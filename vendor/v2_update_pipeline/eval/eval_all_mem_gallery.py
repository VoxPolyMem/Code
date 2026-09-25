"""
Mem-Gallery 全量批量测评 V2

V2 改动（相对 v1）：
  setup_eval_db: 索引路径从 TEXT/IMAGE → UNIFIED/CAPTION
    - 删除 TEXT_INDEX_PATH / IMAGE_INDEX_PATH
    - 新增 UNIFIED_INDEX_PATH（config 里已定义，这里覆盖到 eval db 目录）
    - CAPTION_INDEX_PATH 保留不变
    - 清理时只 unlink UNIFIED/CAPTION（不再 unlink TEXT/IMAGE）

其余逻辑不变：全 turn ingest → 全 QA 检索+评分。
从 v2 目录跑时，import 的 RetrievalOrchestrator / RetrievalTools / config 都是 v2 版本。

用法：
  python -m eval.eval_all_mem_gallery                          # 全部case
  python -m eval.eval_all_mem_gallery --start 0 --end 2        # case 0-1
  python -m eval.eval_all_mem_gallery --case 0                 # 单个case
"""
import sys
import os
import json
import glob
import base64
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.llm_client import call_llm, extract_json, print_usage_summary
from adapters.mem_gallery import MemGalleryAdapter
from retrieval.tools import RetrievalTools
from eval.bench_prompts import (get_hint, get_format_constraint,
                                 answer_messages, answer_messages_with_image,
                                 answer_messages_with_memory_images,
                                 answer_system_prompt, answer_user_prompt,
                                 judge_messages, format_memory_context)
from eval.retrieval_logger import RetrievalLogger

EVAL_RUNS_DIR = config.BASE_DIR / "storage" / "eval_runs"
# 记忆库统一目录（跨 mode 共享）：ingest 结果只依赖 case 数据，不依赖 mode/top_k/route，
# 所以按 case 放一份即可，不同 mode 复用同一份 db + faiss 索引，省去重复 embedding。
INGEST_DIR = EVAL_RUNS_DIR / "_ingest"
LOGS_DIR = config.BASE_DIR / "logs"
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT",
    "data/Mem-Gallery"))
# DB 目录前缀：Mem-Gallery 用 "mem_gallery_"，MemEyeBench 用 "memeye_"
DB_PREFIX = os.environ.get("DB_PREFIX", "mem_gallery_")


def _compress_image_to_b64(image_path: str,
                           max_long_edge: int = 1568,
                           quality: int = 85) -> str:
    """用 PIL 压缩图片返回 base64，避免 413 Request Entity Too Large。

    美团 API 网关（openresty）对 request body 有大小限制，原始高清图片
    base64 编码后常超限触发 413。这里做温和压缩：
    - 最长边 > max_long_edge 才缩小，否则保持原尺寸（不压缩过头）
    - JPEG quality=85（视觉近无损级别）
    - RGB 转换（JPEG 不支持 alpha）
    - 若压缩后 base64 仍 > 3.5MB，降质量到 75 再试一次

    max_long_edge=1568 是 gpt-4o high detail 模式的合理上限
    （短边 768 的 2x2 tile 组合），保证模型能看清图片细节。
    """
    from PIL import Image
    import io

    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    long_edge = max(w, h)
    if long_edge > max_long_edge:
        ratio = max_long_edge / long_edge
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 极端大图：quality=85 仍超限，降到 75
    if len(b64) > 3_500_000:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        print(f"  [img compress] {Path(image_path).name} still large, q=75 → "
              f"{len(b64)/1024/1024:.1f}MB base64", flush=True)
    elif long_edge > max_long_edge:
        print(f"  [img compress] {Path(image_path).name} {w}x{h}→"
              f"{img.size[0]}x{img.size[1]}, q={quality}", flush=True)

    return b64


def _collect_memory_images(context: list[dict], max_k: int = 5,
                           max_long_edge: int = 1568, quality: int = 85) -> list[dict]:
    """从 context 收集记忆原图，压缩成 base64，限 max_k 张（with_visual 实验用）。

    遍历已排序的 context，取有 image_paths 的记忆，每张图压缩成 base64。
    image_id 从 caption dict 的 key 取（格式如 "D1:IMG_001"）；若 caption 无对应 key
    则从路径名推断。返回 [{"mem_id", "image_id", "base64"}]。

    max_long_edge/quality 传给 _compress_image_to_b64（MemEye 高清截图需更激进压缩）。
    """
    collected = []
    for m in context:
        if len(collected) >= max_k:
            break
        image_paths = m.get("image_paths") or []
        if not image_paths:
            continue
        mem_id = m.get("mem_id", "?")
        caps = m.get("caption", {})
        cap_keys = list(caps.keys()) if isinstance(caps, dict) else []
        for idx, img_path in enumerate(image_paths):
            if len(collected) >= max_k:
                break
            if not os.path.exists(img_path):
                continue
            try:
                b64 = _compress_image_to_b64(img_path, max_long_edge=max_long_edge,
                                             quality=quality)
            except Exception as e:
                print(f"  [warn] compress memory image failed {Path(img_path).name}: {e}",
                      flush=True)
                continue
            image_id = cap_keys[idx] if idx < len(cap_keys) else Path(img_path).stem
            collected.append({"mem_id": mem_id, "image_id": image_id, "base64": b64})
    return collected


def setup_eval_db(case_name: str, force_ingest: bool = False) -> bool:
    """把 config 的存储路径指向跨 mode 共享的记忆库目录 _ingest/{case}。

    记忆库内容只依赖 case 数据（不依赖 mode/top_k/route），所以统一放一份。
    - 库已完整写入（semantic.db + 两个 faiss index 都在）且未 force → 返回 True（跳过 ingest）
    - 否则删除旧库 + 重置 memid counter，返回 False（需要重新 ingest）
    """
    ingest_dir = INGEST_DIR / f"{DB_PREFIX}{case_name}"
    if not ingest_dir.exists():
        # 尝试另一种前缀（MemEyeBench 用 memeye_）
        for prefix in ["mem_gallery_", "memeye_"]:
            alt = INGEST_DIR / f"{prefix}{case_name}"
            if alt.exists():
                ingest_dir = alt
                break
    ingest_dir.mkdir(parents=True, exist_ok=True)
    config.STORAGE_DIR = ingest_dir
    config.RAW_DB_PATH = ingest_dir / "raw.db"
    config.SEMANTIC_DB_PATH = ingest_dir / "semantic.db"
    config.EPISODIC_DB_PATH = ingest_dir / "episodic.db"
    config.REGISTRY_DB_PATH = ingest_dir / "registry.db"
    config.FAISS_DIR = ingest_dir / "faiss"
    config.FAISS_DIR.mkdir(exist_ok=True)
    # V2: 2 路索引（unified + caption），删除 text/image
    config.UNIFIED_INDEX_PATH = config.FAISS_DIR / "unified_emb.index"
    config.CAPTION_INDEX_PATH = config.FAISS_DIR / "caption_emb.index"
    config.VOICEPRINT_INDEX_PATH = config.FAISS_DIR / "voiceprint.index"

    _exists = (config.SEMANTIC_DB_PATH.exists()
               and config.UNIFIED_INDEX_PATH.exists()
               and config.CAPTION_INDEX_PATH.exists())
    if _exists and not force_ingest:
        print(f"  [reuse] ingest db exists, skip ingest", flush=True)
        return True

    for p in [config.RAW_DB_PATH, config.SEMANTIC_DB_PATH, config.EPISODIC_DB_PATH,
              config.REGISTRY_DB_PATH, config.UNIFIED_INDEX_PATH, config.CAPTION_INDEX_PATH]:
        if p.exists():
            p.unlink()
    from utils import memid
    memid._counters = {config.PREFIX_RAW: 0, config.PREFIX_SEM: 0,
                       config.PREFIX_EPI: 0, config.PREFIX_COR: 0}
    return False


def _read_reused_stats(case_name: str) -> dict:
    """跳过 ingest 时，从共享记忆库读取 n_turns / n_memories（仅用于打印/汇总）。"""
    import sqlite3
    ingest_dir = INGEST_DIR / f"{DB_PREFIX}{case_name}"
    if not ingest_dir.exists():
        for prefix in ["mem_gallery_", "memeye_"]:
            alt = INGEST_DIR / f"{prefix}{case_name}"
            if alt.exists():
                ingest_dir = alt
                break
    n_mem = n_turn = 0
    try:
        c = sqlite3.connect(str(ingest_dir / "semantic.db"))
        n_mem = c.execute("SELECT COUNT(*) FROM semantic_memory").fetchone()[0]
        c.close()
    except Exception:
        pass
    try:
        c = sqlite3.connect(str(ingest_dir / "raw.db"))
        n_turn = c.execute("SELECT COUNT(*) FROM raw_turns").fetchone()[0]
        c.close()
    except Exception:
        pass
    return {"n_turns": n_turn, "n_memories": n_mem}


def eval_one_case(case_path: str, baseline: bool = False, force_ingest: bool = False) -> dict:
    """测评单个 case：全 turn 写入 → 全 QA 检索+评分。

    baseline=True: 只用 Round 0 直接 search_memory（和 LTMemory 一样），不走多轮。
    baseline=False: 走 orchestrator 多轮（Round 0 → sufficiency → rewrite+route）。
    force_ingest=True: 强制重新 ingest（忽略已存在的共享记忆库）。
    """
    case_name = Path(case_path).stem
    mode = "baseline" if baseline else "multi-round"
    # 环境变量后缀：用于隔离不同 model 的结果（如 gpt-4o-mini baseline）
    _suffix = os.environ.get("EVAL_MODE_SUFFIX", "")
    if _suffix:
        mode = f"{mode}-{_suffix}"
    # 结果目录按 mode 隔离（答题/judge 结果随 mode 不同）；记忆库在共享 _ingest 目录
    result_dir = EVAL_RUNS_DIR / mode / f"{DB_PREFIX}{case_name}"
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Case: {case_name} [{mode}]")
    print(f"{'='*60}", flush=True)

    can_reuse = setup_eval_db(case_name, force_ingest=force_ingest)

    # 1. 全 turn 写入（不限制 max_turns）；共享库已存在则跳过
    if can_reuse:
        stats = _read_reused_stats(case_name)
        print(f"[1] Reusing memory db: {stats['n_turns']} turns, "
              f"{stats['n_memories']} memories (skip ingest)", flush=True)
    else:
        print(f"[1] Ingesting ALL turns (dense caption + fast_mode)...", flush=True)
        ad = MemGalleryAdapter()
        stats = ad.ingest_case(case_path, fast_mode=True)  # 全 turn
        ad.ingestor.sem_store.flush()
        print(f"  ingested: {stats['n_turns']} turns, {stats['n_memories']} memories", flush=True)

    # 2. 全 QA
    case_data = json.load(open(case_path))
    all_qa = case_data["human-annotated QAs"]
    character = case_data.get("character_profile", {}).get("name", "unknown")
    sessions = case_data.get("multi_session_dialogues", [])
    last_date = sessions[-1].get("date", "2024-12-31") if sessions else "2024-12-31"
    print(f"[2] Evaluating {len(all_qa)} QA (character={character})", flush=True)

    tools = RetrievalTools()
    # orchestrator 复用：避免每个 QA 创建新实例导致 SQLite 连接堆积 → "database is locked"
    _orch = None
    if not baseline:
        from retrieval.orchestrator import RetrievalOrchestrator
        _orch = RetrievalOrchestrator()
    # log 也按 mode 隔离，避免不同配置互相覆盖
    logger = RetrievalLogger(LOGS_DIR / mode, f"{DB_PREFIX}{case_name}")

    # 断点续跑：加载已有结果，跳过已完成的题
    _resume_path = result_dir / f"eval_results_{mode}.json"
    existing_qa_ids = set()
    results = []
    if _resume_path.exists():
        try:
            _existing = json.load(open(_resume_path))
            existing_qa_ids = {r.get("qa_id") for r in _existing}
            results = list(_existing)
            if existing_qa_ids:
                print(f"  [resume] {len(existing_qa_ids)} existing results, skipping", flush=True)
        except Exception:
            pass

    for i, qa in enumerate(all_qa):
        q = qa["question"]
        gt = str(qa.get("answer", ""))
        point = qa.get("point", "?")
        qa_id = qa.get("question_id", f"Q{i+1}")

        # 断点续跑：跳过已完成的题
        if qa_id in existing_qa_ids:
            continue

        format_constraint = get_format_constraint(point)

        # V2: 读 question_image（对齐 LTMemory run_bench.py）
        question_image_path = qa.get("question_image", "")
        question_image_caption = qa.get("image_caption", None)
        question_image_abs = None
        if question_image_path:
            if not os.path.isabs(question_image_path):
                if question_image_path.startswith("../image/"):
                    rel = question_image_path.replace("../image/", "")
                    question_image_abs = str(DATASET_ROOT / "data" / "image" / rel)
                else:
                    question_image_abs = str(DATASET_ROOT / "data" / "image" / question_image_path)
            else:
                question_image_abs = question_image_path

        # V2: query 拼接 question image caption（对齐 LTMemory memory_recall）
        # LTMemory: observation += "\nquestion's image:\nimage_caption: " + caption
        search_query = q
        if question_image_caption:
            search_query = q + "\nquestion's image:\nimage_caption: " + str(question_image_caption)

        logger.start_qa(qa_id, q, gt, point)

        import io, contextlib
        orch_iters = 0

        if baseline:
            # baseline: 直接 search_memory(search_query)（和 LTMemory 一样），不走多轮
            retrieved = tools.search_memory(search_query)
            need_imgs = False   # baseline 不走 orchestrator，无 need_memory_images
            logger.log_round(0, {"query_type": "baseline", "tool_choice": "search_memory"},
                             retrieved, len(retrieved), {}, None,
                             label="Final memory for answer")
        else:
            # 多轮 orchestrator（Round 0 → sufficiency → rewrite+route）
            orch = _orch  # 复用单例，避免 SQLite 连接堆积
            log_capture = io.StringIO()
            with contextlib.redirect_stdout(log_capture):
                orch_result = orch.retrieve(search_query, point=point)
            retrieved = orch_result.get("context", [])
            orch_iters = orch_result.get("iter", 0)
            need_imgs = orch_result.get("need_memory_images", False)  # with_visual 实验
            logger.log_round(0, {"query_type": "orchestrator", "tool_choice": "multi-round"},
                             retrieved, len(retrieved),
                             {"confidence": orch_result.get("confidence", 0),
                              "need_memory_images": need_imgs},
                             None,
                             label="Final memory for answer")
            orch_log = log_capture.getvalue()
            if orch_log:
                logger._lines.append(f"\n[Orchestrator Log]")
                logger._lines.append(orch_log)
            if need_imgs:
                print(f"  [with_visual] need_memory_images=True, will pass memory images to answer model",
                      flush=True)

        # V2: 答题（对齐 LTMemory fast_run_with_textual_memory）
        # caption 已拼进 search_query（检索）+ 记忆 text（ingest），answer 模型能看到 caption 文本。
        # need_imgs=True（with_visual 实验）：额外传 top-K 记忆原图 + 问题图给答题模型。
        # 否则：有 question_image 传问题图（answer_messages_with_image），无则纯文本。
        if need_imgs:
            mem_imgs = _collect_memory_images(retrieved, max_k=5)
            qimg_b64 = None
            if question_image_abs and os.path.exists(question_image_abs):
                try:
                    qimg_b64 = _compress_image_to_b64(question_image_abs)
                except Exception:
                    qimg_b64 = None
            if mem_imgs or qimg_b64:
                try:
                    msgs = answer_messages_with_memory_images(
                        q, retrieved, mem_imgs,
                        question_image_base64=qimg_b64,
                        character=character, last_date=last_date,
                        hint=format_constraint)
                    prediction = call_llm(msgs).strip()
                except Exception as e:
                    # fallback: 图片传不过去（如模型不支持 image_url）退回纯文本
                    print(f"  [warn] answer with memory images failed ({e}), fallback to text-only",
                          flush=True)
                    msgs = answer_messages(q, retrieved, character=character,
                                           last_date=last_date, hint=format_constraint)
                    prediction = call_llm(msgs).strip()
            else:
                # 没有可用记忆图/问题图，退回纯文本
                msgs = answer_messages(q, retrieved, character=character,
                                       last_date=last_date, hint=format_constraint)
                prediction = call_llm(msgs).strip()
        elif question_image_abs and os.path.exists(question_image_abs):
            try:
                img_b64 = _compress_image_to_b64(question_image_abs)
                msgs = answer_messages_with_image(q, retrieved, img_b64,
                                                  character=character, last_date=last_date,
                                                  hint=format_constraint)
                prediction = call_llm(msgs).strip()
            except Exception as e:
                # fallback: gpt-4.1 不支持 image_url 时退回文本模式
                print(f"  [warn] answer with image failed ({e}), fallback to text-only", flush=True)
                msgs = answer_messages(q, retrieved, character=character,
                                       last_date=last_date, hint=format_constraint)
                prediction = call_llm(msgs).strip()
        else:
            msgs = answer_messages(q, retrieved, character=character,
                                   last_date=last_date, hint=format_constraint)
            prediction = call_llm(msgs).strip()

        # LLM-judge（对齐官方 llm_judge.txt）
        try:
            judge_resp = call_llm(judge_messages(q, gt, prediction))
            judge_data = extract_json(judge_resp)
        except Exception:
            judge_data = {"score": 0.0, "reasoning": "judge parse failed"}
        score = float(judge_data.get("score", 0.0))

        logger.log_answer(prediction, judge_data)

        results.append({
            "qa_id": qa_id, "point": point, "question": q, "gt": gt,
            "prediction": prediction[:300], "score": score,
            "n_retrieved": len(retrieved),
            "retrieved_mem_ids": [r.get("mem_id", "") for r in retrieved],
            "top_score": retrieved[0].get("score", 0) if retrieved else 0,
            "orch_iters": orch_iters,
            "mode": mode,
        })

        # 增量保存（避免崩溃丢数据）
        _save_path = result_dir / f"eval_results_{mode}.json"
        json.dump(results, open(_save_path, "w"), ensure_ascii=False, indent=2)

        status = "✅" if score >= 0.75 else ("🟡" if score >= 0.5 else "❌")
        print(f"  [{i+1}/{len(all_qa)}] {point} {status} {score} "
              f"Q={q[:40]}...", flush=True)
        logger.flush()

    # 汇总
    by_point = defaultdict(list)
    for r in results:
        by_point[r["point"]].append(r["score"])

    print(f"\n  --- {case_name} Summary ---", flush=True)
    print(f"  {'Point':<6} {'#QA':<5} {'Avg':<8} {'Acc@0.75':<8}", flush=True)
    case_avg = sum(r["score"] for r in results) / max(len(results), 1)
    case_acc = sum(1 for r in results if r["score"] >= 0.75) / max(len(results), 1)
    for p in sorted(by_point.keys()):
        scores = by_point[p]
        avg = sum(scores) / len(scores)
        acc = sum(1 for s in scores if s >= 0.75) / len(scores)
        print(f"  {p:<6} {len(scores):<5} {avg:<8.3f} {acc:<8.3f}", flush=True)
    print(f"  {'TOTAL':<6} {len(results):<5} {case_avg:<8.3f} {case_acc:<8.3f}", flush=True)

    # 存结果
    result_path = result_dir / f"eval_results_{mode}.json"
    json.dump(results, open(result_path, "w"), ensure_ascii=False, indent=2)

    return {
        "case": case_name, "n_turns": stats["n_turns"],
        "n_memories": stats["n_memories"], "n_qa": len(results),
        "avg_score": case_avg, "acc_075": case_acc,
        "by_point": {p: {"n": len(s), "avg": sum(s)/len(s),
                         "acc": sum(1 for x in s if x >= 0.75)/len(s)}
                     for p, s in by_point.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=None, action="append",
                        help="case index (can repeat)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--baseline", action="store_true",
                        help="baseline: 只用 Round 0 直接 search_memory（和 LTMemory 一样），不走多轮")
    parser.add_argument("--force-ingest", action="store_true",
                        help="强制重新 ingest（忽略已存在的共享记忆库，默认会自动复用）")
    args = parser.parse_args()

    cases = sorted(glob.glob(str(DATASET_ROOT / "data" / "dialog" / "*.json")))
    if args.case is not None:
        cases = [cases[idx] for idx in args.case]
    else:
        end = args.end or len(cases)
        cases = cases[args.start:end]

    mode_tag = "BASELINE (LTMemory-style, single-round)" if args.baseline else "MULTI-ROUND (orchestrator)"
    print(f"=== Mem-Gallery Eval V2 [{mode_tag}]: {len(cases)} cases ===", flush=True)

    all_summaries = []
    for i, case_path in enumerate(cases):
        print(f"\n{'#'*60}")
        print(f"# Case {i+1}/{len(cases)}: {Path(case_path).stem}")
        print(f"{'#'*60}", flush=True)
        try:
            summary = eval_one_case(case_path, baseline=args.baseline, force_ingest=args.force_ingest)
            all_summaries.append(summary)
        except Exception as e:
            print(f"  ❌ ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
            all_summaries.append({"case": Path(case_path).stem, "error": str(e)})

    # 总汇总
    print(f"\n{'='*60}")
    print("=== ALL CASES SUMMARY ===")
    print(f"{'Case':<40} {'#QA':<5} {'Avg':<8} {'Acc@0.75':<8}")
    valid = [s for s in all_summaries if "avg_score" in s]
    for s in valid:
        print(f"{s['case'][:40]:<40} {s['n_qa']:<5} {s['avg_score']:<8.3f} {s['acc_075']:<8.3f}")
    if valid:
        total_avg = sum(s["avg_score"] for s in valid) / len(valid)
        total_acc = sum(s["acc_075"] for s in valid) / len(valid)
        print(f"{'OVERALL':<40} {'':<5} {total_avg:<8.3f} {total_acc:<8.3f}")

    summary_path = LOGS_DIR / "mem_gallery_all_summary.json"
    json.dump(all_summaries, open(summary_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nSummary: {summary_path}")
    print_usage_summary()


if __name__ == "__main__":
    main()
