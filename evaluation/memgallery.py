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
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path
from collections import defaultdict

import numpy as np

V2_PIPELINE_ROOT = Path(os.environ.get(
    "V2_PIPELINE_ROOT",
    Path(__file__).resolve().parent.parent / "vendor/v2_update_pipeline",
))
sys.path.insert(0, str(V2_PIPELINE_ROOT))

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
from core.memgallery_executor import RouteExecutor
from core.evidence_bundle import answer_view, evidence_bundle_active
from core.iterative_retrieval import (
    configured_top_k,
    iterative_retrieval_enabled,
    retrieve_iteratively,
)
from core.planner import (
    heuristic_plan,
    mg_keyword_r1_plan,
    model_type_fixed_plan,
    two_level_dynamic_plan,
    unified_hybrid_plan,
)

V2_EVAL_RUNS_DIR = config.BASE_DIR / "storage" / "eval_runs"
EVAL_RUNS_DIR = Path(os.environ.get(
    "FUSION_EVAL_RUNS_DIR",
    Path(__file__).resolve().parent.parent / "artifacts/evaluation/memgallery",
))
# 记忆库统一目录（跨 mode 共享）：ingest 结果只依赖 case 数据，不依赖 mode/top_k/route，
# 所以按 case 放一份即可，不同 mode 复用同一份 db + faiss 索引，省去重复 embedding。
INGEST_DIR = V2_EVAL_RUNS_DIR / "_ingest"
LOGS_DIR = EVAL_RUNS_DIR / "logs"
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT",
    Path(__file__).resolve().parent.parent / "data/Mem-Gallery"))
# DB 目录前缀：Mem-Gallery 用 "mem_gallery_"，MemEyeBench 用 "memeye_"
DB_PREFIX = os.environ.get("DB_PREFIX", "mem_gallery_")
FACT_ROOT = Path(os.environ.get(
    "FACT_ROOT",
    Path(__file__).resolve().parent.parent / "artifacts/memory/memgallery_contextual_v1/compatible_views/no_role_metadata",
))
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _batch_encode(texts: list[str]) -> np.ndarray:
    """Use the same Qwen3-VL embedding service as v2 in bounded batches."""
    url = os.environ.get("EMBEDDING_SERVER_URL", "http://localhost:9981")
    batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))
    chunks = []
    for start in range(0, len(texts), batch_size):
        payload = {"type": "batch_text", "texts": texts[start:start + batch_size],
                   "instr": "Represent the text for retrieval."}
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as response:
            out = json.loads(response.read())
        chunk = np.frombuffer(base64.b64decode(out["emb"]), dtype=np.float32)
        chunks.append(chunk.reshape(tuple(out["shape"])))
    arr = np.vstack(chunks)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def _batch_encode_images(paths: list[str]) -> np.ndarray:
    """Encode real images in the same Qwen3-VL space, in small GPU-safe batches."""
    url = os.environ.get("EMBEDDING_SERVER_URL", "http://localhost:9981")
    batch_size = int(os.environ.get("IMAGE_EMBEDDING_BATCH_SIZE", "8"))
    chunks = []
    for start in range(0, len(paths), batch_size):
        payload = {
            "type": "batch_image",
            "images": paths[start:start + batch_size],
            "instr": "Represent the image for retrieval.",
        }
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as response:
            out = json.loads(response.read())
        chunk = np.frombuffer(base64.b64decode(out["emb"]), dtype=np.float32)
        chunks.append(chunk.reshape(tuple(out["shape"])))
    arr = np.vstack(chunks)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


class FactFusion:
    """Fact retrieval is an evidence-rescue/rerank channel over v2 memories.

    Facts never replace the source evidence shown to the answer model.  A fact hit is
    mapped back to its source v2 semantic turn, then fused with v2's original ranking.
    This keeps image IDs, dense captions, raw text, timestamps, and provenance intact.
    """

    def __init__(self, case_name: str, tools: RetrievalTools):
        self.case_name = case_name
        self.tools = tools
        self.facts = []
        self.image_to_visual_sets = defaultdict(list)
        self.last_route = {}
        for path in sorted((FACT_ROOT / case_name).glob("D*/event_cards.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            session_facts = []
            for card in doc.get("event_cards", []):
                for fact in card.get("atomic_facts", []):
                    text = str(fact.get("text", "")).strip()
                    refs = [str(x) for x in fact.get("source_turn_ids", [])]
                    if text and refs:
                        fact_row = {
                            "text": text,
                            "source_turn_ids": refs,
                            "refer_ids": [str(x) for x in fact.get(
                                "refer_ids", refs)],
                            "source_image_ids": list(fact.get("source_image_ids", [])),
                        }
                        self.facts.append(fact_row)
                        session_facts.append(fact_row)
            visual_sets = path.parent / "visual_sets.json"
            if visual_sets.exists():
                visual_doc = json.loads(visual_sets.read_text(encoding="utf-8"))
                for item in visual_doc.get("visual_sets", []):
                    text = str(item.get("text", "")).strip()
                    refs = [str(x) for x in item.get("source_turn_ids", [])]
                    image_ids = [str(x) for x in item.get("image_ids", [])]
                    if text and refs and image_ids:
                        set_label = str(item.get("label", "")).strip()
                        turn_numbers = [
                            int(match.group(1)) for ref in refs
                            if (match := re.search(r":(\d+)$", ref))
                        ]
                        nearby = []
                        if turn_numbers:
                            lo, hi = min(turn_numbers), max(turn_numbers)
                            for fact in session_facts:
                                fact_turns = [
                                    int(match.group(1))
                                    for ref in fact.get("source_turn_ids", [])
                                    if (match := re.search(r":(\d+)$", ref))
                                ]
                                if fact_turns and any(lo <= value <= hi for value in fact_turns):
                                    nearby.append(fact["text"])
                        self.facts.append({
                            "text": text,
                            "retrieval_text": text + " " + " ".join(
                                list(dict.fromkeys(nearby))),
                            "source_turn_ids": refs,
                            "refer_ids": [str(x) for x in item.get(
                                "refer_ids", refs)],
                            "source_image_ids": image_ids,
                            "fact_kind": "visual_set",
                            "set_label": set_label,
                        })
                        for image_id in image_ids:
                            if set_label:
                                self.image_to_visual_sets[image_id].append(set_label)
        if not self.facts:
            raise RuntimeError(f"No Fact cards found for {case_name} under {FACT_ROOT}")
        cache = FACT_ROOT / case_name / "fact_embeddings.npy"
        meta = FACT_ROOT / case_name / "fact_embeddings.meta.json"
        signature = [f.get("retrieval_text", f["text"]) for f in self.facts]
        reuse = False
        if cache.exists() and meta.exists():
            try:
                old = json.loads(meta.read_text(encoding="utf-8"))
                reuse = old.get("texts") == signature
            except Exception:
                pass
        if reuse:
            self.embeddings = np.load(cache)
        else:
            self.embeddings = _batch_encode(signature)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.embeddings)
            meta.write_text(json.dumps({"texts": signature}, ensure_ascii=False), encoding="utf-8")
        self.docs = [self._tokens(f.get("retrieval_text", f["text"]))
                     for f in self.facts]
        self.counts = [Counter(row) for row in self.docs]
        self.lengths = np.asarray([len(row) for row in self.docs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if len(self.lengths) else 1.0
        df = Counter(token for row in self.docs for token in set(row))
        n = max(len(self.docs), 1)
        self.idf = {t: math.log(1 + (n-c+0.5)/(c+0.5)) for t, c in df.items()}
        self.turn_to_mem = self._build_turn_map()
        self.mem_to_refer_ids = defaultdict(list)
        for refer_id, mem_id in self.turn_to_mem.items():
            self.mem_to_refer_ids[mem_id].append(refer_id)
        for fact in self.facts:
            fact["refer_mem_ids"] = list(dict.fromkeys(
                self.turn_to_mem[ref]
                for ref in fact.get("refer_ids", fact["source_turn_ids"])
                if ref in self.turn_to_mem
            ))
        self._build_visual_card_index()
        print(f"  [FactFusion] {len(self.facts)} atomic facts, "
              f"{len(self.turn_to_mem)} source turns, "
              f"{len(self.visual_cards)} visual cards", flush=True)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def _build_turn_map(self) -> dict[str, str]:
        mapping = {}
        for mem_id in self.tools.sem_store.unified_ids:
            for raw_id in self.tools.sem_store.get_refer_ids(mem_id):
                match = re.search(r":(D\d+):(\d+)$", str(raw_id))
                if match:
                    mapping[f"{match.group(1)}:{int(match.group(2))+1}"] = mem_id
        return mapping

    def _build_visual_card_index(self) -> None:
        """Aggregate all evidence belonging to one image into one retrieval unit.

        Atomic facts are useful for textual recall but are too narrow for visual
        selection: a single image may have separate facts for object, colour,
        relation, mood, and time.  A visual card keeps those attributes together
        with the original dense caption and dialogue turn.
        """
        caption_parts = defaultdict(list)
        dialogue_parts = defaultdict(list)
        fact_parts = defaultdict(list)
        image_paths = defaultdict(list)
        dates = defaultdict(set)
        mem_ids = defaultdict(set)
        for mem_id in self.tools.sem_store.unified_ids:
            mem = self.tools.sem_store.get_memory(mem_id)
            if not mem:
                continue
            captions = mem.get("image_captions", {}) or {}
            paths = mem.get("image_paths", []) or []
            for image_index, (image_id, caption) in enumerate(captions.items()):
                image_id = str(image_id)
                if caption:
                    caption_parts[image_id].append(str(caption).strip())
                if image_index < len(paths) and os.path.exists(str(paths[image_index])):
                    image_paths[image_id].append(str(paths[image_index]))
                if mem.get("text"):
                    dialogue_parts[image_id].append(str(mem["text"]).strip())
                if mem.get("date"):
                    dates[image_id].add(str(mem["date"]))
                mem_ids[image_id].add(mem_id)
        for fact in self.facts:
            if fact.get("fact_kind") == "visual_set":
                continue
            for image_id in fact.get("source_image_ids", []):
                fact_parts[str(image_id)].append(fact["text"])

        self.visual_cards = []
        all_image_ids = set(caption_parts) | set(dialogue_parts) | set(fact_parts)
        for image_id in sorted(all_image_ids):
            captions = list(dict.fromkeys(x for x in caption_parts[image_id] if x))
            dialogues = list(dict.fromkeys(x for x in dialogue_parts[image_id] if x))
            facts = list(dict.fromkeys(x for x in fact_parts[image_id] if x))
            text = f"image_id={image_id}. " + " ".join(captions + dialogues + facts)
            annotation_text = f"image_id={image_id}; " + " ".join(captions + facts)
            self.visual_cards.append({
                "image_id": image_id,
                "text": text,
                "annotation_text": annotation_text,
                "mem_ids": sorted(mem_ids[image_id]),
                "dates": sorted(dates[image_id]),
                "image_paths": list(dict.fromkeys(image_paths[image_id])),
            })
        self.image_to_card_index = {
            row["image_id"]: index for index, row in enumerate(self.visual_cards)
        }

        signature = [row["text"] for row in self.visual_cards]
        cache = FACT_ROOT / self.case_name / "visual_card_embeddings.npy"
        meta = FACT_ROOT / self.case_name / "visual_card_embeddings.meta.json"
        reuse = False
        if cache.exists() and meta.exists():
            try:
                reuse = json.loads(meta.read_text(encoding="utf-8")).get("texts") == signature
            except Exception:
                pass
        if reuse:
            self.visual_embeddings = np.load(cache)
        else:
            self.visual_embeddings = _batch_encode(signature)
            np.save(cache, self.visual_embeddings)
            meta.write_text(json.dumps({"texts": signature}, ensure_ascii=False),
                            encoding="utf-8")
        self.visual_docs = [self._tokens(row["text"]) for row in self.visual_cards]
        self.visual_counts = [Counter(row) for row in self.visual_docs]
        self.visual_lengths = np.asarray(
            [len(row) for row in self.visual_docs], dtype=np.float32)
        self.visual_avg_len = float(self.visual_lengths.mean()) if len(self.visual_lengths) else 1.0
        visual_df = Counter(token for row in self.visual_docs for token in set(row))
        n_visual = max(len(self.visual_docs), 1)
        self.visual_idf = {
            token: math.log(1 + (n_visual-count+0.5)/(count+0.5))
            for token, count in visual_df.items()
        }

        # Real-image embeddings are cached separately from text/caption cards.
        # Each card keeps a direct path to its bottom-level image evidence.
        image_signature = [
            row["image_paths"][0] if row.get("image_paths") else ""
            for row in self.visual_cards
        ]
        image_cache = FACT_ROOT / self.case_name / "visual_image_embeddings.npy"
        image_meta = FACT_ROOT / self.case_name / "visual_image_embeddings.meta.json"
        reuse_images = False
        if image_cache.exists() and image_meta.exists():
            try:
                reuse_images = (
                    json.loads(image_meta.read_text(encoding="utf-8")).get("paths")
                    == image_signature
                )
            except Exception:
                pass
        if reuse_images:
            self.visual_image_embeddings = np.load(image_cache)
        else:
            valid_indices = [i for i, path in enumerate(image_signature) if path]
            encoded = _batch_encode_images([image_signature[i] for i in valid_indices])
            dim = encoded.shape[1]
            self.visual_image_embeddings = np.zeros(
                (len(self.visual_cards), dim), dtype=np.float32)
            for index, vector in zip(valid_indices, encoded):
                self.visual_image_embeddings[index] = vector
            np.save(image_cache, self.visual_image_embeddings)
            image_meta.write_text(
                json.dumps({"paths": image_signature}, ensure_ascii=False),
                encoding="utf-8")

    def rank_visual_cards(self, query: str,
                          allowed_image_ids: set[str] | None = None,
                          query_image_path: str | None = None,
                          top_k: int = 5) -> list[dict]:
        query_emb = _batch_encode([query])[0]
        text_dense = self.visual_embeddings @ query_emb
        image_dense = None
        if query_image_path and os.path.exists(query_image_path):
            image_query_emb = _batch_encode_images([query_image_path])[0]
            image_dense = self.visual_image_embeddings @ image_query_emb
        bm25 = np.zeros(len(self.visual_docs), dtype=np.float32)
        for token in set(self._tokens(query)):
            if token not in self.visual_idf:
                continue
            for i, counts in enumerate(self.visual_counts):
                freq = counts.get(token, 0)
                if not freq:
                    continue
                denom = freq + 1.5 * (
                    0.25 + 0.75 * self.visual_lengths[i] / max(self.visual_avg_len, 1e-6))
                bm25[i] += self.visual_idf[token] * freq * 2.5 / denom
        eligible = [
            i for i, row in enumerate(self.visual_cards)
            if not allowed_image_ids or row["image_id"] in allowed_image_ids
        ]
        text_dense_order = sorted(eligible, key=lambda i: text_dense[i], reverse=True)
        image_dense_order = (
            sorted(eligible, key=lambda i: image_dense[i], reverse=True)
            if image_dense is not None else [])
        bm25_order = sorted(eligible, key=lambda i: bm25[i], reverse=True)
        rrf = defaultdict(float)
        channels = [(text_dense_order, False, 0.75), (bm25_order, True, 0.75)]
        if image_dense_order:
            channels.append((image_dense_order, False, 2.0))
        for order, is_sparse, weight in channels:
            for rank, idx in enumerate(order[:30], 1):
                if is_sparse and bm25[idx] <= 0:
                    continue
                rrf[idx] += weight / (60 + rank)
        rows = []
        for idx in sorted(rrf, key=rrf.get, reverse=True)[:top_k]:
            row = dict(self.visual_cards[idx])
            row.update({"retrieval_score": rrf[idx],
                        "dense_score": float(text_dense[idx]),
                        "image_score": (
                            float(image_dense[idx]) if image_dense is not None else None),
                        "bm25_score": float(bm25[idx])})
            rows.append(row)
        return rows

    def rank_visual_sets_by_image(self, query_image_path: str,
                                  text_ranked_sets: list[dict]) -> list[dict]:
        """Rank sets by the mean similarity of their two nearest member images."""
        query_emb = _batch_encode_images([query_image_path])[0]
        member_scores = self.visual_image_embeddings @ query_emb
        text_rank = {
            fact.get("set_label"): rank
            for rank, fact in enumerate(text_ranked_sets, 1)
        }
        rows = []
        for fact in self.facts:
            if fact.get("fact_kind") != "visual_set":
                continue
            indices = [
                self.image_to_card_index[image_id]
                for image_id in fact.get("source_image_ids", [])
                if image_id in self.image_to_card_index
            ]
            if not indices:
                continue
            scores = sorted(
                (float(member_scores[index]) for index in indices), reverse=True)
            k = min(2, len(scores))
            visual_score = sum(scores[:k]) / k
            label = fact.get("set_label")
            text_score = 1.0 / (60 + text_rank.get(label, len(text_rank) + 1))
            row = dict(fact)
            row["visual_set_score"] = visual_score
            row["retrieval_score"] = 0.9 * visual_score + 0.1 * text_score
            rows.append(row)
        return sorted(rows, key=lambda row: row["retrieval_score"], reverse=True)

    def _bm25(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.docs), dtype=np.float32)
        for token in set(self._tokens(query)):
            if token not in self.idf:
                continue
            for i, counts in enumerate(self.counts):
                freq = counts.get(token, 0)
                if not freq:
                    continue
                denom = freq + 1.5 * (0.25 + 0.75 * self.lengths[i] / max(self.avg_len, 1e-6))
                scores[i] += self.idf[token] * freq * 2.5 / denom
        return scores

    def rank_facts(self, query: str, allow_visual_sets: bool,
                   top_facts: int = 30) -> list[dict]:
        query_emb = _batch_encode([query])[0]
        dense = self.embeddings @ query_emb
        bm25 = self._bm25(query)
        eligible = np.asarray([
            allow_visual_sets or fact.get("fact_kind") != "visual_set"
            for fact in self.facts
        ], dtype=bool)
        dense_order = [int(i) for i in np.argsort(-dense) if eligible[int(i)]][:top_facts]
        bm25_order = [int(i) for i in np.argsort(-bm25) if eligible[int(i)]][:top_facts]
        rrf = defaultdict(float)
        for source, order in (("dense", dense_order), ("bm25", bm25_order)):
            for rank, idx in enumerate(order, 1):
                if source == "bm25" and bm25[idx] <= 0:
                    continue
                rrf[idx] += 1.0 / (60 + rank)
        return [dict(self.facts[idx], retrieval_score=rrf[idx])
                for idx in sorted(rrf, key=rrf.get, reverse=True)]

    def rank_visual_sets(self, query: str) -> list[dict]:
        """Rank visual-set facts against each other, never behind atomic facts."""
        indices = [i for i, fact in enumerate(self.facts)
                   if fact.get("fact_kind") == "visual_set"]
        if not indices:
            return []
        query_emb = _batch_encode([query])[0]
        dense = self.embeddings @ query_emb
        bm25 = self._bm25(query)
        dense_order = sorted(indices, key=lambda i: dense[i], reverse=True)
        bm25_order = sorted(indices, key=lambda i: bm25[i], reverse=True)
        rrf = defaultdict(float)
        for order, is_sparse in ((dense_order, False), (bm25_order, True)):
            for rank, idx in enumerate(order, 1):
                if is_sparse and bm25[idx] <= 0:
                    continue
                rrf[idx] += 1.0 / (60 + rank)
        return [dict(self.facts[idx], retrieval_score=rrf[idx],
                     dense_score=float(dense[idx]), bm25_score=float(bm25[idx]))
                for idx in sorted(rrf, key=rrf.get, reverse=True)]

    def rank_source_memories(self, query: str, top_facts: int = 30,
                             allow_visual_sets: bool = True,
                             max_per_source: bool = False) -> list[dict]:
        query_emb = _batch_encode([query])[0]
        dense = self.embeddings @ query_emb
        bm25 = self._bm25(query)
        eligible = np.asarray([
            allow_visual_sets or fact.get("fact_kind") != "visual_set"
            for fact in self.facts
        ], dtype=bool)
        dense_order = [int(i) for i in np.argsort(-dense) if eligible[int(i)]][:top_facts]
        bm25_order = [int(i) for i in np.argsort(-bm25) if eligible[int(i)]][:top_facts]
        rrf = defaultdict(float)
        for source, order in (("dense", dense_order), ("bm25", bm25_order)):
            for rank, idx in enumerate(order, 1):
                if source == "bm25" and bm25[idx] <= 0:
                    continue
                rrf[int(idx)] += 1.0 / (60 + rank)
        ranked_facts = sorted(rrf, key=rrf.get, reverse=True)
        mem_scores = defaultdict(float)
        mem_facts = defaultdict(list)
        mem_visual = defaultdict(bool)
        mem_visual_sets = defaultdict(list)
        for rank, idx in enumerate(ranked_facts, 1):
            for turn_id in self.facts[idx]["source_turn_ids"]:
                mem_id = self.turn_to_mem.get(turn_id)
                if mem_id:
                    contribution = 1.0 / (60 + rank)
                    if max_per_source:
                        mem_scores[mem_id] = max(mem_scores[mem_id], contribution)
                    else:
                        mem_scores[mem_id] += contribution
                    mem_facts[mem_id].append(self.facts[idx]["text"])
                    mem_visual[mem_id] = mem_visual[mem_id] or bool(
                        self.facts[idx].get("source_image_ids"))
                    if self.facts[idx].get("fact_kind") == "visual_set":
                        mem_visual_sets[mem_id].append(self.facts[idx]["text"])
        rows = []
        for mem_id in sorted(mem_scores, key=mem_scores.get, reverse=True):
            mem = self.tools.sem_store.get_memory(mem_id)
            if not mem:
                continue
            row = self.tools._to_retrieved(mem)
            row["fact_score"] = mem_scores[mem_id]
            row["matched_facts"] = mem_facts[mem_id][:3]
            row["fact_has_image"] = mem_visual[mem_id]
            row["matched_visual_sets"] = list(dict.fromkeys(mem_visual_sets[mem_id]))
            rows.append(row)
        return rows

    def fuse(self, query: str, base: list[dict],
             question_image_path: str | None = None
             ) -> tuple[list[dict], list[str], list[str]]:
        lowered = query.lower()
        collection_query = bool(re.search(
            r"\bhow many\b|\btotal\b|\bnumber of\b|"
            r"\b(?:search for|find) all\b|"
            r"\ball\s+(?:\w+\s+){0,3}(?:images|photos|pictures)\b|"
            r"\bwhich\s+(?:\w+\s+){0,3}(?:images|photos|pictures)\b",
            lowered,
        ))
        # Candidate prototype: do not consume the benchmark's hidden type label to
        # select a layer.  Visual intent comes only from observable query/image
        # inputs; raw evidence remains the fallback in every branch.
        visual_query = bool(question_image_path) or bool(re.search(
            r"\b(image|images|photo|photos|picture|pictures|visual|look like|depicts)\b",
            lowered,
        ))
        self.last_route = {
            "raw": 1.0,
            "fact": 0.0 if visual_query else 0.75,
            "visual": 1.0 if collection_query else 0.0,
            "collection": 1.0 if collection_query else 0.0,
            "used_benchmark_point": False,
        }
        if visual_query:
            # Keep v2's exact top-30.  Fact is an evidence annotation channel, not
            # a competing memory slot: this prevents similar images from crowding
            # out or reordering the original multimodal evidence.
            ordered = list(base[:30])
            annotations = []
            chosen_sets = []
            covered_ids = set()
            # A single-image visual card is only a richer representation of one
            # raw turn, so making it consume a context slot cannot add coverage.
            # Keep ordinary visual questions on v2's raw multimodal context.  A
            # hierarchy action is useful only for an explicit collection request,
            # where one node can expand to several grounded bottom-level turns.
            if not collection_query:
                self.last_route["route_reason"] = "raw_visual_fallback"
                return ordered, [], []
            if collection_query:
                self.last_route["route_reason"] = "explicit_collection_intent"
                ranked = self.rank_visual_sets(query)
                multi_set_query = bool(re.search(r"\b(total|all|overall)\b", lowered))
                same_species_query = bool(re.search(
                    r"\bsame\s+(?:plant\s+)?species\b", lowered))
                query_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query))
                def fact_dates(fact):
                    out = set()
                    for turn_id in fact.get("source_turn_ids", []):
                        mem_id = self.turn_to_mem.get(turn_id)
                        mem = self.tools.sem_store.get_memory(mem_id) if mem_id else None
                        if mem and mem.get("date"):
                            out.add(str(mem["date"]))
                    return out
                ranked_sets = [
                    f for f in ranked
                    if not query_dates or query_dates & fact_dates(f)
                ]

                generic_label_tokens = {
                    "the", "a", "an", "and", "or", "of", "in", "at", "from",
                    "user", "shared", "additional", "collection", "option", "options",
                    "image", "images", "photo", "photos", "picture", "pictures",
                    "flower", "flowers", "dress", "dresses", "view", "views",
                }
                def label_keywords(label):
                    return set(self._tokens(label)) - generic_label_tokens
                set_facts_by_label = defaultdict(list)
                for fact in self.facts:
                    if fact.get("fact_kind") == "visual_set" and fact.get("set_label"):
                        set_facts_by_label[fact["set_label"]].append(fact)

                # Query-image collection questions first classify the image to a
                # candidate visual group, then expose the group's complete ID set.
                if same_species_query:
                    if question_image_path and os.path.exists(question_image_path):
                        candidate_rows = self.rank_visual_sets_by_image(
                            question_image_path, ranked_sets)[:3]
                    else:
                        candidate_rows = ranked_sets[:3]
                    candidate_labels = [
                        row.get("set_label") for row in candidate_rows
                        if row.get("set_label")]
                    label_scores = {
                        row.get("set_label"): float(row.get("retrieval_score", 0.0))
                        for row in candidate_rows if row.get("set_label")
                    }
                    self.last_route["visual_set_candidates"] = [
                        {"label": row.get("set_label"),
                         "score": float(row.get("retrieval_score", 0.0))}
                        for row in candidate_rows
                    ]
                    chosen_sets = []
                    covered_ids = set()
                    for candidate_rank, label in enumerate(candidate_labels[:1], 1):
                        ids = {
                            image_id for fact in set_facts_by_label[label]
                            for image_id in fact.get("source_image_ids", [])
                        }
                        keys = label_keywords(label)
                        if len(keys) >= 2:
                            for row in self.visual_cards:
                                card_tokens = set(self._tokens(row["annotation_text"]))
                                if len(keys & card_tokens) >= 2:
                                    ids.add(row["image_id"])
                        role = "primary" if candidate_rank == 1 else "alternative"
                        chosen_sets.extend(set_facts_by_label[label])
                        covered_ids.update(ids)
                        annotations.append(
                            f"visual_species_{role}_group: "
                            f"soft_score={label_scores[label]:.4f}; category='{label}'; "
                            f"{len(ids)} distinct image_ids = " + ", ".join(sorted(ids)))
                else:
                    chosen_sets = ranked_sets[:1]
                    # Broad totals such as "all dress pictures" can span adjacent
                    # named subcategories; named entities such as moon orchid cannot.
                    broad_query_tokens = set(self._tokens(query)) & {
                        "dress", "dresses", "skirt", "skirts", "kimono", "kimonos",
                        "chart", "charts", "plot", "plots", "script", "scripts",
                    }
                    if multi_set_query and broad_query_tokens and chosen_sets:
                        for fact in ranked_sets[1:]:
                            if broad_query_tokens & set(self._tokens(fact.get("set_label", ""))):
                                chosen_sets.append(fact)
                            if len(chosen_sets) >= 3:
                                break
                    covered_ids = {
                        image_id for fact in chosen_sets
                        for image_id in fact.get("source_image_ids", [])
                    }
                    annotations.extend(
                        "visual_set: " + fact["text"] + " Supporting evidence: "
                        + fact.get("retrieval_text", fact["text"])
                        for fact in chosen_sets)

                    # Cross-session expansion is allowed only for explicit total/all
                    # requests and requires at least two entity-keyword matches.
                    if multi_set_query and chosen_sets:
                        keys = label_keywords(chosen_sets[0].get("set_label", ""))
                        if len(keys) >= 2:
                            for row in self.visual_cards:
                                if row["image_id"] in covered_ids:
                                    continue
                                if query_dates and not query_dates.intersection(row.get("dates", [])):
                                    continue
                                card_tokens = set(self._tokens(row["annotation_text"]))
                                if len(keys & card_tokens) >= 2:
                                    annotations.append(
                                        "visual_set_extra_candidate: " + row["annotation_text"])
                                    covered_ids.add(row["image_id"])

                    if covered_ids:
                        annotations.append(
                            "fact_layer_verified_count: Across the matched visual "
                            f"evidence, the user shared {len(covered_ids)} images "
                            "matching the requested collection. The distinct image_ids "
                            "are " + ", ".join(sorted(covered_ids)) + ".")
            if ordered and annotations:
                node_refer_ids = []
                for fact in chosen_sets:
                    node_refer_ids.extend(
                        fact.get("refer_ids", fact.get("source_turn_ids", [])))
                for row in self.visual_cards:
                    if row["image_id"] not in covered_ids:
                        continue
                    for mem_id in row.get("mem_ids", []):
                        node_refer_ids.extend(self.mem_to_refer_ids.get(mem_id, []))
                hierarchy_node = {
                    "mem_id": "hier:retrieved_collection",
                    "text": (
                        "[Retrieved collection node]\n"
                        "This node is the highest-scoring hierarchical memory and "
                        "is grounded by the refer_ids below.\n" +
                        "\n".join(annotations)
                    ),
                    "date": None,
                    "caption": {},
                    "refer_ids": list(dict.fromkeys(node_refer_ids)),
                    "score": 1.0,
                }
                promoted = []
                promoted_mem_ids = set()
                for refer_id in hierarchy_node["refer_ids"]:
                    mem_id = self.turn_to_mem.get(refer_id)
                    if not mem_id or mem_id in promoted_mem_ids:
                        continue
                    mem = self.tools.sem_store.get_memory(mem_id)
                    if not mem:
                        continue
                    promoted.append(self.tools._to_retrieved(mem))
                    promoted_mem_ids.add(mem_id)
                remaining = [
                    row for row in ordered
                    if row.get("mem_id") not in promoted_mem_ids
                ]
                ordered = ([hierarchy_node] + promoted + remaining)[:30]
                self.last_route["expanded_refer_count"] = len(promoted)
            return ordered, [], annotations
        fact_rows = self.rank_source_memories(
            query, allow_visual_sets=False, max_per_source=False)
        by_id = {row.get("mem_id"): row for row in base if row.get("mem_id")}
        for row in fact_rows:
            by_id.setdefault(row.get("mem_id"), row)
        score = defaultdict(float)
        # Weighted RRF: v2 remains the dominant channel; Fact rescues missed evidence.
        for rank, row in enumerate(base, 1):
            score[row["mem_id"]] += 1.0 / (60 + rank)
        fact_weight = float(os.environ.get("FACT_RRF_WEIGHT", "0.75"))
        for rank, row in enumerate(fact_rows, 1):
            score[row["mem_id"]] += fact_weight / (60 + rank)
        ordered = sorted(by_id.values(), key=lambda row: score[row["mem_id"]], reverse=True)[:30]
        base_ids = {row.get("mem_id") for row in base}
        injected = [row["mem_id"] for row in ordered if row["mem_id"] not in base_ids]
        temporal_query = bool(re.search(
            r"\b(before|after|earlier|later|first|last|then|timeline|"
            r"chronological|in order|sequence)\b", lowered))
        if temporal_query:
            def mem_num(row):
                try: return int(row.get("mem_id", "0").split(":")[-1])
                except Exception: return 0
            ordered.sort(key=lambda row: (row.get("timestamp", 0) or 0, mem_num(row)))
            self.last_route["temporal_sort"] = 1.0
        return ordered, injected, []


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
                           max_long_edge: int = 768, quality: int = 75) -> list[dict]:
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
    route_strategy = os.environ.get("ROUTE_ABLATION_STRATEGY", "").strip()
    mode = "baseline" if baseline else "multi-round"
    if route_strategy:
        mode = f"route-{route_strategy}"
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
    all_qa = [dict(qa, _eval_qa_id=qa.get("question_id", f"Q{i+1}"))
              for i, qa in enumerate(case_data["human-annotated QAs"])]
    qa_points = {x.strip() for x in os.environ.get("QA_POINTS", "").split(",") if x.strip()}
    if qa_points:
        all_qa = [qa for qa in all_qa if qa.get("point") in qa_points]
    qa_ids = {x.strip() for x in os.environ.get("QA_IDS", "").split(",") if x.strip()}
    if qa_ids:
        all_qa = [qa for qa in all_qa if qa.get("_eval_qa_id") in qa_ids]
    character = case_data.get("character_profile", {}).get("name", "unknown")
    sessions = case_data.get("multi_session_dialogues", [])
    last_date = sessions[-1].get("date", "2024-12-31") if sessions else "2024-12-31"
    print(f"[2] Evaluating {len(all_qa)} QA (character={character})", flush=True)

    tools = RetrievalTools()
    fact_fusion = FactFusion(case_name, tools)
    final_top_k = configured_top_k()
    route_executor = RouteExecutor(
        tools, fact_fusion, _batch_encode, top_k=final_top_k
    )
    # orchestrator 复用：避免每个 QA 创建新实例导致 SQLite 连接堆积 → "database is locked"
    base_result_file = os.environ.get("BASE_RESULT_FILE")
    base_results = {}
    if base_result_file:
        base_results = {row["qa_id"]: row for row in json.load(open(base_result_file))}
        print(f"  [paired] fixed v2 contexts from {base_result_file}", flush=True)
    route_plan_result_file = os.environ.get("ROUTE_PLAN_RESULT_FILE")
    frozen_route_plans = {}
    if route_plan_result_file:
        frozen_route_plans = {
            str(row["qa_id"]): row["route_plan"]
            for row in json.load(open(route_plan_result_file))
            if row.get("qa_id") and row.get("route_plan")
        }
        print(
            f"  [paired] {len(frozen_route_plans)} frozen route plans from "
            f"{route_plan_result_file}", flush=True,
        )
    _orch = None
    if not baseline and not base_results:
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
        qa_id = qa["_eval_qa_id"]

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
        fact_injected_mem_ids = []
        fact_annotations = []
        base_row = None
        route_plan = None
        route_execution = None

        if route_strategy:
            has_question_image = bool(
                question_image_abs and os.path.exists(question_image_abs)
            )

            def _router(messages):
                return call_llm(messages, max_tokens=512).strip()

            if frozen_route_plans:
                if qa_id not in frozen_route_plans:
                    raise KeyError(f"missing frozen route plan for {case_name}/{qa_id}")
                route_plan = json.loads(json.dumps(frozen_route_plans[qa_id]))
            elif route_strategy == "heuristic_gate":
                route_plan = heuristic_plan(search_query, has_question_image)
            elif route_strategy == "mg_keyword_r1_port":
                route_plan = mg_keyword_r1_plan(search_query, has_question_image)
            elif route_strategy == "model_type_fixed_route":
                route_plan = model_type_fixed_plan(
                    search_query, has_question_image, _router
                )
            elif route_strategy == "two_level_dynamic_route":
                route_plan = two_level_dynamic_plan(
                    search_query, has_question_image, _router
                )
            elif route_strategy == "unified_hybrid_route":
                route_plan = unified_hybrid_plan(
                    search_query, has_question_image, _router
                )
            else:
                raise ValueError(f"unknown ROUTE_ABLATION_STRATEGY={route_strategy}")
            if (
                route_strategy == "unified_hybrid_route"
                and iterative_retrieval_enabled()
            ):
                route_execution = retrieve_iteratively(
                    question=search_query,
                    has_question_image=has_question_image,
                    initial_plan=route_plan,
                    execute=lambda round_plan: route_executor.execute(
                        round_plan, question_image_path=question_image_abs
                    ),
                    call_router=_router,
                    top_k=final_top_k,
                )
                route_plan = route_execution["route_plan"]
                orch_iters = max(
                    0,
                    int(route_execution["iterative_retrieval"]["rounds_executed"]) - 1,
                )
            else:
                route_execution = route_executor.execute(
                    route_plan, question_image_path=question_image_abs
                )
            retrieved = route_execution["context"]
            need_imgs = bool(route_execution.get("need_memory_images"))
            fact_fusion.last_route = {
                "strategy": route_strategy,
                "plan": route_plan,
                "execution": {
                    "trace": route_execution.get("trace", []),
                    "expanded_refer_count": route_execution.get(
                        "expanded_refer_count", 0
                    ),
                    "selected_collection": route_execution.get(
                        "selected_collection"
                    ),
                },
                "used_benchmark_point": False,
            }
            logger.log_round(
                0,
                {"query_type": route_strategy, "tool_choice": "route_plan"},
                retrieved,
                len(retrieved),
                {"route_plan": route_plan, "route_trace": route_execution.get("trace", [])},
                None,
                label="Final memory for answer",
            )
        elif baseline:
            # baseline: 直接 search_memory(search_query)（和 LTMemory 一样），不走多轮
            retrieved = tools.search_memory(search_query)
            need_imgs = False   # baseline 不走 orchestrator，无 need_memory_images
            logger.log_round(0, {"query_type": "baseline", "tool_choice": "search_memory"},
                             retrieved, len(retrieved), {}, None,
                             label="Final memory for answer")
        else:
            # 多轮 orchestrator（Round 0 → sufficiency → rewrite+route）
            if base_results:
                base_row = base_results[qa_id]
                retrieved = []
                for mem_id in base_row.get("retrieved_mem_ids", []):
                    mem = tools.sem_store.get_memory(mem_id)
                    if mem:
                        retrieved.append(tools._to_retrieved(mem))
                orch_result = {
                    "context": retrieved,
                    "iter": base_row.get("orch_iters", 0),
                    "confidence": 0,
                    "need_memory_images": False,
                }
                log_capture = io.StringIO()
            else:
                orch = _orch  # 复用单例，避免 SQLite 连接堆积
                log_capture = io.StringIO()
                with contextlib.redirect_stdout(log_capture):
                    orch_result = orch.retrieve(search_query, point=point)
                retrieved = orch_result.get("context", [])
            retrieved, fact_injected_mem_ids, fact_annotations = fact_fusion.fuse(
                search_query, retrieved, question_image_path=question_image_abs)
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

        has_fact_annotation = bool(fact_annotations)
        answer_memory_image_ids = []
        answer_question_image_abs = question_image_abs
        if (
            fact_fusion.last_route.get("collection", 0.0) > 0
            and any(row.get("mem_id") == "hier:retrieved_collection"
                    for row in retrieved)
            and question_image_abs
        ):
            # The selected node has promoted its referenced bottom evidence, so
            # the existing multimodal answer prompt can compare those memory
            # images with the query image instead of relying on captions alone.
            need_imgs = True
            fact_fusion.last_route["memory_images_for_answer"] = True
        reuse_control_output = os.environ.get("REUSE_CONTROL_OUTPUT", "1") != "0"
        answer_retrieved, bundle_diagnostics = answer_view(retrieved, plan=route_plan)
        bundle_active, _ = evidence_bundle_active(route_plan)
        same_as_fixed_control = bool(
            reuse_control_output
            and not bundle_active
            and
            base_row is not None
            and not has_fact_annotation
            and [row.get("mem_id") for row in retrieved]
                == list(base_row.get("retrieved_mem_ids", []))
        )

        # V2: 答题（对齐 LTMemory fast_run_with_textual_memory）
        # caption 已拼进 search_query（检索）+ 记忆 text（ingest），answer 模型能看到 caption 文本。
        # need_imgs=True（with_visual 实验）：额外传 top-K 记忆原图 + 问题图给答题模型。
        # 否则：有 question_image 传问题图（answer_messages_with_image），无则纯文本。
        if same_as_fixed_control:
            prediction = str(base_row.get("prediction", ""))
        elif need_imgs:
            image_context = (
                route_execution.get("memory_image_candidates")
                if route_execution else None
            ) or retrieved
            mem_imgs = _collect_memory_images(image_context, max_k=5)
            answer_memory_image_ids = [row["image_id"] for row in mem_imgs]
            qimg_b64 = None
            if answer_question_image_abs and os.path.exists(answer_question_image_abs):
                try:
                    qimg_b64 = _compress_image_to_b64(
                        answer_question_image_abs,
                        max_long_edge=768,
                        quality=75,
                    )
                except Exception:
                    qimg_b64 = None
            if mem_imgs or qimg_b64:
                try:
                    msgs = answer_messages_with_memory_images(
                        q, answer_retrieved, mem_imgs,
                        question_image_base64=qimg_b64,
                        character=character, last_date=last_date,
                        hint=format_constraint)
                    prediction = call_llm(msgs).strip()
                except Exception as e:
                    # fallback: 图片传不过去（如模型不支持 image_url）退回纯文本
                    print(f"  [warn] answer with memory images failed ({e}), fallback to text-only",
                          flush=True)
                    msgs = answer_messages(q, answer_retrieved, character=character,
                                           last_date=last_date, hint=format_constraint)
                    prediction = call_llm(msgs).strip()
            else:
                # 没有可用记忆图/问题图，退回纯文本
                msgs = answer_messages(q, answer_retrieved, character=character,
                                       last_date=last_date, hint=format_constraint)
                prediction = call_llm(msgs).strip()
        elif answer_question_image_abs and os.path.exists(answer_question_image_abs):
            try:
                img_b64 = _compress_image_to_b64(answer_question_image_abs)
                msgs = answer_messages_with_image(q, answer_retrieved, img_b64,
                                                  character=character, last_date=last_date,
                                                  hint=format_constraint)
                prediction = call_llm(msgs).strip()
            except Exception as e:
                # fallback: gpt-4.1 不支持 image_url 时退回文本模式
                print(f"  [warn] answer with image failed ({e}), fallback to text-only", flush=True)
                msgs = answer_messages(q, answer_retrieved, character=character,
                                       last_date=last_date, hint=format_constraint)
                prediction = call_llm(msgs).strip()
        else:
            msgs = answer_messages(q, answer_retrieved, character=character,
                                   last_date=last_date, hint=format_constraint)
            prediction = call_llm(msgs).strip()

        # LLM-judge（对齐官方 llm_judge.txt）
        if same_as_fixed_control:
            judge_data = {
                "score": float(base_row.get("score", 0.0)),
                "reasoning": "reused fixed-control output because Fact left context unchanged",
            }
        else:
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
            "retrieval_top_k": final_top_k,
            "retrieved_mem_ids": [r.get("mem_id", "") for r in retrieved],
            "retrieved_refer_ids": list(dict.fromkeys(
                refer_id
                for row in retrieved
                for refer_id in (
                    row.get("refer_ids", []) or
                    fact_fusion.mem_to_refer_ids.get(row.get("mem_id", ""), [])
                )
            )),
            "fact_injected_mem_ids": fact_injected_mem_ids,
            "n_fact_injected": len(fact_injected_mem_ids),
            "fact_annotations": fact_annotations,
            "hierarchy_route": dict(fact_fusion.last_route),
            "evidence_bundle": bundle_diagnostics,
            "route_plan": route_plan,
            "route_execution": (
                {
                    "trace": route_execution.get("trace", []),
                    "expanded_refer_count": route_execution.get(
                        "expanded_refer_count", 0
                    ),
                    "selected_collection": route_execution.get(
                        "selected_collection"
                    ),
                    "answer_memory_image_ids": answer_memory_image_ids,
                    "iterative_retrieval": route_execution.get(
                        "iterative_retrieval",
                        {"enabled": False, "rounds_executed": 1},
                    ),
                }
                if route_execution else None
            ),
            "reused_control_output": same_as_fixed_control,
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
