#!/usr/bin/env python3
"""Default-off full-case matched R12 versus Audio relation/profile v2 evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
V2_EXPERIMENT = HERE.parents[1]
sys.path.insert(0, str(V2_EXPERIMENT))

from audit_retrieval import AuditContractError, freeze_retrieval, read_object  # noqa: E402
from evaluate_matched25 import (  # noqa: E402
    ExactMessageCache,
    EvaluationContractError,
    MODEL,
    TOP_K,
    _answer_context,
    _canonical_qa,
    _judge,
    sha256_file,
    sha256_json,
    write_json_atomic,
)


ARMS = ("current_r12", "audio_relation_profile_v2")


def run(
    *,
    config_path: Path,
    out_dir: Path,
    seed_cache_path: Path | None,
    max_attempts: int,
    retry_delay_seconds: float,
    call_llm: Any,
    extract_json: Any,
) -> dict[str, Any]:
    from adapters.voxpoly import gold_evidence
    from eval.bench_prompts import answer_messages, judge_messages

    config = read_object(config_path, "full-case config")
    cases = config.get("cases") or []
    if len(cases) != 1 or int(config.get("top_k", -1)) != TOP_K:
        raise EvaluationContractError("full75 evaluation requires one case and Top30")
    case_id = str(cases[0]["case_id"])
    requested = list(cases[0].get("panel_qa_ids") or [])
    expected_count = len(requested)
    if not requested or len(set(requested)) != expected_count:
        raise EvaluationContractError("full-case config must contain nonempty unique QA IDs")

    # Contexts for all requested questions are frozen before canonical answers/gold are opened.
    frozen = freeze_retrieval(config, config_path)
    if int(frozen.get("qa_count", -1)) != expected_count:
        raise EvaluationContractError("retrieval freeze did not produce every requested QA")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "frozen_contexts_before_gold.json", frozen)

    prompt_module = Path(os.environ.get(
        "V2_PIPELINE_ROOT",
        HERE.parents[5] / "vendor/v2_update_pipeline",
    )) / "eval" / "bench_prompts.py"
    run_config = {
        "schema_version": "voxpoly-audio-v2-fullcase-config.v1",
        "config_id": f"{case_id.lower()}_audio_relation_profile_v2_fullcase_gpt41mini_top30",
        "case_id": case_id,
        "model": MODEL,
        "top_k": TOP_K,
        "qa_count": expected_count,
        "control": "frozen_current_r12_context",
        "treatment": "audio_relation_profile_v2",
        "same_answer_prompt": True,
        "same_judge_prompt": True,
        "same_frozen_route": True,
        "same_audio_derived_character": True,
        "uses_oracle_asker": False,
        "uses_benchmark_label_for_retrieval": False,
        "uses_answer_for_retrieval": False,
        "uses_gold_for_retrieval": False,
        "answer_and_judge_prompt_sha256": sha256_file(prompt_module),
        "retrieval_freeze_sha256": frozen["retrieval_freeze_sha256"],
    }
    partial_path = out_dir / "matched_pairs.partial.json"
    final_path = out_dir / "matched_pairs.json"
    if final_path.exists():
        final = read_object(final_path, "full-case result")
        if final.get("status") != "complete" or final.get("run_config") != run_config:
            raise EvaluationContractError("existing final output has a different contract")
        return final
    if partial_path.exists():
        document = read_object(partial_path, "full-case partial")
        if document.get("status") != "partial" or document.get("run_config") != run_config:
            raise EvaluationContractError("existing partial output has a different contract")
    else:
        document = {
            "schema_version": "voxpoly-audio-v2-fullcase-result.v1",
            "status": "partial",
            "run_config": run_config,
            "pairs": [],
        }
    done = {(row["case_id"], row["qa_id"]) for row in document["pairs"]}
    if len(done) != len(document["pairs"]):
        raise EvaluationContractError("partial output contains duplicate QAs")

    cache_path = out_dir / "llm_call_cache.json"
    if not cache_path.exists() and seed_cache_path and seed_cache_path.exists():
        seed = read_object(seed_cache_path, "matched25 seed cache")
        if seed.get("schema_version") != "exact-message-cache.v1":
            raise EvaluationContractError("seed cache schema mismatch")
        write_json_atomic(cache_path, seed)
    canonical = _canonical_qa(config)
    cache = ExactMessageCache(
        cache_path,
        call_llm,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    for position, row in enumerate(frozen["rows"], 1):
        key = (row["case_id"], row["qa_id"])
        if key in done:
            continue
        qa = canonical.get(key)
        if qa is None:
            raise EvaluationContractError(f"canonical QA missing: {key}")
        reference = str(qa.get("answer") or "")
        identity = row.get("query_identity") or {}
        character = (
            str(identity.get("profile_display"))
            if identity.get("confidence") == "high" and identity.get("profile_display")
            else "user"
        )
        dates = [
            str(item.get("date") or "")
            for item in row["arms"]["current_r12"]["context"]
            if item.get("date")
        ]
        last_date = max(dates) if dates else ""
        arms: dict[str, Any] = {}
        for arm_name in ARMS:
            arm = row["arms"][arm_name]
            prediction = cache.call(answer_messages(
                row["question"], _answer_context(arm), character=character, last_date=last_date
            ))
            judged = _judge(cache, extract_json, judge_messages(row["question"], reference, prediction))
            arms[arm_name] = {
                "prediction": prediction,
                "score": judged["score"],
                "judge": judged,
                "context_node_ids": arm["context_node_ids"],
                "retrieved_refer_ids": arm["retrieved_refer_ids"],
                "context_sha256": sha256_json(arm["context"]),
            }
        document["pairs"].append({
            "case_id": key[0],
            "qa_id": key[1],
            "fine_category": qa.get("category"),
            "four_way_category": qa.get("four_way_category"),
            "question": row["question"],
            "reference_answer": reference,
            "gold_evidence": gold_evidence(qa),
            "runtime_character": character,
            "query_identity_confidence": identity.get("confidence", "unresolved"),
            "arms": arms,
        })
        write_json_atomic(partial_path, document)
        print(
            f"[{position}/{expected_count}] {key[0]}/{key[1]} "
            f"r12={arms[ARMS[0]]['score']:.2f} v2={arms[ARMS[1]]['score']:.2f}",
            flush=True,
        )
    if len(document["pairs"]) != expected_count:
        raise EvaluationContractError("evaluation ended without every requested QA pair")
    document["status"] = "complete"
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in document["pairs"]:
        categories[str(pair.get("four_way_category"))].append(pair)
    document["summary"] = {
        "overall": {
            arm: sum(pair["arms"][arm]["score"] for pair in document["pairs"]) / expected_count
            for arm in ARMS
        },
        "by_four_way_category": {
            category: {
                "count": len(rows),
                **{
                    arm: sum(pair["arms"][arm]["score"] for pair in rows) / len(rows)
                    for arm in ARMS
                },
            }
            for category, rows in sorted(categories.items())
        },
    }
    write_json_atomic(final_path, document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-paid-evaluation", action="store_true")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed-cache", type=Path)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--retry-delay-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if not args.enable_paid_evaluation:
        print("paid full-case evaluation is default-off; pass --enable-paid-evaluation")
        return 2
    if not 1 <= args.max_attempts <= 8:
        print("--max-attempts must be in [1,8]")
        return 2
    from lane_llm import call_llm
    from utils.llm_client import extract_json, print_usage_summary

    try:
        result = run(
            config_path=args.config.resolve(strict=True),
            out_dir=args.out_dir.resolve(),
            seed_cache_path=args.seed_cache.resolve(strict=True) if args.seed_cache else None,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            call_llm=call_llm,
            extract_json=extract_json,
        )
    except (AuditContractError, EvaluationContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"full-case evaluation failed closed: {exc}")
        return 2
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print_usage_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
