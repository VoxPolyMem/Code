#!/usr/bin/env python3
"""Freeze one arbitrary full-case R12 route set before gold access.

The isolated all18 runner uses the same planner and raw Top-30 execution as
the frozen full75 entrypoint.  Only the verified number of QA rows is made
data-dependent so legacy expanded Persona units are not discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[3]
WORK_ROOT = EXPERIMENT.parents[1]
V2_PIPELINE = Path(os.environ.get(
    "V2_PIPELINE_ROOT",
    WORK_ROOT / "vendor/v2_update_pipeline",
))
for candidate in (EXPERIMENT, WORK_ROOT, V2_PIPELINE):
    sys.path.insert(0, str(candidate))

from build_memory import atomic_json  # noqa: E402
from evaluate_matched_panel import PersistentCallCache  # noqa: E402


MODEL = "gpt-4.1-mini"
TOP_K = 30


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def safe_seed_rows(path: Path, questions: dict[str, str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    document = read_object(path, "seed matched result")
    result: dict[str, dict[str, Any]] = {}
    for row in document.get("pairs") or []:
        qa_id = str(row.get("qa_id") or "")
        arm = row.get("content_only") or {}
        node_ids = [str(value) for value in arm.get("context_node_ids") or []]
        question = str(row.get("question") or "")
        if qa_id in questions and question == questions[qa_id] and len(node_ids) == TOP_K:
            result[qa_id] = {
                "question": question,
                "route_plan": row.get("route_plan"),
                "context_node_ids": node_ids,
                "route_source": "validated_prior_matched_panel",
            }
    del document
    return result


def run(
    *,
    case_id: str,
    question_manifest_path: Path,
    memory_path: Path,
    output_dir: Path,
    seed_result_path: Path | None,
    seed_cache_path: Path | None,
    call_llm: Any,
    max_attempts: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    from core.planner import unified_hybrid_plan
    from evaluation.voxpoly import LayerIndex

    question_manifest = read_object(question_manifest_path, "question manifest")
    rows = question_manifest.get("rows")
    if (
        question_manifest.get("status") != "complete"
        or question_manifest.get("case_id") != case_id
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("question manifest is not a complete nonempty case")
    forbidden = {"answer", "gold", "evidence", "category", "four_way_category"}
    if any(forbidden & set(row) for row in rows if isinstance(row, dict)):
        raise ValueError("question manifest contains forbidden retrieval fields")
    questions = {str(row["qa_id"]): str(row["question"]) for row in rows}
    expected_count = len(rows)
    if len(questions) != expected_count or any(not value for value in questions.values()):
        raise ValueError("question manifest contains duplicate/empty entries")

    memory_path = memory_path.resolve(strict=True)
    memory = read_object(memory_path, "R12 memory")
    if memory.get("case_id") != case_id:
        raise ValueError("memory case_id mismatch")
    raw_by_id = {str(row["node_id"]): row for row in memory.get("raw") or []}
    if len(raw_by_id) != len(memory.get("raw") or []):
        raise ValueError("memory raw IDs are not unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "router_call_cache.json"
    if not cache_path.exists() and seed_cache_path and seed_cache_path.exists():
        seed_cache = read_object(seed_cache_path, "seed call cache")
        if seed_cache.get("schema_version") != "exact-message-cache.v1":
            raise ValueError("seed cache schema mismatch")
        atomic_json(cache_path, seed_cache)
    cache = PersistentCallCache(
        cache_path,
        call_llm,
        model=MODEL,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    seed = safe_seed_rows(seed_result_path, questions) if seed_result_path else {}
    for qa_id, item in seed.items():
        if len(set(item["context_node_ids"])) != TOP_K:
            raise ValueError(f"seed context is not unique Top30: {qa_id}")
        if any(node_id not in raw_by_id for node_id in item["context_node_ids"]):
            raise ValueError(f"seed context references foreign raw node: {qa_id}")

    run_config = {
        "schema_version": "voxpoly-fullcase-r12-freeze.v1",
        "case_id": case_id,
        "model": MODEL,
        "top_k": TOP_K,
        "qa_count": expected_count,
        "planner": "unified_hybrid_plan",
        "question_manifest_sha256": sha256_file(question_manifest_path),
        "memory_sha256": sha256_file(memory_path),
        "seed_result_sha256": sha256_file(seed_result_path) if seed_result_path and seed_result_path.exists() else None,
        "uses_benchmark_label": False,
        "uses_answer": False,
        "uses_gold": False,
    }
    partial_path = output_dir / "saved_r12.partial.json"
    final_path = output_dir / "saved_r12.json"
    if final_path.exists():
        final = read_object(final_path, "frozen R12 result")
        if final.get("status") != "complete" or final.get("run_config") != run_config:
            raise ValueError("existing frozen R12 result has a different contract")
        return final
    if partial_path.exists():
        document = read_object(partial_path, "partial R12 freeze")
        if document.get("status") != "partial" or document.get("run_config") != run_config:
            raise ValueError("existing partial R12 freeze has a different contract")
    else:
        document = {
            "schema_version": "voxpoly-fullcase-saved-r12.v1",
            "status": "partial",
            "run_config": run_config,
            "pairs": [],
        }
    done = {str(row["qa_id"]) for row in document["pairs"]}
    if len(done) != len(document["pairs"]):
        raise ValueError("partial R12 freeze contains duplicate QA IDs")
    index = LayerIndex(memory_path)

    for position, source in enumerate(rows, 1):
        qa_id = str(source["qa_id"])
        question = str(source["question"])
        if qa_id in done:
            continue
        if qa_id in seed:
            frozen = seed[qa_id]
            context_ids = frozen["context_node_ids"]
            route_plan = frozen["route_plan"]
            route_source = frozen["route_source"]
            trace = {"source": route_source}
        else:
            def router(messages: list[dict[str, Any]]) -> str:
                return cache.call(messages, max_tokens=512).strip()

            route_plan = unified_hybrid_plan(question, False, router)
            if route_plan.get("router_parse_ok") is not True:
                raise ValueError(f"planner parse failed closed: {case_id}/{qa_id}")
            if route_plan.get("uses_benchmark_label") is not False:
                raise ValueError(f"planner label contract failed: {case_id}/{qa_id}")
            context, trace, _selected = index.execute(route_plan, top_k=TOP_K)
            context_ids = [str(row.get("node_id") or "") for row in context]
            route_source = "new_frozen_gpt41mini_route"
        if len(context_ids) != TOP_K or len(set(context_ids)) != TOP_K:
            raise ValueError(f"R12 context is not unique Top30: {case_id}/{qa_id}")
        if any(node_id not in raw_by_id for node_id in context_ids):
            raise ValueError(f"R12 context is not raw-projected: {case_id}/{qa_id}")
        document["pairs"].append({
            "case_id": case_id,
            "qa_id": qa_id,
            "question": question,
            "route_plan": route_plan,
            "route_source": route_source,
            "content_only": {
                "context_node_ids": context_ids,
                "retrieved_refer_ids": [
                    str(ref)
                    for node_id in context_ids
                    for ref in raw_by_id[node_id].get("refer_ids") or []
                ],
                "trace": trace,
            },
        })
        atomic_json(partial_path, document)
        print(f"[{position}/{expected_count}] {case_id}/{qa_id} source={route_source}", flush=True)

    if len(document["pairs"]) != expected_count:
        raise ValueError("R12 freeze ended without every requested QA")
    document["status"] = "complete"
    document["summary"] = {
        "qa_count": expected_count,
        "seeded_routes": sum(row["route_source"] == "validated_prior_matched_panel" for row in document["pairs"]),
        "new_routes": sum(row["route_source"] == "new_frozen_gpt41mini_route" for row in document["pairs"]),
    }
    atomic_json(final_path, document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-paid-routing", action="store_true")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--question-manifest", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed-result", type=Path)
    parser.add_argument("--seed-cache", type=Path)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--retry-delay-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if not args.enable_paid_routing:
        print("paid route freeze is default-off; pass --enable-paid-routing")
        return 2
    if not 1 <= args.max_attempts <= 8:
        print("--max-attempts must be in [1,8]")
        return 2
    os.environ.setdefault("EMBEDDING_SERVER_URL", "http://127.0.0.1:9981")
    from lane_llm import call_llm
    from utils.llm_client import print_usage_summary

    try:
        result = run(
            case_id=args.case_id,
            question_manifest_path=args.question_manifest.resolve(strict=True),
            memory_path=args.memory,
            output_dir=args.out_dir.resolve(),
            seed_result_path=args.seed_result.resolve(strict=True) if args.seed_result else None,
            seed_cache_path=args.seed_cache.resolve(strict=True) if args.seed_cache else None,
            call_llm=call_llm,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"full-case R12 freeze failed closed: {exc}")
        return 2
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print_usage_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
