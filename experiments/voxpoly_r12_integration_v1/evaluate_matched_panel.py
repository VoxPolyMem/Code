#!/usr/bin/env python3
"""Matched content-only vs order-only soft-identity VoxPoly evaluation.

The same router plan and exact same Top-30 candidate set are used for both
arms. Identity can only reorder those candidates using ``speaker_hint`` and
fact-level ``source_speaker_ref`` / ``addressee_refs`` relations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parents[1]
V2_ROOT = Path(os.environ.get(
    "V2_PIPELINE_ROOT",
    WORK_ROOT / "vendor/v2_update_pipeline",
))
for path in (HERE, WORK_ROOT, V2_ROOT):
    while str(path) in sys.path:
        sys.path.remove(str(path))
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(WORK_ROOT))
sys.path.insert(2, str(V2_ROOT))

from build_memory import atomic_json  # noqa: E402
from soft_identity import soft_identity_rerank  # noqa: E402


SCHEMA_VERSION = "voxpoly-soft-identity-matched.v1"
TOP_K = 30


class EvaluationContractError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationContractError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationContractError(f"{label} must be an object")
    return value


def load_panel(path: Path) -> dict[str, Any]:
    panel = read_object(path, "panel manifest")
    if panel.get("schema_version") != "voxpoly-matched-panel.v1":
        raise EvaluationContractError("unsupported panel manifest")
    qa_ids = panel.get("qa_ids")
    if not isinstance(qa_ids, list) or not qa_ids or len(qa_ids) != len(set(qa_ids)):
        raise EvaluationContractError("panel QA IDs must be a nonempty unique list")
    if int(panel.get("panel_count", -1)) != len(qa_ids):
        raise EvaluationContractError("panel count mismatch")
    digest = hashlib.sha256("\n".join(map(str, qa_ids)).encode("utf-8")).hexdigest()
    if panel.get("qa_ids_sha256") != digest:
        raise EvaluationContractError("panel QA-ID hash mismatch")
    if panel.get("selection_inputs") != ["qa_id"]:
        raise EvaluationContractError("panel was not selected from QA IDs alone")
    return panel


class PersistentCallCache:
    """Atomic exact-message cache around a bounded LLM caller."""

    def __init__(
        self,
        path: Path,
        caller: Callable[..., str],
        *,
        model: str,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> None:
        self.path = path
        self.caller = caller
        self.model = model
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        if path.exists():
            document = read_object(path, "LLM call cache")
            if document.get("schema_version") != "exact-message-cache.v1":
                raise EvaluationContractError("unsupported LLM cache schema")
            self.rows = document.get("rows")
            if not isinstance(self.rows, dict):
                raise EvaluationContractError("LLM cache rows must be an object")
        else:
            self.rows: dict[str, dict[str, Any]] = {}

    def _contract(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], str]:
        contract = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        return contract, sha256_json(contract)

    def invalidate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> None:
        _, key = self._contract(messages, max_tokens, response_format)
        if self.rows.pop(key, None) is not None:
            atomic_json(self.path, {
                "schema_version": "exact-message-cache.v1",
                "rows": self.rows,
            })

    def call(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        contract, key = self._contract(messages, max_tokens, response_format)
        if key in self.rows:
            return str(self.rows[key]["response"])
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                kwargs: dict[str, Any] = {"model": self.model}
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                if response_format is not None:
                    kwargs["response_format"] = response_format
                response = str(self.caller(messages, **kwargs))
                self.rows[key] = {
                    "contract_sha256": key,
                    "contract": contract,
                    "response": response,
                    "attempts_for_this_call": attempt,
                }
                atomic_json(self.path, {
                    "schema_version": "exact-message-cache.v1",
                    "rows": self.rows,
                })
                return response
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts and self.retry_delay_seconds > 0:
                    time.sleep(self.retry_delay_seconds * attempt)
        raise RuntimeError(
            f"LLM call failed after {self.max_attempts} finite attempts: {last_error}"
        ) from last_error


def _context_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [
        str(row.get("node_id") or row.get("mem_id") or f"row:{index}")
        for index, row in enumerate(rows)
    ]


def _retrieved_refs(rows: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(
        str(ref) for row in rows for ref in row.get("refer_ids", []) if str(ref)
    ))


def _answer_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "mem_id": row.get("mem_id") or row["node_id"],
            "text": (
                f"[{row.get('speaker', 'unknown')}; {row.get('date', '')}]: "
                f"{row.get('text', '')}"
            ),
        }
        for row in rows
    ]


def _arm_record(
    *,
    prediction: str,
    judged: dict[str, Any],
    context: list[dict[str, Any]],
    route_plan_sha256: str,
    evidence_recall: float | None,
    identity_trace: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prediction": prediction,
        "score": float(judged["score"]),
        "judge": judged,
        "route_plan_sha256": route_plan_sha256,
        "context_node_ids": _context_ids(context),
        "retrieved_refer_ids": _retrieved_refs(context),
        "evidence_recall_at_30": evidence_recall,
        "n_context": len(context),
        "identity_trace": identity_trace,
    }


def _publish_outputs(
    out_dir: Path,
    document: dict[str, Any],
) -> None:
    atomic_json(out_dir / "matched_pairs.json", document)
    common = {
        "schema_version": SCHEMA_VERSION,
        "config": document["config"],
        "panel_manifest_sha256": document["panel_manifest_sha256"],
    }
    for arm in ("content_only", "soft_identity"):
        atomic_json(out_dir / f"{arm}.json", {
            **common,
            "arm": arm,
            "results": [
                {
                    "qa_id": pair["qa_id"],
                    "question": pair["question"],
                    "reference_answer": pair["reference_answer"],
                    "gold_evidence": pair["gold_evidence"],
                    "route_plan": pair["route_plan"],
                    **pair[arm],
                }
                for pair in document["pairs"]
            ],
        })


def run_panel(
    *,
    memory_path: Path,
    panel_path: Path,
    out_dir: Path,
    model: str,
    max_attempts: int,
    retry_delay_seconds: float,
    call_llm: Callable[..., str],
    extract_json: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    # Heavy evaluator imports stay inside the paid entrypoint; zero-API tests
    # exercise the intervention without initializing embeddings or services.
    from adapters.voxpoly import gold_evidence, qa_pairs
    from core.planner import unified_hybrid_plan
    from eval.bench_prompts import answer_messages, judge_messages
    from evaluation.voxpoly import LayerIndex, question_text

    panel = load_panel(panel_path)
    source_case_path = Path(panel["source_case"]).resolve(strict=True)
    if sha256_file(source_case_path) != panel.get("source_case_sha256"):
        raise EvaluationContractError("source case changed after panel selection")
    case = read_object(source_case_path, "source case")
    qa_by_id = {
        str(row.get("qa_id")): row for row in qa_pairs(case)
        if isinstance(row, dict) and row.get("qa_id")
    }
    missing = [qa_id for qa_id in panel["qa_ids"] if qa_id not in qa_by_id]
    if missing:
        raise EvaluationContractError(f"panel QA IDs missing from source: {missing}")

    memory_path = memory_path.resolve(strict=True)
    memory = read_object(memory_path, "r12 memory")
    if memory.get("meta", {}).get("identity_retrieval_enabled") is not False:
        raise EvaluationContractError("base memory must keep identity retrieval disabled")
    index = LayerIndex(memory_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_sha = sha256_file(panel_path)
    config = {
        "model": model,
        "strategy": "unified_hybrid_route",
        "top_k": TOP_K,
        "identity_intervention": "order_only_soft_boost",
        "candidate_set_must_match": True,
        "uses_benchmark_label_for_retrieval": False,
        "uses_gold_for_retrieval": False,
    }
    pairs_path = out_dir / "matched_pairs.json"
    if pairs_path.exists():
        document = read_object(pairs_path, "matched results")
        if (
            document.get("schema_version") != SCHEMA_VERSION
            or document.get("panel_manifest_sha256") != panel_sha
            or document.get("memory_sha256") != sha256_file(memory_path)
            or document.get("config") != config
        ):
            raise EvaluationContractError("existing matched output has a different contract")
    else:
        document = {
            "schema_version": SCHEMA_VERSION,
            "panel_manifest_sha256": panel_sha,
            "memory_sha256": sha256_file(memory_path),
            "config": config,
            "pairs": [],
        }
    done = {str(row["qa_id"]) for row in document["pairs"]}
    if len(done) != len(document["pairs"]):
        raise EvaluationContractError("existing output contains duplicate QA IDs")
    cache = PersistentCallCache(
        out_dir / "llm_call_cache.json",
        call_llm,
        model=model,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    dates = [str(row.get("date") or "") for row in memory["raw"] if row.get("date")]
    last_date = max(dates) if dates else ""

    for position, qa_id in enumerate(panel["qa_ids"], 1):
        qa_id = str(qa_id)
        if qa_id in done:
            continue
        qa = qa_by_id[qa_id]
        question = question_text(qa)
        if not question:
            raise EvaluationContractError(f"empty question: {qa_id}")

        def router(messages: list[dict[str, Any]]) -> str:
            return cache.call(messages, max_tokens=512).strip()

        plan = unified_hybrid_plan(question, False, router)
        if plan.get("router_parse_ok") is not True:
            raise EvaluationContractError(
                f"router did not produce a validated plan for {qa_id}; refusing fallback evaluation"
            )
        if plan.get("uses_benchmark_label") is not False:
            raise EvaluationContractError(f"route plan lacks the no-label contract: {qa_id}")
        route_plan_sha = sha256_json(plan)
        content_context, route_trace, selected_collection = index.execute(plan, top_k=TOP_K)
        if not content_context:
            raise EvaluationContractError(f"retrieval returned no context: {qa_id}")
        identity_context, identity_trace = soft_identity_rerank(
            content_context, plan, memory, enabled=True
        )
        content_ids = _context_ids(content_context)
        identity_ids = _context_ids(identity_context)
        if len(content_ids) > TOP_K or len(identity_ids) > TOP_K:
            raise EvaluationContractError("retrieval exceeded Top-30")
        if len(content_ids) != len(set(content_ids)):
            raise EvaluationContractError("content retrieval returned duplicate candidates")
        if len(content_ids) != len(identity_ids) or set(content_ids) != set(identity_ids):
            raise EvaluationContractError("identity arm changed the raw safety candidate set")

        # Reference fields are accessed only after both retrieval contexts have
        # been frozen. They are used solely by judge/recall accounting.
        reference_answer = str(qa.get("answer") or "")
        gold = gold_evidence(qa)
        retrieved = set(_retrieved_refs(content_context))
        recall = len(retrieved & set(gold)) / len(set(gold)) if gold else None

        content_answer_messages = answer_messages(
            question, _answer_context(content_context), character="user", last_date=last_date
        )
        identity_answer_messages = answer_messages(
            question, _answer_context(identity_context), character="user", last_date=last_date
        )
        content_prediction = cache.call(content_answer_messages).strip()
        identity_prediction = cache.call(identity_answer_messages).strip()

        def judge(prediction: str) -> dict[str, Any]:
            messages = judge_messages(question, reference_answer, prediction)
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    raw = cache.call(messages)
                    value = extract_json(raw)
                    if not isinstance(value, dict):
                        raise EvaluationContractError("judge output is not an object")
                    score = float(value.get("score"))
                    if not 0.0 <= score <= 1.0:
                        raise EvaluationContractError(
                            f"judge score outside [0,1]: {score}"
                        )
                    value["score"] = score
                    return value
                except RuntimeError:
                    # The cache wrapper has already exhausted its bounded
                    # transport retries; do not multiply that retry budget.
                    raise
                except Exception as exc:
                    last_error = exc
                    cache.invalidate(messages)
                    if attempt < max_attempts and retry_delay_seconds > 0:
                        time.sleep(retry_delay_seconds * attempt)
            raise EvaluationContractError(
                f"judge parse failed after {max_attempts} attempts: {last_error}"
            ) from last_error

        content_judge = judge(content_prediction)
        identity_judge = judge(identity_prediction)
        off_trace = {
            "enabled": False,
            "candidate_set_preserved": True,
            "uses_question_text": False,
            "uses_benchmark_label": False,
            "uses_answer": False,
            "uses_gold": False,
            "changed_positions": 0,
        }
        pair = {
            "qa_id": qa_id,
            "question": question,
            "reference_answer": reference_answer,
            "gold_evidence": gold,
            "route_plan": plan,
            "route_plan_sha256": route_plan_sha,
            "route_trace": route_trace,
            "selected_collection": (
                selected_collection.get("node_id") if selected_collection else None
            ),
            "content_only": _arm_record(
                prediction=content_prediction,
                judged=content_judge,
                context=content_context,
                route_plan_sha256=route_plan_sha,
                evidence_recall=recall,
                identity_trace=off_trace,
            ),
            "soft_identity": _arm_record(
                prediction=identity_prediction,
                judged=identity_judge,
                context=identity_context,
                route_plan_sha256=route_plan_sha,
                evidence_recall=recall,
                identity_trace=identity_trace,
            ),
        }
        document["pairs"].append(pair)
        _publish_outputs(out_dir, document)
        print(
            f"[{position}/{len(panel['qa_ids'])}] {qa_id}: "
            f"content={content_judge['score']:.2f} identity={identity_judge['score']:.2f} "
            f"moved={identity_trace.get('changed_positions', 0)}",
            flush=True,
        )
    return document


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-paid-evaluation", action="store_true")
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-delay-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.enable_paid_evaluation:
        print(
            "matched evaluation is default-off; pass --enable-paid-evaluation",
            file=sys.stderr,
        )
        return 2
    if not 1 <= args.max_attempts <= 8:
        print("--max-attempts must be in [1,8]", file=sys.stderr)
        return 2
    if not 0 <= args.retry_delay_seconds <= 300:
        print("--retry-delay-seconds must be in [0,300]", file=sys.stderr)
        return 2

    from utils.llm_client import call_llm, extract_json, print_usage_summary

    try:
        document = run_panel(
            memory_path=args.memory,
            panel_path=args.panel,
            out_dir=args.out_dir,
            model=args.model,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            call_llm=call_llm,
            extract_json=extract_json,
        )
    except (EvaluationContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"matched evaluation failed: {exc}", file=sys.stderr)
        return 2
    pairs = document["pairs"]
    summary = {
        "n": len(pairs),
        "content_only_llm_score": (
            sum(row["content_only"]["score"] for row in pairs) / max(len(pairs), 1)
        ),
        "soft_identity_llm_score": (
            sum(row["soft_identity"]["score"] for row in pairs) / max(len(pairs), 1)
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print_usage_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
