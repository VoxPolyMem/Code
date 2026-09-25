#!/usr/bin/env python3
"""Zero-API verifier for matched soft-identity evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_memory import atomic_json
from evaluate_matched_panel import SCHEMA_VERSION, load_panel, read_object, sha256_json
from soft_identity import MAX_IDENTITY_BOOST


class VerificationError(ValueError):
    pass


def verify(out_dir: Path, panel_path: Path, *, require_complete: bool) -> dict[str, Any]:
    panel = load_panel(panel_path)
    document = read_object(out_dir / "matched_pairs.json", "matched pairs")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise VerificationError("matched-pair schema mismatch")
    panel_sha = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    if document.get("panel_manifest_sha256") != panel_sha:
        raise VerificationError("matched output belongs to a different panel")
    config = document.get("config") or {}
    if (
        config.get("top_k") != 30
        or config.get("identity_intervention") != "order_only_soft_boost"
        or config.get("candidate_set_must_match") is not True
        or config.get("uses_benchmark_label_for_retrieval") is not False
        or config.get("uses_gold_for_retrieval") is not False
    ):
        raise VerificationError("evaluation config violates the soft-identity contract")

    pairs = document.get("pairs")
    if not isinstance(pairs, list):
        raise VerificationError("pairs must be a list")
    ids = [str(row.get("qa_id")) for row in pairs]
    if len(ids) != len(set(ids)):
        raise VerificationError("duplicate QA IDs in matched output")
    expected_prefix = [str(value) for value in panel["qa_ids"][:len(ids)]]
    if ids != expected_prefix:
        raise VerificationError("completed QA IDs are not the frozen panel prefix")
    if require_complete and ids != list(map(str, panel["qa_ids"])):
        raise VerificationError(f"panel incomplete: {len(ids)}/{len(panel['qa_ids'])}")

    moved_questions = 0
    hint_resolved_questions = 0
    for row in pairs:
        route_sha = sha256_json(row.get("route_plan"))
        if route_sha != row.get("route_plan_sha256"):
            raise VerificationError(f"route-plan hash mismatch: {row.get('qa_id')}")
        content = row.get("content_only") or {}
        identity = row.get("soft_identity") or {}
        if content.get("route_plan_sha256") != route_sha or identity.get("route_plan_sha256") != route_sha:
            raise VerificationError(f"arms do not share one route plan: {row.get('qa_id')}")
        content_ids = content.get("context_node_ids") or []
        identity_ids = identity.get("context_node_ids") or []
        if len(content_ids) > 30 or len(identity_ids) > 30:
            raise VerificationError(f"Top-30 violation: {row.get('qa_id')}")
        if len(content_ids) != len(identity_ids) or set(content_ids) != set(identity_ids):
            raise VerificationError(f"candidate-set change: {row.get('qa_id')}")
        if set(content.get("retrieved_refer_ids") or []) != set(identity.get("retrieved_refer_ids") or []):
            raise VerificationError(f"raw safety recall set changed: {row.get('qa_id')}")
        if content.get("evidence_recall_at_30") != identity.get("evidence_recall_at_30"):
            raise VerificationError(f"recall accounting differs: {row.get('qa_id')}")
        trace = identity.get("identity_trace") or {}
        if any(trace.get(key) is not False for key in (
            "uses_question_text", "uses_benchmark_label", "uses_answer", "uses_gold"
        )):
            raise VerificationError(f"forbidden rerank input recorded: {row.get('qa_id')}")
        if trace.get("candidate_set_preserved") is not True:
            raise VerificationError(f"candidate preservation not asserted: {row.get('qa_id')}")
        for detail in trace.get("boost_details") or []:
            boost = float(detail.get("boost", -1))
            if not 0.0 < boost <= MAX_IDENTITY_BOOST:
                raise VerificationError(f"boost outside cap: {row.get('qa_id')}")
            if not set(detail.get("reasons") or []).issubset(
                {"source_speaker_ref", "addressee_refs"}
            ):
                raise VerificationError(f"unknown boost reason: {row.get('qa_id')}")
        moved_questions += int(int(trace.get("changed_positions", 0)) > 0)
        hint_resolved_questions += int(bool(trace.get("resolved_speaker_refs")))

    content_score = sum(float(row["content_only"]["score"]) for row in pairs) / max(len(pairs), 1)
    identity_score = sum(float(row["soft_identity"]["score"]) for row in pairs) / max(len(pairs), 1)
    report = {
        "schema_version": "voxpoly-soft-identity-verification.v1",
        "status": "complete" if len(pairs) == len(panel["qa_ids"]) else "partial_valid",
        "verified_pairs": len(pairs),
        "expected_pairs": len(panel["qa_ids"]),
        "same_question_order": True,
        "same_route_plan": True,
        "same_candidate_set": True,
        "same_raw_recall_set": True,
        "top_k": 30,
        "identity_hints_resolved": hint_resolved_questions,
        "questions_with_reordered_context": moved_questions,
        "content_only_llm_score": content_score,
        "soft_identity_llm_score": identity_score,
        "delta": identity_score - content_score,
        "uses_benchmark_label_for_retrieval": False,
        "uses_gold_for_retrieval": False,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify(args.out_dir, args.panel, require_complete=args.require_complete)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 2
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

