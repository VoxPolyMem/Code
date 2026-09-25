#!/usr/bin/env python3
"""Build the deterministic, no-API summary for the frozen matched-25 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ARM_R12 = "current_r12"
ARM_V2 = "audio_relation_profile_v2"
CATEGORIES = (
    "retrieval_reasoning",
    "memory_evolution_conflict",
    "personalized_qa",
    "interaction_attribution",
)
ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Keep machine-readable provenance stable across local/remote mirrors."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def verdict(new: float, old: float) -> str:
    if new > old:
        return "win"
    if new < old:
        return "loss"
    return "tie"


def wtl(rows: list[dict[str, Any]], new_key: str, old_key: str) -> dict[str, int]:
    counts = {"wins": 0, "ties": 0, "losses": 0}
    for row in rows:
        counts[{"win": "wins", "tie": "ties", "loss": "losses"}[verdict(row[new_key], row[old_key])]] += 1
    return counts


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows)}
    for prefix in ("old", "current_r12", "audio_relation_profile_v2"):
        score_key = f"{prefix}_score"
        result[prefix] = {
            "llm_score": mean(float(row[score_key]) for row in rows),
            "correct_at_0_75": sum(bool(row[f"{prefix}_correct_at_0_75"]) for row in rows),
            "accuracy_at_0_75": mean(float(row[f"{prefix}_correct_at_0_75"]) for row in rows),
        }
    for prefix in ("current_r12", "audio_relation_profile_v2"):
        recall_key = f"{prefix}_evidence_recall_at_30"
        result[prefix]["mean_evidence_recall_at_30"] = mean(float(row[recall_key]) for row in rows)
    result["comparisons"] = {
        "v2_vs_r12_score_wtl": wtl(rows, "audio_relation_profile_v2_score", "current_r12_score"),
        "v2_vs_old_score_wtl": wtl(rows, "audio_relation_profile_v2_score", "old_score"),
        "r12_vs_old_score_wtl": wtl(rows, "current_r12_score", "old_score"),
        "v2_vs_r12_accuracy_at_0_75_wtl": wtl(
            rows, "audio_relation_profile_v2_correct_at_0_75", "current_r12_correct_at_0_75"
        ),
        "v2_vs_old_accuracy_at_0_75_wtl": wtl(
            rows, "audio_relation_profile_v2_correct_at_0_75", "old_correct_at_0_75"
        ),
        "r12_vs_old_accuracy_at_0_75_wtl": wtl(
            rows, "current_r12_correct_at_0_75", "old_correct_at_0_75"
        ),
        "v2_vs_r12_recall_wtl": wtl(
            rows,
            "audio_relation_profile_v2_evidence_recall_at_30",
            "current_r12_evidence_recall_at_30",
        ),
    }
    return result


def parse_usage(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8")
    usage = re.search(
        r"\[LLM usage\]\s+calls=(\d+)\s+prompt=(\d+)\s+completion=(\d+)\s+total_tokens=(\d+)",
        text,
    )
    cost = re.search(r"gpt-4\.1-mini:\s+\d+ calls,\s+\d+\+\d+ tokens,\s+¥([0-9.]+)", text)
    if not usage or not cost:
        raise ValueError(f"usage footer is missing from {log_path}")
    calls, prompt, completion, total = (int(value) for value in usage.groups())
    if prompt + completion != total:
        raise ValueError("prompt + completion token count does not equal total")
    return {
        "provider": "AIGC native",
        "model": "gpt-4.1-mini",
        "calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "aigc_equivalent_cost_cny": float(cost.group(1)),
        "velen_paid_cost_cny": 0.0,
        "http_429_retries": len(re.findall(r"\[429 retry", text)),
        "velen_calls": len(re.findall(r"VELEN-USED", text)),
    }


def build_summary(result_path: Path, audit_path: Path, old_dir: Path, log_path: Path) -> dict[str, Any]:
    result = load_json(result_path)
    audit = load_json(audit_path)
    if result.get("status") != "complete":
        raise ValueError(f"matched result status is not complete: {result.get('status')!r}")
    pairs = result.get("pairs", [])
    if len(pairs) != 25:
        raise ValueError(f"expected 25 matched pairs, found {len(pairs)}")

    keys = [(pair["case_id"], pair["qa_id"]) for pair in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("matched result contains duplicate (case_id, qa_id) keys")

    audit_by_key = {(row["case_id"], row["qa_id"]): row for row in audit["rows"]}
    old_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    old_paths: dict[str, Path] = {}
    for path in sorted(old_dir.glob("CASE_G_*.gpt41mini.json")):
        match = re.search(r"(CASE_G_\d{3})", path.name)
        if not match:
            continue
        case_id = match.group(1)
        old_paths[case_id] = path
        document = load_json(path)
        if document.get("status", "").upper() != "COMPLETE":
            raise ValueError(f"old reference is incomplete: {path}")
        if document.get("model") != "gpt-4.1-mini":
            raise ValueError(f"old reference model mismatch: {path}")
        for decision in document["decisions"]:
            old_by_key[(case_id, decision["qa_id"])] = decision

    aligned: list[dict[str, Any]] = []
    for pair in pairs:
        key = (pair["case_id"], pair["qa_id"])
        if key not in audit_by_key:
            raise KeyError(f"retrieval audit is missing {key}")
        if key not in old_by_key:
            raise KeyError(f"old reference is missing {key}")
        audit_row = audit_by_key[key]
        old = old_by_key[key]
        r12_score = float(pair["arms"][ARM_R12]["score"])
        v2_score = float(pair["arms"][ARM_V2]["score"])
        old_score = float(old["score"])
        r12_recall = float(audit_row["arms"][ARM_R12]["evidence_recall_at_30"])
        v2_recall = float(audit_row["arms"][ARM_V2]["evidence_recall_at_30"])
        aligned.append(
            {
                "case_id": key[0],
                "qa_id": key[1],
                "four_way_category": old["four_way_category"],
                "fine_category": old["fine_category"],
                "old_score": old_score,
                "current_r12_score": r12_score,
                "audio_relation_profile_v2_score": v2_score,
                "old_correct_at_0_75": old_score >= 0.75,
                "current_r12_correct_at_0_75": r12_score >= 0.75,
                "audio_relation_profile_v2_correct_at_0_75": v2_score >= 0.75,
                "current_r12_evidence_recall_at_30": r12_recall,
                "audio_relation_profile_v2_evidence_recall_at_30": v2_recall,
                "v2_minus_r12_score": v2_score - r12_score,
                "v2_minus_r12_recall": v2_recall - r12_recall,
                "v2_vs_r12_score_verdict": verdict(v2_score, r12_score),
                "v2_vs_r12_recall_verdict": verdict(v2_recall, r12_recall),
            }
        )

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aligned:
        by_category[row["four_way_category"]].append(row)
    if set(by_category) != set(CATEGORIES):
        raise ValueError(f"unexpected four-way categories: {sorted(by_category)}")

    overall = aggregate(aligned)
    categories = {category: aggregate(by_category[category]) for category in CATEGORIES}
    acceptance = {
        "personalized_qa_v2_llm_score_gte_old_0_75": categories["personalized_qa"][ARM_V2]["llm_score"] >= 0.75,
        "interaction_attribution_v2_llm_score_gte_old_0_770833": categories["interaction_attribution"][ARM_V2]["llm_score"] >= 0.7708333333333334,
        "retrieval_reasoning_v2_llm_score_gte_r12_0_928571": categories["retrieval_reasoning"][ARM_V2]["llm_score"] >= 0.9285714285714286,
        "retrieval_reasoning_v2_llm_score_gt_old_0_821429": categories["retrieval_reasoning"][ARM_V2]["llm_score"] > 0.8214285714285714,
        "memory_evolution_conflict_v2_llm_score_eq_1": categories["memory_evolution_conflict"][ARM_V2]["llm_score"] == 1.0,
    }
    acceptance["all_pass"] = all(acceptance.values())

    source_files = {
        "matched_result": {"path": display_path(result_path), "sha256": sha256(result_path)},
        "retrieval_audit": {"path": display_path(audit_path), "sha256": sha256(audit_path)},
        "run_log": {"path": display_path(log_path), "sha256": sha256(log_path)},
        "old_references": {
            case_id: {"path": display_path(path), "sha256": sha256(path)}
            for case_id, path in sorted(old_paths.items())
        },
    }
    return {
        "schema_version": "voxpoly-audio-v2-matched25-summary.v1",
        "status": "complete",
        "qa_count": len(aligned),
        "top_k": 30,
        "formal_comparison": {
            "arms": [ARM_R12, ARM_V2],
            "provider": "AIGC native",
            "model": "gpt-4.1-mini",
            "same_answer_and_judge_prompts": True,
            "same_frozen_route": True,
            "audio_identity": "waveform_to_ECAPA_to_frozen_EMA_registry; no oracle asker",
        },
        "old_reference_scope": {
            "role": "diagnostic historical threshold and per-QA alignment only",
            "model": "gpt-4.1-mini",
            "provider": "Velen",
            "not_formally_matched_because": [
                "historical answer context/pipeline differs",
                "historical Persona run used oracle-derived asker identity",
                "provider differs from the formal AIGC matched arms",
            ],
        },
        "overall": overall,
        "by_four_way_category": categories,
        "acceptance": acceptance,
        "usage": parse_usage(log_path),
        "rows": aligned,
        "source_files": source_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT
    parser.add_argument("--result", type=Path, default=base / "artifacts/matched25_eval/matched_pairs.json")
    parser.add_argument("--audit", type=Path, default=base / "artifacts/retrieval_audit.json")
    parser.add_argument("--old-reference-dir", type=Path, default=base / "artifacts/old_reference")
    parser.add_argument("--log", type=Path, default=base / "artifacts/matched25_eval/run_aigc_gpt41mini.log")
    parser.add_argument("--output", type=Path, default=base / "artifacts/matched25_eval/summary.json")
    args = parser.parse_args()
    summary = build_summary(args.result, args.audit, args.old_reference_dir, args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "acceptance": summary["acceptance"]}, indent=2))


if __name__ == "__main__":
    main()
