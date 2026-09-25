#!/usr/bin/env python3
"""Derive a reproducible four-way AudioMem label sidecar from frozen results.

The original matched-pair artifacts are immutable evaluation records.  This
script therefore writes a separate label sidecar rather than modifying their
scores, prompts, or retrieval traces.  It preserves the formal Gen3 labels
when present and applies the same four-way taxonomy to legacy fine-grained
labels and to the two older ID-only case formats.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


FORMAL = {
    "interaction_attribution",
    "memory_evolution_conflict",
    "personalized_qa",
    "retrieval_reasoning",
}

FINE_TO_FOUR_WAY = {
    "attribution": "interaction_attribution",
    "conflict": "memory_evolution_conflict",
    "conflict_detection": "memory_evolution_conflict",
    "supersede": "memory_evolution_conflict",
    "personalized": "personalized_qa",
    "personalized_retrieval": "personalized_qa",
    "personalized_adversarial": "personalized_qa",
    "adversarial": "retrieval_reasoning",
    "multi_hop": "retrieval_reasoning",
    "multi_hop_retrieval": "retrieval_reasoning",
    "single_hop": "retrieval_reasoning",
    "single_hop_retrieval": "retrieval_reasoning",
    "temporal": "retrieval_reasoning",
    "temporal_order": "retrieval_reasoning",
}

ID_PREFIX_TO_FOUR_WAY = {
    "ADDRESSEE_ATTRIBUTION": "interaction_attribution",
    "SPEAKER_ATTRIBUTION": "interaction_attribution",
    "PERSONALIZATION": "personalized_qa",
    "PERSONALIZATION_APPLICATION": "personalized_qa",
    "CONFLICT_DETECTION": "memory_evolution_conflict",
    "ADVERSARIAL": "retrieval_reasoning",
    "MULTI_HOP": "retrieval_reasoning",
    "SINGLE_HOP": "retrieval_reasoning",
    "TEMPORAL": "retrieval_reasoning",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def from_qa_id(qa_id: str) -> str | None:
    for prefix, category in ID_PREFIX_TO_FOUR_WAY.items():
        if qa_id == prefix or qa_id.startswith(prefix + "_"):
            return category
    return None


def label_pair(pair: dict[str, Any]) -> tuple[str, str]:
    formal = pair.get("four_way_category")
    if formal in FORMAL:
        return str(formal), "formal_four_way_category"
    fine = pair.get("fine_category")
    if fine in FINE_TO_FOUR_WAY:
        return FINE_TO_FOUR_WAY[str(fine)], "legacy_fine_category"
    recovered = from_qa_id(str(pair.get("qa_id") or ""))
    if recovered is not None:
        return recovered, "qa_id_prefix"
    raise ValueError(
        "no four-way mapping for "
        f"{pair.get('case_id')}/{pair.get('qa_id')} fine={fine!r} formal={formal!r}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-cases", type=int, default=18)
    parser.add_argument("--expected-qa", type=int, default=1527)
    args = parser.parse_args()

    result_paths = sorted(args.results_root.glob("CASE_G_*/matched_pairs.json"))
    if len(result_paths) != args.expected_cases:
        raise ValueError(f"expected {args.expected_cases} completed cases, found {len(result_paths)}")

    labels: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for result_path in result_paths:
        document = json.loads(result_path.read_text(encoding="utf-8"))
        if document.get("status") != "complete":
            raise ValueError(f"incomplete result: {result_path}")
        for pair in document.get("pairs") or []:
            case_id = str(pair.get("case_id") or "")
            qa_id = str(pair.get("qa_id") or "")
            key = (case_id, qa_id)
            if not all(key) or key in seen:
                raise ValueError(f"invalid or duplicate QA key: {key}")
            seen.add(key)
            category, source = label_pair(pair)
            score = float((pair.get("arm") or {}).get("score"))
            labels.append({
                "case_id": case_id,
                "qa_id": qa_id,
                "four_way_category": category,
                "label_source": source,
                "original_fine_category": pair.get("fine_category"),
                "original_four_way_category": pair.get("four_way_category"),
                "llm_score": score,
                "correct_at_0_75": score >= 0.75,
            })
    if len(labels) != args.expected_qa:
        raise ValueError(f"expected {args.expected_qa} QA rows, found {len(labels)}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_counts: dict[str, int] = defaultdict(int)
    for row in labels:
        groups[row["four_way_category"]].append(row)
        source_counts[row["label_source"]] += 1
    if set(groups) != FORMAL:
        raise ValueError(f"four-way categories are incomplete: {sorted(groups)}")
    summary = {
        category: {
            "count": len(rows),
            "average_llm_score": sum(float(row["llm_score"]) for row in rows) / len(rows),
            "correct_at_0_75": sum(bool(row["correct_at_0_75"]) for row in rows),
            "accuracy_at_0_75": sum(bool(row["correct_at_0_75"]) for row in rows) / len(rows),
        }
        for category, rows in sorted(groups.items())
    }
    document = {
        "schema_version": "voxpoly-four-way-relabel.v1",
        "taxonomy": {
            "interaction_attribution": ["attribution", "speaker attribution", "addressee attribution"],
            "memory_evolution_conflict": ["conflict", "conflict detection", "supersede"],
            "personalized_qa": [
                "personalized", "personalized_retrieval", "personalized_adversarial",
                "personalization", "personalization_application",
            ],
            "retrieval_reasoning": [
                "single-hop", "multi-hop", "temporal", "adversarial", "retrieval variants",
            ],
        },
        "mapping": {
            "formal_categories": sorted(FORMAL),
            "fine_to_four_way": FINE_TO_FOUR_WAY,
            "qa_id_prefix_to_four_way": ID_PREFIX_TO_FOUR_WAY,
        },
        "input": {
            "results_root": str(args.results_root.resolve()),
            "case_count": len(result_paths),
            "qa_count": len(labels),
        },
        "label_source_counts": dict(sorted(source_counts.items())),
        "summary": summary,
        "labels": labels,
    }
    atomic_json(args.output, document)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
