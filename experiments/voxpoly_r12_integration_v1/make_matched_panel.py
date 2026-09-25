#!/usr/bin/env python3
"""Select a deterministic matched QA panel using QA IDs only.

Selection never inspects category, answer, gold, evidence, question wording, or
the memory system's retrieval output. Both evaluation arms consume the exact
same frozen manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_memory import atomic_json


class PanelError(ValueError):
    pass


def qa_ids_only(case_path: Path) -> list[str]:
    doc = json.loads(case_path.read_text(encoding="utf-8"))
    qa = doc.get("qa")
    rows = qa.get("qa_pairs", []) if isinstance(qa, dict) else qa
    if not isinstance(rows, list):
        raise PanelError("case QA must be a list")
    ids = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PanelError(f"QA row {index} is not an object")
        qa_id = str(row.get("qa_id") or "").strip()
        if not qa_id:
            raise PanelError(f"QA row {index} has no qa_id")
        ids.append(qa_id)
    if len(ids) != len(set(ids)):
        raise PanelError("QA IDs are not unique")
    return ids


def build_manifest(case_path: Path, case_name: str, count: int, seed: str) -> dict:
    if count <= 0:
        raise PanelError("panel count must be positive")
    ids = qa_ids_only(case_path)
    if count > len(ids):
        raise PanelError(f"panel count {count} exceeds QA count {len(ids)}")
    ranked = sorted(
        ids,
        key=lambda qa_id: hashlib.sha256(
            f"{seed}|{case_name}|{qa_id}".encode("utf-8")
        ).hexdigest(),
    )
    selected = ranked[:count]
    return {
        "schema_version": "voxpoly-matched-panel.v1",
        "case_name": case_name,
        "source_case": str(case_path.resolve()),
        "source_case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
        "selection_policy": "smallest SHA256(seed|case_name|qa_id)",
        "selection_inputs": ["qa_id"],
        "forbidden_selection_inputs": [
            "question", "category", "benchmark_label", "answer", "gold", "evidence",
            "retrieval_result", "model_score",
        ],
        "seed": seed,
        "source_qa_count": len(ids),
        "panel_count": count,
        "qa_ids": selected,
        "qa_ids_sha256": hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", default="voxpoly-r12-panel-v1")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"refusing to overwrite panel manifest: {args.out}", file=sys.stderr)
        return 2
    try:
        manifest = build_manifest(args.case, args.case_name, args.count, args.seed)
    except (OSError, json.JSONDecodeError, PanelError) as exc:
        print(f"panel selection failed: {exc}", file=sys.stderr)
        return 2
    atomic_json(args.out, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

