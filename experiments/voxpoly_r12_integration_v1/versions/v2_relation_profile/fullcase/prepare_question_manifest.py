#!/usr/bin/env python3
"""Project a canonical VoxPoly case onto QA-id/question-only retrieval input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def project(case_path: Path) -> dict[str, Any]:
    document = json.loads(case_path.read_text(encoding="utf-8"))
    qa = document.get("qa")
    rows = qa.get("qa_pairs") if isinstance(qa, dict) else qa
    if not isinstance(rows, list) or len(rows) != 75:
        raise ValueError(f"expected exactly 75 QA rows, found {len(rows) if isinstance(rows, list) else 'invalid'}")
    safe = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"QA row {index} is not an object")
        qa_id = str(row.get("qa_id") or "").strip()
        question = str(row.get("question_text") or row.get("question") or "").strip()
        if not qa_id or not question:
            raise ValueError(f"QA row {index} lacks qa_id/question")
        safe.append({"qa_id": qa_id, "question": question})
    if len({row["qa_id"] for row in safe}) != 75:
        raise ValueError("QA IDs are not unique")
    case_id = str(document.get("case_id") or "").strip()
    del document
    return {
        "schema_version": "voxpoly-full75-question-manifest.v1",
        "status": "complete",
        "case_id": case_id,
        "count": 75,
        "source_case_sha256": sha256_file(case_path),
        "projection_fields": ["qa_id", "question"],
        "forbidden_retrieval_fields": [
            "answer", "gold", "evidence", "category", "four_way_category",
        ],
        "rows": safe,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite question manifest: {args.output}")
        return 2
    try:
        payload = project(args.case.resolve(strict=True))
        atomic_json(args.output.resolve(), payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"question projection failed: {exc}")
        return 2
    print(json.dumps({"output": str(args.output), "case_id": payload["case_id"], "count": 75}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
