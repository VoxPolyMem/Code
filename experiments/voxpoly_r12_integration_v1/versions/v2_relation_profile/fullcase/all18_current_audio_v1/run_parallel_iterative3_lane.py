#!/usr/bin/env python3
"""Evaluate a disjoint, resumable case subset as each memory becomes ready.

This runner is intentionally separate from the original serial lane.  It only
splits cases only after their own immutable memory artifact is complete.  It
does not wait for unrelated cases in the same build lane, preserving the
frozen memory and QA protocol while minimizing idle evaluation time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get("PYTHON_BIN", sys.executable))


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def state_write(path: Path, **fields: Any) -> None:
    prior = json.loads(path.read_text()) if path.exists() else {}
    prior.update(fields)
    prior["updated_at"] = now()
    atomic_json(path, prior)


def memory_is_ready(path: Path, case_id: str) -> bool:
    """Match the immutable contextual-memory build contract without QA access."""

    if not path.exists():
        return False
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    meta = document.get("meta") or {}
    return (
        document.get("schema_version") == "voxpoly-r12-contextual-facts-v1"
        and document.get("case_id") == case_id
        and meta.get("qa_consumed") is False
        and meta.get("gold_evidence_consumed") is False
        and meta.get("gt_speaker_consumed") is False
        and meta.get("gt_addressee_consumed") is False
        and isinstance(document.get("raw"), list)
        and isinstance(document.get("facts"), list)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--provider", choices=("aigc", "velen"), required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--aigc-key-index", type=int)
    args = parser.parse_args()
    if args.provider == "aigc" and args.aigc_key_index is None:
        raise ValueError("AIGC parallel evaluator requires --aigc-key-index")
    if args.provider == "velen" and args.aigc_key_index is not None:
        raise ValueError("Velen evaluator must not receive --aigc-key-index")
    if len(args.cases) != len(set(args.cases)):
        raise ValueError("parallel evaluator received duplicate cases")

    root = args.run_root.resolve(strict=True)
    state = root / "state" / f"iterative3_evaluation_{args.provider}_{args.worker}.json"
    memory_state = root / "state" / f"memory_{args.provider}.json"
    state_write(
        state, status="waiting_for_memory_case", provider=args.provider,
        worker=args.worker, cases=args.cases, aigc_key_index=args.aigc_key_index,
    )

    for position, case_id in enumerate(args.cases, 1):
        memory_path = root / "memory" / f"{case_id}.json"
        while not memory_is_ready(memory_path, case_id):
            memory = json.loads(memory_state.read_text()) if memory_state.exists() else {}
            if memory.get("status") == "failed":
                state_write(state, status="blocked", memory_error=memory.get("error"))
                return 2
            state_write(
                state, status="waiting_for_memory_case", current_case=case_id,
                memory_case=memory.get("current_case"), position=position,
            )
            time.sleep(60)
        state_write(state, status="evaluating", current_case=case_id, position=position)
        command = [
            str(PYTHON), "-B", str(HERE / "evaluate_case_iterative3.py"),
            "--run-root", str(root), "--case", case_id, "--provider", args.provider,
            "--state-suffix", args.worker,
        ]
        if args.aigc_key_index is not None:
            command.extend(["--aigc-key-index", str(args.aigc_key_index)])
        result = subprocess.run(command)
        if result.returncode != 0:
            state_write(state, status="failed", current_case=case_id, exit_code=result.returncode)
            return result.returncode
    state_write(state, status="complete", current_case=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
