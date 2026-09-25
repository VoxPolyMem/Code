#!/usr/bin/env python3
"""Wait for one memory lane, then serially evaluate its completed cases."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--provider", choices=("aigc", "velen"), required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    state = root / "state" / f"evaluation_{args.provider}.json"
    memory_state = root / "state" / f"memory_{args.provider}.json"
    state_write(state, status="waiting_for_memory_lane", provider=args.provider, cases=args.cases)
    while True:
        if not memory_state.exists():
            time.sleep(60)
            continue
        memory = json.loads(memory_state.read_text())
        if memory.get("status") == "complete":
            break
        if memory.get("status") == "failed":
            state_write(state, status="blocked", memory_error=memory.get("error"))
            return 2
        state_write(state, status="waiting_for_memory_lane", memory_case=memory.get("current_case"))
        time.sleep(120)
    for position, case_id in enumerate(args.cases, 1):
        state_write(state, status="evaluating", current_case=case_id, position=position)
        result = subprocess.run([
            str(PYTHON), "-B", str(HERE / "evaluate_case.py"),
            "--run-root", str(root), "--case", case_id, "--provider", args.provider,
        ])
        if result.returncode != 0:
            state_write(state, status="failed", current_case=case_id, exit_code=result.returncode)
            return result.returncode
    state_write(state, status="complete", current_case=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
