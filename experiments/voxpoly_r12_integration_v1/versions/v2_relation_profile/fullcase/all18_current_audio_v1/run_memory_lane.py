#!/usr/bin/env python3
"""One bounded, resumable provider lane for all18 contextual-memory builds."""

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
PACKAGE_ROOT = HERE.parents[5]
AGENT_MEMORY = Path(os.environ.get("AGENT_MEMORY_ROOT", PACKAGE_ROOT))
WORKSPACE = PACKAGE_ROOT
EXPERIMENT = WORKSPACE / "experiments/voxpoly_r12_integration_v1"
VOXPOLY = Path(os.environ.get(
    "VOXPOLYBENCH_HOME", AGENT_MEMORY / "audio_mem_bench/VoxPolyBench"
))
PYTHON = Path(os.environ.get("PYTHON_BIN", sys.executable))


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_state(path: Path, **fields: Any) -> None:
    prior = json.loads(path.read_text()) if path.exists() else {}
    prior.update(fields)
    prior["updated_at"] = now()
    atomic_json(path, prior)


def provider_env(provider: str, *, aigc_key_index: int | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["EVAL_PIN_MODEL"] = "gpt-4.1-mini"
    if provider == "aigc":
        # Do not silently spend Velen if the AIGC lane is rate limited.
        environment["NO_VELEN_FALLBACK"] = "1"
        environment.pop("FORCE_VELEN_MODEL", None)
        if aigc_key_index is None:
            environment.pop("AIGC_EVAL_KEY_INDEX", None)
        elif aigc_key_index >= 0:
            # Each concurrent evaluator receives a disjoint configured AIGC
            # key.  The value is an index only; no credential is persisted.
            environment["AIGC_EVAL_KEY_INDEX"] = str(aigc_key_index)
        else:
            raise ValueError("aigc_key_index must be non-negative")
    elif provider == "velen":
        # Separate serial lane with explicit accounting in llm_client logs.
        environment["FORCE_VELEN_MODEL"] = "gpt-4.1-mini"
        environment.pop("NO_VELEN_FALLBACK", None)
        environment.pop("AIGC_EVAL_KEY_INDEX", None)
    else:
        raise ValueError(f"unsupported provider: {provider}")
    return environment


def run_case(row: dict[str, Any], run_root: Path, provider: str, state_path: Path) -> None:
    case_id = str(row["case_id"])
    memory = run_root / "memory" / f"{case_id}.json"
    checkpoints = run_root / "checkpoints" / case_id
    log = run_root / "logs" / f"{provider}_{case_id}.log"
    registry = VOXPOLY / "logs/speaker/online_all_cases_v1" / case_id / "online_registry_state.npz"
    preflight = Path(str(row["bundle_audit"])).with_name(f"{case_id}.preflight.json")
    if not preflight.exists():
        raise FileNotFoundError(f"missing zero-API preflight: {preflight}")
    preflight_doc = json.loads(preflight.read_text())
    if preflight_doc.get("status") != "ready_for_contextual_memory_build":
        raise RuntimeError(f"preflight did not pass: {case_id}")
    if memory.exists():
        try:
            existing = json.loads(memory.read_text())
            if existing.get("case_id") == case_id and existing.get("meta", {}).get("qa_consumed") is False:
                write_state(state_path, current_case=case_id, case_status="already_complete")
                return
        except (OSError, json.JSONDecodeError):
            pass
        raise RuntimeError(f"existing memory output is incomplete or incompatible: {memory}")

    command = [
        str(PYTHON), "-B", str(EXPERIMENT / "build_memory.py"),
        "--enable-experimental-audio",
        "--case-view-dir", str(row["case_view_dir"]),
        "--registry-state", str(registry),
        "--out", str(memory),
        "--checkpoints", str(checkpoints),
        "--model", "gpt-4.1-mini",
        "--resume-output",
        "--max-attempts", "8",
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 9):
        write_state(
            state_path, current_case=case_id, case_status="building_memory",
            process_attempt=attempt, memory_path=str(memory), log_path=str(log),
        )
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] start {case_id} attempt={attempt}/8 provider={provider}\n")
            handle.flush()
            completed = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT,
                env=provider_env(provider), timeout=4 * 3600,
            )
            handle.write(f"[{now()}] exit={completed.returncode} {case_id} attempt={attempt}/8\n")
        if completed.returncode == 0 and memory.exists():
            built = json.loads(memory.read_text())
            meta = built.get("meta") or {}
            if built.get("case_id") == case_id and meta.get("qa_consumed") is False and meta.get("gt_addressee_consumed") is False:
                write_state(
                    state_path, current_case=case_id, case_status="memory_complete",
                    process_attempt=attempt, facts=len(built.get("facts") or []),
                    raw_turns=len(built.get("raw") or []),
                )
                return
        if attempt < 8:
            write_state(state_path, current_case=case_id, case_status="retry_backoff", process_attempt=attempt)
            time.sleep(300)
    raise RuntimeError(f"memory build exhausted bounded retry budget: {case_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--provider", choices=("aigc", "velen"), required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve(strict=True)
    manifest = json.loads((run_root / "all18_inputs.json").read_text())
    rows = {str(row["case_id"]): dict(row) for row in manifest.get("cases") or []}
    if len(args.cases) != len(set(args.cases)) or any(case not in rows for case in args.cases):
        raise ValueError("lane contains duplicate or unknown case ID")
    state = run_root / "state" / f"memory_{args.provider}.json"
    write_state(
        state, status="running", provider=args.provider, cases=args.cases,
        model="gpt-4.1-mini", top_k=30, retry_policy="8 process attempts, 300-second backoff",
    )
    try:
        for case in args.cases:
            run_case(rows[case], run_root, args.provider, state)
    except BaseException as exc:
        write_state(state, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    write_state(state, status="complete", current_case=None, case_status="all_memory_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
