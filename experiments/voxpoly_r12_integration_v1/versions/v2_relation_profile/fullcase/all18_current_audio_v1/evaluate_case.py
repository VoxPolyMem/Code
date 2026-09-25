#!/usr/bin/env python3
"""Run one completed all18 memory through the current unified Audio evaluation.

This is intentionally an isolated wrapper.  It uses the original query,
Top-30, waveform-derived asker identity, shared planner scope selection, and
the generic known-asker answer hint selected by the four-case audit.
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

from run_memory_lane import AGENT_MEMORY, EXPERIMENT, PYTHON, VOXPOLY, provider_env


HERE = Path(__file__).resolve().parent


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


def update_state(path: Path, **fields: Any) -> None:
    prior = json.loads(path.read_text()) if path.exists() else {}
    prior.update(fields)
    prior["updated_at"] = now()
    atomic_json(path, prior)


def run_bounded(
    command: list[str], *, environment: dict[str, str], log: Path, phase: str,
    state: Path, case_id: str,
) -> None:
    for attempt in range(1, 9):
        update_state(state, current_case=case_id, phase=phase, process_attempt=attempt, status="running")
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] {phase} attempt={attempt}/8\n")
            handle.flush()
            finished = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT,
                env=environment, timeout=4 * 3600,
            )
            handle.write(f"[{now()}] {phase} exit={finished.returncode}\n")
        if finished.returncode == 0:
            return
        if attempt < 8:
            update_state(state, current_case=case_id, phase=phase, process_attempt=attempt, status="retry_backoff")
            time.sleep(300)
    raise RuntimeError(f"{case_id} phase exhausted finite retries: {phase}")


def required_complete(path: Path, label: str) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("status") != "complete":
        raise RuntimeError(f"{label} is not complete: {path}")
    return document


def required_memory(path: Path, case_id: str) -> dict[str, Any]:
    """Validate the builder's explicit memory contract.

    Contextual memories are immutable artifacts rather than task-result
    envelopes, so they deliberately have no ``status: complete`` field.  The
    original lane already validates this metadata before declaring a build
    complete; evaluation repeats the same contract here.
    """

    document = json.loads(path.read_text())
    meta = document.get("meta") or {}
    if (
        document.get("schema_version") != "voxpoly-r12-contextual-facts-v1"
        or document.get("case_id") != case_id
        or meta.get("qa_consumed") is not False
        or meta.get("gold_evidence_consumed") is not False
        or meta.get("gt_speaker_consumed") is not False
        or meta.get("gt_addressee_consumed") is not False
        or not isinstance(document.get("raw"), list)
        or not isinstance(document.get("facts"), list)
    ):
        raise RuntimeError(f"memory is incomplete or has an incompatible contract: {path}")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--provider", choices=("aigc", "velen"), required=True)
    parser.add_argument("--aigc-key-index", type=int)
    parser.add_argument("--state-suffix")
    args = parser.parse_args()
    if args.aigc_key_index is not None and args.provider != "aigc":
        raise ValueError("--aigc-key-index is valid only for the AIGC provider")
    run_root = args.run_root.resolve(strict=True)
    rows = {
        str(row["case_id"]): dict(row)
        for row in json.loads((run_root / "all18_inputs.json").read_text()).get("cases") or []
    }
    row = rows.get(args.case)
    if row is None:
        raise ValueError(f"unknown case: {args.case}")
    case_id = args.case
    memory = run_root / "memory" / f"{case_id}.json"
    memory_doc = required_memory(memory, case_id)
    if memory_doc.get("case_id") != case_id or memory_doc.get("meta", {}).get("qa_consumed") is not False:
        raise RuntimeError("memory contract is incompatible")

    suffix = f"_{args.state_suffix}" if args.state_suffix else ""
    state = run_root / "state" / f"evaluation_{args.provider}{suffix}.json"
    log = run_root / "logs" / f"{args.provider}_{case_id}.evaluation.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = provider_env(args.provider, aigc_key_index=args.aigc_key_index)
    identity = run_root / "query_identity" / f"{case_id}.json"
    if not identity.exists():
        run_bounded([
            str(PYTHON), "-B", str(HERE / "build_audio_sidecar_from_view.py"),
            "--manifest", str(row["query_audio_manifest"]), "--output", str(identity),
        ], environment=environment, log=log, phase="query_waveform_identity", state=state, case_id=case_id)
    identity_doc = json.loads(identity.read_text())
    if identity_doc.get("external_llm_calls") != 0:
        raise RuntimeError("query identity sidecar used an LLM")

    questions = json.loads(Path(str(row["question_manifest"])).read_text())
    qa_ids = [str(item.get("qa_id") or "") for item in questions.get("rows") or []]
    if len(qa_ids) != int(row["qa_count"]) or not all(qa_ids) or len(qa_ids) != len(set(qa_ids)):
        raise RuntimeError("question manifest is incomplete or duplicated")
    routes = run_root / "routes" / case_id
    saved_r12 = routes / "saved_r12.json"
    config = run_root / "configs" / f"{case_id}.json"
    if not config.exists():
        atomic_json(config, {
            "schema_version": "voxpoly-audio-v2-fullcase-config.v1",
            "top_k": 30,
            "embedding_server": "http://127.0.0.1:9981",
            "query_identity_sidecar": f"query_identity/{case_id}.json",
            "cases": [{
                "case_id": case_id,
                "memory": str(memory),
                "saved_r12": str(saved_r12),
                "canonical_case": str(row["canonical_case"]),
                "panel_qa_ids": qa_ids,
                "attribution_audit_qa_ids": [item for item in qa_ids if item.startswith("ATTRIBUTION_")],
                "regression_audit_qa_ids": [],
            }],
        })
    if not saved_r12.exists():
        run_bounded([
            str(PYTHON), "-B", str(HERE / "freeze_r12_fullcase.py"),
            "--enable-paid-routing", "--case-id", case_id,
            "--question-manifest", str(row["question_manifest"]),
            "--memory", str(memory), "--out-dir", str(routes),
            "--max-attempts", "8", "--retry-delay-seconds", "300",
        ], environment=environment, log=log, phase="qa_blind_route_freeze", state=state, case_id=case_id)
    saved = required_complete(saved_r12, "saved route")
    if int(saved.get("summary", {}).get("qa_count", -1)) != len(qa_ids):
        raise RuntimeError("saved route QA count differs from input manifest")

    base_out = run_root / "base_v2" / case_id
    base_result = base_out / "matched_pairs.json"
    if not base_result.exists():
        run_bounded([
            str(PYTHON), "-B", str(HERE / "evaluate_fullcase_base.py"),
            "--enable-paid-evaluation", "--config", str(config), "--out-dir", str(base_out),
            "--max-attempts", "8", "--retry-delay-seconds", "300",
        ], environment=environment, log=log, phase="base_v2_answer_judge", state=state, case_id=case_id)
    required_complete(base_result, "base v2 evaluation")

    out = run_root / "current_unified" / case_id
    result = out / "matched_pairs.json"
    if not result.exists():
        run_bounded([
            str(PYTHON), "-B", str(HERE / "evaluate_fullcase_shared_scope.py"),
            "--enable-paid-evaluation", "--config", str(config),
            "--base-result", str(base_result), "--out-dir", str(out),
            "--answer-query-source", "original",
            "--asker-context-strategy", "speaker_only",
            "--include-known-asker-hint",
            "--max-attempts", "8", "--retry-delay-seconds", "300",
        ], environment=environment, log=log, phase="current_shared_scope_answer_judge", state=state, case_id=case_id)
    final = required_complete(result, "current unified evaluation")
    update_state(
        state, current_case=case_id, phase="complete", status="case_complete",
        qa_count=len(final.get("pairs") or []), overall_llm_score=final.get("summary", {}).get("overall_llm_score"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
