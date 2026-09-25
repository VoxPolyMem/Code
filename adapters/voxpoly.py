"""Normalize all VoxPolyBench case generations without using QA labels in memory.

The normalized bottom-level ID is the only provenance namespace used by raw,
fact, and collection nodes. Legacy cases use ``S1:12``; relational cases keep
their explicit ``S1_T012`` IDs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def resolve_case(cases_root: str | Path, case_name: str) -> Path:
    path = Path(cases_root) / case_name
    if path.is_dir():
        path = path / "full_case.json"
    elif path.suffix != ".json":
        path = path / "full_case.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def load_case(cases_root: str | Path, case_name: str) -> tuple[Path, dict[str, Any]]:
    path = resolve_case(cases_root, case_name)
    return path, json.loads(path.read_text(encoding="utf-8"))


def _session_date(data: dict[str, Any], session: dict[str, Any], sid: str) -> str:
    value = data.get("session_dates", {}).get(sid)
    if isinstance(value, dict):
        return str(value.get("date") or "")
    if isinstance(value, str):
        return value
    return str(session.get("date") or "")


def iter_bottom_nodes(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield verbatim turns with stable, nonempty self provenance."""
    dialogues = data.get("dialogues") or {}
    if not isinstance(dialogues, dict):
        raise TypeError("VoxPolyBench dialogues must be a session-keyed object")
    sessions = data.get("sessions") or [
        {"session_id": sid} for sid in dialogues
    ]
    seen = set()
    for session in sessions:
        sid = str(session.get("session_id") or "")
        if not sid:
            continue
        for ordinal, turn in enumerate(dialogues.get(sid, []), 1):
            # Legacy VoxPolyBench evidence uses zero-based ``S8:0`` offsets.
            # New relational generations expose their own ``S8_T001`` IDs.
            bottom_id = str(turn.get("turn_id") or f"{sid}:{ordinal - 1}")
            if bottom_id in seen:
                raise ValueError(f"duplicate bottom evidence ID: {bottom_id}")
            seen.add(bottom_id)
            yield {
                "node_id": f"raw:{bottom_id}",
                "layer": "raw",
                "text": str(turn.get("text") or ""),
                "speaker": str(
                    turn.get("speaker_name")
                    or turn.get("speaker_id")
                    or turn.get("speaker_role")
                    or "unknown"
                ),
                "session_id": sid,
                "ordinal": int(turn.get("ordinal") or ordinal),
                "date": _session_date(data, session, sid),
                "refer_ids": [bottom_id],
            }


def qa_pairs(data: dict[str, Any]) -> list[dict[str, Any]]:
    qa = data.get("qa")
    if isinstance(qa, dict):
        rows = qa.get("qa_pairs", [])
    elif isinstance(qa, list):
        rows = qa
    else:
        rows = data.get("qa_pairs", [])
    if not isinstance(rows, list):
        raise TypeError("VoxPolyBench QA collection must be a list")
    return rows


def gold_evidence(qa: dict[str, Any]) -> list[str]:
    for key in ("evidence_ids", "evidence", "gold_evidence_ids", "gold_evidence"):
        value = qa.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def integrity_report(data: dict[str, Any]) -> dict[str, Any]:
    nodes = list(iter_bottom_nodes(data))
    bottom_ids = {node["refer_ids"][0] for node in nodes}
    qas = qa_pairs(data)
    evidence = [gold_evidence(qa) for qa in qas]
    referenced = {item for group in evidence for item in group}
    return {
        "turns": len(nodes),
        "qa": len(qas),
        "qa_with_evidence": sum(bool(group) for group in evidence),
        "missing_evidence_ids": sorted(referenced - bottom_ids),
        "empty_raw_refer_ids": sum(not node["refer_ids"] for node in nodes),
    }
