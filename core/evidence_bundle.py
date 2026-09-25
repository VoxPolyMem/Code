"""Answer-only organization of provenance-linked raw evidence.

The retriever still chooses exactly the same top-k raw memories. This module
only places raw turns supported by the same contextual fact next to each other
and orders those turns chronologically. It never creates a summary and never
uses benchmark labels, answers, or gold evidence.
"""
from __future__ import annotations

from datetime import date
import os
import re
from typing import Any, Mapping


EVIDENCE_BUNDLE_ENV = "MPMEM_EVIDENCE_BUNDLE_V1"
EVIDENCE_BUNDLE_SOFT_GATE_ENV = "MPMEM_EVIDENCE_BUNDLE_SOFT_GATE_V1"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_ISO_DATE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})(?:[Tt ].*)?\s*$")


def evidence_bundle_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if environ is None else environ
    return str(source.get(EVIDENCE_BUNDLE_ENV, "")).strip().lower() in _TRUE_VALUES


def evidence_bundle_soft_gate_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if environ is None else environ
    return str(source.get(EVIDENCE_BUNDLE_SOFT_GATE_ENV, "")).strip().lower() in _TRUE_VALUES


def evidence_bundle_metadata_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    return evidence_bundle_enabled(environ) or evidence_bundle_soft_gate_enabled(environ)


def evidence_bundle_active(
    plan: dict[str, Any] | None,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """Decide presentation use from existing model route decisions only."""
    if evidence_bundle_enabled(environ):
        return True, "global_v1"
    if not evidence_bundle_soft_gate_enabled(environ):
        return False, "disabled"
    plan = plan or {}
    has_decomposition = len(plan.get("sub_questions", []) or []) >= 2
    if plan.get("model_needs_temporal_reasoning"):
        return False, "global_temporal_order_preserved"
    cross_turn = bool(
        plan.get("model_needs_event_scope")
    )
    if has_decomposition and cross_turn:
        return True, "model_cross_turn_decomposition"
    return False, "model_gate_rejected"


def make_evidence_group(
    group_id: str,
    refer_ids: list[Any],
    *,
    source_rank: int,
) -> dict[str, Any] | None:
    """Create a bounded fact-provenance group or reject an unsafe broad group."""
    refs = list(dict.fromkeys(str(value) for value in refer_ids if value))
    if not 2 <= len(refs) <= 6:
        return None
    return {
        "group_id": str(group_id),
        "refer_ids": refs,
        "source_rank": int(source_rank),
    }


def merge_evidence_groups(
    left: list[dict[str, Any]] | None,
    right: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in list(left or []) + list(right or []):
        identity = str(group.get("group_id") or "")
        if not identity or identity in seen:
            continue
        seen.add(identity)
        merged.append(dict(group))
    return merged


def _identity(row: dict[str, Any], index: int) -> str:
    return str(row.get("mem_id") or row.get("node_id") or f"row:{index}")


def _date_key(row: dict[str, Any]) -> tuple[Any, ...]:
    value = row.get("date") or row.get("timestamp") or ""
    match = _ISO_DATE.fullmatch(str(value))
    normalized = match.group(1) if match else ""
    try:
        if normalized:
            date.fromisoformat(normalized)
    except ValueError:
        normalized = ""
    ordinal = row.get("ordinal", row.get("turn_seq"))
    try:
        ordinal_key = int(ordinal)
    except (TypeError, ValueError):
        ordinal_key = 2**31 - 1
    return (
        0 if normalized else 1,
        normalized or "9999-12-31",
        str(row.get("session_id") or row.get("session") or ""),
        ordinal_key,
    )


def organize_evidence_bundles(
    rows: list[dict[str, Any]],
    *,
    enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return an answer-only view with grounded multi-turn groups made adjacent.

    Membership is invariant: every input row appears exactly once in the output.
    Candidate groups come only from stored fact provenance attached by retrieval.
    Overlapping groups are resolved greedily so no raw turn is duplicated.
    """
    if not enabled or len(rows) < 2:
        return list(rows), {
            "enabled": bool(enabled),
            "candidate_groups": 0,
            "selected_groups": 0,
            "bundled_rows": 0,
            "membership_preserved": True,
        }

    members_by_group: dict[str, set[int]] = {}
    rank_by_group: dict[str, int] = {}
    for index, row in enumerate(rows):
        for group in row.get("evidence_groups", []) or []:
            group_id = str(group.get("group_id") or "")
            if not group_id:
                continue
            members_by_group.setdefault(group_id, set()).add(index)
            rank_by_group[group_id] = min(
                rank_by_group.get(group_id, 2**31 - 1),
                int(group.get("source_rank", 2**31 - 1)),
            )

    candidates = [
        (group_id, indices)
        for group_id, indices in members_by_group.items()
        if 2 <= len(indices) <= 6
    ]
    candidates.sort(key=lambda item: (
        min(item[1]),
        rank_by_group.get(item[0], 2**31 - 1),
        -len(item[1]),
        item[0],
    ))

    selected: list[tuple[str, list[int]]] = []
    claimed: set[int] = set()
    for group_id, indices in candidates:
        available = sorted(indices - claimed)
        if len(available) < 2:
            continue
        selected.append((group_id, available))
        claimed.update(available)

    group_for_index = {
        index: (group_number, member_indices)
        for group_number, (_, member_indices) in enumerate(selected, 1)
        for index in member_indices
    }
    emitted_groups: set[int] = set()
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        assignment = group_for_index.get(index)
        if assignment is None:
            output.append(row)
            continue
        group_number, member_indices = assignment
        if group_number in emitted_groups:
            continue
        emitted_groups.add(group_number)
        ordered_members = sorted(
            member_indices,
            key=lambda member_index: (_date_key(rows[member_index]), member_index),
        )
        for offset, member_index in enumerate(ordered_members):
            copied = dict(rows[member_index])
            copied["evidence_bundle_id"] = group_number
            if offset == 0:
                copied["evidence_bundle_header"] = (
                    f"[Evidence bundle {group_number}: related source turns; "
                    "shown chronologically]"
                )
            output.append(copied)

    before = sorted(_identity(row, index) for index, row in enumerate(rows))
    after = sorted(_identity(row, index) for index, row in enumerate(output))
    preserved = before == after and len(output) == len(rows)
    if not preserved:
        raise AssertionError("evidence bundling changed top-k membership")
    return output, {
        "enabled": True,
        "candidate_groups": len(candidates),
        "selected_groups": len(selected),
        "bundled_rows": len(claimed),
        "membership_preserved": preserved,
    }


def answer_view(
    rows: list[dict[str, Any]],
    *,
    plan: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the opt-in prompt view without mutating retrieval rows."""
    active, reason = evidence_bundle_active(plan)
    organized, diagnostics = organize_evidence_bundles(
        rows, enabled=active
    )
    diagnostics["activation_reason"] = reason
    answer_rows: list[dict[str, Any]] = []
    for row in organized:
        copied = dict(row)
        header = copied.get("evidence_bundle_header")
        if header:
            copied["text"] = f"{header}\n{copied.get('text', '')}"
        answer_rows.append(copied)
    return answer_rows, diagnostics
