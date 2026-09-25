"""Shared, opt-in chronological ordering for unified-memory executors.

The frozen r12 behavior is preserved unless ``MPMEM_TEMPORAL_CONTRACT_V1`` is
explicitly enabled.  An earlier structured-range prototype did not add either
missing clue in the zero-LLM retrieval gate, so the retained candidate is
deliberately sort-only.  Frozen exact-date filtering remains unchanged.
"""
from __future__ import annotations

from datetime import date
import os
import re
from typing import Any, Callable, Mapping, Sequence


TEMPORAL_CONTRACT_ENV = "MPMEM_TEMPORAL_CONTRACT_V1"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_ISO_DATETIME = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})(?:[Tt ].*)?\s*$")


def temporal_contract_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return true only for an explicit candidate opt-in."""

    source = os.environ if environ is None else environ
    return str(source.get(TEMPORAL_CONTRACT_ENV, "")).strip().lower() in _TRUE_VALUES


def _valid_date(value: str) -> str | None:
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return value


def _date_prefix(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = _ISO_DATETIME.fullmatch(value)
    return _valid_date(match.group(1)) if match else None


def apply_frozen_time_range(
    rows: list[dict[str, Any]],
    value: Any,
    *,
    date_getter: Callable[[dict[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Reproduce frozen r12 exact-date filtering byte-for-byte.

    List/dict/range values remain fail-open.  This function is shared to prevent
    executor drift, but is not part of the opt-in sort candidate.
    """

    getter = date_getter or (lambda row: row.get("date"))
    if not isinstance(value, str) or len(value) != 10:
        return rows
    filtered = [row for row in rows if str(getter(row) or "") == value]
    return filtered or rows


def _natural_key(value: Any) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", str(value or ""))
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in parts if part
    )


def _ref_coordinates(row: dict[str, Any]) -> tuple[str, int | None]:
    refs: Sequence[Any] = row.get("refer_ids") or row.get("source_turn_ids") or ()
    ref = str(refs[0]) if refs else str(row.get("node_id") or row.get("mem_id") or "")
    match = re.search(r"(?:^|:)(session\d+):(\d+)$", ref, re.IGNORECASE)
    if match:
        return match.group(1), int(match.group(2))
    match = re.search(r"(?:^|:)(D\d+):(\d+)$", ref, re.IGNORECASE)
    if match:
        return match.group(1), int(match.group(2))
    match = re.search(r"(?:^|:)(S\d+)_T(\d+)$", ref, re.IGNORECASE)
    if match:
        return match.group(1), int(match.group(2))
    return "", None


def _is_answer_visible_raw(row: dict[str, Any]) -> bool:
    layer = str(row.get("memory_layer") or row.get("layer") or "").lower()
    if layer:
        return layer == "raw"
    identity = str(row.get("node_id") or row.get("mem_id") or "").lower()
    return not identity.startswith(("collection:", "fact:", "profile:", "hier:"))


def stable_sort_answer_visible_raw(
    rows: list[dict[str, Any]],
    *,
    sort_by: Any,
    enabled: bool,
    date_getter: Callable[[dict[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Chronologically sort only answer-visible raw slots.

    Upper nodes keep their slots.  Raw membership, row objects, provenance, and
    image fields are preserved byte-for-byte; only raw order can change.
    """

    if not enabled or str(sort_by).lower() != "time":
        return rows
    getter = date_getter or (lambda row: row.get("date"))
    raw_positions = [i for i, row in enumerate(rows) if _is_answer_visible_raw(row)]
    if len(raw_positions) < 2:
        return rows

    indexed = [(position, rows[position]) for position in raw_positions]

    def key(item: tuple[int, dict[str, Any]]):
        original_position, row = item
        normalized_date = _date_prefix(getter(row))
        session, inferred_ordinal = _ref_coordinates(row)
        session = str(row.get("session_id") or row.get("session") or session)
        ordinal = row.get("ordinal", inferred_ordinal)
        try:
            ordinal_key = int(ordinal)
        except (TypeError, ValueError):
            ordinal_key = 2**31 - 1
        return (
            0 if normalized_date else 1,
            normalized_date or "9999-12-31",
            _natural_key(session),
            ordinal_key,
            original_position,
        )

    sorted_rows = [row for _, row in sorted(indexed, key=key)]
    if all(rows[position] is row for position, row in zip(raw_positions, sorted_rows)):
        return rows
    result = list(rows)
    for position, row in zip(raw_positions, sorted_rows):
        result[position] = row
    return result
