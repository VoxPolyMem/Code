"""Project upper memories to grounded raw evidence with local continuity."""
from __future__ import annotations

from typing import Any


def project_grounding(
    upper: dict[str, Any],
    raw_by_ref: dict[str, dict[str, Any]],
    raw_by_image: dict[str, list[dict[str, Any]]],
    raw_by_session_ordinal: dict[tuple[str, int], dict[str, Any]],
    neighbor_radius: int = 1,
) -> list[tuple[dict[str, Any], float, str]]:
    """Return direct, media-linked, and bounded adjacent raw support.

    The expansion is driven only by stored provenance.  It never consumes QA
    types, answers, or gold evidence.
    """
    selected: dict[str, tuple[dict[str, Any], float, str]] = {}

    def keep(row: dict[str, Any] | None, weight: float, relation: str) -> None:
        if not row:
            return
        identity = str(row.get("node_id") or "")
        current = selected.get(identity)
        if identity and (current is None or weight > current[1]):
            selected[identity] = (row, weight, relation)

    direct = []
    for ref in upper.get("refer_ids", []):
        row = raw_by_ref.get(str(ref))
        if row:
            direct.append(row)
            keep(row, 1.0, "direct")

    for image_id in upper.get("images", []) or upper.get("source_image_ids", []):
        for row in raw_by_image.get(str(image_id), []):
            keep(row, 1.0, "media")

    if neighbor_radius > 0:
        for row in direct:
            session = str(row.get("session_id") or "")
            ordinal = row.get("ordinal")
            if not session or not isinstance(ordinal, int):
                continue
            for delta in range(-neighbor_radius, neighbor_radius + 1):
                if delta:
                    keep(
                        raw_by_session_ordinal.get((session, ordinal + delta)),
                        0.35,
                        "adjacent",
                    )
    return list(selected.values())
