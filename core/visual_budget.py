"""Choose a small, relevant image payload from the retrieved top-k context."""
from __future__ import annotations

from typing import Any, Callable


def prioritize_visual_rows(
    visual_ranked: list[dict[str, Any]],
    context: list[dict[str, Any]],
    identity: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    """Order context images by the dedicated visual channel, then context rank.

    Only rows already admitted to the final top-k context are eligible.  This
    changes the bounded multimodal payload, not retrieval coverage or the
    benchmark's top-k contract.
    """
    context_ids = {identity(row) for row in context}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in [*visual_ranked, *context]:
        row_id = identity(row)
        if not row_id or row_id in seen or row_id not in context_ids:
            continue
        if not row.get("image_paths"):
            continue
        selected.append(row)
        seen.add(row_id)
    return selected
