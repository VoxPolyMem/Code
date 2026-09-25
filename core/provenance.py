"""Provenance-aware context packing for hierarchical memory retrieval."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any


def _refs(row: dict[str, Any]) -> set[str]:
    return {str(value) for value in row.get("refer_ids", []) if value}


def _is_upper(row: dict[str, Any]) -> bool:
    identity = str(row.get("mem_id") or row.get("node_id") or "")
    return row.get("memory_layer") == "collection" or identity.startswith(
        ("collection:", "hier:retrieved_collection")
    )


def _effective_support(counts: Counter[str]) -> tuple[float, float]:
    """Return Shannon entropy and its effective number of evidence sources."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0, 0.0
    entropy = -sum(
        (count / total) * math.log(count / total)
        for count in counts.values() if count > 0
    )
    return entropy, math.exp(entropy)


def select_context(
    candidates: list[dict[str, Any]],
    top_k: int = 30,
    support_quota: int = 2,
    expose_upper: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pack relevant, provenance-diverse context with bounded verification redundancy.

    Ranking is deterministic and never consumes gold evidence. If an upper node
    is selected, at most ``support_quota`` overlapping bottom rows are reserved
    so the answer model can verify the summary against source evidence. Remaining
    slots maximize a mixture of original relevance rank, new refer-id coverage,
    and low provenance overlap.
    """
    if top_k <= 0 or not candidates:
        return [], {
            "candidate_count": len(candidates), "selected_count": 0,
            "unique_refer_ids": 0, "refer_mentions": 0,
            "redundancy_ratio": 0.0, "support_rows": 0,
        }

    unique_candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(candidates):
        identity = str(row.get("mem_id") or row.get("node_id") or f"row:{index}")
        if identity in seen_ids:
            continue
        seen_ids.add(identity)
        unique_candidates.append(row)

    # When an upper node is retrieval-only, select one extra item internally so
    # removing that scaffold still leaves up to ``top_k`` answer-visible rows.
    selection_limit = top_k if expose_upper else top_k + 1
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    covered: set[str] = set()
    coverage_counts: Counter[str] = Counter()
    support_rows = 0

    upper_index = next(
        (index for index, row in enumerate(unique_candidates) if _is_upper(row)), None
    )
    if upper_index is not None:
        upper = unique_candidates[upper_index]
        selected.append(upper)
        selected_indices.add(upper_index)
        upper_refs = _refs(upper)
        covered.update(upper_refs)
        coverage_counts.update(upper_refs)
        for index, row in enumerate(unique_candidates):
            if len(selected) >= selection_limit or support_rows >= support_quota:
                break
            if index in selected_indices:
                continue
            refs = _refs(row)
            if refs and refs & upper_refs:
                selected.append(row)
                selected_indices.add(index)
                covered.update(refs)
                coverage_counts.update(refs)
                support_rows += 1

    while (
        len(selected) < selection_limit
        and len(selected_indices) < len(unique_candidates)
    ):
        best_index = None
        best_gain = float("-inf")
        for index, row in enumerate(unique_candidates):
            if index in selected_indices:
                continue
            refs = _refs(row)
            rank_relevance = 1.0 / (1.0 + index)
            novelty = len(refs - covered) / max(len(refs), 1) if refs else 0.0
            _, effective_before = _effective_support(coverage_counts)
            proposed_counts = coverage_counts.copy()
            proposed_counts.update(refs)
            _, effective_after = _effective_support(proposed_counts)
            entropy_gain = (effective_after - effective_before) / max(len(refs), 1)
            entropy_gain = max(-1.0, min(1.0, entropy_gain))
            max_overlap = 0.0
            if refs:
                for chosen in selected:
                    chosen_refs = _refs(chosen)
                    union = refs | chosen_refs
                    if union:
                        max_overlap = max(max_overlap, len(refs & chosen_refs) / len(union))
            gain = (
                0.50 * rank_relevance
                + 0.25 * novelty
                + 0.25 * entropy_gain
                - 0.10 * max_overlap
            )
            if gain > best_gain:
                best_gain, best_index = gain, index
        if best_index is None:
            break
        row = unique_candidates[best_index]
        selected.append(row)
        selected_indices.add(best_index)
        covered.update(_refs(row))
        coverage_counts.update(_refs(row))

    upper_scaffold_hidden = 0
    if not expose_upper:
        answer_visible = [row for row in selected if not _is_upper(row)]
        upper_scaffold_hidden = len(selected) - len(answer_visible)
        selected = answer_visible[:top_k]

    final_counts: Counter[str] = Counter()
    for row in selected:
        final_counts.update(_refs(row))
    final_refs = set(final_counts)
    refer_mentions = sum(final_counts.values())
    entropy, effective_support = _effective_support(final_counts)
    diagnostics = {
        "candidate_count": len(unique_candidates),
        "selected_count": len(selected),
        "unique_refer_ids": len(final_refs),
        "refer_mentions": refer_mentions,
        "redundancy_ratio": (
            1.0 - len(final_refs) / refer_mentions if refer_mentions else 0.0
        ),
        "support_rows": support_rows,
        "support_quota": support_quota,
        "provenance_entropy": entropy,
        "effective_refer_support": effective_support,
        "upper_scaffold_hidden": upper_scaffold_hidden,
        "uses_gold": False,
    }
    return selected, diagnostics
