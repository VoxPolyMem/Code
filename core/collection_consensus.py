"""Cross-layer consensus for selecting a grounded collection node."""
from __future__ import annotations

from typing import Any


def _refs(row: dict[str, Any]) -> set[str]:
    return {str(value) for value in row.get("refer_ids", []) if value}


def rerank_collections(
    collections: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    support_weight: float = 0.25,
    max_support_hits: int = 8,
) -> list[dict[str, Any]]:
    """Prefer collections independently supported by raw/fact retrieval.

    The collection channel remains primary.  A small reciprocal-rank bonus is
    added from up to ``max_support_hits`` lower-layer rows whose provenance
    overlaps the collection.  This is benchmark-agnostic and consumes neither
    answer text nor gold evidence.
    """
    reranked = []
    for collection in collections:
        collection_refs = _refs(collection)
        support_ranks = []
        if collection_refs:
            for rank, row in enumerate(evidence, 1):
                if collection_refs & _refs(row):
                    support_ranks.append(rank)
                    if len(support_ranks) >= max_support_hits:
                        break
        support = sum(1.0 / (60 + rank) for rank in support_ranks)
        row = dict(collection)
        row["collection_base_score"] = float(collection.get("score", 0.0))
        row["cross_layer_support"] = support
        row["cross_layer_support_ranks"] = support_ranks
        row["score"] = row["collection_base_score"] + support_weight * support
        reranked.append(row)
    return sorted(reranked, key=lambda row: row["score"], reverse=True)
