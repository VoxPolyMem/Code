"""Normalize query variants before cross-channel retrieval fusion.

Several rewrites of one query are evidence variants of the same retrieval
channel, not independent votes.  This module first fuses those variants into
one ranked list, so a layer cannot gain extra weight merely by issuing more
rewrites.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def _identity(row: dict[str, Any]) -> str:
    return str(
        row.get("mem_id")
        or row.get("node_id")
        or row.get("set_label")
        or row.get("text")
        or ""
    )


def normalize_query_variants(
    channels: list[tuple[str, float, list[dict[str, Any]]]],
) -> tuple[list[tuple[str, float, list[dict[str, Any]]]], dict[str, int]]:
    """Collapse repeated ``(layer, retriever)`` lists with an inner RRF.

    The returned channel keeps exactly one outer-fusion vote.  Distinct dense,
    BM25, image, and caption channels remain independent.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for name, weight, ranked in channels:
        entry = grouped.setdefault(name, {"weight": weight, "ranked": []})
        entry["weight"] = max(float(entry["weight"]), float(weight))
        entry["ranked"].append(ranked)

    normalized = []
    collapsed = 0
    for name, entry in grouped.items():
        variants = entry["ranked"]
        if len(variants) == 1:
            normalized.append((name, entry["weight"], variants[0]))
            continue
        collapsed += len(variants) - 1
        scores: dict[str, float] = defaultdict(float)
        rows: dict[str, dict[str, Any]] = {}
        sources: dict[str, list[str]] = defaultdict(list)
        for variant_index, ranked in enumerate(variants):
            source = f"{name}.query{variant_index}"
            for rank, row in enumerate(ranked, 1):
                key = _identity(row)
                if not key:
                    continue
                rows.setdefault(key, row)
                scores[key] += 1.0 / (60 + rank)
                sources[key].append(source)
        fused = []
        for key in sorted(scores, key=scores.get, reverse=True):
            row = dict(rows[key])
            row["score"] = scores[key]
            row["query_variant_channels"] = sources[key]
            fused.append(row)
        normalized.append((name, entry["weight"], fused))
    return normalized, {
        "input_channels": len(channels),
        "normalized_channels": len(normalized),
        "collapsed_query_votes": collapsed,
    }
