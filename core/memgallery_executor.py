"""Execute a RoutePlan against the frozen Mem-Gallery 4.3 memory/indexes.

All planners share this executor. Upper-layer hits are projected through
``refer_ids`` to the same bottom evidence before answer generation.
"""
from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from core.provenance import select_context
from core.query_fusion import normalize_query_variants
from core.temporal_contract import (
    apply_frozen_time_range,
    stable_sort_answer_visible_raw,
    temporal_contract_enabled,
)
from core.visual_budget import prioritize_visual_rows
from core.collection_consensus import rerank_collections
from core.evidence_bundle import (
    evidence_bundle_metadata_enabled,
    make_evidence_group,
    merge_evidence_groups,
)

import numpy as np


class RouteExecutor:
    def __init__(
        self,
        tools: Any,
        fact_fusion: Any,
        encode_texts: Any,
        top_k: int = 30,
        candidate_k: int = 30,
    ):
        self.tools = tools
        self.ff = fact_fusion
        self.encode_texts = encode_texts
        self.top_k = top_k
        self.candidate_k = candidate_k

    def _upper_fact(self, fact: dict[str, Any], index: int) -> dict[str, Any]:
        row = dict(fact)
        row.update({
            "mem_id": str(fact.get("mem_id") or f"fact:pcp:{index}"),
            "memory_layer": (
                "collection" if fact.get("fact_kind") == "visual_set" else "fact"
            ),
            "refer_ids": list(
                fact.get("refer_ids", fact.get("source_turn_ids", []))
            ),
        })
        return row

    def _with_refs(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        if not out.get("refer_ids"):
            out["refer_ids"] = list(
                self.ff.mem_to_refer_ids.get(out.get("mem_id", ""), [])
            )
        return out

    def _row_date(self, row: dict[str, Any]) -> str | None:
        if row.get("date"):
            return str(row["date"])
        refs = row.get("refer_ids") or row.get("source_turn_ids") or []
        for ref in refs:
            mem_id = self.ff.turn_to_mem.get(str(ref))
            if not mem_id:
                continue
            mem = self.tools.sem_store.get_memory(mem_id)
            if mem and getattr(mem, "date", None):
                return str(mem.date)
            if mem and isinstance(mem, dict) and mem.get("date"):
                return str(mem["date"])
            if mem:
                converted = self.tools._to_retrieved(mem)
                if converted.get("date"):
                    return str(converted["date"])
        return None

    @staticmethod
    def _rrf(channels: list[tuple[str, float, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
        score: dict[str, float] = defaultdict(float)
        rows: dict[str, dict[str, Any]] = {}
        provenance: dict[str, list[str]] = defaultdict(list)
        for channel, weight, ranked in channels:
            for rank, row in enumerate(ranked, 1):
                mem_id = str(row.get("mem_id") or "")
                if not mem_id:
                    continue
                if mem_id not in rows:
                    rows[mem_id] = row
                elif evidence_bundle_metadata_enabled():
                    rows[mem_id] = dict(
                        rows[mem_id],
                        evidence_groups=merge_evidence_groups(
                            rows[mem_id].get("evidence_groups"),
                            row.get("evidence_groups"),
                        ),
                    )
                score[mem_id] += weight / (60 + rank)
                provenance[mem_id].append(channel)
        ordered = []
        for mem_id in sorted(score, key=score.get, reverse=True):
            row = dict(rows[mem_id])
            row["score"] = score[mem_id]
            row["retrieval_channels"] = provenance[mem_id]
            ordered.append(row)
        return ordered

    def _raw(self, query: str, retriever: str, question_image_path: str | None):
        if retriever == "dense":
            return self.tools.search_memory(query, top_k=self.candidate_k)
        if retriever == "bm25":
            return self.tools.search_bm25(query, top_k=self.candidate_k)
        if retriever == "caption":
            return self.tools.search_by_caption(query, top_k=self.candidate_k)
        if retriever == "image":
            cards = self.ff.rank_visual_cards(
                query, query_image_path=question_image_path,
                top_k=self.candidate_k,
            )
            rows = []
            seen = set()
            for card in cards:
                for mem_id in card.get("mem_ids", []):
                    if mem_id in seen:
                        continue
                    mem = self.tools.sem_store.get_memory(mem_id)
                    if mem:
                        row = self.tools._to_retrieved(mem)
                        row["score"] = float(card.get("retrieval_score", 0.0))
                        row["matched_image_id"] = card.get("image_id")
                        rows.append(row)
                        seen.add(mem_id)
            return rows
        return []

    def _fact(self, query: str, retriever: str):
        if retriever not in {"dense", "bm25"}:
            return []
        eligible = [
            i for i, fact in enumerate(self.ff.facts)
            if fact.get("fact_kind") != "visual_set"
        ]
        if retriever == "dense":
            query_emb = self.encode_texts([query])[0]
            values = self.ff.embeddings @ query_emb
        else:
            values = self.ff._bm25(query)
        order = sorted(eligible, key=lambda i: float(values[i]), reverse=True)
        mem_score: dict[str, float] = defaultdict(float)
        matched: dict[str, list[str]] = defaultdict(list)
        evidence_groups_by_mem: dict[str, list[dict[str, Any]]] = defaultdict(list)
        image_to_mem_ids: dict[str, list[str]] = defaultdict(list)
        for card in self.ff.visual_cards:
            for mem_id in card.get("mem_ids", []):
                image_to_mem_ids[str(card.get("image_id") or "")].append(mem_id)
        for rank, index in enumerate(order[: self.candidate_k], 1):
            if retriever == "bm25" and float(values[index]) <= 0:
                continue
            fact = self.ff.facts[index]
            linked: dict[str, float] = {}
            direct_mem_ids = []
            for turn_id in fact.get("source_turn_ids", []):
                mem_id = self.ff.turn_to_mem.get(turn_id)
                if mem_id:
                    linked[mem_id] = max(linked.get(mem_id, 0.0), 1.0)
                    direct_mem_ids.append(mem_id)
                match = re.match(r"^(.*:)(\d+)$", str(turn_id))
                if match:
                    prefix, ordinal = match.group(1), int(match.group(2))
                    for neighbor in (ordinal - 1, ordinal + 1):
                        neighbor_mem = self.ff.turn_to_mem.get(f"{prefix}{neighbor}")
                        if neighbor_mem:
                            linked[neighbor_mem] = max(
                                linked.get(neighbor_mem, 0.0), 0.35
                            )
            for image_id in fact.get("source_image_ids", []):
                for mem_id in image_to_mem_ids.get(str(image_id), []):
                    linked[mem_id] = max(linked.get(mem_id, 0.0), 1.0)
                    direct_mem_ids.append(mem_id)
            group = (
                make_evidence_group(
                    str(fact.get("fact_id") or fact.get("node_id") or f"fact:mg:{index}"),
                    fact.get("refer_ids", fact.get("source_turn_ids", [])),
                    source_rank=rank,
                )
                if evidence_bundle_metadata_enabled() else None
            )
            if group is not None:
                for mem_id in dict.fromkeys(direct_mem_ids):
                    evidence_groups_by_mem[mem_id].append(group)
            for mem_id, weight in linked.items():
                mem_score[mem_id] += weight / (60 + rank)
                matched[mem_id].append(fact["text"])
        rows = []
        for mem_id in sorted(mem_score, key=mem_score.get, reverse=True):
            mem = self.tools.sem_store.get_memory(mem_id)
            if mem:
                row = self.tools._to_retrieved(mem)
                row["score"] = mem_score[mem_id]
                row["matched_facts"] = matched[mem_id][:3]
                if evidence_groups_by_mem[mem_id]:
                    row["evidence_groups"] = merge_evidence_groups(
                        [], evidence_groups_by_mem[mem_id]
                    )
                rows.append(row)
        return rows

    def _collection_facts(
        self, query: str, retriever: str, question_image_path: str | None
    ) -> list[dict[str, Any]]:
        indices = [
            i for i, fact in enumerate(self.ff.facts)
            if fact.get("fact_kind") == "visual_set"
        ]
        if not indices:
            return []
        if retriever == "image" and question_image_path:
            text_ranked = self.ff.rank_visual_sets(query)
            image_ranked = self.ff.rank_visual_sets_by_image(
                question_image_path, text_ranked
            )
            by_key = {
                str(fact.get("set_label") or fact.get("text") or ""): index
                for index, fact in enumerate(self.ff.facts)
            }
            return [
                self._upper_fact(
                    row,
                    by_key.get(str(row.get("set_label") or row.get("text") or ""), -1),
                )
                for row in image_ranked
            ]
        if retriever in {"dense", "caption", "image"}:
            query_emb = self.encode_texts([query])[0]
            values = self.ff.embeddings @ query_emb
        elif retriever == "bm25":
            values = self.ff._bm25(query)
        else:
            return []
        order = sorted(indices, key=lambda i: float(values[i]), reverse=True)
        rows = []
        for index in order:
            if retriever == "bm25" and float(values[index]) <= 0:
                continue
            row = self._upper_fact(self.ff.facts[index], index)
            row["retrieval_score"] = float(values[index])
            rows.append(row)
        return rows

    @staticmethod
    def _rrf_facts(channels: list[tuple[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
        scores: dict[str, float] = defaultdict(float)
        rows: dict[str, dict[str, Any]] = {}
        for name, ranked in channels:
            for rank, row in enumerate(ranked, 1):
                key = str(row.get("set_label") or row.get("text") or "")
                if not key:
                    continue
                rows.setdefault(key, row)
                scores[key] += 1.0 / (60 + rank)
        return [dict(rows[key], route_score=scores[key])
                for key in sorted(scores, key=scores.get, reverse=True)]

    def _expand_collection(self, fact: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        refs = list(dict.fromkeys(
            str(x) for x in fact.get("refer_ids", fact.get("source_turn_ids", []))
        ))
        node = {
            "mem_id": "hier:retrieved_collection",
            "text": "[Retrieved collection node]\n" + str(fact.get("text") or ""),
            "caption": {},
            "image_paths": [],
            "date": None,
            "timestamp": 0,
            "refer_ids": refs,
            "score": 1.0,
            "retrieval_channels": ["collection"],
            "memory_layer": "collection",
        }
        promoted = []
        seen = set()
        for ref in refs:
            mem_id = self.ff.turn_to_mem.get(ref)
            if not mem_id or mem_id in seen:
                continue
            mem = self.tools.sem_store.get_memory(mem_id)
            if mem:
                promoted.append(self._with_refs(self.tools._to_retrieved(mem)))
                seen.add(mem_id)
        return node, promoted

    def execute(
        self,
        plan: dict[str, Any],
        question_image_path: str | None = None,
    ) -> dict[str, Any]:
        channels = []
        collection_channels = []
        visual_ranked = []
        trace = []
        time_range = plan.get("time_range")
        temporal_contract = temporal_contract_enabled()
        for route_index, route in enumerate(plan.get("routes", [])):
            layer = route.get("layer")
            query = str(route.get("query") or plan.get("rewritten_query") or "")
            for retriever in route.get("retrievers", []):
                name = f"{layer}.{retriever}"
                if layer == "raw":
                    rows = self._raw(query, retriever, question_image_path)
                    weight = 1.0
                    if retriever == "image":
                        visual_ranked.extend(rows)
                elif layer == "fact":
                    rows = self._fact(query, retriever)
                    weight = 0.75
                elif layer == "collection":
                    rows = self._collection_facts(
                        query, retriever, question_image_path
                    )
                else:
                    rows = []
                rows = apply_frozen_time_range(
                    rows,
                    time_range,
                    date_getter=self._row_date,
                )
                if layer == "collection":
                    collection_channels.append((name, 1.0, rows))
                elif layer in {"raw", "fact"}:
                    channels.append((name, weight, rows))
                trace.append({
                    "route_index": route_index,
                    "channel": name,
                    "n": len(rows),
                })

        channels, query_fusion = normalize_query_variants(channels)
        collection_channels, collection_query_fusion = normalize_query_variants(
            collection_channels
        )
        trace.append({"channel": "query_variant_fusion", **query_fusion})
        trace.append({
            "channel": "collection_query_variant_fusion", **collection_query_fusion
        })
        fused = [self._with_refs(row) for row in self._rrf(channels)]
        collection_ranked = self._rrf_facts([
            (name, ranked) for name, _, ranked in collection_channels
        ])
        collection_ranked = rerank_collections(collection_ranked, fused)
        collection_node = None
        promoted = []
        if collection_ranked:
            collection_node, promoted = self._expand_collection(collection_ranked[0])

        # A malformed/empty route may never remove the raw safety net.
        if not fused and collection_node is None:
            fallback = self._raw(plan.get("rewritten_query") or "", "dense", None)
            fused = [self._with_refs(row) for row in fallback]
            trace.append({"channel": "fallback.raw.dense", "n": len(fused)})

        if collection_node is not None:
            promoted_ids = {row.get("mem_id") for row in promoted}
            rest = [row for row in fused if row.get("mem_id") not in promoted_ids]
            context = [collection_node] + promoted + rest
        else:
            context = fused
        selection = None
        if plan.get("strategy") == "unified_hybrid_route":
            support_quota = min(
                8,
                max(2, len(collection_node.get("refer_ids", [])))
                if collection_node else 2,
            )
            context, selection = select_context(
                context, top_k=self.top_k, support_quota=support_quota,
                expose_upper=False,
            )
            trace.append({"channel": "provenance_selector", **selection})
        else:
            context = context[: self.top_k]
        if temporal_contract:
            context = stable_sort_answer_visible_raw(
                context,
                sort_by=plan.get("sort_by"),
                enabled=True,
                date_getter=self._row_date,
            )
        elif plan.get("sort_by") == "time" and collection_node is None:
            context.sort(key=lambda row: (row.get("timestamp", 0) or 0, row.get("mem_id", "")))
        return {
            "context": context,
            "memory_image_candidates": prioritize_visual_rows(
                visual_ranked, context, lambda row: str(row.get("mem_id") or "")
            ),
            "trace": trace,
            "expanded_refer_count": len(promoted),
            "selected_collection": (
                collection_ranked[0].get("set_label") if collection_ranked else None
            ),
            # The router's image action controls whether original retrieved memory
            # images are exposed to the answer model. This also covers questions
            # that ask about a remembered image but do not attach a query image.
            "need_memory_images": any(
                "image" in route.get("retrievers", [])
                for route in plan.get("routes", [])
            ),
            "selection": selection,
        }
