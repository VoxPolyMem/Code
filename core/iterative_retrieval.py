"""Bounded sufficiency-driven retrieval over the shared hierarchical executor.

Round 0 uses the normal unified plan.  After every retrieval action, accumulated
evidence is deduplicated and repacked to the same TopK contract before the model
decides whether to stop or issue one targeted rewrite-and-route action.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os
from typing import Any, Callable, Mapping

from core.evidence_bundle import merge_evidence_groups
from core.planner import iterative_refinement_decision
from core.visual_budget import prioritize_visual_rows


ITERATIVE_RETRIEVAL_ENV = "MPMEM_ITERATIVE_RETRIEVAL_V1"
MAX_ROUNDS_ENV = "MPMEM_ITERATIVE_MAX_ROUNDS"
ROUND_BUDGETS_ENV = "MPMEM_ITERATIVE_ROUND_BUDGETS"
FINAL_TOP_K_ENV = "MPMEM_FINAL_TOP_K"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_DEFAULT_BUDGETS = (30, 8, 5, 3, 2)


def iterative_retrieval_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if environ is None else environ
    return str(source.get(ITERATIVE_RETRIEVAL_ENV, "")).strip().lower() in _TRUE_VALUES


def iterative_limits(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, tuple[int, ...]]:
    source = os.environ if environ is None else environ
    try:
        max_rounds = int(source.get(MAX_ROUNDS_ENV, "5"))
    except (TypeError, ValueError):
        max_rounds = 5
    max_rounds = max(1, min(5, max_rounds))
    values = []
    for value in str(source.get(
        ROUND_BUDGETS_ENV, ",".join(map(str, _DEFAULT_BUDGETS))
    )).split(","):
        try:
            values.append(max(1, int(value.strip())))
        except (TypeError, ValueError):
            continue
    if not values:
        values = list(_DEFAULT_BUDGETS)
    while len(values) < max_rounds:
        values.append(max(1, values[-1] // 2))
    # Later rounds may inspect fewer newly ranked candidates, never more.
    for index in range(1, len(values)):
        values[index] = min(values[index], values[index - 1])
    return max_rounds, tuple(values[:max_rounds])


def configured_top_k(
    environ: Mapping[str, str] | None = None,
) -> int:
    """Return a bounded final retrieval size; the frozen default remains 30."""
    source = os.environ if environ is None else environ
    try:
        value = int(source.get(FINAL_TOP_K_ENV, "30"))
    except (TypeError, ValueError):
        value = 30
    return max(10, min(60, value))


def _identity(row: dict[str, Any], fallback: str) -> str:
    return str(row.get("mem_id") or row.get("node_id") or fallback)


def _merge_row_metadata(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    merged["evidence_groups"] = merge_evidence_groups(
        left.get("evidence_groups"), right.get("evidence_groups")
    )
    merged["retrieval_channels"] = list(dict.fromkeys(
        list(left.get("retrieval_channels", []))
        + list(right.get("retrieval_channels", []))
    ))
    return merged


def merge_round_contexts(
    contexts: list[list[dict[str, Any]]],
    budgets: tuple[int, ...],
    *,
    top_k: int = 30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Admit a decreasing number of new rows while preserving the round-0 core."""
    packed: list[dict[str, Any]] = []
    all_candidate_ids: set[str] = set()
    seen_per_round = []
    all_seen: set[str] = set()
    for round_index, ranked in enumerate(contexts):
        budget = budgets[min(round_index, len(budgets) - 1)]
        round_seen: set[str] = set()
        candidates: list[dict[str, Any]] = []
        for rank, row in enumerate(ranked[:budget], 1):
            identity = _identity(row, f"round:{round_index}:row:{rank}")
            if identity in round_seen:
                continue
            round_seen.add(identity)
            candidates.append(dict(row))
        new_ids = round_seen - all_seen
        packed_by_id = {
            _identity(row, f"packed:{index}"): index
            for index, row in enumerate(packed)
        }
        for row in candidates:
            identity = _identity(row, "")
            if identity in packed_by_id:
                index = packed_by_id[identity]
                packed[index] = _merge_row_metadata(packed[index], row)
        admitted = [
            row for row in candidates
            if _identity(row, "") not in all_seen
        ][:budget]
        if round_index == 0:
            packed = admitted[:top_k]
        elif admitted:
            # Targeted later evidence gets bounded slots; it cannot globally
            # rerank or replace the high-ranked round-0 core.
            keep = max(0, top_k - len(admitted))
            packed = packed[:keep] + admitted[:top_k]
        seen_per_round.append({
            "round": round_index,
            "budget": budget,
            "considered_unique": len(round_seen),
            "new_unique": len(new_ids),
            "admitted_new": len(admitted),
        })
        all_seen.update(round_seen)
        all_candidate_ids.update(round_seen)
    return packed, {
        "round_stats": seen_per_round,
        "candidate_unique": len(all_candidate_ids),
        "packed": len(packed),
        "deduplicated": sum(item["considered_unique"] for item in seen_per_round)
        - len(all_candidate_ids),
    }


def _action_signature(plan: dict[str, Any]) -> str:
    payload = {
        "query": str(plan.get("rewritten_query") or "").strip().lower(),
        "routes": [
            {
                "layer": route.get("layer"),
                "retrievers": sorted(set(route.get("retrievers", []))),
                "query": str(route.get("query") or "").strip().lower(),
            }
            for route in plan.get("routes", [])
        ],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _combined_plan(plans: list[dict[str, Any]], question: str) -> dict[str, Any]:
    sub_questions = []
    routes = []
    for plan in plans:
        for value in plan.get("sub_questions", []) or []:
            if value and value not in sub_questions:
                sub_questions.append(value)
        routes.extend(dict(route) for route in plan.get("routes", []))
    return {
        "strategy": "unified_hybrid_route",
        "intent": None,
        "rewritten_query": plans[-1].get("rewritten_query", question),
        "routes": routes,
        "sub_questions": sub_questions[:8],
        "time_range": plans[-1].get("time_range"),
        "speaker_hint": plans[-1].get("speaker_hint"),
        "sort_by": "time" if any(
            plan.get("sort_by") == "time" for plan in plans
        ) else "score",
        "model_needs_visual_memory": any(
            plan.get("model_needs_visual_memory") for plan in plans
        ),
        "model_needs_event_scope": any(
            plan.get("model_needs_event_scope") for plan in plans
        ),
        "model_needs_temporal_reasoning": any(
            plan.get("model_needs_temporal_reasoning") for plan in plans
        ),
        "uses_benchmark_label": False,
        "iterative_round_count": len(plans),
    }


def prioritize_selected_evidence(
    rows: list[dict[str, Any]],
    supporting_memory_ids: list[str],
    supporting_image_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Move model-selected evidence first without changing TopK membership."""
    memory_order = {
        str(identity): index for index, identity in enumerate(supporting_memory_ids)
    }
    image_order = {
        str(identity): index for index, identity in enumerate(supporting_image_ids)
    }

    def row_image_ids(row: dict[str, Any]) -> list[str]:
        captions = row.get("image_captions", row.get("caption", {})) or {}
        ids = list(captions) if isinstance(captions, dict) else []
        ids.extend(str(value) for value in row.get("image_ids", []) if value)
        return list(dict.fromkeys(ids))

    ranked = []
    selected_rows = 0
    selected_images = set()
    for original_index, row in enumerate(rows):
        identity = _identity(row, f"row:{original_index}")
        images = row_image_ids(row)
        mem_rank = memory_order.get(identity)
        image_ranks = [image_order[value] for value in images if value in image_order]
        if mem_rank is not None or image_ranks:
            selected_rows += 1
            selected_images.update(value for value in images if value in image_order)
        ranked.append((
            0 if mem_rank is not None else 1,
            mem_rank if mem_rank is not None else 2**31 - 1,
            0 if image_ranks else 1,
            min(image_ranks) if image_ranks else 2**31 - 1,
            original_index,
            row,
        ))
    ranked.sort(key=lambda item: item[:-1])
    output = [item[-1] for item in ranked]
    before = sorted(_identity(row, f"row:{i}") for i, row in enumerate(rows))
    after = sorted(_identity(row, f"row:{i}") for i, row in enumerate(output))
    if before != after:
        raise AssertionError("evidence selection changed TopK membership")
    return output, {
        "requested_memory_ids": list(memory_order),
        "requested_image_ids": list(image_order),
        "matched_rows": selected_rows,
        "matched_image_ids": sorted(selected_images),
        "membership_preserved": True,
    }


def retrieve_iteratively(
    *,
    question: str,
    has_question_image: bool,
    initial_plan: dict[str, Any],
    execute: Callable[[dict[str, Any]], dict[str, Any]],
    call_router: Callable[[list[dict[str, str]]], str],
    top_k: int = 30,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run at most five retrieval rounds, with one combined decision per retry."""
    max_rounds, budgets = iterative_limits(environ)
    plans = [initial_plan]
    outputs = [execute(initial_plan)]
    contexts = [list(outputs[0].get("context", []))]
    packed, merge_info = merge_round_contexts(contexts, budgets, top_k=top_k)
    history = [{
        "round": 0,
        "plan": initial_plan,
        "new_unique": merge_info["round_stats"][0]["new_unique"],
        "top30_ids": [_identity(row, f"row:{i}") for i, row in enumerate(packed)],
        "missing_evidence": [],
    }]
    signatures = {_action_signature(initial_plan)}
    decisions = []
    final_supporting_memory_ids: list[str] = []
    final_supporting_image_ids: list[str] = []
    stop_reason = "max_rounds_reached"

    for round_index in range(1, max_rounds):
        decision = iterative_refinement_decision(
            question,
            has_question_image,
            packed,
            history,
            round_index - 1,
            max_rounds,
            call_router,
        )
        decisions.append({key: value for key, value in decision.items() if key != "plan"})
        if decision.get("sufficient"):
            stop_reason = str(decision.get("stop_reason") or "evidence_sufficient")
            final_supporting_memory_ids = list(
                decision.get("supporting_memory_ids", [])
            )
            final_supporting_image_ids = list(
                decision.get("supporting_image_ids", [])
            )
            break
        plan = decision.get("plan")
        if not plan:
            stop_reason = "missing_refinement_plan"
            break
        signature = _action_signature(plan)
        if signature in signatures:
            stop_reason = "duplicate_retrieval_action"
            break
        signatures.add(signature)
        output = execute(plan)
        prior_ids = {
            _identity(row, f"prior:{i}")
            for current in contexts
            for i, row in enumerate(current)
        }
        new_ids = {
            _identity(row, f"new:{i}")
            for i, row in enumerate(output.get("context", []))
        } - prior_ids
        plans.append(plan)
        outputs.append(output)
        contexts.append(list(output.get("context", [])))
        packed, merge_info = merge_round_contexts(contexts, budgets, top_k=top_k)
        admitted_new = int(merge_info["round_stats"][-1]["admitted_new"])
        history.append({
            "round": round_index,
            "plan": plan,
            "new_unique": admitted_new,
            "discovered_unique": len(new_ids),
            "top30_ids": [_identity(row, f"row:{i}") for i, row in enumerate(packed)],
            "missing_evidence": decision.get("missing_evidence", []),
        })
        if not admitted_new:
            stop_reason = "no_new_evidence"
            break

    combined_plan = _combined_plan(plans, question)
    relevance_order, selector_info = prioritize_selected_evidence(
        list(packed),
        final_supporting_memory_ids,
        final_supporting_image_ids,
    )
    packed = list(relevance_order)
    if combined_plan.get("sort_by") == "time":
        def time_key(row: dict[str, Any]) -> tuple[Any, ...]:
            return (
                str(row.get("date") or "9999-12-31"),
                row.get("timestamp", 0) or 0,
                str(row.get("session_id") or row.get("session") or ""),
                row.get("ordinal", row.get("turn_seq", 0)) or 0,
            )
        packed = sorted(packed, key=time_key)
    selected_visual_rows = []
    selected_image_set = set(final_supporting_image_ids)
    for row in relevance_order:
        captions = row.get("image_captions", row.get("caption", {})) or {}
        image_ids = set(captions) if isinstance(captions, dict) else set()
        image_ids.update(str(value) for value in row.get("image_ids", []) if value)
        if image_ids & selected_image_set:
            selected_visual_rows.append(row)
    visual_ranked = selected_visual_rows + [
        row
        for output in outputs
        for row in output.get("memory_image_candidates", [])
    ]
    image_candidates = prioritize_visual_rows(
        visual_ranked,
        relevance_order,
        lambda row: _identity(row, ""),
    )
    return {
        "context": packed,
        "memory_image_candidates": image_candidates,
        "trace": [
            {"round": index, "events": output.get("trace", [])}
            for index, output in enumerate(outputs)
        ],
        "need_memory_images": any(
            output.get("need_memory_images") for output in outputs
        ),
        "selected_collection": [
            output.get("selected_collection") for output in outputs
            if output.get("selected_collection")
        ],
        "expanded_refer_count": sum(
            int(output.get("expanded_refer_count", 0)) for output in outputs
        ),
        "route_plan": combined_plan,
        "iterative_retrieval": {
            "enabled": True,
            "max_rounds": max_rounds,
            "round_budgets": list(budgets),
            "rounds_executed": len(outputs),
            "stop_reason": stop_reason,
            "history": history,
            "decisions": decisions,
            "merge": merge_info,
            "evidence_selector": selector_info,
            "uses_benchmark_label": False,
        },
    }
