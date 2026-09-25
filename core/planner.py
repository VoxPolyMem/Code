"""Shared route-plan strategies for hierarchical memory ablations.

The strategy chooses *what to search*. Retrieval, fusion, refer-id expansion,
deduplication, and top-k budgeting are owned by the common environment.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable


VALID_LAYERS = {"raw", "fact", "collection"}
VALID_RETRIEVERS = {"dense", "bm25", "image", "caption"}
LAYER_RETRIEVERS = {
    "raw": {"dense", "bm25", "image", "caption"},
    "fact": {"dense", "bm25"},
    "collection": {"dense", "bm25", "image", "caption"},
}

_VISUAL_RE = re.compile(
    r"\b(image|images|photo|photos|picture|pictures|visual|look like|depicts)\b",
    re.IGNORECASE,
)
_COLLECTION_RE = re.compile(
    r"\bhow many\b|\btotal\b|\bnumber of\b|"
    r"\b(?:search for|find) all\b|"
    r"\ball\s+(?:\w+\s+){0,3}(?:images|photos|pictures)\b|"
    r"\bwhich\s+(?:\w+\s+){0,3}(?:images|photos|pictures)\b",
    re.IGNORECASE,
)
_TEMPORAL_RE = re.compile(
    r"\b(before|after|earlier|later|first|last|then|timeline|"
    r"chronological|in order|sequence)\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


TYPE_TO_FIXED_ROUTES = {
    "TEXT_FACT": [
        {"layer": "raw", "retrievers": ["dense", "bm25"]},
        {"layer": "fact", "retrievers": ["dense", "bm25"]},
    ],
    "VISUAL_SINGLE": [
        {"layer": "raw", "retrievers": ["image", "caption", "dense"]},
    ],
    "VISUAL_COLLECTION": [
        {"layer": "collection", "retrievers": ["image", "dense", "bm25"]},
        {"layer": "raw", "retrievers": ["caption", "dense"]},
    ],
    "TEMPORAL": [
        {"layer": "raw", "retrievers": ["dense", "bm25"]},
        {"layer": "fact", "retrievers": ["dense"]},
    ],
    "CONFLICT": [
        {"layer": "raw", "retrievers": ["bm25", "dense"]},
        {"layer": "fact", "retrievers": ["bm25", "dense"]},
    ],
    "MULTI_HOP": [
        {"layer": "raw", "retrievers": ["dense", "bm25"]},
        {"layer": "fact", "retrievers": ["dense", "bm25"]},
    ],
    "EVENT_SCOPED": [
        {"layer": "collection", "retrievers": ["dense", "bm25"]},
        {"layer": "raw", "retrievers": ["dense", "bm25"]},
    ],
}


def _route(layer: str, retrievers: list[str], query: str) -> dict[str, Any]:
    return {"layer": layer, "retrievers": retrievers, "query": query}


def _validated_sub_questions(payload: dict[str, Any]) -> list[str]:
    values = payload.get("sub_questions", [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values[:2] if value]


def _load_json_object(raw: Any) -> tuple[dict[str, Any], bool]:
    """Accept strict JSON and the fenced/noisy JSON often emitted by chat models."""
    if isinstance(raw, dict):
        return raw, True
    text = str(raw or "").strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            return payload, True
    return {}, False


def heuristic_plan(question: str, has_question_image: bool = False) -> dict[str, Any]:
    """Current 4.3 observable keyword/image gate, expressed as a RoutePlan."""
    visual = has_question_image or bool(_VISUAL_RE.search(question))
    collection = visual and bool(_COLLECTION_RE.search(question))
    if collection:
        routes = [
            _route("collection", ["image", "dense", "bm25"], question),
            _route("raw", ["caption", "dense"], question),
        ]
        intent = "VISUAL_COLLECTION"
    elif visual:
        routes = [_route("raw", ["image", "caption", "dense"], question)]
        intent = "VISUAL_SINGLE"
    else:
        routes = [
            _route("raw", ["dense", "bm25"], question),
            _route("fact", ["dense", "bm25"], question),
        ]
        intent = "TEXT_FACT"
    return {
        "strategy": "heuristic_gate",
        "intent": intent,
        "rewritten_query": question,
        "routes": routes,
        "sub_questions": [],
        "time_range": None,
        "speaker_hint": None,
        "sort_by": "score",
    }


def mg_keyword_r1_plan(
    question: str, has_question_image: bool = False
) -> dict[str, Any]:
    """Portable Mem-Gallery soft-hierarchy r1 keyword gate."""
    visual = has_question_image or bool(_VISUAL_RE.search(question))
    collection = visual and bool(_COLLECTION_RE.search(question))
    if collection:
        routes = [
            _route("collection", ["image", "caption", "dense", "bm25"], question),
            _route("raw", ["image", "caption", "dense", "bm25"], question),
        ]
        intent = "VISUAL_COLLECTION"
    elif visual:
        routes = [
            _route("raw", ["image", "caption", "dense", "bm25"], question),
        ]
        intent = "VISUAL_SINGLE"
    else:
        routes = [
            _route("raw", ["dense", "bm25"], question),
            _route("fact", ["dense", "bm25"], question),
        ]
        intent = "TEMPORAL" if _TEMPORAL_RE.search(question) else "TEXT_FACT"
    explicit_date = _ISO_DATE_RE.search(question)
    return {
        "strategy": "mg_keyword_r1_port",
        "intent": intent,
        "rewritten_query": question,
        "routes": routes,
        "sub_questions": [],
        "time_range": explicit_date.group(1) if explicit_date else None,
        "speaker_hint": None,
        "sort_by": "time" if intent == "TEMPORAL" else "score",
        "uses_benchmark_label": False,
    }


def classifier_messages(question: str, has_question_image: bool) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Classify a memory question by retrieval intent. Do not answer it. "
                "Choose exactly one label using these definitions:\n"
                "VISUAL_SINGLE: identify, compare, or describe a particular image; "
                "a date inside an image question does not make it TEMPORAL.\n"
                "VISUAL_COLLECTION: count, enumerate, or retrieve an explicit set of images.\n"
                "TEMPORAL: the answer primarily asks when, before/after, or chronology.\n"
                "CONFLICT: reconcile changed or contradictory information.\n"
                "MULTI_HOP: combine multiple distinct memories to infer the answer.\n"
                "EVENT_SCOPED: retrieve details from one coherent named episode, "
                "meeting, trip, project phase, or dialogue session; use this when the "
                "relevant atomic fact is easier to find after locating that event.\n"
                "TEXT_FACT: all other factual questions.\n"
                "Return JSON only as "
                '{"intent":"LABEL"}.'
            ),
        },
        {
            "role": "user",
            "content": f"question_image_present={has_question_image}\nQuestion: {question}",
        },
    ]


def model_type_fixed_plan(
    question: str,
    has_question_image: bool,
    call_router: Callable[[list[dict[str, str]]], str],
) -> dict[str, Any]:
    """Model predicts one intent; a frozen table supplies the complete route."""
    try:
        raw_output = call_router(classifier_messages(question, has_question_image))
        payload, parse_ok = _load_json_object(raw_output)
    except Exception as exc:
        raw_output, payload, parse_ok = f"router_error: {exc}", {}, False
    intent = str(payload.get("intent", "TEXT_FACT")).upper()
    if intent not in TYPE_TO_FIXED_ROUTES:
        intent = "TEXT_FACT"
    routes = [dict(row, query=question) for row in TYPE_TO_FIXED_ROUTES[intent]]
    return {
        "strategy": "model_type_fixed_route",
        "intent": intent,
        "rewritten_query": question,
        "routes": routes,
        "sub_questions": [],
        "time_range": None,
        "speaker_hint": None,
        "sort_by": "time" if intent == "TEMPORAL" else "score",
        "router_parse_ok": parse_ok,
        "router_raw": str(raw_output)[:2000],
    }


def dynamic_messages(question: str, has_question_image: bool) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": """Plan hierarchical memory retrieval; do not answer the question.
Return strict JSON with rewritten_query, routes, sub_questions, time_range,
speaker_hint, sort_by, needs_visual_memory, needs_event_scope, and
needs_temporal_reasoning. The three needs_* fields are required booleans and are
your retrieval decisions, not benchmark labels. routes has at most two entries. Each entry is
{layer, retrievers, query}. layer is raw, fact, or collection. retrievers are
dense, bm25, image, or caption. Raw stores verbatim turns; fact stores grounded
atomic facts; collection stores grounded event, session, and image groups. Choose
only useful routes. Use collection when locating a coherent named episode/session
first will make its details easier to retrieve, or for an explicit group/count.
For chronology, keep raw evidence because it preserves order; fact can supplement
it. If question_image_present=true or visual matching is required, include image
or caption retrieval on raw/collection. Visual matching is required whenever the
question asks what a remembered image/picture/photo shows, requests an image id,
asks which image caused a later reaction, or asks about a visible attribute, even
when question_image_present=false. For image-to-reaction questions, retrieve both
the visual evidence and grounded facts that connect the image to the reaction.
Set needs_visual_memory=true for remembered photographs, screenshots, plots,
curves, charts, visible attributes, image identifiers, and image-caused
reactions, whether or not the question attaches an image. Set
needs_event_scope=true when adjacent turns jointly express an image/event and
its reaction. Set needs_temporal_reasoning=true only when ordering or comparing
events over time is required.
A date filter alone does not imply
chronological sorting. Use sort_by=time only for chronology. Never infer from
benchmark labels. Do not emit the same layer twice; combine its retrievers.""",
        },
        {
            "role": "user",
            "content": f"question_image_present={has_question_image}\nQuestion: {question}",
        },
    ]


def _validated_dynamic(
    payload: dict[str, Any], question: str, has_question_image: bool
) -> dict[str, Any]:
    routes_by_layer: dict[str, dict[str, Any]] = {}
    for candidate in payload.get("routes", [])[:4]:
        if not isinstance(candidate, dict):
            continue
        layer = str(candidate.get("layer", "")).lower()
        if layer not in VALID_LAYERS:
            continue
        retrievers = []
        for value in candidate.get("retrievers", [])[:3]:
            value = str(value).lower()
            if (
                value in VALID_RETRIEVERS
                and value in LAYER_RETRIEVERS[layer]
                and value not in retrievers
            ):
                retrievers.append(value)
        if not retrievers:
            continue
        query = str(candidate.get("query") or payload.get("rewritten_query") or question)
        if layer in routes_by_layer:
            current = routes_by_layer[layer]["retrievers"]
            current.extend(value for value in retrievers if value not in current)
        elif len(routes_by_layer) < 2:
            routes_by_layer[layer] = _route(layer, retrievers, query)
    routes = list(routes_by_layer.values())
    if not routes:
        routes = [_route("raw", ["dense", "bm25"], question)]
    def model_bool(key: str) -> bool:
        value = payload.get(key, False)
        return value is True or str(value).strip().lower() == "true"

    needs_visual = model_bool("needs_visual_memory")
    needs_event = model_bool("needs_event_scope")
    needs_temporal = model_bool("needs_temporal_reasoning")
    sort_by = (
        "time"
        if needs_temporal or str(payload.get("sort_by", "")).lower() == "time"
        else "score"
    )

    # Observable-input safety contracts. These do not use benchmark labels.
    if sort_by == "time" and not any(route["layer"] == "raw" for route in routes):
        raw_route = _route("raw", ["dense", "bm25"], str(payload.get("rewritten_query") or question))
        routes = routes[:1] + [raw_route]
    if has_question_image:
        visual = next(
            (route for route in routes if route["layer"] in {"raw", "collection"}), None
        )
        if visual is None:
            visual = _route("raw", [], str(payload.get("rewritten_query") or question))
            routes = routes[:1] + [visual]
        for retriever in ("image", "caption"):
            if retriever not in visual["retrievers"]:
                visual["retrievers"].append(retriever)
    if needs_visual:
        visual = next(
            (route for route in routes if route["layer"] in {"raw", "collection"}), None
        )
        if visual is None and len(routes) < 2:
            visual = _route(
                "raw", [], str(payload.get("rewritten_query") or question)
            )
            routes.append(visual)
        if visual is not None:
            for retriever in ("image", "caption"):
                if retriever not in visual["retrievers"]:
                    visual["retrievers"].append(retriever)
    if needs_event and not any(route["layer"] == "collection" for route in routes):
        if len(routes) < 2:
            routes.append(_route(
                "collection", ["dense", "bm25"],
                str(payload.get("rewritten_query") or question),
            ))
    explicit_date = _ISO_DATE_RE.search(question)
    return {
        "strategy": "two_level_dynamic_route",
        "intent": None,
        "rewritten_query": str(payload.get("rewritten_query") or question),
        "routes": routes,
        "sub_questions": _validated_sub_questions(payload),
        # An explicit ISO date is a safe observable constraint and takes
        # precedence over a router paraphrase. It is not a benchmark label.
        "time_range": explicit_date.group(1) if explicit_date else payload.get("time_range"),
        "speaker_hint": payload.get("speaker_hint"),
        "sort_by": sort_by,
        "model_needs_visual_memory": needs_visual,
        "model_needs_event_scope": needs_event,
        "model_needs_temporal_reasoning": needs_temporal,
    }


def two_level_dynamic_plan(
    question: str,
    has_question_image: bool,
    call_router: Callable[[list[dict[str, str]]], str],
) -> dict[str, Any]:
    """Base-model precursor of the later RL rewrite-and-route policy."""
    try:
        raw_output = call_router(dynamic_messages(question, has_question_image))
        payload, parse_ok = _load_json_object(raw_output)
    except Exception as exc:
        raw_output, payload, parse_ok = f"router_error: {exc}", {}, False
    if not parse_ok:
        payload = {"rewritten_query": question, "routes": []}
    plan = _validated_dynamic(payload, question, has_question_image)
    plan["router_parse_ok"] = parse_ok
    plan["router_raw"] = str(raw_output)[:4000]
    return plan


def unified_hybrid_plan(
    question: str,
    has_question_image: bool,
    call_router: Callable[[list[dict[str, str]]], str],
) -> dict[str, Any]:
    """Model-selected hierarchy plus an original-question raw safety route."""
    plan = two_level_dynamic_plan(question, has_question_image, call_router)
    return _apply_unified_contract(plan, question)


def _apply_unified_contract(
    plan: dict[str, Any], question: str,
) -> dict[str, Any]:
    """Apply the shared safety routes to an initial or refinement plan."""
    # A model-declared visual event needs both coarse scope and a grounded
    # relation path.  If the model selected only collection retrieval, add the
    # fact layer whose refer_ids connect a reaction/detail back to nearby raw
    # media.  This contract is driven by the model's explicit decisions, not by
    # benchmark labels or question keywords.
    if (
        plan.get("model_needs_visual_memory")
        and plan.get("model_needs_event_scope")
        and not any(route["layer"] == "fact" for route in plan["routes"])
    ):
        plan["routes"].append(_route(
            "fact", ["dense", "bm25"],
            str(plan.get("rewritten_query") or question),
        ))
    original_raw = next(
        (
            route for route in plan["routes"]
            if route["layer"] == "raw"
            and str(route.get("query") or "").strip() == question.strip()
        ),
        None,
    )
    if original_raw is None:
        plan["routes"].append(_route("raw", ["dense", "bm25"], question))
    else:
        for retriever in ("dense", "bm25"):
            if retriever not in original_raw["retrievers"]:
                original_raw["retrievers"].append(retriever)
    for route in plan["routes"]:
        if route["layer"] in {"fact", "collection"}:
            for retriever in ("dense", "bm25"):
                if retriever not in route["retrievers"]:
                    route["retrievers"].append(retriever)
    plan["strategy"] = "unified_hybrid_route"
    plan["retrieval_floor"] = (
        "raw_dense_bm25_on_original_question; model_selects_upper_layers_and_queries; "
        "same-channel_query_variants_share_one_fusion_vote"
    )
    plan["uses_benchmark_label"] = False
    return plan


def _compact_retrieval_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for item in history:
        plan = item.get("plan") or {}
        compact.append({
            "round": item.get("round"),
            "query": plan.get("rewritten_query"),
            "actions": [
                {
                    "layer": route.get("layer"),
                    "retrievers": route.get("retrievers", []),
                    "query": route.get("query"),
                }
                for route in plan.get("routes", [])
            ],
            "sub_questions": plan.get("sub_questions", []),
            "new_unique": item.get("new_unique", 0),
            "packed_context_ids": item.get("top30_ids", []),
            "missing_evidence": item.get("missing_evidence", []),
        })
    return compact


def _compact_evidence(rows: list[dict[str, Any]]) -> str:
    lines = []
    for index, row in enumerate(rows[:60], 1):
        identity = str(row.get("mem_id") or row.get("node_id") or f"row:{index}")
        captions = row.get("image_captions", row.get("caption", {})) or {}
        if isinstance(captions, dict):
            image_ids = list(captions)[:6]
            caption_text = " | ".join(str(value) for value in captions.values())
        else:
            image_ids = []
            caption_text = str(captions)
        if not image_ids:
            image_ids = [
                str(value) for value in row.get("image_ids", [])[:6]
            ]
        text = str(row.get("text") or "").replace("\n", " ")[:520]
        caption_text = caption_text.replace("\n", " ")[:320]
        lines.append(
            f"[{index}] id={identity}; speaker={row.get('speaker', 'unknown')}; "
            f"date={row.get('date', '')}; refer_ids={list(row.get('refer_ids', []))[:8]}; "
            f"image_ids={image_ids}; text={text}; captions={caption_text}"
        )
    return "\n".join(lines)


def retrieval_refinement_messages(
    question: str,
    has_question_image: bool,
    current_context: list[dict[str, Any]],
    history: list[dict[str, Any]],
    round_index: int,
    max_rounds: int,
) -> list[dict[str, str]]:
    """Ask once whether to stop or perform one targeted retrieval action."""
    return [
        {
            "role": "system",
            "content": """Control a bounded multi-round hierarchical memory retriever. Do not answer the question.
Judge whether the CURRENT PACKED EVIDENCE is sufficient to answer the question accurately and with the requested identifiers, entities, attributes, chronology, or image-event relation. Previous retrieval actions and their results are visible. Do not repeat an action unless it targets a clearly unresolved gap. Evidence is sufficient when the answer can be composed from the available grounded source rows; do not require the memory to already contain a prewritten final summary, timeline, comparison, or answer. A missing_evidence item must name a missing source fact or source relation, never a desired summary of evidence that is already present.

If evidence is sufficient, return strict JSON:
{"sufficient":true,"reasoning":"brief reason","missing_evidence":[],"supporting_memory_ids":["exact id from evidence"],"supporting_image_ids":["exact image id from evidence"]}

If evidence is insufficient, return strict JSON with sufficient=false plus rewritten_query, routes, sub_questions, time_range, speaker_hint, sort_by, needs_visual_memory, needs_event_scope, and needs_temporal_reasoning. missing_evidence must name the unresolved evidence needed by the next round. routes contains at most two entries of {layer,retrievers,query}. layer is raw, fact, or collection. retrievers are dense, bm25, image, or caption. Raw stores verbatim turns; fact stores grounded atomic facts; collection stores event/session/image groups. Use image or caption for remembered visual evidence and use fact/collection only when their refer_ids help recover grounded raw turns. Use sort_by=time only for actual chronology. Do not use benchmark labels or guess the answer. The next action must target missing evidence rather than merely paraphrase the original query.

supporting_memory_ids and supporting_image_ids are evidence selectors, not an answer. Copy only exact IDs visibly present in CURRENT PACKED EVIDENCE. If the question requests an exact identifier and that identifier is not explicitly present, sufficient must be false even when the likely answer can be inferred.""",
        },
        {
            "role": "user",
            "content": (
                f"retrieval_round={round_index + 1}/{max_rounds}\n"
                f"question_image_present={has_question_image}\n"
                f"Question: {question}\n\n"
                "PREVIOUS RETRIEVAL ACTIONS:\n"
                f"{json.dumps(_compact_retrieval_history(history), ensure_ascii=False)}\n\n"
                f"CURRENT PACKED TOP-{len(current_context)} EVIDENCE:\n"
                f"{_compact_evidence(current_context)}"
            ),
        },
    ]


def iterative_refinement_decision(
    question: str,
    has_question_image: bool,
    current_context: list[dict[str, Any]],
    history: list[dict[str, Any]],
    round_index: int,
    max_rounds: int,
    call_router: Callable[[list[dict[str, str]]], str],
) -> dict[str, Any]:
    """Return a validated stop/refine decision without using benchmark labels."""
    try:
        raw_output = call_router(retrieval_refinement_messages(
            question,
            has_question_image,
            current_context,
            history,
            round_index,
            max_rounds,
        ))
        payload, parse_ok = _load_json_object(raw_output)
    except Exception as exc:
        raw_output, payload, parse_ok = f"refinement_error: {exc}", {}, False
    if not parse_ok:
        return {
            "sufficient": True,
            "parse_ok": False,
            "stop_reason": "refinement_parse_failed",
            "missing_evidence": [],
            "raw": str(raw_output)[:4000],
            "plan": None,
        }
    sufficient = payload.get("sufficient") is True or str(
        payload.get("sufficient", "")
    ).strip().lower() == "true"
    missing = payload.get("missing_evidence", [])
    if not isinstance(missing, list):
        missing = [str(missing)] if missing else []
    decision = {
        "sufficient": sufficient,
        "parse_ok": True,
        "stop_reason": "evidence_sufficient" if sufficient else None,
        "reasoning": str(payload.get("reasoning") or "")[:1000],
        "missing_evidence": [str(value) for value in missing[:6] if value],
        "supporting_memory_ids": [
            str(value) for value in payload.get("supporting_memory_ids", [])[:12]
            if value
        ] if isinstance(payload.get("supporting_memory_ids", []), list) else [],
        "supporting_image_ids": [
            str(value) for value in payload.get("supporting_image_ids", [])[:12]
            if value
        ] if isinstance(payload.get("supporting_image_ids", []), list) else [],
        "raw": str(raw_output)[:4000],
        "plan": None,
    }
    if not sufficient:
        plan = _validated_dynamic(payload, question, has_question_image)
        decision["plan"] = _apply_unified_contract(plan, question)
    return decision
