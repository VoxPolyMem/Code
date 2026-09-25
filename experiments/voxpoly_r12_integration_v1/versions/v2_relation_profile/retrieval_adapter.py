"""Pure AudioMem Top-30 relation/profile projection.

The adapter consumes only the question, retrieval ranks, an optional
query-acoustic identity observation, and QA-free memory metadata.  Answers,
gold evidence, benchmark categories, and QA-type labels are deliberately not
accepted by any public function.

The final answer context contains raw turns only.  Facts and profiles are
navigation indexes whose ``refer_ids`` are projected back to raw evidence.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


TOP_K = 30
RAW_SAFETY_PREFIX = 24
MAX_RELATION_ROWS = 4
MAX_PROFILE_STATE_ROWS = 2
MAX_ANCHORS = 2
MIN_ANCHOR_COVERAGE = 0.38

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
QUOTE_RE = re.compile(r"[\"“”‘’]([^\"“”‘’]{8,})[\"“”‘’]")
STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "did", "do",
    "for", "from", "had", "has", "have", "he", "her", "hers", "him",
    "his", "how", "i", "in", "is", "it", "its", "me", "my", "of",
    "on", "or", "our", "she", "that", "the", "their", "them", "they",
    "this", "to", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "whom", "why", "with", "you", "your",
}


class RetrievalContractError(ValueError):
    """Raised when a memory or retrieval input violates the isolated contract."""


@dataclass(frozen=True)
class Expansion:
    row: dict[str, Any]
    anchor_node_id: str | None
    reason: str
    priority: float


def _node_id(row: Mapping[str, Any], fallback: str = "") -> str:
    return str(row.get("node_id") or row.get("mem_id") or fallback)


def _refs(row: Mapping[str, Any]) -> list[str]:
    values = row.get("refer_ids") or []
    if not isinstance(values, (list, tuple)):
        return []
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _tokens(text: Any) -> list[str]:
    return TOKEN_RE.findall(str(text or "").casefold())


def _informative_tokens(text: Any) -> set[str]:
    return {token for token in _tokens(text) if token not in STOPWORDS and len(token) > 1}


def _normalized_text(text: Any) -> str:
    return " ".join(_tokens(text))


def _lexical_coverage(question: str, text: str) -> float:
    query = _informative_tokens(question)
    if not query:
        return 0.0
    return len(query & set(_tokens(text))) / len(query)


def _copy_raw(row: Mapping[str, Any], **trace: Any) -> dict[str, Any]:
    copied = dict(row)
    if trace:
        copied["audio_retrieval_v2"] = trace
    return copied


def validate_memory(memory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate the raw/provenance namespace and return ``raw_by_ref``."""

    raw = memory.get("raw")
    facts = memory.get("facts")
    profiles = memory.get("profiles")
    if not all(isinstance(value, list) for value in (raw, facts, profiles)):
        raise RetrievalContractError("memory must contain raw/facts/profiles lists")
    raw_by_ref: dict[str, dict[str, Any]] = {}
    node_ids: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise RetrievalContractError(f"raw[{index}] is not an object")
        node_id = _node_id(row)
        refs = _refs(row)
        if not node_id or len(refs) != 1:
            raise RetrievalContractError(f"raw[{index}] lacks one stable refer_id")
        if node_id in node_ids or refs[0] in raw_by_ref:
            raise RetrievalContractError("raw node_id/refer_id is duplicated")
        node_ids.add(node_id)
        raw_by_ref[refs[0]] = row
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            raise RetrievalContractError(f"facts[{index}] is not an object")
        dangling = set(_refs(fact)) - set(raw_by_ref)
        if dangling:
            raise RetrievalContractError(f"facts[{index}] has dangling refs: {sorted(dangling)}")
    profile_ids: set[str] = set()
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            raise RetrievalContractError(f"profiles[{index}] is not an object")
        stable = str(profile.get("stable_speaker_id") or "").strip()
        if not stable or stable in profile_ids:
            raise RetrievalContractError("profile stable IDs are missing or duplicated")
        profile_ids.add(stable)
        dangling = set(_refs(profile)) - set(raw_by_ref)
        if dangling:
            raise RetrievalContractError(f"profiles[{index}] has dangling refs")
    return raw_by_ref


def raw_multichannel_rrf_topk(
    channels_input: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    *,
    top_k: int = TOP_K,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse named raw rank lists without inspecting any QA annotation.

    A row may appear in an arbitrary number of channels.  This makes profile,
    modality, and rewritten-query views composable while preserving a single
    explicit Top-K budget and a raw-only output contract.
    """

    if top_k <= 0:
        return []
    scores: dict[str, float] = defaultdict(float)
    rows: dict[str, Mapping[str, Any]] = {}
    channels: dict[str, list[str]] = defaultdict(list)
    for channel, ranked in channels_input:
        seen: set[str] = set()
        for rank, row in enumerate(ranked, 1):
            identity = _node_id(row)
            if not identity or identity in seen:
                continue
            seen.add(identity)
            rows.setdefault(identity, row)
            scores[identity] += 1.0 / (rrf_k + rank)
            channels[identity].append(channel)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    return [
        _copy_raw(
            rows[identity],
            source="raw_hybrid_global",
            hybrid_score=scores[identity],
            channels=channels[identity],
        )
        for identity in ordered
    ]


def raw_hybrid_topk(
    dense_rows: Sequence[Mapping[str, Any]],
    bm25_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int = TOP_K,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse raw dense/BM25 ranks without inspecting any QA annotation."""

    return raw_multichannel_rrf_topk(
        (("raw.dense", dense_rows), ("raw.bm25", bm25_rows)),
        top_k=top_k,
        rrf_k=rrf_k,
    )


def select_semantic_anchors(
    question: str,
    raw_ranked: Sequence[Mapping[str, Any]],
    *,
    max_anchors: int = MAX_ANCHORS,
    min_coverage: float = MIN_ANCHOR_COVERAGE,
) -> list[dict[str, Any]]:
    """Select strong raw utterance anchors from observable query content.

    Quoted utterances receive exact normalized matching.  Otherwise the gate
    uses lexical coverage over the already semantic-ranked raw list.  It does
    not inspect a task label or infer a relation direction from keywords.
    """

    quotes = [_normalized_text(value) for value in QUOTE_RE.findall(question)]
    quotes = [value for value in quotes if value]
    scored: list[tuple[float, int, dict[str, Any], str]] = []
    for rank, source in enumerate(raw_ranked):
        text = str(source.get("text") or source.get("retrieval_text") or "")
        normalized = _normalized_text(text)
        exact_quote = any(quote in normalized or normalized in quote for quote in quotes)
        coverage = _lexical_coverage(question, text)
        if quotes and not exact_quote:
            continue
        if not exact_quote and coverage < min_coverage:
            continue
        confidence = (1.0 if exact_quote else coverage) + 1.0 / (1000 + rank)
        scored.append((confidence, rank, dict(source), "exact_quote" if exact_quote else "lexical_semantic"))
    scored.sort(key=lambda value: (-value[0], value[1], _node_id(value[2])))
    output = []
    for confidence, rank, row, source in scored[:max_anchors]:
        output.append(_copy_raw(
            row,
            source="semantic_anchor",
            anchor_source=source,
            anchor_confidence=min(confidence, 1.0),
            raw_rank=rank + 1,
        ))
    return output


def _raw_indexes(
    raw_by_ref: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[str, int], Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    by_position: dict[tuple[str, int], Mapping[str, Any]] = {}
    children: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_by_ref.values():
        session = str(row.get("session_id") or "")
        ordinal = row.get("ordinal")
        if session and isinstance(ordinal, int):
            by_position[(session, ordinal)] = row
        parent = str(row.get("reply_to_turn_id") or "").strip()
        if parent:
            children[parent].append(row)
    for rows in children.values():
        rows.sort(key=lambda row: (
            str(row.get("session_id") or ""), int(row.get("ordinal") or 0), _node_id(row)
        ))
    return by_position, children


def _relation_expansions(
    anchors: Sequence[Mapping[str, Any]],
    memory: Mapping[str, Any],
    raw_by_ref: Mapping[str, Mapping[str, Any]],
    *,
    max_rows: int = MAX_RELATION_ROWS,
) -> list[Expansion]:
    by_position, children = _raw_indexes(raw_by_ref)
    facts_by_ref: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for fact in memory.get("facts") or []:
        refs = _refs(fact)
        addressees = {
            str(value) for value in fact.get("addressee_refs") or []
            if str(value) not in {"", "unknown", "group"}
        }
        if not 1 <= len(refs) <= 4 or not addressees:
            continue
        for ref in refs:
            facts_by_ref[ref].append(fact)

    candidates: dict[str, Expansion] = {}

    def keep(row: Mapping[str, Any] | None, anchor: Mapping[str, Any], reason: str, priority: float) -> None:
        if row is None:
            return
        identity = _node_id(row)
        if not identity or identity == _node_id(anchor):
            return
        proposed = Expansion(
            _copy_raw(row, source="relation_expansion", reason=reason, anchor_node_id=_node_id(anchor)),
            _node_id(anchor),
            reason,
            priority,
        )
        current = candidates.get(identity)
        if current is None or proposed.priority > current.priority:
            candidates[identity] = proposed

    for anchor_rank, anchor in enumerate(anchors):
        refs = _refs(anchor)
        if len(refs) != 1:
            continue
        ref = refs[0]
        anchor_weight = 1.0 / (1 + anchor_rank)
        parent = str(anchor.get("reply_to_turn_id") or "").strip()
        if parent:
            keep(raw_by_ref.get(parent), anchor, "reply_to", 5.0 + anchor_weight)
        for child in children.get(ref, []):
            keep(child, anchor, "replied_by", 4.8 + anchor_weight)
        for fact in facts_by_ref.get(ref, []):
            for sibling in _refs(fact):
                if sibling != ref:
                    keep(raw_by_ref.get(sibling), anchor, "fact_addressee_sibling", 4.0 + anchor_weight)
        session = str(anchor.get("session_id") or "")
        ordinal = anchor.get("ordinal")
        if session and isinstance(ordinal, int):
            # Responses after an anchor are slightly preferred, but both sides
            # remain visible for reverse-reply questions.
            for position, delta in enumerate((1, -1, 2, -2)):
                keep(
                    by_position.get((session, ordinal + delta)),
                    anchor,
                    f"local_{delta:+d}",
                    2.0 - 0.15 * position + anchor_weight,
                )
    return sorted(
        candidates.values(),
        key=lambda item: (-item.priority, item.anchor_node_id or "", _node_id(item.row)),
    )[:max_rows]


def _identity_fields(query_identity: Mapping[str, Any] | None) -> tuple[str | None, str]:
    if not query_identity:
        return None, "unresolved"
    stable = str(query_identity.get("stable_asker_ref") or "").strip() or None
    confidence = str(query_identity.get("confidence") or "unresolved")
    if confidence not in {"high", "low", "unresolved"}:
        raise RetrievalContractError(f"unknown query identity confidence: {confidence}")
    if confidence != "unresolved" and not stable:
        raise RetrievalContractError("resolved identity lacks stable_asker_ref")
    return stable, confidence


def _profile_state_expansions(
    question: str,
    memory: Mapping[str, Any],
    raw_by_ref: Mapping[str, Mapping[str, Any]],
    query_identity: Mapping[str, Any] | None,
    ranked_facts: Sequence[Mapping[str, Any]],
    raw_ranked: Sequence[Mapping[str, Any]],
    *,
    max_rows: int = MAX_PROFILE_STATE_ROWS,
) -> list[Expansion]:
    stable, confidence = _identity_fields(query_identity)
    if confidence != "high" or not stable or max_rows <= 0:
        return []
    known_profiles = {
        str(profile.get("stable_speaker_id") or "")
        for profile in memory.get("profiles") or []
    }
    if stable not in known_profiles:
        raise RetrievalContractError("query identity is absent from memory profiles")

    rank_by_id = {
        _node_id(fact): rank for rank, fact in enumerate(ranked_facts)
        if _node_id(fact)
    }
    # Tie identity-relevant state back to the event neighborhood already found
    # by raw semantic retrieval.  This resolves same-speaker, same-action but
    # different-event collisions without any benchmark keyword or QA label.
    semantic_positions: list[tuple[str, int, int]] = []
    for raw_rank, row in enumerate(raw_ranked[:12]):
        session = str(row.get("session_id") or "")
        ordinal = row.get("ordinal")
        if session and isinstance(ordinal, int):
            semantic_positions.append((session, ordinal, raw_rank))

    def event_coherence(fact: Mapping[str, Any]) -> float:
        positions = []
        for ref in _refs(fact):
            raw = raw_by_ref.get(ref)
            if raw is None:
                continue
            session = str(raw.get("session_id") or "")
            ordinal = raw.get("ordinal")
            if session and isinstance(ordinal, int):
                positions.append((session, ordinal))
        best = 0.0
        for session, ordinal in positions:
            for semantic_session, semantic_ordinal, raw_rank in semantic_positions:
                if semantic_session != session:
                    continue
                distance = abs(ordinal - semantic_ordinal)
                locality = max(0.0, 1.0 - distance / 8.0)
                rank_discount = 1.0 / (1.0 + raw_rank / 6.0)
                best = max(best, locality * rank_discount)
        return best

    candidates: dict[str, Expansion] = {}
    for fact in memory.get("facts") or []:
        source_match = str(fact.get("source_speaker_ref") or "") == stable
        addressee_match = stable in {
            str(value) for value in fact.get("addressee_refs") or []
        }
        if not (source_match or addressee_match):
            continue
        coverage = _lexical_coverage(question, str(fact.get("text") or ""))
        rank = rank_by_id.get(_node_id(fact), 10**6)
        # A fact must either have substantive lexical support or have appeared
        # in the semantic fact rank.  Identity alone never selects a state.
        if coverage < 0.18 and rank >= 30:
            continue
        role = "asker_spoke" if source_match else "assigned_to_asker"
        coherence = event_coherence(fact)
        priority = (
            3.0 * coverage
            + (1.0 if source_match else 0.75)
            + 0.6 * coherence
            + 1.0 / (60 + rank)
        )
        for ref in _refs(fact):
            row = raw_by_ref.get(ref)
            if row is None:
                continue
            identity = _node_id(row)
            proposed = Expansion(
                _copy_raw(
                    row,
                    source="profile_state",
                    reason=role,
                    fact_node_id=_node_id(fact),
                    stable_asker_ref=stable,
                    event_coherence=coherence,
                ),
                None,
                role,
                priority,
            )
            current = candidates.get(identity)
            if current is None or proposed.priority > current.priority:
                candidates[identity] = proposed
    return sorted(candidates.values(), key=lambda item: (-item.priority, _node_id(item.row)))[:max_rows]


def _soft_identity_reorder(
    rows: Sequence[Mapping[str, Any]],
    question: str,
    memory: Mapping[str, Any],
    stable: str,
) -> list[dict[str, Any]]:
    """Capped membership-preserving boost for a low-confidence acoustic match."""

    related_refs: set[str] = set()
    for fact in memory.get("facts") or []:
        source_match = str(fact.get("source_speaker_ref") or "") == stable
        addressee_match = stable in {str(value) for value in fact.get("addressee_refs") or []}
        if (source_match or addressee_match) and _lexical_coverage(question, str(fact.get("text") or "")) >= 0.18:
            related_refs.update(_refs(fact))
    scored = []
    for rank, source in enumerate(rows):
        refs = set(_refs(source))
        direct = str(source.get("speaker_ref") or "") == stable
        relation = bool(refs & related_refs)
        lexical = _lexical_coverage(question, str(source.get("text") or ""))
        # At most a small, two-rank-scale nudge; unrelated identity history is
        # not promoted merely because the same speaker appears there.
        boost = min(0.018, (0.010 * direct + 0.008 * relation) * min(1.0, lexical / 0.2))
        base = 1.0 - rank / max(4 * len(rows), 1)
        scored.append((-(base + boost), rank, _copy_raw(
            source,
            source="low_confidence_identity_soft",
            stable_asker_ref=stable,
            boost=boost,
        )))
    scored.sort(key=lambda value: (value[0], value[1]))
    return [value[2] for value in scored]


def _pack_with_anchors(
    base_rows: Sequence[Mapping[str, Any]],
    anchors: Sequence[Mapping[str, Any]],
    relation: Sequence[Expansion],
    state: Sequence[Expansion],
    *,
    top_k: int,
    safety_prefix: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if top_k <= 0:
        return [], {"top_k": top_k, "selected": 0}
    unique_base: list[dict[str, Any]] = []
    base_ids: set[str] = set()
    for source in base_rows:
        identity = _node_id(source)
        if identity and identity not in base_ids:
            base_ids.add(identity)
            unique_base.append(dict(source))

    base_by_id = {_node_id(row): row for row in unique_base}
    prefix = unique_base[: min(safety_prefix, top_k)]
    prefix_ids = {_node_id(row) for row in prefix}

    # A selected structural row consumes one of the bounded non-prefix slots
    # even when it was already present near the tail of ``base_rows``.  This is
    # important: otherwise an evidence row at rank 29 can be silently evicted
    # when a new expansion is inserted ahead of it.
    candidates: list[Expansion] = []
    candidate_ids: set[str] = set()
    for anchor in anchors:
        identity = _node_id(anchor)
        if identity and identity not in prefix_ids and identity not in candidate_ids:
            candidates.append(Expansion(dict(anchor), None, "semantic_anchor", 10.0))
            candidate_ids.add(identity)
    for expansion in list(relation) + list(state):
        identity = _node_id(expansion.row)
        if identity and identity not in prefix_ids and identity not in candidate_ids:
            candidates.append(expansion)
            candidate_ids.add(identity)

    slot_budget = max(top_k - len(prefix), 0)
    selected = candidates[:slot_budget]
    selected_ids = {_node_id(item.row) for item in selected}
    filler = [
        row for row in unique_base[len(prefix):]
        if _node_id(row) not in selected_ids
    ][: max(slot_budget - len(selected), 0)]

    # Materialize the base row when present so all frozen raw fields survive;
    # attach only the v2 trace carried by the structural candidate.
    selected_rows: dict[str, dict[str, Any]] = {}
    for expansion in selected:
        identity = _node_id(expansion.row)
        row = dict(base_by_id.get(identity, expansion.row))
        if expansion.row.get("audio_retrieval_v2"):
            row["audio_retrieval_v2"] = dict(expansion.row["audio_retrieval_v2"])
        selected_rows[identity] = row

    expansions_by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tail: list[dict[str, Any]] = []
    for expansion in selected:
        row = selected_rows[_node_id(expansion.row)]
        if expansion.anchor_node_id:
            expansions_by_anchor[expansion.anchor_node_id].append(row)
        else:
            tail.append(row)
    packed: list[dict[str, Any]] = []
    emitted: set[str] = set()
    for row in prefix:
        identity = _node_id(row)
        if identity not in emitted:
            packed.append(row)
            emitted.add(identity)
        for neighbor in expansions_by_anchor.get(identity, []):
            neighbor_id = _node_id(neighbor)
            if neighbor_id not in emitted:
                packed.append(neighbor)
                emitted.add(neighbor_id)
    # Anchors selected from either the old tail or raw-global are emitted
    # before their packet.  Existing structural rows are protected, not lost.
    for anchor in anchors:
        identity = _node_id(anchor)
        if identity in selected_ids and identity not in emitted:
            packed.append(selected_rows[identity])
            emitted.add(identity)
            for neighbor in expansions_by_anchor.get(identity, []):
                neighbor_id = _node_id(neighbor)
                if neighbor_id not in emitted:
                    packed.append(neighbor)
                    emitted.add(neighbor_id)
    for row in tail:
        identity = _node_id(row)
        if identity not in emitted:
            packed.append(row)
            emitted.add(identity)
    # A relation whose anchor was already selected but emitted through another
    # path must still be included before generic relevance filler.
    for rows in expansions_by_anchor.values():
        for row in rows:
            identity = _node_id(row)
            if identity not in emitted:
                packed.append(row)
                emitted.add(identity)
    for row in filler:
        if len(packed) >= top_k:
            break
        identity = _node_id(row)
        if identity not in emitted:
            packed.append(row)
            emitted.add(identity)
    packed = packed[:top_k]
    return packed, {
        "top_k": top_k,
        "selected": len(packed),
        "safety_prefix_requested": safety_prefix,
        "base_prefix_kept": len(prefix),
        "base_count": len(unique_base),
        "structural_slot_count": len(selected),
        "new_membership_count": sum(
            _node_id(item.row) not in base_ids for item in selected
        ),
        "protected_base_tail_count": sum(
            _node_id(item.row) in base_ids for item in selected
        ),
        "anchor_ids": [_node_id(row) for row in anchors],
        "relation_additions": [
            {"node_id": _node_id(item.row), "anchor_node_id": item.anchor_node_id, "reason": item.reason}
            for item in selected if item.anchor_node_id
        ],
        "profile_state_additions": [
            {"node_id": _node_id(item.row), "reason": item.reason}
            for item in selected if item.anchor_node_id is None and item.reason != "semantic_anchor"
        ],
        "uses_question": True,
        "uses_query_identity": bool(state),
        "uses_benchmark_label": False,
        "uses_answer": False,
        "uses_gold": False,
        "raw_only_output": all(str(row.get("layer") or "raw") == "raw" for row in packed),
    }


def relation_profile_context(
    base_context: Sequence[Mapping[str, Any]],
    raw_global: Sequence[Mapping[str, Any]],
    memory: Mapping[str, Any],
    question: str,
    *,
    query_identity: Mapping[str, Any] | None = None,
    ranked_facts: Sequence[Mapping[str, Any]] = (),
    top_k: int = TOP_K,
    safety_prefix: int = RAW_SAFETY_PREFIX,
    relation_budget: int = MAX_RELATION_ROWS,
    profile_state_budget: int = MAX_PROFILE_STATE_ROWS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return raw Top-K with gated local relation/profile completion.

    High-confidence query identity may select asker-owned or assigned-to state
    facts.  Low confidence changes order only, and unresolved identity has no
    effect.  In every case a raw relevance prefix is retained.
    """

    if top_k != TOP_K:
        raise RetrievalContractError("AudioMem v2 is frozen to TopK=30")
    if not 0 <= safety_prefix <= top_k:
        raise RetrievalContractError("invalid raw safety prefix")
    raw_by_ref = validate_memory(memory)
    base_ids = [_node_id(row, f"row:{index}") for index, row in enumerate(base_context)]
    if len(base_ids) != len(set(base_ids)):
        raise RetrievalContractError("base context contains duplicate rows")
    anchors = select_semantic_anchors(question, raw_global)
    relation = _relation_expansions(
        anchors, memory, raw_by_ref, max_rows=max(0, relation_budget)
    )
    state = _profile_state_expansions(
        question,
        memory,
        raw_by_ref,
        query_identity,
        ranked_facts,
        raw_global,
        max_rows=max(0, profile_state_budget),
    )
    packed, trace = _pack_with_anchors(
        base_context,
        anchors,
        relation,
        state,
        top_k=top_k,
        safety_prefix=safety_prefix,
    )
    stable, confidence = _identity_fields(query_identity)
    if confidence == "low" and stable:
        before = {_node_id(row) for row in packed}
        packed = _soft_identity_reorder(packed, question, memory, stable)
        if {_node_id(row) for row in packed} != before:
            raise AssertionError("low-confidence soft boost changed membership")
        trace["low_confidence_soft_reorder"] = True
    else:
        trace["low_confidence_soft_reorder"] = False
    trace.update({
        "schema_version": "voxpoly-audio-retrieval-adapter.v2",
        "query_identity_confidence": confidence,
        "stable_asker_ref": stable,
        "relation_budget": relation_budget,
        "profile_state_budget": profile_state_budget,
    })
    if len(packed) > top_k or len({_node_id(row) for row in packed}) != len(packed):
        raise AssertionError("v2 output violates unique Top-30")
    if any(len(_refs(row)) != 1 for row in packed):
        raise AssertionError("v2 output must retain self-grounded raw refer_ids")
    return packed, trace
