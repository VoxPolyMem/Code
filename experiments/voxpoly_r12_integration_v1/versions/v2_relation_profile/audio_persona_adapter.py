"""QA-blind speaker-owned raw retrieval for the optional Audio planner action.

The candidate path is selected by the shared planner, which receives an
optional high-confidence waveform-derived asker and chooses asker_only or
global. This module executes only the former action using profile provenance.
The legacy first-person gate and answer instruction below remain for
reproducible diagnostic ablations; they are not used by the candidate method.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from retrieval_adapter import (
    TOP_K,
    RetrievalContractError,
    _node_id,
    _refs,
    raw_hybrid_topk,
    raw_multichannel_rrf_topk,
    validate_memory,
)


FIRST_PERSON_RE = re.compile(r"\b(?:i|me|my|mine|myself)\b", re.IGNORECASE)
SCHEMA_VERSION = "voxpoly-audio-persona-adapter.v2.2"


def is_audio_persona_query(
    question: str, query_identity: Mapping[str, Any] | None
) -> bool:
    """Legacy static gate used only by the retained diagnostic ablation."""

    return has_high_confidence_identity(query_identity) and bool(
        FIRST_PERSON_RE.search(question or "")
    )


def has_high_confidence_identity(query_identity: Mapping[str, Any] | None) -> bool:
    """Return whether waveform matching resolved one safe, stable asker ID."""

    return bool(
        query_identity
        and query_identity.get("confidence") == "high"
        and str(query_identity.get("stable_asker_ref") or "").strip()
    )


def _profile(memory: Mapping[str, Any], stable: str) -> Mapping[str, Any]:
    matches = [
        row for row in memory.get("profiles") or []
        if str(row.get("stable_speaker_id") or "") == stable
    ]
    if len(matches) != 1:
        raise RetrievalContractError(
            f"expected exactly one profile for high-confidence speaker {stable!r}"
        )
    return matches[0]


def speaker_owned_raw_context(
    *,
    question: str,
    query_identity: Mapping[str, Any],
    memory: Mapping[str, Any],
    dense_raw: Sequence[Mapping[str, Any]],
    bm25_raw: Sequence[Mapping[str, Any]],
    top_k: int = TOP_K,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve at most Top-K raw turns grounded to the identified speaker.

    Dense and BM25 rankings are computed upstream over all raw turns, then
    restricted using the profile's `refer_ids`.  No answer, gold evidence,
    question category, or identity-bearing file name is accepted.
    """

    if top_k != TOP_K:
        raise RetrievalContractError("Audio Persona adapter is frozen to TopK=30")
    if not has_high_confidence_identity(query_identity):
        raise RetrievalContractError("speaker-owned route requires high-confidence identity")
    raw_by_ref = validate_memory(memory)
    stable = str(query_identity["stable_asker_ref"])
    profile = _profile(memory, stable)
    allowed_refs = {str(value) for value in profile.get("refer_ids") or [] if str(value)}
    if not allowed_refs:
        raise RetrievalContractError("identified speaker profile has no raw provenance")
    if not allowed_refs <= set(raw_by_ref):
        raise RetrievalContractError("speaker profile has dangling raw provenance")

    def keep(row: Mapping[str, Any]) -> bool:
        refs = {str(value) for value in row.get("refer_ids") or [] if str(value)}
        return len(refs) == 1 and bool(refs & allowed_refs)

    ranked = raw_hybrid_topk(
        [row for row in dense_raw if keep(row)],
        [row for row in bm25_raw if keep(row)],
        top_k=top_k,
    )
    if not ranked:
        raise RetrievalContractError("speaker-owned raw retrieval returned no evidence")
    if any(not keep(row) for row in ranked):
        raise AssertionError("speaker-owned context contains foreign raw evidence")
    trace = {
        "schema_version": SCHEMA_VERSION,
        "route": "profile_to_speaker_owned_raw_dense_bm25_rrf",
        "top_k_limit": top_k,
        "n_context": len(ranked),
        "query_identity_confidence": "high",
        "stable_asker_ref": stable,
        "profile_node_id": _node_id(profile),
        "profile_grounded_raw_count": len(allowed_refs),
        "uses_benchmark_label": False,
        "uses_answer": False,
        "uses_gold": False,
        "raw_only_output": True,
    }
    return ranked, trace


def speaker_profile_prior_raw_context(
    *,
    question: str,
    query_identity: Mapping[str, Any],
    memory: Mapping[str, Any],
    dense_raw: Sequence[Mapping[str, Any]],
    bm25_raw: Sequence[Mapping[str, Any]],
    top_k: int = TOP_K,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply a soft profile prior while retaining cross-speaker raw evidence.

    The exact same raw dense/BM25 query is viewed twice: globally and after
    profile-provenance projection.  Four-way RRF makes speaker-owned evidence
    more competitive, without discarding a delegation or acknowledgement
    uttered by someone else.  The operation uses no category or answer data.
    """

    if top_k != TOP_K:
        raise RetrievalContractError("Audio Persona adapter is frozen to TopK=30")
    if not has_high_confidence_identity(query_identity):
        raise RetrievalContractError("profile prior requires high-confidence identity")
    raw_by_ref = validate_memory(memory)
    stable = str(query_identity["stable_asker_ref"])
    profile = _profile(memory, stable)
    allowed_refs = {str(value) for value in profile.get("refer_ids") or [] if str(value)}
    if not allowed_refs or not allowed_refs <= set(raw_by_ref):
        raise RetrievalContractError("speaker profile has invalid raw provenance")

    def keep(row: Mapping[str, Any]) -> bool:
        refs = {str(value) for value in row.get("refer_ids") or [] if str(value)}
        return len(refs) == 1 and bool(refs & allowed_refs)

    ranked = raw_multichannel_rrf_topk(
        (
            ("global.dense", dense_raw),
            ("global.bm25", bm25_raw),
            ("profile.dense", [row for row in dense_raw if keep(row)]),
            ("profile.bm25", [row for row in bm25_raw if keep(row)]),
        ),
        top_k=top_k,
    )
    if len(ranked) != top_k:
        raise RetrievalContractError("profile-prior context underfilled Top-K")
    trace = {
        "schema_version": SCHEMA_VERSION,
        "route": "raw_dense_bm25_rrf_with_profile_provenance_prior",
        "top_k_limit": top_k,
        "n_context": len(ranked),
        "query_identity_confidence": "high",
        "stable_asker_ref": stable,
        "profile_node_id": _node_id(profile),
        "profile_grounded_raw_count": len(allowed_refs),
        "uses_benchmark_label": False,
        "uses_answer": False,
        "uses_gold": False,
        "raw_only_output": True,
    }
    return ranked, trace


def annotate_raw_relation_context(
    memory: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Attach grounded addressee labels to raw rows without another model call.

    Raw turns carry a reliable speaker label.  Addressee is stored on grounded
    fact nodes, so a raw turn inherits an addressee only when it is spoken by
    the fact's grounded source speaker and all such fact votes agree. This
    prevents a multi-turn fact's semantic target from being copied onto the
    earlier question that elicited it. Ambiguous or absent evidence remains
    explicitly ``unknown`` rather than fabricating a directed edge.
    """

    labels_by_ref: dict[str, list[tuple[str, tuple[str, ...], str]]] = defaultdict(list)
    for fact in memory.get("facts") or []:
        label = str(fact.get("addressee") or "").strip()
        refs = tuple(str(value) for value in fact.get("addressee_refs") or [] if str(value))
        source_speaker_ref = str(
            fact.get("source_speaker_ref") or fact.get("speaker_ref") or ""
        ).strip()
        if not label or not source_speaker_ref:
            continue
        for refer_id in _refs(fact):
            labels_by_ref[refer_id].append((label, refs, source_speaker_ref))

    annotated = []
    for row in raw_rows:
        copied = dict(row)
        raw_speaker_ref = str(copied.get("speaker_ref") or "").strip()
        candidates = [
            value for refer_id in _refs(copied)
            for value in labels_by_ref.get(refer_id, [])
            if raw_speaker_ref and value[2] == raw_speaker_ref
        ]
        counts = Counter(label for label, _, _ in candidates)
        if len(counts) == 1:
            addressee = next(iter(counts))
            ref_values = next(
                refs for label, refs, _ in candidates if label == addressee
            )
            source = "grounded_fact_source_speaker_unanimous"
        elif not counts:
            addressee, ref_values, source = "unknown", (), "no_grounded_fact"
        else:
            addressee, ref_values, source = "unknown", (), "grounded_fact_ambiguous"
        copied["relation_addressee"] = addressee
        copied["relation_addressee_refs"] = list(ref_values)
        copied["relation_label_source"] = source
        annotated.append(copied)
    return annotated


def answer_messages_with_audio_persona_identity(
    answer_messages: Any,
    *,
    question: str,
    context: Sequence[Mapping[str, Any]],
    character: str,
    last_date: str,
    query_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the historical Audio-only answer-prompt diagnostic.

    It is deliberately not called by the candidate shared-planner path, whose
    answer prompt is identical across text, image, and audio evaluation.
    """

    if not is_audio_persona_query(question, query_identity):
        raise RetrievalContractError("identity prompt requires the Persona runtime gate")
    if not character or character.casefold() == "user":
        raise RetrievalContractError("identity prompt requires a named profile display")
    messages = answer_messages(question, list(context), character=character, last_date=last_date)
    if len(messages) < 2 or not isinstance(messages[1], Mapping):
        raise RetrievalContractError("shared answer prompt has an unexpected message shape")
    content = messages[1].get("content")
    if not isinstance(content, list):
        raise RetrievalContractError("shared answer prompt has non-list user content")
    instruction = (
        f"The person ASKING this question was identified from their voice as \"{character}\". "
        f"All retrieved memories are utterances by \"{character}\". Interpret I/me/my in the "
        f"question as \"{character}\" and answer only from these utterances. Do not attribute "
        "the user's statement to anyone else and do not infer facts absent from these memories.\n"
    )
    content.insert(1, {"type": "text", "text": instruction})
    return messages
