"""Pure, order-only soft identity reranking for an existing Top-30 context.

The intervention never changes the candidate set. It consumes only the
planner's ``speaker_hint`` and structured identity relations stored in memory;
question text, benchmark labels, answers, and gold evidence are not inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


SOURCE_SPEAKER_BOOST = 0.020
ADDRESSEE_BOOST = 0.010
MAX_IDENTITY_BOOST = 0.025


def _key(value: Any) -> str:
    return " ".join(str(value or "").strip().lstrip("@").casefold().split())


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [] if value is None else [value]


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        normalized = _key(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(text)
    return output


@dataclass(frozen=True)
class HintResolution:
    speaker_refs: tuple[str, ...]
    source: str
    raw_hints: tuple[str, ...]


def resolve_speaker_hint(
    speaker_hint: Any,
    profiles: list[dict[str, Any]],
) -> HintResolution:
    """Resolve exact or unique first-token aliases; ambiguous hints stay inactive."""

    hints = _ordered_unique(_as_list(speaker_hint))
    if not hints:
        return HintResolution((), "no_hint", ())

    exact: dict[str, set[str]] = {}
    first_tokens: dict[str, set[str]] = {}
    for profile in profiles:
        stable = str(
            profile.get("stable_speaker_id")
            or profile.get("speaker_ref")
            or profile.get("node_id", "").removeprefix("profile:")
        ).strip()
        if not stable:
            continue
        aliases = _ordered_unique([
            stable,
            profile.get("speaker_name"),
            *(_as_list(profile.get("aliases"))),
        ])
        for alias in aliases:
            exact.setdefault(_key(alias), set()).add(stable)
            tokens = _key(alias).split()
            if tokens:
                first_tokens.setdefault(tokens[0], set()).add(stable)

    resolved: list[str] = []
    used_first_token = False
    for hint in hints:
        key = _key(hint)
        candidates = exact.get(key, set())
        if len(candidates) == 1:
            stable = next(iter(candidates))
        elif len(key.split()) == 1 and len(first_tokens.get(key, set())) == 1:
            stable = next(iter(first_tokens[key]))
            used_first_token = True
        else:
            continue
        if stable not in resolved:
            resolved.append(stable)
    if not resolved:
        return HintResolution((), "unresolved_or_ambiguous", tuple(hints))
    return HintResolution(
        tuple(resolved),
        "unique_alias_token" if used_first_token else "exact_profile_alias",
        tuple(hints),
    )


def _relation_hits_by_ref(
    facts: list[dict[str, Any]], target_refs: set[str]
) -> dict[str, tuple[bool, bool]]:
    """Return (source-speaker hit, addressee hit) for each grounded raw ref."""

    hits: dict[str, list[bool]] = {}
    for fact in facts:
        refs = _ordered_unique(fact.get("refer_ids") or [])
        if not refs:
            continue
        source_ref = str(
            fact.get("source_speaker_ref") or fact.get("speaker_ref") or ""
        ).strip()
        addressee_refs = {
            value for value in _ordered_unique(fact.get("addressee_refs") or [])
            if value not in {"group", "unknown"}
        }
        source_hit = source_ref in target_refs
        addressee_hit = bool(addressee_refs & target_refs)
        if not source_hit and not addressee_hit:
            continue
        for ref in refs:
            row = hits.setdefault(ref, [False, False])
            row[0] = row[0] or source_hit
            row[1] = row[1] or addressee_hit
    return {key: (value[0], value[1]) for key, value in hits.items()}


def soft_identity_rerank(
    context: list[dict[str, Any]],
    plan: dict[str, Any],
    memory: dict[str, Any],
    *,
    enabled: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reorder an existing context with a capped identity boost.

    Candidate membership is invariant by construction. This preserves the
    exact raw safety-recall set produced by the shared retrieval pipeline.
    """

    node_ids = [
        str(row.get("node_id") or row.get("mem_id") or f"row:{index}")
        for index, row in enumerate(context)
    ]
    base_trace = {
        "enabled": bool(enabled),
        "candidate_count": len(context),
        "candidate_set_preserved": True,
        "uses_question_text": False,
        "uses_benchmark_label": False,
        "uses_answer": False,
        "uses_gold": False,
        "source_speaker_boost": SOURCE_SPEAKER_BOOST,
        "addressee_boost": ADDRESSEE_BOOST,
        "max_identity_boost": MAX_IDENTITY_BOOST,
    }
    if not enabled or len(context) < 2:
        return list(context), {
            **base_trace,
            "hint_resolution": "feature_off" if not enabled else "too_few_candidates",
            "resolved_speaker_refs": [],
            "changed_positions": 0,
            "boosted_candidates": 0,
        }

    resolution = resolve_speaker_hint(plan.get("speaker_hint"), memory.get("profiles") or [])
    if not resolution.speaker_refs:
        return list(context), {
            **base_trace,
            "hint_resolution": resolution.source,
            "raw_speaker_hints": list(resolution.raw_hints),
            "resolved_speaker_refs": [],
            "changed_positions": 0,
            "boosted_candidates": 0,
        }

    hits = _relation_hits_by_ref(memory.get("facts") or [], set(resolution.speaker_refs))
    n = len(context)
    scored: list[tuple[float, int, dict[str, Any], float, list[str]]] = []
    for rank, row in enumerate(context):
        source_hit = False
        addressee_hit = False
        for refer_id in _ordered_unique(row.get("refer_ids") or []):
            relation = hits.get(refer_id, (False, False))
            source_hit = source_hit or relation[0]
            addressee_hit = addressee_hit or relation[1]
        boost = min(
            MAX_IDENTITY_BOOST,
            SOURCE_SPEAKER_BOOST * int(source_hit)
            + ADDRESSEE_BOOST * int(addressee_hit),
        )
        reasons = []
        if source_hit:
            reasons.append("source_speaker_ref")
        if addressee_hit:
            reasons.append("addressee_refs")
        # Base relevance spans only 0.25 over the complete list; the capped
        # boost can move a candidate a few ranks but cannot overwhelm content.
        base_score = 1.0 - rank / max(4 * n, 1)
        scored.append((base_score + boost, rank, row, boost, reasons))
    scored.sort(key=lambda value: (-value[0], value[1]))
    reranked = [value[2] for value in scored]
    reranked_ids = [
        str(row.get("node_id") or row.get("mem_id") or f"row:{index}")
        for index, row in enumerate(reranked)
    ]
    if set(reranked_ids) != set(node_ids) or len(reranked_ids) != len(node_ids):
        raise RuntimeError("soft identity reranking changed the candidate set")
    details = [
        {
            "node_id": str(value[2].get("node_id") or value[2].get("mem_id") or ""),
            "original_rank": value[1],
            "new_rank": new_rank,
            "boost": value[3],
            "reasons": value[4],
        }
        for new_rank, value in enumerate(scored)
        if value[3] > 0
    ]
    return reranked, {
        **base_trace,
        "hint_resolution": resolution.source,
        "raw_speaker_hints": list(resolution.raw_hints),
        "resolved_speaker_refs": list(resolution.speaker_refs),
        "changed_positions": sum(a != b for a, b in zip(node_ids, reranked_ids)),
        "boosted_candidates": len(details),
        "boost_details": details,
    }

