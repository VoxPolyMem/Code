"""Ground role identities in the bottom-level turns selected by the model.

The LLM chooses semantic facts and their ``refer_ids``.  This module performs
only schema-level identity binding: it never adds evidence or guesses a person
that is absent from the cited turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


UNKNOWN_LABELS = frozenset({"", "unknown", "unk", "none", "null", "n/a", "unspecified"})
GROUP_LABELS = frozenset({"group", "everyone", "all", "all participants", "multiple people"})


@dataclass(frozen=True)
class IdentityBinding:
    value: str
    source: str
    candidates: tuple[str, ...]


def _key(value: Any) -> str:
    return " ".join(str(value or "").strip().lstrip("@").casefold().split())


def _ordered_unique(values: Sequence[Any]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = str(value or "").strip()
        key = _key(label)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(label)
    return tuple(output)


def bind_speaker_from_references(
    proposed_speaker: Any,
    refer_ids: Sequence[str],
    rows_by_ref: Mapping[str, Mapping[str, Any]],
    *,
    speaker_field: str = "speaker",
) -> IdentityBinding:
    """Return a speaker that is grounded by every model-selected reference.

    A single cited speaker is deterministic and therefore overrides an LLM
    ``unknown`` or formatting error.  With multiple cited speakers, the model's
    proposal must exactly match one candidate after harmless normalization.
    """

    missing = [ref for ref in refer_ids if ref not in rows_by_ref]
    if missing:
        raise ValueError(f"unknown refer_ids for identity binding: {missing!r}")
    candidates = _ordered_unique(
        [rows_by_ref[ref].get(speaker_field) for ref in refer_ids]
    )
    if not candidates:
        raise ValueError("cited turns contain no speaker identity")
    if len(candidates) == 1:
        return IdentityBinding(candidates[0], "single_cited_speaker", candidates)

    proposed_key = _key(proposed_speaker)
    for candidate in candidates:
        if proposed_key == _key(candidate):
            return IdentityBinding(candidate, "model_grounded_multi_speaker", candidates)
    raise ValueError(
        f"speaker {str(proposed_speaker)!r} is not one of cited speakers {list(candidates)!r}"
    )


def bind_addressee_to_candidates(
    proposed_addressee: Any,
    candidates: Sequence[Any],
) -> IdentityBinding:
    """Fail closed to ``unknown`` unless an addressee is an allowed identity/group."""

    canonical = _ordered_unique(candidates)
    proposed_key = _key(proposed_addressee)
    if proposed_key in GROUP_LABELS:
        return IdentityBinding("group", "model_group", canonical)
    if proposed_key in UNKNOWN_LABELS:
        return IdentityBinding("unknown", "model_unknown", canonical)
    for candidate in canonical:
        if proposed_key == _key(candidate):
            return IdentityBinding(candidate, "model_grounded_candidate", canonical)
    return IdentityBinding("unknown", "unresolved_candidate", canonical)
