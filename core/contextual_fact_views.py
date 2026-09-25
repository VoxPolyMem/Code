"""Validation and zero-call ablation views for contextual atomic facts."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable


CONTEXT_OPERATIONS = {
    "coreference",
    "relative_time",
    "reply_resolution",
    "state_transition",
    "multi_turn_answer",
    "speaker_attribution",
    "multi_turn_synthesis",
}


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _canonical_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in {"user", "human", "character", "participant"}:
        return "user"
    if role in {"assistant", "agent", "ai", "chatbot"}:
        return "assistant"
    raise ValueError(f"invalid source_role: {value!r}")


def normalize_fact(
    item: dict[str, Any],
    *,
    character: str,
    valid_refer_ids: set[str],
    valid_image_ids: set[str],
) -> dict[str, Any]:
    """Normalize one model-emitted fact without inventing provenance."""

    text = " ".join(str(item.get("text") or "").split())
    if not text:
        raise ValueError("fact text is empty")

    refer_ids = _ordered_unique(item.get("refer_ids") or [])
    if not refer_ids or any(ref not in valid_refer_ids for ref in refer_ids):
        raise ValueError(f"invalid refer_ids: {refer_ids!r}")

    image_ids = _ordered_unique(item.get("image_ids") or [])
    if any(image_id not in valid_image_ids for image_id in image_ids):
        raise ValueError(f"invalid image_ids: {image_ids!r}")

    source_role = _canonical_role(item.get("source_role"))
    speaker = character if source_role == "user" else "assistant"
    addressee = "assistant" if source_role == "user" else character

    operations = _ordered_unique(item.get("context_operations") or [])
    unknown = set(operations) - CONTEXT_OPERATIONS
    if unknown:
        raise ValueError(f"unknown context operations: {sorted(unknown)}")
    if len(refer_ids) > 1 and not operations:
        operations = ["multi_turn_synthesis"]

    normalized = {
        "layer": "fact",
        "text": text,
        "retrieval_text": f"speaker={speaker}; addressee={addressee}; {text}",
        "subject": str(item.get("subject") or "").strip(),
        "predicate": str(item.get("predicate") or "").strip(),
        "object": str(item.get("object") or "").strip(),
        "fact_type": str(item.get("fact_type") or "fact").strip(),
        "speaker": speaker,
        "addressee": addressee,
        "claimant": str(item.get("claimant") or "").strip() or None,
        "participants": _ordered_unique(item.get("participants") or []),
        "valid_time": str(item.get("valid_time") or "").strip() or None,
        "original_time_expression": str(
            item.get("original_time_expression") or ""
        ).strip() or None,
        "context_operations": operations,
        "refer_ids": refer_ids,
        "image_ids": image_ids,
    }
    signature = json.dumps(
        {
            "speaker": speaker,
            "text": re.sub(r"\W+", " ", text.lower()).strip(),
            "refer_ids": refer_ids,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    normalized["node_id"] = "fact:" + hashlib.sha1(
        signature.encode("utf-8")
    ).hexdigest()[:16]
    return normalized


def deduplicate_facts(facts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate overlapping-window paraphrases conservatively.

    Exact normalized speaker/text duplicates keep the version with the richest
    valid provenance. Ties are resolved deterministically.
    """

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for fact in facts:
        key = (
            str(fact["speaker"]),
            re.sub(r"\W+", " ", str(fact["text"]).lower()).strip(),
        )
        old = selected.get(key)
        rank = (
            len(fact.get("refer_ids", [])),
            len(fact.get("context_operations", [])),
            str(fact.get("node_id", "")),
        )
        old_rank = (-1, -1, "") if old is None else (
            len(old.get("refer_ids", [])),
            len(old.get("context_operations", [])),
            str(old.get("node_id", "")),
        )
        if old is None or rank > old_rank:
            selected[key] = fact
    return sorted(selected.values(), key=lambda row: row["node_id"])


def materialize_ablation_view(
    facts: Iterable[dict[str, Any]], view: str
) -> list[dict[str, Any]]:
    """Derive an ablation without another LLM extraction call."""

    rows = [dict(fact) for fact in facts]
    if view == "full":
        return rows
    if view == "no_coreference":
        return [
            row for row in rows
            if "coreference" not in row.get("context_operations", [])
        ]
    if view == "single_turn_only":
        return [row for row in rows if len(row.get("refer_ids", [])) == 1]
    if view == "no_role_metadata":
        for row in rows:
            row["retrieval_text"] = row["text"]
        return rows
    raise ValueError(f"unknown ablation view: {view}")

