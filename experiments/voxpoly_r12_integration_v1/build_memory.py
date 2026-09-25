#!/usr/bin/env python3
"""Build shared r12 raw/fact/collection memory for a QA-free VoxPoly view.

This experiment is default-off and writes only to an explicitly selected,
isolated output.  The LLM selects fact semantics and provenance; deterministic
shared core code grounds identities to the cited bottom turns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from adapter import (  # noqa: E402
    ADAPTER_VERSION,
    AdapterContractError,
    CaseBundle,
    load_case_bundle,
)
from core.identity_binding import (  # noqa: E402
    bind_addressee_to_candidates,
    bind_speaker_from_references,
)


PROMPT_VERSION = "voxpoly-r12-contextual-facts-v1"
WINDOW_SIZE = 12
STRIDE = 6
VALID_OPERATIONS = {
    "coreference", "relative_time", "reply_resolution", "state_transition",
    "multi_turn_answer", "speaker_attribution", "multi_turn_synthesis",
}
SYSTEM = (
    "You extract grounded atomic memory from a multi-speaker dialogue. "
    "Return strict JSON only. Preserve attribution and never infer unsupported facts."
)
USER_PROMPT = """Session date: {date}
Known predicted speaker profiles in this window:
{profiles}

Dialogue window:
{dialogue}

Extract durable, self-contained atomic facts. Use adjacent turns only when
needed to resolve meaning; never merge unrelated statements merely because
they are nearby or share a topic.

Create a multi-turn fact only for coreference, relative_time,
reply_resolution, state_transition, multi_turn_answer, or
speaker_attribution. Preserve exact entities, dates, quantities, negation,
uncertainty, decisions, and changes of state.

Rules:
1. refer_ids must be nonempty and copied only from raw IDs printed above. The
   model selects every supporting turn; cite no irrelevant turn.
2. source_speaker_ref is the principal asserter and must be one printed stable
   ID belonging to at least one cited turn.
3. addressee_refs must be a JSON list containing only printed stable IDs,
   ["group"], or ["unknown"]. Infer it from the observable dialogue only. Do
   not force a person when the addressee is unclear.
4. Multiple refer_ids require at least one context_operation.
5. claimant is optional and differs from the source speaker only for a
   reported belief or statement.
6. Images may be used only through printed image IDs and captions.
7. Skip greetings, pure questions, and unsupported speculation.

Return exactly:
{{"facts":[{{
  "text":"self-contained fact",
  "subject":"explicit entity",
  "predicate":"normalized relation",
  "object":"value or event",
  "fact_type":"event|state|preference|relationship|plan|belief|decision",
  "source_speaker_ref":"stable ID",
  "addressee_refs":["stable ID|group|unknown"],
  "claimant":null,
  "participants":["explicit entity"],
  "valid_time":null,
  "original_time_expression":null,
  "context_operations":["coreference"],
  "refer_ids":["S1_T001"],
  "image_ids":[]
}}]}}
"""


class MemoryContractError(ValueError):
    """Raised when an extraction cannot be grounded to the shared schema."""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def ordered_unique(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    return list(dict.fromkeys(
        str(value).strip() for value in (values or []) if str(value).strip()
    ))


def make_windows(rows: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    if not rows:
        return []
    starts = list(range(0, max(len(rows) - WINDOW_SIZE + 1, 1), STRIDE))
    final_start = max(0, len(rows) - WINDOW_SIZE)
    if final_start not in starts:
        starts.append(final_start)
    return [(start, rows[start:start + WINDOW_SIZE]) for start in sorted(set(starts))]


def render_window(rows: list[dict[str, Any]]) -> tuple[str, str]:
    profile_map: dict[str, str] = {}
    blocks = []
    for row in rows:
        profile_map.setdefault(row["speaker_ref"], row["speaker"])
        block = (
            f"[{row['refer_ids'][0]}|speaker_ref={row['speaker_ref']}] "
            f"{row['speaker']}: {row['text']}"
        )
        for image_id in row.get("image_ids") or []:
            caption = (row.get("image_captions") or {}).get(image_id, "")
            block += f"\n  [image_id={image_id}; caption={caption or 'unknown'}]"
        blocks.append(block)
    profiles = "\n".join(
        f"- {speaker_ref}: {profile_map[speaker_ref]}" for speaker_ref in sorted(profile_map)
    )
    return "\n".join(blocks), profiles


def extraction_messages(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    dialogue, profiles = render_window(rows)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_PROMPT.format(
            date=str(rows[0].get("date") or "unknown"),
            profiles=profiles,
            dialogue=dialogue,
        )},
    ]


def _bind_addressees(
    proposed: Any,
    *,
    candidates: list[str],
) -> tuple[list[str], list[str]]:
    proposals = proposed if isinstance(proposed, list) else [proposed]
    bound: list[str] = []
    sources: list[str] = []
    for value in proposals or ["unknown"]:
        result = bind_addressee_to_candidates(value, candidates)
        sources.append(result.source)
        if result.value == "group":
            return ["group"], sources
        if result.value != "unknown" and result.value not in bound:
            bound.append(result.value)
    return (bound or ["unknown"]), sources


def normalize_fact(
    item: dict[str, Any],
    *,
    rows_by_ref: dict[str, dict[str, Any]],
    valid_image_ids: set[str],
) -> dict[str, Any]:
    text = " ".join(str(item.get("text") or "").split())
    refs = ordered_unique(item.get("refer_ids"))
    if not text or not refs or any(ref not in rows_by_ref for ref in refs):
        raise MemoryContractError(f"invalid fact text/refer_ids: {refs!r}")
    sessions = {str(rows_by_ref[ref]["session_id"]) for ref in refs}
    if len(sessions) != 1:
        raise MemoryContractError("a contextual fact cannot cite multiple sessions")

    proposed_speaker = (
        item.get("source_speaker_ref")
        or item.get("speaker_ref")
        or item.get("speaker")
    )
    try:
        speaker_binding = bind_speaker_from_references(
            proposed_speaker,
            refs,
            rows_by_ref,
            speaker_field="speaker_ref",
        )
    except ValueError as exc:
        raise MemoryContractError(str(exc)) from exc
    speaker_ref = speaker_binding.value
    speaker_display = next(
        row["speaker"] for row in rows_by_ref.values()
        if row["speaker_ref"] == speaker_ref
    )
    evidence_speaker_refs = ordered_unique(
        rows_by_ref[ref]["speaker_ref"] for ref in refs
    )
    addressee_candidates = ordered_unique(
        row["speaker_ref"] for row in rows_by_ref.values()
        if row["speaker_ref"] != speaker_ref
    )
    addressee_refs, addressee_sources = _bind_addressees(
        item.get("addressee_refs", ["unknown"]), candidates=addressee_candidates
    )
    display_by_ref = {
        row["speaker_ref"]: row["speaker"] for row in rows_by_ref.values()
    }
    addressee_display = (
        addressee_refs[0]
        if addressee_refs[0] in {"unknown", "group"}
        else ", ".join(display_by_ref[value] for value in addressee_refs)
    )

    operations = ordered_unique(item.get("context_operations"))
    unknown_operations = set(operations) - VALID_OPERATIONS
    if unknown_operations:
        raise MemoryContractError(f"unknown context operations: {sorted(unknown_operations)}")
    if len(refs) > 1 and not operations:
        operations = ["multi_turn_synthesis"]
    image_ids = ordered_unique(item.get("image_ids"))
    if any(image_id not in valid_image_ids for image_id in image_ids):
        raise MemoryContractError(f"invalid image IDs: {image_ids!r}")

    first = rows_by_ref[refs[0]]
    normalized = {
        "layer": "fact",
        "text": text,
        # Match the best r12 setting: identity remains structured metadata and
        # does not pollute generic dense/BM25 content retrieval.
        "retrieval_text": text,
        "subject": str(item.get("subject") or "").strip(),
        "predicate": str(item.get("predicate") or "").strip(),
        "object": str(item.get("object") or "").strip(),
        "fact_type": str(item.get("fact_type") or "fact").strip(),
        "speaker": speaker_display,
        "source_speaker_ref": speaker_ref,
        # Compatibility alias for callers that use the shorter shared name.
        "speaker_ref": speaker_ref,
        "speaker_binding_source": speaker_binding.source,
        "evidence_speaker_refs": evidence_speaker_refs,
        "addressee": addressee_display,
        "addressee_refs": addressee_refs,
        "addressee_binding_sources": addressee_sources,
        "claimant": str(item.get("claimant") or "").strip() or None,
        "participants": ordered_unique(item.get("participants")),
        "valid_time": str(item.get("valid_time") or "").strip() or None,
        "original_time_expression": str(
            item.get("original_time_expression") or ""
        ).strip() or None,
        "context_operations": operations,
        "refer_ids": refs,
        "image_ids": image_ids,
        "session_id": first["session_id"],
        "date": first.get("date", ""),
    }
    signature = json.dumps(
        {
            "speaker_ref": speaker_ref,
            "text": re.sub(r"\W+", " ", text.casefold()).strip(),
            "refer_ids": refs,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    normalized["node_id"] = "fact:" + hashlib.sha1(
        signature.encode("utf-8")
    ).hexdigest()[:16]
    return normalized


def deduplicate_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for fact in facts:
        key = (
            fact["source_speaker_ref"],
            re.sub(r"\W+", " ", fact["text"].casefold()).strip(),
        )
        old = selected.get(key)
        rank = (len(fact["refer_ids"]), len(fact["context_operations"]), fact["node_id"])
        old_rank = (-1, -1, "") if old is None else (
            len(old["refer_ids"]), len(old["context_operations"]), old["node_id"]
        )
        if old is None or rank > old_rank:
            selected[key] = fact
    return sorted(selected.values(), key=lambda row: row["node_id"])


def build_collections(
    facts: list[dict[str, Any]], raw: list[dict[str, Any]], case_name: str
) -> list[dict[str, Any]]:
    raw_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        raw_by_session[row["session_id"]].append(row)
    collections = []
    for session_id, source_rows in raw_by_session.items():
        members = [fact for fact in facts if fact["session_id"] == session_id]
        refs = ordered_unique(ref for fact in members for ref in fact["refer_ids"])
        if not members or not refs:
            continue
        participant_refs = ordered_unique(
            value
            for fact in members
            for value in (
                fact["evidence_speaker_refs"]
                + [ref for ref in fact["addressee_refs"] if ref not in {"group", "unknown"}]
            )
        )
        collections.append({
            "node_id": f"collection:{case_name}:{session_id}",
            "layer": "collection",
            "text": f"Grounded event collection for session {session_id}.",
            "retrieval_text": " ".join(fact["text"] for fact in members),
            "session_id": session_id,
            "date": source_rows[0].get("date", ""),
            "member_fact_ids": [fact["node_id"] for fact in members],
            "participant_refs": participant_refs,
            "refer_ids": refs,
        })
    return collections


def _fingerprint(bundle: CaseBundle, model: str) -> dict[str, Any]:
    return {
        "adapter_version": ADAPTER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "source_view_sha256": bundle.source_view_sha256,
        "speaker_sidecar_sha256": bundle.sidecar_sha256,
        "identity_prediction_sha256": bundle.prediction_sha256,
        "identity_prediction_protocol": bundle.prediction_protocol,
        "registry_state_sha256": bundle.registry_state_sha256,
        "model": model,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "role_metadata_in_retrieval_text": False,
        "identity_retrieval_enabled": False,
    }


def _assert_referential_integrity(memory: dict[str, Any]) -> None:
    bottom = {row["refer_ids"][0] for row in memory["raw"]}
    fact_ids = {row["node_id"] for row in memory["facts"]}
    if len(bottom) != len(memory["raw"]):
        raise MemoryContractError("raw bottom provenance IDs are not unique")
    for row in memory["raw"]:
        reply_to_turn_id = row.get("reply_to_turn_id")
        if reply_to_turn_id is not None and reply_to_turn_id not in bottom:
            raise MemoryContractError(
                f"raw node replies to an unknown bottom turn: {reply_to_turn_id}"
            )
    for layer in ("facts", "collections", "profiles"):
        for row in memory[layer]:
            refs = row.get("refer_ids")
            if not isinstance(refs, list) or not refs or not set(refs).issubset(bottom):
                raise MemoryContractError(f"{layer} node has invalid refer_ids: {row}")
    for row in memory["collections"]:
        if not set(row.get("member_fact_ids") or []).issubset(fact_ids):
            raise MemoryContractError("collection references an unknown fact")


Extractor = Callable[[list[dict[str, str]]], dict[str, Any]]


def build_case(
    *,
    case_view_dir: Path,
    registry_state: Path,
    output_path: Path,
    checkpoint_dir: Path,
    model: str,
    extractor: Extractor,
    enabled: bool = False,
    force: bool = False,
    allow_existing_output: bool = False,
    require_strict_causal_aliases: bool = False,
    max_attempts: int = 4,
) -> dict[str, Any]:
    if not 1 <= max_attempts <= 8:
        raise MemoryContractError("max_attempts must be between 1 and 8")
    bundle = load_case_bundle(
        case_view_dir,
        registry_state,
        enabled=enabled,
        require_strict_causal_aliases=require_strict_causal_aliases,
    )
    output_path = output_path.expanduser().resolve(strict=False)
    checkpoint_dir = checkpoint_dir.expanduser().resolve(strict=False)
    if output_path.exists() and not allow_existing_output:
        raise MemoryContractError(f"refusing to overwrite existing output: {output_path}")
    fingerprint = _fingerprint(bundle, model)
    raw = [dict(row) for row in bundle.raw]
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        by_session[row["session_id"]].append(row)

    all_facts: list[dict[str, Any]] = []
    calls = 0
    for session_id, rows in by_session.items():
        rows.sort(key=lambda row: (row["ordinal"], row["refer_ids"][0]))
        for start, window in make_windows(rows):
            window_by_ref = {row["refer_ids"][0]: row for row in window}
            valid_images = {
                image_id for row in window for image_id in row.get("image_ids") or []
            }
            checkpoint = checkpoint_dir / session_id / f"{start}.json"
            if checkpoint.exists() and not force:
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if saved.get("fingerprint") != fingerprint:
                    raise MemoryContractError(f"checkpoint fingerprint mismatch: {checkpoint}")
                facts = saved.get("facts")
                if not isinstance(facts, list) or not facts or any(
                    not set(fact.get("refer_ids") or []).issubset(window_by_ref)
                    for fact in facts
                ):
                    raise MemoryContractError(f"invalid checkpoint contents: {checkpoint}")
                all_facts.extend(facts)
                continue

            last_error: Exception | None = None
            facts: list[dict[str, Any]] = []
            messages = extraction_messages(window)
            for _attempt in range(1, max_attempts + 1):
                calls += 1
                try:
                    parsed = extractor(messages)
                    if not isinstance(parsed, dict):
                        raise MemoryContractError("extractor output root must be an object")
                    raw_facts = parsed.get("facts", [])
                    if not isinstance(raw_facts, list) or any(
                        not isinstance(item, dict) for item in raw_facts
                    ):
                        raise MemoryContractError("facts must be a list of objects")
                    facts = [
                        normalize_fact(
                            item,
                            rows_by_ref=window_by_ref,
                            valid_image_ids=valid_images,
                        )
                        for item in raw_facts
                    ]
                    if not facts:
                        raise MemoryContractError("extractor returned no grounded facts")
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    messages = extraction_messages(window)
                    messages[-1]["content"] += (
                        "\nYour previous output failed grounding validation: "
                        f"{exc}. Regenerate the complete JSON for this same window."
                    )
            if last_error is not None:
                raise MemoryContractError(
                    f"failed grounding {bundle.case_name}/{session_id}@{start}: {last_error}"
                ) from last_error
            atomic_json(checkpoint, {
                "fingerprint": fingerprint,
                "case_id": bundle.case_id,
                "session_id": session_id,
                "window_start": start,
                "refer_ids": list(window_by_ref),
                "facts": facts,
            })
            all_facts.extend(facts)

    facts = deduplicate_facts(all_facts)
    collections = build_collections(facts, raw, bundle.case_name)
    profiles = [dict(row) for row in bundle.profiles]
    memory = {
        "schema_version": PROMPT_VERSION,
        "case": bundle.case_name,
        "case_id": bundle.case_id,
        "raw": raw,
        "facts": facts,
        "collections": collections,
        "profiles": profiles,
        "meta": {
            "fingerprint": fingerprint,
            "turns": len(raw),
            "facts": len(facts),
            "cross_turn_facts": sum(len(row["refer_ids"]) > 1 for row in facts),
            "collections": len(collections),
            "profiles": len(profiles),
            "complete_windows": sum(len(make_windows(rows)) for rows in by_session.values()),
            "extractor_calls_this_run": calls,
            "qa_consumed": False,
            "gold_evidence_consumed": False,
            "gt_speaker_consumed": False,
            "gt_addressee_consumed": False,
            "identity_retrieval_enabled": False,
            "profile_retrieval_enabled": False,
            "alias_timing": (
                "strict_turn_causal" if bundle.alias_is_causal_at_each_turn else "batch_final"
            ),
        },
    }
    _assert_referential_integrity(memory)
    atomic_json(output_path, memory)
    return memory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-experimental-audio", action="store_true")
    parser.add_argument("--case-view-dir", required=True, type=Path)
    parser.add_argument("--registry-state", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume-output", action="store_true")
    parser.add_argument("--require-strict-causal-aliases", action="store_true")
    parser.add_argument("--max-attempts", type=int, choices=range(1, 9), default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.enable_experimental_audio:
        print(
            "VoxPoly r12 integration is default-off; pass --enable-experimental-audio",
            file=sys.stderr,
        )
        return 2
    v2_root = Path(os.environ.get(
        "V2_PIPELINE_ROOT",
        WORK_ROOT / "vendor/v2_update_pipeline",
    ))
    sys.path.insert(0, str(v2_root))
    from utils.llm_client import call_llm, extract_json, print_usage_summary

    def extractor(messages: list[dict[str, str]]) -> dict[str, Any]:
        response = call_llm(
            messages,
            model=args.model,
            temperature=0,
            max_tokens=6000,
            response_format={"type": "json_object"},
        )
        return extract_json(response)

    try:
        memory = build_case(
            case_view_dir=args.case_view_dir,
            registry_state=args.registry_state,
            output_path=args.out,
            checkpoint_dir=args.checkpoints,
            model=args.model,
            extractor=extractor,
            enabled=True,
            force=args.force,
            allow_existing_output=args.resume_output,
            require_strict_causal_aliases=args.require_strict_causal_aliases,
            max_attempts=args.max_attempts,
        )
    except (AdapterContractError, MemoryContractError, OSError) as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.out), **memory["meta"]}, ensure_ascii=False))
    print_usage_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
