"""Build QA-blind contextual atomic memory for H2HMem.

This is the multi-party counterpart of the Mem-Gallery r12 builder.  It reads
only the frozen turn stream and dense image captions from the existing H2H
ingest artifacts; the old extracted facts and benchmark questions are never
opened.  Facts are generated from overlapping within-session windows and keep
model-selected provenance to bottom-level raw nodes.
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
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
V3_ROOT = Path(os.environ.get(
    "V3_PIPELINE_ROOT", PACKAGE_ROOT / "vendor/v3_update_pipeline"
))
SOURCE_ROOT = Path(os.environ.get(
    "H2H_SOURCE_ROOT", PACKAGE_ROOT / "data/h2h_source"
))
DATA_ROOT = Path(os.environ.get(
    "H2HMEM_ROOT", PACKAGE_ROOT / "data/H2HMEM"
))
CAPTION_PATH = Path(os.environ.get(
    "H2HMEM_CAPTIONS", PACKAGE_ROOT / "data/dense_captions_h2hmem.json"
))
DEFAULT_OUT = PACKAGE_ROOT / "artifacts/memory/h2h_contextual_v1/no_role_metadata"

sys.path.insert(0, str(V3_ROOT))
from utils.llm_client import call_llm, extract_json, print_usage_summary  # noqa: E402
from core.identity_binding import bind_speaker_from_references  # noqa: E402


PROMPT_VERSION = "h2h-contextual-atomic-facts-v1"
WINDOW_SIZE = 12
STRIDE = 6
VALID_OPERATIONS = {
    "coreference",
    "relative_time",
    "reply_resolution",
    "state_transition",
    "multi_turn_answer",
    "speaker_attribution",
    "multi_turn_synthesis",
}

SYSTEM = """You extract grounded atomic memory from a multi-party dialogue.
Return strict JSON only. Preserve speaker attribution and never infer unsupported facts."""

USER_PROMPT = """Session date: {date}

Dialogue window:
{dialogue}

Extract durable, self-contained atomic facts. Use adjacent turns only when
needed to resolve meaning; do not merge unrelated statements just because they
are nearby or share a topic.

Create a multi-turn fact only when at least one operation is necessary:
- coreference: resolve it/this/that/he/she/the second one;
- relative_time: resolve expressions such as yesterday or last week;
- reply_resolution: a reply accepts, rejects, modifies, or answers an earlier turn;
- state_transition: a later turn updates, corrects, or contradicts an earlier state;
- multi_turn_answer: a conclusion requires a question and its answer together;
- speaker_attribution: neighboring turns are needed to identify speaker/addressee.

Rules:
1. speaker is the participant who asserted the fact, copied exactly from a
   printed speaker name. addressee is an explicit participant, group, or unknown.
2. refer_ids must be nonempty and use only raw IDs printed in this window. The
   model, not the program, chooses every supporting ID.
3. Cite every turn required to support or resolve the fact and no irrelevant turn.
4. Multiple refer_ids require at least one context_operation.
5. claimant is optional and differs from speaker only for reported beliefs.
6. Preserve names, image IDs, dates, quantities, negation, uncertainty,
   decisions, changes of state, and the distinction between proposal and acceptance.
7. Resolve relative time only when the date and dialogue make it safe.
8. Images may be used only through the printed image_id and caption.
9. Skip greetings, pure questions, and unsupported speculation.

Return exactly:
{{"facts":[{{
  "text":"self-contained fact",
  "subject":"explicit entity",
  "predicate":"normalized relation",
  "object":"value or event",
  "fact_type":"event|state|preference|relationship|plan|belief|decision",
  "speaker":"exact printed participant name",
  "addressee":"participant|group|unknown",
  "claimant":null,
  "participants":["explicit entity"],
  "valid_time":null,
  "original_time_expression":null,
  "context_operations":["coreference"],
  "refer_ids":["raw:dialogue1:session1:0"],
  "image_ids":[]
}}]}}
"""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def ordered_unique(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    return list(dict.fromkeys(str(value).strip() for value in (values or []) if str(value).strip()))


def make_windows(rows: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    if not rows:
        return []
    starts = list(range(0, max(len(rows) - WINDOW_SIZE + 1, 1), STRIDE))
    final_start = max(0, len(rows) - WINDOW_SIZE)
    if final_start not in starts:
        starts.append(final_start)
    return [(start, rows[start:start + WINDOW_SIZE]) for start in sorted(set(starts))]


def strip_speaker_prefix(text: str, speaker: str) -> str:
    return re.sub(rf"^\s*{re.escape(speaker)}\s*:\s*", "", text, count=1).strip()


def render(window: list[dict[str, Any]]) -> str:
    blocks = []
    for row in window:
        block = f"[{row['node_id']}] {row['speaker']}: {row['text']}"
        for image_id, caption in (row.get("image_captions") or {}).items():
            block += f"\n  [image_id={image_id}; caption={caption or 'unknown'}]"
        blocks.append(block)
    return "\n".join(blocks)


def normalize_fact(
    item: dict[str, Any],
    *,
    window_by_ref: dict[str, dict[str, Any]],
    valid_image_ids: set[str],
) -> dict[str, Any]:
    text = " ".join(str(item.get("text") or "").split())
    refs = ordered_unique(item.get("refer_ids"))
    if not text or not refs or any(ref not in window_by_ref for ref in refs):
        raise ValueError(f"invalid text/refer_ids: {refs!r}")

    speaker_binding = bind_speaker_from_references(
        item.get("speaker"), refs, window_by_ref, speaker_field="speaker"
    )
    speaker = speaker_binding.value
    addressee = str(item.get("addressee") or "unknown").strip() or "unknown"
    operations = ordered_unique(item.get("context_operations"))
    if set(operations) - VALID_OPERATIONS:
        raise ValueError(f"unknown context operations: {operations!r}")
    if len(refs) > 1 and not operations:
        operations = ["multi_turn_synthesis"]
    image_ids = ordered_unique(item.get("image_ids"))
    if any(image_id not in valid_image_ids for image_id in image_ids):
        raise ValueError(f"invalid image_ids: {image_ids!r}")

    first = window_by_ref[refs[0]]
    normalized = {
        "layer": "fact",
        "text": text,
        # Match Mem-Gallery's best r12 view: roles remain structured metadata
        # but do not pollute dense/BM25 retrieval text.
        "retrieval_text": text,
        "subject": str(item.get("subject") or "").strip(),
        "predicate": str(item.get("predicate") or "").strip(),
        "object": str(item.get("object") or "").strip(),
        "fact_type": str(item.get("fact_type") or "fact").strip(),
        "speaker": speaker,
        "speaker_binding": speaker_binding.source,
        "addressee": addressee,
        "claimant": str(item.get("claimant") or "").strip() or None,
        "participants": ordered_unique(item.get("participants")),
        "valid_time": str(item.get("valid_time") or "").strip() or None,
        "original_time_expression": str(item.get("original_time_expression") or "").strip() or None,
        "context_operations": operations,
        "refer_ids": refs,
        "image_ids": image_ids,
        "session_id": first["session_id"],
        "date": first["date"],
    }
    signature = json.dumps(
        {"speaker": speaker, "text": re.sub(r"\W+", " ", text.lower()).strip(), "refer_ids": refs},
        ensure_ascii=False,
        sort_keys=True,
    )
    normalized["node_id"] = "fact:" + hashlib.sha1(signature.encode()).hexdigest()[:16]
    return normalized


def deduplicate(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for fact in facts:
        key = (fact["speaker"], re.sub(r"\W+", " ", fact["text"].lower()).strip())
        old = selected.get(key)
        rank = (len(fact["refer_ids"]), len(fact["context_operations"]), fact["node_id"])
        old_rank = (-1, -1, "") if old is None else (
            len(old["refer_ids"]), len(old["context_operations"]), old["node_id"]
        )
        if old is None or rank > old_rank:
            selected[key] = fact
    return sorted(selected.values(), key=lambda row: row["node_id"])


def raw_nodes(dialogue: str, turns: list[dict[str, Any]], captions: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for turn in turns:
        sid, ordinal = str(turn["turn_id"]).split("#", 1)
        node_id = f"raw:{dialogue}:{sid}:{ordinal}"
        image_captions: dict[str, str] = {}
        image_paths = []
        for image in turn.get("images", []):
            image_id = str(image.get("image_id") or "")
            if not image_id:
                continue
            image_sid, filename = image_id.split(":", 1)
            caption_key = f"h2hmem:multi-party/{dialogue}/scenes/{image_sid}/image/{filename}"
            image_captions[image_id] = str(captions.get(caption_key) or image.get("caption") or "")
            image_paths.append(str(DATA_ROOT / "multi-party" / dialogue / "scenes" / image_sid / "image" / filename))
        speaker = str(turn.get("speaker") or "unknown")
        text = strip_speaker_prefix(str(turn.get("text") or ""), speaker)
        retrieval_text = text
        if image_captions:
            retrieval_text += "\n" + "\n".join(
                f"image_id={image_id}; caption={caption}"
                for image_id, caption in image_captions.items()
            )
        rows.append({
            "node_id": node_id,
            "layer": "raw",
            "text": text,
            "retrieval_text": retrieval_text,
            "speaker": speaker,
            "session_id": sid,
            "ordinal": int(ordinal),
            "date": str(turn.get("date") or ""),
            "image_captions": image_captions,
            "image_paths": image_paths,
            "refer_ids": [node_id],
        })
    return rows


def build_dialogue(dialogue: str, out_dir: Path, model: str, force: bool) -> dict[str, Any]:
    source = SOURCE_ROOT / f"multi-party_{dialogue}.json"
    source_doc = json.loads(source.read_text(encoding="utf-8"))
    captions = json.loads(CAPTION_PATH.read_text(encoding="utf-8"))
    raw = raw_nodes(dialogue, source_doc["turns"], captions)
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        by_session[row["session_id"]].append(row)

    checkpoint_root = out_dir.parent / "checkpoints" / dialogue
    all_facts = []
    for session_id, rows in by_session.items():
        rows.sort(key=lambda row: row["ordinal"])
        valid_image_ids = {
            image_id for row in rows for image_id in row.get("image_captions", {})
        }
        for start, window in make_windows(rows):
            checkpoint = checkpoint_root / session_id / f"{start}.json"
            window_by_ref = {row["node_id"]: row for row in window}
            if checkpoint.exists() and not force:
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if saved.get("prompt_version") != PROMPT_VERSION:
                    raise RuntimeError(f"prompt version mismatch: {checkpoint}")
                facts = saved.get("facts", [])
                if facts and all(
                    fact.get("refer_ids")
                    and set(fact["refer_ids"]).issubset(window_by_ref)
                    for fact in facts
                ):
                    rebound_facts = []
                    for fact in facts:
                        rebound = dict(fact)
                        binding = bind_speaker_from_references(
                            rebound.get("speaker"),
                            rebound["refer_ids"],
                            window_by_ref,
                            speaker_field="speaker",
                        )
                        rebound["speaker"] = binding.value
                        rebound["speaker_binding"] = binding.source
                        rebound_facts.append(rebound)
                    all_facts.extend(rebound_facts)
                    continue

            prompt = USER_PROMPT.format(date=rows[0]["date"], dialogue=render(window))
            last_error: Exception | None = None
            facts = []
            for attempt in range(1, 5):
                correction = "" if last_error is None else (
                    "\nYour previous output failed grounding validation: "
                    f"{last_error}. Regenerate the complete JSON for this window."
                )
                response = call_llm(
                    [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt + correction}],
                    model=model,
                    temperature=0,
                    max_tokens=6000,
                    response_format={"type": "json_object"},
                )
                try:
                    parsed = extract_json(response)
                    facts = [
                        normalize_fact(
                            item,
                            window_by_ref=window_by_ref,
                            valid_image_ids=valid_image_ids,
                        )
                        for item in parsed.get("facts", [])
                    ]
                    if not facts:
                        raise ValueError("no grounded facts")
                    last_error = None
                    break
                except (KeyError, TypeError, ValueError) as exc:
                    last_error = exc
                    print(f"[{dialogue}] {session_id}@{start} validation {attempt}/4: {exc}", flush=True)
            if last_error is not None:
                raise RuntimeError(f"failed grounding {dialogue}/{session_id}@{start}") from last_error
            atomic_json(checkpoint, {
                "prompt_version": PROMPT_VERSION,
                "dialogue": dialogue,
                "session_id": session_id,
                "window_start": start,
                "facts": facts,
            })
            all_facts.extend(facts)
            print(f"[{dialogue}] {session_id}@{start}: {len(window)} turns -> {len(facts)} facts", flush=True)

    facts = deduplicate(all_facts)
    collections = []
    for session_id, rows in by_session.items():
        members = [fact for fact in facts if fact["session_id"] == session_id]
        refs = list(dict.fromkeys(ref for fact in members for ref in fact["refer_ids"]))
        if members and refs:
            collections.append({
                "node_id": f"collection:{dialogue}:{session_id}",
                "layer": "collection",
                "text": f"Grounded event collection for {session_id}.",
                "retrieval_text": " ".join(fact["text"] for fact in members),
                "session_id": session_id,
                "date": rows[0]["date"],
                "member_fact_ids": [fact["node_id"] for fact in members],
                "refer_ids": refs,
            })
    meta = {
        "prompt_version": PROMPT_VERSION,
        "source": str(source),
        "turns": len(raw),
        "facts": len(facts),
        "cross_turn_facts": sum(len(fact["refer_ids"]) > 1 for fact in facts),
        "collections": len(collections),
        "speaker_missing": sum(not fact.get("speaker") for fact in facts),
        "addressee_missing": sum(not fact.get("addressee") for fact in facts),
        "empty_refer_ids": sum(
            not row.get("refer_ids") for layer in (raw, facts, collections) for row in layer
        ),
        "old_fact_cache_consumed": False,
        "qa_consumed": False,
    }
    if any(meta[key] for key in ("speaker_missing", "addressee_missing", "empty_refer_ids")):
        raise RuntimeError(f"memory integrity failure: {meta}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"multi-party_{dialogue}.json"
    atomic_json(target, {
        "split": "multi-party",
        "dialogue": dialogue,
        "raw": raw,
        "facts": facts,
        "collections": collections,
        "meta": meta,
    })
    print(json.dumps({"path": str(target), **meta}, ensure_ascii=False), flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogue", action="append", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for dialogue in args.dialogue:
        build_dialogue(dialogue, args.out, args.model, args.force)
    print_usage_summary()


if __name__ == "__main__":
    main()
