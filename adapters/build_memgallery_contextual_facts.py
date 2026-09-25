"""Build speaker-aware, cross-turn contextual facts for Mem-Gallery.

The builder reads dialogue only. It never reads QA, answers, benchmark types,
or gold evidence. Overlapping windows are checkpointed and all ablations are
derived from one set of model outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
V2_ROOT = Path(os.environ.get(
    "V2_PIPELINE_ROOT", PACKAGE_ROOT / "vendor/v2_update_pipeline"
))
DATA_ROOT = Path(os.environ.get(
    "MEMGALLERY_ROOT", PACKAGE_ROOT / "data/Mem-Gallery"
))
DEFAULT_OUT = PACKAGE_ROOT / "artifacts/memory/memgallery_contextual_v1"
LEGACY_FACT_ROOT = Path(os.environ.get(
    "LEGACY_MEMGALLERY_FACT_ROOT",
    PACKAGE_ROOT / "data/memgallery_legacy_facts",
))
sys.path.insert(0, str(V2_ROOT))

from utils.llm_client import call_llm, extract_json, print_usage_summary  # noqa: E402
from core.contextual_fact_views import (  # noqa: E402
    deduplicate_facts,
    materialize_ablation_view,
    normalize_fact,
)


PROMPT_VERSION = "contextual-atomic-facts-v1"
WINDOW_SIZE = 12
STRIDE = 6
VIEWS = ("full", "no_coreference", "single_turn_only", "no_role_metadata")
WINDOW_VALIDATION_ATTEMPTS = 4


SYSTEM = """You extract grounded atomic memory from a dialogue window.
Return strict JSON only. Preserve attribution and never infer unsupported facts."""


USER_PROMPT = """Character name: {character}
Session date: {date}

Dialogue window:
{dialogue}

Extract durable, self-contained atomic facts. Use the surrounding turns to
resolve context, but never merge unrelated information merely because it is
nearby or shares a topic.

Create a multi-turn fact only when at least one operation is necessary:
- coreference: resolve it/this/that/he/she/the second one to an explicit entity;
- relative_time: resolve expressions such as yesterday or last week;
- reply_resolution: a reply accepts, rejects, modifies, or answers an earlier turn;
- state_transition: a later turn updates or contradicts an earlier state;
- multi_turn_answer: a conclusion requires a question and its answer together;
- speaker_attribution: neighboring turns are needed to identify who said what.

Rules:
1. Every fact must identify the principal source_role as exactly user or assistant.
2. refer_ids must be nonempty and use only D#:N identifiers printed below.
3. Cite every turn required to support or resolve the fact, but no irrelevant turn.
4. If two or more refer_ids are used, context_operations must be nonempty.
5. Keep speaker attribution separate from claimant. claimant is optional and is
   used only when the utterance reports somebody else's belief or statement.
6. Preserve exact entities, image IDs, dates, quantities, negation, uncertainty,
   decisions, and changes of state.
7. Resolve relative time only when the session date and dialogue make it safe;
   preserve the original phrase in original_time_expression.
8. Images may be used only through the printed image_id and caption.
9. Skip greetings, pure questions, and unsupported assistant speculation.

Return exactly:
{{"facts":[{{
  "text":"self-contained fact",
  "subject":"explicit entity",
  "predicate":"normalized relation",
  "object":"value or event",
  "fact_type":"event|state|preference|relationship|plan|belief|decision",
  "source_role":"user|assistant",
  "claimant":null,
  "participants":["explicit entity"],
  "valid_time":null,
  "original_time_expression":null,
  "context_operations":["coreference"],
  "refer_ids":["D1:1"],
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


def windows(rows: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    if not rows:
        return []
    starts = list(range(0, max(len(rows) - WINDOW_SIZE + 1, 1), STRIDE))
    final_start = max(0, len(rows) - WINDOW_SIZE)
    if final_start not in starts:
        starts.append(final_start)
    return [(start, rows[start:start + WINDOW_SIZE]) for start in sorted(set(starts))]


def render(rows: list[dict[str, Any]], character: str) -> str:
    blocks = []
    for row in rows:
        turn_id = str(row["round"])
        block = (
            f"[{turn_id}|user] {character}: {row.get('user', '')}\n"
            f"[{turn_id}|assistant] assistant: {row.get('assistant', '')}"
        )
        image_ids = row.get("image_id") or []
        captions = row.get("image_caption") or []
        if isinstance(image_ids, str):
            image_ids = [image_ids]
        if isinstance(captions, str):
            captions = [captions]
        for image_id, caption in zip(image_ids, captions):
            block += f"\n[{turn_id}|image] image_id={image_id}; caption={caption}"
        blocks.append(block)
    return "\n\n".join(blocks)


def canonicalize_refer_ids(item: dict[str, Any]) -> dict[str, Any]:
    """Collapse a model-selected rendered span ID to its canonical turn ID.

    The prompt renders ``D1:2|user``, ``D1:2|assistant``, and ``D1:2|image``
    so the model can see the source channel, while the memory contract stores
    the shared underlying evidence ID ``D1:2``.  This changes only notation;
    it never chooses, adds, or replaces an evidence ID.
    """

    value = dict(item)
    refs = value.get("refer_ids") or []
    if isinstance(refs, str):
        refs = [refs]
    value["refer_ids"] = [str(ref).strip().split("|", 1)[0] for ref in refs]
    return value


def export_compatible_view(
    *, case_name: str, view: str, facts: list[dict[str, Any]], out_root: Path
) -> Path:
    """Export one view in the frozen FactFusion directory contract."""

    root = out_root / "compatible_views" / view / case_name
    by_session: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        sessions = {str(ref).split(":", 1)[0] for ref in fact["refer_ids"]}
        if len(sessions) != 1:
            raise RuntimeError(f"cross-session atomic fact is not allowed: {fact}")
        by_session.setdefault(next(iter(sessions)), []).append(fact)

    for session_id, session_facts in by_session.items():
        session_root = root / session_id
        cards = []
        for index, fact in enumerate(session_facts, 1):
            atomic = {
                "text": fact["retrieval_text"],
                "display_text": fact["text"],
                "speaker": fact["speaker"],
                "addressee": fact["addressee"],
                "claimant": fact.get("claimant"),
                "context_operations": fact.get("context_operations", []),
                "source_turn_ids": fact["refer_ids"],
                "refer_ids": fact["refer_ids"],
                "source_image_ids": fact.get("image_ids", []),
            }
            cards.append({
                "event_id": f"CF{index:04d}",
                "card_id": fact["node_id"],
                "primary_subject": fact.get("subject") or fact["speaker"],
                "event_label": fact.get("predicate") or fact.get("fact_type") or "fact",
                "valid_time": fact.get("valid_time"),
                "source_turn_ids": fact["refer_ids"],
                "refer_ids": fact["refer_ids"],
                "source_image_ids": fact.get("image_ids", []),
                "atomic_facts": [atomic],
                "relations": [],
                "summary": fact["text"],
                "retrieval_text": fact["retrieval_text"],
                "confidence": 1.0,
            })
        atomic_json(session_root / "event_cards.json", {
            "schema_version": PROMPT_VERSION,
            "case": case_name,
            "session": session_id,
            "view": view,
            "event_cards": cards,
        })
        old_visual = LEGACY_FACT_ROOT / case_name / session_id / "visual_sets.json"
        if old_visual.exists():
            session_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_visual, session_root / "visual_sets.json")
    return root.parent


def build_case(case_name: str, out_root: Path, model: str, force: bool) -> dict[str, Any]:
    source = DATA_ROOT / "data/dialog" / f"{case_name}.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    character = str(data["character_profile"]["name"])
    case_root = out_root / case_name
    all_facts = []

    for session in data["multi_session_dialogues"]:
        session_id = str(session["session_id"])
        rows = list(session.get("dialogues", []))
        valid_images = {
            str(image_id)
            for row in rows
            for image_id in (
                row.get("image_id") or []
                if isinstance(row.get("image_id") or [], list)
                else [row.get("image_id")]
            )
        }
        for start, window in windows(rows):
            window_refs = {str(row["round"]) for row in window}
            checkpoint = case_root / "windows" / f"{session_id}@{start}.json"
            if checkpoint.exists() and not force:
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if saved.get("prompt_version") != PROMPT_VERSION:
                    raise RuntimeError(f"prompt version mismatch: {checkpoint}")
                saved_facts = saved.get("facts", [])
                refs_are_local = all(
                    fact.get("refer_ids")
                    and set(map(str, fact["refer_ids"])).issubset(window_refs)
                    for fact in saved_facts
                )
                if refs_are_local:
                    all_facts.extend(saved_facts)
                    continue
                print(f"[{case_name}] rebuilding non-local checkpoint: {checkpoint.name}", flush=True)

            prompt = USER_PROMPT.format(
                character=character,
                date=str(session.get("date") or "unknown"),
                dialogue=render(window, character),
            )
            facts = []
            validation_error: Exception | None = None
            for attempt in range(1, WINDOW_VALIDATION_ATTEMPTS + 1):
                correction = ""
                if validation_error is not None:
                    correction = (
                        "\n\nYour previous output failed grounding validation: "
                        f"{validation_error}. Regenerate the complete JSON for this same "
                        "window. Select refer_ids only from IDs printed in this window."
                    )
                response = call_llm(
                    [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt + correction},
                    ],
                    model=model,
                    temperature=0,
                    max_tokens=6000,
                    response_format={"type": "json_object"},
                )
                try:
                    parsed = extract_json(response)
                    facts = [
                        normalize_fact(
                            canonicalize_refer_ids(item),
                            character=character,
                            valid_refer_ids=window_refs,
                            valid_image_ids=valid_images,
                        )
                        for item in parsed.get("facts", [])
                    ]
                    if not facts:
                        raise ValueError("the model returned no grounded facts")
                    validation_error = None
                    break
                except (KeyError, TypeError, ValueError) as exc:
                    validation_error = exc
                    print(
                        f"[{case_name}] {session_id}@{start}: grounding retry "
                        f"{attempt}/{WINDOW_VALIDATION_ATTEMPTS}: {exc}",
                        flush=True,
                    )
            if validation_error is not None:
                raise RuntimeError(
                    f"grounding validation failed for {case_name}/{session_id}@{start}"
                ) from validation_error
            if not facts:
                raise RuntimeError(f"empty fact window: {case_name}/{session_id}@{start}")
            atomic_json(checkpoint, {
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "case": case_name,
                "session": session_id,
                "window_start": start,
                "facts": facts,
            })
            all_facts.extend(facts)
            print(
                f"[{case_name}] {session_id}@{start}: {len(window)} turns -> "
                f"{len(facts)} facts",
                flush=True,
            )

    facts = deduplicate_facts(all_facts)
    views = {}
    for view in VIEWS:
        rows = materialize_ablation_view(facts, view)
        path = case_root / "views" / f"{view}.json"
        atomic_json(path, {
            "prompt_version": PROMPT_VERSION,
            "case": case_name,
            "view": view,
            "facts": rows,
        })
        compatible_root = export_compatible_view(
            case_name=case_name, view=view, facts=rows, out_root=out_root
        )
        views[view] = {
            "path": str(path),
            "compatible_fact_root": str(compatible_root),
            "facts": len(rows),
        }

    meta = {
        "turns": sum(len(x.get("dialogues", [])) for x in data["multi_session_dialogues"]),
        "facts": len(facts),
        "cross_turn_facts": sum(len(x["refer_ids"]) > 1 for x in facts),
        "speaker_missing": sum(not x.get("speaker") for x in facts),
        "addressee_missing": sum(not x.get("addressee") for x in facts),
        "empty_refer_ids": sum(not x.get("refer_ids") for x in facts),
        "views": views,
    }
    if any(meta[key] for key in ("speaker_missing", "addressee_missing", "empty_refer_ids")):
        raise RuntimeError(f"contextual fact integrity failure: {meta}")
    atomic_json(case_root / "memory.json", {
        "prompt_version": PROMPT_VERSION,
        "case": case_name,
        "character": character,
        "source": str(source),
        "facts": facts,
        "meta": meta,
    })
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for case_name in args.case:
        print(json.dumps(
            {"case": case_name, **build_case(case_name, args.out, args.model, args.force)},
            ensure_ascii=False,
        ))
    print_usage_summary()


if __name__ == "__main__":
    main()
