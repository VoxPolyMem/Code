#!/usr/bin/env python3
"""Prepare a QA-free, identity-free VoxPoly source with stable bottom IDs.

The all18 evaluation projection intentionally keeps only dialogue text and an
oracle ``speaker_id``.  It omits ``turn_id``, while the unified r12 memory
contract requires every bottom node to have a stable provenance ID.  This
zero-API adapter removes QA/oracle identity and deterministically reconstructs
``<session>_T<one-based index>`` IDs.  An optional canonical source is used
only to audit those IDs and text coordinates; none of its other fields can be
copied into the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "voxpoly-r12-qa-free-source.v1"
SAFE_ROOT_FIELDS = ("schema_version", "case_id", "theme", "session_dates", "sessions")
SAFE_TURN_FIELDS = (
    "turn_id", "ordinal", "text", "timestamp",
    "image_id", "image_ids", "image_caption", "image_captions", "images",
    "audio_path", "audio_file", "audio_ref",
    "speech_act", "interaction_id", "reply_to_turn_id",
)


class PreparationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreparationError(f"{label} must be a JSON object")
    return value


def deterministic_turn_id(session: str, turn_idx: int) -> str:
    return f"{session}_T{turn_idx + 1:03d}"


def dialogue_rows(document: dict[str, Any], label: str) -> dict[str, list[dict[str, Any]]]:
    dialogues = document.get("dialogues")
    if not isinstance(dialogues, dict) or not dialogues:
        raise PreparationError(f"{label}.dialogues must be a nonempty session map")
    result: dict[str, list[dict[str, Any]]] = {}
    for session_value, turns in dialogues.items():
        session = str(session_value).strip()
        if not session or not isinstance(turns, list) or not turns:
            raise PreparationError(f"invalid dialogue session in {label}: {session_value!r}")
        normalized: list[dict[str, Any]] = []
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise PreparationError(f"{label}.{session}[{index}] is not an object")
            if not str(turn.get("text") or "").strip():
                raise PreparationError(f"{label}.{session}[{index}] has no text")
            normalized.append(turn)
        result[session] = normalized
    return result


def prepare(
    source_path: Path,
    output_dir: Path,
    *,
    canonical_id_source: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    source_path = source_path.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    output_path = output_dir / "full_case.json"
    manifest_path = output_dir / "preparation_manifest.json"
    if not force and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"refusing to overwrite prepared source in {output_dir}")

    source = read_object(source_path, "all18 source")
    source_dialogues = dialogue_rows(source, "all18 source")
    case_id = str(source.get("case_id") or "").strip()
    if not case_id:
        raise PreparationError("all18 source has no case_id")

    canonical_path: Path | None = None
    canonical_dialogues: dict[str, list[dict[str, Any]]] | None = None
    if canonical_id_source is not None:
        canonical_path = canonical_id_source.expanduser().resolve(strict=True)
        canonical = read_object(canonical_path, "canonical ID audit source")
        if str(canonical.get("case_id") or "") != case_id:
            raise PreparationError("canonical ID audit source has a different case_id")
        canonical_dialogues = dialogue_rows(canonical, "canonical ID audit source")
        if list(canonical_dialogues) != list(source_dialogues):
            raise PreparationError("canonical session order differs from all18 source")

    prepared: dict[str, Any] = {
        key: source[key] for key in SAFE_ROOT_FIELDS if key in source
    }
    prepared["case_id"] = case_id
    prepared["dialogues"] = {}
    turn_ids: list[str] = []
    coordinate_text: list[list[Any]] = []
    generated_count = 0
    canonical_id_matches = 0

    for session, turns in source_dialogues.items():
        canonical_turns = canonical_dialogues.get(session) if canonical_dialogues else None
        if canonical_turns is not None and len(canonical_turns) != len(turns):
            raise PreparationError(f"canonical turn count differs in {session}")
        output_turns: list[dict[str, Any]] = []
        for turn_idx, turn in enumerate(turns):
            expected_id = deterministic_turn_id(session, turn_idx)
            existing_id = str(turn.get("turn_id") or "").strip()
            turn_id = existing_id or expected_id
            if not existing_id:
                generated_count += 1
            if turn_id != expected_id:
                raise PreparationError(
                    f"noncanonical turn_id at {(session, turn_idx)}: {turn_id!r} != {expected_id!r}"
                )
            text = str(turn["text"])
            if canonical_turns is not None:
                canonical_turn = canonical_turns[turn_idx]
                canonical_id = str(canonical_turn.get("turn_id") or "").strip()
                if canonical_id != turn_id:
                    raise PreparationError(
                        f"generated ID disagrees with canonical source at {(session, turn_idx)}"
                    )
                if str(canonical_turn.get("text") or "") != text:
                    raise PreparationError(
                        f"all18 text disagrees with canonical source at {(session, turn_idx)}"
                    )
                canonical_id_matches += 1
            clean = {key: turn[key] for key in SAFE_TURN_FIELDS if key in turn}
            clean["turn_id"] = turn_id
            clean["ordinal"] = int(turn.get("ordinal") or turn_idx + 1)
            clean["text"] = text
            output_turns.append(clean)
            turn_ids.append(turn_id)
            coordinate_text.append([session, turn_idx, text])
        prepared["dialogues"][session] = output_turns

    if len(turn_ids) != len(set(turn_ids)):
        raise PreparationError("prepared turn IDs are not globally unique")
    if any(key.lower() in {"qa", "answer", "gold", "evidence"} for key in prepared):
        raise PreparationError("forbidden QA/gold root survived preparation")

    atomic_json(output_path, prepared)
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.manifest",
        "status": "complete",
        "zero_api": True,
        "case_id": case_id,
        "source": {
            "all18_path": str(source_path),
            "all18_sha256": sha256_file(source_path),
            "all18_root_keys": sorted(source),
            "canonical_id_audit_path": str(canonical_path) if canonical_path else None,
            "canonical_id_audit_sha256": sha256_file(canonical_path) if canonical_path else None,
        },
        "output": {
            "full_case": str(output_path),
            "full_case_sha256": sha256_file(output_path),
        },
        "counts": {
            "sessions": len(source_dialogues),
            "turns": len(turn_ids),
            "generated_turn_ids": generated_count,
            "canonical_id_and_text_matches": canonical_id_matches,
        },
        "turn_id_policy": "existing ID or <session>_T<one-based index padded to 3 digits>",
        "coordinate_text_sha256": hashlib.sha256(
            json.dumps(coordinate_text, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "removed_root_keys": sorted(set(source) - set(SAFE_ROOT_FIELDS) - {"dialogues"}),
        "retained_turn_fields": sorted({key for turns in prepared["dialogues"].values() for row in turns for key in row}),
        "invariants": {
            "qa_free": True,
            "oracle_identity_free": True,
            "coordinate_order_preserved": True,
            "source_text_preserved": True,
            "bottom_ids_unique": True,
            "canonical_id_audit_passed": canonical_dialogues is not None,
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--canonical-id-source", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        report = prepare(
            args.source,
            args.output_dir,
            canonical_id_source=args.canonical_id_source,
            force=args.force,
        )
    except (OSError, json.JSONDecodeError, PreparationError) as exc:
        print(f"source preparation failed: {exc}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
