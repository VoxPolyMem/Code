#!/usr/bin/env python3
"""Create QA-free VoxPoly case views for speaker-source ablations.

This module deliberately sits in an experiment-only directory.  It does not
modify the frozen r12 adapter, memory builder, evaluator, or memory schema.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "voxpoly-speaker-view.v1"
MODES = {"oracle", "online_predicted"}
ONLINE_PROTOCOL = "Online Acoustic + Final Predicted Alias"

# These annotations reveal the source speaker identity and must never survive
# into an online-predicted view.  Natural names inside turn text are dialogue
# content, not hidden labels, and are intentionally preserved.
TURN_IDENTITY_FIELDS = {
    "addressee",
    "addressee_id",
    "addressee_ids",
    "addressee_name",
    "addressee_names",
    "cluster",
    "cluster_id",
    "cluster_label",
    "speaker",
    "speaker_alias",
    "speaker_aliases",
    "speaker_id",
    "speaker_key",
    "speaker_label",
    "speaker_name",
    "speaker_role",
    "tts_speaker",
    "voice",
    "voice_id",
    "voice_name",
}

# The frozen VoxPoly adapter needs only sessions/session_dates/dialogues.  A
# whitelist prevents QA, participant rosters, hidden labels, and audit metadata
# from entering the memory-construction view.
SAFE_TOP_LEVEL_FIELDS = (
    "schema_version",
    "case_id",
    "theme",
    "session_dates",
)
SAFE_SESSION_FIELDS = ("session_id", "date")


class ViewValidationError(ValueError):
    """Raised when a speaker view cannot be joined without ambiguity."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON with fsync + same-directory replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def resolve_source_case(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_dir():
        candidate = candidate / "full_case.json"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate.resolve()


def _source_speaker(turn: dict[str, Any]) -> str:
    for key in ("speaker_name", "speaker_id", "speaker_role", "speaker"):
        value = turn.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ViewValidationError("oracle turn has no nonempty speaker identity")


def _coordinates(dialogues: dict[str, Any]) -> list[tuple[str, int]]:
    if not isinstance(dialogues, dict):
        raise ViewValidationError("dialogues must be a session-keyed object")
    coordinates: list[tuple[str, int]] = []
    for session, turns in dialogues.items():
        if not isinstance(turns, list):
            raise ViewValidationError(f"dialogues[{session!r}] must be a list")
        coordinates.extend((str(session), index) for index in range(len(turns)))
    if len(coordinates) != len(set(coordinates)):
        raise ViewValidationError("source dialogue contains duplicate coordinates")
    return coordinates


def _load_online_predictions(
    path: Path, expected_case_id: str, expected_coordinates: Iterable[tuple[str, int]]
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    prediction_case_id = str(document.get("case_id") or "")
    if prediction_case_id != expected_case_id:
        raise ViewValidationError(
            f"case_id mismatch: source={expected_case_id!r}, "
            f"predictions={prediction_case_id!r}"
        )
    rows = document.get("predictions_frozen")
    if not isinstance(rows, list):
        raise ViewValidationError("predictions_frozen must be a list")

    joined: dict[tuple[str, int], dict[str, Any]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ViewValidationError(f"prediction {position} must be an object")
        session = row.get("session")
        turn_idx = row.get("turn_idx")
        if session is None or isinstance(turn_idx, bool) or not isinstance(turn_idx, int):
            raise ViewValidationError(
                f"prediction {position} requires string session and integer turn_idx"
            )
        key = (str(session), turn_idx)
        if key in joined:
            raise ViewValidationError(f"duplicate prediction coordinate: {key}")
        stable_id = str(row.get("stable_identity_id") or "").strip()
        acoustic_id = str(row.get("acoustic_speaker_id") or "").strip()
        if not stable_id or not acoustic_id:
            raise ViewValidationError(
                f"prediction {key} lacks stable_identity_id/acoustic_speaker_id"
            )
        joined[key] = row

    expected = set(expected_coordinates)
    actual = set(joined)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ViewValidationError(
            f"coordinate coverage mismatch: missing={missing[:8]} "
            f"({len(missing)} total), extra={extra[:8]} ({len(extra)} total)"
        )
    return joined, document


def _safe_sessions(source: dict[str, Any], dialogue_ids: list[str]) -> list[dict[str, Any]]:
    rows = source.get("sessions")
    if not isinstance(rows, list):
        return [{"session_id": session} for session in dialogue_ids]
    safe = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        reduced = {key: copy.deepcopy(row[key]) for key in SAFE_SESSION_FIELDS if key in row}
        if reduced.get("session_id") is not None:
            reduced["session_id"] = str(reduced["session_id"])
            safe.append(reduced)
    return safe or [{"session_id": session} for session in dialogue_ids]


def _is_identity_annotation_key(key: str) -> bool:
    normalized = key.lower()
    return (
        key in TURN_IDENTITY_FIELDS
        or "speaker" in normalized
        or "addressee" in normalized
        or "voice" in normalized
        or "cluster" in normalized
    )


def _strip_identity(turn: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    clean = copy.deepcopy(turn)
    removed = sorted(key for key in clean if _is_identity_annotation_key(key))
    for key in removed:
        clean.pop(key, None)
    return clean, removed


def build_case_view(
    source_case: Path,
    output_dir: Path,
    mode: str = "oracle",
    identity_predictions: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ViewValidationError(f"unsupported speaker mode: {mode!r}")
    source_path = resolve_source_case(source_case)
    output_dir = output_dir.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    case_id = str(source.get("case_id") or source.get("source_case_id") or "").strip()
    if not case_id:
        raise ViewValidationError("source case has no case_id")
    dialogues = source.get("dialogues")
    coordinates = _coordinates(dialogues)

    prediction_path: Path | None = None
    predictions: dict[tuple[str, int], dict[str, Any]] = {}
    prediction_document: dict[str, Any] | None = None
    if mode == "online_predicted":
        if identity_predictions is None:
            raise ViewValidationError(
                "online_predicted mode requires --identity-predictions"
            )
        prediction_path = identity_predictions.expanduser().resolve()
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        predictions, prediction_document = _load_online_predictions(
            prediction_path, case_id, coordinates
        )
        language_model_calls = prediction_document.get("language_model_calls")
        if language_model_calls not in (None, 0):
            raise ViewValidationError(
                "online identity artifact is not zero-LLM: "
                f"language_model_calls={language_model_calls!r}"
            )
    elif identity_predictions is not None:
        raise ViewValidationError(
            "identity predictions are not accepted in oracle mode"
        )

    view: dict[str, Any] = {
        key: copy.deepcopy(source[key])
        for key in SAFE_TOP_LEVEL_FIELDS
        if key in source
    }
    view["case_id"] = case_id
    removed_top_level_keys = sorted(
        set(source) - set(SAFE_TOP_LEVEL_FIELDS) - {"sessions", "dialogues"}
    )
    dialogue_ids = [str(session) for session in dialogues]
    view["sessions"] = _safe_sessions(source, dialogue_ids)
    view["dialogues"] = {}
    sidecar_rows = []
    stripped_counts: dict[str, int] = {}
    named_predictions = 0
    stable_id_fallbacks = 0

    for session, source_turns in dialogues.items():
        session = str(session)
        output_turns = []
        for turn_idx, source_turn in enumerate(source_turns):
            if not isinstance(source_turn, dict):
                raise ViewValidationError(f"turn {(session, turn_idx)} must be an object")
            clean, removed = _strip_identity(source_turn)
            for field in removed:
                stripped_counts[field] = stripped_counts.get(field, 0) + 1
            turn_id = str(source_turn.get("turn_id") or f"{session}:{turn_idx}")

            if mode == "oracle":
                # Oracle identity is intentionally accessed only in the Oracle
                # ablation.  The online path must also work from a source view
                # in which every hidden identity annotation was removed.
                oracle_label = _source_speaker(source_turn)
                display_label = oracle_label
                predicted_name = None
                stable_id = None
                acoustic_id = None
                clean["speaker_name"] = display_label
            else:
                prediction = predictions[(session, turn_idx)]
                predicted_name = str(prediction.get("speaker_name") or "").strip() or None
                stable_id = str(prediction["stable_identity_id"]).strip()
                acoustic_id = str(prediction["acoustic_speaker_id"]).strip()
                display_label = predicted_name or stable_id
                named_predictions += int(predicted_name is not None)
                stable_id_fallbacks += int(predicted_name is None)
                clean["speaker_name"] = display_label
                clean["speaker_id"] = stable_id

            output_turns.append(clean)
            sidecar_rows.append({
                "session": session,
                "turn_idx": turn_idx,
                "turn_id": turn_id,
                "speaker_display": display_label,
                "predicted_speaker_name": predicted_name,
                "stable_identity_id": stable_id,
                "acoustic_speaker_id": acoustic_id,
            })
        view["dialogues"][session] = output_turns

    protocol_name = "Oracle Speaker" if mode == "oracle" else ONLINE_PROTOCOL
    online_alias_is_causal = False if mode == "online_predicted" else None
    sidecar = {
        "schema_version": f"{SCHEMA_VERSION}.sidecar",
        "case_id": case_id,
        "mode": mode,
        "protocol_name": protocol_name,
        "online_alias_is_causal_at_each_turn": online_alias_is_causal,
        "turns": sidecar_rows,
    }

    case_output = output_dir / "full_case.json"
    sidecar_output = output_dir / "speaker_identity_sidecar.json"
    manifest_output = output_dir / "speaker_view_manifest.json"
    atomic_write_json(case_output, view)
    atomic_write_json(sidecar_output, sidecar)

    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.manifest",
        "case_id": case_id,
        "mode": mode,
        "protocol_name": protocol_name,
        "online_alias_is_causal_at_each_turn": online_alias_is_causal,
        "qa_free": True,
        "strict_join_key": ["session", "turn_idx"],
        "removed_top_level_keys": removed_top_level_keys,
        "removed_turn_keys": sorted(stripped_counts),
        "inputs": {
            "source_case": str(source_path),
            "source_case_sha256": sha256_file(source_path),
            "identity_predictions": str(prediction_path) if prediction_path else None,
            "identity_predictions_sha256": (
                sha256_file(prediction_path) if prediction_path else None
            ),
            "identity_prediction_protocol": (
                prediction_document.get("protocol") if prediction_document else None
            ),
            "identity_prediction_policy": (
                prediction_document.get("policy") if prediction_document else None
            ),
            "identity_prediction_language_model_calls": (
                prediction_document.get("language_model_calls")
                if prediction_document
                else None
            ),
            "config": str(config_path.resolve()) if config_path else None,
            "config_sha256": sha256_file(config_path.resolve()) if config_path else None,
        },
        "outputs": {
            "full_case": str(case_output),
            "full_case_sha256": sha256_file(case_output),
            "speaker_identity_sidecar": str(sidecar_output),
            "speaker_identity_sidecar_sha256": sha256_file(sidecar_output),
        },
        "counts": {
            "sessions": len(view["dialogues"]),
            "turns": len(coordinates),
            "prediction_rows": len(predictions),
            "predicted_names": named_predictions,
            "stable_id_fallbacks": stable_id_fallbacks,
            "stripped_identity_fields": stripped_counts,
        },
        "invariants": {
            "coordinate_coverage_complete": True,
            "coordinate_join_unique": True,
            "source_turn_ids_preserved": True,
            "source_text_preserved": True,
            "qa_or_gold_loaded_into_view": False,
            "frozen_r12_files_modified": False,
        },
        "code": {
            "make_case_view": str(Path(__file__).resolve()),
            "make_case_view_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_write_json(manifest_output, manifest)
    return manifest


def load_config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ViewValidationError("config must be a JSON object")
    mode = document.get("speaker_mode")
    if mode not in MODES:
        raise ViewValidationError(f"invalid config speaker_mode: {mode!r}")
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=sorted(MODES))
    parser.add_argument("--identity-predictions", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else {}
    configured_mode = config.get("speaker_mode")
    if args.mode and configured_mode and args.mode != configured_mode:
        raise ViewValidationError(
            f"--mode {args.mode!r} conflicts with config mode {configured_mode!r}"
        )
    # Default-off: without an explicit online config/mode, use Oracle Speaker.
    mode = args.mode or configured_mode or "oracle"
    manifest = build_case_view(
        source_case=args.source_case,
        output_dir=args.output_dir,
        mode=mode,
        identity_predictions=args.identity_predictions,
        config_path=args.config,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
