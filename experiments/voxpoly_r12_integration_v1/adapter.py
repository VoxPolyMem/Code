"""Fail-closed adapter from a QA-free VoxPoly view to shared raw/profile nodes.

The adapter consumes only artifacts produced by ``make_case_view.py`` in
``online_predicted`` mode.  Hidden speaker/addressee labels, QA, answers, and
gold evidence are rejected before any memory is constructed.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ADAPTER_VERSION = "voxpoly-r12-adapter-v1.1-reply-edge"
SAFE_ROOT_FIELDS = {
    "schema_version", "case_id", "theme", "session_dates", "sessions", "dialogues"
}
SAFE_TURN_FIELDS = {
    "turn_id", "ordinal", "text", "timestamp",
    # These two labels are injected from the predicted online sidecar.
    "speaker_id", "speaker_name",
    # Observable media fields. Audio is referenced, never copied into JSON.
    "image_id", "image_ids", "image_caption", "image_captions", "images",
    "audio_path", "audio_file", "audio_ref",
    # Generator bookkeeping is tolerated in the clean view but deliberately
    # ignored by the prompt and retrieval text.  The observable reply link is
    # retained as optional raw provenance; speech_act/interaction_id are not.
    "speech_act", "interaction_id", "reply_to_turn_id",
}
FORBIDDEN_ROOT_MARKERS = (
    "qa", "answer", "gold", "evidence", "character", "participant_roster",
    "hidden", "audit", "source_provenance", "query_contract",
)
FORBIDDEN_TURN_MARKERS = (
    "addressee", "oracle", "ground_truth", "gold", "evidence", "answer",
    "voice", "cluster", "speaker_role", "tts_speaker",
)
REGISTRY_MEMBERS = {
    "speaker_ids.npy", "assigned_turns.npy", "updates.npy", "prototypes.npy"
}


class AdapterContractError(ValueError):
    """Raised when the input view can no longer guarantee a QA-blind join."""


@dataclass(frozen=True)
class CaseBundle:
    case_id: str
    case_name: str
    raw: tuple[dict[str, Any], ...]
    profiles: tuple[dict[str, Any], ...]
    source_view_sha256: str
    sidecar_sha256: str
    prediction_sha256: str
    prediction_protocol: str
    registry_state_sha256: str
    ema_policy: dict[str, float]
    alias_is_causal_at_each_turn: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterContractError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterContractError(f"{label} root must be an object")
    return value


def _nonempty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise AdapterContractError(f"{label} must be nonempty")
    return text


def _optional_turn_ref(value: Any, label: str) -> str | None:
    """Normalize one observable bottom-turn reference without guessing it."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise AdapterContractError(f"{label} must be a string or null")
    return value.strip() or None


def _has_marker(key: str, markers: tuple[str, ...]) -> bool:
    normalized = key.strip().lower()
    return any(
        normalized == marker
        or normalized.startswith(marker + "_")
        or normalized.endswith("_" + marker)
        for marker in markers
    )


def _session_date(view: dict[str, Any], session_id: str) -> str:
    value = (view.get("session_dates") or {}).get(session_id)
    if isinstance(value, dict):
        return str(value.get("date") or "")
    if isinstance(value, str):
        return value
    for row in view.get("sessions") or []:
        if isinstance(row, dict) and str(row.get("session_id")) == session_id:
            return str(row.get("date") or "")
    return ""


def _media_fields(turn: dict[str, Any]) -> tuple[list[str], dict[str, str], str | None]:
    image_ids: list[str] = []
    captions: dict[str, str] = {}
    raw_ids = turn.get("image_ids", turn.get("image_id", []))
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    if isinstance(raw_ids, list):
        image_ids.extend(str(value).strip() for value in raw_ids if str(value).strip())
    raw_captions = turn.get("image_captions", turn.get("image_caption", []))
    if isinstance(raw_captions, dict):
        captions.update({str(k): str(v) for k, v in raw_captions.items()})
    else:
        if isinstance(raw_captions, str):
            raw_captions = [raw_captions]
        if isinstance(raw_captions, list):
            captions.update({key: str(value) for key, value in zip(image_ids, raw_captions)})
    images = turn.get("images")
    if isinstance(images, list):
        for row in images:
            if not isinstance(row, dict):
                continue
            image_id = str(row.get("image_id") or row.get("id") or "").strip()
            if image_id and image_id not in image_ids:
                image_ids.append(image_id)
            if image_id and row.get("caption") is not None:
                captions[image_id] = str(row["caption"])
    audio_ref = next(
        (str(turn[key]).strip() for key in ("audio_ref", "audio_path", "audio_file")
         if turn.get(key) is not None and str(turn[key]).strip()),
        None,
    )
    return list(dict.fromkeys(image_ids)), captions, audio_ref


def _validate_registry(path: Path) -> str:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != REGISTRY_MEMBERS:
                raise AdapterContractError(
                    f"EMA registry members differ: expected={sorted(REGISTRY_MEMBERS)}, "
                    f"actual={sorted(set(names))}"
                )
            corrupt = archive.testzip()
            if corrupt is not None:
                raise AdapterContractError(f"EMA registry member is corrupt: {corrupt}")
    except (OSError, zipfile.BadZipFile) as exc:
        raise AdapterContractError(f"invalid EMA registry: {path}: {exc}") from exc
    return sha256_file(path)


def load_case_bundle(
    case_view_dir: Path,
    registry_state: Path,
    *,
    enabled: bool = False,
    require_strict_causal_aliases: bool = False,
) -> CaseBundle:
    """Load and validate one predicted-speaker bundle.

    ``enabled`` is intentionally false by default so importing this experiment
    cannot alter any Mem-Gallery/H2H run.
    """

    if not enabled:
        raise AdapterContractError(
            "VoxPoly r12 integration is default-off; pass the explicit enable flag"
        )
    root = case_view_dir.expanduser().resolve(strict=True)
    view_path = root / "full_case.json"
    sidecar_path = root / "speaker_identity_sidecar.json"
    manifest_path = root / "speaker_view_manifest.json"
    view = _read_object(view_path, "QA-free case view")
    sidecar = _read_object(sidecar_path, "speaker sidecar")
    manifest = _read_object(manifest_path, "speaker manifest")

    if manifest.get("mode") != "online_predicted" or manifest.get("qa_free") is not True:
        raise AdapterContractError("only a manifest-declared QA-free online_predicted view is allowed")
    causal = manifest.get("online_alias_is_causal_at_each_turn") is True
    if require_strict_causal_aliases and not causal:
        raise AdapterContractError("the selected final-alias view is batch-final, not strictly causal")
    if manifest.get("strict_join_key") != ["session", "turn_idx"]:
        raise AdapterContractError("speaker view must use the exact (session, turn_idx) join")
    if manifest.get("invariants", {}).get("qa_or_gold_loaded_into_view") is not False:
        raise AdapterContractError("manifest does not prove that QA/gold stayed outside the view")

    extras = set(view) - SAFE_ROOT_FIELDS
    if extras:
        raise AdapterContractError(f"case view has non-whitelisted root fields: {sorted(extras)}")
    leaking_roots = [key for key in view if _has_marker(key, FORBIDDEN_ROOT_MARKERS)]
    if leaking_roots:
        raise AdapterContractError(f"case view contains forbidden root fields: {leaking_roots}")

    view_sha = sha256_file(view_path)
    sidecar_sha = sha256_file(sidecar_path)
    declared_outputs = manifest.get("outputs") or {}
    if declared_outputs.get("full_case_sha256") != view_sha:
        raise AdapterContractError("case-view SHA256 does not match its manifest")
    if declared_outputs.get("speaker_identity_sidecar_sha256") != sidecar_sha:
        raise AdapterContractError("speaker-sidecar SHA256 does not match its manifest")

    case_id = _nonempty(view.get("case_id"), "case_id")
    if sidecar.get("case_id") != case_id or manifest.get("case_id") != case_id:
        raise AdapterContractError("case_id mismatch across view, sidecar, and manifest")
    if sidecar.get("mode") != "online_predicted":
        raise AdapterContractError("sidecar is not online_predicted")

    dialogues = view.get("dialogues")
    sidecar_rows = sidecar.get("turns")
    if not isinstance(dialogues, dict) or not isinstance(sidecar_rows, list):
        raise AdapterContractError("dialogues and sidecar turns must be collections")
    by_coordinate: dict[tuple[str, int], dict[str, Any]] = {}
    for index, row in enumerate(sidecar_rows):
        if not isinstance(row, dict):
            raise AdapterContractError(f"sidecar turn {index} is not an object")
        session = _nonempty(row.get("session"), f"sidecar[{index}].session")
        turn_idx = row.get("turn_idx")
        if isinstance(turn_idx, bool) or not isinstance(turn_idx, int) or turn_idx < 0:
            raise AdapterContractError(f"sidecar[{index}].turn_idx must be nonnegative")
        coordinate = (session, turn_idx)
        if coordinate in by_coordinate:
            raise AdapterContractError(f"duplicate sidecar coordinate: {coordinate}")
        for key in ("turn_id", "speaker_display", "stable_identity_id", "acoustic_speaker_id"):
            _nonempty(row.get(key), f"sidecar[{index}].{key}")
        by_coordinate[coordinate] = row

    raw: list[dict[str, Any]] = []
    expected_coordinates: list[tuple[str, int]] = []
    seen_turn_ids: set[str] = set()
    for session_value, turns in dialogues.items():
        session = _nonempty(session_value, "dialogue session")
        if not isinstance(turns, list):
            raise AdapterContractError(f"dialogues[{session}] must be a list")
        for turn_idx, turn in enumerate(turns):
            coordinate = (session, turn_idx)
            expected_coordinates.append(coordinate)
            if coordinate not in by_coordinate:
                raise AdapterContractError(f"missing speaker sidecar coordinate: {coordinate}")
            if not isinstance(turn, dict):
                raise AdapterContractError(f"turn {coordinate} must be an object")
            unknown = set(turn) - SAFE_TURN_FIELDS
            if unknown:
                raise AdapterContractError(f"turn {coordinate} has non-whitelisted fields: {sorted(unknown)}")
            leaking = [key for key in turn if _has_marker(key, FORBIDDEN_TURN_MARKERS)]
            if leaking:
                raise AdapterContractError(f"turn {coordinate} has hidden identity fields: {leaking}")
            side = by_coordinate[coordinate]
            turn_id = _nonempty(turn.get("turn_id"), f"turn {coordinate}.turn_id")
            if turn_id != str(side["turn_id"]):
                raise AdapterContractError(f"turn_id mismatch at {coordinate}")
            if turn_id in seen_turn_ids:
                raise AdapterContractError(f"duplicate bottom turn_id: {turn_id}")
            seen_turn_ids.add(turn_id)
            speaker_ref = _nonempty(side.get("stable_identity_id"), "stable_identity_id")
            speaker = _nonempty(side.get("speaker_display"), "speaker_display")
            if str(turn.get("speaker_id") or "").strip() != speaker_ref:
                raise AdapterContractError(f"predicted speaker_id mismatch at {coordinate}")
            if str(turn.get("speaker_name") or "").strip() != speaker:
                raise AdapterContractError(f"predicted speaker_name mismatch at {coordinate}")
            text = _nonempty(turn.get("text"), f"turn {coordinate}.text")
            reply_to_turn_id = _optional_turn_ref(
                turn.get("reply_to_turn_id"),
                f"turn {coordinate}.reply_to_turn_id",
            )
            image_ids, image_captions, audio_ref = _media_fields(turn)
            retrieval_text = text
            if image_captions:
                retrieval_text += "\n" + "\n".join(
                    f"image_id={image_id}; caption={image_captions.get(image_id, '')}"
                    for image_id in image_ids
                )
            raw.append({
                "node_id": f"raw:{turn_id}",
                "layer": "raw",
                "text": text,
                "retrieval_text": retrieval_text,
                "speaker": speaker,
                "speaker_ref": speaker_ref,
                "session_id": session,
                "source_turn_idx": turn_idx,
                "ordinal": int(turn.get("ordinal") or turn_idx + 1),
                "date": _session_date(view, session),
                "timestamp": turn.get("timestamp"),
                "image_ids": image_ids,
                "image_captions": image_captions,
                "audio_ref": audio_ref,
                "reply_to_turn_id": reply_to_turn_id,
                "refer_ids": [turn_id],
            })

    if set(expected_coordinates) != set(by_coordinate):
        missing = sorted(set(by_coordinate) - set(expected_coordinates))
        raise AdapterContractError(f"sidecar has extra coordinates: {missing}")
    if len(raw) != int((manifest.get("counts") or {}).get("turns", -1)):
        raise AdapterContractError("turn count disagrees with manifest")
    for row in raw:
        reply_to_turn_id = row["reply_to_turn_id"]
        if reply_to_turn_id is not None and reply_to_turn_id not in seen_turn_ids:
            raise AdapterContractError(
                f"raw turn {row['refer_ids'][0]} replies to unknown bottom turn: "
                f"{reply_to_turn_id}"
            )

    registry_path = registry_state.expanduser().resolve(strict=True)
    registry_sha = _validate_registry(registry_path)
    policy = (manifest.get("inputs") or {}).get("identity_prediction_policy")
    if not isinstance(policy, dict):
        raise AdapterContractError("manifest has no online EMA policy")
    ema_policy: dict[str, float] = {}
    for key in ("match_cosine_threshold", "ema_update_cosine_threshold", "ema_alpha"):
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AdapterContractError(f"invalid EMA policy value: {key}")
        ema_policy[key] = float(value)
        if not 0.0 <= ema_policy[key] <= 1.0:
            raise AdapterContractError(f"EMA policy value is outside [0,1]: {key}")
    if ema_policy["ema_alpha"] <= 0.0:
        raise AdapterContractError("EMA alpha must be greater than zero")
    if ema_policy["ema_update_cosine_threshold"] < ema_policy["match_cosine_threshold"]:
        raise AdapterContractError("EMA update threshold cannot be below match threshold")

    raw_by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in raw:
        raw_by_speaker.setdefault(row["speaker_ref"], []).append(row)
    profiles = []
    for speaker_ref, rows in sorted(raw_by_speaker.items()):
        aliases = list(dict.fromkeys(row["speaker"] for row in rows if row["speaker"]))
        acoustic_ids = list(dict.fromkeys(
            str(by_coordinate[(row["session_id"], row["source_turn_idx"])]["acoustic_speaker_id"])
            for row in rows
        ))
        profiles.append({
            "node_id": f"profile:{speaker_ref}",
            "layer": "profile",
            "stable_speaker_id": speaker_ref,
            "speaker_name": aliases[0] if aliases else None,
            "aliases": aliases,
            "acoustic_speaker_ids": acoustic_ids,
            "refer_ids": [row["refer_ids"][0] for row in rows],
            "registry_state_sha256": registry_sha,
            "ema_policy": ema_policy,
            "retrieval_enabled": False,
        })

    inputs = manifest.get("inputs") or {}
    return CaseBundle(
        case_id=case_id,
        case_name=root.name,
        raw=tuple(raw),
        profiles=tuple(profiles),
        source_view_sha256=view_sha,
        sidecar_sha256=sidecar_sha,
        prediction_sha256=_nonempty(
            inputs.get("identity_predictions_sha256"), "identity prediction SHA256"
        ),
        prediction_protocol=str(inputs.get("identity_prediction_protocol") or ""),
        registry_state_sha256=registry_sha,
        ema_policy=ema_policy,
        alias_is_causal_at_each_turn=causal,
    )
