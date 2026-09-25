#!/usr/bin/env python3
"""Prepare legacy VoxPoly cases for unified r12 memory and expanded QA evaluation.

Legacy dialogue evidence uses zero-based IDs such as ``S3:0`` and Persona
templates contain several speaker-conditioned answer versions.  This adapter
is deliberately evaluation-local: it creates a QA-free source with unified
``S3_T001`` bottom IDs, an evaluation-only expanded canonical file, and an
audio manifest with one item per actual Persona version.  It never calls an
external model and never copies QA or oracle speaker labels into the memory
construction view.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "voxpoly-legacy-r12-preparation.v1"
LEGACY_REF = re.compile(r"^(S\d+):(\d+)$")
SAFE_ROOT_FIELDS = ("schema_version", "case_id", "theme", "session_dates", "sessions")
SAFE_TURN_FIELDS = (
    "turn_id", "ordinal", "text", "timestamp",
    "image_id", "image_ids", "image_caption", "image_captions", "images",
    "audio_path", "audio_file", "audio_ref",
    "speech_act", "interaction_id", "reply_to_turn_id",
)
EVIDENCE_FIELDS = ("evidence_ids", "evidence", "gold_evidence_ids", "gold_evidence")


class LegacyPreparationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LegacyPreparationError(f"{label} must be a JSON object")
    return value


def qa_rows(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = document.get("qa")
    rows = value.get("qa_pairs") if isinstance(value, Mapping) else value
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise LegacyPreparationError("canonical QA must be a list of objects")
    return rows


def dialogues(document: Mapping[str, Any], label: str) -> dict[str, list[dict[str, Any]]]:
    value = document.get("dialogues")
    if not isinstance(value, dict) or not value:
        raise LegacyPreparationError(f"{label}.dialogues must be a nonempty session map")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for session_value, turns in value.items():
        session = str(session_value)
        if not session or not isinstance(turns, list) or not turns:
            raise LegacyPreparationError(f"invalid dialogue session in {label}: {session_value!r}")
        if any(not isinstance(turn, dict) or not str(turn.get("text") or "").strip() for turn in turns):
            raise LegacyPreparationError(f"{label}.{session} has a malformed/empty turn")
        normalized[session] = turns
    return normalized


def unified_turn_id(session: str, turn_idx: int) -> str:
    return f"{session}_T{turn_idx + 1:03d}"


def build_id_map(source_dialogues: Mapping[str, list[dict[str, Any]]]) -> dict[str, str]:
    return {
        f"{session}:{turn_idx}": unified_turn_id(session, turn_idx)
        for session, turns in source_dialogues.items()
        for turn_idx in range(len(turns))
    }


def evidence_values(row: Mapping[str, Any]) -> list[str]:
    present = [field for field in EVIDENCE_FIELDS if field in row]
    if len(present) > 1:
        raise LegacyPreparationError(f"QA row has multiple evidence fields: {present}")
    if not present:
        return []
    value = row[present[0]]
    if not isinstance(value, list):
        raise LegacyPreparationError(f"QA evidence is not a list: {value!r}")
    return [str(item) for item in value]


def translate_evidence(values: list[str], id_map: Mapping[str, str]) -> list[str]:
    translated = []
    for value in values:
        match = LEGACY_REF.fullmatch(value)
        if match is None:
            raise LegacyPreparationError(f"non-legacy evidence ID encountered: {value!r}")
        if value not in id_map:
            raise LegacyPreparationError(f"legacy evidence ID does not close to a turn: {value!r}")
        translated.append(id_map[value])
    return list(dict.fromkeys(translated))


def question_text(row: Mapping[str, Any]) -> str:
    value = str(row.get("question_text") or row.get("question") or "").strip()
    if not value:
        raise LegacyPreparationError(f"QA {row.get('qa_id')!r} has no question")
    return value


def expand_templates(
    templates: list[dict[str, Any]], id_map: Mapping[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    units: list[dict[str, Any]] = []
    template_map: list[dict[str, Any]] = []
    persona_audio: list[dict[str, str]] = []
    seen_templates: set[str] = set()
    seen_units: set[str] = set()

    for template_index, template in enumerate(templates):
        template_id = str(template.get("qa_id") or "").strip()
        if not template_id or template_id in seen_templates:
            raise LegacyPreparationError(f"missing/duplicate template QA ID: {template_id!r}")
        seen_templates.add(template_id)
        versions = template.get("versions")
        version_rows = versions if isinstance(versions, list) and versions else []
        expanded_ids: list[str] = []
        speakers: list[str] = []

        if version_rows:
            for version_index, version in enumerate(version_rows, 1):
                if not isinstance(version, dict):
                    raise LegacyPreparationError(f"{template_id} version {version_index} is malformed")
                speaker = str(version.get("speaker_name") or version.get("expected_speaker") or "").strip()
                answer = str(version.get("answer") or "").strip()
                if not speaker or not answer:
                    raise LegacyPreparationError(f"{template_id} version {version_index} lacks speaker/answer")
                unit_id = f"{template_id}__V{version_index:02d}"
                legacy_evidence = evidence_values(version)
                unit = {
                    key: copy.deepcopy(value)
                    for key, value in template.items()
                    if key not in {"qa_id", "versions", *EVIDENCE_FIELDS, "answer", "question"}
                }
                unit.update({
                    "qa_id": unit_id,
                    "template_qa_id": template_id,
                    "template_index": template_index,
                    "version_index": version_index,
                    "question_text": question_text(template),
                    "answer": answer,
                    "legacy_evidence": legacy_evidence,
                    "evidence": translate_evidence(legacy_evidence, id_map),
                    "speaker_name": speaker,
                    "expected_speaker": str(version.get("expected_speaker") or speaker),
                    "addressee": copy.deepcopy(version.get("addressee")),
                })
                if unit_id in seen_units:
                    raise LegacyPreparationError(f"duplicate expanded QA ID: {unit_id}")
                seen_units.add(unit_id)
                units.append(unit)
                expanded_ids.append(unit_id)
                speakers.append(speaker)
                persona_audio.append({
                    "qa_id": unit_id,
                    "template_qa_id": template_id,
                    "speaker_name": speaker,
                })
        else:
            answer = str(template.get("answer") or "").strip()
            if not answer:
                raise LegacyPreparationError(f"ordinary QA {template_id} lacks answer")
            legacy_evidence = evidence_values(template)
            unit = copy.deepcopy(template)
            for field in (*EVIDENCE_FIELDS, "question"):
                unit.pop(field, None)
            unit.update({
                "qa_id": template_id,
                "template_qa_id": template_id,
                "template_index": template_index,
                "version_index": None,
                "question_text": question_text(template),
                "answer": answer,
                "legacy_evidence": legacy_evidence,
                "evidence": translate_evidence(legacy_evidence, id_map),
            })
            if template_id in seen_units:
                raise LegacyPreparationError(f"duplicate expanded QA ID: {template_id}")
            seen_units.add(template_id)
            units.append(unit)
            expanded_ids.append(template_id)

        template_map.append({
            "template_index": template_index,
            "template_qa_id": template_id,
            "kind": "persona_versioned" if version_rows else "ordinary",
            "version_count": len(version_rows),
            "expanded_qa_ids": expanded_ids,
            "speaker_names": speakers,
        })
    return units, template_map, persona_audio


def safe_source(
    source: Mapping[str, Any],
    source_dialogues: Mapping[str, list[dict[str, Any]]],
    id_map: Mapping[str, str],
) -> dict[str, Any]:
    result = {key: copy.deepcopy(source[key]) for key in SAFE_ROOT_FIELDS if key in source}
    result["case_id"] = str(source.get("case_id") or "")
    result["dialogues"] = {}
    for session, turns in source_dialogues.items():
        output_turns = []
        for turn_idx, turn in enumerate(turns):
            clean = {key: copy.deepcopy(turn[key]) for key in SAFE_TURN_FIELDS if key in turn}
            clean["turn_id"] = unified_turn_id(session, turn_idx)
            clean["ordinal"] = int(turn.get("ordinal") or turn_idx + 1)
            clean["text"] = str(turn["text"])
            reply = clean.get("reply_to_turn_id")
            if reply is not None:
                reply = str(reply)
                if reply in id_map:
                    clean["reply_to_turn_id"] = id_map[reply]
                elif reply not in set(id_map.values()):
                    raise LegacyPreparationError(f"reply_to_turn_id does not close: {reply!r}")
            output_turns.append(clean)
        result["dialogues"][session] = output_turns
    return result


def require_new_outputs(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite prepared artifacts: {existing}")


def prepare(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve(strict=True)
    config = read_object(config_path, "legacy build config")
    case_id = str(config.get("case_id") or "").strip()
    expected = config.get("expected") or {}
    inputs = config.get("inputs") or {}
    outputs = config.get("outputs") or {}

    source_path = Path(inputs["all18_source"]).resolve(strict=True)
    canonical_path = Path(inputs["canonical_case"]).resolve(strict=True)
    audio_dir = Path(inputs["persona_audio_dir"]).resolve(strict=True)
    registry_path = Path(inputs["registry_state"]).resolve(strict=True)
    encoder_path = Path(inputs["encoder_module"]).resolve(strict=True)
    prepared_dir = Path(outputs["prepared_source_dir"]).resolve()
    prepared_path = prepared_dir / "full_case.json"
    preparation_manifest_path = prepared_dir / "preparation_manifest.json"
    id_map_path = Path(outputs["legacy_id_map"]).resolve()
    expanded_path = Path(outputs["expanded_canonical"]).resolve()
    question_path = Path(outputs["question_manifest"]).resolve()
    template_map_path = Path(outputs["template_unit_map"]).resolve()
    query_audio_path = Path(outputs["query_audio_manifest"]).resolve()
    require_new_outputs([
        prepared_path, preparation_manifest_path, id_map_path, expanded_path,
        question_path, template_map_path, query_audio_path,
    ])

    source = read_object(source_path, "all18 source")
    canonical = read_object(canonical_path, "legacy canonical")
    if str(source.get("case_id") or "") != case_id or str(canonical.get("case_id") or "") != case_id:
        raise LegacyPreparationError("case_id differs across config/source/canonical")
    source_dialogues = dialogues(source, "all18 source")
    canonical_dialogues = dialogues(canonical, "legacy canonical")
    if list(source_dialogues) != list(canonical_dialogues):
        raise LegacyPreparationError("session order differs between all18 and canonical")
    for session, turns in source_dialogues.items():
        other = canonical_dialogues[session]
        if len(turns) != len(other):
            raise LegacyPreparationError(f"turn count differs in {session}")
        for turn_idx, (left, right) in enumerate(zip(turns, other)):
            if str(left.get("text") or "") != str(right.get("text") or ""):
                raise LegacyPreparationError(f"text differs at {(session, turn_idx)}")

    id_map = build_id_map(source_dialogues)
    prepared = safe_source(source, source_dialogues, id_map)
    templates = qa_rows(canonical)
    units, template_map, persona_rows = expand_templates(templates, id_map)
    template_count = len(templates)
    versioned_templates = sum(row["kind"] == "persona_versioned" for row in template_map)
    persona_unit_count = len(persona_rows)
    counts = {
        "sessions": len(source_dialogues),
        "turns": len(id_map),
        "templates": template_count,
        "ordinary_templates": template_count - versioned_templates,
        "persona_templates": versioned_templates,
        "expanded_qa_units": len(units),
        "persona_query_audio": persona_unit_count,
    }
    for key, actual in counts.items():
        configured = expected.get(key)
        if configured is not None and int(configured) != actual:
            raise LegacyPreparationError(f"expected {key}={configured}, observed {actual}")

    expected_audio: set[Path] = set()
    query_items = []
    for row in persona_rows:
        path = (audio_dir / f"{row['template_qa_id']}_{row['speaker_name']}.wav").resolve()
        if not path.is_file():
            raise LegacyPreparationError(f"Persona audio is missing: {path}")
        if path in expected_audio:
            raise LegacyPreparationError(f"Persona audio is reused twice: {path}")
        expected_audio.add(path)
        query_items.append({
            "case_id": case_id,
            "qa_id": row["qa_id"],
            "audio_path": str(path),
            "case_view_dir": str(Path(outputs["case_view_dir"]).resolve()),
            "registry_path": str(registry_path),
            "encoder_module_path": str(encoder_path),
        })
    actual_audio = {path.resolve() for path in audio_dir.glob("*.wav")}
    if expected_audio != actual_audio:
        raise LegacyPreparationError(
            f"Persona audio inventory mismatch: missing={sorted(map(str, expected_audio - actual_audio))}, "
            f"extra={sorted(map(str, actual_audio - expected_audio))}"
        )

    atomic_json(prepared_path, prepared)
    expanded = {
        "schema_version": "voxpoly-legacy-expanded-evaluation.v1",
        "case_id": case_id,
        "source_canonical_sha256": sha256_file(canonical_path),
        "turn_id_policy": "legacy S#:zero-based -> unified S#_T<one-based,3 digits>",
        "session_dates": copy.deepcopy(prepared.get("session_dates") or {}),
        "sessions": copy.deepcopy(prepared.get("sessions") or []),
        "dialogues": copy.deepcopy(prepared["dialogues"]),
        "qa": {
            "qa_pairs": units,
            "stats": counts,
        },
    }
    atomic_json(expanded_path, expanded)
    question_manifest = {
        "schema_version": "voxpoly-fullcase-question-manifest.v2-expanded",
        "status": "complete",
        "case_id": case_id,
        "count": len(units),
        "template_count": template_count,
        "source_canonical_sha256": sha256_file(canonical_path),
        "expanded_canonical_sha256": sha256_file(expanded_path),
        "projection_fields": ["qa_id", "question"],
        "forbidden_retrieval_fields": [
            "answer", "gold", "evidence", "category", "four_way_category",
            "speaker_name", "expected_speaker", "template_qa_id", "version_index",
        ],
        "rows": [
            {"qa_id": str(row["qa_id"]), "question": question_text(row)}
            for row in units
        ],
    }
    atomic_json(question_path, question_manifest)
    template_payload = {
        "schema_version": "voxpoly-legacy-template-unit-map.v1",
        "case_id": case_id,
        "counts": counts,
        "template_to_units": template_map,
    }
    atomic_json(template_map_path, template_payload)
    query_manifest = {
        "schema_version": "voxpoly-audio-query-from-view-batch.v1",
        "device": "cpu",
        "expected_per_case": {case_id: persona_unit_count},
        "items": query_items,
    }
    atomic_json(query_audio_path, query_manifest)
    id_payload = {
        "schema_version": "voxpoly-legacy-bottom-id-map.v1",
        "case_id": case_id,
        "policy": "S#:zero-based -> S#_T<one-based,3 digits>",
        "count": len(id_map),
        "rows": [
            {
                "legacy_id": legacy_id,
                "turn_id": turn_id,
                "session": legacy_id.split(":", 1)[0],
                "turn_idx": int(legacy_id.split(":", 1)[1]),
            }
            for legacy_id, turn_id in id_map.items()
        ],
    }
    atomic_json(id_map_path, id_payload)

    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.manifest",
        "status": "complete",
        "zero_api": True,
        "case_id": case_id,
        "source": {
            "all18_path": str(source_path),
            "all18_sha256": sha256_file(source_path),
            "all18_root_keys": sorted(source),
            "canonical_id_audit_path": str(canonical_path),
            "canonical_id_audit_sha256": sha256_file(canonical_path),
        },
        "output": {
            "full_case": str(prepared_path),
            "full_case_sha256": sha256_file(prepared_path),
            "legacy_id_map": str(id_map_path),
            "legacy_id_map_sha256": sha256_file(id_map_path),
            "expanded_canonical": str(expanded_path),
            "expanded_canonical_sha256": sha256_file(expanded_path),
            "question_manifest": str(question_path),
            "question_manifest_sha256": sha256_file(question_path),
            "template_unit_map": str(template_map_path),
            "template_unit_map_sha256": sha256_file(template_map_path),
            "query_audio_manifest": str(query_audio_path),
            "query_audio_manifest_sha256": sha256_file(query_audio_path),
        },
        "counts": {
            "sessions": counts["sessions"],
            "turns": counts["turns"],
            "generated_turn_ids": counts["turns"],
            "canonical_id_and_text_matches": counts["turns"],
            **{key: value for key, value in counts.items() if key not in {"sessions", "turns"}},
        },
        "turn_id_policy": "legacy S#:zero-based -> unified S#_T<one-based,3 digits>",
        "removed_root_keys": sorted(set(source) - set(SAFE_ROOT_FIELDS) - {"dialogues"}),
        "retained_turn_fields": sorted({
            key for rows in prepared["dialogues"].values() for row in rows for key in row
        }),
        "invariants": {
            "qa_free": True,
            "oracle_identity_free": True,
            "coordinate_order_preserved": True,
            "source_text_preserved": True,
            "bottom_ids_unique": True,
            "canonical_id_audit_passed": True,
            "legacy_evidence_closed_before_translation": True,
            "expanded_evidence_uses_unified_bottom_ids": True,
            "persona_templates_expanded": True,
            "persona_audio_inventory_exact": True,
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
        },
    }
    atomic_json(preparation_manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.config)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"legacy preparation failed: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
