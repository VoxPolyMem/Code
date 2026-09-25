#!/usr/bin/env python3
"""Independent zero-API audit for a prepared VoxPoly r12 case-view bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from adapter import REGISTRY_MEMBERS, SAFE_ROOT_FIELDS, SAFE_TURN_FIELDS, load_case_bundle
from build_memory import atomic_json


class BundleAuditError(ValueError):
    pass


def read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BundleAuditError(f"{label} must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qa_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    qa = document.get("qa")
    rows = qa.get("qa_pairs", []) if isinstance(qa, dict) else qa
    if not isinstance(rows, list):
        raise BundleAuditError("all18 QA field is malformed")
    return rows


def verify(
    *,
    case_view_dir: Path,
    prepared_source_dir: Path,
    all18_source: Path,
    raw_prediction: Path,
    run_manifest: Path,
    registry_state: Path,
) -> dict[str, Any]:
    case_view_dir = case_view_dir.resolve(strict=True)
    prepared_source_dir = prepared_source_dir.resolve(strict=True)
    all18_source = all18_source.resolve(strict=True)
    raw_prediction = raw_prediction.resolve(strict=True)
    run_manifest = run_manifest.resolve(strict=True)
    registry_state = registry_state.resolve(strict=True)

    bundle = load_case_bundle(case_view_dir, registry_state, enabled=True)
    view = read_object(case_view_dir / "full_case.json", "case view")
    sidecar = read_object(case_view_dir / "speaker_identity_sidecar.json", "speaker sidecar")
    view_manifest = read_object(case_view_dir / "speaker_view_manifest.json", "speaker manifest")
    prepared = read_object(prepared_source_dir / "full_case.json", "prepared source")
    preparation = read_object(prepared_source_dir / "preparation_manifest.json", "preparation manifest")
    source = read_object(all18_source, "all18 source")
    prediction = read_object(raw_prediction, "raw identity prediction")
    resolved = read_object(case_view_dir / "online_identity_predictions_alias_resolved.json", "resolved aliases")
    run = read_object(run_manifest, "online speaker run manifest")

    case_id = bundle.case_id
    if any(str(doc.get("case_id") or "") != case_id for doc in (view, sidecar, source, prediction, resolved, run)):
        raise BundleAuditError("case_id differs across the preparation chain")
    if preparation.get("case_id") != case_id:
        raise BundleAuditError("preparation manifest case_id differs")
    if set(view) - SAFE_ROOT_FIELDS:
        raise BundleAuditError(f"non-whitelisted view roots: {sorted(set(view) - SAFE_ROOT_FIELDS)}")
    if "qa" in view or "qa" in prepared:
        raise BundleAuditError("QA leaked into a memory-construction source")
    if not qa_rows(source):
        raise BundleAuditError("all18 source has no QA; leak test is not meaningful")
    if preparation.get("source", {}).get("all18_sha256") != sha256_file(all18_source):
        raise BundleAuditError("all18 source hash differs from preparation manifest")
    if preparation.get("output", {}).get("full_case_sha256") != sha256_file(prepared_source_dir / "full_case.json"):
        raise BundleAuditError("prepared source hash differs from its manifest")
    invariants = preparation.get("invariants") or {}
    for key in ("qa_free", "oracle_identity_free", "coordinate_order_preserved", "source_text_preserved", "bottom_ids_unique", "canonical_id_audit_passed"):
        if invariants.get(key) is not True:
            raise BundleAuditError(f"preparation invariant is not true: {key}")

    side_by_coordinate: dict[tuple[str, int], dict[str, Any]] = {}
    for row in sidecar.get("turns") or []:
        key = (str(row.get("session")), int(row.get("turn_idx")))
        if key in side_by_coordinate:
            raise BundleAuditError(f"duplicate sidecar coordinate: {key}")
        side_by_coordinate[key] = row
    prediction_by_coordinate: dict[tuple[str, int], dict[str, Any]] = {}
    for row in resolved.get("predictions_frozen") or []:
        key = (str(row.get("session")), int(row.get("turn_idx")))
        if key in prediction_by_coordinate:
            raise BundleAuditError(f"duplicate prediction coordinate: {key}")
        prediction_by_coordinate[key] = row

    coordinates: list[tuple[str, int]] = []
    ids: list[str] = []
    for session_value, source_turns in (source.get("dialogues") or {}).items():
        session = str(session_value)
        view_turns = (view.get("dialogues") or {}).get(session)
        prepared_turns = (prepared.get("dialogues") or {}).get(session)
        if not isinstance(view_turns, list) or not isinstance(prepared_turns, list):
            raise BundleAuditError(f"session missing after preparation: {session}")
        if len(source_turns) != len(view_turns) or len(source_turns) != len(prepared_turns):
            raise BundleAuditError(f"turn count changed in session {session}")
        for turn_idx, source_turn in enumerate(source_turns):
            key = (session, turn_idx)
            coordinates.append(key)
            if key not in side_by_coordinate or key not in prediction_by_coordinate:
                raise BundleAuditError(f"identity coordinate missing: {key}")
            expected_id = f"{session}_T{turn_idx + 1:03d}"
            prepared_turn = prepared_turns[turn_idx]
            view_turn = view_turns[turn_idx]
            if prepared_turn.get("turn_id") != expected_id or view_turn.get("turn_id") != expected_id:
                raise BundleAuditError(f"bottom turn_id mismatch at {key}")
            if prepared_turn.get("text") != source_turn.get("text") or view_turn.get("text") != source_turn.get("text"):
                raise BundleAuditError(f"source text changed at {key}")
            if set(view_turn) - SAFE_TURN_FIELDS:
                raise BundleAuditError(f"non-whitelisted fields at {key}")
            side = side_by_coordinate[key]
            predicted = prediction_by_coordinate[key]
            if str(side.get("turn_id")) != expected_id:
                raise BundleAuditError(f"sidecar turn_id mismatch at {key}")
            if side.get("stable_identity_id") != predicted.get("stable_identity_id"):
                raise BundleAuditError(f"stable identity changed at {key}")
            if side.get("acoustic_speaker_id") != predicted.get("acoustic_speaker_id"):
                raise BundleAuditError(f"acoustic identity changed at {key}")
            if str(view_turn.get("speaker_id")) != str(side.get("stable_identity_id")):
                raise BundleAuditError(f"view speaker_id differs from sidecar at {key}")
            ids.append(expected_id)
    if len(ids) != len(set(ids)):
        raise BundleAuditError("bottom IDs are not unique")
    if set(coordinates) != set(side_by_coordinate) or set(coordinates) != set(prediction_by_coordinate):
        raise BundleAuditError("coordinate sets are not exactly equal")

    policy = prediction.get("policy")
    if not isinstance(policy, dict) or policy != run.get("parameters"):
        raise BundleAuditError("prediction policy differs from run manifest parameters")
    if policy != view_manifest.get("inputs", {}).get("identity_prediction_policy"):
        raise BundleAuditError("view EMA policy differs from frozen prediction policy")
    if prediction.get("language_model_calls") != 0 or run.get("language_model_calls") != 0:
        raise BundleAuditError("speaker inference was not zero-LLM")
    alias = resolved.get("alias_resolver") or {}
    if alias.get("language_model_calls") != 0:
        raise BundleAuditError("final alias resolution was not zero-LLM")
    if alias.get("input_prediction_sha256") != sha256_file(raw_prediction):
        raise BundleAuditError("final alias input hash differs from raw prediction")
    if view_manifest.get("inputs", {}).get("identity_predictions_sha256") != sha256_file(case_view_dir / "online_identity_predictions_alias_resolved.json"):
        raise BundleAuditError("resolved alias hash differs from speaker manifest")
    if str(Path(run.get("artifacts", {}).get("registry_state", "")).resolve()) != str(registry_state):
        raise BundleAuditError("run manifest points to a different EMA registry")

    with np.load(registry_state, allow_pickle=False) as registry:
        if {f"{name}.npy" for name in registry.files} != REGISTRY_MEMBERS:
            raise BundleAuditError(f"EMA registry member mismatch: {registry.files}")
        speaker_ids = registry["speaker_ids"]
        assigned_turns = registry["assigned_turns"]
        updates = registry["updates"]
        prototypes = registry["prototypes"]
        if speaker_ids.ndim != 1 or assigned_turns.ndim != 1 or updates.ndim != 1 or prototypes.ndim != 2:
            raise BundleAuditError("EMA registry arrays have invalid ranks")
        if not (
            len(speaker_ids)
            == len(assigned_turns)
            == len(updates)
            == prototypes.shape[0]
        ):
            raise BundleAuditError("EMA registry array cardinalities differ")
        registry_acoustic_ids = [str(value) for value in speaker_ids.tolist()]
        profile_acoustic_ids = [
            str(acoustic_id)
            for profile in bundle.profiles
            for acoustic_id in profile.get("acoustic_speaker_ids") or []
        ]
        if len(profile_acoustic_ids) != len(set(profile_acoustic_ids)):
            raise BundleAuditError(
                "one acoustic speaker ID is assigned to multiple stable profiles"
            )
        if set(profile_acoustic_ids) != set(registry_acoustic_ids):
            raise BundleAuditError(
                "stable profiles do not exactly cover the EMA acoustic registry"
            )
        if int(np.asarray(assigned_turns).sum()) != len(coordinates):
            raise BundleAuditError("EMA registry assigned-turn counts do not cover the case")
        if np.any(np.asarray(updates) < 0) or not np.isfinite(prototypes).all():
            raise BundleAuditError("EMA registry contains invalid update counts or prototypes")
        norms = np.linalg.norm(prototypes.astype(float), axis=1)
        if np.any(norms <= 0):
            raise BundleAuditError("EMA registry contains an empty prototype")

    return {
        "schema_version": "voxpoly-r12-case-view-audit.v1",
        "status": "complete",
        "zero_api": True,
        "case_id": case_id,
        "sessions": len(source.get("dialogues") or {}),
        "turns": len(coordinates),
        "profiles": len(bundle.profiles),
        "source_qa_rows_excluded": len(qa_rows(source)),
        "coordinates_exact": True,
        "bottom_ids_unique_and_canonical": True,
        "all18_text_preserved": True,
        "qa_gold_leakage": False,
        "oracle_identity_leakage": False,
        "speaker_inference_language_model_calls": 0,
        "final_alias_language_model_calls": 0,
        "ema_policy": policy,
        "ema_registry_sha256": sha256_file(registry_state),
        "ema_registry_acoustic_profiles": len(registry_acoustic_ids),
        "stable_identity_profiles": len(bundle.profiles),
        "ema_assigned_turns": len(coordinates),
        "alias_timing": "batch_final",
        "profile_retrieval_enabled": False,
        "hashes": {
            "all18_source": sha256_file(all18_source),
            "prepared_source": sha256_file(prepared_source_dir / "full_case.json"),
            "case_view": sha256_file(case_view_dir / "full_case.json"),
            "speaker_sidecar": sha256_file(case_view_dir / "speaker_identity_sidecar.json"),
            "raw_identity_prediction": sha256_file(raw_prediction),
            "resolved_identity_prediction": sha256_file(case_view_dir / "online_identity_predictions_alias_resolved.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-view-dir", required=True, type=Path)
    parser.add_argument("--prepared-source-dir", required=True, type=Path)
    parser.add_argument("--all18-source", required=True, type=Path)
    parser.add_argument("--raw-prediction", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--registry-state", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = verify(
            case_view_dir=args.case_view_dir,
            prepared_source_dir=args.prepared_source_dir,
            all18_source=args.all18_source,
            raw_prediction=args.raw_prediction,
            run_manifest=args.run_manifest,
            registry_state=args.registry_state,
        )
    except (OSError, json.JSONDecodeError, BundleAuditError, ValueError) as exc:
        print(f"case-view audit failed: {exc}", file=sys.stderr)
        return 2
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
