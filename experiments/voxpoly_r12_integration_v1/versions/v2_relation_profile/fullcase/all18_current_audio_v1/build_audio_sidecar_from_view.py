#!/usr/bin/env python3
"""Build a zero-LLM query-speaker sidecar from a validated QA-free case view.

This is the pre-memory counterpart of ``build_audio_sidecar.py``.  It reads
the same frozen ECAPA registry mapping from the case-view bundle, allowing
legacy cases with more than ten speaker-conditioned QA units to be validated
before paid memory construction.  Audio filenames are never parsed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping


# This copy lives one level below ``fullcase`` in an isolated all18 run.
# Keep imports pointed at the shared, frozen v2 relation helpers and adapter.
HERE = Path(__file__).resolve().parent
V2_ROOT = HERE.parents[1]
EXPERIMENT = HERE.parents[3]
for candidate in (V2_ROOT, EXPERIMENT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from adapter import load_case_bundle  # noqa: E402
from audio_query_adapter import (  # noqa: E402
    AudioIdentityContractError,
    IdentityObservation,
    _load_ecapa_encoder,
    _reject_forbidden_keys,
    load_registry,
    match_embedding,
    sha256_file,
    write_json_atomic,
)


def profile_index(profiles: list[Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    acoustic_to_stable: dict[str, str] = {}
    stable_to_display: dict[str, str] = {}
    for profile in profiles:
        stable = str(profile.get("stable_speaker_id") or "").strip()
        if not stable:
            raise AudioIdentityContractError("case-view profile lacks stable_speaker_id")
        stable_to_display[stable] = str(profile.get("speaker_name") or stable)
        for value in profile.get("acoustic_speaker_ids") or []:
            acoustic = str(value).strip()
            prior = acoustic_to_stable.get(acoustic)
            if not acoustic or (prior is not None and prior != stable):
                raise AudioIdentityContractError(
                    f"invalid acoustic-to-stable profile mapping: {acoustic!r}"
                )
            acoustic_to_stable[acoustic] = stable
    return acoustic_to_stable, stable_to_display


def build(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _reject_forbidden_keys(manifest)
    items = manifest.get("items")
    expected = manifest.get("expected_per_case")
    if not isinstance(items, list) or not items or not isinstance(expected, Mapping):
        raise AudioIdentityContractError("manifest requires items and expected_per_case")
    device = str(manifest.get("device") or "cpu")
    if device not in {"cpu", "cuda:1"}:
        raise AudioIdentityContractError("only CPU or cuda:1 is allowed")

    encoders: dict[tuple[str, str], Any] = {}
    bundles: dict[tuple[str, str], tuple[Any, list[str], list[list[float]], dict[str, str], dict[str, str]]] = {}
    observations = []
    seen: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}

    for index, item in enumerate(items):
        required = {
            "qa_id", "case_id", "audio_path", "case_view_dir",
            "registry_path", "encoder_module_path",
        }
        missing = required - set(item)
        if missing:
            raise AudioIdentityContractError(f"item {index} lacks {sorted(missing)}")
        case_id = str(item["case_id"])
        qa_id = str(item["qa_id"])
        key = (case_id, qa_id)
        if key in seen:
            raise AudioIdentityContractError(f"duplicate case/QA: {key}")
        seen.add(key)
        item_device = str(item.get("device") or device)
        if item_device not in {"cpu", "cuda:1"}:
            raise AudioIdentityContractError("only CPU or cuda:1 is allowed")

        encoder_key = (str(item["encoder_module_path"]), item_device)
        if encoder_key not in encoders:
            encoders[encoder_key] = _load_ecapa_encoder(*encoder_key)

        bundle_key = (str(item["case_view_dir"]), str(item["registry_path"]))
        if bundle_key not in bundles:
            registry_sha = sha256_file(item["registry_path"])
            bundle = load_case_bundle(
                Path(item["case_view_dir"]), Path(item["registry_path"]), enabled=True
            )
            acoustic_to_stable, stable_to_display = profile_index(list(bundle.profiles))
            speaker_ids, prototypes = load_registry(item["registry_path"])
            if set(speaker_ids) != set(acoustic_to_stable):
                raise AudioIdentityContractError(
                    "case-view profiles do not exactly cover the EMA acoustic registry"
                )
            bundles[bundle_key] = (
                bundle, speaker_ids, prototypes, acoustic_to_stable, stable_to_display,
            )
        bundle, speaker_ids, prototypes, acoustic_to_stable, stable_to_display = bundles[bundle_key]
        if bundle.case_id != case_id:
            raise AudioIdentityContractError(f"case-view case_id differs for {key}")

        audio_path = Path(str(item["audio_path"])).resolve(strict=True)
        embedding = encoders[encoder_key].encode(str(audio_path))
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        matched = match_embedding(
            embedding,
            acoustic_speaker_ids=speaker_ids,
            prototypes=prototypes,
            acoustic_to_stable=acoustic_to_stable,
            stable_to_display=stable_to_display,
        )
        observations.append(asdict(IdentityObservation(
            qa_id=qa_id,
            case_id=case_id,
            stable_asker_ref=matched["stable_asker_ref"],
            confidence=matched["confidence"],
            cosine=matched["cosine"],
            margin=matched["margin"],
            runner_up_ref=matched["runner_up_ref"],
            profile_display=matched["profile_display"],
            audio_sha256=sha256_file(audio_path),
            registry_sha256=sha256_file(item["registry_path"]),
            device=item_device,
        )))
        counts[case_id] = counts.get(case_id, 0) + 1

    normalized_expected = {str(key): int(value) for key, value in expected.items()}
    if counts != normalized_expected:
        raise AudioIdentityContractError(
            f"query-audio counts differ: actual={counts}, expected={normalized_expected}"
        )
    return {
        "schema_version": "voxpoly.audio-query-identity.v2",
        "method": "waveform_ecapa_frozen_ema_registry_case_view_profile_mapping",
        "uses_filename_identity": False,
        "uses_question_text": False,
        "uses_answer": False,
        "uses_gold": False,
        "external_llm_calls": 0,
        "observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite identity sidecar: {args.output}")
        return 2
    try:
        payload = build(json.loads(args.manifest.read_text(encoding="utf-8")))
        write_json_atomic(args.output, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"audio sidecar failed: {exc}")
        return 2
    print(json.dumps({
        "output": str(args.output),
        "observations": len(payload["observations"]),
        "confidence": {
            label: sum(row["confidence"] == label for row in payload["observations"])
            for label in ("high", "low", "unresolved")
        },
        "external_llm_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
