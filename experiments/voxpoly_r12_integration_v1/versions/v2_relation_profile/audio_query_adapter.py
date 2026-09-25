"""Audio-only query-speaker binding for the isolated VoxPoly v2 adapter.

The formal path is waveform -> frozen ECAPA encoder -> frozen online EMA
registry -> stable speaker reference.  Question text, answer annotations,
benchmark types, gold evidence, and identity-bearing filenames are never used
to infer the asker.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SIDECAR_SCHEMA = "voxpoly.audio-query-identity.v2"
DEFAULT_MATCH_THRESHOLD = 0.35
DEFAULT_HIGH_THRESHOLD = 0.40
DEFAULT_HIGH_MARGIN = 0.02
FORBIDDEN_INPUT_KEYS = {
    "answer",
    "asker_name",
    "category",
    "evidence",
    "evidence_ids",
    "gold",
    "gold_evidence",
    "gold_evidence_ids",
    "question",
    "question_type",
    "task_type",
}


class AudioIdentityContractError(ValueError):
    """Raised when an audio identity input breaks the QA-blind contract."""


@dataclass(frozen=True)
class IdentityObservation:
    qa_id: str
    case_id: str
    stable_asker_ref: str | None
    confidence: str
    cosine: float | None
    margin: float | None
    runner_up_ref: str | None
    profile_display: str | None
    audio_sha256: str
    registry_sha256: str
    device: str
    schema_version: str = SIDECAR_SCHEMA
    inference_source: str = "query_audio_ecapa_ema_registry"
    uses_filename_identity: bool = False
    uses_question_text: bool = False
    uses_answer: bool = False
    uses_gold: bool = False


def _reject_forbidden_keys(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in FORBIDDEN_INPUT_KEYS:
                raise AudioIdentityContractError(f"forbidden identity input {path}.{key}")
            _reject_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, path=f"{path}[{index}]")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if not values or norm <= 1e-12:
        raise AudioIdentityContractError("speaker embedding is empty or zero")
    return [value / norm for value in values]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    a = _normalize(left)
    b = _normalize(right)
    if len(a) != len(b):
        raise AudioIdentityContractError("query and registry embedding dimensions differ")
    return sum(x * y for x, y in zip(a, b))


def _profile_index(
    memory: Mapping[str, Any], registry_sha256: str
) -> tuple[dict[str, str], dict[str, str]]:
    acoustic_to_stable: dict[str, str] = {}
    stable_to_display: dict[str, str] = {}
    profiles = memory.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise AudioIdentityContractError("memory has no profile registry mapping")
    for profile in profiles:
        expected_sha = str(profile.get("registry_state_sha256") or "")
        if expected_sha != registry_sha256:
            raise AudioIdentityContractError(
                "memory profile and online EMA registry SHA do not match"
            )
        stable = str(profile.get("stable_speaker_id") or "").strip()
        if not stable:
            raise AudioIdentityContractError("profile lacks stable_speaker_id")
        display = str(profile.get("speaker_name") or stable)
        stable_to_display[stable] = display
        for acoustic in profile.get("acoustic_speaker_ids") or []:
            acoustic = str(acoustic).strip()
            if not acoustic:
                continue
            prior = acoustic_to_stable.get(acoustic)
            if prior and prior != stable:
                raise AudioIdentityContractError("acoustic speaker ID maps to two profiles")
            acoustic_to_stable[acoustic] = stable
    return acoustic_to_stable, stable_to_display


def match_embedding(
    query_embedding: Sequence[float],
    *,
    acoustic_speaker_ids: Sequence[str],
    prototypes: Sequence[Sequence[float]],
    acoustic_to_stable: Mapping[str, str],
    stable_to_display: Mapping[str, str],
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    high_margin: float = DEFAULT_HIGH_MARGIN,
) -> dict[str, Any]:
    """Match one embedding without using a filename, question, or QA label."""

    if len(acoustic_speaker_ids) != len(prototypes) or not acoustic_speaker_ids:
        raise AudioIdentityContractError("registry speaker/prototype arrays are invalid")
    scored = []
    for acoustic, prototype in zip(acoustic_speaker_ids, prototypes):
        acoustic = str(acoustic)
        if acoustic not in acoustic_to_stable:
            raise AudioIdentityContractError(f"registry ID {acoustic} is absent from profiles")
        scored.append((_cosine(query_embedding, prototype), acoustic))
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_acoustic = scored[0]
    runner_score, runner_acoustic = scored[1] if len(scored) > 1 else (-1.0, "")
    margin = best_score - runner_score
    if best_score < match_threshold:
        return {
            "stable_asker_ref": None,
            "confidence": "unresolved",
            "cosine": best_score,
            "margin": margin,
            "runner_up_ref": acoustic_to_stable.get(runner_acoustic),
            "profile_display": None,
        }
    stable = acoustic_to_stable[best_acoustic]
    confidence = "high" if best_score >= high_threshold and margin >= high_margin else "low"
    return {
        "stable_asker_ref": stable,
        "confidence": confidence,
        "cosine": best_score,
        "margin": margin,
        "runner_up_ref": acoustic_to_stable.get(runner_acoustic),
        "profile_display": stable_to_display.get(stable, stable),
    }


def load_registry(path: str | os.PathLike[str]) -> tuple[list[str], list[list[float]]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise AudioIdentityContractError("numpy is required to read the EMA registry") from exc
    registry = np.load(path, allow_pickle=False)
    required = {"speaker_ids", "prototypes"}
    if not required.issubset(registry.files):
        raise AudioIdentityContractError("EMA registry lacks speaker_ids/prototypes")
    speaker_ids = [str(value) for value in registry["speaker_ids"].tolist()]
    prototypes = [[float(value) for value in row] for row in registry["prototypes"].tolist()]
    return speaker_ids, prototypes


def _load_ecapa_encoder(module_path: str | os.PathLike[str], device: str) -> Any:
    spec = importlib.util.spec_from_file_location("voxpoly_frozen_ecapa_encoder", module_path)
    if spec is None or spec.loader is None:
        raise AudioIdentityContractError("cannot load frozen ECAPA encoder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder_type = getattr(module, "EcapaEncoder", None)
    if encoder_type is None:
        raise AudioIdentityContractError("frozen module has no EcapaEncoder")
    return encoder_type(device=device)


def resolve_query_audio(
    *,
    qa_id: str,
    case_id: str,
    audio_path: str,
    memory_path: str,
    registry_path: str,
    encoder_module_path: str,
    device: str = "cpu",
) -> IdentityObservation:
    """Resolve a stable asker reference from waveform bytes only."""

    if device not in {"cpu", "cuda:1"}:
        raise AudioIdentityContractError("v2 permits only CPU or low-memory GPU1 ECAPA")
    registry_sha = sha256_file(registry_path)
    with open(memory_path, encoding="utf-8") as handle:
        memory = json.load(handle)
    acoustic_to_stable, stable_to_display = _profile_index(memory, registry_sha)
    speaker_ids, prototypes = load_registry(registry_path)
    encoder = _load_ecapa_encoder(encoder_module_path, device)
    embedding = encoder.encode(audio_path)
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    match = match_embedding(
        embedding,
        acoustic_speaker_ids=speaker_ids,
        prototypes=prototypes,
        acoustic_to_stable=acoustic_to_stable,
        stable_to_display=stable_to_display,
    )
    return IdentityObservation(
        qa_id=qa_id,
        case_id=case_id,
        stable_asker_ref=match["stable_asker_ref"],
        confidence=match["confidence"],
        cosine=match["cosine"],
        margin=match["margin"],
        runner_up_ref=match["runner_up_ref"],
        profile_display=match["profile_display"],
        audio_sha256=sha256_file(audio_path),
        registry_sha256=registry_sha,
        device=device,
    )


def write_json_atomic(path: str | os.PathLike[str], payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_sidecar(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _reject_forbidden_keys(manifest)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise AudioIdentityContractError("manifest.items must be a non-empty list")
    default_device = str(manifest.get("device") or "cpu")
    observations = []
    for item in items:
        required = {
            "qa_id", "case_id", "audio_path", "memory_path",
            "registry_path", "encoder_module_path",
        }
        missing = sorted(required - set(item))
        if missing:
            raise AudioIdentityContractError(f"identity item lacks {missing}")
        observations.append(asdict(resolve_query_audio(
            qa_id=str(item["qa_id"]),
            case_id=str(item["case_id"]),
            audio_path=str(item["audio_path"]),
            memory_path=str(item["memory_path"]),
            registry_path=str(item["registry_path"]),
            encoder_module_path=str(item["encoder_module_path"]),
            device=str(item.get("device") or default_device),
        )))
    return {
        "schema_version": SIDECAR_SCHEMA,
        "method": "waveform_ecapa_frozen_ema_registry",
        "uses_filename_identity": False,
        "uses_question_text": False,
        "uses_answer": False,
        "uses_gold": False,
        "observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    payload = build_sidecar(manifest)
    write_json_atomic(args.output, payload)
    print(json.dumps({
        "output": args.output,
        "observations": len(payload["observations"]),
        "external_llm_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
