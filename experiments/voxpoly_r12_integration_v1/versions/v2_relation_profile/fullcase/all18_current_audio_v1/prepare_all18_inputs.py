#!/usr/bin/env python3
"""Prepare and verify QA-blind, audio-identified inputs for all VoxPoly cases.

This is a zero-LLM preparation step for the isolated all18 current-audio run.
Dialogue content is read from the benchmark-provided text projection.  Turn
speaker identities come only from the frozen online ECAPA/EMA sidecar and
final alias resolver; addressees are deliberately *not* supplied here and are
instead inferred by the memory writer from dialogue context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[5]
AGENT_MEMORY = Path(os.environ.get("AGENT_MEMORY_ROOT", PACKAGE_ROOT))
WORKSPACE = PACKAGE_ROOT
EXPERIMENT = WORKSPACE / "experiments/voxpoly_r12_integration_v1"
SPEAKER_EXPERIMENT = WORKSPACE / "experiments/voxpoly_speaker_ablation_v1"
VOXPOLY = Path(os.environ.get(
    "VOXPOLYBENCH_HOME", AGENT_MEMORY / "audio_mem_bench/VoxPolyBench"
))
ALL18 = VOXPOLY / "evaluation/versions/id_label_natural_names_aigc_gpt41mini_all18_20260913_v1"
SPEAKER_ROOT = Path(os.environ.get(
    "VOXPOLY_SPEAKER_ROOT", VOXPOLY / "logs/speaker/online_all_cases_v1"
))
PYTHON = Path(os.environ.get("PYTHON_BIN", sys.executable))

LEGACY = ("001", "002", "003", "005", "006", "009", "010", "012")
NEW = ("013", "014", "015", "016", "019", "020", "022", "023", "024", "025")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(*args: str | Path) -> None:
    command = [str(value) for value in args]
    print("[zero-api]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def canonical(case_id: str, is_legacy: bool) -> Path:
    if is_legacy:
        return VOXPOLY / ("output" if case_id in {"CASE_G_001", "CASE_G_002"} else "output/gen2") / case_id / "full_case.json"
    return VOXPOLY / "output/gen3" / case_id / "full_case.json"


def source(case_id: str) -> Path:
    return ALL18 / "inputs" / case_id / "full_case.json"


def online_paths(case_id: str) -> dict[str, Path]:
    online = SPEAKER_ROOT / case_id
    return {
        "raw": online / "online_identity_predictions.json",
        "run": online / "run_manifest.json",
        "registry": online / "online_registry_state.npz",
        "resolved": SPEAKER_EXPERIMENT / "artifacts/final_alias_v1" / case_id / "online_identity_predictions_alias_resolved.json",
    }


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required input(s): " + "; ".join(missing))


def make_view(run_root: Path, case_id: str, prepared: Path) -> tuple[Path, Path]:
    paths = online_paths(case_id)
    view = run_root / "case_views" / case_id
    view.mkdir(parents=True, exist_ok=True)
    resolved_copy = view / "online_identity_predictions_alias_resolved.json"
    if not resolved_copy.exists():
        shutil.copy2(paths["resolved"], resolved_copy)
    invoke(
        PYTHON,
        SPEAKER_EXPERIMENT / "make_case_view.py",
        "--source-case", prepared / "full_case.json",
        "--output-dir", view,
        "--config", SPEAKER_EXPERIMENT / "configs/online_predicted.json",
        "--identity-predictions", resolved_copy,
    )
    report = run_root / "audits" / f"{case_id}.bundle.json"
    invoke(
        PYTHON,
        EXPERIMENT / "verify_case_view_bundle.py",
        "--case-view-dir", view,
        "--prepared-source-dir", prepared,
        "--all18-source", source(case_id),
        "--raw-prediction", paths["raw"],
        "--run-manifest", paths["run"],
        "--registry-state", paths["registry"],
        "--report", report,
    )
    preflight = run_root / "audits" / f"{case_id}.preflight.json"
    invoke(
        PYTHON,
        EXPERIMENT / "preflight.py",
        "--enable-experimental-audio",
        "--case-view-dir", view,
        "--registry-state", paths["registry"],
        "--out", preflight,
    )
    return view, report


def prepare_new(run_root: Path, case_id: str) -> dict[str, Any]:
    prepared = run_root / "prepared" / case_id
    invoke(
        PYTHON,
        EXPERIMENT / "prepare_qa_free_source.py",
        "--source", source(case_id),
        "--canonical-id-source", canonical(case_id, False),
        "--output-dir", prepared,
    )
    view, audit = make_view(run_root, case_id, prepared)
    questions = run_root / "questions" / f"{case_id}.json"
    invoke(
        PYTHON,
        EXPERIMENT / "versions/v2_relation_profile/fullcase/prepare_question_manifest.py",
        "--case", canonical(case_id, False),
        "--output", questions,
    )
    persona = json.loads((canonical(case_id, False).parent / "audio_persona/persona_manifest.json").read_text())
    paths = online_paths(case_id)
    query_items = [
        {
            "case_id": case_id,
            "qa_id": str(item["qa_id"]),
            "audio_path": str(Path(item["audio_path"]).resolve()),
            "case_view_dir": str(view),
            "registry_path": str(paths["registry"]),
            "encoder_module_path": str(VOXPOLY / "tools/speaker/ecapa_encoder.py"),
        }
        for item in persona.get("items") or []
    ]
    if len(query_items) != 10 or len({item["qa_id"] for item in query_items}) != 10:
        raise ValueError(f"{case_id} does not expose exactly ten distinct Persona waveforms")
    query_manifest = run_root / "query_audio" / f"{case_id}.json"
    atomic_json(query_manifest, {
        "schema_version": "voxpoly-audio-query-from-view-batch.v1",
        "device": "cpu",
        "expected_per_case": {case_id: len(query_items)},
        "items": query_items,
    })
    return {
        "case_id": case_id,
        "collection": "new",
        "canonical_case": str(canonical(case_id, False)),
        "case_view_dir": str(view),
        "prepared_source_dir": str(prepared),
        "bundle_audit": str(audit),
        "question_manifest": str(questions),
        "query_audio_manifest": str(query_manifest),
        "qa_count": 75,
    }


def prepare_legacy(run_root: Path, case_id: str) -> dict[str, Any]:
    canonical_case = canonical(case_id, True)
    paths = online_paths(case_id)
    view = run_root / "case_views" / case_id
    prepared = view / "prepared_source"
    legacy_config = run_root / "legacy_configs" / f"{case_id}.json"
    outputs = {
        "case_view_dir": str(view),
        "prepared_source_dir": str(prepared),
        "legacy_id_map": str(run_root / "legacy_maps" / f"{case_id}.turn_ids.json"),
        "expanded_canonical": str(run_root / "expanded_canonical" / f"{case_id}.json"),
        "question_manifest": str(run_root / "questions" / f"{case_id}.expanded.json"),
        "template_unit_map": str(run_root / "template_maps" / f"{case_id}.json"),
        "query_audio_manifest": str(run_root / "query_audio" / f"{case_id}.json"),
    }
    atomic_json(legacy_config, {
        "schema_version": "voxpoly-r12-legacy-fullcase-build-config.v1",
        "case_id": case_id,
        "inputs": {
            "all18_source": str(source(case_id)),
            "canonical_case": str(canonical_case),
            "persona_audio_dir": str(canonical_case.parent / "audio_persona"),
            "registry_state": str(paths["registry"]),
            "encoder_module": str(VOXPOLY / "tools/speaker/ecapa_encoder.py"),
        },
        "outputs": outputs,
    })
    invoke(
        PYTHON,
        EXPERIMENT / "versions/v2_relation_profile/fullcase/prepare_legacy_fullcase.py",
        "--config", legacy_config,
    )
    view, audit = make_view(run_root, case_id, prepared)
    expanded = json.loads(Path(outputs["expanded_canonical"]).read_text())
    qa_count = len((expanded.get("qa") or {}).get("qa_pairs") or [])
    if qa_count <= 0:
        raise ValueError(f"{case_id} legacy QA expansion is empty")
    return {
        "case_id": case_id,
        "collection": "legacy",
        "canonical_case": outputs["expanded_canonical"],
        "case_view_dir": str(view),
        "prepared_source_dir": str(prepared),
        "bundle_audit": str(audit),
        "question_manifest": outputs["question_manifest"],
        "query_audio_manifest": outputs["query_audio_manifest"],
        "qa_count": qa_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"refusing to write into a nonempty run root: {run_root}")
    require_files([PYTHON, EXPERIMENT / "build_memory.py", SPEAKER_EXPERIMENT / "make_case_view.py"])
    for suffix in (*LEGACY, *NEW):
        case_id = f"CASE_G_{suffix}"
        require_files([
            source(case_id), canonical(case_id, suffix in LEGACY),
            *online_paths(case_id).values(),
        ])
    run_root.mkdir(parents=True)
    atomic_json(run_root / "RUN_PROTOCOL.json", {
        "schema_version": "voxpoly-all18-current-audio-run.v1",
        "status": "preparing",
        "top_k": 30,
        "text_input": {
            "mode": "benchmark_provided_text",
            "asr_enabled": False,
            "asr_interface": "transcribe(audio_path: str) -> str",
            "note": "ASR is intentionally not invoked in this GT-text run.",
        },
        "audio_identity": {
            "turn_speaker": "online ECAPA + guarded EMA + final alias resolver",
            "query_asker": "query waveform ECAPA against frozen EMA registry",
            "external_llm_calls": 0,
        },
        "addressee": "LLM-inferred from observable dialogue context during memory construction",
        "qa_or_gold_used_for_memory_or_retrieval": False,
    })
    rows = []
    for suffix in LEGACY:
        rows.append(prepare_legacy(run_root, f"CASE_G_{suffix}"))
    for suffix in NEW:
        rows.append(prepare_new(run_root, f"CASE_G_{suffix}"))
    atomic_json(run_root / "all18_inputs.json", {
        "schema_version": "voxpoly-all18-current-audio-inputs.v1",
        "status": "complete",
        "case_count": len(rows),
        "qa_count_total": sum(int(row["qa_count"]) for row in rows),
        "cases": rows,
    })
    protocol_path = run_root / "RUN_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["status"] = "zero_api_inputs_verified"
    protocol["input_manifest_sha256"] = sha256(run_root / "all18_inputs.json")
    atomic_json(protocol_path, protocol)
    print(json.dumps({
        "run_root": str(run_root), "case_count": len(rows),
        "qa_count_total": sum(int(row["qa_count"]) for row in rows),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
