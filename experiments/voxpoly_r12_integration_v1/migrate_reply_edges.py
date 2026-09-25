#!/usr/bin/env python3
"""Create a zero-API candidate copy with validated observable reply edges.

This migration never edits its source memory.  It is intended for an already
built Audio r12 memory whose QA-free case view contained ``reply_to_turn_id``
but whose legacy adapter omitted that optional field from raw nodes.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any

from adapter import (
    ADAPTER_VERSION,
    AdapterContractError,
    CaseBundle,
    load_case_bundle,
    sha256_file,
)


MIGRATION_VERSION = "voxpoly-reply-edge-migration-v1"


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterContractError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterContractError(f"{label} must be an object")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _validate_source(memory: dict[str, Any], bundle: CaseBundle) -> None:
    if memory.get("case_id") != bundle.case_id:
        raise AdapterContractError("source memory and case view have different case_id")
    meta = memory.get("meta")
    fingerprint = meta.get("fingerprint") if isinstance(meta, dict) else None
    if not isinstance(fingerprint, dict):
        raise AdapterContractError("source memory has no reproducibility fingerprint")
    expected = {
        "source_view_sha256": bundle.source_view_sha256,
        "speaker_sidecar_sha256": bundle.sidecar_sha256,
        "identity_prediction_sha256": bundle.prediction_sha256,
        "registry_state_sha256": bundle.registry_state_sha256,
    }
    for key, value in expected.items():
        if fingerprint.get(key) != value:
            raise AdapterContractError(
                f"source memory fingerprint differs for {key}: "
                f"{fingerprint.get(key)!r} != {value!r}"
            )

    source_raw = memory.get("raw")
    if not isinstance(source_raw, list) or not all(isinstance(row, dict) for row in source_raw):
        raise AdapterContractError("source memory raw layer must be a list of objects")
    bundle_by_id = {str(row["refer_ids"][0]): row for row in bundle.raw}
    source_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(source_raw):
        refs = row.get("refer_ids")
        if not isinstance(refs, list) or len(refs) != 1 or not str(refs[0]).strip():
            raise AdapterContractError(f"source raw[{index}] has invalid self provenance")
        ref = str(refs[0]).strip()
        if ref in source_by_id:
            raise AdapterContractError(f"source memory duplicates raw ID: {ref}")
        source_by_id[ref] = row
    if set(source_by_id) != set(bundle_by_id):
        raise AdapterContractError("source memory and validated bundle have different raw IDs")

    # Prove that this is a field-only enrichment, not a silent raw-data rewrite.
    for ref, source_row in source_by_id.items():
        bundle_row = bundle_by_id[ref]
        for key, value in bundle_row.items():
            if key == "reply_to_turn_id":
                continue
            if source_row.get(key) != value:
                raise AdapterContractError(
                    f"source memory differs from validated bundle at {ref}.{key}"
                )


def migrate_reply_edges(
    *,
    source_memory_path: Path,
    case_view_dir: Path,
    registry_state: Path,
    output_path: Path,
    enabled: bool = False,
) -> dict[str, Any]:
    if not enabled:
        raise AdapterContractError(
            "reply-edge migration is default-off; pass the explicit enable flag"
        )
    source_path = source_memory_path.expanduser().resolve(strict=True)
    output = output_path.expanduser().resolve(strict=False)
    if output == source_path:
        raise AdapterContractError("migration output must differ from the frozen source")
    if output.exists():
        raise AdapterContractError(f"refusing to overwrite migration output: {output}")

    source_sha256 = sha256_file(source_path)
    source = _read_object(source_path, "source memory")
    bundle = load_case_bundle(
        case_view_dir,
        registry_state,
        enabled=True,
    )
    _validate_source(source, bundle)

    migrated = copy.deepcopy(source)
    edges = {
        str(row["refer_ids"][0]): row["reply_to_turn_id"] for row in bundle.raw
    }
    for row in migrated["raw"]:
        row["reply_to_turn_id"] = edges[str(row["refer_ids"][0])]

    meta = migrated["meta"]
    fingerprint = meta["fingerprint"]
    previous_adapter = fingerprint.get("adapter_version")
    fingerprint["adapter_version"] = ADAPTER_VERSION
    meta["reply_edge_migration"] = {
        "schema_version": MIGRATION_VERSION,
        "source_memory_sha256": source_sha256,
        "source_adapter_version": previous_adapter,
        "target_adapter_version": ADAPTER_VERSION,
        "reply_edges": sum(value is not None for value in edges.values()),
        "llm_calls": 0,
        "api_calls": 0,
    }
    _atomic_json(output, migrated)
    return {
        "schema_version": MIGRATION_VERSION,
        "status": "complete",
        "case_id": bundle.case_id,
        "source_memory_sha256": source_sha256,
        "output_memory_sha256": sha256_file(output),
        "raw_turns": len(migrated["raw"]),
        "reply_edges": sum(value is not None for value in edges.values()),
        "llm_calls": 0,
        "api_calls": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-experimental-audio", action="store_true")
    parser.add_argument("--source-memory", required=True, type=Path)
    parser.add_argument("--case-view-dir", required=True, type=Path)
    parser.add_argument("--registry-state", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = migrate_reply_edges(
            source_memory_path=args.source_memory,
            case_view_dir=args.case_view_dir,
            registry_state=args.registry_state,
            output_path=args.out,
            enabled=args.enable_experimental_audio,
        )
    except (OSError, ValueError) as exc:
        print(f"reply-edge migration failed: {exc}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
