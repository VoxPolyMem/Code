#!/usr/bin/env python3
"""Independent zero-API gate for the completed G014 r12 memory artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_memory import PROMPT_VERSION, atomic_json


class MemoryVerificationError(ValueError):
    pass


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MemoryVerificationError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MemoryVerificationError(f"{label} must be an object")
    return value


def verify_memory(
    memory_path: Path,
    checkpoint_dir: Path,
    preflight_path: Path,
    *,
    expected_turns: int,
    expected_windows: int,
    expected_profiles: int,
) -> dict[str, Any]:
    memory = read_object(memory_path, "memory")
    preflight = read_object(preflight_path, "preflight")
    if memory.get("schema_version") != PROMPT_VERSION:
        raise MemoryVerificationError("memory prompt/schema version mismatch")
    meta = memory.get("meta")
    if not isinstance(meta, dict):
        raise MemoryVerificationError("memory meta must be an object")
    fingerprint = meta.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise MemoryVerificationError("memory fingerprint is missing")
    expected_fingerprint = {
        "prompt_version": PROMPT_VERSION,
        "source_view_sha256": preflight.get("source_view_sha256"),
        "speaker_sidecar_sha256": preflight.get("sidecar_sha256"),
        "identity_prediction_sha256": preflight.get("identity_prediction_sha256"),
        "identity_prediction_protocol": preflight.get("prediction_protocol"),
        "registry_state_sha256": preflight.get("registry_state_sha256"),
        "model": "gpt-4.1-mini",
        "window_size": 12,
        "stride": 6,
        "role_metadata_in_retrieval_text": False,
        "identity_retrieval_enabled": False,
    }
    for key, expected in expected_fingerprint.items():
        if fingerprint.get(key) != expected:
            raise MemoryVerificationError(
                f"fingerprint mismatch for {key}: {fingerprint.get(key)!r} != {expected!r}"
            )
    raw = memory.get("raw")
    facts = memory.get("facts")
    collections = memory.get("collections")
    profiles = memory.get("profiles")
    if not all(isinstance(rows, list) for rows in (raw, facts, collections, profiles)):
        raise MemoryVerificationError("all four memory layers must be lists")
    if len(raw) != expected_turns or int(meta.get("turns", -1)) != expected_turns:
        raise MemoryVerificationError(f"raw turn count mismatch: {len(raw)}/{expected_turns}")
    if len(profiles) != expected_profiles or int(meta.get("profiles", -1)) != expected_profiles:
        raise MemoryVerificationError(
            f"profile count mismatch: {len(profiles)}/{expected_profiles}"
        )
    if int(meta.get("complete_windows", -1)) != expected_windows:
        raise MemoryVerificationError("memory complete-window count mismatch")

    raw_by_ref: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(raw):
        refs = row.get("refer_ids") if isinstance(row, dict) else None
        if not isinstance(refs, list) or len(refs) != 1 or not str(refs[0]):
            raise MemoryVerificationError(f"raw[{index}] has invalid self provenance")
        ref = str(refs[0])
        if ref in raw_by_ref:
            raise MemoryVerificationError(f"duplicate raw refer ID: {ref}")
        if not str(row.get("speaker_ref") or ""):
            raise MemoryVerificationError(f"raw[{index}] has no predicted speaker_ref")
        raw_by_ref[ref] = row
    for index, row in enumerate(raw):
        reply_to_turn_id = row.get("reply_to_turn_id")
        if reply_to_turn_id is not None and reply_to_turn_id not in raw_by_ref:
            raise MemoryVerificationError(
                f"raw[{index}] replies to unknown bottom turn: {reply_to_turn_id}"
            )
    profile_ids = {
        str(row.get("stable_speaker_id") or "") for row in profiles if isinstance(row, dict)
    }
    if "" in profile_ids or len(profile_ids) != expected_profiles:
        raise MemoryVerificationError("profile stable IDs are missing or duplicated")
    for row in profiles:
        if row.get("retrieval_enabled") is not False:
            raise MemoryVerificationError("profile retrieval must remain disabled")
        if not set(row.get("refer_ids") or []).issubset(raw_by_ref):
            raise MemoryVerificationError("profile refer closure failed")
        if "prototypes" in row:
            raise MemoryVerificationError("voice prototype vectors leaked into JSON")

    fact_ids: set[str] = set()
    cross_turn = 0
    addressee_unknown = 0
    addressee_group = 0
    for index, fact in enumerate(facts):
        node_id = str(fact.get("node_id") or "")
        refs = [str(value) for value in fact.get("refer_ids") or []]
        if not node_id or node_id in fact_ids or not refs or not set(refs).issubset(raw_by_ref):
            raise MemoryVerificationError(f"fact[{index}] identity/refer closure failed")
        fact_ids.add(node_id)
        if fact.get("retrieval_text") != fact.get("text"):
            raise MemoryVerificationError(f"fact[{index}] pollutes content retrieval text")
        cited_speakers = list(dict.fromkeys(raw_by_ref[ref]["speaker_ref"] for ref in refs))
        source_ref = str(fact.get("source_speaker_ref") or "")
        if source_ref not in cited_speakers or fact.get("speaker_ref") != source_ref:
            raise MemoryVerificationError(f"fact[{index}] principal speaker is ungrounded")
        if fact.get("evidence_speaker_refs") != cited_speakers:
            raise MemoryVerificationError(f"fact[{index}] evidence speaker set mismatch")
        addressees = [str(value) for value in fact.get("addressee_refs") or []]
        if not addressees:
            raise MemoryVerificationError(f"fact[{index}] has no addressee scope")
        invalid = set(addressees) - profile_ids - {"group", "unknown"}
        if invalid:
            raise MemoryVerificationError(f"fact[{index}] invalid addressee refs: {invalid}")
        addressee_unknown += int(addressees == ["unknown"])
        addressee_group += int(addressees == ["group"])
        cross_turn += int(len(refs) > 1)

    for index, row in enumerate(collections):
        refs = set(map(str, row.get("refer_ids") or []))
        members = set(map(str, row.get("member_fact_ids") or []))
        if not refs or not refs.issubset(raw_by_ref) or not members.issubset(fact_ids):
            raise MemoryVerificationError(f"collection[{index}] refer closure failed")

    checkpoint_paths = sorted(checkpoint_dir.glob("*/*.json"))
    if len(checkpoint_paths) != expected_windows:
        raise MemoryVerificationError(
            f"checkpoint count mismatch: {len(checkpoint_paths)}/{expected_windows}"
        )
    for path in checkpoint_paths:
        checkpoint = read_object(path, "checkpoint")
        if checkpoint.get("fingerprint") != fingerprint:
            raise MemoryVerificationError(f"checkpoint fingerprint mismatch: {path}")
        refs = set(map(str, checkpoint.get("refer_ids") or []))
        if not refs or not refs.issubset(raw_by_ref):
            raise MemoryVerificationError(f"checkpoint refer IDs invalid: {path}")
        rows = checkpoint.get("facts")
        if not isinstance(rows, list) or not rows:
            raise MemoryVerificationError(f"checkpoint has no facts: {path}")

    for key in (
        "qa_consumed", "gold_evidence_consumed", "gt_speaker_consumed",
        "gt_addressee_consumed", "identity_retrieval_enabled",
        "profile_retrieval_enabled",
    ):
        if meta.get(key) is not False:
            raise MemoryVerificationError(f"unsafe meta flag: {key}={meta.get(key)!r}")
    return {
        "schema_version": "voxpoly-r12-memory-verification.v1",
        "status": "complete",
        "case_id": memory.get("case_id"),
        "raw": len(raw),
        "facts": len(facts),
        "cross_turn_facts": cross_turn,
        "collections": len(collections),
        "profiles": len(profiles),
        "checkpoints": len(checkpoint_paths),
        "addressee_unknown": addressee_unknown,
        "addressee_group": addressee_group,
        "refer_closure": True,
        "fingerprint_verified": True,
        "identity_retrieval_enabled": False,
        "qa_consumed": False,
        "gold_consumed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--expected-turns", type=int, default=480)
    parser.add_argument("--expected-windows", type=int, default=72)
    parser.add_argument("--expected-profiles", type=int, default=5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify_memory(
            args.memory,
            args.checkpoints,
            args.preflight,
            expected_turns=args.expected_turns,
            expected_windows=args.expected_windows,
            expected_profiles=args.expected_profiles,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"memory verification failed: {exc}", file=sys.stderr)
        return 2
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
