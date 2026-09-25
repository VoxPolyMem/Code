#!/usr/bin/env python3
"""Zero-API validation of a real VoxPoly predicted-speaker input bundle."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from adapter import AdapterContractError, load_case_bundle
from build_memory import make_windows, atomic_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-experimental-audio", action="store_true")
    parser.add_argument("--case-view-dir", required=True, type=Path)
    parser.add_argument("--registry-state", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-strict-causal-aliases", action="store_true")
    args = parser.parse_args(argv)
    try:
        bundle = load_case_bundle(
            args.case_view_dir,
            args.registry_state,
            enabled=args.enable_experimental_audio,
            require_strict_causal_aliases=args.require_strict_causal_aliases,
        )
    except (AdapterContractError, OSError) as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2
    sessions = Counter(row["session_id"] for row in bundle.raw)
    windows = sum(
        len(make_windows([row for row in bundle.raw if row["session_id"] == sid]))
        for sid in sessions
    )
    report = {
        "status": "ready_for_contextual_memory_build",
        "zero_api": True,
        "case_id": bundle.case_id,
        "case_name": bundle.case_name,
        "sessions": len(sessions),
        "turns": len(bundle.raw),
        "windows_12_stride_6": windows,
        "profiles": len(bundle.profiles),
        "turns_per_session": dict(sorted(sessions.items())),
        "source_view_sha256": bundle.source_view_sha256,
        "sidecar_sha256": bundle.sidecar_sha256,
        "identity_prediction_sha256": bundle.prediction_sha256,
        "registry_state_sha256": bundle.registry_state_sha256,
        "prediction_protocol": bundle.prediction_protocol,
        "alias_timing": (
            "strict_turn_causal" if bundle.alias_is_causal_at_each_turn else "batch_final"
        ),
        "profile_retrieval_enabled": False,
        "qa_consumed": False,
        "gold_evidence_consumed": False,
        "gt_addressee_consumed": False,
    }
    if args.out:
        atomic_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
