#!/usr/bin/env python3
"""Recompute v1 benchmark aggregates directly from the frozen output archive."""

from __future__ import annotations

import argparse
import json
import math
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = ROOT / "reference/replay_outputs_v1.tgz"
DEFAULT_EXPECTED = ROOT / "reference/expected_metrics.json"


def _json_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Any:
    stream = archive.extractfile(member)
    if stream is None:
        raise RuntimeError(f"cannot read archive member: {member.name}")
    return json.load(stream)


def recompute(archive_path: Path) -> dict[str, dict[str, float | int]]:
    memgallery_docs: list[list[dict[str, Any]]] = []
    h2h_docs: list[list[dict[str, Any]]] = []
    vox_docs: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            name = member.name
            if not member.isfile() or not name.endswith(".json"):
                continue
            if (
                name.startswith("memgallery_iterative_topk20_full20/")
                and "/mem_gallery_" in name
                and "/logs/" not in name
                and "/eval_results_" in name
            ):
                document = _json_member(archive, member)
                if isinstance(document, list):
                    memgallery_docs.append(document)
            elif (
                name.startswith("h2h_iterative_topk30_full190/unified_hybrid_route/")
                and name.endswith(".json")
            ):
                document = _json_member(archive, member)
                if isinstance(document, list):
                    h2h_docs.append(document)
            elif (
                name.startswith("current_unified_iterative3/CASE_G_")
                and name.endswith("/matched_pairs.json")
            ):
                document = _json_member(archive, member)
                if isinstance(document, dict) and document.get("status") == "complete":
                    vox_docs.append(document)

    memgallery_rows = [row for document in memgallery_docs for row in document]
    h2h_rows = [row for document in h2h_docs for row in document]
    vox_rows = [row for document in vox_docs for row in document.get("pairs", [])]

    return {
        "mem_gallery": {
            "topics": len(memgallery_docs),
            "qa": len(memgallery_rows),
            "mean_llm_score": sum(float(row["score"]) for row in memgallery_rows)
            / len(memgallery_rows),
        },
        "h2hmem": {
            "dialogues": len(h2h_docs),
            "qa": len(h2h_rows),
            "mean_llm_score": sum(float(row["score"]) for row in h2h_rows)
            / len(h2h_rows),
        },
        "voxpolybench": {
            "cases": len(vox_docs),
            "qa": len(vox_rows),
            "mean_llm_score": sum(float(row["arm"]["score"]) for row in vox_rows)
            / len(vox_rows),
        },
    }


def verify(actual: dict[str, dict[str, float | int]], expected_path: Path) -> None:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))["benchmarks"]
    for benchmark, actual_metrics in actual.items():
        expected_metrics = expected[benchmark]
        for key, value in actual_metrics.items():
            expected_value = expected_metrics[key]
            if isinstance(value, float):
                if not math.isclose(value, float(expected_value), rel_tol=0, abs_tol=1e-12):
                    raise AssertionError(
                        f"{benchmark}.{key}: expected {expected_value}, got {value}"
                    )
            elif value != expected_value:
                raise AssertionError(
                    f"{benchmark}.{key}: expected {expected_value}, got {value}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    args = parser.parse_args()
    actual = recompute(args.archive)
    verify(actual, args.expected)
    print(json.dumps(actual, indent=2, sort_keys=True))
    print("VoxPolyMem v1 exact replay: PASS")


if __name__ == "__main__":
    main()

