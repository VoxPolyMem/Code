#!/usr/bin/env python3
"""Build a deterministic, score-only public v1 replay archive.

The input is an expanded copy of the original experiment outputs. Only the
43 completed result documents consumed by ``replay_reference_metrics.py`` are
included. Questions, answers, predictions, dialogue evidence, logs, caches,
partial files, and machine-specific metadata are excluded.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reference/replay_outputs_v1.tgz"
PRIVATE_MARKERS = (
    b"/mnt/" + b"dolphinfs/",
    b"/home/" + b"hadoop-",
    b"jiawen" + b"xu02",
)
SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9]{16,}"),
    re.compile(rb"\b220[0-9]{10,}\b"),
)


def select_members(source: Path) -> list[Path]:
    members: list[Path] = []
    members.extend(
        source.glob(
            "memgallery_iterative_topk20_full20/*/mem_gallery_*/eval_results_*.json"
        )
    )
    members.extend(
        source.glob("h2h_iterative_topk30_full190/unified_hybrid_route/*.json")
    )
    members.extend(
        source.glob("current_unified_iterative3/CASE_G_*/matched_pairs.json")
    )
    members = sorted(members, key=lambda path: path.relative_to(source).as_posix())

    group_counts = {
        "mem_gallery": sum("memgallery_" in str(path) for path in members),
        "h2hmem": sum("h2h_iterative_" in str(path) for path in members),
        "voxpolybench": sum("current_unified_" in str(path) for path in members),
    }
    expected = {"mem_gallery": 20, "h2hmem": 5, "voxpolybench": 18}
    if group_counts != expected:
        raise AssertionError(f"expected {expected}, found {group_counts}")
    return members


def validate_public_payload(relative: Path, payload: bytes) -> None:
    for marker in PRIVATE_MARKERS:
        if marker in payload:
            raise AssertionError(f"private path marker in {relative}: {marker!r}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(payload):
            raise AssertionError(f"credential-like token in {relative}")


def score_only_document(relative: Path, source: bytes) -> bytes:
    document = json.loads(source)
    group = relative.parts[0]
    if group.startswith("memgallery_"):
        reduced = [
            {"qa_id": row["qa_id"], "point": row.get("point"), "score": row["score"]}
            for row in document
        ]
    elif group.startswith("h2h_iterative_"):
        reduced = [
            {"qa_id": row["qa_id"], "sub_type": row.get("sub_type"), "score": row["score"]}
            for row in document
        ]
    elif group.startswith("current_unified_"):
        if document.get("status") != "complete":
            raise AssertionError(f"incomplete result in {relative}")
        reduced = {
            "status": "complete",
            "pairs": [
                {
                    "case_id": row["case_id"],
                    "qa_id": row["qa_id"],
                    "four_way_category": row.get("four_way_category"),
                    "arm": {"score": row["arm"]["score"]},
                }
                for row in document["pairs"]
            ],
        }
    else:
        raise AssertionError(f"unexpected result tree: {relative}")
    payload = (json.dumps(reduced, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    validate_public_payload(relative, payload)
    return payload


def build(source: Path, output: Path) -> None:
    members = select_members(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_stream, compresslevel=9, mtime=0
        ) as compressed_stream:
            with tarfile.open(
                fileobj=compressed_stream, mode="w", format=tarfile.PAX_FORMAT
            ) as archive:
                for path in members:
                    relative = path.relative_to(source)
                    payload = score_only_document(relative, path.read_bytes())
                    info = tarfile.TarInfo(relative.as_posix())
                    info.size = len(payload)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    archive.addfile(info, io.BytesIO(payload))
    print(f"wrote {len(members)} public replay files to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=Path,
        help="expanded root containing the three frozen experiment result trees",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
