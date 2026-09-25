#!/usr/bin/env python3
"""Extract round-0 route plans from an optional full internal result archive."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def _read(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Any:
    stream = archive.extractfile(member)
    if stream is None:
        raise RuntimeError(member.name)
    return json.load(stream)


def _initial_plan(row: dict[str, Any]) -> dict[str, Any]:
    iterative = (row.get("route_execution") or {}).get("iterative_retrieval") or {}
    history = iterative.get("history") or []
    if history and isinstance(history[0].get("plan"), dict):
        return history[0]["plan"]
    plan = row.get("route_plan")
    if not isinstance(plan, dict):
        raise ValueError(f"missing route plan for {row.get('qa_id')}")
    return plan


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive", type=Path, required=True,
        help="full result archive containing route plans; not the public score-only replay archive",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/frozen_routes_v1")
    args = parser.parse_args()

    memgallery_count = 0
    h2h_count = 0
    with tarfile.open(args.archive, "r:gz") as archive:
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
                rows = _read(archive, member)
                case_name = name.split("/mem_gallery_", 1)[1].split("/", 1)[0]
                routes = [
                    {"qa_id": row["qa_id"], "route_plan": _initial_plan(row)}
                    for row in rows
                ]
                _write(args.out / "memgallery" / f"{case_name}.json", routes)
                memgallery_count += 1
            elif name.startswith(
                "h2h_iterative_topk30_full190/unified_hybrid_route/"
            ):
                rows = _read(archive, member)
                routes = [
                    {"qa_id": row["qa_id"], "route_plan": _initial_plan(row)}
                    for row in rows
                ]
                _write(
                    args.out / "h2h" / "unified_hybrid_route" / Path(name).name,
                    routes,
                )
                h2h_count += 1
    if memgallery_count != 20 or h2h_count != 5:
        raise AssertionError(
            f"unexpected route coverage: Mem-Gallery={memgallery_count}, H2H={h2h_count}"
        )
    print(f"wrote frozen routes to {args.out}")


if __name__ == "__main__":
    main()
