#!/usr/bin/env python3
"""Offline structural, security, syntax, test, and replay gate for v1."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_ARCHIVE_SHA256 = "1b1f8125f4d4796d7a3335cd350b9a1533f3d9d756a6d3bc9d6ef0fbf65435fc"
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"\b220[0-9]{10,}\b"),
)
PRIVATE_PATH_MARKERS = ("/mnt/dolphinfs/", "/home/hadoop-", "jiawenxu02")


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> None:
    symlinks = [path for path in ROOT.rglob("*") if path.is_symlink()]
    if symlinks:
        fail("release contains symlinks: " + ", ".join(map(str, symlinks[:10])))

    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".sh", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            fail(f"credential-like token in {path.relative_to(ROOT)}")
        if path.resolve() != Path(__file__).resolve() and any(
            marker in text for marker in PRIVATE_PATH_MARKERS
        ):
            fail(f"private absolute path in {path.relative_to(ROOT)}")
        if path.suffix == ".py":
            py_compile.compile(str(path), doraise=True)

    archive = ROOT / "reference/replay_outputs_v1.tgz"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != EXPECTED_ARCHIVE_SHA256:
        fail(f"reference archive hash mismatch: {digest}")

    manifest = json.loads(
        (ROOT / "reference/code_manifest.json").read_text(encoding="utf-8")
    )
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            fail(f"code manifest mismatch: {relative}: {actual}")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT),
            str(ROOT / "vendor/v2_update_pipeline"),
            str(ROOT / "vendor/v3_update_pipeline"),
        ]
    )
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/replay_reference_metrics.py"],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    print("VoxPolyMem v1 release verification: PASS")


if __name__ == "__main__":
    main()
