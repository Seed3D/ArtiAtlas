# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Locate and verify ArtiBench benchmark suites.

You download and unzip suites yourself (see the README); this module is the single
place that knows where an unzipped suite lives.

Resolution order for a suite root (first hit wins):

1. an explicit CLI argument (``--prompt_suite_dir`` / ``--gt_dir``), which never
   reaches this module
2. ``$ARTIBENCH_DATA_ROOT/<suite>/``

CLI::

    python -m artibench.data list
    python -m artibench.data verify --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Inside the package, not at the repo root: an installed wheel has no repo root,
# so resolving relative to it made every `artibench-data` call fail after pip install.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
RELEASE_FILE = ASSETS_DIR / "RELEASE.json"

# ``category_rich`` has 3D ground truth and runs the full paired protocol; ``in_the_wild`` has none, so C1/C2 are skipped.
SUITES = ("category_rich", "in_the_wild")


class DataError(RuntimeError):
    """Raised when a suite cannot be located, downloaded or verified."""


@dataclass(frozen=True)
class SuiteSpec:
    """One suite's entry in assets/RELEASE.json."""

    name: str
    version: str
    has_gt: bool = True
    note: str = ""


def load_release() -> Dict[str, SuiteSpec]:
    """Parse assets/RELEASE.json into ``{suite_name: SuiteSpec}``."""
    if not RELEASE_FILE.is_file():
        raise DataError(f"release manifest not found: {RELEASE_FILE}")
    payload = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
    out: Dict[str, SuiteSpec] = {}
    for name, entry in payload.get("suites", {}).items():
        out[name] = SuiteSpec(
            name=name,
            version=str(entry.get("version", "0")),
            has_gt=bool(entry.get("has_gt", True)),
            note=entry.get("note", ""),
        )
    return out


def _spec(name: str) -> SuiteSpec:
    specs = load_release()
    if name not in specs:
        raise DataError(
            f"unknown suite {name!r}; this release supports: {', '.join(SUITES)}"
        )
    return specs[name]


def _candidate_roots(spec: SuiteSpec) -> List[Path]:
    roots: List[Path] = []
    env_root = os.environ.get("ARTIBENCH_DATA_ROOT", "").strip()
    if env_root:
        roots.append(Path(env_root) / spec.name)
    return roots


def suite_root(name: str) -> Path:
    """Return the directory holding suite ``name``."""
    spec = _spec(name)

    searched = _candidate_roots(spec)
    for root in searched:
        if (root / "manifest.json").is_file():
            return root

    where = (
        f"Searched: {[r.as_posix() for r in searched]}"
        if searched else "ARTIBENCH_DATA_ROOT is not set, so nothing was searched."
    )
    raise DataError(
        f"suite {name!r} not found. Download it from the link in the README and\n"
        f"unzip it, then point ARTIBENCH_DATA_ROOT at the directory you unzipped into:\n"
        f"  export ARTIBENCH_DATA_ROOT=/scratch/artibench\n"
        f"  unzip {name}.zip -d \"$ARTIBENCH_DATA_ROOT\"\n"
        f"{where}"
    )


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(name: str) -> int:
    """Check a suite against its checksum manifest.

    A mismatch on a regenerable file is *drift* (warn, name the cache to
    delete); a mismatch on anything else is *corruption* (hard fail).
    Regenerable entries carry a ``# rebuildable:`` prefix.

    Returns the number of corrupt files.
    """
    root = suite_root(name)
    manifest = root / "files.sha256"
    if not manifest.is_file():
        raise DataError(f"no checksum manifest at {manifest}")

    corrupt, drift, missing = 0, 0, 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        rebuildable = raw.startswith("# rebuildable:")
        if rebuildable:
            raw = raw[len("# rebuildable:"):].strip()
        elif raw.startswith("#"):
            continue
        try:
            digest, rel = raw.split(None, 1)
        except ValueError:
            continue
        path = root / rel.strip()
        if not path.is_file():
            missing += 1
            print(f"MISSING  {rel.strip()}")
            continue
        if _sha256(path) != digest:
            if rebuildable:
                drift += 1
                print(f"DRIFT    {rel.strip()}  (cache; delete it and re-run the eval)")
            else:
                corrupt += 1
                print(f"CORRUPT  {rel.strip()}")

    print(f"\n{name}: {corrupt} corrupt, {drift} drift, {missing} missing")
    return corrupt


def _cmd_list() -> int:
    specs = load_release()
    width = max((len(n) for n in specs), default=4)
    for name in SUITES:
        spec = specs.get(name)
        if spec is None:
            print(f"{name:<{width}}  not described in assets/RELEASE.json")
            continue
        try:
            root = suite_root(name)
            where = f"present at {root}"
        except DataError:
            where = "not present"
        gt = "paired + self" if spec.has_gt else "self + alignment + VLM (no 3D GT)"
        print(f"{name:<{width}}  v{spec.version:<8}  {gt:<34}  {where}")
        if spec.note:
            print(f"{'':<{width}}  {spec.note}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m artibench.data",
        description="Locate and verify ArtiBench benchmark suites.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show every suite and whether it is present")

    p_verify = sub.add_parser("verify", help="checksum a suite")
    p_verify.add_argument("suite", nargs="?", choices=SUITES)
    p_verify.add_argument("--all", action="store_true", help="verify every present suite")

    args = parser.parse_args(argv)

    try:
        if args.command == "list":
            return _cmd_list()
        if args.command == "verify":
            if args.all:
                worst = 0
                for name in SUITES:
                    try:
                        worst += verify(name)
                    except DataError as exc:
                        print(f"{name}: skipped ({exc.args[0].splitlines()[0]})")
                return 1 if worst else 0
            if not args.suite:
                parser.error("give a suite name or --all")
            return 1 if verify(args.suite) else 0
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
