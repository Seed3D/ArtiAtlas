#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Score one evaluation run into the numbers you report.

`evaluate.sh` writes per-sample results. This turns them into one row of
`filled_mean` values, which is what the paper's cells are: a metric your method
failed to produce on an object is imputed with the worst value any published
baseline scored *on that object*, so crashing on the hard cases is not rewarded.
The worst-value table is frozen and shipped in `results/<suite>/fill_reference.json`.

    python scripts/score_run.py --suite category_rich --run out/mymodel
    python scripts/score_run.py --suite category_rich --run out/mymodel --json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Metric key -> the column it is called in the paper. Reporting a number under
# any other name makes it hard to compare, so this is the naming to use.
CATEGORY_RICH_LAYOUT: List[tuple] = [
    ("Object Geometry", "whole_mesh.rest_overall_cd", "R-obj CD", "lower"),
    ("Object Geometry", "whole_mesh.clip_score", "CLIP", "higher"),
    ("Object Geometry", "whole_mesh.clip_n_score", "CLIP-N", "higher"),
    ("Object Geometry", "whole_mesh.ulip2_score", "ULIP-2", "higher"),
    ("Object Geometry", "whole_mesh.uni3d_score", "Uni3D", "higher"),
    ("Part Decomposition", "part_meshes.rest_per_part_cd", "R-part CD", "lower"),
    ("Part Decomposition", "part_meshes.rest_per_part_giou", "R-part gIoU", "higher"),
    ("Part Decomposition", "part_meshes.vlm_part_decomp_score", "decomp", "higher"),
    ("Part Decomposition", "part_meshes.vlm_part_complete_score", "compl.", "higher"),
    ("Part Decomposition", "part_meshes.rest_overlap_ratio", "R-overlap", "lower"),
    ("Articulation", "articulation.success_rate", "j-succ.", "higher"),
    ("Articulation", "articulation.limit_error_mean", "limit err", "lower"),
    ("Articulation", "articulation.articulated_per_part_cd", "A-part CD", "lower"),
    ("Articulation", "articulation.articulated_per_part_giou", "A-part gIoU", "higher"),
    ("Articulation", "articulation.vlm_joint_type_score", "j-type", "higher"),
    ("Articulation", "articulation.vlm_joint_axis_score", "j-axis", "higher"),
    ("Articulation", "articulation.vlm_motion_score", "motion", "higher"),
    ("Articulation", "articulation.articulated_overlap_mean", "A-overlap", "lower"),
]

IN_THE_WILD_LAYOUT: List[tuple] = [
    row for row in CATEGORY_RICH_LAYOUT
    if row[1] not in {
        "whole_mesh.rest_overall_cd",
        "part_meshes.rest_per_part_cd",
        "part_meshes.rest_per_part_giou",
        "articulation.success_rate",
        "articulation.limit_error_mean",
        "articulation.articulated_per_part_cd",
        "articulation.articulated_per_part_giou",
    }
]

LAYOUTS = {"category_rich": CATEGORY_RICH_LAYOUT, "in_the_wild": IN_THE_WILD_LAYOUT}


def flatten(obj: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for dim, val in obj.items():
        if dim in ("object_id", "_raw") or not isinstance(val, dict):
            continue
        for metric, value in val.items():
            if isinstance(value, (int, float)):
                out[f"{dim}.{metric}"] = float(value)
    return out


def find_result_jsonls(target: Path) -> List[Path]:
    """Locate every results.jsonl under a run directory.

    Accepts the out-dir evaluate.sh produces (which holds self_quality/ and
    paired/ side by side), either of those subdirectories, or a jsonl directly.
    Pointing at paired/ alone silently drops the two overlap metrics, which come
    out of eval_self — pass the out-dir.
    """
    if target.is_file():
        return [target]
    found = [target / "results.jsonl"] if (target / "results.jsonl").is_file() else []
    for sub in ("paired", "self_quality"):
        p = target / sub / "results.jsonl"
        if p.is_file():
            found.append(p)
    return found


def score_against_reference(jsonls: List[Path], fill_ref: Dict[str, Any]) -> Dict[str, float]:
    """filled_mean for one method, using the frozen per-sample worst values."""
    per_oid: Dict[str, Dict[str, float]] = {}
    for jsonl in jsonls:
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            oid = obj.get("object_id")
            if oid is not None:
                per_oid.setdefault(str(oid), {}).update(flatten(obj))

    row: Dict[str, float] = {}
    for metric, worst in fill_ref["worst"].items():
        # Opt-out rule: a method that never emits a metric is not scored on it,
        # rather than being filled to worst on every sample.
        if not any(metric in v for v in per_oid.values()):
            continue
        vals = [
            per_oid[oid][metric] if (oid in per_oid and metric in per_oid[oid]) else w
            for oid, w in sorted(worst.items())
        ]
        if vals:
            row[metric] = sum(vals) / len(vals)
    return row


def fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if abs(v) < 0.01 or abs(v) >= 1000:
        return f"{v:.3e}"
    return f"{v:.4f}".rstrip("0").rstrip(".")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=sorted(LAYOUTS))
    ap.add_argument("--run", type=Path, required=True,
                    help="The --out-dir from evaluate.sh (not one of its subdirectories).")
    ap.add_argument("--fill-reference", type=Path, default=None,
                    help="Defaults to results/<suite>/fill_reference.json.")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = ap.parse_args()

    jsonls = find_result_jsonls(args.run)
    if not jsonls:
        raise SystemExit(f"no results.jsonl found under {args.run}")

    ref_path = args.fill_reference or REPO_ROOT / "results" / args.suite / "fill_reference.json"
    if not ref_path.is_file():
        raise SystemExit(f"missing fill reference: {ref_path}")
    fill_ref = json.loads(ref_path.read_text(encoding="utf-8"))

    row = score_against_reference(jsonls, fill_ref)
    layout = LAYOUTS[args.suite]

    if args.json:
        print(json.dumps(
            {column: row.get(key) for _, key, column, _ in layout},
            indent=2,
        ))
        return

    print(f"suite: {args.suite}   n_objects: {fill_ref.get('n_oids', '?')}   "
          f"read: {', '.join(str(p.relative_to(args.run)) for p in jsonls)}\n")
    width = max(len(c) for _, _, c, _ in layout)
    group = None
    for grp, key, column, direction in layout:
        if grp != group:
            group = grp
            print(f"  {grp}")
        arrow = "↓" if direction == "lower" else "↑"
        print(f"    {column:<{width}} {arrow}  {fmt(row.get(key))}")

    missing = [c for _, k, c, _ in layout if k not in row]
    if missing:
        print(f"\n  not scored (your run never emitted these): {', '.join(missing)}")
        if any(c in ("R-overlap", "A-overlap") for c in missing):
            print("  R-overlap / A-overlap come from eval_self — check you did not pass "
                  "--skip-self, and point --run at the out-dir rather than paired/.")


if __name__ == "__main__":
    main()
