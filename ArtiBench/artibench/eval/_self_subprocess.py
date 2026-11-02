#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Subprocess-isolated eval_self: a CUDA or mesh failure on one object loses that
object rather than the whole run.

Runs each object in a separate subprocess so a segfault doesn't kill the batch.
Aggregates per-object dimension results at the end.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--expected_root",
        default=None,
        help="Optional root containing expected object IDs for completeness "
             "accounting, usually the GT root.",
    )
    parser.add_argument(
        "--expected_ids",
        default=None,
        help="Suite manifest.json, for suites with no GT root to enumerate.",
    )
    parser.add_argument("--timeout", type=int, default=120, help="Per-object timeout in seconds")
    parser.add_argument(
        "--object_ids", nargs="+", default=None,
        help="Optional subset of object IDs to evaluate (default: every dir "
             "under input_root with a mobility.urdf). Useful for salvage "
             "runs that target only the previously-segfaulted samples.",
    )
    args, extra_args = parser.parse_known_args()
    args.extra_args = extra_args

    python = sys.executable
    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover objects
    discovered = sorted([
        d.name for d in input_root.iterdir()
        if d.is_dir() and (d / "mobility.urdf").exists()
    ])
    if args.object_ids:
        wanted = set(args.object_ids)
        objects = [o for o in discovered if o in wanted]
        unknown = sorted(wanted - set(discovered))
        if unknown:
            print(f"[Subprocess eval_self] WARNING: {len(unknown)} requested "
                  f"object_ids not found under {input_root}: {unknown[:10]}"
                  f"{'...' if len(unknown) > 10 else ''}")
    else:
        objects = discovered
    print(f"[Subprocess eval_self] Found {len(objects)} objects in {input_root}"
          f"{' (filtered from %d)' % len(discovered) if args.object_ids else ''}")

    per_object = []
    ok = 0
    crash = 0
    fail = 0
    start = time.time()

    for oid in objects:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                python, "-m", "artibench.eval.eval_self",
                "--input_root", str(input_root),
                "--out_dir", tmpdir,
                "--object_id", oid,
            ] + args.extra_args
            if args.expected_root:
                cmd.extend(["--expected_root", str(args.expected_root)])

            try:
                result = subprocess.run(cmd, timeout=args.timeout, capture_output=True, text=True)
                if result.returncode == 0:
                    res_path = os.path.join(tmpdir, "results.json")
                    if os.path.exists(res_path):
                        with open(res_path) as f:
                            data = json.load(f)
                        per_object.append({
                            "object_id": oid,
                            "whole_mesh": data.get("whole_mesh", {}),
                            "part_meshes": data.get("part_meshes", {}),
                            "articulation": data.get("articulation", {}),
                        })
                        ok += 1
                        print(f"  [{oid}] OK")
                    else:
                        fail += 1
                        print(f"  [{oid}] no results.json")
                elif result.returncode == -11 or result.returncode == 139:
                    crash += 1
                    print(f"  [{oid}] SEGFAULT")
                else:
                    fail += 1
                    print(f"  [{oid}] FAIL (rc={result.returncode})")
                    if result.stderr:
                        for line in result.stderr.strip().split("\n")[-3:]:
                            print(f"    {line}")
            except subprocess.TimeoutExpired:
                fail += 1
                print(f"  [{oid}] TIMEOUT ({args.timeout}s)")

    # Aggregate using the existing dimension_report module
    from artibench.reporting.dimension_report import aggregate_dimensions, generate_analysis, write_results

    params = {"subprocess_isolated": True, "extra_args": args.extra_args}
    elapsed = time.time() - start

    # Objects that segfaulted, errored or timed out are listed under missing_oids, so a failure stays distinct from a missing metric.
    evaluated_oids = [obj.get("object_id") for obj in per_object]
    if args.expected_root:
        expected_root = Path(args.expected_root)
        expected_oids = sorted([
            d.name for d in expected_root.iterdir()
            if d.is_dir() and (d / "mobility.urdf").exists()
        ])
    elif args.expected_ids:
        manifest = json.loads(Path(args.expected_ids).read_text(encoding="utf-8"))
        expected_oids = sorted(m["model_name"] for m in manifest["models"])
    else:
        expected_oids = objects
    pred_missing_oids = [o for o in expected_oids if o not in set(objects)]
    missing_oids = [o for o in objects if o not in set(evaluated_oids)]
    completeness = {
        "gt_total": len(expected_oids),
        "pred_found": len(objects),
        "pred_missing": len(pred_missing_oids),
        "missing_oids": pred_missing_oids,
        "eval_succeeded": len(evaluated_oids),
        "eval_errored_oids": missing_oids,
        "coverage_rate": len(evaluated_oids) / max(1, len(expected_oids)),
        "segfault_count": crash,
        "other_failure_count": fail,
    }

    aggregate = aggregate_dimensions(per_object)
    analysis = generate_analysis(per_object)
    written = write_results(
        str(out_dir), aggregate, per_object, analysis, params, elapsed,
        completeness=completeness,
    )

    print()
    for dim_name in ("whole_mesh", "part_meshes", "articulation"):
        dim = aggregate.get(dim_name, {})
        if dim:
            items = ", ".join(f"{k}={v:.4f}" for k, v in list(dim.items())[:5])
            print(f"  [{dim_name}] {items}")

    for p in written:
        print(f"  Wrote {p}")

    print(f"\n[Done] {ok} OK, {crash} segfaults, {fail} other failures out of {len(objects)} objects (elapsed: {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
