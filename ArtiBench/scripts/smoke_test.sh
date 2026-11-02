#!/usr/bin/env bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Score the two bundled objects and compare against the stored fixture.
#
# Runs the paired evaluator over the two-object example that ships in
# examples/ and compares the result against examples/expected/results.json.
#
# Needs the base install plus pytorch3d.
#
#   bash scripts/smoke_test.sh
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT="${1:-$(mktemp -d)}"
ATOL="${ARTIBENCH_SMOKE_ATOL:-1e-3}"
RTOL="${ARTIBENCH_SMOKE_RTOL:-0.05}"

echo "==> evaluating examples/predictions against examples/mini_suite/gt"
$PYTHON -m artibench.eval.eval_paired \
    --pred_dir "$REPO/examples/predictions" \
    --gt_dir   "$REPO/examples/mini_suite/gt" \
    --out_dir  "$OUT" \
    --recon_norm mesh \
    --skip_alignment

echo
echo "==> comparing against examples/expected/results.json  (atol $ATOL, rtol $RTOL)"
$PYTHON - "$OUT/results.json" "$REPO/examples/expected/results.json" "$ATOL" "$RTOL" <<'PY'
import json, sys

got_path, want_path = sys.argv[1], sys.argv[2]
atol, rtol = float(sys.argv[3]), float(sys.argv[4])
got = json.load(open(got_path))
want = json.load(open(want_path))

bad, checked = [], 0
if got.get("n_objects") != want.get("n_objects"):
    bad.append(f"n_objects: got {got.get('n_objects')}, expected {want['n_objects']}")

for dim, metrics in want.items():
    if not isinstance(metrics, dict):
        continue
    for key, expected in metrics.items():
        actual = got.get(dim, {}).get(key)
        if actual is None:
            bad.append(f"{dim}.{key}: missing from output")
            continue
        checked += 1
        # Geometry metrics move by ~1% across torch/pytorch3d builds, and the
        # near-zero ones move a lot in relative terms while staying ~0. Accept
        # either bound: a broken install shows up as a missing metric or as a
        # value far outside both.
        if abs(actual - expected) > atol and abs(actual - expected) > rtol * abs(expected):
            bad.append(f"{dim}.{key}: got {actual:.6g}, expected {expected:.6g}")

if bad:
    print(f"FAIL — {len(bad)} mismatch(es) out of {checked} metrics:\n")
    for b in bad:
        print("  " + b)
    print("\nMissing metrics mean a dependency is absent. Values far outside both")
    print("tolerances mean the install is broken; the expectation was recorded on")
    print("torch 2.4 / pytorch3d 0.7.8.")
    sys.exit(1)

print(f"OK — {checked} metrics match.")
PY
