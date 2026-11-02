#!/bin/bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# ArtiBench — run the full evaluation protocol on one set of predictions.
#
# Runs, in order: the URDF convention check, eval_self (self-quality), and
# eval_paired (reconstruction + alignment + VLM-as-judge), against a suite
# downloaded and unzipped from the links in the README.
#
# Usage:
#   bash scripts/evaluate.sh --suite <suite> --pred-dir <dir> [options]
#
# Required:
#   --suite <name>        category_rich   — 3D ground truth: the full protocol
#                         in_the_wild — no 3D GT: self-quality, alignment and
#                                       the VLM judge; C1/C2 are skipped
#   --pred-dir <dir>      Your predictions: <dir>/<object_id>/mobility.urdf
#                         plus its meshes. Object ids must match the suite's.
#
# Strongly recommended:
#   --input-modality <m>  How your method is conditioned (mesh|image).
#                         Selects the reconstruction normalization:
#                           mesh  -> recon_norm=none  (your output is already
#                                    in the GT frame; do not re-normalize)
#                           image -> recon_norm=mesh  (your output has an
#                                    arbitrary scale/placement; normalize both
#                                    sides before comparing)
#                         Default: image. Getting this wrong makes your numbers
#                         incomparable.
#
# Optional:
#   --out-dir <dir>       Where results go (default: out/<suite>)
#   --expected-root <dir> Object ids that SHOULD exist, so self-quality reports
#                         real coverage instead of counting only what you
#                         produced. Defaults to the suite's gt/ when it has one.
#   --recon-norm <mode>   Set normalization directly (none|bbox|mesh).
#                         Overrides --input-modality.
#   --vlm-model <name>    Judge model (default: gemini-3.1-pro-preview, what the paper used)
#   --max-qpm <n>         Client-side QPM cap for the judge (default 60)
#   --skip-self           Skip eval_self
#   --skip-paired         Skip eval_paired
#   --no-vlm              Skip the VLM judge (alignment still runs)
#   --no-alignment        Skip CLIP/CLIP-N/ULIP-2/Uni3D (the judge still runs)
#   --object-id <id>      Evaluate a single object (debugging)
#   --no-check-urdf       Skip the URDF convention check + repair pass
#                         (default: on — it is fast on already-clean inputs)
#
# Env vars:
#   ARTIBENCH_DATA_ROOT   Suite lookup root, probed as <root>/<suite>/. Required
#                         unless SUITE_DIR is set. Nothing here downloads a
#                         suite; the README has the links.
#   SUITE_DIR             Override the suite directory entirely
#   GT_DIR                Override the ground-truth directory
#   GOOGLE_API_KEY        Required unless --no-vlm
#   PYTHON                Interpreter (default: python3)
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

SUITE=""
PRED_DIR=""
OUT_DIR=""
EXPECTED_ROOT=""
INPUT_MODALITY="image"
RECON_NORM="auto"
VLM_MODEL="gemini-3.1-pro-preview"
MAX_QPM=60
SKIP_SELF=false
SKIP_PAIRED=false
NO_VLM=false
NO_ALIGN=false
OBJECT_ID=""
CHECK_URDF=true

usage() { sed -n '/^# ArtiBench/,/^# ====/p' "$0" | sed '$d'; exit "${1:-1}"; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --suite)           SUITE="$2"; shift 2 ;;
        --pred-dir)        PRED_DIR="$2"; shift 2 ;;
        --out-dir)         OUT_DIR="$2"; shift 2 ;;
        --expected-root)   EXPECTED_ROOT="$2"; shift 2 ;;
        --input-modality)  INPUT_MODALITY="$2"; shift 2 ;;
        --recon-norm)      RECON_NORM="$2"; shift 2 ;;
        --vlm-model)       VLM_MODEL="$2"; shift 2 ;;
        --max-qpm)         MAX_QPM="$2"; shift 2 ;;
        --skip-self)       SKIP_SELF=true; shift ;;
        --skip-paired)     SKIP_PAIRED=true; shift ;;
        --no-vlm)          NO_VLM=true; shift ;;
        --no-alignment)    NO_ALIGN=true; shift ;;
        --object-id)       OBJECT_ID="$2"; shift 2 ;;
        --no-check-urdf)   CHECK_URDF=false; shift ;;
        -h|--help)         usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

[ -n "$SUITE" ]    || { echo "--suite is required" >&2; usage; }
[ -n "$PRED_DIR" ] || { echo "--pred-dir is required" >&2; usage; }
[ -d "$PRED_DIR" ] || { echo "--pred-dir does not exist: $PRED_DIR" >&2; exit 1; }

# The run happens from $REPO_DIR, so paths the caller gave relative to their own
# working directory have to be resolved before that cd.
PRED_DIR="$(cd "$PRED_DIR" && pwd)"

case "$SUITE" in
    category_rich|in_the_wild) ;;
    *) echo "Unknown suite: $SUITE  (expected: category_rich | in_the_wild)" >&2; exit 1 ;;
esac

# Locate the suite through artibench.data, which reads ARTIBENCH_DATA_ROOT.
# SUITE_DIR short-circuits it for a suite kept elsewhere.
if [ -z "${SUITE_DIR:-}" ]; then
    SUITE_DIR="$($PYTHON -c "
import sys
from artibench.data import suite_root, DataError
try:
    print(suite_root('$SUITE'))
except DataError as exc:
    print(exc, file=sys.stderr); sys.exit(1)
")" || {
        exit 1
    }
fi

# in_the_wild has no 3D ground truth by design.
if [ "$SUITE" = "in_the_wild" ]; then
    : "${GT_DIR=}"
else
    : "${GT_DIR=${SUITE_DIR}/gt}"
fi
NO_GT=false
[ -z "$GT_DIR" ] && NO_GT=true

: "${OUT_DIR:=${PWD}/out/${SUITE}}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
SELF_OUT="${OUT_DIR}/self_quality"
PAIRED_OUT="${OUT_DIR}/paired"
[ -n "$GT_DIR" ] && [ -d "$GT_DIR" ] && GT_DIR="$(cd "$GT_DIR" && pwd)"
[ -d "$SUITE_DIR" ] && SUITE_DIR="$(cd "$SUITE_DIR" && pwd)"

# Reconstruction normalization, shared by C1 and C2.
if [ "$RECON_NORM" = "auto" ]; then
    case "$INPUT_MODALITY" in
        mesh)   RECON_NORM="none" ;;
        image)  RECON_NORM="mesh" ;;
        *) echo "Unknown --input-modality: $INPUT_MODALITY (expected: mesh | image)" >&2; exit 1 ;;
    esac
fi

# Sanity checks
if [ "$NO_GT" != "true" ] && [ ! -d "$GT_DIR" ]; then
    echo "GT dir missing: $GT_DIR" >&2; exit 1
fi
if [ ! -f "${SUITE_DIR}/manifest.json" ]; then
    echo "Suite manifest missing: ${SUITE_DIR}/manifest.json" >&2
    echo "Download and unzip it first; see the README for the link." >&2
    exit 1
fi
if [ "$NO_VLM" != "true" ] && [ -z "${GOOGLE_API_KEY:-}${GEMINI_API_KEY:-}" ]; then
    echo "The VLM judge needs GOOGLE_API_KEY (or GEMINI_API_KEY); pass --no-vlm to skip it." >&2
    exit 1
fi

# Self-quality coverage is only meaningful against a list of ids that SHOULD
# exist; otherwise gt_total just counts whatever you produced.
if [ -z "$EXPECTED_ROOT" ] && [ "$NO_GT" != "true" ]; then
    EXPECTED_ROOT="$GT_DIR"
fi
# A suite without 3D ground truth has no gt/ to enumerate, so the expected set
# comes from its manifest. Otherwise gt_total counts whatever you produced and
# coverage is always 100%.
EXPECTED_IDS=""
if [ "$NO_GT" = "true" ] && [ -f "$SUITE_DIR/manifest.json" ]; then
    EXPECTED_IDS="$SUITE_DIR/manifest.json"
fi

mkdir -p "$PAIRED_OUT" "$SELF_OUT"

echo "============================================="
echo "  ArtiBench"
echo "============================================="
echo "  Suite:           $SUITE  ($SUITE_DIR)"
echo "  Predictions:     $PRED_DIR"
[ "$CHECK_URDF" = "true" ] && echo "                   (the URDF repair pass edits these in place;"
[ "$CHECK_URDF" = "true" ] && echo "                    a <name>_backup.urdf is kept beside each one)"
if [ "$NO_GT" = "true" ]; then
    echo "  Ground truth:    none (this suite has no 3D GT; C1/C2 skipped)"
else
    echo "  Ground truth:    $GT_DIR"
fi
echo "  Output:          $OUT_DIR"
echo "  Input modality:  $INPUT_MODALITY  ->  recon_norm=$RECON_NORM"
echo "  Alignment:       $([ "$NO_ALIGN" = true ] && echo skipped || echo on)"
echo "  VLM judge:       $([ "$NO_VLM" = true ] && echo skipped || echo "$VLM_MODEL (max_qpm=$MAX_QPM)")"
[ -n "$OBJECT_ID" ] && echo "  Object id:       $OBJECT_ID"
echo "============================================="

cd "$REPO_DIR"

# ── URDF convention check ──
if [ "$CHECK_URDF" = "true" ]; then
    echo ""
    echo "--- URDF convention check ---"
    $PYTHON scripts/check_urdf_conventions.py --root "$PRED_DIR" --enable-urdf-fixing
fi

# ── eval_self ──
# Full-set runs go through the per-sample subprocess wrapper so a CUDA or mesh
# segfault on one object only loses that object instead of the whole run.
if [ "$SKIP_SELF" != "true" ]; then
    echo ""
    echo "--- eval_self ---"
    if [ -n "$OBJECT_ID" ]; then
        $PYTHON -m artibench.eval.eval_self \
            --input_root "$PRED_DIR" --out_dir "$SELF_OUT" \
            --object_id "$OBJECT_ID"
    else
        SELF_ARGS=(--input_root "$PRED_DIR" --out_dir "$SELF_OUT")
        [ -n "$EXPECTED_ROOT" ] && SELF_ARGS+=(--expected_root "$EXPECTED_ROOT")
        [ -n "$EXPECTED_IDS" ]  && SELF_ARGS+=(--expected_ids "$EXPECTED_IDS")
        $PYTHON -m artibench.eval._self_subprocess "${SELF_ARGS[@]}"
    fi
fi

# ── eval_paired ──
if [ "$SKIP_PAIRED" != "true" ]; then
    echo ""
    echo "--- eval_paired (recon_norm=$RECON_NORM) ---"
    PAIRED_ARGS=(
        --pred_dir         "$PRED_DIR"
        --out_dir          "$PAIRED_OUT"
        --recon_norm       "$RECON_NORM"
        --prompt_suite_dir "$SUITE_DIR"
        --vlm_max_qpm      "$MAX_QPM"
    )
    # Pre-sampled GT surface points. Without these the geometry metrics re-sample
    # every run, so two evaluations of one prediction disagree.
    [ -d "$SUITE_DIR/cached_link_point_clouds" ] && \
        PAIRED_ARGS+=(--gt_pcl_dir "$SUITE_DIR/cached_link_point_clouds")
    if [ "$NO_GT" = "true" ]; then
        PAIRED_ARGS+=(--no_gt)
    else
        PAIRED_ARGS+=(--gt_dir "$GT_DIR")
    fi
    [ "$NO_ALIGN" = "true" ] && PAIRED_ARGS+=(--skip_alignment)
    [ "$NO_VLM"   != "true" ] && PAIRED_ARGS+=(--vlm_judge --vlm_model "$VLM_MODEL")
    [ -n "$OBJECT_ID" ] && PAIRED_ARGS+=(--object_id "$OBJECT_ID")
    [ -n "$EXPECTED_IDS" ] && PAIRED_ARGS+=(--expected_ids "$EXPECTED_IDS")
    $PYTHON -u -m artibench.eval.eval_paired "${PAIRED_ARGS[@]}"
fi

echo ""
echo "Done. Results in $OUT_DIR"
