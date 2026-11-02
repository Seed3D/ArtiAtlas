# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
artibench.eval.eval_paired
===========================
Paired reconstruction evaluation (pred <-> GT).

Evaluates matched pairs using C1 (joint params), C2 (part geometry),
and optional image-alignment metrics (CLIP, CLIP-N, ULIP-2, Uni3D),
then organizes results by output dimension: whole_mesh, part_meshes, articulation.

The reference image (for alignment metrics and VLM-as-judge) is resolved
**exclusively** through the prompt suite's ``manifest.json`` — exactly one
image per sample, fixed by the released suite. There are no fallbacks to
``preview.jpg`` or to specific view indices; if the prompt suite has no
entry for a sample the eval errors out loudly.

Usage:
    python -m artibench.eval.eval_paired \
        --pred_dir /path/to/predictions \
        --gt_dir  $ARTIBENCH_DATA_ROOT/category_rich/gt \
        --out_dir out/category_rich/paired \
        [--object_id 103425] \
        [--prompt_suite_dir $ARTIBENCH_DATA_ROOT/category_rich]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from artibench.core.urdf_parser import parse_urdf
from artibench.metrics.reconstruction import eval_joint_params, eval_part_geometry
from artibench.utils.timing import PhaseTimer, phase, push_timer, pop_timer
from artibench.eval._dimensions import (
    extract_whole_mesh_paired,
    extract_part_meshes_paired,
    extract_articulation_paired,
)
from artibench.eval._prompt_suite import PromptEntry, load_prompt_suite
from artibench.reporting.dimension_report import (
    aggregate_dimensions,
    generate_analysis,
    write_results,
)


def discover_pairs(
    pred_dir: str, gt_dir: Optional[str], urdf_name: str = "mobility.urdf",
    object_id: str | None = None,
    object_ids: Optional[List[str]] = None,
    reverse: bool = False,
) -> List[Tuple[str, str, Optional[str]]]:
    """Find pred/gt URDF pairs by object ID.

    When ``gt_dir`` is None (no-GT / wild mode), the gt-side path is omitted
    and entries are returned as ``(oid, pred_urdf, None)``.

    ``object_ids`` filters the discovered pairs to a specific subset, preserving
    the input order. ``reverse`` reverses the iteration order of a normal
    full-directory scan; ignored when ``object_id`` or ``object_ids`` is set.
    """
    pred_root = Path(pred_dir)
    gt_root = Path(gt_dir) if gt_dir else None
    pairs: List[Tuple[str, str, Optional[str]]] = []

    if object_id is not None:
        pred_urdf = pred_root / object_id / urdf_name
        if not pred_urdf.is_file():
            return pairs
        if gt_root is None:
            pairs.append((object_id, str(pred_urdf), None))
            return pairs
        gt_urdf = gt_root / object_id / urdf_name
        if gt_urdf.is_file():
            pairs.append((object_id, str(pred_urdf), str(gt_urdf)))
        return pairs

    if object_ids is not None:
        for oid in object_ids:
            pred_urdf = pred_root / oid / urdf_name
            if not pred_urdf.is_file():
                continue
            if gt_root is None:
                pairs.append((oid, str(pred_urdf), None))
                continue
            gt_urdf = gt_root / oid / urdf_name
            if gt_urdf.is_file():
                pairs.append((oid, str(pred_urdf), str(gt_urdf)))
        return pairs

    entries = sorted(pred_root.iterdir(), reverse=reverse)
    for entry in entries:
        if not entry.is_dir():
            continue
        pred_urdf = entry / urdf_name
        if not pred_urdf.is_file():
            continue
        if gt_root is None:
            pairs.append((entry.name, str(pred_urdf), None))
            continue
        gt_urdf = gt_root / entry.name / urdf_name
        if gt_urdf.is_file():
            pairs.append((entry.name, str(pred_urdf), str(gt_urdf)))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ArtiBench: Paired reconstruction evaluation."
    )
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", default=None,
                        help="Ground-truth root. Omit (or pass --no_gt) for "
                             "pred-only suites like in_the_wild — C1/C2 "
                             "will auto-skip and only alignment + VLM run.")
    parser.add_argument("--no_gt", action="store_true",
                        help="Pred-only mode: no GT available, skip C1/C2.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--urdf_name", default="mobility.urdf")
    parser.add_argument("--object_id", default=None, help="Evaluate a single object.")
    parser.add_argument("--object_ids", nargs="+", default=None,
                        help="Evaluate this explicit list of object IDs (space-separated). "
                             "Preserves the given order; used by per-GPU shard launchers.")
    parser.add_argument("--object_ids_file", default=None,
                        help="Path to a newline-separated file of object IDs. Merged with "
                             "--object_ids when both are provided.")
    parser.add_argument("--reverse", action="store_true",
                        help="Iterate the discovered pair list in reverse sorted order. "
                             "Ignored when --object_id or --object_ids is set.")
    parser.add_argument("--max_objects", type=int, default=None,
                        help="Limit evaluation to first N objects (for quick sanity checks).")

    parser.add_argument("--n_states", type=int, default=5)
    parser.add_argument("--n_link_points", type=int, default=4096)
    parser.add_argument("--n_whole_points", type=int, default=32768,
                        help="Surface-area-weighted point budget for the "
                             "rest-state whole-assembled-mesh CD "
                             "(rest_overall_cd). Decoupled from "
                             "--n_link_points so over-/under-segmented "
                             "models all sample the same surface uniformly.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--axis_cutoff", type=float, default=0.25)
    parser.add_argument("--origin_cutoff_ratio", type=float, default=0.03,
                        help="Per-joint origin threshold = ratio × min(parent, child) "
                             "link bbox diagonal (default 0.03; matches reference).")
    parser.add_argument("--origin_cutoff_fallback", type=float, default=0.05,
                        help="Absolute origin threshold (m) used when the GT joint "
                             "has no surrounding geometry.")
    parser.add_argument("--gt_pcl_dir", default=None)
    parser.add_argument("--expected_ids", default=None,
                        help="Suite manifest.json. Without 3D ground truth there is no gt/ "
                             "to enumerate, so this supplies the id set coverage is measured "
                             "against; otherwise it is whatever you produced and always 100%.")
    # Per-side (center, scale) normalization, applied to both C1 joint params and C2 per-link CD/IoU.
    parser.add_argument("--recon_norm", "--c2_norm",
                        dest="recon_norm",
                        default="none", choices=["none", "bbox", "mesh"])
    parser.add_argument("--skip_c1", action="store_true")
    parser.add_argument("--skip_c2", action="store_true")

    # Prompt suite (single source of truth for the reference image)
    parser.add_argument("--prompt_suite_dir", default=None,
                        help="Path to a downloaded suite directory (see "
                             "the README links). Required for "
                             "the alignment metrics and the VLM judge: it is "
                             "the sole source of each sample's reference image.")

    # Alignment metrics
    parser.add_argument("--skip_alignment", action="store_true",
                        help="Skip all alignment metrics (CLIP, CLIP-N, ULIP-2, Uni3D)")
    parser.add_argument("--skip_clip", action="store_true")
    parser.add_argument("--skip_clip_n", action="store_true")
    parser.add_argument("--skip_ulip2", action="store_true")
    parser.add_argument("--skip_uni3d", action="store_true")
    parser.add_argument("--alignment_n_pcl", type=int, default=10000,
                        help="Number of points for ULIP-2/Uni3D point cloud (default 10000)")
    parser.add_argument("--canonical_num_views", type=int, default=4)
    parser.add_argument("--canonical_azimuth_start", type=float, default=30.0)
    parser.add_argument("--canonical_azimuth_span", type=float, default=270.0)
    parser.add_argument("--canonical_elevation", type=float, default=30.0)

    # VLM-as-judge
    parser.add_argument("--vlm_judge", action="store_true",
                        help="Enable VLM-as-judge evaluation")
    parser.add_argument("--vlm_backend", default="", choices=["", "google"],
                        help="VLM backend (only the official Google GenAI API is supported)")
    parser.add_argument("--vlm_model", default="gemini-3.1-pro-preview")
    parser.add_argument("--vlm_max_qpm", type=int, default=60,
                        help="Client-side QPM cap (per process). Set to "
                             "total_budget/num_workers when running distributed "
                             "to stay under your API quota.")
    parser.add_argument("--skip_vlm_part_decomp", action="store_true")
    parser.add_argument("--skip_vlm_part_complete", action="store_true")
    parser.add_argument("--skip_vlm_motion", action="store_true")
    parser.add_argument("--skip_vlm_joint_type", action="store_true")
    parser.add_argument("--skip_vlm_joint_axis", action="store_true")

    return parser.parse_args()


def _lookup_prompt_image(
    prompt_suite: Dict[str, PromptEntry] | None, oid: str,
) -> str | None:
    """Return the path to the prompt image for ``oid`` (same image used for both
    alignment and VLM). None if no prompt suite was provided.

    Raises on prompt-suite-present-but-missing-oid so silent skips don't hide
    a typo in the suite directory.
    """
    if prompt_suite is None:
        return None
    if oid not in prompt_suite:
        raise RuntimeError(
            f"prompt suite has no entry for {oid!r}. Your predictions must "
            "use the suite's object ids; check --prompt_suite_dir points at "
            "the suite you are evaluating against."
        )
    return prompt_suite[oid].image_path


def _compute_alignment(
    oid: str,
    pred_urdf: str,
    gt_urdf: str,
    ref_image_path: str,
    args: argparse.Namespace,
    alignment_config,
) -> dict | None:
    """Compute alignment metrics for one pair."""
    from artibench.metrics._rendering import render_canonical_views
    from artibench.metrics.image_alignment import compute_all_alignment
    from artibench.core.point_cloud import sample_link_pcl_with_colors, apply_transform
    from artibench.core.forward_kinematics import compute_fk

    # Render canonical views for pred mesh model
    try:
        with phase("align.render_canonical"):
            pred_rgb, pred_normals = render_canonical_views(
                pred_urdf,
                num_views=args.canonical_num_views,
                azimuth_start=args.canonical_azimuth_start,
                azimuth_span=args.canonical_azimuth_span,
                elevation=args.canonical_elevation,
                render_normal=not args.skip_clip_n,
            )
    except Exception as e:
        print(f"  [WARN] Pred rendering failed for {oid}: {e}")
        pred_rgb, pred_normals = [], []

    # Prepare colored point cloud for ULIP-2 / Uni3D
    pcl_6d = None
    if not (args.skip_ulip2 and args.skip_uni3d):
        try:
            with phase("align.pcl_assembly"):
                pred_obj = parse_urdf(pred_urdf, prefer_collision=False)
                # Assemble whole-object colored point cloud at rest pose
                link_pcls_6d = {}
                for i, link in enumerate(pred_obj.links):
                    if not link.mesh_paths:
                        continue
                    pcl = sample_link_pcl_with_colors(
                        link.mesh_paths, link.origins,
                        args.alignment_n_pcl, seed=args.seed + i,
                    )
                    if np.any(pcl[:, :3] != 0):
                        link_pcls_6d[link.name] = pcl

                if link_pcls_6d:
                    rest_fk = compute_fk(pred_obj, {})
                    parts = []
                    for link_name, pcl in link_pcls_6d.items():
                        T = rest_fk.get(link_name, np.eye(4))
                        xyz_t = apply_transform(pcl[:, :3], T)
                        rgb = pcl[:, 3:]
                        parts.append(np.concatenate([xyz_t, rgb], axis=1))
                    merged = np.concatenate(parts, axis=0)
                    # Subsample to target count
                    if len(merged) > args.alignment_n_pcl:
                        rng = np.random.default_rng(args.seed)
                        idx = rng.choice(len(merged), args.alignment_n_pcl, replace=False)
                        pcl_6d = merged[idx]
                    elif len(merged) > 0:
                        rng = np.random.default_rng(args.seed)
                        idx = rng.choice(len(merged), args.alignment_n_pcl, replace=True)
                        pcl_6d = merged[idx]
        except Exception as e:
            print(f"  [WARN] PCL assembly failed for {oid}: {e}")

    alignment_result = compute_all_alignment(
        ref_image_path=ref_image_path,
        rendered_view_paths=pred_rgb,
        normal_map_paths=pred_normals,
        point_cloud_6d=pcl_6d,
        config=alignment_config,
        skip_clip=args.skip_clip,
        skip_clip_n=args.skip_clip_n,
        skip_ulip2=args.skip_ulip2,
        skip_uni3d=args.skip_uni3d,
    )
    return alignment_result


def _eval_one_pair(
    oid: str, pred_urdf: str, gt_urdf: Optional[str], args: argparse.Namespace,
    gt_pcl_root, alignment_config=None, vlm_config=None,
    prompt_suite: Dict[str, PromptEntry] | None = None,
) -> dict | None:
    """Evaluate a single pred/gt pair, return dimension-organized result.

    When ``gt_urdf`` is None, runs in no-GT mode (alignment + VLM only).
    """
    timer = PhaseTimer(tag=oid)
    push_timer(timer)
    try:
        try:
            with phase("parse"):
                # Visual meshes, not collision proxies: cached GT point clouds were sampled from them.
                pred_obj = parse_urdf(pred_urdf, prefer_collision=False)
                gt_obj = (
                    parse_urdf(gt_urdf, prefer_collision=False)
                    if gt_urdf is not None else None
                )
        except Exception as e:
            print(f"  [WARN] Failed to parse {oid}: {e}")
            return None

        gt_cached_pcls = None
        if gt_pcl_root is not None:
            npz_path = gt_pcl_root / oid / "link_pcls.npz"
            if npz_path.exists():
                gt_cached_pcls = dict(np.load(npz_path))

        c1_result = None
        c2_result = None

        pred_norm = None
        gt_norm = None
        if gt_obj is not None and args.recon_norm != "none":
            try:
                from artibench.metrics._shared.assembly import (
                    compute_rest_normalization_bbox,
                    compute_rest_normalization_mesh,
                )
                if args.recon_norm == "mesh":
                    pred_norm = compute_rest_normalization_mesh(pred_obj, seed=args.seed)
                    gt_norm = compute_rest_normalization_mesh(gt_obj, seed=args.seed)
                elif args.recon_norm == "bbox":
                    # bbox mode needs per-link point clouds, so eval_part_geometry computes it internally.
                    pred_norm = None
                    gt_norm = None
            except Exception as e:
                print(f"  [WARN] {oid}: recon normalization failed ({e}); falling back to raw frame")

        if gt_obj is not None and not args.skip_c1:
            try:
                with phase("C1"):
                    c1_result = eval_joint_params(
                        pred_obj, gt_obj,
                        axis_cutoff=args.axis_cutoff,
                        origin_cutoff_ratio=args.origin_cutoff_ratio,
                        origin_cutoff_fallback=args.origin_cutoff_fallback,
                        pred_norm=pred_norm,
                        gt_norm=gt_norm,
                    )
            except Exception as e:
                print(f"  [WARN] C1 failed for {oid}: {e}")

        if gt_obj is not None and not args.skip_c2:
            try:
                with phase("C2"):
                    c2_result = eval_part_geometry(
                        pred_obj, gt_obj,
                        n_states=args.n_states,
                        n_link_points=args.n_link_points,
                        n_whole_points=args.n_whole_points,
                        seed=args.seed,
                        gt_cached_pcls=gt_cached_pcls,
                        recon_norm=args.recon_norm,
                    )
            except Exception as e:
                print(f"  [WARN] C2 failed for {oid}: {e}")

        # Threads work here because both stages block on Blender subprocesses and HTTP, releasing the GIL.
        # The prompt-suite manifest is the only source of the reference image; there is no GT-side fallback.
        alignment_result = None
        vlm_result = None
        prompt_image = _lookup_prompt_image(prompt_suite, oid)

        def _run_alignment():
            return _compute_alignment(
                oid, pred_urdf, gt_urdf, prompt_image, args, alignment_config,
            )

        def _run_vlm():
            from artibench.vlm.judge import compute_vlm_scores
            return compute_vlm_scores(
                urdf_path=pred_urdf,
                ref_image_path=prompt_image,
                config=vlm_config,
            )

        jobs: List[Tuple[str, Any]] = []  # (name, callable)
        if (
            alignment_config is not None
            and not args.skip_alignment
            and prompt_image is not None
        ):
            jobs.append(("alignment", _run_alignment))
        if vlm_config is not None and prompt_image is not None:
            jobs.append(("vlm", _run_vlm))

        if jobs:
            with phase("alignment+vlm.parallel"):
                with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                    futs = {name: ex.submit(fn) for name, fn in jobs}
                    for name, fut in futs.items():
                        try:
                            if name == "alignment":
                                alignment_result = fut.result()
                            elif name == "vlm":
                                vlm_result = fut.result()
                        except Exception as e:
                            print(f"  [WARN] {name} failed for {oid}: {e}")
    finally:
        pop_timer()
        print(timer.summary(), flush=True)

    return {
        "object_id": oid,
        "whole_mesh": extract_whole_mesh_paired(c2_result, alignment_result),
        "part_meshes": extract_part_meshes_paired(c1_result, c2_result, vlm_result),
        "articulation": extract_articulation_paired(c1_result, c2_result, vlm_result),
        "_raw": {
            "c1_joint_params": c1_result,
            "c2_part_geometry": c2_result,
            "alignment": alignment_result,
            "vlm_judge": vlm_result,
        },
    }


def main() -> None:
    args = parse_args()
    start_time = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    no_gt = bool(args.no_gt or args.gt_dir is None)
    if no_gt:
        args.skip_c1 = True
        args.skip_c2 = True

    object_ids: Optional[List[str]] = None
    if args.object_ids or args.object_ids_file:
        merged: List[str] = list(args.object_ids or [])
        if args.object_ids_file:
            with open(args.object_ids_file, "r", encoding="utf-8") as f:
                merged.extend(line.strip() for line in f if line.strip())
        # Preserve order; drop duplicates.
        seen: set[str] = set()
        object_ids = []
        for oid in merged:
            if oid not in seen:
                seen.add(oid)
                object_ids.append(oid)
    pairs = discover_pairs(
        args.pred_dir, args.gt_dir, args.urdf_name,
        object_id=args.object_id,
        object_ids=object_ids,
        reverse=bool(args.reverse),
    )
    single_mode = args.object_id is not None
    if args.max_objects is not None and len(pairs) > args.max_objects:
        pairs = pairs[: args.max_objects]

    # Completeness baseline: the expected sample set the predictions are reported against.
    if no_gt and args.expected_ids:
        manifest = json.loads(Path(args.expected_ids).read_text(encoding="utf-8"))
        all_gt_oids: List[str] = sorted(m["model_name"] for m in manifest["models"])
    elif no_gt:
        all_gt_oids = [oid for oid, _, _ in pairs]
    elif single_mode:
        all_gt_oids = [args.object_id]  # type: ignore[list-item]
    else:
        gt_root_path = Path(args.gt_dir)
        all_gt_oids = sorted(
            p.name for p in gt_root_path.iterdir()
            if p.is_dir() and (p / args.urdf_name).is_file()
        )
    found_oids = [oid for oid, _, _ in pairs]
    missing_oids = [oid for oid in all_gt_oids if oid not in set(found_oids)]
    if no_gt:
        print(
            f"[Paired Eval] no-GT mode — found {len(pairs)} pred sample(s); "
            "running alignment + VLM only"
        )
    else:
        print(
            f"[Paired Eval] Found {len(pairs)} pair(s)  (recon_norm={args.recon_norm})"
            f"   gt_total={len(all_gt_oids)}  missing={len(missing_oids)}"
        )

    if not pairs:
        raise RuntimeError(
            "No samples found." if no_gt else "No matched pairs found."
        )

    gt_pcl_root = Path(args.gt_pcl_dir) if args.gt_pcl_dir else None

    # Load the prompt suite once — hard-errors on an incomplete manifest.
    # This is the sole source of reference images for alignment and VLM metrics.
    wants_ref_image = (not args.skip_alignment) or args.vlm_judge
    prompt_suite: Dict[str, PromptEntry] | None = None
    if wants_ref_image:
        if not args.prompt_suite_dir:
            raise RuntimeError(
                "Alignment / VLM metrics require --prompt_suite_dir. "
                "Pass it, or disable both with --skip_alignment and omit --vlm_judge."
            )
        prompt_suite = load_prompt_suite(args.prompt_suite_dir)
        print(
            f"[Paired Eval] Prompt suite: {args.prompt_suite_dir} "
            f"({len(prompt_suite)} entries, seed-assigned views)"
        )

    # Initialize alignment config
    alignment_config = None
    if not args.skip_alignment:
        from artibench.metrics.image_alignment import AlignmentConfig, prewarm_alignment_models
        alignment_config = AlignmentConfig(n_pcl_points=args.alignment_n_pcl)
        print("[Paired Eval] Alignment metrics enabled")
        # Eagerly load CLIP / ULIP-2 / Uni3D once here so the first sample doesn't absorb ~120s of model loads inside a timed phase.
        prewarm_t0 = time.time()
        print("[Paired Eval] Pre-warming alignment models (CLIP / ULIP-2 / Uni3D)...")
        prewarm_alignment_models(
            alignment_config,
            skip_clip=args.skip_clip,
            skip_clip_n=args.skip_clip_n,
            skip_ulip2=args.skip_ulip2,
            skip_uni3d=args.skip_uni3d,
        )
        print(f"[Paired Eval] Pre-warm done in {time.time() - prewarm_t0:.1f}s")

    # Initialize VLM judge config
    vlm_config = None
    if args.vlm_judge:
        from artibench.vlm.judge import VLMJudgeConfig, ALL_ASPECTS
        from artibench.vlm.client import VLMClientConfig
        skip_map = {
            "PART_DECOMP": args.skip_vlm_part_decomp,
            "PART_COMPLETE": args.skip_vlm_part_complete,
            "MOTION": args.skip_vlm_motion,
            "JOINT_TYPE": args.skip_vlm_joint_type,
            "JOINT_AXIS": args.skip_vlm_joint_axis,
        }
        aspects = [a for a in ALL_ASPECTS if not skip_map.get(a, False)]
        if aspects:
            vlm_config = VLMJudgeConfig(
                aspects=aspects,
                client=VLMClientConfig(
                    backend=args.vlm_backend,
                    model=args.vlm_model,
                    max_qpm=int(args.vlm_max_qpm),
                ),
            )
            print(f"[Paired Eval] VLM judge enabled (aspects={aspects}, "
                  f"model={args.vlm_model}, max_qpm={args.vlm_max_qpm})")

    per_object = []
    for oid, pred_urdf, gt_urdf in tqdm(pairs, desc="Eval pairs", disable=single_mode):
        result = _eval_one_pair(
            oid, pred_urdf, gt_urdf, args, gt_pcl_root,
            alignment_config, vlm_config,
            prompt_suite=prompt_suite,
        )
        if result is not None:
            per_object.append(result)

    # Build aggregate
    params = {
        "n_states": args.n_states,
        "n_link_points": args.n_link_points,
        "n_whole_points": args.n_whole_points,
        "seed": args.seed,
        "axis_cutoff": args.axis_cutoff,
        "origin_cutoff_ratio": args.origin_cutoff_ratio,
        "origin_cutoff_fallback": args.origin_cutoff_fallback,
        "recon_norm": args.recon_norm,
        "mesh_source": "visual",
    }

    elapsed = time.time() - start_time

    evaluated_oids = [obj.get("object_id") for obj in per_object]
    pred_errored_oids = [
        oid for oid in found_oids if oid not in evaluated_oids
    ]
    completeness = {
        "gt_total": len(all_gt_oids),
        "pred_found": len(found_oids),
        "pred_missing": len(missing_oids),
        "missing_oids": missing_oids,
        "eval_succeeded": len(evaluated_oids),
        "eval_errored_oids": pred_errored_oids,
        "coverage_rate": (
            len(evaluated_oids) / max(1, len(all_gt_oids))
        ),
    }

    if single_mode:
        # Single-ID: aggregate IS the single sample
        aggregate = aggregate_dimensions(per_object)
        written = write_results(
            str(out_dir), aggregate, None, None, params, elapsed,
            completeness=completeness,
        )
    else:
        aggregate = aggregate_dimensions(per_object)
        analysis = generate_analysis(per_object)
        written = write_results(
            str(out_dir), aggregate, per_object, analysis, params, elapsed,
            completeness=completeness,
        )

    # Print summary
    for dim_name in ("whole_mesh", "part_meshes", "articulation"):
        dim = aggregate.get(dim_name, {})
        if dim:
            items = ", ".join(f"{k}={v:.4f}" for k, v in dim.items())
            print(f"  [{dim_name}] {items}")

    for p in written:
        print(f"  Wrote {p}")

    print(
        f"\n[Done] {len(per_object)} / {len(all_gt_oids)} pairs evaluated  "
        f"(coverage={completeness['coverage_rate']:.1%}, "
        f"pred_missing={completeness['pred_missing']}, "
        f"elapsed: {elapsed:.1f}s)"
    )


if __name__ == "__main__":
    main()
