# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
artibench.eval.eval_self
=========================
Self-quality evaluation for articulated URDFs (no GT needed).

Results organized by output dimension: whole_mesh, part_meshes, articulation.

Usage:
    python -m artibench.eval.eval_self \
        --input_root /path/to/predictions \
        --out_dir out/self_quality \
        [--object_id 103425]
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from artibench.core.forward_kinematics import validate_kinematic_tree
from artibench.core.point_cloud import load_link_mesh
from artibench.core.urdf_parser import parse_urdf
from artibench.metrics.mesh_quality import (
    StaticMeshQualityConfig,
    compute_link_watertightness,
)
from artibench.metrics.segmentation_quality import compute_segmentation_quality
from artibench.utils.timing import PhaseTimer, phase, push_timer, pop_timer
from artibench.eval._dimensions import extract_whole_mesh_self, extract_part_meshes_self, extract_articulation_self
from artibench.reporting.dimension_report import (
    aggregate_dimensions,
    generate_analysis,
    write_results,
)

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class AuditConfig:
    static_mesh: StaticMeshQualityConfig = StaticMeshQualityConfig()
    # Segmentation quality (SDF-based part overlap)
    skip_segmentation_quality: bool = False
    seg_num_samples: int = 50000
    seg_n_states: int = 3
    seg_device: str = "cuda:0"
    seg_collect_viz: bool = False
    seg_seed: int = 0


def audit_urdf(
    urdf_path: str,
    *,
    config: AuditConfig,
    object_id: Optional[str] = None,
) -> Dict[str, Any]:
    urdf_path = os.path.abspath(urdf_path)
    if object_id is None:
        parent = os.path.basename(os.path.dirname(urdf_path))
        stem = os.path.splitext(os.path.basename(urdf_path))[0]
        object_id = parent or stem

    started = time.time()

    # Default-empty so a partial report still carries the fields the extractors read.
    # still serialize cleanly through the dimension extractors.
    kinematic: Dict[str, Any] = {}
    static_mesh: Dict[str, Any] = {}
    link_watertight_quality: Dict[str, Any] = {}
    seg_quality: Optional[Dict[str, Any]] = None
    link_meshes: Dict[str, Any] = {}

    with phase("parse"):
        parsed_obj = parse_urdf(urdf_path)
        tree_validation = validate_kinematic_tree(parsed_obj)

    try:
        with phase("mesh.load_link_meshes"):
            for link in parsed_obj.links:
                if not link.mesh_paths:
                    continue
                mesh = load_link_mesh(link.mesh_paths, link.origins)
                if mesh is not None:
                    link_meshes[link.name] = mesh
    except Exception as e:
        logger.warning("link mesh loading failed for %s: %s", object_id, e)
        traceback.print_exc()

    if bool(config.static_mesh.enabled) and any(link.mesh_paths for link in parsed_obj.links):
        try:
            with phase("mesh.whole_assembled"):
                link_watertight_quality = compute_link_watertightness(
                    parsed_obj,
                    config=config.static_mesh,
                )
        except Exception as e:
            logger.warning("whole mesh quality failed for %s: %s", object_id, e)
            traceback.print_exc()

    if not config.skip_segmentation_quality:
        try:
            if len(link_meshes) >= 2:
                with phase("seg.sdf_overlap"):
                    seg_quality = compute_segmentation_quality(
                        parsed_obj, link_meshes,
                        num_samples=config.seg_num_samples,
                        n_states=config.seg_n_states,
                        device=config.seg_device,
                        collect_collision_points=config.seg_collect_viz,
                        seed=config.seg_seed,
                    )
        except Exception as e:
            logger.warning("Segmentation quality failed for %s: %s", object_id, e)
            traceback.print_exc()

    finished = time.time()

    return {
        "object_id": object_id,
        "urdf_path": urdf_path,
        "runtime_sec": float(finished - started),
        "config": {
            "static_mesh": asdict(config.static_mesh),
            "segmentation_quality": {
                "skip": bool(config.skip_segmentation_quality),
                "num_samples": int(config.seg_num_samples),
                "n_states": int(config.seg_n_states),
                "device": str(config.seg_device),
                "seed": int(config.seg_seed),
            },
        },
        "metrics": {
            "urdf_valid": tree_validation.to_dict(),
            "kinematic": kinematic,
            "static_mesh": static_mesh,
            "link_watertight_quality": link_watertight_quality,
            "segmentation_quality": seg_quality,
        },
        "_link_meshes": link_meshes,  # transient, not serialized to JSON
        "_parsed_obj": parsed_obj,
    }


# URDF discovery

def _find_urdfs(
    input_root: str,
    *,
    urdf_name: Optional[str],
    object_id: Optional[str] = None,
) -> List[str]:
    """Discover URDFs under input_root.

    Enumerates only direct children ``<input_root>/<oid>/<urdf>`` — never
    recurses into subdirs. This keeps backup directories like
    ``<oid>/_pre_remap_backup/mobility.urdf`` (and any other ``_*``-prefixed
    or hidden dir we drop alongside an oid) out of the eval; previously
    they were picked up by ``os.walk`` and showed up as spurious extra
    object_ids in results.jsonl
    twice).
    """
    root = Path(input_root)

    if object_id is not None:
        candidates = [root / object_id / urdf_name] if urdf_name else []
        if not urdf_name:
            for f in (root / object_id).glob("*.urdf"):
                candidates.append(f)
        return [str(c) for c in candidates if c.is_file()]

    urdf_paths: List[str] = []
    for child in root.iterdir():
        # Skip hidden and backup dirs at the oid level.
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        if urdf_name is not None:
            urdf_path = child / urdf_name
            if urdf_path.is_file():
                urdf_paths.append(str(urdf_path))
        else:
            for f in child.glob("*.urdf"):
                urdf_paths.append(str(f))
    urdf_paths.sort()
    return urdf_paths


# Dimension extraction

def _extract_dimensions(report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract 3-dimension metrics from an audit report."""
    metrics = report.get("metrics", {})

    kinematic = metrics.get("kinematic", {}) or {}
    static_mesh = metrics.get("static_mesh", {}) or {}
    link_watertight_quality = metrics.get("link_watertight_quality", {}) or {}
    tree_valid = metrics.get("urdf_valid", {}) or {}
    seg_quality = metrics.get("segmentation_quality")

    return {
        "whole_mesh": extract_whole_mesh_self(static_mesh, link_watertight_quality),
        "part_meshes": extract_part_meshes_self(
            static_mesh,
            seg_quality_result=seg_quality,
            link_watertight_quality_result=link_watertight_quality,
        ),
        "articulation": extract_articulation_self(kinematic, tree_valid, seg_quality_result=seg_quality),
    }


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ArtiBench: Self-quality evaluation."
    )
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--urdf_name", default="mobility.urdf")
    parser.add_argument("--object_id", default=None, help="Evaluate a single object.")
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument(
        "--expected_root",
        default=None,
        help="Optional root containing the expected object IDs for completeness "
             "accounting (typically the GT root for benchmark runs).",
    )

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--skip_static_mesh", action="store_true")

    # Segmentation quality (SDF-based)
    parser.add_argument("--skip_segmentation_quality", action="store_true",
                        help="Skip SDF-based part overlap analysis.")
    parser.add_argument("--seg_num_samples", type=int, default=50000,
                        help="Number of volume samples for SDF overlap (default 50000).")
    parser.add_argument("--seg_n_states", type=int, default=3,
                        help="Number of articulation states per joint for overlap (default 3).")
    parser.add_argument("--seg_device", type=str, default="cuda:0",
                        help="CUDA device for SDF queries (default cuda:0).")

    # Visualization
    parser.add_argument("--export_overlap_viz", action="store_true",
                        help="Export SDF-based overlap visualization as GLB files.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = AuditConfig(
        static_mesh=StaticMeshQualityConfig(enabled=not args.skip_static_mesh),
        skip_segmentation_quality=args.skip_segmentation_quality,
        seg_num_samples=args.seg_num_samples,
        seg_n_states=args.seg_n_states,
        seg_device=args.seg_device,
        seg_collect_viz=args.export_overlap_viz,
        seg_seed=args.seed,
    )

    urdf_paths = _find_urdfs(
        args.input_root, urdf_name=args.urdf_name, object_id=args.object_id,
    )
    if args.max_objects > 0:
        urdf_paths = urdf_paths[:args.max_objects]

    single_mode = args.object_id is not None
    print(f"[Self Eval] Found {len(urdf_paths)} URDF(s)")

    if not urdf_paths:
        raise RuntimeError("No URDFs found.")

    per_object: List[Dict[str, Any]] = []
    failed_oids: List[str] = []

    for urdf_path in urdf_paths:
        obj_id = os.path.basename(os.path.dirname(os.path.abspath(urdf_path)))

        timer = PhaseTimer(tag=obj_id)
        push_timer(timer)
        try:
            try:
                report = audit_urdf(urdf_path, config=config, object_id=obj_id)
            except Exception as e:
                print(f"  [WARN] Failed to audit {obj_id}: {e}")
                traceback.print_exc()
                failed_oids.append(obj_id)
                continue
        finally:
            pop_timer()
            print(timer.summary(), flush=True)

        # Export overlap visualization if requested
        if args.export_overlap_viz:
            _meshes = report.get("_link_meshes", {})
            _obj = report.get("_parsed_obj")
            _seg_q = report.get("metrics", {}).get("segmentation_quality")
            if _meshes and _obj and _seg_q:
                from artibench.visualization.overlap_viz import export_overlap_glb
                viz_dir = out_dir / "visualization" / "overlap" / obj_id
                try:
                    glb_paths = export_overlap_glb(
                        _obj, _meshes, _seg_q, str(viz_dir),
                        n_states=args.seg_n_states,
                    )
                    print(f"  [{obj_id}] overlap viz -> {len(glb_paths)} GLB(s) in {viz_dir}")
                except Exception as e:
                    print(f"  [{obj_id}] overlap viz failed: {e}")
                    traceback.print_exc()

        # Strip collision_points (ndarray) before JSON serialization
        _seg_q = report.get("metrics", {}).get("segmentation_quality")
        if _seg_q:
            for pp in _seg_q.get("rest_overlap", {}).get("per_pair", []):
                pp.pop("collision_points", None)
            for jt in _seg_q.get("articulated_overlap", {}).get("per_joint", []):
                for st in jt.get("states", []):
                    st.pop("collision_points", None)

        # Extract dimension metrics
        dims = _extract_dimensions(report)
        per_object.append({
            "object_id": obj_id,
            **dims,
        })

        if not single_mode:
            print(f"  [{obj_id}] done ({report.get('runtime_sec', 0):.1f}s)")

    # Aggregate and write dimension report
    params = {
        "seed": args.seed,
        "seg_num_samples": args.seg_num_samples,
        "seg_n_states": args.seg_n_states,
    }
    elapsed = time.time() - start_time

    found_oids = [
        os.path.basename(os.path.dirname(os.path.abspath(p))) for p in urdf_paths
    ]
    if args.expected_root:
        expected_paths = _find_urdfs(
            args.expected_root, urdf_name=args.urdf_name, object_id=args.object_id,
        )
        expected_oids = [
            os.path.basename(os.path.dirname(os.path.abspath(p)))
            for p in expected_paths
        ]
    else:
        expected_oids = found_oids
    evaluated_oids = [obj.get("object_id") for obj in per_object]
    missing_oids = [oid for oid in expected_oids if oid not in set(found_oids)]
    eval_errored_oids = sorted(set(failed_oids) | (set(found_oids) - set(evaluated_oids)))
    completeness = {
        "gt_total": len(expected_oids),
        "pred_found": len(found_oids),
        "pred_missing": len(missing_oids),
        "missing_oids": missing_oids,
        "eval_succeeded": len(evaluated_oids),
        "eval_errored_oids": eval_errored_oids,
        "coverage_rate": len(evaluated_oids) / max(1, len(expected_oids)),
    }

    if single_mode:
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

    for dim_name in ("whole_mesh", "part_meshes", "articulation"):
        dim = aggregate.get(dim_name, {})
        if dim:
            items = ", ".join(f"{k}={v:.4f}" for k, v in list(dim.items())[:5])
            print(f"  [{dim_name}] {items}")

    for p_path in written:
        print(f"  Wrote {p_path}")

    print(f"\n[Done] {len(per_object)} objects evaluated  (elapsed: {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
