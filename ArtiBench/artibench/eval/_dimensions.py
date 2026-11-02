# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Dimension extractors: reorganize raw metric results into the 3 output dimensions
(whole_mesh, part_meshes, articulation).

These are thin functions that pick fields from C1/C2/kinematic/static_mesh results
and group them by what aspect of the output they measure.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


# ── Whole Mesh

def _add_if_present(
    out: Dict[str, Any],
    out_key: str,
    data: Dict[str, Any],
    data_key: Optional[str] = None,
) -> None:
    key = data_key or out_key
    if key in data and data[key] is not None:
        val = data[key]
        if isinstance(val, float) and not math.isfinite(val):
            return
        out[out_key] = val


def extract_whole_mesh_self(
    static_mesh_result: Dict[str, Any],
    link_watertight_quality_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Self-quality has no whole-object signal: there is no GT to compare
    against (paired uses ``rest_overall_cd`` / alignment scores; self-
    quality has no analogue) and articulated assemblies are not designed
    as a single closed surface. Returns an empty dict so the dimension
    stays present in the report schema but contributes no headline."""
    return {}


def extract_whole_mesh_paired(
    c2_result: Optional[Dict[str, Any]],
    alignment_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if c2_result:
        _add_if_present(result, "rest_overall_cd", c2_result)
    if alignment_result:
        for key in ("clip_score", "clip_n_score", "ulip2_score", "uni3d_score"):
            val = alignment_result.get(key)
            if val is not None:
                result[key] = val
    return result


# ── Part Meshes

def extract_part_meshes_self(
    static_mesh_result: Dict[str, Any],
    seg_quality_result: Optional[Dict[str, Any]] = None,
    link_watertight_quality_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    # Headline per-link watertightness.
    lwq = link_watertight_quality_result or {}
    if lwq.get("evaluated") and lwq.get("metrics"):
        _add_if_present(result, "link_watertight_volume_ratio", lwq["metrics"])
        _add_if_present(result, "fraction_watertight_links", lwq["metrics"])
        # num_links comes from the watertight path.
        n_le = lwq["metrics"].get("num_links_evaluated")
        if isinstance(n_le, int):
            result["num_links"] = n_le
    per_link = (static_mesh_result or {}).get("per_link") or []
    evaluated = [l for l in per_link if l.get("evaluated") and l.get("metrics")]
    if static_mesh_result and (per_link or evaluated):
        per_link_summary = []
        for link in evaluated:
            m = link["metrics"]
            per_link_summary.append({
                "link_name": link.get("link_name", ""),
                "num_vertices": m.get("num_vertices", 0),
                "num_faces": m.get("num_faces", 0),
                "is_watertight": m.get("is_watertight", False),
                "is_winding_consistent": m.get("is_winding_consistent", False),
                "boundary_edges": m.get("boundary_edges", 0),
                "degenerate_faces": m.get("degenerate_faces", 0),
            })
        # Pybullet-side count overrides the watertight-path count when available (it is the authoritative per-link enumeration).
        result["num_links"] = len(evaluated)
        result["per_link"] = per_link_summary
    # Omit the ratio when no pair was evaluated (missing ≠ a perfect 0.0), but always emit failure counts so an all-fail sample stays distinguishable.
    if seg_quality_result:
        rest = seg_quality_result.get("rest_overlap", {})
        n_eval = int(rest.get("num_pairs_evaluated", 0))
        n_fail = int(rest.get("num_pairs_failed", 0))
        if n_eval > 0:
            _add_if_present(result, "rest_overlap_ratio", rest, "overlap_ratio")
        if n_eval > 0 or n_fail > 0:
            result["rest_overlap_n_pairs_evaluated"] = n_eval
            if n_fail > 0:
                result["rest_overlap_n_pairs_failed"] = n_fail
                result["rest_overlap_partial"] = True
        debug = rest.get("_debug")
        if debug:
            result["_rest_overlap_debug"] = debug
    return result


def extract_part_meshes_paired(
    c1_result: Optional[Dict[str, Any]],
    c2_result: Optional[Dict[str, Any]],
    vlm_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if c1_result is not None:
        _add_if_present(result, "count_accuracy", c1_result)
    if c2_result is not None:
        for key in ("rest_per_part_cd", "rest_per_part_miou", "rest_per_part_giou"):
            _add_if_present(result, key, c2_result)
    # VLM-as-judge scores (GT-free, uses reference image only)
    if vlm_result:
        for aspect_key in ("PART_DECOMP", "PART_COMPLETE"):
            data = vlm_result.get(aspect_key)
            if data and "score" in data and data["score"] > 0:
                metric_key = f"vlm_{aspect_key.lower()}_score"
                result[metric_key] = data["score"]
    return result


# ── Articulation

def extract_articulation_self(
    kinematic_result: Dict[str, Any],
    tree_validation: Dict[str, Any],
    seg_quality_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    # tree_validation is always present
    if tree_validation:
        result["kinematic_tree_valid"] = tree_validation.get("valid", False)
        result["kinematic_tree_valid_rate"] = (
            1.0 if tree_validation.get("valid", False) else 0.0
        )
        result["kinematic_tree_warnings"] = tree_validation.get("warnings", [])
    # SDF-based: a collision-free-rate style check cannot reliably detect concave-mesh penetrations.
    if seg_quality_result:
        artic = seg_quality_result.get("articulated_overlap", {})
        n_eval = int(artic.get("num_evaluations", 0))
        n_fail = int(artic.get("num_evaluations_failed", 0))
        if n_eval > 0:
            _add_if_present(result, "articulated_overlap_mean", artic, "mean_overlap_ratio")
            _add_if_present(result, "articulated_overlap_max",  artic, "max_overlap_ratio")
        if n_eval > 0 or n_fail > 0:
            result["articulated_overlap_n_evaluations"] = n_eval
            if n_fail > 0:
                result["articulated_overlap_n_evaluations_failed"] = n_fail
                result["articulated_overlap_partial"] = True
        per_joint = artic.get("per_joint")
        if per_joint:
            result["_articulated_overlap_debug"] = per_joint
    return result


def extract_articulation_paired(
    c1_result: Optional[Dict[str, Any]],
    c2_result: Optional[Dict[str, Any]],
    vlm_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if c1_result is not None:
        for key in (
            "type_accuracy",
            "axis_error_rad_mean",
            "origin_error_mean",
            "limit_error_mean",
            "success_rate",
        ):
            _add_if_present(result, key, c1_result)
    if c2_result is not None:
        for key in (
            "articulated_per_part_cd",
            "articulated_per_part_miou",
            "articulated_per_part_giou",
        ):
            _add_if_present(result, key, c2_result)
    # VLM-as-judge articulation scores (GT-free)
    if vlm_result:
        for aspect_key in ("JOINT_TYPE", "JOINT_AXIS", "MOTION"):
            data = vlm_result.get(aspect_key)
            if data and "score" in data and data["score"] > 0:
                metric_key = f"vlm_{aspect_key.lower()}_score"
                result[metric_key] = data["score"]
    return result
