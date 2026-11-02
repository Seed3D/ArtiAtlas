# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
artibench.reporting.dimension_report
=====================================
Report generation for the 3-dimension output format:
whole_mesh, part_meshes, articulation.

Produces:
- results.json: averages by dimension (or single-sample)
- results.jsonl: per-sample metrics (full-set only)
- analysis.json: best/worst K per metric (full-set only)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _collect_numeric_keys(
    per_object: List[Dict[str, Any]],
    dimension: str,
) -> Dict[str, List[Tuple[str, float]]]:
    """Collect (object_id, value) pairs for each numeric metric in a dimension."""
    result: Dict[str, List[Tuple[str, float]]] = {}
    for obj in per_object:
        oid = obj.get("object_id", "")
        dim_data = obj.get(dimension, {})
        if not dim_data:
            continue
        for key, val in dim_data.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                import math
                if not math.isfinite(val):
                    continue
                result.setdefault(key, []).append((oid, val))
    return result


# Metrics where lower is better (errors, distances); the single registry of metric direction, never re-derive it from names.
_LOWER_IS_BETTER = {
    "rest_overall_cd",
    "rest_per_part_cd",
    "articulated_per_part_cd",
    "axis_error_rad_mean", "origin_error_mean", "limit_error_mean",
    "rest_overlap_ratio", "articulated_overlap_mean", "articulated_overlap_max",
}

# Diagnostic / metadata leaves that are not quality signals — best/worst ordering is meaningless for them,
# so analysis.json leaves them unranked.
_NEUTRAL_LEAVES = {
    "num_links", "num_links_evaluated", "num_parts_total",
    "object_diag", "bbox_diag", "max_part_diag",
    "weld_digits", "weld_eps_abs", "weld_tol_rel",
    "thickness_threshold_rel", "small_part_diag_ratio",
    "total_surface_area",
    "rest_overlap_n_pairs_evaluated", "rest_overlap_n_pairs_failed",
    "rest_overlap_partial",
    "articulated_overlap_n_evaluations", "articulated_overlap_n_evaluations_failed",
    "articulated_overlap_partial",
}


def aggregate_dimensions(
    per_object: List[Dict[str, Any]],
    dimensions: List[str] = ("whole_mesh", "part_meshes", "articulation"),
) -> Dict[str, Dict[str, float]]:
    """Compute mean of each metric across objects, grouped by dimension."""
    import numpy as np

    result = {}
    for dim in dimensions:
        metrics_data = _collect_numeric_keys(per_object, dim)
        dim_avg = {}
        for key, pairs in metrics_data.items():
            values = [v for _, v in pairs]
            if values:
                dim_avg[key] = float(np.mean(values))
        result[dim] = dim_avg
    return result


def metric_counts(
    per_object: List[Dict[str, Any]],
    dimensions: List[str] = ("whole_mesh", "part_meshes", "articulation"),
) -> Dict[str, Dict[str, int]]:
    """Count finite numeric values contributing to each aggregate metric."""
    result: Dict[str, Dict[str, int]] = {}
    for dim in dimensions:
        metrics_data = _collect_numeric_keys(per_object, dim)
        result[dim] = {key: len(pairs) for key, pairs in metrics_data.items()}
    return result


def generate_analysis(
    per_object: List[Dict[str, Any]],
    top_k: int = 5,
    dimensions: List[str] = ("whole_mesh", "part_meshes", "articulation"),
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Generate best/worst-K analysis per metric.

    Returns:
        {"dimension.metric": {"best": [...], "worst": [...]}}
        where each entry is {"object_id": str, "value": float}.
    """
    result = {}
    for dim in dimensions:
        metrics_data = _collect_numeric_keys(per_object, dim)
        for key, pairs in metrics_data.items():
            if len(pairs) < 2:
                continue
            # Counters such as num_links or *_n_pairs_failed have no meaningful best/worst ordering.
            if key in _NEUTRAL_LEAVES:
                continue
            lower_better = key in _LOWER_IS_BETTER
            sorted_asc = sorted(pairs, key=lambda x: x[1])
            best = sorted_asc[:top_k] if lower_better else sorted_asc[-top_k:][::-1]
            worst = sorted_asc[-top_k:][::-1] if lower_better else sorted_asc[:top_k]

            result[f"{dim}.{key}"] = {
                "best": [{"object_id": oid, "value": v} for oid, v in best],
                "worst": [{"object_id": oid, "value": v} for oid, v in worst],
            }
    return result


def write_results(
    out_dir: str,
    aggregate: Dict[str, Dict[str, float]],
    per_object: Optional[List[Dict[str, Any]]],
    analysis: Optional[Dict],
    params: Dict[str, Any],
    runtime_sec: float = 0.0,
    completeness: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Write results files.

    If per_object is None (single-ID mode), only writes results.json.
    Otherwise writes results.json + results.jsonl + analysis.json.

    If *completeness* is provided, include it at the top of results.json so
    downstream aggregators can see the per-model coverage rate + which
    sample IDs were missing.

    Returns:
        List of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []

    # results.json
    n_objects = 1 if per_object is None else len(per_object)
    results: Dict[str, Any] = {
        "n_objects": n_objects,
        "params": params,
    }
    if completeness is not None:
        results["completeness"] = completeness
    results.update(aggregate)
    if per_object is not None:
        results["metric_counts"] = metric_counts(per_object)
    results["runtime_sec"] = runtime_sec
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    written.append(results_path)

    # Full-set only: results.jsonl + analysis.json
    if per_object is not None:
        jsonl_path = os.path.join(out_dir, "results.jsonl")
        with open(jsonl_path, "w") as f:
            for obj in per_object:
                f.write(json.dumps(obj, default=str) + "\n")
        written.append(jsonl_path)

        analysis_path = os.path.join(out_dir, "analysis.json")
        with open(analysis_path, "w") as f:
            json.dump(analysis or {}, f, indent=2, default=str)
        written.append(analysis_path)

    return written
