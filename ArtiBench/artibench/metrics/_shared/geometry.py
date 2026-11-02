# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Geometric primitives: Chamfer Distance, bbox IoU/gIoU, and numeric helpers."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def bbox_center(pcl: np.ndarray) -> np.ndarray:
    """AABB center of a point cloud."""
    return (pcl.min(axis=0) + pcl.max(axis=0)) / 2.0


def bbox_bounds(pcl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box min/max corners for a point cloud."""
    return pcl.min(axis=0), pcl.max(axis=0)


def bbox_iou_from_bounds(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> float:
    """Axis-aligned bounding box IoU from precomputed min/max corners.

    For boxes that are degenerate on one or more axes (e.g. paper-thin door
    panels, 1-D wires), the IoU is computed in the *supporting subspace* —
    axes where both boxes have zero extent are dropped from the product.
    This keeps IoU=1 for two identical planar parts (where the volume-based
    formula would give 0 because inter_vol scales with the thickness axis).
    """
    return _bbox_iou_and_union_from_bounds(min_a, max_a, min_b, max_b)[0]


def _bbox_iou_and_union_from_bounds(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> Tuple[float, float, float]:
    """Internal helper: IoU + union_vol + enclosing_vol in the supporting subspace.

    All three returned volumes are measured in the *same* (sub)space — axes
    where both boxes are degenerate are dropped from every product, so gIoU
    can compose them safely without mixing dimensionalities.
    """
    ext_a = max_a - min_a
    ext_b = max_b - min_b
    keep = (ext_a > 0) | (ext_b > 0)
    # On a dropped axis the boxes still only intersect if their coordinates match: parallel panels at z=0 and z=1 do not.
    drop = ~keep
    if np.any(drop):
        if not (
            np.allclose(min_a[drop], min_b[drop])
            and np.allclose(max_a[drop], max_b[drop])
        ):
            # Disjoint along a dropped axis → no intersection.
            # Use the full 3-D box union so gIoU's enclosing volume is meaningful (it's what shrinks the gIoU below 0 for far-apart boxes).
            vol_a_full = float(np.prod(np.clip(ext_a, 1e-12, None)))
            vol_b_full = float(np.prod(np.clip(ext_b, 1e-12, None)))
            enc_min = np.minimum(min_a, min_b)
            enc_max = np.maximum(max_a, max_b)
            enc_full = float(np.prod(np.clip(enc_max - enc_min, 1e-12, None)))
            return 0.0, vol_a_full + vol_b_full, enc_full
    if not np.any(keep):
        # Both boxes are points at the same location.
        return 1.0, 1.0, 1.0
    min_a = min_a[keep]; max_a = max_a[keep]
    min_b = min_b[keep]; max_b = max_b[keep]
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    if np.any(inter_max < inter_min):
        inter_vol = 0.0
    else:
        inter_vol = float(np.prod(inter_max - inter_min))
    # Use 1e-12 as the per-axis floor only on axes that survived the drop — these axes are guaranteed to have positive extent in at least one box,
    # so the residual ε bias is a relative ~ε per dim, not a 0×∞ cancellation.
    vol_a = float(np.prod(np.clip(max_a - min_a, 1e-12, None)))
    vol_b = float(np.prod(np.clip(max_b - min_b, 1e-12, None)))
    union_vol = vol_a + vol_b - inter_vol

    enclosing_min = np.minimum(min_a, min_b)
    enclosing_max = np.maximum(max_a, max_b)
    enclosing_vol = float(np.prod(np.clip(enclosing_max - enclosing_min, 1e-12, None)))

    iou = inter_vol / union_vol if union_vol > 0 else 0.0
    return iou, union_vol, enclosing_vol


def bbox_iou(pcl_a: np.ndarray, pcl_b: np.ndarray) -> float:
    """Axis-aligned bounding box IoU between two point clouds."""
    min_a, max_a = bbox_bounds(pcl_a)
    min_b, max_b = bbox_bounds(pcl_b)
    return bbox_iou_from_bounds(min_a, max_a, min_b, max_b)


def bbox_giou(pcl_a: np.ndarray, pcl_b: np.ndarray) -> float:
    """Axis-aligned bounding box gIoU between two point clouds."""
    min_a, max_a = pcl_a.min(axis=0), pcl_a.max(axis=0)
    min_b, max_b = pcl_b.min(axis=0), pcl_b.max(axis=0)

    iou, union_vol, enclosing_vol = _bbox_iou_and_union_from_bounds(min_a, max_a, min_b, max_b)
    if enclosing_vol <= 0:
        return iou
    return iou - ((enclosing_vol - union_vol) / enclosing_vol)


def chamfer_distance_np(pcl_a: np.ndarray, pcl_b: np.ndarray) -> float:
    """Symmetric Chamfer Distance using pytorch3d."""
    import torch
    from pytorch3d.loss import chamfer_distance as _cd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        a = torch.from_numpy(pcl_a).float().to(device).unsqueeze(0)
        b = torch.from_numpy(pcl_b).float().to(device).unsqueeze(0)
        cd, _ = _cd(a, b, batch_reduction=None)
    return float(cd.cpu().numpy())


def safe_mean(values: List[float], fallback: float) -> float:
    return float(np.mean(values)) if values else fallback


def average_state_series(a: List[float], b: List[float]) -> List[float]:
    return [float((x + y) / 2.0) for x, y in zip(a, b)]
