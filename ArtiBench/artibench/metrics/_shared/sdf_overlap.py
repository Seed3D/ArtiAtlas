# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
artibench.metrics._shared.sdf_overlap
======================================
GPU-accelerated SDF-based mesh penetration analysis using cubvh.

SDF-based penetration analysis for ArtiBench segmentation quality
evaluation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import cubvh
import numpy as np
import torch
import trimesh

logger = logging.getLogger(__name__)

# ── Constants

COLLISION_SIGN_BACKEND = "cubvh"
COLLISION_CUBVH_MODE = "raystab"  # robust to non-watertight meshes


# ── Core BVH / SDF helpers

def _build_bvh(mesh: trimesh.Trimesh, device: str = "cuda:0"):
    """Build a cubvh BVH for signed-distance queries (CUDA only)."""
    vertices_t = torch.as_tensor(
        np.asarray(mesh.vertices, dtype=np.float32),
        dtype=torch.float32, device=device,
    )
    faces_t = torch.as_tensor(
        np.asarray(mesh.faces, dtype=np.int32),
        dtype=torch.int32, device=device,
    )
    return cubvh.cuBVH(vertices_t, faces_t)


def _check_sign(
    verts: torch.Tensor,
    faces: torch.Tensor,
    pts: torch.Tensor,
    *,
    backend: str = COLLISION_SIGN_BACKEND,
    mode: str = COLLISION_CUBVH_MODE,
) -> torch.Tensor:
    """Return boolean inside mask for points using cubvh."""
    if backend == "cubvh":
        bvh = cubvh.cuBVH(verts, faces.int())
        with torch.no_grad():
            sdf, _, _ = bvh.signed_distance(pts, return_uvw=False, mode=mode)
        return sdf < 0
    raise ValueError(f"Unknown backend for _check_sign: {backend}")


def _signed_distance(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    *,
    mode: str = COLLISION_CUBVH_MODE,
    device: str = "cuda:0",
) -> np.ndarray:
    """Compute signed distances using cubvh."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    bvh = _build_bvh(mesh, device=device)
    pts_t = torch.as_tensor(pts, dtype=torch.float32, device=device)
    with torch.no_grad():
        distances_t, _, _ = bvh.signed_distance(pts_t, return_uvw=False, mode=mode)
    return distances_t.cpu().numpy()


# ── Sampling helpers

def _sample_volume_points(
    mesh: trimesh.Trimesh,
    num_samples: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Robust GPU-accelerated volume sampling with cascaded fallbacks.

    Pipeline: GPU rejection sampling -> GPU surface nudge -> CPU surface fallback.
    """
    def _cpu_fallback(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
        rng_state = None
        if seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(int(seed) % (2**31))
        try:
            try:
                pts, face_idx = trimesh.sample.sample_surface_even(mesh, n)
                face_normals = mesh.face_normals[face_idx]
                ext = float(np.min(mesh.extents)) if mesh.extents.size else 1.0
                pts_in = pts - (0.01 * ext) * face_normals
                return np.asarray(pts_in, dtype=np.float32)
            except Exception:
                return trimesh.sample.sample_surface(mesh, n)[0].astype(np.float32)
        finally:
            if rng_state is not None:
                np.random.set_state(rng_state)

    try:
        device = torch.device("cuda")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        verts = torch.as_tensor(
            np.asarray(mesh.vertices, dtype=np.float32),
            dtype=torch.float32, device=device,
        )
        faces = torch.as_tensor(
            np.asarray(mesh.faces, dtype=np.int64),
            dtype=torch.int64, device=device,
        )

        min_xyz = verts.min(dim=0).values
        max_xyz = verts.max(dim=0).values
        extent = max_xyz - min_xyz
        margin = 0.01 * extent
        min_bb = min_xyz - margin
        max_bb = max_xyz + margin
        bb_size = max_bb - min_bb

        collected_list = []
        current_count = 0
        batch_size = min(max(num_samples * 2, 8192), 2_000_000)
        max_iters = 5

        for i in range(max_iters):
            if current_count >= num_samples:
                break
            rand_pts = torch.rand(
                (batch_size, 3), device=device, dtype=torch.float32,
                generator=generator,
            )
            rand_pts = rand_pts * bb_size + min_bb
            inside_mask = _check_sign(verts, faces, rand_pts)
            valid_pts = rand_pts[inside_mask]
            if valid_pts.shape[0] > 0:
                collected_list.append(valid_pts)
                current_count += valid_pts.shape[0]
            hit_rate = max(valid_pts.shape[0] / batch_size, 1e-4)
            if i == 0 and hit_rate < 0.005:
                logger.debug(
                    "Volume hit rate too low (%.2f%%); switching to surface.",
                    hit_rate * 100,
                )
                break
            needed = num_samples - current_count
            if needed > 0:
                batch_size = int(min(needed / hit_rate * 1.2, 2_000_000))

        # Stage 2: GPU surface nudge fallback
        if current_count < num_samples:
            needed = num_samples - current_count
            try:
                from pytorch3d.ops import sample_points_from_meshes
                from pytorch3d.structures import Meshes
                mesh_t = Meshes(verts=[verts], faces=[faces])
                # pytorch3d.ops.sample_points_from_meshes has no generator= kwarg,
                # so to keep this call's seeding scoped we wrap it in fork_rng — restoring global torch RNG state on exit.
                if seed is not None:
                    fork_devices = (
                        [device.index if device.index is not None else 0]
                        if device.type == "cuda" else []
                    )
                    with torch.random.fork_rng(devices=fork_devices):
                        torch.manual_seed(int(seed) + 1)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(int(seed) + 1)
                        pts, normals = sample_points_from_meshes(
                            mesh_t, num_samples=needed, return_normals=True,
                        )
                else:
                    pts, normals = sample_points_from_meshes(
                        mesh_t, num_samples=needed, return_normals=True,
                    )
                min_ext = float(extent.min())
                pts_in = pts[0] - (0.01 * min_ext) * normals[0]
                collected_list.append(pts_in)
                current_count += needed
            except (ImportError, Exception):
                pass

        if len(collected_list) > 0:
            all_pts = torch.cat(collected_list, dim=0)
            if all_pts.shape[0] >= num_samples:
                return all_pts[:num_samples].cpu().numpy()
            remaining = num_samples - all_pts.shape[0]
            cpu_pts = _cpu_fallback(mesh, remaining)
            return np.concatenate([all_pts.cpu().numpy(), cpu_pts], axis=0)

    except Exception as e:
        logger.warning(
            "sample_volume_points GPU pipeline failed: %s; CPU fallback.", e,
        )

    return _cpu_fallback(mesh, num_samples)


# ── AABB helper

def _aabb_overlap(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    pad: float = 0.02,
) -> bool:
    """Return True if the padded AABBs of two meshes overlap."""
    bounds_a = mesh_a.bounds  # (2, 3)
    bounds_b = mesh_b.bounds
    a_min = bounds_a[0] - pad
    a_max = bounds_a[1] + pad
    b_min = bounds_b[0] - pad
    b_max = bounds_b[1] + pad
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    return bool(np.all(inter_min < inter_max))


# ── Pair penetration

def compute_pair_penetration(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    *,
    num_samples: int = 50000,
    device: str = "cuda:0",
    collect_collision_points: bool = False,
    collision_target_samples: int = 512,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Compute mutual penetration between two meshes using SDF.

    Returns dict with:
        ratio_ab: fraction of A's volume points inside B
        ratio_ba: fraction of B's volume points inside A
        penetration_ratio: max(ratio_ab, ratio_ba)
        mean_penetration_depth: mean |SDF| of penetrating points
        max_penetration_depth: max |SDF| of penetrating points
        collision_points: (M, 3) ndarray if collect_collision_points=True
    """
    samples_each = max(1, num_samples // 2)
    pts_a = _sample_volume_points(mesh_a, samples_each, seed=seed)
    pts_b = _sample_volume_points(
        mesh_b,
        samples_each,
        seed=None if seed is None else int(seed) + 1,
    )

    sdf_a_on_b = _signed_distance(mesh_b, pts_a, device=device)  # negative => inside B
    sdf_b_on_a = _signed_distance(mesh_a, pts_b, device=device)  # negative => inside A

    ratio_ab = float(np.mean(sdf_a_on_b < 0.0))
    ratio_ba = float(np.mean(sdf_b_on_a < 0.0))

    # Penetrating point depths
    pen_a = -sdf_a_on_b[sdf_a_on_b < 0.0]  # positive depth
    pen_b = -sdf_b_on_a[sdf_b_on_a < 0.0]
    all_depths = np.concatenate([pen_a, pen_b]) if (pen_a.size + pen_b.size) > 0 else np.array([0.0])

    result: Dict[str, Any] = {
        "ratio_ab": ratio_ab,
        "ratio_ba": ratio_ba,
        "penetration_ratio": max(ratio_ab, ratio_ba),
        "mean_penetration_depth": float(np.mean(all_depths)),
        "max_penetration_depth": float(np.max(all_depths)),
    }

    if collect_collision_points:
        cpts_list = []
        if pen_a.size > 0:
            cpts_list.append(pts_a[sdf_a_on_b < 0.0])
        if pen_b.size > 0:
            cpts_list.append(pts_b[sdf_b_on_a < 0.0])
        if cpts_list:
            all_cpts = np.concatenate(cpts_list, axis=0)
            # Subsample if too many
            if len(all_cpts) > collision_target_samples:
                rng = np.random.default_rng(None if seed is None else int(seed) + 2)
                idx = rng.choice(len(all_cpts), collision_target_samples, replace=False)
                all_cpts = all_cpts[idx]
            result["collision_points"] = all_cpts.astype(np.float32)
        else:
            result["collision_points"] = np.empty((0, 3), dtype=np.float32)

    return result


# from compute_pair_penetration (the actual penetrating volume points that determine the metric).


__all__ = ["compute_pair_penetration"]
