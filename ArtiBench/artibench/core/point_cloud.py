# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
artibench.core.point_cloud
==========================
Mesh loading, surface sampling, and point cloud transforms.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# Suppress trimesh warnings about degenerate faces etc.
logging.getLogger("trimesh").setLevel(logging.ERROR)


def load_link_mesh(
    mesh_paths: List[str],
    origins: List[np.ndarray],
) -> trimesh.Trimesh | None:
    """Load per-link meshes, apply origins, and combine into one Trimesh.

    Args:
        mesh_paths: List of absolute paths to mesh files (.obj/.ply/.stl).
        origins:    List of 4x4 origin transforms for each mesh.

    Returns:
        Combined trimesh.Trimesh, or None if loading fails.
    """
    if not mesh_paths:
        return None

    parts: list[trimesh.Trimesh] = []
    for path, origin in zip(mesh_paths, origins):
        try:
            # process=False to keep raw geometry: SDF overlap consumes these meshes via cubvh and depends on stable face winding.
            # The whole-mesh quality metric does its own per-OBJ load with position-only vertex dedup and is not affected.
            mesh = trimesh.load(path, force="mesh", process=False)
        except Exception as e:
            logger.warning("Failed to load mesh %s: %s", path, e)
            continue
        if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            continue
        if not np.allclose(origin, np.eye(4), atol=1e-8):
            mesh.apply_transform(origin)
        parts.append(mesh)

    if not parts:
        return None

    combined = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    if not hasattr(combined, "faces") or len(combined.faces) == 0:
        return None
    return combined


def sample_link_pcl(
    mesh_paths: List[str],
    origins: List[np.ndarray],
    n_points: int,
    seed: int | None = None,
) -> np.ndarray:
    """Load per-link meshes, combine, and sample surface points.

    Args:
        mesh_paths: List of absolute paths to mesh files (.obj/.ply/.stl).
        origins:    List of 4×4 origin transforms to apply to each mesh
                    before combining (from URDF <origin> under <collision>/<visual>).
        n_points:   Number of surface points to sample from the combined mesh.

    Returns:
        np.ndarray of shape (n_points, 3).
        If loading fails or mesh has no faces, returns zeros.
    """
    if not mesh_paths:
        return np.zeros((n_points, 3), dtype=np.float32)

    parts: list[trimesh.Trimesh] = []
    for path, origin in zip(mesh_paths, origins):
        try:
            mesh = trimesh.load(path, force="mesh", process=False)
        except Exception as e:
            logger.warning("Failed to load mesh %s: %s", path, e)
            continue
        if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            continue
        # Apply origin transform
        if not np.allclose(origin, np.eye(4), atol=1e-8):
            mesh.apply_transform(origin)
        parts.append(mesh)

    if not parts:
        logger.warning("No valid mesh parts; returning zero cloud.")
        return np.zeros((n_points, 3), dtype=np.float32)

    combined = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]

    if not hasattr(combined, "faces") or len(combined.faces) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    # Sample surface points trimesh uses numpy legacy random internally; seed for reproducibility
    if seed is not None:
        _rng_state = np.random.get_state()
        np.random.seed(seed % (2**31))
    points, _ = trimesh.sample.sample_surface(combined, n_points)
    if seed is not None:
        np.random.set_state(_rng_state)
    points = np.asarray(points, dtype=np.float32)

    # If we got fewer points than requested, pad by repeating
    if len(points) < n_points:
        repeats = int(np.ceil(n_points / max(len(points), 1)))
        points = np.tile(points, (repeats, 1))[:n_points]

    return points[:n_points]


def sample_link_pcl_with_colors(
    mesh_paths: List[str],
    origins: List[np.ndarray],
    n_points: int,
    seed: int | None = None,
) -> np.ndarray:
    """Load per-link meshes, combine, and sample surface points with interpolated colors.

    Colors are obtained from materials (MTL Kd, texture maps, or vertex colors).
    Falls back to white (1,1,1) for meshes without material information.

    Args:
        mesh_paths: List of absolute paths to mesh files (.obj/.ply/.stl).
        origins:    List of 4x4 origin transforms for each mesh.
        n_points:   Number of surface points to sample.

    Returns:
        np.ndarray of shape (n_points, 6) with [x, y, z, r, g, b] in [0, 1].
        If loading fails, returns zeros for xyz and ones for rgb.
    """
    if not mesh_paths:
        out = np.zeros((n_points, 6), dtype=np.float32)
        out[:, 3:] = 1.0
        return out

    parts: list[trimesh.Trimesh] = []
    for path, origin in zip(mesh_paths, origins):
        try:
            loaded = trimesh.load(path, force="mesh", process=False)
        except Exception as e:
            logger.warning("Failed to load mesh %s: %s", path, e)
            continue
        if not hasattr(loaded, "faces") or len(loaded.faces) == 0:
            continue

        # Resolve vertex colors from whatever visual representation exists.
        vc_resolved = False
        vis = loaded.visual

        # Strategy 1: try to_color() which handles UV-textured, vertex-colored, etc.
        if not vc_resolved:
            try:
                color_vis = vis.to_color()
                # Verify vertex_colors is actually accessible
                vc = color_vis.vertex_colors
                if vc is not None and len(vc) == len(loaded.vertices):
                    loaded.visual = color_vis
                    vc_resolved = True
            except Exception:
                pass

        # Strategy 2: extract diffuse color from material
        if not vc_resolved and hasattr(vis, "material"):
            try:
                diffuse = getattr(vis.material, "diffuse", None)
                if diffuse is not None:
                    diffuse = np.asarray(diffuse, dtype=np.uint8).flatten()
                    color = np.array([255, 255, 255, 255], dtype=np.uint8)
                    if len(diffuse) >= 3:
                        color[:len(diffuse)] = diffuse[:4]
                    loaded.visual = trimesh.visual.ColorVisuals(
                        mesh=loaded,
                        vertex_colors=np.tile(color, (len(loaded.vertices), 1)),
                    )
                    vc_resolved = True
            except Exception:
                pass

        # Strategy 3: fallback to white
        if not vc_resolved:
            loaded.visual = trimesh.visual.ColorVisuals(
                mesh=loaded,
                vertex_colors=np.full((len(loaded.vertices), 4), 255, dtype=np.uint8),
            )

        if not np.allclose(origin, np.eye(4), atol=1e-8):
            loaded.apply_transform(origin)
        parts.append(loaded)

    if not parts:
        logger.warning("No valid mesh parts; returning white cloud.")
        out = np.zeros((n_points, 6), dtype=np.float32)
        out[:, 3:] = 1.0
        return out

    combined = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]

    if not hasattr(combined, "faces") or len(combined.faces) == 0:
        out = np.zeros((n_points, 6), dtype=np.float32)
        out[:, 3:] = 1.0
        return out

    if seed is not None:
        _rng_state = np.random.get_state()
        np.random.seed(seed % (2**31))

    points, face_indices = trimesh.sample.sample_surface(combined, n_points)

    # Interpolate vertex colors at sampled points
    try:
        vc = combined.visual.vertex_colors
        if vc is not None and len(vc) == len(combined.vertices):
            vc_float = np.asarray(vc[:, :3], dtype=np.float32) / 255.0
            # Get per-face vertex colors and interpolate (use face centroid approx)
            face_vc = vc_float[combined.faces[face_indices]]  # (N, 3, 3)
            colors = face_vc.mean(axis=1)  # (N, 3) average of triangle vertices
        else:
            colors = np.ones((n_points, 3), dtype=np.float32)
    except Exception:
        colors = np.ones((n_points, 3), dtype=np.float32)

    if seed is not None:
        np.random.set_state(_rng_state)

    points = np.asarray(points, dtype=np.float32)
    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)

    if len(points) < n_points:
        repeats = int(np.ceil(n_points / max(len(points), 1)))
        points = np.tile(points, (repeats, 1))[:n_points]
        colors = np.tile(colors, (repeats, 1))[:n_points]

    return np.concatenate([points[:n_points], colors[:n_points]], axis=1)


def apply_transform(pcl: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4×4 SE(3) transform to a point cloud.

    Args:
        pcl: (N, 3) points.
        T:   (4, 4) transformation matrix.

    Returns:
        (N, 3) transformed points.
    """
    ones = np.ones((pcl.shape[0], 1), dtype=pcl.dtype)
    homo = np.concatenate([pcl, ones], axis=1)  # (N, 4)
    transformed = (T @ homo.T).T  # (N, 4)
    return transformed[:, :3].astype(np.float32)


def subsample(pcl: np.ndarray, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random subsample (or repeat-pad) to exactly n points.

    Args:
        pcl: (M, 3) input cloud.
        n:   desired number of points.
        rng: optional numpy random generator.

    Returns:
        (n, 3) subsampled cloud.
    """
    if rng is None:
        rng = np.random.default_rng()

    m = pcl.shape[0]
    if m == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if m >= n:
        idx = rng.choice(m, size=n, replace=False)
    else:
        idx = rng.choice(m, size=n, replace=True)
    return pcl[idx].astype(np.float32)
