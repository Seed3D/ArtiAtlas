# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Point cloud assembly from URDF + FK: sampling, state assembly, normalization."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from artibench.core.forward_kinematics import compute_fk
from artibench.core.point_cloud import apply_transform, sample_link_pcl
from artibench.core.urdf_parser import ArticulatedObject

logger = logging.getLogger(__name__)


def prepare_link_pcls(
    obj: ArticulatedObject,
    n_link_points: int = 4096,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """Sample rest-pose per-link point clouds."""
    link_pcls: Dict[str, np.ndarray] = {}
    for i, link in enumerate(obj.links):
        if not link.mesh_paths:
            continue
        link_seed = None if seed is None else seed + i
        pcl = sample_link_pcl(link.mesh_paths, link.origins, n_link_points, seed=link_seed)
        if np.any(pcl != 0):
            link_pcls[link.name] = pcl
    return link_pcls


def assemble_at_state(
    obj: ArticulatedObject,
    link_pcls: Dict[str, np.ndarray],
    joint_states: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Assemble per-link clouds at a given joint state.

    Returns:
        (whole_object_pcl, {link_name: transformed_pcl})
    """
    fk = compute_fk(obj, joint_states)
    per_link = {}
    parts = []
    for ln, pcl in link_pcls.items():
        T = fk.get(ln, np.eye(4))
        t_pcl = apply_transform(pcl, T)
        per_link[ln] = t_pcl
        parts.append(t_pcl)
    whole = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.float32)
    return whole, per_link


def normalize_assembly(
    whole: np.ndarray,
    per_link: Dict[str, np.ndarray],
    center: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Recenter and rescale assembled point clouds."""
    whole_n = (whole - center) / scale if len(whole) > 0 else whole
    per_link_n = {k: (v - center) / scale for k, v in per_link.items()}
    return whole_n, per_link_n


def compute_rest_normalization_bbox(
    obj: ArticulatedObject,
    link_pcls: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, float]:
    """Normalization via bbox center + half bbox diagonal."""
    whole, _ = assemble_at_state(obj, link_pcls, {})
    if len(whole) == 0:
        return np.zeros(3, dtype=np.float64), 1.0
    lo, hi = whole.min(axis=0), whole.max(axis=0)
    center = (lo + hi) / 2.0
    scale = float(np.linalg.norm(hi - lo)) / 2.0  # half bbox diagonal
    return center.astype(np.float64), max(scale, 1e-8)


def compute_rest_normalization_mesh(
    obj: ArticulatedObject,
    n_points: int = 100_000,
    seed: Optional[int] = 0,
) -> Tuple[np.ndarray, float]:
    """Normalization via centroid + max-radius of the whole-object mesh.

    ``seed`` controls trimesh's surface sampling RNG so the centroid /
    max-radius are reproducible across runs (trimesh uses numpy's legacy
    ``np.random`` global state internally — we save/restore it around the
    call to avoid leaking the seed to the caller).
    """
    import trimesh
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    fk = compute_fk(obj, {})
    parts: list = []
    for link in obj.links:
        if not link.mesh_paths:
            continue
        T_link = fk.get(link.name, np.eye(4))
        for mesh_path, origin in zip(link.mesh_paths, link.origins):
            try:
                mesh = trimesh.load(mesh_path, force="mesh", process=False)
            except Exception:
                continue
            if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
                continue
            mesh.apply_transform(T_link @ origin)
            parts.append(mesh)

    if not parts:
        return np.zeros(3, dtype=np.float64), 1.0

    combined = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    if not hasattr(combined, "faces") or len(combined.faces) == 0:
        return np.zeros(3, dtype=np.float64), 1.0

    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(int(seed) % (2**31))
    try:
        points, _ = trimesh.sample.sample_surface(combined, n_points)
    finally:
        if seed is not None:
            np.random.set_state(rng_state)
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    scale = float(np.max(np.linalg.norm(points - centroid, axis=1)))
    return centroid, max(scale, 1e-8)


def load_link_meshes_for_whole(
    obj: ArticulatedObject,
) -> Dict[str, list]:
    """Load each link's mesh files once for repeated whole-assembly sampling.

    Returns ``{link_name: [(trimesh.Trimesh, origin_4x4), ...]}`` — meshes are
    returned in their own local frame (origin not yet applied) so the caller
    can re-use them across articulation states by composing FK with origin.
    """
    import trimesh
    logging.getLogger("trimesh").setLevel(logging.ERROR)
    out: Dict[str, list] = {}
    for link in obj.links:
        if not link.mesh_paths:
            continue
        ml = []
        for mesh_path, origin in zip(link.mesh_paths, link.origins):
            try:
                mesh = trimesh.load(mesh_path, force="mesh", process=False)
            except Exception:
                continue
            if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
                continue
            ml.append((mesh, origin))
        if ml:
            out[link.name] = ml
    return out


def sample_whole_assembled_pcl(
    obj: ArticulatedObject,
    link_meshes_cache: Dict[str, list],
    joint_states: Dict[str, float],
    n_total_points: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Surface-area-weighted sample of the whole assembled mesh at a state.

    Decouples the whole-mesh point budget from per-link counts so over- or
    under-segmented models all sample the same surface uniformly. trimesh's
    ``sample_surface`` is area-weighted by face triangle, so concatenating
    all transformed link meshes and sampling once gives a true uniform
    surface sample.
    """
    import trimesh
    fk = compute_fk(obj, joint_states)
    parts = []
    for link_name, mesh_list in link_meshes_cache.items():
        T_link = fk.get(link_name, np.eye(4))
        for mesh, origin in mesh_list:
            m = mesh.copy()
            m.apply_transform(T_link @ origin)
            parts.append(m)
    if not parts:
        return np.zeros((0, 3), dtype=np.float32)
    combined = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    if not hasattr(combined, "faces") or len(combined.faces) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(int(seed) % (2**31))
    try:
        pts, _ = trimesh.sample.sample_surface(combined, n_total_points)
    finally:
        if seed is not None:
            np.random.set_state(rng_state)
    return np.asarray(pts, dtype=np.float32)


def load_cached_link_pcls(
    cached: Dict[str, np.ndarray],
    n_link_points: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Deterministically subsample cached per-link point clouds."""
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    for link_name, pcl in cached.items():
        n = pcl.shape[0]
        if n <= n_link_points:
            out[link_name] = pcl.astype(np.float32)
        else:
            idx = rng.choice(n, size=n_link_points, replace=False)
            out[link_name] = pcl[idx].astype(np.float32)
    return out
