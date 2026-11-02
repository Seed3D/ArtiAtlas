# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh


@dataclass(frozen=True)
class StaticMeshQualityConfig:
    enabled: bool = True


def _load_obj_with_position_dedup(mesh_path: str) -> Optional[trimesh.Trimesh]:
    """Load one OBJ and dedup vertices by 3D position only.

    Undoes the OBJ format's (v, vt, vn) split-vertex artifact: trimesh
    materialises one Python vertex per (v, vt, vn) tuple, so adjacent faces
    that share a 3D edge but live in different UV islands appear as
    distinct vertex IDs and break is_watertight downstream. Disk file is
    never modified.
    """
    try:
        m = trimesh.load(mesh_path, force="mesh", process=False)
    except Exception:
        return None
    if not isinstance(m, trimesh.Trimesh) or len(m.vertices) == 0 or len(m.faces) == 0:
        return None
    verts = np.asarray(m.vertices, dtype=np.float64).round(6)
    _, idx_first, inverse = np.unique(verts, axis=0, return_index=True, return_inverse=True)
    return trimesh.Trimesh(
        vertices=m.vertices[idx_first], faces=inverse[m.faces], process=False
    )


def _size_weight(mesh: trimesh.Trimesh, object_diag: float) -> float:
    """Per-link size weight for the volume-weighted aggregate.

    Returns ``max(convex_hull.volume, surface_area × object_diag · 1e-3)``,
    with a final fallback to AABB volume if both fail. Taking the max (not
    a switch on hull == 0) means even near-degenerate panels with a tiny
    positive hull volume — common when CAD exports introduce micron-level
    thickness noise on what is logically a flat surface — still contribute
    an area-proportional weight equivalent to "if the panel were 0.1% of
    object_diag thick". Without this, a 1 m × 1 m panel with 0.5 µm of
    numerical thickness would weigh ≈ 5e-7 m³ when it should weigh ≈ 1e-3.
    """
    try:
        hull_v = float(mesh.convex_hull.volume)
    except Exception:
        hull_v = 0.0
    if not np.isfinite(hull_v) or hull_v < 0.0:
        hull_v = 0.0
    fallback_thickness = max(float(object_diag) * 1e-3, 1e-9)
    try:
        area = float(mesh.area)
    except Exception:
        area = 0.0
    area_v = area * fallback_thickness if (area > 0.0 and np.isfinite(area)) else 0.0
    w = max(hull_v, area_v)
    if w > 0.0:
        return w
    try:
        return float(mesh.bounding_box.volume)
    except Exception:
        return 0.0


def compute_link_watertightness(
    obj: Any,
    *,
    config: StaticMeshQualityConfig,
    weld_tol_rel: float = 1e-5,
) -> Dict[str, Any]:
    """Per-link watertightness for the FK-assembled rest pose.

    Pipeline per link:
      1. Load every ``<mesh>`` element listed under the link with
         ``process=False`` and undo OBJ split-vertex artifacts via a
         position-only dedup. Apply each mesh's ``<origin>`` transform.
      2. Concatenate the link's parts and apply the link's FK rest-pose
         transform. The link is now expressed in the object frame.
      3. Weld vertices across the concatenated link via
         ``trimesh.merge_vertices(digits_vertex=…)`` with precision
         ``object_diag × weld_tol_rel`` (default 0.001% of the assembled
         object's AABB diagonal). This bridges authoring-noise gaps
         between OBJs that the artist meant to share an edge, without
         collapsing intentionally distinct vertices.
      4. Run ``is_watertight`` and convex-hull volume on the welded link
         mesh. Convex-hull volume is well-defined whether the mesh is
         watertight or not; it is the per-link "size weight" used by the
         volume-weighted aggregate.

    Aggregates (both higher-better, dimensionless, in [0, 1]):
      * ``link_watertight_volume_ratio`` —
            ``Σ hull_vol(link) · 1[link is watertight] / Σ hull_vol(link)``.
            Volume-weighted so a tiny broken handle can't dominate a clean
            cabinet, but a broken cabinet body cannot be hidden by clean
            handles either.
      * ``fraction_watertight_links`` — simple count fraction, useful as a
            secondary diagnostic.

    Both aggregates are per-link properties of the object and live in the
    ``part_meshes`` dimension, not ``whole_mesh``: they describe how many /
    how much of the object's per-link decomposition is geometrically valid,
    not the assembled object surface as a whole. Self-quality has no
    meaningful single "whole-object" watertightness signal because most
    articulated objects are not designed as a single closed surface
    (joints are deliberate gaps).

    Centering / sphere-normalisation are unnecessary: watertightness, area,
    and volume are intrinsic to the surface, and the weld tolerance is
    already expressed in object-relative units.
    """
    result: Dict[str, Any] = {
        "evaluated": False,
        "num_links_total": int(len(getattr(obj, "links", []))),
        "num_links_evaluated": 0,
        "metrics": None,
    }
    if not bool(config.enabled):
        return result

    from artibench.core.forward_kinematics import compute_fk

    rest_fk = compute_fk(obj, {})

    # Step 1+2: load OBJs of each link, dedup, apply origin + FK; concat without cross-OBJ welding yet.
    link_meshes_fk: Dict[str, trimesh.Trimesh] = {}
    for link in getattr(obj, "links", []):
        if not link.mesh_paths:
            continue
        parts: List[trimesh.Trimesh] = []
        for mesh_path, mesh_origin in zip(link.mesh_paths, link.origins):
            m = _load_obj_with_position_dedup(mesh_path)
            if m is None:
                continue
            if not np.allclose(mesh_origin, np.eye(4), atol=1e-8):
                m.apply_transform(mesh_origin)
            parts.append(m)
        if not parts:
            continue
        combined = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
        link_T = rest_fk.get(link.name, np.eye(4))
        if not np.allclose(link_T, np.eye(4), atol=1e-8):
            combined.apply_transform(link_T)
        link_meshes_fk[link.name] = combined

    result["num_links_evaluated"] = len(link_meshes_fk)
    if not link_meshes_fk:
        return result

    # Object scale: AABB diagonal across all FK-applied link meshes.
    all_mins = np.array(
        [m.bounds[0] for m in link_meshes_fk.values()], dtype=np.float64
    )
    all_maxs = np.array(
        [m.bounds[1] for m in link_meshes_fk.values()], dtype=np.float64
    )
    object_diag = float(np.linalg.norm(all_maxs.max(axis=0) - all_mins.min(axis=0)))
    if object_diag < 1e-9:
        object_diag = 1.0
    weld_eps = max(object_diag * weld_tol_rel, 1e-12)
    weld_digits = int(max(0, min(8, np.ceil(-np.log10(weld_eps)))))

    # Step 3+4: per-link weld + watertight + size weight
    n_watertight = 0
    sum_w = 0.0
    sum_w_wt = 0.0
    per_link_summary: List[Dict[str, Any]] = []
    for name, m in link_meshes_fk.items():
        m_welded = m.copy()
        try:
            m_welded.merge_vertices(digits_vertex=weld_digits)
        except Exception:
            pass
        wt = bool(m_welded.is_watertight)
        w = _size_weight(m_welded, object_diag)
        if wt:
            n_watertight += 1
            sum_w_wt += w
        sum_w += w
        per_link_summary.append({
            "link_name": name,
            "is_watertight": wt,
            "size_weight": w,
            "num_vertices_welded": int(len(m_welded.vertices)),
            "num_faces": int(len(m_welded.faces)),
        })

    link_watertight_volume_ratio = (sum_w_wt / sum_w) if sum_w > 0 else 0.0
    fraction_watertight_links = (
        float(n_watertight) / float(len(link_meshes_fk))
        if link_meshes_fk else 0.0
    )

    result["evaluated"] = True
    result["metrics"] = {
        "link_watertight_volume_ratio": float(link_watertight_volume_ratio),
        "fraction_watertight_links": float(fraction_watertight_links),
        "num_links_evaluated": int(len(link_meshes_fk)),
        "object_diag": float(object_diag),
        "weld_eps_abs": float(weld_eps),
        "weld_digits": int(weld_digits),
        "weld_tol_rel": float(weld_tol_rel),
        "_per_link": per_link_summary,
    }
    return result


