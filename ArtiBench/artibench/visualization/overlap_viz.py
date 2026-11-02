# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
artibench.visualization.overlap_viz
====================================
GLB visualization of SDF-based part overlap.

Exports translucent part meshes with opaque red collision-point spheres.
The collision points are pre-computed by ``compute_segmentation_quality``
(with ``collect_collision_points=True``) and passed in — this module only
renders them, ensuring the visualization reflects exactly what the metric
measures.

Output structure per object:
    {output_dir}/
        rest_overlap.glb                # all parts + collision spheres at rest
        {joint_name}_{pct}pct.glb       # connected pair at each state
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh

from artibench.core.urdf_parser import ArticulatedObject
from artibench.core.forward_kinematics import compute_fk
from artibench.metrics.segmentation_quality import (
    _deterministic_joint_values,
    _get_joint_link_pairs,
    _geom_has_data,
)

logger = logging.getLogger(__name__)

# Distinct colors for up to 12 parts
_PART_COLORS = [
    [31, 119, 180],   # blue
    [255, 127, 14],   # orange
    [44, 160, 44],    # green
    [148, 103, 189],  # purple
    [140, 86, 75],    # brown
    [227, 119, 194],  # pink
    [127, 127, 127],  # gray
    [188, 189, 34],   # olive
    [23, 190, 207],   # cyan
    [255, 187, 120],  # light orange
    [152, 223, 138],  # light green
    [174, 199, 232],  # light blue
]

_MAX_VIZ_POINTS = 512


def _make_translucent(
    mesh: trimesh.Trimesh,
    color_rgb: List[int],
    alpha: int = 100,
) -> trimesh.Trimesh:
    """Return a copy of the mesh with uniform translucent vertex colors."""
    m = mesh.copy()
    rgba = np.array(
        [color_rgb[0], color_rgb[1], color_rgb[2], alpha], dtype=np.uint8,
    )
    m.visual = trimesh.visual.ColorVisuals(
        mesh=m,
        vertex_colors=np.tile(rgba, (len(m.vertices), 1)),
    )
    return m


def _sphere_radius_for_pair(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
) -> float:
    """Compute sphere radius based on the smaller mesh's extent."""
    ext_a = np.linalg.norm(mesh_a.bounds[1] - mesh_a.bounds[0])
    ext_b = np.linalg.norm(mesh_b.bounds[1] - mesh_b.bounds[0])
    return float(min(ext_a, ext_b)) * 0.01


def _collision_spheres_mesh(
    points: np.ndarray,
    radius: float,
) -> Optional[trimesh.Trimesh]:
    """Create a single mesh of small red icospheres at collision points."""
    if points is None or len(points) == 0:
        return None

    if len(points) > _MAX_VIZ_POINTS:
        indices = np.random.choice(len(points), _MAX_VIZ_POINTS, replace=False)
        points = points[indices]

    unit_sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    spheres = []
    for center in points:
        s = unit_sphere.copy()
        s.apply_translation(center)
        spheres.append(s)

    combined = trimesh.util.concatenate(spheres)
    combined.visual = trimesh.visual.ColorVisuals(
        mesh=combined,
        vertex_colors=np.tile(
            [255, 0, 0, 255], (len(combined.vertices), 1),
        ).astype(np.uint8),
    )
    return combined


def export_overlap_glb(
    obj: ArticulatedObject,
    link_meshes: Dict[str, trimesh.Trimesh],
    seg_quality_result: Dict[str, Any],
    output_dir: str,
    n_states: int = 3,
) -> List[str]:
    """Export overlap visualization as GLB files using pre-computed collision points.

    Args:
        obj:                 ArticulatedObject.
        link_meshes:         {link_name: trimesh.Trimesh} in rest pose (pre-FK).
        seg_quality_result:  Output of compute_segmentation_quality(collect_collision_points=True).
        output_dir:          Directory to write GLBs.
        n_states:            Number of articulation states per joint.

    Returns:
        List of written file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    # ── Rest overlap
    rest_data = seg_quality_result.get("rest_overlap", {})
    per_pair = rest_data.get("per_pair", [])

    # Build assembled meshes (rest FK)
    rest_fk = compute_fk(obj, {})
    names = [n for n in link_meshes if _geom_has_data(link_meshes[n])]
    assembled: Dict[str, trimesh.Trimesh] = {}
    for name in names:
        m = link_meshes[name].copy()
        T = rest_fk.get(name, np.eye(4))
        m.apply_transform(T)
        assembled[name] = m

    scene = trimesh.Scene()
    for i, name in enumerate(names):
        color = _PART_COLORS[i % len(_PART_COLORS)]
        scene.add_geometry(
            _make_translucent(assembled[name], color, alpha=100),
            node_name=f"part_{name}",
        )

    # Add collision spheres from pre-computed per_pair data
    for pp in per_pair:
        cpts = pp.get("collision_points")
        if cpts is None or (hasattr(cpts, '__len__') and len(cpts) == 0):
            continue
        cpts = np.asarray(cpts, dtype=np.float32)
        if len(cpts) == 0:
            continue
        pair_names = pp.get("pair", ("?", "?"))
        # Determine sphere radius from the pair meshes
        na, nb = pair_names
        if na in assembled and nb in assembled:
            radius = _sphere_radius_for_pair(assembled[na], assembled[nb])
        else:
            radius = 0.002
        sphere_mesh = _collision_spheres_mesh(cpts, radius)
        if sphere_mesh is not None:
            scene.add_geometry(sphere_mesh, node_name=f"collision_{na}_{nb}")

    if len(scene.geometry) > 0:
        rest_path = str(out / "rest_overlap.glb")
        scene.export(rest_path, file_type="glb")
        written.append(rest_path)

    # ── Articulated overlap
    artic_data = seg_quality_result.get("articulated_overlap", {})
    per_joint = artic_data.get("per_joint", [])
    joint_lookup = {j.name: j for j in obj.joints}

    for joint_info in per_joint:
        joint_name = joint_info["joint_name"]
        parent_link = joint_info["parent_link"]
        child_link = joint_info["child_link"]

        if parent_link not in link_meshes or child_link not in link_meshes:
            continue

        # Old result files carry a single joint name and no `chain_joints`; new ones carry "j1+j2+..." plus the chain list.
        chain_joint_names = joint_info.get("chain_joints") or [joint_name]
        chain = [joint_lookup[n] for n in chain_joint_names if n in joint_lookup]
        if len(chain) != len(chain_joint_names):
            continue  # joint missing from URDF — skip this entry

        for i, state_data in enumerate(joint_info.get("states", [])):
            jv = state_data.get("joint_value")
            # New schema stores joint_value as dict {joint_name: float}; old schema stored a scalar.
            # Normalise to dict for compute_fk.
            if isinstance(jv, dict):
                joint_vals = jv
            elif isinstance(jv, (int, float)):
                joint_vals = {chain_joint_names[0]: float(jv)}
            else:
                # Fall back to a deterministic state for this single-joint case
                if len(chain) == 1:
                    fallback = _deterministic_joint_values(chain[0], n_states=n_states)
                    joint_vals = {chain[0].name: fallback[i] if i < len(fallback) else 0.0}
                else:
                    continue  # multi-joint chain with missing values — skip

            # Transform meshes to this articulation state
            fk = compute_fk(obj, joint_vals)
            mesh_p = link_meshes[parent_link].copy()
            mesh_c = link_meshes[child_link].copy()
            mesh_p.apply_transform(fk.get(parent_link, np.eye(4)))
            mesh_c.apply_transform(fk.get(child_link, np.eye(4)))

            pair_scene = trimesh.Scene()
            pair_scene.add_geometry(
                _make_translucent(mesh_p, _PART_COLORS[0], alpha=100),
                node_name=f"part_{parent_link}",
            )
            pair_scene.add_geometry(
                _make_translucent(mesh_c, _PART_COLORS[1], alpha=100),
                node_name=f"part_{child_link}",
            )

            # Add collision spheres from pre-computed data
            cpts = state_data.get("collision_points")
            if cpts is not None:
                cpts = np.asarray(cpts, dtype=np.float32)
                if len(cpts) > 0:
                    radius = _sphere_radius_for_pair(mesh_p, mesh_c)
                    sphere_mesh = _collision_spheres_mesh(cpts, radius)
                    if sphere_mesh is not None:
                        pair_scene.add_geometry(sphere_mesh, node_name="collision_points")

            pct = int(round(i / max(len(joint_info["states"]) - 1, 1) * 100))
            label = f"{pct:03d}pct"
            glb_path = str(out / f"{joint_name}_{label}.glb")
            pair_scene.export(glb_path, file_type="glb")
            written.append(glb_path)

    return written
