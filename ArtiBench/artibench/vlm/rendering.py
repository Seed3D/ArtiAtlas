# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
VLM judge phase-2 rendering: MOTION, JOINT_TYPE, JOINT_AXIS.

Provides rendering functions that produce annotated frame images
(each frame = 2 camera views concatenated side-by-side with text overlay)
for VLM-as-judge articulation quality evaluation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from artibench.core.forward_kinematics import compute_fk
from artibench.core.urdf_parser import ArticulatedObject, JointInfo, parse_urdf

logger = logging.getLogger(__name__)

# Cache directory names
MOTION_CACHE_DIR = "vlm_motion_frames"
JOINT_HIGHLIGHT_CACHE_DIR = "vlm_joint_highlight_frames"

# Default azimuths for 2-view rendering
DEFAULT_AZIMUTHS = (150.0, 240.0)


def _render_proxy_decimate_target() -> int:
    """Read ARTIBENCH_RENDER_DECIMATE_FACES env var; 0 disables the proxy."""
    try:
        return max(0, int(os.environ.get("ARTIBENCH_RENDER_DECIMATE_FACES", "0")))
    except ValueError:
        return 0


def _maybe_decimate_for_render(
    mesh: trimesh.Trimesh, target_faces: int,
) -> trimesh.Trimesh:
    """Quadric-decimate ``mesh`` to at most ``target_faces`` faces.

    Used to speed up Blender Cycles renders on high-poly AIGC-input link
    meshes. Affects only the diagnostic / alignment-side renders that go
    through this helper; the paired C1/C2 metrics read the URDF link meshes
    directly elsewhere and are unaffected. No-op when ``target_faces`` is
    non-positive or the input mesh is already at or below the target.
    """
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    try:
        import open3d as o3d
    except Exception:
        return mesh
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int64))
    try:
        dec = m.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
    except Exception:
        return mesh
    verts = np.asarray(dec.vertices)
    faces = np.asarray(dec.triangles)
    if faces.size == 0 or verts.size == 0:
        return mesh
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

# Zoom-inset threshold: child bbox diagonal / whole model diagonal
ZOOM_INSET_THRESHOLD = 0.20  # child diag / whole diag below this triggers zoom
ZOOM_INSET_SIZE_FRAC = 0.45  # inset height as fraction of view height

# Ordered so consecutive indices have maximum hue separation, since adjacent URDF links are usually neighboring parts.
_PART_PALETTE = [
    "#e74c3c",  #   0: red          H≈  6
    "#3498db",  #   1: blue         H≈204
    "#f1c40f",  #   2: yellow       H≈ 48
    "#9b59b6",  #   3: purple       H≈283
    "#2ecc71",  #   4: green        H≈145
    "#e91e63",  #   5: pink         H≈340
    "#ff9800",  #   6: amber        H≈ 36
    "#009688",  #   7: teal         H≈174
    "#ff5722",  #   8: deep orange  H≈ 14
    "#3f51b5",  #   9: indigo       H≈232
    "#8bc34a",  #  10: lime         H≈ 88
    "#d81b60",  #  11: dark pink    H≈336
    "#00bcd4",  #  12: cyan         H≈187
    "#ff6f00",  #  13: dark amber   H≈ 26
    "#7c4dff",  #  14: deep purple  H≈258
    "#00e676",  #  15: bright green H≈152
    "#f39c12",  #  16: orange       H≈ 37
    "#26c6da",  #  17: light cyan   H≈186
    "#6200ea",  #  18: violet       H≈262
    "#aeea00",  #  19: yellow-green H≈ 73
]


# ── Text overlay

def draw_overlay_top_right(
    img: Image.Image,
    text: str,
    size_scale: float = 0.04,
) -> Image.Image:
    """Draw text label in top-right corner with black outline + white fill.

    Follows the ArtiAtlas annotation convention: DejaVuSans-Bold font scaled to
    4% of the smallest image dimension.  Returns a new image.
    """
    result = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(result)

    min_dim = min(result.width, result.height)
    font_size = max(int(min_dim * size_scale), 10)

    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
            )
        except OSError:
            font = ImageFont.load_default()

    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
    text_w = bbox[2] - bbox[0]
    margin = max(int(font_size * 0.4), 4)
    x = result.width - text_w - margin
    y = margin

    # Black outline
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            draw.multiline_text(
                (x + dx, y + dy), text,
                fill=(0, 0, 0, 255), font=font, spacing=2,
            )
    # White text
    draw.multiline_text(
        (x, y), text,
        fill=(255, 255, 255, 255), font=font, spacing=2,
    )
    return result.convert("RGB")


# ── View concatenation

_SEP_WIDTH = 4
_SEP_COLOR = (180, 180, 180)


def _concat_pil_images(
    imgs: List[Image.Image],
    label: str | None = None,
    size_scale: float = 0.04,
) -> Image.Image:
    """Concatenate PIL images side-by-side with a gray separator + text overlay."""
    max_h = max(im.height for im in imgs)
    n_seps = max(len(imgs) - 1, 0)
    total_w = sum(im.width for im in imgs) + n_seps * _SEP_WIDTH
    canvas = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x_off = 0
    for idx, im in enumerate(imgs):
        canvas.paste(im, (x_off, 0))
        x_off += im.width
        if idx < len(imgs) - 1:
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([x_off, 0, x_off + _SEP_WIDTH - 1, max_h - 1], fill=_SEP_COLOR)
            x_off += _SEP_WIDTH
    if label:
        canvas = draw_overlay_top_right(canvas, label, size_scale=size_scale)
    return canvas


def concat_views(
    paths: List[str],
    label: str | None = None,
    size_scale: float = 0.04,
) -> Image.Image:
    """Concatenate camera-view image files side-by-side with separator + label."""
    imgs = [Image.open(p).convert("RGB") for p in paths]
    return _concat_pil_images(imgs, label=label, size_scale=size_scale)


# ── Joint ranking

def rank_joints_by_importance(
    joints: List[JointInfo],
    link_volumes: Dict[str, float],
) -> List[Tuple[JointInfo, float]]:
    """Rank non-fixed joints by importance for VLM evaluation.

    importance(j) = vol(parent)/count(parent) + vol(child)/count(child)
    where count(link) = number of non-fixed joints the link participates in.
    Links appearing in many joints contribute less per joint.

    Returns list of (JointInfo, importance) sorted descending.
    """
    actuated = [j for j in joints if j.type != "fixed"]
    if not actuated:
        return []

    # Count how many actuated joints each link participates in
    link_joint_count: Dict[str, int] = {}
    for j in actuated:
        link_joint_count[j.parent_link] = link_joint_count.get(j.parent_link, 0) + 1
        link_joint_count[j.child_link] = link_joint_count.get(j.child_link, 0) + 1

    ranked = []
    for j in actuated:
        parent_vol = link_volumes.get(j.parent_link, 0.0)
        child_vol = link_volumes.get(j.child_link, 0.0)
        parent_count = max(link_joint_count.get(j.parent_link, 1), 1)
        child_count = max(link_joint_count.get(j.child_link, 1), 1)
        importance = parent_vol / parent_count + child_vol / child_count
        ranked.append((j, importance))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ── Link export

def _state_from_percent(
    obj: ArticulatedObject,
    percent: float,
) -> Dict[str, float]:
    """Interpolate all joints from rest (state=0) toward fully open.

    In URDF convention, state=0 is the rest pose.  We clamp 0 into
    [lower, upper] to get the rest value, then interpolate toward the
    limit endpoint that is farthest from rest.

    percent=0 → rest (state nearest 0),  percent=1 → fully open.
    """
    state: Dict[str, float] = {}
    for joint in obj.joints:
        if joint.type == "fixed":
            continue
        rest = max(joint.lower, min(0.0, joint.upper))  # clamp(0, lo, hi)
        # Pick the limit endpoint with larger displacement from rest
        if abs(joint.upper - rest) >= abs(rest - joint.lower):
            open_val = joint.upper
        else:
            open_val = joint.lower
        state[joint.name] = rest + (open_val - rest) * percent
    return state


def _export_link_objs_at_state(
    urdf_path: str,
    joint_states: Dict[str, float],
    *,
    prefer_collision: bool = False,
) -> Tuple[List[dict], str]:
    """Export per-link STL meshes at an arbitrary FK state.

    Returns (link_entries, tmp_dir) matching the format from
    one OBJ per link, at an arbitrary joint configuration rather than rest.
    """
    obj = parse_urdf(urdf_path, prefer_collision=prefer_collision)
    fk = compute_fk(obj, joint_states)

    tmp_dir = Path(tempfile.mkdtemp(prefix="artibench_vlm_links_"))
    entries: List[dict] = []

    for link_idx, link in enumerate(obj.links):
        link_tf = fk.get(link.name, np.eye(4))
        meshes: List[trimesh.Trimesh] = []
        for i, mesh_path in enumerate(link.mesh_paths):
            if not os.path.isfile(mesh_path):
                continue
            try:
                mesh = trimesh.load(mesh_path, force="mesh", process=False)
            except Exception:
                continue
            if i < len(link.origins):
                mesh.apply_transform(link.origins[i])
            mesh.apply_transform(link_tf)
            meshes.append(mesh)

        if not meshes:
            continue
        combined = trimesh.util.concatenate(meshes)

        # Optional render-proxy decimation: keep the original URDF link mesh untouched but emit a low-poly proxy STL for Blender.
        # Gated on the ARTIBENCH_RENDER_DECIMATE_FACES env var so the default behaviour is unchanged.
        bbox_vol = float(np.prod(combined.extents)) if hasattr(combined, "extents") else 0.0
        target = _render_proxy_decimate_target()
        if target > 0:
            combined = _maybe_decimate_for_render(combined, target)

        obj_out = tmp_dir / f"link_{link_idx:03d}_{link.name}.stl"
        combined.export(str(obj_out), file_type="stl")
        entries.append({
            "obj_path": str(obj_out),
            "link_index": link_idx,
            "link_name": link.name,
            "bbox_volume": bbox_vol,
        })

    return entries, str(tmp_dir)


# ── Joint axis cylinder

def _rotation_matrix_from_z_to(target: np.ndarray) -> np.ndarray:
    """3x3 rotation that maps the Z-axis to `target` (unit vector)."""
    t = target / max(float(np.linalg.norm(target)), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, t)
    s = float(np.linalg.norm(v))
    c = float(np.dot(z, t))
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def create_axis_cylinder(
    joint: JointInfo,
    fk: Dict[str, np.ndarray],
    child_mesh: trimesh.Trimesh | None,
    whole_mesh: trimesh.Trimesh | None,
    smaller_diag: float,
    whole_diag: float,
    output_path: str,
) -> str | None:
    """Create an axis cylinder STL for a non-fixed joint.

    For revolute/continuous: axis centered at the midpoint of the child
    mesh's projection onto the axis (visually centered on the child part).
    For prismatic: axis centered at child mesh centroid (slide region).

    Length = 1.2 × the whole scene's span projected onto the axis direction.

    Returns the output STL path, or None on failure.
    """
    if joint.type == "fixed":
        return None

    world_T_parent = fk.get(joint.parent_link, np.eye(4))
    joint_frame = world_T_parent @ joint.origin
    axis_dir = joint_frame[:3, :3] @ joint.axis
    axis_dir = axis_dir / max(float(np.linalg.norm(axis_dir)), 1e-12)

    # Axis center positioning
    if joint.type == "prismatic" and child_mesh is not None:
        axis_center = np.asarray(child_mesh.centroid, dtype=np.float64)
    elif child_mesh is not None:
        joint_origin = joint_frame[:3, 3]
        child_verts = np.asarray(child_mesh.vertices, dtype=np.float64)
        projections = (child_verts - joint_origin[None, :]) @ axis_dir
        t_mid = 0.5 * (float(projections.min()) + float(projections.max()))
        axis_center = joint_origin + t_mid * axis_dir
    else:
        axis_center = joint_frame[:3, 3].copy()

    # Length = 1.2 × scene span on the axis direction
    if whole_mesh is not None:
        all_verts = np.asarray(whole_mesh.vertices, dtype=np.float64)
        proj = all_verts @ axis_dir
        scene_span = float(proj.max() - proj.min())
    else:
        scene_span = whole_diag
    length = max(1.2 * scene_span, 1.5 * smaller_diag)

    # Radius: proportional to the smaller link so it's visible but not overwhelming.
    # Clamped to [0.1%, 0.5%] of whole model diagonal.
    radius = 0.04 * smaller_diag
    radius = max(radius, 0.001 * whole_diag)  # absolute min visibility
    radius = min(radius, 0.005 * whole_diag)  # cap to avoid oversize

    try:
        cyl = trimesh.creation.cylinder(
            radius=radius, height=length, sections=16,
        )
        # Transform: rotate Z→axis_dir, translate to axis_center
        T = np.eye(4)
        T[:3, :3] = _rotation_matrix_from_z_to(axis_dir)
        T[:3, 3] = axis_center
        cyl.apply_transform(T)
        cyl.export(output_path, file_type="stl")
        return output_path
    except Exception as e:
        logger.warning("Failed to create axis cylinder: %s", e)
        return None


# ── Occlusion-based view selection

def _load_mesh_from_stl(path: str) -> trimesh.Trimesh | None:
    """Load an STL mesh, return None on failure."""
    try:
        m = trimesh.load(path, force="mesh", process=False)
        if hasattr(m, "faces") and len(m.faces) > 0:
            return m
    except Exception:
        pass
    return None


def _spherical_to_cart(r: float, az_deg: float, el_deg: float) -> np.ndarray:
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    return np.array([
        r * math.cos(el) * math.cos(az),
        r * math.cos(el) * math.sin(az),
        r * math.sin(el),
    ], dtype=np.float64)


def _compute_child_occlusion_ratio(
    camera_pos: np.ndarray,
    child_mesh: trimesh.Trimesh,
    occluder_mesh: trimesh.Trimesh,
    n_samples: int = 128,
    rng: np.random.Generator | None = None,
) -> float:
    """Fraction of child surface points occluded by occluder from camera_pos.

    Returns ratio in [0, 1].  Returns 1.0 on failure (assume worst case).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Build double-sided occluder for robust ray casting
    try:
        occ = occluder_mesh.copy()
        if occ.faces is not None and len(occ.faces) > 0:
            occ.faces = np.vstack([occ.faces, occ.faces[:, ::-1]])
        _ = occ.ray
    except Exception:
        return 0.5  # unknown

    # Sample child surface points
    try:
        pts, face_idx = trimesh.sample.sample_surface(
            child_mesh, count=min(max(3 * n_samples, 64), 4096),
        )
        pts = np.asarray(pts, dtype=np.float64)
        if pts.size == 0:
            return 1.0
        if pts.shape[0] > n_samples:
            sel = rng.choice(pts.shape[0], size=n_samples, replace=False)
            pts = pts[sel]
    except Exception:
        return 0.5

    cam = camera_pos.reshape(3)
    dirs = pts - cam
    dists = np.linalg.norm(dirs, axis=1)
    valid = dists > 1e-8
    if not np.any(valid):
        return 1.0
    dirs = dirs[valid] / dists[valid, None]
    dists_v = dists[valid]

    origins = np.tile(cam, (dirs.shape[0], 1))
    try:
        loc_all, idx_ray_all, _ = occ.ray.intersects_location(
            ray_origins=origins, ray_directions=dirs,
        )
    except Exception:
        return 0.5

    if loc_all is None or len(loc_all) == 0:
        return 0.0

    loc_all = np.asarray(loc_all, dtype=np.float64)
    idx_ray_all = np.asarray(idx_ray_all, dtype=np.int64)
    d_hit = np.linalg.norm(loc_all - origins[idx_ray_all], axis=1)

    hit_min = np.full(dirs.shape[0], np.inf, dtype=np.float64)
    np.minimum.at(hit_min, idx_ray_all, d_hit)

    eps = 1e-4
    occluded = (hit_min > eps) & (hit_min < (dists_v - eps))
    return float(np.mean(occluded))


def _projected_bbox_area(
    mesh: trimesh.Trimesh,
    cam_pos: np.ndarray,
) -> float:
    """Approximate projected 2D bbox area of a mesh from a camera position.

    Larger area means the mesh is "expanded" (seen face-on); smaller area
    means it's edge-on (e.g., a thin door seen from the side).
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) < 3:
        return 0.0
    cam = cam_pos.reshape(3)
    forward = -cam / max(float(np.linalg.norm(cam)), 1e-12)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= rn
    up = np.cross(right, forward)
    up /= max(np.linalg.norm(up), 1e-12)

    # Project vertices onto right/up plane
    rel = verts - mesh.centroid[None, :]
    x = rel @ right
    y = rel @ up
    return float((x.max() - x.min()) * (y.max() - y.min()))


def _select_best_azimuths(
    smaller_mesh: trimesh.Trimesh,
    larger_mesh: trimesh.Trimesh,
    *,
    num_views: int = 2,
    n_azimuths: int = 12,
    elevations: Tuple[float, ...] = (20.0, 35.0),
    camera_r: float = 3.0,
    n_samples: int = 128,
) -> List[float]:
    """Select camera azimuths that maximize smaller-link visibility.

    Two-stage scoring:
      1. Occlusion: ray-cast smaller link against larger link. Filter ≤ 0.5.
      2. Expansion: projected 2D bbox area of the smaller link. Keep top 50%.
    Then farthest-point sampling on 3D direction vectors for diversity.

    Falls back to DEFAULT_AZIMUTHS on failure.
    """
    rng = np.random.default_rng(42)
    az_start = 15.0
    az_list = [(az_start + i * 360.0 / n_azimuths) % 360.0 for i in range(n_azimuths)]

    candidates: List[dict] = []
    for el in elevations:
        for az in az_list:
            cam = _spherical_to_cart(camera_r, az, el)
            d = _spherical_to_cart(1.0, az, el)
            d = d / max(float(np.linalg.norm(d)), 1e-12)
            occ = _compute_child_occlusion_ratio(cam, smaller_mesh, larger_mesh, n_samples, rng)
            expansion = _projected_bbox_area(smaller_mesh, cam)
            candidates.append({
                "azimuth": az, "elevation": el,
                "dir": d, "occlusion": occ, "expansion": expansion,
            })

    # Stage 1: filter by occlusion (≤ 0.5)
    kept = [c for c in candidates if c["occlusion"] <= 0.5]
    if not kept:
        kept = sorted(candidates, key=lambda c: c["occlusion"])[:max(num_views * 2, 4)]

    # Stage 2: among remaining, keep top 50% by expansion (face-on views)
    if len(kept) > num_views * 2:
        kept.sort(key=lambda c: c["expansion"], reverse=True)
        kept = kept[: max(len(kept) // 2, num_views * 2)]

    # Farthest-point sampling on 3D direction vectors for diversity
    chosen = _farthest_point_sample(kept, num_views)

    if len(chosen) < num_views:
        return list(DEFAULT_AZIMUTHS[:num_views])

    return [c["azimuth"] for c in chosen[:num_views]]


def _farthest_point_sample(
    candidates: List[dict],
    k: int,
) -> List[dict]:
    """Greedy farthest-point sampling on unit direction vectors.

    Each candidate must have 'dir' (unit 3D vector) and 'occlusion' fields.
    Starts with the lowest-occlusion candidate, then greedily picks the
    candidate maximizing min angular distance to already-chosen views.
    """
    if not candidates or k <= 0:
        return []
    if len(candidates) <= k:
        return list(candidates)

    # Start with lowest occlusion
    occs = [c.get("occlusion", 0.5) for c in candidates]
    chosen_idx = [int(np.argmin(occs))]

    dirs = [np.asarray(c["dir"], dtype=np.float64).reshape(3) for c in candidates]

    while len(chosen_idx) < k:
        best_i: int | None = None
        best_ang = -1.0
        best_occ = 1e9
        for i, d in enumerate(dirs):
            if i in chosen_idx:
                continue
            # Min angular distance to any already-chosen view
            min_ang = min(
                float(np.arccos(np.clip(np.dot(d, dirs[j]), -1.0, 1.0)))
                for j in chosen_idx
            )
            occ_i = occs[i]
            # Maximize angular distance; break ties by lower occlusion
            if (min_ang > best_ang + 1e-9) or (
                abs(min_ang - best_ang) <= 1e-9 and occ_i < best_occ
            ):
                best_ang = min_ang
                best_occ = occ_i
                best_i = i
        if best_i is None:
            break
        chosen_idx.append(best_i)

    return [candidates[i] for i in chosen_idx]


def _look_at_cam_T_world(
    eye: np.ndarray, target: np.ndarray, up_world: np.ndarray,
) -> np.ndarray:
    """Build a 4x4 camera-from-world transform (look-at convention)."""
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up_world)
    n = np.linalg.norm(right)
    if n < 1e-12:
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        n = np.linalg.norm(right)
    right = right / (n + 1e-12)
    up_proj = np.cross(right, forward)
    y_down = -up_proj
    R_cw = np.stack([right, y_down, forward], axis=0)
    t_cw = -R_cw @ eye.reshape(3)
    cam_T = np.eye(4, dtype=np.float64)
    cam_T[:3, :3] = R_cw
    cam_T[:3, 3] = t_cw
    return cam_T


def _project_mesh_bbox_px(
    mesh: trimesh.Trimesh,
    *,
    norm_scale: float,
    norm_translation: Tuple[float, float, float],
    camera_r: float,
    azimuth: float,
    elevation: float,
    target_z: float,
    width: int,
    height: int,
    lens_mm: float = 50.0,
    sensor_width_mm: float = 36.0,
) -> Optional[Tuple[float, float, float, float]]:
    """Project a mesh's vertices to pixel space and return (x0, y0, x1, y1).

    Applies the same normalization and camera model as the Blender templates.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) < 3:
        return None

    # Apply normalization (same as Blender template's apply_static_norm)
    verts = verts * norm_scale + np.array(norm_translation, dtype=np.float64)

    # Camera setup
    sensor_h_mm = sensor_width_mm * (height / width)
    tan_h = (sensor_width_mm * 0.5) / max(lens_mm, 1e-6)
    tan_v = (sensor_h_mm * 0.5) / max(lens_mm, 1e-6)

    eye = np.array(_spherical_to_cart(camera_r, azimuth, elevation), dtype=np.float64)
    target = np.array([0.0, 0.0, target_z], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cam_T = _look_at_cam_T_world(eye, target, up)
    R, t = cam_T[:3, :3], cam_T[:3, 3]

    # Project all vertices
    pts_cam = (R @ verts.T).T + t  # (N, 3)
    valid = pts_cam[:, 2] > 1e-6
    if not np.any(valid):
        return None
    pts_cam = pts_cam[valid]

    u = pts_cam[:, 0] / (pts_cam[:, 2] * tan_h)
    v = pts_cam[:, 1] / (pts_cam[:, 2] * tan_v)
    px = (u + 1.0) * 0.5 * width
    py = (v + 1.0) * 0.5 * height

    x0, x1 = float(np.clip(px.min(), 0, width - 1)), float(np.clip(px.max(), 0, width - 1))
    y0, y1 = float(np.clip(py.min(), 0, height - 1)), float(np.clip(py.max(), 0, height - 1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1, y1)


def _choose_inset_position(
    *,
    width: int,
    height: int,
    inset_w: int,
    inset_h: int,
    child_bbox_px: Optional[Tuple[int, int, int, int]],
    margin_px: int,
) -> Tuple[int, int]:
    """Choose inset placement to minimize overlap with the child region.

    Prefers the quadrant opposite to the child center. Falls back to the
    corner with least overlap if all corners intersect.
    """
    if child_bbox_px is None:
        # Default: bottom-right
        return (width - inset_w - margin_px, height - inset_h - margin_px)

    cx = 0.5 * (child_bbox_px[0] + child_bbox_px[2])
    cy = 0.5 * (child_bbox_px[1] + child_bbox_px[3])
    midx, midy = 0.5 * width, 0.5 * height

    # Preferred: opposite quadrant from child center
    if cx < midx and cy < midy:
        prefer = "br"
    elif cx >= midx and cy < midy:
        prefer = "bl"
    elif cx < midx and cy >= midy:
        prefer = "tr"
    else:
        prefer = "tl"

    def pos(key: str) -> Tuple[int, int]:
        if key == "tl":
            return (margin_px, margin_px)
        if key == "tr":
            return (width - inset_w - margin_px, margin_px)
        if key == "bl":
            return (margin_px, height - inset_h - margin_px)
        return (width - inset_w - margin_px, height - inset_h - margin_px)

    def overlap(x: int, y: int) -> float:
        ix0 = max(float(x), float(child_bbox_px[0]))
        iy0 = max(float(y), float(child_bbox_px[1]))
        ix1 = min(float(x + inset_w), float(child_bbox_px[2]))
        iy1 = min(float(y + inset_h), float(child_bbox_px[3]))
        return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)

    order = [prefer] + [k for k in ("tl", "tr", "bl", "br") if k != prefer]
    best_xy = pos(order[0])
    best_overlap = float("inf")
    for k in order:
        x, y = pos(k)
        ov = overlap(x, y)
        if ov <= 1e-6:
            return (x, y)
        if ov < best_overlap:
            best_overlap = ov
            best_xy = (x, y)
    return best_xy


def _draw_callout_lines(
    draw: ImageDraw.ImageDraw,
    crop_bbox: Tuple[float, float, float, float],
    inset_bbox: Tuple[float, float, float, float],
    stroke: int = 2,
    callout_offset: int = 8,
) -> None:
    """Draw connector lines from crop rectangle to inset, avoiding crossings.

    Uses a 4-point polyline per connector: crop_edge → outward offset →
    inward offset → inset_edge.  Two lines at 25%/75% along the chosen edge.
    Routing is vertical if dy >= dx, horizontal otherwise.
    """
    crop_cx = 0.5 * (crop_bbox[0] + crop_bbox[2])
    crop_cy = 0.5 * (crop_bbox[1] + crop_bbox[3])
    inset_cx = 0.5 * (inset_bbox[0] + inset_bbox[2])
    inset_cy = 0.5 * (inset_bbox[1] + inset_bbox[3])
    dx = inset_cx - crop_cx
    dy = inset_cy - crop_cy

    use_vertical = abs(dy) >= abs(dx)

    def _edge_points(rect, side):
        rx0, ry0, rx1, ry1 = rect
        if side == "top":
            return [(rx0 + 0.25 * (rx1 - rx0), ry0), (rx0 + 0.75 * (rx1 - rx0), ry0)]
        if side == "bottom":
            return [(rx0 + 0.25 * (rx1 - rx0), ry1), (rx0 + 0.75 * (rx1 - rx0), ry1)]
        if side == "left":
            return [(rx0, ry0 + 0.25 * (ry1 - ry0)), (rx0, ry0 + 0.75 * (ry1 - ry0))]
        return [(rx1, ry0 + 0.25 * (ry1 - ry0)), (rx1, ry0 + 0.75 * (ry1 - ry0))]

    def _normal(side):
        return {"top": (0, -1), "bottom": (0, 1), "left": (-1, 0), "right": (1, 0)}[side]

    if use_vertical:
        crop_side = "top" if dy < 0 else "bottom"
        inset_side = "bottom" if dy < 0 else "top"
    else:
        crop_side = "left" if dx < 0 else "right"
        inset_side = "right" if dx < 0 else "left"

    crop_pts = _edge_points(crop_bbox, crop_side)
    inset_pts = _edge_points(inset_bbox, inset_side)
    cn = _normal(crop_side)
    in_ = _normal(inset_side)
    off = float(max(1, callout_offset))

    for p, q in zip(crop_pts, inset_pts):
        p_out = (p[0] + cn[0] * off, p[1] + cn[1] * off)
        q_out = (q[0] + in_[0] * off, q[1] + in_[1] * off)
        path = [p, p_out, q_out, q]
        draw.line(path, fill=(0, 0, 0, 200), width=max(1, stroke), joint="curve")


def _composite_zoom_inset(
    base_img: Image.Image,
    inset_img: Image.Image,
    child_bbox_px: Optional[Tuple[float, float, float, float]] = None,
    inset_frac: float = ZOOM_INSET_SIZE_FRAC,
) -> Image.Image:
    """Composite a zoom inset with crop box and callout lines.

    Args:
        child_bbox_px: Geometrically projected (x0, y0, x1, y1) of the child
            link in pixel space. Used for crop rectangle and adaptive placement.
    """
    w, h = base_img.size
    inset_side = int(h * inset_frac)
    if inset_side < 32:
        return base_img

    inset_resized = inset_img.convert("RGB").resize(
        (inset_side, inset_side), Image.LANCZOS,
    )

    margin = max(int(h * 0.02), 4)
    # Convert float bbox to int for _choose_inset_position
    int_bbox: Optional[Tuple[int, int, int, int]] = None
    if child_bbox_px is not None:
        int_bbox = (
            int(child_bbox_px[0]), int(child_bbox_px[1]),
            int(child_bbox_px[2]), int(child_bbox_px[3]),
        )
    ix, iy = _choose_inset_position(
        width=w, height=h,
        inset_w=inset_side, inset_h=inset_side,
        child_bbox_px=int_bbox,
        margin_px=margin,
    )

    result = base_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    stroke = max(1, int(h * 0.004))
    inset_bbox = (float(ix), float(iy), float(ix + inset_side), float(iy + inset_side))

    # Draw square crop rectangle around child region
    if child_bbox_px is not None:
        # Make square: use the longer side, expand by 80%, then center
        bw = child_bbox_px[2] - child_bbox_px[0]
        bh = child_bbox_px[3] - child_bbox_px[1]
        side = max(max(bw, bh) * 1.8, w * 0.05)
        cx = 0.5 * (child_bbox_px[0] + child_bbox_px[2])
        cy = 0.5 * (child_bbox_px[1] + child_bbox_px[3])
        half = side * 0.5
        crop_rect = (cx - half, cy - half, cx + half, cy + half)
        draw.rectangle(list(crop_rect), outline=(0, 0, 0, 200), width=stroke)

        # Callout lines
        _draw_callout_lines(draw, crop_rect, inset_bbox, stroke=stroke,
                            callout_offset=max(6, int(h * 0.015)))

    result = Image.alpha_composite(result, overlay)

    # Paste inset on top, then draw border over it
    inset_layer = Image.new("RGBA", result.size, (0, 0, 0, 0))
    inset_layer.paste(inset_resized.convert("RGBA"), (ix, iy))
    result = Image.alpha_composite(result, inset_layer)

    # Draw inset border ON TOP of the pasted inset
    border_layer = Image.new("RGBA", result.size, (0, 0, 0, 0))
    border_draw = ImageDraw.Draw(border_layer)
    border_draw.rectangle(list(inset_bbox), outline=(0, 0, 0, 240), width=stroke)
    result = Image.alpha_composite(result, border_layer)

    return result.convert("RGB")


# ── Motion rendering

def render_motion_frames(
    urdf_path: str,
    *,
    azimuths: Tuple[float, float] = DEFAULT_AZIMUTHS,
    percents: Tuple[float, ...] = (0.0, 0.33, 0.67, 1.0),
    labels: Tuple[str, ...] = ("Rest state", "33% open", "67% open", "Fully open"),
    resolution: int = 518,
    elevation: float = 30.0,
    force_render: bool = False,
    blender_bin: Optional[str] = None,
) -> List[str]:
    """Render motion sequence frames (all joints driven simultaneously).

    Each frame = 2 azimuth views concatenated side-by-side + text label.
    Uses the existing blender_render_part_colored.py.template per state.

    Returns list of PNG paths (one per articulation percentage).
    Cached under {urdf_dir}/vlm_motion_frames/.
    """
    from artibench.visualization.blender_driver import (
        create_temp_render_dir,
        resolve_blender_bin,
        run_blender_template,
    )

    urdf_path = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(urdf_path)
    cache_dir = Path(urdf_dir) / MOTION_CACHE_DIR
    obj_name = Path(urdf_dir).name or "object"

    n_frames = len(percents)
    output_paths = [
        str(cache_dir / f"{obj_name}_motion_{i:03d}.png") for i in range(n_frames)
    ]

    # Check cache
    if not force_render:
        meta_path = cache_dir / "render_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("num_frames", 0) >= n_frames and all(
                    os.path.isfile(p) for p in output_paths
                ):
                    return output_paths
            except Exception:
                pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    blender = resolve_blender_bin(blender_bin)
    parsed = parse_urdf(urdf_path, prefer_collision=False)

    # Phase 1: export meshes for all states
    all_state_entries: List[List[dict]] = []
    tmp_dirs: List[str] = []
    try:
        for pct in percents:
            state = _state_from_percent(parsed, pct)
            link_entries, tmp_link_dir = _export_link_objs_at_state(
                urdf_path, state, prefer_collision=False,
            )
            tmp_dirs.append(tmp_link_dir)
            if not link_entries:
                raise RuntimeError(f"No link meshes at state {pct}")
            all_state_entries.append(link_entries)

        # Phase 2: compute global normalization across ALL states so the camera and scale stay consistent across the motion sequence.
        global_min = np.full(3, np.inf, dtype=np.float64)
        global_max = np.full(3, -np.inf, dtype=np.float64)
        for entries in all_state_entries:
            for e in entries:
                m = _load_mesh_from_stl(e["obj_path"])
                if m is not None and hasattr(m, "bounds"):
                    global_min = np.minimum(global_min, m.bounds[0])
                    global_max = np.maximum(global_max, m.bounds[1])

        ranges = global_max - global_min
        max_range = max(float(ranges[0]), float(ranges[1]), float(ranges[2]), 1e-8)
        norm_scale = 1.0 / max_range
        center = 0.5 * (global_min + global_max)
        norm_translation = (-norm_scale * center[0], -norm_scale * center[1], -norm_scale * center[2])

        # Last frame first: its widest pose sets the camera radius that the template then reuses for every other frame.
        render_order = list(range(n_frames))
        if n_frames > 1:
            render_order = [n_frames - 1] + [i for i in range(n_frames - 1)]

        batch_work_dir = create_temp_render_dir("artibench_vlm_motion_batch_")
        try:
            per_frame_solid_paths: Dict[int, List[str]] = {}
            jobs: List[dict] = []
            for frame_idx in render_order:
                frame_solid_paths = [
                    str(batch_work_dir / f"frame{frame_idx}_solid_az{j}.png")
                    for j in range(len(azimuths))
                ]
                frame_trans_paths: List[str] = []
                per_frame_solid_paths[frame_idx] = frame_solid_paths
                jobs.append({
                    "link_meshes": all_state_entries[frame_idx],
                    "solid_paths": frame_solid_paths,
                    "translucent_paths": frame_trans_paths,
                    "per_link_alpha": None,
                    "norm_override": {
                        "scale": float(norm_scale),
                        "translation": list(norm_translation),
                    },
                })

            batch_manifest = {
                "palette": _PART_PALETTE,
                "azimuths": list(azimuths),
                "jobs": jobs,
            }
            manifest_path = batch_work_dir / "batch_manifest.json"
            manifest_path.write_text(
                json.dumps(batch_manifest, ensure_ascii=False), encoding="utf-8",
            )
            state_json_path = batch_work_dir / "batch_state.json"

            replacements = {
                "MANIFEST_PATH": manifest_path.as_posix(),
                "STATE_PATH": state_json_path.as_posix(),
                "FIT_MARGIN": repr(0.92),
                "CAMERA_R": "None",
                "CAMERA_ELEVATION": repr(float(elevation)),
                "LENS_MM": repr(50.0),
                "SENSOR_WIDTH_MM": repr(36.0),
                "TARGET_Z": "0.0",
                "RES_X": str(int(resolution)),
                "RES_Y": str(int(resolution)),
            }

            ok, log_path = run_blender_template(
                template_name="blender_render_part_colored.py.template",
                replacements=replacements,
                blender_bin=blender,
                work_dir=batch_work_dir,
                log_name="blender_motion_batch.log",
                script_name="blender_motion_batch.py",
            )
            if not ok:
                tail = ""
                try:
                    tail = "\n".join(
                        log_path.read_text(errors="ignore").splitlines()[-60:]
                    )
                except Exception:
                    pass
                raise RuntimeError(
                    f"Blender motion batch render failed. Log: {log_path}\n{tail}"
                )

            # Post-process: compose side-by-side views + label per frame.
            for frame_idx in range(n_frames):
                pct = percents[frame_idx]
                label = labels[frame_idx] if frame_idx < len(labels) else f"{int(pct*100)}%"
                frame_img = concat_views(
                    per_frame_solid_paths[frame_idx], label=label,
                )
                frame_img.save(output_paths[frame_idx])

        finally:
            shutil.rmtree(batch_work_dir, ignore_errors=True)

    finally:
        for td in tmp_dirs:
            shutil.rmtree(td, ignore_errors=True)

    # Write cache metadata
    meta = {
        "object_name": obj_name,
        "urdf_path": urdf_path,
        "num_frames": n_frames,
        "percents": list(percents),
        "azimuths": list(azimuths),
        "elevation": float(elevation),
        "resolution": int(resolution),
    }
    (cache_dir / "render_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return output_paths


# ── Joint-highlight rendering

def render_joint_highlight_frames(
    urdf_path: str,
    *,
    num_views: int = 2,
    max_joints: int = 4,
    resolution: int = 518,
    elevation: float = 30.0,
    force_render: bool = False,
    blender_bin: Optional[str] = None,
) -> Tuple[List[str], List[dict]]:
    """Render per-joint highlighted frames (parent=blue, child=red, rest=gray).

    Joints are ranked by importance (bbox volume with sharing discount).
    Up to max_joints are rendered.  For each joint, camera azimuths are
    selected via occlusion scoring to maximize child-link visibility.
    Small child parts automatically get a zoomed inset.

    Each frame = 2 azimuth views concatenated side-by-side.

    Returns:
        (frame_paths, joint_metas) where joint_metas = [
            {"joint_name", "type", "axis", "parent_link", "child_link",
             "importance", "label", "azimuths"}, ...
        ]
    """
    from artibench.visualization.blender_driver import (
        create_temp_render_dir,
        resolve_blender_bin,
        run_blender_template,
    )

    LENS_MM, SENSOR_WIDTH_MM = 50.0, 36.0

    urdf_path = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(urdf_path)
    cache_dir = Path(urdf_dir) / JOINT_HIGHLIGHT_CACHE_DIR
    obj_name = Path(urdf_dir).name or "object"

    # Export link meshes at rest state
    link_entries, tmp_link_dir = _export_link_objs_at_state(urdf_path, {})

    try:
        if not link_entries:
            raise RuntimeError(f"No link meshes in {urdf_path}")

        # Build link_name → (bbox_volume, mesh) maps
        link_volumes: Dict[str, float] = {}
        link_meshes: Dict[str, trimesh.Trimesh] = {}
        for e in link_entries:
            link_volumes[e["link_name"]] = e["bbox_volume"]
            m = _load_mesh_from_stl(e["obj_path"])
            if m is not None:
                link_meshes[e["link_name"]] = m

        # Compute whole-model diagonal for zoom-inset threshold
        all_mesh_list = [m for m in link_meshes.values()]
        if all_mesh_list:
            whole_mesh = trimesh.util.concatenate(all_mesh_list)
            whole_diag = float(np.linalg.norm(whole_mesh.extents))
        else:
            whole_diag = 1.0

        # Parse URDF for joint info and rank
        parsed = parse_urdf(urdf_path, prefer_collision=False)
        ranked = rank_joints_by_importance(parsed.joints, link_volumes)

        if not ranked:
            logger.warning("No actuated joints found in %s", urdf_path)
            return [], []

        # Select top-K joints
        total_actuated = len(ranked)
        selected = ranked[:max_joints]
        n_joints = len(selected)

        # Build type breakdown for all actuated joints (not just shown)
        from collections import Counter
        type_counts = Counter(j.type for j, _ in ranked)
        type_summary = ", ".join(
            f"{cnt} {jtype}" for jtype, cnt in sorted(type_counts.items())
        )

        # Per-joint decisions key off the smaller/larger link rather than parent/child, since the child can be the larger link (a big door on a small hinge).
        per_joint_azimuths: List[List[float]] = []
        per_joint_smaller_name: List[str | None] = []  # for zoom inset focus

        for joint, _imp in selected:
            child_m = link_meshes.get(joint.child_link)
            parent_m = link_meshes.get(joint.parent_link)
            if child_m is None and parent_m is None:
                per_joint_azimuths.append(list(DEFAULT_AZIMUTHS[:num_views]))
                per_joint_smaller_name.append(None)
                continue

            # Determine smaller vs larger link by bbox diagonal
            child_diag = float(np.linalg.norm(child_m.extents)) if child_m is not None else 0.0
            parent_diag = float(np.linalg.norm(parent_m.extents)) if parent_m is not None else 0.0
            if child_diag <= parent_diag:
                smaller_m, larger_m = child_m, parent_m
                smaller_name = joint.child_link
            else:
                smaller_m, larger_m = parent_m, child_m
                smaller_name = joint.parent_link
            per_joint_smaller_name.append(smaller_name)

            # Occlusion + expansion scoring: smaller link against larger link
            if smaller_m is not None and larger_m is not None:
                azimuths = _select_best_azimuths(
                    smaller_m, larger_m,
                    num_views=num_views,
                    camera_r=3.0,
                    elevations=(float(elevation) - 10.0, float(elevation), float(elevation) + 10.0),
                )
            else:
                azimuths = list(DEFAULT_AZIMUTHS[:num_views])
            per_joint_azimuths.append(azimuths)

        # ── Create axis cylinders for each joint ──
        fk_rest = compute_fk(parsed, {})
        axis_stl_paths: List[str | None] = []
        axis_tmp_dir = Path(tempfile.mkdtemp(prefix="artibench_axes_"))
        for i, (joint, _imp) in enumerate(selected):
            child_m = link_meshes.get(joint.child_link)
            sname = per_joint_smaller_name[i]
            sm = link_meshes.get(sname) if sname else None
            sm_diag = float(np.linalg.norm(sm.extents)) if sm is not None else 0.1

            axis_path = create_axis_cylinder(
                joint, fk_rest, child_m,
                whole_mesh=whole_mesh if all_mesh_list else None,
                smaller_diag=sm_diag,
                whole_diag=whole_diag,
                output_path=str(axis_tmp_dir / f"axis_{i:03d}.stl"),
            )
            axis_stl_paths.append(axis_path)

        # Build axis entries for the manifest (separate from link meshes)
        axis_entries = []
        for i, path in enumerate(axis_stl_paths):
            if path is not None:
                axis_entries.append({
                    "obj_path": path,
                    "joint_index": i,
                })

        # Build output paths and metadata
        output_paths = [
            str(cache_dir / f"{obj_name}_joint_{i:03d}.png")
            for i in range(n_joints)
        ]
        joint_metas = []
        for i, (joint, importance) in enumerate(selected):
            label = f"J{i+1}: {joint.type}"
            joint_metas.append({
                "joint_name": joint.name,
                "type": joint.type,
                "axis": joint.axis.tolist(),
                "parent_link": joint.parent_link,
                "child_link": joint.child_link,
                "importance": float(importance),
                "label": label,
                "azimuths": per_joint_azimuths[i],
                "total_actuated": total_actuated,
                "type_summary": type_summary,
            })

        # Check cache
        if not force_render:
            meta_path = cache_dir / "render_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    cached_joints = meta.get("joint_names", [])
                    if (
                        cached_joints == [jm["joint_name"] for jm in joint_metas]
                        and all(os.path.isfile(p) for p in output_paths)
                    ):
                        return output_paths, joint_metas
                except Exception:
                    pass

        cache_dir.mkdir(parents=True, exist_ok=True)
        blender = resolve_blender_bin(blender_bin)

        # Build manifest with per-joint azimuths
        work_dir = create_temp_render_dir("artibench_vlm_joint_hl_")

        joint_frame_entries = []
        per_joint_raw_paths: List[List[str]] = []
        for i, (joint, _importance) in enumerate(selected):
            azs = per_joint_azimuths[i]
            az_paths = [
                str(work_dir / f"joint_{i:03d}_az{j}.png")
                for j in range(len(azs))
            ]
            per_joint_raw_paths.append(az_paths)
            joint_frame_entries.append({
                "parent_link_name": joint.parent_link,
                "child_link_name": joint.child_link,
                "azimuths": azs,
                "output_paths": az_paths,
            })

        # ── Identify which joints need a zoom inset ── Trigger based on the SMALLER link's diagonal (not always the child).
        zoom_inset_paths: dict[int, List[str]] = {}
        joints_needing_zoom: List[int] = []
        for i in range(n_joints):
            sname = per_joint_smaller_name[i]
            if sname and sname in link_meshes:
                sm = link_meshes[sname]
                sm_diag = float(np.linalg.norm(sm.extents))
                if (sm_diag / max(whole_diag, 1e-8)) < ZOOM_INSET_THRESHOLD:
                    joints_needing_zoom.append(i)

        # Build per-zoom-phase entries (one phase per zoom joint,
        # reusing the same Blender process for ALL phases).
        zoom_phases: List[dict] = []
        for i in joints_needing_zoom:
            joint = selected[i][0]
            sname = per_joint_smaller_name[i]
            if not sname or sname not in link_meshes:
                continue

            sm = link_meshes[sname]
            zmin, zmax = sm.bounds
            zranges = zmax - zmin
            zmax_range = max(
                float(zranges[0]), float(zranges[1]), float(zranges[2]), 1e-8,
            )
            zscale = 1.0 / zmax_range
            zcenter = 0.5 * (zmin + zmax)
            zoom_norm = {
                "scale": float(zscale),
                "translation": [float(-zscale * zcenter[k]) for k in range(3)],
            }

            azs = per_joint_azimuths[i]
            zoom_az_paths = [
                str(work_dir / f"zoom_{i:03d}_az{j}.png")
                for j in range(len(azs))
            ]
            zoom_inset_paths[i] = zoom_az_paths

            zoom_phases.append({
                "norm_override": zoom_norm,
                "fit_margin": 0.65,
                "joints": [{
                    "parent_link_name": joint.parent_link,
                    "child_link_name": joint.child_link,
                    "azimuths": azs,
                    "output_paths": zoom_az_paths,
                    # The template hides axes for other joint_idx;
                    # remap so it compares against the ORIGINAL joint index.
                    "axis_joint_index": i,
                }],
                "azimuths": list(DEFAULT_AZIMUTHS[:num_views]),
            })

        # ── Batch: main phase + all zoom phases in one Blender call ──
        phases: List[dict] = [{
            "norm_override": None,
            "fit_margin": 0.92,
            "joints": joint_frame_entries,
            "azimuths": list(DEFAULT_AZIMUTHS[:num_views]),
        }]
        phases.extend(zoom_phases)

        manifest = {
            "link_meshes": link_entries,
            "axis_meshes": axis_entries,
            "parent_color": "#3498db",
            "child_color": "#e74c3c",
            "azimuths": list(DEFAULT_AZIMUTHS[:num_views]),
            "phases": phases,
        }
        manifest_path = work_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8",
        )
        state_path = work_dir / "render_state.json"

        replacements = {
            "MANIFEST_PATH": manifest_path.as_posix(),
            "STATE_PATH": state_path.as_posix(),
            "FIT_MARGIN": repr(0.92),
            "CAMERA_ELEVATION": repr(float(elevation)),
            "LENS_MM": repr(LENS_MM),
            "SENSOR_WIDTH_MM": repr(SENSOR_WIDTH_MM),
            "TARGET_Z": "0.0",
            "RES_X": str(int(resolution)),
            "RES_Y": str(int(resolution)),
        }

        ok, log_path = run_blender_template(
            template_name="blender_render_joint_highlight.py.template",
            replacements=replacements,
            blender_bin=blender,
            work_dir=work_dir,
            log_name="blender_joint_highlight.log",
            script_name="blender_joint_highlight.py",
        )
        if not ok:
            tail = ""
            try:
                tail = "\n".join(
                    log_path.read_text(errors="ignore").splitlines()[-60:]
                )
            except Exception:
                pass
            raise RuntimeError(
                f"Blender joint-highlight render failed. Log: {log_path}\n{tail}"
            )

        # If any zoom's output PNGs were not produced,
        # drop that entry and fall back to the main-view composite (preserves prior behaviour).
        for i in list(zoom_inset_paths):
            if not all(os.path.isfile(p) for p in zoom_inset_paths[i]):
                logger.warning(
                    "Zoom inset render missing for joint %d; using main view.", i,
                )
                zoom_inset_paths.pop(i, None)

        # Read camera_r from render state for geometric projection. Batch mode writes {"states": [...]};
        # we want the MAIN (phase 0) state since zoom inset bbox projection is computed against the main view.
        _render_camera_r: Optional[float] = None
        _render_scale: Optional[float] = None
        _render_translation: Optional[Tuple[float, float, float]] = None
        if state_path.exists():
            try:
                _rs = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(_rs, dict) and "states" in _rs:
                    _rs = _rs["states"][0]
                _render_camera_r = float(_rs["camera_r"])
                _render_scale = float(_rs["scale"])
                _render_translation = tuple(_rs["translation"])
            except Exception:
                pass

        # ── Compose frames: concat views + zoom inset + overlay label ──
        for i in range(n_joints):
            view_imgs: List[Image.Image] = []
            for az_idx in range(len(per_joint_azimuths[i])):
                view_img = Image.open(per_joint_raw_paths[i][az_idx]).convert("RGB")

                # Composite zoom inset if available
                if i in zoom_inset_paths:
                    zoom_path = zoom_inset_paths[i][az_idx]
                    if os.path.isfile(zoom_path):
                        zoom_img = Image.open(zoom_path).convert("RGB")

                        # Compute geometric child bbox for this view
                        child_bbox_px = None
                        sname = per_joint_smaller_name[i]
                        sm = link_meshes.get(sname) if sname else None
                        if (sm is not None and _render_camera_r is not None
                                and _render_scale is not None
                                and _render_translation is not None):
                            child_bbox_px = _project_mesh_bbox_px(
                                sm,
                                norm_scale=_render_scale,
                                norm_translation=_render_translation,
                                camera_r=_render_camera_r,
                                azimuth=per_joint_azimuths[i][az_idx],
                                elevation=float(elevation),
                                target_z=0.0,
                                width=int(resolution),
                                height=int(resolution),
                                lens_mm=LENS_MM,
                                sensor_width_mm=SENSOR_WIDTH_MM,
                            )

                        view_img = _composite_zoom_inset(
                            view_img, zoom_img, child_bbox_px=child_bbox_px,
                        )

                view_imgs.append(view_img)

            # Concatenate views with separator + label
            canvas = _concat_pil_images(view_imgs, joint_metas[i]["label"])
            canvas.save(output_paths[i])

        shutil.rmtree(work_dir, ignore_errors=True)

    finally:
        shutil.rmtree(tmp_link_dir, ignore_errors=True)
        if 'axis_tmp_dir' in dir():
            shutil.rmtree(str(axis_tmp_dir), ignore_errors=True)

    # Write cache metadata
    meta = {
        "object_name": obj_name,
        "urdf_path": urdf_path,
        "num_joints": n_joints,
        "joint_names": [jm["joint_name"] for jm in joint_metas],
        "per_joint_azimuths": per_joint_azimuths,
        "elevation": float(elevation),
        "resolution": int(resolution),
    }
    (cache_dir / "render_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return output_paths, joint_metas


# ── Part-colored rendering

PART_COLORED_CACHE_DIR = "part_colored_rendering_images"


def _classify_internal_external(
    link_entries: List[dict],
    meshes: List[Optional[trimesh.Trimesh]],
    azimuths: Tuple[float, ...] = (120.0, 210.0),
    elevation: float = 30.0,
    camera_r: float = 3.0,
) -> Tuple[List[int], List[int]]:
    """Classify links as external or internal based on camera visibility.

    For each part, sample 256 surface points and ray-cast from 12 camera
    azimuths (every 30°) against ALL other parts merged as one occluder.
    If the best (least-occluded) view has <50% of points visible, the part
    is internal — it cannot be clearly seen from any angle.

    Returns (external_indices, internal_indices).
    """
    n = len(link_entries)
    rng = np.random.default_rng(42)
    visibility_azimuths = [float(a) for a in range(0, 360, 30)]
    is_internal = [False] * n

    for i in range(n):
        mi = meshes[i]
        if mi is None or mi.vertices.shape[0] == 0:
            continue
        other_meshes = [meshes[j] for j in range(n)
                       if j != i and meshes[j] is not None]
        if not other_meshes:
            continue  # no occluders → external
        occluder = trimesh.util.concatenate(other_meshes)

        best_vis = 0.0
        for az in visibility_azimuths:
            cam = _spherical_to_cart(camera_r, az, elevation)
            occ = _compute_child_occlusion_ratio(
                cam, mi, occluder, n_samples=256, rng=rng,
            )
            vis = 1.0 - occ
            if vis > best_vis:
                best_vis = vis
            if best_vis >= 0.50:
                break  # already confirmed external

        if best_vis < 0.50:
            is_internal[i] = True
            logger.debug("Link %s classified as internal (best_vis=%.0f%%)",
                         link_entries[i]["link_name"], best_vis * 100)

    return (
        [i for i in range(n) if not is_internal[i]],
        [i for i in range(n) if is_internal[i]],
    )


def render_part_colored_views(
    urdf_path: str,
    *,
    num_views: int = 2,
    azimuth_start: float = 120.0,
    azimuth_span: float = 90.0,
    elevation: float = 30.0,
    resolution: int = 518,
    force_render: bool = False,
    blender_bin: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Render split external/internal part-colored views via Blender.

    External images: outer parts only (solid | sep | translucent).
    Internal images (if any internal parts exist): inner parts colored on
    a ghost outline of the whole object.

    Returns (external_paths, internal_paths).
    Each list has num_views images.  internal_paths is empty if no internal
    parts were detected.  Results are cached.
    """
    from artibench.visualization.blender_driver import (
        create_temp_render_dir,
        resolve_blender_bin,
        run_blender_template,
    )

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    cache_dir = Path(urdf_dir) / PART_COLORED_CACHE_DIR
    obj_name = Path(urdf_dir).name or "object"

    n = num_views
    if n <= 1:
        azimuths = [float(azimuth_start)]
    else:
        azimuths = np.linspace(
            float(azimuth_start),
            float(azimuth_start) + float(azimuth_span),
            int(n),
            endpoint=True,
        ).tolist()

    ext_output_paths = [str(cache_dir / f"{obj_name}_ext_{i:03d}.png") for i in range(n)]
    int_output_paths = [str(cache_dir / f"{obj_name}_int_{i:03d}.png") for i in range(n)]

    # Check cache
    if not force_render:
        meta_path = cache_dir / "render_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                has_internal = meta.get("has_internal", False)
                cached_ext = all(os.path.isfile(p) for p in ext_output_paths)
                cached_int = (not has_internal) or all(os.path.isfile(p) for p in int_output_paths)
                if meta.get("num_views", 0) >= n and cached_ext and cached_int:
                    logger.info("Using cached part-colored renders from %s", cache_dir)
                    return (ext_output_paths,
                            int_output_paths if has_internal else [])
            except Exception:
                pass

    # Export per-link STL files at rest state
    link_entries, tmp_link_dir = _export_link_objs_at_state(urdf_path, {})
    if not link_entries:
        raise RuntimeError(f"No link meshes found in {urdf_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Classify internal vs external
    meshes = [_load_mesh_from_stl(e["obj_path"]) for e in link_entries]
    ext_idx, int_idx = _classify_internal_external(
        link_entries, meshes, azimuths=tuple(azimuths), elevation=elevation,
    )
    if not ext_idx:
        # Fully nested models classify every link as internal, which would leave `min(ext_volumes)` with an empty sequence downstream.
        logger.warning(
            "All %d links classified as internal; falling back to all-external "
            "(no internal callout pass) for %s.",
            len(link_entries), urdf_path,
        )
        ext_idx = list(range(len(link_entries)))
        int_idx = []
    has_internal = len(int_idx) > 0
    logger.info("Part split: %d external, %d internal", len(ext_idx), len(int_idx))

    # ── External pass: only external links, solid + translucent ──
    ext_entries = [link_entries[i] for i in ext_idx]
    ext_volumes = [ext_entries[j].get("bbox_volume", 0.0) for j in range(len(ext_entries))]
    vol_lo, vol_hi = min(ext_volumes), max(ext_volumes)
    if vol_hi - vol_lo < 1e-12:
        ext_alpha = [0.3] * len(ext_entries)
    else:
        ext_alpha = [
            0.5 - 0.4 * (v - vol_lo) / (vol_hi - vol_lo)
            for v in ext_volumes
        ]
    # ── Batch: external + (optional ghost + color) passes in ONE Blender call ──

    batch_work_dir = create_temp_render_dir("artibench_part_colored_batch_")
    try:
        # ── Job 1: external pass (solid + translucent) ──
        ext_solid = [str(batch_work_dir / f"ext_solid_{i}.png") for i in range(n)]
        ext_trans = [str(batch_work_dir / f"ext_trans_{i}.png") for i in range(n)]
        ext_job = {
            "link_meshes": ext_entries,
            "solid_paths": ext_solid,
            "translucent_paths": ext_trans,
            "per_link_alpha": ext_alpha,
            "palette": _PART_PALETTE,
        }
        jobs: List[dict] = [ext_job]

        ghost_solid: List[str] = []
        ghost_trans: List[str] = []
        int_solid: List[str] = []
        int_trans: List[str] = []

        if has_internal:
            # ── Job 2: ghost pass (all links, uniform gray, very low alpha) ──
            ghost_solid = [str(batch_work_dir / f"ghost_solid_{i}.png") for i in range(n)]
            ghost_trans = [str(batch_work_dir / f"ghost_trans_{i}.png") for i in range(n)]
            ghost_palette = ["#c8c8c8"] * len(link_entries)
            ghost_alpha = [0.08] * len(link_entries)
            jobs.append({
                "link_meshes": link_entries,
                "solid_paths": ghost_solid,
                "translucent_paths": ghost_trans,
                "per_link_alpha": ghost_alpha,
                "palette": ghost_palette,
                # No norm/camera_r override — the template reuses the ones cached from the external (first)
                # job so internal mesh is framed identically to external.
            })

            # ── Job 3: color pass (internal links only, full color) ──
            int_solid = [str(batch_work_dir / f"int_solid_{i}.png") for i in range(n)]
            int_trans = [str(batch_work_dir / f"int_trans_{i}.png") for i in range(n)]
            int_entries_p = [link_entries[i] for i in int_idx]
            jobs.append({
                "link_meshes": int_entries_p,
                "solid_paths": int_solid,
                "translucent_paths": int_trans,
                "per_link_alpha": [1.0] * len(int_entries_p),
                "palette": _PART_PALETTE,
            })

        batch_manifest = {
            "palette": _PART_PALETTE,
            "azimuths": azimuths,
            "jobs": jobs,
        }
        manifest_path = batch_work_dir / "batch_manifest.json"
        manifest_path.write_text(
            json.dumps(batch_manifest, ensure_ascii=False), encoding="utf-8",
        )
        state_json_path = batch_work_dir / "batch_state.json"

        blender = resolve_blender_bin(blender_bin)
        replacements = {
            "MANIFEST_PATH": manifest_path.as_posix(),
            "STATE_PATH": state_json_path.as_posix(),
            "FIT_MARGIN": repr(0.92),
            "CAMERA_R": "None",
            "CAMERA_ELEVATION": repr(float(elevation)),
            "LENS_MM": repr(50.0),
            "SENSOR_WIDTH_MM": repr(36.0),
            "TARGET_Z": "0.0",
            "RES_X": str(int(resolution)),
            "RES_Y": str(int(resolution)),
        }
        ok, log_path = run_blender_template(
            template_name="blender_render_part_colored.py.template",
            replacements=replacements,
            blender_bin=blender,
            work_dir=batch_work_dir,
            log_name="blender_part_colored_batch.log",
            script_name="blender_part_colored_batch.py",
        )
        if not ok:
            tail = ""
            try:
                tail = "\n".join(
                    log_path.read_text(errors="ignore").splitlines()[-60:]
                )
            except Exception:
                pass
            raise RuntimeError(
                f"Blender part-colored batch render failed. Log: {log_path}\n{tail}"
            )

        # ── PIL composition (same as before, just from the batched outputs) ──

        # External: concat solid | translucent
        for i in range(n):
            s = Image.open(ext_solid[i]).convert("RGB")
            t = Image.open(ext_trans[i]).convert("RGB")
            canvas = _concat_pil_images([s, t], label=f"External")
            canvas.save(ext_output_paths[i])

        if has_internal:
            for i in range(n):
                ghost_img = Image.open(ghost_trans[i]).convert("RGBA")
                color_solid = Image.open(int_solid[i]).convert("RGBA")

                # Make white bg transparent in color overlay
                arr = np.array(color_solid)
                white = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
                arr[white, 3] = 0
                color_overlay = Image.fromarray(arr)

                # Left panel: solid colored parts on ghost
                left = Image.alpha_composite(ghost_img, color_overlay).convert("RGB")

                # Right panel: semi-transparent colored parts on ghost
                arr_t = arr.copy()
                visible = arr_t[:, :, 3] > 0
                arr_t[visible, 3] = (arr_t[visible, 3].astype(float) * 0.55).astype(np.uint8)
                color_trans = Image.fromarray(arr_t)
                right = Image.alpha_composite(ghost_img.copy(), color_trans).convert("RGB")

                canvas = _concat_pil_images([left, right], label="Internal")
                canvas.save(int_output_paths[i])
        else:
            int_output_paths = []
    finally:
        shutil.rmtree(batch_work_dir, ignore_errors=True)

    shutil.rmtree(tmp_link_dir, ignore_errors=True)

    # Save metadata
    ext_link_names = [link_entries[i]["link_name"] for i in ext_idx]
    int_link_names = [link_entries[i]["link_name"] for i in int_idx]
    meta = {
        "object_name": obj_name,
        "urdf_path": os.path.abspath(urdf_path),
        "num_views": n,
        "azimuths": azimuths,
        "elevation": float(elevation),
        "resolution": int(resolution),
        "has_internal": has_internal,
        "external_links": ext_link_names,
        "internal_links": int_link_names,
        "palette": _PART_PALETTE,
        "format": "split_external_internal",
    }
    (cache_dir / "render_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    return ext_output_paths, int_output_paths
