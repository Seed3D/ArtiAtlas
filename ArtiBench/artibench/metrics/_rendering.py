# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Canonical rendering helper for alignment metrics.

Renders RGB and optional normal-map views of a URDF's assembled mesh
using the Blender pipeline.

Results are cached under {urdf_dir}/canonical_rendering_images/ so that
repeated evaluations skip the rendering step.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from artibench.core.forward_kinematics import compute_fk
from artibench.core.urdf_parser import parse_urdf

logger = logging.getLogger(__name__)

# All PartNet-Mobility URDFs share root fixed-joint rpy=(π/2, 0, -π/2); its inverse recovers the native Y-up, Z-front mesh space that Blender's OBJ importer expects.
_PARTNET_ROOT_RPY = np.array(
    [[0, 0, -1, 0],
     [-1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]],
    dtype=np.float64,
)
_CANONICAL_CORRECTION = np.linalg.inv(_PARTNET_ROOT_RPY)

# Default canonical viewpoints from Trellis 2.0: 4 views, yaw 30-300°, elevation 30°.
DEFAULT_NUM_VIEWS = 4
DEFAULT_AZIMUTH_START = 30.0
DEFAULT_AZIMUTH_SPAN = 270.0
DEFAULT_ELEVATION = 30.0
DEFAULT_RESOLUTION = 518

CANONICAL_DIR_NAME = "canonical_rendering_images"


def _canonical_dir(urdf_dir: str) -> Path:
    return Path(urdf_dir) / CANONICAL_DIR_NAME


def _urdf_cache_signature(urdf_path: str) -> Optional[Dict[str, Any]]:
    """Content fingerprint for cache invalidation.

    Returns ``{"urdf_sha256": str, "urdf_size": int}`` or None if the URDF
    can't be read. We deliberately only fingerprint the URDF file (not every
    referenced mesh) to keep the cache-hit path I/O-cheap; a URDF re-fix (the
    dominant invalidation cause in practice) changes its content hash.

    Content-addressed on purpose: unpacking a downloaded dataset rewrites
    every mtime, and an mtime-keyed signature would invalidate every shipped
    canonical rendering on first use, forcing a full Blender re-render per
    object.
    """
    try:
        st = os.stat(urdf_path)
        with open(urdf_path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None
    return {"urdf_sha256": digest, "urdf_size": int(st.st_size)}


def _signature_matches(sig_now: Dict[str, Any], sig_cached: Dict[str, Any]) -> bool:
    """Compare cache signatures, tolerating the legacy mtime-based form.

    Caches written before the switch to content hashing record
    ``urdf_mtime``. Those cannot be re-verified by hash, so fall back to
    comparing size alone rather than discarding an otherwise valid cache.
    """
    if int(sig_cached.get("urdf_size", -1)) != int(sig_now.get("urdf_size", -2)):
        return False
    if "urdf_sha256" in sig_cached:
        return sig_cached.get("urdf_sha256") == sig_now.get("urdf_sha256")
    return "urdf_mtime" in sig_cached  # legacy entry, size already matched


def get_cached_renderings(
    urdf_dir: str,
    num_views: int = DEFAULT_NUM_VIEWS,
    azimuth_start: float = DEFAULT_AZIMUTH_START,
    azimuth_span: float = DEFAULT_AZIMUTH_SPAN,
    elevation: float = DEFAULT_ELEVATION,
    resolution: int = DEFAULT_RESOLUTION,
    render_normal: bool = True,
    urdf_path: Optional[str] = None,
) -> Optional[Tuple[List[str], List[str]]]:
    """Check if canonical renderings already exist.

    When ``urdf_path`` is provided, also compares the cache's recorded URDF
    mtime/size against the current URDF — invalidates on mismatch so a
    URDF re-fix forces fresh renders instead of silently reusing stale
    images.

    Returns:
        (rgb_paths, normal_paths) if cache is complete, else None.
    """
    canon_dir = _canonical_dir(urdf_dir)
    meta_path = canon_dir / "render_meta.json"
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # URDF identity check — invalidate cache if the source URDF has changed since these renders were produced (content hash or size delta).
    if urdf_path is not None:
        sig_now = _urdf_cache_signature(urdf_path)
        sig_cached = meta.get("urdf_signature")
        if sig_now is not None and sig_cached is not None:
            if not _signature_matches(sig_now, sig_cached):
                return None

    if meta.get("num_views", 0) < num_views:
        return None
    expected_azimuths = (
        [float(azimuth_start)]
        if num_views <= 1
        else np.linspace(
            float(azimuth_start),
            float(azimuth_start) + float(azimuth_span),
            int(num_views),
            endpoint=True,
        ).tolist()
    )
    cached_azimuths = meta.get("azimuths")
    if cached_azimuths is None or len(cached_azimuths) < num_views:
        return None
    if not np.allclose(
        np.asarray(cached_azimuths[:num_views], dtype=np.float64),
        np.asarray(expected_azimuths, dtype=np.float64),
        atol=1e-6,
    ):
        return None
    if abs(float(meta.get("elevation", float("nan"))) - float(elevation)) > 1e-6:
        return None
    if int(meta.get("resolution", -1)) != int(resolution):
        return None
    if bool(meta.get("render_normal", False)) is not bool(render_normal) and render_normal:
        return None

    obj_name = meta.get("object_name", "object")
    rgb_paths = []
    normal_paths = []
    for i in range(num_views):
        rgb = canon_dir / f"{obj_name}_{i:03d}_processed.png"
        if not rgb.exists():
            return None
        rgb_paths.append(str(rgb))

        if render_normal:
            normal = canon_dir / f"{obj_name}_{i:03d}_normal.png"
            if not normal.exists():
                return None
            normal_paths.append(str(normal))

    return rgb_paths, normal_paths


def render_canonical_views(
    urdf_path: str,
    *,
    num_views: int = DEFAULT_NUM_VIEWS,
    azimuth_start: float = DEFAULT_AZIMUTH_START,
    azimuth_span: float = DEFAULT_AZIMUTH_SPAN,
    elevation: float = DEFAULT_ELEVATION,
    resolution: int = DEFAULT_RESOLUTION,
    render_normal: bool = True,
    blender_bin: Optional[str] = None,
    force: bool = False,
) -> Tuple[List[str], List[str]]:
    """Render canonical views of a URDF's visual meshes.

    Uses the canonical-view Blender template. Caches results
    under {urdf_dir}/canonical_rendering_images/.

    Args:
        urdf_path: Path to the URDF file.
        num_views: Number of views to render.
        azimuth_start: Starting azimuth angle (degrees).
        azimuth_span: Total span of azimuth angles.
        elevation: Camera elevation (degrees).
        resolution: Image resolution (square).
        render_normal: Also render normal maps.
        blender_bin: Optional Blender binary path.
        force: Force re-rendering even if cache exists.

    Returns:
        (rgb_paths, normal_paths) — lists of absolute paths to rendered images.
    """
    from artibench.visualization.blender_driver import (
        create_temp_render_dir,
        resolve_blender_bin,
        run_blender_template,
    )
    from artibench.utils.obj_merge_export import export_part_meshes_obj_dir

    urdf_path = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(urdf_path)
    canon_dir = _canonical_dir(urdf_dir)

    # Check cache (URDF identity check guards against stale renders after the URDF has been re-fixed in place).
    if not force:
        cached = get_cached_renderings(
            urdf_dir,
            num_views=num_views,
            azimuth_start=azimuth_start,
            azimuth_span=azimuth_span,
            elevation=elevation,
            resolution=resolution,
            render_normal=render_normal,
            urdf_path=urdf_path,
        )
        if cached is not None:
            return cached

    obj_name = Path(urdf_dir).name or "object"

    obj = parse_urdf(urdf_path, prefer_collision=False)
    fk_zero = compute_fk(obj, {})

    part_dict: dict[int, list[dict]] = defaultdict(list)
    all_instances: list[dict] = []
    for idx, link in enumerate(obj.links):
        if not link.mesh_paths:
            continue
        m = re.search(r"(?:^|_)(\d+)(?:$|_)", link.name)
        part_id = int(m.group(1)) if m else idx
        world_T = fk_zero.get(link.name, np.eye(4, dtype=np.float64))
        for mesh_path, origin in zip(link.mesh_paths, link.origins):
            if not os.path.isfile(mesh_path):
                continue
            total = _CANONICAL_CORRECTION @ world_T @ origin
            inst = {
                "mesh_path": os.path.abspath(mesh_path),
                "transform": total.astype(np.float64).tolist(),
            }
            all_instances.append(inst)
            part_dict[part_id].append(inst)

    if not all_instances:
        raise RuntimeError(f"No visual meshes found in {urdf_path}")

    part_entries = [
        {"part_id": pid, "instances": insts}
        for pid, insts in sorted(part_dict.items())
        if insts
    ]

    # Compute azimuths
    if num_views <= 1:
        azimuths = [float(azimuth_start)]
    else:
        azimuths = np.linspace(
            float(azimuth_start),
            float(azimuth_start) + float(azimuth_span),
            int(num_views),
            endpoint=True,
        ).tolist()

    # Build render manifest
    canon_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = create_temp_render_dir("artibench_canonical_")
    frame_tmp = tmp_dir / "mask_tmp"
    frame_tmp.mkdir(parents=True, exist_ok=True)
    state_path = tmp_dir / "render_state.json"

    # Export properly assembled per-part OBJ files into a temp directory
    tmp_part_dir = tmp_dir / "assembled_parts"
    export_part_meshes_obj_dir(part_entries, tmp_part_dir, verbose=False)

    # Read back the manifest to build part_meshes_info for Blender
    _part_manifest = json.loads(
        (tmp_part_dir / "manifest.json").read_text(encoding="utf-8")
    )
    part_meshes_info = [
        {
            "part_id": int(p["part_id"]),
            "obj_path": (tmp_part_dir / p["obj_path"]).resolve().as_posix(),
        }
        for p in _part_manifest["parts"]
    ]

    frame_entries = []
    for i, azim in enumerate(azimuths):
        rgb_path = canon_dir / f"{obj_name}_{i:03d}_processed.png"
        normal_path = canon_dir / f"{obj_name}_{i:03d}_normal.png"
        frame_entries.append({
            "frame_index": i,
            "azimuth_deg": float(azim),
            "rgb_path": rgb_path.as_posix(),
            "normal_path": normal_path.as_posix() if render_normal else "",
            "mask_npy_path": "",
            "mask_tmp_base": "",
            "mask_exr_output_path": "",
        })

    manifest_path = tmp_dir / "render_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "frames": frame_entries,
            "part_meshes": part_meshes_info,
            "render_masks": False,
            "render_normals": render_normal,
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    blender = resolve_blender_bin(blender_bin)

    camera_r_token = "None"
    replacements = {
        "MANIFEST_PATH": manifest_path.as_posix(),
        "STATE_PATH": state_path.as_posix(),
        "RES_X": str(int(resolution)),
        "RES_Y": str(int(resolution)),
        "FIT_MARGIN": repr(1.0),
        "CAMERA_R": camera_r_token,
        "CAMERA_ELEVATION": repr(float(elevation)),
        "LENS_MM": repr(50.0),
        "SENSOR_WIDTH_MM": repr(36.0),
        "TARGET_Z": "0.0",
    }

    ok, log_path = run_blender_template(
        template_name="blender_render_mesh_custom_infer.py.template",
        replacements=replacements,
        blender_bin=blender,
        work_dir=tmp_dir,
        log_name="blender_canonical.log",
        script_name="blender_canonical.py",
    )

    if not ok:
        tail = ""
        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            tail = "\n".join(lines[-60:])
        except Exception:
            pass
        raise RuntimeError(
            f"Blender canonical rendering failed for {urdf_path}. Log: {log_path}\n{tail}"
        )

    # ``urdf_signature`` invalidates cached renders when the URDF is re-fixed in place; ``urdf_name`` is stored instead of the absolute path so the cache survives relocation.
    meta = {
        "object_name": obj_name,
        "urdf_name": Path(urdf_path).name,
        "urdf_signature": _urdf_cache_signature(urdf_path),
        "num_views": num_views,
        "azimuths": azimuths,
        "elevation": float(elevation),
        "resolution": int(resolution),
        "render_normal": render_normal,
    }
    (canon_dir / "render_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Cleanup temp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Collect output paths
    rgb_paths = [str(canon_dir / f"{obj_name}_{i:03d}_processed.png") for i in range(num_views)]
    normal_paths = []
    if render_normal:
        normal_paths = [str(canon_dir / f"{obj_name}_{i:03d}_normal.png") for i in range(num_views)]

    return rgb_paths, normal_paths
