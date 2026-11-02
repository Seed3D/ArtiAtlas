# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
artibench.core.urdf_parser
==========================
Parse a URDF file into a lightweight `ArticulatedObject` dataclass.

Works for PartNet-Mobility URDFs and any standard URDF with mesh geometry.

Post-parse normalization:
  - ``continuous`` joints: limits set to [0, 2π].
  - ``revolute`` joints with range ≥ 360°: reclassified as ``continuous``.
  - **Screw collapse**: a prismatic + continuous pair connected via an empty
    helper link is collapsed into a single ``screw`` joint.  The helper link
    is removed from the link list.
"""

from __future__ import annotations

import hashlib
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Dataclasses

@dataclass
class LinkInfo:
    """A single URDF link with resolved mesh paths."""
    name: str
    mesh_paths: List[str] = field(default_factory=list)   # absolute paths
    origins: List[np.ndarray] = field(default_factory=list)  # per-mesh 4×4 origin transforms


@dataclass
class JointInfo:
    """A single URDF joint with extracted parameters.

    Supported types: revolute, prismatic, continuous, screw, fixed.
    """
    name: str
    type: str
    axis: np.ndarray                # unit direction (3,)
    lower: float = 0.0
    upper: float = 0.0
    origin: np.ndarray = field(default_factory=lambda: np.eye(4))  # 4×4 parent→joint
    parent_link: str = ""
    child_link: str = ""


@dataclass
class ArticulatedObject:
    """Complete kinematic description of an articulated object."""
    links: List[LinkInfo] = field(default_factory=list)
    joints: List[JointInfo] = field(default_factory=list)
    urdf_dir: str = ""
    urdf_path: str = ""


# ── Parsing helpers

def _parse_origin(elem: Optional[ET.Element]) -> np.ndarray:
    """Parse an <origin xyz="..." rpy="..."/> element into a 4×4 matrix."""
    T = np.eye(4)
    if elem is None:
        return T

    xyz_str = elem.get("xyz", "0 0 0")
    rpy_str = elem.get("rpy", "0 0 0")

    xyz = np.array([float(v) for v in xyz_str.split()])
    rpy = np.array([float(v) for v in rpy_str.split()])

    T[:3, 3] = xyz

    # Roll-Pitch-Yaw → rotation matrix (URDF convention: rpy = [roll, pitch, yaw])
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    T[:3, :3] = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])
    return T


def _parse_axis(joint_elem: ET.Element) -> np.ndarray:
    """Parse <axis xyz="..."/> defaulting to [1, 0, 0]."""
    axis_elem = joint_elem.find("axis")
    if axis_elem is None:
        return np.array([1.0, 0.0, 0.0])
    vals = [float(v) for v in axis_elem.get("xyz", "1 0 0").split()]
    a = np.array(vals)
    norm = np.linalg.norm(a)
    return a / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])


def _parse_limits(joint_elem: ET.Element) -> Tuple[float, float]:
    """Parse <limit lower="..." upper="..."/>."""
    limit_elem = joint_elem.find("limit")
    if limit_elem is None:
        return 0.0, 0.0
    lower = float(limit_elem.get("lower", "0"))
    upper = float(limit_elem.get("upper", "0"))
    return lower, upper


def _resolve_mesh_path(urdf_dir: str, filename: str) -> str:
    """Resolve a mesh filename relative to the URDF directory."""
    if os.path.isabs(filename):
        return filename
    # Handle common ROS/Gazebo URI schemes by stripping the prefix
    for prefix in ("package://", "model://"):
        if filename.startswith(prefix):
            filename = filename[len(prefix):]
            break
    return os.path.normpath(os.path.join(urdf_dir, filename))


"""Cache for primitives tessellated into .obj files.

URDF geometry may be a `<box>`, `<cylinder>` or `<sphere>` instead of a
`<mesh>`. The rest of ArtiBench consumes link geometry exclusively as mesh
*file paths*, so rather than threading a second geometry representation
through every downstream consumer (point_cloud, mesh_quality, the
Blender render manifests), primitives are tessellated once into .obj files
here and handed onward as ordinary meshes.

Files land in a shared cache outside the dataset, keyed by a hash of the
primitive's type and parameters, so prediction directories stay byte-identical
to what the model emitted and identical primitives are only built once.
"""
PRIMITIVE_CACHE_DIR = Path(
    os.environ.get(
        "ARTIBENCH_PRIMITIVE_CACHE",
        Path(__file__).resolve().parents[2] / "intermediate" / "primitive_meshes",
    )
)


_PRIMITIVE_WARNED: set = set()


def _primitive_mesh_path(geom_elem: ET.Element) -> Optional[str]:
    """Tessellate a URDF primitive into a cached .obj; None if not a primitive.

    The returned mesh is in the primitive's own local frame, exactly like a
    `<mesh>` file would be, so the caller applies the visual/collision origin
    unchanged. URDF primitive conventions: box is centred on its origin with
    full extents `size`; cylinder is centred and aligned to +Z with `length`
    and `radius`; sphere is centred with `radius`.
    """
    tag = geom_elem.tag
    if tag not in ("box", "cylinder", "sphere"):
        return None
    try:
        if tag == "box":
            size = [float(v) for v in (geom_elem.get("size") or "").split()]
            if len(size) < 3:
                return None
            key = ("box", round(size[0], 9), round(size[1], 9), round(size[2], 9))
        elif tag == "cylinder":
            radius = float(geom_elem.get("radius"))
            length = float(geom_elem.get("length"))
            key = ("cylinder", round(radius, 9), round(length, 9))
        else:
            radius = float(geom_elem.get("radius"))
            key = ("sphere", round(radius, 9))
    except (TypeError, ValueError):
        return None

    digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:16]
    out_path = PRIMITIVE_CACHE_DIR / f"{tag}_{digest}.obj"
    if out_path.is_file():
        return str(out_path)

    try:
        import trimesh

        if tag == "box":
            mesh = trimesh.creation.box(extents=key[1:4])
        elif tag == "cylinder":
            mesh = trimesh.creation.cylinder(radius=key[1], height=key[2], sections=32)
        else:
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=key[1])
        PRIMITIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Write via a unique temp name then rename, so concurrent eval workers never observe a half-written .obj.
        # The temp name must keep the .obj suffix — trimesh picks its exporter from the extension.
        tmp = out_path.with_name(f"{out_path.stem}.tmp-{os.getpid()}.obj")
        mesh.export(tmp)
        os.replace(tmp, out_path)
    except Exception as exc:
        if key not in _PRIMITIVE_WARNED:
            _PRIMITIVE_WARNED.add(key)
            print(f"  [WARN] urdf_parser: could not tessellate {key}: "
                  f"{type(exc).__name__}: {exc}")
        return None
    return str(out_path)


def _collect_mesh_info(
    link_elem: ET.Element,
    urdf_dir: str,
    prefer_collision: bool = True,
) -> Tuple[List[str], List[np.ndarray]]:
    """Collect mesh paths and per-mesh origin transforms from a link element.

    Order controlled by ``prefer_collision``.
    `<box>`/`<cylinder>`/`<sphere>` primitives are tessellated to cached .obj
    files and returned alongside real meshes.
    """
    mesh_paths: List[str] = []
    origins: List[np.ndarray] = []

    # Try collision first, then visual
    geom_sources = ["collision", "visual"] if prefer_collision else ["visual", "collision"]

    for source in geom_sources:
        elems = link_elem.findall(source)
        for elem in elems:
            geom = elem.find("geometry")
            if geom is None:
                continue
            mesh_el = geom.find("mesh")
            if mesh_el is None:
                # Not a <mesh> — may still be a primitive, which carries real geometry (e.g.
                # bicycle frame tubes as <cylinder>). Dropping these silently loses whole parts of an object.
                for child in geom:
                    resolved = _primitive_mesh_path(child)
                    if resolved is None:
                        continue
                    mesh_paths.append(resolved)
                    origins.append(_parse_origin(elem.find("origin")))
                continue
            filename = mesh_el.get("filename", "")
            if not filename:
                continue
            resolved = _resolve_mesh_path(urdf_dir, filename)
            if os.path.isfile(resolved):
                origin = _parse_origin(elem.find("origin"))
                scale_str = mesh_el.get("scale")
                if scale_str:
                    try:
                        vals = [float(v) for v in scale_str.split()]
                        if len(vals) >= 3:
                            scale = np.eye(4)
                            scale[0, 0] = vals[0]
                            scale[1, 1] = vals[1]
                            scale[2, 2] = vals[2]
                            # URDF applies mesh-local scale before the visual / collision origin transform.
                            origin = origin @ scale
                    except ValueError:
                        pass
                mesh_paths.append(resolved)
                origins.append(origin)

        if mesh_paths:
            break  # found meshes in the preferred source

    return mesh_paths, origins


# ── Post-parse normalization

def _normalize_continuous(joints: List[JointInfo]) -> None:
    """Set continuous joint limits to [0, 2π] and reclassify wide revolute."""
    TWO_PI = 2.0 * math.pi
    for j in joints:
        if j.type == "continuous":
            j.lower, j.upper = 0.0, TWO_PI
        elif j.type == "revolute":
            if (j.upper - j.lower) >= TWO_PI - 1e-6:
                j.type = "continuous"
                j.lower, j.upper = 0.0, TWO_PI


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0.0


def _collapse_screw_joints(
    links: List[LinkInfo],
    joints: List[JointInfo],
) -> Tuple[List[LinkInfo], List[JointInfo]]:
    """Collapse prismatic+continuous pairs through empty helper links into screw joints.

    A helper link is an empty link (no meshes) with exactly one incoming and one
    outgoing non-fixed joint whose types are {prismatic, continuous} with
    (anti-)parallel axes.

    The screw joint inherits:
      - axis and limits from the prismatic component
      - origin = incoming.origin @ outgoing.origin
      - parent = real parent, child = real child (helper removed)

    Pitch convention: full prismatic range ↔ one 360° rotation.
    State=0 → no translation, no rotation.

    Returns (filtered_links, normalized_joints).
    """
    empty_links = {l.name for l in links if not l.mesh_paths}
    non_fixed = [j for j in joints if j.type != "fixed"]

    as_child: Dict[str, List[JointInfo]] = {}
    as_parent: Dict[str, List[JointInfo]] = {}
    for j in non_fixed:
        as_child.setdefault(j.child_link, []).append(j)
        as_parent.setdefault(j.parent_link, []).append(j)

    consumed_joints: set = set()
    consumed_links: set = set()
    screw_joints: List[JointInfo] = []

    for helper in empty_links:
        incoming = [j for j in as_child.get(helper, []) if j.name not in consumed_joints]
        outgoing = [j for j in as_parent.get(helper, []) if j.name not in consumed_joints]
        if len(incoming) != 1 or len(outgoing) != 1:
            continue

        j_in, j_out = incoming[0], outgoing[0]
        types = {j_in.type.lower(), j_out.type.lower()}
        if types != {"prismatic", "continuous"}:
            continue

        # Verify (anti-)parallel axes
        dot = abs(float(np.dot(_normalize_vec(j_in.axis), _normalize_vec(j_out.axis))))
        if dot < 0.95:
            continue

        j_pris = j_in if j_in.type.lower() == "prismatic" else j_out
        screw_joints.append(JointInfo(
            name=f"screw({j_in.name}+{j_out.name})",
            type="screw",
            axis=j_pris.axis.copy(),
            lower=j_pris.lower,
            upper=j_pris.upper,
            origin=j_in.origin @ j_out.origin,
            parent_link=j_in.parent_link,
            child_link=j_out.child_link,
        ))
        consumed_joints.add(j_in.name)
        consumed_joints.add(j_out.name)
        consumed_links.add(helper)

    if not consumed_joints:
        return links, joints

    # Rebuild joint list: screw replacements + all non-consumed joints
    remaining_joints = [j for j in joints if j.name not in consumed_joints]
    new_joints = remaining_joints + screw_joints

    # Remove helper links
    new_links = [l for l in links if l.name not in consumed_links]

    # Remove fixed joints that referenced consumed helper links
    new_joints = [
        j for j in new_joints
        if j.parent_link not in consumed_links and j.child_link not in consumed_links
    ]

    return new_links, new_joints


# ── Main API

def parse_urdf(urdf_path: str, *, prefer_collision: bool = True) -> ArticulatedObject:
    """Parse a URDF file into an ArticulatedObject.

    Post-parse normalization is applied automatically:
      1. ``continuous`` joints get limits [0, 2π].
      2. ``revolute`` joints with range ≥ 360° are reclassified as ``continuous``.
      3. Screw patterns (prismatic+continuous via empty helper link) are collapsed
         into single ``screw`` joints; helper links are removed.

    Args:
        urdf_path: Path to the .urdf file.
        prefer_collision: If True, parse collision meshes first and fall back to
            visual meshes. If False, parse visual meshes first and fall back to
            collision meshes.

    Returns:
        ArticulatedObject with links (mesh paths resolved) and joints.
    """
    urdf_path = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(urdf_path)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # ── Parse links
    links: List[LinkInfo] = []
    for link_elem in root.findall("link"):
        name = link_elem.get("name", "")
        mesh_paths, origins = _collect_mesh_info(
            link_elem,
            urdf_dir,
            prefer_collision=prefer_collision,
        )
        links.append(LinkInfo(name=name, mesh_paths=mesh_paths, origins=origins))

    # ── Parse joints
    joints: List[JointInfo] = []
    for joint_elem in root.findall("joint"):
        jname = joint_elem.get("name", "")
        jtype = joint_elem.get("type", "fixed")
        axis = _parse_axis(joint_elem)
        lower, upper = _parse_limits(joint_elem)
        origin = _parse_origin(joint_elem.find("origin"))

        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        parent_link = parent_elem.get("link", "") if parent_elem is not None else ""
        child_link = child_elem.get("link", "") if child_elem is not None else ""

        joints.append(JointInfo(
            name=jname,
            type=jtype,
            axis=axis,
            lower=lower,
            upper=upper,
            origin=origin,
            parent_link=parent_link,
            child_link=child_link,
        ))

    # ── Normalize
    _normalize_continuous(joints)
    links, joints = _collapse_screw_joints(links, joints)

    return ArticulatedObject(
        links=links,
        joints=joints,
        urdf_dir=urdf_dir,
        urdf_path=urdf_path,
    )
