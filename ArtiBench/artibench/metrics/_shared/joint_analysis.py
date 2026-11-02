# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Joint parameter analysis: screw collapse, axis/origin error helpers."""

from __future__ import annotations

import dataclasses
from typing import Dict, List

import numpy as np

from artibench.core.urdf_parser import ArticulatedObject, JointInfo


def normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > eps else v * 0.0


def axis_angle_error(a1: np.ndarray, a2: np.ndarray) -> float:
    """Undirected angle between two axes in radians [0, pi/2]."""
    u = normalize_vec(a1)
    v = normalize_vec(a2)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.arccos(abs(dot)))


def revolute_axisline_distance(
    pos1: np.ndarray, axis1: np.ndarray,
    pos2: np.ndarray, axis2: np.ndarray,
) -> float:
    """Local perpendicular offset from (pos1, axis1) to (pos2, axis2), using
    axis1 as the reference "along" direction.

    Returns ||(pos2 - pos1) projected onto the plane perpendicular to axis1||.
    For exactly-parallel lines this is the true parallel-line distance. For
    nearly-parallel lines it is numerically stable (unlike the skew-line
    closest-approach formula, which collapses to ~0 as ||axis1 × axis2|| → 0
    even when the two lines are obviously offset in space).

    axis2 is kept in the signature for call-site compatibility but not used;
    the direction mismatch is already captured by ``axis_angle_error``.
    """
    a = normalize_vec(axis1)
    diff = pos2 - pos1
    perp = diff - np.dot(diff, a) * a
    return float(np.linalg.norm(perp))


def canonicalize_joint_axes(obj: ArticulatedObject) -> ArticulatedObject:
    """Pin axis sign so the rest-frame world axis has ``axis[argmax(|axis|)]
    >= 0``; swap (lower, upper) accordingly. Two equivalent encodings of the
    same physical motion collapse onto one representation:

      * ``axis_local = a, range = [l, u]`` and ``axis_local = -a,
        range = [-u, -l]`` (sign-flip in the local frame), and
      * ``(joint.origin = R, axis_local = a)`` vs
        ``(joint.origin = R · Rπ, axis_local = a, range = [-u, -l])`` for any
        Rπ that flips the axis direction (sign-flip routed through
        ``joint.origin``'s rotation).

    Both produce the same effective world-frame axis ``R_parent_world ·
    R_origin · axis_local`` after sign-fix, so paired metrics that
    interpolate states as ``lower + f·(upper-lower)`` (C2 / eval_part_geometry)
    compare matching physical poses regardless of how the URDF chose to
    encode sign.

    Implementation: compute rest-pose FK once to obtain each joint's parent
    world transform; flip the *local* axis + range when the *effective*
    world axis points negative. ``joint.origin`` itself is left intact
    (changing it would also drift the child link's rest-pose mesh).

    Fixed joints and degenerate (zero-norm) axes are left untouched.

    Note: continuous joints have range forced to [0, 2π] by parse_urdf; after
    canonicalize they may end up at [-2π, 0]. By SO(3) periodicity the
    fraction-by-fraction physical poses agree (mod 2π).
    """
    # Lazy import to avoid any chance of an init-time cycle through artibench.metrics → core.forward_kinematics.
    from artibench.core.forward_kinematics import compute_fk

    fk_rest = compute_fk(obj, {})
    new_joints = []
    for j in obj.joints:
        if j.type == "fixed":
            new_joints.append(j)
            continue
        a_local = np.asarray(j.axis, dtype=float)
        if a_local.shape != (3,) or float(np.linalg.norm(a_local)) < 1e-12:
            new_joints.append(j)
            continue
        T_parent = fk_rest.get(j.parent_link, np.eye(4))
        R_world = T_parent[:3, :3] @ j.origin[:3, :3]
        eff_axis = R_world @ a_local
        if float(np.linalg.norm(eff_axis)) < 1e-12:
            new_joints.append(j)
            continue
        k = int(np.argmax(np.abs(eff_axis)))
        if eff_axis[k] < 0:
            new_joints.append(dataclasses.replace(
                j,
                axis=-a_local,
                lower=-float(j.upper),
                upper=-float(j.lower),
            ))
        else:
            new_joints.append(j)
    return dataclasses.replace(obj, joints=new_joints)


def collapse_screw_joints(obj: ArticulatedObject) -> List[JointInfo]:
    """Collapse prismatic+continuous pairs via helper links into screw joints.

    A *screw joint pattern* is detected when an empty link (no mesh geometry)
    has exactly one incoming and one outgoing non-fixed joint whose types are
    {prismatic, continuous} with (anti-)parallel axes.

    Returns:
        List of logical joints -- screw-collapsed plus remaining non-fixed.
    """
    empty_links = {link.name for link in obj.links if not link.mesh_paths}
    non_fixed = [j for j in obj.joints if j.type != "fixed"]

    as_child: Dict[str, List[JointInfo]] = {}
    as_parent: Dict[str, List[JointInfo]] = {}
    for j in non_fixed:
        as_child.setdefault(j.child_link, []).append(j)
        as_parent.setdefault(j.parent_link, []).append(j)

    consumed: set = set()
    screw_joints: List[JointInfo] = []

    for helper in empty_links:
        incoming = [j for j in as_child.get(helper, []) if j.name not in consumed]
        outgoing = [j for j in as_parent.get(helper, []) if j.name not in consumed]
        if len(incoming) != 1 or len(outgoing) != 1:
            continue

        j_in, j_out = incoming[0], outgoing[0]
        types = {j_in.type.lower(), j_out.type.lower()}
        if types != {"prismatic", "continuous"}:
            continue

        # Verify (anti-)parallel axes
        dot = abs(float(np.dot(normalize_vec(j_in.axis), normalize_vec(j_out.axis))))
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
        consumed.add(j_in.name)
        consumed.add(j_out.name)

    remaining = [j for j in non_fixed if j.name not in consumed]
    return screw_joints + remaining
