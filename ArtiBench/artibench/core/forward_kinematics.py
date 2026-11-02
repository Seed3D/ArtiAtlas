# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
artibench.core.forward_kinematics
=================================
Forward kinematics: given an ArticulatedObject and joint state values,
compute the world-frame SE(3) transform for every link.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .urdf_parser import ArticulatedObject, JointInfo

logger = logging.getLogger(__name__)


# ── Joint-level SE(3)

def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation: 3×3 matrix for rotating `angle` radians about `axis`."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0,        -axis[2],  axis[1]],
        [axis[2],   0,       -axis[0]],
        [-axis[1],  axis[0],  0],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def _joint_transform(joint: JointInfo, value: float) -> np.ndarray:
    """SE(3) transform produced by a joint at a given value.

    For revolute/continuous: rotation about joint axis by `value` radians.
    For prismatic: translation along joint axis by `value` meters.
    For screw: coupled translation (`value` meters) + rotation (proportional
        to the fraction of the prismatic range completed — full range = 360°).
    For fixed: identity.

    The result is composed with the joint origin:
        T_parent_child = joint.origin @ T_motion
    """
    T_motion = np.eye(4)

    if joint.type in ("revolute", "continuous"):
        T_motion[:3, :3] = _rotation_about_axis(joint.axis, value)
    elif joint.type == "prismatic":
        T_motion[:3, 3] = joint.axis * value
    elif joint.type == "screw":
        # Translation along axis
        T_motion[:3, 3] = joint.axis * value
        # Rotation: full range → one 360° turn
        full_range = joint.upper - joint.lower
        if abs(full_range) > 1e-12:
            angle = (value - joint.lower) / full_range * 2.0 * np.pi
        else:
            angle = 0.0
        T_motion[:3, :3] = _rotation_about_axis(joint.axis, angle)
    # fixed: T_motion stays identity

    return joint.origin @ T_motion


# ── Public API

def compute_fk(
    obj: ArticulatedObject,
    joint_states: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """Compute forward kinematics for every link.

    Args:
        obj:          An ArticulatedObject from parse_urdf.
        joint_states: {joint_name: value} for actuated joints.
                      Missing joints default to 0.

    Returns:
        {link_name: world_T_link (4×4 np.ndarray)} for every link.
    """
    # Build lookup structures
    link_names = {link.name for link in obj.links}
    child_to_joint: Dict[str, JointInfo] = {}
    child_to_parent: Dict[str, str] = {}
    # Adjacency list: parent_link -> list of joints whose parent is that link
    parent_to_joints: Dict[str, List[JointInfo]] = {}

    for joint in obj.joints:
        child_to_joint[joint.child_link] = joint
        child_to_parent[joint.child_link] = joint.parent_link
        parent_to_joints.setdefault(joint.parent_link, []).append(joint)

    # Find root links (links that are never a child)
    child_links = set(child_to_parent.keys())
    root_links = link_names - child_links

    # BFS from roots
    world_T: Dict[str, np.ndarray] = {}
    queue: deque[str] = deque()

    for root in root_links:
        world_T[root] = np.eye(4)
        queue.append(root)

    while queue:
        current = queue.popleft()
        # Find all children of current via adjacency list
        for joint in parent_to_joints.get(current, ()):
            if joint.child_link not in world_T:
                value = joint_states.get(joint.name, 0.0)
                T_parent_child = _joint_transform(joint, value)
                world_T[joint.child_link] = world_T[current] @ T_parent_child
                queue.append(joint.child_link)

    return world_T


@dataclass
class KinematicTreeValidation:
    """Result of kinematic tree structural validation."""
    valid: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"valid": self.valid, "warnings": self.warnings}


def validate_kinematic_tree(obj: ArticulatedObject) -> KinematicTreeValidation:
    """Validate kinematic tree structure for a single-object benchmark.

    Checks:
        1. Single connected component (no disconnected sub-trees).
        2. No cycles (every link reachable via BFS from the single root).
        3. No dangling references (joints referencing undefined links).

    Returns:
        KinematicTreeValidation with ``valid=True`` iff the tree passes all
        checks, plus a list of human-readable warning strings.
    """
    result = KinematicTreeValidation()

    link_names = {link.name for link in obj.links}

    # ── Dangling references
    for joint in obj.joints:
        if joint.parent_link and joint.parent_link not in link_names:
            msg = (f"Joint '{joint.name}' references undefined parent link "
                   f"'{joint.parent_link}'")
            result.warnings.append(msg)
            result.valid = False
        if joint.child_link and joint.child_link not in link_names:
            msg = (f"Joint '{joint.name}' references undefined child link "
                   f"'{joint.child_link}'")
            result.warnings.append(msg)
            result.valid = False

    # ── Build adjacency & find roots
    child_links: set[str] = set()
    child_incoming: Dict[str, List[str]] = {}
    parent_to_children: Dict[str, List[str]] = {}
    for joint in obj.joints:
        if joint.child_link in link_names and joint.parent_link in link_names:
            child_links.add(joint.child_link)
            child_incoming.setdefault(joint.child_link, []).append(joint.name)
            parent_to_children.setdefault(joint.parent_link, []).append(
                joint.child_link)

    for child, incoming in sorted(child_incoming.items()):
        if len(incoming) > 1:
            msg = (f"Link '{child}' has multiple parent joints "
                   f"({', '.join(sorted(incoming))}); expected a tree.")
            result.warnings.append(msg)
            result.valid = False

    root_links = link_names - child_links

    # ── Multiple roots = disconnected components
    if len(root_links) > 1:
        msg = (f"Disconnected kinematic tree: {len(root_links)} root links "
               f"found ({', '.join(sorted(root_links))}). "
               f"A single-object URDF must have exactly one root.")
        result.warnings.append(msg)
        result.valid = False
    elif len(root_links) == 0:
        msg = "No root link found — every link is a child of some joint (cycle)"
        result.warnings.append(msg)
        result.valid = False

    # ── BFS reachability → detect cycles / orphaned links
    visited: set[str] = set()
    queue: deque[str] = deque(root_links)
    visited.update(root_links)

    while queue:
        current = queue.popleft()
        for child in parent_to_children.get(current, ()):
            if child not in visited:
                visited.add(child)
                queue.append(child)

    unreachable = link_names - visited
    if unreachable:
        msg = (f"Unreachable links (cycle or missing parent joint): "
               f"{', '.join(sorted(unreachable))}")
        result.warnings.append(msg)
        result.valid = False

    # ── Log warnings
    for w in result.warnings:
        logger.warning("Kinematic tree validation: %s", w)

    return result


def sample_random_states(
    obj: ArticulatedObject,
    n_states: int,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Sample n_states random joint configurations within URDF limits.

    Args:
        obj:      An ArticulatedObject from parse_urdf.
        n_states: Number of configurations to sample.
        seed:     Random seed.

    Returns:
        List of {joint_name: value} dicts, one per state.
    """
    rng = np.random.default_rng(seed)
    actuated = [j for j in obj.joints if j.type != "fixed"]

    states: List[Dict[str, float]] = []
    for _ in range(n_states):
        state: Dict[str, float] = {}
        for joint in actuated:
            lo, hi = joint.lower, joint.upper
            value = rng.uniform(lo, hi)
            state[joint.name] = float(value)
        states.append(state)

    return states
