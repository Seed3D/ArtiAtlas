# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
artibench.metrics.segmentation_quality
======================================
Self-quality metrics for part segmentation (no GT required).

Evaluates how well an articulated object's part decomposition avoids
inter-part overlap, using GPU-accelerated signed distance field (SDF)
analysis via cubvh.

Metrics:
    - **Part Overlap Ratio (rest)**: maximum pairwise penetration ratio
      across all part pairs at rest pose.
    - **Part Overlap Ratio (articulated)**: same analysis per *connected*
      link pair, across deterministic articulation states (lo / mid / hi).

Design note -- articulated overlap:
    When joints are opened, unrelated parts may collide (e.g. a plate inside
    an oven touched by the swinging door). These are **expected** collisions
    and should not penalize the segmentation.  Therefore, articulated overlap
    is evaluated **only between the two links connected by each joint**.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

from artibench.core.urdf_parser import ArticulatedObject
from artibench.core.forward_kinematics import compute_fk
from artibench.metrics._shared.sdf_overlap import (
    _aabb_overlap,
    compute_pair_penetration,
)

logger = logging.getLogger(__name__)


# Helpers

def _geom_has_data(mesh: trimesh.Trimesh) -> bool:
    """Return True if the mesh has meaningful face data."""
    return isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0


def _apply_transform_mesh(mesh: trimesh.Trimesh, T: np.ndarray) -> trimesh.Trimesh:
    """Apply a 4x4 transform to a mesh copy."""
    m = mesh.copy()
    m.apply_transform(T)
    return m


# Cap on cartesian-product joint states per chain: a 3-DOF chain at n_states=5 would give 5^3 = 125 combinations.
_CHAIN_STATE_CAP = 64

# `screw` is the repo-internal 1-DOF pseudo-type produced by `urdf_parser._collapse_screw_joints`; `floating` and `planar` cannot be swept deterministically and their chains are dropped.
_SAMPLEABLE_MOBILE_TYPES = ("revolute", "continuous", "prismatic", "screw")


def _walk_helper_chain(obj: ArticulatedObject, start_joint) -> Optional[Tuple[str, list]]:
    """Walk a joint chain starting from a mesh-bearing parent, through any
    helper links (no mesh) following single-child links, until a mesh-bearing
    descendant is reached.

    Used for PhysX-3D-style URDFs where each multi-DOF articulation is
    decomposed into 1-DOF mobile joints + ``abstract_<id>`` helper links +
    a final fixed offset. The chain reaches the real child; the metric can
    then sweep all mobile joints in the chain together.

    Returns:
        (real_child_link_name, [joints_in_chain]) — chain starts with the
        passed ``start_joint`` and ends with the joint whose child is the
        first mesh-bearing descendant. Returns None when the chain
        branches (helper has multiple outgoing joints) or dead-ends in
        a helper link with no further joints.
    """
    import collections
    mesh_links = {l.name for l in obj.links if l.mesh_paths}
    helper_links = {l.name for l in obj.links if not l.mesh_paths}
    children_of: Dict[str, list] = collections.defaultdict(list)
    for j in obj.joints:
        children_of[j.parent_link].append(j)

    chain = [start_joint]
    cur = start_joint.child_link
    visited = {cur}
    while cur in helper_links:
        outgoing = children_of.get(cur, [])
        if len(outgoing) != 1:
            return None  # branching or dead-end helper
        nxt = outgoing[0]
        if nxt.child_link in visited:
            return None  # cycle (shouldn't happen in valid URDFs)
        visited.add(nxt.child_link)
        chain.append(nxt)
        cur = nxt.child_link
    return (cur, chain) if cur in mesh_links else None


def _get_joint_link_pairs(
    obj: ArticulatedObject,
) -> List[Tuple[str, str, str, list]]:
    """Resolve (chain_id, parent_real, child_real, joints_in_chain) tuples
    for every actuated articulation, walking through helper-link chains so
    PhysX-3D-style decomposed joints (`parent → mobile → abstract → fixed →
    child`, possibly with multiple chained mobile joints for ball / screw /
    compound DOFs) are evaluated as a single logical pair.

    A "chain" always starts with a non-fixed mobile joint whose parent has
    a mesh, and ends at the first mesh-bearing descendant. Fixed joints
    inside the chain (e.g. PhysX-3D's terminal offset transform) are
    preserved so the FK call applies their static offset correctly.

    Joint chains that hit a branching helper or never reach a mesh-bearing
    link are dropped (matches the prior behaviour of skipping unevaluable
    pairs rather than guessing).
    """
    mesh_links = {l.name for l in obj.links if l.mesh_paths}
    pairs: List[Tuple[str, str, str, list]] = []
    for j in obj.joints:
        if j.type not in _SAMPLEABLE_MOBILE_TYPES:
            continue  # skip fixed / floating / planar starts
        if j.parent_link not in mesh_links:
            continue
        result = _walk_helper_chain(obj, j)
        if result is None:
            continue
        child_real, chain = result
        if any(
            jc.type not in _SAMPLEABLE_MOBILE_TYPES and jc.type != "fixed"
            for jc in chain
        ):
            continue
        chain_id = "+".join(jc.name for jc in chain)
        pairs.append((chain_id, j.parent_link, child_real, chain))
    return pairs


def _deterministic_joint_values(joint, n_states: int = 3) -> List[float]:
    """Return n_states uniformly spaced values from lo to hi of joint range."""
    lo, hi = joint.lower, joint.upper
    if joint.type == "continuous":
        if n_states <= 1:
            return [0.0]
        return np.linspace(0.0, 2.0 * np.pi, n_states, endpoint=False).tolist()
    if abs(hi - lo) < 1e-8:
        return [float(lo)]
    return [lo + f * (hi - lo) for f in np.linspace(0, 1, n_states).tolist()]


# Rest-state overlap (all pairs, SDF-based)

def compute_rest_overlap(
    link_meshes: Dict[str, trimesh.Trimesh],
    *,
    num_samples: int = 50000,
    device: str = "cuda:0",
    collect_collision_points: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """All-pairs SDF penetration at rest pose.

    Returns dict with:
        overlap_ratio: max penetration_ratio across all pairs (primary metric)
        mean_pair_ratio: mean across pairs
        num_pairs_evaluated: int
        num_parts: int
        per_pair: list of per-pair detail dicts
        _debug: debug info
    """
    names = [n for n in link_meshes if _geom_has_data(link_meshes[n])]
    if len(names) < 2:
        return {
            "overlap_ratio": 0.0,
            "mean_pair_ratio": 0.0,
            "num_pairs_evaluated": 0,
            "num_parts": len(names),
            "per_pair": [],
            "_debug": {},
        }

    pair_results: List[Dict[str, Any]] = []
    all_ratios: List[float] = []
    pair_failures: List[Dict[str, Any]] = []  # pair-level failures (cubvh / CUDA / mesh)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            ma, mb = link_meshes[na], link_meshes[nb]

            # AABB early-exit
            if not _aabb_overlap(ma, mb, pad=0.02):
                pair_results.append({
                    "pair": (na, nb),
                    "penetration_ratio": 0.0,
                    "skipped": "aabb_disjoint",
                })
                all_ratios.append(0.0)
                continue

            # Pair-level try: a single bad cubvh/CUDA mesh shouldn't drop the whole sample's overlap_ratio.
            # Record the failure so downstream can see the partial-coverage denominator.
            try:
                pen = compute_pair_penetration(
                    ma, mb,
                    num_samples=num_samples,
                    device=device,
                    collect_collision_points=collect_collision_points,
                    seed=int(seed) + len(all_ratios) * 1009,
                )
            except Exception as e:
                pair_results.append({
                    "pair": (na, nb),
                    "failed": True,
                    "error": str(e)[:200],
                })
                pair_failures.append({"pair": (na, nb), "error": str(e)[:200]})
                continue

            pair_results.append({
                "pair": (na, nb),
                **{k: v for k, v in pen.items() if k != "collision_points"},
            })
            if collect_collision_points and "collision_points" in pen:
                pair_results[-1]["collision_points"] = pen["collision_points"]
            all_ratios.append(pen["penetration_ratio"])

    return {
        "overlap_ratio": float(max(all_ratios)) if all_ratios else 0.0,
        "mean_pair_ratio": float(np.mean(all_ratios)) if all_ratios else 0.0,
        "num_pairs_evaluated": len(all_ratios),
        "num_pairs_failed": len(pair_failures),
        "num_parts": len(names),
        "per_pair": pair_results,
        "_debug": {
            "parts_evaluated": names,
            "num_samples": num_samples,
            "failures": pair_failures,
        },
    }


# Articulated overlap (connected link pairs only, SDF-based)

def compute_articulated_overlap(
    obj: ArticulatedObject,
    link_meshes: Dict[str, trimesh.Trimesh],
    *,
    num_samples: int = 50000,
    n_states: int = 3,
    device: str = "cuda:0",
    collect_collision_points: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """Connected-pairs SDF penetration across articulation states.

    Resolves PhysX-3D-style helper-link chains (parent → [mobile joints
    + abstract helpers + terminal fixed offset] → child) into a single
    logical pair via :func:`_get_joint_link_pairs`, then sweeps every
    mobile joint in the chain together (cartesian product of per-joint
    states, capped at :data:`_CHAIN_STATE_CAP` randomly-selected combos
    to keep the evaluation tractable on 3-DOF chains).

    Returns dict with:
        mean_overlap_ratio, max_overlap_ratio, per_joint, num_evaluations
    """
    from itertools import product as _product

    joint_link_pairs = _get_joint_link_pairs(obj)
    if not joint_link_pairs:
        return {
            "mean_overlap_ratio": 0.0,
            "max_overlap_ratio": 0.0,
            "per_joint": [],
            "num_evaluations": 0,
        }

    all_ratios: List[float] = []
    per_joint_results: List[Dict] = []
    state_failures: List[Dict[str, Any]] = []  # state-level failures
    eval_idx = 0
    rng = np.random.default_rng(int(seed))

    for chain_id, parent_link, child_link, chain in joint_link_pairs:
        if parent_link not in link_meshes or child_link not in link_meshes:
            continue
        if not _geom_has_data(link_meshes[parent_link]) or not _geom_has_data(link_meshes[child_link]):
            continue

        # Per-joint state values; fixed joints contribute a single 0.0 (their <origin> still applies via FK,
        # just no DOF to sweep).
        per_joint_states = []
        for jc in chain:
            if jc.type == "fixed":
                per_joint_states.append([0.0])
            else:
                per_joint_states.append(_deterministic_joint_values(jc, n_states=n_states))

        combos = list(_product(*per_joint_states))
        if len(combos) > _CHAIN_STATE_CAP:
            # Always retain the corner configurations (every mobile joint at a bound) so the worst-case interpenetration pose is never sampled away.
            corner_axes = []
            for jc in chain:
                if jc.type == "fixed":
                    corner_axes.append([0.0])
                else:
                    jvals = _deterministic_joint_values(jc, n_states=n_states)
                    corner_axes.append([jvals[0], jvals[-1]] if len(jvals) > 1 else [jvals[0]])
            corner_combos = set(_product(*corner_axes))
            other_combos = [c for c in combos if c not in corner_combos]
            n_extra = max(_CHAIN_STATE_CAP - len(corner_combos), 0)
            if n_extra > 0 and other_combos:
                idxs = rng.choice(len(other_combos), min(n_extra, len(other_combos)), replace=False)
                combos = list(corner_combos) + [other_combos[i] for i in idxs]
            else:
                combos = list(corner_combos)[:_CHAIN_STATE_CAP]

        joint_result: Dict[str, Any] = {
            "joint_name": chain_id,
            "parent_link": parent_link,
            "child_link": child_link,
            "chain_joints": [jc.name for jc in chain],
            "states": [],
        }

        for combo in combos:
            joint_vals = {jc.name: float(v) for jc, v in zip(chain, combo) if jc.type != "fixed"}
            fk = compute_fk(obj, joint_vals)

            T_parent = fk.get(parent_link, np.eye(4))
            T_child = fk.get(child_link, np.eye(4))
            mesh_p = _apply_transform_mesh(link_meshes[parent_link], T_parent)
            mesh_c = _apply_transform_mesh(link_meshes[child_link], T_child)

            # AABB early-exit
            if not _aabb_overlap(mesh_p, mesh_c, pad=0.02):
                all_ratios.append(0.0)
                joint_result["states"].append({
                    "joint_value": joint_vals,
                    "penetration_ratio": 0.0,
                    "_debug": {"note": "aabb_disjoint"},
                })
                continue

            # State-level try: a cubvh/CUDA failure on one joint state shouldn't drop the whole sample's articulated_overlap.
            try:
                pen = compute_pair_penetration(
                    mesh_p, mesh_c,
                    num_samples=num_samples,
                    device=device,
                    collect_collision_points=collect_collision_points,
                    seed=int(seed) + eval_idx * 1009,
                )
            except Exception as e:
                eval_idx += 1
                state_failures.append({
                    "joint_chain": chain_id, "joint_value": joint_vals,
                    "error": str(e)[:200],
                })
                joint_result["states"].append({
                    "joint_value": joint_vals,
                    "failed": True,
                    "error": str(e)[:200],
                })
                continue
            eval_idx += 1

            ratio = pen["penetration_ratio"]
            all_ratios.append(ratio)

            state_info: Dict[str, Any] = {
                "joint_value": joint_vals,
                "penetration_ratio": float(ratio),
                "ratio_ab": pen["ratio_ab"],
                "ratio_ba": pen["ratio_ba"],
                "mean_penetration_depth": pen["mean_penetration_depth"],
                "max_penetration_depth": pen["max_penetration_depth"],
            }
            if collect_collision_points and "collision_points" in pen:
                state_info["collision_points"] = pen["collision_points"]
            joint_result["states"].append(state_info)

        per_joint_results.append(joint_result)

    if not all_ratios:
        return {
            "mean_overlap_ratio": 0.0,
            "max_overlap_ratio": 0.0,
            "per_joint": per_joint_results,
            "num_evaluations": 0,
            "num_evaluations_failed": len(state_failures),
            "_debug": {"failures": state_failures},
        }

    return {
        "mean_overlap_ratio": float(np.mean(all_ratios)),
        "max_overlap_ratio": float(np.max(all_ratios)),
        "per_joint": per_joint_results,
        "num_evaluations": len(all_ratios),
        "num_evaluations_failed": len(state_failures),
        "_debug": {"failures": state_failures},
    }


# Unified interface

def compute_segmentation_quality(
    obj: ArticulatedObject,
    link_meshes: Dict[str, trimesh.Trimesh],
    *,
    num_samples: int = 50000,
    n_states: int = 3,
    device: str = "cuda:0",
    collect_collision_points: bool = False,
    seed: int = 0,
) -> Dict[str, object]:
    """Compute all segmentation quality metrics (rest + articulated overlap).

    Applies rest-pose FK before rest overlap so parts are in assembled world frame.

    Returns:
        Dict with keys:
            rest_overlap: rest-state overlap results
            articulated_overlap: articulated overlap results (connected pairs only)
    """
    # Apply rest-pose FK so parts are in assembled world frame
    rest_fk = compute_fk(obj, {})
    assembled_meshes: Dict[str, trimesh.Trimesh] = {}
    for name, mesh in link_meshes.items():
        T = rest_fk.get(name, np.eye(4))
        assembled_meshes[name] = _apply_transform_mesh(mesh, T)

    rest = compute_rest_overlap(
        assembled_meshes,
        num_samples=num_samples,
        device=device,
        collect_collision_points=collect_collision_points,
        seed=seed,
    )
    articulated = compute_articulated_overlap(
        obj, link_meshes,
        num_samples=num_samples,
        n_states=n_states,
        device=device,
        collect_collision_points=collect_collision_points,
        seed=seed + 1_000_003,
    )

    return {
        "rest_overlap": rest,
        "articulated_overlap": articulated,
    }
