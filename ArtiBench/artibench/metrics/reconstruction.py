# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
artibench.metrics.reconstruction
=================================
Reconstruction metrics for paired pred/GT evaluation.

**C1 — Joint Parameter Evaluation**:
    Screw-joint patterns (prismatic+continuous via empty helper links) are
    collapsed into single logical joints.  Joints matched via Hungarian
    assignment on child-link bbox centers (rest state, world frame) — this
    is robust to URDFs that bake part placement into mesh vertices with a
    zero joint.origin (common in PartNet-Mobility). Measures type accuracy,
    axis angle error, origin position error, limit error, and composite
    success rate.  Extra predicted joints are penalised.

    Type accuracy is asymmetric on continuous ↔ revolute: ``pred=revolute``
    spanning > 330° matched against ``gt=continuous`` counts as type-correct
    (absorbing models that emit wide revolute in lieu of continuous), but
    ``pred=continuous`` matched against any finite-range revolute is always
    a type error (annotation deliberately encodes "this can't go all the way").

    Origin success threshold is adaptive: per joint, threshold = ratio × the
    smaller non-zero world-frame bbox diagonal of the GT joint's parent/child
    link meshes (default ratio 0.03; falls back to an absolute 0.05 m when no
    surrounding geometry is available). The raw origin_error value (axis-line
    distance, m) is unchanged — only success-rate gating scales with part size.

    Limit error dispatches on joint category (length / rotational), where
    ``length`` covers both prismatic and screw (range is in meters in both):
      • cross-category               → constant π penalty (units incompatible).
      • length ↔ length              → endpoint L1 of swept segment projected
        onto gt axis (m). Works for prismatic ↔ prismatic / screw / etc.
      • rotational with ≥1 continuous → |range_pred − range_gt| (continuous start
        position is arbitrary; parser guarantees rotational range ≤ 2π).
      • revolute ↔ revolute          → endpoint L1 (rad) with anti-parallel axis swap.

**C2 — Part Geometry Evaluation**:
    Parts matched via Hungarian assignment on rest-pose bbox centers.
    Per-part Chamfer Distance + bbox mIoU / gIoU at rest and across
    articulation states. Symmetric (pred→gt + gt→pred) / 2.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

from artibench.core.forward_kinematics import compute_fk
from artibench.core.urdf_parser import ArticulatedObject, JointInfo

from artibench.metrics._shared.geometry import (
    bbox_center as _bbox_center,
    bbox_iou as _bbox_iou,
    bbox_giou as _bbox_giou,
    chamfer_distance_np as _chamfer_distance_np,
    safe_mean as _safe_mean,
    average_state_series as _average_state_series,
)
from artibench.metrics._shared.matching import (
    hungarian_match_bbox as _hungarian_match_bbox,
)
from artibench.metrics._shared.assembly import (
    prepare_link_pcls as _prepare_link_pcls,
    assemble_at_state as _assemble_at_state,
    normalize_assembly as _normalize_assembly,
    compute_rest_normalization_bbox as _compute_rest_normalization_bbox,
    compute_rest_normalization_mesh as _compute_rest_normalization_mesh,
    load_cached_link_pcls as _load_cached_link_pcls,
    load_link_meshes_for_whole as _load_link_meshes_for_whole,
    sample_whole_assembled_pcl as _sample_whole_assembled_pcl,
)
from artibench.metrics._shared.joint_analysis import (
    normalize_vec as _normalize,
    axis_angle_error as _axis_angle_error,
    revolute_axisline_distance as _revolute_axisline_distance,
    canonicalize_joint_axes as _canonicalize_joint_axes,
)

logger = logging.getLogger(__name__)


# C1: Joint Parameter Metrics

_CONTINUOUS_EQUIV_RANGE_RAD = 11.0 * math.pi / 6.0  # 330° in radians


def _type_equiv(pj: JointInfo, gj: JointInfo) -> bool:
    """Return True if pred and gt joint types are considered equivalent.

    Equal strings always match. The only cross-string lenience is for the
    asymmetric case ``pred=revolute, gt=continuous`` when the revolute side
    spans more than ``_CONTINUOUS_EQUIV_RANGE_RAD``: many models do not emit
    a ``continuous`` element and instead express full-rotation joints as wide
    revolute. The reverse direction (``pred=continuous, gt=revolute``) is
    NEVER equivalent — annotation deliberately marks finite-range revolute as
    "this can't go all the way" (e.g., a 355° hinge that hits a stop), and a
    model claiming free continuous rotation is a real semantic error.
    """
    pt = pj.type.lower()
    gt = gj.type.lower()
    if pt == gt:
        return True
    if pt == "revolute" and gt == "continuous":
        return (pj.upper - pj.lower) > _CONTINUOUS_EQUIV_RANGE_RAD
    return False


def _category(joint_type: str) -> str:
    """Return the unit category of a joint type for limit_err dispatch.

    - ``length``     (prismatic, screw) → limits in meters. For prismatic
                     it's pure translation; for screw it's the along-axis
                     distance limit (parser keeps the prismatic component's
                     lower/upper). Both can be compared directly in m.
    - ``rotational`` (revolute, continuous) → limits in radians.
    Cross-category matches (length ↔ rotational) use a constant π penalty
    because endpoint differences across categories are unit-incompatible.
    """
    t = joint_type.lower()
    if t in ("prismatic", "screw"):
        return "length"
    if t in ("revolute", "continuous"):
        return "rotational"
    return "other"


def eval_joint_params(
    pred: ArticulatedObject,
    gt: ArticulatedObject,
    *,
    axis_cutoff: float = 0.25,
    origin_cutoff_ratio: float = 0.03,
    origin_cutoff_fallback: float = 0.05,
    pred_norm: Optional[Tuple[np.ndarray, float]] = None,
    gt_norm: Optional[Tuple[np.ndarray, float]] = None,
) -> dict:
    """C1: Evaluate joint parameters with geometric matching.

    Screw joints are already collapsed at parse time by ``parse_urdf``.
    Matching uses the Hungarian algorithm on child-link world-frame
    positions (rest state), not joint names.  Extra predicted joints
    are penalised alongside missing GT joints.

    Args:
        pred:                  Predicted ArticulatedObject.
        gt:                    Ground-truth ArticulatedObject.
        axis_cutoff:           Axis angle error threshold (rad) for success.
        origin_cutoff_ratio:   Per-joint origin threshold = ratio × the (possibly
            normalized) bbox diagonal of the GT joint's parent/child link
            meshes. Scales with the local part size (mirrors
            default 0.03).
        origin_cutoff_fallback: Absolute origin threshold (m, raw GT scale)
            used when the GT bbox diagonal is zero/missing. Divided by
            ``gt_norm[1]`` when normalization is active to stay in the same
            units as the normalized origin distance.
        pred_norm / gt_norm:   Optional ``(center, scale)`` per side (typically
            from ``compute_rest_normalization_mesh``). When provided, every
            world-frame *position* (matching cost, origin distance, threshold
            bbox diag) and every *length-typed joint value* (prismatic/screw
            lower/upper) is divided by the respective side's scale before
            comparison, so the metric is robust to differences in each URDF's
            natural unit scale. Axes (unit directions) and rotational limits
            (radians) are untouched. When both are ``None`` (default) the
            metric runs in raw world-frame, preserving prior behavior.

    Returns:
        Dict with per-joint and aggregate results.
    """
    # Normalization: positions are (p - center) / scale, lengths are v / scale;
    # direction vectors are scale-invariant.
    def _norm_pt(p: np.ndarray, norm: Optional[Tuple[np.ndarray, float]]) -> np.ndarray:
        if norm is None:
            return p
        center, scale = norm
        return (p - center) / float(scale)

    def _norm_len(v: float, norm: Optional[Tuple[np.ndarray, float]]) -> float:
        if norm is None:
            return v
        return float(v) / float(norm[1])

    pred_scale = float(pred_norm[1]) if pred_norm is not None else 1.0
    gt_scale = float(gt_norm[1]) if gt_norm is not None else 1.0
    # Non-fixed joints (screw already collapsed by parse_urdf)
    pred_joints = [j for j in pred.joints if j.type != "fixed"]
    gt_joints = [j for j in gt.joints if j.type != "fixed"]

    n_pred = len(pred_joints)
    n_gt = len(gt_joints)

    # 2. FK at rest for world-frame positions / axes
    pred_fk = compute_fk(pred, {})
    gt_fk = compute_fk(gt, {})

    def _world_axis(joint: JointInfo, fk: Dict[str, np.ndarray]) -> np.ndarray:
        R = fk.get(joint.child_link, np.eye(4))[:3, :3]
        return _normalize(R @ _normalize(joint.axis))

    def _world_pos(joint: JointInfo, fk: Dict[str, np.ndarray]) -> np.ndarray:
        return fk.get(joint.child_link, np.eye(4))[:3, 3].copy()

    # Vertex caches and per-link world-frame vertex collection.
    # Used both for child-link bbox center matching (Hungarian) and for the per-joint adaptive origin threshold.
    _mesh_vert_cache: Dict[str, np.ndarray] = {}

    def _link_world_verts(
        link_name: str, fk: Dict[str, np.ndarray], obj: ArticulatedObject,
    ) -> np.ndarray:
        """All world-frame vertices belonging to a link's visual meshes.
        Returns an (N, 3) array; empty (0, 3) if link missing or has no mesh."""
        link = next((l for l in obj.links if l.name == link_name), None)
        if link is None or not link.mesh_paths:
            return np.zeros((0, 3), dtype=np.float64)
        T_world = fk.get(link_name, np.eye(4))
        verts_world = []
        for mp, origin in zip(link.mesh_paths, link.origins):
            v = _mesh_vert_cache.get(mp)
            if v is None:
                try:
                    m = trimesh.load(mp, force="mesh", process=False)
                    v = np.asarray(m.vertices, dtype=np.float64)
                except Exception:
                    v = np.zeros((0, 3), dtype=np.float64)
                _mesh_vert_cache[mp] = v
            if len(v) == 0:
                continue
            v_h = np.c_[v, np.ones(len(v))]
            v_world = (T_world @ origin @ v_h.T).T[:, :3]
            verts_world.append(v_world)
        if not verts_world:
            return np.zeros((0, 3), dtype=np.float64)
        return np.concatenate(verts_world, 0)

    # Many URDFs bake part placement into mesh vertices with zero joint.origin, collapsing every sibling's FK origin onto the parent's; bbox centers stay distinct.
    def _child_link_bbox_center(
        joint: JointInfo, fk: Dict[str, np.ndarray], obj: ArticulatedObject,
    ) -> np.ndarray:
        all_v = _link_world_verts(joint.child_link, fk, obj)
        if len(all_v) == 0:
            return fk.get(joint.child_link, np.eye(4))[:3, 3].copy()
        return (all_v.min(0) + all_v.max(0)) / 2.0

    def _joint_local_bbox_diag(
        joint: JointInfo, fk: Dict[str, np.ndarray], obj: ArticulatedObject,
    ) -> float:
        """Local-part bbox diagonal in world frame for the joint's two links.

        Computes per-link bbox diagonals for parent and child, returns the
        ``min`` of the two non-zero values. Rationale:

        - For URDFs that bake the entire body into a single root link
          (PartNet-Mobility), the parent (root) is huge while the moving part
          is small. ``min`` falls naturally to the moving-part scale.
        - For balanced URDFs (ArtiAtlas), it picks the smaller of the two
          surrounding parts — the tightest physically meaningful gauge for
          this joint's success threshold.
        - Reference: the motion-axis error convention used by the ArtiAtlas annotation tooling
          uses child-only; we honour the "use both links" intent by also
          considering parent and taking the conservative min.

        Returns 0.0 when neither link has any geometry — callers fall back to
        the absolute origin_cutoff_fallback.
        """
        def _diag(verts: np.ndarray) -> float:
            if len(verts) == 0:
                return 0.0
            return float(np.linalg.norm(verts.max(0) - verts.min(0)))

        parent_diag = _diag(_link_world_verts(joint.parent_link, fk, obj))
        child_diag = _diag(_link_world_verts(joint.child_link, fk, obj))
        diags = [d for d in (parent_diag, child_diag) if d > 0.0]
        if not diags:
            return 0.0
        return float(min(diags))

    # ── 3. Hungarian matching on child-link bbox centers, each side in its own normalized frame so differing URDF unit scales don't bias the cost matrix.
    from scipy.optimize import linear_sum_assignment

    gt_to_pred: Dict[int, int] = {}
    if n_pred > 0 and n_gt > 0:
        INF = 1e9
        cost = np.full((n_pred, n_gt), INF, dtype=np.float64)
        pred_centers = [_norm_pt(_child_link_bbox_center(pj, pred_fk, pred), pred_norm)
                        for pj in pred_joints]
        gt_centers = [_norm_pt(_child_link_bbox_center(gj, gt_fk, gt), gt_norm)
                      for gj in gt_joints]
        for i, p_pos in enumerate(pred_centers):
            for j, g_pos in enumerate(gt_centers):
                cost[i, j] = float(np.linalg.norm(p_pos - g_pos))
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < INF:
                gt_to_pred[c] = r  # gt index → pred index

    # 4. Evaluate
    per_joint: List[dict] = []
    type_ok_list: List[bool] = []
    axis_err_list: List[float] = []
    origin_err_list: List[float] = []
    limit_err_list: List[float] = []
    success_list: List[bool] = []

    WORST = dict(
        type_correct=False, axis_error_rad=math.pi / 2,
        origin_error=1.0, limit_error=math.pi, success=False,
    )

    matched_pred: set = set(gt_to_pred.values())

    def _append_penalty(entry: dict) -> None:
        per_joint.append(entry)
        type_ok_list.append(False)
        axis_err_list.append(WORST["axis_error_rad"])
        origin_err_list.append(WORST["origin_error"])
        limit_err_list.append(WORST["limit_error"])
        success_list.append(False)

    # 4a. GT joints (matched + unmatched)
    for gi, gj in enumerate(gt_joints):
        pi = gt_to_pred.get(gi)
        if pi is None:
            _append_penalty({"gt_joint": gj.name, "pred_joint": None, "matched": False, **WORST})
            continue

        pj = pred_joints[pi]

        type_ok = _type_equiv(pj, gj)
        type_ok_list.append(type_ok)

        a_pred = _world_axis(pj, pred_fk)
        a_gt = _world_axis(gj, gt_fk)
        axis_err = _axis_angle_error(a_pred, a_gt)
        axis_err_list.append(axis_err)

        # Position normalization: each side's origin is divided by its own mesh-derived scale,
        # so origin_err is dimensionless and comparable across methods at different natural URDF scales.
        p_pred = _norm_pt(_world_pos(pj, pred_fk), pred_norm)
        p_gt = _norm_pt(_world_pos(gj, gt_fk), gt_norm)

        # Both the raw-world bbox diagonal and the fallback constant (raw GT meters) are divided by gt_scale so the threshold shares units with the normalized origin error.
        gt_bbox_diag_raw = _joint_local_bbox_diag(gj, gt_fk, gt)
        gt_bbox_diag = gt_bbox_diag_raw / gt_scale
        if gt_bbox_diag > 0.0:
            origin_threshold = origin_cutoff_ratio * gt_bbox_diag
        else:
            origin_threshold = origin_cutoff_fallback / gt_scale

        gt_t = gj.type.lower()
        pj_t = pj.type.lower()
        gt_cat = _category(gt_t)
        pj_cat = _category(pj_t)

        if gt_cat != pj_cat:
            # Endpoints are unit-incompatible across categories (m vs rad), hence the constant π penalty; prismatic gt has a 3-DOF perpendicular gauge, hence no origin_err.
            if gt_t == "prismatic":
                origin_err = None
                origin_ok = True
            else:
                origin_err = _revolute_axisline_distance(p_gt, a_gt, p_pred, a_pred)
                origin_ok = origin_err <= origin_threshold
            limit_err = math.pi
        elif gt_cat == "length":
            # Projecting the swept endpoints onto the gt axis from p_gt is invariant to the (origin + d·a, lower-d, upper-d) gauge, and the sort absorbs axis sign flips.
            lo_pred = _norm_len(pj.lower, pred_norm)
            hi_pred = _norm_len(pj.upper, pred_norm)
            lo_world = p_pred + lo_pred * a_pred
            hi_world = p_pred + hi_pred * a_pred
            t_a = float(np.dot(lo_world - p_gt, a_gt))
            t_b = float(np.dot(hi_world - p_gt, a_gt))
            t_lo, t_hi = (t_a, t_b) if t_a <= t_b else (t_b, t_a)
            gj_lo_n = _norm_len(gj.lower, gt_norm)
            gj_hi_n = _norm_len(gj.upper, gt_norm)
            limit_err = abs(t_lo - gj_lo_n) + abs(t_hi - gj_hi_n)

            # origin_err: prismatic has 3-DOF perpendicular gauge → None.
            # screw has a fixed rotation axis line → axis-line distance is meaningful and the same helper works.
            if gt_t == "prismatic":
                origin_err = None
                origin_ok = True
            else:  # gt_t == "screw"
                origin_err = _revolute_axisline_distance(p_gt, a_gt, p_pred, a_pred)
                origin_ok = origin_err <= origin_threshold
        elif gt_cat == "rotational":
            # Only an along-axis gauge exists here, so the axis line, not the origin point, is the physically meaningful quantity.
            origin_err = _revolute_axisline_distance(p_gt, a_gt, p_pred, a_pred)
            origin_ok = origin_err <= origin_threshold

            if pj_t == "continuous" or gt_t == "continuous":
                # A continuous joint's limits are a parser placeholder ([0, 2π]) with an arbitrary start, so only the swept-range magnitude is comparable.
                range_pred = pj.upper - pj.lower
                range_gt = gj.upper - gj.lower
                limit_err = abs(range_pred - range_gt)
            else:
                # Both revolute (with finite limits). Limits are angles. When axes are anti-parallel,
                # negate & swap pred limits (axis +x [0,1] ≡ axis -x [-1,0]).
                if float(np.dot(a_pred, a_gt)) < 0:
                    pl, pu = -pj.upper, -pj.lower
                else:
                    pl, pu = pj.lower, pj.upper
                limit_err = abs(pl - gj.lower) + abs(pu - gj.upper)
        else:
            origin_err = _revolute_axisline_distance(p_gt, a_gt, p_pred, a_pred)
            origin_ok = origin_err <= origin_threshold
            limit_err = math.pi

        origin_err_list.append(origin_err)
        limit_err_list.append(limit_err)

        axis_ok = axis_err <= axis_cutoff
        success = type_ok and axis_ok and origin_ok
        success_list.append(success)

        per_joint.append({
            "gt_joint": gj.name, "pred_joint": pj.name, "matched": True,
            "type_correct": type_ok,
            "axis_error_rad": round(axis_err, 6),
            "origin_error": None if origin_err is None else round(origin_err, 6),
            "origin_threshold": round(origin_threshold, 6),
            "gt_bbox_diag": round(gt_bbox_diag, 6),
            "limit_error": round(limit_err, 6),
            "success": success,
        })

    # 4b. Extra pred joints (no GT match) → penalty
    for pi, pj in enumerate(pred_joints):
        if pi not in matched_pred:
            _append_penalty({"gt_joint": None, "pred_joint": pj.name, "matched": False, **WORST})

    # 5. Part count accuracy
    n_pred_parts = sum(1 for link in pred.links if link.mesh_paths)
    n_gt_parts = sum(1 for link in gt.links if link.mesh_paths)

    # 6. Aggregate
    valid_origin = [x for x in origin_err_list if x is not None]
    return {
        "n_gt_joints": n_gt,
        "n_pred_joints": n_pred,
        "n_matched": sum(1 for j in per_joint if j["matched"]),
        "count_accuracy": int(n_pred_parts == n_gt_parts),
        "type_accuracy": float(np.mean(type_ok_list)) if type_ok_list else 0.0,
        "axis_error_rad_mean": float(np.mean(axis_err_list)) if axis_err_list else 0.0,
        "origin_error_mean": float(np.mean(valid_origin)) if valid_origin else 0.0,
        "limit_error_mean": float(np.mean(limit_err_list)) if limit_err_list else 0.0,
        "success_rate": float(np.mean(success_list)) if success_list else 0.0,
        "cutoffs": {
            "axis_rad": axis_cutoff,
            "origin_ratio": origin_cutoff_ratio,
            "origin_fallback": origin_cutoff_fallback,
        },
        "per_joint": per_joint,
    }


# C2: Part Geometry Metrics

def _eval_one_direction(
    src_obj: ArticulatedObject,
    tgt_obj: ArticulatedObject,
    src_link_pcls: Dict[str, np.ndarray],
    tgt_link_pcls: Dict[str, np.ndarray],
    n_states: int,
    seed: int,
    src_norm: Optional[Tuple[np.ndarray, float]] = None,
    tgt_norm: Optional[Tuple[np.ndarray, float]] = None,
    src_whole_meshes: Optional[Dict[str, list]] = None,
    tgt_whole_meshes: Optional[Dict[str, list]] = None,
    n_whole_points: int = 32768,
) -> dict:
    """Evaluate src→tgt direction (one side of symmetric evaluation)."""
    # Rest-state assembly
    whole_src_rest, per_link_src_rest = _assemble_at_state(src_obj, src_link_pcls, {})
    whole_tgt_rest, per_link_tgt_rest = _assemble_at_state(tgt_obj, tgt_link_pcls, {})
    if src_norm is not None:
        whole_src_rest, per_link_src_rest = _normalize_assembly(
            whole_src_rest, per_link_src_rest, *src_norm)
    if tgt_norm is not None:
        whole_tgt_rest, per_link_tgt_rest = _normalize_assembly(
            whole_tgt_rest, per_link_tgt_rest, *tgt_norm)

    if len(whole_src_rest) == 0 or len(whole_tgt_rest) == 0:
        return {
            "rest_cd": float("inf"),
            "rest_miou": 0.0,
            "rest_giou": -1.0,
            "articulated_cd": float("inf"),
            "articulated_miou": 0.0,
            "articulated_giou": -1.0,
            "overall_rest_cd": float("inf"),
            "per_state_cd": [],
            "per_state_miou": [],
            "per_state_giou": [],
        }

    # Part matching (Hungarian on rest-pose bbox centers)
    src_names = list(per_link_src_rest.keys())
    tgt_names = list(per_link_tgt_rest.keys())
    src_centers = [_bbox_center(per_link_src_rest[n]) for n in src_names]
    tgt_centers = [_bbox_center(per_link_tgt_rest[n]) for n in tgt_names]
    mapping = _hungarian_match_bbox(src_centers, tgt_centers)

    # Max CD penalty for unmatched parts
    bbox_diag = float(np.linalg.norm(whole_src_rest.max(0) - whole_src_rest.min(0)))
    max_cd = bbox_diag / 2.0

    # Uniform articulation states via linspace interpolation of joint limits
    fractions = np.linspace(0, 1, n_states)

    def _make_uniform_states(obj: ArticulatedObject, fracs: np.ndarray):
        states_list = []
        for f in fracs:
            s = {}
            for j in obj.joints:
                if j.type == "fixed":
                    continue
                s[j.name] = j.lower + f * (j.upper - j.lower)
            states_list.append(s)
        return states_list

    src_states_list = _make_uniform_states(src_obj, fractions)
    tgt_states_list = _make_uniform_states(tgt_obj, fractions)

    per_state_cd = []
    per_state_miou = []
    per_state_giou = []
    overall_rest_cd = None

    all_states = [{}] + src_states_list  # rest + articulated
    all_tgt_states = [{}] + tgt_states_list

    for si, (src_js, tgt_js) in enumerate(zip(all_states, all_tgt_states)):
        whole_src_s, per_link_src = _assemble_at_state(src_obj, src_link_pcls, src_js)
        whole_tgt_s, per_link_tgt = _assemble_at_state(tgt_obj, tgt_link_pcls, tgt_js)
        if src_norm is not None:
            whole_src_s, per_link_src = _normalize_assembly(
                whole_src_s, per_link_src, *src_norm)
        if tgt_norm is not None:
            whole_tgt_s, per_link_tgt = _normalize_assembly(
                whole_tgt_s, per_link_tgt, *tgt_norm)

        # Per-part metrics
        cds = []
        ious = []
        gious = []
        for i, src_ln in enumerate(src_names):
            j_idx = int(mapping[i, 0])
            if j_idx < 0:
                cds.append(max_cd)
                ious.append(0.0)
                gious.append(-1.0)
                continue
            tgt_ln = tgt_names[j_idx]
            src_pcl = per_link_src.get(src_ln)
            tgt_pcl = per_link_tgt.get(tgt_ln)
            if src_pcl is None or tgt_pcl is None or len(src_pcl) < 3 or len(tgt_pcl) < 3:
                cds.append(max_cd)
                ious.append(0.0)
                gious.append(-1.0)
                continue
            cd = _chamfer_distance_np(src_pcl, tgt_pcl)
            iou = _bbox_iou(src_pcl, tgt_pcl)
            giou = _bbox_giou(src_pcl, tgt_pcl)
            cds.append(cd)
            ious.append(iou)
            gious.append(giou)
        per_state_cd.append(float(np.mean(cds)) if cds else max_cd)
        per_state_miou.append(float(np.mean(ious)) if ious else 0.0)
        per_state_giou.append(float(np.mean(gious)) if gious else -1.0)

        # Whole-object CD is rest-state only, and src/tgt share one seed per state so identical inputs give CD = 0.
        if si != 0:
            continue
        if src_whole_meshes is not None and tgt_whole_meshes is not None:
            state_seed = int(seed) + si
            whole_src_overall = _sample_whole_assembled_pcl(
                src_obj, src_whole_meshes, src_js,
                n_total_points=n_whole_points,
                seed=state_seed,
            )
            whole_tgt_overall = _sample_whole_assembled_pcl(
                tgt_obj, tgt_whole_meshes, tgt_js,
                n_total_points=n_whole_points,
                seed=state_seed,
            )
            if src_norm is not None and len(whole_src_overall) > 0:
                whole_src_overall = (whole_src_overall - src_norm[0]) / src_norm[1]
            if tgt_norm is not None and len(whole_tgt_overall) > 0:
                whole_tgt_overall = (whole_tgt_overall - tgt_norm[0]) / tgt_norm[1]
        else:
            whole_src_overall = np.concatenate(
                [per_link_src[n] for n in src_names if per_link_src.get(n) is not None],
                axis=0,
            )
            whole_tgt_overall = np.concatenate(
                [per_link_tgt[n] for n in tgt_names if per_link_tgt.get(n) is not None],
                axis=0,
            )
        if len(whole_src_overall) > 0 and len(whole_tgt_overall) > 0:
            overall_rest_cd = _chamfer_distance_np(whole_src_overall, whole_tgt_overall)
        else:
            overall_rest_cd = max_cd

    return {
        "rest_cd": per_state_cd[0],
        "rest_miou": per_state_miou[0],
        "rest_giou": per_state_giou[0],
        "overall_rest_cd": overall_rest_cd,
        "articulated_cd": float(np.mean(per_state_cd[1:])) if len(per_state_cd) > 1 else per_state_cd[0],
        "articulated_miou": float(np.mean(per_state_miou[1:])) if len(per_state_miou) > 1 else per_state_miou[0],
        "articulated_giou": float(np.mean(per_state_giou[1:])) if len(per_state_giou) > 1 else per_state_giou[0],
        "per_state_cd": per_state_cd,
        "per_state_miou": per_state_miou,
        "per_state_giou": per_state_giou,
    }


def eval_part_geometry(
    pred: ArticulatedObject,
    gt: ArticulatedObject,
    *,
    n_states: int = 5,
    n_link_points: int = 4096,
    n_whole_points: int = 32768,
    seed: int = 42,
    gt_cached_pcls: Optional[Dict[str, np.ndarray]] = None,
    recon_norm: str = "none",
) -> dict:
    """C2: Evaluate part geometry with Hungarian matching.

    Symmetric: computes pred→gt and gt→pred, averages.

    ``recon_norm`` is the per-side normalization mode shared with C1
    (``eval_joint_params``).
    """
    if recon_norm not in ("none", "bbox", "mesh"):
        raise ValueError(f"Unknown recon_norm mode: {recon_norm!r}. "
                         f"Expected 'none', 'bbox', or 'mesh'.")

    # Canonicalize axis sign on both sides so per-fraction state interpolation (`lower + f·(upper-lower)`)
    # compares matching physical poses regardless of the (a,[l,u]) ≡ (-a,[-u,-l]) representation choice.
    pred = _canonicalize_joint_axes(pred)
    gt = _canonicalize_joint_axes(gt)

    pred_link_pcls = _prepare_link_pcls(pred, n_link_points, seed=seed)
    if gt_cached_pcls is not None:
        gt_link_pcls = _load_cached_link_pcls(gt_cached_pcls, n_link_points, seed)
    else:
        gt_link_pcls = _prepare_link_pcls(gt, n_link_points, seed=seed)

    if not pred_link_pcls or not gt_link_pcls:
        logger.warning("Empty link point clouds for C2 evaluation")
        return {
            "status": "invalid",
            "invalid_reason": "empty_link_point_clouds",
            "c2_debug": {"invalid_reason": "empty_link_point_clouds"},
        }

    pred_norm: Optional[Tuple[np.ndarray, float]] = None
    gt_norm: Optional[Tuple[np.ndarray, float]] = None
    if recon_norm == "bbox":
        pred_norm = _compute_rest_normalization_bbox(pred, pred_link_pcls)
        gt_norm = _compute_rest_normalization_bbox(gt, gt_link_pcls)
    elif recon_norm == "mesh":
        pred_norm = _compute_rest_normalization_mesh(pred, seed=seed)
        gt_norm = _compute_rest_normalization_mesh(gt, seed=seed)

    # Load each side's link-mesh cache once for area-weighted whole-mesh CD sampling (decouples whole-CD point budget from per-link counts so over-/under-segmented models all sample the same surface uniformly).
    pred_whole_meshes = _load_link_meshes_for_whole(pred)
    gt_whole_meshes = _load_link_meshes_for_whole(gt)

    fwd = _eval_one_direction(pred, gt, pred_link_pcls, gt_link_pcls, n_states, seed,
                              src_norm=pred_norm, tgt_norm=gt_norm,
                              src_whole_meshes=pred_whole_meshes,
                              tgt_whole_meshes=gt_whole_meshes,
                              n_whole_points=n_whole_points)
    bwd = _eval_one_direction(gt, pred, gt_link_pcls, pred_link_pcls, n_states, seed,
                              src_norm=gt_norm, tgt_norm=pred_norm,
                              src_whole_meshes=gt_whole_meshes,
                              tgt_whole_meshes=pred_whole_meshes,
                              n_whole_points=n_whole_points)

    return {
        "rest_per_part_cd": (fwd["rest_cd"] + bwd["rest_cd"]) / 2.0,
        "rest_per_part_miou": (fwd["rest_miou"] + bwd["rest_miou"]) / 2.0,
        "rest_per_part_giou": (fwd["rest_giou"] + bwd["rest_giou"]) / 2.0,
        "articulated_per_part_cd": (fwd["articulated_cd"] + bwd["articulated_cd"]) / 2.0,
        "articulated_per_part_miou": (fwd["articulated_miou"] + bwd["articulated_miou"]) / 2.0,
        "articulated_per_part_giou": (fwd["articulated_giou"] + bwd["articulated_giou"]) / 2.0,
        "rest_overall_cd": (fwd["overall_rest_cd"] + bwd["overall_rest_cd"]) / 2.0,
        "forward": {
            k: v for k, v in fwd.items() if k not in (
                "per_state_cd",
                "per_state_miou",
                "per_state_giou",
            )
        },
        "backward": {
            k: v for k, v in bwd.items() if k not in (
                "per_state_cd",
                "per_state_miou",
                "per_state_giou",
            )
        },
        "c2_debug": {
            "per_state_cd": _average_state_series(fwd["per_state_cd"], bwd["per_state_cd"]),
            "per_state_miou": _average_state_series(fwd["per_state_miou"], bwd["per_state_miou"]),
            "per_state_giou": _average_state_series(fwd["per_state_giou"], bwd["per_state_giou"]),
        },
    }
