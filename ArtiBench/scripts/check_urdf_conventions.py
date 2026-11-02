#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Check and optionally fix URDF convention violations for ArtiBench submissions.

Convention: each submitted URDF must have exactly one fixed joint, connecting
``base`` to the root link.  All other joints must be non-fixed (revolute,
prismatic, continuous), and the kinematic graph must be a tree.

Root-joint origin: submissions that already follow the convention often carry
the global-pose-adjustment ``rpy="1.570796326794897 0 -1.570796326794897"``.
Those are left exactly as they are.  When this script has to *insert* a
missing root fixed joint it writes ``rpy="0 0 0"`` instead — a synthetic base
joint exists to satisfy the base→root convention, not to re-orient the model,
and writing the 90°/-90° rotation would tip over predictions that were already
correctly oriented.

Three classes of violations are detected:

  1. **Missing or malformed root fixed joint** — the root link is not connected
     to ``base`` via a fixed joint (or ``base`` link is missing entirely).
     Includes the multi-root case (multiple ``base→X`` fixed joints): one is
     picked as the canonical root by descendant count, the rest are
     reported as extras.
  2. **Extra fixed joints** — fixed joints other than the single canonical
     ``base→root_link`` one (e.g. a handle attached to a door via a fixed
     joint, or a leftover ``base→knob0`` from a multi-root output).
  3. **Not a tree** — some link has two or more parent joints.  Reported but
     never "repaired": it means broken model output, and it is also the
     tripwire that stops a fix from being reported clean if it breaks the
     tree itself.

With ``--enable-urdf-fixing``, the script repairs 1 and 2:

  - Issue 1: inserts ``base`` link + canonical fixed joint (identity rpy)
    when missing.  A link already *named* ``base`` is only accepted as the
    placeholder when it is genuinely synthetic — no visual/collision
    geometry and no non-fixed child joints.  Otherwise it is the model's
    real root, so it is renamed (``base`` → ``base_root``, all joint
    endpoints repointed) and a fresh synthetic ``base`` is inserted above
    it.  Without that check the fixer would hunt for the root among the
    real root's own children and hand one of them a second parent.
  - Issue 2: iteratively collapses every extra fixed joint, leaf or
    non-leaf:
        * merges child link's visual/collision into the parent (with the
          fixed transform composed into each geom origin)
        * reparents every joint whose parent==child_link to point at
          parent_link, multiplying the joint's origin by the fixed
          transform so absolute pose is preserved
        * removes the child link and the fixed joint
    Multi-root extras are first re-routed (parent rewritten from ``base``
    to the canonical root link, origin re-expressed in the new frame),
    then collapsed by the same loop.

  Out of scope (the script intentionally does NOT touch these — they
  indicate broken model output that should surface via the
  ``kinematic_tree_valid_rate`` metric, not be silently fixed):
    * self-loop fixed joints (parent == child)
    * cyclic kinematic graphs (a sub-joint whose child is the link the
      collapse would remove — e.g. ``body <-> base``)

  ``[FIXED]`` is printed only when the post-fix audit returns clean;
  ``[PARTIAL]`` indicates remaining violations the fixer couldn't
  safely resolve.

Usage:

    # Audit only (dry-run)
    python scripts/check_urdf_conventions.py \\
        --root /path/to/predictions

    # Audit + fix
    python scripts/check_urdf_conventions.py \\
        --root /path/to/predictions \\
        --enable-urdf-fixing
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET

# Identity on purpose: an inserted base joint satisfies the base→root convention without re-orienting the model.
ROOT_JOINT_RPY = "0 0 0"
ROOT_JOINT_XYZ = "0 0 0"
BASE_LINK_NAMES = ("base", "base_link")


# Transform helpers

def _origin_to_transform(origin_elem) -> "np.ndarray":
    import numpy as np

    transform = np.eye(4, dtype=np.float64)
    if origin_elem is None:
        return transform

    xyz = [float(v) for v in origin_elem.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in origin_elem.get("rpy", "0 0 0").split()]
    transform[:3, 3] = xyz

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    transform[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return transform


def _clean_small(value: float, eps: float = 1e-12) -> float:
    return 0.0 if abs(value) < eps else float(value)


def _transform_to_xyz_rpy(transform) -> Tuple[List[float], List[float]]:
    R = transform[:3, :3]
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    xyz = [_clean_small(v) for v in transform[:3, 3]]
    rpy = [_clean_small(roll), _clean_small(pitch), _clean_small(yaw)]
    return xyz, rpy


def _fmt_vec(v) -> str:
    return f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}"


def _set_origin_from_transform(elem, transform) -> None:
    xyz, rpy = _transform_to_xyz_rpy(transform)
    origin = elem.find("origin")
    if origin is None:
        origin = ET.Element("origin")
        elem.insert(0, origin)
    origin.set("xyz", _fmt_vec(xyz))
    origin.set("rpy", _fmt_vec(rpy))


def _to_pretty_xml(root) -> str:
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(
        root, encoding="unicode"
    ) + "\n"


# Audit

def audit_urdf(urdf_path: str) -> dict:
    """Audit a single URDF for convention violations.

    Returns dict with keys:
        root_fixed_joint_ok: bool
        root_fixed_joint_detail: str
        extra_fixed_joints: list of (joint_name, parent_link, child_link)
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {link.get("name") for link in root.findall("link") if link.get("name")}
    joints = root.findall("joint")

    # Find the base link
    base_link = None
    for name in BASE_LINK_NAMES:
        if name in links:
            base_link = name
            break

    # Find the fixed joint from base
    root_fixed_joint = None
    extra_fixed: list = []

    for joint in joints:
        if joint.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link", "")
        child_name = child.get("link", "")

        if base_link and parent_name == base_link:
            if root_fixed_joint is None:
                root_fixed_joint = joint
            else:
                extra_fixed.append((
                    joint.get("name", "?"),
                    parent_name,
                    child_name,
                ))
        else:
            extra_fixed.append((
                joint.get("name", "?"),
                parent_name,
                child_name,
            ))

    # Determine root fixed joint status
    if base_link is None:
        root_ok = False
        root_detail = "no 'base' or 'base_link' link found"
    elif root_fixed_joint is None:
        root_ok = False
        root_detail = f"'{base_link}' link exists but no fixed joint from it to any child"
    else:
        root_ok = True
        child_name = root_fixed_joint.find("child").get("link", "?")
        root_detail = f"fixed joint from '{base_link}' to '{child_name}'"

    # A link with two parent joints is a graph, not a tree, and is rejected downstream by kinematic_tree_valid_rate.
    parent_counts: dict = {}
    for joint in joints:
        child = joint.find("child")
        if child is None:
            continue
        name = child.get("link", "")
        parent_counts.setdefault(name, []).append(joint.get("name", "?"))
    multi_parent = {k: v for k, v in parent_counts.items() if len(v) > 1}

    return {
        "root_fixed_joint_ok": root_ok,
        "root_fixed_joint_detail": root_detail,
        "extra_fixed_joints": extra_fixed,
        "multi_parent_links": multi_parent,
    }


# Fix

def _is_synthetic_base(root_elem, name: str) -> bool:
    """Whether the link ``name`` is the convention's placeholder base link.

    A link merely *called* ``base`` is not necessarily the synthetic base the
    convention asks for: plenty of predictions name their real, geometry-
    bearing root link ``base`` and hang revolute joints straight off it.
    Treating such a link as the placeholder makes the fixer look for the
    "root" among its own children and then attach a second, fixed parent to
    one of them — turning a valid tree into a graph with a two-parent link.

    A genuine placeholder carries no geometry and articulates nothing: every
    joint leaving it is fixed.
    """
    link = next(
        (l for l in root_elem.findall("link") if l.get("name") == name), None
    )
    if link is None:
        return False
    if link.findall("visual") or link.findall("collision"):
        return False
    for joint in root_elem.findall("joint"):
        parent = joint.find("parent")
        if parent is None or parent.get("link", "") != name:
            continue
        if joint.get("type") != "fixed":
            return False
    return True


def _free_link_name(root_elem, preferred: str) -> str:
    """Return ``preferred``, suffixed if needed, so it collides with no link."""
    taken = {l.get("name") for l in root_elem.findall("link")}
    if preferred not in taken:
        return preferred
    i = 2
    while f"{preferred}_{i}" in taken:
        i += 1
    return f"{preferred}_{i}"


def _rename_link(root_elem, old: str, new: str) -> None:
    """Rename a link and repoint every joint endpoint that referenced it."""
    for link in root_elem.findall("link"):
        if link.get("name") == old:
            link.set("name", new)
    for joint in root_elem.findall("joint"):
        for endpoint in ("parent", "child"):
            elem = joint.find(endpoint)
            if elem is not None and elem.get("link", "") == old:
                elem.set("link", new)


def _find_kinematic_root(root_elem) -> str | None:
    """Find the root link of the kinematic tree (excluding base)."""
    children = set()
    for joint in root_elem.findall("joint"):
        child = joint.find("child")
        if child is not None:
            children.add(child.get("link", ""))

    parents = set()
    for joint in root_elem.findall("joint"):
        parent = joint.find("parent")
        if parent is not None:
            parents.add(parent.get("link", ""))

    all_links = {link.get("name") for link in root_elem.findall("link") if link.get("name")}
    non_base = all_links - set(BASE_LINK_NAMES)

    non_base_children = set()
    for joint in root_elem.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        p = parent.get("link", "")
        c = child.get("link", "")
        if p not in BASE_LINK_NAMES:
            non_base_children.add(c)

    roots = non_base - non_base_children
    if len(roots) == 1:
        return roots.pop()
    # Fallback: link that is child of base fixed joint
    for joint in root_elem.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        if parent is not None and parent.get("link", "") in BASE_LINK_NAMES:
            child = joint.find("child")
            if child is not None:
                return child.get("link", "")
    # Deterministic fallback: pick lexicographically smallest root
    return min(sorted(roots)) if roots else None


def fix_root_fixed_joint(root_elem) -> bool:
    """Ensure a synthetic base link + its single fixed joint exist.

    Returns True if the URDF was modified.
    """
    links = {link.get("name") for link in root_elem.findall("link") if link.get("name")}

    # Check if base already exists
    base_name = None
    for name in BASE_LINK_NAMES:
        if name in links:
            base_name = name
            break

    modified = False

    # A base-named link carrying geometry or articulated children is the real root, so rename it away before a synthetic base is inserted above it.
    if base_name is not None and not _is_synthetic_base(root_elem, base_name):
        _rename_link(root_elem, base_name, _free_link_name(root_elem, f"{base_name}_root"))
        base_name = None
        modified = True

    # Check if root fixed joint already exists
    if base_name is not None:
        for joint in root_elem.findall("joint"):
            if joint.get("type") != "fixed":
                continue
            parent = joint.find("parent")
            if parent is not None and parent.get("link", "") in BASE_LINK_NAMES:
                return modified  # already has root fixed joint

    # Find the kinematic root
    kin_root = _find_kinematic_root(root_elem)
    if kin_root is None:
        return modified

    # Add base link if missing
    if base_name is None:
        base_name = "base"
        base_elem = ET.Element("link", {"name": base_name})
        root_elem.insert(0, base_elem)

    # Add fixed joint
    joint = ET.SubElement(root_elem, "joint", {
        "name": f"joint_{base_name}_{kin_root}",
        "type": "fixed",
    })
    ET.SubElement(joint, "parent", {"link": base_name})
    ET.SubElement(joint, "child", {"link": kin_root})
    ET.SubElement(joint, "origin", {"xyz": ROOT_JOINT_XYZ, "rpy": ROOT_JOINT_RPY})

    return True


def _pick_canonical_root_joint(root_elem, actual_base):
    """Return the ONE base→X fixed joint to preserve.

    When multiple base→X fixed joints exist (a multi-root URDF), pick the
    one whose child has the most descendant joints — that's the "real"
    articulation root; the others are extras that should be re-routed
    into the canonical root and collapsed.
    """
    if actual_base is None:
        return None
    candidates = []
    for joint in root_elem.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        p = joint.find("parent")
        c = joint.find("child")
        if p is None or c is None:
            continue
        if p.get("link") != actual_base:
            continue
        candidates.append((joint, c.get("link", "")))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]

    # Multi-root: count descendants for each candidate child
    def _descendant_count(start: str) -> int:
        seen = {start}
        queue = [start]
        n = 0
        while queue:
            cur = queue.pop()
            for j in root_elem.findall("joint"):
                p = j.find("parent")
                c = j.find("child")
                if p is None or c is None:
                    continue
                if p.get("link") != cur:
                    continue
                cn = c.get("link", "")
                if cn in seen or not cn:
                    continue
                seen.add(cn)
                queue.append(cn)
                n += 1
        return n

    candidates.sort(key=lambda jc: (-_descendant_count(jc[1]), jc[1]))
    return candidates[0][0]


def _normalize_multi_base_fixed_joints(root_elem, actual_base, canonical_root_joint) -> int:
    """Re-route extra base→X fixed joints so their parent becomes the
    canonical root link instead of `base`.

    Mathematically: extra joint origin in base frame becomes
        new_origin_in_canonical_frame = inv(canonical_origin_in_base) @ extra_origin_in_base
    After this, the extra joint can be collapsed by the standard non-leaf
    collapse loop (parent is no longer the protected `actual_base`).
    """
    import numpy as np
    if canonical_root_joint is None or actual_base is None:
        return 0

    canonical_child_elem = canonical_root_joint.find("child")
    if canonical_child_elem is None:
        return 0
    canonical_root_link = canonical_child_elem.get("link", "")
    if not canonical_root_link:
        return 0

    canonical_origin = _origin_to_transform(canonical_root_joint.find("origin"))
    inv_canonical = np.linalg.inv(canonical_origin)

    n_normalized = 0
    for joint in list(root_elem.findall("joint")):
        if joint is canonical_root_joint:
            continue
        if joint.get("type") != "fixed":
            continue
        p_elem = joint.find("parent")
        c_elem = joint.find("child")
        if p_elem is None or c_elem is None:
            continue
        if p_elem.get("link") != actual_base:
            continue
        # extra base→X fixed joint — re-express in canonical_root_link frame
        original_origin = _origin_to_transform(joint.find("origin"))
        new_origin = inv_canonical @ original_origin
        p_elem.set("link", canonical_root_link)
        _set_origin_from_transform(joint, new_origin)
        n_normalized += 1
    return n_normalized


def collapse_extra_fixed_joints(root_elem) -> dict:
    """Iteratively collapse extra fixed joints (leaf and non-leaf).

    The script's job is strictly *fixed-joint formatting fixup* — collapse
    extras that violate the convention "exactly one fixed joint base→root_link".
    It does NOT attempt to fix logically-broken URDFs (self-loops, cyclic
    kinematic graphs); those are left in place so the downstream
    ``kinematic_tree_valid_rate`` metric honestly reflects the model output.

    Step-by-step:
      1. Identify the canonical root fixed joint (base→X). When multiple
         base→X fixed joints exist (multi-root URDF), pick the one whose
         child has the most descendant joints — that's the heuristic
         "real" articulation root; the others are extras.

         NOTE: the descendant-count heuristic is not a strict semantic
         guarantee. It works well when one branch carries the actual
         articulation tree and the others are static decoration links;
         degenerate inputs with multiple equally-articulated branches
         could pick the "wrong" root. For eval sanitisation this is
         acceptable — both choices preserve geometry.

      2. Re-route any extra base→X fixed joints so their parent becomes
         the canonical root link (not `base`); they then look like normal
         non-base extra fixed joints.

      3. Iteratively collapse every fixed joint EXCEPT the canonical root:
            - merge child link's visual/collision into parent (with the
              fixed transform applied to each geom origin)
            - reparent every joint whose parent==child_link to have
              parent=parent_link, multiplying its origin by the fixed
              transform (so the absolute pose of the rest of the chain
              is preserved)
            - remove the child link and the fixed joint
         Repeats until no more collapsible fixed joints. Each iteration
         picks one target so the tree stays consistent across mutations.

    Returns a dict with structured counters:
        {
          "base_fixed_normalized":   N,   # extra base→X re-routed
          "fixed_joints_collapsed":  N,   # actually merged + removed
          "stale_fixed_removed":     N,   # joints pointing at deleted links
        }

    Skipped (left for ``kinematic_tree_valid_rate`` to flag, not silently fixed):
      * self-loop fixed joints (parent == child)
      * cyclic kinematic graphs (a sub-joint whose child = the link we'd remove)
    """
    import numpy as np  # noqa: F811

    counters = {
        "base_fixed_normalized": 0,
        "fixed_joints_collapsed": 0,
        "stale_fixed_removed": 0,
    }

    # Identify the actual kinematic base link (first BASE_LINK_NAMES entry that exists in this URDF).
    all_links = {link.get("name") for link in root_elem.findall("link") if link.get("name")}
    actual_base = next((n for n in BASE_LINK_NAMES if n in all_links), None)

    # Pick the canonical root joint and re-route extras
    canonical_root_joint = _pick_canonical_root_joint(root_elem, actual_base)
    counters["base_fixed_normalized"] = _normalize_multi_base_fixed_joints(
        root_elem, actual_base, canonical_root_joint,
    )

    # Iteratively collapse one fixed joint at a time so the tree state stays consistent across mutations (reparenting may create new collapsible leaves).
    SAFETY_LIMIT = 1000  # plenty for any realistic URDF
    for _ in range(SAFETY_LIMIT):
        links_by_name = {
            link.get("name"): link
            for link in root_elem.findall("link")
            if link.get("name")
        }

        target = None
        for joint in root_elem.findall("joint"):
            if joint is canonical_root_joint:
                continue
            if joint.get("type") != "fixed":
                continue
            p_elem = joint.find("parent")
            c_elem = joint.find("child")
            if p_elem is None or c_elem is None:
                continue
            p_name = p_elem.get("link", "")
            c_name = c_elem.get("link", "")
            if not p_name or not c_name:
                continue
            # Self-loop: skip — `kinematic_tree_valid_rate`
            # should flag this as broken model output rather than us silently fixing it.
            if p_name == c_name:
                continue
            if p_name not in links_by_name or c_name not in links_by_name:
                # Stale joint pointing at a previously-removed link — prune.
                root_elem.remove(joint)
                counters["stale_fixed_removed"] += 1
                target = "stale"
                break
            # Cycle guard: if any joint OTHER than `joint` itself has child=p_name and parent=c_name,
            # removing c_name would create a cycle. Skip cyclic patterns and let audit flag them.
            cyclic = False
            for j2 in root_elem.findall("joint"):
                if j2 is joint:
                    continue
                p2 = j2.find("parent"); c2 = j2.find("child")
                if p2 is None or c2 is None:
                    continue
                if p2.get("link") == c_name and c2.get("link") == p_name:
                    cyclic = True; break
            if cyclic:
                continue
            target = (joint, links_by_name[p_name], links_by_name[c_name],
                      p_name, c_name)
            break

        if target is None:
            break
        if target == "stale":
            continue

        joint, parent_link, child_link, p_name, c_name = target
        parent_to_child = _origin_to_transform(joint.find("origin"))

        # Step 1: merge child geometry into parent (transformed)
        for tag in ("visual", "collision"):
            for geom_elem in list(child_link.findall(tag)):
                geom_copy = copy.deepcopy(geom_elem)
                child_geom_transform = _origin_to_transform(geom_copy.find("origin"))
                merged_transform = parent_to_child @ child_geom_transform
                _set_origin_from_transform(geom_copy, merged_transform)
                parent_link.append(geom_copy)

        # Step 2: reparent every joint whose parent==child_link to parent_link
        for sub_joint in list(root_elem.findall("joint")):
            if sub_joint is joint:
                continue
            sp = sub_joint.find("parent")
            if sp is None or sp.get("link") != c_name:
                continue
            sp.set("link", p_name)
            sub_origin = _origin_to_transform(sub_joint.find("origin"))
            new_origin = parent_to_child @ sub_origin
            _set_origin_from_transform(sub_joint, new_origin)

        # Step 3: remove the joint and the now-orphan child link.
        # Duplicate joint elements pointing at the deleted child are pruned via the stale-joint branch on subsequent iterations.
        root_elem.remove(joint)
        if child_link in list(root_elem):
            root_elem.remove(child_link)
        counters["fixed_joints_collapsed"] += 1

    return counters


# Backward-compat shim: external callers may still import the old name.
# Returns the total integer count to preserve the prior int return type expected by anything that integer-checked the result.


def fix_urdf(urdf_path: str, backup: bool = True) -> dict:
    """Fix convention violations in a URDF. Returns summary.

    The returned dict carries:
      * structured action counters (``fixed_joints_collapsed``,
        ``base_fixed_normalized``, ``stale_fixed_removed``,
        ``root_fixed_joint_added``)
      * a post-fix audit (``post_fix_audit``) so callers can decide
        whether the fix was complete or only partial — when the URDF has
        a self-loop or cyclic kinematic graph the collapse loop
        intentionally leaves them untouched, so the audit may still
        report extras even after a successful "fix" run.
    """
    import shutil

    path = Path(urdf_path)
    if backup:
        backup_path = path.with_name(path.stem + "_backup" + path.suffix)
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    root_fixed = fix_root_fixed_joint(root)
    counters = collapse_extra_fixed_joints(root)

    remaining_fixed = sum(
        1 for j in root.findall("joint") if j.get("type") == "fixed"
    )

    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(_to_pretty_xml(root))

    # Re-audit the just-written URDF so the caller can distinguish "fully fixed" from "partial"
    # (e.g. cycles / self-loops left in).
    post_fix_audit = audit_urdf(urdf_path)

    return {
        "root_fixed_joint_added": root_fixed,
        # New structured counters (preferred by new callers):
        "fixed_joints_collapsed": counters["fixed_joints_collapsed"],
        "base_fixed_normalized": counters["base_fixed_normalized"],
        "stale_fixed_removed": counters["stale_fixed_removed"],
        # Backward-compat alias (sum of all collapse-related actions);
        # NOTE: name is kept for old downstream tooling but is no longer accurate — the implementation handles non-leaf collapses too.
        "leaf_fixed_joints_collapsed": sum(counters.values()),
        "remaining_fixed_joints": remaining_fixed,
        "post_fix_audit": post_fix_audit,
        "fully_fixed": (
            post_fix_audit["root_fixed_joint_ok"]
            and not post_fix_audit["extra_fixed_joints"]
            and not post_fix_audit["multi_parent_links"]
        ),
    }


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check and optionally fix URDF convention violations for ArtiBench.",
    )
    parser.add_argument(
        "--root", type=str, required=True,
        help="Root directory containing per-object URDF folders.",
    )
    parser.add_argument(
        "--urdf_name", type=str, default="mobility.urdf",
        help="URDF filename to look for under each object directory.",
    )
    parser.add_argument(
        "--object_ids", nargs="*", default=None,
        help="Optional specific object IDs to check.",
    )
    parser.add_argument(
        "--enable-urdf-fixing", action="store_true",
        help="Fix detected violations: add root fixed joint if missing; "
             "collapse extra fixed joints (leaf and non-leaf); re-route "
             "extra base→X fixed joints into the canonical root link. "
             "Does NOT touch self-loops or cyclic kinematic graphs — "
             "those are left for kinematic_tree_valid_rate to flag.",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Skip creating backup files when fixing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root)
    if not root_dir.is_dir():
        print(f"ERROR: {root_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Discover URDFs
    if args.object_ids:
        urdf_paths = [root_dir / oid / args.urdf_name for oid in args.object_ids]
        urdf_paths = [p for p in urdf_paths if p.is_file()]
    else:
        urdf_paths = sorted(root_dir.glob(f"*/{args.urdf_name}"))

    if not urdf_paths:
        print(f"No {args.urdf_name} files found under {root_dir}")
        sys.exit(1)

    print(f"[check_urdf] Found {len(urdf_paths)} URDFs under {root_dir}")

    n_root_issues = 0
    n_extra_fixed = 0
    n_multi_parent = 0
    n_clean = 0
    n_fix_failed = 0

    for urdf_path in urdf_paths:
        oid = urdf_path.parent.name
        result = audit_urdf(str(urdf_path))

        has_issue = False

        if not result["root_fixed_joint_ok"]:
            print(f"  [WARN] {oid}: root fixed joint missing — {result['root_fixed_joint_detail']}")
            n_root_issues += 1
            has_issue = True

        if result["extra_fixed_joints"]:
            n_extra = len(result["extra_fixed_joints"])
            n_extra_fixed += 1
            has_issue = True
            details = ", ".join(
                f"{name}({p}->{c})" for name, p, c in result["extra_fixed_joints"]
            )
            print(f"  [WARN] {oid}: {n_extra} extra fixed joint(s): {details}")

        if result["multi_parent_links"]:
            n_multi_parent += 1
            has_issue = True
            details = ", ".join(
                f"{link}({'+'.join(js)})"
                for link, js in sorted(result["multi_parent_links"].items())
            )
            print(f"  [WARN] {oid}: not a tree — link(s) with 2+ parent joints: {details}")

        if not has_issue:
            n_clean += 1

        # Fix if requested
        if args.enable_urdf_fixing and has_issue:
            try:
                fix_result = fix_urdf(str(urdf_path), backup=not args.no_backup)
            except Exception as fix_err:
                # Do not let one malformed URDF kill the whole batch — the downstream eval (eval_self / eval_paired)
                # already tolerates per-sample failures, so we just log and move on.
                n_fix_failed += 1
                print(f"  [ERROR] {oid}: fix_urdf crashed — {fix_err.__class__.__name__}: {fix_err}")
                continue
            parts = []
            if fix_result["root_fixed_joint_added"]:
                parts.append("added root fixed joint")
            if fix_result["base_fixed_normalized"] > 0:
                parts.append(
                    f"normalized {fix_result['base_fixed_normalized']} extra base→X fixed joint(s)"
                )
            if fix_result["fixed_joints_collapsed"] > 0:
                parts.append(
                    f"collapsed {fix_result['fixed_joints_collapsed']} fixed joint(s)"
                )
            if fix_result["stale_fixed_removed"] > 0:
                parts.append(
                    f"removed {fix_result['stale_fixed_removed']} stale fixed joint(s)"
                )
            parts.append(f"{fix_result['remaining_fixed_joints']} fixed joint(s) remaining")

            # Distinguish complete vs partial fix based on post-fix audit.
            # Partial = self-loops or cyclic kinematic graph remain (script intentionally doesn't touch those — they should be flagged via kinematic_tree_valid_rate).
            tag = "FIXED" if fix_result["fully_fixed"] else "PARTIAL"
            if not fix_result["fully_fixed"]:
                pa = fix_result["post_fix_audit"]
                still: List[str] = []
                if not pa["root_fixed_joint_ok"]:
                    still.append("root fixed joint still missing/malformed")
                if pa["extra_fixed_joints"]:
                    detail = ", ".join(
                        f"{n}({p}->{c})" for n, p, c in pa["extra_fixed_joints"]
                    )
                    still.append(f"extra fixed joint(s) remain: {detail}")
                if pa["multi_parent_links"]:
                    detail = ", ".join(
                        f"{link}({'+'.join(js)})"
                        for link, js in sorted(pa["multi_parent_links"].items())
                    )
                    still.append(f"not a tree — 2+ parent joints on: {detail}")
                parts.append("UNRESOLVED: " + "; ".join(still))

            print(f"  [{tag}] {oid}: {'; '.join(parts)}")

    # Summary
    total = len(urdf_paths)
    print(f"\n[Summary] {total} URDFs checked:")
    print(f"  Clean: {n_clean}")
    print(f"  Missing root fixed joint: {n_root_issues}")
    print(f"  Extra fixed joints: {n_extra_fixed}")
    print(f"  Not a tree (2+ parent joints): {n_multi_parent}")
    if args.enable_urdf_fixing:
        print(f"  (fixes applied)")
        if n_fix_failed:
            print(f"  Fix crashed (skipped): {n_fix_failed}")


if __name__ == "__main__":
    main()
