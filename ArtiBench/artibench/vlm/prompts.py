# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
VLM-as-judge prompt system for ArtiBench.

Provides:
  - SYSTEM_PROMPT: shared domain knowledge for articulated object evaluation
  - build_user_content(): constructs interleaved text/image content parts
  - ASPECT_RUBRICS: per-aspect scoring rubrics (1-5 scale)
  - ASPECT_RENDERING_TYPE: maps each aspect to its rendering pipeline

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .client import image_part, text_part

# --------------------------------------------------------------------------- System prompt (shared across all aspects)
SYSTEM_PROMPT_BASE = """\
<role>
You are an expert evaluator of computer-generated 3D articulated objects.
You assess how faithfully a generated model reproduces the mechanical structure \
visible in a reference photograph.
</role>

<knowledge_base>
**Joint taxonomy** (the five mechanical joint types):
- **Revolute**: rotation with angular limits (door hinges, lid pivots, levers).
- **Prismatic**: linear translation (drawers sliding out, push buttons).
- **Continuous**: unlimited rotation (knobs, wheels, turntable swivels).
- **Screw**: coupled rotation + translation (threaded bottle caps, office-chair \
seat height adjustment, pump nozzles).
- **Fixed**: no motion — used only for the root/base part or static decorative \
sub-parts that were incorrectly split during segmentation.

**Axis placement**:
- Use the reference photograph and your physical intuition to infer where \
each joint's axis of rotation or translation should be.
- Revolute joints: the axis should be at the **pivot edge** — the edge where \
the part is attached to the body. For doors with visible handles, the hinge \
is typically on the opposite side; but many revolute parts (lids, levers, \
flaps) have no handle — infer the pivot from the part's shape and attachment.
- Prismatic joints: the axis should align with the **sliding direction** \
(e.g., along the rail for drawers, perpendicular to the panel for buttons).
- Continuous joints: the axis passes through the **center of rotation** \
(e.g., center of a wheel or knob).
- Screw joints: treated the same as continuous/prismatic for axis placement \
— the axis passes through the rotation center AND aligns with the \
translation direction (e.g., a bottle cap's screw axis runs along the \
threaded spindle).

**Part completeness**:
- A drawer part should include the **container slot** (the box behind the panel). \
If the container is missing, the drawer is incomplete.
- A door part should be a **flat panel** only. If it has an incorrectly attached \
container box, that is a decomposition error.

**Small controls — leniency vs expectation**:
- **Small flat buttons and printed labels** on a control panel are surface \
textures. The model is NOT required to reproduce them as separate parts — \
but if it does generate them, that is fine (do NOT penalize extra buttons).
- **Prominent knobs/dials** that visibly protrude from the surface in the \
reference (e.g., a large washing-machine program selector, an oven \
temperature dial) ARE expected as independent articulable parts. Missing \
a clearly prominent knob is a completeness error.
- Rule of thumb: if the control is **flush/flat** on the panel, it is \
optional; if it **protrudes in 3D**, it is expected.

**Assembly quality**:
- Parts should fit together tightly, as in a real manufactured object. \
Large visible gaps between parts that should be touching (e.g., a door \
that floats away from the body, or panels that don't align) indicate \
poor part geometry or placement.

**Handle rules**:
- Handles that are **rigidly fixed** to a door or drawer are NOT \
separate articulated parts — they belong to the same link as the panel they \
are attached to.
- Only count a handle as a separate part if it has its own independent motion \
(e.g., a rotating latch).

**Symmetric rotation** (do NOT penalize):
- A geometrically symmetric part (e.g., a perfectly round wheel or knob \
with uniform spokes) rotating around its symmetry axis will show **no \
visible change** between frames. This is correct physics — the rotation \
is real but visually undetectable. Do NOT treat this as "no motion." \
Conversely, if a supposedly symmetric wheel shows visible wobble or offset \
during rotation, that indicates an off-center axis (penalize).
</knowledge_base>

<output_format>
You MUST respond with ONLY a valid JSON object — no other text before or after.
The object has exactly two fields in this order: `score` (integer 1-5) \
first, then `analysis` (a plain-text explanation). Always open the \
analysis with "The reference image shows ..." describing the real \
object, then compare the generated model against the reference and \
apply the scoring rubric.
Example:
```json
{"score": 4, "analysis": "The reference image shows a cabinet with two doors and a drawer. The generated model captures all three parts but ..."}
```
Replace the values with your actual score and analysis.
</output_format>"""


# Backward-compat alias (deprecated — prefer build_system_prompt())
SYSTEM_PROMPT = SYSTEM_PROMPT_BASE


# Aspect rubrics
@dataclass
class AspectRubric:
    name: str
    description: str
    rubric: str
    analysis_guide: str


_REF_MISMATCH_RULE = """\
FIRST CHECK: If the generated model is an entirely different type of object \
from the reference photograph (e.g., a chair when the reference shows a \
washing machine), score **1** immediately — do not evaluate further."""

ASPECT_RUBRICS: Dict[str, AspectRubric] = {
    "PART_DECOMP": AspectRubric(
        name="PART_DECOMP",
        description="Part Decomposition Quality",
        rubric=f"""\
{_REF_MISMATCH_RULE}

Scoring procedure:
1. Identify all distinct **functional regions** in the reference (e.g., a \
washing machine has door / drum / knob / buttons = 4 regions).
2. For the decomposition of parts that ARE present, start from 5:
   - Parts at correct mechanical boundaries, tight assembly: 5.
   - One minor boundary error or small gap: 4.
   - 1-2 notable errors (wrong merge/split, floating parts): 3.
   - Multiple wrong boundaries: 2.
3. Apply a **missing-part penalty**: deduct 1 point if any functional \
region from the reference is missing (merged into body or absent). Deduct \
at most 1 regardless of how many are missing — this dimension focuses on \
decomposition quality, not completeness.
4. Functionally reasonable extra parts: no penalty. Unreasonable splits: -1.
Minimum score is 1.""",
        analysis_guide="""\
Compare the generated model's part structure to the reference.
IMPORTANT: Meshes with the **same color** that are **adjacent or touching** \
belong to the **same rigid link** — they are NOT separate parts. However, two \
distant regions that happen to share a similar color may be different parts \
(the palette has limited colors). Use spatial adjacency, not color alone, \
to decide part identity.
Check: (1) Are parts separated at the right mechanical boundaries? \
(2) Are fixed handles correctly merged with their panel? \
(3) Do parts fit together tightly, or are there visible gaps? \
(4) Are any parts from the reference missing or incorrectly merged?""",
    ),

    "PART_COMPLETE": AspectRubric(
        name="PART_COMPLETE",
        description="Part Completeness",
        rubric=f"""\
{_REF_MISMATCH_RULE}

Scoring procedure:
1. Identify all distinct **functional regions** in the reference (e.g., a \
washing machine: door / drum / knob / buttons = 4 regions; a microwave: \
door / knob = 2 regions).
2. Count how many are present in the generated model. Score by the \
fraction present:
   - All present: 5.
   - One missing: 4.
   - About half missing: 3.
   - Most missing: 2.
   - Nearly all missing: 1.
3. Drawers must include the container slot; doors must be flat panels.
4. Functionally reasonable extra parts: no penalty.""",
        analysis_guide="""\
Count the functional regions in the reference image (doors, drawers, lids, \
knobs, drums, etc.). Check if each has a corresponding part in the model. \
Score based on the fraction present.""",
    ),

    "MOTION": AspectRubric(
        name="MOTION",
        description="Articulation Motion Quality",
        rubric=f"""\
{_REF_MISMATCH_RULE}

Scoring procedure:
1. From the reference, identify expected motions and assign a **point budget** \
(total = 4). Distribute points across functional parts roughly by importance \
(e.g., a microwave with door + knob → door=2.5, knob=1.5; a cabinet with door + \
drawer → door=2, drawer=2).

2. For each expected part, compare rest state vs fully-open state. Award \
its full points if the motion is **accurate**: correct axis placement \
(e.g., hinge at the correct edge), correct direction, plausible range \
(e.g., ~90° for a microwave door), and no interpenetration.

3. Deduct from that part's budget:
   - **Minor issue** (slight clipping at extremes, range slightly off, \
small drift from seam): deduct ~half of that part's points.
   - **Major issue** (wrong direction + penetration, severe skew, \
translating instead of rotating, or completely wrong range): deduct \
all of that part's points.
   - **Part entirely missing** (expected from reference but absent or \
static in the sequence): deduct all of that part's points.

4. **Extra parts** not expected from the reference: no penalty if their \
motion is reasonable and physically plausible. If their motion is broken \
or they are functionally unrelated to the object, deduct 1 point.

Note: A geometrically symmetric part (round wheel, uniform knob) rotating \
around its symmetry axis will look identical across frames — correct physics, \
NOT a lack of motion.

Floor: minimum **2** if the model is the correct type of object and has at \
least one correctly moving part. Score **1** only for completely nonsensical \
motion or an entirely different object.""",
        analysis_guide="""\
Compare the rest state (Frame 1) with the open states, especially the fully-open state (last frame). \
For each moving part:
1. Is the motion type correct (rotation vs translation)?
2. Is the axis/pivot at the right location (hinge at edge, not center)?
3. Is the direction correct (outward, not inward)?
4. Is the range plausible (doors ~90-120°, drawers extend their depth)?
5. Is there any interpenetration? Check BOTH camera views.
Also check: are there expected motions from the reference that are missing?""",
    ),

    "JOINT_TYPE": AspectRubric(
        name="JOINT_TYPE",
        description="Joint Type Correctness",
        rubric=f"""\
{_REF_MISMATCH_RULE}

The model has {{total_joints}} actuated joint(s) in total \
({{type_summary}}). The top {{n_joints}} by importance are shown above.

Score in two steps:

**Step 1 — Type correctness** of the SHOWN joints (fraction correct):
- All correct: 5.
- One wrong: 4.
- About half wrong: 3.
- Most wrong: 2.

**Step 2 — Completeness**: compare the model's total joint count to the \
number of functional regions expected from the reference. If the model has \
significantly fewer joints than expected (e.g., ref has 4 drawers but model \
has only 1), reduce proportionally (even if the shown joints are correct).

Floor: if the model resembles the reference and has at least some correct \
joints, minimum score is **2**. Score 1 is reserved for all-wrong types or \
an entirely different object.""",
        analysis_guide="""\
Step 1: Look at the reference photograph. Identify the expected articulable \
parts and their types (doors=revolute, drawers=prismatic, knobs=continuous, \
etc.).
Step 2: You are shown {n_joints} joint-highlight image(s) out of the \
model's {total_joints} total joints ({type_summary}). For each shown joint, \
check if its claimed type matches the expected type from the reference.
Step 3: Compare the model's total joint count to what the reference suggests. \
If many expected parts are missing, reduce the score proportionally.""",
    ),

    "JOINT_AXIS": AspectRubric(
        name="JOINT_AXIS",
        description="Joint Axis Placement and Orientation",
        rubric=f"""\
{_REF_MISMATCH_RULE}

The model has {{total_joints}} actuated joint(s) in total \
({{type_summary}}). The top {{n_joints}} by importance are shown above.

Scoring procedure:

**Step 1 — Build a point budget (total = 4, plus 1 base point).**
Distribute the 4 points across the shown joints by importance in the \
reference object. For example:
- Single dominant joint (e.g., a microwave door): 4 points → direction=2, \
position=2.
- Two equally important joints (e.g., cabinet door + drawer): 2 points \
each → direction=1, position=1 per joint.
- Four joints (e.g., 2 oven doors + 2 racks): 1 point each → direction=0.5, \
position=0.5 per joint.
- If some joints are clearly more important than others, give them a \
larger share (e.g., main door = 2.5, secondary buttons = 0.5 each).

**Step 2 — Per-joint deductions.** For each shown joint, compare the \
predicted green axis against the expected axis and deduct from that \
joint's allocated points:
  (a) **Direction / orientation** (half of the joint's points):
      - Correct direction: no deduction.
      - Slightly off (a few degrees of skew): lose half of the direction \
sub-budget.
      - Completely wrong axis (e.g., vertical instead of horizontal, or \
perpendicular to correct): lose the full direction sub-budget.
  (b) **Origin / position** (the other half):
      - Exactly on the expected mechanical location: no deduction.
      - Slightly offset (axis floats near but not on the seam, a small \
shift from the knob center): lose half of the position sub-budget.
      - Clearly on the wrong side of the part, or far away from the \
correct location: lose the full position sub-budget.

Inspect the zoomed-in inset (if present) and BOTH camera views — an axis \
may look right from one angle but float from another.

**Step 3 — Completeness**: compare the model's total joint count to the \
number of functional regions expected from the reference. If the model has \
significantly fewer joints than expected (e.g., ref has 4 drawers but the \
model has only 1), reduce proportionally (even if the shown joint axes \
are all correct).

Sum the remaining points and add the 1 base point to obtain the final \
1-5 score (round to the nearest integer).

Floor: minimum **2** if the model resembles the reference and has at \
least some correct axes. Score **1** only for random axes or an entirely \
different object.""",
        analysis_guide="""\
Follow the rubric's point-budget procedure.  First inspect the reference \
to identify how many joints to expect and how to distribute the 4-point \
budget across them.  Then, for each shown joint, examine the green axis \
line in BOTH camera views and the zoom inset (if present), applying the \
direction and position deductions:
- **Revolute**: direction = parallel to the hinge axis; position = on the \
pivot edge/seam (not floating, not at center).
- **Prismatic**: direction = along the sliding axis; position = anywhere \
on that line is acceptable.
- **Continuous**: direction = symmetry axis; position = through the \
rotation center.
- **Screw**: direction = along the threaded spindle (the combined \
rotation + translation axis); position = through the spindle center \
(same criteria as continuous).
A common failure mode is correct direction but offset position — the axis \
floats parallel to where it should be.""",
    ),
}

def build_system_prompt(
    aspect: str,
    joint_metas: Optional[List[Dict[str, Any]]] = None,
    n_eval_images: int = 0,
) -> str:
    """Build a system prompt that appends the current dimension's rubric.

    Moves the bulky "Scoring procedure" block into the system message so it
    is not repeated across every user turn (especially wasteful with
    multi-turn few-shot examples).  The user messages only need to reference
    the dimension name to evaluate against; the rubric itself lives here.

    Template variables for joint-highlight aspects ({n_joints},
    {total_joints}, {type_summary}) are expanded using the provided
    joint_metas and n_eval_images of the query being scored.
    """
    rubric = ASPECT_RUBRICS[aspect]
    rendered_rubric = rubric.rubric
    rendered_guide = rubric.analysis_guide

    rtype = ASPECT_RENDERING_TYPE.get(aspect, "")
    if rtype == "joint_highlight" and joint_metas:
        n_j = n_eval_images or len(joint_metas)
        total_j = joint_metas[0].get("total_actuated", n_j)
        type_sum = joint_metas[0].get("type_summary", "")
        for tpl_var, val in [("{n_joints}", str(n_j)),
                             ("{total_joints}", str(total_j)),
                             ("{type_summary}", type_sum)]:
            rendered_rubric = rendered_rubric.replace(tpl_var, val)
            rendered_guide = rendered_guide.replace(tpl_var, val)

    rubric_block = (
        f"\n\n<scoring_rubric aspect=\"{rubric.name}\">\n"
        f"You will be asked to evaluate the **{rubric.name}** dimension — "
        f"{rubric.description}.\n\n"
        f"**Scoring rubric:**\n{rendered_rubric}\n\n"
        f"**Analysis guide:**\n{rendered_guide}\n"
        f"</scoring_rubric>"
    )
    return SYSTEM_PROMPT_BASE + rubric_block


# Mapping from aspect to rendering pipeline
ASPECT_RENDERING_TYPE: Dict[str, str] = {
    "PART_DECOMP": "part_colored",
    "PART_COMPLETE": "part_colored",
    "MOTION": "motion",
    "JOINT_TYPE": "joint_highlight",
    "JOINT_AXIS": "joint_highlight",
}


# Interleaved content builders
FEW_SHOT_ROOT = Path(__file__).resolve().parent / "few_shot_examples"


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_few_shot_examples(
    aspect: str,
    num_shots: int,
) -> List[Dict[str, Any]]:
    """Load up to num_shots few-shot examples for the given aspect.

    Each example dict contains:
        ref_image: PIL.Image
        eval_images: List[PIL.Image]
        internal_images: List[PIL.Image]  (only for part_colored)
        joint_metas: List[Dict] or None   (only for joint_highlight)
        response: {"analysis": str, "score": int}
    """
    aspect_dir = FEW_SHOT_ROOT / aspect
    if not aspect_dir.is_dir() or num_shots <= 0:
        return []

    rtype = ASPECT_RENDERING_TYPE.get(aspect, "")
    example_dirs = sorted(d for d in aspect_dir.iterdir() if d.is_dir())[:num_shots]
    examples: List[Dict[str, Any]] = []

    for ex_dir in example_dirs:
        ex: Dict[str, Any] = {}
        ex["ref_image"] = _load_image(ex_dir / "ref.png")
        ex["response"] = json.loads((ex_dir / "response.json").read_text(encoding="utf-8"))

        if rtype == "part_colored":
            ext_paths = sorted(ex_dir.glob("ext_view_*.png"))
            int_paths = sorted(ex_dir.glob("int_view_*.png"))
            ex["eval_images"] = [_load_image(p) for p in ext_paths]
            ex["internal_images"] = [_load_image(p) for p in int_paths]
            # Optional: a joint_metas.json carrying {"total_parts": N} gives the same sample-level hint the query receives.
            metas_path = ex_dir / "joint_metas.json"
            if metas_path.is_file():
                metas_blob = json.loads(metas_path.read_text(encoding="utf-8"))
                if metas_blob.get("total_parts", 0) > 0:
                    ex["joint_metas"] = [{
                        "total_parts": int(metas_blob["total_parts"]),
                    }]
                else:
                    ex["joint_metas"] = None
            else:
                ex["joint_metas"] = None
        elif rtype == "motion":
            paths = sorted(ex_dir.glob("motion_frame_*.png"))
            ex["eval_images"] = [_load_image(p) for p in paths]
            ex["internal_images"] = []
            # Optional: a joint_metas.json carrying only aggregate stats ({"total_actuated": N,
            # "type_summary": "..."}) lets the example's user message surface the same context the query gets.
            metas_path = ex_dir / "joint_metas.json"
            if metas_path.is_file():
                metas_blob = json.loads(metas_path.read_text(encoding="utf-8"))
                ex["joint_metas"] = [{
                    "total_actuated": metas_blob.get("total_actuated", 0),
                    "type_summary": metas_blob.get("type_summary", ""),
                }]
            else:
                ex["joint_metas"] = None
        elif rtype == "joint_highlight":
            paths = sorted(ex_dir.glob("joint_view_*.png"))
            ex["eval_images"] = [_load_image(p) for p in paths]
            ex["internal_images"] = []
            metas_path = ex_dir / "joint_metas.json"
            if metas_path.is_file():
                metas_blob = json.loads(metas_path.read_text(encoding="utf-8"))
                # Reconstruct joint_metas list with the fields build_user_content reads
                joints = metas_blob.get("joints", [])
                total = metas_blob.get("total_actuated", len(joints))
                summary = metas_blob.get("type_summary", "")
                ex["joint_metas"] = [
                    {
                        **j,
                        "total_actuated": total,
                        "type_summary": summary,
                    }
                    for j in joints
                ]
            else:
                ex["joint_metas"] = None
        examples.append(ex)
    return examples


def build_messages_with_few_shot(
    aspect: str,
    *,
    ref_image: Image.Image,
    eval_images: List[Image.Image],
    joint_metas: Optional[List[Dict[str, Any]]] = None,
    internal_images: Optional[List[Image.Image]] = None,
    num_shots: int = 3,
) -> List[Dict[str, Any]]:
    """Build the full message list (system + few-shot pairs + query).

    Returns OpenAI-format messages: [{role, content}, ...].
    Few-shot examples are loaded from artibench/vlm/few_shot_examples/{aspect}/.
    """
    system_prompt = build_system_prompt(
        aspect,
        joint_metas=joint_metas,
        n_eval_images=len(eval_images),
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]

    examples = _load_few_shot_examples(aspect, num_shots)
    for ex in examples:
        ex_user = build_user_content(
            aspect,
            ref_image=ex["ref_image"],
            eval_images=ex["eval_images"],
            joint_metas=ex["joint_metas"],
            internal_images=ex["internal_images"],
        )
        messages.append({"role": "user", "content": ex_user})
        messages.append({
            "role": "assistant",
            "content": json.dumps(ex["response"], ensure_ascii=False),
        })

    # Final query
    query_user = build_user_content(
        aspect,
        ref_image=ref_image,
        eval_images=eval_images,
        joint_metas=joint_metas,
        internal_images=internal_images,
    )
    messages.append({"role": "user", "content": query_user})
    return messages


def build_user_content(
    aspect: str,
    *,
    ref_image: Image.Image,
    eval_images: List[Image.Image],
    joint_metas: Optional[List[Dict[str, Any]]] = None,
    internal_images: Optional[List[Image.Image]] = None,
) -> List[Dict[str, Any]]:
    """Build interleaved text/image content parts for a VLM evaluation call.

    Returns a list of content parts (OpenAI format) ready to be used as
    the ``content`` field of a user message::

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(...)},
        ]
    """
    rubric = ASPECT_RUBRICS[aspect]
    rtype = ASPECT_RENDERING_TYPE[aspect]

    parts: List[Dict[str, Any]] = []
    int_imgs = internal_images or []

    # ── Reference image ──
    if rtype == "part_colored":
        parts.append(text_part(
            "**Image 1 — Reference photograph** of the real object. Use this to "
            "understand what parts the object should have and how they are arranged:"
        ))
    elif rtype == "motion":
        parts.append(text_part(
            "**Image 1 — Reference photograph** of the real object. Use this to "
            "understand what motions (opening, sliding, rotating) the parts should "
            "exhibit:"
        ))
    elif rtype == "joint_highlight":
        parts.append(text_part(
            "**Image 1 — Reference photograph** of the real object. Use this to "
            "infer the expected number of joints, their types, and axis placements:"
        ))
    else:
        parts.append(text_part("**Image 1 — Reference photograph** of the real object:"))
    parts.append(image_part(ref_image))

    # ── Evaluation images (interleaved with per-image descriptions) ──
    if rtype == "part_colored":
        n_ext = len(eval_images)
        n_int = len(int_imgs)
        n_total = n_ext + n_int
        img_idx = 2  # next image number (Image 1 = ref)

        # External views
        parts.append(text_part(
            f"**External structure** — {n_ext} view(s) showing the **outer parts** "
            "(body, doors, drawers, panels) of the generated 3D model at rest state. "
            "Each distinct color represents a separate rigid part (link). "
            f"All {n_ext} images show the SAME decomposition from different angles.\n"
            "Image layout: left half = **solid opaque**; right half = **translucent**.\n"
            "Note: whether a link is rendered as external vs internal is a "
            "rendering-time heuristic based on ray-cast visibility from multiple "
            "azimuths. It is NOT evidence about where the part physically sits "
            "inside the object; a part may be shown as internal purely because "
            "the occlusion test placed it there for clearer visualization."
        ))
        for i, img in enumerate(eval_images):
            parts.append(text_part(f"Image {img_idx} — External view {i+1}:"))
            parts.append(image_part(img))
            img_idx += 1

        # Internal views (if present)
        if int_imgs:
            parts.append(text_part(
                f"**Internal structure** — {n_int} view(s) showing **inner parts** "
                "(trays, racks, internal compartments) that are enclosed inside the "
                "outer shell. The outer shell is shown as a faint gray ghost outline "
                "for spatial reference. Each color represents a separate internal part.\n"
                "Image layout: left half = **solid**; right half = **translucent**.\n"
                "IMPORTANT: The internal and external images show **different sets of "
                "parts**. Do NOT match colors across external and internal images — "
                "they use independent color assignments.\n"
                "Note: the external/internal split is a rendering-time heuristic "
                "(visibility-based), not a physical location claim — a part \n"
                "appearing in the internal view does NOT necessarily mean the real "
                "model places it inside the outer shell."
            ))
            for i, img in enumerate(int_imgs):
                parts.append(text_part(f"Image {img_idx} — Internal view {i+1}:"))
                parts.append(image_part(img))
                img_idx += 1

    elif rtype == "motion":
        labels = ["Rest state (closed)", "33% open", "67% open", "Fully open"]
        n = len(eval_images)
        parts.append(text_part(
            f"**Images 2-{n+1} — Motion sequence** ({n} frames) of the generated "
            "3D model, showing all joints articulating from closed to fully open. "
            "Each distinct color represents a separate link.\n"
            "Image layout: each frame shows two camera angles side-by-side.\n"
            "The frames are ordered chronologically from rest to fully open."
        ))
        for i, img in enumerate(eval_images):
            label = labels[i] if i < len(labels) else f"{int(i / max(n - 1, 1) * 100)}% open"
            parts.append(text_part(f"Image {i+2} — Frame {i+1} of {n}: **{label}**"))
            parts.append(image_part(img))

    elif rtype == "joint_highlight":
        n_joints = len(eval_images)
        parts.append(text_part(
            f"**Images 2-{n_joints+1} — {n_joints} per-joint highlighted view(s)** "
            "of the generated 3D model at rest state. Each image highlights "
            "ONE joint:\n"
            "- **Blue** = parent link (static side)\n"
            "- **Red** = child link (moving side)\n"
            "- **Green line** = predicted joint axis (direction and origin)\n"
            "- Other links are translucent gray.\n"
            "Image layout: two camera angles side-by-side. Small child parts \n"
            "may include a zoomed-in inset for axis-position inspection."
        ))
        for i, img in enumerate(eval_images):
            jtype = joint_metas[i]["type"] if joint_metas and i < len(joint_metas) else "?"
            parts.append(text_part(f"Image {i+2} — Joint {i+1}: {jtype}"))
            parts.append(image_part(img))

    # The scoring rubric lives in the system prompt; this message only names the dimension and adds sample-specific joint counts.
    extra_context = ""
    if rtype == "joint_highlight" and joint_metas:
        n_j = len(eval_images)
        total_j = joint_metas[0].get("total_actuated", n_j)
        type_sum = joint_metas[0].get("type_summary", "")
        extra_context = (
            f"\n\nThe model has {total_j} actuated joint(s) in total "
            f"({type_sum}). The top {n_j} by importance are shown above."
        )
    elif rtype == "motion" and joint_metas:
        total_j = joint_metas[0].get("total_actuated", 0)
        type_sum = joint_metas[0].get("type_summary", "")
        if total_j > 0:
            extra_context = (
                f"\n\nFor reference, the model has {total_j} actuated "
                f"joint(s) in total ({type_sum}). All of them animate "
                "together across the motion sequence above."
            )
    elif rtype == "part_colored" and joint_metas:
        total_p = joint_metas[0].get("total_parts", 0)
        if total_p > 0:
            extra_context = (
                f"\n\nFor reference, the model decomposes into {total_p} "
                "part(s) in total (counting every link with mesh geometry). "
                "The external + internal views above together cover all of "
                "them — remember the split is a rendering heuristic, not a "
                "physical-location claim."
            )

    parts.append(text_part(
        f"\nEvaluate the **{rubric.name}** dimension "
        f"({rubric.description}) for the model above, following the "
        f"scoring rubric given in the system instructions.{extra_context}\n\n"
        "Respond with ONLY valid JSON in the format specified by "
        'the system output_format: {"score": <1-5>, "analysis": "..."}'
    ))

    return parts
