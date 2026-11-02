# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
VLM-as-judge evaluation for ArtiBench.

Orchestrates rendering + VLM scoring for all 5 aspects:
  Phase 1: PART_DECOMP, PART_COMPLETE (part-colored views)
  Phase 2: MOTION, JOINT_TYPE, JOINT_AXIS (motion / joint-highlight frames)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .client import (
    RateLimiter,
    VLMCallResult,
    VLMClientConfig,
    parse_json_response,
    vlm_call_messages,
)
from .prompts import (
    ASPECT_RUBRICS, ASPECT_RENDERING_TYPE, SYSTEM_PROMPT,
    build_messages_with_few_shot, build_user_content,
)
from artibench.utils.timing import phase

logger = logging.getLogger(__name__)


# Configuration
ALL_ASPECTS = ["PART_DECOMP", "PART_COMPLETE", "MOTION", "JOINT_TYPE", "JOINT_AXIS"]


@dataclass
class VLMJudgeConfig:
    """Configuration for VLM-as-judge evaluation."""

    aspects: List[str] = field(
        default_factory=lambda: list(ALL_ASPECTS)
    )
    client: VLMClientConfig = field(default_factory=VLMClientConfig)
    num_views: int = 2
    azimuth_start: float = 120.0
    azimuth_span: float = 90.0
    elevation: float = 30.0
    resolution: int = 518
    force_render: bool = False
    blender_bin: Optional[str] = None
    json_retries: int = 3
    num_shots: int = 3


# Logging
VLM_LOG_DIR_NAME = "vlm_judge_logs"


def _save_vlm_log(
    log_dir: Path,
    aspect: str,
    call_result: VLMCallResult,
    parsed_result: Dict[str, Any],
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    if call_result.request_payload is not None:
        (log_dir / f"{aspect}_payload.json").write_text(
            json.dumps(call_result.request_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if call_result.raw_response is not None:
        (log_dir / f"{aspect}_response.json").write_text(
            json.dumps(call_result.raw_response, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (log_dir / f"{aspect}_result.json").write_text(
        json.dumps(parsed_result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _save_input_images(
    log_dir: Path,
    ref_image_path: str,
    view_paths: List[str],
    *,
    prefix: str = "part_view",
) -> None:
    images_dir = log_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    ref_dst = images_dir / f"ref{Path(ref_image_path).suffix}"
    shutil.copy2(ref_image_path, ref_dst)
    for i, vp in enumerate(view_paths):
        dst = images_dir / f"{prefix}_{i:03d}.png"
        shutil.copy2(vp, dst)


# Single-aspect evaluation
def _evaluate_aspect(
    *,
    aspect: str,
    ref_image: Image.Image,
    eval_images: List[Image.Image],
    config: VLMJudgeConfig,
    rate_limiter: Optional[RateLimiter] = None,
    log_dir: Optional[Path] = None,
    prompt_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score a single aspect via VLM with interleaved text/image content."""
    if aspect not in ASPECT_RUBRICS:
        raise ValueError(f"Unknown aspect: {aspect}. Available: {list(ASPECT_RUBRICS)}")

    messages = build_messages_with_few_shot(
        aspect,
        ref_image=ref_image,
        eval_images=eval_images,
        num_shots=config.num_shots,
        **(prompt_kwargs or {}),
    )

    def _save_failure(
        err_msg: str, err_str: str, call_result: Optional[VLMCallResult],
    ) -> Dict[str, Any]:
        # Always drop a {aspect}_result.json on failure so resume logic can find it and retry later.
        result = {
            "aspect": aspect,
            "score": 0,
            "analysis": err_msg,
            "error": err_str,
        }
        if log_dir is not None:
            _save_vlm_log(
                log_dir, aspect,
                call_result or VLMCallResult(text=""),
                result,
            )
        return result

    for attempt in range(1, config.json_retries + 1):
        try:
            with phase(f"vlm.api.{aspect}"):
                call_result = vlm_call_messages(
                    messages=messages,
                    config=config.client,
                    rate_limiter=rate_limiter,
                )
        except Exception as api_err:
            # HTTP error, quota exhausted after 429 backoff retries, network timeout, etc.
            # Never let _evaluate_aspect raise — record the failure and return a score=0 stub so the caller sees a well-formed result per aspect.
            logger.error(
                "VLM API call failed for %s (attempt %d/%d): %s",
                aspect, attempt, config.json_retries, api_err,
            )
            if attempt >= config.json_retries:
                return _save_failure(
                    err_msg=(
                        f"VLM API call failed after {config.json_retries} "
                        f"attempt(s): {api_err}"
                    ),
                    err_str=str(api_err),
                    call_result=None,
                )
            continue
        try:
            parsed = parse_json_response(call_result.text)
            score = int(parsed.get("score", 0))
            if not 1 <= score <= 5:
                raise ValueError(f"Score {score} not in [1, 5]")
            result = {
                "aspect": aspect,
                "score": score,
                "analysis": str(parsed.get("analysis", "")),
            }
            if log_dir is not None:
                _save_vlm_log(log_dir, aspect, call_result, result)
            return result
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(
                "VLM JSON parse attempt %d/%d failed for %s: %s",
                attempt, config.json_retries, aspect, e,
            )
            if attempt >= config.json_retries:
                logger.error(
                    "All JSON parse attempts failed. Raw: %s",
                    call_result.text[:500],
                )
                return _save_failure(
                    err_msg=(
                        f"JSON parse failed after {config.json_retries} "
                        "attempts"
                    ),
                    err_str=str(e),
                    call_result=call_result,
                )

    # Unreachable — the loop always returns. Defensive fallback.
    return _save_failure(
        err_msg="unreachable VLM evaluation fallthrough",
        err_str="unreachable",
        call_result=None,
    )


# Public API
def compute_vlm_scores(
    *,
    urdf_path: str,
    ref_image_path: str,
    config: Optional[VLMJudgeConfig] = None,
    log_dir: Optional[str] = None,
    rate_limiter: Optional[RateLimiter] = None,
) -> Dict[str, Any]:
    """Run VLM-as-judge evaluation on a generated articulated object.

    Returns Dict mapping aspect names to {"score": int, "analysis": str}.

    ``rate_limiter`` may be passed in to share a single QPM budget across
    multiple concurrent compute_vlm_scores calls (e.g. when an outer loop
    parallelizes samples). If None, a fresh limiter is created from
    ``config.client.max_qpm``.
    """
    from .rendering import (
        render_joint_highlight_frames,
        render_motion_frames,
        render_part_colored_views,
    )

    config = config or VLMJudgeConfig()
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    log_path = Path(log_dir) if log_dir else Path(urdf_dir) / VLM_LOG_DIR_NAME

    ref_image_path_abs = os.path.abspath(ref_image_path)
    ref_image = Image.open(ref_image_path_abs).convert("RGB")
    if rate_limiter is None:
        rate_limiter = RateLimiter(config.client.max_qpm)

    results: Dict[str, Any] = {}

    # Resume from cached per-aspect results; failed aspects are re-attempted, and force_render bypasses the cache entirely.
    aspects_to_run: List[str] = list(config.aspects)
    if not config.force_render:
        resumed: List[str] = []
        for aspect in list(aspects_to_run):
            result_file = log_path / f"{aspect}_result.json"
            if not result_file.is_file():
                continue
            try:
                prior = json.loads(result_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            score = prior.get("score")
            if not isinstance(score, int) or not 1 <= score <= 5:
                continue
            if "error" in prior:
                continue
            results[aspect] = {
                "score": int(score),
                "analysis": str(prior.get("analysis", "")),
            }
            aspects_to_run.remove(aspect)
            resumed.append(aspect)
        if resumed:
            logger.info(
                "VLM resume: reusing %d cached aspect result(s) for %s: %s",
                len(resumed), urdf_path, resumed,
            )

    if not aspects_to_run:
        logger.info("All aspects already cached for %s; skipping renders/API.", urdf_path)
        return results

    # Group remaining aspects by rendering type (only render what's needed).
    aspects_by_type: Dict[str, List[str]] = {}
    for aspect in aspects_to_run:
        rtype = ASPECT_RENDERING_TYPE.get(aspect, "part_colored")
        aspects_by_type.setdefault(rtype, []).append(aspect)

    # ── Early exit: if URDF has no actuated joints, score 1 for all pending dims ──
    from artibench.core.urdf_parser import parse_urdf as _parse_urdf
    _parsed = _parse_urdf(urdf_path, prefer_collision=False)
    _has_actuated = any(j.type != "fixed" for j in _parsed.joints)
    if not _has_actuated:
        logger.warning("No actuated joints in %s; all dimensions score 1", urdf_path)
        no_actuated_analysis = (
            "The model has no actuated joints. All parts are fixed."
        )
        for aspect in aspects_to_run:
            result = {"aspect": aspect, "score": 1, "analysis": no_actuated_analysis}
            if log_path is not None:
                _save_vlm_log(log_path, aspect, VLMCallResult(text=""), result)
            results[aspect] = {"score": 1, "analysis": no_actuated_analysis}
        return results

    # Threads, not processes: the renders must share the parent logger, the module-level caches and the current PhaseTimer.
    part_aspects = aspects_by_type.get("part_colored", [])
    motion_aspects = aspects_by_type.get("motion", [])
    joint_aspects = aspects_by_type.get("joint_highlight", [])

    def _timed_render(phase_name, fn):
        with phase(phase_name):
            return fn()

    render_results: Dict[str, Any] = {}
    render_errors: Dict[str, Exception] = {}
    render_submits: Dict[str, Any] = {}
    if part_aspects or motion_aspects or joint_aspects:
        logger.info("Dispatching VLM renders in parallel for %s", urdf_path)
        with phase("vlm.render.parallel"):
            with ThreadPoolExecutor(max_workers=3) as ex:
                if part_aspects:
                    render_submits["part_colored"] = ex.submit(
                        _timed_render, "vlm.render.part_colored",
                        lambda: render_part_colored_views(
                            urdf_path,
                            num_views=config.num_views,
                            azimuth_start=config.azimuth_start,
                            azimuth_span=config.azimuth_span,
                            elevation=config.elevation,
                            resolution=config.resolution,
                            force_render=config.force_render,
                            blender_bin=config.blender_bin,
                        ),
                    )
                if motion_aspects:
                    render_submits["motion"] = ex.submit(
                        _timed_render, "vlm.render.motion",
                        lambda: render_motion_frames(
                            urdf_path,
                            resolution=config.resolution,
                            elevation=config.elevation,
                            force_render=config.force_render,
                            blender_bin=config.blender_bin,
                        ),
                    )
                if joint_aspects:
                    render_submits["joint_highlight"] = ex.submit(
                        _timed_render, "vlm.render.joint_highlight",
                        lambda: render_joint_highlight_frames(
                            urdf_path,
                            resolution=config.resolution,
                            elevation=config.elevation,
                            force_render=config.force_render,
                            blender_bin=config.blender_bin,
                        ),
                    )
                for rtype, fut in render_submits.items():
                    try:
                        render_results[rtype] = fut.result()
                    except Exception as e:
                        render_errors[rtype] = e
                        logger.error("%s rendering failed: %s", rtype, e)

    # ── Stage 2: assemble per-aspect evaluation tasks from render results ──
    aspect_tasks: List[tuple] = []  # (aspect, eval_images, prompt_kwargs)

    if part_aspects:
        if "part_colored" in render_results:
            ext_paths, int_paths = render_results["part_colored"]
            _save_input_images(log_path, ref_image_path, ext_paths, prefix="part_view")
            if int_paths:
                _save_input_images(log_path, ref_image_path, int_paths, prefix="part_int_view")
            ext_views = [Image.open(p).convert("RGB") for p in ext_paths]
            int_views = [Image.open(p).convert("RGB") for p in int_paths]
            # Sample-level context: total_parts so the VLM knows how many distinct colored links to expect across the ext/int views.
            part_joint_metas = None
            try:
                summary = _urdf_joint_summary(urdf_path)
                if summary.get("total_parts", 0) > 0:
                    part_joint_metas = [summary]
            except Exception as e:
                logger.warning("Could not build URDF summary for PART_*: %s", e)
            for aspect in part_aspects:
                aspect_tasks.append(
                    (aspect, ext_views, {
                        "internal_images": int_views,
                        "joint_metas": part_joint_metas,
                    }),
                )
        else:
            err = render_errors.get("part_colored")
            for aspect in part_aspects:
                results[aspect] = {
                    "score": 0,
                    "analysis": f"Part-colored rendering failed: {err}",
                    "error": str(err),
                }

    if motion_aspects:
        if "motion" in render_results:
            motion_paths = render_results["motion"]
            _save_input_images(log_path, ref_image_path, motion_paths, prefix="motion_frame")
            motion_views = [Image.open(p).convert("RGB") for p in motion_paths]
            # Attach an aggregate joint summary so the MOTION user message can surface "The model has N actuated joint(s)
            # in total (x continuous, y prismatic, …)" — same context JOINT_TYPE/JOINT_AXIS get.
            motion_prompt_kwargs = None
            try:
                summary = _urdf_joint_summary(urdf_path)
                if summary["total_actuated"] > 0:
                    motion_prompt_kwargs = {"joint_metas": [summary]}
            except Exception as e:
                logger.warning("Could not build URDF joint summary for MOTION: %s", e)
            for aspect in motion_aspects:
                aspect_tasks.append((aspect, motion_views, motion_prompt_kwargs))
        else:
            err = render_errors.get("motion")
            for aspect in motion_aspects:
                results[aspect] = {
                    "score": 0,
                    "analysis": f"Motion rendering failed: {err}",
                    "error": str(err),
                }

    if joint_aspects:
        if "joint_highlight" in render_results:
            jh_paths, joint_metas = render_results["joint_highlight"]
            if not jh_paths:
                # No actuated joints passed the importance filter;
                # fall back to the same explanation the old sequential path used.
                logger.warning("No joint-highlight frames for %s; scoring 1", urdf_path)
                for aspect in joint_aspects:
                    results[aspect] = {
                        "score": 1,
                        "analysis": (
                            "The model has no actuated joints at all. "
                            "Since the reference object has articulable parts, "
                            "this is the worst case."
                        ),
                    }
            else:
                _save_input_images(log_path, ref_image_path, jh_paths, prefix="joint_highlight")
                jh_views = [Image.open(p).convert("RGB") for p in jh_paths]
                for aspect in joint_aspects:
                    aspect_tasks.append(
                        (aspect, jh_views, {"joint_metas": joint_metas}),
                    )
        else:
            err = render_errors.get("joint_highlight")
            for aspect in joint_aspects:
                results[aspect] = {
                    "score": 0,
                    "analysis": f"Joint-highlight rendering failed: {err}",
                    "error": str(err),
                }

    # ── Stage 3: run all aspect API calls concurrently ──
    if aspect_tasks:
        logger.info(
            "Dispatching %d VLM aspect call(s) in parallel", len(aspect_tasks),
        )
        with phase("vlm.api.parallel"):
            with ThreadPoolExecutor(max_workers=len(aspect_tasks)) as ex:
                futs = {
                    aspect: ex.submit(
                        _evaluate_aspect,
                        aspect=aspect,
                        ref_image=ref_image,
                        eval_images=imgs,
                        config=config,
                        rate_limiter=rate_limiter,
                        log_dir=log_path,
                        prompt_kwargs=kw,
                    )
                    for aspect, imgs, kw in aspect_tasks
                }
                for aspect, fut in futs.items():
                    try:
                        results[aspect] = _extract_result(fut.result())
                    except Exception as e:
                        logger.error("VLM call failed for %s: %s", aspect, e)
                        results[aspect] = {
                            "score": 0,
                            "analysis": f"VLM call failed: {e}",
                            "error": str(e),
                        }

    logger.info("VLM logs saved to %s", log_path)
    return results


def _extract_result(result: Dict[str, Any]) -> Dict[str, Any]:
    out = {"score": result["score"], "analysis": result["analysis"]}
    if "error" in result:
        out["error"] = result["error"]
    return out


def _urdf_joint_summary(urdf_path: str) -> Dict[str, Any]:
    """Return an aggregate URDF summary used for sample-level prompt hints.

    Keys:
      - ``total_actuated``: number of non-fixed joints (covers revolute,
        prismatic, continuous, screw).
      - ``type_summary``: e.g. ``"10 continuous, 1 revolute, 1 screw"``.
      - ``total_parts``: number of links that carry any mesh geometry
        (visual or collision). This is the count the VLM actually sees
        in the part-colored renders — base-only / empty helper links are
        excluded.
    """
    from collections import Counter
    from artibench.core.urdf_parser import parse_urdf
    obj = parse_urdf(urdf_path, prefer_collision=False)
    counts = Counter(j.type for j in obj.joints if j.type != "fixed")
    total_actuated = int(sum(counts.values()))
    type_summary = ", ".join(
        f"{cnt} {jtype}" for jtype, cnt in sorted(counts.items())
    )
    total_parts = int(sum(1 for link in obj.links if link.mesh_paths))
    return {
        "total_actuated": total_actuated,
        "type_summary": type_summary,
        "total_parts": total_parts,
    }
