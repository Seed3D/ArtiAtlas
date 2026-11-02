# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""VLM-as-judge evaluation package for ArtiBench."""

from .client import (
    RateLimiter,
    VLMCallResult,
    VLMClientConfig,
    encode_pil_image,
    parse_json_response,
    vlm_call,
    vlm_call_messages,
)
from .judge import ALL_ASPECTS, VLMJudgeConfig, compute_vlm_scores
from .prompts import ASPECT_RUBRICS, ASPECT_RENDERING_TYPE, SYSTEM_PROMPT, build_user_content

__all__ = [
    # client
    "RateLimiter",
    "VLMCallResult",
    "VLMClientConfig",
    "encode_pil_image",
    "parse_json_response",
    "vlm_call",
    "vlm_call_messages",
    # judge
    "ALL_ASPECTS",
    "VLMJudgeConfig",
    "compute_vlm_scores",
    # prompts
    "ASPECT_RUBRICS",
    "ASPECT_RENDERING_TYPE",
    "SYSTEM_PROMPT",
    "build_user_content",
]
