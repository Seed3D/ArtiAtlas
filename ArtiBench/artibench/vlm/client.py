# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Gemini VLM client for ArtiBench.

A thin wrapper around the official Google GenAI SDK (``google-genai``) that adds
what a benchmark harness needs on top of a bare ``generate_content`` call: a
shared client-side QPM limiter, retry with exponential backoff, a separate
cooldown for rate-limit responses that does not consume the ordinary retry
budget, PIL image encoding, and image-redacted request payloads suitable for
writing to disk alongside results.

Setup::

    pip install google-genai
    export GOOGLE_API_KEY=...        # or GEMINI_API_KEY

Public API:
  - vlm_call_messages(): full multi-turn support (system + few-shot + query)
  - vlm_call(): convenience wrapper for single-turn (system + images + text)

Messages use the OpenAI content-part shape (``{"type": "text"}`` /
``{"type": "image_url"}``) and are converted to ``google-genai`` types on the
way out, so callers can build prompts with the helpers in this module without
depending on the SDK's object model.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


# Call result
@dataclass
class VLMCallResult:
    """Structured result from a VLM call, including logging data."""

    text: str  # extracted response text
    raw_response: Any = None  # full API response
    request_payload: Any = None  # loggable payload (images redacted)


# Constants
# The model the published numbers were judged with. Changing it changes five of
# the eighteen columns, so it is pinned rather than tracking the latest release.
DEFAULT_MODEL = "gemini-3.1-pro-preview"

# Gemini 3 models require temperature=1.0.
_GEMINI3_PREFIXES = ("gemini-3-", "gemini-3.0-", "gemini-3.1-")

# Rate-limit (HTTP 429 / RESOURCE_EXHAUSTED) handling. These retries do not count against `max_retries`
# — one throttled minute should not consume the whole retry budget of a long evaluation run.
_MAX_RATE_LIMIT_RETRIES = 5
_RATE_LIMIT_BACKOFF_SEC = 60.0


def _is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection of a quota / rate-limit error from the SDK."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    text = f"{type(exc).__name__}: {exc}"
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


# Configuration
@dataclass
class VLMClientConfig:
    """Configuration for the VLM client."""

    backend: str = ""  # only "google" is supported; "" means the default
    model: str = DEFAULT_MODEL
    max_tokens: int = 16384
    max_retries: int = 3
    retry_backoff: float = 2.0
    timeout: float = 120.0
    max_qpm: int = 60

    @property
    def temperature(self) -> float:
        """Gemini 3 requires temperature=1.0; others default to 0.5."""
        if any(self.model.startswith(p) for p in _GEMINI3_PREFIXES):
            return 1.0
        return 0.5


# Rate limiter
class RateLimiter:
    """Thread-safe QPM rate limiter with rolling window."""

    def __init__(self, max_qpm: int):
        self.max_qpm = int(max_qpm)
        self._lock = threading.Lock()
        self._calls: deque = deque()
        self._next_allowed: float = 0.0

    def acquire(self) -> None:
        if self.max_qpm <= 0:
            return
        interval = 60.0 / float(self.max_qpm)
        wait = 0.0
        with self._lock:
            now = time.time()
            if now < self._next_allowed:
                wait = max(wait, self._next_allowed - now)
                now = self._next_allowed
            self._next_allowed = now + interval
            while self._calls and now - self._calls[0] >= 60.0:
                self._calls.popleft()
            if len(self._calls) >= self.max_qpm:
                wait = max(wait, 60.0 - (now - self._calls[0]))
        if wait > 0:
            time.sleep(wait)
        with self._lock:
            now = time.time()
            while self._calls and now - self._calls[0] >= 60.0:
                self._calls.popleft()
            self._calls.append(now)


# Helpers
def encode_pil_image(img: Image.Image) -> str:
    """Encode a PIL Image as a base64 JPEG data URL."""
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from VLM output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json\n", "", 1)
        stripped = stripped.replace("json\r\n", "", 1)
    return stripped.strip()


def parse_json_response(text: str) -> dict:
    """Parse JSON from VLM response text, handling common formatting issues."""
    raw = strip_code_fences(text)
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1], strict=False)
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("Cannot parse JSON from VLM response", raw, 0)


# --------------------------------------------------------------------------- Content-part builders (for constructing messages)
def text_part(text: str) -> Dict[str, Any]:
    """Create a text content part (OpenAI format)."""
    return {"type": "text", "text": text}


def image_part(img: Image.Image) -> Dict[str, Any]:
    """Create an image content part from a PIL Image."""
    return {"type": "image_url", "image_url": {"url": encode_pil_image(img)}}


# Payload redaction for logging
def _redact_payload(payload: dict) -> dict:
    """Create a loggable copy of the request payload with image data replaced.

    Rebuilds the structure instead of deep-copying it: a multi-turn call can
    carry a dozen base64 images, and deep-copying them only to discard them
    doubles peak memory for nothing.
    """

    def _redact_part(part: Any) -> Any:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            return part
        data_url = (part.get("image_url") or {}).get("url", "")
        if not data_url.startswith("data:"):
            return part
        try:
            header, b64data = data_url.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
            size_bytes = len(base64.b64decode(b64data))
        except Exception:
            mime = "unknown"
            size_bytes = 0
        return {
            "type": "image_url",
            "image_url": {
                "data": "<omitted>",
                "mime_type": mime,
                "size_bytes": size_bytes,
            },
        }

    redacted = dict(payload)
    messages = payload.get("messages")
    if isinstance(messages, list):
        out_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                out_messages.append(msg)
                continue
            new_msg = dict(msg)
            content = msg.get("content")
            if isinstance(content, list):
                new_msg["content"] = [_redact_part(p) for p in content]
            out_messages.append(new_msg)
        redacted["messages"] = out_messages
    return redacted


# Google GenAI backend
def _convert_parts_to_genai(parts_list, types):
    """Convert OpenAI-format content parts to google-genai Part objects."""
    out = []
    for part in parts_list:
        if part.get("type") == "text":
            out.append(types.Part(text=part["text"]))
        elif part.get("type") == "image_url":
            data_url = part["image_url"]["url"]
            header, b64data = data_url.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
            image_bytes = base64.b64decode(b64data)
            out.append(
                types.Part(inline_data=types.Blob(data=image_bytes, mime_type=mime))
            )
    return out


def _resolve_api_key() -> str:
    api_key = (os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError(
            "No API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) to a "
            "Google AI Studio key: https://aistudio.google.com/apikey"
        )
    return api_key


def _call_google_messages(
    *,
    messages: List[Dict[str, Any]],
    config: VLMClientConfig,
    rate_limiter: Optional[RateLimiter] = None,
) -> VLMCallResult:
    """Call the Gemini API with a full messages list."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError(
            "google-genai is required for the VLM judge. "
            "Install it with: pip install google-genai"
        ) from e

    client = genai.Client(api_key=_resolve_api_key())

    # Separate system prompt from conversation turns
    system_instruction = None
    contents = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            system_instruction = content if isinstance(content, str) else json.dumps(content)
            continue

        # Map "assistant" -> "model" for google-genai
        genai_role = "model" if role == "assistant" else "user"

        if isinstance(content, str):
            parts = [types.Part(text=content)]
        elif isinstance(content, list):
            parts = _convert_parts_to_genai(content, types)
        else:
            parts = [types.Part(text=str(content))]

        contents.append(types.Content(role=genai_role, parts=parts))

    loggable_payload = {
        "messages": messages,
        "max_tokens": int(config.max_tokens),
        "temperature": config.temperature,
        "model": config.model,
    }

    attempt = 0
    rate_limit_retries = 0
    max_retries = max(1, config.max_retries)
    while True:
        attempt += 1
        if rate_limiter is not None:
            rate_limiter.acquire()
        try:
            response = client.models.generate_content(
                model=config.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=config.temperature,
                    max_output_tokens=config.max_tokens,
                ),
            )
            text = response.text or ""
            if not text.strip():
                for cand in getattr(response, "candidates", []) or []:
                    cand_content = getattr(cand, "content", None)
                    if not cand_content:
                        continue
                    for p in getattr(cand_content, "parts", []) or []:
                        t = getattr(p, "text", None)
                        if t:
                            text += t
            if not text.strip():
                raise RuntimeError("Empty response from the Gemini API")

            raw_resp: Dict[str, Any] = {"text": text, "model": config.model}
            usage = getattr(response, "usage_metadata", None)
            if usage:
                raw_resp["usage_metadata"] = {
                    "prompt_token_count": getattr(usage, "prompt_token_count", None),
                    "candidates_token_count": getattr(usage, "candidates_token_count", None),
                    "total_token_count": getattr(usage, "total_token_count", None),
                }

            return VLMCallResult(
                text=text,
                raw_response=raw_resp,
                request_payload=_redact_payload(loggable_payload),
            )
        except Exception as exc:
            if _is_rate_limit_error(exc) and rate_limit_retries < _MAX_RATE_LIMIT_RETRIES:
                rate_limit_retries += 1
                logger.warning(
                    "Rate limited by the Gemini API (cooldown %d/%d), sleeping %.0fs",
                    rate_limit_retries, _MAX_RATE_LIMIT_RETRIES, _RATE_LIMIT_BACKOFF_SEC,
                )
                time.sleep(_RATE_LIMIT_BACKOFF_SEC)
                attempt -= 1  # a throttled call does not consume the retry budget
                continue
            if attempt >= max_retries:
                raise
            logger.warning(
                "Gemini API attempt %d/%d failed, retrying...\n%s",
                attempt, max_retries, traceback.format_exc(),
            )
            time.sleep(config.retry_backoff ** attempt)


# Public API
def vlm_call_messages(
    *,
    messages: List[Dict[str, Any]],
    config: VLMClientConfig,
    rate_limiter: Optional[RateLimiter] = None,
) -> VLMCallResult:
    """Call the VLM with a full messages list.

    Supports multi-turn conversations with interleaved text/image content::

        messages = [
            {"role": "system", "content": "You are ..."},
            # few-shot example
            {"role": "user", "content": [
                {"type": "text", "text": "Evaluate this:"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
            ]},
            {"role": "assistant", "content": '{"score": 4, "analysis": "..."}'},
            # actual query
            {"role": "user", "content": [...]},
        ]
    """
    backend = (config.backend or os.getenv("ARTIBENCH_VLM_BACKEND", "") or "google").strip().lower()
    if backend != "google":
        raise ValueError(
            f"Unsupported VLM backend {backend!r}. This release supports the "
            "official Google GenAI backend only ('google')."
        )
    logger.info("VLM call (model=%s)", config.model)
    return _call_google_messages(
        messages=messages, config=config, rate_limiter=rate_limiter,
    )


def vlm_call(
    *,
    system_prompt: str,
    images: List[Image.Image],
    user_text: str,
    config: VLMClientConfig,
    rate_limiter: Optional[RateLimiter] = None,
    image_paths: Optional[List[str]] = None,
) -> VLMCallResult:
    """Convenience wrapper: single-turn call with system prompt + images + text.

    For multi-turn or interleaved text/image, use vlm_call_messages() directly.
    """
    user_parts: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    for img in images:
        user_parts.append(image_part(img))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_parts},
    ]
    return vlm_call_messages(
        messages=messages, config=config, rate_limiter=rate_limiter,
    )
