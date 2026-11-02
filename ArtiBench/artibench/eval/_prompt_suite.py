# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Prompt-suite manifest loader.

A *prompt suite* is an unzipped suite directory (see ``artibench.data``)
containing:

- ``manifest.json`` — describes each sample and records the one chosen
  reference view per sample in the ``prompt_view_index`` /
  ``prompt_image_path`` fields. The choice is shipped **as data**: it is
  fixed for a released suite, because changing it changes every
  image-based score.
- ``images/<oid>/<oid>_prompt.png`` — copy of the chosen view.

Eval code should NEVER look at individual ``<oid>_00N_processed.png`` files
or at GT ``preview.jpg`` — it always routes through ``load_prompt_suite``
so there is exactly one reference image per sample and path errors surface
loudly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class PromptEntry:
    """Resolved prompt image for one sample."""

    oid: str
    view_index: int
    image_path: str
    mask_path: Optional[str]


class PromptSuiteError(RuntimeError):
    """Raised when a prompt suite is missing required fields or files."""


def _resolve(suite_dir: Path, raw: Optional[str]) -> Optional[Path]:
    """Resolve a manifest path against the suite directory.

    Paths are stored relative to the suite root so a suite stays loadable
    after it is downloaded, unpacked, or moved. Absolute paths are accepted
    for backward compatibility with manifests written in place; when such a
    path does not exist, its basename is re-rooted under ``suite_dir`` so a
    relocated suite still resolves.
    """
    if raw is None:
        return None
    p = Path(raw)
    if not p.is_absolute():
        return suite_dir / p
    if p.exists():
        return p
    # Legacy absolute manifest, relocated tree: re-root the tail of the path at the suite directory (…/<suite>/images/<oid>/<oid>_prompt.png).
    parts = p.parts
    for i in range(len(parts) - 1, 0, -1):
        if parts[i] == suite_dir.name:
            candidate = suite_dir.joinpath(*parts[i + 1:])
            if candidate.exists():
                return candidate
            break
    return p


def load_prompt_suite(prompt_suite_dir: str | Path) -> Dict[str, PromptEntry]:
    """Load the prompt-suite manifest and return ``{oid: PromptEntry}``.

    Manifest paths are interpreted relative to ``prompt_suite_dir``; absolute
    paths written by older tooling are still honored (see ``_resolve``).

    Hard errors (``PromptSuiteError``) on:
      - missing ``manifest.json``
      - a manifest without ``prompt_seed`` (not a released suite)
      - any model entry missing ``prompt_view_index`` / ``prompt_image_path``
      - a ``prompt_image_path`` that does not resolve on disk
    """
    suite_dir = Path(prompt_suite_dir).resolve()
    manifest_path = suite_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PromptSuiteError(f"manifest.json not found under {suite_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "prompt_seed" not in manifest:
        raise PromptSuiteError(
            f"{manifest_path}: no 'prompt_seed'. This does not look like a "
            "released ArtiBench suite; re-download it with "
            "the links in the README."
        )

    out: Dict[str, PromptEntry] = {}
    for m in manifest.get("models", []):
        oid = m.get("model_name")
        if oid is None:
            raise PromptSuiteError(f"{manifest_path}: model entry missing 'model_name'")
        for required in ("prompt_view_index", "prompt_image_path"):
            if required not in m:
                raise PromptSuiteError(
                    f"{manifest_path}: model {oid!r} is missing '{required}'. "
                    "The suite manifest is incomplete; re-download the suite."
                )
        image_path = _resolve(suite_dir, m["prompt_image_path"])
        if image_path is None or not image_path.is_file():
            raise PromptSuiteError(
                f"prompt image missing: {m['prompt_image_path']}  (model={oid!r}, "
                f"resolved against {suite_dir})"
            )
        mask_path = _resolve(suite_dir, m.get("prompt_mask_path"))
        if mask_path is not None and not mask_path.is_file():
            # Mask is optional; drop it silently.
            mask_path = None
        out[str(oid)] = PromptEntry(
            oid=str(oid),
            view_index=int(m["prompt_view_index"]),
            image_path=str(image_path),
            mask_path=str(mask_path) if mask_path is not None else None,
        )

    if not out:
        raise PromptSuiteError(f"{manifest_path}: no models found.")
    return out
