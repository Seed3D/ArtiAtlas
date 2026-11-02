# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Helpers for launching Blender-based rendering pipelines in ArtiBench."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional


_THIS_DIR = Path(__file__).resolve().parent
_TEMPLATE_DIR = _THIS_DIR / "static"

def resolve_blender_bin(blender_bin: Optional[str] = None) -> Path:
    """Resolve Blender executable path and mirror it into environment variables.

    Priority:
    1) explicit `blender_bin`
    2) `$ARTIBENCH_BLENDER_BIN`
    3) `$BLENDER_BIN`
    4) `blender` from PATH

    The published benchmark numbers were produced with Blender 4.2.2 LTS.
    Renders from a different build are not guaranteed to be bit-identical,
    which shifts the image-based metrics (CLIP / CLIP-N / VLM judge).
    """

    candidates = []
    if blender_bin:
        candidates.append(Path(blender_bin))

    env_artibench = os.environ.get("ARTIBENCH_BLENDER_BIN")
    if env_artibench:
        candidates.append(Path(env_artibench))

    env_blender = os.environ.get("BLENDER_BIN")
    if env_blender:
        candidates.append(Path(env_blender))

    which_blender = shutil.which("blender")
    if which_blender:
        candidates.append(Path(which_blender))

    for candidate in candidates:
        if candidate and candidate.exists() and os.access(candidate, os.X_OK):
            resolved = candidate.resolve()
            # Keep a dedicated variable for ArtiBench and compatibility with existing tools.
            os.environ["ARTIBENCH_BLENDER_BIN"] = resolved.as_posix()
            os.environ["BLENDER_BIN"] = resolved.as_posix()
            return resolved

    raise FileNotFoundError(
        "Unable to locate Blender. Export $ARTIBENCH_BLENDER_BIN (or "
        "$BLENDER_BIN) pointing at the executable, or put `blender` on PATH. "
        "The published numbers were produced with Blender 4.2.2 LTS."
    )


def _load_template(template_name: str) -> str:
    path = _TEMPLATE_DIR / template_name
    if not path.exists():
        raise FileNotFoundError(f"Blender template not found: {path}")
    return path.read_text(encoding="utf-8")


def _render_script(template_text: str, replacements: Mapping[str, str]) -> str:
    text = template_text
    for key, value in replacements.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def run_blender_template(
    *,
    template_name: str,
    replacements: Mapping[str, str],
    blender_bin: Path,
    work_dir: Path,
    log_name: str,
    script_name: str,
    extra_env: Optional[Dict[str, str]] = None,
) -> tuple[bool, Path]:
    """Run a Blender template and return `(success, log_path)`."""

    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / script_name
    log_path = work_dir / log_name

    template = _load_template(template_name)
    script_text = _render_script(template, replacements)
    script_path.write_text(script_text, encoding="utf-8")

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    # The templates read ARTIBENCH_BLENDER_DEVICE themselves; hiding the GPUs as
    # well keeps Blender from spending its CUDA init before it gets there.
    if os.environ.get("ARTIBENCH_BLENDER_DEVICE", "").upper() == "CPU":
        env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        blender_bin.as_posix(),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        script_path.as_posix(),
    ]

    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=f, env=env, check=False)

    return proc.returncode == 0, log_path


def create_temp_render_dir(prefix: str = "artibench_blender_") -> Path:
    """Create a temporary working directory for Blender scripts/logs."""

    return Path(tempfile.mkdtemp(prefix=prefix))
