# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Metric implementations for ArtiBench.

Submodules are imported lazily: ``eval_self`` needs only the SDF
paths, so importing this package must not drag in torch, open_clip or the
CUDA-only SDF backend before they are actually used.
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    # Self-Quality metrics
    "StaticMeshQualityConfig",
    "compute_segmentation_quality",
    # Reconstruction (paired) metrics
    "eval_joint_params",
    "eval_part_geometry",
    # Alignment metrics (paired, requires reference image)
    "AlignmentConfig",
    "compute_all_alignment",
]

_EXPORTS = {
    "StaticMeshQualityConfig": "artibench.metrics.mesh_quality",
    "compute_segmentation_quality": "artibench.metrics.segmentation_quality",
    "eval_joint_params": "artibench.metrics.reconstruction",
    "eval_part_geometry": "artibench.metrics.reconstruction",
    "AlignmentConfig": "artibench.metrics.image_alignment",
    "compute_all_alignment": "artibench.metrics.image_alignment",
}


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .image_alignment import AlignmentConfig, compute_all_alignment
    from .mesh_quality import StaticMeshQualityConfig
    from .reconstruction import eval_joint_params, eval_part_geometry
    from .segmentation_quality import compute_segmentation_quality
