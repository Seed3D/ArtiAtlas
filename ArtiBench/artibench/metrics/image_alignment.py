# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
artibench.metrics.image_alignment
==================================
Alignment metrics measuring how well a generated 3D object matches its
conditioning input image.

Metrics:
    - **CLIP**:   Cosine similarity between CLIP embeddings of rendered views
                  and the reference image.
    - **CLIP-N**: Same as CLIP but on rendered normal maps.
    - **ULIP-2**: Cosine similarity between ULIP-2 point cloud embedding
                  and CLIP image embedding of the reference image.
    - **Uni3D**:  Cosine similarity between Uni3D point cloud embedding
                  and CLIP image embedding of the reference image.

References:
    - TRELLIS 2.0 Table 2 (Alignment metrics)
    - Uni3D: baaivision/Uni3D (ICLR'24 Spotlight)
    - ULIP-2: salesforce/ULIP (CVPR'24)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from artibench.utils.timing import phase

logger = logging.getLogger(__name__)

# Model cache and loading

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "artibench" / "models"


def _open_image_rgb(
    path: str, bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Open an image and convert to RGB, compositing alpha onto *bg_color*.

    Args:
        path: Image file path.
        bg_color: Background fill for transparent pixels.
                  White (255,255,255) for RGB renders / reference images;
                  Black (0,0,0) for normal maps.
    """
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, bg_color)
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def _normalize_pc_xyz(pc_6d: np.ndarray) -> np.ndarray:
    """Rotate point cloud to canonical orientation, then normalize to unit sphere.

    ArtiBench assembles point clouds in URDF world space, which has a fixed
    root RPY = (π/2, 0, -π/2) baked into the forward kinematics.  ULIP-2 and
    Uni3D were trained on Objaverse / ShapeNet data in Y-up canonical pose, so
    we undo the root RPY before feeding points to the encoders.

    The rotation is: R_root = Rz(-π/2) @ Rx(π/2), canonical = inv(R_root) @ xyz.

    After rotation, xyz is centered and scaled to the unit sphere, matching the
    official ``pc_norm`` / ``normalize_pc`` used during training.
    RGB channels (columns 3+) are left unchanged.
    """
    # inv(R_root) where R_root = Rz(-π/2) @ Rx(π/2), i.e. RPY = (π/2, 0, -π/2)
    _INV_ROOT_RPY = np.array(
        [[-0., -1., -0.],
         [ 0.,  0.,  1.],
         [-1., -0., -0.]],
        dtype=np.float64,
    )

    pc = pc_6d.copy()
    xyz = pc[:, :3]
    # Undo URDF root RPY to recover Y-up canonical orientation
    xyz = (_INV_ROOT_RPY @ xyz.T).T
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    max_norm = np.linalg.norm(xyz, axis=1).max()
    if max_norm > 1e-6:
        xyz = xyz / max_norm
    pc[:, :3] = xyz
    return pc


# Singleton model caches
_clip_model_cache: dict = {}
_uni3d_model_cache: dict = {}
_ulip2_model_cache: dict = {}


@dataclass
class AlignmentConfig:
    """Configuration for alignment metric computation."""
    clip_model_name: str = "ViT-bigG-14"
    clip_pretrained: str = "laion2b_s39b_b160k"
    uni3d_ckpt: str = "BAAI/Uni3D"
    uni3d_scale: str = "giant"  # giant | large | base
    ulip2_ckpt: str = "SFXX/ulip"
    n_pcl_points: int = 10000
    cache_dir: str = ""
    device: str = ""

    def __post_init__(self):
        if not self.cache_dir:
            self.cache_dir = str(_DEFAULT_CACHE_DIR)
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# CLIP model loading and inference

def _load_clip_model(config: AlignmentConfig):
    """Load OpenCLIP model and preprocessing transform.

    Uses the default HuggingFace cache. Pre-download models before running:
        huggingface-cli download \\
            laion/CLIP-ViT-bigG-14-laion2B-39B-b160k open_clip_model.safetensors
    """
    key = (config.clip_model_name, config.clip_pretrained)
    if key in _clip_model_cache:
        return _clip_model_cache[key]

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.clip_model_name,
        pretrained=config.clip_pretrained,
    )
    model = model.eval().to(config.device)

    _clip_model_cache[key] = (model, preprocess)
    logger.info("Loaded CLIP model: %s / %s", config.clip_model_name, config.clip_pretrained)
    return model, preprocess


@torch.no_grad()
def _clip_encode_images(
    model, preprocess, image_paths: List[str], device: str,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> torch.Tensor:
    """Encode images into normalized CLIP embeddings.

    Returns:
        (N, D) tensor of L2-normalized embeddings.
    """
    images = []
    for p in image_paths:
        img = _open_image_rgb(p, bg_color=bg_color)
        images.append(preprocess(img))
    batch = torch.stack(images).to(device)
    features = model.encode_image(batch)
    features = F.normalize(features, dim=-1)
    return features


def compute_clip_score(
    ref_image_path: str,
    rendered_view_paths: List[str],
    config: AlignmentConfig,
) -> float:
    """Compute CLIP alignment score: avg cosine sim between ref and rendered views.

    Args:
        ref_image_path: Path to the conditioning/reference image.
        rendered_view_paths: Paths to rendered RGB views of the 3D output.
        config: Alignment configuration.

    Returns:
        Mean cosine similarity in [0, 1].
    """
    model, preprocess = _load_clip_model(config)

    ref_feat = _clip_encode_images(model, preprocess, [ref_image_path], config.device)
    view_feats = _clip_encode_images(model, preprocess, rendered_view_paths, config.device)

    # Cosine similarities: (1, D) @ (D, N) -> (1, N)
    sims = (ref_feat @ view_feats.T).squeeze(0)
    return float(sims.mean().cpu())


def compute_clip_n_score(
    ref_image_path: str,
    normal_map_paths: List[str],
    config: AlignmentConfig,
) -> float:
    """Compute CLIP-N score: avg cosine sim between ref image and rendered normal maps.

    Args:
        ref_image_path: Path to the conditioning/reference image.
        normal_map_paths: Paths to rendered normal map views.
        config: Alignment configuration.

    Returns:
        Mean cosine similarity in [0, 1].
    """
    model, preprocess = _load_clip_model(config)

    ref_feat = _clip_encode_images(model, preprocess, [ref_image_path], config.device)
    # Normal maps use black background — standard convention for normal visualization and avoids washing out the light-blue camera-facing normals.
    normal_feats = _clip_encode_images(
        model, preprocess, normal_map_paths, config.device,
        bg_color=(0, 0, 0),
    )

    sims = (ref_feat @ normal_feats.T).squeeze(0)
    return float(sims.mean().cpu())


# Uni3D model loading and inference

_UNI3D_SCALE_CONFIGS = {
    "giant": {
        "pc_model": "eva_giant_patch14_560",
        "pc_feat_dim": 1408,
    },
    "large": {
        "pc_model": "eva02_large_patch14_448",
        "pc_feat_dim": 1024,
    },
    "base": {
        "pc_model": "eva02_base_patch14_448",
        "pc_feat_dim": 768,
    },
}


def _download_uni3d_checkpoint(config: AlignmentConfig) -> str:
    """Download Uni3D checkpoint from HuggingFace."""
    from huggingface_hub import hf_hub_download

    scale = config.uni3d_scale
    # The model zoo paths on HF: BAAI/Uni3D -> modelzoo/uni3d-{g,l,b}/model.pt
    scale_map = {"giant": "g", "large": "l", "base": "b"}
    short = scale_map.get(scale, "g")
    filename = f"modelzoo/uni3d-{short}/model.pt"

    cache_dir = os.path.join(config.cache_dir, "uni3d")
    path = hf_hub_download(
        repo_id=config.uni3d_ckpt,
        filename=filename,
        cache_dir=cache_dir,
    )
    return path


def _load_uni3d_model(config: AlignmentConfig):
    """Load Uni3D model and the paired CLIP model for image encoding."""
    key = config.uni3d_scale
    if key in _uni3d_model_cache:
        return _uni3d_model_cache[key]

    import open_clip
    from easydict import EasyDict

    scale_cfg = _UNI3D_SCALE_CONFIGS[config.uni3d_scale]

    # Import Uni3D components — we inline the minimal code to avoid requiring the Uni3D repo as a dependency.
    # The model needs: pointnet2_ops for FPS. If not available, fall back.
    try:
        from artibench.metrics._uni3d_model import create_uni3d, Uni3D
    except ImportError:
        raise ImportError(
            "Uni3D model code not found. Please ensure "
            "artibench/metrics/_uni3d_model.py exists and pointnet2_ops is installed.\n"
            "Install: pip install 'git+https://github.com/erikwijmans/Pointnet2_PyTorch.git"
            "#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib'"
        )

    ckpt_path = _download_uni3d_checkpoint(config)

    # Build model
    args = EasyDict(
        pc_model=scale_cfg["pc_model"],
        pc_feat_dim=scale_cfg["pc_feat_dim"],
        embed_dim=1024,
        group_size=64,
        num_group=512,
        pc_encoder_dim=512,
        patch_dropout=0.0,
        pretrained_pc="",
        drop_path_rate=0.0,
    )
    uni3d_model = create_uni3d(args)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "module" in checkpoint:
        state_dict = checkpoint["module"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v
    uni3d_model.load_state_dict(cleaned, strict=False)
    uni3d_model = uni3d_model.eval().to(config.device)

    # Load paired CLIP model for image encoding
    clip_model_name = "EVA02-E-14-plus"
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained="laion2b_s9b_b144k",
    )
    clip_model = clip_model.eval().to(config.device)

    result = (uni3d_model, clip_model, clip_preprocess)
    _uni3d_model_cache[key] = result
    logger.info("Loaded Uni3D model (scale=%s) from %s", config.uni3d_scale, ckpt_path)
    return result


@torch.no_grad()
def compute_uni3d_score(
    ref_image_path: str,
    point_cloud_6d: np.ndarray,
    config: AlignmentConfig,
) -> float:
    """Compute Uni3D alignment score.

    Args:
        ref_image_path: Path to the reference/conditioning image.
        point_cloud_6d: (N, 6) array [x, y, z, r, g, b], colors in [0, 1].
        config: Alignment configuration.

    Returns:
        Cosine similarity between the point cloud and image embeddings.
    """
    uni3d_model, clip_model, clip_preprocess = _load_uni3d_model(config)

    # Encode reference image via CLIP
    img = _open_image_rgb(ref_image_path)
    img_tensor = clip_preprocess(img).unsqueeze(0).to(config.device)
    image_embed = clip_model.encode_image(img_tensor)
    image_embed = F.normalize(image_embed, dim=-1)

    # Normalize xyz to unit sphere (matches official Uni3D preprocessing)
    pc_normed = _normalize_pc_xyz(point_cloud_6d)
    pc = torch.from_numpy(pc_normed).float().unsqueeze(0).to(config.device)
    pc_embed = uni3d_model.encode_pc(pc)
    pc_embed = F.normalize(pc_embed, dim=-1)

    sim = (pc_embed @ image_embed.T).squeeze()
    return float(sim.cpu())


# ULIP-2 model loading and inference

def _download_ulip2_checkpoint(config: AlignmentConfig) -> str:
    """Download ULIP-2 PointBERT checkpoint from HuggingFace."""
    from huggingface_hub import hf_hub_download

    cache_dir = os.path.join(config.cache_dir, "ulip2")
    # ULIP-2 CVPR2024 ensembled model (10k xyzrgb, ViT-G)
    path = hf_hub_download(
        repo_id=config.ulip2_ckpt,
        filename="ULIP-2/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return path


def _load_ulip2_model(config: AlignmentConfig):
    """Load ULIP-2 (PointBERT) model and CLIP for image encoding.

    ULIP-2 (10k xyzrgb, ViT-G variant) uses ViT-bigG-14 as the image encoder.
    We reuse the same CLIP model loaded for CLIP score when possible.
    """
    if "ulip2" in _ulip2_model_cache:
        return _ulip2_model_cache["ulip2"]

    try:
        from artibench.metrics._ulip2_model import create_ulip2
    except ImportError:
        raise ImportError(
            "ULIP-2 model code not found. Please ensure "
            "artibench/metrics/_ulip2_model.py exists."
        )

    ckpt_path = _download_ulip2_checkpoint(config)

    ulip2_model = create_ulip2()

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "module" in checkpoint:
        state_dict = checkpoint["module"]
    else:
        state_dict = checkpoint

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v
    ulip2_model.load_state_dict(cleaned, strict=False)
    ulip2_model = ulip2_model.eval().to(config.device)

    # ULIP-2 (ViT-G variant) uses ViT-bigG-14 for image encoding. Reuse the main CLIP model if it matches,
    # otherwise load separately.
    clip_key = (config.clip_model_name, config.clip_pretrained)
    if clip_key in _clip_model_cache and config.clip_model_name == "ViT-bigG-14":
        clip_model, clip_preprocess = _clip_model_cache[clip_key]
    else:
        # Load the CLIP model using the same robust loader
        _ulip_clip_config = AlignmentConfig(
            clip_model_name="ViT-bigG-14",
            clip_pretrained="laion2b_s39b_b160k",
            cache_dir=config.cache_dir,
            device=config.device,
        )
        clip_model, clip_preprocess = _load_clip_model(_ulip_clip_config)

    result = (ulip2_model, clip_model, clip_preprocess)
    _ulip2_model_cache["ulip2"] = result
    logger.info("Loaded ULIP-2 model from %s", ckpt_path)
    return result


@torch.no_grad()
def compute_ulip2_score(
    ref_image_path: str,
    point_cloud_6d: np.ndarray,
    config: AlignmentConfig,
) -> float:
    """Compute ULIP-2 alignment score.

    Args:
        ref_image_path: Path to the reference/conditioning image.
        point_cloud_6d: (N, 6) array [x, y, z, r, g, b], colors in [0, 1].
        config: Alignment configuration.

    Returns:
        Cosine similarity between the point cloud and image embeddings.
    """
    ulip2_model, clip_model, clip_preprocess = _load_ulip2_model(config)

    # Encode image
    img = _open_image_rgb(ref_image_path)
    img_tensor = clip_preprocess(img).unsqueeze(0).to(config.device)
    image_embed = clip_model.encode_image(img_tensor)
    image_embed = F.normalize(image_embed, dim=-1)

    # Normalize xyz to unit sphere (matches official ULIP-2 preprocessing)
    pc_normed = _normalize_pc_xyz(point_cloud_6d)
    pc = torch.from_numpy(pc_normed).float().unsqueeze(0).to(config.device)
    pc_embed = ulip2_model.encode_pc(pc)
    pc_embed = F.normalize(pc_embed, dim=-1)

    sim = (pc_embed @ image_embed.T).squeeze()
    return float(sim.cpu())


# Unified interface

def prewarm_alignment_models(
    config: AlignmentConfig,
    *,
    skip_clip: bool = False,
    skip_clip_n: bool = False,
    skip_ulip2: bool = False,
    skip_uni3d: bool = False,
) -> None:
    """Eagerly load CLIP / ULIP-2 / Uni3D weights into the module-level caches.

    Called once at evaluator startup to avoid paying the ~120s first-sample
    model-load cost inside a timed phase. Safe to call more than once; the
    underlying loaders are idempotent.
    """
    # CLIP is needed for CLIP, CLIP-N, and ULIP-2 (shares the same ViT-bigG-14).
    if not (skip_clip and skip_clip_n and skip_ulip2):
        try:
            _load_clip_model(config)
        except Exception as e:
            logger.warning("Pre-warm CLIP failed: %s", e)
    if not skip_ulip2:
        try:
            _load_ulip2_model(config)
        except Exception as e:
            logger.warning("Pre-warm ULIP-2 failed: %s", e)
    if not skip_uni3d:
        try:
            _load_uni3d_model(config)
        except Exception as e:
            logger.warning("Pre-warm Uni3D failed: %s", e)


def compute_all_alignment(
    ref_image_path: str,
    rendered_view_paths: List[str],
    normal_map_paths: List[str],
    point_cloud_6d: np.ndarray,
    config: AlignmentConfig,
    skip_clip: bool = False,
    skip_clip_n: bool = False,
    skip_ulip2: bool = False,
    skip_uni3d: bool = False,
) -> Dict[str, Optional[float]]:
    """Compute all alignment metrics.

    Args:
        ref_image_path: Path to the conditioning/reference image.
        rendered_view_paths: Paths to rendered RGB views.
        normal_map_paths: Paths to rendered normal map views.
        point_cloud_6d: (N, 6) point cloud with xyz+rgb.
        config: Alignment configuration.
        skip_*: Flags to skip individual metrics.

    Returns:
        Dict with keys: clip_score, clip_n_score, ulip2_score, uni3d_score.
        Skipped metrics are None.
    """
    results: Dict[str, Optional[float]] = {
        "clip_score": None,
        "clip_n_score": None,
        "ulip2_score": None,
        "uni3d_score": None,
    }

    if not skip_clip and rendered_view_paths:
        try:
            with phase("align.clip"):
                results["clip_score"] = compute_clip_score(
                    ref_image_path, rendered_view_paths, config,
                )
        except Exception as e:
            logger.warning("CLIP score failed: %s", e)

    if not skip_clip_n and normal_map_paths:
        try:
            with phase("align.clip_n"):
                results["clip_n_score"] = compute_clip_n_score(
                    ref_image_path, normal_map_paths, config,
                )
        except Exception as e:
            logger.warning("CLIP-N score failed: %s", e)

    if not skip_ulip2 and point_cloud_6d is not None:
        try:
            with phase("align.ulip2"):
                results["ulip2_score"] = compute_ulip2_score(
                    ref_image_path, point_cloud_6d, config,
                )
        except Exception as e:
            logger.warning("ULIP-2 score failed: %s", e)

    if not skip_uni3d and point_cloud_6d is not None:
        try:
            with phase("align.uni3d"):
                results["uni3d_score"] = compute_uni3d_score(
                    ref_image_path, point_cloud_6d, config,
                )
        except Exception as e:
            logger.warning("Uni3D score failed: %s", e)

    return results
