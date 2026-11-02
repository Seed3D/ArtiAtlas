# Copyright (c) 2022, Salesforce.com, Inc.
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: BSD-3-Clause
#
# This file has been modified by ByteDance Ltd. and/or its affiliates: it is a minimized,
# inference-only subset of the upstream PointBERT model definition.
#
# The original file was released by salesforce/ULIP under the BSD 3-Clause License.
# Full license text, including the required disclaimer:
# third_party_licenses/ULIP-LICENSE.txt
# Upstream: https://github.com/salesforce/ULIP/blob/main/LICENSE.txt
#
# This modified file is released under the same license.
"""
Minimal ULIP-2 (PointBERT, colored) model for inference only.

Adapted from salesforce/ULIP (CVPR'24).
The ULIP-2 ensembled model uses PointBERT with 6D input (xyz+rgb), 10k points.

Requires: open_clip, timm.
Optional: pointnet2_ops for GPU-accelerated FPS (falls back to CPU iterative FPS).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.layers import DropPath


# ── Point grouping utilities (shared with _uni3d_model)

def _fps_torch(data: torch.Tensor, number: int) -> torch.Tensor:
    """Farthest Point Sampling with pointnet2_ops or CPU fallback."""
    try:
        from pointnet2_ops import pointnet2_utils
        fps_idx = pointnet2_utils.furthest_point_sample(data, number)
        fps_data = pointnet2_utils.gather_operation(
            data.transpose(1, 2).contiguous(), fps_idx
        ).transpose(1, 2).contiguous()
        return fps_data
    except ImportError:
        print("Warning: pointnet2_ops not found, using CPU fallback for FPS (slow for large point clouds).")
        B, N, C = data.shape
        centroids = torch.zeros(B, number, dtype=torch.long, device=data.device)
        distance = torch.ones(B, N, device=data.device) * 1e10
        farthest = torch.randint(0, N, (B,), device=data.device)
        batch_indices = torch.arange(B, device=data.device)
        for i in range(number):
            centroids[:, i] = farthest
            centroid = data[batch_indices, farthest, :].unsqueeze(1)
            dist = torch.sum((data - centroid) ** 2, dim=-1)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, dim=-1)[1]
        batch_idx = torch.arange(B, device=data.device).unsqueeze(1).expand(-1, number)
        return data[batch_idx, centroids]


def _square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def _knn_point(nsample, xyz, new_xyz):
    sqrdists = _square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


# ── PointBERT components

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        """Input: B N C (C >= 3, first 3 are xyz, rest are features like rgb)."""
        batch_size, num_points, C = xyz.shape
        if C > 3:
            data = xyz
            xyz_only = data[:, :, :3].contiguous()
            rgb = data[:, :, 3:]
        else:
            xyz_only = xyz
            rgb = None

        center = _fps_torch(xyz_only, self.num_group)
        idx = _knn_point(self.group_size, xyz_only, center)
        idx_base = torch.arange(0, batch_size, device=xyz_only.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx_flat = idx.view(-1)
        neighborhood = xyz_only.view(batch_size * num_points, -1)[idx_flat, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)

        if rgb is not None:
            neighborhood_rgb = rgb.reshape(batch_size * num_points, -1)[idx_flat, :]
            neighborhood_rgb = neighborhood_rgb.view(
                batch_size, self.num_group, self.group_size, -1
            ).contiguous()
            neighborhood = torch.cat((neighborhood, neighborhood_rgb), dim=-1)

        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel, input_dim=3):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate)
            for i in range(depth)
        ])

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x


class PointTransformerColored(nn.Module):
    """PointBERT encoder for colored (xyz+rgb) point clouds."""

    def __init__(self, trans_dim=384, depth=12, drop_path_rate=0.1,
                 num_heads=6, group_size=32, num_group=512, encoder_dims=256):
        super().__init__()
        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims, input_dim=6)
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim, depth=self.depth,
            drop_path_rate=dpr, num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

    def forward(self, pts):
        """pts: (B, N, 6) xyz+rgb."""
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        group_input_tokens = self.reduce_dim(group_input_tokens)
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        return concat_f


# ── ULIP-2 with OpenCLIP

class ULIP2(nn.Module):
    """ULIP-2 model: PointBERT (colored) + projection to CLIP space."""

    def __init__(self, point_encoder, pc_feat_dims=768, embed_dim=1280):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder
        self.pc_projection = nn.Parameter(torch.empty(pc_feat_dims, embed_dim))
        nn.init.normal_(self.pc_projection, std=embed_dim ** -0.5)

    def encode_pc(self, pc):
        """pc: (B, N, 6) xyz+rgb."""
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed


def create_ulip2():
    """Create ULIP-2 model with default PointBERT-Colored architecture.

    Default config matches the CVPR'24 ensembled model:
    - 10k points, xyz+rgb (6D)
    - PointBERT: trans_dim=384, depth=18, num_heads=6
    - group_size=32, num_group=512, encoder_dims=256
    - pc_feat_dims = trans_dim * 2 = 768 (cls + max pool concat)
    - Project to 1280D (ViT-bigG-14 CLIP space)
    """
    point_encoder = PointTransformerColored(
        trans_dim=384,
        depth=18,
        drop_path_rate=0.1,
        num_heads=6,
        group_size=32,
        num_group=512,
        encoder_dims=256,
    )
    model = ULIP2(
        point_encoder=point_encoder,
        pc_feat_dims=768,   # 384 * 2 (cls token + max pool)
        embed_dim=1280,     # ViT-bigG-14 CLIP embed dim
    )
    return model
