# Copyright (c) 2023 BAAI-Vision
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates: it is a minimized,
# inference-only subset of the upstream model definition.
#
# The original file was released by baaivision/Uni3D under the MIT License.
# Full license text: third_party_licenses/Uni3D-LICENSE.txt
# Upstream: https://github.com/baaivision/Uni3D/blob/main/LICENSE
#
# This modified file is released under the same license.
"""
Minimal Uni3D model for inference only.

Adapted from baaivision/Uni3D (ICLR'24 Spotlight).
Requires: timm, pointnet2_ops (for FPS).

Install pointnet2_ops:
    pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import timm


# ── Point grouping utilities

def _fps_torch(data: torch.Tensor, number: int) -> torch.Tensor:
    """Farthest Point Sampling using pointnet2_ops (CUDA) with CPU fallback."""
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


class PatchDropout(nn.Module):
    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        self.prob = prob
        self.exclude_first_token = exclude_first_token

    def forward(self, x):
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = x[:, :1]

        batch = x.size()[0]
        num_tokens = x.size()[1]
        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))
        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices
        batch_indices = torch.arange(batch)[..., None]
        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)
        return x


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        batch_size, num_points, _ = xyz.shape
        center = _fps_torch(xyz, self.num_group)
        idx = _knn_point(self.group_size, xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
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
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 6)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class PointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, args):
        super().__init__()
        self.trans_dim = args.pc_feat_dim
        self.embed_dim = args.embed_dim
        self.group_size = args.group_size
        self.num_group = args.num_group
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder_dim = args.pc_encoder_dim
        self.encoder = Encoder(encoder_channel=self.encoder_dim)
        self.encoder2trans = nn.Linear(self.encoder_dim, self.trans_dim)
        self.trans2embed = nn.Linear(self.trans_dim, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        self.patch_dropout = (
            PatchDropout(args.patch_dropout) if args.patch_dropout > 0.0 else nn.Identity()
        )
        self.visual = point_transformer

    def forward(self, pts, colors):
        _, center, features = self.group_divider(pts, colors)
        group_input_tokens = self.encoder(features)
        group_input_tokens = self.encoder2trans(group_input_tokens)
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = x + pos
        x = self.patch_dropout(x)
        x = self.visual.pos_drop(x)
        for blk in self.visual.blocks:
            x = blk(x)
        x = self.visual.norm(x[:, 0, :])
        x = self.visual.fc_norm(x)
        x = self.trans2embed(x)
        return x


class Uni3D(nn.Module):
    def __init__(self, point_encoder):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    def encode_pc(self, pc):
        xyz = pc[:, :, :3].contiguous()
        color = pc[:, :, 3:].contiguous()
        pc_feat = self.point_encoder(xyz, color)
        return pc_feat


def create_uni3d(args):
    point_transformer = timm.create_model(
        args.pc_model,
        checkpoint_path=args.pretrained_pc if args.pretrained_pc else "",
        drop_path_rate=args.drop_path_rate,
    )
    point_encoder = PointcloudEncoder(point_transformer, args)
    model = Uni3D(point_encoder=point_encoder)
    return model
