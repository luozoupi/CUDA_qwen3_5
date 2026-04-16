"""Vision patch embed + positional embedding + merger for Qwen3-VL."""
from __future__ import annotations

import torch
from torch import nn

from cuda_qwen3_vl.kernels import conv3d_patch, gelu_tanh
from cuda_qwen3_vl.modules.embedding import CudaEmbedding
from cuda_qwen3_vl.modules.linear import CudaLinear
from cuda_qwen3_vl.modules.norms import CudaLayerNorm


class CudaVisionPatchEmbed(nn.Module):
    """Qwen3-VL patch embed: Conv3d(3, hidden, [T_patch, H_patch, W_patch]) with bias."""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int, temporal_patch_size: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        kernel_shape = (temporal_patch_size, patch_size, patch_size)
        # Weight layout matches nn.Conv3d: (out_channels, in_channels, T, H, W)
        self.weight = nn.Parameter(torch.empty(hidden_size, in_channels, *kernel_shape))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N_patches, in_channels * T * H * W) or (N_patches, C, T, H, W) — we expect latter
        if x.dim() == 2:
            x = x.reshape(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return conv3d_patch(x, self.weight, self.bias)


class CudaVisionPositionEmbed(nn.Module):
    """Positional embedding table for vision patches."""

    def __init__(self, num_positions: int, hidden_size: int) -> None:
        super().__init__()
        self.emb = CudaEmbedding(num_positions, hidden_size)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(position_ids)


class CudaVisionPatchMerger(nn.Module):
    """Vision projector: LN + Linear + GELU + Linear."""

    def __init__(self, hidden_size: int, out_hidden_size: int, spatial_merge_size: int = 2, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        self.merged_hidden = hidden_size * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.merged_hidden if use_postshuffle_norm else hidden_size
        self.norm = CudaLayerNorm(norm_dim, eps=1e-6, bias=True)
        self.linear_fc1 = CudaLinear(self.merged_hidden, self.merged_hidden, bias=True)
        self.linear_fc2 = CudaLinear(self.merged_hidden, out_hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape(-1, self.merged_hidden))
        else:
            x = self.norm(x).reshape(-1, self.merged_hidden)
        return self.linear_fc2(gelu_tanh(self.linear_fc1(x)))
