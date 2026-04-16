"""Vision tower shared by dense and MoE Qwen3-VL variants."""
from __future__ import annotations

import torch
from torch import nn

from cuda_qwen3_vl.configs import VisionConfig
from cuda_qwen3_vl.modules import (
    CudaVisionBlock,
    CudaVisionPatchEmbed,
    CudaVisionPatchMerger,
    CudaVisionPositionEmbed,
    Vision2DRoPE,
)


class CudaVisionTower(nn.Module):
    """27-layer Qwen3-VL vision transformer with patch embed + position embed + deepstack mergers."""

    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = CudaVisionPatchEmbed(
            in_channels=cfg.in_channels,
            hidden_size=cfg.hidden_size,
            patch_size=cfg.patch_size,
            temporal_patch_size=cfg.temporal_patch_size,
        )
        self.pos_embed = CudaVisionPositionEmbed(cfg.num_position_embeddings, cfg.hidden_size)
        head_dim = cfg.hidden_size // cfg.num_heads
        self.rotary = Vision2DRoPE(head_dim // 2, theta=cfg.rope_theta)
        self.blocks = nn.ModuleList([
            CudaVisionBlock(cfg.hidden_size, cfg.num_heads, cfg.intermediate_size, eps=cfg.rms_norm_eps)
            for _ in range(cfg.num_layers)
        ])
        self.merger = CudaVisionPatchMerger(
            hidden_size=cfg.hidden_size,
            out_hidden_size=cfg.out_hidden_size,
            spatial_merge_size=cfg.spatial_merge_size,
            use_postshuffle_norm=False,
        )
        # Deepstack mergers (3x at layers 8, 16, 24) — output into text hidden dim
        self.deepstack_mergers = nn.ModuleList([
            CudaVisionPatchMerger(
                hidden_size=cfg.hidden_size,
                out_hidden_size=cfg.out_hidden_size,
                spatial_merge_size=cfg.spatial_merge_size,
                use_postshuffle_norm=True,
            )
            for _ in cfg.deepstack_layers
        ])

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """pixel_values: (N_patches, C, T, H, W). position_ids: (N_patches,).

        Returns (main_features, deepstack_features_list) — all projected to text hidden_size.
        """
        x = self.patch_embed(pixel_values)  # (N, hidden)
        x = x + self.pos_embed(position_ids)
        x = x.unsqueeze(0)  # add batch dim for blocks: (1, N, hidden)

        # Precompute rotary cos/sin for vision RoPE
        seqlen = x.shape[1]
        freqs = self.rotary(seqlen)  # (seqlen, dim//2)
        cos = freqs.cos().unsqueeze(0)  # (1, seqlen, dim//2)
        sin = freqs.sin().unsqueeze(0)

        deepstack_outs: list[torch.Tensor] = []
        deepstack_idx = {layer_idx: i for i, layer_idx in enumerate(self.cfg.deepstack_layers)}

        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin)
            if i in deepstack_idx:
                ds_i = deepstack_idx[i]
                deepstack_outs.append(self.deepstack_mergers[ds_i](x.squeeze(0)))

        main = self.merger(x.squeeze(0))
        return main, deepstack_outs
