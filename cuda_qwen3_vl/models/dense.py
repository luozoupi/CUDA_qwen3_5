"""Qwen3-VL dense conditional-generation model (e.g. 8B)."""
from __future__ import annotations

import torch
from torch import nn

from cuda_qwen3_vl.configs import Qwen3VLConfig
from cuda_qwen3_vl.modules import (
    CudaEmbedding,
    CudaLinear,
    CudaRMSNorm,
    CudaTextDecoderLayer,
    TextMRoPE,
)
from cuda_qwen3_vl.models.common import CudaVisionTower


class CudaQwen3VLDenseModel(nn.Module):
    """Dense Qwen3-VL: vision tower + text decoder stack + lm_head."""

    def __init__(self, cfg: Qwen3VLConfig) -> None:
        super().__init__()
        assert cfg.family == "dense", f"expected dense config, got {cfg.family}"
        self.cfg = cfg
        # Vision
        self.visual = CudaVisionTower(cfg.vision)
        # Text
        t = cfg.text
        self.embed_tokens = CudaEmbedding(t.vocab_size, t.hidden_size)
        self.rotary = TextMRoPE(t.head_dim, theta=t.rope_theta, mrope_section=t.mrope_section)
        self.layers = nn.ModuleList([
            CudaTextDecoderLayer(
                hidden_size=t.hidden_size,
                num_heads=t.num_heads,
                num_kv_heads=t.num_kv_heads,
                head_dim=t.head_dim,
                intermediate_size=t.intermediate_size,
                rms_norm_eps=t.rms_norm_eps,
                use_moe=False,
                attention_bias=t.attention_bias,
            )
            for _ in range(t.num_layers)
        ])
        self.norm = CudaRMSNorm(t.hidden_size, eps=t.rms_norm_eps)
        self.lm_head = CudaLinear(t.hidden_size, t.vocab_size, bias=False)
        if t.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _text_forward(self, input_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Run decoder stack given text-space embeddings and (3, B, S) position IDs."""
        x = input_embeds
        mrope_fn = lambda q, k: self.rotary.apply(q, k, position_ids)
        for layer in self.layers:
            x, _ = layer(x, mrope_fn)
        return self.norm(x)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        vision_position_ids: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids: (B, S) text tokens. If pixel_values given, vision features replace
        positions marked by image_token_mask.
        position_ids: (3, B, S) 3D positions for MRoPE.
        """
        assert input_ids is not None
        inputs_embeds = self.embed_tokens(input_ids)
        if pixel_values is not None and image_token_mask is not None:
            vision_feats, _deepstack = self.visual(pixel_values, vision_position_ids)
            # vision_feats: (N_tokens, hidden) — scatter into image-token positions.
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[image_token_mask] = vision_feats.to(inputs_embeds.dtype)
        if position_ids is None:
            # Default: same position for all 3 axes
            B, S = input_ids.shape
            pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
            position_ids = torch.stack([pos, pos, pos], dim=0)  # (3, B, S)

        hidden = self._text_forward(inputs_embeds, position_ids)
        return self.lm_head(hidden)
