"""Qwen3-VL-MoE conditional-generation model (e.g. 30B-A3B)."""
from __future__ import annotations

import torch
from torch import nn

from cuda_qwen3_vl.configs import Qwen3VLConfig, MoETextConfig
from cuda_qwen3_vl.modules import (
    CudaEmbedding,
    CudaLinear,
    CudaRMSNorm,
    CudaTextDecoderLayer,
    TextMRoPE,
)
from cuda_qwen3_vl.models.common import CudaVisionTower


def _layer_uses_moe(cfg: MoETextConfig, layer_idx: int) -> bool:
    if cfg.num_experts <= 0:
        return False
    if layer_idx in cfg.mlp_only_layers:
        return False
    return ((layer_idx + 1) % cfg.decoder_sparse_step) == 0


class CudaQwen3VLMoeModel(nn.Module):
    """MoE Qwen3-VL: vision tower + text decoder stack with MoE MLPs + lm_head."""

    def __init__(self, cfg: Qwen3VLConfig) -> None:
        super().__init__()
        assert cfg.family == "moe", f"expected MoE config, got {cfg.family}"
        assert isinstance(cfg.text, MoETextConfig)
        self.cfg = cfg
        self.visual = CudaVisionTower(cfg.vision)
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
                use_moe=_layer_uses_moe(t, i),
                num_experts=t.num_experts,
                top_k=t.num_experts_per_tok,
                moe_intermediate_size=t.moe_intermediate_size,
                attention_bias=t.attention_bias,
            )
            for i in range(t.num_layers)
        ])
        self.norm = CudaRMSNorm(t.hidden_size, eps=t.rms_norm_eps)
        self.lm_head = CudaLinear(t.hidden_size, t.vocab_size, bias=False)
        if t.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _text_forward(
        self, input_embeds: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = input_embeds
        mrope_fn = lambda q, k: self.rotary.apply(q, k, position_ids)
        router_logits_all: list[torch.Tensor] = []
        for layer in self.layers:
            x, rl = layer(x, mrope_fn)
            if rl is not None:
                router_logits_all.append(rl)
        return self.norm(x), router_logits_all

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        vision_position_ids: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        assert input_ids is not None
        inputs_embeds = self.embed_tokens(input_ids)
        if pixel_values is not None and image_token_mask is not None:
            vision_feats, _deepstack = self.visual(pixel_values, vision_position_ids)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[image_token_mask] = vision_feats.to(inputs_embeds.dtype)
        if position_ids is None:
            B, S = input_ids.shape
            pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
            position_ids = torch.stack([pos, pos, pos], dim=0)
        hidden, router_logits = self._text_forward(inputs_embeds, position_ids)
        return self.lm_head(hidden), router_logits
