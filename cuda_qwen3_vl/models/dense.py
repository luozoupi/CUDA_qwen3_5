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
    """Dense Qwen3-VL. Mirrors HF Qwen3VLForConditionalGeneration's text-model forward
    so we can test our CUDA text stack against the HF reference using HF's processor,
    HF's vision tower, and HF's rope-index computation.
    """

    def __init__(self, cfg: Qwen3VLConfig) -> None:
        super().__init__()
        assert cfg.family == "dense", f"expected dense config, got {cfg.family}"
        self.cfg = cfg
        # Vision tower (optional — end-to-end uses our vision tower; smoke uses HF's)
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

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """hidden_states[visual_pos_masks] += visual_embeds. Matches HF _deepstack_process."""
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :] + visual_embeds
        return hidden_states

    def _text_forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids_3d: torch.Tensor,  # (3, B, S) for MRoPE
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = inputs_embeds
        mrope_fn = lambda q, k: self.rotary.apply(q, k, position_ids_3d)
        n_deepstack = len(deepstack_visual_embeds) if deepstack_visual_embeds is not None else 0
        for i, layer in enumerate(self.layers):
            x, _ = layer(x, mrope_fn)
            if deepstack_visual_embeds is not None and visual_pos_masks is not None and i < n_deepstack:
                x = self._deepstack_process(x, visual_pos_masks, deepstack_visual_embeds[i])
        return self.norm(x)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        input_ids OR inputs_embeds: one of them.
        position_ids: (4, B, S) with rows [text, T, H, W] matching HF, or (3, B, S) with
                      rows [T, H, W], or None (defaults to causal 1D).
        visual_pos_masks: (B, S) bool — which positions are image tokens (for deepstack fusion).
        deepstack_visual_embeds: list of (N_image_tokens, hidden) tensors — one per early layer.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids / inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Normalize position_ids to (3, B, S) for MRoPE
        if position_ids is None:
            B, S, _ = inputs_embeds.shape
            pos = torch.arange(S, device=inputs_embeds.device).unsqueeze(0).expand(B, S)
            position_ids_3d = torch.stack([pos, pos, pos], dim=0)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids_3d = position_ids[1:]  # drop text-only row
        elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
            position_ids_3d = position_ids
        else:
            raise ValueError(f"Unsupported position_ids shape {tuple(position_ids.shape)}")

        hidden = self._text_forward(inputs_embeds, position_ids_3d, visual_pos_masks, deepstack_visual_embeds)
        return self.lm_head(hidden)
