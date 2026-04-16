"""HuggingFace weight loader for CUDA Qwen3-VL models.

Maps HF parameter names to our internal layout:
- HF vision:  `visual.patch_embed.proj.weight` -> `visual.patch_embed.weight`
- HF text:    `model.layers.N.*` -> `layers.N.*`
- HF MoE:     `model.layers.N.mlp.experts.gate_up_proj` -> `layers.N.mlp.gate_up_proj`
- HF gate:    `model.layers.N.mlp.gate.weight` -> `layers.N.mlp.gate_weight`
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open


def _strip_prefixes(name: str) -> str:
    for p in ("language_model.", "model.language_model.", "model."):
        if name.startswith(p):
            return name[len(p):]
    return name


def _map_name(name: str) -> str | None:
    """Translate HF parameter name to our internal name. Returns None if unmapped."""
    # Vision tower
    if name.startswith("visual."):
        # HF: visual.patch_embed.proj.{weight,bias} -> visual.patch_embed.{weight,bias}
        name = name.replace("visual.patch_embed.proj.", "visual.patch_embed.")
        # HF: visual.blocks.N.norm1.{weight,bias} etc -> visual.blocks.N.norm1.{weight,bias} (match)
        # HF: visual.blocks.N.attn.qkv.{weight,bias} -> visual.blocks.N.attn.qkv.{weight,bias} (match)
        # HF: visual.blocks.N.attn.proj.{weight,bias} -> visual.blocks.N.attn.proj.{weight,bias} (match)
        # HF: visual.blocks.N.mlp.linear_fc1/fc2.{weight,bias} -> visual.blocks.N.mlp.linear_fc1/fc2.{weight,bias} (match)
        # HF: visual.merger.ln_q.{weight,bias} -> visual.merger.norm.{weight,bias}
        name = name.replace("visual.merger.ln_q.", "visual.merger.norm.")
        # HF: visual.merger.mlp.0.{weight,bias} -> visual.merger.linear_fc1.{weight,bias}
        name = name.replace("visual.merger.mlp.0.", "visual.merger.linear_fc1.")
        name = name.replace("visual.merger.mlp.2.", "visual.merger.linear_fc2.")
        # HF: visual.deepstack_merger_list.N.{norm,linear_fc1,linear_fc2}
        name = name.replace("visual.deepstack_merger_list.", "visual.deepstack_mergers.")
        # HF: visual.pos_embed.weight -> visual.pos_embed.emb.weight
        name = name.replace("visual.pos_embed.weight", "visual.pos_embed.emb.weight")
        return name

    stripped = _strip_prefixes(name)
    # Text stack:
    # HF: layers.N.self_attn.{q,k,v,o}_proj.weight -> layers.N.self_attn.{q,k,v,o}_proj.weight (match)
    # HF: layers.N.self_attn.q_norm.weight -> layers.N.self_attn.q_norm.weight (match)
    # HF: layers.N.mlp.{gate,up,down}_proj.weight -> layers.N.mlp.{gate,up,down}_proj.weight (match)
    # HF MoE: layers.N.mlp.experts.{gate_up_proj,down_proj} -> layers.N.mlp.{gate_up_proj,down_proj}
    stripped = stripped.replace("mlp.experts.gate_up_proj", "mlp.gate_up_proj")
    stripped = stripped.replace("mlp.experts.down_proj", "mlp.down_proj")
    # HF MoE: layers.N.mlp.gate.weight -> layers.N.mlp.gate_weight
    stripped = stripped.replace("mlp.gate.weight", "mlp.gate_weight")
    # HF: embed_tokens.weight -> embed_tokens.weight (match)
    # HF: norm.weight -> norm.weight (match)
    # HF: lm_head.weight -> lm_head.weight (at top level — handled by caller if needed)
    return stripped


def load_hf_weights(model: torch.nn.Module, snapshot_path: str | Path) -> dict:
    """Load safetensors from snapshot_path into model.

    Returns a dict with keys: 'loaded', 'missing', 'unexpected', 'mismatched'.
    """
    snapshot = Path(snapshot_path)
    files = sorted(snapshot.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors in {snapshot_path}")

    state_dict = dict(model.state_dict())
    remaining = set(state_dict.keys())
    loaded: list[str] = []
    unexpected: list[str] = []
    mismatched: list[tuple[str, tuple, tuple]] = []

    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                mapped = _map_name(key)
                if mapped is None or mapped not in state_dict:
                    unexpected.append(key)
                    continue
                tensor = sf.get_tensor(key)
                target = state_dict[mapped]
                if tensor.shape != target.shape:
                    mismatched.append((key, tuple(tensor.shape), tuple(target.shape)))
                    continue
                with torch.no_grad():
                    target.copy_(tensor.to(target.dtype).to(target.device))
                loaded.append(mapped)
                remaining.discard(mapped)

    return {
        "loaded": loaded,
        "missing": sorted(remaining),
        "unexpected": unexpected,
        "mismatched": mismatched,
    }
