from __future__ import annotations

import torch
import torch.nn.functional as F

from ._loader import load_op, maybe_strict_raise


def _ensure() -> bool:
    return load_op("flash_attention")


def _fallback(q, k, v, scale, is_causal, num_kv_groups):
    if num_kv_groups > 1:
        B, nkv, S, D = k.shape
        k = k[:, :, None].expand(B, nkv, num_kv_groups, S, D).reshape(B, nkv * num_kv_groups, S, D)
        v = v[:, :, None].expand(B, nkv, num_kv_groups, S, D).reshape(B, nkv * num_kv_groups, S, D)
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)


class _FlashAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, is_causal, num_kv_groups):
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.num_kv_groups = num_kv_groups
        if not _ensure():
            return _fallback(q, k, v, scale, is_causal, num_kv_groups)
        try:
            outs = torch.ops.cuda_qwen3_vl.flash_attention_forward(
                q.contiguous(), k.contiguous(), v.contiguous(),
                scale, is_causal, num_kv_groups
            )
            return outs[0]
        except Exception as exc:
            maybe_strict_raise("flash_attention", exc)
            return _fallback(q, k, v, scale, is_causal, num_kv_groups)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward kernel not yet implemented — fall back to torch autograd on reference impl
        from ._loader import record_fallback
        record_fallback("flash_attention", "backward_not_implemented_uses_sdpa_autograd")
        q, k, v = ctx.saved_tensors
        with torch.enable_grad():
            q2 = q.detach().requires_grad_(True)
            k2 = k.detach().requires_grad_(True)
            v2 = v.detach().requires_grad_(True)
            o = _fallback(q2, k2, v2, ctx.scale, ctx.is_causal, ctx.num_kv_groups)
            gq, gk, gv = torch.autograd.grad(o, (q2, k2, v2), grad_output)
        return gq, gk, gv, None, None, None


def flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    scale: float, is_causal: bool = True, num_kv_groups: int = 1,
) -> torch.Tensor:
    if not q.is_cuda:
        return _fallback(q, k, v, scale, is_causal, num_kv_groups)
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return _FlashAttnFunction.apply(q, k, v, scale, is_causal, num_kv_groups)
    if not _ensure():
        return _fallback(q, k, v, scale, is_causal, num_kv_groups)
    try:
        outs = torch.ops.cuda_qwen3_vl.flash_attention_forward(
            q.contiguous(), k.contiguous(), v.contiguous(),
            scale, is_causal, num_kv_groups
        )
        return outs[0]
    except Exception as exc:
        maybe_strict_raise("flash_attention", exc)
        return _fallback(q, k, v, scale, is_causal, num_kv_groups)
