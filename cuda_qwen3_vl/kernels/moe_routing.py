from __future__ import annotations

import torch

from ._loader import load_op, maybe_strict_raise, record_fallback


def _ensure() -> bool:
    return load_op("moe_routing")


def cuda_topk(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda or not _ensure():
        if x.is_cuda and not _ensure():
            record_fallback("moe_routing.topk", "build_or_load_failed")
        return torch.topk(x, k, dim=-1)
    try:
        vals, idxs = torch.ops.cuda_qwen3_vl.topk_forward(x.contiguous(), k)
        return vals, idxs
    except Exception as exc:
        maybe_strict_raise("moe_routing.topk", exc)
        return torch.topk(x, k, dim=-1)


def cuda_index_add(target: torch.Tensor, source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """In-place: target[index[i]] += source[i]. Returns target."""
    if not target.is_cuda or not _ensure():
        if target.is_cuda and not _ensure():
            record_fallback("moe_routing.index_add", "build_or_load_failed")
        target.index_add_(0, index, source)
        return target
    try:
        return torch.ops.cuda_qwen3_vl.index_add_forward(target, source.contiguous(), index.contiguous())
    except Exception as exc:
        maybe_strict_raise("moe_routing.index_add", exc)
        target.index_add_(0, index, source)
        return target


def cuda_batched_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Batched GEMM: Y[e] = X[e] @ W[e].T. x: (E,M,K), weight: (E,N,K), out: (E,M,N)."""
    if not x.is_cuda or not _ensure():
        if x.is_cuda and not _ensure():
            record_fallback("moe_routing.batched_gemm", "build_or_load_failed")
        return torch.einsum("emk,enk->emn", x, weight)
    try:
        return torch.ops.cuda_qwen3_vl.batched_gemm_forward(x.contiguous(), weight.contiguous())
    except Exception as exc:
        maybe_strict_raise("moe_routing.batched_gemm", exc)
        return torch.einsum("emk,enk->emn", x, weight)
