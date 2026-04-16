"""Module-level sanity tests: CudaLinear, CudaEmbedding, norms, MLP, attention."""
import pytest
import torch
import torch.nn.functional as F

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda_only
def test_cuda_linear_matches_nn_linear():
    from cuda_qwen3_vl.modules import CudaLinear
    torch.manual_seed(0)
    ref = torch.nn.Linear(64, 32, bias=True).cuda()
    ours = CudaLinear(64, 32, bias=True).cuda()
    ours.weight.data.copy_(ref.weight.data)
    ours.bias.data.copy_(ref.bias.data)
    x = torch.randn(8, 16, 64, device="cuda")
    torch.testing.assert_close(ours(x), ref(x), atol=1e-3, rtol=1e-3)


@cuda_only
def test_cuda_embedding_matches_nn_embedding():
    from cuda_qwen3_vl.modules import CudaEmbedding
    torch.manual_seed(0)
    ref = torch.nn.Embedding(100, 64).cuda()
    ours = CudaEmbedding(100, 64).cuda()
    ours.weight.data.copy_(ref.weight.data)
    ids = torch.randint(0, 100, (4, 16), device="cuda", dtype=torch.int64)
    torch.testing.assert_close(ours(ids), ref(ids), atol=1e-5, rtol=1e-5)


@cuda_only
def test_cuda_rmsnorm():
    from cuda_qwen3_vl.modules import CudaRMSNorm
    torch.manual_seed(0)
    norm = CudaRMSNorm(256, eps=1e-6).cuda()
    x = torch.randn(4, 16, 256, device="cuda")
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    torch.testing.assert_close(norm(x), expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_cuda_layernorm_matches_nn_layernorm():
    from cuda_qwen3_vl.modules import CudaLayerNorm
    torch.manual_seed(0)
    ref = torch.nn.LayerNorm(256, eps=1e-6).cuda()
    ours = CudaLayerNorm(256, eps=1e-6).cuda()
    ours.weight.data.copy_(ref.weight.data)
    ours.bias.data.copy_(ref.bias.data)
    x = torch.randn(4, 16, 256, device="cuda")
    torch.testing.assert_close(ours(x), ref(x), atol=1e-4, rtol=1e-4)


@cuda_only
def test_swiglu_mlp_forward_finite():
    """MLP forward produces finite outputs for a realistic small config."""
    from cuda_qwen3_vl.modules import CudaSwiGLUMLP
    torch.manual_seed(0)
    mlp = CudaSwiGLUMLP(hidden_size=64, intermediate_size=128).cuda()
    x = torch.randn(2, 8, 64, device="cuda")
    out = mlp(x)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


@cuda_only
def test_vision_block_forward_finite():
    from cuda_qwen3_vl.modules import CudaVisionBlock
    torch.manual_seed(0)
    block = CudaVisionBlock(hidden_size=64, num_heads=4, intermediate_size=128).cuda()
    x = torch.randn(1, 32, 64, device="cuda")
    head_dim = 64 // 4
    cos = torch.randn(1, 32, head_dim, device="cuda")
    sin = torch.randn(1, 32, head_dim, device="cuda")
    out = block(x, cos, sin)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@cuda_only
def test_moe_forward_finite():
    """MoE forward produces finite outputs with correct shape."""
    from cuda_qwen3_vl.modules import CudaSparseMoE
    torch.manual_seed(0)
    moe = CudaSparseMoE(
        hidden_size=64, moe_intermediate_size=32,
        num_experts=4, top_k=2, norm_topk_prob=True,
    ).cuda()
    x = torch.randn(2, 8, 64, device="cuda")
    out, router_logits = moe(x)
    assert out.shape == (2, 8, 64)
    assert router_logits.shape == (16, 4)
    assert torch.isfinite(out).all()
