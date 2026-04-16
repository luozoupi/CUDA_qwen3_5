#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> flash_attention_forward_cuda(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    double scale, bool is_causal, int64_t num_kv_groups);

TORCH_LIBRARY_FRAGMENT(cuda_qwen3_vl, m) {
    m.def("flash_attention_forward(Tensor q, Tensor k, Tensor v, float scale, bool is_causal, int num_kv_groups) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(cuda_qwen3_vl, CUDA, m) {
    m.impl("flash_attention_forward", &flash_attention_forward_cuda);
}
