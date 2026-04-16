#include "common.h"
#include <cfloat>

// Flash Attention v2 forward (tiled, online softmax).
// Inputs:
//   Q: (B, H_q, S_q, D)
//   K: (B, H_kv, S_k, D)  — GQA: H_q = H_kv * num_kv_groups
//   V: (B, H_kv, S_k, D)
// Output: (B, H_q, S_q, D)
// scale: softmax scale (1/sqrt(D) typically)
// is_causal: causal mask inside kernel (col <= row)

namespace {
constexpr int kBlockM = 16;
constexpr int kBlockN = 32;

template <typename scalar_t, int BLOCK_D>
__global__ void flash_attn_fwd_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ O,
    float* __restrict__ LSE,  // (B, H_q, S_q) — logsumexp for backward
    int64_t B, int64_t H_q, int64_t H_kv, int64_t S_q, int64_t S_k, int64_t D,
    int64_t num_kv_groups, float scale, bool is_causal) {
    const int pid_m = blockIdx.x;  // Q tile (rows of size kBlockM)
    const int pid_bh = blockIdx.y; // batch * H_q
    const int tid = threadIdx.x;

    const int64_t b = pid_bh / H_q;
    const int64_t hq = pid_bh % H_q;
    const int64_t hkv = hq / num_kv_groups;

    if (b >= B) return;

    const scalar_t* Q_bh = Q + ((b * H_q + hq) * S_q) * D;
    const scalar_t* K_bh = K + ((b * H_kv + hkv) * S_k) * D;
    const scalar_t* V_bh = V + ((b * H_kv + hkv) * S_k) * D;
    scalar_t* O_bh = O + ((b * H_q + hq) * S_q) * D;
    float* LSE_bh = LSE + (b * H_q + hq) * S_q;

    // Shared memory: Q tile (kBlockM x BLOCK_D), K tile (kBlockN x BLOCK_D), V tile (kBlockN x BLOCK_D)
    __shared__ float Qs[kBlockM][BLOCK_D];
    __shared__ float Ks[kBlockN][BLOCK_D];
    __shared__ float Vs[kBlockN][BLOCK_D];
    __shared__ float Ss[kBlockM][kBlockN];  // attention scores

    // Each thread handles one Q row within the Q block
    const int qrow = tid;  // 0..kBlockM-1 (assume blockDim.x == kBlockM)
    const int64_t global_q = pid_m * kBlockM + qrow;
    const bool valid_q = global_q < S_q;

    // Load Q tile
    if (valid_q) {
        for (int d = 0; d < D && d < BLOCK_D; ++d) {
            Qs[qrow][d] = static_cast<float>(Q_bh[global_q * D + d]);
        }
        for (int d = D; d < BLOCK_D; ++d) Qs[qrow][d] = 0.0f;
    } else {
        for (int d = 0; d < BLOCK_D; ++d) Qs[qrow][d] = 0.0f;
    }

    // Running softmax state
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float acc[BLOCK_D];
    #pragma unroll
    for (int d = 0; d < BLOCK_D; ++d) acc[d] = 0.0f;

    const int64_t kv_end = is_causal
        ? min((int64_t)((pid_m + 1) * kBlockM), S_k)
        : S_k;

    for (int64_t kv_start = 0; kv_start < kv_end; kv_start += kBlockN) {
        __syncthreads();
        // Load K, V tiles cooperatively
        const int items = kBlockN * BLOCK_D;
        for (int i = tid; i < items; i += blockDim.x) {
            const int n = i / BLOCK_D;
            const int d = i % BLOCK_D;
            const int64_t global_n = kv_start + n;
            if (global_n < S_k && d < D) {
                Ks[n][d] = static_cast<float>(K_bh[global_n * D + d]);
                Vs[n][d] = static_cast<float>(V_bh[global_n * D + d]);
            } else {
                Ks[n][d] = 0.0f;
                Vs[n][d] = 0.0f;
            }
        }
        __syncthreads();

        if (!valid_q) continue;

        // Compute scores S[qrow, n] = dot(Q[qrow], K[n]) * scale
        for (int n = 0; n < kBlockN; ++n) {
            float s = 0.0f;
            for (int d = 0; d < D && d < BLOCK_D; ++d) s += Qs[qrow][d] * Ks[n][d];
            s *= scale;
            const int64_t global_n = kv_start + n;
            if (global_n >= S_k) s = -FLT_MAX;
            if (is_causal && global_n > global_q) s = -FLT_MAX;
            Ss[qrow][n] = s;
        }

        // Online softmax update
        float m_ij = -FLT_MAX;
        for (int n = 0; n < kBlockN; ++n) m_ij = fmaxf(m_ij, Ss[qrow][n]);
        const float m_new = fmaxf(m_i, m_ij);
        const float alpha = (m_i == -FLT_MAX) ? 0.0f : __expf(m_i - m_new);
        float l_ij = 0.0f;
        for (int n = 0; n < kBlockN; ++n) {
            Ss[qrow][n] = __expf(Ss[qrow][n] - m_new);
            l_ij += Ss[qrow][n];
        }
        l_i = l_i * alpha + l_ij;
        for (int d = 0; d < D && d < BLOCK_D; ++d) acc[d] *= alpha;
        // acc += P @ V for this Q row
        for (int n = 0; n < kBlockN; ++n) {
            const float p = Ss[qrow][n];
            for (int d = 0; d < D && d < BLOCK_D; ++d) {
                acc[d] += p * Vs[n][d];
            }
        }
        m_i = m_new;
    }

    // Write output
    if (valid_q) {
        const float inv_l = 1.0f / l_i;
        for (int d = 0; d < D && d < BLOCK_D; ++d) {
            O_bh[global_q * D + d] = static_cast<scalar_t>(acc[d] * inv_l);
        }
        LSE_bh[global_q] = m_i + logf(l_i);
    }
}
}  // namespace

std::vector<torch::Tensor> flash_attention_forward_cuda(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    double scale, bool is_causal, int64_t num_kv_groups) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    c10::cuda::CUDAGuard guard(q.device());

    const int64_t B = q.size(0);
    const int64_t H_q = q.size(1);
    const int64_t H_kv = k.size(1);
    const int64_t S_q = q.size(2);
    const int64_t S_k = k.size(2);
    const int64_t D = q.size(3);
    TORCH_CHECK(D <= 128, "head dim > 128 not supported (got ", D, ")");

    auto out = torch::empty_like(q);
    auto lse = torch::empty({B, H_q, S_q}, q.options().dtype(torch::kFloat32));

    const dim3 grid(static_cast<unsigned>(CEIL_DIV(S_q, kBlockM)),
                    static_cast<unsigned>(B * H_q));
    const dim3 block(kBlockM);  // one thread per Q row
    const auto stream = at::cuda::getCurrentCUDAStream();

    auto launch = [&](auto block_d_tag) {
        constexpr int BD = decltype(block_d_tag)::value;
        DISPATCH_FLOAT_TYPES(q.scalar_type(), "flash_attn_fwd", [&] {
            flash_attn_fwd_kernel<scalar_t, BD><<<grid, block, 0, stream>>>(
                q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), lse.data_ptr<float>(),
                B, H_q, H_kv, S_q, S_k, D, num_kv_groups,
                static_cast<float>(scale), is_causal);
        });
    };

    if (D <= 32) launch(std::integral_constant<int, 32>{});
    else if (D <= 64) launch(std::integral_constant<int, 64>{});
    else if (D <= 96) launch(std::integral_constant<int, 96>{});
    else launch(std::integral_constant<int, 128>{});

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, lse};
}
