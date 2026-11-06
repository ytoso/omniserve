// NVFP4 W4A4 cuBLASLt GEMM (Atom-style), stateful backend.
//
// See README.md in this directory for the format contract. This file provides
// the compiled backend used by NVFP4Linear:
//
//   out[M, OC] = dequant(x) @ dequant(W)^T + bias
//
// Both operands are E2M1 (NVFP4) with per-16 UE4M3 block scales; the GEMM is
// delegated to cuBLASLt so we hit Blackwell FP4 tensor cores without writing
// tcgen05 ourselves.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "gemm_cuda.h"

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t e = (call);                                                     \
    if (e != cudaSuccess) {                                                     \
      throw std::runtime_error(std::string("CUDA error: ") +                    \
                               cudaGetErrorString(e));                          \
    }                                                                           \
  } while (0)

#define CHECK_CUBLASLT(call)                                                    \
  do {                                                                          \
    cublasStatus_t e = (call);                                                  \
    if (e != CUBLAS_STATUS_SUCCESS) {                                           \
      throw std::runtime_error(std::string("cuBLASLt error: ") +                \
                               cublasLtGetStatusString(e));                     \
    }                                                                           \
  } while (0)

// ---------------------------------------------------------------------------
// Device-side E2M1 / E4M3 conversions (match omniserve/utils/nvfp4.py).
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned e2m1_encode(float x) {
  const float bounds[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
  bool sign = x < 0.0f;
  float a = fabsf(x);
  a = fminf(a, 6.0f);
  int idx = 0;
  while (idx < 7 && a > bounds[idx]) idx++;
  return (unsigned)idx | (sign ? 8u : 0u);
}

__device__ __forceinline__ unsigned e4m3_encode(float x) {
  bool sign = x < 0.0f;
  float a = fabsf(x);
  a = fminf(a, 240.0f);
  int e;
  float m = frexpf(a, &e);
  int E = e + 6;
  int frac = lrintf((m * 2.0f - 1.0f) * 8.0f);
  int carry = frac >> 3;
  E += carry;
  frac &= 7;
  if (E < 1) E = 1;
  if (E > 14) E = 14;
  unsigned bits;
  if (a > 0.0f && e <= -6) {
    int sub = lrintf(a * 512.0f);
    if (sub < 0) sub = 0;
    if (sub > 8) sub = 8;
    bits = (sub == 8) ? (1u << 3) : (unsigned)sub;
  } else {
    bits = ((unsigned)E << 3) | (unsigned)frac;
  }
  if (a == 0.0f) bits = 0;
  if (sign) bits |= 0x80u;
  return bits;
}

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

__device__ __forceinline__ float e4m3_decode(unsigned b) {
  int sign = (b >> 7) & 1;
  int exp = (b >> 3) & 0xF;
  int man = b & 7;
  float val;
  if (exp == 0) {
    val = (man / 8.0f) * powf(2.0f, -6.0f);
  } else {
    val = (1.0f + man / 8.0f) * powf(2.0f, (float)(exp - 7));
  }
  return sign ? -val : val;
}

__device__ __forceinline__ float pow2i(int k) {
  return __int_as_float((k + 127) << 23);
}

// Branch-light E4M3 decode (exact, no powf).
__device__ __forceinline__ float e4m3_decode_fast(unsigned b) {
  int sign = (b >> 7) & 1;
  int exp = (b >> 3) & 0xF;
  int man = b & 7;
  float val = (exp == 0) ? (float)man * pow2i(-9)
                         : (float)(8 + man) * pow2i(exp - 10);
  return sign ? -val : val;
}

// E2M1 (1s,2e,1m, bias 1) decoded arithmetically to avoid a local array.
__device__ __forceinline__ float e2m1_decode(unsigned v) {
  unsigned mag = v & 7u;
  unsigned exp = (mag >> 1) & 3u;
  unsigned man = mag & 1u;
  float val = (exp == 0) ? ((float)man * 0.5f)
                         : ((float)(2u + man) * pow2i((int)exp - 2));
  return (v & 8u) ? -val : val;
}

// value of FP4 element `d` given packed bytes and per-16 scales of that row.
// Empirical MSE-optimal E2M1 block scale.
// E2M1 has a single mantissa bit, so the largest magnitudes in a block are
// poorly served by scale = max/6. Searching a few divisors of the block max
// and keeping the one with the smallest squared error is free (3x ALU) and
// typically recovers several percent of relative error.
__device__ __forceinline__ unsigned best_e4m3_scale(const float* x, int n,
                                                    float* out_scale) {
  const float cands[3] = {6.0f, 4.5f, 3.0f};
  float best_err = INFINITY;
  unsigned best_bits = 0x01u;
  float best_s = 0.0f;
  #pragma unroll
  for (int c = 0; c < 3; c++) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) mx = fmaxf(mx, fabsf(x[i]));
    unsigned bits = (mx == 0.0f) ? 0x01u : e4m3_encode(mx / cands[c]);
    float s = e4m3_decode_fast(bits);
    if (s < 1e-12f) s = 1e-12f;
    float err = 0.0f;
    for (int i = 0; i < n; i++) {
      float d = e2m1_decode(e2m1_encode(x[i] / s)) * s - x[i];
      err += d * d;
    }
    if (err < best_err) {
      best_err = err;
      best_bits = bits;
      best_s = s;
    }
  }
  *out_scale = best_s;
  return best_bits;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Fused activation quantization: one thread per (row, 16-element block).
// Computes the UE4M3 scale, writes it directly into cuBLASLt's tiled layout,
// and writes the E2M1 packed values into the transposed A operand.
template <typename T>
__global__ void quantize_act_fused(const T* __restrict__ x, int M, int K,
                                   int block_size, int nblocks,
                                   unsigned char* __restrict__ a_scales,
                                   unsigned char* __restrict__ a_stored) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * nblocks;
  if (tid >= total) return;

  int j = tid / nblocks;
  int b = tid % nblocks;
  const T* row = x + (size_t)j * K;
  int start = b * block_size;

  float vals[16];
  for (int e = 0; e < block_size; e++) vals[e] = to_float(row[start + e]);
  float S = 0.0f;
  unsigned s = best_e4m3_scale(vals, block_size, &S);
  if (S < 1e-12f) S = 1e-12f;

  // Tiled VEC16_UE4M3 scale layout.
  int tiles_inner = (nblocks + 3) / 4;
  int o = j % 128;
  int c = b % 4;
  int off = (o % 32) * 16 + (o / 32) * 4 + c;
  int tiled_idx = ((j / 128) * tiles_inner + (b / 4)) * 512 + off;
  a_scales[tiled_idx] = (unsigned char)s;

  // Transposed/packed A operand: byte (i + j*K)/2, i in [start, start+16).
  int K2 = K / 2;
  unsigned char* dst = a_stored + (size_t)j * K2 + start / 2;
  for (int e = 0; e < block_size; e += 2) {
    unsigned lo = e2m1_encode(vals[e] / S);
    unsigned hi = e2m1_encode(vals[e + 1] / S);
    dst[e / 2] = (unsigned char)(lo | (hi << 4));
  }
}

__global__ void tile_scales(const unsigned char* __restrict__ simple, int outer,
                            int inner, int block_size,
                            unsigned char* __restrict__ tiled) {
  int tiles_inner = (inner + 63) / 64;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = ((outer + 127) / 128) * tiles_inner * 512;
  if (tid >= total) return;

  int off = tid % 512;
  int tmp = tid / 512;
  int ti = tmp % tiles_inner;
  int to = tmp / tiles_inner;

  int o = ((off % 16) / 4) * 32 + (off / 16);
  int c = off % 4;
  int mat_o = to * 128 + o;
  int mat_b = ti * 4 + c;

  unsigned char v = 0;
  if (mat_o < outer && mat_b < inner / block_size) {
    v = simple[(size_t)mat_o * (inner / block_size) + mat_b];
  }
  tiled[tid] = v;
}

__global__ void transpose_weight_bytes(const unsigned char* __restrict__ qw,
                                       int OC, int K2,
                                       unsigned char* __restrict__ b_stored) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (size_t)OC * K2;
  if (tid >= total) return;
  int j = tid / K2;
  int i2 = tid % K2;
  b_stored[(size_t)i2 + (size_t)j * K2] = qw[(size_t)j * K2 + i2];
}

__global__ void output_transpose_f32(const float* __restrict__ dC, int M, int N,
                                     float* __restrict__ out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (size_t)M * N;
  if (tid >= total) return;
  int i = tid / N;
  int j = tid % N;
  out[(size_t)i * N + j] = dC[(size_t)i + (size_t)j * M];
}

__global__ void output_kernel_h(const float* __restrict__ dC, int M, int OC,
                                const __half* __restrict__ bias, int has_bias,
                                __half* __restrict__ out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (size_t)M * OC;
  if (tid >= total) return;
  int i = tid / OC;
  int j = tid % OC;
  float v = dC[(size_t)i + (size_t)j * M];
  if (has_bias) v += __half2float(bias[j]);
  out[(size_t)i * OC + j] = __float2half(v);
}

__global__ void output_kernel_f(const float* __restrict__ dC, int M, int OC,
                                const float* __restrict__ bias, int has_bias,
                                float* __restrict__ out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (size_t)M * OC;
  if (tid >= total) return;
  int i = tid / OC;
  int j = tid % OC;
  float v = dC[(size_t)i + (size_t)j * M];
  if (has_bias) v += bias[j];
  out[(size_t)i * OC + j] = v;
}

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// NVFP4GEMM
// ---------------------------------------------------------------------------

NVFP4GEMM::NVFP4GEMM(at::Tensor qweight, at::Tensor w_scale,
                     c10::optional<at::Tensor> bias) {
  TORCH_CHECK(qweight.is_cuda() && w_scale.is_cuda(),
              "qweight/w_scale must be CUDA tensors");
  TORCH_CHECK(qweight.dim() == 2 && w_scale.dim() == 2, "expected 2D tensors");
  OC_ = qweight.size(0);
  K_ = qweight.size(1) * 2;
  TORCH_CHECK(w_scale.size(0) == OC_ && w_scale.size(1) == K_ / 16,
              "w_scale must be [OC, K/16]");
  out_f32_ = bias.has_value() && bias->scalar_type() == at::kFloat;

  auto opts_u8 = qweight.options().dtype(at::kByte);
  b_stored_ = at::empty({(int64_t)((size_t)OC_ * (K_ / 2))}, opts_u8);
  b_scales_ = at::empty(
      {(int64_t)((size_t)cdiv(OC_, 128) * cdiv(K_, 64) * 512)}, opts_u8);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int total = (size_t)OC_ * (K_ / 2);
  int threads = 256;
  transpose_weight_bytes<<<cdiv(total, threads), threads, 0, stream>>>(
      qweight.data_ptr<unsigned char>(), OC_, K_ / 2,
      b_stored_.data_ptr<unsigned char>());
  total = cdiv(OC_, 128) * cdiv(K_, 64) * 512;
  tile_scales<<<cdiv(total, threads), threads, 0, stream>>>(
      w_scale.data_ptr<unsigned char>(), OC_, K_, 16,
      b_scales_.data_ptr<unsigned char>());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUBLASLT(cublasLtCreate(&lt_));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&Bdesc_, CUDA_R_4F_E2M1, K_, OC_, K_));
  int order_col = CUBLASLT_ORDER_COL;
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      Bdesc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));

  if (bias.has_value()) {
    bias_ = bias->to(out_f32_ ? at::kFloat : at::kHalf).contiguous();
  }
}

NVFP4GEMM::~NVFP4GEMM() {
  for (auto& kv : plans_) {
    cublasLtMatmulDescDestroy(kv.second.opDesc);
    cublasLtMatrixLayoutDestroy(kv.second.Adesc);
    cublasLtMatrixLayoutDestroy(kv.second.Cdesc);
    cublasLtMatrixLayoutDestroy(kv.second.Ddesc);
  }
  if (Bdesc_) cublasLtMatrixLayoutDestroy(Bdesc_);
  if (lt_) cublasLtDestroy(lt_);
}

void NVFP4GEMM::ensure_plan(int M) {
  NVFP4Plan plan;
  int order_col = CUBLASLT_ORDER_COL;
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&plan.Adesc, CUDA_R_4F_E2M1, K_, M, K_));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&plan.Cdesc, CUDA_R_32F, M, OC_, M));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&plan.Ddesc, CUDA_R_32F, M, OC_, M));
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      plan.Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      plan.Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      plan.Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));

  CHECK_CUBLASLT(
      cublasLtMatmulDescCreate(&plan.opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t opA = CUBLAS_OP_T, opB = CUBLAS_OP_N;
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      plan.opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      plan.opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
  cublasLtMatmulMatrixScale_t smode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  void* pB = b_scales_.data_ptr();
  void* pA_dummy = pB;
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      plan.opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pA_dummy,
      sizeof(pA_dummy)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      plan.opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pB, sizeof(pB)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      plan.opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &smode, sizeof(smode)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      plan.opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &smode, sizeof(smode)));

  cublasLtMatmulPreference_t pref = nullptr;
  CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&pref));
  int ret = 0;
  CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
      lt_, plan.opDesc, plan.Adesc, Bdesc_, plan.Cdesc, plan.Ddesc, pref, 1,
      &plan.heur, &ret));
  cublasLtMatmulPreferenceDestroy(pref);

  // Reusable per-M workspace/buffers.
  auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);
  auto opts_f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  plan.a_stored = at::empty({(int64_t)((size_t)M * (K_ / 2))}, opts_u8);
  plan.a_scales = at::zeros(
      {(int64_t)((size_t)cdiv(M, 128) * cdiv(K_, 64) * 512)}, opts_u8);
  plan.dC = at::empty({(int64_t)((size_t)M * OC_)}, opts_f32);
  if (plan.heur.workspaceSize > 0) {
    plan.workspace = at::empty({(int64_t)plan.heur.workspaceSize}, opts_u8);
  }
  plans_[M] = plan;
}

at::Tensor NVFP4GEMM::forward(at::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == K_, "x must be [*, K]");

  int M = x.size(0);
  int M_pad = ((M + 3) / 4) * 4;

  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  ensure_plan(M_pad);
  NVFP4Plan& plan = plans_[M_pad];

  at::Tensor x_c = x.contiguous();
  at::Tensor x_pad;
  if (M_pad != M) {
    x_pad = at::zeros({M_pad, K_}, x_c.options());
    x_pad.narrow(0, 0, M).copy_(x_c);
    x_c = x_pad;
  }

  int threads = 256;
  int total = M_pad * (K_ / 16);
  if (x_c.scalar_type() == at::kHalf) {
    quantize_act_fused<__half><<<cdiv(total, threads), threads, 0, stream>>>(
        reinterpret_cast<const __half*>(x_c.data_ptr<at::Half>()), M_pad, K_, 16,
        K_ / 16, plan.a_scales.data_ptr<unsigned char>(),
        plan.a_stored.data_ptr<unsigned char>());
  } else {
    quantize_act_fused<float><<<cdiv(total, threads), threads, 0, stream>>>(
        x_c.data_ptr<float>(), M_pad, K_, 16, K_ / 16,
        plan.a_scales.data_ptr<unsigned char>(),
        plan.a_stored.data_ptr<unsigned char>());
  }
  CHECK_CUDA(cudaGetLastError());

  void* pA_scales = plan.a_scales.data_ptr();
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      plan.opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pA_scales,
      sizeof(pA_scales)));

  void* ws_ptr = plan.workspace.defined() ? plan.workspace.data_ptr() : nullptr;
  float alpha = 1.0f, beta = 0.0f;
  void* pA = plan.a_stored.data_ptr();
  void* pB = b_stored_.data_ptr();
  void* pC = plan.dC.data_ptr();
  CHECK_CUBLASLT(cublasLtMatmul(
      lt_, plan.opDesc, &alpha, pA, plan.Adesc, pB, Bdesc_, &beta, pC,
      plan.Ddesc, pC, plan.Ddesc, &plan.heur.algo, ws_ptr,
      plan.heur.workspaceSize, stream));

  at::Tensor out = at::empty({M_pad, OC_}, x.options());
  total = (size_t)M_pad * OC_;
  if (out_f32_) {
    output_kernel_f<<<cdiv(total, threads), threads, 0, stream>>>(
        plan.dC.data_ptr<float>(), M_pad, OC_,
        bias_.has_value() ? bias_->data_ptr<float>() : nullptr,
        bias_.has_value() ? 1 : 0, out.data_ptr<float>());
  } else {
    output_kernel_h<<<cdiv(total, threads), threads, 0, stream>>>(
        plan.dC.data_ptr<float>(), M_pad, OC_,
        bias_.has_value() ? reinterpret_cast<const __half*>(bias_->data_ptr<at::Half>())
                          : nullptr,
        bias_.has_value() ? 1 : 0,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()));
  }
  CHECK_CUDA(cudaGetLastError());

  if (M_pad != M) out = out.narrow(0, 0, M);
  return out;
}

// ---------------------------------------------------------------------------
// Dynamic-activation NVFP4 GEMM (used by NVFP4 attention).

struct ActPlan {
  cublasLtMatrixLayout_t Adesc = nullptr;
  cublasLtMatrixLayout_t Bdesc = nullptr;
  cublasLtMatrixLayout_t Cdesc = nullptr;
  cublasLtMatrixLayout_t Ddesc = nullptr;
  cublasLtMatmulDesc_t opDesc = nullptr;
  cublasLtMatmulHeuristicResult_t heur{};
  at::Tensor workspace;
};
static std::unordered_map<std::string, ActPlan> g_act_plans;

// ---------------------------------------------------------------------------
// Computes C[M, N] = x @ y^T, where both x [M, K] and y [N, K] are quantized
// to E2M1 with per-16 UE4M3 block scales on the fly.
// ---------------------------------------------------------------------------

static std::tuple<at::Tensor, at::Tensor> quantize_act_for_gemm(
    at::Tensor a, int outer, int inner, cudaStream_t stream) {
  // a is [outer, inner], fp16 or fp32. Returns [a_stored, a_scales].
  auto opts_u8 = a.options().dtype(at::kByte);
  at::Tensor a_stored =
      at::empty({(int64_t)((size_t)outer * (inner / 2))}, opts_u8);
  at::Tensor a_scales = at::zeros(
      {(int64_t)((size_t)cdiv(outer, 128) * cdiv(inner, 64) * 512)}, opts_u8);
  int threads = 256;
  int total = outer * (inner / 16);
  if (a.scalar_type() == at::kHalf) {
    quantize_act_fused<__half><<<cdiv(total, threads), threads, 0, stream>>>(
        reinterpret_cast<const __half*>(a.data_ptr<at::Half>()), outer, inner,
        16, inner / 16, a_scales.data_ptr<unsigned char>(),
        a_stored.data_ptr<unsigned char>());
  } else {
    quantize_act_fused<float><<<cdiv(total, threads), threads, 0, stream>>>(
        a.data_ptr<float>(), outer, inner, 16, inner / 16,
        a_scales.data_ptr<unsigned char>(), a_stored.data_ptr<unsigned char>());
  }
  CHECK_CUDA(cudaGetLastError());
  return std::make_tuple(a_stored, a_scales);
}

at::Tensor gemm_act_forward_cuda(at::Tensor x, at::Tensor y) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "x/y must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && x.size(1) == y.size(1),
              "x [M,K] and y [N,K] expected");

  int M = x.size(0), N = y.size(0), K = x.size(1);
  int M_pad = ((M + 3) / 4) * 4;
  int N_pad = ((N + 3) / 4) * 4;
  int K_pad = ((K + 15) / 16) * 16;

  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  at::Tensor x_c = x.contiguous();
  if (M_pad != M) {
    at::Tensor p = at::zeros({M_pad, K}, x_c.options());
    p.narrow(0, 0, M).copy_(x_c);
    x_c = p;
  }
  if (K_pad != K) {
    at::Tensor p = at::zeros({x_c.size(0), K_pad}, x_c.options());
    p.narrow(1, 0, K).copy_(x_c);
    x_c = p;
  }
  at::Tensor y_c = y.contiguous();
  if (N_pad != N) {
    at::Tensor p = at::zeros({N_pad, K}, y_c.options());
    p.narrow(0, 0, N).copy_(y_c);
    y_c = p;
  }
  if (K_pad != K) {
    at::Tensor p = at::zeros({y_c.size(0), K_pad}, y_c.options());
    p.narrow(1, 0, K).copy_(y_c);
    y_c = p;
  }

  auto [a_stored, a_scales] = quantize_act_for_gemm(x_c, M_pad, K_pad, stream);
  auto [b_stored, b_scales] = quantize_act_for_gemm(y_c, N_pad, K_pad, stream);

  static cublasLtHandle_t lt = nullptr;
  if (!lt) CHECK_CUBLASLT(cublasLtCreate(&lt));
  std::string key = std::to_string(M_pad) + "_" + std::to_string(N_pad) + "_" + std::to_string(K_pad);
  ActPlan* plan;
  auto it = g_act_plans.find(key);
  if (it == g_act_plans.end()) {
    ActPlan p;
    int order_col = CUBLASLT_ORDER_COL;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&p.Adesc, CUDA_R_4F_E2M1, K_pad, M_pad, K_pad));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&p.Bdesc, CUDA_R_4F_E2M1, K_pad, N_pad, K_pad));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&p.Cdesc, CUDA_R_32F, M_pad, N_pad, M_pad));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&p.Ddesc, CUDA_R_32F, M_pad, N_pad, M_pad));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(p.Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(p.Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(p.Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(p.Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&p.opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opA = CUBLAS_OP_T, opB = CUBLAS_OP_N;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(p.opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(p.opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    cublasLtMatmulMatrixScale_t smode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(p.opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &smode, sizeof(smode)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(p.opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &smode, sizeof(smode)));
    void* pA_s0 = a_scales.data_ptr();
    void* pB_s0 = b_scales.data_ptr();
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(p.opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pA_s0, sizeof(pA_s0)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(p.opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pB_s0, sizeof(pB_s0)));
    cublasLtMatmulPreference_t pref = nullptr;
    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&pref));
    int ret = 0;
    CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(lt, p.opDesc, p.Adesc, p.Bdesc, p.Cdesc, p.Ddesc, pref, 1, &p.heur, &ret));
    cublasLtMatmulPreferenceDestroy(pref);
    if (p.heur.workspaceSize > 0) {
      p.workspace = at::empty({(int64_t)p.heur.workspaceSize}, x.options().dtype(at::kByte));
    }
    g_act_plans[key] = p;
    plan = &g_act_plans[key];
  } else {
    plan = &it->second;
  }
  void* pA_s = a_scales.data_ptr();
  void* pB_s = b_scales.data_ptr();
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(plan->opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pA_s, sizeof(pA_s)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(plan->opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pB_s, sizeof(pB_s)));

  auto opts_f32 = x.options().dtype(at::kFloat);
  at::Tensor dC = at::empty({(int64_t)((size_t)M_pad * N_pad)}, opts_f32);
  void* ws = plan->workspace.defined() ? plan->workspace.data_ptr() : nullptr;
  float alpha = 1.0f, beta = 0.0f;
  void* pA = a_stored.data_ptr();
  void* pB = b_stored.data_ptr();
  void* pC = dC.data_ptr();
  CHECK_CUBLASLT(cublasLtMatmul(lt, plan->opDesc, &alpha, pA, plan->Adesc, pB,
                                plan->Bdesc, &beta, pC, plan->Ddesc, pC,
                                plan->Ddesc, &plan->heur.algo, ws,
                                plan->heur.workspaceSize, stream));

  at::Tensor out = at::empty({M_pad, N_pad}, opts_f32);
  int threads = 256;
  int total = (size_t)M_pad * N_pad;
  output_transpose_f32<<<cdiv(total, threads), threads, 0, stream>>>(
      dC.data_ptr<float>(), M_pad, N_pad, out.data_ptr<float>());
  CHECK_CUDA(cudaGetLastError());

  if (M_pad != M || N_pad != N) out = out.narrow(0, 0, M).narrow(1, 0, N);
  return out;
}

// ---------------------------------------------------------------------------
// Fused NVFP4 decode attention.
//
// KV cache is kept in NVFP4: packed E2M1 nibbles + one UE4M3 scale per 16
// elements. Decode attention is memory bound, so halving/quartering the KV
// bytes read matters far more than tensor-core FLOPs. Everything (scores,
// masked softmax, PV) is fused into one kernel launch per layer.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float deq_fp4(const unsigned char* __restrict__ p,
                                        const unsigned char* __restrict__ s,
                                        int d) {
  unsigned byte = p[d >> 1];
  unsigned nib = (d & 1) ? (byte >> 4) : (byte & 0xFu);
  return e2m1_decode(nib) * e4m3_decode_fast(s[d >> 4]);
}

// ----------------------------------------------------------------------
// KV-cache element fetch, templated on the storage format.
//   FP4: 2 E2M1 per byte, one UE4M3 scale per 16 elements.
//   FP8: 1 E4M3 per byte, one fp32 scale per (token, kv-head) row.
// ----------------------------------------------------------------------
template <bool FP8>
struct KVFetch;

template <>
struct KVFetch<false> {
  __device__ __forceinline__ static float at(const unsigned char* __restrict__ row,
                                             const void* __restrict__ scales,
                                             int d) {
    unsigned byte = row[d >> 1];
    unsigned nib = (d & 1) ? (byte >> 4) : (byte & 0xFu);
    const unsigned char* sc = static_cast<const unsigned char*>(scales);
    return e2m1_decode(nib) * e4m3_decode_fast(sc[d >> 4]);
  }
};

template <>
struct KVFetch<true> {
  __device__ __forceinline__ static float at(const unsigned char* __restrict__ row,
                                             const void* __restrict__ scales,
                                             int d) {
    const float* sc = static_cast<const float*>(scales);
    return e4m3_decode_fast(row[d]) * sc[0];
  }
};


// Quantize one [outer, inner] half/float matrix into (packed, scales).
__global__ void kv_quantize_kernel(const float* __restrict__ x, int outer,
                                   int inner, int nblocks,
                                   unsigned char* __restrict__ packed,
                                   unsigned char* __restrict__ scales) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * nblocks;
  if (tid >= total) return;
  int o = tid / nblocks;
  int b = tid % nblocks;
  const float* row = x + (size_t)o * inner;
  int start = b * 16;

  float vals[16];
  for (int e = 0; e < 16; e++) vals[e] = row[start + e];
  float S = 0.0f;
  unsigned s = best_e4m3_scale(vals, 16, &S);
  if (S < 1e-12f) S = 1e-12f;
  scales[(size_t)o * nblocks + b] = (unsigned char)s;

  unsigned char* dst = packed + (size_t)o * (inner / 2) + start / 2;
  for (int e = 0; e < 16; e += 2) {
    unsigned lo = e2m1_encode(vals[e] / S);
    unsigned hi = e2m1_encode(vals[e + 1] / S);
    dst[e / 2] = (unsigned char)(lo | (hi << 4));
  }
}

__global__ void kv_quantize_kernel_h(const __half* __restrict__ x, int outer,
                                     int inner, int nblocks,
                                     unsigned char* __restrict__ packed,
                                     unsigned char* __restrict__ scales) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * nblocks;
  if (tid >= total) return;
  int o = tid / nblocks;
  int b = tid % nblocks;
  const __half* row = x + (size_t)o * inner;
  int start = b * 16;

  float vals[16];
  for (int e = 0; e < 16; e++) vals[e] = __half2float(row[start + e]);
  float S = 0.0f;
  unsigned s = best_e4m3_scale(vals, 16, &S);
  if (S < 1e-12f) S = 1e-12f;
  scales[(size_t)o * nblocks + b] = (unsigned char)s;

  unsigned char* dst = packed + (size_t)o * (inner / 2) + start / 2;
  for (int e = 0; e < 16; e += 2) {
    unsigned lo = e2m1_encode(vals[e] / S);
    unsigned hi = e2m1_encode(vals[e + 1] / S);
    dst[e / 2] = (unsigned char)(lo | (hi << 4));
  }
}

// Quantize [L, H, D] to E4M3 with one fp32 scale per (token, head) row.
__global__ void kv_quantize_fp8_kernel(const float* __restrict__ x, int rows,
                                       int D, float* __restrict__ scales,
                                       unsigned char* __restrict__ packed) {
  int r = blockIdx.x;
  if (r >= rows) return;
  extern __shared__ float sh[];
  const float* row = x + (size_t)r * D;
  float mx = 0.0f;
  for (int d = threadIdx.x; d < D; d += blockDim.x)
    mx = fmaxf(mx, fabsf(row[d]));
  for (int off = 16; off > 0; off >>= 1)
    mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
  __syncthreads();
  sh[threadIdx.x >> 5] = mx;   // only meaningful for the first warp
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = sh[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); w++) m = fmaxf(m, sh[w]);
    float s = (m == 0.0f) ? 1.0f : (m / 240.0f);
    scales[r] = s;
    sh[0] = s;
  }
  __syncthreads();
  float s = sh[0];
  for (int d = threadIdx.x; d < D; d += blockDim.x)
    packed[(size_t)r * D + d] = (unsigned char)e4m3_encode(row[d] / s);
}

__global__ void kv_quantize_fp8_kernel_h(const __half* __restrict__ x, int rows,
                                         int D, float* __restrict__ scales,
                                         unsigned char* __restrict__ packed) {
  int r = blockIdx.x;
  if (r >= rows) return;
  extern __shared__ float sh[];
  const __half* row = x + (size_t)r * D;
  float mx = 0.0f;
  for (int d = threadIdx.x; d < D; d += blockDim.x)
    mx = fmaxf(mx, fabsf(__half2float(row[d])));
  for (int off = 16; off > 0; off >>= 1)
    mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
  __syncthreads();
  sh[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = sh[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); w++) m = fmaxf(m, sh[w]);
    float s = (m == 0.0f) ? 1.0f : (m / 240.0f);
    scales[r] = s;
    sh[0] = s;
  }
  __syncthreads();
  float s = sh[0];
  for (int d = threadIdx.x; d < D; d += blockDim.x)
    packed[(size_t)r * D + d] = (unsigned char)e4m3_encode(__half2float(row[d]) / s);
}

std::tuple<at::Tensor, at::Tensor> kv_quantize_fp8_cuda(at::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 3, "x must be a [L, H, D] CUDA tensor");
  int L = x.size(0), H = x.size(1), D = x.size(2);
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor xc = x.contiguous();
  auto opts_u8 = xc.options().dtype(at::kByte);
  at::Tensor packed = at::empty({L, H, D}, opts_u8);
  at::Tensor scales = at::empty({L, H}, xc.options().dtype(at::kFloat));
  int threads = 128;
  int smem = 32 * sizeof(float);
  if (xc.scalar_type() == at::kHalf) {
    kv_quantize_fp8_kernel_h<<<L * H, threads, smem, stream>>>(
        reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()), L * H, D,
        scales.data_ptr<float>(), packed.data_ptr<unsigned char>());
  } else {
    at::Tensor xf = xc.to(at::kFloat);
    kv_quantize_fp8_kernel<<<L * H, threads, smem, stream>>>(
        xf.data_ptr<float>(), L * H, D, scales.data_ptr<float>(),
        packed.data_ptr<unsigned char>());
  }
  CHECK_CUDA(cudaGetLastError());
  return std::make_tuple(packed, scales);
}

std::tuple<at::Tensor, at::Tensor> kv_quantize_cuda(at::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(x.dim() == 3, "x must be [L, H, D]");
  int L = x.size(0), H = x.size(1), D = x.size(2);
  TORCH_CHECK(D % 16 == 0, "head_dim must be a multiple of 16");

  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor xc = x.contiguous();
  auto opts_u8 = xc.options().dtype(at::kByte);
  at::Tensor packed = at::empty({L, H, D / 2}, opts_u8);
  at::Tensor scales = at::empty({L, H, D / 16}, opts_u8);

  int total = L * H * (D / 16);
  int threads = 256;
  if (xc.scalar_type() == at::kHalf) {
    kv_quantize_kernel_h<<<cdiv(total, threads), threads, 0, stream>>>(
        reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()), L * H, D,
        D / 16, packed.data_ptr<unsigned char>(), scales.data_ptr<unsigned char>());
  } else {
    at::Tensor xf = xc.to(at::kFloat);
    kv_quantize_kernel<<<cdiv(total, threads), threads, 0, stream>>>(
        xf.data_ptr<float>(), L * H, D, D / 16,
        packed.data_ptr<unsigned char>(), scales.data_ptr<unsigned char>());
  }
  CHECK_CUDA(cudaGetLastError());
  return std::make_tuple(packed, scales);
}

// Vectorized fused decode attention over an FP4 KV cache.
//
// A 4-byte word holds 8 FP4 values, which always live inside a single 16-wide
// scale block, so one uint32 load + one scale byte decode 8 head dims. That
// cuts the load-instruction count by 8x versus byte-at-a-time decoding, which
// is what limits this memory/latency-bound path.
constexpr int kMaxDecodeChunk = 64;  // key-range splits (runtime chosen)

// Stage 1: scores, split over key chunks. grid = (H, nchunks).
// Each warp scores 2 keys at once: the lanes are split into `ndw`-wide groups
// (ndw = D/8 dwords), each group handling one key and consuming 8 FP4 values
// per uint32 load. Q is staged in shared memory.
template <bool FP8>
__global__ void decode_scores_kernel(
    const __half* __restrict__ q, int H, int KH, int Lk, int D,
    const unsigned char* __restrict__ kp, const void* __restrict__ ks,
    float scale, int nchunks, float* __restrict__ prob) {
  extern __shared__ float shm[];
  __half* qsh = reinterpret_cast<__half*>(shm);   // [D] staged query
  const int h = blockIdx.x;
  const int chunk = blockIdx.y;
  const int kh = h / (H / KH);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;

  for (int d = threadIdx.x; d < D; d += blockDim.x) qsh[d] = q[(size_t)h * D + d];
  __syncthreads();

  const size_t kstride = FP8 ? (size_t)KH * D : (size_t)KH * (D / 2);
  const int ndw = D / 8;
  const int groups = 32 / ndw;               // keys per warp
  const int jh = lane / ndw;                 // which key this lane serves
  const int dw = lane % ndw;                 // which dword
  const unsigned gmask = (ndw >= 32) ? 0xffffffffu
                                     : (((1u << ndw) - 1u) << (jh * ndw));
  float* srow = prob + (size_t)h * Lk;

  const int per = (Lk + nchunks - 1) / nchunks;
  int j0 = chunk * per;
  int j1 = min(j0 + per, Lk);

  for (int base = j0 + warp * groups; base < j1; base += nwarps * groups) {
    int j = base + jh;
    float acc = 0.0f;
    if (j < j1) {
      const unsigned char* kb =
          kp + (size_t)j * kstride + (size_t)kh * (FP8 ? D : (D / 2));
      const void* sb = FP8
          ? (const void*)(static_cast<const float*>(ks) + (size_t)j * KH + kh)
          : (const void*)(static_cast<const unsigned char*>(ks) +
                          (size_t)j * (size_t)KH * (D / 16) + (size_t)kh * (D / 16));
      #pragma unroll
      for (int k = 0; k < 8; k++)
        acc += __half2float(qsh[dw * 8 + k]) * KVFetch<FP8>::at(kb, sb, dw * 8 + k);
    }
    for (int off = ndw / 2; off > 0; off >>= 1)
      acc += __shfl_down_sync(gmask, acc, off);
    if (dw == 0 && j < j1) srow[j] = acc * scale;
  }
}

// Stage 1b: masked softmax over the full key range (one block per head).
__global__ void decode_softmax_kernel(float* __restrict__ prob, int H, int Lk) {
  extern __shared__ float shm[];
  float* wred = shm;
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = blockDim.x >> 5;
  float* srow = prob + (size_t)h * Lk;

  float m = -INFINITY;
  for (int j = tid; j < Lk; j += blockDim.x) m = fmaxf(m, srow[j]);
  for (int off = 16; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
  __syncthreads();
  if (lane == 0) wred[warp] = m;
  __syncthreads();
  __shared__ float bmax;
  __shared__ float bsum;
  if (tid == 0) {
    float r = wred[0];
    for (int w = 1; w < nwarps; w++) r = fmaxf(r, wred[w]);
    bmax = r;
  }
  __syncthreads();

  float sum = 0.0f;
  for (int j = tid; j < Lk; j += blockDim.x) {
    float e = __expf(srow[j] - bmax);
    srow[j] = e;
    sum += e;
  }
  for (int off = 16; off > 0; off >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, off);
  __syncthreads();
  if (lane == 0) wred[warp] = sum;
  __syncthreads();
  if (tid == 0) {
    float r = 0.f;
    for (int w = 0; w < nwarps; w++) r += wred[w];
    bsum = r;
  }
  __syncthreads();

  const float inv = 1.0f / bsum;
  for (int j = tid; j < Lk; j += blockDim.x) srow[j] *= inv;
}

// Stage 2: partial PV over a key chunk. 128 threads = 16 dwords x 8 j-streams.
template <bool FP8>
__global__ void decode_pv_kernel(
    const float* __restrict__ prob, int H, int KH, int Lk, int D,
    const unsigned char* __restrict__ vp, const void* __restrict__ vs,
    int nchunks, float* __restrict__ part) {
  const int h = blockIdx.x;
  const int chunk = blockIdx.y;
  const int kh = h / (H / KH);
  const int tid = threadIdx.x;
  const int ndw = D / 8;
  const int dw = tid % ndw;
  const int js = tid / ndw;
  const int nstreams = blockDim.x / ndw;

  const int per = (Lk + nchunks - 1) / nchunks;
  int j0 = chunk * per;
  int j1 = min(j0 + per, Lk);
  const float* prow = prob + (size_t)h * Lk;
  const size_t vstride = FP8 ? (size_t)KH * D : (size_t)KH * (D / 2);
  const size_t sstride = FP8 ? (size_t)KH : (size_t)KH * (D / 16);

  float acc[8];
  #pragma unroll
  for (int k = 0; k < 8; k++) acc[k] = 0.0f;

  for (int j = j0 + js; j < j1; j += nstreams) {
    const unsigned char* vb = vp + (size_t)j * vstride + (size_t)kh * (FP8 ? D : (D / 2));
    const void* sb = FP8
        ? (const void*)(static_cast<const float*>(vs) + (size_t)j * KH + kh)
        : (const void*)(static_cast<const unsigned char*>(vs) +
                        (size_t)j * sstride + (size_t)kh * (D / 16));
    float p = prow[j];
    #pragma unroll
    for (int k = 0; k < 8; k++) acc[k] += p * KVFetch<FP8>::at(vb, sb, dw * 8 + k);
  }

  // combine the j-streams of this dword through shared memory
  extern __shared__ float psh[];
  #pragma unroll
  for (int k = 0; k < 8; k++) psh[js * D + dw * 8 + k] = acc[k];
  __syncthreads();
  if (js == 0) {
    float* dst = part + ((size_t)h * nchunks + chunk) * D;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
      float t = 0.0f;
      for (int s2 = 0; s2 < nstreams; s2++) t += psh[s2 * D + dw * 8 + k];
      dst[dw * 8 + k] = t;
    }
  }
}

__global__ void decode_combine_kernel(const float* __restrict__ part, int H,
                                      int D, int nchunks,
                                      __half* __restrict__ out) {
  int h = blockIdx.x;
  int d = threadIdx.x;
  if (d >= D) return;
  float acc = 0.f;
  for (int c = 0; c < nchunks; c++)
    acc += part[((size_t)h * nchunks + c) * D + d];
  out[(size_t)h * D + d] = __float2half(acc);
}

at::Tensor decode_attention_cuda(at::Tensor q, at::Tensor k_packed,
                                 at::Tensor k_scales, at::Tensor v_packed,
                                 at::Tensor v_scales, double scale) {
  TORCH_CHECK(q.is_cuda() && k_packed.is_cuda(), "tensors must be CUDA");
  int H = q.size(0), D = q.size(1);
  int Lk = k_packed.size(0), KH = k_packed.size(1);
  const bool fp8_shape = k_scales.scalar_type() == at::kFloat;
  TORCH_CHECK(fp8_shape ? (k_packed.size(2) == D)
                        : (k_packed.size(2) * 2 == D),
              "head_dim mismatch");
  TORCH_CHECK(H % KH == 0, "H must be a multiple of KH");
  TORCH_CHECK(D % 8 == 0, "head_dim must be a multiple of 8");
  TORCH_CHECK(Lk > 0, "empty KV cache");
  const bool fp8 = fp8_shape;

  c10::cuda::CUDAGuard guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Enough blocks to fill the GPU: split the key range adaptively.
  int nchunks = (Lk + 127) / 128;
  if (nchunks < 4) nchunks = 4;
  if (nchunks > kMaxDecodeChunk) nchunks = kMaxDecodeChunk;

  at::Tensor out = at::empty({H, D}, q.options());
  at::Tensor prob = at::empty({H, Lk}, q.options().dtype(at::kFloat));
  at::Tensor part = at::empty({H, nchunks, D}, q.options().dtype(at::kFloat));

  dim3 score_grid(H, nchunks);
  const void* ksc = k_scales.data_ptr();
  const void* vsc = v_scales.data_ptr();
  if (fp8) {
    decode_scores_kernel<true><<<score_grid, 128, (size_t)D * sizeof(__half), stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()), H, KH, Lk, D,
        k_packed.data_ptr<unsigned char>(), ksc, (float)scale, nchunks,
        prob.data_ptr<float>());
  } else {
    decode_scores_kernel<false><<<score_grid, 128, (size_t)D * sizeof(__half), stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()), H, KH, Lk, D,
        k_packed.data_ptr<unsigned char>(), ksc, (float)scale, nchunks,
        prob.data_ptr<float>());
  }
  CHECK_CUDA(cudaGetLastError());

  decode_softmax_kernel<<<H, 256, 32 * sizeof(float), stream>>>(
      prob.data_ptr<float>(), H, Lk);
  CHECK_CUDA(cudaGetLastError());

  int pv_threads = 128;
  int pv_smem = (pv_threads / (D / 8)) * D * sizeof(float);
  dim3 pv_grid(H, nchunks);
  if (fp8) {
    decode_pv_kernel<true><<<pv_grid, pv_threads, pv_smem, stream>>>(
        prob.data_ptr<float>(), H, KH, Lk, D,
        v_packed.data_ptr<unsigned char>(), vsc, nchunks, part.data_ptr<float>());
  } else {
    decode_pv_kernel<false><<<pv_grid, pv_threads, pv_smem, stream>>>(
        prob.data_ptr<float>(), H, KH, Lk, D,
        v_packed.data_ptr<unsigned char>(), vsc, nchunks, part.data_ptr<float>());
  }
  CHECK_CUDA(cudaGetLastError());

  decode_combine_kernel<<<H, 128, 0, stream>>>(
      part.data_ptr<float>(), H, D, nchunks,
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()));
  CHECK_CUDA(cudaGetLastError());
  return out;
}
