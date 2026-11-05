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

  float mx = 0.0f;
  for (int e = 0; e < block_size; e++) {
    float v = to_float(row[start + e]);
    mx = fmaxf(mx, fabsf(v));
  }
  unsigned s = (mx == 0.0f) ? 0x01u : e4m3_encode(mx / 6.0f);
  float S = e4m3_decode(s);
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
    unsigned lo = e2m1_encode(to_float(row[start + e]) / S);
    unsigned hi = e2m1_encode(to_float(row[start + e + 1]) / S);
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
