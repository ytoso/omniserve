// NVFP4 W4A4 cuBLASLt GEMM (Atom-style).
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
#include <vector>

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

// Compute simple activation block scales: out[j * nblocks + b] =
// UE4M3(max|block| / 6), for x [M, K] row-major.
__global__ void compute_act_scales(const float* __restrict__ x, int M, int K,
                                   int block_size, int nblocks,
                                   unsigned char* __restrict__ scales) {
  int j = blockIdx.x;
  int b = blockIdx.y;
  if (j >= M || b >= nblocks) return;
  const float* row = x + (size_t)j * K;
  int start = b * block_size;
  float mx = 0.0f;
  for (int e = 0; e < block_size; e++) {
    float v = fabsf(row[start + e]);
    mx = fmaxf(mx, v);
  }
  scales[(size_t)j * nblocks + b] = (unsigned char)e4m3_encode(mx / 6.0f);
}

// Convert simple [outer, inner/16] UE4M3 scales into cuBLASLt's tiled
// VEC16_UE4M3 layout (one 128x4 tile per 128x64 block).
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
  int mat_b = ti * 4 + c;  // inner block index (each block = block_size elements)

  unsigned char v = 0;
  if (mat_o < outer && mat_b < inner / block_size) {
    v = simple[(size_t)mat_o * (inner / block_size) + mat_b];
  }
  tiled[tid] = v;
}

// Quantize activations x [M, K] (row-major) and write A in cuBLASLt layout:
// A stored [K, M] column-major, packed 2 E2M1 per byte along K (low = first).
__global__ void quantize_act_stored(const float* __restrict__ x, int M, int K,
                                    int block_size,
                                    const unsigned char* __restrict__ scales,
                                    unsigned char* __restrict__ a_stored) {
  int K2 = K / 2;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (size_t)M * K2;
  if (tid >= total) return;

  int j = tid / K2;   // M index
  int i2 = tid % K2;  // packed K index
  int i = i2 * 2;
  int b0 = i / block_size;
  int b1 = (i + 1) / block_size;
  // We recompute the scale from the source row (cheap 16-element reduction)
  // so quantization here exactly matches the UE4M3-rounded scale used by
  // cuBLASLt and by the Python reference path.
  const float* row = x + (size_t)j * K;
  float mx0 = 0.0f, mx1 = 0.0f;
  for (int e = 0; e < block_size; e++) {
    mx0 = fmaxf(mx0, fabsf(row[b0 * block_size + e]));
    mx1 = fmaxf(mx1, fabsf(row[b1 * block_size + e]));
  }
  float S0f = e4m3_decode(e4m3_encode(mx0 / 6.0f));
  float S1f = e4m3_decode(e4m3_encode(mx1 / 6.0f));
  if (S0f < 1e-12f) S0f = 1e-12f;
  if (S1f < 1e-12f) S1f = 1e-12f;
  unsigned lo = e2m1_encode(row[i] / S0f);
  unsigned hi = e2m1_encode(row[i + 1] / S1f);
  // A stored column-major: byte index = (i + j*K)/2 = i2 + j*K2
  a_stored[(size_t)j * K2 + i2] = (unsigned char)(lo | (hi << 4));
}

// Byte transpose qweight [OC, K/2] -> b_stored [K/2, OC] column-major.
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

// Convert cuBLASLt column-major FP32 output [M, OC] into row-major half/float
// output and optionally add bias.
__global__ void output_kernel(const float* __restrict__ dC, int M, int OC,
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

__global__ void output_kernel_f32(const float* __restrict__ dC, int M, int OC,
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

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

at::Tensor gemm_forward_cuda(at::Tensor x, at::Tensor qweight,
                             at::Tensor w_scale,
                             c10::optional<at::Tensor> bias) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(qweight.is_cuda(), "qweight must be a CUDA tensor");
  TORCH_CHECK(w_scale.is_cuda(), "w_scale must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "x must be [M, K]");
  TORCH_CHECK(qweight.dim() == 2, "qweight must be [OC, K/2]");
  TORCH_CHECK(w_scale.dim() == 2, "w_scale must be [OC, K/16]");

  const bool out_f32 = x.scalar_type() == at::kFloat;
  at::Tensor x_f = x.to(at::kFloat).contiguous();
  at::Tensor qw = qweight.contiguous();
  at::Tensor ws = w_scale.contiguous();

  const int M = x_f.size(0);
  const int K = x_f.size(1);
  const int OC = qweight.size(0);
  const int K2 = K / 2;
  const int block_size = 16;
  TORCH_CHECK(qweight.size(1) == K2, "qweight last dim must be K/2");
  TORCH_CHECK(w_scale.size(0) == OC && w_scale.size(1) == K / block_size,
              "w_scale must be [OC, K/16]");
  TORCH_CHECK(K % block_size == 0, "K must be divisible by block_size");

  c10::cuda::CUDAGuard guard(x.device());

  // Allocate workspaces.
  auto opts_u8 = x.options().dtype(at::kByte);
  auto opts_f32 = x.options().dtype(at::kFloat);
  at::Tensor a_stored = at::empty({(int64_t)((size_t)M * K2)}, opts_u8);
  at::Tensor b_stored = at::empty({(int64_t)((size_t)OC * K2)}, opts_u8);
  at::Tensor act_scales_simple = at::empty({M, K / block_size}, opts_u8);
  int tiles_outer_A = cdiv(M, 128), tiles_inner_A = cdiv(K, 64);
  int tiles_outer_B = cdiv(OC, 128), tiles_inner_B = cdiv(K, 64);
  at::Tensor a_scales = at::zeros(
      {(int64_t)((size_t)tiles_outer_A * tiles_inner_A * 512)}, opts_u8);
  at::Tensor b_scales = at::zeros(
      {(int64_t)((size_t)tiles_outer_B * tiles_inner_B * 512)}, opts_u8);
  at::Tensor dC = at::empty({(int64_t)((size_t)M * OC)}, opts_f32);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // 1. Activation block scales.
  {
    dim3 grid(M, K / block_size);
    compute_act_scales<<<grid, 1, 0, stream>>>(
        x_f.data_ptr<float>(), M, K, block_size, K / block_size,
        act_scales_simple.data_ptr<unsigned char>());
  }
  // 2. Tile activation scales into cuBLASLt layout.
  {
    int total = tiles_outer_A * tiles_inner_A * 512;
    int threads = 256;
    tile_scales<<<cdiv(total, threads), threads, 0, stream>>>(
        act_scales_simple.data_ptr<unsigned char>(), M, K, block_size,
        a_scales.data_ptr<unsigned char>());
  }
  // 3. Quantize activations into transposed/packed A layout.
  {
    int total = (size_t)M * K2;
    int threads = 256;
    quantize_act_stored<<<cdiv(total, threads), threads, 0, stream>>>(
        x_f.data_ptr<float>(), M, K, block_size,
        act_scales_simple.data_ptr<unsigned char>(),
        a_stored.data_ptr<unsigned char>());
  }
  // 4. Transpose weights into B layout.
  {
    int total = (size_t)OC * K2;
    int threads = 256;
    transpose_weight_bytes<<<cdiv(total, threads), threads, 0, stream>>>(
        qw.data_ptr<unsigned char>(), OC, K2,
        b_stored.data_ptr<unsigned char>());
  }
  // 5. Tile weight scales into cuBLASLt layout.
  {
    int total = tiles_outer_B * tiles_inner_B * 512;
    int threads = 256;
    tile_scales<<<cdiv(total, threads), threads, 0, stream>>>(
        ws.data_ptr<unsigned char>(), OC, K, block_size,
        b_scales.data_ptr<unsigned char>());
  }

  // 6. cuBLASLt FP4 GEMM (TN, VEC16_UE4M3, FP32 accumulate).
  cublasLtHandle_t lt;
  CHECK_CUBLASLT(cublasLtCreate(&lt));

  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr,
                         Ddesc = nullptr;
  cublasLtMatmulDesc_t opDesc = nullptr;
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, K, M, K));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, K, OC, K));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, OC, M));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, M, OC, M));
  int order_col = CUBLASLT_ORDER_COL;
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));

  CHECK_CUBLASLT(
      cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t opA = CUBLAS_OP_T, opB = CUBLAS_OP_N;
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

  cublasLtMatmulMatrixScale_t smode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  void* pA_scales = a_scales.data_ptr();
  void* pB_scales = b_scales.data_ptr();
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pA_scales, sizeof(pA_scales)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pB_scales, sizeof(pB_scales)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &smode, sizeof(smode)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &smode, sizeof(smode)));

  cublasLtMatmulHeuristicResult_t heur;
  int returnedResults = 0;
  cublasLtMatmulPreference_t pref = nullptr;
  CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
      lt, opDesc, Adesc, Bdesc, Cdesc, Ddesc, pref, 1, &heur, &returnedResults));

  // Some heuristics (especially the small-M decode shapes) require workspace.
  at::Tensor workspace = at::empty(
      {(int64_t)heur.workspaceSize}, opts_u8);

  float alpha = 1.0f, beta = 0.0f;
  void* pA = a_stored.data_ptr();
  void* pB = b_stored.data_ptr();
  void* pC = dC.data_ptr();
  CHECK_CUBLASLT(cublasLtMatmul(lt, opDesc, &alpha, pA, Adesc, pB, Bdesc, &beta,
                                pC, Ddesc, pC, Ddesc, &heur.algo,
                                workspace.data_ptr(), heur.workspaceSize, stream));

  // 7. Transpose output + optional bias.
  at::Tensor out = at::empty({M, OC}, x.options());
  int total = (size_t)M * OC;
  int threads = 256;
  if (out_f32) {
    at::Tensor bias_f = bias.has_value() ? bias.value().to(at::kFloat).contiguous()
                                         : at::Tensor();
    output_kernel_f32<<<cdiv(total, threads), threads, 0, stream>>>(
        dC.data_ptr<float>(), M, OC,
        bias_f.defined() ? bias_f.data_ptr<float>() : nullptr,
        bias_f.defined() ? 1 : 0, out.data_ptr<float>());
  } else {
    at::Tensor bias_h = bias.has_value() ? bias.value().to(at::kHalf).contiguous()
                                         : at::Tensor();
    output_kernel<<<cdiv(total, threads), threads, 0, stream>>>(
        dC.data_ptr<float>(), M, OC,
        bias_h.defined() ? reinterpret_cast<const __half*>(bias_h.data_ptr<at::Half>())
                         : nullptr,
        bias_h.defined() ? 1 : 0, reinterpret_cast<__half*>(out.data_ptr<at::Half>()));
  }

  CHECK_CUDA(cudaGetLastError());

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatmulDescDestroy(opDesc);
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
  cublasLtMatrixLayoutDestroy(Ddesc);
  cublasLtDestroy(lt);

  return out;
}
