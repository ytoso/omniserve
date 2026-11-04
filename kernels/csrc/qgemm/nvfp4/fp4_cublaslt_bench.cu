#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CHECK_CUBLAS(call) do { cublasStatus_t e = (call); if (e != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)e); exit(1); } } while(0)

static uint8_t e4m3_one() { return 0x38; }

static double now_s() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

// FP4 GEMM via cuBLASLt, TN, VEC16_UE4M3, C/D FP32.
static double bench_fp4(int m, int n, int k, int iters) {
  // allocate device tensors: A k x m, B k x n packed FP4, scales, C m x n fp32
  size_t a_bytes = (size_t)k * m / 2, b_bytes = (size_t)k * n / 2;
  int tiles_outer_A = (m + 127) / 128, tiles_inner_A = (k + 63) / 64;
  int tiles_outer_B = (n + 127) / 128, tiles_inner_B = (k + 63) / 64;
  size_t sa = (size_t)tiles_outer_A * tiles_inner_A * 512;
  size_t sb = (size_t)tiles_outer_B * tiles_inner_B * 512;

  uint8_t *dA, *dB, *dSA, *dSB; float *dC;
  CHECK_CUDA(cudaMalloc(&dA, a_bytes));
  CHECK_CUDA(cudaMalloc(&dB, b_bytes));
  CHECK_CUDA(cudaMalloc(&dSA, sa));
  CHECK_CUDA(cudaMalloc(&dSB, sb));
  CHECK_CUDA(cudaMalloc(&dC, (size_t)m * n * sizeof(float)));
  CHECK_CUDA(cudaMemset(dA, 0, a_bytes));
  CHECK_CUDA(cudaMemset(dB, 0, b_bytes));
  CHECK_CUDA(cudaMemset(dSA, e4m3_one(), sa));
  CHECK_CUDA(cudaMemset(dSB, e4m3_one(), sb));

  cublasLtHandle_t lt; CHECK_CUBLAS(cublasLtCreate(&lt));
  cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc; cublasLtMatmulDesc_t opDesc;
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, k, m, k));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, k, n, k));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, m));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, m, n, m));
  int order_col = CUBLASLT_ORDER_COL;
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
  CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t opA = CUBLAS_OP_T, opB = CUBLAS_OP_N;
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
  cublasLtMatmulMatrixScale_t smode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dSA, sizeof(dSA)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dSB, sizeof(dSB)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &smode, sizeof(smode)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &smode, sizeof(smode)));

  cublasLtMatmulHeuristicResult_t heur; int ret = 0;
  cublasLtMatmulPreference_t pref = nullptr;
  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(lt, opDesc, Adesc, Bdesc, Cdesc, Ddesc, pref, 1, &heur, &ret));

  float alpha = 1.0f, beta = 0.0f;
  // warmup
  for (int i = 0; i < 10; i++)
    CHECK_CUBLAS(cublasLtMatmul(lt, opDesc, &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Ddesc, dC, Ddesc, &heur.algo, nullptr, 0, 0));
  CHECK_CUDA(cudaDeviceSynchronize());
  double t0 = now_s();
  for (int i = 0; i < iters; i++)
    CHECK_CUBLAS(cublasLtMatmul(lt, opDesc, &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Ddesc, dC, Ddesc, &heur.algo, nullptr, 0, 0));
  CHECK_CUDA(cudaDeviceSynchronize());
  double dt = now_s() - t0;

  double flops = 2.0 * m * n * k;
  printf("FP4   m=%5d n=%5d k=%5d  %.3f ms  %.1f TFLOPS\n", m, n, k, dt/iters*1e3, flops / (dt/iters) / 1e12);

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatmulDescDestroy(opDesc);
  cublasLtMatrixLayoutDestroy(Adesc); cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Cdesc); cublasLtMatrixLayoutDestroy(Ddesc);
  cublasLtDestroy(lt);
  cudaFree(dA); cudaFree(dB); cudaFree(dSA); cudaFree(dSB); cudaFree(dC);
  return dt/iters;
}

// FP16 GEMM via cuBLAS (A^T * B to match TN), C/D FP32.
static double bench_fp16(int m, int n, int k, int iters) {
  size_t a = (size_t)k * m, b = (size_t)k * n, c = (size_t)m * n;
  __half *dA, *dB; float *dC;
  CHECK_CUDA(cudaMalloc(&dA, a * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dB, b * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dC, c * sizeof(float)));
  CHECK_CUDA(cudaMemset(dA, 0, a * sizeof(__half)));
  CHECK_CUDA(cudaMemset(dB, 0, b * sizeof(__half)));

  cublasHandle_t h; CHECK_CUBLAS(cublasCreate(&h));
  float alpha = 1.0f, beta = 0.0f;
  // C = A^T * B, A is k x m (so A^T is m x k), B is k x n
  for (int i = 0; i < 10; i++)
    CHECK_CUBLAS(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                              dA, CUDA_R_16F, k, dB, CUDA_R_16F, k, &beta,
                              dC, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  CHECK_CUDA(cudaDeviceSynchronize());
  double t0 = now_s();
  for (int i = 0; i < iters; i++)
    CHECK_CUBLAS(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                              dA, CUDA_R_16F, k, dB, CUDA_R_16F, k, &beta,
                              dC, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  CHECK_CUDA(cudaDeviceSynchronize());
  double dt = now_s() - t0;
  double flops = 2.0 * m * n * k;
  printf("FP16  m=%5d n=%5d k=%5d  %.3f ms  %.1f TFLOPS\n", m, n, k, dt/iters*1e3, flops / (dt/iters) / 1e12);

  cublasDestroy(h);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return dt/iters;
}

int main(int argc, char** argv) {
  // shapes: m n k
  struct S { int m, n, k; };
  std::vector<S> shapes = {
    {1, 4096, 4096}, {1, 13824, 14336}, {8, 4096, 4096},
    {128, 4096, 4096}, {2048, 4096, 4096}, {2048, 13824, 14336},
    {4096, 4096, 4096}, {4096, 14336, 14336},
  };
  if (argc >= 4) shapes = {{atoi(argv[1]), atoi(argv[2]), atoi(argv[3])}};
  for (auto s : shapes) {
    int iters = (s.m <= 8) ? 200 : 50;
    bench_fp16(s.m, s.n, s.k, iters);
    bench_fp4(s.m, s.n, s.k, iters);
    printf("\n");
  }
  return 0;
}
