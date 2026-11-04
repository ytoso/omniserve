#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define CHECK_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CHECK_CUBLAS(call) do { cublasStatus_t e = (call); if (e != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)e); exit(1); } } while(0)

static uint8_t e2m1_one() { return 2; } // 1.0
static uint8_t e4m3_one() { return 0x38; } // 1.0

int main(int argc, char** argv) {
  int m = 256, n = 256, k = 256;
  if (argc > 1) m = atoi(argv[1]);
  if (argc > 2) n = atoi(argv[2]);
  if (argc > 3) k = atoi(argv[3]);

  // all ones data
  std::vector<uint8_t> A_packed((size_t)k * m / 2, (uint8_t)(e2m1_one() | (e2m1_one() << 4)));
  std::vector<uint8_t> B_packed((size_t)k * n / 2, (uint8_t)(e2m1_one() | (e2m1_one() << 4)));

  // scales: all ones, sized to full tiles
  int tiles_outer_A = (k + 127) / 128;
  int tiles_inner_A = (m + 63) / 64;
  int tiles_outer_B = (k + 127) / 128;
  int tiles_inner_B = (n + 63) / 64;
  std::vector<uint8_t> scalesA((size_t)tiles_outer_A * tiles_inner_A * 512, e4m3_one());
  std::vector<uint8_t> scalesB((size_t)tiles_outer_B * tiles_inner_B * 512, e4m3_one());

  uint8_t *dA, *dB, *dScaleA, *dScaleB; float *dC;
  CHECK_CUDA(cudaMalloc(&dA, A_packed.size()));
  CHECK_CUDA(cudaMalloc(&dB, B_packed.size()));
  CHECK_CUDA(cudaMalloc(&dScaleA, scalesA.size()));
  CHECK_CUDA(cudaMalloc(&dScaleB, scalesB.size()));
  CHECK_CUDA(cudaMalloc(&dC, (size_t)m * n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA, A_packed.data(), A_packed.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, B_packed.data(), B_packed.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dScaleA, scalesA.data(), scalesA.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dScaleB, scalesB.data(), scalesB.size(), cudaMemcpyHostToDevice));

  cublasLtHandle_t lt;
  CHECK_CUBLAS(cublasLtCreate(&lt));

  cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
  cublasLtMatmulDesc_t opDesc;
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
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dScaleA, sizeof(dScaleA)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dScaleB, sizeof(dScaleB)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &smode, sizeof(smode)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &smode, sizeof(smode)));

  cublasLtMatmulHeuristicResult_t heur;
  int returnedResults = 0;
  cublasLtMatmulPreference_t pref = nullptr;
  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
  cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, opDesc, Adesc, Bdesc, Cdesc, Ddesc, pref, 1, &heur, &returnedResults);
  printf("heuristic st=%d returned=%d\n", (int)st, returnedResults);
  if (st != CUBLAS_STATUS_SUCCESS) return 1;

  float alpha = 1.0f, beta = 0.0f;
  st = cublasLtMatmul(lt, opDesc, &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Ddesc, dC, Ddesc, &heur.algo, nullptr, 0, 0);
  printf("matmul st=%d (%s)\n", (int)st, cublasLtGetStatusString(st));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> C_out((size_t)m * n);
  CHECK_CUDA(cudaMemcpy(C_out.data(), dC, C_out.size()*sizeof(float), cudaMemcpyDeviceToHost));

  // expected all k (with scale 1 and values 1)
  double err = 0;
  int bad = 0;
  for (size_t i = 0; i < C_out.size(); i++) {
    double d = C_out[i] - (double)k;
    err += d*d;
    if (std::fabs(d) > 1e-3 && bad < 10) { printf("bad[%zu]=%f (want %d)\n", i, C_out[i], k); bad++; }
  }
  printf("shape %dx%dx%d, nbad=%d, RMSE=%f\n", m, n, k, bad, std::sqrt(err / C_out.size()));
  return 0;
}
