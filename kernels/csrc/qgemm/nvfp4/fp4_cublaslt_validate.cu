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

// ---- E2M1 (NVFP4) host conversion (mirrors omniserve/utils/nvfp4.py) ----
static const float E2M1_MAGS[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
static const float E2M1_BOUNDS[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};

static inline float e2m1_decode(uint8_t v) {
  float m = E2M1_MAGS[v & 7];
  return (v & 8) ? -m : m;
}
static inline uint8_t e2m1_encode(float x) {
  bool sign = x < 0.0f;
  float a = std::fabs(x);
  if (a > 6.0f) a = 6.0f;
  int idx = 0;
  while (idx < 7 && a > E2M1_BOUNDS[idx]) idx++;
  return (uint8_t)(idx | (sign ? 8 : 0));
}

// ---- E4M3 (FP8) host conversion ----
static inline float e4m3_decode(uint8_t b) {
  int sign = (b >> 7) & 1;
  int exp = (b >> 3) & 0xF;
  int man = b & 7;
  float val;
  if (exp == 0) val = (man / 8.0f) * std::pow(2.0f, -6.0f);
  else val = (1.0f + man / 8.0f) * std::pow(2.0f, exp - 7);
  return sign ? -val : val;
}
static inline uint8_t e4m3_encode(float x) {
  bool sign = x < 0.0f;
  float a = std::fabs(x);
  if (a > 240.0f) a = 240.0f;
  int exp;
  float m = std::frexp(a, &exp);
  int E = exp + 6;
  int frac = (int)std::lround((m * 2.0f - 1.0f) * 8.0f);
  int carry = frac >> 3;
  E += carry;
  frac &= 7;
  if (E < 1) E = 1;
  if (E > 14) E = 14;
  int bits;
  if (a > 0.0f && exp <= -6) {
    int sub_man = (int)std::lround(a * 512.0f);
    if (sub_man < 0) sub_man = 0;
    if (sub_man > 8) sub_man = 8;
    bits = (sub_man == 8) ? (1 << 3) : sub_man;
  } else {
    bits = (E << 3) | frac;
  }
  if (a == 0.0f) bits = 0;
  if (sign) bits |= 0x80;
  return (uint8_t)bits;
}

static void pack_e2m1_rowmajor_k(const std::vector<float>& in, int outer, int inner,
                                 std::vector<uint8_t>& out) {
  // in is outer x inner row-major; pack 2 consecutive inner elements per byte, low nibble first
  out.resize((size_t)outer * inner / 2);
  for (int o = 0; o < outer; o++)
    for (int i = 0; i < inner; i += 2)
      out[(size_t)o * (inner/2) + i/2] = (uint8_t)(e2m1_encode(in[(size_t)o*inner+i]) |
                                                   (e2m1_encode(in[(size_t)o*inner+i+1]) << 4));
}

// Build VEC16_UE4M3 scale tensor for a math matrix of shape [outer, inner].
// Scales are defined on the math matrix (outer = M for A, N for B; inner = K).
// Each 16-element block along inner gets one UE4M3 scale S = block_absmax / 6.
// Tiled layout: each 128x64 block -> 128x4 tile (512 bytes).
// within tile: off = (o%32)*16 + (o/32)*4 + c
static void build_vec16_scales(const std::vector<float>& mat, int outer, int inner,
                               std::vector<uint8_t>& scales) {
  int tiles_outer = (outer + 127) / 128;
  int tiles_inner = (inner + 63) / 64;
  scales.assign((size_t)tiles_outer * tiles_inner * 512, 0);
  for (int to = 0; to < tiles_outer; to++) {
    for (int ti = 0; ti < tiles_inner; ti++) {
      for (int o = 0; o < 128; o++) {
        int mat_o = to * 128 + o;
        for (int c = 0; c < 4; c++) {
          float mx = 0.0f;
          for (int e = 0; e < 16; e++) {
            int mat_i = ti * 64 + c * 16 + e;
            if (mat_o < outer && mat_i < inner) {
              float v = std::fabs(mat[(size_t)mat_o * inner + mat_i]);
              if (v > mx) mx = v;
            }
          }
          float S = mx / 6.0f;
          int off = (o % 32) * 16 + (o / 32) * 4 + c;
          size_t tile_off = ((size_t)to * tiles_inner + ti) * 512 + off;
          scales[tile_off] = e4m3_encode(S);
        }
      }
    }
  }
}

// quantize math matrix [outer, inner] to NVFP4 with per-16 block UE4M3 scales.
// data[i] = round(x / S), S = block_absmax / 6
static void quantize_rowmajor(const std::vector<float>& mat, int outer, int inner,
                              std::vector<uint8_t>& packed, std::vector<uint8_t>& scales) {
  packed.resize((size_t)outer * inner / 2);
  build_vec16_scales(mat, outer, inner, scales);
  // reconstruct scale per block for quantization (decode not needed; recompute S)
  for (int o = 0; o < outer; o++) {
    for (int blk = 0; blk < inner / 16; blk++) {
      float mx = 0.0f;
      for (int e = 0; e < 16; e++) {
        float v = std::fabs(mat[(size_t)o * inner + blk * 16 + e]);
        if (v > mx) mx = v;
      }
      float S = mx / 6.0f;
      if (S < 1e-12f) S = 1e-12f;
      for (int e = 0; e < 16; e++) {
        float q = mat[(size_t)o * inner + blk * 16 + e] / S;
        int i = blk * 16 + e;
        int half = i / 2;
        uint8_t nib = e2m1_encode(q);
        if ((i & 1) == 0) packed[(size_t)o * (inner/2) + half] = nib;
        else packed[(size_t)o * (inner/2) + half] |= (nib << 4);
      }
    }
  }
}

static void ref_gemm(const std::vector<float>& A, const std::vector<float>& B,
                     std::vector<float>& C, int m, int n, int k) {
  C.assign((size_t)m * n, 0.0f);
  for (int i = 0; i < m; i++)
    for (int p = 0; p < k; p++)
      for (int j = 0; j < n; j++)
        C[(size_t)i * n + j] += A[(size_t)i * k + p] * B[(size_t)p * n + j];
}

int main(int argc, char** argv) {
  int m = 256, n = 256, k = 256;
  if (argc > 1) m = atoi(argv[1]);
  if (argc > 2) n = atoi(argv[2]);
  if (argc > 3) k = atoi(argv[3]);

  std::vector<float> A_math((size_t)m * k), B_math((size_t)k * n);
  srand(1234);
  for (auto& v : A_math) v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
  for (auto& v : B_math) v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

  std::vector<float> C_ref;
  ref_gemm(A_math, B_math, C_ref, m, n, k);

  // quantize math matrices
  std::vector<uint8_t> A_packed_row, B_packed_row, scalesA, scalesB;
  quantize_rowmajor(A_math, m, k, A_packed_row, scalesA);
  quantize_rowmajor(B_math, n, k, B_packed_row, scalesB); // B math is n x k? wait B is k x n

  // B math matrix is [k, n]; its outer = N = n, inner = K = k.
  // We need B in "row-major math" [n, k] for the scale/quant helpers (outer=n, inner=k).
  std::vector<float> B_math_T((size_t)n * k);
  for (int j = 0; j < n; j++)
    for (int p = 0; p < k; p++)
      B_math_T[(size_t)j * k + p] = B_math[(size_t)p * n + j];
  quantize_rowmajor(B_math_T, n, k, B_packed_row, scalesB);

  // Now convert packed row-major (outer x inner/2) to cuBLASLt storage:
  // A stored k x m col-major (op=T). B stored k x n col-major (op=N).
  std::vector<uint8_t> A_packed((size_t)k * m / 2), B_packed((size_t)k * n / 2);
  for (int j = 0; j < m; j++)          // A stored col j = math row j
    for (int i = 0; i < k; i += 2) {   // A stored rows i,i+1 in same byte
      // A_stored[i + j*k] is memory position (i + j*k). Byte index = (i + j*k)/2 (since i even)
      A_packed[(size_t)(i + j*k)/2] = A_packed_row[(size_t)j*(k/2) + i/2];
    }
  for (int j = 0; j < n; j++)
    for (int i = 0; i < k; i += 2)
      B_packed[(size_t)(i + j*k)/2] = B_packed_row[(size_t)j*(k/2) + i/2];

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

  // Dequantized reference: A_deq[m,k], B_deq[k,n] from packed + scales
  auto deq_scale = [](const std::vector<uint8_t>& scales, int outer, int inner, int o, int i) -> float {
    int to = o / 128, ti = i / 64;
    int oo = o % 128, ii = (i / 16) % 4;
    int tiles_inner = (inner + 63) / 64;
    int off = (oo % 32) * 16 + (oo / 32) * 4 + ii;
    size_t idx = ((size_t)to * tiles_inner + ti) * 512 + off;
    return e4m3_decode(scales[idx]);
  };
  std::vector<float> A_deq((size_t)m * k), B_deq((size_t)k * n);
  for (int o = 0; o < m; o++)
    for (int i = 0; i < k; i++) {
      int half = i / 2;
      uint8_t byte = A_packed_row[(size_t)o * (k/2) + half];
      uint8_t nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
      A_deq[(size_t)o * k + i] = e2m1_decode(nib) * deq_scale(scalesA, m, k, o, i);
    }
  for (int o = 0; o < n; o++)
    for (int i = 0; i < k; i++) {
      int half = i / 2;
      uint8_t byte = B_packed_row[(size_t)o * (k/2) + half];
      uint8_t nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
      B_deq[(size_t)i * n + o] = e2m1_decode(nib) * deq_scale(scalesB, n, k, o, i);
    }
  std::vector<float> C_deqref;
  ref_gemm(A_deq, B_deq, C_deqref, m, n, k);

  double err = 0, norm = 0, qerr = 0, qnorm = 0;
  int bad = 0;
  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++) {
      double c = C_out[(size_t)i + (size_t)j * m];
      double r = C_ref[(size_t)i * n + j];
      double q = C_deqref[(size_t)i * n + j];
      err += (c - r) * (c - r); norm += r * r;
      qerr += (c - q) * (c - q); qnorm += q * q;
      if (std::fabs(c - q) > 1e-2 && bad < 5) {
        printf("bad[%d,%d] got=%f deqref=%f\n", i, j, c, q); bad++;
      }
    }
  printf("rel RMSE vs fp32 = %.4f | vs deqref = %.6f  (shape %dx%dx%d)\n",
         std::sqrt(err / norm), std::sqrt(qerr / qnorm), m, n, k);
  return 0;
}
