// NVFP4 W4A4 cuBLASLt GEMM (Atom-style), stateful backend.
//
// The Python layer creates one NVFP4GEMM per quantized linear layer. Weight
// layout (byte transpose + tiled UE4M3 scales) is precomputed once in the
// constructor; activation quantization and the cuBLASLt call happen in
// forward(). cuBLASLt descriptors/heuristics are cached per output row count.

#pragma once

#include <torch/extension.h>

#include <cublasLt.h>
#include <unordered_map>
#include <cstdint>

struct NVFP4Plan {
  cublasLtMatrixLayout_t Adesc = nullptr;
  cublasLtMatrixLayout_t Cdesc = nullptr;
  cublasLtMatrixLayout_t Ddesc = nullptr;
  cublasLtMatmulDesc_t opDesc = nullptr;
  cublasLtMatmulHeuristicResult_t heur{};
};

class NVFP4GEMM {
 public:
  NVFP4GEMM(at::Tensor qweight, at::Tensor w_scale,
            c10::optional<at::Tensor> bias);
  ~NVFP4GEMM();

  at::Tensor forward(at::Tensor x);

 private:
  int OC_ = 0;
  int K_ = 0;
  bool out_f32_ = false;

  at::Tensor b_stored_;   // [K/2 * OC] column-major packed FP4 (B operand)
  at::Tensor b_scales_;   // tiled VEC16_UE4M3 scales for B
  c10::optional<at::Tensor> bias_;

  cublasLtHandle_t lt_ = nullptr;
  cublasLtMatrixLayout_t Bdesc_ = nullptr;
  std::unordered_map<int, NVFP4Plan> plans_;

  void ensure_plan(int M, void* a_scales_ptr);
};
