// NVFP4 W4A4 cuBLASLt GEMM pybind module.

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "gemm_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<NVFP4GEMM>(m, "NVFP4GEMM")
      .def(pybind11::init<at::Tensor, at::Tensor, c10::optional<at::Tensor>>())
      .def("forward", &NVFP4GEMM::forward);
  m.def("gemm_act_forward", &gemm_act_forward_cuda,
        "NVFP4 W4A4 GEMM for two dynamic activations: C = x @ y^T");
  m.def("kv_quantize_fp8", &kv_quantize_fp8_cuda,
        "Quantize a [L, H, D] KV tensor to E4M3 + one fp32 scale per (token, head)");
  m.def("kv_quantize", &kv_quantize_cuda,
        "Quantize a [L, H, D] KV tensor to NVFP4 (packed E2M1 + UE4M3 scales)");
  m.def("decode_attention", &decode_attention_cuda,
        "Fused NVFP4 decode attention over an FP4 KV cache");
}
