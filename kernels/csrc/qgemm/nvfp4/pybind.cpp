// NVFP4 W4A4 cuBLASLt GEMM pybind module.

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "gemm_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_forward_cuda", &gemm_forward_cuda,
        "NVFP4 W4A4 GEMM (cuBLASLt FP4 tensor cores)");
}
