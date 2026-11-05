// NVFP4 W4A4 cuBLASLt GEMM pybind module.

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "gemm_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<NVFP4GEMM>(m, "NVFP4GEMM")
      .def(pybind11::init<at::Tensor, at::Tensor, c10::optional<at::Tensor>>())
      .def("forward", &NVFP4GEMM::forward);
}
