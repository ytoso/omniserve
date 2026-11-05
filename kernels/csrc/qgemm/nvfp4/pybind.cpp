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
}
