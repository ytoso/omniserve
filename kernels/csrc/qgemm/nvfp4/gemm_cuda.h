// NVFP4 W4A4 cuBLASLt GEMM (Atom-style) forward declaration.
//
// Activations are quantized to E2M1 with per-16 UE4M3 block scales on device;
// weights are assumed to be already quantized to E2M1 + UE4M3 by the Python
// layer. The GEMM itself is delegated to cuBLASLt (TN, VEC16_UE4M3, FP32
// accumulate), which on Blackwell uses the 5th-gen tensor cores.

#pragma once

#include <torch/extension.h>

at::Tensor gemm_forward_cuda(
    at::Tensor x,        // [M, K] float16/float32 activations, CUDA
    at::Tensor qweight,  // [OC, K/2] uint8, packed E2M1 (low nibble first), CUDA
    at::Tensor w_scale,  // [OC, K/16] uint8, UE4M3 block scales, CUDA
    c10::optional<at::Tensor> bias);  // [OC] float16/float32, optional
