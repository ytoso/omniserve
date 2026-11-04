"""NVFP4 quantized linear layer (experimental, efficiency-first).

Replaces ``nn.Linear`` with a single-stage FP16 -> NVFP4 (E2M1) weight
quantization. Unlike the original QServe ``W4A8OF16LinearDynamicInputScale``,
this layer has **no** progressive INT8 stage and **no** zero point: weights go
straight to 4-bit floating point with per-block E4M3 scales + one global FP32
scale.

The forward pass is currently a *reference* implementation (dequantize to float
and call ``F.linear``). Part B will replace it with a fused SM100 FP4 GEMM
kernel; see ``README_NVFP4.md``.
"""

import torch

from omniserve.utils.nvfp4 import (
    dequantize_weight_nvfp4,
    quantize_weight_nvfp4,
)


class NVFP4Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        if in_features % block_size != 0:
            raise ValueError(
                f"in_features {in_features} not divisible by block_size {block_size}"
            )

        # 2 E2M1 values packed per byte, low nibble first.
        self.register_buffer(
            "qweight",
            torch.zeros((out_features, in_features // 2), dtype=torch.uint8),
        )
        # Per-block E4M3 scales (one per `block_size` elements along in_features).
        self.register_buffer(
            "w_scale",
            torch.zeros((out_features, in_features // block_size), dtype=torch.uint8),
        )
        # Single FP32 global scale for the whole weight tensor.
        self.register_buffer("w_global_scale", torch.zeros((), dtype=torch.float32))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reference path: dequantize weights and use a standard GEMM.
        # Part B replaces this with `omniserve_backend.qgemm_nvfp4`.
        w = dequantize_weight_nvfp4(self.qweight, self.w_scale, self.w_global_scale)
        bias = None if self.bias is None else self.bias.float()
        out = torch.nn.functional.linear(x.float(), w.float(), bias)
        return out.to(x.dtype)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, block_size: int = 16) -> "NVFP4Linear":
        q_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            block_size,
        )
        qweight, w_scale, global_scale = quantize_weight_nvfp4(
            linear.weight.data, block_size
        )
        q_linear.qweight.copy_(qweight)
        q_linear.w_scale.copy_(w_scale)
        q_linear.w_global_scale.copy_(global_scale)
        if linear.bias is not None:
            q_linear.bias.copy_(linear.bias.data.half())
        return q_linear

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, block_size={self.block_size}"
        )
