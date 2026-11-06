"""NVFP4 quantized linear layer (experimental, efficiency-first).

W4A4 (Atom-style): both weights and activations are quantized to NVFP4 (E2M1)
with per-16-block UE4M3 scales, then multiplied with a Blackwell FP4 GEMM.

- No progressive INT8 stage, no zero points.
- Accuracy is intentionally ignored; the goal is to measure efficiency.
- The forward pass below is a *reference* dequantize+``F.linear`` path. The
  compiled ``omniserve_backend.qgemm_nvfp4`` (cuBLASLt) path is wired when the
  backend is available.
"""

import torch

from omniserve.utils.nvfp4 import (
    dequantize_nvfp4_abs,
    quantize_nvfp4_abs,
)


class NVFP4Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 16,
        quant_acts: bool = True,
    ):
        super().__init__()
        self.quant_acts = quant_acts
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        if in_features % block_size != 0:
            raise ValueError(
                f"in_features {in_features} not divisible by block_size {block_size}"
            )

        # 2 E2M1 values packed per byte, low nibble first (row-major [OC, IC/2]).
        self.register_buffer(
            "qweight",
            torch.zeros((out_features, in_features // 2), dtype=torch.uint8),
        )
        # Absolute per-block UE4M3 scales, one per `block_size` input elements.
        self.register_buffer(
            "w_scale",
            torch.zeros((out_features, in_features // block_size), dtype=torch.uint8),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
        self._backend = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quant_acts:
            # W4A16: exact activations, dequantized 4-bit weights.
            w = dequantize_nvfp4_abs(self.qweight, self.w_scale)
            bias = None if self.bias is None else self.bias.float()
            out = torch.nn.functional.linear(x.float(), w, bias)
            return out.to(x.dtype)
        try:
            import omniserve_backend.qgemm_nvfp4 as qgemm_nvfp4

            if self._backend is None:
                qweight = self.qweight
                w_scale = self.w_scale
                if not qweight.is_cuda:
                    qweight = qweight.cuda()
                    w_scale = w_scale.cuda()
                self._backend = qgemm_nvfp4.NVFP4GEMM(
                    qweight, w_scale, self.bias
                )
            return self._backend.forward(x)
        except Exception:
            return self._reference_forward(x)

    def _reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        qact, act_scale = quantize_nvfp4_abs(x.float(), self.block_size)
        x_deq = dequantize_nvfp4_abs(qact, act_scale)
        w_deq = dequantize_nvfp4_abs(self.qweight, self.w_scale)
        bias = None if self.bias is None else self.bias.float()
        out = torch.nn.functional.linear(x_deq, w_deq, bias)
        return out.to(x.dtype)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, block_size: int = 16,
                    quant_acts: bool = True) -> "NVFP4Linear":
        q_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            block_size,
            quant_acts=quant_acts,
        )
        qweight, w_scale = quantize_nvfp4_abs(linear.weight.data, block_size)
        q_linear.qweight.copy_(qweight)
        q_linear.w_scale.copy_(w_scale)
        if linear.bias is not None:
            q_linear.bias.copy_(linear.bias.data.half())
        return q_linear

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, block_size={self.block_size}"
        )
