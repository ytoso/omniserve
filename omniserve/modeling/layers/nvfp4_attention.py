"""NVFP4 (W4A4) attention.

Q and K are quantized to NVFP4 (E2M1) with per-16 UE4M3 block scales and the
QK^T product runs on Blackwell FP4 tensor cores. Softmax probabilities are
kept in higher precision for the PV product by default: FP4's single mantissa
bit is far too coarse for the very wide dynamic range of softmax output (it
costs ~1.3 relative error), while FP4 K only costs ~0.13.

Set ``pv_fp4=True`` for the fully-FP4 variant.
"""

from typing import Optional

import torch
import torch.nn as nn


class NVFP4Attention(nn.Module):
    def __init__(self, head_dim: int, scale: Optional[float] = None,
                 causal: bool = True, pv_fp4: bool = False):
        super().__init__()
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim ** -0.5
        self.causal = causal
        self.pv_fp4 = pv_fp4

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                causal_start: int = 0) -> torch.Tensor:
        """q [Lq, H, D], k/v [Lk, KH, D] -> [Lq, H, D] fp16.

        ``causal_start`` is the absolute position of the first query token; it
        is 0 for prefill and ``past_len`` for decode with a KV cache.
        """
        import omniserve_backend.qgemm_nvfp4 as qgemm_nvfp4

        Lq, H, D = q.shape
        Lk = k.shape[0]
        KH = k.shape[1]
        n_rep = H // KH
        out = []
        for h in range(H):
            kh = h // n_rep
            s = qgemm_nvfp4.gemm_act_forward(q[:, h, :], k[:, kh, :])  # [Lq,Lk]
            if self.causal:
                i = torch.arange(Lq, device=s.device).unsqueeze(1)
                j = torch.arange(Lk, device=s.device).unsqueeze(0)
                s = s.masked_fill(j > (causal_start + i), float("-inf"))
            p = torch.softmax(s * self.scale, dim=-1)
            if self.pv_fp4:
                o = qgemm_nvfp4.gemm_act_forward(p, v[:, kh, :].t().contiguous())
            else:
                # Exact probabilities x exact values (V is already FP4-quantized
                # upstream when it comes from the KV cache).
                o = torch.matmul(p.half(), v[:, kh, :])
            out.append(o)
        return torch.stack(out, dim=1).half()
