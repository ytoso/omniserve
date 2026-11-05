"""NVFP4 (W4A4) attention.

Quantizes Q, K, V and the post-softmax probabilities P to NVFP4 (E2M1) with
per-16 UE4M3 block scales, and computes both attention GEMMs (QK^T and PV) on
Blackwell FP4 tensor cores through ``omniserve_backend.qgemm_nvfp4``.

Accuracy is intentionally traded for efficiency.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NVFP4Attention(nn.Module):
    def __init__(self, head_dim: int, scale: Optional[float] = None,
                 causal: bool = True):
        super().__init__()
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim ** -0.5
        self.causal = causal

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        """q [L, H, D], k/v [L, KH, D] -> [L, H, D] fp16."""
        import omniserve_backend.qgemm_nvfp4 as qgemm_nvfp4

        L, H, D = q.shape
        KH = k.shape[1]
        n_rep = H // KH
        out = []
        for h in range(H):
            kh = h // n_rep
            s = qgemm_nvfp4.gemm_act_forward(q[:, h, :], k[:, kh, :])  # [L,L] f32
            s = s * self.scale
            if self.causal:
                mask = torch.triu(torch.ones(L, L, device=s.device, dtype=torch.bool),
                                  diagonal=1)
                s = s.masked_fill(mask, float("-inf"))
            p = torch.softmax(s, dim=-1)
            # O = P @ V^T -> gemm_act_forward(p, V.t()) with V.t() [D, L].
            o = qgemm_nvfp4.gemm_act_forward(p, v[:, kh, :].t().contiguous())
            out.append(o)
        return torch.stack(out, dim=1).half()
