"""NVFP4 (W4A4) attention.

Quantizes Q, K, V and the post-softmax probabilities P to NVFP4 (E2M1) with
per-16 UE4M3 block scales, and computes both attention GEMMs (QK^T and PV) on
Blackwell FP4 tensor cores through ``omniserve_backend.qgemm_nvfp4``.

Accuracy is intentionally traded for efficiency.
"""

from typing import Optional

import torch
import torch.nn as nn


class NVFP4Attention(nn.Module):
    def __init__(self, head_dim: int, scale: Optional[float] = None,
                 causal: bool = True):
        super().__init__()
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim ** -0.5
        self.causal = causal

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
            s = s * self.scale
            if self.causal:
                i = torch.arange(Lq, device=s.device).unsqueeze(1)  # [Lq,1]
                j = torch.arange(Lk, device=s.device).unsqueeze(0)  # [1,Lk]
                mask = j > (causal_start + i)                        # [Lq,Lk]
                s = s.masked_fill(mask, float("-inf"))
            p = torch.softmax(s, dim=-1)
            # O = P @ V -> gemm_act_forward(p, V.t()) with V.t() [D, Lk].
            o = qgemm_nvfp4.gemm_act_forward(p, v[:, kh, :].t().contiguous())
            out.append(o)
        return torch.stack(out, dim=1).half()
