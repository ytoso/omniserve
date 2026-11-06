"""NVFP4 (E2M1) weight-format utilities for the W4A8-FP experiment.

NVFP4 is NVIDIA's 4-bit floating-point format (1 sign, 2 exponent, 1 mantissa),
used for 4-bit weights on Blackwell (SM100) GPUs. This module implements the
pure-Python/torch data-format side so it can be validated on CPU without the
compiled ``omniserve_backend`` extension.

Contents
--------
- E2M1 (NVFP4) <-> float conversion + 2-values-per-byte packing
- E4M3 (FP8) <-> float conversion (used for the per-block weight scales)
- single-stage FP16 -> NVFP4 weight quantization (no INT8 intermediate,
  no zero point)

Accuracy is deliberately traded for efficiency in this experiment.
"""

import torch

# ---------------------------------------------------------------------------
# E2M1 (NVFP4): 1 sign bit, 2 exponent bits, 1 mantissa bit.
# Positive magnitudes (exponent bias = 1): 0, 0.5, 1, 1.5, 2, 3, 4, 6
# ---------------------------------------------------------------------------
_E2M1_MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Midpoints between consecutive magnitudes (7 boundaries -> 8 bins).
_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def float_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Convert float tensor to E2M1 nibbles (``uint8`` values in [0, 15]).

    Values are clamped to [-6, 6] and rounded to the nearest representable
    magnitude (boundary values round up).
    """
    x = x.float()
    sign = (x < 0).to(torch.int32)
    a = x.abs().clamp(max=6.0)
    idx = torch.searchsorted(_E2M1_BOUNDS.to(a.device), a.reshape(-1))
    idx = idx.reshape(a.shape)
    return (idx.to(torch.int32) | (sign << 3)).to(torch.uint8)


def e2m1_to_float(bits: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 nibbles (``uint8``) back to float."""
    bits = bits.to(torch.int32)
    sign = (bits >> 3) & 1
    mags = _E2M1_MAGS.to(bits.device)[bits & 7]
    return mags * (1.0 - 2.0 * sign.to(torch.float32))


def pack_e2m1(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 nibbles 2-per-byte along the last dim (even -> low nibble).

    Input shape ``[..., N]`` (N even) -> output shape ``[..., N/2]`` uint8.
    """
    lo = nibbles[..., 0::2]
    hi = nibbles[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Reverse of :func:`pack_e2m1` (``[..., N/2]`` -> ``[..., N]`` nibbles)."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)


# ---------------------------------------------------------------------------
# E4M3 (FP8): 1 sign bit, 4 exponent bits, 3 mantissa bits, bias = 7.
# Max normal = 240, min normal = 2^-6, min subnormal = 2^-9.
# ---------------------------------------------------------------------------


def float_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Convert float tensor to E4M3 (``uint8``), saturating to [-240, 240]."""
    x = x.float()
    sign = (x < 0).to(torch.int32)
    a = x.abs().clamp(max=240.0)

    m, e = torch.frexp(a)  # a = m * 2^e, m in [0.5, 1)
    e = e.to(torch.int32)
    E = e + 6  # stored exponent for normal numbers
    frac = torch.round((m * 2.0 - 1.0) * 8.0).to(torch.int32)  # 0..8
    carry = frac >> 3
    E = E + carry
    frac = frac & 7
    E = E.clamp(1, 14)
    normal_bits = (E << 3) | frac

    # Subnormals: a < 2^-6  <=>  frexp exponent <= -6.
    sub = (e <= -6) & (a > 0)
    sub_man = torch.round(a * 512.0).to(torch.int32).clamp(0, 8)
    sub_bits = sub_man
    sub_bits = torch.where(
        sub_man == 8, torch.full_like(sub_bits, 1 << 3), sub_bits
    )  # rounds up to smallest normal

    bits = torch.where(sub, sub_bits, normal_bits)
    bits = bits | (sign << 7)
    bits = torch.where(a == 0, torch.zeros_like(bits), bits)
    return bits.to(torch.uint8)


def e4m3_to_float(bits: torch.Tensor) -> torch.Tensor:
    """Decode E4M3 (``uint8``) back to float."""
    bits = bits.to(torch.int32)
    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0xF
    man = bits & 7
    manf = man.to(torch.float32)
    val = torch.where(
        exp == 0,
        (manf / 8.0) * (2.0 ** -6),
        (1.0 + manf / 8.0) * (2.0 ** (exp - 7)),
    )
    return val * (1.0 - 2.0 * sign.to(torch.float32))


# ---------------------------------------------------------------------------
# Weight quantization (single-stage FP16 -> NVFP4, no INT8 / zero point)
# ---------------------------------------------------------------------------


def quantize_weight_nvfp4(w: torch.Tensor, block_size: int = 16):
    """Quantize a ``[OC, IC]`` float weight to NVFP4.

    Returns
    -------
    qweight : uint8 ``[OC, IC // 2]`` (2 E2M1 values per byte, low nibble first)
    w_scale : uint8 ``[OC, IC // block_size]`` (E4M3 per-block scales)
    global_scale : float32 scalar (FP32 global scale)

    Dequantization is ``w = e2m1(qweight) * e4m3(w_scale) * global_scale``.
    """
    w = w.float()
    OC, IC = w.shape
    if IC % block_size != 0:
        raise ValueError(f"in_features {IC} not divisible by block_size {block_size}")

    w_r = w.reshape(OC, IC // block_size, block_size)
    block_max = w_r.abs().amax(dim=-1, keepdim=True)
    w_max = w.abs().max().clamp(min=1e-9)

    # global_scale covers the whole tensor range: w_max -> +-6 in FP4.
    global_scale = (w_max / 6.0).float()

    # Block scale is *relative* to the global scale (block_max / w_max in [0, 1])
    # and is stored in E4M3. This is the MXFP4-style two-level scheme:
    #   w = fp4 * block_scale_fp8 * global_scale
    block_scale = block_max / w_max
    scale_fp8 = float_to_e4m3(block_scale).reshape(OC, IC // block_size)
    block_scale_deq = e4m3_to_float(scale_fp8)

    denom = (block_scale_deq.reshape(OC, IC // block_size, 1) * global_scale).clamp(
        min=1e-12
    )
    normalized = w_r / denom
    fp4 = float_to_e2m1(normalized)  # [OC, IC//block_size, block_size] nibbles
    qweight = pack_e2m1(fp4.reshape(OC, IC))
    return qweight, scale_fp8, global_scale


def dequantize_weight_nvfp4(qweight: torch.Tensor, scale_fp8: torch.Tensor,
                            global_scale: torch.Tensor) -> torch.Tensor:
    """Reconstruct a ``[OC, IC]`` float weight from its NVFP4 representation."""
    nibbles = unpack_e2m1(qweight)
    w_fp4 = e2m1_to_float(nibbles)
    scale = e4m3_to_float(scale_fp8)
    block_size = (qweight.shape[1] * 2) // scale_fp8.shape[1]
    scale = scale.repeat_interleave(block_size, dim=-1)
    return w_fp4 * scale * global_scale


# ---------------------------------------------------------------------------
# W4A4 (Atom-style) helpers.
#
# cuBLASLt block-scaled FP4 GEMM requires *both* operands to be
# ``CUDA_R_4F_E2M1`` with per-16 ``UE4M3`` block scales. The absolute-scale
# helpers below match that contract: ``x = e2m1(q) * e4m3(scale)`` with
# ``scale = block_absmax / 6`` for each 16-element block along the last dim.
# ---------------------------------------------------------------------------


# Candidate divisors of the block max used by the MSE-optimal scale search.
_SCALE_CANDIDATES = (6.0, 4.5, 3.0)


def quantize_nvfp4_abs(w: torch.Tensor, block_size: int = 16,
                       scale_search: bool = True):
    """Quantize a ``[..., K]`` float tensor to NVFP4 with absolute UE4M3 scales.

    Returns
    -------
    q : uint8 ``[..., K // 2]`` (2 E2M1 values per byte, low nibble first)
    scale : uint8 ``[..., K // block_size]`` (UE4M3 per-block scales, sign bit 0)

    Dequantization is ``x = e2m1(q) * e4m3(scale)``.

    With ``scale_search`` the per-block UE4M3 scale is chosen to minimise the
    squared reconstruction error over a few divisors of the block max (E2M1's
    single mantissa bit makes the classic ``max/6`` clipping suboptimal).
    """
    x = w.float()
    *lead, K = x.shape
    if K % block_size != 0:
        raise ValueError(f"last dim {K} not divisible by block_size {block_size}")

    xr = x.reshape(-1, K // block_size, block_size)
    block_max = xr.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9)
    cands = _SCALE_CANDIDATES if scale_search else (6.0,)

    best_err = None
    best_bits = None
    best_s = None
    for c in cands:
        bits = float_to_e4m3(block_max / c)
        s = e4m3_to_float(bits).clamp(min=1e-12)
        rec = e2m1_to_float(float_to_e2m1(xr / s)) * s
        err = ((rec - xr) ** 2).sum(dim=-1, keepdim=True)
        if best_err is None:
            best_err, best_bits, best_s = err, bits, s
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_bits = torch.where(m, bits, best_bits)
            best_s = torch.where(m, s, best_s)

    fp4 = float_to_e2m1(xr / best_s)
    q = pack_e2m1(fp4.reshape(*lead, K))
    scale_fp8 = best_bits.reshape(*lead, K // block_size)
    return q, scale_fp8


def dequantize_nvfp4_abs(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reconstruct ``[..., K]`` floats from ``q`` (``[..., K/2]``) and
    ``scale`` (``[..., K/block_size]``)."""
    nibbles = unpack_e2m1(q)
    S = e4m3_to_float(scale)
    block_size = (q.shape[-1] * 2) // scale.shape[-1]
    S = S.repeat_interleave(block_size, dim=-1)
    return e2m1_to_float(nibbles) * S
