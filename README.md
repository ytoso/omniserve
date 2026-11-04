# NVFP4 (W4A8-FP) Experiment

> **Status:** Part A (data-format layer) done and tested on CPU.
> Part B (fused GEMM kernel) pending — blocked on Blackwell hardware + CUDA toolchain.

## Goal

Replace QServe's progressive **W4A8** quantization (FP16 → INT8 → INT4 with zero
points) with a **single-stage FP16 → NVFP4** path:

- **No INT8 intermediate** (drop progressive quantization).
- **No zero point** — use 4-bit *floating point* (NVFP4 / E2M1) instead of unsigned INT4.
- **Accuracy is intentionally ignored**; the goal is to evaluate efficiency.

## Why NVFP4 matters (and the hard constraint)

NVFP4 (1 sign + 2 exponent + 1 mantissa) is NVIDIA's 4-bit float format. Its
tensor-core MMA instructions **only exist on SM100 (Blackwell: B100/B200/GB200,
RTX 50-series)**. On Ampere/Hopper/Ada there is no FP4 MMA — FP4 can only be
*emulated* by dequantizing to FP16, which is exactly the slow path QServe exists
to avoid.

The existing W4A8 kernels (`kernels/csrc/qgemm/w4a8_per_*`) target SM 8.0–9.0 and
use `mma.sync ... .s8.s8.s32` (int8 tensor cores). They cannot be "swapped" to
FP4; Part B is a full kernel effort requiring CUDA ≥ 12.8.

## What's done (Part A)

Pure-Python/torch data-format layer, testable without the CUDA backend.

| File | Change |
|---|---|
| `omniserve/utils/nvfp4.py` | **new** — E2M1 ↔ float, E4M3 ↔ float, 2-per-byte packing, single-stage weight quantization |
| `omniserve/modeling/layers/quantized_linear/nvfp4_linear.py` | **new** — `NVFP4Linear` layer (reference forward + `from_linear`) |
| `omniserve/modeling/layers/quantized_linear/__init__.py` | export `NVFP4Linear` unconditionally; guard backend-dependent imports so the package imports on CPU |
| `omniserve/utils/quant_config.py` | **new** `NVFP4QuantConfig` (`get_name() == "nvfp4"`) |
| `unit_tests/test_nvfp4.py` | **new** — standalone CPU tests (no pytest / transformers / backend needed) |

The original `W4A8OF16LinearDynamicInputScale` and the INT4 kernels are **kept
as a compile-time fallback** and are unchanged.

## Format design

Weight matrix `W` with shape `[OC, IC]`:

```
W = e2m1(qweight) * e4m3(w_scale) * w_global_scale
```

- `qweight`: `uint8 [OC, IC // 2]` — two E2M1 values per byte, **low nibble = first element** (matches NVIDIA NVFP4 packing).
- `w_scale`: `uint8 [OC, IC // block_size]` — per-block E4M3 scales, one per `block_size` (default 16) elements along `IC`. Stored **relative to the global scale** (`block_max / w_max ∈ [0, 1]`).
- `w_global_scale`: `float32` scalar — `w_max / 6` so the full tensor fits in FP4's `[-6, 6]` range.

Single-stage quantization (no INT8, no zero point):

1. `global_scale = max|W| / 6`
2. per block: `block_scale = max|block| / max|W|`, rounded to E4M3
3. `fp4 = round_to_e2m1(W / (block_scale * global_scale))`, then nibble-pack

## Validation

```bash
python3 unit_tests/test_nvfp4.py
```

Results (CPU, torch 2.8):
- E2M1 / E4M3 round-trips exact on their representable grids.
- Weight round-trip: max abs err = 0.541, scale-relative err = 0.132.
- `nn.Linear` vs `NVFP4Linear`: **10.4%** relative output error (expected for FP4; accuracy is out of scope).

## Part B — remaining work (blocked)

1. **Target GPU / toolchain** — confirm Blackwell (SM100) access and CUDA ≥ 12.8
   (e.g., the `user@syt` server). The current machine has no CUDA at all.
2. **Kernel strategy** — pick one:
   - cuBLAS / cuBLASLt FP4 GEMM (least code, loses fused epilogue),
   - CUTLASS SM100 FP4 kernels (`cutlass::nv_float4_t` / MXFP4),
   - hand-written SM100 kernel (full rewrite of `qgemm/w4a8_per_*`).
3. Add `omniserve_backend.qgemm_nvfp4` and dispatch from `NVFP4Linear.forward`.
4. Wire `NVFP4Linear` into `llama_w4a8_unpad.py` (or a new model) and a
   checkpoint converter (direct HF FP16 → NVFP4, no calibration needed).
5. Benchmark prefill/decoding throughput vs the W4A8 INT4 baseline.

## Open questions

- Activation side: current A8 is per-token **INT8**. Blackwell FP4 GEMMs want
  **FP8 (E4M3)** activations — do we also switch the activation quant kernel?
- Per-tensor vs per-output-channel global scale (NVIDIA uses per-tensor for
  weights; keep per-tensor for now).
