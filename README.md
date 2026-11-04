# NVFP4 (W4A4-FP, Atom-style) Experiment

> **Status:** W4A4-FP path is implemented and validated end-to-end on a
> Blackwell (Jetson AGX Thor, sm_110) GPU: FP4 weights + FP4 activations with
> per-16 UE4M3 block scales, dispatched through `omniserve_backend.qgemm_nvfp4`
> (cuBLASLt FP4 tensor cores). Reference dequant+GEMM matches the kernel
> bit-exactly.

## Goal

Replace QServe's progressive **W4A8** quantization (FP16 → INT8 → INT4 with zero
points) with a **single-stage W4A4-FP** path (Atom-style):

- **No INT8 intermediate** and **no zero point** — weights *and* activations
  are 4-bit floating point (NVFP4 / E2M1) with per-16 UE4M3 block scales.
- **Accuracy is intentionally ignored**; the goal is to evaluate efficiency.

> Why W4A4 instead of W4A8-FP: cuBLASLt's block-scaled FP4 GEMM requires both
> operands to be `CUDA_R_4F_E2M1`. A true W4A8-FP (FP8 × FP4) would need a
> hand-written tcgen05/TMEM kernel on this hardware.

## Why NVFP4 matters (and the hard constraint)

NVFP4 (1 sign + 2 exponent + 1 mantissa) is NVIDIA's 4-bit float format. Its
tensor-core MMA instructions **only exist on Blackwell** (B100/B200/GB200 =
10.0, Blackwell Ultra = 11.0, RTX 50-series = 12.0). On Ampere/Hopper/Ada there
is no FP4 MMA — FP4 can only be *emulated* by dequantizing to FP16, which is
exactly the slow path QServe exists to avoid.

The existing W4A8 kernels (`kernels/csrc/qgemm/w4a8_per_*`) target SM 8.0–9.0 and
use `mma.sync ... .s8.s8.s32` (int8 tensor cores). They cannot be "swapped" to
FP4; Part B is a full kernel effort requiring CUDA ≥ 12.8 (13.0 for sm_110).

This machine is a **Jetson AGX Thor Developer Kit** (Blackwell, compute
capability **11.0**, 20 SMs, CUDA 13.2 toolkit installed to
`/usr/local/cuda`).

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

## Part B — progress

1. **Target GPU / toolchain** ✅ — Jetson AGX Thor (sm_110), CUDA 13.2.
2. **Kernel strategy** — validated the **cuBLASLt FP4 GEMM** path first:
   - `kernels/csrc/qgemm/nvfp4/fp4_cublaslt_validate.cu` matches the
     dequantized E2M1 reference exactly.
   - `kernels/csrc/qgemm/nvfp4/fp4_cublaslt_bench.cu` measures FP4 vs FP16
     (table below).
   - ⚠️ cuBLASLt FP4 is **W4A4** (both operands FP4), so the experiment is
     now **W4A4-FP (Atom-style)**. A true W4A8-FP (FP8 × FP4) would need a
     hand-written tcgen05/TMEM kernel (the warp-level `mma.sync` FP4 forms are
     `sm_120a`-only and are rejected on sm_110).
3. `omniserve_backend.qgemm_nvfp4` (pybind + cuBLASLt) — ✅ implemented and
   dispatched from `NVFP4Linear.forward`; kernel output matches the W4A4
   reference dequant+GEMM **bit-exactly** (`max abs diff = 0.0`).
4. Wire `NVFP4Linear` into `llama_w4a8_unpad.py` (or a new model) and a
   checkpoint converter (direct HF FP16 → NVFP4, no calibration needed) — next.
5. Benchmark prefill/decoding throughput vs the W4A8 INT4 baseline.

### cuBLASLt FP4 (W4A4) vs FP16 on Jetson AGX Thor (sm_110, 20 SMs)

| M | N | K | FP16 TFLOPS | FP4 TFLOPS | speedup |
|---|---|---|---|---|---|
| 1 | 4096 | 4096 | 0.3 | 2.5 | ~8x (mem-bound) |
| 8 | 4096 | 4096 | 2.1 | 9.7 | 4.6x |
| 128 | 4096 | 4096 | 33.7 | 320.0 | 9.5x |
| 2048 | 4096 | 4096 | 154.5 | 375.5 | 2.4x |
| 2048 | 13824 | 14336 | 99.4 | 527.4 | 5.3x |
| 4096 | 4096 | 4096 | 114.9 | 342.1 | 3.0x |
| 4096 | 14336 | 14336 | 101.9 | 556.1 | 5.5x |

Numbers are wall-clock TFLOPS over warm loops, no fused epilogue.

## Open questions

- Activation side: current A8 is per-token **INT8**. Blackwell FP4 GEMMs want
  **FP8 (E4M3)** activations — do we also switch the activation quant kernel?
  (The cuBLASLt path forces FP4 activations.)
- Per-tensor vs per-output-channel global scale (NVIDIA uses per-tensor for
  weights; keep per-tensor for now).
