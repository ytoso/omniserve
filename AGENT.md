# AGENT.md — NVFP4 (W4A4-FP, Atom-style) Experiment Handoff

> Handoff for the agent continuing this experiment on a Blackwell (SM100) server.

## Mission

Replace QServe's progressive **W4A8** quantization (FP16 → INT8 → INT4 with zero
points) with a **single-stage FP16 → NVFP4** path:

- **No INT8 intermediate** (progressive quantization removed).
- **No zero point** — 4-bit *floating point* (NVFP4 / E2M1) instead of unsigned INT4.
- **Accuracy is ignored**; the goal is to measure efficiency.

## TL;DR status

| Part | Status |
|---|---|
| A — data-format layer (Python/torch) | ✅ W4A4 helpers + MSE-optimal scale search |
| B — fused FP4 GEMM kernel (CUDA) | ✅ `qgemm_nvfp4` cuBLASLt kernel |
| C — NVFP4 attention + KV cache | ✅ fused decode kernels, FP4/FP8 KV |
| D — end-to-end model + benchmark | ✅ `llama_nvfp4_unpad.py`, `scripts/nvfp4_bench.py`, `scripts/nvfp4_accuracy.py` |

The existing INT4/INT8 kernels are **kept untouched** as a fallback; the NVFP4
path is additive.

## Hard constraint (read this first)

NVFP4 (E2M1) tensor-core MMA instructions **only exist on Blackwell**
(B100/B200/GB200 = 10.0, Blackwell Ultra = 11.0, RTX 50-series = 12.0). On
Ampere/Hopper/Ada there is no FP4 MMA — FP4 can only be *emulated* by dequant to
FP16, which is the slow path QServe exists to avoid.

**This machine is a Jetson AGX Thor Developer Kit**: `NVIDIA Thor`,
compute capability **11.0** (SM110), 20 SMs, driver 595.78 / CUDA **13.2**.
CUDA 13.2 toolkit has been installed to `/usr/local/cuda` from NVIDIA's
`ubuntu2404/sbsa` repo (toolkit only, no data-center drivers).

Existing kernels target SM 8.0–9.0 and use `mma.sync ... .s8.s8.s32` (int8
tensor cores). They cannot be adapted to FP4; Part B is a new kernel.

Toolchain requirements for Part B:
- CUDA ≥ 12.8 (12.9 for cuBLAS/cuBLASLt FP4 GEMM); **13.0+ for sm_110**.
- Recent PyTorch (the repo pins `torch==2.2.0`; Blackwell FP8/FP4 needs 2.7+).
- `kernels/setup.py` `SUPPORTED_ARCHS` now includes `"10.0"`, `"11.0"`, `"12.0"`
  and the gencode loop handles 3-digit arch names.

## Part A — what's done

| File | Role |
|---|---|
| `omniserve/utils/nvfp4.py` | E2M1 ↔ float, E4M3 ↔ float, 2-per-byte packing, single-stage weight quantization |
| `omniserve/modeling/layers/quantized_linear/nvfp4_linear.py` | `NVFP4Linear` layer: reference `forward` + `from_linear` |
| `omniserve/modeling/layers/quantized_linear/__init__.py` | exports `NVFP4Linear` unconditionally; backend imports guarded |
| `omniserve/utils/quant_config.py` | `NVFP4QuantConfig` (`get_name() == "nvfp4"`) |
| `unit_tests/test_nvfp4.py` | standalone CPU tests (no pytest/transformers/backend needed) |

## Format spec — the contract Part B must match

Weight `W` with shape `[OC, IC]` (M = OC, K = IC). Dequantization is:

```
W = e2m1(qweight) * e4m3(w_scale) * w_global_scale
```

- `qweight`: `uint8 [OC, IC // 2]` — **two E2M1 values per byte, low nibble =
  first element** (matches NVIDIA NVFP4 packing). Keep this bit-exact when the
  kernel unpacks.
- `w_scale`: `uint8 [OC, IC // block_size]` — per-block **E4M3** scales, one per
  `block_size` (default 16) elements along `IC`. Stored **relative to the global
  scale** (`block_max / w_max ∈ [0, 1]`), i.e. the MXFP4-style two-level scheme.
- `w_global_scale`: `float32` scalar — `w_max / 6`, so the full tensor fits in
  FP4's `[-6, 6]` range.

E2M1 (1 sign + 2 exp + 1 mant, bias 1) positive magnitudes:
`0, 0.5, 1, 1.5, 2, 3, 4, 6`.

E4M3 (1 sign + 4 exp + 3 mant, bias 7): max normal 240, min normal 2^-6,
min subnormal 2^-9.

Single-stage quantization (already implemented in `quantize_weight_nvfp4`):

1. `global_scale = max|W| / 6`
2. per block: `block_scale = max|block| / max|W|`, rounded to E4M3
3. `fp4 = round_to_e2m1(W / (block_scale * global_scale))`, then nibble-pack

> ⚠️ The kernel author must reproduce this *exact* dequant in the epilogue, or
> swap `from_linear`/`quantize_weight_nvfp4` to whatever layout the kernel wants.
> Right now `nvfp4_linear.py::from_linear` is the single source of truth.

## Part B — remaining work

1. **Kernel strategy** — validated the **cuBLASLt FP4 GEMM** path first
   (least code). See `kernels/csrc/qgemm/nvfp4/`:
   - `fp4_cublaslt_validate.cu`: `CUDA_R_4F_E2M1` × `CUDA_R_4F_E2M1`,
     `VEC16_UE4M3` block scales, TN layout. Matches dequantized reference
     **exactly** (`vs deqref = 0.000000`).
   - `fp4_cublaslt_bench.cu`: FP4 vs FP16 throughput on Thor (see README there).
   - ⚠️ **cuBLASLt FP4 = W4A4** (A and B both FP4). cuBLASLt does **not**
     expose FP8 (E4M3) × FP4 (E2M1). A true W4A8-FP needs a hand-written
     `mma.sync ... kind::mxf8f6f4 ... .e4m3.e2m1 ...` kernel (or CUTLASS).
   - ⚠️ CUDA 13.2 ptxas only accepts those warp-level `mma.sync` FP4 forms on
     **`sm_120a`**; they are rejected on `sm_100`/`sm_110`/`sm_120`/`sm_121`.
     On this Thor (`sm_110`) FP4 tensor cores are reachable only via
     **tcgen05** (5th-gen tensor cores) — i.e., what cuBLASLt uses internally.
2. Add `omniserve_backend.qgemm_nvfp4` (new `pybind.cpp` + `.cu` under
   `kernels/csrc/qgemm/nvfp4/`, register in `kernels/setup.py`). The standalone
   cuBLASLt probes are already there; the pybind wrapper is the next step.
3. Dispatch from `NVFP4Linear.forward` — ✅ `omniserve_backend.qgemm_nvfp4`
   now dispatches the cuBLASLt W4A4 path; kernel matches reference bit-exactly.
4. **Activation side:** now **W4A4** — activations are quantized to FP4 (E2M1)
   with per-16 UE4M3 scales (Atom-style), matching cuBLASLt's requirement.
5. Wire `NVFP4Linear` into a model (clone `llama_w4a8_unpad.py` → e.g.
   `llama_nvfp4_unpad.py`) and add a checkpoint converter (direct HF FP16 →
   NVFP4, no calibration needed since accuracy is aside).
6. Benchmark prefill/decoding throughput vs the W4A8 INT4 baseline.

## Validation (runs anywhere, no backend needed)

```bash
python3 unit_tests/test_nvfp4.py
```

Expected (CPU, torch 2.8):
- E2M1/E4M3 round-trips exact on representable grids.
- weight round-trip: max abs err 0.541, scale-rel err 0.132.
- `nn.Linear` vs `NVFP4Linear`: **10.4%** relative output error.

## Repo / git

- Upstream: `https://github.com/mit-han-lab/omniserve.git` (read-only here).
- Fork (push target): `git@github.com:ytoso/omniserve.git`, remote name `fork`,
  branch `main`.
- Latest commit: `e17b6e4` "Add NVFP4 (W4A8-FP) quantization...".
- `README.md` = this experiment's progress doc; original README backed up as
  `README.md.bak`.
- Ignore the `pi-session-*.html` file in the working tree (session artifact).

## Gotchas

- `omniserve/__init__.py` imports the engine (needs `transformers`, the compiled
  backend). The NVFP4 util/layer files are pure-torch; import them directly or
  via the guarded `quantized_linear/__init__.py`.
- The reference `forward` dequantizes every call — correct but not fast. It is
  intentionally only a placeholder until the kernel lands.
- Per-tensor vs per-output-channel `w_global_scale`: NVIDIA uses per-tensor for
  weights; the current code is per-tensor. Keep unless the kernel wants otherwise.

## Open questions for the next agent

1. ~~Which Blackwell GPU exactly~~ → **Jetson AGX Thor, sm_110, CUDA 13.2**.
2. cuBLAS vs CUTLASS vs hand-written kernel? → **cuBLASLt validated (W4A4)**;
   for true W4A8-FP use a hand-written `mxf8f6f4 .e4m3.e2m1` kernel or CUTLASS.
3. Switch activations from INT8 to FP8 E4M3, or keep INT8 activations?
   (cuBLASLt path forces FP4 activations.)
