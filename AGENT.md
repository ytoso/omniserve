# AGENT.md — NVFP4 (W4A8-FP) Experiment Handoff

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
| A — data-format layer (Python/torch) | ✅ done, tested on CPU |
| B — fused FP4 GEMM kernel (CUDA) | ⛔ pending — requires SM100 + CUDA ≥ 12.8 |

The existing INT4/INT8 kernels are **kept untouched** as a fallback; the NVFP4
path is additive.

## Hard constraint (read this first)

NVFP4 (E2M1) tensor-core MMA instructions **only exist on SM100** (Blackwell:
B100/B200/GB200 = compute capability 10.0; RTX 50-series = 12.0). On
Ampere/Hopper/Ada there is no FP4 MMA — FP4 can only be *emulated* by dequant to
FP16, which is the slow path QServe exists to avoid.

Existing kernels target SM 8.0–9.0 and use `mma.sync ... .s8.s8.s32` (int8
tensor cores). They cannot be adapted to FP4; Part B is a new kernel.

Toolchain requirements for Part B:
- CUDA ≥ 12.8 (12.9 for cuBLAS/cuBLASLt FP4 GEMM).
- Recent PyTorch (the repo pins `torch==2.2.0`; Blackwell FP8/FP4 needs 2.7+).
- `kernels/setup.py` `SUPPORTED_ARCHS` currently = `{"8.0","8.6","8.7","8.9","9.0"}`;
  add `"10.0"` (and `"12.0"` if targeting consumer Blackwell).

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

1. **Kernel strategy** (pick one, in increasing effort):
   - cuBLAS / cuBLASLt FP4 GEMM — least code, loses fused epilogue.
   - CUTLASS SM100 FP4 kernels (`cutlass::nv_float4_t` / MXFP4).
   - hand-written SM100 kernel (full rewrite of `qgemm/w4a8_per_*`).
2. Add `omniserve_backend.qgemm_nvfp4` (new `pybind.cpp` + `.cu` under
   `kernels/csrc/qgemm/nvfp4/`, register in `kernels/setup.py`).
3. Dispatch from `NVFP4Linear.forward` (currently a reference dequant+`F.linear`).
4. **Activation side (open decision):** current A8 is per-token **INT8**
   (`fused_kernels.cu::invoke_quant*`). Blackwell FP4 GEMMs want **FP8 (E4M3)**
   activations — decide whether to switch the activation quant kernel too.
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

1. Which Blackwell GPU exactly (B200 vs GB200 vs RTX 50) and CUDA version?
2. cuBLAS vs CUTLASS vs hand-written kernel?
3. Switch activations from INT8 to FP8 E4M3, or keep INT8 activations?
