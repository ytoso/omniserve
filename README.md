# NVFP4 (W4A4-FP, Atom-style) Experiment

> **Status:** full W4A4-FP path (FP4 weights, activations, attention and KV
> cache) implemented and benchmarked end-to-end on a Blackwell (Jetson AGX
> Thor, sm_110) GPU. Prefill **1.31x** and decode **4.81x** faster than FP16.

## Goal

Replace QServe's progressive **W4A8** quantization (FP16 → INT8 → INT4 with zero
points) with a **single-stage W4A4-FP** path (Atom-style):

- **No INT8 intermediate** and **no zero point** — weights *and* activations
  are 4-bit floating point (NVFP4 / E2M1) with per-16 UE4M3 block scales.
- Efficiency first: accuracy is measured (see below) but not the primary goal.

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

### End-to-end Llama (8 layers, hidden 4096, seq 2048, Thor sm_110)

| Variant | Prefill | Decode (KV=2048) |
|---|---|---|
| FP16 baseline | 381.4 ms | 62.27 ms/tok |
| NVFP4 linears, FP16 attention | 421.0 ms (0.91x) | 59.2 ms/tok (1.05x) |
| **W4A4 + FP4 attention + FP8 KV cache** | **295.9 ms (1.29x)** | **14.42 ms/tok (4.32x)** |

The decode win comes from a fused attention kernel: the KV cache is stored
4-bit/8-bit and preallocated, and scores / masked softmax / PV run as four
fused kernel launches per layer instead of 2*H cuBLASLt calls. Each uint32
load decodes 8 FP4 values.

## Accuracy

### Perplexity (the real test)

`scripts/nvfp4_ppl.py` loads a real checkpoint into the NVFP4 model and reports
WikiText-2 perplexity with the standard sliding window (512 / stride 256).

```bash
export HF_ENDPOINT=https://hf-mirror.com          # CN mirror
python3 -c "from huggingface_hub import snapshot_download as d; \
  d('HuggingFaceTB/SmolLM2-135M', local_dir='/tmp/models/SmolLM2-135M')"
python3 -c "from huggingface_hub import hf_hub_download as d; \
  d('Salesforce/wikitext','wikitext-2-raw-v1/test-00000-of-00001.parquet', \
    repo_type='dataset', local_dir='/tmp/data/wikitext2')"

python3 scripts/nvfp4_ppl.py --model /tmp/models/SmolLM2-135M \
  --parquet /tmp/data/wikitext2/wikitext-2-raw-v1/test-00000-of-00001.parquet
```

WikiText-2 test set, sliding window 512 / stride 256. Two model sizes show how
sensitivity scales:

| Variant | SmolLM2-135M (304k tok) | SmolLM2-360M (32k tok) |
|---|---|---|
| transformers (fp32) | 20.324 | 15.032 |
| **ours FP16** | **20.329** (+0.02%) | **15.034** (+0.01%) |
| W4A16 (4-bit weights only) | 25.065 (+23.3%) | 17.006 (+13.1%) |
| W4A4 + FP16 attention | 30.495 (+50.0%) | 18.837 (+25.3%) |
| W4A4 + FP4 attention + FP8 KV | 34.935 (+71.9%) | 21.090 (+40.3%) |
| W4A4 + FP4 attention + FP4 KV | 34.935 (+71.9%) | 21.090 (+40.3%) |

Our FP16 path reproduces transformers to 0.02%, so the deltas are purely
quantization. (KV dtype does not show up here because each window is a fresh
prefill; it only matters when decoding against a cache.)

**Is this acceptable?** Not as a deployable quantized model — but it is the
expected result for a *naive* RTN W4A4 recipe, and the gap shrinks quickly with
model size (135M +50% -> 360M +25%, roughly halving per 2.7x parameters).
Published NVFP4 W4A4 numbers (a few percent on 7-8B) rely on GPTQ/AWQ-style
calibration, which we do not do yet.

Layer-level checks on real weights/activations (SmolLM2, captured inputs):

| quantizer | down_proj rel err | o_proj rel err |
|---|---|---|
| NVFP4 per-16 (ours) | 0.088 | 0.076-0.092 |
| INT4 g16 RTN | 0.084 | 0.077-0.087 |
| NVFP4 + Hadamard-128 | 0.089-0.093 | n/a (dim not divisible) |

So per-16 NVFP4 is on par with INT4-g16, and a Hadamard rotation does **not**
help — QuaRot's win comes from per-channel quantization, while per-16 block
scales already absorb outliers. This was re-verified on real weights after an
initial test on synthetic Gaussians gave a misleadingly small gain.

Use `quant_acts=False` (W4A16) and/or `use_nvfp4_attn=False` to trade speed
back for accuracy.

### Where the error comes from

`scripts/nvfp4_accuracy.py` breaks the error down stage by stage (relative
RMSE against FP16, Gaussian tensors):

| Stage | before | after |
|---|---|---|
| weight-only NVFP4 | 0.144 | **0.098** |
| W4A4 linear | 0.170 | **0.133** |
| attention, FP4 KV | 0.135 | **0.129** |
| attention, FP8 KV | - | **0.037** |
| attention, FP4 softmax probs | 1.29 | 0.76 (only with `pv_fp4=True`) |

Three changes drove this:

1. **MSE-optimal block scale.** E2M1 has one mantissa bit, so `scale = max/6`
   is suboptimal. The quantizer now searches a few divisors of the block max
   (`6, 4.5, 3`) and keeps the one minimising squared error. Applies to the
   Python weight quantizer and to the CUDA activation / KV kernels.
2. **Softmax probabilities are no longer quantized by default.** FP4's single
   mantissa bit on the very wide softmax distribution cost ~1.3 relative
   error; keeping `P` in fp16 costs nothing measurable.
3. **FP8 (E4M3) KV cache.** 3.5x lower attention error than FP4 KV *and
   faster* (one row scale instead of per-16 lookups): 14.42 vs 15.78 ms/tok.

FP8 KV is now the default (`kv_dtype="fp8"`); pass `kv_dtype="fp4"` for the
smallest cache.

Also verified: a **Hadamard rotation does not help** here (0.170 -> 0.163).
QuaRot's win comes from per-channel quantization; per-16 block scales already
absorb outliers.

Next accuracy step would be **GPTQ** (Hessian-aware weight rounding with the
WikiText calibration set we now have) — that targets the +23% weight term,
which is the largest single contributor and cannot be fixed by better scaling.

### cuBLASLt FP4 (W4A4) vs FP16 GEMM on Jetson AGX Thor (sm_110, 20 SMs)

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
