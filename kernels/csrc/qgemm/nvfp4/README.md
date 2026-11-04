# NVFP4 (W4A4-FP) cuBLASLt standalone validation

Part B (fused GEMM) standalone probes for the NVFP4 experiment. These are
**plain CUDA/C++ programs** (no PyTorch, no pybind) used to validate and
benchmark the FP4 tensor-core path on a Blackwell GPU before wiring the
Python/pybind layer.

## Why cuBLASLt (and why W4A4, not W4A8)

- The existing `w4a8_per_*` kernels use `mma.sync ... .s8.s8.s32` (INT8 tensor
  cores) and cannot be adapted to FP4.
- cuBLASLt (CUDA ≥ 12.9) is the lowest-code FP4 GEMM path.
- Important finding: **cuBLASLt's block-scaled FP4 kernels require A == B ==
  `CUDA_R_4F_E2M1`** (both operands FP4). FP8 (E4M3) × FP4 (E2M1) is *not*
  exposed by cuBLASLt. So the cuBLASLt path is **W4A4-FP**, not W4A8-FP.
  A true W4A8-FP (FP8 activations × FP4 weights) would need a hand-written
  `mma.sync ... kind::mxf8f6f4 ... .e4m3.e2m1 ...` kernel or CUTLASS.

## Files

| File | Role |
|---|---|
| `fp4_cublaslt_validate.cu` | quantizes random FP32 A/B to E2M1 + per-16 UE4M3 block scales, runs `cublasLtMatmul`, and compares against the dequantized reference |
| `fp4_cublaslt_bench.cu` | FP4 (`CUDA_R_4F_E2M1`, `VEC16_UE4M3`) vs FP16 (`cublasGemmEx`) TFLOPS for LLM-like shapes |
| `fp4_debug.cu` | all-ones sanity probe to isolate data-layout/scale-layout bugs |

## Build / run

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

nvcc -arch=sm_110 -lcublasLt -lcublas fp4_cublaslt_validate.cu -o validate
./validate 256 256 256     # M N K

nvcc -arch=sm_110 -lcublasLt -lcublas fp4_cublaslt_bench.cu -o bench
./bench
```

Use `-arch=sm_100` / `sm_120` on B200 / RTX 50-series respectively. Requires
CUDA ≥ 12.8 (12.9+ for FP4 cuBLASLt), and CUDA ≥ 13.0 for `sm_110`.

## Validation result (NVIDIA Thor, compute capability 11.0, CUDA 13.2)

`./fp4_cublaslt_validate 256 256 256`:

```
heuristic st=0 returned=1
matmul st=0 (success)
rel RMSE vs fp32 = 0.1505 | vs deqref = 0.000000  (shape 256x256x256)
```

- `vs deqref = 0.000000` means the cuBLASLt result matches the dequantized
  E2M1 reference exactly — the data packing, TN layout, and `VEC16_UE4M3`
  tiled scale layout below are all correct.
- The 15% FP32 error is the expected E2M1 quantization error (1 mantissa bit).

## Format contract implemented here

- Data: `CUDA_R_4F_E2M1`, 2 E2M1 values per byte, **low nibble = first
  element** along the innermost (memory-contiguous) dimension.
- GEMM: TN layout (`op(A)=T`, `op(B)=N`), column-major storage.
- Scales: `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` (one unsigned E4M3 per
  16 inner elements), tiled as documented in the cuBLAS "1D Block Scaling
  Factors Layout" section:

  ```text
  tile: 128x4 scales covers a 128x64 matrix block
  within tile: offset = (outer % 32) * 16 + (outer / 32) * 4 + inner
  tiles are row-major over (outer/128) x (inner/64)
  ```

  where `outer` = M (A) / N (B), `inner` = K.

## Throughput (NVIDIA Thor, 20 SMs)

| M | N | K | FP16 TFLOPS | FP4 (W4A4) TFLOPS | speedup |
|---|---|---|---|---|---|
| 1 | 4096 | 4096 | 0.3 | 2.5 | ~8x (mem-bound) |
| 1 | 13824 | 14336 | 0.2 | 0.8 | ~4x (mem-bound) |
| 8 | 4096 | 4096 | 2.1 | 9.7 | 4.6x |
| 128 | 4096 | 4096 | 33.7 | 320.0 | 9.5x |
| 2048 | 4096 | 4096 | 154.5 | 375.5 | 2.4x |
| 2048 | 13824 | 14336 | 99.4 | 527.4 | 5.3x |
| 4096 | 4096 | 4096 | 114.9 | 342.1 | 3.0x |
| 4096 | 14336 | 14336 | 101.9 | 556.1 | 5.5x |

Numbers are wall-clock TFLOPS over warm loops; they include the block-scale
tensor reads but no fused activation epilogue (cuBLASLt loses the fused
epilogue the QServe INT4 kernels have).
