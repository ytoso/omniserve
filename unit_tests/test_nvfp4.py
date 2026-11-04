"""Standalone CPU tests for the NVFP4 (Part A) data-format layer.

This file loads the NVFP4 modules directly from disk (via ``importlib``) so it
can run without the compiled ``omniserve_backend`` extension and without the
heavy ``omniserve`` package ``__init__`` (which needs ``transformers``).

Run with:  python3 unit_tests/test_nvfp4.py
"""

import importlib.util
import os
import sys
import types

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, relpath))
    assert spec and spec.loader, f"cannot find {relpath}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Dummy packages so `from omniserve.utils.nvfp4 import ...` resolves without
# executing omniserve/__init__.py (which requires transformers / the backend).
_omniserve_pkg = types.ModuleType("omniserve")
_omniserve_pkg.__path__ = []
sys.modules["omniserve"] = _omniserve_pkg
_utils_pkg = types.ModuleType("omniserve.utils")
_utils_pkg.__path__ = [os.path.join(REPO, "omniserve", "utils")]
sys.modules["omniserve.utils"] = _utils_pkg

nvfp4 = _load("omniserve.utils.nvfp4", "omniserve/utils/nvfp4.py")
nvfp4_linear = _load(
    "omniserve.modeling.layers.quantized_linear.nvfp4_linear",
    "omniserve/modeling/layers/quantized_linear/nvfp4_linear.py",
)


def test_e2m1_grid_roundtrip():
    mags = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    vals = torch.cat([mags, -mags[1:]])  # 15 distinct values (+0 and -0 coincide)
    bits = nvfp4.float_to_e2m1(vals)
    assert bits.max() <= 15
    assert torch.allclose(nvfp4.e2m1_to_float(bits), vals, atol=0.0), (
        (bits, nvfp4.e2m1_to_float(bits), vals)
    )


def test_e2m1_rounding():
    x = torch.tensor([0.1, 0.7, 1.3, 5.5, -2.5, 7.0, -7.0])
    dec = nvfp4.e2m1_to_float(nvfp4.float_to_e2m1(x))
    # Values are clamped to [-6, 6] and rounded to the nearest E2M1 magnitude.
    assert torch.allclose(dec, torch.tensor([0.0, 0.5, 1.5, 6.0, -2.0, 6.0, -6.0]))


def test_e4m3_grid_roundtrip():
    vals = torch.tensor(
        [0.0, 1.0, -1.0, 0.5, 240.0, -240.0, 0.015625, 0.001953125, 2.0, 448.0]
    )
    bits = nvfp4.float_to_e4m3(vals)
    dec = nvfp4.e4m3_to_float(bits)
    # 448 saturates to 240.
    assert torch.allclose(dec[:-1], vals[:-1], atol=0.0), (bits, dec, vals)
    assert dec[-1].item() == 240.0


def test_pack_unpack():
    torch.manual_seed(0)
    nibbles = torch.randint(0, 16, (4, 8), dtype=torch.uint8)
    packed = nvfp4.pack_e2m1(nibbles)
    assert packed.shape == (4, 4)
    assert torch.equal(nvfp4.unpack_e2m1(packed), nibbles)


def test_weight_quant_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    qweight, w_scale, global_scale = nvfp4.quantize_weight_nvfp4(w, block_size=16)
    assert qweight.shape == (64, 64)
    assert w_scale.shape == (64, 8)
    deq = nvfp4.dequantize_weight_nvfp4(qweight, w_scale, global_scale)
    abs_err = (deq - w).abs().max().item()
    # Scale-relative error: worst absolute error vs the tensor's max magnitude.
    rel_err = abs_err / w.abs().max().item()
    print(f"[test_weight_quant_roundtrip] max abs err = {abs_err:.3f}, "
          f"scale-rel err = {rel_err:.3f}")
    # FP4 has ~1/3 granularity near the top of its range; 0.5 is a safe bound.
    assert rel_err < 0.5


def test_abs_quant_roundtrip():
    torch.manual_seed(2)
    x = torch.randn(32, 64)
    q, scale = nvfp4.quantize_nvfp4_abs(x, block_size=16)
    assert q.shape == (32, 32)
    assert scale.shape == (32, 4)
    deq = nvfp4.dequantize_nvfp4_abs(q, scale)
    abs_err = (deq - x).abs().max().item()
    rel_err = abs_err / x.abs().max().item()
    print(f"[test_abs_quant_roundtrip] max abs err = {abs_err:.3f}, "
          f"scale-rel err = {rel_err:.3f}")
    assert rel_err < 0.5


def test_linear():
    torch.manual_seed(1)
    lin = torch.nn.Linear(128, 64, bias=True)
    q = nvfp4_linear.NVFP4Linear.from_linear(lin, block_size=16)
    x = torch.randn(8, 128)
    y_ref = lin(x)
    y_q = q(x)
    diff = (y_q.float() - y_ref.float()).abs()
    rel = (diff.norm() / y_ref.float().norm()).item()
    print(f"[test_linear] W4A4 relative output error = {rel:.3f}")
    assert rel < 0.6


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"== {t.__name__} ==")
        t()
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
