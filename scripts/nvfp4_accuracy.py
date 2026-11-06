"""Accuracy harness for the NVFP4 W4A4 path.

Measures the error introduced at each stage against an FP16 reference so
improvements can be attributed and verified:

  1. weight-only NVFP4
  2. weight + activation NVFP4 (W4A4 linear)
  3. + FP4 KV cache in attention
  4. + FP4 softmax probabilities (full W4A4 attention)
  5. end-to-end LM logit agreement

Run:  python3 scripts/nvfp4_accuracy.py
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import LlamaConfig  # noqa: E402

from omniserve.modeling.layers.quantized_linear import NVFP4Linear  # noqa: E402
from omniserve.modeling.models.llama_nvfp4_unpad import LlamaForCausalLM  # noqa: E402
from omniserve.utils.nvfp4 import (  # noqa: E402
    dequantize_nvfp4_abs,
    quantize_nvfp4_abs,
)


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_linear(verbose=True):
    torch.manual_seed(0)
    dev = "cuda"
    rows = []
    for K, OC, M in [(4096, 4096, 512), (4096, 11008, 512), (11008, 4096, 512),
                     (4096, 4096, 1)]:
        lin = torch.nn.Linear(K, OC, bias=False).to(dev).half()
        x = torch.randn(M, K, device=dev).half()

        # weight-only NVFP4 (activations kept exact)
        qw, sw = quantize_nvfp4_abs(lin.weight.data.float(), 16)
        w_deq = dequantize_nvfp4_abs(qw, sw).half()
        y_wonly = torch.nn.functional.linear(x, w_deq)

        # W4A4
        q = NVFP4Linear.from_linear(lin, block_size=16).to(dev).half()
        y_w4a4 = q(x)

        y_ref = torch.nn.functional.linear(x, lin.weight.data)
        rows.append((f"{M}x{K}->{OC}", rel_err(y_wonly, y_ref),
                     rel_err(y_w4a4, y_ref)))
    if verbose:
        print("  linear (rel RMSE vs FP16)")
        for name, w, wa in rows:
            print(f"    {name:>18}  w-only {w:.4f}   w4a4 {wa:.4f}")
    return rows


@torch.no_grad()
def test_attention(verbose=True):
    torch.manual_seed(0)
    dev = "cuda"
    import omniserve_backend.qgemm_nvfp4 as B

    H, KH, D, Lk = 32, 8, 128, 2048
    q = torch.randn(1, H, D, device=dev).half()
    k = torch.randn(Lk, KH, D, device=dev).half()
    v = torch.randn(Lk, KH, D, device=dev).half()
    scale = D ** -0.5

    kr = k.repeat_interleave(H // KH, dim=1)
    vr = v.repeat_interleave(H // KH, dim=1)
    s = (q.float() @ kr.float().permute(1, 2, 0)) * scale
    p = torch.softmax(s, dim=-1)
    ref = (p @ vr.float().permute(1, 0, 2)).squeeze(1)

    # FP4 KV cache + fused FP4 decode attention
    kp, ks = B.kv_quantize(k)
    vp, vs = B.kv_quantize(v)
    out = B.decode_attention(q[0], kp, ks, vp, vs, scale)

    # FP4 KV cache only (exact softmax / PV in fp32)
    kd = dequantize_nvfp4_abs(
        kp.reshape(Lk * KH, D // 2), ks.reshape(Lk * KH, D // 16)
    ).reshape(Lk, KH, D)
    vd = dequantize_nvfp4_abs(
        vp.reshape(Lk * KH, D // 2), vs.reshape(Lk * KH, D // 16)
    ).reshape(Lk, KH, D)
    krd = kd.repeat_interleave(H // KH, dim=1)
    vrd = vd.repeat_interleave(H // KH, dim=1)
    s2 = (q.float() @ krd.permute(1, 2, 0)) * scale
    p2 = torch.softmax(s2, dim=-1)
    mid = (p2 @ vrd.permute(1, 0, 2)).squeeze(1)

    e_kv = rel_err(mid, ref)
    e_all = rel_err(out, ref)
    if verbose:
        print("  attention (rel RMSE vs FP16)")
        print(f"    FP4 KV cache only     {e_kv:.4f}")
        print(f"    + FP4 probabilities   {e_all:.4f}")
    return e_kv, e_all


@torch.no_grad()
def test_e2e(verbose=True):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32000, hidden_size=4096,
                      intermediate_size=11008, num_hidden_layers=4,
                      num_attention_heads=32, num_key_value_heads=8,
                      rms_norm_eps=1e-6)
    fp16 = LlamaForCausalLM(cfg).cuda().half().eval()
    variants = {}
    for name, attn in [("w4a4 + fp16 attn", False), ("w4a4 + fp4 attn", True)]:
        m = LlamaForCausalLM(cfg, use_nvfp4_attn=attn).cuda().half().eval()
        m.load_state_dict(fp16.state_dict())
        m.replace_linears_with_nvfp4()
        variants[name] = m

    ids = torch.randint(0, 32000, (512,), device="cuda")
    l_ref, _ = fp16(ids)
    out = {}
    for name, m in variants.items():
        l, _ = m(ids)
        cos = torch.nn.functional.cosine_similarity(
            l[-1].float(), l_ref[-1].float(), dim=0
        ).item()
        out[name] = (rel_err(l, l_ref), cos)
    if verbose:
        print("  end-to-end logits (512-token prefill)")
        for name, (e, c) in out.items():
            print(f"    {name:>20}  rel {e:.4f}  cosine {c:.5f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-e2e", action="store_true")
    args = ap.parse_args()
    print("== accuracy ==")
    test_linear()
    test_attention()
    if not args.skip_e2e:
        test_e2e()


if __name__ == "__main__":
    main()
