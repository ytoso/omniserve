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
    from omniserve.utils.nvfp4 import pack_e2m1, e2m1_to_float

    H, KH, D, Lk = 32, 8, 128, 2048
    q = torch.randn(1, H, D, device=dev).half()
    k = torch.randn(Lk, KH, D, device=dev).half()
    v = torch.randn(Lk, KH, D, device=dev).half()
    scale = D ** -0.5

    def kv4(t):
        a, b = quantize_nvfp4_abs(t.float().reshape(-1, D), 16)
        return dequantize_nvfp4_abs(a, b).reshape(t.shape)

    def p4(t):
        # quantize the softmax probabilities the same way the kernels do
        a, b = quantize_nvfp4_abs(t.float(), 16)
        return dequantize_nvfp4_abs(a, b)

    def att(kk, vv, pq=False):
        kr = kk.repeat_interleave(H // KH, dim=1)
        vr = vv.repeat_interleave(H // KH, dim=1)
        s = (q.float() @ kr.float().permute(1, 2, 0)) * scale
        p = torch.softmax(s, dim=-1)
        if pq:
            p = p4(p)
        return (p @ vr.float().permute(1, 0, 2)).squeeze(1)

    ref = att(k, v)
    e_k = rel_err(att(kv4(k), v), ref)
    e_kv = rel_err(att(kv4(k), kv4(v)), ref)
    e_all = rel_err(att(kv4(k), kv4(v), pq=True), ref)
    if verbose:
        print("  attention (rel RMSE vs FP16)")
        print(f"    FP4 K only             {e_k:.4f}")
        print(f"    FP4 K + V (default)    {e_kv:.4f}")
        print(f"    + FP4 probabilities    {e_all:.4f}   <- pv_fp4=True")
    return e_k, e_kv, e_all


@torch.no_grad()
def test_e2e(verbose=True):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32000, hidden_size=4096,
                      intermediate_size=11008, num_hidden_layers=4,
                      num_attention_heads=32, num_key_value_heads=8,
                      rms_norm_eps=1e-6)
    fp16 = LlamaForCausalLM(cfg).cuda().half().eval()
    variants = {}
    specs = [
        ("w4a4 + fp16 attn", False, "fp4"),
        ("w4a4 + fp4 attn / fp4 kv", True, "fp4"),
        ("w4a4 + fp4 attn / fp8 kv", True, "fp8"),
    ]
    for name, attn, kv in specs:
        m = LlamaForCausalLM(cfg, use_nvfp4_attn=attn, kv_dtype=kv)
        m = m.cuda().half().eval()
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

    # greedy-token agreement after prefill + one decode step
    tok_ref = l_ref[-1].argmax()
    for name, m in variants.items():
        _, cache = m(ids)
        l2, _ = m(tok_ref.view(1), cache)
        out[name] = out[name] + (int(l2[-1].argmax() == tok_ref),)

    if verbose:
        print("  end-to-end logits (512-token prefill + 1 decode step)")
        for name, (e, c, agree) in out.items():
            print(f"    {name:>28}  rel {e:.4f}  cosine {c:.5f}  "
                  f"next-token match {agree}")
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
