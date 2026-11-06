"""Standalone W4A4-FP vs FP16 Llama throughput benchmark.

Builds a LlamaForCausalLM twice (FP16 baseline and NVFP4 W4A4), times
prefill and decode. Uses random weights so no checkpoint is required; the
goal is efficiency, not accuracy.
"""

import argparse
import time

import torch
from transformers import LlamaConfig

from omniserve.modeling.models.llama_nvfp4_unpad import LlamaForCausalLM


def build_models(config, device, use_nvfp4_attn=False, kv_dtype="fp8"):
    torch.manual_seed(0)
    fp16 = LlamaForCausalLM(config).to(device).half().eval()
    nvfp4 = LlamaForCausalLM(config, use_nvfp4_attn=use_nvfp4_attn,
                             kv_dtype=kv_dtype).to(device).half().eval()
    nvfp4.load_state_dict(fp16.state_dict())
    nvfp4.replace_linears_with_nvfp4()
    return fp16, nvfp4


@torch.no_grad()
def run_prefill(model, input_ids):
    logits, _ = model(input_ids)
    return logits


@torch.no_grad()
def run_decode(model, prompt_ids, steps):
    """Prefill `prompt_ids`, then time single-token decode steps."""
    logits, cache = model(prompt_ids)
    lat = []
    for _ in range(steps):
        tok = logits[-1:].argmax(dim=-1)
        t0 = time.perf_counter()
        logits, cache = model(tok, cache)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)
    return lat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--intermediate-size", type=int, default=11008)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--prefill-len", type=int, default=2048)
    p.add_argument("--decode-steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--kv-dtype", type=str, default="fp8", choices=["fp4", "fp8"])
    args = p.parse_args()

    device = "cuda"
    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        rms_norm_eps=1e-6,
    )
    print("building models ...")
    fp16, nvfp4 = build_models(config, device)
    _, nvfp4_attn = build_models(config, device, use_nvfp4_attn=True,
                                 kv_dtype=args.kv_dtype)
    n_params = sum(p.numel() for p in fp16.parameters()) / 1e9
    print(f"params: {n_params:.2f}B")

    input_ids = torch.randint(0, args.vocab_size, (args.prefill_len,), device=device)

    def bench_prefill(model):
        for _ in range(args.warmup):
            run_prefill(model, input_ids)
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            run_prefill(model, input_ids)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        return min(ts)

    fp16_ms = bench_prefill(fp16)
    nvfp4_ms = bench_prefill(nvfp4)
    nvfp4_attn_ms = bench_prefill(nvfp4_attn)
    print(f"\nprefill len={args.prefill_len}:")
    print(f"  fp16        {fp16_ms:.2f} ms")
    print(f"  nvfp4       {nvfp4_ms:.2f} ms  ({fp16_ms/nvfp4_ms:.2f}x)")
    print(f"  nvfp4+attn  {nvfp4_attn_ms:.2f} ms  ({fp16_ms/nvfp4_attn_ms:.2f}x)"
          f"   [kv={args.kv_dtype}]")

    def bench_decode(model):
        for _ in range(args.warmup):
            run_decode(model, input_ids, args.decode_steps)
        torch.cuda.synchronize()
        lat = run_decode(model, input_ids, args.decode_steps)
        return sum(lat) / len(lat)

    fp16_lat = bench_decode(fp16)
    nvfp4_lat = bench_decode(nvfp4)
    nvfp4_attn_lat = bench_decode(nvfp4_attn)
    print(f"decode (KV cache = {args.prefill_len} tokens):")
    print(f"  fp16        {fp16_lat:.2f} ms/tok")
    print(f"  nvfp4       {nvfp4_lat:.2f} ms/tok  ({fp16_lat/nvfp4_lat:.2f}x)")
    print(f"  nvfp4+attn  {nvfp4_attn_lat:.2f} ms/tok  ({fp16_lat/nvfp4_attn_lat:.2f}x)"
          f"   [kv={args.kv_dtype}]")


if __name__ == "__main__":
    main()
