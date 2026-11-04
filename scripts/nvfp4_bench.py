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


def build_models(config, device):
    torch.manual_seed(0)
    fp16 = LlamaForCausalLM(config).to(device).half().eval()
    # Clone the fp16 weights into a fresh model, then quantize linears to NVFP4.
    nvfp4 = LlamaForCausalLM(config).to(device).half().eval()
    nvfp4.load_state_dict(fp16.state_dict())
    nvfp4.replace_linears_with_nvfp4()
    return fp16, nvfp4


@torch.no_grad()
def run_prefill(model, input_ids):
    logits, _ = model(input_ids)
    return logits


@torch.no_grad()
def run_decode(model, input_ids, steps):
    cache = None
    logits, cache = model(input_ids)
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
    n_params = sum(p.numel() for p in fp16.parameters()) / 1e9
    print(f"params: {n_params:.2f}B")

    input_ids = torch.randint(0, args.vocab_size, (args.prefill_len,), device=device)

    # warmup
    for _ in range(args.warmup):
        run_prefill(fp16, input_ids)
        run_prefill(nvfp4, input_ids)
    torch.cuda.synchronize()

    def bench_prefill(model):
        ts = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            run_prefill(model, input_ids)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        return min(ts)

    fp16_ms = bench_prefill(fp16)
    nvfp4_ms = bench_prefill(nvfp4)
    print(f"\nprefill len={args.prefill_len}: fp16 {fp16_ms:.2f} ms | "
          f"nvfp4 {nvfp4_ms:.2f} ms | speedup {fp16_ms/nvfp4_ms:.2f}x")

    # decode
    for _ in range(args.warmup):
        run_decode(fp16, input_ids[:1], args.decode_steps)
        run_decode(nvfp4, input_ids[:1], args.decode_steps)
    fp16_lat = run_decode(fp16, input_ids[:1], args.decode_steps)
    nvfp4_lat = run_decode(nvfp4, input_ids[:1], args.decode_steps)
    fp16_avg = sum(fp16_lat) / len(fp16_lat)
    nvfp4_avg = sum(nvfp4_lat) / len(nvfp4_lat)
    print(f"decode: fp16 {fp16_avg:.2f} ms/tok | nvfp4 {nvfp4_avg:.2f} ms/tok | "
          f"speedup {fp16_avg/nvfp4_avg:.2f}x")


if __name__ == "__main__":
    main()
