"""Perplexity evaluation: FP16 vs NVFP4 W4A4 (optionally FP4 attention).

Loads a real HF Llama checkpoint into the NVFP4 model, then reports token-level
perplexity on WikiText-2 (or a local text corpus) using a sliding window.

    python3 scripts/nvfp4_ppl.py --model /tmp/models/SmolLM2-135M \
        --parquet /tmp/data/wikitext2/wikitext-2-raw-v1/test-00000-of-00001.parquet

For reference the script also evaluates the stock transformers model so the
FP16 path of our implementation can be sanity-checked against upstream.
"""

import argparse
import glob
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from omniserve.modeling.models.llama_nvfp4_unpad import LlamaForCausalLM  # noqa: E402


def load_corpus(args):
    if args.parquet:
        import pyarrow.parquet as pq
        table = pq.read_table(args.parquet)
        col = "text" if "text" in table.column_names else table.column_names[0]
        return "\n\n".join(t for t in table[col].to_pylist() if t and t.strip())
    if args.text_dir:
        files = sorted(glob.glob(os.path.join(args.text_dir, "*.txt")))
        return "\n\n".join(open(f, encoding="utf-8", errors="ignore").read()
                           for f in files)
    raise SystemExit("need --parquet or --text-dir")


def to_ids(tokenizer, text):
    return tokenizer(text, return_tensors="pt").input_ids[0]


@torch.no_grad()
def ppl_local(model, ids, window, stride, device="cuda"):
    """Standard sliding-window perplexity (non-overlapping scoring region)."""
    total_nll = 0.0
    total_tok = 0
    n = ids.numel()
    for start in range(0, n - window, stride):
        chunk = ids[start:start + window].to(device)
        target = ids[start + 1:start + window + 1].to(device)
        logits, _ = model(chunk)
        logits = logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), target.view(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tok += target.numel()
    return math.exp(total_nll / total_tok) if total_tok else float("nan")


@torch.no_grad()
def ppl_hf(model, ids, window, stride, device="cuda"):
    total_nll = 0.0
    total_tok = 0
    n = ids.numel()
    for start in range(0, n - window, stride):
        chunk = ids[start:start + window].unsqueeze(0).to(device)
        target = ids[start + 1:start + window + 1].to(device)
        out = model(input_ids=chunk, use_cache=False)
        logits = out.logits[0].float()
        total_nll += torch.nn.functional.cross_entropy(
            logits, target, reduction="sum"
        ).item()
        total_tok += target.numel()
    return math.exp(total_nll / total_tok) if total_tok else float("nan")


def build_local(cfg, state, device, use_nvfp4_attn=False, kv_dtype="fp8",
                quantize=True, quant_acts=True):
    m = LlamaForCausalLM(cfg, use_nvfp4_attn=use_nvfp4_attn,
                         kv_dtype=kv_dtype)
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing:
        print(f"  ! missing keys: {len(missing)} (e.g. {missing[:2]})")
    if unexpected:
        print(f"  ! unexpected keys: {len(unexpected)}")
    if cfg.tie_word_embeddings:
        m.lm_head.weight = m.model.embed_tokens.weight
    m = m.to(device).half().eval()
    if quantize:
        m.replace_linears_with_nvfp4(quant_acts=quant_acts)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/models/SmolLM2-135M")
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--text-dir", default=None)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="truncate the corpus (0 = all)")
    ap.add_argument("--dtype", default="float32",
                    help="dtype for the transformers reference model")
    args = ap.parse_args()

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = load_corpus(args)
    ids = to_ids(tokenizer, text)
    if args.max_tokens:
        ids = ids[: args.max_tokens]
    print(f"corpus: {ids.numel()} tokens, window {args.window}, "
          f"stride {args.stride}\n")

    cfg = AutoConfig.from_pretrained(args.model)
    from safetensors.torch import load_file
    state = load_file(os.path.join(args.model, "model.safetensors"))

    results = []

    # 1) stock transformers (sanity reference)
    hf_dtype = getattr(torch, args.dtype)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=hf_dtype
    ).to(device).eval()
    results.append((f"transformers ({args.dtype})",
                    ppl_hf(hf, ids, args.window, args.stride)))
    del hf
    torch.cuda.empty_cache()

    # 2) our implementation, FP16
    m = build_local(cfg, state, device, quantize=False)
    results.append(("ours FP16", ppl_local(m, ids, args.window, args.stride)))
    del m
    torch.cuda.empty_cache()

    # 3a) W4A16: 4-bit weights, exact activations
    m = build_local(cfg, state, device, quantize=True, quant_acts=False)
    results.append(("W4A16 + FP16 attn",
                    ppl_local(m, ids, args.window, args.stride)))
    del m
    torch.cuda.empty_cache()

    # 3b) W4A4 linears, FP16 attention
    m = build_local(cfg, state, device, quantize=True)
    results.append(("W4A4 + FP16 attn",
                    ppl_local(m, ids, args.window, args.stride)))
    del m
    torch.cuda.empty_cache()

    # 4) W4A4 + FP4 attention, FP8 KV cache
    m = build_local(cfg, state, device, use_nvfp4_attn=True, kv_dtype="fp8",
                    quantize=True)
    results.append(("W4A4 + FP4 attn / FP8 KV",
                    ppl_local(m, ids, args.window, args.stride)))
    del m
    torch.cuda.empty_cache()

    # 5) W4A4 + FP4 attention, FP4 KV cache
    m = build_local(cfg, state, device, use_nvfp4_attn=True, kv_dtype="fp4",
                    quantize=True)
    results.append(("W4A4 + FP4 attn / FP4 KV",
                    ppl_local(m, ids, args.window, args.stride)))

    base = results[1][1]
    print("== perplexity (WikiText-2) ==")
    for name, v in results:
        print(f"  {name:>28}  ppl {v:8.3f}   (+{100 * (v / base - 1):6.2f}% vs FP16)")


if __name__ == "__main__":
    main()
