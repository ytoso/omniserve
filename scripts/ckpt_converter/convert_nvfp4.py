"""Convert an HF Llama FP16 checkpoint to the NVFP4 W4A4 (Atom-style) format.

No calibration is required: weights are quantized single-stage FP16 -> NVFP4
(E2M1) with per-16 UE4M3 block scales. The resulting ``pytorch_model.bin`` has
``qweight`` / ``w_scale`` buffers for every projection and keeps all other
parameters (embedding, norms, lm_head) in FP16.

Usage:
    python convert_nvfp4.py --model-path /path/to/hf --output-path /out
"""

import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from omniserve.modeling.layers.quantized_linear import NVFP4Linear
from quant_utils import get_blocks, get_named_linears, set_op_by_name


def skip(*args, **kwargs):
    pass


torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.kaiming_normal_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="path to the HF model")
    parser.add_argument("--output-path", type=str, required=True,
                        help="directory for the converted checkpoint")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config)

    layers = get_blocks(model)
    for i, layer in enumerate(layers):
        named_linears = get_named_linears(layer)
        for name, module in named_linears.items():
            if not any(name.endswith(s) for s in (
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            )):
                continue
            module = module.to(args.device)
            q_linear = NVFP4Linear.from_linear(module, block_size=args.block_size)
            q_linear = q_linear.to(device=module.weight.device,
                                   dtype=module.weight.dtype)
            set_op_by_name(layer, name, q_linear)

    # Load the FP16 weights for everything that is not a quantized projection.
    # `from_pretrained` would call the patched init functions (no-op), so load
    # the state dict explicitly from the checkpoint shards.
    model = model.cpu()
    # Rebuild quantized modules state_dict by leaving qweight/w_scale in place
    # and copying the remaining HF weights below.
    import transformers.modeling_utils as modeling_utils
    # Suppress weight init already done; just load real weights from disk.
    state = model.state_dict()
    loaded = {}
    from safetensors.torch import load_file
    import glob
    for f in sorted(glob.glob(os.path.join(args.model_path, "*.safetensors"))):
        loaded.update(load_file(f))

    # Map quantized projection weights: the converter stores qweight/w_scale,
    # so ignore the original .weight keys for those modules.
    quantized_suffixes = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    for key, val in loaded.items():
        parts = key.split(".")
        if parts[-1] == "weight" and parts[-2] in quantized_suffixes:
            continue
        if key in state:
            state[key].copy_(val)

    os.makedirs(args.output_path, exist_ok=True)
    torch.save(state, os.path.join(args.output_path, "pytorch_model.bin"))
    print(f"wrote {os.path.join(args.output_path, 'pytorch_model.bin')}")


if __name__ == "__main__":
    main()
