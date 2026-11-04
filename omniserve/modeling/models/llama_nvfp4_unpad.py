"""Llama model with NVFP4 (W4A4-FP, Atom-style) linear layers.

A minimal, self-contained Llama implementation used to benchmark the NVFP4
cuBLASLt GEMM in a realistic transformer stack. It keeps the HF naming scheme
(``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` / ``gate_proj`` /
``up_proj`` / ``down_proj``) so a checkpoint converter can simply replace every
``nn.Linear`` with :class:`NVFP4Linear`.

Attention uses ``torch.nn.functional.scaled_dot_product_attention`` (no
flash-attn dependency) and a trivial growable KV cache for decode.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig

from omniserve.modeling.layers.quantized_linear import NVFP4Linear
from omniserve.modeling.layers.layernorm import RMSNorm


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=False)

    def _reshape(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        return x.view(-1, heads, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        q = self._reshape(self.q_proj(hidden_states), self.num_heads)
        k = self._reshape(self.k_proj(hidden_states), self.num_kv_heads)
        v = self._reshape(self.v_proj(hidden_states), self.num_kv_heads)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=0)
            v = torch.cat([past_v, v], dim=0)
        new_cache = (k, v)

        # GQA: repeat KV heads to match query heads.
        n_rep = self.num_heads // self.num_kv_heads
        k_rep = k.repeat_interleave(n_rep, dim=1)
        v_rep = v.repeat_interleave(n_rep, dim=1)

        attn = F.scaled_dot_product_attention(
            q.transpose(0, 1), k_rep.transpose(0, 1), v_rep.transpose(0, 1),
            is_causal=True,
        ).transpose(0, 1)

        out = self.o_proj(attn.reshape(-1, self.q_size))
        return out, new_cache


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, kv_cache=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_cache = self.self_attn(hidden_states, kv_cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, new_cache


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, kv_caches=None):
        hidden_states = self.embed_tokens(input_ids)
        new_caches = []
        for i, layer in enumerate(self.layers):
            kv_cache = None if kv_caches is None else kv_caches[i]
            hidden_states, new_cache = layer(hidden_states, kv_cache)
            new_caches.append(new_cache)
        hidden_states = self.norm(hidden_states)
        return hidden_states, new_caches


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, kv_caches=None):
        hidden_states, new_caches = self.model(input_ids, kv_caches)
        logits = self.lm_head(hidden_states)
        return logits, new_caches

    @torch.no_grad()
    def replace_linears_with_nvfp4(self):
        """Replace every projection nn.Linear with NVFP4Linear (W4A4)."""
        for name, module in list(self.named_modules()):
            if isinstance(module, nn.Linear) and (
                name.endswith("q_proj") or name.endswith("k_proj")
                or name.endswith("v_proj") or name.endswith("o_proj")
                or name.endswith("gate_proj") or name.endswith("up_proj")
                or name.endswith("down_proj")
            ):
                parent = self
                parts = name.split(".")[:-1]
                for p in parts:
                    parent = getattr(parent, p)
                q = NVFP4Linear.from_linear(module, block_size=16)
                q = q.to(device=module.weight.device, dtype=module.weight.dtype)
                setattr(parent, name.split(".")[-1], q)
        return self
