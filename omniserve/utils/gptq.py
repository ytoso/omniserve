"""GPTQ weight quantization for the NVFP4 (E2M1 + per-16 UE4M3) format.

Plain round-to-nearest is the dominant accuracy loss in our W4A4 path (see the
perplexity table in README.md). GPTQ fixes it by rounding each column against
the layer's input Hessian ``H = X^T X`` and propagating the rounding error into
the not-yet-quantized columns, so the *layer output* error is minimised rather
than the weight error.

This produces exactly the buffers ``NVFP4Linear`` consumes: 2 E2M1 nibbles per
byte plus one E4M3 scale per 16 input elements, chosen with the same
MSE-optimal scale search as the runtime quantizer.
"""

import torch

from omniserve.utils.nvfp4 import (
    e2m1_to_float,
    e4m3_to_float,
    float_to_e2m1,
    float_to_e4m3,
    pack_e2m1,
)

_SCALE_CANDIDATES = (6.0, 4.5, 3.0)


@torch.no_grad()
def quantize_block(w: torch.Tensor):
    """Quantize a ``[OC, block_size]`` block with a per-row UE4M3 scale.

    Returns ``(nibbles [OC, bs] uint8, scale_bits [OC] uint8, deq [OC, bs])``.
    """
    mx = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9)
    best_err = best_bits = best_s = None
    for c in _SCALE_CANDIDATES:
        bits = float_to_e4m3(mx / c).squeeze(-1)
        s = e4m3_to_float(bits).clamp(min=1e-12)
        rec = e2m1_to_float(float_to_e2m1(w / s.unsqueeze(-1))) * s.unsqueeze(-1)
        err = ((rec - w) ** 2).sum(dim=-1)
        if best_err is None:
            best_err, best_bits, best_s = err, bits, s
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_bits = torch.where(m, bits, best_bits)
            best_s = torch.where(m, s, best_s)
    nib = float_to_e2m1(w / best_s.unsqueeze(-1))
    deq = e2m1_to_float(nib) * best_s.unsqueeze(-1)
    return nib, best_bits, deq


@torch.no_grad()
def gptq_quantize(w: torch.Tensor, H: torch.Tensor, block_size: int = 16,
                  blocksize: int = 128, damp: float = 0.01):
    """GPTQ-quantize ``w [OC, K]`` given the calibration Hessian ``H [K, K]``.

    Returns ``(qweight [OC, K/2] uint8, w_scale [OC, K/block_size] uint8)``.
    """
    W = w.detach().float().clone()
    OC, K = W.shape
    assert K % block_size == 0
    H = H.float().clone()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * torch.diag(torch.diag(H).clone())

    Hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True
    )

    nibbles = torch.zeros(OC, K, dtype=torch.uint8, device=W.device)
    scales = torch.zeros(OC, K // block_size, dtype=torch.uint8, device=W.device)

    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = W[:, i1:i2].clone()
        Hinv1 = Hinv[i1:i2, i1:i2]
        Err1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            col = i1 + i
            if (col - i1) % block_size == 0:
                # (re)fix the UE4M3 scale for this 16-wide group from the
                # current, error-compensated weights
                lo = i
                hi = min(i + block_size, i2 - i1)
                _, bits, _ = quantize_block(W1[:, lo:hi])
                scales[:, col // block_size] = bits
                cur_s = e4m3_to_float(bits).unsqueeze(-1)
                cur_nib = None
            w_col = W1[:, i]
            nib = float_to_e2m1(w_col.unsqueeze(-1) / cur_s)
            q = (e2m1_to_float(nib) * cur_s).squeeze(-1)
            nibbles[:, col] = nib.squeeze(-1)
            Err1[:, i] = (w_col - q) / Hinv1[i, i]
            if i + 1 < (i2 - i1):
                W1[:, i + 1:] -= Err1[:, i].unsqueeze(1) @ Hinv1[i, i + 1:].unsqueeze(0)
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    qweight = pack_e2m1(nibbles)
    return qweight, scales


@torch.no_grad()
def collect_hessian(linear: torch.nn.Linear, xs, device="cuda"):
    """Accumulate ``H = sum(x^T x)`` for a LINEAR from calibration inputs."""
    K = linear.in_features
    H = torch.zeros(K, K, dtype=torch.float32, device=device)
    n = 0
    for x in xs:
        x = x.reshape(-1, K).float().to(device)
        H += x.T @ x
        n += x.shape[0]
    return H / max(n, 1)


_PROJ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")


@torch.no_grad()
def gptq_quantize_model(model, calib_ids, seq_len=512, n_batches=8,
                        block_size=16, blocksize=128, damp=0.01,
                        device="cuda", verbose=True):
    """Sequentially GPTQ-quantize every projection of a Llama-architecture model.

    Layer ``i`` is calibrated with the already-quantized layers ``< i`` in place,
    which is what the deployed model sees. Returns the number of layers done.
    """
    import torch.nn as nn
    from omniserve.modeling.layers.quantized_linear import NVFP4Linear

    layers = model.model.layers
    for li, layer in enumerate(layers):
        linears = [(n, m) for n, m in layer.named_modules()
                   if isinstance(m, nn.Linear) and n.endswith(_PROJ_SUFFIXES)]
        if not linears:
            continue
        Hs = {n: torch.zeros(m.in_features, m.in_features, dtype=torch.float32,
                             device=device) for n, m in linears}
        seen = [0]

        def make_hook(name, mod):
            def hook(_mod, inp, _out):
                x = inp[0].reshape(-1, mod.in_features).float()
                Hs[name] += x.T @ x
                return None
            return hook

        hooks = [m.register_forward_hook(make_hook(n, m)) for n, m in linears]
        for b in range(n_batches):
            chunk = calib_ids[b * seq_len:(b + 1) * seq_len].to(device)
            if chunk.numel() < seq_len:
                break
            model(chunk)
            seen[0] += chunk.numel()
        for h in hooks:
            h.remove()
        if seen[0] == 0:
            raise RuntimeError("no calibration tokens")

        for name, mod in linears:
            H = Hs[name] / seen[0]
            qw, ws = gptq_quantize(mod.weight.data, H, block_size=block_size,
                                   blocksize=blocksize, damp=damp)
            q = NVFP4Linear.from_linear(mod, block_size=block_size)
            q = q.to(device=mod.weight.device, dtype=mod.weight.dtype)
            q.qweight.copy_(qw.to(q.qweight.device))
            q.w_scale.copy_(ws.to(q.w_scale.device))
            q._backend = None
            parent = layer
            parts = name.split(".")[:-1]
            for p in parts:
                parent = getattr(parent, p)
            setattr(parent, name.split(".")[-1], q)
        if verbose:
            print(f"  gptq layer {li + 1}/{len(layers)}", end="\r", flush=True)
    if verbose:
        print()
    return len(layers)
