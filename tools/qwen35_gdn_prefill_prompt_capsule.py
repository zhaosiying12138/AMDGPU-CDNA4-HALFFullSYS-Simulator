#!/usr/bin/env python3
"""Compare the real Qwen3.5-9B layer-0 prefill GDN chain to CPU.

This is deliberately narrower than a model run: it loads only the two prompt
embedding rows and layer-0 GDN weights, then checks projection, causal conv,
chunked recurrent attention, gated norm, and output projection.  It is the
first numerical boundary for a 9B token mismatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/Qwen3.5-9B"
PROMPT_IDS = [248044, 266]
SHARD1 = MODEL / "model.safetensors-00001-of-00004.safetensors"
SHARD3 = MODEL / "model.safetensors-00003-of-00004.safetensors"
SHARD4 = MODEL / "model.safetensors-00004-of-00004.safetensors"


def load_tensor(shards: tuple[Path, ...], name: str) -> torch.Tensor:
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as source:
            if name in source.keys():
                return source.get_tensor(name).contiguous()
    raise KeyError(name)


def load_embedding_rows() -> torch.Tensor:
    with safe_open(str(SHARD1), framework="pt", device="cpu") as source:
        view = source.get_slice("model.language_model.embed_tokens.weight")
        return view[PROMPT_IDS].contiguous()


def metric(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    a = actual.detach().float().cpu().reshape(-1)
    b = expected.detach().float().cpu().reshape(-1)
    delta = a - b
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype).removeprefix("torch."),
        "max_abs_error": float(delta.abs().max().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(delta).item()
            / max(torch.linalg.vector_norm(b).item(), 1e-30)
        ),
        "cosine": float(F.cosine_similarity(a[None], b[None]).item()),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CAPSULE ERROR: simulated CUDA device unavailable", flush=True)
        return 2

    hidden = load_embedding_rows()
    qkv = load_tensor(
        (SHARD3, SHARD4),
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    )
    z_weight = load_tensor(
        (SHARD3, SHARD4),
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
    )
    b_weight = load_tensor(
        (SHARD3, SHARD4),
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
    )
    a_weight = load_tensor(
        (SHARD3, SHARD4),
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
    )
    conv_weight = load_tensor(
        (SHARD3, SHARD4),
        "model.language_model.layers.0.linear_attn.conv1d.weight",
    )
    norm_weight = load_tensor(
        (SHARD3, SHARD4),
        "model.language_model.layers.0.linear_attn.norm.weight",
    )
    out_weight = load_tensor(
        (SHARD3, SHARD4),
        "model.language_model.layers.0.linear_attn.out_proj.weight",
    )
    a_log = load_tensor(
        (SHARD3, SHARD4),
        "model.language_model.layers.0.linear_attn.A_log",
    )
    dt_bias = load_tensor(
        (SHARD3, SHARD4),
        "model.language_model.layers.0.linear_attn.dt_bias",
    )

    # Independent CPU formula follows Transformers Qwen3.5GatedDeltaNet.
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule,
    )

    h = hidden.to(torch.bfloat16)
    qkv_cpu = F.linear(h, qkv)
    z_cpu = F.linear(h, z_weight).reshape(1, 2, 32, 128)
    b_cpu = F.linear(h, b_weight).reshape(1, 2, 32)
    a_cpu = F.linear(h, a_weight).reshape(1, 2, 32)
    mixed_cpu = F.conv1d(
        qkv_cpu.transpose(0, 1).unsqueeze(0),
        conv_weight,
        bias=None,
        padding=3,
        groups=8192,
    )[:, :, :2]
    mixed_cpu = F.silu(mixed_cpu).transpose(1, 2).squeeze(0)
    q_cpu, k_cpu, v_cpu = mixed_cpu.split([2048, 2048, 4096], dim=-1)
    q_cpu = q_cpu.reshape(1, 2, 16, 128)
    k_cpu = k_cpu.reshape(1, 2, 16, 128)
    v_cpu = v_cpu.reshape(1, 2, 32, 128)
    q_cpu = q_cpu.repeat_interleave(2, dim=2)
    k_cpu = k_cpu.repeat_interleave(2, dim=2)
    beta_cpu = b_cpu.sigmoid()
    g_cpu = -a_log.float().exp() * F.softplus(a_cpu.float() + dt_bias)
    core_cpu, _ = torch_chunk_gated_delta_rule(
        q_cpu,
        k_cpu,
        v_cpu,
        g=g_cpu,
        beta=beta_cpu,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.tensor([0, 2], dtype=torch.long),
    )
    core_cpu = core_cpu.reshape(2, 32, 128)
    z_rows_cpu = z_cpu.reshape(2, 32, 128)
    norm_cpu = (
        core_cpu.float()
        * torch.rsqrt(core_cpu.float().square().mean(-1, keepdim=True) + 1e-6)
        * norm_weight.float()
        * (z_rows_cpu.float() * torch.sigmoid(z_rows_cpu.float()))
    ).to(torch.bfloat16)
    out_cpu = F.linear(norm_cpu.reshape(2, 4096), out_weight)

    device = torch.device("cuda")
    h_d = h.to(device)
    qkv_d = qkv.to(device)
    z_weight_d = z_weight.to(device)
    b_weight_d = b_weight.to(device)
    a_weight_d = a_weight.to(device)
    conv_weight_d = conv_weight.to(device)
    norm_weight_d = norm_weight.to(device)
    out_weight_d = out_weight.to(device)
    a_log_d = a_log.to(device)
    dt_bias_d = dt_bias.to(device)
    idx = torch.tensor([0], device=device, dtype=torch.int32)
    cu = torch.tensor([0, 2], device=device, dtype=torch.int32)
    conv_state = torch.zeros((1, 8192, 3), device=device, dtype=torch.bfloat16)
    states = torch.zeros((1, 32, 128, 128), device=device, dtype=torch.float32)

    qkv_d_out = F.linear(h_d, qkv_d)
    z_d = F.linear(h_d, z_weight_d).reshape(1, 2, 32, 128)
    b_d = F.linear(h_d, b_weight_d).reshape(1, 2, 32)
    a_d = F.linear(h_d, a_weight_d).reshape(1, 2, 32)
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    mixed_d = causal_conv1d_fn(
        qkv_d_out.transpose(0, 1).contiguous(),
        conv_weight_d.squeeze(1),
        None,
        conv_states=conv_state,
        query_start_loc=cu,
        seq_lens_cpu=[2],
        cache_indices=idx,
        has_initial_state=torch.tensor([False], device=device),
        activation="silu",
    ).transpose(0, 1)
    q_d, k_d, v_d = mixed_d.split([2048, 2048, 4096], dim=-1)
    q_d = q_d.reshape(1, 2, 16, 128)
    k_d = k_d.reshape(1, 2, 16, 128)
    v_d = v_d.reshape(1, 2, 32, 128)
    beta_d = b_d.sigmoid()
    g_d = -a_log_d.float().exp() * F.softplus(a_d.float() + dt_bias_d)
    core_d, _, _ = TritonGDNKernel().extend(
        q=q_d,
        k=k_d,
        v=v_d,
        g=g_d,
        beta=beta_d,
        ssm_states=states,
        cache_indices=idx,
        query_start_loc=cu,
    )
    torch.cuda.synchronize()
    core_d = core_d.reshape(2, 32, 128)
    norm_d = (
        core_d.float()
        * torch.rsqrt(core_d.float().square().mean(-1, keepdim=True) + 1e-6)
        * norm_weight_d.float()
        * (z_d.float() * torch.sigmoid(z_d.float()))
    ).to(torch.bfloat16)
    out_d = F.linear(norm_d.reshape(2, 4096), out_weight_d)
    torch.cuda.synchronize()

    stages = {
        "in_proj_qkv": metric(qkv_d_out.cpu(), qkv_cpu),
        "in_proj_z": metric(z_d.cpu(), z_cpu),
        "in_proj_b": metric(b_d.cpu(), b_cpu),
        "in_proj_a": metric(a_d.cpu(), a_cpu),
        "causal_conv": metric(mixed_d.cpu(), mixed_cpu),
        "gdn_core": metric(core_d.cpu(), core_cpu),
        "gated_norm": metric(norm_d.cpu(), norm_cpu),
        "out_proj": metric(out_d.cpu(), out_cpu),
    }
    # Projection BF16 noise is bounded; recurrent output needs the same
    # tolerance used by the existing layer-0 capsule.
    tolerances = {
        "in_proj_qkv": 0.02,
        "in_proj_z": 0.02,
        "in_proj_b": 0.02,
        "in_proj_a": 0.02,
        "causal_conv": 0.02,
        "gdn_core": 0.05,
        "gated_norm": 0.05,
        "out_proj": 0.05,
    }
    passed = all(float(stages[name]["relative_l2"]) <= limit for name, limit in tolerances.items())
    report = {
        "schema": "amdgpu-sim.qwen35-gdn-prefill-prompt-capsule.v1",
        "prompt_token_ids": PROMPT_IDS,
        "stages": stages,
        "tolerances": tolerances,
        "passed": passed,
        "reference": "Transformers CPU Qwen3_5GatedDeltaNet formula",
        "tensor_sha256": hashlib.sha256(out_cpu.view(torch.uint16).numpy().tobytes()).hexdigest(),
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
