#!/usr/bin/env python3
"""Differential capsule for the two GDN decode-only Triton kernels.

The decode-phase layer gate (CP-0092) proved decode step 1's layer-0 INPUT
is bit-exact against the NVIDIA golden while its OUTPUT diverges
(rel_l2 1.42, cos 0.37, no clean scale factor).  Layer 0 is a GDN layer and
its decode path runs exactly two Triton kernels the prefill never executes:

  1. ``_causal_conv1d_update_kernel`` via
     ``sglang.kernels.ops.mamba.causal_conv1d_triton.causal_conv1d_update``
  2. ``fused_recurrent_gated_delta_rule_packed_decode_kernel`` via
     ``sglang.kernels.ops.attention.fla.fused_recurrent.
     fused_recurrent_gated_delta_rule_packed_decode``

This capsule runs each kernel standalone through the normal Triton launch
path on the simulator (import the SGLang wrapper, call it with device
tensors -- same as tools/qwen35_aiter_attn_capsule.py) with deterministic
seeded synthetic activations, real layer-0 weights (conv1d / A_log /
dt_bias) and multi-slot state pools with a *non-zero* selected slot, then
compares outputs and in-place state updates against independent CPU
float32 references written from the algorithm (below), plus hypothesis
tables and per-workgroup error maps to characterise any wrong pattern.

CPU float32 references (documented from the Triton source):

conv update (decode: seqlen=1, KERNEL_WIDTH=4, no bias, silu; window
state_len=3; ``w = conv1d.weight[d, :4]``, ``s = conv_state[slot, d, :3]``):
    acc[d]  = s[0]*w[0] + s[1]*w[1] + s[2]*w[2] + x[d]*w[3]
    out[d]  = silu(acc) = acc / (1 + exp(-acc))          (stored bf16)
    state'[slot, d, :] = [s[1], s[2], x[d]]              (left shift)

packed recurrent decode (single-token gated delta rule recurrence; per
batch row n, v-head hv; q/k head i_h = hv // (HV//H)):
    q = mixed[n, i_h*K:(i_h+1)*K];  k = mixed[n, H*K + i_h*K : ...]
    v = mixed[n, 2*H*K + hv*V : 2*H*K + (hv+1)*V]        (all -> fp32)
    q = q / sqrt(sum(q^2) + 1e-6); k = k / sqrt(sum(k^2) + 1e-6); q *= scale
    g    = -exp(A_log[hv]) * softplus(a[n,hv] + dt_bias[hv]),
           softplus(t) = log(1+exp(t)) if t <= 20 else t
    beta = sigmoid(b[n,hv]) rounded through bf16
    S = ssm_state[slot, hv]                              (fp32 [V, K])
    S *= exp(g)
    u = v - S @ k;  u *= beta;  S += outer(u, k)
    o = S @ q
    out[n, 0, hv] = o (bf16);  state'[slot, hv] = S (fp32)

Layer-0 geometry (Qwen3.5-0.8B, TP1): H=HV=16, K=V=128, conv dim 6144,
width 4, scale = K**-0.5.  Production grids: conv (1, 24) num_warps 4,
packed decode (NV=4, B*HV) num_warps 1.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]

# Qwen3.5-0.8B layer-0 GDN geometry (TP1).
H = 16
HV = 16
K = 128
V = 128
DIM = 2 * H * K + HV * V  # 6144 packed q|k|v
WIDTH = 4  # linear_conv_kernel_dim
STATE_LEN = WIDTH - 1  # 3
SCALE = float(K) ** -0.5
MODEL_FILE = ROOT / "models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
LAYER_PREFIX = "model.language_model.layers.0.linear_attn"


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    a = actual.detach().float().reshape(-1)
    b = reference.detach().float().reshape(-1)
    delta = a - b
    denom = torch.linalg.vector_norm(b).item()
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype).removeprefix("torch."),
        "max_abs_error": float(delta.abs().max().item()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(delta).item() / max(denom, 1.0e-30)
        ),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(a[None], b[None]).item()
        ),
        "bitexact_fraction": (
            float(
                (
                    actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
                    == reference.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
                )
                .float()
                .mean()
                .item()
            )
            if actual.dtype == reference.dtype
            else None
        ),
        "actual_norm": float(torch.linalg.vector_norm(a).item()),
        "reference_norm": float(torch.linalg.vector_norm(b).item()),
    }


def _sha(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# CPU float32 reference: causal_conv1d_update (decode path)
# ---------------------------------------------------------------------------
def ref_conv_update(
    x: torch.Tensor,
    pool: torch.Tensor,
    weight: torch.Tensor,
    indices: torch.Tensor,
    variant: str = "correct",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (out fp32 [bs, DIM], new_pool same dtype/shape as pool).

    Kernel-faithful numerics: each tap product is rounded to bf16 (Triton
    computes bf16*bf16 -> bf16) and accumulated in fp32; silu in fp32; the
    updated state keeps bf16 inputs verbatim.
    """
    bs = x.shape[0]
    out = torch.empty(bs, DIM, dtype=torch.float32)
    new_pool = pool.clone()
    for n in range(bs):
        slot = 0 if variant == "slot0_state" else int(indices[n])
        s = pool[slot].float()  # [DIM, 3]
        w = weight.float()  # [DIM, 4]
        xf = x[n].float()
        taps = torch.stack(
            [s[:, 0] * w[:, 0], s[:, 1] * w[:, 1], s[:, 2] * w[:, 2], xf * w[:, 3]],
            dim=1,
        )
        if variant == "reversed_window":
            taps = torch.stack(
                [xf * w[:, 0], s[:, 2] * w[:, 1], s[:, 1] * w[:, 2], s[:, 0] * w[:, 3]],
                dim=1,
            )
        elif variant == "reversed_weights":
            taps = torch.stack(
                [s[:, 0] * w[:, 3], s[:, 1] * w[:, 2], s[:, 2] * w[:, 1], xf * w[:, 0]],
                dim=1,
            )
        elif variant == "no_x_term":
            taps = taps[:, :3]
        taps = taps.to(torch.bfloat16).float()
        acc = taps.sum(dim=1)
        out[n] = acc if variant == "no_silu" else acc / (1.0 + torch.exp(-acc))
        if variant == "stale_state_write":
            continue
        new_pool[slot, :, 0] = pool[slot, :, 1]
        new_pool[slot, :, 1] = pool[slot, :, 2]
        new_pool[slot, :, 2] = x[n]
    return out, new_pool


# ---------------------------------------------------------------------------
# CPU float32 reference: fused_recurrent_gated_delta_rule_packed_decode
# ---------------------------------------------------------------------------
def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 20.0, torch.log(1.0 + torch.exp(x)), x)


def ref_packed_decode(
    mixed: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    pool: torch.Tensor,
    indices: torch.Tensor,
    variant: str = "correct",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (out fp32 [bs, HV, V], new_pool fp32 clone of pool).

    All arithmetic in fp32 mirroring the kernel's explicit .to(tl.float32)
    casts; beta is rounded through bf16 exactly like the kernel; final out
    and state are stored as the kernel stores them (bf16 out / fp32 state).
    """
    bs = mixed.shape[0]
    out = torch.empty(bs, HV, V, dtype=torch.float32)
    new_pool = pool.clone()
    for n in range(bs):
        slot = 0 if variant == "state_slot0" else int(indices[n])
        for hv in range(HV):
            i_h = hv // (HV // H)
            q = mixed[n, i_h * K : (i_h + 1) * K].float()
            k = mixed[n, H * K + i_h * K : H * K + (i_h + 1) * K].float()
            v = mixed[n, 2 * H * K + hv * V : 2 * H * K + (hv + 1) * V].float()
            if variant != "no_l2norm":
                q = q / torch.sqrt(torch.sum(q * q) + 1e-6)
                k = k / torch.sqrt(torch.sum(k * k) + 1e-6)
            q = q * SCALE
            g = -torch.exp(A_log[hv].float()) * _softplus(
                a[n, hv].float() + dt_bias[hv].float()
            )
            beta = torch.sigmoid(b[n, hv].float()).to(torch.bfloat16).float()
            if variant == "no_decay":
                g = torch.zeros_like(g)
            if variant == "beta_one":
                beta = torch.ones_like(beta)
            S0 = pool[slot, hv].float()
            if variant == "state_T":
                S0 = S0.t().contiguous()
            S = S0 * torch.exp(g)
            u = v - S @ k
            u = u * beta
            S = S + torch.outer(u, k)
            o = S @ (k if variant == "out_dot_k" else q)
            if variant == "out_from_decayed":
                o = (S0 * torch.exp(g)) @ q
            out[n, hv] = o
            new_pool[slot, hv] = S
    return out, new_pool


def _hypothesis_table(
    sim: torch.Tensor,
    hypotheses: dict[str, torch.Tensor],
) -> dict[str, float]:
    """rel_l2 of the simulator result against each alternative reference."""
    table = {}
    sim_f = sim.detach().float().reshape(-1)
    for name, ref in hypotheses.items():
        ref_f = ref.detach().float().reshape(-1)
        delta = sim_f - ref_f
        table[name] = float(
            torch.linalg.vector_norm(delta).item()
            / max(torch.linalg.vector_norm(ref_f).item(), 1.0e-30)
        )
    return dict(sorted(table.items(), key=lambda kv: kv[1]))


def _block_errors(sim: torch.Tensor, ref: torch.Tensor, n_blocks: int) -> list[float]:
    s = sim.detach().float().reshape(-1)
    r = ref.detach().float().reshape(-1)
    chunk = s.numel() // n_blocks
    return [
        float((s[i * chunk : (i + 1) * chunk] - r[i * chunk : (i + 1) * chunk])
              .abs()
              .max()
              .item())
        for i in range(n_blocks)
    ]


# ---------------------------------------------------------------------------
# Minimal Triton op probes (normal compile path; mirrors the packed decode
# kernel's op mix: single-wave num_warps=1 reductions + transcendentals).
# ---------------------------------------------------------------------------
def _build_op_probes():
    import triton
    import triton.language as tl

    @triton.jit
    def _rowsum_kernel(x_ptr, o_ptr, BV: tl.constexpr, BK: tl.constexpr):
        # Same structure as b_h * b_k[None, :] -> tl.sum(axis=1) in the
        # packed decode kernel: [BV, BK] fp32 block, reduce along the
        # 128-wide axis, num_warps=1.
        o_v = tl.arange(0, BV)
        o_k = tl.arange(0, BK)
        x = tl.load(x_ptr + o_v[:, None] * BK + o_k[None, :])
        tl.store(o_ptr + o_v, tl.sum(x, 1))

    @triton.jit
    def _vecsum_kernel(x_ptr, o_ptr, BK: tl.constexpr):
        # Same structure as the l2norm sum(q*q) reduction: 128-wide fp32.
        o_k = tl.arange(0, BK)
        x = tl.load(x_ptr + o_k)
        tl.store(o_ptr, tl.sum(x, 0))

    @triton.jit
    def _trans_kernel(x_ptr, y_ptr, e_ptr, sp_ptr, sg_ptr, sq_ptr, N: tl.constexpr):
        i = tl.arange(0, N)
        x = tl.load(x_ptr + i)
        y = tl.load(y_ptr + i)
        tl.store(e_ptr + i, tl.exp(x))
        tl.store(sp_ptr + i, tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x))
        tl.store(sg_ptr + i, tl.sigmoid(x))
        tl.store(sq_ptr + i, tl.sqrt(y))

    return _rowsum_kernel, _vecsum_kernel, _trans_kernel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            __import__("os").environ.get(
                "QWEN35_GDN_DECODE_DIFF_OUTPUT",
                str(ROOT / "artifacts/qwen35-gdn-decode-diff/20260821-v1"),
            )
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--slots", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        print("CAPSULE ERROR: simulated HIP torch device is unavailable", flush=True)
        return 2
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    weights = load_file(str(MODEL_FILE), device="cpu")
    conv_w = weights[f"{LAYER_PREFIX}.conv1d.weight"].view(DIM, WIDTH).contiguous()
    A_log = weights[f"{LAYER_PREFIX}.A_log"].contiguous()  # fp32 [16]
    dt_bias = weights[f"{LAYER_PREFIX}.dt_bias"].contiguous()  # bf16 [16]
    assert conv_w.shape == (DIM, WIDTH) and conv_w.dtype == torch.bfloat16
    assert A_log.dtype == torch.float32 and dt_bias.dtype == torch.bfloat16

    # Deterministic synthetic activations (layer-0 magnitudes).
    slots = args.slots
    x_pre = torch.randn(1, DIM).to(torch.bfloat16)  # in_proj mixed_qkv row
    conv_pool = (torch.randn(slots, DIM, STATE_LEN) * 0.7).to(torch.bfloat16)
    ssm_pool = (torch.randn(slots, HV, V, K) * 0.3).to(torch.float32)
    a_gating = torch.randn(1, HV).to(torch.bfloat16)
    b_gating = torch.randn(1, HV).to(torch.bfloat16)
    idx1 = torch.tensor([1], dtype=torch.int32)  # non-zero slot on purpose
    idx2 = torch.tensor([0, slots - 1], dtype=torch.int32)

    # Secondary bs=2 inputs (index-structure probe).
    x_pre2 = torch.randn(2, DIM).to(torch.bfloat16)
    conv_pool2 = (torch.randn(slots, DIM, STATE_LEN) * 0.7).to(torch.bfloat16)
    ssm_pool2 = (torch.randn(slots, HV, V, K) * 0.3).to(torch.float32)
    a2 = torch.randn(2, HV).to(torch.bfloat16)
    b2 = torch.randn(2, HV).to(torch.bfloat16)

    # Import the production wrappers only after the lane env is up.
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    results: dict[str, object] = {
        "schema": "amdgpu-sim.qwen35-gdn-decode-diff.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "seed": args.seed,
            "geometry": {"H": H, "HV": HV, "K": K, "V": V, "DIM": DIM, "WIDTH": WIDTH},
            "scale": SCALE,
            "slots": slots,
            "weights": {
                "conv1d": _sha(conv_w),
                "A_log": _sha(A_log),
                "dt_bias": _sha(dt_bias),
                "A_log_range": [
                    float(A_log.min().item()),
                    float(A_log.max().item()),
                ],
            },
        },
    }
    tensors: dict[str, torch.Tensor] = {}

    def _run_conv(tag, x_cpu, pool_cpu, idx_cpu):
        t0 = time.time()
        ref_out, ref_pool = ref_conv_update(x_cpu, pool_cpu, conv_w, idx_cpu)
        x_d = x_cpu.to(device)
        pool_d = pool_cpu.clone().to(device)
        idx_d = idx_cpu.to(device)
        out_d = causal_conv1d_update(
            x_d,
            pool_d,
            conv_w.to(device),
            None,
            "silu",
            conv_state_indices=idx_d,
        )
        torch.cuda.synchronize()
        sim_out = out_d.detach().cpu()
        sim_pool = pool_d.detach().cpu()
        wall = time.time() - t0
        sel = [int(i) for i in idx_cpu]
        untouched_ok = all(
            torch.equal(sim_pool[s], pool_cpu[s])
            for s in range(slots)
            if s not in sel
        )
        entry = {
            "wall_seconds": round(wall, 3),
            "out_sim_vs_ref_fp32": _metrics(sim_out, ref_out),
            "out_sim_vs_ref_bf16store": _metrics(
                sim_out, ref_out.to(torch.bfloat16)
            ),
            "state_sim_vs_ref": _metrics(
                sim_pool[sel].float(), ref_pool[sel].float()
            ),
            "untouched_slots_bitexact": bool(untouched_ok),
            "out_block_maxabs_24x256": _block_errors(sim_out, ref_out, DIM // 256),
        }
        hypotheses = {}
        for variant in (
            "slot0_state",
            "no_silu",
            "no_x_term",
            "reversed_window",
            "reversed_weights",
        ):
            alt_out, _ = ref_conv_update(x_cpu, pool_cpu, conv_w, idx_cpu, variant)
            hypotheses[variant] = alt_out
        entry["out_hypothesis_rel_l2"] = _hypothesis_table(sim_out, hypotheses)
        _, stale_pool = ref_conv_update(
            x_cpu, pool_cpu, conv_w, idx_cpu, "stale_state_write"
        )
        entry["state_hypothesis_rel_l2"] = _hypothesis_table(
            sim_pool[sel], {"correct": ref_pool[sel], "state_unrolled": stale_pool[sel]}
        )
        tensors.update(
            {
                f"conv_{tag}_x": x_cpu.contiguous(),
                f"conv_{tag}_pool_before": pool_cpu.contiguous(),
                f"conv_{tag}_sim_out": sim_out.contiguous(),
                f"conv_{tag}_ref_out": ref_out.to(torch.bfloat16).contiguous(),
                f"conv_{tag}_sim_pool": sim_pool.contiguous(),
            }
        )
        return entry, sim_out

    def _run_recurrent(tag, mixed_cpu, a_cpu, b_cpu, pool_cpu, idx_cpu):
        t0 = time.time()
        ref_out, ref_pool = ref_packed_decode(
            mixed_cpu, a_cpu, b_cpu, A_log, dt_bias, pool_cpu, idx_cpu
        )
        mixed_d = mixed_cpu.to(device)
        a_d = a_cpu.to(device)
        b_d = b_cpu.to(device)
        pool_d = pool_cpu.clone().to(device)
        idx_d = idx_cpu.to(device)
        out_d = mixed_d.new_empty(mixed_d.shape[0], 1, HV, V)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_d,
            a=a_d,
            b=b_d,
            A_log=A_log.to(device),
            dt_bias=dt_bias.to(device),
            scale=SCALE,
            initial_state=pool_d,
            out=out_d,
            ssm_state_indices=idx_d,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()
        sim_out = out_d.detach().cpu()
        sim_pool = pool_d.detach().cpu()
        wall = time.time() - t0
        sel = [int(i) for i in idx_cpu]
        untouched_ok = all(
            torch.equal(sim_pool[s], pool_cpu[s])
            for s in range(slots)
            if s not in sel
        )
        entry = {
            "wall_seconds": round(wall, 3),
            "out_sim_vs_ref_fp32": _metrics(sim_out.squeeze(1), ref_out),
            "out_sim_vs_ref_bf16store": _metrics(
                sim_out.squeeze(1), ref_out.to(torch.bfloat16)
            ),
            "state_sim_vs_ref": _metrics(
                sim_pool[sel], ref_pool[sel]
            ),
            "untouched_slots_bitexact": bool(untouched_ok),
            "out_head_maxabs_16": _block_errors(sim_out.squeeze(1), ref_out, HV),
            "out_vblock_maxabs_4": _block_errors(
                sim_out.squeeze(1).reshape(-1), ref_out.reshape(-1), 4
            ),
        }
        hypotheses = {}
        for variant in (
            "no_decay",
            "beta_one",
            "no_l2norm",
            "out_from_decayed",
            "state_slot0",
            "state_T",
            "out_dot_k",
        ):
            alt_out, _ = ref_packed_decode(
                mixed_cpu, a_cpu, b_cpu, A_log, dt_bias, pool_cpu, idx_cpu, variant
            )
            hypotheses[variant] = alt_out
        entry["out_hypothesis_rel_l2"] = _hypothesis_table(
            sim_out.squeeze(1), hypotheses
        )
        tensors.update(
            {
                f"rec_{tag}_mixed": mixed_cpu.contiguous(),
                f"rec_{tag}_a": a_cpu.contiguous(),
                f"rec_{tag}_b": b_cpu.contiguous(),
                f"rec_{tag}_pool_before": pool_cpu.contiguous(),
                f"rec_{tag}_sim_out": sim_out.contiguous(),
                f"rec_{tag}_ref_out": ref_out.to(torch.bfloat16).contiguous(),
                f"rec_{tag}_sim_pool": sim_pool.contiguous(),
                f"rec_{tag}_ref_pool": ref_pool.contiguous(),
            }
        )
        return entry, sim_out, pool_d

    # --- probe A: causal_conv1d_update (bs=1, slot 1) ----------------------
    conv_entry, conv_sim_out = _run_conv("bs1", x_pre, conv_pool, idx1)
    results["conv_update_bs1"] = conv_entry
    print(
        "CONV bs1 out rel_l2={:.6g} cos={:.6f} state rel_l2={:.6g} untouched={}".format(
            conv_entry["out_sim_vs_ref_fp32"]["relative_l2_error"],
            conv_entry["out_sim_vs_ref_fp32"]["cosine_similarity"],
            conv_entry["state_sim_vs_ref"]["relative_l2_error"],
            conv_entry["untouched_slots_bitexact"],
        ),
        flush=True,
    )

    # --- probe B: packed recurrent decode (bs=1, slot 1) -------------------
    # Standalone: feed the CPU conv reference output (production dataflow).
    rec_entry, rec_sim_out, rec_pool_d = _run_recurrent(
        "bs1", ref_conv_update(x_pre, conv_pool, conv_w, idx1)[0].to(torch.bfloat16),
        a_gating, b_gating, ssm_pool, idx1,
    )
    results["packed_decode_bs1"] = rec_entry
    print(
        "REC bs1 out rel_l2={:.6g} cos={:.6f} state rel_l2={:.6g} untouched={}".format(
            rec_entry["out_sim_vs_ref_fp32"]["relative_l2_error"],
            rec_entry["out_sim_vs_ref_fp32"]["cosine_similarity"],
            rec_entry["state_sim_vs_ref"]["relative_l2_error"],
            rec_entry["untouched_slots_bitexact"],
        ),
        flush=True,
    )

    # --- probe C: on-device chain conv -> recurrent (bs=1) -----------------
    mixed_chain = conv_sim_out.to(device)  # sim conv output, like production
    out_chain = mixed_chain.new_empty(1, 1, HV, V)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_chain,
        a=a_gating.to(device),
        b=b_gating.to(device),
        A_log=A_log.to(device),
        dt_bias=dt_bias.to(device),
        scale=SCALE,
        initial_state=ssm_pool.clone().to(device),
        out=out_chain,
        ssm_state_indices=idx1.to(device),
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    chain_ref_out, chain_ref_pool = ref_packed_decode(
        conv_sim_out, a_gating, b_gating, A_log, dt_bias, ssm_pool, idx1
    )
    results["chain_bs1"] = {
        "out_sim_vs_ref": _metrics(out_chain.detach().cpu().squeeze(1), chain_ref_out),
        "note": "sim conv output fed to sim recurrent; ref chain from same input",
    }
    tensors["chain_bs1_sim_out"] = out_chain.detach().cpu().contiguous()
    tensors["chain_bs1_ref_out"] = (
        chain_ref_out.to(torch.bfloat16).contiguous()
    )
    print(
        "CHAIN bs1 out rel_l2={:.6g}".format(
            results["chain_bs1"]["out_sim_vs_ref"]["relative_l2_error"]
        ),
        flush=True,
    )

    # --- probe D: bs=2 index-structure variants -----------------------------
    conv2_entry, _ = _run_conv("bs2", x_pre2, conv_pool2, idx2)
    results["conv_update_bs2"] = conv2_entry
    rec2_entry, _, _ = _run_recurrent(
        "bs2",
        ref_conv_update(x_pre2, conv_pool2, conv_w, idx2)[0].to(torch.bfloat16),
        a2, b2, ssm_pool2, idx2,
    )
    results["packed_decode_bs2"] = rec2_entry
    print(
        "CONV bs2 rel_l2={:.6g}  REC bs2 rel_l2={:.6g}".format(
            conv2_entry["out_sim_vs_ref_fp32"]["relative_l2_error"],
            rec2_entry["out_sim_vs_ref_fp32"]["relative_l2_error"],
        ),
        flush=True,
    )

    # --- probe E: minimal Triton op probes (reduction/transcendentals) ------
    rowsum_kernel, vecsum_kernel, trans_kernel = _build_op_probes()
    op_results = {}
    xr = torch.randn(32, K, dtype=torch.float32)
    xv = torch.randn(K, dtype=torch.float32)
    xt = (torch.randn(K) * 2.0).to(torch.float32)
    xs = (xt.abs() + 0.05).to(torch.float32)
    xr_d, xv_d, xt_d, xs_d = xr.to(device), xv.to(device), xt.to(device), xs.to(device)
    row_out = torch.zeros(32, dtype=torch.float32, device=device)
    vec_out = torch.zeros(1, dtype=torch.float32, device=device)
    e_out = torch.zeros(K, dtype=torch.float32, device=device)
    sp_out = torch.zeros(K, dtype=torch.float32, device=device)
    sg_out = torch.zeros(K, dtype=torch.float32, device=device)
    sq_out = torch.zeros(K, dtype=torch.float32, device=device)
    rowsum_kernel[(1,)](xr_d, row_out, BV=32, BK=K, num_warps=1, num_stages=3)
    vecsum_kernel[(1,)](xv_d, vec_out, BK=K, num_warps=1, num_stages=3)
    trans_kernel[(1,)](xt_d, xs_d, e_out, sp_out, sg_out, sq_out, N=K,
                       num_warps=1, num_stages=3)
    torch.cuda.synchronize()
    op_results["rowsum_32x128_axis1"] = _metrics(
        row_out.cpu(), xr.sum(dim=1)
    )
    op_results["vecsum_128"] = _metrics(vec_out.cpu(), xv.sum())
    op_results["tl_exp"] = _metrics(e_out.cpu(), torch.exp(xt))
    op_results["softplus_log1p_exp"] = _metrics(sp_out.cpu(), _softplus(xt))
    op_results["tl_sigmoid"] = _metrics(sg_out.cpu(), torch.sigmoid(xt))
    op_results["tl_sqrt"] = _metrics(sq_out.cpu(), torch.sqrt(xs))
    results["op_probes"] = op_results
    for name, value in op_results.items():
        print(
            "OP {} rel_l2={:.6g} max_abs={:.6g}".format(
                name, value["relative_l2_error"], value["max_abs_error"]
            ),
            flush=True,
        )
    tensors.update(
        {
            "op_rowsum_x": xr.contiguous(),
            "op_rowsum_sim": row_out.cpu().contiguous(),
            "op_vecsum_x": xv.contiguous(),
            "op_vecsum_sim": vec_out.cpu().contiguous(),
            "op_trans_x": xt.contiguous(),
            "op_trans_exp_sim": e_out.cpu().contiguous(),
            "op_trans_softplus_sim": sp_out.cpu().contiguous(),
            "op_trans_sigmoid_sim": sg_out.cpu().contiguous(),
            "op_trans_sqrt_sim": sq_out.cpu().contiguous(),
        }
    )

    # --- verdict ------------------------------------------------------------
    conv_bad = conv_entry["out_sim_vs_ref_fp32"]["relative_l2_error"] > 0.02 or (
        conv_entry["state_sim_vs_ref"]["relative_l2_error"] > 0.01
    )
    rec_bad = rec_entry["out_sim_vs_ref_fp32"]["relative_l2_error"] > 0.02 or (
        rec_entry["state_sim_vs_ref"]["relative_l2_error"] > 1e-3
    )
    op_bad = [
        name
        for name, value in op_results.items()
        if value["relative_l2_error"] > 1e-4
    ]
    verdict = {
        "conv_update_diverges": bool(conv_bad),
        "packed_decode_diverges": bool(rec_bad),
        "op_probes_divergent": op_bad,
    }
    if conv_bad and not rec_bad:
        verdict["kernel_at_fault"] = "causal_conv1d_update"
    elif rec_bad and not conv_bad:
        verdict["kernel_at_fault"] = "packed_decode"
    elif conv_bad and rec_bad:
        verdict["kernel_at_fault"] = "both"
    else:
        verdict["kernel_at_fault"] = (
            "none (op-level: " + ",".join(op_bad) + ")") if op_bad else "none"
    results["verdict"] = verdict
    print("VERDICT " + json.dumps(verdict, sort_keys=True), flush=True)

    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    save_file(
        {k: v.contiguous() for k, v in tensors.items()},
        str(args.output_dir / "tensors.safetensors"),
        metadata={"schema": results["schema"]},
    )
    results["tensors_sha256"] = hashlib.sha256(
        (args.output_dir / "tensors.safetensors").read_bytes()
    ).hexdigest()
    (args.output_dir / "result.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    return 1 if (conv_bad or rec_bad or op_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
