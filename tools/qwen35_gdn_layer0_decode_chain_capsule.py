#!/usr/bin/env python3
"""Layer-0 GDN decode chain capsule: find the first diverging m=1 op.

The decode-kernel differential (tools/qwen35_gdn_decode_diff_capsule.py)
CLEARED both decode-only Triton kernels bit-exactly:
``_causal_conv1d_update_kernel`` and
``fused_recurrent_gated_delta_rule_packed_decode_kernel``.  The decode-phase
layer gate still shows layer 0's output diverging (rel_l2 1.42) with a
bit-exact input, so this capsule stages the FULL layer-0 GDN decode forward
at m=1 exactly as production runs it (Qwen3_5LinearDecoderLayer.forward):

  in_proj_qkvz (m=1 torch F.linear, bf16)      [1, 8192]
  in_proj_ba   (m=1 torch F.linear, bf16)      [1, 32]
  fused_qkvzba_split_reshape_cat_contiguous     -> mixed_qkv/z/b/a
  causal_conv1d_update        (CLEARED)
  packed_decode               (CLEARED)
  RMSNormGated (fla Triton, M=16 rows)          [16, 128]
  out_proj (m=1 torch F.linear, bf16)           [1, 1024]

with real layer-0 weights and compares EVERY stage against CPU float32
references two ways: end-to-end (vs the all-CPU chain) and locally (vs a
reference recomputed from the simulator's own previous-stage output), so
the first genuinely wrong op is named even if earlier drift is only
rounding.  Reference formulas are documented per stage; the conv/recurrent
references are reused from the decode-diff capsule module.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
LAYER_PREFIX = "model.language_model.layers.0.linear_attn"
HIDDEN = 1024
H = 16
HV = 16
K = 128
V = 128
DIM = 2 * H * K + HV * V  # 6144
ZDIM = HV * V  # 2048
WIDTH = 4
SCALE = float(K) ** -0.5
EPS = 1e-6


def _load_decode_diff_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_gdn_decode_diff_capsule",
        str(ROOT / "tools/qwen35_gdn_decode_diff_capsule.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    a = actual.detach().float().reshape(-1)
    b = reference.detach().float().reshape(-1)
    delta = a - b
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype).removeprefix("torch."),
        "max_abs_error": float(delta.abs().max().item()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(delta).item()
            / max(torch.linalg.vector_norm(b).item(), 1.0e-30)
        ),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(a[None], b[None]).item()
        ),
    }


def ref_split(qkvz: torch.Tensor, ba: torch.Tensor):
    """Reference for fused_qkvzba_split_reshape_cat_contiguous.

    Pure data movement (the Triton kernel is an identity copy for the
    Qwen3.5 geometry NUM_HEADS_QK == NUM_HEADS_V):
      mixed_qkv[n, 0:6144] = qkvz[n, 0:6144]   (q|k|v passed through)
      z[n, hv, :]          = qkvz[n, 6144 + hv*128 : ... + 128]
      b[n, hv]             = ba[n, hv]
      a[n, hv]             = ba[n, 16 + hv]
    """
    bs = qkvz.shape[0]
    mixed = qkvz[:, :DIM].clone()
    z = qkvz[:, DIM : DIM + ZDIM].reshape(bs, HV, V).clone()
    b = ba[:, :HV].clone()
    a = ba[:, HV:].clone()
    return mixed, z, b, a


def ref_rmsnorm_gated(x: torch.Tensor, w: torch.Tensor, z: torch.Tensor):
    """Reference for fla RMSNormGated (rms variant, norm_before_gate, swish).

    Per row r of 128 (fp32 math, matching the kernel's casts):
      var = mean(x^2); rstd = rsqrt(var + eps)
      y = x * rstd * w
      y = y * silu(z)          (z * sigmoid(z))
    stored to the input dtype (bf16 in production).
    """
    xf = x.float()
    zf = z.float()
    wf = w.float()
    rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + EPS)
    y = xf * rstd * wf
    y = y * (zf * torch.sigmoid(zf))
    return y


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            __import__("os").environ.get(
                "QWEN35_GDN_CHAIN_OUTPUT",
                str(ROOT / "artifacts/qwen35-gdn-decode-chain/20260821-v1"),
            )
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--slots", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        print("CAPSULE ERROR: simulated HIP torch device is unavailable", flush=True)
        return 2
    device = torch.device(args.device)

    diff = _load_decode_diff_module()

    torch.manual_seed(args.seed)
    weights = load_file(str(MODEL_FILE), device="cpu")
    W_qkvz = torch.cat(
        [weights[f"{LAYER_PREFIX}.in_proj_qkv.weight"], weights[f"{LAYER_PREFIX}.in_proj_z.weight"]],
        dim=0,
    ).contiguous()  # bf16 [8192, 1024] = [q|k|v|z]
    W_ba = torch.cat(
        [weights[f"{LAYER_PREFIX}.in_proj_b.weight"], weights[f"{LAYER_PREFIX}.in_proj_a.weight"]],
        dim=0,
    ).contiguous()  # bf16 [32, 1024] = [b|a]
    norm_w = weights[f"{LAYER_PREFIX}.norm.weight"].to(torch.bfloat16).contiguous()
    W_out = weights[f"{LAYER_PREFIX}.out_proj.weight"].contiguous()  # bf16 [1024, 2048]
    conv_w = weights[f"{LAYER_PREFIX}.conv1d.weight"].view(DIM, WIDTH).contiguous()
    A_log = weights[f"{LAYER_PREFIX}.A_log"].contiguous()
    dt_bias = weights[f"{LAYER_PREFIX}.dt_bias"].contiguous()

    h = torch.randn(1, HIDDEN).to(torch.bfloat16)
    conv_pool = (torch.randn(args.slots, DIM, WIDTH - 1) * 0.7).to(torch.bfloat16)
    ssm_pool = (torch.randn(args.slots, HV, V, K) * 0.3).to(torch.float32)
    idx = torch.tensor([1], dtype=torch.int32)

    # Import production wrappers only after the lane env is up.
    import torch.nn as nn
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated

    # ------------------------- CPU reference chain -------------------------
    # GEMM refs in fp32 from the bf16 operands; conv/recurrent refs reused
    # from the decode-diff capsule (kernel-faithful numerics).
    t0 = time.time()
    qkvz_ref = h.float() @ W_qkvz.float().t()
    ba_ref = h.float() @ W_ba.float().t()
    mixed_ref, z_ref, b_ref, a_ref = ref_split(
        qkvz_ref.to(torch.bfloat16), ba_ref.to(torch.bfloat16)
    )
    conv_out_ref, conv_pool_ref = diff.ref_conv_update(
        mixed_ref, conv_pool, conv_w, idx
    )
    core_ref, ssm_pool_ref = diff.ref_packed_decode(
        conv_out_ref.to(torch.bfloat16), a_ref, b_ref, A_log, dt_bias, ssm_pool, idx
    )
    core_rows_ref = core_ref.reshape(HV * 1, V).to(torch.bfloat16)
    z_rows_ref = z_ref.reshape(HV * 1, V)
    normed_ref = ref_rmsnorm_gated(core_rows_ref, norm_w, z_rows_ref)
    out_ref = (
        normed_ref.reshape(1, HV * V).to(torch.bfloat16).float() @ W_out.float().t()
    )
    ref_wall = time.time() - t0

    # --------------------------- device chain ------------------------------
    stage: dict[str, dict] = {}

    def checkpoint(name, sim, ref_e2e, sim_prev, local_ref_fn, tol):
        local = local_ref_fn(sim_prev) if local_ref_fn is not None else ref_e2e
        entry = {
            "e2e_vs_cpu_chain": _metrics(sim, ref_e2e),
            "local_vs_ref": _metrics(sim, local),
            "tolerance": tol,
        }
        entry["diverges"] = bool(
            entry["local_vs_ref"]["relative_l2_error"] > tol
        )
        stage[name] = entry
        print(
            "STAGE {:14s} local rel_l2={:.6g} cos={:.8f} e2e rel_l2={:.6g}{}".format(
                name,
                entry["local_vs_ref"]["relative_l2_error"],
                entry["local_vs_ref"]["cosine_similarity"],
                entry["e2e_vs_cpu_chain"]["relative_l2_error"],
                "  <-- DIVERGES" if entry["diverges"] else "",
            ),
            flush=True,
        )
        return sim

    h_d = h.to(device)
    W_qkvz_d = W_qkvz.to(device)
    W_ba_d = W_ba.to(device)
    W_out_d = W_out.to(device)
    t1 = time.time()

    qkvz_d = F.linear(h_d, W_qkvz_d)
    torch.cuda.synchronize()
    checkpoint(
        "in_proj_qkvz", qkvz_d.detach().cpu(), qkvz_ref.to(torch.bfloat16), h,
        lambda prev: (prev.float() @ W_qkvz.float().t()).to(torch.bfloat16), 0.02,
    )
    ba_d = F.linear(h_d, W_ba_d)
    torch.cuda.synchronize()
    checkpoint(
        "in_proj_ba", ba_d.detach().cpu(), ba_ref.to(torch.bfloat16), h,
        lambda prev: (prev.float() @ W_ba.float().t()).to(torch.bfloat16), 0.02,
    )

    mixed_d, z_d, b_d, a_d = fused_qkvzba_split_reshape_cat_contiguous(
        qkvz_d, ba_d, H, HV, K, V
    )
    torch.cuda.synchronize()
    checkpoint(
        "qkvzba_split", mixed_d.detach().cpu(), mixed_ref, qkvz_d.detach().cpu(),
        lambda prev: ref_split(prev, ba_d.detach().cpu())[0], 1e-6,
    )
    checkpoint(
        "split_z", z_d.detach().cpu(), z_ref, qkvz_d.detach().cpu(),
        lambda prev: ref_split(prev, ba_d.detach().cpu())[1], 1e-6,
    )
    checkpoint(
        "split_a", a_d.detach().cpu(), a_ref, ba_d.detach().cpu(),
        lambda prev: ref_split(qkvz_d.detach().cpu(), prev)[3], 1e-6,
    )
    checkpoint(
        "split_b", b_d.detach().cpu(), b_ref, ba_d.detach().cpu(),
        lambda prev: ref_split(qkvz_d.detach().cpu(), prev)[2], 1e-6,
    )

    conv_pool_d = conv_pool.clone().to(device)
    conv_out_d = causal_conv1d_update(
        mixed_d, conv_pool_d, conv_w.to(device), None, "silu", conv_state_indices=idx.to(device)
    )
    torch.cuda.synchronize()
    checkpoint(
        "conv_update", conv_out_d.detach().cpu(), conv_out_ref.to(torch.bfloat16),
        mixed_d.detach().cpu().to(torch.bfloat16),
        lambda prev: diff.ref_conv_update(prev, conv_pool, conv_w, idx)[0].to(torch.bfloat16),
        0.02,
    )
    checkpoint_state_conv = _metrics(
        conv_pool_d.detach().cpu()[1], conv_pool_ref[1].float()
    )

    core_d = mixed_d.new_empty(1, 1, HV, V)
    ssm_pool_d = ssm_pool.clone().to(device)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=conv_out_d,
        a=a_d,
        b=b_d,
        A_log=A_log.to(device),
        dt_bias=dt_bias.to(device),
        scale=SCALE,
        initial_state=ssm_pool_d,
        out=core_d,
        ssm_state_indices=idx.to(device),
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    checkpoint(
        "packed_decode", core_d.detach().cpu().squeeze(1), core_ref.to(torch.bfloat16),
        (conv_out_d.detach().cpu(), a_d.detach().cpu(), b_d.detach().cpu()),
        lambda prev: diff.ref_packed_decode(
            prev[0], prev[1], prev[2], A_log, dt_bias, ssm_pool, idx
        )[0].to(torch.bfloat16),
        0.02,
    )
    checkpoint_state_ssm = _metrics(
        ssm_pool_d.detach().cpu()[1], ssm_pool_ref[1]
    )

    norm_mod = RMSNormGated(
        V, eps=EPS, group_size=None, norm_before_gate=True,
        device=device, dtype=torch.bfloat16,
    )
    with torch.no_grad():
        norm_mod.weight.copy_(norm_w.to(device))
    core_rows_d = core_d.detach().reshape(HV, V)
    z_rows_d = z_d.detach().reshape(HV, V)
    normed_d = norm_mod(core_rows_d, z_rows_d)
    torch.cuda.synchronize()
    checkpoint(
        "rmsnorm_gated", normed_d.detach().cpu(), normed_ref.to(torch.bfloat16),
        (core_rows_d.detach().cpu(), z_rows_d.detach().cpu()),
        lambda prev: ref_rmsnorm_gated(prev[0], norm_w, prev[1]).to(torch.bfloat16),
        0.02,
    )

    normed_2d = normed_d.reshape(1, HV * V)
    out_d = F.linear(normed_2d, W_out_d)
    torch.cuda.synchronize()
    checkpoint(
        "out_proj", out_d.detach().cpu(), out_ref.to(torch.bfloat16),
        normed_2d.detach().cpu(),
        lambda prev: (prev.float().to(torch.bfloat16).float() @ W_out.float().t()).to(torch.bfloat16),
        0.02,
    )
    dev_wall = time.time() - t1

    first_bad = next((name for name, e in stage.items() if e["diverges"]), None)
    verdict = {
        "first_diverging_stage": first_bad,
        "all_stages_clean": first_bad is None,
        "cleared_kernels_note": (
            "conv_update and packed_decode were cleared bit-exact by "
            "qwen35_gdn_decode_diff_capsule.py; this chain checks the m=1 "
            "dispatches around them (in_proj GEMMs, split, rmsnorm, out_proj)"
        ),
    }
    print("VERDICT " + json.dumps(verdict, sort_keys=True), flush=True)

    result = {
        "schema": "amdgpu-sim.qwen35-gdn-decode-chain.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "seed": args.seed,
            "slots": args.slots,
            "selected_slot": 1,
            "weights_sha256": {
                "in_proj_qkvz": hashlib.sha256(
                    W_qkvz.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "in_proj_ba": hashlib.sha256(
                    W_ba.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "norm": hashlib.sha256(
                    norm_w.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "out_proj": hashlib.sha256(
                    W_out.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            },
        },
        "stages": stage,
        "conv_state_after": checkpoint_state_conv,
        "ssm_state_after": checkpoint_state_ssm,
        "ref_wall_seconds": round(ref_wall, 3),
        "device_wall_seconds": round(dev_wall, 3),
        "verdict": verdict,
    }
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    save_file(
        {
            "h": h.contiguous(),
            "qkvz_sim": qkvz_d.detach().cpu().contiguous(),
            "qkvz_ref": qkvz_ref.to(torch.bfloat16).contiguous(),
            "ba_sim": ba_d.detach().cpu().contiguous(),
            "mixed_sim": mixed_d.detach().cpu().contiguous(),
            "z_sim": z_d.detach().cpu().contiguous(),
            "a_sim": a_d.detach().cpu().contiguous(),
            "b_sim": b_d.detach().cpu().contiguous(),
            "conv_out_sim": conv_out_d.detach().cpu().contiguous(),
            "core_sim": core_d.detach().cpu().contiguous(),
            "normed_sim": normed_d.detach().cpu().contiguous(),
            "layer_out_sim": out_d.detach().cpu().contiguous(),
            "layer_out_ref": out_ref.to(torch.bfloat16).contiguous(),
            "conv_state_sim": conv_pool_d.detach().cpu().contiguous(),
            "ssm_state_sim": ssm_pool_d.detach().cpu().contiguous(),
        },
        str(args.output_dir / "tensors.safetensors"),
        metadata={"schema": result["schema"]},
    )
    result["tensors_sha256"] = hashlib.sha256(
        (args.output_dir / "tensors.safetensors").read_bytes()
    ).hexdigest()
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 1 if first_bad is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
