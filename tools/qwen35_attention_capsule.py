#!/usr/bin/env python3
"""Replay the Qwen3.5 layer-3 aiter attention defect in isolation.

The layer gate captured the layer-3 boundary tensors at the first numerical
mismatch (``before_hidden`` / ``before_residual``).  An fp64 oracle
adjudication localized the defect to the aiter attention call itself: for a
2-token prefill on the first full-attention layer, token 1's per-head
attention probabilities come out squashed toward uniform while every
projection, norm, RoPE, V read and the output gate are correct.

This capsule rebuilds q/k/v from the captured boundary tensors with plain
torch ops (those stages were proven correct on the simulator), then invokes
THE SAME production aiter entry the SGLang backend uses for this case:

    aiter.mha_batch_prefill_func(
        q.view(-1, 8, 256), k_cache, v_cache,
        qo_indptr, kv_indptr, kv_indices,
        max_q_len, max_kv_len, causal=True, ...)

with the same argument shapes and layout as
``AiterAttnBackend.forward_extend`` (NHD pool, page_size=1, bf16, no
descales, no sliding window, no sink).  The dispatched kernel is the CK-tile
``FmhaBatchPrefillWithPagedKVCacheKernel`` (QRKSVS_ASYNC pipeline, d256,
LINEAR_LAYOUT, SGLANG_PAGE_TABLE_1D).

Exit semantics: exit 0 when the attention output matches the fp64 reference;
exit 1 when the pinned defect signature reproduces (token0 correct, token1
wrong) or the output is structurally broken.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = (
    ROOT / "artifacts/lanes/sglang-tp1-layer-gate-v9/layer-gate/first-mismatch"
)
DEFAULT_MODEL = (
    ROOT / "models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
)
WEIGHT_PREFIX = "model.language_model.layers.3."

# Model geometry for Qwen3.5-0.8B layer 3 (first full-attention layer).
NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIM = 64  # head_dim * partial_rotary_factor(0.25)
ROPE_THETA = 10000000.0
RMS_EPS = 1e-6
SCALE = HEAD_DIM**-0.5
SEQ_LEN = 2

# Pinned fp64-oracle adjudication numbers (see /tmp/layer3_oracle.py and
# /tmp/layer3_kvhyp.py).  Used only as a plumbing sanity check in
# --reference-only mode; the device-mode gate compares against the fp64
# reference computed from the same inputs.
ORACLE_P0 = [0.14, 0.07, 0.45, 0.11, 0.08, 0.02, 0.15, 0.01]
ORACLE_S0 = [3.58, 3.02, 6.68, 4.22, 4.90, 5.63, 5.75, 7.48]
ORACLE_S1 = [5.38, 5.54, 6.88, 6.35, 7.37, 9.79, 7.47, 11.76]
DEFECT_P0 = [0.47, 0.37, 0.30, 0.38, 0.42, 0.38, 0.33, 0.28]


def _sha(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    a = actual.double().reshape(-1)
    b = reference.double().reshape(-1)
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
        "actual_sha256": _sha(actual),
        "reference_sha256": _sha(reference),
    }


def _gemm_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """bf16-in / fp32-accumulate / bf16-out matmul (mirrors hipblas GEMM)."""
    return (x.float() @ w.float().T).to(torch.bfloat16)


def _gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """GemmaRMSNorm forward_native semantics: fp32 internals, (1 + w)."""
    orig_dtype = x.dtype
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + RMS_EPS)
    xf = xf * (1.0 + weight.float())
    return xf.to(orig_dtype)


def _rope_neox(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """NeoX-style partial RoPE (rotary_dim=64, theta=1e7), fp32 internals."""
    orig_dtype = x.dtype
    xf = x.float()
    half = ROTARY_DIM // 2
    t = torch.arange(half, dtype=torch.float32, device=x.device)
    inv_freq = 1.0 / (ROPE_THETA ** (2.0 * t / ROTARY_DIM))
    freqs = positions.float()[:, None] * inv_freq[None, :]  # [T, half]
    cos = freqs.cos()[:, None, :]
    sin = freqs.sin()[:, None, :]
    x1 = xf[..., :half]
    x2 = xf[..., half:ROTARY_DIM]
    rotated = torch.cat(
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin, xf[..., ROTARY_DIM:]], dim=-1
    )
    return rotated.to(orig_dtype)


def _load_layer3_weights(model_path: Path, device: torch.device) -> dict:
    keys = (
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    weights = {}
    with safe_open(str(model_path), framework="pt") as f:
        available = set(f.keys())
        for short in keys:
            full = WEIGHT_PREFIX + short
            if full not in available:
                raise RuntimeError(f"missing model weight: {full}")
            weights[short] = f.get_tensor(full).to(device)
    return weights


def _compute_qkv(
    tensors: dict, weights: dict, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Rebuild layer-3 q/k/v/gate exactly as the model does up to attention.

    These stages (fused add + GemmaRMSNorm, merged qkv projection with the
    per-head [q|gate] interleave, per-head q/k GemmaRMSNorm, NeoX RoPE) were
    all adjudicated correct on the simulator; the object under test is the
    aiter attention call only.
    """
    hidden = tensors["before_hidden"].to(device)
    residual = tensors["before_residual"].to(device)

    # input_layernorm with fused residual add (bf16 add, fp32 norm internals).
    r1 = hidden + residual
    hn = _gemma_rmsnorm(r1, weights["input_layernorm.weight"])

    # Merged qkv projection; q_proj output is per-head [q(256) | gate(256)].
    q_gate = _gemm_bf16(hn, weights["self_attn.q_proj.weight"]).view(
        SEQ_LEN, NUM_HEADS, 2 * HEAD_DIM
    )
    q = q_gate[..., :HEAD_DIM].contiguous()
    gate = q_gate[..., HEAD_DIM:].contiguous()
    k = _gemm_bf16(hn, weights["self_attn.k_proj.weight"]).view(
        SEQ_LEN, NUM_KV_HEADS, HEAD_DIM
    )
    v = (
        _gemm_bf16(hn, weights["self_attn.v_proj.weight"])
        .view(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM)
        .contiguous()
    )

    # Per-head q/k GemmaRMSNorm then NeoX RoPE at positions [0, 1].
    q = _gemma_rmsnorm(q, weights["self_attn.q_norm.weight"])
    k = _gemma_rmsnorm(k, weights["self_attn.k_norm.weight"])
    positions = torch.arange(SEQ_LEN, device=device)
    q = _rope_neox(q, positions).contiguous()
    k = _rope_neox(k, positions).contiguous()
    return q, k, v, gate, r1, hn


def _reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 CPU causal GQA softmax attention; returns (output, probs)."""
    qd = q.detach().cpu().double()
    kd = k.detach().cpu().double()
    vd = v.detach().cpu().double()
    rep = NUM_HEADS // NUM_KV_HEADS
    k_e = kd.repeat_interleave(rep, dim=1)
    v_e = vd.repeat_interleave(rep, dim=1)
    qh = qd.permute(1, 0, 2)  # [H, T, D]
    kh = k_e.permute(1, 0, 2)
    vh = v_e.permute(1, 0, 2)
    scores = qh @ kh.transpose(-1, -2) * SCALE
    mask = torch.triu(
        torch.full((SEQ_LEN, SEQ_LEN), float("-inf"), dtype=torch.float64),
        diagonal=1,
    )
    probs = torch.softmax(scores + mask, dim=-1)
    out = (probs @ vh).permute(1, 0, 2)  # [T, H, D]
    return out, probs


def _implied_p0(
    out_token1: torch.Tensor, v: torch.Tensor
) -> list[float]:
    """Per-head fitted p(attend tok0) from o = p0*v0 + (1-p0)*v1."""
    rep = NUM_HEADS // NUM_KV_HEADS
    v_e = v.detach().cpu().double().repeat_interleave(rep, dim=1)  # [T, H, D]
    o = out_token1.detach().cpu().double()  # [H, D]
    v0, v1 = v_e[0], v_e[1]
    diff = v0 - v1
    num = ((o - v1) * diff).sum(-1)
    den = (diff * diff).sum(-1).clamp_min(1e-30)
    return [float(x) for x in (num / den)]


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "QWEN35_ATTENTION_CAPSULE_OUTPUT",
                str(ROOT / "artifacts/qwen35-attention-capsule/20260820-layer3-v1"),
            )
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help=(
            "skip the aiter call: verify tensor plumbing + fp64 reference "
            "against the pinned oracle numbers (runs on CPU)"
        ),
    )
    parser.add_argument(
        "--kv-pool-slots",
        type=int,
        default=17,
        help=(
            "NHD KV pool slot count (lane runs max_total_tokens=16, "
            "page_size=1: pool = size + page_size = 17 slots, slot 0 reserved)"
        ),
    )
    parser.add_argument(
        "--kv-slot-offset",
        type=int,
        default=1,
        help="first pool slot used by the 2 prefill tokens (allocator skips 0)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse()
    capture = args.capture.resolve(strict=True)
    tensors_path = capture / "tensors.safetensors"
    if not tensors_path.is_file():
        raise RuntimeError(f"incomplete first-mismatch capture: {capture}")
    tensors = load_file(str(tensors_path), device="cpu")
    required = {"before_hidden", "before_residual"}
    if not required.issubset(tensors):
        raise RuntimeError(f"capture missing {required - set(tensors)}: {capture}")

    if args.reference_only:
        device = torch.device("cpu")
    else:
        if str(args.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("simulated HIP torch device is unavailable")
        device = torch.device(args.device)

    # The q/k/v preamble stages were adjudicated correct on the simulator, and
    # running them as ~30 torch kernels on the simulated device once stalled a
    # capsule for 30 minutes before the kernel under test was even reached.
    # Compute them on the host and ship only the attention inputs across; the
    # object under test is the single aiter attention kernel.
    host = torch.device("cpu")
    weights = _load_layer3_weights(args.model.resolve(strict=True), host)
    q, k, v, gate, r1, hn = _compute_qkv(tensors, weights, host)

    ref_out, ref_probs = _reference_attention(q, k, v)
    ref_p0 = [float(x) for x in ref_probs[:, 1, 0]]
    # fp64 scaled scores, recomputed from the (bf16-rounded) q/k actually fed
    # to the kernel; diagnostic parity with the oracle adjudication.
    rep = NUM_HEADS // NUM_KV_HEADS
    k_e = k.detach().cpu().double().repeat_interleave(rep, dim=1)
    q1 = q.detach().cpu().double()[1]
    s0 = [float(x) for x in (q1 * k_e[0]).sum(-1) * SCALE]
    s1 = [float(x) for x in (q1 * k_e[1]).sum(-1) * SCALE]

    result = {
        "schema": "amdgpu-sim.qwen35-attention-capsule.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "capture": str(capture),
        "model": str(args.model),
        "device": str(device),
        "mode": "reference-only" if args.reference_only else "replay",
        "kv_pool_slots": args.kv_pool_slots,
        "kv_slot_offset": args.kv_slot_offset,
        "reference_p0_token1": ref_p0,
        "reference_scaled_scores_token1_vs_tok0": s0,
        "reference_scaled_scores_token1_vs_self": s1,
        "oracle_pinned_p0": ORACLE_P0,
        "defect_pinned_p0": DEFECT_P0,
    }

    saved = {
        "q": q.detach().cpu().contiguous(),
        "k": k.detach().cpu().contiguous(),
        "v": v.detach().cpu().contiguous(),
        "gate": gate.detach().cpu().contiguous(),
        "reference_out": ref_out.contiguous(),
        "reference_probs": ref_probs.contiguous(),
    }

    if args.reference_only:
        # Plumbing gate: the fp64 reference computed from the rebuilt q/k/v
        # must land on the pinned oracle numbers (bf16 stage rounding gives
        # small drift versus the all-fp64 oracle pipeline).
        p0_err = max(abs(a - b) for a, b in zip(ref_p0, ORACLE_P0))
        s0_err = max(abs(a - b) for a, b in zip(s0, ORACLE_S0))
        s1_err = max(abs(a - b) for a, b in zip(s1, ORACLE_S1))
        result["reference_vs_oracle"] = {
            "p0_max_abs_err": p0_err,
            "s0_max_abs_err": s0_err,
            "s1_max_abs_err": s1_err,
        }
        ok = p0_err < 0.02 and s0_err < 0.2 and s1_err < 0.2
        result["state"] = "reference_verified" if ok else "reference_mismatch"
        _finish(args, result, saved)
        if not ok:
            print(
                "CAPSULE REFERENCE FAIL: rebuilt q/k/v do not reproduce the "
                "pinned oracle numbers (plumbing drift)",
                flush=True,
            )
            return 1
        print(
            "CAPSULE REFERENCE PASS: tensor plumbing and fp64 reference match "
            "the pinned oracle adjudication",
            flush=True,
        )
        return 0

    # ---- production aiter call ------------------------------------------
    # Import only inside the lane: importing aiter outside the simulator
    # environment would bind the wrong backend identity.
    from aiter import mha_batch_prefill_func

    # NHD KV pool, page_size=1, exactly as MHATokenToKVPool allocates it and
    # AiterAttnBackend.forward_extend consumes it.  The two prefill tokens
    # occupy consecutive slots starting at kv_slot_offset (slot 0 reserved).
    pool_slots = args.kv_pool_slots
    off = args.kv_slot_offset
    if off < 0 or off + SEQ_LEN > pool_slots:
        raise RuntimeError(
            f"kv slots [{off}, {off + SEQ_LEN}) do not fit pool of {pool_slots}"
        )
    k_cache = torch.zeros(
        (pool_slots, NUM_KV_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    v_cache = torch.zeros_like(k_cache)
    cache_loc = torch.arange(off, off + SEQ_LEN, device=device)
    k_cache[cache_loc] = k.to(device)
    v_cache[cache_loc] = v.to(device)

    qo_indptr = torch.tensor([0, SEQ_LEN], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, SEQ_LEN], dtype=torch.int32, device=device)
    # AiterIndicesUpdaterPrefill allocates seq_lens_sum + 256 entries and pads
    # the tail with kv_indices[0] (the mha_batch_prefill 128-token read WA).
    kv_indices = torch.empty(SEQ_LEN + 256, dtype=torch.int32, device=device)
    kv_indices[:SEQ_LEN] = cache_loc.to(torch.int32)
    kv_indices[SEQ_LEN:] = kv_indices[0]

    print(
        "CAPSULE input",
        json.dumps(
            {
                "capture": str(capture),
                "device": str(device),
                "q_shape": list(q.shape),
                "k_cache_shape": list(k_cache.shape),
                "kv_indices_len": int(kv_indices.numel()),
                "cache_slots": [int(x) for x in cache_loc],
                "max_q_len": SEQ_LEN,
                "max_kv_len": SEQ_LEN,
                "function": "aiter.mha_batch_prefill_func",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # Same call as AiterAttnBackend.forward_extend (non-MLA, NHD, bf16 path):
    # softmax_scale defaults to head_dim**-0.5 inside the wrapper, matching
    # the backend which does not pass it either.
    o = mha_batch_prefill_func(
        q.to(device).contiguous().view(-1, NUM_HEADS, HEAD_DIM),
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        SEQ_LEN,
        SEQ_LEN,
        causal=True,
        logits_soft_cap=0.0,
        alibi_slopes=None,
        return_lse=False,
        return_attn_probs=False,
        window_size=(-1, -1),
        sink_ptr=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    o = o.detach().cpu().contiguous()
    saved["replayed_out"] = o

    structural_ok = (
        tuple(o.shape) == (SEQ_LEN, NUM_HEADS, HEAD_DIM)
        and torch.isfinite(o.float()).all().item()
    )
    result["structural_ok"] = bool(structural_ok)
    result["output_dtype"] = str(o.dtype).removeprefix("torch.")

    per_token = []
    per_head = []
    if structural_ok:
        for t in range(SEQ_LEN):
            per_token.append(_metrics(o[t], ref_out[t].to(o.dtype)))
            heads = []
            for h in range(NUM_HEADS):
                a = o[t, h].double()
                b = ref_out[t, h]
                heads.append(
                    float(
                        torch.linalg.vector_norm(a - b).item()
                        / max(torch.linalg.vector_norm(b).item(), 1e-30)
                    )
                )
            per_head.append(heads)
        result["replayed_implied_p0_token1"] = _implied_p0(o[1], v)
    result["per_token_vs_reference"] = per_token
    result["per_head_rel_l2"] = per_head

    if structural_ok:
        tok0 = per_token[0]["relative_l2_error"]
        tok1 = per_token[1]["relative_l2_error"]
        print(f"token0 rel_l2 vs fp64 reference: {tok0:.6f}", flush=True)
        print(f"token1 rel_l2 vs fp64 reference: {tok1:.6f}", flush=True)
        for t in range(SEQ_LEN):
            pretty = ", ".join(f"{x:.4f}" for x in per_head[t])
            print(f"token{t} per-head rel_l2: [{pretty}]", flush=True)
        print(
            "token1 implied p(attend tok0) per head:",
            [f"{x:.3f}" for x in result["replayed_implied_p0_token1"]],
            flush=True,
        )
        print(
            "reference p(attend tok0) per head:   ",
            [f"{x:.3f}" for x in ref_p0],
            flush=True,
        )

    if not structural_ok:
        result["state"] = "structural_failure"
        verdict, code = "CAPSULE FAIL: structurally broken attention output", 1
    elif tok0 < 0.02 and tok1 > 0.05:
        result["state"] = "defect_reproduced"
        verdict, code = (
            "CAPSULE FAIL: defect signature reproduced "
            "(token0 correct, token1 attention wrong)",
            1,
        )
    elif tok0 < 0.02 and tok1 <= 0.05:
        result["state"] = "attention_correct"
        verdict, code = (
            "CAPSULE PASS: aiter attention matches fp64 reference for both tokens",
            0,
        )
    else:
        result["state"] = "unexpected_mismatch"
        verdict, code = (
            "CAPSULE FAIL: token0 also mismatches "
            "(not the pinned signature; environment or plumbing drift)",
            1,
        )

    _finish(args, result, saved)
    print(verdict, flush=True)
    return code


def _finish(args: argparse.Namespace, result: dict, saved: dict) -> None:
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    save_file(
        {name: t for name, t in saved.items()},
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


if __name__ == "__main__":
    raise SystemExit(main())
