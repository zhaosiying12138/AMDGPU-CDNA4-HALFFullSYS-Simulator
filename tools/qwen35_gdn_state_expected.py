#!/usr/bin/env python3
"""Offline expected-GDN-state analyzer for the second-token defect.

Reimplements the NVIDIA golden generator's layer-0 recurrence (zero initial
state, sequential token updates) on the host CPU and compares, per position,
against the engine's captured decode tensors and journal fingerprints:

  expected state after pos1 == what prefill must commit to slot 2
                              (journal decode_entry_pool row2 fingerprint)
  expected state after pos2 == engine decode2_packed_decode_states (post row)
  expected state after pos3 == engine decode3_packed_decode_states + golden

This splits "prefill committed a wrong state" from "the decode update wrote
a wrong state" without another simulator run.  CPU bf16 rounding differs
from the NVIDIA reference in the projections, so comparisons use cosine
similarity; the trajectory sanity check (expected pos3 vs golden
recurrent_state) must come out near 1.0 before any conclusion is read.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "artifacts/qwen35-nvidia-golden/20260812-decode4-max24-v1/results.safetensors"
CHECKPOINT = ROOT / "models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
LANE = ROOT / "artifacts/lanes/zcode-decode-15/layer-gate"

TOKEN_IDS = [248044, 266, 27841, 27841]
EPSILON = 1.0e-6
NUM_HEADS, HEAD_DIM = 16, 128
QKV_DIM = 3 * NUM_HEADS * HEAD_DIM
Z_DIM = NUM_HEADS * HEAD_DIM

LAYER = "model.language_model.layers.0"
NAMES = {
    "embedding": "model.language_model.embed_tokens.weight",
    "input_norm": f"{LAYER}.input_layernorm.weight",
    "qkv": f"{LAYER}.linear_attn.in_proj_qkv.weight",
    "z": f"{LAYER}.linear_attn.in_proj_z.weight",
    "b": f"{LAYER}.linear_attn.in_proj_b.weight",
    "a": f"{LAYER}.linear_attn.in_proj_a.weight",
    "conv": f"{LAYER}.linear_attn.conv1d.weight",
    "a_log": f"{LAYER}.linear_attn.A_log",
    "dt_bias": f"{LAYER}.linear_attn.dt_bias",
}


def bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x.float(), weight.float().T).to(torch.bfloat16)


def fingerprint(t: torch.Tensor) -> str:
    f = t.detach().cpu().contiguous().float().flatten()
    return hashlib.sha256(
        struct.pack("<%df" % f.numel(), *f.tolist())
    ).hexdigest()[:16]


def main() -> int:
    cos = torch.nn.functional.cosine_similarity
    ckpt = safe_open(str(CHECKPOINT), framework="pt", device="cpu")
    emb = ckpt.get_tensor(NAMES["embedding"])
    w = {k: ckpt.get_tensor(n) for k, n in NAMES.items() if k != "embedding"}
    hidden = torch.cat(
        [emb[t : t + 1] for t in TOKEN_IDS], dim=0
    ).contiguous()

    x = hidden.float()
    variance = torch.mean(x * x, dim=-1, keepdim=True)
    normalized = (
        x * torch.rsqrt(variance + EPSILON) * (1.0 + w["input_norm"].float())
    ).to(torch.bfloat16)

    qkvz = torch.cat(
        [
            bf16_linear(normalized, w["qkv"]),
            bf16_linear(normalized, w["z"]),
        ],
        dim=-1,
    )
    ba = torch.cat(
        [bf16_linear(normalized, w["b"]), bf16_linear(normalized, w["a"])],
        dim=-1,
    )
    mixed_qkv = qkvz[:, :QKV_DIM].contiguous()
    b = ba[:, :NUM_HEADS].contiguous()
    a = ba[:, NUM_HEADS:].contiguous()

    conv_w = w["conv"].float().reshape(QKV_DIM, 4)
    selected = torch.zeros((QKV_DIM, 3), dtype=torch.bfloat16)
    conv_rows = []
    for t in range(len(TOKEN_IDS)):
        cur = mixed_qkv[t].float()
        acc = (
            selected[:, 0].float() * conv_w[:, 0]
            + selected[:, 1].float() * conv_w[:, 1]
            + selected[:, 2].float() * conv_w[:, 2]
            + cur * conv_w[:, 3]
        )
        conv_rows.append((acc * torch.sigmoid(acc)).to(torch.bfloat16))
        selected = torch.stack(
            (selected[:, 1], selected[:, 2], mixed_qkv[t]), dim=1
        )
    mixed_after_conv = torch.stack(conv_rows, dim=0)

    state = torch.zeros((NUM_HEADS, HEAD_DIM, HEAD_DIM), dtype=torch.float32)
    states = {}
    kernel_inputs = {}
    for t in range(len(TOKEN_IDS)):
        mixed = mixed_after_conv[t].float().view(-1)
        q = mixed[: NUM_HEADS * HEAD_DIM].view(NUM_HEADS, HEAD_DIM)
        k = mixed[NUM_HEADS * HEAD_DIM : 2 * NUM_HEADS * HEAD_DIM].view(
            NUM_HEADS, HEAD_DIM
        )
        v = mixed[2 * NUM_HEADS * HEAD_DIM :].view(NUM_HEADS, HEAD_DIM)
        q = q * torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + EPSILON)
        k = k * torch.rsqrt(torch.sum(k * k, dim=-1, keepdim=True) + EPSILON)
        q = q * (HEAD_DIM**-0.5)
        sp_in = a[t].float() + w["dt_bias"].float()
        softplus = torch.where(
            sp_in <= 20.0, torch.log1p(torch.exp(sp_in)), sp_in
        )
        g = -torch.exp(w["a_log"]) * softplus
        decay = torch.exp(g)
        beta = torch.sigmoid(b[t].float()).to(torch.bfloat16).float()
        state = state * decay.view(NUM_HEADS, 1, 1)
        pred = torch.sum(state * k[:, None, :], dim=-1)
        delta = (v - pred) * beta[:, None]
        state = state + delta[:, :, None] * k[:, None, :]
        states[t] = state.clone()
        kernel_inputs[t] = mixed_after_conv[t]

    # sanity: expected pos3 state vs golden recurrent_state
    g = safe_open(str(GOLDEN), framework="pt", device="cpu")
    grs = g.get_tensor("layers.0.recurrent_state").float()
    gcs = g.get_tensor("layers.0.conv_state")
    print(
        "[sanity] expected pos3 state vs golden:",
        round(float(cos(states[3].reshape(-1), grs.reshape(-1), dim=0)), 6),
    )
    gline = gcs[gcs.dim() == 3 and 1 or 0] if gcs.dim() == 3 else gcs
    if gcs.dim() == 3:
        gline = gcs[1].float()
    else:
        gline = gcs.float()
    print(
        "[sanity] expected pos3 conv vs golden line1:",
        round(float(cos(selected.reshape(-1).float(), gline.reshape(-1), dim=0)), 6),
    )

    print()
    print("expected state fingerprints (fp32-sha256-16):")
    for t in range(4):
        print(f"  pos{t}: ssm fp {fingerprint(states[t])}")

    print()
    e2 = safe_open(str(LANE / "decode2_packed_decode_states.safetensors"), framework="pt")
    e3 = safe_open(str(LANE / "decode3_packed_decode_states.safetensors"), framework="pt")
    ssm2 = e2.get_tensor("ssm_states_selected").float()
    ssm3 = e3.get_tensor("ssm_states_selected").float()
    ki2 = e2.get_tensor("first_kernel_input").float().reshape(-1)
    ki3 = e3.get_tensor("first_kernel_input").float().reshape(-1)
    c2 = safe_open(str(LANE / "decode2_conv_update_states.safetensors"), framework="pt")
    conv2 = c2.get_tensor("conv_states_selected").float()

    rows = [
        ("engine ssm post-pos2", ssm2.reshape(-1), states[2].reshape(-1)),
        ("engine ssm post-pos3", ssm3.reshape(-1), states[3].reshape(-1)),
        (
            "engine kernel_input pos2",
            ki2,
            kernel_inputs[2].float().reshape(-1),
        ),
        (
            "engine kernel_input pos3",
            ki3,
            kernel_inputs[3].float().reshape(-1),
        ),
        (
            "engine conv line post-pos2",
            conv2.reshape(-1),
            None,
        ),
    ]
    print("engine vs expected (cosine):")
    for label, eng, exp in rows:
        if exp is None:
            # conv expected line after pos2 = window (pos0, pos1, pos2)
            exp = torch.stack(
                (mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]), dim=1
            ).float().reshape(-1)
        print(f"  {label}: {round(float(cos(eng, exp, dim=0)), 6)}")

    # conv line after pos3 for engine decode3
    c3 = safe_open(str(LANE / "decode3_conv_update_states.safetensors"), framework="pt")
    conv3 = c3.get_tensor("conv_states_selected").float().reshape(-1)
    exp3 = torch.stack(
        (mixed_qkv[1], mixed_qkv[2], mixed_qkv[3]), dim=1
    ).float().reshape(-1)
    print(
        f"  engine conv line post-pos3: {round(float(cos(conv3, exp3, dim=0)), 6)}"
    )

    # journal reference fingerprints
    journal = [json.loads(l) for l in open(LANE / "state-journal.jsonl")]
    entries = [
        e
        for e in journal
        if e.get("event") == "decode_entry_pool" and e.get("layer") == 0
    ]
    for i, e in enumerate(entries):
        print(
            f"journal decode_entry L0 row{i + 1}: slot fp {e.get('slot_sha256_16')}"
        )
    print("  (expected after pos1 =", fingerprint(states[1]) + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
