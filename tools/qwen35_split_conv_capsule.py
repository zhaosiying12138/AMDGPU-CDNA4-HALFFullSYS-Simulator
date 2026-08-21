#!/usr/bin/env python3
"""Capsule the last two layer-0 decode stages with REAL trajectory data.

Proven so far (all on the simulator, cos ~1.0): the decode embedding input,
the plain gemma RMSNorm at m=1/m=2, and the m=1 projection matmuls
(n=8192, n=32).  Proven wrong in the engine: the post-conv mixed_qkv
entering the recurrent decode kernel (cos 0.012 vs the expected
trajectory).  Only two production kernels sit between them:

  1. ``fused_qkvzba_split_reshape_cat_contiguous`` (grid [m, 16]) -- splits
     the fused projection into mixed_qkv / z / b / a;
  2. ``causal_conv1d_update`` (Triton, grid [m, 24]-class) -- rolls the conv
     state line and produces the mixed_qkv the recurrent kernel consumes.

Both are replayed here with the exact golden trajectory tensors: the
offline expected pipeline (embeddings -> gemma norm -> bf16 projections)
recomputes the pre-conv mixed_qkv rows and the conv line states after each
position.  Each kernel runs at m=2 (prefill-proven) and m=1 (the decode
shape) and is compared against the host recurrence from the same inputs.
The conv case starts from the true post-prefill line (zero, t0, t1) and
feeds t2, checking BOTH the kernel's output row and the rolled line.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = (
    ROOT
    / "artifacts/qwen35-nvidia-golden/20260812-decode4-max24-v1/results.safetensors"
)
CHECKPOINT = (
    ROOT / "models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
)
EPSILON = 1.0e-6
LAYER = "model.language_model.layers.0"
NUM_HEADS, HEAD_DIM = 16, 128
QKV_DIM = 3 * NUM_HEADS * HEAD_DIM


def bf16_linear(x, w):
    return torch.matmul(x.float(), w.float().T).to(torch.bfloat16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            __import__("os").environ.get(
                "QWEN35_SPLIT_CONV_OUTPUT",
                str(ROOT / "artifacts/qwen35-split-conv-capsule/v1"),
            )
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import (
        causal_conv1d_update,
    )

    cos = torch.nn.functional.cosine_similarity
    ckpt = safe_open(str(CHECKPOINT), framework="pt", device="cpu")
    emb = ckpt.get_tensor("model.language_model.embed_tokens.weight")
    W = {
        k: ckpt.get_tensor(f"{LAYER}.{n}")
        for k, n in (
            ("norm", "input_layernorm.weight"),
            ("qkv", "linear_attn.in_proj_qkv.weight"),
            ("z", "linear_attn.in_proj_z.weight"),
            ("b", "linear_attn.in_proj_b.weight"),
            ("a", "linear_attn.in_proj_a.weight"),
            ("conv", "linear_attn.conv1d.weight"),
        )
    }
    conv_bias = None
    try:
        conv_bias = ckpt.get_tensor(f"{LAYER}.linear_attn.conv1d.bias")
    except Exception:
        pass

    tokens = [248044, 266, 27841, 27841]
    hidden = torch.cat([emb[t : t + 1] for t in tokens], 0).contiguous()
    x = hidden.float()
    normalized = (
        x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPSILON)
        * (1.0 + W["norm"].float())
    ).to(torch.bfloat16)
    qkvz = torch.cat(
        [bf16_linear(normalized, W["qkv"]), bf16_linear(normalized, W["z"])],
        dim=-1,
    ).contiguous()
    ba = torch.cat(
        [bf16_linear(normalized, W["b"]), bf16_linear(normalized, W["a"])],
        dim=-1,
    ).contiguous()
    pre_conv = qkvz[:, :QKV_DIM].contiguous()

    report = {
        "schema": "amdgpu-sim.qwen35-split-conv-capsule.v1",
        "cases": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    artifacts: dict[str, torch.Tensor] = {}
    failures = 0

    def record(label, dev, ref):
        nonlocal failures
        dev_f, ref_f = dev.float(), ref.float()
        c = float(cos(dev_f.reshape(-1), ref_f.reshape(-1), dim=0))
        maxd = float((dev_f.double() - ref_f.double()).abs().max())
        ok = c >= 0.999 and maxd <= 0.05
        report["cases"].append(
            {"label": label, "cos": round(c, 6), "max_abs_diff": round(maxd, 6), "ok": ok}
        )
        artifacts[f"{label}_dev"] = dev_f.cpu()
        artifacts[f"{label}_ref"] = ref_f.cpu()
        if not ok:
            failures += 1
        print(f"[splitconv] {label}: cos={c:.6f} max={maxd:.6f} ok={ok}", flush=True)

    # ---- Stage 1: fused split kernel at m=2 and m=1 ----
    for m, rows in ((2, (0, 1)), (1, (2,))):
        qkvz_m = qkvz[list(rows)].contiguous()
        ba_m = ba[list(rows)].contiguous()
        out = fused_qkvzba_split_reshape_cat_contiguous(
            qkvz_m.to("cuda"), ba_m.to("cuda"), NUM_HEADS, NUM_HEADS, HEAD_DIM, HEAD_DIM
        )
        dev_qkv, dev_z, dev_b, dev_a = (t.cpu() for t in out)
        record(f"split_qkv_m{m}", dev_qkv, qkvz_m[:, :QKV_DIM])
        record(
            f"split_z_m{m}",
            dev_z,
            qkvz_m[:, QKV_DIM:].reshape(m, NUM_HEADS, HEAD_DIM),
        )
        record(f"split_b_m{m}", dev_b, ba_m[:, :NUM_HEADS])
        record(f"split_a_m{m}", dev_a, ba_m[:, NUM_HEADS:])

    # ---- Stage 2: causal_conv1d_update on the real trajectory ----
    # Conv line after prefill (positions 0,1) = (zero, t0, t1).
    line_after_prefill = torch.stack(
        (
            torch.zeros(QKV_DIM, dtype=torch.bfloat16),
            pre_conv[0],
            pre_conv[1],
        ),
        dim=1,
    ).contiguous()  # [6144, 3]
    conv_w = W["conv"].float().reshape(QKV_DIM, 4)

    def conv_step_host(line, current_pre):
        cur = current_pre.float()
        acc = (
            line[:, 0].float() * conv_w[:, 0]
            + line[:, 1].float() * conv_w[:, 1]
            + line[:, 2].float() * conv_w[:, 2]
            + cur * conv_w[:, 3]
        )
        if conv_bias is not None:
            acc = acc + conv_bias.float()
        out = (acc * torch.sigmoid(acc)).to(torch.bfloat16)
        new_line = torch.stack((line[:, 1], line[:, 2], current_pre), dim=1)
        return out, new_line

    exp_out2, exp_line2 = conv_step_host(line_after_prefill, pre_conv[2])
    exp_out3, exp_line3 = conv_step_host(exp_line2, pre_conv[3])

    for m, tag, cur_row, start_line, exp_out, exp_line in (
        (1, "pos2", 2, line_after_prefill, exp_out2, exp_line2),
        (1, "pos3", 3, exp_line2, exp_out3, exp_line3),
    ):
        pool = start_line.unsqueeze(0).to("cuda")  # [1, 6144, 3]
        indices = torch.zeros(1, dtype=torch.int32, device="cuda")
        dev = causal_conv1d_update(
            pre_conv[cur_row : cur_row + 1].to("cuda"),
            pool,
            W["conv"].to("cuda"),
            conv_bias.to("cuda") if conv_bias is not None else None,
            "silu",
            conv_state_indices=indices,
        ).cpu()
        record(f"convupd_{tag}_out", dev, exp_out.unsqueeze(0))
        record(f"convupd_{tag}_line", pool.cpu()[0], exp_line)

    report["passed"] = failures == 0
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_file(artifacts, str(args.output_dir / "tensors.safetensors"))
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="ascii"
    )
    print(f"[splitconv] passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
