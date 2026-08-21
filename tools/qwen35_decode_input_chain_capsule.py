#!/usr/bin/env python3
"""Capsule the layer-0 decode-input chain kernels at m=1 vs m=2.

The decode row's layer-0 input is bit-perfect against golden while its
output collapses (cos 0.37), and the post-conv mixed_qkv entering the
recurrent kernel is already uncorrelated (cos 0.012).  Between the perfect
input and the garbage output, only these run:

  1. the plain Triton ``_gemma_rmsnorm_kernel`` (grid [m], 8 warps) -- the
     final/input norm path that prefill exercises at m=2 and decode at m=1;
  2. the m=1 bf16 projection matmuls (n=8192 k=1024 for qkvz, n=32 k=1024
     for ba) whose n=8192 shape the lane notes once had no valid tiling.

Both are replayed here with golden inputs at m=2 (the proven-correct
prefill shape) and m=1 (the accused decode shape), each against a float64
host reference computed from identical bf16 inputs.  A case fails when its
max deviation exceeds the bf16 rounding band or its cosine leaves 0.999.

Exit 0 when every case matches; exit 1 on the first failing case, with the
m=2/m=1 contrast printed so the verdict names the phase.
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


def host_gemma_norm(x_bf16: torch.Tensor, w_bf16: torch.Tensor) -> torch.Tensor:
    x = x_bf16.float()
    var = (x * x).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + EPSILON) * (1.0 + w_bf16.float())).to(
        torch.bfloat16
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            __import__("os").environ.get(
                "QWEN35_DECODE_INPUT_CHAIN_OUTPUT",
                str(ROOT / "artifacts/qwen35-decode-input-chain-capsule/v1"),
            )
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The lane imports sglang through the overlay; the capsule rides the
    # lane environment, so the production kernel module is importable.
    from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import gemma_rmsnorm

    cos = torch.nn.functional.cosine_similarity
    ckpt = safe_open(str(CHECKPOINT), framework="pt", device="cpu")
    emb = ckpt.get_tensor("model.language_model.embed_tokens.weight")
    in_w = ckpt.get_tensor(f"{LAYER}.input_layernorm.weight")
    qkv_w = ckpt.get_tensor(f"{LAYER}.linear_attn.in_proj_qkv.weight")
    z_w = ckpt.get_tensor(f"{LAYER}.linear_attn.in_proj_z.weight")
    g = safe_open(str(GOLDEN), framework="pt", device="cpu")
    hidden = g.get_tensor("hidden_input")  # [4,1024] bf16 embeddings

    report = {
        "schema": "amdgpu-sim.qwen35-decode-input-chain-capsule.v1",
        "cases": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    artifacts: dict[str, torch.Tensor] = {}
    failures = 0

    def record(label, dev, ref):
        nonlocal failures
        dev_f = dev.float()
        ref_f = ref.float()
        c = float(cos(dev_f.reshape(-1), ref_f.reshape(-1), dim=0))
        maxd = float((dev_f.double() - ref_f.double()).abs().max())
        ok = c >= 0.999 and maxd <= 0.05
        report["cases"].append(
            {
                "label": label,
                "cos": round(c, 6),
                "max_abs_diff": round(maxd, 6),
                "ok": ok,
            }
        )
        artifacts[f"{label}_dev"] = dev_f.cpu()
        artifacts[f"{label}_ref"] = ref_f.cpu()
        if not ok:
            failures += 1
        print(
            f"[chain] {label}: cos={c:.6f} max_diff={maxd:.6f} ok={ok}",
            flush=True,
        )

    # Case 1: gemma_rmsnorm at m=2 (prefill shape) and m=1 (decode shape),
    # golden embeddings rows 0-1 / row 2 as inputs.
    for m, rows in ((2, (0, 1)), (1, (2,))):
        x = hidden[list(rows)].contiguous()
        ref = host_gemma_norm(x, in_w)
        dev = gemma_rmsnorm(x.to("cuda"), in_w.to("cuda"), EPSILON).cpu()
        record(f"gemmarmsnorm_m{m}", dev, ref)

    # Case 2: bf16 matmul m=1 projections vs m=2, random bf16 data (the
    # projections are pure linear maps; random inputs suffice) plus one
    # golden-driven case: norm(rows0-1) @ [qkv|z] for m=2 and norm(row2)
    # for m=1, reference = fp64 host from identical bf16 inputs.
    torch.manual_seed(0)
    for m, rows in ((2, (0, 1)), (1, (2,))):
        x = hidden[list(rows)].contiguous()
        xn = host_gemma_norm(x, in_w)
        w_cat = torch.cat([qkv_w, z_w], dim=0).contiguous()  # [8192,1024]
        w_gpu = w_cat.to("cuda")
        ref = (xn.double() @ w_cat.double().T).to(torch.bfloat16)
        dev = torch.matmul(xn.to("cuda"), w_gpu.T).cpu()
        record(f"proj8192_m{m}", dev, ref)

    for m in (2, 1):
        x = (torch.randn(m, 1024) * 0.5).to(torch.bfloat16)
        w = (torch.randn(32, 1024) * 0.02).to(torch.bfloat16)
        ref = (x.double() @ w.double().T).to(torch.bfloat16)
        dev = torch.matmul(x.to("cuda"), w.to("cuda").T).cpu()
        record(f"proj32_m{m}", dev, ref)

    report["passed"] = failures == 0
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_file(artifacts, str(args.output_dir / "tensors.safetensors"))
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="ascii"
    )
    print(f"[chain] passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
