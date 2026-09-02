#!/usr/bin/env python3
"""Isolate the post-attention Gemma RMSNorm mismatch against a CPU reference.

The layer gate's ordinal-19 ``post_attention_rms_norm`` mismatch could be
either an upstream GEMM-tolerance input difference propagating through the
norm or a defect inside the fused-add Gemma RMSNorm Triton kernel itself.
This capsule separates the two: it replays the pinned
``gemma_fused_add_rmsnorm`` kernel on the simulator with the gate capture's
own inputs (input_0=x, input_1=residual, input_2=weight), computes the same
Gemma formula in CPU float32, and compares both against each other and the
frozen golden.  If the kernel matches the CPU reference within bf16
tolerance, its math is correct and the gate divergence is input propagation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = (
    ROOT / "artifacts/lanes/zcode-lgate-3/layer-gate/first-mismatch"
)


def _metrics(actual: torch.Tensor, reference: torch.Tensor, atol, rtol) -> dict:
    a = actual.float().reshape(-1)
    b = reference.float().reshape(-1)
    delta = a - b
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(delta.abs().max().item()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(delta).item()
            / max(torch.linalg.vector_norm(b).item(), 1.0e-30)
        ),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(a[None], b[None]).item()
        ),
        "mismatch_count": int(
            ((delta.abs() > (atol + rtol * b.abs())).sum().item())
        ),
        "atol": atol,
        "rtol": rtol,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            __import__("os").environ.get(
                "QWEN35_NORM_ISOLATION_OUTPUT",
                str(ROOT / "artifacts/qwen35-norm-isolation/20260820-layer0-v1"),
            )
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    tensors = load_file(str(args.capture / "tensors.safetensors"))
    device = torch.device(args.device)
    # Import only after the lane has installed its backend and simulator.
    from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (
        gemma_fused_add_rmsnorm,
    )

    x = tensors["input_0"].to(device)
    residual = tensors["input_1"].to(device)
    weight = tensors["input_2"].to(device)
    weight_cpu = tensors["input_2"].float().cpu()

    normed, res_out = gemma_fused_add_rmsnorm(x, residual, weight, args.eps)
    torch.cuda.synchronize()
    normed = normed.detach().cpu()
    res_out = res_out.detach().cpu()

    # CPU float32 reference of the same Gemma formula the kernel implements:
    # new_residual = x + residual; normed = new_residual * rsqrt(mean(sqr) +
    # eps) * (1 + weight), with the accumulation in float32 throughout.
    xf = x.float().cpu()
    rf = residual.float().cpu()
    new_residual_ref = (xf + rf).to(torch.bfloat16)
    variance = (xf + rf).pow(2).mean(dim=-1, keepdim=True)
    normed_ref = (
        (xf + rf) * torch.rsqrt(variance + args.eps) * (1.0 + weight_cpu)
    ).to(torch.bfloat16)

    result = {
        "schema": "amdgpu-sim.qwen35-norm-isolation-capsule.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "capture": str(args.capture),
        "kernel_vs_cpu_reference": _metrics(
            normed.reshape(-1, normed.shape[-1]),
            normed_ref.reshape(-1, normed_ref.shape[-1]),
            0.01,
            0.01,
        ),
        "kernel_vs_golden": _metrics(
            normed, tensors["expected"], 0.01, 0.01
        ),
        "cpu_reference_vs_golden": _metrics(
            normed_ref, tensors["expected"], 0.01, 0.01
        ),
        "residual_out_vs_reference": _metrics(
            res_out, new_residual_ref, 0.01, 0.01
        ),
    }
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    save_file(
        {"kernel_normed": normed, "cpu_normed_ref": normed_ref.contiguous()},
        str(args.output_dir / "tensors.safetensors"),
        metadata={"schema": result["schema"]},
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
