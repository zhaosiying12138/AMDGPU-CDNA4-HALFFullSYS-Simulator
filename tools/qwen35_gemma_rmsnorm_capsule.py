#!/usr/bin/env python3
"""Exercise the exact 9B Gemma RMSNorm Triton kernel without model weights."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(
    os.environ.get(
        "QWEN35_GEMMA_RMSNORM_OUTPUT",
        str(ROOT / "artifacts/qwen35-gemma-rmsnorm-capsule/v1"),
    )
)


def main() -> int:
    torch.manual_seed(20260830)
    x = torch.randn((2, 8192), device="cuda", dtype=torch.bfloat16)
    if os.environ.get("QWEN35_GEMMA_RMSNORM_NONCONTIG"):
        x = x[:, ::2]
    else:
        x = x[:, :4096]
    weight = torch.randn((4096,), device="cuda", dtype=torch.bfloat16)
    from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import gemma_rmsnorm

    actual = gemma_rmsnorm(x, weight, 1.0e-6)
    torch.cuda.synchronize()
    expected = (
        x.float()
        * torch.rsqrt((x.float() * x.float()).mean(dim=-1, keepdim=True) + 1.0e-6)
        * (1.0 + weight)
    ).to(torch.bfloat16)
    delta = (actual.float() - expected.float()).abs()
    report = {
        "schema": "amdgpu-sim.qwen35-gemma-rmsnorm-capsule.v1",
        "shape": list(x.shape),
        "max_abs_error": float(delta.max().item()),
        "mismatch_count": int(
            (delta > 0.03125 + 0.03 * expected.float().abs()).sum().item()
        ),
    }
    report["passed"] = report["mismatch_count"] == 0
    OUTPUT.mkdir(mode=0o700, parents=True, exist_ok=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
