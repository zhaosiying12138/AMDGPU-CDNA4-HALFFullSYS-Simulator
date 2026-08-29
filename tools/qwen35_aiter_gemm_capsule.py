#!/usr/bin/env python3
"""Minimal AITER BF16 GEMM probe for the first Qwen3.5-9B MLP shape."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(
    __import__("os").environ.get(
        "QWEN35_AITER_GEMM_OUTPUT",
        str(ROOT / "artifacts/qwen35-aiter-gemm-capsule/v1"),
    )
)


def main() -> int:
    torch.manual_seed(0)
    x = torch.randn((2, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((3072, 4096), device="cuda", dtype=torch.bfloat16)

    from aiter.tuned_gemm import tgemm

    actual = tgemm.mm(x, weight, otype=torch.bfloat16)
    torch.cuda.synchronize()
    expected = torch.nn.functional.linear(x.float(), weight.float()).to(
        torch.bfloat16
    )
    actual_cpu = actual.detach().cpu().contiguous()
    expected_cpu = expected.detach().cpu().contiguous()
    delta = (actual_cpu.float() - expected_cpu.float()).abs()
    report = {
        "schema": "amdgpu-sim.qwen35-aiter-gemm-capsule.v1",
        "shape": [2, 3072, 4096],
        "dtype": "bfloat16",
        "max_abs_error": float(delta.max().item()),
        "mismatch_count": int(
            (delta > 0.03125 + 0.03 * expected_cpu.float().abs()).sum().item()
        ),
        "actual_sha256": hashlib.sha256(
            actual_cpu.view(torch.uint16).numpy().tobytes()
        ).hexdigest(),
        "expected_sha256": hashlib.sha256(
            expected_cpu.view(torch.uint16).numpy().tobytes()
        ).hexdigest(),
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
