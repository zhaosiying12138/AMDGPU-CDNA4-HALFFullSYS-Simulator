#!/usr/bin/env python3
"""Reproduce the 9B first-MLP GEMM under model-sized allocation pressure."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(
    os.environ.get(
        "QWEN35_AITER_PRESSURE_OUTPUT",
        str(ROOT / "artifacts/qwen35-aiter-gemm-pressure-capsule/v1"),
    )
)
PRESSURE_GIB = float(os.environ.get("QWEN35_AITER_PRESSURE_GIB", "0"))
CHUNK_BYTES = int(
    os.environ.get("QWEN35_AITER_PRESSURE_CHUNK_MIB", "256")
) * 1024 * 1024


def memory_snapshot() -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "free": int(free),
        "total": int(total),
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }


def main() -> int:
    if PRESSURE_GIB < 0:
        raise ValueError("QWEN35_AITER_PRESSURE_GIB must be non-negative")
    if CHUNK_BYTES < 2 or CHUNK_BYTES % 2:
        raise ValueError("QWEN35_AITER_PRESSURE_CHUNK_MIB must be positive")
    pressure_bytes = int(PRESSURE_GIB * (1024**3))
    pressure: list[torch.Tensor] = []
    before = memory_snapshot()
    remaining = pressure_bytes
    while remaining:
        size = min(CHUNK_BYTES, remaining)
        pressure.append(torch.empty((size // 2,), device="cuda", dtype=torch.bfloat16))
        remaining -= size
    after_pressure = memory_snapshot()

    torch.manual_seed(0)
    x = torch.randn((2, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((3072, 4096), device="cuda", dtype=torch.bfloat16)
    before_gemm = memory_snapshot()
    from aiter.tuned_gemm import tgemm

    actual = tgemm.mm(x, weight, otype=torch.bfloat16)
    torch.cuda.synchronize()
    after_gemm = memory_snapshot()
    expected = torch.nn.functional.linear(x.float(), weight.float()).to(torch.bfloat16)
    actual_cpu = actual.detach().cpu().contiguous()
    expected_cpu = expected.detach().cpu().contiguous()
    delta = (actual_cpu.float() - expected_cpu.float()).abs()
    report = {
        "schema": "amdgpu-sim.qwen35-aiter-gemm-pressure-capsule.v1",
        "shape": [2, 3072, 4096],
        "dtype": "bfloat16",
        "pressure_gib": PRESSURE_GIB,
        "pressure_bytes": pressure_bytes,
        "pressure_chunk_bytes": CHUNK_BYTES,
        "memory": {
            "before": before,
            "after_pressure": after_pressure,
            "before_gemm": before_gemm,
            "after_gemm": after_gemm,
        },
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
    del pressure
    gc.collect()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
