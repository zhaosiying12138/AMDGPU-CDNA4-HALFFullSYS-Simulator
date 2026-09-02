#!/usr/bin/env python3
"""Reproduce concurrent four-device Triton Gemma RMSNorm initialization."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(
    os.environ.get(
        "QWEN35_GEMMA_RMSNORM_TP4_OUTPUT",
        str(ROOT / "artifacts/qwen35-gemma-rmsnorm-tp4-capsule/v1"),
    )
)


def _worker(rank: int, reports: mp.Queue) -> None:
    import torch

    torch.cuda.set_device(rank)
    torch.manual_seed(20260830 + rank)
    x = torch.randn((2, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((4096,), device="cuda", dtype=torch.bfloat16)
    from aiter.tuned_gemm import tgemm

    gemm_weight = torch.randn((3072, 4096), device="cuda", dtype=torch.bfloat16)
    tgemm.mm(x, gemm_weight, otype=torch.bfloat16)
    torch.cuda.synchronize()
    from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import gemma_rmsnorm

    actual = gemma_rmsnorm(x, weight, 1.0e-6)
    torch.cuda.synchronize()
    reports.put({"rank": rank, "shape": list(actual.shape), "ok": True})


def main() -> int:
    ctx = mp.get_context("spawn")
    reports = ctx.Queue()
    workers = [ctx.Process(target=_worker, args=(rank, reports)) for rank in range(4)]
    for worker in workers:
        worker.start()
    deadline = time.monotonic() + 240.0
    while any(worker.is_alive() for worker in workers) and time.monotonic() < deadline:
        time.sleep(0.1)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        worker.join(10.0)
    observed = []
    while True:
        try:
            observed.append(reports.get_nowait())
        except queue.Empty:
            break
    observed.sort(key=lambda item: item["rank"])
    report = {
        "schema": "amdgpu-sim.qwen35-gemma-rmsnorm-tp4-capsule.v1",
        "observed": observed,
        "exitcodes": [worker.exitcode for worker in workers],
        "passed": len(observed) == 4 and all(worker.exitcode == 0 for worker in workers),
    }
    OUTPUT.mkdir(mode=0o700, parents=True, exist_ok=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
