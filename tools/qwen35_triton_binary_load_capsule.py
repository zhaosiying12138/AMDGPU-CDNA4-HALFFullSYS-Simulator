#!/usr/bin/env python3
"""Load a selected cached Triton HSACO directly on one simulated device."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = next(
    (ROOT / "artifacts/zcode-cache/triton").glob(
        "3GS*/_gemma_rmsnorm_kernel.hsaco"
    ),
    None,
)


def main() -> int:
    if DEFAULT_BINARY is None:
        raise RuntimeError("3GS gemma HSACO is absent")
    device = int(os.environ.get("QWEN35_TRITON_BINARY_DEVICE", "2"))
    torch.cuda.set_device(device)
    binary = Path(os.environ.get("QWEN35_TRITON_BINARY", str(DEFAULT_BINARY)))
    from triton.runtime import driver

    data = binary.read_bytes()
    module, function, n_regs, n_spills, n_threads = driver.active.utils.load_binary(
        "_gemma_rmsnorm_kernel", data, 0, device
    )
    report = {
        "schema": "amdgpu-sim.qwen35-triton-binary-load-capsule.v1",
        "device": device,
        "binary": str(binary),
        "bytes": len(data),
        "n_regs": n_regs,
        "n_spills": n_spills,
        "n_threads": n_threads,
        "passed": module is not None and function is not None,
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
