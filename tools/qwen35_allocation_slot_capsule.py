#!/usr/bin/env python3
"""Find the first simulated allocation failure with small GPU buffers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    count = int(os.environ.get("QWEN35_ALLOC_SLOT_COUNT", "12000"))
    bytes_per = int(os.environ.get("QWEN35_ALLOC_SLOT_BYTES", "4096"))
    buffers = []
    failure = None
    for index in range(count):
        try:
            buffers.append(torch.empty(bytes_per // 2, dtype=torch.bfloat16, device="cuda"))
        except BaseException as error:
            failure = {"index": index, "type": type(error).__name__, "message": str(error)}
            break
        if index % 1000 == 0:
            print(f"ALLOC_SLOT progress={index + 1}", flush=True)
    torch.cuda.synchronize()
    result = {
        "schema": "amdgpu-sim.qwen35-allocation-slot-capsule.v1",
        "requested": count,
        "bytes_per_allocation": bytes_per,
        "allocated": len(buffers),
        "failure": failure,
        "passed": failure is None,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    del buffers
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
