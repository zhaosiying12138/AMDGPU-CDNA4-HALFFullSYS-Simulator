#!/usr/bin/env python3
"""Enumerate and dispatch on every simulated GPU the topology publishes.

Tensor-parallel inference is blocked long before a collective if the stack
exposes one agent: an engine calls set_device(rank) on every rank and dies with
"invalid device ordinal". This is the smallest probe that answers the question
directly -- how many agents exist, and does each one really execute -- without
a model, an engine, or a collective.

Point HSA_MODEL_TOPOLOGY at a tree from
projects/self-amdgpu-runtime/tools/hsakmt-model-topology.py to choose the
width. One managed gem5 session serves every logical GPU in this process, so a
single process is enough; separate ranks get separate simulators.
"""

from __future__ import annotations

import json
import os

import torch


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("the ROCm simulator device is unavailable")

    count = torch.cuda.device_count()
    devices = []
    checks = {"every_device_dispatched": count > 0}
    for index in range(count):
        torch.cuda.set_device(index)
        host_input = torch.arange(8, dtype=torch.float32)
        device_input = host_input.to(f"cuda:{index}")
        # A distinct bias per device makes a result that silently came from
        # another agent's buffer impossible to mistake for a pass.
        bias = float(index) + 100.0
        device_output = device_input + bias
        torch.cuda.synchronize()
        actual = device_output.cpu()
        expected = host_input + bias
        exact = torch.equal(actual, expected)
        checks["every_device_dispatched"] = (
            checks["every_device_dispatched"] and exact
        )
        devices.append(
            {
                "ordinal": index,
                "name": torch.cuda.get_device_name(index),
                "current_device": torch.cuda.current_device(),
                "bias": bias,
                "values": actual.tolist(),
                "bitwise_exact": exact,
                "output_is_cuda": device_output.is_cuda,
                "output_fresh": (
                    device_output.untyped_storage().data_ptr()
                    != device_input.untyped_storage().data_ptr()
                ),
            }
        )

    checks["outputs_fresh"] = all(entry["output_fresh"] for entry in devices)
    checks["outputs_on_device"] = all(entry["output_is_cuda"] for entry in devices)
    checks["set_device_followed"] = all(
        entry["current_device"] == entry["ordinal"] for entry in devices
    )
    result = {
        "schema": "amdgpu-sim.multi-gpu-quickstart.v1",
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "topology": os.environ.get("HSA_MODEL_TOPOLOGY", ""),
        "device_count": count,
        "devices": devices,
        "checks": checks,
        "correct": all(checks.values()),
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if not result["correct"]:
        raise RuntimeError("multi-GPU quickstart output mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
