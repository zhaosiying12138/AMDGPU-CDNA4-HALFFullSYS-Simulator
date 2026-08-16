#!/usr/bin/env python3
"""Run ordinary upstream ROCm PyTorch operations on the simulated device."""

from __future__ import annotations

import hashlib
import json

import torch


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = (
        tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("the ROCm simulator device is unavailable")

    host_input = torch.arange(-8, 8, dtype=torch.float32)
    device_input = host_input.to("cuda")
    device_added = torch.add(device_input, 1.25)
    device_sigmoid = torch.sigmoid(device_input)
    device_sum = torch.sum(device_input)
    torch.cuda.synchronize()

    roundtrip = device_input.cpu()
    added = device_added.cpu()
    sigmoid = device_sigmoid.cpu()
    reduced = device_sum.cpu()
    expected_added = torch.add(host_input, 1.25)
    expected_sigmoid = torch.sigmoid(host_input)
    expected_sum = torch.sum(host_input)

    outputs = (device_added, device_sigmoid, device_sum)
    output_storage_addresses = [
        output.untyped_storage().data_ptr() for output in outputs
    ]
    checks = {
        "copy_bitwise": torch.equal(roundtrip, host_input),
        "add_bitwise": torch.equal(added, expected_added),
        "sigmoid_bitwise": torch.equal(sigmoid, expected_sigmoid),
        "sum_bitwise": torch.equal(reduced, expected_sum),
        "input_unchanged": torch.equal(roundtrip, host_input),
        "outputs_are_cuda": all(output.is_cuda for output in outputs),
        "outputs_fresh": all(
            address != device_input.untyped_storage().data_ptr()
            for address in output_storage_addresses
        ),
        "outputs_nonalias": len(set(output_storage_addresses)) == len(outputs),
    }
    actual_tensors = {
        "copy": roundtrip,
        "add": added,
        "sigmoid": sigmoid,
        "sum": reduced,
    }
    expected_tensors = {
        "copy": host_input,
        "add": expected_added,
        "sigmoid": expected_sigmoid,
        "sum": expected_sum,
    }
    result = {
        "schema": "amdgpu-sim.upstream-rocm-pytorch-quickstart.v1",
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "operations": ["copy", "add", "sigmoid", "sum"],
        "tensor_contract": {
            "dtype": "float32",
            "input_shape": [16],
            "input_device": str(device_input.device),
            "actual_sha256": {
                name: tensor_sha256(tensor) for name, tensor in actual_tensors.items()
            },
            "expected_sha256": {
                name: tensor_sha256(tensor) for name, tensor in expected_tensors.items()
            },
        },
        "checks": checks,
        "correct": all(checks.values()),
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if not result["correct"]:
        raise RuntimeError("ROCm PyTorch quickstart output mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
