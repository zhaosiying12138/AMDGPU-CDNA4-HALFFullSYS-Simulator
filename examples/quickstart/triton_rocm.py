#!/usr/bin/env python3
"""Compile and run ordinary upstream Triton AMD kernels on the ROCm device."""

from __future__ import annotations

import hashlib
import json

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(left, right, output, size: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    values = tl.load(left + offsets, mask=mask) + tl.load(right + offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


@triton.jit
def transform_kernel(source, output, size: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    values = tl.load(source + offsets, mask=mask)
    transformed = tl.where(values > 0.0, values * values + 0.5, -values + 0.25)
    tl.store(output + offsets, transformed, mask=mask)


@triton.jit
def reduce_kernel(source, output, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(source + offsets)
    tl.store(output, tl.sum(values, axis=0))


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = (
        tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def compiled_kernel_record(kernel) -> dict[str, object]:
    target = kernel.metadata.target
    binary = bytes(kernel.kernel)
    return {
        "name": kernel.name,
        "cache_hash": kernel.hash,
        "binary_bytes": len(binary),
        "binary_sha256": hashlib.sha256(binary).hexdigest(),
        "target": {
            "backend": target.backend,
            "arch": target.arch,
            "warp_size": target.warp_size,
        },
        "num_warps": kernel.metadata.num_warps,
        "shared_memory_bytes": kernel.metadata.shared,
    }


def main() -> int:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("the upstream ROCm PyTorch device is unavailable")
    driver = triton.runtime.driver.active
    target = driver.get_current_target()
    if target.backend != "hip" or target.arch != "gfx950":
        raise RuntimeError(f"unexpected upstream Triton target: {target}")
    if type(driver).__module__ != "triton.backends.amd.driver":
        raise RuntimeError(f"unexpected upstream Triton driver: {type(driver)}")

    size = 256
    host_left = torch.arange(size, dtype=torch.float32) / 8.0 - 16.0
    host_right = torch.arange(size - 1, -1, -1, dtype=torch.float32) / 16.0
    host_reduce = torch.remainder(torch.arange(size, dtype=torch.float32), 17.0) - 8.0
    left = host_left.to("cuda")
    right = host_right.to("cuda")
    reduction_input = host_reduce.to("cuda")
    added = torch.empty_like(left)
    transformed = torch.empty_like(left)
    reduced = torch.empty((), dtype=torch.float32, device="cuda")

    compiled_add = add_kernel[(triton.cdiv(size, 128),)](
        left, right, added, size=size, BLOCK=128
    )
    compiled_transform = transform_kernel[(triton.cdiv(size, 128),)](
        left, transformed, size=size, BLOCK=128
    )
    compiled_reduce = reduce_kernel[(1,)](reduction_input, reduced, BLOCK=256)
    torch.cuda.synchronize()

    input_roundtrip = {
        "left": left.cpu(),
        "right": right.cpu(),
        "reduction_input": reduction_input.cpu(),
    }
    actual = {
        "add": added.cpu(),
        "transform": transformed.cpu(),
        "reduce": reduced.cpu(),
    }
    expected = {
        "add": host_left + host_right,
        "transform": torch.where(
            host_left > 0.0,
            host_left * host_left + 0.5,
            -host_left + 0.25,
        ),
        "reduce": torch.sum(host_reduce),
    }
    checks = {
        "add_bitwise": torch.equal(actual["add"], expected["add"]),
        "transform_bitwise": torch.equal(
            actual["transform"], expected["transform"]
        ),
        "reduce_bitwise": torch.equal(actual["reduce"], expected["reduce"]),
        "inputs_unchanged": (
            torch.equal(input_roundtrip["left"], host_left)
            and torch.equal(input_roundtrip["right"], host_right)
            and torch.equal(input_roundtrip["reduction_input"], host_reduce)
        ),
        "outputs_are_cuda": all(
            tensor.is_cuda for tensor in (added, transformed, reduced)
        ),
        "outputs_nonalias": len(
            {
                tensor.untyped_storage().data_ptr()
                for tensor in (left, right, reduction_input, added, transformed, reduced)
            }
        )
        == 6,
    }
    result = {
        "schema": "amdgpu-sim.upstream-triton-amd-quickstart.v1",
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "triton": triton.__version__,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "driver": {
            "module": type(driver).__module__,
            "class": type(driver).__name__,
            "backend": target.backend,
            "arch": target.arch,
            "warp_size": target.warp_size,
        },
        "kernels": ["add_kernel", "transform_kernel", "reduce_kernel"],
        "compiled_kernels": [
            compiled_kernel_record(compiled_add),
            compiled_kernel_record(compiled_transform),
            compiled_kernel_record(compiled_reduce),
        ],
        "tensor_contract": {
            "dtype": "float32",
            "element_count": size,
            "input_actual_sha256": {
                name: tensor_sha256(tensor)
                for name, tensor in input_roundtrip.items()
            },
            "input_expected_sha256": {
                "left": tensor_sha256(host_left),
                "right": tensor_sha256(host_right),
                "reduction_input": tensor_sha256(host_reduce),
            },
            "actual_sha256": {
                name: tensor_sha256(tensor) for name, tensor in actual.items()
            },
            "expected_sha256": {
                name: tensor_sha256(tensor) for name, tensor in expected.items()
            },
        },
        "checks": checks,
        "target_feedback_from_oracle": False,
        "correct": all(checks.values()),
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if not result["correct"]:
        raise RuntimeError("upstream Triton AMD quickstart output mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
