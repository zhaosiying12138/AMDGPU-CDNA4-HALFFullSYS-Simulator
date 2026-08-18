#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run an ordinary upstream Triton AMD softmax kernel on the simulated device.

This is deliberately a *small* end-to-end smoke, not a model run.  It exists
because a row-wise softmax is the cheapest kernel that still exercises the
parts of the stack a vector add cannot reach:

* two cross-lane block reductions (``tl.max`` and ``tl.sum``) which lower to
  DPP/permute sequences rather than plain VALU work,
* a masked load with a ``-inf`` fill so a non-power-of-two row width is legal,
* transcendental ``exp`` on the vector path,
* two different launch geometries compiled from one ``@triton.jit`` function.

Nothing here is simulator-aware.  It uses the unchanged upstream
``triton.backends.amd`` driver, ordinary ``torch.cuda`` allocation and copies,
and it checks its own numbers against a CPU reference computed in float64.
The caller supplies the runtime environment; see
``scripts/sandbox_triton_smoke.sh``.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
import triton
import triton.language as tl


# Softmax is not bitwise reproducible against a CPU reference: the device path
# uses exp2 with a fused multiply and reduces in a different order, so the
# result differs in the last few units in the last place. The gate is therefore
# an explicit float32 tolerance, and the row-sum identity is checked separately
# because it catches a wrong reduction that a loose element tolerance would not.
FLOAT32_RELATIVE_TOLERANCE = 1e-6
ROW_SUM_TOLERANCE = 1e-6


@triton.jit
def softmax_kernel(
    input_pointer,
    output_pointer,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row; the row is held in registers for both reductions."""

    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols
    values = tl.load(
        input_pointer + row * input_row_stride + columns,
        mask=mask,
        other=float("-inf"),
    )
    shifted = values - tl.max(values, axis=0)
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    tl.store(
        output_pointer + row * output_row_stride + columns,
        numerator / denominator,
        mask=mask,
    )


def reference_softmax(source: torch.Tensor) -> torch.Tensor:
    """Independent CPU oracle in float64, rounded back to float32."""

    wide = source.detach().to(torch.float64)
    shifted = wide - wide.max(dim=-1, keepdim=True).values
    exponentiated = torch.exp(shifted)
    return (exponentiated / exponentiated.sum(dim=-1, keepdim=True)).to(torch.float32)


def run_case(rows: int, cols: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    host_input = torch.randn((rows, cols), generator=generator, dtype=torch.float32)
    # A wide dynamic range makes an unshifted or wrongly reduced exponent
    # overflow to inf instead of quietly landing inside the tolerance.
    host_input[0] += 40.0
    host_input[-1] -= 40.0

    device_input = host_input.to("cuda")
    device_output = torch.empty_like(device_input)
    block_size = triton.next_power_of_2(cols)
    compiled = softmax_kernel[(rows,)](
        device_input,
        device_output,
        device_input.stride(0),
        device_output.stride(0),
        cols,
        BLOCK_SIZE=block_size,
    )
    torch.cuda.synchronize()

    actual = device_output.cpu()
    roundtrip = device_input.cpu()
    expected = reference_softmax(host_input)

    difference = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
    max_absolute = float(difference.max())
    denominator = expected.to(torch.float64).abs().clamp_min(torch.finfo(torch.float32).tiny)
    max_relative = float((difference / denominator).max())
    row_sums = actual.to(torch.float64).sum(dim=-1)
    max_row_sum_error = float((row_sums - 1.0).abs().max())

    checks = {
        "finite": bool(torch.isfinite(actual).all()),
        "within_relative_tolerance": max_relative <= FLOAT32_RELATIVE_TOLERANCE,
        "rows_sum_to_one": max_row_sum_error <= ROW_SUM_TOLERANCE,
        "input_unchanged": torch.equal(roundtrip, host_input),
        "output_is_device_resident": device_output.is_cuda,
        "output_does_not_alias_input": (
            device_output.untyped_storage().data_ptr()
            != device_input.untyped_storage().data_ptr()
        ),
    }
    return {
        "rows": rows,
        "cols": cols,
        "block_size": block_size,
        "masked": block_size != cols,
        "kernel_name": compiled.name,
        "kernel_hash": compiled.hash,
        "num_warps": compiled.metadata.num_warps,
        "shared_memory_bytes": compiled.metadata.shared,
        "max_absolute_error": max_absolute,
        "max_relative_error": max_relative,
        "max_row_sum_error": max_row_sum_error,
        "checks": checks,
        "correct": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260818)
    arguments = parser.parse_args()

    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("the upstream ROCm PyTorch device is unavailable")
    driver = triton.runtime.driver.active
    target = driver.get_current_target()
    if target.backend != "hip":
        raise RuntimeError(f"unexpected upstream Triton backend: {target}")
    # The point of this smoke is that *unchanged* upstream Triton reaches the
    # simulated device, so refuse the project's gemsim_hip subclass here.
    if type(driver).__module__ != "triton.backends.amd.driver":
        raise RuntimeError(f"unexpected upstream Triton driver: {type(driver)}")

    cases = [
        # A power-of-two row: no masking, the plain reduction path.
        run_case(rows=4, cols=64, seed=arguments.seed),
        # A non-power-of-two row: the -inf masked tail must not reach the max.
        run_case(rows=3, cols=100, seed=arguments.seed + 1),
    ]
    result = {
        "schema": "amdgpu-sim.upstream-triton-amd-softmax.v1",
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "triton": triton.__version__,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "driver": {
            "module": type(driver).__module__,
            "class": type(driver).__name__,
            "backend": target.backend,
            "arch": target.arch,
            "warp_size": target.warp_size,
        },
        "float32_relative_tolerance": FLOAT32_RELATIVE_TOLERANCE,
        "row_sum_tolerance": ROW_SUM_TOLERANCE,
        "cases": cases,
        "correct": all(case["correct"] for case in cases),
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if not result["correct"]:
        raise RuntimeError("upstream Triton AMD softmax output mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
