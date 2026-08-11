#!/usr/bin/env python3

import json

import torch
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()
INPUT_ELEMENTS_PER_ROW = 7168
OUTPUT_ELEMENTS_PER_ROW = 3584
BLOCK_SIZE = 1024


@triton.jit
def silu_and_mul_kernel(x_ptr, out_ptr, scratch_ptr,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // 4
    block = pid % 4
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < 3584
    input_base = row * 7168
    gate = tl.load(x_ptr + input_base + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(
        x_ptr + input_base + 3584 + offsets,
        mask=mask,
        other=0,
    ).to(tl.float32)
    value = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + row * 3584 + offsets,
             value.to(tl.bfloat16), mask=mask)


def silu_and_mul(x: torch.Tensor, output: torch.Tensor,
                 scratch: torch.Tensor, rows: int):
    grid = (rows * 4,)
    silu_and_mul_kernel[grid](x, output, scratch, BLOCK_SIZE=BLOCK_SIZE)


def run_case(rows: int, seed: int) -> dict:
    torch.manual_seed(seed)
    x = torch.randn(
        (rows, INPUT_ELEMENTS_PER_ROW), device=DEVICE, dtype=torch.bfloat16
    )
    output = torch.full_like(x, 0x3f80)
    scratch = torch.full_like(x, 0x7e00)
    output_flat = output.flatten()
    written_elements = rows * OUTPUT_ELEMENTS_PER_ROW
    before_tail = output_flat[written_elements:].clone()
    silu_and_mul(x, output, scratch, rows)
    gate = x[:, :OUTPUT_ELEMENTS_PER_ROW].to(torch.float32)
    up = x[:, OUTPUT_ELEMENTS_PER_ROW:].to(torch.float32)
    reference = gate * torch.sigmoid(gate) * up
    actual = output_flat[:written_elements].view(
        rows, OUTPUT_ELEMENTS_PER_ROW
    ).to(torch.float32)
    error = torch.abs(actual - reference)
    tolerance = 0.015625 + 0.02 * torch.abs(reference)
    finite = (
        torch.isfinite(actual)
        & torch.isfinite(reference)
        & torch.isfinite(error)
    )
    tail_unchanged = bool(
        torch.equal(output_flat[written_elements:], before_tail)
    )
    mismatch = int(torch.count_nonzero((~finite) | (error > tolerance)).item())
    nonfinite = int(torch.count_nonzero(~finite).item())
    finite_error = torch.where(finite, error, torch.zeros_like(error))
    max_error = float(torch.max(finite_error).item())
    return {
        "rows": rows,
        "seed": seed,
        "output_correct": mismatch == 0,
        "mismatch_count": mismatch,
        "nonfinite_count": nonfinite,
        "all_values_finite": nonfinite == 0,
        "max_abs_error": max_error,
        "absolute_tolerance": 0.015625,
        "relative_tolerance": 0.02,
        "tail_unchanged": tail_unchanged,
        "program_count": rows * 4,
    }


def main() -> int:
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "gemsim_amd" or target.arch != "gfx950":
        raise RuntimeError(f"unexpected Triton execution target: {target}")
    if DEVICE.type != "cpu":
        raise RuntimeError(f"gemsim_amd must expose a CPU staging device, got {DEVICE}")
    launch_results = [run_case(1, 0), run_case(7, 1)]
    correct = all(
        result["output_correct"] and result["tail_unchanged"]
        for result in launch_results
    )
    print(json.dumps({
        "schema": "amdgpu-sim.triton-silu-and-mul.v1",
        "backend": "gemsim_amd",
        "arch": target.arch,
        "kernel": "silu_and_mul_kernel",
        "input_shape_decode": [1, INPUT_ELEMENTS_PER_ROW],
        "input_shape_prefill": [7, INPUT_ELEMENTS_PER_ROW],
        "output_width": OUTPUT_ELEMENTS_PER_ROW,
        "block_size": BLOCK_SIZE,
        "launch_count": len(launch_results),
        "reuse": True,
        "launch_results": launch_results,
        "output_correct": correct,
        "mismatch_count": sum(item["mismatch_count"] for item in launch_results),
        "max_abs_error": max(item["max_abs_error"] for item in launch_results),
        "fallback_count": 0,
    }, sort_keys=True))
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
