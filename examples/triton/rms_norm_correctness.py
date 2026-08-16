#!/usr/bin/env python3

__import__("runpy").run_path(__file__.replace("rms_norm_correctness.py", "_gemsim_bootstrap.py"))["bootstrap"](__file__, "qwen35-rms-norm")
import json
import torch
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()
HIDDEN_SIZE = 1024
EPSILON = 1.0e-6
BLOCK_SIZE = 1024
NUM_WARPS = 8
ABSOLUTE_TOLERANCE = 0.015625
RELATIVE_TOLERANCE = 0.02
RELATIVE_L2_TOLERANCE = 0.005


@triton.jit
def qwen35_plain_gemma_rms_norm_kernel(
    x_ptr,
    raw_weight_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + row * BLOCK_SIZE + cols).to(tl.float32)
    raw_weight = tl.load(raw_weight_ptr + cols).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / BLOCK_SIZE
    rstd = 1.0 / tl.sqrt(variance + 1.0e-6)
    value = x * rstd * (1.0 + raw_weight)
    tl.store(
        out_ptr + row * BLOCK_SIZE + cols,
        value.to(tl.bfloat16),
    )


def plain_gemma_rms_norm(
    x: torch.Tensor,
    output: torch.Tensor,
    raw_weight: torch.Tensor,
    rows: int,
) -> None:
    qwen35_plain_gemma_rms_norm_kernel[(rows,)](
        x,
        raw_weight,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )


import hashlib


def run_case(rows: int, seed: int) -> dict:
    torch.manual_seed(seed)
    x = torch.randn((rows, HIDDEN_SIZE), device=DEVICE, dtype=torch.bfloat16)
    raw_weight = (
        0.125
        * torch.randn((HIDDEN_SIZE,), device=DEVICE, dtype=torch.bfloat16)
    ).to(torch.bfloat16)
    output_storage = torch.full(
        ((rows + 1) * HIDDEN_SIZE,),
        17.0,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    output = output_storage[: rows * HIDDEN_SIZE].view(rows, HIDDEN_SIZE)
    guard = output_storage[rows * HIDDEN_SIZE :].clone()
    input_before = x.clone()
    weight_before = raw_weight.clone()

    plain_gemma_rms_norm(x, output, raw_weight, rows)

    x_float = x.to(torch.float32)
    weight_float = raw_weight.to(torch.float32)
    variance = torch.mean(x_float * x_float, dim=-1, keepdim=True)
    reference = (
        x_float
        * torch.rsqrt(variance + EPSILON)
        * (1.0 + weight_float)
    )
    actual = output.to(torch.float32)
    error = torch.abs(actual - reference)
    tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * torch.abs(reference)
    finite = (
        torch.isfinite(actual)
        & torch.isfinite(reference)
        & torch.isfinite(error)
    )
    mismatch = int(torch.count_nonzero((~finite) | (error > tolerance)).item())
    nonfinite = int(torch.count_nonzero(~finite).item())
    finite_error = torch.where(finite, error, torch.zeros_like(error))
    reference_l2 = float(torch.linalg.vector_norm(reference).item())
    error_l2 = float(torch.linalg.vector_norm(finite_error).item())
    relative_l2 = error_l2 / reference_l2 if reference_l2 != 0.0 else error_l2
    input_unchanged = bool(torch.equal(x, input_before))
    weight_unchanged = bool(torch.equal(raw_weight, weight_before))
    guard_unchanged = bool(
        torch.equal(output_storage[rows * HIDDEN_SIZE :], guard)
    )
    correct = (
        mismatch == 0
        and nonfinite == 0
        and relative_l2 < RELATIVE_L2_TOLERANCE
        and input_unchanged
        and weight_unchanged
        and guard_unchanged
    )
    output_sha256 = hashlib.sha256(
        output.contiguous().view(torch.uint16).numpy().tobytes(order="C")
    ).hexdigest()
    return {
        "rows": rows,
        "seed": seed,
        "input_shape": [rows, HIDDEN_SIZE],
        "output_shape": [rows, HIDDEN_SIZE],
        "weight_shape": [HIDDEN_SIZE],
        "input_strides": [HIDDEN_SIZE, 1],
        "output_strides": [HIDDEN_SIZE, 1],
        "weight_strides": [1],
        "output_correct": correct,
        "mismatch_count": mismatch,
        "nonfinite_count": nonfinite,
        "all_values_finite": nonfinite == 0,
        "max_abs_error": float(torch.max(finite_error).item()),
        "relative_l2_error": relative_l2,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "relative_l2_tolerance": RELATIVE_L2_TOLERANCE,
        "input_unchanged": input_unchanged,
        "weight_unchanged": weight_unchanged,
        "guard_unchanged": guard_unchanged,
        "output_sha256": output_sha256,
        "program_count": rows,
    }


def main() -> int:
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "gemsim_amd" or target.arch != "gfx950":
        raise RuntimeError(f"unexpected Triton execution target: {target}")
    if DEVICE.type != "cpu":
        raise RuntimeError(
            f"gemsim_amd must expose a CPU staging device, got {DEVICE}"
        )

    launch_results = [run_case(1, 31), run_case(7, 37)]
    correct = all(result["output_correct"] for result in launch_results)
    print(
        json.dumps(
            {
                "schema": "amdgpu-sim.triton-qwen35-plain-rms-norm.v1",
                "backend": "gemsim_amd",
                "arch": target.arch,
                "kernel": "qwen35_plain_gemma_rms_norm_kernel",
                "dtype": "bfloat16",
                "accumulation_dtype": "float32",
                "epsilon": EPSILON,
                "effective_weight": "1.0 + raw_weight.float()",
                "input_shape_decode": [1, HIDDEN_SIZE],
                "input_shape_prefill": [7, HIDDEN_SIZE],
                "block_size": BLOCK_SIZE,
                "num_warps": NUM_WARPS,
                "launch_count": len(launch_results),
                "reuse": True,
                "launch_results": launch_results,
                "output_correct": correct,
                "mismatch_count": sum(
                    item["mismatch_count"] for item in launch_results
                ),
                "nonfinite_count": sum(
                    item["nonfinite_count"] for item in launch_results
                ),
                "max_abs_error": max(
                    item["max_abs_error"] for item in launch_results
                ),
                "max_relative_l2_error": max(
                    item["relative_l2_error"] for item in launch_results
                ),
                "fallback_count": 0,
            },
            sort_keys=True,
        )
    )
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
