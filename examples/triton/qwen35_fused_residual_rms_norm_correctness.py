#!/usr/bin/env python3

import runpy

runpy.run_path(
    __file__.replace(
        "qwen35_fused_residual_rms_norm_correctness.py",
        "_gemsim_bootstrap.py",
    )
)["bootstrap"](__file__, "qwen35-rms-norm")

import hashlib
import json

import torch
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()
HIDDEN_SIZE = 1024
EPSILON = 1.0e-6
BLOCK_SIZE = 1024
NUM_WARPS = 8
ABSOLUTE_TOLERANCE = 0.01
RELATIVE_TOLERANCE = 0.01
RELATIVE_L2_TOLERANCE = 0.005


@triton.jit
def qwen35_fused_residual_gemma_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    raw_weight_ptr,
    y_ptr,
    residual_out_ptr,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + row * BLOCK_SIZE + cols).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * BLOCK_SIZE + cols
    ).to(tl.float32)
    summed = x + residual
    mean_square = tl.sum(summed * summed, axis=0) / BLOCK_SIZE
    inverse_rms = 1.0 / tl.sqrt(mean_square + EPSILON)
    raw_weight = tl.load(raw_weight_ptr + cols).to(tl.float32)
    normalized = summed * inverse_rms * (1.0 + raw_weight)
    tl.store(
        y_ptr + row * BLOCK_SIZE + cols,
        normalized.to(tl.bfloat16),
    )
    tl.store(
        residual_out_ptr + row * BLOCK_SIZE + cols,
        summed.to(tl.bfloat16),
    )


def fused_residual_gemma_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    raw_weight: torch.Tensor,
    y: torch.Tensor,
    residual_out: torch.Tensor,
    rows: int,
) -> None:
    qwen35_fused_residual_gemma_rms_norm_kernel[(rows,)](
        x,
        residual,
        raw_weight,
        y,
        residual_out,
        BLOCK_SIZE=BLOCK_SIZE,
        EPSILON=EPSILON,
        num_warps=NUM_WARPS,
    )


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.contiguous().view(torch.uint16).numpy().tobytes(order="C")
    ).hexdigest()


def compare_output(
    actual_bf16: torch.Tensor,
    reference_float: torch.Tensor,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict:
    actual = actual_bf16.to(torch.float32)
    error = torch.abs(actual - reference_float)
    tolerance = absolute_tolerance + relative_tolerance * torch.abs(
        reference_float
    )
    finite = (
        torch.isfinite(actual)
        & torch.isfinite(reference_float)
        & torch.isfinite(error)
    )
    mismatch = (~finite) | (error > tolerance)
    finite_error = torch.where(finite, error, torch.zeros_like(error))
    reference_l2 = float(torch.linalg.vector_norm(reference_float).item())
    error_l2 = float(torch.linalg.vector_norm(finite_error).item())
    relative_l2 = (
        error_l2 / reference_l2 if reference_l2 != 0.0 else error_l2
    )
    return {
        "all_values_finite": bool(torch.all(finite).item()),
        "nonfinite_count": int(torch.count_nonzero(~finite).item()),
        "mismatch_count": int(torch.count_nonzero(mismatch).item()),
        "max_abs_error": float(torch.max(finite_error).item()),
        "relative_l2_error": relative_l2,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
    }


def guarded_output(
    rows: int, sentinel: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    guard_elements = HIDDEN_SIZE
    storage = torch.full(
        ((rows + 2) * HIDDEN_SIZE,),
        sentinel,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    begin = guard_elements
    end = begin + rows * HIDDEN_SIZE
    output = storage[begin:end].view(rows, HIDDEN_SIZE)
    return output, storage, storage[:begin].clone(), storage[end:].clone()


def guards_unchanged(
    storage: torch.Tensor,
    rows: int,
    prefix_before: torch.Tensor,
    suffix_before: torch.Tensor,
) -> bool:
    begin = HIDDEN_SIZE
    end = begin + rows * HIDDEN_SIZE
    return bool(
        torch.equal(storage[:begin], prefix_before)
        and torch.equal(storage[end:], suffix_before)
    )


def run_case(phase: str, rows: int, seed: int) -> dict:
    torch.manual_seed(seed)
    x = (
        0.5
        * torch.randn(
            (rows, HIDDEN_SIZE), device=DEVICE, dtype=torch.bfloat16
        )
    ).to(torch.bfloat16)
    residual = (
        0.25
        * torch.randn(
            (rows, HIDDEN_SIZE), device=DEVICE, dtype=torch.bfloat16
        )
    ).to(torch.bfloat16)
    raw_weight = (
        0.125
        * torch.randn(
            (HIDDEN_SIZE,), device=DEVICE, dtype=torch.bfloat16
        )
    ).to(torch.bfloat16)
    y, y_storage, y_prefix, y_suffix = guarded_output(rows, 17.0)
    residual_out, residual_storage, residual_prefix, residual_suffix = (
        guarded_output(rows, -19.0)
    )
    x_before = x.clone()
    residual_before = residual.clone()
    raw_weight_before = raw_weight.clone()

    fused_residual_gemma_rms_norm(
        x, residual, raw_weight, y, residual_out, rows
    )

    summed_float = x_before.to(torch.float32) + residual_before.to(
        torch.float32
    )
    variance = torch.mean(
        summed_float * summed_float, dim=-1, keepdim=True
    )
    y_reference = (
        summed_float
        * torch.rsqrt(variance + EPSILON)
        * (1.0 + raw_weight_before.to(torch.float32))
    )
    residual_reference = summed_float.to(torch.bfloat16)
    y_comparison = compare_output(
        y,
        y_reference,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    )
    residual_comparison = compare_output(
        residual_out,
        residual_reference.to(torch.float32),
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    )
    x_unchanged = bool(torch.equal(x, x_before))
    residual_input_unchanged = bool(
        torch.equal(residual, residual_before)
    )
    raw_weight_unchanged = bool(
        torch.equal(raw_weight, raw_weight_before)
    )
    y_guard_unchanged = guards_unchanged(
        y_storage, rows, y_prefix, y_suffix
    )
    residual_guard_unchanged = guards_unchanged(
        residual_storage, rows, residual_prefix, residual_suffix
    )
    outputs_disjoint = (
        y.untyped_storage().data_ptr()
        != residual_out.untyped_storage().data_ptr()
    )
    residual_exact = bool(torch.equal(residual_out, residual_reference))
    residual_actual_bits = residual_out.view(torch.uint16)
    residual_expected_bits = residual_reference.view(torch.uint16)
    residual_bit_mismatch = residual_actual_bits != residual_expected_bits
    residual_bit_mismatch_count = int(
        torch.count_nonzero(residual_bit_mismatch).item()
    )
    residual_first_mismatch = None
    if residual_bit_mismatch_count:
        first_flat = int(
            torch.nonzero(residual_bit_mismatch.view(-1), as_tuple=False)[
                0, 0
            ].item()
        )
        row, column = divmod(first_flat, HIDDEN_SIZE)
        residual_first_mismatch = {
            "row": row,
            "column": column,
            "summed_float": float(summed_float[row, column].item()),
            "actual": float(residual_out[row, column].to(torch.float32).item()),
            "expected": float(
                residual_reference[row, column].to(torch.float32).item()
            ),
            "actual_bits": int(residual_actual_bits[row, column].item()),
            "expected_bits": int(residual_expected_bits[row, column].item()),
        }
    correct = (
        y_comparison["mismatch_count"] == 0
        and y_comparison["nonfinite_count"] == 0
        and y_comparison["relative_l2_error"] < RELATIVE_L2_TOLERANCE
        and residual_comparison["mismatch_count"] == 0
        and residual_comparison["nonfinite_count"] == 0
        and residual_comparison["relative_l2_error"] < RELATIVE_L2_TOLERANCE
        and residual_exact
        and x_unchanged
        and residual_input_unchanged
        and raw_weight_unchanged
        and y_guard_unchanged
        and residual_guard_unchanged
        and outputs_disjoint
    )
    return {
        "phase": phase,
        "rows": rows,
        "seed": seed,
        "input_shape": [rows, HIDDEN_SIZE],
        "residual_shape": [rows, HIDDEN_SIZE],
        "raw_weight_shape": [HIDDEN_SIZE],
        "y_shape": [rows, HIDDEN_SIZE],
        "residual_out_shape": [rows, HIDDEN_SIZE],
        "input_strides": [HIDDEN_SIZE, 1],
        "residual_strides": [HIDDEN_SIZE, 1],
        "raw_weight_strides": [1],
        "output_strides": [HIDDEN_SIZE, 1],
        "program_count": rows,
        "x_unchanged": x_unchanged,
        "residual_input_unchanged": residual_input_unchanged,
        "raw_weight_unchanged": raw_weight_unchanged,
        "outputs_disjoint": outputs_disjoint,
        "y_guard_unchanged": y_guard_unchanged,
        "residual_out_guard_unchanged": residual_guard_unchanged,
        "y": y_comparison,
        "residual_out": residual_comparison,
        "residual_out_bitwise_equal_to_reference": residual_exact,
        "residual_out_bit_mismatch_count": residual_bit_mismatch_count,
        "residual_out_first_mismatch": residual_first_mismatch,
        "y_sha256": tensor_sha256(y),
        "residual_out_sha256": tensor_sha256(residual_out),
        "output_correct": correct,
        "fallback_count": 0,
    }


def main() -> int:
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "gemsim_amd" or target.arch != "gfx950":
        raise RuntimeError(f"unexpected Triton execution target: {target}")
    if DEVICE.type != "cpu":
        raise RuntimeError(
            f"gemsim_amd must expose a CPU staging device, got {DEVICE}"
        )

    launch_results = [
        run_case("decode", 1, 71),
        run_case("prefill", 7, 73),
    ]
    correct = all(result["output_correct"] for result in launch_results)
    payload = {
        "schema": (
            "amdgpu-sim.triton-qwen35-fused-residual-rms-norm.v1"
        ),
        "backend": target.backend,
        "arch": target.arch,
        "kernel": "qwen35_fused_residual_gemma_rms_norm_kernel",
        "dtype": "bfloat16",
        "sum_dtype": "float32",
        "accumulation_dtype": "float32",
        "epsilon": EPSILON,
        "effective_weight": "1.0 + raw_weight.float()",
        "outputs": ["y", "residual_out"],
        "residual_out_semantics": "(x.float() + residual.float()).bfloat16()",
        "input_shape_decode": [1, HIDDEN_SIZE],
        "input_shape_prefill": [7, HIDDEN_SIZE],
        "block_size": BLOCK_SIZE,
        "num_warps": NUM_WARPS,
        "launch_count": len(launch_results),
        "persistent_cache": "qwen35-rms-norm",
        "reuse": True,
        "launch_results": launch_results,
        "output_correct": correct,
        "mismatch_count": sum(
            result["y"]["mismatch_count"]
            + result["residual_out"]["mismatch_count"]
            for result in launch_results
        ),
        "nonfinite_count": sum(
            result["y"]["nonfinite_count"]
            + result["residual_out"]["nonfinite_count"]
            for result in launch_results
        ),
        "max_y_abs_error": max(
            result["y"]["max_abs_error"] for result in launch_results
        ),
        "max_y_relative_l2_error": max(
            result["y"]["relative_l2_error"]
            for result in launch_results
        ),
        "relative_l2_tolerance": RELATIVE_L2_TOLERANCE,
        "fallback_count": 0,
        "cpu_fallback_count": 0,
        "nvidia_fallback_count": 0,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
