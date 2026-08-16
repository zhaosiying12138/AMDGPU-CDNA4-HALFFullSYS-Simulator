#!/usr/bin/env python3

import runpy

runpy.run_path(
    __file__.replace(
        "qwen35_gdn_output_norm_gate_correctness.py",
        "_gemsim_bootstrap.py",
    )
)["bootstrap"](__file__, "qwen35-gdn-output-norm-gate")

import hashlib
import json

import torch
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()
NUM_HEADS = 16
HEAD_DIM = 128
BLOCK_SIZE = 128
NUM_WARPS = 4
EPSILON = 1.0e-6
ABSOLUTE_TOLERANCE = 0.01
RELATIVE_TOLERANCE = 0.01
RELATIVE_L2_TOLERANCE = 0.005
GUARD_ELEMENTS = HEAD_DIM


@triton.jit
def qwen35_gdn_output_norm_gate_kernel(
    x_ptr,
    z_ptr,
    weight_ptr,
    output_ptr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HEAD_DIM
    offsets = row * HEAD_DIM + cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    variance = tl.sum(x * x, axis=0) / HEAD_DIM
    normalized = x * (1.0 / tl.sqrt(variance + EPSILON))
    weighted = normalized * weight
    gated = weighted * (z * tl.sigmoid(z))
    tl.store(
        output_ptr + offsets,
        gated.to(tl.bfloat16),
        mask=mask,
    )


def gdn_output_norm_gate(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    rows: int,
) -> None:
    qwen35_gdn_output_norm_gate_kernel[(rows * NUM_HEADS,)](
        x,
        z,
        weight,
        output,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SIZE=BLOCK_SIZE,
        EPSILON=EPSILON,
        num_warps=NUM_WARPS,
    )


def bf16_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.contiguous().view(torch.uint16).numpy().tobytes(order="C")
    ).hexdigest()


def compare_output(
    actual_bf16: torch.Tensor, reference_float: torch.Tensor
) -> dict:
    actual = actual_bf16.to(torch.float32)
    error = torch.abs(actual - reference_float)
    tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * torch.abs(
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
        "mismatch_count": int(torch.count_nonzero(mismatch).item()),
        "nonfinite_count": int(torch.count_nonzero(~finite).item()),
        "all_values_finite": bool(torch.all(finite).item()),
        "max_abs_error": float(torch.max(finite_error).item()),
        "relative_l2_error": relative_l2,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
    }


def run_case(phase: str, rows: int, seed: int) -> dict:
    torch.manual_seed(seed)
    x = torch.randn(
        (rows, NUM_HEADS, HEAD_DIM),
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    z = torch.randn(
        (rows, NUM_HEADS, HEAD_DIM),
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    weight = 1.0 + 0.125 * torch.randn(
        (HEAD_DIM,), device=DEVICE, dtype=torch.float32
    )
    output_elements = rows * NUM_HEADS * HEAD_DIM
    output_storage = torch.full(
        (GUARD_ELEMENTS + output_elements + GUARD_ELEMENTS,),
        31.0,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    begin = GUARD_ELEMENTS
    end = begin + output_elements
    output = output_storage[begin:end].view(rows, NUM_HEADS, HEAD_DIM)
    prefix_before = output_storage[:begin].clone()
    suffix_before = output_storage[end:].clone()
    x_before = x.clone()
    z_before = z.clone()
    weight_before = weight.clone()

    gdn_output_norm_gate(x, z, weight, output, rows)

    x_float = x_before.to(torch.float32)
    z_float = z_before.to(torch.float32)
    variance = torch.mean(x_float * x_float, dim=-1, keepdim=True)
    normalized = x_float * torch.rsqrt(variance + EPSILON)
    weighted = normalized * weight_before
    reference = weighted * (z_float * torch.sigmoid(z_float))
    comparison = compare_output(output, reference)
    x_unchanged = bool(torch.equal(x, x_before))
    z_unchanged = bool(torch.equal(z, z_before))
    weight_unchanged = bool(torch.equal(weight, weight_before))
    output_guard_unchanged = bool(
        torch.equal(output_storage[:begin], prefix_before)
        and torch.equal(output_storage[end:], suffix_before)
    )
    correct = (
        comparison["mismatch_count"] == 0
        and comparison["nonfinite_count"] == 0
        and comparison["relative_l2_error"] < RELATIVE_L2_TOLERANCE
        and x_unchanged
        and z_unchanged
        and weight_unchanged
        and output_guard_unchanged
    )
    return {
        "phase": phase,
        "rows": rows,
        "seed": seed,
        "x_shape": [rows, NUM_HEADS, HEAD_DIM],
        "z_shape": [rows, NUM_HEADS, HEAD_DIM],
        "weight_shape": [HEAD_DIM],
        "output_shape": [rows, NUM_HEADS, HEAD_DIM],
        "x_strides": [NUM_HEADS * HEAD_DIM, HEAD_DIM, 1],
        "z_strides": [NUM_HEADS * HEAD_DIM, HEAD_DIM, 1],
        "weight_strides": [1],
        "output_strides": [NUM_HEADS * HEAD_DIM, HEAD_DIM, 1],
        "program_count": rows * NUM_HEADS,
        "x_unchanged": x_unchanged,
        "z_unchanged": z_unchanged,
        "weight_unchanged": weight_unchanged,
        "output_guard_unchanged": output_guard_unchanged,
        "output": comparison,
        "output_sha256": bf16_sha256(output),
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

    results = [
        run_case("decode", 1, 97),
        run_case("prefill", 7, 101),
    ]
    correct = all(result["output_correct"] for result in results)
    payload = {
        "schema": "amdgpu-sim.triton-qwen35-gdn-output-norm-gate.v1",
        "backend": target.backend,
        "arch": target.arch,
        "kernel": "qwen35_gdn_output_norm_gate_kernel",
        "dtype": "bfloat16",
        "weight_dtype": "float32",
        "accumulation_dtype": "float32",
        "epsilon": EPSILON,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "norm_before_gate": True,
        "activation": "silu",
        "effective_weight": "weight.float()",
        "formula": "rms_norm(x.float()) * weight * silu(z.float())",
        "input_shape_decode": [1, NUM_HEADS, HEAD_DIM],
        "input_shape_prefill": [7, NUM_HEADS, HEAD_DIM],
        "block_size": BLOCK_SIZE,
        "num_warps": NUM_WARPS,
        "launch_count": len(results),
        "persistent_cache": "qwen35-gdn-output-norm-gate",
        "results": results,
        "mismatch_count": sum(
            result["output"]["mismatch_count"] for result in results
        ),
        "nonfinite_count": sum(
            result["output"]["nonfinite_count"] for result in results
        ),
        "max_abs_error": max(
            result["output"]["max_abs_error"] for result in results
        ),
        "max_relative_l2_error": max(
            result["output"]["relative_l2_error"] for result in results
        ),
        "relative_l2_tolerance": RELATIVE_L2_TOLERANCE,
        "fallback_count": 0,
        "cpu_fallback_count": 0,
        "nvidia_fallback_count": 0,
        "output_correct": correct,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
