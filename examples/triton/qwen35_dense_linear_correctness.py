#!/usr/bin/env python3

import runpy

runpy.run_path(
    __file__.replace("qwen35_dense_linear_correctness.py", "_gemsim_bootstrap.py")
)["bootstrap"](__file__, "qwen35-dense-linear")

import argparse
import json
import math

import torch
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32
NUM_WARPS = 4
ABSOLUTE_TOLERANCE = 0.03125
RELATIVE_TOLERANCE = 0.03


@triton.jit
def dense_linear_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + reduction
        x = tl.load(
            x_ptr + rows[:, None] * K + k[None, :],
            mask=(rows[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + cols[:, None] * K + k[None, :],
            mask=(cols[:, None] < N) & (k[None, :] < K),
            other=0.0,
        )
        accumulator = tl.dot(x, tl.trans(weight), accumulator)

    tl.store(
        output_ptr + rows[:, None] * N + cols[None, :],
        accumulator.to(tl.bfloat16),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def dense_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> None:
    m, k = x.shape
    n, weight_k = weight.shape
    assert k == weight_k
    dense_linear_kernel[(triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))](
        x,
        weight,
        output,
        M=m,
        N=n,
        K=k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
    )


def run_case(name: str, rows: int, input_width: int, output_width: int,
             seed: int, structured: bool = False,
             active_chunks: tuple[int, ...] | None = None) -> dict:
    torch.manual_seed(seed)
    if structured:
        x = (torch.arange(
            1, input_width + 1, device=DEVICE, dtype=torch.float32
        ) / 16.0).view(rows, input_width).to(torch.bfloat16)
        weight = torch.zeros(
            (output_width, input_width), device=DEVICE, dtype=torch.bfloat16
        )
        for chunk, offset in enumerate(range(0, input_width, BLOCK_K)):
            if active_chunks is not None and chunk not in active_chunks:
                continue
            weight[
                torch.arange(output_width),
                offset + 2 * torch.arange(output_width),
            ] = 1
    else:
        x = (0.0625 * torch.randn(
            (rows, input_width), device=DEVICE, dtype=torch.bfloat16
        )).to(torch.bfloat16)
        weight = (0.0625 * torch.randn(
            (output_width, input_width), device=DEVICE, dtype=torch.bfloat16
        )).to(torch.bfloat16)
    output_storage = torch.full(
        ((rows + 1) * output_width,),
        17.0,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    output = output_storage[: rows * output_width].view(rows, output_width)
    guard_before = output_storage[rows * output_width :].clone()
    x_before = x.clone()
    weight_before = weight.clone()

    dense_linear(x, weight, output)

    reference = torch.matmul(x.to(torch.float32), weight.to(torch.float32).T)
    actual = output.to(torch.float32)
    error = torch.abs(actual - reference)
    tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * torch.abs(reference)
    finite = torch.isfinite(actual) & torch.isfinite(reference) & torch.isfinite(error)
    mismatch = (error > tolerance) | ~finite
    mismatch_count = int(torch.count_nonzero(mismatch).item())
    first_mismatches = []
    for row, col in torch.nonzero(mismatch, as_tuple=False)[:8].tolist():
        first_mismatches.append({
            "row": row,
            "col": col,
            "actual": float(actual[row, col].item()),
            "expected": float(reference[row, col].item()),
            "absolute_error": float(error[row, col].item()),
        })
    result = {
        "name": name,
        "rows": rows,
        "input_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "output_shape": list(output.shape),
        "grid": [math.ceil(rows / BLOCK_M), math.ceil(output_width / BLOCK_N)],
        "input_unchanged": bool(torch.equal(x, x_before)),
        "weight_unchanged": bool(torch.equal(weight, weight_before)),
        "guard_unchanged": bool(torch.equal(
            output_storage[rows * output_width :], guard_before
        )),
        "all_values_finite": bool(torch.all(finite).item()),
        "nonfinite_count": int(torch.count_nonzero(~finite).item()),
        "mismatch_count": mismatch_count,
        "first_mismatches": first_mismatches,
        "max_abs_error": float(torch.max(error).item()),
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "output_correct": mismatch_count == 0,
    }
    if structured:
        result["active_chunks"] = (
            list(active_chunks) if active_chunks is not None
            else list(range(input_width // BLOCK_K))
        )
        result["actual_values"] = actual[0].tolist()
        result["expected_values"] = reference[0].tolist()
    return result


def main() -> int:
    default_cases = [
        ("mlp_gate_up_decode", 1, 1024, 7168, 41),
        ("mlp_gate_up_prefill", 7, 1024, 7168, 43),
        ("mlp_down_decode", 1, 3584, 1024, 47),
        ("mlp_down_prefill", 7, 3584, 1024, 53),
    ]
    attention_cases = [
        ("gdn_qkvz_decode", 1, 1024, 8192, 71),
        ("gdn_qkvz_prefill", 7, 1024, 8192, 73),
        ("gdn_ba_decode", 1, 1024, 32, 79),
        ("gdn_ba_prefill", 7, 1024, 32, 83),
        ("gdn_out_decode", 1, 2048, 1024, 89),
        ("gdn_out_prefill", 7, 2048, 1024, 97),
        ("full_attn_qkv_gate_decode", 1, 1024, 5120, 101),
        ("full_attn_qkv_gate_prefill", 7, 1024, 5120, 103),
        ("full_attn_out_decode", 1, 2048, 1024, 107),
        ("full_attn_out_prefill", 7, 2048, 1024, 109),
    ]
    diagnostic_cases = [
        ("mfma_layout_k32", 1, 32, 16, 59, True),
        ("mfma_layout_k64_first_only", 1, 64, 16, 61, True, (0,)),
        ("mfma_layout_k64_second_only", 1, 64, 16, 61, True, (1,)),
        ("mfma_layout_k64_both", 1, 64, 16, 61, True, (0, 1)),
        ("mfma_layout_k128", 1, 128, 16, 67, True),
    ]
    all_cases = [*default_cases, *attention_cases, *diagnostic_cases]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        choices=[case[0] for case in all_cases],
        help="run only the selected case; repeat to select multiple cases",
    )
    selected = parser.parse_args().case
    if selected:
        selected_names = set(selected)
        cases = [case for case in all_cases if case[0] in selected_names]
    else:
        cases = default_cases
    results = [run_case(*case) for case in cases]
    payload = {
        "schema": "amdgpu-sim.triton-qwen35-dense-linear.v1",
        "backend": triton.runtime.driver.active.get_current_target().backend,
        "arch": triton.runtime.driver.active.get_current_target().arch,
        "dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "kernel": "dense_linear_kernel",
        "block": [BLOCK_M, BLOCK_N, BLOCK_K],
        "num_warps": NUM_WARPS,
        "launch_count": len(results),
        "launch_results": results,
        "mismatch_count": sum(result["mismatch_count"] for result in results),
        "nonfinite_count": sum(result["nonfinite_count"] for result in results),
        "fallback_count": 0,
        "output_correct": all(result["output_correct"] for result in results),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["output_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
