#!/usr/bin/env python3

import runpy

runpy.run_path(
    __file__.replace(
        "qwen35_gdn_conv_decode_correctness.py",
        "_gemsim_bootstrap.py",
    )
)["bootstrap"](__file__, "qwen35-gdn-conv-decode")

import hashlib
import json

import torch
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()
CHANNELS = 6144
KERNEL_WIDTH = 4
STATE_WIDTH = KERNEL_WIDTH - 1
STATE_CACHE_LINES = 3
BLOCK_SIZE = 256
NUM_WARPS = 4
ABSOLUTE_TOLERANCE = 0.05
RELATIVE_TOLERANCE = 0.01
RELATIVE_L2_TOLERANCE = 0.005
GUARD_ELEMENTS = 256


@triton.jit
def qwen35_gdn_conv_decode_kernel(
    x_ptr,
    weight_ptr,
    state_cache_ptr,
    state_index_ptr,
    CHANNELS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = channels < CHANNELS
    state_index = tl.load(state_index_ptr + batch).to(tl.int64)
    state_base = (state_index * CHANNELS + channels) * 3
    weight_base = channels * 4
    old0 = tl.load(state_cache_ptr + state_base, mask=mask, other=0.0).to(
        tl.float32
    )
    old1 = tl.load(state_cache_ptr + state_base + 1, mask=mask, other=0.0).to(
        tl.float32
    )
    old2 = tl.load(state_cache_ptr + state_base + 2, mask=mask, other=0.0).to(
        tl.float32
    )
    x_offsets = batch * CHANNELS + channels
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    w0 = tl.load(weight_ptr + weight_base, mask=mask, other=0.0).to(
        tl.float32
    )
    w1 = tl.load(weight_ptr + weight_base + 1, mask=mask, other=0.0).to(
        tl.float32
    )
    w2 = tl.load(weight_ptr + weight_base + 2, mask=mask, other=0.0).to(
        tl.float32
    )
    w3 = tl.load(weight_ptr + weight_base + 3, mask=mask, other=0.0).to(
        tl.float32
    )
    accumulator = old0 * w0 + old1 * w1 + old2 * w2 + x * w3
    output = accumulator * tl.sigmoid(accumulator)
    tl.store(x_ptr + x_offsets, output.to(tl.bfloat16), mask=mask)
    tl.store(
        state_cache_ptr + state_base,
        old1.to(tl.bfloat16),
        mask=mask,
    )
    tl.store(
        state_cache_ptr + state_base + 1,
        old2.to(tl.bfloat16),
        mask=mask,
    )
    tl.store(
        state_cache_ptr + state_base + 2,
        x.to(tl.bfloat16),
        mask=mask,
    )


def gdn_conv_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    state_cache: torch.Tensor,
    state_index: torch.Tensor,
) -> torch.Tensor:
    qwen35_gdn_conv_decode_kernel[
        (x.shape[0], triton.cdiv(CHANNELS, BLOCK_SIZE))
    ](
        x,
        weight,
        state_cache,
        state_index,
        CHANNELS=CHANNELS,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return x


def bf16_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.contiguous().view(torch.uint16).numpy().tobytes(order="C")
    ).hexdigest()


def guarded_tensor(
    shape: tuple[int, ...], sentinel: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    elements = 1
    for extent in shape:
        elements *= extent
    storage = torch.full(
        (GUARD_ELEMENTS + elements + GUARD_ELEMENTS,),
        sentinel,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    begin = GUARD_ELEMENTS
    end = begin + elements
    value = storage[begin:end].view(shape)
    return value, storage, storage[:begin].clone(), storage[end:].clone()


def guards_unchanged(
    storage: torch.Tensor,
    value_elements: int,
    prefix_before: torch.Tensor,
    suffix_before: torch.Tensor,
) -> bool:
    begin = GUARD_ELEMENTS
    end = begin + value_elements
    return bool(
        torch.equal(storage[:begin], prefix_before)
        and torch.equal(storage[end:], suffix_before)
    )


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


def run_transition(
    index: int,
    input_value: torch.Tensor,
    weight: torch.Tensor,
    state_cache: torch.Tensor,
    state_index: torch.Tensor,
    state_storage: torch.Tensor,
    state_prefix: torch.Tensor,
    state_suffix: torch.Tensor,
    expected_state_cache: torch.Tensor,
) -> tuple[dict, torch.Tensor]:
    x, x_storage, x_prefix, x_suffix = guarded_tensor(
        (1, CHANNELS), 23.0 + index
    )
    x.copy_(input_value)
    x_before = x.clone()
    weight_before = weight.clone()
    state_index_before = state_index.clone()
    selected_index = int(state_index[0].item())
    state_before_sha256 = bf16_sha256(state_cache)
    state_before = state_cache.clone()
    expected_before = expected_state_cache.clone()
    expected_selected = expected_before[selected_index]

    output = gdn_conv_decode(x, weight, state_cache, state_index)

    accumulator = (
        expected_selected[:, 0].to(torch.float32)
        * weight_before[:, 0].to(torch.float32)
        + expected_selected[:, 1].to(torch.float32)
        * weight_before[:, 1].to(torch.float32)
        + expected_selected[:, 2].to(torch.float32)
        * weight_before[:, 2].to(torch.float32)
        + x_before[0].to(torch.float32)
        * weight_before[:, 3].to(torch.float32)
    )
    output_reference = accumulator * torch.sigmoid(accumulator)
    next_expected_state_cache = expected_before.clone()
    next_expected_state_cache[selected_index] = torch.stack(
        (
            expected_selected[:, 1],
            expected_selected[:, 2],
            x_before[0],
        ),
        dim=1,
    ).to(torch.bfloat16)
    output_comparison = compare_output(
        output, output_reference.view(1, CHANNELS)
    )
    state_finite = torch.isfinite(state_cache) & torch.isfinite(
        next_expected_state_cache
    )
    state_mismatch = state_cache != next_expected_state_cache
    state_mismatch_count = int(torch.count_nonzero(state_mismatch).item())
    state_nonfinite_count = int(torch.count_nonzero(~state_finite).item())
    output_aliases_input = (
        output.data_ptr() == x.data_ptr()
        and output.untyped_storage().data_ptr()
        == x.untyped_storage().data_ptr()
        and output.storage_offset() == x.storage_offset()
        and output.shape == x.shape
        and output.stride() == x.stride()
    )
    input_overwritten = not torch.equal(x, x_before)
    weight_unchanged = bool(torch.equal(weight, weight_before))
    state_index_unchanged = bool(torch.equal(state_index, state_index_before))
    unselected_state_lines_unchanged = all(
        torch.equal(state_cache[line], state_before[line])
        for line in range(STATE_CACHE_LINES)
        if line != selected_index
    )
    input_output_guard_unchanged = guards_unchanged(
        x_storage,
        CHANNELS,
        x_prefix,
        x_suffix,
    )
    state_guard_unchanged = guards_unchanged(
        state_storage,
        STATE_CACHE_LINES * CHANNELS * STATE_WIDTH,
        state_prefix,
        state_suffix,
    )
    correct = (
        output_comparison["mismatch_count"] == 0
        and output_comparison["nonfinite_count"] == 0
        and output_comparison["relative_l2_error"] < RELATIVE_L2_TOLERANCE
        and state_mismatch_count == 0
        and state_nonfinite_count == 0
        and output_aliases_input
        and input_overwritten
        and weight_unchanged
        and state_index_unchanged
        and unselected_state_lines_unchanged
        and input_output_guard_unchanged
        and state_guard_unchanged
    )
    result = {
        "transition": index,
        "input_shape": [1, CHANNELS],
        "output_shape": [1, CHANNELS],
        "state_cache_shape": [STATE_CACHE_LINES, CHANNELS, STATE_WIDTH],
        "selected_state_index": selected_index,
        "input_strides": [CHANNELS, 1],
        "weight_strides": [KERNEL_WIDTH, 1],
        "state_cache_strides": [CHANNELS * STATE_WIDTH, STATE_WIDTH, 1],
        "grid": [1, triton.cdiv(CHANNELS, BLOCK_SIZE)],
        "output_aliases_input": output_aliases_input,
        "input_overwritten": input_overwritten,
        "weight_unchanged": weight_unchanged,
        "state_index_unchanged": state_index_unchanged,
        "unselected_state_lines_unchanged": unselected_state_lines_unchanged,
        "input_output_guard_unchanged": input_output_guard_unchanged,
        "state_guard_unchanged": state_guard_unchanged,
        "output": output_comparison,
        "state_mismatch_count": state_mismatch_count,
        "state_nonfinite_count": state_nonfinite_count,
        "state_all_values_finite": state_nonfinite_count == 0,
        "state_before_sha256": state_before_sha256,
        "state_after_sha256": bf16_sha256(state_cache),
        "expected_state_after_sha256": bf16_sha256(
            next_expected_state_cache
        ),
        "input_before_sha256": bf16_sha256(x_before),
        "output_sha256": bf16_sha256(output),
        "output_correct": correct,
        "fallback_count": 0,
    }
    return result, next_expected_state_cache


def main() -> int:
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "gemsim_amd" or target.arch != "gfx950":
        raise RuntimeError(f"unexpected Triton execution target: {target}")
    if DEVICE.type != "cpu":
        raise RuntimeError(
            f"gemsim_amd must expose a CPU staging device, got {DEVICE}"
        )

    torch.manual_seed(89)
    weight = (
        0.25
        * torch.randn(
            (CHANNELS, KERNEL_WIDTH),
            device=DEVICE,
            dtype=torch.bfloat16,
        )
    ).to(torch.bfloat16)
    state_cache, state_storage, state_prefix, state_suffix = guarded_tensor(
        (STATE_CACHE_LINES, CHANNELS, STATE_WIDTH), -29.0
    )
    state_cache.copy_(
        (
            0.25
            * torch.randn(
                (STATE_CACHE_LINES, CHANNELS, STATE_WIDTH),
                device=DEVICE,
                dtype=torch.bfloat16,
            )
        ).to(torch.bfloat16)
    )
    state_index = torch.tensor([1], device=DEVICE, dtype=torch.int32)
    inputs = [
        (
            0.25
            * torch.randn(
                (1, CHANNELS), device=DEVICE, dtype=torch.bfloat16
            )
        ).to(torch.bfloat16)
        for _ in range(2)
    ]
    weight_before = weight.clone()
    state_index_before = state_index.clone()
    initial_state_cache = state_cache.clone()
    initial_state_nonzero_count = int(
        torch.count_nonzero(initial_state_cache).item()
    )
    inputs_distinct = not torch.equal(inputs[0], inputs[1])
    expected_state_cache = initial_state_cache.clone()
    transitions = []
    for index, x in enumerate(inputs, start=1):
        result, expected_state_cache = run_transition(
            index,
            x,
            weight,
            state_cache,
            state_index,
            state_storage,
            state_prefix,
            state_suffix,
            expected_state_cache,
        )
        transitions.append(result)

    state_chain_preserved = (
        transitions[1]["state_before_sha256"]
        == transitions[0]["state_after_sha256"]
    )
    weight_unchanged_after_sequence = bool(torch.equal(weight, weight_before))
    state_index_unchanged_after_sequence = bool(
        torch.equal(state_index, state_index_before)
    )
    unselected_state_lines_unchanged_after_sequence = bool(
        torch.equal(state_cache[0], initial_state_cache[0])
        and torch.equal(state_cache[2], initial_state_cache[2])
    )
    correct = (
        initial_state_nonzero_count > 0
        and inputs_distinct
        and state_chain_preserved
        and weight_unchanged_after_sequence
        and state_index_unchanged_after_sequence
        and unselected_state_lines_unchanged_after_sequence
        and all(result["output_correct"] for result in transitions)
    )
    payload = {
        "schema": "amdgpu-sim.triton-qwen35-gdn-conv-decode.v1",
        "backend": target.backend,
        "arch": target.arch,
        "kernel": "qwen35_gdn_conv_decode_kernel",
        "dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "activation": "silu",
        "channels": CHANNELS,
        "kernel_width": KERNEL_WIDTH,
        "state_cache_shape": [STATE_CACHE_LINES, CHANNELS, STATE_WIDTH],
        "selected_state_index": int(state_index.item()),
        "state_update": "[old1, old2, x]",
        "input_output_alias": "in_place",
        "formula": "silu(old0*w0 + old1*w1 + old2*w2 + x*w3)",
        "block_size": BLOCK_SIZE,
        "num_warps": NUM_WARPS,
        "transition_count": len(transitions),
        "initial_state_nonzero_count": initial_state_nonzero_count,
        "initial_state_sha256": bf16_sha256(initial_state_cache),
        "inputs_distinct": inputs_distinct,
        "state_chain_preserved": state_chain_preserved,
        "weight_unchanged_after_sequence": weight_unchanged_after_sequence,
        "state_index_unchanged_after_sequence": (
            state_index_unchanged_after_sequence
        ),
        "unselected_state_lines_unchanged_after_sequence": (
            unselected_state_lines_unchanged_after_sequence
        ),
        "persistent_cache": "qwen35-gdn-conv-decode",
        "transitions": transitions,
        "mismatch_count": sum(
            result["output"]["mismatch_count"]
            + result["state_mismatch_count"]
            for result in transitions
        ),
        "nonfinite_count": sum(
            result["output"]["nonfinite_count"]
            + result["state_nonfinite_count"]
            for result in transitions
        ),
        "max_abs_error": max(
            result["output"]["max_abs_error"] for result in transitions
        ),
        "max_relative_l2_error": max(
            result["output"]["relative_l2_error"]
            for result in transitions
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
