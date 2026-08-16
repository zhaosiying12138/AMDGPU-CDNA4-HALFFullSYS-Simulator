#!/usr/bin/env python3

import runpy

runpy.run_path(
    __file__.replace(
        "qwen35_gdn_recurrent_decode_correctness.py",
        "_gemsim_bootstrap.py",
    )
)["bootstrap"](__file__, "qwen35-gdn-recurrent")

import hashlib
import json
import math

import torch
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()
NUM_HEADS = 16
KEY_DIM = 128
VALUE_DIM = 128
MIXED_QKV_DIM = 3 * NUM_HEADS * KEY_DIM
VALUE_BLOCK = 32
SCALE = KEY_DIM**-0.5
EPSILON = 1.0e-6
SOFTPLUS_THRESHOLD = 20.0
OUTPUT_ATOL = 0.015625
OUTPUT_RTOL = 0.02
STATE_ATOL = 2.0e-5
STATE_RTOL = 2.0e-4


@triton.jit
def qwen35_gdn_recurrent_decode_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    a_log_ptr,
    dt_bias_ptr,
    state_ptr,
    output_ptr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    VALUE_BLOCK: tl.constexpr,
    SCALE: tl.constexpr,
    EPSILON: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    value_block = tl.program_id(0)
    head = tl.program_id(1)
    key_offsets = tl.arange(0, KEY_DIM)
    value_offsets = value_block * VALUE_BLOCK + tl.arange(0, VALUE_BLOCK)
    value_mask = value_offsets < VALUE_DIM
    state_mask = value_mask[:, None]

    state_offsets = (
        head * VALUE_DIM * KEY_DIM
        + value_offsets[:, None] * KEY_DIM
        + key_offsets[None, :]
    )
    state = tl.load(
        state_ptr + state_offsets,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)

    query_offset = head * KEY_DIM + key_offsets
    key_offset = NUM_HEADS * KEY_DIM + head * KEY_DIM + key_offsets
    value_offset = 2 * NUM_HEADS * KEY_DIM + head * VALUE_DIM + value_offsets
    query = tl.load(mixed_qkv_ptr + query_offset).to(tl.float32)
    key = tl.load(mixed_qkv_ptr + key_offset).to(tl.float32)
    value = tl.load(
        mixed_qkv_ptr + value_offset,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)

    query *= tl.rsqrt(tl.sum(query * query, axis=0) + EPSILON)
    key *= tl.rsqrt(tl.sum(key * key, axis=0) + EPSILON)
    query *= SCALE

    a = tl.load(a_ptr + head).to(tl.float32)
    b = tl.load(b_ptr + head).to(tl.float32)
    a_log = tl.load(a_log_ptr + head).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + head).to(tl.float32)
    softplus_input = a + dt_bias
    softplus = tl.where(
        softplus_input <= SOFTPLUS_THRESHOLD,
        tl.log(1.0 + tl.exp(softplus_input)),
        softplus_input,
    )
    decay_log = -tl.exp(a_log) * softplus
    state *= tl.exp(decay_log)

    prediction = tl.sum(state * key[None, :], axis=1)
    beta = tl.sigmoid(b).to(b_ptr.dtype.element_ty).to(tl.float32)
    delta = (value - prediction) * beta
    state += delta[:, None] * key[None, :]
    output = tl.sum(state * query[None, :], axis=1)

    tl.store(
        output_ptr + head * VALUE_DIM + value_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=value_mask,
    )
    tl.store(state_ptr + state_offsets, state, mask=state_mask)


def gdn_recurrent_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    output: torch.Tensor,
) -> None:
    qwen35_gdn_recurrent_decode_kernel[
        (triton.cdiv(VALUE_DIM, VALUE_BLOCK), NUM_HEADS)
    ](
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
        output,
        KEY_DIM=KEY_DIM,
        VALUE_DIM=VALUE_DIM,
        NUM_HEADS=NUM_HEADS,
        VALUE_BLOCK=VALUE_BLOCK,
        SCALE=SCALE,
        EPSILON=EPSILON,
        SOFTPLUS_THRESHOLD=SOFTPLUS_THRESHOLD,
        num_warps=1,
        num_stages=3,
    )


def tensor_sha256(value: torch.Tensor) -> str:
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return hashlib.sha256(
        value.contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def compare(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict:
    actual_float = actual.to(torch.float32)
    expected_float = expected.to(torch.float32)
    error = torch.abs(actual_float - expected_float)
    tolerance = atol + rtol * torch.abs(expected_float)
    finite = (
        torch.isfinite(actual_float)
        & torch.isfinite(expected_float)
        & torch.isfinite(error)
    )
    mismatch = (~finite) | (error > tolerance)
    finite_error = torch.where(finite, error, torch.zeros_like(error))
    expected_norm = float(torch.linalg.vector_norm(expected_float).item())
    error_norm = float(torch.linalg.vector_norm(finite_error).item())
    return {
        "all_values_finite": bool(torch.all(finite).item()),
        "nonfinite_count": int(torch.count_nonzero(~finite).item()),
        "mismatch_count": int(torch.count_nonzero(mismatch).item()),
        "max_abs_error": float(torch.max(finite_error).item()),
        "relative_l2_error": (
            error_norm / expected_norm if expected_norm != 0.0 else error_norm
        ),
        "atol": atol,
        "rtol": rtol,
    }


def guarded_tensor(
    elements: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    sentinel: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    guard_elements = 257
    storage = torch.full(
        (elements + 2 * guard_elements,),
        sentinel,
        dtype=dtype,
        device=DEVICE,
    )
    value = storage[guard_elements : guard_elements + elements].view(shape)
    return (
        value,
        storage,
        storage[:guard_elements].clone(),
        storage[guard_elements + elements :].clone(),
    )


def guards_unchanged(
    storage: torch.Tensor,
    elements: int,
    prefix: torch.Tensor,
    suffix: torch.Tensor,
) -> bool:
    guard_elements = prefix.numel()
    return bool(
        torch.equal(storage[:guard_elements], prefix)
        and torch.equal(storage[guard_elements + elements :], suffix)
    )


def reference_transition(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mixed = mixed_qkv.to(torch.float32).view(-1)
    query = mixed[: NUM_HEADS * KEY_DIM].view(NUM_HEADS, KEY_DIM)
    key = mixed[
        NUM_HEADS * KEY_DIM : 2 * NUM_HEADS * KEY_DIM
    ].view(NUM_HEADS, KEY_DIM)
    value = mixed[2 * NUM_HEADS * KEY_DIM :].view(
        NUM_HEADS, VALUE_DIM
    )
    query = query * torch.rsqrt(
        torch.sum(query * query, dim=-1, keepdim=True) + EPSILON
    )
    key = key * torch.rsqrt(
        torch.sum(key * key, dim=-1, keepdim=True) + EPSILON
    )
    query *= SCALE
    a_float = a.to(torch.float32).view(NUM_HEADS)
    b_float = b.to(torch.float32).view(NUM_HEADS)
    softplus_input = a_float + dt_bias.to(torch.float32)
    softplus = torch.where(
        softplus_input <= SOFTPLUS_THRESHOLD,
        torch.log1p(torch.exp(softplus_input)),
        softplus_input,
    )
    decay = torch.exp(-torch.exp(a_log) * softplus)
    beta = torch.sigmoid(b_float).to(torch.bfloat16).to(
        torch.float32
    )
    next_state = state.to(torch.float32) * decay.view(NUM_HEADS, 1, 1)
    prediction = torch.einsum("hvk,hk->hv", next_state, key)
    delta = (value - prediction) * beta.view(NUM_HEADS, 1)
    next_state = next_state + delta[:, :, None] * key[:, None, :]
    output = torch.einsum("hvk,hk->hv", next_state, query)
    return output.to(torch.bfloat16), next_state


def run_transition(
    transition: int,
    seed: int,
    state: torch.Tensor,
    state_storage: torch.Tensor,
    state_prefix: torch.Tensor,
    state_suffix: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> dict:
    torch.manual_seed(seed)
    mixed_qkv = (
        0.25
        * torch.randn(
            (1, MIXED_QKV_DIM), dtype=torch.bfloat16, device=DEVICE
        )
    ).to(torch.bfloat16)
    a = (
        0.25
        * torch.randn((1, NUM_HEADS), dtype=torch.bfloat16, device=DEVICE)
    ).to(torch.bfloat16)
    b = (
        0.25
        * torch.randn((1, NUM_HEADS), dtype=torch.bfloat16, device=DEVICE)
    ).to(torch.bfloat16)
    output, output_storage, output_prefix, output_suffix = guarded_tensor(
        NUM_HEADS * VALUE_DIM,
        (1, NUM_HEADS, VALUE_DIM),
        torch.bfloat16,
        -23.0,
    )
    mixed_before = mixed_qkv.clone()
    a_before = a.clone()
    b_before = b.clone()
    a_log_before = a_log.clone()
    dt_bias_before = dt_bias.clone()
    state_before = state.clone()
    expected_output, expected_state = reference_transition(
        mixed_before,
        a_before,
        b_before,
        a_log_before,
        dt_bias_before,
        state_before,
    )

    gdn_recurrent_decode(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
        output,
    )

    output_comparison = compare(
        output, expected_output, OUTPUT_ATOL, OUTPUT_RTOL
    )
    state_comparison = compare(
        state, expected_state, STATE_ATOL, STATE_RTOL
    )
    output_guard_unchanged = guards_unchanged(
        output_storage,
        NUM_HEADS * VALUE_DIM,
        output_prefix,
        output_suffix,
    )
    state_guard_unchanged = guards_unchanged(
        state_storage,
        NUM_HEADS * VALUE_DIM * KEY_DIM,
        state_prefix,
        state_suffix,
    )
    inputs_unchanged = bool(
        torch.equal(mixed_qkv, mixed_before)
        and torch.equal(a, a_before)
        and torch.equal(b, b_before)
        and torch.equal(a_log, a_log_before)
        and torch.equal(dt_bias, dt_bias_before)
    )
    state_changed = not torch.equal(state, state_before)
    correct = (
        output_comparison["mismatch_count"] == 0
        and output_comparison["nonfinite_count"] == 0
        and state_comparison["mismatch_count"] == 0
        and state_comparison["nonfinite_count"] == 0
        and output_guard_unchanged
        and state_guard_unchanged
        and inputs_unchanged
        and state_changed
    )
    return {
        "transition": transition,
        "seed": seed,
        "output": output_comparison,
        "state": state_comparison,
        "output_guard_unchanged": output_guard_unchanged,
        "state_guard_unchanged": state_guard_unchanged,
        "inputs_unchanged": inputs_unchanged,
        "state_changed": state_changed,
        "output_sha256": tensor_sha256(output),
        "state_before_sha256": tensor_sha256(state_before),
        "state_after_sha256": tensor_sha256(state),
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

    torch.manual_seed(109)
    state, state_storage, state_prefix, state_suffix = guarded_tensor(
        NUM_HEADS * VALUE_DIM * KEY_DIM,
        (NUM_HEADS, VALUE_DIM, KEY_DIM),
        torch.float32,
        29.0,
    )
    state.copy_(
        0.005
        * torch.randn(
            (NUM_HEADS, VALUE_DIM, KEY_DIM),
            dtype=torch.float32,
            device=DEVICE,
        )
    )
    a_log = (
        -0.25
        + 0.05
        * torch.randn((NUM_HEADS,), dtype=torch.float32, device=DEVICE)
    )
    dt_bias = (
        0.1
        * torch.randn((NUM_HEADS,), dtype=torch.bfloat16, device=DEVICE)
    ).to(torch.bfloat16)
    initial_state_sha256 = tensor_sha256(state)
    results = [
        run_transition(
            1,
            113,
            state,
            state_storage,
            state_prefix,
            state_suffix,
            a_log,
            dt_bias,
        ),
        run_transition(
            2,
            127,
            state,
            state_storage,
            state_prefix,
            state_suffix,
            a_log,
            dt_bias,
        ),
    ]
    correct = all(result["output_correct"] for result in results)
    payload = {
        "schema": "amdgpu-sim.triton-qwen35-gdn-recurrent-decode.v1",
        "backend": target.backend,
        "arch": target.arch,
        "kernel": "qwen35_gdn_recurrent_decode_kernel",
        "dtype": "bfloat16",
        "state_dtype": "float32",
        "accumulation_dtype": "float32",
        "num_heads": NUM_HEADS,
        "key_dim": KEY_DIM,
        "value_dim": VALUE_DIM,
        "mixed_qkv_shape": [1, MIXED_QKV_DIM],
        "state_shape": [NUM_HEADS, VALUE_DIM, KEY_DIM],
        "output_shape": [1, NUM_HEADS, VALUE_DIM],
        "transition_count": len(results),
        "program_count_per_transition": (
            triton.cdiv(VALUE_DIM, VALUE_BLOCK) * NUM_HEADS
        ),
        "initial_state_nonzero": True,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": tensor_sha256(state),
        "persistent_cache": "qwen35-gdn-recurrent",
        "transition_results": results,
        "output_correct": correct,
        "mismatch_count": sum(
            result["output"]["mismatch_count"]
            + result["state"]["mismatch_count"]
            for result in results
        ),
        "nonfinite_count": sum(
            result["output"]["nonfinite_count"]
            + result["state"]["nonfinite_count"]
            for result in results
        ),
        "fallback_count": 0,
        "cpu_fallback_count": 0,
        "nvidia_fallback_count": 0,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
