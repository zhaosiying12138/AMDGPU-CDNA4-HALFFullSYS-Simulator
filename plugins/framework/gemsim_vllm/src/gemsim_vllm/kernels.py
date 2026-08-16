"""Project-owned Triton kernels used by the formal framework plugin."""

from __future__ import annotations

import triton
import triton.language as tl


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


@triton.jit
def embedding_kernel(
    token_ids_ptr,
    weight_ptr,
    output_ptr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_position = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = columns < HIDDEN
    token_id = tl.load(token_ids_ptr + token_position).to(tl.int64)
    values = tl.load(
        weight_ptr + token_id * HIDDEN + columns,
        mask=mask,
        other=0.0,
    )
    tl.store(
        output_ptr + token_position * HIDDEN + columns,
        values,
        mask=mask,
    )


@triton.jit
def rotary_embedding_kernel(
    positions_ptr,
    query_ptr,
    key_ptr,
    cache_ptr,
    query_out_ptr,
    key_out_ptr,
    query_stride_t,
    key_stride_t,
    query_out_stride_t,
    key_out_stride_t,
    cache_stride_p,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_key = head >= NUM_Q_HEADS
    local_head = tl.where(is_key, head - NUM_Q_HEADS, head)
    if is_key:
        input_base = key_ptr + token * key_stride_t + local_head * HEAD_DIM
        output_base = key_out_ptr + token * key_out_stride_t + local_head * HEAD_DIM
    else:
        input_base = query_ptr + token * query_stride_t + local_head * HEAD_DIM
        output_base = (
            query_out_ptr + token * query_out_stride_t + local_head * HEAD_DIM
        )

    tail = tl.arange(0, HEAD_DIM)
    values = tl.load(input_base + tail)
    tl.store(output_base + tail, values, mask=tail >= ROTARY_DIM)

    offsets = tl.arange(0, HALF_ROTARY)
    first = tl.load(input_base + offsets).to(tl.float32)
    second = tl.load(input_base + HALF_ROTARY + offsets).to(tl.float32)
    position = tl.load(positions_ptr + token).to(tl.int64)
    cache_base = position * cache_stride_p
    cosine = tl.load(cache_ptr + cache_base + offsets).to(tl.float32)
    sine = tl.load(cache_ptr + cache_base + HALF_ROTARY + offsets).to(tl.float32)
    tl.store(output_base + offsets, first * cosine - second * sine)
    tl.store(
        output_base + HALF_ROTARY + offsets,
        second * cosine + first * sine,
    )


@triton.jit
def gemma_rms_norm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    EPSILON: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < HIDDEN
    x = tl.load(x_ptr + row * HIDDEN + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / HIDDEN
    output = x * tl.rsqrt(variance + EPSILON) * (1.0 + weight)
    tl.store(
        output_ptr + row * HIDDEN + columns,
        output.to(tl.bfloat16),
        mask=mask,
    )


@triton.jit
def fused_add_gemma_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    residual_out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    EPSILON: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < HIDDEN
    x = tl.load(x_ptr + row * HIDDEN + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    residual = tl.load(
        residual_ptr + row * HIDDEN + columns, mask=mask, other=0.0
    ).to(tl.float32)
    summed = x + residual
    variance = tl.sum(summed * summed, axis=0) / HIDDEN
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    output = summed * tl.rsqrt(variance + EPSILON) * (1.0 + weight)
    tl.store(
        output_ptr + row * HIDDEN + columns,
        output.to(tl.bfloat16),
        mask=mask,
    )
    tl.store(
        residual_out_ptr + row * HIDDEN + columns,
        summed.to(tl.bfloat16),
        mask=mask,
    )


@triton.jit
def silu_and_mul_kernel(
    x_ptr,
    output_ptr,
    HALF_WIDTH: tl.constexpr,
    FULL_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    columns = block * BLOCK + tl.arange(0, BLOCK)
    mask = columns < HALF_WIDTH
    base = row * FULL_WIDTH
    gate = tl.load(x_ptr + base + columns, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(
        x_ptr + base + HALF_WIDTH + columns, mask=mask, other=0.0
    ).to(tl.float32)
    output = gate * tl.sigmoid(gate) * up
    tl.store(
        output_ptr + row * HALF_WIDTH + columns,
        output.to(tl.bfloat16),
        mask=mask,
    )


@triton.jit
def sigmoid_output_gate_kernel(
    attention_ptr,
    gate_ptr,
    output_ptr,
    ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS
    attention = tl.load(attention_ptr + offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    # The pinned vLLM expression materializes sigmoid(gate_bf16) as BF16
    # before the multiply.  Preserve that boundary across the target backend.
    sigmoid_gate = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    output = attention * sigmoid_gate
    tl.store(output_ptr + offsets, output.to(tl.bfloat16), mask=mask)


@triton.jit
def rms_norm_gated_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    output_ptr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    EPSILON: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < HEAD_DIM
    offsets = row * HEAD_DIM + columns
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    variance = tl.sum(x * x, axis=0) / HEAD_DIM
    normalized = x * tl.rsqrt(variance + EPSILON) * weight
    output = normalized * (gate * tl.sigmoid(gate))
    tl.store(output_ptr + offsets, output.to(tl.bfloat16), mask=mask)


@triton.jit
def gdn_conv_decode_kernel(
    x_ptr,
    weight_ptr,
    state_ptr,
    state_indices_ptr,
    output_ptr,
    state_stride_line,
    state_stride_channel,
    state_stride_time,
    TOKENS: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = channels < CHANNELS
    state_index = tl.load(state_indices_ptr).to(tl.int64)
    state_base = (
        state_index * state_stride_line + channels * state_stride_channel
    )
    old0 = tl.load(state_ptr + state_base, mask=mask, other=0.0).to(tl.float32)
    old1 = tl.load(
        state_ptr + state_base + state_stride_time, mask=mask, other=0.0
    ).to(tl.float32)
    old2 = tl.load(
        state_ptr + state_base + 2 * state_stride_time,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight_base = channels * 4
    w0 = tl.load(weight_ptr + weight_base, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight_ptr + weight_base + 1, mask=mask, other=0.0).to(
        tl.float32
    )
    w2 = tl.load(weight_ptr + weight_base + 2, mask=mask, other=0.0).to(
        tl.float32
    )
    w3 = tl.load(weight_ptr + weight_base + 3, mask=mask, other=0.0).to(
        tl.float32
    )
    for token in range(0, TOKENS):
        x_offsets = token * CHANNELS + channels
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        accumulator = old0 * w0 + old1 * w1 + old2 * w2 + x * w3
        output = accumulator * tl.sigmoid(accumulator)
        tl.store(output_ptr + x_offsets, output.to(tl.bfloat16), mask=mask)
        old0 = old1
        old1 = old2
        old2 = x
    tl.store(
        state_ptr + state_base,
        old0.to(tl.bfloat16),
        mask=mask,
    )
    tl.store(
        state_ptr + state_base + state_stride_time,
        old1.to(tl.bfloat16),
        mask=mask,
    )
    tl.store(
        state_ptr + state_base + 2 * state_stride_time,
        old2.to(tl.bfloat16),
        mask=mask,
    )


@triton.jit
def gdn_recurrent_decode_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    a_log_ptr,
    dt_bias_ptr,
    state_ptr,
    state_indices_ptr,
    output_ptr,
    TOKENS: tl.constexpr,
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
    state_index = tl.load(state_indices_ptr).to(tl.int64)
    state_offsets = (
        ((state_index * NUM_HEADS + head) * VALUE_DIM + value_offsets[:, None])
        * KEY_DIM
        + key_offsets[None, :]
    )
    state = tl.load(
        state_ptr + state_offsets,
        mask=value_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    a_log = tl.load(a_log_ptr + head).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + head).to(tl.float32)
    for token in range(0, TOKENS):
        token_qkv = token * (2 * NUM_HEADS * KEY_DIM + NUM_HEADS * VALUE_DIM)
        query_offset = token_qkv + head * KEY_DIM + key_offsets
        key_offset = token_qkv + NUM_HEADS * KEY_DIM + head * KEY_DIM + key_offsets
        value_offset = (
            token_qkv
            + 2 * NUM_HEADS * KEY_DIM
            + head * VALUE_DIM
            + value_offsets
        )
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

        a = tl.load(a_ptr + token * NUM_HEADS + head).to(tl.float32)
        b = tl.load(b_ptr + token * NUM_HEADS + head).to(tl.float32)
        softplus_input = a + dt_bias
        softplus = tl.where(
            softplus_input <= SOFTPLUS_THRESHOLD,
            tl.log(1.0 + tl.exp(softplus_input)),
            softplus_input,
        )
        state *= tl.exp(-tl.exp(a_log) * softplus)
        prediction = tl.sum(state * key[None, :], axis=1)
        beta = tl.sigmoid(b).to(b_ptr.dtype.element_ty).to(tl.float32)
        delta = (value - prediction) * beta
        state += delta[:, None] * key[None, :]
        output = tl.sum(state * query[None, :], axis=1)
        tl.store(
            output_ptr
            + token * NUM_HEADS * VALUE_DIM
            + head * VALUE_DIM
            + value_offsets,
            output.to(tl.bfloat16),
            mask=value_mask,
        )
    tl.store(
        state_ptr + state_offsets,
        state,
        mask=value_mask[:, None],
    )
