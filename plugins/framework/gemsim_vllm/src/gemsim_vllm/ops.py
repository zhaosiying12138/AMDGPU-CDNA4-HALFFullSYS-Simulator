"""torch.library operators backed only by the generic GemSim Triton path."""

from __future__ import annotations

import math

import torch
import triton

from .kernels import (
    dense_linear_kernel,
    embedding_kernel,
    fused_add_gemma_rms_norm_kernel,
    gdn_conv_decode_kernel,
    gdn_recurrent_decode_kernel,
    gemma_rms_norm_kernel,
    rotary_embedding_kernel,
    rms_norm_gated_kernel,
    sigmoid_output_gate_kernel,
    silu_and_mul_kernel,
)


# vLLM's upstream Inductor functionalization pass resolves these canonical
# namespace symbols while it is constructing its target set, even when the
# selected OOT layers emit their own custom ops.  The CPU-only simulator
# product has no compiled vLLM `_C` extension, so provide declaration-only
# compatibility symbols at registration time.  They are never used as a
# fallback implementation; actual execution stays in the `gemsim::*` ops.
_UPSTREAM_COMPILE_SYMBOL_SCHEMAS = {
    "rotary_embedding": (
        "rotary_embedding(Tensor query, Tensor key, Tensor cos_sin_cache, "
        "Tensor positions, int head_size, int rotary_dim, bool is_neox_style) "
        "-> (Tensor, Tensor)"
    ),
    "fused_add_rms_norm": (
        "fused_add_rms_norm(Tensor input, Tensor residual, Tensor weight, "
        "float epsilon) -> ()"
    ),
    "fused_add_rms_norm_static_fp8_quant": (
        "fused_add_rms_norm_static_fp8_quant(Tensor input, Tensor residual, "
        "Tensor weight, Tensor out, Tensor scale, float epsilon) -> ()"
    ),
    "rms_norm_dynamic_per_token_quant": (
        "rms_norm_dynamic_per_token_quant(Tensor input, Tensor weight, "
        "float epsilon) -> (Tensor, Tensor)"
    ),
    "rms_norm": "rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor",
    "rms_norm_static_fp8_quant": (
        "rms_norm_static_fp8_quant(Tensor input, Tensor weight, Tensor out, "
        "Tensor scale, float epsilon) -> ()"
    ),
    "silu_and_mul": "silu_and_mul(Tensor input) -> Tensor",
    "silu_and_mul_quant": (
        "silu_and_mul_quant(Tensor input, Tensor scale) -> (Tensor, Tensor)"
    ),
    "fused_qk_norm_rope": (
        "fused_qk_norm_rope(Tensor query, Tensor key, Tensor q_weight, "
        "Tensor k_weight, Tensor cos_sin_cache, Tensor positions, float epsilon) "
        "-> (Tensor, Tensor)"
    ),
}
_upstream_compile_library: torch.library.Library | None = None


def register_upstream_compile_symbols() -> None:
    """Expose the canonical symbols expected by upstream vLLM graph passes."""
    global _upstream_compile_library
    if _upstream_compile_library is None:
        _upstream_compile_library = torch.library.Library("_C", "FRAGMENT")
    for name, schema in _UPSTREAM_COMPILE_SYMBOL_SCHEMAS.items():
        if not hasattr(torch.ops._C, name):
            _upstream_compile_library.define(schema)


def _require_staging_tensor(name: str, value: torch.Tensor) -> None:
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CPU staging tensor")


def validate_target() -> None:
    """Validate the selected Triton backend at the eager registration boundary."""
    target = triton.runtime.driver.active.get_current_target()
    if (target.backend, target.arch) != ("gemsim_amd", "gfx950"):
        raise RuntimeError(f"unexpected Triton target: {target}")


def _require_target() -> None:
    """Compatibility hook for eager callers; never run driver code in graphs.

    The target is validated once by :func:`gemsim_vllm.register_ops` before
    torch.compile captures the model.  Calling the Triton driver's Python
    introspection API from an attention/update method makes Dynamo try to
    inline the driver and reject an otherwise valid upstream graph.
    """
    return


def _require_bf16(name: str, value: torch.Tensor) -> None:
    _require_staging_tensor(name, value)
    if value.dtype != torch.bfloat16:
        raise ValueError(f"{name} must have dtype bfloat16")


@torch.library.custom_op("gemsim::dense_linear", mutates_args=())
def dense_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _require_target()
    _require_bf16("x", x)
    _require_bf16("weight", weight)
    if x.dim() != 2 or weight.dim() != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("dense_linear expects x[M,K] and weight[N,K]")
    m, k = x.shape
    n = weight.shape[0]
    output = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    dense_linear_kernel[(triton.cdiv(m, 16), triton.cdiv(n, 16))](
        x,
        weight,
        output,
        M=m,
        N=n,
        K=k,
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=32,
        num_warps=4,
    )
    return output


@dense_linear.register_fake
def _dense_linear_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2 or weight.dim() != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("dense_linear expects x[M,K] and weight[N,K]")
    return torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)


@torch.library.custom_op("gemsim::embedding", mutates_args=())
def embedding(token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _require_target()
    _require_staging_tensor("token_ids", token_ids)
    _require_bf16("weight", weight)
    if token_ids.dtype != torch.int64 or weight.dim() != 2:
        raise ValueError("embedding expects int64 token_ids and BF16 weight[V,H]")
    if token_ids.numel() and (
        int(torch.min(token_ids).item()) < 0
        or int(torch.max(token_ids).item()) >= weight.shape[0]
    ):
        raise ValueError("embedding token ID is outside the weight vocabulary")
    hidden = weight.shape[1]
    output = torch.empty(
        (*token_ids.shape, hidden), dtype=torch.bfloat16, device=weight.device
    )
    flat_ids = token_ids.view(-1)
    flat_output = output.view(flat_ids.numel(), hidden)
    embedding_kernel[(flat_ids.numel(), triton.cdiv(hidden, 256))](
        flat_ids,
        weight,
        flat_output,
        HIDDEN=hidden,
        BLOCK=256,
        num_warps=4,
    )
    return output


@embedding.register_fake
def _embedding_fake(token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError("embedding weight must have rank 2")
    return torch.empty(
        (*token_ids.shape, weight.shape[1]), dtype=weight.dtype, device=weight.device
    )


@torch.library.custom_op("gemsim::rotary_embedding", mutates_args=())
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_target()
    _require_staging_tensor("positions", positions)
    _require_bf16("query", query)
    _require_bf16("key", key)
    _require_bf16("cos_sin_cache", cos_sin_cache)
    if positions.dtype not in (torch.int32, torch.int64) or positions.dim() != 1:
        raise ValueError("GemSim text RoPE expects one-dimensional integer positions")
    if not is_neox_style:
        raise NotImplementedError("GemSim Qwen RoPE requires NeoX-style pairing")
    if head_size != 256 or rotary_dim != 64:
        raise NotImplementedError("bounded Qwen RoPE requires head=256, rotary=64")
    if query.dim() != 2 or key.dim() != 2 or query.shape[0] != key.shape[0]:
        raise ValueError("GemSim RoPE expects flattened query/key token matrices")
    if positions.numel() != query.shape[0]:
        raise ValueError("GemSim RoPE token/position count mismatch")
    if query.shape[1] % head_size or key.shape[1] % head_size:
        raise ValueError("GemSim RoPE hidden dimensions must contain full heads")
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[1] != rotary_dim:
        raise ValueError("GemSim RoPE cache shape mismatch")
    if positions.numel() and (
        int(torch.min(positions).item()) < 0
        or int(torch.max(positions).item()) >= cos_sin_cache.shape[0]
    ):
        raise ValueError("GemSim RoPE position is outside the cosine/sine cache")

    num_q_heads = query.shape[1] // head_size
    num_kv_heads = key.shape[1] // head_size
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)
    rotary_embedding_kernel[
        (query.shape[0], num_q_heads + num_kv_heads)
    ](
        positions,
        query,
        key,
        cos_sin_cache,
        query_out,
        key_out,
        query.stride(0),
        key.stride(0),
        query_out.stride(0),
        key_out.stride(0),
        cos_sin_cache.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_size,
        ROTARY_DIM=rotary_dim,
        HALF_ROTARY=rotary_dim // 2,
        num_warps=4,
    )
    return query_out, key_out


@rotary_embedding.register_fake
def _rotary_embedding_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del positions, cos_sin_cache, head_size, rotary_dim, is_neox_style
    return torch.empty_like(query), torch.empty_like(key)


def _norm_contract(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> tuple[int, int, int]:
    _require_bf16("x", x)
    _require_bf16("weight", weight)
    if x.dim() < 1 or weight.dim() != 1 or x.shape[-1] != weight.shape[0]:
        raise ValueError("Gemma RMSNorm expects x[...,H] and weight[H]")
    if epsilon != 1.0e-6:
        raise ValueError("the Qwen3.5 RMSNorm contract requires epsilon=1e-6")
    hidden = weight.shape[0]
    block = triton.next_power_of_2(hidden)
    if block > 65536:
        raise ValueError("RMSNorm hidden width exceeds the supported Triton block")
    return x.numel() // hidden, hidden, block


@torch.library.custom_op("gemsim::gemma_rms_norm", mutates_args=())
def gemma_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    _require_target()
    rows, hidden, block = _norm_contract(x, weight, epsilon)
    output = torch.empty_like(x)
    gemma_rms_norm_kernel[(rows,)](
        x,
        weight,
        output,
        HIDDEN=hidden,
        BLOCK=block,
        EPSILON=epsilon,
        num_warps=8 if block >= 1024 else 4,
    )
    return output


@gemma_rms_norm.register_fake
def _gemma_rms_norm_fake(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    if x.shape[-1] != weight.shape[0]:
        raise ValueError("Gemma RMSNorm shape mismatch")
    return torch.empty_like(x)


@torch.library.custom_op("gemsim::fused_add_gemma_rms_norm", mutates_args=())
def fused_add_gemma_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_target()
    if x.shape != residual.shape:
        raise ValueError("fused RMSNorm x and residual shapes must match")
    _require_bf16("residual", residual)
    rows, hidden, block = _norm_contract(x, weight, epsilon)
    output = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    fused_add_gemma_rms_norm_kernel[(rows,)](
        x,
        residual,
        weight,
        output,
        residual_out,
        HIDDEN=hidden,
        BLOCK=block,
        EPSILON=epsilon,
        num_warps=8 if block >= 1024 else 4,
    )
    return output, residual_out


@fused_add_gemma_rms_norm.register_fake
def _fused_add_gemma_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape != residual.shape or x.shape[-1] != weight.shape[0]:
        raise ValueError("fused Gemma RMSNorm shape mismatch")
    return torch.empty_like(x), torch.empty_like(x)


@torch.library.custom_op("gemsim::silu_and_mul", mutates_args=())
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    _require_target()
    _require_bf16("x", x)
    if x.dim() < 1 or x.shape[-1] % 2:
        raise ValueError("SiluAndMul expects an even final dimension")
    half = x.shape[-1] // 2
    rows = x.numel() // x.shape[-1]
    output = torch.empty((*x.shape[:-1], half), dtype=x.dtype, device=x.device)
    silu_and_mul_kernel[(rows, math.ceil(half / 1024))](
        x,
        output,
        HALF_WIDTH=half,
        FULL_WIDTH=x.shape[-1],
        BLOCK=1024,
        num_warps=4,
    )
    return output


@silu_and_mul.register_fake
def _silu_and_mul_fake(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2:
        raise ValueError("SiluAndMul expects an even final dimension")
    return torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device)


@torch.library.custom_op("gemsim::sigmoid_output_gate", mutates_args=())
def sigmoid_output_gate(
    attention: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    _require_target()
    _require_bf16("attention", attention)
    _require_bf16("gate", gate)
    if attention.shape != gate.shape:
        raise ValueError("GemSim attention output and gate shapes must match")
    output = torch.empty_like(attention)
    sigmoid_output_gate_kernel[(triton.cdiv(attention.numel(), 256),)](
        attention,
        gate,
        output,
        ELEMENTS=attention.numel(),
        BLOCK=256,
        num_warps=4,
    )
    return output


@sigmoid_output_gate.register_fake
def _sigmoid_output_gate_fake(
    attention: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    if attention.shape != gate.shape:
        raise ValueError("GemSim attention output and gate shapes must match")
    return torch.empty_like(attention)


@torch.library.custom_op("gemsim::rms_norm_gated", mutates_args=())
def rms_norm_gated(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    _require_target()
    _require_bf16("x", x)
    _require_bf16("gate", gate)
    _require_staging_tensor("weight", weight)
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("rms_norm_gated weight must be BF16 or FP32")
    if x.shape != gate.shape or x.dim() < 2 or x.shape[-1] != weight.shape[0]:
        raise ValueError("GDN RMSNormGated expects matching [...,head_dim] tensors")
    if weight.numel() <= 0:
        raise ValueError("GDN RMSNormGated weight must be nonempty")
    head_dim = int(weight.shape[0])
    block = triton.next_power_of_2(head_dim)
    output = torch.empty_like(x)
    rms_norm_gated_kernel[(x.numel() // head_dim,)](
        x,
        gate,
        weight,
        output,
        HEAD_DIM=head_dim,
        BLOCK=block,
        EPSILON=epsilon,
        num_warps=4,
    )
    return output


@rms_norm_gated.register_fake
def _rms_norm_gated_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if x.shape != gate.shape or x.shape[-1] != weight.shape[0]:
        raise ValueError("bounded Qwen GDN RMSNormGated shape mismatch")
    return torch.empty_like(x)


@torch.library.custom_op(
    "gemsim::gdn_conv_decode", mutates_args=("state_cache",)
)
def gdn_conv_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    _require_target()
    _require_bf16("x", x)
    _require_bf16("weight", weight)
    _require_bf16("state_cache", state_cache)
    _require_staging_tensor("state_indices", state_indices)
    if x.dim() != 2 or weight.dim() != 2 or weight.shape[1] != 4 or x.shape[1] != weight.shape[0]:
        raise ValueError("GDN conv expects [tokens,channels] and [channels,4]")
    channels = int(x.shape[1])
    tokens = x.shape[0]
    if not 1 <= tokens <= 16:
        raise NotImplementedError("bounded Qwen GDN conv supports 1..16 tokens")
    if state_indices.shape != (1,) or state_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("GDN conv state index must be one integer")
    if state_cache.dim() != 3:
        raise ValueError("GDN conv state cache must have rank 3")
    if state_cache.shape[1:] == (channels, 3):
        channel_axis, time_axis = 1, 2
    elif state_cache.shape[1:] == (3, channels):
        channel_axis, time_axis = 2, 1
    else:
        raise ValueError("GDN conv state cache must use DS or SD layout")
    state_index = int(state_indices[0].item())
    if state_index < 0 or state_index >= state_cache.shape[0]:
        raise ValueError("GDN conv state index is outside the cache")
    output = torch.empty_like(x)
    gdn_conv_decode_kernel[(triton.cdiv(channels, 256),)](
        x,
        weight,
        state_cache,
        state_indices,
        output,
        state_cache.stride(0),
        state_cache.stride(channel_axis),
        state_cache.stride(time_axis),
        TOKENS=tokens,
        CHANNELS=channels,
        BLOCK=256,
        num_warps=4,
    )
    return output


@gdn_conv_decode.register_fake
def _gdn_conv_decode_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "gemsim::gdn_recurrent_decode", mutates_args=("state_cache",)
)
def gdn_recurrent_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    _require_target()
    for name, value in (
        ("mixed_qkv", mixed_qkv),
        ("a", a),
        ("b", b),
        ("dt_bias", dt_bias),
    ):
        _require_bf16(name, value)
    _require_staging_tensor("a_log", a_log)
    _require_staging_tensor("state_cache", state_cache)
    _require_staging_tensor("state_indices", state_indices)
    if a_log.dtype != torch.float32 or state_cache.dtype != torch.float32:
        raise ValueError("GDN recurrent A_log/state cache must be FP32")
    if mixed_qkv.dim() != 2 or mixed_qkv.shape[1] <= 0:
        raise ValueError("GDN recurrent expects [tokens,channels]")
    channels = int(mixed_qkv.shape[1])
    if channels % 3 != 0:
        raise ValueError("GDN recurrent channel width must be divisible by 3")
    num_heads = int(a_log.shape[0])
    if channels != 3 * num_heads * 128:
        raise ValueError("GDN recurrent channel/head dimensions do not match")
    tokens = mixed_qkv.shape[0]
    if not 1 <= tokens <= 16:
        raise NotImplementedError("bounded Qwen GDN recurrent supports 1..16 tokens")
    if a.shape != (tokens, num_heads) or b.shape != (tokens, num_heads):
        raise ValueError("bounded Qwen GDN recurrent a/b shape mismatch")
    if a_log.shape != (num_heads,) or dt_bias.shape != (num_heads,):
        raise ValueError("GDN recurrent parameter shape mismatch")
    if state_cache.dim() != 4 or state_cache.shape[1:] != (num_heads, 128, 128):
        raise ValueError("GDN recurrent state cache shape mismatch")
    if state_indices.shape != (1,) or state_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("GDN recurrent state index must be one integer")
    state_index = int(state_indices[0].item())
    if state_index < 0 or state_index >= state_cache.shape[0]:
        raise ValueError("GDN recurrent state index is outside the cache")
    output = torch.empty(
        (tokens, num_heads, 128), dtype=torch.bfloat16, device=mixed_qkv.device
    )
    gdn_recurrent_decode_kernel[(4, num_heads)](
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state_cache,
        state_indices,
        output,
        TOKENS=tokens,
        KEY_DIM=128,
        VALUE_DIM=128,
        NUM_HEADS=num_heads,
        VALUE_BLOCK=32,
        SCALE=128**-0.5,
        EPSILON=1.0e-6,
        SOFTPLUS_THRESHOLD=20.0,
        num_warps=1,
        num_stages=3,
    )
    return output


@gdn_recurrent_decode.register_fake
def _gdn_recurrent_decode_fake(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (mixed_qkv.shape[0], a.shape[1], 128),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )


__all__ = [
    "dense_linear",
    "embedding",
    "fused_add_gemma_rms_norm",
    "gdn_conv_decode",
    "gdn_recurrent_decode",
    "gemma_rms_norm",
    "rotary_embedding",
    "rms_norm_gated",
    "sigmoid_output_gate",
    "silu_and_mul",
]
