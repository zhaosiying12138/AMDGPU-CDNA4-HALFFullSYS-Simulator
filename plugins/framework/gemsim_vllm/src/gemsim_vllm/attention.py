"""Bounded GemSim full-attention backend for the formal vLLM plugin.

The backend supports one-request decode and empty-cache causal prefill with a
BF16 K/V cache and a maximum context of 128 tokens. Unsupported vLLM metadata
is rejected before target work is launched; there is no CPU attention fallback
hidden behind this class.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.v1.attention.backend import AttentionImpl, AttentionType
from vllm.v1.attention.backends.cpu_attn import (
    CPUAttentionBackend,
    CPUAttentionMetadata,
    CPUAttentionMetadataBuilder,
)

from .ops import _require_bf16, _require_target


MAX_CONTEXT = 128
MAX_PREFILL_TOKENS = 16


@triton.jit
def _store_kv_kernel(
    key_ptr,
    value_ptr,
    cache_ptr,
    slot_ptr,
    key_stride_t,
    key_stride_h,
    value_stride_t,
    value_stride_h,
    cache_stride_b,
    cache_stride_s,
    cache_stride_h,
    cache_stride_c,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, head_dim)
    slot = tl.load(slot_ptr + token).to(tl.int64)
    block = slot // block_size
    block_slot = slot % block_size
    cache_base = (
        block * cache_stride_b
        + block_slot * cache_stride_s
        + head * cache_stride_h
    )
    key = tl.load(key_ptr + token * key_stride_t + head * key_stride_h + offsets)
    value = tl.load(
        value_ptr + token * value_stride_t + head * value_stride_h + offsets
    )
    tl.store(cache_ptr + cache_base + offsets * cache_stride_c, key)
    tl.store(
        cache_ptr + cache_base + (head_dim + offsets) * cache_stride_c,
        value,
    )


@triton.jit
def _decode_attention_kernel_impl(
    query_ptr,
    cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    output_ptr,
    query_stride_t,
    query_stride_h,
    output_stride_t,
    output_stride_h,
    cache_stride_b,
    cache_stride_s,
    cache_stride_h,
    cache_stride_c,
    block_table_stride,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_context: tl.constexpr,
    scale: tl.constexpr,
):
    q_head = tl.program_id(0)
    kv_group = num_q_heads // num_kv_heads
    kv_head = q_head // kv_group
    dim = tl.arange(0, 64)
    positions = tl.arange(0, max_context)
    seq_len = tl.load(seq_lens_ptr).to(tl.int64)
    valid_positions = positions < seq_len
    block_ids = tl.load(
        block_table_ptr + (positions // block_size) * block_table_stride,
        mask=valid_positions,
        other=0,
    ).to(tl.int64)
    block_slots = positions % block_size
    cache_base = (
        block_ids[:, None] * cache_stride_b
        + block_slots[:, None] * cache_stride_s
        + kv_head * cache_stride_h
    )
    scores = tl.zeros((max_context,), dtype=tl.float32)
    for dim_start in range(0, head_dim, 64):
        d = dim_start + dim
        dmask = d < head_dim
        q = tl.load(
            query_ptr + q_head * query_stride_h + d,
            mask=dmask,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            cache_ptr
            + cache_base
            + d[None, :] * cache_stride_c,
            mask=valid_positions[:, None] & dmask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores += tl.sum(k * q[None, :], axis=1)
    scores = scores * scale
    scores = tl.where(valid_positions, scores, -float("inf"))
    scores -= tl.max(scores, axis=0)
    probabilities = tl.exp(scores)
    probabilities /= tl.sum(probabilities, axis=0)
    for dim_start in range(0, head_dim, 64):
        d = dim_start + dim
        dmask = d < head_dim
        v = tl.load(
            cache_ptr
            + cache_base
            + (head_dim + d)[None, :] * cache_stride_c,
            mask=valid_positions[:, None] & dmask[None, :],
            other=0.0,
        ).to(tl.float32)
        result = tl.sum(probabilities[:, None] * v, axis=0)
        tl.store(
            output_ptr + q_head * output_stride_h + d,
            result.to(tl.bfloat16),
            mask=dmask,
        )


@triton.jit
def _empty_prefill_attention_kernel_impl(
    query_ptr,
    cache_ptr,
    block_table_ptr,
    output_ptr,
    query_stride_t,
    query_stride_h,
    output_stride_t,
    output_stride_h,
    cache_stride_b,
    cache_stride_s,
    cache_stride_h,
    cache_stride_c,
    block_table_stride,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_context: tl.constexpr,
    scale: tl.constexpr,
):
    token = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_group = num_q_heads // num_kv_heads
    kv_head = q_head // kv_group
    dim = tl.arange(0, 64)
    positions = tl.arange(0, max_context)
    valid_positions = positions <= token
    block_ids = tl.load(
        block_table_ptr + (positions // block_size) * block_table_stride,
        mask=valid_positions,
        other=0,
    ).to(tl.int64)
    block_slots = positions % block_size
    cache_base = (
        block_ids[:, None] * cache_stride_b
        + block_slots[:, None] * cache_stride_s
        + kv_head * cache_stride_h
    )
    scores = tl.zeros((max_context,), dtype=tl.float32)
    for dim_start in range(0, head_dim, 64):
        d = dim_start + dim
        dmask = d < head_dim
        query = tl.load(
            query_ptr + token * query_stride_t + q_head * query_stride_h + d,
            mask=dmask,
            other=0.0,
        ).to(tl.float32)
        key = tl.load(
            cache_ptr + cache_base + d[None, :] * cache_stride_c,
            mask=valid_positions[:, None] & dmask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores += tl.sum(key * query[None, :], axis=1)
    scores = tl.where(valid_positions, scores * scale, -float("inf"))
    scores -= tl.max(scores, axis=0)
    probabilities = tl.exp(scores)
    probabilities /= tl.sum(probabilities, axis=0)
    for dim_start in range(0, head_dim, 64):
        d = dim_start + dim
        dmask = d < head_dim
        value = tl.load(
            cache_ptr
            + cache_base
            + (head_dim + d)[None, :] * cache_stride_c,
            mask=valid_positions[:, None] & dmask[None, :],
            other=0.0,
        ).to(tl.float32)
        result = tl.sum(probabilities[:, None] * value, axis=0)
        tl.store(
            output_ptr
            + token * output_stride_t
            + q_head * output_stride_h
            + d,
            result.to(tl.bfloat16),
            mask=dmask,
        )


@torch.library.custom_op("gemsim::kv_cache_update", mutates_args=("kv_cache",))
def _kv_cache_update_op(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """Opaque bridge launch for the cache write used by upstream attention."""
    _store_kv_kernel[(key.shape[0], num_kv_heads)](
        key,
        value,
        kv_cache,
        slot_mapping,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_warps=4,
    )
    # A tensor return keeps the custom-op ABI explicit while the mutation is
    # represented by mutates_args.  The caller only needs the side effect.
    return torch.empty((0,), dtype=kv_cache.dtype, device=kv_cache.device)


@_kv_cache_update_op.register_fake
def _kv_cache_update_op_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    del key, value, slot_mapping, num_kv_heads, head_dim, block_size
    return torch.empty((0,), dtype=kv_cache.dtype, device=kv_cache.device)


@torch.library.custom_op("gemsim::decode_attention", mutates_args=("output",))
def _decode_attention_op(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_context: int,
    scale: float,
) -> None:
    _decode_attention_kernel_impl[(num_q_heads,)](
        query,
        kv_cache,
        block_table,
        seq_lens,
        output,
        query.stride(0),
        query.stride(1),
        output.stride(0),
        output.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        block_table.stride(1),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_context=max_context,
        scale=scale,
        num_warps=4,
    )
    return None


@_decode_attention_op.register_fake
def _decode_attention_op_fake(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_context: int,
    scale: float,
) -> None:
    del query, kv_cache, block_table, seq_lens, output, num_q_heads, num_kv_heads
    del head_dim, block_size, max_context, scale
    return None


@torch.library.custom_op("gemsim::prefill_attention", mutates_args=("output",))
def _prefill_attention_op(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_context: int,
    scale: float,
) -> None:
    _empty_prefill_attention_kernel_impl[(query.shape[0], num_q_heads)](
        query,
        kv_cache,
        block_table,
        output,
        query.stride(0),
        query.stride(1),
        output.stride(0),
        output.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        block_table.stride(1),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_context=max_context,
        scale=scale,
        num_warps=4,
    )
    return None


@_prefill_attention_op.register_fake
def _prefill_attention_op_fake(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_context: int,
    scale: float,
) -> None:
    del query, kv_cache, block_table, output, num_q_heads, num_kv_heads
    del head_dim, block_size, max_context, scale
    return None


class GemsimAttentionBackend(CPUAttentionBackend):
    """The bounded target backend, exposed under vLLM's CPU_ATTN enum slot."""

    @staticmethod
    def get_name() -> str:
        # Attention.__init__ indexes this name in AttentionBackendEnum.  The
        # class path is still GemSim-owned and its implementation is not CPU.
        return "CPU_ATTN"

    @staticmethod
    def get_impl_cls() -> type["GemsimAttentionImpl"]:
        return GemsimAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[CPUAttentionMetadataBuilder]:
        return CPUAttentionMetadataBuilder

    @classmethod
    def get_required_kv_cache_layout(cls) -> str:
        return "NHD"

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 3, 2, 4)
        return (0, 2, 1, 3)


class GemsimAttentionImpl(AttentionImpl[CPUAttentionMetadata]):
    """Single-request decode and empty-cache prefill through GemSim Triton."""

    supports_dcp = False
    supports_pcp = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **_: object,
    ) -> None:
        del alibi_slopes, sliding_window, logits_soft_cap, kv_sharing_target_layer_name
        if attn_type != AttentionType.DECODER:
            raise ValueError("GemSim attention currently supports decoder attention only")
        if num_kv_heads is None or num_heads % num_kv_heads:
            raise ValueError("GemSim attention requires divisible Q/KV head counts")
        if head_size != 256 or num_heads < 1 or num_kv_heads < 1:
            raise ValueError(
                "GemSim attention requires positive local Q/KV heads and head_dim=256"
            )
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("GemSim attention requires an unquantized BF16 KV cache")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.num_queries_per_kv = num_heads // num_kv_heads

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        del layer
        _require_target()
        # Qwen packs Q/gate/K/V into one projection. K/V can therefore be
        # strided views for multi-token requests; own contiguous staging
        # buffers before crossing the managed-runtime boundary.
        key = key.contiguous()
        value = value.contiguous()
        _require_bf16("key", key)
        _require_bf16("value", value)
        _require_bf16("kv_cache", kv_cache)
        if key.ndim != 3 or value.shape != key.shape:
            raise ValueError("GemSim KV update expects [tokens, kv_heads, head_dim]")
        tokens = key.shape[0]
        if slot_mapping.numel() != tokens or not 1 <= tokens <= MAX_PREFILL_TOKENS:
            raise NotImplementedError(
                f"GemSim KV updates support 1..{MAX_PREFILL_TOKENS} tokens"
            )
        if kv_cache.ndim != 4:
            raise ValueError("GemSim KV cache must be rank 4")
        block_size = kv_cache.shape[1]
        capacity = kv_cache.shape[0] * block_size
        # Value-dependent tensor reductions are eager input validation only.
        # Dynamo cannot specialize on arbitrary slot contents while capturing
        # the upstream model graph; the Triton kernel still receives the exact
        # validated shape/stride contract.
        if not torch._dynamo.is_compiling() and (
            int(slot_mapping.min().item()) < 0
            or int(slot_mapping.max().item()) >= capacity
        ):
            raise ValueError("GemSim KV slot mapping is outside the cache")
        _kv_cache_update_op(
            key,
            value,
            kv_cache,
            slot_mapping,
            self.num_kv_heads,
            self.head_size,
            block_size,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CPUAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, key, value
        _require_target()
        _require_bf16("query", query)
        _require_bf16("kv_cache", kv_cache)
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("GemSim attention has no fused output quantization")
        if attn_metadata is None:
            raise RuntimeError("GemSim attention requires runtime metadata")
        tokens = query.shape[0]
        if not 1 <= tokens <= MAX_PREFILL_TOKENS:
            raise NotImplementedError(
                f"GemSim attention supports 1..{MAX_PREFILL_TOKENS} tokens"
            )
        if not torch._dynamo.is_compiling():
            if (
                attn_metadata.num_actual_tokens != tokens
                or attn_metadata.query_start_loc.numel() != 2
                or int(attn_metadata.query_start_loc[0].item()) != 0
                or int(attn_metadata.query_start_loc[-1].item()) != tokens
            ):
                raise NotImplementedError("GemSim attention requires one request")
            if int(attn_metadata.seq_lens.max().item()) > MAX_CONTEXT:
                raise NotImplementedError(
                    f"GemSim attention context exceeds bounded limit {MAX_CONTEXT}"
                )
        if query.ndim != 3 or query.shape != (
            tokens,
            self.num_heads,
            self.head_size,
        ):
            raise ValueError("GemSim query shape does not match local attention heads")
        if output.shape != query.shape:
            raise ValueError("GemSim attention output shape mismatch")
        block_size = kv_cache.shape[1]
        if block_size <= 0 or kv_cache.shape[-1] != 2 * self.head_size:
            raise ValueError("GemSim KV cache shape mismatch")
        if tokens == 1:
            _decode_attention_op(
                query,
                kv_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                output,
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                block_size,
                MAX_CONTEXT,
                self.scale,
            )
        else:
            expected_slots = torch.arange(tokens, dtype=torch.int32)
            slot_mapping = getattr(attn_metadata, "slot_mapping", None)
            if not torch._dynamo.is_compiling() and (
                getattr(attn_metadata, "num_decode_tokens", 0) != 0
                or getattr(attn_metadata, "max_query_len", tokens) != tokens
                or getattr(attn_metadata, "max_seq_len", tokens) != tokens
                or attn_metadata.seq_lens.shape != (1,)
                or int(attn_metadata.seq_lens[0].item()) != tokens
                or slot_mapping is None
                or not torch.equal(slot_mapping.cpu(), expected_slots)
            ):
                raise NotImplementedError(
                    "GemSim prefill requires one empty-cache causal request"
                )
            _prefill_attention_op(
                query,
                kv_cache,
                attn_metadata.block_table,
                output,
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                block_size,
                MAX_CONTEXT,
                self.scale,
            )
        return output


__all__ = ["GemsimAttentionBackend", "GemsimAttentionImpl"]
