"""Formal vLLM OOT replacements backed by registered GemSim Torch ops."""

from __future__ import annotations

import torch

from vllm.forward_context import get_forward_context
from vllm.distributed import get_tp_group
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from .row_parallel import validate_row_parallel_contract


class _GemsimLinearForward:
    """Shared unquantized local-GEMM path for vLLM linear layers.

    The upstream layer owns sharding, bias policy, and any collective around
    this method.  This method only executes the local ``[M,K] x [N,K]``
    projection, so the same bridge applies to replicated, column, merged, and
    QKV projections at every supported tensor-parallel rank.
    """

    def _gemsim_forward(self, x: torch.Tensor):
        if self.quant_config is not None:
            raise NotImplementedError("GemSim linear does not accept quantized weights")
        if self.bias is not None:
            raise NotImplementedError("bounded Qwen GemSim linear requires bias=False")
        if x.shape[-1] != self.weight.shape[1]:
            raise ValueError("GemSim linear input/weight shape mismatch")

        output_shape = (*x.shape[:-1], self.weight.shape[0])
        flat_input = x.reshape(-1, x.shape[-1]).contiguous()
        output = torch.ops.gemsim.dense_linear(flat_input, self.weight).view(
            output_shape
        )
        if not self.return_bias:
            return output
        return output, None


@PluggableLayer.register_oot(name="VocabParallelEmbedding")
class GemsimVocabParallelEmbedding(VocabParallelEmbedding):
    """Vocabulary-sharded embedding through the generic device SUM path."""

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                "GemSim embedding does not accept quantized weights"
            )
        if self.num_added_embeddings != 0:
            raise NotImplementedError("GemSim embedding does not accept LoRA rows")
        if self.tp_size == 1:
            return torch.ops.gemsim.embedding(input_.long(), self.weight)
        # Match upstream VocabParallelEmbedding: mask IDs outside this rank's
        # shard, run the local lookup, then sum the mutually-exclusive partials
        # through the device communicator.
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            get_masked_input_and_mask,
        )

        masked_input, input_mask = get_masked_input_and_mask(
            input_,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            self.shard_indices.num_org_vocab_padding,
            self.shard_indices.added_vocab_start_index,
            self.shard_indices.added_vocab_end_index,
        )
        output_parallel = torch.ops.gemsim.embedding(masked_input.long(), self.weight)
        output_parallel = output_parallel.masked_fill(input_mask.unsqueeze(-1), 0)
        return get_tp_group().all_reduce(output_parallel)


@CustomOp.register_oot(name="SiluAndMul")
class GemsimSiluAndMul(SiluAndMul):
    """Replace vLLM's SwiGLU activation without changing vLLM source."""

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.gemsim.silu_and_mul(x)


@CustomOp.register_oot(name="GemmaRMSNorm")
class GemsimGemmaRMSNorm(GemmaRMSNorm):
    """Replace plain and fused Gemma RMSNorm through the OOT dispatch hook."""

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Q/K projections are views into a packed per-token projection. For
        # more than one token their leading stride is larger than the logical
        # row, so materialize the project-owned staging buffer explicitly.
        x = x.contiguous()
        if residual is None:
            return torch.ops.gemsim.gemma_rms_norm(
                x, self.weight, self.variance_epsilon
            )
        return torch.ops.gemsim.fused_add_gemma_rms_norm(
            x, residual.contiguous(), self.weight, self.variance_epsilon
        )


@CustomOp.register_oot(name="RotaryEmbedding")
class GemsimRotaryEmbedding(RotaryEmbedding):
    """Replace Qwen text partial RoPE with the GemSim Triton operator."""

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if key is None:
            raise NotImplementedError("GemSim RoPE requires an explicit key tensor")
        cache = self._match_cos_sin_cache_dtype(query)
        return torch.ops.gemsim.rotary_embedding(
            positions,
            query,
            key,
            cache,
            self.head_size,
            self.rotary_dim,
            self.is_neox_style,
        )


@CustomOp.register_oot(name="MRotaryEmbedding")
class GemsimMRotaryEmbedding(MRotaryEmbedding):
    """Route text-only Qwen3.5 MRoPE through the GemSim Triton operator."""

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if positions.ndim != 1:
            raise NotImplementedError(
                "GemSim text architecture does not accept 2D multimodal MRoPE"
            )
        if offsets is not None:
            raise NotImplementedError("GemSim text MRoPE does not accept offsets")
        if key is None:
            raise NotImplementedError("GemSim text MRoPE requires an explicit key")
        cache = self._match_cos_sin_cache_dtype(query)
        return torch.ops.gemsim.rotary_embedding(
            positions,
            query,
            key,
            cache,
            self.head_size,
            self.rotary_dim,
            self.is_neox_style,
        )


@CustomOp.register_oot(name="RMSNormGated")
class GemsimRMSNormGated(RMSNormGated):
    """Replace Qwen3.5 GDN output norm+SiLU gate through Triton."""

    def forward_oot(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z is None:
            raise NotImplementedError("GemSim GDN RMSNormGated requires a gate")
        if self.group_size is not None or not self.norm_before_gate:
            raise NotImplementedError("unsupported GemSim RMSNormGated variant")
        if self.activation not in ("silu", "swish"):
            raise NotImplementedError("GemSim GDN gate requires SiLU")
        return torch.ops.gemsim.rms_norm_gated(x, z, self.weight, self.eps)


@PluggableLayer.register_oot(name="QwenGatedDeltaNetAttention")
class GemsimQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    """Bounded single-request Qwen3.5 GDN through project Triton ops."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            hidden_states.dim() != 2
            or hidden_states.shape[1] != self.hidden_size
            or not 1 <= hidden_states.shape[0] <= 16
        ):
            raise NotImplementedError(
                "GemSim GDN accepts one request with 1..16 tokens"
            )
        tokens = hidden_states.shape[0]
        metadata_by_layer = get_forward_context().attn_metadata
        if not isinstance(metadata_by_layer, dict) or self.prefix not in metadata_by_layer:
            raise RuntimeError("GemSim GDN requires vLLM forward metadata")
        metadata = metadata_by_layer[self.prefix]
        if not isinstance(metadata, GDNAttentionMetadata):
            raise TypeError("GemSim GDN metadata type mismatch")
        decode = (
            tokens == 1
            and metadata.num_actual_tokens == 1
            and metadata.num_decodes == 1
            and metadata.num_decode_tokens == 1
            and metadata.num_prefills == 0
        )
        prefill = (
            tokens > 1
            and metadata.num_actual_tokens == tokens
            and metadata.num_prefills == 1
            and metadata.num_prefill_tokens == tokens
            and metadata.num_decodes == 0
            and metadata.num_decode_tokens == 0
            and metadata.non_spec_query_start_loc is not None
            and metadata.non_spec_query_start_loc.shape == (2,)
            and (
                torch._dynamo.is_compiling()
                or (
                    int(metadata.non_spec_query_start_loc[0].item()) == 0
                    and int(metadata.non_spec_query_start_loc[1].item()) == tokens
                )
            )
        )
        if (
            not (decode or prefill)
            or metadata.num_spec_decodes != 0
            or metadata.spec_sequence_masks is not None
        ):
            raise NotImplementedError("unsupported GemSim GDN request metadata")
        state_indices = (
            metadata.prefill_state_indices
            if prefill and metadata.prefill_state_indices is not None
            else metadata.non_spec_state_indices_tensor
        )
        if state_indices is None or state_indices.numel() < 1:
            raise ValueError("GemSim GDN state index is absent")
        state_indices = state_indices[:1].contiguous()
        if not hasattr(self, "kv_cache") or len(self.kv_cache) != 2:
            raise RuntimeError("GemSim GDN cache is not bound")
        conv_state, recurrent_state = self.kv_cache

        mixed_qkvz, projection_bias = self.in_proj_qkvz(hidden_states)
        ba, ba_bias = self.in_proj_ba(hidden_states)
        if projection_bias is not None or ba_bias is not None:
            raise RuntimeError("GemSim Qwen GDN projections must be bias-free")
        local_key_dim = int(self.key_dim // self.tp_size)
        local_value_dim = int(self.value_dim // self.tp_size)
        local_heads = int(self.num_v_heads // self.tp_size)
        qkv_width = local_key_dim * 2 + local_value_dim
        mixed_qkv, z = mixed_qkvz.split([qkv_width, local_value_dim], dim=-1)
        b, a = ba.chunk(2, dim=-1)
        conv_weight = self.conv1d.weight.view(qkv_width, self.conv_kernel_size)
        convolved = torch.ops.gemsim.gdn_conv_decode(
            mixed_qkv.contiguous(),
            conv_weight,
            conv_state,
            state_indices,
        )
        core = torch.ops.gemsim.gdn_recurrent_decode(
            convolved,
            a.contiguous(),
            b.contiguous(),
            self.A_log,
            self.dt_bias,
            recurrent_state,
            state_indices,
        )
        return self._output_projection(core, z.view(tokens, local_heads, self.head_v_dim))


@PluggableLayer.register_oot(name="ReplicatedLinear")
class GemsimReplicatedLinear(_GemsimLinearForward, ReplicatedLinear):
    def forward(self, x: torch.Tensor):
        return self._gemsim_forward(x)


@PluggableLayer.register_oot(name="ColumnParallelLinear")
class GemsimColumnParallelLinear(_GemsimLinearForward, ColumnParallelLinear):
    def forward(self, x: torch.Tensor):
        return self._gemsim_forward(x)


@PluggableLayer.register_oot(name="MergedColumnParallelLinear")
class GemsimMergedColumnParallelLinear(
    _GemsimLinearForward, MergedColumnParallelLinear
):
    def forward(self, x: torch.Tensor):
        return self._gemsim_forward(x)


@PluggableLayer.register_oot(name="QKVParallelLinear")
class GemsimQKVParallelLinear(_GemsimLinearForward, QKVParallelLinear):
    def forward(self, x: torch.Tensor):
        return self._gemsim_forward(x)


@PluggableLayer.register_oot(name="RowParallelLinear")
class GemsimRowParallelLinear(RowParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )
        if (
            quant_config is not None
            or type(self.quant_method) is not UnquantizedLinearMethod
        ):
            raise NotImplementedError(
                "GemSim row parallel does not accept quantized weights"
            )
        validate_row_parallel_contract(
            input_size=self.input_size,
            output_size=self.output_size,
            input_size_per_partition=self.input_size_per_partition,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            weight=self.weight,
        )
        if self.bias is not None and not self.skip_bias_add:
            raise NotImplementedError(
                "GemSim row parallel requires deferred bias until device bias add is available"
            )
        self.quant_method = GemsimUnquantizedRowParallelMethod()


class GemsimUnquantizedRowParallelMethod(UnquantizedLinearMethod):
    """Local GEMM injected into the unchanged pinned RowParallelLinear forward."""

    def apply(
        self,
        layer: RowParallelLinear,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise NotImplementedError(
                "GemSim row parallel does not perform host-side bias arithmetic"
            )
        validate_row_parallel_contract(
            input_size=layer.input_size,
            output_size=layer.output_size,
            input_size_per_partition=layer.input_size_per_partition,
            tp_size=layer.tp_size,
            tp_rank=layer.tp_rank,
            weight=layer.weight,
        )
        if (
            not isinstance(x, torch.Tensor)
            or x.dim() < 1
            or x.shape[-1] != layer.weight.shape[1]
        ):
            raise ValueError("GemSim row-parallel local input/weight shape mismatch")
        if (
            x.dtype != torch.bfloat16
            or layer.weight.dtype != torch.bfloat16
            or x.device.type != "cpu"
            or layer.weight.device != x.device
            or not layer.weight.is_contiguous()
        ):
            raise ValueError(
                "GemSim row-parallel local GEMM requires contiguous CPU BF16 weights"
            )

        output_shape = (*x.shape[:-1], layer.weight.shape[0])
        flat_input = x.reshape(-1, x.shape[-1]).contiguous()
        output = torch.ops.gemsim.dense_linear(flat_input, layer.weight)
        if (
            not isinstance(output, torch.Tensor)
            or tuple(output.shape) != (flat_input.shape[0], layer.weight.shape[0])
            or output.dtype != x.dtype
            or output.device != x.device
            or not output.is_contiguous()
        ):
            raise RuntimeError("GemSim row-parallel local GEMM returned an invalid tensor")
        # Storage identity and input immutability are backend/runtime contracts
        # checked by the device executor and host tests.  They must not be
        # inspected here: this method is compiled by upstream torch.compile,
        # and UntypedStorage.data_ptr is intentionally not traceable by Dynamo.
        return output.view(output_shape)


__all__ = [
    "GemsimColumnParallelLinear",
    "GemsimGemmaRMSNorm",
    "GemsimMergedColumnParallelLinear",
    "GemsimMRotaryEmbedding",
    "GemsimQwenGatedDeltaNetAttention",
    "GemsimQKVParallelLinear",
    "GemsimReplicatedLinear",
    "GemsimRotaryEmbedding",
    "GemsimRMSNormGated",
    "GemsimRowParallelLinear",
    "GemsimUnquantizedRowParallelMethod",
    "GemsimSiluAndMul",
    "GemsimVocabParallelEmbedding",
]
