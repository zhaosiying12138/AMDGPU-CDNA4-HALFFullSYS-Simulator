"""GemSim-owned Qwen3.5 text architecture registered through vLLM's API.

This module deliberately does not replace the upstream multimodal
``Qwen3_5ForConditionalGeneration`` architecture.  The plugin registers a
separate text-only architecture and reuses the upstream model implementation,
parameter names, weight loader, and attention modules.  Only the Qwen3.5
full-attention output gate is routed through the project-owned Triton op.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention


class GemsimQwen3NextAttention(nn.Module):
    """Parameter-compatible wrapper around an initialized upstream attention."""

    _MODULE_NAMES = (
        "qkv_proj",
        "o_proj",
        "rotary_emb",
        "attn",
        "q_norm",
        "k_norm",
    )
    _VALUE_NAMES = (
        "config",
        "hidden_size",
        "total_num_heads",
        "num_heads",
        "total_num_kv_heads",
        "num_kv_heads",
        "head_dim",
        "q_size",
        "kv_size",
        "scaling",
        "dual_chunk_attention_config",
        "attn_output_gate",
        "use_fused_qk_norm_rope_gate",
    )

    def __init__(self, source: Qwen3NextAttention) -> None:
        super().__init__()
        if type(source) is not Qwen3NextAttention:
            raise TypeError(f"unexpected attention source type: {type(source)}")
        for name in self._MODULE_NAMES:
            setattr(self, name, getattr(source, name))
        for name in self._VALUE_NAMES:
            setattr(self, name, getattr(source, name))

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return Qwen3NextAttention._project_qkv_gate(self, qkv, positions)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        attn_output = self.attn(q, k, v)
        if gate is not None:
            attn_output = torch.ops.gemsim.sigmoid_output_gate(attn_output, gate)
        output, _ = self.o_proj(attn_output)
        return output


class GemsimQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    """Dense Qwen3.5 text model with formal GemSim full-attention gating."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        if config.model_type != "qwen3_5_text":
            raise ValueError(
                "GemsimQwen3_5ForCausalLM accepts only dense qwen3_5_text"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        replaced = 0
        fp32_gdn_norms = 0
        for layer_index, layer in enumerate(self.model.layers):
            layer_type = config.layer_types[layer_index]
            if layer_type == "linear_attention":
                weight = layer.linear_attn.norm.weight
                if weight.shape != (config.linear_value_head_dim,):
                    raise RuntimeError("unexpected Qwen3.5 GDN norm weight shape")
                weight.data = weight.data.to(dtype=torch.float32)
                fp32_gdn_norms += 1
                continue
            if layer_type != "full_attention":
                raise ValueError(f"unsupported Qwen3.5 layer type: {layer_type}")
            layer.self_attn = GemsimQwen3NextAttention(layer.self_attn)
            replaced += 1
        expected = sum(kind == "full_attention" for kind in config.layer_types)
        if replaced != expected:
            raise RuntimeError(
                f"full-attention replacement mismatch: {replaced} != {expected}"
            )
        expected_gdn = sum(kind == "linear_attention" for kind in config.layer_types)
        if fp32_gdn_norms != expected_gdn:
            raise RuntimeError(
                f"FP32 GDN norm count mismatch: {fp32_gdn_norms} != {expected_gdn}"
            )


__all__ = ["GemsimQwen3NextAttention", "GemsimQwen3_5ForCausalLM"]
