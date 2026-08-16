#!/usr/bin/env python3
"""Build a source-grounded operator contract for Qwen3.5-0.8B.

This tool deliberately stops at a contract gate.  It does not execute a model,
and a CPU or NVIDIA execution is never reported as an AMD pass.  The manifest
is useful before the host-native backend exists because it makes the exact
operator closure and its source evidence reproducible from the pinned local
checkouts.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from qwen35_operator_work_queue import (  # noqa: E402
    derive_work_item_status,
    queue_summary,
    runtime_evidence_policy,
    validate_manifest,
    work_item_spec_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen3.5-0.8B"
VLLM_ROOT = ROOT / "projects" / "vllm"
TRITON_ROOT = ROOT / "projects" / "triton"
PYTORCH_ROOT = ROOT / "projects" / "pytorch"
SOURCE_LOCK_PATH = ROOT / "SOURCE_LOCK.json"
MODEL_MANIFEST_PATH = MODEL_DIR / "manifest.json"


# These are the source files on the text-only Qwen3.5 path.  Keeping the list
# explicit is intentional: adding a new dependency requires an auditable
# manifest change instead of silently widening the closure.
SOURCE_FILES = {
    "qwen_model": "vllm/model_executor/models/qwen3_5.py",
    "qwen_next": "vllm/model_executor/models/qwen3_next.py",
    "qwen_mlp": "vllm/model_executor/models/qwen2_moe.py",
    "linear": "vllm/model_executor/layers/linear.py",
    "embedding": "vllm/model_executor/layers/vocab_parallel_embedding.py",
    "layernorm": "vllm/model_executor/layers/layernorm.py",
    "qk_rope": "vllm/model_executor/layers/fused_qk_norm_rope.py",
    "rotary": "vllm/model_executor/layers/rotary_embedding/common.py",
    "rotary_base": "vllm/model_executor/layers/rotary_embedding/base.py",
    "mrope": "vllm/model_executor/layers/rotary_embedding/mrope.py",
    "attention": "vllm/model_executor/layers/attention/attention.py",
    "gdn": "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    "conv": "vllm/model_executor/layers/mamba/ops/causal_conv1d.py",
    "fla_post_conv": "vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py",
    "fla_chunk": "vllm/third_party/flash_linear_attention/ops/chunk.py",
    "fla_cumsum": "vllm/third_party/flash_linear_attention/ops/cumsum.py",
    "fla_kkt": "vllm/third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py",
    "fla_solve": "vllm/third_party/flash_linear_attention/ops/solve_tril.py",
    "fla_wy": "vllm/third_party/flash_linear_attention/ops/wy_fast.py",
    "fla_h": "vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py",
    "fla_o": "vllm/third_party/flash_linear_attention/ops/chunk_o.py",
    "fla_recurrent": "vllm/third_party/flash_linear_attention/ops/fused_recurrent.py",
    "fla_sigmoid": "vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py",
    "fla_norm_gate": "vllm/third_party/flash_linear_attention/ops/fused_norm_gate.py",
    "fla_l2norm": "vllm/third_party/flash_linear_attention/ops/l2norm.py",
    "activation": "vllm/model_executor/layers/activation.py",
    "registry": "vllm/model_executor/models/registry.py",
    "platform": "vllm/platforms/interface.py",
    "rocm_platform": "vllm/platforms/rocm.py",
    "gdn_backend": "vllm/v1/attention/backends/gdn_attn.py",
    "attn_backend_registry": "vllm/v1/attention/backends/registry.py",
    "rocm_attn": "vllm/v1/attention/backends/rocm_attn.py",
    "triton_attn_backend": "vllm/v1/attention/backends/triton_attn.py",
    "rocm_paged": "vllm/v1/attention/ops/chunked_prefill_paged_decode.py",
    "rocm_prefix": "vllm/v1/attention/ops/prefix_prefill.py",
    "triton_prefill": "vllm/v1/attention/ops/triton_prefill_attention.py",
    "triton_cache": "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py",
    "triton_unified": "vllm/v1/attention/ops/triton_unified_attention.py",
    "triton_helpers": "vllm/v1/attention/ops/triton_attention_helpers.py",
}


# Each contract describes one independently testable model operation.  The
# source scanner below fills in line numbers and presence; this table only
# defines the intended closure and its platform boundary.
CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "id": "embedding.lookup",
        "stage": "input",
        "description": "Vocabulary lookup; tied LM head for this checkpoint.",
        "source_keys": ("qwen_model", "qwen_next", "embedding"),
        "required_symbols": ("Qwen3_5ForCausalLM", "Qwen3_5Model", "VocabParallelEmbedding"),
        "registrations": ("vocab_parallel_embedding",),
        "triton_symbols": (),
        "backend_class": "pluggable_layer_and_gemm",
        "amd_lowering": "backend_required",
        "execution_gate": "amd_device_tensor_and_gemm",
    },
    {
        "id": "decoder.rms_norm",
        "stage": "decoder",
        "description": "Gemma RMSNorm before attention, after attention, and final norm.",
        "source_keys": ("qwen_model", "qwen_next", "layernorm"),
        "required_symbols": ("Qwen3_5RMSNorm", "GemmaRMSNorm"),
        "registrations": ("gemma_rms_norm",),
        "triton_symbols": (),
        "backend_class": "custom_op",
        "amd_lowering": "backend_required",
        "execution_gate": "amd_rms_norm",
    },
    {
        "id": "gdn.input_projection",
        "stage": "linear_attention",
        "description": "QKVZ and BA projections, followed by Qwen3.5 non-interleaved split.",
        "source_keys": ("qwen_model", "gdn", "linear"),
        "required_symbols": ("QwenGatedDeltaNetAttention", "create_qkvz_proj", "create_ba_proj", "MergedColumnParallelLinear"),
        "registrations": ("qwen_gated_delta_net_attention",),
        "triton_symbols": (),
        "backend_class": "column_parallel_gemm",
        "amd_lowering": "backend_required",
        "execution_gate": "amd_gemm",
    },
    {
        "id": "gdn.conv.prefill",
        "stage": "linear_attention.prefill",
        "description": "Causal width-4 convolution over packed QKV.",
        "source_keys": ("gdn", "conv"),
        "required_symbols": ("causal_conv1d_fn", "_causal_conv1d_fwd_kernel"),
        "registrations": (),
        "triton_symbols": ("_causal_conv1d_fwd_kernel",),
        "backend_class": "triton_kernel",
        "amd_lowering": "generic_triton_candidate",
        "execution_gate": "amd_triton_compile_and_run",
    },
    {
        "id": "gdn.conv.decode",
        "stage": "linear_attention.decode",
        "description": "Single-token causal convolution state update.",
        "source_keys": ("gdn", "conv"),
        "required_symbols": ("causal_conv1d_update", "_causal_conv1d_update_kernel"),
        "registrations": (),
        "triton_symbols": ("_causal_conv1d_update_kernel",),
        "backend_class": "triton_kernel",
        "amd_lowering": "generic_triton_candidate",
        "execution_gate": "amd_triton_compile_and_run",
    },
    {
        "id": "gdn.post_conv_prep",
        "stage": "linear_attention.prefill",
        "description": "Fused Q/K normalization, V layout, decay and beta preparation.",
        "source_keys": ("gdn", "fla_post_conv"),
        "required_symbols": ("fused_post_conv_prep", "_fused_post_conv_kernel"),
        "registrations": (),
        "triton_symbols": ("_fused_post_conv_kernel",),
        "backend_class": "triton_kernel",
        "amd_lowering": "generic_triton_candidate",
        "execution_gate": "amd_triton_compile_and_run",
    },
    {
        "id": "gdn.chunk_prefill",
        "stage": "linear_attention.prefill",
        "description": "Chunked gated-delta rule and recurrent-state update.",
        "source_keys": ("gdn", "fla_chunk", "fla_cumsum", "fla_kkt", "fla_solve", "fla_wy", "fla_h", "fla_o"),
        "required_symbols": (
            "ChunkGatedDeltaRule",
            "chunk_gated_delta_rule",
            "chunk_gated_delta_rule_fwd",
            "chunk_local_cumsum",
            "chunk_scaled_dot_kkt_fwd",
            "solve_tril",
            "recompute_w_u_fwd",
            "chunk_gated_delta_rule_fwd_h",
            "chunk_fwd_o",
        ),
        "registrations": ("chunk_gated_delta_rule",),
        "triton_symbols": (
            "chunk_local_cumsum_scalar_kernel",
            "chunk_local_cumsum_vector_kernel",
            "chunk_scaled_dot_kkt_fwd_kernel",
            "solve_tril_16x16_kernel",
            "merge_16x16_to_32x32_inverse_kernel",
            "merge_16x16_to_64x64_inverse_kernel",
            "recompute_w_u_fwd_kernel",
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            "chunk_fwd_kernel_o",
        ),
        "backend_class": "triton_kernel_family",
        "amd_lowering": "generic_triton_candidate_with_arch_review",
        "execution_gate": "amd_triton_compile_and_run",
    },
    {
        "id": "gdn.recurrent_decode",
        "stage": "linear_attention.decode",
        "description": "Packed recurrent gated-delta update with persistent state.",
        "source_keys": ("gdn", "fla_recurrent", "fla_sigmoid"),
        "required_symbols": ("fused_recurrent_gated_delta_rule_packed_decode", "fused_sigmoid_gating_delta_rule_update"),
        "registrations": ("qwen_gdn_attention_core",),
        "triton_symbols": (
            "fused_recurrent_gated_delta_rule_packed_decode_kernel",
            "fused_sigmoid_gating_delta_rule_update_kernel",
        ),
        "backend_class": "triton_kernel_family",
        "amd_lowering": "generic_triton_candidate",
        "execution_gate": "amd_triton_compile_and_run",
    },
    {
        "id": "gdn.auxiliary_triton_variants",
        "stage": "linear_attention.prefill_decode_variants",
        "description": "Optional recurrent/gating and standalone L2-normalization kernels exposed by the GDN implementation.",
        "source_keys": ("gdn", "fla_recurrent", "fla_l2norm"),
        "required_symbols": ("fused_gdn_gating", "l2norm_fwd"),
        "registrations": (),
        "triton_symbols": (
            "fused_gdn_gating_kernel",
            "fused_recurrent_gated_delta_rule_fwd_kernel",
            "l2norm_fwd_kernel1",
            "l2norm_fwd_kernel",
            "l2norm_fwd_kernel2",
        ),
        "backend_class": "triton_kernel_family",
        "amd_lowering": "generic_triton_candidate_with_arch_review",
        "execution_gate": "amd_triton_compile_and_run",
    },
    {
        "id": "gdn.output_norm_gate",
        "stage": "linear_attention",
        "description": "RMSNormGated over recurrent output and z gate.",
        "source_keys": ("gdn", "layernorm", "fla_norm_gate"),
        "required_symbols": ("RMSNormGated", "rms_norm_gated", "layer_norm_gated_fwd_kernel"),
        "registrations": ("rms_norm_gated",),
        "triton_symbols": ("layer_norm_gated_fwd_kernel", "layer_norm_gated_fwd_kernel1"),
        "backend_class": "triton_kernel_or_native",
        "amd_lowering": "generic_triton_candidate_with_arch_review",
        "execution_gate": "amd_triton_compile_and_run",
    },
    {
        "id": "full_attention.qkv_qk_norm_rope",
        "stage": "full_attention",
        "description": "QKV projection, per-head Q/K RMSNorm, partial RoPE and output gate split.",
        "source_keys": ("qwen_next", "qk_rope", "rotary", "rotary_base", "mrope", "layernorm"),
        "required_symbols": (
            "Qwen3NextAttention",
            "_project_qkv_gate",
            "_fused_qk_rmsnorm_rope_gate_kernel",
            "ApplyRotaryEmb",
            "MRotaryEmbedding",
            "RotaryEmbeddingBase",
        ),
        "registrations": ("apply_rotary_emb",),
        "triton_symbols": ("_fused_qk_rmsnorm_rope_gate_kernel",),
        # Qwen3.5 text inputs use 1-D positions, so MRotaryEmbedding's
        # 3-D/vision kernel is an alternate branch rather than a text-path
        # execution requirement.  Keep it visible without pretending it ran.
        "alternate_triton_symbols": ("_triton_mrope_forward",),
        "alternate_path": "vision_or_3d_positions_deferred",
        "backend_class": "triton_kernel_or_native",
        "amd_lowering": "cuda_guarded_fused_path; hip_native_or_flash_attn_path",
        "execution_gate": "amd_qk_norm_rope_without_cpu_fallback",
    },
    {
        "id": "full_attention.kv_cache_attention",
        "stage": "full_attention",
        "description": "KV-cache update and causal paged attention for six full-attention layers.",
        "source_keys": (
            "qwen_next",
            "attention",
            "rocm_attn",
            "rocm_platform",
            "rocm_paged",
            "rocm_prefix",
            "triton_prefill",
            "triton_cache",
            "triton_attn_backend",
            "triton_unified",
            "triton_helpers",
            "attn_backend_registry",
        ),
        "required_symbols": (
            "Attention",
            "unified_kv_cache_update",
            "unified_attention_with_output",
            "RocmAttentionBackend",
            "TritonAttentionBackend",
            "GDN_ATTN",
            "chunked_prefill_paged_decode",
            "kernel_paged_attention_2d",
            "context_attention_fwd",
            "reshape_and_cache_kernel_flash",
            "triton_reshape_and_cache_flash",
            "triton_reshape_and_cache_flash_per_token_head_quant",
            "unified_attention",
            "kernel_unified_attention",
            "reduce_segments",
            "has_native_kv_cache_layout",
        ),
        "registrations": ("unified_kv_cache_update", "unified_attention_with_output"),
        "triton_symbols": (
            # ROCmAttentionBackend path.
            "cdiv_fn",
            "kernel_paged_attention_2d",
            "_paged_kv_cache_offsets",
            "_fwd_kernel",
            "_fwd_kernel_alibi",
            "reshape_and_cache_kernel_flash",
            "_reshape_cache_per_token_head",
            "reshape_and_cache_kernel_flash_diffkv",
            # TritonAttentionBackend fallback path.
            "_cast_kv_tile",
            "_load_q_td",
            "_load_kv_tile_td",
            "_store_output_td",
            "kernel_unified_attention",
            "reduce_segments",
            "apply_alibi_to_score",
            "apply_softcap",
            "compute_kv_seq_mask",
            "compute_tile_loop_bounds",
            "find_seq_idx",
            "init_softmax_M",
            "load_qq_bias_tile",
            "resolve_seq_and_query_len",
            "softmax_step",
            "store_segm_reduce_scalars",
        ),
        "alternate_path": "ROCM_ATTN_default; TRITON_ATTN_fallback; AITER_external_optional",
        "backend_class": "attention_backend",
        "amd_lowering": "rocm_backend_or_triton_attention_required",
        "execution_gate": "amd_kv_cache_and_attention",
    },
    {
        "id": "full_attention.output_gate_projection",
        "stage": "full_attention",
        "description": "Sigmoid output gate followed by row-parallel projection.",
        "source_keys": ("qwen_next", "linear"),
        "required_symbols": ("attn_output_gate", "RowParallelLinear"),
        "registrations": ("row_parallel_linear",),
        "triton_symbols": (),
        "backend_class": "elementwise_and_gemm",
        "amd_lowering": "backend_required",
        "execution_gate": "amd_elementwise_and_gemm",
    },
    {
        "id": "mlp.gate_up_silu_down",
        "stage": "decoder",
        "description": "Dense gate/up projection, SiLU-and-mul, and down projection.",
        "source_keys": ("qwen_model", "qwen_next", "qwen_mlp", "activation", "linear"),
        "required_symbols": ("Qwen3NextMLP", "Qwen2MoeMLP", "SiluAndMul", "gate_up_proj", "down_proj"),
        "registrations": ("row_parallel_linear",),
        "triton_symbols": (),
        # The checkpoint selects hidden_act=silu, whose vLLM implementation is
        # the registered C++ ``silu_and_mul`` custom op.  This Triton kernel is
        # retained only for the alternate SwigluStep activation family.
        "alternate_triton_symbols": ("_swiglustep_and_mul_kernel",),
        "alternate_path": "SwigluStepAndMul_only; Qwen3.5_dense_uses_SiluAndMul",
        "backend_class": "gemm_and_elementwise",
        "amd_lowering": "backend_required",
        "execution_gate": "amd_gemm_and_activation",
    },
    {
        "id": "lm_head.logits",
        "stage": "output",
        "description": "Final tied LM head and logits processor; sampling is outside this contract.",
        "source_keys": ("qwen_model", "qwen_next"),
        "required_symbols": ("compute_logits", "LogitsProcessor", "ParallelLMHead"),
        "registrations": (),
        "triton_symbols": (),
        "backend_class": "vocab_parallel_gemm_and_logits",
        "amd_lowering": "backend_required",
        "execution_gate": "amd_logits",
    },
)


def _read(relative: str) -> str:
    path = VLLM_ROOT / relative
    return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty(repo: Path) -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _line_for(text: str, needle: str) -> int | None:
    match = re.search(needle, text, flags=re.MULTILINE)
    if match is None:
        return None
    return text.count("\n", 0, match.start()) + 1


def _source_inventory(text: str) -> dict[str, Any]:
    """Extract symbols and registrations without importing vLLM."""
    tree = ast.parse(text)
    classes: dict[str, int] = {}
    functions: dict[str, int] = {}
    imports: dict[str, int] = {}
    references: dict[str, int] = {}
    decorators: dict[str, list[dict[str, Any]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes[node.name] = node.lineno
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node.lineno
            for decorator in node.decorator_list:
                rendered = ast.unparse(decorator)
                if "triton." in rendered:
                    decorators.setdefault("triton", []).append(
                        {"name": node.name, "line": node.lineno, "decorator": rendered}
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[-1]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports[alias.asname or alias.name] = node.lineno
        if isinstance(node, ast.Name):
            references.setdefault(node.id, getattr(node, "lineno", 1))
        elif isinstance(node, ast.Attribute):
            references.setdefault(node.attr, getattr(node, "lineno", 1))
    registrations: list[dict[str, Any]] = []
    patterns = (
        r"@(?:CustomOp|PluggableLayer)\.register\(\s*[\"']([^\"']+)[\"']",
        r"op_name\s*=\s*[\"']([^\"']+)[\"']",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            registrations.append(
                {
                    "name": match.group(1),
                    "line": text.count("\n", 0, match.start()) + 1,
                    "pattern": pattern,
                }
            )
    torch_ops = sorted(set(re.findall(r"torch\.ops\.vllm\.([A-Za-z0-9_]+)", text)))
    return {
        "classes": classes,
        "functions": functions,
        "imports": imports,
        "references": references,
        "triton": decorators.get("triton", []),
        "registrations": sorted(registrations, key=lambda item: (item["name"], item["line"])),
        "torch_ops_vllm": torch_ops,
    }


def _symbol_present(inventory: dict[str, Any], symbol: str) -> tuple[bool, str | None, int | None]:
    kind_names = {
        "classes": "class",
        "functions": "function",
        "imports": "import",
        "references": "reference",
    }
    for kind in ("classes", "functions", "imports", "references"):
        if symbol in inventory.get(kind, {}):
            return True, kind_names[kind], inventory[kind][symbol]
    return False, None, None


def _inventory_counts(inventories: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize the statically discovered operator surface.

    Counts are intentionally source-qualified for Triton symbols: names such
    as ``_fwd_kernel`` occur in several independent modules and must not be
    accidentally collapsed into one operator.
    """
    triton_occurrences = 0
    triton_symbols: set[tuple[str, str]] = set()
    registration_names: set[str] = set()
    torch_ops: set[str] = set()
    classes = functions = imports = references = 0
    for key, inventory in inventories.items():
        classes += len(inventory.get("classes", {}))
        functions += len(inventory.get("functions", {}))
        imports += len(inventory.get("imports", {}))
        references += len(inventory.get("references", {}))
        for item in inventory.get("triton", []):
            triton_occurrences += 1
            triton_symbols.add((key, item["name"]))
        registration_names.update(item["name"] for item in inventory.get("registrations", []))
        torch_ops.update(inventory.get("torch_ops_vllm", []))
    return {
        "source_file_count": len(inventories),
        "class_symbol_count": classes,
        "function_symbol_count": functions,
        "import_symbol_count": imports,
        "reference_symbol_count": references,
        "triton_decorator_occurrence_count": triton_occurrences,
        "triton_source_symbol_count": len(triton_symbols),
        "custom_registration_count": len(registration_names),
        "torch_ops_vllm_count": len(torch_ops),
        "custom_registration_names": sorted(registration_names),
        "torch_ops_vllm_names": sorted(torch_ops),
    }


def _evidence_for_symbols(
    inventories: dict[str, dict[str, Any]],
    source_keys: tuple[str, ...],
    symbols: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return source/line evidence for a backend decision."""
    evidence: list[dict[str, Any]] = []
    for key in source_keys:
        inventory = inventories.get(key, {})
        relative = SOURCE_FILES[key]
        for symbol in symbols:
            present, kind, line = _symbol_present(inventory, symbol)
            if present:
                evidence.append(
                    {"source": relative, "symbol": symbol, "kind": kind, "line": line}
                )
    return sorted(
        evidence,
        key=lambda item: (item["source"], item.get("line", 0), item["symbol"]),
    )


def _backend_support(inventories: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Describe backend boundaries without importing vLLM or executing code.

    ``unsupported`` is deliberately explicit.  A CPU implementation or a
    CUDA device can be useful for debugging, but neither is allowed to satisfy
    the AMD operator gate.  Optional AITER/FlashInfer/CuteDSL paths are also
    kept separate because they are external or CUDA-only dependencies.
    """
    target = {
        "backend": "gemsim_amd",
        "status": "work_queue_incomplete",
        "counts_as_pass": False,
        "reason": (
            "Only accepted per-item gemsim_amd evidence can complete a contract; "
            "ambient physical accelerator availability is not part of this spec."
        ),
        "source_evidence": _evidence_for_symbols(
            inventories,
            ("gdn", "conv", "fla_chunk", "triton_unified", "triton_cache"),
            ("_causal_conv1d_fwd_kernel", "chunk_gated_delta_rule_fwd_kernel", "kernel_unified_attention"),
        ),
    }
    unsupported = [
        {
            "backend": "cpu",
            "status": "forbidden_for_gate",
            "counts_as_pass": False,
            "reason": "CPU GDN/custom-op branches are diagnostic fallbacks and are explicitly out of scope.",
            "source_evidence": _evidence_for_symbols(
                inventories, ("gdn", "activation"), ("forward_cpu", "cpu_gdn_attention_core")
            ),
        },
        {
            "backend": "cuda_nvidia",
            "status": "forbidden_for_amd_gate",
            "counts_as_pass": False,
            "reason": "A CUDA build/device is not HIP; NVIDIA execution cannot be promoted to an AMD result.",
            "source_evidence": _evidence_for_symbols(
                inventories,
                ("qwen_next", "qk_rope", "gdn"),
                ("is_cuda", "forward_cuda", "use_fused_qk_norm_rope_gate"),
            ),
        },
        {
            "backend": "gdn_flashinfer",
            "status": "unsupported_on_rocm",
            "counts_as_pass": False,
            "reason": "The local resolver selects FlashInfer only on CUDA-capability branches; it is not an AMD Triton closure.",
            "source_evidence": _evidence_for_symbols(
                inventories,
                ("gdn",),
                ("_resolve_gdn_prefill_backend", "fi_chunk_gated_delta_rule"),
            ),
        },
        {
            "backend": "gdn_cutedsl",
            "status": "unsupported_on_rocm",
            "counts_as_pass": False,
            "reason": "CuteDSL is an opt-in CUDA/Blackwell branch in the checked-out resolver, not a generic AMD Triton backend.",
            "source_evidence": _evidence_for_symbols(
                inventories, ("gdn",), ("forward_cutedsl", "_resolve_gdn_prefill_backend")
            ),
        },
    ]
    external = [
        {
            "backend": "rocm_aiter_triton",
            "status": "external_dependency_unverified",
            "counts_as_pass": False,
            "reason": "AITER GDN/attention kernels are imported conditionally from the external aiter package; this manifest only audits vLLM source.",
            "source_evidence": _evidence_for_symbols(
                inventories,
                ("gdn", "rocm_attn", "rocm_platform"),
                ("rocm_aiter_ops", "GDN_AITER_TRITON_AVAILABLE", "ROCM_AITER_UNIFIED_ATTN"),
            ),
        },
        {
            "backend": "flash_attn_rotary_external",
            "status": "external_dependency_unverified",
            "counts_as_pass": False,
            "reason": "ROCm ApplyRotaryEmb may import flash_attn.ops.triton.rotary; the package is not part of this repository lock.",
            "source_evidence": _evidence_for_symbols(
                inventories, ("rotary",), ("apply_rotary_emb_flash_attn", "forward_hip")
            ),
        },
    ]
    return {
        "target": target,
        "unsupported": unsupported,
        "external_unverified": external,
        "unsupported_backend_names": [item["backend"] for item in unsupported],
    }


def _tensor(
    name: str,
    role: str,
    dtype: str,
    shape: list[int],
    strides: list[int],
    access: str,
    *,
    storage_elements: int | None = None,
) -> dict[str, Any]:
    elements = 1
    for extent in shape:
        elements *= extent
    return {
        "name": name,
        "role": role,
        "dtype": dtype,
        "shape": shape,
        "strides": strides,
        "stride_unit": "elements",
        "storage_offset": 0,
        "storage_elements": storage_elements if storage_elements is not None else elements,
        "access": access,
        "alias_group": name,
    }


def _fallback_policy() -> dict[str, Any]:
    return {
        "allowed": False,
        "forbidden_backends": [
            "cpu",
            "cuda_nvidia",
            "gdn_flashinfer",
            "gdn_cutedsl",
            "production_rocm",
        ],
        "required_zero_counters": [
            "fallback_count",
            "cpu_fallback_count",
            "nvidia_fallback_count",
        ],
        "forbidden_dsos": [
            "/opt/rocm",
            "libcuda.so",
            "libamdhip64.so",
            "libhsa-runtime64.so",
        ],
        "forbidden_device_nodes": ["/dev/kfd", "/dev/dri"],
    }


def _source_entries(
    contract: dict[str, Any], source_meta: dict[str, Any]
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for evidence in contract["source_evidence"]:
        path = evidence["source"]
        key = (path, evidence.get("symbol"), evidence.get("registration"))
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "repo": "vllm",
                "path": path,
                "symbol": evidence.get("symbol"),
                "registration": evidence.get("registration"),
                "line": evidence.get("line"),
                "file_sha256": source_meta[path]["sha256"],
            }
        )
    return entries


def _root_source(path: str, symbol: str) -> dict[str, Any]:
    absolute = ROOT / path
    return {
        "repo": "root",
        "path": path,
        "symbol": symbol,
        "registration": None,
        "line": None,
        "file_sha256": _sha256(absolute),
    }


def _repository_source(repo: str, path: str, symbol: str) -> dict[str, Any]:
    roots = {
        "vllm": VLLM_ROOT,
        "triton": TRITON_ROOT,
        "pytorch": PYTORCH_ROOT,
    }
    absolute = roots[repo] / path
    return {
        "repo": repo,
        "path": path,
        "symbol": symbol,
        "registration": None,
        "line": None,
        "file_sha256": _sha256(absolute),
    }


def _common_provenance(
    source_revisions: dict[str, Any], model_manifest: dict[str, Any]
) -> dict[str, Any]:
    weights = model_manifest["files"]["model.safetensors-00001-of-00001.safetensors"]
    return {
        "model_revision": model_manifest["revision"],
        "config_sha256": model_manifest["files"]["config.json"]["sha256"],
        "model_manifest_sha256": _sha256(MODEL_MANIFEST_PATH),
        "weight_bytes": weights["bytes"],
        "weight_sha256": weights["sha256"],
        "source_lock_sha256": _sha256(SOURCE_LOCK_PATH),
        "required_repo_commits": {
            name: source_revisions[name]["head"]
            for name in ("vllm", "triton", "pytorch")
        },
    }


def _empty_execution_contract() -> dict[str, Any]:
    return {
        "tensors": [],
        "mutation": {
            "allowed": [],
            "must_remain_bitwise_equal": [],
            "must_be_disjoint": [],
        },
        "state": {
            "kind": "unconfigured",
            "transition_count": 0,
            "slots": [],
            "initial_state": None,
            "expected_transition": None,
        },
        "oracle": {
            "reference": "",
            "accumulation_dtype": "",
            "comparisons": [],
            "invariants": [],
        },
        "kernel": {
            "entrypoint": "",
            "expected_symbols": [],
            "target": {
                "backend": "gemsim_amd",
                "arch": "gfx950",
                "wavefront_size": 64,
            },
            "launch": {"grid": [], "workgroup": [], "num_warps": None},
            "code_objects": {"count": 0, "identity_policy": "recorded_sha256"},
        },
    }


def _make_unconfigured_item(
    contract: dict[str, Any],
    item_id: str,
    phase: str,
    variant: str,
    reason: str,
    source_meta: dict[str, Any],
    provenance: dict[str, Any],
    *,
    dependencies: tuple[str, ...] = (),
) -> dict[str, Any]:
    item = {
        "id": item_id,
        "contract_id": contract["id"],
        "phase": phase,
        "variant": variant,
        "acceptance_role": "required",
        "required_for_contract": True,
        "configuration_status": "unconfigured",
        "configuration_errors": [reason],
        **_empty_execution_contract(),
        "dependencies": {
            "all_of_work_items": list(dependencies),
            "capabilities": ["normal_python", "gemsim_amd", "gfx950"],
            "state_inputs": [],
        },
        "source": {
            "entrypoints": _source_entries(contract, source_meta),
            "runner": None,
        },
        "provenance": provenance,
        "runtime_evidence": runtime_evidence_policy(),
        "fallback": _fallback_policy(),
    }
    item["spec_sha256"] = work_item_spec_sha256(item)
    item["status"] = derive_work_item_status(item)
    return item


def _make_rms_norm_item(
    contract: dict[str, Any],
    rows: int,
    phase: str,
    source_meta: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    item = {
        "id": f"decoder.rms_norm.{phase}.no_residual.bf16.c.v1",
        "contract_id": contract["id"],
        "phase": phase,
        "variant": "no_residual_gemma_weight_plus_one",
        "acceptance_role": "required",
        "required_for_contract": True,
        "configuration_status": "configured",
        "configuration_errors": [],
        "tensors": [
            _tensor("x", "input", "bfloat16", [rows, 1024], [1024, 1], "read_only"),
            _tensor("raw_weight", "parameter", "bfloat16", [1024], [1], "read_only"),
            _tensor(
                "out",
                "output",
                "bfloat16",
                [rows, 1024],
                [1024, 1],
                "write_only",
                storage_elements=(rows + 1) * 1024,
            ),
        ],
        "mutation": {
            "allowed": [{"tensor": "out", "region": "all", "kind": "write"}],
            "must_remain_bitwise_equal": ["x", "raw_weight"],
            "must_be_disjoint": ["x", "raw_weight", "out"],
        },
        "state": {
            "kind": "stateless",
            "transition_count": 0,
            "slots": [],
            "initial_state": None,
            "expected_transition": None,
        },
        "oracle": {
            "reference": (
                "float32(x) * rsqrt(mean(float32(x)^2)+1e-6) * "
                "(1+float32(raw_weight)), rounded to bfloat16"
            ),
            "accumulation_dtype": "float32",
            "comparisons": [
                {
                    "actual": "out",
                    "expected": "reference",
                    "mode": "atol_rtol",
                    "atol": 0.015625,
                    "rtol": 0.02,
                    "max_mismatches": 0,
                    "finite_required": True,
                    "equal_nan": False,
                }
            ],
            "invariants": [
                {"name": "relative_l2_error", "operator": "lt", "value": 0.005},
                {"name": "guard_unchanged", "operator": "eq", "value": True},
            ],
        },
        "dependencies": {
            "all_of_work_items": [],
            "capabilities": ["normal_python", "triton_jit", "gemsim_amd", "gfx950"],
            "state_inputs": [],
        },
        "kernel": {
            "entrypoint": (
                "examples/triton/rms_norm_correctness.py:"
                "qwen35_plain_gemma_rms_norm_kernel"
            ),
            "expected_symbols": ["qwen35_plain_gemma_rms_norm_kernel"],
            "target": {"backend": "gemsim_amd", "arch": "gfx950", "wavefront_size": 64},
            "launch": {
                "grid": [rows, 1, 1],
                "workgroup": [512, 1, 1],
                "num_warps": 8,
            },
            "code_objects": {
                "count": 1,
                "identity_policy": "recorded_sha256",
            },
        },
        "source": {
            "entrypoints": _source_entries(contract, source_meta)
            + [
                _repository_source(
                    "vllm",
                    "vllm/models/minimax_m3/amd/ops/gemma_rmsnorm.py",
                    "_gemma_rmsnorm_kernel",
                ),
                _root_source(
                    "examples/triton/rms_norm_correctness.py",
                    "qwen35_plain_gemma_rms_norm_kernel",
                )
            ],
            "runner": {
                "path": "examples/triton/rms_norm_correctness.py",
                "sha256": _sha256(ROOT / "examples/triton/rms_norm_correctness.py"),
            },
        },
        "provenance": provenance,
        "runtime_evidence": runtime_evidence_policy(),
        "fallback": _fallback_policy(),
    }
    item["spec_sha256"] = work_item_spec_sha256(item)
    item["status"] = derive_work_item_status(item)
    return item


def _build_work_queue(
    contracts: list[dict[str, Any]],
    source_meta: dict[str, Any],
    source_revisions: dict[str, Any],
    model_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id = {contract["id"]: contract for contract in contracts}
    provenance = _common_provenance(source_revisions, model_manifest)
    item_specs: dict[str, list[tuple[str, str, str]]] = {
        "embedding.lookup": [
            ("embedding.lookup.decode.bf16.c.v1", "decode", "decode_lookup"),
            ("embedding.lookup.prefill.bf16.c.v1", "prefill", "prefill_lookup"),
        ],
        "gdn.input_projection": [
            ("gdn.input_projection.qkvz_decode.bf16.c.v1", "decode", "qkvz_decode"),
            ("gdn.input_projection.qkvz_prefill.bf16.c.v1", "prefill", "qkvz_prefill"),
            ("gdn.input_projection.ba_decode.bf16.c.v1", "decode", "ba_decode"),
            ("gdn.input_projection.ba_prefill.bf16.c.v1", "prefill", "ba_prefill"),
        ],
        "gdn.conv.prefill": [("gdn.conv.prefill.bf16.c.v1", "prefill", "width4")],
        "gdn.conv.decode": [("gdn.conv.decode.bf16.state.v1", "decode", "width4_state")],
        "gdn.post_conv_prep": [("gdn.post_conv_prep.prefill.bf16.c.v1", "prefill", "fused")],
        "gdn.chunk_prefill": [("gdn.chunk_prefill.bf16.state.v1", "prefill", "kernel_family")],
        "gdn.recurrent_decode": [("gdn.recurrent_decode.bf16.state.v1", "decode", "kernel_family")],
        "gdn.auxiliary_triton_variants": [
            ("gdn.auxiliary_triton_variants.applicability.v1", "prefill_decode", "resolve_model_path")
        ],
        "gdn.output_norm_gate": [
            ("gdn.output_norm_gate.decode.bf16.c.v1", "decode", "decode"),
            ("gdn.output_norm_gate.prefill.bf16.c.v1", "prefill", "prefill"),
        ],
        "full_attention.qkv_qk_norm_rope": [
            ("full_attention.qkv_qk_norm_rope.decode.bf16.c.v1", "decode", "text_1d_positions"),
            ("full_attention.qkv_qk_norm_rope.prefill.bf16.c.v1", "prefill", "text_1d_positions"),
        ],
        "full_attention.kv_cache_attention": [
            ("full_attention.kv_cache_update.bf16.state.v1", "prefill_decode", "cache_update"),
            ("full_attention.attention.prefill.bf16.state.v1", "prefill", "causal_prefill"),
            ("full_attention.attention.decode.bf16.state.v1", "decode", "paged_decode"),
        ],
        "full_attention.output_gate_projection": [
            ("full_attention.output_gate_projection.decode.bf16.c.v1", "decode", "decode"),
            ("full_attention.output_gate_projection.prefill.bf16.c.v1", "prefill", "prefill"),
        ],
        "mlp.gate_up_silu_down": [
            ("mlp.gate_up_projection.decode.bf16.c.v1", "decode", "gate_up_projection"),
            ("mlp.gate_up_projection.prefill.bf16.c.v1", "prefill", "gate_up_projection"),
            ("mlp.silu_and_mul.integration.decode.bf16.c.v1", "decode", "model_integration"),
            ("mlp.silu_and_mul.integration.prefill.bf16.c.v1", "prefill", "model_integration"),
            ("mlp.down_projection.decode.bf16.c.v1", "decode", "down_projection"),
            ("mlp.down_projection.prefill.bf16.c.v1", "prefill", "down_projection"),
        ],
        "lm_head.logits": [("lm_head.logits.decode.bf16.c.v1", "output", "selected_token")],
    }
    items: list[dict[str, Any]] = []
    for contract_id, specs in item_specs.items():
        contract = by_id[contract_id]
        reason = (
            "exact dtype/shape/stride/mutation/state/oracle/kernel contract "
            "has not yet been captured from the pinned text graph"
        )
        if contract_id == "gdn.auxiliary_triton_variants":
            reason = (
                "model applicability is unresolved; optional variants cannot count "
                "until a pinned text-graph selection guard names the required path"
            )
        for item_id, phase, variant in specs:
            items.append(
                _make_unconfigured_item(
                    contract,
                    item_id,
                    phase,
                    variant,
                    reason,
                    source_meta,
                    provenance,
                )
            )

    rms = by_id["decoder.rms_norm"]
    items.extend(
        [
            _make_rms_norm_item(rms, 1, "decode", source_meta, provenance),
            _make_rms_norm_item(rms, 7, "prefill", source_meta, provenance),
            _make_unconfigured_item(
                rms,
                "decoder.rms_norm.decode.fused_residual.bf16.c.v1",
                "decode",
                "fused_residual",
                "fused residual output and mutation semantics require a separate exact runner",
                source_meta,
                provenance,
            ),
            _make_unconfigured_item(
                rms,
                "decoder.rms_norm.prefill.fused_residual.bf16.c.v1",
                "prefill",
                "fused_residual",
                "fused residual output and mutation semantics require a separate exact runner",
                source_meta,
                provenance,
            ),
        ]
    )
    return sorted(items, key=lambda item: item["id"])


def build_manifest() -> dict[str, Any]:
    config_path = MODEL_DIR / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", {})
    layer_types = list(text_config.get("layer_types", []))
    counts = Counter(layer_types)

    inventories: dict[str, dict[str, Any]] = {}
    source_meta: dict[str, Any] = {}
    missing_files: list[str] = []
    for key, relative in SOURCE_FILES.items():
        path = VLLM_ROOT / relative
        if not path.is_file():
            missing_files.append(relative)
            continue
        text = path.read_text(encoding="utf-8")
        inventories[key] = _source_inventory(text)
        source_meta[relative] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "triton_decorated_symbols": sorted(
                {item["name"] for item in inventories[key]["triton"]}
            ),
            "triton_decorator_occurrence_count": len(inventories[key]["triton"]),
            "custom_registration_names": sorted(
                {item["name"] for item in inventories[key]["registrations"]}
            ),
            "torch_ops_vllm_names": sorted(inventories[key]["torch_ops_vllm"]),
        }

    contracts: list[dict[str, Any]] = []
    for contract in CONTRACTS:
        source_evidence: list[dict[str, Any]] = []
        missing_symbols: list[str] = []
        missing_registrations: list[str] = []
        missing_triton_symbols: list[str] = []
        alternate_triton_symbols = tuple(contract.get("alternate_triton_symbols", ()))
        alternate_source_evidence: list[dict[str, Any]] = []
        missing_alternate_triton_symbols: list[str] = []
        for key in contract["source_keys"]:
            inventory = inventories.get(key)
            relative = SOURCE_FILES[key]
            if inventory is None:
                continue
            for symbol in contract["required_symbols"]:
                present, kind, line = _symbol_present(inventory, symbol)
                if present:
                    source_evidence.append(
                        {"source": relative, "symbol": symbol, "kind": kind, "line": line}
                    )
            for symbol in contract["triton_symbols"]:
                present, kind, line = _symbol_present(inventory, symbol)
                if present:
                    source_evidence.append(
                        {"source": relative, "symbol": symbol, "kind": kind, "line": line, "triton": True}
                    )
            for symbol in alternate_triton_symbols:
                present, kind, line = _symbol_present(inventory, symbol)
                if present:
                    alternate_source_evidence.append(
                        {
                            "source": relative,
                            "symbol": symbol,
                            "kind": kind,
                            "line": line,
                            "triton": True,
                            "alternate": True,
                        }
                    )
            names = {item["name"] for item in inventory["registrations"]}
            for registration in contract["registrations"]:
                if registration not in names and registration not in inventory["torch_ops_vllm"]:
                    # The registration may live in a different source key in
                    # this contract; defer final missing check until below.
                    continue
                source_evidence.append(
                    {"source": relative, "registration": registration}
                )

        # Required symbols are allowed to be distributed over source_keys.
        for symbol in contract["required_symbols"]:
            if not any(_symbol_present(inventories.get(key, {}), symbol)[0] for key in contract["source_keys"]):
                missing_symbols.append(symbol)
        for symbol in contract["triton_symbols"]:
            if not any(_symbol_present(inventories.get(key, {}), symbol)[0] for key in contract["source_keys"]):
                missing_triton_symbols.append(symbol)
        for symbol in alternate_triton_symbols:
            if not any(_symbol_present(inventories.get(key, {}), symbol)[0] for key in contract["source_keys"]):
                missing_alternate_triton_symbols.append(symbol)
        all_registrations = {
            item["name"]
            for key in contract["source_keys"]
            for item in inventories.get(key, {}).get("registrations", [])
        }
        all_registrations.update(
            op
            for key in contract["source_keys"]
            for op in inventories.get(key, {}).get("torch_ops_vllm", [])
        )
        for registration in contract["registrations"]:
            if registration not in all_registrations:
                missing_registrations.append(registration)
        missing_triton_symbols = sorted(set(missing_triton_symbols))
        static_source_ok = not missing_files and not missing_symbols and not missing_registrations and not missing_triton_symbols
        contracts.append(
            {
                "id": contract["id"],
                "stage": contract["stage"],
                "description": contract["description"],
                "backend_class": contract["backend_class"],
                "amd_lowering": contract["amd_lowering"],
                "execution_gate": contract["execution_gate"],
                "cpu_fallback_forbidden": True,
                "nvidia_runtime_is_not_pass": True,
                "source_keys": list(contract["source_keys"]),
                "required_symbols": list(contract["required_symbols"]),
                "registrations": list(contract["registrations"]),
                "triton_symbols": list(contract["triton_symbols"]),
                "alternate_triton_symbols": list(alternate_triton_symbols),
                "alternate_path": contract.get("alternate_path"),
                "source_evidence": sorted(
                    source_evidence,
                    key=lambda item: (item["source"], item.get("line", 0), item.get("symbol", item.get("registration", ""))),
                ),
                "alternate_source_evidence": sorted(
                    alternate_source_evidence,
                    key=lambda item: (item["source"], item.get("line", 0), item["symbol"]),
                ),
                "static_source_ok": static_source_ok,
                "missing_files": sorted(set(missing_files)),
                "missing_symbols": sorted(set(missing_symbols)),
                "missing_registrations": sorted(set(missing_registrations)),
                "missing_triton_symbols": missing_triton_symbols,
                "missing_alternate_triton_symbols": sorted(
                    set(missing_alternate_triton_symbols)
                ),
                "amd_runtime_status": "not_executed",
            }
        )

    model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    source_lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    locked_sources = {
        source["id"]: source
        for source in source_lock["sources"]
        if source.get("id") in {"vllm", "triton", "pytorch"}
    }
    if set(locked_sources) != {"vllm", "triton", "pytorch"}:
        raise RuntimeError("SOURCE_LOCK.json lacks the required model source locks")
    source_revisions = {
        name: {
            "head": locked_sources[name]["work_head"],
            "tree": locked_sources[name]["work_tree"],
        }
        for name in ("vllm", "triton", "pytorch")
    }
    work_items = _build_work_queue(
        contracts, source_meta, source_revisions, model_manifest
    )
    for contract in contracts:
        contract_id = contract["id"]
        required_ids = sorted(
            item["id"]
            for item in work_items
            if item["contract_id"] == contract_id and item["required_for_contract"]
        )
        contract["completion_rule"] = "all_required_items_accepted"
        contract["required_work_item_ids"] = required_ids
        contract["partial_work_item_ids"] = []
        contract["work_queue_status"] = "not_accepted"
    static_ok = not missing_files and all(item["static_source_ok"] for item in contracts)
    inventory_counts = _inventory_counts(inventories)
    contract_counts = {
        "total": len(contracts),
        "static_source_ok": sum(1 for item in contracts if item["static_source_ok"]),
        "amd_runtime_not_executed": sum(
            1 for item in contracts if item["amd_runtime_status"] == "not_executed"
        ),
        "required_triton_symbol_count": sum(
            len(item["triton_symbols"]) for item in contracts
        ),
        "alternate_triton_symbol_count": sum(
            len(item["alternate_triton_symbols"]) for item in contracts
        ),
    }
    backend_support = _backend_support(inventories)
    manifest = {
        "schema": "amdgpu-sim.qwen35.operator-manifest.v2",
        "purpose": (
            "model-specific text-only executable operator work queue; "
            "no CPU, NVIDIA, production ROCm, or GDN fallback"
        ),
        "acceptance_policy": {
            "contract_completion": "all_required_items_accepted",
            "environment_probe_completes_contracts": False,
            "cpu_oracle_is_execution": False,
            "runtime_target": {
                "backend": "gemsim_amd",
                "arch": "gfx950",
                "wavefront_size": 64,
            },
            "required_run_kinds": ["fresh", "repeat"],
            "fallback_allowed": False,
        },
        "model": {
            "id": "Qwen/Qwen3.5-0.8B",
            "revision": model_manifest["revision"],
            "config_sha256": _sha256(config_path),
            "architecture": config.get("architectures", []),
            "model_type": config.get("model_type"),
            "text_model_type": text_config.get("model_type"),
            "scope": {
                "text_only_first_gate": True,
                "vision_path_deferred": True,
                "full_rocm_opencl_cts_required": False,
                "cpu_fallback_allowed": False,
                "nvidia_runtime_counts_as_amd_pass": False,
            },
        },
        "topology": {
            "num_hidden_layers": text_config.get("num_hidden_layers"),
            "layer_type_counts": dict(sorted(counts.items())),
            "linear_attention_layers": [i for i, value in enumerate(layer_types) if value == "linear_attention"],
            "full_attention_layers": [i for i, value in enumerate(layer_types) if value == "full_attention"],
            "hidden_size": text_config.get("hidden_size"),
            "intermediate_size": text_config.get("intermediate_size"),
            "vocab_size": text_config.get("vocab_size"),
            "num_attention_heads": text_config.get("num_attention_heads"),
            "num_key_value_heads": text_config.get("num_key_value_heads"),
            "head_dim": text_config.get("head_dim"),
            "linear_num_key_heads": text_config.get("linear_num_key_heads"),
            "linear_num_value_heads": text_config.get("linear_num_value_heads"),
            "linear_key_head_dim": text_config.get("linear_key_head_dim"),
            "linear_value_head_dim": text_config.get("linear_value_head_dim"),
            "linear_conv_kernel_dim": text_config.get("linear_conv_kernel_dim"),
        },
        "source_revisions": source_revisions,
        "counts": {
            **inventory_counts,
            "configured_layer_count": len(layer_types),
            "configured_layer_type_counts": dict(sorted(counts.items())),
            "contract_count": contract_counts["total"],
            "contract_static_source_ok_count": contract_counts["static_source_ok"],
            "contract_amd_runtime_not_executed_count": contract_counts[
                "amd_runtime_not_executed"
            ],
            "contract_required_triton_symbol_count": contract_counts[
                "required_triton_symbol_count"
            ],
            "contract_alternate_triton_symbol_count": contract_counts[
                "alternate_triton_symbol_count"
            ],
        },
        "source_files": source_meta,
        "contracts": contracts,
        "work_items": work_items,
        "backend_support": backend_support,
        # Keep a short, stable top-level projection for CI/reporting tools.
        # The detailed evidence remains under ``backend_support``.
        "unsupported_backends": backend_support["unsupported"],
        "summary": {
            "contract_count": contract_counts["total"],
            "static_source_ok": static_ok,
            "amd_runtime_executed": False,
            "amd_runtime_pass": False,
            "accepted_contract_count": 0,
            "all_contracts_accepted": False,
            "runtime_status": "work_queue_incomplete",
            "blocker": "32/32 required work items still require external generic results",
        },
    }
    work_queue_summary = queue_summary(manifest)
    manifest["counts"].update(
        {
            "work_item_count": work_queue_summary["work_item_count"],
            "work_item_configured_count": work_queue_summary[
                "configured_work_item_count"
            ],
            "work_item_ready_count": work_queue_summary["ready_work_item_count"],
            "work_item_accepted_count": work_queue_summary[
                "accepted_work_item_count"
            ],
        }
    )
    manifest["summary"].update(
        {
            "accepted_contract_count": work_queue_summary[
                "accepted_contract_count"
            ],
            "all_contracts_accepted": work_queue_summary[
                "all_contracts_accepted"
            ],
        }
    )
    validation_errors = validate_manifest(manifest)
    if validation_errors:
        raise RuntimeError(
            "generated work queue failed validation: " + "; ".join(validation_errors)
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tools" / "qwen35_operator_manifest.json",
        help="write the deterministic manifest here (default: tools/qwen35_operator_manifest.json)",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="return non-zero unless all 15 operator contracts are accepted",
    )
    args = parser.parse_args()
    manifest = build_manifest()
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.output == Path("-"):
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "contract_count": manifest["summary"]["contract_count"],
                    "static_source_ok": manifest["summary"]["static_source_ok"],
                    "runtime_status": manifest["summary"]["runtime_status"],
                    "blocker": manifest["summary"]["blocker"],
                    "accepted_contract_count": manifest["summary"][
                        "accepted_contract_count"
                    ],
                },
                sort_keys=True,
            )
        )
    if not manifest["summary"]["static_source_ok"]:
        return 1
    if args.require_complete and not manifest["summary"]["all_contracts_accepted"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
