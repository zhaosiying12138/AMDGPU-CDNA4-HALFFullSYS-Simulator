#!/usr/bin/env python3

import runpy
from pathlib import Path


HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE / "_gemsim_bootstrap.py"))["bootstrap"](
    __file__, "qwen35-model-decode"
)

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import tempfile

import torch
import triton
from safetensors import safe_open
from safetensors.torch import save as serialize_safetensors


ROOT = HERE.parents[1]
MODEL_DIR = ROOT / "models/Qwen3.5-0.8B"
MODEL_FILENAME = "model.safetensors-00001-of-00001.safetensors"
MODEL_FILE = MODEL_DIR / MODEL_FILENAME
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DEVICE = triton.runtime.driver.active.get_active_torch_device()
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3584
GATE_UP_SIZE = 7168
GDN_HEADS = 16
GDN_HEAD_DIM = 128
GDN_QKV_DIM = 6144
GDN_Z_DIM = 2048
GDN_QKVZ_DIM = 8192
FULL_Q_HEADS = 8
FULL_KV_HEADS = 2
FULL_HEAD_DIM = 256
FULL_Q_SIZE = 2048
FULL_KV_SIZE = 512
FULL_Q_GATE_SIZE = 4096
FULL_QKV_GATE_SIZE = 5120
ROTARY_DIM = 64
ROPE_THETA = 10_000_000.0
KV_BLOCK_SIZE = 16
KV_CONTENT_SIZE = 512
EPSILON = 1.0e-6
VOCAB_SIZE = 248320
EMBEDDING_WEIGHT_NAME = "model.language_model.embed_tokens.weight"
EXPECTED_SELECTED_TENSOR_COUNT = 320
EXPECTED_LAYER_TYPES = tuple(
    layer_type
    for _ in range(6)
    for layer_type in (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )
)
EXPECTED_ARTIFACTS = {
    "config.json": {
        "bytes": 2907,
        "sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    },
    "model.safetensors.index.json": {
        "bytes": 50900,
        "sha256": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
    },
    MODEL_FILENAME: {
        "bytes": 1746942600,
        "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    },
    "manifest.json": {
        "bytes": 1008,
        "sha256": "de2281cc73a1329d13245cb9658be910cf435e72c4ea0277c4f8811a24edf762",
    },
}
NVIDIA_GOLDEN_SCHEMA = "amdgpu-sim.qwen35-nvidia-backbone-golden.v1"
NVIDIA_GOLDEN_KIND = "independent_torch_cuda_backbone_golden"
NVIDIA_GOLDEN_ACCEPTED_SHA256 = {
    4: {
        "metadata.json": "33be05f6c7bb07b949f050dd68c875d03baf9e3a018a89aee5fffaaeb78cbccc",
        "results.safetensors": "0314d1c4cbd16fd5cb5c6298dd6df86a1e84172b6e21539519870f5d6fe66996",
        "script_sha256": "bd630ee9693f5ea79c7ba7c05d4981b985dcb5b8e4cf119ee8ea8aaa20535231",
    },
    24: {
        "metadata.json": "2a5d43d9c8b068ad15027916db4120782fc99b7ffa48bf225073bc09f909a9fb",
        "results.safetensors": "43a6b9f8d2cc29c728444ead69f7a0df575d634b35593bc1b2490b9ed0adfb9b",
        "script_sha256": "bd630ee9693f5ea79c7ba7c05d4981b985dcb5b8e4cf119ee8ea8aaa20535231",
    },
}
NVIDIA_GOLDEN_VLLM_HEAD = "8d9b52f7c2514490bdadfd5eb0c931e58625df2e"
NVIDIA_GOLDEN_GPU = {
    "name": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "uuid": "GPU-64aae36b-ef77-b0d4-b1c7-f7ab17a729f1",
    "compute_capability": [12, 0],
}
NVIDIA_GOLDEN_TOLERANCES = {
    "returned_hidden": (0.03125, 0.03, 0.03),
    "returned_residual": (0.03125, 0.03, 0.03),
    # This boundary includes accumulated upstream BF16 drift before RMS
    # amplification. The relative-L2 gate prevents a broad-error pass.
    "final_norm": (0.125, 0.03, 0.03),
}
AT_FDCWD = -100
RENAME_NOREPLACE = 1
ATOMIC_OUTPUT_WRITE_POLICY = (
    "same_parent_temp_directory_fsync_renameat2_noreplace_parent_fsync"
)


layer0 = runpy.run_path(str(HERE / "qwen35_layer0_decode_correctness.py"))
layer3 = runpy.run_path(str(HERE / "qwen35_layer3_decode_correctness.py"))
plain_rms_norm = layer0["plain_rms_norm"]
dense_linear = layer0["dense_linear"]
gdn_conv_decode = layer0["gdn_conv_decode"]
gdn_recurrent_decode = layer0["gdn_recurrent_decode"]
gdn_output_norm_gate = layer0["gdn_output_norm_gate"]
fused_residual_rms_norm = layer0["fused_residual_rms_norm"]
silu_and_mul = layer0["silu_and_mul"]
qk_rmsnorm_rope_gate = layer3["qk_rmsnorm_rope_gate"]
kv_cache_store_kernel = layer3["kv_cache_store_kernel"]
gqa_decode_scores_kernel = layer3["gqa_decode_scores_kernel"]
gqa_decode_output_kernel = layer3["gqa_decode_output_kernel"]
apply_sigmoid_output_gate = layer3["apply_sigmoid_output_gate"]


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().contiguous()
    raw = (
        contiguous.view(torch.uint16)
        if value.dtype == torch.bfloat16
        else contiguous
    )
    return hashlib.sha256(raw.numpy().tobytes(order="C")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_descriptor(value: torch.Tensor) -> dict:
    return {
        "dtype": str(value.dtype).replace("torch.", ""),
        "shape": list(value.shape),
        "sha256": tensor_sha256(value),
    }


def compare_nvidia_golden_tensor(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    tolerance_name: str,
) -> dict:
    if actual.dtype != torch.bfloat16 or tuple(actual.shape) != (1, HIDDEN_SIZE):
        raise RuntimeError(
            f"actual NVIDIA golden boundary contract mismatch for {name}: "
            f"{actual.dtype} {tuple(actual.shape)}"
        )
    if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(expected.shape):
        raise RuntimeError(
            f"NVIDIA golden tensor contract mismatch for {name}: "
            f"actual={actual.dtype} {tuple(actual.shape)} "
            f"expected={expected.dtype} {tuple(expected.shape)}"
        )
    atol, rtol, max_relative_l2 = NVIDIA_GOLDEN_TOLERANCES[tolerance_name]
    actual_float = actual.to(torch.float32)
    expected_float = expected.to(torch.float32)
    error = torch.abs(actual_float - expected_float)
    finite = (
        torch.isfinite(actual_float)
        & torch.isfinite(expected_float)
        & torch.isfinite(error)
    )
    mismatch = (~finite) | (
        error > atol + rtol * torch.abs(expected_float)
    )
    finite_error = torch.where(finite, error, torch.zeros_like(error))
    expected_l2 = float(torch.linalg.vector_norm(expected_float).item())
    error_l2 = float(torch.linalg.vector_norm(finite_error).item())
    relative_l2 = error_l2 / expected_l2 if expected_l2 else error_l2
    mismatch_count = int(torch.count_nonzero(mismatch).item())
    nonfinite_count = int(torch.count_nonzero(~finite).item())
    return {
        "name": name,
        "shape": list(actual.shape),
        "dtype": "bfloat16",
        "actual_sha256": tensor_sha256(actual),
        "expected_sha256": tensor_sha256(expected),
        "atol": atol,
        "rtol": rtol,
        "max_relative_l2": max_relative_l2,
        "max_abs_error": float(torch.max(finite_error).item()),
        "relative_l2_error": relative_l2,
        "mismatch_count": mismatch_count,
        "nonfinite_count": nonfinite_count,
        "all_values_finite": nonfinite_count == 0,
        "correct": (
            mismatch_count == 0
            and nonfinite_count == 0
            and relative_l2 <= max_relative_l2
        ),
    }


def load_nvidia_golden(
    golden_dir: Path,
    max_layers: int,
    token_id: int,
    checkpoint_provenance: dict,
) -> dict:
    accepted_hashes = NVIDIA_GOLDEN_ACCEPTED_SHA256.get(max_layers)
    if accepted_hashes is None:
        raise RuntimeError(
            "--nvidia-golden-dir is accepted only with --max-layers 4 or 24"
        )
    expanded_dir = golden_dir.expanduser()
    if expanded_dir.is_symlink():
        raise RuntimeError("NVIDIA golden directory must not be a symlink")
    resolved_dir = expanded_dir.resolve(strict=True)
    if not resolved_dir.is_dir():
        raise RuntimeError(f"NVIDIA golden path is not a directory: {resolved_dir}")
    entries = {entry.name: entry for entry in resolved_dir.iterdir()}
    if set(entries) != {"metadata.json", "results.safetensors"}:
        raise RuntimeError(
            f"NVIDIA golden directory contents mismatch: {sorted(entries)!r}"
        )
    for name, path in entries.items():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"NVIDIA golden entry is not a regular file: {name}")
        observed_hash = file_sha256(path)
        if observed_hash != accepted_hashes[name]:
            raise RuntimeError(
                f"unaccepted NVIDIA golden {name} SHA-256: {observed_hash}"
            )

    metadata_path = entries["metadata.json"]
    results_path = entries["results.safetensors"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError("NVIDIA golden metadata must be a JSON object")
    if metadata.get("schema") != NVIDIA_GOLDEN_SCHEMA:
        raise RuntimeError("NVIDIA golden schema mismatch")
    if metadata.get("kind") != NVIDIA_GOLDEN_KIND:
        raise RuntimeError("NVIDIA golden kind mismatch")
    if metadata.get("results_file_sha256") != accepted_hashes[
        "results.safetensors"
    ]:
        raise RuntimeError("NVIDIA golden results hash metadata mismatch")

    expected_case = {
        "scope": "decoder_backbone_prefix",
        "max_layers": max_layers,
        "layer_types": list(EXPECTED_LAYER_TYPES[:max_layers]),
        "linear_attention_layers": sum(
            value == "linear_attention"
            for value in EXPECTED_LAYER_TYPES[:max_layers]
        ),
        "full_attention_layers": sum(
            value == "full_attention"
            for value in EXPECTED_LAYER_TYPES[:max_layers]
        ),
        "token_id": token_id,
        "tokens": 1,
        "position": 0,
        "tensor_layout": "NHD",
        "cache": "empty_per_layer",
        "state_initialization": "zero_first_token",
        "selected_conv_state": 1,
        "linear_weight_layout": "checkpoint [out,in], matmul uses weight.T",
        "outer_rms_weight": "1 + checkpoint raw weight",
        "gdn_output_norm_weight": "checkpoint weight directly",
        "returned_pair": ["mlp_down", "post_attention_residual"],
        "final_norm_applied": max_layers == 24,
        "final_norm_scope": (
            "full_backbone"
            if max_layers == 24
            else "not_applied_for_backbone_prefix"
        ),
        "excludes": ["final vocabulary projection", "logits"],
    }
    if metadata.get("case") != expected_case:
        raise RuntimeError("NVIDIA golden case contract mismatch")

    artifact_hashes = checkpoint_provenance["artifact_hashes"]
    expected_model = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "config_sha256": artifact_hashes["config.json"]["sha256"],
        "index_sha256": artifact_hashes[
            "model.safetensors.index.json"
        ]["sha256"],
        "index_total_size": 1746882752,
        "shard": MODEL_FILENAME,
        "shard_bytes": artifact_hashes[MODEL_FILENAME]["bytes"],
        "shard_manifest_sha256": artifact_hashes[MODEL_FILENAME]["sha256"],
    }
    model = metadata.get("model")
    if not isinstance(model, dict) or any(
        model.get(name) != value for name, value in expected_model.items()
    ):
        raise RuntimeError("NVIDIA golden model provenance mismatch")
    if not isinstance(model.get("directory"), str) or not model["directory"]:
        raise RuntimeError("NVIDIA golden model directory provenance is missing")

    script = metadata.get("script")
    if script != {
        "path": "tools/qwen35_nvidia_golden.py",
        "sha256": accepted_hashes["script_sha256"],
    }:
        raise RuntimeError("NVIDIA golden generator identity mismatch")
    formula_source = metadata.get("formula_source")
    if (
        not isinstance(formula_source, dict)
        or formula_source.get("vllm_git_head") != NVIDIA_GOLDEN_VLLM_HEAD
        or not isinstance(formula_source.get("formula_source_sha256"), dict)
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in formula_source["formula_source_sha256"].values()
        )
    ):
        raise RuntimeError("NVIDIA golden formula source identity mismatch")
    environment = metadata.get("environment")
    gpu = environment.get("gpu") if isinstance(environment, dict) else None
    if not isinstance(gpu, dict) or any(
        gpu.get(name) != value for name, value in NVIDIA_GOLDEN_GPU.items()
    ):
        raise RuntimeError("NVIDIA golden GPU identity mismatch")
    if (
        environment.get("deterministic_algorithms") is not True
        or environment.get("tf32_cudnn") is not False
        or environment.get("tf32_matmul") is not False
        or environment.get("float32_matmul_precision") != "highest"
    ):
        raise RuntimeError("NVIDIA golden deterministic environment mismatch")

    expected_embedded_provenance = {
        "layer_types": list(EXPECTED_LAYER_TYPES[:max_layers]),
        "max_layers": max_layers,
        "model_id": MODEL_ID,
        "position": 0,
        "revision": MODEL_REVISION,
        "schema": NVIDIA_GOLDEN_SCHEMA,
        "token_id": token_id,
    }
    results_metadata = metadata.get("results")
    if not isinstance(results_metadata, dict):
        raise RuntimeError("NVIDIA golden results metadata is missing")
    with safe_open(results_path, framework="pt", device="cpu") as tensors:
        tensor_names = set(tensors.keys())
        if tensor_names != set(results_metadata):
            raise RuntimeError("NVIDIA golden tensor set mismatch")
        embedded_metadata = tensors.metadata()
        if not isinstance(embedded_metadata, dict) or set(embedded_metadata) != {
            "provenance"
        }:
            raise RuntimeError("NVIDIA golden embedded metadata mismatch")
        try:
            embedded_provenance = json.loads(embedded_metadata["provenance"])
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "NVIDIA golden embedded provenance is malformed"
            ) from error
        if embedded_provenance != expected_embedded_provenance:
            raise RuntimeError("NVIDIA golden embedded provenance mismatch")
        loaded = {
            name: tensors.get_tensor(name).clone().contiguous()
            for name in tensor_names
        }

    observed_descriptors = {
        name: tensor_descriptor(value) for name, value in loaded.items()
    }
    if observed_descriptors != results_metadata:
        raise RuntimeError("NVIDIA golden per-tensor descriptor mismatch")
    comparison_names = [
        name
        for layer in range(max_layers)
        for name in (
            f"layers.{layer}.returned_hidden",
            f"layers.{layer}.returned_residual",
        )
    ]
    if max_layers == 24:
        comparison_names.append("final_norm")
    result_order = metadata.get("result_order")
    if (
        not isinstance(result_order, list)
        or len(result_order) != len(tensor_names)
        or len(set(result_order)) != len(result_order)
        or set(result_order) != tensor_names
        or result_order[-len(comparison_names) :] != comparison_names
    ):
        raise RuntimeError("NVIDIA golden result order mismatch")
    last_layer = max_layers - 1
    if not torch.equal(
        loaded["final_hidden"],
        loaded[f"layers.{last_layer}.returned_hidden"],
    ) or not torch.equal(
        loaded["final_residual"],
        loaded[f"layers.{last_layer}.returned_residual"],
    ):
        raise RuntimeError("NVIDIA golden top-level final pair alias mismatch")
    nonfinite_counts = metadata.get("result_nonfinite_counts")
    if (
        metadata.get("all_results_finite") is not True
        or not isinstance(nonfinite_counts, dict)
        or set(nonfinite_counts) != tensor_names
        or any(value != 0 for value in nonfinite_counts.values())
        or any(
            not bool(torch.all(torch.isfinite(value.to(torch.float32))).item())
            for value in loaded.values()
        )
    ):
        raise RuntimeError("NVIDIA golden contains nonfinite results")
    for name in comparison_names:
        value = loaded.get(name)
        if value is None or value.dtype != torch.bfloat16 or tuple(value.shape) != (
            1,
            HIDDEN_SIZE,
        ):
            raise RuntimeError(f"NVIDIA golden comparison tensor mismatch: {name}")

    layer_results = metadata.get("layer_results")
    expected_layer_results = [
        {
            "index": layer,
            "type": EXPECTED_LAYER_TYPES[layer],
            "returned_hidden": observed_descriptors[
                f"layers.{layer}.returned_hidden"
            ],
            "returned_residual": observed_descriptors[
                f"layers.{layer}.returned_residual"
            ],
        }
        for layer in range(max_layers)
    ]
    if layer_results != expected_layer_results:
        raise RuntimeError("NVIDIA golden layer result metadata mismatch")
    if max_layers == 24:
        if metadata.get("final_norm") != observed_descriptors["final_norm"]:
            raise RuntimeError("NVIDIA golden final norm metadata mismatch")
    elif "final_norm" in observed_descriptors or "final_norm" in metadata:
        raise RuntimeError("prefix NVIDIA golden unexpectedly contains final norm")

    return {
        "directory": str(resolved_dir),
        "metadata": metadata,
        "metadata_sha256": accepted_hashes["metadata.json"],
        "results_sha256": accepted_hashes["results.safetensors"],
        "tensors": {
            name: loaded[name] for name in comparison_names
        },
    }


def selected_tensor_names(layer_types: tuple[str, ...]) -> tuple[str, ...]:
    names = [EMBEDDING_WEIGHT_NAME]
    for layer, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{layer}"
        names.extend(
            (
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            )
        )
        if layer_type == "linear_attention":
            names.extend(
                (
                    f"{prefix}.linear_attn.in_proj_qkv.weight",
                    f"{prefix}.linear_attn.in_proj_z.weight",
                    f"{prefix}.linear_attn.in_proj_b.weight",
                    f"{prefix}.linear_attn.in_proj_a.weight",
                    f"{prefix}.linear_attn.conv1d.weight",
                    f"{prefix}.linear_attn.A_log",
                    f"{prefix}.linear_attn.dt_bias",
                    f"{prefix}.linear_attn.norm.weight",
                    f"{prefix}.linear_attn.out_proj.weight",
                )
            )
        elif layer_type == "full_attention":
            names.extend(
                (
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                    f"{prefix}.self_attn.q_norm.weight",
                    f"{prefix}.self_attn.k_norm.weight",
                    f"{prefix}.self_attn.o_proj.weight",
                )
            )
        else:
            raise RuntimeError(f"unsupported layer type: {layer_type}")
    names.append("model.language_model.norm.weight")
    if len(names) != len(set(names)):
        raise RuntimeError("selected checkpoint tensor names are not unique")
    return tuple(names)


def load_tensor(
    tensors,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    source_tensor_hashes: dict[str, str],
):
    value = tensors.get_tensor(name).clone().contiguous()
    if value.dtype != dtype or tuple(value.shape) != shape:
        raise RuntimeError(
            f"checkpoint tensor contract mismatch for {name}: "
            f"{value.dtype} {tuple(value.shape)}"
        )
    if value.untyped_storage().nbytes() != value.numel() * value.element_size():
        raise RuntimeError(f"checkpoint tensor does not own exact storage: {name}")
    source_tensor_hashes[name] = tensor_sha256(value)
    return value


def load_common_weights(
    tensors,
    layer: int,
    source_tensor_hashes: dict[str, str],
    assembled_weight_hashes: dict[str, str],
) -> dict:
    prefix = f"model.language_model.layers.{layer}"
    gate = load_tensor(
        tensors,
        f"{prefix}.mlp.gate_proj.weight",
        torch.bfloat16,
        (INTERMEDIATE_SIZE, HIDDEN_SIZE),
        source_tensor_hashes,
    )
    up = load_tensor(
        tensors,
        f"{prefix}.mlp.up_proj.weight",
        torch.bfloat16,
        (INTERMEDIATE_SIZE, HIDDEN_SIZE),
        source_tensor_hashes,
    )
    gate_up = torch.cat((gate, up), dim=0).contiguous()
    assembled_weight_hashes[f"layer.{layer}.mlp.gate_up"] = tensor_sha256(
        gate_up
    )
    return {
        "input_norm": load_tensor(
            tensors,
            f"{prefix}.input_layernorm.weight",
            torch.bfloat16,
            (HIDDEN_SIZE,),
            source_tensor_hashes,
        ),
        "post_norm": load_tensor(
            tensors,
            f"{prefix}.post_attention_layernorm.weight",
            torch.bfloat16,
            (HIDDEN_SIZE,),
            source_tensor_hashes,
        ),
        "gate_up": gate_up,
        "down": load_tensor(
            tensors,
            f"{prefix}.mlp.down_proj.weight",
            torch.bfloat16,
            (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            source_tensor_hashes,
        ),
    }


def input_norm(
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    raw_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = torch.empty_like(hidden)
    if residual is None:
        residual = hidden.clone()
        plain_rms_norm(hidden, normalized, raw_weight, hidden.shape[0])
        return normalized, residual
    next_residual = torch.empty_like(residual)
    fused_residual_rms_norm(
        hidden,
        residual,
        raw_weight,
        normalized,
        next_residual,
        hidden.shape[0],
    )
    return normalized, next_residual


def mlp(
    normalized: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    gate_up = torch.empty((1, GATE_UP_SIZE), dtype=torch.bfloat16, device=DEVICE)
    dense_linear(normalized, gate_up_weight, gate_up)
    activated = torch.empty(
        (1, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=DEVICE
    )
    scratch = torch.zeros((1,), dtype=torch.uint8, device=DEVICE)
    silu_and_mul(gate_up, activated, scratch, 1)
    output = torch.empty((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=DEVICE)
    dense_linear(activated, down_weight, output)
    return output


def run_gdn_layer(
    tensors,
    layer: int,
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    source_tensor_hashes: dict[str, str],
    assembled_weight_hashes: dict[str, str],
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    prefix = f"model.language_model.layers.{layer}"
    weights = load_common_weights(
        tensors,
        layer,
        source_tensor_hashes,
        assembled_weight_hashes,
    )
    qkv = load_tensor(
        tensors,
        f"{prefix}.linear_attn.in_proj_qkv.weight",
        torch.bfloat16,
        (GDN_QKV_DIM, HIDDEN_SIZE),
        source_tensor_hashes,
    )
    z_weight = load_tensor(
        tensors,
        f"{prefix}.linear_attn.in_proj_z.weight",
        torch.bfloat16,
        (GDN_Z_DIM, HIDDEN_SIZE),
        source_tensor_hashes,
    )
    b_weight = load_tensor(
        tensors,
        f"{prefix}.linear_attn.in_proj_b.weight",
        torch.bfloat16,
        (GDN_HEADS, HIDDEN_SIZE),
        source_tensor_hashes,
    )
    a_weight = load_tensor(
        tensors,
        f"{prefix}.linear_attn.in_proj_a.weight",
        torch.bfloat16,
        (GDN_HEADS, HIDDEN_SIZE),
        source_tensor_hashes,
    )
    conv_weight = load_tensor(
        tensors,
        f"{prefix}.linear_attn.conv1d.weight",
        torch.bfloat16,
        (GDN_QKV_DIM, 1, 4),
        source_tensor_hashes,
    ).view(GDN_QKV_DIM, 4)
    a_log = load_tensor(
        tensors,
        f"{prefix}.linear_attn.A_log",
        torch.float32,
        (GDN_HEADS,),
        source_tensor_hashes,
    )
    dt_bias = load_tensor(
        tensors,
        f"{prefix}.linear_attn.dt_bias",
        torch.bfloat16,
        (GDN_HEADS,),
        source_tensor_hashes,
    )
    norm_weight = load_tensor(
        tensors,
        f"{prefix}.linear_attn.norm.weight",
        torch.float32,
        (GDN_HEAD_DIM,),
        source_tensor_hashes,
    )
    out_weight = load_tensor(
        tensors,
        f"{prefix}.linear_attn.out_proj.weight",
        torch.bfloat16,
        (HIDDEN_SIZE, GDN_Z_DIM),
        source_tensor_hashes,
    )

    qkvz_weight = torch.cat((qkv, z_weight), dim=0).contiguous()
    ba_weight = torch.cat((b_weight, a_weight), dim=0).contiguous()
    assembled_weight_hashes[
        f"layer.{layer}.linear_attn.qkvz"
    ] = tensor_sha256(qkvz_weight)
    assembled_weight_hashes[
        f"layer.{layer}.linear_attn.ba"
    ] = tensor_sha256(ba_weight)
    assembled_weight_hashes[
        f"layer.{layer}.linear_attn.conv1d_flat"
    ] = tensor_sha256(conv_weight)

    normalized, residual = input_norm(hidden, residual, weights["input_norm"])
    qkvz = torch.empty((1, GDN_QKVZ_DIM), dtype=torch.bfloat16, device=DEVICE)
    ba = torch.empty((1, 2 * GDN_HEADS), dtype=torch.bfloat16, device=DEVICE)
    dense_linear(normalized, qkvz_weight, qkvz)
    dense_linear(normalized, ba_weight, ba)
    mixed_qkv = qkvz[:, :GDN_QKV_DIM]
    z = qkvz[:, GDN_QKV_DIM:].view(1, GDN_HEADS, GDN_HEAD_DIM)
    conv_state = torch.zeros(
        (1, GDN_QKV_DIM, 3), dtype=torch.bfloat16, device=DEVICE
    )
    state_index = torch.zeros((1,), dtype=torch.int32, device=DEVICE)
    gdn_conv_decode(mixed_qkv, conv_weight, conv_state, state_index)
    recurrent_state = torch.zeros(
        (GDN_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM),
        dtype=torch.float32,
        device=DEVICE,
    )
    recurrent_output = torch.empty(
        (1, GDN_HEADS, GDN_HEAD_DIM), dtype=torch.bfloat16, device=DEVICE
    )
    gdn_recurrent_decode(
        mixed_qkv,
        ba[:, GDN_HEADS:],
        ba[:, :GDN_HEADS],
        a_log,
        dt_bias,
        recurrent_state,
        recurrent_output,
    )
    gated = torch.empty_like(recurrent_output)
    gdn_output_norm_gate(
        recurrent_output, z, norm_weight, gated, hidden.shape[0]
    )
    attention_out = torch.empty_like(hidden)
    dense_linear(gated.view(1, GDN_Z_DIM), out_weight, attention_out)
    post_norm = torch.empty_like(hidden)
    post_residual = torch.empty_like(residual)
    fused_residual_rms_norm(
        attention_out,
        residual,
        weights["post_norm"],
        post_norm,
        post_residual,
        1,
    )
    output = mlp(post_norm, weights["gate_up"], weights["down"])
    state = {
        "conv_state_sha256": tensor_sha256(conv_state),
        "recurrent_state_sha256": tensor_sha256(recurrent_state),
    }
    return output, post_residual, state


def first_token_cos_sin() -> torch.Tensor:
    inverse_frequency = 1.0 / (
        ROPE_THETA
        ** (
            torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32, device=DEVICE)
            / ROTARY_DIM
        )
    )
    frequencies = torch.zeros((1, 1), dtype=torch.float32, device=DEVICE)
    frequencies = frequencies * inverse_frequency[None, :]
    return torch.cat((torch.cos(frequencies), torch.sin(frequencies)), dim=1).to(
        torch.bfloat16
    )


def first_token_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    slot = torch.zeros((1,), dtype=torch.int32, device=DEVICE)
    scores = torch.empty((FULL_Q_HEADS, 1), dtype=torch.float32, device=DEVICE)
    output = torch.empty((1, FULL_Q_SIZE), dtype=torch.bfloat16, device=DEVICE)
    kv_cache_store_kernel[(FULL_KV_HEADS,)](
        k,
        v,
        cache,
        slot,
        NUM_KV_HEADS=FULL_KV_HEADS,
        HEAD_DIM=FULL_HEAD_DIM,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        KV_CONTENT_SIZE=KV_CONTENT_SIZE,
        num_warps=4,
    )
    gqa_decode_scores_kernel[(FULL_Q_HEADS,)](
        q,
        cache,
        scores,
        NUM_Q_HEADS=FULL_Q_HEADS,
        NUM_KV_HEADS=FULL_KV_HEADS,
        HEAD_DIM=FULL_HEAD_DIM,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        KV_CONTENT_SIZE=KV_CONTENT_SIZE,
        SEQUENCE_LENGTH=1,
        SCORE_BLOCK=1,
        DIM_BLOCK=64,
        num_warps=4,
    )
    gqa_decode_output_kernel[(FULL_Q_HEADS, 4)](
        scores,
        cache,
        output,
        NUM_Q_HEADS=FULL_Q_HEADS,
        NUM_KV_HEADS=FULL_KV_HEADS,
        HEAD_DIM=FULL_HEAD_DIM,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        KV_CONTENT_SIZE=KV_CONTENT_SIZE,
        SEQUENCE_LENGTH=1,
        SCORE_BLOCK=1,
        DIM_BLOCK=64,
        num_warps=4,
    )
    return output, scores


def run_full_attention_layer(
    tensors,
    layer: int,
    hidden: torch.Tensor,
    residual: torch.Tensor,
    source_tensor_hashes: dict[str, str],
    assembled_weight_hashes: dict[str, str],
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    prefix = f"model.language_model.layers.{layer}"
    weights = load_common_weights(
        tensors,
        layer,
        source_tensor_hashes,
        assembled_weight_hashes,
    )
    q_weight = load_tensor(
        tensors,
        f"{prefix}.self_attn.q_proj.weight",
        torch.bfloat16,
        (FULL_Q_GATE_SIZE, HIDDEN_SIZE),
        source_tensor_hashes,
    )
    k_weight = load_tensor(
        tensors,
        f"{prefix}.self_attn.k_proj.weight",
        torch.bfloat16,
        (FULL_KV_SIZE, HIDDEN_SIZE),
        source_tensor_hashes,
    )
    v_weight = load_tensor(
        tensors,
        f"{prefix}.self_attn.v_proj.weight",
        torch.bfloat16,
        (FULL_KV_SIZE, HIDDEN_SIZE),
        source_tensor_hashes,
    )
    q_raw = load_tensor(
        tensors,
        f"{prefix}.self_attn.q_norm.weight",
        torch.bfloat16,
        (FULL_HEAD_DIM,),
        source_tensor_hashes,
    )
    k_raw = load_tensor(
        tensors,
        f"{prefix}.self_attn.k_norm.weight",
        torch.bfloat16,
        (FULL_HEAD_DIM,),
        source_tensor_hashes,
    )
    out_weight = load_tensor(
        tensors,
        f"{prefix}.self_attn.o_proj.weight",
        torch.bfloat16,
        (HIDDEN_SIZE, FULL_Q_SIZE),
        source_tensor_hashes,
    )

    qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0).contiguous()
    q_gamma = (q_raw.to(torch.float32) + 1.0).contiguous()
    k_gamma = (k_raw.to(torch.float32) + 1.0).contiguous()
    assembled_weight_hashes[
        f"layer.{layer}.self_attn.qkv_gate"
    ] = tensor_sha256(qkv_weight)
    assembled_weight_hashes[
        f"layer.{layer}.self_attn.q_gamma"
    ] = tensor_sha256(q_gamma)
    assembled_weight_hashes[
        f"layer.{layer}.self_attn.k_gamma"
    ] = tensor_sha256(k_gamma)

    normalized, residual = input_norm(hidden, residual, weights["input_norm"])
    qkv_gate = torch.empty(
        (1, FULL_QKV_GATE_SIZE), dtype=torch.bfloat16, device=DEVICE
    )
    dense_linear(normalized, qkv_weight, qkv_gate)
    q_gate = qkv_gate[:, :FULL_Q_GATE_SIZE]
    k_source = qkv_gate[:, FULL_Q_GATE_SIZE : FULL_Q_GATE_SIZE + FULL_KV_SIZE]
    v = qkv_gate[:, FULL_Q_GATE_SIZE + FULL_KV_SIZE :]
    q = torch.empty((1, FULL_Q_SIZE), dtype=torch.bfloat16, device=DEVICE)
    k = torch.empty((1, FULL_KV_SIZE), dtype=torch.bfloat16, device=DEVICE)
    gate = torch.empty((1, FULL_Q_SIZE), dtype=torch.bfloat16, device=DEVICE)
    position = torch.zeros((1,), dtype=torch.int32, device=DEVICE)
    qk_rmsnorm_rope_gate(
        q_gate,
        k_source,
        q_gamma,
        k_gamma,
        first_token_cos_sin(),
        position,
        q,
        k,
        gate,
    )
    cache = torch.zeros(
        (1, KV_BLOCK_SIZE, FULL_KV_HEADS, KV_CONTENT_SIZE),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    attention, scores = first_token_attention(q, k, v, cache)
    gated = torch.empty_like(attention)
    apply_sigmoid_output_gate(attention, gate, gated)
    attention_out = torch.empty_like(hidden)
    dense_linear(gated, out_weight, attention_out)
    post_norm = torch.empty_like(hidden)
    post_residual = torch.empty_like(residual)
    fused_residual_rms_norm(
        attention_out,
        residual,
        weights["post_norm"],
        post_norm,
        post_residual,
        1,
    )
    output = mlp(post_norm, weights["gate_up"], weights["down"])
    state = {
        "kv_cache_sha256": tensor_sha256(cache),
        "scores_sha256": tensor_sha256(scores),
    }
    return output, post_residual, state


def validate_checkpoint(max_layers: int) -> tuple[tuple[str, ...], dict]:
    artifact_hashes = {}
    for filename, expected in EXPECTED_ARTIFACTS.items():
        path = MODEL_DIR / filename
        observed = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        if observed != expected:
            raise RuntimeError(
                f"checkpoint artifact mismatch for {filename}: {observed!r}"
            )
        artifact_hashes[filename] = observed

    config = json.loads((MODEL_DIR / "config.json").read_text())
    text = config.get("text_config", {})
    root_contract = {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
    }
    expected_root_contract = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": True,
    }
    text_keys = (
        "attention_bias",
        "attention_dropout",
        "attn_output_gate",
        "dtype",
        "eos_token_id",
        "full_attention_interval",
        "head_dim",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "max_position_embeddings",
        "mlp_only_layers",
        "model_type",
        "mtp_num_hidden_layers",
        "mtp_use_dedicated_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rms_norm_eps",
        "tie_word_embeddings",
        "use_cache",
        "vocab_size",
        "mamba_ssm_dtype",
        "rope_parameters",
        "layer_types",
    )
    text_contract = {key: text.get(key) for key in text_keys}
    expected_text_contract = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": FULL_HEAD_DIM,
        "hidden_act": "silu",
        "model_type": "qwen3_5_text",
        "hidden_size": HIDDEN_SIZE,
        "initializer_range": 0.02,
        "intermediate_size": INTERMEDIATE_SIZE,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": GDN_HEAD_DIM,
        "linear_num_key_heads": GDN_HEADS,
        "linear_num_value_heads": GDN_HEADS,
        "linear_value_head_dim": GDN_HEAD_DIM,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": FULL_Q_HEADS,
        "num_hidden_layers": 24,
        "num_key_value_heads": FULL_KV_HEADS,
        "vocab_size": VOCAB_SIZE,
        "rms_norm_eps": EPSILON,
        "tie_word_embeddings": True,
        "use_cache": True,
        "mamba_ssm_dtype": "float32",
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "default",
            "rope_theta": int(ROPE_THETA),
            "partial_rotary_factor": 0.25,
        },
        "layer_types": list(EXPECTED_LAYER_TYPES),
    }
    if root_contract != expected_root_contract:
        raise RuntimeError(f"pinned root config changed: {root_contract!r}")
    if text_contract != expected_text_contract:
        raise RuntimeError(f"pinned text config changed: {text_contract!r}")
    if max_layers < 1 or max_layers > len(EXPECTED_LAYER_TYPES):
        raise ValueError(
            f"max-layers must be in [1, {len(EXPECTED_LAYER_TYPES)}]"
        )

    index = json.loads(
        (MODEL_DIR / "model.safetensors.index.json").read_text()
    )
    if index.get("metadata") != {"total_size": 1746882752}:
        raise RuntimeError(
            f"checkpoint index metadata mismatch: {index.get('metadata')!r}"
        )
    weight_map = index.get("weight_map", {})
    selected_names = selected_tensor_names(EXPECTED_LAYER_TYPES)
    if len(selected_names) != EXPECTED_SELECTED_TENSOR_COUNT:
        raise RuntimeError(
            f"selected tensor contract count mismatch: {len(selected_names)}"
        )
    bad_mappings = {
        name: weight_map.get(name)
        for name in selected_names
        if weight_map.get(name) != MODEL_FILENAME
    }
    if bad_mappings:
        raise RuntimeError(
            f"selected tensor index mapping mismatch: {bad_mappings!r}"
        )
    independent_lm_heads = sorted(
        name for name in weight_map if name.endswith("lm_head.weight")
    )
    if independent_lm_heads:
        raise RuntimeError(
            f"tied checkpoint contains independent LM head weights: {independent_lm_heads!r}"
        )

    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    if (
        manifest.get("model_id") != MODEL_ID
        or manifest.get("revision") != MODEL_REVISION
        or manifest.get("files", {}).get(MODEL_FILENAME)
        != EXPECTED_ARTIFACTS[MODEL_FILENAME]
        or manifest.get("files", {}).get("config.json")
        != EXPECTED_ARTIFACTS["config.json"]
        or manifest.get("files", {}).get("model.safetensors.index.json")
        != EXPECTED_ARTIFACTS["model.safetensors.index.json"]
    ):
        raise RuntimeError("checkpoint manifest provenance contract mismatch")

    selected_names_json = json.dumps(
        selected_names, separators=(",", ":")
    ).encode()
    provenance = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "artifact_hashes": artifact_hashes,
        "root_config_contract": root_contract,
        "text_config_contract": text_contract,
        "config_contract_verified": True,
        "manifest_provenance_verified": True,
        "index_metadata_verified": True,
        "selected_tensor_mapping_verified": True,
        "selected_tensor_count": len(selected_names),
        "selected_tensor_names": list(selected_names),
        "selected_tensor_names_sha256": hashlib.sha256(
            selected_names_json
        ).hexdigest(),
        "selected_tensor_single_shard": MODEL_FILENAME,
        "independent_lm_head_present": False,
    }
    return EXPECTED_LAYER_TYPES, provenance


def write_exclusive(path: Path, content: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOSYS,
            "renameat2 is unavailable; refusing non-atomic model output publish",
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    ) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def publish_output_directory(
    output_dir: Path,
    final_state_bytes: bytes,
    metadata_bytes: bytes,
) -> None:
    try:
        os.lstat(output_dir)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(
            errno.EEXIST, "model output directory already exists", output_dir
        )

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        write_exclusive(temporary_dir / "final_state.safetensors", final_state_bytes)
        write_exclusive(temporary_dir / "execution_metadata.json", metadata_bytes)
        fsync_directory(temporary_dir)
        rename_noreplace(temporary_dir, output_dir)
        temporary_dir = None
        fsync_directory(output_dir.parent)
    finally:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir)


def save_outputs_exclusive(
    output_dir: Path,
    final_hidden: torch.Tensor,
    final_residual: torch.Tensor,
    comparison_final_norm: torch.Tensor | None,
    payload: dict,
) -> dict:
    resolved_parent = output_dir.parent.resolve(strict=True)
    resolved_output_dir = resolved_parent / output_dir.name
    final_state_path = resolved_output_dir / "final_state.safetensors"
    final_state_tensors = {
        "final_hidden": final_hidden.detach().clone().contiguous(),
        "final_residual": final_residual.detach().clone().contiguous(),
    }
    if comparison_final_norm is not None:
        final_state_tensors["nvidia_comparison_final_norm"] = (
            comparison_final_norm.detach().clone().contiguous()
        )
    final_state_bytes = serialize_safetensors(
        final_state_tensors,
        metadata={
            "schema": "amdgpu-sim.qwen35-model-final-state.v1",
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "token_id": str(payload["token_id"]),
            "layers_executed": str(payload["layers_executed"]),
            "correctness_verified": str(
                payload["correctness_verified"]
            ).lower(),
        },
    )
    final_state_record = {
        "path": str(final_state_path),
        "bytes": len(final_state_bytes),
        "sha256": hashlib.sha256(final_state_bytes).hexdigest(),
        "tensor_keys": list(final_state_tensors),
    }

    metadata_path = resolved_output_dir / "execution_metadata.json"
    persisted_payload = {
        **payload,
        "output_artifacts": {
            "write_policy": ATOMIC_OUTPUT_WRITE_POLICY,
            "final_state": final_state_record,
            "execution_metadata": {"path": str(metadata_path)},
        },
    }
    metadata_bytes = (
        json.dumps(persisted_payload, indent=2, sort_keys=True) + "\n"
    ).encode()
    publish_output_directory(
        resolved_output_dir,
        final_state_bytes,
        metadata_bytes,
    )
    return {
        "write_policy": ATOMIC_OUTPUT_WRITE_POLICY,
        "directory": str(resolved_output_dir),
        "final_state": final_state_record,
        "execution_metadata": {
            "path": str(metadata_path),
            "bytes": len(metadata_bytes),
            "sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "contains_layer_records": True,
        },
    }


def run_teacher_forced_layer(
    layer: int,
    token_id: int,
    layer_types: tuple[str, ...],
    checkpoint_provenance: dict,
    nvidia_golden: dict,
) -> int:
    if layer < 0 or layer >= len(layer_types):
        raise ValueError(f"teacher-forced layer is out of range: {layer}")
    source_tensor_hashes: dict[str, str] = {}
    assembled_weight_hashes: dict[str, str] = {}
    with safe_open(MODEL_FILE, framework="pt", device="cpu") as tensors:
        if layer == 0:
            embedding = tensors.get_slice(EMBEDDING_WEIGHT_NAME)
            hidden = embedding[token_id : token_id + 1].clone().contiguous()
            residual = None
            input_source = "checkpoint_embedding_row"
        else:
            hidden = nvidia_golden["tensors"][
                f"layers.{layer - 1}.returned_hidden"
            ].clone().contiguous()
            residual = nvidia_golden["tensors"][
                f"layers.{layer - 1}.returned_residual"
            ].clone().contiguous()
            input_source = f"nvidia_golden_layer_{layer - 1}_returned_pair"
        hidden_before = hidden.clone()
        residual_before = residual.clone() if residual is not None else None
        if layer_types[layer] == "linear_attention":
            actual_hidden, actual_residual, state = run_gdn_layer(
                tensors,
                layer,
                hidden,
                residual,
                source_tensor_hashes,
                assembled_weight_hashes,
            )
        else:
            if residual is None:
                raise RuntimeError("full-attention teacher input has no residual")
            actual_hidden, actual_residual, state = run_full_attention_layer(
                tensors,
                layer,
                hidden,
                residual,
                source_tensor_hashes,
                assembled_weight_hashes,
            )
    comparisons = {
        "returned_hidden": compare_nvidia_golden_tensor(
            f"layers.{layer}.returned_hidden",
            actual_hidden,
            nvidia_golden["tensors"][f"layers.{layer}.returned_hidden"],
            "returned_hidden",
        ),
        "returned_residual": compare_nvidia_golden_tensor(
            f"layers.{layer}.returned_residual",
            actual_residual,
            nvidia_golden["tensors"][f"layers.{layer}.returned_residual"],
            "returned_residual",
        ),
    }
    finite = bool(
        torch.all(torch.isfinite(actual_hidden.float())).item()
        and torch.all(torch.isfinite(actual_residual.float())).item()
    )
    inputs_unchanged = bool(
        torch.equal(hidden, hidden_before)
        and (
            residual is None
            or torch.equal(residual, residual_before)
        )
    )
    correct = bool(
        finite
        and inputs_unchanged
        and all(item["correct"] for item in comparisons.values())
    )
    payload = {
        "schema": "amdgpu-sim.triton-qwen35-teacher-forced-layer.v1",
        "backend": "gemsim_amd",
        "arch": "gfx950",
        "model": MODEL_ID,
        "token_id": token_id,
        "layer": layer,
        "layer_type": layer_types[layer],
        "input_source": input_source,
        "input_hidden_sha256": tensor_sha256(hidden_before),
        "input_residual_sha256": (
            tensor_sha256(residual_before)
            if residual_before is not None
            else None
        ),
        "inputs_unchanged": inputs_unchanged,
        "comparisons": comparisons,
        "state": state,
        "all_values_finite": finite,
        "source_tensor_hashes": source_tensor_hashes,
        "assembled_weight_hashes": assembled_weight_hashes,
        "checkpoint_provenance": checkpoint_provenance,
        "nvidia_golden": {
            "metadata_sha256": nvidia_golden["metadata_sha256"],
            "results_sha256": nvidia_golden["results_sha256"],
            "role": "teacher_input_and_external_output_oracle_only",
        },
        "fallback_count": 0,
        "cpu_fallback_count": 0,
        "nvidia_fallback_count": 0,
        "output_correct": correct,
        "claim_boundary": (
            "isolates one layer from accumulated upstream cross-backend "
            "BF16 drift; it is not a sequential model execution"
        ),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if correct else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the pinned Qwen3.5-0.8B first-token backbone and record "
            "execution evidence; verify model correctness when an accepted "
            "NVIDIA golden is supplied"
        )
    )
    parser.add_argument("--token-id", type=int, default=248044)
    parser.add_argument("--max-layers", type=int, default=24)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "create a new directory and exclusively save final state plus "
            "per-layer execution metadata"
        ),
    )
    parser.add_argument(
        "--nvidia-golden-dir",
        type=Path,
        help=(
            "accepted independent NVIDIA 4-layer or 24-layer golden "
            "directory containing metadata.json and results.safetensors"
        ),
    )
    parser.add_argument(
        "--teacher-forced-layer",
        type=int,
        help=(
            "run exactly one layer with the accepted NVIDIA previous-layer "
            "hidden/residual pair; requires the 24-layer NVIDIA golden"
        ),
    )
    args = parser.parse_args()
    if (
        args.nvidia_golden_dir is not None
        and args.max_layers not in NVIDIA_GOLDEN_ACCEPTED_SHA256
    ):
        parser.error("--nvidia-golden-dir requires --max-layers 4 or 24")
    if args.teacher_forced_layer is not None:
        if args.nvidia_golden_dir is None or args.max_layers != 24:
            parser.error(
                "--teacher-forced-layer requires --max-layers 24 and "
                "--nvidia-golden-dir"
            )
        if args.output_dir is not None:
            parser.error("--teacher-forced-layer does not publish model output")
    target = triton.runtime.driver.active.get_current_target()
    if (target.backend, target.arch) != ("gemsim_amd", "gfx950"):
        raise RuntimeError(f"unexpected Triton execution target: {target}")
    if DEVICE.type != "cpu":
        raise RuntimeError(f"unexpected staging device: {DEVICE}")
    if args.token_id < 0 or args.token_id >= VOCAB_SIZE:
        raise ValueError(f"token ID is out of range: {args.token_id}")
    layer_types, checkpoint_provenance = validate_checkpoint(args.max_layers)
    nvidia_golden = (
        load_nvidia_golden(
            args.nvidia_golden_dir,
            args.max_layers,
            args.token_id,
            checkpoint_provenance,
        )
        if args.nvidia_golden_dir is not None
        else None
    )
    if args.teacher_forced_layer is not None:
        if nvidia_golden is None:
            raise RuntimeError("teacher-forced NVIDIA golden is unavailable")
        return run_teacher_forced_layer(
            args.teacher_forced_layer,
            args.token_id,
            layer_types,
            checkpoint_provenance,
            nvidia_golden,
        )

    layer_records = []
    nvidia_layer_comparisons = []
    source_tensor_hashes = {}
    assembled_weight_hashes = {}
    with safe_open(MODEL_FILE, framework="pt", device="cpu") as tensors:
        embedding = tensors.get_slice(EMBEDDING_WEIGHT_NAME)
        if tuple(embedding.get_shape()) != (VOCAB_SIZE, HIDDEN_SIZE):
            raise RuntimeError("embedding shape mismatch")
        hidden = embedding[args.token_id : args.token_id + 1].clone().contiguous()
        if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (1, HIDDEN_SIZE):
            raise RuntimeError(
                f"embedding row contract mismatch: {hidden.dtype} {tuple(hidden.shape)}"
            )
        if hidden.untyped_storage().nbytes() != hidden.numel() * hidden.element_size():
            raise RuntimeError("embedding row does not own exact storage")
        source_tensor_hashes[
            f"{EMBEDDING_WEIGHT_NAME}[row={args.token_id}]"
        ] = tensor_sha256(hidden)
        if (
            nvidia_golden is not None
            and tensor_descriptor(hidden)
            != nvidia_golden["metadata"].get("input", {}).get("hidden")
        ):
            raise RuntimeError("NVIDIA golden hidden input identity mismatch")
        residual = None
        input_sha256 = tensor_sha256(hidden)
        for layer, layer_type in enumerate(layer_types[: args.max_layers]):
            if layer_type == "linear_attention":
                hidden, residual, state = run_gdn_layer(
                    tensors,
                    layer,
                    hidden,
                    residual,
                    source_tensor_hashes,
                    assembled_weight_hashes,
                )
            else:
                if residual is None:
                    raise RuntimeError(
                        "full-attention layer cannot start without residual"
                    )
                hidden, residual, state = run_full_attention_layer(
                    tensors,
                    layer,
                    hidden,
                    residual,
                    source_tensor_hashes,
                    assembled_weight_hashes,
                )
            finite = bool(
                torch.all(torch.isfinite(hidden.to(torch.float32))).item()
                and torch.all(torch.isfinite(residual.to(torch.float32))).item()
            )
            if not finite:
                raise RuntimeError(f"nonfinite output after layer {layer}")
            if nvidia_golden is not None:
                nvidia_layer_comparisons.append(
                    {
                        "layer": layer,
                        "type": layer_type,
                        "returned_hidden": compare_nvidia_golden_tensor(
                            f"layers.{layer}.returned_hidden",
                            hidden,
                            nvidia_golden["tensors"][
                                f"layers.{layer}.returned_hidden"
                            ],
                            "returned_hidden",
                        ),
                        "returned_residual": compare_nvidia_golden_tensor(
                            f"layers.{layer}.returned_residual",
                            residual,
                            nvidia_golden["tensors"][
                                f"layers.{layer}.returned_residual"
                            ],
                            "returned_residual",
                        ),
                    }
                )
            layer_records.append(
                {
                    "layer": layer,
                    "type": layer_type,
                    "hidden_sha256": tensor_sha256(hidden),
                    "residual_sha256": tensor_sha256(residual),
                    "finite": finite,
                    **state,
                }
            )

        final_hidden = hidden
        final_residual = residual
        final_norm_applied = args.max_layers == len(layer_types)
        comparison_final_norm = None
        if final_norm_applied:
            final_weight = load_tensor(
                tensors,
                "model.language_model.norm.weight",
                torch.bfloat16,
                (HIDDEN_SIZE,),
                source_tensor_hashes,
            )
            normalized = torch.empty_like(hidden)
            ignored_residual = torch.empty_like(residual)
            fused_residual_rms_norm(
                hidden,
                residual,
                final_weight,
                normalized,
                ignored_residual,
                1,
            )
            if nvidia_golden is not None:
                comparison_final_norm = normalized
            final_hidden = normalized
            final_residual = ignored_residual

    nvidia_final_norm_comparison = None
    if nvidia_golden is not None:
        expected_source_hashes = nvidia_golden["metadata"].get(
            "selected_checkpoint_tensor_sha256"
        )
        observed_source_hashes = dict(source_tensor_hashes)
        observed_embedding_name = (
            f"{EMBEDDING_WEIGHT_NAME}[row={args.token_id}]"
        )
        expected_embedding_name = f"{EMBEDDING_WEIGHT_NAME}[{args.token_id}]"
        observed_source_hashes[expected_embedding_name] = (
            observed_source_hashes.pop(observed_embedding_name)
        )
        if observed_source_hashes != expected_source_hashes:
            raise RuntimeError(
                "NVIDIA golden selected checkpoint tensor hashes mismatch"
            )
        if args.max_layers == 24:
            if comparison_final_norm is None:
                raise RuntimeError("NVIDIA golden final norm was not executed")
            nvidia_final_norm_comparison = compare_nvidia_golden_tensor(
                "final_norm",
                comparison_final_norm,
                nvidia_golden["tensors"]["final_norm"],
                "final_norm",
            )

    final_state_finite = bool(
        torch.all(torch.isfinite(final_hidden.to(torch.float32))).item()
        and torch.all(torch.isfinite(final_residual.to(torch.float32))).item()
    )
    if not final_state_finite:
        raise RuntimeError("nonfinite final model state")

    if nvidia_golden is None:
        nvidia_golden_correct = None
        nvidia_golden_payload = {
            "enabled": False,
            "role": "external_oracle_not_provided",
        }
    else:
        golden_comparisons = [
            comparison
            for layer_comparison in nvidia_layer_comparisons
            for comparison in (
                layer_comparison["returned_hidden"],
                layer_comparison["returned_residual"],
            )
        ]
        if nvidia_final_norm_comparison is not None:
            golden_comparisons.append(nvidia_final_norm_comparison)
        nvidia_golden_correct = all(
            comparison["correct"] for comparison in golden_comparisons
        )
        golden_metadata = nvidia_golden["metadata"]
        nvidia_golden_payload = {
            "enabled": True,
            "role": (
                "external_host_oracle_excluded_from_target_execution_and_"
                "fallback_counts"
            ),
            "fallback_counters_include_oracle": False,
            "directory": nvidia_golden["directory"],
            "schema": golden_metadata["schema"],
            "kind": golden_metadata["kind"],
            "metadata_sha256": nvidia_golden["metadata_sha256"],
            "results_sha256": nvidia_golden["results_sha256"],
            "generator": golden_metadata["script"],
            "gpu": golden_metadata["environment"]["gpu"],
            "comparison_count": len(golden_comparisons),
            "layer_comparisons": nvidia_layer_comparisons,
            "final_norm": nvidia_final_norm_comparison,
            "mismatch_count": sum(
                comparison["mismatch_count"]
                for comparison in golden_comparisons
            ),
            "nonfinite_count": sum(
                comparison["nonfinite_count"]
                for comparison in golden_comparisons
            ),
            "all_comparisons_correct": nvidia_golden_correct,
        }

    payload = {
        "schema": "amdgpu-sim.triton-qwen35-model-backbone-decode.v1",
        "backend": target.backend,
        "arch": target.arch,
        "model": MODEL_ID,
        "mode": "checkpoint_first_token_empty_cache",
        "token_id": args.token_id,
        "layers_executed": args.max_layers,
        "checkpoint_provenance": checkpoint_provenance,
        "layer_records": layer_records,
        "selected_tensor_hashes": source_tensor_hashes,
        "selected_tensor_hash_count": len(source_tensor_hashes),
        "selected_tensor_hash_scope": (
            "raw bytes of tensors loaded by this execution; the embedding "
            "entry covers only the selected token row"
        ),
        "assembled_weight_hashes": assembled_weight_hashes,
        "assembled_weight_hash_count": len(assembled_weight_hashes),
        "assembled_weight_hash_scope": (
            "raw bytes of concatenated or dtype-transformed weights passed "
            "to kernels in this execution"
        ),
        "input_sha256": input_sha256,
        "final_norm_applied": final_norm_applied,
        "final_hidden_sha256": tensor_sha256(final_hidden),
        "final_residual_sha256": tensor_sha256(final_residual),
        "nvidia_comparison_final_norm_sha256": (
            tensor_sha256(comparison_final_norm)
            if comparison_final_norm is not None
            else None
        ),
        "all_layer_states_finite": all(
            record["finite"] for record in layer_records
        ),
        "final_state_finite": final_state_finite,
        "all_finite": (
            all(record["finite"] for record in layer_records)
            and final_state_finite
        ),
        "fallback_count": 0,
        "cpu_fallback_count": 0,
        "nvidia_fallback_count": 0,
        "execution_complete": True,
        "requested_layer_range_complete": True,
        "full_24_layer_backbone_complete": args.max_layers == 24,
        "execution_status": (
            "completed_with_external_nvidia_golden"
            if nvidia_golden is not None
            else "completed_without_model_golden"
        ),
        "correctness_verified": nvidia_golden_correct is True,
        "correctness_status": (
            "verified_against_external_nvidia_golden"
            if nvidia_golden_correct is True
            else (
                "failed_external_nvidia_golden"
                if nvidia_golden_correct is False
                else "not_verified_no_external_golden"
            )
        ),
        "golden_provided": nvidia_golden is not None,
        "golden_compared": nvidia_golden is not None,
        "nvidia_golden": nvidia_golden_payload,
        "scope": {
            "in_scope": [
                "checkpoint text backbone",
                "one first-token decode step",
                "position zero",
                "empty per-layer convolution, recurrent, and KV caches",
                f"layers zero through {args.max_layers - 1}",
                "final RMSNorm only when all 24 layers execute",
                "execution and finite-value evidence",
            ],
            "out_of_scope": [
                "prefill",
                "multi-token decode",
                "nonempty cache history",
                "vision path",
                "tied LM head and logits",
                "sampling",
                "performance claims",
                "model-level numerical correctness without external golden",
            ],
        },
        "output_correct": nvidia_golden_correct,
    }
    payload["output_artifacts"] = (
        save_outputs_exclusive(
            args.output_dir,
            final_hidden,
            final_residual,
            comparison_final_norm,
            payload,
        )
        if args.output_dir is not None
        else None
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if nvidia_golden_correct is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
