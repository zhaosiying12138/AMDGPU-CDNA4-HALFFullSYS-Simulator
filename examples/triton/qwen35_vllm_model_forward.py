#!/usr/bin/env python3
"""Run one checkpoint-backed Qwen3.5 token through the formal vLLM plugin."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import uuid


HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE / "_gemsim_bootstrap.py"))["bootstrap"](
    __file__, "qwen35-vllm-model-forward"
)
# Python 3.14 isolated mode intentionally omits the script directory.  Trust
# only this resolved, repository-owned directory for the sibling debug helpers;
# never inherit an ambient PYTHONPATH.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch
import triton
from safetensors import safe_open
from safetensors.torch import save as save_safetensors

import gemsim_vllm
from vllm import ModelRegistry
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


ROOT = HERE.parents[1]
MODEL_DIR = ROOT / "models/Qwen3.5-0.8B"
MODEL_FILE = MODEL_DIR / "model.safetensors-00001-of-00001.safetensors"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TOKEN_ID = 248044
EXPECTED_ARTIFACTS = {
    "config.json": (
        2907,
        "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    ),
    "model.safetensors.index.json": (
        50900,
        "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
    ),
    MODEL_FILE.name: (
        1746942600,
        "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    ),
    "manifest.json": (
        1008,
        "de2281cc73a1329d13245cb9658be910cf435e72c4ea0277c4f8811a24edf762",
    ),
}
PREFILL_GOLDEN_DIR = (
    ROOT / "artifacts/qwen35-nvidia-golden/20260812-prefill2-max24-v1"
)
PREFILL_GOLDEN_FILES = {
    "metadata.json": (
        153970,
        "ff326833bc2a47f760c240af5f441f48a187e686e523a4c0778185ce392d2251",
    ),
    "results.safetensors": (
        21284456,
        "c401c34db3f137ad4b2e371b32e3bcc9c796ba56a2101d50e7e8fc927091cbe7",
    ),
}
PREFILL_GOLDEN_SCRIPT_SHA256 = (
    "32706a3f9235c6279c74adba136e813243b8aadaad3001d2f0b3c038f675506a"
)
PREFILL_GOLDEN_SCHEMA = "amdgpu-sim.qwen35-nvidia-prefill-golden.v1"
PREFILL_TOKEN_IDS = [248044, 266]
DECODE_WINDOW_GOLDEN_DIR = (
    ROOT / "artifacts/qwen35-nvidia-golden/20260812-decode4-max24-v1"
)
DECODE_WINDOW_GOLDEN_FILES = {
    "metadata.json": (
        154152,
        "8df6b70919203c7bd0369db8191b231abe4bf36e026b7c68fb305619ece65a66",
    ),
    "results.safetensors": (
        21497464,
        "60e9ba36e659d5ef93297e3beb4e056558dd5fdabdc4bbf2b6b3a65b9fb3210f",
    ),
}
DECODE_WINDOW_TOKEN_IDS = [248044, 266, 27841, 27841]
INFERENCE_MODES = (
    "production",
    "debug-layer-diff",
    "debug-resume",
    "evidence",
)
NVIDIA_ORACLE_SCRIPT = ROOT / "tools/qwen35_nvidia_layer_oracle.py"
NVIDIA_ORACLE_PYTHON = Path(
    "/home/zhaosiying/miniforge3/envs/triton-dev/bin/python3"
)
PLUGIN_SOURCE_PATHS = tuple(
    ROOT / relative
    for relative in (
        "plugins/framework/gemsim_vllm/src/gemsim_vllm/adapters.py",
        "plugins/framework/gemsim_vllm/src/gemsim_vllm/attention.py",
        "plugins/framework/gemsim_vllm/src/gemsim_vllm/kernels.py",
        "plugins/framework/gemsim_vllm/src/gemsim_vllm/model.py",
        "plugins/framework/gemsim_vllm/src/gemsim_vllm/ops.py",
    )
)
FORMULA_SOURCE_PATHS = tuple(
    ROOT / relative
    for relative in (
        "projects/vllm/vllm/model_executor/layers/layernorm.py",
        "projects/vllm/vllm/model_executor/models/qwen3_5.py",
        "projects/vllm/vllm/model_executor/models/qwen3_next.py",
        (
            "projects/vllm/vllm/model_executor/layers/mamba/gdn/"
            "qwen_gdn_linear_attn.py"
        ),
    )
)


class IdentityFinalNorm(torch.nn.Module):
    """Expose a decoder layer's returned hidden tensor for bounded diagnosis."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states, residual


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(value)).hexdigest()


def layer0_parameter_hashes(model) -> dict[str, dict[str, object]]:
    records = {}
    for name, value in model.named_parameters():
        if name.startswith("model.layers.0.") or name == "model.norm.weight":
            records[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_sha256(value),
            }
    embedding = model.model.embed_tokens.weight[TOKEN_ID : TOKEN_ID + 1]
    records[f"model.embed_tokens.weight[row={TOKEN_ID}]"] = {
        "shape": list(embedding.shape),
        "dtype": str(embedding.dtype),
        "sha256": tensor_sha256(embedding),
    }
    return records


def validate_checkpoint() -> dict[str, object]:
    observed = {}
    for name, (expected_bytes, expected_sha256) in EXPECTED_ARTIFACTS.items():
        path = MODEL_DIR / name
        record = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        if record != {"bytes": expected_bytes, "sha256": expected_sha256}:
            raise RuntimeError(f"checkpoint artifact mismatch: {name}")
        observed[name] = record
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    if (
        manifest.get("model_id") != MODEL_ID
        or manifest.get("revision") != MODEL_REVISION
    ):
        raise RuntimeError("checkpoint manifest identity mismatch")
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "artifacts": observed,
    }


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def git_object_id(repository: Path, revision: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", revision],
        text=True,
    ).strip()


def source_set_identity(paths: tuple[Path, ...]) -> dict[str, str]:
    records = {}
    for path in paths:
        resolved = path.resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file():
            raise RuntimeError(f"identity source is unsafe: {path}")
        records[str(resolved.relative_to(ROOT))] = file_sha256(resolved)
    return records


def load_prefix_manifest() -> tuple[Path, dict[str, object]]:
    prefix = Path(os.environ["ROCM_SIM_ROOT"]).resolve(strict=True)
    manifest_path = prefix / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("prefix manifest is unsafe")
    manifest = json.loads(manifest_path.read_text())
    return manifest_path, manifest


def exact_execution_identity(
    *,
    architecture: str,
    checkpoint: dict[str, object],
    layer_types: list[str],
    conv_shape: tuple[int, ...],
    recurrent_shape: tuple[int, ...],
    all_source_names: list[str],
    source_names: list[str],
    loaded_count: int,
    world_size: int = 1,
    rank: int = 0,
    kv_cache_shape: tuple[int, ...] = (1, 16, 2, 512),
) -> dict[str, object]:
    """Bind restart state to every implementation and weight input."""

    manifest_path, manifest = load_prefix_manifest()
    artifacts = manifest.get("artifacts")
    managed_inputs = manifest.get("managed_inputs")
    if not isinstance(artifacts, dict) or not isinstance(managed_inputs, dict):
        raise RuntimeError("prefix manifest artifact tables are missing")
    runtime_record = artifacts.get("runtime_library")
    gem5_record = managed_inputs.get("gem5_binary")
    if not isinstance(runtime_record, dict) or not isinstance(gem5_record, dict):
        raise RuntimeError("prefix runtime/gem5 identities are missing")
    runtime_path = Path(runtime_record.get("path", "")).resolve(strict=True)
    gem5_path = Path(gem5_record.get("path", "")).resolve(strict=True)
    if not runtime_path.is_file() or not gem5_path.is_file():
        raise RuntimeError("prefix runtime/gem5 identity path is invalid")

    vllm_repository = ROOT / "projects/vllm"
    vllm_head = git_object_id(vllm_repository, "HEAD")
    vllm_tree = git_object_id(vllm_repository, "HEAD^{tree}")
    formula_sources = source_set_identity(FORMULA_SOURCE_PATHS)
    plugin_sources = source_set_identity(PLUGIN_SOURCE_PATHS)
    runner_path = Path(__file__).resolve(strict=True)
    return {
        "model": {
            "id": checkpoint["model_id"],
            "revision": checkpoint["revision"],
            "artifacts": checkpoint["artifacts"],
        },
        "implementation": {
            "architecture": architecture,
            "runner_sha256": file_sha256(runner_path),
            "plugin_sha256": canonical_sha256(plugin_sources),
            "vllm_git_head": vllm_head,
            "vllm_tree_sha256": canonical_sha256(
                {
                    "git_tree": vllm_tree,
                    "formula_sources": formula_sources,
                }
            ),
            "gem5_binary_sha256": file_sha256(gem5_path),
            "runtime_dso_sha256": file_sha256(runtime_path),
            "prefix_manifest_sha256": file_sha256(manifest_path),
        },
        "target": {
            "backend": "gemsim_amd",
            "arch": "gfx950",
            "device": "cpu",
            "fallback_allowed": False,
            "stochastic_ops": False,
        },
        "decoder": {
            "layer_count": 24,
            "hidden_size": 1024,
            "activation_dtype": "torch.bfloat16",
            "layer_types": list(layer_types),
            "gdn_conv_state_shape": [1, *conv_shape],
            "gdn_recurrent_state_shape": [1, *recurrent_shape],
            "kv_cache_shape": list(kv_cache_shape),
            "gdn_conv_state_dtype": "torch.bfloat16",
            "gdn_recurrent_state_dtype": "torch.float32",
            "kv_cache_dtype": "torch.bfloat16",
        },
        "parallelism": {
            "world_size": world_size,
            "rank": rank,
            "tensor_parallel_size": world_size,
            "pipeline_parallel_size": 1,
        },
        "weights": {
            "checkpoint_tensor_count": len(all_source_names),
            "source_tensor_count": len(source_names),
            "loaded_tensor_count": loaded_count,
            "source_names_sha256": canonical_sha256(sorted(source_names)),
        },
    }


def request_identity(metadata: dict, slot_mapping: dict) -> dict[str, object]:
    scheduler = {
        "metadata_names": sorted(metadata),
        "slot_mapping": {
            name: value.detach().cpu().tolist()
            for name, value in sorted(slot_mapping.items())
        },
        "num_tokens": 2,
        "cache_mode": "empty_per_layer_then_mutated_in_order",
    }
    return {
        "sequence_id": "qwen35-prefill2-248044-266",
        "phase": "prefill",
        "step_index": 0,
        "context_length_before": 0,
        "context_length_after": 2,
        "scheduler": scheduler,
    }


@contextmanager
def distributed_world(
    vllm_config: VllmConfig,
    *,
    world_size: int,
    rank: int,
    distributed_init_method: str | None,
):
    created_store: Path | None = None
    if distributed_init_method is None:
        fd, local_store = tempfile.mkstemp(
            prefix="qwen35-vllm-forward-",
            suffix=".store",
            dir=os.environ["TMPDIR"],
        )
        os.close(fd)
        created_store = Path(local_store)
        distributed_init_method = created_store.as_uri()
    initialized = False
    try:
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                distributed_init_method=distributed_init_method,
                local_rank=rank,
                backend="gloo",
            )
            initialize_model_parallel(world_size, 1, backend="gloo")
            initialized = True
            yield
    finally:
        if initialized:
            destroy_model_parallel()
            destroy_distributed_environment()
        if created_store is not None:
            created_store.unlink(missing_ok=True)


def load_reference(path: Path, key: str) -> tuple[torch.Tensor, dict[str, object]]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise RuntimeError("reference safetensors must not be a symlink")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError("reference safetensors is not a regular file")
    with safe_open(resolved, framework="pt", device="cpu") as tensors:
        value = tensors.get_tensor(key).clone().contiguous()
    if value.dtype != torch.bfloat16 or value.shape != (1, 1024):
        raise RuntimeError("reference tensor must be BF16 [1,1024]")
    return value, {
        "path": str(resolved),
        "file_sha256": file_sha256(resolved),
        "tensor_key": key,
        "tensor_sha256": tensor_sha256(value),
    }


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    actual_float = actual.float()
    expected_float = expected.float()
    error = torch.abs(actual_float - expected_float)
    finite = (
        torch.isfinite(actual_float)
        & torch.isfinite(expected_float)
        & torch.isfinite(error)
    )
    mismatch = (~finite) | (error > 0.125 + 0.03 * torch.abs(expected_float))
    expected_norm = float(torch.linalg.vector_norm(expected_float).item())
    relative_l2 = float(torch.linalg.vector_norm(error).item()) / max(
        expected_norm, 1.0e-30
    )
    return {
        "actual_sha256": tensor_sha256(actual),
        "expected_sha256": tensor_sha256(expected),
        "mismatch_count": int(torch.count_nonzero(mismatch).item()),
        "nonfinite_count": int(torch.count_nonzero(~finite).item()),
        "max_abs_error": float(torch.where(finite, error, 0.0).max().item()),
        "relative_l2_error": relative_l2,
        "atol": 0.125,
        "rtol": 0.03,
        "max_relative_l2": 0.03,
        "correct": bool(
            torch.count_nonzero(mismatch).item() == 0 and relative_l2 <= 0.03
        ),
    }


def compare_cross_arch(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, object]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(
            "cross-architecture tensor contract mismatch: "
            f"actual={actual.dtype}{tuple(actual.shape)}, "
            f"expected={expected.dtype}{tuple(expected.shape)}"
        )
    actual_float = actual.float()
    expected_float = expected.float()
    error = torch.abs(actual_float - expected_float)
    finite = (
        torch.isfinite(actual_float)
        & torch.isfinite(expected_float)
        & torch.isfinite(error)
    )
    atol = 0.25
    rtol = 0.15
    relative_l2_limit = 0.15
    mismatch = (~finite) | (error > atol + rtol * torch.abs(expected_float))
    actual_norm = float(torch.linalg.vector_norm(actual_float).item())
    expected_norm = float(torch.linalg.vector_norm(expected_float).item())
    relative_l2 = float(torch.linalg.vector_norm(error).item()) / max(
        expected_norm, 1.0e-30
    )
    cosine = float(
        torch.sum(actual_float * expected_float).item()
        / max(actual_norm * expected_norm, 1.0e-30)
    )
    mismatch_count = int(torch.count_nonzero(mismatch).item())
    nonfinite_count = int(torch.count_nonzero(~finite).item())
    return {
        "actual_sha256": tensor_sha256(actual),
        "expected_sha256": tensor_sha256(expected),
        "mismatch_count": mismatch_count,
        "nonfinite_count": nonfinite_count,
        "max_abs_error": float(torch.where(finite, error, 0.0).max().item()),
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
        "atol": atol,
        "rtol": rtol,
        "max_relative_l2": relative_l2_limit,
        "correct": bool(
            mismatch_count == 0
            and nonfinite_count == 0
            and relative_l2 <= relative_l2_limit
            and cosine >= 0.98
        ),
    }


def compare_with_limits(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float = 0.03,
    max_relative_l2: float = 0.03,
) -> dict[str, object]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(
            "diagnostic tensor contract mismatch: "
            f"actual={actual.dtype}{tuple(actual.shape)}, "
            f"expected={expected.dtype}{tuple(expected.shape)}"
        )
    actual_float = actual.float()
    expected_float = expected.float()
    error = torch.abs(actual_float - expected_float)
    finite = (
        torch.isfinite(actual_float)
        & torch.isfinite(expected_float)
        & torch.isfinite(error)
    )
    mismatch = (~finite) | (error > atol + rtol * torch.abs(expected_float))
    expected_norm = float(torch.linalg.vector_norm(expected_float).item())
    actual_norm = float(torch.linalg.vector_norm(actual_float).item())
    relative_l2 = float(torch.linalg.vector_norm(error).item()) / max(
        expected_norm, 1.0e-30
    )
    cosine = float(
        torch.sum(actual_float * expected_float).item()
        / max(actual_norm * expected_norm, 1.0e-30)
    )
    mismatch_count = int(torch.count_nonzero(mismatch).item())
    nonfinite_count = int(torch.count_nonzero(~finite).item())
    return {
        "actual_sha256": tensor_sha256(actual),
        "expected_sha256": tensor_sha256(expected),
        "mismatch_count": mismatch_count,
        "nonfinite_count": nonfinite_count,
        "max_abs_error": float(torch.where(finite, error, 0.0).max().item()),
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
        "atol": atol,
        "rtol": rtol,
        "max_relative_l2": max_relative_l2,
        "correct": bool(
            mismatch_count == 0
            and nonfinite_count == 0
            and relative_l2 <= max_relative_l2
            and cosine >= 0.98
        ),
    }


def load_prefill_golden(
    path: Path,
    *,
    expected_token_ids: list[int] | None = None,
    expected_files: dict[str, tuple[int, str]] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if expected_token_ids is None:
        expected_token_ids = PREFILL_TOKEN_IDS
    if expected_files is None:
        expected_files = PREFILL_GOLDEN_FILES
    expected_positions = list(range(len(expected_token_ids)))
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise RuntimeError("prefill golden directory must not be a symlink")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_dir():
        raise RuntimeError("prefill golden path is not a directory")
    files = sorted(item.name for item in resolved.iterdir())
    if files != sorted(expected_files):
        raise RuntimeError(f"prefill golden file set mismatch: {files}")
    file_records = {}
    for name, (expected_bytes, expected_sha256) in expected_files.items():
        candidate = resolved / name
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"unsafe prefill golden file: {name}")
        observed_sha256 = file_sha256(candidate)
        if (
            candidate.stat().st_size != expected_bytes
            or observed_sha256 != expected_sha256
        ):
            raise RuntimeError(f"prefill golden hash mismatch: {name}")
        file_records[name] = {
            "bytes": candidate.stat().st_size,
            "sha256": observed_sha256,
        }
    metadata = json.loads((resolved / "metadata.json").read_text())
    if (
        metadata.get("schema") != PREFILL_GOLDEN_SCHEMA
        or metadata.get("kind")
        != "independent_torch_cuda_empty_cache_prefill_golden"
        or metadata.get("model", {}).get("id") != MODEL_ID
        or metadata.get("model", {}).get("revision") != MODEL_REVISION
        or metadata.get("case", {}).get("token_ids") != expected_token_ids
        or metadata.get("case", {}).get("positions") != expected_positions
        or metadata.get("case", {}).get("max_layers") != 24
        or metadata.get("case", {}).get("cache") != "empty_per_layer"
        or metadata.get("script", {}).get("sha256")
        != PREFILL_GOLDEN_SCRIPT_SHA256
        or metadata.get("results_file_sha256")
        != expected_files["results.safetensors"][1]
        or metadata.get("all_results_finite") is not True
    ):
        raise RuntimeError("prefill golden metadata contract mismatch")
    expected_checkpoint_artifacts = {
        name: {
            "expected": {"bytes": size, "sha256": sha256},
            "observed": {"bytes": size, "sha256": sha256},
        }
        for name, (size, sha256) in EXPECTED_ARTIFACTS.items()
    }
    if (
        metadata.get("model", {}).get("pinned_artifacts")
        != expected_checkpoint_artifacts
    ):
        raise RuntimeError("prefill golden checkpoint provenance mismatch")

    result_path = resolved / "results.safetensors"
    loaded = {}
    with safe_open(result_path, framework="pt", device="cpu") as tensors:
        provenance = json.loads(tensors.metadata().get("provenance", "null"))
        if (
            provenance.get("schema") != PREFILL_GOLDEN_SCHEMA
            or provenance.get("token_ids") != expected_token_ids
            or provenance.get("positions") != expected_positions
            or provenance.get("max_layers") != 24
        ):
            raise RuntimeError("prefill golden safetensors provenance mismatch")
        keys = list(tensors.keys())
        if sorted(keys) != sorted(metadata.get("results", {})):
            raise RuntimeError("prefill golden result key set mismatch")
        for key in keys:
            value = tensors.get_tensor(key).clone().contiguous()
            record = metadata["results"][key]
            if (
                list(value.shape) != record.get("shape")
                or value.dtype
                not in (torch.bfloat16, torch.float32)
                or tensor_sha256(value) != record.get("sha256")
                or not torch.all(torch.isfinite(value.float())).item()
            ):
                raise RuntimeError(f"prefill golden tensor mismatch: {key}")
            loaded[key] = value
    return loaded, {
        "path": str(resolved),
        "files": file_records,
        "schema": metadata["schema"],
        "script": metadata["script"],
        "gpu": metadata["environment"]["gpu"],
        "token_ids": expected_token_ids,
        "positions": expected_positions,
        "tensor_count": len(loaded),
    }


def bind_runtime_state(
    model,
    vllm_config: VllmConfig,
    *,
    tokens: int = 1,
    prefill: bool = False,
    start_position: int = 0,
    caches: list[tuple] | None = None,
):
    if not 1 <= tokens <= 16:
        raise ValueError("runtime state supports 1..16 tokens")
    if prefill and (tokens == 1 or start_position != 0):
        raise ValueError("prefill must contain 2..16 tokens from an empty cache")
    metadata = {}
    slot_mapping = {}
    if caches is None:
        caches = []
        allocate_caches = True
    else:
        if len(caches) != len(model.model.layers):
            raise ValueError("runtime cache count mismatch")
        allocate_caches = False
    state_index = torch.tensor([0], dtype=torch.int32)
    conv_shape, recurrent_shape = model.get_mamba_state_shape_from_config(
        vllm_config
    )
    layer_types = vllm_config.model_config.hf_text_config.layer_types
    for index, layer in enumerate(model.model.layers):
        if layer_types[index] == "linear_attention":
            gdn = layer.linear_attn
            if allocate_caches:
                conv_state = torch.zeros((1, *conv_shape), dtype=torch.bfloat16)
                recurrent_state = torch.zeros(
                    (1, *recurrent_shape), dtype=torch.float32
                )
                gdn.kv_cache = (conv_state, recurrent_state)
                caches.append((gdn.prefix, conv_state, recurrent_state))
            else:
                record = caches[index]
                if record[0] != gdn.prefix or len(record) != 3:
                    raise RuntimeError("GDN runtime cache identity mismatch")
                conv_state, recurrent_state = record[1:]
                gdn.kv_cache = (conv_state, recurrent_state)
            metadata[gdn.prefix] = GDNAttentionMetadata(
                num_prefills=1 if prefill else 0,
                num_prefill_tokens=tokens if prefill else 0,
                num_decodes=0 if prefill else 1,
                num_decode_tokens=0 if prefill else 1,
                num_spec_decodes=0,
                num_spec_decode_tokens=0,
                num_actual_tokens=tokens,
                non_spec_query_start_loc=torch.tensor(
                    [0, tokens], dtype=torch.int32
                ),
                non_spec_state_indices_tensor=state_index,
                prefill_query_start_loc=(
                    torch.tensor([0, tokens], dtype=torch.int32)
                    if prefill
                    else None
                ),
                prefill_state_indices=state_index if prefill else None,
                prefill_has_initial_state=(
                    torch.tensor([False], dtype=torch.bool) if prefill else None
                ),
            )
        else:
            attention = layer.self_attn.attn
            if allocate_caches:
                # The upstream attention constructor already resolves the
                # rank-local KV head count.  Keep the runtime cache layout
                # generic across TP2/TP4 rather than encoding the single-rank
                # Qwen dimensions here.
                cache = torch.zeros(
                    (
                        1,
                        16,
                        int(attention.num_kv_heads),
                        2 * int(attention.head_size),
                    ),
                    dtype=torch.bfloat16,
                )
                attention.kv_cache = cache
                caches.append((attention.layer_name, cache))
            else:
                record = caches[index]
                if record[0] != attention.layer_name or len(record) != 2:
                    raise RuntimeError("attention runtime cache identity mismatch")
                cache = record[1]
                attention.kv_cache = cache
            positions = torch.arange(
                start_position, start_position + tokens, dtype=torch.int32
            )
            metadata[attention.layer_name] = SimpleNamespace(
                num_actual_tokens=tokens,
                max_query_len=tokens,
                query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
                max_seq_len=start_position + tokens,
                seq_lens=torch.tensor([start_position + tokens], dtype=torch.int32),
                block_table=torch.tensor([[0]], dtype=torch.int32),
                slot_mapping=positions,
                num_decode_tokens=0 if prefill else tokens,
            )
            slot_mapping[attention.layer_name] = positions
    return metadata, slot_mapping, caches, tuple(conv_shape), tuple(recurrent_shape)


def clone_runtime_caches(caches: list[tuple]) -> list[tuple]:
    return [tuple([record[0], *(value.clone() for value in record[1:])]) for record in caches]


def compare_runtime_caches(
    actual: list[tuple], expected: list[tuple], executed_layers: int
) -> dict[str, object]:
    if len(actual) != len(expected):
        raise ValueError("runtime cache list length mismatch")
    records = []
    all_correct = True
    for index, (actual_record, expected_record) in enumerate(zip(actual, expected)):
        if actual_record[0] != expected_record[0] or len(actual_record) != len(
            expected_record
        ):
            raise RuntimeError("runtime cache record identity mismatch")
        tensors = []
        for actual_tensor, expected_tensor in zip(
            actual_record[1:], expected_record[1:]
        ):
            error = torch.abs(actual_tensor.float() - expected_tensor.float())
            exact = torch.equal(actual_tensor, expected_tensor)
            finite = bool(torch.all(torch.isfinite(error)).item())
            tolerance = 1.0e-5 if actual_tensor.dtype == torch.float32 else 0.0
            correct = bool(
                finite
                and torch.all(error <= tolerance).item()
                and index < executed_layers
            )
            if index >= executed_layers:
                correct = exact
            tensors.append(
                {
                    "shape": list(actual_tensor.shape),
                    "dtype": str(actual_tensor.dtype),
                    "exact": exact,
                    "max_abs_error": float(error.max().item()),
                    "tolerance": tolerance,
                    "correct": correct,
                }
            )
            all_correct = all_correct and correct
        records.append({"layer": index, "name": actual_record[0], "tensors": tensors})
    return {"layers": records, "correct": all_correct}


def compare_runtime_caches_to_golden(
    actual: list[tuple], golden: dict[str, torch.Tensor]
) -> dict[str, object]:
    if len(actual) != 24:
        raise RuntimeError("full prefill requires 24 runtime cache records")
    records = []
    all_correct = True
    for layer_index, record in enumerate(actual):
        if len(record) == 3:
            golden_conv = golden[f"layers.{layer_index}.conv_state"]
            if golden_conv.shape != (3, 6144, 3):
                raise RuntimeError("golden GDN conv state contract mismatch")
            expected_tensors = [
                golden_conv[1].T.contiguous().unsqueeze(0),
                golden[f"layers.{layer_index}.recurrent_state"].unsqueeze(0),
            ]
            state_names = ["conv_state", "recurrent_state"]
        elif len(record) == 2:
            expected_tensors = [golden[f"layers.{layer_index}.kv_cache"]]
            state_names = ["kv_cache"]
        else:
            raise RuntimeError("runtime cache record shape is unsupported")
        tensor_records = []
        for state_name, actual_tensor, expected_tensor in zip(
            state_names, record[1:], expected_tensors
        ):
            comparison = compare_cross_arch(actual_tensor, expected_tensor)
            comparison["name"] = state_name
            tensor_records.append(comparison)
            all_correct = all_correct and comparison["correct"]
        records.append(
            {
                "layer": layer_index,
                "name": record[0],
                "tensors": tensor_records,
            }
        )
    return {"layers": records, "correct": all_correct}


def expected_cache_tensors_for_layer(
    golden: dict[str, torch.Tensor], layer_index: int
) -> tuple[list[str], list[torch.Tensor]]:
    if layer_index % 4 != 3:
        golden_conv = golden[f"layers.{layer_index}.conv_state"]
        if golden_conv.shape != (3, 6144, 3):
            raise RuntimeError("golden GDN conv state contract mismatch")
        return ["conv_state", "recurrent_state"], [
            golden_conv[1].T.contiguous().unsqueeze(0),
            golden[f"layers.{layer_index}.recurrent_state"].unsqueeze(0),
        ]
    return ["kv_cache"], [golden[f"layers.{layer_index}.kv_cache"]]


def compare_layer_cache_to_golden(
    actual_record: tuple,
    golden: dict[str, torch.Tensor],
    layer_index: int,
    comparison_function,
) -> dict[str, object]:
    state_names, expected_tensors = expected_cache_tensors_for_layer(
        golden, layer_index
    )
    if len(actual_record) != len(expected_tensors) + 1:
        raise RuntimeError("teacher-forced runtime cache arity mismatch")
    records = []
    for state_name, actual_tensor, expected_tensor in zip(
        state_names, actual_record[1:], expected_tensors
    ):
        comparison = comparison_function(actual_tensor, expected_tensor)
        comparison["name"] = state_name
        records.append(comparison)
    return {
        "layer": layer_index,
        "name": actual_record[0],
        "tensors": records,
        "correct": all(record["correct"] for record in records),
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=int(os.environ.get("GSIM_TENSOR_PARALLEL_SIZE", "1")),
        help="upstream vLLM tensor-parallel world size (1, 2, or 4 for bring-up)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=int(os.environ.get("GSIM_RANK", "0")),
        help="rank of this externally launched worker",
    )
    parser.add_argument(
        "--dist-init-method",
        help="shared upstream torch.distributed rendezvous URI for TP workers",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="use upstream vLLM CompilationMode.NONE for fast TP bring-up",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help=(
            "diagnostic-only multi-rank bring-up: execute target layers without "
            "the single-rank NVIDIA oracle protocol"
        ),
    )
    parser.add_argument(
        "--inference-mode",
        choices=INFERENCE_MODES,
        default="production",
        help="explicit production, online differential, restart, or evidence mode",
    )
    parser.add_argument(
        "--debug-output-dir",
        type=Path,
        help="new absent-only directory for diagnostic artifacts and checkpoints",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="strict layer-boundary checkpoint for debug-resume",
    )
    parser.add_argument(
        "--debug-stop-after-layer",
        type=int,
        help="last decoder layer for a bounded debug/evidence suffix",
    )
    parser.add_argument(
        "--describe-inference-modes",
        action="store_true",
        help="print mode contracts without initializing a model",
    )
    parser.add_argument("--reference-final-state", type=Path)
    parser.add_argument("--reference-key", default="final_hidden")
    parser.add_argument("--stop-after-layer", type=int)
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument(
        "--prefill-differential",
        action="store_true",
        help="compare bounded empty-cache prefill against serial decode",
    )
    parser.add_argument(
        "--prefill-token-ids",
        type=int,
        nargs="+",
        default=[TOKEN_ID, 266],
    )
    parser.add_argument(
        "--prefill-golden-dir",
        type=Path,
        help=(
            "run one full 24-layer two-token empty-cache prefill and compare "
            "with the pinned independent NVIDIA golden"
        ),
    )
    parser.add_argument(
        "--teacher-force-layer",
        type=int,
        help=(
            "execute one decoder layer from the pinned NVIDIA previous-layer "
            "state; requires --prefill-golden-dir"
        ),
    )
    parser.add_argument(
        "--decode-window-golden-dir",
        type=Path,
        help=(
            "run the pinned two-token prefill plus two cache-preserving "
            "single-token decodes and compare with the four-token golden"
        ),
    )
    return parser


def validate_mode_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    debug_mode = args.inference_mode != "production"
    if type(args.tensor_parallel_size) is not int or not 1 <= args.tensor_parallel_size <= 16:
        parser.error("--tensor-parallel-size must be in 1..16")
    if type(args.rank) is not int or not 0 <= args.rank < args.tensor_parallel_size:
        parser.error("--rank must be in the tensor-parallel world")
    if args.tensor_parallel_size > 1 and not args.dist_init_method:
        parser.error("TP>1 requires --dist-init-method")
    legacy_execution_selected = any(
        (
            args.reference_final_state is not None,
            args.stop_after_layer is not None,
            args.weights_only,
            args.prefill_differential,
            args.prefill_golden_dir is not None,
            args.teacher_force_layer is not None,
            args.decode_window_golden_dir is not None,
        )
    )
    if debug_mode:
        if legacy_execution_selected:
            parser.error(
                "debug/evidence inference modes are exclusive with all legacy "
                "production execution options"
            )
        if args.debug_output_dir is None:
            parser.error("non-production modes require --debug-output-dir")
        if args.inference_mode == "debug-resume":
            if args.resume_checkpoint is None:
                parser.error("debug-resume requires --resume-checkpoint")
        elif args.resume_checkpoint is not None:
            parser.error("--resume-checkpoint is accepted only by debug-resume")
        if args.debug_stop_after_layer is None:
            args.debug_stop_after_layer = 23
        if not 0 <= args.debug_stop_after_layer < 24:
            parser.error("--debug-stop-after-layer must be in [0,23]")
    else:
        if any(
            (
                args.debug_output_dir is not None,
                args.resume_checkpoint is not None,
                args.debug_stop_after_layer is not None,
            )
        ):
            parser.error(
                "production forbids debug output, oracle/checkpoint, and resume options"
            )
    return args


def inference_mode_description() -> dict[str, object]:
    return {
        "schema": "amdgpu-sim.qwen35-vllm-inference-modes.v1",
        "default": "production",
        "modes": {
            "production": {
                "oracle": False,
                "checkpoint_restore": False,
                "layer_checkpoint_publish": False,
                "acceptance_eligible": True,
            },
            "debug-layer-diff": {
                "oracle": True,
                "checkpoint_restore": False,
                "layer_checkpoint_publish": True,
                "stop_first_mismatch": True,
                "acceptance_eligible": False,
            },
            "debug-resume": {
                "oracle": True,
                "checkpoint_restore": True,
                "layer_checkpoint_publish": True,
                "stop_first_mismatch": True,
                "acceptance_eligible": False,
            },
            "evidence": {
                "oracle": False,
                "checkpoint_restore": False,
                "layer_checkpoint_publish": True,
                "explicit_layer_loop": True,
                "acceptance_eligible": False,
            },
        },
        "common": {
            "tokens": PREFILL_TOKEN_IDS,
            "positions": [0, 1],
            "target_fallback": False,
            "oracle_feedback_to_target": False,
        },
    }


def create_debug_output(path: Path) -> Path:
    requested = path.expanduser()
    parent = requested.parent.resolve(strict=True)
    destination = parent / requested.name
    if requested.name in ("", ".", "..") or requested != destination:
        raise RuntimeError("--debug-output-dir must be normalized and symlink-free")
    try:
        os.mkdir(destination, 0o700)
    except FileExistsError as error:
        raise RuntimeError("--debug-output-dir must be absent") from error
    if destination.resolve(strict=True) != destination:
        raise RuntimeError("debug output directory resolved unexpectedly")
    return destination


def publish_result(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o400)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def publish_actual_diagnostic(
    output_dir: Path,
    *,
    layer_index: int,
    hidden: torch.Tensor,
    residual: torch.Tensor,
    state: dict[str, torch.Tensor],
    request_record: dict[str, object],
    response_record: dict[str, object],
    oracle_execution: dict[str, object],
    identity_sha256: str,
) -> dict[str, object]:
    """Atomically retain target outputs without creating a resume boundary."""

    tensors = {
        "hidden_after": hidden.detach().cpu().contiguous(),
        "residual_after": residual.detach().cpu().contiguous(),
        **{
            f"mutable_state.{name}": value.detach().cpu().contiguous()
            for name, value in sorted(state.items())
        },
    }
    descriptors = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": tensor_sha256(value),
            "nonfinite_count": int(
                torch.count_nonzero(~torch.isfinite(value.float())).item()
            ),
        }
        for name, value in tensors.items()
    }
    embedded = {
        "schema": "amdgpu-sim.qwen35-amd-layer-actual.v1",
        "layer": str(layer_index),
        "diagnostic_only": "true",
        "resume_eligible": "false",
        "target_feedback": "false",
    }
    tensor_payload = save_safetensors(tensors, metadata=embedded)
    manifest = {
        "schema": "amdgpu-sim.qwen35-amd-layer-actual.v1",
        "kind": "first_mismatch_amd_after_layer_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "resume_eligible": False,
        "target_feedback": False,
        "layer": layer_index,
        "identity_sha256": identity_sha256,
        "oracle_request": request_record,
        "oracle_response": response_record,
        "oracle_execution": oracle_execution,
        "tensor_roles": list(tensors),
        "tensors": descriptors,
        "payload": {
            "filename": "actual.safetensors",
            "bytes": len(tensor_payload),
            "sha256": hashlib.sha256(tensor_payload).hexdigest(),
            "embedded_metadata": embedded,
        },
    }
    manifest_payload = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    parent = output_dir.parent.resolve(strict=True)
    if output_dir != parent / output_dir.name:
        raise RuntimeError("actual diagnostic output must be normalized")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=parent))
    try:
        for filename, content in (
            ("actual.safetensors", tensor_payload),
            ("manifest.json", manifest_payload),
        ):
            descriptor = os.open(
                temporary / filename,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary / filename, 0o400)
        directory = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.chmod(temporary, 0o500)
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(
                errno.ENOSYS,
                "renameat2 is unavailable; refusing non-atomic diagnostic publish",
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
            -100,
            os.fsencode(temporary),
            -100,
            os.fsencode(output_dir),
            1,
        ) != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), output_dir)
        temporary = None
        parent_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if temporary is not None:
            os.chmod(temporary, 0o700)
            for candidate in temporary.iterdir():
                os.chmod(candidate, 0o600)
                candidate.unlink()
            os.rmdir(temporary)
    return {
        "path": str(output_dir),
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "payload_sha256": hashlib.sha256(tensor_payload).hexdigest(),
        "diagnostic_only": True,
        "resume_eligible": False,
        "target_feedback": False,
    }


def mutable_state(record: tuple) -> dict[str, torch.Tensor]:
    if len(record) == 3:
        return {"conv_state": record[1], "recurrent_state": record[2]}
    if len(record) == 2:
        return {"kv_cache": record[1]}
    raise RuntimeError("unsupported runtime cache record")


def snapshot_runtime_cache_guard(caches: list[tuple]) -> list[dict[str, object]]:
    return [
        {
            "name": record[0],
            "tensors": [
                {
                    "object_id": id(value),
                    "data_ptr": value.data_ptr(),
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "storage_offset": value.storage_offset(),
                    "value": value.detach().clone(),
                }
                for value in record[1:]
            ],
        }
        for record in caches
    ]


def validate_runtime_cache_guard(
    caches: list[tuple],
    snapshot: list[dict[str, object]],
    current_layer: int,
) -> dict[str, object]:
    if len(caches) != len(snapshot):
        raise RuntimeError("cache guard record count changed")
    records = []
    identity_correct = True
    non_target_unchanged = True
    for layer_index, (record, before) in enumerate(zip(caches, snapshot)):
        if record[0] != before["name"] or len(record) - 1 != len(before["tensors"]):
            raise RuntimeError("cache guard module/arity changed")
        tensor_records = []
        for value, tensor_before in zip(record[1:], before["tensors"]):
            storage_identity = bool(
                id(value) == tensor_before["object_id"]
                and value.data_ptr() == tensor_before["data_ptr"]
                and str(value.dtype) == tensor_before["dtype"]
                and list(value.shape) == tensor_before["shape"]
                and list(value.stride()) == tensor_before["stride"]
                and value.storage_offset() == tensor_before["storage_offset"]
            )
            before_sha256 = tensor_sha256(tensor_before["value"])
            after_sha256 = tensor_sha256(value)
            unchanged = before_sha256 == after_sha256
            tensor_records.append(
                {
                    "storage_identity_preserved": storage_identity,
                    "content_unchanged": unchanged,
                    "before_sha256": before_sha256,
                    "after_sha256": after_sha256,
                }
            )
            identity_correct = identity_correct and storage_identity
            if layer_index != current_layer:
                non_target_unchanged = non_target_unchanged and unchanged
        records.append({"layer": layer_index, "tensors": tensor_records})
    return {
        "records": records,
        "all_storage_identity_preserved": identity_correct,
        "non_target_cache_unchanged": non_target_unchanged,
        "correct": identity_correct and non_target_unchanged,
    }


def oracle_execution_record(value) -> dict[str, object]:
    if value is None:
        raise RuntimeError("oracle response lacks subprocess execution evidence")
    return {
        "argv": list(value.argv),
        "environment": value.environment,
        "exit_code": value.exit_code,
        "launcher_identity": value.launcher_identity,
        "stdout": value.stdout,
        "stderr": value.stderr,
        "stdout_sha256": hashlib.sha256(value.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(value.stderr.encode("utf-8")).hexdigest(),
    }


def compare_online_layer(
    *,
    actual_hidden: torch.Tensor,
    actual_residual: torch.Tensor,
    actual_state: dict[str, torch.Tensor],
    response,
    cache_guard: dict[str, object],
) -> dict[str, object]:
    if (
        actual_hidden.dtype != response.hidden_after.dtype
        or actual_hidden.shape != response.hidden_after.shape
        or actual_residual.dtype != response.residual_after.dtype
        or actual_residual.shape != response.residual_after.shape
    ):
        raise RuntimeError("online hidden/residual contract mismatch")
    hidden_comparison = compare_with_limits(
        actual_hidden, response.hidden_after, atol=0.03125
    )
    residual_comparison = compare_with_limits(
        actual_residual, response.residual_after, atol=0.03125
    )
    if set(actual_state) != set(response.mutable_state_after):
        raise RuntimeError("online mutable-state role mismatch")
    state_comparisons = {}
    for name in sorted(actual_state):
        atol = 1.0e-4 if actual_state[name].dtype == torch.float32 else 0.03125
        state_comparisons[name] = compare_with_limits(
            actual_state[name],
            response.mutable_state_after[name],
            atol=atol,
        )
    return {
        "hidden": hidden_comparison,
        "residual": residual_comparison,
        "mutable_state": state_comparisons,
        "correct": bool(
            hidden_comparison["correct"]
            and residual_comparison["correct"]
            and all(value["correct"] for value in state_comparisons.values())
            and cache_guard["correct"]
        ),
        "cache_mutation_guard": cache_guard,
        "target_feedback": False,
    }


def mismatch_evidence_policy(comparison_correct: bool) -> dict[str, bool]:
    return {
        "publish_actual_diagnostic": not comparison_correct,
        "publish_resume_checkpoint": comparison_correct,
        "stop": not comparison_correct,
        "target_feedback": False,
    }


def validate_resume_suffix(
    *, next_layer: int, stop_after_layer: int, resume_action: str
) -> bool:
    resume_final_norm = (
        next_layer == 24
        and stop_after_layer == 23
        and resume_action == "final_norm"
    )
    if next_layer > stop_after_layer and not resume_final_norm:
        raise RuntimeError("resume checkpoint leaves an empty debug suffix")
    return resume_final_norm


def run_explicit_layer_mode(
    *,
    args: argparse.Namespace,
    model,
    vllm_config: VllmConfig,
    architecture: str,
    checkpoint: dict[str, object],
    all_source_names: list[str],
    source_names: list[str],
    loaded_count: int,
    target,
) -> int:
    from _qwen35_layer_checkpoint import (
        load_layer_checkpoint,
        publish_layer_checkpoint,
        restore_layer_checkpoint,
    )

    use_oracle = (
        args.inference_mode in ("debug-layer-diff", "debug-resume")
        and not args.skip_oracle
    )
    if use_oracle:
        from _qwen35_layer_oracle_protocol import (
            LayerOracleExecutionError,
            current_layer_oracle_request_identity,
            expected_local_oracle_identity,
            publish_layer_oracle_request,
            run_layer_oracle,
        )

    output_dir = create_debug_output(args.debug_output_dir)
    input_ids = torch.tensor(PREFILL_TOKEN_IDS, dtype=torch.int64)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    metadata, slot_mapping, caches, conv_shape, recurrent_shape = bind_runtime_state(
        model,
        vllm_config,
        tokens=2,
        prefill=True,
        start_position=0,
    )
    layer_types = list(vllm_config.model_config.hf_text_config.layer_types)
    kv_shapes = [tuple(record[1].shape) for record in caches if len(record) == 2]
    if kv_shapes and any(shape != kv_shapes[0] for shape in kv_shapes):
        raise RuntimeError("rank-local full-attention KV cache shapes disagree")
    identity = exact_execution_identity(
        architecture=architecture,
        checkpoint=checkpoint,
        layer_types=layer_types,
        conv_shape=conv_shape,
        recurrent_shape=recurrent_shape,
        all_source_names=all_source_names,
        source_names=source_names,
        loaded_count=loaded_count,
        world_size=args.tensor_parallel_size,
        rank=args.rank,
        kv_cache_shape=kv_shapes[0] if kv_shapes else (1, 16, 2, 512),
    )
    request_record = request_identity(metadata, slot_mapping)
    oracle_identity = (
        current_layer_oracle_request_identity(Path(__file__))
        if use_oracle
        else None
    )
    expected_oracle = expected_local_oracle_identity() if use_oracle else None
    run_id = uuid.uuid4().hex
    layer_records: list[dict[str, object]] = []
    checkpoint_records: list[dict[str, object]] = []
    start_layer = 0
    previous_layer: int | None = None
    previous_manifest_sha256: str | None = None

    if args.inference_mode == "debug-resume":
        loaded_checkpoint = load_layer_checkpoint(args.resume_checkpoint)
        restored = restore_layer_checkpoint(
            args.resume_checkpoint,
            expected_identity=identity,
            expected_request_identity=request_record,
            expected_input_ids=input_ids,
            expected_positions=positions,
            expected_after_layer=loaded_checkpoint.after_layer,
            expected_manifest_sha256=loaded_checkpoint.manifest_sha256,
            caches=caches,
        )
        hidden_states = restored.hidden_states
        residual = restored.residual
        start_layer = restored.next_layer
        run_id = restored.manifest["lineage"]["run_id"]
        previous_layer = loaded_checkpoint.after_layer
        previous_manifest_sha256 = loaded_checkpoint.manifest_sha256
        checkpoint_records.append({"kind": "restored", **restored.result})
        validate_resume_suffix(
            next_layer=start_layer,
            stop_after_layer=args.debug_stop_after_layer,
            resume_action=restored.result["resume_action"],
        )
    else:
        hidden_states = model.embed_input_ids(input_ids)
        residual = None
        initial_checkpoint = publish_layer_checkpoint(
            output_dir / "checkpoint-after-embedded",
            identity=identity,
            request_identity=request_record,
            lineage={
                "run_id": run_id,
                "previous_after_layer": None,
                "previous_manifest_sha256": None,
            },
            after_layer=-1,
            hidden_states=hidden_states,
            residual=None,
            input_ids=input_ids,
            positions=positions,
            caches=caches,
        )
        checkpoint_records.append({"kind": "published", **initial_checkpoint})
        previous_layer = -1
        previous_manifest_sha256 = initial_checkpoint["manifest_sha256"]

    final_hidden = None
    failed = None
    end_layer = args.debug_stop_after_layer
    for layer_index in range(start_layer, end_layer + 1):
        record: dict[str, object] = {
            "layer": layer_index,
            "layer_type": layer_types[layer_index],
            "operation": "prefill_2",
            "positions": [0, 1],
            "oracle_feedback_to_target": False,
        }
        try:
            cache_guard_before = snapshot_runtime_cache_guard(caches)
            response = None
            if use_oracle:
                request = publish_layer_oracle_request(
                    output_dir / f"layer-{layer_index:02d}-request",
                    identity=oracle_identity,
                    layer_index=layer_index,
                    operation="prefill_2",
                    token_positions=[0, 1],
                    hidden_before=hidden_states,
                    residual_before=residual,
                    mutable_state_before=mutable_state(caches[layer_index]),
                )
                record["oracle_request"] = {
                    "path": str(request.path),
                    "request_id": request.request_id,
                    "package_sha256": request.package_sha256,
                    "identity_sha256": request.identity_sha256,
                    "payload_sha256": request.payload_sha256,
                }
                response = run_layer_oracle(
                    request=request,
                    response_dir=output_dir / f"layer-{layer_index:02d}-response",
                    expected_oracle_identity=expected_oracle,
                    oracle_script=NVIDIA_ORACLE_SCRIPT,
                    python=NVIDIA_ORACLE_PYTHON,
                )
                record["oracle_response"] = {
                    "path": str(response.path),
                    "request_id": response.request_id,
                    "package_sha256": response.package_sha256,
                    "identity_sha256": response.identity_sha256,
                    "payload_sha256": response.payload_sha256,
                    "target_feedback": False,
                }
                record["oracle_execution"] = oracle_execution_record(
                    response.execution_record
                )

            with set_forward_context(
                metadata,
                vllm_config,
                num_tokens=2,
                slot_mapping=slot_mapping,
            ):
                next_hidden, next_residual = model.model.layers[layer_index](
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )
            if not all(
                torch.all(torch.isfinite(value.float())).item()
                for value in (next_hidden, next_residual)
            ):
                raise RuntimeError("AMD layer returned a nonfinite activation")
            hidden_states, residual = next_hidden, next_residual
            cache_guard = validate_runtime_cache_guard(
                caches, cache_guard_before, layer_index
            )
            record["cache_mutation_guard"] = cache_guard
            if not cache_guard["correct"] and response is None:
                failed = {
                    "kind": "cache_mutation_guard_failure",
                    "layer": layer_index,
                }
                layer_records.append(record)
                break
            if response is not None:
                comparison = compare_online_layer(
                    actual_hidden=hidden_states,
                    actual_residual=residual,
                    actual_state=mutable_state(caches[layer_index]),
                    response=response,
                    cache_guard=cache_guard,
                )
                record["comparison"] = comparison
                record["mismatch_evidence_policy"] = mismatch_evidence_policy(
                    comparison["correct"]
                )
                if not comparison["correct"]:
                    actual_diagnostic = publish_actual_diagnostic(
                        output_dir / f"layer-{layer_index:02d}-amd-actual",
                        layer_index=layer_index,
                        hidden=hidden_states,
                        residual=residual,
                        state=mutable_state(caches[layer_index]),
                        request_record=record["oracle_request"],
                        response_record=record["oracle_response"],
                        oracle_execution=record["oracle_execution"],
                        identity_sha256=canonical_sha256(identity),
                    )
                    record["amd_actual_diagnostic"] = actual_diagnostic
                    failed = {
                        "kind": "first_numerical_mismatch",
                        "layer": layer_index,
                    }
                    layer_records.append(record)
                    break

            completed_checkpoint = publish_layer_checkpoint(
                output_dir / f"checkpoint-after-layer-{layer_index:02d}",
                identity=identity,
                request_identity=request_record,
                lineage={
                    "run_id": run_id,
                    "previous_after_layer": previous_layer,
                    "previous_manifest_sha256": previous_manifest_sha256,
                },
                after_layer=layer_index,
                hidden_states=hidden_states,
                residual=residual,
                input_ids=input_ids,
                positions=positions,
                caches=caches,
            )
            checkpoint_records.append({"kind": "published", **completed_checkpoint})
            previous_layer = layer_index
            previous_manifest_sha256 = completed_checkpoint["manifest_sha256"]
            record["checkpoint"] = {
                "path": completed_checkpoint["path"],
                "manifest_sha256": completed_checkpoint["manifest_sha256"],
                "state_sha256": completed_checkpoint["state_sha256"],
            }
            layer_records.append(record)
        except Exception as error:
            execution = getattr(error, "execution_record", None)
            if execution is not None:
                record["oracle_execution"] = oracle_execution_record(execution)
            failed = {
                "kind": "layer_execution_or_protocol_failure",
                "layer": layer_index,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            record["failure"] = failed
            layer_records.append(record)
            break

    if failed is None and end_layer == 23:
        if start_layer > 23:
            if residual is None:
                raise RuntimeError("final-norm resume boundary lacks residual")
        final_hidden, _ = model.model.norm(hidden_states, residual)
        if not torch.all(torch.isfinite(final_hidden.float())).item():
            failed = {"kind": "nonfinite_final_norm", "layer": 23}

    all_caches_finite = all(
        torch.all(torch.isfinite(value.float())).item()
        for cache_record in caches
        for value in cache_record[1:]
    )
    mutated_cache_count = sum(
        any(torch.count_nonzero(value).item() > 0 for value in record[1:])
        for record in caches
    )
    expected_mutated_cache_count = end_layer + 1
    cache_mutation_complete = mutated_cache_count == expected_mutated_cache_count
    payload = {
        "schema": "amdgpu-sim.gemsim-vllm-qwen35-explicit-layer-mode.v1",
        "inference_mode": args.inference_mode,
        "backend": target.backend,
        "arch": target.arch,
        "model": MODEL_ID,
        "architecture": architecture,
        "identity_sha256": canonical_sha256(identity),
        "request_identity_sha256": canonical_sha256(request_record),
        "run_id": run_id,
        "start_layer": start_layer,
        "stop_after_layer": end_layer,
        "layers_completed": len([item for item in layer_records if "checkpoint" in item]),
        "layer_records": layer_records,
        "checkpoints": checkpoint_records,
        "first_failure": failed,
        "final_norm_executed": final_hidden is not None,
        "final_hidden_sha256": (
            None if final_hidden is None else tensor_sha256(final_hidden)
        ),
        "all_caches_finite": all_caches_finite,
        "mutated_cache_count": mutated_cache_count,
        "expected_mutated_cache_count": expected_mutated_cache_count,
        "cache_mutation_complete": cache_mutation_complete,
        "oracle_used": use_oracle,
        "oracle_skipped": bool(args.skip_oracle),
        "oracle_feedback_to_target": False,
        "checkpoint_restore_used": args.inference_mode == "debug-resume",
        "acceptance_eligible": False,
        "fallback_count": 0,
        "cpu_fallback_count": 0,
        "nvidia_fallback_count": 0,
        "execution_success": (
            failed is None and all_caches_finite and cache_mutation_complete
        ),
        "output_correct": (
            failed is None and all_caches_finite and cache_mutation_complete
            if use_oracle
            else None
        ),
        "claim_boundary": (
            "diagnostic two-token explicit decoder-layer loop; NVIDIA responses "
            "are comparison-only and never target inputs. This result is not "
            "production acceptance evidence"
        ),
    }
    publish_result(output_dir / "result.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["execution_success"] else 1


def main() -> int:
    parser = argument_parser()
    args = validate_mode_args(parser, parser.parse_args())
    if args.describe_inference_modes:
        print(json.dumps(inference_mode_description(), sort_keys=True))
        return 0
    if args.prefill_differential and args.stop_after_layer is None:
        args.stop_after_layer = 3
    if args.stop_after_layer is not None and not 0 <= args.stop_after_layer < 24:
        raise ValueError("--stop-after-layer must be in [0, 23]")
    if args.decode_window_golden_dir is not None:
        if (
            args.prefill_golden_dir is not None
            or args.teacher_force_layer is not None
            or args.prefill_differential
            or args.reference_final_state is not None
            or args.weights_only
            or args.stop_after_layer is not None
        ):
            raise ValueError(
                "decode-window mode is exclusive with other execution modes"
            )
    elif args.prefill_golden_dir is not None:
        if args.teacher_force_layer is not None and not 0 <= args.teacher_force_layer < 24:
            raise ValueError("--teacher-force-layer must be in [0,23]")
        if (
            args.prefill_differential
            or args.reference_final_state is not None
            or args.weights_only
            or args.stop_after_layer is not None
        ):
            raise ValueError(
                "prefill golden mode is exclusive with other execution modes"
            )
    elif args.teacher_force_layer is not None:
        raise ValueError("--teacher-force-layer requires --prefill-golden-dir")
    elif args.prefill_differential:
        if args.reference_final_state is not None or args.weights_only:
            raise ValueError("prefill differential is exclusive with reference/weights")
        if not 2 <= len(args.prefill_token_ids) <= 16:
            raise ValueError("prefill differential requires 2..16 token IDs")
        if any(not 0 <= token < 248320 for token in args.prefill_token_ids):
            raise ValueError("prefill token ID is outside the Qwen vocabulary")
    elif (
        args.inference_mode == "production"
        and not args.weights_only
        and args.reference_final_state is None
    ):
        parser.error("--reference-final-state is required outside prefill mode")

    target = triton.runtime.driver.active.get_current_target()
    if (target.backend, target.arch) != ("gemsim_amd", "gfx950"):
        raise RuntimeError(f"unexpected Triton target: {target}")
    checkpoint = validate_checkpoint()
    reference = None
    reference_record = None
    prefill_golden = None
    prefill_golden_record = None
    decode_window_golden = None
    decode_window_golden_record = None
    decode_window_prefix_comparison = None
    if args.decode_window_golden_dir is not None:
        prefill_golden, prefill_golden_record = load_prefill_golden(
            PREFILL_GOLDEN_DIR
        )
        decode_window_golden, decode_window_golden_record = load_prefill_golden(
            args.decode_window_golden_dir,
            expected_token_ids=DECODE_WINDOW_TOKEN_IDS,
            expected_files=DECODE_WINDOW_GOLDEN_FILES,
        )
        decode_window_prefix_comparison = compare(
            decode_window_golden["final_norm"][: len(PREFILL_TOKEN_IDS)],
            prefill_golden["final_norm"],
        )
        if not decode_window_prefix_comparison["correct"]:
            raise RuntimeError(
                "independent CUDA prefill prefix is not causally stable"
            )
    if args.prefill_golden_dir is not None:
        prefill_golden, prefill_golden_record = load_prefill_golden(
            args.prefill_golden_dir
        )
    if args.reference_final_state is not None:
        reference, reference_record = load_reference(
            args.reference_final_state, args.reference_key
        )
    gemsim_vllm.register_ops()

    model_config = ModelConfig(
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=(
            2
            if args.inference_mode != "production"
            else (
                len(DECODE_WINDOW_TOKEN_IDS)
                if args.decode_window_golden_dir is not None
                else (
                    2
                    if args.prefill_golden_dir is not None
                    else (
                        len(args.prefill_token_ids)
                        if args.prefill_differential
                        else 1
                    )
                )
            )
        ),
        enforce_eager=args.eager,
        trust_remote_code=False,
    )
    parallel_config = ParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        distributed_executor_backend="external_launcher",
    )
    compilation_config = (
        CompilationConfig(mode=CompilationMode.NONE) if args.eager else None
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        compilation_config=compilation_config,
    )
    current_platform.check_and_update_config(vllm_config)
    old_dtype = torch.get_default_dtype()
    with distributed_world(
        vllm_config,
        world_size=args.tensor_parallel_size,
        rank=args.rank,
        distributed_init_method=args.dist_init_method,
    ), torch.no_grad():
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cpu"):
                model_class, architecture = ModelRegistry.resolve_model_cls(
                    ["GemsimQwen3_5ForCausalLM"], model_config
                )
                model = model_class(vllm_config=vllm_config)
        finally:
            torch.set_default_dtype(old_dtype)

        with safe_open(MODEL_FILE, framework="pt", device="cpu") as tensors:
            all_source_names = list(tensors.keys())
            unexpected_names = [
                name
                for name in all_source_names
                if not name.startswith("model.language_model.")
                and not name.startswith("model.visual.")
                and not name.startswith("mtp.")
            ]
            if unexpected_names:
                raise RuntimeError(
                    f"unexpected checkpoint tensor namespaces: {unexpected_names[:8]}"
                )
            source_names = [
                name
                for name in all_source_names
                if name.startswith("model.language_model.")
            ]
            visual_names = [
                name
                for name in all_source_names
                if name.startswith("model.visual.")
            ]
            mtp_names = [name for name in all_source_names if name.startswith("mtp.")]
            if len(source_names) != 320:
                raise RuntimeError(
                    f"text checkpoint tensor count mismatch: {len(source_names)}"
                )
            if len(visual_names) != 153 or len(mtp_names) != 15:
                raise RuntimeError(
                    "excluded checkpoint tensor count mismatch: "
                    f"visual={len(visual_names)}, mtp={len(mtp_names)}"
                )
            loaded = model.load_weights(
                (name, tensors.get_tensor(name)) for name in source_names
            )
        parameter_hashes = layer0_parameter_hashes(model)
        if args.weights_only:
            print(
                json.dumps(
                    {
                        "schema": "amdgpu-sim.gemsim-vllm-weight-probe.v1",
                        "inference_mode": args.inference_mode,
                        "architecture": architecture,
                        "checkpoint": checkpoint,
                        "weights_checkpoint_count": len(all_source_names),
                        "weights_source_count": len(source_names),
                        "weights_loaded_count": len(loaded),
                        "parameters": parameter_hashes,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.inference_mode != "production":
            return run_explicit_layer_mode(
                args=args,
                model=model,
                vllm_config=vllm_config,
                architecture=architecture,
                checkpoint=checkpoint,
                all_source_names=all_source_names,
                source_names=source_names,
                loaded_count=len(loaded),
                target=target,
            )
        if args.stop_after_layer is not None:
            model.model.end_layer = args.stop_after_layer + 1
            model.model.norm = IdentityFinalNorm()
        if args.decode_window_golden_dir is not None:
            assert prefill_golden is not None
            assert prefill_golden_record is not None
            assert decode_window_golden is not None
            assert decode_window_golden_record is not None
            assert decode_window_prefix_comparison is not None
            metadata, slot_mapping, caches, conv_shape, recurrent_shape = (
                bind_runtime_state(
                    model,
                    vllm_config,
                    tokens=len(PREFILL_TOKEN_IDS),
                    prefill=True,
                    start_position=0,
                )
            )
            with set_forward_context(
                metadata,
                vllm_config,
                num_tokens=len(PREFILL_TOKEN_IDS),
                slot_mapping=slot_mapping,
            ):
                prefill_output = model(
                    input_ids=torch.tensor(PREFILL_TOKEN_IDS, dtype=torch.int64),
                    positions=torch.arange(
                        len(PREFILL_TOKEN_IDS), dtype=torch.int64
                    ),
                ).clone()
            step_comparisons = [
                {
                    "kind": "empty_cache_prefill",
                    "positions": [0, 1],
                    "comparison": compare_cross_arch(
                        prefill_output, prefill_golden["final_norm"]
                    ),
                }
            ]
            prefill_cache_comparison = compare_runtime_caches_to_golden(
                caches, prefill_golden
            )
            for position in range(len(PREFILL_TOKEN_IDS), len(DECODE_WINDOW_TOKEN_IDS)):
                token_id = DECODE_WINDOW_TOKEN_IDS[position]
                metadata, slot_mapping, caches, conv_shape, recurrent_shape = (
                    bind_runtime_state(
                        model,
                        vllm_config,
                        tokens=1,
                        prefill=False,
                        start_position=position,
                        caches=caches,
                    )
                )
                with set_forward_context(
                    metadata,
                    vllm_config,
                    num_tokens=1,
                    slot_mapping=slot_mapping,
                ):
                    decode_output = model(
                        input_ids=torch.tensor([token_id], dtype=torch.int64),
                        positions=torch.tensor([position], dtype=torch.int64),
                    ).clone()
                step_comparisons.append(
                    {
                        "kind": "cache_preserving_decode",
                        "positions": [position],
                        "input_token_id": token_id,
                        "comparison": compare_cross_arch(
                            decode_output,
                            decode_window_golden["final_norm"][
                                position : position + 1
                            ],
                        ),
                    }
                )
            final_cache_comparison = compare_runtime_caches_to_golden(
                caches, decode_window_golden
            )
            caches_finite = all(
                torch.all(torch.isfinite(value.float())).item()
                for record in caches
                for value in record[1:]
            )
            mutated_cache_count = sum(
                any(
                    torch.count_nonzero(value).item() > 0
                    for value in record[1:]
                )
                for record in caches
            )
            comparison_correct = all(
                record["comparison"]["correct"] for record in step_comparisons
            )
            execution_complete = bool(
                caches_finite and mutated_cache_count == 24
            )
            correct = bool(
                execution_complete
                and comparison_correct
                and prefill_cache_comparison["correct"]
                and final_cache_comparison["correct"]
            )
            print(
                json.dumps(
                    {
                        "schema": (
                            "amdgpu-sim.gemsim-vllm-qwen35-decode-window.v1"
                        ),
                        "inference_mode": args.inference_mode,
                        "backend": target.backend,
                        "arch": target.arch,
                        "model": MODEL_ID,
                        "architecture": architecture,
                        "checkpoint": checkpoint,
                        "token_ids": DECODE_WINDOW_TOKEN_IDS,
                        "positions": list(range(len(DECODE_WINDOW_TOKEN_IDS))),
                        "prefill_tokens": len(PREFILL_TOKEN_IDS),
                        "decode_tokens": (
                            len(DECODE_WINDOW_TOKEN_IDS) - len(PREFILL_TOKEN_IDS)
                        ),
                        "layers_executed_per_step": 24,
                        "weights_loaded_count": len(loaded),
                        "prefill_golden": prefill_golden_record,
                        "decode_window_golden": decode_window_golden_record,
                        "golden_prefix_causality_comparison": (
                            decode_window_prefix_comparison
                        ),
                        "step_comparisons": step_comparisons,
                        "prefill_cache_comparison": prefill_cache_comparison,
                        "final_cache_comparison": final_cache_comparison,
                        "all_caches_finite": caches_finite,
                        "mutated_cache_count": mutated_cache_count,
                        "gdn_conv_state_shape": [1, *conv_shape],
                        "gdn_recurrent_state_shape": [1, *recurrent_shape],
                        "execution_complete": execution_complete,
                        "external_nvidia_correct": bool(
                            comparison_correct
                            and prefill_cache_comparison["correct"]
                            and final_cache_comparison["correct"]
                        ),
                        "fallback_count": 0,
                        "cpu_fallback_count": 0,
                        "nvidia_fallback_count": 0,
                        "claim_boundary": (
                            "one registered vLLM two-token empty-cache prefill "
                            "followed by two single-token forwards that retain "
                            "the same 24-layer GDN/NHD caches; independent CUDA "
                            "is external comparison only. Scheduler workers, "
                            "batching, logits, sampling, CCL, and TP are not "
                            "claimed"
                        ),
                        "output_correct": correct,
                    },
                    sort_keys=True,
                )
            )
            return 0 if correct else 1
        if args.prefill_golden_dir is not None:
            assert prefill_golden is not None
            assert prefill_golden_record is not None
            if args.teacher_force_layer is not None:
                layer_index = args.teacher_force_layer
                metadata, slot_mapping, caches, conv_shape, recurrent_shape = (
                    bind_runtime_state(
                        model,
                        vllm_config,
                        tokens=len(PREFILL_TOKEN_IDS),
                        prefill=True,
                        start_position=0,
                    )
                )
                if layer_index == 0:
                    hidden_input = prefill_golden["hidden_input"]
                    residual_input = None
                else:
                    hidden_input = prefill_golden[
                        f"layers.{layer_index - 1}.returned_hidden"
                    ]
                    residual_input = prefill_golden[
                        f"layers.{layer_index - 1}.returned_residual"
                    ]
                with set_forward_context(
                    metadata,
                    vllm_config,
                    num_tokens=len(PREFILL_TOKEN_IDS),
                    slot_mapping=slot_mapping,
                ):
                    actual_hidden, actual_residual = model.model.layers[
                        layer_index
                    ](
                        hidden_input,
                        residual_input,
                        positions=torch.arange(
                            len(PREFILL_TOKEN_IDS), dtype=torch.int64
                        ),
                    )
                hidden_comparison = compare(
                    actual_hidden,
                    prefill_golden[f"layers.{layer_index}.returned_hidden"],
                )
                residual_comparison = compare(
                    actual_residual,
                    prefill_golden[f"layers.{layer_index}.returned_residual"],
                )
                cache_comparison = compare_layer_cache_to_golden(
                    caches[layer_index], prefill_golden, layer_index, compare
                )
                caches_finite = all(
                    torch.all(torch.isfinite(value.float())).item()
                    for record in caches
                    for value in record[1:]
                )
                mutated_layers = [
                    index
                    for index, record in enumerate(caches)
                    if any(
                        torch.count_nonzero(value).item() > 0
                        for value in record[1:]
                    )
                ]
                correct = bool(
                    hidden_comparison["correct"]
                    and residual_comparison["correct"]
                    and cache_comparison["correct"]
                    and caches_finite
                    and mutated_layers == [layer_index]
                )
                print(
                    json.dumps(
                        {
                            "schema": (
                                "amdgpu-sim.gemsim-vllm-qwen35-"
                                "teacher-forced-layer.v1"
                            ),
                            "inference_mode": args.inference_mode,
                            "backend": target.backend,
                            "arch": target.arch,
                            "model": MODEL_ID,
                            "architecture": architecture,
                            "checkpoint": checkpoint,
                            "golden": prefill_golden_record,
                            "layer": layer_index,
                            "layer_type": (
                                "full_attention"
                                if layer_index % 4 == 3
                                else "linear_attention"
                            ),
                            "token_ids": PREFILL_TOKEN_IDS,
                            "positions": [0, 1],
                            "hidden_comparison": hidden_comparison,
                            "residual_comparison": residual_comparison,
                            "cache_comparison": cache_comparison,
                            "all_caches_finite": caches_finite,
                            "mutated_layers": mutated_layers,
                            "gdn_conv_state_shape": [1, *conv_shape],
                            "gdn_recurrent_state_shape": [1, *recurrent_shape],
                            "fallback_count": 0,
                            "cpu_fallback_count": 0,
                            "nvidia_fallback_count": 0,
                            "claim_boundary": (
                                "one teacher-forced decoder layer from the "
                                "independent NVIDIA previous-layer state; this "
                                "isolates intrinsic layer correctness from "
                                "cross-architecture prefix accumulation"
                            ),
                            "output_correct": correct,
                        },
                        sort_keys=True,
                    )
                )
                return 0 if correct else 1
            metadata, slot_mapping, caches, conv_shape, recurrent_shape = (
                bind_runtime_state(
                    model,
                    vllm_config,
                    tokens=len(PREFILL_TOKEN_IDS),
                    prefill=True,
                    start_position=0,
                )
            )
            with set_forward_context(
                metadata,
                vllm_config,
                num_tokens=len(PREFILL_TOKEN_IDS),
                slot_mapping=slot_mapping,
            ):
                output = model(
                    input_ids=torch.tensor(PREFILL_TOKEN_IDS, dtype=torch.int64),
                    positions=torch.arange(
                        len(PREFILL_TOKEN_IDS), dtype=torch.int64
                    ),
                )
            if output.dtype != torch.bfloat16 or output.shape != (2, 1024):
                raise RuntimeError("full prefill output contract mismatch")
            comparison = compare_cross_arch(output, prefill_golden["final_norm"])
            cache_comparison = compare_runtime_caches_to_golden(
                caches, prefill_golden
            )
            caches_finite = all(
                torch.all(torch.isfinite(value.float())).item()
                for record in caches
                for value in record[1:]
            )
            mutated_cache_count = sum(
                any(
                    torch.count_nonzero(value).item() > 0
                    for value in record[1:]
                )
                for record in caches
            )
            correct = bool(
                comparison["correct"]
                and cache_comparison["correct"]
                and caches_finite
                and mutated_cache_count == 24
            )
            print(
                json.dumps(
                    {
                        "schema": (
                            "amdgpu-sim.gemsim-vllm-qwen35-prefill-golden.v1"
                        ),
                        "inference_mode": args.inference_mode,
                        "backend": target.backend,
                        "arch": target.arch,
                        "model": MODEL_ID,
                        "architecture": architecture,
                        "checkpoint": checkpoint,
                        "token_ids": PREFILL_TOKEN_IDS,
                        "positions": [0, 1],
                        "layers_executed": 24,
                        "weights_loaded_count": len(loaded),
                        "golden": prefill_golden_record,
                        "output_comparison": comparison,
                        "cache_comparison": cache_comparison,
                        "all_caches_finite": caches_finite,
                        "mutated_cache_count": mutated_cache_count,
                        "gdn_conv_state_shape": [1, *conv_shape],
                        "gdn_recurrent_state_shape": [1, *recurrent_shape],
                        "fallback_count": 0,
                        "cpu_fallback_count": 0,
                        "nvidia_fallback_count": 0,
                        "claim_boundary": (
                            "one registered vLLM 24-layer empty-cache causal "
                            "prefill with two text tokens; independent NVIDIA "
                            "CUDA is used only as an external cross-architecture "
                            "golden. Worker scheduling, prior-context chunked "
                            "prefill, batching, CCL, TP, logits, and sampling "
                            "are not claimed"
                        ),
                        "output_correct": correct,
                    },
                    sort_keys=True,
                )
            )
            return 0 if correct else 1
        if args.prefill_differential:
            executed_layers = args.stop_after_layer + 1
            serial_outputs = []
            serial_caches = None
            for position, token_id in enumerate(args.prefill_token_ids):
                metadata, slot_mapping, serial_caches, conv_shape, recurrent_shape = (
                    bind_runtime_state(
                        model,
                        vllm_config,
                        tokens=1,
                        prefill=False,
                        start_position=position,
                        caches=serial_caches,
                    )
                )
                with set_forward_context(
                    metadata,
                    vllm_config,
                    num_tokens=1,
                    slot_mapping=slot_mapping,
                ):
                    serial_outputs.append(
                        model(
                            input_ids=torch.tensor([token_id], dtype=torch.int64),
                            positions=torch.tensor([position], dtype=torch.int64),
                        ).clone()
                    )
            serial_output = torch.cat(serial_outputs, dim=0)
            serial_snapshot = clone_runtime_caches(serial_caches)

            metadata, slot_mapping, prefill_caches, conv_shape, recurrent_shape = (
                bind_runtime_state(
                    model,
                    vllm_config,
                    tokens=len(args.prefill_token_ids),
                    prefill=True,
                    start_position=0,
                )
            )
            with set_forward_context(
                metadata,
                vllm_config,
                num_tokens=len(args.prefill_token_ids),
                slot_mapping=slot_mapping,
            ):
                prefill_output = model(
                    input_ids=torch.tensor(
                        args.prefill_token_ids, dtype=torch.int64
                    ),
                    positions=torch.arange(
                        len(args.prefill_token_ids), dtype=torch.int64
                    ),
                )
            comparison = compare(prefill_output, serial_output)
            cache_comparison = compare_runtime_caches(
                prefill_caches, serial_snapshot, executed_layers
            )
            caches_finite = all(
                torch.all(torch.isfinite(value.float())).item()
                for record in prefill_caches
                for value in record[1:]
            )
            mutated_cache_count = sum(
                any(torch.count_nonzero(value).item() > 0 for value in record[1:])
                for record in prefill_caches
            )
            correct = bool(
                comparison["correct"]
                and cache_comparison["correct"]
                and caches_finite
                and mutated_cache_count == executed_layers
            )
            print(
                json.dumps(
                    {
                        "schema": "amdgpu-sim.gemsim-vllm-qwen35-prefill-differential.v1",
                        "inference_mode": args.inference_mode,
                        "backend": target.backend,
                        "arch": target.arch,
                        "model": MODEL_ID,
                        "architecture": architecture,
                        "checkpoint": checkpoint,
                        "token_ids": args.prefill_token_ids,
                        "positions": list(range(len(args.prefill_token_ids))),
                        "layers_executed": executed_layers,
                        "weights_loaded_count": len(loaded),
                        "comparison": comparison,
                        "cache_comparison": cache_comparison,
                        "all_caches_finite": caches_finite,
                        "mutated_cache_count": mutated_cache_count,
                        "gdn_conv_state_shape": [1, *conv_shape],
                        "gdn_recurrent_state_shape": [1, *recurrent_shape],
                        "fallback_count": 0,
                        "cpu_fallback_count": 0,
                        "nvidia_fallback_count": 0,
                        "claim_boundary": (
                            "one-request empty-cache causal prefill of 2..16 "
                            "tokens compared with serial decode through the same "
                            "formal registered model; worker scheduling, chunked "
                            "prefill with prior context, batching, CCL, and TP are "
                            "not claimed"
                        ),
                        "output_correct": correct,
                    },
                    sort_keys=True,
                )
            )
            return 0 if correct else 1
        metadata, slot_mapping, caches, conv_shape, recurrent_shape = (
            bind_runtime_state(model, vllm_config)
        )
        input_ids = torch.tensor([TOKEN_ID], dtype=torch.int64)
        positions = torch.tensor([0], dtype=torch.int64)
        with set_forward_context(
            metadata,
            vllm_config,
            num_tokens=1,
            slot_mapping=slot_mapping,
        ):
            output = model(input_ids=input_ids, positions=positions)

    if output.dtype != torch.bfloat16 or output.shape != (1, 1024):
        raise RuntimeError("formal vLLM forward output contract mismatch")
    assert reference is not None
    comparison = compare(output, reference)
    caches_finite = all(
        torch.all(torch.isfinite(value.float())).item()
        for record in caches
        for value in record[1:]
    )
    mutated_cache_count = sum(
        any(torch.count_nonzero(value).item() > 0 for value in record[1:])
        for record in caches
    )
    expected_mutated_cache_count = (
        24 if args.stop_after_layer is None else args.stop_after_layer + 1
    )
    state_mutated = mutated_cache_count == expected_mutated_cache_count
    correct = bool(comparison["correct"] and caches_finite and state_mutated)
    payload = {
        "schema": "amdgpu-sim.gemsim-vllm-qwen35-model-forward.v1",
        "inference_mode": args.inference_mode,
        "backend": target.backend,
        "arch": target.arch,
        "model": MODEL_ID,
        "architecture": architecture,
        "checkpoint": checkpoint,
        "token_id": TOKEN_ID,
        "position": 0,
        "layers_executed": expected_mutated_cache_count,
        "stop_after_layer": args.stop_after_layer,
        "weights_checkpoint_count": len(all_source_names),
        "weights_source_count": len(source_names),
        "visual_weights_excluded_count": len(visual_names),
        "mtp_weights_excluded_count": len(mtp_names),
        "weights_loaded_count": len(loaded),
        "layer0_parameter_hashes": parameter_hashes,
        "metadata_layer_count": len(metadata),
        "gdn_layer_count": sum(len(record) == 3 for record in caches),
        "full_attention_layer_count": sum(len(record) == 2 for record in caches),
        "gdn_conv_state_shape": [1, *conv_shape],
        "gdn_recurrent_state_shape": [1, *recurrent_shape],
        "reference": reference_record,
        "comparison": comparison,
        "all_caches_finite": caches_finite,
        "all_selected_caches_mutated": state_mutated,
        "mutated_cache_count": mutated_cache_count,
        "fallback_count": 0,
        "cpu_fallback_count": 0,
        "nvidia_fallback_count": 0,
        "claim_boundary": (
            "actual registered vLLM text-model forward with manually bound "
            "single-token scheduler metadata; a stop-after-layer run exposes "
            "the returned hidden tensor before final norm; worker scheduling, "
            "prefill, multi-token decode, logits, sampling, CCL, and TP are "
            "not claimed"
        ),
        "output_correct": correct,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
