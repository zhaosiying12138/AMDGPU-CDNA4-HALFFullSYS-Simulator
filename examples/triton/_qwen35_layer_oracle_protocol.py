"""Fail-closed packages for per-layer NVIDIA diagnostic comparisons.

This host-side module serializes an exact before-layer AMD snapshot for the
independent NVIDIA oracle and validates the returned after-layer diagnostic.
It never copies a returned NVIDIA tensor into target state.  Callers may only
compare the returned tensors with already-computed AMD results.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
from datetime import datetime
import errno
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors


ROOT = Path(__file__).resolve().parents[2]
PINNED_ORACLE_PYTHON = Path(
    "/home/zhaosiying/miniforge3/envs/triton-dev/bin/python3"
)
PINNED_NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")
DEFAULT_ORACLE_SCRIPT = ROOT / "tools/qwen35_nvidia_layer_oracle.py"
DEFAULT_GOLDEN_SCRIPT = ROOT / "tools/qwen35_nvidia_golden.py"

REQUEST_SCHEMA = "amdgpu-sim.qwen35-nvidia-layer-oracle-request.v1"
RESPONSE_SCHEMA = "amdgpu-sim.qwen35-nvidia-layer-oracle-response.v1"
REQUEST_KIND = "exact_live_amd_before_layer_diagnostic"
RESPONSE_KIND = "independent_torch_cuda_after_layer_diagnostic"
FINAL_NORM_REQUEST_KIND = "exact_live_amd_before_final_norm_diagnostic"
FINAL_NORM_RESPONSE_KIND = "independent_torch_cuda_after_final_norm_diagnostic"
REQUEST_JSON_FILENAME = "request.json"
RESPONSE_JSON_FILENAME = "response.json"
TENSOR_FILENAME = "tensors.safetensors"
WRITE_POLICY = (
    "same_parent_temp_directory_fsync_renameat2_noreplace_parent_fsync_read_only"
)
TARGET_FEEDBACK_POLICY = "prohibited"
REQUEST_ID_DOMAIN = b"amdgpu-sim.qwen35-layer-oracle-request-id.v1\0"
PACKAGE_HASH_DOMAIN = b"amdgpu-sim.qwen35-layer-oracle-package.v1\0"

AT_FDCWD = -100
RENAME_NOREPLACE = 1
MAX_JSON_BYTES = 1024 * 1024
MAX_TENSOR_BYTES = 32 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_SHARD = "model.safetensors-00001-of-00001.safetensors"
NUM_LAYERS = 24
HIDDEN_SIZE = 1024
GDN_QKV_DIM = 6144
GDN_HEADS = 16
GDN_HEAD_DIM = 128
KV_CACHE_SLOTS = 16
FULL_KV_HEADS = 2
FULL_KV_CONTENT = 512
LAYER_TYPES = tuple(
    layer_type
    for _ in range(6)
    for layer_type in (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )
)
PINNED_CHECKPOINT_ARTIFACTS = {
    "config.json": {
        "bytes": 2907,
        "sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    },
    "manifest.json": {
        "bytes": 1008,
        "sha256": "de2281cc73a1329d13245cb9658be910cf435e72c4ea0277c4f8811a24edf762",
    },
    MODEL_SHARD: {
        "bytes": 1746942600,
        "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    },
    "model.safetensors.index.json": {
        "bytes": 50900,
        "sha256": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
    },
}
PINNED_GPU = {
    "name": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "uuid": "GPU-64aae36b-ef77-b0d4-b1c7-f7ab17a729f1",
    "compute_capability": [12, 0],
}
PLUGIN_ROOT = "plugins/framework/gemsim_vllm"
PLUGIN_SOURCE_PATHS = {
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/adapters.py",
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/attention.py",
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/kernels.py",
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/model.py",
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/ops.py",
}
FORMULA_SOURCE_PATHS = {
    "projects/vllm/vllm/model_executor/layers/layernorm.py",
    "projects/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    "projects/vllm/vllm/model_executor/models/qwen3_5.py",
    "projects/vllm/vllm/model_executor/models/qwen3_next.py",
}
EXECUTION_BOUNDARY = {
    "allowed_framework_internal_modules": ["triton"],
    "direct_triton_import": False,
    "forbidden_project_module_roots": ["gemsim_vllm", "m5", "gem5"],
    "implementation": "pure_torch_cuda_formula",
    "non_target_execution": True,
}

_IDENTITY_KEYS = {"checkpoint", "formula", "model", "plugin", "runner"}
_REQUEST_CORE_KEYS = {
    "diagnostic_only",
    "identities",
    "kind",
    "layer_index",
    "layer_type",
    "operation",
    "residual_input",
    "schema",
    "state_snapshot",
    "target_feedback",
    "tensor_roles",
    "tensors",
    "token_positions",
}
_REQUEST_KEYS = _REQUEST_CORE_KEYS | {"payload", "request_id"}
_SOURCE_CHECKPOINT_KEYS = {
    "after_layer",
    "binding_sha256",
    "directory",
    "hidden",
    "identity_sha256",
    "manifest_sha256",
    "next_layer",
    "request_sha256",
    "residual",
    "resume_action",
    "schema",
    "state_sha256",
    "token_positions",
}
_TENSOR_DESCRIPTOR_KEYS = {"bytes", "dtype", "finite", "sha256", "shape"}


class LayerOracleProtocolError(RuntimeError):
    """An oracle request/response is malformed, unsafe, or misbound."""


@dataclass(frozen=True)
class LayerOracleExecutionRecord:
    """Exact subprocess evidence retained by the AMD-side diagnostic runner."""

    argv: tuple[str, ...]
    environment: dict[str, str]
    exit_code: int | None
    launcher_identity: dict[str, Any]
    stdout: str
    stderr: str


class LayerOracleExecutionError(LayerOracleProtocolError):
    """An oracle subprocess failed; ``execution_record`` retains its evidence."""

    def __init__(
        self,
        message: str,
        execution_record: LayerOracleExecutionRecord,
    ) -> None:
        super().__init__(message)
        self.execution_record = execution_record


@dataclass(frozen=True)
class LayerOracleRequest:
    path: Path
    document: dict[str, Any]
    request_id: str
    package_sha256: str
    request_json_sha256: str
    payload_sha256: str
    identity_sha256: str
    identities: dict[str, Any]
    layer_index: int
    layer_type: str
    operation: str
    token_positions: tuple[int, ...]
    hidden_before: torch.Tensor
    residual_before: torch.Tensor | None
    mutable_state_before: dict[str, torch.Tensor]
    source_checkpoint: dict[str, Any] | None


@dataclass(frozen=True)
class LayerOracleResponse:
    path: Path
    document: dict[str, Any]
    request_id: str
    package_sha256: str
    response_json_sha256: str
    payload_sha256: str
    identity_sha256: str
    identities: dict[str, Any]
    oracle_identity: dict[str, Any]
    layer_index: int
    layer_type: str
    operation: str
    token_positions: tuple[int, ...]
    hidden_after: torch.Tensor | None
    residual_after: torch.Tensor | None
    final_hidden_after: torch.Tensor | None
    mutable_state_after: dict[str, torch.Tensor]
    execution_record: LayerOracleExecutionRecord | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LayerOracleProtocolError(message)


def _require_exact_keys(value: object, expected: set[str], name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    observed = set(value)
    _require(
        observed == expected,
        f"{name} keys mismatch: missing={sorted(expected - observed)}, "
        f"extra={sorted(observed - expected)}",
    )
    return value


def _require_sha256(value: object, name: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256",
    )
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=lambda _item: (_ for _ in ()).throw(
                TypeError("non-JSON value")
            ),
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise LayerOracleProtocolError(f"value is not canonical JSON: {error}") from error


def _strict_json_load(value: bytes, name: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise LayerOracleProtocolError(f"duplicate JSON key in {name}: {key}")
            result[key] = item
        return result

    try:
        return json.loads(
            value.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {constant}")
            ),
        )
    except LayerOracleProtocolError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise LayerOracleProtocolError(f"invalid JSON in {name}: {error}") from error


def _normalized_json(value: object) -> Any:
    return _strict_json_load(_canonical_json(value), "canonical value")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(
        order="C"
    )


def _tensor_descriptor(value: torch.Tensor) -> dict[str, Any]:
    raw = _tensor_bytes(value)
    return {
        "bytes": len(raw),
        "dtype": str(value.dtype).removeprefix("torch."),
        "finite": bool(torch.all(torch.isfinite(value.float())).item()),
        "sha256": _sha256_bytes(raw),
        "shape": list(value.shape),
    }


def _package_sha256(files: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    digest.update(PACKAGE_HASH_DOMAIN)
    for filename, content in files:
        encoded = filename.encode("ascii")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<Q", len(content)))
        digest.update(content)
    return digest.hexdigest()


def _request_core(document: Mapping[str, Any]) -> dict[str, Any]:
    core_keys = set(_REQUEST_CORE_KEYS)
    if document.get("operation") == "final_norm":
        core_keys.add("source_checkpoint")
    _require(
        core_keys <= set(document),
        "request lacks fields required to derive request_id",
    )
    return {key: document[key] for key in sorted(core_keys)}


def derive_request_id(document_or_core: Mapping[str, Any]) -> str:
    """Derive the acyclic request ID from request semantics and tensor hashes."""

    core = _request_core(document_or_core)
    return _sha256_bytes(REQUEST_ID_DOMAIN + _canonical_json(core))


def _validate_artifacts(value: object, name: str) -> dict[str, Any]:
    artifacts = _require_exact_keys(
        value, set(PINNED_CHECKPOINT_ARTIFACTS), name
    )
    for filename, expected in PINNED_CHECKPOINT_ARTIFACTS.items():
        _require(artifacts[filename] == expected, f"{name}.{filename} mismatch")
    return artifacts


def _validate_identity(value: object) -> dict[str, Any]:
    identity = _normalized_json(value)
    identity = _require_exact_keys(identity, _IDENTITY_KEYS, "identity")

    checkpoint = _require_exact_keys(
        identity["checkpoint"], {"artifacts", "model_id", "revision"}, "identity.checkpoint"
    )
    _require(checkpoint["model_id"] == MODEL_ID, "identity checkpoint model mismatch")
    _require(checkpoint["revision"] == MODEL_REVISION, "identity checkpoint revision mismatch")
    _validate_artifacts(checkpoint["artifacts"], "identity.checkpoint.artifacts")

    model = _require_exact_keys(
        identity["model"], {"id", "layer_types", "num_layers", "revision"}, "identity.model"
    )
    _require(
        model
        == {
            "id": MODEL_ID,
            "layer_types": list(LAYER_TYPES),
            "num_layers": NUM_LAYERS,
            "revision": MODEL_REVISION,
        },
        "identity model topology mismatch",
    )

    formula = _require_exact_keys(
        identity["formula"], {"formula_source_sha256", "vllm_git_head"}, "identity.formula"
    )
    _require(
        isinstance(formula["vllm_git_head"], str)
        and GIT_OBJECT_RE.fullmatch(formula["vllm_git_head"]) is not None,
        "identity formula vLLM object ID is invalid",
    )
    source_hashes = _require_exact_keys(
        formula["formula_source_sha256"], FORMULA_SOURCE_PATHS, "identity.formula sources"
    )
    for path, digest in source_hashes.items():
        _require_sha256(digest, f"identity.formula source {path}")
        _require(
            _sha256_file(ROOT / path) == digest,
            f"identity.formula source changed: {path}",
        )
    observed_vllm_head = subprocess.check_output(
        ["git", "-C", str(ROOT / "projects/vllm"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    _require(
        formula["vllm_git_head"] == observed_vllm_head,
        "identity formula vLLM object ID mismatch",
    )

    plugin = _require_exact_keys(
        identity["plugin"], {"root", "source_sha256"}, "identity.plugin"
    )
    _require(plugin["root"] == PLUGIN_ROOT, "identity plugin root mismatch")
    plugin_hashes = _require_exact_keys(
        plugin["source_sha256"], PLUGIN_SOURCE_PATHS, "identity.plugin sources"
    )
    for path, digest in plugin_hashes.items():
        _require_sha256(digest, f"identity.plugin source {path}")
        _require(
            _sha256_file(ROOT / path) == digest,
            f"identity.plugin source changed: {path}",
        )

    runner = _require_exact_keys(identity["runner"], {"path", "sha256"}, "identity.runner")
    _require(
        isinstance(runner["path"], str)
        and bool(runner["path"])
        and not Path(runner["path"]).is_absolute()
        and ".." not in Path(runner["path"]).parts,
        "identity runner path must be a safe workspace-relative path",
    )
    _require_sha256(runner["sha256"], "identity.runner.sha256")
    runner_path = (ROOT / runner["path"]).resolve(strict=True)
    _require(runner_path.is_relative_to(ROOT), "identity runner path escapes workspace")
    _require(runner_path.is_file() and not runner_path.is_symlink(), "identity runner is unsafe")
    _require(
        _sha256_file(runner_path) == runner["sha256"],
        "identity runner source changed",
    )
    return identity


def current_layer_oracle_request_identity(
    runner_path: Path,
) -> dict[str, Any]:
    """Build the exact current request identity without importing target code."""

    runner = Path(runner_path).resolve(strict=True)
    _require(runner.is_relative_to(ROOT), "runner must be under the workspace")
    _require(runner.is_file() and not runner.is_symlink(), "runner source is unsafe")
    vllm_head = subprocess.check_output(
        ["git", "-C", str(ROOT / "projects/vllm"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    identity = {
        "checkpoint": {
            "artifacts": PINNED_CHECKPOINT_ARTIFACTS,
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
        },
        "formula": {
            "formula_source_sha256": {
                relative: _sha256_file(ROOT / relative)
                for relative in sorted(FORMULA_SOURCE_PATHS)
            },
            "vllm_git_head": vllm_head,
        },
        "model": {
            "id": MODEL_ID,
            "layer_types": list(LAYER_TYPES),
            "num_layers": NUM_LAYERS,
            "revision": MODEL_REVISION,
        },
        "plugin": {
            "root": PLUGIN_ROOT,
            "source_sha256": {
                relative: _sha256_file(ROOT / relative)
                for relative in sorted(PLUGIN_SOURCE_PATHS)
            },
        },
        "runner": {
            "path": str(runner.relative_to(ROOT)),
            "sha256": _sha256_file(runner),
        },
    }
    return _validate_identity(identity)


def _validate_operation(
    layer_index: object,
    operation: object,
    token_positions: object,
) -> tuple[int, str, str, tuple[int, ...]]:
    _require(
        type(layer_index) is int and 0 <= layer_index < NUM_LAYERS,
        "layer_index must be in [0,23]",
    )
    layer_type = LAYER_TYPES[layer_index]
    _require(
        operation in ("prefill_2", "decode_1", "final_norm"),
        "operation must be prefill_2, decode_1, or final_norm",
    )
    _require(
        isinstance(token_positions, (list, tuple))
        and all(type(position) is int for position in token_positions),
        "token_positions must be integer positions",
    )
    positions = tuple(token_positions)
    if operation == "final_norm":
        _require(layer_index == 23, "final_norm is only valid after layer 23")
        _require(positions == (0, 1), "final_norm positions must be exactly [0,1]")
        layer_type = "final_norm"
    elif operation == "prefill_2":
        _require(positions == (0, 1), "prefill_2 positions must be exactly [0,1]")
    else:
        _require(
            len(positions) == 1 and 2 <= positions[0] < KV_CACHE_SLOTS,
            "decode_1 position must be in [2,15]",
        )
    return layer_index, layer_type, operation, positions


def _cpu_contiguous(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    result = value.detach().cpu().contiguous()
    _require(
        bool(torch.all(torch.isfinite(result.float())).item()),
        f"{name} contains nonfinite values",
    )
    return result


def _prepare_request_tensors(
    *,
    layer_index: int,
    layer_type: str,
    operation: str,
    hidden_before: torch.Tensor,
    residual_before: torch.Tensor | None,
    mutable_state_before: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    tokens = 2 if operation in ("prefill_2", "final_norm") else 1
    hidden = _cpu_contiguous(hidden_before, "hidden_before")
    _require(
        hidden.dtype == torch.bfloat16 and tuple(hidden.shape) == (tokens, HIDDEN_SIZE),
        "hidden_before must be BF16 [tokens,1024]",
    )
    residual: torch.Tensor | None = None
    if operation == "final_norm":
        residual = _cpu_contiguous(residual_before, "residual_before")
        _require(
            residual.dtype == torch.bfloat16 and residual.shape == hidden.shape,
            "final_norm residual_before must match BF16 hidden_before",
        )
    elif layer_index == 0:
        _require(residual_before is None, "layer 0 residual_before must be absent")
    else:
        residual = _cpu_contiguous(residual_before, "residual_before")
        _require(
            residual.dtype == torch.bfloat16 and residual.shape == hidden.shape,
            "residual_before must match BF16 hidden_before",
        )

    payload = {"hidden_before": hidden}
    if residual is not None:
        payload["residual_before"] = residual
    runner_state: dict[str, torch.Tensor] = {}
    if operation == "final_norm":
        _require_exact_keys(mutable_state_before, set(), "final_norm mutable state")
    elif layer_type == "linear_attention":
        _require_exact_keys(mutable_state_before, {"conv_state", "recurrent_state"}, "GDN mutable state")
        conv = _cpu_contiguous(mutable_state_before["conv_state"], "conv_state")
        recurrent = _cpu_contiguous(
            mutable_state_before["recurrent_state"], "recurrent_state"
        )
        _require(
            conv.dtype == torch.bfloat16
            and tuple(conv.shape) == (1, 3, GDN_QKV_DIM),
            "runner GDN conv_state must be BF16 [1,3,6144] SD",
        )
        _require(
            recurrent.dtype == torch.float32
            and tuple(recurrent.shape) == (1, GDN_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM),
            "GDN recurrent_state must be FP32 [1,16,128,128]",
        )
        runner_state = {"conv_state": conv, "recurrent_state": recurrent}
        payload["gdn_conv_state_before"] = conv
        payload["gdn_recurrent_state_before"] = recurrent
    else:
        _require_exact_keys(mutable_state_before, {"kv_cache"}, "full-attention mutable state")
        cache = _cpu_contiguous(mutable_state_before["kv_cache"], "kv_cache")
        _require(
            cache.dtype == torch.bfloat16
            and tuple(cache.shape)
            == (1, KV_CACHE_SLOTS, FULL_KV_HEADS, FULL_KV_CONTENT),
            "full-attention kv_cache must be BF16 [1,16,2,512] NHD",
        )
        runner_state = {"kv_cache": cache}
        payload["full_attention_kv_cache_before"] = cache
    if operation == "prefill_2":
        _require(
            all(torch.count_nonzero(value).item() == 0 for value in runner_state.values()),
            "prefill_2 mutable state must be exactly zero",
        )
    return payload, runner_state


def _load_checkpoint_module():
    path = ROOT / "examples/triton/_qwen35_layer_checkpoint.py"
    spec = importlib.util.spec_from_file_location(
        "_qwen35_final_norm_checkpoint_protocol", path
    )
    _require(spec is not None and spec.loader is not None, "checkpoint loader is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_checkpoint_binding(source_checkpoint: Path) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    checkpoint_path = Path(source_checkpoint).expanduser()
    _require(checkpoint_path.is_absolute(), "source checkpoint path must be absolute")
    checkpoint = _load_checkpoint_module().load_layer_checkpoint(checkpoint_path)
    boundary = checkpoint.manifest["boundary"]
    _require(
        checkpoint.after_layer == 23
        and checkpoint.next_layer == 24
        and boundary["resume_action"] == "final_norm"
        and boundary["residual_present"] is True,
        "source checkpoint is not the exact post-layer-23 final_norm boundary",
    )
    positions = checkpoint.manifest["request"]["position_values"]
    _require(positions == [0, 1], "source checkpoint is not a two-token prefill")
    hidden_descriptor = boundary["hidden_states"]
    residual_descriptor = boundary["residual"]
    _require(isinstance(residual_descriptor, dict), "source checkpoint residual is absent")
    hidden = checkpoint.tensors[hidden_descriptor["key"]].clone().contiguous()
    residual = checkpoint.tensors[residual_descriptor["key"]].clone().contiguous()
    _require(
        hidden.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and tuple(hidden.shape) == (2, HIDDEN_SIZE)
        and residual.shape == hidden.shape,
        "source checkpoint final_norm activation contract mismatch",
    )
    binding = {
        "after_layer": checkpoint.after_layer,
        "binding_sha256": checkpoint.manifest["binding_sha256"],
        "directory": str(checkpoint.path),
        "hidden": _tensor_descriptor(hidden),
        "identity_sha256": checkpoint.manifest["identity_sha256"],
        "manifest_sha256": checkpoint.manifest_sha256,
        "next_layer": checkpoint.next_layer,
        "request_sha256": checkpoint.manifest["request_sha256"],
        "residual": _tensor_descriptor(residual),
        "resume_action": boundary["resume_action"],
        "schema": checkpoint.manifest["schema"],
        "state_sha256": checkpoint.manifest["artifacts"]["state"]["sha256"],
        "token_positions": positions,
    }
    _require_exact_keys(binding, _SOURCE_CHECKPOINT_KEYS, "source checkpoint binding")
    return binding, hidden, residual


def _embedded_request_metadata(
    request_id: str, operation: str = "prefill_2"
) -> dict[str, str]:
    return {
        "diagnostic_only": "true",
        "request_id": request_id,
        "role": (
            "before_final_norm_exact_live_amd_checkpoint_snapshot"
            if operation == "final_norm"
            else "before_layer_exact_live_amd_snapshot"
        ),
        "schema": REQUEST_SCHEMA,
        "state_snapshot": (
            "before_final_norm" if operation == "final_norm" else "before_layer"
        ),
    }


def _embedded_response_metadata(
    request_id: str, operation: str = "prefill_2"
) -> dict[str, str]:
    return {
        "diagnostic_only": "true",
        "request_id": request_id,
        "role": (
            "after_final_norm_independent_nvidia_diagnostic"
            if operation == "final_norm"
            else "after_layer_independent_nvidia_diagnostic"
        ),
        "schema": RESPONSE_SCHEMA,
        "state_snapshot": (
            "after_final_norm" if operation == "final_norm" else "after_layer"
        ),
        "target_feedback": TARGET_FEEDBACK_POLICY,
    }


def _safetensors_metadata(value: bytes) -> dict[str, str]:
    _require(len(value) >= 8, "safetensors payload is truncated")
    header_size = struct.unpack("<Q", value[:8])[0]
    _require(2 <= header_size <= MAX_JSON_BYTES, "safetensors header size is invalid")
    end = 8 + header_size
    _require(end <= len(value), "safetensors header exceeds payload")
    header = _strict_json_load(value[8:end], "safetensors header")
    _require(isinstance(header, dict), "safetensors header must be an object")
    metadata = header.get("__metadata__")
    _require(isinstance(metadata, dict), "safetensors metadata is absent")
    _require(
        all(isinstance(key, str) and isinstance(item, str) for key, item in metadata.items()),
        "safetensors metadata must contain strings",
    )
    return metadata


def _exclusive_write(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o400)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable; refusing non-atomic publish")
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


def _private_parent(path: Path) -> Path:
    _require(path.is_absolute(), "package path must be absolute")
    _require(path.name not in ("", ".", ".."), "package path must name a directory")
    parent = path.parent.resolve(strict=True)
    _require(path.parent == parent, "package parent path must be normalized and symlink-free")
    metadata = os.lstat(parent)
    _require(stat.S_ISDIR(metadata.st_mode), "package parent must be a real directory")
    _require(metadata.st_uid == os.getuid(), "package parent owner mismatch")
    _require(
        stat.S_IMODE(metadata.st_mode) & 0o077 == 0,
        "package parent must be private to its owner",
    )
    return parent


def _publish_package(
    output_dir: Path,
    *,
    json_filename: str,
    json_bytes: bytes,
    tensor_bytes: bytes,
) -> Path:
    output = Path(output_dir).expanduser()
    parent = _private_parent(output)
    destination = parent / output.name
    try:
        os.lstat(destination)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(errno.EEXIST, "oracle package already exists", destination)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent)
    )
    try:
        _exclusive_write(temporary / TENSOR_FILENAME, tensor_bytes)
        _exclusive_write(temporary / json_filename, json_bytes)
        _fsync_directory(temporary)
        os.chmod(temporary, 0o500)
        _rename_noreplace(temporary, destination)
        temporary = None
        _fsync_directory(parent)
    finally:
        if temporary is not None:
            try:
                os.chmod(temporary, 0o700)
            except FileNotFoundError:
                pass
            shutil.rmtree(temporary, ignore_errors=True)
    return destination


def _read_package(
    path: Path,
    *,
    json_filename: str,
) -> tuple[Path, bytes, bytes]:
    requested = Path(path).expanduser()
    _require(requested.is_absolute(), "oracle package path must be absolute")
    metadata = os.lstat(requested)
    _require(stat.S_ISDIR(metadata.st_mode), "oracle package must be a real directory")
    _require(metadata.st_uid == os.getuid(), "oracle package owner mismatch")
    _require(stat.S_IMODE(metadata.st_mode) == 0o500, "oracle package must have mode 0500")
    resolved = requested.resolve(strict=True)
    _require(resolved == requested, "oracle package path must be normalized and symlink-free")
    _private_parent(resolved)
    directory_fd = os.open(resolved, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        entries = list(os.scandir(directory_fd))
        _require(
            {entry.name for entry in entries} == {json_filename, TENSOR_FILENAME},
            "oracle package must contain exactly two named files",
        )
        _require(
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "oracle package entries must be regular files, not symlinks",
        )

        def read_entry(filename: str, maximum: int) -> bytes:
            descriptor = os.open(
                filename,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                info = os.fstat(descriptor)
                _require(stat.S_ISREG(info.st_mode), f"{filename} is not regular")
                _require(info.st_uid == os.getuid(), f"{filename} owner mismatch")
                _require(stat.S_IMODE(info.st_mode) == 0o400, f"{filename} must have mode 0400")
                _require(info.st_nlink == 1, f"{filename} must not be hard linked")
                _require(0 < info.st_size <= maximum, f"{filename} size is invalid")
                blocks: list[bytes] = []
                remaining = info.st_size
                while remaining:
                    block = os.read(descriptor, min(remaining, 1024 * 1024))
                    _require(bool(block), f"short read from {filename}")
                    blocks.append(block)
                    remaining -= len(block)
                _require(os.read(descriptor, 1) == b"", f"{filename} grew during read")
                return b"".join(blocks)
            finally:
                os.close(descriptor)

        json_bytes = read_entry(json_filename, MAX_JSON_BYTES)
        payload_bytes = read_entry(TENSOR_FILENAME, MAX_TENSOR_BYTES)
    finally:
        os.close(directory_fd)
    return resolved, json_bytes, payload_bytes


def _load_tensor_payload(
    payload_bytes: bytes,
    *,
    expected_metadata: Mapping[str, str],
    descriptors: object,
    roles: object,
) -> dict[str, torch.Tensor]:
    _require(
        _safetensors_metadata(payload_bytes) == dict(expected_metadata),
        "safetensors embedded metadata mismatch",
    )
    try:
        tensors = {
            key: value.clone().contiguous()
            for key, value in load_safetensors(payload_bytes).items()
        }
    except Exception as error:
        raise LayerOracleProtocolError(f"invalid safetensors payload: {error}") from error
    _require(
        isinstance(roles, list)
        and set(roles) == set(tensors)
        and len(roles) == len(set(roles)),
        "tensor role order/set mismatch",
    )
    records = _require_exact_keys(descriptors, set(tensors), "tensor descriptors")
    for key, tensor in tensors.items():
        record = _require_exact_keys(records[key], _TENSOR_DESCRIPTOR_KEYS, f"tensor {key}")
        observed = _tensor_descriptor(tensor)
        _require(observed == record, f"tensor descriptor mismatch: {key}")
        _require(record["finite"] is True, f"tensor contains nonfinite values: {key}")
    return tensors


def publish_layer_oracle_request(
    output_dir: Path,
    *,
    identity: Mapping[str, Any],
    layer_index: int,
    operation: str,
    token_positions: Sequence[int],
    hidden_before: torch.Tensor,
    residual_before: torch.Tensor | None,
    mutable_state_before: Mapping[str, torch.Tensor],
    source_checkpoint: Path | None = None,
) -> LayerOracleRequest:
    """Atomically publish an immutable exact before-layer request package."""

    identities = _validate_identity(identity)
    layer_index, layer_type, operation, positions = _validate_operation(
        layer_index, operation, token_positions
    )
    source_binding: dict[str, Any] | None = None
    if operation == "final_norm":
        _require(source_checkpoint is not None, "final_norm requires a source checkpoint")
        source_binding, checkpoint_hidden, checkpoint_residual = _source_checkpoint_binding(
            source_checkpoint
        )
        _require(
            torch.equal(_cpu_contiguous(hidden_before, "hidden_before"), checkpoint_hidden)
            and torch.equal(
                _cpu_contiguous(residual_before, "residual_before"), checkpoint_residual
            ),
            "final_norm tensors do not exactly match the source checkpoint",
        )
    else:
        _require(source_checkpoint is None, "decoder layer request cannot bind a source checkpoint")
    tensors, _runner_state = _prepare_request_tensors(
        layer_index=layer_index,
        layer_type=layer_type,
        operation=operation,
        hidden_before=hidden_before,
        residual_before=residual_before,
        mutable_state_before=mutable_state_before,
    )
    tensor_records = {key: _tensor_descriptor(value) for key, value in tensors.items()}
    core = {
        "diagnostic_only": True,
        "identities": identities,
        "kind": FINAL_NORM_REQUEST_KIND if operation == "final_norm" else REQUEST_KIND,
        "layer_index": layer_index,
        "layer_type": layer_type,
        "operation": operation,
        "residual_input": "absent" if residual_before is None else "present",
        "schema": REQUEST_SCHEMA,
        "state_snapshot": "before_final_norm" if operation == "final_norm" else "before_layer",
        "target_feedback": TARGET_FEEDBACK_POLICY,
        "tensor_roles": list(tensors),
        "tensors": tensor_records,
        "token_positions": list(positions),
    }
    if source_binding is not None:
        core["source_checkpoint"] = source_binding
    request_id = derive_request_id(core)
    embedded = _embedded_request_metadata(request_id, operation)
    payload_bytes = save_safetensors(tensors, metadata=embedded)
    document = {
        **core,
        "payload": {
            "bytes": len(payload_bytes),
            "embedded_metadata": embedded,
            "filename": TENSOR_FILENAME,
            "sha256": _sha256_bytes(payload_bytes),
        },
        "request_id": request_id,
    }
    json_bytes = _canonical_json(document) + b"\n"
    destination = _publish_package(
        output_dir,
        json_filename=REQUEST_JSON_FILENAME,
        json_bytes=json_bytes,
        tensor_bytes=payload_bytes,
    )
    return load_layer_oracle_request(destination)


def publish_final_norm_oracle_request(
    output_dir: Path,
    *,
    identity: Mapping[str, Any],
    source_checkpoint: Path,
) -> LayerOracleRequest:
    """Publish the exact post-layer-23 checkpoint boundary for final norm."""

    _, hidden, residual = _source_checkpoint_binding(source_checkpoint)
    return publish_layer_oracle_request(
        output_dir,
        identity=identity,
        layer_index=23,
        operation="final_norm",
        token_positions=[0, 1],
        hidden_before=hidden,
        residual_before=residual,
        mutable_state_before={},
        source_checkpoint=source_checkpoint,
    )


def load_layer_oracle_request(path: Path) -> LayerOracleRequest:
    """Read and fully validate an immutable request without target mutation."""

    resolved, json_bytes, payload_bytes = _read_package(
        path, json_filename=REQUEST_JSON_FILENAME
    )
    document = _strict_json_load(json_bytes, "oracle request")
    expected_request_keys = set(_REQUEST_KEYS)
    if isinstance(document, dict) and document.get("operation") == "final_norm":
        expected_request_keys.add("source_checkpoint")
    document = _require_exact_keys(document, expected_request_keys, "oracle request")
    _require(json_bytes == _canonical_json(document) + b"\n", "request JSON is not canonical")
    _require(document["schema"] == REQUEST_SCHEMA, "request schema mismatch")
    expected_kind = FINAL_NORM_REQUEST_KIND if document["operation"] == "final_norm" else REQUEST_KIND
    _require(document["kind"] == expected_kind, "request kind mismatch")
    _require(document["diagnostic_only"] is True, "request is not diagnostic-only")
    expected_snapshot = "before_final_norm" if document["operation"] == "final_norm" else "before_layer"
    _require(document["state_snapshot"] == expected_snapshot, "request snapshot mismatch")
    _require(document["target_feedback"] == TARGET_FEEDBACK_POLICY, "request target feedback is not prohibited")
    layer_index, layer_type, operation, positions = _validate_operation(
        document["layer_index"], document["operation"], document["token_positions"]
    )
    _require(document["layer_type"] == layer_type, "request layer type mismatch")
    expected_residual_input = "absent" if layer_index == 0 else "present"
    _require(document["residual_input"] == expected_residual_input, "request residual presence mismatch")
    identities = _validate_identity(document["identities"])
    request_id = _require_sha256(document["request_id"], "request_id")
    _require(request_id == derive_request_id(document), "request_id derivation mismatch")
    payload = _require_exact_keys(
        document["payload"], {"bytes", "embedded_metadata", "filename", "sha256"}, "request payload"
    )
    embedded = _embedded_request_metadata(request_id, operation)
    _require(payload["filename"] == TENSOR_FILENAME, "request payload filename mismatch")
    _require(payload["bytes"] == len(payload_bytes), "request payload byte count mismatch")
    _require(payload["sha256"] == _sha256_bytes(payload_bytes), "request payload hash mismatch")
    _require(payload["embedded_metadata"] == embedded, "request payload metadata declaration mismatch")
    tensors = _load_tensor_payload(
        payload_bytes,
        expected_metadata=embedded,
        descriptors=document["tensors"],
        roles=document["tensor_roles"],
    )
    hidden = tensors.get("hidden_before")
    residual = tensors.get("residual_before")
    source_binding: dict[str, Any] | None = None
    if operation == "final_norm":
        _require(
            document["tensor_roles"] == ["hidden_before", "residual_before"],
            "final_norm request tensor roles mismatch",
        )
        source_binding = _require_exact_keys(
            document["source_checkpoint"],
            _SOURCE_CHECKPOINT_KEYS,
            "source checkpoint binding",
        )
        observed_binding, checkpoint_hidden, checkpoint_residual = _source_checkpoint_binding(
            Path(source_binding["directory"])
        )
        _require(source_binding == observed_binding, "source checkpoint binding mismatch")
        _require(
            torch.equal(hidden, checkpoint_hidden)
            and torch.equal(residual, checkpoint_residual),
            "final_norm request tensors do not match the source checkpoint",
        )
        mutable = {}
    elif layer_type == "linear_attention":
        expected_roles = ["hidden_before"]
        if layer_index != 0:
            expected_roles.append("residual_before")
        expected_roles.extend(("gdn_conv_state_before", "gdn_recurrent_state_before"))
        _require(document["tensor_roles"] == expected_roles, "GDN request tensor roles mismatch")
        conv_oracle = tensors["gdn_conv_state_before"]
        recurrent = tensors["gdn_recurrent_state_before"]
        _require(
            conv_oracle.dtype == torch.bfloat16
            and tuple(conv_oracle.shape) == (1, 3, GDN_QKV_DIM),
            "oracle GDN conv state must be BF16 [1,3,6144]",
        )
        _require(
            recurrent.dtype == torch.float32
            and tuple(recurrent.shape) == (1, GDN_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM),
            "oracle GDN recurrent state contract mismatch",
        )
        mutable = {
            "conv_state": conv_oracle,
            "recurrent_state": recurrent,
        }
    else:
        expected_roles = ["hidden_before"]
        if layer_index != 0:
            expected_roles.append("residual_before")
        expected_roles.append("full_attention_kv_cache_before")
        _require(document["tensor_roles"] == expected_roles, "attention request tensor roles mismatch")
        cache = tensors["full_attention_kv_cache_before"]
        _require(
            cache.dtype == torch.bfloat16
            and tuple(cache.shape) == (1, KV_CACHE_SLOTS, FULL_KV_HEADS, FULL_KV_CONTENT),
            "attention cache must be BF16 [1,16,2,512] NHD",
        )
        mutable = {"kv_cache": cache}
        if operation == "decode_1":
            position = positions[0]
            _require(
                torch.count_nonzero(cache[:, position:]).item() == 0,
                "decode attention cache must be empty from the current slot onward",
            )
    _require(
        isinstance(hidden, torch.Tensor)
        and hidden.dtype == torch.bfloat16
        and tuple(hidden.shape)
        == ((2 if operation in ("prefill_2", "final_norm") else 1), HIDDEN_SIZE),
        "request hidden tensor contract mismatch",
    )
    if operation == "final_norm":
        _require(
            isinstance(residual, torch.Tensor)
            and residual.dtype == torch.bfloat16
            and residual.shape == hidden.shape,
            "final_norm request residual tensor contract mismatch",
        )
    elif layer_index == 0:
        _require(residual is None, "layer 0 request unexpectedly contains residual")
    else:
        _require(
            isinstance(residual, torch.Tensor)
            and residual.dtype == torch.bfloat16
            and residual.shape == hidden.shape,
            "request residual tensor contract mismatch",
        )
    if operation == "prefill_2":
        _require(
            all(torch.count_nonzero(value).item() == 0 for value in mutable.values()),
            "prefill request mutable state must be exactly zero",
        )
    return LayerOracleRequest(
        path=resolved,
        document=document,
        request_id=request_id,
        package_sha256=_package_sha256(
            ((REQUEST_JSON_FILENAME, json_bytes), (TENSOR_FILENAME, payload_bytes))
        ),
        request_json_sha256=_sha256_bytes(json_bytes),
        payload_sha256=_sha256_bytes(payload_bytes),
        identity_sha256=_sha256_bytes(_canonical_json(identities)),
        identities=identities,
        layer_index=layer_index,
        layer_type=layer_type,
        operation=operation,
        token_positions=positions,
        hidden_before=hidden,
        residual_before=residual,
        mutable_state_before=mutable,
        source_checkpoint=source_binding,
    )


def expected_local_oracle_identity(
    oracle_script: Path = DEFAULT_ORACLE_SCRIPT,
    golden_script: Path = DEFAULT_GOLDEN_SCRIPT,
) -> dict[str, Any]:
    """Return the exact local script and pinned RTX identity expected in a response."""

    oracle = Path(oracle_script).resolve(strict=True)
    golden = Path(golden_script).resolve(strict=True)
    _require(oracle.is_relative_to(ROOT) and golden.is_relative_to(ROOT), "oracle scripts must be under the workspace")
    _require(oracle.is_file() and not oracle.is_symlink(), "oracle script is unsafe")
    _require(golden.is_file() and not golden.is_symlink(), "golden script is unsafe")
    return {
        "gpu": dict(PINNED_GPU),
        "launcher": _oracle_launcher_identity(PINNED_ORACLE_PYTHON),
        "script": {
            "golden_path": str(golden.relative_to(ROOT)),
            "golden_sha256": _sha256_file(golden),
            "path": str(oracle.relative_to(ROOT)),
            "sha256": _sha256_file(oracle),
        },
    }


def _validate_expected_oracle_identity(value: object) -> dict[str, Any]:
    identity = _normalized_json(value)
    identity = _require_exact_keys(
        identity, {"gpu", "launcher", "script"}, "expected oracle identity"
    )
    _require(identity["gpu"] == PINNED_GPU, "expected oracle GPU is not the pinned RTX 5090")
    _require(
        identity["launcher"] == _oracle_launcher_identity(PINNED_ORACLE_PYTHON),
        "expected oracle launcher identity mismatch",
    )
    script = _require_exact_keys(
        identity["script"], {"golden_path", "golden_sha256", "path", "sha256"}, "expected oracle script"
    )
    for path_key in ("golden_path", "path"):
        _require(
            isinstance(script[path_key], str)
            and bool(script[path_key])
            and not Path(script[path_key]).is_absolute()
            and ".." not in Path(script[path_key]).parts,
            f"expected oracle {path_key} is unsafe",
        )
    _require_sha256(script["golden_sha256"], "expected oracle golden SHA-256")
    _require_sha256(script["sha256"], "expected oracle script SHA-256")
    return identity


def _oracle_environment() -> dict[str, str]:
    home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    _require(home.is_dir(), "oracle HOME is not a directory")
    return {
        "HOME": str(home),
        "LC_ALL": "C",
        "PATH": f"{PINNED_NVIDIA_SMI.parent}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }


def _oracle_launcher_identity(python: Path) -> dict[str, Any]:
    requested = Path(python).expanduser()
    _require(requested.is_absolute(), "oracle Python path must be absolute")
    executable = requested.resolve(strict=True)
    smi = PINNED_NVIDIA_SMI.resolve(strict=True)
    _require(executable.is_file() and os.access(executable, os.X_OK), "oracle Python is not executable")
    _require(smi == PINNED_NVIDIA_SMI and smi.is_file() and os.access(smi, os.X_OK), "pinned nvidia-smi is unsafe")
    return {
        "environment": _oracle_environment(),
        "nvidia_smi": {
            "bytes": smi.stat().st_size,
            "path": str(smi),
            "sha256": _sha256_file(smi),
        },
        "python": {
            "bytes": executable.stat().st_size,
            "path": str(requested),
            "resolved_path": str(executable),
            "sha256": _sha256_file(executable),
        },
    }


def _validate_response_checkpoint(value: object, request: LayerOracleRequest) -> None:
    checkpoint = _require_exact_keys(
        value,
        {"artifacts", "directory", "id", "index_total_size", "revision", "shard", "shard_manifest"},
        "response checkpoint",
    )
    _require(checkpoint["id"] == MODEL_ID, "response checkpoint model mismatch")
    _require(checkpoint["revision"] == MODEL_REVISION, "response checkpoint revision mismatch")
    _require(isinstance(checkpoint["directory"], str) and bool(checkpoint["directory"]), "response checkpoint directory is absent")
    _require(checkpoint["index_total_size"] == 1746882752, "response checkpoint index size mismatch")
    _require(checkpoint["shard"] == MODEL_SHARD, "response checkpoint shard mismatch")
    _require(
        checkpoint["shard_manifest"] == PINNED_CHECKPOINT_ARTIFACTS[MODEL_SHARD],
        "response checkpoint shard manifest mismatch",
    )
    artifacts = _require_exact_keys(
        checkpoint["artifacts"], set(PINNED_CHECKPOINT_ARTIFACTS), "response checkpoint artifacts"
    )
    for filename, expected in request.identities["checkpoint"]["artifacts"].items():
        _require(
            artifacts[filename] == {"expected": expected, "observed": expected},
            f"response checkpoint artifact mismatch: {filename}",
        )


def load_layer_oracle_response(
    path: Path,
    *,
    request: LayerOracleRequest,
    expected_oracle_identity: Mapping[str, Any],
) -> LayerOracleResponse:
    """Validate a diagnostic response exactly; never mutate the AMD snapshot."""

    expected_oracle = _validate_expected_oracle_identity(expected_oracle_identity)
    resolved, json_bytes, payload_bytes = _read_package(
        path, json_filename=RESPONSE_JSON_FILENAME
    )
    _require(resolved.parent == request.path.parent, "response is not a sibling of its request")
    document = _strict_json_load(json_bytes, "oracle response")
    expected_response_keys = {
        "created_utc",
        "diagnostic_only",
        "environment",
        "input_package",
        "kind",
        "layer_index",
        "layer_type",
        "operation",
        "oracle",
        "payload",
        "request_id",
        "result_roles",
        "schema",
        "state_snapshot",
        "target_feedback",
        "tensors",
        "timing",
        "token_positions",
    }
    document = _require_exact_keys(document, expected_response_keys, "oracle response")
    _require(json_bytes == _canonical_json(document) + b"\n", "response JSON is not canonical")
    _require(document["schema"] == RESPONSE_SCHEMA, "response schema mismatch")
    expected_response_kind = (
        FINAL_NORM_RESPONSE_KIND
        if request.operation == "final_norm"
        else RESPONSE_KIND
    )
    _require(document["kind"] == expected_response_kind, "response kind mismatch")
    _require(document["diagnostic_only"] is True, "response is not diagnostic-only")
    expected_response_snapshot = (
        "after_final_norm" if request.operation == "final_norm" else "after_layer"
    )
    _require(document["state_snapshot"] == expected_response_snapshot, "response snapshot mismatch")
    _require(document["target_feedback"] == TARGET_FEEDBACK_POLICY, "response target feedback is not prohibited")
    _require(document["request_id"] == request.request_id, "response request_id binding mismatch")
    _require(document["layer_index"] == request.layer_index, "response layer binding mismatch")
    _require(document["layer_type"] == request.layer_type, "response layer type binding mismatch")
    _require(document["operation"] == request.operation, "response operation binding mismatch")
    _require(tuple(document["token_positions"]) == request.token_positions, "response position binding mismatch")
    try:
        created = datetime.fromisoformat(document["created_utc"])
    except (TypeError, ValueError) as error:
        raise LayerOracleProtocolError("response created_utc is invalid") from error
    _require(created.tzinfo is not None, "response created_utc must be timezone-aware")

    input_package = _require_exact_keys(
        document["input_package"],
        {"directory", "package_sha256", "request_json_sha256", "tensor_file_sha256"},
        "response input package",
    )
    _require(
        input_package
        == {
            "directory": str(request.path),
            "package_sha256": request.package_sha256,
            "request_json_sha256": request.request_json_sha256,
            "tensor_file_sha256": request.payload_sha256,
        },
        "response input package binding mismatch",
    )
    oracle = _require_exact_keys(
        document["oracle"],
        {
            "checkpoint",
            "execution_boundary",
            "formula",
            "plugin_identity_from_request",
            "request_identities",
            "runner_identity_from_request",
            "script",
        },
        "response oracle identity",
    )
    _validate_response_checkpoint(oracle["checkpoint"], request)
    _require(oracle["formula"] == request.identities["formula"], "response formula identity binding mismatch")
    _require(oracle["plugin_identity_from_request"] == request.identities["plugin"], "response plugin identity binding mismatch")
    _require(oracle["request_identities"] == request.identities, "response complete request identity binding mismatch")
    _require(oracle["runner_identity_from_request"] == request.identities["runner"], "response runner identity binding mismatch")
    _require(oracle["script"] == expected_oracle["script"], "response oracle script identity mismatch")
    _require(oracle["execution_boundary"] == EXECUTION_BOUNDARY, "response execution boundary mismatch")

    environment = _require_exact_keys(
        document["environment"],
        {
            "cuda_runtime_version",
            "cudnn_version",
            "deterministic_algorithms",
            "float32_matmul_precision",
            "gpu",
            "platform",
            "python_executable",
            "python_version",
            "tf32_cudnn",
            "tf32_matmul",
            "torch_version",
        },
        "response environment",
    )
    gpu = environment["gpu"]
    _require(isinstance(gpu, dict), "response GPU identity is absent")
    _require(
        all(gpu.get(key) == value for key, value in expected_oracle["gpu"].items()),
        "response GPU is not the expected pinned RTX 5090",
    )
    _require(
        environment["deterministic_algorithms"] is True
        and environment["tf32_cudnn"] is False
        and environment["tf32_matmul"] is False
        and environment["float32_matmul_precision"] == "highest",
        "response deterministic CUDA environment mismatch",
    )
    timing = _require_exact_keys(
        document["timing"],
        {"compute_seconds", "model_checkpoint_validate_and_load_seconds"},
        "response timing",
    )
    _require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            for value in timing.values()
        ),
        "response timing is invalid",
    )

    payload = _require_exact_keys(
        document["payload"], {"bytes", "embedded_metadata", "filename", "sha256"}, "response payload"
    )
    embedded = _embedded_response_metadata(request.request_id, request.operation)
    _require(payload["filename"] == TENSOR_FILENAME, "response payload filename mismatch")
    _require(payload["bytes"] == len(payload_bytes), "response payload byte count mismatch")
    _require(payload["sha256"] == _sha256_bytes(payload_bytes), "response payload hash mismatch")
    _require(payload["embedded_metadata"] == embedded, "response payload metadata declaration mismatch")
    tensors = _load_tensor_payload(
        payload_bytes,
        expected_metadata=embedded,
        descriptors=document["tensors"],
        roles=document["result_roles"],
    )
    tokens = 2 if request.operation in ("prefill_2", "final_norm") else 1
    hidden: torch.Tensor | None = tensors.get("hidden_after")
    residual: torch.Tensor | None = tensors.get("residual_after")
    final_hidden: torch.Tensor | None = tensors.get("final_hidden_after")
    if request.operation == "final_norm":
        _require(
            document["result_roles"] == ["final_hidden_after"],
            "final_norm response must contain only final_hidden_after",
        )
        _require(
            hidden is None
            and residual is None
            and isinstance(final_hidden, torch.Tensor)
            and final_hidden.dtype == torch.bfloat16
            and tuple(final_hidden.shape) == (2, HIDDEN_SIZE),
            "final_norm response tensor contract mismatch",
        )
        mutable = {}
    else:
        _require(
            isinstance(hidden, torch.Tensor)
            and hidden.dtype == torch.bfloat16
            and tuple(hidden.shape) == (tokens, HIDDEN_SIZE),
            "response hidden tensor contract mismatch",
        )
        _require(
            isinstance(residual, torch.Tensor)
            and residual.dtype == torch.bfloat16
            and residual.shape == hidden.shape,
            "response residual tensor contract mismatch",
        )
        _require(final_hidden is None, "decoder layer response contains final_hidden_after")
    if request.operation != "final_norm" and request.layer_type == "linear_attention":
        expected_roles = [
            "hidden_after",
            "residual_after",
            "gdn_conv_state_after",
            "gdn_recurrent_state_after",
        ]
        _require(document["result_roles"] == expected_roles, "response GDN tensor roles mismatch")
        conv = tensors["gdn_conv_state_after"]
        recurrent = tensors["gdn_recurrent_state_after"]
        _require(
            conv.dtype == torch.bfloat16 and tuple(conv.shape) == (1, 3, GDN_QKV_DIM),
            "response oracle GDN conv state contract mismatch",
        )
        _require(
            recurrent.dtype == torch.float32
            and tuple(recurrent.shape) == (1, GDN_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM),
            "response GDN recurrent state contract mismatch",
        )
        mutable = {
            "conv_state": conv,
            "recurrent_state": recurrent,
        }
    elif request.operation != "final_norm":
        expected_roles = ["hidden_after", "residual_after", "full_attention_kv_cache_after"]
        _require(document["result_roles"] == expected_roles, "response attention tensor roles mismatch")
        cache = tensors["full_attention_kv_cache_after"]
        _require(
            cache.dtype == torch.bfloat16
            and tuple(cache.shape) == (1, KV_CACHE_SLOTS, FULL_KV_HEADS, FULL_KV_CONTENT),
            "response attention cache contract mismatch",
        )
        mutable = {"kv_cache": cache}
    return LayerOracleResponse(
        path=resolved,
        document=document,
        request_id=request.request_id,
        package_sha256=_package_sha256(
            ((RESPONSE_JSON_FILENAME, json_bytes), (TENSOR_FILENAME, payload_bytes))
        ),
        response_json_sha256=_sha256_bytes(json_bytes),
        payload_sha256=_sha256_bytes(payload_bytes),
        identity_sha256=request.identity_sha256,
        identities=request.identities,
        oracle_identity=expected_oracle,
        layer_index=request.layer_index,
        layer_type=request.layer_type,
        operation=request.operation,
        token_positions=request.token_positions,
        hidden_after=hidden,
        residual_after=residual,
        final_hidden_after=final_hidden,
        mutable_state_after=mutable,
    )


def run_layer_oracle(
    *,
    request: LayerOracleRequest,
    response_dir: Path,
    expected_oracle_identity: Mapping[str, Any],
    oracle_script: Path = DEFAULT_ORACLE_SCRIPT,
    python: Path = PINNED_ORACLE_PYTHON,
    device: int = 0,
    timeout_seconds: float = 300.0,
) -> LayerOracleResponse:
    """Run one oracle subprocess, then return only a fully validated diagnostic."""

    expected_oracle = _validate_expected_oracle_identity(expected_oracle_identity)
    _require(type(device) is int and device >= 0, "oracle CUDA device must be nonnegative")
    _require(
        isinstance(timeout_seconds, (int, float))
        and not isinstance(timeout_seconds, bool)
        and math.isfinite(timeout_seconds)
        and timeout_seconds > 0,
        "oracle timeout must be positive and finite",
    )
    requested_executable = Path(python).expanduser()
    _require(requested_executable.is_absolute(), "oracle Python path must be absolute")
    executable = requested_executable.resolve(strict=True)
    script = Path(oracle_script).resolve(strict=True)
    _require(executable.is_file() and os.access(executable, os.X_OK), "oracle Python is not executable")
    _require(script.is_file() and not script.is_symlink(), "oracle script is unsafe")
    response = Path(response_dir).expanduser()
    _require(response.is_absolute(), "oracle response path must be absolute")
    _require(response.parent.resolve(strict=True) == request.path.parent, "oracle response must be a request sibling")
    launcher_identity = _oracle_launcher_identity(requested_executable)
    _require(
        launcher_identity == expected_oracle["launcher"],
        "requested oracle launcher does not match the expected identity",
    )
    oracle_environment = launcher_identity["environment"]
    argv = (
        str(requested_executable),
        str(script),
        "--input-dir",
        str(request.path),
        "--output-dir",
        str(response),
        "--device",
        str(device),
    )
    try:
        completed = subprocess.run(
            argv,
            check=True,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(timeout_seconds),
            env=oracle_environment,
        )
    except subprocess.CalledProcessError as error:
        record = LayerOracleExecutionRecord(
            argv=argv,
            environment=oracle_environment,
            exit_code=error.returncode,
            launcher_identity=launcher_identity,
            stdout=error.stdout or "",
            stderr=error.stderr or "",
        )
        raise LayerOracleExecutionError(
            f"NVIDIA layer oracle exited with status {error.returncode}", record
        ) from error
    except subprocess.TimeoutExpired as error:
        record = LayerOracleExecutionRecord(
            argv=argv,
            environment=oracle_environment,
            exit_code=None,
            launcher_identity=launcher_identity,
            stdout=error.stdout or "",
            stderr=error.stderr or "",
        )
        raise LayerOracleExecutionError(
            f"NVIDIA layer oracle exceeded {float(timeout_seconds)} seconds",
            record,
        ) from error
    except OSError as error:
        record = LayerOracleExecutionRecord(
            argv=argv,
            environment=oracle_environment,
            exit_code=None,
            launcher_identity=launcher_identity,
            stdout="",
            stderr=str(error),
        )
        raise LayerOracleExecutionError(
            f"NVIDIA layer oracle could not start: {error}", record
        ) from error
    result = load_layer_oracle_response(
        response,
        request=request,
        expected_oracle_identity=expected_oracle,
    )
    return replace(
        result,
        execution_record=LayerOracleExecutionRecord(
            argv=argv,
            environment=oracle_environment,
            exit_code=completed.returncode,
            launcher_identity=launcher_identity,
            stdout=completed.stdout,
            stderr=completed.stderr,
        ),
    )


__all__ = [
    "DEFAULT_GOLDEN_SCRIPT",
    "DEFAULT_ORACLE_SCRIPT",
    "LayerOracleExecutionError",
    "LayerOracleExecutionRecord",
    "LayerOracleProtocolError",
    "LayerOracleRequest",
    "LayerOracleResponse",
    "current_layer_oracle_request_identity",
    "derive_request_id",
    "expected_local_oracle_identity",
    "load_layer_oracle_request",
    "load_layer_oracle_response",
    "publish_final_norm_oracle_request",
    "publish_layer_oracle_request",
    "run_layer_oracle",
]
