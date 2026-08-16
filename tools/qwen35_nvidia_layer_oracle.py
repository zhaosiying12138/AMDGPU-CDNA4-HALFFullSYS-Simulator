#!/usr/bin/env python3
"""Run one diagnostic-only Qwen3.5 layer oracle on pinned NVIDIA CUDA.

The request is an immutable two-file package containing an exact live AMD
before-layer snapshot.  The response is a new same-parent, atomically
published two-file package.  This source never imports vLLM, project Triton,
GemSim, gem5, or an AMD target implementation.  PyTorch may internally load
upstream Triton as a CUDA framework detail; it is never invoked as a target or
fallback by this oracle.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRITON_DEV_PYTHON = Path(
    "/home/zhaosiying/miniforge3/envs/triton-dev/bin/python3"
)
DEFAULT_MODEL_DIR = ROOT / "models/Qwen3.5-0.8B"
REQUEST_SCHEMA = "amdgpu-sim.qwen35-nvidia-layer-oracle-request.v1"
RESPONSE_SCHEMA = "amdgpu-sim.qwen35-nvidia-layer-oracle-response.v1"
REQUEST_KIND = "exact_live_amd_before_layer_diagnostic"
RESPONSE_KIND = "independent_torch_cuda_after_layer_diagnostic"
FINAL_NORM_REQUEST_KIND = "exact_live_amd_before_final_norm_diagnostic"
FINAL_NORM_RESPONSE_KIND = "independent_torch_cuda_after_final_norm_diagnostic"
REQUEST_TENSOR_FILE = "tensors.safetensors"
REQUEST_JSON_FILE = "request.json"
RESPONSE_TENSOR_FILE = "tensors.safetensors"
RESPONSE_JSON_FILE = "response.json"
PACKAGE_HASH_DOMAIN = b"amdgpu-sim.qwen35-layer-oracle-package.v1\0"
REQUEST_ID_DOMAIN = b"amdgpu-sim.qwen35-layer-oracle-request-id.v1\0"
AT_FDCWD = -100
RENAME_NOREPLACE = 1
MAX_JSON_BYTES = 1024 * 1024
MAX_TENSOR_BYTES = 32 * 1024 * 1024
KV_CACHE_SLOTS = 16
PINNED_GPU = {
    "name": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "uuid": "GPU-64aae36b-ef77-b0d4-b1c7-f7ab17a729f1",
    "compute_capability": [12, 0],
}
PLUGIN_SOURCE_PATHS = (
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/adapters.py",
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/attention.py",
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/kernels.py",
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/model.py",
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/ops.py",
)
FORBIDDEN_PROJECT_MODULE_ROOTS = ("gemsim_vllm", "m5", "gem5")
ALLOWED_FRAMEWORK_INTERNAL_MODULE_ROOTS = ("triton",)
EXECUTION_BOUNDARY = {
    "allowed_framework_internal_modules": list(
        ALLOWED_FRAMEWORK_INTERNAL_MODULE_ROOTS
    ),
    "direct_triton_import": False,
    "forbidden_project_module_roots": list(FORBIDDEN_PROJECT_MODULE_ROOTS),
    "implementation": "pure_torch_cuda_formula",
    "non_target_execution": True,
}
REQUEST_CORE_KEYS = {
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
SOURCE_CHECKPOINT_KEYS = {
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


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Compute a diagnostic-only RTX 5090 golden for one pinned "
            "Qwen3.5-0.8B decoder layer"
        )
    )
    value.add_argument("--input-dir", type=Path)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    value.add_argument("--device", type=int, default=0)
    value.add_argument(
        "--describe-schema",
        action="store_true",
        help="print the exact request/response contract without loading CUDA",
    )
    return value


def maybe_reexec_triton_dev() -> None:
    required = TRITON_DEV_PYTHON.resolve(strict=True)
    if Path(sys.executable).resolve() == required:
        return
    os.execv(required, [str(required), str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    maybe_reexec_triton_dev()


import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import qwen35_nvidia_golden as backbone_golden


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def assert_independent_imports() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if name.split(".", 1)[0] in FORBIDDEN_PROJECT_MODULE_ROOTS
    )
    require(not imported, f"forbidden target/backend modules were imported: {imported}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_bytes(value: bytes, description: str) -> dict[str, Any]:
    try:
        decoded = value.decode("utf-8")
        result = json.loads(decoded, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"malformed {description}: {error}") from error
    require(isinstance(result, dict), f"{description} must be a JSON object")
    return result


def canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def derive_request_id(value: dict[str, Any]) -> str:
    core_keys = set(REQUEST_CORE_KEYS)
    if value.get("operation") == "final_norm":
        core_keys.add("source_checkpoint")
    require(
        core_keys <= set(value),
        "request lacks fields required to derive request_id",
    )
    core = {key: value[key] for key in sorted(core_keys)}
    canonical = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256_bytes(REQUEST_ID_DOMAIN + canonical)


def exact_keys(value: object, keys: set[str], description: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{description} must be an object")
    observed = set(value)
    require(
        observed == keys,
        f"{description} key set mismatch: expected {sorted(keys)}, got {sorted(observed)}",
    )
    return value


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(
        order="C"
    )


def tensor_descriptor(value: torch.Tensor) -> dict[str, Any]:
    raw = tensor_bytes(value)
    return {
        "bytes": len(raw),
        "dtype": str(value.dtype).removeprefix("torch."),
        "finite": bool(torch.all(torch.isfinite(value.float())).item()),
        "sha256": sha256_bytes(raw),
        "shape": list(value.shape),
    }


def package_sha256(files: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    digest.update(PACKAGE_HASH_DOMAIN)
    for name, content in files:
        encoded_name = name.encode("ascii")
        digest.update(struct.pack("<Q", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<Q", len(content)))
        digest.update(content)
    return digest.hexdigest()


def check_path_chain_is_resolved(path: Path, *, must_exist: bool) -> Path:
    require(path.is_absolute(), f"path must be absolute: {path}")
    if must_exist:
        resolved = path.resolve(strict=True)
    else:
        resolved_parent = path.parent.resolve(strict=True)
        resolved = resolved_parent / path.name
    require(path == resolved, f"path must be normalized and contain no symlink: {path}")
    return resolved


def validate_private_parent(parent: Path) -> None:
    check_path_chain_is_resolved(parent, must_exist=True)
    info = os.lstat(parent)
    require(stat.S_ISDIR(info.st_mode), f"package parent is not a directory: {parent}")
    require(info.st_uid == os.getuid(), f"package parent has a foreign owner: {parent}")
    require(
        info.st_mode & 0o022 == 0,
        f"package parent must not be group/other writable: {parent}",
    )


def validate_immutable_input_directory(path: Path) -> Path:
    resolved = check_path_chain_is_resolved(path, must_exist=True)
    info = os.lstat(resolved)
    require(stat.S_ISDIR(info.st_mode), f"request package is not a directory: {path}")
    require(info.st_uid == os.getuid(), "request package has a foreign owner")
    require(info.st_mode & 0o222 == 0, "request package directory is mutable")
    entries = {entry.name for entry in os.scandir(resolved)}
    require(
        entries == {REQUEST_JSON_FILE, REQUEST_TENSOR_FILE},
        f"request package file set mismatch: {sorted(entries)}",
    )
    validate_private_parent(resolved.parent)
    return resolved


def read_immutable_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), f"package entry is not regular: {path.name}")
        require(info.st_uid == os.getuid(), f"package entry has a foreign owner: {path.name}")
        require(info.st_mode & 0o222 == 0, f"package entry is mutable: {path.name}")
        require(0 < info.st_size <= maximum_bytes, f"package entry size is invalid: {path.name}")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            require(block != b"", f"short read from package entry: {path.name}")
            chunks.append(block)
            remaining -= len(block)
        extra = os.read(descriptor, 1)
        require(extra == b"", f"package entry grew while reading: {path.name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def safetensors_metadata(value: bytes) -> dict[str, str]:
    require(len(value) >= 8, "safetensors payload is truncated")
    header_bytes = struct.unpack("<Q", value[:8])[0]
    require(2 <= header_bytes <= MAX_JSON_BYTES, "safetensors header size is invalid")
    end = 8 + header_bytes
    require(end <= len(value), "safetensors header exceeds the payload")
    header = parse_json_bytes(value[8:end], "safetensors header")
    metadata = header.get("__metadata__")
    require(isinstance(metadata, dict), "safetensors metadata is missing")
    require(
        all(isinstance(key, str) and isinstance(item, str) for key, item in metadata.items()),
        "safetensors metadata must contain only strings",
    )
    return metadata


def plugin_identity() -> dict[str, Any]:
    records = {}
    for relative in PLUGIN_SOURCE_PATHS:
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"plugin source is unsafe: {relative}")
        records[relative] = sha256_file(path)
    return {
        "root": "plugins/framework/gemsim_vllm",
        "source_sha256": records,
    }


def checkpoint_request_identity() -> dict[str, Any]:
    return {
        "artifacts": backbone_golden.PINNED_ARTIFACTS,
        "model_id": backbone_golden.PINNED_MODEL_ID,
        "revision": backbone_golden.PINNED_REVISION,
    }


def model_request_identity() -> dict[str, Any]:
    return {
        "id": backbone_golden.PINNED_MODEL_ID,
        "layer_types": backbone_golden.LAYER_TYPES,
        "num_layers": backbone_golden.NUM_LAYERS,
        "revision": backbone_golden.PINNED_REVISION,
    }


def validate_runner_identity(value: object) -> dict[str, str]:
    runner = exact_keys(value, {"path", "sha256"}, "runner identity")
    relative = runner["path"]
    require(isinstance(relative, str) and relative != "", "runner path is invalid")
    candidate = ROOT / relative
    require(not Path(relative).is_absolute(), "runner path must be workspace-relative")
    resolved = candidate.resolve(strict=True)
    require(resolved.is_relative_to(ROOT), "runner path escapes the workspace")
    require(resolved == candidate and resolved.is_file(), "runner path is unsafe")
    require(not resolved.is_symlink(), "runner path must not be a symlink")
    require(is_sha256(runner["sha256"]), "runner SHA-256 is malformed")
    require(sha256_file(resolved) == runner["sha256"], "runner identity mismatch")
    return {"path": relative, "sha256": runner["sha256"]}


def expected_embedded_request_metadata(
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


def input_contract(
    layer_index: int,
    operation: str,
    residual_input: str,
) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    tokens = 2 if operation in ("prefill_2", "final_norm") else 1
    contracts: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {
        "hidden_before": (torch.bfloat16, (tokens, backbone_golden.HIDDEN_SIZE)),
    }
    if residual_input == "present":
        contracts["residual_before"] = (
            torch.bfloat16,
            (tokens, backbone_golden.HIDDEN_SIZE),
        )
    if operation == "final_norm":
        return contracts
    layer_type = backbone_golden.LAYER_TYPES[layer_index]
    if layer_type == "linear_attention":
        contracts.update(
            {
                "gdn_conv_state_before": (
                    torch.bfloat16,
                    (1, backbone_golden.CONV_STATE_WIDTH, backbone_golden.QKV_DIM),
                ),
                "gdn_recurrent_state_before": (
                    torch.float32,
                    (1, backbone_golden.NUM_HEADS, backbone_golden.HEAD_DIM, backbone_golden.HEAD_DIM),
                ),
            }
        )
    else:
        contracts["full_attention_kv_cache_before"] = (
            torch.bfloat16,
            (
                1,
                KV_CACHE_SLOTS,
                backbone_golden.FULL_NUM_KV_HEADS,
                2 * backbone_golden.FULL_HEAD_DIM,
            ),
        )
    return contracts


def load_checkpoint_protocol():
    path = ROOT / "examples/triton/_qwen35_layer_checkpoint.py"
    spec = importlib.util.spec_from_file_location(
        "_qwen35_nvidia_final_norm_checkpoint_protocol", path
    )
    require(spec is not None and spec.loader is not None, "checkpoint loader is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_source_checkpoint(
    value: object,
    request_tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    binding = exact_keys(value, SOURCE_CHECKPOINT_KEYS, "source checkpoint binding")
    directory_value = binding["directory"]
    require(
        isinstance(directory_value, str) and Path(directory_value).is_absolute(),
        "source checkpoint path must be absolute",
    )
    checkpoint = load_checkpoint_protocol().load_layer_checkpoint(Path(directory_value))
    boundary = checkpoint.manifest["boundary"]
    require(
        checkpoint.after_layer == 23
        and checkpoint.next_layer == 24
        and boundary["resume_action"] == "final_norm"
        and boundary["residual_present"] is True,
        "source checkpoint is not the exact post-layer-23 final_norm boundary",
    )
    positions = checkpoint.manifest["request"]["position_values"]
    require(positions == [0, 1], "source checkpoint is not a two-token prefill")
    hidden_record = boundary["hidden_states"]
    residual_record = boundary["residual"]
    require(isinstance(residual_record, dict), "source checkpoint residual is absent")
    hidden = checkpoint.tensors[hidden_record["key"]].clone().contiguous()
    residual = checkpoint.tensors[residual_record["key"]].clone().contiguous()
    require(
        hidden.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and tuple(hidden.shape) == (2, backbone_golden.HIDDEN_SIZE)
        and residual.shape == hidden.shape,
        "source checkpoint final_norm activation contract mismatch",
    )
    observed = {
        "after_layer": checkpoint.after_layer,
        "binding_sha256": checkpoint.manifest["binding_sha256"],
        "directory": str(checkpoint.path),
        "hidden": tensor_descriptor(hidden),
        "identity_sha256": checkpoint.manifest["identity_sha256"],
        "manifest_sha256": checkpoint.manifest_sha256,
        "next_layer": checkpoint.next_layer,
        "request_sha256": checkpoint.manifest["request_sha256"],
        "residual": tensor_descriptor(residual),
        "resume_action": boundary["resume_action"],
        "schema": checkpoint.manifest["schema"],
        "state_sha256": checkpoint.manifest["artifacts"]["state"]["sha256"],
        "token_positions": positions,
    }
    require(binding == observed, "source checkpoint binding mismatch")
    require(
        torch.equal(request_tensors["hidden_before"], hidden)
        and torch.equal(request_tensors["residual_before"], residual),
        "final_norm request tensors do not match the source checkpoint",
    )
    return observed


def validate_identities(value: object) -> dict[str, Any]:
    identities = exact_keys(
        value,
        {"checkpoint", "formula", "model", "plugin", "runner"},
        "request identities",
    )
    runner = validate_runner_identity(identities["runner"])
    require(identities["model"] == model_request_identity(), "model identity mismatch")
    require(
        identities["checkpoint"] == checkpoint_request_identity(),
        "checkpoint request identity mismatch",
    )
    expected_formula = backbone_golden.source_identity()
    require(identities["formula"] == expected_formula, "formula source identity mismatch")
    expected_plugin = plugin_identity()
    require(identities["plugin"] == expected_plugin, "plugin source identity mismatch")
    return {
        "checkpoint": checkpoint_request_identity(),
        "formula": expected_formula,
        "model": model_request_identity(),
        "plugin": expected_plugin,
        "runner": runner,
    }


class RequestPackage:
    def __init__(
        self,
        directory: Path,
        document: dict[str, Any],
        tensors: dict[str, torch.Tensor],
        request_bytes: bytes,
        tensor_file_bytes: bytes,
        identities: dict[str, Any],
    ) -> None:
        self.directory = directory
        self.document = document
        self.tensors = tensors
        self.request_bytes = request_bytes
        self.tensor_file_bytes = tensor_file_bytes
        self.identities = identities
        self.package_sha256 = package_sha256(
            [
                (REQUEST_JSON_FILE, request_bytes),
                (REQUEST_TENSOR_FILE, tensor_file_bytes),
            ]
        )


def load_request_package(path: Path) -> RequestPackage:
    directory = validate_immutable_input_directory(path)
    request_bytes = read_immutable_file(directory / REQUEST_JSON_FILE, MAX_JSON_BYTES)
    tensor_file_bytes = read_immutable_file(
        directory / REQUEST_TENSOR_FILE, MAX_TENSOR_BYTES
    )
    request = parse_json_bytes(request_bytes, REQUEST_JSON_FILE)
    require(
        request_bytes == canonical_json_bytes(request),
        "request.json is not canonical sorted compact JSON with one trailing newline",
    )
    request_keys = {
        "diagnostic_only",
        "identities",
        "kind",
        "layer_index",
        "layer_type",
        "operation",
        "payload",
        "request_id",
        "residual_input",
        "schema",
        "state_snapshot",
        "target_feedback",
        "tensor_roles",
        "tensors",
        "token_positions",
    }
    if isinstance(request, dict) and request.get("operation") == "final_norm":
        request_keys.add("source_checkpoint")
    exact_keys(request, request_keys, "request")
    require(request["schema"] == REQUEST_SCHEMA, "request schema mismatch")
    operation = request["operation"]
    require(
        operation in ("prefill_2", "decode_1", "final_norm"),
        "unsupported operation",
    )
    expected_kind = FINAL_NORM_REQUEST_KIND if operation == "final_norm" else REQUEST_KIND
    require(request["kind"] == expected_kind, "request kind mismatch")
    require(request["diagnostic_only"] is True, "request is not diagnostic-only")
    require(request["target_feedback"] == "prohibited", "target feedback must be prohibited")
    expected_snapshot = "before_final_norm" if operation == "final_norm" else "before_layer"
    require(request["state_snapshot"] == expected_snapshot, "request state snapshot mismatch")
    request_id = request["request_id"]
    require(is_sha256(request_id), "request_id must be a lowercase 64-hex value")
    require(request_id == derive_request_id(request), "request_id derivation mismatch")
    layer_index = request["layer_index"]
    require(
        isinstance(layer_index, int)
        and not isinstance(layer_index, bool)
        and 0 <= layer_index < backbone_golden.NUM_LAYERS,
        "layer_index is outside [0,23]",
    )
    expected_layer_type = (
        "final_norm" if operation == "final_norm" else backbone_golden.LAYER_TYPES[layer_index]
    )
    require(request["layer_type"] == expected_layer_type, "layer type schedule mismatch")
    positions = request["token_positions"]
    if operation == "final_norm":
        require(layer_index == 23, "final_norm is only valid after layer 23")
        require(positions == [0, 1], "final_norm positions must be exactly [0,1]")
    elif operation == "prefill_2":
        require(positions == [0, 1], "prefill_2 positions must be exactly [0,1]")
    else:
        require(
            isinstance(positions, list)
            and len(positions) == 1
            and isinstance(positions[0], int)
            and not isinstance(positions[0], bool)
            and 2 <= positions[0] < KV_CACHE_SLOTS,
            f"decode_1 position must be in [2,{KV_CACHE_SLOTS - 1}]",
        )
    residual_input = request["residual_input"]
    expected_residual = "absent" if layer_index == 0 else "present"
    require(residual_input == expected_residual, "residual input presence mismatch")
    identities = validate_identities(request["identities"])

    embedded_metadata = expected_embedded_request_metadata(request_id, operation)
    payload = exact_keys(
        request["payload"],
        {"bytes", "embedded_metadata", "filename", "sha256"},
        "request payload",
    )
    require(payload["filename"] == REQUEST_TENSOR_FILE, "request tensor filename mismatch")
    require(payload["bytes"] == len(tensor_file_bytes), "request tensor byte count mismatch")
    require(payload["sha256"] == sha256_bytes(tensor_file_bytes), "request tensor file hash mismatch")
    require(payload["embedded_metadata"] == embedded_metadata, "request embedded metadata declaration mismatch")
    require(safetensors_metadata(tensor_file_bytes) == embedded_metadata, "request safetensors metadata mismatch")
    try:
        tensors = {
            name: tensor.clone().contiguous()
            for name, tensor in load_safetensors(tensor_file_bytes).items()
        }
    except Exception as error:
        raise RuntimeError(f"invalid request safetensors payload: {error}") from error
    contracts = input_contract(layer_index, operation, residual_input)
    require(set(tensors) == set(contracts), "request tensor role set mismatch")
    require(request["tensor_roles"] == list(contracts), "request tensor role order mismatch")
    tensor_records = exact_keys(request["tensors"], set(contracts), "request tensor descriptors")
    for name, (dtype, shape) in contracts.items():
        value = tensors[name]
        require(value.dtype == dtype, f"request tensor dtype mismatch: {name}")
        require(tuple(value.shape) == shape, f"request tensor shape mismatch: {name}")
        observed = tensor_descriptor(value)
        require(observed == tensor_records[name], f"request tensor descriptor mismatch: {name}")
        require(observed["finite"] is True, f"request tensor contains nonfinite values: {name}")

    if operation == "final_norm":
        require(
            request["tensor_roles"] == ["hidden_before", "residual_before"],
            "final_norm request tensor roles mismatch",
        )
        validate_source_checkpoint(request["source_checkpoint"], tensors)

    if operation == "prefill_2":
        state_names = [name for name in contracts if name.endswith("_before") and name != "hidden_before" and name != "residual_before"]
        require(
            all(torch.count_nonzero(tensors[name]).item() == 0 for name in state_names),
            "prefill_2 requires an exact zero before-layer mutable state",
        )
    elif operation == "decode_1" and request["layer_type"] == "full_attention":
        position = positions[0]
        cache = tensors["full_attention_kv_cache_before"]
        require(
            torch.count_nonzero(cache[:, position:]).item() == 0,
            "decode_1 full-attention cache must be empty from the current slot onward",
        )
    return RequestPackage(
        directory,
        request,
        tensors,
        request_bytes,
        tensor_file_bytes,
        identities,
    )


def configure_cuda(device_index: int) -> tuple[torch.device, dict[str, Any]]:
    require(torch.cuda.is_available(), "CUDA is unavailable in triton-dev PyTorch")
    require(0 <= device_index < torch.cuda.device_count(), "CUDA device index is invalid")
    torch.cuda.set_device(device_index)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    gpu = backbone_golden.gpu_identity(device_index)
    require(
        all(gpu.get(name) == value for name, value in PINNED_GPU.items()),
        f"CUDA device is not the pinned RTX 5090: {gpu}",
    )
    environment = {
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "gpu": gpu,
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "torch_version": torch.__version__,
    }
    return torch.device("cuda", device_index), environment


def apply_text_rope_positions(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        backbone_golden.FULL_ROPE_THETA
        ** (
            torch.arange(
                0,
                backbone_golden.FULL_ROTARY_DIM,
                2,
                dtype=torch.float32,
            )
            / backbone_golden.FULL_ROTARY_DIM
        )
    )
    position_tensor = torch.tensor(positions, dtype=torch.float32)
    frequencies = torch.einsum("i,j->ij", position_tensor, inv_freq)
    cosine = frequencies.cos().to(torch.bfloat16).to(query.device).float()
    sine = frequencies.sin().to(torch.bfloat16).to(query.device).float()

    def rotate(value: torch.Tensor) -> torch.Tensor:
        output = value.clone()
        half = backbone_golden.FULL_ROTARY_DIM // 2
        first = value[..., :half].float()
        second = value[..., half : 2 * half].float()
        cos = cosine[:, None, :]
        sin = sine[:, None, :]
        output[..., :half] = (first * cos - second * sin).to(torch.bfloat16)
        output[..., half : 2 * half] = (second * cos + first * sin).to(torch.bfloat16)
        return output

    return rotate(query), rotate(key)


def gdn_attention_live(
    normalized: torch.Tensor,
    weights: dict[str, torch.Tensor],
    conv_state_before: torch.Tensor,
    recurrent_state_before: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = normalized.shape[0]
    qkvz = backbone_golden.bf16_linear(normalized, weights["qkvz"])
    ba = backbone_golden.bf16_linear(normalized, weights["ba"])
    mixed_input = qkvz[:, : backbone_golden.QKV_DIM]
    selected = conv_state_before[0].transpose(0, 1).contiguous()
    conv_weight = weights["conv"].float()
    mixed_rows = []
    for token in range(tokens):
        current = mixed_input[token].float()
        accumulator = (
            selected[:, 0].float() * conv_weight[:, 0]
            + selected[:, 1].float() * conv_weight[:, 1]
            + selected[:, 2].float() * conv_weight[:, 2]
            + current * conv_weight[:, 3]
        )
        mixed_rows.append((accumulator * torch.sigmoid(accumulator)).to(torch.bfloat16))
        selected = torch.stack((selected[:, 1], selected[:, 2], mixed_input[token]), dim=1)
    mixed_qkv = torch.stack(mixed_rows, dim=0)
    conv_state_after = conv_state_before.clone()
    conv_state_after[0] = selected.transpose(0, 1)

    recurrent_state = recurrent_state_before[0].clone()
    recurrent_rows = []
    for token in range(tokens):
        mixed = mixed_qkv[token].float()
        query = mixed[: backbone_golden.NUM_HEADS * backbone_golden.HEAD_DIM].view(
            backbone_golden.NUM_HEADS, backbone_golden.HEAD_DIM
        )
        key = mixed[
            backbone_golden.NUM_HEADS * backbone_golden.HEAD_DIM :
            2 * backbone_golden.NUM_HEADS * backbone_golden.HEAD_DIM
        ].view(backbone_golden.NUM_HEADS, backbone_golden.HEAD_DIM)
        recurrent_value = mixed[
            2 * backbone_golden.NUM_HEADS * backbone_golden.HEAD_DIM :
        ].view(backbone_golden.NUM_HEADS, backbone_golden.HEAD_DIM)
        query = query * torch.rsqrt(torch.sum(query * query, dim=-1, keepdim=True) + epsilon)
        key = key * torch.rsqrt(torch.sum(key * key, dim=-1, keepdim=True) + epsilon)
        query = query * (backbone_golden.HEAD_DIM**-0.5)
        b = ba[token, : backbone_golden.NUM_HEADS].float()
        a = ba[token, backbone_golden.NUM_HEADS :].float()
        softplus_input = a + weights["dt_bias"].float()
        softplus = torch.where(
            softplus_input <= 20.0,
            torch.log(1.0 + torch.exp(softplus_input)),
            softplus_input,
        )
        decay = torch.exp(-torch.exp(weights["a_log"]) * softplus)
        recurrent_state = recurrent_state * decay.view(backbone_golden.NUM_HEADS, 1, 1)
        prediction = torch.sum(recurrent_state * key[:, None, :], dim=-1)
        beta = torch.sigmoid(b).to(torch.bfloat16).float()
        delta = (recurrent_value - prediction) * beta[:, None]
        recurrent_state = recurrent_state + delta[:, :, None] * key[:, None, :]
        recurrent_rows.append(
            torch.sum(recurrent_state * query[:, None, :], dim=-1).to(torch.bfloat16)
        )
    recurrent_output = torch.stack(recurrent_rows, dim=0)
    recurrent_float = recurrent_output.float()
    variance = torch.mean(recurrent_float * recurrent_float, dim=-1, keepdim=True)
    normalized_recurrent = recurrent_float * torch.rsqrt(variance + epsilon)
    z = qkvz[:, backbone_golden.QKV_DIM :].view(
        tokens, backbone_golden.NUM_HEADS, backbone_golden.HEAD_DIM
    ).float()
    gated = (
        normalized_recurrent
        * weights["output_norm"].float()
        * (z * torch.sigmoid(z))
    ).to(torch.bfloat16)
    output_weight = weights.get("attention_out", weights.get("gdn_out"))
    require(output_weight is not None, "GDN output projection weight is absent")
    attention = backbone_golden.bf16_linear(
        gated.view(tokens, backbone_golden.Z_DIM), output_weight
    )
    recurrent_state_after = recurrent_state_before.clone()
    recurrent_state_after[0] = recurrent_state
    return attention, conv_state_after, recurrent_state_after


def full_attention_live(
    normalized: torch.Tensor,
    weights: dict[str, torch.Tensor],
    cache_before: torch.Tensor,
    positions: list[int],
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = normalized.shape[0]
    q_gate_projection = backbone_golden.bf16_linear(normalized, weights["q_gate"])
    q_gate = q_gate_projection.view(
        tokens,
        backbone_golden.FULL_NUM_HEADS,
        2 * backbone_golden.FULL_HEAD_DIM,
    )
    query, gate = torch.chunk(q_gate, 2, dim=-1)
    key = backbone_golden.bf16_linear(normalized, weights["k"]).view(
        tokens, backbone_golden.FULL_NUM_KV_HEADS, backbone_golden.FULL_HEAD_DIM
    )
    value = backbone_golden.bf16_linear(normalized, weights["v"]).view(
        tokens, backbone_golden.FULL_NUM_KV_HEADS, backbone_golden.FULL_HEAD_DIM
    )
    query = backbone_golden.gemma_rms_formula(query, weights["q_norm"], epsilon)
    key = backbone_golden.gemma_rms_formula(key, weights["k_norm"], epsilon)
    query, key = apply_text_rope_positions(query, key, positions)
    cache_after = cache_before.clone()
    for token, position in enumerate(positions):
        cache_after[0, position, :, : backbone_golden.FULL_HEAD_DIM] = key[token]
        cache_after[0, position, :, backbone_golden.FULL_HEAD_DIM :] = value[token]

    rows = []
    for token, position in enumerate(positions):
        heads = []
        for query_head in range(backbone_golden.FULL_NUM_HEADS):
            kv_head = query_head // backbone_golden.FULL_GQA_GROUP_SIZE
            cached_key = cache_after[
                0, : position + 1, kv_head, : backbone_golden.FULL_HEAD_DIM
            ].float()
            cached_value = cache_after[
                0, : position + 1, kv_head, backbone_golden.FULL_HEAD_DIM :
            ].float()
            scores = torch.sum(query[token, query_head].float()[None, :] * cached_key, dim=-1)
            scores = scores * (backbone_golden.FULL_HEAD_DIM**-0.5)
            probabilities = torch.softmax(scores, dim=0)
            heads.append(
                torch.sum(probabilities[:, None] * cached_value, dim=0).to(torch.bfloat16)
            )
        rows.append(torch.stack(heads, dim=0))
    attention = torch.stack(rows, dim=0)
    sigmoid_gate = torch.sigmoid(gate.float()).to(torch.bfloat16).float()
    gated = (attention.float() * sigmoid_gate).to(torch.bfloat16)
    projected = backbone_golden.bf16_linear(
        gated.view(tokens, backbone_golden.FULL_Q_SIZE), weights["attention_out"]
    )
    return projected, cache_after


class OracleModel:
    """Validated 24-layer CUDA weights reusable across sequential requests."""

    def __init__(self, model_dir: Path, device: torch.device) -> None:
        self.model_dir = model_dir.resolve(strict=True)
        self.device = device
        load_start = time.perf_counter()
        config, index, manifest, shard, artifacts = backbone_golden.validate_checkpoint(
            self.model_dir
        )
        _, layer0, _ = backbone_golden.load_inputs(
            self.model_dir, backbone_golden.DEFAULT_TOKEN_ID, shard
        )
        additional, final_norm, _, _ = backbone_golden.load_additional_backbone_weights(
            index, shard, backbone_golden.NUM_LAYERS
        )
        require(final_norm is not None, "final norm weight load is incomplete")
        cpu_layers = {0: layer0, **additional}
        require(set(cpu_layers) == set(range(backbone_golden.NUM_LAYERS)), "24-layer weight load is incomplete")
        self.layers = {
            layer: {name: value.to(device) for name, value in weights.items()}
            for layer, weights in cpu_layers.items()
        }
        self.final_norm = final_norm.to(device)
        del cpu_layers, layer0, additional, final_norm
        torch.cuda.synchronize(device)
        self.epsilon = float(config["text_config"]["rms_norm_eps"])
        self.checkpoint = {
            "artifacts": artifacts,
            "directory": str(self.model_dir),
            "id": backbone_golden.PINNED_MODEL_ID,
            "index_total_size": index.get("metadata", {}).get("total_size"),
            "revision": backbone_golden.PINNED_REVISION,
            "shard": shard.name,
            "shard_manifest": manifest["files"][shard.name],
        }
        self.formula = backbone_golden.source_identity()
        self.load_seconds = time.perf_counter() - load_start

    def execute(self, package: RequestPackage) -> tuple[dict[str, torch.Tensor], float]:
        request = package.document
        values = {name: tensor.to(self.device) for name, tensor in package.tensors.items()}
        hidden = values["hidden_before"]
        if request["operation"] == "final_norm":
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            final_hidden, _ = backbone_golden.fused_gemma_rms_formula(
                hidden,
                values["residual_before"],
                self.final_norm,
                self.epsilon,
            )
            torch.cuda.synchronize(self.device)
            compute_seconds = time.perf_counter() - start
            results = {"final_hidden_after": final_hidden}
            cpu_results = {
                name: value.detach().cpu().contiguous()
                for name, value in results.items()
            }
            require(
                all(tensor_descriptor(value)["finite"] for value in cpu_results.values()),
                "oracle produced a nonfinite final norm result",
            )
            return cpu_results, compute_seconds

        layer_index = request["layer_index"]
        weights = self.layers[layer_index]
        if request["residual_input"] == "absent":
            normalized = backbone_golden.gemma_rms_formula(
                hidden, weights["input_norm"], self.epsilon
            )
            residual = hidden.clone()
        else:
            normalized, residual = backbone_golden.fused_gemma_rms_formula(
                hidden,
                values["residual_before"],
                weights["input_norm"],
                self.epsilon,
            )
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        if request["layer_type"] == "linear_attention":
            attention, conv_after, recurrent_after = gdn_attention_live(
                normalized,
                weights,
                values["gdn_conv_state_before"],
                values["gdn_recurrent_state_before"],
                self.epsilon,
            )
            state_results = {
                "gdn_conv_state_after": conv_after,
                "gdn_recurrent_state_after": recurrent_after,
            }
        else:
            attention, cache_after = full_attention_live(
                normalized,
                weights,
                values["full_attention_kv_cache_before"],
                request["token_positions"],
                self.epsilon,
            )
            state_results = {"full_attention_kv_cache_after": cache_after}
        post_norm, residual_after = backbone_golden.fused_gemma_rms_formula(
            attention, residual, weights["post_attention_norm"], self.epsilon
        )
        gate_up = backbone_golden.bf16_linear(post_norm, weights["gate_up"])
        gate = gate_up[:, : backbone_golden.INTERMEDIATE_SIZE].float()
        up = gate_up[:, backbone_golden.INTERMEDIATE_SIZE :].float()
        activated = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
        hidden_after = backbone_golden.bf16_linear(activated, weights["down"])
        torch.cuda.synchronize(self.device)
        compute_seconds = time.perf_counter() - start
        results = {
            "hidden_after": hidden_after,
            "residual_after": residual_after,
            **state_results,
        }
        cpu_results = {
            name: value.detach().cpu().contiguous() for name, value in results.items()
        }
        require(
            all(tensor_descriptor(value)["finite"] for value in cpu_results.values()),
            "oracle produced a nonfinite result",
        )
        return cpu_results, compute_seconds


def source_script_identity() -> dict[str, str]:
    script = Path(__file__).resolve()
    golden_script = Path(backbone_golden.__file__).resolve()
    return {
        "golden_path": str(golden_script.relative_to(ROOT)),
        "golden_sha256": sha256_file(golden_script),
        "path": str(script.relative_to(ROOT)),
        "sha256": sha256_file(script),
    }


def write_exclusive(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o400)


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
        raise OSError(errno.ENOSYS, "renameat2 is unavailable; refusing non-atomic publish")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        observed_errno = ctypes.get_errno()
        raise OSError(observed_errno, os.strerror(observed_errno), destination)


def validate_output_destination(input_dir: Path, output_dir: Path) -> Path:
    output = check_path_chain_is_resolved(output_dir, must_exist=False)
    require(output.parent == input_dir.parent, "response must be a sibling of the request package")
    require(output.name not in ("", ".", ".."), "response directory name is invalid")
    require(not output.exists() and not output.is_symlink(), "response path already exists")
    validate_private_parent(output.parent)
    return output


def publish_response(
    output_dir: Path,
    package: RequestPackage,
    results: dict[str, torch.Tensor],
    model: OracleModel,
    environment: dict[str, Any],
    compute_seconds: float,
) -> dict[str, Any]:
    result_roles = list(results)
    operation = package.document["operation"]
    final_norm = operation == "final_norm"
    embedded = {
        "diagnostic_only": "true",
        "request_id": package.document["request_id"],
        "role": (
            "after_final_norm_independent_nvidia_diagnostic"
            if final_norm
            else "after_layer_independent_nvidia_diagnostic"
        ),
        "schema": RESPONSE_SCHEMA,
        "state_snapshot": "after_final_norm" if final_norm else "after_layer",
        "target_feedback": "prohibited",
    }
    tensor_file_bytes = save_safetensors(results, metadata=embedded)
    descriptors = {name: tensor_descriptor(value) for name, value in results.items()}
    response = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "environment": environment,
        "input_package": {
            "directory": str(package.directory),
            "package_sha256": package.package_sha256,
            "request_json_sha256": sha256_bytes(package.request_bytes),
            "tensor_file_sha256": sha256_bytes(package.tensor_file_bytes),
        },
        "kind": FINAL_NORM_RESPONSE_KIND if final_norm else RESPONSE_KIND,
        "layer_index": package.document["layer_index"],
        "layer_type": package.document["layer_type"],
        "operation": package.document["operation"],
        "oracle": {
            "checkpoint": model.checkpoint,
            "execution_boundary": EXECUTION_BOUNDARY,
            "formula": model.formula,
            "plugin_identity_from_request": package.identities["plugin"],
            "request_identities": package.identities,
            "runner_identity_from_request": package.identities["runner"],
            "script": source_script_identity(),
        },
        "payload": {
            "bytes": len(tensor_file_bytes),
            "embedded_metadata": embedded,
            "filename": RESPONSE_TENSOR_FILE,
            "sha256": sha256_bytes(tensor_file_bytes),
        },
        "request_id": package.document["request_id"],
        "result_roles": result_roles,
        "schema": RESPONSE_SCHEMA,
        "state_snapshot": "after_final_norm" if final_norm else "after_layer",
        "target_feedback": "prohibited",
        "tensors": descriptors,
        "timing": {
            "compute_seconds": compute_seconds,
            "model_checkpoint_validate_and_load_seconds": model.load_seconds,
        },
        "token_positions": package.document["token_positions"],
    }
    response_bytes = canonical_json_bytes(response)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=".qwen35-layer-oracle-", dir=output_dir.parent)
    )
    try:
        write_exclusive(temporary / RESPONSE_TENSOR_FILE, tensor_file_bytes)
        write_exclusive(temporary / RESPONSE_JSON_FILE, response_bytes)
        fsync_directory(temporary)
        os.chmod(temporary, 0o500)
        rename_noreplace(temporary, output_dir)
        temporary = None
        fsync_directory(output_dir.parent)
    finally:
        if temporary is not None:
            try:
                os.chmod(temporary, 0o700)
            except FileNotFoundError:
                pass
            shutil.rmtree(temporary, ignore_errors=True)
    require(
        {entry.name for entry in os.scandir(output_dir)}
        == {RESPONSE_JSON_FILE, RESPONSE_TENSOR_FILE},
        "published response file set mismatch",
    )
    return response


def schema_description() -> dict[str, Any]:
    return {
        "cli": {
            "single_request": (
                f"{TRITON_DEV_PYTHON} tools/qwen35_nvidia_layer_oracle.py "
                "--input-dir /abs/private/request --output-dir /abs/private/response"
            ),
            "atomicity": "response is a new same-parent directory published with renameat2(RENAME_NOREPLACE)",
        },
        "operations": {
            "decode_1": {"positions": "one integer in [2,15]", "tokens": 1},
            "final_norm": {
                "layer_index": 23,
                "positions": [0, 1],
                "source": "exact post-layer-23 checkpoint",
                "tokens": 2,
            },
            "prefill_2": {"positions": [0, 1], "state": "exact zero", "tokens": 2},
        },
        "request": {
            "files": [REQUEST_JSON_FILE, REQUEST_TENSOR_FILE],
            "immutable": "directory and files have no write bits; no symlinks or extra entries",
            "schema": REQUEST_SCHEMA,
            "state_snapshot": {
                "decoder_layer": "before_layer",
                "final_norm": "before_final_norm",
            },
            "tensor_roles": {
                "common": ["hidden_before", "residual_before except layer 0"],
                "final_norm": ["hidden_before", "residual_before"],
                "full_attention": ["full_attention_kv_cache_before"],
                "linear_attention": ["gdn_conv_state_before", "gdn_recurrent_state_before"],
            },
        },
        "response": {
            "files": [RESPONSE_JSON_FILE, RESPONSE_TENSOR_FILE],
            "schema": RESPONSE_SCHEMA,
            "state_snapshot": {
                "decoder_layer": "after_layer",
                "final_norm": "after_final_norm",
            },
            "tensor_roles": {
                "common": ["hidden_after", "residual_after"],
                "final_norm": ["final_hidden_after"],
                "full_attention": ["full_attention_kv_cache_after"],
                "linear_attention": ["gdn_conv_state_after", "gdn_recurrent_state_after"],
            },
        },
        "safety": {
            "diagnostic_only": True,
            "execution_boundary": EXECUTION_BOUNDARY,
            "fallback": "none",
            "target_feedback": "prohibited",
        },
    }


def main() -> int:
    args = parser().parse_args()
    assert_independent_imports()
    if args.describe_schema:
        require(args.input_dir is None and args.output_dir is None, "schema mode accepts no package paths")
        print(json.dumps(schema_description(), indent=2, sort_keys=True))
        return 0
    require(args.input_dir is not None, "--input-dir is required")
    require(args.output_dir is not None, "--output-dir is required")
    package = load_request_package(args.input_dir)
    output_dir = validate_output_destination(package.directory, args.output_dir)
    device, environment = configure_cuda(args.device)
    model = OracleModel(args.model_dir, device)
    require(model.formula == package.identities["formula"], "loaded formula identity changed after request validation")
    results, compute_seconds = model.execute(package)
    response = publish_response(
        output_dir, package, results, model, environment, compute_seconds
    )
    assert_independent_imports()
    print(
        json.dumps(
            {
                "diagnostic_only": True,
                "input_package_sha256": package.package_sha256,
                "output_dir": str(output_dir),
                "payload_sha256": response["payload"]["sha256"],
                "request_id": package.document["request_id"],
                "state_snapshot": (
                    "after_final_norm"
                    if package.document["operation"] == "final_norm"
                    else "after_layer"
                ),
                "target_feedback": "prohibited",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
