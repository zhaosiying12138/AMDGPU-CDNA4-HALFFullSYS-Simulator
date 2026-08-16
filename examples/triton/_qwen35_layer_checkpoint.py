"""Fail-closed layer checkpoints for the Qwen3.5 vLLM forward runner.

The initial checkpoint is after embedding and before decoder layer 0, with an
absent residual.  Subsequent checkpoints are after a decoder layer has returned
its ``(hidden_states, residual)`` pair and before the next action (another layer
or final norm).  Every checkpoint contains all runtime cache records, including
untouched suffix layers.  This module deliberately has no vLLM or simulator
dependency so its persistence and restore contracts can be tested on the host.

Version 1 restore intentionally requires exact implementation identity and
rejects a gem5, runtime, plugin, vLLM, prefix, or runner change.  It supports
pure tensor parallel groups (pipeline parallelism remains one); continuing
after an implementation change still requires checkpoint regeneration.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import stat
import struct
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
import torch


CHECKPOINT_SCHEMA = "amdgpu-sim.qwen35-vllm-layer-checkpoint.v1"
PUBLISH_RESULT_SCHEMA = "amdgpu-sim.qwen35-vllm-layer-checkpoint-publish.v1"
RESTORE_RESULT_SCHEMA = "amdgpu-sim.qwen35-vllm-layer-checkpoint-restore.v1"
CHECKPOINT_KIND = "decoder_layer_boundary_before_next_action"
WRITE_POLICY = (
    "same_parent_temp_directory_fsync_renameat2_noreplace_parent_fsync"
)
MANIFEST_FILENAME = "manifest.json"
STATE_FILENAME = "state.safetensors"

AT_FDCWD = -100
RENAME_NOREPLACE = 1
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_STATE_BYTES = 512 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GENERATOR_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")

_IDENTITY_KEYS = {
    "model",
    "implementation",
    "target",
    "decoder",
    "parallelism",
    "weights",
}
_REQUEST_IDENTITY_KEYS = {
    "sequence_id",
    "phase",
    "step_index",
    "context_length_before",
    "context_length_after",
    "scheduler",
}
_MANIFEST_CORE_KEYS = {
    "schema",
    "kind",
    "created_utc",
    "write_policy",
    "identity",
    "identity_sha256",
    "request",
    "request_sha256",
    "boundary",
    "runtime_caches",
    "rng",
    "lineage",
}
_MANIFEST_KEYS = _MANIFEST_CORE_KEYS | {"binding_sha256", "artifacts"}
_IMPLEMENTATION_KEYS = {
    "architecture",
    "runner_sha256",
    "plugin_sha256",
    "vllm_git_head",
    "vllm_tree_sha256",
    "gem5_binary_sha256",
    "runtime_dso_sha256",
    "prefix_manifest_sha256",
}
_TENSOR_DESCRIPTOR_KEYS = {
    "key",
    "role",
    "dtype",
    "shape",
    "source_device",
    "source_stride",
    "source_storage_offset",
    "stored_stride",
    "numel",
    "bytes",
    "sha256",
    "nonfinite_count",
}


class LayerCheckpointError(RuntimeError):
    """The checkpoint is malformed, incompatible, or unsafe to restore."""


@dataclass(frozen=True)
class LoadedLayerCheckpoint:
    path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    tensors: dict[str, torch.Tensor]

    @property
    def after_layer(self) -> int:
        return self.manifest["boundary"]["after_layer"]

    @property
    def next_layer(self) -> int:
        return self.manifest["boundary"]["next_layer"]


@dataclass(frozen=True)
class RestoredLayerCheckpoint:
    hidden_states: torch.Tensor
    residual: torch.Tensor | None
    input_ids: torch.Tensor
    positions: torch.Tensor
    next_layer: int
    manifest: dict[str, Any]
    result: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LayerCheckpointError(message)


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], name: str) -> None:
    observed = set(value)
    _require(
        observed == keys,
        f"{name} keys mismatch: missing={sorted(keys - observed)}, "
        f"extra={sorted(observed - keys)}",
    )


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _require_after_layer(value: Any, layer_count: int, name: str) -> int:
    _require(
        type(value) is int and -1 <= value < layer_count,
        f"{name} must be an integer in [-1,{layer_count - 1}]",
    )
    return value


def _require_string(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"{name} must be lowercase SHA-256")
    return value


def _json_default(_value: object) -> object:
    raise TypeError("checkpoint identity values must be JSON-native")


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=_json_default,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise LayerCheckpointError(f"value is not canonical JSON: {error}") from error


def _normalized_json(value: Any) -> Any:
    return _strict_json_load(_canonical_json(value), "canonical value")


def _strict_json_load(value: bytes, name: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise LayerCheckpointError(f"duplicate JSON key in {name}: {key}")
            result[key] = item
        return result

    try:
        return json.loads(value.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except LayerCheckpointError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LayerCheckpointError(f"invalid JSON in {name}: {error}") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _validate_artifact_records(value: Any, name: str) -> None:
    _require(isinstance(value, dict) and bool(value), f"{name} must be a non-empty object")
    for filename, record in value.items():
        _require_string(filename, f"{name} filename")
        _require(Path(filename).name == filename, f"{name} filename must be a basename: {filename}")
        _require(isinstance(record, dict), f"{name}.{filename} must be an object")
        _require_exact_keys(record, {"bytes", "sha256"}, f"{name}.{filename}")
        _require_int(record["bytes"], f"{name}.{filename}.bytes", minimum=1)
        _require_sha256(record["sha256"], f"{name}.{filename}.sha256")


def _validate_identity(value: Any) -> dict[str, Any]:
    identity = _normalized_json(value)
    _require(isinstance(identity, dict), "identity must be an object")
    _require_exact_keys(identity, _IDENTITY_KEYS, "identity")

    model = identity["model"]
    _require(isinstance(model, dict), "identity.model must be an object")
    _require_exact_keys(model, {"id", "revision", "artifacts"}, "identity.model")
    _require_string(model["id"], "identity.model.id")
    _require_string(model["revision"], "identity.model.revision")
    _validate_artifact_records(model["artifacts"], "identity.model.artifacts")

    implementation = identity["implementation"]
    _require(isinstance(implementation, dict), "identity.implementation must be an object")
    _require_exact_keys(
        implementation, _IMPLEMENTATION_KEYS, "identity.implementation"
    )
    _require_string(implementation["architecture"], "identity.implementation.architecture")
    for key in _IMPLEMENTATION_KEYS - {"architecture", "vllm_git_head"}:
        _require_sha256(
            implementation[key], f"identity.implementation.{key}"
        )
    git_head = implementation["vllm_git_head"]
    _require(
        isinstance(git_head, str)
        and len(git_head) in (40, 64)
        and all(character in "0123456789abcdef" for character in git_head),
        "identity.implementation.vllm_git_head must be a lowercase object ID",
    )

    target = identity["target"]
    _require(isinstance(target, dict), "identity.target must be an object")
    _require_exact_keys(
        target,
        {"backend", "arch", "device", "fallback_allowed", "stochastic_ops"},
        "identity.target",
    )
    for key in ("backend", "arch", "device"):
        _require_string(target[key], f"identity.target.{key}")
    _require(target["fallback_allowed"] is False, "checkpoint restore forbids fallback")
    _require(target["stochastic_ops"] is False, "Qwen inference checkpoint requires stochastic_ops=false")

    decoder = identity["decoder"]
    _require(isinstance(decoder, dict), "identity.decoder must be an object")
    _require_exact_keys(
        decoder,
        {
            "layer_count",
            "hidden_size",
            "activation_dtype",
            "layer_types",
            "gdn_conv_state_shape",
            "gdn_recurrent_state_shape",
            "kv_cache_shape",
            "gdn_conv_state_dtype",
            "gdn_recurrent_state_dtype",
            "kv_cache_dtype",
        },
        "identity.decoder",
    )
    layer_count = _require_int(decoder["layer_count"], "identity.decoder.layer_count", minimum=1)
    _require_int(decoder["hidden_size"], "identity.decoder.hidden_size", minimum=1)
    _require_string(decoder["activation_dtype"], "identity.decoder.activation_dtype")
    layer_types = decoder["layer_types"]
    _require(
        isinstance(layer_types, list)
        and len(layer_types) == layer_count
        and all(kind in ("linear_attention", "full_attention") for kind in layer_types),
        "identity.decoder.layer_types contract mismatch",
    )
    for shape_name in (
        "gdn_conv_state_shape",
        "gdn_recurrent_state_shape",
        "kv_cache_shape",
    ):
        shape = decoder[shape_name]
        _require(
            isinstance(shape, list)
            and bool(shape)
            and all(type(dimension) is int and dimension > 0 for dimension in shape),
            f"identity.decoder.{shape_name} must be a concrete positive shape",
        )
    for dtype_name in (
        "gdn_conv_state_dtype",
        "gdn_recurrent_state_dtype",
        "kv_cache_dtype",
    ):
        _require_string(decoder[dtype_name], f"identity.decoder.{dtype_name}")

    parallelism = identity["parallelism"]
    _require(isinstance(parallelism, dict), "identity.parallelism must be an object")
    _require_exact_keys(
        parallelism,
        {"world_size", "rank", "tensor_parallel_size", "pipeline_parallel_size"},
        "identity.parallelism",
    )
    world_size = _require_int(parallelism["world_size"], "parallelism.world_size", minimum=1)
    rank = _require_int(parallelism["rank"], "parallelism.rank", minimum=0)
    tensor_parallel_size = _require_int(
        parallelism["tensor_parallel_size"],
        "parallelism.tensor_parallel_size",
        minimum=1,
    )
    pipeline_parallel_size = _require_int(
        parallelism["pipeline_parallel_size"],
        "parallelism.pipeline_parallel_size",
        minimum=1,
    )
    _require(
        1 <= world_size <= 16
        and 1 <= tensor_parallel_size <= 16
        and pipeline_parallel_size == 1
        and world_size == tensor_parallel_size
        and rank < world_size,
        "layer checkpoint supports pure tensor parallel groups with PP=1 and world=TP in 1..16",
    )

    weights = identity["weights"]
    _require(isinstance(weights, dict), "identity.weights must be an object")
    _require_exact_keys(
        weights,
        {
            "checkpoint_tensor_count",
            "source_tensor_count",
            "loaded_tensor_count",
            "source_names_sha256",
        },
        "identity.weights",
    )
    for key in ("checkpoint_tensor_count", "source_tensor_count", "loaded_tensor_count"):
        _require_int(weights[key], f"identity.weights.{key}", minimum=1)
    _require_sha256(weights["source_names_sha256"], "identity.weights.source_names_sha256")
    return identity


def _validate_request_identity(
    value: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, Any]:
    request = _normalized_json(value)
    _require(isinstance(request, dict), "request identity must be an object")
    _require_exact_keys(request, _REQUEST_IDENTITY_KEYS, "request identity")
    _require_string(request["sequence_id"], "request.sequence_id")
    _require(request["phase"] in ("prefill", "decode"), "request.phase must be prefill or decode")
    _require_int(request["step_index"], "request.step_index")
    before = _require_int(request["context_length_before"], "request.context_length_before")
    after = _require_int(request["context_length_after"], "request.context_length_after", minimum=1)
    _require(
        isinstance(request["scheduler"], dict) and bool(request["scheduler"]),
        "request.scheduler must be a non-empty exact scheduler identity",
    )

    _require(
        isinstance(input_ids, torch.Tensor)
        and input_ids.dtype == torch.int64
        and input_ids.ndim == 1
        and input_ids.numel() > 0,
        "input_ids must be a non-empty int64 vector",
    )
    _require(
        isinstance(positions, torch.Tensor)
        and positions.dtype == torch.int64
        and positions.shape == input_ids.shape,
        "positions must be an int64 vector matching input_ids",
    )
    input_values = input_ids.detach().cpu().tolist()
    position_values = positions.detach().cpu().tolist()
    _require(all(type(token) is int and token >= 0 for token in input_values), "input_ids must be non-negative")
    expected_positions = list(range(before, before + len(input_values)))
    _require(position_values == expected_positions, "positions are not contiguous from context_length_before")
    _require(after == before + len(input_values), "context_length_after does not match token count")
    if request["phase"] == "prefill":
        _require(before == 0, "prefill checkpoint must start from empty context")
    return request


def _validate_lineage(value: Any, after_layer: int) -> dict[str, Any]:
    lineage = _normalized_json(value)
    _require(isinstance(lineage, dict), "lineage must be an object")
    _require_exact_keys(
        lineage,
        {"run_id", "previous_after_layer", "previous_manifest_sha256"},
        "lineage",
    )
    _require_string(lineage["run_id"], "lineage.run_id")
    previous_layer = lineage["previous_after_layer"]
    previous_hash = lineage["previous_manifest_sha256"]
    if after_layer == -1:
        _require(
            previous_layer is None and previous_hash is None,
            "initial boundary lineage must have null previous layer and hash",
        )
    else:
        _require(
            previous_layer is not None and previous_hash is not None,
            "non-initial boundary lineage must identify the immediately preceding layer and manifest hash",
        )
        _require(
            type(previous_layer) is int and previous_layer >= -1,
            "lineage.previous_after_layer must be an integer >= -1",
        )
        _require(previous_layer == after_layer - 1, "lineage must point to the immediately preceding layer")
        _require_sha256(previous_hash, "lineage.previous_manifest_sha256")
    return lineage


def _prepare_tensor(
    tensors: dict[str, torch.Tensor],
    key: str,
    role: str,
    value: torch.Tensor,
    *,
    require_finite: bool,
) -> dict[str, Any]:
    _require(key not in tensors, f"duplicate checkpoint tensor key: {key}")
    _require(isinstance(value, torch.Tensor), f"{role} must be a tensor")
    _require(value.layout == torch.strided and not value.is_quantized, f"{role} must be a dense strided tensor")
    _require(value.is_contiguous(), f"{role} must be contiguous")
    stored = value.detach().cpu().contiguous().clone()
    nonfinite_count = 0
    if stored.is_floating_point() or stored.is_complex():
        nonfinite_count = int(torch.count_nonzero(~torch.isfinite(stored)).item())
    _require(not require_finite or nonfinite_count == 0, f"{role} contains {nonfinite_count} nonfinite values")
    tensors[key] = stored
    return {
        "key": key,
        "role": role,
        "dtype": str(stored.dtype),
        "shape": list(stored.shape),
        "source_device": str(value.device),
        "source_stride": list(value.stride()),
        "source_storage_offset": value.storage_offset(),
        "stored_stride": list(stored.stride()),
        "numel": stored.numel(),
        "bytes": stored.numel() * stored.element_size(),
        "sha256": _sha256_bytes(_tensor_bytes(stored)),
        "nonfinite_count": nonfinite_count,
    }


def _cache_contract(identity: Mapping[str, Any], layer_type: str) -> list[tuple[str, str, list[int]]]:
    decoder = identity["decoder"]
    if layer_type == "linear_attention":
        return [
            (
                "conv_state",
                decoder["gdn_conv_state_dtype"],
                decoder["gdn_conv_state_shape"],
            ),
            (
                "recurrent_state",
                decoder["gdn_recurrent_state_dtype"],
                decoder["gdn_recurrent_state_shape"],
            ),
        ]
    return [("kv_cache", decoder["kv_cache_dtype"], decoder["kv_cache_shape"])]


def _prepare_caches(
    tensors: dict[str, torch.Tensor],
    caches: Sequence[tuple[Any, ...]],
    identity: Mapping[str, Any],
    *,
    require_finite: bool,
) -> dict[str, Any]:
    layer_types = identity["decoder"]["layer_types"]
    _require(len(caches) == len(layer_types), "runtime cache record count mismatch")
    identities: set[str] = set()
    records: list[dict[str, Any]] = []
    for layer_index, (record, layer_type) in enumerate(zip(caches, layer_types)):
        _require(isinstance(record, tuple), f"cache record {layer_index} must be a tuple")
        contract = _cache_contract(identity, layer_type)
        _require(len(record) == len(contract) + 1, f"cache record {layer_index} arity mismatch")
        module_identity = _require_string(record[0], f"cache record {layer_index} identity")
        _require(module_identity not in identities, f"duplicate cache module identity: {module_identity}")
        identities.add(module_identity)
        states = []
        for state_index, ((state_name, dtype, shape), value) in enumerate(
            zip(contract, record[1:])
        ):
            _require(isinstance(value, torch.Tensor), f"cache {layer_index}.{state_name} must be a tensor")
            _require(str(value.dtype) == dtype, f"cache {layer_index}.{state_name} dtype mismatch")
            _require(list(value.shape) == shape, f"cache {layer_index}.{state_name} shape mismatch")
            key = f"cache.layers.{layer_index}.{state_name}"
            states.append(
                {
                    "name": state_name,
                    "ordinal": state_index,
                    "tensor": _prepare_tensor(
                        tensors,
                        key,
                        f"layer {layer_index} {state_name}",
                        value,
                        require_finite=require_finite,
                    ),
                }
            )
        records.append(
            {
                "layer_index": layer_index,
                "layer_type": layer_type,
                "module_identity": module_identity,
                "states": states,
            }
        )
    return {"record_count": len(records), "records": records}


def _capture_rng(
    tensors: dict[str, torch.Tensor],
    named_generators: Mapping[str, torch.Generator],
) -> dict[str, Any]:
    python_version, python_internal, python_gauss = random.getstate()
    _require(
        isinstance(python_internal, tuple)
        and bool(python_internal)
        and all(type(value) is int for value in python_internal),
        "unsupported Python random state",
    )
    _require(
        python_gauss is None
        or (isinstance(python_gauss, float) and math.isfinite(python_gauss)),
        "Python random gaussian cache is nonfinite",
    )
    python_descriptor = _prepare_tensor(
        tensors,
        "rng.python_mt_state",
        "Python random state",
        torch.tensor(python_internal, dtype=torch.int64),
        require_finite=True,
    )

    numpy_algorithm, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached = (
        np.random.get_state()
    )
    _require(numpy_algorithm == "MT19937", "unsupported NumPy global RNG algorithm")
    _require(math.isfinite(float(numpy_cached)), "NumPy gaussian cache is nonfinite")
    numpy_descriptor = _prepare_tensor(
        tensors,
        "rng.numpy_mt_state",
        "NumPy global random state",
        torch.from_numpy(numpy_keys.astype(np.int64, copy=True)),
        require_finite=True,
    )

    torch_descriptor = _prepare_tensor(
        tensors,
        "rng.torch_cpu_state",
        "torch CPU random state",
        torch.get_rng_state(),
        require_finite=True,
    )
    generator_records = []
    for name in sorted(named_generators):
        _require(
            isinstance(name, str) and GENERATOR_NAME_RE.fullmatch(name) is not None,
            f"invalid named generator name: {name!r}",
        )
        generator = named_generators[name]
        _require(isinstance(generator, torch.Generator), f"named generator {name} is not torch.Generator")
        generator_records.append(
            {
                "name": name,
                "device": str(generator.device),
                "state": _prepare_tensor(
                    tensors,
                    f"rng.generators.{name}",
                    f"named torch generator {name}",
                    generator.get_state(),
                    require_finite=True,
                ),
            }
        )
    return {
        "policy": "restore_exact_host_globals_and_named_torch_generators",
        "python": {
            "version": python_version,
            "gauss_next": python_gauss,
            "state": python_descriptor,
        },
        "numpy": {
            "algorithm": numpy_algorithm,
            "position": int(numpy_position),
            "has_gauss": int(numpy_has_gauss),
            "cached_gaussian": float(numpy_cached),
            "state": numpy_descriptor,
        },
        "torch_cpu": {"state": torch_descriptor},
        "named_torch_generators": generator_records,
        "accelerator_global": {
            "captured": False,
            "reason": "formal target inference forbids stochastic operators and CUDA access",
        },
    }


def _exclusive_write(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


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
        raise OSError(errno.ENOSYS, "renameat2 is unavailable; refusing non-atomic checkpoint publish")
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


def _publish_directory(output_dir: Path, state_bytes: bytes, manifest_bytes: bytes) -> Path:
    output = output_dir.expanduser()
    _require(output.name not in ("", ".", ".."), "checkpoint output must name a directory")
    parent = output.parent.resolve(strict=True)
    _require(parent.is_dir(), "checkpoint parent is not a directory")
    destination = parent / output.name
    try:
        os.lstat(destination)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(errno.EEXIST, "checkpoint directory already exists", destination)

    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    try:
        _exclusive_write(temporary_dir / STATE_FILENAME, state_bytes)
        _exclusive_write(temporary_dir / MANIFEST_FILENAME, manifest_bytes)
        _fsync_directory(temporary_dir)
        _rename_noreplace(temporary_dir, destination)
        temporary_dir = None
        _fsync_directory(parent)
    finally:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir)
    return destination


def publish_layer_checkpoint(
    output_dir: Path,
    *,
    identity: Mapping[str, Any],
    request_identity: Mapping[str, Any],
    lineage: Mapping[str, Any],
    after_layer: int,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    caches: Sequence[tuple[Any, ...]],
    named_generators: Mapping[str, torch.Generator] | None = None,
    require_finite: bool = True,
) -> dict[str, Any]:
    """Publish one immutable, atomically visible decoder-layer checkpoint."""

    normalized_identity = _validate_identity(identity)
    normalized_request = _validate_request_identity(request_identity, input_ids, positions)
    layer_count = normalized_identity["decoder"]["layer_count"]
    _require_after_layer(after_layer, layer_count, "after_layer")
    normalized_lineage = _validate_lineage(lineage, after_layer)

    activation_dtype = normalized_identity["decoder"]["activation_dtype"]
    hidden_size = normalized_identity["decoder"]["hidden_size"]
    token_count = input_ids.numel()
    _require(isinstance(hidden_states, torch.Tensor), "hidden_states must be a tensor")
    _require(str(hidden_states.dtype) == activation_dtype, "hidden_states dtype mismatch")
    _require(
        list(hidden_states.shape) == [token_count, hidden_size],
        "hidden_states shape mismatch",
    )
    if after_layer == -1:
        _require(residual is None, "initial boundary residual must be None")
    else:
        _require(isinstance(residual, torch.Tensor), "residual must be a tensor after layer execution")
        _require(str(residual.dtype) == activation_dtype, "residual dtype mismatch")
        _require(
            list(residual.shape) == [token_count, hidden_size],
            "residual shape mismatch",
        )

    tensor_payload: dict[str, torch.Tensor] = {}
    input_descriptor = _prepare_tensor(
        tensor_payload,
        "request.input_ids",
        "request input IDs",
        input_ids,
        require_finite=True,
    )
    position_descriptor = _prepare_tensor(
        tensor_payload,
        "request.positions",
        "request positions",
        positions,
        require_finite=True,
    )
    hidden_descriptor = _prepare_tensor(
        tensor_payload,
        "boundary.hidden_states",
        "returned hidden states",
        hidden_states,
        require_finite=require_finite,
    )
    residual_descriptor = (
        None
        if residual is None
        else _prepare_tensor(
            tensor_payload,
            "boundary.residual",
            "returned residual",
            residual,
            require_finite=require_finite,
        )
    )
    runtime_caches = _prepare_caches(
        tensor_payload,
        caches,
        normalized_identity,
        require_finite=require_finite,
    )
    rng = _capture_rng(tensor_payload, named_generators or {})

    request = {
        "identity": normalized_request,
        "input_ids": input_descriptor,
        "positions": position_descriptor,
        "input_id_values": input_ids.detach().cpu().tolist(),
        "position_values": positions.detach().cpu().tolist(),
    }
    boundary = {
        "after_layer": after_layer,
        "next_layer": after_layer + 1,
        "remaining_decoder_layer_count": layer_count - after_layer - 1,
        "resume_action": (
            "final_norm"
            if after_layer + 1 == layer_count
            else "decoder_layers_then_final_norm"
        ),
        "completed_layer_count": after_layer + 1,
        "layer_count": layer_count,
        "semantics": CHECKPOINT_KIND,
        "hidden_states": hidden_descriptor,
        "residual_present": residual is not None,
        "residual": residual_descriptor,
    }
    identity_sha256 = _sha256_bytes(_canonical_json(normalized_identity))
    request_sha256 = _sha256_bytes(_canonical_json(request))
    core = {
        "schema": CHECKPOINT_SCHEMA,
        "kind": CHECKPOINT_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "write_policy": WRITE_POLICY,
        "identity": normalized_identity,
        "identity_sha256": identity_sha256,
        "request": request,
        "request_sha256": request_sha256,
        "boundary": boundary,
        "runtime_caches": runtime_caches,
        "rng": rng,
        "lineage": normalized_lineage,
    }
    binding_sha256 = _sha256_bytes(_canonical_json(core))
    state_bytes = save_safetensors(
        tensor_payload,
        metadata={
            "schema": CHECKPOINT_SCHEMA,
            "binding_sha256": binding_sha256,
            "identity_sha256": identity_sha256,
            "request_sha256": request_sha256,
        },
    )
    _require(len(state_bytes) <= MAX_STATE_BYTES, "checkpoint state exceeds size limit")
    state_sha256 = _sha256_bytes(state_bytes)
    manifest = {
        **core,
        "binding_sha256": binding_sha256,
        "artifacts": {
            "state": {
                "filename": STATE_FILENAME,
                "bytes": len(state_bytes),
                "sha256": state_sha256,
                "tensor_keys": sorted(tensor_payload),
            },
            "manifest": {"filename": MANIFEST_FILENAME},
        },
    }
    manifest_bytes = _canonical_json(manifest) + b"\n"
    _require(len(manifest_bytes) <= MAX_MANIFEST_BYTES, "checkpoint manifest exceeds size limit")
    destination = _publish_directory(Path(output_dir), state_bytes, manifest_bytes)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    cache_tensor_count = sum(len(record["states"]) for record in runtime_caches["records"])
    return {
        "schema": PUBLISH_RESULT_SCHEMA,
        "path": str(destination),
        "after_layer": after_layer,
        "next_layer": after_layer + 1,
        "remaining_decoder_layer_count": layer_count - after_layer - 1,
        "resume_action": boundary["resume_action"],
        "residual_present": residual is not None,
        "manifest_sha256": manifest_sha256,
        "state_sha256": state_sha256,
        "identity_sha256": identity_sha256,
        "request_sha256": request_sha256,
        "tensor_count": len(tensor_payload),
        "cache_record_count": len(runtime_caches["records"]),
        "cache_tensor_count": cache_tensor_count,
        "state_bytes": len(state_bytes),
        "manifest_bytes": len(manifest_bytes),
        "write_policy": WRITE_POLICY,
        "atomic_publish": True,
        "strict_restore_eligible": require_finite,
    }


def _read_regular_file_nofollow(path: Path, maximum: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{name} is not a regular file")
        _require(0 < metadata.st_size <= maximum, f"{name} size is outside the accepted range")
        chunks = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            _require(bool(chunk), f"short read from {name}")
            chunks.append(chunk)
            remaining -= len(chunk)
        _require(os.read(descriptor, 1) == b"", f"{name} grew while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _safetensors_metadata(value: bytes) -> dict[str, str]:
    _require(len(value) >= 8, "safetensors payload is truncated")
    header_length = struct.unpack("<Q", value[:8])[0]
    _require(0 < header_length <= len(value) - 8, "invalid safetensors header length")
    header = _strict_json_load(value[8 : 8 + header_length], "safetensors header")
    _require(isinstance(header, dict), "safetensors header must be an object")
    metadata = header.get("__metadata__")
    _require(isinstance(metadata, dict), "safetensors metadata is missing")
    _require(
        all(isinstance(key, str) and isinstance(item, str) for key, item in metadata.items()),
        "safetensors metadata must contain string pairs",
    )
    return metadata


def _validate_tensor_descriptor(
    descriptor: Any,
    tensor: torch.Tensor,
    expected_key: str,
) -> None:
    _require(isinstance(descriptor, dict), f"tensor descriptor {expected_key} must be an object")
    _require_exact_keys(descriptor, _TENSOR_DESCRIPTOR_KEYS, f"tensor descriptor {expected_key}")
    _require(descriptor["key"] == expected_key, f"tensor descriptor key mismatch: {expected_key}")
    _require_string(descriptor["role"], f"tensor descriptor {expected_key}.role")
    _require(descriptor["dtype"] == str(tensor.dtype), f"tensor {expected_key} dtype mismatch")
    _require(descriptor["shape"] == list(tensor.shape), f"tensor {expected_key} shape mismatch")
    _require_string(descriptor["source_device"], f"tensor {expected_key}.source_device")
    source_stride = descriptor["source_stride"]
    _require(
        isinstance(source_stride, list)
        and len(source_stride) == tensor.ndim
        and all(type(value) is int and value >= 0 for value in source_stride),
        f"tensor {expected_key} source stride is invalid",
    )
    source_storage_offset = _require_int(
        descriptor["source_storage_offset"],
        f"tensor {expected_key}.source_storage_offset",
    )
    try:
        source_layout = torch.empty_strided(
            tuple(tensor.shape), tuple(source_stride), device="meta"
        )
    except (RuntimeError, ValueError) as error:
        raise LayerCheckpointError(
            f"tensor {expected_key} source layout is invalid: {error}"
        ) from error
    _require(
        source_layout.is_contiguous(),
        f"tensor {expected_key} source layout is not contiguous",
    )
    maximum_offset = source_storage_offset
    for dimension, stride in zip(tensor.shape, source_stride):
        if dimension:
            maximum_offset += (dimension - 1) * stride
    source_storage_bytes = (maximum_offset + 1) * tensor.element_size()
    _require(
        source_storage_bytes <= MAX_STATE_BYTES,
        f"tensor {expected_key} source layout exceeds the restore size limit",
    )
    _require(descriptor["stored_stride"] == list(tensor.stride()), f"tensor {expected_key} stored stride mismatch")
    _require(descriptor["numel"] == tensor.numel(), f"tensor {expected_key} numel mismatch")
    _require(
        descriptor["bytes"] == tensor.numel() * tensor.element_size(),
        f"tensor {expected_key} byte count mismatch",
    )
    _require_sha256(descriptor["sha256"], f"tensor {expected_key}.sha256")
    _require(descriptor["sha256"] == _sha256_bytes(_tensor_bytes(tensor)), f"tensor {expected_key} SHA-256 mismatch")
    observed_nonfinite = 0
    if tensor.is_floating_point() or tensor.is_complex():
        observed_nonfinite = int(torch.count_nonzero(~torch.isfinite(tensor)).item())
    _require(descriptor["nonfinite_count"] == observed_nonfinite, f"tensor {expected_key} finite descriptor mismatch")


def _all_descriptors(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    descriptors = [
        manifest["request"]["input_ids"],
        manifest["request"]["positions"],
        manifest["boundary"]["hidden_states"],
    ]
    if manifest["boundary"]["residual_present"]:
        descriptors.append(manifest["boundary"]["residual"])
    for record in manifest["runtime_caches"]["records"]:
        descriptors.extend(state["tensor"] for state in record["states"])
    rng = manifest["rng"]
    descriptors.extend(
        [
            rng["python"]["state"],
            rng["numpy"]["state"],
            rng["torch_cpu"]["state"],
        ]
    )
    descriptors.extend(record["state"] for record in rng["named_torch_generators"])
    return descriptors


def _validate_manifest_structure(manifest: Any) -> dict[str, Any]:
    _require(isinstance(manifest, dict), "checkpoint manifest must be an object")
    _require_exact_keys(manifest, _MANIFEST_KEYS, "checkpoint manifest")
    _require(manifest["schema"] == CHECKPOINT_SCHEMA, "checkpoint schema mismatch")
    _require(manifest["kind"] == CHECKPOINT_KIND, "checkpoint boundary kind mismatch")
    created_utc = _require_string(manifest["created_utc"], "checkpoint created_utc")
    try:
        created = datetime.fromisoformat(created_utc)
    except ValueError as error:
        raise LayerCheckpointError("checkpoint created_utc is not ISO-8601") from error
    _require(
        created.tzinfo is not None and created.utcoffset().total_seconds() == 0,
        "checkpoint created_utc must carry an explicit UTC offset",
    )
    _require(manifest["write_policy"] == WRITE_POLICY, "checkpoint write policy mismatch")
    identity = _validate_identity(manifest["identity"])
    _require_sha256(manifest["identity_sha256"], "identity_sha256")
    _require(
        manifest["identity_sha256"] == _sha256_bytes(_canonical_json(identity)),
        "checkpoint identity hash mismatch",
    )

    request = manifest["request"]
    _require(isinstance(request, dict), "checkpoint request must be an object")
    _require_exact_keys(
        request,
        {"identity", "input_ids", "positions", "input_id_values", "position_values"},
        "checkpoint request",
    )
    _require_sha256(manifest["request_sha256"], "request_sha256")
    _require(
        manifest["request_sha256"] == _sha256_bytes(_canonical_json(request)),
        "checkpoint request hash mismatch",
    )

    boundary = manifest["boundary"]
    _require(isinstance(boundary, dict), "checkpoint boundary must be an object")
    _require_exact_keys(
        boundary,
        {
            "after_layer",
            "next_layer",
            "remaining_decoder_layer_count",
            "resume_action",
            "completed_layer_count",
            "layer_count",
            "semantics",
            "hidden_states",
            "residual_present",
            "residual",
        },
        "checkpoint boundary",
    )
    layer_count = identity["decoder"]["layer_count"]
    after_layer = _require_after_layer(
        boundary["after_layer"], layer_count, "boundary.after_layer"
    )
    _require(boundary["next_layer"] == after_layer + 1, "checkpoint next_layer mismatch")
    remaining_layers = layer_count - after_layer - 1
    _require(
        boundary["remaining_decoder_layer_count"] == remaining_layers,
        "remaining decoder layer count mismatch",
    )
    expected_resume_action = (
        "final_norm" if remaining_layers == 0 else "decoder_layers_then_final_norm"
    )
    _require(
        boundary["resume_action"] == expected_resume_action,
        "checkpoint resume action mismatch",
    )
    _require(boundary["completed_layer_count"] == after_layer + 1, "completed layer count mismatch")
    _require(boundary["layer_count"] == layer_count, "checkpoint layer count mismatch")
    _require(boundary["semantics"] == CHECKPOINT_KIND, "checkpoint boundary semantics mismatch")
    if after_layer == -1:
        _require(
            boundary["residual_present"] is False and boundary["residual"] is None,
            "initial boundary must encode an absent residual",
        )
    else:
        _require(
            boundary["residual_present"] is True
            and isinstance(boundary["residual"], dict),
            "post-layer boundary must encode a residual tensor",
        )
    _validate_lineage(manifest["lineage"], after_layer)

    caches = manifest["runtime_caches"]
    _require(isinstance(caches, dict), "runtime_caches must be an object")
    _require_exact_keys(caches, {"record_count", "records"}, "runtime_caches")
    records = caches["records"]
    _require(isinstance(records, list) and len(records) == layer_count, "runtime cache list length mismatch")
    _require(caches["record_count"] == len(records), "runtime cache record_count mismatch")
    module_identities: set[str] = set()
    for layer_index, (record, layer_type) in enumerate(
        zip(records, identity["decoder"]["layer_types"])
    ):
        _require(isinstance(record, dict), f"cache manifest record {layer_index} must be an object")
        _require_exact_keys(
            record,
            {"layer_index", "layer_type", "module_identity", "states"},
            f"cache manifest record {layer_index}",
        )
        _require(record["layer_index"] == layer_index, f"cache record {layer_index} index mismatch")
        _require(record["layer_type"] == layer_type, f"cache record {layer_index} type mismatch")
        module_identity = _require_string(record["module_identity"], f"cache record {layer_index} identity")
        _require(module_identity not in module_identities, f"duplicate cache identity: {module_identity}")
        module_identities.add(module_identity)
        contract = _cache_contract(identity, layer_type)
        states = record["states"]
        _require(isinstance(states, list) and len(states) == len(contract), f"cache record {layer_index} state count mismatch")
        for ordinal, (state, expected) in enumerate(zip(states, contract)):
            _require(isinstance(state, dict), f"cache state {layer_index}.{ordinal} must be an object")
            _require_exact_keys(state, {"name", "ordinal", "tensor"}, f"cache state {layer_index}.{ordinal}")
            _require(state["name"] == expected[0], f"cache state {layer_index}.{ordinal} name mismatch")
            _require(state["ordinal"] == ordinal, f"cache state {layer_index}.{ordinal} ordinal mismatch")

    rng = manifest["rng"]
    _require(isinstance(rng, dict), "checkpoint RNG must be an object")
    _require_exact_keys(
        rng,
        {"policy", "python", "numpy", "torch_cpu", "named_torch_generators", "accelerator_global"},
        "checkpoint RNG",
    )
    _require(
        rng["policy"] == "restore_exact_host_globals_and_named_torch_generators",
        "checkpoint RNG policy mismatch",
    )
    _require(
        rng["accelerator_global"]
        == {
            "captured": False,
            "reason": "formal target inference forbids stochastic operators and CUDA access",
        },
        "checkpoint accelerator RNG policy mismatch",
    )
    generator_records = rng["named_torch_generators"]
    _require(isinstance(generator_records, list), "named generator records must be a list")
    generator_names = [record.get("name") for record in generator_records if isinstance(record, dict)]
    _require(
        len(generator_names) == len(generator_records)
        and generator_names == sorted(generator_names)
        and len(set(generator_names)) == len(generator_names),
        "named generator records must be unique and sorted",
    )
    for record in generator_records:
        _require_exact_keys(record, {"name", "device", "state"}, f"named generator {record.get('name')}")
        _require(GENERATOR_NAME_RE.fullmatch(record["name"]) is not None, "invalid named generator name")
        _require_string(record["device"], f"named generator {record['name']} device")

    artifacts = manifest["artifacts"]
    _require(isinstance(artifacts, dict), "checkpoint artifacts must be an object")
    _require_exact_keys(artifacts, {"state", "manifest"}, "checkpoint artifacts")
    _require_exact_keys(
        artifacts["state"],
        {"filename", "bytes", "sha256", "tensor_keys"},
        "checkpoint state artifact",
    )
    _require(artifacts["state"]["filename"] == STATE_FILENAME, "state filename mismatch")
    _require_int(artifacts["state"]["bytes"], "state artifact bytes", minimum=1)
    _require_sha256(artifacts["state"]["sha256"], "state artifact sha256")
    _require_exact_keys(artifacts["manifest"], {"filename"}, "checkpoint manifest artifact")
    _require(artifacts["manifest"]["filename"] == MANIFEST_FILENAME, "manifest filename mismatch")

    _require_sha256(manifest["binding_sha256"], "binding_sha256")
    core = {key: manifest[key] for key in _MANIFEST_CORE_KEYS}
    _require(
        manifest["binding_sha256"] == _sha256_bytes(_canonical_json(core)),
        "checkpoint binding hash mismatch",
    )
    return manifest


def _validate_loaded_tensor_contracts(
    manifest: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]
) -> None:
    identity = manifest["identity"]
    decoder = identity["decoder"]
    request = manifest["request"]
    input_ids = tensors[request["input_ids"]["key"]]
    positions = tensors[request["positions"]["key"]]
    token_count = input_ids.numel()
    _require(
        input_ids.dtype == torch.int64
        and input_ids.ndim == 1
        and token_count > 0,
        "checkpoint input_ids tensor contract mismatch",
    )
    _require(
        positions.dtype == torch.int64 and positions.shape == input_ids.shape,
        "checkpoint positions tensor contract mismatch",
    )

    boundary = manifest["boundary"]
    activation_shape = [token_count, decoder["hidden_size"]]
    for name in ("hidden_states", "residual"):
        descriptor = boundary[name]
        if descriptor is None:
            continue
        tensor = tensors[descriptor["key"]]
        _require(
            str(tensor.dtype) == decoder["activation_dtype"]
            and list(tensor.shape) == activation_shape,
            f"checkpoint boundary {name} tensor contract mismatch",
        )
        _require(
            descriptor["source_device"] == identity["target"]["device"],
            f"checkpoint boundary {name} source device mismatch",
        )

    for record in manifest["runtime_caches"]["records"]:
        contract = _cache_contract(identity, record["layer_type"])
        for state, (_state_name, expected_dtype, expected_shape) in zip(
            record["states"], contract
        ):
            descriptor = state["tensor"]
            tensor = tensors[descriptor["key"]]
            _require(
                str(tensor.dtype) == expected_dtype
                and list(tensor.shape) == expected_shape,
                f"checkpoint cache tensor contract mismatch: {descriptor['key']}",
            )
            _require(
                descriptor["source_device"] == identity["target"]["device"],
                f"checkpoint cache source device mismatch: {descriptor['key']}",
            )

    rng = manifest["rng"]
    python_state = tensors[rng["python"]["state"]["key"]]
    numpy_state = tensors[rng["numpy"]["state"]["key"]]
    torch_state = tensors[rng["torch_cpu"]["state"]["key"]]
    _require(
        python_state.dtype == torch.int64
        and python_state.ndim == 1
        and python_state.numel() > 0,
        "checkpoint Python RNG tensor contract mismatch",
    )
    _require(
        numpy_state.dtype == torch.int64
        and numpy_state.ndim == 1
        and numpy_state.numel() == 624,
        "checkpoint NumPy MT19937 tensor contract mismatch",
    )
    _require(
        torch_state.dtype == torch.uint8
        and torch_state.ndim == 1
        and torch_state.numel() > 0,
        "checkpoint torch CPU RNG tensor contract mismatch",
    )
    for record in rng["named_torch_generators"]:
        state = tensors[record["state"]["key"]]
        _require(
            state.dtype == torch.uint8 and state.ndim == 1 and state.numel() > 0,
            f"checkpoint named generator tensor contract mismatch: {record['name']}",
        )


def load_layer_checkpoint(path: Path) -> LoadedLayerCheckpoint:
    """Read and completely validate a checkpoint without mutating process state."""

    requested = Path(path).expanduser()
    metadata = os.lstat(requested)
    _require(stat.S_ISDIR(metadata.st_mode), "checkpoint path must be a real directory")
    resolved = requested.resolve(strict=True)
    entries = list(os.scandir(resolved))
    _require(
        {entry.name for entry in entries} == {MANIFEST_FILENAME, STATE_FILENAME},
        "checkpoint directory entry set mismatch",
    )
    _require(all(entry.is_file(follow_symlinks=False) for entry in entries), "checkpoint entries must be regular files")
    manifest_bytes = _read_regular_file_nofollow(
        resolved / MANIFEST_FILENAME, MAX_MANIFEST_BYTES, "checkpoint manifest"
    )
    state_bytes = _read_regular_file_nofollow(
        resolved / STATE_FILENAME, MAX_STATE_BYTES, "checkpoint state"
    )
    manifest = _validate_manifest_structure(
        _strict_json_load(manifest_bytes, "checkpoint manifest")
    )
    state_record = manifest["artifacts"]["state"]
    _require(state_record["bytes"] == len(state_bytes), "checkpoint state byte count mismatch")
    _require(state_record["sha256"] == _sha256_bytes(state_bytes), "checkpoint state file SHA-256 mismatch")
    metadata_record = _safetensors_metadata(state_bytes)
    _require(
        metadata_record
        == {
            "schema": CHECKPOINT_SCHEMA,
            "binding_sha256": manifest["binding_sha256"],
            "identity_sha256": manifest["identity_sha256"],
            "request_sha256": manifest["request_sha256"],
        },
        "safetensors metadata binding mismatch",
    )
    try:
        tensors = {
            key: value.clone().contiguous()
            for key, value in load_safetensors(state_bytes).items()
        }
    except Exception as error:
        raise LayerCheckpointError(f"invalid safetensors state: {error}") from error
    expected_keys = state_record["tensor_keys"]
    _require(
        isinstance(expected_keys, list)
        and expected_keys == sorted(expected_keys)
        and len(set(expected_keys)) == len(expected_keys),
        "state tensor key manifest must be sorted and unique",
    )
    _require(set(tensors) == set(expected_keys), "safetensors tensor key set mismatch")
    descriptors = _all_descriptors(manifest)
    _require(len(descriptors) == len(expected_keys), "checkpoint tensor descriptor count mismatch")
    descriptor_keys = [descriptor.get("key") for descriptor in descriptors if isinstance(descriptor, dict)]
    _require(
        len(descriptor_keys) == len(descriptors)
        and len(set(descriptor_keys)) == len(descriptor_keys)
        and set(descriptor_keys) == set(expected_keys),
        "checkpoint tensor descriptors are not one-to-one with state tensors",
    )
    for descriptor in descriptors:
        key = descriptor["key"]
        _validate_tensor_descriptor(descriptor, tensors[key], key)
    _require(
        all(descriptor["nonfinite_count"] == 0 for descriptor in descriptors),
        "checkpoint contains nonfinite tensors and is not restore eligible",
    )
    _validate_loaded_tensor_contracts(manifest, tensors)

    request = manifest["request"]
    _validate_request_identity(
        request["identity"],
        tensors[request["input_ids"]["key"]],
        tensors[request["positions"]["key"]],
    )
    _require(
        request["input_id_values"] == tensors[request["input_ids"]["key"]].tolist(),
        "checkpoint input ID values mismatch",
    )
    _require(
        request["position_values"] == tensors[request["positions"]["key"]].tolist(),
        "checkpoint position values mismatch",
    )
    return LoadedLayerCheckpoint(
        path=resolved,
        manifest=manifest,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        tensors=tensors,
    )


def _validate_restore_caches(
    loaded: LoadedLayerCheckpoint,
    caches: Sequence[tuple[Any, ...]],
) -> list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]:
    records = loaded.manifest["runtime_caches"]["records"]
    _require(len(caches) == len(records), "restore cache record count mismatch")
    copies = []
    for layer_index, (target_record, source_record) in enumerate(zip(caches, records)):
        _require(isinstance(target_record, tuple), f"restore cache record {layer_index} must be a tuple")
        _require(
            len(target_record) == len(source_record["states"]) + 1,
            f"restore cache record {layer_index} arity mismatch",
        )
        _require(
            target_record[0] == source_record["module_identity"],
            f"restore cache record {layer_index} module identity mismatch",
        )
        for target, state in zip(target_record[1:], source_record["states"]):
            descriptor = state["tensor"]
            source = loaded.tensors[descriptor["key"]]
            _require(isinstance(target, torch.Tensor), f"restore target {descriptor['key']} must be a tensor")
            _require(str(target.dtype) == descriptor["dtype"], f"restore target {descriptor['key']} dtype mismatch")
            _require(list(target.shape) == descriptor["shape"], f"restore target {descriptor['key']} shape mismatch")
            _require(str(target.device) == descriptor["source_device"], f"restore target {descriptor['key']} device mismatch")
            _require(list(target.stride()) == descriptor["source_stride"], f"restore target {descriptor['key']} stride mismatch")
            _require(target.storage_offset() == descriptor["source_storage_offset"], f"restore target {descriptor['key']} storage offset mismatch")
            copies.append((target, source, descriptor))
    return copies


def _validate_rng_restore(
    loaded: LoadedLayerCheckpoint,
    named_generators: Mapping[str, torch.Generator],
) -> None:
    rng = loaded.manifest["rng"]
    records = rng["named_torch_generators"]
    _require(set(named_generators) == {record["name"] for record in records}, "named generator set mismatch")
    for record in records:
        generator = named_generators[record["name"]]
        _require(isinstance(generator, torch.Generator), f"restore generator {record['name']} is not torch.Generator")
        _require(str(generator.device) == record["device"], f"restore generator {record['name']} device mismatch")
    python = rng["python"]
    _require_exact_keys(python, {"version", "gauss_next", "state"}, "Python RNG record")
    _require_int(python["version"], "Python RNG version", minimum=1)
    _require(
        python["gauss_next"] is None
        or (isinstance(python["gauss_next"], float) and math.isfinite(python["gauss_next"])),
        "Python RNG gaussian cache is invalid",
    )
    numpy_rng = rng["numpy"]
    _require_exact_keys(
        numpy_rng,
        {"algorithm", "position", "has_gauss", "cached_gaussian", "state"},
        "NumPy RNG record",
    )
    _require(numpy_rng["algorithm"] == "MT19937", "NumPy RNG algorithm mismatch")
    _require_int(numpy_rng["position"], "NumPy RNG position")
    _require(numpy_rng["has_gauss"] in (0, 1), "NumPy RNG has_gauss is invalid")
    _require(
        isinstance(numpy_rng["cached_gaussian"], (int, float))
        and math.isfinite(float(numpy_rng["cached_gaussian"])),
        "NumPy RNG gaussian cache is invalid",
    )
    _require_exact_keys(rng["torch_cpu"], {"state"}, "torch CPU RNG record")

    try:
        python_values = loaded.tensors[python["state"]["key"]].tolist()
        python_probe = random.Random()
        python_probe.setstate(
            (
                python["version"],
                tuple(int(value) for value in python_values),
                python["gauss_next"],
            )
        )
        numpy_values = loaded.tensors[numpy_rng["state"]["key"]].numpy().astype(
            np.uint32, copy=True
        )
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(
            (
                numpy_rng["algorithm"],
                numpy_values,
                numpy_rng["position"],
                numpy_rng["has_gauss"],
                numpy_rng["cached_gaussian"],
            )
        )
        torch_probe = torch.Generator(device="cpu")
        torch_probe.set_state(loaded.tensors[rng["torch_cpu"]["state"]["key"]])
        for record in records:
            generator_probe = torch.Generator(device=record["device"])
            generator_probe.set_state(
                loaded.tensors[record["state"]["key"]].to(
                    device=record["device"]
                )
            )
    except Exception as error:
        raise LayerCheckpointError(
            f"checkpoint RNG state cannot be restored: {error}"
        ) from error


def _materialize_cache_sources(
    copies: Sequence[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    materialized = []
    for target, source, descriptor in copies:
        source_device = source.to(device=target.device)
        source_stride = tuple(descriptor["source_stride"])
        source_offset = descriptor["source_storage_offset"]
        maximum_offset = source_offset
        for dimension, stride in zip(source.shape, source_stride):
            if dimension:
                maximum_offset += (dimension - 1) * stride
        storage = torch.empty(
            maximum_offset + 1,
            dtype=source.dtype,
            device=target.device,
        )
        restored_layout = torch.as_strided(
            storage,
            tuple(source.shape),
            source_stride,
            source_offset,
        )
        restored_layout.copy_(source_device)
        materialized.append((target, restored_layout))
    return materialized


def _apply_rng(
    loaded: LoadedLayerCheckpoint,
    named_generators: Mapping[str, torch.Generator],
) -> None:
    rng = loaded.manifest["rng"]
    python = rng["python"]
    python_values = loaded.tensors[python["state"]["key"]].tolist()
    random.setstate(
        (python["version"], tuple(int(value) for value in python_values), python["gauss_next"])
    )
    numpy_rng = rng["numpy"]
    numpy_values = loaded.tensors[numpy_rng["state"]["key"]].numpy().astype(
        np.uint32, copy=True
    )
    np.random.set_state(
        (
            numpy_rng["algorithm"],
            numpy_values,
            numpy_rng["position"],
            numpy_rng["has_gauss"],
            numpy_rng["cached_gaussian"],
        )
    )
    torch.set_rng_state(loaded.tensors[rng["torch_cpu"]["state"]["key"]])
    for record in rng["named_torch_generators"]:
        named_generators[record["name"]].set_state(
            loaded.tensors[record["state"]["key"]]
        )


def restore_layer_checkpoint(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_request_identity: Mapping[str, Any],
    expected_input_ids: torch.Tensor,
    expected_positions: torch.Tensor,
    expected_after_layer: int,
    caches: Sequence[tuple[Any, ...]],
    named_generators: Mapping[str, torch.Generator] | None = None,
    expected_manifest_sha256: str | None = None,
) -> RestoredLayerCheckpoint:
    """Validate everything, then transactionally restore caches and all RNG state."""

    loaded = load_layer_checkpoint(path)
    identity = _validate_identity(expected_identity)
    request_identity = _validate_request_identity(
        expected_request_identity, expected_input_ids, expected_positions
    )
    _require(loaded.manifest["identity"] == identity, "restore model/runtime identity mismatch")
    _require(loaded.manifest["request"]["identity"] == request_identity, "restore request identity mismatch")
    _require_after_layer(
        expected_after_layer,
        identity["decoder"]["layer_count"],
        "expected_after_layer",
    )
    _require(loaded.after_layer == expected_after_layer, "restore layer boundary mismatch")
    if expected_manifest_sha256 is not None:
        _require_sha256(expected_manifest_sha256, "expected_manifest_sha256")
        _require(loaded.manifest_sha256 == expected_manifest_sha256, "restore manifest SHA-256 mismatch")
    request = loaded.manifest["request"]
    checkpoint_input_ids = loaded.tensors[request["input_ids"]["key"]]
    checkpoint_positions = loaded.tensors[request["positions"]["key"]]
    _require(torch.equal(checkpoint_input_ids, expected_input_ids.detach().cpu()), "restore input IDs mismatch")
    _require(torch.equal(checkpoint_positions, expected_positions.detach().cpu()), "restore positions mismatch")

    cache_copies = _validate_restore_caches(loaded, caches)
    generators = named_generators or {}
    _validate_rng_restore(loaded, generators)
    materialized_cache_copies = _materialize_cache_sources(cache_copies)

    cache_backups = [
        (target, target.detach().clone())
        for target, _source in materialized_cache_copies
    ]
    rng_backup = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "generators": {name: generator.get_state() for name, generator in generators.items()},
    }
    try:
        with torch.no_grad():
            for target, source in materialized_cache_copies:
                target.copy_(source)
        _apply_rng(loaded, generators)
    except Exception:
        with torch.no_grad():
            for target, backup in cache_backups:
                target.copy_(backup)
        random.setstate(rng_backup["python"])
        np.random.set_state(rng_backup["numpy"])
        torch.set_rng_state(rng_backup["torch_cpu"])
        for name, state in rng_backup["generators"].items():
            generators[name].set_state(state)
        raise

    hidden_descriptor = loaded.manifest["boundary"]["hidden_states"]
    residual_descriptor = loaded.manifest["boundary"]["residual"]
    hidden = loaded.tensors[hidden_descriptor["key"]].clone()
    residual = (
        None
        if residual_descriptor is None
        else loaded.tensors[residual_descriptor["key"]].clone()
    )
    cache_tensor_count = len(materialized_cache_copies)
    result = {
        "schema": RESTORE_RESULT_SCHEMA,
        "path": str(loaded.path),
        "after_layer": loaded.after_layer,
        "next_layer": loaded.next_layer,
        "remaining_decoder_layer_count": loaded.manifest["boundary"][
            "remaining_decoder_layer_count"
        ],
        "resume_action": loaded.manifest["boundary"]["resume_action"],
        "residual_present": loaded.manifest["boundary"]["residual_present"],
        "manifest_sha256": loaded.manifest_sha256,
        "state_sha256": loaded.manifest["artifacts"]["state"]["sha256"],
        "identity_sha256": loaded.manifest["identity_sha256"],
        "request_sha256": loaded.manifest["request_sha256"],
        "restored_cache_record_count": len(caches),
        "restored_cache_tensor_count": cache_tensor_count,
        "rng_restored": True,
        "strict_validation_passed": True,
    }
    return RestoredLayerCheckpoint(
        hidden_states=hidden,
        residual=residual,
        input_ids=checkpoint_input_ids.clone(),
        positions=checkpoint_positions.clone(),
        next_layer=loaded.next_layer,
        manifest=loaded.manifest,
        result=result,
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "PUBLISH_RESULT_SCHEMA",
    "RESTORE_RESULT_SCHEMA",
    "WRITE_POLICY",
    "LayerCheckpointError",
    "LoadedLayerCheckpoint",
    "RestoredLayerCheckpoint",
    "load_layer_checkpoint",
    "publish_layer_checkpoint",
    "restore_layer_checkpoint",
]
