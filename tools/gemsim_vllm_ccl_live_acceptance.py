#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent acceptance authority for the standalone vLLM CCL live gate.

The runner and rank workers never grant acceptance.  This verifier only reads
an already completed evidence tree, independently replays the CCL plan and
BF16 arithmetic, rehashes every artifact, and publishes into an absent output
directory.  It never starts gem5 or imports the executed vLLM environment.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
BASE_VERIFIER_FILE = ROOT / "tools/gemsim_ccl_live_allreduce_acceptance.py"
RUNNER_FILE = ROOT / "scripts/run_gemsim_vllm_ccl_live.py"
WORKER_FILE = ROOT / "examples/triton/vllm_ccl_live_rank.py"
BOOTSTRAP_FILE = ROOT / "examples/triton/_gemsim_bootstrap.py"
DESIGN_FILE = ROOT / "tools/gemsim_ccl_live_allreduce.py"
VLLM_CHECKOUT = ROOT / "projects/vllm"


def _load_base():
    name = "_gemsim_vllm_ccl_acceptance_base"
    specification = importlib.util.spec_from_file_location(name, BASE_VERIFIER_FILE)
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load the bound standalone CCL verifier")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


BASE = _load_base()
AcceptanceError = BASE.AcceptanceError
EXPECTED_SCHEMA = BASE.EXPECTED_SCHEMA
DESIGN_SCHEMA = BASE.DESIGN_SCHEMA
RANK_LAUNCH_SCHEMA = BASE.RANK_LAUNCH_SCHEMA
TRACE_SCHEMA = BASE.TRACE_SCHEMA
RUN_SCHEMA = "amdgpu-sim.vllm-ccl-live-run.v1"
RANK_RESULT_SCHEMA = "amdgpu-sim.vllm-ccl-live-rank-result.v1"
ADAPTER_SCHEMA = "amdgpu-sim.vllm-ccl-live-adapter-evidence.v1"
EVENT_SCHEMA = "amdgpu-sim.vllm-ccl-live-adapter-event.v1"
BOOTSTRAP_SCHEMA = "amdgpu-sim.vllm-ccl-bootstrap.v1"
ACCEPTANCE_SCHEMA = "amdgpu-sim.vllm-ccl-live-acceptance.v1"
MANIFEST_SCHEMA = "amdgpu-sim.vllm-ccl-live-acceptance-manifest.v1"
WORKLOAD_SCHEMA = "amdgpu-sim.vllm-ccl-workload.v1"
PROCESS_GROUP_AUDIT_SCHEMA = "amdgpu-sim.vllm-gloo-process-group-audit.v1"
DISPATCH_CAPTURE_SCHEMA = "amdgpu-sim.torch-dispatch-output-capture.v1"
ROW_PARALLEL_LOCAL_OPERATOR = "gemsim.dense_linear.default"
PINNED_VLLM_HEAD = "8d9b52f7c2514490bdadfd5eb0c931e58625df2e"
PINNED_VLLM_TREE = "d7f16cac8369098d7fde19003ab2577171116ecb"
PINNED_VLLM_VERSION = "0.0.dev0+g8d9b52f7c2"
FORMAL_WORLDS = (2, 3, 4, 8, 16)
ALL_WORLDS = tuple(range(2, 17))
MAX_ARTIFACT_BYTES = BASE.MAX_ARTIFACT_BYTES
MAX_JSON_BYTES = BASE.MAX_JSON_BYTES
MAX_JOURNAL_BYTES = BASE.MAX_JOURNAL_BYTES
HEX32 = re.compile(r"[0-9a-f]{32}")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")

RANK_FILES = (
    "worker-result.json",
    "adapter-evidence.json",
    "adapter-events.jsonl",
    "bootstrap-descriptor.json",
    "dispatch-trace.jsonl",
    "stats.txt",
    "gem5.log",
    "rank-launch.json",
    "input.bin",
    "output.bin",
    "worker-stdout.log",
    "worker-stderr.log",
)

TENSOR_COLLECTIVE_APIS = (
    "all_reduce",
    "all_reduce_coalesced",
    "broadcast",
    "broadcast_object_list",
    "barrier",
    "reduce",
    "all_gather",
    "all_gather_into_tensor",
    "_all_gather_base",
    "all_gather_coalesced",
    "all_gather_object",
    "send",
    "recv",
    "isend",
    "irecv",
    "batch_isend_irecv",
    "gather",
    "gather_object",
    "scatter",
    "scatter_object_list",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "_reduce_scatter_base",
    "all_to_all",
    "all_to_all_single",
)

ACTUAL_IMPORT_ROLES = (
    "vllm_parallel_state",
    "vllm_base_communicator",
    "vllm_communication_op",
    "vllm_version",
    "vllm_plugin_init",
    "vllm_communicator",
    "vllm_platform",
    "ccl_engine",
    "triton_driver",
)
ROW_PARALLEL_IMPORT_ROLES = ACTUAL_IMPORT_ROLES + (
    "vllm_linear",
    "vllm_config_vllm",
    "vllm_config_parallel",
    "vllm_config_model",
    "vllm_adapters",
    "vllm_row_parallel",
    "vllm_ops",
    "vllm_kernels",
)

# Keep this list explicit.  Any launcher identity expansion is a schema change
# until the verifier is updated to understand the new role.
IDENTITY_ROLES = (
    "product_manifest",
    "source_lock",
    "runtime_library",
    "ccl_native",
    "ccl_device",
    "ccl_engine",
    "triton_driver",
    "gem5_binary",
    "gem5_config",
    "vllm_plugin_init",
    "vllm_communicator",
    "vllm_ccl_bootstrap",
    "vllm_platform",
    "vllm_parallel_state",
    "vllm_base_communicator",
    "vllm_communication_op",
    "vllm_version",
    "vllm_metadata",
    "vllm_checkout_parallel_state",
    "vllm_checkout_base_communicator",
    "vllm_checkout_communication_op",
    "vllm_checkout_version",
    "vllm_linear",
    "vllm_config_vllm",
    "vllm_config_parallel",
    "vllm_config_model",
    "vllm_adapters",
    "vllm_row_parallel",
    "vllm_ops",
    "vllm_kernels",
    "ccl_acceptance_base",
    "verifier",
    "runner",
    "worker",
    "bootstrap",
    "design",
    "rank_registry",
)


def streaming_file_record(path: Path, *, limit: int) -> dict[str, Any]:
    """Hash a large immutable input without retaining it in verifier memory."""
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AcceptanceError(f"cannot open regular artifact {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"artifact is not regular: {path}")
        require(before.st_uid == os.getuid(), f"artifact has wrong owner: {path}")
        require(0 < before.st_size <= limit, f"artifact size is invalid: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            require(size <= limit, f"artifact exceeds size limit: {path}")
            digest.update(chunk)
        after = os.fstat(descriptor)
        stable = lambda value: (
            value.st_dev, value.st_ino, value.st_mode, value.st_uid,
            value.st_gid, value.st_size, value.st_mtime_ns, value.st_ctime_ns,
        )
        require(stable(before) == stable(after) and size == before.st_size,
                f"artifact changed while hashed: {path}")
        return {"bytes": size, "sha256": digest.hexdigest()}
    finally:
        os.close(descriptor)

LIVE_EXECUTION_IDENTITY_ROLES = tuple(
    role for role in IDENTITY_ROLES if role != "verifier"
)

RESULT_KEYS = (
    "schema",
    "status",
    "rank",
    "world_size",
    "acceptance_authority",
    "live_adapter_accepted",
    "public_result_published",
    "input_sha256_before",
    "input_sha256_after",
    "output_sha256",
    "output_storage_fresh",
    "bootstrap_descriptor_sha256",
    "adapter_evidence_sha256",
    "managed_session",
    "first_error",
    "product",
)

ADAPTER_KEYS = (
    "schema",
    "rank",
    "world_size",
    "entrypoint",
    "coordinator_class",
    "communicator_class",
    "platform_class",
    "unique_name",
    "control_backend",
    "control_process_groups",
    "tensor_data_backend",
    "message_queue_broadcaster",
    "use_custom_op_call",
    "coordinator_methods_unmodified",
    "gloo_tensor_api_counts",
    "gloo_tensor_api_total",
    "gloo_control_records",
    "capability_fd_identity",
    "bootstrap_descriptor_sha256",
    "input_sha256_before",
    "input_sha256_after",
    "output_sha256",
    "output_storage_fresh",
    "engine_rank",
    "engine_world_size",
    "engine_state_after_collective",
    "actual_imports",
    "vllm_installed_version",
    "managed_session",
    "coordinator_destroyed",
    "default_group_destroyed",
    "workload_evidence",
)

RUN_KEYS = (
    "schema",
    "status",
    "acceptance_authority",
    "live_adapter_accepted",
    "expected",
    "workload",
    "world_size",
    "element_count",
    "dtype",
    "unique_name",
    "job_uuid",
    "group_uuid",
    "epoch",
    "group_generation",
    "started_at_ns",
    "completed_at_ns",
    "absolute_deadline_ns",
    "target_execution_completed",
    "target_feedback",
    "oracle_phase",
    "oracle_feedback",
    "first_error",
    "supervisor_cleanup",
    "source_identity_preflight",
    "source_identity_postflight",
    "vllm_checkout",
    "ranks",
)


def build_row_parallel_workload(model_root: Path) -> dict[str, Any]:
    """Build the first external workload anchor without changing production code."""
    root = model_root.resolve(strict=True)
    config_path = root / "config.json"
    shard_path = root / "model.safetensors-00001-of-00001.safetensors"
    config_payload, config_record = BASE.file_record(config_path)
    shard_record = streaming_file_record(
        shard_path, limit=2 * 1024 * 1024 * 1024
    )
    del config_payload
    from safetensors import safe_open

    key = "model.language_model.layers.0.mlp.down_proj.weight"
    with safe_open(shard_path, framework="pt", device="cpu") as tensors:
        require(key in tensors.keys(), "RowParallel checkpoint tensor is absent")
        weight = tensors.get_tensor(key)
    require(str(weight.dtype) == "torch.bfloat16"
            and list(weight.shape) == [1024, 3584]
            and weight.is_contiguous(),
            "RowParallel checkpoint tensor contract differs")
    weight_payload = weight.view(__import__("torch").uint8).numpy().tobytes()
    input_payloads = [row_parallel_input(rank) for rank in range(2)]
    document = {
        "schema": WORKLOAD_SCHEMA,
        "kind": "vllm-row-parallel",
        "model": {
            "id": "Qwen/Qwen3.5-0.8B",
            "root": str(root),
            "config": {"path": str(config_path), **config_record},
            "weight_shard": {"path": str(shard_path), **shard_record},
            "tensor_key": key,
            "tensor_dtype": "bfloat16",
            "tensor_shape": [1024, 3584],
            "tensor_sha256": hashlib.sha256(weight_payload).hexdigest(),
        },
        "layer": {
            "upstream_class": "vllm.model_executor.layers.linear.RowParallelLinear",
            "oot_class": "gemsim_vllm.adapters.GemsimRowParallelLinear",
            "input_size": 3584,
            "output_size": 1024,
            "bias": False,
            "input_is_parallel": True,
            "reduce_results": True,
            "return_bias": True,
            "tp_world_size": 2,
        },
        "input": {
            "policy": "affine-mod127-v1",
            "dtype": "bfloat16",
            "shape": [1, 1792],
            "sha256_by_rank": [
                hashlib.sha256(payload).hexdigest() for payload in input_payloads
            ],
        },
        "collective": {"dtype": "bfloat16", "element_count": 1024},
    }
    return validate_workload(document, live=True)


def validate_workload(value: Any, *, live: bool) -> dict[str, Any]:
    workload = exact_keys(
        value, ("schema", "kind", "model", "layer", "input", "collective"),
        "workload",
    )
    require(workload["schema"] == WORKLOAD_SCHEMA
            and workload["kind"] in {"standalone-allreduce", "vllm-row-parallel"},
            "workload schema or kind differs")
    if workload["kind"] == "standalone-allreduce":
        require(workload["model"] is None and workload["layer"] is None
                and workload["input"] == {"policy": "rank-affine-mod127-v1"},
                "standalone workload contract differs")
        exact_keys(workload["collective"], ("dtype", "element_count"),
                   "standalone collective")
        return dict(workload)

    model = exact_keys(
        workload["model"],
        ("id", "root", "config", "weight_shard", "tensor_key", "tensor_dtype",
         "tensor_shape", "tensor_sha256"),
        "RowParallel model",
    )
    layer = exact_keys(
        workload["layer"],
        ("upstream_class", "oot_class", "input_size", "output_size", "bias",
         "input_is_parallel", "reduce_results", "return_bias", "tp_world_size"),
        "RowParallel layer",
    )
    input_spec = exact_keys(workload["input"],
                            ("policy", "dtype", "shape", "sha256_by_rank"),
                            "RowParallel input")
    collective = exact_keys(workload["collective"], ("dtype", "element_count"),
                            "RowParallel collective")
    require(model["id"] == "Qwen/Qwen3.5-0.8B"
            and model["tensor_key"]
            == "model.language_model.layers.0.mlp.down_proj.weight"
            and model["tensor_dtype"] == "bfloat16"
            and model["tensor_shape"] == [1024, 3584]
            and HEX64.fullmatch(str(model["tensor_sha256"])) is not None,
            "RowParallel model anchor differs")
    require(layer == {
        "upstream_class": "vllm.model_executor.layers.linear.RowParallelLinear",
        "oot_class": "gemsim_vllm.adapters.GemsimRowParallelLinear",
        "input_size": 3584, "output_size": 1024, "bias": False,
        "input_is_parallel": True, "reduce_results": True,
        "return_bias": True, "tp_world_size": 2,
    }, "RowParallel upstream layer contract differs")
    require(input_spec["policy"] == "affine-mod127-v1"
            and input_spec["dtype"] == "bfloat16"
            and input_spec["shape"] == [1, 1792]
            and isinstance(input_spec["sha256_by_rank"], list)
            and len(input_spec["sha256_by_rank"]) == 2
            and all(HEX64.fullmatch(str(value)) is not None
                    for value in input_spec["sha256_by_rank"])
            and collective == {"dtype": "bfloat16", "element_count": 1024},
            "RowParallel input/collective contract differs")
    for name in ("config", "weight_shard"):
        record = exact_keys(model[name], ("path", "bytes", "sha256"),
                            f"RowParallel {name}")
        require(Path(record["path"]).is_absolute()
                and type(record["bytes"]) is int and record["bytes"] > 0
                and HEX64.fullmatch(str(record["sha256"])) is not None,
                f"RowParallel {name} identity differs")
        if live:
            if name == "weight_shard":
                observed = streaming_file_record(
                    Path(record["path"]), limit=2 * 1024 * 1024 * 1024
                )
            else:
                _, observed = BASE.file_record(
                    Path(record["path"]), limit=MAX_JSON_BYTES
                )
            require(observed == {"bytes": record["bytes"], "sha256": record["sha256"]},
                    f"RowParallel {name} drifted")
    return dict(workload)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def exact_keys(value: Any, expected: Iterable[str], label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    require(set(value) == set(expected), f"{label} fields differ")
    return value


def integer(value: Any, minimum: int, maximum: int, label: str) -> int:
    require(type(value) is int and minimum <= value <= maximum,
            f"{label} must be in [{minimum}, {maximum}]")
    return int(value)


def hex_string(value: Any, pattern: re.Pattern[str], label: str) -> str:
    require(isinstance(value, str) and pattern.fullmatch(value) is not None,
            f"{label} is not canonical lowercase hex")
    return str(value)


def canonical_json(value: object) -> bytes:
    return BASE.canonical_json(value)


def object_sha256(value: object) -> str:
    return BASE.object_sha256(value)


def _regular_private(path: Path, label: str) -> os.stat_result:
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode) and not path.is_symlink(),
            f"{label} is not a regular file")
    require(metadata.st_uid == os.getuid(), f"{label} owner differs")
    require(not metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO),
            f"{label} is not private")
    return metadata


def validate_absent_output(path: Path) -> None:
    BASE.validate_absent_output(path)


def validate_source_inventory(source: Path, world: int) -> None:
    require(source.is_absolute() and source == Path(os.path.normpath(source)),
            "source path is not normalized and absolute")
    require(source.is_dir() and not source.is_symlink(), "source is not a directory")
    expected_root = {"result-manifest.json"} | {
        f"rank-{rank:02d}" for rank in range(world)
    }
    require({entry.name for entry in os.scandir(source)} == expected_root,
            "source inventory differs")
    _regular_private(source / "result-manifest.json", "run manifest")
    for rank in range(world):
        directory = source / f"rank-{rank:02d}"
        require(directory.is_dir() and not directory.is_symlink(),
                f"rank {rank} source is not a directory")
        require({entry.name for entry in os.scandir(directory)} == set(RANK_FILES),
                f"rank {rank} source inventory differs")
        for name in RANK_FILES:
            _regular_private(directory / name, f"rank {rank} {name}")


def _identity_path(record: Mapping[str, Any], role: str) -> Path:
    exact_keys(record, ("path", "bytes", "sha256"), f"identity {role}")
    path = Path(str(record["path"]))
    require(path.is_absolute() and path == Path(os.path.normpath(path)),
            f"identity {role} path is not canonical")
    integer(record["bytes"], 1, MAX_ARTIFACT_BYTES, f"identity {role} bytes")
    hex_string(record["sha256"], HEX64, f"identity {role} SHA")
    return path


def validate_identity_snapshot(value: Any, *, live: bool) -> dict[str, dict[str, Any]]:
    require(isinstance(value, Mapping) and set(value) == set(IDENTITY_ROLES),
            "source identity roles differ")
    result: dict[str, dict[str, Any]] = {}
    for role in IDENTITY_ROLES:
        record = dict(value[role])
        path = _identity_path(record, role)
        if live and role in LIVE_EXECUTION_IDENTITY_ROLES:
            _, observed = BASE.file_record(path)
            require(observed == {"bytes": record["bytes"], "sha256": record["sha256"]},
                    f"live source identity drifted: {role}")
        result[role] = record
    expected_paths = {
        "ccl_acceptance_base": BASE_VERIFIER_FILE,
        "verifier": THIS_FILE,
        "runner": RUNNER_FILE,
        "worker": WORKER_FILE,
        "bootstrap": BOOTSTRAP_FILE,
        "design": DESIGN_FILE,
        "rank_registry": ROOT / "scripts/gemsim_live_registry.py",
    }
    for role, expected in expected_paths.items():
        require(Path(result[role]["path"]) == expected,
                f"source identity {role} path mismatch")
    if live:
        validate_static_call_chain(result)
    return result


def _module(tree: ast.AST, name: str, kind: type[ast.AST]) -> ast.AST:
    matches = [node for node in getattr(tree, "body", [])
               if isinstance(node, kind) and getattr(node, "name", None) == name]
    require(len(matches) == 1, f"static source needs exactly one {name}")
    return matches[0]


def _call_name(call: ast.Call) -> str:
    parts: list[str] = []
    node: ast.AST = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_subscript_call(call: ast.Call, name: str) -> bool:
    return (
        isinstance(call.func, ast.Subscript)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == name
    )


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    matches = [node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == name
                       for target in node.targets)]
    require(len(matches) == 1, f"static source needs exactly one {name} assignment")
    try:
        return ast.literal_eval(matches[0].value)
    except (ValueError, TypeError) as error:
        raise AcceptanceError(f"static {name} is not literal") from error


def _parse_identity_source(identity: Mapping[str, Mapping[str, Any]], role: str) -> ast.Module:
    try:
        return ast.parse(Path(identity[role]["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise AcceptanceError(f"static source is invalid: {role}") from error


def _parse_frozen_ccl_source(
    identity: Mapping[str, Mapping[str, Any]], filename: str
) -> ast.Module:
    engine_path = Path(identity["ccl_engine"]["path"])
    path = engine_path.with_name(filename)
    require(path.parent == engine_path.parent,
            "shared CCL source escaped the frozen package")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise AcceptanceError("shared CCL source is unavailable") from error
    require(stat.S_ISREG(metadata.st_mode) and not path.is_symlink(),
            "shared CCL source is not a regular file")

    manifest_path = Path(identity["product_manifest"]["path"])
    manifest, _ = BASE.read_json(manifest_path, "product manifest")
    prefix = manifest_path.parent
    try:
        relative = path.relative_to(prefix).as_posix()
    except ValueError as error:
        raise AcceptanceError("shared CCL source escaped the product") from error
    inventory = manifest.get("inventory")
    require(isinstance(inventory, list), "product inventory is missing")
    matches = [
        item for item in inventory
        if isinstance(item, Mapping) and item.get("path") == relative
    ]
    _, observed = BASE.file_record(path)
    require(
        len(matches) == 1
        and matches[0].get("kind") == "regular"
        and matches[0].get("bytes") == observed["bytes"]
        and matches[0].get("sha256") == observed["sha256"],
        "shared CCL source differs from the frozen product inventory",
    )
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise AcceptanceError("shared CCL source is invalid") from error


def validate_static_call_chain(identity: Mapping[str, Mapping[str, Any]]) -> None:
    worker = _parse_identity_source(identity, "worker")
    require(tuple(_literal_assignment(worker, "TENSOR_COLLECTIVE_APIS"))
            == TENSOR_COLLECTIVE_APIS,
            "worker Gloo tensor API audit surface differs")
    run_rank = _module(worker, "run_rank", ast.FunctionDef)
    worker_calls = [_call_name(node) for node in ast.walk(run_rank)
                    if isinstance(node, ast.Call)]
    require("dist.init_process_group" in worker_calls,
            "worker does not initialize the explicit Gloo control group")
    require("load_general_plugins" in worker_calls,
            "worker does not activate the OOT vLLM plugin")
    require("GroupCoordinator" in worker_calls,
            "worker does not construct a real vLLM GroupCoordinator")
    require("coordinator.all_reduce" in worker_calls,
            "worker does not enter GroupCoordinator.all_reduce")
    require(not any(name.endswith("GemsimDeviceCommunicator") for name in worker_calls),
            "worker directly constructs GemsimDeviceCommunicator")
    group_calls = [node for node in ast.walk(run_rank)
                   if isinstance(node, ast.Call) and _call_name(node) == "GroupCoordinator"]
    require(len(group_calls) == 1, "worker GroupCoordinator construction is ambiguous")
    keywords = {item.arg: item.value for item in group_calls[0].keywords if item.arg}
    required_literals = {
        "torch_distributed_backend": "gloo",
        "use_device_communicator": True,
        "use_message_queue_broadcaster": False,
        "group_name": "tp",
        "use_all2all": False,
    }
    for name, expected in required_literals.items():
        require(name in keywords and ast.literal_eval(keywords[name]) == expected,
                f"worker GroupCoordinator {name} differs")
    with_calls = [item.context_expr for node in ast.walk(run_rank)
                  if isinstance(node, ast.With) for item in node.items]
    require(any(isinstance(value, ast.Call)
                and _call_name(value) == "reject_tensor_collectives"
                for value in with_calls),
            "worker does not wrap the adapter path in the Gloo fail-fast audit")

    row_rank = _module(worker, "run_row_parallel_rank", ast.FunctionDef)
    row_calls = [_call_name(node) for node in ast.walk(row_rank)
                 if isinstance(node, ast.Call)]
    for required in (
        "load_general_plugins", "ModelConfig", "VllmConfig",
        "init_distributed_environment", "initialize_model_parallel", "get_tp_group",
        "RowParallelLinear", "layer.weight.weight_loader", "layer",
        "destroy_model_parallel", "destroy_distributed_environment",
    ):
        require(required in row_calls,
                f"RowParallel worker call chain omits {required}")
    model_calls = [node for node in ast.walk(row_rank)
                   if isinstance(node, ast.Call) and _call_name(node) == "ModelConfig"]
    require(len(model_calls) == 1, "RowParallel ModelConfig construction is ambiguous")
    model_keywords = {item.arg: item.value for item in model_calls[0].keywords if item.arg}
    require("model" in model_keywords and "tokenizer" in model_keywords
            and ast.literal_eval(model_keywords["skip_tokenizer_init"]) is True
            and ast.literal_eval(model_keywords["enforce_eager"]) is True,
            "RowParallel worker does not bind a dense local ModelConfig")
    require("AutoWeightsLoader" not in row_calls
            and not any(name.endswith("GemsimDeviceCommunicator") for name in row_calls),
            "RowParallel worker bypasses an upstream loader or communicator hook")
    row_with_calls = [item.context_expr for node in ast.walk(row_rank)
                      if isinstance(node, ast.With) for item in node.items]
    require(any(isinstance(value, ast.Call)
                and _call_name(value) == "audit_standard_vllm_control"
                for value in row_with_calls),
            "RowParallel worker lacks the bounded upstream Gloo control audit")
    require(any(isinstance(value, ast.Call)
                and _call_name(value) == "capture_operator_output"
                for value in row_with_calls),
            "RowParallel worker does not observe the actual local projection")

    linear = _parse_identity_source(identity, "vllm_linear")
    row_class = _module(linear, "RowParallelLinear", ast.ClassDef)
    row_methods = {node.name: node for node in row_class.body
                   if isinstance(node, ast.FunctionDef)}
    require(all(name in row_methods for name in ("weight_loader", "forward")),
            "installed upstream RowParallel loader/forward differs")
    forward_calls = [_call_name(node) for node in ast.walk(row_methods["forward"])
                     if isinstance(node, ast.Call)]
    require("self.quant_method.apply" in forward_calls
            and "tensor_model_parallel_all_reduce" in forward_calls,
            "upstream RowParallel forward no longer uses method and TP hooks")

    adapters = _parse_identity_source(identity, "vllm_adapters")
    oot_row = _module(adapters, "GemsimRowParallelLinear", ast.ClassDef)
    require(not any(isinstance(node, ast.FunctionDef) and node.name == "forward"
                    for node in oot_row.body),
            "OOT RowParallel duplicates the upstream forward state machine")
    method = _module(adapters, "GemsimUnquantizedRowParallelMethod", ast.ClassDef)
    apply_method = next(
        (node for node in method.body
         if isinstance(node, ast.FunctionDef) and node.name == "apply"), None
    )
    require(apply_method is not None and any(
        isinstance(node, ast.Call)
        and _call_name(node) == "torch.ops.gemsim.dense_linear"
        for node in ast.walk(apply_method)
    ), "OOT RowParallel method does not delegate local GEMM to the generic op")

    communicator = _parse_identity_source(identity, "vllm_communicator")
    cls = _module(communicator, "GemsimDeviceCommunicator", ast.ClassDef)
    require(any(isinstance(base, ast.Name) and base.id == "DeviceCommunicatorBase"
                for base in cls.bases),
            "Gemsim communicator base class differs")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    require("all_reduce" in methods and any(
        isinstance(node, ast.Call) and _call_name(node) == "self._engine.all_reduce"
        for node in ast.walk(methods["all_reduce"])
    ), "Gemsim communicator does not delegate all_reduce to the engine")
    build = _module(communicator, "_build_engine", ast.FunctionDef)
    shared_imports = [
        node for node in communicator.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "gemsim_ccl.bootstrap"
        and len(node.names) == 1
        and node.names[0].name == "build_engine"
        and node.names[0].asname == "_shared_build_engine"
    ]
    build_calls = [node for node in ast.walk(build) if isinstance(node, ast.Call)]
    require(
        len(shared_imports) == 1
        and len(build_calls) == 1
        and _call_name(build_calls[0]) == "_shared_build_engine",
        "Gemsim communicator does not delegate engine construction to shared CCL",
    )
    ccl_bootstrap = _parse_frozen_ccl_source(identity, "bootstrap.py")
    shared_build = _module(ccl_bootstrap, "build_engine", ast.FunctionDef)
    require(any(
        isinstance(node, ast.ImportFrom)
        and node.module == "engine"
        and node.level == 1
        and any(alias.name == "AllReduceEngine" for alias in node.names)
        for node in ast.walk(shared_build)
    ), "shared CCL bootstrap does not import the reusable engine")
    join_calls = [
        node for node in ast.walk(shared_build)
        if isinstance(node, ast.Call) and _call_name(node) == "AllReduceEngine.join"
    ]
    require(len(join_calls) == 1,
            "shared CCL bootstrap does not uniquely join the reusable engine")

    parallel = _parse_identity_source(identity, "vllm_parallel_state")
    coordinator = _module(parallel, "GroupCoordinator", ast.ClassDef)
    coordinator_methods = {node.name: node for node in coordinator.body
                           if isinstance(node, ast.FunctionDef)}
    require(all(name in coordinator_methods for name in
                ("all_reduce", "_all_reduce_out_place")),
            "installed GroupCoordinator all_reduce methods differ")
    require(any(isinstance(node, ast.Call)
                and _call_name(node) == "self._all_reduce_out_place"
                for node in ast.walk(coordinator_methods["all_reduce"])),
            "GroupCoordinator all_reduce does not use its out-of-place path")
    require(any(isinstance(node, ast.Call)
                and _call_name(node) == "self.device_communicator.all_reduce"
                for node in ast.walk(coordinator_methods["_all_reduce_out_place"])),
            "GroupCoordinator does not delegate to the selected communicator")

    platform_text = Path(identity["vllm_platform"]["path"]).read_text(encoding="ascii")
    require("gemsim_vllm.communicator.GemsimDeviceCommunicator" in platform_text,
            "Gemsim platform communicator selection differs")

    runner = _parse_identity_source(identity, "runner")
    spawn = _module(runner, "spawn_rank_process", ast.FunctionDef)
    popen_calls = [node for node in ast.walk(spawn)
                   if isinstance(node, ast.Call) and _call_name(node) == "popen_factory"]
    require(len(popen_calls) == 1, "runner worker spawn is ambiguous")
    spawn_keywords = {item.arg: item.value for item in popen_calls[0].keywords if item.arg}
    require(ast.literal_eval(spawn_keywords.get("close_fds")) is True,
            "runner does not close ambient worker FDs")
    pass_fds = spawn_keywords.get("pass_fds")
    require(isinstance(pass_fds, ast.Tuple) and len(pass_fds.elts) == 1
            and isinstance(pass_fds.elts[0], ast.Attribute)
            and pass_fds.elts[0].attr == "capability_fd",
            "runner pass_fds is not exactly the rank capability")

    device = _parse_identity_source(identity, "ccl_device")
    executor = _module(device, "DeviceSumExecutor", ast.ClassDef)
    executor_methods = {
        node.name: node for node in executor.body if isinstance(node, ast.FunctionDef)
    }
    require("sum_in_place" in executor_methods and any(
        isinstance(node, ast.Call) and _is_subscript_call(node, "_sum_kernel")
        for node in ast.walk(executor_methods["sum_in_place"])
    ), "CCL device executor does not launch the SUM kernel")
    require("counters" in executor_methods and any(
        isinstance(node, ast.keyword) and node.arg == "host_reduction_count"
        and isinstance(node.value, ast.Constant) and node.value.value == 0
        for node in ast.walk(executor_methods["counters"])
    ), "CCL device executor host-reduction counter differs")

    engine = _parse_identity_source(identity, "ccl_engine")
    engine_class = _module(engine, "AllReduceEngine", ast.ClassDef)
    engine_methods = {
        node.name: node
        for node in engine_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    require(set(("join", "all_reduce", "close", "destroy", "abort"))
            <= set(engine_methods), "reusable CCL engine lifecycle differs")
    require(any(
        isinstance(node, ast.Call) and _call_name(node) == "runtime.join_rank"
        for node in ast.walk(engine_methods["join"])
    ), "reusable CCL engine does not join through the live runtime")
    require(any(
        isinstance(node, ast.Call) and _call_name(node) == "self._executor.sum_in_place"
        for node in ast.walk(engine_methods["_execute_segment"])
    ), "reusable CCL engine does not execute device SUM")


def _git_output(arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(VLLM_CHECKOUT), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    require(result.returncode == 0, "could not verify the pinned vLLM checkout")
    return result.stdout


def validate_vllm_checkout(identity: Mapping[str, Mapping[str, Any]]) -> None:
    require(_git_output(("rev-parse", "HEAD")).strip() == PINNED_VLLM_HEAD,
            "vLLM checkout HEAD differs from the pin")
    require(_git_output(("rev-parse", "HEAD^{tree}")).strip() == PINNED_VLLM_TREE,
            "vLLM checkout tree differs from the pin")
    require(_git_output(("status", "--porcelain=v1", "--untracked-files=no")) == "",
            "vLLM checkout tracked files are dirty")
    pairs = {
        "vllm_parallel_state": "vllm_checkout_parallel_state",
        "vllm_base_communicator": "vllm_checkout_base_communicator",
        "vllm_communication_op": "vllm_checkout_communication_op",
        "vllm_version": "vllm_checkout_version",
    }
    for installed, checkout in pairs.items():
        require(identity[installed]["bytes"] == identity[checkout]["bytes"]
                and identity[installed]["sha256"] == identity[checkout]["sha256"],
                f"installed {installed} differs from the pinned checkout")


def validate_product_identity(identity: Mapping[str, Mapping[str, Any]]) -> None:
    BASE.validate_product_identity(identity)
    manifest, manifest_record = BASE.read_json(
        Path(identity["product_manifest"]["path"]), "product manifest"
    )
    require(manifest_record == {
        "bytes": identity["product_manifest"]["bytes"],
        "sha256": identity["product_manifest"]["sha256"],
    }, "product manifest record differs")
    source_lock = manifest.get("source_lock")
    require(isinstance(source_lock, Mapping)
            and source_lock.get("path") == identity["source_lock"]["path"]
            and source_lock.get("bytes") == identity["source_lock"]["bytes"]
            and source_lock.get("sha256") == identity["source_lock"]["sha256"],
            "product source lock binding differs")
    lock, lock_record = BASE.read_json(Path(identity["source_lock"]["path"]),
                                       "product source lock")
    require(lock_record == {"bytes": identity["source_lock"]["bytes"],
                            "sha256": identity["source_lock"]["sha256"]},
            "product source lock record differs")
    plugins = manifest.get("plugins")
    snapshots = plugins.get("snapshots") if isinstance(plugins, Mapping) else None
    snapshot = snapshots.get("gemsim-vllm") if isinstance(snapshots, Mapping) else None
    require(isinstance(snapshot, Mapping), "product vLLM plugin snapshot is missing")
    package = Path(str(snapshot.get("package_path")))
    require(package == Path(identity["vllm_plugin_init"]["path"]).parent,
            "product vLLM package path differs")
    inventory = manifest.get("inventory")
    require(isinstance(inventory, list), "product inventory is missing")
    by_path = {item.get("path"): item for item in inventory if isinstance(item, Mapping)}
    prefix = Path(manifest["prefix"])
    for role in ("vllm_plugin_init", "vllm_communicator",
                 "vllm_ccl_bootstrap", "vllm_platform"):
        path = Path(identity[role]["path"])
        relative = path.relative_to(prefix).as_posix()
        require(by_path.get(relative) == {
            "path": relative,
            "kind": "regular",
            "mode": by_path.get(relative, {}).get("mode"),
            "bytes": identity[role]["bytes"],
            "sha256": identity[role]["sha256"],
        }, f"product inventory {role} differs")
    lock_plugins = lock.get("plugins")
    locked = lock_plugins.get("gemsim-vllm") if isinstance(lock_plugins, Mapping) else None
    require(isinstance(locked, Mapping)
            and locked.get("source_set_sha256") == snapshot.get("source_set_sha256"),
            "vLLM plugin source-set binding differs")
    metadata = Path(identity["vllm_metadata"]["path"]).read_text(encoding="utf-8")
    require(f"Version: {PINNED_VLLM_VERSION}\n" in metadata,
            "installed vLLM version differs from the pin")
    validate_vllm_checkout(identity)


def artifact_descriptor(value: Any, expected_path: str,
                        observed: Mapping[str, Any], label: str) -> None:
    BASE.artifact_descriptor(value, expected_path, observed, label)


def validate_manifest_header(manifest: Mapping[str, Any],
                             expected_record: Mapping[str, Any],
                             design: Mapping[str, Any],
                             workload: Mapping[str, Any]) -> None:
    exact_keys(manifest, RUN_KEYS, "run manifest")
    require(manifest["schema"] == RUN_SCHEMA
            and manifest["status"] == "success"
            and manifest["acceptance_authority"] is False
            and manifest["live_adapter_accepted"] is False,
            "runner status/authority boundary differs")
    exact_keys(manifest["expected"], ("schema", "bytes", "sha256"),
               "expected binding")
    require(manifest["expected"] == {"schema": EXPECTED_SCHEMA, **expected_record},
            "run expected binding differs")
    workload_payload = canonical_json(workload)
    require(manifest["workload"] == {
        "schema": WORKLOAD_SCHEMA,
        "bytes": len(workload_payload),
        "sha256": hashlib.sha256(workload_payload).hexdigest(),
        "document": workload,
    }, "run workload binding differs")
    config = design["config"]
    if workload["kind"] == "vllm-row-parallel":
        require(
            config["model_identity_sha256"]
            == hashlib.sha256(workload_payload).hexdigest()
            and config["world_size"] == workload["layer"]["tp_world_size"]
            and config["element_count"] == workload["collective"]["element_count"],
            "RowParallel workload is not bound to the collective group identity",
        )
    require(manifest["world_size"] == config["world_size"]
            and manifest["element_count"] == config["element_count"]
            and manifest["dtype"] == "bfloat16" == config["dtype"]
            and manifest["unique_name"] == "tp:0"
            and manifest["job_uuid"] == config["job_uuid"]
            and manifest["group_uuid"] == config["group_uuid"]
            and manifest["epoch"] == config["epoch"]
            and manifest["group_generation"] == config["group_generation"],
            "run collective identity differs")
    started = integer(manifest["started_at_ns"], 1, (1 << 63) - 1, "started_at_ns")
    completed = integer(manifest["completed_at_ns"], started, (1 << 63) - 1,
                        "completed_at_ns")
    integer(manifest["absolute_deadline_ns"], started + 1, (1 << 63) - 1,
            "absolute_deadline_ns")
    require(completed < manifest["absolute_deadline_ns"]
            and manifest["target_execution_completed"] is True
            and manifest["target_feedback"] is False
            and manifest["oracle_phase"] == "post_target"
            and manifest["oracle_feedback"] is False
            and manifest["first_error"] is None,
            "run execution/oracle boundary differs")
    checkout = exact_keys(
        manifest["vllm_checkout"],
        ("path", "head", "tree", "tracked_clean", "installed_version"),
        "vLLM checkout binding",
    )
    require(checkout == {
        "path": str(VLLM_CHECKOUT.resolve(strict=True)),
        "head": PINNED_VLLM_HEAD,
        "tree": PINNED_VLLM_TREE,
        "tracked_clean": True,
        "installed_version": PINNED_VLLM_VERSION,
    }, "vLLM checkout binding differs")


def _fd_identity(value: Any, label: str) -> dict[str, Any]:
    record = exact_keys(value, BASE.FD_IDENTITY_KEYS, label)
    result = {
        "fd": integer(record["fd"], 0, (1 << 31) - 1, f"{label} fd"),
        "device": integer(record["device"], 0, (1 << 63) - 1, f"{label} device"),
        "inode": integer(record["inode"], 1, (1 << 63) - 1, f"{label} inode"),
        "mode": integer(record["mode"], 0, (1 << 32) - 1, f"{label} mode"),
        "target": record["target"],
    }
    require(stat.S_ISSOCK(result["mode"]), f"{label} is not a socket")
    require(isinstance(result["target"], str)
            and result["target"] == f"socket:[{result['inode']}]",
            f"{label} procfs socket identity differs")
    return result


def validate_bootstrap(value: Mapping[str, Any], *, rank: int,
                       design: Mapping[str, Any],
                       identity: Mapping[str, Mapping[str, Any]],
                       capability: Mapping[str, Any]) -> dict[str, Any]:
    exact_keys(value, ("schema", "product", "groups"),
               f"rank {rank} bootstrap descriptor")
    require(value["schema"] == BOOTSTRAP_SCHEMA,
            f"rank {rank} bootstrap schema differs")
    product = exact_keys(value["product"], ("prefix", "manifest", "runtime_library"),
                         f"rank {rank} bootstrap product")
    manifest_path = Path(identity["product_manifest"]["path"])
    require(product["prefix"] == str(manifest_path.parent)
            and product["manifest"] == identity["product_manifest"]
            and product["runtime_library"] == identity["runtime_library"],
            f"rank {rank} bootstrap product differs")
    require(isinstance(value["groups"], list) and len(value["groups"]) == 1,
            f"rank {rank} bootstrap group count differs")
    group = exact_keys(value["groups"][0], ("unique_name", "identity", "rank"),
                       f"rank {rank} bootstrap group")
    config = design["config"]
    group_identity = exact_keys(
        group["identity"],
        ("world_size", "epoch", "group_generation", "job_uuid", "group_uuid",
         "model_identity_sha256"),
        f"rank {rank} bootstrap group identity",
    )
    require(group["unique_name"] == "tp:0" and group_identity == {
        "world_size": config["world_size"],
        "epoch": config["epoch"],
        "group_generation": config["group_generation"],
        "job_uuid": config["job_uuid"],
        "group_uuid": config["group_uuid"],
        "model_identity_sha256": config["model_identity_sha256"],
    }, f"rank {rank} bootstrap collective identity differs")
    rank_binding = exact_keys(
        group["rank"],
        ("rank", "capability_fd", "broker_pid", "broker_start_time_ticks",
         "join_timeout_ns", "collective_timeout_ns", "credits_per_peer"),
        f"rank {rank} bootstrap rank binding",
    )
    parent = _fd_identity(capability["parent_fd_identity"],
                          f"rank {rank} parent capability")
    require(rank_binding["rank"] == rank
            and rank_binding["capability_fd"] == parent["fd"]
            and capability["pass_fds"] == [parent["fd"]],
            f"rank {rank} pass_fds/capability binding differs")
    integer(rank_binding["broker_pid"], 1, (1 << 31) - 1, "broker pid")
    integer(rank_binding["broker_start_time_ticks"], 1, (1 << 63) - 1,
            "broker start time")
    join = integer(rank_binding["join_timeout_ns"], 1, (1 << 63) - 1,
                   "join timeout")
    collective = integer(rank_binding["collective_timeout_ns"], 1,
                         (1 << 63) - 1, "collective timeout")
    require(join == collective
            and rank_binding["credits_per_peer"] == design["limits"]["credits_per_peer"],
            f"rank {rank} bootstrap timeout/credit policy differs")
    return parent


def validate_events(records: Sequence[Mapping[str, Any]], rank: int) -> None:
    expected = (
        ("worker_started", ()),
        ("default_gloo_group_initialized", ()),
        ("coordinator_ready", ("unique_name",)),
        ("coordinator_all_reduce_returned", ()),
        ("cleanup_complete", ()),
    )
    require(len(records) == len(expected), f"rank {rank} adapter event count differs")
    for ordinal, (record, (event, extra)) in enumerate(zip(records, expected)):
        exact_keys(record, ("schema", "ordinal", "rank", "event", *extra),
                   f"rank {rank} adapter event {ordinal}")
        require(record["schema"] == EVENT_SCHEMA
                and record["ordinal"] == ordinal
                and record["rank"] == rank
                and record["event"] == event,
                f"rank {rank} adapter event order differs")
        if extra:
            require(record["unique_name"] == "tp:0",
                    f"rank {rank} coordinator event name differs")


def validate_row_parallel_events(
    records: Sequence[Mapping[str, Any]], rank: int
) -> None:
    expected = (
        "worker_started",
        "standard_model_parallel_initialized",
        "row_parallel_layer_ready",
        "upstream_row_parallel_forward_returned",
        "cleanup_complete",
    )
    require(len(records) == len(expected), f"rank {rank} RowParallel event count differs")
    for ordinal, (record, event) in enumerate(zip(records, expected)):
        exact_keys(record, ("schema", "ordinal", "rank", "event"),
                   f"rank {rank} RowParallel event {ordinal}")
        require(record == {
            "schema": EVENT_SCHEMA, "ordinal": ordinal, "rank": rank, "event": event,
        }, f"rank {rank} RowParallel event order differs")


def validate_product_binding(value: Any, identity: Mapping[str, Mapping[str, Any]],
                             rank: int) -> None:
    product = exact_keys(
        value,
        ("product_id", "manifest_sha256", "prefix", "ccl_engine",
         "vllm_plugin_init", "vllm_communicator"),
        f"rank {rank} product execution binding",
    )
    manifest, record = BASE.read_json(Path(identity["product_manifest"]["path"]),
                                      f"rank {rank} product manifest")
    require(product["product_id"] == manifest.get("product_id")
            and product["manifest_sha256"] == record["sha256"]
            and product["prefix"]
            == str(Path(identity["product_manifest"]["path"]).parent)
            and product["ccl_engine"] == identity["ccl_engine"]
            and product["vllm_plugin_init"] == identity["vllm_plugin_init"]
            and product["vllm_communicator"] == identity["vllm_communicator"],
            f"rank {rank} product execution binding differs")


def validate_adapter(value: Any, *, rank: int, world: int,
                     parent_capability: Mapping[str, Any],
                     bootstrap_sha: str, input_record: Mapping[str, Any],
                     output_record: Mapping[str, Any],
                     managed: Mapping[str, Any],
                     identity: Mapping[str, Mapping[str, Any]],
                     workload: Mapping[str, Any]) -> dict[str, Any]:
    adapter = exact_keys(value, ADAPTER_KEYS, f"rank {rank} adapter evidence")
    row_parallel = workload["kind"] == "vllm-row-parallel"
    process_groups = adapter["control_process_groups"]
    if row_parallel:
        process_groups = validate_process_group_audit(process_groups, rank, world)
    require(adapter["schema"] == ADAPTER_SCHEMA
            and adapter["rank"] == rank
            and adapter["world_size"] == world
            and adapter["entrypoint"] == (
                "vllm.model_executor.layers.linear.RowParallelLinear.forward"
                if row_parallel else
                "vllm.distributed.parallel_state.GroupCoordinator.all_reduce"
            )
            and adapter["coordinator_class"]
            == "vllm.distributed.parallel_state.GroupCoordinator"
            and adapter["communicator_class"]
            == "gemsim_vllm.communicator.GemsimDeviceCommunicator"
            and adapter["platform_class"] == "gemsim_vllm.platform.GemsimPlatform"
            and adapter["unique_name"] == "tp:0"
            and adapter["control_backend"] == "gloo"
            and (row_parallel or adapter["control_process_groups"] == {
                "default": "gloo", "device_group": "gloo", "cpu_group": "gloo"
            })
            and adapter["tensor_data_backend"] == "gemsim_ccl_engine"
            and adapter["message_queue_broadcaster"] is row_parallel
            and adapter["use_custom_op_call"] is False,
            f"rank {rank} vLLM adapter path differs")
    require(adapter["coordinator_methods_unmodified"] == {
        "broadcast": "upstream-object-identity-preserved",
        "broadcast_tensor_dict": "upstream-object-identity-preserved",
    }, f"rank {rank} upstream coordinator method identity differs")
    counts = exact_keys(adapter["gloo_tensor_api_counts"], TENSOR_COLLECTIVE_APIS,
                        f"rank {rank} Gloo tensor API counters")
    if row_parallel:
        records = adapter["gloo_control_records"]
        require(isinstance(records, list) and records,
                f"rank {rank} standard initialization control evidence is absent")
        require(all(set(record) == {"api", "phase", "dtype", "shape", "bytes"}
                    and record.get("phase") == "initialization"
                    and record.get("api") in {
                        "all_reduce", "barrier", "broadcast_object_list", "broadcast"
                    }
                    and type(record.get("bytes")) is int
                    and 0 <= record["bytes"] <= 64 * 1024
                    for record in records),
                f"rank {rank} standard initialization Gloo allowlist differs")
        require(sum(counts.values()) == len(records)
                and adapter["gloo_tensor_api_total"] == len(records),
                f"rank {rank} standard initialization control counts differ")
        evidence = exact_keys(
            adapter["workload_evidence"],
            ("kind", "layer_class", "forward_inherited", "loader", "loaded_parameters",
             "weight_shard_columns", "weight_shard_sha256_before",
             "weight_shard_sha256_after", "local_projection"),
            f"rank {rank} RowParallel workload evidence",
        )
        expected_columns = [0, 1792] if rank == 0 else [1792, 3584]
        require(evidence["kind"] == "vllm-row-parallel"
                and evidence["layer_class"]
                == "gemsim_vllm.adapters.GemsimRowParallelLinear"
                and evidence["forward_inherited"] is True
                and evidence["loader"]
                == "vllm RowParallelLinear parameter weight_loader hook"
                and evidence["loaded_parameters"] == ["weight"]
                and evidence["weight_shard_columns"] == expected_columns
                and evidence["weight_shard_sha256_before"]
                == evidence["weight_shard_sha256_after"]
                == row_parallel_weight_shard_sha256(workload, rank),
                f"rank {rank} upstream RowParallel loader/forward evidence differs")
        validate_local_projection(
            evidence["local_projection"], rank, workload["layer"]["output_size"]
        )
    else:
        require(all(counts[name] == 0 for name in TENSOR_COLLECTIVE_APIS)
                and adapter["gloo_tensor_api_total"] == 0
                and adapter["gloo_control_records"] == []
                and adapter["workload_evidence"] == {"kind": "standalone-allreduce"},
                f"rank {rank} Gloo carried or attempted tensor payload")
    child_capability = _fd_identity(adapter["capability_fd_identity"],
                                    f"rank {rank} child capability")
    require(child_capability == parent_capability,
            f"rank {rank} inherited capability identity differs")
    require(adapter["bootstrap_descriptor_sha256"] == bootstrap_sha
            and adapter["input_sha256_before"] == input_record["sha256"]
            and adapter["input_sha256_after"] == input_record["sha256"]
            and adapter["output_sha256"] == output_record["sha256"]
            and adapter["output_storage_fresh"] is True,
            f"rank {rank} adapter tensor/artifact binding differs")
    require(adapter["engine_rank"] == rank
            and adapter["engine_world_size"] == world
            and adapter["engine_state_after_collective"] == "ready"
            and adapter["managed_session"] == managed
            and adapter["coordinator_destroyed"] is True
            and adapter["default_group_destroyed"] is True,
            f"rank {rank} adapter engine/cleanup contract differs")
    import_roles = ROW_PARALLEL_IMPORT_ROLES if row_parallel else ACTUAL_IMPORT_ROLES
    actual_imports = exact_keys(
        adapter["actual_imports"], import_roles,
        f"rank {rank} actual imported source identities",
    )
    require(dict(actual_imports) == {
        role: identity[role] for role in import_roles
    }, f"rank {rank} actual imports differ from execution preflight")
    require(adapter["vllm_installed_version"] == PINNED_VLLM_VERSION,
            f"rank {rank} imported vLLM version differs from the pin")
    return dict(adapter)


def validate_process_group_audit(
    value: Any, rank: int, world: int
) -> dict[str, Any]:
    audit = exact_keys(
        value,
        (
            "schema", "init", "new", "destroy", "local_tokens_created",
            "local_tokens_destroyed", "default_destroyed",
            "all_local_groups_destroyed",
        ),
        f"rank {rank} Gloo process-group audit",
    )
    require(audit["schema"] == PROCESS_GROUP_AUDIT_SCHEMA,
            f"rank {rank} Gloo process-group audit schema differs")
    require(audit["init"] == [{
        "ordinal": 0,
        "phase": "initialization",
        "backend": "gloo",
        "rank": rank,
        "world_size": world,
    }], f"rank {rank} default Gloo process-group initialization differs")
    groups = audit["new"]
    require(isinstance(groups, list) and groups,
            f"rank {rank} standard vLLM process groups are absent")
    local_tokens: list[int] = []
    for ordinal, group in enumerate(groups):
        exact_keys(
            group,
            (
                "ordinal", "phase", "backend", "ranks", "local_member",
                "local_token",
            ),
            f"rank {rank} Gloo new_group {ordinal}",
        )
        ranks = group["ranks"]
        require(
            group["ordinal"] == ordinal
            and group["phase"] == "initialization"
            and group["backend"] == "gloo"
            and isinstance(ranks, list) and ranks
            and all(type(value) is int and 0 <= value < world for value in ranks)
            and len(set(ranks)) == len(ranks)
            and type(group["local_member"]) is bool,
            f"rank {rank} Gloo new_group {ordinal} differs",
        )
        token = group["local_token"]
        if group["local_member"]:
            require(type(token) is int and token > 0 and rank in ranks,
                    f"rank {rank} local Gloo group token differs")
            local_tokens.append(token)
        else:
            require(token is None and rank not in ranks,
                    f"rank {rank} nonmember Gloo group differs")
    require(local_tokens == list(range(1, len(local_tokens) + 1))
            and audit["local_tokens_created"] == local_tokens
            and isinstance(audit["local_tokens_destroyed"], list)
            and sorted(audit["local_tokens_destroyed"]) == local_tokens
            and len(set(audit["local_tokens_destroyed"])) == len(local_tokens),
            f"rank {rank} local Gloo group token lifecycle differs")
    destroys = audit["destroy"]
    require(isinstance(destroys, list) and len(destroys) == len(local_tokens) + 1,
            f"rank {rank} Gloo process-group destruction count differs")
    targets = []
    for ordinal, record in enumerate(destroys):
        exact_keys(record, ("ordinal", "phase", "target"),
                   f"rank {rank} Gloo destroy {ordinal}")
        require(record["ordinal"] == ordinal and record["phase"] == "cleanup",
                f"rank {rank} Gloo destroy phase/order differs")
        targets.append(record["target"])
    require(targets.count("default") == 1
            and sorted(value for value in targets if type(value) is int)
            == local_tokens
            and audit["default_destroyed"] is True
            and audit["all_local_groups_destroyed"] is True,
            f"rank {rank} Gloo process groups were not all destroyed")
    return dict(audit)


def validate_local_projection(
    value: Any, rank: int, output_size: int
) -> bytes:
    record = exact_keys(
        value,
        ("schema", "operator", "dtype", "shape", "bytes", "sha256", "payload_hex"),
        f"rank {rank} local projection capture",
    )
    require(
        record["schema"] == DISPATCH_CAPTURE_SCHEMA
        and record["operator"] == ROW_PARALLEL_LOCAL_OPERATOR
        and record["dtype"] == "bfloat16"
        and record["shape"] == [1, output_size]
        and record["bytes"] == output_size * 2
        and isinstance(record["payload_hex"], str)
        and len(record["payload_hex"]) == output_size * 4,
        f"rank {rank} local projection capture contract differs",
    )
    try:
        payload = bytes.fromhex(record["payload_hex"])
    except ValueError as error:
        raise AcceptanceError(
            f"rank {rank} local projection payload is not canonical hex"
        ) from error
    require(payload.hex() == record["payload_hex"]
            and hashlib.sha256(payload).hexdigest() == record["sha256"],
            f"rank {rank} local projection payload/hash differs")
    return payload


def _device_steps(plans: Sequence[Sequence[Mapping[str, Any]]], rank: int) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for segment in plans[rank]:
        for step in segment["steps"]:
            count = int(step["receive_count_elements"])
            if int(step["phase"]) == BASE.PHASE_REDUCE_SCATTER and count:
                result.append((int(segment["segment_id"]), int(step["ordinal"]), count))
    return result


def deterministic_input(rank: int, element_count: int) -> bytes:
    """Independently reconstruct the versioned rank stimulus."""
    return b"".join(
        BASE._encode_value(
            (((index * 13 + rank * 29) % 127) - 63) / 16.0,
            "bfloat16",
        )
        for index in range(element_count)
    )


def row_parallel_input(rank: int) -> bytes:
    require(type(rank) is int and 0 <= rank < 2, "RowParallel input rank differs")
    full = b"".join(
        BASE._encode_value((((index * 13 + 7) % 127) - 63) / 16.0, "bfloat16")
        for index in range(3584)
    )
    shard_bytes = 1792 * 2
    return full[rank * shard_bytes:(rank + 1) * shard_bytes]


def row_parallel_oracles(
    workload: Mapping[str, Any]
) -> tuple[list[bytes], bytes]:
    import torch
    from safetensors import safe_open

    with safe_open(
        workload["model"]["weight_shard"]["path"], framework="pt", device="cpu"
    ) as tensors:
        weight = tensors.get_tensor(workload["model"]["tensor_key"])
    partials = []
    for rank in range(2):
        start = rank * 1792
        shard = weight[:, start:start + 1792].contiguous()
        values = torch.frombuffer(
            bytearray(row_parallel_input(rank)), dtype=torch.bfloat16
        ).view(1, 1792)
        partials.append(
            values.float().matmul(shard.float().t())
            .to(torch.bfloat16)
        )
    partial_payloads = [
        value.contiguous().view(torch.uint8).numpy().tobytes()
        for value in partials
    ]
    output = (partials[0].float() + partials[1].float()).to(torch.bfloat16)
    return partial_payloads, output.contiguous().view(torch.uint8).numpy().tobytes()


def row_parallel_weight_shard_sha256(
    workload: Mapping[str, Any], rank: int
) -> str:
    import torch
    from safetensors import safe_open

    require(type(rank) is int and 0 <= rank < 2, "RowParallel weight rank differs")
    with safe_open(
        workload["model"]["weight_shard"]["path"], framework="pt", device="cpu"
    ) as tensors:
        weight = tensors.get_tensor(workload["model"]["tensor_key"])
    shard = weight[:, rank * 1792:(rank + 1) * 1792].contiguous()
    return hashlib.sha256(shard.view(torch.uint8).numpy().tobytes()).hexdigest()


def row_parallel_oracle(workload: Mapping[str, Any]) -> bytes:
    return row_parallel_oracles(workload)[1]


def compare_bf16(
    actual: bytes, expected: bytes, *, label: str = "RowParallel output"
) -> dict[str, Any]:
    import torch

    left = torch.frombuffer(bytearray(actual), dtype=torch.bfloat16).float()
    right = torch.frombuffer(bytearray(expected), dtype=torch.bfloat16).float()
    require(left.numel() == right.numel() and torch.isfinite(left).all().item()
            and torch.isfinite(right).all().item(),
            f"{label} extent/finite contract differs")
    difference = (left - right).abs()
    allowed = 0.03125 + 0.03 * right.abs()
    mismatch = int((difference > allowed).sum().item())
    relative_l2 = float(torch.linalg.vector_norm(left - right).item()) / max(
        float(torch.linalg.vector_norm(right).item()), 1e-30
    )
    require(mismatch == 0 and relative_l2 <= 0.03,
            f"{label} differs from the independent sharded oracle")
    return {
        "atol": 0.03125, "rtol": 0.03, "max_relative_l2": 0.03,
        "mismatch_count": mismatch, "relative_l2": relative_l2,
        "max_abs": float(difference.max().item()),
        "expected_sha256": hashlib.sha256(expected).hexdigest(),
    }


def verify(source: Path, expected_path: Path, output: Path,
           *, live_identity: bool = True) -> dict[str, Any]:
    validate_absent_output(output)
    expected, expected_record = BASE.read_json(expected_path, "expected wrapper")
    design, plans = BASE.validate_expected(expected)
    manifest, manifest_record = BASE.read_json(source / "result-manifest.json",
                                               "run manifest")
    workload_binding = exact_keys(
        manifest.get("workload"), ("schema", "bytes", "sha256", "document"),
        "run workload binding",
    )
    workload = validate_workload(workload_binding["document"], live=live_identity)
    workload_payload = canonical_json(workload)
    require(workload_binding == {
        "schema": WORKLOAD_SCHEMA, "bytes": len(workload_payload),
        "sha256": hashlib.sha256(workload_payload).hexdigest(), "document": workload,
    }, "run workload identity differs")
    world = int(design["config"]["world_size"])
    require(world in ALL_WORLDS and design["config"]["dtype"] == "bfloat16",
            "vLLM adapter verifier accepts BF16 world_size 2..16")
    validate_source_inventory(source, world)
    validate_manifest_header(manifest, expected_record, design, workload)
    preflight = validate_identity_snapshot(manifest["source_identity_preflight"],
                                           live=live_identity)
    postflight = validate_identity_snapshot(manifest["source_identity_postflight"],
                                            live=live_identity)
    require(preflight == postflight, "source identity drifted during execution")
    require(design["runtime"]["path"] == preflight["runtime_library"]["path"]
            and design["runtime"]["sha256"] == preflight["runtime_library"]["sha256"],
            "expected runtime differs from the executed product")
    if live_identity:
        validate_product_identity(preflight)

    rank_entries = manifest["ranks"]
    require(isinstance(rank_entries, list)
            and [entry.get("rank") for entry in rank_entries] == list(range(world)),
            "run ranks are not canonical")
    inputs: list[bytes] = []
    outputs: list[bytes] = []
    adapters: list[dict[str, Any]] = []
    trace_summaries: list[dict[str, Any]] = []
    rank_results: list[dict[str, Any]] = []
    all_artifacts: dict[str, tuple[bytes, dict[str, Any]]] = {}
    row_parallel = workload["kind"] == "vllm-row-parallel"
    input_bytes = (1792 if row_parallel else int(design["config"]["element_count"])) * 2
    output_bytes = int(design["config"]["element_count"]) * 2

    for rank, entry in enumerate(rank_entries):
        exact_keys(entry,
                   ("rank", "worker_pid", "worker_start_time_ticks", "returncode",
                    "capability", "artifacts", "cleanup"),
                   f"rank {rank} manifest entry")
        require(entry["rank"] == rank and entry["returncode"] == 0,
                f"rank {rank} worker did not exit successfully")
        artifact_map = entry["artifacts"]
        require(isinstance(artifact_map, Mapping)
                and set(artifact_map) == set(RANK_FILES),
                f"rank {rank} artifact set differs")
        for name in RANK_FILES:
            relative = f"rank-{rank:02d}/{name}"
            limit = MAX_JOURNAL_BYTES if name.endswith(".jsonl") else MAX_ARTIFACT_BYTES
            payload, observed = BASE.file_record(source / relative, limit=limit)
            artifact_descriptor(artifact_map[name], relative, observed,
                                f"rank {rank} {name}")
            all_artifacts[relative] = (payload, observed)

        result_payload, result_record = all_artifacts[f"rank-{rank:02d}/worker-result.json"]
        result = BASE.parse_json_bytes(result_payload, f"rank {rank} worker result")
        require(result_payload == canonical_json(result),
                f"rank {rank} worker result is not canonical")
        exact_keys(result, RESULT_KEYS, f"rank {rank} worker result")
        require(result["schema"] == RANK_RESULT_SCHEMA
                and result["status"] == "success"
                and result["rank"] == rank
                and result["world_size"] == world
                and result["acceptance_authority"] is False
                and result["live_adapter_accepted"] is False
                and result["public_result_published"] is True
                and result["first_error"] is None,
                f"rank {rank} worker status/authority differs")
        validate_product_binding(result["product"], preflight, rank)

        launch_payload, _ = all_artifacts[f"rank-{rank:02d}/rank-launch.json"]
        launch = BASE.parse_json_bytes(launch_payload, f"rank {rank} launch")
        require(launch_payload == canonical_json(launch)
                and launch == design["ranks"][rank]["rank_launch"]
                and launch["schema"] == RANK_LAUNCH_SCHEMA
                and object_sha256(launch)
                == design["ranks"][rank]["rank_launch_sha256"],
                f"rank {rank} launch descriptor differs")

        capability = exact_keys(
            entry["capability"],
            ("parent_fd_identity", "pass_fds", "bootstrap_descriptor_sha256"),
            f"rank {rank} capability evidence",
        )
        bootstrap_payload, bootstrap_record = all_artifacts[
            f"rank-{rank:02d}/bootstrap-descriptor.json"
        ]
        bootstrap = BASE.parse_json_bytes(bootstrap_payload,
                                          f"rank {rank} bootstrap descriptor")
        require(bootstrap_payload == canonical_json(bootstrap)
                and capability["bootstrap_descriptor_sha256"]
                == bootstrap_record["sha256"],
                f"rank {rank} bootstrap encoding/hash differs")
        parent_capability = validate_bootstrap(
            bootstrap, rank=rank, design=design, identity=preflight,
            capability=capability,
        )

        events_payload, _ = all_artifacts[f"rank-{rank:02d}/adapter-events.jsonl"]
        parsed_events = BASE.parse_jsonl(events_payload, f"rank {rank} adapter events")
        (validate_row_parallel_events if row_parallel else validate_events)(
            parsed_events, rank
        )

        input_payload, input_record = all_artifacts[f"rank-{rank:02d}/input.bin"]
        output_payload, output_record = all_artifacts[f"rank-{rank:02d}/output.bin"]
        require(len(input_payload) == input_bytes and len(output_payload) == output_bytes,
                f"rank {rank} tensor extent differs")
        require(
            input_payload == (
                row_parallel_input(rank) if row_parallel else
                deterministic_input(rank, int(design["config"]["element_count"]))
            ),
            f"rank {rank} input differs from the independent deterministic stimulus",
        )
        require(result["input_sha256_before"] == input_record["sha256"]
                and result["input_sha256_after"] == input_record["sha256"]
                and result["output_sha256"] == output_record["sha256"]
                and result["output_storage_fresh"] is True,
                f"rank {rank} result tensor binding differs")

        managed = BASE.validate_managed_session(
            result["managed_session"], rank, design["config"], launch,
            design["runtime"], preflight["runtime_library"],
        )
        adapter_payload, adapter_record = all_artifacts[
            f"rank-{rank:02d}/adapter-evidence.json"
        ]
        adapter = BASE.parse_json_bytes(adapter_payload,
                                        f"rank {rank} adapter evidence")
        require(adapter_payload == canonical_json(adapter)
                and result["adapter_evidence_sha256"] == adapter_record["sha256"]
                and result["bootstrap_descriptor_sha256"] == bootstrap_record["sha256"],
                f"rank {rank} adapter/result artifact binding differs")
        adapters.append(validate_adapter(
            adapter, rank=rank, world=world, parent_capability=parent_capability,
            bootstrap_sha=bootstrap_record["sha256"], input_record=input_record,
            output_record=output_record, managed=managed, identity=preflight,
            workload=workload,
        ))

        log_summary = BASE.validate_log(
            all_artifacts[f"rank-{rank:02d}/gem5.log"][0], rank,
            launch, managed, preflight,
        )
        trace = BASE.parse_trace_jsonl(
            all_artifacts[f"rank-{rank:02d}/dispatch-trace.jsonl"][0],
            f"rank {rank} trace",
        )
        expected_steps = _device_steps(plans, rank)
        require(expected_steps, f"rank {rank} has no independently planned device SUM")
        if row_parallel:
            trace_kernels = [{
                "kernel": "dense_linear_kernel",
                "grid": [256, 64, 1], "workgroup": [256, 1, 1],
                "allocation_count": 3, "workgroups_completed": 64,
                "global_reads": None,
                "global_writes": math.ceil(output_bytes / 4),
            }] + [{
                "kernel": "_sum_kernel",
                "grid": [math.ceil(count / 256) * 256, 1, 1],
                "workgroup": [256, 1, 1], "allocation_count": 2,
                "workgroups_completed": math.ceil(count / 256),
                "global_reads": 2 * count, "global_writes": count,
            } for _, _, count in expected_steps]
            trace_summary = BASE.validate_kernel_trace(
                trace, trace_kernels, rank, managed, design["config"], log_summary
            )
        else:
            trace_summary = BASE.validate_trace(
                trace, expected_steps, rank, managed, design["config"], log_summary,
            )
        stats_summary = BASE.parse_stats(
            all_artifacts[f"rank-{rank:02d}/stats.txt"][0], rank,
            trace_summary["dispatch_count"],
        )
        require(stats_summary["sim_ticks"] == log_summary["exit_tick"]
                and trace_summary["terminal_sim_tick"] == stats_summary["sim_ticks"],
                f"rank {rank} trace/log/stats simulator tick differs")
        trace_summaries.append(trace_summary)
        rank_results.append(result)
        inputs.append(input_payload)
        outputs.append(output_payload)

    if row_parallel:
        require(outputs[0] == outputs[1], "RowParallel ranks do not converge bitwise")
        local_projections = [
            validate_local_projection(
                adapter["workload_evidence"]["local_projection"],
                rank,
                workload["layer"]["output_size"],
            )
            for rank, adapter in enumerate(adapters)
        ]
        expected_partials, expected_output = row_parallel_oracles(workload)
        partial_comparisons = [
            compare_bf16(
                local_projections[rank], expected_partials[rank],
                label=f"rank {rank} local projection",
            )
            for rank in range(world)
        ]
        oracle_outputs = BASE.ring_oracle(local_projections, "bfloat16", plans)
        require(outputs == oracle_outputs,
                "RowParallel allreduce differs from the actual-partial BF16 ring oracle")
        comparison = {
            "local_projection_by_rank": partial_comparisons,
            "collective_actual_partial_bitwise": True,
            "final_model_oracle": compare_bf16(outputs[0], expected_output),
        }
    else:
        oracle_outputs = BASE.ring_oracle(inputs, "bfloat16", plans)
        require(outputs == oracle_outputs,
                "rank outputs differ from the independent BF16 ring-step oracle")
        comparison = None
    require(len({hashlib.sha256(value).hexdigest() for value in outputs}) == 1,
            "rank outputs do not converge to one common SHA")
    oracle_sha = hashlib.sha256(oracle_outputs[0]).hexdigest()
    cleanup_summary = BASE.validate_supervisor_cleanup(
        manifest["supervisor_cleanup"], rank_entries, rank_results, world
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for relative, (payload, _) in all_artifacts.items():
            BASE.write_bytes(temporary / "source" / relative, payload)
        BASE.write_bytes(temporary / "source/result-manifest.json", canonical_json(manifest))
        BASE.write_bytes(temporary / "expected.json", canonical_json(expected))
        verifier_payload, _ = BASE.file_record(THIS_FILE)
        accepted_verifier = BASE.write_bytes(
            temporary / "acceptance/verifier.py", verifier_payload
        )
        accepted_verifier["path"] = "acceptance/verifier.py"
        base_payload, _ = BASE.file_record(BASE_VERIFIER_FILE)
        accepted_base = BASE.write_bytes(
            temporary / "acceptance/ccl-verifier-base.py", base_payload
        )
        accepted_base["path"] = "acceptance/ccl-verifier-base.py"
        result = {
            "schema": ACCEPTANCE_SCHEMA,
            "status": "success",
            "world_size": world,
            "formal_live_acceptance_world": world in FORMAL_WORLDS,
            "schema_valid_world": world in ALL_WORLDS,
            "expected_sha256": expected_record["sha256"],
            "source_manifest_sha256": manifest_record["sha256"],
            "source_identity": preflight,
            "acceptance_verifier": accepted_verifier,
            "ccl_verifier_base": accepted_base,
            "identity_unchanged_postflight": True,
            "authoritative_artifact_rehash": True,
            "planner_independently_recomputed": True,
            "descriptor_independently_recomputed": True,
            "bf16_ring_oracle_sha256": oracle_sha,
            "gloo_tensor_api_total": 0,
            "gloo_control_api_total": sum(
                adapter["gloo_tensor_api_total"] for adapter in adapters
            ) if row_parallel else 0,
            "group_coordinator_entrypoint_proven": True,
            "device_reduction_dispatch_count": sum(
                item["dispatch_count"] for item in trace_summaries
            ),
            "host_reduction_count": 0,
            "fallback_count": 0,
            "rank_adapters": adapters,
            "rank_traces": trace_summaries,
            "supervisor_cleanup": cleanup_summary,
            "live_adapter_accepted": world in FORMAL_WORLDS,
            "workload": workload,
            "row_parallel_comparison": comparison,
            "claim_boundary": (
                "One upstream vLLM RowParallelLinear with OOT local GEMM and device allreduce; "
                "this is not full-model tensor parallel acceptance."
                if row_parallel else
                "Standalone vLLM GroupCoordinator-to-GemSim communicator allreduce only; "
                "this is not vLLM tensor-parallel model or Qwen acceptance. Synthetic host "
                "fixtures validate the schema only and are never live evidence."
            ),
        }
        BASE.write_bytes(temporary / "result.json", canonical_json(result))
        artifact_manifest: dict[str, dict[str, Any]] = {}
        for artifact in sorted(path for path in temporary.rglob("*") if path.is_file()):
            _, record = BASE.file_record(artifact)
            artifact_manifest[str(artifact.relative_to(temporary))] = record
        BASE.write_bytes(temporary / "manifest.json", canonical_json({
            "schema": MANIFEST_SCHEMA,
            "artifacts": artifact_manifest,
            "complete": True,
        }))
        BASE.fsync_tree(temporary)
        BASE.rename_noreplace(temporary, output)
        temporary = None
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return result
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify one vLLM GemSim communicator workload"
    )
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--emit-row-parallel-workload", type=Path)
    parser.add_argument("--model-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.emit_row_parallel_workload is not None:
        if args.model_root is None or any(
            value is not None for value in (args.source_dir, args.expected, args.output_dir)
        ):
            raise SystemExit(
                "--emit-row-parallel-workload requires --model-root and no verify paths"
            )
        output = Path(os.path.abspath(args.emit_row_parallel_workload))
        validate_absent_output(output)
        workload = build_row_parallel_workload(
            Path(os.path.abspath(args.model_root))
        )
        record = BASE.write_bytes(output, canonical_json(workload))
        print(json.dumps({"path": str(output), **record}, sort_keys=True))
        return 0
    if args.model_root is not None or any(
        value is None for value in (args.source_dir, args.expected, args.output_dir)
    ):
        raise SystemExit("verification requires --source-dir, --expected, and --output-dir")
    result = verify(
        Path(os.path.abspath(args.source_dir)),
        Path(os.path.abspath(args.expected)),
        Path(os.path.abspath(args.output_dir)),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
