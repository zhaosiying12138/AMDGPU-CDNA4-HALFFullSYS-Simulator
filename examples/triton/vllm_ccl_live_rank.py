#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""One frozen-product rank for the formal vLLM CCL adapter live gate."""

from __future__ import annotations

if __name__ == "__main__":
    __import__("runpy").run_path(
        __file__.replace("vllm_ccl_live_rank.py", "_gemsim_bootstrap.py")
    )["bootstrap"](__file__, "vllm-ccl-live")

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterator, Mapping, Sequence
import pickle


CONFIG_SCHEMA = "amdgpu-sim.vllm-ccl-live-rank-config.v1"
RESULT_SCHEMA = "amdgpu-sim.vllm-ccl-live-rank-result.v1"
ADAPTER_SCHEMA = "amdgpu-sim.vllm-ccl-live-adapter-evidence.v1"
EVENT_SCHEMA = "amdgpu-sim.vllm-ccl-live-adapter-event.v1"
PROCESS_GROUP_AUDIT_SCHEMA = "amdgpu-sim.vllm-gloo-process-group-audit.v1"
DISPATCH_CAPTURE_SCHEMA = "amdgpu-sim.torch-dispatch-output-capture.v1"
ROW_PARALLEL_LOCAL_OPERATOR = "gemsim.dense_linear.default"
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


class RankError(RuntimeError):
    pass


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_size <= 0
            or metadata.st_size > (1 << 30)
        ):
            raise RankError(f"unsafe imported source identity: {path}")
        chunks = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RankError(f"imported source was truncated: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RankError(f"imported source changed while reading: {path}")
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def tensor_bytes(tensor: Any) -> bytes:
    torch = __import__("torch")
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def deterministic_input(torch: Any, rank: int, element_count: int) -> Any:
    values = [
        (((index * 13 + rank * 29) % 127) - 63) / 16.0
        for index in range(element_count)
    ]
    return torch.tensor(values, dtype=torch.bfloat16)


def fd_identity(descriptor: int) -> dict[str, int | str]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISSOCK(metadata.st_mode):
        raise RankError("inherited CCL capability FD is not a socket")
    return {
        "fd": descriptor,
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": int(metadata.st_mode),
        "target": os.readlink(f"/proc/self/fd/{descriptor}"),
    }


def exclusive_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_private(path: Path, maximum: int, label: str) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RankError(f"could not safely open {label}") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
            or metadata.st_size <= 0
            or metadata.st_size > maximum
        ):
            raise RankError(f"{label} is not a private owned regular file")
        chunks = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise RankError(f"{label} was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RankError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def read_config(path: Path, capability_fd: int) -> dict[str, Any]:
    payload = read_private(path, 1024 * 1024, "worker config")
    try:
        config = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RankError("worker config is invalid ASCII JSON") from error
    required = {
        "schema", "rank", "world_size", "element_count", "dtype",
        "unique_name", "rendezvous_path", "bootstrap_descriptor_path",
        "bootstrap_descriptor_sha256", "result_path", "adapter_evidence_path",
        "journal_path", "input_path", "output_path", "runtime_library",
        "rank_launch_sha256", "epoch", "group_generation", "job_uuid",
        "group_uuid", "model_identity_sha256", "expected_imports", "product",
        "workload",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise RankError("worker config fields are invalid")
    if config["schema"] != CONFIG_SCHEMA or payload != canonical_json(config):
        raise RankError("worker config schema or canonical encoding is invalid")
    rank, world = config["rank"], config["world_size"]
    if (
        type(world) is not int or not 2 <= world <= 16
        or type(rank) is not int or not 0 <= rank < world
        or type(config["element_count"]) is not int
        or config["element_count"] <= 0
        or config["dtype"] != "bfloat16"
        or config["unique_name"] != "tp:0"
    ):
        raise RankError("worker topology or tensor contract is invalid")
    descriptor_path = Path(config["bootstrap_descriptor_path"])
    descriptor_payload = read_private(
        descriptor_path, 1024 * 1024, "CCL bootstrap descriptor"
    )
    if sha256_bytes(descriptor_payload) != config["bootstrap_descriptor_sha256"]:
        raise RankError("CCL bootstrap descriptor identity mismatch")
    descriptor = json.loads(descriptor_payload.decode("ascii"))
    groups = descriptor.get("groups") if isinstance(descriptor, Mapping) else None
    if (
        not isinstance(groups, list) or len(groups) != 1
        or groups[0].get("unique_name") != "tp:0"
        or groups[0].get("rank", {}).get("capability_fd") != capability_fd
    ):
        raise RankError("CCL bootstrap descriptor does not bind this capability")
    return config


def _gloo_backend_name(value: Any) -> str:
    text = str(value).lower()
    return "gloo" if text in {"gloo", "backend.gloo"} else text


def _bound_call_argument(
    function: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    name: str,
    default: Any,
) -> Any:
    try:
        arguments = inspect.signature(function).bind_partial(
            *args, **kwargs
        ).arguments
    except (TypeError, ValueError) as error:
        raise RankError(
            f"could not bind torch.distributed call arguments: {name}"
        ) from error
    return arguments.get(name, default)


@contextmanager
def audit_standard_vllm_control(
    dist: Any, world_size: int, rank: int
) -> Iterator[dict[str, Any]]:
    """Allow only bounded upstream initialization metadata on Gloo."""
    import torch.distributed.distributed_c10d as distributed_c10d

    if (
        type(world_size) is not int or not 2 <= world_size <= 16
        or type(rank) is not int or not 0 <= rank < world_size
    ):
        raise RankError("Gloo audit topology is invalid")
    counters = {name: 0 for name in TENSOR_COLLECTIVE_APIS}
    records: list[dict[str, Any]] = []
    originals: dict[str, Any] = {}
    state = {"phase": "initialization", "object_broadcast_depth": 0}
    process_groups: dict[str, Any] = {
        "schema": PROCESS_GROUP_AUDIT_SCHEMA,
        "init": [],
        "new": [],
        "destroy": [],
        "local_tokens_created": [],
        "local_tokens_destroyed": [],
        "default_destroyed": False,
        "all_local_groups_destroyed": False,
    }
    default_active = False
    local_groups: dict[int, int] = {}
    next_local_token = 1

    for name in ("init_process_group", "new_group", "destroy_process_group"):
        if not hasattr(dist, name):
            raise RankError(f"required torch.distributed process-group symbol is absent: {name}")
        originals[name] = getattr(dist, name)

    original_init = originals["init_process_group"]
    original_new_group = originals["new_group"]
    original_destroy = originals["destroy_process_group"]

    def audited_init_process_group(*args: Any, **kwargs: Any) -> Any:
        nonlocal default_active
        if state["phase"] != "initialization" or default_active:
            raise RankError("default process group initialization is out of phase")
        backend = _bound_call_argument(
            original_init, args, kwargs, "backend", None
        )
        observed_rank = _bound_call_argument(
            original_init, args, kwargs, "rank", -1
        )
        observed_world = _bound_call_argument(
            original_init, args, kwargs, "world_size", -1
        )
        if (
            _gloo_backend_name(backend) != "gloo"
            or observed_rank != rank
            or observed_world != world_size
        ):
            raise RankError("default process group is not the exact Gloo topology")
        result = original_init(*args, **kwargs)
        if _gloo_backend_name(dist.get_backend()) != "gloo":
            raise RankError("default process group actual backend is not Gloo")
        default_active = True
        process_groups["init"].append({
            "ordinal": 0,
            "phase": "initialization",
            "backend": "gloo",
            "rank": rank,
            "world_size": world_size,
        })
        return result

    def audited_new_group(*args: Any, **kwargs: Any) -> Any:
        nonlocal next_local_token
        if state["phase"] != "initialization" or not default_active:
            raise RankError("new process group is out of phase")
        ranks = _bound_call_argument(
            original_new_group, args, kwargs, "ranks", None
        )
        backend = _bound_call_argument(
            original_new_group, args, kwargs, "backend", None
        )
        if (
            not isinstance(ranks, (list, tuple))
            or not ranks
            or any(type(value) is not int for value in ranks)
            or len(set(ranks)) != len(ranks)
            or any(value < 0 or value >= world_size for value in ranks)
            or _gloo_backend_name(backend) != "gloo"
        ):
            raise RankError("new process group is not a canonical Gloo rank set")
        group = original_new_group(*args, **kwargs)
        non_member = getattr(
            getattr(dist, "GroupMember", object()), "NON_GROUP_MEMBER", object()
        )
        local_member = group is not None and group is not non_member
        token = None
        if local_member:
            if _gloo_backend_name(dist.get_backend(group)) != "gloo":
                raise RankError("new process group actual backend is not Gloo")
            identity = id(group)
            if identity in local_groups:
                raise RankError("new process group reused a live local handle")
            token = next_local_token
            next_local_token += 1
            local_groups[identity] = token
            process_groups["local_tokens_created"].append(token)
        process_groups["new"].append({
            "ordinal": len(process_groups["new"]),
            "phase": "initialization",
            "backend": "gloo",
            "ranks": list(ranks),
            "local_member": local_member,
            "local_token": token,
        })
        return group

    def audited_destroy_process_group(*args: Any, **kwargs: Any) -> Any:
        nonlocal default_active
        if state["phase"] != "cleanup":
            raise RankError("process group destruction is out of phase")
        group = _bound_call_argument(
            original_destroy, args, kwargs, "group", None
        )
        world_group = getattr(getattr(dist, "group", object()), "WORLD", object())
        if group is None or group is world_group:
            if not default_active:
                raise RankError("default process group was destroyed more than once")
            result = original_destroy(*args, **kwargs)
            default_active = False
            target: str | int = "default"
            process_groups["default_destroyed"] = True
        else:
            identity = id(group)
            if identity not in local_groups:
                raise RankError("unknown or already-destroyed local process group")
            token = local_groups[identity]
            result = original_destroy(*args, **kwargs)
            del local_groups[identity]
            process_groups["local_tokens_destroyed"].append(token)
            target = token
        process_groups["destroy"].append({
            "ordinal": len(process_groups["destroy"]),
            "phase": "cleanup",
            "target": target,
        })
        return result

    dist.init_process_group = audited_init_process_group
    dist.new_group = audited_new_group
    dist.destroy_process_group = audited_destroy_process_group
    allowed = {"all_reduce", "barrier", "broadcast_object_list"}
    for name in TENSOR_COLLECTIVE_APIS:
        if not hasattr(dist, name):
            raise RankError(f"required torch.distributed audit symbol is absent: {name}")
        original = getattr(dist, name)
        originals[name] = original

        def audited(*args: Any, _name: str = name, _original: Any = original,
                    **kwargs: Any) -> Any:
            counters[_name] += 1
            if state["phase"] != "initialization" or (
                _name not in allowed
            ):
                raise RankError(
                    f"Gloo tensor/model payload is forbidden during {state['phase']}: {_name}"
                )
            record: dict[str, Any] = {
                "api": _name, "phase": "initialization"
            }
            if _name == "all_reduce":
                tensor = args[0] if args else kwargs.get("tensor")
                if (
                    tensor is None or tensor.device.type != "cpu"
                    or tensor.dtype != __import__("torch").int32
                    or tuple(tensor.shape) != (world_size,)
                ):
                    raise RankError("upstream initialization all_reduce is not int32 control data")
                record.update(dtype="int32", shape=[world_size], bytes=world_size * 4)
            elif _name == "broadcast_object_list":
                objects = args[0] if args else kwargs.get("object_list")
                if not isinstance(objects, list) or len(objects) != 1:
                    raise RankError("upstream initialization object broadcast shape differs")
                serialized = pickle.dumps(objects, protocol=4)
                if len(serialized) > 64 * 1024:
                    raise RankError("upstream initialization object metadata exceeds 64KiB")
                record.update(dtype="python-control-object", shape=[1], bytes=len(serialized))
            else:
                record.update(dtype="none", shape=[], bytes=0)
            records.append(record)
            if _name == "broadcast_object_list":
                state["object_broadcast_depth"] += 1
                try:
                    return _original(*args, **kwargs)
                finally:
                    state["object_broadcast_depth"] -= 1
            return _original(*args, **kwargs)

        setattr(dist, name, audited)
    internal_broadcast = distributed_c10d.broadcast

    def audited_object_broadcast(*args: Any, **kwargs: Any) -> Any:
        counters["broadcast"] += 1
        if state["phase"] != "initialization" or state["object_broadcast_depth"] == 0:
            raise RankError("internal Gloo broadcast is outside object metadata exchange")
        tensor = args[0] if args else kwargs.get("tensor")
        torch = __import__("torch")
        if (
            tensor is None or tensor.device.type != "cpu"
            or tensor.dtype not in (torch.uint8, torch.int64)
            or tensor.numel() * tensor.element_size() > 64 * 1024
        ):
            raise RankError("object broadcast payload is not bounded CPU metadata")
        records.append({
            "api": "broadcast",
            "phase": "initialization",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "bytes": tensor.numel() * tensor.element_size(),
        })
        return internal_broadcast(*args, **kwargs)

    distributed_c10d.broadcast = audited_object_broadcast
    audit = {
        "phase": state,
        "counts": counters,
        "records": records,
        "process_groups": process_groups,
    }
    try:
        yield audit
        if (
            len(process_groups["init"]) != 1
            or default_active
            or local_groups
            or sorted(process_groups["local_tokens_destroyed"])
            != process_groups["local_tokens_created"]
            or process_groups["default_destroyed"] is not True
        ):
            raise RankError("Gloo process-group lifecycle did not close exactly")
        process_groups["all_local_groups_destroyed"] = True
    finally:
        for name, original in originals.items():
            setattr(dist, name, original)
        distributed_c10d.broadcast = internal_broadcast


def row_parallel_input(torch: Any, rank: int) -> Any:
    full = torch.tensor(
        [(((index * 13 + 7) % 127) - 63) / 16.0 for index in range(3584)],
        dtype=torch.bfloat16,
    ).view(1, 3584)
    return full[:, rank * 1792:(rank + 1) * 1792].contiguous()


@contextmanager
def capture_operator_output(operator: str) -> Iterator[list[dict[str, Any]]]:
    """Capture one operator result without replacing an upstream or OOT symbol."""
    from torch.utils._python_dispatch import TorchDispatchMode

    if not isinstance(operator, str) or not operator:
        raise RankError("dispatch capture operator is invalid")
    records: list[dict[str, Any]] = []

    class CaptureMode(TorchDispatchMode):
        def __torch_dispatch__(
            self, function: Any, types: Any, args: tuple[Any, ...] = (),
            kwargs: Mapping[str, Any] | None = None,
        ) -> Any:
            del types
            output = function(*args, **({} if kwargs is None else dict(kwargs)))
            if str(function) == operator:
                torch = __import__("torch")
                if not isinstance(output, torch.Tensor):
                    raise RankError("captured operator did not return one tensor")
                payload = tensor_bytes(output)
                records.append({
                    "schema": DISPATCH_CAPTURE_SCHEMA,
                    "operator": operator,
                    "dtype": str(output.dtype).removeprefix("torch."),
                    "shape": list(output.shape),
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                    "payload_hex": payload.hex(),
                })
            return output

    with CaptureMode():
        yield records
    if len(records) != 1:
        raise RankError(
            f"expected exactly one {operator} result, observed {len(records)}"
        )


def _actual_imports(*, row_parallel: bool = False) -> dict[str, dict[str, Any]]:
    import gemsim_ccl.engine as ccl_engine
    import gemsim_vllm
    import gemsim_vllm.adapters as vllm_adapters
    import gemsim_vllm.communicator as vllm_communicator
    import gemsim_vllm.kernels as vllm_kernels
    import gemsim_vllm.ops as vllm_ops
    import gemsim_vllm.platform as vllm_platform
    import gemsim_vllm.row_parallel as vllm_row_parallel
    import triton.backends.gemsim_amd.driver as triton_driver
    import vllm.config.parallel as vllm_config_parallel
    import vllm.config.model as vllm_config_model
    import vllm.config.vllm as vllm_config_vllm
    import vllm.distributed.communication_op as communication_op
    import vllm.distributed.device_communicators.base_device_communicator as base_communicator
    import vllm.distributed.parallel_state as parallel_state
    import vllm.model_executor.layers.linear as vllm_linear
    import vllm.version as vllm_version

    modules = {
        "vllm_parallel_state": parallel_state,
        "vllm_base_communicator": base_communicator,
        "vllm_communication_op": communication_op,
        "vllm_version": vllm_version,
        "vllm_plugin_init": gemsim_vllm,
        "vllm_communicator": vllm_communicator,
        "vllm_platform": vllm_platform,
        "ccl_engine": ccl_engine,
        "triton_driver": triton_driver,
        "vllm_linear": vllm_linear,
        "vllm_config_vllm": vllm_config_vllm,
        "vllm_config_parallel": vllm_config_parallel,
        "vllm_config_model": vllm_config_model,
        "vllm_adapters": vllm_adapters,
        "vllm_row_parallel": vllm_row_parallel,
        "vllm_ops": vllm_ops,
        "vllm_kernels": vllm_kernels,
    }
    if not row_parallel:
        modules = {name: module for name, module in modules.items() if name in {
            "vllm_parallel_state", "vllm_base_communicator", "vllm_communication_op",
            "vllm_version", "vllm_plugin_init", "vllm_communicator", "vllm_platform",
            "ccl_engine", "triton_driver",
        }}
    return {name: file_record(Path(module.__file__)) for name, module in modules.items()}


def run_row_parallel_rank(
    config: Mapping[str, Any], capability_fd: int, inherited_capability: Mapping[str, Any]
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    import triton

    rank = int(config["rank"])
    world = int(config["world_size"])
    workload = config["workload"]
    if (
        not isinstance(workload, Mapping)
        or workload.get("schema") != "amdgpu-sim.vllm-ccl-workload.v1"
        or workload.get("kind") != "vllm-row-parallel"
        or world != 2
    ):
        raise RankError("RowParallel workload contract differs")
    input_tensor = row_parallel_input(torch, rank)
    input_before = tensor_bytes(input_tensor)
    if sha256_bytes(input_before) != workload["input"]["sha256_by_rank"][rank]:
        raise RankError("RowParallel deterministic input identity differs")
    exclusive_write(Path(config["input_path"]), input_before)
    append_event(Path(config["journal_path"]), rank, 0, "worker_started")
    process_group_initialized = False
    model_parallel_initialized = False
    model_parallel_destroyed = False
    default_group_destroyed = False
    adapter: dict[str, Any] = {}
    audit_counts = {name: 0 for name in TENSOR_COLLECTIVE_APIS}
    try:
        from vllm.config import (
            ModelConfig, ParallelConfig, VllmConfig, set_current_vllm_config,
        )
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
            get_tp_group,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.model_executor.layers.linear import RowParallelLinear
        from vllm.platforms import current_platform
        from vllm.plugins import load_general_plugins
        from safetensors import safe_open

        load_general_plugins()
        coordinator_methods_before = {
            name: getattr(__import__(
                "vllm.distributed.parallel_state", fromlist=["GroupCoordinator"]
            ).GroupCoordinator, name)
            for name in ("broadcast", "broadcast_tensor_dict")
        }
        coordinator_methods = coordinator_method_evidence(
            __import__(
                "vllm.distributed.parallel_state", fromlist=["GroupCoordinator"]
            ).GroupCoordinator,
            coordinator_methods_before,
        )
        if (
            type(current_platform).__module__ != "gemsim_vllm.platform"
            or type(current_platform).__qualname__ != "GemsimPlatform"
        ):
            raise RankError("active vLLM platform is not GemsimPlatform")
        parallel = ParallelConfig(
            tensor_parallel_size=world,
            distributed_executor_backend="external_launcher",
        )
        vllm_config = VllmConfig(
            model_config=ModelConfig(
                model=workload["model"]["root"],
                tokenizer=workload["model"]["root"],
                skip_tokenizer_init=True,
                dtype="bfloat16",
                enforce_eager=True,
            ),
            parallel_config=parallel,
        )
        current_platform.check_and_update_config(vllm_config)
        with audit_standard_vllm_control(dist, world, rank) as control_audit:
            with set_current_vllm_config(vllm_config):
                init_distributed_environment(
                    world_size=world,
                    rank=rank,
                    distributed_init_method=Path(config["rendezvous_path"]).as_uri(),
                    local_rank=rank,
                    backend="gloo",
                )
                process_group_initialized = True
                initialize_model_parallel(world, 1, backend="gloo")
                model_parallel_initialized = True
                coordinator = get_tp_group()
                communicator = coordinator.device_communicator
                if (
                    coordinator.unique_name != "tp:0"
                    or type(communicator).__module__ != "gemsim_vllm.communicator"
                    or type(communicator).__qualname__ != "GemsimDeviceCommunicator"
                    or coordinator.use_custom_op_call is not False
                    or coordinator.mq_broadcaster is None
                ):
                    raise RankError("standard vLLM TP initialization did not select GemSim")
                append_event(Path(config["journal_path"]), rank, 1,
                             "standard_model_parallel_initialized")
                layer_spec = workload["layer"]
                layer = RowParallelLinear(
                    layer_spec["input_size"],
                    layer_spec["output_size"],
                    bias=layer_spec["bias"],
                    input_is_parallel=layer_spec["input_is_parallel"],
                    params_dtype=torch.bfloat16,
                    reduce_results=layer_spec["reduce_results"],
                    return_bias=layer_spec["return_bias"],
                    prefix="model.language_model.layers.0.mlp.down_proj",
                )
                if (
                    type(layer).__module__ != "gemsim_vllm.adapters"
                    or type(layer).__qualname__ != "GemsimRowParallelLinear"
                    or "forward" in type(layer).__dict__
                ):
                    raise RankError("OOT RowParallel layer did not inherit upstream forward")
                model = workload["model"]
                with safe_open(
                    model["weight_shard"]["path"], framework="pt", device="cpu"
                ) as tensors:
                    full_weight = tensors.get_tensor(model["tensor_key"])
                full_weight_payload = tensor_bytes(full_weight)
                if sha256_bytes(full_weight_payload) != model["tensor_sha256"]:
                    raise RankError("RowParallel full checkpoint tensor drifted")
                layer.weight.weight_loader(layer.weight, full_weight)
                loaded = {"weight"}
                shard_before = tensor_bytes(layer.weight)
                expected_columns = [0, 1792] if rank == 0 else [1792, 3584]
                expected_shard = full_weight[:, expected_columns[0]:expected_columns[1]]
                if shard_before != tensor_bytes(expected_shard.contiguous()):
                    raise RankError("upstream RowParallel weight loader shard differs")
                append_event(Path(config["journal_path"]), rank, 2,
                             "row_parallel_layer_ready")
                control_audit["phase"]["phase"] = "model_ready"
                with capture_operator_output(ROW_PARALLEL_LOCAL_OPERATOR) as captures:
                    result = layer(input_tensor)
                if not isinstance(result, tuple) or len(result) != 2 or result[1] is not None:
                    raise RankError("upstream RowParallel return contract differs")
                output = result[0]
                local_projection = captures[0]
                if (
                    local_projection["dtype"] != "bfloat16"
                    or local_projection["shape"] != [1, layer_spec["output_size"]]
                    or local_projection["bytes"] != layer_spec["output_size"] * 2
                ):
                    raise RankError("captured RowParallel local projection contract differs")
                control_audit["phase"]["phase"] = "cleanup"
                append_event(Path(config["journal_path"]), rank, 3,
                             "upstream_row_parallel_forward_returned")
                destroy_model_parallel()
                model_parallel_initialized = False
                model_parallel_destroyed = True
                destroy_distributed_environment()
                process_group_initialized = False
                default_group_destroyed = True
        input_after = tensor_bytes(input_tensor)
        output_payload = tensor_bytes(output)
        shard_after = tensor_bytes(layer.weight)
        if input_after != input_before or shard_after != shard_before:
            raise RankError("RowParallel forward modified public input or weight shard")
        actual_imports = _actual_imports(row_parallel=True)
        if actual_imports != config["expected_imports"]:
            raise RankError("actual imported sources differ from runner preflight")
        driver = triton.runtime.driver.active
        driver.runtime._ensure_session()
        managed = managed_session_record(driver, config["rank_launch_sha256"])
        installed_version = __import__(
            "importlib.metadata", fromlist=["version"]
        ).version("vllm")
        audit_counts = dict(control_audit["counts"])
        control_records = list(control_audit["records"])
        adapter = {
            "schema": ADAPTER_SCHEMA,
            "rank": rank,
            "world_size": world,
            "entrypoint": "vllm.model_executor.layers.linear.RowParallelLinear.forward",
            "coordinator_class": "vllm.distributed.parallel_state.GroupCoordinator",
            "communicator_class": "gemsim_vllm.communicator.GemsimDeviceCommunicator",
            "platform_class": "gemsim_vllm.platform.GemsimPlatform",
            "unique_name": "tp:0",
            "control_backend": "gloo",
            "control_process_groups": control_audit["process_groups"],
            "tensor_data_backend": "gemsim_ccl_engine",
            "message_queue_broadcaster": True,
            "use_custom_op_call": False,
            "coordinator_methods_unmodified": coordinator_methods,
            "gloo_tensor_api_counts": audit_counts,
            "gloo_tensor_api_total": sum(audit_counts.values()),
            "gloo_control_records": control_records,
            "capability_fd_identity": inherited_capability,
            "bootstrap_descriptor_sha256": config["bootstrap_descriptor_sha256"],
            "input_sha256_before": sha256_bytes(input_before),
            "input_sha256_after": sha256_bytes(input_after),
            "output_sha256": sha256_bytes(output_payload),
            "output_storage_fresh": output.untyped_storage().data_ptr()
            != input_tensor.untyped_storage().data_ptr(),
            "engine_rank": rank,
            "engine_world_size": world,
            "engine_state_after_collective": "ready",
            "actual_imports": actual_imports,
            "vllm_installed_version": installed_version,
            "managed_session": managed,
            "coordinator_destroyed": model_parallel_destroyed,
            "default_group_destroyed": default_group_destroyed,
            "workload_evidence": {
                "kind": "vllm-row-parallel",
                "layer_class": f"{type(layer).__module__}.{type(layer).__qualname__}",
                "forward_inherited": "forward" not in type(layer).__dict__,
                "loader": "vllm RowParallelLinear parameter weight_loader hook",
                "loaded_parameters": sorted(loaded),
                "weight_shard_columns": expected_columns,
                "weight_shard_sha256_before": sha256_bytes(shard_before),
                "weight_shard_sha256_after": sha256_bytes(shard_after),
                "local_projection": local_projection,
            },
        }
        exclusive_write(Path(config["adapter_evidence_path"]), canonical_json(adapter))
        exclusive_write(Path(config["output_path"]), output_payload)
        append_event(Path(config["journal_path"]), rank, 4, "cleanup_complete")
        rank_result = {
            "schema": RESULT_SCHEMA, "status": "success", "rank": rank,
            "world_size": world, "acceptance_authority": False,
            "live_adapter_accepted": False, "public_result_published": True,
            "input_sha256_before": sha256_bytes(input_before),
            "input_sha256_after": sha256_bytes(input_after),
            "output_sha256": sha256_bytes(output_payload),
            "output_storage_fresh": adapter["output_storage_fresh"],
            "bootstrap_descriptor_sha256": config["bootstrap_descriptor_sha256"],
            "adapter_evidence_sha256": sha256_bytes(canonical_json(adapter)),
            "managed_session": managed, "first_error": None,
            "product": config["product"],
        }
        exclusive_write(Path(config["result_path"]), canonical_json(rank_result))
        return rank_result
    except BaseException:
        if model_parallel_initialized:
            try:
                destroy_model_parallel()
            except Exception:
                pass
        if process_group_initialized:
            try:
                destroy_distributed_environment()
            except Exception:
                pass
        raise


def managed_session_record(driver: Any, rank_launch_sha256: str) -> dict[str, Any]:
    runtime = driver.runtime
    info = runtime.session_info
    if info is None or not runtime.session.value:
        raise RankError("managed simulator session is unavailable")
    return {
        "child_pid": int(info.child_pid),
        "connection_id": int(info.connection_id),
        "epoch": int(info.epoch),
        "rank": int(info.rank),
        "world_size": int(info.world_size),
        "daemon_uuid": bytes(info.daemon_uuid).hex(),
        "job_uuid": bytes(info.job_uuid).hex(),
        "runtime_library": str(Path(runtime.path).resolve(strict=True)),
        "rank_launch_sha256": rank_launch_sha256,
    }


@contextmanager
def reject_tensor_collectives(dist: Any) -> Iterator[dict[str, int]]:
    """Fail on every public torch.distributed tensor payload path."""
    counters = {name: 0 for name in TENSOR_COLLECTIVE_APIS}
    originals: dict[str, Any] = {}
    for name in TENSOR_COLLECTIVE_APIS:
        if not hasattr(dist, name):
            raise RankError(f"required torch.distributed audit symbol is absent: {name}")
        originals[name] = getattr(dist, name)

        def rejected(*_args: Any, _name: str = name, **_kwargs: Any) -> Any:
            counters[_name] += 1
            raise RankError(
                f"Gloo tensor collective is forbidden in adapter gate: {_name}"
            )

        setattr(dist, name, rejected)
    try:
        yield counters
    finally:
        for name, original in originals.items():
            setattr(dist, name, original)


def coordinator_method_evidence(
    group_coordinator: Any, before: Mapping[str, Any]
) -> dict[str, str]:
    """Prove plugin registration did not replace upstream coordinator methods."""
    expected_names = ("broadcast", "broadcast_tensor_dict")
    if set(before) != set(expected_names):
        raise RankError("upstream GroupCoordinator method baseline differs")
    result = {}
    for name in expected_names:
        method = getattr(group_coordinator, name)
        if method is not before[name]:
            raise RankError(f"GemSim plugin replaced GroupCoordinator.{name}")
        result[name] = "upstream-object-identity-preserved"
    return result


def append_event(path: Path, rank: int, ordinal: int, event: str, **fields: Any) -> None:
    document = {
        "schema": EVENT_SCHEMA,
        "ordinal": ordinal,
        "rank": rank,
        "event": event,
        **fields,
    }
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, canonical_json(document))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_rank(config_path: Path, capability_fd: int) -> dict[str, Any]:
    config = read_config(config_path, capability_fd)
    rank = int(config["rank"])
    world = int(config["world_size"])
    inherited_capability = fd_identity(capability_fd)
    os.environ["GEMSIM_CCL_BOOTSTRAP_DESCRIPTOR"] = config[
        "bootstrap_descriptor_path"
    ]
    if config["workload"].get("kind") == "vllm-row-parallel":
        return run_row_parallel_rank(config, capability_fd, inherited_capability)

    import torch
    import torch.distributed as dist
    import triton

    input_tensor = deterministic_input(torch, rank, int(config["element_count"]))
    input_before = tensor_bytes(input_tensor)
    exclusive_write(Path(config["input_path"]), input_before)
    append_event(Path(config["journal_path"]), rank, 0, "worker_started")
    coordinator = None
    output = None
    audit_counts = {name: 0 for name in TENSOR_COLLECTIVE_APIS}
    adapter: dict[str, Any] = {}
    process_group_initialized = False
    coordinator_destroyed = False
    default_group_destroyed = False
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=Path(config["rendezvous_path"]).as_uri(),
            rank=rank,
            world_size=world,
        )
        process_group_initialized = True
        append_event(Path(config["journal_path"]), rank, 1,
                     "default_gloo_group_initialized")
        with reject_tensor_collectives(dist) as audit_counts:
            from vllm.distributed.parallel_state import GroupCoordinator
            import vllm.distributed.parallel_state as parallel_state

            coordinator_methods_before = {
                name: getattr(GroupCoordinator, name)
                for name in ("broadcast", "broadcast_tensor_dict")
            }
            from vllm.plugins import load_general_plugins

            load_general_plugins()
            import vllm.distributed.communication_op as communication_op
            import vllm.version as vllm_version
            import vllm.distributed.device_communicators.base_device_communicator as base_communicator
            import triton.backends.gemsim_amd.driver as triton_driver
            from vllm.platforms import current_platform
            import gemsim_ccl.engine as ccl_engine
            import gemsim_vllm
            import gemsim_vllm.communicator as vllm_communicator
            import gemsim_vllm.platform as vllm_platform

            coordinator_methods = coordinator_method_evidence(
                GroupCoordinator, coordinator_methods_before
            )
            if (
                type(current_platform).__module__ != "gemsim_vllm.platform"
                or type(current_platform).__qualname__ != "GemsimPlatform"
            ):
                raise RankError("active vLLM platform is not GemsimPlatform")
            coordinator = GroupCoordinator(
                group_ranks=[list(range(world))],
                local_rank=rank,
                torch_distributed_backend="gloo",
                use_device_communicator=True,
                use_message_queue_broadcaster=False,
                group_name="tp",
                use_all2all=False,
            )
            communicator = coordinator.device_communicator
            if (
                coordinator.unique_name != "tp:0"
                or type(communicator).__module__ != "gemsim_vllm.communicator"
                or type(communicator).__qualname__ != "GemsimDeviceCommunicator"
                or coordinator.use_custom_op_call is not False
                or coordinator.mq_broadcaster is not None
            ):
                raise RankError("vLLM coordinator did not select the exact GemSim adapter")
            append_event(Path(config["journal_path"]), rank, 2,
                         "coordinator_ready", unique_name=coordinator.unique_name)
            output = coordinator.all_reduce(input_tensor)
            append_event(Path(config["journal_path"]), rank, 3,
                         "coordinator_all_reduce_returned")
            output_payload = tensor_bytes(output)
            input_after = tensor_bytes(input_tensor)
            fresh = (
                output.untyped_storage().data_ptr()
                != input_tensor.untyped_storage().data_ptr()
            )
            if input_after != input_before or not fresh:
                raise RankError("adapter violated input/fresh-output contract")
            if any(audit_counts.values()):
                raise RankError("a Gloo tensor collective was attempted")
            driver = triton.runtime.driver.active
            driver.runtime._ensure_session()
            managed = managed_session_record(driver, config["rank_launch_sha256"])
            if (
                managed["rank"] != rank
                or managed["world_size"] != world
                or managed["epoch"] != config["epoch"]
                or managed["job_uuid"] != config["job_uuid"]
                or managed["runtime_library"] != config["runtime_library"]
            ):
                raise RankError("managed simulator session identity mismatch")
            actual_imports = _actual_imports()
            if actual_imports != config["expected_imports"]:
                raise RankError("actual imported sources differ from runner preflight")
            installed_version = __import__(
                "importlib.metadata", fromlist=["version"]
            ).version("vllm")
            if installed_version != "0.0.dev0+g8d9b52f7c2":
                raise RankError("actual imported vLLM version differs from the pin")
            adapter = {
                "schema": ADAPTER_SCHEMA,
                "rank": rank,
                "world_size": world,
                "entrypoint": "vllm.distributed.parallel_state.GroupCoordinator.all_reduce",
                "coordinator_class": (
                    f"{type(coordinator).__module__}.{type(coordinator).__qualname__}"
                ),
                "communicator_class": (
                    f"{type(communicator).__module__}.{type(communicator).__qualname__}"
                ),
                "platform_class": (
                    f"{type(current_platform).__module__}.{type(current_platform).__qualname__}"
                ),
                "unique_name": coordinator.unique_name,
                "control_backend": "gloo",
                "control_process_groups": {
                    "default": "gloo",
                    "device_group": "gloo",
                    "cpu_group": "gloo",
                },
                "tensor_data_backend": "gemsim_ccl_engine",
                "message_queue_broadcaster": False,
                "use_custom_op_call": False,
                "coordinator_methods_unmodified": coordinator_methods,
                "gloo_tensor_api_counts": dict(audit_counts),
                "gloo_tensor_api_total": sum(audit_counts.values()),
                "gloo_control_records": [],
                "capability_fd_identity": inherited_capability,
                "bootstrap_descriptor_sha256": config[
                    "bootstrap_descriptor_sha256"
                ],
                "input_sha256_before": sha256_bytes(input_before),
                "input_sha256_after": sha256_bytes(input_after),
                "output_sha256": sha256_bytes(output_payload),
                "output_storage_fresh": fresh,
                "engine_rank": communicator._engine.rank,
                "engine_world_size": communicator._engine.world_size,
                "engine_state_after_collective": communicator._engine.state.value,
                "actual_imports": actual_imports,
                "vllm_installed_version": installed_version,
                "managed_session": managed,
                "coordinator_destroyed": False,
                "default_group_destroyed": False,
                "workload_evidence": {"kind": "standalone-allreduce"},
            }
            coordinator.destroy()
            coordinator_destroyed = True
            coordinator = None
            dist.destroy_process_group()
            default_group_destroyed = True
            process_group_initialized = False
            adapter["coordinator_destroyed"] = True
            adapter["default_group_destroyed"] = True
            if any(audit_counts.values()):
                raise RankError("a Gloo tensor collective was attempted during cleanup")
        exclusive_write(Path(config["adapter_evidence_path"]), canonical_json(adapter))
        exclusive_write(Path(config["output_path"]), output_payload)
        append_event(Path(config["journal_path"]), rank, 4, "cleanup_complete")
        result = {
            "schema": RESULT_SCHEMA,
            "status": "success",
            "rank": rank,
            "world_size": world,
            "acceptance_authority": False,
            "live_adapter_accepted": False,
            "public_result_published": True,
            "input_sha256_before": sha256_bytes(input_before),
            "input_sha256_after": sha256_bytes(input_after),
            "output_sha256": sha256_bytes(output_payload),
            "output_storage_fresh": True,
            "bootstrap_descriptor_sha256": config["bootstrap_descriptor_sha256"],
            "adapter_evidence_sha256": sha256_bytes(canonical_json(adapter)),
            "managed_session": managed,
            "first_error": None,
            "product": config["product"],
        }
        exclusive_write(Path(config["result_path"]), canonical_json(result))
        return result
    except BaseException as error:
        if coordinator is not None:
            try:
                coordinator.destroy()
                coordinator_destroyed = True
            except Exception:
                pass
        if process_group_initialized and dist.is_initialized():
            try:
                dist.destroy_process_group()
                default_group_destroyed = True
            except Exception:
                pass
        failure = {
            "schema": ADAPTER_SCHEMA,
            "rank": rank,
            "world_size": world,
            "status": "device_failure",
            "error_type": type(error).__name__,
            "error": str(error),
            "gloo_tensor_api_counts": dict(audit_counts),
            "gloo_tensor_api_total": sum(audit_counts.values()),
            "capability_fd_identity": inherited_capability,
            "bootstrap_descriptor_sha256": config["bootstrap_descriptor_sha256"],
            "coordinator_destroyed": coordinator_destroyed,
            "default_group_destroyed": default_group_destroyed,
        }
        evidence_path = Path(config["adapter_evidence_path"])
        if not evidence_path.exists():
            exclusive_write(evidence_path, canonical_json(failure))
        result = {
            "schema": RESULT_SCHEMA,
            "status": "device_failure",
            "rank": rank,
            "world_size": world,
            "acceptance_authority": False,
            "live_adapter_accepted": False,
            "public_result_published": False,
            "input_sha256_before": sha256_bytes(input_before),
            "input_sha256_after": sha256_bytes(tensor_bytes(input_tensor)),
            "output_sha256": None,
            "output_storage_fresh": None,
            "bootstrap_descriptor_sha256": config["bootstrap_descriptor_sha256"],
            "adapter_evidence_sha256": sha256_bytes(evidence_path.read_bytes()),
            "first_error": {"type": type(error).__name__, "message": str(error)},
            "product": config["product"],
        }
        result_path = Path(config["result_path"])
        if not result_path.exists():
            exclusive_write(result_path, canonical_json(result))
        output_path = Path(config["output_path"])
        if not output_path.exists():
            exclusive_write(output_path, b"")
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--capability-fd", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_rank(args.config, args.capability_fd)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RankError, OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
