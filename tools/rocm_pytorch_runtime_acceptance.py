#!/usr/bin/env python3
"""Verify and atomically publish the ordinary upstream ROCm PyTorch gate."""

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
import tempfile
from typing import Any, Callable, Mapping

import sys


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
SCRIPTS = ROOT / "scripts"
for directory in (TOOLS, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import upstream_rocr_aql_acceptance as base  # noqa: E402


THIS_FILE = Path(__file__).resolve()
RUNNER = ROOT / "scripts/run_rocm_pytorch_runtime.py"
LIFECYCLE = ROOT / "scripts/gemsim_single_rank_lifecycle.py"
QUICKSTART = ROOT / "examples/quickstart/torch_rocm.py"
PRODUCT_TOOL = ROOT / "tools/rocm_pytorch_product_environment.py"
ACTIVE_PRODUCT = ROOT / "env/conda/active-rocm-pytorch"
GEM5_REPOSITORY = ROOT / "projects/gem5"
RUNTIME_REPOSITORY = ROOT / "projects/self-amdgpu-runtime"
GEM5_BINARY = GEM5_REPOSITORY / "build/VEGA_X86/gem5.opt"
GEM5_CONFIG = GEM5_REPOSITORY / "configs/example/gemsim/host_dispatch.py"

RUN_SCHEMA = "amdgpu-sim.rocm-pytorch-runtime-run.v1"
RESULT_SCHEMA = "amdgpu-sim.rocm-pytorch-runtime-acceptance.v1"
MANIFEST_SCHEMA = "amdgpu-sim.rocm-pytorch-runtime-evidence-manifest.v1"
QUICKSTART_SCHEMA = "amdgpu-sim.upstream-rocm-pytorch-quickstart.v1"
TRACE_SCHEMA = "amdgpu-sim.native-kernel-execution-trace.v1"
CLAIM_SCOPE = "ordinary_upstream_rocm_pytorch_eager_multiop_on_runtime_gem5"
SOURCE_ARTIFACTS = (
    "worker.log",
    "gem5.log",
    "dispatch-trace.jsonl",
    "m5out/stats.txt",
    "m5out/config.ini",
    "m5out/config.json",
)
UUID_RE = re.compile(r"^[0-9a-f]{32}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_TENSOR_SHA256 = {
    "copy": "8b831d777e8026aef565fc02e2f65c588613538bee9a738e9b7cc6b25945d6ed",
    "add": "71ed4d77ec47881cee8c86cec80ce7a2b426afeadd17c6bafee9b7952599b2fc",
    "sigmoid": "d81957eb31a34479cdc5437f4279dfd060dd302fbf00dc4aa204f71dc841d13d",
    "sum": "053ac09e0109c6d84088a0106d301d00ebe129455cb7889c911000d6f462271f",
}
EXPECTED_QUEUE_INDEX = [0, 1, 2, 3, 5, 6, 7, 8]
EXPECTED_KERNARG_SIZE = [48, 32, 24, 976, 48, 48, 48, 48]
EXPECTED_GRID = [
    [512, 1, 1],
    [256, 1, 1],
    [256, 1, 1],
    [16, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
]
EXPECTED_WORKGROUP = [
    [512, 1, 1],
    [256, 1, 1],
    [256, 1, 1],
    [16, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
]


class AcceptanceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def canonical_json(value: object) -> bytes:
    return base.canonical_json(value)


def _load_product_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_rocm_pytorch_product", PRODUCT_TOOL
    )
    require(spec is not None and spec.loader is not None, "product tool is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bound_file(path: Path, *, executable: bool = False) -> dict[str, Any]:
    require(path.is_absolute(), f"bound file path is not absolute: {path}")
    metadata = path.lstat()
    require(
        stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode),
        f"bound path is not a file: {path}",
    )
    require(metadata.st_uid == os.getuid(), f"bound file has the wrong owner: {path}")
    resolved = path.resolve(strict=True)
    record = base.file_record(resolved, executable=executable)
    result = {
        "path": str(path),
        "resolved_path": str(resolved),
        "bytes": record["bytes"],
        "sha256": record["sha256"],
    }
    if path.is_symlink():
        result["symlink_target"] = os.readlink(path)
    return result


def active_product() -> dict[str, Any]:
    module = _load_product_module()
    try:
        active, active_payload = module.read_canonical(ACTIVE_PRODUCT)
        require(active.get("schema") == module.ACTIVE_SCHEMA, "active product schema differs")
        prefix = Path(active.get("prefix", ""))
        require(prefix.is_absolute(), "active product prefix is not absolute")
        require(prefix.parent == (ROOT / "env/conda").resolve(strict=True), "active product prefix is outside env/conda")
        manifest_path = prefix / "amdgpu-sim-rocm-pytorch-manifest.json"
        manifest, manifest_payload = module.read_canonical(manifest_path)
        require(manifest.get("schema") == module.SCHEMA, "product manifest schema differs")
        require(manifest.get("prefix") == str(prefix), "product manifest prefix differs")
        require(manifest.get("product_id") == active.get("product_id"), "product ID differs")
        require(
            module.sha256_bytes(manifest_payload) == active.get("manifest_sha256"),
            "active product manifest SHA differs",
        )
        profile = active.get("profile")
        require(isinstance(profile, str), "active product profile is missing")
        current_identity, native_manifest = module.identity(ROOT, profile)
        require(manifest.get("identity") == current_identity, "product build identity drifted")
        require(module.product_id(current_identity) == active["product_id"], "product identity hash differs")
        provider = current_identity["rocm_provider"]
        sdk_libraries = module.provider_library_records(
            prefix, Path(manifest["sdk_root"]), provider
        )
        require(sdk_libraries == manifest.get("sdk_libraries"), "official SDK library identity drifted")
        native_prefix = Path(current_identity["native_product"]["prefix"])
        native_manifest_path = native_prefix / "manifest.json"
        loaded_native, native_payload = module.read_canonical(native_manifest_path)
        require(loaded_native == native_manifest, "native product manifest differs")
        require(
            module.sha256_bytes(native_payload)
            == current_identity["native_product"]["manifest_sha256"],
            "native product manifest SHA differs",
        )
    except (OSError, KeyError, ValueError, module.ProductError) as error:
        raise AcceptanceError(f"ROCm PyTorch product is invalid: {error}") from error
    return {
        "module": module,
        "active": active,
        "active_payload": active_payload,
        "prefix": prefix,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_payload": manifest_payload,
        "native_manifest": native_manifest,
        "native_manifest_path": native_manifest_path,
        "native_manifest_payload": native_payload,
        "sdk_libraries": sdk_libraries,
    }


def identity_snapshot() -> dict[str, Any]:
    product = active_product()
    manifest = product["manifest"]
    native = product["native_manifest"]
    prefix = product["prefix"]
    entry = manifest["entry"]
    runtime_probe = manifest["runtime_probe"]
    torch_root = Path(runtime_probe["torch_file"]).parent
    torch_lib = torch_root / "lib"
    files = {
        "verifier": _bound_file(THIS_FILE),
        "runner": _bound_file(RUNNER),
        "lifecycle": _bound_file(LIFECYCLE),
        "quickstart": _bound_file(QUICKSTART),
        "product_tool": _bound_file(PRODUCT_TOOL),
        "active_product": _bound_file(ACTIVE_PRODUCT),
        "product_manifest": _bound_file(product["manifest_path"]),
        "native_manifest": _bound_file(product["native_manifest_path"]),
        "activation": _bound_file(Path(entry["activate"])),
        "python": _bound_file(Path(entry["python"]), executable=True),
        "torch_init": _bound_file(Path(runtime_probe["torch_file"])),
        "triton_init": _bound_file(Path(runtime_probe["triton_file"])),
        "libtorch_hip": _bound_file(torch_lib / "libtorch_hip.so"),
        "libc10_hip": _bound_file(torch_lib / "libc10_hip.so"),
        "libtorch_python": _bound_file(torch_lib / "libtorch_python.so"),
        "gem5_binary": _bound_file(GEM5_BINARY, executable=True),
        "gem5_config": _bound_file(GEM5_CONFIG),
    }
    for role, descriptor in product["sdk_libraries"].items():
        files[role] = _bound_file(Path(descriptor["path"]))
    for role in (
        "runtime_library",
        "rocr_library",
        "hsakmt_model_library",
        "topology_manifest",
    ):
        files[role] = _bound_file(Path(native["artifacts"][role]["path"]))
    topology_root = Path(native["artifacts"]["topology_manifest"]["path"]).parent
    return {
        "product": {
            "schema": manifest["schema"],
            "product_id": manifest["product_id"],
            "prefix": str(prefix),
            "manifest_sha256": hashlib.sha256(product["manifest_payload"]).hexdigest(),
            "installed_tree": manifest["installed_tree"],
            "runtime_probe": runtime_probe,
            "sdk_libraries": product["sdk_libraries"],
            "native_product_id": native["product_id"],
            "native_prefix": native["prefix"],
            "native_role": manifest["identity"]["native_product"]["role"],
        },
        "files": files,
        "topology": base.directory_record(topology_root),
        "repositories": {
            "gem5": base.source_set_summary(GEM5_REPOSITORY),
            "self_runtime": base.source_set_summary(RUNTIME_REPOSITORY),
            "upstream": manifest["identity"]["upstream"],
        },
    }


def read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        return base.read_json(path, label)
    except base.AcceptanceError as error:
        raise AcceptanceError(str(error)) from error


def validate_quickstart_source(path: Path) -> dict[str, Any]:
    payload = path.read_text(encoding="ascii")
    tree = ast.parse(payload, filename=str(path))
    imported: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            imported.add(node.module or "")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                calls.add(f"{node.func.value.id}.{node.func.attr}")
    require(imported == {"hashlib", "json", "torch"}, "quickstart imports are not upstream-only")
    require(
        {"torch.add", "torch.sigmoid", "torch.sum"}.issubset(calls),
        "quickstart lacks the fixed ordinary PyTorch operations",
    )
    for forbidden in ("torch.ops", "gemsim", "self_amdgpu", "subprocess", "ctypes"):
        require(forbidden not in payload, f"quickstart contains a forbidden hook: {forbidden}")
    return {"upstream_only_imports": sorted(imported), "ordinary_torch_calls": sorted(calls)}


def validate_artifacts(
    source: Path, manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), "run artifacts are missing")
    require(set(artifacts) == set(SOURCE_ARTIFACTS), "run artifact set differs")
    observed: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_ARTIFACTS:
        expected = artifacts.get(relative)
        require(isinstance(expected, dict), f"artifact record is invalid: {relative}")
        actual = base.artifact_record(source, relative)
        require(actual == expected, f"artifact content drifted: {relative}")
        observed[relative] = actual
    require(
        {entry.name for entry in source.iterdir()}
        == {"result-manifest.json", "worker.log", "gem5.log", "dispatch-trace.jsonl", "m5out"},
        "source root file set differs",
    )
    require(
        {entry.name for entry in (source / "m5out").iterdir()}
        == {"stats.txt", "config.ini", "config.json"},
        "m5out file set differs",
    )
    return observed


def validate_worker(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    payload, raw = read_json(path, "PyTorch worker result")
    require(raw == canonical_json(payload), "PyTorch worker result is not canonical JSON")
    expected_keys = {
        "schema",
        "torch",
        "torch_hip",
        "device_count",
        "device_name",
        "capability",
        "operations",
        "tensor_contract",
        "checks",
        "correct",
    }
    require(set(payload) == expected_keys, "PyTorch worker result keys differ")
    runtime = identity["product"]["runtime_probe"]
    require(payload["schema"] == QUICKSTART_SCHEMA, "PyTorch quickstart schema differs")
    require(payload["torch"] == runtime["torch"], "PyTorch version differs")
    require(payload["torch_hip"] == runtime["torch_hip"], "PyTorch HIP version differs")
    require(payload["device_count"] == 1, "PyTorch device count differs")
    require(payload["device_name"] == "AMD Instinct MI350X", "PyTorch device name differs")
    require(payload["capability"] == [9, 5], "PyTorch capability differs")
    require(payload["operations"] == ["copy", "add", "sigmoid", "sum"], "PyTorch operation sequence differs")
    tensor = payload["tensor_contract"]
    require(
        isinstance(tensor, dict)
        and set(tensor)
        == {"dtype", "input_shape", "input_device", "actual_sha256", "expected_sha256"},
        "PyTorch tensor contract differs",
    )
    require(tensor["dtype"] == "float32" and tensor["input_shape"] == [16], "PyTorch input contract differs")
    require(tensor["input_device"] == "cuda:0", "PyTorch input device differs")
    require(tensor["expected_sha256"] == EXPECTED_TENSOR_SHA256, "PyTorch expected tensor identity differs")
    require(tensor["actual_sha256"] == EXPECTED_TENSOR_SHA256, "PyTorch actual tensor identity differs")
    expected_checks = {
        "copy_bitwise": True,
        "add_bitwise": True,
        "sigmoid_bitwise": True,
        "sum_bitwise": True,
        "input_unchanged": True,
        "outputs_are_cuda": True,
        "outputs_fresh": True,
        "outputs_nonalias": True,
    }
    require(payload["checks"] == expected_checks, "PyTorch correctness checks differ")
    require(payload["correct"] is True, "PyTorch worker output is incorrect")
    return {
        "torch": payload["torch"],
        "torch_hip": payload["torch_hip"],
        "device": payload["device_name"],
        "capability": payload["capability"],
        "operations": payload["operations"],
        "tensor_sha256": EXPECTED_TENSOR_SHA256,
        "input_unchanged": True,
        "outputs_fresh_nonalias": True,
        "bitwise_correct": True,
    }


def parse_trace(path: Path, execution: Mapping[str, Any]) -> dict[str, Any]:
    lines = base._read_text(path, "PyTorch dispatch trace").splitlines()
    require(len(lines) == 9, "PyTorch dispatch trace record count differs")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AcceptanceError(f"trace line {index} is invalid JSON") from error
        require(isinstance(value, dict), f"trace line {index} is not an object")
        records.append(value)
    retired = [value for value in records if value.get("event") == "native_execution_retired"]
    terminal = [value for value in records if value.get("event") == "native_execution_session_complete"]
    require(len(retired) == 8 and len(terminal) == 1, "PyTorch trace event counts differ")
    job_uuid = execution.get("job_uuid")
    require(isinstance(job_uuid, str) and UUID_RE.fullmatch(job_uuid), "execution job UUID is invalid")
    identity_fields = (
        "daemon_uuid",
        "job_uuid",
        "epoch",
        "rank",
        "world_size",
        "connection_id",
        "owner_fd",
        "owner_generation",
    )
    first = records[0]
    for record in records:
        require(record.get("schema") == TRACE_SCHEMA, "PyTorch trace schema differs")
        require(record.get("source") == "upstream_rocr_kmt_aql", "PyTorch trace source differs")
        require(record.get("job_uuid") == job_uuid, "PyTorch trace job UUID differs")
        require(record.get("epoch") == 1 and record.get("rank") == 0 and record.get("world_size") == 1, "PyTorch trace topology differs")
        require(isinstance(record.get("daemon_uuid"), str) and UUID_RE.fullmatch(record["daemon_uuid"]), "PyTorch daemon UUID is invalid")
        require(isinstance(record.get("connection_id"), int) and record["connection_id"] > 0, "PyTorch connection identity is invalid")
        require(isinstance(record.get("owner_fd"), int) and record["owner_fd"] >= 0, "PyTorch owner FD is invalid")
        require(isinstance(record.get("owner_generation"), int) and record["owner_generation"] > 0, "PyTorch owner generation is invalid")
        require(all(record.get(field) == first.get(field) for field in identity_fields), "PyTorch trace identity drifted")
        require(record.get("kernel_executed") is True, "PyTorch trace did not execute on device")
    require([value.get("execution_ticket") for value in retired] == list(range(1, 9)), "PyTorch execution ticket order differs")
    require([value.get("queue_index") for value in retired] == EXPECTED_QUEUE_INDEX, "PyTorch queue index order differs")
    require([value.get("descriptor_abi") for value in retired] == [3] * 8, "PyTorch descriptor ABI differs")
    require([value.get("kernarg_size") for value in retired] == EXPECTED_KERNARG_SIZE, "PyTorch kernarg sequence differs")
    require([value.get("grid") for value in retired] == EXPECTED_GRID, "PyTorch grid sequence differs")
    require([value.get("workgroup") for value in retired] == EXPECTED_WORKGROUP, "PyTorch workgroup sequence differs")
    expected_signals = [(1, 0), (0, 0), (0, 0), (0, 0)] + [(1, 0)] * 4
    require([(value.get("signal_before"), value.get("signal_after")) for value in retired] == expected_signals, "PyTorch signal sequence differs")
    request_ids: set[int] = set()
    previous_retire = 0
    queue_ids: set[int] = set()
    for record in retired:
        require(record.get("dispatch_id") == 32, "PyTorch dispatch ID differs")
        require(record.get("packet_fetches") == 1, "PyTorch packet fetch count differs")
        require(record.get("command_processor_submissions") == 1, "PyTorch CP submission count differs")
        require(record.get("dispatcher_starts") == 1, "PyTorch dispatcher start count differs")
        require(record.get("workgroups_completed") == 1, "PyTorch workgroup completion count differs")
        require(record.get("doorbell_ack_durable") is True, "PyTorch doorbell ACK is not durable")
        require(record.get("queue_retired") is True and record.get("pins_released") is True, "PyTorch queue/pin retirement differs")
        require(record.get("cleanup_complete") is False, "PyTorch per-dispatch cleanup scope differs")
        request_id = record.get("source_request_id")
        require(isinstance(request_id, int) and request_id > 0 and request_id not in request_ids, "PyTorch request identity differs")
        request_ids.add(request_id)
        queue_id = record.get("queue_object_id")
        require(isinstance(queue_id, int) and queue_id > 0, "PyTorch queue identity is invalid")
        queue_ids.add(queue_id)
        start, end, retire, sim_tick = (
            record.get("start_tick"),
            record.get("end_tick"),
            record.get("retire_tick"),
            record.get("sim_tick"),
        )
        require(all(isinstance(value, int) and value > 0 for value in (start, end, retire, sim_tick)), "PyTorch trace ticks are invalid")
        require(previous_retire <= start <= end == retire == sim_tick, "PyTorch trace tick order differs")
        previous_retire = retire
    require(len(queue_ids) == 1, "PyTorch quickstart used more than one queue")
    final = terminal[0]
    require(final.get("retired_dispatches") == 8, "PyTorch terminal retirement count differs")
    require(final.get("owner_disconnected") is True and final.get("cleanup_complete") is True, "PyTorch terminal cleanup differs")
    require(final.get("sim_tick", 0) >= previous_retire, "PyTorch terminal tick precedes retirement")
    return {
        "record_count": 9,
        "retired_dispatches": 8,
        "session_complete": 1,
        "source": "upstream_rocr_kmt_aql",
        "descriptor_abi": 3,
        "execution_tickets": list(range(1, 9)),
        "queue_indices": EXPECTED_QUEUE_INDEX,
        "terminal_tick": final["sim_tick"],
        "daemon_uuid": final["daemon_uuid"],
        "connection_id": final["connection_id"],
        "host_fallback_count": 0,
        "all_invariants_correct": True,
    }


def validate_stats(path: Path, terminal_tick: int) -> dict[str, Any]:
    text = base._read_text(path, "PyTorch gem5 stats")
    values: dict[str, str] = {}
    required = {
        "simTicks",
        "finalTick",
        "hostSeconds",
        "system.host_gpu_bridge.host_fallback_count",
    }
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in required:
            values[fields[0]] = fields[1]
    require(set(values) == required, "required PyTorch gem5 stats are missing")
    sim_ticks = int(values["simTicks"])
    final_tick = int(values["finalTick"])
    host_seconds = float(values["hostSeconds"])
    fallback = int(values["system.host_gpu_bridge.host_fallback_count"])
    require(sim_ticks == final_tick == terminal_tick, "PyTorch gem5 terminal tick differs")
    require(math.isfinite(host_seconds) and host_seconds > 0.0, "PyTorch gem5 hostSeconds is invalid")
    require(fallback == 0, "PyTorch gem5 host fallback is nonzero")
    return {"sim_ticks": sim_ticks, "host_seconds": host_seconds, "host_fallback_count": 0, "correct": True}


def validate_gem5_log(
    path: Path, execution: Mapping[str, Any], trace: Mapping[str, Any]
) -> dict[str, Any]:
    text = base._read_text(path, "PyTorch gem5 log")
    require("panic:" not in text and "fatal:" not in text, "PyTorch gem5 log contains a fatal failure")
    require(text.count("host-gpu-handshake status=OK") == 1, "PyTorch gem5 handshake count differs")
    match = re.search(
        r"host-gpu-dispatch-exit cause=host GPU dispatch session complete code=0 tick=([0-9]+)",
        text,
    )
    require(match is not None and int(match.group(1)) == trace["terminal_tick"], "PyTorch clean exit tick differs")
    require(f"job_uuid={execution['job_uuid']}" in text, "PyTorch gem5 job UUID differs")
    require(f"daemon_uuid={trace['daemon_uuid']}" in text, "PyTorch gem5 daemon UUID differs")
    require(str(GEM5_BINARY.resolve()) in text and str(GEM5_CONFIG.resolve()) in text, "PyTorch gem5 command paths differ")
    gem5_pid = execution.get("gem5_pid")
    require(isinstance(gem5_pid, int) and gem5_pid > 1, "PyTorch gem5 PID is invalid")
    require(re.search(rf"gem5 executing on .+, pid {gem5_pid}(?:\n|$)", text) is not None, "PyTorch gem5 log PID differs")
    command = "command line: " + " ".join(execution["gem5_argv"])
    require(command in text, "PyTorch gem5 argv differs")
    return {"handshake_ok": True, "clean_exit": True, "exit_tick": trace["terminal_tick"]}


def validate_execution(
    execution: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    required_execution = {
        "job_uuid", "epoch", "rank", "world_size", "execution_root", "endpoint",
        "trace_path", "m5out_path", "gem5_argv", "worker_argv", "gem5_environment",
        "worker_environment", "gem5_pid", "gem5_start_time_ticks", "gem5_process_group",
        "worker_pid", "worker_start_time_ticks", "worker_process_group", "worker_exit_code",
        "gem5_exit_code", "worker_timeout_seconds", "gem5_exit_timeout_seconds",
        "startup_timeout_seconds",
    }
    require(set(execution) == required_execution, "PyTorch execution keys differ")
    require(execution["epoch"] == 1 and execution["rank"] == 0 and execution["world_size"] == 1, "PyTorch execution topology differs")
    require(execution["worker_exit_code"] == 0 and execution["gem5_exit_code"] == 0, "PyTorch process exit differs")
    for role in ("gem5", "worker"):
        pid = execution[f"{role}_pid"]
        require(isinstance(pid, int) and pid > 1, f"PyTorch {role} PID is invalid")
        require(execution[f"{role}_process_group"] == pid, f"PyTorch {role} process group differs")
        require(isinstance(execution[f"{role}_start_time_ticks"], int) and execution[f"{role}_start_time_ticks"] > 0, f"PyTorch {role} start time is invalid")
    root = Path(execution["execution_root"])
    require(root.parent == Path("/tmp") and root.name.startswith("gs-rocm-pytorch-"), "PyTorch execution root differs")
    require(Path(execution["endpoint"]) == root / "bridge.sock", "PyTorch endpoint differs")
    require(Path(execution["trace_path"]) == root / "dispatch-trace.jsonl", "PyTorch trace path differs")
    require(Path(execution["m5out_path"]) == root / "m5out", "PyTorch m5out path differs")
    worker_argv = execution["worker_argv"]
    require(
        worker_argv
        == [
            "/bin/bash", "--noprofile", "--norc", "-c",
            'set -eu; source "$1"; shift; exec "$@"',
            "rocm-pytorch-worker", identity["files"]["activation"]["path"],
            identity["files"]["python"]["path"], identity["files"]["quickstart"]["path"],
        ],
        "PyTorch worker argv differs",
    )
    worker_environment = execution["worker_environment"]
    require(worker_environment.get("SAGR_GENERIC_BRIDGE_ENDPOINT") == execution["endpoint"], "PyTorch worker endpoint differs")
    require(worker_environment.get("PYTHONDONTWRITEBYTECODE") == "1" and worker_environment.get("PYTHONNOUSERSITE") == "1", "PyTorch worker isolation differs")
    for forbidden in ("CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX", "PYTHONPATH"):
        require(forbidden not in worker_environment, f"PyTorch worker inherited {forbidden}")
    cleanup_expected = {
        "worker_reaped": True,
        "gem5_reaped": True,
        "worker_process_group_absent": True,
        "gem5_process_group_absent": True,
        "endpoint_absent": True,
        "worker_forced_termination": False,
        "gem5_forced_termination": False,
        "all_clear": True,
    }
    require(cleanup == cleanup_expected, "PyTorch process cleanup differs")


def validate_source(
    source: Path,
    *,
    snapshot: Callable[[], dict[str, Any]] = identity_snapshot,
) -> dict[str, Any]:
    source = source.resolve(strict=True)
    manifest, manifest_payload = read_json(source / "result-manifest.json", "PyTorch run manifest")
    require(manifest.get("schema") == RUN_SCHEMA, "PyTorch run schema differs")
    require(manifest.get("status") == "success", "PyTorch run did not succeed")
    require(manifest.get("claim_scope") == CLAIM_SCOPE, "PyTorch claim scope differs")
    require(manifest.get("ordinary_upstream_pytorch_eager_executed") is True, "ordinary PyTorch path did not execute")
    require(manifest.get("runtime_gem5_bridge_modified_for_profile") is False, "run claims a profile-specific bridge change")
    for name in (
        "pytorch_rocm_multiop_accepted", "triton_upstream_amd_accepted",
        "torch_compile_accepted", "vllm_accepted", "sglang_accepted", "model_accepted",
    ):
        require(manifest.get(name) is False, f"PyTorch run overclaims {name}")
    identity = manifest.get("identity_preflight")
    require(isinstance(identity, dict) and identity == manifest.get("identity_postflight"), "PyTorch run identity drifted")
    require(snapshot() == identity, "live PyTorch execution identity differs")
    require(identity["product"]["native_role"] == "rocr_kmd_boundary_only", "PyTorch product boundary role differs")
    for role in ("hip_library", "comgr_library", "rccl_library"):
        provider = identity["product"]["sdk_libraries"][role].get("provider")
        require(
            provider in {"official_rocm_sdk", "official_rocm_apt_sysroot"},
            f"{role} is not an approved official ROCm provider",
        )
    validate_quickstart_source(Path(identity["files"]["quickstart"]["path"]))
    execution = manifest.get("execution")
    cleanup = manifest.get("cleanup")
    require(isinstance(execution, dict) and isinstance(cleanup, dict), "PyTorch execution metadata is invalid")
    validate_execution(execution, cleanup, identity)
    artifacts = validate_artifacts(source, manifest)
    worker = validate_worker(source / "worker.log", identity)
    trace = parse_trace(source / "dispatch-trace.jsonl", execution)
    stats = validate_stats(source / "m5out/stats.txt", trace["terminal_tick"])
    gem5_log = validate_gem5_log(source / "gem5.log", execution, trace)
    return {
        "manifest_bytes": len(manifest_payload),
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "identity": identity,
        "artifacts": artifacts,
        "worker": worker,
        "trace": trace,
        "stats": stats,
        "gem5_log": gem5_log,
        "output_correct": True,
    }


def publish(
    source: Path, output: Path, validation: Mapping[str, Any]
) -> dict[str, Any]:
    require(output.is_absolute() and not os.path.lexists(output), "PyTorch acceptance output must be absent and absolute")
    parent = output.parent.resolve(strict=True)
    require(output.parent == parent, "PyTorch acceptance parent contains a symlink")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        copied: dict[str, dict[str, Any]] = {}
        for relative in ("result-manifest.json",) + SOURCE_ARTIFACTS:
            record = base._copy_file(source / relative, temporary / relative)
            record["path"] = relative
            copied[relative] = record
        result = {
            "schema": RESULT_SCHEMA,
            "status": "accepted",
            "claim_scope": CLAIM_SCOPE,
            "source_manifest": {
                "bytes": validation["manifest_bytes"],
                "sha256": validation["manifest_sha256"],
            },
            "identity": validation["identity"],
            "worker": validation["worker"],
            "trace": validation["trace"],
            "stats": validation["stats"],
            "gem5_log": validation["gem5_log"],
            "ordinary_upstream_rocm_pytorch_eager_multiop_accepted": True,
            "official_rocm_hip_comgr_rccl_retained": True,
            "rocr_kmd_boundary_replaced_only": True,
            "runtime_gem5_bridge_modified_for_profile": False,
            "target_feedback_from_oracle": False,
            "triton_upstream_amd_accepted": False,
            "torch_compile_accepted": False,
            "vllm_accepted": False,
            "sglang_accepted": False,
            "model_accepted": False,
            "claim_boundary": (
                "ordinary upstream ROCm PyTorch eager CPU-to-GPU/GPU-to-CPU copy, "
                "add, sigmoid, and sum on one gfx950 simulated device with bitwise "
                "host-oracle agreement, eight retired AQL dispatches, zero host "
                "fallback, and clean session teardown; no complete PyTorch API, "
                "upstream Triton AMD, torch.compile, framework, TP, or model acceptance"
            ),
            "output_correct": True,
        }
        copied["result.json"] = base._write_bytes(
            temporary / "result.json", canonical_json(result)
        )
        evidence_manifest = {
            "schema": MANIFEST_SCHEMA,
            "artifacts": copied,
            "complete": True,
        }
        base._write_bytes(
            temporary / "manifest.json", canonical_json(evidence_manifest)
        )
        for directory in sorted(
            {path.parent for path in temporary.rglob("*") if path.is_file()},
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            descriptor = os.open(
                directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        base.rename_noreplace(temporary, output)
        temporary = None
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return result
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        validation = validate_source(arguments.source)
        result = publish(arguments.source.resolve(), arguments.output, validation)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (
        AcceptanceError,
        base.AcceptanceError,
        FileExistsError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"ROCm PyTorch acceptance failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
