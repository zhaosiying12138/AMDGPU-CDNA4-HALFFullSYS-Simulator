#!/usr/bin/env python3
"""Verify and atomically publish the standard upstream HIP kernel gate."""

from __future__ import annotations

import argparse
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
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import upstream_rocr_aql_acceptance as base


ROOT = TOOLS.parent
THIS_FILE = Path(__file__).resolve()
RUNNER = ROOT / "scripts/run_hip_facade_runtime.py"
SMOKE = ROOT / "tools/hip_facade_runtime_smoke.py"
CONDA_TOOL = ROOT / "tools/conda_product_environment.py"
ACTIVE_CONDA = ROOT / "env/conda/active-product"
GEM5_REPOSITORY = ROOT / "projects/gem5"
RUNTIME_REPOSITORY = ROOT / "projects/self-amdgpu-runtime"
ROCM_REPOSITORY = ROOT / "projects/rocm-systems"
TRITON_REPOSITORY = ROOT / "projects/triton"
GEM5_BINARY = GEM5_REPOSITORY / "build/VEGA_X86/gem5.opt"
GEM5_CONFIG = GEM5_REPOSITORY / "configs/example/gemsim/host_dispatch.py"

RUN_SCHEMA = "amdgpu-sim.hip-facade-runtime-run.v1"
RESULT_SCHEMA = "amdgpu-sim.hip-facade-runtime-acceptance.v1"
MANIFEST_SCHEMA = "amdgpu-sim.hip-facade-runtime-evidence-manifest.v1"
SMOKE_SCHEMA = "amdgpu-sim.hip-facade-runtime-smoke.v1"
TRACE_SCHEMA = "amdgpu-sim.native-kernel-execution-trace.v1"
CLAIM_SCOPE = "standard_upstream_hip_generic_kernel_functional_checkpoint"
SOURCE_ARTIFACTS = (
    "worker.log",
    "gem5.log",
    "dispatch-trace.jsonl",
    "m5out/stats.txt",
    "m5out/config.ini",
    "m5out/config.json",
)
EXPECTED_OUTPUT_SHA256 = "6a93153f49a44c8bbfe2495c975af06c6ec3e99484e6e9cbb13940858ea48247"


class AcceptanceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def canonical_json(value: object) -> bytes:
    return base.canonical_json(value)


def _load_conda_module() -> Any:
    spec = importlib.util.spec_from_file_location("_hip_conda_product", CONDA_TOOL)
    require(spec is not None and spec.loader is not None, "conda product tool is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def active_product() -> dict[str, Any]:
    module = _load_conda_module()
    prefix = module.verify(ROOT)
    active, active_payload = module.read_canonical_json(ACTIVE_CONDA)
    manifest_path = prefix / "amdgpu-sim-manifest.json"
    manifest, manifest_payload = module.read_canonical_json(manifest_path)
    native_prefix = Path(manifest["native"]["prefix"])
    native_manifest_path = native_prefix / "manifest.json"
    native_manifest, native_manifest_payload = module.read_canonical_json(native_manifest_path)
    return {
        "prefix": prefix,
        "active": active,
        "active_payload": active_payload,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_payload": manifest_payload,
        "native_prefix": native_prefix,
        "native_manifest": native_manifest,
        "native_manifest_path": native_manifest_path,
        "native_manifest_payload": native_manifest_payload,
    }


def product_paths(product: Mapping[str, Any]) -> dict[str, Path]:
    manifest = product["manifest"]
    native = product["native_manifest"]
    prefix = Path(product["prefix"])
    return {
        "python": Path(manifest["entry"]["python"]),
        "activation": Path(manifest["entry"]["activate"]),
        "conda_manifest": Path(product["manifest_path"]),
        "native_manifest": Path(product["native_manifest_path"]),
        "hip_library": Path(native["artifacts"]["hip_library"]["path"]),
        "rocr_library": Path(native["artifacts"]["rocr_library"]["path"]),
        "model_library": Path(native["artifacts"]["hsakmt_model_library"]["path"]),
        "topology_manifest": Path(native["artifacts"]["topology_manifest"]["path"]),
        "compiler": Path(native["base"]["prefix"]) / "bin/clang++",
        "product_root": prefix,
    }


def identity_snapshot() -> dict[str, Any]:
    product = active_product()
    paths = product_paths(product)
    compiler_entry = paths["compiler"]
    compiler_resolved = compiler_entry.resolve(strict=True)
    compiler_record = base.file_record(compiler_resolved, executable=True)
    compiler_record["path"] = str(compiler_entry)
    compiler_record["resolved_path"] = str(compiler_resolved)
    files = {
        "verifier": base.file_record(THIS_FILE),
        "runner": base.file_record(RUNNER),
        "smoke": base.file_record(SMOKE),
        "conda_tool": base.file_record(CONDA_TOOL),
        "active_conda": base.file_record(ACTIVE_CONDA),
        "conda_manifest": base.file_record(paths["conda_manifest"]),
        "native_manifest": base.file_record(paths["native_manifest"]),
        "activation": base.file_record(paths["activation"]),
        "python": base.file_record(paths["python"], executable=True),
        "gem5_binary": base.file_record(GEM5_BINARY, executable=True),
        "gem5_config": base.file_record(GEM5_CONFIG),
        "hip_library": base.file_record(paths["hip_library"]),
        "rocr_library": base.file_record(paths["rocr_library"]),
        "model_library": base.file_record(paths["model_library"]),
        "topology_manifest": base.file_record(paths["topology_manifest"]),
        "compiler": compiler_record,
    }
    return {
        "product": {
            "product_id": product["manifest"]["product_id"],
            "prefix": str(product["prefix"]),
            "native_product_id": product["native_manifest"]["product_id"],
            "native_prefix": str(product["native_prefix"]),
        },
        "files": files,
        "repositories": {
            "gem5": base.source_set_summary(GEM5_REPOSITORY),
            "self_runtime": base.source_set_summary(RUNTIME_REPOSITORY),
            "rocm_systems": base.clean_repository_identity(ROCM_REPOSITORY),
            "triton": base.clean_repository_identity(TRITON_REPOSITORY),
        },
    }


def read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        return base.read_json(path, label)
    except base.AcceptanceError as error:
        raise AcceptanceError(str(error)) from error


def validate_artifacts(source: Path, manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
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
    text = base._read_text(path, "HIP worker log")
    require(len(text.splitlines()) == 1, "HIP worker log line count differs")
    try:
        result = json.loads(text)
    except json.JSONDecodeError as error:
        raise AcceptanceError("HIP worker result is invalid JSON") from error
    require(isinstance(result, dict), "HIP worker result is not an object")
    require(result.get("schema") == SMOKE_SCHEMA, "HIP worker schema differs")
    require(result.get("mode") == "vector-add", "HIP worker mode differs")
    require(result.get("device_count") == 1, "HIP device count differs")
    require(result.get("correct") is True, "HIP worker did not report success")
    require(
        result.get("path")
        == [
            "upstream_hip_api",
            "upstream_rocr",
            "upstream_hsakmt_model_interface",
            "self_runtime_provider",
            "runtime_gem5_bridge",
            "gem5_gpu_model",
        ],
        "HIP worker path differs",
    )
    require(
        result.get("fallback")
        == {"cpu": 0, "cuda": 0, "privateuse1": 0, "project_operator": 0},
        "HIP worker fallback differs",
    )
    execution = result.get("execution")
    compilation = result.get("compilation")
    hip_library = result.get("hip_library")
    require(isinstance(execution, dict), "HIP execution record is missing")
    require(isinstance(compilation, dict), "HIP compilation record is missing")
    require(isinstance(hip_library, dict), "HIP library record is missing")
    require(
        execution
        == {
            "count": 256,
            "grid": [4, 1, 1],
            "block": [64, 1, 1],
            "mismatch_count": 0,
            "output_sha256": EXPECTED_OUTPUT_SHA256,
            "correct": True,
        },
        "HIP numerical execution differs",
    )
    require(compilation.get("target") == "gfx950", "HIP compiler target differs")
    require(compilation.get("device_only") is True, "HIP compilation mode differs")
    require(compilation.get("output_format") == "elf64-amdgpu", "HIP output format differs")
    smoke_spec = importlib.util.spec_from_file_location("_hip_smoke_contract", SMOKE)
    require(smoke_spec is not None and smoke_spec.loader is not None, "HIP smoke module is unavailable")
    smoke_module = importlib.util.module_from_spec(smoke_spec)
    smoke_spec.loader.exec_module(smoke_module)
    require(
        compilation.get("source_sha256")
        == hashlib.sha256(smoke_module.KERNEL_SOURCE.encode("ascii")).hexdigest(),
        "HIP source SHA differs",
    )
    files = identity["files"]
    require(compilation.get("compiler_path") == files["compiler"]["path"], "HIP compiler path differs")
    require(compilation.get("compiler_sha256") == files["compiler"]["sha256"], "HIP compiler SHA differs")
    require(hip_library == {
        "path": files["hip_library"]["path"],
        "sha256": files["hip_library"]["sha256"],
    }, "HIP library identity differs")
    image_sha = compilation.get("image_sha256")
    require(isinstance(image_sha, str) and base.SHA256_RE.fullmatch(image_sha), "HIP image SHA invalid")
    require(isinstance(compilation.get("image_bytes"), int) and compilation["image_bytes"] > 0, "HIP image size invalid")
    return {
        "device_count": 1,
        "element_count": 256,
        "image_sha256": image_sha,
        "image_bytes": compilation["image_bytes"],
        "output_sha256": EXPECTED_OUTPUT_SHA256,
        "mismatch_count": 0,
        "fallback": result["fallback"],
        "correct": True,
    }


def parse_trace(path: Path, execution: Mapping[str, Any]) -> dict[str, Any]:
    lines = base._read_text(path, "HIP dispatch trace").splitlines()
    require(len(lines) == 2, "HIP dispatch trace record count differs")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AcceptanceError(f"HIP dispatch trace line {index} is invalid") from error
        require(isinstance(value, dict), f"HIP dispatch trace line {index} is not an object")
        records.append(value)
    retired = records[0]
    terminal = records[1]
    require(retired.get("event") == "native_execution_retired", "HIP retire record differs")
    require(terminal.get("event") == "native_execution_session_complete", "HIP terminal record differs")
    for record in records:
        require(record.get("schema") == TRACE_SCHEMA, "HIP trace schema differs")
        require(record.get("source") == "upstream_rocr_kmt_aql", "HIP trace source differs")
        require(record.get("job_uuid") == execution["job_uuid"], "HIP trace job differs")
        require(record.get("epoch") == 1 and record.get("rank") == 0 and record.get("world_size") == 1, "HIP trace topology differs")
        require(record.get("kernel_executed") is True, "HIP trace did not execute a kernel")
        require(isinstance(record.get("daemon_uuid"), str) and base.UUID_RE.fullmatch(record["daemon_uuid"]), "HIP daemon UUID invalid")
        require(isinstance(record.get("connection_id"), int) and record["connection_id"] > 0, "HIP connection ID invalid")
    require(retired.get("execution_ticket") == 1, "HIP execution ticket differs")
    require(retired.get("dispatch_id") == 32, "HIP dispatch ID differs")
    require(retired.get("descriptor_abi") == 3, "HIP descriptor ABI differs")
    require(retired.get("queue_index") == 0, "HIP queue index differs")
    require(retired.get("kernarg_size") == 288, "HIP kernarg size differs")
    require(retired.get("grid") == [256, 1, 1], "HIP trace grid differs")
    require(retired.get("workgroup") == [64, 1, 1], "HIP trace workgroup differs")
    require(retired.get("workgroups_completed") == 4, "HIP completed workgroups differ")
    for name in ("packet_fetches", "command_processor_submissions", "dispatcher_starts"):
        require(retired.get(name) == 1, f"HIP trace counter differs: {name}")
    require(retired.get("signal_before") == 0 and retired.get("signal_after") == 0, "HIP signal state differs")
    require(retired.get("doorbell_ack_durable") is True, "HIP doorbell was not durable")
    require(retired.get("queue_retired") is True and retired.get("pins_released") is True, "HIP queue resources were not retired")
    require(retired.get("cleanup_complete") is False, "HIP dispatch cleanup scope differs")
    require(terminal.get("retired_dispatches") == 1, "HIP terminal retirement count differs")
    require(terminal.get("owner_disconnected") is True and terminal.get("cleanup_complete") is True, "HIP terminal cleanup differs")
    require(terminal.get("daemon_uuid") == retired.get("daemon_uuid"), "HIP daemon identity changed")
    require(terminal.get("connection_id") == retired.get("connection_id"), "HIP connection identity changed")
    start, end, retire = retired.get("start_tick"), retired.get("end_tick"), retired.get("retire_tick")
    require(all(isinstance(value, int) and value > 0 for value in (start, end, retire)), "HIP dispatch ticks invalid")
    require(start <= end <= retire <= terminal.get("sim_tick", 0), "HIP trace tick order differs")
    return {
        "record_count": 2,
        "retired_dispatches": 1,
        "session_complete": 1,
        "execution_ticket": 1,
        "terminal_tick": terminal["sim_tick"],
        "daemon_uuid": terminal["daemon_uuid"],
        "connection_id": terminal["connection_id"],
        "all_invariants_correct": True,
    }


def validate_stats(path: Path, terminal_tick: int) -> dict[str, Any]:
    try:
        return base.validate_stats(path, terminal_tick)
    except base.AcceptanceError as error:
        raise AcceptanceError(str(error)) from error


def validate_gem5_log(path: Path, execution: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    text = base._read_text(path, "HIP gem5 log")
    require("panic:" not in text and "fatal:" not in text, "HIP gem5 log contains fatal diagnostics")
    require(text.count("host-gpu-handshake status=OK") == 1, "HIP handshake count differs")
    match = re.search(r"host-gpu-dispatch-exit cause=host GPU dispatch session complete code=0 tick=([0-9]+)", text)
    require(match is not None and int(match.group(1)) == trace["terminal_tick"], "HIP clean exit tick differs")
    require(f"job_uuid={execution['job_uuid']}" in text, "HIP gem5 job UUID differs")
    require(f"daemon_uuid={trace['daemon_uuid']}" in text, "HIP gem5 daemon UUID differs")
    require("command line: " + " ".join(execution["gem5_argv"]) in text, "HIP gem5 argv differs")
    require(re.search(rf"gem5 executing on .+, pid {execution['gem5_pid']}(?:\n|$)", text) is not None, "HIP gem5 PID differs")
    return {"handshake_ok": True, "clean_exit": True, "exit_tick": trace["terminal_tick"]}


def _option(argv: list[str], name: str) -> str:
    try:
        return base._option(argv, name)
    except base.AcceptanceError as error:
        raise AcceptanceError(str(error)) from error


def validate_execution(execution: Mapping[str, Any], cleanup: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    expected_keys = {
        "job_uuid", "epoch", "rank", "world_size", "execution_root", "endpoint",
        "trace_path", "m5out_path", "gem5_argv", "worker_argv", "gem5_environment",
        "worker_environment", "gem5_pid", "gem5_start_time_ticks", "gem5_process_group",
        "worker_pid", "worker_start_time_ticks", "worker_process_group", "worker_exit_code",
        "gem5_exit_code", "worker_timeout_seconds", "gem5_exit_timeout_seconds",
    }
    require(set(execution) == expected_keys, "HIP execution keys differ")
    require(execution.get("epoch") == 1 and execution.get("rank") == 0 and execution.get("world_size") == 1, "HIP execution topology differs")
    require(isinstance(execution.get("job_uuid"), str) and base.UUID_RE.fullmatch(execution["job_uuid"]), "HIP execution job invalid")
    root = Path(execution["execution_root"])
    require(root.is_absolute(), "HIP execution root is not absolute")
    require(Path(execution["endpoint"]) == root / "bridge.sock", "HIP endpoint differs")
    require(Path(execution["trace_path"]) == root / "dispatch-trace.jsonl", "HIP trace path differs")
    require(Path(execution["m5out_path"]) == root / "m5out", "HIP m5out differs")
    gem5_argv = execution["gem5_argv"]
    require(gem5_argv[:2] == [identity["files"]["gem5_binary"]["path"], "--listener-mode=on"], "HIP gem5 argv prefix differs")
    require(gem5_argv[gem5_argv.index("--outdir") + 2] == identity["files"]["gem5_config"]["path"], "HIP gem5 config differs")
    require(_option(gem5_argv, "--endpoint") == execution["endpoint"], "HIP gem5 endpoint differs")
    require(_option(gem5_argv, "--dispatch-trace-path") == execution["trace_path"], "HIP gem5 trace differs")
    require(_option(gem5_argv, "--job-uuid") == execution["job_uuid"], "HIP gem5 job differs")
    require(_option(gem5_argv, "--epoch") == "1" and _option(gem5_argv, "--rank") == "0" and _option(gem5_argv, "--world-size") == "1", "HIP gem5 topology argv differs")
    worker_argv = execution["worker_argv"]
    require(isinstance(worker_argv, list) and worker_argv[:4] == ["/bin/bash", "--noprofile", "--norc", "-c"], "HIP worker argv prefix differs")
    require(identity["files"]["activation"]["path"] in worker_argv, "HIP activation is not bound")
    require(identity["files"]["python"]["path"] in worker_argv, "HIP product Python is not bound")
    require(identity["files"]["smoke"]["path"] in worker_argv, "HIP smoke worker is not bound")
    require(worker_argv[-4:] == ["--mode", "vector-add", "--count", "256"], "HIP worker options differ")
    require(execution["worker_environment"].get("SAGR_GENERIC_BRIDGE_ENDPOINT") == execution["endpoint"], "HIP worker endpoint environment differs")
    for role in ("gem5", "worker"):
        pid = execution.get(f"{role}_pid")
        start = execution.get(f"{role}_start_time_ticks")
        require(isinstance(pid, int) and pid > 1, f"HIP {role} PID invalid")
        require(isinstance(start, int) and start > 0, f"HIP {role} start time invalid")
        require(execution.get(f"{role}_process_group") == pid, f"HIP {role} group differs")
        require(base._proc_start_time(pid) != start, f"HIP {role} process is still live")
    require(execution.get("worker_exit_code") == 0 and execution.get("gem5_exit_code") == 0, "HIP child exit code differs")
    required_cleanup = {
        "worker_reaped", "gem5_reaped", "worker_process_group_absent",
        "gem5_process_group_absent", "endpoint_absent", "worker_forced_termination",
        "gem5_forced_termination", "all_clear",
    }
    require(set(cleanup) == required_cleanup, "HIP cleanup keys differ")
    for key in ("worker_reaped", "gem5_reaped", "worker_process_group_absent", "gem5_process_group_absent", "endpoint_absent", "all_clear"):
        require(cleanup.get(key) is True, f"HIP cleanup invariant differs: {key}")
    require(cleanup.get("worker_forced_termination") is False and cleanup.get("gem5_forced_termination") is False, "HIP child required forced termination")


def validate_source(source: Path, *, snapshot: Callable[[], dict[str, Any]] = identity_snapshot) -> dict[str, Any]:
    require(source.is_absolute(), "HIP source must be absolute")
    source = source.resolve(strict=True)
    require(stat.S_ISDIR(source.lstat().st_mode) and not source.is_symlink(), "HIP source is not a real directory")
    manifest, manifest_payload = read_json(source / "result-manifest.json", "HIP run manifest")
    require(manifest.get("schema") == RUN_SCHEMA, "HIP run schema differs")
    require(manifest.get("status") == "success", "HIP run did not succeed")
    require(manifest.get("claim_scope") == CLAIM_SCOPE, "HIP run claim scope differs")
    require(manifest.get("standard_hip_kernel_executed") is True, "HIP run did not execute the standard kernel")
    for name in ("pytorch_rocm_accepted", "triton_upstream_hip_accepted", "framework_accepted", "model_accepted"):
        require(manifest.get(name) is False, f"HIP run overclaims {name}")
    identity = manifest.get("identity_preflight")
    require(isinstance(identity, dict) and identity == manifest.get("identity_postflight"), "HIP run identity drifted")
    current_before = snapshot()
    require(current_before == identity, "live HIP identity differs from run")
    execution = manifest.get("execution")
    cleanup = manifest.get("cleanup")
    require(isinstance(execution, dict) and isinstance(cleanup, dict), "HIP execution or cleanup is missing")
    validate_execution(execution, cleanup, identity)
    artifacts = validate_artifacts(source, manifest)
    worker = validate_worker(source / "worker.log", identity)
    trace = parse_trace(source / "dispatch-trace.jsonl", execution)
    stats = validate_stats(source / "m5out/stats.txt", trace["terminal_tick"])
    gem5_log = validate_gem5_log(source / "gem5.log", execution, trace)
    require(snapshot() == current_before, "live HIP identity drifted during verification")
    return {
        "manifest_bytes": len(manifest_payload),
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "identity": identity,
        "artifacts": artifacts,
        "worker": worker,
        "trace": trace,
        "stats": stats,
        "gem5_log": gem5_log,
        "correct": True,
    }


def publish(source: Path, output: Path, validation: Mapping[str, Any]) -> dict[str, Any]:
    require(output.is_absolute() and not os.path.lexists(output), "HIP acceptance output must be absent and absolute")
    parent = output.parent.resolve(strict=True)
    require(output.parent == parent, "HIP acceptance parent contains a symlink")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        copied: dict[str, dict[str, Any]] = {}
        for relative in ("result-manifest.json",) + SOURCE_ARTIFACTS:
            copied[f"source/{relative}"] = base._copy_file(source / relative, temporary / "source" / relative)
        result = {
            "schema": RESULT_SCHEMA,
            "status": "accepted",
            "claim_scope": CLAIM_SCOPE,
            "source": str(source.resolve()),
            "source_manifest": {"bytes": validation["manifest_bytes"], "sha256": validation["manifest_sha256"]},
            "execution_identity": validation["identity"],
            "worker": validation["worker"],
            "trace": validation["trace"],
            "stats": validation["stats"],
            "gem5_log": validation["gem5_log"],
            "standard_hip_facade_kernel_accepted": True,
            "complete_hip_runtime_accepted": False,
            "pytorch_rocm_accepted": False,
            "triton_upstream_hip_accepted": False,
            "vllm_accepted": False,
            "sglang_accepted": False,
            "model_accepted": False,
            "host_fallback_count": 0,
            "target_feedback": False,
            "claim_boundary": "one ordinary public-HIP gfx950 module kernel with allocator, copies, stream, synchronization and clean runtime-gem5 retirement; no event-completeness, PyTorch, Triton, RCCL, framework, TP, or model acceptance",
            "output_correct": True,
        }
        copied["result.json"] = base._write_bytes(temporary / "result.json", canonical_json(result))
        manifest = {"schema": MANIFEST_SCHEMA, "artifacts": copied, "complete": True}
        base._write_bytes(temporary / "manifest.json", canonical_json(manifest))
        for directory in sorted({path.parent for path in temporary.rglob("*") if path.is_file()}, reverse=True):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
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
    except (AcceptanceError, base.AcceptanceError, FileExistsError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"HIP facade acceptance failed: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
