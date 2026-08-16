#!/usr/bin/env python3
"""Verify and atomically publish the unchanged-upstream ROCr AQL gate."""

from __future__ import annotations

import argparse
import ctypes
import errno
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


ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
RUNNER = ROOT / "scripts/run_upstream_rocr_aql.py"
GEM5_REPOSITORY = ROOT / "projects/gem5"
RUNTIME_REPOSITORY = ROOT / "projects/self-amdgpu-runtime"
ROCM_REPOSITORY = ROOT / "projects/rocm-systems"
SOURCE_SET_TOOL = ROOT / "tools/repository_source_set.py"
GEM5_BINARY = GEM5_REPOSITORY / "build/VEGA_X86/gem5.opt"
GEM5_CONFIG = GEM5_REPOSITORY / "configs/example/gemsim/host_dispatch.py"
RUNTIME_BUILD = RUNTIME_REPOSITORY / "build/upstream-rocr-smoke-v1"
RUNTIME_LIBRARY = RUNTIME_BUILD / "libself_amdgpu_runtime.so.0.8.0"
MODEL_LIBRARY = RUNTIME_BUILD / "libself_amdgpu_hsakmt_model.so.1.1.0"
WORKER = RUNTIME_BUILD / "tests/self_amdgpu_runtime_upstream_rocr_model_test"
TOPOLOGY = RUNTIME_BUILD / (
    "tests/hsakmt-model-topology-"
    "409fbe2c0b22a3c3fbbc932dd735128097224885870ae2a84103f7648988abec"
)
ROCR_LIBRARY = ROOT / "env/rocm/hip-facade-stage-core-v1/lib/libhsa-runtime64.so.1"
KERNEL = ROCM_REPOSITORY / (
    "projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/gfx950/"
    "gpuReadWrite_kernels.hsaco"
)

RUN_SCHEMA = "amdgpu-sim.upstream-rocr-aql-run.v1"
RESULT_SCHEMA = "amdgpu-sim.upstream-rocr-aql-acceptance.v1"
TRACE_SCHEMA = "amdgpu-sim.native-kernel-execution-trace.v1"
MANIFEST_SCHEMA = "amdgpu-sim.upstream-rocr-aql-evidence-manifest.v1"
MAX_TEXT_BYTES = 64 * 1024 * 1024
AT_FDCWD = -100
RENAME_NOREPLACE = 1
UUID_RE = re.compile(r"^[0-9a-f]{32}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CLAIM_SCOPE = "unchanged_upstream_rocr_standard_aql_functional_checkpoint"
SOURCE_ARTIFACTS = (
    "worker.log",
    "gem5.log",
    "dispatch-trace.jsonl",
    "m5out/stats.txt",
    "m5out/config.ini",
    "m5out/config.json",
)
FILE_ROLES = {
    "verifier": THIS_FILE,
    "runner": RUNNER,
    "source_set_tool": SOURCE_SET_TOOL,
    "gem5_binary": GEM5_BINARY,
    "gem5_config": GEM5_CONFIG,
    "runtime_library": RUNTIME_LIBRARY,
    "model_library": MODEL_LIBRARY,
    "rocr_library": ROCR_LIBRARY,
    "worker": WORKER,
    "kernel": KERNEL,
}


class AcceptanceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path, *, executable: bool = False) -> os.stat_result:
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"not a regular file: {path}")
    require(not path.is_symlink(), f"symlink is forbidden: {path}")
    require(metadata.st_uid == os.getuid(), f"wrong file owner: {path}")
    if executable:
        require(os.access(path, os.X_OK), f"file is not executable: {path}")
    return metadata


def file_record(path: Path, *, executable: bool = False) -> dict[str, Any]:
    path = path.resolve(strict=True)
    metadata = regular_file(path, executable=executable)
    return {
        "path": str(path),
        "bytes": metadata.st_size,
        "sha256": file_sha256(path),
    }


def directory_record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    metadata = path.lstat()
    require(stat.S_ISDIR(metadata.st_mode), f"not a directory: {path}")
    require(not path.is_symlink(), f"directory symlink is forbidden: {path}")
    require(metadata.st_uid == os.getuid(), f"wrong directory owner: {path}")
    files: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*")):
        candidate_metadata = candidate.lstat()
        require(
            stat.S_ISDIR(candidate_metadata.st_mode)
            or stat.S_ISREG(candidate_metadata.st_mode),
            f"special topology entry is forbidden: {candidate}",
        )
        require(not candidate.is_symlink(), f"topology symlink is forbidden: {candidate}")
        require(candidate_metadata.st_uid == os.getuid(), f"wrong topology owner: {candidate}")
        if stat.S_ISREG(candidate_metadata.st_mode):
            files.append(
                {
                    "path": candidate.relative_to(path).as_posix(),
                    "bytes": candidate_metadata.st_size,
                    "sha256": file_sha256(candidate),
                }
            )
    core = {"path": str(path), "files": files}
    return {
        "path": str(path),
        "file_count": len(files),
        "tree_sha256": hashlib.sha256(canonical_json(core)).hexdigest(),
        "files": files,
    }


def _load_source_set_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_upstream_rocr_source_set", SOURCE_SET_TOOL
    )
    require(spec is not None and spec.loader is not None, "source-set tool is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_set_summary(repository: Path) -> dict[str, Any]:
    try:
        value = _load_source_set_module().source_set(repository)
    except Exception as error:
        raise AcceptanceError(f"could not hash repository source set: {repository}") from error
    return {
        "schema": value["schema"],
        "repository": value["repository"],
        "head": value["head"],
        "tree": value["tree"],
        "file_count": value["file_count"],
        "source_set_sha256": value["source_set_sha256"],
    }


def clean_repository_identity(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    commands = {
        "head": ["/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"],
        "tree": ["/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD^{tree}"],
        "status": ["/usr/bin/git", "-C", str(repository), "status", "--porcelain=v1", "-uno"],
    }
    outputs: dict[str, str] = {}
    for name, argv in commands.items():
        completed = subprocess.run(argv, check=True, capture_output=True, text=True)
        require(not completed.stderr, f"git {name} wrote stderr")
        outputs[name] = completed.stdout.strip()
    require(not outputs["status"], f"upstream repository is dirty: {repository}")
    return {
        "repository": str(repository),
        "head": outputs["head"],
        "tree": outputs["tree"],
        "tracked_clean": True,
    }


def identity_snapshot() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for role, configured in FILE_ROLES.items():
        files[role] = file_record(
            configured,
            executable=role in {"gem5_binary", "worker"},
        )
    return {
        "files": files,
        "topology": directory_record(TOPOLOGY),
        "repositories": {
            "gem5": source_set_summary(GEM5_REPOSITORY),
            "self_runtime": source_set_summary(RUNTIME_REPOSITORY),
            "rocm_systems": clean_repository_identity(ROCM_REPOSITORY),
        },
    }


def read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    regular_file(path)
    payload = path.read_bytes()
    require(0 < len(payload) <= MAX_TEXT_BYTES, f"{label} has invalid size")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"{label} is invalid ASCII JSON") from error
    require(isinstance(value, dict), f"{label} is not an object")
    return value, payload


def artifact_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    metadata = regular_file(path)
    return {"path": relative, "bytes": metadata.st_size, "sha256": file_sha256(path)}


def validate_artifacts(source: Path, manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), "run artifacts are missing")
    require(set(artifacts) == set(SOURCE_ARTIFACTS), "run artifact set differs")
    observed: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_ARTIFACTS:
        record = artifacts.get(relative)
        require(isinstance(record, dict), f"artifact record is invalid: {relative}")
        require(set(record) == {"path", "bytes", "sha256"}, f"artifact keys differ: {relative}")
        require(record.get("path") == relative, f"artifact path differs: {relative}")
        require(isinstance(record.get("bytes"), int) and record["bytes"] >= 0, f"artifact size invalid: {relative}")
        require(isinstance(record.get("sha256"), str) and SHA256_RE.fullmatch(record["sha256"]), f"artifact SHA invalid: {relative}")
        value = artifact_record(source, relative)
        require(value == record, f"artifact content drifted: {relative}")
        observed[relative] = value
    root_entries = {entry.name for entry in source.iterdir()}
    require(root_entries == {"result-manifest.json", "worker.log", "gem5.log", "dispatch-trace.jsonl", "m5out"}, "source root file set differs")
    require({entry.name for entry in (source / "m5out").iterdir()} == {"stats.txt", "config.ini", "config.json"}, "m5out file set differs")
    return observed


def _read_text(path: Path, label: str) -> str:
    regular_file(path)
    payload = path.read_bytes()
    require(0 < len(payload) <= MAX_TEXT_BYTES, f"{label} has invalid size")
    try:
        return payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise AcceptanceError(f"{label} is not ASCII") from error


def validate_worker_log(path: Path, execution: Mapping[str, Any]) -> dict[str, Any]:
    text = _read_text(path, "worker log")
    marker = "upstream ROCr standard AQL execution passed: elements=64 "
    require(text.count(marker) == 1, "worker success marker count differs")
    require(text.count("upstream-rocr-execution phase=verified") == 1, "worker verification phase differs")
    require(text.count("agent device=1 name=gfx950") == 1, "worker gfx950 agent differs")
    require("standard AQL dispatch did not retire" not in text, "worker reported retirement failure")
    require("execution mismatch" not in text, "worker reported numerical mismatch")
    worker_pid = execution.get("worker_pid")
    require(isinstance(worker_pid, int) and worker_pid > 1, "worker PID is invalid")
    observed_pids = {
        int(value) for value in re.findall(r"hsakmt-model pid=([0-9]+)", text)
    }
    require(observed_pids == {worker_pid}, "worker log PID differs")
    queue_pattern = re.compile(
        rf"^hsakmt-model pid={worker_pid} phase=queue-(doorbell|retired) "
        r"queue_id=([0-9]+) slot=([0-9]+) doorbell=([0-9]+) "
        r"notification=([0-9]+) completion=([0-9]+) status=(-?[0-9]+)$",
        re.MULTILINE,
    )
    queue_events = [
        (kind,) + tuple(int(value) for value in values)
        for kind, *values in queue_pattern.findall(text)
    ]
    require(
        queue_events
        == [
            ("doorbell", 1, 0, 0, 1, 0, 0),
            ("doorbell", 1, 0, 1, 2, 1, 0),
            ("retired", 1, 0, 1, 2, 1, 0),
            ("retired", 1, 0, 1, 2, 2, 0),
            ("doorbell", 2, 1, 0, 1, 0, 0),
            ("retired", 2, 1, 0, 1, 1, 0),
            ("doorbell", 1, 0, 2, 3, 2, 0),
            ("retired", 1, 0, 2, 3, 3, 0),
        ],
        "worker queue lifecycle differs",
    )
    return {
        "success_marker_count": 1,
        "queue_doorbell_observations": 4,
        "queue_retirement_observations": 4,
        "standard_control_packets_retired": 1,
        "kernel_dispatches_retired": 3,
        "numerical_oracle": "exact_int32_external_user_process",
        "correct": True,
    }


def parse_trace(path: Path, execution: Mapping[str, Any]) -> dict[str, Any]:
    text = _read_text(path, "dispatch trace")
    lines = text.splitlines()
    require(len(lines) == 4, "dispatch trace record count differs")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AcceptanceError(f"dispatch trace line {index} is invalid JSON") from error
        require(isinstance(value, dict), f"dispatch trace line {index} is not an object")
        records.append(value)
    retired = [record for record in records if record.get("event") == "native_execution_retired"]
    terminal = [record for record in records if record.get("event") == "native_execution_session_complete"]
    require(len(retired) == 3 and len(terminal) == 1, "dispatch trace event counts differ")
    job_uuid = execution.get("job_uuid")
    require(isinstance(job_uuid, str) and UUID_RE.fullmatch(job_uuid), "execution job UUID invalid")
    connection_ids: set[int] = set()
    for record in records:
        require(record.get("schema") == TRACE_SCHEMA, "dispatch trace schema differs")
        require(record.get("source") == "upstream_rocr_kmt_aql", "dispatch trace source differs")
        require(record.get("job_uuid") == job_uuid, "dispatch trace job UUID differs")
        require(record.get("epoch") == 1 and record.get("rank") == 0 and record.get("world_size") == 1, "dispatch trace topology differs")
        require(isinstance(record.get("daemon_uuid"), str) and UUID_RE.fullmatch(record["daemon_uuid"]), "dispatch trace daemon UUID invalid")
        require(isinstance(record.get("connection_id"), int) and record["connection_id"] > 0, "dispatch trace connection ID invalid")
        require(record.get("owner_fd", -1) >= 0 and record.get("owner_generation", 0) > 0, "dispatch trace owner invalid")
        require(record.get("kernel_executed") is True, "dispatch trace did not execute kernel")
        connection_ids.add(record["connection_id"])
    require(len(connection_ids) == 1, "dispatch trace connection identity differs")
    require([record.get("execution_ticket") for record in retired] == [1, 2, 3], "execution ticket order differs")
    require([record.get("descriptor_abi") for record in retired] == [2, 3, 2], "descriptor ABI sequence differs")
    require([record.get("queue_index") for record in retired] == [1, 0, 2], "queue index sequence differs")
    require([record.get("kernarg_size") for record in retired] == [0, 280, 0], "kernarg size sequence differs")
    require([record.get("grid") for record in retired] == [[65536, 1, 1], [64, 1, 1], [65536, 1, 1]], "grid sequence differs")
    require(all(record.get("workgroup") == [64, 1, 1] for record in retired), "workgroup sequence differs")
    require(retired[0].get("queue_object_id") == retired[2].get("queue_object_id") != retired[1].get("queue_object_id"), "ROCr blit/user queue identity differs")
    for record in retired:
        require(record.get("dispatch_id") == 32, "native dispatch ID differs")
        require(record.get("packet_fetches") == 1, "packet fetch count differs")
        require(record.get("command_processor_submissions") == 1, "CP submission count differs")
        require(record.get("dispatcher_starts") == 1, "dispatcher start count differs")
        require(record.get("workgroups_completed", 0) > 0, "no workgroup completed")
        require(record.get("signal_before") == 1 and record.get("signal_after") == 0, "completion signal transition differs")
        require(record.get("doorbell_ack_durable") is True, "doorbell ACK was not durable")
        require(record.get("queue_retired") is True and record.get("pins_released") is True, "queue/pin retirement differs")
        require(record.get("cleanup_complete") is False, "dispatch cleanup scope differs")
        start, end, retire = record.get("start_tick"), record.get("end_tick"), record.get("retire_tick")
        require(all(isinstance(value, int) and value > 0 for value in (start, end, retire)), "dispatch ticks invalid")
        require(start <= end <= retire, "dispatch tick order differs")
    final = terminal[0]
    require(final.get("retired_dispatches") == 3, "terminal retirement count differs")
    require(final.get("owner_disconnected") is True and final.get("cleanup_complete") is True, "terminal cleanup differs")
    require(final.get("sim_tick", 0) >= max(record["retire_tick"] for record in retired), "terminal tick precedes retirement")
    require(final.get("daemon_uuid") == retired[0].get("daemon_uuid"), "terminal daemon identity differs")
    return {
        "record_count": 4,
        "retired_dispatches": 3,
        "session_complete": 1,
        "descriptor_abi_sequence": [2, 3, 2],
        "execution_tickets": [1, 2, 3],
        "terminal_tick": final["sim_tick"],
        "daemon_uuid": final["daemon_uuid"],
        "connection_id": final["connection_id"],
        "all_invariants_correct": True,
    }


def validate_stats(path: Path, terminal_tick: int) -> dict[str, Any]:
    text = _read_text(path, "gem5 stats")
    values: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in {
            "simTicks",
            "finalTick",
            "hostSeconds",
            "system.host_gpu_bridge.host_fallback_count",
        }:
            values[fields[0]] = fields[1]
    require(set(values) == {"simTicks", "finalTick", "hostSeconds", "system.host_gpu_bridge.host_fallback_count"}, "required gem5 stats are missing")
    sim_ticks = int(values["simTicks"])
    final_tick = int(values["finalTick"])
    host_seconds = float(values["hostSeconds"])
    fallback = int(values["system.host_gpu_bridge.host_fallback_count"])
    require(sim_ticks == final_tick == terminal_tick, "gem5 terminal tick differs")
    require(math.isfinite(host_seconds) and host_seconds > 0.0, "gem5 hostSeconds invalid")
    require(fallback == 0, "gem5 host fallback is nonzero")
    return {
        "sim_ticks": sim_ticks,
        "host_seconds": host_seconds,
        "host_fallback_count": fallback,
        "correct": True,
    }


def validate_gem5_log(path: Path, execution: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    text = _read_text(path, "gem5 log")
    require("panic:" not in text and "fatal:" not in text, "gem5 log contains a fatal failure")
    require(text.count("host-gpu-handshake status=OK") == 1, "gem5 handshake count differs")
    exit_match = re.search(r"host-gpu-dispatch-exit cause=host GPU dispatch session complete code=0 tick=([0-9]+)", text)
    require(exit_match is not None, "gem5 clean exit record is missing")
    require(int(exit_match.group(1)) == trace["terminal_tick"], "gem5 log exit tick differs")
    require(f"job_uuid={execution['job_uuid']}" in text, "gem5 log job UUID differs")
    require(f"daemon_uuid={trace['daemon_uuid']}" in text, "gem5 log daemon UUID differs")
    require(str(GEM5_BINARY.resolve()) in text, "gem5 log binary path differs")
    require(str(GEM5_CONFIG.resolve()) in text, "gem5 log config path differs")
    gem5_pid = execution.get("gem5_pid")
    require(isinstance(gem5_pid, int) and gem5_pid > 1, "gem5 PID is invalid")
    require(
        re.search(rf"gem5 executing on .+, pid {gem5_pid}(?:\n|$)", text) is not None,
        "gem5 log PID differs",
    )
    argv = execution.get("gem5_argv")
    require(isinstance(argv, list) and all(isinstance(value, str) for value in argv), "gem5 argv is invalid")
    require("command line: " + " ".join(argv) in text, "gem5 command line differs")
    return {"handshake_ok": True, "clean_exit": True, "exit_tick": trace["terminal_tick"]}


def _proc_start_time(pid: int) -> int | None:
    try:
        payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return None
    closing = payload.rfind(")")
    require(closing >= 0, f"process stat is malformed: {pid}")
    fields = payload[closing + 2 :].split()
    require(len(fields) > 19, f"process stat is truncated: {pid}")
    return int(fields[19])


def _option(argv: list[str], name: str) -> str:
    require(argv.count(name) == 1, f"gem5 option count differs: {name}")
    index = argv.index(name)
    require(index + 1 < len(argv), f"gem5 option value is absent: {name}")
    return argv[index + 1]


def validate_execution(execution: Mapping[str, Any], cleanup: Mapping[str, Any]) -> None:
    require(set(execution) == {
        "job_uuid", "epoch", "rank", "world_size", "execution_root",
        "endpoint", "trace_path", "m5out_path", "gem5_argv", "worker_argv",
        "gem5_environment", "worker_environment", "gem5_pid",
        "gem5_start_time_ticks", "gem5_process_group", "worker_pid",
        "worker_start_time_ticks", "worker_process_group", "worker_exit_code",
        "gem5_exit_code", "worker_timeout_seconds", "gem5_exit_timeout_seconds",
    }, "execution keys differ")
    require(
        execution.get("epoch") == 1
        and execution.get("rank") == 0
        and execution.get("world_size") == 1,
        "execution topology differs",
    )
    job_uuid = execution.get("job_uuid")
    require(isinstance(job_uuid, str) and UUID_RE.fullmatch(job_uuid), "execution job UUID invalid")
    execution_root = execution.get("execution_root")
    endpoint = execution.get("endpoint")
    trace_path = execution.get("trace_path")
    m5out_path = execution.get("m5out_path")
    require(
        all(isinstance(value, str) and Path(value).is_absolute()
            for value in (execution_root, endpoint, trace_path, m5out_path)),
        "execution paths are not absolute",
    )
    root = Path(execution_root)
    require(Path(endpoint) == root / "bridge.sock", "execution endpoint differs")
    require(Path(trace_path) == root / "dispatch-trace.jsonl", "execution trace path differs")
    require(Path(m5out_path) == root / "m5out", "execution m5out path differs")
    require(len(os.fsencode(endpoint)) < 108, "execution endpoint exceeds AF_UNIX capacity")

    gem5_argv = execution.get("gem5_argv")
    worker_argv = execution.get("worker_argv")
    require(isinstance(gem5_argv, list) and all(isinstance(value, str) for value in gem5_argv), "gem5 argv invalid")
    require(isinstance(worker_argv, list) and all(isinstance(value, str) for value in worker_argv), "worker argv invalid")
    require(gem5_argv[:2] == [str(GEM5_BINARY.resolve()), "--listener-mode=on"], "gem5 argv prefix differs")
    require(_option(gem5_argv, "--outdir") == m5out_path, "gem5 outdir differs")
    require(gem5_argv[gem5_argv.index("--outdir") + 2] == str(GEM5_CONFIG.resolve()), "gem5 config argv differs")
    require(_option(gem5_argv, "--endpoint") == endpoint, "gem5 endpoint argv differs")
    require(_option(gem5_argv, "--dispatch-trace-path") == trace_path, "gem5 trace argv differs")
    require(_option(gem5_argv, "--job-uuid") == job_uuid, "gem5 job argv differs")
    require(_option(gem5_argv, "--epoch") == "1", "gem5 epoch argv differs")
    require(_option(gem5_argv, "--rank") == "0", "gem5 rank argv differs")
    require(_option(gem5_argv, "--world-size") == "1", "gem5 world argv differs")
    require(_option(gem5_argv, "--startup-timeout-ms") == "86400000", "gem5 startup timeout differs")
    require(_option(gem5_argv, "--handshake-timeout-ms") == "15000", "gem5 handshake timeout differs")
    require(_option(gem5_argv, "--run-timeout-ms") == "86400000", "gem5 run timeout differs")
    require(worker_argv == [str(WORKER.resolve()), "--execute", str(KERNEL.resolve())], "worker argv differs")

    gem5_environment = execution.get("gem5_environment")
    worker_environment = execution.get("worker_environment")
    require(isinstance(gem5_environment, dict), "gem5 environment is missing")
    require(isinstance(worker_environment, dict), "worker environment is missing")
    common_environment = {
        "HOME": str(root / "home"),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TMPDIR": str(root / "tmp"),
        "XDG_CACHE_HOME": str(root / "xdg-cache"),
        "XDG_CONFIG_HOME": str(root / "xdg-config"),
        "XDG_DATA_HOME": str(root / "xdg-data"),
    }
    require(gem5_environment == common_environment, "gem5 environment differs")
    expected_worker_environment = dict(common_environment)
    expected_worker_environment.update({
        "HSA_ENABLE_DTIF_FAST_COPY": "0",
        "HSA_ENABLE_DXG_DETECTION": "0",
        "HSA_ENABLE_INTERRUPT": "0",
        "HSA_MODEL_LIB": str(MODEL_LIBRARY.resolve()),
        "HSA_MODEL_TOPOLOGY": str(TOPOLOGY.resolve()),
        "LD_LIBRARY_PATH": f"{RUNTIME_BUILD.resolve()}:{ROCR_LIBRARY.resolve().parent}",
        "SAGR_GENERIC_BRIDGE_ENDPOINT": endpoint,
        "SAGR_HSAKMT_MODEL_TRACE": "1",
        "SAGR_UPSTREAM_ROCR_EXECUTION_TRACE": "1",
    })
    require(worker_environment == expected_worker_environment, "worker environment differs")

    for role in ("gem5", "worker"):
        pid = execution.get(f"{role}_pid")
        start_time = execution.get(f"{role}_start_time_ticks")
        process_group = execution.get(f"{role}_process_group")
        require(isinstance(pid, int) and pid > 1, f"{role} PID invalid")
        require(isinstance(start_time, int) and start_time > 0, f"{role} start time invalid")
        require(process_group == pid, f"{role} process group differs")
        require(_proc_start_time(pid) != start_time, f"{role} process is still live")
    require(execution.get("worker_exit_code") == 0 and execution.get("gem5_exit_code") == 0, "child exit code differs")
    require(
        isinstance(execution.get("worker_timeout_seconds"), int)
        and 1 <= execution["worker_timeout_seconds"] <= 600,
        "worker timeout invalid",
    )
    require(
        isinstance(execution.get("gem5_exit_timeout_seconds"), int)
        and 1 <= execution["gem5_exit_timeout_seconds"] <= 120,
        "gem5 exit timeout invalid",
    )

    require(set(cleanup) == {
        "worker_reaped", "gem5_reaped", "worker_process_group_absent",
        "gem5_process_group_absent", "endpoint_absent",
        "worker_forced_termination", "gem5_forced_termination", "all_clear",
    }, "cleanup keys differ")
    for key in (
        "worker_reaped",
        "gem5_reaped",
        "worker_process_group_absent",
        "gem5_process_group_absent",
        "endpoint_absent",
        "all_clear",
    ):
        require(cleanup.get(key) is True, f"cleanup invariant differs: {key}")
    require(
        cleanup.get("worker_forced_termination") is False
        and cleanup.get("gem5_forced_termination") is False,
        "a child required forced termination",
    )


def validate_source(
    source: Path,
    *,
    snapshot: Callable[[], dict[str, Any]] = identity_snapshot,
) -> dict[str, Any]:
    require(source.is_absolute(), "source must be absolute")
    resolved_source = source.resolve(strict=True)
    require(source == resolved_source, "source path contains a symlink")
    source = resolved_source
    metadata = source.lstat()
    require(stat.S_ISDIR(metadata.st_mode) and not source.is_symlink(), "source is not a real directory")
    manifest, manifest_bytes = read_json(source / "result-manifest.json", "run manifest")
    require(manifest.get("schema") == RUN_SCHEMA, "run schema differs")
    require(manifest.get("status") == "success", "run did not succeed")
    require(manifest.get("claim_scope") == CLAIM_SCOPE, "run claim scope differs")
    require(manifest.get("hip_runtime_accepted") is False, "run overclaims HIP")
    require(manifest.get("pytorch_rocm_accepted") is False, "run overclaims PyTorch")
    require(manifest.get("model_accepted") is False, "run overclaims model support")
    execution = manifest.get("execution")
    cleanup = manifest.get("cleanup")
    require(isinstance(execution, dict), "execution record is missing")
    require(isinstance(cleanup, dict), "cleanup record is missing")
    validate_execution(execution, cleanup)
    preflight = manifest.get("identity_preflight")
    postflight = manifest.get("identity_postflight")
    require(isinstance(preflight, dict) and preflight == postflight, "execution identity drifted during run")
    current_before = snapshot()
    require(current_before == preflight, "live execution identity differs from run")
    artifacts = validate_artifacts(source, manifest)
    worker = validate_worker_log(source / "worker.log", execution)
    trace = parse_trace(source / "dispatch-trace.jsonl", execution)
    stats = validate_stats(source / "m5out/stats.txt", trace["terminal_tick"])
    gem5_log = validate_gem5_log(source / "gem5.log", execution, trace)
    current_after = snapshot()
    require(current_after == current_before, "live execution identity drifted during verification")
    return {
        "manifest_bytes": len(manifest_bytes),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "identity": current_before,
        "artifacts": artifacts,
        "worker": worker,
        "trace": trace,
        "stats": stats,
        "gem5_log": gem5_log,
        "identity_unchanged": True,
        "correct": True,
    }


def _write_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError(errno.EIO, "short evidence write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _copy_file(source: Path, destination: Path) -> dict[str, Any]:
    regular_file(source)
    return _write_bytes(destination, source.read_bytes())


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.renameat2
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(AT_FDCWD, os.fsencode(source), AT_FDCWD, os.fsencode(destination), RENAME_NOREPLACE) != 0:
        value = ctypes.get_errno()
        if value == errno.EEXIST:
            raise FileExistsError(value, os.strerror(value), destination)
        raise OSError(value, os.strerror(value), destination)


def publish(source: Path, output: Path, validation: Mapping[str, Any]) -> dict[str, Any]:
    require(output.is_absolute(), "output must be absolute")
    require(not os.path.lexists(output), "output already exists")
    parent = output.parent.resolve(strict=True)
    require(output.parent == parent, "output parent contains a symlink")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        copied: dict[str, dict[str, Any]] = {}
        for relative in ("result-manifest.json",) + SOURCE_ARTIFACTS:
            copied[f"source/{relative}"] = _copy_file(source / relative, temporary / "source" / relative)
        result = {
            "schema": RESULT_SCHEMA,
            "status": "accepted",
            "source": str(source.resolve()),
            "source_manifest": {
                "bytes": validation["manifest_bytes"],
                "sha256": validation["manifest_sha256"],
            },
            "execution_identity": validation["identity"],
            "worker": validation["worker"],
            "trace": validation["trace"],
            "stats": validation["stats"],
            "gem5_log": validation["gem5_log"],
            "standard_rocr_aql_accepted": True,
            "hip_runtime_accepted": False,
            "pytorch_rocm_accepted": False,
            "triton_accepted": False,
            "vllm_accepted": False,
            "sglang_accepted": False,
            "model_accepted": False,
            "host_fallback_count": 0,
            "target_feedback": False,
            "claim_scope": CLAIM_SCOPE,
            "claim_boundary": "standard ROCr Model Interface, device blit and one user AQL kernel only; no HIP API, PyTorch device, Triton, framework, TP, or model acceptance",
            "output_correct": True,
        }
        copied["result.json"] = _write_bytes(temporary / "result.json", canonical_json(result))
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "artifacts": copied,
            "complete": True,
        }
        _write_bytes(temporary / "manifest.json", canonical_json(manifest))
        for directory in sorted({path.parent for path in temporary.rglob("*") if path.is_file()}, reverse=True):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        rename_noreplace(temporary, output)
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
    except (AcceptanceError, FileExistsError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"upstream ROCr AQL acceptance failed: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
