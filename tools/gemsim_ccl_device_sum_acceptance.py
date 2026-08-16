#!/usr/bin/env python3
"""Run and atomically bind the standalone device-SUM acceptance evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
RUNNER = ROOT / "examples/triton/ccl_device_sum_correctness.py"
BOOTSTRAP = ROOT / "examples/triton/_gemsim_bootstrap.py"
SETUP = ROOT / "scripts/setup_rocm_env.sh"
EXPECTED_COUNTS = (0, 1, 3, 127, 128, 129, 255, 256, 257, 1024, 1027, 2048, 7168)
EXPECTED_PREFIX_SCHEMA = "amdgpu-sim.product-prefix.v1"
EXPECTED_RUNTIME_VERSION = "0.8.0"
EXPECTED_RUNTIME_ABI = (1 << 16) | 8
AT_FDCWD = -100
RENAME_NOREPLACE = 1


class AcceptanceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path, *, executable: bool = False) -> None:
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"not a regular file: {path}")
    require(not path.is_symlink(), f"symlink is forbidden: {path}")
    require(metadata.st_uid == os.getuid(), f"wrong file owner: {path}")
    if executable:
        require(os.access(path, os.X_OK), f"file is not executable: {path}")


def prefix_path() -> Path:
    completed = subprocess.run(
        [str(SETUP), "--print-prefix"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    require(not completed.stderr, "prefix discovery wrote stderr")
    lines = completed.stdout.splitlines()
    require(len(lines) == 1, "prefix discovery did not return one path")
    prefix = Path(lines[0])
    require(prefix.is_absolute() and prefix == Path(os.path.normpath(prefix)), "invalid prefix")
    return prefix.resolve(strict=True)


def artifact_matches(
    record: Any,
    expected: dict[str, Any],
    role: str,
) -> None:
    require(isinstance(record, dict), f"prefix manifest {role} is not an object")
    require(record.get("path") == expected["path"], f"prefix manifest {role} path drifted")
    require(record.get("sha256") == expected["sha256"], f"prefix manifest {role} SHA drifted")


def validate_prefix_manifest(identity: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path = Path(identity["prefix_manifest"]["path"])
    try:
        document = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AcceptanceError("prefix manifest is unreadable") from error
    require(isinstance(document, dict), "prefix manifest is not an object")
    require(document.get("schema") == EXPECTED_PREFIX_SCHEMA, "wrong prefix schema")
    require(document.get("prefix") == identity["prefix"]["path"], "prefix path drifted")
    require(document.get("product_id") == identity["product_id"]["sha256"], "product id drifted")
    require(document.get("setup_schema") == 1, "wrong product setup schema")
    artifacts = document.get("artifacts")
    managed = document.get("managed_inputs")
    base = document.get("base")
    plugins = document.get("plugins")
    require(isinstance(artifacts, dict), "prefix artifacts are missing")
    require(isinstance(managed, dict), "prefix managed inputs are missing")
    require(isinstance(base, dict), "prefix base binding is missing")
    require(isinstance(plugins, dict), "prefix plugin binding is missing")
    artifact_matches(
        artifacts.get("runtime_library"), identity["runtime_library"], "runtime library"
    )
    soname = artifacts.get("runtime_soname")
    require(isinstance(soname, dict), "prefix manifest runtime soname is not an object")
    expected_soname = Path(identity["prefix"]["path"]) / "lib/libself_amdgpu_runtime.so.1"
    require(soname.get("path") == str(expected_soname), "prefix runtime soname path drifted")
    require(expected_soname.resolve(strict=True) == Path(identity["runtime_library"]["path"]), "runtime soname target drifted")
    require(soname.get("sha256") == identity["runtime_library"]["sha256"], "runtime soname SHA drifted")
    artifact_matches(managed.get("gem5_binary"), identity["gem5_binary"], "gem5 binary")
    artifact_matches(managed.get("gem5_config"), identity["gem5_config"], "gem5 config")
    artifact_matches(artifacts.get("triton_plugin_driver"), identity["triton_driver"], "Triton driver")
    artifact_matches(artifacts.get("ccl_plugin_init"), identity["package_init"], "CCL plugin")
    artifact_matches(artifacts.get("python_bootstrap"), identity["product_bootstrap"], "product bootstrap")
    artifact_matches(base.get("python"), identity["private_python"], "base Python")
    snapshots = plugins.get("snapshots")
    require(isinstance(snapshots, dict), "plugin snapshots are missing")
    require(
        snapshots.get("gemsim-ccl", {}).get("package_path")
        == str(Path(identity["package_init"]["path"]).parent),
        "CCL package snapshot path drifted",
    )
    require(
        snapshots.get("triton-gemsim-amd", {}).get("package_path")
        == str(Path(identity["triton_driver"]["path"]).parent),
        "Triton package snapshot path drifted",
    )
    runtime = ctypes.CDLL(identity["runtime_library"]["path"], mode=ctypes.RTLD_LOCAL)
    runtime.sagr_abi_version.argtypes = []
    runtime.sagr_abi_version.restype = ctypes.c_uint32
    runtime.sagr_version_string.argtypes = []
    runtime.sagr_version_string.restype = ctypes.c_char_p
    abi = int(runtime.sagr_abi_version())
    raw_version = runtime.sagr_version_string()
    require(raw_version is not None, "runtime version string is absent")
    version = raw_version.decode("ascii", "strict")
    require(abi == EXPECTED_RUNTIME_ABI, "runtime ABI is not exactly 1.8")
    require(version == EXPECTED_RUNTIME_VERSION, "runtime version is not exactly 0.8.0")
    return {
        "schema": document["schema"],
        "sha256": identity["prefix_manifest"]["sha256"],
        "runtime_version": version,
        "runtime_abi": f"{abi >> 16}.{abi & 0xffff}",
        "managed_inputs_match_live_files": True,
        "complete": True,
    }


def identity_snapshot() -> dict[str, dict[str, Any]]:
    prefix = prefix_path()
    manifest_path = prefix / "manifest.json"
    regular_file(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AcceptanceError("product manifest is unreadable") from error
    require(isinstance(manifest, dict), "product manifest is not an object")
    require(manifest.get("schema") == EXPECTED_PREFIX_SCHEMA, "wrong product schema")
    artifacts = manifest.get("artifacts")
    managed = manifest.get("managed_inputs")
    base = manifest.get("base")
    require(isinstance(artifacts, dict), "product artifacts are missing")
    require(isinstance(managed, dict), "product managed inputs are missing")
    require(isinstance(base, dict) and isinstance(base.get("python"), dict), "base Python is missing")

    def recorded_path(container: dict[str, Any], name: str) -> Path:
        record = container.get(name)
        require(isinstance(record, dict) and isinstance(record.get("path"), str), f"missing path: {name}")
        return Path(record["path"])

    paths = {
        "validator": THIS_FILE,
        "runner": RUNNER,
        "bootstrap": BOOTSTRAP,
        "product_bootstrap": recorded_path(artifacts, "python_bootstrap"),
        "device_plugin": prefix / "python/gemsim_ccl/device.py",
        "package_init": recorded_path(artifacts, "ccl_plugin_init"),
        "triton_driver": recorded_path(artifacts, "triton_plugin_driver"),
        "prefix_manifest": manifest_path,
        "runtime_library": recorded_path(artifacts, "runtime_library"),
        "private_python": Path(base["python"]["path"]),
        "gem5_binary": recorded_path(managed, "gem5_binary"),
        "gem5_config": recorded_path(managed, "gem5_config"),
    }
    result: dict[str, dict[str, Any]] = {}
    for role, path in paths.items():
        regular_file(path, executable=role in {"private_python", "gem5_binary"})
        result[role] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    result["prefix"] = {
        "path": str(prefix),
        "bytes": 0,
        "sha256": result["prefix_manifest"]["sha256"],
    }
    product_id = manifest.get("product_id")
    require(isinstance(product_id, str) and re.fullmatch(r"[0-9a-f]{64}", product_id) is not None, "invalid product id")
    result["product_id"] = {"path": "", "bytes": 0, "sha256": product_id}
    return result


def strict_result(stdout: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    require(len(lines) == 1, "runner stdout is not exactly one JSON record")
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise AcceptanceError("runner stdout is invalid JSON") from error
    require(isinstance(result, dict), "runner result is not an object")
    require(result.get("schema") == "amdgpu-sim.ccl-device-sum.v1", "wrong result schema")
    require(result.get("backend") == "gemsim_amd" and result.get("arch") == "gfx950", "wrong target")
    require(result.get("output_correct") is True, "runner numerical result failed")
    require(result.get("claim_scope") == "standalone_device_sum_primitive", "wrong claim scope")
    require(result.get("planner_binding_accepted") is False, "runner overclaims planner binding")
    require(result.get("trace_evidence_bound") is False, "runner may not self-accept trace")
    require(result.get("live_collective_accepted") is False, "runner overclaims a collective")
    require(result.get("device_reduction_launch_count") == 24, "wrong self-reported launch count")
    counters = result.get("self_reported_counters")
    require(isinstance(counters, dict) and counters.get("acceptance_authority") is False, "counter scope invalid")
    require(all(counters.get(name) == 0 for name in (
        "host_reduction_count", "fallback_count", "cpu_fallback_count", "nvidia_fallback_count"
    )), "runner reported fallback or host reduction")
    cases = result.get("cases")
    require(isinstance(cases, list) and len(cases) == 2 * len(EXPECTED_COUNTS), "wrong case count")
    expected_pairs = [(dtype, count) for dtype in ("bfloat16", "float32") for count in EXPECTED_COUNTS]
    observed_pairs = [(case.get("dtype"), case.get("element_count")) for case in cases]
    require(observed_pairs == expected_pairs, "case order or coverage mismatch")
    for case in cases:
        require(case.get("output_correct") is True, "case failed")
        require(case.get("comparison", {}).get("mismatch_count") == 0, "case mismatch")
        require(case.get("comparison", {}).get("finite") is True, "case nonfinite")
        require(case.get("right_unchanged") is True and case.get("tail_unchanged") is True, "input changed")
        require(case.get("guards_unchanged") is True, "guard changed")
        require(case.get("output_aliases_destination") is True, "in-place identity failed")
        if case["element_count"] == 0:
            require(case.get("extent") == 0 and case.get("program_count") == 0, "zero chunk dispatched")
        if case["dtype"] == "bfloat16" and case["element_count"] >= 4:
            tie = case.get("bf16_tie_contract")
            require(isinstance(tie, dict) and tie.get("correct") is True, "BF16 tie contract failed")
            require(
                tie.get("actual_bits")
                == ["0x3f80", "0x3f82", "0xbf80", "0xbf82"],
                "BF16 RTNE bits differ",
            )
    negative = result.get("negative_contracts")
    require(isinstance(negative, dict) and negative and all(value is True for value in negative.values()), "negative contract failed")
    return result


def validate_session_paths(result: dict[str, Any], identity: dict[str, dict[str, Any]]) -> dict[str, Path]:
    session = result.get("managed_session")
    require(isinstance(session, dict), "managed session record missing")
    require(session.get("rank") == 0 and session.get("world_size") == 1, "primitive session topology changed")
    require(isinstance(session.get("epoch"), int) and session["epoch"] > 0, "session epoch invalid")
    for name in ("job_uuid", "daemon_uuid"):
        value = session.get(name)
        require(isinstance(value, str) and len(value) == 32 and value != "0" * 32, f"{name} invalid")
    require(session.get("gem5_path") == identity["gem5_binary"]["path"], "gem5 path mismatch")
    require(session.get("config_path") == identity["gem5_config"]["path"], "config path mismatch")
    require(session.get("runtime_library") == identity["runtime_library"]["path"], "runtime path mismatch")
    require(session.get("python_executable") == identity["private_python"]["path"], "private Python mismatch")
    require(session.get("prefix") == identity["prefix"]["path"], "prefix mismatch")
    paths = {name: Path(session[name]) for name in (
        "run_directory", "output_directory", "trace_path", "stats_path", "log_path", "triton_cache_directory"
    )}
    require(all(path.is_absolute() and path == Path(os.path.normpath(path)) for path in paths.values()), "noncanonical session path")
    run = paths["run_directory"]
    require(paths["output_directory"] == run / "m5out", "output directory escaped run")
    require(paths["trace_path"] == run / "dispatch-trace.jsonl", "trace escaped run")
    require(paths["stats_path"] == run / "m5out/stats.txt", "stats escaped run")
    require(paths["log_path"] == run / "gem5.log", "log escaped run")
    for role in ("trace_path", "stats_path", "log_path"):
        regular_file(paths[role])
    return paths


TRACE_MATCH_FIELDS = (
    "owner_generation", "request_id", "trace_id", "ticket_id", "dispatch_id",
    "kernel", "image_sha256", "grid", "workgroup", "fixed_shared_memory_bytes",
    "dynamic_shared_memory_bytes", "total_shared_memory_bytes", "kernarg_size",
    "packet_crc32c", "allocation_count", "allocations", "packet_fetches",
    "command_processor_submissions", "gpu_dispatcher_starts", "waves_started",
    "instructions_started", "instruction_wave_count", "scalar_reads", "global_reads",
    "global_writes", "store_events", "store_dwords", "workgroups_completed",
    "signal_before", "signal_after", "admission_tick", "start_tick", "end_tick",
    "retire_tick", "compatibility_completion_token_crc32c", "kernel_executed",
    "native_queue_retired", "application_pins_released", "adapter_released",
)


def validate_trace(path: Path, session: dict[str, Any]) -> dict[str, Any]:
    records = []
    with path.open("r", encoding="ascii") as stream:
        for line in stream:
            require(line.endswith("\n"), "trace line is unterminated")
            value = json.loads(line)
            require(isinstance(value, dict), "trace record is not an object")
            records.append(value)
    require(len(records) == 72, "trace must contain exactly 72 records")
    expected_events = []
    for index in range(24):
        expected_events.extend(("generic_execution_retired", "generic_execution_type20_durable"))
        expected_events.append("generic_execution_session_complete" if index == 23 else "generic_execution_reuse_complete")
    require([record.get("event") for record in records] == expected_events, "trace lifecycle order mismatch")
    request_ids = []
    image_counts: Counter[str] = Counter()
    for index in range(24):
        group = records[index * 3:index * 3 + 3]
        base = group[0]
        request_ids.append(base.get("request_id"))
        require(all(record.get("schema") == "amdgpu-sim.generic-kernel-execution-trace.v1" for record in group), "trace schema mismatch")
        require(all(record.get("kernel") == "_sum_kernel" for record in group), "wrong trace kernel")
        image = base.get("image_sha256")
        require(isinstance(image, str) and re.fullmatch(r"[0-9a-f]{64}", image) is not None, "image SHA invalid")
        image_counts[image] += 1
        for record in group[1:]:
            require(all(record.get(field) == base.get(field) for field in TRACE_MATCH_FIELDS), "lifecycle record fields differ")
        require(base.get("type20_durable") is False, "retired record is prematurely durable")
        require(group[1].get("type20_durable") is True, "type20 record is not durable")
        require(group[2].get("type20_durable") is True, "terminal record lost durability")
        require(all(base.get(field) is True for field in (
            "kernel_executed", "native_queue_retired", "application_pins_released", "adapter_released"
        )), "execution lifecycle incomplete")
        require(base.get("signal_before") == 1 and base.get("signal_after") == 0, "signal lifecycle mismatch")
        require(base.get("packet_fetches") == 1 and base.get("command_processor_submissions") == 1 and base.get("gpu_dispatcher_starts") == 1, "dispatch stages mismatch")
        require(base.get("allocation_count") == 2 and len(base.get("allocations", [])) == 2, "SUM allocation count mismatch")
        require(base.get("global_reads", 0) > 0 and base.get("global_writes", 0) > 0, "SUM memory traffic missing")
        require(base.get("store_events", 0) > 0 and base.get("store_dwords", 0) > 0, "SUM stores missing")
        require(base.get("workgroups_completed", 0) > 0 and base.get("waves_started", 0) > 0, "SUM execution missing")
        terminal = group[2]
        if index < 23:
            require(terminal.get("unmap_durable") is False and terminal.get("cleanup_complete") is False, "reuse is terminal")
        else:
            require(terminal.get("unmap_durable") is True, "final unmap is not durable")
            require(terminal.get("owner_disconnected") is True, "owner did not disconnect")
            require(terminal.get("cleanup_complete") is True, "terminal cleanup missing")
            require(terminal.get("owner_quarantined") is False, "owner was quarantined")
    require(len(set(request_ids)) == 24, "request IDs are not unique")
    return {
        "records": len(records),
        "retired": 24,
        "type20_durable": 24,
        "reuse_complete": 23,
        "session_complete": 1,
        "unique_request_ids": 24,
        "image_sha256_counts": dict(sorted(image_counts.items())),
        "trace_sha256": file_sha256(path),
        "job_uuid": session["job_uuid"],
        "epoch": session["epoch"],
        "rank": session["rank"],
        "world_size": session["world_size"],
        "correct": True,
    }


def bind_kernel_images(
    cache_directory: Path,
    image_counts: dict[str, int],
) -> dict[str, Any]:
    require(cache_directory.is_dir(), "Triton cache directory is missing")
    require(not cache_directory.is_symlink(), "Triton cache directory is a symlink")
    require(cache_directory.resolve(strict=True) == cache_directory, "Triton cache is noncanonical")
    expected = set(image_counts)
    matches: dict[str, list[Path]] = {digest: [] for digest in expected}
    for candidate in cache_directory.rglob("*.hsaco"):
        regular_file(candidate)
        digest = file_sha256(candidate)
        if digest in matches:
            matches[digest].append(candidate)
    require(all(matches[digest] for digest in expected), "trace image is absent from Triton cache")
    images: dict[str, Any] = {}
    for digest in sorted(expected):
        selected = sorted(matches[digest], key=lambda path: str(path))[0]
        directory = selected.parent
        siblings: dict[str, Any] = {}
        for artifact in sorted(directory.iterdir(), key=lambda path: path.name):
            regular_file(artifact)
            siblings[artifact.name] = {
                "bytes": artifact.stat().st_size,
                "sha256": file_sha256(artifact),
            }
        require(selected.name == "_sum_kernel.hsaco", "trace image has wrong cache basename")
        require(
            {"_sum_kernel.ttir", "_sum_kernel.ttgir", "_sum_kernel.llir", "_sum_kernel.amdgcn"}
            <= set(siblings),
            "trace image cache entry is incomplete",
        )
        images[digest] = {
            "launch_count": image_counts[digest],
            "selected_path": str(selected),
            "relative_path": str(selected.relative_to(cache_directory)),
            "artifacts": siblings,
        }
    return {
        "cache_directory": str(cache_directory),
        "image_count": len(images),
        "images": images,
        "all_trace_images_bound": True,
    }


def validate_log(path: Path, session: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    require("fatal:" not in text.lower() and "panic:" not in text.lower(), "gem5 log has fatal or panic")
    require("host-gpu-dispatch-exit cause=host GPU dispatch session complete code=0" in text, "gem5 did not exit cleanly")
    ready = (
        f"job_uuid={session['job_uuid']} epoch={session['epoch']} "
        f"rank={session['rank']} world={session['world_size']}"
    )
    require(ready in text, "gem5 ready identity mismatch")
    require(f"--outdir {paths['output_directory']}" in text, "gem5 log output path mismatch")
    require(f"--dispatch-trace-path {paths['trace_path']}" in text, "gem5 log trace path mismatch")
    return {"sha256": file_sha256(path), "clean_exit": True, "identity_match": True}


def validate_stats(path: Path) -> dict[str, Any]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            values[fields[0]] = fields[1]
    require(values.get("system.host_gpu_bridge.host_fallback_count") == "0", "authoritative host fallback is nonzero")
    sim_ticks = int(values["simTicks"])
    host_seconds = float(values["hostSeconds"])
    require(sim_ticks > 0 and math.isfinite(host_seconds) and host_seconds > 0, "stats timing invalid")
    return {
        "sha256": file_sha256(path),
        "sim_ticks": sim_ticks,
        "host_seconds": host_seconds,
        "host_fallback_count": 0,
        "correct": True,
    }


def copy_file(source: Path, destination: Path) -> dict[str, Any]:
    payload = source.read_bytes()
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": destination.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("ascii") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": path.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(AT_FDCWD, os.fsencode(source), AT_FDCWD, os.fsencode(destination), RENAME_NOREPLACE) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    require(output.is_absolute() and output == Path(os.path.normpath(output)), "output path must be canonical absolute")
    require(not output.exists() and not output.is_symlink(), "output path already exists")
    parent = output.parent.resolve(strict=True)
    require(output.parent == parent and parent.is_relative_to(ROOT), "output parent must be inside the repository")
    before = identity_snapshot()
    prefix = validate_prefix_manifest(before)
    command = ["/usr/bin/python3", str(RUNNER)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    result = strict_result(completed.stdout)
    require(completed.returncode == 0, f"runner failed: {completed.returncode}")
    require(completed.stderr == "", "runner wrote stderr")
    after = identity_snapshot()
    require(before == after, "execution identity drifted during run")
    require(prefix == validate_prefix_manifest(after), "prefix identity drifted during run")
    session = result["managed_session"]
    paths = validate_session_paths(result, before)
    trace = validate_trace(paths["trace_path"], session)
    kernels = bind_kernel_images(
        paths["triton_cache_directory"], trace["image_sha256_counts"]
    )
    log = validate_log(paths["log_path"], session, paths)
    stats = validate_stats(paths["stats_path"])
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        artifacts = {
            "runner_stdout": copy_file_from_bytes(completed.stdout.encode("ascii"), temporary / "runner.stdout"),
            "runner_stderr": copy_file_from_bytes(completed.stderr.encode("ascii"), temporary / "runner.stderr"),
            "dispatch_trace": copy_file(paths["trace_path"], temporary / "dispatch-trace.jsonl"),
            "gem5_log": copy_file(paths["log_path"], temporary / "gem5.log"),
            "stats": copy_file(paths["stats_path"], temporary / "stats.txt"),
        }
        payload = {
            "schema": "amdgpu-sim.ccl-device-sum-acceptance.v1",
            "command": command,
            "runner_exit_code": completed.returncode,
            "execution_identity": before,
            "prefix_validation": prefix,
            "identity_unchanged_postflight": True,
            "managed_session": session,
            "numerical_result": result,
            "trace": trace,
            "kernel_images": kernels,
            "stats": stats,
            "gem5_log": log,
            "artifacts": artifacts,
            "host_reduction_count": 0,
            "fallback_count": 0,
            "cpu_fallback_count": 0,
            "nvidia_fallback_count": 0,
            "target_feedback": False,
            "planner_binding_accepted": False,
            "live_collective_accepted": False,
            "vllm_communicator_accepted": False,
            "output_correct": True,
            "claim_scope": "standalone_private_workspace_device_sum_primitive",
            "claim_boundary": "BF16 and FP32 pairwise device SUM only; no planner ring, authenticated peers, public-output collective commit, vLLM communicator, or TP",
        }
        write_json(temporary / "result.json", payload)
        manifest_artifacts = {}
        for artifact in sorted(temporary.iterdir()):
            regular_file(artifact)
            manifest_artifacts[artifact.name] = {
                "bytes": artifact.stat().st_size,
                "sha256": file_sha256(artifact),
            }
        write_json(temporary / "manifest.json", {
            "schema": "amdgpu-sim.ccl-device-sum-evidence-manifest.v1",
            "artifacts": manifest_artifacts,
            "complete": True,
        })
        directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        rename_noreplace(temporary, output)
        temporary = None
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        print(json.dumps(payload, sort_keys=True))
        return 0
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def copy_file_from_bytes(payload: bytes, destination: Path) -> dict[str, Any]:
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": destination.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


if __name__ == "__main__":
    raise SystemExit(main())
