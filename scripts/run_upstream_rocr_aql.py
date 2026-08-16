#!/usr/bin/env python3
"""Run unchanged upstream ROCr over the standard AQL runtime-gem5 path."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import upstream_rocr_aql_acceptance as contract  # noqa: E402


class RunnerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def validate_output(output: Path) -> Path:
    require(output.is_absolute(), "output directory must be absolute")
    require(not os.path.lexists(output), "output directory must be absent")
    parent = output.parent.resolve(strict=True)
    require(output.parent == parent, "output parent contains a symlink")
    metadata = parent.lstat()
    require(metadata.st_uid == os.getuid(), "output parent has the wrong owner")
    require(not parent.is_symlink(), "output parent is a symlink")
    return output


def private_directories(execution_root: Path) -> None:
    for relative in (
        "home",
        "tmp",
        "xdg-cache",
        "xdg-config",
        "xdg-data",
        "m5out",
    ):
        path = execution_root / relative
        path.mkdir(mode=0o700)
        path.chmod(0o700)


def common_environment(execution_root: Path) -> dict[str, str]:
    return {
        "HOME": str(execution_root / "home"),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TMPDIR": str(execution_root / "tmp"),
        "XDG_CACHE_HOME": str(execution_root / "xdg-cache"),
        "XDG_CONFIG_HOME": str(execution_root / "xdg-config"),
        "XDG_DATA_HOME": str(execution_root / "xdg-data"),
    }


def worker_environment(execution_root: Path, endpoint: Path) -> dict[str, str]:
    environment = common_environment(execution_root)
    environment.update(
        {
            "HSA_ENABLE_DTIF_FAST_COPY": "0",
            "HSA_ENABLE_DXG_DETECTION": "0",
            "HSA_ENABLE_INTERRUPT": "0",
            "HSA_MODEL_LIB": str(contract.MODEL_LIBRARY.resolve()),
            "HSA_MODEL_TOPOLOGY": str(contract.TOPOLOGY.resolve()),
            "LD_LIBRARY_PATH": (
                f"{contract.RUNTIME_BUILD.resolve()}:"
                f"{contract.ROCR_LIBRARY.resolve().parent}"
            ),
            "SAGR_GENERIC_BRIDGE_ENDPOINT": str(endpoint),
            "SAGR_HSAKMT_MODEL_TRACE": "1",
            "SAGR_UPSTREAM_ROCR_EXECUTION_TRACE": "1",
        }
    )
    return environment


def gem5_argv(
    execution_root: Path, endpoint: Path, trace_path: Path, job_uuid: str
) -> list[str]:
    return [
        str(contract.GEM5_BINARY.resolve()),
        "--listener-mode=on",
        "--outdir",
        str(execution_root / "m5out"),
        str(contract.GEM5_CONFIG.resolve()),
        "--endpoint",
        str(endpoint),
        "--dispatch-trace-path",
        str(trace_path),
        "--epoch",
        "1",
        "--job-uuid",
        job_uuid,
        "--rank",
        "0",
        "--world-size",
        "1",
        "--startup-timeout-ms",
        "86400000",
        "--handshake-timeout-ms",
        "15000",
        "--run-timeout-ms",
        "86400000",
    ]


def worker_argv() -> list[str]:
    return [
        str(contract.WORKER.resolve()),
        "--execute",
        str(contract.KERNEL.resolve()),
    ]


def _proc_fields(pid: int) -> tuple[int, int] | None:
    try:
        payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return None
    closing = payload.rfind(")")
    require(closing >= 0, f"process stat is malformed: {pid}")
    fields = payload[closing + 2 :].split()
    require(len(fields) > 19, f"process stat is truncated: {pid}")
    return int(fields[2]), int(fields[19])


def process_identity(process: subprocess.Popen[bytes], role: str) -> tuple[int, int, int]:
    fields = _proc_fields(process.pid)
    require(fields is not None, f"{role} exited before identity capture")
    process_group, start_time = fields
    require(process_group == process.pid, f"{role} did not create a private process group")
    return process.pid, start_time, process_group


def process_group_members(process_group: int) -> list[int]:
    members: list[int] = []
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        fields = _proc_fields(int(candidate.name))
        if fields is not None and fields[0] == process_group:
            members.append(int(candidate.name))
    return sorted(members)


def terminate_group(
    process: subprocess.Popen[bytes] | None,
    process_group: int | None,
    grace_seconds: float = 3.0,
) -> bool:
    if process is None or process_group is None:
        return False
    if process.poll() is not None and not process_group_members(process_group):
        return False
    forced = True
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None and not process_group_members(process_group):
            return forced
        time.sleep(0.02)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    return forced


def wait_for_endpoint(
    endpoint: Path,
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RunnerError(f"gem5 exited before endpoint publication: {return_code}")
        try:
            metadata = endpoint.lstat()
        except FileNotFoundError:
            time.sleep(0.02)
            continue
        require(metadata.st_uid == os.getuid(), "gem5 endpoint has the wrong owner")
        require(stat.S_ISSOCK(metadata.st_mode), "gem5 endpoint is not a socket")
        return
    raise RunnerError("gem5 endpoint publication timed out")


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
                raise OSError(errno.EIO, "short runner artifact write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _copy_file(source: Path, destination: Path) -> dict[str, Any]:
    metadata = source.lstat()
    require(not source.is_symlink() and metadata.st_uid == os.getuid(), f"unsafe artifact: {source}")
    require((metadata.st_mode & 0o170000) == 0o100000, f"artifact is not regular: {source}")
    record = _write_bytes(destination, source.read_bytes())
    return {"path": destination.name, **record}


def _fsync_tree(root: Path) -> None:
    directories = {root}
    directories.update(path for path in root.rglob("*") if path.is_dir())
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def publish_source(
    output: Path,
    execution_root: Path,
    manifest_core: Mapping[str, Any],
    error: str | None,
) -> dict[str, Any]:
    parent = output.parent.resolve(strict=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        for relative in contract.SOURCE_ARTIFACTS:
            source = execution_root / relative
            if not source.is_file() or source.is_symlink():
                continue
            destination = temporary / relative
            record = _copy_file(source, destination)
            record["path"] = relative
            artifacts[relative] = record
        if error is not None:
            payload = (error.rstrip() + "\n").encode("ascii", errors="backslashreplace")
            record = _write_bytes(temporary / "runner-error.txt", payload)
            artifacts["runner-error.txt"] = {"path": "runner-error.txt", **record}
        manifest = dict(manifest_core)
        manifest["artifacts"] = artifacts
        _write_bytes(temporary / "result-manifest.json", contract.canonical_json(manifest))
        _fsync_tree(temporary)
        contract.rename_noreplace(temporary, output)
        temporary = None
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return manifest
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)


def run_gate(
    output: Path,
    *,
    worker_timeout_seconds: int,
    gem5_exit_timeout_seconds: int,
    startup_timeout_seconds: int,
) -> dict[str, Any]:
    validate_output(output)
    require(1 <= worker_timeout_seconds <= 600, "worker timeout is outside 1..600")
    require(1 <= gem5_exit_timeout_seconds <= 120, "gem5 exit timeout is outside 1..120")
    require(1 <= startup_timeout_seconds <= 120, "startup timeout is outside 1..120")

    identity_preflight = contract.identity_snapshot()
    execution_root = Path(tempfile.mkdtemp(prefix="gs-rocr-aql-", dir="/tmp"))
    execution_root.chmod(0o700)
    private_directories(execution_root)
    endpoint = execution_root / "bridge.sock"
    trace_path = execution_root / "dispatch-trace.jsonl"
    require(len(os.fsencode(endpoint)) < 108, "private endpoint exceeds AF_UNIX capacity")
    job_uuid = secrets.token_hex(16)
    gem5_command = gem5_argv(execution_root, endpoint, trace_path, job_uuid)
    worker_command = worker_argv()
    gem5_env = common_environment(execution_root)
    worker_env = worker_environment(execution_root, endpoint)

    gem5_process: subprocess.Popen[bytes] | None = None
    worker_process: subprocess.Popen[bytes] | None = None
    gem5_log = None
    worker_log = None
    gem5_pid = gem5_start = gem5_group = None
    worker_pid = worker_start = worker_group = None
    gem5_exit_code: int | None = None
    worker_exit_code: int | None = None
    gem5_forced = False
    worker_forced = False
    failure: str | None = None
    identity_postflight: dict[str, Any] | None = None

    try:
        gem5_log = (execution_root / "gem5.log").open("wb", buffering=0)
        gem5_process = subprocess.Popen(
            gem5_command,
            cwd=ROOT,
            env=gem5_env,
            stdin=subprocess.DEVNULL,
            stdout=gem5_log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
        gem5_pid, gem5_start, gem5_group = process_identity(gem5_process, "gem5")
        wait_for_endpoint(endpoint, gem5_process, startup_timeout_seconds)

        worker_log = (execution_root / "worker.log").open("wb", buffering=0)
        worker_process = subprocess.Popen(
            worker_command,
            cwd=ROOT,
            env=worker_env,
            stdin=subprocess.DEVNULL,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
        worker_pid, worker_start, worker_group = process_identity(worker_process, "worker")
        try:
            worker_exit_code = worker_process.wait(timeout=worker_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise RunnerError("upstream ROCr worker timed out") from error
        if worker_exit_code != 0:
            raise RunnerError(f"upstream ROCr worker exited {worker_exit_code}")
        try:
            gem5_exit_code = gem5_process.wait(timeout=gem5_exit_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise RunnerError("gem5 did not exit after the standard ROCr session") from error
        if gem5_exit_code != 0:
            raise RunnerError(f"gem5 exited {gem5_exit_code}")
    except Exception as error:
        failure = str(error)
    finally:
        worker_forced = terminate_group(worker_process, worker_group)
        gem5_forced = terminate_group(gem5_process, gem5_group)
        if worker_process is not None:
            try:
                worker_exit_code = worker_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                failure = failure or "worker could not be reaped"
        if gem5_process is not None:
            try:
                gem5_exit_code = gem5_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                failure = failure or "gem5 could not be reaped"
        if worker_log is not None:
            worker_log.close()
        if gem5_log is not None:
            gem5_log.close()
        try:
            identity_postflight = contract.identity_snapshot()
        except Exception as error:
            failure = failure or f"postflight identity failed: {error}"

    worker_reaped = worker_process is not None and worker_process.poll() is not None
    gem5_reaped = gem5_process is not None and gem5_process.poll() is not None
    worker_group_absent = worker_group is not None and not process_group_members(worker_group)
    gem5_group_absent = gem5_group is not None and not process_group_members(gem5_group)
    endpoint_absent = not os.path.lexists(endpoint)
    cleanup = {
        "worker_reaped": worker_reaped,
        "gem5_reaped": gem5_reaped,
        "worker_process_group_absent": worker_group_absent,
        "gem5_process_group_absent": gem5_group_absent,
        "endpoint_absent": endpoint_absent,
        "worker_forced_termination": worker_forced,
        "gem5_forced_termination": gem5_forced,
        "all_clear": (
            worker_reaped
            and gem5_reaped
            and worker_group_absent
            and gem5_group_absent
            and endpoint_absent
            and not worker_forced
            and not gem5_forced
        ),
    }
    required_artifacts = all((execution_root / value).is_file() for value in contract.SOURCE_ARTIFACTS)
    identity_unchanged = identity_postflight == identity_preflight
    success = (
        failure is None
        and worker_exit_code == 0
        and gem5_exit_code == 0
        and cleanup["all_clear"]
        and required_artifacts
        and identity_unchanged
    )
    if not success and failure is None:
        if not identity_unchanged:
            failure = "execution identity drifted"
        elif not cleanup["all_clear"]:
            failure = "process cleanup was incomplete"
        elif not required_artifacts:
            failure = "required execution artifacts are missing"
        else:
            failure = "execution did not meet the success contract"

    execution = {
        "job_uuid": job_uuid,
        "epoch": 1,
        "rank": 0,
        "world_size": 1,
        "execution_root": str(execution_root),
        "endpoint": str(endpoint),
        "trace_path": str(trace_path),
        "m5out_path": str(execution_root / "m5out"),
        "gem5_argv": gem5_command,
        "worker_argv": worker_command,
        "gem5_environment": gem5_env,
        "worker_environment": worker_env,
        "gem5_pid": gem5_pid,
        "gem5_start_time_ticks": gem5_start,
        "gem5_process_group": gem5_group,
        "worker_pid": worker_pid,
        "worker_start_time_ticks": worker_start,
        "worker_process_group": worker_group,
        "worker_exit_code": worker_exit_code,
        "gem5_exit_code": gem5_exit_code,
        "worker_timeout_seconds": worker_timeout_seconds,
        "gem5_exit_timeout_seconds": gem5_exit_timeout_seconds,
    }
    manifest_core = {
        "schema": contract.RUN_SCHEMA,
        "status": "success" if success else "failed",
        "claim_scope": contract.CLAIM_SCOPE,
        "hip_runtime_accepted": False,
        "pytorch_rocm_accepted": False,
        "model_accepted": False,
        "identity_preflight": identity_preflight,
        "identity_postflight": identity_postflight,
        "execution": execution,
        "cleanup": cleanup,
    }
    try:
        manifest = publish_source(output, execution_root, manifest_core, failure)
    finally:
        shutil.rmtree(execution_root)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-timeout-seconds", type=int, default=120)
    parser.add_argument("--gem5-exit-timeout-seconds", type=int, default=20)
    parser.add_argument("--startup-timeout-seconds", type=int, default=30)
    arguments = parser.parse_args()
    try:
        result = run_gate(
            arguments.output_dir,
            worker_timeout_seconds=arguments.worker_timeout_seconds,
            gem5_exit_timeout_seconds=arguments.gem5_exit_timeout_seconds,
            startup_timeout_seconds=arguments.startup_timeout_seconds,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "success" else 1
    except (RunnerError, FileExistsError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"upstream ROCr AQL run failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
