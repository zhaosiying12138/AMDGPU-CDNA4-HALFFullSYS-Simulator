#!/usr/bin/env python3
"""Own one isolated rank-0/world-1 gem5 session for acceptance runners."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import os
from pathlib import Path
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


class LifecycleError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LifecycleError(message)


def validate_absent_output(output: Path) -> Path:
    require(output.is_absolute(), "output directory must be absolute")
    require(not os.path.lexists(output), "output directory must be absent")
    parent = output.parent.resolve(strict=True)
    require(output.parent == parent, "output parent contains a symlink")
    metadata = parent.lstat()
    require(stat.S_ISDIR(metadata.st_mode), "output parent is not a directory")
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


def isolated_environment(execution_root: Path) -> dict[str, str]:
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


def process_identity(
    process: subprocess.Popen[bytes], role: str
) -> tuple[int, int, int]:
    fields = _proc_fields(process.pid)
    require(fields is not None, f"{role} exited before identity capture")
    process_group, start_time = fields
    require(process_group == process.pid, f"{role} lacks a private process group")
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
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None and not process_group_members(process_group):
            return True
        time.sleep(0.02)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    return True


def wait_for_endpoint(
    endpoint: Path,
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise LifecycleError(
                f"gem5 exited before endpoint publication: {return_code}"
            )
        try:
            metadata = endpoint.lstat()
        except FileNotFoundError:
            time.sleep(0.02)
            continue
        require(metadata.st_uid == os.getuid(), "gem5 endpoint has the wrong owner")
        require(stat.S_ISSOCK(metadata.st_mode), "gem5 endpoint is not a socket")
        return
    raise LifecycleError("gem5 endpoint publication timed out")


def gem5_argv(
    *,
    binary: Path,
    config: Path,
    execution_root: Path,
    endpoint: Path,
    trace_path: Path,
    job_uuid: str,
    startup_timeout_ms: int = 86_400_000,
    handshake_timeout_ms: int = 15_000,
    run_timeout_ms: int = 86_400_000,
) -> list[str]:
    return [
        str(binary.resolve(strict=True)),
        "--listener-mode=on",
        "--outdir",
        str(execution_root / "m5out"),
        str(config.resolve(strict=True)),
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
        str(startup_timeout_ms),
        "--handshake-timeout-ms",
        str(handshake_timeout_ms),
        "--run-timeout-ms",
        str(run_timeout_ms),
    ]


@dataclass(frozen=True)
class SessionResult:
    execution_root: Path
    execution: dict[str, Any]
    cleanup: dict[str, bool]
    failure: str | None
    process_success: bool


def run_session(
    *,
    repository_root: Path,
    execution_prefix: str,
    gem5_binary: Path,
    gem5_config: Path,
    worker_argv: Sequence[str],
    worker_environment_factory: Callable[[Path, Path], Mapping[str, str]],
    worker_label: str,
    worker_timeout_seconds: int,
    gem5_exit_timeout_seconds: int,
    startup_timeout_seconds: int,
) -> SessionResult:
    require(1 <= worker_timeout_seconds <= 3600, "worker timeout is outside 1..3600")
    require(1 <= gem5_exit_timeout_seconds <= 300, "gem5 exit timeout is outside 1..300")
    require(1 <= startup_timeout_seconds <= 300, "startup timeout is outside 1..300")
    require(worker_argv and all(isinstance(value, str) and value for value in worker_argv), "worker argv is invalid")

    execution_root = Path(tempfile.mkdtemp(prefix=execution_prefix, dir="/tmp"))
    execution_root.chmod(0o700)
    private_directories(execution_root)
    endpoint = execution_root / "bridge.sock"
    trace_path = execution_root / "dispatch-trace.jsonl"
    require(len(os.fsencode(endpoint)) < 108, "private endpoint exceeds AF_UNIX capacity")
    job_uuid = secrets.token_hex(16)
    gem5_command = gem5_argv(
        binary=gem5_binary,
        config=gem5_config,
        execution_root=execution_root,
        endpoint=endpoint,
        trace_path=trace_path,
        job_uuid=job_uuid,
    )
    gem5_environment = isolated_environment(execution_root)
    worker_environment = dict(worker_environment_factory(execution_root, endpoint))

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

    try:
        gem5_log = (execution_root / "gem5.log").open("wb", buffering=0)
        gem5_process = subprocess.Popen(
            gem5_command,
            cwd=repository_root,
            env=gem5_environment,
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
            list(worker_argv),
            cwd=repository_root,
            env=worker_environment,
            stdin=subprocess.DEVNULL,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
        worker_pid, worker_start, worker_group = process_identity(
            worker_process, worker_label
        )
        try:
            worker_exit_code = worker_process.wait(timeout=worker_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise LifecycleError(f"{worker_label} timed out") from error
        if worker_exit_code != 0:
            raise LifecycleError(f"{worker_label} exited {worker_exit_code}")
        try:
            gem5_exit_code = gem5_process.wait(timeout=gem5_exit_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise LifecycleError(
                f"gem5 did not exit after the {worker_label} session"
            ) from error
        if gem5_exit_code != 0:
            raise LifecycleError(f"gem5 exited {gem5_exit_code}")
    except Exception as error:
        failure = str(error)
    finally:
        worker_forced = terminate_group(worker_process, worker_group)
        gem5_forced = terminate_group(gem5_process, gem5_group)
        if worker_process is not None:
            try:
                worker_exit_code = worker_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                failure = failure or f"{worker_label} could not be reaped"
        if gem5_process is not None:
            try:
                gem5_exit_code = gem5_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                failure = failure or "gem5 could not be reaped"
        if worker_log is not None:
            worker_log.close()
        if gem5_log is not None:
            gem5_log.close()

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
    process_success = (
        failure is None
        and worker_exit_code == 0
        and gem5_exit_code == 0
        and cleanup["all_clear"]
    )
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
        "worker_argv": list(worker_argv),
        "gem5_environment": gem5_environment,
        "worker_environment": worker_environment,
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
        "startup_timeout_seconds": startup_timeout_seconds,
    }
    return SessionResult(
        execution_root=execution_root,
        execution=execution,
        cleanup=cleanup,
        failure=failure,
        process_success=process_success,
    )


def write_bytes(path: Path, payload: bytes) -> dict[str, Any]:
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


def copy_file(source: Path, destination: Path) -> dict[str, Any]:
    metadata = source.lstat()
    require(not source.is_symlink(), f"artifact is a symlink: {source}")
    require(stat.S_ISREG(metadata.st_mode), f"artifact is not regular: {source}")
    require(metadata.st_uid == os.getuid(), f"artifact has the wrong owner: {source}")
    return write_bytes(destination, source.read_bytes())


def fsync_tree(root: Path) -> None:
    directories = {root}
    directories.update(path for path in root.rglob("*") if path.is_dir())
    for directory in sorted(
        directories, key=lambda value: len(value.parts), reverse=True
    ):
        descriptor = os.open(
            directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def publish_source(
    *,
    output: Path,
    execution_root: Path,
    artifact_paths: Sequence[str],
    manifest: Mapping[str, Any],
    canonical_json: Callable[[object], bytes],
    rename_noreplace: Callable[[Path, Path], None],
    error: str | None,
) -> dict[str, Any]:
    validate_absent_output(output)
    parent = output.parent.resolve(strict=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        for relative in artifact_paths:
            source = execution_root / relative
            if not source.is_file() or source.is_symlink():
                continue
            record = copy_file(source, temporary / relative)
            artifacts[relative] = {"path": relative, **record}
        if error is not None:
            payload = (error.rstrip() + "\n").encode("ascii", errors="backslashreplace")
            record = write_bytes(temporary / "runner-error.txt", payload)
            artifacts["runner-error.txt"] = {"path": "runner-error.txt", **record}
        document = dict(manifest)
        document["artifacts"] = artifacts
        write_bytes(temporary / "result-manifest.json", canonical_json(document))
        fsync_tree(temporary)
        rename_noreplace(temporary, output)
        temporary = None
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return document
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
