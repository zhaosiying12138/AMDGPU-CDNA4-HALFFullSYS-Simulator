#!/usr/bin/env python3
"""Run the pinned AgentENV server as a feature-local process.

The wrapper deliberately avoids system-wide installation and service managers.
Its config, dependency cache, runtime sockets, logs, and ownership record all
live below ``build/agentenv-integration/server`` in this worktree.

Only a process whose PID, Linux start-time tick, and executable all match the
feature-local ownership record may be signalled by ``stop``.  This prevents a
stale or edited PID file from targeting another workload after PID reuse.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Iterator, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SERVICE_SCHEMA = "amdgpu-sim.agentenv-service.v1"
PROCESS_SCHEMA = "amdgpu-sim.agentenv-service-process.v1"
DEFAULT_API_ADDR = "127.0.0.1:18080"
DEFAULT_GUEST_IMAGE = "ubuntu:26.04"
DEFAULT_STATE_RELATIVE = Path("build/agentenv-integration/server")
DEFAULT_BINARY_RELATIVE = Path("build/agentenv-cargo/release/server")
LEGACY_BINARY_RELATIVE = Path("projects/AgentENV/build/agentenv-cargo/release/server")
DEFAULT_SOURCE_CONFIG_RELATIVE = Path("projects/AgentENV/config/default.toml")
DEFAULT_SERVER_SOURCE_RELATIVE = Path("projects/AgentENV")
IMAGE_LINE_RE = re.compile(r'^default_image\s*=\s*"[^"]+"\s*$', re.MULTILINE)
IMAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]*$")


class ServiceError(RuntimeError):
    """Raised when a lifecycle operation cannot be completed safely."""


def now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        temporary.unlink(missing_ok=True)


def validate_api_addr(value: str) -> str:
    host, separator, port_text = value.rpartition(":")
    if not separator or not port_text.isdigit():
        raise ServiceError(f"invalid API address {value!r}; expected 127.0.0.1:PORT")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ServiceError(f"invalid API address {value!r}") from error
    port = int(port_text)
    if address != ipaddress.ip_address("127.0.0.1"):
        raise ServiceError("AgentENV API must bind exactly to loopback 127.0.0.1")
    if not 1 <= port <= 65535:
        raise ServiceError(f"invalid API port {port}")
    return f"127.0.0.1:{port}"


def validate_guest_image(value: str) -> str:
    if not IMAGE_NAME_RE.fullmatch(value):
        raise ServiceError(f"invalid guest image reference {value!r}")
    return value


def _path_from_repo(repo: Path, value: str | None, default: Path) -> Path:
    candidate = Path(value) if value else default
    if not candidate.is_absolute():
        candidate = repo / candidate
    return absolute(candidate)


@dataclass(frozen=True)
class ServicePaths:
    repo: Path
    root: Path
    binary: Path
    source_config: Path
    source_cwd: Path
    config: Path
    environment_json: Path
    environment_sh: Path
    credentials: Path
    credentials_metadata: Path
    xdg_config: Path
    home: Path
    runtime: Path
    deps: Path
    tmp: Path
    cache: Path
    logs: Path
    server_log: Path
    pidfile: Path
    last_stop: Path
    lockfile: Path


def service_paths(repo: Path, server_binary: str | None = None) -> ServicePaths:
    repo = absolute(repo)
    root = repo / DEFAULT_STATE_RELATIVE
    if server_binary:
        binary = _path_from_repo(repo, server_binary, DEFAULT_BINARY_RELATIVE)
    else:
        binary = repo / DEFAULT_BINARY_RELATIVE
        if not binary.exists() and (repo / LEGACY_BINARY_RELATIVE).exists():
            binary = repo / LEGACY_BINARY_RELATIVE
    return ServicePaths(
        repo=repo,
        root=root,
        binary=absolute(binary),
        source_config=repo / DEFAULT_SOURCE_CONFIG_RELATIVE,
        source_cwd=repo / DEFAULT_SERVER_SOURCE_RELATIVE,
        config=root / "config" / "default.toml",
        environment_json=root / "environment.json",
        environment_sh=root / "server.env",
        credentials=root / "xdg-config" / "aenv" / "credentials",
        credentials_metadata=root / "credentials.json",
        xdg_config=root / "xdg-config",
        home=root / "home",
        runtime=root / "runtime",
        deps=root / "deps",
        tmp=root / "tmp",
        cache=root / "cache",
        logs=root / "logs",
        server_log=root / "logs" / "server.log",
        pidfile=root / "process.json",
        last_stop=root / "last-stop.json",
        lockfile=root / "control.lock",
    )


def _assert_no_state_symlink(paths: ServicePaths) -> None:
    """Reject an existing symlink that would redirect feature-local state."""

    cursor = paths.repo
    for component in DEFAULT_STATE_RELATIVE.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ServiceError(f"refusing symlink in AgentENV state path: {cursor}")
    for candidate in (
        paths.root,
        paths.config,
        paths.config.parent,
        paths.environment_json,
        paths.environment_sh,
        paths.credentials,
        paths.credentials_metadata,
        paths.xdg_config,
        paths.home,
        paths.runtime,
        paths.deps,
        paths.tmp,
        paths.cache,
        paths.logs,
        paths.server_log,
        paths.pidfile,
        paths.last_stop,
        paths.lockfile,
    ):
        if candidate.is_symlink():
            raise ServiceError(f"refusing symlink in AgentENV state path: {candidate}")


def credential_metadata(api_addr: str) -> dict[str, Any]:
    api_addr = validate_api_addr(api_addr)
    api_url = f"http://{api_addr}"
    return {
        "schema": "amdgpu-sim.agentenv-local-credentials.v1",
        "scope": "loopback-development-only",
        "api_url": api_url,
        "sandbox_url": api_url,
        # AgentENV's manual local mode requires non-empty placeholder values.
        "api_key": "e2b_000000",
        "access_token": "dummy",
    }


def environment_overrides(paths: ServicePaths, api_addr: str) -> dict[str, str]:
    api_addr = validate_api_addr(api_addr)
    credentials = credential_metadata(api_addr)
    return {
        "AENV_CONFIG_PATH": str(paths.config),
        "AENV_API_URL": credentials["api_url"],
        "AENV_API_KEY": credentials["api_key"],
        "AENV_DEPS_PATH": str(paths.deps),
        "AENV_HOME_PATH": str(paths.home),
        "AENV_LOG_FORMAT": "json",
        "AENV_RUNTIME_PATH": str(paths.runtime),
        "AENV_VIRTUALIZATION_MODE": "kvm",
        "API_ADDR": api_addr,
        "E2B_ACCESS_TOKEN": credentials["access_token"],
        "E2B_API_KEY": credentials["api_key"],
        "E2B_API_URL": credentials["api_url"],
        "E2B_SANDBOX_URL": credentials["sandbox_url"],
        "TMPDIR": str(paths.tmp),
        "XDG_CACHE_HOME": str(paths.cache),
        "XDG_CONFIG_HOME": str(paths.xdg_config),
        "XDG_RUNTIME_DIR": str(paths.runtime / "xdg"),
    }


def render_config(source: bytes, guest_image: str) -> bytes:
    guest_image = validate_guest_image(guest_image)
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ServiceError("AgentENV source config is not UTF-8") from error
    replacement = f'default_image = "{guest_image}"'
    rendered, count = IMAGE_LINE_RE.subn(replacement, text)
    if count != 1:
        raise ServiceError(
            "expected exactly one image.resolver default_image in AgentENV config"
        )
    header = (
        "# Generated by tools/agentenv_service.py; do not edit in place.\n"
        "# Regenerate by running the service start command.\n"
    )
    return (header + rendered).encode("utf-8")


def shell_environment(payload: dict[str, str]) -> bytes:
    lines = [f"export {key}={shlex.quote(payload[key])}" for key in sorted(payload)]
    return ("\n".join(lines) + "\n").encode("utf-8")


def desired_plan(
    paths: ServicePaths, api_addr: str, guest_image: str
) -> dict[str, Any]:
    api_addr = validate_api_addr(api_addr)
    guest_image = validate_guest_image(guest_image)
    environment = environment_overrides(paths, api_addr)
    return {
        "schema": SERVICE_SCHEMA,
        "operation": "plan",
        "repo": str(paths.repo),
        "state_root": str(paths.root),
        "api_addr": api_addr,
        "api_url": f"http://{api_addr}",
        "guest_image": guest_image,
        "server_binary": {
            "path": str(paths.binary),
            "exists": paths.binary.is_file(),
            "executable": os.access(paths.binary, os.X_OK),
        },
        "source_config": {
            "path": str(paths.source_config),
            "exists": paths.source_config.is_file(),
        },
        "generated_config": str(paths.config),
        "environment_file": str(paths.environment_sh),
        "credentials_file": str(paths.credentials),
        "credentials_metadata_file": str(paths.credentials_metadata),
        "environment": environment,
        "argv": [str(paths.binary), "--config", str(paths.config)],
        "log": str(paths.server_log),
        "pidfile": str(paths.pidfile),
        "safety": {
            "loopback_only": True,
            "system_install": False,
            "wsl_configuration_changes": False,
            "wsl_shutdown": False,
            "signal_identity": ["pid", "starttime_ticks", "executable"],
        },
    }


def prepare_layout(
    paths: ServicePaths, api_addr: str, guest_image: str
) -> dict[str, Any]:
    _assert_no_state_symlink(paths)
    if not paths.source_config.is_file():
        raise ServiceError(f"AgentENV source config is missing: {paths.source_config}")
    source = paths.source_config.read_bytes()
    rendered = render_config(source, guest_image)
    environment = environment_overrides(paths, api_addr)

    directories = (
        paths.root,
        paths.config.parent,
        paths.home,
        paths.runtime,
        paths.runtime / "xdg",
        paths.xdg_config / "aenv",
        paths.deps,
        paths.tmp,
        paths.cache,
        paths.logs,
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)

    atomic_write(paths.config, rendered, mode=0o600)
    atomic_write(paths.environment_json, canonical_json(environment), mode=0o600)
    atomic_write(paths.environment_sh, shell_environment(environment), mode=0o600)
    credentials = credential_metadata(api_addr)
    cli_credentials = (
        f"url = {json.dumps(credentials['api_url'])}\n"
        f"api_key = {json.dumps(credentials['api_key'])}\n"
    ).encode("utf-8")
    atomic_write(paths.credentials, cli_credentials, mode=0o600)
    atomic_write(paths.credentials_metadata, canonical_json(credentials), mode=0o600)
    return {
        "config_sha256": sha256_bytes(rendered),
        "source_config_sha256": sha256_bytes(source),
        "environment_sha256": sha256_bytes(canonical_json(environment)),
        "credentials_sha256": sha256_bytes(cli_credentials),
        "credentials_metadata_sha256": sha256_bytes(canonical_json(credentials)),
    }


def prerequisite_report() -> dict[str, Any]:
    devices: dict[str, Any] = {}
    for name, path in (("kvm", Path("/dev/kvm")), ("ublk_control", Path("/dev/ublk-control"))):
        try:
            metadata = path.stat()
            present = True
            error = None
        except OSError as exception:
            metadata = None
            present = False
            error = str(exception)
        is_character_device = bool(metadata and stat.S_ISCHR(metadata.st_mode))
        readable = present and os.access(path, os.R_OK)
        writable = present and os.access(path, os.W_OK)
        devices[name] = {
            "path": str(path),
            "present": present,
            "character_device": is_character_device,
            "readable": readable,
            "writable": writable,
            "ready": present and is_character_device and readable and writable,
            "error": error,
        }
    blockers = [name for name, value in devices.items() if not value["ready"]]
    return {"ready": not blockers, "blockers": blockers, "devices": devices}


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    starttime_ticks: int
    executable: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "starttime_ticks": self.starttime_ticks,
            "executable": self.executable,
        }


def process_identity(pid: int) -> ProcessIdentity | None:
    if pid <= 0:
        return None
    directory = Path("/proc") / str(pid)
    try:
        stat_text = (directory / "stat").read_text(encoding="ascii")
        executable = os.readlink(directory / "exe")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, UnicodeError):
        return None
    closing_parenthesis = stat_text.rfind(")")
    if closing_parenthesis < 0:
        return None
    fields = stat_text[closing_parenthesis + 1 :].split()
    if len(fields) <= 19:
        return None
    try:
        starttime_ticks = int(fields[19])
    except ValueError:
        return None
    return ProcessIdentity(
        pid=pid,
        starttime_ticks=starttime_ticks,
        executable=os.path.realpath(executable),
    )


def _expected_executable(paths: ServicePaths) -> str:
    return os.path.realpath(paths.binary)


def _read_process_record(paths: ServicePaths) -> dict[str, Any] | None:
    try:
        raw = paths.pidfile.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ServiceError(f"cannot read AgentENV process record: {error}") from error
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ServiceError(f"invalid AgentENV process record: {paths.pidfile}") from error
    if not isinstance(record, dict) or record.get("schema") != PROCESS_SCHEMA:
        raise ServiceError(f"unsupported AgentENV process record: {paths.pidfile}")
    if not isinstance(record.get("pid"), int) or not isinstance(
        record.get("starttime_ticks"), int
    ):
        raise ServiceError(f"incomplete AgentENV process identity: {paths.pidfile}")
    if not isinstance(record.get("executable"), str):
        raise ServiceError(f"missing AgentENV executable identity: {paths.pidfile}")
    return record


def _record_match(
    paths: ServicePaths, record: dict[str, Any]
) -> tuple[bool, str, ProcessIdentity | None]:
    expected = _expected_executable(paths)
    if os.path.realpath(record["executable"]) != expected:
        return False, "recorded executable differs from configured server binary", None
    current = process_identity(record["pid"])
    if current is None:
        return False, "recorded process no longer exists or is not inspectable", None
    if current.starttime_ticks != record["starttime_ticks"]:
        return False, "PID was reused (Linux start time differs)", current
    if current.executable != expected:
        return False, "live executable differs from the ownership record", current
    return True, "pid, start time, and executable match", current


def health_probe(api_addr: str, timeout: float = 0.5) -> dict[str, Any]:
    api_addr = validate_api_addr(api_addr)
    request = Request(f"http://{api_addr}/health", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(4096)
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "body": raw.decode("utf-8", errors="replace"),
            }
    except HTTPError as error:
        return {"ok": False, "status": error.code, "error": str(error)}
    except (URLError, TimeoutError, OSError) as error:
        return {"ok": False, "error": str(error)}


def service_status(
    paths: ServicePaths, api_addr: str, *, check_health: bool = True
) -> dict[str, Any]:
    _assert_no_state_symlink(paths)
    record = _read_process_record(paths)
    result: dict[str, Any] = {
        "schema": SERVICE_SCHEMA,
        "operation": "status",
        "state_root": str(paths.root),
        "pidfile": str(paths.pidfile),
    }
    if record is None:
        result.update({"state": "stopped", "owned": False, "record": None})
        return result
    matched, reason, current = _record_match(paths, record)
    result.update(
        {
            "state": "running" if matched else "stale",
            "owned": matched,
            "reason": reason,
            "record": record,
            "current_identity": current.as_dict() if current else None,
        }
    )
    if matched and check_health:
        result["api"] = health_probe(api_addr)
    return result


@contextmanager
def control_lock(paths: ServicePaths) -> Iterator[None]:
    _assert_no_state_symlink(paths)
    paths.root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(paths.lockfile, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with os.fdopen(descriptor, "r+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield
    finally:
        # fdopen closes descriptor on both normal and exceptional exits.
        pass


def _validate_start_inputs(paths: ServicePaths) -> None:
    if not paths.binary.is_file():
        raise ServiceError(f"AgentENV server binary is missing: {paths.binary}")
    if not os.access(paths.binary, os.X_OK):
        raise ServiceError(f"AgentENV server binary is not executable: {paths.binary}")
    if not paths.source_cwd.is_dir():
        raise ServiceError(f"AgentENV source directory is missing: {paths.source_cwd}")


def start_service(
    paths: ServicePaths,
    api_addr: str,
    guest_image: str,
    *,
    dry_run: bool,
    allow_missing_prereqs: bool = False,
) -> dict[str, Any]:
    plan = desired_plan(paths, api_addr, guest_image)
    plan["operation"] = "start"
    plan["dry_run"] = dry_run
    prerequisites = prerequisite_report()
    plan["prerequisites"] = prerequisites
    plan["allow_missing_prereqs"] = allow_missing_prereqs
    existing = service_status(paths, api_addr, check_health=False)
    if existing["state"] == "running":
        plan.update({"result": "already-running", "status": existing})
        return plan
    if existing["state"] == "stale":
        raise ServiceError(
            "stale AgentENV process record requires `stop --confirm` before restart"
        )
    if not prerequisites["ready"] and not allow_missing_prereqs:
        if not dry_run:
            names = ", ".join(prerequisites["blockers"])
            raise ServiceError(
                f"AgentENV host prerequisites are unavailable: {names}; "
                "use --allow-missing-prereqs only for controlled diagnostics"
            )
        plan["result"] = "would-refuse-missing-prereqs"
        return plan
    if dry_run:
        plan["result"] = "would-start"
        return plan

    _validate_start_inputs(paths)
    _assert_no_state_symlink(paths)
    with control_lock(paths):
        existing = service_status(paths, api_addr, check_health=False)
        if existing["state"] != "stopped":
            raise ServiceError(f"AgentENV service state changed: {existing['state']}")
        generated = prepare_layout(paths, api_addr, guest_image)
        environment = os.environ.copy()
        environment.update(environment_overrides(paths, api_addr))
        argv = [str(paths.binary), "--config", str(paths.config)]
        log_descriptor = os.open(
            paths.server_log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
        )
        try:
            process = subprocess.Popen(
                argv,
                cwd=paths.source_cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_descriptor,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            os.close(log_descriptor)

        identity = process_identity(process.pid)
        if process.poll() is not None:
            raise ServiceError(
                f"AgentENV server exited during launch; inspect {paths.server_log}"
            )
        expected = _expected_executable(paths)
        if identity is None or identity.executable != expected:
            raise ServiceError(
                "launched AgentENV PID could not be verified; no ownership record was written"
            )
        record = {
            "schema": PROCESS_SCHEMA,
            "pid": identity.pid,
            "starttime_ticks": identity.starttime_ticks,
            "executable": identity.executable,
            "argv": argv,
            "api_addr": validate_api_addr(api_addr),
            "state_root": str(paths.root),
            "started_at": now_utc(),
            "config_sha256": generated["config_sha256"],
            "environment_sha256": generated["environment_sha256"],
        }
        atomic_write(paths.pidfile, canonical_json(record), mode=0o600)

    plan.update({"result": "started", "process": record, "generated": generated})
    return plan


def _remove_record(paths: ServicePaths, record: dict[str, Any], result: str) -> None:
    last_stop = dict(record)
    last_stop.update({"stopped_at": now_utc(), "stop_result": result})
    atomic_write(paths.last_stop, canonical_json(last_stop), mode=0o600)
    paths.pidfile.unlink(missing_ok=True)


def _wait_until_identity_changes(record: dict[str, Any], timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        current = process_identity(record["pid"])
        if current is None:
            return True
        if (
            current.starttime_ticks != record["starttime_ticks"]
            or current.executable != os.path.realpath(record["executable"])
        ):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def stop_service(
    paths: ServicePaths,
    api_addr: str,
    *,
    dry_run: bool,
    confirm: bool,
    timeout: float,
    kill_after_timeout: bool,
) -> dict[str, Any]:
    validate_api_addr(api_addr)
    if timeout < 0:
        raise ServiceError("stop timeout must be non-negative")
    _assert_no_state_symlink(paths)
    record = _read_process_record(paths)
    base: dict[str, Any] = {
        "schema": SERVICE_SCHEMA,
        "operation": "stop",
        "dry_run": dry_run,
        "confirmed": confirm,
        "pidfile": str(paths.pidfile),
    }
    if record is None:
        return {**base, "result": "already-stopped"}
    matched, reason, current = _record_match(paths, record)
    base.update(
        {
            "record": record,
            "owned": matched,
            "reason": reason,
            "current_identity": current.as_dict() if current else None,
        }
    )
    if dry_run:
        base["result"] = "would-signal-sigterm" if matched else "would-clear-stale-record"
        return base
    if not confirm:
        raise ServiceError("stop requires explicit --confirm; no signal was sent")

    with control_lock(paths):
        record = _read_process_record(paths)
        if record is None:
            return {**base, "result": "already-stopped"}
        matched, reason, current = _record_match(paths, record)
        if not matched:
            _remove_record(paths, record, "stale-record-cleared-without-signal")
            return {
                **base,
                "result": "stale-record-cleared-without-signal",
                "owned": False,
                "reason": reason,
                "current_identity": current.as_dict() if current else None,
            }

        # Revalidation above is immediately adjacent to the only signal site.
        os.kill(record["pid"], signal.SIGTERM)
        if _wait_until_identity_changes(record, timeout):
            _remove_record(paths, record, "stopped-after-sigterm")
            return {**base, "result": "stopped-after-sigterm", "owned": True}

        if not kill_after_timeout:
            raise ServiceError(
                "AgentENV did not exit before timeout; ownership record was retained"
            )

        matched, reason, _ = _record_match(paths, record)
        if not matched:
            _remove_record(paths, record, "stopped-before-sigkill")
            return {**base, "result": "stopped-before-sigkill", "owned": True}
        os.kill(record["pid"], signal.SIGKILL)
        if not _wait_until_identity_changes(record, min(max(timeout, 0.1), 5.0)):
            raise ServiceError(
                "AgentENV still appears live after SIGKILL; ownership record was retained"
            )
        _remove_record(paths, record, "stopped-after-sigkill")
        return {**base, "result": "stopped-after-sigkill", "owned": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parents[1]),
        help="integration worktree root",
    )
    parser.add_argument(
        "--server-binary",
        help=(
            "server executable, absolute or relative to --repo "
            f"(default: {DEFAULT_BINARY_RELATIVE})"
        ),
    )
    parser.add_argument("--api-addr", default=DEFAULT_API_ADDR)
    parser.add_argument("--guest-image", default=DEFAULT_GUEST_IMAGE)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("plan", help="print paths, environment, and command")

    status_parser = subparsers.add_parser("status", help="inspect the owned server")
    status_parser.add_argument(
        "--no-health", action="store_true", help="skip the loopback HTTP health probe"
    )

    start_parser = subparsers.add_parser("start", help="start the feature-local server")
    start_parser.add_argument("--dry-run", action="store_true")
    start_parser.add_argument(
        "--allow-missing-prereqs",
        action="store_true",
        help="diagnostic override for unavailable /dev/kvm or /dev/ublk-control",
    )

    stop_parser = subparsers.add_parser("stop", help="stop only the owned server")
    stop_parser.add_argument("--dry-run", action="store_true")
    stop_parser.add_argument(
        "--confirm", action="store_true", help="required before any signal is sent"
    )
    stop_parser.add_argument("--timeout", type=float, default=30.0)
    stop_parser.add_argument(
        "--kill-after-timeout",
        action="store_true",
        help="after revalidating ownership, use SIGKILL if graceful stop times out",
    )
    return parser


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    paths = service_paths(Path(arguments.repo), arguments.server_binary)
    if arguments.command == "plan":
        return desired_plan(paths, arguments.api_addr, arguments.guest_image)
    if arguments.command == "status":
        validate_guest_image(arguments.guest_image)
        return service_status(
            paths, arguments.api_addr, check_health=not arguments.no_health
        )
    if arguments.command == "start":
        return start_service(
            paths,
            arguments.api_addr,
            arguments.guest_image,
            dry_run=arguments.dry_run,
            allow_missing_prereqs=arguments.allow_missing_prereqs,
        )
    if arguments.command == "stop":
        validate_guest_image(arguments.guest_image)
        return stop_service(
            paths,
            arguments.api_addr,
            dry_run=arguments.dry_run,
            confirm=arguments.confirm,
            timeout=arguments.timeout,
            kill_after_timeout=arguments.kill_after_timeout,
        )
    raise ServiceError(f"unsupported command: {arguments.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = run(arguments)
    except ServiceError as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
