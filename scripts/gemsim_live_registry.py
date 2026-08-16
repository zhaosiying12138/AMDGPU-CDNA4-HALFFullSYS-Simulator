#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical rank-launch and lease-backed gemsim live-registry support."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid
from typing import Any, Iterable, Mapping


RANK_LAUNCH_SCHEMA = "amdgpu-sim.gemsim-rank-launch.v1"
LIVE_REGISTRY_SCHEMA = "amdgpu-sim.gemsim-live-registry.v1"
LEASE_SCHEMA = "amdgpu-sim.gemsim-live-registry-lease.v1"
REGISTRY_STATES = frozenset(("STARTING", "READY", "OFF"))
RANK_PATH_KEYS = (
    "instance_directory",
    "triton_cache_directory",
    "runtime_directory",
    "endpoint",
    "gem5_output_directory",
    "dispatch_trace_path",
    "gem5_log_path",
    "gem5_cache_directory",
)
_UUID_PATTERN = re.compile(r"[0-9a-f]{32}")
_ZERO_UUID = "0" * 32
_MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
MIN_GROUP_WORLD_SIZE = 2
MAX_GROUP_WORLD_SIZE = 16
MANAGED_ENDPOINT_BYTES = 108


class RegistryError(RuntimeError):
    pass


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], context: str) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        missing = sorted(wanted - actual)
        extra = sorted(actual - wanted)
        raise RegistryError(f"{context} keys differ: missing={missing} extra={extra}")


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RegistryError(f"{context} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RegistryError(f"{context} must be a nonnegative integer")
    return value


def _canonical_uuid(value: Any, context: str, *, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if not isinstance(value, str) or _UUID_PATTERN.fullmatch(value) is None:
        raise RegistryError(f"{context} must be a canonical lowercase 32-hex UUID")
    if value == _ZERO_UUID:
        raise RegistryError(f"{context} must be nonzero")
    return value


def _absolute_normal_path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RegistryError(f"{context} must be a nonempty path string")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.normpath(value)) != path:
        raise RegistryError(f"{context} must be an absolute normalized path")
    return path


def validate_private_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RegistryError(f"private directory is unavailable: {path}: {error}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise RegistryError(f"private directory is not a real directory: {path}")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise RegistryError(f"private directory must have mode 0700: {path}")
    if metadata.st_uid != os.getuid():
        raise RegistryError(f"private directory is not owned by the current user: {path}")


def ensure_private_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, mode=0o700)
    except FileExistsError:
        validate_private_directory(path)
        return
    os.chmod(path, 0o700, follow_symlinks=False)
    validate_private_directory(path)


def _validate_regular_file_metadata(metadata: os.stat_result, path: Path) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise RegistryError(f"expected a regular file without symlinks: {path}")
    if metadata.st_uid != os.getuid():
        raise RegistryError(f"file is not owned by the current user: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise RegistryError(f"file must not be accessible by group or other: {path}")


def _read_regular_nofollow(path: Path) -> bytes:
    validate_private_directory(path.parent)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise RegistryError(f"refusing symlink: {path}") from error
        raise RegistryError(f"could not open {path}: {error}") from error
    try:
        _validate_regular_file_metadata(os.fstat(descriptor), path)
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > _MAX_DOCUMENT_BYTES:
                raise RegistryError(f"document exceeds size limit: {path}")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _decode_json(data: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RegistryError(f"{context} is not canonical ASCII JSON: {error}") from error
    if not isinstance(value, dict):
        raise RegistryError(f"{context} must be a JSON object")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise RegistryError(f"value cannot be encoded as canonical ASCII JSON: {error}") from error


def _atomic_write(path: Path, data: bytes) -> None:
    validate_private_directory(path.parent)
    try:
        existing = path.lstat()
    except FileNotFoundError:
        pass
    else:
        _validate_regular_file_metadata(existing, path)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def make_rank_launch(
    *,
    job_uuid: str,
    epoch: int,
    rank: int,
    world_size: int,
    instance_directory: Path,
    triton_cache_directory: Path,
) -> dict[str, Any]:
    runtime = instance_directory / "runtime"
    value = {
        "schema": RANK_LAUNCH_SCHEMA,
        "job_uuid": job_uuid,
        "epoch": epoch,
        "rank": rank,
        "world_size": world_size,
        "paths": {
            "instance_directory": str(instance_directory),
            "triton_cache_directory": str(triton_cache_directory),
            "runtime_directory": str(runtime),
            "endpoint": str(runtime / "bridge.sock"),
            "gem5_output_directory": str(runtime / "m5out"),
            "dispatch_trace_path": str(runtime / "dispatch-trace.jsonl"),
            "gem5_log_path": str(runtime / "gem5.log"),
            "gem5_cache_directory": str(runtime / "cache"),
        },
    }
    validate_rank_launch(value)
    return value


def validate_rank_launch(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError("rank launch descriptor must be an object")
    _exact_keys(
        value,
        ("schema", "job_uuid", "epoch", "rank", "world_size", "paths"),
        "rank launch descriptor",
    )
    if value["schema"] != RANK_LAUNCH_SCHEMA:
        raise RegistryError("rank launch descriptor schema mismatch")
    _canonical_uuid(value["job_uuid"], "rank launch job_uuid")
    _positive_integer(value["epoch"], "rank launch epoch")
    world = _positive_integer(value["world_size"], "rank launch world_size")
    if not MIN_GROUP_WORLD_SIZE <= world <= MAX_GROUP_WORLD_SIZE:
        raise RegistryError("rank launch world_size must be in 2..16")
    rank = _nonnegative_integer(value["rank"], "rank launch rank")
    if rank >= world:
        raise RegistryError("rank launch rank must be less than world_size")
    paths = value["paths"]
    if not isinstance(paths, Mapping):
        raise RegistryError("rank launch paths must be an object")
    _exact_keys(paths, RANK_PATH_KEYS, "rank launch paths")
    normalized = {
        key: _absolute_normal_path(paths[key], f"rank launch paths.{key}")
        for key in RANK_PATH_KEYS
    }
    instance = normalized["instance_directory"]
    runtime = normalized["runtime_directory"]
    if runtime.parent != instance:
        raise RegistryError("runtime_directory must be directly inside instance_directory")
    if normalized["triton_cache_directory"].parent.parent != instance.parent:
        raise RegistryError("triton cache must be private to its instance namespace")
    for key in (
        "endpoint",
        "gem5_output_directory",
        "dispatch_trace_path",
        "gem5_log_path",
        "gem5_cache_directory",
    ):
        if normalized[key].parent != runtime:
            raise RegistryError(f"{key} must be directly inside runtime_directory")
    if len(os.fsencode(normalized["endpoint"])) >= MANAGED_ENDPOINT_BYTES:
        raise RegistryError(
            "rank launch endpoint exceeds the managed runtime path limit"
        )
    return dict(value)


def validate_rank_launch_group(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    descriptors = [validate_rank_launch(value) for value in values]
    if len(descriptors) < 2:
        raise RegistryError("group launch requires at least two ranks")
    first = descriptors[0]
    identity = (first["job_uuid"], first["epoch"], first["world_size"])
    if len(descriptors) != first["world_size"]:
        raise RegistryError("group launch descriptor count does not equal world_size")
    for value in descriptors:
        current = (value["job_uuid"], value["epoch"], value["world_size"])
        if current != identity:
            raise RegistryError("group launch descriptors have mixed identity")
    ranks = [value["rank"] for value in descriptors]
    if sorted(ranks) != list(range(first["world_size"])):
        raise RegistryError("group launch ranks must be exactly 0..world_size-1 once")
    for key in RANK_PATH_KEYS:
        paths = [value["paths"][key] for value in descriptors]
        if len(paths) != len(set(paths)):
            raise RegistryError(f"group launch path is shared across ranks: {key}")
    return sorted(descriptors, key=lambda value: value["rank"])


def write_rank_launch(path: Path, value: Mapping[str, Any]) -> None:
    validate_rank_launch(value)
    _atomic_write(path, _canonical_json(value))
    os.chmod(path, 0o400, follow_symlinks=False)


def load_rank_launch(path: Path) -> dict[str, Any]:
    data = _read_regular_nofollow(path)
    value = validate_rank_launch(_decode_json(data, str(path)))
    if data != _canonical_json(value):
        raise RegistryError("rank launch descriptor is not canonical ASCII JSON")
    return value


def _validate_registry_rank(value: Mapping[str, Any], state: str, world: int) -> None:
    _exact_keys(
        value,
        (
            "rank",
            "world_size",
            "state",
            "worker_pid",
            "daemon_pid",
            "daemon_uuid",
            "endpoint",
            "runtime_directory",
            "triton_cache_directory",
            "gem5_cache_directory",
        ),
        "live registry rank",
    )
    rank = _nonnegative_integer(value["rank"], "live registry rank.rank")
    if rank >= world or value["world_size"] != world:
        raise RegistryError("live registry rank/world identity mismatch")
    if value["state"] != state:
        raise RegistryError("live registry rank state does not match registry state")
    worker_pid = value["worker_pid"]
    if worker_pid is not None:
        _positive_integer(worker_pid, "live registry rank.worker_pid")
    daemon_pid = value["daemon_pid"]
    daemon_uuid = value["daemon_uuid"]
    if state == "READY":
        _positive_integer(worker_pid, "READY rank worker_pid")
        _positive_integer(daemon_pid, "READY rank daemon_pid")
        _canonical_uuid(daemon_uuid, "READY rank daemon_uuid")
    else:
        if daemon_pid is not None:
            _positive_integer(daemon_pid, "live registry rank.daemon_pid")
        _canonical_uuid(daemon_uuid, "live registry rank.daemon_uuid", allow_none=True)
    for key in (
        "endpoint",
        "runtime_directory",
        "triton_cache_directory",
        "gem5_cache_directory",
    ):
        _absolute_normal_path(value[key], f"live registry rank.{key}")


def validate_live_registry(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError("live registry must be an object")
    _exact_keys(
        value,
        (
            "schema",
            "generation",
            "state",
            "updated_at_ns",
            "supervisor_pid",
            "supervisor_start_time",
            "lease_path",
            "job_uuid",
            "epoch",
            "world_size",
            "ranks",
        ),
        "live registry",
    )
    if value["schema"] != LIVE_REGISTRY_SCHEMA:
        raise RegistryError("live registry schema mismatch")
    _positive_integer(value["generation"], "live registry generation")
    state = value["state"]
    if state not in REGISTRY_STATES:
        raise RegistryError(f"invalid live registry state: {state}")
    _positive_integer(value["updated_at_ns"], "live registry updated_at_ns")
    _positive_integer(value["supervisor_pid"], "live registry supervisor_pid")
    _positive_integer(value["supervisor_start_time"], "live registry supervisor_start_time")
    _absolute_normal_path(value["lease_path"], "live registry lease_path")
    _canonical_uuid(value["job_uuid"], "live registry job_uuid")
    _positive_integer(value["epoch"], "live registry epoch")
    world = _positive_integer(value["world_size"], "live registry world_size")
    ranks = value["ranks"]
    if not isinstance(ranks, list) or len(ranks) != world:
        raise RegistryError("live registry ranks must contain exactly world_size entries")
    for rank_value in ranks:
        if not isinstance(rank_value, Mapping):
            raise RegistryError("live registry rank entry must be an object")
        _validate_registry_rank(rank_value, state, world)
    observed = [rank_value["rank"] for rank_value in ranks]
    if observed != list(range(world)):
        raise RegistryError("live registry ranks must be sorted and exactly 0..world_size-1")
    return dict(value)


def _validate_lease(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        value,
        ("schema", "session_uuid", "registry_generation", "registry_sha256"),
        "live registry lease",
    )
    if value["schema"] != LEASE_SCHEMA:
        raise RegistryError("live registry lease schema mismatch")
    _canonical_uuid(value["session_uuid"], "live registry lease session_uuid")
    _positive_integer(value["registry_generation"], "lease registry_generation")
    digest = value["registry_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RegistryError("lease registry_sha256 must be lowercase SHA-256")
    return dict(value)


def _proc_start_time(pid: int) -> int:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError as error:
        raise RegistryError(f"could not read supervisor process identity: {error}") from error
    close = text.rfind(")")
    fields = text[close + 2 :].split() if close >= 0 else []
    if len(fields) <= 19:
        raise RegistryError("could not parse supervisor process identity")
    return _positive_integer(int(fields[19]), "supervisor process start time")


@dataclass(frozen=True)
class LiveSnapshot:
    registry: dict[str, Any]
    lease_held: bool

    @property
    def effective_status(self) -> str:
        return "ON" if self.lease_held and self.registry["state"] == "READY" else "OFF"


class LiveRegistryPublisher:
    def __init__(self, registry_path: Path):
        self.registry_path = Path(os.path.abspath(registry_path))
        self.lease_path = self.registry_path.with_name(self.registry_path.name + ".lease")
        validate_private_directory(self.registry_path.parent)
        for path in (self.registry_path, self.lease_path):
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            raise RegistryError(f"live registry path must not already exist: {path} ({metadata.st_mode:o})")
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self._lease_fd = os.open(self.lease_path, flags, 0o600)
        _validate_regular_file_metadata(os.fstat(self._lease_fd), self.lease_path)
        fcntl.flock(self._lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.session_uuid = uuid.uuid4().hex
        self.generation = 0
        self.closed = False

    def publish(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if self.closed:
            raise RegistryError("live registry publisher is closed")
        candidate = dict(value)
        expected = self.generation + 1
        if candidate.get("generation") != expected:
            raise RegistryError(
                f"live registry generation must advance exactly once: expected {expected}"
            )
        validated = validate_live_registry(candidate)
        if Path(validated["lease_path"]) != self.lease_path:
            raise RegistryError("live registry lease_path does not match held lease")
        encoded = _canonical_json(validated)
        _atomic_write(self.registry_path, encoded)
        lease = {
            "schema": LEASE_SCHEMA,
            "session_uuid": self.session_uuid,
            "registry_generation": expected,
            "registry_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        lease_data = _canonical_json(lease)
        os.lseek(self._lease_fd, 0, os.SEEK_SET)
        os.ftruncate(self._lease_fd, 0)
        offset = 0
        while offset < len(lease_data):
            offset += os.write(self._lease_fd, lease_data[offset:])
        os.fsync(self._lease_fd)
        self.generation = expected
        return validated

    def base_document(
        self,
        *,
        state: str,
        job_uuid: str,
        epoch: int,
        world_size: int,
        ranks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema": LIVE_REGISTRY_SCHEMA,
            "generation": self.generation + 1,
            "state": state,
            "updated_at_ns": time.time_ns(),
            "supervisor_pid": os.getpid(),
            "supervisor_start_time": _proc_start_time(os.getpid()),
            "lease_path": str(self.lease_path),
            "job_uuid": job_uuid,
            "epoch": epoch,
            "world_size": world_size,
            "ranks": ranks,
        }

    def close(self) -> None:
        if self.closed:
            return
        fcntl.flock(self._lease_fd, fcntl.LOCK_UN)
        os.close(self._lease_fd)
        self.closed = True

    def __enter__(self) -> "LiveRegistryPublisher":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def _read_lease(path: Path) -> tuple[dict[str, Any], bool]:
    validate_private_directory(path.parent)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise RegistryError(f"refusing symlink: {path}") from error
        raise RegistryError(f"could not open registry lease {path}: {error}") from error
    try:
        _validate_regular_file_metadata(os.fstat(descriptor), path)
        lease_held = True
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            lease_held = False
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.lseek(descriptor, 0, os.SEEK_SET)
        data = os.read(descriptor, _MAX_DOCUMENT_BYTES + 1)
        if len(data) > _MAX_DOCUMENT_BYTES:
            raise RegistryError("registry lease exceeds size limit")
        return _validate_lease(_decode_json(data, str(path))), lease_held
    finally:
        os.close(descriptor)


def read_live_snapshot(registry_path: Path, retries: int = 20) -> LiveSnapshot:
    registry_path = Path(os.path.abspath(registry_path))
    last_error: RegistryError | None = None
    for _attempt in range(retries):
        try:
            first_data = _read_regular_nofollow(registry_path)
            registry = validate_live_registry(_decode_json(first_data, str(registry_path)))
            lease_path = Path(registry["lease_path"])
            if lease_path != registry_path.with_name(registry_path.name + ".lease"):
                raise RegistryError("registry lease_path is outside the canonical namespace")
            lease, held = _read_lease(lease_path)
            second_data = _read_regular_nofollow(registry_path)
            if first_data != second_data:
                raise RegistryError("live registry changed during read")
            digest = hashlib.sha256(first_data).hexdigest()
            if lease["registry_generation"] != registry["generation"]:
                raise RegistryError("live registry generation rollback or torn publish detected")
            if lease["registry_sha256"] != digest:
                raise RegistryError("live registry tamper or torn publish detected")
            return LiveSnapshot(registry=registry, lease_held=held)
        except RegistryError as error:
            last_error = error
            time.sleep(0.005)
    assert last_error is not None
    raise last_error
