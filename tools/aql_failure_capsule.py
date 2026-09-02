#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Capture and verify a host-native ROCr/AQL failure capsule.

The capture side is intentionally outside gem5.  It consumes the generic KMT
trace emitted by ``SAGR_HSAKMT_MODEL_TRACE=1``, reads the stopped producer's
shared backing/process memory, and freezes the queue head plus every resource
that can be identified without knowing a framework, operator, kernel name, or
program counter.

This is a diagnostic snapshot, not an executable replay format.  The manifest
states that boundary explicitly and the verifier checks every frozen byte.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, field
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import struct
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


CAPSULE_SCHEMA = "amdgpu-sim.aql-failure-capsule.v1"
REGISTRY_SCHEMA = "amdgpu-sim.kmt-allocation-snapshot.v1"
QUEUE_SCHEMA = "amdgpu-sim.aql-queue-snapshot.v1"
PACKET_SCHEMA = "amdgpu-sim.aql-packet-snapshot.v1"
MANIFEST_NAME = "manifest.json"
MANIFEST_SHA_NAME = "manifest.sha256"
PACKET_BYTES = 64
DESCRIPTOR_BYTES = 64
LEGACY_V2_DESCRIPTOR_BYTES = 256
SIGNAL_BYTES = 64
MQD_BYTES = 256
MQD_READ_INDEX_OFFSET = 128
USERPTR_FLAG = 4
KERNEL_DISPATCH_PACKET_TYPE = 2
PACKET_TYPE_MASK = 0xFF
PACKET_HEADER_RESERVED_MASK = 0xE000
KERNEL_SETUP_DIMENSIONS_MASK = 0x3
KERNEL_SETUP_RESERVED_MASK = 0xFFFC
DOORBELL_INITIAL_VALUE = (1 << 64) - 1
COMPLETION_OFFSET_FROM_DOORBELL = 1024
DEFAULT_MAX_HASH_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_CODE_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_KERNARG_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_LOG_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_TOTAL_LOG_BYTES = 512 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
AT_FDCWD = -100
RENAME_NOREPLACE = 1
HEX64_RE = re.compile(r"[0-9a-f]{64}")
ROLE_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}")
LOG_ROLE_RE = re.compile(r"[a-z][a-z0-9_-]{0,59}")
REPLAY_BLOCKERS = (
    "simulator_event_queue_and_cu_pipeline_state_not_captured",
    "allocation_mutation_history_and_external_host_state_not_captured",
    "no_generation_safe_resource_reinstantiation_contract",
    "no_verified_completion_signal_or_output_oracle_for_replay",
)

ALLOC_RE = re.compile(
    r"^hsakmt-model pid=(?P<pid>[1-9][0-9]*) phase=leave "
    r"request=0x(?P<request>[0-9a-f]+) result=0 errno=0 "
    r"gpu_id=(?P<gpu>[0-9]+) flags=0x(?P<flags>[0-9a-f]+) "
    r"va=0x(?P<va>[0-9a-f]+) size=(?P<size>[1-9][0-9]*) "
    r"mmap_offset=0x(?P<offset>[0-9a-f]+) "
    r"handle=(?P<handle>[1-9][0-9]*)$"
)
QUEUE_CREATE_RE = re.compile(
    r"^hsakmt-model pid=(?P<pid>[1-9][0-9]*) phase=leave "
    r"request=0x(?P<request>[0-9a-f]+) result=0 errno=0 "
    r"gpu_id=(?P<gpu>[0-9]+) queue_type=(?P<type>[0-9]+) "
    r"ring=0x(?P<ring>[0-9a-f]+) ring_size=(?P<size>[1-9][0-9]*) "
    r"read=0x(?P<read>[0-9a-f]+) write=0x(?P<write>[0-9a-f]+) "
    r"percentage=(?P<percentage>[0-9]+) priority=(?P<priority>[0-9]+) "
    r"queue_id=(?P<queue>[1-9][0-9]*) "
    r"doorbell_offset=0x(?P<doorbell>[0-9a-f]+)$"
)
QUEUE_UPDATE_RE = re.compile(
    r"^hsakmt-model pid=(?P<pid>[1-9][0-9]*) phase=leave "
    r"request=0x(?P<request>[0-9a-f]+) result=0 errno=0 "
    r"queue_id=(?P<queue>[1-9][0-9]*) ring=0x(?P<ring>[0-9a-f]+) "
    r"ring_size=(?P<size>[1-9][0-9]*) percentage=(?P<percentage>[0-9]+) "
    r"priority=(?P<priority>[0-9]+)$"
)
QUEUE_DESTROY_RE = re.compile(
    r"^hsakmt-model pid=(?P<pid>[1-9][0-9]*) phase=leave "
    r"request=0x(?P<request>[0-9a-f]+) result=0 errno=0 "
    r"queue_id=(?P<queue>[1-9][0-9]*)$"
)
QUEUE_PROGRESS_RE = re.compile(
    r"^hsakmt-model pid=(?P<pid>[1-9][0-9]*) "
    r"phase=(?P<phase>queue-doorbell|queue-retired) "
    r"queue_id=(?P<queue>[1-9][0-9]*) slot=(?P<slot>[0-9]+) "
    r"doorbell=(?P<doorbell>[0-9]+) notification=(?P<notification>[0-9]+) "
    r"completion=(?P<completion>[0-9]+) status=(?P<status>-?[0-9]+)$"
)


class CapsuleError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CapsuleError(message)


def canonical_json(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise CapsuleError(f"value is not canonical JSON: {error}") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def crc32c(payload: bytes) -> int:
    value = 0xFFFFFFFF
    for byte in payload:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
    return value ^ 0xFFFFFFFF


def checked_range(base: int, count: int, label: str) -> tuple[int, int]:
    require(type(base) is int and base >= 0, f"{label} base is invalid")
    require(type(count) is int and count > 0, f"{label} size is invalid")
    end = base + count
    require(end <= (1 << 64), f"{label} range overflows uint64")
    return base, end


@dataclass(frozen=True)
class Allocation:
    line_number: int
    ordinal: int
    pid: int
    gpu_id: int
    flags: int
    gpu_va: int
    byte_count: int
    mmap_offset: int
    handle: int

    @property
    def userptr(self) -> bool:
        return (self.flags & USERPTR_FLAG) != 0

    def contains(self, address: int, count: int) -> bool:
        if count <= 0 or address < self.gpu_va:
            return False
        offset = address - self.gpu_va
        return offset <= self.byte_count and count <= self.byte_count - offset

    def identity(self) -> dict[str, Any]:
        return {
            "allocation_id": self.handle,
            "allocation_log_ordinal": self.ordinal,
            "allocation_log_line": self.line_number,
            "generation": None,
            "generation_status": "not_exposed_by_hsakmt_ioctl_trace",
            "liveness_status": "not_reconstructable_because_free_trace_omits_handle",
            "gpu_id": self.gpu_id,
            "gpu_va": self.gpu_va,
            "bytes": self.byte_count,
            "flags": self.flags,
            "storage": "process_userptr" if self.userptr else "shared_memfd",
            "storage_offset": self.mmap_offset,
        }


@dataclass
class Queue:
    line_number: int
    pid: int
    gpu_id: int
    queue_type: int
    queue_id: int
    ring_va: int
    ring_bytes: int
    read_pointer_va: int
    write_pointer_va: int
    doorbell_offset: int
    percentage: int
    priority: int
    active: bool = True
    progress: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Registry:
    pid: int
    allocations: tuple[Allocation, ...]
    queues: tuple[Queue, ...]
    complete_lines: int
    trailing_partial_line: bool

    def resolve(self, address: int, count: int, label: str) -> Allocation:
        checked_range(address, count, label)
        candidates = [
            allocation
            for allocation in self.allocations
            if allocation.contains(address, count)
        ]
        require(candidates, f"{label} is outside every observed KMT allocation")
        # HostGPUKmtState rejects overlapping live mappings.  Therefore the
        # newest trace occurrence is the only legal current owner when logs
        # retain older, already-freed allocations.
        return max(candidates, key=lambda allocation: allocation.line_number)


def parse_registry(payload: bytes, expected_pid: int | None) -> Registry:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise CapsuleError(f"worker KMT trace is not UTF-8: {error}") from error
    trailing_partial = bool(text) and not text.endswith("\n")
    lines = text.splitlines()
    parsed_lines = lines[:-1] if trailing_partial else lines
    allocations: list[Allocation] = []
    queues: dict[int, Queue] = {}
    observed_pids: set[int] = set()
    ordinal = 0
    for line_number, line in enumerate(parsed_lines, 1):
        match = ALLOC_RE.fullmatch(line)
        if match:
            pid = int(match["pid"])
            if expected_pid is not None and pid != expected_pid:
                continue
            observed_pids.add(pid)
            ordinal += 1
            allocations.append(
                Allocation(
                    line_number=line_number,
                    ordinal=ordinal,
                    pid=pid,
                    gpu_id=int(match["gpu"]),
                    flags=int(match["flags"], 16),
                    gpu_va=int(match["va"], 16),
                    byte_count=int(match["size"]),
                    mmap_offset=int(match["offset"], 16),
                    handle=int(match["handle"]),
                )
            )
            continue
        match = QUEUE_CREATE_RE.fullmatch(line)
        if match:
            pid = int(match["pid"])
            if expected_pid is not None and pid != expected_pid:
                continue
            observed_pids.add(pid)
            queue_id = int(match["queue"])
            queues[queue_id] = Queue(
                line_number=line_number,
                pid=pid,
                gpu_id=int(match["gpu"]),
                queue_type=int(match["type"]),
                queue_id=queue_id,
                ring_va=int(match["ring"], 16),
                ring_bytes=int(match["size"]),
                read_pointer_va=int(match["read"], 16),
                write_pointer_va=int(match["write"], 16),
                doorbell_offset=int(match["doorbell"], 16),
                percentage=int(match["percentage"]),
                priority=int(match["priority"]),
            )
            continue
        match = QUEUE_UPDATE_RE.fullmatch(line)
        if match:
            pid = int(match["pid"])
            if expected_pid is not None and pid != expected_pid:
                continue
            observed_pids.add(pid)
            queue = queues.get(int(match["queue"]))
            if queue is not None:
                queue.line_number = line_number
                queue.ring_va = int(match["ring"], 16)
                queue.ring_bytes = int(match["size"])
                queue.percentage = int(match["percentage"])
                queue.priority = int(match["priority"])
            continue
        match = QUEUE_DESTROY_RE.fullmatch(line)
        if match:
            pid = int(match["pid"])
            if expected_pid is not None and pid != expected_pid:
                continue
            observed_pids.add(pid)
            queue = queues.get(int(match["queue"]))
            if queue is not None:
                queue.active = False
                queue.line_number = line_number
            continue
        match = QUEUE_PROGRESS_RE.fullmatch(line)
        if match:
            pid = int(match["pid"])
            if expected_pid is not None and pid != expected_pid:
                continue
            observed_pids.add(pid)
            queue = queues.get(int(match["queue"]))
            if queue is not None:
                queue.progress.append(
                    {
                        "line": line_number,
                        "phase": match["phase"],
                        "slot": int(match["slot"]),
                        "doorbell": int(match["doorbell"]),
                        "notification": int(match["notification"]),
                        "completion": int(match["completion"]),
                        "status": int(match["status"]),
                    }
                )

    require(allocations, "worker log contains no successful KMT allocations")
    require(queues, "worker log contains no successful KMT queue creation")
    require(len(observed_pids) == 1, "worker log mixes KMT process identities")
    pid = next(iter(observed_pids))
    if expected_pid is not None:
        require(pid == expected_pid, "worker log PID differs from requested PID")
    return Registry(
        pid=pid,
        allocations=tuple(allocations),
        queues=tuple(sorted(queues.values(), key=lambda item: item.queue_id)),
        complete_lines=len(parsed_lines),
        trailing_partial_line=trailing_partial,
    )


def process_start_time(proc_dir: Path) -> int:
    payload = (proc_dir / "stat").read_text(encoding="ascii", errors="strict")
    close = payload.rfind(")")
    require(close >= 0, "process stat has no command terminator")
    fields = payload[close + 2 :].split()
    require(len(fields) >= 20 and fields[19].isdigit(), "process stat is truncated")
    return int(fields[19])


def process_state(proc_dir: Path) -> str:
    text = (proc_dir / "status").read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("State:"):
            fields = line.split()
            require(len(fields) >= 2 and len(fields[1]) == 1, "invalid process state")
            return fields[1]
    raise CapsuleError("process status has no State field")


def stable_file_prefix(path: Path, max_bytes: int) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CapsuleError(f"could not open {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"source is not a regular file: {path}")
        require(before.st_size <= max_bytes, f"source exceeds capture budget: {path}")
        remaining = before.st_size
        chunks: list[bytes] = []
        offset = 0
        while remaining:
            chunk = os.pread(descriptor, min(COPY_CHUNK_BYTES, remaining), offset)
            require(chunk, f"source shrank while reading: {path}")
            chunks.append(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
            and after.st_size >= before.st_size,
            f"source identity changed while reading: {path}",
        )
        return b"".join(chunks), {
            "source": str(path.resolve()),
            "device": before.st_dev,
            "inode": before.st_ino,
            "captured_prefix_bytes": before.st_size,
            "size_after": after.st_size,
            "append_observed": after.st_size != before.st_size,
        }
    finally:
        os.close(descriptor)


class PidFd:
    """Identity-stable signaling for the exact live Linux process."""

    def __init__(self, pid: int):
        libc = ctypes.CDLL(None, use_errno=True)
        pidfd_open = getattr(libc, "pidfd_open", None)
        pidfd_send_signal = getattr(libc, "pidfd_send_signal", None)
        require(
            pidfd_open is not None and pidfd_send_signal is not None,
            "pidfd support is required for --freeze-target",
        )
        pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
        pidfd_open.restype = ctypes.c_int
        pidfd_send_signal.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        pidfd_send_signal.restype = ctypes.c_int
        descriptor = pidfd_open(pid, 0)
        if descriptor < 0:
            error_number = ctypes.get_errno()
            raise CapsuleError(
                f"could not open pidfd for process {pid}: {os.strerror(error_number)}"
            )
        self.fd = descriptor
        self.pid = pid
        self._send = pidfd_send_signal

    def signal(self, signal_number: int, *, ignore_missing: bool = False) -> None:
        result = self._send(self.fd, signal_number, None, 0)
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if ignore_missing and error_number == errno.ESRCH:
            return
        raise CapsuleError(
            f"pidfd signal {signal_number} for process {self.pid} failed: "
            f"{os.strerror(error_number)}"
        )

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class ByteSource:
    def __init__(self, path: Path, label: str):
        self.path = path
        self.label = label
        try:
            self.fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        except OSError as error:
            raise CapsuleError(f"could not open {label} {path}: {error}") from error
        metadata = os.fstat(self.fd)
        require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file")
        self.identity = {
            "path": str(path),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "bytes": metadata.st_size,
            "mode": stat.S_IMODE(metadata.st_mode),
        }

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def read_at(self, offset: int, count: int) -> bytes:
        checked_range(offset, count, self.label)
        chunks: list[bytes] = []
        completed = 0
        while completed < count:
            try:
                chunk = os.pread(
                    self.fd,
                    min(COPY_CHUNK_BYTES, count - completed),
                    offset + completed,
                )
            except OSError as error:
                raise CapsuleError(
                    f"could not read {self.label} at 0x{offset + completed:x}: {error}"
                ) from error
            require(chunk, f"short read from {self.label} at 0x{offset + completed:x}")
            chunks.append(chunk)
            completed += len(chunk)
        return b"".join(chunks)

    def stream(self, offset: int, count: int) -> Iterable[bytes]:
        checked_range(offset, count, self.label)
        completed = 0
        while completed < count:
            chunk = self.read_at(offset + completed, min(COPY_CHUNK_BYTES, count - completed))
            completed += len(chunk)
            yield chunk


class GPUMemory:
    def __init__(
        self,
        registry: Registry,
        backing: ByteSource,
        process_memory: ByteSource | None,
    ):
        self.registry = registry
        self.backing = backing
        self.process_memory = process_memory

    def source_range(self, allocation: Allocation, address: int, count: int) -> tuple[ByteSource, int]:
        require(allocation.contains(address, count), "allocation range mismatch")
        relative = address - allocation.gpu_va
        if allocation.userptr:
            require(self.process_memory is not None, "userptr capture requires readable process memory")
            return self.process_memory, allocation.mmap_offset + relative
        return self.backing, allocation.mmap_offset + relative

    def read(self, address: int, count: int, label: str) -> tuple[bytes, Allocation]:
        allocation = self.registry.resolve(address, count, label)
        source, offset = self.source_range(allocation, address, count)
        return source.read_at(offset, count), allocation

    def hash_allocation(self, allocation: Allocation) -> str:
        source, offset = self.source_range(allocation, allocation.gpu_va, allocation.byte_count)
        digest = hashlib.sha256()
        for chunk in source.stream(offset, allocation.byte_count):
            digest.update(chunk)
        return digest.hexdigest()

    def copy_allocation(self, allocation: Allocation, destination: Path) -> str:
        source, offset = self.source_range(allocation, allocation.gpu_va, allocation.byte_count)
        digest = hashlib.sha256()
        with destination.open("xb") as stream:
            os.chmod(destination, 0o600)
            for chunk in source.stream(offset, allocation.byte_count):
                stream.write(chunk)
                digest.update(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        return digest.hexdigest()


class CapsuleDirectory:
    def __init__(self, output: Path):
        self.output = output.resolve()
        require(not self.output.exists(), f"output already exists: {self.output}")
        self.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        os.chmod(self.stage, 0o700)
        self.artifacts: list[dict[str, Any]] = []
        self.committed = False

    def close(self) -> None:
        if not self.committed and self.stage.exists():
            shutil.rmtree(self.stage)

    def _target(self, relative: str) -> Path:
        require(relative and not relative.startswith("/"), "artifact path is not relative")
        parts = Path(relative).parts
        require(all(part not in ("", ".", "..") for part in parts), "artifact path escapes capsule")
        target = self.stage.joinpath(*parts)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        return target

    def add_bytes(self, relative: str, payload: bytes, role: str) -> dict[str, Any]:
        target = self._target(relative)
        with target.open("xb") as stream:
            os.chmod(target, 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        record = {
            "path": relative,
            "role": role,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
        self.artifacts.append(record)
        return record

    def add_json(self, relative: str, value: object, role: str) -> dict[str, Any]:
        return self.add_bytes(relative, canonical_json(value), role)

    def reserve_path(self, relative: str) -> Path:
        return self._target(relative)

    def record_path(self, relative: str, role: str) -> dict[str, Any]:
        target = self._target(relative)
        metadata = target.stat()
        record = {
            "path": relative,
            "role": role,
            "bytes": metadata.st_size,
            "sha256": file_sha256(target),
        }
        self.artifacts.append(record)
        return record

    def commit(self, manifest: Mapping[str, Any]) -> None:
        payload = canonical_json(manifest)
        manifest_path = self.stage / MANIFEST_NAME
        digest = sha256_bytes(payload)
        sha_path = self.stage / MANIFEST_SHA_NAME
        controls = (
            (manifest_path, payload),
            (sha_path, f"{digest}  {MANIFEST_NAME}\n".encode("ascii")),
        )
        for path, contents in controls:
            with path.open("xb") as stream:
                os.chmod(path, 0o600)
                stream.write(contents)
                stream.flush()
                os.fsync(stream.fileno())
        stage_fd = os.open(self.stage, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        require(renameat2 is not None, "renameat2 is required for absent-only publication")
        result = renameat2(
            AT_FDCWD,
            os.fsencode(self.stage),
            AT_FDCWD,
            os.fsencode(self.output),
            RENAME_NOREPLACE,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise CapsuleError(f"output already exists: {self.output}")
            raise CapsuleError(
                f"could not publish capsule {self.output}: {os.strerror(error_number)}"
            )
        parent_fd = os.open(
            self.output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        self.committed = True


def decode_packet(
    payload: bytes, *, require_kernel_dispatch: bool = True
) -> dict[str, Any]:
    require(len(payload) == PACKET_BYTES, "AQL packet is not 64 bytes")
    fields = struct.unpack("<6H5I4Q", payload)
    packet_type = fields[0] & PACKET_TYPE_MASK
    value = {
        "schema": PACKET_SCHEMA,
        "packet_type": packet_type,
        "header": fields[0],
        "sha256": sha256_bytes(payload),
        "crc32c": crc32c(payload),
    }
    if packet_type != KERNEL_DISPATCH_PACKET_TYPE:
        require(
            not require_kernel_dispatch,
            f"queue head packet type {packet_type} is not a kernel dispatch",
        )
        value["resource_decode"] = "unsupported_non_kernel_packet"
        return value
    value.update(
        {
            "resource_decode": "kernel_dispatch",
            "setup": fields[1],
            "workgroup": [fields[2], fields[3], fields[4]],
            "reserved0": fields[5],
            "grid": [fields[6], fields[7], fields[8]],
            "private_segment_size": fields[9],
            "group_segment_size": fields[10],
            "kernel_object": fields[11],
            "kernarg_address": fields[12],
            "reserved1": fields[13],
            "completion_signal": fields[14],
        }
    )
    dimensions = fields[1] & KERNEL_SETUP_DIMENSIONS_MASK
    require(fields[0] & PACKET_HEADER_RESERVED_MASK == 0, "AQL header reserved bits are set")
    require(fields[1] & KERNEL_SETUP_RESERVED_MASK == 0, "AQL setup reserved bits are set")
    require(dimensions in (1, 2, 3), "AQL grid dimension count is invalid")
    require(value["reserved0"] == 0 and value["reserved1"] == 0, "AQL reserved fields are nonzero")
    require(value["kernel_object"] != 0, "AQL packet has no kernel object")
    require(value["kernel_object"] % 64 == 0, "AQL kernel object is not 64-byte aligned")
    require(value["kernarg_address"] % 8 == 0, "AQL kernarg is not 8-byte aligned")
    require(all(item > 0 for item in value["workgroup"]), "AQL workgroup is empty")
    require(all(item > 0 for item in value["grid"]), "AQL grid is empty")
    require(
        all(item == 1 for item in value["grid"][dimensions:]),
        "AQL unused grid dimensions are not one",
    )
    return value


def versioned_v2_descriptor(prefix: bytes) -> bool:
    require(len(prefix) >= DESCRIPTOR_BYTES, "kernel descriptor prefix is truncated")
    version_major, version_minor = struct.unpack_from("<II", prefix, 0)
    machine, machine_major, machine_minor, machine_stepping = struct.unpack_from(
        "<4H", prefix, 8
    )
    return (
        version_major == 1
        and version_minor <= 2
        and machine == 1
        and machine_major == 9
        and machine_minor == 5
        and machine_stepping == 0
    )


def decode_descriptor(payload: bytes, descriptor_va: int) -> dict[str, Any]:
    require(len(payload) >= DESCRIPTOR_BYTES, "kernel descriptor is shorter than 64 bytes")
    if versioned_v2_descriptor(payload[:DESCRIPTOR_BYTES]):
        require(
            len(payload) == LEGACY_V2_DESCRIPTOR_BYTES,
            "versioned code-object V2 descriptor is not 256 bytes",
        )
        entry_offset = struct.unpack_from("<q", payload, 16)[0]
        private_bytes = struct.unpack_from("<I", payload, 60)[0]
        group_bytes = struct.unpack_from("<I", payload, 64)[0]
        gds_bytes = struct.unpack_from("<I", payload, 68)[0]
        kernarg_bytes = struct.unpack_from("<Q", payload, 72)[0]
        require(gds_bytes == 0 and kernarg_bytes <= 0xFFFFFFFF, "invalid code-object V2 descriptor")
        require(entry_offset >= LEGACY_V2_DESCRIPTOR_BYTES and entry_offset % 256 == 0, "invalid code-object V2 entry")
        descriptor_abi = "code_object_v2"
        preload_length = 0
        preload_offset = 0
    else:
        require(len(payload) == DESCRIPTOR_BYTES, "code-object V3 descriptor is not 64 bytes")
        group_bytes, private_bytes, kernarg_bytes, entry_offset = struct.unpack_from(
            "<III4xq", payload, 0
        )
        preload_word = struct.unpack_from("<H", payload, 58)[0]
        preload_length = preload_word & 0x7F
        preload_offset = (preload_word >> 7) & 0x1FF
        descriptor_abi = "code_object_v3_or_common_v2_v3"
    raw_entry = descriptor_va + entry_offset
    require(0 <= raw_entry < (1 << 64), "descriptor entry relation overflows")
    effective_entry = raw_entry + (256 if preload_length else 0)
    require(effective_entry < (1 << 64), "effective entry overflows")
    preload_end_dwords = preload_offset + preload_length
    kernarg_read_bytes = kernarg_bytes if kernarg_bytes else preload_end_dwords * 4
    return {
        "descriptor_abi": descriptor_abi,
        "descriptor_bytes": len(payload),
        "descriptor_va": descriptor_va,
        "group_segment_fixed_size": group_bytes,
        "private_segment_fixed_size": private_bytes,
        "kernarg_size": kernarg_bytes,
        "kernel_code_entry_byte_offset": entry_offset,
        "raw_entry_va": raw_entry,
        "effective_entry_va": effective_entry,
        "kernarg_preload_spec_length": preload_length,
        "kernarg_preload_spec_offset": preload_offset,
        "kernarg_preload_end_dwords": preload_end_dwords,
        "kernarg_read_bytes": kernarg_read_bytes,
        "sha256": sha256_bytes(payload),
    }


def parse_trace_summary(payload: bytes) -> dict[str, Any]:
    complete = payload[: payload.rfind(b"\n") + 1] if b"\n" in payload else b""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(complete.splitlines(), 1):
        if not line:
            continue
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise CapsuleError(f"dispatch trace line {line_number} is invalid: {error}") from error
        require(isinstance(value, dict), f"dispatch trace line {line_number} is not an object")
        records.append(value)
    events: dict[str, int] = {}
    for record in records:
        event = record.get("event")
        if isinstance(event, str):
            events[event] = events.get(event, 0) + 1
    last = records[-1] if records else None
    return {
        "complete_records": len(records),
        "trailing_partial_bytes": len(payload) - len(complete),
        "event_counts": dict(sorted(events.items())),
        "last_record": last,
    }


def first_failure(logs: Sequence[tuple[str, bytes]]) -> dict[str, Any] | None:
    markers = ("panic:", "fatal:", "Traceback (most recent call last):")
    for role, payload in logs:
        text = payload.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), 1):
            if any(marker in line for marker in markers):
                return {"log_role": role, "line": line_number, "text": line[:4096]}
    return None


def parse_role_path(value: str) -> tuple[str, Path]:
    role, separator, path = value.partition("=")
    require(
        separator == "=" and LOG_ROLE_RE.fullmatch(role) is not None and path,
        "log must be ROLE=PATH with a 1..60 character role",
    )
    return role, Path(path).resolve()


def read_proc_snapshot(proc_dir: Path) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    snapshots: dict[str, bytes] = {}
    for name in ("maps", "status", "stat", "cmdline"):
        try:
            snapshots[name] = (proc_dir / name).read_bytes()
        except OSError as error:
            raise CapsuleError(f"could not snapshot {proc_dir / name}: {error}") from error
    try:
        exe = os.readlink(proc_dir / "exe")
    except OSError as error:
        raise CapsuleError(f"could not read process executable: {error}") from error
    snapshots["exe"] = (exe + "\n").encode("utf-8", errors="surrogateescape")

    records: list[dict[str, Any]] = []
    try:
        entries = sorted(
            (entry for entry in (proc_dir / "fd").iterdir() if entry.name.isdigit()),
            key=lambda entry: int(entry.name),
        )
    except OSError as error:
        raise CapsuleError(f"could not enumerate process descriptors: {error}") from error
    for entry in entries:
        record: dict[str, Any] = {"fd": int(entry.name)}
        try:
            record["target"] = os.readlink(entry)
        except OSError as error:
            record["target_error"] = f"{error.__class__.__name__}:{error.errno}"
        try:
            metadata = entry.stat()
            record.update(
                {
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "mode": stat.S_IFMT(metadata.st_mode),
                    "bytes": metadata.st_size,
                }
            )
        except OSError as error:
            record["stat_error"] = f"{error.__class__.__name__}:{error.errno}"
        try:
            record["fdinfo"] = (proc_dir / "fdinfo" / entry.name).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError as error:
            record["fdinfo_error"] = f"{error.__class__.__name__}:{error.errno}"
        records.append(record)
    return snapshots, records


def choose_backing(
    proc_dir: Path,
    fd_records: Sequence[Mapping[str, Any]],
    explicit_path: Path | None,
    explicit_fd: int | None,
    required_bytes: int,
) -> Path:
    require(not (explicit_path is not None and explicit_fd is not None), "choose one backing source")
    if explicit_path is not None:
        return explicit_path.resolve()
    if explicit_fd is not None:
        require(explicit_fd >= 0, "backing fd must be nonnegative")
        return proc_dir / "fd" / str(explicit_fd)
    candidates: list[tuple[int, str]] = []
    for record in fd_records:
        if record.get("mode") != stat.S_IFREG or int(record.get("bytes", -1)) < required_bytes:
            continue
        target = str(record.get("target", ""))
        if "memfd:" not in target and "/memfd:" not in target:
            continue
        score = 1 + (10 if "hsakmt" in target.lower() or "amdgpu" in target.lower() else 0)
        candidates.append((score, str(record["fd"])))
    require(candidates, "could not find the process-owned KMT backing memfd")
    best = max(score for score, _ in candidates)
    selected = [fd for score, fd in candidates if score == best]
    require(len(selected) == 1, f"KMT backing memfd is ambiguous: fds={selected}")
    return proc_dir / "fd" / selected[0]


@dataclass(frozen=True)
class QueueFrontiers:
    producer_read: int
    producer_reserved_write: int
    doorbell: int
    published_write: int
    bridge_completion: int
    completion_offset: int


def queue_frontiers(
    memory: GPUMemory, backing: ByteSource, queue: Queue
) -> QueueFrontiers:
    read_payload, _ = memory.read(queue.read_pointer_va, 8, "queue read index")
    write_payload, _ = memory.read(queue.write_pointer_va, 8, "queue write index")
    read_index = struct.unpack("<Q", read_payload)[0]
    write_index = struct.unpack("<Q", write_payload)[0]
    depth = queue.ring_bytes // PACKET_BYTES
    require(queue.ring_bytes >= 4096 and queue.ring_bytes % PACKET_BYTES == 0, "queue ring geometry is invalid")
    require(write_index >= read_index, "queue reserved write index regressed behind producer read index")
    require(write_index - read_index <= depth, "queue reserved range exceeds ring depth")
    doorbell = struct.unpack("<Q", backing.read_at(queue.doorbell_offset, 8))[0]
    completion_offset = queue.doorbell_offset + COMPLETION_OFFSET_FROM_DOORBELL
    completion = struct.unpack("<Q", backing.read_at(completion_offset, 8))[0]
    published_write = 0 if doorbell == DOORBELL_INITIAL_VALUE else doorbell + 1
    require(published_write < (1 << 64), "queue published frontier overflows")
    require(write_index >= published_write, "queue doorbell exceeds reserved write index")
    require(completion <= published_write, "queue completion exceeds published frontier")
    require(read_index <= completion, "producer read index exceeds bridge completion")
    require(published_write - completion <= depth, "queue published pending range exceeds ring depth")
    return QueueFrontiers(
        producer_read=read_index,
        producer_reserved_write=write_index,
        doorbell=doorbell,
        published_write=published_write,
        bridge_completion=completion,
        completion_offset=completion_offset,
    )


def select_queue(
    memory: GPUMemory,
    backing: ByteSource,
    registry: Registry,
    requested: int | None,
) -> tuple[Queue, QueueFrontiers]:
    candidates: list[tuple[Queue, QueueFrontiers]] = []
    for queue in registry.queues:
        if not queue.active or (requested is not None and queue.queue_id != requested):
            continue
        frontiers = queue_frontiers(memory, backing, queue)
        if frontiers.published_write > frontiers.bridge_completion:
            candidates.append((queue, frontiers))
    if requested is not None:
        require(candidates, f"queue {requested} has no pending AQL packet")
    else:
        require(candidates, "no active queue has a pending AQL packet")
        require(
            len(candidates) == 1,
            "multiple queues have pending packets; select one with --queue-id",
        )
    return candidates[0]


def allocation_role(
    roles: dict[tuple[int, int], set[str]], allocation: Allocation, role: str
) -> None:
    roles.setdefault((allocation.handle, allocation.ordinal), set()).add(role)


def capture(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    proc_root = Path(args.proc_root).resolve()
    proc_dir = proc_root / str(args.pid)
    gem5_proc_dir = proc_root / str(args.gem5_pid)
    require(args.pid > 0 and proc_dir.is_dir(), f"process {args.pid} does not exist")
    require(args.gem5_pid > 0 and args.gem5_pid != args.pid, "gem5 PID is invalid")
    require(not gem5_proc_dir.exists(), "gem5 must exit before failure-capsule capture")
    before_start = process_start_time(proc_dir)
    initial_state = process_state(proc_dir)
    require(
        initial_state in ("T", "t") or args.freeze_target,
        "target must already be stopped or --freeze-target must be used",
    )
    stopped_by_us = False
    capsule = CapsuleDirectory(output)
    backing: ByteSource | None = None
    process_memory: ByteSource | None = None
    process_handle: PidFd | None = None
    try:
        if args.freeze_target:
            require(proc_root == Path("/proc"), "--freeze-target requires the live /proc")
            process_handle = PidFd(args.pid)
            require(
                process_start_time(proc_dir) == before_start,
                "target process identity changed before stop",
            )
            if initial_state not in ("T", "t"):
                process_handle.signal(signal.SIGSTOP)
                stopped_by_us = True
                deadline = time.monotonic() + args.freeze_timeout
                while time.monotonic() < deadline and process_state(proc_dir) not in ("T", "t"):
                    time.sleep(0.01)
                require(process_state(proc_dir) in ("T", "t"), "target did not stop before capture")
        capture_state = process_state(proc_dir)
        require(capture_state in ("T", "t"), "target is not stopped for capture")

        snapshots, fd_records = read_proc_snapshot(proc_dir)
        worker_payload, worker_source = stable_file_prefix(
            Path(args.worker_log).resolve(), args.max_log_bytes
        )
        trace_payload, trace_source = stable_file_prefix(
            Path(args.dispatch_trace).resolve(), args.max_log_bytes
        )
        captured_log_bytes = len(worker_payload) + len(trace_payload)
        require(
            captured_log_bytes <= args.max_total_log_bytes,
            "logs exceed total capture budget",
        )
        registry = parse_registry(worker_payload, args.pid)

        non_user = [allocation for allocation in registry.allocations if not allocation.userptr]
        required_backing = max(
            (allocation.mmap_offset + allocation.byte_count for allocation in non_user),
            default=1,
        )
        backing_path = choose_backing(
            proc_dir,
            fd_records,
            Path(args.backing) if args.backing else None,
            args.backing_fd,
            required_backing,
        )
        backing = ByteSource(backing_path, "KMT shared backing")
        if any(allocation.userptr for allocation in registry.allocations):
            try:
                process_memory = ByteSource(proc_dir / "mem", "worker process memory")
            except CapsuleError:
                process_memory = None
        memory = GPUMemory(registry, backing, process_memory)

        queue, frontiers = select_queue(memory, backing, registry, args.queue_id)
        read_index = frontiers.bridge_completion
        write_index = frontiers.published_write
        depth = queue.ring_bytes // PACKET_BYTES
        failing_va = queue.ring_va + (read_index % depth) * PACKET_BYTES
        packet_payload, packet_allocation = memory.read(failing_va, PACKET_BYTES, "failing AQL packet")
        packet = decode_packet(packet_payload)
        next_packet: dict[str, Any] | None = None
        next_payload: bytes | None = None
        next_allocation: Allocation | None = None
        if read_index + 1 < write_index:
            next_va = queue.ring_va + ((read_index + 1) % depth) * PACKET_BYTES
            next_payload, next_allocation = memory.read(next_va, PACKET_BYTES, "next AQL packet")
            next_packet = decode_packet(next_payload, require_kernel_dispatch=False)
            next_packet["packet_va"] = next_va

        descriptor_prefix, descriptor_allocation = memory.read(
            int(packet["kernel_object"]), DESCRIPTOR_BYTES, "kernel descriptor"
        )
        descriptor_capture_bytes = (
            LEGACY_V2_DESCRIPTOR_BYTES
            if versioned_v2_descriptor(descriptor_prefix)
            else DESCRIPTOR_BYTES
        )
        if descriptor_capture_bytes == DESCRIPTOR_BYTES:
            descriptor_payload = descriptor_prefix
        else:
            descriptor_payload, descriptor_allocation = memory.read(
                int(packet["kernel_object"]),
                descriptor_capture_bytes,
                "code-object V2 descriptor",
            )
        descriptor = decode_descriptor(descriptor_payload, int(packet["kernel_object"]))
        require(
            descriptor_allocation.contains(int(descriptor["effective_entry_va"]), 1),
            "kernel entry is outside the resident code allocation",
        )

        kernarg_payload = b""
        kernarg_allocation: Allocation | None = None
        kernarg_read_bytes = int(descriptor["kernarg_read_bytes"])
        kernarg_capture_bytes = (
            kernarg_read_bytes
            if kernarg_read_bytes != 0
            else (1 if int(packet["kernarg_address"]) != 0 else 0)
        )
        if kernarg_capture_bytes != 0:
            require(
                kernarg_capture_bytes <= args.max_kernarg_bytes,
                "kernarg exceeds capture budget",
            )
            require(int(packet["kernarg_address"]) != 0, "descriptor requires a missing kernarg")
            kernarg_payload, kernarg_allocation = memory.read(
                int(packet["kernarg_address"]), kernarg_capture_bytes, "kernarg"
            )
        signal_payload = b""
        signal_allocation: Allocation | None = None
        if int(packet["completion_signal"]) != 0:
            signal_payload, signal_allocation = memory.read(
                int(packet["completion_signal"]), SIGNAL_BYTES, "completion signal"
            )

        mqd_base = queue.read_pointer_va - MQD_READ_INDEX_OFFSET
        require(mqd_base >= 0, "MQD base underflows")
        mqd_payload, mqd_allocation = memory.read(mqd_base, MQD_BYTES, "MQD")
        mqd_read = struct.unpack_from("<Q", mqd_payload, 128)[0]
        mqd_offset = struct.unpack_from("<I", mqd_payload, 136)[0]
        mqd_write = struct.unpack_from("<Q", mqd_payload, 56)[0]
        require(mqd_offset == MQD_READ_INDEX_OFFSET, "MQD read-index offset is invalid")
        require(
            mqd_read == frontiers.producer_read
            and mqd_write == frontiers.producer_reserved_write,
            "MQD indices differ from queue controls",
        )
        require(
            queue.write_pointer_va == mqd_base + 56,
            "MQD write-index pointer relation is invalid",
        )
        scratch_va = struct.unpack_from("<Q", mqd_payload, 160)[0]
        scratch_bytes = struct.unpack_from("<Q", mqd_payload, 168)[0]
        inactive_signal = struct.unpack_from("<Q", mqd_payload, 192)[0]

        roles: dict[tuple[int, int], set[str]] = {}
        allocations_by_key = {
            (allocation.handle, allocation.ordinal): allocation
            for allocation in registry.allocations
        }
        allocation_role(roles, packet_allocation, "queue_ring")
        if next_allocation is not None:
            allocation_role(roles, next_allocation, "queue_ring")
        allocation_role(roles, descriptor_allocation, "resident_code")
        allocation_role(roles, mqd_allocation, "mqd")
        if kernarg_allocation is not None:
            allocation_role(
                roles,
                kernarg_allocation,
                "kernarg" if kernarg_read_bytes else "kernarg_residency_probe",
            )
        if signal_allocation is not None:
            allocation_role(roles, signal_allocation, "completion_signal")
        pointer_candidates: list[dict[str, Any]] = []
        for offset in range(0, len(kernarg_payload) - 7, 8):
            pointer = struct.unpack_from("<Q", kernarg_payload, offset)[0]
            candidates = [allocation for allocation in registry.allocations if allocation.contains(pointer, 1)]
            if not candidates:
                continue
            allocation = max(candidates, key=lambda item: item.line_number)
            allocation_role(roles, allocation, "kernarg_pointer_candidate")
            pointer_candidates.append(
                {
                    "kernarg_offset": offset,
                    "pointer": pointer,
                    "allocation_id": allocation.handle,
                    "allocation_log_ordinal": allocation.ordinal,
                    "allocation_offset": pointer - allocation.gpu_va,
                }
            )
        require(bool(scratch_va) == bool(scratch_bytes), "scratch address/size presence differs")
        if scratch_va and scratch_bytes:
            scratch_allocation = registry.resolve(
                scratch_va, scratch_bytes, "scratch backing"
            )
            allocation_role(roles, scratch_allocation, "scratch")
        if inactive_signal:
            inactive_allocation = registry.resolve(inactive_signal, SIGNAL_BYTES, "queue inactive signal")
            allocation_role(roles, inactive_allocation, "queue_inactive_signal")

        total_hash_bytes = sum(allocations_by_key[key].byte_count for key in roles)
        require(total_hash_bytes <= args.max_hash_bytes, "referenced allocation hash budget exceeded")

        capsule.add_bytes("objects/failing-packet.bin", packet_payload, "failing_packet")
        packet["packet_va"] = failing_va
        capsule.add_json("objects/failing-packet.json", packet, "failing_packet_metadata")
        if next_payload is not None and next_packet is not None:
            capsule.add_bytes("objects/next-packet.bin", next_payload, "next_packet")
            capsule.add_json("objects/next-packet.json", next_packet, "next_packet_metadata")
        capsule.add_bytes("objects/kernel-descriptor.bin", descriptor_payload, "kernel_descriptor")
        capsule.add_json("objects/kernel-descriptor.json", descriptor, "kernel_descriptor_metadata")
        if kernarg_payload:
            capsule.add_bytes("objects/kernarg.bin", kernarg_payload, "kernarg")
        if signal_payload:
            capsule.add_bytes("objects/completion-signal.bin", signal_payload, "completion_signal")
        capsule.add_bytes("objects/mqd.bin", mqd_payload, "mqd")

        require(descriptor_allocation.byte_count <= args.max_code_bytes, "resident code allocation exceeds copy budget")
        code_path = capsule.reserve_path("objects/resident-code-allocation.bin")
        code_sha = memory.copy_allocation(descriptor_allocation, code_path)
        code_artifact = capsule.record_path("objects/resident-code-allocation.bin", "resident_code_allocation")
        require(code_artifact["sha256"] == code_sha, "resident code hash changed during capture")

        allocation_records: list[dict[str, Any]] = []
        for key in sorted(roles, key=lambda item: (allocations_by_key[item].gpu_va, item)):
            allocation = allocations_by_key[key]
            digest = code_sha if allocation == descriptor_allocation else memory.hash_allocation(allocation)
            record = allocation.identity()
            record.update(
                {
                    "roles": sorted(roles[key]),
                    "content_sha256": digest,
                    "content_hash_range": {"gpu_va": allocation.gpu_va, "bytes": allocation.byte_count},
                    "bytes_copied": allocation == descriptor_allocation,
                    "copy_artifact": code_artifact["path"] if allocation == descriptor_allocation else None,
                }
            )
            allocation_records.append(record)
        registry_snapshot = {
            "schema": REGISTRY_SCHEMA,
            "pid": registry.pid,
            "source_complete_lines": registry.complete_lines,
            "source_trailing_partial_line": registry.trailing_partial_line,
            "selection_rule": "newest_containing_allocation_by_nonoverlapping_live_va_invariant",
            "allocation_liveness_verified": False,
            "allocation_liveness_limit": "successful_free_trace_omits_handle_and_generation",
            "observed_allocations": [
                allocation.identity() for allocation in registry.allocations
            ],
            "referenced_allocations": allocation_records,
            "observed_queues": [
                {
                    "queue_id": item.queue_id,
                    "active": item.active,
                    "gpu_id": item.gpu_id,
                    "queue_type": item.queue_type,
                    "ring_va": item.ring_va,
                    "ring_bytes": item.ring_bytes,
                    "read_pointer_va": item.read_pointer_va,
                    "write_pointer_va": item.write_pointer_va,
                    "doorbell_offset": item.doorbell_offset,
                    "last_log_line": item.line_number,
                    "progress": item.progress,
                }
                for item in registry.queues
            ],
            "kernarg_pointer_candidates": pointer_candidates,
        }
        registry_artifact = capsule.add_json("kmt-registry.json", registry_snapshot, "kmt_registry")

        queue_snapshot = {
            "schema": QUEUE_SCHEMA,
            "selection_semantics": "first_unretired_published_packet_after_simulator_exit",
            "queue_id": queue.queue_id,
            "queue_type": queue.queue_type,
            "gpu_id": queue.gpu_id,
            "ring_va": queue.ring_va,
            "ring_bytes": queue.ring_bytes,
            "depth": depth,
            "read_pointer_va": queue.read_pointer_va,
            "write_pointer_va": queue.write_pointer_va,
            "read_index": read_index,
            "write_index": write_index,
            "pending_count": write_index - read_index,
            "failing_packet_index": read_index,
            "failing_packet_va": failing_va,
            "next_packet_index": read_index + 1 if next_packet is not None else None,
            "doorbell_offset": queue.doorbell_offset,
            "completion_offset": frontiers.completion_offset,
            "frontiers": {
                "producer_read": frontiers.producer_read,
                "producer_reserved_write": frontiers.producer_reserved_write,
                "doorbell": frontiers.doorbell,
                "published_write": frontiers.published_write,
                "bridge_completion": frontiers.bridge_completion,
            },
            "mqd_va": mqd_base,
            "mqd_read_index": mqd_read,
            "mqd_write_index": mqd_write,
            "scratch_va": scratch_va,
            "scratch_bytes": scratch_bytes,
            "queue_inactive_signal": inactive_signal,
            "trace_progress": queue.progress,
        }
        queue_artifact = capsule.add_json("queue-snapshot.json", queue_snapshot, "queue_snapshot")

        for name, payload in sorted(snapshots.items()):
            suffix = "bin" if name == "cmdline" else "txt"
            capsule.add_bytes(f"process/{name}.{suffix}", payload, f"process_{name}")
        capsule.add_json("process/fds.json", fd_records, "process_fds")

        log_payloads: list[tuple[str, bytes]] = [("worker", worker_payload), ("dispatch_trace", trace_payload)]
        log_sources: dict[str, dict[str, Any]] = {
            "worker": worker_source,
            "dispatch_trace": trace_source,
        }
        capsule.add_bytes("logs/worker.log", worker_payload, "worker_log")
        capsule.add_bytes("logs/dispatch-trace.jsonl", trace_payload, "dispatch_trace")
        seen_roles = {"worker", "dispatch_trace"}
        for role_path in args.log:
            role, path = parse_role_path(role_path)
            require(role not in seen_roles, f"duplicate log role: {role}")
            seen_roles.add(role)
            payload, source = stable_file_prefix(path, args.max_log_bytes)
            captured_log_bytes += len(payload)
            require(
                captured_log_bytes <= args.max_total_log_bytes,
                "logs exceed total capture budget",
            )
            log_payloads.append((role, payload))
            log_sources[role] = source
            capsule.add_bytes(f"logs/{role}.log", payload, f"{role}_log")

        after_frontiers = queue_frontiers(memory, backing, queue)
        after_start = process_start_time(proc_dir)
        final_state = process_state(proc_dir)
        require(after_frontiers == frontiers, "queue frontiers changed during capture")
        require(after_start == before_start, "target process identity changed during capture")
        require(final_state in ("T", "t"), "target resumed during capture")
        require(not gem5_proc_dir.exists(), "gem5 PID appeared during capture")
        trace_summary = parse_trace_summary(trace_payload)
        failure = first_failure(log_payloads)

        tool_path = Path(__file__).resolve()
        manifest: dict[str, Any] = {
            "schema": CAPSULE_SCHEMA,
            "status": "captured",
            "capture_time_unix_ns": time.time_ns(),
            "implementation": {
                "path": str(tool_path),
                "sha256": file_sha256(tool_path),
            },
            "process": {
                "pid": args.pid,
                "start_time_ticks": before_start,
                "initial_state": initial_state,
                "captured_state": capture_state,
                "final_state": final_state,
                "stopped_by_capture": stopped_by_us,
                "proc_root": str(proc_root),
            },
            "simulator": {
                "kind": "gem5",
                "pid": args.gem5_pid,
                "state": "exited_before_and_after_capture",
            },
            "consistency": {
                "process_identity_stable": True,
                "producer_stopped": True,
                "simulator_exited": True,
                "queue_frontiers_stable": True,
                "queue_frontiers_before": queue_snapshot["frontiers"],
                "queue_frontiers_after": queue_snapshot["frontiers"],
                "log_capture_semantics": "stable_inode_prefix",
            },
            "backing": backing.identity,
            "worker_trace": {
                "source": worker_source,
                "allocation_records": len(registry.allocations),
                "queue_records": len(registry.queues),
            },
            "dispatch_trace": {"source": trace_source, "summary": trace_summary},
            "first_failure": failure,
            "selection": queue_snapshot,
            "packet": packet,
            "next_packet": next_packet,
            "descriptor": descriptor,
            "allocation_snapshot": registry_artifact["path"],
            "allocation_count": len(allocation_records),
            "observed_allocation_count": len(registry.allocations),
            "allocation_hashed_bytes": total_hash_bytes,
            "queue_snapshot": queue_artifact["path"],
            "log_sources": log_sources,
            "capture_limits": {
                "max_hash_bytes": args.max_hash_bytes,
                "max_code_bytes": args.max_code_bytes,
                "max_kernarg_bytes": args.max_kernarg_bytes,
                "max_log_bytes": args.max_log_bytes,
                "max_total_log_bytes": args.max_total_log_bytes,
                "captured_log_bytes": captured_log_bytes,
            },
            "replay": {
                "eligible": False,
                "classification": "diagnostic_failure_capsule_only",
                "blockers": list(REPLAY_BLOCKERS),
            },
            "artifacts": sorted(capsule.artifacts, key=lambda item: item["path"]),
        }
        capsule.commit(manifest)
        return manifest
    finally:
        if process_memory is not None:
            process_memory.close()
        if backing is not None:
            backing.close()
        capsule.close()
        if process_handle is not None:
            try:
                if stopped_by_us:
                    process_handle.signal(signal.SIGCONT, ignore_missing=True)
            finally:
                process_handle.close()


def exact_keys(value: Any, expected: Iterable[str], label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} is not an object")
    actual = set(value)
    wanted = set(expected)
    require(actual == wanted, f"{label} keys differ: missing={sorted(wanted-actual)} extra={sorted(actual-wanted)}")
    return value


def verify(capsule_path: Path) -> dict[str, Any]:
    requested = capsule_path.absolute()
    require(not requested.is_symlink(), "capsule path is a symlink")
    root = requested.resolve()
    root_metadata = root.lstat()
    require(stat.S_ISDIR(root_metadata.st_mode), "capsule is not a directory")
    require(stat.S_IMODE(root_metadata.st_mode) == 0o700, "capsule mode is not 0700")
    require(root_metadata.st_uid == os.geteuid(), "capsule is not owned by the current euid")
    manifest_path = root / MANIFEST_NAME
    sha_path = root / MANIFEST_SHA_NAME
    for control in (manifest_path, sha_path):
        metadata = control.lstat()
        require(stat.S_ISREG(metadata.st_mode) and not control.is_symlink(), f"capsule control is not a regular file: {control.name}")
        require(stat.S_IMODE(metadata.st_mode) == 0o600, f"capsule control mode is not 0600: {control.name}")
        require(metadata.st_uid == os.geteuid(), f"capsule control owner differs: {control.name}")
    payload = manifest_path.read_bytes()
    try:
        manifest = json.loads(payload)
    except json.JSONDecodeError as error:
        raise CapsuleError(f"manifest is invalid JSON: {error}") from error
    require(payload == canonical_json(manifest), "manifest is not canonical JSON")
    require(isinstance(manifest, dict) and manifest.get("schema") == CAPSULE_SCHEMA, "manifest schema differs")
    exact_keys(
        manifest,
        (
            "schema",
            "status",
            "capture_time_unix_ns",
            "implementation",
            "process",
            "simulator",
            "consistency",
            "backing",
            "worker_trace",
            "dispatch_trace",
            "first_failure",
            "selection",
            "packet",
            "next_packet",
            "descriptor",
            "allocation_snapshot",
            "allocation_count",
            "observed_allocation_count",
            "allocation_hashed_bytes",
            "queue_snapshot",
            "log_sources",
            "capture_limits",
            "replay",
            "artifacts",
        ),
        "manifest",
    )
    require(manifest["status"] == "captured", "manifest status is not captured")
    process = exact_keys(
        manifest["process"],
        (
            "pid",
            "start_time_ticks",
            "initial_state",
            "captured_state",
            "final_state",
            "stopped_by_capture",
            "proc_root",
        ),
        "process",
    )
    require(type(process["pid"]) is int and process["pid"] > 0, "process PID is invalid")
    require(process["captured_state"] in ("T", "t"), "captured process was not stopped")
    require(process["final_state"] in ("T", "t"), "final process state was not stopped")
    simulator = exact_keys(manifest["simulator"], ("kind", "pid", "state"), "simulator")
    require(simulator["kind"] == "gem5", "simulator kind differs")
    require(type(simulator["pid"]) is int and simulator["pid"] > 0, "simulator PID is invalid")
    require(
        simulator["state"] == "exited_before_and_after_capture",
        "simulator quiescence is not established",
    )
    consistency = exact_keys(
        manifest["consistency"],
        (
            "process_identity_stable",
            "producer_stopped",
            "simulator_exited",
            "queue_frontiers_stable",
            "queue_frontiers_before",
            "queue_frontiers_after",
            "log_capture_semantics",
        ),
        "consistency",
    )
    require(
        consistency["process_identity_stable"] is True
        and consistency["producer_stopped"] is True
        and consistency["simulator_exited"] is True
        and consistency["queue_frontiers_stable"] is True,
        "capture consistency claims are incomplete",
    )
    require(
        consistency["queue_frontiers_before"]
        == consistency["queue_frontiers_after"],
        "captured queue frontiers differ",
    )
    require(
        consistency["log_capture_semantics"] == "stable_inode_prefix",
        "log capture semantics differ",
    )
    capture_limits = exact_keys(
        manifest["capture_limits"],
        (
            "max_hash_bytes",
            "max_code_bytes",
            "max_kernarg_bytes",
            "max_log_bytes",
            "max_total_log_bytes",
            "captured_log_bytes",
        ),
        "capture limits",
    )
    require(
        all(type(value) is int and value > 0 for value in capture_limits.values()),
        "capture limit is invalid",
    )
    require(
        capture_limits["captured_log_bytes"]
        <= capture_limits["max_total_log_bytes"],
        "captured logs exceed manifest limit",
    )
    expected_sha = f"{sha256_bytes(payload)}  {MANIFEST_NAME}\n".encode("ascii")
    require(sha_path.read_bytes() == expected_sha, "manifest checksum differs")
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, list) and artifacts, "manifest artifacts are missing")
    expected_files = {MANIFEST_NAME, MANIFEST_SHA_NAME}
    artifacts_by_path: dict[str, Mapping[str, Any]] = {}
    previous = ""
    for index, raw in enumerate(artifacts):
        record = exact_keys(raw, ("path", "role", "bytes", "sha256"), f"artifact {index}")
        relative = record["path"]
        require(isinstance(relative, str) and relative > previous, "artifact paths are not strictly sorted")
        previous = relative
        require(not relative.startswith("/") and ".." not in Path(relative).parts, "artifact path escapes capsule")
        require(isinstance(record["role"], str) and ROLE_RE.fullmatch(record["role"]) is not None, "artifact role is invalid")
        require(type(record["bytes"]) is int and record["bytes"] >= 0, "artifact size is invalid")
        require(isinstance(record["sha256"], str) and HEX64_RE.fullmatch(record["sha256"]) is not None, "artifact hash is invalid")
        path = root / relative
        current = root
        for component in Path(relative).parts[:-1]:
            current /= component
            directory_metadata = current.lstat()
            require(stat.S_ISDIR(directory_metadata.st_mode) and not current.is_symlink(), f"artifact parent is not a directory: {relative}")
            require(stat.S_IMODE(directory_metadata.st_mode) == 0o700, f"artifact parent mode is not 0700: {relative}")
        metadata = path.lstat()
        require(stat.S_ISREG(metadata.st_mode) and not path.is_symlink(), f"artifact is not a regular file: {relative}")
        require(stat.S_IMODE(metadata.st_mode) == 0o600, f"artifact mode is not 0600: {relative}")
        require(metadata.st_uid == os.geteuid(), f"artifact owner differs: {relative}")
        require(metadata.st_size == record["bytes"], f"artifact size differs: {relative}")
        require(file_sha256(path) == record["sha256"], f"artifact hash differs: {relative}")
        expected_files.add(relative)
        artifacts_by_path[relative] = record

    selection = manifest["selection"]
    require(isinstance(selection, Mapping) and selection.get("schema") == QUEUE_SCHEMA, "queue selection schema differs")
    require(
        selection.get("selection_semantics")
        == "first_unretired_published_packet_after_simulator_exit",
        "queue selection semantics differ",
    )
    integer_fields = (
        "ring_va",
        "ring_bytes",
        "depth",
        "read_index",
        "write_index",
        "pending_count",
        "failing_packet_index",
        "failing_packet_va",
        "doorbell_offset",
        "completion_offset",
        "mqd_read_index",
        "mqd_write_index",
    )
    require(
        all(type(selection.get(name)) is int for name in integer_fields),
        "queue selection integer field is invalid",
    )
    require(selection["depth"] == selection["ring_bytes"] // PACKET_BYTES, "queue depth differs")
    require(selection["pending_count"] > 0, "queue selection has no pending packet")
    require(
        selection["write_index"] - selection["read_index"]
        == selection["pending_count"],
        "queue pending count differs",
    )
    require(selection["failing_packet_index"] == selection["read_index"], "queue head index differs")
    require(
        selection["failing_packet_va"]
        == selection["ring_va"]
        + (selection["read_index"] % selection["depth"]) * PACKET_BYTES,
        "queue head address differs",
    )
    require(
        selection["completion_offset"]
        == selection["doorbell_offset"] + COMPLETION_OFFSET_FROM_DOORBELL,
        "queue completion offset differs",
    )
    frontiers = exact_keys(
        selection.get("frontiers"),
        (
            "producer_read",
            "producer_reserved_write",
            "doorbell",
            "published_write",
            "bridge_completion",
        ),
        "queue frontiers",
    )
    require(all(type(value) is int and value >= 0 for value in frontiers.values()), "queue frontier is invalid")
    require(
        frontiers["producer_read"]
        <= frontiers["bridge_completion"]
        < frontiers["published_write"]
        <= frontiers["producer_reserved_write"],
        "queue frontier ordering differs",
    )
    require(selection["read_index"] == frontiers["bridge_completion"], "queue head is not bridge completion")
    require(selection["write_index"] == frontiers["published_write"], "queue tail is not published frontier")
    require(selection["mqd_read_index"] == frontiers["producer_read"], "MQD read frontier differs")
    require(selection["mqd_write_index"] == frontiers["producer_reserved_write"], "MQD write frontier differs")
    require(consistency["queue_frontiers_before"] == frontiers, "manifest queue consistency cross-link differs")

    packet = manifest["packet"]
    require(isinstance(packet, Mapping) and packet.get("schema") == PACKET_SCHEMA, "packet schema differs")
    require(packet.get("packet_type") == KERNEL_DISPATCH_PACKET_TYPE, "queue head is not a kernel dispatch")
    require(packet.get("resource_decode") == "kernel_dispatch", "packet resource decode differs")
    require(packet.get("packet_va") == selection["failing_packet_va"], "packet address cross-link differs")
    require(isinstance(packet.get("sha256"), str) and HEX64_RE.fullmatch(packet["sha256"]) is not None, "packet hash is invalid")
    descriptor = manifest["descriptor"]
    require(isinstance(descriptor, Mapping), "descriptor is not an object")
    require(descriptor.get("descriptor_va") == packet.get("kernel_object"), "descriptor address cross-link differs")
    require(isinstance(descriptor.get("sha256"), str) and HEX64_RE.fullmatch(descriptor["sha256"]) is not None, "descriptor hash is invalid")
    require(type(descriptor.get("kernarg_size")) is int and descriptor["kernarg_size"] >= 0, "descriptor kernarg size is invalid")
    require(
        type(descriptor.get("kernarg_preload_spec_length")) is int
        and type(descriptor.get("kernarg_preload_spec_offset")) is int
        and type(descriptor.get("kernarg_preload_end_dwords")) is int
        and type(descriptor.get("kernarg_read_bytes")) is int,
        "descriptor kernarg read metadata is invalid",
    )
    require(
        descriptor["kernarg_preload_end_dwords"]
        == descriptor["kernarg_preload_spec_offset"]
        + descriptor["kernarg_preload_spec_length"],
        "descriptor preload end differs",
    )
    expected_kernarg_read_bytes = (
        descriptor["kernarg_size"]
        if descriptor["kernarg_size"] != 0
        else descriptor["kernarg_preload_end_dwords"] * 4
    )
    require(
        descriptor["kernarg_read_bytes"] == expected_kernarg_read_bytes,
        "descriptor kernarg read size differs",
    )
    require(type(packet.get("completion_signal")) is int and packet["completion_signal"] >= 0, "packet completion signal is invalid")

    required_paths = {
        "kmt-registry.json": "kmt_registry",
        "queue-snapshot.json": "queue_snapshot",
        "logs/worker.log": "worker_log",
        "logs/dispatch-trace.jsonl": "dispatch_trace",
        "objects/failing-packet.bin": "failing_packet",
        "objects/failing-packet.json": "failing_packet_metadata",
        "objects/kernel-descriptor.bin": "kernel_descriptor",
        "objects/kernel-descriptor.json": "kernel_descriptor_metadata",
        "objects/mqd.bin": "mqd",
        "objects/resident-code-allocation.bin": "resident_code_allocation",
        "process/maps.txt": "process_maps",
        "process/status.txt": "process_status",
        "process/stat.txt": "process_stat",
        "process/cmdline.bin": "process_cmdline",
        "process/exe.txt": "process_exe",
        "process/fds.json": "process_fds",
    }
    expected_kernarg_capture_bytes = (
        descriptor["kernarg_read_bytes"]
        if descriptor["kernarg_read_bytes"] != 0
        else (1 if packet.get("kernarg_address") != 0 else 0)
    )
    if expected_kernarg_capture_bytes > 0:
        required_paths["objects/kernarg.bin"] = "kernarg"
    if packet["completion_signal"] != 0:
        required_paths["objects/completion-signal.bin"] = "completion_signal"
    next_packet = manifest["next_packet"]
    if selection["pending_count"] > 1:
        require(isinstance(next_packet, Mapping), "next packet metadata is missing")
        require(selection.get("next_packet_index") == selection["read_index"] + 1, "next packet index differs")
        required_paths["objects/next-packet.bin"] = "next_packet"
        required_paths["objects/next-packet.json"] = "next_packet_metadata"
    else:
        require(next_packet is None and selection.get("next_packet_index") is None, "unexpected next packet metadata")
    for relative, role in required_paths.items():
        record = artifacts_by_path.get(relative)
        require(record is not None and record["role"] == role, f"required artifact differs: {relative}")
    require(manifest["allocation_snapshot"] == "kmt-registry.json", "allocation snapshot path differs")
    require(manifest["queue_snapshot"] == "queue-snapshot.json", "queue snapshot path differs")

    mqd_payload = (root / "objects/mqd.bin").read_bytes()
    require(len(mqd_payload) == MQD_BYTES, "MQD artifact size differs")
    require(
        struct.unpack_from("<Q", mqd_payload, 56)[0]
        == selection["mqd_write_index"],
        "MQD write index differs from selection",
    )
    require(
        struct.unpack_from("<Q", mqd_payload, 128)[0]
        == selection["mqd_read_index"],
        "MQD read index differs from selection",
    )
    require(
        struct.unpack_from("<I", mqd_payload, 136)[0] == MQD_READ_INDEX_OFFSET,
        "MQD read-index offset differs",
    )
    require(
        struct.unpack_from("<Q", mqd_payload, 160)[0]
        == selection.get("scratch_va")
        and struct.unpack_from("<Q", mqd_payload, 168)[0]
        == selection.get("scratch_bytes"),
        "MQD scratch range differs from selection",
    )
    require(
        struct.unpack_from("<Q", mqd_payload, 192)[0]
        == selection.get("queue_inactive_signal"),
        "MQD inactive signal differs from selection",
    )

    registry = exact_keys(
        json.loads((root / "kmt-registry.json").read_bytes()),
        (
            "schema",
            "pid",
            "source_complete_lines",
            "source_trailing_partial_line",
            "selection_rule",
            "allocation_liveness_verified",
            "allocation_liveness_limit",
            "observed_allocations",
            "referenced_allocations",
            "observed_queues",
            "kernarg_pointer_candidates",
        ),
        "KMT registry",
    )
    require(registry["schema"] == REGISTRY_SCHEMA, "KMT registry schema differs")
    require(registry["pid"] == process["pid"], "KMT registry PID differs")
    require(
        registry["selection_rule"]
        == "newest_containing_allocation_by_nonoverlapping_live_va_invariant",
        "KMT allocation selection rule differs",
    )
    require(
        registry["allocation_liveness_verified"] is False
        and registry["allocation_liveness_limit"]
        == "successful_free_trace_omits_handle_and_generation",
        "KMT allocation liveness boundary differs",
    )
    observed = registry["observed_allocations"]
    referenced = registry["referenced_allocations"]
    observed_queues = registry["observed_queues"]
    require(isinstance(observed, list) and isinstance(referenced, list), "KMT allocation lists are invalid")
    require(isinstance(observed_queues, list), "KMT queue list is invalid")
    require(
        len(observed) == manifest["observed_allocation_count"],
        "observed allocation count differs",
    )
    require(
        len(referenced) == manifest["allocation_count"],
        "referenced allocation count differs",
    )
    worker_trace = exact_keys(
        manifest["worker_trace"],
        ("source", "allocation_records", "queue_records"),
        "worker trace",
    )
    require(
        worker_trace["allocation_records"] == len(observed)
        and worker_trace["queue_records"] == len(observed_queues),
        "worker trace registry counts differ",
    )
    identity_keys = (
        "allocation_id",
        "allocation_log_ordinal",
        "allocation_log_line",
        "generation",
        "generation_status",
        "liveness_status",
        "gpu_id",
        "gpu_va",
        "bytes",
        "flags",
        "storage",
        "storage_offset",
    )
    observed_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for index, raw in enumerate(observed):
        allocation = exact_keys(raw, identity_keys, f"observed allocation {index}")
        key = (allocation["allocation_id"], allocation["allocation_log_ordinal"])
        require(
            all(type(value) is int and value >= 0 for value in key),
            f"observed allocation {index} identity is invalid",
        )
        require(key not in observed_by_key, "observed allocation identity is duplicated")
        require(type(allocation["gpu_va"]) is int and allocation["gpu_va"] >= 0, "observed allocation VA is invalid")
        require(type(allocation["bytes"]) is int and allocation["bytes"] > 0, "observed allocation size is invalid")
        require(allocation["generation"] is None, "unavailable allocation generation was invented")
        observed_by_key[key] = allocation

    referenced_total = 0
    referenced_roles: set[str] = set()
    resident_code_records: list[Mapping[str, Any]] = []
    for index, raw in enumerate(referenced):
        allocation = exact_keys(
            raw,
            identity_keys
            + (
                "roles",
                "content_sha256",
                "content_hash_range",
                "bytes_copied",
                "copy_artifact",
            ),
            f"referenced allocation {index}",
        )
        key = (allocation["allocation_id"], allocation["allocation_log_ordinal"])
        observed_allocation = observed_by_key.get(key)
        require(observed_allocation is not None, "referenced allocation identity was not observed")
        require(
            all(allocation[name] == observed_allocation[name] for name in identity_keys),
            "referenced allocation identity differs from observed registry",
        )
        roles = allocation["roles"]
        require(
            isinstance(roles, list)
            and roles == sorted(set(roles))
            and all(isinstance(role, str) and ROLE_RE.fullmatch(role) for role in roles),
            "referenced allocation roles are invalid",
        )
        referenced_roles.update(roles)
        content_range = exact_keys(
            allocation["content_hash_range"], ("gpu_va", "bytes"), "content hash range"
        )
        require(
            content_range["gpu_va"] == allocation["gpu_va"]
            and content_range["bytes"] == allocation["bytes"],
            "allocation content hash range differs",
        )
        require(
            isinstance(allocation["content_sha256"], str)
            and HEX64_RE.fullmatch(allocation["content_sha256"]) is not None,
            "allocation content hash is invalid",
        )
        require(type(allocation["bytes_copied"]) is bool, "allocation copy flag is invalid")
        referenced_total += allocation["bytes"]
        if "resident_code" in roles:
            resident_code_records.append(allocation)
    require(
        referenced_total == manifest["allocation_hashed_bytes"],
        "allocation hashed-byte total differs",
    )
    require(
        referenced_total <= capture_limits["max_hash_bytes"],
        "allocation hashes exceed manifest limit",
    )
    require(
        {"queue_ring", "resident_code", "mqd"}.issubset(referenced_roles),
        "required allocation roles are missing",
    )
    require(len(resident_code_records) == 1, "resident code allocation identity is ambiguous")
    resident_code = resident_code_records[0]
    require(
        resident_code["bytes_copied"] is True
        and resident_code["copy_artifact"]
        == "objects/resident-code-allocation.bin",
        "resident code copy relation differs",
    )
    require(
        resident_code["content_sha256"]
        == artifacts_by_path["objects/resident-code-allocation.bin"]["sha256"],
        "resident code allocation hash differs",
    )
    require(
        resident_code["bytes"]
        == artifacts_by_path["objects/resident-code-allocation.bin"]["bytes"],
        "resident code allocation size differs",
    )
    require(
        resident_code["bytes"] <= capture_limits["max_code_bytes"],
        "resident code exceeds manifest copy limit",
    )
    require(
        all(
            allocation["bytes_copied"] is False
            and allocation["copy_artifact"] is None
            for allocation in referenced
            if allocation is not resident_code
        ),
        "non-code allocation unexpectedly claims copied bytes",
    )
    selected_queue = [
        queue
        for queue in observed_queues
        if isinstance(queue, Mapping) and queue.get("queue_id") == selection.get("queue_id")
    ]
    require(len(selected_queue) == 1, "selected queue identity is absent or ambiguous")
    selected_queue_record = selected_queue[0]
    require(
        selected_queue_record.get("active") is True
        and selected_queue_record.get("ring_va") == selection["ring_va"]
        and selected_queue_record.get("ring_bytes") == selection["ring_bytes"]
        and selected_queue_record.get("read_pointer_va")
        == selection.get("read_pointer_va")
        and selected_queue_record.get("write_pointer_va")
        == selection.get("write_pointer_va")
        and selected_queue_record.get("doorbell_offset")
        == selection["doorbell_offset"],
        "selected queue registry cross-link differs",
    )

    require(json.loads((root / "queue-snapshot.json").read_bytes()) == selection, "queue snapshot cross-link differs")
    require(json.loads((root / "objects/failing-packet.json").read_bytes()) == packet, "packet metadata cross-link differs")
    require(json.loads((root / "objects/kernel-descriptor.json").read_bytes()) == descriptor, "descriptor metadata cross-link differs")
    require(artifacts_by_path["objects/failing-packet.bin"]["sha256"] == packet["sha256"], "packet bytes hash cross-link differs")
    require(artifacts_by_path["objects/kernel-descriptor.bin"]["sha256"] == descriptor["sha256"], "descriptor bytes hash cross-link differs")
    decoded_packet = decode_packet((root / "objects/failing-packet.bin").read_bytes())
    decoded_packet["packet_va"] = selection["failing_packet_va"]
    require(decoded_packet == packet, "packet binary decode differs from metadata")
    decoded_descriptor = decode_descriptor(
        (root / "objects/kernel-descriptor.bin").read_bytes(),
        packet["kernel_object"],
    )
    require(decoded_descriptor == descriptor, "descriptor binary decode differs from metadata")
    descriptor_offset = packet["kernel_object"] - resident_code["gpu_va"]
    require(
        0 <= descriptor_offset
        and descriptor_offset + descriptor["descriptor_bytes"]
        <= artifacts_by_path["objects/resident-code-allocation.bin"]["bytes"],
        "descriptor is outside copied resident code",
    )
    with (root / "objects/resident-code-allocation.bin").open("rb") as stream:
        stream.seek(descriptor_offset)
        resident_descriptor = stream.read(descriptor["descriptor_bytes"])
    require(
        resident_descriptor == (root / "objects/kernel-descriptor.bin").read_bytes(),
        "descriptor bytes differ from resident code allocation",
    )
    if expected_kernarg_capture_bytes > 0:
        require(
            artifacts_by_path["objects/kernarg.bin"]["bytes"]
            == expected_kernarg_capture_bytes,
            "kernarg artifact size differs",
        )
    if packet["completion_signal"] != 0:
        require(
            artifacts_by_path["objects/completion-signal.bin"]["bytes"]
            == SIGNAL_BYTES,
            "completion signal artifact size differs",
        )
    if next_packet is not None:
        require(json.loads((root / "objects/next-packet.json").read_bytes()) == next_packet, "next packet metadata cross-link differs")
        require(artifacts_by_path["objects/next-packet.bin"]["sha256"] == next_packet.get("sha256"), "next packet bytes hash cross-link differs")
        decoded_next = decode_packet(
            (root / "objects/next-packet.bin").read_bytes(),
            require_kernel_dispatch=False,
        )
        decoded_next["packet_va"] = (
            selection["ring_va"]
            + ((selection["read_index"] + 1) % selection["depth"])
            * PACKET_BYTES
        )
        require(decoded_next == next_packet, "next packet binary decode differs from metadata")

    log_artifacts = [
        record
        for relative, record in artifacts_by_path.items()
        if relative.startswith("logs/")
    ]
    require(
        sum(record["bytes"] for record in log_artifacts)
        == capture_limits["captured_log_bytes"],
        "captured log byte count differs",
    )
    require(
        all(record["bytes"] <= capture_limits["max_log_bytes"] for record in log_artifacts),
        "captured log exceeds per-file limit",
    )

    replay = exact_keys(manifest["replay"], ("eligible", "classification", "blockers"), "replay")
    require(replay["eligible"] is False, "v1 capsule must not claim replay eligibility")
    require(replay["classification"] == "diagnostic_failure_capsule_only", "replay classification differs")
    require(replay["blockers"] == list(REPLAY_BLOCKERS), "replay blockers differ")
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    require(actual_files == expected_files, f"capsule file set differs: missing={sorted(expected_files-actual_files)} extra={sorted(actual_files-expected_files)}")
    return {
        "schema": CAPSULE_SCHEMA,
        "status": "verified",
        "capsule": str(root),
        "manifest_sha256": sha256_bytes(payload),
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(record["bytes"] for record in artifacts),
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subcommands = command.add_subparsers(dest="command", required=True)
    capture_parser = subcommands.add_parser("capture", help="capture one pending AQL queue head")
    capture_parser.add_argument("--pid", type=int, required=True, help="ROCr producer PID")
    capture_parser.add_argument(
        "--gem5-pid",
        type=int,
        required=True,
        help="gem5 PID, which must already have exited",
    )
    capture_parser.add_argument("--worker-log", required=True, help="log with SAGR_HSAKMT_MODEL_TRACE=1")
    capture_parser.add_argument("--dispatch-trace", required=True, help="gem5 dispatch-trace.jsonl")
    capture_parser.add_argument("--log", action="append", default=[], metavar="ROLE=PATH", help="additional log to freeze")
    capture_parser.add_argument("--output", required=True, help="absent output directory")
    capture_parser.add_argument("--queue-id", type=int, help="required when multiple queues are pending")
    capture_parser.add_argument("--backing", help="explicit KMT shared backing file")
    capture_parser.add_argument("--backing-fd", type=int, help="producer fd for the KMT backing")
    capture_parser.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)
    capture_parser.add_argument("--freeze-target", action="store_true", help="SIGSTOP only the exact producer during capture and restore it afterwards")
    capture_parser.add_argument("--freeze-timeout", type=float, default=2.0)
    capture_parser.add_argument("--max-hash-bytes", type=int, default=DEFAULT_MAX_HASH_BYTES)
    capture_parser.add_argument("--max-code-bytes", type=int, default=DEFAULT_MAX_CODE_BYTES)
    capture_parser.add_argument("--max-kernarg-bytes", type=int, default=DEFAULT_MAX_KERNARG_BYTES)
    capture_parser.add_argument("--max-log-bytes", type=int, default=DEFAULT_MAX_LOG_BYTES)
    capture_parser.add_argument("--max-total-log-bytes", type=int, default=DEFAULT_MAX_TOTAL_LOG_BYTES)
    verify_parser = subcommands.add_parser("verify", help="verify a completed capsule without launching gem5")
    verify_parser.add_argument("capsule")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "capture":
            require(args.freeze_timeout > 0, "freeze timeout must be positive")
            require(
                args.max_hash_bytes > 0
                and args.max_code_bytes > 0
                and args.max_kernarg_bytes > 0
                and args.max_log_bytes > 0
                and args.max_total_log_bytes > 0,
                "capture budgets must be positive",
            )
            require(args.queue_id is None or args.queue_id > 0, "queue id must be positive")
            result = capture(args)
            summary = {
                "schema": CAPSULE_SCHEMA,
                "status": result["status"],
                "output": str(Path(args.output).resolve()),
                "failing_packet_index": result["selection"]["failing_packet_index"],
                "pending_count": result["selection"]["pending_count"],
                "replay_eligible": False,
            }
        else:
            summary = verify(Path(args.capsule))
    except (CapsuleError, OSError, ValueError) as error:
        print(f"aql-failure-capsule: {error}", file=os.sys.stderr)
        return 1
    print(canonical_json(summary).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
