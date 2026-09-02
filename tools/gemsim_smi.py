#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Report the 16-slot simulator device inventory without ROCm/KMD access."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any, Sequence


OUTPUT_SCHEMA = "amdgpu-sim.gemsim-smi.v2"
REGISTRY_OUTPUT_SCHEMA = "amdgpu-sim.gemsim-smi.v1"
DEVICE_COUNT = 16
RECORD_BYTES = 320
RECORD_PAYLOAD_BYTES = 288
ENDPOINT_BYTES = 112
RECORD_MAGIC = b"SAGRSMI1"
RECORD_VERSION = 1


class SMIError(RuntimeError):
    pass


def default_state_directory() -> Path:
    return Path(f"/tmp/amdgpu-sim-smi-{os.getuid()}")


def _proc_start_time(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError:
        return None
    close = text.rfind(")")
    fields = text[close + 2 :].split() if close >= 0 else []
    if len(fields) <= 19:
        return None
    try:
        value = int(fields[19], 10)
    except ValueError:
        return None
    return value if value > 0 else None


def _record_path(directory: Path, device: int) -> Path:
    return directory / f"device-{device:02d}.bin"


def _empty_device(device: int, path: Path, reason: str) -> dict[str, Any]:
    return {
        "device": device,
        "status": "OFF",
        "reason": reason,
        "owner_pid": None,
        "daemon_pid": None,
        "daemon_uuid": None,
        "job_uuid": None,
        "epoch": None,
        "connection_id": None,
        "rank": None,
        "world_size": None,
        "exact_topology": None,
        "endpoint": None,
        "record_path": str(path),
    }


def _lock_is_held(descriptor: int) -> bool:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False


def _decode_record(payload: bytes, device: int, path: Path) -> dict[str, Any]:
    if len(payload) != RECORD_BYTES:
        raise SMIError(f"locked simulator device record has wrong size: {path}")
    if not hashlib.sha256(payload[:RECORD_PAYLOAD_BYTES]).digest() == payload[RECORD_PAYLOAD_BYTES:]:
        raise SMIError(f"locked simulator device record digest differs: {path}")
    if payload[:8] != RECORD_MAGIC:
        raise SMIError(f"locked simulator device record magic differs: {path}")
    version, record_bytes, slot, flags = struct.unpack_from(">IIII", payload, 8)
    if version != RECORD_VERSION or record_bytes != RECORD_BYTES or slot != device:
        raise SMIError(f"locked simulator device record identity differs: {path}")
    if flags not in (0, 1):
        raise SMIError(f"locked simulator device record flags differ: {path}")
    owner_pid, daemon_pid = struct.unpack_from(">II", payload, 24)
    (
        owner_start_time,
        daemon_start_time,
        epoch,
        connection_id,
    ) = struct.unpack_from(">QQQQ", payload, 32)
    rank, world_size = struct.unpack_from(">II", payload, 64)
    job_uuid = payload[72:88]
    daemon_uuid = payload[88:104]
    gem5_device, gem5_inode, published_at_ns = struct.unpack_from(">QQQ", payload, 104)
    endpoint_size = struct.unpack_from(">I", payload, 128)[0]
    if (
        owner_pid == 0
        or daemon_pid == 0
        or owner_start_time == 0
        or daemon_start_time == 0
        or epoch == 0
        or connection_id == 0
        or world_size == 0
        or world_size > DEVICE_COUNT
        or rank >= world_size
        or job_uuid == bytes(16)
        or daemon_uuid == bytes(16)
        or gem5_inode == 0
        or published_at_ns == 0
        or endpoint_size == 0
        or endpoint_size >= ENDPOINT_BYTES
        or payload[132:136] != bytes(4)
        or payload[136 + endpoint_size : RECORD_PAYLOAD_BYTES] != bytes(
            RECORD_PAYLOAD_BYTES - 136 - endpoint_size
        )
    ):
        raise SMIError(f"locked simulator device record fields are invalid: {path}")
    try:
        endpoint = payload[136 : 136 + endpoint_size].decode("ascii")
    except UnicodeDecodeError as error:
        raise SMIError(f"locked simulator endpoint is not ASCII: {path}") from error
    endpoint_path = Path(endpoint)
    if not endpoint_path.is_absolute() or endpoint_path != Path(os.path.normpath(endpoint)):
        raise SMIError(f"locked simulator endpoint is not normalized: {path}")
    return {
        "device": device,
        "status": "ON",
        "reason": "managed_gem5_ready",
        "owner_pid": owner_pid,
        "owner_start_time": owner_start_time,
        "daemon_pid": daemon_pid,
        "daemon_start_time": daemon_start_time,
        "daemon_uuid": daemon_uuid.hex(),
        "job_uuid": job_uuid.hex(),
        "epoch": epoch,
        "connection_id": connection_id,
        "rank": rank,
        "world_size": world_size,
        "exact_topology": bool(flags & 1),
        "endpoint": endpoint,
        "gem5_executable_device": gem5_device,
        "gem5_executable_inode": gem5_inode,
        "published_at_monotonic_ns": published_at_ns,
        "record_path": str(path),
    }


def _validate_live_identity(record: dict[str, Any]) -> str | None:
    if _proc_start_time(record["owner_pid"]) != record["owner_start_time"]:
        return "owner_identity_unavailable"
    if _proc_start_time(record["daemon_pid"]) != record["daemon_start_time"]:
        return "daemon_identity_unavailable"
    try:
        executable = Path(f"/proc/{record['daemon_pid']}/exe").stat()
    except OSError:
        return "daemon_executable_unavailable"
    if (
        executable.st_dev != record["gem5_executable_device"]
        or executable.st_ino != record["gem5_executable_inode"]
    ):
        return "daemon_executable_identity_differs"
    try:
        endpoint = Path(record["endpoint"]).lstat()
    except OSError:
        return "daemon_endpoint_unavailable"
    if (
        not stat.S_ISSOCK(endpoint.st_mode)
        or endpoint.st_uid != os.getuid()
        or endpoint.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
    ):
        return "daemon_endpoint_identity_differs"
    return None


def _read_device(directory: Path, device: int) -> dict[str, Any]:
    path = _record_path(directory, device)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return _empty_device(device, path, "unused")
    except OSError as error:
        raise SMIError(f"could not open simulator device record {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            raise SMIError(f"unsafe simulator device record: {path}")
        if not _lock_is_held(descriptor):
            return _empty_device(device, path, "unused")
        payload = b""
        while len(payload) < RECORD_BYTES:
            chunk = os.read(descriptor, RECORD_BYTES - len(payload))
            if not chunk:
                break
            payload += chunk
        record = _decode_record(payload, device, path)
        reason = _validate_live_identity(record)
        if reason is not None or not _lock_is_held(descriptor):
            return _empty_device(device, path, reason or "lease_released")
        return record
    finally:
        os.close(descriptor)


def _validate_state_directory(directory: Path) -> None:
    if not directory.exists():
        return
    metadata = directory.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise SMIError(f"simulator device state directory is unsafe: {directory}")


def device_document(directory: Path) -> dict[str, Any]:
    directory = Path(os.path.abspath(directory))
    _validate_state_directory(directory)
    devices = [_read_device(directory, device) for device in range(DEVICE_COUNT)]
    on_count = sum(device["status"] == "ON" for device in devices)
    return {
        "schema": OUTPUT_SCHEMA,
        "logical_device_count": DEVICE_COUNT,
        "device_model": DEVICE_MODEL,
        "on_count": on_count,
        "off_count": DEVICE_COUNT - on_count,
        "state_directory": str(directory),
        "devices": devices,
    }


def registry_document(registry_path: Path) -> dict[str, Any]:
    from gemsim_live_registry import RegistryError, read_live_snapshot

    try:
        snapshot = read_live_snapshot(registry_path)
    except RegistryError as error:
        raise SMIError(str(error)) from error
    registry = snapshot.registry
    status = snapshot.effective_status
    ranks = []
    for rank in registry["ranks"]:
        ranks.append(
            {
                "status": status,
                "registry_state": rank["state"],
                "daemon_pid": rank["daemon_pid"],
                "daemon_uuid": rank["daemon_uuid"],
                "job_uuid": registry["job_uuid"],
                "epoch": registry["epoch"],
                "rank": rank["rank"],
                "world_size": rank["world_size"],
                "endpoint": rank["endpoint"],
            }
        )
    return {
        "schema": REGISTRY_OUTPUT_SCHEMA,
        "status": status,
        "registry_state": registry["state"],
        "lease_held": snapshot.lease_held,
        "generation": registry["generation"],
        "registry_path": str(registry_path.absolute()),
        "job_uuid": registry["job_uuid"],
        "epoch": registry["epoch"],
        "world_size": registry["world_size"],
        "ranks": ranks,
    }


def _cell(value: Any) -> str:
    return "-" if value is None else str(value)


DEVICE_MODEL = "AMD Instinct MI350X（虞书欣粉丝特供版）"


def _device_table(document: dict[str, Any]) -> str:
    headers = ("GPU", "MODEL", "STATUS", "DAEMON_PID", "DAEMON_UUID", "RANK", "WORLD", "JOB_UUID")
    rows = [
        (
            str(device["device"]),
            DEVICE_MODEL,
            device["status"],
            _cell(device["daemon_pid"]),
            _cell(device["daemon_uuid"]),
            _cell(device["rank"]),
            _cell(device["world_size"]),
            _cell(device["job_uuid"]),
        )
        for device in document["devices"]
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    summary = (
        f"GemSim devices: {document['on_count']} ON, {document['off_count']} OFF"
        f"  model: {DEVICE_MODEL}"
    )
    return "\n".join((summary, render(headers), render(tuple("-" * width for width in widths)), *(render(row) for row in rows)))


def _registry_table(document: dict[str, Any]) -> str:
    headers = ("STATUS", "STATE", "DAEMON_PID", "DAEMON_UUID", "RANK", "WORLD", "ENDPOINT")
    rows = [
        (
            rank["status"],
            rank["registry_state"],
            _cell(rank["daemon_pid"]),
            _cell(rank["daemon_uuid"]),
            str(rank["rank"]),
            str(rank["world_size"]),
            rank["endpoint"],
        )
        for rank in document["ranks"]
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    return "\n".join((render(headers), render(tuple("-" * width for width in widths)), *(render(row) for row in rows)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report up to 16 managed GemSim devices. This command reads only "
            "runtime leases and procfs; it never opens GPU device nodes."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--state-dir", type=Path, default=default_state_directory())
    group.add_argument("--registry", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = (
            registry_document(args.registry)
            if args.registry is not None
            else device_document(args.state_dir)
        )
    except SMIError as error:
        print(f"rocm-smi: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(document, sort_keys=True))
    else:
        print(
            _registry_table(document)
            if args.registry is not None
            else _device_table(document)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
