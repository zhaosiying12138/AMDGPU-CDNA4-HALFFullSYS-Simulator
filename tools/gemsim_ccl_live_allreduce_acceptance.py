#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Authoritative verifier for future device-backed live allreduce evidence.

The verifier never launches gem5.  It consumes a completed, immutable runner
directory and a separately supplied design wrapper, validates every byte, and
publishes a new absent-only evidence directory with ``renameat2(NOREPLACE)``.
Runner summaries and counters are deliberately not acceptance authority.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import shlex
import stat
import struct
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
RUNNER_FILE = ROOT / "scripts/run_gemsim_ccl_live_allreduce.py"
WORKER_FILE = ROOT / "examples/triton/ccl_live_allreduce_rank.py"
BOOTSTRAP_FILE = ROOT / "examples/triton/_gemsim_bootstrap.py"
DESIGN_FILE = ROOT / "tools/gemsim_ccl_live_allreduce.py"
EXPECTED_SCHEMA = "amdgpu-sim.ccl-live-allreduce-expected.v1"
RUN_SCHEMA = "amdgpu-sim.ccl-live-allreduce-run.v1"
RANK_RESULT_SCHEMA = "amdgpu-sim.ccl-live-allreduce-rank-result.v1"
JOURNAL_SCHEMA = "amdgpu-sim.ccl-live-allreduce-step-event.v1"
ACCEPTANCE_SCHEMA = "amdgpu-sim.ccl-live-allreduce-acceptance.v1"
MANIFEST_SCHEMA = "amdgpu-sim.ccl-live-allreduce-acceptance-manifest.v1"
DESIGN_SCHEMA = "amdgpu-sim.ccl-live-allreduce-design.v1"
RANK_LAUNCH_SCHEMA = "amdgpu-sim.gemsim-rank-launch.v1"
TRACE_SCHEMA = "amdgpu-sim.generic-kernel-execution-trace.v1"
FORMAL_WORLDS = (2, 3, 4, 8, 16)
ALL_WORLDS = tuple(range(2, 17))
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_JOURNAL_BYTES = 512 * 1024 * 1024
MAX_SEGMENT_BYTES = 16 * 1024 * 1024
UINT32_MAX = (1 << 32) - 1
AT_FDCWD = -100
RENAME_NOREPLACE = 1
HEX32 = re.compile(r"[0-9a-f]{32}")
HEX64 = re.compile(r"[0-9a-f]{64}")
READY_PATTERN = re.compile(
    r"host-gpu-ready endpoint=(\S+) daemon_uuid=([0-9a-f]{32}) "
    r"job_uuid=([0-9a-f]{32}) epoch=([1-9][0-9]*) rank=([0-9]+) "
    r"world=([0-9]+) max_record=([1-9][0-9]*)"
)
HANDSHAKE_PATTERN = re.compile(
    r"host-gpu-handshake status=OK fd=([0-9]+) generation=([1-9][0-9]*)"
)
EXIT_PATTERN = re.compile(
    r"host-gpu-dispatch-exit cause=host GPU dispatch session complete "
    r"code=0 tick=([1-9][0-9]*) stats=(\S+)"
)
PID_PATTERN = re.compile(r"^gem5 executing on .+, pid ([1-9][0-9]*)$", re.MULTILINE)
DTYPES = {"bfloat16": (1, 2), "float32": (2, 4)}
PHASE_REDUCE_SCATTER = 1
PHASE_ALL_GATHER = 2
ACTION_SEND_RECEIVE = 3

ARITHMETIC_POLICY = {
    "schema": "amdgpu-sim.ccl-ring-sum-arithmetic.v1",
    "bfloat16": "decode-binary32, add-binary32, round-to-bfloat16-rne-after-each-ring-reduce-step",
    "float32": "decode-binary32, add-binary32, round-to-binary32-rne-after-each-ring-reduce-step",
    "all_gather": "bitwise-copy",
    "oracle_phase": "post_target",
    "oracle_feedback": False,
}

RANK_FILES = (
    "worker-result.json",
    "step-journal.jsonl",
    "dispatch-trace.jsonl",
    "stats.txt",
    "gem5.log",
    "rank-launch.json",
    "input.bin",
    "output.bin",
)

IDENTITY_ROLES = (
    "product_manifest",
    "runtime_library",
    "ccl_native",
    "ccl_device",
    "ccl_engine",
    "triton_driver",
    "gem5_binary",
    "gem5_config",
    "verifier",
    "runner",
    "worker",
    "bootstrap",
    "design",
    "rank_registry",
)

# The runner records the verifier that performed its preflight, but acceptance is
# intentionally a later, read-only phase.  A verifier bug fix must not require a
# second target run; the accepted bundle separately carries the exact verifier
# bytes that made the acceptance decision.
LIVE_EXECUTION_IDENTITY_ROLES = tuple(
    role for role in IDENTITY_ROLES if role != "verifier"
)

TRACE_KEYS = (
    "schema", "event", "sim_tick", "daemon_uuid", "job_uuid", "epoch",
    "rank", "world_size", "connection_id", "owner_fd", "owner_generation",
    "request_id", "trace_id", "ticket_id", "dispatch_id", "kernel",
    "image_sha256", "grid", "workgroup", "fixed_shared_memory_bytes",
    "dynamic_shared_memory_bytes", "total_shared_memory_bytes", "kernarg_va",
    "kernarg_size", "packet_va", "packet_crc32c", "allocation_count",
    "allocations", "packet_fetches", "command_processor_submissions",
    "gpu_dispatcher_starts", "waves_started", "instructions_started",
    "instruction_wave_count", "scalar_reads", "global_reads", "global_writes",
    "store_events", "store_dwords", "workgroups_completed", "signal_before",
    "signal_after", "admission_tick", "start_tick", "end_tick", "retire_tick",
    "compatibility_completion_token_crc32c", "native_queue_retired",
    "application_pins_released", "adapter_released", "type20_durable",
    "unmap_durable", "owner_disconnected", "owner_quarantined",
    "cleanup_complete", "kernel_executed", "numerical_oracle",
)
TRACE_ALLOCATION_KEYS = ("allocation_id", "generation", "gpu_va", "bytes")

TRANSFER_KEYS = (
    "descriptor_sha256",
    "sequence",
    "phase",
    "step_index",
    "chunk_index",
    "source_rank",
    "destination_rank",
    "slot_index",
    "slot_generation",
)

STEP_EVENTS_COMMON = (
    "outbound_prepared",
    "outbound_DATA_sent",
    "inbound_DATA_received",
    "inbound_staged",
)

FIRST_ERROR_KEYS = (
    "status",
    "native_status",
    "reporter_rank",
    "failed_rank",
    "context_sequence",
)

FD_IDENTITY_KEYS = ("fd", "device", "inode", "mode", "target")
PROCESS_IDENTITY_KEYS = ("rank", "role", "pid", "start_time_ticks")
SUPERVISOR_CLEANUP_KEYS = (
    "baseline_fds",
    "baseline_fd_count",
    "baseline_fd_sha256",
    "post_fds",
    "post_fd_count",
    "post_fd_sha256",
    "added_fds",
    "removed_fds",
    "measured_fd_delta",
    "children_exhausted",
    "workers_reaped",
    "new_child_identities",
    "orphan_identities",
    "all_clear",
)


class AcceptanceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def exact_keys(value: Any, expected: Iterable[str], label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    actual = set(value)
    wanted = set(expected)
    require(actual == wanted, f"{label} keys differ: missing={sorted(wanted-actual)} extra={sorted(actual-wanted)}")
    return value


def integer(value: Any, minimum: int, maximum: int, label: str) -> int:
    require(type(value) is int and minimum <= value <= maximum,
            f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def hex_string(value: Any, pattern: re.Pattern[str], label: str) -> str:
    require(isinstance(value, str) and pattern.fullmatch(value) is not None,
            f"{label} must be canonical lowercase hex")
    return value


def canonical_json(value: object) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise AcceptanceError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _open_regular(path: Path, limit: int) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AcceptanceError(f"cannot open regular artifact {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode), f"artifact is not a regular file: {path}")
        require(metadata.st_uid == os.getuid(), f"artifact has wrong owner: {path}")
        require(metadata.st_size <= limit, f"artifact exceeds size limit: {path}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            require(size <= limit, f"artifact exceeds size limit: {path}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        stable = lambda value: (
            value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns,
        )
        require(stable(after) == stable(metadata), f"artifact changed while read: {path}")
        return b"".join(chunks), metadata
    finally:
        os.close(descriptor)


def file_record(path: Path, *, limit: int = MAX_ARTIFACT_BYTES) -> tuple[bytes, dict[str, Any]]:
    payload, metadata = _open_regular(path, limit)
    return payload, {
        "bytes": metadata.st_size,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def parse_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"{label} is not ASCII JSON: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def read_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, record = file_record(path, limit=MAX_JSON_BYTES)
    value = parse_json_bytes(payload, label)
    require(payload == canonical_json(value), f"{label} is not canonical JSON")
    return value, record


def parse_jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    if not payload:
        return []
    require(payload.endswith(b"\n"), f"{label} has an unterminated record")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        require(line, f"{label} contains a blank record")
        try:
            value = json.loads(line.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AcceptanceError(f"{label}:{line_number} is invalid JSON: {error}") from error
        require(isinstance(value, dict), f"{label}:{line_number} is not an object")
        require(line + b"\n" == canonical_json(value),
                f"{label}:{line_number} is not canonical JSON")
        records.append(value)
    return records


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def trace_json(value: Mapping[str, Any]) -> bytes:
    require(tuple(value) == TRACE_KEYS, "trace field order differs from the emitter contract")
    allocations = value.get("allocations")
    require(isinstance(allocations, list)
            and all(isinstance(item, Mapping)
                    and tuple(item) == TRACE_ALLOCATION_KEYS for item in allocations),
            "trace allocation field order differs from the emitter contract")
    try:
        text = json.dumps(value, sort_keys=False, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise AcceptanceError(f"trace is not strict JSON: {error}") from error
    return text.encode("ascii") + b"\n"


def parse_trace_jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    if not payload:
        return []
    require(payload.endswith(b"\n"), f"{label} has an unterminated record")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        require(line, f"{label} contains a blank record")
        try:
            value = json.loads(
                line.decode("ascii"), object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise AcceptanceError(
                f"{label}:{line_number} is invalid strict JSON: {error}"
            ) from error
        require(isinstance(value, dict), f"{label}:{line_number} is not an object")
        require(line + b"\n" == trace_json(value),
                f"{label}:{line_number} does not match the fixed emitter encoding")
        records.append(value)
    return records


def validate_absent_output(path: Path) -> None:
    require(path.is_absolute() and path == Path(os.path.normpath(path)),
            "output path must be canonical absolute")
    require(not path.exists() and not path.is_symlink(), "output path must be absent")
    parent = path.parent.resolve(strict=True)
    require(path.parent == parent, "output parent must be canonical")


def validate_source_inventory(source: Path, world: int) -> None:
    require(source.is_absolute() and source == Path(os.path.normpath(source)),
            "source path must be canonical absolute")
    metadata = source.lstat()
    require(stat.S_ISDIR(metadata.st_mode) and not source.is_symlink(),
            "source must be a real directory")
    expected_dirs = {f"rank-{rank:02d}" for rank in range(world)}
    expected_files = {"result-manifest.json"}
    actual_dirs: set[str] = set()
    actual_files: set[str] = set()
    for child in os.scandir(source):
        if child.is_symlink():
            raise AcceptanceError(f"source symlink is forbidden: {child.path}")
        if child.is_dir(follow_symlinks=False):
            actual_dirs.add(child.name)
        elif child.is_file(follow_symlinks=False):
            actual_files.add(child.name)
        else:
            raise AcceptanceError(f"source contains non-file entry: {child.path}")
    require(actual_dirs == expected_dirs and actual_files == expected_files,
            f"source inventory differs: dirs={sorted(actual_dirs)} files={sorted(actual_files)}")
    for rank_dir_name in sorted(expected_dirs):
        rank_dir = source / rank_dir_name
        metadata = rank_dir.lstat()
        require(stat.S_ISDIR(metadata.st_mode) and not rank_dir.is_symlink(),
                f"rank directory is unsafe: {rank_dir}")
        children = list(os.scandir(rank_dir))
        require({child.name for child in children} == set(RANK_FILES),
                f"rank artifact inventory differs: {rank_dir}")
        for child in children:
            require(child.is_file(follow_symlinks=False) and not child.is_symlink(),
                    f"rank artifact is not a regular file: {child.path}")


def descriptor_sha256(config: Mapping[str, Any], sequence: int, count: int) -> str:
    dtype_code = DTYPES[str(config["dtype"])][0]
    wire = bytearray(160)
    struct.pack_into(">I", wire, 0, 0x53434331)
    struct.pack_into(">H", wire, 4, 1)
    struct.pack_into(">H", wire, 6, 0)
    for offset, value in (
        (8, 160), (12, 0), (16, 1), (20, 1), (24, dtype_code),
        (28, 0), (32, int(config["world_size"])), (36, UINT32_MAX),
    ):
        struct.pack_into(">I", wire, offset, value)
    for offset, value in (
        (40, sequence), (48, count), (56, count),
        (64, int(config["epoch"])), (72, int(config["group_generation"])),
    ):
        struct.pack_into(">Q", wire, offset, value)
    wire[80:96] = bytes.fromhex(str(config["job_uuid"]))
    wire[96:112] = bytes.fromhex(str(config["group_uuid"]))
    wire[112:144] = bytes.fromhex(str(config["model_identity_sha256"]))
    struct.pack_into(">I", wire, 144, crc32c(wire[:144]))
    return hashlib.sha256(wire).hexdigest()


def crc32c(payload: bytes | bytearray) -> int:
    value = UINT32_MAX
    for byte in payload:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
    return value ^ UINT32_MAX


def chunk_range(total: int, world: int, chunk: int) -> tuple[int, int]:
    base, remainder = divmod(total, world)
    return chunk * base + min(chunk, remainder), base + (chunk < remainder)


def independent_plan(rank: int, world: int, count: int, dtype_bytes: int,
                     global_base: int) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    for phase in (PHASE_REDUCE_SCATTER, PHASE_ALL_GATHER):
        for phase_step in range(world - 1):
            if phase == PHASE_REDUCE_SCATTER:
                send_chunk = (rank + world - phase_step - 1) % world
                receive_chunk = (rank + world - phase_step - 2) % world
            else:
                send_chunk = (rank + world - phase_step) % world
                receive_chunk = (rank + world - phase_step - 1) % world
            send_offset, send_count = chunk_range(count, world, send_chunk)
            receive_offset, receive_count = chunk_range(count, world, receive_chunk)
            result.append({
                "ordinal": len(result),
                "phase": phase,
                "phase_step_index": phase_step,
                "action": ACTION_SEND_RECEIVE,
                "send_rank": (rank + 1) % world,
                "receive_rank": (rank + world - 1) % world,
                "send_chunk": send_chunk,
                "receive_chunk": receive_chunk,
                "send_offset_elements": send_offset,
                "send_count_elements": send_count,
                "receive_offset_elements": receive_offset,
                "receive_count_elements": receive_count,
                "global_send_offset_elements": global_base + send_offset,
                "global_receive_offset_elements": global_base + receive_offset,
                "send_payload_bytes": send_count * dtype_bytes,
                "receive_payload_bytes": receive_count * dtype_bytes,
            })
    return result


def validate_expected(expected: Mapping[str, Any]) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    exact_keys(expected, ("schema", "arithmetic_policy", "design"), "expected wrapper")
    require(expected["schema"] == EXPECTED_SCHEMA, "expected wrapper schema mismatch")
    require(expected["arithmetic_policy"] == ARITHMETIC_POLICY,
            "arithmetic policy is not exact version v1")
    design = expected["design"]
    require(isinstance(design, Mapping) and design.get("schema") == DESIGN_SCHEMA,
            "design schema mismatch")
    config = design.get("config")
    require(isinstance(config, Mapping), "design config missing")
    world = integer(config.get("world_size"), 2, 16, "world_size")
    total = integer(config.get("element_count"), 1, (1 << 63) - 1, "element_count")
    dtype = config.get("dtype")
    require(dtype in DTYPES and config.get("dtype_bytes") == DTYPES[str(dtype)][1],
            "dtype contract mismatch")
    hex_string(config.get("job_uuid"), HEX32, "job_uuid")
    hex_string(config.get("group_uuid"), HEX32, "group_uuid")
    hex_string(config.get("model_identity_sha256"), HEX64, "model identity")
    dtype_bytes = DTYPES[str(dtype)][1]
    segmentation = design.get("segmentation")
    require(isinstance(segmentation, Mapping), "design segmentation missing")
    segments = segmentation.get("segments")
    require(isinstance(segments, list) and segments, "design segments missing")
    base = 0
    independent: list[list[dict[str, Any]]] = [[] for _ in range(world)]
    ranks = design.get("ranks")
    require(isinstance(ranks, list) and [item.get("rank") for item in ranks] == list(range(world)),
            "design ranks are not canonical")
    for rank, item in enumerate(ranks):
        rank_segments = item.get("segments") if isinstance(item, Mapping) else None
        require(isinstance(rank_segments, list) and len(rank_segments) == len(segments),
                f"design rank {rank} segment chain mismatch")
    for segment_index, segment in enumerate(segments):
        require(isinstance(segment, Mapping), "segment summary is not an object")
        count = integer(segment.get("element_count"), 1, MAX_SEGMENT_BYTES // dtype_bytes,
                        "segment element_count")
        require(segment.get("index") == segment_index
                and segment.get("sequence") == segment_index + 1
                and segment.get("base_offset_elements") == base
                and segment.get("byte_count") == count * dtype_bytes
                and count * dtype_bytes <= MAX_SEGMENT_BYTES,
                "segment extent/sequence is not contiguous and 16MiB bounded")
        digest = descriptor_sha256(config, segment_index + 1, count)
        require(segment.get("descriptor_sha256") == digest,
                "segment descriptor digest was not independently reproduced")
        for rank in range(world):
            expected_steps = independent_plan(rank, world, count, dtype_bytes, base)
            rank_segment = ranks[rank]["segments"][segment_index]
            require(isinstance(rank_segment, Mapping),
                    f"design rank {rank} segment is not an object")
            require(rank_segment.get("descriptor_sha256") == digest
                    and rank_segment.get("steps") == expected_steps
                    and rank_segment.get("plan_sha256") == object_sha256(expected_steps),
                    "runtime planner design was not independently reproduced")
            independent[rank].append({
                "segment_id": segment_index,
                "sequence": segment_index + 1,
                "base_offset_elements": base,
                "element_count": count,
                "descriptor_sha256": digest,
                "steps": expected_steps,
                "plan_sha256": object_sha256(expected_steps),
            })
        base += count
    require(base == total, "segments do not exactly cover the public tensor")
    require(config.get("first_sequence") == 1
            and config.get("last_sequence") == len(segments),
            "design descriptor sequence bounds mismatch")
    return dict(design), independent


def artifact_descriptor(value: Any, expected_path: str, actual: Mapping[str, Any], label: str) -> None:
    exact_keys(value, ("path", "bytes", "sha256"), label)
    require(value.get("path") == expected_path
            and value.get("bytes") == actual["bytes"]
            and value.get("sha256") == actual["sha256"],
            f"{label} descriptor/hash mismatch")


def validate_identity_snapshot(snapshot: Any, live: bool) -> dict[str, Any]:
    require(isinstance(snapshot, Mapping) and set(snapshot) == set(IDENTITY_ROLES),
            "source identity roles differ")
    result: dict[str, Any] = {}
    for role in IDENTITY_ROLES:
        record = exact_keys(snapshot[role], ("path", "bytes", "sha256"), f"identity {role}")
        path = Path(str(record["path"]))
        require(path.is_absolute() and path == Path(os.path.normpath(path)),
                f"identity {role} path is not canonical")
        hex_string(record["sha256"], HEX64, f"identity {role} SHA")
        integer(record["bytes"], 0, MAX_ARTIFACT_BYTES, f"identity {role} bytes")
        if live and role in LIVE_EXECUTION_IDENTITY_ROLES:
            _, observed = file_record(path)
            require(observed == {"bytes": record["bytes"], "sha256": record["sha256"]},
                    f"live source identity drifted: {role}")
        result[role] = dict(record)
    require(Path(result["verifier"]["path"]) == THIS_FILE,
            "source identity verifier path mismatch")
    require(Path(result["runner"]["path"]) == RUNNER_FILE,
            "source identity runner path mismatch")
    require(Path(result["worker"]["path"]) == WORKER_FILE,
            "source identity worker path mismatch")
    require(Path(result["bootstrap"]["path"]) == BOOTSTRAP_FILE,
            "source identity bootstrap path mismatch")
    require(Path(result["design"]["path"]) == DESIGN_FILE,
            "source identity design path mismatch")
    require(
        Path(result["rank_registry"]["path"])
        == ROOT / "scripts/gemsim_live_registry.py",
        "source identity rank registry path mismatch",
    )
    require(
        Path(result["ccl_engine"]["path"]).name == "engine.py"
        and Path(result["ccl_engine"]["path"]).parent
        == Path(result["ccl_native"]["path"]).parent
        == Path(result["ccl_device"]["path"]).parent,
        "source identity CCL package topology mismatch",
    )
    if live:
        device_source = Path(result["ccl_device"]["path"]).read_text(
            encoding="ascii"
        )
        require(
            "class DeviceSumExecutor" in device_source
            and "target.backend != \"gemsim_amd\"" in device_source
            and "target.arch != \"gfx950\"" in device_source
            and "_sum_kernel[grid]" in device_source
            and "self._launch_count += 1" in device_source
            and "host_reduction_count=0" in device_source,
            "static device target path audit failed",
        )
        try:
            engine_tree = ast.parse(
                Path(result["ccl_engine"]["path"]).read_text(encoding="ascii")
            )
        except (OSError, UnicodeError, SyntaxError) as error:
            raise AcceptanceError("reusable CCL engine source is invalid") from error
        engine_classes = [
            node
            for node in engine_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AllReduceEngine"
        ]
        engine_methods = {
            node.name: node
            for node in engine_classes[0].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        } if len(engine_classes) == 1 else {}
        require(
            set(("join", "all_reduce", "close", "destroy", "abort"))
            <= set(engine_methods)
            and any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "join_rank"
                for node in ast.walk(engine_methods["join"])
            ),
            "static reusable CCL engine path audit failed",
        )
    return result


def validate_product_identity(identity: Mapping[str, Any]) -> None:
    manifest_path = Path(identity["product_manifest"]["path"])
    manifest, observed = read_json(manifest_path, "product manifest")
    require(observed["sha256"] == identity["product_manifest"]["sha256"]
            and manifest.get("schema") == "amdgpu-sim.product-prefix.v1"
            and manifest.get("prefix") == str(manifest_path.parent),
            "product manifest identity mismatch")
    hex_string(manifest.get("product_id"), HEX64, "product manifest product id")
    artifacts = manifest.get("artifacts")
    managed = manifest.get("managed_inputs")
    plugins = manifest.get("plugins")
    require(
        isinstance(artifacts, Mapping)
        and isinstance(managed, Mapping)
        and isinstance(plugins, Mapping)
        and isinstance(artifacts.get("ccl_plugin_init"), Mapping)
        and isinstance(artifacts["ccl_plugin_init"].get("path"), str),
        "product manifest artifact bindings are missing",
    )
    snapshots = plugins.get("snapshots", {}) if isinstance(plugins, Mapping) else {}
    ccl_snapshot = snapshots.get("gemsim-ccl", {}) if isinstance(snapshots, Mapping) else {}
    ccl_package = ccl_snapshot.get("package_path") if isinstance(ccl_snapshot, Mapping) else None
    require(
        isinstance(ccl_package, str)
        and Path(ccl_package).is_absolute()
        and Path(ccl_package) == Path(os.path.normpath(ccl_package))
        and Path(ccl_package) == Path(artifacts["ccl_plugin_init"]["path"]).parent,
        "product manifest CCL package binding mismatch",
    )
    bindings = {
        "runtime_library": artifacts.get("runtime_library"),
        "ccl_native": {"path": str(Path(artifacts["ccl_plugin_init"]["path"]).with_name("native.py"))},
        "ccl_device": {"path": str(Path(artifacts["ccl_plugin_init"]["path"]).with_name("device.py"))},
        "ccl_engine": {"path": str(Path(ccl_package) / "engine.py")},
        "triton_driver": artifacts.get("triton_plugin_driver"),
        "gem5_binary": managed.get("gem5_binary"),
        "gem5_config": managed.get("gem5_config"),
    }
    for role, binding in bindings.items():
        require(isinstance(binding, Mapping) and binding.get("path") == identity[role]["path"],
                f"product manifest {role} path mismatch")
        _, observed = file_record(Path(identity[role]["path"]))
        require(observed == {"bytes": identity[role]["bytes"], "sha256": identity[role]["sha256"]},
                f"product {role} bytes drifted")
        if "sha256" in binding:
            require(binding["sha256"] == identity[role]["sha256"],
                    f"product manifest {role} SHA mismatch")
    inventory = manifest.get("inventory")
    require(isinstance(inventory, list), "product manifest inventory is missing")
    inventory_by_path = {
        item.get("path"): item
        for item in inventory
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }
    prefix = Path(manifest_path).parent
    for role in ("ccl_native", "ccl_device", "ccl_engine"):
        try:
            relative = Path(identity[role]["path"]).relative_to(prefix).as_posix()
        except ValueError as error:
            raise AcceptanceError(
                f"product manifest inventory {role} path escaped prefix"
            ) from error
        binding = inventory_by_path.get(relative)
        require(
            isinstance(binding, Mapping)
            and binding.get("kind") == "regular"
            and binding.get("bytes") == identity[role]["bytes"]
            and binding.get("sha256") == identity[role]["sha256"],
            f"product manifest inventory {role} mismatch",
        )


def validate_rank_product(
    value: Any, identity: Mapping[str, Mapping[str, Any]], rank: int
) -> None:
    product = exact_keys(
        value,
        ("product_id", "manifest_sha256", "prefix", "ccl_engine"),
        f"rank {rank} product execution binding",
    )
    manifest_path = Path(identity["product_manifest"]["path"])
    manifest, manifest_record = read_json(manifest_path, "rank product manifest")
    require(
        isinstance(product["product_id"], str)
        and HEX64.fullmatch(product["product_id"]) is not None,
        f"rank {rank} product id is not canonical",
    )
    hex_string(
        product["manifest_sha256"], HEX64, f"rank {rank} product manifest SHA"
    )
    require(
        manifest_record["sha256"] == identity["product_manifest"]["sha256"]
        and manifest.get("schema") == "amdgpu-sim.product-prefix.v1"
        and product["product_id"] == manifest.get("product_id")
        and product["manifest_sha256"] == identity["product_manifest"]["sha256"]
        and product["prefix"] == str(manifest_path.parent)
        and product["ccl_engine"] == identity["ccl_engine"],
        f"rank {rank} product/CCL engine execution binding mismatch",
    )


def transfer_tuple(record: Mapping[str, Any], direction: str) -> dict[str, Any]:
    value = record.get("transfer")
    exact_keys(value, TRANSFER_KEYS, f"{direction} transfer tuple")
    result = dict(value)
    hex_string(result["descriptor_sha256"], HEX64, "transfer descriptor SHA")
    for name in ("sequence", "phase", "step_index", "chunk_index", "source_rank",
                 "destination_rank", "slot_index", "slot_generation"):
        integer(result[name], 0 if name not in ("sequence", "slot_generation") else 1,
                (1 << 63) - 1, f"transfer {name}")
    return result


def expected_transfer(segment: Mapping[str, Any], step: Mapping[str, Any], rank: int,
                      direction: str) -> dict[str, Any]:
    outbound = direction == "outbound"
    return {
        "descriptor_sha256": segment["descriptor_sha256"],
        "sequence": segment["sequence"],
        "phase": step["phase"],
        "step_index": step["ordinal"],
        "chunk_index": step["send_chunk"] if outbound else step["receive_chunk"],
        "source_rank": rank if outbound else step["receive_rank"],
        "destination_rank": step["send_rank"] if outbound else rank,
    }


def validate_journal(records: list[dict[str, Any]], rank: int,
                     segments: list[dict[str, Any]], dtype_bytes: int) -> dict[str, Any]:
    require(records, f"rank {rank} journal is empty")
    require([record.get("ordinal") for record in records] == list(range(len(records))),
            f"rank {rank} journal ordinals are not exact")
    last_ns = 0
    by_step: dict[tuple[int, int], list[dict[str, Any]]] = {}
    public_commits: list[dict[str, Any]] = []
    for record in records:
        require(record.get("schema") == JOURNAL_SCHEMA and record.get("rank") == rank,
                f"rank {rank} journal identity mismatch")
        now = integer(record.get("monotonic_ns"), 1, (1 << 63) - 1, "journal monotonic_ns")
        require(now >= last_ns, f"rank {rank} journal monotonic order regressed")
        last_ns = now
        if record.get("event") == "public_commit":
            public_commits.append(record)
            continue
        segment_id = integer(record.get("segment_id"), 0, len(segments) - 1, "journal segment_id")
        step_ordinal = integer(record.get("step_ordinal"), 0,
                               len(segments[segment_id]["steps"]) - 1, "journal step")
        by_step.setdefault((segment_id, step_ordinal), []).append(record)
    require(len(public_commits) == 1 and public_commits[0]["ordinal"] == len(records) - 1,
            f"rank {rank} must have one final public commit")
    dispatch_steps: list[tuple[int, int, int]] = []
    transfers: list[dict[str, Any]] = []
    credit_last: dict[tuple[str, int, int], int] = {}
    tuple_uniqueness: set[tuple[Any, ...]] = set()
    for segment in segments:
        for step in segment["steps"]:
            key = (segment["segment_id"], step["ordinal"])
            events = by_step.get(key)
            require(events is not None, f"rank {rank} journal missing step {key}")
            names = [record.get("event") for record in events]
            nonzero_rs = step["phase"] == PHASE_REDUCE_SCATTER and step["receive_count_elements"] > 0
            middle = ("device_call_enter", "device_call_returned") if nonzero_rs else (
                ("zero_no_dispatch",) if step["phase"] == PHASE_REDUCE_SCATTER else ("copy_complete",)
            )
            expected_names = [
                "outbound_prepared", "outbound_DATA_sent", "inbound_DATA_received",
                "inbound_staged", *middle, "inbound_CONSUMED_send_attempt",
                "inbound_CONSUMED_sent", "outbound_CONSUMED_received_credit_released",
                "step_complete",
            ]
            require(names == expected_names, f"rank {rank} step {key} event sequence mismatch")
            for record in events:
                require(record.get("descriptor_sequence") == segment["sequence"]
                        and record.get("phase") == step["phase"]
                        and record.get("phase_step_index") == step["phase_step_index"]
                        and record.get("planner") == {
                            "send_rank": step["send_rank"],
                            "receive_rank": step["receive_rank"],
                            "send_chunk": step["send_chunk"],
                            "receive_chunk": step["receive_chunk"],
                            "send_offset_elements": step["send_offset_elements"],
                            "send_count_elements": step["send_count_elements"],
                            "receive_offset_elements": step["receive_offset_elements"],
                            "receive_count_elements": step["receive_count_elements"],
                        },
                        f"rank {rank} step {key} planner identity mismatch")
                event = str(record["event"])
                direction = "outbound" if event.startswith("outbound") else "inbound"
                if event in {"outbound_prepared", "outbound_DATA_sent",
                             "outbound_CONSUMED_received_credit_released",
                             "inbound_DATA_received", "inbound_staged",
                             "inbound_CONSUMED_send_attempt", "inbound_CONSUMED_sent"}:
                    observed = transfer_tuple(record, direction)
                    fixed = expected_transfer(segment, step, rank, direction)
                    require(all(observed[name] == value for name, value in fixed.items()),
                            f"rank {rank} step {key} exact transfer tuple mismatch")
                    identity = tuple(observed[name] for name in TRANSFER_KEYS)
                    if event in {"outbound_prepared", "inbound_DATA_received"}:
                        require(identity not in tuple_uniqueness,
                                f"rank {rank} duplicate transfer tuple")
                        tuple_uniqueness.add(identity)
            outbound = [transfer_tuple(event, "outbound") for event in events
                        if str(event["event"]).startswith("outbound") and "transfer" in event]
            inbound = [transfer_tuple(event, "inbound") for event in events
                       if str(event["event"]).startswith("inbound") and "transfer" in event]
            require(outbound and all(item == outbound[0] for item in outbound),
                    f"rank {rank} outbound tuple changed within step")
            require(inbound and all(item == inbound[0] for item in inbound),
                    f"rank {rank} inbound tuple changed within step")
            for direction, observed in (("outbound", outbound[0]), ("inbound", inbound[0])):
                require(observed["slot_index"] < 2,
                        f"rank {rank} {direction} transfer exceeds credit slots")
                peer = (observed["destination_rank"] if direction == "outbound"
                        else observed["source_rank"])
                credit_key = (direction, peer, observed["slot_index"])
                require(observed["slot_generation"] > credit_last.get(credit_key, 0),
                        f"rank {rank} {direction} slot generation did not advance")
                credit_last[credit_key] = observed["slot_generation"]
            transfers.append({
                "segment_id": segment["segment_id"],
                "step_ordinal": step["ordinal"],
                "outbound": outbound[0],
                "inbound": inbound[0],
            })
            staging = events[names.index("inbound_staged")]
            hex_string(staging.get("staging_sha256"), HEX64,
                       f"rank {rank} step {key} staging SHA")
            require(staging.get("immutable_bytes") is True,
                    f"rank {rank} step {key} staging is not immutable")
            if "payload_bytes" in staging:
                require(
                    staging["payload_bytes"]
                    == step["receive_count_elements"] * dtype_bytes,
                    f"rank {rank} step {key} staging extent mismatch",
                )
            if nonzero_rs:
                returned = names.index("device_call_returned")
                ack_attempt = names.index("inbound_CONSUMED_send_attempt")
                require(returned < ack_attempt, "inbound ACK precedes synchronous device return")
                dispatch_steps.append((segment["segment_id"], step["ordinal"],
                                       step["receive_count_elements"]))
            elif step["phase"] == PHASE_ALL_GATHER:
                require("device_call_enter" not in names and "device_call_returned" not in names,
                        "all-gather performed SUM")
    return {
        "records": len(records),
        "steps": len(by_step),
        "device_steps": dispatch_steps,
        "public_commit_count": 1,
        "last_monotonic_ns": last_ns,
        "transfers": transfers,
    }


def validate_peer_transfers(summaries: Sequence[Mapping[str, Any]]) -> None:
    outbound: list[tuple[Any, ...]] = []
    inbound: list[tuple[Any, ...]] = []
    for summary in summaries:
        for step in summary["transfers"]:
            outbound.append(tuple(step["outbound"][name] for name in TRANSFER_KEYS))
            inbound.append(tuple(step["inbound"][name] for name in TRANSFER_KEYS))
    require(sorted(outbound) == sorted(inbound),
            "cross-rank outbound/inbound transfer tuples are not peer-exact")


TRACE_MATCH_FIELDS = (
    "daemon_uuid", "job_uuid", "epoch", "rank", "world_size", "connection_id",
    "owner_fd", "owner_generation", "request_id", "trace_id", "ticket_id", "dispatch_id",
    "kernel", "image_sha256", "grid", "workgroup", "packet_crc32c",
    "allocation_count", "allocations", "packet_fetches",
    "command_processor_submissions", "gpu_dispatcher_starts", "waves_started",
    "instructions_started", "global_reads", "global_writes", "store_events",
    "workgroups_completed", "signal_before", "signal_after", "admission_tick",
    "start_tick", "end_tick", "retire_tick", "kernel_executed",
    "native_queue_retired", "type20_durable",
)


def validate_kernel_trace(
    records: list[dict[str, Any]],
    kernels: Sequence[Mapping[str, Any]],
    rank: int,
    session: Mapping[str, Any],
    config: Mapping[str, Any],
    log_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one ordered sequence of ordinary generic kernel dispatches.

    Kernel-specific traffic expectations are supplied by the external evidence
    verifier.  The lifecycle, identity, retirement, reuse, and cleanup contract
    remains shared across every operation family.
    """
    if not kernels:
        require(not records, f"rank {rank} zero-dispatch trace is not empty")
        return {
            "dispatch_count": 0, "lifecycle_records": 0,
            "reuse_complete_count": 0, "session_complete_count": 0,
            "terminal_sim_tick": log_identity["exit_tick"],
        }
    require(records, f"rank {rank} trace is empty")
    terminal = [record for record in records
                if record.get("event") == "generic_execution_session_complete"]
    reuse = [record for record in records
             if record.get("event") == "generic_execution_reuse_complete"]
    require(len(terminal) == 1 and records[-1] is terminal[0],
            f"rank {rank} trace needs one final session_complete")
    require(len(reuse) == max(0, len(kernels) - 1),
            f"rank {rank} trace reuse lifecycle count mismatch")
    require(len(records) == 3 * len(kernels),
            f"rank {rank} authoritative trace dispatch count mismatch")
    identities: set[tuple[Any, ...]] = set()
    expected_identity = {
        "daemon_uuid": session["daemon_uuid"],
        "job_uuid": config["job_uuid"],
        "epoch": config["epoch"],
        "rank": rank,
        "world_size": config["world_size"],
        "connection_id": session["connection_id"],
        "owner_fd": log_identity["owner_fd"],
        "owner_generation": log_identity["owner_generation"],
    }
    require(all(all(record.get(name) == value for name, value in expected_identity.items())
                for record in records),
            f"rank {rank} trace run/rank identity mismatch")
    for launch_index, expectation in enumerate(kernels):
        require(isinstance(expectation, Mapping),
                f"rank {rank} kernel expectation is not an object")
        retired, durable, completion = records[3 * launch_index:3 * launch_index + 3]
        require(retired.get("schema") == TRACE_SCHEMA and durable.get("schema") == TRACE_SCHEMA,
                f"rank {rank} trace schema mismatch")
        require(retired.get("event") == "generic_execution_retired"
                and durable.get("event") == "generic_execution_type20_durable",
                f"rank {rank} lifecycle pair order mismatch")
        require(completion.get("event") == (
            "generic_execution_session_complete"
            if launch_index == len(kernels) - 1
            else "generic_execution_reuse_complete"
        ), f"rank {rank} dispatch terminal event mismatch")
        identity = tuple(retired.get(name) for name in ("request_id", "trace_id", "ticket_id", "dispatch_id"))
        require(identity not in identities, f"rank {rank} duplicate trace dispatch identity")
        identities.add(identity)
        require(all(durable.get(field) == retired.get(field)
                    for field in TRACE_MATCH_FIELDS if field != "type20_durable"),
                f"rank {rank} retired/type20 records do not match")
        require(all(completion.get(field) == retired.get(field)
                    for field in TRACE_MATCH_FIELDS if field not in ("type20_durable",)),
                f"rank {rank} completion record does not match dispatch")
        require(retired.get("kernel") == expectation.get("kernel")
                and isinstance(retired.get("image_sha256"), str)
                and HEX64.fullmatch(retired["image_sha256"]) is not None
                and retired.get("type20_durable") is False
                and durable.get("type20_durable") is True
                and retired.get("native_queue_retired") is True
                and retired.get("kernel_executed") is True
                and retired.get("packet_fetches") == 1
                and retired.get("command_processor_submissions") == 1
                and retired.get("gpu_dispatcher_starts") == 1
                and retired.get("waves_started", 0) > 0
                and retired.get("instructions_started", 0) > 0
                and retired.get("global_reads", 0) > 0
                and retired.get("global_writes", 0) > 0
                and retired.get("store_events", 0) > 0
                and retired.get("workgroups_completed")
                == expectation.get("workgroups_completed")
                and retired.get("signal_before") == 1
                and retired.get("signal_after") == 0,
                f"rank {rank} trace does not prove normal device execution")
        grid = retired.get("grid")
        require(isinstance(grid, list) and len(grid) == 3
                and grid == expectation.get("grid"),
                f"rank {rank} trace grid does not match planner element count")
        require(retired.get("workgroup") == expectation.get("workgroup")
                and retired.get("allocation_count") == expectation.get("allocation_count")
                and isinstance(retired.get("allocations"), list)
                and len(retired["allocations"]) == expectation.get("allocation_count"),
                f"rank {rank} trace allocation contract mismatch")
        for metric in ("global_reads", "global_writes"):
            expected_metric = expectation.get(metric)
            if expected_metric is not None:
                require(retired.get(metric) == expected_metric,
                        f"rank {rank} trace {metric} differs")
        admission = integer(retired.get("admission_tick"), 1, (1 << 63) - 1,
                            "trace admission_tick")
        start = integer(retired.get("start_tick"), admission, (1 << 63) - 1,
                        "trace start_tick")
        end = integer(retired.get("end_tick"), start, (1 << 63) - 1,
                      "trace end_tick")
        retire = integer(retired.get("retire_tick"), end, (1 << 63) - 1,
                         "trace retire_tick")
        retired_tick = integer(retired.get("sim_tick"), retire, (1 << 63) - 1,
                               "retired sim_tick")
        durable_tick = integer(durable.get("sim_tick"), retired_tick,
                               (1 << 63) - 1, "durable sim_tick")
        completion_tick = integer(completion.get("sim_tick"), durable_tick,
                                  (1 << 63) - 1, "completion sim_tick")
        require(end == retire == retired_tick,
                f"rank {rank} trace retirement/durability order mismatch")
        if launch_index + 1 < len(kernels):
            next_retired = records[3 * (launch_index + 1)]
            require(completion_tick == next_retired.get("admission_tick"),
                    f"rank {rank} reuse/next-admission tick handoff mismatch")
    final = terminal[0]
    require(final.get("schema") == TRACE_SCHEMA
            and final.get("type20_durable") is True
            and final.get("unmap_durable") is True
            and final.get("owner_disconnected") is True
            and final.get("owner_quarantined") is False
            and final.get("cleanup_complete") is True,
            f"rank {rank} terminal cleanup trace incomplete")
    require(final.get("sim_tick") == log_identity["exit_tick"],
            f"rank {rank} terminal trace tick differs from gem5 exit")
    return {"dispatch_count": len(kernels), "lifecycle_records": len(records),
            "reuse_complete_count": len(reuse), "session_complete_count": 1,
            "terminal_sim_tick": final["sim_tick"],
            "kernels": [str(item["kernel"]) for item in kernels]}


def validate_trace(records: list[dict[str, Any]], device_steps: Sequence[tuple[int, int, int]],
                   rank: int, session: Mapping[str, Any], config: Mapping[str, Any],
                   log_identity: Mapping[str, Any]) -> dict[str, Any]:
    kernels = [
        {
            "kernel": "_sum_kernel",
            "grid": [math.ceil(count / 256) * 256, 1, 1],
            "workgroup": [256, 1, 1],
            "allocation_count": 2,
            "workgroups_completed": math.ceil(count / 256),
            "global_reads": 2 * count,
            "global_writes": count,
        }
        for _, _, count in device_steps
    ]
    return validate_kernel_trace(
        records, kernels, rank, session, config, log_identity
    )


def parse_stats(payload: bytes, rank: int, dispatches: int) -> dict[str, Any]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise AcceptanceError(f"rank {rank} stats are not ASCII") from error
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            values[parts[0]] = parts[1]
    require(values.get("system.host_gpu_bridge.host_fallback_count") == "0",
            f"rank {rank} authoritative host fallback stat is nonzero")
    require(int(values.get("simTicks", "0")) > 0, f"rank {rank} simTicks missing")
    require(float(values.get("hostSeconds", "0")) > 0, f"rank {rank} hostSeconds missing")
    if dispatches:
        require(any(int(float(value)) > 0 for key, value in values.items()
                    if "numInstrExecuted" in key),
                f"rank {rank} stats lack device instruction execution")
    return {"host_fallback_count": 0, "sim_ticks": int(values["simTicks"])}


def validate_managed_session(value: Any, rank: int, config: Mapping[str, Any],
                             launch: Mapping[str, Any], runtime: Mapping[str, Any],
                             runtime_identity: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "child_pid", "connection_id", "epoch", "rank", "world_size",
        "daemon_uuid", "job_uuid", "runtime_library", "rank_launch_sha256",
    }
    require(isinstance(value, Mapping) and set(value) == expected_keys,
            f"rank {rank} managed session keys differ")
    integer(value.get("child_pid"), 1, (1 << 31) - 1, "managed child_pid")
    integer(value.get("connection_id"), 1, (1 << 64) - 1, "managed connection_id")
    hex_string(value.get("daemon_uuid"), HEX32, "managed daemon_uuid")
    require(value.get("job_uuid") == config["job_uuid"]
            and value.get("epoch") == config["epoch"]
            and value.get("rank") == rank
            and value.get("world_size") == config["world_size"]
            and value.get("runtime_library") == runtime["path"]
            and value.get("rank_launch_sha256") == object_sha256(launch)
            and runtime_identity.get("path") == runtime["path"]
            and runtime_identity.get("sha256") == runtime["sha256"],
            f"rank {rank} managed session collective/product identity mismatch")
    return dict(value)


def validate_log(payload: bytes, rank: int, launch: Mapping[str, Any],
                 session: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise AcceptanceError(f"rank {rank} gem5 log is not ASCII") from error
    lowered = text.lower()
    require("fatal:" not in lowered and "panic:" not in lowered
            and "traceback (most recent call last)" not in lowered,
            f"rank {rank} gem5 log contains fatal diagnostics")
    ready = READY_PATTERN.findall(text)
    handshake = HANDSHAKE_PATTERN.findall(text)
    exits = EXIT_PATTERN.findall(text)
    pids = PID_PATTERN.findall(text)
    require(len(ready) == len(handshake) == len(exits) == len(pids) == 1,
            f"rank {rank} gem5 lifecycle identity count mismatch")
    endpoint, daemon_uuid, job_uuid, epoch, ready_rank, world, max_record = ready[0]
    owner_fd, owner_generation = handshake[0]
    exit_tick, stats_path = exits[0]
    paths = launch["paths"]
    require(endpoint == paths["endpoint"]
            and daemon_uuid == session["daemon_uuid"]
            and job_uuid == launch["job_uuid"]
            and int(epoch) == launch["epoch"]
            and int(ready_rank) == rank
            and int(world) == launch["world_size"]
            and int(max_record) > 0
            and int(pids[0]) == session["child_pid"]
            and stats_path == f'{paths["gem5_output_directory"]}/stats.txt',
            f"rank {rank} gem5 ready/exit identity mismatch")
    command_lines = [line[len("command line: "):] for line in text.splitlines()
                     if line.startswith("command line: ")]
    require(len(command_lines) == 1, f"rank {rank} gem5 command line count mismatch")
    try:
        command = shlex.split(command_lines[0])
    except ValueError as error:
        raise AcceptanceError(f"rank {rank} gem5 command line is malformed") from error
    require(len(command) >= 5
            and command[0] == identity["gem5_binary"]["path"]
            and command[1] == "--listener-mode=on"
            and command[2] == "--outdir"
            and command[3] == paths["gem5_output_directory"]
            and command[4] == identity["gem5_config"]["path"],
            f"rank {rank} gem5 binary/config command identity mismatch")
    expected_options = {
        "--outdir": paths["gem5_output_directory"],
        "--endpoint": paths["endpoint"],
        "--dispatch-trace-path": paths["dispatch_trace_path"],
        "--epoch": str(launch["epoch"]),
        "--job-uuid": launch["job_uuid"],
        "--rank": str(rank),
        "--world-size": str(launch["world_size"]),
    }
    for option, expected in expected_options.items():
        require(command.count(option) == 1 and command.index(option) + 1 < len(command)
                and command[command.index(option) + 1] == expected,
                f"rank {rank} gem5 command {option} mismatch")
    return {
        "fatal_diagnostic_absent": True,
        "successful_dispatch_exit": True,
        "daemon_uuid": daemon_uuid,
        "owner_fd": int(owner_fd),
        "owner_generation": int(owner_generation),
        "exit_tick": int(exit_tick),
    }


def _decode_values(payload: bytes, dtype: str) -> list[float]:
    width = DTYPES[dtype][1]
    require(len(payload) % width == 0, "tensor byte length is not dtype aligned")
    if dtype == "float32":
        return [item[0] for item in struct.iter_unpack("<f", payload)]
    values: list[float] = []
    for (bits,) in struct.iter_unpack("<H", payload):
        values.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
    return values


def _encode_value(value: float, dtype: str) -> bytes:
    fp32 = struct.unpack("<I", struct.pack("<f", value))[0]
    if dtype == "float32":
        return struct.pack("<I", fp32)
    upper = fp32 >> 16
    lower = fp32 & 0xFFFF
    upper = (upper + (lower > 0x8000 or (lower == 0x8000 and upper & 1))) & 0xFFFF
    return struct.pack("<H", upper)


def ring_oracle(inputs: Sequence[bytes], dtype: str,
                plans: Sequence[Sequence[Mapping[str, Any]]]) -> list[bytes]:
    decoded = [_decode_values(payload, dtype) for payload in inputs]
    total = len(decoded[0])
    world = len(inputs)
    require(all(len(values) == total for values in decoded), "input tensor extents differ")
    outputs = [bytearray() for _ in range(world)]
    for segment_index in range(len(plans[0])):
        segment = plans[0][segment_index]
        base = int(segment["base_offset_elements"])
        count = int(segment["element_count"])
        workspaces = [list(values[base:base + count]) for values in decoded]
        for ordinal in range(2 * (world - 1)):
            sends: list[tuple[int, int, list[float]]] = []
            for source in range(world):
                step = plans[source][segment_index]["steps"][ordinal]
                offset = int(step["send_offset_elements"])
                extent = int(step["send_count_elements"])
                sends.append((source, int(step["send_rank"]),
                              list(workspaces[source][offset:offset + extent])))
            for source, destination, payload in sends:
                step = plans[destination][segment_index]["steps"][ordinal]
                require(step["receive_rank"] == source
                        and step["receive_count_elements"] == len(payload),
                        "oracle planner peer delivery mismatch")
                offset = int(step["receive_offset_elements"])
                if step["phase"] == PHASE_REDUCE_SCATTER:
                    for index, value in enumerate(payload):
                        summed = struct.unpack(
                            "<f", struct.pack("<f", workspaces[destination][offset + index] + value)
                        )[0]
                        rounded = _decode_values(_encode_value(summed, dtype), dtype)[0]
                        workspaces[destination][offset + index] = rounded
                else:
                    workspaces[destination][offset:offset + len(payload)] = payload
        for rank in range(world):
            for value in workspaces[rank]:
                outputs[rank].extend(_encode_value(value, dtype))
    return [bytes(output) for output in outputs]


def validate_rank_result(value: Mapping[str, Any], rank: int, world: int,
                         status: str) -> None:
    require(value.get("schema") == RANK_RESULT_SCHEMA and value.get("rank") == rank
            and value.get("world_size") == world and value.get("status") == status,
            f"rank {rank} worker result identity/status mismatch")
    require(value.get("acceptance_authority") is False
            and value.get("live_collective_accepted") is False,
            f"rank {rank} worker falsely claims acceptance authority")


def validate_manifest_header(manifest: Mapping[str, Any], expected_record: Mapping[str, Any],
                             design: Mapping[str, Any]) -> tuple[str, int]:
    require(manifest.get("schema") == RUN_SCHEMA, "run manifest schema mismatch")
    require(manifest.get("acceptance_authority") is False
            and manifest.get("live_collective_accepted") is False,
            "runner falsely claims live acceptance")
    binding = manifest.get("expected")
    exact_keys(binding, ("schema", "bytes", "sha256"), "expected binding")
    require(binding == {"schema": EXPECTED_SCHEMA, **expected_record},
            "run manifest expected wrapper binding mismatch")
    config = design["config"]
    require(manifest.get("world_size") == config["world_size"]
            and manifest.get("element_count") == config["element_count"]
            and manifest.get("dtype") == config["dtype"]
            and manifest.get("job_uuid") == config["job_uuid"]
            and manifest.get("group_uuid") == config["group_uuid"]
            and manifest.get("epoch") == config["epoch"]
            and manifest.get("group_generation") == config["group_generation"],
            "run manifest collective identity mismatch")
    status = manifest.get("status")
    require(status in ("success", "device_failure", "peer_lost", "timed_out"),
            "run status is not canonical")
    return str(status), int(config["world_size"])


def _validate_fd_identities(value: Any, label: str) -> list[dict[str, Any]]:
    require(isinstance(value, list), f"{label} must be a list")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        record = exact_keys(item, FD_IDENTITY_KEYS, f"{label}[{index}]")
        result.append({
            "fd": integer(record["fd"], 0, (1 << 31) - 1, f"{label} fd"),
            "device": integer(record["device"], 0, (1 << 63) - 1,
                              f"{label} device"),
            "inode": integer(record["inode"], 0, (1 << 63) - 1,
                             f"{label} inode"),
            "mode": integer(record["mode"], 0, (1 << 32) - 1,
                            f"{label} mode"),
            "target": record["target"],
        })
        require(isinstance(record["target"], str) and record["target"],
                f"{label} target must be a nonempty string")
    require([item["fd"] for item in result] == sorted(item["fd"] for item in result)
            and len({item["fd"] for item in result}) == len(result),
            f"{label} is not uniquely ordered by descriptor")
    return result


def _validate_process_identities(value: Any, world: int,
                                 label: str) -> list[dict[str, Any]]:
    require(isinstance(value, list), f"{label} must be a list")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        record = exact_keys(item, PROCESS_IDENTITY_KEYS, f"{label}[{index}]")
        role = record["role"]
        require(role in ("worker", "daemon_or_descendant"),
                f"{label} role is not canonical")
        result.append({
            "rank": integer(record["rank"], 0, world - 1, f"{label} rank"),
            "role": role,
            "pid": integer(record["pid"], 1, (1 << 31) - 1, f"{label} pid"),
            "start_time_ticks": integer(
                record["start_time_ticks"], 1, (1 << 63) - 1,
                f"{label} start_time_ticks",
            ),
        })
    order = [(item["rank"], item["role"], item["pid"], item["start_time_ticks"])
             for item in result]
    require(order == sorted(order) and len(set(order)) == len(order),
            f"{label} is not unique and canonical")
    return result


def validate_supervisor_cleanup(value: Any, rank_entries: Sequence[Mapping[str, Any]],
                                rank_results: Sequence[Mapping[str, Any]],
                                world: int) -> dict[str, Any]:
    cleanup = exact_keys(value, SUPERVISOR_CLEANUP_KEYS, "supervisor cleanup")
    baseline = _validate_fd_identities(cleanup["baseline_fds"], "baseline fds")
    post = _validate_fd_identities(cleanup["post_fds"], "post fds")
    added = _validate_fd_identities(cleanup["added_fds"], "added fds")
    removed = _validate_fd_identities(cleanup["removed_fds"], "removed fds")
    baseline_set = {canonical_json(item) for item in baseline}
    post_set = {canonical_json(item) for item in post}
    require(added == [item for item in post if canonical_json(item) not in baseline_set]
            and removed == [item for item in baseline if canonical_json(item) not in post_set],
            "supervisor cleanup FD set algebra mismatch")
    require(cleanup["baseline_fd_count"] == len(baseline)
            and cleanup["post_fd_count"] == len(post)
            and cleanup["baseline_fd_sha256"] == object_sha256(baseline)
            and cleanup["post_fd_sha256"] == object_sha256(post),
            "supervisor cleanup FD count/hash mismatch")
    hex_string(cleanup["baseline_fd_sha256"], HEX64, "baseline FD SHA")
    hex_string(cleanup["post_fd_sha256"], HEX64, "post FD SHA")
    measured_delta = integer(cleanup["measured_fd_delta"], 0, (1 << 31) - 1,
                             "measured FD delta")
    require(measured_delta == len(added) + len(removed),
            "supervisor measured FD delta is not derived")

    children = _validate_process_identities(
        cleanup["new_child_identities"], world, "new child identities"
    )
    orphans = _validate_process_identities(
        cleanup["orphan_identities"], world, "orphan identities"
    )
    child_keys = {tuple(item[name] for name in PROCESS_IDENTITY_KEYS) for item in children}
    require(all(tuple(item[name] for name in PROCESS_IDENTITY_KEYS) in child_keys
                for item in orphans), "orphan identity was not an owned new child")
    expected_workers = {
        (rank, "worker", entry.get("worker_pid"), entry.get("worker_start_time_ticks"))
        for rank, entry in enumerate(rank_entries)
        if entry.get("worker_pid") is not None
    }
    observed_workers = {
        tuple(item[name] for name in PROCESS_IDENTITY_KEYS)
        for item in children if item["role"] == "worker"
    }
    require(observed_workers == expected_workers,
            "supervisor worker process identities differ from rank launches")
    for rank, result in enumerate(rank_results):
        session = result.get("managed_session")
        if isinstance(session, Mapping) and type(session.get("child_pid")) is int:
            require(any(item["rank"] == rank
                        and item["role"] == "daemon_or_descendant"
                        and item["pid"] == session["child_pid"] for item in children),
                    f"rank {rank} managed daemon is absent from supervisor ownership")

    for rank, entry in enumerate(rank_entries):
        rank_cleanup = exact_keys(entry.get("cleanup"),
                                  ("worker_reaped", "daemon_reaped"),
                                  f"rank {rank} cleanup")
        require(rank_cleanup["worker_reaped"] is True
                and rank_cleanup["daemon_reaped"] is True,
                f"rank {rank} lifecycle cleanup is incomplete")
    derived_clear = (
        not added and not removed and measured_delta == 0 and not orphans
        and cleanup["children_exhausted"] is True
        and cleanup["workers_reaped"] is True
    )
    require(cleanup["all_clear"] is derived_clear and derived_clear,
            "supervisor cleanup is not independently all-clear")
    return {
        "baseline_fd_count": len(baseline),
        "post_fd_count": len(post),
        "measured_fd_delta": measured_delta,
        "new_child_count": len(children),
        "orphan_count": len(orphans),
        "all_clear": True,
    }


def validate_failure(manifest: Mapping[str, Any], ranks: Sequence[Mapping[str, Any]],
                     status: str, world: int) -> dict[str, Any]:
    first_error = manifest.get("first_error")
    failed_tuple = manifest.get("failed_transfer")
    require(isinstance(first_error, Mapping) and first_error.get("status") == status,
            "failure first_error is not exact/canonical")
    sequence = integer(first_error.get("context_sequence"), 1, (1 << 63) - 1,
                       "failure context sequence")
    require(isinstance(failed_tuple, Mapping), "failure transfer tuple missing")
    transfer_tuple({"transfer": failed_tuple}, "failed")
    require(failed_tuple.get("sequence") == sequence, "failed tuple sequence differs")
    require(all(rank.get("first_error") == first_error
                and rank.get("public_commit_count") == 0
                and rank.get("public_result_published") is False
                and rank.get("failed_transfer") == failed_tuple
                for rank in ranks),
            "failure is not exact and group-wide")
    require(manifest.get("failed_ack_sent") is False
            and manifest.get("public_commit_count") == 0,
            "failed DATA was ACKed or committed")
    started = integer(manifest.get("started_at_ns"), 1, (1 << 63) - 1, "started_at_ns")
    completed = integer(manifest.get("completed_at_ns"), started, (1 << 63) - 1,
                        "completed_at_ns")
    deadline = integer(manifest.get("absolute_deadline_ns"), started + 1,
                       (1 << 63) - 1, "absolute_deadline_ns")
    require((completed >= deadline) is (status == "timed_out"),
            "timeout completion/deadline relation is incorrect")
    return {"status": status, "context_sequence": sequence,
            "completed_at_ns": completed, "absolute_deadline_ns": deadline}


def verify(source: Path, expected_path: Path, output: Path, *, live_identity: bool = True) -> dict[str, Any]:
    validate_absent_output(output)
    expected, expected_record = read_json(expected_path, "expected wrapper")
    design, plans = validate_expected(expected)
    world = int(design["config"]["world_size"])
    validate_source_inventory(source, world)
    manifest, manifest_record = read_json(source / "result-manifest.json", "run manifest")
    status, _ = validate_manifest_header(manifest, expected_record, design)
    preflight = validate_identity_snapshot(manifest.get("source_identity_preflight"), live_identity)
    postflight = validate_identity_snapshot(manifest.get("source_identity_postflight"), live_identity)
    require(preflight == postflight, "source identity drifted during execution")
    design_runtime = design.get("runtime")
    require(isinstance(design_runtime, Mapping)
            and design_runtime.get("path") == preflight["runtime_library"]["path"]
            and design_runtime.get("sha256") == preflight["runtime_library"]["sha256"],
            "expected design runtime differs from executed product identity")
    if live_identity:
        validate_product_identity(preflight)
    rank_entries = manifest.get("ranks")
    require(isinstance(rank_entries, list)
            and [entry.get("rank") for entry in rank_entries] == list(range(world)),
            "run manifest ranks are not canonical")
    all_artifacts: dict[str, tuple[bytes, dict[str, Any]]] = {}
    rank_results: list[dict[str, Any]] = []
    journal_summaries: list[dict[str, Any]] = []
    trace_summaries: list[dict[str, Any]] = []
    inputs: list[bytes] = []
    outputs: list[bytes] = []
    dtype = str(design["config"]["dtype"])
    tensor_bytes = int(design["config"]["element_count"]) * DTYPES[dtype][1]
    for rank, entry in enumerate(rank_entries):
        artifact_map = entry.get("artifacts")
        require(isinstance(artifact_map, Mapping) and set(artifact_map) == set(RANK_FILES),
                f"rank {rank} manifest artifact set differs")
        for name in RANK_FILES:
            relative = f"rank-{rank:02d}/{name}"
            payload, observed = file_record(source / relative,
                                            limit=MAX_JOURNAL_BYTES if name.endswith(".jsonl") else MAX_ARTIFACT_BYTES)
            artifact_descriptor(artifact_map[name], relative, observed, f"rank {rank} {name}")
            all_artifacts[relative] = (payload, observed)
        result = parse_json_bytes(all_artifacts[f"rank-{rank:02d}/worker-result.json"][0],
                                  f"rank {rank} worker result")
        require(all_artifacts[f"rank-{rank:02d}/worker-result.json"][0]
                == canonical_json(result), f"rank {rank} worker result is not canonical")
        validate_rank_result(result, rank, world, status)
        validate_rank_product(result.get("product"), preflight, rank)
        rank_results.append(result)
        launch = parse_json_bytes(all_artifacts[f"rank-{rank:02d}/rank-launch.json"][0],
                                  f"rank {rank} launch")
        require(all_artifacts[f"rank-{rank:02d}/rank-launch.json"][0]
                == canonical_json(launch), f"rank {rank} launch is not canonical")
        require(launch == design["ranks"][rank]["rank_launch"]
                and launch.get("schema") == RANK_LAUNCH_SCHEMA
                and object_sha256(launch) == design["ranks"][rank]["rank_launch_sha256"],
                f"rank {rank} launch descriptor mismatch")
        journal = parse_jsonl(all_artifacts[f"rank-{rank:02d}/step-journal.jsonl"][0],
                              f"rank {rank} journal")
        if status == "success":
            journal_summary = validate_journal(
                journal, rank, plans[rank], DTYPES[dtype][1]
            )
            require(journal_summary["device_steps"],
                    f"rank {rank} has zero device dispatches; formal live shutdown is unprovable")
            session = validate_managed_session(
                result.get("managed_session"), rank, design["config"], launch,
                design["runtime"], preflight["runtime_library"],
            )
            log_summary = validate_log(
                all_artifacts[f"rank-{rank:02d}/gem5.log"][0], rank,
                launch, session, preflight,
            )
            trace = parse_trace_jsonl(
                all_artifacts[f"rank-{rank:02d}/dispatch-trace.jsonl"][0],
                f"rank {rank} trace",
            )
            trace_summary = validate_trace(
                trace, journal_summary["device_steps"], rank, session,
                design["config"], log_summary,
            )
            stats_summary = parse_stats(
                all_artifacts[f"rank-{rank:02d}/stats.txt"][0], rank,
                trace_summary["dispatch_count"],
            )
            require(stats_summary["sim_ticks"] == log_summary["exit_tick"]
                    and (not trace_summary["dispatch_count"]
                         or trace_summary["terminal_sim_tick"] == stats_summary["sim_ticks"]),
                    f"rank {rank} trace/log/stats simulator tick mismatch")
            journal_summaries.append(journal_summary)
            trace_summaries.append(trace_summary)
        input_payload = all_artifacts[f"rank-{rank:02d}/input.bin"][0]
        output_payload = all_artifacts[f"rank-{rank:02d}/output.bin"][0]
        require(len(input_payload) == tensor_bytes, f"rank {rank} input extent mismatch")
        if status == "success":
            require(len(output_payload) == tensor_bytes, f"rank {rank} output extent mismatch")
        else:
            require(output_payload == b"", f"rank {rank} failure published output bytes")
        input_record = artifact_map["input.bin"]
        require(result.get("input_sha256_before") == input_record["sha256"]
                and result.get("input_sha256_after") == input_record["sha256"],
                f"rank {rank} input unchanged claim does not bind reopened input")
        inputs.append(input_payload)
        outputs.append(output_payload)
    if status == "success":
        validate_peer_transfers(journal_summaries)
        oracle_outputs = ring_oracle(inputs, dtype, plans)
        require(len(set(oracle_outputs)) == 1,
                "independent planner replay did not converge across ranks")
        oracle = oracle_outputs[0]
        oracle_sha = hashlib.sha256(oracle).hexdigest()
        require(outputs == oracle_outputs,
                "rank outputs are not bitwise equal to the versioned ring-step oracle")
        require(len({hashlib.sha256(output).hexdigest() for output in outputs}) == 1,
                "rank outputs do not have one common SHA")
        require(all(result.get("output_sha256") == oracle_sha
                    and result.get("output_storage_fresh") is True
                    and result.get("public_commit_count") == 1
                    and result.get("public_result_published") is True
                    for result in rank_results),
                "fresh single public commit contract mismatch")
        require(manifest.get("target_execution_completed") is True
                and manifest.get("target_feedback") is False
                and manifest.get("oracle_phase") == "post_target"
                and manifest.get("oracle_feedback") is False
                and manifest.get("public_commit_count") == world,
                "target/oracle/commit boundary mismatch")
        failure = None
    else:
        failure = validate_failure(manifest, rank_results, status, world)
        oracle_sha = None
    cleanup_summary = validate_supervisor_cleanup(
        manifest.get("supervisor_cleanup"), rank_entries, rank_results, world
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        evidence_artifacts: dict[str, dict[str, Any]] = {}
        for relative, (payload, _) in sorted(all_artifacts.items()):
            destination = temporary / relative
            destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            evidence_artifacts[relative] = write_bytes(destination, payload)
        evidence_artifacts["source/result-manifest.json"] = write_bytes(
            temporary / "source/result-manifest.json", canonical_json(manifest))
        evidence_artifacts["expected.json"] = write_bytes(
            temporary / "expected.json", canonical_json(expected))
        verifier_payload, _ = file_record(THIS_FILE)
        accepted_verifier = write_bytes(
            temporary / "acceptance/verifier.py", verifier_payload
        )
        accepted_verifier["path"] = "acceptance/verifier.py"
        result = {
            "schema": ACCEPTANCE_SCHEMA,
            "status": status,
            "world_size": world,
            "formal_live_acceptance_world": world in FORMAL_WORLDS,
            "schema_valid_world": world in ALL_WORLDS,
            "expected_sha256": expected_record["sha256"],
            "source_manifest_sha256": manifest_record["sha256"],
            "source_identity": preflight,
            "execution_preflight_verifier": preflight["verifier"],
            "acceptance_verifier": accepted_verifier,
            "identity_unchanged_postflight": True,
            "authoritative_artifact_rehash": True,
            "planner_independently_recomputed": True,
            "descriptor_independently_recomputed": True,
            "host_reduction_count": 0 if status == "success" else None,
            "host_reduction_evidence": (
                "static DeviceSumExecutor target-path audit plus normal device trace/stats"
                if status == "success" else None
            ),
            "fallback_count": 0 if status == "success" else None,
            "oracle": {
                "schema": ARITHMETIC_POLICY["schema"],
                "phase": "post_target",
                "feedback": False,
                "output_sha256": oracle_sha,
                "arithmetic_separate_from_target": True,
            },
            "rank_journals": journal_summaries,
            "rank_traces": trace_summaries,
            "supervisor_cleanup": cleanup_summary,
            "failure": failure,
            "live_collective_accepted": status == "success" and world in FORMAL_WORLDS,
            "claim_boundary": (
                "Synthetic host fixtures validate this verifier schema only and are never real live evidence."
            ),
        }
        write_bytes(temporary / "result.json", canonical_json(result))
        manifest_artifacts: dict[str, dict[str, Any]] = {}
        for artifact in sorted(path for path in temporary.rglob("*") if path.is_file()):
            payload, record = file_record(artifact)
            manifest_artifacts[str(artifact.relative_to(temporary))] = record
        write_bytes(temporary / "manifest.json", canonical_json({
            "schema": MANIFEST_SCHEMA,
            "artifacts": manifest_artifacts,
            "complete": True,
        }))
        fsync_tree(temporary)
        rename_noreplace(temporary, output)
        temporary = None
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return result
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def write_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"path": str(path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def fsync_tree(root: Path) -> None:
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()),
                            key=lambda path: len(path.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(AT_FDCWD, os.fsencode(source), AT_FDCWD,
                 os.fsencode(destination), RENAME_NOREPLACE) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a completed device-backed live allreduce run")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(Path(os.path.abspath(args.source_dir)),
                    Path(os.path.abspath(args.expected)),
                    Path(os.path.abspath(args.output_dir)))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
