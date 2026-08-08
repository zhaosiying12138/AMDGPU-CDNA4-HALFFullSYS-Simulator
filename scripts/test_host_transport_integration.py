#!/usr/bin/env python3
"""Run the CP-0004 through CP-0008 host-transport integration matrix."""

from __future__ import annotations

import argparse
import array
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import re
import resource
import select
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocol/host-transport-v1.json"
DISPATCH_PROTOCOL_PATH = ROOT / "protocol/host-transport-v1-dispatch.json"
DEFAULT_CONFIG = (
    ROOT / "projects/gem5/configs/example/gemsim/host_bridge.py"
)
DEFAULT_DISPATCH_CONFIG = (
    ROOT / "projects/gem5/configs/example/gemsim/host_dispatch.py"
)
WORLD_SIZES = (1, 2, 3, 4, 8)
RUN_IDENTITY_NAMESPACE = uuid.uuid4()
READY_TOKEN = "host-gpu-ready"
SUCCESS_EXIT_TOKEN = "host GPU transport handshake complete"
DISPATCH_FIXTURE = "gfx950-xor-u8-v1"
DISPATCH_TRACE_SCHEMA = "amdgpu-sim.cp8-dispatch-trace.v1"
# Bit 4 is CP-0008 PINNED_DISPATCH_V1; CP4's unsupported-capability probe must
# use the next reserved bit so the runtime can form a valid open request.
UNSUPPORTED_CAPABILITY_PROBE_BIT = 5
FORBIDDEN_NATIVE_GPU_MARKERS = (
    "libhsa",
    "libamdhip64",
    "/dev/kfd",
    "/dev/dri",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
STAT_INTEGER_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)")
DISPATCH_EXIT_PATTERN = re.compile(
    r"host-gpu-dispatch-exit cause=host GPU dispatch session complete "
    r"code=0 tick=(0|[1-9][0-9]*) stats=([^\r\n]+)"
)
LOWER_HEX_64_PATTERN = re.compile(r"[0-9a-f]{128}")
LOWER_HEX_16_PATTERN = re.compile(r"[0-9a-f]{32}")
MAX_DISPATCH_JSON_BYTES = 1 << 20
MAX_DISPATCH_TRACE_BYTES = 1 << 20
MAX_DISPATCH_TRACE_RECORDS = 64
MAX_GEM5_STATS_BYTES = 64 << 20
TRACE_U64_RULES = frozenset({
    "u64-equals-sim_tick",
    "nonzero-u64",
    "same-u64",
    "u64",
    "u64-all-ones",
    "ticket-u64",
    "same-as-fetch-u64",
    "ack-u64",
    "completion-u64",
    "retire-plus-one-u64",
})
TRACE_U32_RULES = frozenset({"u32", "ack-u32"})


class CheckFailure(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Identity:
    daemon_uuid: str
    job_uuid: str
    epoch: int
    rank: int
    world_size: int


@dataclasses.dataclass(frozen=True)
class ClientResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclasses.dataclass(frozen=True)
class RawAck:
    status: int
    request_id: int
    daemon_uuid: str
    connection_id: int
    epoch: int
    client_nonce: bytes
    server_nonce: bytes
    selected_capabilities: bytes
    maximum_record: int
    topology_job_uuid: str | None
    topology_rank: int | None
    topology_world_size: int | None


@dataclasses.dataclass(frozen=True)
class ProcessAudit:
    maps: str
    open_paths: tuple[str, ...]
    log: str


@dataclasses.dataclass(frozen=True)
class ProcessAuditCounts:
    daemon_samples: int
    runtime_samples: int


@dataclasses.dataclass(frozen=True)
class DispatchEvidence:
    request_id: int
    trace_id: int
    fixture_id: int
    queue_id: int
    queue_generation: int
    queue_sequence: int
    input_allocation_id: int
    input_generation: int
    input_gpu_va: int
    output_allocation_id: int
    output_generation: int
    output_gpu_va: int
    signal_id: int
    signal_generation: int
    packet_crc32c: int
    output_crc32c: int
    admission_tick: int
    start_tick: int
    end_tick: int
    retire_tick: int
    signal_completion_tick: int
    materialized_aql_sha256: str = ""


DISPATCH_STAT_SEMANTICS = frozenset({
    "dispatches_admitted",
    "hsapp_packets_fetched",
    "gpu_command_processor_submissions",
    "gpu_dispatcher_starts",
    "gpu_dispatcher_completions",
    "workgroups_completed",
    "waves_started",
    "packets_retired",
    "dispatches_completed",
    "retired_instructions",
    "global_store_instructions",
    "global_store_bytes",
    "host_fallback_count",
})
DISPATCH_STAT_NAMES = {
    "dispatches_admitted": (
        "system.host_gpu_bridge.cp8_dispatches_admitted"
    ),
    "hsapp_packets_fetched": (
        "system.host_gpu_bridge.cp8_hsapp_packets_fetched"
    ),
    "gpu_command_processor_submissions": (
        "system.host_gpu_bridge.cp8_gpu_command_processor_submissions"
    ),
    "gpu_dispatcher_starts": (
        "system.host_gpu_bridge.cp8_gpu_dispatcher_starts"
    ),
    "gpu_dispatcher_completions": (
        "system.host_gpu_bridge.cp8_gpu_dispatcher_completions"
    ),
    "workgroups_completed": (
        "system.host_gpu_bridge.cp8_cu_workgroups_completed"
    ),
    "waves_started": "system.host_gpu_bridge.cp8_cu_waves_started",
    "packets_retired": "system.host_gpu_bridge.cp8_packets_retired",
    "dispatches_completed": (
        "system.host_gpu_bridge.cp8_dispatches_completed"
    ),
    # The single-CU host_dispatch configuration exposes the CU group without
    # an indexed child name.
    "retired_instructions": "system.cpu1.CUs.numInstrExecuted",
    "global_store_instructions": (
        "system.host_gpu_bridge.cp8_cu_global_store_instructions"
    ),
    "global_store_bytes": (
        "system.host_gpu_bridge.cp8_cu_global_store_bytes"
    ),
    "host_fallback_count": "system.host_gpu_bridge.host_fallback_count",
}
DISPATCH_RESULT_KEYS = frozenset({
    "status",
    "fixture",
    "fixture_id",
    "fixture_manifest_sha256",
    "input_crc32c",
    "output_sentinel_crc32c",
    "input_hex",
    "initial_output_hex",
    "expected_output_hex",
    "d2h_output_hex",
    "ticket",
    "first_wait",
    "completion",
    "signal",
    "output_crc32c",
    "output_match",
    "cleanup",
})
DISPATCH_CLIENT_RESULT_KEYS = frozenset({
    "status",
    "selected_version",
    "capability_words",
    "daemon_uuid",
    "job_uuid",
    "connection_id",
    "epoch",
    "rank",
    "world_size",
    "maximum_record_bytes",
    "request_id",
    "peer_uid",
    "peer_pid",
    "dispatch",
})
DISPATCH_TICKET_KEYS = frozenset({
    "request_id",
    "queue_id",
    "queue_generation",
    "queue_sequence",
    "input_allocation_id",
    "input_generation",
    "output_allocation_id",
    "output_generation",
    "signal_id",
    "signal_generation",
    "trace_id",
    "input_gpu_va",
    "output_gpu_va",
    "packet_crc32c",
    "admission_tick",
})
DISPATCH_COMPLETION_KEYS = frozenset({
    "status",
    "wire_status",
    "request_id",
    "queue_id",
    "queue_generation",
    "queue_sequence",
    "fixture_id",
    "input_allocation_id",
    "input_generation",
    "output_allocation_id",
    "output_generation",
    "signal_id",
    "signal_generation",
    "trace_id",
    "input_gpu_va",
    "output_gpu_va",
    "packet_crc32c",
    "output_crc32c",
    "admission_tick",
    "start_tick",
    "end_tick",
    "retire_tick",
})
DISPATCH_FIRST_WAIT_KEYS = frozenset({
    "status", "status_name", "wire_status", "retried_without_send",
})
DISPATCH_SIGNAL_KEYS = frozenset({
    "armed_wait_status",
    "armed_wait_wire_status",
    "armed_wait_status_name",
    "observed_value",
    "signal_completion_tick",
    "retried_without_send",
})
DISPATCH_CLEANUP_KEYS = frozenset({
    "queue_destroyed", "input_freed", "output_freed", "signal_destroyed",
})


def load_json_document(path: Path, expected_schema: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise CheckFailure(f"could not load {path}: {error}") from error
    require(isinstance(document, dict), f"{path} is not a JSON object")
    require(document.get("schema") == expected_schema,
            f"unexpected protocol schema in {path}")
    return document


def parse_json_integer(value: Any, source: str) -> int:
    require(not isinstance(value, bool), f"{source} is a boolean, not an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        require(value != "" and value.strip() == value,
                f"{source} is not a canonical integer: {value!r}")
        try:
            result = int(value, 0)
        except ValueError as error:
            raise CheckFailure(f"{source} is not an integer: {value!r}") from error
    else:
        raise CheckFailure(f"{source} is not an integer: {value!r}")
    return result


def require_u64(value: Any, source: str, *, nonzero: bool = False) -> int:
    result = parse_json_integer(value, source)
    require(0 <= result <= 0xFFFFFFFFFFFFFFFF,
            f"{source} is outside uint64: {result}")
    if nonzero:
        require(result != 0, f"{source} must be nonzero")
    return result


def require_u32(value: Any, source: str) -> int:
    result = parse_json_integer(value, source)
    require(0 <= result <= 0xFFFFFFFF,
            f"{source} is outside uint32: {result}")
    return result


def require_json_unsigned(value: Any, source: str, bits: int) -> int:
    require(type(value) is int, f"{source} is not a JSON unsigned integer")
    maximum = (1 << bits) - 1
    require(0 <= value <= maximum,
            f"{source} is outside uint{bits}: {value}")
    return value


def validate_trace_event_field(
    value: Any, rule: Any, source: str, sim_tick: int
) -> None:
    if not isinstance(rule, str):
        require(type(value) is type(rule) and value == rule,
                f"{source} differs from frozen value {rule!r}: {value!r}")
        return
    if rule in TRACE_U64_RULES:
        integer = require_json_unsigned(value, source, 64)
        if rule == "nonzero-u64":
            require(integer != 0, f"{source} must be nonzero")
        elif rule == "u64-all-ones":
            require(integer == 0xFFFFFFFFFFFFFFFF,
                    f"{source} is not the all-lanes wave64 mask")
        elif rule == "u64-equals-sim_tick":
            require(integer == sim_tick,
                    f"{source} does not equal the event sim_tick")
        return
    if rule in TRACE_U32_RULES:
        require_json_unsigned(value, source, 32)
        return
    if rule == "exact-64-byte-lowercase-hex":
        require(isinstance(value, str) and
                LOWER_HEX_64_PATTERN.fullmatch(value) is not None,
                f"{source} is not exact lowercase 64-byte hex")
        return
    if rule == "exact-16-byte-lowercase-hex":
        require(isinstance(value, str) and
                LOWER_HEX_16_PATTERN.fullmatch(value) is not None,
                f"{source} is not exact lowercase 16-byte hex")
        return
    if rule == "lowercase-sha256":
        require_sha256(value, source)
        return
    require(type(value) is str and value == rule,
            f"{source} differs from frozen value {rule!r}: {value!r}")


def require_sha256(value: Any, source: str) -> str:
    require(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
            f"{source} is not a lowercase SHA-256 digest: {value!r}")
    return value


def require_exact_object(
    value: Any, expected_keys: Iterable[str], source: str
) -> dict[str, Any]:
    require(isinstance(value, dict), f"{source} is not a JSON object")
    keys = set(value)
    expected = set(expected_keys)
    require(keys == expected,
            f"{source} keys differ from the frozen surface: "
            f"missing={sorted(expected - keys)} extra={sorted(keys - expected)}")
    return value


def require_fixed_hex_unsigned(
    value: Any, source: str, bits: int, *, nonzero: bool = False
) -> int:
    digits = bits // 4
    require(bits in (32, 64), f"unsupported fixed hex width for {source}")
    require(isinstance(value, str) and
            re.fullmatch(rf"0x[0-9a-f]{{{digits}}}", value) is not None,
            f"{source} is not canonical lowercase uint{bits} hex: {value!r}")
    result = int(value, 16)
    if nonzero:
        require(result != 0, f"{source} must be nonzero")
    return result


def require_lower_hex_bytes(value: Any, source: str, byte_count: int) -> bytes:
    require(isinstance(value, str) and
            re.fullmatch(rf"[0-9a-f]{{{byte_count * 2}}}", value) is not None,
            f"{source} is not exact lowercase {byte_count}-byte hex")
    return bytes.fromhex(value)


def validate_dispatch_result(
    value: Any, authority: dict[str, Any]
) -> DispatchEvidence:
    dispatch = require_exact_object(value, DISPATCH_RESULT_KEYS,
                                    "runtime dispatch result")
    fixture = authority["fixture_authority"]
    golden = fixture["golden_buffers"]
    require(type(dispatch["status"]) is int and dispatch["status"] == 0,
            "runtime dispatch status is not canonical success")
    require(dispatch["fixture"] == DISPATCH_FIXTURE,
            "runtime dispatch fixture name differs from the pinned CLI")
    fixture_id = require_fixed_hex_unsigned(
        dispatch["fixture_id"], "runtime dispatch fixture_id", 64, nonzero=True
    )
    require(fixture_id == fixture["fixture_id"],
            "runtime dispatch fixture ID differs from authority")
    require(require_sha256(
        dispatch["fixture_manifest_sha256"],
        "runtime dispatch fixture_manifest_sha256",
    ) == fixture["manifest"]["sha256_hex"],
            "runtime dispatch fixture manifest hash differs from authority")

    ticket = require_exact_object(
        dispatch["ticket"], DISPATCH_TICKET_KEYS, "runtime dispatch ticket"
    )
    ticket_u64 = {
        name: require_fixed_hex_unsigned(
            ticket[name], f"runtime dispatch ticket.{name}", 64, nonzero=True
        )
        for name in DISPATCH_TICKET_KEYS
        if name != "packet_crc32c"
    }
    packet_crc32c = require_fixed_hex_unsigned(
        ticket["packet_crc32c"],
        "runtime dispatch ticket.packet_crc32c",
        32,
    )
    require(packet_crc32c == int(
        authority["golden"]["identity"]["packet_crc32c_hex"], 16
    ), "runtime dispatch ticket packet CRC differs from fixture authority")
    input_allocation_id = ticket_u64["input_allocation_id"]
    output_allocation_id = ticket_u64["output_allocation_id"]
    signal_id = ticket_u64["signal_id"]
    require(1 <= input_allocation_id <= 1024 and
            1 <= output_allocation_id <= 1024 and
            input_allocation_id != output_allocation_id,
            "runtime dispatch allocation handles are aliased or out of range")
    require(1 <= signal_id <= 1024,
            "runtime dispatch signal handle is out of range")
    slot_base = 0x0000100000000000
    slot_stride = 0x80000000
    require(
        ticket_u64["input_gpu_va"] ==
        slot_base + (input_allocation_id - 1) * slot_stride and
        ticket_u64["output_gpu_va"] ==
        slot_base + (output_allocation_id - 1) * slot_stride,
        "runtime dispatch packet VAs differ from their CP-0006 slots",
    )

    first_wait = require_exact_object(
        dispatch["first_wait"], DISPATCH_FIRST_WAIT_KEYS,
        "runtime dispatch first_wait",
    )
    require(type(first_wait["status"]) is int and
            first_wait["status"] == 19 and
            first_wait["status_name"] == "cancelled" and
            type(first_wait["wire_status"]) is int and
            first_wait["wire_status"] == -1 and
            first_wait["retried_without_send"] is True,
            "runtime dispatch first wait was not locally cancelled and retried")

    completion = require_exact_object(
        dispatch["completion"], DISPATCH_COMPLETION_KEYS,
        "runtime dispatch completion",
    )
    require(type(completion["status"]) is int and completion["status"] == 0 and
            type(completion["wire_status"]) is int and
            completion["wire_status"] == 0,
            "runtime dispatch completion status is not canonical success")
    completion_u64 = {
        name: require_fixed_hex_unsigned(
            completion[name], f"runtime dispatch completion.{name}", 64,
            nonzero=True,
        )
        for name in DISPATCH_COMPLETION_KEYS
        if name not in {"status", "wire_status", "packet_crc32c",
                        "output_crc32c"}
    }
    completion_packet_crc = require_fixed_hex_unsigned(
        completion["packet_crc32c"],
        "runtime dispatch completion.packet_crc32c",
        32,
    )
    completion_output_crc = require_fixed_hex_unsigned(
        completion["output_crc32c"],
        "runtime dispatch completion.output_crc32c",
        32,
    )
    ticket_echoes = (
        "request_id",
        "queue_id",
        "queue_generation",
        "queue_sequence",
        "input_allocation_id",
        "input_generation",
        "output_allocation_id",
        "output_generation",
        "signal_id",
        "signal_generation",
        "trace_id",
        "input_gpu_va",
        "output_gpu_va",
        "admission_tick",
    )
    for name in ticket_echoes:
        require(completion_u64[name] == ticket_u64[name],
                f"runtime dispatch completion.{name} differs from ticket")
    require(completion_u64["fixture_id"] == fixture_id,
            "runtime dispatch completion fixture ID differs from ticket")
    require(completion_packet_crc == packet_crc32c,
            "runtime dispatch completion packet CRC differs from ticket")

    admission_tick = ticket_u64["admission_tick"]
    start_tick = completion_u64["start_tick"]
    end_tick = completion_u64["end_tick"]
    retire_tick = completion_u64["retire_tick"]
    require(admission_tick < start_tick <= end_tick <= retire_tick and
            retire_tick > admission_tick + 1,
            "runtime dispatch completion ticks are not canonical")
    require(retire_tick != 0xFFFFFFFFFFFFFFFF,
            "runtime dispatch retirement cannot form R+1")

    signal_result = require_exact_object(
        dispatch["signal"], DISPATCH_SIGNAL_KEYS, "runtime dispatch signal"
    )
    signal_completion_tick = require_fixed_hex_unsigned(
        signal_result["signal_completion_tick"],
        "runtime dispatch signal.signal_completion_tick",
        64,
        nonzero=True,
    )
    require(type(signal_result["armed_wait_status"]) is int and
            signal_result["armed_wait_status"] == 11 and
            type(signal_result["armed_wait_wire_status"]) is int and
            signal_result["armed_wait_wire_status"] == -1 and
            signal_result["armed_wait_status_name"] == "timed out" and
            type(signal_result["observed_value"]) is int and
            signal_result["observed_value"] == 0 and
            signal_result["retried_without_send"] is True,
            "runtime dispatch signal was not armed-one/EQ0 timeout then zero")
    require(signal_completion_tick == retire_tick + 1,
            "runtime dispatch signal completion tick is not exactly R+1")

    expected_output_crc = int(golden["output_crc32c_hex"], 16)
    require(completion_output_crc == expected_output_crc,
            "runtime dispatch completion output CRC differs from authority")

    # Execution and signal completion are canonical before accepting D2H bytes.
    input_bytes = require_lower_hex_bytes(
        dispatch["input_hex"], "runtime dispatch input_hex", 64
    )
    initial_output = require_lower_hex_bytes(
        dispatch["initial_output_hex"],
        "runtime dispatch initial_output_hex",
        64,
    )
    expected_output = require_lower_hex_bytes(
        dispatch["expected_output_hex"],
        "runtime dispatch expected_output_hex",
        64,
    )
    d2h_output = require_lower_hex_bytes(
        dispatch["d2h_output_hex"], "runtime dispatch d2h_output_hex", 64
    )
    require(input_bytes.hex() == golden["input_hex"] and
            input_bytes == bytes(range(64)),
            "runtime dispatch input bytes differ from exact 00..3f")
    require(initial_output == bytes(64),
            "runtime dispatch initial output is not the exact zero sentinel")
    xor_output = bytes(byte ^ fixture["xor_byte"] for byte in input_bytes)
    require(expected_output == xor_output and
            expected_output.hex() == golden["output_hex"] and
            d2h_output == expected_output and
            d2h_output != input_bytes and d2h_output != initial_output,
            "runtime dispatch D2H bytes differ from the XOR fixture oracle")
    input_crc = require_fixed_hex_unsigned(
        dispatch["input_crc32c"], "runtime dispatch input_crc32c", 32
    )
    sentinel_crc = require_fixed_hex_unsigned(
        dispatch["output_sentinel_crc32c"],
        "runtime dispatch output_sentinel_crc32c",
        32,
    )
    output_crc = require_fixed_hex_unsigned(
        dispatch["output_crc32c"], "runtime dispatch output_crc32c", 32
    )
    require(input_crc == WireProtocol.crc32c(input_bytes) ==
            int(golden["input_crc32c_hex"], 16),
            "runtime dispatch input CRC differs from exact input bytes")
    require(sentinel_crc == WireProtocol.crc32c(initial_output) ==
            int(golden["output_initial_crc32c_hex"], 16),
            "runtime dispatch sentinel CRC differs from zero bytes")
    require(output_crc == WireProtocol.crc32c(d2h_output) ==
            expected_output_crc == completion_output_crc,
            "runtime dispatch D2H CRC differs from bytes or completion")
    require(hashlib.sha256(input_bytes).hexdigest() ==
            golden["input_sha256_hex"] and
            hashlib.sha256(initial_output).hexdigest() ==
            golden["output_initial_sha256_hex"] and
            hashlib.sha256(d2h_output).hexdigest() ==
            golden["output_sha256_hex"],
            "runtime dispatch buffer hashes differ from authority")
    require(dispatch["output_match"] is True,
            "runtime dispatch did not publish exact output_match")
    cleanup = require_exact_object(
        dispatch["cleanup"], DISPATCH_CLEANUP_KEYS, "runtime dispatch cleanup"
    )
    require(all(cleanup[name] is True for name in DISPATCH_CLEANUP_KEYS),
            "runtime dispatch resource cleanup is incomplete")

    return DispatchEvidence(
        request_id=ticket_u64["request_id"],
        trace_id=ticket_u64["trace_id"],
        fixture_id=fixture_id,
        queue_id=ticket_u64["queue_id"],
        queue_generation=ticket_u64["queue_generation"],
        queue_sequence=ticket_u64["queue_sequence"],
        input_allocation_id=input_allocation_id,
        input_generation=ticket_u64["input_generation"],
        input_gpu_va=ticket_u64["input_gpu_va"],
        output_allocation_id=output_allocation_id,
        output_generation=ticket_u64["output_generation"],
        output_gpu_va=ticket_u64["output_gpu_va"],
        signal_id=signal_id,
        signal_generation=ticket_u64["signal_generation"],
        packet_crc32c=packet_crc32c,
        output_crc32c=output_crc,
        admission_tick=admission_tick,
        start_tick=start_tick,
        end_tick=end_tick,
        retire_tick=retire_tick,
        signal_completion_tick=signal_completion_tick,
    )


def strict_json_object(text: str, source: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CheckFailure(f"{source} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise CheckFailure(f"{source} contains non-finite JSON number {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise CheckFailure(f"{source} is invalid JSON: {error}") from error
    require(isinstance(value, dict), f"{source} is not a JSON object")
    return value


def load_dispatch_trace(path: Path) -> list[dict[str, Any]]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise CheckFailure(f"could not stat dispatch trace {path}: {error}") from error
    require(size <= MAX_DISPATCH_TRACE_BYTES,
            f"dispatch trace {path} exceeds {MAX_DISPATCH_TRACE_BYTES} bytes")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CheckFailure(f"could not read dispatch trace {path}: {error}") from error
    require(text.endswith("\n"),
            f"dispatch trace {path} does not end at a JSONL record boundary")
    lines = text.splitlines()
    require(lines, f"dispatch trace {path} is empty")
    require(len(lines) <= MAX_DISPATCH_TRACE_RECORDS,
            f"dispatch trace {path} exceeds {MAX_DISPATCH_TRACE_RECORDS} records")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        require(line != "", f"dispatch trace {path}:{index} is blank")
        records.append(strict_json_object(line, f"dispatch trace {path}:{index}"))
    return records


def parse_gem5_stats(path: Path) -> dict[str, str]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise CheckFailure(f"could not stat gem5 stats {path}: {error}") from error
    require(size <= MAX_GEM5_STATS_BYTES,
            f"gem5 stats {path} exceeds {MAX_GEM5_STATS_BYTES} bytes")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CheckFailure(f"could not read gem5 stats {path}: {error}") from error
    stats: dict[str, str] = {}
    in_section = False
    saw_section = False
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "---------- Begin Simulation Statistics ----------":
            require(not in_section and not saw_section,
                    f"gem5 stats {path} contains multiple statistics sections")
            in_section = True
            saw_section = True
            continue
        if stripped == "---------- End Simulation Statistics   ----------":
            require(in_section, f"gem5 stats {path}:{line_number} has an unmatched end")
            in_section = False
            continue
        if stripped.startswith("#") or not in_section:
            continue
        columns = stripped.split()
        require(len(columns) >= 2,
                f"gem5 stats {path}:{line_number} is malformed: {line!r}")
        name, value = columns[:2]
        require(name not in stats,
                f"gem5 stats {path} repeats statistic {name}")
        stats[name] = value
    require(saw_section and not in_section,
            f"gem5 stats {path} has no complete statistics section")
    return stats


def require_stat_integer(stats: dict[str, str], name: str) -> int:
    require(name in stats, f"gem5 stats is missing {name}")
    value = stats[name]
    require(STAT_INTEGER_PATTERN.fullmatch(value) is not None,
            f"gem5 statistic {name} is not an unsigned integer: {value!r}")
    return int(value, 10)


def capture_process_audit(
    pid: int, log_path: Path | None = None
) -> ProcessAudit:
    try:
        maps = Path(f"/proc/{pid}/maps").read_text(
            encoding="utf-8", errors="replace"
        )
        descriptor_entries = list(Path(f"/proc/{pid}/fd").iterdir())
    except (FileNotFoundError, ProcessLookupError) as error:
        raise CheckFailure(f"process {pid} exited during provenance audit") from error
    except OSError as error:
        raise CheckFailure(f"could not audit process {pid}: {error}") from error
    open_paths_list: list[str] = []
    for entry in descriptor_entries:
        try:
            open_paths_list.append(os.readlink(entry))
        except FileNotFoundError:
            continue
        except OSError as error:
            raise CheckFailure(
                f"could not audit process {pid} descriptor {entry.name}: {error}"
            ) from error
    return ProcessAudit(
        maps=maps,
        open_paths=tuple(sorted(open_paths_list)),
        log=read_text(log_path) if log_path is not None else "",
    )


def validate_process_audit(audit: ProcessAudit) -> None:
    sources = {
        "process maps": audit.maps,
        "open file descriptors": "\n".join(audit.open_paths),
        "gem5 log": audit.log,
    }
    for source, contents in sources.items():
        lowered = contents.lower()
        for marker in FORBIDDEN_NATIVE_GPU_MARKERS:
            require(marker not in lowered,
                    f"{source} contains forbidden native GPU dependency {marker}")


def validate_evidence_file(
    path: Path, source: str, *, exact_mode: int | None = None
) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise CheckFailure(f"could not stat {source} {path}: {error}") from error
    require(stat.S_ISREG(metadata.st_mode),
            f"{source} is not a regular file: {path}")
    require(metadata.st_uid == os.geteuid(),
            f"{source} is not owned by the current euid: {path}")
    mode = stat.S_IMODE(metadata.st_mode)
    if exact_mode is not None:
        require(mode == exact_mode,
                f"{source} mode is {mode:04o}, expected {exact_mode:04o}")
    else:
        require(mode & 0o022 == 0,
                f"{source} is group/world writable: {path}")
    require(metadata.st_size > 0, f"{source} is empty: {path}")
    return metadata


def validate_dispatch_exit_log(
    log: str, stats_path: Path, earliest_tick: int
) -> int:
    marker_lines = [
        line for line in log.splitlines()
        if line.startswith("host-gpu-dispatch-exit ")
    ]
    require(len(marker_lines) == 1,
            "dispatch daemon log does not contain exactly one exit marker")
    match = DISPATCH_EXIT_PATTERN.fullmatch(marker_lines[0])
    require(match is not None,
            "dispatch daemon did not exit for canonical session completion")
    assert match is not None
    exit_tick = int(match.group(1), 10)
    require(exit_tick >= earliest_tick,
            "dispatch daemon exit tick precedes signal completion")
    require(match.group(2) == str(stats_path),
            "dispatch daemon exit marker names a foreign stats path")
    return exit_tick


def validate_dispatch_stats(
    stats: dict[str, str], metric_names: dict[str, str]
) -> dict[str, int]:
    require(set(metric_names) == DISPATCH_STAT_SEMANTICS,
            "dispatch statistic map does not cover the exact CP-0008 semantics")
    require(len(set(metric_names.values())) == len(metric_names),
            "dispatch statistic map aliases two semantic counters")
    values = {
        semantic: require_stat_integer(stats, name)
        for semantic, name in metric_names.items()
    }
    exact_one = (
        "dispatches_admitted",
        "hsapp_packets_fetched",
        "gpu_command_processor_submissions",
        "gpu_dispatcher_starts",
        "gpu_dispatcher_completions",
        "workgroups_completed",
        "waves_started",
        "packets_retired",
        "dispatches_completed",
    )
    for semantic in exact_one:
        require(values[semantic] == 1,
                f"dispatch statistic {semantic} is not exactly one: "
                f"{values[semantic]}")
    require(values["retired_instructions"] > 0,
            "dispatch retired no GPU instructions")
    require(values["global_store_instructions"] > 0,
            "dispatch executed no GPU global-store instructions")
    require(values["global_store_bytes"] == 64,
            f"dispatch stored {values['global_store_bytes']} bytes, expected 64")
    require(values["host_fallback_count"] == 0,
            "dispatch used a host fallback path")
    return values


def validate_dispatch_trace(
    records: list[dict[str, Any]],
    evidence: DispatchEvidence,
    authority: dict[str, Any],
) -> dict[str, Any]:
    fixture = authority["fixture_authority"]
    trace_contract = authority["trace_contract"]
    expected_events = trace_contract["required_ordered_events"]
    require(len(records) == len(expected_events),
            f"dispatch trace has {len(records)} events, expected "
            f"{len(expected_events)}")
    require([record.get("event") for record in records] == expected_events,
            "dispatch trace event order is missing, duplicated, or foreign")

    expected_common: dict[str, Any] = {
        "schema": trace_contract["schema"],
        "trace_id": evidence.trace_id,
        "request_id": evidence.request_id,
        "fixture_id": evidence.fixture_id,
        "fixture_manifest_sha256": fixture["manifest"]["sha256_hex"],
        "code_image_sha256": fixture["code_image"]["sha256_hex"],
        "aql_template_sha256": fixture["aql_template"]["sha256_hex"],
        "materialized_aql_sha256": evidence.materialized_aql_sha256,
        "queue_id": evidence.queue_id,
        "queue_generation": evidence.queue_generation,
        "queue_sequence": evidence.queue_sequence,
        "input_allocation_id": evidence.input_allocation_id,
        "input_generation": evidence.input_generation,
        "output_allocation_id": evidence.output_allocation_id,
        "output_generation": evidence.output_generation,
        "signal_id": evidence.signal_id,
        "signal_generation": evidence.signal_generation,
    }
    common_fields = set(trace_contract["common_fields"])
    require(common_fields == set(expected_common) | {"event", "sim_tick"},
            "dispatch trace authority common_fields differ from the validator")
    require(set(trace_contract["event_fields"]) == set(expected_events),
            "dispatch trace authority event_fields differ from ordered events")
    require(trace_contract["schema"] == DISPATCH_TRACE_SCHEMA,
            "dispatch trace authority schema differs from CP-0008")
    previous_tick = -1
    indexed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        event = expected_events[index]
        indexed[event] = record
        event_fields = trace_contract["event_fields"].get(event)
        require(isinstance(event_fields, dict),
                f"dispatch authority lacks event_fields for {event}")
        expected_keys = common_fields | set(event_fields)
        require(set(record) == expected_keys,
                f"dispatch trace event {event} keys differ from authority: "
                f"missing={sorted(expected_keys - set(record))} "
                f"extra={sorted(set(record) - expected_keys)}")
        for name, expected in expected_common.items():
            actual = record.get(name)
            if isinstance(expected, int):
                actual = require_u64(actual, f"trace {event}.{name}")
            elif name.endswith("sha256"):
                actual = require_sha256(actual, f"trace {event}.{name}")
            require(actual == expected,
                    f"dispatch trace {event}.{name} is inconsistent: "
                    f"{actual!r} != {expected!r}")
        sim_tick = require_u64(record.get("sim_tick"), f"trace {event}.sim_tick")
        require(sim_tick >= previous_tick,
                f"dispatch trace tick regressed at {event}")
        previous_tick = sim_tick
        for name, rule in event_fields.items():
            validate_trace_event_field(
                record[name], rule, f"trace {event}.{name}", sim_tick
            )

    require(
        require_u64(indexed["dispatch_admitted"].get("sim_tick"),
                    "trace dispatch_admitted.sim_tick") ==
        require_u64(indexed["dispatch_admitted"].get("admission_tick"),
                    "trace dispatch_admitted.admission_tick"),
        "dispatch admission event does not self-correlate its tick",
    )
    require(
        require_u64(indexed["dispatch_admitted"].get("admission_tick"),
                    "trace dispatch_admitted.admission_tick") ==
        evidence.admission_tick,
        "dispatch trace admission tick differs from the ACK ticket",
    )
    require(
        require_u64(indexed["gpu_dispatcher_started"].get("sim_tick"),
                    "trace gpu_dispatcher_started.sim_tick") ==
        evidence.start_tick,
        "dispatch trace GPU start tick differs from completion",
    )
    require(
        require_u64(indexed["cu_global_store_completed"].get("sim_tick"),
                    "trace cu_global_store_completed.sim_tick") ==
        evidence.end_tick,
        "dispatch trace final-store tick differs from completion",
    )
    require(
        require_u64(indexed["packet_retired"].get("sim_tick"),
                    "trace packet_retired.sim_tick") ==
        evidence.retire_tick,
        "dispatch trace retire tick differs from completion",
    )

    published = indexed["aql_packet_published"]
    packet_hex = published.get("materialized_aql_hex")
    require(isinstance(packet_hex, str),
            "dispatch trace does not retain materialized_aql_hex")
    try:
        packet = bytes.fromhex(packet_hex)
    except ValueError as error:
        raise CheckFailure("dispatch trace materialized AQL is not hex") from error
    require(len(packet) == fixture["aql_template"]["bytes"],
            f"materialized AQL is {len(packet)} bytes, expected 64")
    template = bytes.fromhex(fixture["aql_template"]["bytes_hex"])
    require(packet[:32] == template[:32] and packet[48:] == template[48:],
            "materialized AQL changed a field outside its two permitted VAs")
    packet_fields = struct.unpack("<6H5I4Q", packet)
    kernel_object, kernarg_address, reserved, completion_signal = packet_fields[11:]
    require(kernel_object != 0 and kernarg_address != 0,
            "materialized AQL contains a zero code or kernarg VA")
    require(reserved == 0 and completion_signal == 0,
            "materialized AQL reserved/completion-signal field is nonzero")
    actual_materialized_hash = hashlib.sha256(packet).hexdigest()
    require(actual_materialized_hash == evidence.materialized_aql_sha256,
            "materialized AQL bytes and retained hash disagree")
    require(WireProtocol.crc32c(packet) == evidence.packet_crc32c,
            "materialized AQL bytes and ACK packet CRC disagree")
    require(require_u64(published.get("kernel_object"),
                        "trace aql_packet_published.kernel_object") ==
            kernel_object,
            "retained materialized AQL kernel-object VA disagrees")
    require(require_u64(published.get("kernarg_address"),
                        "trace aql_packet_published.kernarg_address") ==
            kernarg_address,
            "retained materialized AQL kernarg VA disagrees")
    require(require_u64(published.get("completion_signal"),
                        "trace aql_packet_published.completion_signal") == 0,
            "retained AQL completion signal is not the zero special handle")
    require(require_u32(published.get("packet_crc32c"),
                        "trace aql_packet_published.packet_crc32c") ==
            evidence.packet_crc32c,
            "retained materialized AQL packet CRC differs from ACK")
    require(require_u32(published.get("header"),
                        "trace aql_packet_published.header") == 0x1402 and
            require_u32(published.get("setup"),
                        "trace aql_packet_published.setup") == 1,
            "retained AQL header/setup differs from the pinned packet")
    require(published.get("grid_size") == fixture["grid_size"] and
            published.get("workgroup_size") == fixture["workgroup_size"],
            "retained AQL geometry differs from the pinned packet")

    kernarg_hex = published.get("materialized_kernarg_hex")
    require(isinstance(kernarg_hex, str),
            "dispatch trace does not retain materialized_kernarg_hex")
    try:
        kernarg = bytes.fromhex(kernarg_hex)
    except ValueError as error:
        raise CheckFailure("dispatch trace materialized kernarg is not hex") from error
    require(len(kernarg) == fixture["kernarg_template"]["bytes"],
            "materialized kernarg does not have the pinned size")
    require(int.from_bytes(kernarg[:8], "little") == evidence.input_gpu_va and
            int.from_bytes(kernarg[8:], "little") == evidence.output_gpu_va,
            "materialized kernarg does not bind the two ticket VAs")
    require(require_sha256(
        published.get("materialized_kernarg_sha256"),
        "trace aql_packet_published.materialized_kernarg_sha256",
    ) == hashlib.sha256(kernarg).hexdigest(),
            "materialized kernarg bytes and retained hash disagree")

    registered = indexed["aql_queue_registered"]
    require(registered.get("component") == "HSAPacketProcessor" and
            registered.get("active") is True,
            "AQL queue registration is not attributed to active "
            "HSAPacketProcessor state")
    internal_queue_id = require_u64(
        registered.get("internal_queue_id"),
        "trace aql_queue_registered.internal_queue_id",
        nonzero=True,
    )
    internal_queue_generation = require_u64(
        registered.get("internal_queue_generation"),
        "trace aql_queue_registered.internal_queue_generation",
        nonzero=True,
    )
    packet_va = require_u64(published.get("packet_va"),
                            "trace aql_packet_published.packet_va", nonzero=True)
    for event in ("aql_packet_published", "hsapp_packet_fetched",
                  "packet_retired"):
        record = indexed[event]
        require(require_u64(record.get("internal_queue_id"),
                            f"trace {event}.internal_queue_id") ==
                internal_queue_id,
                f"dispatch trace {event} crossed internal AQL queues")
        require(require_u64(record.get("internal_queue_generation"),
                            f"trace {event}.internal_queue_generation") ==
                internal_queue_generation,
                f"dispatch trace {event} crossed internal queue generations")
        require(require_u64(record.get("packet_va"),
                            f"trace {event}.packet_va") == packet_va,
                f"dispatch trace {event} crossed AQL packet VAs")

    submitted = indexed["gpu_command_processor_submitted"]
    require(indexed["hsapp_packet_fetched"].get("component") ==
            "HSAPacketProcessor",
            "packet fetch is not attributed to HSAPacketProcessor")
    require(submitted.get("component") == "GPUCommandProcessor",
            "GPU task submission is not attributed to GPUCommandProcessor")
    gpu_task_id = require_u64(submitted.get("gpu_task_id"),
                              "trace gpu_command_processor_submitted.gpu_task_id",
                              nonzero=True)
    for event in ("gpu_dispatcher_started", "cu_wave_started",
                  "cu_global_store_completed", "gpu_dispatcher_completed",
                  "packet_retired"):
        require(require_u64(indexed[event].get("gpu_task_id"),
                            f"trace {event}.gpu_task_id") == gpu_task_id,
                f"dispatch trace {event} crossed GPU task IDs")

    dispatcher = indexed["gpu_dispatcher_started"]
    require(dispatcher.get("component") == "GPUDispatcher" and
            indexed["gpu_dispatcher_completed"].get("component") ==
            "GPUDispatcher",
            "GPU dispatch start/completion is not attributed to GPUDispatcher")
    require(dispatcher.get("grid_size") == fixture["grid_size"] and
            dispatcher.get("workgroup_size") == fixture["workgroup_size"],
            "GPUDispatcher trace geometry differs from the pinned fixture")
    require(require_u64(dispatcher.get("workgroups"),
                        "trace gpu_dispatcher_started.workgroups") == 1 and
            require_u64(dispatcher.get("waves"),
                        "trace gpu_dispatcher_started.waves") == 1,
            "GPUDispatcher trace did not schedule one workgroup and one wave")

    wave = indexed["cu_wave_started"]
    require(wave.get("component") == "ComputeUnit" and
            indexed["cu_global_store_completed"].get("component") ==
            "ComputeUnit",
            "wave/store execution is not attributed to ComputeUnit")
    require(require_u64(wave.get("cu_id"), "trace cu_wave_started.cu_id") == 0,
            "dispatch trace used a compute unit other than CU 0")
    require(wave.get("workgroup_id") == [0, 0, 0],
            "dispatch trace workgroup coordinates are not zero")
    require(require_u64(wave.get("wavefront_size"),
                        "trace cu_wave_started.wavefront_size") == 64,
            "dispatch trace wavefront is not wave64")
    require(require_u64(wave.get("lane_mask"),
                        "trace cu_wave_started.lane_mask") ==
            0xFFFFFFFFFFFFFFFF,
            "dispatch trace lane mask is not all 64 lanes")

    store = indexed["cu_global_store_completed"]
    require(require_u64(store.get("output_gpu_va"),
                        "trace cu_global_store_completed.output_gpu_va") ==
            evidence.output_gpu_va,
            "dispatch trace global-store base differs from the output ticket VA")
    require(require_u64(store.get("store_bytes"),
                        "trace cu_global_store_completed.store_bytes") == 64,
            "dispatch trace global-store range is not exactly 64 bytes")

    fetched = indexed["hsapp_packet_fetched"]
    retired = indexed["packet_retired"]
    read_index = require_u64(fetched.get("read_index"),
                             "trace hsapp_packet_fetched.read_index")
    require(require_u64(retired.get("finish_pkt_read_index"),
                        "trace packet_retired.finish_pkt_read_index") ==
            read_index,
            "finishPkt retirement crossed the fetched AQL read index")
    require(require_u64(retired.get("completion_signal"),
                        "trace packet_retired.completion_signal") == 0,
            "retired packet did not retain the zero completion signal")

    mirrored = indexed["cp7_signal_mirrored"]
    require(require_u64(mirrored.get("sim_tick"),
                        "trace cp7_signal_mirrored.sim_tick") ==
            evidence.retire_tick,
            "CP-0007 signal mirror did not occur at packet retire tick R")
    require(parse_json_integer(mirrored.get("value_before"),
                               "trace cp7_signal_mirrored.value_before") == 1 and
            parse_json_integer(mirrored.get("value_after"),
                               "trace cp7_signal_mirrored.value_after") == 0,
            "CP-0007 trace mirror is not the signed one-to-zero transition")

    final = indexed["wire_completion_emitted"]
    require(evidence.retire_tick != 0xFFFFFFFFFFFFFFFF and
            evidence.signal_completion_tick == evidence.retire_tick + 1,
            "CP-0007 signal completion tick is not exactly R+1")
    require(require_u64(final.get("sim_tick"),
                        "trace wire_completion_emitted.sim_tick") >=
            evidence.signal_completion_tick,
            "dispatch completion was emitted before CP-0007 completion R+1")
    final_summary = {
        "packet_crc32c": evidence.packet_crc32c,
        "output_crc32c": evidence.output_crc32c,
        "admission_tick": evidence.admission_tick,
        "start_tick": evidence.start_tick,
        "end_tick": evidence.end_tick,
        "retire_tick": evidence.retire_tick,
        "signal_completion_tick": evidence.signal_completion_tick,
        "input_gpu_va": evidence.input_gpu_va,
        "output_gpu_va": evidence.output_gpu_va,
    }
    for name, expected in final_summary.items():
        maximum = require_u32 if name.endswith("crc32c") else require_u64
        require(maximum(final.get(name), f"trace wire_completion_emitted.{name}") ==
                expected,
                f"wire completion trace summary disagrees on {name}")
    require(evidence.admission_tick < evidence.start_tick <= evidence.end_tick <=
            evidence.retire_tick and
            evidence.retire_tick > evidence.admission_tick + 1,
            "dispatch completion ticks are not canonical")
    return {
        "events": len(records),
        "request_id": evidence.request_id,
        "trace_id": evidence.trace_id,
        "gpu_task_id": gpu_task_id,
        "internal_queue_id": internal_queue_id,
        "packet_va": packet_va,
        "materialized_aql_sha256": actual_materialized_hash,
    }


class WireProtocol:
    """Independent JSON-driven host-transport v1 wire oracle."""

    def __init__(self, document: dict[str, Any]) -> None:
        self.document = document
        self.header_bytes = int(document["header"]["bytes"])
        self.hello_bytes = int(document["messages"]["HELLO"]["fixed_payload_bytes"])
        self.ack_bytes = int(document["messages"]["HELLO_ACK"]["fixed_payload_bytes"])
        self.maximum_record = int(document["transport"]["max_record_bytes"])
        self.magic = bytes.fromhex(self.header_field("magic")["constant_hex"])
        self.hello_type = int(document["messages"]["HELLO"]["value"])
        self.ack_type = int(document["messages"]["HELLO_ACK"]["value"])
        self.topology_type = int(
            document["tlv"]["types"]["TOPOLOGY_IDENTITY"]["value"]
        )
        self.tlv_header_bytes = int(document["tlv"]["header_bytes"])
        self.tlv_alignment = int(document["tlv"]["alignment_bytes"])
        self.critical_flag = int(document["tlv"]["critical_flag"])
        self.capability_bytes = int(document["capabilities"]["bitmap_bytes"])
        self.topology_bit = int(document["capabilities"]["TOPOLOGY_IDENTITY_V1"])
        self.statuses = {
            name: int(value) for name, value in document["statuses"].items()
        }

    @classmethod
    def load(cls, path: Path = PROTOCOL_PATH) -> "WireProtocol":
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        require(
            document.get("schema") == "amdgpu-sim.host-transport.v1",
            f"unexpected protocol schema in {path}",
        )
        return cls(document)

    def header_field(self, name: str) -> dict[str, Any]:
        for field in self.document["header"]["fields"]:
            if field["name"] == name:
                return field
        raise CheckFailure(f"protocol JSON has no header field {name}")

    def message_field(self, message: str, name: str) -> dict[str, Any]:
        for field in self.document["messages"][message]["fields"]:
            if field["name"] == name:
                return field
        raise CheckFailure(f"protocol JSON has no {message} field {name}")

    @staticmethod
    def _write_integer(target: bytearray, field: dict[str, Any], value: int) -> None:
        size = int(field["bytes"])
        target[int(field["offset"]):int(field["offset"]) + size] = value.to_bytes(
            size, "big"
        )

    @staticmethod
    def _read_integer(source: bytes, field: dict[str, Any], base: int = 0) -> int:
        offset = base + int(field["offset"])
        size = int(field["bytes"])
        return int.from_bytes(source[offset:offset + size], "big")

    @staticmethod
    def _write_bytes(target: bytearray, field: dict[str, Any], value: bytes) -> None:
        size = int(field["bytes"])
        require(len(value) == size, f"field {field['name']} requires {size} bytes")
        offset = int(field["offset"])
        target[offset:offset + size] = value

    @staticmethod
    def crc32c(data: bytes) -> int:
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
        return crc ^ 0xFFFFFFFF

    def _capabilities(self, *bits: int) -> bytes:
        value = bytearray(self.capability_bytes)
        for bit in bits:
            require(0 <= bit < self.capability_bytes * 8,
                    f"capability bit is out of range: {bit}")
            value[bit // 8] |= 1 << (bit % 8)
        return bytes(value)

    def _topology_tlv(self, identity: Identity) -> bytes:
        definition = self.document["tlv"]["types"]["TOPOLOGY_IDENTITY"]
        value = bytearray(int(definition["value_bytes"]))
        fields = {field["name"]: field for field in definition["fields"]}
        self._write_bytes(value, fields["job_uuid"], bytes.fromhex(identity.job_uuid))
        self._write_integer(value, fields["rank"], identity.rank)
        self._write_integer(value, fields["world_size"], identity.world_size)
        unpadded = self.tlv_header_bytes + len(value)
        padded = (
            (unpadded + self.tlv_alignment - 1) // self.tlv_alignment
        ) * self.tlv_alignment
        return (
            struct.pack(">HHI", self.topology_type, self.critical_flag, len(value))
            + bytes(value)
            + bytes(padded - unpadded)
        )

    def _frame(
        self,
        *,
        message_type: int,
        request_id: int,
        daemon_uuid: bytes,
        connection_id: int,
        epoch: int,
        payload: bytes,
    ) -> bytes:
        require(request_id != 0, "raw request ID must be nonzero")
        header = bytearray(self.header_bytes)
        self._write_bytes(header, self.header_field("magic"), self.magic)
        for name in ("framing_major", "framing_minor", "header_bytes", "flags"):
            field = self.header_field(name)
            self._write_integer(header, field, int(field["constant"]))
        self._write_integer(header, self.header_field("message_type"), message_type)
        self._write_integer(header, self.header_field("payload_bytes"), len(payload))
        self._write_integer(header, self.header_field("request_id"), request_id)
        self._write_bytes(header, self.header_field("daemon_instance_uuid"), daemon_uuid)
        self._write_integer(header, self.header_field("connection_id"), connection_id)
        self._write_integer(header, self.header_field("job_epoch"), epoch)
        frame = header + payload
        checksum_field = self.header_field("crc32c")
        self._write_integer(frame, checksum_field, self.crc32c(bytes(frame)))
        return bytes(frame)

    def encode_hello(
        self,
        identity: Identity,
        *,
        request_id: int,
        client_nonce: bytes,
        minimum_version: tuple[int, int] = (1, 0),
        maximum_version: tuple[int, int] = (1, 0),
        role: int | None = None,
        receive_maximum: int | None = None,
    ) -> bytes:
        payload = bytearray(self.hello_bytes)
        fields = {
            field["name"]: field
            for field in self.document["messages"]["HELLO"]["fields"]
        }
        self._write_integer(payload, fields["minimum_major"], minimum_version[0])
        self._write_integer(payload, fields["minimum_minor"], minimum_version[1])
        self._write_integer(payload, fields["maximum_major"], maximum_version[0])
        self._write_integer(payload, fields["maximum_minor"], maximum_version[1])
        self._write_bytes(payload, fields["client_nonce"], client_nonce)
        selected = self._capabilities(self.topology_bit)
        self._write_bytes(payload, fields["offered_capabilities"], selected)
        self._write_bytes(payload, fields["required_capabilities"], selected)
        self._write_integer(
            payload, fields["rx_max_record"],
            self.maximum_record if receive_maximum is None else receive_maximum,
        )
        self._write_integer(
            payload, fields["role"],
            int(fields["role"]["constant"]) if role is None else role,
        )
        self._write_integer(payload, fields["reserved"], 0)
        payload.extend(self._topology_tlv(identity))
        return self._frame(
            message_type=self.hello_type,
            request_id=request_id,
            daemon_uuid=bytes.fromhex(identity.daemon_uuid),
            connection_id=0,
            epoch=identity.epoch,
            payload=bytes(payload),
        )

    def decode_ack(
        self,
        frame: bytes,
        *,
        request_id: int,
        client_nonce: bytes,
        expected_status: int,
        identity: Identity,
        configured_maximum: int,
    ) -> RawAck:
        require(len(frame) >= self.header_bytes + self.ack_bytes,
                f"short raw ACK: {len(frame)} bytes")
        require(len(frame) <= self.maximum_record,
                f"oversized raw ACK: {len(frame)} bytes")
        require(frame[:len(self.magic)] == self.magic, "raw ACK magic mismatch")
        for name in ("framing_major", "framing_minor", "header_bytes", "flags",
                     "reserved0", "reserved1"):
            field = self.header_field(name)
            require(
                self._read_integer(frame, field) == int(field["constant"]),
                f"raw ACK header field {name} mismatch",
            )
        require(self._read_integer(frame, self.header_field("message_type")) ==
                self.ack_type, "raw response is not HELLO_ACK")
        require(self._read_integer(frame, self.header_field("payload_bytes")) ==
                len(frame) - self.header_bytes, "raw ACK payload length mismatch")
        checksum_field = self.header_field("crc32c")
        checksum = self._read_integer(frame, checksum_field)
        zeroed = bytearray(frame)
        offset = int(checksum_field["offset"])
        zeroed[offset:offset + int(checksum_field["bytes"])] = bytes(
            int(checksum_field["bytes"])
        )
        require(self.crc32c(bytes(zeroed)) == checksum, "raw ACK CRC32C mismatch")
        actual_request = self._read_integer(frame, self.header_field("request_id"))
        require(actual_request == request_id, "raw ACK request ID mismatch")

        fields = {
            field["name"]: field
            for field in self.document["messages"]["HELLO_ACK"]["fields"]
        }
        base = self.header_bytes
        status = self._read_integer(frame, fields["status"], base)
        require(status == expected_status,
                f"raw ACK status {status}, expected {expected_status}")
        nonce_offset = base + int(fields["client_nonce_echo"]["offset"])
        actual_nonce = frame[nonce_offset:nonce_offset + len(client_nonce)]
        require(actual_nonce == client_nonce, "raw ACK client nonce mismatch")
        daemon_field = self.header_field("daemon_instance_uuid")
        daemon_offset = int(daemon_field["offset"])
        daemon_uuid = frame[daemon_offset:daemon_offset + int(daemon_field["bytes"])]
        require(daemon_uuid.hex() == identity.daemon_uuid,
                "raw ACK daemon UUID mismatch")
        epoch = self._read_integer(frame, self.header_field("job_epoch"))
        require(epoch == identity.epoch, "raw ACK epoch mismatch")
        connection_id = self._read_integer(frame, self.header_field("connection_id"))
        server_field = fields["server_nonce"]
        server_offset = base + int(server_field["offset"])
        server_nonce = frame[server_offset:server_offset + int(server_field["bytes"])]
        selected_field = fields["selected_capabilities"]
        selected_offset = base + int(selected_field["offset"])
        selected = frame[selected_offset:selected_offset + int(selected_field["bytes"])]
        maximum = self._read_integer(frame, fields["max_record"], base)
        role = self._read_integer(frame, fields["role"], base)
        reserved = self._read_integer(frame, fields["reserved"], base)
        require(role == int(fields["role"]["constant"]), "raw ACK role mismatch")
        require(reserved == 0, "raw ACK reserved field is nonzero")

        topology_job: str | None = None
        topology_rank: int | None = None
        topology_world: int | None = None
        trailing = frame[base + self.ack_bytes:]
        if status == self.statuses["OK"]:
            require(
                self._read_integer(frame, fields["selected_major"], base) == 1 and
                self._read_integer(frame, fields["selected_minor"], base) == 0,
                "successful raw ACK selected the wrong version",
            )
            require(connection_id != 0, "successful raw ACK has zero connection ID")
            require(any(server_nonce), "successful raw ACK has zero server nonce")
            require(selected == self._capabilities(self.topology_bit),
                    "successful raw ACK selected capabilities mismatch")
            require(maximum == configured_maximum,
                    "successful raw ACK maximum record mismatch")
            require(len(trailing) == 32, "successful raw ACK topology TLV length mismatch")
            tlv_type, tlv_flags, value_bytes = struct.unpack(">HHI", trailing[:8])
            require(
                (tlv_type, tlv_flags, value_bytes) ==
                (self.topology_type, self.critical_flag, 24),
                "successful raw ACK topology TLV header mismatch",
            )
            topology_job = trailing[8:24].hex()
            topology_rank, topology_world = struct.unpack(">II", trailing[24:32])
            require(
                (topology_job, topology_rank, topology_world) ==
                (identity.job_uuid, identity.rank, identity.world_size),
                "successful raw ACK topology mismatch",
            )
        else:
            require(
                self._read_integer(frame, fields["selected_major"], base) == 0 and
                self._read_integer(frame, fields["selected_minor"], base) == 0,
                "failed raw ACK selected a version",
            )
            require(connection_id == 0, "failed raw ACK has a connection ID")
            require(not any(server_nonce), "failed raw ACK has a server nonce")
            require(not any(selected), "failed raw ACK selected capabilities")
            require(maximum == configured_maximum,
                    "failed raw ACK maximum record is not daemon configured limit")
            require(not trailing, "failed raw ACK contains TLVs")
        return RawAck(
            status=status,
            request_id=actual_request,
            daemon_uuid=daemon_uuid.hex(),
            connection_id=connection_id,
            epoch=epoch,
            client_nonce=actual_nonce,
            server_nonce=server_nonce,
            selected_capabilities=selected,
            maximum_record=maximum,
            topology_job_uuid=topology_job,
            topology_rank=topology_rank,
            topology_world_size=topology_world,
        )

    def assert_canonical_golden(self) -> None:
        golden = self.document["golden"]
        values = golden["identity"]
        identity = Identity(
            daemon_uuid=values["daemon_instance_uuid_hex"],
            job_uuid=values["job_uuid_hex"],
            epoch=int(values["job_epoch_hex"], 16),
            rank=int(values["rank"]),
            world_size=int(values["world_size"]),
        )
        frame = self.encode_hello(
            identity,
            request_id=int(values["request_id_hex"], 16),
            client_nonce=bytes.fromhex(values["client_nonce_hex"]),
            receive_maximum=int(values["rx_max_record"]),
        )
        expected = bytes.fromhex(golden["hello_success_request"]["frame_hex"])
        require(frame == expected, "dynamic raw HELLO differs from canonical golden")


class RawSeqpacketClient:
    def __init__(
        self,
        endpoint: Path,
        timeout: float,
        trace=None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.trace = trace or (lambda _event, **_details: None)
        self.socket: socket.socket | None = None
        self.deadline: float | None = None

    def _arm_deadline(self) -> None:
        assert self.socket is not None
        assert self.deadline is not None
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"raw socket deadline expired for {self.endpoint}")
        self.socket.settimeout(remaining)

    def __enter__(self) -> "RawSeqpacketClient":
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.deadline = time.monotonic() + self.timeout
        try:
            self._arm_deadline()
            self.socket.connect(str(self.endpoint))
        except Exception:
            self.socket.close()
            self.socket = None
            self.deadline = None
            raise
        self.trace("connect", endpoint=str(self.endpoint))
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        self.deadline = None

    def send(self, record: bytes) -> None:
        assert self.socket is not None
        self._arm_deadline()
        sent = self.socket.send(record)
        require(sent == len(record), "raw seqpacket send was partial")
        self.trace("send", bytes=sent)

    def send_rights(self, record: bytes, descriptor_count: int) -> None:
        assert self.socket is not None
        require(descriptor_count > 0, "descriptor count must be positive")
        self._arm_deadline()
        source = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
        try:
            rights = array.array("i", [source] * descriptor_count)
            sent = self.socket.sendmsg(
                [record], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)]
            )
        finally:
            os.close(source)
        require(sent == len(record), "raw ancillary seqpacket send was partial")
        self.trace("send_rights", bytes=sent, descriptors=descriptor_count)

    def receive(self) -> bytes:
        assert self.socket is not None
        self._arm_deadline()
        frame = self.socket.recv(65537)
        self.trace("receive", bytes=len(frame))
        return frame

    def expect_closed(self) -> None:
        frame = self.receive()
        require(frame == b"", f"peer returned {len(frame)} bytes instead of closing")

    def expect_open_and_idle(self, duration: float) -> None:
        assert self.socket is not None
        assert self.deadline is not None
        remaining = self.deadline - time.monotonic()
        require(remaining > duration,
                "raw client deadline is too short for idle-open probe")
        readable, _, _ = select.select([self.socket], [], [], duration)
        require(not readable,
                "peer closed or responded before the delayed HELLO was sent")
        self.trace("idle_open", duration_seconds=duration)



def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def write_retained_stream(path: Path, content: str) -> None:
    """Persist an audited child stream as a private, durable artifact."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def tail(path: Path, lines: int = 80) -> str:
    return "\n".join(read_text(path).splitlines()[-lines:])


def parse_last_json(text: str, source: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise CheckFailure(f"{source} did not contain a JSON object: {text[-1000:]}")


def parse_single_json_object(text: str, source: str) -> dict[str, Any]:
    require(len(text) <= MAX_DISPATCH_JSON_BYTES,
            f"{source} exceeds {MAX_DISPATCH_JSON_BYTES} bytes")
    require(text.endswith("\n"), f"{source} is not newline terminated")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    require(len(lines) == 1,
            f"{source} did not contain exactly one JSON record")
    return strict_json_object(lines[0], source)


def wait_until(predicate, timeout: float, description: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise CheckFailure(f"timed out waiting for {description}")


def terminate_process(process: subprocess.Popen[Any], grace: float = 1.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGCONT)
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=grace)


def kill_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=2.0)


def process_connected_unix_socket_count(pid: int) -> int:
    socket_inodes: set[str] = set()
    try:
        for entry in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(entry)
            except OSError:
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                socket_inodes.add(target[8:-1])
    except FileNotFoundError:
        return 0
    if not socket_inodes:
        return 0
    try:
        lines = Path("/proc/net/unix").read_text(encoding="ascii").splitlines()
    except OSError:
        return 0
    connected: set[str] = set()
    for line in lines[1:]:
        fields = line.split()
        if len(fields) >= 7 and fields[4] == "0005" and fields[5] == "03":
            if fields[6] in socket_inodes:
                connected.add(fields[6])
    return len(connected)


def process_has_connected_unix_socket(pid: int) -> bool:
    return process_connected_unix_socket_count(pid) > 0


def process_fd_count(pid: int) -> int:
    try:
        return len(list(Path(f"/proc/{pid}/fd").iterdir()))
    except FileNotFoundError as error:
        raise CheckFailure(f"process {pid} exited while counting descriptors") from error


def make_identity(tag: int, world_size: int, rank: int) -> Identity:
    daemon_uuid = uuid.uuid5(
        RUN_IDENTITY_NAMESPACE, f"daemon:{tag}:{world_size}:{rank}"
    ).hex
    job_uuid = uuid.uuid5(
        RUN_IDENTITY_NAMESPACE, f"job:{tag}:{world_size}"
    ).hex
    epoch = (
        int.from_bytes(RUN_IDENTITY_NAMESPACE.bytes[:8], "big")
        ^ ((tag & 0xFFFF) << 32)
        ^ world_size
    ) or 1
    return Identity(
        daemon_uuid=daemon_uuid,
        job_uuid=job_uuid,
        epoch=epoch,
        rank=rank,
        world_size=world_size,
    )


class Daemon:
    def __init__(
        self,
        harness: "Harness",
        name: str,
        identity: Identity,
        *,
        exit_on_handshake: bool,
        endpoint: Path | None = None,
        maximum_record: int = 65536,
        run_timeout_ms: int | None = None,
        handshake_timeout_ms: int | None = None,
        config: Path | None = None,
        extra_args: Iterable[str] = (),
    ) -> None:
        self.harness = harness
        self.name = name
        self.identity = identity
        self.exit_on_handshake = exit_on_handshake
        self.maximum_record = maximum_record
        self.config = config or harness.gem5_config
        self.extra_args = tuple(extra_args)
        self.run_timeout_ms = run_timeout_ms or harness.run_timeout_ms
        self.handshake_timeout_ms = (
            handshake_timeout_ms or harness.handshake_timeout_ms
        )
        self.endpoint = endpoint or (harness.work_dir / f"{name}.sock")
        self.lock_path = Path(f"{self.endpoint}.lock")
        self.out_dir = harness.work_dir / f"{name}.m5out"
        self.log_path = harness.work_dir / f"{name}.gem5.log"
        self.process: subprocess.Popen[Any] | None = None

    def argv(self, listener_mode: str = "on") -> list[str]:
        argv = [
            str(self.harness.gem5),
            f"--listener-mode={listener_mode}",
            f"--outdir={self.out_dir}",
            str(self.config),
            "--endpoint", str(self.endpoint),
            "--daemon-uuid", self.identity.daemon_uuid,
            "--job-uuid", self.identity.job_uuid,
            "--epoch", str(self.identity.epoch),
            "--rank", str(self.identity.rank),
            "--world-size", str(self.identity.world_size),
            "--startup-timeout-ms", str(self.harness.startup_timeout_ms),
            "--handshake-timeout-ms", str(self.handshake_timeout_ms),
            "--run-timeout-ms", str(self.run_timeout_ms),
            "--max-record", str(self.maximum_record),
            *self.extra_args,
        ]
        if self.exit_on_handshake:
            argv.append("--exit-on-handshake")
        return argv

    def launch(self, *, wait_ready: bool = True, listener_mode: str = "on") -> None:
        require(self.process is None, f"daemon {self.name} was launched twice")
        require(
            len(os.fsencode(self.endpoint)) < 108,
            f"AF_UNIX endpoint is too long: {self.endpoint}",
        )
        self.out_dir.mkdir(mode=0o700)
        os.chmod(self.out_dir, 0o700)
        with self.log_path.open("wb", buffering=0) as log_stream:
            self.process = subprocess.Popen(
                self.argv(listener_mode),
                cwd=ROOT,
                env=self.harness.environment,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
        self.harness.daemons.append(self)
        if wait_ready:
            self.wait_ready()

    def wait_ready(self) -> None:
        assert self.process is not None

        def ready() -> bool:
            if self.process is None or self.process.poll() is not None:
                raise CheckFailure(
                    f"daemon {self.name} exited before readiness "
                    f"(rc={self.process.returncode if self.process else 'unknown'}):\n"
                    f"{tail(self.log_path)}"
                )
            try:
                metadata = self.endpoint.lstat()
            except FileNotFoundError:
                return False
            if READY_TOKEN not in read_text(self.log_path):
                return False
            require(stat.S_ISSOCK(metadata.st_mode),
                    f"daemon {self.name} ready endpoint is not a socket")
            require(stat.S_IMODE(metadata.st_mode) == 0o600,
                    f"daemon {self.name} endpoint mode is not 0600")
            return True

        wait_until(ready, self.harness.start_wait_seconds,
                   f"daemon {self.name} readiness")

    def wait(self, timeout: float | None = None) -> int:
        assert self.process is not None
        try:
            return self.process.wait(timeout=timeout or self.harness.process_timeout)
        except subprocess.TimeoutExpired as error:
            raise CheckFailure(
                f"daemon {self.name} did not exit:\n{tail(self.log_path)}"
            ) from error

    def expect_success_exit(self) -> None:
        returncode = self.wait()
        log = read_text(self.log_path)
        require(returncode == 0,
                f"daemon {self.name} exited rc={returncode}:\n{tail(self.log_path)}")
        require(SUCCESS_EXIT_TOKEN in log,
                f"daemon {self.name} did not report handshake exit:\n{tail(self.log_path)}")

    def stop(self) -> None:
        if self.process is not None:
            terminate_process(self.process)

    def kill(self) -> None:
        if self.process is not None:
            kill_process(self.process)


class Harness:
    def __init__(self, args: argparse.Namespace, work_dir: Path) -> None:
        self.gem5 = args.gem5
        self.runtime_cli = args.runtime_cli
        self.gem5_config = args.gem5_config
        self.dispatch_gem5_config = args.dispatch_gem5_config
        self.work_dir = work_dir
        self.start_wait_seconds = args.start_wait_seconds
        self.process_timeout = args.process_timeout_seconds
        self.dispatch_process_timeout = args.dispatch_process_timeout_seconds
        self.client_timeout_ms = args.client_timeout_ms
        self.startup_timeout_ms = args.server_startup_timeout_ms
        self.handshake_timeout_ms = args.server_handshake_timeout_ms
        self.run_timeout_ms = args.server_run_timeout_ms
        self.dispatch_run_timeout_ms = args.dispatch_server_run_timeout_ms
        self.hold_ms = args.hold_ms
        self.environment = os.environ.copy()
        self.environment.update({
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        })
        self.daemons: list[Daemon] = []
        self.checks: list[dict[str, Any]] = []
        self.wire = WireProtocol.load()
        self.raw_trace_path = work_dir / "raw-wire.jsonl"
        self.raw_sequence = 0

    def add_check(self, name: str, **details: Any) -> None:
        self.checks.append({"name": name, "status": "passed", **details})

    def trace_raw(self, event: str, **details: Any) -> None:
        entry = {
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
            **details,
        }
        with self.raw_trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")

    def raw_material(self, tag: str) -> tuple[int, bytes]:
        self.raw_sequence += 1
        material = uuid.uuid5(
            RUN_IDENTITY_NAMESPACE, f"raw:{self.raw_sequence}:{tag}"
        ).bytes
        request_id = int.from_bytes(material[:8], "big") or 1
        nonce = uuid.uuid5(
            RUN_IDENTITY_NAMESPACE, f"raw-nonce:{self.raw_sequence}:{tag}"
        ).bytes
        return request_id, nonce

    def raw_client(self, daemon: Daemon, timeout: float | None = None) -> RawSeqpacketClient:
        return RawSeqpacketClient(
            daemon.endpoint,
            timeout or self.process_timeout,
            lambda event, **details: self.trace_raw(
                event, daemon=daemon.name, **details
            ),
        )

    def raw_round_trip(
        self,
        client: RawSeqpacketClient,
        daemon: Daemon,
        *,
        tag: str,
        expected_status: int,
        minimum_version: tuple[int, int] = (1, 0),
        maximum_version: tuple[int, int] = (1, 0),
        role: int | None = None,
    ) -> RawAck:
        request_id, nonce = self.raw_material(tag)
        hello = self.wire.encode_hello(
            daemon.identity,
            request_id=request_id,
            client_nonce=nonce,
            minimum_version=minimum_version,
            maximum_version=maximum_version,
            role=role,
            receive_maximum=daemon.maximum_record,
        )
        client.send(hello)
        response = client.receive()
        require(response, f"raw {tag} peer closed without an ACK")
        ack = self.wire.decode_ack(
            response,
            request_id=request_id,
            client_nonce=nonce,
            expected_status=expected_status,
            identity=daemon.identity,
            configured_maximum=daemon.maximum_record,
        )
        self.trace_raw("ack", daemon=daemon.name, tag=tag, status=ack.status)
        return ack

    def wait_for_daemon_fd_count(
        self,
        daemon: Daemon,
        expected: int,
        description: str,
    ) -> None:
        assert daemon.process is not None
        observed = -1

        def restored() -> bool:
            nonlocal observed
            require(daemon.process is not None and daemon.process.poll() is None,
                    f"daemon {daemon.name} exited while waiting for {description}:\n"
                    f"{tail(daemon.log_path)}")
            observed = process_fd_count(daemon.process.pid)
            return observed == expected

        try:
            wait_until(restored, self.process_timeout, description)
        except CheckFailure as error:
            raise CheckFailure(
                f"{error}; expected {expected} descriptors, observed {observed}:\n"
                f"{tail(daemon.log_path)}"
            ) from error

    def client_argv(
        self,
        endpoint: Path,
        identity: Identity,
        *,
        daemon_uuid: str | None = None,
        job_uuid: str | None = None,
        epoch: int | None = None,
        rank: int | None = None,
        world_size: int | None = None,
        timeout_ms: int | None = None,
        extra: Iterable[str] = (),
    ) -> list[str]:
        return [
            str(self.runtime_cli),
            "--endpoint", str(endpoint),
            "--expected-daemon-uuid", daemon_uuid or identity.daemon_uuid,
            "--expected-job-uuid", job_uuid or identity.job_uuid,
            "--expected-epoch", str(identity.epoch if epoch is None else epoch),
            "--expected-rank", str(identity.rank if rank is None else rank),
            "--expected-world", str(
                identity.world_size if world_size is None else world_size
            ),
            "--timeout-ms", str(timeout_ms or self.client_timeout_ms),
            *extra,
        ]

    def run_client_argv(self, argv: list[str], timeout: float | None = None) -> ClientResult:
        try:
            result = subprocess.run(
                argv,
                cwd=ROOT,
                env=self.environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                close_fds=True,
                timeout=timeout or self.process_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise CheckFailure(f"runtime CLI timed out: {' '.join(argv)}") from error
        return ClientResult(argv, result.returncode, result.stdout, result.stderr)

    def run_client_argv_audited(
        self,
        argv: list[str],
        daemon: Daemon,
        *,
        timeout: float,
        retained_name: str | None = None,
    ) -> tuple[ClientResult, ProcessAuditCounts]:
        require(timeout > 0, "audited runtime timeout must be positive")
        require(daemon.process is not None and daemon.process.poll() is None,
                f"daemon {daemon.name} is not live for the process audit")
        daemon_samples = 0
        runtime_samples = 0

        def audit_daemon() -> None:
            nonlocal daemon_samples
            assert daemon.process is not None
            try:
                audit = capture_process_audit(
                    daemon.process.pid, daemon.log_path
                )
            except CheckFailure:
                if daemon.process.poll() is not None:
                    return
                raise
            validate_process_audit(audit)
            daemon_samples += 1

        audit_daemon()
        require(daemon_samples == 1,
                f"daemon {daemon.name} exited before its live audit sample")
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=self.environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            close_fds=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                try:
                    audit = capture_process_audit(process.pid)
                except CheckFailure:
                    try:
                        process.wait(timeout=0.05)
                    except subprocess.TimeoutExpired:
                        raise
                    else:
                        break
                validate_process_audit(audit)
                runtime_samples += 1
                if time.monotonic() >= deadline:
                    raise CheckFailure(
                        f"runtime CLI timed out under process audit: "
                        f"{' '.join(argv)}"
                    )
                if daemon.process.poll() is None:
                    audit_daemon()
                time.sleep(0.01)
            stdout, stderr = process.communicate(timeout=1.0)
        except Exception:
            terminate_process(process)
            raise
        if daemon.process.poll() is None:
            audit_daemon()
        require(runtime_samples > 0,
                "runtime CLI exited before a live provenance audit sample")
        validate_process_audit(ProcessAudit(
            maps="", open_paths=(), log=read_text(daemon.log_path)
        ))
        validate_process_audit(ProcessAudit(
            maps="", open_paths=(), log=stdout + "\n" + stderr
        ))
        if retained_name is not None:
            write_retained_stream(
                self.work_dir / f"{retained_name}.runtime.stdout", stdout
            )
            write_retained_stream(
                self.work_dir / f"{retained_name}.runtime.stderr", stderr
            )
        return (
            ClientResult(argv, process.returncode, stdout, stderr),
            ProcessAuditCounts(daemon_samples, runtime_samples),
        )

    def expect_failure(
        self,
        result: ClientResult,
        status: int,
        status_name: str,
        wire_status: int,
    ) -> dict[str, Any]:
        require(result.returncode == 1,
                f"runtime CLI returned {result.returncode}, expected 1: "
                f"stdout={result.stdout!r} stderr={result.stderr!r}")
        payload = parse_last_json(result.stderr, "runtime CLI stderr")
        require(payload.get("status") == status,
                f"wrong runtime status for {status_name}: {payload}")
        require(payload.get("status_name") == status_name,
                f"wrong runtime status name: {payload}")
        require(payload.get("wire_status") == wire_status,
                f"wrong wire status for {status_name}: {payload}")
        return payload

    def validate_success(
        self,
        result: ClientResult,
        identity: Identity,
        daemon: Daemon,
        capability_words: list[str] | None = None,
    ) -> dict[str, Any]:
        require(result.returncode == 0,
                f"runtime CLI failed: stdout={result.stdout!r} stderr={result.stderr!r}")
        payload = parse_last_json(result.stdout, "runtime CLI stdout")
        require(payload.get("status") == 0, f"success status is wrong: {payload}")
        require(payload.get("selected_version") == "1.0",
                f"selected version is wrong: {payload}")
        expected_capabilities = capability_words or [
            "0x0000000000000001", "0x0000000000000000",
            "0x0000000000000000", "0x0000000000000000",
        ]
        require(payload.get("capability_words") == expected_capabilities,
                f"selected capabilities are wrong: {payload}")
        require(payload.get("daemon_uuid") == identity.daemon_uuid,
                f"daemon UUID is wrong: {payload}")
        require(payload.get("job_uuid") == identity.job_uuid,
                f"job UUID is wrong: {payload}")
        require(int(payload.get("epoch", "0"), 0) == identity.epoch,
                f"epoch is wrong: {payload}")
        require(payload.get("rank") == identity.rank and
                payload.get("world_size") == identity.world_size,
                f"topology is wrong: {payload}")
        require(payload.get("maximum_record_bytes") == daemon.maximum_record,
                f"maximum record is wrong: {payload}")
        require(int(payload.get("connection_id", "0"), 0) != 0,
                f"connection ID is zero: {payload}")
        require(int(payload.get("request_id", "0"), 0) != 0,
                f"request ID is zero: {payload}")
        require(payload.get("peer_uid") == os.geteuid(),
                f"peer UID is wrong: {payload}")
        require(daemon.process is not None and
                payload.get("peer_pid") == daemon.process.pid,
                f"runtime connected to the wrong daemon process: {payload}")
        return payload

    def validate_dispatch_success(
        self,
        result: ClientResult,
        identity: Identity,
        daemon: Daemon,
        authority: dict[str, Any],
    ) -> tuple[dict[str, Any], DispatchEvidence]:
        require(result.returncode == 0,
                f"dispatch runtime CLI failed: stdout={result.stdout!r} "
                f"stderr={result.stderr!r}")
        require(result.stderr == "",
                f"dispatch runtime CLI emitted unexpected stderr: "
                f"{result.stderr!r}")
        payload = parse_single_json_object(
            result.stdout, "dispatch runtime CLI stdout"
        )
        require_exact_object(
            payload, DISPATCH_CLIENT_RESULT_KEYS,
            "dispatch runtime CLI result",
        )
        ordinary = self.validate_success(
            result,
            identity,
            daemon,
            [
                "0x000000000000001f",
                "0x0000000000000000",
                "0x0000000000000000",
                "0x0000000000000000",
            ],
        )
        require(ordinary == payload,
                "strict and ordinary runtime result parsing disagreed")
        evidence = validate_dispatch_result(payload["dispatch"], authority)
        hello_request_id = require_fixed_hex_unsigned(
            payload["request_id"], "dispatch runtime HELLO request_id", 64,
            nonzero=True,
        )
        require(evidence.request_id > hello_request_id,
                "dispatch request ID did not advance beyond HELLO")
        return payload, evidence

    def validate_memory_success(
        self,
        payload: dict[str, Any],
        *,
        expected_bytes: int,
        expected_alignment: int,
        require_reuse: bool,
    ) -> dict[str, Any]:
        memory = payload.get("memory")
        require(isinstance(memory, dict), f"memory result is missing: {payload}")
        require(memory.get("status") == 0, f"memory status is wrong: {memory}")
        allocation_id = int(memory.get("allocation_id", "0"), 0)
        generation = int(memory.get("generation", "0"), 0)
        simulated_va = int(memory.get("simulated_va", "0"), 0)
        require(1 <= allocation_id <= 1024,
                f"memory allocation ID is out of range: {memory}")
        require(generation != 0, f"memory generation is zero: {memory}")
        expected_va = 0x0000100000000000 + (allocation_id - 1) * 0x80000000
        require(simulated_va == expected_va,
                f"memory simulated VA does not match its slot: {memory}")
        require(memory.get("size_bytes") == expected_bytes and
                memory.get("alignment_bytes") == expected_alignment,
                f"memory size or alignment is wrong: {memory}")
        require(memory.get("initial_zero") is True,
                f"new allocation was not zero-filled: {memory}")
        require(memory.get("match") is True,
                f"memory roundtrip bytes did not match: {memory}")
        require(memory.get("freed") is True,
                f"memory allocation was not freed: {memory}")
        pattern_crc = int(memory.get("pattern_crc32c", "-1"), 0)
        returned_crc = int(memory.get("returned_crc32c", "-2"), 0)
        require(0 <= pattern_crc <= 0xFFFFFFFF and
                pattern_crc == returned_crc,
                f"memory roundtrip CRC is wrong: {memory}")

        reuse = memory.get("reuse")
        if require_reuse:
            require(isinstance(reuse, dict),
                    f"memory reuse result is missing: {memory}")
            require(int(reuse.get("allocation_id", "0"), 0) == allocation_id,
                    f"memory slot was not deterministically reused: {reuse}")
            reuse_generation = int(reuse.get("generation", "0"), 0)
            require(reuse_generation != 0 and reuse_generation != generation,
                    f"memory reuse generation did not change: {reuse}")
            require(int(reuse.get("simulated_va", "0"), 0) == simulated_va,
                    f"memory reuse changed the slot VA: {reuse}")
            require(reuse.get("initial_zero") is True and
                    reuse.get("freed") is True,
                    f"reused allocation was not zero-filled and freed: {reuse}")
        else:
            require(reuse is None, f"unexpected memory reuse result: {memory}")
        return memory

    def validate_signal_success(
        self,
        payload: dict[str, Any],
        *,
        expected_initial: int,
        expected_stored: int,
    ) -> dict[str, Any]:
        signal_result = payload.get("signal")
        require(isinstance(signal_result, dict),
                f"signal result is missing: {payload}")
        require(signal_result.get("status") == 0,
                f"signal status is wrong: {signal_result}")
        signal_id = int(signal_result.get("signal_id", "0"), 0)
        generation = int(signal_result.get("generation", "0"), 0)
        require(1 <= signal_id <= 1024,
                f"signal ID is out of range: {signal_result}")
        require(generation != 0,
                f"signal generation is zero: {signal_result}")
        require(signal_result.get("initial_value") == expected_initial and
                signal_result.get("load_before") == expected_initial,
                f"signal initial load is wrong: {signal_result}")
        require(signal_result.get("stored_value") == expected_stored and
                signal_result.get("load_after") == expected_stored,
                f"signal stored load is wrong: {signal_result}")
        require(signal_result.get("destroyed") is True,
                f"signal was not destroyed: {signal_result}")

        wait = signal_result.get("wait")
        require(isinstance(wait, dict),
                f"signal wait result is missing: {signal_result}")
        require(wait.get("condition") == "gte" and wait.get("compare") == 0,
                f"signal wait predicate is wrong: {wait}")
        require(wait.get("first_status") == 11 and
                wait.get("first_status_name") == "timed out",
                f"signal wait did not first time out: {wait}")
        require(wait.get("completion_status") == 0 and
                wait.get("observed_value") == expected_stored,
                f"signal completion is not canonical: {wait}")
        require(int(wait.get("sequence", "0"), 0) != 0,
                f"signal wait sequence is zero: {wait}")
        require(wait.get("retried_without_send") is True,
                f"signal wait retry sent a second request: {wait}")

        reuse = signal_result.get("reuse")
        require(isinstance(reuse, dict),
                f"signal reuse result is missing: {signal_result}")
        require(int(reuse.get("signal_id", "0"), 0) == signal_id,
                f"signal slot was not deterministically reused: {reuse}")
        reuse_generation = int(reuse.get("generation", "0"), 0)
        require(reuse_generation > generation,
                f"signal reuse generation did not advance: {reuse}")
        require(reuse.get("initial_value") == expected_initial and
                reuse.get("destroyed") is True,
                f"reused signal lifecycle is incomplete: {reuse}")
        return signal_result

    def exact_client(self, daemon: Daemon, **kwargs: Any) -> ClientResult:
        return self.run_client_argv(
            self.client_argv(daemon.endpoint, daemon.identity, **kwargs)
        )

    def check_reject_then_success(self) -> None:
        identity = make_identity(1, 1, 0)
        daemon = Daemon(self, "reject-success", identity,
                        exit_on_handshake=True, maximum_record=32768)
        daemon.launch()

        result = self.exact_client(
            daemon, extra=("--min-version", "2.0", "--max-version", "2.0")
        )
        self.expect_failure(result, 4, "version mismatch", 2)

        result = self.exact_client(
            daemon,
            extra=(
                "--offer-cap-bit", str(UNSUPPORTED_CAPABILITY_PROBE_BIT),
                "--require-cap-bit", str(UNSUPPORTED_CAPABILITY_PROBE_BIT),
            ),
        )
        self.expect_failure(result, 5, "capability mismatch", 3)

        wrong_instance = make_identity(2, 1, 0).daemon_uuid
        result = self.exact_client(daemon, daemon_uuid=wrong_instance)
        self.expect_failure(result, 7, "instance mismatch", 4)

        wrong_job = make_identity(2, 1, 0).job_uuid
        result = self.exact_client(daemon, job_uuid=wrong_job)
        self.expect_failure(result, 8, "topology mismatch", 5)

        success = self.validate_success(self.exact_client(daemon), identity, daemon)
        daemon.expect_success_exit()
        log = read_text(daemon.log_path)
        for status_name in (
            "UNSUPPORTED_VERSION", "UNSUPPORTED_CAPABILITY",
            "INSTANCE_MISMATCH", "TOPOLOGY_MISMATCH", "OK",
        ):
            require(f"status={status_name}" in log,
                    f"daemon did not log {status_name}:\n{tail(daemon.log_path)}")
        self.add_check(
            "reject_then_success",
            wire_statuses=[2, 3, 4, 5, 0],
            daemon_uuid=success["daemon_uuid"],
            peer_pid=success["peer_pid"],
        )

    def check_queue_control(self) -> None:
        expected_capabilities = [
            "0x0000000000000003", "0x0000000000000000",
            "0x0000000000000000", "0x0000000000000000",
        ]
        success_identity = make_identity(20, 1, 0)
        success_daemon = Daemon(
            self, "queue-control-success", success_identity,
            exit_on_handshake=False, run_timeout_ms=8000,
        )
        success_daemon.launch()
        success_result = self.exact_client(
            success_daemon,
            extra=(
                "--queue-depth", "4",
                "--doorbells", "3",
                "--command-kind", "1",
            ),
        )
        success = self.validate_success(
            success_result, success_identity, success_daemon,
            expected_capabilities,
        )
        queue = success.get("queue")
        require(isinstance(queue, dict), f"queue result is missing: {success}")
        require(queue.get("status") == 0 and queue.get("depth") == 4,
                f"queue success metadata is wrong: {queue}")
        require(queue.get("command_kind") == 1,
                f"queue command kind is wrong: {queue}")
        require(queue.get("completion_status") == 0 and
                queue.get("completion_wire_status") == 0 and
                queue.get("completion_error_code") == 0,
                f"queue completion is not canonical success: {queue}")
        require(int(queue.get("queue_id", "0"), 0) != 0 and
                int(queue.get("generation", "0"), 0) != 0,
                f"queue handle is zero: {queue}")
        sequences = [int(value, 0) for value in queue.get("sequences", [])]
        require(sequences == [1, 2, 3],
                f"queue sequences are not strictly ordered: {queue}")
        success_daemon.stop()

        error_identity = make_identity(21, 1, 0)
        error_daemon = Daemon(
            self, "queue-control-error", error_identity,
            exit_on_handshake=False, run_timeout_ms=8000,
        )
        error_daemon.launch()
        error_result = self.exact_client(
            error_daemon,
            extra=(
                "--queue-depth", "2",
                "--doorbells", "1",
                "--command-kind", "2",
            ),
        )
        error = self.validate_success(
            error_result, error_identity, error_daemon,
            expected_capabilities,
        )
        error_queue = error.get("queue")
        require(isinstance(error_queue, dict),
                f"queue error result is missing: {error}")
        require(error_queue.get("command_kind") == 2 and
                error_queue.get("completion_status") == 3 and
                error_queue.get("completion_wire_status") == 10 and
                error_queue.get("completion_error_code") == 1,
                f"queue error completion is not canonical: {error_queue}")
        error_daemon.stop()
        self.add_check(
            "queue_control_lifecycle",
            success_sequences=sequences,
            success_peer_pid=success["peer_pid"],
            error_peer_pid=error["peer_pid"],
            error_wire_status=error_queue["completion_wire_status"],
        )

    def check_memory_transfer(self) -> None:
        expected_capabilities = [
            "0x0000000000000005", "0x0000000000000000",
            "0x0000000000000000", "0x0000000000000000",
        ]
        memory_bytes = 2 * 65536 + 17
        alignment = 65536
        identity = make_identity(22, 1, 0)
        daemon = Daemon(
            self, "simulated-memory", identity,
            exit_on_handshake=False, run_timeout_ms=12000,
        )
        daemon.launch()
        result = self.exact_client(
            daemon,
            extra=(
                "--memory-bytes", str(memory_bytes),
                "--memory-alignment", str(alignment),
                "--memory-reuse",
            ),
        )
        success = self.validate_success(
            result, identity, daemon, expected_capabilities
        )
        memory = self.validate_memory_success(
            success,
            expected_bytes=memory_bytes,
            expected_alignment=alignment,
            require_reuse=True,
        )
        daemon.stop()
        self.add_check(
            "simulated_memory_roundtrip",
            bytes=memory_bytes,
            allocation_id=memory["allocation_id"],
            generation=memory["generation"],
            simulated_va=memory["simulated_va"],
            crc32c=memory["returned_crc32c"],
            peer_pid=success["peer_pid"],
        )

    def check_signal_lifecycle(self) -> None:
        expected_capabilities = [
            "0x0000000000000009", "0x0000000000000000",
            "0x0000000000000000", "0x0000000000000000",
        ]
        initial_value = -7
        stored_value = 42
        identity = make_identity(23, 1, 0)
        daemon = Daemon(
            self, "signal-lifecycle", identity,
            exit_on_handshake=False, run_timeout_ms=12000,
        )
        daemon.launch()
        result = self.exact_client(
            daemon,
            extra=(
                "--signal-initial", str(initial_value),
                "--signal-wait-condition", "gte",
                "--signal-wait-compare", "0",
                "--signal-wait-timeout-ms", "50",
                "--signal-store", str(stored_value),
                "--signal-reuse",
            ),
        )
        success = self.validate_success(
            result, identity, daemon, expected_capabilities
        )
        signal_result = self.validate_signal_success(
            success,
            expected_initial=initial_value,
            expected_stored=stored_value,
        )
        daemon.stop()
        wait = signal_result["wait"]
        reuse = signal_result["reuse"]
        self.add_check(
            "signal_event_lifecycle",
            signal_id=signal_result["signal_id"],
            generation=signal_result["generation"],
            wait_sequence=wait["sequence"],
            first_wait_status=wait["first_status_name"],
            observed_value=wait["observed_value"],
            reused_generation=reuse["generation"],
            peer_pid=success["peer_pid"],
        )

    def check_pinned_dispatch(self) -> None:
        authority = load_json_document(
            DISPATCH_PROTOCOL_PATH,
            "amdgpu-sim.host-transport-v1.dispatch.v1",
        )
        identity = make_identity(29, 1, 0)
        trace_path = self.work_dir / "dispatch-trace.jsonl"
        daemon = Daemon(
            self,
            "pinned-dispatch",
            identity,
            exit_on_handshake=False,
            run_timeout_ms=self.dispatch_run_timeout_ms,
            config=self.dispatch_gem5_config,
            extra_args=("--dispatch-trace-path", str(trace_path)),
        )
        daemon.launch()
        argv = self.client_argv(
            daemon.endpoint,
            identity,
            timeout_ms=self.dispatch_run_timeout_ms,
            extra=("--dispatch-fixture", DISPATCH_FIXTURE),
        )
        result, audit_counts = self.run_client_argv_audited(
            argv,
            daemon,
            timeout=self.dispatch_process_timeout,
            retained_name="pinned-dispatch",
        )
        payload, evidence = self.validate_dispatch_success(
            result, identity, daemon, authority
        )

        daemon_wait = max(
            self.dispatch_process_timeout,
            self.dispatch_run_timeout_ms / 1000.0 + self.start_wait_seconds,
        )
        returncode = daemon.wait(timeout=daemon_wait)
        daemon_log = read_text(daemon.log_path)
        stats_path = daemon.out_dir / "stats.txt"
        require(returncode == 0,
                f"dispatch daemon exited rc={returncode}:\n"
                f"{tail(daemon.log_path)}")
        exit_tick = validate_dispatch_exit_log(
            daemon_log, stats_path, evidence.signal_completion_tick
        )
        validate_process_audit(ProcessAudit(
            maps="", open_paths=(), log=daemon_log
        ))

        require(not daemon.endpoint.exists(),
                "dispatch daemon retained its endpoint after clean exit")
        try:
            lock_metadata = daemon.lock_path.lstat()
        except OSError as error:
            raise CheckFailure(
                f"dispatch endpoint lock is missing: {daemon.lock_path}: "
                f"{error}"
            ) from error
        require(stat.S_ISREG(lock_metadata.st_mode) and
                lock_metadata.st_uid == os.geteuid() and
                stat.S_IMODE(lock_metadata.st_mode) == 0o600,
                "dispatch endpoint lock is not a same-euid 0600 regular file")

        validate_evidence_file(
            trace_path, "dispatch trace", exact_mode=0o600
        )
        validate_evidence_file(stats_path, "dispatch gem5 stats")
        records = load_dispatch_trace(trace_path)
        materialized_hash = require_sha256(
            records[0].get("materialized_aql_sha256"),
            "dispatch trace materialized_aql_sha256",
        )
        evidence = dataclasses.replace(
            evidence, materialized_aql_sha256=materialized_hash
        )
        trace_summary = validate_dispatch_trace(
            records, evidence, authority
        )
        stat_values = validate_dispatch_stats(
            parse_gem5_stats(stats_path), DISPATCH_STAT_NAMES
        )

        dispatch_result = payload["dispatch"]
        self.add_check(
            "pinned_gfx950_dispatch",
            fixture=dispatch_result["fixture"],
            request_id=f"0x{evidence.request_id:016x}",
            trace_id=f"0x{evidence.trace_id:016x}",
            gpu_task_id=f"0x{trace_summary['gpu_task_id']:016x}",
            packet_crc32c=f"0x{evidence.packet_crc32c:08x}",
            output_crc32c=f"0x{evidence.output_crc32c:08x}",
            admission_tick=evidence.admission_tick,
            start_tick=evidence.start_tick,
            end_tick=evidence.end_tick,
            retire_tick=evidence.retire_tick,
            signal_completion_tick=evidence.signal_completion_tick,
            daemon_exit_tick=exit_tick,
            first_wait_status=dispatch_result["first_wait"]["status_name"],
            signal_first_wait_status=(
                dispatch_result["signal"]["armed_wait_status_name"]
            ),
            dispatch_requests=stat_values["dispatches_admitted"],
            retired_instructions=stat_values["retired_instructions"],
            global_store_instructions=(
                stat_values["global_store_instructions"]
            ),
            global_store_bytes=stat_values["global_store_bytes"],
            daemon_audit_samples=audit_counts.daemon_samples,
            runtime_audit_samples=audit_counts.runtime_samples,
            dispatch_trace=str(trace_path),
            gem5_stats=str(stats_path),
            gem5_log=str(daemon.log_path),
        )

    def check_busy(self) -> None:
        identity = make_identity(3, 1, 0)
        daemon = Daemon(self, "busy", identity, exit_on_handshake=False)
        daemon.launch()
        holder_stdout = self.work_dir / "busy.holder.stdout"
        holder_stderr = self.work_dir / "busy.holder.stderr"
        holder_argv = self.client_argv(
            daemon.endpoint,
            identity,
            extra=("--hold-ms", str(self.hold_ms)),
        )
        with holder_stdout.open("wb", buffering=0) as stdout_stream, \
                holder_stderr.open("wb", buffering=0) as stderr_stream:
            holder = subprocess.Popen(
                holder_argv,
                cwd=ROOT,
                env=self.environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                close_fds=True,
                start_new_session=True,
            )
        try:
            def holder_ready() -> bool:
                if holder.poll() is not None:
                    raise CheckFailure(
                        f"BUSY holder exited early rc={holder.returncode}: "
                        f"{read_text(holder_stderr)}"
                    )
                try:
                    parse_last_json(read_text(holder_stdout), "BUSY holder stdout")
                    return True
                except CheckFailure:
                    return False

            wait_until(holder_ready, self.process_timeout, "established BUSY holder")
            holder_payload = parse_last_json(read_text(holder_stdout),
                                             "BUSY holder stdout")
            require(holder_payload.get("peer_pid") == daemon.process.pid,
                    f"BUSY holder reached the wrong daemon: {holder_payload}")
            busy = self.expect_failure(
                self.exact_client(daemon), 18, "busy", 7
            )
            require(holder.wait(timeout=self.process_timeout) == 0,
                    f"BUSY holder failed: {read_text(holder_stderr)}")
            wait_until(
                lambda: "status=BUSY" in read_text(daemon.log_path),
                self.process_timeout,
                "daemon BUSY log",
            )
        finally:
            terminate_process(holder)
            daemon.stop()
        self.add_check(
            "busy",
            holder_peer_pid=holder_payload["peer_pid"],
            wire_status=busy["wire_status"],
        )

    def check_raw_golden_consistency(self) -> None:
        self.wire.assert_canonical_golden()
        self.add_check(
            "raw_dynamic_hello_matches_golden",
            protocol_schema=self.wire.document["schema"],
            frame_bytes=len(bytes.fromhex(
                self.wire.document["golden"]["hello_success_request"]["frame_hex"]
            )),
        )

    def check_raw_correlatable_malformed(self) -> None:
        identity = make_identity(30, 1, 0)
        daemon = Daemon(
            self, "raw-malformed", identity,
            exit_on_handshake=True, maximum_record=32768,
        )
        daemon.launch()
        with self.raw_client(daemon) as client:
            malformed = self.raw_round_trip(
                client,
                daemon,
                tag="correlatable-malformed-role",
                expected_status=self.wire.statuses["MALFORMED"],
                role=2,
            )
            client.expect_closed()
        with self.raw_client(daemon) as client:
            success = self.raw_round_trip(
                client,
                daemon,
                tag="after-correlatable-malformed",
                expected_status=self.wire.statuses["OK"],
            )
        daemon.expect_success_exit()
        self.add_check(
            "raw_correlatable_malformed_ack",
            malformed_status=malformed.status,
            recovery_connection_id=hex(success.connection_id),
        )

    def check_raw_control_and_fd_cleanup(self) -> None:
        identity = make_identity(31, 1, 0)
        daemon = Daemon(
            self, "raw-control", identity,
            exit_on_handshake=True, maximum_record=32768,
        )
        daemon.launch()
        assert daemon.process is not None
        baseline = process_fd_count(daemon.process.pid)

        def hello(tag: str) -> bytes:
            request_id, nonce = self.raw_material(tag)
            return self.wire.encode_hello(
                identity,
                request_id=request_id,
                client_nonce=nonce,
                receive_maximum=daemon.maximum_record,
            )

        def reject_control(
            tag: str,
            record: bytes,
            descriptor_count: int | None,
        ) -> None:
            with self.raw_client(daemon) as client:
                if descriptor_count is None:
                    client.send(record)
                else:
                    client.send_rights(record, descriptor_count)
                client.expect_closed()
            self.wait_for_daemon_fd_count(
                daemon, baseline, f"{tag} descriptor baseline recovery"
            )
            self.trace_raw(
                "fd_baseline_restored", daemon=daemon.name,
                tag=tag, descriptors=baseline,
            )

        reject_control("scm_rights", hello("scm-rights"), 1)
        reject_control("zero_length", b"", None)
        reject_control("zero_length_scm_rights", b"", 1)
        reject_control("control_truncation", hello("control-truncation"), 64)
        require(daemon.process.poll() is None,
                f"raw control rejection terminated daemon:\n{tail(daemon.log_path)}")
        with self.raw_client(daemon) as client:
            success = self.raw_round_trip(
                client,
                daemon,
                tag="after-control-rejections",
                expected_status=self.wire.statuses["OK"],
            )
        daemon.expect_success_exit()
        self.add_check(
            "raw_control_rejection_fd_cleanup",
            cases=[
                "scm_rights", "zero_length", "zero_length_scm_rights",
                "control_truncation",
            ],
            daemon_fd_baseline=baseline,
            recovery_connection_id=hex(success.connection_id),
        )

    def check_raw_handshake_deadline(self) -> None:
        identity = make_identity(32, 1, 0)
        handshake_timeout_ms = 250
        daemon = Daemon(
            self, "raw-deadline", identity,
            exit_on_handshake=False,
            handshake_timeout_ms=handshake_timeout_ms,
        )
        daemon.launch()
        assert daemon.process is not None
        baseline = process_fd_count(daemon.process.pid)
        started = time.monotonic()
        with self.raw_client(daemon) as client:
            client.expect_closed()
        elapsed = time.monotonic() - started
        require(elapsed < self.process_timeout,
                f"stalled raw peer closure exceeded deadline: {elapsed:.3f}s")
        self.wait_for_daemon_fd_count(
            daemon, baseline, "stalled peer descriptor baseline recovery"
        )
        wait_until(
            lambda: "host-gpu-timeout" in read_text(daemon.log_path),
            self.process_timeout,
            "stalled peer timeout log",
        )
        require(daemon.process.poll() is None,
                f"stalled peer deadline terminated daemon:\n{tail(daemon.log_path)}")
        daemon.stop()
        self.add_check(
            "raw_stalled_peer_deadline",
            configured_timeout_ms=handshake_timeout_ms,
            observed_seconds=round(elapsed, 3),
            daemon_fd_baseline=baseline,
        )

    def check_raw_protocol_state(self) -> None:
        identity = make_identity(33, 1, 0)
        daemon = Daemon(
            self, "raw-protocol-state", identity, exit_on_handshake=False
        )
        daemon.launch()
        with self.raw_client(daemon) as client:
            success = self.raw_round_trip(
                client,
                daemon,
                tag="protocol-state-first",
                expected_status=self.wire.statuses["OK"],
            )
            rejected = self.raw_round_trip(
                client,
                daemon,
                tag="protocol-state-second-unsupported-version",
                expected_status=self.wire.statuses["PROTOCOL_STATE"],
                minimum_version=(2, 0),
                maximum_version=(2, 0),
            )
            client.expect_closed()
        require(daemon.process is not None and daemon.process.poll() is None,
                f"PROTOCOL_STATE closed daemon instead of client:\n"
                f"{tail(daemon.log_path)}")
        daemon.stop()

        malformed_identity = make_identity(35, 1, 0)
        malformed_daemon = Daemon(
            self, "raw-established-malformed", malformed_identity,
            exit_on_handshake=False,
        )
        malformed_daemon.launch()
        with self.raw_client(malformed_daemon) as client:
            malformed_success = self.raw_round_trip(
                client,
                malformed_daemon,
                tag="established-malformed-first",
                expected_status=self.wire.statuses["OK"],
            )
            malformed = self.raw_round_trip(
                client,
                malformed_daemon,
                tag="established-malformed-second-role",
                expected_status=self.wire.statuses["MALFORMED"],
                role=2,
            )
            client.expect_closed()
        require(
            malformed_daemon.process is not None and
            malformed_daemon.process.poll() is None,
            f"established MALFORMED closed daemon instead of client:\n"
            f"{tail(malformed_daemon.log_path)}",
        )
        malformed_daemon.stop()
        self.add_check(
            "raw_later_state_precedence",
            first_connection_id=hex(success.connection_id),
            second_status=rejected.status,
            malformed_first_connection_id=hex(
                malformed_success.connection_id
            ),
            malformed_second_status=malformed.status,
            precedence_probes=[
                "structurally-valid-unsupported-version",
                "structurally-malformed-role",
            ],
        )

    def check_raw_resource_exhausted(self) -> None:
        identity = make_identity(34, 1, 0)
        daemon = Daemon(
            self,
            "raw-resource-exhausted",
            identity,
            exit_on_handshake=True,
            handshake_timeout_ms=max(
                10000, self.handshake_timeout_ms,
                int(self.process_timeout * 4000),
            ),
            run_timeout_ms=max(
                30000, self.run_timeout_ms,
                int(self.process_timeout * 8000),
            ),
        )
        daemon.launch()
        assert daemon.process is not None
        baseline = process_fd_count(daemon.process.pid)
        baseline_connections = process_connected_unix_socket_count(
            daemon.process.pid
        )

        def open_stalled() -> list[RawSeqpacketClient]:
            stalled: list[RawSeqpacketClient] = []
            try:
                for _ in range(8):
                    client = self.raw_client(daemon)
                    client.__enter__()
                    stalled.append(client)
                wait_until(
                    lambda: process_connected_unix_socket_count(
                        daemon.process.pid
                    ) >= baseline_connections + 8,
                    self.process_timeout,
                    "eight stalled raw clients to be accepted",
                )
                return stalled
            except Exception:
                for client in reversed(stalled):
                    client.__exit__(None, None, None)
                raise

        startup_stalled = open_stalled()
        try:
            with self.raw_client(daemon) as overflow:
                startup_rejected = self.raw_round_trip(
                    overflow,
                    daemon,
                    tag="startup-ninth-client-resource-exhausted",
                    expected_status=self.wire.statuses["RESOURCE_EXHAUSTED"],
                )
                overflow.expect_closed()
        finally:
            for client in reversed(startup_stalled):
                client.__exit__(None, None, None)

        self.wait_for_daemon_fd_count(
            daemon, baseline, "startup resource descriptor baseline recovery"
        )

        steady_stalled = open_stalled()
        try:
            with self.raw_client(daemon) as overflow:
                wait_until(
                    lambda: process_connected_unix_socket_count(
                        daemon.process.pid
                    ) >= baseline_connections + 9,
                    self.process_timeout,
                    "delayed overflow client to be accepted",
                )
                overflow.expect_open_and_idle(0.1)
                steady_rejected = self.raw_round_trip(
                    overflow,
                    daemon,
                    tag="steady-delayed-ninth-client-resource-exhausted",
                    expected_status=self.wire.statuses["RESOURCE_EXHAUSTED"],
                )
                overflow.expect_closed()
        finally:
            for client in reversed(steady_stalled):
                client.__exit__(None, None, None)

        self.wait_for_daemon_fd_count(
            daemon, baseline, "steady resource descriptor baseline recovery"
        )
        with self.raw_client(daemon) as client:
            success = self.raw_round_trip(
                client,
                daemon,
                tag="after-resource-exhaustion",
                expected_status=self.wire.statuses["OK"],
            )
        daemon.expect_success_exit()
        self.add_check(
            "raw_ninth_client_resource_exhausted",
            stalled_clients=8,
            startup_overflow_status=startup_rejected.status,
            delayed_overflow_status=steady_rejected.status,
            delayed_hello_idle_seconds=0.1,
            daemon_fd_baseline=baseline,
            daemon_connection_baseline=baseline_connections,
            recovery_connection_id=hex(success.connection_id),
        )

    def run_parallel_clients(
        self, daemons: list[Daemon], operation
    ) -> list[Any]:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(daemons), thread_name_prefix="cp4-client"
        ) as executor:
            futures = [executor.submit(operation, daemon) for daemon in daemons]
            return [future.result() for future in futures]

    def check_world_isolation(self, world_size: int) -> None:
        tag = 100 + world_size
        daemons = [
            Daemon(
                self,
                f"world-{world_size}-rank-{rank}",
                make_identity(tag, world_size, rank),
                exit_on_handshake=True,
            )
            for rank in range(world_size)
        ]
        for daemon in daemons:
            daemon.launch(wait_ready=False)
        for daemon in daemons:
            daemon.wait_ready()
        require(len({daemon.endpoint for daemon in daemons}) == world_size,
                f"world {world_size} reused an endpoint")
        require(len({daemon.identity.daemon_uuid for daemon in daemons}) == world_size,
                f"world {world_size} reused a daemon UUID")
        require(len({daemon.process.pid for daemon in daemons}) == world_size,
                f"world {world_size} reused a daemon PID")

        def wrong_instance(daemon: Daemon) -> dict[str, Any]:
            wrong_uuid = make_identity(tag + 1000, world_size,
                                       daemon.identity.rank).daemon_uuid
            return self.expect_failure(
                self.exact_client(daemon, daemon_uuid=wrong_uuid),
                7, "instance mismatch", 4,
            )

        def wrong_topology(daemon: Daemon) -> dict[str, Any]:
            wrong_job = make_identity(tag + 1000, world_size,
                                      daemon.identity.rank).job_uuid
            return self.expect_failure(
                self.exact_client(daemon, job_uuid=wrong_job),
                8, "topology mismatch", 5,
            )

        self.run_parallel_clients(daemons, wrong_instance)
        self.run_parallel_clients(daemons, wrong_topology)

        def success(daemon: Daemon) -> dict[str, Any]:
            return self.validate_success(self.exact_client(daemon),
                                         daemon.identity, daemon)

        payloads = self.run_parallel_clients(daemons, success)
        for daemon in daemons:
            daemon.expect_success_exit()
        require(
            {payload["peer_pid"] for payload in payloads} ==
            {daemon.process.pid for daemon in daemons},
            f"world {world_size} peer PID routing crossed daemon boundaries",
        )
        require(
            {(payload["rank"], payload["daemon_uuid"]) for payload in payloads} ==
            {(daemon.identity.rank, daemon.identity.daemon_uuid)
             for daemon in daemons},
            f"world {world_size} rank identity routing crossed boundaries",
        )
        self.add_check(
            f"world_size_{world_size}_isolation",
            daemon_count=world_size,
            ranks=list(range(world_size)),
            peer_pids=sorted(payload["peer_pid"] for payload in payloads),
        )

    def check_runtime_timeout(self) -> None:
        identity = make_identity(4, 1, 0)
        daemon = Daemon(self, "runtime-timeout", identity,
                        exit_on_handshake=False)
        daemon.launch()
        assert daemon.process is not None
        os.killpg(daemon.process.pid, signal.SIGSTOP)
        try:
            timeout = self.expect_failure(
                self.run_client_argv(
                    self.client_argv(
                        daemon.endpoint, identity,
                        timeout_ms=min(200, self.client_timeout_ms),
                    ),
                    timeout=self.process_timeout,
                ),
                11, "timed out", -1,
            )
        finally:
            daemon.kill()
        self.add_check("runtime_timeout", wire_status=timeout["wire_status"])

    def check_eof_and_stale_recovery(self) -> None:
        original_identity = make_identity(5, 1, 0)
        original = Daemon(self, "eof-original", original_identity,
                          exit_on_handshake=False)
        original.launch()
        assert original.process is not None
        os.killpg(original.process.pid, signal.SIGSTOP)
        argv = self.client_argv(
            original.endpoint, original_identity,
            timeout_ms=max(2000, self.client_timeout_ms),
        )
        client = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=self.environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            close_fds=True,
            start_new_session=True,
        )
        try:
            wait_until(
                lambda: process_has_connected_unix_socket(client.pid),
                self.process_timeout,
                "runtime CLI connected seqpacket before EOF",
            )
            original.kill()
            stdout, stderr = client.communicate(timeout=self.process_timeout)
        except Exception:
            terminate_process(client)
            original.kill()
            raise
        eof = self.expect_failure(
            ClientResult(argv, client.returncode, stdout, stderr),
            13, "connection lost", -1,
        )
        require(original.endpoint.exists() and
                stat.S_ISSOCK(original.endpoint.lstat().st_mode),
                "SIGKILL did not leave the expected stale socket")

        replacement_identity = make_identity(6, 1, 0)
        replacement = Daemon(
            self,
            "eof-replacement",
            replacement_identity,
            exit_on_handshake=True,
            endpoint=original.endpoint,
        )
        replacement.launch()
        payload = self.validate_success(
            self.exact_client(replacement), replacement_identity, replacement
        )
        replacement.expect_success_exit()
        require(not replacement.endpoint.exists(),
                "clean replacement shutdown retained its socket")
        require(replacement.lock_path.exists() and
                stat.S_IMODE(replacement.lock_path.stat().st_mode) == 0o600,
                "endpoint lock was not retained as a 0600 regular file")
        self.add_check(
            "eof_and_stale_recovery",
            eof_status=eof["status_name"],
            original_daemon_uuid=original_identity.daemon_uuid,
            replacement_daemon_uuid=payload["daemon_uuid"],
            replacement_peer_pid=payload["peer_pid"],
        )

    def check_live_endpoint_preserved(self) -> None:
        endpoint = self.work_dir / "external-live.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        listener.settimeout(self.process_timeout)
        listener.bind(str(endpoint))
        listener.listen(2)
        before = endpoint.lstat()
        identity = make_identity(7, 1, 0)
        daemon = Daemon(self, "live-endpoint", identity,
                        exit_on_handshake=True, endpoint=endpoint)
        try:
            daemon.launch(wait_ready=False)
            returncode = daemon.wait()
            require(returncode != 0,
                    "gem5 replaced an already-live endpoint")
            log = read_text(daemon.log_path)
            require("existing endpoint is live or cannot be proven stale" in log,
                    f"gem5 did not diagnose the live endpoint:\n{tail(daemon.log_path)}")
            after = endpoint.lstat()
            require((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
                    "gem5 changed or unlinked the live endpoint")
            accepted, _ = listener.accept()
            accepted.close()
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            try:
                probe.connect(str(endpoint))
                accepted, _ = listener.accept()
                accepted.close()
            finally:
                probe.close()
        finally:
            listener.close()
            try:
                endpoint.unlink()
            except FileNotFoundError:
                pass
        self.add_check("live_endpoint_preserved", inode=before.st_ino)

    def check_endpoint_security_rejections(self) -> None:
        observations: list[str] = []

        unsafe_parent = self.work_dir / "unsafe-parent"
        unsafe_parent.mkdir(mode=0o755)
        os.chmod(unsafe_parent, 0o755)
        unsafe_endpoint = unsafe_parent / "bridge.sock"
        daemon = Daemon(
            self, "unsafe-parent", make_identity(20, 1, 0),
            exit_on_handshake=True, endpoint=unsafe_endpoint,
        )
        daemon.launch(wait_ready=False)
        require(daemon.wait() != 0, "0755 endpoint parent was accepted")
        require("endpoint parent must be owned by euid" in read_text(daemon.log_path),
                f"0755 parent failure was not diagnostic:\n{tail(daemon.log_path)}")
        require(not unsafe_endpoint.exists(),
                "0755 endpoint parent acquired a socket")
        observations.append("parent_mode")

        real_parent = self.work_dir / "real-parent"
        real_parent.mkdir(mode=0o700)
        linked_parent = self.work_dir / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        linked_endpoint = linked_parent / "bridge.sock"
        daemon = Daemon(
            self, "symlink-parent", make_identity(21, 1, 0),
            exit_on_handshake=True, endpoint=linked_endpoint,
        )
        daemon.launch(wait_ready=False)
        require(daemon.wait() != 0, "symlink endpoint parent was accepted")
        require("cannot securely open endpoint parent" in read_text(daemon.log_path),
                f"symlink parent failure was not diagnostic:\n{tail(daemon.log_path)}")
        require(linked_parent.is_symlink() and not (real_parent / "bridge.sock").exists(),
                "symlink parent or target was altered")
        observations.append("parent_symlink")

        regular_endpoint = self.work_dir / "regular-endpoint.sock"
        regular_bytes = b"cp4-regular-endpoint\n"
        regular_endpoint.write_bytes(regular_bytes)
        os.chmod(regular_endpoint, 0o600)
        daemon = Daemon(
            self, "regular-endpoint", make_identity(22, 1, 0),
            exit_on_handshake=True, endpoint=regular_endpoint,
        )
        daemon.launch(wait_ready=False)
        require(daemon.wait() != 0, "regular endpoint was replaced")
        require("existing endpoint is not a socket" in read_text(daemon.log_path),
                f"regular endpoint failure was not diagnostic:\n{tail(daemon.log_path)}")
        require(regular_endpoint.read_bytes() == regular_bytes,
                "regular endpoint contents changed")
        observations.append("endpoint_regular")

        symlink_target = self.work_dir / "endpoint-symlink-target"
        symlink_target_bytes = b"cp4-endpoint-symlink\n"
        symlink_target.write_bytes(symlink_target_bytes)
        symlink_endpoint = self.work_dir / "symlink-endpoint.sock"
        symlink_endpoint.symlink_to(symlink_target.name)
        daemon = Daemon(
            self, "symlink-endpoint", make_identity(23, 1, 0),
            exit_on_handshake=True, endpoint=symlink_endpoint,
        )
        daemon.launch(wait_ready=False)
        require(daemon.wait() != 0, "symlink endpoint was replaced")
        require("existing endpoint is not a socket" in read_text(daemon.log_path),
                f"symlink endpoint failure was not diagnostic:\n{tail(daemon.log_path)}")
        require(symlink_endpoint.is_symlink() and
                symlink_target.read_bytes() == symlink_target_bytes,
                "symlink endpoint or target changed")
        observations.append("endpoint_symlink")

        lock_symlink_endpoint = self.work_dir / "lock-symlink.sock"
        lock_target = self.work_dir / "lock-symlink-target"
        lock_target.write_bytes(b"cp4-lock-target\n")
        os.chmod(lock_target, 0o600)
        lock_symlink = Path(f"{lock_symlink_endpoint}.lock")
        lock_symlink.symlink_to(lock_target.name)
        daemon = Daemon(
            self, "lock-symlink", make_identity(24, 1, 0),
            exit_on_handshake=True, endpoint=lock_symlink_endpoint,
        )
        daemon.launch(wait_ready=False)
        require(daemon.wait() != 0, "symlink lock was accepted")
        require("cannot open endpoint lock" in read_text(daemon.log_path),
                f"symlink lock failure was not diagnostic:\n{tail(daemon.log_path)}")
        require(lock_symlink.is_symlink(), "symlink lock was altered")
        observations.append("lock_symlink")

        weak_lock_endpoint = self.work_dir / "weak-lock.sock"
        weak_lock = Path(f"{weak_lock_endpoint}.lock")
        weak_lock.write_bytes(b"")
        os.chmod(weak_lock, 0o644)
        daemon = Daemon(
            self, "weak-lock", make_identity(25, 1, 0),
            exit_on_handshake=True, endpoint=weak_lock_endpoint,
        )
        daemon.launch(wait_ready=False)
        require(daemon.wait() != 0, "non-0600 lock was accepted")
        require("endpoint lock must be a same-euid 0600 regular file" in
                read_text(daemon.log_path),
                f"weak lock failure was not diagnostic:\n{tail(daemon.log_path)}")
        require(stat.S_IMODE(weak_lock.stat().st_mode) == 0o644,
                "weak lock mode was changed")
        observations.append("lock_mode")

        self.add_check("endpoint_security_rejections", cases=observations)

    def check_competing_daemon_lock(self) -> None:
        endpoint = self.work_dir / "competing.sock"
        identity = make_identity(26, 1, 0)
        primary = Daemon(
            self, "competing-primary", identity,
            exit_on_handshake=True, endpoint=endpoint,
        )
        primary.launch()
        before = endpoint.lstat()
        competitor = Daemon(
            self, "competing-secondary", make_identity(27, 1, 0),
            exit_on_handshake=True, endpoint=endpoint,
        )
        competitor.launch(wait_ready=False)
        require(competitor.wait() != 0,
                "second gem5 acquired an already-held endpoint lock")
        require("endpoint lock is held by another process" in
                read_text(competitor.log_path),
                f"lock contention failure was not diagnostic:\n"
                f"{tail(competitor.log_path)}")
        require(primary.process is not None and primary.process.poll() is None,
                "competing daemon terminated the primary")
        after = endpoint.lstat()
        require((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
                "competing daemon replaced the primary endpoint")
        payload = self.validate_success(self.exact_client(primary), identity, primary)
        primary.expect_success_exit()
        self.add_check(
            "competing_daemon_lock",
            primary_peer_pid=payload["peer_pid"],
            competitor_returncode=competitor.process.returncode,
        )

    def check_shutdown_replacement_preserved(self) -> None:
        endpoint = self.work_dir / "shutdown-replacement.sock"
        identity = make_identity(28, 1, 0)
        daemon = Daemon(
            self,
            "shutdown-replacement",
            identity,
            exit_on_handshake=False,
            endpoint=endpoint,
            run_timeout_ms=1500,
        )
        daemon.launch()
        rejection = self.exact_client(
            daemon, extra=("--min-version", "2.0", "--max-version", "2.0")
        )
        self.expect_failure(rejection, 4, "version mismatch", 2)
        require(daemon.process is not None and daemon.process.poll() is None,
                "daemon exited before endpoint replacement")

        endpoint.unlink()
        replacement = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        replacement.settimeout(self.process_timeout)
        replacement.bind(str(endpoint))
        replacement.listen(1)
        replacement_stat = endpoint.lstat()
        try:
            require(daemon.wait(timeout=self.process_timeout) == 0,
                    f"daemon did not reach bounded config exit:\n{tail(daemon.log_path)}")
            current = endpoint.lstat()
            require(
                (current.st_dev, current.st_ino) ==
                (replacement_stat.st_dev, replacement_stat.st_ino),
                "daemon shutdown unlinked a same-UID replacement socket",
            )
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            try:
                probe.connect(str(endpoint))
                accepted, _ = replacement.accept()
                accepted.close()
            finally:
                probe.close()
        finally:
            replacement.close()
            try:
                endpoint.unlink()
            except FileNotFoundError:
                pass
        self.add_check(
            "shutdown_replacement_preserved",
            replacement_inode=replacement_stat.st_ino,
        )

    def check_listener_disabled(self) -> None:
        identity = make_identity(8, 1, 0)
        daemon = Daemon(self, "listener-disabled", identity,
                        exit_on_handshake=True)
        daemon.launch(wait_ready=False, listener_mode="off")
        returncode = daemon.wait()
        require(returncode != 0,
                "HostGPUBridge ran with gem5 listeners disabled")
        require("host GPU bridge requires --listener-mode=on" in
                read_text(daemon.log_path),
                f"listener-disabled failure was not diagnostic:\n{tail(daemon.log_path)}")
        require(not daemon.endpoint.exists(),
                "listener-disabled initialization created an endpoint")
        self.add_check("listener_disabled_rejected", returncode=returncode)

    def run(self) -> None:
        self.check_raw_golden_consistency()
        self.check_listener_disabled()
        self.check_endpoint_security_rejections()
        self.check_competing_daemon_lock()
        self.check_shutdown_replacement_preserved()
        self.check_raw_correlatable_malformed()
        self.check_raw_control_and_fd_cleanup()
        self.check_raw_handshake_deadline()
        self.check_raw_protocol_state()
        self.check_raw_resource_exhausted()
        self.check_reject_then_success()
        self.check_queue_control()
        self.check_memory_transfer()
        self.check_signal_lifecycle()
        self.check_pinned_dispatch()
        self.check_busy()
        for world_size in WORLD_SIZES:
            self.check_world_isolation(world_size)
        self.check_runtime_timeout()
        self.check_eof_and_stale_recovery()
        self.check_live_endpoint_preserved()

    def cleanup_processes(self) -> list[str]:
        errors: list[str] = []
        for daemon in reversed(self.daemons):
            try:
                daemon.stop()
            except (OSError, subprocess.SubprocessError) as error:
                errors.append(f"{daemon.name}: {type(error).__name__}: {error}")
        return errors


def executable_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise argparse.ArgumentTypeError(f"not an executable file: {value}")
    return path


def file_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {value}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real CP-0004 through CP-0008 runtime CLI against "
            "HostGPUBridge processes. This validates handshake, bounded "
            "queue-control events, bridge-owned simulated-memory byte "
            "transfer, signal/event completion, and one source-pinned "
            "gfx950 dispatch through gem5 GPU execution."
        )
    )
    parser.add_argument("--gem5", required=True, type=executable_path)
    parser.add_argument("--runtime-cli", required=True, type=executable_path)
    parser.add_argument("--gem5-config", type=file_path,
                        default=DEFAULT_CONFIG)
    parser.add_argument("--dispatch-gem5-config", type=file_path,
                        default=DEFAULT_DISPATCH_CONFIG)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help=(
            "Compatibility flag; CP-0008 acceptance evidence now retains "
            "every run directory"
        ),
    )
    parser.add_argument("--start-wait-seconds", type=float, default=15.0)
    parser.add_argument("--process-timeout-seconds", type=float, default=8.0)
    parser.add_argument(
        "--dispatch-process-timeout-seconds", type=float, default=90.0,
        help="Wall-clock bound for the one real pinned-dispatch client",
    )
    parser.add_argument("--client-timeout-ms", type=int, default=1500)
    parser.add_argument("--server-startup-timeout-ms", type=int, default=20000)
    parser.add_argument("--server-handshake-timeout-ms", type=int, default=2000)
    parser.add_argument("--server-run-timeout-ms", type=int, default=20000)
    parser.add_argument(
        "--dispatch-server-run-timeout-ms", type=int, default=60000,
        help="Wall-clock service bound for the dispatch-specific gem5 config",
    )
    parser.add_argument("--hold-ms", type=int, default=1500)
    args = parser.parse_args()
    for name in (
        "start_wait_seconds", "process_timeout_seconds",
        "dispatch_process_timeout_seconds", "client_timeout_ms",
        "server_startup_timeout_ms", "server_handshake_timeout_ms",
        "server_run_timeout_ms", "dispatch_server_run_timeout_ms", "hold_ms",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.work_root is not None:
        args.work_root = args.work_root.expanduser().resolve()
        if not args.work_root.is_dir():
            parser.error(f"--work-root is not a directory: {args.work_root}")
    for option, path in (
        ("--gem5-config", args.gem5_config),
        ("--dispatch-gem5-config", args.dispatch_gem5_config),
    ):
        if not path.is_file():
            parser.error(f"{option} is not a file: {path}")
    return args


def main() -> int:
    args = parse_args()
    soft_fds, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_fds != resource.RLIM_INFINITY and soft_fds < 64:
        print(json.dumps({
            "schema": "amdgpu-sim.host-transport.integration.v1",
            "status": "failed",
            "error": f"RLIMIT_NOFILE {soft_fds} is below the required 64",
        }, sort_keys=True))
        return 1

    work_dir = Path(tempfile.mkdtemp(
        prefix="cp8-host-",
        dir=str(args.work_root) if args.work_root is not None else None,
    )).resolve()
    os.chmod(work_dir, 0o700)
    harness = Harness(args, work_dir)
    started = time.monotonic()
    status = "failed"
    error_message: str | None = None
    try:
        harness.run()
        status = "passed"
    except KeyboardInterrupt:
        error_message = "KeyboardInterrupt: integration run interrupted"
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
    finally:
        cleanup_errors = harness.cleanup_processes()
    if cleanup_errors:
        cleanup_message = "; ".join(cleanup_errors)
        if error_message is None:
            error_message = f"process cleanup failed: {cleanup_message}"
            status = "failed"
        else:
            error_message += f"; process cleanup failed: {cleanup_message}"

    # CP-0008 trace, stats, process logs, and raw-wire evidence are acceptance
    # provenance, so successful and failed run directories are both retained.
    retain = True
    summary: dict[str, Any] = {
        "schema": "amdgpu-sim.host-transport.integration.v1",
        "status": status,
        "scope": (
            "CP-0004 handshake, CP-0005 bounded queue control, CP-0006 "
            "bridge-owned simulated allocation and byte transfer, and "
            "CP-0007 signal/event completion, plus one CP-0008 source-pinned "
            "gfx950 AQL dispatch through gem5 GPU execution"
        ),
        "checks": harness.checks,
        "world_sizes": list(WORLD_SIZES),
        "resource_bounds": {
            "maximum_parallel_gem5_daemons": max(WORLD_SIZES),
            "maximum_clients_per_daemon_phase": 9,
            "runtime_cli_process_retries": 0,
            "wire_request_retries": 0,
            "dispatch_wait_retries_without_send": 1,
            "raw_client_retries": 0,
        },
        "duration_seconds": round(time.monotonic() - started, 3),
        "work_dir": str(work_dir),
        "work_dir_retained": retain,
        "retained_work_dir": str(work_dir),
    }
    if error_message is not None:
        summary["error"] = error_message
    print(json.dumps(summary, indent=2, sort_keys=True))
    if status != "passed":
        print(f"integration logs retained at {work_dir}", file=sys.stderr)
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
