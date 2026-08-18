#!/usr/bin/env python3
"""Decode a gem5 bridge wire tape into readable JSONL.

The dispatch trace answers "which dispatch failed"; it stores descriptors and
never bytes, so it cannot rebuild an execution. The bridge tape written by
``HostGPUBridge`` with ``--bridge-tape-path`` stores the exact bytes of every
record the bridge accepted and every record it transmitted, so a failing
dispatch can be located and, later, replayed without repeating a multi-hour
model run.

This decoder is deliberately transport-level. It parses only the frozen
80-byte transport header plus the two fixed operation selectors that the
versioned contract defines (KMT operation id, generic dispatch opcode). It
never inspects kernel names, shapes, code hashes, or program counters, so it
stays valid for any kernel and any model.

Usage::

    tools/bridge_tape_decode.py <tape> [--limit N] [--payload-bytes N]
    tools/bridge_tape_decode.py <tape> --summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import BinaryIO, Iterator


SCHEMA = "amdgpu-sim.bridge-tape.v1"

TAPE_MAGIC = b"GSBRTAPE"
TAPE_FILE_HEADER_BYTES = 80
TAPE_RECORD_HEADER_BYTES = 48
TAPE_FORMAT_MAJOR = 1

WIRE_HEADER_BYTES = 80
WIRE_MAGIC = bytes.fromhex("4753494d52504300")
WIRE_CRC_OFFSET = 64

DIRECTIONS = {0: "ingress", 1: "egress"}

# Frozen values of the versioned transport contract. Kept as a table so the
# decoder runs standalone on a tape copied off the machine; tests/ cross-checks
# it against protocol/ and the generated headers so it cannot drift.
MESSAGE_TYPES = {
    1: "HELLO",
    2: "HELLO_ACK",
    3: "QUEUE_REQUEST",
    4: "QUEUE_ACK",
    5: "QUEUE_COMPLETION",
    6: "MEMORY_REQUEST",
    7: "MEMORY_ACK",
    8: "SIGNAL_REQUEST",
    9: "SIGNAL_ACK",
    10: "SIGNAL_COMPLETION",
    11: "DISPATCH_REQUEST",
    12: "DISPATCH_ACK",
    13: "DISPATCH_COMPLETION",
    14: "KMT_REQUEST",
    15: "KMT_ACK",
    16: "CODE_OBJECT_REQUEST",
    17: "CODE_OBJECT_ACK",
    18: "GENERIC_DISPATCH_REQUEST",
    19: "GENERIC_DISPATCH_ACK",
    20: "GENERIC_DISPATCH_COMPLETION",
}

KMT_MESSAGE_TYPES = (14, 15)
# Payload offset of the KMT operation selector; see protocol/host-transport-v1-kmt.json.
KMT_OPERATION_OFFSET = 4
KMT_OPERATIONS = {
    1: "OPEN_KFD",
    2: "CLOSE_KFD",
    3: "GET_VERSION",
    4: "TOPOLOGY_SNAPSHOT",
    5: "ALLOC_MEMORY",
    6: "FREE_MEMORY",
    7: "COPY_MEMORY",
    8: "QUEUE_CREATE",
    9: "QUEUE_DESTROY",
    10: "QUEUE_DOORBELL",
    11: "EVENT_CREATE",
    12: "EVENT_DESTROY",
    13: "EVENT_SET",
    14: "EVENT_RESET",
    15: "EVENT_QUERY",
    16: "EVENT_WAIT",
    17: "POINTER_INFO",
    18: "MODEL_DRM_CALL",
    19: "PROCESS_APERTURES",
    20: "ACQUIRE_VM",
    21: "SET_MEMORY_POLICY",
    22: "ALLOC_MEMORY_OF_GPU",
    23: "FREE_MEMORY_OF_GPU",
    24: "MAP_MEMORY_TO_GPU",
    25: "UNMAP_MEMORY_FROM_GPU",
    26: "SET_SCRATCH_BACKING_VA",
    27: "EXPORT_BACKING",
    28: "GET_CLOCK_COUNTERS",
}


class TapeError(Exception):
    """The tape is not a readable bridge tape."""


def _crc32c_table() -> list[int]:
    table = []
    for index in range(256):
        value = index
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
        table.append(value)
    return table


# A full model tape holds tens of megabytes of records, so the per-byte loop is
# table driven rather than bit driven.
CRC32C_TABLE = _crc32c_table()


def crc32c(data: bytes) -> int:
    """CRC-32C Castagnoli, reflected polynomial 0x82f63b78."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc = CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def read_file_header(stream: BinaryIO) -> dict[str, object]:
    raw = stream.read(TAPE_FILE_HEADER_BYTES)
    if len(raw) != TAPE_FILE_HEADER_BYTES:
        raise TapeError("tape is shorter than one file header")
    if raw[:8] != TAPE_MAGIC:
        raise TapeError("tape magic mismatch")
    major = int.from_bytes(raw[8:10], "big")
    if major != TAPE_FORMAT_MAJOR:
        raise TapeError(f"unsupported tape format major {major}")
    file_header_bytes = int.from_bytes(raw[12:14], "big")
    record_header_bytes = int.from_bytes(raw[14:16], "big")
    if file_header_bytes != TAPE_FILE_HEADER_BYTES:
        raise TapeError(f"unexpected file header size {file_header_bytes}")
    if record_header_bytes != TAPE_RECORD_HEADER_BYTES:
        raise TapeError(f"unexpected record header size {record_header_bytes}")
    return {
        "schema": SCHEMA,
        "kind": "file_header",
        "format_major": major,
        "format_minor": int.from_bytes(raw[10:12], "big"),
        "record_header_bytes": record_header_bytes,
        "tick_frequency_hz": int.from_bytes(raw[16:24], "big"),
        "epoch": int.from_bytes(raw[24:32], "big"),
        "rank": int.from_bytes(raw[32:36], "big"),
        "world_size": int.from_bytes(raw[36:40], "big"),
        "maximum_record": int.from_bytes(raw[40:44], "big"),
        "daemon_uuid": raw[48:64].hex(),
        "job_uuid": raw[64:80].hex(),
    }


def decode_wire_header(record: bytes) -> dict[str, object]:
    """Decode the frozen 80-byte transport header of one wire record."""
    if len(record) < WIRE_HEADER_BYTES:
        return {"wire_header": None}
    message_type = int.from_bytes(record[14:16], "big")
    stored_crc = int.from_bytes(record[64:68], "big")
    zeroed = bytearray(record)
    zeroed[WIRE_CRC_OFFSET:WIRE_CRC_OFFSET + 4] = b"\0\0\0\0"
    header: dict[str, object] = {
        "magic_ok": record[:8] == WIRE_MAGIC,
        "framing_major": int.from_bytes(record[8:10], "big"),
        "framing_minor": int.from_bytes(record[10:12], "big"),
        "header_bytes": int.from_bytes(record[12:14], "big"),
        "message_type": message_type,
        "message": MESSAGE_TYPES.get(message_type, f"UNKNOWN_{message_type}"),
        "flags": int.from_bytes(record[16:20], "big"),
        "payload_bytes": int.from_bytes(record[20:24], "big"),
        "request_id": int.from_bytes(record[24:32], "big"),
        "daemon_instance_uuid": record[32:48].hex(),
        "connection_id": int.from_bytes(record[48:56], "big"),
        "job_epoch": int.from_bytes(record[56:64], "big"),
        "crc32c": stored_crc,
        "crc_ok": crc32c(bytes(zeroed)) == stored_crc,
    }
    if message_type in KMT_MESSAGE_TYPES:
        offset = WIRE_HEADER_BYTES + KMT_OPERATION_OFFSET
        if len(record) >= offset + 2:
            operation = int.from_bytes(record[offset:offset + 2], "big")
            header["kmt_operation"] = operation
            header["kmt_operation_name"] = KMT_OPERATIONS.get(
                operation, f"UNKNOWN_{operation}"
            )
    return header


def iter_records(
    stream: BinaryIO, payload_bytes: int
) -> Iterator[dict[str, object]]:
    """Yield one decoded record per tape entry, then a trailer."""
    sequence = 0
    truncated = False
    while True:
        header = stream.read(TAPE_RECORD_HEADER_BYTES)
        if not header:
            break
        if len(header) != TAPE_RECORD_HEADER_BYTES:
            truncated = True
            break
        length = int.from_bytes(header[8:12], "big")
        record = stream.read(length)
        if len(record) != length:
            truncated = True
            break
        sequence += 1
        direction = header[0]
        entry: dict[str, object] = {
            "schema": SCHEMA,
            "kind": "record",
            "sequence": int.from_bytes(header[24:32], "big"),
            "direction": DIRECTIONS.get(direction, f"unknown_{direction}"),
            "carrier": bool(header[1]),
            "flags": int.from_bytes(header[2:4], "big"),
            "client_fd": int.from_bytes(header[4:8], "big"),
            "generation": int.from_bytes(header[16:24], "big"),
            "sim_tick": int.from_bytes(header[32:40], "big"),
            "monotonic_ns": int.from_bytes(header[40:48], "big"),
            "record_bytes": length,
            "record_sha256": hashlib.sha256(record).hexdigest(),
        }
        entry.update(decode_wire_header(record))
        if payload_bytes > 0:
            entry["record_prefix"] = record[:payload_bytes].hex()
        yield entry
    yield {
        "schema": SCHEMA,
        "kind": "trailer",
        "records": sequence,
        "truncated": truncated,
    }


def summarize(entries: Iterator[dict[str, object]]) -> dict[str, object]:
    counts: dict[str, int] = {}
    directions: dict[str, int] = {}
    connections: dict[str, int] = {}
    total_bytes = 0
    carriers = 0
    crc_failures = 0
    first_tick = None
    last_tick = None
    trailer: dict[str, object] = {}
    for entry in entries:
        if entry["kind"] == "trailer":
            trailer = entry
            continue
        message = str(entry.get("message", "MALFORMED"))
        counts[message] = counts.get(message, 0) + 1
        direction = str(entry["direction"])
        directions[direction] = directions.get(direction, 0) + 1
        owner = f"{entry['client_fd']}:{entry['generation']}"
        connections[owner] = connections.get(owner, 0) + 1
        total_bytes += int(entry["record_bytes"])
        carriers += 1 if entry["carrier"] else 0
        if entry.get("crc_ok") is False:
            crc_failures += 1
        tick = int(entry["sim_tick"])
        first_tick = tick if first_tick is None else first_tick
        last_tick = tick
    return {
        "schema": SCHEMA,
        "kind": "summary",
        "records": trailer.get("records", 0),
        "truncated": trailer.get("truncated", False),
        "record_bytes": total_bytes,
        "carrier_records": carriers,
        "crc_failures": crc_failures,
        "first_sim_tick": first_tick,
        "last_sim_tick": last_tick,
        "by_direction": dict(sorted(directions.items())),
        "by_message": dict(sorted(counts.items())),
        "by_connection": dict(sorted(connections.items())),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tape", type=Path, help="Binary tape written by gem5")
    parser.add_argument(
        "--output", type=Path, default=None, help="JSONL output (default stdout)"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after N records; 0 is all"
    )
    parser.add_argument(
        "--payload-bytes",
        type=int,
        default=32,
        help="Hex prefix of each raw record to emit; 0 omits it",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Emit one aggregate object instead of per-record lines",
    )
    args = parser.parse_args(argv)
    if args.limit < 0 or args.payload_bytes < 0:
        parser.error("--limit and --payload-bytes must not be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with args.tape.open("rb") as stream:
            file_header = read_file_header(stream)
            entries = iter_records(stream, args.payload_bytes)
            if args.summary:
                lines = [file_header, summarize(entries)]
            else:
                lines = [file_header]
                emitted = 0
                for entry in entries:
                    lines.append(entry)
                    if entry["kind"] != "record":
                        continue
                    emitted += 1
                    if args.limit and emitted >= args.limit:
                        break
    except TapeError as error:
        print(f"bridge-tape-decode error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"bridge-tape-decode error: {error}", file=sys.stderr)
        return 2

    text = "".join(json.dumps(line, sort_keys=False) + "\n" for line in lines)
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
