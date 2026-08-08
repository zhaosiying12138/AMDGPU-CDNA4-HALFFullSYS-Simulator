#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
from pathlib import Path
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocol" / "host-transport-v1.json"

MAGIC = b"GSIMRPC\0"
HEADER_BYTES = 80
CRC_OFFSET = 64
MAX_RECORD = 65536
MAX_HANDSHAKE_RECORD = 4096
HELLO = 1
HELLO_ACK = 2
RUNTIME = 1
DAEMON = 2
OK = 0
MALFORMED = 1
UNSUPPORTED_VERSION = 2
UNSUPPORTED_CAPABILITY = 3
INSTANCE_MISMATCH = 4
TOPOLOGY_MISMATCH = 5
PROTOCOL_STATE = 9
INTERNAL = 10
KNOWN_STATUSES = set(range(OK, INTERNAL + 1))
TOPOLOGY_IDENTITY = 1
CRITICAL = 1
CAP_TOPOLOGY = 0
CAP_BYTES = 32

CLIENT_NONCE = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
SERVER_NONCE = bytes.fromhex("f0e0d0c0b0a090807060504030201001")
DAEMON_UUID = bytes.fromhex("00112233445566778899aabbccddeeff")
JOB_UUID = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
REQUEST_ID = 0x0123456789ABCDEF
CONNECTION_ID = 0x1122334455667788
JOB_EPOCH = 0x0102030405060708
RANK = 3
WORLD_SIZE = 8


class ProtocolError(ValueError):
    def __init__(self, message: str, status: int = MALFORMED) -> None:
        super().__init__(message)
        self.status = status


class DropFrame(ProtocolError):
    pass


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            mask = -(crc & 1) & 0xFFFFFFFF
            crc = (crc >> 1) ^ (0x82F63B78 & mask)
    return crc ^ 0xFFFFFFFF


def capability_bitmap(*bits: int) -> bytes:
    result = bytearray(CAP_BYTES)
    for bit in bits:
        if bit < 0 or bit >= CAP_BYTES * 8:
            raise ValueError("capability bit out of range")
        result[bit // 8] |= 1 << (bit % 8)
    return bytes(result)


def encode_tlv(tlv_type: int, flags: int, value: bytes) -> bytes:
    prefix = struct.pack(">HHI", tlv_type, flags, len(value)) + value
    return prefix + bytes((-len(prefix)) % 8)


def topology_tlv(job_uuid: bytes = JOB_UUID, rank: int = RANK,
                 world_size: int = WORLD_SIZE) -> bytes:
    value = job_uuid + struct.pack(">II", rank, world_size)
    return encode_tlv(TOPOLOGY_IDENTITY, CRITICAL, value)


def hello_payload(*, offered: bytes | None = None,
                  required: bytes | None = None,
                  minimum: tuple[int, int] = (1, 0),
                  maximum: tuple[int, int] = (1, 0),
                  nonce: bytes = CLIENT_NONCE, rx_max: int = MAX_RECORD,
                  role: int = RUNTIME, reserved: int = 0,
                  tlvs: bytes | None = None) -> bytes:
    caps = capability_bitmap(CAP_TOPOLOGY)
    return struct.pack(
        ">HHHH16s32s32sIHH",
        minimum[0], minimum[1], maximum[0], maximum[1], nonce,
        offered if offered is not None else caps,
        required if required is not None else caps,
        rx_max, role, reserved,
    ) + (topology_tlv() if tlvs is None else tlvs)


def ack_payload(*, status: int = OK, nonce_echo: bytes = CLIENT_NONCE,
                server_nonce: bytes | None = None,
                selected: bytes | None = None, max_record: int = MAX_RECORD,
                role: int = DAEMON, reserved: int = 0,
                tlvs: bytes | None = None,
                selected_version: tuple[int, int] | None = None) -> bytes:
    caps = capability_bitmap(CAP_TOPOLOGY)
    success = status == OK
    if selected_version is None:
        selected_version = (1, 0) if success else (0, 0)
    if server_nonce is None:
        server_nonce = SERVER_NONCE if success else bytes(16)
    if selected is None:
        selected = caps if success else bytes(CAP_BYTES)
    if tlvs is None:
        tlvs = topology_tlv() if success else b""
    return struct.pack(
        ">HHI16s16s32sIHH",
        selected_version[0], selected_version[1], status, nonce_echo,
        server_nonce, selected,
        max_record, role, reserved,
    ) + tlvs


def encode_frame(message_type: int, payload: bytes, *,
                 request_id: int = REQUEST_ID,
                 daemon_uuid: bytes = DAEMON_UUID,
                 connection_id: int = 0, job_epoch: int = JOB_EPOCH,
                 magic: bytes = MAGIC, framing: tuple[int, int] = (1, 0),
                 header_bytes: int = HEADER_BYTES, flags: int = 0,
                 reserved0: int = 0, reserved1: int = 0) -> bytes:
    header = struct.pack(
        ">8sHHHHIIQ16sQQIIQ",
        magic, framing[0], framing[1], header_bytes, message_type, flags,
        len(payload), request_id, daemon_uuid, connection_id, job_epoch, 0,
        reserved0, reserved1,
    )
    if len(header) != HEADER_BYTES:
        raise AssertionError("test encoder header layout drifted")
    frame = header + payload
    checksum = crc32c(frame)
    return frame[:CRC_OFFSET] + struct.pack(">I", checksum) + frame[68:]


def with_recomputed_crc(frame: bytes) -> bytes:
    zeroed = frame[:CRC_OFFSET] + bytes(4) + frame[68:]
    checksum = crc32c(zeroed)
    return zeroed[:CRC_OFFSET] + struct.pack(">I", checksum) + zeroed[68:]


def parse_tlvs(data: bytes) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[int] = set()
    offset = 0
    while offset < len(data):
        if len(data) - offset < 8:
            raise ProtocolError("truncated TLV header")
        tlv_type, flags, value_length = struct.unpack_from(">HHI", data, offset)
        if flags & ~CRITICAL:
            raise ProtocolError("invalid TLV flags")
        remaining = len(data) - offset
        if value_length > remaining - 8:
            raise ProtocolError("invalid TLV length")
        value_start = offset + 8
        value_end = value_start + value_length
        padding = (-value_length) % 8
        if padding > remaining - 8 - value_length:
            raise ProtocolError("invalid TLV padding length")
        padded_end = value_end + padding
        if any(data[value_end:padded_end]):
            raise ProtocolError("nonzero TLV padding")
        if tlv_type in seen:
            raise ProtocolError("duplicate TLV")
        seen.add(tlv_type)
        value = data[value_start:value_end]
        if tlv_type == TOPOLOGY_IDENTITY:
            if flags != CRITICAL or value_length != 24:
                raise ProtocolError("malformed topology TLV")
            job_uuid = value[:16]
            rank, world_size = struct.unpack_from(">II", value, 16)
            result.append({"type": tlv_type, "flags": flags,
                           "job_uuid": job_uuid, "rank": rank,
                           "world_size": world_size})
        elif flags & CRITICAL:
            result.append({"type": tlv_type, "flags": flags,
                           "value": value, "unknown_critical": True})
        else:
            result.append({"type": tlv_type, "flags": flags,
                           "value": value})
        offset = padded_end
    return result


def decode_frame(frame: bytes) -> dict[str, object]:
    if len(frame) < HEADER_BYTES or len(frame) > MAX_RECORD:
        raise DropFrame("record size")
    fields = struct.unpack_from(">8sHHHHIIQ16sQQIIQ", frame)
    (magic, major, minor, header_bytes, message_type, flags, payload_bytes,
     request_id, daemon_uuid, connection_id, job_epoch, stored_crc,
     reserved0, reserved1) = fields
    if magic != MAGIC or (major, minor) != (1, 0):
        raise DropFrame("framing")
    if header_bytes != HEADER_BYTES or payload_bytes != len(frame) - HEADER_BYTES:
        raise DropFrame("length")
    if len(frame) > MAX_HANDSHAKE_RECORD:
        raise DropFrame("handshake size")
    if message_type not in (HELLO, HELLO_ACK) or flags != 0:
        raise DropFrame("type or flags")
    if request_id == 0 or reserved0 != 0 or reserved1 != 0:
        raise DropFrame("request or reserved")
    zeroed = frame[:CRC_OFFSET] + bytes(4) + frame[68:]
    if crc32c(zeroed) != stored_crc:
        raise DropFrame("CRC32C")
    payload = frame[HEADER_BYTES:]
    if message_type == HELLO:
        if connection_id != 0:
            raise DropFrame("HELLO connection")
        if len(payload) < 96:
            raise ProtocolError("short HELLO")
        fixed = struct.unpack_from(">HHHH16s32s32sIHH", payload)
        (min_major, min_minor, max_major, max_minor, nonce, offered, required,
         rx_max, role, reserved) = fixed
        if nonce == bytes(16) or role != RUNTIME or reserved != 0:
            raise ProtocolError("HELLO fixed fields")
        if rx_max < 4096 or rx_max > MAX_RECORD:
            raise ProtocolError("HELLO maximum record")
        tlvs = parse_tlvs(payload[96:])
        return {"type": HELLO, "request_id": request_id,
                "daemon_uuid": daemon_uuid, "connection_id": connection_id,
                "job_epoch": job_epoch, "minimum": (min_major, min_minor),
                "maximum": (max_major, max_minor), "nonce": nonce,
                "offered": offered, "required": required, "rx_max": rx_max,
                "tlvs": tlvs, "crc32c": stored_crc}
    if len(payload) < 80:
        raise ProtocolError("short ACK")
    fixed = struct.unpack_from(">HHI16s16s32sIHH", payload)
    (selected_major, selected_minor, status, nonce_echo, server_nonce, selected,
     max_record, role, reserved) = fixed
    if role != DAEMON or reserved != 0 or status not in KNOWN_STATUSES:
        raise ProtocolError("ACK fixed fields")
    if daemon_uuid == bytes(16):
        raise DropFrame("ACK daemon identity")
    tlvs = parse_tlvs(payload[80:])
    if status == OK:
        if (connection_id == 0 or job_epoch == 0 or
                server_nonce == bytes(16) or
                (selected_major, selected_minor) != (1, 0)):
            raise DropFrame("ACK session")
        topology = [tlv for tlv in tlvs
                    if tlv["type"] == TOPOLOGY_IDENTITY]
        if not (selected[0] & 1) or len(topology) != 1:
            raise ProtocolError("ACK topology capability")
    elif (connection_id != 0 or job_epoch == 0 or
          server_nonce != bytes(16) or
          (selected_major, selected_minor) != (0, 0) or
          selected != bytes(CAP_BYTES) or tlvs):
        raise DropFrame("noncanonical failed ACK")
    if max_record < 4096 or max_record > MAX_RECORD:
        raise ProtocolError("ACK maximum record")
    return {"type": HELLO_ACK, "request_id": request_id,
            "daemon_uuid": daemon_uuid, "connection_id": connection_id,
            "job_epoch": job_epoch, "selected_version": (selected_major,
            selected_minor), "status": status, "nonce_echo": nonce_echo,
            "server_nonce": server_nonce, "selected": selected,
            "max_record": max_record, "tlvs": tlvs, "crc32c": stored_crc}


def negotiation_status(hello: dict[str, object]) -> int:
    required = hello["required"]
    offered = hello["offered"]
    assert isinstance(required, bytes) and isinstance(offered, bytes)
    if any(required[index] & ~offered[index] for index in range(CAP_BYTES)):
        return MALFORMED
    if hello["minimum"] > hello["maximum"]:
        return MALFORMED
    topology = [tlv for tlv in hello["tlvs"]
                if tlv["type"] == TOPOLOGY_IDENTITY]
    topology_offered = bool(offered[0] & 1)
    if topology_offered != (len(topology) == 1):
        return MALFORMED
    if not (hello["minimum"] <= (1, 0) <= hello["maximum"]):
        return UNSUPPORTED_VERSION
    if any(tlv.get("unknown_critical", False) for tlv in hello["tlvs"]):
        return UNSUPPORTED_CAPABILITY
    supported = capability_bitmap(CAP_TOPOLOGY)
    selected = bytes(offered[index] & supported[index]
                     for index in range(CAP_BYTES))
    if any(required[index] & ~selected[index] for index in range(CAP_BYTES)):
        return UNSUPPORTED_CAPABILITY
    if not (required[0] & 1) or not (selected[0] & 1):
        return UNSUPPORTED_CAPABILITY
    if hello["daemon_uuid"] not in (bytes(16), DAEMON_UUID):
        return INSTANCE_MISMATCH
    value = topology[0]
    wildcard = (value["job_uuid"] == bytes(16) and
                value["rank"] == 0xFFFFFFFF and value["world_size"] == 0)
    exact = (value["job_uuid"] == JOB_UUID and value["rank"] == RANK and
             value["world_size"] == WORLD_SIZE)
    if not wildcard and not exact:
        return TOPOLOGY_MISMATCH
    if hello["job_epoch"] not in (0, JOB_EPOCH):
        return TOPOLOGY_MISMATCH
    return OK


def hello_result(frame: bytes) -> int:
    try:
        return negotiation_status(decode_frame(frame))
    except DropFrame:
        raise
    except ProtocolError as error:
        return error.status


def hello_server_action(
        frame: bytes, *, established: bool = False) -> tuple[str, int | None]:
    try:
        decoded = decode_frame(frame)
        if decoded["type"] != HELLO:
            return ("drop", None)
        status = negotiation_status(decoded)
        if established and status != MALFORMED:
            status = PROTOCOL_STATE
        return ("ack", status)
    except DropFrame:
        return ("drop", None)
    except ProtocolError as error:
        if (len(frame) < 16 or
                struct.unpack_from(">H", frame, 14)[0] != HELLO):
            return ("drop", None)
        if len(frame) < HEADER_BYTES + 96:
            return ("drop", None)
        payload = frame[HEADER_BYTES:]
        if payload[8:24] == bytes(16):
            return ("drop", None)
        status = error.status
        if established and status != MALFORMED:
            status = PROTOCOL_STATE
        return ("ack", status)


def capability_subset_bytes(subset: bytes, superset: bytes) -> bool:
    return all((subset[index] & ~superset[index]) == 0
               for index in range(CAP_BYTES))


def validate_ack(hello_frame: bytes, ack_frame: bytes) -> dict[str, object]:
    hello = decode_frame(hello_frame)
    ack = decode_frame(ack_frame)
    if hello["type"] != HELLO or ack["type"] != HELLO_ACK:
        raise ProtocolError("invalid correlation message types")
    if ack["request_id"] != hello["request_id"]:
        raise ProtocolError("wrong ACK request ID")
    if ack["nonce_echo"] != hello["nonce"]:
        raise ProtocolError("wrong ACK nonce")
    if ack["status"] != OK:
        return ack
    if hello["daemon_uuid"] not in (bytes(16), ack["daemon_uuid"]):
        raise ProtocolError("wrong ACK daemon identity")
    if hello["job_epoch"] not in (0, ack["job_epoch"]):
        raise ProtocolError("wrong ACK epoch")
    offered = hello["offered"]
    required = hello["required"]
    selected = ack["selected"]
    assert isinstance(offered, bytes) and isinstance(required, bytes)
    assert isinstance(selected, bytes)
    if not capability_subset_bytes(selected, offered):
        raise ProtocolError("ACK selected unoffered capability")
    supported = capability_bitmap(CAP_TOPOLOGY)
    expected_selected = bytes(offered[index] & supported[index]
                              for index in range(CAP_BYTES))
    if selected != expected_selected:
        raise ProtocolError("ACK selected unsupported capability")
    if not capability_subset_bytes(required, selected):
        raise ProtocolError("ACK omitted required capability")
    if ack["max_record"] > hello["rx_max"]:
        raise ProtocolError("ACK maximum exceeds client receive limit")
    hello_topology = next(
        tlv for tlv in hello["tlvs"] if tlv["type"] == TOPOLOGY_IDENTITY)
    ack_topology = next(
        tlv for tlv in ack["tlvs"] if tlv["type"] == TOPOLOGY_IDENTITY)
    wildcard = (hello_topology["job_uuid"] == bytes(16) and
                hello_topology["rank"] == 0xFFFFFFFF and
                hello_topology["world_size"] == 0)
    if not wildcard and ack_topology != hello_topology:
        raise ProtocolError("wrong ACK topology")
    if (ack_topology["job_uuid"] == bytes(16) or
            ack_topology["world_size"] == 0 or
            ack_topology["rank"] >= ack_topology["world_size"]):
        raise ProtocolError("invalid ACK topology")
    return ack


class HostTransportProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        cls.golden_hello = bytes.fromhex(
            cls.spec["golden"]["hello_success_request"]["frame_hex"])
        cls.golden_ack = bytes.fromhex(
            cls.spec["golden"]["hello_success_ack"]["frame_hex"])

    def test_schema_offsets_are_contiguous_and_exact(self) -> None:
        self.assertEqual(self.spec["schema"], "amdgpu-sim.host-transport.v1")
        self.assertEqual(self.spec["transport"]["socket_type"], "SOCK_SEQPACKET")
        self.assertEqual(self.spec["transport"]["byte_order"], "big-endian")
        fields = self.spec["header"]["fields"]
        self.assertEqual([(field["name"], field["offset"], field["bytes"])
                          for field in fields], [
            ("magic", 0, 8), ("framing_major", 8, 2),
            ("framing_minor", 10, 2), ("header_bytes", 12, 2),
            ("message_type", 14, 2), ("flags", 16, 4),
            ("payload_bytes", 20, 4), ("request_id", 24, 8),
            ("daemon_instance_uuid", 32, 16), ("connection_id", 48, 8),
            ("job_epoch", 56, 8), ("crc32c", 64, 4),
            ("reserved0", 68, 4), ("reserved1", 72, 8),
        ])
        self.assertEqual(fields[-1]["offset"] + fields[-1]["bytes"], 80)

    def test_schema_enums_capability_and_tlv_contract(self) -> None:
        self.assertEqual(self.spec["messages"]["HELLO"]["value"], HELLO)
        self.assertEqual(self.spec["messages"]["HELLO_ACK"]["value"], HELLO_ACK)
        self.assertEqual(self.spec["roles"], {"RUNTIME": 1, "DAEMON": 2})
        self.assertEqual(self.spec["statuses"], {
            "OK": 0, "MALFORMED": 1, "UNSUPPORTED_VERSION": 2,
            "UNSUPPORTED_CAPABILITY": 3, "INSTANCE_MISMATCH": 4,
            "TOPOLOGY_MISMATCH": 5, "UNAUTHORIZED": 6, "BUSY": 7,
            "RESOURCE_EXHAUSTED": 8, "PROTOCOL_STATE": 9, "INTERNAL": 10,
        })
        self.assertEqual(self.spec["capabilities"]["TOPOLOGY_IDENTITY_V1"], 0)
        topology = self.spec["tlv"]["types"]["TOPOLOGY_IDENTITY"]
        self.assertEqual((topology["value"], topology["required_flags"],
                          topology["value_bytes"]), (1, 1, 24))
        self.assertEqual(capability_bitmap(0, 8), b"\x01\x01" + bytes(30))
        self.assertEqual(
            self.spec["handshake_rules"]["required_session_capabilities"],
            ["TOPOLOGY_IDENTITY_V1"])
        self.assertEqual(
            self.spec["handshake_rules"]["failure_ack"]["tlvs"], "none")
        self.assertIn(
            "silent-drop",
            self.spec["handshake_rules"]["state_machine"]["server_initial"])
        self.assertEqual(
            self.spec["handshake_rules"]["state_machine"]
                     ["server_established"],
            "after-envelope-and-structural-validation-second-HELLO-canonical-"
            "PROTOCOL_STATE-before-negotiation-failure-ACK-then-close",
        )
        self.assertEqual(
            self.spec["handshake_rules"]["await_hello_error_precedence"],
            ["MALFORMED", "UNSUPPORTED_VERSION", "UNSUPPORTED_CAPABILITY",
             "INSTANCE_MISMATCH", "TOPOLOGY_MISMATCH", "BUSY",
             "RESOURCE_EXHAUSTED", "INTERNAL"],
        )
        self.assertEqual(
            self.spec["handshake_rules"]["established_error_precedence"],
            ["MALFORMED", "PROTOCOL_STATE", "INTERNAL"],
        )
        self.assertEqual(self.spec["deadline"], {
            "clock": "CLOCK_MONOTONIC",
            "scope": (
                "one-absolute-deadline-for-connect-send-hello-receive-and-"
                "validate-ack"),
            "client_input": (
                "absolute-nanoseconds-or-relative-duration-converted-once-"
                "after-synchronous-validation-before-randomness-or-transport-"
                "io"),
            "cancellation": (
                "optional-caller-owned-cloexec-pollable-fd-readability-"
                "hangup-or-error"),
            "deadline_cancellation_precedence": (
                "expired-deadline-before-ready-cancellation"),
            "retries": 0,
            "eintr": "recompute-remaining-from-original-absolute-deadline",
        })

    def test_crc32c_check_vector(self) -> None:
        self.assertEqual(crc32c(b"123456789"), 0xE3069283)
        self.assertEqual(self.spec["crc32c"]["check_value_hex"], "e3069283")

    def test_hello_golden_is_independently_reconstructed(self) -> None:
        frame = encode_frame(HELLO, hello_payload())
        self.assertEqual(frame, self.golden_hello)
        self.assertEqual(len(frame), 208)
        self.assertEqual(frame[CRC_OFFSET:68].hex(), "508ae012")
        decoded = decode_frame(frame)
        self.assertEqual(hello_result(frame), OK)
        self.assertEqual(decoded["request_id"], REQUEST_ID)
        self.assertEqual(decoded["connection_id"], 0)
        self.assertEqual(decoded["tlvs"][0]["world_size"], WORLD_SIZE)
        metadata = self.spec["golden"]["hello_success_request"]
        identity = self.spec["golden"]["identity"]
        self.assertEqual(metadata["frame_bytes"], len(frame))
        self.assertEqual(metadata["payload_bytes"], len(frame) - HEADER_BYTES)
        self.assertEqual(metadata["crc32c_hex"], frame[CRC_OFFSET:68].hex())
        self.assertEqual(int(identity["request_id_hex"], 16), REQUEST_ID)
        self.assertEqual(bytes.fromhex(
            identity["daemon_instance_uuid_hex"]), DAEMON_UUID)
        self.assertEqual(bytes.fromhex(identity["job_uuid_hex"]), JOB_UUID)
        self.assertEqual(identity["rank"], RANK)
        self.assertEqual(identity["world_size"], WORLD_SIZE)

    def test_ack_golden_is_independently_reconstructed(self) -> None:
        frame = encode_frame(HELLO_ACK, ack_payload(), connection_id=CONNECTION_ID)
        self.assertEqual(frame, self.golden_ack)
        self.assertEqual(len(frame), 192)
        self.assertEqual(frame[CRC_OFFSET:68].hex(), "c09c2612")
        decoded = decode_frame(frame)
        self.assertEqual(decoded["status"], OK)
        self.assertEqual(decoded["nonce_echo"], CLIENT_NONCE)
        self.assertEqual(decoded["server_nonce"], SERVER_NONCE)
        self.assertEqual(decoded["connection_id"], CONNECTION_ID)
        metadata = self.spec["golden"]["hello_success_ack"]
        identity = self.spec["golden"]["identity"]
        self.assertEqual(metadata["frame_bytes"], len(frame))
        self.assertEqual(metadata["payload_bytes"], len(frame) - HEADER_BYTES)
        self.assertEqual(metadata["crc32c_hex"], frame[CRC_OFFSET:68].hex())
        self.assertEqual(int(identity["connection_id_hex"], 16), CONNECTION_ID)
        self.assertEqual(int(identity["job_epoch_hex"], 16), JOB_EPOCH)
        self.assertEqual(identity["rx_max_record"], MAX_RECORD)
        self.assertEqual(bytes.fromhex(
            identity["client_nonce_hex"]), CLIENT_NONCE)
        self.assertEqual(bytes.fromhex(
            identity["server_nonce_hex"]), SERVER_NONCE)
        validate_ack(self.golden_hello, frame)

    def test_invalid_envelope_mutations_drop_without_ack(self) -> None:
        mutations: list[bytes] = []
        mutations.append(self.golden_hello[:7])
        bad_magic = bytearray(self.golden_hello)
        bad_magic[0] ^= 1
        mutations.append(bytes(bad_magic))
        bad_payload_length = bytearray(self.golden_hello)
        struct.pack_into(">I", bad_payload_length, 20, 127)
        mutations.append(with_recomputed_crc(bytes(bad_payload_length)))
        bad_crc = bytearray(self.golden_hello)
        bad_crc[-1] ^= 1
        mutations.append(bytes(bad_crc))
        zero_request = bytearray(self.golden_hello)
        zero_request[24:32] = bytes(8)
        mutations.append(with_recomputed_crc(bytes(zero_request)))
        nonzero_connection = bytearray(self.golden_hello)
        struct.pack_into(">Q", nonzero_connection, 48, 1)
        mutations.append(with_recomputed_crc(bytes(nonzero_connection)))
        nonzero_reserved = bytearray(self.golden_hello)
        struct.pack_into(">I", nonzero_reserved, 68, 1)
        mutations.append(with_recomputed_crc(bytes(nonzero_reserved)))
        for frame in mutations:
            with self.subTest(frame=frame.hex()[:32]):
                with self.assertRaises(DropFrame):
                    decode_frame(frame)

    def test_valid_envelope_payload_mutations_are_malformed(self) -> None:
        cases = [
            hello_payload(rx_max=4095),
            hello_payload(role=DAEMON),
            hello_payload(reserved=1),
            hello_payload(tlvs=topology_tlv() + topology_tlv()),
            hello_payload(tlvs=encode_tlv(TOPOLOGY_IDENTITY, 2,
                                          JOB_UUID + struct.pack(">II", RANK, WORLD_SIZE))),
            hello_payload(tlvs=struct.pack(">HHI", TOPOLOGY_IDENTITY,
                                           CRITICAL, 25) + bytes(24)),
            hello_payload(tlvs=struct.pack(">HHI", 99, 0, 0xFFFFFFFF)),
            hello_payload(tlvs=struct.pack(">HHI", 99, 0, 0xFFFFFFF8)),
        ]
        padded = bytearray(encode_tlv(99, 0, b"x"))
        padded[-1] = 1
        cases.append(hello_payload(tlvs=bytes(padded)))
        for payload in cases:
            with self.subTest(payload=payload.hex()[-48:]):
                self.assertEqual(
                    hello_result(encode_frame(HELLO, payload)), MALFORMED)

    def test_correlatable_malformed_boundary_is_unambiguous(self) -> None:
        for payload_size in (0, 23, 24, 95):
            frame = encode_frame(HELLO, bytes(payload_size))
            self.assertEqual(hello_server_action(frame), ("drop", None))
        zero_nonce = encode_frame(HELLO, hello_payload(nonce=bytes(16)))
        self.assertEqual(hello_server_action(zero_nonce), ("drop", None))
        bad_role = encode_frame(HELLO, hello_payload(role=DAEMON))
        self.assertEqual(hello_server_action(bad_role), ("ack", MALFORMED))
        self.assertEqual(
            hello_server_action(self.golden_ack), ("drop", None))

    def test_unknown_optional_is_skipped_and_unknown_critical_rejected(self) -> None:
        optional = encode_tlv(99, 0, b"future")
        decoded = decode_frame(encode_frame(
            HELLO, hello_payload(tlvs=topology_tlv() + optional)))
        self.assertEqual(len(decoded["tlvs"]), 2)
        critical = encode_tlv(99, CRITICAL, b"future")
        self.assertEqual(hello_result(encode_frame(
            HELLO, hello_payload(tlvs=topology_tlv() + critical))),
            UNSUPPORTED_CAPABILITY)
        version_first = encode_frame(HELLO, hello_payload(
            minimum=(2, 0), maximum=(2, 0),
            tlvs=topology_tlv() + critical))
        self.assertEqual(hello_result(version_first), UNSUPPORTED_VERSION)

    def test_established_state_precedes_negotiation_but_not_structure(self) -> None:
        incompatible = encode_frame(
            HELLO, hello_payload(minimum=(2, 0), maximum=(2, 0)))
        critical = encode_frame(
            HELLO,
            hello_payload(
                tlvs=topology_tlv() + encode_tlv(99, CRITICAL, b"future")),
        )
        malformed = encode_frame(HELLO, hello_payload(role=DAEMON))
        for frame in (self.golden_hello, incompatible, critical):
            with self.subTest(frame=frame[-32:].hex()):
                self.assertEqual(
                    hello_server_action(frame, established=True),
                    ("ack", PROTOCOL_STATE),
                )
        self.assertEqual(
            hello_server_action(malformed, established=True),
            ("ack", MALFORMED),
        )

    def test_deterministic_negotiation_failures(self) -> None:
        unsupported_bit = capability_bitmap(1)
        cases = [
            (hello_payload(required=unsupported_bit), MALFORMED),
            (hello_payload(offered=capability_bitmap(0, 1),
                           required=capability_bitmap(0, 1)),
             UNSUPPORTED_CAPABILITY),
            (hello_payload(offered=bytes(CAP_BYTES), required=bytes(CAP_BYTES),
                           tlvs=b""), UNSUPPORTED_CAPABILITY),
            (hello_payload(offered=capability_bitmap(0),
                           required=bytes(CAP_BYTES)),
             UNSUPPORTED_CAPABILITY),
            (hello_payload(offered=bytes(CAP_BYTES), required=bytes(CAP_BYTES)),
             MALFORMED),
            (hello_payload(offered=capability_bitmap(0),
                           required=capability_bitmap(0), tlvs=b""),
             MALFORMED),
            (hello_payload(minimum=(1, 1), maximum=(1, 0)), MALFORMED),
            (hello_payload(minimum=(1, 1), maximum=(1, 1)),
             UNSUPPORTED_VERSION),
            (hello_payload(minimum=(2, 0), maximum=(3, 0)),
             UNSUPPORTED_VERSION),
        ]
        for payload, expected in cases:
            decoded = decode_frame(encode_frame(HELLO, payload))
            self.assertEqual(negotiation_status(decoded), expected)

        compatible_range = decode_frame(encode_frame(
            HELLO, hello_payload(minimum=(1, 0), maximum=(1, 1))))
        self.assertEqual(negotiation_status(compatible_range), OK)

        wrong_instance = decode_frame(encode_frame(
            HELLO, hello_payload(), daemon_uuid=b"x" * 16))
        self.assertEqual(negotiation_status(wrong_instance), INSTANCE_MISMATCH)
        wrong_topology = decode_frame(encode_frame(
            HELLO, hello_payload(tlvs=topology_tlv(rank=4))))
        self.assertEqual(negotiation_status(wrong_topology), TOPOLOGY_MISMATCH)

    def test_identity_wildcards_and_partial_wildcard(self) -> None:
        wildcard = topology_tlv(bytes(16), 0xFFFFFFFF, 0)
        decoded = decode_frame(encode_frame(
            HELLO, hello_payload(tlvs=wildcard), daemon_uuid=bytes(16),
            job_epoch=0))
        self.assertEqual(negotiation_status(decoded), OK)
        partial = topology_tlv(bytes(16), RANK, 0)
        decoded = decode_frame(encode_frame(
            HELLO, hello_payload(tlvs=partial), daemon_uuid=bytes(16),
            job_epoch=0))
        self.assertEqual(negotiation_status(decoded), TOPOLOGY_MISMATCH)

    def test_ack_correlation_and_session_mutations_rejected(self) -> None:
        zero_connection = bytearray(self.golden_ack)
        zero_connection[48:56] = bytes(8)
        with self.assertRaises(DropFrame):
            decode_frame(with_recomputed_crc(bytes(zero_connection)))
        zero_server_nonce = bytearray(self.golden_ack)
        zero_server_nonce[HEADER_BYTES + 24:HEADER_BYTES + 40] = bytes(16)
        with self.assertRaises(DropFrame):
            decode_frame(with_recomputed_crc(bytes(zero_server_nonce)))

        mutations: list[bytes] = []
        wrong_request = bytearray(self.golden_ack)
        struct.pack_into(">Q", wrong_request, 24, REQUEST_ID + 1)
        mutations.append(with_recomputed_crc(bytes(wrong_request)))
        wrong_nonce = bytearray(self.golden_ack)
        wrong_nonce[HEADER_BYTES + 8] ^= 1
        mutations.append(with_recomputed_crc(bytes(wrong_nonce)))
        wrong_instance = bytearray(self.golden_ack)
        wrong_instance[32] ^= 1
        mutations.append(with_recomputed_crc(bytes(wrong_instance)))
        wrong_epoch = bytearray(self.golden_ack)
        struct.pack_into(">Q", wrong_epoch, 56, JOB_EPOCH + 1)
        mutations.append(with_recomputed_crc(bytes(wrong_epoch)))
        wrong_topology = bytearray(self.golden_ack)
        struct.pack_into(">I", wrong_topology, HEADER_BYTES + 80 + 8 + 16,
                         RANK + 1)
        mutations.append(with_recomputed_crc(bytes(wrong_topology)))
        for frame in mutations:
            with self.subTest(frame=frame.hex()[48:96]):
                with self.assertRaises(ProtocolError):
                    validate_ack(self.golden_hello, frame)

        smaller_hello = encode_frame(
            HELLO, hello_payload(rx_max=4096))
        with self.assertRaises(ProtocolError):
            validate_ack(smaller_hello, self.golden_ack)

        offered_extra = encode_frame(HELLO, hello_payload(
            offered=capability_bitmap(0, 1), required=capability_bitmap(0)))
        selected_extra = encode_frame(
            HELLO_ACK,
            ack_payload(selected=capability_bitmap(0, 1)),
            connection_id=CONNECTION_ID)
        with self.assertRaises(ProtocolError):
            validate_ack(offered_extra, selected_extra)

    def test_failure_ack_is_correlated_but_establishes_no_session(self) -> None:
        payload = ack_payload(status=UNSUPPORTED_VERSION)
        frame = encode_frame(HELLO_ACK, payload, connection_id=0)
        decoded = validate_ack(self.golden_hello, frame)
        self.assertEqual(decoded["status"], UNSUPPORTED_VERSION)
        self.assertEqual(decoded["request_id"], REQUEST_ID)
        self.assertEqual(decoded["nonce_echo"], CLIENT_NONCE)
        self.assertEqual(decoded["connection_id"], 0)
        self.assertEqual(decoded["server_nonce"], bytes(16))
        self.assertEqual(decoded["selected_version"], (0, 0))
        self.assertEqual(decoded["selected"], bytes(CAP_BYTES))
        self.assertEqual(decoded["tlvs"], [])

        noncanonical = ack_payload(
            status=UNSUPPORTED_VERSION, selected_version=(1, 0))
        with self.assertRaises(DropFrame):
            decode_frame(encode_frame(HELLO_ACK, noncanonical))
        unknown = ack_payload(status=0xFFFFFFFF)
        with self.assertRaises(ProtocolError):
            decode_frame(encode_frame(HELLO_ACK, unknown))
        zero_epoch = encode_frame(
            HELLO_ACK, ack_payload(status=UNSUPPORTED_VERSION), job_epoch=0)
        with self.assertRaises(DropFrame):
            decode_frame(zero_epoch)


if __name__ == "__main__":
    unittest.main()
