"""Decoder gates for the gem5 bridge wire tape.

The tape is written by C++ and read by Python, so the two ends can drift
silently. These tests pin the on-disk layout against the C++ header, pin the
message and operation tables against the versioned protocol contract, and prove
that a truncated tape is reported rather than hidden.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DECODER = ROOT / "tools/bridge_tape_decode.py"
TAPE_HEADER = ROOT / "projects/gem5/src/dev/amdgpu/host_gpu_bridge_tape.hh"
PROTOCOL_HEADER = ROOT / "projects/gem5/src/dev/amdgpu/host_gpu_protocol.hh"
KMT_HEADER = (
    ROOT
    / "projects/self-amdgpu-runtime/include/self_amdgpu_runtime/generated"
    / "bridge_kmt_v5.h"
)
GENERIC_HEADER = (
    ROOT
    / "projects/self-amdgpu-runtime/include/self_amdgpu_runtime/generated"
    / "bridge_generic_v2.h"
)
BASE_PROTOCOL = ROOT / "protocol/host-transport-v1.json"


def load_decoder():
    spec = importlib.util.spec_from_file_location("bridge_tape_decode", DECODER)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load bridge tape decoder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decoder = load_decoder()


def file_header(
    tick_frequency_hz: int = 1_000_000_000_000,
    epoch: int = 7,
    rank: int = 0,
    world_size: int = 1,
    maximum_record: int = 65536,
) -> bytes:
    header = bytearray(decoder.TAPE_FILE_HEADER_BYTES)
    header[0:8] = decoder.TAPE_MAGIC
    header[8:10] = (1).to_bytes(2, "big")
    header[10:12] = (0).to_bytes(2, "big")
    header[12:14] = decoder.TAPE_FILE_HEADER_BYTES.to_bytes(2, "big")
    header[14:16] = decoder.TAPE_RECORD_HEADER_BYTES.to_bytes(2, "big")
    header[16:24] = tick_frequency_hz.to_bytes(8, "big")
    header[24:32] = epoch.to_bytes(8, "big")
    header[32:36] = rank.to_bytes(4, "big")
    header[36:40] = world_size.to_bytes(4, "big")
    header[40:44] = maximum_record.to_bytes(4, "big")
    header[48:64] = bytes(range(16))
    header[64:80] = bytes(range(16, 32))
    return bytes(header)


def record_header(
    direction: int,
    carrier: bool,
    client_fd: int,
    generation: int,
    sequence: int,
    sim_tick: int,
    monotonic_ns: int,
    record_bytes: int,
) -> bytes:
    header = bytearray(decoder.TAPE_RECORD_HEADER_BYTES)
    header[0] = direction
    header[1] = 1 if carrier else 0
    header[4:8] = client_fd.to_bytes(4, "big")
    header[8:12] = record_bytes.to_bytes(4, "big")
    header[16:24] = generation.to_bytes(8, "big")
    header[24:32] = sequence.to_bytes(8, "big")
    header[32:40] = sim_tick.to_bytes(8, "big")
    header[40:48] = monotonic_ns.to_bytes(8, "big")
    return bytes(header)


def wire_frame(
    message_type: int,
    payload: bytes = b"",
    request_id: int = 1,
    connection_id: int = 0,
    job_epoch: int = 7,
    valid_crc: bool = True,
) -> bytes:
    frame = bytearray(decoder.WIRE_HEADER_BYTES + len(payload))
    frame[0:8] = decoder.WIRE_MAGIC
    frame[8:10] = (1).to_bytes(2, "big")
    frame[10:12] = (0).to_bytes(2, "big")
    frame[12:14] = decoder.WIRE_HEADER_BYTES.to_bytes(2, "big")
    frame[14:16] = message_type.to_bytes(2, "big")
    frame[20:24] = len(payload).to_bytes(4, "big")
    frame[24:32] = request_id.to_bytes(8, "big")
    frame[48:56] = connection_id.to_bytes(8, "big")
    frame[56:64] = job_epoch.to_bytes(8, "big")
    frame[decoder.WIRE_HEADER_BYTES:] = payload
    crc = decoder.crc32c(bytes(frame)) if valid_crc else 0
    frame[64:68] = crc.to_bytes(4, "big")
    return bytes(frame)


def kmt_payload(operation: int) -> bytes:
    payload = bytearray(256)
    payload[0:2] = (1).to_bytes(2, "big")
    payload[2:4] = (5).to_bytes(2, "big")
    payload[4:6] = operation.to_bytes(2, "big")
    return bytes(payload)


def sample_tape() -> bytes:
    hello = wire_frame(1, b"\x11" * 96, request_id=101)
    hello_ack = wire_frame(2, b"\x22" * 80, request_id=101)
    doorbell = wire_frame(14, kmt_payload(10), request_id=102)
    doorbell_ack = wire_frame(15, kmt_payload(10), request_id=102)
    tape = bytearray(file_header())
    for index, (direction, carrier, frame) in enumerate(
        [
            (0, False, hello),
            (1, False, hello_ack),
            (0, True, doorbell),
            (1, False, doorbell_ack),
        ],
        start=1,
    ):
        tape += record_header(
            direction, carrier, 9, 1, index, 1000 * index, 5000 * index,
            len(frame),
        )
        tape += frame
    return bytes(tape)


def decode(tape: bytes, payload_bytes: int = 0) -> list[dict]:
    stream = io.BytesIO(tape)
    lines = [decoder.read_file_header(stream)]
    lines.extend(decoder.iter_records(stream, payload_bytes))
    return lines


def constant(header: Path, pattern: str) -> int:
    match = re.search(pattern, header.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(f"{pattern} not found in {header}")
    return int(match.group(1), 0)


class BridgeTapeLayoutTest(unittest.TestCase):
    def test_layout_constants_match_the_writer(self) -> None:
        self.assertEqual(
            decoder.TAPE_FILE_HEADER_BYTES,
            constant(TAPE_HEADER, r"FileHeaderBytes = (\d+)"),
        )
        self.assertEqual(
            decoder.TAPE_RECORD_HEADER_BYTES,
            constant(TAPE_HEADER, r"RecordHeaderBytes = (\d+)"),
        )
        self.assertEqual(
            decoder.TAPE_FORMAT_MAJOR,
            constant(TAPE_HEADER, r"FormatMajor = (\d+)"),
        )
        magic = re.search(
            r"Magic = \{\s*([^}]*)\}", TAPE_HEADER.read_text(encoding="utf-8")
        )
        self.assertIsNotNone(magic)
        letters = re.findall(r"'(.)'", magic.group(1))
        self.assertEqual("".join(letters).encode("ascii"), decoder.TAPE_MAGIC)

    def test_wire_header_matches_the_base_protocol(self) -> None:
        base = json.loads(BASE_PROTOCOL.read_text(encoding="ascii"))
        self.assertEqual(base["header"]["bytes"], decoder.WIRE_HEADER_BYTES)
        self.assertEqual(base["header"]["crc32c_offset"], decoder.WIRE_CRC_OFFSET)
        fields = {field["name"]: field for field in base["header"]["fields"]}
        self.assertEqual(
            bytes.fromhex(fields["magic"]["constant_hex"]), decoder.WIRE_MAGIC
        )
        self.assertEqual(fields["message_type"]["offset"], 14)
        self.assertEqual(fields["payload_bytes"]["offset"], 20)
        self.assertEqual(fields["request_id"]["offset"], 24)
        self.assertEqual(fields["connection_id"]["offset"], 48)
        self.assertEqual(fields["job_epoch"]["offset"], 56)
        self.assertEqual(base["crc32c"]["check_value_hex"], "e3069283")
        self.assertEqual(
            decoder.crc32c(base["crc32c"]["check_input_ascii"].encode("ascii")),
            int(base["crc32c"]["check_value_hex"], 16),
        )


class BridgeTapeContractTest(unittest.TestCase):
    def test_message_table_matches_the_gem5_enumeration(self) -> None:
        text = PROTOCOL_HEADER.read_text(encoding="utf-8")
        block = re.search(r"enum class MessageType : uint16_t\s*\{(.*?)\}", text, re.S)
        self.assertIsNotNone(block)
        named = dict(
            re.findall(r"(\w+)\s*=\s*(\d+),", block.group(1))
        )
        for value, name in decoder.MESSAGE_TYPES.items():
            camel = name.title().replace("_", "")
            if camel in named:
                self.assertEqual(int(named[camel]), value, name)
        # The three externally defined types come from the generated contract.
        self.assertEqual(
            decoder.MESSAGE_TYPES[
                constant(KMT_HEADER, r"KMT_MESSAGE_REQUEST UINT16_C\((\d+)\)")
            ],
            "KMT_REQUEST",
        )
        self.assertEqual(
            decoder.MESSAGE_TYPES[
                constant(KMT_HEADER, r"KMT_MESSAGE_ACK UINT16_C\((\d+)\)")
            ],
            "KMT_ACK",
        )
        for suffix, name in (
            ("REQUEST", "GENERIC_DISPATCH_REQUEST"),
            ("ACK", "GENERIC_DISPATCH_ACK"),
            ("COMPLETION", "GENERIC_DISPATCH_COMPLETION"),
        ):
            value = constant(
                GENERIC_HEADER,
                rf"MESSAGE_GENERIC_DISPATCH_{suffix} UINT16_C\((\d+)\)",
            )
            self.assertEqual(decoder.MESSAGE_TYPES[value], name)

    def test_kmt_operation_table_matches_the_generated_contract(self) -> None:
        text = KMT_HEADER.read_text(encoding="utf-8")
        generated = {
            int(value): name
            for name, value in re.findall(
                r"SAGR_BRIDGE_KMT_OPERATION_(\w+) UINT16_C\((\d+)\)", text
            )
        }
        self.assertEqual(decoder.KMT_OPERATIONS, generated)
        self.assertEqual(decoder.KMT_MESSAGE_TYPES, (14, 15))


class BridgeTapeDecodeTest(unittest.TestCase):
    def test_decodes_both_directions_with_wire_detail(self) -> None:
        lines = decode(sample_tape(), payload_bytes=16)
        header, *rest = lines
        self.assertEqual(header["kind"], "file_header")
        self.assertEqual(header["tick_frequency_hz"], 1_000_000_000_000)
        self.assertEqual(header["epoch"], 7)
        self.assertEqual(header["daemon_uuid"], bytes(range(16)).hex())
        self.assertEqual(header["job_uuid"], bytes(range(16, 32)).hex())

        records = [entry for entry in rest if entry["kind"] == "record"]
        trailer = rest[-1]
        self.assertEqual(len(records), 4)
        self.assertEqual(trailer["kind"], "trailer")
        self.assertEqual(trailer["records"], 4)
        self.assertFalse(trailer["truncated"])

        self.assertEqual(
            [entry["direction"] for entry in records],
            ["ingress", "egress", "ingress", "egress"],
        )
        self.assertEqual(
            [entry["message"] for entry in records],
            ["HELLO", "HELLO_ACK", "KMT_REQUEST", "KMT_ACK"],
        )
        self.assertEqual([entry["sequence"] for entry in records], [1, 2, 3, 4])
        self.assertEqual(
            [entry["sim_tick"] for entry in records], [1000, 2000, 3000, 4000]
        )
        self.assertEqual(
            [entry["carrier"] for entry in records], [False, False, True, False]
        )
        self.assertTrue(all(entry["magic_ok"] for entry in records))
        self.assertTrue(all(entry["crc_ok"] for entry in records))
        self.assertEqual(records[2]["kmt_operation_name"], "QUEUE_DOORBELL")
        self.assertEqual(records[3]["kmt_operation_name"], "QUEUE_DOORBELL")
        self.assertEqual(len(records[0]["record_prefix"]), 32)
        self.assertEqual(records[0]["record_prefix"][:16], decoder.WIRE_MAGIC.hex())

    def test_reports_a_corrupt_frame_instead_of_hiding_it(self) -> None:
        frame = wire_frame(3, b"\x00" * 32, valid_crc=False)
        tape = bytearray(file_header())
        tape += record_header(0, False, 9, 1, 1, 10, 20, len(frame))
        tape += frame
        records = [
            entry for entry in decode(bytes(tape)) if entry["kind"] == "record"
        ]
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["crc_ok"])
        self.assertEqual(records[0]["message"], "QUEUE_REQUEST")

    def test_short_record_has_no_wire_header(self) -> None:
        short = b"\x01\x02\x03"
        tape = bytearray(file_header())
        tape += record_header(1, False, 9, 1, 1, 10, 20, len(short))
        tape += short
        records = [
            entry for entry in decode(bytes(tape)) if entry["kind"] == "record"
        ]
        self.assertIsNone(records[0]["wire_header"])
        self.assertEqual(records[0]["record_bytes"], 3)

    def test_truncated_tape_is_reported(self) -> None:
        tape = sample_tape()[:-7]
        lines = decode(tape)
        trailer = lines[-1]
        self.assertTrue(trailer["truncated"])
        self.assertEqual(trailer["records"], 3)

    def test_rejects_a_foreign_file(self) -> None:
        with self.assertRaises(decoder.TapeError):
            decode(b"not-a-tape" + bytes(128))
        with self.assertRaises(decoder.TapeError):
            decode(decoder.TAPE_MAGIC)

    def test_summary_counts_every_record(self) -> None:
        stream = io.BytesIO(sample_tape())
        decoder.read_file_header(stream)
        summary = decoder.summarize(decoder.iter_records(stream, 0))
        self.assertEqual(summary["records"], 4)
        self.assertFalse(summary["truncated"])
        self.assertEqual(summary["by_direction"], {"egress": 2, "ingress": 2})
        self.assertEqual(summary["carrier_records"], 1)
        self.assertEqual(summary["crc_failures"], 0)
        self.assertEqual(summary["first_sim_tick"], 1000)
        self.assertEqual(summary["last_sim_tick"], 4000)
        self.assertEqual(summary["by_connection"], {"9:1": 4})
        self.assertEqual(
            summary["by_message"],
            {"HELLO": 1, "HELLO_ACK": 1, "KMT_ACK": 1, "KMT_REQUEST": 1},
        )


class BridgeTapeCommandTest(unittest.TestCase):
    def test_command_line_emits_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tape.bin"
            path.write_bytes(sample_tape())
            completed = subprocess.run(
                ["/usr/bin/python3", str(DECODER), str(path), "--limit", "2"],
                capture_output=True,
                check=True,
                text=True,
            )
            lines = [json.loads(line) for line in completed.stdout.splitlines()]
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0]["kind"], "file_header")
            self.assertEqual(lines[1]["message"], "HELLO")
            self.assertEqual(lines[2]["message"], "HELLO_ACK")

            summary = subprocess.run(
                ["/usr/bin/python3", str(DECODER), str(path), "--summary"],
                capture_output=True,
                check=True,
                text=True,
            )
            aggregate = json.loads(summary.stdout.splitlines()[1])
            self.assertEqual(aggregate["records"], 4)

    def test_command_line_rejects_a_foreign_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tape.bin"
            path.write_bytes(bytes(128))
            completed = subprocess.run(
                ["/usr/bin/python3", str(DECODER), str(path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("magic mismatch", completed.stderr)


if __name__ == "__main__":
    unittest.main()
