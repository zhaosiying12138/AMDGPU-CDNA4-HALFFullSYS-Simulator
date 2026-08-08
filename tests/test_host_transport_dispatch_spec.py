#!/usr/bin/env python3
"""Independent checks for the CP-0008 pinned-dispatch wire authority."""

from __future__ import annotations

import hashlib
import json
import struct
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(name: str) -> dict[str, Any]:
    with (ROOT / "protocol" / name).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


class WireOracle:
    """Build frames only from the published base and extension field tables."""

    def __init__(self, base: dict[str, Any], spec: dict[str, Any]) -> None:
        self.base = base
        self.spec = spec
        self.header = {field["name"]: field for field in base["header"]["fields"]}
        self.request = {
            field["name"]: field for field in spec["request_payload"]["fields"]
        }
        self.result = {
            field["name"]: field for field in spec["result_payload"]["fields"]
        }
        self.messages = spec["transport"]["message_types"]

    @staticmethod
    def store_integer(target: bytearray, field: dict[str, Any], value: int) -> None:
        size = int(field["bytes"])
        offset = int(field["offset"])
        target[offset:offset + size] = (value % (1 << (size * 8))).to_bytes(
            size, "big"
        )

    @staticmethod
    def store_bytes(target: bytearray, field: dict[str, Any], value: bytes) -> None:
        size = int(field["bytes"])
        if len(value) != size:
            raise ValueError(f"{field['name']} requires {size} bytes")
        offset = int(field["offset"])
        target[offset:offset + size] = value

    @staticmethod
    def load_integer(source: bytes, field: dict[str, Any], base: int = 0) -> int:
        offset = base + int(field["offset"])
        size = int(field["bytes"])
        return int.from_bytes(source[offset:offset + size], "big")

    def payload(
        self, definition: dict[str, dict[str, Any]], size: int,
        values: dict[str, int | bytes],
    ) -> bytes:
        payload = bytearray(size)
        for name, field in definition.items():
            if "constant_hex" in field:
                self.store_bytes(payload, field, bytes.fromhex(field["constant_hex"]))
            elif "constant" in field:
                self.store_integer(payload, field, int(field["constant"]))
            elif name in values:
                value = values[name]
                if isinstance(value, bytes):
                    self.store_bytes(payload, field, value)
                else:
                    self.store_integer(payload, field, int(value))
        return bytes(payload)

    def frame(
        self,
        message_name: str,
        request_id: int,
        daemon_uuid: bytes,
        connection_id: int,
        job_epoch: int,
        payload: bytes,
    ) -> bytes:
        header = bytearray(int(self.base["header"]["bytes"]))
        values = {
            "message_type": int(self.messages[message_name]),
            "payload_bytes": len(payload),
            "request_id": request_id,
            "connection_id": connection_id,
            "job_epoch": job_epoch,
            "crc32c": 0,
        }
        for name, field in self.header.items():
            if "constant_hex" in field:
                self.store_bytes(header, field, bytes.fromhex(field["constant_hex"]))
            elif name == "daemon_instance_uuid":
                self.store_bytes(header, field, daemon_uuid)
            elif "constant" in field:
                self.store_integer(header, field, int(field["constant"]))
            elif name in values:
                self.store_integer(header, field, values[name])
        frame = header + payload
        self.store_integer(frame, self.header["crc32c"], crc32c(bytes(frame)))
        return bytes(frame)


class HostTransportDispatchSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = load_json("host-transport-v1.json")
        cls.queue = load_json("host-transport-v1-queue.json")
        cls.memory = load_json("host-transport-v1-memory.json")
        cls.signal = load_json("host-transport-v1-signal.json")
        cls.spec = load_json("host-transport-v1-dispatch.json")
        cls.oracle = WireOracle(cls.base, cls.spec)

    @staticmethod
    def as_hex(identity: dict[str, Any], name: str) -> int:
        return int(identity[name], 16)

    def identity_values(self) -> dict[str, int]:
        identity = self.spec["golden"]["identity"]
        return {
            "queue_id": self.as_hex(identity, "queue_id_hex"),
            "queue_generation": self.as_hex(identity, "queue_generation_hex"),
            "queue_sequence": self.as_hex(identity, "queue_sequence_hex"),
            "fixture_id": int(identity["fixture_id"]),
            "input_allocation_id": int(identity["input_allocation_id"]),
            "input_generation": self.as_hex(identity, "input_generation_hex"),
            "output_allocation_id": int(identity["output_allocation_id"]),
            "output_generation": self.as_hex(identity, "output_generation_hex"),
            "signal_id": int(identity["signal_id"]),
            "signal_generation": self.as_hex(identity, "signal_generation_hex"),
        }

    def golden_frames(self) -> dict[str, bytes]:
        identity = self.spec["golden"]["identity"]
        values = self.identity_values()
        request_id = self.as_hex(identity, "request_id_hex")
        daemon_uuid = bytes.fromhex(identity["daemon_instance_uuid_hex"])
        connection_id = self.as_hex(identity, "connection_id_hex")
        job_epoch = self.as_hex(identity, "job_epoch_hex")

        request = self.oracle.payload(
            self.oracle.request,
            int(self.spec["request_payload"]["bytes"]),
            {**values, "opcode": 1, "expected_signal_value": 0},
        )
        common_result = {
            **values,
            "status": 0,
            "opcode": 1,
            "trace_id": self.as_hex(identity, "trace_id_hex"),
            "input_gpu_va": self.as_hex(identity, "input_gpu_va_hex"),
            "output_gpu_va": self.as_hex(identity, "output_gpu_va_hex"),
            "packet_crc32c": int(identity["packet_crc32c_hex"], 16),
            "admission_tick": self.as_hex(identity, "admission_tick_hex"),
        }
        ack = self.oracle.payload(
            self.oracle.result,
            int(self.spec["result_payload"]["bytes"]),
            common_result,
        )
        completion = self.oracle.payload(
            self.oracle.result,
            int(self.spec["result_payload"]["bytes"]),
            {
                **common_result,
                "output_crc32c": int(
                    self.spec["fixture_authority"]["golden_buffers"][
                        "output_crc32c_hex"
                    ],
                    16,
                ),
                "start_tick": self.as_hex(identity, "start_tick_hex"),
                "end_tick": self.as_hex(identity, "end_tick_hex"),
                "retire_tick": self.as_hex(identity, "retire_tick_hex"),
            },
        )
        return {
            "dispatch_request": self.oracle.frame(
                "DISPATCH_REQUEST", request_id, daemon_uuid, connection_id,
                job_epoch, request,
            ),
            "dispatch_ack": self.oracle.frame(
                "DISPATCH_ACK", request_id, daemon_uuid, connection_id,
                job_epoch, ack,
            ),
            "dispatch_completion": self.oracle.frame(
                "DISPATCH_COMPLETION", request_id, daemon_uuid, connection_id,
                job_epoch, completion,
            ),
        }

    def test_capability_and_record_sizes_are_additive(self) -> None:
        self.assertEqual(
            self.spec["schema"], "amdgpu-sim.host-transport-v1.dispatch.v1"
        )
        capability = self.spec["capability"]
        self.assertEqual(
            (capability["name"], capability["bit"], capability["mask_hex"]),
            ("PINNED_DISPATCH_V1", 4, "10"),
        )
        self.assertIn("QUEUE_CONTROL_V1", capability["negotiation"])
        transport = self.spec["transport"]
        self.assertEqual(transport["header_bytes"], self.base["header"]["bytes"])
        self.assertEqual(transport["byte_order"], "big-endian")
        self.assertEqual(
            transport["message_types"],
            {"DISPATCH_REQUEST": 11, "DISPATCH_ACK": 12,
             "DISPATCH_COMPLETION": 13},
        )
        self.assertEqual(
            (transport["request_payload_bytes"], transport["request_frame_bytes"]),
            (128, 208),
        )
        self.assertEqual(
            (transport["result_payload_bytes"], transport["result_frame_bytes"]),
            (160, 240),
        )
        self.assertEqual(transport["ancillary_descriptors"], 0)
        old_types = set(self.queue["transport"]["message_types"].values())
        old_types.update(self.memory["transport"]["message_types"].values())
        old_types.update(self.signal["transport"]["message_types"].values())
        self.assertTrue(old_types.isdisjoint(transport["message_types"].values()))

    def test_payload_layouts_are_exact_and_non_overlapping(self) -> None:
        expected_request = {
            "queue_id": 8,
            "queue_generation": 16,
            "queue_sequence": 24,
            "fixture_id": 32,
            "input_allocation_id": 40,
            "input_generation": 48,
            "output_allocation_id": 56,
            "output_generation": 64,
            "signal_id": 72,
            "signal_generation": 80,
            "expected_signal_value": 88,
            "fixture_manifest_sha256": 96,
        }
        expected_result = {
            "queue_id": 16,
            "queue_generation": 24,
            "queue_sequence": 32,
            "fixture_id": 40,
            "input_allocation_id": 48,
            "input_generation": 56,
            "output_allocation_id": 64,
            "output_generation": 72,
            "signal_id": 80,
            "signal_generation": 88,
            "trace_id": 96,
            "input_gpu_va": 104,
            "output_gpu_va": 112,
            "packet_crc32c": 120,
            "output_crc32c": 124,
            "admission_tick": 128,
            "start_tick": 136,
            "end_tick": 144,
            "retire_tick": 152,
        }
        for name, expected in (
            ("request_payload", expected_request),
            ("result_payload", expected_result),
        ):
            payload = self.spec[name]
            cursor = 0
            offsets = {}
            for field in payload["fields"]:
                self.assertEqual(field["offset"], cursor)
                offsets[field["name"]] = field["offset"]
                cursor += field["bytes"]
            self.assertEqual(cursor, payload["bytes"])
            for field, offset in expected.items():
                self.assertEqual(offsets[field], offset)
        request_hash = self.oracle.request["fixture_manifest_sha256"]
        self.assertEqual(request_hash["bytes"], 32)
        self.assertEqual(
            request_hash["constant_hex"],
            self.spec["fixture_authority"]["manifest"]["sha256_hex"],
        )

    def test_fixture_code_aql_and_kernarg_are_pinned(self) -> None:
        fixture = self.spec["fixture_authority"]
        self.assertEqual(
            (fixture["architecture"], fixture["wavefront_size"],
             fixture["active_compute_units"]),
            ("gfx950", 64, 1),
        )
        self.assertEqual(fixture["workgroup_size"], [64, 1, 1])
        self.assertEqual(fixture["grid_size"], [64, 1, 1])
        self.assertEqual((fixture["workgroups"], fixture["waves"]), (1, 1))
        self.assertIn("already-admitted", fixture["signal_precondition"])
        self.assertIn("EQ-zero wait", fixture["signal_precondition"])

        code = bytes.fromhex(fixture["code_image"]["bytes_hex"])
        self.assertEqual(len(code), fixture["code_image"]["bytes"])
        self.assertEqual(len(code[:64]), fixture["code_image"]["descriptor_bytes"])
        self.assertEqual(len(code[64:]), fixture["code_image"]["machine_code_bytes"])
        self.assertEqual(hashlib.sha256(code).hexdigest(),
                         fixture["code_image"]["sha256_hex"])
        self.assertEqual(hashlib.sha256(code[:64]).hexdigest(),
                         fixture["code_image"]["descriptor_sha256_hex"])
        self.assertEqual(hashlib.sha256(code[64:]).hexdigest(),
                         fixture["code_image"]["machine_code_sha256_hex"])
        self.assertEqual(int.from_bytes(code[8:12], "little"), 16)
        self.assertEqual(int.from_bytes(code[16:24], "little"), 64)

        packet = bytes.fromhex(fixture["aql_template"]["bytes_hex"])
        self.assertEqual(len(packet), 64)
        self.assertEqual(hashlib.sha256(packet).hexdigest(),
                         fixture["aql_template"]["sha256_hex"])
        unpacked = struct.unpack("<6H5I4Q", packet)
        self.assertEqual(unpacked[:11],
                         (0x1402, 1, 64, 1, 1, 0, 64, 1, 1, 0, 0))
        self.assertEqual(unpacked[11:], (0, 0, 0, 0))
        self.assertIn("completion_signal remains", fixture["aql_template"]["materialization"])

        kernarg = bytes.fromhex(fixture["kernarg_template"]["bytes_hex"])
        self.assertEqual(kernarg, bytes(16))
        self.assertEqual(hashlib.sha256(kernarg).hexdigest(),
                         fixture["kernarg_template"]["sha256_hex"])

    def test_manifest_is_rebuilt_from_authoritative_components(self) -> None:
        fixture = self.spec["fixture_authority"]
        manifest = fixture["manifest"]
        scalar_values = {
            "manifest_version": fixture["manifest_version"],
            "gfx_ip_decimal": fixture["gfx_ip_decimal"],
            "wavefront_size": fixture["wavefront_size"],
            "active_compute_units": fixture["active_compute_units"],
            "dimensions": fixture["dimensions"],
            "workgroup_x": fixture["workgroup_size"][0],
            "workgroup_y": fixture["workgroup_size"][1],
            "workgroup_z": fixture["workgroup_size"][2],
            "grid_x": fixture["grid_size"][0],
            "grid_y": fixture["grid_size"][1],
            "grid_z": fixture["grid_size"][2],
            "input_offset": fixture["input_offset"],
            "output_offset": fixture["output_offset"],
            "output_bytes": fixture["output_bytes"],
            "xor_byte": fixture["xor_byte"],
        }
        hash_values = {
            "code_image_sha256": fixture["code_image"]["sha256_hex"],
            "aql_template_sha256": fixture["aql_template"]["sha256_hex"],
            "kernarg_template_sha256": fixture["kernarg_template"]["sha256_hex"],
            "golden_input_sha256": fixture["golden_buffers"]["input_sha256_hex"],
            "golden_output_sha256": fixture["golden_buffers"]["output_sha256_hex"],
        }
        encoded = manifest["domain_ascii"].encode("ascii") + bytes([0])
        encoded += b"".join(
            struct.pack(">I", scalar_values[name])
            for name in manifest["scalar_order"]
        )
        encoded += b"".join(
            bytes.fromhex(hash_values[name]) for name in manifest["hash_order"]
        )
        self.assertEqual(len(encoded), manifest["bytes"])
        self.assertEqual(encoded.hex(), manifest["bytes_hex"])
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), manifest["sha256_hex"])

    def test_golden_buffers_are_non_identity_and_hashed(self) -> None:
        fixture = self.spec["fixture_authority"]
        golden = fixture["golden_buffers"]
        source = bytes.fromhex(golden["input_hex"])
        output = bytes.fromhex(golden["output_hex"])
        initial = bytes(fixture["output_bytes"])
        self.assertEqual(source, bytes(range(64)))
        self.assertEqual(output, bytes(value ^ fixture["xor_byte"] for value in source))
        self.assertNotEqual(output, source)
        self.assertNotEqual(output, initial)
        for name, data in (("input", source), ("output", output)):
            self.assertEqual(hashlib.sha256(data).hexdigest(),
                             golden[f"{name}_sha256_hex"])
            self.assertEqual(f"{crc32c(data):08x}", golden[f"{name}_crc32c_hex"])
        self.assertEqual(hashlib.sha256(initial).hexdigest(),
                         golden["output_initial_sha256_hex"])
        self.assertEqual(f"{crc32c(initial):08x}",
                         golden["output_initial_crc32c_hex"])

    def test_materialized_packet_keeps_completion_signal_zero(self) -> None:
        identity = self.spec["golden"]["identity"]
        packet = bytes.fromhex(identity["materialized_aql_hex"])
        unpacked = struct.unpack("<6H5I4Q", packet)
        self.assertEqual(unpacked[11], self.as_hex(identity, "internal_code_va_hex"))
        self.assertEqual(unpacked[12], self.as_hex(identity, "internal_kernarg_va_hex"))
        self.assertEqual(unpacked[13], 0)
        self.assertEqual(unpacked[14], 0)
        self.assertNotIn("internal_native_signal_va_hex", identity)
        self.assertEqual(hashlib.sha256(packet).hexdigest(),
                         identity["materialized_aql_sha256_hex"])
        self.assertEqual(f"{crc32c(packet):08x}", identity["packet_crc32c_hex"])

    def test_golden_frames_rebuild_byte_for_byte(self) -> None:
        frames = self.golden_frames()
        expected_types = {
            "dispatch_request": 11,
            "dispatch_ack": 12,
            "dispatch_completion": 13,
        }
        for name, frame in frames.items():
            golden = self.spec["golden"][name]
            self.assertEqual(frame.hex(), golden["frame_hex"])
            self.assertEqual(len(frame), golden["frame_bytes"])
            self.assertEqual(int.from_bytes(frame[14:16], "big"), expected_types[name])
            self.assertEqual(int.from_bytes(frame[20:24], "big"),
                             golden["payload_bytes"])
            checksum = int(golden["crc32c_hex"], 16)
            self.assertEqual(int.from_bytes(frame[64:68], "big"), checksum)
            zeroed = bytearray(frame)
            zeroed[64:68] = bytes(4)
            self.assertEqual(crc32c(bytes(zeroed)), checksum)
            self.assertEqual(hashlib.sha256(frame).hexdigest(),
                             golden["frame_sha256_hex"])

    def test_ack_is_admission_and_completion_is_real_execution(self) -> None:
        frames = self.golden_frames()
        base = self.base["header"]["bytes"]
        ack = frames["dispatch_ack"]
        completion = frames["dispatch_completion"]
        result = self.oracle.result
        echo_names = [
            "queue_id", "queue_generation", "queue_sequence", "fixture_id",
            "input_allocation_id", "input_generation", "output_allocation_id",
            "output_generation", "signal_id", "signal_generation", "trace_id",
            "input_gpu_va", "output_gpu_va", "packet_crc32c", "admission_tick",
        ]
        for name in echo_names:
            self.assertEqual(
                self.oracle.load_integer(ack, result[name], base),
                self.oracle.load_integer(completion, result[name], base),
            )
        for name in ("output_crc32c", "start_tick", "end_tick", "retire_tick"):
            self.assertEqual(self.oracle.load_integer(ack, result[name], base), 0)
        output_crc = self.oracle.load_integer(completion, result["output_crc32c"], base)
        self.assertEqual(output_crc, 0x796671EC)
        ticks = [
            self.oracle.load_integer(completion, result[name], base)
            for name in ("admission_tick", "start_tick", "end_tick", "retire_tick")
        ]
        self.assertLess(ticks[0], ticks[1])
        self.assertLessEqual(ticks[1], ticks[2])
        self.assertLessEqual(ticks[2], ticks[3])
        self.assertGreater(ticks[3], ticks[0] + 1)
        rules = self.spec["result_rules"]
        self.assertIn("admission, never execution", rules["ack_success"])
        self.assertIn("non-OK DISPATCH_COMPLETION is noncanonical",
                      rules["completion_failure"])

    def test_failure_order_shared_ids_and_two_phase_wait_are_explicit(self) -> None:
        precedence = self.spec["error_precedence"]
        prefixes = [
            "DROP_AND_CLOSE", "MALFORMED", "UNSUPPORTED_VERSION",
            "INSTANCE_MISMATCH", "PROTOCOL_STATE", "TOPOLOGY_MISMATCH",
            "UNSUPPORTED_CAPABILITY", "PROTOCOL_STATE", "PROTOCOL_STATE",
            "PROTOCOL_STATE", "PROTOCOL_STATE", "BUSY", "PROTOCOL_STATE",
            "RESOURCE_EXHAUSTED", "INTERNAL",
        ]
        self.assertEqual(len(precedence), len(prefixes))
        for entry, prefix in zip(precedence, prefixes):
            self.assertTrue(entry.startswith(prefix))
        self.assertIn("shared by queue, memory, signal, and dispatch",
                      self.spec["semantics"]["request_id"])
        self.assertIn("every field from trace_id through retire_tick is zero",
                      self.spec["result_rules"]["ack_failure"])
        wait = self.spec["runtime_wait"]
        self.assertIn("submit call forms one", wait["deadline"])
        self.assertIn("each later wait call forms its own", wait["deadline"])
        self.assertIn("without allocating a request ID or sending", wait["retry"])
        self.assertIn("admission ticket", wait["publication"])
        self.assertIn("request ID", wait["publication"])
        self.assertIn("timeout or cancellation", wait["after_ack"])
        ordering = self.spec["interleave"]["ordering"]
        self.assertIn("tick R+1", ordering)
        self.assertIn("before wire_completion_emitted", ordering)

    def test_trace_requires_cu_provenance_and_forbids_shortcuts(self) -> None:
        events = self.spec["trace_contract"]["required_ordered_events"]
        event_fields = self.spec["trace_contract"]["event_fields"]
        self.assertEqual(list(event_fields), events)
        for event in events:
            self.assertTrue(event_fields[event])
        disconnect = self.spec["trace_contract"]["disconnect_event_fields"]
        self.assertEqual(
            list(disconnect),
            ["dispatch_owner_disconnected", "abandoned_packet_retired"],
        )
        self.assertEqual(
            disconnect["dispatch_owner_disconnected"]["phase"],
            ["admitted_unissued", "issued_task_pending",
             "issued_task_known", "retired_pre_wire"],
        )
        self.assertIs(
            disconnect["dispatch_owner_disconnected"]
                      ["dispatch_completion_emitted"],
            False,
        )
        self.assertIs(
            disconnect["abandoned_packet_retired"]
                      ["cp7_signal_mirrored"],
            False,
        )
        self.assertIn(
            "exactly one later abandoned_packet_retired",
            self.spec["trace_contract"]["disconnect_ordering"],
        )
        disconnect = self.spec["trace_contract"]["disconnect_event_fields"]
        self.assertEqual(
            list(disconnect),
            ["dispatch_owner_disconnected", "abandoned_packet_retired"],
        )
        self.assertEqual(
            disconnect["dispatch_owner_disconnected"]["phase"],
            ["admitted_unissued", "issued_task_pending",
             "issued_task_known", "retired_pre_wire"],
        )
        self.assertIs(
            disconnect["dispatch_owner_disconnected"]
                      ["dispatch_completion_emitted"],
            False,
        )
        self.assertIs(
            disconnect["abandoned_packet_retired"]
                      ["cp7_signal_mirrored"],
            False,
        )
        self.assertIn(
            "exactly one later abandoned_packet_retired",
            self.spec["trace_contract"]["disconnect_ordering"],
        )
        self.assertEqual(events[0], "dispatch_admitted")
        self.assertLess(events.index("hsapp_packet_fetched"),
                        events.index("gpu_command_processor_submitted"))
        self.assertLess(events.index("gpu_dispatcher_started"),
                        events.index("cu_wave_started"))
        self.assertLess(events.index("cu_global_store_completed"),
                        events.index("packet_retired"))
        self.assertLess(events.index("packet_retired"),
                        events.index("cp7_signal_mirrored"))
        self.assertLess(events.index("cp7_signal_mirrored"),
                        events.index("wire_completion_emitted"))
        provenance = self.spec["trace_contract"]["gpu_provenance"]
        self.assertIn("CU ID zero", provenance)
        self.assertIn("zero completion-signal field", provenance)
        self.assertEqual(event_fields["aql_packet_published"]["completion_signal"], 0)
        self.assertEqual(event_fields["cu_wave_started"]["wavefront_size"], 64)
        self.assertEqual(event_fields["wire_completion_emitted"]["signal_completion_tick"],
                         "retire-plus-one-u64")
        shortcut = self.spec["execution_path"]["forbidden_shortcut"]
        self.assertIn("bridge-side byte arithmetic", shortcut)
        self.assertIn("one-tick synthetic completion", shortcut)
        issue_order = self.spec["execution_path"]["issue_order"]
        self.assertIn("only after the complete successful ACK send", issue_order)
        self.assertIn("rolls back the admission and queue sequence", issue_order)
        issue_order = self.spec["execution_path"]["issue_order"]
        self.assertIn("only after the complete successful ACK send", issue_order)
        self.assertIn("rolls back the admission and queue sequence", issue_order)
        forbidden = " ".join(self.spec["forbidden_scope"])
        for phrase in ("generic code-object", "ROCr", "P2", "HIP", "vLLM"):
            self.assertIn(phrase, forbidden)


if __name__ == "__main__":
    unittest.main()
