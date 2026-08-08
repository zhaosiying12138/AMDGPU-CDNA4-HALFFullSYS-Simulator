#!/usr/bin/env python3
"""Independent checks for the host-transport signal/event v1 contract."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "protocol/host-transport-v1.json"
QUEUE_PATH = ROOT / "protocol/host-transport-v1-queue.json"
SIGNAL_PATH = ROOT / "protocol/host-transport-v1-signal.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


class WireOracle:
    """Construct signal frames from the published field tables only."""

    def __init__(self, base: dict[str, Any], signal: dict[str, Any]) -> None:
        self.base = base
        self.signal = signal
        self.header = {field["name"]: field for field in base["header"]["fields"]}
        self.request = {
            field["name"]: field for field in signal["request_payload"]["fields"]
        }
        self.result = {
            field["name"]: field for field in signal["result_payload"]["fields"]
        }
        self.messages = signal["transport"]["message_types"]
        self.opcodes = self.request["opcode"]["values"]
        self.conditions = self.request["condition"]["values"]

    @staticmethod
    def crc32c(data: bytes) -> int:
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
        return crc ^ 0xFFFFFFFF

    @staticmethod
    def _store_integer(target: bytearray, field: dict[str, Any], value: int) -> None:
        size = int(field["bytes"])
        modulus = 1 << (size * 8)
        offset = int(field["offset"])
        target[offset:offset + size] = (value % modulus).to_bytes(size, "big")

    @staticmethod
    def _store_bytes(target: bytearray, field: dict[str, Any], value: bytes) -> None:
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

    def _payload(
        self, definition: dict[str, dict[str, Any]], values: dict[str, int]
    ) -> bytes:
        payload = bytearray(int(self.signal["transport"]["payload_bytes"]))
        for name, field in definition.items():
            if "constant" in field:
                self._store_integer(payload, field, int(field["constant"]))
            elif name in values:
                self._store_integer(payload, field, int(values[name]))
        return bytes(payload)

    def _frame(
        self,
        message_name: str,
        request_id: int,
        daemon_uuid: bytes,
        connection_id: int,
        epoch: int,
        payload: bytes,
    ) -> bytes:
        header = bytearray(int(self.base["header"]["bytes"]))
        supplied: dict[str, int] = {
            "message_type": int(self.messages[message_name]),
            "payload_bytes": len(payload),
            "request_id": request_id,
            "connection_id": connection_id,
            "job_epoch": epoch,
            "crc32c": 0,
        }
        for name, field in self.header.items():
            if "constant_hex" in field:
                self._store_bytes(header, field, bytes.fromhex(field["constant_hex"]))
            elif name == "daemon_instance_uuid":
                self._store_bytes(header, field, daemon_uuid)
            elif "constant" in field:
                self._store_integer(header, field, int(field["constant"]))
            elif name in supplied:
                self._store_integer(header, field, supplied[name])
        frame = header + payload
        checksum = self.header["crc32c"]
        self._store_integer(frame, checksum, self.crc32c(bytes(frame)))
        return bytes(frame)

    def request_frame(
        self,
        opcode_name: str,
        request_id: int,
        daemon_uuid: bytes,
        connection_id: int,
        epoch: int,
        *,
        signal_id: int,
        generation: int,
        sequence: int,
        value: int,
        condition: int,
    ) -> bytes:
        payload = self._payload(
            self.request,
            {
                "opcode": int(self.opcodes[opcode_name]),
                "signal_id": signal_id,
                "generation": generation,
                "sequence": sequence,
                "value": value,
                "condition": condition,
            },
        )
        return self._frame(
            "SIGNAL_REQUEST", request_id, daemon_uuid, connection_id, epoch, payload
        )

    def result_frame(
        self,
        message_name: str,
        opcode_name: str,
        request_id: int,
        daemon_uuid: bytes,
        connection_id: int,
        epoch: int,
        *,
        status: int,
        signal_id: int,
        generation: int,
        sequence: int,
        value: int,
        ready: int,
        sim_tick: int,
    ) -> bytes:
        payload = self._payload(
            self.result,
            {
                "status": status,
                "opcode": int(self.opcodes[opcode_name]),
                "signal_id": signal_id,
                "generation": generation,
                "sequence": sequence,
                "value": value,
                "ready": ready,
                "sim_tick": sim_tick,
            },
        )
        return self._frame(
            message_name, request_id, daemon_uuid, connection_id, epoch, payload
        )


class HostTransportSignalSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = load_json(BASE_PATH)
        cls.queue = load_json(QUEUE_PATH)
        cls.signal = load_json(SIGNAL_PATH)
        cls.oracle = WireOracle(cls.base, cls.signal)

    @staticmethod
    def _hex(identity: dict[str, Any], name: str) -> int:
        return int(identity[name], 16)

    def _golden_frames(self) -> dict[str, bytes]:
        identity = self.signal["golden"]["identity"]
        daemon = bytes.fromhex(identity["daemon_instance_uuid_hex"])
        connection = self._hex(identity, "connection_id_hex")
        epoch = self._hex(identity, "job_epoch_hex")
        signal_id = self._hex(identity, "signal_id_hex")
        generation = self._hex(identity, "generation_hex")
        sequence = self._hex(identity, "sequence_hex")
        initial = int(identity["initial_value"])
        compare = int(identity["wait_compare_value"])
        stored = int(identity["stored_value"])
        condition = int(self.oracle.conditions[identity["wait_condition"]])
        ok = int(self.base["statuses"]["OK"])

        create_request = self._hex(identity, "create_request_id_hex")
        load_request = self._hex(identity, "load_request_id_hex")
        wait_request = self._hex(identity, "wait_request_id_hex")
        store_request = self._hex(identity, "store_request_id_hex")
        destroy_request = self._hex(identity, "destroy_request_id_hex")

        common = (daemon, connection, epoch)
        return {
            "create_request": self.oracle.request_frame(
                "CREATE", create_request, *common, signal_id=0, generation=0,
                sequence=0, value=initial, condition=0
            ),
            "create_ack": self.oracle.result_frame(
                "SIGNAL_ACK", "CREATE", create_request, *common, status=ok,
                signal_id=signal_id, generation=generation, sequence=0,
                value=initial, ready=0,
                sim_tick=self._hex(identity, "create_tick_hex")
            ),
            "load_request": self.oracle.request_frame(
                "LOAD", load_request, *common, signal_id=signal_id,
                generation=generation, sequence=0, value=0, condition=0
            ),
            "load_ack": self.oracle.result_frame(
                "SIGNAL_ACK", "LOAD", load_request, *common, status=ok,
                signal_id=signal_id, generation=generation, sequence=0,
                value=initial, ready=0,
                sim_tick=self._hex(identity, "load_tick_hex")
            ),
            "wait_request": self.oracle.request_frame(
                "WAIT", wait_request, *common, signal_id=signal_id,
                generation=generation, sequence=sequence, value=compare,
                condition=condition
            ),
            "wait_ack": self.oracle.result_frame(
                "SIGNAL_ACK", "WAIT", wait_request, *common, status=ok,
                signal_id=signal_id, generation=generation, sequence=sequence,
                value=initial, ready=0,
                sim_tick=self._hex(identity, "wait_tick_hex")
            ),
            "store_request": self.oracle.request_frame(
                "STORE", store_request, *common, signal_id=signal_id,
                generation=generation, sequence=0, value=stored, condition=0
            ),
            "store_ack": self.oracle.result_frame(
                "SIGNAL_ACK", "STORE", store_request, *common, status=ok,
                signal_id=signal_id, generation=generation, sequence=0,
                value=stored, ready=0,
                sim_tick=self._hex(identity, "store_tick_hex")
            ),
            "wait_completion": self.oracle.result_frame(
                "SIGNAL_COMPLETION", "WAIT", wait_request, *common, status=ok,
                signal_id=signal_id, generation=generation, sequence=sequence,
                value=stored, ready=0,
                sim_tick=self._hex(identity, "completion_tick_hex")
            ),
            "destroy_request": self.oracle.request_frame(
                "DESTROY", destroy_request, *common, signal_id=signal_id,
                generation=generation, sequence=0, value=0, condition=0
            ),
            "destroy_ack": self.oracle.result_frame(
                "SIGNAL_ACK", "DESTROY", destroy_request, *common, status=ok,
                signal_id=signal_id, generation=generation, sequence=0,
                value=0, ready=0,
                sim_tick=self._hex(identity, "destroy_tick_hex")
            ),
        }

    def test_schema_transport_capability_and_status_authority(self) -> None:
        self.assertEqual(
            self.signal["schema"], "amdgpu-sim.host-transport-v1.signal.v1"
        )
        transport = self.signal["transport"]
        self.assertEqual(transport["header_bytes"], self.base["header"]["bytes"])
        self.assertEqual(transport["payload_bytes"], 64)
        self.assertEqual(
            transport["frame_bytes"],
            transport["header_bytes"] + transport["payload_bytes"],
        )
        self.assertEqual(
            transport["message_types"],
            {"SIGNAL_REQUEST": 8, "SIGNAL_ACK": 9, "SIGNAL_COMPLETION": 10},
        )
        self.assertEqual(transport["ancillary_descriptors"], 0)
        capability = self.signal["capability"]
        self.assertEqual(capability["bit"], 3)
        self.assertEqual(capability["byte_index"], capability["bit"] // 8)
        self.assertEqual(int(capability["mask_hex"], 16), 1 << (capability["bit"] % 8))
        self.assertEqual(
            self.base["statuses"],
            {
                "OK": 0,
                "MALFORMED": 1,
                "UNSUPPORTED_VERSION": 2,
                "UNSUPPORTED_CAPABILITY": 3,
                "INSTANCE_MISMATCH": 4,
                "TOPOLOGY_MISMATCH": 5,
                "UNAUTHORIZED": 6,
                "BUSY": 7,
                "RESOURCE_EXHAUSTED": 8,
                "PROTOCOL_STATE": 9,
                "INTERNAL": 10,
            },
        )

    def test_payload_layouts_are_contiguous_complete_and_unique(self) -> None:
        expected_names = {
            "request_payload": [
                "signal_major", "signal_minor", "opcode", "flags",
                "signal_id", "generation", "sequence", "value", "condition",
                "reserved0", "reserved1",
            ],
            "result_payload": [
                "signal_major", "signal_minor", "status", "opcode",
                "reserved_opcode", "reserved_status", "signal_id", "generation",
                "sequence", "value", "ready", "sim_tick",
            ],
        }
        for payload_name, names in expected_names.items():
            definition = self.signal[payload_name]
            fields = definition["fields"]
            self.assertEqual([field["name"] for field in fields], names)
            cursor = 0
            for field in fields:
                self.assertEqual(field["offset"], cursor)
                self.assertIn(field["bytes"], (2, 4, 8))
                cursor += field["bytes"]
            self.assertEqual(cursor, definition["bytes"])
            self.assertEqual(cursor, self.signal["transport"]["payload_bytes"])
            self.assertEqual(len(names), len(set(names)))

    def test_opcode_condition_and_canonical_request_matrix(self) -> None:
        self.assertEqual(
            self.oracle.opcodes,
            {"CREATE": 1, "DESTROY": 2, "LOAD": 3, "STORE": 4, "WAIT": 5},
        )
        self.assertEqual(
            self.oracle.conditions, {"EQ": 0, "NE": 1, "LT": 2, "GTE": 3}
        )
        constraints = self.signal["opcode_constraints"]
        self.assertEqual(
            set(constraints),
            {"CREATE", "DESTROY", "LOAD", "STORE", "WAIT", "failure_ack"},
        )
        self.assertEqual(
            constraints["CREATE"]["request"],
            {
                "signal_id": 0,
                "generation": 0,
                "sequence": 0,
                "value": "arbitrary-i64-bits",
                "condition": 0,
            },
        )
        for opcode in ("DESTROY", "LOAD"):
            request = constraints[opcode]["request"]
            self.assertEqual(request["sequence"], 0)
            self.assertEqual(request["value"], 0)
            self.assertEqual(request["condition"], 0)
        self.assertEqual(constraints["STORE"]["request"]["sequence"], 0)
        self.assertEqual(constraints["STORE"]["request"]["condition"], 0)
        self.assertEqual(
            constraints["WAIT"]["request"]["condition"], "integer-0-through-3"
        )
        self.assertIn("value, ready, and sim_tick are zero", constraints["failure_ack"])

    def test_limits_generation_and_outbound_accounting_are_bounded(self) -> None:
        limits = self.signal["limits"]
        self.assertEqual(limits["maximum_live_signals"], 1024)
        self.assertEqual(limits["minimum_signal_id"], 1)
        self.assertEqual(limits["maximum_signal_id"], 1024)
        self.assertEqual(limits["maximum_pending_waits"], 8)
        self.assertEqual(limits["maximum_pending_waits_per_signal"], 1)
        self.assertEqual(limits["completion_latency_ticks"], 1)
        self.assertEqual(
            limits["maximum_outbound_records"],
            2 * self.queue["limits"]["maximum_inflight_commands"]
            + limits["maximum_pending_waits"]
            + 1,
        )
        self.assertEqual(limits["maximum_outbound_records"], 25)
        semantics = self.signal["semantics"]
        self.assertIn("lowest-free", semantics["signal_identity"])
        self.assertIn("never wraps", semantics["signal_identity"])
        self.assertIn("do not advance", semantics["wait_resources"])
        self.assertIn("complete completion frame is sent", semantics["wait_resources"])
        self.assertIn("UINT64_MAX", semantics["wait_admission"])
        self.assertIn("unchanged", semantics["store_overflow"])

    def test_sim_tick_rules_separate_opaque_ack_ticks_from_known_correlations(self) -> None:
        rules = self.signal["sim_tick_rules"]
        self.assertEqual(
            set(rules),
            {
                "opaque_lifecycle_ack",
                "store_ack",
                "wait_ack",
                "immediate_completion",
                "armed_completion",
                "failure_ack",
            },
        )
        self.assertIn("opaque", rules["opaque_lifecycle_ack"])
        self.assertIn("including UINT64_MAX", rules["opaque_lifecycle_ack"])
        self.assertIn("does not predict", rules["wait_ack"])
        self.assertIn("below UINT64_MAX", rules["wait_ack"])
        self.assertIn("satisfying STORE ACK", rules["armed_completion"])
        self.assertIn("WAIT ACK", rules["immediate_completion"])
        self.assertIn("sim_tick is validated only by sim_tick_rules", self.signal["ack_rules"]["runtime_validation"])
        for opcode in ("CREATE", "LOAD", "DESTROY"):
            self.assertIn("opaque", self.signal["opcode_constraints"][opcode]["success_ack"]["sim_tick"])
        self.assertIn("opaque", self.signal["opcode_constraints"]["STORE"]["success_ack"]["sim_tick"])
        self.assertIn("strictly-below-UINT64_MAX", self.signal["opcode_constraints"]["WAIT"]["success_ack"]["sim_tick"])

    def test_error_precedence_and_failure_ack_zero_policy(self) -> None:
        expected_prefixes = [
            "DROP_AND_CLOSE", "MALFORMED", "UNSUPPORTED_VERSION",
            "INSTANCE_MISMATCH", "PROTOCOL_STATE connection",
            "TOPOLOGY_MISMATCH", "UNSUPPORTED_CAPABILITY",
            "PROTOCOL_STATE non-increasing", "PROTOCOL_STATE stale", "BUSY",
            "RESOURCE_EXHAUSTED", "INTERNAL",
        ]
        precedence = self.signal["error_precedence"]
        self.assertEqual(len(precedence), len(expected_prefixes))
        for entry, prefix in zip(precedence, expected_prefixes):
            self.assertTrue(entry.startswith(prefix), entry)

        identity = self.signal["golden"]["identity"]
        frame = self.oracle.result_frame(
            "SIGNAL_ACK",
            "WAIT",
            self._hex(identity, "wait_request_id_hex"),
            bytes.fromhex(identity["daemon_instance_uuid_hex"]),
            self._hex(identity, "connection_id_hex"),
            self._hex(identity, "job_epoch_hex"),
            status=int(self.base["statuses"]["BUSY"]),
            signal_id=self._hex(identity, "signal_id_hex"),
            generation=self._hex(identity, "generation_hex"),
            sequence=self._hex(identity, "sequence_hex"),
            value=0,
            ready=0,
            sim_tick=0,
        )
        payload_base = int(self.signal["transport"]["header_bytes"])
        for name in ("value", "ready", "sim_tick"):
            self.assertEqual(
                self.oracle.load_integer(frame, self.oracle.result[name], payload_base), 0
            )
        self.assertEqual(
            self.oracle.load_integer(frame, self.oracle.result["signal_id"], payload_base),
            self._hex(identity, "signal_id_hex"),
        )

    def test_signed_predicates_cover_edges_without_arithmetic(self) -> None:
        conditions = self.oracle.conditions

        def matches(name: str, observed: int, compare: int) -> bool:
            operation = conditions[name]
            if operation == conditions["EQ"]:
                return observed == compare
            if operation == conditions["NE"]:
                return observed != compare
            if operation == conditions["LT"]:
                return observed < compare
            if operation == conditions["GTE"]:
                return observed >= compare
            raise AssertionError(operation)

        minimum = -(1 << 63)
        maximum = (1 << 63) - 1
        cases = [
            ("EQ", minimum, minimum, True),
            ("EQ", maximum, minimum, False),
            ("NE", -1, 0, True),
            ("NE", maximum, maximum, False),
            ("LT", minimum, maximum, True),
            ("LT", -1, 0, True),
            ("LT", maximum, minimum, False),
            ("GTE", maximum, minimum, True),
            ("GTE", -1, 0, False),
            ("GTE", minimum, minimum, True),
        ]
        for name, observed, compare, expected in cases:
            with self.subTest(name=name, observed=observed, compare=compare):
                self.assertEqual(matches(name, observed, compare), expected)
        self.assertEqual(
            self.signal["condition_rules"]["value_domain"],
            "signed 64-bit two's-complement",
        )

    def test_all_eleven_goldens_are_independently_reconstructed(self) -> None:
        rebuilt = self._golden_frames()
        self.assertEqual(len(rebuilt), 11)
        checksum = self.oracle.header["crc32c"]
        for name, frame in rebuilt.items():
            with self.subTest(frame=name):
                golden = self.signal["golden"][name]
                expected = bytes.fromhex(golden["frame_hex"])
                self.assertEqual(frame, expected)
                self.assertEqual(len(frame), golden["frame_bytes"])
                stored_crc = self.oracle.load_integer(frame, checksum)
                self.assertEqual(stored_crc, int(golden["crc32c_hex"], 16))
                zeroed = bytearray(frame)
                self.oracle._store_integer(zeroed, checksum, 0)
                self.assertEqual(self.oracle.crc32c(bytes(zeroed)), stored_crc)

    def test_golden_wait_is_armed_then_store_completes_one_tick_later(self) -> None:
        frames = self._golden_frames()
        identity = self.signal["golden"]["identity"]
        base = int(self.signal["transport"]["header_bytes"])
        result = self.oracle.result
        header = self.oracle.header
        wait_ack = frames["wait_ack"]
        store_ack = frames["store_ack"]
        completion = frames["wait_completion"]

        self.assertEqual(self.oracle.load_integer(wait_ack, result["ready"], base), 0)
        self.assertEqual(
            self.oracle.load_integer(wait_ack, result["value"], base),
            int(identity["initial_value"]) % (1 << 64),
        )
        self.assertEqual(
            self.oracle.load_integer(store_ack, result["value"], base),
            int(identity["stored_value"]),
        )
        self.assertEqual(
            self.oracle.load_integer(completion, result["value"], base),
            int(identity["stored_value"]),
        )
        self.assertEqual(self.oracle.load_integer(completion, result["ready"], base), 0)
        self.assertEqual(
            self.oracle.load_integer(completion, result["sim_tick"], base),
            self.oracle.load_integer(store_ack, result["sim_tick"], base)
            + self.signal["limits"]["completion_latency_ticks"],
        )
        self.assertEqual(
            self.oracle.load_integer(completion, header["request_id"]),
            self._hex(identity, "wait_request_id_hex"),
        )

    def test_retry_interleave_poison_and_forbidden_scope_are_explicit(self) -> None:
        runtime = self.signal["runtime_wait"]
        self.assertIn("poisons", runtime["before_ack"])
        self.assertIn("session reusable", runtime["after_ack"])
        self.assertIn("without allocating a request ID", runtime["retry"])
        self.assertIn("before consuming", runtime["deadline"])
        interleave = self.signal["interleave"]
        self.assertIn("previously acknowledged", interleave["allowed"])
        self.assertIn("active request_id before its ACK", interleave["forbidden"])
        self.assertIn("leaves it pending", interleave["buffer_retry"])
        self.assertEqual(
            set(self.signal["forbidden_scope"]),
            {
                "separate KFD event handles",
                "signal arithmetic or compare-and-swap operations",
                "GPU-visible signal memory",
                "queue or packet completion linkage",
                "GPU packet submission",
                "code objects",
                "kernel execution",
                "collectives",
            },
        )
        self.assertIn("no separate event handle", self.signal["semantics"]["event_handle"])


if __name__ == "__main__":
    unittest.main()
