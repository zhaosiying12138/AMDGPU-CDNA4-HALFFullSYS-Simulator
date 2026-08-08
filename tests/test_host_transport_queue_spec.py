import json
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class QueueSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(
            (ROOT / "protocol/host-transport-v1-queue.json").read_text()
        )

    def test_extension_is_additive(self):
        self.assertEqual(self.spec["base_protocol"], "protocol/host-transport-v1.json")
        self.assertEqual(self.spec["transport"]["header_bytes"], 80)
        self.assertEqual(self.spec["transport"]["payload_bytes"], 64)
        self.assertEqual(self.spec["capability"]["bit"], 1)
        self.assertEqual(self.spec["capability"]["mask_hex"], "02")

    def test_message_types_are_unique_and_stable(self):
        types = self.spec["transport"]["message_types"]
        self.assertEqual(types, {
            "QUEUE_REQUEST": 3,
            "QUEUE_ACK": 4,
            "QUEUE_COMPLETION": 5,
        })
        self.assertEqual(len(set(types.values())), len(types))

    def test_request_layout_is_non_overlapping(self):
        fields = self.spec["request_payload"]["fields"]
        self.assertEqual(fields[0]["offset"], 0)
        end = 0
        for field in fields:
            self.assertEqual(field["offset"], end)
            end += field["bytes"]
        self.assertEqual(end, self.spec["request_payload"]["bytes"])

    def test_result_layout_has_explicit_reserved_words(self):
        fields = self.spec["result_payload"]["fields"]
        end = 0
        for field in fields:
            self.assertEqual(field["offset"], end)
            end += field["bytes"]
        self.assertEqual(end, self.spec["result_payload"]["bytes"])
        self.assertEqual(fields[4]["name"], "reserved_opcode")
        self.assertEqual(fields[4]["constant"], 0)
        self.assertEqual(fields[5]["name"], "reserved_status")
        self.assertEqual(fields[5]["constant"], 0)

    def test_limits_and_forbidden_scope(self):
        limits = self.spec["limits"]
        self.assertEqual(limits["maximum_queues"], 8)
        self.assertEqual(limits["maximum_depth"], 64)
        self.assertEqual(limits["maximum_inflight_commands"], 8)
        self.assertIn("request_id", self.spec["semantics"]["completion"])
        self.assertIn("strictly increasing", self.spec["semantics"]["request_id"])
        self.assertIn("without wrap", self.spec["semantics"]["request_id"])
        self.assertIn("poisons", self.spec["semantics"]["indeterminate_ack"])
        self.assertIn("retried", self.spec["semantics"]["completion_retry"])
        self.assertIn("buffered", self.spec["semantics"]["completion_retry"])
        forbidden = set(self.spec["forbidden_scope"])
        self.assertIn("memory allocation", forbidden)
        self.assertIn("GPU packet submission", forbidden)
        self.assertIn("kernel execution", forbidden)

    def test_opcode_constraints_are_total(self):
        constraints = self.spec["opcode_constraints"]
        self.assertEqual(set(constraints), {
            "CREATE", "DESTROY", "DOORBELL", "failure_ack"
        })
        create = constraints["CREATE"]["request"]
        self.assertEqual(
            (create["queue_id"], create["generation"], create["sequence"],
             create["arg1"]),
            (0, 0, 0, 0),
        )
        destroy = constraints["DESTROY"]["request"]
        self.assertEqual(
            (destroy["sequence"], destroy["arg0"], destroy["arg1"]),
            (0, 0, 0),
        )
        doorbell = constraints["DOORBELL"]["request"]
        self.assertEqual(doorbell["arg1"], 0)
        self.assertEqual(
            constraints["DOORBELL"]["success_ack"]["value"], 0
        )
        self.assertEqual(
            constraints["DOORBELL"]["error_completion"],
            {
                "command_kind": 2,
                "status": "INTERNAL",
                "queue_id": "request-echo",
                "generation": "request-echo",
                "sequence": "request-echo",
                "value": 2,
                "error_code": 1,
                "sim_tick": "admission-tick-plus-one",
            },
        )
        self.assertIn("actual daemon UUID", constraints["failure_ack"])

    def test_error_precedence_is_explicit(self):
        precedence = self.spec["error_precedence"]
        self.assertTrue(precedence[0].startswith("DROP_AND_CLOSE"))
        self.assertTrue(precedence[1].startswith("MALFORMED"))
        self.assertTrue(precedence[2].startswith("UNSUPPORTED_VERSION"))
        self.assertIn("request ID", precedence[-2])
        self.assertIn(
            "tracker advances only after", self.spec["semantics"]["request_id"]
        )

    @staticmethod
    def crc32c(data):
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
        return crc ^ 0xFFFFFFFF

    def test_golden_frames_are_independently_valid(self):
        golden = self.spec["golden"]
        identity = golden["identity"]
        frames = []
        for name, message_type in (
            ("doorbell_request", 3),
            ("doorbell_ack", 4),
            ("doorbell_completion", 5),
        ):
            entry = golden[name]
            frame = bytearray.fromhex(entry["frame_hex"])
            self.assertEqual(len(frame), entry["frame_bytes"])
            self.assertEqual(int.from_bytes(frame[14:16], "big"), message_type)
            self.assertEqual(int.from_bytes(frame[20:24], "big"), 64)
            self.assertEqual(frame[24:32].hex(), identity["request_id_hex"])
            expected_crc = int(entry["crc32c_hex"], 16)
            self.assertEqual(int.from_bytes(frame[64:68], "big"), expected_crc)
            frame[64:68] = bytes(4)
            self.assertEqual(self.crc32c(frame), expected_crc)
            frames.append(bytes.fromhex(entry["frame_hex"]))

        request = struct.unpack(">HHHHQQQQQQQ", frames[0][80:])
        ack = struct.unpack(">HHIHHIQQQQQQ", frames[1][80:])
        completion = struct.unpack(">HHIHHIQQQQQQ", frames[2][80:])
        self.assertEqual(request[2], 3)
        self.assertEqual((ack[3], completion[3]), (3, 3))
        self.assertEqual((ack[9], completion[9]), (0, identity["command_kind"]))
        self.assertEqual(completion[11], ack[11] + 1)


if __name__ == "__main__":
    unittest.main()
