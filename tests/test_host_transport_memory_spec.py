import json
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MemorySpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(
            (ROOT / "protocol/host-transport-v1-memory.json").read_text()
        )
        cls.base = json.loads(
            (ROOT / "protocol/host-transport-v1.json").read_text()
        )

    def test_extension_is_additive_and_independent(self):
        self.assertEqual(self.spec["base_protocol"], "protocol/host-transport-v1.json")
        self.assertEqual(
            self.spec["queue_protocol"], "protocol/host-transport-v1-queue.json"
        )
        capability = self.spec["capability"]
        self.assertEqual((capability["bit"], capability["mask_hex"]), (2, "04"))
        self.assertIn("independent", capability["negotiation"])
        self.assertEqual(
            self.spec["transport"]["message_types"],
            {"MEMORY_REQUEST": 6, "MEMORY_ACK": 7},
        )
        self.assertEqual(self.spec["transport"]["frame_bytes"], 144)

    def test_payload_layouts_are_exact_and_non_overlapping(self):
        for payload_name in ("request_payload", "ack_payload"):
            payload = self.spec[payload_name]
            end = 0
            for field in payload["fields"]:
                self.assertEqual(field["offset"], end)
                end += field["bytes"]
            self.assertEqual(end, payload["bytes"])
            self.assertEqual(end, 64)

    def test_opcode_shapes_and_fd_cardinality_are_total(self):
        constraints = self.spec["opcode_constraints"]
        self.assertEqual(
            set(constraints),
            {"ALLOC", "FREE", "COPY_H2D", "COPY_D2H", "failure_ack"},
        )
        self.assertEqual(constraints["ALLOC"]["request"]["fd_count"], 0)
        self.assertEqual(constraints["FREE"]["request"]["fd_count"], 0)
        self.assertEqual(constraints["COPY_H2D"]["request"]["fd_count"], 1)
        self.assertEqual(constraints["COPY_D2H"]["request"]["fd_count"], 1)
        self.assertEqual(constraints["COPY_D2H"]["request"]["argument"], 0)
        self.assertEqual(constraints["ALLOC"]["request"]["byte_count"], "nonzero-u64")
        self.assertIn(
            "no-wrap", constraints["COPY_H2D"]["request"]["byte_count"]
        )
        self.assertIn("actual daemon UUID", constraints["failure_ack"])

    def test_ack_contract_is_total_and_poisoning(self):
        rules = self.spec["ack_rules"]
        self.assertEqual(
            set(rules),
            {
                "common", "ALLOC_success", "FREE_success",
                "COPY_H2D_success", "COPY_D2H_success", "failure",
                "runtime_validation",
            },
        )
        self.assertIn("request_id echo", rules["common"])
        self.assertIn("1 through 1024", rules["ALLOC_success"])
        self.assertIn("known nonzero", rules["failure"])
        self.assertIn("poisons and closes", rules["runtime_validation"])

    def test_resource_and_sparse_storage_limits_are_explicit(self):
        limits = self.spec["limits"]
        self.assertEqual(limits["maximum_live_allocations"], 1024)
        self.assertEqual(limits["maximum_single_allocation_bytes"], 2 << 30)
        self.assertEqual(limits["maximum_total_live_bytes"], 4 << 30)
        self.assertEqual(limits["maximum_transfer_bytes"], 16 << 20)
        self.assertEqual(limits["allowed_alignment_bytes"], [4096, 65536])
        self.assertEqual(limits["sparse_chunk_bytes"], 65536)
        self.assertIn("missing chunks read as zero", self.spec["semantics"]["storage"])
        self.assertIn(
            "base + (allocation_id - 1) * slot stride",
            self.spec["semantics"]["simulated_va"],
        )

    def test_carrier_states_and_atomicity_are_mechanical(self):
        carrier = self.spec["carrier"]
        self.assertEqual(
            carrier["COPY_H2D"]["exact_seals"],
            ["F_SEAL_SHRINK", "F_SEAL_GROW", "F_SEAL_WRITE", "F_SEAL_SEAL"],
        )
        self.assertEqual(carrier["COPY_D2H"]["initial_access"], "O_RDWR")
        self.assertEqual(
            carrier["COPY_D2H"]["initial_exact_seals"],
            ["F_SEAL_SHRINK", "F_SEAL_GROW"],
        )
        self.assertEqual(
            carrier["COPY_D2H"]["final_exact_seals"],
            carrier["COPY_H2D"]["exact_seals"],
        )
        self.assertIn("changes no simulated bytes", self.spec["semantics"]["copy_atomicity"])
        self.assertIn("changes no caller bytes", self.spec["semantics"]["copy_atomicity"])
        self.assertIn("re-reads", carrier["COPY_D2H"]["commit"])
        self.assertIn("RESOURCE_EXHAUSTED", carrier["COPY_D2H"]["failure_status"])

    def test_error_precedence_and_indeterminate_ack_are_explicit(self):
        precedence = self.spec["error_precedence"]
        self.assertTrue(precedence[0].startswith("DROP_AND_CLOSE"))
        self.assertTrue(precedence[1].startswith("MALFORMED"))
        self.assertTrue(precedence[2].startswith("UNSUPPORTED_VERSION"))
        self.assertIn("non-increasing request ID", precedence[7])
        self.assertTrue(precedence[-1].startswith("MALFORMED"))
        self.assertIn("poisons and closes", self.spec["semantics"]["indeterminate_ack"])
        self.assertIn("memfd creation", self.spec["semantics"]["deadline"])
        self.assertIn(
            "only successful COPY_D2H",
            self.spec["semantics"]["deadline"],
        )
        self.assertIn(
            "canonical CP-0005 QUEUE_COMPLETION",
            self.spec["semantics"]["queue_completion_interleave"],
        )
        self.assertIn(
            "active memory request_id",
            self.spec["semantics"]["queue_completion_interleave"],
        )
        self.assertIn("carrier-IO", self.spec["semantics"]["request_id"])

    @staticmethod
    def crc32c(data):
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
        return crc ^ 0xFFFFFFFF

    def build_frame(self, message_type, request_id, payload):
        identity = self.spec["golden"]["identity"]
        header = struct.pack(
            ">8sHHHHIIQ16sQQIIQ",
            bytes.fromhex(
                next(
                    field["constant_hex"]
                    for field in self.base["header"]["fields"]
                    if field["name"] == "magic"
                )
            ),
            1,
            0,
            80,
            message_type,
            0,
            len(payload),
            request_id,
            bytes.fromhex(identity["daemon_instance_uuid_hex"]),
            int(identity["connection_id_hex"], 16),
            int(identity["job_epoch_hex"], 16),
            0,
            0,
            0,
        )
        frame = bytearray(header + payload)
        checksum = self.crc32c(frame)
        frame[64:68] = checksum.to_bytes(4, "big")
        return bytes(frame), checksum

    def test_golden_frames_and_carrier_crc_are_independently_valid(self):
        golden = self.spec["golden"]
        identity = golden["identity"]
        carrier = bytes.fromhex(identity["carrier_hex"])
        self.assertEqual(
            self.crc32c(carrier), int(identity["carrier_crc32c_hex"], 16)
        )

        request_format = ">HHHHQQQQQQQ"
        ack_format = ">HHIHHIQQQQQQ"
        allocation_id = int(identity["allocation_id_hex"], 16)
        generation = int(identity["generation_hex"], 16)
        simulated_va = int(identity["simulated_va_hex"], 16)
        size = identity["size_bytes"]
        alignment = identity["alignment_bytes"]
        offset = identity["copy_offset"]
        count = len(carrier)
        carrier_crc = int(identity["carrier_crc32c_hex"], 16)
        request_ids = {
            "alloc": 0x0123456789ABCDF1,
            "h2d": 0x0123456789ABCDF2,
            "d2h": 0x0123456789ABCDF3,
            "free": 0x0123456789ABCDF4,
        }
        ticks = {
            "alloc": 0x123456789ABCDEF2,
            "h2d": 0x123456789ABCDEF3,
            "d2h": 0x123456789ABCDEF4,
            "free": 0x123456789ABCDEF5,
        }
        expected_payloads = {
            "alloc_request": struct.pack(
                request_format, 1, 0, 1, 0, 0, 0, 0, size, alignment, 0, 0
            ),
            "alloc_ack": struct.pack(
                ack_format, 1, 0, 0, 1, 0, 0, allocation_id, generation,
                simulated_va, size, alignment, ticks["alloc"],
            ),
            "h2d_request": struct.pack(
                request_format, 1, 0, 3, 0, allocation_id, generation,
                offset, count, carrier_crc, 0, 0,
            ),
            "h2d_ack": struct.pack(
                ack_format, 1, 0, 0, 3, 0, 0, allocation_id, generation,
                offset, count, carrier_crc, ticks["h2d"],
            ),
            "d2h_request": struct.pack(
                request_format, 1, 0, 4, 0, allocation_id, generation,
                offset, count, 0, 0, 0,
            ),
            "d2h_ack": struct.pack(
                ack_format, 1, 0, 0, 4, 0, 0, allocation_id, generation,
                offset, count, carrier_crc, ticks["d2h"],
            ),
            "free_request": struct.pack(
                request_format, 1, 0, 2, 0, allocation_id, generation,
                0, 0, 0, 0, 0,
            ),
            "free_ack": struct.pack(
                ack_format, 1, 0, 0, 2, 0, 0, allocation_id, generation,
                0, 0, 0, ticks["free"],
            ),
        }
        request_for = {
            "alloc_request": "alloc", "alloc_ack": "alloc",
            "h2d_request": "h2d", "h2d_ack": "h2d",
            "d2h_request": "d2h", "d2h_ack": "d2h",
            "free_request": "free", "free_ack": "free",
        }
        frames = {}
        for name, payload in expected_payloads.items():
            message_type = 6 if name.endswith("request") else 7
            expected, checksum = self.build_frame(
                message_type, request_ids[request_for[name]], payload
            )
            entry = golden[name]
            self.assertEqual(entry["frame_bytes"], 144)
            self.assertEqual(entry["crc32c_hex"], f"{checksum:08x}")
            self.assertEqual(bytes.fromhex(entry["frame_hex"]), expected)
            self.assertEqual(expected[68:80], bytes(12))
            frames[name] = expected

        alloc_request = struct.unpack(request_format, frames["alloc_request"][80:])
        alloc_ack = struct.unpack(ack_format, frames["alloc_ack"][80:])
        h2d_request = struct.unpack(request_format, frames["h2d_request"][80:])
        h2d_ack = struct.unpack(ack_format, frames["h2d_ack"][80:])
        d2h_request = struct.unpack(request_format, frames["d2h_request"][80:])
        d2h_ack = struct.unpack(ack_format, frames["d2h_ack"][80:])
        free_request = struct.unpack(request_format, frames["free_request"][80:])
        free_ack = struct.unpack(ack_format, frames["free_ack"][80:])
        self.assertEqual(alloc_request[2], 1)
        self.assertEqual(alloc_ack[6], 7)
        expected_va = int(self.spec["limits"]["simulated_va_base_hex"], 16) + (
            alloc_ack[6] - 1
        ) * self.spec["limits"]["simulated_va_slot_stride_bytes"]
        self.assertEqual(alloc_ack[8], expected_va)
        self.assertEqual((alloc_request[8], alloc_ack[10]), (65536, 65536))
        self.assertEqual((h2d_request[2], h2d_ack[3], d2h_ack[3]), (3, 3, 4))
        self.assertEqual(h2d_request[8], int(identity["carrier_crc32c_hex"], 16))
        self.assertEqual(h2d_ack[10], h2d_request[8])
        self.assertEqual(d2h_ack[10], h2d_request[8])
        self.assertEqual(d2h_request[8:], (0, 0, 0))
        self.assertEqual(free_request[2:], (2, 0, 7, generation, 0, 0, 0, 0, 0))
        self.assertEqual(free_ack[3:], (2, 0, 0, 7, generation, 0, 0, 0, ticks["free"]))


if __name__ == "__main__":
    unittest.main()
