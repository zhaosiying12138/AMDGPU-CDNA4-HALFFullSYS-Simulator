#!/usr/bin/env python3
"""Independent oracle checks for the CP-0013 A1 code-object envelope."""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocol" / "host-transport-v1-codeobj-transport.json"
BASE_PATH = ROOT / "protocol" / "host-transport-v1.json"
ABI_PATH = ROOT / "protocol" / "host-transport-v1-codeobj.json"
DOC_PATH = ROOT / "docs" / "host-transport-v1-codeobj-transport.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def field_map(fields: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {field["name"]: field for field in fields}


def assert_contiguous(testcase: unittest.TestCase,
                      fields: list[dict[str, Any]], total: int) -> None:
    end = 0
    for field in fields:
        testcase.assertEqual(field["offset"], end, field["name"])
        testcase.assertGreater(field["bytes"], 0, field["name"])
        end += field["bytes"]
    testcase.assertEqual(end, total)


class CodeObjectTransportSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_json(SPEC_PATH)
        cls.base = load_json(BASE_PATH)
        cls.abi = load_json(ABI_PATH)

    def test_scope_keeps_source_authority_additive(self) -> None:
        self.assertEqual(
            self.spec["schema"],
            "amdgpu-sim.host-transport-v1.codeobj-transport.v1",
        )
        self.assertEqual(self.spec["checkpoint"], "CP-0013")
        self.assertEqual(self.spec["lane"], "P3-CODEOBJ-03")
        self.assertEqual(self.spec["envelope_version"], {"major": 1, "minor": 0})
        self.assertEqual(
            self.spec["base_protocol"], "protocol/host-transport-v1.json"
        )
        self.assertEqual(
            self.spec["code_object_authority"],
            "protocol/host-transport-v1-codeobj.json",
        )
        self.assertEqual(self.abi["capability"]["bit"], 6)
        self.assertEqual(self.abi["capability"]["mask_hex"], "40")
        self.assertEqual(self.abi["transport"]["message_types"], {})

    def test_capability_and_message_ids_are_unused_and_non_overlapping(self) -> None:
        capability = self.spec["capability"]
        self.assertEqual(
            (capability["name"], capability["bit"], capability["mask_hex"]),
            ("CODE_OBJECT_TRANSPORT_V1", 7, "80"),
        )
        self.assertEqual(capability["byte_index"], 0)
        self.assertIn("TOPOLOGY_IDENTITY_V1", capability["requires"])
        self.assertIn("bit 6", capability["negotiation"])

        used = {
            int(message["value"])
            for message in self.base["messages"].values()
        }
        new_ids = list(self.spec["transport"]["message_types"].values())
        self.assertEqual(new_ids, [16, 17])
        self.assertEqual(len(set(new_ids)), len(new_ids))
        self.assertTrue(set(new_ids).isdisjoint(used))
        self.assertEqual(
            self.spec["transport"]["reserved_message_types"],
            {
                "KERNEL_DISPATCH_REQUEST": 18,
                "KERNEL_DISPATCH_ACK": 19,
                "KERNEL_DISPATCH_COMPLETION": 20,
            },
        )
        self.assertIn("no A1 payload", self.spec["transport"]["reserved_message_type_policy"])
        self.assertEqual(self.spec["transport"]["ancillary_descriptors"], 0)

    def test_fixed_record_and_request_prefix_layout(self) -> None:
        transport = self.spec["transport"]
        self.assertEqual(transport["header_bytes"], self.base["header"]["bytes"])
        self.assertEqual(transport["record_bytes"], 4096)
        self.assertEqual(transport["payload_bytes"], 4016)
        self.assertEqual(
            transport["record_bytes"],
            transport["header_bytes"] + transport["payload_bytes"],
        )
        self.assertEqual(transport["chunk_data_offset"], 48)
        self.assertEqual(transport["chunk_data_bytes"], 3968)
        assert_contiguous(self, self.spec["request_prefix"]["fields"], 48)
        prefix = field_map(self.spec["request_prefix"]["fields"])
        self.assertEqual(prefix["major"]["constant"], 1)
        self.assertEqual(prefix["minor"]["constant"], 0)
        self.assertEqual(prefix["flags"]["constant"], 0)
        self.assertEqual(prefix["reserved"]["constant"], 0)
        for name in ("object_id", "object_generation", "image_offset"):
            self.assertEqual(prefix[name]["type"], "u64")
            self.assertEqual(prefix[name]["bytes"], 8)
        serialized = json.dumps(self.spec["request_prefix"]).lower()
        self.assertNotIn("pointer", serialized)
        self.assertNotIn("file descriptor", serialized)

    def test_begin_manifest_and_segment_record_offsets(self) -> None:
        begin = self.spec["begin_payload"]
        self.assertEqual(begin["bytes"], 4016)
        self.assertEqual(begin["prefix_constraints"]["opcode"], "BEGIN")
        manifest = begin["manifest_fields"]
        self.assertEqual(manifest[0]["offset"], 48)
        self.assertEqual(manifest[-1]["offset"] + manifest[-1]["bytes"], 544)
        fields = field_map(manifest)
        self.assertEqual(fields["image_sha256"]["bytes"], 32)
        self.assertEqual(fields["descriptor"]["bytes"], 64)
        self.assertEqual(fields["kernel_name"]["offset"], 224)
        self.assertEqual(fields["kernel_symbol"]["offset"], 352)
        self.assertEqual(begin["segment_table"]["offset"], 544)
        self.assertEqual(begin["segment_table"]["record_bytes"], 48)
        self.assertEqual(begin["segment_table"]["bytes"], 16 * 48)
        assert_contiguous(self, begin["segment_table"]["fields"], 48)
        self.assertEqual(begin["reserved_tail"], {
            "offset": 1312,
            "bytes": 2704,
            "must_be": "all-zero",
        })
        self.assertEqual(fields["relocation_count"]["a1_constant"], 0)
        self.assertEqual(fields["uses_dynamic_stack"]["a1_constant"], 0)

    def test_chunk_commit_and_ack_layouts_are_fixed(self) -> None:
        chunk = self.spec["chunk_payload"]
        self.assertEqual(chunk["bytes"], 4016)
        self.assertEqual(chunk["copied_bytes"], {
            "offset": 48,
            "bytes": 3968,
            "type": "bytes",
            "active_prefix": "byte_count",
            "inactive_tail": "all-zero",
        })
        commit = self.spec["commit_payload"]
        self.assertEqual(commit["bytes"], 4016)
        commit_end = commit["fields"][0]["offset"]
        for field in commit["fields"]:
            self.assertEqual(field["offset"], commit_end)
            commit_end += field["bytes"]
        self.assertEqual(commit_end, 4016)
        self.assertEqual(commit["fields"][0]["offset"], 48)
        self.assertEqual(commit["fields"][1]["offset"], 80)
        ack = self.spec["ack_payload"]
        self.assertEqual(ack["bytes"], 4016)
        assert_contiguous(self, ack["fields"], 4016)
        ack_fields = field_map(ack["fields"])
        for name in ("mapped_base_va", "descriptor_va", "code_va", "kernarg_va"):
            self.assertEqual(ack_fields[name]["a1_constant"], 0)

    def test_fixed_frame_crc_and_chunk_padding_oracle(self) -> None:
        identity = self.spec["golden"]["identity"]
        copied = bytes.fromhex(self.spec["golden"]["chunk"]["copied_hex"])
        expected_crc = int(
            self.spec["golden"]["chunk"]["copied_crc32c_hex"], 16
        )
        self.assertEqual(len(copied), self.spec["golden"]["chunk"]["copied_bytes"])
        self.assertEqual(crc32c(copied), expected_crc)

        payload = bytearray(self.spec["transport"]["payload_bytes"])
        struct.pack_into(
            ">HHHHQQQIIII",
            payload,
            0,
            1,
            0,
            self.spec["transport"]["request_opcodes"]["CHUNK"],
            0,
            int(identity["object_id_hex"], 16),
            int(identity["object_generation_hex"], 16),
            self.spec["golden"]["chunk"]["image_offset"],
            len(copied),
            self.spec["golden"]["chunk"]["index"],
            expected_crc,
            0,
        )
        payload[48 : 48 + len(copied)] = copied
        self.assertTrue(all(value == 0 for value in payload[48 + len(copied) :]))

        header = self.base["header"]
        frame = bytearray(self.spec["transport"]["record_bytes"])
        frame[:80] = struct.pack(
            ">8sHHHHIIQ16sQQIIQ",
            bytes.fromhex(next(
                field["constant_hex"]
                for field in header["fields"]
                if field["name"] == "magic"
            )),
            1,
            0,
            80,
            self.spec["transport"]["message_types"]["CODE_OBJECT_REQUEST"],
            0,
            len(payload),
            int(identity["request_id_hex"], 16),
            bytes.fromhex(identity["daemon_instance_uuid_hex"]),
            int(identity["connection_id_hex"], 16),
            int(identity["job_epoch_hex"], 16),
            0,
            0,
            0,
        )
        frame[80:] = payload
        checksum = crc32c(frame)
        frame[64:68] = checksum.to_bytes(4, "big")
        self.assertEqual(len(frame), 4096)
        self.assertEqual(int.from_bytes(frame[20:24], "big"), 4016)
        self.assertEqual(crc32c(frame[:64] + b"\0\0\0\0" + frame[68:]), checksum)
        self.assertEqual(
            int.from_bytes(frame[14:16], "big"),
            self.spec["transport"]["message_types"]["CODE_OBJECT_REQUEST"],
        )

    def test_chunk_count_and_digest_identity_are_deterministic(self) -> None:
        candidate = self.spec["observed_vecadd_candidate"]
        size = candidate["image_bytes"]
        capacity = self.spec["transport"]["chunk_data_bytes"]
        expected = (size + capacity - 1) // capacity
        self.assertEqual(expected, candidate["chunk_count"])
        self.assertEqual(sum(candidate["chunk_byte_counts"]), size)
        self.assertEqual(candidate["chunk_byte_counts"][0], capacity)
        self.assertEqual(
            candidate["image_sha256_hex"],
            "b0e07d4d34826177f7261b706dc9bac139d7a59b5217e8094fa714371874efaf",
        )
        self.assertNotEqual(candidate["image_sha256_hex"], "0" * 64)

    def test_descriptor_relation_and_segment_facts(self) -> None:
        kernel = self.spec["observed_vecadd_candidate"]["kernel"]
        descriptor = int(kernel["descriptor_address_hex"], 16)
        entry = int(kernel["descriptor_kernel_code_entry_byte_offset_hex"], 16)
        code = int(kernel["code_address_hex"], 16)
        self.assertEqual(descriptor + entry, code)
        self.assertEqual(kernel["entry_relation"], "0x8c0 + 0x1140 = 0x1a00")
        self.assertEqual(len(self.spec["observed_vecadd_candidate"]["segments"]), 3)
        for segment in self.spec["observed_vecadd_candidate"]["segments"]:
            self.assertLessEqual(
                int(segment["file_size_hex"], 16),
                int(segment["memory_size_hex"], 16),
            )
            self.assertEqual(int(segment["alignment_hex"], 16), 0x1000)

    def test_a1_is_explicitly_non_mapping_and_non_execution(self) -> None:
        boundary = self.spec["a1_boundary"]
        for key in (
            "pt_load_mapping",
            "process_page_mapping",
            "relocation_application",
            "kernarg_publication",
            "aql_construction",
            "queue_submission",
            "gem5_execution",
            "isa_supported_by_gemsim",
        ):
            self.assertFalse(boundary[key], key)
        self.assertIn("zero in every A1 ACK", boundary["ack_address_rule"])
        self.assertEqual(boundary["future_gates"], [
            "A2 independently validates and atomically maps PT_LOAD ranges, applies required relocations, and zero-fills memory_size-file_size",
            "A3 builds complete explicit plus hidden kernarg bytes and a daemon-owned AQL packet from validated handles",
            "A4 submits through the real HSA queue/fetch/decoder/retirement path and retains output and signal evidence",
        ])
        forbidden = " ".join(self.spec["forbidden_scope"])
        for token in ("host pointers", "file descriptors", "PT_LOAD", "CPU arithmetic fallback"):
            self.assertIn(token, forbidden)

    def test_source_constraints_and_document_cross_reference(self) -> None:
        source = self.spec["source_constraints"]
        self.assertTrue((ROOT / "projects/self-amdgpu-runtime/src/code_object.c").exists())
        self.assertTrue((ROOT / "projects/gem5/src/dev/amdgpu/host_gpu_protocol.cc").exists())
        self.assertIn("sagr_code_object_validate", source["runtime_parser"]["functions"])
        self.assertIn("existing message types end at 15", source["gem5_protocol"]["constraints"])
        self.assertIn("kernel_object", source["descriptor_execution_rule_for_future_gates"])
        docs = DOC_PATH.read_text(encoding="utf-8")
        self.assertIn("host-transport-v1-codeobj-transport.json", docs)
        self.assertIn("A1 is transport staging only", docs)
        self.assertIn("0x1140", docs)

    def test_error_order_and_ownership_limits_are_explicit(self) -> None:
        precedence = self.spec["validation_precedence"]
        expected = [
            "DROP_AND_CLOSE",
            "MALFORMED",
            "UNSUPPORTED_VERSION",
            "INSTANCE_MISMATCH",
            "PROTOCOL_STATE",
            "TOPOLOGY_MISMATCH",
            "UNSUPPORTED_CAPABILITY",
            "UNAUTHORIZED",
            "BUSY",
            "RESOURCE_EXHAUSTED",
            "INTERNAL",
        ]
        positions = [
            next(index for index, value in enumerate(precedence) if token in value)
            for token in expected
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("whole-image digest", precedence[10])
        ownership = self.spec["ownership_and_atomicity"]
        self.assertEqual(ownership["staging_ceiling_bytes"], 64 << 20)
        self.assertTrue(ownership["one_transaction_per_owner"])
        self.assertIn("exactly image_size staged bytes", ownership["sha256_rule"])


if __name__ == "__main__":
    unittest.main()
