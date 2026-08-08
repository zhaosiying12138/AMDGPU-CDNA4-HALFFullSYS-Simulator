#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocol" / "host-transport-v1-kmt.json"


def crc32c(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


class KmtSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(SPEC_PATH.read_text())

    @staticmethod
    def _assert_contiguous(testcase, fields, total):
        end = 0
        for field in fields:
            testcase.assertEqual(field["offset"], end, field["name"])
            testcase.assertGreater(field["bytes"], 0)
            end += field["bytes"]
        testcase.assertEqual(end, total)

    def test_scope_and_capability_are_pinned(self):
        self.assertEqual(self.spec["checkpoint"], "CP-0010")
        self.assertEqual(self.spec["base_protocol"], "protocol/host-transport-v1.json")
        self.assertEqual(self.spec["provider_authority"], "protocol/host-transport-v1-provider.json")
        self.assertEqual(self.spec["capability"], {
            "name": "KMT_OPERATION_V1",
            "bit": 5,
            "byte_index": 0,
            "mask_hex": "20",
            "negotiation": "selected if and only if offered and required by the client; required-not-offered is the base subset failure; absence returns UNSUPPORTED_CAPABILITY and creates no KMT owner or object state",
            "requires": ["TOPOLOGY_IDENTITY_V1"],
        })

    def test_fixed_transport_and_payload_sizes(self):
        transport = self.spec["transport"]
        self.assertEqual(transport["header_bytes"], 80)
        self.assertEqual(transport["request_payload_bytes"], 256)
        self.assertEqual(transport["result_payload_bytes"], 256)
        self.assertEqual(transport["request_frame_bytes"], 336)
        self.assertEqual(transport["result_frame_bytes"], 336)
        self.assertEqual(transport["message_types"], {"KMT_REQUEST": 14, "KMT_ACK": 15})
        self.assertEqual(transport["ancillary_descriptors"], 0)
        self.assertTrue(transport["no_fragmentation"])

    def test_payload_layouts_are_contiguous_and_fixed(self):
        self._assert_contiguous(self, self.spec["request_payload"]["fields"], 256)
        self._assert_contiguous(self, self.spec["result_payload"]["fields"], 256)
        request = {field["name"]: field for field in self.spec["request_payload"]["fields"]}
        result = {field["name"]: field for field in self.spec["result_payload"]["fields"]}
        self.assertEqual(request["copied_buffer"]["bytes"], 128)
        self.assertEqual(result["copied_result"]["bytes"], 128)
        self.assertEqual(request["reserved"]["bytes"], 24)
        self.assertEqual(result["reserved"]["bytes"], 16)
        self.assertEqual(request["copied_buffer"]["offset"], 104)
        self.assertEqual(result["copied_result"]["offset"], 112)
        self.assertEqual(request["arg_words"]["type"], "u32[8]")
        self.assertEqual(result["result_words"]["type"], "u32[8]")

    def test_all_wire_handles_are_fixed_width_u64_pairs(self):
        for payload_name in ("request_payload", "result_payload"):
            fields = {field["name"]: field for field in self.spec[payload_name]["fields"]}
            for name in ("owner_id", "owner_generation", "object_id", "object_generation", "aux_id", "aux_generation"):
                self.assertEqual(fields[name]["type"], "u64")
                self.assertEqual(fields[name]["bytes"], 8)
            self.assertNotIn("pointer", " ".join(fields))

    def test_operation_ids_are_unique_and_stable(self):
        operations = self.spec["operations"]
        ids = [entry["id"] for entry in operations.values()]
        self.assertEqual(ids, list(range(1, 19)))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(operations["OPEN_KFD"]["id"], 1)
        self.assertEqual(operations["MODEL_DRM_CALL"]["id"], 18)
        surface_ids = [entry["operation_id"] for entry in self.spec["operation_wrapper_surface"]]
        self.assertEqual(surface_ids, ids)
        implementation = self.spec["cp0010_implementation_table"]
        expected_wrappers = [
            "sagr_kmt_open_kfd", "sagr_kmt_close_kfd", "sagr_kmt_get_version",
            "sagr_kmt_topology_snapshot", "sagr_kmt_alloc_memory",
            "sagr_kmt_free_memory", "sagr_kmt_copy_memory",
            "sagr_kmt_queue_create", "sagr_kmt_queue_destroy",
            "sagr_kmt_queue_doorbell", "sagr_kmt_event_create",
            "sagr_kmt_event_destroy", "sagr_kmt_event_set",
            "sagr_kmt_event_reset", "sagr_kmt_event_query",
            "sagr_kmt_event_wait", "sagr_kmt_pointer_info",
            "sagr_kmt_model_drm_call",
        ]
        self.assertEqual(implementation["operation_wrappers_implemented"], expected_wrappers)
        self.assertEqual(implementation["provider_semantics_unsupported_by_default"], list(operations))
        self.assertIn("HSAKMT_STATUS_NOT_SUPPORTED", implementation["unsupported_rule"])
        layouts = self.spec["operation_layouts"]
        self.assertEqual(list(layouts), list(operations))
        for name, layout in layouts.items():
            self.assertEqual(layout["id"], operations[name]["id"])
            self.assertEqual(layout["allowed_flags"], [0])
            for field_name in ("arguments", "results"):
                words = [entry["word"] for entry in layout[field_name]]
                self.assertEqual(len(words), len(set(words)), name)
                self.assertTrue(all(0 <= word < 8 for word in words), name)
                self.assertTrue(all(entry["bits"] == 32 for entry in layout[field_name]), name)
                by_word = {entry["word"]: entry["name"] for entry in layout[field_name]}
                for word, field in by_word.items():
                    if field.endswith("_high"):
                        self.assertEqual(by_word.get(word + 1), field[:-5] + "_low", name)
            self.assertIn(layout["request_buffer"]["direction"], ("none", "client_to_daemon", "client_to_daemon only for H2D"))
            self.assertIn(layout["result_buffer"]["direction"], ("none", "daemon_to_client", "daemon_to_client only for D2H"))
        fixture = self.spec["deterministic_fixture"]
        version = bytes.fromhex(fixture["version_record"]["hex"])
        topology = bytes.fromhex(fixture["topology_record"]["hex"])
        self.assertEqual(len(version), fixture["version_record"]["bytes"])
        self.assertEqual(len(topology), fixture["topology_record"]["bytes"])
        self.assertEqual(f"{crc32c(version):08x}", fixture["version_record"]["crc32c_hex"])
        self.assertEqual(f"{crc32c(topology):08x}", fixture["topology_record"]["crc32c_hex"])
        self.assertEqual(fixture["topology"]["topology_generation"], "echo-base-header-job_epoch")
        self.assertIn("prefix", fixture["first_objects"]["id_rule"])
        self.assertIn("fixture job_epoch", fixture["first_objects"]["generation_rule"])

    def test_model_callback_version_policy_is_exact(self):
        policy = self.spec["model_callback_policy"]
        self.assertEqual(policy["supported_version"], {"major": 1, "minor": 1})
        self.assertEqual(policy["wire_callback_ids"], {"NONE": 0, "KFD_IOCTL": 1, "DRM_CALL": 2})
        commands = policy["drm_command_argument_bytes"]
        self.assertEqual([command["id"] for command in commands], list(range(15)))
        provider = json.loads((ROOT / "protocol" / "host-transport-v1-provider.json").read_text())
        authority_commands = provider["model_abi"]["drm_commands"]
        self.assertEqual([(command["name"], command["bytes"]) for command in commands], [(command["name"], command["sizeof"]) for command in authority_commands])
        self.assertIn("exactly 1.1", policy["negotiation"])
        self.assertIn("function address", policy["callback_ownership"])
        self.assertIn("15 source-defined", policy["drm_command"])
        self.assertIn("IDs 8 through 14", policy["hardware_command_policy"])
        self.assertIn("never opens", policy["hardware_command_policy"])

    def test_status_domains_and_precedence_are_explicit(self):
        status = self.spec["status_encoding"]
        self.assertIn("result.wire_status", status["transport_status"])
        self.assertIn("result.status", status["provider_status"])
        self.assertIn("wire_status", status["success"])
        precedence = self.spec["error_precedence"]
        self.assertGreaterEqual(len(precedence), 10)
        expected_order = ("DROP_AND_CLOSE", "MALFORMED", "UNSUPPORTED_VERSION", "INSTANCE_MISMATCH", "PROTOCOL_STATE", "TOPOLOGY_MISMATCH", "UNSUPPORTED_CAPABILITY", "UNAUTHORIZED", "INVALID_HANDLE", "INVALID_PARAMETER", "RESOURCE_EXHAUSTED")
        for earlier, later in zip(expected_order, expected_order[1:]):
            self.assertLess(next(i for i, value in enumerate(precedence) if earlier in value), next(i for i, value in enumerate(precedence) if later in value))

    def test_atomicity_and_sequence_contract(self):
        atomicity = self.spec["atomicity"]
        for key in ("validation", "unsupported_call", "failure", "result_copy", "sequence"):
            self.assertTrue(atomicity[key])
        self.assertIn("does not advance", atomicity["sequence"])
        self.assertIn("exactly once", atomicity["sequence"])

    def test_forbidden_scope_covers_pointer_fd_and_device_paths(self):
        forbidden = " ".join(self.spec["forbidden_scope"])
        for token in ("raw host pointers", "file descriptors", "/dev/kfd", "/dev/dri", "libhsakmt", "libdrm"):
            self.assertIn(token, forbidden)
        serialized = SPEC_PATH.read_text()
        self.assertFalse(re.search(r'"name"\s*:\s*"(?:host_)?(?:pointer|fd|dma_buf)', serialized, re.I))

    def test_document_and_authority_are_cross_referenced(self):
        docs = (ROOT / "docs" / "host-transport-v1-kmt.md").read_text()
        self.assertIn("protocol/host-transport-v1-kmt.json", docs)
        self.assertIn("KMT_OPERATION_V1", docs)
        self.assertIn("HSAKMT_STATUS_NOT_SUPPORTED", docs)
        self.assertIn("CP-0010", docs)


if __name__ == "__main__":
    unittest.main()
