# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import argparse
import array
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "host_transport_integration",
    ROOT / "scripts/test_host_transport_integration.py",
)
assert SPEC and SPEC.loader
integration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = integration
SPEC.loader.exec_module(integration)


def harness_args() -> argparse.Namespace:
    return argparse.Namespace(
        gem5=Path("/bin/true"),
        runtime_cli=Path("/bin/true"),
        gem5_config=Path("/fixture/host_bridge.py"),
        dispatch_gem5_config=Path("/fixture/host_dispatch.py"),
        start_wait_seconds=1.0,
        process_timeout_seconds=1.0,
        dispatch_process_timeout_seconds=5.0,
        client_timeout_ms=500,
        server_startup_timeout_ms=2000,
        server_handshake_timeout_ms=500,
        server_run_timeout_ms=2000,
        dispatch_server_run_timeout_ms=5000,
        hold_ms=250,
    )


def dispatch_fixture():
    authority = integration.load_json_document(
        integration.DISPATCH_PROTOCOL_PATH,
        "amdgpu-sim.host-transport-v1.dispatch.v1",
    )
    fixture = authority["fixture_authority"]
    identity = authority["golden"]["identity"]
    evidence = integration.DispatchEvidence(
        request_id=int(identity["request_id_hex"], 16),
        trace_id=int(identity["trace_id_hex"], 16),
        fixture_id=1,
        queue_id=int(identity["queue_id_hex"], 16),
        queue_generation=int(identity["queue_generation_hex"], 16),
        queue_sequence=int(identity["queue_sequence_hex"], 16),
        input_allocation_id=identity["input_allocation_id"],
        input_generation=int(identity["input_generation_hex"], 16),
        input_gpu_va=int(identity["input_gpu_va_hex"], 16),
        output_allocation_id=identity["output_allocation_id"],
        output_generation=int(identity["output_generation_hex"], 16),
        output_gpu_va=int(identity["output_gpu_va_hex"], 16),
        signal_id=identity["signal_id"],
        signal_generation=int(identity["signal_generation_hex"], 16),
        packet_crc32c=int(identity["packet_crc32c_hex"], 16),
        output_crc32c=int(fixture["golden_buffers"]["output_crc32c_hex"], 16),
        admission_tick=int(identity["admission_tick_hex"], 16),
        start_tick=int(identity["start_tick_hex"], 16),
        end_tick=int(identity["end_tick_hex"], 16),
        retire_tick=int(identity["retire_tick_hex"], 16),
        signal_completion_tick=int(identity["retire_tick_hex"], 16) + 1,
        materialized_aql_sha256=identity["materialized_aql_sha256_hex"],
    )
    common = {
        "schema": authority["trace_contract"]["schema"],
        "trace_id": hex(evidence.trace_id),
        "request_id": hex(evidence.request_id),
        "fixture_id": evidence.fixture_id,
        "fixture_manifest_sha256": fixture["manifest"]["sha256_hex"],
        "code_image_sha256": fixture["code_image"]["sha256_hex"],
        "aql_template_sha256": fixture["aql_template"]["sha256_hex"],
        "materialized_aql_sha256": evidence.materialized_aql_sha256,
        "queue_id": hex(evidence.queue_id),
        "queue_generation": hex(evidence.queue_generation),
        "queue_sequence": hex(evidence.queue_sequence),
        "input_allocation_id": evidence.input_allocation_id,
        "input_generation": hex(evidence.input_generation),
        "output_allocation_id": evidence.output_allocation_id,
        "output_generation": hex(evidence.output_generation),
        "signal_id": evidence.signal_id,
        "signal_generation": hex(evidence.signal_generation),
    }
    events = authority["trace_contract"]["required_ordered_events"]
    ticks = [
        evidence.admission_tick,
        evidence.admission_tick + 1,
        evidence.admission_tick + 2,
        evidence.admission_tick + 3,
        evidence.admission_tick + 4,
        evidence.start_tick,
        evidence.start_tick,
        evidence.end_tick,
        evidence.end_tick,
        evidence.retire_tick,
        evidence.retire_tick,
        evidence.retire_tick + 1,
    ]
    records = [
        {**common, "event": event, "sim_tick": tick}
        for event, tick in zip(events, ticks)
    ]
    by_event = {record["event"]: record for record in records}
    by_event["dispatch_admitted"]["admission_tick"] = evidence.admission_tick
    queue_fields = {
        "internal_queue_id": 0x31,
        "internal_queue_generation": 0x9,
    }
    by_event["aql_queue_registered"].update(queue_fields)
    by_event["aql_queue_registered"].update({
        "component": "HSAPacketProcessor",
        "active": True,
    })
    by_event["aql_packet_published"].update({
        **queue_fields,
        "packet_va": 0x200000003000,
        "materialized_aql_hex": identity["materialized_aql_hex"],
        "materialized_kernarg_hex": identity["materialized_kernarg_hex"],
        "materialized_kernarg_sha256": identity[
            "materialized_kernarg_sha256_hex"
        ],
        "kernel_object": int(identity["internal_code_va_hex"], 16),
        "kernarg_address": int(identity["internal_kernarg_va_hex"], 16),
        "completion_signal": 0,
        "packet_crc32c": int(identity["packet_crc32c_hex"], 16),
        "header": 0x1402,
        "setup": 1,
        "grid_size": [64, 1, 1],
        "workgroup_size": [64, 1, 1],
    })
    by_event["hsapp_packet_fetched"].update({
        **queue_fields,
        "packet_va": 0x200000003000,
        "read_index": 0,
        "component": "HSAPacketProcessor",
    })
    by_event["gpu_command_processor_submitted"].update({
        "gpu_task_id": 0x45,
        "component": "GPUCommandProcessor",
    })
    by_event["gpu_dispatcher_started"].update({
        "gpu_task_id": 0x45,
        "grid_size": [64, 1, 1],
        "workgroup_size": [64, 1, 1],
        "workgroups": 1,
        "waves": 1,
        "component": "GPUDispatcher",
    })
    by_event["cu_wave_started"].update({
        "gpu_task_id": 0x45,
        "cu_id": 0,
        "workgroup_id": [0, 0, 0],
        "wavefront_size": 64,
        "lane_mask": 0xFFFFFFFFFFFFFFFF,
        "component": "ComputeUnit",
    })
    by_event["cu_global_store_completed"].update({
        "gpu_task_id": 0x45,
        "output_gpu_va": int(identity["output_gpu_va_hex"], 16),
        "store_bytes": 64,
        "component": "ComputeUnit",
    })
    by_event["gpu_dispatcher_completed"].update({
        "gpu_task_id": 0x45,
        "component": "GPUDispatcher",
    })
    by_event["packet_retired"].update({
        **queue_fields,
        "packet_va": 0x200000003000,
        "gpu_task_id": 0x45,
        "finish_pkt_read_index": 0,
        "completion_signal": 0,
    })
    by_event["cp7_signal_mirrored"].update({
        "value_before": 1,
        "value_after": 0,
    })
    by_event["wire_completion_emitted"].update({
        "packet_crc32c": int(identity["packet_crc32c_hex"], 16),
        "output_crc32c": int(
            fixture["golden_buffers"]["output_crc32c_hex"], 16
        ),
        "admission_tick": evidence.admission_tick,
        "start_tick": evidence.start_tick,
        "end_tick": evidence.end_tick,
        "retire_tick": evidence.retire_tick,
        "signal_completion_tick": evidence.signal_completion_tick,
        "input_gpu_va": int(identity["input_gpu_va_hex"], 16),
        "output_gpu_va": int(identity["output_gpu_va_hex"], 16),
    })
    return authority, evidence, records


def dispatch_result_fixture():
    authority, evidence, _ = dispatch_fixture()
    fixture = authority["fixture_authority"]
    golden = fixture["golden_buffers"]

    def u64(value):
        return f"0x{value:016x}"

    def u32(value):
        return f"0x{value:08x}"

    ticket = {
        "request_id": u64(evidence.request_id),
        "queue_id": u64(evidence.queue_id),
        "queue_generation": u64(evidence.queue_generation),
        "queue_sequence": u64(evidence.queue_sequence),
        "input_allocation_id": u64(evidence.input_allocation_id),
        "input_generation": u64(evidence.input_generation),
        "output_allocation_id": u64(evidence.output_allocation_id),
        "output_generation": u64(evidence.output_generation),
        "signal_id": u64(evidence.signal_id),
        "signal_generation": u64(evidence.signal_generation),
        "trace_id": u64(evidence.trace_id),
        "input_gpu_va": u64(evidence.input_gpu_va),
        "output_gpu_va": u64(evidence.output_gpu_va),
        "packet_crc32c": u32(evidence.packet_crc32c),
        "admission_tick": u64(evidence.admission_tick),
    }
    completion = {
        "status": 0,
        "wire_status": 0,
        **ticket,
        "fixture_id": u64(evidence.fixture_id),
        "output_crc32c": u32(evidence.output_crc32c),
        "start_tick": u64(evidence.start_tick),
        "end_tick": u64(evidence.end_tick),
        "retire_tick": u64(evidence.retire_tick),
    }
    value = {
        "status": 0,
        "fixture": integration.DISPATCH_FIXTURE,
        "fixture_id": u64(evidence.fixture_id),
        "fixture_manifest_sha256": fixture["manifest"]["sha256_hex"],
        "input_crc32c": "0x" + golden["input_crc32c_hex"],
        "output_sentinel_crc32c": (
            "0x" + golden["output_initial_crc32c_hex"]
        ),
        "input_hex": golden["input_hex"],
        "initial_output_hex": "00" * 64,
        "expected_output_hex": golden["output_hex"],
        "d2h_output_hex": golden["output_hex"],
        "ticket": ticket,
        "first_wait": {
            "status": 19,
            "status_name": "cancelled",
            "wire_status": -1,
            "retried_without_send": True,
        },
        "completion": completion,
        "signal": {
            "armed_wait_status": 11,
            "armed_wait_wire_status": -1,
            "armed_wait_status_name": "timed out",
            "observed_value": 0,
            "signal_completion_tick": u64(evidence.signal_completion_tick),
            "retried_without_send": True,
        },
        "output_crc32c": u32(evidence.output_crc32c),
        "output_match": True,
        "cleanup": {
            "queue_destroyed": True,
            "input_freed": True,
            "output_freed": True,
            "signal_destroyed": True,
        },
    }
    return authority, evidence, value


class HostTransportIntegrationHarnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.wire = integration.WireProtocol.load()

    def test_identity_fixture_is_exact_and_unique(self) -> None:
        identities = [
            integration.make_identity(101 + world, world, rank)
            for world in integration.WORLD_SIZES
            for rank in range(world)
        ]
        self.assertEqual(len({value.daemon_uuid for value in identities}), 18)
        self.assertEqual(
            integration.make_identity(109, 8, 3),
            integration.make_identity(109, 8, 3),
        )
        self.assertTrue(all(value.epoch != 0 for value in identities))
        for world in integration.WORLD_SIZES:
            selected = [value for value in identities if value.world_size == world]
            self.assertEqual({value.rank for value in selected}, set(range(world)))
            self.assertEqual(len({value.job_uuid for value in selected}), 1)

    def test_daemon_and_runtime_argv_carry_exact_topology(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cp4-harness-test-") as temp:
            harness = integration.Harness(harness_args(), Path(temp))
            identity = integration.make_identity(44, 3, 2)
            daemon = integration.Daemon(
                harness, "rank-2", identity, exit_on_handshake=True
            )
            daemon_argv = daemon.argv()
            client_argv = harness.client_argv(daemon.endpoint, identity)

        self.assertIn("--listener-mode=on", daemon_argv)
        self.assertEqual(
            daemon_argv[daemon_argv.index("--run-timeout-ms") + 1], "2000"
        )
        self.assertEqual(
            daemon_argv[daemon_argv.index("--world-size") + 1], "3"
        )
        self.assertEqual(
            client_argv[client_argv.index("--expected-daemon-uuid") + 1],
            identity.daemon_uuid,
        )
        self.assertEqual(
            client_argv[client_argv.index("--expected-rank") + 1], "2"
        )

        deadline_daemon = integration.Daemon(
            harness,
            "deadline",
            identity,
            exit_on_handshake=False,
            handshake_timeout_ms=123,
        )
        deadline_argv = deadline_daemon.argv()
        self.assertEqual(
            deadline_argv[deadline_argv.index("--handshake-timeout-ms") + 1],
            "123",
        )
        dispatch_daemon = integration.Daemon(
            harness,
            "dispatch",
            identity,
            exit_on_handshake=False,
            config=harness.dispatch_gem5_config,
            extra_args=("--dispatch-trace-path", "/fixture/trace.jsonl"),
        )
        dispatch_argv = dispatch_daemon.argv()
        self.assertIn(str(harness.dispatch_gem5_config), dispatch_argv)
        self.assertEqual(
            dispatch_argv[dispatch_argv.index("--dispatch-trace-path") + 1],
            "/fixture/trace.jsonl",
        )

    def test_raw_wire_crc_and_dynamic_hello_match_canonical_golden(self) -> None:
        self.assertEqual(self.wire.crc32c(b"123456789"), 0xE3069283)
        self.wire.assert_canonical_golden()

        golden = self.wire.document["golden"]
        identity_values = golden["identity"]
        identity = integration.Identity(
            daemon_uuid=identity_values["daemon_instance_uuid_hex"],
            job_uuid=identity_values["job_uuid_hex"],
            epoch=int(identity_values["job_epoch_hex"], 16),
            rank=identity_values["rank"],
            world_size=identity_values["world_size"],
        )
        malformed = self.wire.encode_hello(
            identity,
            request_id=int(identity_values["request_id_hex"], 16),
            client_nonce=bytes.fromhex(identity_values["client_nonce_hex"]),
            role=2,
        )
        role = self.wire.message_field("HELLO", "role")
        role_offset = self.wire.header_bytes + int(role["offset"])
        self.assertEqual(int.from_bytes(malformed[role_offset:role_offset + 2], "big"), 2)
        checksum = self.wire.header_field("crc32c")
        checksum_offset = int(checksum["offset"])
        expected_crc = int.from_bytes(
            malformed[checksum_offset:checksum_offset + int(checksum["bytes"])],
            "big",
        )
        zeroed = bytearray(malformed)
        zeroed[checksum_offset:checksum_offset + int(checksum["bytes"])] = bytes(
            int(checksum["bytes"])
        )
        self.assertEqual(self.wire.crc32c(bytes(zeroed)), expected_crc)

    def test_raw_wire_decodes_canonical_success_ack(self) -> None:
        golden = self.wire.document["golden"]
        values = golden["identity"]
        identity = integration.Identity(
            daemon_uuid=values["daemon_instance_uuid_hex"],
            job_uuid=values["job_uuid_hex"],
            epoch=int(values["job_epoch_hex"], 16),
            rank=values["rank"],
            world_size=values["world_size"],
        )
        ack = self.wire.decode_ack(
            bytes.fromhex(golden["hello_success_ack"]["frame_hex"]),
            request_id=int(values["request_id_hex"], 16),
            client_nonce=bytes.fromhex(values["client_nonce_hex"]),
            expected_status=self.wire.statuses["OK"],
            identity=identity,
            configured_maximum=values["rx_max_record"],
        )
        self.assertEqual(ack.connection_id, int(values["connection_id_hex"], 16))
        self.assertEqual(ack.topology_job_uuid, identity.job_uuid)
        self.assertEqual(ack.topology_rank, 3)
        self.assertEqual(ack.topology_world_size, 8)

    def test_raw_wire_decodes_correlated_canonical_malformed_ack(self) -> None:
        golden = self.wire.document["golden"]
        values = golden["identity"]
        identity = integration.Identity(
            daemon_uuid=values["daemon_instance_uuid_hex"],
            job_uuid=values["job_uuid_hex"],
            epoch=int(values["job_epoch_hex"], 16),
            rank=values["rank"],
            world_size=values["world_size"],
        )
        frame = bytearray(
            bytes.fromhex(golden["hello_success_ack"]["frame_hex"])[
                :self.wire.header_bytes + self.wire.ack_bytes
            ]
        )

        def store(field, value: int, base: int = 0) -> None:
            offset = base + int(field["offset"])
            size = int(field["bytes"])
            frame[offset:offset + size] = value.to_bytes(size, "big")

        header = self.wire.header_field
        fields = {
            field["name"]: field
            for field in self.wire.document["messages"]["HELLO_ACK"]["fields"]
        }
        store(header("payload_bytes"), self.wire.ack_bytes)
        store(header("connection_id"), 0)
        base = self.wire.header_bytes
        store(fields["selected_major"], 0, base)
        store(fields["selected_minor"], 0, base)
        store(fields["status"], self.wire.statuses["MALFORMED"], base)
        server_offset = base + int(fields["server_nonce"]["offset"])
        frame[server_offset:server_offset + int(fields["server_nonce"]["bytes"])] = (
            bytes(int(fields["server_nonce"]["bytes"]))
        )
        selected_offset = base + int(fields["selected_capabilities"]["offset"])
        frame[
            selected_offset:
            selected_offset + int(fields["selected_capabilities"]["bytes"])
        ] = bytes(int(fields["selected_capabilities"]["bytes"]))
        checksum = header("crc32c")
        store(checksum, 0)
        store(checksum, self.wire.crc32c(bytes(frame)))

        ack = self.wire.decode_ack(
            bytes(frame),
            request_id=int(values["request_id_hex"], 16),
            client_nonce=bytes.fromhex(values["client_nonce_hex"]),
            expected_status=self.wire.statuses["MALFORMED"],
            identity=identity,
            configured_maximum=values["rx_max_record"],
        )
        self.assertEqual(ack.status, self.wire.statuses["MALFORMED"])
        self.assertEqual(ack.connection_id, 0)
        self.assertFalse(any(ack.server_nonce))

    def test_raw_material_changes_without_becoming_zero(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cp4-harness-test-") as temp:
            harness = integration.Harness(harness_args(), Path(temp))
            first = harness.raw_material("fixture")
            second = harness.raw_material("fixture")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first[0], 0)
        self.assertTrue(any(first[1]))

    def test_status_json_and_peer_pid_are_strict(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cp4-harness-test-") as temp:
            harness = integration.Harness(harness_args(), Path(temp))
            identity = integration.make_identity(9, 1, 0)
            daemon = types.SimpleNamespace(
                maximum_record=32768,
                process=types.SimpleNamespace(pid=os.getpid()),
            )
            payload = {
                "status": 0,
                "selected_version": "1.0",
                "capability_words": [
                    "0x0000000000000001",
                    "0x0000000000000000",
                    "0x0000000000000000",
                    "0x0000000000000000",
                ],
                "daemon_uuid": identity.daemon_uuid,
                "job_uuid": identity.job_uuid,
                "connection_id": "0x1",
                "epoch": hex(identity.epoch),
                "rank": 0,
                "world_size": 1,
                "maximum_record_bytes": 32768,
                "request_id": "0x2",
                "peer_uid": os.geteuid(),
                "peer_pid": os.getpid(),
            }
            result = integration.ClientResult(
                ["fixture"], 0, json.dumps(payload) + "\n", ""
            )
            self.assertEqual(
                harness.validate_success(result, identity, daemon), payload
            )

            crossed = dict(payload, peer_pid=os.getpid() + 1)
            with self.assertRaises(integration.CheckFailure):
                harness.validate_success(
                    integration.ClientResult(
                        ["fixture"], 0, json.dumps(crossed) + "\n", ""
                    ),
                    identity,
                    daemon,
                )

            failure = integration.ClientResult(
                ["fixture"],
                1,
                "",
                json.dumps({
                    "status": 18,
                    "status_name": "busy",
                    "wire_status": 7,
                    "native_errno": 0,
                    "message": "daemon rejected handshake",
                }) + "\n",
            )
            self.assertEqual(
                harness.expect_failure(failure, 18, "busy", 7)["wire_status"],
                7,
            )

    def test_memory_result_validation_is_generation_and_va_strict(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cp6-harness-test-") as temp:
            harness = integration.Harness(harness_args(), Path(temp))
            payload = {
                "memory": {
                    "status": 0,
                    "allocation_id": "0x0000000000000007",
                    "generation": "0x0000000000000011",
                    "simulated_va": "0x0000100300000000",
                    "size_bytes": 131089,
                    "alignment_bytes": 65536,
                    "initial_zero": True,
                    "pattern_crc32c": "0x48dfe982",
                    "returned_crc32c": "0x48dfe982",
                    "match": True,
                    "freed": True,
                    "reuse": {
                        "allocation_id": "0x0000000000000007",
                        "generation": "0x0000000000000012",
                        "simulated_va": "0x0000100300000000",
                        "initial_zero": True,
                        "freed": True,
                    },
                }
            }
            self.assertEqual(
                harness.validate_memory_success(
                    payload,
                    expected_bytes=131089,
                    expected_alignment=65536,
                    require_reuse=True,
                ),
                payload["memory"],
            )

            bad_generation = json.loads(json.dumps(payload))
            bad_generation["memory"]["reuse"]["generation"] = "0x11"
            with self.assertRaises(integration.CheckFailure):
                harness.validate_memory_success(
                    bad_generation,
                    expected_bytes=131089,
                    expected_alignment=65536,
                    require_reuse=True,
                )

            bad_va = json.loads(json.dumps(payload))
            bad_va["memory"]["simulated_va"] = "0x0000100380000000"
            with self.assertRaises(integration.CheckFailure):
                harness.validate_memory_success(
                    bad_va,
                    expected_bytes=131089,
                    expected_alignment=65536,
                    require_reuse=True,
                )

    def test_signal_result_validation_requires_retry_and_new_generation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cp7-harness-test-") as temp:
            harness = integration.Harness(harness_args(), Path(temp))
            payload = {
                "signal": {
                    "status": 0,
                    "signal_id": "0x0000000000000007",
                    "generation": "0x8877665544332211",
                    "initial_value": -7,
                    "load_before": -7,
                    "wait": {
                        "condition": "gte",
                        "compare": 0,
                        "first_status": 11,
                        "first_status_name": "timed out",
                        "completion_status": 0,
                        "observed_value": 42,
                        "sequence": "0x0000000000000001",
                        "retried_without_send": True,
                    },
                    "stored_value": 42,
                    "load_after": 42,
                    "destroyed": True,
                    "reuse": {
                        "signal_id": "0x0000000000000007",
                        "generation": "0x8877665544332212",
                        "initial_value": -7,
                        "destroyed": True,
                    },
                }
            }
            self.assertEqual(
                harness.validate_signal_success(
                    payload, expected_initial=-7, expected_stored=42
                ),
                payload["signal"],
            )

            repeated = json.loads(json.dumps(payload))
            repeated["signal"]["reuse"]["generation"] = (
                repeated["signal"]["generation"]
            )
            with self.assertRaises(integration.CheckFailure):
                harness.validate_signal_success(
                    repeated, expected_initial=-7, expected_stored=42
                )

            regressed = json.loads(json.dumps(payload))
            regressed["signal"]["reuse"]["generation"] = "0x1"
            with self.assertRaises(integration.CheckFailure):
                harness.validate_signal_success(
                    regressed, expected_initial=-7, expected_stored=42
                )

            replayed = json.loads(json.dumps(payload))
            replayed["signal"]["wait"]["retried_without_send"] = False
            with self.assertRaises(integration.CheckFailure):
                harness.validate_signal_success(
                    replayed, expected_initial=-7, expected_stored=42
                )

    def test_dispatch_trace_fixture_is_strict_and_cross_correlated(self) -> None:
        authority, evidence, records = dispatch_fixture()
        summary = integration.validate_dispatch_trace(
            records, evidence, authority
        )
        self.assertEqual(summary["events"], 12)
        self.assertEqual(summary["request_id"], evidence.request_id)
        self.assertEqual(summary["trace_id"], evidence.trace_id)
        self.assertEqual(
            summary["materialized_aql_sha256"],
            evidence.materialized_aql_sha256,
        )

        mutations = {}
        crossed_request = json.loads(json.dumps(records))
        crossed_request[5]["request_id"] = "0x99"
        mutations["request correlation"] = crossed_request
        reordered = json.loads(json.dumps(records))
        reordered[5], reordered[6] = reordered[6], reordered[5]
        mutations["event ordering"] = reordered
        host_packet = json.loads(json.dumps(records))
        packet = bytearray(bytes.fromhex(
            host_packet[2]["materialized_aql_hex"]
        ))
        packet[16] ^= 1
        host_packet[2]["materialized_aql_hex"] = packet.hex()
        mutations["packet materialization boundary"] = host_packet
        wrong_task = json.loads(json.dumps(records))
        wrong_task[7]["gpu_task_id"] = "0x46"
        mutations["GPU task correlation"] = wrong_task
        wrong_signal = json.loads(json.dumps(records))
        wrong_signal[10]["value_before"] = 0
        mutations["CP7 mirror transition"] = wrong_signal
        extra_field = json.loads(json.dumps(records))
        extra_field[4]["alias_task_id"] = 0x45
        mutations["unfrozen event alias"] = extra_field
        wrong_type = json.loads(json.dumps(records))
        wrong_type[2]["header"] = "0x1402"
        mutations["event field JSON type"] = wrong_type
        missing_signal_tick = json.loads(json.dumps(records))
        del missing_signal_tick[-1]["signal_completion_tick"]
        mutations["signal completion summary"] = missing_signal_tick
        for name, mutation in mutations.items():
            with self.subTest(name=name), self.assertRaises(
                integration.CheckFailure
            ):
                integration.validate_dispatch_trace(
                    mutation, evidence, authority
                )

        bad_ticks = dataclasses.replace(evidence, retire_tick=evidence.end_tick - 1)
        with self.assertRaises(integration.CheckFailure):
            integration.validate_dispatch_trace(records, bad_ticks, authority)
        bad_signal_tick = dataclasses.replace(
            evidence, signal_completion_tick=evidence.retire_tick + 2
        )
        with self.assertRaises(integration.CheckFailure):
            integration.validate_dispatch_trace(
                records, bad_signal_tick, authority
            )
        expanded_authority = json.loads(json.dumps(authority))
        expanded_authority["trace_contract"]["common_fields"].append(
            "unfrozen_alias"
        )
        with self.assertRaises(integration.CheckFailure):
            integration.validate_dispatch_trace(
                records, evidence, expanded_authority
            )

    def test_dispatch_runtime_result_is_exact_and_byte_authoritative(self) -> None:
        authority, expected, value = dispatch_result_fixture()
        actual = integration.validate_dispatch_result(value, authority)
        self.assertEqual(
            dataclasses.replace(actual, materialized_aql_sha256=(
                expected.materialized_aql_sha256
            )),
            expected,
        )

        mutations = {}
        extra = json.loads(json.dumps(value))
        extra["first_wait"]["request_count"] = 1
        mutations["unfrozen wait alias"] = extra
        resent = json.loads(json.dumps(value))
        resent["first_wait"]["retried_without_send"] = False
        mutations["dispatch resend"] = resent
        wrong_echo = json.loads(json.dumps(value))
        wrong_echo["completion"]["request_id"] = "0x0000000000000001"
        mutations["completion echo"] = wrong_echo
        wrong_signal = json.loads(json.dumps(value))
        wrong_signal["signal"]["signal_completion_tick"] = (
            wrong_signal["completion"]["retire_tick"]
        )
        mutations["signal completion tick"] = wrong_signal
        wrong_output = json.loads(json.dumps(value))
        wrong_output["d2h_output_hex"] = "00" * 64
        mutations["D2H byte oracle"] = wrong_output
        wrong_crc = json.loads(json.dumps(value))
        wrong_crc["output_crc32c"] = "0x00000000"
        mutations["D2H CRC"] = wrong_crc
        leaked = json.loads(json.dumps(value))
        leaked["cleanup"]["signal_destroyed"] = False
        mutations["resource cleanup"] = leaked
        for name, mutation in mutations.items():
            with self.subTest(name=name), self.assertRaises(
                integration.CheckFailure
            ):
                integration.validate_dispatch_result(mutation, authority)

        identity = integration.Identity(
            daemon_uuid="11" * 16,
            job_uuid="22" * 16,
            epoch=3,
            rank=0,
            world_size=1,
        )
        payload = {
            "status": 0,
            "selected_version": "1.0",
            "capability_words": [
                "0x000000000000001f",
                "0x0000000000000000",
                "0x0000000000000000",
                "0x0000000000000000",
            ],
            "daemon_uuid": identity.daemon_uuid,
            "job_uuid": identity.job_uuid,
            "connection_id": "0x0000000000000004",
            "epoch": "0x0000000000000003",
            "rank": 0,
            "world_size": 1,
            "maximum_record_bytes": 65536,
            "request_id": "0x0000000000000001",
            "peer_uid": os.geteuid(),
            "peer_pid": os.getpid(),
            "dispatch": value,
        }
        daemon = types.SimpleNamespace(
            maximum_record=65536,
            process=types.SimpleNamespace(pid=os.getpid()),
        )
        with tempfile.TemporaryDirectory(prefix="cp8-result-test-") as temp:
            harness = integration.Harness(harness_args(), Path(temp))
            parsed, live = harness.validate_dispatch_success(
                integration.ClientResult(
                    ["sagr-handshake"], 0, json.dumps(payload) + "\n", ""
                ),
                identity,
                daemon,
                authority,
            )
            self.assertEqual(parsed, payload)
            self.assertEqual(live.request_id, expected.request_id)
            foreign = dict(payload)
            foreign["dispatch_alias"] = foreign["dispatch"]
            with self.assertRaises(integration.CheckFailure):
                harness.validate_dispatch_success(
                    integration.ClientResult(
                        ["sagr-handshake"], 0,
                        json.dumps(foreign) + "\n", "",
                    ),
                    identity,
                    daemon,
                    authority,
                )
        with self.assertRaises(integration.CheckFailure):
            integration.parse_single_json_object(
                '{"status":0,"status":0}\n', "duplicate fixture"
            )
        with self.assertRaises(integration.CheckFailure):
            integration.parse_single_json_object(
                '{"status":0}', "unterminated fixture"
            )

    def test_dispatch_stats_fixture_requires_real_gpu_work_and_no_fallback(self) -> None:
        names = integration.DISPATCH_STAT_NAMES
        self.assertEqual(
            names["retired_instructions"],
            "system.cpu1.CUs.numInstrExecuted",
        )
        self.assertTrue(all(
            name.startswith("system.host_gpu_bridge.")
            for semantic, name in names.items()
            if semantic != "retired_instructions"
        ))
        stats = {name: "1" for name in names.values()}
        stats[names["retired_instructions"]] = "7"
        stats[names["global_store_bytes"]] = "64"
        stats[names["host_fallback_count"]] = "0"
        values = integration.validate_dispatch_stats(stats, names)
        self.assertEqual(values["global_store_bytes"], 64)
        self.assertGreater(values["retired_instructions"], 0)

        for semantic, value in (
            ("hsapp_packets_fetched", "0"),
            ("workgroups_completed", "2"),
            ("retired_instructions", "0"),
            ("global_store_instructions", "0"),
            ("global_store_bytes", "63"),
            ("host_fallback_count", "1"),
        ):
            broken = dict(stats)
            broken[names[semantic]] = value
            with self.subTest(semantic=semantic), self.assertRaises(
                integration.CheckFailure
            ):
                integration.validate_dispatch_stats(broken, names)

        missing = dict(stats)
        del missing[names["waves_started"]]
        with self.assertRaises(integration.CheckFailure):
            integration.validate_dispatch_stats(missing, names)

    def test_unsupported_capability_probe_stays_outside_cp8_bit(self) -> None:
        authority = integration.load_json_document(
            integration.DISPATCH_PROTOCOL_PATH,
            "amdgpu-sim.host-transport-v1.dispatch.v1",
        )
        self.assertEqual(authority["capability"]["bit"], 4)
        self.assertGreater(
            integration.UNSUPPORTED_CAPABILITY_PROBE_BIT,
            authority["capability"]["bit"],
        )

    def test_missing_dispatch_children_fail_before_creating_work(self) -> None:
        script = ROOT / "scripts/test_host_transport_integration.py"
        missing_binary = subprocess.run(
            [
                sys.executable,
                str(script),
                "--gem5", "/definitely/missing/gem5.opt",
                "--runtime-cli", "/bin/true",
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        self.assertEqual(missing_binary.returncode, 2)
        self.assertIn("not an executable file", missing_binary.stderr)

        missing_config = subprocess.run(
            [
                sys.executable,
                str(script),
                "--gem5", "/bin/true",
                "--runtime-cli", "/bin/true",
                "--dispatch-gem5-config",
                "/definitely/missing/host_dispatch.py",
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        self.assertEqual(missing_config.returncode, 2)
        self.assertIn("not a file", missing_config.stderr)

    def test_dispatch_trace_jsonl_and_gem5_stats_parsers_are_bounded(self) -> None:
        _, _, records = dispatch_fixture()
        with tempfile.TemporaryDirectory(prefix="cp8-parser-test-") as temp:
            root = Path(temp)
            trace_path = root / "dispatch-trace.jsonl"
            trace_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            self.assertEqual(integration.load_dispatch_trace(trace_path), records)

            trace_path.write_text('{"event":"a","event":"b"}\n',
                                  encoding="utf-8")
            with self.assertRaises(integration.CheckFailure):
                integration.load_dispatch_trace(trace_path)
            trace_path.write_text('{"event":NaN}\n', encoding="utf-8")
            with self.assertRaises(integration.CheckFailure):
                integration.load_dispatch_trace(trace_path)
            trace_path.write_text('{"event":"partial"}', encoding="utf-8")
            with self.assertRaises(integration.CheckFailure):
                integration.load_dispatch_trace(trace_path)
            trace_path.write_text(
                "{}\n" * (integration.MAX_DISPATCH_TRACE_RECORDS + 1),
                encoding="utf-8",
            )
            with self.assertRaises(integration.CheckFailure):
                integration.load_dispatch_trace(trace_path)

            stats_path = root / "stats.txt"
            stats_path.write_text(
                "---------- Begin Simulation Statistics ----------\n"
                "system.dispatch.workgroups_started 1 # fixture\n"
                "system.dispatch.global_store_bytes 64 # fixture\n"
                "---------- End Simulation Statistics   ----------\n",
                encoding="utf-8",
            )
            self.assertEqual(
                integration.parse_gem5_stats(stats_path),
                {
                    "system.dispatch.workgroups_started": "1",
                    "system.dispatch.global_store_bytes": "64",
                },
            )
            stats_path.write_text(
                "---------- Begin Simulation Statistics ----------\n"
                "system.dispatch.workgroups_started 1\n"
                "system.dispatch.workgroups_started 2\n"
                "---------- End Simulation Statistics   ----------\n",
                encoding="utf-8",
            )
            with self.assertRaises(integration.CheckFailure):
                integration.parse_gem5_stats(stats_path)

            trace_path.write_text("{}\n", encoding="utf-8")
            os.chmod(trace_path, 0o600)
            self.assertEqual(
                integration.validate_evidence_file(
                    trace_path, "dispatch trace", exact_mode=0o600
                ).st_size,
                3,
            )
            evidence_link = root / "trace-link.jsonl"
            evidence_link.symlink_to(trace_path.name)
            with self.assertRaises(integration.CheckFailure):
                integration.validate_evidence_file(
                    evidence_link, "dispatch trace", exact_mode=0o600
                )
            empty_stats = root / "empty-stats.txt"
            empty_stats.touch(mode=0o600)
            with self.assertRaises(integration.CheckFailure):
                integration.validate_evidence_file(empty_stats, "gem5 stats")

    def test_dispatch_process_audit_rejects_native_gpu_dependencies(self) -> None:
        clean = integration.ProcessAudit(
            maps="/usr/bin/gem5.opt\n/lib/x86_64-linux-gnu/libc.so.6",
            open_paths=("/tmp/dispatch.sock", "/tmp/dispatch-trace.jsonl"),
            log="host-gpu-ready",
        )
        integration.validate_process_audit(clean)
        for source, marker in (
            ("maps", "/opt/rocm/lib/libhsa-runtime64.so"),
            ("open_paths", "/dev/kfd"),
            ("open_paths", "/dev/dri/renderD128"),
            ("log", "loaded libamdhip64.so"),
        ):
            values = dataclasses.asdict(clean)
            if source == "open_paths":
                values[source] = (*values[source], marker)
            else:
                values[source] += "\n" + marker
            with self.subTest(source=source, marker=marker), self.assertRaises(
                integration.CheckFailure
            ):
                integration.validate_process_audit(
                    integration.ProcessAudit(**values)
                )

    def test_dispatch_exit_log_requires_exact_causal_marker(self) -> None:
        stats_path = Path("/tmp/cp8-evidence/pinned-dispatch.m5out/stats.txt")
        marker = (
            "host-gpu-dispatch-exit cause=host GPU dispatch session complete "
            f"code=0 tick=42 stats={stats_path}"
        )
        self.assertEqual(
            integration.validate_dispatch_exit_log(
                "gem5 prelude\n" + marker + "\n", stats_path, 41
            ),
            42,
        )
        invalid = {
            "timeout cause": marker.replace(
                "host GPU dispatch session complete",
                "host wall-clock service timeout",
            ),
            "nonzero code": marker.replace("code=0", "code=1"),
            "foreign stats": marker.replace(str(stats_path), "/tmp/stats.txt"),
            "noncanonical tick": marker.replace("tick=42", "tick=042"),
            "early tick": marker.replace("tick=42", "tick=40"),
            "duplicate": marker + "\n" + marker,
            "missing": "host-gpu-ready",
        }
        for name, log in invalid.items():
            with self.subTest(name=name), self.assertRaises(
                integration.CheckFailure
            ):
                integration.validate_dispatch_exit_log(
                    log + "\n", stats_path, 41
                )

    @unittest.skipUnless(Path("/proc/self/maps").exists(), "procfs is required")
    def test_audited_client_runner_is_bounded_and_samples_live_process(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cp8-audit-runner-") as temp:
            root = Path(temp)
            log_path = root / "gem5.log"
            log_path.write_text("host-gpu-ready\n", encoding="utf-8")
            daemon_process = subprocess.Popen(
                ["/bin/sleep", "2"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
            daemon = types.SimpleNamespace(
                name="audit-fixture",
                process=daemon_process,
                log_path=log_path,
            )
            try:
                harness = integration.Harness(harness_args(), root)
                result, samples = harness.run_client_argv_audited(
                    ["/bin/sleep", "0.05"], daemon, timeout=1.0
                )
                self.assertEqual(result.returncode, 0)
                self.assertGreaterEqual(samples.daemon_samples, 1)
                self.assertGreaterEqual(samples.runtime_samples, 1)
            finally:
                integration.terminate_process(daemon_process, grace=0.5)

    @unittest.skipUnless(
        hasattr(socket, "SOCK_SEQPACKET") and Path("/proc/net/unix").exists(),
        "Linux seqpacket and procfs are required",
    )
    def test_connected_seqpacket_probe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cp4-harness-test-") as temp:
            endpoint = str(Path(temp) / "socket")
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            accepted = None
            try:
                listener.bind(endpoint)
                listener.listen(1)
                client.connect(endpoint)
                self.assertTrue(
                    integration.process_has_connected_unix_socket(os.getpid())
                )
                self.assertGreaterEqual(
                    integration.process_connected_unix_socket_count(os.getpid()),
                    1,
                )
                accepted, _ = listener.accept()
            finally:
                if accepted is not None:
                    accepted.close()
                client.close()
                listener.close()

    @unittest.skipUnless(
        hasattr(socket, "SOCK_SEQPACKET") and hasattr(socket, "SCM_RIGHTS"),
        "Linux seqpacket descriptor passing is required",
    )
    def test_raw_seqpacket_control_fixture(self) -> None:
        sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client = integration.RawSeqpacketClient(Path("/fixture"), 0.5)
        client.socket = sender
        client.deadline = time.monotonic() + client.timeout

        def close_received(ancillary) -> int:
            count = 0
            for level, kind, payload in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    usable = len(payload) - (len(payload) % array.array("i").itemsize)
                    descriptors = array.array("i")
                    descriptors.frombytes(payload[:usable])
                    for descriptor in descriptors:
                        os.close(descriptor)
                        count += 1
            return count

        try:
            client.send_rights(b"record", 1)
            payload, ancillary, flags, _ = receiver.recvmsg(
                64, socket.CMSG_SPACE(array.array("i").itemsize)
            )
            self.assertEqual(payload, b"record")
            self.assertEqual(flags & socket.MSG_CTRUNC, 0)
            self.assertEqual(close_received(ancillary), 1)

            client.send_rights(b"", 1)
            payload, ancillary, flags, _ = receiver.recvmsg(
                64, socket.CMSG_SPACE(array.array("i").itemsize)
            )
            self.assertEqual(payload, b"")
            self.assertEqual(flags & socket.MSG_CTRUNC, 0)
            self.assertEqual(close_received(ancillary), 1)

            client.send_rights(b"overflow", 8)
            payload, ancillary, flags, _ = receiver.recvmsg(
                64, socket.CMSG_SPACE(array.array("i").itemsize)
            )
            self.assertEqual(payload, b"overflow")
            self.assertNotEqual(flags & socket.MSG_CTRUNC, 0)
            self.assertGreaterEqual(close_received(ancillary), 1)
        finally:
            client.__exit__(None, None, None)
            receiver.close()

    @unittest.skipUnless(
        hasattr(socket, "SOCK_SEQPACKET"), "Linux seqpacket is required"
    )
    def test_raw_seqpacket_receive_deadline_is_bounded(self) -> None:
        sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client = integration.RawSeqpacketClient(Path("/fixture"), 0.05)
        client.socket = sender
        client.deadline = time.monotonic() + client.timeout
        started = time.monotonic()
        try:
            with self.assertRaises(TimeoutError):
                client.receive()
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            client.__exit__(None, None, None)
            receiver.close()

    def test_process_group_cleanup_is_bounded(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        started = time.monotonic()
        integration.terminate_process(process, grace=0.5)
        self.assertIsNotNone(process.returncode)
        self.assertLess(time.monotonic() - started, 2.0)


if __name__ == "__main__":
    unittest.main()
