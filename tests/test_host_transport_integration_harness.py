# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import argparse
import array
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
        start_wait_seconds=1.0,
        process_timeout_seconds=1.0,
        client_timeout_ms=500,
        server_startup_timeout_ms=2000,
        server_handshake_timeout_ms=500,
        server_run_timeout_ms=2000,
        hold_ms=250,
    )


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
