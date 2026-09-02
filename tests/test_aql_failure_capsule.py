#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aql_failure_capsule", ROOT / "tools/aql_failure_capsule.py"
)
assert SPEC is not None and SPEC.loader is not None
CAPSULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CAPSULE
SPEC.loader.exec_module(CAPSULE)


class Fixture:
    PID = 4242
    GEM5_PID = 4343
    MQD_VA = 0x10000
    RING_VA = 0x20000
    CODE_VA = 0x30000
    KERNARG_VA = 0x40000
    SIGNAL_VA = 0x50000
    DATA_VA = 0x60000
    PAGE = 4096
    DOORBELL_BASE = 10 * PAGE

    def __init__(self, root: Path):
        self.root = root
        self.proc_root = root / "proc"
        self.proc = self.proc_root / str(self.PID)
        (self.proc / "fd").mkdir(parents=True)
        (self.proc / "fdinfo").mkdir()
        stat_fields = ["T"] + ["0"] * 18 + ["123456"]
        (self.proc / "stat").write_text(
            f"{self.PID} (fixture worker) " + " ".join(stat_fields) + "\n",
            encoding="ascii",
        )
        (self.proc / "status").write_text(
            "Name:\tfixture\nState:\tT (stopped)\nUid:\t1000\t1000\t1000\t1000\n",
            encoding="ascii",
        )
        (self.proc / "maps").write_text(
            "10000-70000 rw-s 00000000 00:01 1 /memfd:self-amdgpu-hsakmt-model (deleted)\n",
            encoding="ascii",
        )
        (self.proc / "cmdline").write_bytes(b"fixture-worker\x00--test\x00")
        (self.proc / "exe").symlink_to("/fixture/worker")

        self.backing = root / "backing.memfd"
        self.backing.write_bytes(b"\x00" * (12 * self.PAGE))
        self.worker = root / "worker.log"
        self.trace = root / "dispatch-trace.jsonl"
        self.gem5_log = root / "gem5.log"
        self.output = root / "capsule"
        self._write_memory()
        self._write_logs()

    def _write_at(self, offset: int, payload: bytes) -> None:
        with self.backing.open("r+b") as stream:
            stream.seek(offset)
            stream.write(payload)

    def _packet(self, *, data_pointer: int) -> bytes:
        return struct.pack(
            "<6H5I4Q",
            2,
            1,
            64,
            1,
            1,
            0,
            64,
            1,
            1,
            0,
            0,
            self.CODE_VA,
            self.KERNARG_VA,
            0,
            self.SIGNAL_VA,
        )

    def _write_memory(self) -> None:
        mqd = bytearray(self.PAGE)
        struct.pack_into("<Q", mqd, 56, 4)
        struct.pack_into("<Q", mqd, 128, 2)
        struct.pack_into("<I", mqd, 136, 128)
        self._write_at(0, bytes(mqd))

        failing = self._packet(data_pointer=self.DATA_VA + 16)
        following = self._packet(data_pointer=self.DATA_VA + 24)
        self._write_at(self.PAGE + 2 * 64, failing)
        self._write_at(self.PAGE + 3 * 64, following)

        descriptor = bytearray(64)
        struct.pack_into("<III4xq", descriptor, 0, 0, 0, 16, 64)
        code = bytearray(self.PAGE)
        code[:64] = descriptor
        code[64:72] = b"CODETEST"
        self._write_at(2 * self.PAGE, bytes(code))
        self._write_at(
            3 * self.PAGE,
            struct.pack("<QQ", self.DATA_VA + 16, 0x1122334455667788),
        )
        signal_payload = bytearray(64)
        struct.pack_into("<qq", signal_payload, 0, 1, 1)
        self._write_at(4 * self.PAGE, bytes(signal_payload))
        self._write_at(5 * self.PAGE, bytes(range(256)) * 16)
        self._write_at(self.DOORBELL_BASE, struct.pack("<Q", 3))
        self._write_at(
            self.DOORBELL_BASE + CAPSULE.COMPLETION_OFFSET_FROM_DOORBELL,
            struct.pack("<Q", 2),
        )

    def _allocation_line(
        self, va: int, offset: int, handle: int, flags: int = 2
    ) -> str:
        return (
            f"hsakmt-model pid={self.PID} phase=leave request=0xc0284b16 "
            f"result=0 errno=0 gpu_id=38144 flags=0x{flags:x} va=0x{va:x} "
            f"size={self.PAGE} mmap_offset=0x{offset:x} handle={handle}"
        )

    def _write_logs(self) -> None:
        lines = [
            self._allocation_line(self.MQD_VA, 0, 1001),
            self._allocation_line(self.RING_VA, self.PAGE, 1002),
            self._allocation_line(self.CODE_VA, 2 * self.PAGE, 1003),
            self._allocation_line(self.KERNARG_VA, 3 * self.PAGE, 1004),
            self._allocation_line(self.SIGNAL_VA, 4 * self.PAGE, 1005),
            self._allocation_line(self.DATA_VA, 5 * self.PAGE, 1006),
            (
                f"hsakmt-model pid={self.PID} phase=leave request=0xc0604b02 "
                f"result=0 errno=0 gpu_id=38144 queue_type=2 ring=0x{self.RING_VA:x} "
                f"ring_size={self.PAGE} read=0x{self.MQD_VA + 128:x} "
                f"write=0x{self.MQD_VA + 56:x} percentage=100 priority=7 "
                f"queue_id=1 doorbell_offset=0x{self.DOORBELL_BASE:x}"
            ),
            (
                f"hsakmt-model pid={self.PID} phase=queue-doorbell queue_id=1 "
                "slot=0 doorbell=3 notification=4 completion=2 status=0"
            ),
        ]
        self.worker.write_text("\n".join(lines) + "\n", encoding="ascii")
        self.trace.write_text(
            json.dumps(
                {
                    "schema": "amdgpu-sim.native-kernel-execution-trace.v1",
                    "event": "native_execution_retired",
                    "queue_index": 1,
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="ascii",
        )
        self.gem5_log.write_text("gem5 started\npanic: fixture failure\n", encoding="ascii")

    def argv(self, output: Path | None = None) -> list[str]:
        return [
            "capture",
            "--pid",
            str(self.PID),
            "--gem5-pid",
            str(self.GEM5_PID),
            "--worker-log",
            str(self.worker),
            "--dispatch-trace",
            str(self.trace),
            "--log",
            f"gem5={self.gem5_log}",
            "--backing",
            str(self.backing),
            "--proc-root",
            str(self.proc_root),
            "--output",
            str(output or self.output),
        ]


class AqlFailureCapsuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = Fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_capture_freezes_queue_head_resources_and_verifies(self) -> None:
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 0)
        manifest = json.loads((self.fixture.output / "manifest.json").read_text())
        self.assertEqual(manifest["schema"], CAPSULE.CAPSULE_SCHEMA)
        self.assertEqual(manifest["selection"]["failing_packet_index"], 2)
        self.assertEqual(manifest["selection"]["next_packet_index"], 3)
        self.assertEqual(manifest["selection"]["pending_count"], 2)
        self.assertEqual(manifest["selection"]["frontiers"]["doorbell"], 3)
        self.assertEqual(
            manifest["selection"]["frontiers"]["bridge_completion"], 2
        )
        self.assertEqual(manifest["packet"]["kernel_object"], Fixture.CODE_VA)
        self.assertEqual(manifest["descriptor"]["effective_entry_va"], Fixture.CODE_VA + 64)
        self.assertFalse(manifest["replay"]["eligible"])
        self.assertEqual(manifest["first_failure"]["log_role"], "gem5")

        registry = json.loads((self.fixture.output / "kmt-registry.json").read_text())
        target = next(
            allocation
            for allocation in registry["referenced_allocations"]
            if "kernarg_pointer_candidate" in allocation["roles"]
        )
        self.assertEqual(target["gpu_va"], Fixture.DATA_VA)
        self.assertEqual(len(target["content_sha256"]), 64)
        self.assertTrue(
            (self.fixture.output / "objects/resident-code-allocation.bin").is_file()
        )
        verified = CAPSULE.verify(self.fixture.output)
        self.assertEqual(verified["status"], "verified")
        self.assertGreater(verified["artifact_count"], 10)

    def test_selection_excludes_reserved_but_unpublished_slots(self) -> None:
        with self.fixture.backing.open("r+b") as stream:
            stream.seek(56)
            stream.write(struct.pack("<Q", 5))
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 0)
        manifest = json.loads((self.fixture.output / "manifest.json").read_text())
        self.assertEqual(manifest["selection"]["write_index"], 4)
        self.assertEqual(manifest["selection"]["pending_count"], 2)
        self.assertEqual(
            manifest["selection"]["frontiers"]["producer_reserved_write"], 5
        )

    def test_selection_uses_bridge_completion_when_producer_read_lags(self) -> None:
        self.fixture._write_at(
            Fixture.DOORBELL_BASE + CAPSULE.COMPLETION_OFFSET_FROM_DOORBELL,
            struct.pack("<Q", 3),
        )
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 0)
        manifest = json.loads((self.fixture.output / "manifest.json").read_text())
        self.assertEqual(manifest["selection"]["failing_packet_index"], 3)
        self.assertEqual(manifest["selection"]["pending_count"], 1)
        self.assertEqual(manifest["selection"]["frontiers"]["producer_read"], 2)

    def test_next_non_kernel_packet_is_frozen_without_kernel_decode(self) -> None:
        with self.fixture.backing.open("r+b") as stream:
            stream.seek(Fixture.PAGE + 3 * CAPSULE.PACKET_BYTES)
            stream.write(struct.pack("<H", 3))
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 0)
        manifest = json.loads((self.fixture.output / "manifest.json").read_text())
        self.assertEqual(manifest["next_packet"]["packet_type"], 3)
        self.assertEqual(
            manifest["next_packet"]["resource_decode"],
            "unsupported_non_kernel_packet",
        )

    def test_non_kernel_queue_head_fails_closed(self) -> None:
        with self.fixture.backing.open("r+b") as stream:
            stream.seek(Fixture.PAGE + 2 * CAPSULE.PACKET_BYTES)
            stream.write(struct.pack("<H", 3))
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 1)
        self.assertFalse(self.fixture.output.exists())

    def test_scratch_range_must_fit_one_allocation(self) -> None:
        with self.fixture.backing.open("r+b") as stream:
            stream.seek(160)
            stream.write(struct.pack("<QQ", Fixture.DATA_VA + Fixture.PAGE - 8, 16))
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 1)
        self.assertFalse(self.fixture.output.exists())

    def test_kernarg_budget_is_checked_before_read(self) -> None:
        with self.fixture.backing.open("r+b") as stream:
            stream.seek(2 * Fixture.PAGE + 8)
            stream.write(struct.pack("<I", 128))
        argv = self.fixture.argv() + ["--max-kernarg-bytes", "64"]
        self.assertEqual(CAPSULE.main(argv), 1)
        self.assertFalse(self.fixture.output.exists())

    def test_zero_sized_kernarg_captures_descriptor_preload_range(self) -> None:
        preload_length = 2
        preload_offset = 1
        with self.fixture.backing.open("r+b") as stream:
            stream.seek(2 * Fixture.PAGE + 8)
            stream.write(struct.pack("<I", 0))
            stream.seek(2 * Fixture.PAGE + 58)
            stream.write(
                struct.pack("<H", preload_length | (preload_offset << 7))
            )
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 0)
        manifest = json.loads((self.fixture.output / "manifest.json").read_text())
        self.assertEqual(manifest["descriptor"]["kernarg_size"], 0)
        self.assertEqual(manifest["descriptor"]["kernarg_read_bytes"], 12)
        self.assertEqual(
            (self.fixture.output / "objects/kernarg.bin").stat().st_size, 12
        )
        self.assertEqual(CAPSULE.verify(self.fixture.output)["status"], "verified")

    def test_capture_requires_gem5_to_have_exited(self) -> None:
        (self.fixture.proc_root / str(Fixture.GEM5_PID)).mkdir()
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 1)
        self.assertFalse(self.fixture.output.exists())

    def test_log_role_length_preserves_artifact_role_contract(self) -> None:
        accepted = self.root / "accepted-role"
        accepted_argv = self.fixture.argv(accepted) + [
            "--log",
            f"{'a' * 60}={self.fixture.gem5_log}",
        ]
        self.assertEqual(CAPSULE.main(accepted_argv), 0)
        self.assertEqual(CAPSULE.verify(accepted)["status"], "verified")

        rejected = self.root / "rejected-role"
        rejected_argv = self.fixture.argv(rejected) + [
            "--log",
            f"{'a' * 61}={self.fixture.gem5_log}",
        ]
        self.assertEqual(CAPSULE.main(rejected_argv), 1)
        self.assertFalse(rejected.exists())

    def test_existing_output_is_never_replaced(self) -> None:
        self.fixture.output.mkdir()
        marker = self.fixture.output / "marker"
        marker.write_text("keep", encoding="ascii")
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 1)
        self.assertEqual(marker.read_text(encoding="ascii"), "keep")

    def test_verifier_rejects_artifact_tamper(self) -> None:
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 0)
        packet = self.fixture.output / "objects/failing-packet.bin"
        payload = bytearray(packet.read_bytes())
        payload[-1] ^= 1
        packet.write_bytes(payload)
        with self.assertRaisesRegex(CAPSULE.CapsuleError, "artifact hash differs"):
            CAPSULE.verify(self.fixture.output)

    def test_verifier_rejects_semantically_incomplete_manifest(self) -> None:
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 0)
        manifest_path = self.fixture.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["status"]
        payload = CAPSULE.canonical_json(manifest)
        manifest_path.write_bytes(payload)
        (self.fixture.output / "manifest.sha256").write_text(
            f"{CAPSULE.sha256_bytes(payload)}  manifest.json\n", encoding="ascii"
        )
        with self.assertRaisesRegex(CAPSULE.CapsuleError, "manifest keys differ"):
            CAPSULE.verify(self.fixture.output)

    def test_verifier_redecodes_packet_binary(self) -> None:
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 0)
        manifest_path = self.fixture.output / "manifest.json"
        packet_path = self.fixture.output / "objects/failing-packet.json"
        manifest = json.loads(manifest_path.read_text())
        packet_metadata = json.loads(packet_path.read_text())
        packet_metadata["grid"] = [123456789, 1, 1]
        packet_payload = CAPSULE.canonical_json(packet_metadata)
        packet_path.write_bytes(packet_payload)
        manifest["packet"]["grid"] = [123456789, 1, 1]
        artifact = next(
            record
            for record in manifest["artifacts"]
            if record["path"] == "objects/failing-packet.json"
        )
        artifact["bytes"] = len(packet_payload)
        artifact["sha256"] = CAPSULE.sha256_bytes(packet_payload)
        manifest_payload = CAPSULE.canonical_json(manifest)
        manifest_path.write_bytes(manifest_payload)
        (self.fixture.output / "manifest.sha256").write_text(
            f"{CAPSULE.sha256_bytes(manifest_payload)}  manifest.json\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(CAPSULE.CapsuleError, "packet binary decode differs"):
            CAPSULE.verify(self.fixture.output)

    def test_verifier_redecodes_mqd_binary(self) -> None:
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 0)
        manifest_path = self.fixture.output / "manifest.json"
        mqd_path = self.fixture.output / "objects/mqd.bin"
        manifest = json.loads(manifest_path.read_text())
        mqd_payload = bytearray(mqd_path.read_bytes())
        struct.pack_into("<Q", mqd_payload, 128, 999)
        mqd_path.write_bytes(mqd_payload)
        artifact = next(
            record
            for record in manifest["artifacts"]
            if record["path"] == "objects/mqd.bin"
        )
        artifact["sha256"] = CAPSULE.sha256_bytes(mqd_payload)
        manifest_payload = CAPSULE.canonical_json(manifest)
        manifest_path.write_bytes(manifest_payload)
        (self.fixture.output / "manifest.sha256").write_text(
            f"{CAPSULE.sha256_bytes(manifest_payload)}  manifest.json\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(CAPSULE.CapsuleError, "MQD read index differs"):
            CAPSULE.verify(self.fixture.output)

    def test_verifier_validates_registry_hash_ranges(self) -> None:
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 0)
        manifest_path = self.fixture.output / "manifest.json"
        registry_path = self.fixture.output / "kmt-registry.json"
        manifest = json.loads(manifest_path.read_text())
        registry = json.loads(registry_path.read_text())
        allocation = registry["referenced_allocations"][0]
        allocation["content_hash_range"]["bytes"] -= 1
        registry_payload = CAPSULE.canonical_json(registry)
        registry_path.write_bytes(registry_payload)
        artifact = next(
            record
            for record in manifest["artifacts"]
            if record["path"] == "kmt-registry.json"
        )
        artifact["bytes"] = len(registry_payload)
        artifact["sha256"] = CAPSULE.sha256_bytes(registry_payload)
        manifest_payload = CAPSULE.canonical_json(manifest)
        manifest_path.write_bytes(manifest_payload)
        (self.fixture.output / "manifest.sha256").write_text(
            f"{CAPSULE.sha256_bytes(manifest_payload)}  manifest.json\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(CAPSULE.CapsuleError, "content hash range differs"):
            CAPSULE.verify(self.fixture.output)

    def test_verifier_rejects_truncated_resident_code_copy(self) -> None:
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 0)
        manifest_path = self.fixture.output / "manifest.json"
        registry_path = self.fixture.output / "kmt-registry.json"
        code_path = self.fixture.output / "objects/resident-code-allocation.bin"
        manifest = json.loads(manifest_path.read_text())
        registry = json.loads(registry_path.read_text())
        code_payload = code_path.read_bytes()[:64]
        code_path.write_bytes(code_payload)
        code_sha = CAPSULE.sha256_bytes(code_payload)
        code_artifact = next(
            record
            for record in manifest["artifacts"]
            if record["path"] == "objects/resident-code-allocation.bin"
        )
        code_artifact["bytes"] = len(code_payload)
        code_artifact["sha256"] = code_sha
        resident_code = next(
            allocation
            for allocation in registry["referenced_allocations"]
            if "resident_code" in allocation["roles"]
        )
        resident_code["content_sha256"] = code_sha
        registry_payload = CAPSULE.canonical_json(registry)
        registry_path.write_bytes(registry_payload)
        registry_artifact = next(
            record
            for record in manifest["artifacts"]
            if record["path"] == "kmt-registry.json"
        )
        registry_artifact["bytes"] = len(registry_payload)
        registry_artifact["sha256"] = CAPSULE.sha256_bytes(registry_payload)
        manifest_payload = CAPSULE.canonical_json(manifest)
        manifest_path.write_bytes(manifest_payload)
        (self.fixture.output / "manifest.sha256").write_text(
            f"{CAPSULE.sha256_bytes(manifest_payload)}  manifest.json\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(CAPSULE.CapsuleError, "allocation size differs"):
            CAPSULE.verify(self.fixture.output)

    def test_versioned_v2_descriptor_uses_legacy_segment_offsets(self) -> None:
        payload = bytearray(256)
        struct.pack_into("<II4H", payload, 0, 1, 2, 1, 9, 5, 0)
        struct.pack_into("<q", payload, 16, 256)
        struct.pack_into("<I", payload, 60, 32)
        struct.pack_into("<I", payload, 64, 64)
        struct.pack_into("<I", payload, 68, 0)
        struct.pack_into("<Q", payload, 72, 280)
        decoded = CAPSULE.decode_descriptor(bytes(payload), Fixture.CODE_VA)
        self.assertEqual(decoded["descriptor_abi"], "code_object_v2")
        self.assertEqual(decoded["private_segment_fixed_size"], 32)
        self.assertEqual(decoded["group_segment_fixed_size"], 64)
        self.assertEqual(decoded["kernarg_size"], 280)
        self.assertEqual(decoded["effective_entry_va"], Fixture.CODE_VA + 256)

    def test_v3_descriptor_decodes_properties_and_preload_independently(self) -> None:
        payload = bytearray(64)
        struct.pack_into("<III4xq", payload, 0, 0, 0, 16, 64)
        struct.pack_into("<H", payload, 56, 0x0008)

        without_preload = CAPSULE.decode_descriptor(bytes(payload), Fixture.CODE_VA)
        self.assertEqual(without_preload["kernarg_preload_spec_length"], 0)
        self.assertEqual(without_preload["kernarg_preload_spec_offset"], 0)
        self.assertEqual(without_preload["effective_entry_va"], Fixture.CODE_VA + 64)

        preload_length = 12
        preload_offset = 17
        struct.pack_into("<H", payload, 58, preload_length | (preload_offset << 7))
        with_preload = CAPSULE.decode_descriptor(bytes(payload), Fixture.CODE_VA)
        self.assertEqual(
            with_preload["kernarg_preload_spec_length"], preload_length
        )
        self.assertEqual(
            with_preload["kernarg_preload_spec_offset"], preload_offset
        )
        self.assertEqual(with_preload["effective_entry_va"], Fixture.CODE_VA + 320)

    def test_registry_ignores_other_pids_and_incomplete_tail(self) -> None:
        payload = self.fixture.worker.read_bytes()
        other = self.fixture._allocation_line(Fixture.DATA_VA, 0, 9999).replace(
            f"pid={Fixture.PID}", "pid=999"
        )
        partial = self.fixture._allocation_line(0x90000, 0, 9998)
        registry = CAPSULE.parse_registry(
            other.encode("ascii") + b"\n" + payload + partial.encode("ascii"),
            Fixture.PID,
        )
        self.assertEqual(registry.pid, Fixture.PID)
        self.assertEqual(len(registry.allocations), 6)
        self.assertTrue(registry.trailing_partial_line)

    def test_multiple_pending_queues_require_explicit_selection(self) -> None:
        second_mqd = 0x70000
        second_ring = 0x80000
        with self.fixture.backing.open("ab") as stream:
            stream.write(b"\x00" * (2 * Fixture.PAGE))
        mqd = bytearray(Fixture.PAGE)
        struct.pack_into("<Q", mqd, 56, 1)
        struct.pack_into("<Q", mqd, 128, 0)
        struct.pack_into("<I", mqd, 136, 128)
        with self.fixture.backing.open("r+b") as stream:
            stream.seek(6 * Fixture.PAGE)
            stream.write(mqd)
            stream.seek(7 * Fixture.PAGE)
            stream.write(self.fixture._packet(data_pointer=Fixture.DATA_VA))
            stream.seek(Fixture.DOORBELL_BASE + 8)
            stream.write(struct.pack("<Q", 0))
            stream.seek(
                Fixture.DOORBELL_BASE
                + CAPSULE.COMPLETION_OFFSET_FROM_DOORBELL
                + 8
            )
            stream.write(struct.pack("<Q", 0))
        with self.fixture.worker.open("a", encoding="ascii") as stream:
            stream.write(self.fixture._allocation_line(second_mqd, 6 * Fixture.PAGE, 1007) + "\n")
            stream.write(self.fixture._allocation_line(second_ring, 7 * Fixture.PAGE, 1008) + "\n")
            stream.write(
                f"hsakmt-model pid={Fixture.PID} phase=leave request=0xc0604b02 "
                f"result=0 errno=0 gpu_id=38144 queue_type=2 ring=0x{second_ring:x} "
                f"ring_size={Fixture.PAGE} read=0x{second_mqd + 128:x} "
                f"write=0x{second_mqd + 56:x} percentage=100 priority=7 "
                f"queue_id=2 doorbell_offset=0x{Fixture.DOORBELL_BASE + 8:x}\n"
            )
        self.assertEqual(CAPSULE.main(self.fixture.argv()), 1)
        selected = self.root / "selected"
        argv = self.fixture.argv(selected) + ["--queue-id", "1"]
        self.assertEqual(CAPSULE.main(argv), 0)
        manifest = json.loads((selected / "manifest.json").read_text())
        self.assertEqual(manifest["selection"]["queue_id"], 1)

    @unittest.skipUnless(Path("/proc/self/status").is_file(), "live procfs is required")
    def test_live_proc_capture_stops_and_restores_exact_spawned_pid(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            payload = self.fixture.worker.read_text(encoding="ascii").replace(
                f"pid={Fixture.PID}", f"pid={process.pid}"
            )
            self.fixture.worker.write_text(payload, encoding="ascii")
            output = self.root / "live-capsule"
            argv = [
                "capture",
                "--pid",
                str(process.pid),
                "--gem5-pid",
                "99999999",
                "--worker-log",
                str(self.fixture.worker),
                "--dispatch-trace",
                str(self.fixture.trace),
                "--backing",
                str(self.fixture.backing),
                "--output",
                str(output),
                "--freeze-target",
            ]
            self.assertEqual(CAPSULE.main(argv), 0)
            self.assertIsNone(process.poll())
            state = Path(f"/proc/{process.pid}/status").read_text()
            self.assertNotIn("State:\tT", state)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["process"]["stopped_by_capture"])
            self.assertEqual(manifest["process"]["pid"], process.pid)
        finally:
            process.terminate()
            process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
