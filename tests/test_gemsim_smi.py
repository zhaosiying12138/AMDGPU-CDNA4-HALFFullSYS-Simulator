"""Host-only tests for the KMD-free 16-device simulator inventory."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path
import socket
import stat
import struct
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/gemsim_smi.py"
SPEC = importlib.util.spec_from_file_location("gemsim_smi_for_tests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SMI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMI)


def make_record(
    *,
    device: int,
    endpoint: Path,
    daemon_start_delta: int = 0,
    corrupt_digest: bool = False,
) -> bytes:
    pid = os.getpid()
    start = SMI._proc_start_time(pid)
    assert start is not None
    executable = Path(f"/proc/{pid}/exe").stat()
    endpoint_bytes = str(endpoint).encode("ascii")
    payload = bytearray(SMI.RECORD_BYTES)
    payload[:8] = SMI.RECORD_MAGIC
    struct.pack_into(">IIII", payload, 8, SMI.RECORD_VERSION, SMI.RECORD_BYTES, device, 1)
    struct.pack_into(">II", payload, 24, pid, pid)
    struct.pack_into(">QQQQ", payload, 32, start, start + daemon_start_delta, 7, 11)
    struct.pack_into(">II", payload, 64, 1, 4)
    payload[72:88] = bytes.fromhex("11" * 16)
    payload[88:104] = bytes.fromhex("22" * 16)
    struct.pack_into(">QQQ", payload, 104, executable.st_dev, executable.st_ino, time.monotonic_ns())
    struct.pack_into(">I", payload, 128, len(endpoint_bytes))
    payload[136 : 136 + len(endpoint_bytes)] = endpoint_bytes
    digest = hashlib.sha256(payload[: SMI.RECORD_PAYLOAD_BYTES]).digest()
    payload[SMI.RECORD_PAYLOAD_BYTES :] = digest
    if corrupt_digest:
        payload[-1] ^= 1
    return bytes(payload)


class HeldDeviceRecord:
    def __init__(self, root: Path, device: int, payload: bytes):
        self.path = root / f"device-{device:02d}.bin"
        self.descriptor = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        os.write(self.descriptor, payload)
        os.fsync(self.descriptor)
        fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def close(self) -> None:
        if self.descriptor >= 0:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "HeldDeviceRecord":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class GemsimSMITest(unittest.TestCase):
    def test_absent_state_directory_reports_sixteen_devices_off(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "absent"
            document = SMI.device_document(missing)
        self.assertEqual(document["schema"], "amdgpu-sim.gemsim-smi.v2")
        self.assertEqual(document["logical_device_count"], 16)
        self.assertEqual(document["on_count"], 0)
        self.assertEqual(document["off_count"], 16)
        self.assertEqual([item["device"] for item in document["devices"]], list(range(16)))
        self.assertTrue(all(item["status"] == "OFF" for item in document["devices"]))

    def test_held_live_record_transitions_off_on_off(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chmod(root, 0o700)
            endpoint = root / "bridge.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(endpoint))
            os.chmod(endpoint, 0o600)
            lease = HeldDeviceRecord(root, 3, make_record(device=3, endpoint=endpoint))
            try:
                on = SMI.device_document(root)
                self.assertEqual(on["on_count"], 1)
                device = on["devices"][3]
                self.assertEqual(device["status"], "ON")
                self.assertEqual(device["reason"], "managed_gem5_ready")
                self.assertEqual(device["daemon_pid"], os.getpid())
                self.assertEqual(device["rank"], 1)
                self.assertEqual(device["world_size"], 4)
                self.assertTrue(device["exact_topology"])
            finally:
                lease.close()
                listener.close()
            off = SMI.device_document(root)
            self.assertEqual(off["on_count"], 0)
            self.assertEqual(off["devices"][3]["reason"], "unused")

    def test_locked_record_requires_current_process_and_private_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chmod(root, 0o700)
            endpoint = root / "bridge.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(endpoint))
            os.chmod(endpoint, 0o600)
            with HeldDeviceRecord(
                root,
                2,
                make_record(device=2, endpoint=endpoint, daemon_start_delta=1),
            ):
                stale = SMI.device_document(root)
                self.assertEqual(stale["devices"][2]["status"], "OFF")
                self.assertEqual(
                    stale["devices"][2]["reason"], "daemon_identity_unavailable"
                )
            listener.close()

    def test_locked_record_digest_corruption_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chmod(root, 0o700)
            endpoint = root / "bridge.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(endpoint))
            os.chmod(endpoint, 0o600)
            with HeldDeviceRecord(
                root,
                5,
                make_record(device=5, endpoint=endpoint, corrupt_digest=True),
            ):
                with self.assertRaisesRegex(SMI.SMIError, "digest differs"):
                    SMI.device_document(root)
            listener.close()

    def test_unsafe_state_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chmod(root, 0o755)
            with self.assertRaisesRegex(SMI.SMIError, "directory is unsafe"):
                SMI.device_document(root)

    def test_table_has_one_row_per_logical_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "absent"
            table = SMI._device_table(SMI.device_document(root))
        self.assertIn("GemSim devices: 0 ON, 16 OFF", table)
        rows = table.splitlines()
        self.assertEqual(len(rows), 19)
        self.assertTrue(rows[-1].lstrip().startswith("15"))


if __name__ == "__main__":
    unittest.main()
