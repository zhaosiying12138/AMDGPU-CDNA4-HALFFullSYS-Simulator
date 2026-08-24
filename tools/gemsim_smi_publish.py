#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Publish one GemSim SMI device record and hold its lease.

The runtime's managed-session path writes these records itself; a
standalone listener-mode gem5 started by ``gem5-session`` bypasses it,
so ``rocm-smi`` would show every slot OFF.  This holder daemon writes
the same 320-byte signed-format record into the per-UID SMI state
directory and holds an exclusive flock on it for its lifetime --
exactly the liveness contract the reader enforces -- and unlinks the
record on exit.  Kill the holder and the slot automatically reads OFF.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import signal
import stat
import struct
import sys
import time
from pathlib import Path

RECORD_BYTES = 320
RECORD_PAYLOAD_BYTES = 288
ENDPOINT_BYTES = 112
RECORD_MAGIC = b"SAGRSMI1"
RECORD_VERSION = 1
DEVICE_COUNT = 16


def _proc_start_time(pid: int) -> int:
    text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    close = text.rfind(")")
    fields = text[close + 2 :].split()
    return int(fields[19])


def build_payload(slot: int, daemon_pid: int, rank: int, world: int,
                  endpoint: str, job_uuid: bytes, daemon_uuid: bytes) -> bytes:
    owner_pid = os.getpid()
    buf = bytearray(RECORD_PAYLOAD_BYTES)
    buf[0:8] = RECORD_MAGIC
    struct.pack_into(">IIII", buf, 8, RECORD_VERSION, RECORD_BYTES, slot, 0)
    struct.pack_into(">II", buf, 24, owner_pid, daemon_pid)
    struct.pack_into(
        ">QQQQ", buf, 32,
        _proc_start_time(owner_pid), _proc_start_time(daemon_pid),
        1,  # epoch (nonzero)
        1,  # connection_id (nonzero)
    )
    struct.pack_into(">II", buf, 64, rank, world)
    buf[72:88] = job_uuid
    buf[88:104] = daemon_uuid
    exe = os.stat(f"/proc/{daemon_pid}/exe")
    struct.pack_into(
        ">QQQ", buf, 104,
        exe.st_dev, exe.st_ino,
        time.clock_gettime_ns(time.CLOCK_MONOTONIC),
    )
    encoded = endpoint.encode("ascii")
    struct.pack_into(">I", buf, 128, len(encoded))
    buf[132:136] = bytes(4)
    buf[136 : 136 + len(encoded)] = encoded
    return bytes(buf) + hashlib.sha256(bytes(buf)).digest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--daemon-pid", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--job-uuid", required=True, help="32 hex digits")
    args = parser.parse_args()

    if not (0 <= args.slot < DEVICE_COUNT) or args.rank >= args.world:
        print("invalid slot/rank/world", file=sys.stderr)
        return 2
    if len(args.job_uuid) != 32:
        print("job uuid must be 32 hex digits", file=sys.stderr)
        return 2
    job_uuid = bytes.fromhex(args.job_uuid)
    daemon_uuid = hashlib.sha256(
        f"{args.job_uuid}:{args.slot}".encode()
    ).digest()[:16]

    state = Path(f"/tmp/amdgpu-sim-smi-{os.getuid()}")
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(state, 0o700)
    record = state / f"device-{args.slot:02d}.bin"

    descriptor = os.open(
        record, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600
    )
    os.fchmod(descriptor, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    payload = build_payload(
        args.slot, args.daemon_pid, args.rank, args.world,
        args.endpoint, job_uuid, daemon_uuid,
    )
    os.pwrite(descriptor, payload, 0)
    os.fsync(descriptor)

    def release(signum, frame):
        try:
            record.unlink()
        except FileNotFoundError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, release)
    signal.signal(signal.SIGINT, release)
    print(
        f"[smi-publish] slot {args.slot} ON: daemon {args.daemon_pid} "
        f"rank {args.rank}/{args.world} endpoint {args.endpoint}",
        flush=True,
    )
    while True:
        # Refresh the record's monotonic timestamp so the lease visibly
        # belongs to this holder, and die as soon as the daemon does.
        try:
            os.kill(args.daemon_pid, 0)
        except OSError:
            release(0, None)
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
