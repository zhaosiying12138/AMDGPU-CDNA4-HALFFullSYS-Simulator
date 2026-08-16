#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Lightweight process-tree fixture for the gemsim instance supervisor."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from gemsim_live_registry import load_rank_launch


FAKE_GEM5 = r"""
import os
from pathlib import Path
import signal
import socket
import sys
import time

def argument(name):
    index = sys.argv.index(name)
    return sys.argv[index + 1]

endpoint = argument('--endpoint')
daemon_uuid = os.environ['FAKE_DAEMON_UUID']
job_uuid = argument('--job-uuid')
epoch = argument('--epoch')
rank = argument('--rank')
world = argument('--world-size')
if os.environ.get('FAKE_STALE_IDENTITY') == '1':
    epoch = str(int(epoch) + 1)
listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
listener.bind(endpoint)
os.chmod(endpoint, 0o600)
listener.listen(1)
print(
    'host-gpu-ready endpoint={} daemon_uuid={} job_uuid={} epoch={} rank={} world={} max_record=65536'.format(
        endpoint, daemon_uuid, job_uuid, epoch, rank, world
    ),
    flush=True,
)
while True:
    time.sleep(1)
"""


def launch_fake_gem5(
    stale_identity: bool = False,
    descriptor: dict | None = None,
) -> tuple[subprocess.Popen[bytes], Path]:
    run_dir = (
        Path(descriptor["paths"]["runtime_directory"])
        if descriptor is not None
        else Path(tempfile.mkdtemp(prefix=f"self-amdgpu-opencl-run.{os.getuid()}."))
    )
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_dir = run_dir / "m5out"
    cache_dir = run_dir / "cache"
    output_dir.mkdir(mode=0o700)
    cache_dir.mkdir(mode=0o700)
    (output_dir / "stats.txt").write_text("simTicks 1\n", encoding="ascii")
    endpoint = (
        Path(descriptor["paths"]["endpoint"])
        if descriptor is not None
        else run_dir / "bridge.sock"
    )
    trace_path = (
        Path(descriptor["paths"]["dispatch_trace_path"])
        if descriptor is not None
        else run_dir / "dispatch-trace.jsonl"
    )
    trace_path.write_text("", encoding="utf-8")
    epoch = (
        descriptor["epoch"]
        if descriptor is not None
        else ((time.monotonic_ns() ^ (os.getpid() << 32)) & ((1 << 63) - 1)) or 1
    )
    job_uuid = descriptor["job_uuid"] if descriptor is not None else uuid.uuid4().hex
    rank = descriptor["rank"] if descriptor is not None else 0
    world_size = descriptor["world_size"] if descriptor is not None else 1
    daemon_uuid = uuid.uuid4().hex
    command = [
        sys.executable,
        "-c",
        FAKE_GEM5,
        "--listener-mode=on",
        "--outdir",
        str(output_dir),
        "/fixture/host_dispatch.py",
        "--endpoint",
        str(endpoint),
        "--dispatch-trace-path",
        str(trace_path),
        "--epoch",
        str(epoch),
        "--job-uuid",
        job_uuid,
        "--rank",
        str(rank),
        "--world-size",
        str(world_size),
    ]
    environment = dict(os.environ)
    environment["FAKE_DAEMON_UUID"] = daemon_uuid
    environment["FAKE_STALE_IDENTITY"] = "1" if stale_identity else "0"
    log = (run_dir / "gem5.log").open("wb")
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
        env=environment,
        start_new_session=True,
    )
    log.close()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if endpoint.is_socket():
            return process, run_dir
        if process.poll() is not None:
            raise RuntimeError("fake gem5 exited before publishing its endpoint")
        time.sleep(0.01)
    raise RuntimeError("fake gem5 did not publish its endpoint")


def terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
    process.wait(timeout=2.0)


def main() -> int:
    if any(name.startswith("SAGR_") for name in os.environ):
        return 70
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        if os.environ.get(name) != "":
            return 71
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    if not cache.is_dir():
        return 72
    index = int(os.environ["GEMSIM_SUPERVISOR_INSTANCE"])
    mode = sys.argv[1]
    descriptor_path = os.environ.get("GEMSIM_RANK_LAUNCH_DESCRIPTOR")
    descriptor = load_rank_launch(Path(descriptor_path)) if descriptor_path else None
    if mode == "mutate-cache" and index == 0:
        cache.chmod(0o755)
        (cache / "unexpected-write").write_text("mutation\n", encoding="ascii")
    if mode == "delay-hold":
        time.sleep(0.60)
    process, run_dir = launch_fake_gem5(
        stale_identity=mode == "stale-epoch", descriptor=descriptor
    )
    (Path.cwd() / "worker.json").write_text(
        json.dumps(
            {
                "index": index,
                "cache": str(cache),
                "runtime_pid": process.pid,
                "run_dir": str(run_dir),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if mode == "tamper-descriptor" and descriptor_path and descriptor is not None:
        tampered = json.loads(json.dumps(descriptor))
        tampered["epoch"] += 1
        path = Path(descriptor_path)
        path.chmod(0o600)
        path.write_text(
            json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
    if mode == "fail-one" and index == 0:
        time.sleep(0.15)
        # Deliberately abandon the separate-session daemon. The subreaper owns
        # cleanup even though the runner did not run its normal atexit path.
        return 9
    if mode == "daemon-dies" and index == 0:
        time.sleep(0.30)
        terminate(process)
        time.sleep(60.0)
        return 0
    if mode in ("hold", "delay-hold", "daemon-dies"):
        time.sleep(60.0)
        return 0
    delay = 0.55 if mode == "fail-one" else 0.20
    time.sleep(delay)
    (Path.cwd() / "completed").write_text("yes\n", encoding="ascii")
    terminate(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
