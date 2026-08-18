#!/usr/bin/env python3
"""Measure what gem5 costs while the host gives it no GPU work.

A live Qwen3.5 TP1 trace showed 90.6% of simulated time falling *between*
retired dispatches rather than inside them, with a single 372-billion-tick gap
accounting for 81.8% on its own. Kernel execution was only 9.4%. That points at
the service loop rather than at the GPU model, but simulated ticks alone cannot
prove it: ticks advancing does not mean instructions are retiring, and
instructions retiring does not mean control flow is sane.

So this probe measures three independent quantities across an idle window in
which the host deliberately submits nothing:

  * gem5 host CPU time  -- real seconds burned with no work to do
  * simulated ticks     -- how much simulated time the idle window invents
  * x86 instructions    -- whether the quiesced SE context is actually retiring
                           instructions while idle

An idle simulator should spend approximately none of all three. Any of them
growing linearly with idle wall-clock is the defect.

The probe starts one managed gem5, connects with the ordinary handshake tool,
holds the connection open without submitting work, and samples. It never
dispatches a kernel, so anything it observes is pure idle overhead.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


SCHEMA = "amdgpu-sim.gem5-idle-cost.v1"
CLOCK_TICKS = os.sysconf("SC_CLK_TCK")


def process_cpu_seconds(pid: int) -> float | None:
    """utime+stime for a pid, in seconds."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    # After the comm field, index 11 is utime and 12 is stime.
    try:
        return (int(fields[11]) + int(fields[12])) / CLOCK_TICKS
    except (IndexError, ValueError):
        return None


def read_stat_counter(stats_path: Path, name: str) -> int | None:
    try:
        text = stats_path.read_text()
    except OSError:
        return None
    match = re.search(rf"^{re.escape(name)}\s+(\d+)", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gem5", default=os.environ.get("SAGR_MANAGED_GEM5", ""))
    parser.add_argument("--config", default=os.environ.get("SAGR_MANAGED_GEM5_CONFIG", ""))
    parser.add_argument("--handshake", default="", help="path to sagr-handshake")
    parser.add_argument("--idle-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)

    for name, value in (("--gem5", args.gem5), ("--config", args.config)):
        if not value or not Path(value).exists():
            print(f"gem5-idle-cost: {name} is missing or unset", file=sys.stderr)
            return 2
    handshake = args.handshake or shutil.which("sagr-handshake") or ""
    if not handshake or not Path(handshake).exists():
        print("gem5-idle-cost: sagr-handshake not found", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="gem5-idle-cost.") as directory:
        run = Path(directory)
        run.chmod(0o700)
        endpoint = run / "bridge.sock"
        outdir = run / "m5out"
        process = subprocess.Popen(
            [
                args.gem5, "--listener-mode=on", "--outdir", str(outdir), args.config,
                "--endpoint", str(endpoint),
                "--dispatch-trace-path", str(run / "dispatch-trace.jsonl"),
                "--epoch", "987654321",
                "--job-uuid", "bbbbbbbbbbbb4bbbbbbbbbbbbbbbbbbb",
                "--rank", "0", "--world-size", "1",
                "--startup-timeout-ms", "30000",
                "--handshake-timeout-ms", "30000",
                "--run-timeout-ms", str(int((args.idle_seconds + 8) * 1000)),
            ],
            stdout=(run / "gem5.log").open("w"), stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for _ in range(600):
                if endpoint.exists():
                    break
                if process.poll() is not None:
                    print("gem5-idle-cost: gem5 exited before creating its endpoint",
                          file=sys.stderr)
                    return 1
                time.sleep(0.1)

            # Hold the connection open for the idle window without submitting
            # any work, so everything measured is pure idle overhead.
            hold_ms = int(args.idle_seconds * 1000)
            handshake_process = subprocess.Popen(
                [handshake, "--endpoint", str(endpoint), "--timeout-ms", "30000",
                 "--hold-ms", str(hold_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            time.sleep(1.0)

            before_cpu = process_cpu_seconds(process.pid)
            before_wall = time.monotonic()
            time.sleep(args.idle_seconds)
            after_cpu = process_cpu_seconds(process.pid)
            idle_wall = time.monotonic() - before_wall

            handshake_process.wait(timeout=60)
            # Let gem5 reach its own service deadline so the config dumps
            # stats; killing it leaves stats.txt empty and the instruction
            # counters unobservable.
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                pass
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)

        stats = outdir / "stats.txt"
        record = {
            "schema": SCHEMA,
            "idle_wall_seconds": round(idle_wall, 3),
            "gem5_cpu_seconds_during_idle":
                None if before_cpu is None or after_cpu is None
                else round(after_cpu - before_cpu, 3),
            "sim_ticks": read_stat_counter(stats, "simTicks"),
            "sim_insts": read_stat_counter(stats, "simInsts"),
            "sim_ops": read_stat_counter(stats, "simOps"),
            "host_seconds": None,
        }
        host_seconds = None
        try:
            match = re.search(r"^hostSeconds\s+([0-9.]+)", stats.read_text(), re.MULTILINE)
            host_seconds = float(match.group(1)) if match else None
        except OSError:
            pass
        record["host_seconds"] = host_seconds
        busy = record["gem5_cpu_seconds_during_idle"]
        record["cpu_fraction_while_idle"] = (
            None if busy is None or idle_wall <= 0 else round(busy / idle_wall, 3)
        )
        print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
