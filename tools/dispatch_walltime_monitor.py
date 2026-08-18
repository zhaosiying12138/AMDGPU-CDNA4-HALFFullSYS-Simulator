#!/usr/bin/env python3
"""Attribute a model run's *wall-clock* time to individual kernel dispatches.

gem5's dispatch trace records simulated ticks only, and simulated ticks are not
wall-clock: an idle service loop invents roughly 0.7e12 ticks per real second,
so a gap that looks like 82% of "simulated time" can be half a second of real
time. Reasoning about where a five-hour run went from simTicks alone is wrong.

This monitor polls the live dispatch traces and stamps each newly retired
dispatch with the real time it appeared. That gives, per dispatch: wall seconds
since the previous retirement, and the running total -- which is the only thing
that answers "where did the wall clock go".

It also flags the pathology that cost a 5.5 hour window: no dispatch retiring
while gem5 burns CPU. Run it alongside a lane; it writes JSONL and prints a
periodic summary.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


SCHEMA = "amdgpu-sim.dispatch-walltime.v1"


def live_run_dirs() -> list[Path]:
    """Run directories owned by a currently running gem5 process."""
    found: set[Path] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
            if "gem5" not in comm:
                continue
            cmdline = (entry / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        for field in cmdline.split("\0"):
            if field.startswith("/tmp/self-amdgpu-opencl-run."):
                found.add(Path(field.split("/m5out")[0]))
    return sorted(found)


def total_records() -> tuple[int, dict[str, int]]:
    per: dict[str, int] = {}
    total = 0
    for run in live_run_dirs():
        trace = run / "dispatch-trace.jsonl"
        try:
            count = sum(1 for _ in trace.open("rb"))
        except OSError:
            continue
        per[run.name] = count
        total += count
    return total, per


def gem5_cpu_percent() -> float:
    """Rough aggregate CPU of live gem5 processes, sampled over 0.4s."""
    def snapshot() -> dict[int, int]:
        out: dict[int, int] = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if "gem5" not in (entry / "comm").read_text():
                    continue
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                out[int(entry.name)] = int(fields[11]) + int(fields[12])
            except (OSError, IndexError, ValueError):
                continue
        return out

    ticks = os.sysconf("SC_CLK_TCK")
    first = snapshot()
    time.sleep(0.4)
    second = snapshot()
    delta = sum(second.get(pid, 0) - value for pid, value in first.items())
    return 100.0 * (delta / ticks) / 0.4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/lanes/dispatch-walltime.jsonl")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--summary-every", type=int, default=30)
    parser.add_argument("--stall-seconds", type=float, default=300.0)
    arguments = parser.parse_args(argv)

    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("a", buffering=1)

    start = time.monotonic()
    last_count, _ = total_records()
    last_change = start
    samples = 0
    gaps: list[float] = []

    print(f"{SCHEMA}: watching live gem5 dispatch traces (interval={arguments.interval}s)")
    try:
        while True:
            time.sleep(arguments.interval)
            samples += 1
            now = time.monotonic()
            count, per = total_records()
            if count != last_count:
                gap = now - last_change
                for _ in range(count - last_count):
                    gaps.append(gap / max(count - last_count, 1))
                stream.write(json.dumps({
                    "schema": SCHEMA,
                    "wall_seconds": round(now - start, 2),
                    "dispatches": count,
                    "new": count - last_count,
                    "seconds_since_previous": round(gap, 2),
                    "per_session": per,
                }) + "\n")
                last_count = count
                last_change = now
            elif now - last_change > arguments.stall_seconds:
                cpu = gem5_cpu_percent()
                stream.write(json.dumps({
                    "schema": SCHEMA,
                    "event": "stall",
                    "wall_seconds": round(now - start, 2),
                    "dispatches": count,
                    "stalled_seconds": round(now - last_change, 1),
                    "gem5_cpu_percent": round(cpu, 1),
                }) + "\n")
                print(f"  STALL: no dispatch for {now - last_change:.0f}s "
                      f"at {count} dispatches, gem5 CPU {cpu:.0f}%")
                last_change = now

            if samples % arguments.summary_every == 0:
                elapsed = now - start
                rate = count / elapsed * 60 if elapsed else 0
                mean = sum(gaps) / len(gaps) if gaps else 0
                print(f"  {elapsed/60:6.1f} min  dispatches={count:5d}  "
                      f"{rate:5.1f}/min  mean_gap={mean:5.2f}s")
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
