#!/usr/bin/env python3
"""Parse perf-ablation lanes into one machine-readable table.

Extracts, per config directory under artifacts/perf-ablation-2026-08:
  - wall time            started=/finished= in lane.log
  - weight load          "Load weight end. elapsed=Ns" (per rank, take min)
  - kv cache alloc       "KV Cache is allocated. elapsed"
  - engine init timings  "Init timings" line (load_weight/kv_cache_allocation/scheduler_e2e)
  - prefill / token      timestamps of "Prefill batch" vs engine-ready lines
  - retired dispatches   count of JSONL records in runroot/*/dispatch-trace.jsonl
  - copy blit dispatches  dispatches whose kernel name contains amd_blit
  - token gate verdict   grep "PASS"/token ids from the tail

Usage: parse_results.py [ablation-root]   (prints a markdown + TSV to stdout)
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "artifacts/perf-ablation-2026-08"


def parse_iso(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


def parse_lane(log_text: str) -> dict:
    out = {}
    started = re.findall(r"started=(\S+)", log_text)
    finished = re.findall(r"finished=(\S+) status=(\d+)", log_text)
    if started:
        out["started"] = started[-1]
    if finished:
        out["finished"], out["exit_status"] = finished[-1]
        try:
            delta = parse_iso(out["finished"]) - parse_iso(out["started"])
            out["wall_s"] = round(delta.total_seconds(), 1)
        except ValueError:
            pass
    loads = re.findall(r"Load weight end\. elapsed=([\d.]+)", log_text)
    if loads:
        out["load_weight_s"] = min(float(v) for v in loads)
    kv = re.findall(r"KV Cache is allocated\. elapsed=([\d.]+) s", log_text)
    if kv:
        out["kv_cache_s"] = min(float(v) for v in kv)
    init = re.findall(r"Init timings \(s\): load_weight=([\d.]+), kv_cache_allocation=([\d.]+), scheduler_e2e=([\d.]+)", log_text)
    if init:
        lw, kvc, sched = init[-1]
        out["init_load_weight_s"] = float(lw)
        out["init_kv_s"] = float(kvc)
        out["init_scheduler_s"] = float(sched)
    fastcopy = re.findall(r"fastcopy=HSA_ENABLE_DTIF_FAST_COPY=(\d+) SAGR_HSAKMT_MODEL_FAST_COPY=(\d+)", log_text)
    if fastcopy:
        out["fastcopy"] = "/".join(fastcopy[-1])
    out["timeout"] = "TIMEOUT" in log_text
    ids = re.findall(r'"output_ids": \[([0-9, ]+)\]', log_text)
    if ids:
        out["output_ids"] = ids[-1].strip()
    return out


def parse_runroot(runroot: Path) -> dict:
    traces = sorted(runroot.glob("self-amdgpu-opencl-run.*/dispatch-trace.jsonl"))
    total = 0
    blit = 0
    for trace in traces:
        for line in trace.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            total += 1
            if "blit" in line.lower():
                blit += 1
    return {"dispatches": total, "blit_dispatches": blit}


def main() -> int:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    rows = []
    for cfgdir in sorted(base.iterdir()):
        if not cfgdir.is_dir():
            continue
        log = cfgdir / "lane.log"
        row = {"config": cfgdir.name}
        if log.exists():
            row.update(parse_lane(log.read_text(errors="replace")))
        runroot = cfgdir / "runroot"
        if runroot.exists():
            row.update(parse_runroot(runroot))
        rows.append(row)

    fields = ["config", "wall_s", "load_weight_s", "kv_cache_s",
              "init_scheduler_s", "dispatches", "blit_dispatches",
              "fastcopy", "output_ids", "timeout", "exit_status"]
    print("| " + " | ".join(fields) + " |")
    print("|" + "---|" * len(fields))
    for row in rows:
        print("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    print()
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
