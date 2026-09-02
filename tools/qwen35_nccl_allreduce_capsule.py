#!/usr/bin/env python3
"""Minimal two-process NCCL allreduce on the simulated 2-GPU topology.

The TP2 lanes deadlock symmetrically inside collective kernels on BOTH
transports (SHM/direct at 1656 dispatches per rank, NET/Socket at 1618),
so the defect is transport-independent and lives in the collective
kernel's own logic under gem5.  This capsule reproduces the collective in
isolation: two processes, one per simulated GPU, a single tiny allreduce,
result verified against the arithmetic sum.  Whichever rank prints its
marker last before a freeze names the spin point; with a fix in place
both ranks print PASS and exit 0.

Run through the lane script with --engine sglang --tp 2 --capsule (the
tp=2 path generates the 2-GPU topology) and NCCL_SHM_DISABLE=1 in the
environment to select the NET/Socket transport that outlived the SHM
wall.  The rank-1 child inherits the managed run root, so both gem5
instances land under the lane's own root and the watchdog's process
accounting stays exact.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get(
    "QWEN35_NCCL_ALLREDUCE_OUTPUT",
    str(ROOT / "artifacts/qwen35-nccl-allreduce-capsule/v1")))
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 256  # floats: one NCCL chunk-scale buffer, far below any threshold


def log(msg: str) -> None:
    print(f"[nccl-capsule rank={os.environ.get('RANK', '0')}] {msg}",
          flush=True)


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank: int, world: int, addr: str, port: int) -> int:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = str(port)
    log("start")

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    log(f"device set: {torch.cuda.get_device_name(rank)}")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    log("process group initialized")

    torch.manual_seed(1234 + rank)
    buf = torch.randn(SIZE, device="cuda", dtype=torch.float32)
    log("buffer created; calling all_reduce")

    t0 = time.time()
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    log(f"all_reduce returned after {time.time() - t0:.1f}s")

    # warm-up collective (also the v1-v3 baseline case)
    log("RATE warm allreduce begin")
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    log("RATE warm allreduce ok")

    # The engine's remaining collective vocabulary: all_gather (logits
    # collection, vocab-wide), broadcast (weight distribution), and
    # reduce_scatter.  The lanes froze at 1618 dispatches with allreduce
    # healthy, so the blocker is plausibly one of these.
    # Mixed-collective rate loop: every op logs its index and type BEFORE
    # running; when the intermittent deadlock hits, the last marker in the
    # log names the hanging op, and the completed-op count across runs
    # measures the per-op hang probability under this run's NCCL knobs
    # (the runner sets NCCL_PROTO/NCCL_ALGO for sensitivity comparison).
    LOOP_OPS = int(os.environ.get("CAPSULE_LOOP_OPS", "60"))
    # Minimal-determinism mode: a single reduce_scatter with nothing else
    # in the sequence (the op that hung 3/3 across protocol variants).
    if os.environ.get("CAPSULE_RS_ONLY") == "1":
        rs = torch.randn(2 * 1024, device="cuda", dtype=torch.float32)
        log("RS-ONLY begin")
        dist.reduce_scatter_tensor(rs, rs)
        torch.cuda.synchronize()
        log(f"RS-ONLY ok finite={bool(torch.isfinite(rs).all())}")
        dist.barrier()
        dist.destroy_process_group()
        log("RS-ONLY PASS")
        return 0
    ag_src = torch.randn(1024, device="cuda", dtype=torch.float32)
    ag_dst = torch.empty(2 * 1024, device="cuda", dtype=torch.float32)
    rs_buf = torch.randn(2 * 1024, device="cuda", dtype=torch.float32)
    ar_buf = torch.randn(256, device="cuda", dtype=torch.float32)
    one = torch.ones(1, device="cuda", dtype=torch.float32)
    for i in range(LOOP_OPS):
        kind = ("allreduce", "barrier", "all_gather", "broadcast",
                "reduce_scatter")[i % 5]
        log(f"RATE {i} {kind} begin")
        if kind == "allreduce":
            dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM)
        elif kind == "barrier":
            dist.barrier()
        elif kind == "all_gather":
            dist.all_gather_into_tensor(ag_dst, ag_src)
        elif kind == "broadcast":
            dist.broadcast(one, src=0)
        else:
            dist.reduce_scatter_tensor(rs_buf, rs_buf)
        torch.cuda.synchronize()
        log(f"RATE {i} {kind} ok")
    log(f"rate loop completed: {LOOP_OPS} ops")

    got = buf.cpu()  # checked against the FIRST allreduce result
    log(f"result mean={float(got.mean()):.6f} finite={bool(torch.isfinite(got).all())}")

    ok = bool(torch.isfinite(got).all())
    if rank == 0:
        # Deterministic check: both ranks regenerate the peer tensor from
        # the fixed seeds and compare against the reduced buffer.
        expect = (torch.manual_seed(1234).randn(SIZE)
                  + torch.manual_seed(1235).randn(SIZE))
        err = float((got - expect).abs().max())
        ok = ok and err < 1e-5
        log(f"expected max_abs_err={err:.6e}")
    dist.barrier()
    dist.destroy_process_group()
    log(f"PASS ok={ok}")
    return 0 if ok else 1


def main() -> int:
    rank = int(os.environ.get("CAPSULE_CHILD_RANK", "-1"))
    if rank >= 0:
        return worker(rank, 2, os.environ["CAPSULE_MASTER_ADDR"],
                      int(os.environ["CAPSULE_MASTER_PORT"]))

    port = find_free_port()
    log(f"spawning rank 1 (master 127.0.0.1:{port})")
    env = dict(os.environ)
    env["CAPSULE_CHILD_RANK"] = "1"
    env["CAPSULE_MASTER_ADDR"] = "127.0.0.1"
    env["CAPSULE_MASTER_PORT"] = str(port)
    import subprocess
    child = subprocess.Popen([sys.executable, __file__], env=env)

    code = worker(0, 2, "127.0.0.1", port)
    try:
        child.wait(timeout=120)
    except subprocess.TimeoutExpired:
        child.kill()
        log("rank 1 timed out")
        code = 1
    report = {"schema": "amdgpu-sim.qwen35-nccl-allreduce.v1",
              "rank0_exit": code,
              "child_returncode": child.returncode,
              "passed": code == 0 and child.returncode == 0}
    (OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n",
                                     encoding="ascii")
    log(f"report: {report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
