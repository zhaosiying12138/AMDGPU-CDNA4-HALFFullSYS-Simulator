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

    # The v1 run hung at a LATE barrier while an early barrier passed
    # instantly: intermittent.  Loop collectives with markers and an
    # interleaved host sync so the first op that never returns -- and
    # whether host-side sync between collectives correlates -- lands in
    # the log.
    for i in range(10):
        b = torch.randn(SIZE, device="cuda", dtype=torch.float32)
        t1 = time.time()
        dist.all_reduce(b, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        log(f"iter {i}: all_reduce ok {time.time() - t1:.1f}s")
        host = b.cpu()  # interleaved D2H sync like the real engine does
        t2 = time.time()
        dist.barrier()
        log(f"iter {i}: barrier ok {time.time() - t2:.1f}s")
    log("loop completed")

    # The engine's remaining collective vocabulary: all_gather (logits
    # collection, vocab-wide), broadcast (weight distribution), and
    # reduce_scatter.  The lanes froze at 1618 dispatches with allreduce
    # healthy, so the blocker is plausibly one of these.
    # Escalating all_gather discriminators: 1-element list API, then the
    # into_tensor API the engines actually use, then a bigger one.
    tiny_parts = [torch.zeros(1, device="cuda", dtype=torch.float32)
                  for _ in range(2)]
    t4 = time.time()
    dist.all_gather(tiny_parts, tiny_parts[rank])
    torch.cuda.synchronize()
    log(f"all_gather tiny(list) ok {time.time() - t4:.1f}s")

    big_parts = [torch.randn(1024, device="cuda", dtype=torch.float32)
                 for _ in range(2)]
    t4b = time.time()
    dist.all_gather(big_parts, big_parts[rank])
    torch.cuda.synchronize()
    log(f"all_gather 1024(list) ok {time.time() - t4b:.1f}s")

    src_t = torch.randn(1024, device="cuda", dtype=torch.float32)
    dst_t = torch.empty(2 * 1024, device="cuda", dtype=torch.float32)
    t4c = time.time()
    dist.all_gather_into_tensor(dst_t, src_t)
    torch.cuda.synchronize()
    log(f"all_gather_into_tensor ok {time.time() - t4c:.1f}s "
        f"finite={bool(torch.isfinite(dst_t).all())}")

    bcast = torch.zeros(2048, device="cuda", dtype=torch.float32)
    if rank == 0:
        bcast.normal_()
    t5 = time.time()
    dist.broadcast(bcast, src=0)
    torch.cuda.synchronize()
    log(f"broadcast ok {time.time() - t5:.1f}s "
        f"finite={bool(torch.isfinite(bcast).all())}")

    rs = torch.randn(2 * 1024, device="cuda", dtype=torch.float32)
    t6 = time.time()
    dist.reduce_scatter_tensor(rs, rs)
    torch.cuda.synchronize()
    log(f"reduce_scatter ok {time.time() - t6:.1f}s "
        f"finite={bool(torch.isfinite(rs).all())}")
    log("extended collectives completed")

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
