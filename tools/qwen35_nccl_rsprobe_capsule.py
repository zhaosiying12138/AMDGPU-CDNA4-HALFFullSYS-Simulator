#!/usr/bin/env python3
"""Reduce-scatter forensics probe on the simulated 2-GPU topology.

The committed RS-ONLY oracle calls ``reduce_scatter_tensor(rs, rs)`` with
``rs = 2*1024`` floats, which PyTorch's c10d wrapper rejects client-side
("input tensor must be the same size as output size times world size")
before NCCL is ever entered, so that mode can only measure teardown
behaviour, never the collective.  This probe issues a VALID in-place
reduce_scatter (input 2048 floats, output the rank-local 1024-float
slice) after the proven warm allreduces, verifies the numerics against
the fixed seeds, and prints per-op markers so a hang names its op and
the live processes can be inspected via /proc/PID/mem.

Environment knobs:
  RS_PROBE_ONLY_RS=1   stop after the reduce_scatter (default: continue
                       into all_gather/barrier/mixed loop)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get(
    "QWEN35_NCCL_RSPROBE_OUTPUT",
    str(ROOT / "artifacts/qwen35-nccl-rsprobe/v1")))
OUT.mkdir(parents=True, exist_ok=True)

IN_FLOATS = 2048   # input per rank: world_size x output
OUT_FLOATS = 1024  # output per rank


def log(msg: str) -> None:
    print(f"[rs-probe rank={os.environ.get('RANK', '0')}] {msg}", flush=True)


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank: int, world: int, addr: str, port: int) -> int:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = str(port)
    log(f"start pid={os.getpid()}")

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    log(f"device set: {torch.cuda.get_device_name(rank)}")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    log("process group initialized")

    torch.manual_seed(1234 + rank)
    buf = torch.randn(256, device="cuda", dtype=torch.float32)
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    log("warm allreduce 1 ok")
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    log("warm allreduce 2 ok")

    # Valid reduce_scatter_tensor: input is world_size x output.  Generate
    # on the CPU (torch's CUDA Philox stream cannot be replayed on the CPU
    # for the expectation below), then stage to the device.
    def seeded(seed: int, n: int):
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(n, generator=gen)

    rs_in_cpu = seeded(4321 + rank, IN_FLOATS)
    rs_in = rs_in_cpu.cuda()
    rs_out = torch.empty(OUT_FLOATS, device="cuda", dtype=torch.float32)
    log("RS begin")
    dist.reduce_scatter_tensor(rs_out, rs_in)
    torch.cuda.synchronize()
    log("RS ok")

    # Numeric oracle: rank r receives sum of slice r of both inputs.
    expect = seeded(4321, IN_FLOATS)[rank * OUT_FLOATS:
                                     (rank + 1) * OUT_FLOATS] + \
        seeded(4322, IN_FLOATS)[rank * OUT_FLOATS:
                                (rank + 1) * OUT_FLOATS]
    err = float((rs_out.cpu() - expect).abs().max())
    log(f"RS err={err:.6e}")
    ok = err < 1e-5

    # SGLang-scale collectives: the engine freezes at multi-hundred-KB
    # collectives (vocab-wide logits allgather, weight shards), which push
    # NCCL into multi-chunk ring territory the tiny probe ops never enter.
    BIG = int(os.environ.get("RS_PROBE_BIG_FLOATS", str(2 * 1024 * 1024)))
    if BIG > 0:
        big_src = torch.randn(BIG, device="cuda", dtype=torch.float32)
        big_ag_dst = torch.empty(2 * BIG, device="cuda",
                                 dtype=torch.float32)
        big_rs_out = torch.empty(BIG, device="cuda", dtype=torch.float32)
        for i in range(3):
            log(f"BIG {i} all_gather({2 * BIG * 4 // 1024}KB) begin")
            dist.all_gather_into_tensor(big_ag_dst, big_src)
            torch.cuda.synchronize()
            log(f"BIG {i} all_gather ok")
            log(f"BIG {i} reduce_scatter begin")
            dist.reduce_scatter_tensor(big_rs_out, big_ag_dst)
            torch.cuda.synchronize()
            log(f"BIG {i} reduce_scatter ok")
        ok = ok and bool(torch.isfinite(big_rs_out.cpu()).all())

    if os.environ.get("RS_PROBE_ONLY_RS") == "1":
        dist.barrier()
        dist.destroy_process_group()
        log(f"RS-PROBE PASS ok={ok}")
        return 0 if ok else 1

    ag_src = torch.randn(1024, device="cuda", dtype=torch.float32)
    ag_dst = torch.empty(2 * 1024, device="cuda", dtype=torch.float32)
    for i in range(8):
        kind = ("allreduce", "barrier", "all_gather",
                "reduce_scatter")[i % 4]
        log(f"MIX {i} {kind} begin")
        if kind == "allreduce":
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        elif kind == "barrier":
            dist.barrier()
        elif kind == "all_gather":
            dist.all_gather_into_tensor(ag_dst, ag_src)
        else:
            torch.manual_seed(100 + i)
            dist.reduce_scatter_tensor(
                rs_out, torch.randn(IN_FLOATS, device="cuda"))
        torch.cuda.synchronize()
        log(f"MIX {i} {kind} ok")
    dist.barrier()
    dist.destroy_process_group()
    log(f"RS-PROBE PASS ok={ok}")
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
    child = subprocess.Popen([sys.executable, __file__], env=env)

    code = worker(0, 2, "127.0.0.1", port)
    try:
        child.wait(timeout=300)
    except subprocess.TimeoutExpired:
        child.kill()
        log("rank 1 timed out")
        code = 1
    report = {"schema": "amdgpu-sim.qwen35-nccl-rsprobe.v1",
              "rank0_exit": code,
              "child_returncode": child.returncode,
              "passed": code == 0 and child.returncode == 0}
    (OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n",
                                     encoding="ascii")
    log(f"report: {report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
