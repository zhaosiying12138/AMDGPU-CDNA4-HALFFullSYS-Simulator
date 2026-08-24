#!/usr/bin/env python3
"""One-shot Triton softmax correctness demo on the simulated GPU.

No benchmarking: a single kernel launch per configuration, checked
against ``torch.softmax`` on the same tensor.  Sized small on purpose
-- the simulator executes every instruction, so a 64-row kernel
completes in tens of seconds instead of the minutes a full-benchmark
sweep would take.

Exit 0 on PASS (max error within the bf16 band), exit 1 on FAIL.
"""

from __future__ import annotations

import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr,
    y_ptr,
    n_cols,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=float("-inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den
    tl.store(y_ptr + row * n_cols + cols, y, mask=mask)


def run_one(rows: int, cols: int, block_n: int, dtype: torch.dtype) -> bool:
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    y = torch.empty_like(x)

    grid = (rows,)
    t0 = time.time()
    _softmax_kernel[grid](
        x, y, cols, BLOCK_N=block_n, num_warps=4,
    )
    torch.cuda.synchronize()
    wall = time.time() - t0

    ref = torch.softmax(x.float(), dim=-1).to(dtype)
    err = (y.float() - ref).abs().max()
    ok = bool(torch.isfinite(y.float()).all()) and float(err) < 5e-2
    print(
        f"[softmax] {rows}x{cols} {str(dtype).split('.')[-1]} "
        f"BLOCK_N={block_n}: max_err={float(err):.5f} "
        f"wall={wall:.1f}s -> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--block", type=int, default=256)
    args = parser.parse_args()

    print(f"[softmax] device: {torch.cuda.get_device_name(0)}", flush=True)
    ok = run_one(args.rows, args.cols, args.block, torch.float32)
    ok = run_one(max(1, args.rows // 2), args.cols, args.block, torch.bfloat16) and ok
    print(f"[softmax] overall: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
