#!/usr/bin/env python3
"""One-shot Triton softmax demo on the simulated GPU (single launch).

Designed to finish well under a minute on the simulator:

  - the input is generated on the CPU (torch.randn runs no GPU kernel)
    and copied to the device once;
  - the reference softmax is also computed on the CPU;
  - exactly ONE Triton kernel launch happens on the simulated GPU
    (fp32, rows x cols, default 4 x 128);
  - the result is copied back once and compared against the reference.

Exit 0 on PASS, 1 on FAIL.  If the ambient shell's stray variables
break the simulated device stack (hsa_init finds no devices), the
script re-execs itself through a clean environment automatically, so
plain `python softmax_demo.py` works from any interactive shell in
the AMDGPU-CDNA4-SIM environment.
"""

from __future__ import annotations

import argparse
import os
import sys


def _relaunch_clean() -> None:
    keep = (
        "HOME", "TERM", "PATH", "LD_LIBRARY_PATH", "LD_PRELOAD",
        "HSA_MODEL_LIB", "HSA_MODEL_TOPOLOGY", "HSA_ENABLE_DXG_DETECTION",
        "HSA_ENABLE_INTERRUPT", "ROCM_SIM_ROOT", "HSA_PATH", "HIP_PLATFORM",
        "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "SAGR_GENERIC_BRIDGE_ENDPOINT",
    )
    environ = {key: os.environ[key] for key in keep if os.environ.get(key)}
    environ.setdefault("HOME", "/home/zhaosiying")
    environ.setdefault("TERM", "dumb")
    environ.setdefault("HSA_ENABLE_DXG_DETECTION", "0")
    environ.setdefault("HSA_ENABLE_INTERRUPT", "0")
    environ.setdefault("HIP_PLATFORM", "amd")
    argv = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:]
    os.execve("/usr/bin/unshare", ["unshare", "-r", "-m", *argv], {**environ})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"),
                        default="float32")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        _relaunch_clean()  # execs; never returns
        return 1

    import triton
    import triton.language as tl

    @triton.jit
    def _softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_N: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask,
                    other=float("-inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        den = tl.sum(num, axis=0)
        tl.store(y_ptr + row * n_cols + cols, num / den, mask=mask)

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    torch.manual_seed(0)
    x_cpu = torch.randn(args.rows, args.cols).to(dtype)
    ref = torch.softmax(x_cpu.float(), dim=-1).to(dtype)

    x_gpu = x_cpu.to("cuda")
    y_gpu = torch.empty_like(x_gpu)
    _softmax_kernel[(args.rows,)](
        x_gpu, y_gpu, args.cols, BLOCK_N=args.block, num_warps=4,
    )
    torch.cuda.synchronize()
    got = y_gpu.cpu()

    err = (got.float() - ref.float()).abs().max()
    ok = bool(torch.isfinite(got.float()).all()) and float(err) < 5e-2
    print(
        f"[softmax] device={torch.cuda.get_device_name(0)} "
        f"{args.rows}x{args.cols} {args.dtype}: "
        f"max_err={float(err):.5f} -> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
