#!/usr/bin/env python3
"""Run a normal Triton program after activating the repository conda product."""

from pathlib import Path
import runpy


_BOOTSTRAP = Path(__file__).resolve().parents[1] / "triton/_gemsim_bootstrap.py"
runpy.run_path(str(_BOOTSTRAP))["bootstrap"](__file__, "quickstart-triton-vecadd")

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(left, right, output, size: tl.constexpr):
    offsets = tl.arange(0, 256)
    mask = offsets < size
    tl.store(output + offsets, tl.load(left + offsets, mask=mask) + tl.load(right + offsets, mask=mask), mask=mask)


left = torch.arange(127, dtype=torch.float32)
right = torch.arange(127, dtype=torch.float32).flip(0).contiguous()
output = torch.empty_like(left)
add_kernel[(1,)](left, right, output, left.numel())
torch.testing.assert_close(output, left + right, rtol=0, atol=0)
target = triton.runtime.driver.active.get_current_target()
print(f"triton vecadd passed on {target.backend}:{target.arch}")
