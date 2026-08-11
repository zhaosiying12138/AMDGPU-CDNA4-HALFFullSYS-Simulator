#!/usr/bin/env python3

import json

import torch
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def main() -> int:
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "gemsim_amd" or target.arch != "gfx950":
        raise RuntimeError(f"unexpected Triton execution target: {target}")
    if DEVICE.type != "cpu":
        raise RuntimeError(f"gemsim_amd must expose a CPU staging device, got {DEVICE}")
    n_elements = 98_432
    launch_results = []
    for launch_index, seed in enumerate((0, 1)):
        torch.manual_seed(seed)
        x = torch.rand(n_elements, device=DEVICE, dtype=torch.float32)
        y = torch.rand(n_elements, device=DEVICE, dtype=torch.float32)
        output = add(x, y)
        reference = x + y
        launch_results.append(
            {
                "launch_index": launch_index,
                "seed": seed,
                "output_correct": bool(torch.equal(output, reference)),
                "mismatch_count": int(torch.count_nonzero(output != reference).item()),
                "max_abs_error": float(torch.max(torch.abs(output - reference)).item()),
            }
        )
    correct = all(result["output_correct"] for result in launch_results)
    mismatch_count = sum(result["mismatch_count"] for result in launch_results)
    maximum_error = max(result["max_abs_error"] for result in launch_results)
    print(
        json.dumps(
            {
                "schema": "amdgpu-sim.triton-vecadd.v1",
                "backend": "gemsim_amd",
                "arch": target.arch,
                "kernel": "add_kernel",
                "n_elements": n_elements,
                "block_size": 1024,
                "program_count": triton.cdiv(n_elements, 1024),
                "launch_count": len(launch_results),
                "reuse": True,
                "launch_results": launch_results,
                "output_correct": correct,
                "mismatch_count": mismatch_count,
                "max_abs_error": maximum_error,
                "fallback_count": 0,
            },
            sort_keys=True,
        )
    )
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
