#!/usr/bin/env python3
"""Launch a queued kernel, free its output, and close without synchronizing."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "tools/hybrid_cta_capsule/hybrid_kernels.hsaco"


def main() -> int:
    grid = int(os.environ.get("KMT_RACE_GRID_WGS", "2048"))
    if grid < 2:
        raise ValueError("KMT_RACE_GRID_WGS must be at least 2")
    hip_path = next(
        (
            os.path.join(directory, "libamdhip64.so.7")
            for directory in os.environ.get("LD_LIBRARY_PATH", "").split(":")
            if os.path.exists(os.path.join(directory, "libamdhip64.so.7"))
        ),
        None,
    )
    if hip_path is None:
        raise RuntimeError("libamdhip64.so.7 is not visible")
    lib = ctypes.CDLL(hip_path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    lib.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.hipModuleLoadData.restype = ctypes.c_int
    lib.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p
    ]
    lib.hipModuleGetFunction.restype = ctypes.c_int
    lib.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.hipModuleLaunchKernel.restype = ctypes.c_int

    # Keep a large queue of functional WGs active while the owner tears down.
    output = torch.zeros(grid * 256, dtype=torch.float32, device="cuda")
    module = ctypes.c_void_p()
    image = ctypes.create_string_buffer(IMAGE.read_bytes())
    if lib.hipModuleLoadData(ctypes.byref(module), ctypes.cast(image, ctypes.c_void_p)) != 0:
        raise RuntimeError("hipModuleLoadData failed")
    function = ctypes.c_void_p()
    if lib.hipModuleGetFunction(
        ctypes.byref(function), module, b"_Z8plain_dpPf"
    ) != 0:
        raise RuntimeError("hipModuleGetFunction failed")
    arg0 = ctypes.c_ulonglong(output.data_ptr())
    params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(arg0), ctypes.c_void_p))
    stream = torch.cuda.current_stream().cuda_stream
    if lib.hipModuleLaunchKernel(
        function, grid, 1, 1, 64, 1, 1, 0,
        ctypes.c_void_p(stream), params, None
    ) != 0:
        raise RuntimeError("hipModuleLaunchKernel failed")
    print(f"KMT_RACE_LAUNCHED grid_wgs={grid} ptr=0x{output.data_ptr():x}", flush=True)
    del output
    torch.cuda.empty_cache()
    time.sleep(float(os.environ.get("KMT_RACE_CLOSE_DELAY_SEC", "0.2")))
    print("KMT_RACE_CLOSE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
