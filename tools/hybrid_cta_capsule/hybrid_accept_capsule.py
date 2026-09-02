#!/usr/bin/env python3
"""Dual-mode acceptance capsule for the hybrid CTA executor.

Loads the three acceptance kernels (plain data-parallel, barrier+LDS
cross-wave, memsync-decline) from tools/hybrid_cta_capsule/hybrid_kernels.hsaco,
launches each on the lane's simulator, and writes every output buffer's
sha256 plus a full histogram to result.json.  Run once per mode (accurate
fastwrap, hybrid) via the lane runner; compare the two result.json files
for byte-identical outputs.  Kernel selection via env HYBRID_KERNEL
(plain_dp | barrier_lds | atomic_decline).
"""
import argparse
import ctypes
import faulthandler
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
faulthandler.enable()

KERNELS = {
    "plain_dp": ("_Z8plain_dpPf", 64),
    "barrier_lds": ("_Z11barrier_ldsPf", 128),
    "atomic_decline": ("_Z14atomic_declinePf", 64),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(
        os.environ.get("HYBRID_CAPSULE_OUTPUT",
                       str(ROOT / "artifacts/hybrid-cta-capsule/v1"))))
    args = parser.parse_args()
    name = os.environ.get("HYBRID_KERNEL", "plain_dp")
    symbol, block = KERNELS[name]
    grid = int(os.environ.get("HYBRID_GRID_WGS", "4"))
    if grid < 2:
        raise ValueError("HYBRID_GRID_WGS must be at least 2")
    total_slots = grid * 256  # kernels index wg*256 + lane

    image = (Path(__file__).parent / "hybrid_kernels.hsaco").read_bytes()
    hip_path = None
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        cand = os.path.join(d, "libamdhip64.so.7")
        if os.path.exists(cand):
            hip_path = cand
            break
    lib = ctypes.CDLL(hip_path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    lib.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.hipModuleLoadData.restype = ctypes.c_int
    lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    lib.hipModuleGetFunction.restype = ctypes.c_int
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    lib.hipModuleLaunchKernel.restype = ctypes.c_int

    out = torch.zeros(total_slots, dtype=torch.float32, device="cuda")
    module = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(image)
    assert lib.hipModuleLoadData(ctypes.byref(module), ctypes.cast(buf, ctypes.c_void_p)) == 0
    function = ctypes.c_void_p()
    assert lib.hipModuleGetFunction(ctypes.byref(function), module, symbol.encode()) == 0
    arg0 = ctypes.c_ulonglong(out.data_ptr())
    params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(arg0), ctypes.c_void_p))
    stream = torch.cuda.current_stream().cuda_stream
    rc = lib.hipModuleLaunchKernel(function, grid, 1, 1, block, 1, 1, 0,
                                   ctypes.c_void_p(stream), params, None)
    assert rc == 0, f"launch {rc}"
    torch.cuda.synchronize()
    raw = out.detach().cpu().numpy().tobytes()
    words = out.detach().cpu().tolist()
    import struct
    uvals = [struct.unpack("<I", struct.pack("<f", v))[0] for v in words]
    histogram = {}
    for v in uvals:
        histogram[v] = histogram.get(v, 0) + 1
    oracle_correct = None
    if name == "plain_dp":
        oracle_correct = all(
            uvals[wg * 256 + lane] == wg * 4096 + lane
            for wg in range(grid)
            for lane in range(block)
        ) and all(
            uvals[wg * 256 + lane] == 0
            for wg in range(grid)
            for lane in range(block, 256)
        )
    result = {
        "schema": "amdgpu-sim.hybrid-cta-capsule.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kernel": name,
        "symbol": symbol,
        "grid": grid,
        "block": block,
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "distinct_values": len(histogram),
        "top_values": sorted(histogram.items(), key=lambda x: -x[1])[:6],
        "oracle_correct": oracle_correct,
        "gem5": os.environ.get("SAGR_MANAGED_GEM5", "<unset>"),
    }
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "output.bin").write_bytes(raw)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
