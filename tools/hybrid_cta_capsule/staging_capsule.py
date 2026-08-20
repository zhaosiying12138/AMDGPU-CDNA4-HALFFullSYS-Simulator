#!/usr/bin/env python3
"""Staging-path capsule: four interleaved global dwordx4 tiles -> ALU consume.

The CK attention defect localizes to early-k-block staging: dims [0:128)
vanish while the final k-blocks are exact (sweep/sweeptail/dimsplit
probes).  Every register-level mechanism is verified correct, so the last
untested path is global dwordx4/buffer-load -> waitcnt -> nearby
consumers, the shape of CK's double-buffered tile fetch.  This capsule
loads four 16-byte-per-lane tiles (address-interleaved so earlier tiles
are not hot), consumes them with ALU adds after s_waitcnt vmcnt(0), and
stores per-ticket dwords.  Input tile dwords are pre-filled with
distinct known values; any stale tile fragment shows up as a wrong sum.
"""
import argparse
import ctypes
import faulthandler
import hashlib
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
faulthandler.enable()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(
        os.environ.get("STAGING_CAPSULE_OUTPUT",
                       str(ROOT / "artifacts/qwen35-staging-capsule/v1"))))
    args = parser.parse_args()

    image = (Path(__file__).parent / "stage_dwordx4.hsaco").read_bytes()
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

    n_slots = 256
    out_words = 16384  # out area [0,16384) dwords; inputs start at dword 4096
    total = 16384 + 4096  # inputs: 4 tiles * 1024 dwords
    buf = torch.zeros(total, dtype=torch.float32, device="cuda")
    # fill input tiles with distinct known u32 patterns
    host = torch.arange(total, dtype=torch.float32)  # placeholder
    in_base = 8  # inputs begin at dword 8, right after the pointer slot
    vals = torch.zeros(total, dtype=torch.int32)
    # the kernel s_loads out[0:1] as its input base pointer
    import numpy as np
    ptr = buf.data_ptr() + in_base * 4
    ptr_u64 = np.uint64(ptr)
    lo = int(np.uint32(ptr_u64 & np.uint64(0xFFFFFFFF)))
    hi = int(np.uint32(ptr_u64 >> np.uint64(32)))
    # keep sign bits legal for int32 storage
    vals[0] = lo - (1 << 32) if lo >= (1 << 31) else lo
    vals[1] = hi - (1 << 32) if hi >= (1 << 31) else hi
    for t in range(4):
        base = in_base + t * 1024
        for i in range(1024):
            vals[base + i] = 0x10000 * (t + 1) + i
    # upload as int32 and bit-reinterpret to float32; a numeric
    # .to(torch.float32) conversion would destroy the pointer patterns
    buf_i32 = vals.to("cuda").view(torch.float32)
    buf.copy_(buf_i32)

    module = ctypes.c_void_p()
    raw = ctypes.create_string_buffer(image)
    assert lib.hipModuleLoadData(ctypes.byref(module), ctypes.cast(raw, ctypes.c_void_p)) == 0
    function = ctypes.c_void_p()
    sym = None
    for name in (b"_Z14mfma_agpr_testPf",):
        if lib.hipModuleGetFunction(ctypes.byref(function), module, name) == 0:
            sym = name
            break
    assert sym, "kernel symbol not found"
    # inputs live at dword 4096 of the same buffer; pass their 64-bit
    # address as two u32 kernel arguments (the kernel builds its load
    # pointer from them).
    arg0 = ctypes.c_ulonglong(buf.data_ptr())
    params = (ctypes.c_void_p * 1)(
        ctypes.cast(ctypes.byref(arg0), ctypes.c_void_p),
    )
    stream = torch.cuda.current_stream().cuda_stream
    rc = lib.hipModuleLaunchKernel(function, 1, 1, 1, 256, 1, 1, 0,
                                   ctypes.c_void_p(stream), params, None)
    assert rc == 0, f"launch {rc}"
    torch.cuda.synchronize()
    got = buf[:out_words].detach().cpu()
    u = torch.frombuffer(got.numpy().tobytes(), dtype=torch.int32).tolist()
    # per ticket slots 0..3; expected: sum over t of (0x10000*(t+1) + lane*4 + k)
    # tickets map arbitrarily to lanes; a correct run yields exactly 64
    # distinct sums repeated 4x (256 lanes -> 64 lane_dw values, each seen by
    # 4 tickets); a broken early-tile path collapses many lanes to garbage.
    sums = {}
    for t in range(256):
        for k in range(4):
            v = u[t * 8 + k]
            sums[v] = sums.get(v, 0) + 1
    valid = {}
    for lane_dw in range(256):
        exp = sum(0x10000 * (ti + 1) + lane_dw for ti in range(4))
        valid[exp] = lane_dw
    bad_values = [v for v in sums if v not in valid]
    result = {
        "schema": "amdgpu-sim.qwen35-staging-capsule.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_sha256": hashlib.sha256(got.numpy().tobytes()).hexdigest(),
        "distinct_output_values": len(sums),
        "invalid_values": len(bad_values),
        "expected_sum_first_ticket": sum(0x10000 * (ti + 1) + 0 for ti in range(4)),
        "first_ticket_values": u[0:4],
    }
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "output.bin").write_bytes(got.numpy().tobytes())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
