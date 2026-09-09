#!/usr/bin/env python3
"""Full-LDS serialization capsule (timing-path differential).

Loads full_lds_a (152 KB __shared__ per WG: two WGs do NOT co-fit on one
CU's 160 KB LDS -> forced serial occupancy, the mechanism exercised by the
vLLM 0.8B chunked GDN kernel that mis-executes) and full_lds_b (64 KB:
two WGs co-fit, the control) from full_lds.hsaco, launches each on N
workgroups, and checks the exact wg*77777+lane round-trip oracle.
Run under the lane (exact env + NVML isolation), accurate or fast mode.
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

faulthandler.enable()

SYMBOLS = {
    "full_lds_a": ("_Z10full_lds_aPf", (256, 1)),
    "full_lds_b": ("_Z10full_lds_bPf", (256, 1)),
    "full_lds_c": ("_Z10full_lds_cPf", (64, 4)),
    "full_lds_d": ("_Z10full_lds_dPf", (64, 4)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(
        os.environ.get("FULL_LDS_CAPSULE_OUTPUT", "artifacts/full-lds-capsule/v1")))
    args = parser.parse_args()
    name = os.environ.get("FULL_LDS_KERNEL", "full_lds_a")
    symbol, (bx, by) = SYMBOLS[name]
    block = bx * by
    grid = int(os.environ.get("FULL_LDS_GRID_WGS", "4"))
    total = grid * block

    image = (Path(__file__).parent / "full_lds.hsaco").read_bytes()
    hip_path = None
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        cand = os.path.join(d, "libamdhip64.so.7")
        if os.path.exists(cand):
            hip_path = cand
            break
    lib = ctypes.CDLL(hip_path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    lib.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]

    out = torch.zeros(total, dtype=torch.float32, device="cuda")
    module = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(image)
    assert lib.hipModuleLoadData(ctypes.byref(module), ctypes.cast(buf, ctypes.c_void_p)) == 0
    fn = ctypes.c_void_p()
    assert lib.hipModuleGetFunction(ctypes.byref(fn), module, symbol.encode()) == 0
    arg0 = ctypes.c_ulonglong(out.data_ptr())
    params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(arg0), ctypes.c_void_p))
    stream = torch.cuda.current_stream().cuda_stream
    rc = lib.hipModuleLaunchKernel(fn, grid, 1, 1, bx, by, 1, 0,
                                   ctypes.c_void_p(stream), params, None)
    assert rc == 0, f"launch {rc}"
    torch.cuda.synchronize()

    raw = out.detach().cpu().numpy().tobytes()
    words = out.detach().cpu().tolist()
    uvals = [struct.unpack("<I", struct.pack("<f", v))[0] for v in words]
    # float can't carry big ints exactly; oracle in float: wg*77777+lane is
    # exact up to 2^24, our values stay < 2^17 * 4 -- fine.
    oracle_correct = all(
        words[wg * block + lane] == float(wg * 77777 + lane)
        for wg in range(grid) for lane in range(block)
    )
    histogram = {}
    for v in uvals:
        histogram[v] = histogram.get(v, 0) + 1
    result = {
        "schema": "amdgpu-sim.full-lds-capsule.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kernel": name,
        "grid": grid,
        "block": block,
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "distinct_values": len(histogram),
        "oracle_correct": oracle_correct,
        "first_mismatches": [
            {"wg": i // 256, "lane": i % 256, "got": words[i], "want": (i // 256) * 77777 + (i % 256)}
            for i in range(total) if words[i] != float((i // block) * 77777 + (i % block))
        ][:8],
        "gem5": os.environ.get("SAGR_MANAGED_GEM5", "<unset>"),
    }
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if oracle_correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
