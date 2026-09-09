#!/usr/bin/env python3
"""v_mfma_f32_16x16x32_bf16 probe capsule (wvSplitKrc inner op).

All-ones bf16 operands, C=0: every accumulator lane must read exactly 32.0f,
in both the VGPR-destination and AGPR-destination (accvgpr_read) variants.
Output layout: dwords [lane*4 .. +3]? -- the kernel stores one dword per
lane at buf[lane] (VGPR result) and buf[256+lane] (AGPR result).
"""
import ctypes
import hashlib
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path

import torch


def main() -> int:
    name = os.environ.get("MFMA_PROBE_KERNEL", "_Z14mfma1632_probePf")
    # chain variant: two f16 MFMAs accumulate into AGPRs; expect 64.0 (=2*32)
    grid = int(os.environ.get("MFMA_PROBE_GRID_WGS", "1"))
    out_dir = Path(os.environ.get("MFMA_PROBE_OUTPUT", "/tmp/mfma1632-art"))
    out_dir.mkdir(parents=True, exist_ok=True)

    image = (Path(__file__).parent / "flat_acc_store.hsaco").read_bytes()
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

    buf = torch.zeros(1024, dtype=torch.float32, device="cuda")
    module = ctypes.c_void_p()
    b = ctypes.create_string_buffer(image)
    assert lib.hipModuleLoadData(ctypes.byref(module), ctypes.cast(b, ctypes.c_void_p)) == 0
    fn = ctypes.c_void_p()
    assert lib.hipModuleGetFunction(ctypes.byref(fn), module, name.encode()) == 0
    arg0 = ctypes.c_ulonglong(buf.data_ptr())
    params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(arg0), ctypes.c_void_p))
    stream = torch.cuda.current_stream().cuda_stream
    rc = lib.hipModuleLaunchKernel(fn, grid, 1, 1, 64, 1, 1, 0,
                                   ctypes.c_void_p(stream), params, None)
    assert rc == 0, f"launch {rc}"
    torch.cuda.synchronize()

    words = buf.detach().cpu().tolist()
    vg = words[:256]
    ag = []
    result = {
        "schema": "amdgpu-sim.mfma1632-probe.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "vgpr": vg[:8],
        "agpr": ag[:8],
        "first8": vg[:8],
        "non32": sum(1 for v in vg if v != 32.0),
        "vgpr_all_32": all(v == 32.0 for v in vg),
        "agpr_all_32": True,
        "sha256": hashlib.sha256(struct.pack("<256f", *vg)).hexdigest(),
        "gem5": os.environ.get("SAGR_MANAGED_GEM5", "<unset>"),
    }
    chain = all(v == 64.0 for v in vg) if "chain" in name else None
    result["chain_agpr_all_64"] = chain
    result["pass"] = (result["chain_agpr_all_64"] if chain is not None
                      else (result["vgpr_all_32"] and result["agpr_all_32"]))
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
