#!/usr/bin/env python3
"""splitk primitive bisect capsule (vLLM skinny NaN forensics)."""
import ctypes
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


def main() -> int:
    name = "_Z17splitk_prim_probePf"
    out_dir = Path(os.environ.get("PRIM_PROBE_OUTPUT", "/tmp/prim-art"))
    out_dir.mkdir(parents=True, exist_ok=True)

    image = (Path(__file__).parent / "splitk_prim_probe.hsaco").read_bytes()
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

    buf = torch.zeros(1024 + 64, dtype=torch.float32, device="cuda")
    pat = torch.arange(0, 64, dtype=torch.float32, device="cuda") + 100.0  # 100..163
    buf[1024:1088] = pat
    module = ctypes.c_void_p()
    b = ctypes.create_string_buffer(image)
    assert lib.hipModuleLoadData(ctypes.byref(module), ctypes.cast(b, ctypes.c_void_p)) == 0
    fn = ctypes.c_void_p()
    assert lib.hipModuleGetFunction(ctypes.byref(fn), module, name.encode()) == 0
    arg0 = ctypes.c_ulonglong(buf.data_ptr())
    params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(arg0), ctypes.c_void_p))
    stream = torch.cuda.current_stream().cuda_stream
    rc = lib.hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0,
                                   ctypes.c_void_p(stream), params, None)
    assert rc == 0, f"launch {rc}"
    torch.cuda.synchronize()

    w = buf.detach().cpu().tolist()
    nt = w[512:512 + 64]
    lds = w[640:640 + 64]
    want = [100.0 + i for i in range(64)]
    result = {
        "schema": "amdgpu-sim.splitk-prim-probe.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "S1_atomicAdd_sum": w[0],
        "S1_pass": w[0] == 64.0,
        "S2_atomic_store_f2": [w[1], w[2]],
        "S2_pass": w[1] == 11.0 and w[2] == 22.0,
        "S3_nt_load_first8": [round(v, 1) for v in nt[:8]],
        "S3_pass": nt == want,
        "S4_global_load_lds_first8": [round(v, 1) for v in lds[:8]],
        "S4_pass": lds == want,
        "S5_marker": w[8],
        "S5_pass": w[8] == 77.0,
        "gem5": os.environ.get("SAGR_MANAGED_GEM5", "<unset>"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
