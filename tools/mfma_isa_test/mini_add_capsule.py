#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 AMDGPU-CDNA4-HALFFullSYS-Simulator contributors
# Full license text: LICENSE at the repository root.
import ctypes, json, os
from datetime import datetime, timezone
from pathlib import Path
import torch

def main() -> int:
    which = os.environ.get("MINI_ADD_KERNEL", "_Z8mini_addPf")
    out_dir = Path(os.environ.get("MINI_ADD_OUTPUT", "/tmp/miniart"))
    out_dir.mkdir(parents=True, exist_ok=True)
    image = (Path(__file__).parent / "mini_add.hsaco").read_bytes()
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
    buf = torch.zeros(4, dtype=torch.float32, device="cuda")
    module = ctypes.c_void_p()
    b = ctypes.create_string_buffer(image)
    assert lib.hipModuleLoadData(ctypes.byref(module), ctypes.cast(b, ctypes.c_void_p)) == 0
    fn = ctypes.c_void_p()
    assert lib.hipModuleGetFunction(ctypes.byref(fn), module, which.encode()) == 0
    arg0 = ctypes.c_ulonglong(buf.data_ptr())
    params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(arg0), ctypes.c_void_p))
    stream = torch.cuda.current_stream().cuda_stream
    rc = lib.hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0,
                                   ctypes.c_void_ptr(stream) if False else ctypes.c_void_p(stream), params, None)
    assert rc == 0, f"launch {rc}"
    torch.cuda.synchronize()
    w = buf.detach().cpu().tolist()
    r = {"kernel": which, "result": w[0],
         "pass_mini_add": w[0] == 64.0,   # mini_add: 64 lanes x 1.0
         "pass_mini_add64": abs(w[0] - 2080.0) < 1e-3,  # 1..64 sum
         "created_utc": datetime.now(timezone.utc).isoformat()}
    (out_dir / "result.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
