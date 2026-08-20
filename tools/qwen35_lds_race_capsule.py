#!/usr/bin/env python3
"""Cross-wave LDS visibility probe: wave 0 writes, s_barrier, all read."""
import argparse, ctypes, faulthandler, json, os
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
faulthandler.enable()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(
        __import__("os").environ.get("QWEN35_MFMA_ISA_OUTPUT",
            str(ROOT / "artifacts/qwen35-lds-race-capsule/v1"))))
    args = parser.parse_args()

    image = (ROOT / "tools/mfma_isa_test/mfma_test.hsaco").read_bytes()
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

    out = torch.zeros(256 * 8, dtype=torch.float32, device="cuda")
    module = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(image)
    assert lib.hipModuleLoadData(ctypes.byref(module), ctypes.cast(buf, ctypes.c_void_p)) == 0
    function = ctypes.c_void_p()
    assert lib.hipModuleGetFunction(ctypes.byref(function), module, b"_Z14mfma_agpr_testPf") == 0
    arg0 = ctypes.c_ulonglong(out.data_ptr())
    params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(arg0), ctypes.c_void_p))
    stream = torch.cuda.current_stream().cuda_stream
    rc = lib.hipModuleLaunchKernel(function, 1, 1, 1, 256, 1, 1, 0,
                                   ctypes.c_void_p(stream), params, None)
    assert rc == 0, f"launch {rc}"
    torch.cuda.synchronize()
    words = out.detach().cpu().tolist()
    # every lane stored its read at ticket slot0; expected value = (id&63)+60
    vals = [words[t*8] for t in range(256)]
    import struct as _s
    u = lambda f: _s.unpack("<I", _s.pack("<f", f))[0]
    exps = [u(words[t*8+1]) for t in range(256)]
    raws = [[u(words[t*8+2+j]) for j in range(4)] for t in range(256)]
    uvals = [u(v) for v in vals]
    bad = sum(1 for i, x in enumerate(uvals) if x != exps[i])
    result = {
        "schema": "amdgpu-sim.qwen35-lds-race-capsule.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "correct": bad == 0,
        "stale_reads": bad,
        "first_stale": [i for i, x in enumerate(uvals) if not (60 <= x <= 123)][:8],
        "sample_read": uvals[:8],
        "sample_expected": exps[:8],
        "raw_v0123_ticket0_4": raws[:4],
    }
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if bad == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
