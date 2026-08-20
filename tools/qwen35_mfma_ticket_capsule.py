#!/usr/bin/env python3
"""Ticket-addressed ISA differential: v_mbcnt, AGPR roundtrip, chained MFMA, v_pk_mul."""
import argparse, ctypes, faulthandler, json, os, struct
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
HIP_SUCCESS = 0
faulthandler.enable()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(
        __import__("os").environ.get("QWEN35_MFMA_ISA_OUTPUT",
            str(ROOT / "artifacts/qwen35-mfma-isa-capsule/ticket-v1"))))
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
    raw = out.detach().cpu()
    words = raw.tolist()
    nz = (raw != 0).nonzero().flatten().tolist()
    print(f"output nonzero words: {len(nz)}/2048 first16={nz[:16]}", flush=True)
    u32 = lambda f: struct.unpack("<I", struct.pack("<f", f))[0]
    tickets = sorted(set(int(words[t*8]) for t in range(256)) | {0})
    slots = {}
    for name, want in (("mbcnt", None), ("mfma_a0", 32.0), ("mfma_a1", 32.0),
                       ("accvgpr_rt", 100.0), ("pk_mul_lo", 3.0)):
        idx = ("mbcnt","mfma_a0","mfma_a1","accvgpr_rt","pk_mul_lo").index(name)
        vals = [words[t*8+idx] for t in range(256)]
        bad = sum(1 for v in vals if want is not None and abs(v-want) > 1e-4)
        slots[name] = {"bad": bad, "want": want,
                       "first8": [round(v,4) for v in vals[:8]]}
    mbcnt_values = sorted(u32(words[t*8]) for t in range(256))
    result = {
        "schema": "amdgpu-sim.qwen35-mfma-ticket-capsule.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "slots": slots,
        "mbcnt_sorted_unique": mbcnt_values[:8] + ["..."] + mbcnt_values[-4:],
    }
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
