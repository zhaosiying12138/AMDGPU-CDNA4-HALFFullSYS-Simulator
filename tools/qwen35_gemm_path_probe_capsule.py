#!/usr/bin/env python3
"""ISA-path discriminator for the gem5 flydsl-GEMM zero-output defect.

Loads the hand-built gfx950 code object tools/mfma_isa_test/
gemm_path_probes.hsaco (one 64-lane wavefront) and checks five independent
sections against hardware-predictable results:

  T1 mfma      v_mfma_f32_16x16x32_bf16 all-ones A/B, C=0 -> 32.0f in every
               accumulator element, dst in VGPRs and in AGPRs.
  T2 mubufdma  buffer_load_dwordx4 ... offen sc0 lds into LDS at m0=0x1000
               with the hgemm kernel's rsrc (word2=-1, word3=0x27000),
               read back via ds_read_b128: dword i == pattern[lane*4+i].
  T3 ds        ds_write_b32/ds_read_b32 roundtrip -> 123.456f.
  T4 atom      global_atomic_pk_add_bf16 (2.0,2.0)bf16 x64 lanes onto
               (1.0,1.0)bf16 -> 129.0 in both halves.

The one that fails names the defective instruction class; the others are
cleared.  Exit 1 if any section fails, else 0.
"""

from __future__ import annotations

import argparse
import ctypes
import faulthandler

faulthandler.enable()
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HSACO = ROOT / "tools/mfma_isa_test/gemm_path_probes.hsaco"
HIP_SUCCESS = 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsaco", type=Path, default=DEFAULT_HSACO)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(
            __import__("os").environ.get(
                "QWEN35_GEMM_PATH_PROBE_OUTPUT",
                str(ROOT / "artifacts/qwen35-gemm-path-probe-capsule/v1"))))
    args = parser.parse_args()

    image = args.hsaco.read_bytes()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("simulated HIP device unavailable")
    hip_path = None
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        cand = os.path.join(d, "libamdhip64.so.7")
        if os.path.exists(cand):
            hip_path = cand
            break
    if hip_path is None:
        raise RuntimeError("libamdhip64.so.7 not on LD_LIBRARY_PATH")
    library = ctypes.CDLL(hip_path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    library.hipModuleLoadData.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    library.hipModuleLoadData.restype = ctypes.c_int
    library.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    library.hipModuleGetFunction.restype = ctypes.c_int
    library.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p)]
    library.hipModuleLaunchKernel.restype = ctypes.c_int

    def check(status, op):
        if status != HIP_SUCCESS:
            raise RuntimeError(f"{op} failed: {status}")

    # single buffer: [0,2048) results, [2048,6144) pattern, [6144,6146) atom
    gbuf = torch.zeros(6146, dtype=torch.float32, device="cuda")
    pat = (1.0 + 0.25 * torch.arange(4096, dtype=torch.float32))
    gbuf[2048:6144] = pat.to("cuda")
    gbuf[6144:6146] = 1.0

    module = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(image)
    check(library.hipModuleLoadData(
        ctypes.byref(module), ctypes.cast(buf, ctypes.c_void_p)),
        "hipModuleLoadData")
    function = ctypes.c_void_p()
    check(library.hipModuleGetFunction(
        ctypes.byref(function), module, b"gemm_path_probe"),
        "hipModuleGetFunction")

    holders = [ctypes.c_ulonglong(gbuf.data_ptr())]
    params = (ctypes.c_void_p * 1)(
        *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
    stream = torch.cuda.current_stream().cuda_stream
    check(library.hipModuleLaunchKernel(
        function, 1, 1, 1, 64, 1, 1, 0, ctypes.c_void_p(stream),
        params, None), "hipModuleLaunchKernel")
    torch.cuda.synchronize()

    w = gbuf.detach().cpu()
    p = pat
    a = w[6144:6146]

    result = {"schema": "amdgpu-sim.qwen35-gemm-path-probe.v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "hsaco_sha256": hashlib.sha256(image).hexdigest(),
              "sections": {}, "examples": []}

    def verdict(name, ok_count, total, detail):
        result["sections"][name] = {
            "ok": ok_count, "total": total,
            "pass": ok_count == total, "detail": detail}
        print(f"[gpprobe] {name}: {ok_count}/{total} "
              f"{'PASS' if ok_count == total else 'FAIL'} {detail}",
              flush=True)

    # T1 mfma VGPR dst: lanes*8 slots 0..3 == 32.0
    ok = 0
    for lane in range(64):
        good = all(abs(float(w[lane * 8 + i]) - 32.0) < 1e-3
                   for i in range(4))
        ok += good
        if not good and len(result["examples"]) < 8:
            result["examples"].append(
                {"t": "mfma_vgpr", "lane": lane,
                 "got": [float(w[lane * 8 + i]) for i in range(4)]})
    verdict("mfma_16x16x32_bf16_vgpr_dst", ok, 64, "expect 32.0 x4/lane")

    # T1b mfma AGPR dst dword0
    ok = sum(abs(float(w[lane * 8 + 4]) - 32.0) < 1e-3 for lane in range(64))
    verdict("mfma_16x16x32_bf16_agpr_dst", ok, 64, "expect 32.0")

    # T3 ds roundtrip
    ok = sum(abs(float(w[lane * 8 + 5]) - 123.456) < 1e-2
             for lane in range(64))
    verdict("ds_write_read_b32", ok, 64, "expect 123.456")

    # T2 mubuf LDS-DMA: out[512+lane*4+i] bits == pattern[lane*4+i] bits
    ok = 0
    for lane in range(64):
        good = True
        for i in range(4):
            got = float(w[512 + lane * 4 + i])
            want = float(p[lane * 4 + i])
            if abs(got - want) > 1e-6:
                good = False
                if len(result["examples"]) < 16:
                    result["examples"].append(
                        {"t": "mubuf_lds_dma", "lane": lane, "i": i,
                         "got": got, "want": want})
        ok += good
    verdict("buffer_load_dwordx4_lds", ok, 64, "expect pattern lane*4..+3")

    # T4 atomic pk_add_bf16: 1.0 + 64*2.0 = 129.0 both halves
    ok = sum(abs(float(a[i]) - 129.0) < 0.01 for i in range(2))
    verdict("global_atomic_pk_add_bf16", ok, 2,
            f"got {[float(a[0]), float(a[1])]} expect 129.0 x2")

    result["pass"] = all(s["pass"] for s in result["sections"].values())
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result["sections"], indent=1, sort_keys=True), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
