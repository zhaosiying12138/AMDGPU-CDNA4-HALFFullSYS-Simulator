#!/usr/bin/env python3
"""Adjudicate saveexec -> v_cndmask_b32_e64 under gem5 with exec predication
modeled exactly.

tools/qwen35_cndmask_e64_capsule.py attributes its p3 failure to gem5 reading
a vcc-shaped condition after s_and_saveexec_b64.  gem5-level instrumentation
disproved that: the e64 cndmask reads the correct old-exec value.  This
capsule re-runs the discriminator with hardware-exact expectations:

  p0/p1/p2  s_mov-written s[2:3] conditions (control, unchanged)
  p3a       s_and_saveexec_b64 s[2:3], vcc; the cndmask runs under the NEW
            exec (lanes<32), so only those lanes store; every active lane
            must take the TRUE path (old-exec condition = all ones)
  p3b       s_mov_b64 exec, s[2:3] restores the old exec; the SAME cndmask
            with the SAME s[2:3] now selects src1 on EVERY lane
  raw       s[2:3] dumped as compact qwords (one per lane) -- every word
            must be 0xffffffff, compared as an unsigned 32-bit pattern

Exit 0 = all exact; exit 2 = mismatch (report names pattern and lanes).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HSACO = ROOT / "tools/mfma_isa_test/cndmask_probe2.hsaco"
HIP_SUCCESS = 0


def src0(lane: int) -> int:
    return 1000 + lane * 10


def src1(lane: int) -> int:
    return 2000 + lane


def expected(pattern: int, lane: int) -> int:
    if pattern == 0:
        return src1(lane) if lane < 16 else src0(lane)
    if pattern == 1:
        return src1(lane)
    if pattern == 2:
        return src1(lane) if 16 <= lane < 32 else src0(lane)
    if pattern == 3:
        # exec = vcc & old_exec = lanes<32 at both the cndmask and the store:
        # active lanes must select src1 (old-exec condition), inactive lanes
        # do not store and the zero-initialized buffer keeps 0.
        return src1(lane) if lane < 32 else 0
    if pattern == 4:
        # exec restored to the old exec (all ones): the same cndmask e64 with
        # the same s[2:3] selects src1 on every lane.
        return src1(lane)
    raise ValueError(pattern)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsaco", type=Path, default=DEFAULT_HSACO)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(os.environ.get(
            "QWEN35_CNDMASK_PROBE2_OUTPUT",
            str(ROOT / "artifacts/qwen35-cndmask-probe2-capsule/v1"))))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    library.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    library.hipModuleLoadData.restype = ctypes.c_int
    library.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    library.hipModuleGetFunction.restype = ctypes.c_int
    library.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.hipModuleLaunchKernel.restype = ctypes.c_int

    def check(status, op):
        if status != HIP_SUCCESS:
            raise RuntimeError(f"{op} failed: {status}")

    out = torch.zeros(5 * 64 + 128, dtype=torch.int32, device="cuda")
    module = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(image)
    check(library.hipModuleLoadData(
        ctypes.byref(module), ctypes.cast(buf, ctypes.c_void_p)),
        "hipModuleLoadData")
    function = ctypes.c_void_p()
    check(library.hipModuleGetFunction(
        ctypes.byref(function), module, b"_Z19cndmask_probe2_testPj"),
        "hipModuleGetFunction")
    arg0 = ctypes.c_ulonglong(out.data_ptr())
    params = (ctypes.c_void_p * 1)(
        ctypes.cast(ctypes.byref(arg0), ctypes.c_void_p))
    stream = torch.cuda.current_stream().cuda_stream
    check(library.hipModuleLaunchKernel(
        function, 1, 1, 1, 64, 1, 1, 0, ctypes.c_void_p(stream), params, None),
        "hipModuleLaunchKernel")
    torch.cuda.synchronize()
    words = out.detach().cpu().tolist()

    report = {"schema": "amdgpu-sim.qwen35-cndmask-probe2.v1",
              "hsaco": str(args.hsaco),
              "patterns": {}, "started_at":
                  datetime.now(timezone.utc).isoformat()}
    bad_total = 0
    names = ["p0_mov_ffff", "p1_mov_ones", "p2_mov_ffff0000",
             "p3_saveexec_masked", "p4_saveexec_restored"]
    for pattern in range(5):
        bad = []
        for lane in range(64):
            got = words[pattern * 64 + lane] & 0xFFFFFFFF
            want = expected(pattern, lane) & 0xFFFFFFFF
            if got != want:
                bad.append({"lane": lane, "got": got, "want": want})
        report["patterns"][names[pattern]] = {
            "bad_lanes": len(bad),
            "first": bad[:3],
        }
        bad_total += len(bad)
    # raw s[2:3]: compact qwords, words 320..447, all must be 0xffffffff
    raw = [w & 0xFFFFFFFF for w in words[320:448]]
    bad_raw = [i for i, w in enumerate(raw) if w != 0xFFFFFFFF]
    report["saveexec_dst_raw"] = {
        "bad_words": len(bad_raw),
        "first": bad_raw[:4],
        "unique": sorted(set(raw)),
        "expected": "every word 0xffffffff (old exec, all ones)",
    }
    raw_ok = not bad_raw
    report["passed"] = bad_total == 0 and raw_ok
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="ascii")
    print(f"[cndmask-probe2] bad lanes total={bad_total} "
          f"per-pattern={[report['patterns'][n]['bad_lanes'] for n in names]}",
          flush=True)
    print(f"[cndmask-probe2] saveexec dst raw: {report['saveexec_dst_raw']}",
          flush=True)
    print(f"[cndmask-probe2] passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
