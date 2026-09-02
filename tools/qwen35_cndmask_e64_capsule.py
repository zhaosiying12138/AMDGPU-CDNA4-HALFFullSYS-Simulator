#!/usr/bin/env python3
"""Adjudicate v_cndmask_b32_e64 with a scalar-pair condition under gem5.

The vLLM warmup crash kernel computes its buffer_store_dwordx4 offsets via
    v_cndmask_b32_e64 v94, v111, v14, s[2:3]
with the crash dump showing s2=s3=0xffffffff (every bit set), and the store
then went to a wild unaligned address.  This capsule runs a hand-encoded
kernel (tools/mfma_isa_test/cndmask_e64.hsaco) through the same e64 form
under four condition patterns -- 0x0000FFFF, all-ones (the crash state),
0xFFFF0000, and an s_and_saveexec_b64-saved exec -- with per-lane src1/src2
values, and compares against the hardware select semantics (bit n of the
condition pair selects src1 for lane n).

Exit 0 = all four patterns exact; exit 2 = at least one wrong lane (the
report names the pattern and lanes, which localizes the gem5 decode of the
e64 condition operand).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HSACO = ROOT / "tools/mfma_isa_test/cndmask_e64.hsaco"
HIP_SUCCESS = 0


def expected(pattern: int, lane: int) -> int:
    # Hardware e64 semantics: dst = cond.bit(lane) ? SRC1 : SRC0, and the
    # kernel's operand order is (v5=src0=1000+10*lane, v6=src1=2000+lane).
    src0 = 1000 + lane * 10
    src1 = 2000 + lane
    cond_lanes = {
        0: range(0, 16),
        1: range(0, 64),
        2: range(16, 32),
        3: range(0, 64),  # s_and_saveexec dst must be the OLD exec (all ones)
    }[pattern]
    return src1 if lane in cond_lanes else src0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsaco", type=Path, default=DEFAULT_HSACO)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(os.environ.get(
            "QWEN35_CNDMASK_E64_OUTPUT",
            str(ROOT / "artifacts/qwen35-cndmask-e64-capsule/v1"))))
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

    out = torch.zeros(256 + 128, dtype=torch.int32, device="cuda")
    module = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(image)
    check(library.hipModuleLoadData(
        ctypes.byref(module), ctypes.cast(buf, ctypes.c_void_p)),
        "hipModuleLoadData")
    function = ctypes.c_void_p()
    check(library.hipModuleGetFunction(
        ctypes.byref(function), module, b"_Z16cndmask_e64_testPj"),
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

    report = {"schema": "amdgpu-sim.qwen35-cndmask-e64.v1",
              "hsaco": str(args.hsaco),
              "patterns": {}, "examples": [],
              "started_at": datetime.now(timezone.utc).isoformat()}
    bad_total = 0
    for pattern in range(4):
        bad = []
        for lane in range(64):
            got = words[pattern * 64 + lane]
            want = expected(pattern, lane)
            if got != want:
                bad.append({"lane": lane, "got": got, "want": want})
        report["patterns"][f"p{pattern}"] = {
            "bad_lanes": len(bad),
            "first": bad[:3],
        }
        bad_total += len(bad)
    # pattern 4: raw s[2:3] after s_and_saveexec (lane-consistent pairs)
    raw = words[256:256 + 128]
    uniq = sorted(set(raw))
    report["saveexec_dst_raw"] = {
        "unique_pairs": len(uniq),
        "lo_values": sorted(set(raw[0::2]))[:4],
        "hi_values": sorted(set(raw[1::2]))[:4],
        "expected": "lo=0xffffffff hi=0xffffffff (old exec, all ones)",
    }
    saveexec_ok = uniq == [0xFFFFFFFF]
    report["passed"] = bad_total == 0 and saveexec_ok
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="ascii")
    print(f"[cndmask-e64] bad lanes total={bad_total} "
          f"per-pattern={[report['patterns'][f'p{p}']['bad_lanes'] for p in range(4)]}",
          flush=True)
    print(f"[cndmask-e64] saveexec dst raw: {report['saveexec_dst_raw']}", flush=True)
    print(f"[cndmask-e64] passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
