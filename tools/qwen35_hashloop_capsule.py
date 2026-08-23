#!/usr/bin/env python3
"""Adjudicate gem5's execution of the vLLM generate-phase hash-probe kernel.

The lane fault (zcode-vllm-tp1-v10/v11, dispatch ~1870, kernel 32) is a JIT
hash-table probe whose decoded window is a loop of

    global_load_dword v1, v7, s[24:25] offset:14   (UNALIGNED entry dword)
    global_load_ushort v2, v7, s[2:3]
    s_waitcnt vmcnt(1) -> v_lshrrev/v_and/v_mul_lo_u32 on v1
    s_waitcnt vmcnt(0) -> v_mul_lo_u32 by the ushort, v_add next index
    s_cbranch_scc1 <head>

followed by an exit path that builds two candidate pointers per index with
v_lshl_add_u64 (SGPR-pair bases and the 64-bit (v8 << 2) + base form),
selects per lane with vcc (v_cmp_gt / v_subrev_co borrow), and dereferences
the selection with global_load_dword -- the instruction that faulted at a
wild ~16 GB offset from a kernarg base, i.e. the hash chain produced a
garbage index.

This capsule runs tools/mfma_isa_test/hashloop_probe.hsaco (an
instruction-for-instruction mirror with a KNOWN table whose chain terminates
in exactly 3 iterations) under the lane environment and checks every
per-lane output field.  Exit 0 = all fields exact; exit 2 = at least one
wrong field; a gem5 host-native panic is a third outcome (the run dies and
the runner reports the panic), reproducing the lane fault in minutes.

Table contract: ushort M = 3 at table+12; entries E0 = 0x00050002,
E1 = 0x00040001, E2 = 0x00010001 at table+14/+18/+22 (probe pointer
advances 4 B per iteration).  Chain: v0 = lane -> +30 -> +42 -> +45; the
secondary index v8 = v0 + 3 = lane + 48.  cap = 50: index < 50 reads
A[index], index >= 50 reads B[index-50]; A[i] = 0xA0000000 + i,
B[i] = 0xB0000000 + i.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HSACO = ROOT / "tools/mfma_isa_test/hashloop_probe.hsaco"
HIP_SUCCESS = 0
CAP = 50
M = 3
ENTRIES = (0x00050002, 0x00040001, 0x00010001)
# (lo * hi) * M per entry
PRODUCTS = (30, 12, 3)
# After the last pass v8 = product + v0_entry and v0 <- v8, so both final
# indices equal lane + sum(products): the back edge assigns v0 = v8.
IDX1 = [lane + sum(PRODUCTS) for lane in range(64)]      # v0 final
IDX2 = IDX1                                               # v8 final


def expected(lane: int) -> list[int]:
    def slot(idx: int) -> int:
        if idx < CAP:
            return 0xA0000000 + idx
        return 0xB0000000 + (idx - CAP)

    return [
        IDX1[lane],                       # v0 final
        IDX2[lane],                       # v8 final
        slot(IDX1[lane]),                 # pair-1 selected load
        slot(IDX2[lane]),                 # pair-2 selected load
        ENTRIES[-1],                      # last raw entry dword
        M,                                # header ushort
        PRODUCTS[-1],                     # final hash product
        3,                                # iteration count
        0,                                # v_sub_u32_e64 clamp: lane-2048 -> 0
        0xFFFFFFFF,                       # v_add_u32_e64 clamp: max+lane -> max
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsaco", type=Path, default=DEFAULT_HSACO)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(os.environ.get(
            "QWEN35_HASHLOOP_OUTPUT",
            str(ROOT / "artifacts/qwen35-hashloop-capsule/v1"))))
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

    # --- device buffers -----------------------------------------------------
    table = torch.zeros(64, dtype=torch.int32, device="cuda")
    bufA = torch.zeros(128, dtype=torch.int32, device="cuda")
    bufB = torch.zeros(128, dtype=torch.int32, device="cuda")
    out = torch.full((64 * 10,), -559038737, dtype=torch.int32, device="cuda")
    bufA.copy_(torch.tensor([0xA0000000 + i for i in range(128)],
                            dtype=torch.int64).to(torch.int32))
    bufB.copy_(torch.tensor([0xB0000000 + i for i in range(128)],
                            dtype=torch.int64).to(torch.int32))
    host_table = bytearray(64 * 4)
    struct.pack_into("<H", host_table, 12, M)          # ushort at +12
    struct.pack_into("<I", host_table, 14, ENTRIES[0])  # E0 at +14 (unaligned)
    struct.pack_into("<I", host_table, 18, ENTRIES[1])  # E1 at +18
    struct.pack_into("<I", host_table, 22, ENTRIES[2])  # E2 at +22
    table.copy_(torch.tensor(
        list(struct.unpack("<" + "I" * 64, bytes(host_table))),
        dtype=torch.int32))

    table_addr = table.data_ptr()
    hdr_addr = table_addr + 12                          # the ushort's base

    module = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(image)
    check(library.hipModuleLoadData(
        ctypes.byref(module), ctypes.cast(buf, ctypes.c_void_p)),
        "hipModuleLoadData")
    function = ctypes.c_void_p()
    check(library.hipModuleGetFunction(
        ctypes.byref(function), module, b"_Z14hashloop_probePmS_S_S_Pj"),
        "hipModuleGetFunction")
    args5 = (ctypes.c_ulonglong * 5)(
        table_addr, hdr_addr, bufA.data_ptr(), bufB.data_ptr(), out.data_ptr())
    params = (ctypes.c_void_p * 5)()
    for i in range(5):
        params[i] = ctypes.cast(ctypes.byref(args5, 8 * i), ctypes.c_void_p)
    stream = torch.cuda.current_stream().cuda_stream
    check(library.hipModuleLaunchKernel(
        function, 1, 1, 1, 64, 1, 1, 0, ctypes.c_void_p(stream), params, None),
        "hipModuleLaunchKernel")
    torch.cuda.synchronize()
    words = [w & 0xFFFFFFFF for w in out.detach().cpu().tolist()]

    fields = ["idx1", "idx2", "load1", "load2",
              "raw_entry", "ushort_M", "product", "iters",
              "clamp_sub", "clamp_add"]
    bad_total = 0
    per_field: dict[str, dict] = {}
    examples = []
    for k, name in enumerate(fields):
        bad = []
        for lane in range(64):
            got = words[lane * 10 + k]
            want = expected(lane)[k]
            if got != want:
                bad.append({"lane": lane, "got": got, "want": want})
        per_field[name] = {"bad_lanes": len(bad), "first": bad[:3]}
        bad_total += len(bad)
        examples += bad[:2]

    report = {"schema": "amdgpu-sim.qwen35-hashloop.v1",
              "hsaco": str(args.hsaco),
              "fields": per_field, "examples": examples,
              "passed": bad_total == 0,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "finished_at": datetime.now(timezone.utc).isoformat()}
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="ascii")
    print(f"[hashloop] bad fields total={bad_total} per-field="
          f"{ {n: per_field[n]['bad_lanes'] for n in fields} }", flush=True)
    for e in examples[:6]:
        print(f"[hashloop] mismatch {e}", flush=True)
    print(f"[hashloop] passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
