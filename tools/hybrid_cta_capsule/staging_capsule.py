#!/usr/bin/env python3
"""Staging-path capsule: four interleaved global dwordx4 tiles -> ALU consume.

The CK attention defect localizes to early-k-block staging: dims [0:128)
vanish while the final k-blocks are exact (sweep/sweeptail/dimsplit
probes).  Every register-level mechanism is verified correct, so the last
untested path is global dwordx4 load -> s_waitcnt vmcnt -> nearby ALU
consumers, the shape of CK's double-buffered tile fetch.  Two otherwise
identical kernels from stage_dwordx4.hsaco are launched:

  stage_imm   - consumers immediately after s_waitcnt vmcnt(0)
  stage_delay - 64-deep dependent v_add chain between waitcnt and the
                consumers

Each loads four 16-byte-per-lane tiles (address math interleaved between
the wide loads so earlier tiles are not hot), consumes them with 12
v_add_u32, and stores per-ticket dwords taken from an LDS ds_add_rtn
counter.  Pointer plumbing is SGPR-free: the kernel's C preamble parks
the argument pointers in __shared__ and the asm reads them with
ds_read_b64 (the previous s_load-from-out[0:1] scheme read garbage).

Input dword at tile t, index i: (0x100000 << t) + i.  With 256 threads
(4x wave64, wave-lane L in [0,64)) each lane dword k sees
value(L,k,t) = (0x100000 << t) + (L*4 + k), so the full sum is
0xF00000 + 4*(L*4 + k): 256 distinct values in [0xF00000, 0xF00400),
each expected exactly 4 times (once per wave) over the 256 tickets x 4
dwords.  A vanished tile subset is decodable from the sum's high nibble
(bit t set = tile t survived).
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
faulthandler.enable()

# mangled names verified against stage_dwordx4.hsaco (readelf -sW)
SYMBOLS = {"stage_imm": b"_Z9stage_immPfPKi", "stage_delay": b"_Z11stage_delayPfPKi"}

TILES = 4
TILE_DWORDS = 256          # 64 wave-lanes x 4 dwords (16 B per lane)
IN_DWORDS = TILES * TILE_DWORDS
TICKETS = 256              # one per thread (4 waves x 64)
OUT_DWORDS = TICKETS * 8   # ticket slots: dword ticket*8 + k
FULL_MASK = 0xF            # all four tiles survived
EXPECTED_BASE = FULL_MASK << 20


def alloc_guarded(n, dtype):
    """Allocate with the low 32 bits clear of the kernel's byte offsets.

    The asm builds flat addresses with a 32-bit low-half add and no
    carry into the high half (inputs reach +0x1010, out reaches +0x1FFC),
    so keep 0x4000 of headroom below the 4 GiB boundary.
    """
    for _ in range(16):
        t = torch.zeros(n, dtype=dtype, device="cuda")
        if (t.data_ptr() & 0xFFFFFFFF) <= (1 << 32) - 0x4000:
            return t
    raise RuntimeError("could not place allocation away from 4 GiB boundary")


def run_variant(lib, function, inputs):
    out = alloc_guarded(OUT_DWORDS, torch.int32)
    holders = [ctypes.c_ulonglong(out.data_ptr()), ctypes.c_ulonglong(inputs.data_ptr())]
    params = (ctypes.c_void_p * 2)(
        *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
    stream = torch.cuda.current_stream().cuda_stream
    rc = lib.hipModuleLaunchKernel(function, 1, 1, 1, 256, 1, 1, 0,
                                   ctypes.c_void_p(stream), params, None)
    assert rc == 0, f"launch {rc}"
    torch.cuda.synchronize()
    return out.detach().cpu().numpy().astype(np.int64).tolist(), holders


def analyze(u):
    stored = [u[t * 8 + k] for t in range(TICKETS) for k in range(4)]
    hist = {}
    for v in stored:
        hist[v] = hist.get(v, 0) + 1
    valid = list(range(EXPECTED_BASE, EXPECTED_BASE + 4 * TILE_DWORDS, 4))
    invalid = [v for v in stored if EXPECTED_BASE > v or v >= EXPECTED_BASE + 1024]
    exact = all(hist.get(v, 0) == 4 for v in valid) and \
        sum(hist.get(v, 0) for v in valid) == len(stored)
    # decode surviving-tile masks: v = mask<<20 + popcount(mask)*i
    masks = {}
    undecoded = 0
    for v, c in hist.items():
        m = (v >> 20) & 0xF
        rem = v & 0xFFFFF
        n = bin(m).count("1")
        if m and not (v >> 24) and rem % n == 0 and rem // n < TILE_DWORDS:
            masks[m] = masks.get(m, 0) + c
        else:
            undecoded += c
    return {
        "stored_dwords": len(stored),
        "empty_slots": sum(1 for v in stored if v == 0),
        "distinct_values": len(hist),
        "valid_distinct": sum(1 for v in hist if EXPECTED_BASE <= v < EXPECTED_BASE + 1024),
        "invalid_values": len(invalid),
        "exact_multiset": bool(exact),
        "tile_survival_mask_histogram": {f"0x{m:X}": c for m, c in sorted(masks.items())},
        "undecodable_values": undecoded,
        "first_ticket_values": u[0:4],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(
        os.environ.get("STAGING_CAPSULE_OUTPUT",
                       str(ROOT / "artifacts/qwen35-staging-capsule/v1"))))
    args = parser.parse_args()

    image = (Path(__file__).parent / "stage_dwordx4.hsaco").read_bytes()
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

    # distinct tile tags make any surviving-tile subset decodable from the sums
    host = np.zeros(IN_DWORDS, dtype=np.int32)
    for t in range(TILES):
        host[t * TILE_DWORDS:(t + 1) * TILE_DWORDS] = \
            (0x100000 << t) + np.arange(TILE_DWORDS, dtype=np.int64)
    inputs = torch.from_numpy(host).to("cuda")  # int32 throughout: bit-exact

    module = ctypes.c_void_p()
    raw = ctypes.create_string_buffer(image)
    assert lib.hipModuleLoadData(ctypes.byref(module), ctypes.cast(raw, ctypes.c_void_p)) == 0
    funcs = {}
    for name, sym in SYMBOLS.items():
        f = ctypes.c_void_p()
        assert lib.hipModuleGetFunction(ctypes.byref(f), module, sym) == 0, sym
        funcs[name] = f

    variants = {}
    keepalive = []
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    for sym in ("stage_imm", "stage_delay"):
        u, holders = run_variant(lib, funcs[sym], inputs)
        keepalive.extend(holders)
        variants[sym] = analyze(u)
        (args.output_dir / f"output_{sym}.bin").write_bytes(
            struct.pack(f"<{len(u)}i", *u))

    imm, delay = variants["stage_imm"], variants["stage_delay"]
    if imm["exact_multiset"] and delay["exact_multiset"]:
        verdict = "CLEAR"
        detail = ("staging path exact for both immediate and delayed consumers; "
                  "suspect moves elsewhere")
    elif imm["exact_multiset"] and not delay["exact_multiset"]:
        verdict = "REPRODUCED"
        detail = "delayed-consumer variant only: delay chain corrupts, immediate exact"
    else:
        verdict = "REPRODUCED"
        same = (imm["invalid_values"] == delay["invalid_values"]
                and imm["tile_survival_mask_histogram"] == delay["tile_survival_mask_histogram"])
        detail = ("identical loss in both variants -> load data loss (immediacy "
                  "irrelevant)" if same else
                  "loss differs between variants -> consumer timing matters (RAW "
                  "hazard shape)")

    result = {
        "schema": "amdgpu-sim.qwen35-staging-capsule.v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "verdict_detail": detail,
        "expected_values": {
            "base": EXPECTED_BASE,
            "count": TILE_DWORDS,
            "each_seen": TICKETS * 4 // TILE_DWORDS,
            "note": "0xF00000 + 4*i for i in [0,256), each exactly 4x",
        },
        "variants": variants,
        "gem5": os.environ.get("SAGR_MANAGED_GEM5", "<unset>"),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "result.json.sha256").write_text(
        hashlib.sha256((args.output_dir / "result.json").read_bytes()).hexdigest() + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
