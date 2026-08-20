#!/usr/bin/env python3
"""Direct ISA-level test of the MFMA AGPR chain, accvgpr read/write, v_pk_mul_f32.

The aiter CK attention capsule diverges on every multi-key token while the
single-key identity case is exact, and a Triton tl.dot differential (VGPR
accumulators, same MFMA shapes) is bit-exact -- so the remaining functional
suspect is the AGPR path: v_mfma_f32_32x32x16_bf16 with C/D in AGPR plus
v_accvgpr_read/write.  This capsule loads a hand-assembled gfx950 code
object (tools/mfma_isa_test/mfma_test.s) with uniform constant operands so
the expected result is layout-independent:

  variant A: no MFMA, no pk_mul; roundtrip only (A=B=1.0 bf16, K=16,
  C chained from the first op's 16.0f AGPR result)
  accvgpr write/read roundtrip = 100.0f
  v_pk_mul_f32 (1.0,2.0)*(3.0,1.5) lo = 3.0f

Each of the 64 lanes stores those four dwords, so any AGPR-offset aliasing,
accumulator clobbering, or packed-lane mixup shows up as a wrong lane value.
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
import struct

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HSACO = ROOT / "tools/mfma_isa_test/mfma_test.hsaco"
HIP_SUCCESS = 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsaco", type=Path, default=DEFAULT_HSACO)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            __import__("os").environ.get(
                "QWEN35_MFMA_ISA_OUTPUT",
                str(ROOT / "artifacts/qwen35-mfma-isa-capsule/20260820-v1"),
            )
        ),
    )
    parser.add_argument("--hip-lib", default=str(ROOT / "env/rocm/product-v1-4d9d40454031c7345f25da81b6781995b09a3b10e4dd66026e019306fc7ee39b/lib/libamdhip64.so.7"))
    args = parser.parse_args()

    image = args.hsaco.read_bytes()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("simulated HIP device unavailable")
    # Torch dlopens its own bundled libamdhip64 at import; opening a second
    # HIP instance from another path and mixing its module handles with the
    # device context torch initialized segfaults at launch.  dlopen of the
    # exact file torch uses returns the same shared instance.
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

    out = torch.zeros(256 * 8, dtype=torch.float32, device="cuda")
    module = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(image)
    check(
        library.hipModuleLoadData(ctypes.byref(module), ctypes.cast(buf, ctypes.c_void_p)),
        "hipModuleLoadData",
    )
    function = ctypes.c_void_p()
    check(
        library.hipModuleGetFunction(ctypes.byref(function), module, b"_Z17mfma_accvgpr_onlyPf"),
        "hipModuleGetFunction",
    )
    # The holder must outlive the launch: a temporary would be freed right
    # after the cast and the runtime would dereference a dangling pointer.
    arg0 = ctypes.c_ulonglong(out.data_ptr())
    params = (ctypes.c_void_p * 1)(
        ctypes.cast(ctypes.byref(arg0), ctypes.c_void_p)
    )
    stream = torch.cuda.current_stream().cuda_stream
    check(
        library.hipModuleLaunchKernel(
            function, 1, 1, 1, 256, 1, 1, 0, ctypes.c_void_p(stream), params, None
        ),
        "hipModuleLaunchKernel",
    )
    torch.cuda.synchronize()
    words = tuple(out.detach().cpu().tolist())
    # report slot 4 (plain-VGPR control) separately
    ctrl_bad = sum(1 for lane in range(256) if abs(words[lane*8+4] - 10.0) > 1e-4)
    print(f"control slot (plain v_mov+store, expect 10.0): bad {ctrl_bad}/256", flush=True)
    import struct as _s
    execs, laneids = [], []
    for lane in range(256):
        execs.append(_s.unpack("<I", _s.pack("<f", words[lane*8+5]))[0])
        laneids.append(_s.unpack("<I", _s.pack("<f", words[lane*8+6]))[0])
    print("exec_lo per wave (unique):", sorted(set(execs))[:6], "as hex:", [hex(x) for x in sorted(set(execs))[:6]], flush=True)
    print("laneid pattern first wave:", laneids[:20], flush=True)
    print("exec per lane first wave:", [hex(e) for e in execs[:20]], flush=True)

    expectations = (0.0, 0.0, 100.0, 0.0)
    per_slot_bad = [0, 0, 0, 0]
    examples = []
    for lane in range(256):
        for slot in range(4):
            got = words[lane * 8 + slot]
            if abs(got - expectations[slot]) > 1e-4:
                per_slot_bad[slot] += 1
                if len(examples) < 8:
                    examples.append(
                        {"lane": lane, "slot": slot, "got": got,
                         "want": expectations[slot]}
                    )
    result = {
        "schema": "amdgpu-sim.qwen35-mfma-isa-capsule.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hsaco_sha256": hashlib.sha256(image).hexdigest(),
        "per_slot_bad": dict(
            zip(("mfma_a0", "mfma_a1", "accvgpr_roundtrip", "pk_mul_lo"), per_slot_bad)
        ),
        "examples": examples,
        "correct": all(v == 0 for v in per_slot_bad),
    }
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
