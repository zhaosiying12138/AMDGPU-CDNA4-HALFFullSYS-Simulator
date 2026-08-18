#!/usr/bin/env python3
"""Public-HIP capsule for the gfx950 global-to-LDS DMA load family.

What it covers
--------------
``global_load_lds_{ubyte,ushort,dword,dwordx3,dwordx4}`` executed through the
normal ROCr/HIP path into gem5. The signed sub-dword opcodes
(``global_load_lds_sbyte`` / ``_sshort``) are ISA-legal on gfx950 but have no
LLVM builtin -- the intrinsic carries a byte size and no signedness -- so they
are covered by the gem5 unit test instead of here.

Why it exists
-------------
The vLLM TP1 lane aborted with::

    src/arch/amdgpu/vega/insts/instructions.hh:45526: fatal: op idx 2 out of
    bounds in Inst_FLAT__FLAT_LOAD_LDS_DWORDX4::getOperandSize

That abort is the *loud* half of the defect. The silent half is placement: an
LDS-DMA writes to a slot chosen by M0 and the physical lane id, and getting
the destination address, the per-lane slot pitch, the sub-dword extension or
the lane masking wrong produces a completely plausible LDS image that happens
to be wrong. Every case here therefore compares the whole 2 KiB LDS tile
against a byte-level host oracle and a SHA-256, not just "it did not crash".

Cases
-----
  x4       16-byte transfer, 16-byte slot, all lanes
  x3       12-byte transfer in a 16-byte slot -- the fourth dword of every
           slot must survive untouched. A payload-sized 12-byte pitch would
           misplace every lane but zero.
  b32      4-byte transfer, 4-byte slot
  b8       1-byte transfer zero extended into a 4-byte slot
  b16      2-byte transfer zero extended into a 4-byte slot
  x4off    immediate offset 32, which the ISA applies to BOTH the global
           source address and the LDS destination address
  x4msk    even lanes only -- slot is indexed by physical lane id, so odd
           slots keep the sentinel and active lanes do not compact
  x4odd    wave-uniform LDS base of 2, i.e. M0 is not dword aligned
  x4oob    M0 = 0x40000000, an out-of-range LDS destination that hardware
           discards; the tile must come back untouched and the run must
           survive
  zeroexec eight LDS DMAs issued with EXEC = 0, twice gem5's
           max_cu_tokens = 4, followed by a real DMA. If the zero-lane path
           leaks its coalescer token the CU runs dry and the real DMA never
           lands.

Environment:
  CAPSULE_LDS_DMA_CASES   comma-separated subset of the case names
  CAPSULE_LDS_DMA_GROUPS  workgroups per launch (default 2)
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

HIP_SUCCESS = 0
H2D, D2H = 1, 2

WAVE = 64
TILE_U32 = 512
TILE_BYTES = TILE_U32 * 4
SENT = 0xC0DEC0DE
SENT_BYTES = SENT.to_bytes(4, "little")

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "lds_dma_probe.hip"

GROUPS = int(os.environ.get("CAPSULE_LDS_DMA_GROUPS", "2"))

# name -> (kernel, transfer bytes, immediate offset, lds base, lane predicate,
#          expected mnemonic)
CASES = {
    "x4":       ("lds_dma_x4",      16,  0, 0,          None,
                 "global_load_lds_dwordx4"),
    "x3":       ("lds_dma_x3",      12,  0, 0,          None,
                 "global_load_lds_dwordx3"),
    "b32":      ("lds_dma_b32",      4,  0, 0,          None,
                 "global_load_lds_dword"),
    "b8":       ("lds_dma_b8",       1,  0, 0,          None,
                 "global_load_lds_ubyte"),
    "b16":      ("lds_dma_b16",      2,  0, 0,          None,
                 "global_load_lds_ushort"),
    "x4off":    ("lds_dma_x4off",   16, 32, 0,          None,
                 "global_load_lds_dwordx4"),
    "x4msk":    ("lds_dma_x4msk",   16,  0, 0,          "even",
                 "global_load_lds_dwordx4"),
    "x4odd":    ("lds_dma_x4odd",   16,  0, 2,          None,
                 "global_load_lds_dwordx4"),
    "x4oob":    ("lds_dma_x4oob",   16,  0, 0x40000000, None,
                 "global_load_lds_dwordx4"),
    "zeroexec": ("lds_dma_zeroexec", 16, 0, 0,          None,
                 "global_load_lds_dwordx4"),
}

SELECTED = [c for c in os.environ.get(
    "CAPSULE_LDS_DMA_CASES", ",".join(CASES)).split(",") if c]


def emit(*parts: object) -> None:
    print(*parts, flush=True)


def compiler_env() -> dict:
    env = dict(os.environ)
    deps = "/home/zhaosiying/amdgpu-sim/env/conda/sglang-build-deps/lib"
    if Path(deps, "libxml2.so.2").exists():
        env["LD_LIBRARY_PATH"] = deps + ":" + env.get("LD_LIBRARY_PATH", "")
    env.pop("LD_PRELOAD", None)
    return env


def compile_kernels() -> tuple[bytes, str]:
    """Return (code object, gfx950 assembly)."""
    hip_root = os.environ["HIP_PATH"]
    rocm_root = os.environ["ROCM_PATH"]
    clang = Path(os.environ["HIP_CLANG_PATH"]) / "clang++"
    base = [
        str(clang), "-x", "hip", "--offload-device-only",
        "--no-gpu-bundle-output", "--offload-arch=gfx950",
        f"--hip-path={hip_root}", f"--rocm-path={rocm_root}", "-O3",
        str(SOURCE),
    ]
    with tempfile.TemporaryDirectory(prefix="lds-dma-capsule.") as directory:
        temporary = Path(directory)
        image = temporary / "lds_dma_probe.hsaco"
        listing = temporary / "lds_dma_probe.s"
        for output, extra in ((image, []), (listing, ["-S"])):
            done = subprocess.run(base + extra + ["-o", str(output)],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True,
                                  check=False, env=compiler_env())
            if done.returncode != 0:
                raise SystemExit("compile failed: "
                                 + done.stderr.strip()[:2000])
        payload = image.read_bytes()
        assembly = listing.read_text(encoding="utf-8", errors="replace")
    emit(f"COMPILE ok bytes={len(payload)} "
         f"image_sha256={hashlib.sha256(payload).hexdigest()}")
    return payload, assembly


def audit_isa(assembly: str) -> bool:
    """Prove the intended instructions were actually generated.

    A capsule that silently compiled to ds_write_b32 plus an ordinary
    global_load would pass its oracle while testing nothing.
    """
    ok = True
    counts: dict[str, int] = {}
    for line in assembly.splitlines():
        match = re.search(r"\b(global_load_lds_\w+)\b", line)
        if match:
            counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    emit("ISA " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    for name in SELECTED:
        mnemonic = CASES[name][5]
        if counts.get(mnemonic, 0) < 1:
            emit(f"ISA FAIL case {name} expected {mnemonic}, none emitted")
            ok = False

    # The zero-EXEC window must contain exactly eight LDS DMAs, and no
    # s_cbranch_execz may skip them.
    if "zeroexec" in SELECTED:
        window = re.search(
            r"s_mov_b64 exec, 0(.*?)s_mov_b64 exec, s\[",
            assembly, re.S)
        if not window:
            emit("ISA FAIL zeroexec: no explicit EXEC=0 window found")
            ok = False
        else:
            body = window.group(1)
            issued = len(re.findall(r"global_load_lds_dwordx4", body))
            skipped = len(re.findall(r"s_cbranch_execz", body))
            emit(f"ISA zeroexec window loads={issued} execz_skips={skipped}")
            if issued != 8 or skipped != 0:
                emit("ISA FAIL zeroexec: expected 8 loads and no execz skip")
                ok = False

    # Two immediate-offset checks, both load bearing for the semantics under
    # test: the offset must reach the instruction, and the unaligned LDS base
    # must reach M0 unrounded.
    if "x4off" in SELECTED and "offset:32" not in assembly:
        emit("ISA FAIL x4off: no 'offset:32' in the generated code")
        ok = False
    if "x4odd" in SELECTED and not re.search(r"s_mov_b32 m0, 2\b", assembly):
        emit("ISA FAIL x4odd: M0 was not set to 2")
        ok = False

    emit(f"ISA {'ok' if ok else 'FAIL'}")
    return ok


class Hip:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libamdhip64.so.7",
                               mode=os.RTLD_NOW | os.RTLD_LOCAL)
        seen = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            for token in ("libamdhip64", "libhsa-runtime64", "libself_amdgpu"):
                if token in line:
                    path = line.split()[-1]
                    if path not in seen:
                        seen.add(path)
                        emit(f"IDENTITY loaded={path}")
        lib = self.lib
        lib.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.hipSetDevice.argtypes = [ctypes.c_int]
        lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.c_size_t]
        lib.hipFree.argtypes = [ctypes.c_void_p]
        lib.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.c_int]
        lib.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                          ctypes.c_void_p]
        lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                             ctypes.c_void_p, ctypes.c_char_p]
        lib.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.hipStreamSynchronize.argtypes = [ctypes.c_void_p]

    def malloc(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        status = self.lib.hipMalloc(ctypes.byref(pointer), size)
        if status != HIP_SUCCESS:
            raise SystemExit(f"hipMalloc({size}) failed: {status}")
        return pointer


def source_bytes(groups: int) -> bytes:
    """Deterministic, position-revealing source image."""
    total = groups * WAVE * 16 + 64  # + slack for the immediate offset
    return bytes(((index * 37 + 11) & 0xFF) for index in range(total))


def oracle_tile(case: str, src: bytes, group: int) -> bytes:
    """Byte-level model of one workgroup's LDS tile after the DMA."""
    _, size, imm, lds_base, predicate, _ = CASES[case]
    slot = 4 if size <= 4 else 16

    tile = bytearray(SENT_BYTES * TILE_U32)

    for lane in range(WAVE):
        if predicate == "even" and (lane & 1):
            continue

        # Global source: base + group*1 KiB + lane*16, plus the immediate.
        src_off = group * WAVE * 16 + lane * 16 + imm
        # LDS destination: M0 + immediate + lane * slot.
        dst = lds_base + imm + lane * slot

        if size >= 4:
            payload = src[src_off:src_off + size]
        elif size == 2:
            payload = int.from_bytes(src[src_off:src_off + 2],
                                     "little").to_bytes(4, "little")
        else:
            payload = src[src_off].to_bytes(4, "little")

        # Every dword of the payload is placed independently, and a dword
        # that does not fit entirely inside the chunk is discarded.
        for word in range(len(payload) // 4):
            addr = dst + word * 4
            if addr < 0 or addr + 4 > TILE_BYTES:
                continue
            tile[addr:addr + 4] = payload[word * 4:word * 4 + 4]

    return bytes(tile)


def run_case(hip: Hip, module, case: str, groups: int) -> bool:
    kernel = CASES[case][0]
    function = ctypes.c_void_p()
    status = hip.lib.hipModuleGetFunction(ctypes.byref(function), module,
                                          kernel.encode("ascii"))
    if status != HIP_SUCCESS:
        raise SystemExit(f"hipModuleGetFunction({kernel}) failed: {status}")

    src = source_bytes(groups)
    out_bytes = groups * TILE_BYTES

    want = b"".join(oracle_tile(case, src, group) for group in range(groups))
    want_sha = hashlib.sha256(want).hexdigest()

    device_in = hip.malloc(len(src))
    device_out = hip.malloc(out_bytes)
    host_in = ctypes.create_string_buffer(src, len(src))
    hip.lib.hipMemcpy(device_in, ctypes.cast(host_in, ctypes.c_void_p),
                      len(src), H2D)
    # Poison the output so an unwritten buffer cannot look like a pass.
    poison = ctypes.create_string_buffer(b"\xa7" * out_bytes, out_bytes)
    hip.lib.hipMemcpy(device_out, ctypes.cast(poison, ctypes.c_void_p),
                      out_bytes, H2D)

    parameters = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(device_in), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(device_out), ctypes.c_void_p),
    )
    emit(f"CASE {case} kernel={kernel} transfer_bytes={CASES[case][1]} "
         f"imm_offset={CASES[case][2]} lds_base={CASES[case][3]} "
         f"groups={groups} block={WAVE}")

    start = time.monotonic()
    launch = hip.lib.hipModuleLaunchKernel(function, groups, 1, 1, WAVE, 1, 1,
                                           0, None, parameters, None)
    sync = hip.lib.hipStreamSynchronize(None)
    seconds = time.monotonic() - start
    emit(f"  launch_status={launch} sync_status={sync} "
         f"wall_s={seconds:.2f}")

    got_buffer = ctypes.create_string_buffer(out_bytes)
    hip.lib.hipMemcpy(ctypes.cast(got_buffer, ctypes.c_void_p), device_out,
                      out_bytes, D2H)
    got = got_buffer.raw[:out_bytes]
    got_sha = hashlib.sha256(got).hexdigest()

    emit(f"  ORACLE {case} want_sha256={want_sha}")
    emit(f"  ORACLE {case} got_sha256={got_sha}")

    ok = (got == want and launch == HIP_SUCCESS and sync == HIP_SUCCESS)
    if not ok:
        bad = [(i, got[i], want[i])
               for i in range(min(len(got), len(want)))
               if got[i] != want[i]]
        emit(f"  DIAG {case} mismatched_bytes={len(bad)} "
             f"first={bad[:8]}")
        # A first-mismatch offset is far more diagnostic than the count:
        # 12 says the dwordx3 pitch collapsed, 2 says M0 was rounded down.
        if bad:
            emit(f"  DIAG {case} first_bad_offset={bad[0][0]}")
    hip.lib.hipFree(device_in)
    hip.lib.hipFree(device_out)
    emit(f"  RESULT {case} {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    unknown = [c for c in SELECTED if c not in CASES]
    if unknown:
        raise SystemExit(f"unknown case(s): {unknown}")

    image, assembly = compile_kernels()
    isa_ok = audit_isa(assembly)

    hip = Hip()
    count = ctypes.c_int(0)
    hip.lib.hipGetDeviceCount(ctypes.byref(count))
    emit(f"DEVICE count={count.value}")
    if count.value < 1:
        raise SystemExit("no simulated device")
    hip.lib.hipSetDevice(0)

    module = ctypes.c_void_p()
    buffer = ctypes.create_string_buffer(image, len(image))
    status = hip.lib.hipModuleLoadData(ctypes.byref(module), buffer)
    if status != HIP_SUCCESS:
        raise SystemExit(f"hipModuleLoadData failed: {status}")

    results = {case: run_case(hip, module, case, GROUPS)
               for case in SELECTED}

    ok = isa_ok and all(results.values())
    emit("SUMMARY " + " ".join(f"{k}={'PASS' if v else 'FAIL'}"
                               for k, v in results.items()))
    emit(f"CAPSULE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
