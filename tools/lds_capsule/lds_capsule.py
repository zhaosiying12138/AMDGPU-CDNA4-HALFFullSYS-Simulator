#!/usr/bin/env python3
"""Public-HIP capsule for the group-segment (LDS) dispatch boundary.

Why this exists
---------------
The gfx950 device this product advertises to the compiler has 160 KiB of
local data store per compute unit: the model topology publishes
``lds_size_in_kb 160`` and the pinned toolchain enforces exactly that limit
(163840 bytes; 163844 is rejected).  gem5's ``LdsState`` defaulted to 64 KiB.

A workgroup that asks for more group segment than one compute unit provides
never satisfies ``ComputeUnit::hasDispResources()``.  gem5's dispatcher pushes
such a kernel back onto its queue and returns *without re-arming its own
event*, and the only things that re-arm it -- a workgroup completing, a launch
invalidate finishing, a new kernel arriving -- cannot happen on an idle
device.  The simulator then advances simulated time at full host CPU with zero
retired instructions while the host waits forever for a completion signal.
This is what stopped the unchanged-upstream vLLM TP1 lane at dispatch 441.

Each case launches one 256-thread workgroup that cooperatively fills its
entire group segment, crosses two barriers, and reads slots written by other
waves, so a wrong LDS size, a lost barrier or a truncated allocation all show
up as a wrong answer rather than as "it ran".  Every case carries an
independent host oracle and a SHA-256 over the whole output buffer.

Cases (bytes of static group segment):
  32k   32768 -- control; fits in either configuration.
  96k   98304 -- above the old 64 KiB compute unit, below the real device.
  160k 163840 -- the exact device maximum the toolchain enforces.

Environment:
  CAPSULE_LDS_CASES   comma-separated subset of ``32k,96k,160k``
  CAPSULE_LDS_GROUPS  workgroups per launch (default 2, exercises reuse)
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HIP_SUCCESS = 0
H2D, D2H = 1, 2
SENTINEL = -559038737  # 0xDEADBEEF as a signed int

BLOCK = 256
GROUPS = int(os.environ.get("CAPSULE_LDS_GROUPS", "2"))
CASES = [c for c in os.environ.get("CAPSULE_LDS_CASES", "32k,96k,160k").split(",") if c]

# name -> group segment ints (4 bytes each)
CASE_INTS = {"32k": 8192, "96k": 24576, "160k": 40960}

KERNEL_TEMPLATE = r"""
#include <hip/hip_runtime.h>

// One statically sized group segment per case.  Static rather than dynamic so
// the size lands in the code object's group_segment_fixed_size and the
// compiler itself enforces the target's limit at build time.
extern "C" __global__ void lds_probe_%(tag)s(const int *input, int *out) {
  __shared__ int tile[%(ints)d];
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  const int seed = input[gid];

  // Every lane writes a strided slice, so all %(ints)d words are touched.
  for (int i = tid; i < %(ints)d; i += %(block)d) {
    tile[i] = seed + i * 3;
  }
  __syncthreads();

  // Read words this lane did not write.  The +97 wrap is a subtract, not a
  // modulo, so the case sizes need not be powers of two.
  int acc = 0;
  for (int i = tid; i < %(ints)d; i += %(block)d) {
    int j = i + 97;
    if (j >= %(ints)d) j -= %(ints)d;
    acc += tile[j];
  }
  __syncthreads();

  // Cross-wave read of the last barrier's data.
  tile[tid] = acc;
  __syncthreads();
  out[gid] = acc * 31 + tile[(tid + 64) & (%(block)d - 1)];
}
"""


def emit(*parts: object) -> None:
    print(*parts, flush=True)


def s32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


def sha_of(array) -> str:
    return hashlib.sha256(bytes(memoryview(array).cast("B"))).hexdigest()


def compile_kernels(cases) -> bytes:
    hip_root = os.environ["HIP_PATH"]
    rocm_root = os.environ["ROCM_PATH"]
    clang = Path(os.environ["HIP_CLANG_PATH"]) / "clang++"
    body = "\n".join(
        KERNEL_TEMPLATE % {"tag": tag, "ints": CASE_INTS[tag], "block": BLOCK}
        for tag in cases
    )
    with tempfile.TemporaryDirectory(prefix="lds-capsule.") as directory:
        temporary = Path(directory)
        source = temporary / "lds_probe.hip"
        image = temporary / "lds_probe.hsaco"
        source.write_text(body, encoding="ascii")
        command = [
            str(clang), "-x", "hip", "--offload-device-only",
            "--no-gpu-bundle-output", "--offload-arch=gfx950",
            f"--hip-path={hip_root}", f"--rocm-path={rocm_root}",
            "-O3", str(source), "-o", str(image),
        ]
        compiler_env = dict(os.environ)
        deps = "/home/zhaosiying/amdgpu-sim/env/conda/sglang-build-deps/lib"
        if Path(deps, "libxml2.so.2").exists():
            compiler_env["LD_LIBRARY_PATH"] = \
                deps + ":" + compiler_env.get("LD_LIBRARY_PATH", "")
        compiler_env.pop("LD_PRELOAD", None)
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False, env=compiler_env,
        )
        if completed.returncode != 0:
            raise SystemExit("compile failed: "
                             + completed.stderr.strip()[:2000])
        payload = image.read_bytes()
    emit(f"COMPILE ok bytes={len(payload)} "
         f"image_sha256={hashlib.sha256(payload).hexdigest()}")
    for tag in cases:
        emit(f"COMPILE case {tag} group_segment_bytes={CASE_INTS[tag] * 4}")
    return payload


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

    def h2d(self, device, host, size: int) -> None:
        self.lib.hipMemcpy(device, ctypes.cast(host, ctypes.c_void_p),
                           size, H2D)

    def d2h(self, host, device, size: int) -> None:
        self.lib.hipMemcpy(ctypes.cast(host, ctypes.c_void_p), device,
                           size, D2H)


def oracle(tag: str, source, groups: int):
    ints = CASE_INTS[tag]
    lanes = groups * BLOCK
    want = (ctypes.c_int * lanes)()
    for group in range(groups):
        base = group * BLOCK
        tile = [0] * ints
        for tid in range(BLOCK):
            seed = source[base + tid]
            for i in range(tid, ints, BLOCK):
                tile[i] = s32(seed + i * 3)
        acc = [0] * BLOCK
        for tid in range(BLOCK):
            total = 0
            for i in range(tid, ints, BLOCK):
                j = i + 97
                if j >= ints:
                    j -= ints
                total = s32(total + tile[j])
            acc[tid] = total
        for tid in range(BLOCK):
            want[base + tid] = s32(acc[tid] * 31 + acc[(tid + 64) & (BLOCK - 1)])
    return want


def run_case(hip: Hip, function, tag: str, groups: int) -> bool:
    lanes = groups * BLOCK
    lane_type = ctypes.c_int * lanes
    source = lane_type(*[((index * 13 + 1) & 0xFFFF) for index in range(lanes)])
    nbytes = ctypes.sizeof(source)

    want = oracle(tag, source, groups)
    want_sha = sha_of(want)

    device_in = hip.malloc(nbytes)
    device_out = hip.malloc(nbytes)
    hip.h2d(device_in, source, nbytes)
    hip.h2d(device_out, lane_type(*([SENTINEL] * lanes)), nbytes)

    parameters = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(device_in), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(device_out), ctypes.c_void_p),
    )
    emit(f"CASE {tag} group_segment_bytes={CASE_INTS[tag] * 4} "
         f"workgroups={groups} block={BLOCK}")
    start = time.monotonic()
    status = hip.lib.hipModuleLaunchKernel(function, groups, 1, 1, BLOCK, 1, 1,
                                           0, None, parameters, None)
    sync = hip.lib.hipStreamSynchronize(None)
    seconds = time.monotonic() - start
    emit(f"  launch_status={status} sync_status={sync}")
    emit(f"  KERNEL_WALL_S {tag} {seconds:.2f}")

    got = lane_type()
    hip.d2h(got, device_out, nbytes)
    got_sha = sha_of(got)
    untouched = sum(1 for i in range(lanes) if got[i] == SENTINEL)
    emit(f"  ORACLE {tag} want_sha256={want_sha}")
    emit(f"  ORACLE {tag} got_sha256={got_sha}")
    emit(f"  DIAG {tag} untouched={untouched}/{lanes}")
    ok = got_sha == want_sha and status == HIP_SUCCESS and sync == HIP_SUCCESS
    if not ok:
        bad = [(i, got[i], want[i]) for i in range(lanes)
               if got[i] != want[i]][:4]
        emit(f"  DIAG {tag} first_bad={bad}")
    hip.lib.hipFree(device_in)
    hip.lib.hipFree(device_out)
    emit(f"  RESULT {tag} {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    unknown = [c for c in CASES if c not in CASE_INTS]
    if unknown:
        raise SystemExit(f"unknown case(s): {unknown}")
    image = compile_kernels(CASES)

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

    results = {}
    for tag in CASES:
        function = ctypes.c_void_p()
        status = hip.lib.hipModuleGetFunction(
            ctypes.byref(function), module,
            f"lds_probe_{tag}".encode("ascii"))
        if status != HIP_SUCCESS:
            raise SystemExit(f"hipModuleGetFunction({tag}) failed: {status}")
        results[tag] = run_case(hip, function, tag, GROUPS)

    ok = all(results.values())
    emit("SUMMARY " + " ".join(f"{k}={'PASS' if v else 'FAIL'}"
                               for k, v in results.items()))
    emit(f"CAPSULE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
