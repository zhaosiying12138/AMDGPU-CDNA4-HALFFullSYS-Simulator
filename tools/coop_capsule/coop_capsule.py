#!/usr/bin/env python3
"""Public-HIP capsule for cross-workgroup cooperation (compute-unit count).

Why this exists
---------------
The topology this stack publishes advertises a full gfx950: ``simd_count
1024`` over ``simd_per_cu 4``, i.e. 256 compute units, each with
``max_waves_per_simd 8``.  gem5 built exactly one compute unit.

One compute unit is not a small gfx950, it is a different machine.  A kernel
whose forward progress needs several workgroups running *at the same time*
cannot complete on it: the workgroup slots fill with consumers, the producer
they are all waiting for is never scheduled, and nothing that could release a
slot can happen.  gem5 then advances simulated time at full host CPU with
scalar instructions retiring forever and not one wavefront completing, while
the host waits on a completion signal that can never arrive.  That is exactly
what stopped the unchanged-upstream SGLang TP1 lane at retired dispatch 883,
inside AITER's split-K GEMM, with twelve wavefronts resident -- three
workgroups of four -- every one of them polling
``s_load_dword s75 / s_cmp_eq_u32 s75, 1 / s_cbranch_scc0``.

The ``coop`` cases reproduce that shape in a few seconds:

  * Every workgroup statically allocates 96 KiB of group segment.  A compute
    unit provides 160 KiB, so **exactly one workgroup fits per compute unit**
    and the number of co-resident workgroups equals the number of compute
    units.  The threshold is therefore sharp and does not depend on register
    pressure or wave-slot accounting.
  * The *last* workgroup is the producer: it publishes a payload and then a
    ready flag.  Every other workgroup spins on the ready flag, unbounded.
  * With fewer compute units than workgroups, the producer is never
    dispatched -- gem5 dispatches workgroups in index order -- and the run
    hangs.  With at least as many, it completes and every workgroup's output
    depends on the producer's payload, so a lost handshake is a wrong answer
    rather than a silent pass.

The ``cost`` case is the counterweight the compute-unit count has to be
justified against: a plain data-parallel kernel with many workgroups and no
cooperation at all, which runs at any compute-unit count.  Comparing its
``KERNEL_WALL_S`` across ``GEMSIM_NUM_COMPUTE_UNITS`` is the simulation-cost
delta of raising the count.

Only ordinary loads and stores cross workgroups.  No atomic read-modify-write
is used: the host-native compute unit rejects one
(``unsupported host-native vector command SwapReq``), and that would confound
the result being measured here.

Environment:
  CAPSULE_COOP_CASES    comma-separated subset of ``coop2,coop4,coop8,cost``
                        (default ``coop2,coop4,cost``)
  CAPSULE_COST_GROUPS   workgroups in the ``cost`` case (default 8)
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
# 24576 ints = 98304 bytes = 96 KiB.  A compute unit provides 160 KiB, so one
# workgroup fits and two do not.  This is the same allocation the group-segment
# capsule already proves dispatchable (its "96k" case).
COOP_LDS_INTS = 24576
# The cost case must not be group-segment bound, or it would measure LDS
# residency instead of compute-unit throughput.  8 KiB leaves the wave-slot
# limit (4 SIMDs x 8 slots / 4 waves per workgroup = 8 workgroups) in charge.
COST_LDS_INTS = 2048

PRODUCER_PAYLOAD = 0x5A17C0DE - (1 << 32)  # signed int

CASES = [
    case
    for case in os.environ.get(
        "CAPSULE_COOP_CASES", "coop2,coop4,cost"
    ).split(",")
    if case
]
COST_GROUPS = int(os.environ.get("CAPSULE_COST_GROUPS", "8"))

# name -> (kernel tag, workgroups)
CASE_GROUPS = {
    "coop2": 2,
    "coop4": 4,
    "coop8": 8,
    "coop16": 16,
    "cost": COST_GROUPS,
}

KERNEL_SOURCE = r"""
#include <hip/hip_runtime.h>

// ---------------------------------------------------------------- coop -----
// 96 KiB of static group segment: one workgroup per compute unit, so the
// co-resident workgroup count is exactly the compute-unit count.
//
// flag[0] is the ready word, flag[1] the payload.  Both are read and written
// through volatile pointers so the compiler cannot hoist the poll out of the
// loop or sink the publication past it.  Ordinary loads and stores only: the
// host-native compute unit does not implement an atomic swap, and the point of
// this capsule is the workgroup residency, not the atomic path.
extern "C" __global__ void coop_probe(const int *input, int *out,
                                      int *flag) {
  __shared__ int tile[%(coop_ints)d];
  volatile int *ready = flag;
  volatile int *payload = flag + 1;

  const int tid = threadIdx.x;
  const int wg = blockIdx.x;
  const int gid = wg * blockDim.x + tid;
  const int seed = input[gid];

  // Fill the whole group segment, so the allocation is genuinely used and a
  // truncated one is a wrong answer.
  for (int i = tid; i < %(coop_ints)d; i += %(block)d) {
    tile[i] = seed + i * 3;
  }
  __syncthreads();

  const int producer = gridDim.x - 1;
  if (wg == producer) {
    if (tid == 0) {
      *payload = %(payload)d;
      __threadfence();
      *ready = 1;
    }
  } else {
    if (tid == 0) {
      // Unbounded.  If the producer workgroup is never scheduled this never
      // returns -- which is the failure being reproduced.
      while (*ready != 1) {
      }
      tile[0] = *payload;
    }
  }
  __syncthreads();

  // Every lane's result depends on the producer's payload, so a handshake that
  // silently did not happen cannot look like a pass.
  const int published = (wg == producer) ? %(payload)d : tile[0];
  int acc = 0;
  for (int i = tid; i < %(coop_ints)d; i += %(block)d) {
    int j = i + 97;
    if (j >= %(coop_ints)d) j -= %(coop_ints)d;
    acc += tile[j];
  }
  out[gid] = acc * 31 + published + wg;
}

// ---------------------------------------------------------------- cost -----
// No cooperation at all: pure data-parallel work whose wall-clock cost is
// comparable across compute-unit counts.
extern "C" __global__ void cost_probe(const int *input, int *out) {
  __shared__ int tile[%(cost_ints)d];
  const int tid = threadIdx.x;
  const int wg = blockIdx.x;
  const int gid = wg * blockDim.x + tid;
  const int seed = input[gid];

  for (int i = tid; i < %(cost_ints)d; i += %(block)d) {
    tile[i] = seed + i * 3;
  }
  __syncthreads();

  int acc = 0;
  for (int round = 0; round < 8; ++round) {
    for (int i = tid; i < %(cost_ints)d; i += %(block)d) {
      int j = i + 97 + round;
      if (j >= %(cost_ints)d) j -= %(cost_ints)d;
      acc += tile[j] ^ round;
    }
  }
  out[gid] = acc * 31 + wg;
}
"""


def emit(*parts: object) -> None:
    print(*parts, flush=True)


def s32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


def sha_of(array) -> str:
    return hashlib.sha256(bytes(memoryview(array).cast("B"))).hexdigest()


def compile_kernels() -> bytes:
    hip_root = os.environ["HIP_PATH"]
    rocm_root = os.environ["ROCM_PATH"]
    clang = Path(os.environ["HIP_CLANG_PATH"]) / "clang++"
    body = KERNEL_SOURCE % {
        "coop_ints": COOP_LDS_INTS,
        "cost_ints": COST_LDS_INTS,
        "block": BLOCK,
        "payload": PRODUCER_PAYLOAD,
    }
    with tempfile.TemporaryDirectory(prefix="coop-capsule.") as directory:
        temporary = Path(directory)
        source = temporary / "coop_probe.hip"
        image = temporary / "coop_probe.hsaco"
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
    emit(f"COMPILE coop group_segment_bytes={COOP_LDS_INTS * 4} "
         f"cost group_segment_bytes={COST_LDS_INTS * 4}")
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


def coop_oracle(source, groups: int):
    ints = COOP_LDS_INTS
    lanes = groups * BLOCK
    want = (ctypes.c_int * lanes)()
    producer = groups - 1
    for group in range(groups):
        base = group * BLOCK
        tile = [0] * ints
        for tid in range(BLOCK):
            seed = source[base + tid]
            for i in range(tid, ints, BLOCK):
                tile[i] = s32(seed + i * 3)
        if group != producer:
            tile[0] = PRODUCER_PAYLOAD
        published = PRODUCER_PAYLOAD
        for tid in range(BLOCK):
            total = 0
            for i in range(tid, ints, BLOCK):
                j = i + 97
                if j >= ints:
                    j -= ints
                total = s32(total + tile[j])
            want[base + tid] = s32(total * 31 + published + group)
    return want


def cost_oracle(source, groups: int):
    ints = COST_LDS_INTS
    lanes = groups * BLOCK
    want = (ctypes.c_int * lanes)()
    for group in range(groups):
        base = group * BLOCK
        tile = [0] * ints
        for tid in range(BLOCK):
            seed = source[base + tid]
            for i in range(tid, ints, BLOCK):
                tile[i] = s32(seed + i * 3)
        for tid in range(BLOCK):
            total = 0
            for round_ in range(8):
                for i in range(tid, ints, BLOCK):
                    j = i + 97 + round_
                    if j >= ints:
                        j -= ints
                    total = s32(total + (tile[j] ^ round_))
            want[base + tid] = s32(total * 31 + group)
    return want


def run_case(hip: Hip, module, case: str) -> bool:
    groups = CASE_GROUPS[case]
    cooperative = case != "cost"
    name = b"coop_probe" if cooperative else b"cost_probe"

    function = ctypes.c_void_p()
    status = hip.lib.hipModuleGetFunction(ctypes.byref(function), module, name)
    if status != HIP_SUCCESS:
        raise SystemExit(f"hipModuleGetFunction({name!r}) failed: {status}")

    lanes = groups * BLOCK
    lane_type = ctypes.c_int * lanes
    source = lane_type(*[((index * 13 + 1) & 0xFFFF) for index in range(lanes)])
    nbytes = ctypes.sizeof(source)

    want = coop_oracle(source, groups) if cooperative \
        else cost_oracle(source, groups)
    want_sha = sha_of(want)

    device_in = hip.malloc(nbytes)
    device_out = hip.malloc(nbytes)
    hip.h2d(device_in, source, nbytes)
    hip.h2d(device_out, lane_type(*([SENTINEL] * lanes)), nbytes)

    arguments = [
        ctypes.cast(ctypes.byref(device_in), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(device_out), ctypes.c_void_p),
    ]
    device_flag = None
    if cooperative:
        device_flag = hip.malloc(8)
        hip.h2d(device_flag, (ctypes.c_int * 2)(0, 0), 8)
        arguments.append(
            ctypes.cast(ctypes.byref(device_flag), ctypes.c_void_p))
    parameters = (ctypes.c_void_p * len(arguments))(*arguments)

    segment = (COOP_LDS_INTS if cooperative else COST_LDS_INTS) * 4
    emit(f"CASE {case} kind={'cooperative' if cooperative else 'independent'} "
         f"workgroups={groups} block={BLOCK} group_segment_bytes={segment} "
         f"producer_wg={groups - 1 if cooperative else -1}")
    start = time.monotonic()
    status = hip.lib.hipModuleLaunchKernel(function, groups, 1, 1, BLOCK, 1, 1,
                                           0, None, parameters, None)
    sync = hip.lib.hipStreamSynchronize(None)
    seconds = time.monotonic() - start
    emit(f"  launch_status={status} sync_status={sync}")
    emit(f"  KERNEL_WALL_S {case} {seconds:.2f}")

    got = lane_type()
    hip.d2h(got, device_out, nbytes)
    got_sha = sha_of(got)
    untouched = sum(1 for i in range(lanes) if got[i] == SENTINEL)
    emit(f"  ORACLE {case} want_sha256={want_sha}")
    emit(f"  ORACLE {case} got_sha256={got_sha}")
    emit(f"  DIAG {case} untouched={untouched}/{lanes}")
    ok = got_sha == want_sha and status == HIP_SUCCESS and sync == HIP_SUCCESS
    if not ok:
        bad = [(i, got[i], want[i]) for i in range(lanes)
               if got[i] != want[i]][:4]
        emit(f"  DIAG {case} first_bad={bad}")
    hip.lib.hipFree(device_in)
    hip.lib.hipFree(device_out)
    if device_flag is not None:
        hip.lib.hipFree(device_flag)
    emit(f"  RESULT {case} {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    unknown = [case for case in CASES if case not in CASE_GROUPS]
    if unknown:
        raise SystemExit(f"unknown case(s): {unknown}")
    image = compile_kernels()

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

    results = {case: run_case(hip, module, case) for case in CASES}

    ok = all(results.values())
    emit("SUMMARY " + " ".join(f"{k}={'PASS' if v else 'FAIL'}"
                               for k, v in results.items()))
    emit(f"CAPSULE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
