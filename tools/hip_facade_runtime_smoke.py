#!/usr/bin/env python3
"""Exercise the standard upstream HIP ABI over the runtime-gem5 bridge.

The program deliberately knows nothing about the self-runtime ABI or gem5
protocol.  It loads libamdhip64, compiles an ordinary HIP translation unit,
and uses the public HIP runtime/module APIs.  This is the common prerequisite
for Triton, PyTorch, vLLM, and SGLang rather than a framework-specific shim.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


SCHEMA = "amdgpu-sim.hip-facade-runtime-smoke.v1"
HIP_SUCCESS = 0
HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2
HIP_MEMCPY_DEVICE_TO_DEVICE = 3
KERNEL_SOURCE = r"""
#include <hip/hip_runtime.h>

extern "C" __global__ void facade_vector_add(
    const float *left, const float *right, float *output, unsigned int count) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) output[index] = left[index] + right[index];
}
"""


class HipSmokeError(RuntimeError):
    pass


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_absolute_environment(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw or not raw.startswith("/"):
        raise HipSmokeError(f"{name} must be an absolute path")
    return Path(raw).resolve(strict=True)


def load_hip(hip_root: Path, library_override: Path | None = None) -> tuple[ctypes.CDLL, Path]:
    library_path = (
        library_override.resolve(strict=True)
        if library_override is not None
        else (hip_root / "lib/libamdhip64.so.7").resolve(strict=True)
    )
    if not library_path.is_file():
        raise HipSmokeError(f"HIP runtime library is not a regular file: {library_path}")
    try:
        library = ctypes.CDLL(str(library_path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    except OSError as error:
        raise HipSmokeError(f"could not load private HIP runtime: {error}") from error

    library.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.hipGetDeviceCount.restype = ctypes.c_int
    library.hipSetDevice.argtypes = [ctypes.c_int]
    library.hipSetDevice.restype = ctypes.c_int
    library.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    library.hipMalloc.restype = ctypes.c_int
    library.hipFree.argtypes = [ctypes.c_void_p]
    library.hipFree.restype = ctypes.c_int
    library.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    library.hipMemcpy.restype = ctypes.c_int
    library.hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.hipStreamCreate.restype = ctypes.c_int
    library.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
    library.hipStreamSynchronize.restype = ctypes.c_int
    library.hipStreamDestroy.argtypes = [ctypes.c_void_p]
    library.hipStreamDestroy.restype = ctypes.c_int
    library.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    library.hipModuleLoadData.restype = ctypes.c_int
    library.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p
    ]
    library.hipModuleGetFunction.restype = ctypes.c_int
    library.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.hipModuleLaunchKernel.restype = ctypes.c_int
    library.hipModuleUnload.argtypes = [ctypes.c_void_p]
    library.hipModuleUnload.restype = ctypes.c_int
    return library, library_path


def hip_check(status: int, operation: str) -> None:
    if status != HIP_SUCCESS:
        raise HipSmokeError(f"{operation} failed with hipError_t={status}")


def compile_kernel(hip_root: Path, rocm_root: Path, compiler: Path) -> tuple[bytes, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="hip-facade-smoke.") as directory:
        temporary = Path(directory)
        source = temporary / "facade_vector_add.hip"
        image = temporary / "facade_vector_add.hsaco"
        source.write_text(KERNEL_SOURCE, encoding="ascii")
        command = [
            str(compiler),
            "-x",
            "hip",
            "--offload-device-only",
            "--no-gpu-bundle-output",
            "--offload-arch=gfx950",
            f"--hip-path={hip_root}",
            f"--rocm-path={rocm_root}",
            "-O3",
            str(source),
            "-o",
            str(image),
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise HipSmokeError("upstream HIP kernel compilation failed: " + completed.stderr.strip())
        payload = image.read_bytes()
        if not payload.startswith(b"\x7fELF") or b"gfx950" not in payload:
            raise HipSmokeError("compiler did not produce a gfx950 ELF code object")
        return payload, {
            "compiler_path": str(compiler),
            "compiler_sha256": sha256_file(compiler),
            "source_sha256": sha256_bytes(KERNEL_SOURCE.encode("ascii")),
            "image_bytes": len(payload),
            "image_sha256": sha256_bytes(payload),
            "target": "gfx950",
            "device_only": True,
            "output_format": "elf64-amdgpu",
        }


def discover(library: ctypes.CDLL) -> int:
    count = ctypes.c_int(-1)
    hip_check(library.hipGetDeviceCount(ctypes.byref(count)), "hipGetDeviceCount")
    if count.value < 1:
        raise HipSmokeError(f"HIP reported no simulator device: {count.value}")
    hip_check(library.hipSetDevice(0), "hipSetDevice")
    return count.value


def run_vector_add(library: ctypes.CDLL, image: bytes, count: int) -> dict[str, Any]:
    host_type = ctypes.c_float * count
    left = host_type(*(float((index * 7) % 31 - 15) / 8.0 for index in range(count)))
    right = host_type(*(float((index * 11) % 29 - 14) / 16.0 for index in range(count)))
    output = host_type(*([0.0] * count))
    byte_count = ctypes.sizeof(left)
    device_left = ctypes.c_void_p()
    device_right = ctypes.c_void_p()
    device_output = ctypes.c_void_p()
    stream = ctypes.c_void_p()
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    allocated: list[ctypes.c_void_p] = []
    try:
        hip_check(library.hipStreamCreate(ctypes.byref(stream)), "hipStreamCreate")
        for name, pointer in (
            ("left", device_left),
            ("right", device_right),
            ("output", device_output),
        ):
            hip_check(library.hipMalloc(ctypes.byref(pointer), byte_count), f"hipMalloc({name})")
            allocated.append(pointer)
        hip_check(
            library.hipMemcpy(device_left, ctypes.cast(left, ctypes.c_void_p), byte_count, HIP_MEMCPY_HOST_TO_DEVICE),
            "hipMemcpy(left,H2D)",
        )
        hip_check(
            library.hipMemcpy(device_right, ctypes.cast(right, ctypes.c_void_p), byte_count, HIP_MEMCPY_HOST_TO_DEVICE),
            "hipMemcpy(right,H2D)",
        )
        image_buffer = ctypes.create_string_buffer(image)
        hip_check(
            library.hipModuleLoadData(ctypes.byref(module), ctypes.cast(image_buffer, ctypes.c_void_p)),
            "hipModuleLoadData",
        )
        hip_check(
            library.hipModuleGetFunction(ctypes.byref(function), module, b"facade_vector_add"),
            "hipModuleGetFunction",
        )
        count_value = ctypes.c_uint(count)
        parameters = (ctypes.c_void_p * 4)(
            ctypes.cast(ctypes.byref(device_left), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(device_right), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(device_output), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(count_value), ctypes.c_void_p),
        )
        block = 64
        grid = (count + block - 1) // block
        hip_check(
            library.hipModuleLaunchKernel(
                function, grid, 1, 1, block, 1, 1, 0, stream, parameters, None
            ),
            "hipModuleLaunchKernel",
        )
        hip_check(library.hipStreamSynchronize(stream), "hipStreamSynchronize")
        hip_check(
            library.hipMemcpy(
                ctypes.cast(output, ctypes.c_void_p),
                device_output,
                byte_count,
                HIP_MEMCPY_DEVICE_TO_HOST,
            ),
            "hipMemcpy(output,D2H)",
        )
        expected = [float(left[index] + right[index]) for index in range(count)]
        actual = [float(output[index]) for index in range(count)]
        mismatches = sum(actual[index] != expected[index] for index in range(count))
        return {
            "count": count,
            "grid": [grid, 1, 1],
            "block": [block, 1, 1],
            "mismatch_count": mismatches,
            "output_sha256": sha256_bytes(bytes(output)),
            "correct": mismatches == 0,
        }
    finally:
        if module.value:
            library.hipModuleUnload(module)
        for pointer in reversed(allocated):
            library.hipFree(pointer)
        if stream.value:
            library.hipStreamDestroy(stream)


def run_copy_roundtrip(library: ctypes.CDLL, byte_count: int) -> dict[str, Any]:
    host_type = ctypes.c_ubyte * byte_count
    source = host_type(*((index ^ (index >> 8) ^ 0x5A) & 0xFF for index in range(byte_count)))
    output = host_type(*([0] * byte_count))
    device_source = ctypes.c_void_p()
    device_destination = ctypes.c_void_p()
    allocated: list[ctypes.c_void_p] = []
    timings: dict[str, float] = {}

    def timed_copy(name: str, destination: ctypes.c_void_p, origin: ctypes.c_void_p, kind: int) -> None:
        started = time.perf_counter()
        hip_check(library.hipMemcpy(destination, origin, byte_count, kind), name)
        timings[name] = time.perf_counter() - started

    try:
        for name, pointer in (
            ("source", device_source),
            ("destination", device_destination),
        ):
            hip_check(library.hipMalloc(ctypes.byref(pointer), byte_count), f"hipMalloc({name})")
            allocated.append(pointer)
        timed_copy(
            "h2d",
            device_source,
            ctypes.cast(source, ctypes.c_void_p),
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
        timed_copy("d2d", device_destination, device_source, HIP_MEMCPY_DEVICE_TO_DEVICE)
        timed_copy(
            "d2h",
            ctypes.cast(output, ctypes.c_void_p),
            device_destination,
            HIP_MEMCPY_DEVICE_TO_HOST,
        )
        source_bytes = bytes(source)
        output_bytes = bytes(output)
        return {
            "bytes": byte_count,
            "correct": output_bytes == source_bytes,
            "source_sha256": sha256_bytes(source_bytes),
            "output_sha256": sha256_bytes(output_bytes),
            "wall_seconds": timings,
        }
    finally:
        for pointer in reversed(allocated):
            library.hipFree(pointer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("discover", "vector-add", "copy-roundtrip"), default="discover"
    )
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--copy-bytes", type=int, default=2 * 1024 * 1024)
    arguments = parser.parse_args(argv)
    try:
        if arguments.count < 1 or arguments.count > 1_048_576:
            raise HipSmokeError("count is outside the bounded smoke range")
        if arguments.copy_bytes < 1 or arguments.copy_bytes > 16 * 1024 * 1024:
            raise HipSmokeError("copy byte count is outside the bounded smoke range")
        hip_root = require_absolute_environment("HIP_PATH")
        rocm_root = require_absolute_environment("ROCM_PATH")
        compiler_root = require_absolute_environment("HIP_CLANG_PATH")
        if os.environ.get("HIP_PLATFORM") != "amd":
            raise HipSmokeError("HIP_PLATFORM must be amd")
        library_override_raw = os.environ.get("HIP_RUNTIME_LIBRARY")
        library_override = None
        if library_override_raw:
            if not library_override_raw.startswith("/"):
                raise HipSmokeError("HIP_RUNTIME_LIBRARY must be an absolute path")
            library_override = Path(library_override_raw)
        library, library_path = load_hip(hip_root, library_override)
        device_count = discover(library)
        compilation = None
        execution = None
        if arguments.mode == "vector-add":
            image, compilation = compile_kernel(hip_root, rocm_root, compiler_root / "clang++")
            execution = run_vector_add(library, image, arguments.count)
            if not execution["correct"]:
                raise HipSmokeError("HIP vector-add output mismatch")
        elif arguments.mode == "copy-roundtrip":
            execution = run_copy_roundtrip(library, arguments.copy_bytes)
            if not execution["correct"]:
                raise HipSmokeError("HIP copy roundtrip output mismatch")
        result = {
            "schema": SCHEMA,
            "mode": arguments.mode,
            "device_count": device_count,
            "hip_library": {"path": str(library_path), "sha256": sha256_file(library_path)},
            "compilation": compilation,
            "execution": execution,
            "path": [
                "upstream_hip_api",
                "upstream_rocr",
                "upstream_hsakmt_model_interface",
                "self_runtime_provider",
                "runtime_gem5_bridge",
                "gem5_gpu_model",
            ],
            "fallback": {"cpu": 0, "cuda": 0, "privateuse1": 0, "project_operator": 0},
            "correct": True,
        }
    except (HipSmokeError, OSError) as error:
        print(f"hip-facade-runtime-smoke: {error}", file=sys.stderr)
        return 1
    print(canonical_json(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
