#!/usr/bin/env python3
"""Verify the isolated dependency/stage contract for the upstream HIP facade.

This tool intentionally does not patch ROCr, HIP, CLR, PyTorch, Triton, vLLM,
or SGLang.  It owns only the private build prefix and records artifacts from
upstream CMake builds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from urllib.parse import urlparse


SCHEMA = "amdgpu-sim.hip-facade-build.v1"
LOCK_RELATIVE = Path("config/conda-hip-facade-build-linux-64.lock")
PREFIX_RELATIVE = Path("env/conda/hip-facade-build-deps")
CORE_STAGE_RELATIVE = Path("env/rocm/hip-facade-stage-core-v1")
CORE_BUILD_RELATIVE = Path("projects/self-amdgpu-runtime/build/upstream-rocr-facade-core-v1")
HIP_STAGE_RELATIVE = Path("env/rocm/hip-facade-stage-v1")
LLVM_TOOLS_RELATIVE = Path("env/rocm/hip-facade-llvm-tools-v1")
DEVICE_SDK_RELATIVE = Path(
    "env/rocm/gfx950-v5-llvm-73f2a21fe16b-rocm-92115a294198-"
    "sim-ed808f6a57b8-runtime-749717e1a5c7"
)
READELF = Path("/usr/bin/readelf")

FACADE_ARTIFACTS: dict[str, tuple[Path, tuple[str, ...]]] = {
    "rocr": (
        Path("lib/libhsa-runtime64.so.1"),
        ("hsa_init", "hsa_shut_down", "hsa_iterate_agents", "hsa_queue_create", "hsa_signal_create"),
    ),
    "hsakmt_model": (
        Path("lib/libself_amdgpu_hsakmt_model.so.1"),
        ("get_hsakmt_model_functions",),
    ),
    "hip": (
        Path("lib/libamdhip64.so.7"),
        (
            "hipGetDeviceCount",
            "hipGetDeviceProperties",
            "hipMalloc",
            "hipFree",
            "hipMemcpy",
            "hipStreamCreate",
            "hipStreamSynchronize",
            "hipModuleLoadData",
            "hipModuleLaunchKernel",
        ),
    ),
    "comgr": (
        Path("lib/libamd_comgr.so.3"),
        ("amd_comgr_create_data", "amd_comgr_do_action", "amd_comgr_release_data"),
    ),
    "rocm_core": (
        Path("lib/librocm-core.so.1"),
        ("getROCmVersion", "getROCmInstallPath"),
    ),
    "rccl": (
        Path("lib/librccl.so.1"),
        ("ncclGetUniqueId", "ncclCommInitRank", "ncclAllReduce", "ncclCommDestroy"),
    ),
}


class BuildEnvironmentError(RuntimeError):
    pass


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise BuildEnvironmentError(f"expected owned regular file: {path}")
    return {"path": str(path), "bytes": metadata.st_size, "sha256": sha256(path)}


def resolved_product_file(path: Path, prefix: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(prefix)
    except (OSError, ValueError) as error:
        raise BuildEnvironmentError(f"artifact escapes facade stage: {path}") from error
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise BuildEnvironmentError(f"expected owned regular facade artifact: {path}")
    return resolved


def dynamic_symbols(path: Path) -> set[str]:
    completed = subprocess.run(
        [str(READELF), "--dyn-syms", "--wide", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise BuildEnvironmentError(f"could not inspect dynamic symbols: {path}")
    symbols: set[str] = set()
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[0].rstrip(":").isdigit():
            continue
        symbols.add(fields[7].split("@", 1)[0])
    return symbols


def locked_packages(lock: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for raw in lock.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line == "@EXPLICIT":
            continue
        url, separator, digest = line.partition("#")
        if not separator or len(digest) != 64:
            raise BuildEnvironmentError("lock entries require SHA-256 fragments")
        filename = Path(urlparse(url).path).name
        suffix = ".conda" if filename.endswith(".conda") else ".tar.bz2" if filename.endswith(".tar.bz2") else ""
        if not suffix:
            raise BuildEnvironmentError(f"unsupported package artifact: {filename}")
        stem = filename[: -len(suffix)]
        try:
            name, version, build = stem.rsplit("-", 2)
        except ValueError as error:
            raise BuildEnvironmentError(f"malformed package filename: {filename}") from error
        if name in result:
            raise BuildEnvironmentError(f"duplicate package: {name}")
        result[name] = {"name": name, "version": version, "build": build,
                        "url": url, "sha256": digest, "format": suffix[1:]}
    if not result:
        raise BuildEnvironmentError("empty HIP facade build lock")
    return result


def verify_prefix(root: Path) -> dict[str, object]:
    lock_path = root / LOCK_RELATIVE
    prefix = root / PREFIX_RELATIVE
    expected = locked_packages(lock_path)
    metadata_dir = prefix / "conda-meta"
    if not metadata_dir.is_dir() or metadata_dir.is_symlink():
        raise BuildEnvironmentError(f"dependency prefix is unavailable: {prefix}")
    actual: dict[str, dict[str, str | None]] = {}
    for path in sorted(metadata_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = payload.get("name")
        if not isinstance(name, str) or name in actual:
            raise BuildEnvironmentError("invalid conda metadata")
        actual[name] = {key: payload.get(key) for key in ("name", "version", "build", "url", "sha256")}
    for name, record in expected.items():
        observed = actual.get(name)
        if observed is None or any(observed.get(key) != record.get(key) for key in ("name", "version", "build", "url", "sha256")):
            raise BuildEnvironmentError(f"dependency differs from lock: {name}")
    if set(actual) != set(expected):
        raise BuildEnvironmentError("dependency prefix has packages outside lock")
    required = ("bin/pkg-config", "include/numa.h", "lib/libdrm.so", "lib/libdrm_amdgpu.so", "lib/libnuma.so", "lib/libelf.so")
    for relative in required:
        if not (prefix / relative).exists():
            raise BuildEnvironmentError(f"required dependency artifact is absent: {relative}")
    return {"prefix": str(prefix), "lock": file_record(lock_path), "packages": [expected[name] for name in sorted(expected)]}


def verify_core_stage(root: Path) -> dict[str, object]:
    stage = root / CORE_STAGE_RELATIVE
    artifacts = {}
    for relative in ("lib/libhsa-runtime64.so", "lib/libhsa-runtime64.so.1", "include/hsa/hsa.h", "include/hsakmt/hsakmt.h"):
        path = stage / relative
        if not path.exists():
            raise BuildEnvironmentError(f"ROCr core artifact is absent: {relative}")
        artifacts[relative] = file_record(path.resolve() if path.is_symlink() else path)
    return {"prefix": str(stage), "build": str(root / CORE_BUILD_RELATIVE), "artifacts": artifacts}


def verify_hip_compile_contract(root: Path) -> dict[str, object]:
    """Verify the two-root HIP driver contract used by upstream components.

    The facade stage owns the public HIP API and runtime libraries.  The
    content-addressed device SDK owns clang and AMDGPU bitcode.  Both roots
    must be explicit: passing only the device SDK makes clang treat it as an
    incomplete HIP installation and omits the HIP runtime wrapper.
    """
    hip_stage = root / HIP_STAGE_RELATIVE
    device_sdk = root / DEVICE_SDK_RELATIVE
    compiler = root / LLVM_TOOLS_RELATIVE / "bin/clang++"
    required = (
        hip_stage / "include/hip/hip_runtime.h",
        hip_stage / "lib/libamdhip64.so",
        device_sdk / "amdgcn/bitcode/ocml.bc",
        device_sdk / "amdgcn/bitcode/oclc_isa_version_950.bc",
        compiler,
    )
    for path in required:
        if not path.exists():
            raise BuildEnvironmentError(f"HIP compile-contract artifact is absent: {path}")
    source = """
#include <hip/hip_runtime.h>
__global__ void facade_probe(unsigned long *out) {
  unsigned long a = 7, b = 11;
  out[0] = min(a, b);
}
"""
    command = [
        str(compiler),
        "-x", "hip",
        "--offload-arch=gfx950",
        f"--hip-path={hip_stage}",
        f"--rocm-path={device_sdk}",
        "-fsyntax-only",
        "-",
    ]
    completed = subprocess.run(
        command,
        input=source,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise BuildEnvironmentError(
            "two-root HIP gfx950 compile probe failed: " + completed.stderr.strip()
        )
    dry_run = subprocess.run(
        command[:-2] + ["-###", "-c", "/dev/null"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    trace = dry_run.stdout + dry_run.stderr
    for token in ('"-include" "__clang_hip_runtime_wrapper.h"', '"-target-cpu" "gfx950"'):
        if token not in trace:
            raise BuildEnvironmentError(f"HIP driver trace lacks required binding: {token}")
    return {
        "hip_api_root": str(hip_stage),
        "device_sdk_root": str(device_sdk),
        "compiler": file_record(compiler.resolve()),
        "target": "gfx950",
        "runtime_wrapper_injected": True,
    }


def facade_environment(root: Path) -> dict[str, object]:
    """Return the one clean environment contract consumed by all frameworks."""
    hip_stage = root / HIP_STAGE_RELATIVE
    device_sdk = root / DEVICE_SDK_RELATIVE
    llvm_bin = root / LLVM_TOOLS_RELATIVE / "bin"
    dependency_prefix = root / PREFIX_RELATIVE
    return {
        "set": {
            "HIP_PLATFORM": "amd",
            "HIP_PATH": str(hip_stage),
            "ROCM_PATH": str(device_sdk),
            "HIP_CLANG_PATH": str(llvm_bin),
            "CMAKE_PREFIX_PATH": f"{hip_stage}:{dependency_prefix}",
            "PKG_CONFIG_PATH": f"{hip_stage / 'lib/pkgconfig'}:{dependency_prefix / 'lib/pkgconfig'}",
            "LD_LIBRARY_PATH": f"{hip_stage / 'lib'}:{dependency_prefix / 'lib'}",
        },
        "unset": [
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDACXX",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "ROCM_HOME",
        ],
        "policy": {
            "upstream_amd_api": True,
            "system_rocm_fallback": False,
            "system_cuda_inheritance": False,
            "framework_specific_bridge": False,
        },
    }


def verify_facade_stage(root: Path) -> dict[str, object]:
    """Verify the framework-neutral upstream AMD API stage without executing it."""
    stage = (root / HIP_STAGE_RELATIVE).resolve(strict=True)
    compile_contract = verify_hip_compile_contract(root)
    artifacts: dict[str, object] = {}
    for family, (relative, required_symbols) in FACADE_ARTIFACTS.items():
        public_path = stage / relative
        resolved = resolved_product_file(public_path, stage)
        symbols = dynamic_symbols(resolved)
        missing = sorted(set(required_symbols) - symbols)
        if missing:
            raise BuildEnvironmentError(
                f"facade artifact lacks required symbols ({family}): {', '.join(missing)}"
            )
        record = file_record(resolved)
        record.update(
            {
                "public_path": str(public_path),
                "required_symbols": list(required_symbols),
            }
        )
        artifacts[family] = record

    required_headers = (
        Path("include/hsa/hsa.h"),
        Path("include/hsakmt/hsakmtmodeliface.h"),
        Path("include/hip/hip_runtime.h"),
        Path("include/amd_comgr/amd_comgr.h"),
        Path("include/nccl.h"),
    )
    headers = {}
    for relative in required_headers:
        path = resolved_product_file(stage / relative, stage)
        headers[str(relative)] = file_record(path)

    versions = {
        "rocm": (stage / ".info/version").read_text(encoding="ascii").strip(),
        "hip": (stage / "share/hip/version").read_text(encoding="ascii").strip(),
    }
    if not all(versions.values()):
        raise BuildEnvironmentError("facade version metadata is empty")
    return {
        "prefix": str(stage),
        "target": "gfx950",
        "artifacts": artifacts,
        "headers": headers,
        "versions": versions,
        "compile_contract": compile_contract,
        "environment": facade_environment(root),
        "upstream_sources_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("verify-prefix", "verify-core", "verify-hip-compile", "verify-facade"),
    )
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.resolve(strict=True)
    try:
        if arguments.mode == "verify-prefix":
            result = verify_prefix(root)
        elif arguments.mode == "verify-core":
            result = verify_core_stage(root)
        elif arguments.mode == "verify-hip-compile":
            result = verify_hip_compile_contract(root)
        else:
            result = verify_facade_stage(root)
    except (BuildEnvironmentError, OSError, json.JSONDecodeError) as error:
        print(f"hip-facade-build: {error}", file=sys.stderr)
        return 1
    print(canonical_json({"schema": SCHEMA, "mode": arguments.mode, "result": result}).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
