#!/usr/bin/env python3
"""Build and publish the small, content-addressed simulator product overlay."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import sys
from typing import Any


PRODUCT_SCHEMA = "amdgpu-sim.product-prefix.v1"
IDENTITY_SCHEMA = "amdgpu-sim.product-identity.v1"
SOURCE_LOCK_SCHEMA = "amdgpu-sim.product-source-lock.v1"
ACTIVE_SCHEMA = "amdgpu-sim.active-product.v1"
FROZEN_SCHEMA = "amdgpu-sim.frozen-product.v1"
BASE_SCHEMA = "amdgpu-sim.rocm-prefix.v8"
BASE_SETUP_SCHEMA = 8
PRODUCT_SETUP_SCHEMA = 1
MAX_CONTROL_BYTES = 64 * 1024 * 1024
PRODUCT_NAME_PREFIX = "product-v1-"
DIRECTORY_SOURCE_SET_SCHEMA = "amdgpu-sim.directory-source-set.v1"
FACADE_STAGE_SCHEMA = "amdgpu-sim.facade-stage-source-set.v1"
FACADE_STAGE_RELATIVE = Path("env/rocm/hip-facade-stage-v1")
FACADE_BUILD_LOCK_RELATIVE = Path("config/conda-hip-facade-build-linux-64.lock")
FACADE_BUILD_DEPS_RELATIVE = Path("env/conda/hip-facade-build-deps")
ROCM_SYSTEMS_RELATIVE = Path("projects/rocm-systems")
ROCR_AQL_EVIDENCE_RELATIVE = Path("artifacts/evidence/upstream-rocr-aql-v2-accepted")
TOPOLOGY_TOOL_RELATIVE = Path(
    "projects/self-amdgpu-runtime/tools/hsakmt-model-topology.py"
)
PRODUCT_BUILD_CACHE_RELATIVE = Path("env/pb")

# Upstream libhsakmt formats topology paths into 256-byte arrays.  Keep enough
# room for the longest path it can construct without changing upstream code.
HSAKMT_TOPOLOGY_PATH_BUFFER_BYTES = 256
HSAKMT_TOPOLOGY_LONGEST_RELATIVE_PATH = (
    "/nodes/2147483647/p2p_links/2147483647/properties"
)

# The facade stage is a standard upstream ROCm installation plus the model
# provider.  Project-owned runtime/OpenCL files are rebuilt by _build_runtime;
# absolute LLVM links and the unused Windows shim tree are intentionally not
# part of the Linux product artifact set.
FACADE_STAGE_EXCLUDED_PREFIXES = (
    Path("llvm"),
    Path("hip"),
    Path("include/CL"),
    Path("include/self_amdgpu_runtime"),
    Path("lib/cmake/SelfAmdgpuRuntime"),
    Path("share/self-amdgpu-runtime"),
)

PLUGIN_SOURCES = {
    "triton-gemsim-amd": Path("plugins/triton/gemsim_amd"),
    "gemsim-vllm": Path("plugins/framework/gemsim_vllm"),
    "gemsim-ccl": Path("plugins/collectives/gemsim_ccl"),
}

BASE_CRITICAL_ARTIFACTS = (
    "tool_clang",
    "tool_clangxx",
    "tool_ld_lld",
    "triton_python",
    "triton_library",
    "triton_driver",
    "triton_compiler",
    "triton_llvm_config",
)

BUILD_CONTRACT: dict[str, Any] = {
    "schema": "amdgpu-sim.product-runtime-build.v1",
    "generator": "Ninja",
    "build_type": "RelWithDebInfo",
    "c_compiler": "/usr/lib/ccache/clang",
    "cxx_compiler": "/usr/lib/ccache/clang++",
    "linker": "/usr/bin/ld.lld",
    "linker_flag": "-fuse-ld=lld",
    "parallel_jobs": 24,
    "ccache": True,
    "build_shared_libs": True,
    "build_tests": True,
    "build_tools": True,
    "build_opencl": True,
    "runtime_source": "frozen-source-set-snapshot",
    "install_layout": ["bin", "include", "lib", "libexec", "share"],
    "gem5_ownership": "workspace-reference",
    "base_ownership": "read-only-reference",
}


class ProductEnvironmentError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _runtime_build_root(root: Path, runtime_source_sha: str) -> Path:
    if (
        len(runtime_source_sha) != 64
        or runtime_source_sha != runtime_source_sha.lower()
        or any(character not in "0123456789abcdef" for character in runtime_source_sha)
    ):
        raise ProductEnvironmentError("runtime source-set digest is invalid")
    return root / PRODUCT_BUILD_CACHE_RELATIVE / runtime_source_sha


def _validate_hsakmt_topology_build_path(
    build_dir: Path, topology_tool_sha256: str
) -> Path:
    if (
        len(topology_tool_sha256) != 64
        or topology_tool_sha256 != topology_tool_sha256.lower()
        or any(character not in "0123456789abcdef" for character in topology_tool_sha256)
    ):
        raise ProductEnvironmentError("HSAKMT topology tool digest is invalid")
    topology_directory = (
        build_dir
        / "tests"
        / f"hsakmt-model-topology-{topology_tool_sha256}"
    )
    longest_path = os.fsencode(
        f"{topology_directory}{HSAKMT_TOPOLOGY_LONGEST_RELATIVE_PATH}"
    )
    if len(longest_path) >= HSAKMT_TOPOLOGY_PATH_BUFFER_BYTES:
        raise ProductEnvironmentError(
            "runtime build path exceeds upstream libhsakmt topology path capacity"
        )
    return topology_directory


def _read_owned_file(
    path: Path,
    *,
    maximum: int = MAX_CONTROL_BYTES,
    private: bool,
    label: str,
) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProductEnvironmentError(f"{label} could not be opened safely: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        forbidden_mode = stat.S_IRWXG | stat.S_IRWXO if private else stat.S_IWGRP | stat.S_IWOTH
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & forbidden_mode
            or metadata.st_size <= 0
            or metadata.st_size > maximum
        ):
            raise ProductEnvironmentError(f"{label} is not a safe owned regular file: {path}")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ProductEnvironmentError(f"{label} was truncated while reading: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ProductEnvironmentError(f"{label} changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_json(
    path: Path,
    *,
    canonical: bool,
    private: bool,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    payload = _read_owned_file(path, private=private, label=label)
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductEnvironmentError(f"{label} is not valid ASCII JSON: {path}") from error
    if not isinstance(document, dict):
        raise ProductEnvironmentError(f"{label} is not a JSON object: {path}")
    if canonical and payload != canonical_json(document):
        raise ProductEnvironmentError(f"{label} is not canonical JSON: {path}")
    return document, payload


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise ProductEnvironmentError("short write while publishing control record")
        offset += written


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ProductEnvironmentError("atomic RENAME_NOREPLACE is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), destination)
        raise OSError(error_number, os.strerror(error_number), destination)


def _atomic_write(path: Path, payload: bytes, *, mode: int, replace: bool) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or parent.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise ProductEnvironmentError(f"control directory is unsafe: {path.parent}")
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        _write_all(descriptor, payload)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        if replace:
            os.replace(temporary, path)
        else:
            _rename_noreplace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _regular_sha256(path: Path, *, allowed_symlink_root: Path | None = None) -> dict[str, Any]:
    original = path
    before = path.lstat()
    symlink_target: str | None = None
    if stat.S_ISLNK(before.st_mode):
        if allowed_symlink_root is None:
            raise ProductEnvironmentError(f"symlink artifact is forbidden: {path}")
        symlink_target = os.readlink(path)
        path = path.resolve(strict=True)
        if not _is_within(path, allowed_symlink_root):
            raise ProductEnvironmentError(f"artifact symlink escapes its exact base root: {original}")
        before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
        raise ProductEnvironmentError(f"artifact is not an owned regular file: {original}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
            or opened.st_mtime_ns != before.st_mtime_ns
        ):
            raise ProductEnvironmentError(f"artifact changed before hashing: {original}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
        ):
            raise ProductEnvironmentError(f"artifact changed while hashing: {original}")
    finally:
        os.close(descriptor)
    result: dict[str, Any] = {
        "path": str(original),
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }
    if symlink_target is not None:
        result["symlink_target"] = symlink_target
        result["resolved_path"] = str(path)
    return result


def _load_source_set_module() -> Any:
    path = Path(__file__).with_name("repository_source_set.py")
    specification = importlib.util.spec_from_file_location(
        "amdgpu_sim_repository_source_set", path
    )
    if specification is None or specification.loader is None:
        raise ProductEnvironmentError("repository source-set helper is unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def repository_source_set(
    repository: Path, *, allow_gitlinks: bool = False
) -> dict[str, Any]:
    try:
        return _load_source_set_module().source_set(
            repository, allow_gitlinks=allow_gitlinks
        )
    except Exception as error:
        raise ProductEnvironmentError(f"could not freeze repository source set: {repository}") from error


def directory_source_set(directory: Path) -> dict[str, Any]:
    original = directory
    metadata = original.lstat()
    directory = original.resolve(strict=True)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or not directory.is_dir()
    ):
        raise ProductEnvironmentError(f"plugin source is not a directory: {directory}")
    records: list[dict[str, Any]] = []
    for current, directory_names, file_names in os.walk(
        directory, topdown=True, followlinks=False
    ):
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            if name in {"__pycache__", ".pytest_cache"}:
                continue
            child_metadata = (Path(current) / name).lstat()
            if (
                not stat.S_ISDIR(child_metadata.st_mode)
                or stat.S_ISLNK(child_metadata.st_mode)
                or child_metadata.st_uid != os.getuid()
            ):
                raise ProductEnvironmentError(
                    f"plugin source directory is unsafe: {Path(current) / name}"
                )
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = Path(current) / name
            relative = path.relative_to(directory).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ProductEnvironmentError(
                    f"plugin source symlink is forbidden: {path}"
                )
            artifact = _regular_sha256(path)
            records.append(
                {
                    "path": relative,
                    "executable": bool(metadata.st_mode & stat.S_IXUSR),
                    "bytes": artifact["bytes"],
                    "sha256": artifact["sha256"],
                }
            )
    core = {"schema": DIRECTORY_SOURCE_SET_SCHEMA, "files": records}
    return {
        "schema": DIRECTORY_SOURCE_SET_SCHEMA,
        "directory": str(directory),
        "file_count": len(records),
        "source_set_sha256": sha256_bytes(canonical_json(core)),
        "files": records,
    }


def _path_is_under(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
        return True
    except ValueError:
        return False


def _facade_stage_included(relative: Path) -> bool:
    if any(_path_is_under(relative, prefix) for prefix in FACADE_STAGE_EXCLUDED_PREFIXES):
        return False
    # These are the project-owned files rebuilt by the frozen self-runtime
    # source snapshot rather than copied from the prebuilt upstream stage.
    if relative.parts and relative.parts[0] == "bin":
        if relative.name in {"opencl-vecadd", "sagr-handshake", "sagr-triton-hsaco-probe"}:
            return False
    if relative.parts and relative.parts[0] == "lib":
        if relative.name.startswith((
            "libOpenCL.so",
            "libself_amdgpu_runtime.so",
            "libself_amdgpu_hsakmt_model.so",
        )):
            return False
    return True


def facade_stage_source_set(directory: Path) -> dict[str, Any]:
    """Freeze the relocatable standard ROCm artifact subset used by products."""
    original = directory
    metadata = original.lstat()
    directory = original.resolve(strict=True)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or not directory.is_dir()
    ):
        raise ProductEnvironmentError(f"facade stage is not a directory: {directory}")
    records: list[dict[str, Any]] = []
    for current, directory_names, file_names in os.walk(
        directory, topdown=True, followlinks=False
    ):
        retained: list[str] = []
        for name in sorted(directory_names):
            relative = (Path(current) / name).relative_to(directory)
            child = Path(current) / name
            child_metadata = child.lstat()
            if not _facade_stage_included(relative):
                continue
            if (
                not stat.S_ISDIR(child_metadata.st_mode)
                or stat.S_ISLNK(child_metadata.st_mode)
                or child_metadata.st_uid != os.getuid()
            ):
                raise ProductEnvironmentError(f"facade stage directory is unsafe: {child}")
            retained.append(name)
        directory_names[:] = retained
        for name in sorted(file_names):
            path = Path(current) / name
            relative = path.relative_to(directory)
            if not _facade_stage_included(relative) or name.endswith((".pyc", ".pyo")):
                continue
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(path)
                target_path = Path(os.path.normpath(relative.parent / target))
                if (
                    not target
                    or Path(target).is_absolute()
                    or target_path.is_absolute()
                    or ".." in target_path.parts
                    or not path.resolve(strict=False).exists()
                ):
                    raise ProductEnvironmentError(f"facade stage symlink is unsafe: {path}")
                encoded = os.fsencode(target)
                records.append(
                    {
                        "path": relative.as_posix(),
                        "kind": "symlink",
                        "target": target,
                        "bytes": len(encoded),
                        "sha256": sha256_bytes(encoded),
                    }
                )
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
                raise ProductEnvironmentError(f"facade stage file is unsafe: {path}")
            artifact = _regular_sha256(path)
            records.append(
                {
                    "path": relative.as_posix(),
                    "kind": "regular",
                    "executable": bool(metadata.st_mode & stat.S_IXUSR),
                    "bytes": artifact["bytes"],
                    "sha256": artifact["sha256"],
                }
            )
    records.sort(key=lambda value: value["path"])
    core = {"schema": FACADE_STAGE_SCHEMA, "files": records}
    return {
        "schema": FACADE_STAGE_SCHEMA,
        "directory": str(directory),
        "file_count": len(records),
        "regular_bytes": sum(
            int(record["bytes"])
            for record in records
            if record["kind"] == "regular"
        ),
        "source_set_sha256": sha256_bytes(canonical_json(core)),
        "files": records,
    }


def _facade_build_binding(root: Path) -> dict[str, Any]:
    """Record the exact offline dependency lock used by upstream ROCm builds."""
    lock = root / FACADE_BUILD_LOCK_RELATIVE
    dependency_prefix = root / FACADE_BUILD_DEPS_RELATIVE
    module = _load_hip_facade_build_module()
    try:
        verification = module.verify_prefix(root)
    except Exception as error:
        raise ProductEnvironmentError("upstream HIP facade dependency prefix is invalid") from error
    return {
        "lock": _regular_sha256(lock),
        "prefix": str(dependency_prefix),
        "verification": verification,
    }


def _load_hip_facade_build_module() -> Any:
    path = Path(__file__).with_name("hip_facade_build_environment.py")
    specification = importlib.util.spec_from_file_location(
        "amdgpu_sim_hip_facade_build_environment", path
    )
    if specification is None or specification.loader is None:
        raise ProductEnvironmentError("HIP facade build helper is unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _regular_tree_binding(path: Path) -> dict[str, Any]:
    return _regular_sha256(path.resolve(strict=True))


def _rocr_aql_binding(root: Path) -> dict[str, Any]:
    evidence = root / ROCR_AQL_EVIDENCE_RELATIVE
    result_path = evidence / "result.json"
    manifest_path = evidence / "manifest.json"
    result, result_payload = _read_json(
        result_path, canonical=True, private=False, label="ROCr AQL result"
    )
    manifest, manifest_payload = _read_json(
        manifest_path, canonical=True, private=False, label="ROCr AQL evidence manifest"
    )
    if (
        result.get("schema") != "amdgpu-sim.upstream-rocr-aql-acceptance.v1"
        or result.get("status") != "accepted"
        or result.get("standard_rocr_aql_accepted") is not True
        or result.get("output_correct") is not True
        or result.get("host_fallback_count") != 0
        or any(result.get(name) is not False for name in (
            "hip_runtime_accepted",
            "pytorch_rocm_accepted",
            "triton_accepted",
            "vllm_accepted",
            "sglang_accepted",
            "model_accepted",
        ))
        or manifest.get("complete") is not True
    ):
        raise ProductEnvironmentError("ROCr AQL prerequisite evidence is not accepted")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ProductEnvironmentError("ROCr AQL evidence artifact table is invalid")
    for relative, record in artifacts.items():
        if not isinstance(record, dict):
            raise ProductEnvironmentError("ROCr AQL evidence artifact record is invalid")
        candidate = evidence / relative
        actual = _regular_sha256(candidate)
        if actual["bytes"] != record.get("bytes") or actual["sha256"] != record.get("sha256"):
            raise ProductEnvironmentError(f"ROCr AQL evidence drifted: {relative}")
    return {
        "result": {
            "path": str(result_path),
            "bytes": len(result_payload),
            "sha256": sha256_bytes(result_payload),
        },
        "manifest": {
            "path": str(manifest_path),
            "bytes": len(manifest_payload),
            "sha256": sha256_bytes(manifest_payload),
        },
        "claim_scope": result["claim_scope"],
    }


def _rocm_root(root: Path) -> Path:
    return (root / "env/rocm").resolve()


def _validate_base_prefix(root: Path, base_prefix: Path) -> Path:
    rocm_root = _rocm_root(root)
    base_prefix = base_prefix.resolve(strict=True)
    if base_prefix.parent != rocm_root or base_prefix.name.startswith(PRODUCT_NAME_PREFIX):
        raise ProductEnvironmentError(f"base prefix is not a direct schema-8 child: {base_prefix}")
    return base_prefix


def _base_binding(root: Path, base_prefix: Path) -> dict[str, Any]:
    base_prefix = _validate_base_prefix(root, base_prefix)
    manifest_path = base_prefix / "manifest.json"
    manifest, manifest_payload = _read_json(
        manifest_path, canonical=False, private=False, label="base manifest"
    )
    if (
        manifest.get("schema") != BASE_SCHEMA
        or manifest.get("setup_schema") != BASE_SETUP_SCHEMA
        or manifest.get("prefix") != str(base_prefix)
    ):
        raise ProductEnvironmentError("base manifest is not the installed schema-8 prefix")
    components = manifest.get("components")
    if not isinstance(components, dict) or any(
        components.get(name) is not True
        for name in ("compiler", "device_libs", "runtime", "opencl", "python", "triton")
    ):
        raise ProductEnvironmentError("base schema-8 prefix is incomplete")
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, dict):
        raise ProductEnvironmentError("base manifest artifact table is invalid")
    artifacts: dict[str, dict[str, Any]] = {}
    for name in BASE_CRITICAL_ARTIFACTS:
        recorded = manifest_artifacts.get(name)
        if not isinstance(recorded, dict) or not isinstance(recorded.get("path"), str):
            raise ProductEnvironmentError(f"base manifest lacks critical artifact: {name}")
        candidate = Path(recorded["path"])
        if not candidate.is_absolute() or not _is_within(candidate, base_prefix):
            raise ProductEnvironmentError(f"base critical artifact escapes the base prefix: {name}")
        actual = _regular_sha256(candidate, allowed_symlink_root=base_prefix)
        if actual["sha256"] != recorded.get("sha256"):
            raise ProductEnvironmentError(f"base critical artifact drift: {name}")
        artifacts[name] = actual
    package_root = Path(artifacts["triton_driver"]["path"]).parent
    if package_root != Path(artifacts["triton_compiler"]["path"]).parent:
        raise ProductEnvironmentError("installed Triton plugin package is split across directories")
    return {
        "prefix": str(base_prefix),
        "manifest": {
            "path": str(manifest_path),
            "bytes": len(manifest_payload),
            "sha256": sha256_bytes(manifest_payload),
        },
        "setup_schema": BASE_SETUP_SCHEMA,
        "critical_artifacts": artifacts,
        "python": artifacts["triton_python"],
        "triton_plugin": {
            "package": {
                "root": str(package_root),
                "driver": artifacts["triton_driver"],
                "compiler": artifacts["triton_compiler"],
            },
        },
    }


def _workspace_inputs(root: Path) -> dict[str, dict[str, Any]]:
    gem5 = (root / "projects/gem5").resolve(strict=True)
    binary = gem5 / "build/VEGA_X86/gem5.opt"
    configuration = gem5 / "configs/example/gemsim/host_dispatch.py"
    return {
        "gem5_binary": _regular_sha256(binary),
        "gem5_config": _regular_sha256(configuration),
        "test_binary_search_hsaco": _regular_sha256(
            root
            / "projects/rocm-systems/projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/gfx950/binary_search_kernels.hsaco"
        ),
        "test_gpu_read_write_hsaco": _regular_sha256(
            root
            / "projects/rocm-systems/projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/gfx950/gpuReadWrite_kernels.hsaco"
        ),
        "test_triton_tutorial": _regular_sha256(
            root / "projects/triton/python/tutorials/01-vector-add.py"
        ),
        "test_vecadd_hsaco": _regular_sha256(
            root / "artifacts/evidence/CP-0017/triton-builds/vecadd-gfx950.hsaco"
        ),
    }


def _identity(
    root: Path,
    base: dict[str, Any],
    gem5_source: dict[str, Any],
    runtime_source: dict[str, Any],
    rocm_systems_source: dict[str, Any],
    plugin_sources: dict[str, dict[str, Any]],
    managed_inputs: dict[str, dict[str, Any]],
    facade: dict[str, Any],
) -> dict[str, Any]:
    source_schema = gem5_source.get("schema")
    if source_schema != runtime_source.get("schema"):
        raise ProductEnvironmentError("repository source-set schema mismatch")
    return {
        "schema": IDENTITY_SCHEMA,
        "setup": {
            "product_schema": PRODUCT_SETUP_SCHEMA,
            "base_setup_schema": BASE_SETUP_SCHEMA,
            "repository_source_set_schema": source_schema,
            "product_environment": _regular_sha256(
                Path(__file__).resolve(strict=True)
            ),
            "setup_entry": _regular_sha256(
                root / "scripts/setup_rocm_env.sh"
            ),
        },
        "base": base,
        "source_sets": {
            "gem5": gem5_source["source_set_sha256"],
            "self-amdgpu-runtime": runtime_source["source_set_sha256"],
            "rocm-systems": rocm_systems_source["source_set_sha256"],
            **{
                name: descriptor["source_set_sha256"]
                for name, descriptor in sorted(plugin_sources.items())
            },
        },
        "managed_inputs": managed_inputs,
        "facade": facade,
        "build_contract": BUILD_CONTRACT,
    }


def _product_id(identity: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(identity))


def _product_prefix(root: Path, product_id: str) -> Path:
    if len(product_id) != 64 or any(character not in "0123456789abcdef" for character in product_id):
        raise ProductEnvironmentError("product id is not a lowercase SHA-256")
    return _rocm_root(root) / f"{PRODUCT_NAME_PREFIX}{product_id}"


def _lock_path(root: Path, product_id: str) -> Path:
    return _rocm_root(root) / "product-locks" / f"{product_id}.json"


def _active_path(root: Path) -> Path:
    return _rocm_root(root) / "active-product"


def _frozen_path(root: Path) -> Path:
    return _rocm_root(root) / "frozen-product"


def freeze_product(root: Path, base_prefix: Path) -> Path:
    root = root.resolve(strict=True)
    rocm_root = _rocm_root(root)
    rocm_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    base = _base_binding(root, base_prefix)
    gem5_source = repository_source_set(root / "projects/gem5")
    runtime_source = repository_source_set(root / "projects/self-amdgpu-runtime")
    rocm_systems_source = repository_source_set(
        root / ROCM_SYSTEMS_RELATIVE, allow_gitlinks=True
    )
    plugin_sources = {
        name: directory_source_set(root / relative)
        for name, relative in PLUGIN_SOURCES.items()
    }
    managed_inputs = _workspace_inputs(root)
    facade = {
        "stage": facade_stage_source_set(root / FACADE_STAGE_RELATIVE),
        "build": _facade_build_binding(root),
        "rocm_systems": {
            "repository": str(root / ROCM_SYSTEMS_RELATIVE),
            "head": rocm_systems_source["head"],
            "tree": rocm_systems_source["tree"],
            "source_set_sha256": rocm_systems_source["source_set_sha256"],
        },
        "topology_tool": _regular_tree_binding(root / TOPOLOGY_TOOL_RELATIVE),
        "rocr_aql_prerequisite": _rocr_aql_binding(root),
        "gpu_topology": {"gpu_count": 1, "architecture": "gfx950"},
    }
    identity = _identity(
        root,
        base,
        gem5_source,
        runtime_source,
        rocm_systems_source,
        plugin_sources,
        managed_inputs,
        facade,
    )
    product_id = _product_id(identity)
    prefix = _product_prefix(root, product_id)
    if prefix.exists() or prefix.is_symlink():
        raise ProductEnvironmentError(f"content-addressed product output already exists: {prefix}")
    source_lock = {
        "schema": SOURCE_LOCK_SCHEMA,
        "product_id": product_id,
        "identity": identity,
        "repositories": {
            "gem5": gem5_source,
            "self-amdgpu-runtime": runtime_source,
            "rocm-systems": rocm_systems_source,
        },
        "facade": facade,
        "plugins": plugin_sources,
    }
    lock_payload = canonical_json(source_lock)
    lock_path = _lock_path(root, product_id)
    if lock_path.exists():
        existing = _read_owned_file(
            lock_path, private=True, label="product source lock"
        )
        if existing != lock_payload:
            raise ProductEnvironmentError("existing product source lock differs from canonical freeze")
    else:
        _atomic_write(lock_path, lock_payload, mode=0o400, replace=False)
    frozen = {
        "schema": FROZEN_SCHEMA,
        "product_id": product_id,
        "source_lock": str(lock_path),
        "source_lock_sha256": sha256_bytes(lock_payload),
    }
    _atomic_write(_frozen_path(root), canonical_json(frozen), mode=0o600, replace=True)
    return prefix


def _load_frozen(root: Path, product_id: str | None = None) -> tuple[dict[str, Any], bytes]:
    if product_id is None:
        frozen, _ = _read_json(
            _frozen_path(root), canonical=True, private=True, label="frozen product record"
        )
        if set(frozen) != {
            "schema",
            "product_id",
            "source_lock",
            "source_lock_sha256",
        } or frozen.get("schema") != FROZEN_SCHEMA:
            raise ProductEnvironmentError("frozen product record schema is invalid")
        product_id = frozen.get("product_id")
        if not isinstance(product_id, str):
            raise ProductEnvironmentError("frozen product id is invalid")
        expected_path = _lock_path(root, product_id)
        if frozen.get("source_lock") != str(expected_path):
            raise ProductEnvironmentError("frozen product source-lock path is invalid")
        expected_sha = frozen.get("source_lock_sha256")
    else:
        expected_path = _lock_path(root, product_id)
        expected_sha = None
    source_lock, payload = _read_json(
        expected_path, canonical=True, private=True, label="product source lock"
    )
    if expected_sha is not None and sha256_bytes(payload) != expected_sha:
        raise ProductEnvironmentError("frozen product source-lock digest mismatch")
    if (
        source_lock.get("schema") != SOURCE_LOCK_SCHEMA
        or source_lock.get("product_id") != product_id
        or not isinstance(source_lock.get("identity"), dict)
        or _product_id(source_lock["identity"]) != product_id
    ):
        raise ProductEnvironmentError("product source lock identity is invalid")
    repositories = source_lock.get("repositories")
    if not isinstance(repositories, dict) or set(repositories) != {
        "gem5",
        "self-amdgpu-runtime",
        "rocm-systems",
    }:
        raise ProductEnvironmentError("product source lock repository inventory is invalid")
    if source_lock.get("facade") != source_lock["identity"].get("facade"):
        raise ProductEnvironmentError("product source lock facade binding is invalid")
    plugins = source_lock.get("plugins")
    if not isinstance(plugins, dict) or set(plugins) != set(PLUGIN_SOURCES):
        raise ProductEnvironmentError("product source lock plugin inventory is invalid")
    return source_lock, payload


def _verify_base_binding(root: Path, expected: dict[str, Any]) -> None:
    prefix = expected.get("prefix")
    if not isinstance(prefix, str):
        raise ProductEnvironmentError("product base binding is invalid")
    actual = _base_binding(root, Path(prefix))
    if actual != expected:
        raise ProductEnvironmentError("schema-8 base prefix drifted from the frozen product identity")


def _verify_repository_lock(root: Path, source_lock: dict[str, Any]) -> None:
    repositories = source_lock["repositories"]
    for name, relative in (
        ("gem5", "projects/gem5"),
        ("self-amdgpu-runtime", "projects/self-amdgpu-runtime"),
        ("rocm-systems", str(ROCM_SYSTEMS_RELATIVE)),
    ):
        expected = repositories[name]
        actual = repository_source_set(
            root / relative, allow_gitlinks=name == "rocm-systems"
        )
        if actual != expected:
            raise ProductEnvironmentError(f"{name} actual source set drifted after freeze")


def _verify_plugin_locks(root: Path, source_lock: dict[str, Any]) -> None:
    for name, relative in PLUGIN_SOURCES.items():
        expected = source_lock["plugins"][name]
        actual = directory_source_set(root / relative)
        if actual != expected:
            raise ProductEnvironmentError(
                f"{name} actual source set drifted after freeze"
            )


def _verify_facade_lock(root: Path, source_lock: dict[str, Any]) -> None:
    facade = source_lock.get("facade")
    if not isinstance(facade, dict):
        raise ProductEnvironmentError("frozen facade binding is missing")
    expected_stage = facade.get("stage")
    if not isinstance(expected_stage, dict):
        raise ProductEnvironmentError("frozen facade stage binding is invalid")
    actual_stage = facade_stage_source_set(root / FACADE_STAGE_RELATIVE)
    if actual_stage != expected_stage:
        raise ProductEnvironmentError("upstream HIP facade stage drifted after freeze")
    expected_build = facade.get("build")
    if not isinstance(expected_build, dict):
        raise ProductEnvironmentError("frozen facade build binding is invalid")
    actual_build = _facade_build_binding(root)
    if actual_build != expected_build:
        raise ProductEnvironmentError("upstream HIP facade build dependencies drifted")
    expected_tool = facade.get("topology_tool")
    actual_tool = _regular_tree_binding(root / TOPOLOGY_TOOL_RELATIVE)
    if actual_tool != expected_tool:
        raise ProductEnvironmentError("HSAKMT topology generator drifted after freeze")
    expected_rocm = facade.get("rocm_systems")
    actual_rocm = repository_source_set(
        root / ROCM_SYSTEMS_RELATIVE, allow_gitlinks=True
    )
    if not isinstance(expected_rocm, dict) or any(
        actual_rocm.get(key) != expected_rocm.get(key)
        for key in ("repository", "head", "tree", "source_set_sha256")
    ):
        raise ProductEnvironmentError("ROCm systems facade identity drifted")
    _rocr_aql_binding(root)


def _verify_managed_inputs(root: Path, expected: dict[str, Any]) -> None:
    actual = _workspace_inputs(root)
    if actual != expected:
        raise ProductEnvironmentError("gem5 binary or configuration drifted after freeze")


def _verify_setup_inputs(root: Path, expected: Any) -> None:
    if not isinstance(expected, dict) or any(
        expected.get(name) != value
        for name, value in (
            ("product_schema", PRODUCT_SETUP_SCHEMA),
            ("base_setup_schema", BASE_SETUP_SCHEMA),
        )
    ):
        raise ProductEnvironmentError("frozen product setup contract is invalid")
    current = {
        "product_environment": _regular_sha256(Path(__file__).resolve(strict=True)),
        "setup_entry": _regular_sha256(root / "scripts/setup_rocm_env.sh"),
    }
    if any(expected.get(name) != descriptor for name, descriptor in current.items()):
        raise ProductEnvironmentError("product setup implementation drifted after freeze")


def _validate_frozen_inputs(root: Path, source_lock: dict[str, Any]) -> None:
    identity = source_lock["identity"]
    if identity.get("schema") != IDENTITY_SCHEMA or identity.get("build_contract") != BUILD_CONTRACT:
        raise ProductEnvironmentError("frozen product build contract is invalid")
    _verify_setup_inputs(root, identity.get("setup"))
    _verify_base_binding(root, identity.get("base"))
    _verify_repository_lock(root, source_lock)
    _verify_facade_lock(root, source_lock)
    _verify_plugin_locks(root, source_lock)
    _verify_managed_inputs(root, identity.get("managed_inputs"))


def _clean_build_environment(
    root: Path,
    build_dir: Path,
    staging_root: Path,
    *,
    facade_root: Path | None = None,
    dependency_prefix: Path | None = None,
) -> dict[str, str]:
    state = build_dir / "state"
    ccache = root / "env/ccache/product-runtime"
    for directory in (state / "home", state / "tmp", state / "cache", state / "config", ccache):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment = {
        "HOME": str(state / "home"),
        "TMPDIR": str(state / "tmp"),
        "XDG_CACHE_HOME": str(state / "cache"),
        "XDG_CONFIG_HOME": str(state / "config"),
        "PATH": "/usr/lib/ccache:/usr/bin:/bin",
        "LC_ALL": "C",
        "SOURCE_DATE_EPOCH": "0",
        "CC": BUILD_CONTRACT["c_compiler"],
        "CXX": BUILD_CONTRACT["cxx_compiler"],
        "CCACHE_DIR": str(ccache),
        "CCACHE_BASEDIR": str(root),
        "CCACHE_NOHASHDIR": "1",
        "DESTDIR": str(staging_root),
    }
    if facade_root is not None:
        library_paths = [facade_root / "lib"]
        if dependency_prefix is not None:
            library_paths.append(dependency_prefix / "lib")
        environment.update(
            {
                "HIP_PLATFORM": "amd",
                "HIP_PATH": str(facade_root),
                "HSA_PATH": str(facade_root),
                "ROCM_PATH": str(facade_root),
                "HIP_CLANG_PATH": str(root / "env/rocm/hip-facade-llvm-tools-v1/bin"),
                "CMAKE_PREFIX_PATH": ":".join(
                    str(path) for path in (facade_root, dependency_prefix) if path is not None
                ),
                "PKG_CONFIG_PATH": ":".join(
                    str(path / "lib/pkgconfig")
                    for path in (facade_root, dependency_prefix)
                    if path is not None
                ),
                "LD_LIBRARY_PATH": ":".join(str(path) for path in library_paths),
            }
        )
    return environment


def _run(command: list[str], *, environment: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=environment)


def _copy_exclusive(
    source: Path,
    destination: Path,
    mode: int,
    *,
    expected: dict[str, Any] | None = None,
) -> None:
    if destination.exists() or destination.is_symlink():
        raise ProductEnvironmentError(f"staged output already exists: {destination}")
    source_flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    source_descriptor = os.open(source, source_flags)
    source_before = os.fstat(source_descriptor)
    if not stat.S_ISREG(source_before.st_mode) or source_before.st_uid != os.getuid():
        os.close(source_descriptor)
        raise ProductEnvironmentError(f"copy source is not an owned regular file: {source}")
    if expected is not None and (
        expected.get("bytes") != source_before.st_size
        or expected.get("executable") != bool(source_before.st_mode & stat.S_IXUSR)
    ):
        os.close(source_descriptor)
        raise ProductEnvironmentError(f"copy source metadata drifted: {source}")
    destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        destination_descriptor = os.open(destination, flags, mode)
    except Exception:
        os.close(source_descriptor)
        raise
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            _write_all(destination_descriptor, chunk)
        source_after = os.fstat(source_descriptor)
        if (
            source_after.st_size != source_before.st_size
            or source_after.st_mtime_ns != source_before.st_mtime_ns
            or source_after.st_ctime_ns != source_before.st_ctime_ns
        ):
            raise ProductEnvironmentError(f"copy source changed while reading: {source}")
        if expected is not None and digest.hexdigest() != expected.get("sha256"):
            raise ProductEnvironmentError(f"copy source content drifted: {source}")
        os.fchmod(destination_descriptor, mode)
        os.fsync(destination_descriptor)
    finally:
        os.close(destination_descriptor)
        os.close(source_descriptor)


def _checked_source_relative(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ProductEnvironmentError("repository source-set path is invalid")
    relative = Path(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ProductEnvironmentError(f"unsafe repository source-set path: {value!r}")
    return relative


def _repository_records(source_set: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    raw_records = source_set.get("files")
    if (
        not isinstance(raw_records, list)
        or source_set.get("file_count") != len(raw_records)
    ):
        raise ProductEnvironmentError("repository source-set inventory is invalid")
    records: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for record in raw_records:
        if not isinstance(record, dict):
            raise ProductEnvironmentError("repository source-set record is invalid")
        relative = _checked_source_relative(record.get("path"))
        if relative.as_posix() in seen or record.get("kind") not in {
            "regular",
            "symlink",
            "missing",
        }:
            raise ProductEnvironmentError("repository source-set record is invalid")
        seen.add(relative.as_posix())
        records.append((relative, record))
    return records


def _safe_snapshot_symlink(relative: Path, target: Any) -> str:
    if not isinstance(target, str) or not target or "\0" in target:
        raise ProductEnvironmentError(f"invalid source symlink target: {relative}")
    target_path = Path(target)
    normalized = Path(os.path.normpath(relative.parent / target_path))
    if target_path.is_absolute() or normalized.is_absolute() or ".." in normalized.parts:
        raise ProductEnvironmentError(f"source symlink escapes snapshot: {relative}")
    return target


def _verify_repository_snapshot(snapshot: Path, source_set: dict[str, Any]) -> None:
    expected_paths: set[str] = set()
    for relative, record in _repository_records(source_set):
        path = snapshot / relative
        kind = record["kind"]
        if kind == "missing":
            if path.exists() or path.is_symlink():
                raise ProductEnvironmentError(f"missing source appeared in snapshot: {relative}")
            continue
        expected_paths.add(relative.as_posix())
        metadata = path.lstat()
        if kind == "regular":
            if not stat.S_ISREG(metadata.st_mode):
                raise ProductEnvironmentError(f"snapshot source is not regular: {relative}")
            actual = _regular_sha256(path)
            if (
                actual["bytes"] != record.get("bytes")
                or actual["sha256"] != record.get("sha256")
                or bool(metadata.st_mode & stat.S_IXUSR) != record.get("executable")
            ):
                raise ProductEnvironmentError(f"snapshot source drifted: {relative}")
        else:
            target = _safe_snapshot_symlink(relative, record.get("target"))
            if not stat.S_ISLNK(metadata.st_mode) or os.readlink(path) != target:
                raise ProductEnvironmentError(f"snapshot symlink drifted: {relative}")
            encoded_target = os.fsencode(target)
            if (
                len(encoded_target) != record.get("bytes")
                or sha256_bytes(encoded_target) != record.get("sha256")
            ):
                raise ProductEnvironmentError(f"snapshot symlink identity is invalid: {relative}")

    actual_paths: set[str] = set()
    for current, directory_names, file_names in os.walk(
        snapshot, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        retained: list[str] = []
        for name in directory_names:
            path = current_path / name
            if path.is_symlink():
                actual_paths.add(path.relative_to(snapshot).as_posix())
            else:
                retained.append(name)
        directory_names[:] = retained
        actual_paths.update(
            (current_path / name).relative_to(snapshot).as_posix()
            for name in file_names
        )
    if actual_paths != expected_paths:
        raise ProductEnvironmentError("repository source snapshot inventory drifted")


def _materialize_repository_snapshot(
    source_set: dict[str, Any], destination: Path
) -> Path:
    source_root_text = source_set.get("repository")
    if not isinstance(source_root_text, str):
        raise ProductEnvironmentError("repository source-set root is invalid")
    source_root = Path(source_root_text).resolve(strict=True)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise ProductEnvironmentError(f"source snapshot path is unsafe: {destination}")
        _verify_repository_snapshot(destination, source_set)
        return destination
    temporary = destination.parent / (
        f".{destination.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    temporary.mkdir(mode=0o700)
    try:
        for relative, record in _repository_records(source_set):
            source = source_root / relative
            target = temporary / relative
            if record["kind"] == "missing":
                if source.exists() or source.is_symlink():
                    raise ProductEnvironmentError(f"frozen missing source reappeared: {relative}")
            elif record["kind"] == "regular":
                _copy_exclusive(
                    source,
                    target,
                    0o555 if record["executable"] else 0o444,
                    expected=record,
                )
            else:
                symlink_target = _safe_snapshot_symlink(relative, record.get("target"))
                metadata = source.lstat()
                if (
                    not stat.S_ISLNK(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                    or os.readlink(source) != symlink_target
                ):
                    raise ProductEnvironmentError(f"frozen source symlink drifted: {relative}")
                target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                os.symlink(symlink_target, target)
        for current, directory_names, _ in os.walk(
            temporary, topdown=False, followlinks=False
        ):
            for name in directory_names:
                path = Path(current) / name
                if not path.is_symlink():
                    os.chmod(path, 0o555)
        _verify_repository_snapshot(temporary, source_set)
        os.chmod(temporary, 0o555)
        try:
            _rename_noreplace(temporary, destination)
        except FileExistsError:
            _verify_repository_snapshot(destination, source_set)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            for current, directory_names, _ in os.walk(
                temporary, topdown=False, followlinks=False
            ):
                for name in directory_names:
                    path = Path(current) / name
                    if not path.is_symlink():
                        os.chmod(path, 0o700)
            os.chmod(temporary, 0o700)
        shutil.rmtree(temporary, ignore_errors=True)
    return destination


def _facade_stage_records(source_set: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        source_set.get("schema") != FACADE_STAGE_SCHEMA
        or not isinstance(source_set.get("files"), list)
        or source_set.get("file_count") != len(source_set["files"])
    ):
        raise ProductEnvironmentError("facade stage source-set inventory is invalid")
    records = source_set["files"]
    paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ProductEnvironmentError("facade stage source-set record is invalid")
        relative = _checked_source_relative(record["path"])
        if relative.as_posix() in paths or record.get("kind") not in {"regular", "symlink"}:
            raise ProductEnvironmentError("facade stage source-set record is invalid")
        paths.add(relative.as_posix())
        if record["kind"] == "symlink":
            target = record.get("target")
            target_path = Path(os.path.normpath(relative.parent / str(target)))
            if (
                not isinstance(target, str)
                or not target
                or Path(target).is_absolute()
                or target_path.is_absolute()
                or ".." in target_path.parts
            ):
                raise ProductEnvironmentError("facade stage source symlink escapes tree")
    return records


def _copy_facade_stage(
    source_set: dict[str, Any], staged_prefix: Path, published_prefix: Path
) -> dict[str, Any]:
    source_root_text = source_set.get("directory")
    if not isinstance(source_root_text, str):
        raise ProductEnvironmentError("facade stage source root is invalid")
    source_root = Path(source_root_text).resolve(strict=True)
    copied: list[str] = []
    for record in _facade_stage_records(source_set):
        relative = _checked_source_relative(record["path"])
        source = source_root / relative
        destination = staged_prefix / relative
        if record["kind"] == "regular":
            _copy_exclusive(
                source,
                destination,
                0o555 if record.get("executable") else 0o444,
                expected=record,
            )
        else:
            source_metadata = source.lstat()
            target = record["target"]
            if not stat.S_ISLNK(source_metadata.st_mode) or os.readlink(source) != target:
                raise ProductEnvironmentError(f"facade stage symlink drifted: {relative}")
            destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                raise ProductEnvironmentError(f"facade stage destination already exists: {relative}")
            os.symlink(target, destination)
        copied.append(relative.as_posix())
    return {
        "schema": FACADE_STAGE_SCHEMA,
        "source_set_sha256": source_set["source_set_sha256"],
        "source_snapshot": str(source_root),
        "installed_root": str(published_prefix),
        "file_count": len(copied),
        "paths_sha256": sha256_bytes(canonical_json(copied)),
    }


def _verify_facade_snapshot(snapshot: Path, source_set: dict[str, Any]) -> None:
    """Verify a materialized standard-ROCm stage without following escapes."""
    expected_paths: set[str] = set()
    for record in _facade_stage_records(source_set):
        relative = _checked_source_relative(record["path"])
        path = snapshot / relative
        expected_paths.add(relative.as_posix())
        if record["kind"] == "regular":
            try:
                metadata = path.lstat()
            except FileNotFoundError as error:
                raise ProductEnvironmentError(
                    f"facade snapshot file is missing: {relative}"
                ) from error
            if not stat.S_ISREG(metadata.st_mode):
                raise ProductEnvironmentError(
                    f"facade snapshot entry is not regular: {relative}"
                )
            actual = _regular_sha256(path)
            if (
                actual["bytes"] != record.get("bytes")
                or actual["sha256"] != record.get("sha256")
                or bool(metadata.st_mode & stat.S_IXUSR)
                != bool(record.get("executable"))
            ):
                raise ProductEnvironmentError(f"facade snapshot drifted: {relative}")
        else:
            target = _safe_snapshot_symlink(relative, record.get("target"))
            try:
                metadata = path.lstat()
            except FileNotFoundError as error:
                raise ProductEnvironmentError(
                    f"facade snapshot symlink is missing: {relative}"
                ) from error
            if not stat.S_ISLNK(metadata.st_mode) or os.readlink(path) != target:
                raise ProductEnvironmentError(f"facade snapshot symlink drifted: {relative}")
    actual_paths: set[str] = set()
    for current, directory_names, file_names in os.walk(
        snapshot, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        retained: list[str] = []
        for name in directory_names:
            path = current_path / name
            if path.is_symlink():
                actual_paths.add(path.relative_to(snapshot).as_posix())
            else:
                retained.append(name)
        directory_names[:] = retained
        actual_paths.update(
            (current_path / name).relative_to(snapshot).as_posix() for name in file_names
        )
    if actual_paths != expected_paths:
        raise ProductEnvironmentError("facade snapshot inventory drifted")


def _materialize_facade_snapshot(
    source_set: dict[str, Any], destination: Path
) -> Path:
    """Create a read-only content-addressed copy used by the runtime build."""
    source_root_text = source_set.get("directory")
    if not isinstance(source_root_text, str):
        raise ProductEnvironmentError("facade stage source root is invalid")
    source_root = Path(source_root_text).resolve(strict=True)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise ProductEnvironmentError(f"facade snapshot path is unsafe: {destination}")
        _verify_facade_snapshot(destination, source_set)
        return destination
    temporary = destination.parent / (
        f".{destination.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    temporary.mkdir(mode=0o700)
    try:
        for record in _facade_stage_records(source_set):
            relative = _checked_source_relative(record["path"])
            source = source_root / relative
            target = temporary / relative
            if record["kind"] == "regular":
                _copy_exclusive(
                    source,
                    target,
                    0o555 if record.get("executable") else 0o444,
                    expected=record,
                )
            else:
                symlink_target = _safe_snapshot_symlink(relative, record.get("target"))
                metadata = source.lstat()
                if (
                    not stat.S_ISLNK(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                    or os.readlink(source) != symlink_target
                ):
                    raise ProductEnvironmentError(
                        f"facade source symlink drifted: {relative}"
                    )
                target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                os.symlink(symlink_target, target)
        _verify_facade_snapshot(temporary, source_set)
        for current, directory_names, _ in os.walk(
            temporary, topdown=False, followlinks=False
        ):
            for name in directory_names:
                path = Path(current) / name
                if not path.is_symlink():
                    os.chmod(path, 0o555)
        os.chmod(temporary, 0o555)
        try:
            _rename_noreplace(temporary, destination)
        except FileExistsError:
            _verify_facade_snapshot(destination, source_set)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            for current, directory_names, _ in os.walk(
                temporary, topdown=False, followlinks=False
            ):
                for name in directory_names:
                    path = Path(current) / name
                    if not path.is_symlink():
                        os.chmod(path, 0o700)
            os.chmod(temporary, 0o700)
        shutil.rmtree(temporary, ignore_errors=True)
    return destination


def _source_subtree_records(
    source_set: dict[str, Any], prefix: str
) -> list[tuple[Path, Path, dict[str, Any]]]:
    normalized_prefix = prefix.rstrip("/") + "/"
    prefix_path = Path(prefix)
    selected: list[tuple[Path, Path, dict[str, Any]]] = []
    raw_records = source_set.get("files")
    if (
        not isinstance(raw_records, list)
        or source_set.get("file_count") != len(raw_records)
    ):
        raise ProductEnvironmentError("upstream source-set inventory is invalid")
    seen: set[str] = set()
    for record in raw_records:
        if not isinstance(record, dict):
            raise ProductEnvironmentError("upstream source-set record is invalid")
        source_relative = _checked_source_relative(record.get("path"))
        kind = record.get("kind")
        if (
            source_relative.as_posix() in seen
            or kind not in {"regular", "symlink", "missing", "gitlink"}
        ):
            raise ProductEnvironmentError("upstream source-set record is invalid")
        seen.add(source_relative.as_posix())
        if not source_relative.as_posix().startswith(normalized_prefix):
            continue
        if kind not in {"regular", "symlink"}:
            raise ProductEnvironmentError(
                f"frozen upstream subtree contains {kind}: {source_relative}"
            )
        selected.append(
            (source_relative.relative_to(prefix_path), source_relative, record)
        )
    if not selected:
        raise ProductEnvironmentError(f"frozen upstream subtree is empty: {prefix}")
    return selected


def _verify_source_subtree(
    destination: Path,
    records: list[tuple[Path, Path, dict[str, Any]]],
) -> None:
    expected_paths: set[str] = set()
    for relative, _, record in records:
        path = destination / relative
        expected_paths.add(relative.as_posix())
        try:
            metadata = path.lstat()
        except FileNotFoundError as error:
            raise ProductEnvironmentError(
                f"upstream subtree snapshot entry is missing: {relative}"
            ) from error
        if record["kind"] == "regular":
            actual = _regular_sha256(path)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or actual["bytes"] != record.get("bytes")
                or actual["sha256"] != record.get("sha256")
                or bool(metadata.st_mode & stat.S_IXUSR)
                != bool(record.get("executable"))
            ):
                raise ProductEnvironmentError(
                    f"upstream subtree snapshot drifted: {relative}"
                )
        else:
            target = _safe_snapshot_symlink(relative, record.get("target"))
            if not stat.S_ISLNK(metadata.st_mode) or os.readlink(path) != target:
                raise ProductEnvironmentError(
                    f"upstream subtree snapshot symlink drifted: {relative}"
                )
    actual_paths: set[str] = set()
    for current, directory_names, file_names in os.walk(
        destination, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        retained: list[str] = []
        for name in directory_names:
            path = current_path / name
            if path.is_symlink():
                actual_paths.add(path.relative_to(destination).as_posix())
            else:
                retained.append(name)
        directory_names[:] = retained
        actual_paths.update(
            (current_path / name).relative_to(destination).as_posix()
            for name in file_names
        )
    if actual_paths != expected_paths:
        raise ProductEnvironmentError("upstream subtree snapshot inventory drifted")


def _materialize_source_subtree(
    source_set: dict[str, Any], prefix: str, destination: Path
) -> Path:
    """Materialize one upstream include subtree from an already frozen inventory."""
    source_root_text = source_set.get("repository")
    if not isinstance(source_root_text, str):
        raise ProductEnvironmentError("upstream source root is invalid")
    source_root = Path(source_root_text).resolve(strict=True)
    selected = _source_subtree_records(source_set, prefix)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise ProductEnvironmentError(f"upstream subtree snapshot is unsafe: {destination}")
        _verify_source_subtree(destination, selected)
        return destination
    temporary = destination.parent / (
        f".{destination.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    temporary.mkdir(mode=0o700)
    try:
        for relative, source_relative, record in selected:
            source = source_root / source_relative
            target = temporary / relative
            if record["kind"] == "regular":
                _copy_exclusive(
                    source,
                    target,
                    0o555 if record.get("executable") else 0o444,
                    expected=record,
                )
            else:
                symlink_target = _safe_snapshot_symlink(relative, record.get("target"))
                metadata = source.lstat()
                if not stat.S_ISLNK(metadata.st_mode) or os.readlink(source) != symlink_target:
                    raise ProductEnvironmentError(f"upstream subtree symlink drifted: {source}")
                target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                os.symlink(symlink_target, target)
        for current, directory_names, _ in os.walk(
            temporary, topdown=False, followlinks=False
        ):
            for name in directory_names:
                path = Path(current) / name
                if not path.is_symlink():
                    os.chmod(path, 0o555)
        os.chmod(temporary, 0o555)
        _verify_source_subtree(temporary, selected)
        try:
            _rename_noreplace(temporary, destination)
        except FileExistsError:
            _verify_source_subtree(destination, selected)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            for current, directory_names, _ in os.walk(
                temporary, topdown=False, followlinks=False
            ):
                for name in directory_names:
                    path = Path(current) / name
                    if not path.is_symlink():
                        os.chmod(path, 0o700)
            os.chmod(temporary, 0o700)
        shutil.rmtree(temporary, ignore_errors=True)
    return destination


def _snapshot_plugins(
    source_lock: dict[str, Any], staged_prefix: Path, published_prefix: Path
) -> dict[str, Any]:
    plugin_root = staged_prefix / "python"
    plugin_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    snapshots: dict[str, Any] = {}
    package_paths = {
        "triton-gemsim-amd": Path("triton/backends/gemsim_amd"),
        "gemsim-vllm": Path("gemsim_vllm"),
        "gemsim-ccl": Path("gemsim_ccl"),
    }
    for name, relative in PLUGIN_SOURCES.items():
        source_set = source_lock["plugins"][name]
        source_root = Path(source_set["directory"])
        destination_root = plugin_root / "source" / name
        for record in source_set["files"]:
            source = source_root / record["path"]
            destination = destination_root / record["path"]
            _copy_exclusive(
                source,
                destination,
                0o555 if record["executable"] else 0o444,
                expected=record,
            )
        if name == "triton-gemsim-amd":
            package_source = destination_root / "backend"
        else:
            package_source = destination_root / "src" / package_paths[name].name
        package_destination = plugin_root / package_paths[name]
        if not package_source.is_dir():
            raise ProductEnvironmentError(
                f"plugin snapshot package is missing: {package_source}"
            )
        package_destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        shutil.copytree(
            package_source,
            package_destination,
            symlinks=False,
            copy_function=shutil.copy2,
        )
        snapshots[name] = {
            "source_set_sha256": source_set["source_set_sha256"],
            "source_snapshot": str(published_prefix / "python/source" / name),
            "package_path": str(
                published_prefix / package_destination.relative_to(staged_prefix)
            ),
        }
    return snapshots


def _write_python_launcher(path: Path, source_lock: dict[str, Any]) -> None:
    source_roots = [
        source_lock["plugins"][name]["directory"] for name in sorted(PLUGIN_SOURCES)
    ]
    payload = f"""#!/usr/bin/env python3
import importlib.abc
import importlib.util
import os
from pathlib import Path
import runpy
import sys

product = Path(os.environ["ROCM_SIM_ROOT"]).resolve()
overlay = product / "python"
blocked = set()
for value in {json.dumps(source_roots, ensure_ascii=True)}:
    source = Path(value).resolve()
    blocked.update((source, source / "src"))

def retained(path):
    if not path:
        return True
    candidate = Path(path).resolve()
    return all(candidate != root for root in blocked)

sys.path[:] = [path for path in sys.path if retained(path)]
script = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(overlay), str(script.parent)]

class ProductTritonBackendFinder(importlib.abc.MetaPathFinder):
    prefix = "triton.backends.gemsim_amd"
    package = overlay / "triton/backends/gemsim_amd"

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.prefix:
            return importlib.util.spec_from_file_location(
                fullname,
                self.package / "__init__.py",
                submodule_search_locations=[str(self.package)],
            )
        if fullname.startswith(self.prefix + "."):
            relative = fullname[len(self.prefix) + 1:].replace(".", "/")
            package = self.package / relative / "__init__.py"
            if package.is_file():
                return importlib.util.spec_from_file_location(
                    fullname,
                    package,
                    submodule_search_locations=[str(package.parent)],
                )
            module = self.package / (relative + ".py")
            if module.is_file():
                return importlib.util.spec_from_file_location(fullname, module)
        return None

sys.meta_path.insert(0, ProductTritonBackendFinder())
sys.argv[:] = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name="__main__")
""".encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o500)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _build_runtime(
    root: Path,
    source_lock: dict[str, Any],
    target: Path,
    staging_root: Path,
    staged_prefix: Path,
) -> None:
    runtime_source_sha = source_lock["identity"]["source_sets"]["self-amdgpu-runtime"]
    build_root = _runtime_build_root(root, runtime_source_sha)
    source_dir = _materialize_repository_snapshot(
        source_lock["repositories"]["self-amdgpu-runtime"],
        build_root / "source",
    )
    facade_source = source_lock["facade"]["stage"]
    facade_dir = _materialize_facade_snapshot(
        facade_source,
        build_root / "facade-stage",
    )
    hsakmt_include_dir = _materialize_source_subtree(
        source_lock["repositories"]["rocm-systems"],
        "projects/rocr-runtime/libhsakmt/include",
        build_root / "hsakmt-include",
    )
    dependency_prefix = Path(source_lock["facade"]["build"]["prefix"])
    build_dir = build_root / "build"
    _validate_hsakmt_topology_build_path(
        build_dir, source_lock["facade"]["topology_tool"]["sha256"]
    )
    build_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment = _clean_build_environment(
        root,
        build_dir,
        staging_root,
        facade_root=facade_dir,
        dependency_prefix=dependency_prefix,
    )
    base = source_lock["identity"]["base"]["prefix"]
    managed = source_lock["identity"]["managed_inputs"]
    configure = [
        "/usr/bin/cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-G",
        BUILD_CONTRACT["generator"],
        f"-DCMAKE_BUILD_TYPE={BUILD_CONTRACT['build_type']}",
        f"-DCMAKE_INSTALL_PREFIX={target}",
        f"-DCMAKE_C_COMPILER={BUILD_CONTRACT['c_compiler']}",
        f"-DCMAKE_CXX_COMPILER={BUILD_CONTRACT['cxx_compiler']}",
        f"-DCMAKE_LINKER={BUILD_CONTRACT['linker']}",
        f"-DCMAKE_EXE_LINKER_FLAGS={BUILD_CONTRACT['linker_flag']}",
        f"-DCMAKE_SHARED_LINKER_FLAGS={BUILD_CONTRACT['linker_flag']}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DBUILD_SHARED_LIBS=ON",
        "-DSELF_AMDGPU_RUNTIME_BUILD_TESTS=ON",
        "-DSELF_AMDGPU_RUNTIME_BUILD_TOOLS=ON",
        "-DSELF_AMDGPU_RUNTIME_BUILD_OPENCL=ON",
        "-DSELF_AMDGPU_RUNTIME_BUILD_HSAKMT_MODEL=ON",
        f"-DSELF_AMDGPU_RUNTIME_HSAKMT_INCLUDE_DIR={hsakmt_include_dir}",
        f"-DSELF_AMDGPU_RUNTIME_UPSTREAM_HSAKMT_LIBRARY={facade_dir / 'lib/libhsakmt.a'}",
        f"-DSELF_AMDGPU_RUNTIME_UPSTREAM_HSAKMT_DEPENDENCY_PREFIX={dependency_prefix}",
        f"-DSELF_AMDGPU_RUNTIME_UPSTREAM_ROCR_LIBRARY={facade_dir / 'lib/libhsa-runtime64.so.1'}",
        f"-DSELF_AMDGPU_RUNTIME_UPSTREAM_ROCR_INCLUDE_DIR={facade_dir / 'include'}",
        f"-DSELF_AMDGPU_RUNTIME_OPENCL_PREFIX={base}",
        f"-DSELF_AMDGPU_RUNTIME_REPO_ROOT={root}",
        f"-DSELF_AMDGPU_RUNTIME_OPENCL_GEM5={managed['gem5_binary']['path']}",
        f"-DSELF_AMDGPU_RUNTIME_OPENCL_GEM5_CONFIG={managed['gem5_config']['path']}",
        "-DSELF_AMDGPU_RUNTIME_TEST_INSTALLED_PACKAGE=OFF",
    ]
    _run(configure, environment=environment)
    _run(
        [
            "/usr/bin/cmake",
            "--build",
            str(build_dir),
            "--parallel",
            str(BUILD_CONTRACT["parallel_jobs"]),
        ],
        environment=environment,
    )
    _run(
        ["/usr/bin/ctest", "--test-dir", str(build_dir), "--output-on-failure"],
        environment=environment,
    )
    _run(["/usr/bin/cmake", "--install", str(build_dir)], environment=environment)
    topology_tool = source_dir / "tools/hsakmt-model-topology.py"
    expected_tool = source_lock["facade"]["topology_tool"]
    actual_tool = _regular_sha256(topology_tool)
    if any(
        actual_tool.get(key) != expected_tool.get(key)
        for key in ("bytes", "sha256")
    ):
        raise ProductEnvironmentError("frozen topology tool snapshot differs from source lock")
    topology_output = staged_prefix / "share/self-amdgpu-runtime/hsakmt-topology"
    _run(
        [
            sys.executable,
            str(topology_tool),
            "--output-dir",
            str(topology_output),
            "--gpu-count",
            str(source_lock["facade"]["gpu_topology"]["gpu_count"]),
        ],
        environment=environment,
    )
    facade_product_source = dict(source_lock["facade"]["stage"])
    facade_product_source["directory"] = str(facade_dir)
    _copy_facade_stage(facade_product_source, staged_prefix, target)
    endpoint = build_dir / "tests/self_amdgpu_runtime_generic_dispatch_v2_endpoint_test"
    _copy_exclusive(
        endpoint,
        staged_prefix / "libexec/amdgpu-sim/generic-dispatch-v2-endpoint-test",
        0o555,
    )


def _write_activation(path: Path, prefix: Path, base: Path) -> None:
    content = f"""# Source in a disposable shell; no system files are modified.
for _amdgpu_sim_var in ${{!CUDA_@}} ${{!CONDA_@}} ${{!LD_@}} ${{!HIP_@}} ${{!HSA_@}} ${{!ROCM_@}} ${{!SAGR_@}}; do
    unset "$_amdgpu_sim_var"
done
unset _amdgpu_sim_var PYTHONHOME PYTHONPATH VIRTUAL_ENV
export ROCM_SIM_ROOT={prefix}
    export ROCM_PATH={base}
    export HIP_PATH={prefix}
    export HSA_PATH={prefix}
    export HIP_PLATFORM=amd
    export HIP_CLANG_PATH={base}/bin
    export HSA_ENABLE_DXG_DETECTION=0
    # Model-backed VRAM is not host-accessible; use the standard ROCr blit AQL path.
    export HSA_ENABLE_DTIF_FAST_COPY=0
    export HSA_ENABLE_INTERRUPT=0
    export HSA_MODEL_LIB={prefix}/lib/libself_amdgpu_hsakmt_model.so.1
    export HSA_MODEL_TOPOLOGY={prefix}/share/self-amdgpu-runtime/hsakmt-topology
    export LD_LIBRARY_PATH={prefix}/lib:{base}/lib
    export PATH={base}/venv/bin:{prefix}/bin:{base}/bin:/usr/bin:/bin
export TRITON_DEFAULT_BACKEND=gemsim_amd
export PYTHONNOUSERSITE=1
"""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o500)
    try:
        _write_all(descriptor, content.encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inventory(prefix: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            path = Path(entry.path)
            relative = path.relative_to(prefix).as_posix()
            if relative == "manifest.json":
                continue
            metadata = path.lstat()
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISDIR(metadata.st_mode):
                records.append({"path": relative, "kind": "directory", "mode": f"{mode:04o}"})
                visit(path)
            elif stat.S_ISREG(metadata.st_mode):
                artifact = _regular_sha256(path)
                records.append(
                    {
                        "path": relative,
                        "kind": "regular",
                        "mode": f"{mode:04o}",
                        "bytes": artifact["bytes"],
                        "sha256": artifact["sha256"],
                    }
                )
            elif stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(path)
                records.append(
                    {
                        "path": relative,
                        "kind": "symlink",
                        "target": target,
                        "sha256": sha256_bytes(os.fsencode(target)),
                    }
                )
            else:
                raise ProductEnvironmentError(f"special file in product overlay: {path}")

    visit(prefix)
    return records


def _normalize_product(prefix: Path) -> None:
    directories: list[Path] = []
    for directory, names, files in os.walk(prefix, topdown=True, followlinks=False):
        current = Path(directory)
        directories.append(current)
        names[:] = [name for name in names if not (current / name).is_symlink()]
        for name in files:
            path = current / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ProductEnvironmentError(f"special file in product overlay: {path}")
            if path.name == "source-lock.json":
                mode = 0o400
            else:
                mode = 0o555 if metadata.st_mode & stat.S_IXUSR else 0o444
            os.chmod(path, mode)
    for directory in reversed(directories[1:]):
        os.chmod(directory, 0o555)


def _verify_symlinks(prefix: Path, base: Path, inventory: list[dict[str, Any]]) -> None:
    for record in inventory:
        if record["kind"] != "symlink":
            continue
        path = prefix / record["path"]
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ProductEnvironmentError(f"broken product symlink: {path}") from error
        if not (_is_within(resolved, prefix) or _is_within(resolved, base)):
            raise ProductEnvironmentError(f"product symlink escapes product/base allowlist: {path}")


def _require_product_layout(prefix: Path) -> None:
    required = (
        "bin/sagr-handshake",
        "bin/sagr-triton-hsaco-probe",
        "bin/opencl-vecadd",
        "include/self_amdgpu_runtime/runtime.h",
        "include/CL/cl.h",
        "lib/libself_amdgpu_runtime.so.1",
        "lib/libOpenCL.so.1",
        "lib/libhsa-runtime64.so.1",
        "lib/libamdhip64.so.7",
        "lib/libamd_comgr.so.3",
        "lib/librccl.so.1",
        "lib/libself_amdgpu_hsakmt_model.so.1",
        "libexec/amdgpu-sim/generic-dispatch-v2-endpoint-test",
        "share/self-amdgpu-runtime/opencl/vecadd.cl",
        "share/self-amdgpu-runtime/hsakmt-topology/manifest.json",
        "source-lock.json",
        "activate",
        "python/product_bootstrap.py",
        "python/triton/backends/gemsim_amd/driver.py",
        "python/triton/backends/gemsim_amd/compiler.py",
        "python/gemsim_vllm/__init__.py",
        "python/gemsim_ccl/__init__.py",
    )
    for relative in required:
        candidate = prefix / relative
        if not candidate.exists():
            raise ProductEnvironmentError(f"product overlay is incomplete: {relative}")
    if (prefix / "venv").exists() or (prefix / "build").exists():
        raise ProductEnvironmentError("product overlay copied base/build state")


def _manifest_document(
    root: Path,
    prefix: Path,
    content_prefix: Path,
    source_lock: dict[str, Any],
    source_lock_payload: bytes,
    inventory: list[dict[str, Any]],
    plugin_snapshots: dict[str, Any],
) -> dict[str, Any]:
    product_id = source_lock["product_id"]
    artifacts = _product_artifacts(prefix, content_prefix=content_prefix)
    return {
        "schema": PRODUCT_SCHEMA,
        "setup_schema": PRODUCT_SETUP_SCHEMA,
        "product_id": product_id,
        "prefix": str(prefix),
        "source_lock": {
            "path": str(prefix / "source-lock.json"),
            "bytes": len(source_lock_payload),
            "sha256": sha256_bytes(source_lock_payload),
        },
        "identity": source_lock["identity"],
        "facade": source_lock["facade"],
        "base": source_lock["identity"]["base"],
        "managed_inputs": source_lock["identity"]["managed_inputs"],
        "artifacts": artifacts,
        "plugins": {
            "python_root": str(prefix / "python"),
            "bootstrap": str(prefix / "python/product_bootstrap.py"),
            "sources": {
                name: {
                    "directory": descriptor["directory"],
                    "file_count": descriptor["file_count"],
                    "source_set_sha256": descriptor["source_set_sha256"],
                }
                for name, descriptor in sorted(source_lock["plugins"].items())
            },
            "snapshots": plugin_snapshots,
        },
        "build_contract": BUILD_CONTRACT,
        "components": {
            "runtime": True,
            "opencl": True,
            "headers": True,
            "python": "base-reference",
            "triton": "base-reference",
            "llvm": "base-reference",
            "gem5": "workspace-reference",
        },
        "runtime_state_root": str(root / "env/product-state" / product_id),
        "inventory": inventory,
    }


def _product_artifacts(
    prefix: Path, *, content_prefix: Path | None = None
) -> dict[str, dict[str, Any]]:
    content_prefix = prefix if content_prefix is None else content_prefix
    versioned_runtime = sorted(
        path
        for path in (content_prefix / "lib").glob("libself_amdgpu_runtime.so.*.*.*")
        if path.is_file() and not path.is_symlink()
    )
    versioned_opencl = sorted(
        path
        for path in (content_prefix / "lib").glob("libOpenCL.so.*.*.*")
        if path.is_file() and not path.is_symlink()
    )
    if len(versioned_runtime) != 1 or len(versioned_opencl) != 1:
        raise ProductEnvironmentError("product runtime/OpenCL versioned library set is invalid")
    relative_paths = {
        "runtime_library": versioned_runtime[0].relative_to(content_prefix),
        "runtime_soname": Path("lib/libself_amdgpu_runtime.so.1"),
        "opencl_library": versioned_opencl[0].relative_to(content_prefix),
        "opencl_soname": Path("lib/libOpenCL.so.1"),
        "runtime_handshake": Path("bin/sagr-handshake"),
        "runtime_triton_probe": Path("bin/sagr-triton-hsaco-probe"),
        "opencl_executable": Path("bin/opencl-vecadd"),
        "runtime_endpoint": Path(
            "libexec/amdgpu-sim/generic-dispatch-v2-endpoint-test"
        ),
        "opencl_source": Path("share/self-amdgpu-runtime/opencl/vecadd.cl"),
        "runtime_header": Path("include/self_amdgpu_runtime/runtime.h"),
        "opencl_header": Path("include/CL/cl.h"),
        "python_bootstrap": Path("python/product_bootstrap.py"),
        "triton_plugin_driver": Path("python/triton/backends/gemsim_amd/driver.py"),
        "triton_plugin_compiler": Path(
            "python/triton/backends/gemsim_amd/compiler.py"
        ),
        "vllm_plugin_init": Path("python/gemsim_vllm/__init__.py"),
        "ccl_plugin_init": Path("python/gemsim_ccl/__init__.py"),
        "rocr_library": Path("lib/libhsa-runtime64.so.1"),
        "hip_library": Path("lib/libamdhip64.so.7"),
        "comgr_library": Path("lib/libamd_comgr.so.3"),
        "rccl_library": Path("lib/librccl.so.1"),
        "hsakmt_model_library": Path("lib/libself_amdgpu_hsakmt_model.so.1"),
        "topology_manifest": Path(
            "share/self-amdgpu-runtime/hsakmt-topology/manifest.json"
        ),
        "hip_header": Path("include/hip/hip_runtime.h"),
        "hsa_header": Path("include/hsa/hsa.h"),
        "hsakmt_header": Path("include/hsakmt/hsakmtmodeliface.h"),
        "rccl_header": Path("include/nccl.h"),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, relative in relative_paths.items():
        content_path = content_prefix / relative
        descriptor = _regular_sha256(
            content_path, allowed_symlink_root=content_prefix
        )
        descriptor["path"] = str(prefix / relative)
        descriptor.pop("resolved_path", None)
        result[name] = descriptor
    return result


def _publish_active(root: Path, prefix: Path, manifest_payload: bytes, product_id: str) -> None:
    active = {
        "schema": ACTIVE_SCHEMA,
        "product_id": product_id,
        "prefix": str(prefix),
        "manifest_sha256": sha256_bytes(manifest_payload),
    }
    _atomic_write(_active_path(root), canonical_json(active), mode=0o600, replace=True)


def build_product(root: Path, base_prefix: Path, product_id: str | None = None) -> Path:
    root = root.resolve(strict=True)
    base_prefix = _validate_base_prefix(root, base_prefix)
    source_lock, source_lock_payload = _load_frozen(root, product_id)
    product_id = source_lock["product_id"]
    target = _product_prefix(root, product_id)
    if target.exists() or target.is_symlink():
        raise ProductEnvironmentError(f"content-addressed product output must be absent: {target}")
    if source_lock["identity"]["base"]["prefix"] != str(base_prefix):
        raise ProductEnvironmentError("requested base prefix differs from frozen product base")
    _validate_frozen_inputs(root, source_lock)
    gem5_before = _workspace_inputs(root)
    rocm_root = _rocm_root(root)
    staging_root = rocm_root / f".product-stage.{product_id}.{os.getpid()}.{secrets.token_hex(8)}"
    candidate = rocm_root / (
        f".product-candidate.{product_id}.{os.getpid()}.{secrets.token_hex(8)}"
    )
    staging_root.mkdir(mode=0o700)
    staged_prefix = staging_root / target.relative_to("/")
    try:
        _build_runtime(root, source_lock, target, staging_root, staged_prefix)
        if not staged_prefix.is_dir() or staged_prefix.is_symlink():
            raise ProductEnvironmentError("CMake install did not create the staged product prefix")
        _copy_exclusive(
            _lock_path(root, product_id), staged_prefix / "source-lock.json", 0o400
        )
        plugin_snapshots = _snapshot_plugins(source_lock, staged_prefix, target)
        _write_python_launcher(
            staged_prefix / "python/product_bootstrap.py", source_lock
        )
        _write_activation(staged_prefix / "activate", target, base_prefix)
        _require_product_layout(staged_prefix)
        _normalize_product(staged_prefix)
        inventory = _inventory(staged_prefix)
        _verify_symlinks(staged_prefix, base_prefix, inventory)
        manifest = _manifest_document(
            root,
            target,
            staged_prefix,
            source_lock,
            source_lock_payload,
            inventory,
            plugin_snapshots,
        )
        manifest_payload = canonical_json(manifest)
        _atomic_write(
            staged_prefix / "manifest.json",
            manifest_payload,
            mode=0o400,
            replace=False,
        )
        if _workspace_inputs(root) != gem5_before:
            raise ProductEnvironmentError("gem5 binary/config changed during runtime build")
        _validate_frozen_inputs(root, source_lock)
        _rename_noreplace(staged_prefix, candidate)
        os.chmod(candidate, 0o555)
        _fsync_directory(candidate)
        _rename_noreplace(candidate, target)
        _fsync_directory(rocm_root)
    finally:
        if candidate.exists() and not candidate.is_symlink():
            os.chmod(candidate, 0o700)
            shutil.rmtree(candidate, ignore_errors=True)
        shutil.rmtree(staging_root, ignore_errors=True)
    try:
        verified_manifest = verify_product(root, prefix=target)
        _publish_active(root, target, verified_manifest, product_id)
    except Exception:
        # A failed candidate is never selected over the previously active product.
        raise
    return target


def _load_product_manifest(prefix: Path) -> tuple[dict[str, Any], bytes]:
    manifest, payload = _read_json(
        prefix / "manifest.json",
        canonical=True,
        private=True,
        label="product manifest",
    )
    product_id = manifest.get("product_id")
    if (
        manifest.get("schema") != PRODUCT_SCHEMA
        or manifest.get("setup_schema") != PRODUCT_SETUP_SCHEMA
        or not isinstance(product_id, str)
        or manifest.get("prefix") != str(prefix)
        or prefix.name != f"{PRODUCT_NAME_PREFIX}{product_id}"
        or not isinstance(manifest.get("identity"), dict)
        or _product_id(manifest["identity"]) != product_id
    ):
        raise ProductEnvironmentError("product manifest identity is invalid")
    return manifest, payload


def verify_product(
    root: Path,
    *,
    prefix: Path | None = None,
    product_id: str | None = None,
) -> bytes:
    root = root.resolve(strict=True)
    if prefix is None:
        prefix = _product_prefix(root, product_id) if product_id else active_product_prefix(root)
    prefix = prefix.absolute()
    if prefix.parent != _rocm_root(root) or prefix.is_symlink() or not prefix.is_dir():
        raise ProductEnvironmentError(f"product prefix is not a canonical direct child: {prefix}")
    manifest, manifest_payload = _load_product_manifest(prefix)
    source_lock, source_lock_payload = _read_json(
        prefix / "source-lock.json",
        canonical=True,
        private=True,
        label="installed product source lock",
    )
    if (
        source_lock.get("schema") != SOURCE_LOCK_SCHEMA
        or source_lock.get("product_id") != manifest["product_id"]
        or source_lock.get("identity") != manifest["identity"]
        or sha256_bytes(source_lock_payload) != manifest.get("source_lock", {}).get("sha256")
    ):
        raise ProductEnvironmentError("installed product source lock differs from manifest")
    if manifest.get("base") != manifest["identity"].get("base"):
        raise ProductEnvironmentError("product manifest base binding is inconsistent")
    if manifest.get("managed_inputs") != manifest["identity"].get("managed_inputs"):
        raise ProductEnvironmentError("product manifest managed inputs are inconsistent")
    if manifest.get("facade") != manifest["identity"].get("facade"):
        raise ProductEnvironmentError("product manifest facade binding is inconsistent")
    plugins = manifest.get("plugins")
    if (
        not isinstance(plugins, dict)
        or plugins.get("python_root") != str(prefix / "python")
        or plugins.get("bootstrap") != str(prefix / "python/product_bootstrap.py")
        or not isinstance(plugins.get("snapshots"), dict)
        or set(plugins["snapshots"]) != set(PLUGIN_SOURCES)
        or not isinstance(plugins.get("sources"), dict)
        or set(plugins["sources"]) != set(PLUGIN_SOURCES)
    ):
        raise ProductEnvironmentError("product plugin snapshot binding is invalid")
    expected_plugin_sources = {
        name: {
            "directory": descriptor["directory"],
            "file_count": descriptor["file_count"],
            "source_set_sha256": descriptor["source_set_sha256"],
        }
        for name, descriptor in sorted(source_lock["plugins"].items())
    }
    if plugins["sources"] != expected_plugin_sources:
        raise ProductEnvironmentError("product plugin source binding is invalid")
    for name, snapshot in plugins["snapshots"].items():
        if snapshot.get("source_set_sha256") != source_lock["plugins"][name].get(
            "source_set_sha256"
        ):
            raise ProductEnvironmentError(f"product plugin snapshot identity mismatch: {name}")
    if manifest.get("build_contract") != BUILD_CONTRACT:
        raise ProductEnvironmentError("product manifest build contract is invalid")
    _require_product_layout(prefix)
    actual_inventory = _inventory(prefix)
    if actual_inventory != manifest.get("inventory"):
        raise ProductEnvironmentError("product overlay inventory drifted")
    base_prefix = Path(manifest["base"]["prefix"])
    _verify_symlinks(prefix, base_prefix, actual_inventory)
    if manifest.get("artifacts") != _product_artifacts(prefix):
        raise ProductEnvironmentError("product artifact descriptors drifted")
    _validate_frozen_inputs(root, source_lock)
    return manifest_payload


def active_product_prefix(root: Path) -> Path:
    root = root.resolve(strict=True)
    active, _ = _read_json(
        _active_path(root), canonical=True, private=True, label="active product record"
    )
    if set(active) != {
        "schema",
        "product_id",
        "prefix",
        "manifest_sha256",
    } or active.get("schema") != ACTIVE_SCHEMA:
        raise ProductEnvironmentError("active product record schema is invalid")
    product_id = active.get("product_id")
    prefix_text = active.get("prefix")
    if not isinstance(product_id, str) or not isinstance(prefix_text, str):
        raise ProductEnvironmentError("active product record identity is invalid")
    expected = _product_prefix(root, product_id)
    prefix = Path(prefix_text)
    if prefix != expected or prefix.is_symlink() or not prefix.is_dir():
        raise ProductEnvironmentError("active product prefix is invalid")
    return prefix


def print_prefix(root: Path, base_prefix: Path) -> Path:
    try:
        return active_product_prefix(root)
    except FileNotFoundError:
        return _validate_base_prefix(root, base_prefix)
    except ProductEnvironmentError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            return _validate_base_prefix(root, base_prefix)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--print-prefix", action="store_true")
    modes.add_argument("--freeze-product", action="store_true")
    modes.add_argument("--product-runtime", action="store_true")
    modes.add_argument("--verify-product", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base-prefix", type=Path, required=True)
    parser.add_argument("--product-id")
    parser.add_argument("--jobs", type=int, choices=(BUILD_CONTRACT["parallel_jobs"],), default=24)
    arguments = parser.parse_args(argv)
    try:
        if arguments.print_prefix:
            result = print_prefix(arguments.root, arguments.base_prefix)
        elif arguments.freeze_product:
            result = freeze_product(arguments.root, arguments.base_prefix)
        elif arguments.product_runtime:
            result = build_product(
                arguments.root, arguments.base_prefix, arguments.product_id
            )
        else:
            payload = verify_product(arguments.root, product_id=arguments.product_id)
            manifest = json.loads(payload)
            result = Path(manifest["prefix"])
    except (OSError, ProductEnvironmentError, subprocess.CalledProcessError) as error:
        print(f"product environment error: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
