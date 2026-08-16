#!/usr/bin/env python3
"""Transparent clean-environment bootstrap for repository Triton examples."""

from __future__ import annotations

import os
import json
from pathlib import Path
import pwd
import stat
import subprocess
import sys


_RANK_DESCRIPTOR_MAX_BYTES = 64 * 1024
_PRODUCT_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
_PRODUCT_SCHEMA = "amdgpu-sim.product-prefix.v1"


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )


def _read_private_file(
    path: Path,
    *,
    maximum: int = _RANK_DESCRIPTOR_MAX_BYTES,
    label: str = "rank launch descriptor",
    private: bool = True,
) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{label} could not be opened safely") from error
    try:
        metadata = os.fstat(descriptor)
        forbidden_mode = (
            stat.S_IRWXG | stat.S_IRWXO
            if private
            else stat.S_IWGRP | stat.S_IWOTH
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & forbidden_mode
            or metadata.st_size <= 0
            or metadata.st_size > maximum
        ):
            raise RuntimeError(f"{label} is not a private owned file")
        chunks = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise RuntimeError(f"{label} was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _prefix(root: Path) -> Path:
    value = subprocess.check_output(
        [
            "/usr/bin/bash",
            str(root / "scripts/setup_rocm_env.sh"),
            "--print-prefix",
        ],
        cwd=root,
        text=True,
    ).strip()
    prefix = Path(value).resolve()
    python, _, _, _ = _execution_paths(prefix)
    if not python.is_file():
        raise RuntimeError(
            f"simulator Python is not installed for {prefix}; build schema 8, "
            "then freeze and publish the product runtime"
        )
    return prefix


def _execution_paths(prefix: Path) -> tuple[Path, Path, Path, Path | None]:
    """Return the interpreter, ROCm root, writable state, and plugin launcher."""
    prefix = prefix.resolve()
    manifest_path = prefix / "manifest.json"
    try:
        data = _read_private_file(
            manifest_path,
            maximum=_PRODUCT_MANIFEST_MAX_BYTES,
            label="product manifest",
            private=False,
        )
    except RuntimeError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            return prefix / "venv/bin/python", prefix, prefix, None
        raise
    try:
        document = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("product manifest is invalid JSON") from error
    if not isinstance(document, dict) or document.get("schema") != _PRODUCT_SCHEMA:
        # Schema-8 manifests predate canonical/private product control records.
        return prefix / "venv/bin/python", prefix, prefix, None
    metadata = manifest_path.lstat()
    if metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise RuntimeError("product manifest is not a private owned file")
    if data != _canonical_json(document) or document.get("prefix") != str(prefix):
        raise RuntimeError("product manifest is not canonical for this prefix")
    base = document.get("base")
    if not isinstance(base, dict):
        raise RuntimeError("product manifest base binding is invalid")
    base_text = base.get("prefix")
    python_record = base.get("python")
    state_text = document.get("runtime_state_root")
    plugins = document.get("plugins")
    launcher_text = plugins.get("bootstrap") if isinstance(plugins, dict) else None
    if (
        not isinstance(base_text, str)
        or not isinstance(python_record, dict)
        or not isinstance(python_record.get("path"), str)
        or not isinstance(state_text, str)
        or not isinstance(launcher_text, str)
    ):
        raise RuntimeError("product manifest execution paths are invalid")
    base_prefix = Path(base_text)
    python = Path(python_record["path"])
    state = Path(state_text)
    launcher = Path(launcher_text)
    if any(
        not candidate.is_absolute()
        or candidate != Path(os.path.normpath(candidate))
        for candidate in (base_prefix, python, state, launcher)
    ):
        raise RuntimeError("product manifest execution paths are not normalized")
    try:
        python.relative_to(base_prefix)
    except ValueError as error:
        raise RuntimeError("product Python escapes the schema-8 base prefix") from error
    try:
        launcher.relative_to(prefix)
    except ValueError as error:
        raise RuntimeError("product plugin launcher escapes the product prefix") from error
    if not launcher.is_file():
        raise RuntimeError("product plugin launcher is missing")
    return python, base_prefix, state, launcher


_RANK_PATH_KEYS = frozenset(
    {
        "instance_directory",
        "triton_cache_directory",
        "runtime_directory",
        "endpoint",
        "gem5_output_directory",
        "dispatch_trace_path",
        "gem5_log_path",
        "gem5_cache_directory",
    }
)
_UNIX_SOCKET_PATH_MAX_BYTES = 107
_UUID_TEXT_BYTES = 36


def _rank_descriptor() -> tuple[Path, Path, Path] | None:
    value = os.environ.get("GEMSIM_RANK_LAUNCH_DESCRIPTOR")
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(value)):
        raise RuntimeError("rank launch descriptor path must be absolute and normalized")
    data = _read_private_file(path)
    try:
        document = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("rank launch descriptor is invalid JSON") from error
    if not isinstance(document, dict) or data != _canonical_json(document):
        raise RuntimeError("rank launch descriptor is not canonical")
    if set(document) != {
        "schema",
        "job_uuid",
        "epoch",
        "rank",
        "world_size",
        "paths",
    } or document["schema"] != "amdgpu-sim.gemsim-rank-launch.v1":
        raise RuntimeError("rank launch descriptor schema is invalid")
    paths = document["paths"]
    if not isinstance(paths, dict) or set(paths) != _RANK_PATH_KEYS:
        raise RuntimeError("rank launch descriptor paths are invalid")
    normalized = {name: Path(value) for name, value in paths.items()}
    if any(
        not isinstance(paths[name], str)
        or not path.is_absolute()
        or path != Path(os.path.normpath(path))
        for name, path in normalized.items()
    ):
        raise RuntimeError("rank launch descriptor paths are invalid")
    instance = normalized["instance_directory"]
    runtime = normalized["runtime_directory"]
    cache = normalized["triton_cache_directory"]
    if (
        runtime.parent != instance
        or cache.parent.parent != instance.parent
        or any(
            normalized[name].parent != runtime
            for name in (
                "endpoint",
                "gem5_output_directory",
                "dispatch_trace_path",
                "gem5_log_path",
                "gem5_cache_directory",
            )
        )
    ):
        raise RuntimeError("rank launch descriptor path topology is invalid")
    world = document["world_size"]
    rank = document["rank"]
    if (
        not isinstance(world, int)
        or isinstance(world, bool)
        or not 2 <= world <= 16
        or not isinstance(rank, int)
        or isinstance(rank, bool)
        or not 0 <= rank < world
    ):
        raise RuntimeError("rank launch descriptor topology is invalid")
    ambient_cache = os.environ.get("TRITON_CACHE_DIR")
    if ambient_cache is not None and Path(ambient_cache).resolve() != cache:
        raise RuntimeError(
            "TRITON_CACHE_DIR does not match the immutable rank launch descriptor"
        )
    state = instance / "state"
    temporary = state / "tmp"
    if (
        len(os.fsencode(temporary)) + 1 + _UUID_TEXT_BYTES
        > _UNIX_SOCKET_PATH_MAX_BYTES
    ):
        raise RuntimeError(
            "rank-private TMPDIR cannot represent a vLLM IPC UUID within sun_path"
        )
    return path, cache, state


def _rank_descriptor_path() -> Path | None:
    descriptor = _rank_descriptor()
    return descriptor[0] if descriptor is not None else None


def _ccl_bootstrap_descriptor_path() -> Path | None:
    value = os.environ.get("GEMSIM_CCL_BOOTSTRAP_DESCRIPTOR")
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(value)):
        raise RuntimeError("CCL bootstrap descriptor path must be absolute and normalized")
    data = _read_private_file(
        path,
        maximum=1024 * 1024,
        label="CCL bootstrap descriptor",
    )
    try:
        document = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("CCL bootstrap descriptor is invalid JSON") from error
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "product", "groups"}
        or document.get("schema") != "amdgpu-sim.vllm-ccl-bootstrap.v1"
        or not isinstance(document.get("product"), dict)
        or not isinstance(document.get("groups"), list)
        or not document["groups"]
        or data != _canonical_json(document)
    ):
        raise RuntimeError("CCL bootstrap descriptor envelope is invalid")
    return path


def _runtime_environment(
    prefix: Path,
    cache: Path,
    rank_descriptor: Path | None = None,
    ccl_bootstrap_descriptor: Path | None = None,
    *,
    rocm_prefix: Path | None = None,
    state_root: Path | None = None,
) -> dict[str, str]:
    rocm_prefix = prefix if rocm_prefix is None else rocm_prefix
    state_root = prefix if state_root is None else state_root
    environment = {
        "HOME": str(state_root / "home"),
        "TMPDIR": str(state_root / "tmp"),
        "XDG_CACHE_HOME": str(state_root / "cache"),
        "XDG_CONFIG_HOME": str(state_root / "config"),
        "XDG_DATA_HOME": str(state_root / "data"),
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "ROCM_SIM_ROOT": str(prefix),
        "ROCM_PATH": str(rocm_prefix),
        "HIP_PATH": str(rocm_prefix),
        "HSA_PATH": str(rocm_prefix),
        "HSA_ENABLE_DXG_DETECTION": "0",
        "HSA_ENABLE_INTERRUPT": "0",
        "TRITON_DEFAULT_BACKEND": "gemsim_amd",
        "TRITON_CACHE_DIR": str(cache),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "_AMDGPU_SIM_BOOTSTRAPPED": "1",
    }
    # torch.compile's C++ wrapper needs the headers belonging to the exact
    # base interpreter used by the product.  The pinned base profile keeps
    # these under python-dev rather than /usr/include; expose them through
    # the clean child environment instead of relying on ambient CPATH.
    python_dev = rocm_prefix / "python-dev"
    include_candidates = sorted(
        path.parent
        for path in python_dev.glob("*/include/python*/Python.h")
        if path.is_file()
    )
    if len(include_candidates) == 1:
        include = str(include_candidates[0])
        environment["C_INCLUDE_PATH"] = include
        environment["CPLUS_INCLUDE_PATH"] = include
    if rank_descriptor is not None:
        environment["GEMSIM_RANK_LAUNCH_DESCRIPTOR"] = str(rank_descriptor)
    if ccl_bootstrap_descriptor is not None:
        environment["GEMSIM_CCL_BOOTSTRAP_DESCRIPTOR"] = str(
            ccl_bootstrap_descriptor
        )
    return environment


def bootstrap(script: str, cache_name: str) -> None:
    script_path = Path(script).resolve()
    root = script_path.parents[2]
    configured_prefix = os.environ.get("ROCM_SIM_ROOT")
    configured_cache = os.environ.get("TRITON_CACHE_DIR")
    descriptor = _rank_descriptor()
    rank_descriptor = descriptor[0] if descriptor is not None else None
    descriptor_cache = descriptor[1] if descriptor is not None else None
    descriptor_state = descriptor[2] if descriptor is not None else None
    ccl_bootstrap_descriptor = _ccl_bootstrap_descriptor_path()
    configured_execution = (
        _execution_paths(Path(configured_prefix).resolve())
        if configured_prefix
        else None
    )
    if (
        os.environ.get("_AMDGPU_SIM_BOOTSTRAPPED") == "1"
        and configured_prefix
        and configured_cache
        and configured_execution is not None
        and Path(sys.executable).resolve()
        == configured_execution[0].resolve()
        and dict(os.environ)
        == _runtime_environment(
            Path(configured_prefix).resolve(),
            Path(configured_cache).resolve(),
            rank_descriptor,
            ccl_bootstrap_descriptor,
            rocm_prefix=configured_execution[1],
            state_root=descriptor_state or configured_execution[2],
        )
    ):
        return

    prefix = _prefix(root)
    python, rocm_prefix, product_state_root, launcher = _execution_paths(prefix)
    state_root = descriptor_state or product_state_root
    user_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    cache = descriptor_cache or Path(
        os.environ.get(
            "TRITON_CACHE_DIR",
            str(user_home / ".cache/amdgpu-sim/triton" / cache_name),
        )
    ).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    for relative in ("home", "tmp", "cache", "config", "data"):
        path = state_root / relative
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)

    environment = _runtime_environment(
        prefix,
        cache,
        rank_descriptor,
        ccl_bootstrap_descriptor,
        rocm_prefix=rocm_prefix,
        state_root=state_root,
    )
    os.execve(
        python,
        [
            str(python),
            "-I",
            *([str(launcher)] if launcher is not None else []),
            str(script_path),
            *sys.argv[1:],
        ],
        environment,
    )
