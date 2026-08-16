#!/usr/bin/env python3
"""Rebuild and verify the private upstream ROCr/libhsakmt build input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse


SCHEMA = "amdgpu-sim.rocm-model-build.v1"
ROCM_SYSTEMS_HEAD = "92115a2941982a384de161be3f78cf9bff547027"
LOCK_RELATIVE = Path("config/conda-rocm-build-linux-64.lock")
PREFIX_RELATIVE = Path("env/conda/rocm-build-deps")
SOURCE_RELATIVE = Path("projects/rocm-systems/projects/rocr-runtime/libhsakmt")
REPOSITORY_RELATIVE = Path("projects/rocm-systems")
BUILD_RELATIVE = Path("projects/self-amdgpu-runtime/build/upstream-hsakmt-model")
LIBRARY_RELATIVE = Path("libhsakmt.a")
MANIFEST_RELATIVE = Path("manifest.json")


class BuildEnvironmentError(RuntimeError):
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
        raise BuildEnvironmentError(f"expected an owned regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.lstat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise BuildEnvironmentError(f"file changed while hashing: {path}")
    return {"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest()}


def run(
    arguments: list[str],
    *,
    environment: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=True,
        env=environment,
        text=True,
        capture_output=capture,
    )


def conda_executable() -> str:
    candidate = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if candidate is None:
        fallback = Path.home() / "miniforge3/condabin/conda"
        candidate = str(fallback) if fallback.is_file() else None
    if candidate is None:
        raise BuildEnvironmentError("conda executable is unavailable")
    return candidate


def locked_packages(lock: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for raw in lock.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line == "@EXPLICIT":
            continue
        url, separator, digest = line.partition("#")
        if not separator or len(digest) != 64:
            raise BuildEnvironmentError("build lock must use SHA-256 URL fragments")
        filename = Path(urlparse(url).path).name
        if not filename.endswith(".conda"):
            raise BuildEnvironmentError("build lock contains a non-conda artifact")
        stem = filename[:-6]
        name, version, build = stem.rsplit("-", 2)
        if name in result:
            raise BuildEnvironmentError(f"duplicate build dependency: {name}")
        result[name] = {
            "name": name,
            "version": version,
            "build": build,
            "url": url,
            "sha256": digest,
        }
    if not result:
        raise BuildEnvironmentError("build dependency lock is empty")
    return result


def verify_prefix(root: Path) -> dict[str, Any]:
    lock = root / LOCK_RELATIVE
    prefix = root / PREFIX_RELATIVE
    expected = locked_packages(lock)
    metadata_directory = prefix / "conda-meta"
    if not metadata_directory.is_dir() or metadata_directory.is_symlink():
        raise BuildEnvironmentError("private ROCm build prefix is unavailable")
    actual: dict[str, dict[str, str]] = {}
    for path in sorted(metadata_directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = payload.get("name")
        if not isinstance(name, str) or name in actual:
            raise BuildEnvironmentError("conda metadata has an invalid package name")
        actual[name] = {
            "name": name,
            "version": payload.get("version"),
            "build": payload.get("build"),
            "url": payload.get("url"),
            "sha256": payload.get("sha256"),
        }
    if actual != expected:
        raise BuildEnvironmentError("private ROCm build prefix differs from its lock")
    for relative in (
        "bin/pkg-config",
        "include/numa.h",
        "lib/pkgconfig/libdrm.pc",
        "lib/pkgconfig/libdrm_amdgpu.pc",
        "lib/libdrm.so",
        "lib/libdrm_amdgpu.so",
        "lib/libnuma.so",
    ):
        path = prefix / relative
        if not path.exists():
            raise BuildEnvironmentError(f"required build dependency is absent: {relative}")
    return {
        "prefix": str(prefix),
        "lock": file_record(lock),
        "packages": [expected[name] for name in sorted(expected)],
    }


def install_prefix(root: Path) -> dict[str, Any]:
    prefix = root / PREFIX_RELATIVE
    if prefix.exists() or prefix.is_symlink():
        return verify_prefix(root)
    prefix.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    run(
        [
            conda_executable(),
            "create",
            "--yes",
            "--prefix",
            str(prefix),
            "--file",
            str(root / LOCK_RELATIVE),
        ]
    )
    return verify_prefix(root)


def git(root: Path, *arguments: str) -> str:
    result = run(
        ["/usr/bin/git", "-C", str(root / REPOSITORY_RELATIVE), *arguments],
        capture=True,
    )
    if result.stderr:
        raise BuildEnvironmentError("git wrote stderr")
    return result.stdout.strip()


def source_identity(root: Path) -> dict[str, Any]:
    head = git(root, "rev-parse", "HEAD")
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if head != ROCM_SYSTEMS_HEAD or status:
        raise BuildEnvironmentError("pinned rocm-systems repository drifted")
    tree = git(root, "rev-parse", "HEAD^{tree}")
    return {
        "path": str(root / SOURCE_RELATIVE),
        "repository": str(root / REPOSITORY_RELATIVE),
        "head": head,
        "tree": tree,
        "clean": True,
    }


def isolated_environment(prefix: Path) -> dict[str, str]:
    return {
        "HOME": str(Path.home()),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PKG_CONFIG_PATH": str(prefix / "lib/pkgconfig"),
        "CMAKE_PREFIX_PATH": str(prefix),
    }


def tool_record(path: Path, version_argument: str = "--version") -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid not in {0, os.getuid()}
        or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        or not os.access(resolved, os.X_OK)
    ):
        raise BuildEnvironmentError(f"build tool is not a trusted executable: {path}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    result = run([str(resolved), version_argument], capture=True)
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "bytes": metadata.st_size,
        "sha256": digest.hexdigest(),
        "version": (result.stdout or result.stderr).splitlines()[0],
    }


def expected_manifest(root: Path, library: dict[str, Any]) -> dict[str, Any]:
    prefix = verify_prefix(root)
    return {
        "schema": SCHEMA,
        "source": source_identity(root),
        "dependencies": prefix,
        "tools": {
            "clang": tool_record(Path("/usr/bin/clang")),
            "cmake": tool_record(Path("/usr/bin/cmake")),
            "ninja": tool_record(Path("/usr/bin/ninja")),
        },
        "contract": {
            "build_type": "Release",
            "build_shared_libs": False,
            "hsakmt_werror": False,
            "export_to_user_package_registry": False,
        },
        "artifacts": {"libhsakmt": library},
    }


def build(root: Path, jobs: int) -> dict[str, Any]:
    prefix_record = verify_prefix(root)
    source_identity(root)
    prefix = Path(prefix_record["prefix"])
    build_directory = root / BUILD_RELATIVE
    build_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment = isolated_environment(prefix)
    run(
        [
            "/usr/bin/cmake",
            "-S",
            str(root / SOURCE_RELATIVE),
            "-B",
            str(build_directory),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_C_COMPILER=/usr/bin/clang",
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DHSAKMT_WERROR=OFF",
            "-DEXPORT_TO_USER_PACKAGE_REGISTRY=OFF",
        ],
        environment=environment,
    )
    run(
        [
            "/usr/bin/cmake",
            "--build",
            str(build_directory),
            "--parallel",
            str(jobs),
            "--target",
            "hsakmt",
        ],
        environment=environment,
    )
    source_identity(root)
    library = file_record(build_directory / LIBRARY_RELATIVE)
    document = expected_manifest(root, library)
    payload = canonical_json(document)
    manifest = build_directory / MANIFEST_RELATIVE
    temporary = build_directory / f".{MANIFEST_RELATIVE.name}.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest)
    finally:
        if temporary.exists():
            temporary.unlink()
    return document


def verify_build(root: Path) -> dict[str, Any]:
    build_directory = root / BUILD_RELATIVE
    manifest_path = build_directory / MANIFEST_RELATIVE
    payload = manifest_path.read_bytes()
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildEnvironmentError("upstream libhsakmt manifest is invalid") from error
    if not isinstance(document, dict) or canonical_json(document) != payload:
        raise BuildEnvironmentError("upstream libhsakmt manifest is not canonical")
    library = file_record(build_directory / LIBRARY_RELATIVE)
    expected = expected_manifest(root, library)
    if document != expected:
        raise BuildEnvironmentError("upstream libhsakmt build identity drifted")
    return document


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("install", "verify-prefix", "build", "verify-build"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    arguments = parser.parse_args()
    if arguments.jobs < 1 or arguments.jobs > 256:
        parser.error("--jobs must be in [1, 256]")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    root = arguments.root.resolve(strict=True)
    try:
        if arguments.mode == "install":
            result = install_prefix(root)
        elif arguments.mode == "verify-prefix":
            result = verify_prefix(root)
        elif arguments.mode == "build":
            result = build(root, arguments.jobs)
        else:
            result = verify_build(root)
    except (BuildEnvironmentError, OSError, subprocess.CalledProcessError) as error:
        print(f"rocm-model-build: {error}", file=sys.stderr)
        return 1
    print(canonical_json(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
