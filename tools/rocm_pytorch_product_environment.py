#!/usr/bin/env python3
"""Build and verify the private upstream ROCm PyTorch product.

The product keeps AMD's official gfx950 ROCm SDK, PyTorch, and Triton wheels
unchanged. Activation replaces only the ROCr provider at the KMD boundary;
AMD's official HIP, COMGR, RCCL, and math libraries remain above it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any
from zipfile import BadZipFile, ZipFile


SCHEMA = "amdgpu-sim.rocm-pytorch-product.v3"
IDENTITY_SCHEMA = "amdgpu-sim.rocm-pytorch-product-identity.v3"
ACTIVE_SCHEMA = "amdgpu-sim.active-rocm-pytorch-product.v1"
LOCK_SCHEMA = "amdgpu-sim.rocm-pytorch-wheel-lock.v1"
BASE_LOCK_RELATIVE = Path("config/conda-hip-facade-build-linux-64.lock")
WHEEL_LOCK_RELATIVE = Path("config/rocm-pytorch-gfx950-7.13.0a20260426.json")
VLLM_WHEEL_LOCK_RELATIVE = Path("config/rocm-pytorch-vllm-gfx950-7.2.3.json")
ROCM_723_DEB_LOCK_RELATIVE = Path("config/rocm-deb-sysroot-7.2.3-jammy-amd64.json")
ACTIVE_RELATIVE = Path("env/conda/active-rocm-pytorch")
PREFIX_NAME = "rocm-pytorch-v3-"
DEFAULT_PROFILE = "rocm713"
NATIVE_BOUNDARY_ARTIFACTS = (
    "runtime_library",
    "runtime_soname",
    "rocr_library",
    "hsakmt_model_library",
    "topology_manifest",
)
SDK_LIBRARY_SPECS = {
    "hip_library": ("amdhip64", "libamdhip64.so.7", "hipInit"),
    "comgr_library": ("amd_comgr", "libamd_comgr.so.3", "amd_comgr_get_version"),
    "rccl_library": ("rccl", "librccl.so.1", "ncclGetVersion"),
}
UPSTREAM_REPOSITORIES = {
    "triton": (Path("projects/triton"), "cd513e2798db0f4675b3d1205c8e76eb3381a0b3"),
    "vllm": (Path("projects/vllm"), "8d9b52f7c2514490bdadfd5eb0c931e58625df2e"),
}
PRODUCT_PROFILES = {
    "rocm713": {
        "wheel_lock": WHEEL_LOCK_RELATIVE,
        "rocm_version": "7.13.0a20260426",
        "required_packages": {
            "rocm",
            "torch",
            "triton",
            "rocm-sdk-core",
            "rocm-sdk-devel",
            "rocm-sdk-libraries-gfx950-dcgpu",
        },
        "url_prefixes": ("https://rocm.nightlies.amd.com/",),
        "provider": {"kind": "sdk-wheel"},
    },
    "vllm-rocm723": {
        "wheel_lock": VLLM_WHEEL_LOCK_RELATIVE,
        "rocm_version": "7.2.3",
        "required_packages": {
            "torch",
            "triton",
            "torchvision",
            "torchaudio",
            "vllm",
            "amdsmi",
            "flash-attn",
            "amd-aiter",
        },
        "url_prefixes": (
            "https://wheels.vllm.ai/",
            "https://files.pythonhosted.org/",
        ),
        "provider": {
            "kind": "signed-apt-sysroot",
            "lock": ROCM_723_DEB_LOCK_RELATIVE,
        },
        "system_runtime": "openmpi-jammy-v1",
    },
}

SYSTEM_RUNTIME_BUNDLES = {
    "none": (),
    "openmpi-jammy-v1": (
        {
            "name": "libevent-core-2.1-7",
            "version": "2.1.12-stable-1build3",
            "filename": "libevent-core-2.1-7_2.1.12-stable-1build3_amd64.deb",
            "bytes": 93926,
            "sha256": "9317ae74915b9e4372f7506fbd16ae2a021e0ead79a8d88f96dad49b87d94508",
        },
        {
            "name": "libevent-pthreads-2.1-7",
            "version": "2.1.12-stable-1build3",
            "filename": "libevent-pthreads-2.1-7_2.1.12-stable-1build3_amd64.deb",
            "bytes": 7642,
            "sha256": "078211a125a9ef5a9c7d24eb441761cefde8812b69889f7cee96824b0dc66831",
        },
        {
            "name": "libopenmpi3",
            "version": "4.1.2-2ubuntu1",
            "filename": "libopenmpi3_4.1.2-2ubuntu1_amd64.deb",
            "bytes": 2593592,
            "sha256": "25fc67e365e70abd06146962e97fbb0aead26403f36ba59b41f63954c768d352",
        },
    ),
}
SYSTEM_RUNTIME_CACHE = Path("artifacts/downloads/vllm-rocm-build")


def _load_deb_provider() -> Any:
    path = Path(__file__).resolve().with_name("rocm_deb_sysroot.py")
    spec = importlib.util.spec_from_file_location("amdgpu_sim_rocm_deb_sysroot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load ROCm Debian provider: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROCM_DEB_SYSROOT = _load_deb_provider()


class ProductError(RuntimeError):
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


def file_record(
    path: Path,
    *,
    allow_root: bool = False,
    allow_owner_group_write: bool = False,
) -> dict[str, Any]:
    before = path.lstat()
    allowed_uids = {os.getuid(), 0} if allow_root else {os.getuid()}
    forbidden_write_bits = stat.S_IWOTH
    if not allow_owner_group_write:
        forbidden_write_bits |= stat.S_IWGRP
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid not in allowed_uids
        or before.st_mode & forbidden_write_bits
        or before.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX)
    ):
        raise ProductError(f"expected a trusted regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.lstat()
    if (
        before.st_size,
        before.st_mtime_ns,
        before.st_ino,
        before.st_dev,
    ) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
        after.st_dev,
    ):
        raise ProductError(f"file changed while hashing: {path}")
    return {"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest()}


def read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductError(f"invalid canonical JSON: {path}") from error
    if not isinstance(document, dict) or canonical_json(document) != payload:
        raise ProductError(f"JSON is not a canonical object: {path}")
    return document, payload


def verify_artifact_descriptor(descriptor: dict[str, Any], prefix: Path) -> None:
    path = Path(descriptor.get("path", ""))
    metadata = path.lstat()
    symlink_target = descriptor.get("symlink_target")
    if symlink_target is None:
        if stat.S_ISLNK(metadata.st_mode):
            raise ProductError(f"unexpected artifact symlink: {path}")
        target = path
    else:
        if (
            not isinstance(symlink_target, str)
            or Path(symlink_target).name != symlink_target
            or not stat.S_ISLNK(metadata.st_mode)
            or os.readlink(path) != symlink_target
        ):
            raise ProductError(f"artifact symlink differs: {path}")
        target = path.resolve(strict=True)
        try:
            target.relative_to(prefix)
        except ValueError as error:
            raise ProductError(f"artifact symlink escapes product: {path}") from error
    actual = file_record(target)
    if actual["bytes"] != descriptor.get("bytes") or actual["sha256"] != descriptor.get("sha256"):
        raise ProductError(f"native facade artifact drifted: {path.name}")


def git(root: Path, repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(root / repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stderr:
        raise ProductError(f"git wrote stderr for {repository}")
    return result.stdout.strip()


def upstream_identity(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (relative, expected_head) in UPSTREAM_REPOSITORIES.items():
        head = git(root, relative, "rev-parse", "HEAD")
        status = git(root, relative, "status", "--porcelain=v1", "--untracked-files=all")
        if head != expected_head or status:
            raise ProductError(f"pinned upstream repository drifted: {name}")
        result[name] = {
            "path": str(root / relative),
            "head": head,
            "tree": git(root, relative, "rev-parse", "HEAD^{tree}"),
            "clean": True,
        }
    build_requirements = root / "projects/vllm/requirements/build/rocm.txt"
    result["vllm"]["rocm_build_requirements"] = file_record(build_requirements)
    return result


def active_native_product(root: Path) -> tuple[Path, dict[str, Any], bytes]:
    active, _ = read_canonical(root / "env/rocm/active-product")
    if active.get("schema") != "amdgpu-sim.active-product.v1":
        raise ProductError("active native product schema is invalid")
    prefix = Path(active.get("prefix", ""))
    manifest, payload = read_canonical(prefix / "manifest.json")
    if (
        manifest.get("schema") != "amdgpu-sim.product-prefix.v1"
        or manifest.get("prefix") != str(prefix)
        or manifest.get("product_id") != active.get("product_id")
        or sha256_bytes(payload) != active.get("manifest_sha256")
    ):
        raise ProductError("active native product identity is invalid")
    for name in NATIVE_BOUNDARY_ARTIFACTS:
        descriptor = manifest.get("artifacts", {}).get(name)
        if not isinstance(descriptor, dict):
            raise ProductError(f"native product lacks ROCr boundary artifact: {name}")
        verify_artifact_descriptor(descriptor, prefix)
    return prefix, manifest, payload


def profile_spec(profile: str) -> dict[str, Any]:
    spec = PRODUCT_PROFILES.get(profile)
    if spec is None:
        raise ProductError(f"unknown ROCm PyTorch product profile: {profile}")
    return spec


def system_runtime_identity(root: Path, profile: str) -> dict[str, Any]:
    bundle = profile_spec(profile).get("system_runtime", "none")
    packages = SYSTEM_RUNTIME_BUNDLES.get(bundle)
    if packages is None:
        raise ProductError(f"unknown system runtime bundle: {bundle}")
    records: list[dict[str, Any]] = []
    for package in packages:
        path = root / SYSTEM_RUNTIME_CACHE / package["filename"]
        actual = file_record(path)
        if actual["bytes"] != package["bytes"] or actual["sha256"] != package["sha256"]:
            raise ProductError(f"system runtime package differs: {package['name']}")
        records.append({**package, "path": str(path)})
    return {"kind": bundle, "cache_directory": str(root / SYSTEM_RUNTIME_CACHE), "packages": records}


def wheel_lock(root: Path, profile: str = DEFAULT_PROFILE) -> tuple[dict[str, Any], bytes]:
    spec = profile_spec(profile)
    lock_relative = spec["wheel_lock"]
    document, payload = read_canonical(root / lock_relative)
    if (
        document.get("schema") != LOCK_SCHEMA
        or document.get("python") != {"implementation": "CPython", "version": "3.12", "abi": "cp312"}
        or document.get("architecture") != "gfx950"
        or document.get("rocm_version") != spec["rocm_version"]
        or not isinstance(document.get("packages"), list)
    ):
        raise ProductError("ROCm PyTorch wheel lock contract is invalid")
    cache_relative = document.get("cache_directory")
    if (
        not isinstance(cache_relative, str)
        or not cache_relative
        or Path(cache_relative).is_absolute()
        or ".." in Path(cache_relative).parts
    ):
        raise ProductError("wheel lock cache directory is invalid")
    names: set[str] = set()
    filenames: set[str] = set()
    required = spec["required_packages"]
    for package in document["packages"]:
        if not isinstance(package, dict) or set(package) not in (
            {"name", "version", "filename", "url", "bytes", "sha256"},
            {"name", "version", "filename", "url", "bytes", "sha256", "built_from"},
        ):
            raise ProductError("wheel lock package record is invalid")
        name = package["name"]
        if not isinstance(name, str) or name != name.lower() or "_" in name or name in names:
            raise ProductError("wheel lock package names are invalid")
        names.add(name)
        filename = package["filename"]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".whl")
            or filename in filenames
            or not isinstance(package["version"], str)
            or not package["version"]
            or not isinstance(package["url"], str)
            or not any(package["url"].startswith(prefix) for prefix in spec["url_prefixes"])
            or not isinstance(package["bytes"], int)
            or package["bytes"] <= 0
            or not isinstance(package["sha256"], str)
            or len(package["sha256"]) != 64
        ):
            raise ProductError(f"wheel lock artifact is invalid: {name}")
        filenames.add(filename)
        built_from = package.get("built_from")
        if built_from is not None:
            if name != "rocm" or built_from != {
                "kind": "upstream-sdist",
                "url": "https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/rocm-7.13.0a20260426.tar.gz",
                "bytes": 17787,
                "sha256": "02dfcb9374a27e12d005a91f0cde032bafce49948856fbe96a4cbcff878ddaac",
                "target_family": "gfx950-dcgpu",
                "python": "3.12.3",
                "setuptools": "83.0.0",
                "wheel": "0.47.0",
            }:
                raise ProductError("ROCm meta-package build provenance is invalid")
        elif name == "rocm" and profile == "rocm713":
            raise ProductError("ROCm meta-package lacks build provenance")
        cache = root / cache_relative / filename
        actual = file_record(cache)
        if actual["bytes"] != package["bytes"] or actual["sha256"] != package["sha256"]:
            raise ProductError(f"wheel cache differs from lock: {name}")
        try:
            with ZipFile(cache) as archive:
                metadata_paths = [
                    path
                    for path in archive.namelist()
                    if path.endswith(".dist-info/METADATA") and path.count("/") == 1
                ]
                if len(metadata_paths) != 1:
                    raise ProductError(f"wheel metadata count is invalid: {name}")
                metadata = archive.read(metadata_paths[0]).decode("utf-8")
        except (BadZipFile, UnicodeDecodeError, OSError) as error:
            raise ProductError(f"wheel metadata is unreadable: {name}") from error
        fields: dict[str, str] = {}
        for line in metadata.splitlines():
            if line.startswith("Name: "):
                fields["name"] = line.removeprefix("Name: ").lower().replace("_", "-")
            elif line.startswith("Version: "):
                fields["version"] = line.removeprefix("Version: ")
            if set(fields) == {"name", "version"}:
                break
        if fields != {"name": name, "version": package["version"]}:
            raise ProductError(f"wheel metadata differs from lock: {name}")
    if not required.issubset(names):
        raise ProductError("wheel lock lacks a required upstream package")
    return document, payload


def provider_identity(root: Path, profile: str) -> dict[str, Any]:
    provider = profile_spec(profile)["provider"]
    if provider["kind"] == "sdk-wheel":
        return {"kind": "sdk-wheel"}
    if provider["kind"] != "signed-apt-sysroot":
        raise ProductError("unsupported ROCm provider")
    lock_relative = provider["lock"]
    try:
        lock, payload = ROCM_DEB_SYSROOT.load_lock(
            root, root / lock_relative, require_packages=True
        )
    except ROCM_DEB_SYSROOT.SysrootError as error:
        raise ProductError(f"ROCm Debian provider is invalid: {error}") from error
    return {
        "kind": "signed-apt-sysroot",
        "implementation": file_record(Path(ROCM_DEB_SYSROOT.__file__).resolve()),
        "lock": {
            "path": str(root / lock_relative),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "document": lock,
        },
    }


def identity(
    root: Path, profile: str = DEFAULT_PROFILE
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = profile_spec(profile)
    native_prefix, native, native_payload = active_native_product(root)
    wheels, wheel_payload = wheel_lock(root, profile)
    document = {
        "schema": IDENTITY_SCHEMA,
        "profile": profile,
        "builder": file_record(Path(__file__).resolve()),
        "base_lock": file_record(root / BASE_LOCK_RELATIVE),
        "wheel_lock": {
            "path": str(root / spec["wheel_lock"]),
            "bytes": len(wheel_payload),
            "sha256": sha256_bytes(wheel_payload),
        },
        "wheel_set": wheels,
        "rocm_provider": provider_identity(root, profile),
        "system_runtime": system_runtime_identity(root, profile),
        "native_product": {
            "prefix": str(native_prefix),
            "product_id": native["product_id"],
            "manifest_sha256": sha256_bytes(native_payload),
            "role": "rocr_kmd_boundary_only",
            "artifacts": {
                name: native["artifacts"][name]
                for name in NATIVE_BOUNDARY_ARTIFACTS
            },
        },
        "upstream": upstream_identity(root),
    }
    return document, native


def product_id(document: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(document))


def product_prefix(root: Path, identifier: str) -> Path:
    return root / "env/conda" / f"{PREFIX_NAME}{identifier}"


def conda_executable() -> str:
    candidate = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not candidate:
        fallback = Path.home() / "miniforge3/condabin/conda"
        candidate = str(fallback) if fallback.is_file() else None
    if not candidate:
        raise ProductError("conda executable is unavailable")
    return candidate


def run(
    arguments: list[str],
    *,
    environment: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            check=True,
            env=environment,
            capture_output=capture,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        suffix = f": {detail[-8000:]}" if detail else ""
        raise ProductError(
            f"command failed with exit {error.returncode}: {arguments[0]}{suffix}"
        ) from error


def tree_summary(prefix: Path, *, excluded: set[str]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for current, directories, files in os.walk(prefix, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(name for name in directories if name not in {"__pycache__", ".pytest_cache"})
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(prefix).as_posix()
            if relative in excluded or name.endswith((".pyc", ".pyo")):
                continue
            metadata = path.lstat()
            if stat.S_ISREG(metadata.st_mode):
                record = file_record(path, allow_owner_group_write=True)
                core = {
                    "path": relative,
                    "kind": "regular",
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
                total += metadata.st_size
            elif stat.S_ISLNK(metadata.st_mode):
                core = {"path": relative, "kind": "symlink", "target": os.readlink(path)}
            else:
                raise ProductError(f"special file in product prefix: {path}")
            digest.update(canonical_json(core))
            count += 1
    return {"file_count": count, "regular_bytes": total, "sha256": digest.hexdigest()}


def sdk_library_records(prefix: Path) -> dict[str, dict[str, Any]]:
    shortnames = [spec[0] for spec in SDK_LIBRARY_SPECS.values()]
    script = r"""
import json
import rocm_sdk
import sys

shortnames = json.loads(sys.argv[1])
paths = rocm_sdk.find_libraries(*shortnames)
assert len(paths) == len(shortnames)
print(json.dumps(dict(zip(shortnames, map(str, paths))), sort_keys=True, separators=(",", ":")))
"""
    environment = {
        "HOME": str(prefix),
        "PATH": f"{prefix}/bin:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ROCM_SDK_TARGET_FAMILY": "gfx950-dcgpu",
        "LC_ALL": "C",
    }
    result = run(
        [
            str(prefix / "bin/python"),
            "-I",
            "-c",
            script,
            json.dumps(shortnames, separators=(",", ":")),
        ],
        environment=environment,
        capture=True,
    )
    try:
        located = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ProductError("ROCm SDK library locator did not return JSON") from error
    if not isinstance(located, dict) or set(located) != set(shortnames):
        raise ProductError("ROCm SDK library locator result is invalid")

    records: dict[str, dict[str, Any]] = {}
    for role, (shortname, soname, symbol) in SDK_LIBRARY_SPECS.items():
        raw_path = located.get(shortname)
        if not isinstance(raw_path, str):
            raise ProductError(f"ROCm SDK library path is invalid: {shortname}")
        path = Path(raw_path)
        if not path.is_absolute() or path.name != soname:
            raise ProductError(f"ROCm SDK selected an unexpected library: {shortname}")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(prefix)
        except ValueError as error:
            raise ProductError(f"ROCm SDK library escapes the private product: {shortname}") from error
        record = file_record(resolved, allow_owner_group_write=True)
        records[role] = {
            **record,
            "provider": "official_rocm_sdk",
            "shortname": shortname,
            "soname": soname,
            "symbol": symbol,
        }
    return records


def apt_sysroot_library_records(prefix: Path, sdk_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for role, (shortname, soname, symbol) in SDK_LIBRARY_SPECS.items():
        candidates = sorted(
            path for path in sdk_root.rglob(soname) if path.is_file() or path.is_symlink()
        )
        resolved = {path.resolve(strict=True) for path in candidates}
        if len(resolved) != 1:
            raise ProductError(
                f"ROCm Debian sysroot library set is ambiguous: {soname}"
            )
        path = resolved.pop()
        try:
            path.relative_to(prefix)
        except ValueError as error:
            raise ProductError(f"ROCm Debian library escapes product: {soname}") from error
        record = file_record(path, allow_owner_group_write=True)
        records[role] = {
            **record,
            "provider": "official_rocm_apt_sysroot",
            "shortname": shortname,
            "soname": soname,
            "symbol": symbol,
        }
    return records


def provider_library_records(
    prefix: Path, sdk_root: Path, provider: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if provider["kind"] == "sdk-wheel":
        return sdk_library_records(prefix)
    if provider["kind"] == "signed-apt-sysroot":
        return apt_sysroot_library_records(prefix, sdk_root)
    raise ProductError("unsupported ROCm provider")


def install_system_runtime(
    root: Path, prefix: Path, runtime: dict[str, Any]
) -> dict[str, Any]:
    kind = runtime.get("kind")
    if kind == "none":
        return {"kind": "none"}
    if kind not in SYSTEM_RUNTIME_BUNDLES:
        raise ProductError("unsupported system runtime bundle")
    destination = prefix / "system-runtime"
    if destination.exists() or destination.is_symlink():
        raise ProductError(f"system runtime destination is not absent: {destination}")
    destination.mkdir(mode=0o755, parents=True)
    for package in runtime["packages"]:
        source = Path(package["path"])
        run(["/usr/bin/dpkg-deb", "--extract", str(source), str(destination)])
    return {
        "schema": "amdgpu-sim.system-runtime-install.v1",
        "kind": kind,
        "packages": [package["name"] for package in runtime["packages"]],
        "tree": tree_summary(destination, excluded=set()),
    }


def verify_system_runtime_install(
    prefix: Path, runtime: dict[str, Any], installed: dict[str, Any]
) -> Path | None:
    kind = runtime.get("kind")
    if kind == "none":
        if installed != {"kind": "none"}:
            raise ProductError("system runtime installation differs")
        return None
    destination = prefix / "system-runtime"
    expected = {
        "schema": "amdgpu-sim.system-runtime-install.v1",
        "kind": kind,
        "packages": [package["name"] for package in runtime["packages"]],
        "tree": tree_summary(destination, excluded=set()),
    }
    if expected != installed:
        raise ProductError("system runtime installation differs")
    return destination


def product_environment(
    prefix: Path,
    native: dict[str, Any],
    sdk_root: Path,
    sdk_libraries: dict[str, dict[str, Any]],
    state: Path,
    system_runtime_root: Path | None = None,
) -> dict[str, str]:
    native_prefix = Path(native["prefix"])
    library_directories = [
        Path(sdk_libraries[name]["path"]).parent
        for name in ("hip_library", "comgr_library", "rccl_library")
    ]
    library_directories.extend(
        (
            sdk_root / "lib",
            sdk_root / "lib/rocm_sysdeps/lib",
            prefix / "lib",
            native_prefix / "lib",
        )
    )
    if system_runtime_root is not None:
        library_directories.insert(0, system_runtime_root / "usr/lib/x86_64-linux-gnu")
    if sdk_libraries["hip_library"].get("provider") == "official_rocm_apt_sysroot":
        sysroot = sdk_root.parents[1]
        library_directories.extend(
            (
                sysroot / "usr/lib/x86_64-linux-gnu",
                sysroot / "lib/x86_64-linux-gnu",
            )
        )
    library_path = ":".join(str(path) for path in dict.fromkeys(library_directories))
    pkg_config_directories = (
        sdk_root / "lib/rocm_sysdeps/lib/pkgconfig",
        sdk_root / "lib/pkgconfig",
        sdk_root / "share/pkgconfig",
        prefix / "lib/pkgconfig",
        prefix / "share/pkgconfig",
    )
    pkg_config_path = ":".join(
        str(path) for path in dict.fromkeys(pkg_config_directories)
    )
    rocr_library = native["artifacts"]["rocr_library"]["path"]
    model_library = native["artifacts"]["hsakmt_model_library"]["path"]
    topology_directory = Path(native["artifacts"]["topology_manifest"]["path"]).parent
    return {
        "HOME": str(state / "home"),
        "TMPDIR": str(state / "tmp"),
        "XDG_CACHE_HOME": str(state / "xdg-cache"),
        "PATH": f"{prefix}/bin:{native_prefix}/bin:{sdk_root}/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": library_path,
        "PKG_CONFIG_PATH": pkg_config_path,
        "LD_PRELOAD": rocr_library,
        "ROCM_SIM_ROOT": str(native_prefix),
        "ROCM_PATH": str(sdk_root),
        "HIP_PATH": str(sdk_root),
        "HSA_PATH": str(native_prefix),
        "HIP_PLATFORM": "amd",
        "HIP_CLANG_PATH": str(sdk_root / "lib/llvm/bin"),
        "PYTORCH_ROCM_ARCH": "gfx950",
        "ROCM_SDK_TARGET_FAMILY": "gfx950-dcgpu",
        "HSA_ENABLE_DXG_DETECTION": "0",
        # Model-backed VRAM is not host-accessible; use the standard ROCr blit AQL path.
        "HSA_ENABLE_DTIF_FAST_COPY": "0",
        "HSA_ENABLE_INTERRUPT": "0",
        "HSA_MODEL_LIB": model_library,
        "HSA_MODEL_TOPOLOGY": str(topology_directory),
        "TRITON_CACHE_DIR": str(state / "triton-cache"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "LC_ALL": "C",
    }


def write_activation(
    prefix: Path,
    native: dict[str, Any],
    sdk_root: Path,
    sdk_libraries: dict[str, dict[str, Any]],
    state: Path,
    system_runtime_root: Path | None = None,
) -> None:
    environment = product_environment(
        prefix, native, sdk_root, sdk_libraries, state, system_runtime_root
    )
    selected = {
        key: environment[key]
        for key in (
            "PATH", "LD_LIBRARY_PATH", "PKG_CONFIG_PATH", "LD_PRELOAD", "ROCM_SIM_ROOT", "ROCM_PATH", "HIP_PATH", "HSA_PATH",
            "HIP_PLATFORM", "HIP_CLANG_PATH", "PYTORCH_ROCM_ARCH", "HSA_ENABLE_DXG_DETECTION",
            "HSA_ENABLE_DTIF_FAST_COPY", "HSA_ENABLE_INTERRUPT", "HSA_MODEL_LIB", "HSA_MODEL_TOPOLOGY", "ROCM_SDK_TARGET_FAMILY", "TRITON_CACHE_DIR",
            "PYTHONNOUSERSITE",
        )
    }
    activate = prefix / "etc/conda/activate.d/amdgpu-sim-rocm-pytorch.sh"
    deactivate = prefix / "etc/conda/deactivate.d/amdgpu-sim-rocm-pytorch.sh"
    activate.parent.mkdir(parents=True, exist_ok=True)
    deactivate.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Generated by tools/rocm_pytorch_product_environment.py."]
    restore = ["# Generated by tools/rocm_pytorch_product_environment.py."]
    for name, value in selected.items():
        lines.extend(
            (
                f'if [[ ${{{name}+x}} ]]; then export _AMDGPU_SIM_ROCM_OLD_{name}="${{{name}}}"; else unset _AMDGPU_SIM_ROCM_OLD_{name}; fi',
                f"export {name}={value}",
            )
        )
        restore.extend(
            (
                f'if [[ ${{_AMDGPU_SIM_ROCM_OLD_{name}+x}} ]]; then export {name}="${{_AMDGPU_SIM_ROCM_OLD_{name}}}"; else unset {name}; fi',
                f"unset _AMDGPU_SIM_ROCM_OLD_{name}",
            )
        )
    activate.write_text("\n".join(lines) + "\n", encoding="ascii")
    deactivate.write_text("\n".join(restore) + "\n", encoding="ascii")


def sdk_root_for(prefix: Path) -> Path:
    executable = prefix / "bin/rocm-sdk"
    result = run([str(executable), "path", "--root"], capture=True)
    candidate = Path(result.stdout.strip()).resolve(strict=True)
    try:
        candidate.relative_to(prefix)
    except ValueError as error:
        raise ProductError("ROCm SDK root escapes the private product") from error
    return candidate


def install_rocm_provider(
    root: Path,
    prefix: Path,
    provider: dict[str, Any],
    clean: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
    if provider["kind"] == "sdk-wheel":
        run([str(prefix / "bin/rocm-sdk"), "init"], environment=clean)
        sdk_root = sdk_root_for(prefix)
        return sdk_root, {"kind": "sdk-wheel", "sdk_root": str(sdk_root)}
    if provider["kind"] != "signed-apt-sysroot":
        raise ProductError("unsupported ROCm provider")
    destination = prefix / "rocm-sysroot"
    try:
        installed = ROCM_DEB_SYSROOT.materialize_sysroot(
            root, provider["lock"]["document"], destination
        )
    except ROCM_DEB_SYSROOT.SysrootError as error:
        raise ProductError(f"ROCm Debian sysroot installation failed: {error}") from error
    sdk_root = Path(installed["sdk_root"]).resolve(strict=True)
    try:
        sdk_root.relative_to(prefix)
    except ValueError as error:
        raise ProductError("ROCm Debian SDK root escapes product") from error
    return sdk_root, installed


def verify_provider_install(
    prefix: Path,
    sdk_root: Path,
    provider: dict[str, Any],
    installed: dict[str, Any],
) -> None:
    if not isinstance(installed, dict):
        raise ProductError("ROCm provider installation record is invalid")
    if provider["kind"] == "sdk-wheel":
        if installed != {"kind": "sdk-wheel", "sdk_root": str(sdk_root)}:
            raise ProductError("ROCm SDK wheel installation record differs")
        return
    if provider["kind"] != "signed-apt-sysroot":
        raise ProductError("unsupported ROCm provider")
    sysroot = prefix / "rocm-sysroot"
    expected = {
        "schema": "amdgpu-sim.rocm-deb-sysroot-install.v1",
        "sdk_root": str(sdk_root),
        "packages": [
            package["name"] for package in provider["lock"]["document"]["packages"]
        ],
        "owned_path_count": installed.get("owned_path_count"),
        "tree": ROCM_DEB_SYSROOT.tree_summary(sysroot),
    }
    if not isinstance(expected["owned_path_count"], int) or expected != installed:
        raise ProductError("ROCm Debian sysroot installation record differs")


def probe_product(
    prefix: Path,
    native: dict[str, Any],
    sdk_root: Path,
    sdk_libraries: dict[str, dict[str, Any]],
    state: Path,
    expected_versions: dict[str, str],
    system_runtime_root: Path | None = None,
) -> dict[str, Any]:
    runtime_symbols = {
        sdk_libraries["hip_library"]["soname"]: {
            "path": sdk_libraries["hip_library"]["path"],
            "provider": sdk_libraries["hip_library"]["provider"],
            "symbol": sdk_libraries["hip_library"]["symbol"],
        },
        "libhsa-runtime64.so.1": {
            "path": native["artifacts"]["rocr_library"]["path"],
            "provider": "amdgpu_sim_rocr_boundary",
            "symbol": "hsa_init",
        },
        sdk_libraries["comgr_library"]["soname"]: {
            "path": sdk_libraries["comgr_library"]["path"],
            "provider": sdk_libraries["comgr_library"]["provider"],
            "symbol": sdk_libraries["comgr_library"]["symbol"],
        },
        sdk_libraries["rccl_library"]["soname"]: {
            "path": sdk_libraries["rccl_library"]["path"],
            "provider": sdk_libraries["rccl_library"]["provider"],
            "symbol": sdk_libraries["rccl_library"]["symbol"],
        },
    }
    script = r"""
import ctypes
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

expected_versions = json.loads(sys.argv[1])
expected_libraries = json.loads(sys.argv[2])
prefix = Path(sys.argv[3]).resolve(strict=True)
actual_versions = {name: metadata.version(name) for name in expected_versions}
assert actual_versions == expected_versions, (actual_versions, expected_versions)
import torch
import triton
assert torch.version.hip is not None

class DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]

libdl = ctypes.CDLL("libdl.so.2")
libdl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
libdl.dladdr.restype = ctypes.c_int
loaded = {}
for soname, expected in expected_libraries.items():
    symbol_name = expected["symbol"]
    expected_path = expected["path"]
    library = ctypes.CDLL(soname, mode=os.RTLD_NOW | os.RTLD_LOCAL)
    symbol = getattr(library, symbol_name)
    info = DlInfo()
    assert libdl.dladdr(ctypes.cast(symbol, ctypes.c_void_p), ctypes.byref(info)) != 0
    actual_path = str(Path(info.dli_fname.decode()).resolve(strict=True))
    assert actual_path == str(Path(expected_path).resolve(strict=True)), (
        soname,
        actual_path,
        expected_path,
    )
    loaded[soname] = {
        "path": actual_path,
        "provider": expected["provider"],
        "symbol": symbol_name,
    }

torch_path = Path(torch.__file__).resolve(strict=True)
triton_path = Path(triton.__file__).resolve(strict=True)
torch_path.relative_to(prefix)
triton_path.relative_to(prefix)
print(json.dumps({
    "runtime_libraries": loaded,
    "torch": torch.__version__,
    "torch_file": str(torch_path),
    "torch_git": torch.version.git_version,
    "torch_hip": torch.version.hip,
    "triton": triton.__version__,
    "triton_file": str(triton_path),
}, sort_keys=True, separators=(",", ":")))
"""
    result = run(
        [
            str(prefix / "bin/python"),
            "-I",
            "-c",
            script,
            json.dumps(expected_versions, sort_keys=True),
            json.dumps(runtime_symbols, sort_keys=True),
            str(prefix),
        ],
        environment=product_environment(
            prefix, native, sdk_root, sdk_libraries, state, system_runtime_root
        ),
        capture=True,
    )
    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ProductError("ROCm PyTorch probe did not return JSON") from error
    if not isinstance(probe, dict):
        raise ProductError("ROCm PyTorch probe result is invalid")
    return probe


def build(root: Path, profile: str = DEFAULT_PROFILE) -> Path:
    document, native = identity(root, profile)
    identifier = product_id(document)
    prefix = product_prefix(root, identifier)
    state = root / "env/conda-state/rocm-pytorch" / identifier
    manifest_path = prefix / "amdgpu-sim-rocm-pytorch-manifest.json"
    if manifest_path.is_file():
        verify(root, prefix, profile)
        publish_active(root, prefix, identifier, profile)
        return prefix
    if prefix.exists() or prefix.is_symlink():
        raise ProductError(f"incomplete product requires inspection: {prefix}")
    prefix.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    for path in (state / "home", state / "tmp", state / "xdg-cache", state / "triton-cache"):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    run(
        [
            conda_executable(), "create", "--yes", "--offline", "--prefix", str(prefix),
            "--file", str(root / BASE_LOCK_RELATIVE),
        ]
    )
    wheels = document["wheel_set"]
    cache = root / wheels["cache_directory"]
    artifacts = [str(cache / package["filename"]) for package in wheels["packages"]]
    clean = {
        "HOME": str(state / "home"),
        "TMPDIR": str(state / "tmp"),
        "PATH": f"{prefix}/bin:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "LC_ALL": "C",
    }
    python = prefix / "bin/python"
    run(
        [str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", *artifacts],
        environment=clean,
    )
    sdk_root, provider_install = install_rocm_provider(
        root, prefix, document["rocm_provider"], clean
    )
    system_runtime_install = install_system_runtime(
        root, prefix, document["system_runtime"]
    )
    system_runtime_root = verify_system_runtime_install(
        prefix, document["system_runtime"], system_runtime_install
    )
    sdk_libraries = provider_library_records(
        prefix, sdk_root, document["rocm_provider"]
    )
    write_activation(
        prefix, native, sdk_root, sdk_libraries, state, system_runtime_root
    )
    run([str(python), "-I", "-m", "pip", "check"], environment=clean)
    expected_versions = {
        package["name"]: package["version"] for package in document["wheel_set"]["packages"]
    }
    runtime_probe = probe_product(
        prefix,
        native,
        sdk_root,
        sdk_libraries,
        state,
        expected_versions,
        system_runtime_root,
    )
    installed = tree_summary(prefix, excluded={"amdgpu-sim-rocm-pytorch-manifest.json"})
    manifest = {
        "schema": SCHEMA,
        "product_id": identifier,
        "prefix": str(prefix),
        "identity": document,
        "state_root": str(state),
        "sdk_root": str(sdk_root),
        "provider_install": provider_install,
        "system_runtime_install": system_runtime_install,
        "sdk_libraries": sdk_libraries,
        "installed_tree": installed,
        "runtime_probe": runtime_probe,
        "entry": {
            "python": str(python),
            "activate": str(prefix / "etc/conda/activate.d/amdgpu-sim-rocm-pytorch.sh"),
        },
    }
    manifest_path.write_bytes(canonical_json(manifest))
    verify(root, prefix, profile)
    publish_active(root, prefix, identifier, profile)
    return prefix


def verify(
    root: Path,
    prefix: Path | None = None,
    profile: str | None = None,
) -> Path:
    if prefix is None:
        active, _ = read_canonical(root / ACTIVE_RELATIVE)
        if active.get("schema") != ACTIVE_SCHEMA:
            raise ProductError("active ROCm PyTorch product schema is invalid")
        prefix = Path(active.get("prefix", ""))
    manifest, _ = read_canonical(prefix / "amdgpu-sim-rocm-pytorch-manifest.json")
    manifest_identity = manifest.get("identity")
    manifest_profile = (
        manifest_identity.get("profile") if isinstance(manifest_identity, dict) else None
    )
    if profile is None:
        profile = manifest_profile
    if not isinstance(profile, str) or profile != manifest_profile:
        raise ProductError("ROCm PyTorch product profile differs")
    current, native = identity(root, profile)
    identifier = product_id(current)
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("product_id") != identifier
        or manifest.get("identity") != current
        or manifest.get("prefix") != str(prefix)
        or prefix != product_prefix(root, identifier)
    ):
        raise ProductError("ROCm PyTorch product identity drifted")
    state = root / "env/conda-state/rocm-pytorch" / identifier
    if manifest.get("state_root") != str(state):
        raise ProductError("ROCm PyTorch state root differs")
    sdk_root = Path(manifest.get("sdk_root", "")).resolve(strict=True)
    try:
        sdk_root.relative_to(prefix)
    except ValueError as error:
        raise ProductError("ROCm SDK root escapes product") from error
    verify_provider_install(
        prefix,
        sdk_root,
        current["rocm_provider"],
        manifest.get("provider_install"),
    )
    system_runtime_root = verify_system_runtime_install(
        prefix,
        current["system_runtime"],
        manifest.get("system_runtime_install"),
    )
    observed = tree_summary(prefix, excluded={"amdgpu-sim-rocm-pytorch-manifest.json"})
    if observed != manifest.get("installed_tree"):
        raise ProductError("ROCm PyTorch product contents drifted")
    expected_versions = {package["name"]: package["version"] for package in current["wheel_set"]["packages"]}
    sdk_libraries = provider_library_records(
        prefix, sdk_root, current["rocm_provider"]
    )
    if sdk_libraries != manifest.get("sdk_libraries"):
        raise ProductError("ROCm SDK runtime libraries drifted")
    for path in (state / "home", state / "tmp", state / "xdg-cache", state / "triton-cache"):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    runtime_probe = probe_product(
        prefix,
        native,
        sdk_root,
        sdk_libraries,
        state,
        expected_versions,
        system_runtime_root,
    )
    if runtime_probe != manifest.get("runtime_probe"):
        raise ProductError("ROCm PyTorch runtime probe differs")
    return prefix


def publish_active(root: Path, prefix: Path, identifier: str, profile: str) -> None:
    manifest = file_record(prefix / "amdgpu-sim-rocm-pytorch-manifest.json")
    document = {
        "schema": ACTIVE_SCHEMA,
        "product_id": identifier,
        "profile": profile,
        "prefix": str(prefix),
        "manifest_sha256": manifest["sha256"],
    }
    path = root / ACTIVE_RELATIVE
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(canonical_json(document))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--print-prefix", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--profile", choices=sorted(PRODUCT_PROFILES))
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve(strict=True)
    try:
        if arguments.build:
            result = build(root, arguments.profile or DEFAULT_PROFILE)
        elif arguments.verify:
            result = verify(root, profile=arguments.profile)
        else:
            active, _ = read_canonical(root / ACTIVE_RELATIVE)
            if active.get("schema") != ACTIVE_SCHEMA:
                raise ProductError("active ROCm PyTorch product schema is invalid")
            result = Path(active["prefix"])
    except (ProductError, OSError, subprocess.CalledProcessError, KeyError, ValueError) as error:
        print(f"ROCm PyTorch product error: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
