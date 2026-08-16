#!/usr/bin/env python3
"""Probe the standard private ROCr/HIP/RCCL ABI surface.

This tool is deliberately a capability probe, not a device emulator.  It only
loads libraries from the selected immutable product prefix and never falls
back to /opt/rocm, the system loader path, CPU tensors, or PrivateUse1.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


SCHEMA = "amdgpu-sim.hip-facade-probe.v1"
PRODUCT_SCHEMAS = {"amdgpu-sim.product-prefix.v1", "amdgpu-sim.conda-product.v1"}

LIBRARY_CONTRACT: dict[str, dict[str, Any]] = {
    "rocr": {
        "names": ("libhsa-runtime64.so.1", "libhsa-runtime64.so"),
        "symbols": (
            "hsa_init",
            "hsa_shut_down",
            "hsa_iterate_agents",
            "hsa_queue_create",
            "hsa_signal_create",
        ),
    },
    "hip": {
        "names": ("libamdhip64.so.7", "libamdhip64.so"),
        "symbols": (
            "hipGetDeviceCount",
            "hipGetDeviceProperties",
            "hipMalloc",
            "hipFree",
            "hipMemcpy",
            "hipStreamCreate",
            "hipStreamSynchronize",
            "hipModuleLoadData",
            "hipModuleLaunchKernel",
            "hipGetErrorString",
        ),
    },
    "comgr": {
        "names": ("libamd_comgr.so.3", "libamd_comgr.so"),
        "symbols": (
            "amd_comgr_create_data",
            "amd_comgr_do_action",
            "amd_comgr_release_data",
        ),
    },
    "rccl": {
        "names": ("librccl.so", "librccl.so.1", "libnccl.so.2"),
        "symbols": (
            "ncclGetUniqueId",
            "ncclCommInitRank",
            "ncclAllReduce",
            "ncclCommDestroy",
        ),
    },
}


class FacadeProbeError(RuntimeError):
    """Raised for malformed product metadata or unsafe paths."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_regular(path: Path, prefix: Path) -> tuple[bool, str | None]:
    """Check that a candidate is a product-owned regular file in-prefix."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False, "missing"
    if not stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
        return False, "not_regular"
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False, "broken_symlink"
    try:
        resolved.relative_to(prefix)
    except ValueError:
        return False, "symlink_escapes_product"
    resolved_meta = resolved.stat()
    if not stat.S_ISREG(resolved_meta.st_mode):
        return False, "resolved_not_regular"
    if resolved_meta.st_uid != os.getuid():
        return False, "not_owned"
    return True, None


def _load_manifest(prefix: Path) -> tuple[dict[str, Any] | None, str | None]:
    for name in ("manifest.json", "amdgpu-sim-manifest.json"):
        path = prefix / name
        if not path.is_file() or path.is_symlink():
            continue
        try:
            payload = path.read_bytes()
            document = json.loads(payload.decode("ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FacadeProbeError(f"invalid product manifest: {path}") from error
        if not isinstance(document, dict) or document.get("schema") not in PRODUCT_SCHEMAS:
            raise FacadeProbeError(f"unsupported product manifest schema: {path}")
        return document, hashlib.sha256(payload).hexdigest()
    return None, None


def _candidate_paths(prefix: Path, names: tuple[str, ...]) -> list[Path]:
    directories = (prefix / "lib", prefix / "lib64", prefix / "lib/x86_64-linux-gnu")
    return [directory / name for directory in directories for name in names]


def _find_library(prefix: Path, names: tuple[str, ...]) -> tuple[Path | None, list[dict[str, str]]]:
    rejected: list[dict[str, str]] = []
    for candidate in _candidate_paths(prefix, names):
        ok, reason = _safe_regular(candidate, prefix)
        if ok:
            return candidate.resolve(), rejected
        if reason != "missing":
            rejected.append({"path": str(candidate), "reason": reason or "invalid"})
    return None, rejected


def _load_symbols(path: Path, symbols: tuple[str, ...]) -> tuple[bool, list[str], str | None]:
    try:
        library = ctypes.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    except OSError as error:
        return False, [], str(error)
    missing = [name for name in symbols if not hasattr(library, name)]
    return not missing, missing, None


def probe(prefix: Path, *, load: bool = False, require_manifest: bool = True) -> dict[str, Any]:
    prefix = prefix.expanduser().absolute()
    if not prefix.is_dir() or prefix.is_symlink():
        raise FacadeProbeError(f"product prefix is not a directory: {prefix}")
    manifest, manifest_sha = _load_manifest(prefix)
    errors: list[str] = []
    if require_manifest and manifest is None:
        errors.append("product manifest is missing")

    libraries: dict[str, Any] = {}
    for family, contract in LIBRARY_CONTRACT.items():
        path, rejected = _find_library(prefix, contract["names"])
        record: dict[str, Any] = {
            "names": list(contract["names"]),
            "path": str(path) if path else None,
            "sha256": sha256(path) if path else None,
            "symbols": list(contract["symbols"]),
            "missing_symbols": [],
            "load_error": None,
            "rejected_candidates": rejected,
            "status": "missing" if path is None else "present",
        }
        if path is None:
            errors.append(f"{family}: standard library is missing")
        elif load:
            symbols_ok, missing, load_error = _load_symbols(path, contract["symbols"])
            record["missing_symbols"] = missing
            record["load_error"] = load_error
            if not symbols_ok:
                record["status"] = "invalid"
                if load_error:
                    errors.append(f"{family}: library load failed")
                else:
                    errors.append(f"{family}: required symbol is missing")
            else:
                record["status"] = "loaded"
        libraries[family] = record

    # The product may publish a stronger artifact binding.  Recheck it when
    # available, but keep the probe usable for pre-publication build products.
    binding = manifest.get("hip_facade") if manifest else None
    if binding is not None and not isinstance(binding, dict):
        errors.append("product hip_facade binding is not an object")
    if isinstance(binding, dict):
        for family, expected in binding.get("libraries", {}).items():
            observed = libraries.get(family)
            if not isinstance(expected, dict) or not isinstance(observed, dict):
                errors.append(f"hip_facade binding has unknown family: {family}")
                continue
            if expected.get("path") != observed.get("path") or expected.get("sha256") != observed.get("sha256"):
                errors.append(f"hip_facade binding drifted: {family}")

    return {
        "schema": SCHEMA,
        "prefix": str(prefix),
        "product_id": manifest.get("product_id") if manifest else None,
        "manifest_sha256": manifest_sha,
        "manifest_present": manifest is not None,
        "load_requested": load,
        "libraries": libraries,
        "policy": {
            "standard_abi_only": True,
            "system_loader_fallback": False,
            "cpu_fallback": False,
            "privateuse1_fallback": False,
            "kfd_or_drm_probe": False,
        },
        "errors": errors,
        "correct": not errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--allow-unmanaged", action="store_true")
    args = parser.parse_args(argv)
    prefix = args.prefix
    if prefix is None:
        active = args.root / "env/rocm/active-product"
        try:
            active_document = json.loads(active.read_text(encoding="ascii"))
            prefix = Path(active_document["prefix"])
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
            print(f"facade probe error: active product is unavailable: {error}", file=os.sys.stderr)
            return 1
    try:
        result = probe(prefix, load=args.load, require_manifest=not args.allow_unmanaged)
    except (FacadeProbeError, OSError) as error:
        print(f"facade probe error: {error}", file=os.sys.stderr)
        return 1
    payload = canonical_json(result)
    if args.output:
        args.output.write_bytes(payload)
    else:
        print(payload.decode("ascii"), end="")
    return 0 if result["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
