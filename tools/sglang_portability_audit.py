#!/usr/bin/env python3
"""Audit an unchanged upstream SGLang tree for GemSim portability hooks.

This module deliberately does not import SGLang or patch its modules.  It only
records source identity and the lowest-level extension points available to an
out-of-tree adapter.  A missing SRT hook is a reported extension gap, not a
reason to copy or edit a model implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SCHEMA = "amdgpu-sim.sglang-portability-audit.v1"
MANIFEST_SCHEMA = "amdgpu-sim.sglang-source.v1"
_EXCLUDED_PARTS = {".git", "__pycache__"}


class AuditError(RuntimeError):
    """Raised when a source snapshot cannot be audited safely."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _regular_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise AuditError(f"SGLang source root is not a directory: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in _EXCLUDED_PARTS for part in path.parts):
            continue
        if path.suffix == ".pyc":
            continue
        files.append(path)
    return sorted(files)


def source_identity(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for path in _regular_files(root):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        records.append(
            {
                "path": relative,
                "bytes": len(data),
                "sha256": _sha256(data),
                "mode": path.stat().st_mode & 0o777,
            }
        )
    core = {"schema": MANIFEST_SCHEMA, "files": records}
    return {
        "schema": MANIFEST_SCHEMA,
        "root": str(root),
        "file_count": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "source_set_sha256": _sha256(_canonical(core)),
        "files": records,
    }


def _read(root: Path, relative: str) -> str:
    path = root / relative
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError) as error:
        raise AuditError(f"required upstream file is unavailable: {relative}") from error


def _first_existing(root: Path, relatives: tuple[str, ...]) -> tuple[str, str]:
    """Read the first metadata file present in a source or wheel snapshot."""
    for relative in relatives:
        path = root / relative
        if path.is_file():
            try:
                return relative, path.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                raise AuditError(f"metadata file is not UTF-8: {relative}") from error
    raise AuditError(
        "required SGLang metadata is unavailable: " + ", ".join(relatives)
    )


def _distribution_metadata(root: Path) -> tuple[str, str]:
    candidates = ["PKG-INFO"]
    candidates.extend(
        path.relative_to(root).as_posix()
        for path in sorted(root.glob("*.dist-info/METADATA"))
    )
    return _first_existing(root, tuple(candidates))


def _entry_points_metadata(root: Path) -> tuple[str | None, str]:
    paths = sorted(root.glob("*.dist-info/entry_points.txt"))
    if not paths:
        return None, ""
    path = paths[0]
    return path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")


def _version(root: Path) -> str:
    _, metadata = _distribution_metadata(root)
    match = re.search(r"^Version:\s*([^\n]+)", metadata, re.MULTILINE)
    if match:
        return match.group(1).strip()
    version = _read(root, "sglang/_version.py")
    match = re.search(r"__version__\s*=\s*version\s*=\s*['\"]([^'\"]+)", version)
    if not match:
        raise AuditError("cannot determine upstream SGLang version")
    return match.group(1)


def _contains(root: Path, relative: str, pattern: str) -> bool:
    return re.search(pattern, _read(root, relative), re.MULTILINE) is not None


def audit(root: Path, *, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    root = root.resolve(strict=True)
    identity = source_identity(root)
    metadata_relative, metadata = _distribution_metadata(root)
    entrypoint_relative, entrypoints = _entry_points_metadata(root)
    key_files: dict[str, str] = {}
    for relative in (
        metadata_relative,
        "sglang/srt/distributed/parallel_state.py",
        "sglang/srt/layers/attention/attention_registry.py",
        "sglang/srt/server_args.py",
    ):
        key_files[relative] = _sha256((root / relative).read_bytes())
    parallel = _read(root, "sglang/srt/distributed/parallel_state.py")
    pyproject_path = root / "pyproject.toml"
    pyproject = pyproject_path.read_text(encoding="utf-8") if pyproject_path.is_file() else ""
    attention = _read(root, "sglang/srt/layers/attention/attention_registry.py")
    srt_platform_dir = (root / "sglang/srt/platforms").is_dir()
    platform_discovery = ""
    for relative in (
        "sglang/srt/platforms/__init__.py",
        "sglang/srt/plugins/__init__.py",
    ):
        path = root / relative
        if path.is_file():
            platform_discovery += path.read_text(encoding="utf-8") + "\n"
            key_files[relative] = _sha256(path.read_bytes())
    rocm_relative = "sglang/srt/platforms/rocm.py"
    rocm_path = root / rocm_relative
    rocm_platform = ""
    if rocm_path.is_file():
        rocm_platform = rocm_path.read_text(encoding="utf-8")
        key_files[rocm_relative] = _sha256(rocm_path.read_bytes())
    has_platform_discovery = bool(
        re.search(r"PLATFORM_PLUGINS_GROUP\s*=\s*['\"]sglang\.srt\.platforms['\"]", platform_discovery)
        and "entry_points" in platform_discovery
        and "_load_platform_class" in platform_discovery
    )
    has_entrypoint = bool(
        re.search(
            r"\[sglang\.srt\.platforms\]",
            entrypoints,
            re.I | re.S,
        )
        or re.search(
            r"(?:platform_plugins|entry[_-]?points).*?(?:sglang|srt)|"
            r"(?:sglang|srt).*?(?:platform_plugins|entry[_-]?points)",
            pyproject,
            re.I | re.S,
        )
    )
    has_upstream_rocm_auto_detection = bool(
        rocm_platform
        and "class RocmSRTPlatform" in rocm_platform
        and "class RocmDeviceMixin" in rocm_platform
        and "torch.version.hip is not None" in platform_discovery
        and "RocmSRTPlatform()" in platform_discovery
    )
    has_backend_registration = "Backend.register_backend" in parallel or "Backend.register_backend" in pyproject
    # Internal communicator implementations do not constitute an OOT hook.
    # Require an explicit constructor/factory injection surface instead of a
    # module import such as ``device_communicators.pynccl``.
    has_group_device_hook = bool(
        re.search(
            r"(?:device_communicator_cls|communicator_cls|"
            r"get_device_communicator|register_device_communicator|"
            r"device_communicator_factory)",
            parallel,
            re.I,
        )
    )
    official_oot_hook = (srt_platform_dir and has_platform_discovery) or has_group_device_hook
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "upstream": {
            "version": _version(root),
            "source": identity,
            "key_file_sha256": key_files,
            "metadata_path": metadata_relative,
            "entry_points_path": entrypoint_relative,
        },
        "attention": {
            "aiter_registered": bool(re.search(r"['\"]aiter['\"]", attention)),
            "triton_registered": bool(re.search(r"['\"]triton['\"]", attention)),
            "flashinfer_present": "flashinfer" in attention.lower() or "flashinfer" in pyproject.lower(),
            "policy": "aiter_or_triton_only",
        },
        "srt_extension_points": {
            "platform_directory": srt_platform_dir,
            "platform_discovery_contract": has_platform_discovery,
            "pyproject_entry_point_or_platform_plugin": has_entrypoint,
            "entry_point_group": "sglang.srt.platforms" if has_platform_discovery else None,
            "parallel_state_device_communicator_hook": has_group_device_hook,
            "pytorch_backend_registration_in_upstream": has_backend_registration,
            "official_oot_hook": official_oot_hook,
            "upstream_rocm_auto_detection": has_upstream_rocm_auto_detection,
            "formal_model_platform": (
                "in_tree_RocmSRTPlatform"
                if has_upstream_rocm_auto_detection
                else None
            ),
            "oot_platform_role": "diagnostic_only",
        },
        "parallel_call_chain": {
            "group_coordinator": "sglang.srt.distributed.parallel_state.GroupCoordinator",
            "all_reduce": "GroupCoordinator.all_reduce -> torch.distributed.all_reduce or sgl_kernel.shm_allreduce",
            "group_constructor": "torch.distributed.new_group(device backend) + gloo cpu group",
        },
        "extension_gap": {
            "code": "SRT_NO_PUBLIC_OOT_DEVICE_COMMUNICATOR" if not official_oot_hook else None,
            "severity": "blocking_for_transparent_srt_tp" if not official_oot_hook else "none",
            "resolution": "provide the standard HIP-backed PyTorch device and RCCL/NCCL contracts so unchanged SGLang auto-selects its in-tree RocmSRTPlatform; retain the official OOT platform and generic c10d ProcessGroup only as diagnostics/fallback; do not modify SGLang or runtime-gem5 bridge",
        },
    }
    if expected is not None:
        expected_identity = (
            expected.get("upstream", {}).get("source", {}).get("source_set_sha256")
            or expected.get("upstream", {}).get("source_set_sha256")
            or expected.get("package", {}).get("source_set_sha256")
        )
        if expected_identity and expected_identity != identity["source_set_sha256"]:
            raise AuditError(
                "SGLang source drift: expected "
                f"{expected_identity}, got {identity['source_set_sha256']}"
            )
        expected_version = (
            expected.get("upstream", {}).get("version")
            or expected.get("package", {}).get("version")
        )
        if expected_version and expected_version != result["upstream"]["version"]:
            raise AuditError(f"SGLang version drift: expected {expected_version}, got {result['upstream']['version']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected = json.loads(args.expected.read_text()) if args.expected else None
    result = audit(args.source_root, expected=expected)
    encoded = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="ascii")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
