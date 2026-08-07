#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Download and verify the frozen Qwen model without placing weights in Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import struct
import sys
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_ENDPOINT = "https://huggingface.co"
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


class ModelError(RuntimeError):
    pass


def load_lock() -> dict[str, Any]:
    try:
        lock = json.loads((ROOT / "SOURCE_LOCK.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"cannot read SOURCE_LOCK.json: {exc}") from exc
    if lock.get("status") != "frozen":
        raise ModelError("SOURCE_LOCK.json must be frozen before model download")
    for source in lock.get("sources", []):
        if source.get("id") == "qwen3.5-0.8b":
            return source
    raise ModelError("Qwen model is absent from SOURCE_LOCK.json")


def ignored_storage_root(name: str) -> Path:
    if name not in {"models", "cache"}:
        raise ModelError(f"unsupported ignored storage root: {name!r}")
    lexical = ROOT / name
    if lexical.is_symlink():
        raise ModelError(f"ignored storage root must not be a symlink: {name}/")
    resolved = lexical.resolve()
    if ROOT != resolved and ROOT not in resolved.parents:
        raise ModelError(f"ignored storage root escapes workspace: {name}/")
    return resolved


def model_output_path(value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("models",):
        raise ModelError("model output must be a relative path below models/")
    final = (ROOT / relative).resolve()
    models_root = ignored_storage_root("models")
    if models_root not in final.parents:
        raise ModelError("model output must remain below the ignored models/ directory")
    return final


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_safetensors_header(path: Path) -> int:
    size = path.stat().st_size
    if size < 10:
        raise ModelError(f"safetensors file is too small: {path.name}")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ModelError(f"cannot read safetensors header length: {path.name}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > min(size - 8, 100 * 1024 * 1024):
            raise ModelError(
                f"invalid safetensors header length for {path.name}: {header_size}"
            )
        try:
            header = json.loads(stream.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelError(f"invalid safetensors JSON header: {path.name}: {exc}") from exc
    if not isinstance(header, dict):
        raise ModelError(f"safetensors header is not an object: {path.name}")
    data_size = size - 8 - header_size
    ranges = []
    tensor_count = 0
    for name, descriptor in header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict):
                raise ModelError(f"invalid safetensors metadata: {path.name}")
            continue
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise ModelError(f"invalid safetensors tensor descriptor: {path.name}")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if not isinstance(dtype, str) or not dtype:
            raise ModelError(f"tensor {name!r} has no dtype in {path.name}")
        if dtype not in DTYPE_BYTES:
            raise ModelError(f"tensor {name!r} has unsupported dtype {dtype!r} in {path.name}")
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in shape
        ):
            raise ModelError(f"tensor {name!r} has an invalid shape in {path.name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            raise ModelError(f"tensor {name!r} has invalid data offsets in {path.name}")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise ModelError(f"tensor {name!r} data range escapes {path.name}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        expected_bytes = elements * DTYPE_BYTES[dtype]
        if end - start != expected_bytes:
            raise ModelError(
                f"tensor {name!r} byte length does not match dtype/shape in {path.name}"
            )
        ranges.append((start, end, name))
        tensor_count += 1
    for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
        if current[0] < previous[1]:
            raise ModelError(
                f"overlapping tensors {previous[2]!r} and {current[2]!r} in {path.name}"
            )
    if tensor_count == 0:
        raise ModelError(f"safetensors file has no tensors: {path.name}")
    return tensor_count


def expected_inventory(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inventory = source.get("files")
    if not isinstance(inventory, list) or not inventory:
        raise ModelError("model file inventory is not frozen in SOURCE_LOCK.json")
    result: dict[str, dict[str, Any]] = {}
    for item in inventory:
        relative = item.get("path")
        if not isinstance(relative, str):
            raise ModelError("model inventory entry has no path")
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts or relative in result:
            raise ModelError(f"unsafe or duplicate model path: {relative!r}")
        result[relative] = item
    return result


def reject_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ModelError(f"model snapshot contains a symlink: {path.relative_to(root)}")


def actual_files(root: Path) -> set[str]:
    reject_symlinks(root)
    files = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative == ".cache/huggingface" or relative.startswith(".cache/huggingface/"):
            continue
        if not path.is_file():
            continue
        if relative == ".amdgpu-sim-download.json":
            continue
        files.add(relative)
    return files


def verify_snapshot(
    directory: Path, source: dict[str, Any], *, endpoint: str | None = None
) -> dict[str, Any]:
    marker = directory / ".amdgpu-sim-download.json"
    marker_value: dict[str, Any] | None = None
    if marker.is_symlink():
        raise ModelError("snapshot provenance marker must not be a symlink")
    if marker.is_file():
        try:
            marker_value = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelError(f"invalid download marker: {exc}") from exc
        marker_endpoint = marker_value.get("endpoint")
        if endpoint is not None and endpoint != marker_endpoint:
            raise ModelError(
                f"download endpoint does not match snapshot marker: {endpoint} != {marker_endpoint}"
            )
        endpoint = marker_endpoint
        if marker_value.get("repo_id") != source.get("repo_id"):
            raise ModelError("snapshot marker repository does not match SOURCE_LOCK")
        if marker_value.get("revision") != source.get("official_revision"):
            raise ModelError("snapshot marker revision does not match SOURCE_LOCK")
    elif endpoint is None:
        raise ModelError("snapshot has no provenance marker; provide an explicit endpoint")
    endpoint = endpoint or OFFICIAL_ENDPOINT
    inventory = expected_inventory(source)
    actual = actual_files(directory)
    expected = set(inventory)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ModelError(f"model file set mismatch; missing={missing}, extra={extra}")
    verified = []
    for relative in sorted(expected):
        item = inventory[relative]
        path = directory / relative
        size = path.stat().st_size
        if item.get("size") is not None and size != item["size"]:
            raise ModelError(f"size mismatch for {relative}: {size} != {item['size']}")
        sha256 = file_sha256(path)
        lfs = item.get("lfs")
        if isinstance(lfs, dict) and lfs.get("sha256"):
            if sha256 != lfs["sha256"]:
                raise ModelError(f"LFS SHA-256 mismatch for {relative}")
        elif item.get("blob_id"):
            if git_blob_sha1(path) != item["blob_id"]:
                raise ModelError(f"Git blob SHA-1 mismatch for {relative}")
        else:
            raise ModelError(f"no verifiable content identity for {relative}")
        verified_entry = {"path": relative, "size": size, "sha256": sha256}
        if relative.endswith(".safetensors"):
            verified_entry["safetensors_tensor_count"] = verify_safetensors_header(path)
        verified.append(verified_entry)

    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelError(f"invalid safetensors index: {exc}") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not all(
            isinstance(name, str) and isinstance(shard, str)
            for name, shard in weight_map.items()
        ):
            raise ModelError("safetensors index has an invalid weight_map")
        referenced = set(weight_map.values())
        missing_shards = sorted(name for name in referenced if name not in expected)
        if missing_shards:
            raise ModelError(f"safetensors index references missing shards: {missing_shards}")
    return {
        "schema": "amdgpu-sim.model-materialization.v1",
        "repo_id": source["repo_id"],
        "revision": source["official_revision"],
        "endpoint": endpoint,
        "provenance": "official" if endpoint == OFFICIAL_ENDPOINT else "explicit-mirror",
        "files": verified,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(directory: Path, paths: Iterable[str]) -> None:
    for relative in paths:
        with (directory / relative).open("rb") as stream:
            os.fsync(stream.fileno())
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def download(args: argparse.Namespace) -> dict[str, Any]:
    source = load_lock()
    revision = source.get("official_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ModelError("official model revision is not frozen")
    endpoint = args.endpoint.rstrip("/")
    if endpoint != OFFICIAL_ENDPOINT and not args.allow_mirror:
        raise ModelError("non-official endpoint requires explicit --allow-mirror")
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelError("huggingface_hub is required in the project environment") from exc

    final = model_output_path(args.output)
    inventory = expected_inventory(source)
    if final.exists():
        return verify_snapshot(final, source, endpoint=endpoint)
    stage = final.with_name(f".{final.name}.partial-{revision}")
    if stage.is_symlink():
        raise ModelError(f"model staging path must not be a symlink: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    marker = stage / ".amdgpu-sim-download.json"
    marker_value = {"repo_id": source["repo_id"], "revision": revision, "endpoint": endpoint}
    if marker.is_symlink():
        raise ModelError("model staging marker must not be a symlink")
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != marker_value:
        raise ModelError(f"staging marker mismatch: {stage}")
    atomic_json(marker, marker_value)
    reject_symlinks(stage)
    cache_root = ignored_storage_root("cache")
    huggingface_cache = cache_root / "huggingface"
    if huggingface_cache.is_symlink():
        raise ModelError("Hugging Face cache directory must not be a symlink")
    if cache_root not in huggingface_cache.resolve().parents:
        raise ModelError("Hugging Face cache directory escapes cache/")
    snapshot_download(
        repo_id=source["repo_id"],
        revision=revision,
        allow_patterns=sorted(inventory),
        local_dir=str(stage),
        cache_dir=str(huggingface_cache),
    )
    manifest = verify_snapshot(stage, source, endpoint=endpoint)
    fsync_tree(stage, inventory)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, final)
    parent_fd = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/Qwen3.5-0.8B")
    parser.add_argument("--endpoint", default=OFFICIAL_ENDPOINT)
    parser.add_argument("--allow-mirror", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        source = load_lock()
        if args.verify_only:
            manifest = verify_snapshot(model_output_path(args.output), source)
        else:
            manifest = download(args)
        revision = manifest["revision"]
        atomic_json(ROOT / "artifacts" / "model-manifests" / f"{revision}.json", manifest)
        print(f"verified {manifest['repo_id']} at {revision}")
        return 0
    except (ModelError, OSError, json.JSONDecodeError) as exc:
        print(f"model download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
