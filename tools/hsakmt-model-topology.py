#!/usr/bin/env python3
"""Materialize and verify a deterministic HSA Model gfx950 topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any


SCHEMA = "self-amdgpu-runtime.hsakmt-model-topology.v1"
MANIFEST_NAME = "manifest.json"
GPU_COUNT_MINIMUM = 1
GPU_COUNT_MAXIMUM = 16
GPU_LOCAL_MEMORY_BYTES = 309_237_645_312


class TopologyError(RuntimeError):
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


def regular_file(path: Path) -> tuple[bytes, os.stat_result]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
        raise TopologyError(f"expected an owned regular file: {path}")
    payload = path.read_bytes()
    after = path.lstat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise TopologyError(f"file changed while reading: {path}")
    return payload, after


def properties(values: list[tuple[str, int]]) -> bytes:
    return "".join(f"{name} {value}\n" for name, value in values).encode("ascii")


def link_properties(node_from: int, node_to: int, link_type: int, weight: int) -> bytes:
    return properties(
        [
            ("type", link_type),
            ("version_major", 0),
            ("version_minor", 0),
            ("node_from", node_from),
            ("node_to", node_to),
            ("weight", weight),
            ("min_latency", 0),
            ("max_latency", 0),
            ("min_bandwidth", 0 if link_type == 2 else 50_000),
            ("max_bandwidth", 0 if link_type == 2 else 50_000),
            ("recommended_transfer_size", 0),
            ("num_hops", 1),
            ("flags", 1),
        ]
    )


def cpu_properties(gpu_count: int) -> bytes:
    return properties(
        [
            ("cpu_cores_count", 1),
            ("simd_count", 0),
            ("mem_banks_count", 1),
            ("caches_count", 0),
            ("io_links_count", gpu_count),
            ("cpu_core_id_base", 0),
            ("simd_id_base", 0),
            ("max_waves_per_simd", 0),
            ("lds_size_in_kb", 0),
            ("gds_size_in_kb", 0),
            ("num_gws", 0),
            ("wave_front_size", 0),
            ("array_count", 0),
            ("simd_arrays_per_engine", 0),
            ("cu_per_simd_array", 0),
            ("simd_per_cu", 0),
            ("max_slots_scratch_cu", 0),
            ("gfx_target_version", 0),
            ("vendor_id", 2),
            ("device_id", 0),
            ("location_id", 0),
            ("domain", 0),
            ("drm_render_minor", 0),
            ("hive_id", 0),
            ("num_sdma_engines", 0),
            ("num_sdma_xgmi_engines", 0),
            ("num_sdma_queues_per_engine", 0),
            ("num_cp_queues", 0),
            ("max_engine_clk_fcompute", 0),
            ("max_engine_clk_ccompute", 3000),
            ("local_mem_size", 0),
            ("fw_version", 0),
            ("capability", 0),
            ("sdma_fw_version", 0),
            ("vram_public", 0),
            ("vram_size", 0),
        ]
    )


def gpu_properties(index: int, gpu_count: int) -> bytes:
    return properties(
        [
            ("cpu_cores_count", 0),
            ("simd_count", 1024),
            ("mem_banks_count", 1),
            ("caches_count", 2),
            ("io_links_count", gpu_count),
            ("p2p_links_count", gpu_count - 1),
            ("cpu_core_id_base", 0),
            ("simd_id_base", 2_147_487_744 + index * 4096),
            ("max_waves_per_simd", 8),
            ("lds_size_in_kb", 160),
            ("gds_size_in_kb", 0),
            ("num_gws", 64),
            ("wave_front_size", 64),
            ("array_count", 64),
            ("simd_arrays_per_engine", 2),
            ("cu_per_simd_array", 4),
            ("simd_per_cu", 4),
            ("max_slots_scratch_cu", 32),
            ("gfx_target_version", 90_500),
            ("vendor_id", 4098),
            ("device_id", 30_112),
            ("location_id", (index + 1) << 8),
            ("domain", 0),
            ("drm_render_minor", 128 + index),
            ("hive_id", 0x5341475200000001),
            ("num_sdma_engines", 5),
            ("num_sdma_xgmi_engines", 12),
            ("num_sdma_queues_per_engine", 2),
            ("num_cp_queues", 128),
            ("max_engine_clk_fcompute", 2400),
            ("max_engine_clk_ccompute", 0),
            ("local_mem_size", GPU_LOCAL_MEMORY_BYTES),
            ("fw_version", 0),
            ("capability", 2_889_327_232),
            ("capability2", 1),
            ("debug_prop", 1494),
            ("sdma_fw_version", 0),
            ("unique_id", 0x52414E4B00000001 + index),
            ("num_xcc", 8),
            ("family_id", 160),
            ("vram_public", 1),
            ("vram_size", GPU_LOCAL_MEMORY_BYTES),
        ]
    )


def cache_properties(level: int, size_kib: int, line_size: int, association: int) -> bytes:
    return properties(
        [
            ("processor_id_low", 0),
            ("level", level),
            ("size", size_kib),
            ("cache_line_size", line_size),
            ("cache_lines_per_tag", 1),
            ("association", association),
            ("latency", 0),
            ("type", 9),
        ]
    ) + b"sibling_map " + b",".join([b"0"] * 32) + b"\n"


def topology_files(gpu_count: int) -> dict[str, bytes]:
    if not GPU_COUNT_MINIMUM <= gpu_count <= GPU_COUNT_MAXIMUM:
        raise TopologyError("gpu_count must be in [1, 16]")
    files: dict[str, bytes] = {
        "generation_id": b"1\n",
        "system_properties": properties(
            [
                ("platform_oem", 0),
                ("platform_id", 0),
                ("platform_rev", 0),
                ("num_devices", 1 + gpu_count),
            ]
        ),
        "nodes/0/gpu_id": b"0\n",
        "nodes/0/properties": cpu_properties(gpu_count),
        "nodes/0/mem_banks/0/properties": properties(
            [
                ("heap_type", 0),
                ("size_in_bytes", 68_719_476_736),
                ("flags", 0),
                ("width", 0),
                ("mem_clk_max", 0),
            ]
        ),
    }
    for index in range(gpu_count):
        node = index + 1
        prefix = f"nodes/{node}"
        files[f"nodes/0/io_links/{index}/properties"] = link_properties(0, node, 2, 20)
        files[f"{prefix}/gpu_id"] = f"{38_144 + index}\n".encode("ascii")
        files[f"{prefix}/name"] = b"AMD Instinct MI350X\n"
        files[f"{prefix}/properties"] = gpu_properties(index, gpu_count)
        files[f"{prefix}/mem_banks/0/properties"] = properties(
            [
                ("heap_type", 1),
                ("size_in_bytes", GPU_LOCAL_MEMORY_BYTES),
                ("flags", 0),
                ("width", 8192),
                ("mem_clk_max", 1600),
            ]
        )
        files[f"{prefix}/mem_banks/0/used_memory"] = b"0\n"
        files[f"{prefix}/caches/0/properties"] = cache_properties(1, 32, 128, 4)
        files[f"{prefix}/caches/1/properties"] = cache_properties(2, 4096, 128, 16)
        files[f"{prefix}/io_links/0/properties"] = link_properties(node, 0, 2, 20)
        peer_slot = 1
        for peer_index in range(gpu_count):
            if peer_index == index:
                continue
            peer_node = peer_index + 1
            files[f"{prefix}/io_links/{peer_slot}/properties"] = link_properties(
                node, peer_node, 11, 15
            )
            files[f"{prefix}/p2p_links/{peer_slot - 1}/properties"] = link_properties(
                node, peer_node, 11, 15
            )
            peer_slot += 1
    return files


def expected_directories(files: dict[str, bytes]) -> list[str]:
    result: set[str] = set()
    for name in files:
        parent = Path(name).parent
        while parent != Path("."):
            result.add(parent.as_posix())
            parent = parent.parent
    return sorted(result)


def manifest_document(gpu_count: int, files: dict[str, bytes]) -> dict[str, Any]:
    inventory = [
        {"path": name, "bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(files.items())
    ]
    specification = {
        "architecture": "gfx950",
        "gpu_count": gpu_count,
        "gpu_id_base": 38_144,
        "render_minor_base": 128,
        "wavefront_size": 64,
        "local_memory_bytes_per_gpu": GPU_LOCAL_MEMORY_BYTES,
    }
    return {
        "schema": SCHEMA,
        "specification": specification,
        "specification_sha256": sha256_bytes(canonical_json(specification)),
        "directories": expected_directories(files),
        "files": inventory,
    }


def validate_root(path: Path) -> None:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise TopologyError(f"expected an owned real directory: {path}")


def verify(output: Path, expected_gpu_count: int | None = None) -> dict[str, Any]:
    validate_root(output)
    payload, _ = regular_file(output / MANIFEST_NAME)
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TopologyError("invalid topology manifest") from error
    if not isinstance(document, dict) or canonical_json(document) != payload:
        raise TopologyError("topology manifest is not canonical")
    specification = document.get("specification")
    if (
        document.get("schema") != SCHEMA
        or not isinstance(specification, dict)
        or document.get("specification_sha256") != sha256_bytes(canonical_json(specification))
    ):
        raise TopologyError("topology manifest identity is invalid")
    gpu_count = specification.get("gpu_count")
    if type(gpu_count) is not int or not GPU_COUNT_MINIMUM <= gpu_count <= GPU_COUNT_MAXIMUM:
        raise TopologyError("topology gpu_count is invalid")
    if expected_gpu_count is not None and gpu_count != expected_gpu_count:
        raise TopologyError("existing topology gpu_count differs")
    expected = manifest_document(gpu_count, topology_files(gpu_count))
    if document != expected:
        raise TopologyError("topology manifest differs from the canonical specification")

    actual_directories: list[str] = []
    actual_files: list[str] = []
    for current, directories, names in os.walk(output, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(directories):
            path = current_path / name
            metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
                raise TopologyError(f"non-directory entry in topology: {path}")
            actual_directories.append(path.relative_to(output).as_posix())
        for name in sorted(names):
            path = current_path / name
            regular_file(path)
            relative = path.relative_to(output).as_posix()
            if relative != MANIFEST_NAME:
                actual_files.append(relative)
    if sorted(actual_directories) != document["directories"]:
        raise TopologyError("topology directory set differs")
    expected_names = [record["path"] for record in document["files"]]
    if sorted(actual_files) != expected_names:
        raise TopologyError("topology file set differs")
    for record in document["files"]:
        data, metadata = regular_file(output / record["path"])
        if metadata.st_size != record["bytes"] or sha256_bytes(data) != record["sha256"]:
            raise TopologyError(f"topology file drifted: {record['path']}")
    return document


def materialize(output: Path, gpu_count: int) -> dict[str, Any]:
    if not output.is_absolute():
        raise TopologyError("output path must be absolute")
    parent = output.parent.resolve(strict=True)
    validate_root(parent)
    if output.exists() or output.is_symlink():
        return verify(output, gpu_count)
    files = topology_files(gpu_count)
    document = manifest_document(gpu_count, files)
    created = False
    try:
        output.mkdir(mode=0o700)
        created = True
        for directory in document["directories"]:
            (output / directory).mkdir(mode=0o700, parents=False, exist_ok=False)
        for name, payload in sorted(files.items()):
            path = output / name
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        manifest = canonical_json(document)
        descriptor = os.open(
            output / MANIFEST_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(manifest)
            stream.flush()
            os.fsync(stream.fileno())
        root_descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(root_descriptor)
        finally:
            os.close(root_descriptor)
    except Exception:
        if created:
            shutil.rmtree(output)
        raise
    return verify(output, gpu_count)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify == (arguments.gpu_count is not None):
        parser.error("choose exactly one of --verify or --gpu-count")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.verify:
            document = verify(arguments.output_dir)
        else:
            document = materialize(arguments.output_dir, arguments.gpu_count)
    except (OSError, TopologyError) as error:
        print(f"hsakmt-model-topology: {error}", file=sys.stderr)
        return 1
    print(canonical_json(document).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
