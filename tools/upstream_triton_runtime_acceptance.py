#!/usr/bin/env python3
"""Verify and atomically publish the unchanged upstream Triton AMD gate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Callable, Mapping

import sys


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
SCRIPTS = ROOT / "scripts"
for directory in (TOOLS, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import gemsim_single_rank_lifecycle as lifecycle  # noqa: E402
import rocm_pytorch_runtime_acceptance as product_contract  # noqa: E402


base = product_contract.base
THIS_FILE = Path(__file__).resolve()
RUNNER = ROOT / "scripts/run_upstream_triton_runtime.py"
LIFECYCLE = ROOT / "scripts/gemsim_single_rank_lifecycle.py"
QUICKSTART = ROOT / "examples/quickstart/triton_rocm.py"
PRODUCT_TOOL = ROOT / "tools/rocm_pytorch_product_environment.py"
ACTIVE_PRODUCT = ROOT / "env/conda/active-rocm-pytorch"
GEM5_REPOSITORY = ROOT / "projects/gem5"
RUNTIME_REPOSITORY = ROOT / "projects/self-amdgpu-runtime"
GEM5_BINARY = GEM5_REPOSITORY / "build/VEGA_X86/gem5.opt"
GEM5_CONFIG = GEM5_REPOSITORY / "configs/example/gemsim/host_dispatch.py"

RUN_SCHEMA = "amdgpu-sim.upstream-triton-amd-runtime-run.v1"
RESULT_SCHEMA = "amdgpu-sim.upstream-triton-amd-runtime-acceptance.v1"
MANIFEST_SCHEMA = "amdgpu-sim.upstream-triton-amd-evidence-manifest.v1"
QUICKSTART_SCHEMA = "amdgpu-sim.upstream-triton-amd-quickstart.v1"
TRACE_SCHEMA = "amdgpu-sim.native-kernel-execution-trace.v1"
CLAIM_SCOPE = "unchanged_upstream_triton_amd_multiop_on_runtime_gem5"
CORE_ARTIFACTS = (
    "worker.log",
    "gem5.log",
    "dispatch-trace.jsonl",
    "m5out/stats.txt",
    "m5out/config.ini",
    "m5out/config.json",
)
KERNEL_NAMES = ("add_kernel", "transform_kernel", "reduce_kernel")
KERNEL_CACHE_SUFFIXES = (
    ".source",
    ".ttir",
    ".ttgir",
    ".llir",
    ".amdgcn",
    ".hsaco",
    ".json",
)
OFFICIAL_SDK_PROVIDERS = frozenset(
    {
        "official_rocm_sdk",
        "official_rocm_apt_sysroot",
    }
)
TRITON_HELPER_RE = re.compile(
    r"^(?:__triton_launcher[^/]*\.so|hip_utils[^/]*\.so)$"
)
UUID_RE = re.compile(r"^[0-9a-f]{32}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CACHE_KEY_RE = re.compile(r"^[A-Z2-7]{52}$")
EXPECTED_INPUT_SHA256 = {
    "left": "d47525c00f9a542f1b5c1ec5f1b2e254fba4260c29212541c046cef75c776e39",
    "right": "eccc2172b04d2e42ced45fc64edf0c8ba14b3a1562fbefbbd86c3d40ae8d3367",
    "reduction_input": "75a2180e30970bf5a39c94c6cbda34a4dd3647a580828dcda881ef0d2ed3c437",
}
EXPECTED_OUTPUT_SHA256 = {
    "add": "ec0aa52312acd7f2c591073e576d3a42892cefafd69a0c5fe9363548fcc2c70b",
    "transform": "a52c3344ab332287e38707349f447bc423cee05b34590ddc96c9e2da8f714810",
    "reduce": "053ac09e0109c6d84088a0106d301d00ebe129455cb7889c911000d6f462271f",
}
EXPECTED_QUEUE_INDEX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
EXPECTED_KERNARG_SIZE = [48, 48, 48, 40, 32, 32, 48, 48, 48, 48, 48, 48]
EXPECTED_GRID = [
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [256, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
]
EXPECTED_WORKGROUP = [
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [256, 1, 1],
    [256, 1, 1],
    [256, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
    [512, 1, 1],
]
EXPECTED_WORKGROUP_COMPLETIONS = [1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1]


class AcceptanceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def canonical_json(value: object) -> bytes:
    return base.canonical_json(value)


def _bound_file(path: Path, *, executable: bool = False) -> dict[str, Any]:
    return product_contract._bound_file(path, executable=executable)


def identity_snapshot() -> dict[str, Any]:
    product = product_contract.active_product()
    manifest = product["manifest"]
    native = product["native_manifest"]
    prefix = product["prefix"]
    entry = manifest["entry"]
    runtime_probe = manifest["runtime_probe"]
    torch_root = Path(runtime_probe["torch_file"]).parent
    torch_lib = torch_root / "lib"
    triton_root = Path(runtime_probe["triton_file"]).parent
    files = {
        "verifier": _bound_file(THIS_FILE),
        "runner": _bound_file(RUNNER),
        "lifecycle": _bound_file(LIFECYCLE),
        "quickstart": _bound_file(QUICKSTART),
        "product_tool": _bound_file(PRODUCT_TOOL),
        "active_product": _bound_file(ACTIVE_PRODUCT),
        "product_manifest": _bound_file(product["manifest_path"]),
        "native_manifest": _bound_file(product["native_manifest_path"]),
        "activation": _bound_file(Path(entry["activate"])),
        "python": _bound_file(Path(entry["python"]), executable=True),
        "torch_init": _bound_file(Path(runtime_probe["torch_file"])),
        "triton_init": _bound_file(Path(runtime_probe["triton_file"])),
        "triton_amd_driver": _bound_file(triton_root / "backends/amd/driver.py"),
        "triton_compiler": _bound_file(triton_root / "compiler/compiler.py"),
        "triton_jit": _bound_file(triton_root / "runtime/jit.py"),
        "libtorch_hip": _bound_file(torch_lib / "libtorch_hip.so"),
        "libc10_hip": _bound_file(torch_lib / "libc10_hip.so"),
        "libtorch_python": _bound_file(torch_lib / "libtorch_python.so"),
        "gem5_binary": _bound_file(GEM5_BINARY, executable=True),
        "gem5_config": _bound_file(GEM5_CONFIG),
        "wavefront_header": _bound_file(
            GEM5_REPOSITORY / "src/gpu-compute/wavefront.hh"
        ),
        "wavefront_source": _bound_file(
            GEM5_REPOSITORY / "src/gpu-compute/wavefront.cc"
        ),
    }
    for role, descriptor in product["sdk_libraries"].items():
        files[role] = _bound_file(Path(descriptor["path"]))
    for role in (
        "runtime_library",
        "rocr_library",
        "hsakmt_model_library",
        "topology_manifest",
    ):
        files[role] = _bound_file(Path(native["artifacts"][role]["path"]))
    topology_root = Path(native["artifacts"]["topology_manifest"]["path"]).parent
    return {
        "product": {
            "schema": manifest["schema"],
            "product_id": manifest["product_id"],
            "prefix": str(prefix),
            "manifest_sha256": hashlib.sha256(
                product["manifest_payload"]
            ).hexdigest(),
            "installed_tree": manifest["installed_tree"],
            "runtime_probe": runtime_probe,
            "sdk_libraries": product["sdk_libraries"],
            "native_product_id": native["product_id"],
            "native_prefix": native["prefix"],
            "native_role": manifest["identity"]["native_product"]["role"],
        },
        "files": files,
        "topology": base.directory_record(topology_root),
        "repositories": {
            "gem5": base.source_set_summary(GEM5_REPOSITORY),
            "self_runtime": base.source_set_summary(RUNTIME_REPOSITORY),
            "upstream": manifest["identity"]["upstream"],
        },
    }


def read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        return base.read_json(path, label)
    except base.AcceptanceError as error:
        raise AcceptanceError(str(error)) from error


def read_terminal_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    """Read a canonical JSON result from the final line of a merged worker log."""
    base.regular_file(path)
    raw = path.read_bytes()
    require(0 < len(raw) <= base.MAX_TEXT_BYTES, f"{label} has invalid size")
    try:
        raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise AcceptanceError(f"{label} is not ASCII") from error
    require(raw.endswith(b"\n"), f"{label} does not end with a newline")
    lines = raw.splitlines(keepends=True)
    require(lines and lines[-1].strip(), f"{label} has no terminal result")
    terminal = lines[-1]
    try:
        value = json.loads(terminal.decode("ascii"))
    except json.JSONDecodeError as error:
        raise AcceptanceError(f"{label} terminal line is invalid JSON") from error
    require(isinstance(value, dict), f"{label} terminal result is not an object")
    require(terminal == canonical_json(value), f"{label} terminal result is not canonical JSON")
    return value, raw


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return None if prefix is None else f"{prefix}.{node.attr}"
    return None


def validate_quickstart_source(path: Path) -> dict[str, Any]:
    payload = path.read_text(encoding="ascii")
    tree = ast.parse(payload, filename=str(path))
    imported: set[str] = set()
    functions: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            imported.add(node.module or "")
        elif isinstance(node, ast.FunctionDef):
            functions[node.name] = node
    require(
        imported == {"hashlib", "json", "torch", "triton", "triton.language"},
        "Triton quickstart imports are not upstream-only",
    )
    require(
        set(KERNEL_NAMES).issubset(functions),
        "Triton quickstart kernel definitions are incomplete",
    )
    expected_calls = {
        "add_kernel": {"tl.program_id", "tl.arange", "tl.load", "tl.store"},
        "transform_kernel": {
            "tl.program_id",
            "tl.arange",
            "tl.load",
            "tl.where",
            "tl.store",
        },
        "reduce_kernel": {"tl.arange", "tl.load", "tl.sum", "tl.store"},
    }
    kernel_calls: dict[str, list[str]] = {}
    for name in KERNEL_NAMES:
        function = functions[name]
        decorators = {_dotted_name(value) for value in function.decorator_list}
        require(decorators == {"triton.jit"}, f"{name} is not one upstream Triton JIT kernel")
        calls = {
            value
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and (value := _dotted_name(node.func)) is not None
        }
        require(expected_calls[name].issubset(calls), f"{name} lacks required Triton operations")
        kernel_calls[name] = sorted(calls)
    main = functions.get("main")
    require(main is not None, "Triton quickstart main is missing")
    assigned_launches: dict[str, str] = {}
    for node in ast.walk(main):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.startswith("compiled_"):
            continue
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Subscript):
            launch = _dotted_name(node.value.func.value)
            if launch is not None:
                assigned_launches[target.id] = launch
    require(
        assigned_launches
        == {
            "compiled_add": "add_kernel",
            "compiled_transform": "transform_kernel",
            "compiled_reduce": "reduce_kernel",
        },
        "Triton compiled-kernel launch binding differs",
    )
    for forbidden in (
        "gemsim",
        "self_amdgpu",
        "torch.ops",
        "subprocess",
        "ctypes",
        "SAGR_",
        "ROCM_SIM_ROOT",
    ):
        require(forbidden not in payload, f"Triton quickstart contains a forbidden hook: {forbidden}")
    return {
        "upstream_only_imports": sorted(imported),
        "kernel_calls": kernel_calls,
        "compiled_launches": assigned_launches,
    }


def cache_artifact_paths(root: Path) -> tuple[str, ...]:
    cache = root / "triton-cache"
    require(cache.is_dir() and not cache.is_symlink(), "private Triton cache is missing")
    paths: list[str] = []
    for path in sorted(cache.rglob("*")):
        metadata = path.lstat()
        require(not path.is_symlink(), f"Triton cache path is a symlink: {path}")
        require(metadata.st_uid == os.getuid(), f"Triton cache path has the wrong owner: {path}")
        if path.is_dir():
            continue
        require(stat.S_ISREG(metadata.st_mode), f"Triton cache path is not regular: {path}")
        require(metadata.st_size > 0, f"Triton cache file is empty: {path}")
        paths.append(path.relative_to(root).as_posix())
    # Triton keeps the per-kernel stage set stable, but helper compilation is
    # version-dependent (3.6 emitted three launcher DSOs; 3.7 emits one
    # shared HIP utility).  Validate the semantic inventory below instead of
    # freezing an implementation-specific total file count here.
    minimum = len(KERNEL_NAMES) * (len(KERNEL_CACHE_SUFFIXES) + 1) + 1
    require(len(paths) >= minimum, "private Triton cache is incomplete")
    return tuple(paths)


def validate_artifacts(
    source: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    cache_paths = cache_artifact_paths(source)
    expected_paths = set(CORE_ARTIFACTS) | set(cache_paths)
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), "Triton run artifacts are missing")
    require(set(artifacts) == expected_paths, "Triton run artifact set differs")
    observed: dict[str, dict[str, Any]] = {}
    for relative in sorted(expected_paths):
        expected = artifacts.get(relative)
        require(isinstance(expected, dict), f"artifact record is invalid: {relative}")
        actual = base.artifact_record(source, relative)
        require(actual == expected, f"artifact content drifted: {relative}")
        observed[relative] = actual
    require(
        {entry.name for entry in source.iterdir()}
        == {
            "result-manifest.json",
            "worker.log",
            "gem5.log",
            "dispatch-trace.jsonl",
            "m5out",
            "triton-cache",
        },
        "Triton source root file set differs",
    )
    require(
        {entry.name for entry in (source / "m5out").iterdir()}
        == {"stats.txt", "config.ini", "config.json"},
        "Triton m5out file set differs",
    )
    return observed, cache_paths


def validate_worker(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    payload, _ = read_terminal_json(path, "Triton worker log")
    require(
        set(payload)
        == {
            "schema",
            "torch",
            "torch_hip",
            "triton",
            "device_count",
            "device_name",
            "capability",
            "driver",
            "kernels",
            "compiled_kernels",
            "tensor_contract",
            "checks",
            "target_feedback_from_oracle",
            "correct",
        },
        "Triton worker result keys differ",
    )
    runtime = identity["product"]["runtime_probe"]
    require(payload["schema"] == QUICKSTART_SCHEMA, "Triton quickstart schema differs")
    require(payload["torch"] == runtime["torch"], "Triton worker PyTorch version differs")
    require(payload["torch_hip"] == runtime["torch_hip"], "Triton worker HIP version differs")
    require(payload["triton"] == runtime["triton"], "Triton version differs")
    require(payload["device_count"] == 1, "Triton device count differs")
    require(payload["device_name"] == "AMD Instinct MI350X", "Triton device name differs")
    require(payload["capability"] == [9, 5], "Triton device capability differs")
    require(
        payload["driver"]
        == {
            "module": "triton.backends.amd.driver",
            "class": "HIPDriver",
            "backend": "hip",
            "arch": "gfx950",
            "warp_size": 64,
        },
        "Triton AMD driver identity differs",
    )
    require(payload["kernels"] == list(KERNEL_NAMES), "Triton kernel order differs")
    compiled = payload["compiled_kernels"]
    require(isinstance(compiled, list) and len(compiled) == 3, "compiled kernel records differ")
    shared = [0, 0, 16]
    for index, record in enumerate(compiled):
        require(
            isinstance(record, dict)
            and set(record)
            == {
                "name",
                "cache_hash",
                "binary_bytes",
                "binary_sha256",
                "target",
                "num_warps",
                "shared_memory_bytes",
            },
            "compiled kernel record keys differ",
        )
        require(record["name"] == KERNEL_NAMES[index], "compiled kernel name differs")
        require(
            isinstance(record["cache_hash"], str)
            and SHA256_RE.fullmatch(record["cache_hash"]),
            "compiled kernel cache hash is invalid",
        )
        require(
            isinstance(record["binary_sha256"], str)
            and SHA256_RE.fullmatch(record["binary_sha256"]),
            "compiled kernel binary hash is invalid",
        )
        require(
            isinstance(record["binary_bytes"], int) and record["binary_bytes"] > 0,
            "compiled kernel binary size is invalid",
        )
        require(
            record["target"] == {"backend": "hip", "arch": "gfx950", "warp_size": 64},
            "compiled kernel target differs",
        )
        require(record["num_warps"] == 4, "compiled kernel warp count differs")
        require(record["shared_memory_bytes"] == shared[index], "compiled kernel shared memory differs")
    tensor = payload["tensor_contract"]
    require(
        isinstance(tensor, dict)
        and set(tensor)
        == {
            "dtype",
            "element_count",
            "input_actual_sha256",
            "input_expected_sha256",
            "actual_sha256",
            "expected_sha256",
        },
        "Triton tensor contract differs",
    )
    require(tensor["dtype"] == "float32" and tensor["element_count"] == 256, "Triton tensor shape differs")
    require(tensor["input_expected_sha256"] == EXPECTED_INPUT_SHA256, "Triton input oracle differs")
    require(tensor["input_actual_sha256"] == EXPECTED_INPUT_SHA256, "Triton input mutation differs")
    require(tensor["expected_sha256"] == EXPECTED_OUTPUT_SHA256, "Triton output oracle differs")
    require(tensor["actual_sha256"] == EXPECTED_OUTPUT_SHA256, "Triton output differs")
    require(
        payload["checks"]
        == {
            "add_bitwise": True,
            "transform_bitwise": True,
            "reduce_bitwise": True,
            "inputs_unchanged": True,
            "outputs_are_cuda": True,
            "outputs_nonalias": True,
        },
        "Triton correctness checks differ",
    )
    require(payload["target_feedback_from_oracle"] is False, "Triton oracle fed the target")
    require(payload["correct"] is True, "Triton worker output is incorrect")
    return {
        "torch": payload["torch"],
        "torch_hip": payload["torch_hip"],
        "triton": payload["triton"],
        "driver": payload["driver"],
        "kernels": payload["kernels"],
        "compiled_kernels": compiled,
        "input_sha256": EXPECTED_INPUT_SHA256,
        "output_sha256": EXPECTED_OUTPUT_SHA256,
        "input_unchanged": True,
        "outputs_nonalias": True,
        "bitwise_correct": True,
    }


def validate_jit_cache(
    source: Path,
    cache_paths: tuple[str, ...],
    execution: Mapping[str, Any],
    worker: Mapping[str, Any],
) -> dict[str, Any]:
    cache = source / "triton-cache"
    original_cache = Path(execution["execution_root"]) / "triton-cache"
    compiled_by_name = {
        record["name"]: record for record in worker["compiled_kernels"]
    }
    kernel_records: dict[str, dict[str, Any]] = {}
    consumed: set[str] = set()
    for name in KERNEL_NAMES:
        matches: dict[str, str] = {}
        for suffix in KERNEL_CACHE_SUFFIXES:
            candidates = [
                relative
                for relative in cache_paths
                if Path(relative).name == f"{name}{suffix}"
            ]
            require(len(candidates) == 1, f"Triton cache {name}{suffix} count differs")
            matches[suffix] = candidates[0]
        group_candidates = [
            relative
            for relative in cache_paths
            if Path(relative).name == f"__grp__{name}.json"
        ]
        require(len(group_candidates) == 1, f"Triton cache group count differs for {name}")
        matches["group"] = group_candidates[0]
        parents = {Path(relative).parent for relative in matches.values()}
        require(len(parents) == 1, f"Triton cache files span directories for {name}")
        parent = next(iter(parents))
        require(parent.parent == Path("triton-cache"), f"Triton cache depth differs for {name}")
        require(CACHE_KEY_RE.fullmatch(parent.name) is not None, f"Triton cache key is invalid for {name}")
        consumed.update(matches.values())

        metadata, _ = read_json(source / matches[".json"], f"{name} metadata")
        compiled = compiled_by_name[name]
        require(metadata.get("name") == name, f"Triton metadata name differs for {name}")
        require(metadata.get("hash") == compiled["cache_hash"], f"Triton metadata hash differs for {name}")
        require(metadata.get("target") == compiled["target"], f"Triton metadata target differs for {name}")
        require(metadata.get("arch") == "gfx950", f"Triton metadata arch differs for {name}")
        require(metadata.get("backend_name") == "hip", f"Triton metadata backend differs for {name}")
        require(metadata.get("triton_version") == worker["triton"], f"Triton metadata version differs for {name}")
        require(metadata.get("num_warps") == 4, f"Triton metadata warp count differs for {name}")
        require(metadata.get("shared") == compiled["shared_memory_bytes"], f"Triton metadata shared memory differs for {name}")

        hsaco_path = source / matches[".hsaco"]
        hsaco = hsaco_path.read_bytes()
        require(hsaco.startswith(b"\x7fELF"), f"Triton binary is not ELF for {name}")
        require(len(hsaco) == compiled["binary_bytes"], f"Triton binary size differs for {name}")
        hsaco_sha = hashlib.sha256(hsaco).hexdigest()
        require(hsaco_sha == compiled["binary_sha256"], f"Triton binary SHA differs for {name}")

        group, _ = read_json(source / matches["group"], f"{name} cache group")
        children = group.get("child_paths")
        require(isinstance(children, dict), f"Triton cache group children differ for {name}")
        expected_children = {
            f"{name}{suffix}": str(original_cache / parent.name / f"{name}{suffix}")
            for suffix in KERNEL_CACHE_SUFFIXES
        }
        require(children == expected_children, f"Triton cache group topology differs for {name}")
        for suffix in (".source", ".ttir", ".ttgir", ".llir", ".amdgcn"):
            text = base._read_text(source / matches[suffix], f"{name}{suffix}")
            require(name in text, f"Triton cache stage lacks kernel identity: {name}{suffix}")
        kernel_records[name] = {
            "cache_key": parent.name,
            "cache_hash": compiled["cache_hash"],
            "hsaco_bytes": len(hsaco),
            "hsaco_sha256": hsaco_sha,
            "metadata": {
                "backend": "hip",
                "arch": "gfx950",
                "warp_size": 64,
                "num_warps": 4,
                "shared_memory_bytes": compiled["shared_memory_bytes"],
            },
        }

    helpers = sorted(set(cache_paths) - consumed)
    require(helpers, "Triton helper cache is missing")
    helper_names = [Path(relative).name for relative in helpers]
    require(all(TRITON_HELPER_RE.fullmatch(name) for name in helper_names), "Triton helper cache name differs")
    require(
        sum(name.startswith("hip_utils") and name.endswith(".so") for name in helper_names) == 1,
        "Triton HIP utility cache count differs",
    )
    for relative in helpers:
        parent = Path(relative).parent
        require(parent.parent == Path("triton-cache"), "Triton helper cache depth differs")
        require(CACHE_KEY_RE.fullmatch(parent.name) is not None, "Triton helper cache key is invalid")
        payload = (source / relative).read_bytes()
        require(payload.startswith(b"\x7fELF"), f"Triton helper is not ELF: {relative}")
    return {
        "file_count": len(cache_paths),
        "fresh_private_cache": True,
        "kernels": kernel_records,
        "helper_count": len(helpers),
        "all_invariants_correct": True,
    }


def parse_trace(path: Path, execution: Mapping[str, Any]) -> dict[str, Any]:
    lines = base._read_text(path, "Triton dispatch trace").splitlines()
    require(len(lines) == 13, "Triton dispatch trace record count differs")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AcceptanceError(f"trace line {index} is invalid JSON") from error
        require(isinstance(value, dict), f"trace line {index} is not an object")
        records.append(value)
    retired = [value for value in records if value.get("event") == "native_execution_retired"]
    terminal = [value for value in records if value.get("event") == "native_execution_session_complete"]
    require(len(retired) == 12 and len(terminal) == 1, "Triton trace event counts differ")
    job_uuid = execution.get("job_uuid")
    require(isinstance(job_uuid, str) and UUID_RE.fullmatch(job_uuid), "execution job UUID is invalid")
    identity_fields = (
        "daemon_uuid",
        "job_uuid",
        "epoch",
        "rank",
        "world_size",
        "connection_id",
        "owner_fd",
        "owner_generation",
    )
    first = records[0]
    for record in records:
        require(record.get("schema") == TRACE_SCHEMA, "Triton trace schema differs")
        require(record.get("source") == "upstream_rocr_kmt_aql", "Triton trace source differs")
        require(record.get("job_uuid") == job_uuid, "Triton trace job UUID differs")
        require(record.get("epoch") == 1 and record.get("rank") == 0 and record.get("world_size") == 1, "Triton trace topology differs")
        require(isinstance(record.get("daemon_uuid"), str) and UUID_RE.fullmatch(record["daemon_uuid"]), "Triton daemon UUID is invalid")
        require(isinstance(record.get("connection_id"), int) and record["connection_id"] > 0, "Triton connection identity is invalid")
        require(isinstance(record.get("owner_fd"), int) and record["owner_fd"] >= 0, "Triton owner FD is invalid")
        require(isinstance(record.get("owner_generation"), int) and record["owner_generation"] > 0, "Triton owner generation is invalid")
        require(all(record.get(field) == first.get(field) for field in identity_fields), "Triton trace identity drifted")
        require(record.get("kernel_executed") is True, "Triton trace did not execute on device")
    require([value.get("execution_ticket") for value in retired] == list(range(1, 13)), "Triton execution ticket order differs")
    require([value.get("queue_index") for value in retired] == EXPECTED_QUEUE_INDEX, "Triton queue index order differs")
    require([value.get("descriptor_abi") for value in retired] == [3] * 12, "Triton descriptor ABI differs")
    require([value.get("kernarg_size") for value in retired] == EXPECTED_KERNARG_SIZE, "Triton kernarg sequence differs")
    require([value.get("grid") for value in retired] == EXPECTED_GRID, "Triton grid sequence differs")
    require([value.get("workgroup") for value in retired] == EXPECTED_WORKGROUP, "Triton workgroup sequence differs")
    require([value.get("workgroups_completed") for value in retired] == EXPECTED_WORKGROUP_COMPLETIONS, "Triton completed workgroups differ")
    expected_signals = [(1, 0)] * 3 + [(0, 0)] * 3 + [(1, 0)] * 6
    require([(value.get("signal_before"), value.get("signal_after")) for value in retired] == expected_signals, "Triton signal sequence differs")
    request_ids: list[int] = []
    previous_retire = 0
    queue_ids: set[int] = set()
    for record in retired:
        require(record.get("dispatch_id") == 32, "Triton dispatch ID differs")
        require(record.get("packet_fetches") == 1, "Triton packet fetch count differs")
        require(record.get("command_processor_submissions") == 1, "Triton CP submission count differs")
        require(record.get("dispatcher_starts") == 1, "Triton dispatcher start count differs")
        require(record.get("doorbell_ack_durable") is True, "Triton doorbell ACK is not durable")
        require(record.get("queue_retired") is True and record.get("pins_released") is True, "Triton queue/pin retirement differs")
        require(record.get("cleanup_complete") is False, "Triton per-dispatch cleanup scope differs")
        request_id = record.get("source_request_id")
        require(isinstance(request_id, int) and request_id > 0, "Triton request identity is invalid")
        request_ids.append(request_id)
        queue_id = record.get("queue_object_id")
        require(isinstance(queue_id, int) and queue_id > 0, "Triton queue identity is invalid")
        queue_ids.add(queue_id)
        start, end, retire, sim_tick = (
            record.get("start_tick"),
            record.get("end_tick"),
            record.get("retire_tick"),
            record.get("sim_tick"),
        )
        require(all(isinstance(value, int) and value > 0 for value in (start, end, retire, sim_tick)), "Triton trace ticks are invalid")
        require(previous_retire <= start <= end == retire == sim_tick, "Triton trace tick order differs")
        previous_retire = retire
    require(request_ids == sorted(set(request_ids)), "Triton request IDs are not unique and increasing")
    require(len(queue_ids) == 1, "Triton quickstart used more than one queue")
    blit_objects = {retired[index].get("kernel_object") for index in (0, 1, 2, 6, 7, 8, 9, 10, 11)}
    require(len(blit_objects) == 1 and next(iter(blit_objects), 0) > 0, "Triton copy kernel identity differs")
    user_objects = [retired[index].get("kernel_object") for index in (3, 4, 5)]
    require(all(isinstance(value, int) and value > 0 for value in user_objects), "Triton user kernel identity is invalid")
    require(len(set(user_objects)) == 3 and not set(user_objects) & blit_objects, "Triton user kernel objects are not distinct")
    final = terminal[0]
    require(final.get("retired_dispatches") == 12, "Triton terminal retirement count differs")
    require(final.get("owner_disconnected") is True and final.get("cleanup_complete") is True, "Triton terminal cleanup differs")
    require(final.get("sim_tick", 0) >= previous_retire, "Triton terminal tick precedes retirement")
    return {
        "record_count": 13,
        "retired_dispatches": 12,
        "session_complete": 1,
        "source": "upstream_rocr_kmt_aql",
        "descriptor_abi": 3,
        "execution_tickets": list(range(1, 13)),
        "queue_indices": EXPECTED_QUEUE_INDEX,
        "user_kernel_execution_tickets": dict(zip(KERNEL_NAMES, (4, 5, 6))),
        "terminal_tick": final["sim_tick"],
        "daemon_uuid": final["daemon_uuid"],
        "connection_id": final["connection_id"],
        "host_fallback_count": 0,
        "all_invariants_correct": True,
    }


def validate_stats(path: Path, terminal_tick: int) -> dict[str, Any]:
    text = base._read_text(path, "Triton gem5 stats")
    values: dict[str, str] = {}
    required = {
        "simTicks",
        "finalTick",
        "hostSeconds",
        "system.host_gpu_bridge.host_fallback_count",
    }
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in required:
            values[fields[0]] = fields[1]
    require(set(values) == required, "required Triton gem5 stats are missing")
    sim_ticks = int(values["simTicks"])
    final_tick = int(values["finalTick"])
    host_seconds = float(values["hostSeconds"])
    fallback = int(values["system.host_gpu_bridge.host_fallback_count"])
    require(sim_ticks == final_tick == terminal_tick, "Triton gem5 terminal tick differs")
    require(math.isfinite(host_seconds) and host_seconds > 0.0, "Triton gem5 hostSeconds is invalid")
    require(fallback == 0, "Triton gem5 host fallback is nonzero")
    return {
        "sim_ticks": sim_ticks,
        "host_seconds": host_seconds,
        "host_fallback_count": 0,
        "correct": True,
    }


def validate_gem5_log(
    path: Path, execution: Mapping[str, Any], trace: Mapping[str, Any]
) -> dict[str, Any]:
    text = base._read_text(path, "Triton gem5 log")
    require("panic:" not in text and "fatal:" not in text, "Triton gem5 log contains a fatal failure")
    require(text.count("host-gpu-handshake status=OK") == 1, "Triton gem5 handshake count differs")
    match = re.search(
        r"host-gpu-dispatch-exit cause=host GPU dispatch session complete code=0 tick=([0-9]+)",
        text,
    )
    require(match is not None and int(match.group(1)) == trace["terminal_tick"], "Triton clean exit tick differs")
    require(f"job_uuid={execution['job_uuid']}" in text, "Triton gem5 job UUID differs")
    require(f"daemon_uuid={trace['daemon_uuid']}" in text, "Triton gem5 daemon UUID differs")
    require(str(GEM5_BINARY.resolve()) in text and str(GEM5_CONFIG.resolve()) in text, "Triton gem5 command paths differ")
    gem5_pid = execution.get("gem5_pid")
    require(isinstance(gem5_pid, int) and gem5_pid > 1, "Triton gem5 PID is invalid")
    require(re.search(rf"gem5 executing on .+, pid {gem5_pid}(?:\n|$)", text) is not None, "Triton gem5 log PID differs")
    require("command line: " + " ".join(execution["gem5_argv"]) in text, "Triton gem5 argv differs")
    return {"handshake_ok": True, "clean_exit": True, "exit_tick": trace["terminal_tick"]}


def expected_worker_argv(identity: Mapping[str, Any]) -> list[str]:
    files = identity["files"]
    return [
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
        (
            'set -eu; source "$1"; shift; '
            'export TRITON_CACHE_DIR="$GEMSIM_RUN_TRITON_CACHE_DIR"; '
            "unset GEMSIM_RUN_TRITON_CACHE_DIR; exec \"$@\""
        ),
        "upstream-triton-worker",
        files["activation"]["path"],
        files["python"]["path"],
        files["quickstart"]["path"],
    ]


def validate_execution(
    execution: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    required_execution = {
        "job_uuid", "epoch", "rank", "world_size", "execution_root", "endpoint",
        "trace_path", "m5out_path", "gem5_argv", "worker_argv", "gem5_environment",
        "worker_environment", "gem5_pid", "gem5_start_time_ticks", "gem5_process_group",
        "worker_pid", "worker_start_time_ticks", "worker_process_group", "worker_exit_code",
        "gem5_exit_code", "worker_timeout_seconds", "gem5_exit_timeout_seconds",
        "startup_timeout_seconds",
    }
    require(set(execution) == required_execution, "Triton execution keys differ")
    require(execution["epoch"] == 1 and execution["rank"] == 0 and execution["world_size"] == 1, "Triton execution topology differs")
    require(execution["worker_exit_code"] == 0 and execution["gem5_exit_code"] == 0, "Triton process exit differs")
    for role in ("gem5", "worker"):
        pid = execution[f"{role}_pid"]
        require(isinstance(pid, int) and pid > 1, f"Triton {role} PID is invalid")
        require(execution[f"{role}_process_group"] == pid, f"Triton {role} process group differs")
        require(isinstance(execution[f"{role}_start_time_ticks"], int) and execution[f"{role}_start_time_ticks"] > 0, f"Triton {role} start time is invalid")
    root = Path(execution["execution_root"])
    require(root.parent == Path("/tmp") and root.name.startswith("gs-upstream-triton-"), "Triton execution root differs")
    require(Path(execution["endpoint"]) == root / "bridge.sock", "Triton endpoint differs")
    require(Path(execution["trace_path"]) == root / "dispatch-trace.jsonl", "Triton trace path differs")
    require(Path(execution["m5out_path"]) == root / "m5out", "Triton m5out path differs")
    require(execution["worker_argv"] == expected_worker_argv(identity), "Triton worker argv differs")
    expected_gem5 = lifecycle.gem5_argv(
        binary=GEM5_BINARY,
        config=GEM5_CONFIG,
        execution_root=root,
        endpoint=root / "bridge.sock",
        trace_path=root / "dispatch-trace.jsonl",
        job_uuid=execution["job_uuid"],
    )
    require(execution["gem5_argv"] == expected_gem5, "Triton gem5 argv differs")
    worker_environment = execution["worker_environment"]
    require(worker_environment.get("SAGR_GENERIC_BRIDGE_ENDPOINT") == execution["endpoint"], "Triton worker endpoint differs")
    require(worker_environment.get("GEMSIM_RUN_TRITON_CACHE_DIR") == str(root / "triton-cache"), "Triton private cache path differs")
    require(worker_environment.get("PYTHONDONTWRITEBYTECODE") == "1" and worker_environment.get("PYTHONNOUSERSITE") == "1", "Triton worker isolation differs")
    require("TRITON_CACHE_DIR" not in worker_environment, "ambient Triton cache was supplied before activation")
    for forbidden in ("CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX", "PYTHONPATH"):
        require(forbidden not in worker_environment, f"Triton worker inherited {forbidden}")
    cleanup_expected = {
        "worker_reaped": True,
        "gem5_reaped": True,
        "worker_process_group_absent": True,
        "gem5_process_group_absent": True,
        "endpoint_absent": True,
        "worker_forced_termination": False,
        "gem5_forced_termination": False,
        "all_clear": True,
    }
    require(cleanup == cleanup_expected, "Triton process cleanup differs")


def validate_source(
    source: Path,
    *,
    snapshot: Callable[[], dict[str, Any]] = identity_snapshot,
) -> dict[str, Any]:
    source = source.resolve(strict=True)
    manifest, manifest_payload = read_json(source / "result-manifest.json", "Triton run manifest")
    require(
        set(manifest)
        == {
            "schema",
            "status",
            "claim_scope",
            "ordinary_upstream_triton_amd_executed",
            "runtime_gem5_bridge_modified_for_profile",
            "pytorch_rocm_multiop_accepted",
            "triton_upstream_amd_accepted",
            "torch_compile_accepted",
            "vllm_accepted",
            "sglang_accepted",
            "model_accepted",
            "identity_preflight",
            "identity_postflight",
            "execution",
            "cleanup",
            "artifacts",
        },
        "Triton run manifest keys differ",
    )
    require(manifest["schema"] == RUN_SCHEMA, "Triton run schema differs")
    require(manifest["status"] == "success", "Triton run did not succeed")
    require(manifest["claim_scope"] == CLAIM_SCOPE, "Triton claim scope differs")
    require(manifest["ordinary_upstream_triton_amd_executed"] is True, "ordinary Triton path did not execute")
    require(manifest["runtime_gem5_bridge_modified_for_profile"] is False, "run claims a profile-specific bridge change")
    for name in (
        "pytorch_rocm_multiop_accepted",
        "triton_upstream_amd_accepted",
        "torch_compile_accepted",
        "vllm_accepted",
        "sglang_accepted",
        "model_accepted",
    ):
        require(manifest[name] is False, f"Triton source run overclaims {name}")
    identity = manifest["identity_preflight"]
    require(isinstance(identity, dict) and identity == manifest["identity_postflight"], "Triton run identity drifted")
    require(snapshot() == identity, "live Triton execution identity differs")
    require(identity["product"]["native_role"] == "rocr_kmd_boundary_only", "Triton product boundary role differs")
    for role in ("hip_library", "comgr_library", "rccl_library"):
        provider = identity["product"]["sdk_libraries"][role].get("provider")
        require(provider in OFFICIAL_SDK_PROVIDERS, f"{role} is not an approved official ROCm provider")
    source_contract = validate_quickstart_source(Path(identity["files"]["quickstart"]["path"]))
    execution = manifest["execution"]
    cleanup = manifest["cleanup"]
    require(isinstance(execution, dict) and isinstance(cleanup, dict), "Triton execution metadata is invalid")
    validate_execution(execution, cleanup, identity)
    artifacts, cache_paths = validate_artifacts(source, manifest)
    worker = validate_worker(source / "worker.log", identity)
    jit_cache = validate_jit_cache(source, cache_paths, execution, worker)
    trace = parse_trace(source / "dispatch-trace.jsonl", execution)
    stats = validate_stats(source / "m5out/stats.txt", trace["terminal_tick"])
    gem5_log = validate_gem5_log(source / "gem5.log", execution, trace)
    return {
        "manifest_bytes": len(manifest_payload),
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "identity": identity,
        "artifacts": artifacts,
        "artifact_paths": tuple(sorted(artifacts)),
        "source_contract": source_contract,
        "worker": worker,
        "jit_cache": jit_cache,
        "trace": trace,
        "stats": stats,
        "gem5_log": gem5_log,
        "output_correct": True,
    }


def publish(
    source: Path, output: Path, validation: Mapping[str, Any]
) -> dict[str, Any]:
    require(output.is_absolute() and not os.path.lexists(output), "Triton acceptance output must be absent and absolute")
    parent = output.parent.resolve(strict=True)
    require(output.parent == parent, "Triton acceptance parent contains a symlink")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        copied: dict[str, dict[str, Any]] = {}
        for relative in ("result-manifest.json",) + validation["artifact_paths"]:
            record = base._copy_file(source / relative, temporary / relative)
            record["path"] = relative
            copied[relative] = record
        result = {
            "schema": RESULT_SCHEMA,
            "status": "accepted",
            "claim_scope": CLAIM_SCOPE,
            "source_manifest": {
                "bytes": validation["manifest_bytes"],
                "sha256": validation["manifest_sha256"],
            },
            "identity": validation["identity"],
            "source_contract": validation["source_contract"],
            "worker": validation["worker"],
            "jit_cache": validation["jit_cache"],
            "trace": validation["trace"],
            "stats": validation["stats"],
            "gem5_log": validation["gem5_log"],
            "unchanged_upstream_triton_amd_multiop_accepted": True,
            "official_rocm_pytorch_triton_hip_comgr_retained": True,
            "rocr_kmd_boundary_replaced_only": True,
            "runtime_gem5_bridge_modified_for_profile": False,
            "target_feedback_from_oracle": False,
            "torch_compile_accepted": False,
            "vllm_accepted": False,
            "sglang_accepted": False,
            "model_accepted": False,
            "claim_boundary": (
                f"unchanged upstream Triton {validation['identity']['product']['runtime_probe']['triton']} AMD HIPDriver add, masked "
                "branching transform, and reduction on one gfx950 simulated "
                "device; fresh private JIT cache and exact HSACO identities, "
                "bitwise host-oracle agreement, 12 retired standard AQL "
                "dispatches, zero host fallback, and clean session teardown; "
                "no complete Triton API, torch.compile, framework, TP, or "
                "model acceptance"
            ),
            "output_correct": True,
        }
        copied["result.json"] = base._write_bytes(
            temporary / "result.json", canonical_json(result)
        )
        evidence_manifest = {
            "schema": MANIFEST_SCHEMA,
            "artifacts": copied,
            "complete": True,
        }
        base._write_bytes(
            temporary / "manifest.json", canonical_json(evidence_manifest)
        )
        lifecycle.fsync_tree(temporary)
        base.rename_noreplace(temporary, output)
        temporary = None
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return result
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        validation = validate_source(arguments.source)
        result = publish(arguments.source.resolve(), arguments.output, validation)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (
        AcceptanceError,
        base.AcceptanceError,
        FileExistsError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"upstream Triton AMD acceptance failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
