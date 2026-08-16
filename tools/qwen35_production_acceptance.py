#!/usr/bin/env python3
"""Validate and run the continuous Qwen3.5 backbone acceptance window."""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
RUNNER = ROOT / "examples/triton/qwen35_vllm_model_forward.py"
FINAL_NORM_RUNNER = ROOT / "examples/triton/qwen35_final_norm_live_diff.py"
GOLDEN = ROOT / "artifacts/qwen35-nvidia-golden/20260812-decode4-max24-v1"
PREFILL_GOLDEN = (
    ROOT / "artifacts/qwen35-nvidia-golden/20260812-prefill2-max24-v1"
)
RUNNER_SHA256 = "878acfa8d37a81a204d2aff1844d7618fa571e274a05a8ba47f5f31312649343"
STRICT_EVIDENCE_MANIFEST = ROOT / "artifacts/qwen35-layer-diff/bridge-m4-evidence-manifest.json"
STRICT_IDENTITY_SHA256 = ""
STRICT_REQUEST_SHA256 = ""
STRICT_RUN_ID = ""
STRICT_RESULTS: tuple[tuple[str, int, str, int, int], ...] = ()
FINAL_NORM_RESULT: tuple[str, int, str] = ("", 0, "")
FINAL_CHECKPOINT: tuple[str, int, str, int, str] = ("", 0, "", 0, "")
FINAL_CHECKPOINT_REQUEST_SHA256 = ""
GOLDEN_FILES = {
    "metadata.json": (
        154152,
        "8df6b70919203c7bd0369db8191b231abe4bf36e026b7c68fb305619ece65a66",
    ),
    "results.safetensors": (
        21497464,
        "60e9ba36e659d5ef93297e3beb4e056558dd5fdabdc4bbf2b6b3a65b9fb3210f",
    ),
}
PREFILL_GOLDEN_FILES = {
    "metadata.json": (
        153970,
        "ff326833bc2a47f760c240af5f441f48a187e686e523a4c0778185ce392d2251",
    ),
    "results.safetensors": (
        21284456,
        "c401c34db3f137ad4b2e371b32e3bcc9c796ba56a2101d50e7e8fc927091cbe7",
    ),
}
PACKAGE_HASH_DOMAIN = b"amdgpu-sim.qwen35-layer-oracle-package.v1\0"
TRACE_SCHEMA = "amdgpu-sim.generic-kernel-execution-trace.v1"
TRAJECTORY_POLICY_ID = "qwen35-bf16-continuous-trajectory-v1"
TRAJECTORY_MAX_RELATIVE_L2 = 0.15
TRAJECTORY_MIN_COSINE = 0.98
EXPECTED_DISPATCHES = 834
HEX64 = re.compile(r"[0-9a-f]{64}")
AT_FDCWD = -100
RENAME_NOREPLACE = 1
PLUGIN_SOURCE_PATHS = (
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/adapters.py",
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/attention.py",
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/kernels.py",
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/model.py",
    "plugins/framework/gemsim_vllm/src/gemsim_vllm/ops.py",
)
FORMULA_SOURCE_PATHS = (
    "projects/vllm/vllm/model_executor/layers/layernorm.py",
    "projects/vllm/vllm/model_executor/models/qwen3_5.py",
    "projects/vllm/vllm/model_executor/models/qwen3_next.py",
    "projects/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
)


class AcceptanceError(RuntimeError):
    """The acceptance input or output is malformed, stale, or rejected."""


@dataclass(frozen=True)
class FileRecord:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Gem5Child:
    pid: int
    argv: tuple[str, ...]
    executable: str
    executable_sha256: str
    config: str
    config_sha256: str
    run_dir: str
    trace_path: str
    outdir: str
    endpoint: str
    job_uuid: str
    epoch: int
    rank: int
    world_size: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def source_set_identity(paths: Sequence[str]) -> dict[str, str]:
    records: dict[str, str] = {}
    for relative in paths:
        path = ROOT / relative
        metadata = os.lstat(path)
        require(stat.S_ISREG(metadata.st_mode) and not path.is_symlink(), f"identity source is unsafe: {relative}")
        records[relative] = file_sha256(path)
    return records


def current_runtime_paths() -> tuple[Path, Path, Path, Path, Path]:
    prefix_text = subprocess.check_output(
        ["/usr/bin/bash", str(ROOT / "scripts/setup_rocm_env.sh"), "--print-prefix"],
        cwd=ROOT,
        text=True,
    ).strip()
    prefix = Path(prefix_text).resolve(strict=True)
    manifest_path = prefix / "manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "prefix manifest is unsafe")
    manifest = strict_json(manifest_path)
    runtime_record = manifest.get("artifacts", {}).get("runtime_library")
    gem5_record = manifest.get("managed_inputs", {}).get("gem5_binary")
    config_record = manifest.get("managed_inputs", {}).get("gem5_config")
    require(
        isinstance(runtime_record, dict)
        and isinstance(gem5_record, dict)
        and isinstance(config_record, dict),
        "prefix execution inputs missing",
    )
    runtime_path = Path(str(runtime_record.get("path", ""))).resolve(strict=True)
    gem5_path = Path(str(gem5_record.get("path", ""))).resolve(strict=True)
    config_unresolved = Path(str(config_record.get("path", "")))
    config_metadata = os.lstat(config_unresolved)
    config_path = config_unresolved.resolve(strict=True)
    require(
        runtime_path.is_file()
        and gem5_path.is_file()
        and stat.S_ISREG(config_metadata.st_mode)
        and not config_unresolved.is_symlink()
        and config_path.is_file(),
        "prefix execution input is not a safe regular file",
    )
    require(
        file_sha256(config_path) == config_record.get("sha256"),
        "live gem5 config differs from the prefix manifest",
    )
    return prefix, manifest_path, runtime_path, gem5_path, config_path


def current_execution_identity(checkpoint_identity: Mapping[str, Any]) -> dict[str, Any]:
    expected = checkpoint_identity.get("implementation")
    require(isinstance(expected, dict), "checkpoint implementation identity missing")
    _, manifest_path, runtime_path, gem5_path, _ = current_runtime_paths()
    vllm = ROOT / "projects/vllm"
    vllm_head = subprocess.check_output(
        ["git", "-C", str(vllm), "rev-parse", "HEAD"], text=True
    ).strip()
    vllm_tree = subprocess.check_output(
        ["git", "-C", str(vllm), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    observed = {
        "architecture": "GemsimQwen3_5ForCausalLM",
        "runner_sha256": file_sha256(RUNNER),
        "plugin_sha256": canonical_sha256(source_set_identity(PLUGIN_SOURCE_PATHS)),
        "vllm_git_head": vllm_head,
        "vllm_tree_sha256": canonical_sha256(
            {
                "git_tree": vllm_tree,
                "formula_sources": source_set_identity(FORMULA_SOURCE_PATHS),
            }
        ),
        "gem5_binary_sha256": file_sha256(gem5_path),
        "runtime_dso_sha256": file_sha256(runtime_path),
        "prefix_manifest_sha256": file_sha256(manifest_path),
    }
    require(observed == expected, "live execution identity drifted from strict evidence")
    return observed


def safe_file(relative: str, expected_bytes: int, expected_sha256: str) -> FileRecord:
    require(not Path(relative).is_absolute(), f"artifact path must be relative: {relative}")
    path = ROOT / relative
    metadata = os.lstat(path)
    require(stat.S_ISREG(metadata.st_mode), f"artifact is not a regular file: {relative}")
    require(not path.is_symlink(), f"artifact is a symlink: {relative}")
    require(path.resolve(strict=True).is_relative_to(ROOT), "artifact escapes repository")
    observed_sha256 = file_sha256(path)
    require(metadata.st_size == expected_bytes, f"artifact size mismatch: {relative}")
    require(observed_sha256 == expected_sha256, f"artifact hash mismatch: {relative}")
    return FileRecord(relative, metadata.st_size, observed_sha256)


def strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AcceptanceError(f"invalid JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def package_sha256(path: Path, json_filename: str) -> str:
    digest = hashlib.sha256(PACKAGE_HASH_DOMAIN)
    for filename in (json_filename, "tensors.safetensors"):
        content = (path / filename).read_bytes()
        encoded = filename.encode("ascii")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<Q", len(content)))
        digest.update(content)
    return digest.hexdigest()


def validate_checkpoint_package(
    record: Mapping[str, Any],
    *,
    expected_previous_manifest: str | None,
) -> tuple[list[FileRecord], dict[str, Any]]:
    path = Path(str(record.get("path", "")))
    require(path.is_absolute(), "checkpoint path is not absolute")
    resolved = path.resolve(strict=True)
    require(path == resolved and resolved.is_relative_to(ROOT), "checkpoint path is unsafe")
    require(not path.is_symlink(), "checkpoint directory is a symlink")
    require(
        {item.name for item in path.iterdir()} == {"manifest.json", "state.safetensors"},
        "checkpoint file set mismatch",
    )
    manifest_path = path / "manifest.json"
    state_path = path / "state.safetensors"
    for artifact in (manifest_path, state_path):
        metadata = os.lstat(artifact)
        require(stat.S_ISREG(metadata.st_mode) and not artifact.is_symlink(), "checkpoint artifact is unsafe")
    manifest_sha = file_sha256(manifest_path)
    state_sha = file_sha256(state_path)
    require(manifest_sha == record.get("manifest_sha256"), "checkpoint manifest hash mismatch")
    require(state_sha == record.get("state_sha256"), "checkpoint state hash mismatch")
    manifest = strict_json(manifest_path)
    boundary = manifest.get("boundary")
    lineage = manifest.get("lineage")
    state = manifest.get("artifacts", {}).get("state")
    require(isinstance(boundary, dict) and isinstance(lineage, dict), "checkpoint boundary or lineage missing")
    require(isinstance(state, dict), "checkpoint state binding missing")
    require(manifest.get("identity_sha256") == STRICT_IDENTITY_SHA256, "checkpoint identity mismatch")
    require(manifest.get("request_sha256") == FINAL_CHECKPOINT_REQUEST_SHA256, "checkpoint request mismatch")
    require(lineage.get("run_id") == STRICT_RUN_ID, "checkpoint run ID mismatch")
    require(boundary.get("after_layer") == record.get("after_layer"), "checkpoint boundary mismatch")
    require(lineage.get("previous_manifest_sha256") == expected_previous_manifest, "checkpoint lineage mismatch")
    require(state.get("sha256") == state_sha, "checkpoint state descriptor hash mismatch")
    require(state.get("bytes") == state_path.stat().st_size, "checkpoint state descriptor size mismatch")
    return [
        FileRecord(str(manifest_path.relative_to(ROOT)), manifest_path.stat().st_size, manifest_sha),
        FileRecord(str(state_path.relative_to(ROOT)), state_path.stat().st_size, state_sha),
    ], manifest


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    ) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def numeric_comparison_valid(
    value: object, *, expected_atol: float, exact_sha_required: bool = False
) -> bool:
    if not isinstance(value, dict):
        return False
    numeric_keys = (
        "atol",
        "rtol",
        "max_relative_l2",
        "cosine_similarity",
        "max_abs_error",
        "relative_l2_error",
    )
    if not all(
        isinstance(value.get(key), (int, float))
        and not isinstance(value.get(key), bool)
        and math.isfinite(float(value[key]))
        for key in numeric_keys
    ):
        return False
    hashes = (value.get("actual_sha256"), value.get("expected_sha256"))
    return bool(
        value.get("correct") is True
        and value.get("mismatch_count") == 0
        and value.get("nonfinite_count") == 0
        and float(value["atol"]) == expected_atol
        and float(value["rtol"]) == 0.03
        and float(value["max_relative_l2"]) == 0.03
        and float(value["relative_l2_error"]) <= 0.03
        and float(value["cosine_similarity"]) >= 0.98
        and all(isinstance(item, str) and HEX64.fullmatch(item) for item in hashes)
        and (not exact_sha_required or hashes[0] == hashes[1])
    )


def validate_oracle_packages(layer: Mapping[str, Any]) -> None:
    request = layer.get("oracle_request")
    response = layer.get("oracle_response")
    execution = layer.get("oracle_execution")
    require(isinstance(request, dict) and isinstance(response, dict), "oracle records missing")
    require(isinstance(execution, dict) and execution.get("exit_code") == 0, "oracle failed")
    require(response.get("target_feedback") is False, "oracle response feedback enabled")
    require(layer.get("oracle_feedback_to_target") is False, "oracle feedback enabled")
    require(request.get("request_id") == response.get("request_id"), "oracle request ID mismatch")
    require(request.get("identity_sha256") == response.get("identity_sha256"), "oracle identity mismatch")
    for record, json_name in ((request, "request.json"), (response, "response.json")):
        path = Path(str(record.get("path", "")))
        require(path.is_absolute(), "oracle package path is not absolute")
        resolved = path.resolve(strict=True)
        require(resolved.is_relative_to(ROOT), "oracle package escapes repository")
        require(not path.is_symlink() and path == resolved, "oracle package path is unsafe")
        require(
            {item.name for item in path.iterdir()} == {json_name, "tensors.safetensors"},
            "oracle package file set mismatch",
        )
        for artifact in (path / json_name, path / "tensors.safetensors"):
            metadata = os.lstat(artifact)
            require(
                stat.S_ISREG(metadata.st_mode) and not artifact.is_symlink(),
                "oracle package contains an unsafe artifact",
            )
        document = strict_json(path / json_name)
        payload = document.get("payload")
        require(isinstance(payload, dict), "oracle package payload is missing")
        tensor_path = path / "tensors.safetensors"
        require(tensor_path.stat().st_size == payload.get("bytes"), "oracle tensor size mismatch")
        require(file_sha256(tensor_path) == payload.get("sha256"), "oracle tensor hash mismatch")
        require(record.get("payload_sha256") == payload.get("sha256"), "oracle payload record mismatch")
        require(record.get("package_sha256") == package_sha256(path, json_name), "oracle package hash mismatch")
        require(document.get("request_id") == request.get("request_id"), "oracle document request mismatch")


def load_evidence_manifest() -> dict[str, Any]:
    """Load one immutable layer evidence bundle and derive all bindings from it."""
    require(STRICT_EVIDENCE_MANIFEST.is_file() and not STRICT_EVIDENCE_MANIFEST.is_symlink(), "layer evidence manifest is missing or unsafe")
    manifest = strict_json(STRICT_EVIDENCE_MANIFEST)
    require(manifest.get("schema") == "amdgpu-sim.qwen35-layer-evidence-manifest.v1", "layer evidence manifest schema mismatch")
    segments = manifest.get("segments")
    require(isinstance(segments, list) and segments, "layer evidence segments missing")
    previous_stop = -1
    for segment in segments:
        require(isinstance(segment, dict), "layer evidence segment is malformed")
        start, stop = segment.get("start_layer"), segment.get("stop_after_layer")
        require(type(start) is int and type(stop) is int and start == previous_stop + 1 and start <= stop, "layer evidence layer lineage has a gap or overlap")
        previous_stop = stop
        require(isinstance(segment.get("path"), str) and type(segment.get("bytes")) is int and HEX64.fullmatch(str(segment.get("sha256"))) is not None, "layer evidence segment binding is malformed")
    require(previous_stop == 23, "layer evidence does not cover layers 0..23")
    final_checkpoint = manifest.get("final_checkpoint")
    final_norm = manifest.get("final_norm")
    require(isinstance(final_checkpoint, dict) and isinstance(final_norm, dict), "layer evidence terminal bindings missing")
    require(
        isinstance(final_checkpoint.get("path"), str)
        and type(final_checkpoint.get("manifest_bytes")) is int
        and type(final_checkpoint.get("state_bytes")) is int
        and HEX64.fullmatch(str(final_checkpoint.get("manifest_sha256"))) is not None
        and HEX64.fullmatch(str(final_checkpoint.get("state_sha256"))) is not None,
        "layer evidence checkpoint binding is malformed",
    )
    require(
        isinstance(final_norm.get("path"), str)
        and type(final_norm.get("bytes")) is int
        and HEX64.fullmatch(str(final_norm.get("sha256"))) is not None,
        "layer evidence final norm binding is malformed",
    )
    return manifest


def validate_strict_prerequisite() -> dict[str, Any]:
    global STRICT_IDENTITY_SHA256, STRICT_REQUEST_SHA256, STRICT_RUN_ID
    global STRICT_RESULTS, FINAL_NORM_RESULT, FINAL_CHECKPOINT, FINAL_CHECKPOINT_REQUEST_SHA256
    evidence = load_evidence_manifest()
    STRICT_IDENTITY_SHA256 = str(evidence["identity_sha256"])
    STRICT_REQUEST_SHA256 = str(evidence["request_identity_sha256"])
    STRICT_RUN_ID = str(evidence["run_id"])
    STRICT_RESULTS = tuple(
        (
            str(segment["path"]),
            int(segment["bytes"]),
            str(segment["sha256"]),
            int(segment["start_layer"]),
            int(segment["stop_after_layer"]),
        )
        for segment in evidence["segments"]
    )
    final_checkpoint = evidence["final_checkpoint"]
    FINAL_CHECKPOINT = (
        str(final_checkpoint["path"]),
        int(final_checkpoint["manifest_bytes"]),
        str(final_checkpoint["manifest_sha256"]),
        int(final_checkpoint["state_bytes"]),
        str(final_checkpoint["state_sha256"]),
    )
    FINAL_CHECKPOINT_REQUEST_SHA256 = str(final_checkpoint["request_sha256"])
    final_norm = evidence["final_norm"]
    FINAL_NORM_RESULT = (
        str(final_norm["path"]),
        int(final_norm["bytes"]),
        str(final_norm["sha256"]),
    )
    require(file_sha256(RUNNER) == RUNNER_SHA256, "strict evidence runner identity drifted")
    artifacts: list[FileRecord] = []
    seen_layers: list[int] = []
    previous_manifest: str | None = None
    checkpoint_artifacts: dict[str, FileRecord] = {}
    for index, (relative, size, digest, start, stop) in enumerate(STRICT_RESULTS):
        artifacts.append(safe_file(relative, size, digest))
        result = strict_json(ROOT / relative)
        require(result.get("schema") == "amdgpu-sim.gemsim-vllm-qwen35-explicit-layer-mode.v1", "strict result schema mismatch")
        require(result.get("inference_mode") == ("debug-layer-diff" if index == 0 else "debug-resume"), "strict mode mismatch")
        require(result.get("start_layer") == start and result.get("stop_after_layer") == stop, "strict layer range mismatch")
        require(result.get("layers_completed") == stop - start + 1, "strict layer count mismatch")
        require(result.get("run_id") == STRICT_RUN_ID, "strict run ID mismatch")
        require(result.get("identity_sha256") == STRICT_IDENTITY_SHA256, "strict identity mismatch")
        require(result.get("request_identity_sha256") == STRICT_REQUEST_SHA256, "strict request mismatch")
        require(result.get("execution_success") is True and result.get("output_correct") is True, "strict result failed")
        require(result.get("first_failure") is None, "strict result retained a failure")
        require(result.get("oracle_used") is True and result.get("oracle_feedback_to_target") is False, "strict oracle boundary mismatch")
        require(all(result.get(key) == 0 for key in ("fallback_count", "cpu_fallback_count", "nvidia_fallback_count")), "strict fallback counter nonzero")
        checkpoints = result.get("checkpoints")
        records = result.get("layer_records")
        require(isinstance(checkpoints, list) and isinstance(records, list), "strict records missing")
        if index:
            require(checkpoints[0].get("manifest_sha256") == previous_manifest, "strict checkpoint lineage broken")
            require(checkpoints[0].get("strict_validation_passed") is True, "strict restore was not validated")
            require(checkpoints[0].get("rng_restored") is True, "strict RNG was not restored")
        for checkpoint_index, checkpoint_record in enumerate(checkpoints):
            if index and checkpoint_index == 0:
                checkpoint_path = Path(str(checkpoint_record.get("path"))).resolve(
                    strict=True
                )
                manifest_relative = str(
                    (checkpoint_path / "manifest.json").relative_to(ROOT)
                )
                state_relative = str(
                    (checkpoint_path / "state.safetensors").relative_to(ROOT)
                )
                require(
                    manifest_relative in checkpoint_artifacts
                    and state_relative in checkpoint_artifacts,
                    "restored checkpoint was not validated by the previous segment",
                )
                continue
            expected_previous = (
                None
                if checkpoint_index == 0
                else checkpoints[checkpoint_index - 1].get("manifest_sha256")
            )
            package_artifacts, checkpoint_document = validate_checkpoint_package(
                checkpoint_record,
                expected_previous_manifest=expected_previous,
            )
            for artifact in package_artifacts:
                existing = checkpoint_artifacts.get(artifact.path)
                require(existing is None or existing == artifact, "checkpoint artifact identity conflict")
                checkpoint_artifacts[artifact.path] = artifact
            require(
                checkpoint_document.get("boundary", {}).get("after_layer")
                == checkpoint_record.get("after_layer"),
                "checkpoint result boundary mismatch",
            )
        require([item.get("layer") for item in records] == list(range(start, stop + 1)), "strict layer order mismatch")
        for layer in records:
            seen_layers.append(layer["layer"])
            comparison = layer.get("comparison")
            guard = layer.get("cache_mutation_guard")
            require(isinstance(comparison, dict) and comparison.get("correct") is True, "strict layer comparison failed")
            require(comparison.get("target_feedback") is False, "strict comparison feedback enabled")
            require(isinstance(guard, dict) and guard.get("correct") is True, "strict cache guard failed")
            require(guard.get("all_storage_identity_preserved") is True, "strict cache alias changed")
            require(guard.get("non_target_cache_unchanged") is True, "strict non-target cache changed")
            for cache_record in guard.get("records", []):
                if cache_record.get("layer") != layer["layer"]:
                    require(
                        all(
                            item.get("content_unchanged") is True
                            and item.get("before_sha256") == item.get("after_sha256")
                            for item in cache_record.get("tensors", [])
                        ),
                        "strict non-target cache is not bitwise unchanged",
                    )
            require(numeric_comparison_valid(comparison.get("hidden"), expected_atol=0.03125), "strict hidden comparison invalid")
            require(numeric_comparison_valid(comparison.get("residual"), expected_atol=0.03125), "strict residual comparison invalid")
            mutable = comparison.get("mutable_state")
            require(isinstance(mutable, dict) and mutable, "strict mutable-state comparison missing")
            expected_states = {"kv_cache": 0.03125} if layer.get("layer_type") == "full_attention" else {"conv_state": 0.03125, "recurrent_state": 0.0001}
            require(set(mutable) == set(expected_states), "strict mutable-state roles mismatch")
            for name, atol in expected_states.items():
                require(numeric_comparison_valid(mutable[name], expected_atol=atol), f"strict {name} comparison invalid")
            validate_oracle_packages(layer)
            published = layer.get("checkpoint")
            require(isinstance(published, dict) and published.get("manifest_sha256") == checkpoints[layer["layer"] - start + 1].get("manifest_sha256"), "strict published checkpoint mismatch")
        previous_manifest = checkpoints[-1].get("manifest_sha256")
    require(seen_layers == list(range(24)), "strict evidence does not cover layers 0..23 exactly once")
    artifacts.extend(checkpoint_artifacts.values())

    checkpoint_relative, manifest_size, manifest_sha, state_size, state_sha = FINAL_CHECKPOINT
    safe_file(f"{checkpoint_relative}/manifest.json", manifest_size, manifest_sha)
    safe_file(f"{checkpoint_relative}/state.safetensors", state_size, state_sha)
    checkpoint = strict_json(ROOT / checkpoint_relative / "manifest.json")
    require(checkpoint.get("identity_sha256") == STRICT_IDENTITY_SHA256, "final checkpoint identity mismatch")
    require(
        checkpoint.get("request_sha256") == FINAL_CHECKPOINT_REQUEST_SHA256,
        "final checkpoint request mismatch",
    )
    require(checkpoint.get("lineage", {}).get("run_id") == STRICT_RUN_ID, "final checkpoint run ID mismatch")
    require(checkpoint.get("boundary", {}).get("after_layer") == 23, "final checkpoint boundary mismatch")
    require(checkpoint.get("boundary", {}).get("next_layer") == 24, "final checkpoint next layer mismatch")
    require(checkpoint.get("boundary", {}).get("resume_action") == "final_norm", "final checkpoint action mismatch")
    require(checkpoint.get("artifacts", {}).get("state", {}).get("sha256") == state_sha, "final checkpoint state binding mismatch")
    observed_execution_identity = current_execution_identity(checkpoint.get("identity", {}))

    relative, size, digest = FINAL_NORM_RESULT
    artifacts.append(safe_file(relative, size, digest))
    final_norm = strict_json(ROOT / relative)
    require(final_norm.get("schema") == "amdgpu-sim.qwen35-final-norm-live-diff.v1", "final norm schema mismatch")
    require(final_norm.get("final_norm_runner_sha256") == file_sha256(FINAL_NORM_RUNNER), "final norm runner identity drifted")
    require(final_norm.get("output_correct") is True, "final norm failed")
    require(final_norm.get("oracle_feedback_to_target") is False, "final norm feedback enabled")
    require(final_norm.get("actual_matches_source_model_dispatch_sha256") is True, "final norm source dispatch mismatch")
    require(all(final_norm.get(key) == 0 for key in ("fallback_count", "cpu_fallback_count", "nvidia_fallback_count")), "final norm fallback nonzero")
    require(final_norm.get("source_result", {}).get("sha256") == STRICT_RESULTS[-1][2], "final norm source result mismatch")
    source_checkpoint = final_norm.get("source_checkpoint", {})
    require(source_checkpoint.get("manifest_sha256") == manifest_sha and source_checkpoint.get("state_sha256") == state_sha, "final norm checkpoint binding mismatch")
    require(numeric_comparison_valid(final_norm.get("final_norm_comparison"), expected_atol=0.03125, exact_sha_required=True), "final norm comparison invalid")
    return {
        "policy_id": "qwen35-live-input-strict-prerequisite-v1",
        "layers": list(range(24)),
        "identity_sha256": STRICT_IDENTITY_SHA256,
        "request_identity_sha256": STRICT_REQUEST_SHA256,
        "run_id": STRICT_RUN_ID,
        "final_norm_exact": True,
        "execution_identity": observed_execution_identity,
        "gem5_config": {
            "path": str(current_runtime_paths()[4]),
            "sha256": file_sha256(current_runtime_paths()[4]),
        },
        "artifacts": [record.__dict__ for record in artifacts],
        "correct": True,
    }


def trajectory_metric(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        values = (
            float(record["max_abs_error"]),
            float(record["relative_l2_error"]),
            float(record["cosine_similarity"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        all(math.isfinite(value) for value in values)
        and record.get("nonfinite_count") == 0
        and record.get("atol") == 0.25
        and record.get("rtol") == 0.15
        and record.get("max_relative_l2") == TRAJECTORY_MAX_RELATIVE_L2
        and isinstance(record.get("mismatch_count"), int)
        and record.get("mismatch_count") >= 0
        and isinstance(record.get("actual_sha256"), str)
        and HEX64.fullmatch(record["actual_sha256"]) is not None
        and isinstance(record.get("expected_sha256"), str)
        and HEX64.fullmatch(record["expected_sha256"]) is not None
        and values[1] <= TRAJECTORY_MAX_RELATIVE_L2
        and values[2] >= TRAJECTORY_MIN_COSINE
    )


def validate_trajectory(result: Mapping[str, Any]) -> dict[str, Any]:
    require(result.get("schema") == "amdgpu-sim.gemsim-vllm-qwen35-decode-window.v1", "production result schema mismatch")
    require(result.get("inference_mode") == "production", "production inference mode missing")
    require(result.get("token_ids") == [248044, 266, 27841, 27841], "production token window mismatch")
    require(result.get("positions") == [0, 1, 2, 3], "production positions mismatch")
    require(result.get("execution_complete") is True, "production execution incomplete")
    require(result.get("all_caches_finite") is True and result.get("mutated_cache_count") == 24, "production cache mutation incomplete")
    require(all(result.get(key) == 0 for key in ("fallback_count", "cpu_fallback_count", "nvidia_fallback_count")), "production fallback nonzero")
    steps = result.get("step_comparisons")
    require(isinstance(steps, list) and len(steps) == 3, "production step comparison count mismatch")
    expected_steps = (
        ("empty_cache_prefill", [0, 1], None),
        ("cache_preserving_decode", [2], 27841),
        ("cache_preserving_decode", [3], 27841),
    )
    metrics: list[dict[str, Any]] = []
    for step, (kind, positions, input_token_id) in zip(steps, expected_steps):
        require(isinstance(step, dict), "production step record is malformed")
        require(
            step.get("kind") == kind and step.get("positions") == positions,
            "production step identity mismatch",
        )
        if input_token_id is None:
            require("input_token_id" not in step, "prefill step has a decode token")
        else:
            require(
                step.get("input_token_id") == input_token_id,
                "production decode token mismatch",
            )
        comparison = step.get("comparison") if isinstance(step, dict) else None
        require(trajectory_metric(comparison), "production step trajectory failed")
        metrics.append(comparison)
    for field in ("prefill_cache_comparison", "final_cache_comparison"):
        comparison = result.get(field)
        require(isinstance(comparison, dict), f"production {field} missing")
        layers = comparison.get("layers")
        require(isinstance(layers, list) and len(layers) == 24, f"production {field} layer count mismatch")
        for expected_layer, layer in enumerate(layers):
            require(isinstance(layer, dict) and layer.get("layer") == expected_layer, f"production {field} layer order mismatch")
            tensors = layer.get("tensors")
            expected_roles = (
                {"kv_cache"}
                if expected_layer % 4 == 3
                else {"conv_state", "recurrent_state"}
            )
            expected_count = len(expected_roles)
            require(isinstance(tensors, list) and len(tensors) == expected_count, f"production {field} tensor count mismatch")
            require(
                {tensor.get("name") for tensor in tensors if isinstance(tensor, dict)}
                == expected_roles,
                f"production {field} tensor roles mismatch",
            )
            for tensor in tensors:
                require(trajectory_metric(tensor), f"production {field} trajectory failed")
                metrics.append(tensor)
    return {
        "policy_id": TRAJECTORY_POLICY_ID,
        "max_relative_l2": TRAJECTORY_MAX_RELATIVE_L2,
        "min_cosine_similarity": TRAJECTORY_MIN_COSINE,
        "comparison_count": len(metrics),
        "pointwise_mismatch_comparison_count": sum(
            int(record.get("mismatch_count", 0) > 0) for record in metrics
        ),
        "pointwise_mismatch_total": sum(int(record.get("mismatch_count", 0)) for record in metrics),
        "worst_relative_l2": max(float(record["relative_l2_error"]) for record in metrics),
        "worst_cosine_similarity": min(float(record["cosine_similarity"]) for record in metrics),
        "correct": True,
    }


def parse_result(stdout: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == "amdgpu-sim.gemsim-vllm-qwen35-decode-window.v1":
            candidates.append(value)
    require(len(candidates) == 1, "production stdout does not contain exactly one result")
    return candidates[0]


def parse_ready_identity(gem5_log: Path) -> dict[str, Any]:
    text = gem5_log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"host-gpu-ready .*daemon_uuid=([0-9a-f]{32}) job_uuid=([0-9a-f]{32}) "
        r"epoch=([0-9]+) rank=([0-9]+) world=([0-9]+)",
        text,
    )
    require(len(matches) == 1, "gem5 ready identity is missing or ambiguous")
    daemon, job, epoch, rank, world = matches[0]
    require((int(rank), int(world)) == (0, 1), "gem5 rank/world mismatch")
    require("host-gpu-dispatch-exit cause=host GPU dispatch session complete code=0" in text, "gem5 did not exit cleanly")
    require("fatal:" not in text and "panic:" not in text, "gem5 log contains fatal or panic")
    return {"daemon_uuid": daemon, "job_uuid": job, "epoch": int(epoch), "rank": 0, "world_size": 1}


def validate_trace(trace_path: Path, gem5_log: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise AcceptanceError(f"invalid trace line {line_number}: {error}") from error
            require(isinstance(value, dict) and value.get("schema") == TRACE_SCHEMA, "trace schema mismatch")
            records.append(value)
    counts = Counter(record.get("event") for record in records)
    require(counts == Counter({"generic_execution_retired": EXPECTED_DISPATCHES, "generic_execution_type20_durable": EXPECTED_DISPATCHES, "generic_execution_reuse_complete": EXPECTED_DISPATCHES - 1, "generic_execution_session_complete": 1}), "trace event counts mismatch")
    retired = [record for record in records if record.get("event") == "generic_execution_retired"]
    durable = [record for record in records if record.get("event") == "generic_execution_type20_durable"]
    retired_ids = [record.get("request_id") for record in retired]
    durable_ids = [record.get("request_id") for record in durable]
    require(len(set(retired_ids)) == EXPECTED_DISPATCHES and set(retired_ids) == set(durable_ids), "trace request identity mismatch")
    semantic_fields = (
        "kernel",
        "image_sha256",
        "grid",
        "workgroup",
        "fixed_shared_memory_bytes",
        "dynamic_shared_memory_bytes",
        "total_shared_memory_bytes",
        "kernarg_size",
        "allocation_count",
        "packet_fetches",
        "command_processor_submissions",
        "gpu_dispatcher_starts",
        "waves_started",
        "instructions_started",
        "instruction_wave_count",
        "scalar_reads",
        "global_reads",
        "global_writes",
        "store_events",
        "store_dwords",
        "workgroups_completed",
        "signal_before",
        "signal_after",
        "admission_tick",
        "start_tick",
        "end_tick",
        "retire_tick",
    )
    retired_by_request = {record["request_id"]: record for record in retired}
    durable_by_request = {record["request_id"]: record for record in durable}
    for request_id in retired_ids:
        require(
            all(
                retired_by_request[request_id].get(field)
                == durable_by_request[request_id].get(field)
                for field in semantic_fields
            ),
            "retired/type20 trace semantics differ",
        )
    reuse = [
        record
        for record in records
        if record.get("event") == "generic_execution_reuse_complete"
    ]
    reuse_ids = [record.get("request_id") for record in reuse]
    require(
        reuse_ids == retired_ids[:-1],
        "reuse trace does not correspond to all but the final dispatch",
    )
    for record in records:
        require(
            record.get("owner_generation") == records[0].get("owner_generation")
            and record.get("trace_id") == records[0].get("trace_id"),
            "trace mixes execution sessions",
        )
    for record in retired + durable:
        require(record.get("kernel_executed") is True, "trace contains unexecuted kernel")
        require(record.get("native_queue_retired") is True, "trace queue did not retire")
        require(record.get("application_pins_released") is True, "trace pins not released")
        require(record.get("adapter_released") is True, "trace adapter not released")
        require(record.get("signal_after") == 0, "trace signal did not complete")
    terminal = next(record for record in records if record.get("event") == "generic_execution_session_complete")
    require(all(terminal.get(key) is True for key in ("type20_durable", "unmap_durable", "owner_disconnected", "cleanup_complete", "kernel_executed", "native_queue_retired", "application_pins_released", "adapter_released")), "terminal trace lifecycle incomplete")
    require(terminal.get("owner_quarantined") is False and terminal.get("signal_after") == 0, "terminal trace quarantined or signaled")
    identity = parse_ready_identity(gem5_log)
    return {
        "record_count": len(records),
        "event_counts": dict(sorted(counts.items())),
        "kernel_retired_counts": dict(sorted(Counter(record.get("kernel") for record in retired).items())),
        "terminal": {key: terminal.get(key) for key in ("type20_durable", "unmap_durable", "owner_disconnected", "owner_quarantined", "cleanup_complete", "kernel_executed", "signal_after", "native_queue_retired", "application_pins_released", "adapter_released")},
        "runtime_identity": identity,
        "trace_sha256": file_sha256(trace_path),
        "gem5_log_sha256": file_sha256(gem5_log),
        "correct": True,
    }


def option_value(argv: Sequence[str], name: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == name]
    require(len(positions) == 1, f"gem5 child option is missing or repeated: {name}")
    index = positions[0]
    require(index + 1 < len(argv), f"gem5 child option has no value: {name}")
    return argv[index + 1]


def parse_gem5_child(
    pid: int,
    argv: Sequence[str],
    expected_executable: Path,
    expected_config: Path,
    *,
    executable_sha256: str | None = None,
) -> Gem5Child:
    require(bool(argv), "gem5 child command is empty")
    executable = Path(argv[0]).resolve(strict=True)
    require(executable == expected_executable, "gem5 child executable mismatch")
    config_positions: list[Path] = []
    for value in argv[1:]:
        try:
            candidate = Path(value).resolve(strict=True)
        except OSError:
            continue
        if candidate == expected_config:
            config_positions.append(candidate)
    require(len(config_positions) == 1, "gem5 child config is missing or repeated")
    trace_path = Path(option_value(argv, "--dispatch-trace-path")).resolve()
    outdir = Path(option_value(argv, "--outdir")).resolve()
    endpoint = Path(option_value(argv, "--endpoint")).resolve()
    run_dir = trace_path.parent
    require(trace_path.name == "dispatch-trace.jsonl", "gem5 trace filename mismatch")
    require(outdir == run_dir / "m5out", "gem5 output directory mismatch")
    require(endpoint.parent == run_dir, "gem5 endpoint directory mismatch")
    require(run_dir.parent == Path("/tmp"), "gem5 run directory is outside /tmp")
    require(
        run_dir.name.startswith(f"self-amdgpu-opencl-run.{os.getuid()}."),
        "gem5 run directory namespace mismatch",
    )
    job_uuid = option_value(argv, "--job-uuid")
    require(re.fullmatch(r"[0-9a-f]{32}", job_uuid) is not None, "gem5 job UUID malformed")
    try:
        epoch = int(option_value(argv, "--epoch"))
        rank = int(option_value(argv, "--rank"))
        world_size = int(option_value(argv, "--world-size"))
    except ValueError as error:
        raise AcceptanceError("gem5 child topology is not numeric") from error
    require(epoch > 0 and (rank, world_size) == (0, 1), "gem5 child topology mismatch")
    return Gem5Child(
        pid=pid,
        argv=tuple(argv),
        executable=str(executable),
        executable_sha256=(
            executable_sha256
            if executable_sha256 is not None
            else file_sha256(expected_executable)
        ),
        config=str(expected_config),
        config_sha256=file_sha256(expected_config),
        run_dir=str(run_dir),
        trace_path=str(trace_path),
        outdir=str(outdir),
        endpoint=str(endpoint),
        job_uuid=job_uuid,
        epoch=epoch,
        rank=rank,
        world_size=world_size,
    )


def process_descendants(root_pid: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(encoding="ascii")
            parent_line = next(line for line in status.splitlines() if line.startswith("PPid:"))
            parent = int(parent_line.split()[1])
        except (OSError, StopIteration, ValueError):
            continue
        children.setdefault(parent, []).append(int(entry.name))
    descendants: set[int] = set()
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        for child in children.get(parent, ()):
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    return descendants


def wait_for_gem5_child(
    process: subprocess.Popen[str],
    expected_executable: Path,
    expected_config: Path,
    timeout_seconds: float = 180.0,
) -> Gem5Child:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for pid in sorted(process_descendants(process.pid)):
            try:
                raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
            except OSError:
                continue
            argv = [part.decode("utf-8") for part in raw.split(b"\0") if part]
            if not argv:
                continue
            try:
                executable = Path(argv[0]).resolve(strict=True)
            except OSError:
                continue
            if executable == expected_executable:
                try:
                    executable_sha256 = file_sha256(
                        Path("/proc") / str(pid) / "exe"
                    )
                except OSError:
                    continue
                return parse_gem5_child(
                    pid,
                    argv,
                    expected_executable,
                    expected_config,
                    executable_sha256=executable_sha256,
                )
        require(process.poll() is None, "production runner exited before gem5 child was identified")
        time.sleep(0.05)
    raise AcceptanceError("timed out identifying the production gem5 child")


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def command_line(output_dir: Path) -> list[str]:
    return [
        "/usr/bin/python3",
        str(RUNNER),
        "--inference-mode",
        "production",
        "--decode-window-golden-dir",
        str(GOLDEN),
    ]


def validate_golden() -> list[FileRecord]:
    records = []
    for directory, table in ((GOLDEN, GOLDEN_FILES), (PREFILL_GOLDEN, PREFILL_GOLDEN_FILES)):
        require(directory.resolve(strict=True).is_relative_to(ROOT), "golden directory escapes repository")
        for filename, (size, digest) in table.items():
            records.append(safe_file(str((directory / filename).relative_to(ROOT)), size, digest))
    return records


def validate_postflight(
    strict_before: Mapping[str, Any],
    golden_before: Sequence[FileRecord],
    validator_sha256_before: str,
) -> dict[str, Any]:
    strict_after = validate_strict_prerequisite()
    golden_after = validate_golden()
    require(strict_after == strict_before, "strict prerequisite drifted during production")
    require(golden_after == list(golden_before), "golden artifacts drifted during production")
    require(
        file_sha256(THIS_FILE) == validator_sha256_before,
        "acceptance validator drifted during production",
    )
    return {
        "strict_prerequisite_unchanged": True,
        "golden_artifacts_unchanged": True,
        "execution_identity_unchanged": True,
        "validator_sha256": validator_sha256_before,
        "validator_unchanged": True,
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("ascii") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-prerequisite-only", action="store_true")
    args = parser.parse_args()
    strict = validate_strict_prerequisite()
    golden_records = validate_golden()
    validator_sha256 = file_sha256(THIS_FILE)
    if args.validate_prerequisite_only:
        print(json.dumps({"strict_prerequisite": strict, "golden_artifacts": [record.__dict__ for record in golden_records], "correct": True}, sort_keys=True))
        return 0

    output_dir = args.output_dir.expanduser()
    require(output_dir.is_absolute(), "output directory must be absolute")
    require(not output_dir.exists(), "output directory already exists")
    parent = output_dir.parent.resolve(strict=True)
    require(output_dir.parent == parent and parent.is_relative_to(ROOT), "output directory must have a normalized repository parent")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=parent))
    command = command_line(output_dir)
    process: subprocess.Popen[str] | None = None
    try:
        runtime_paths = current_runtime_paths()
        expected_gem5 = runtime_paths[3]
        expected_config = runtime_paths[4]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        child = wait_for_gem5_child(process, expected_gem5, expected_config)
        require(
            child.executable_sha256
            == strict["execution_identity"]["gem5_binary_sha256"],
            "running gem5 executable identity mismatch",
        )
        require(
            child.config_sha256 == strict["gem5_config"]["sha256"],
            "running gem5 config identity mismatch",
        )
        stdout, stderr = process.communicate()
        completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        (temporary / "runner.stdout").write_text(completed.stdout)
        (temporary / "runner.stderr").write_text(completed.stderr)
        result = parse_result(completed.stdout)
        require(
            completed.returncode in (0, 1),
            f"production runner terminated abnormally: {completed.returncode}",
        )
        require(
            (completed.returncode == 0) == (result.get("output_correct") is True),
            "production runner exit/result relationship mismatch",
        )
        trajectory = validate_trajectory(result)
        run_dir = Path(child.run_dir)
        trace_path = run_dir / "dispatch-trace.jsonl"
        gem5_log = run_dir / "gem5.log"
        require(trace_path.is_file() and gem5_log.is_file(), "gem5 trace or log missing")
        trace = validate_trace(trace_path, gem5_log)
        runtime_identity = trace["runtime_identity"]
        require(
            runtime_identity["job_uuid"] == child.job_uuid
            and runtime_identity["epoch"] == child.epoch
            and runtime_identity["rank"] == child.rank
            and runtime_identity["world_size"] == child.world_size,
            "gem5 command and ready identities differ",
        )
        shutil.copy2(trace_path, temporary / "dispatch-trace.jsonl")
        shutil.copy2(gem5_log, temporary / "gem5.log")
        stats = run_dir / "m5out/stats.txt"
        require(stats.is_file(), "gem5 stats are missing")
        shutil.copy2(stats, temporary / "stats.txt")
        postflight = validate_postflight(
            strict, golden_records, validator_sha256
        )
        payload = {
            "schema": "amdgpu-sim.qwen35-production-backbone-acceptance.v1",
            "command": command,
            "runner_exit_code": completed.returncode,
            "runner_pointwise_output_correct": result.get("output_correct"),
            "strict_live_input_prerequisite": strict,
            "golden_artifacts": [record.__dict__ for record in golden_records],
            "continuous_trajectory": trajectory,
            "trace": trace,
            "gem5_child": child.__dict__,
            "postflight_identity_validation": postflight,
            "fallback_count": result.get("fallback_count"),
            "cpu_fallback_count": result.get("cpu_fallback_count"),
            "nvidia_fallback_count": result.get("nvidia_fallback_count"),
            "target_feedback": False,
            "oracle_process_started_during_production": False,
            "checkpoint_restore_used": False,
            "checkpoint_publish_used": False,
            "acceptance_scope": "continuous_teacher_forced_backbone_final_norm_and_mutable_state",
            "tied_lm_head_executed": False,
            "logits_compared": False,
            "greedy_verified": False,
            "sampling_executed": False,
            "output_correct": bool(strict["correct"] and trajectory["correct"] and trace["correct"]),
            "claim_boundary": "continuous two-token empty-cache prefill plus two cache-preserving teacher-forced decode tokens through all 24 decoder layers, final norm, and mutable state; no checkpoint/oracle injection, logits, sampling, greedy generation, CCL, or TP",
        }
        atomic_write_json(temporary / "result.json", payload)
        for file in temporary.iterdir():
            if file.is_file():
                with file.open("rb") as stream:
                    os.fsync(stream.fileno())
        descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        rename_noreplace(temporary, output_dir)
        temporary = None
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["output_correct"] else 1
    finally:
        if process is not None:
            terminate_process_group(process)
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
