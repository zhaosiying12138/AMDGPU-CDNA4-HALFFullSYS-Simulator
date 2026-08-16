#!/usr/bin/env python3
"""Compare the live post-layer-23 Qwen3.5 final norm with NVIDIA.

This diagnostic is deliberately separate from the model runner whose exact
source hash is embedded in the checkpoint.  NVIDIA output is comparison-only;
it is never copied into target state or used as fallback data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import uuid


HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE / "_gemsim_bootstrap.py"))["bootstrap"](
    __file__, "qwen35-final-norm-live-diff"
)
# Python isolated mode omits the script directory. Trust only this resolved,
# repository-owned directory for sibling helpers after the clean bootstrap.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch
from safetensors import safe_open

import gemsim_vllm
from _qwen35_layer_checkpoint import load_layer_checkpoint
from _qwen35_layer_oracle_protocol import (
    current_layer_oracle_request_identity,
    expected_local_oracle_identity,
    publish_final_norm_oracle_request,
    run_layer_oracle,
)


ROOT = HERE.parents[1]
MAIN_RUNNER = HERE / "qwen35_vllm_model_forward.py"
EXPECTED_MAIN_RUNNER_SHA256 = (
    "878acfa8d37a81a204d2aff1844d7618fa571e274a05a8ba47f5f31312649343"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "artifacts/qwen35-layer-diff/layer2023-resume-v1/"
    "checkpoint-after-layer-23"
)
DEFAULT_SOURCE_RESULT = (
    ROOT / "artifacts/qwen35-layer-diff/layer2023-resume-v1/result.json"
)
MODEL_SHARD = ROOT / "models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
FINAL_NORM_WEIGHT_KEY = "model.language_model.norm.weight"
FINAL_NORM_WEIGHT_SHA256 = (
    "30832e6642a9c1555485d5c9553a35e539278c33ce3b7a9665e06cd2c5598381"
)
EPSILON = 1.0e-6
ATOL = 0.03125
RTOL = 0.03
MAX_RELATIVE_L2 = 0.03
MIN_COSINE = 0.98
PLUGIN_SOURCE_PATHS = tuple(
    ROOT / relative
    for relative in (
        "plugins/framework/gemsim_vllm/src/gemsim_vllm/adapters.py",
        "plugins/framework/gemsim_vllm/src/gemsim_vllm/attention.py",
        "plugins/framework/gemsim_vllm/src/gemsim_vllm/kernels.py",
        "plugins/framework/gemsim_vllm/src/gemsim_vllm/model.py",
        "plugins/framework/gemsim_vllm/src/gemsim_vllm/ops.py",
    )
)
FORMULA_SOURCE_PATHS = tuple(
    ROOT / relative
    for relative in (
        "projects/vllm/vllm/model_executor/layers/layernorm.py",
        "projects/vllm/vllm/model_executor/models/qwen3_5.py",
        "projects/vllm/vllm/model_executor/models/qwen3_next.py",
        (
            "projects/vllm/vllm/model_executor/layers/mamba/gdn/"
            "qwen_gdn_linear_attn.py"
        ),
    )
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def source_set_identity(paths: tuple[Path, ...]) -> dict[str, str]:
    records = {}
    for path in paths:
        resolved = path.resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file():
            raise RuntimeError(f"identity source is unsafe: {path}")
        records[str(resolved.relative_to(ROOT))] = file_sha256(resolved)
    return records


def git_object_id(repository: Path, revision: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", revision],
        text=True,
    ).strip()


def validate_live_checkpoint_identity(checkpoint) -> dict[str, object]:
    identity = checkpoint.manifest["identity"]
    implementation = identity["implementation"]
    model = identity["model"]
    if (
        model["id"] != "Qwen/Qwen3.5-0.8B"
        or model["revision"] != "2fc06364715b967f1860aea9cf38778875588b17"
    ):
        raise RuntimeError("checkpoint model identity mismatch")
    for filename, expected in model["artifacts"].items():
        artifact = ROOT / "models/Qwen3.5-0.8B" / filename
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or artifact.stat().st_size != expected["bytes"]
            or file_sha256(artifact) != expected["sha256"]
        ):
            raise RuntimeError(f"checkpoint model artifact drift: {filename}")
    if identity["weights"] != {
        "checkpoint_tensor_count": 488,
        "source_tensor_count": 320,
        "loaded_tensor_count": 248,
        "source_names_sha256": (
            "01b20f593de553feff7fdcc0ecd487bdd498a5787fcc9addfcb83c1b0bcd7b04"
        ),
    }:
        raise RuntimeError("checkpoint loaded-weight identity mismatch")
    observed_runner = file_sha256(MAIN_RUNNER)
    if observed_runner != EXPECTED_MAIN_RUNNER_SHA256:
        raise RuntimeError("main runner changed since the accepted layer23 execution")
    if implementation["runner_sha256"] != observed_runner:
        raise RuntimeError("checkpoint main-runner identity mismatch")

    prefix = Path(os.environ["ROCM_SIM_ROOT"]).resolve(strict=True)
    prefix_manifest = prefix / "manifest.json"
    manifest = json.loads(prefix_manifest.read_text())
    runtime_path = Path(manifest["artifacts"]["runtime_library"]["path"]).resolve(
        strict=True
    )
    gem5_path = Path(manifest["managed_inputs"]["gem5_binary"]["path"]).resolve(
        strict=True
    )
    vllm = ROOT / "projects/vllm"
    vllm_head = git_object_id(vllm, "HEAD")
    vllm_tree = git_object_id(vllm, "HEAD^{tree}")
    live = {
        "architecture": implementation["architecture"],
        "runner_sha256": observed_runner,
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
        "prefix_manifest_sha256": file_sha256(prefix_manifest),
    }
    if implementation != live:
        mismatches = {
            key: {"checkpoint": implementation.get(key), "live": value}
            for key, value in live.items()
            if implementation.get(key) != value
        }
        raise RuntimeError(f"checkpoint live implementation drift: {mismatches}")
    if identity["target"] != {
        "backend": "gemsim_amd",
        "arch": "gfx950",
        "device": "cpu",
        "fallback_allowed": False,
        "stochastic_ops": False,
    }:
        raise RuntimeError("checkpoint target/fallback identity mismatch")
    if identity["parallelism"] != {
        "world_size": 1,
        "rank": 0,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }:
        raise RuntimeError("checkpoint parallelism identity mismatch")
    return {
        "checkpoint_identity_sha256": checkpoint.manifest["identity_sha256"],
        "implementation": live,
        "target": identity["target"],
        "parallelism": identity["parallelism"],
        "model": model,
        "weights": identity["weights"],
    }


def validate_source_result(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError("source result must be a regular non-symlink file")
    observed_hash = file_sha256(resolved)
    payload = json.loads(resolved.read_text())
    if payload.get("execution_success") is not True or payload.get("output_correct") is not True:
        raise RuntimeError("source result execution contract mismatch")
    if payload.get("fallback_count") != 0 or payload.get("cpu_fallback_count") != 0 or payload.get("nvidia_fallback_count") != 0:
        raise RuntimeError("source result fallback contract mismatch")
    layer_records = payload.get("layer_records")
    if not isinstance(layer_records, list):
        raise RuntimeError("source result lacks layer records")
    layer23 = next((record for record in layer_records if record.get("layer") == 23), None)
    if not isinstance(layer23, dict) or layer23.get("comparison", {}).get("correct") is not True:
        raise RuntimeError("source result lacks a correct layer23 comparison")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("after_layer") != 23:
        checkpoint = next(
            (record for record in reversed(payload.get("checkpoints", [])) if record.get("after_layer") == 23),
            None,
        )
    if not isinstance(checkpoint, dict) or not checkpoint.get("manifest_sha256") or not checkpoint.get("state_sha256"):
        raise RuntimeError("source result lacks the layer23 checkpoint binding")
    final_hidden = payload.get("final_hidden_sha256")
    if (
        payload.get("final_norm_executed") is not True
        or not isinstance(final_hidden, str)
        or len(final_hidden) != 64
    ):
        raise RuntimeError("source result lacks a complete final norm output")
    return {
        "path": str(resolved),
        "sha256": observed_hash,
        "final_hidden_sha256": final_hidden,
        "layer23_hidden_sha256": layer23["comparison"]["hidden"]["actual_sha256"],
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "checkpoint_state_sha256": checkpoint["state_sha256"],
        "source_schema": payload.get("schema"),
    }


def load_final_norm_weight(checkpoint) -> tuple[torch.Tensor, dict[str, object]]:
    model_record = checkpoint.manifest["identity"]["model"]["artifacts"]
    shard_record = model_record[MODEL_SHARD.name]
    if (
        MODEL_SHARD.stat().st_size != shard_record["bytes"]
        or file_sha256(MODEL_SHARD) != shard_record["sha256"]
    ):
        raise RuntimeError("checkpoint-bound model shard identity mismatch")
    with safe_open(MODEL_SHARD, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor(FINAL_NORM_WEIGHT_KEY).clone().contiguous()
    if weight.dtype != torch.bfloat16 or weight.shape != (1024,):
        raise RuntimeError("final norm raw weight must be BF16 [1024]")
    observed = tensor_sha256(weight)
    if observed != FINAL_NORM_WEIGHT_SHA256:
        raise RuntimeError("final norm raw weight identity mismatch")
    return weight, {
        "key": FINAL_NORM_WEIGHT_KEY,
        "dtype": str(weight.dtype),
        "shape": list(weight.shape),
        "sha256": observed,
        "epsilon": EPSILON,
        "semantics": "Gemma raw weight; kernel applies 1 + weight",
    }


def compare_final_hidden(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, object]:
    if (
        actual.dtype != torch.bfloat16
        or expected.dtype != torch.bfloat16
        or actual.shape != (2, 1024)
        or expected.shape != actual.shape
    ):
        raise RuntimeError("final norm comparison requires matching BF16 [2,1024]")
    actual_float = actual.float()
    expected_float = expected.float()
    error = torch.abs(actual_float - expected_float)
    finite = (
        torch.isfinite(actual_float)
        & torch.isfinite(expected_float)
        & torch.isfinite(error)
    )
    mismatch = (~finite) | (error > ATOL + RTOL * torch.abs(expected_float))
    actual_norm = float(torch.linalg.vector_norm(actual_float).item())
    expected_norm = float(torch.linalg.vector_norm(expected_float).item())
    relative_l2 = float(torch.linalg.vector_norm(error).item()) / max(
        expected_norm, 1.0e-30
    )
    cosine = float(
        torch.sum(actual_float * expected_float).item()
        / max(actual_norm * expected_norm, 1.0e-30)
    )
    mismatch_count = int(torch.count_nonzero(mismatch).item())
    nonfinite_count = int(torch.count_nonzero(~finite).item())
    return {
        "actual_sha256": tensor_sha256(actual),
        "expected_sha256": tensor_sha256(expected),
        "mismatch_count": mismatch_count,
        "nonfinite_count": nonfinite_count,
        "max_abs_error": float(torch.where(finite, error, 0.0).max().item()),
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
        "atol": ATOL,
        "rtol": RTOL,
        "max_relative_l2": MAX_RELATIVE_L2,
        "min_cosine_similarity": MIN_COSINE,
        "correct": bool(
            mismatch_count == 0
            and nonfinite_count == 0
            and relative_l2 <= MAX_RELATIVE_L2
            and cosine >= MIN_COSINE
        ),
    }


def execution_record(value) -> dict[str, object]:
    if value is None:
        raise RuntimeError("oracle response lacks execution evidence")
    return {
        "argv": list(value.argv),
        "environment": value.environment,
        "exit_code": value.exit_code,
        "launcher_identity": value.launcher_identity,
        "stdout": value.stdout,
        "stderr": value.stderr,
        "stdout_sha256": hashlib.sha256(value.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(value.stderr.encode()).hexdigest(),
    }


def create_output(path: Path) -> Path:
    requested = path.expanduser()
    parent = requested.parent.resolve(strict=True)
    destination = parent / requested.name
    if requested != destination or requested.name in ("", ".", ".."):
        raise RuntimeError("output directory must be normalized and symlink-free")
    os.mkdir(destination, 0o700)
    return destination


def publish_result(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o400)
        os.link(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--source-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    value.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    target = __import__("triton").runtime.driver.active.get_current_target()
    if (target.backend, target.arch) != ("gemsim_amd", "gfx950"):
        raise RuntimeError(f"unexpected Triton target: {target}")

    checkpoint_path = args.source_checkpoint.resolve(strict=True)
    checkpoint = load_layer_checkpoint(checkpoint_path)
    boundary = checkpoint.manifest["boundary"]
    if not (
        checkpoint.after_layer == 23
        and checkpoint.next_layer == 24
        and boundary["resume_action"] == "final_norm"
        and boundary["residual_present"] is True
    ):
        raise RuntimeError("source checkpoint is not an after-layer23 final_norm boundary")
    live_identity = validate_live_checkpoint_identity(checkpoint)
    source_result = validate_source_result(args.source_result)
    expected_state_sha = checkpoint.manifest["artifacts"]["state"]["sha256"]
    if (
        source_result["checkpoint_manifest_sha256"] != checkpoint.manifest_sha256
        or source_result["checkpoint_state_sha256"] != expected_state_sha
    ):
        raise RuntimeError("source result is not bound to the supplied layer23 checkpoint")
    weight, weight_record = load_final_norm_weight(checkpoint)
    hidden = checkpoint.tensors[boundary["hidden_states"]["key"]].clone().contiguous()
    residual = checkpoint.tensors[boundary["residual"]["key"]].clone().contiguous()

    output = create_output(args.output_dir)
    oracle_identity = current_layer_oracle_request_identity(MAIN_RUNNER)
    expected_oracle = expected_local_oracle_identity()
    request = publish_final_norm_oracle_request(
        output / "final-norm-request",
        identity=oracle_identity,
        source_checkpoint=checkpoint_path,
    )
    response = run_layer_oracle(
        request=request,
        response_dir=output / "final-norm-response",
        expected_oracle_identity=expected_oracle,
    )
    if response.final_hidden_after is None:
        raise RuntimeError("final norm oracle response lacks output")

    gemsim_vllm.register_ops()
    actual, residual_out = torch.ops.gemsim.fused_add_gemma_rms_norm(
        hidden,
        residual,
        weight,
        EPSILON,
    )
    if not torch.equal(residual_out, (hidden.float() + residual.float()).to(torch.bfloat16)):
        raise RuntimeError("formal fused final norm residual output mismatch")
    comparison = compare_final_hidden(actual, response.final_hidden_after)
    source_dispatch_same = tensor_sha256(actual) == source_result["final_hidden_sha256"]
    correct = bool(comparison["correct"] and source_dispatch_same)
    payload = {
        "schema": "amdgpu-sim.qwen35-final-norm-live-diff.v1",
        "kind": "checkpoint_bound_diagnostic_final_norm_cross_architecture_comparison",
        "final_norm_runner_sha256": file_sha256(Path(__file__).resolve()),
        "diagnostic_only": True,
        "acceptance_eligible": False,
        "backend": target.backend,
        "arch": target.arch,
        "source_checkpoint": {
            "path": str(checkpoint.path),
            "manifest_sha256": checkpoint.manifest_sha256,
            "state_sha256": checkpoint.manifest["artifacts"]["state"]["sha256"],
            "after_layer": checkpoint.after_layer,
            "next_layer": checkpoint.next_layer,
            "resume_action": boundary["resume_action"],
        },
        "live_identity": live_identity,
        "source_result": source_result,
        "final_norm_weight": weight_record,
        "oracle_request": {
            "path": str(request.path),
            "request_id": request.request_id,
            "package_sha256": request.package_sha256,
            "identity_sha256": request.identity_sha256,
            "payload_sha256": request.payload_sha256,
        },
        "oracle_response": {
            "path": str(response.path),
            "request_id": response.request_id,
            "package_sha256": response.package_sha256,
            "identity_sha256": response.identity_sha256,
            "payload_sha256": response.payload_sha256,
            "target_feedback": False,
        },
        "oracle_execution": execution_record(response.execution_record),
        "final_norm_comparison": comparison,
        "actual_matches_source_model_dispatch_sha256": source_dispatch_same,
        "actual_final_hidden_sha256": tensor_sha256(actual),
        "source_final_hidden_sha256": source_result["final_hidden_sha256"],
        "oracle_feedback_to_target": False,
        "fallback_count": 0,
        "cpu_fallback_count": 0,
        "nvidia_fallback_count": 0,
        "output_correct": correct,
        "claim_boundary": (
            "diagnostic replay of only the formal fused final-norm operator from "
            "the exact accepted after-layer23 checkpoint; it is not a continuous "
            "empty-cache production acceptance run"
        ),
    }
    publish_result(output / "result.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
