#!/usr/bin/env python3
"""Validate the source-only Qwen3.5 operator queue and generic results."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "amdgpu-sim.qwen35.operator-manifest.v2"
RESULT_SCHEMA = "amdgpu-sim.qwen35.operator-result.v2"
VALID_DTYPES = {"bfloat16", "float32", "int32", "int64", "uint32", "bool"}
VALID_PHASES = {"input", "decode", "prefill", "prefill_decode", "output"}
VALID_ROLES = {"required", "alternate", "deferred"}
VALID_CONFIGURATIONS = {"configured", "unconfigured"}
VALID_SOURCE_STATUSES = {"unconfigured", "ready"}
REQUIRED_RUN_KINDS = ("fresh", "repeat")
REQUIRED_LIFECYCLE_STAGES = (
    "compile",
    "load",
    "launch",
    "wait",
    "d2h",
    "oracle",
    "trace",
    "cleanup",
)
REQUIRED_ARTIFACT_ROLES = ("stdout", "stderr", "code_object", "trace")
REQUIRED_FALLBACK_COUNTERS = (
    "fallback_count",
    "cpu_fallback_count",
    "nvidia_fallback_count",
)
RUNTIME_PROVENANCE_FIELDS = {
    "gem5_tree",
    "runtime_commit",
    "triton_commit",
    "prefix_manifest_sha256",
    "setup_sha256",
    "runner_sha256",
    "code_object_sha256",
    "environment_sha256",
}


class WorkQueueValidationError(ValueError):
    """Raised when an executable work-queue artifact is malformed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def runtime_evidence_policy() -> dict[str, Any]:
    """Return the immutable, operator-independent result policy."""
    return {
        "result_schema": RESULT_SCHEMA,
        "results_external": True,
        "required_run_kinds": list(REQUIRED_RUN_KINDS),
        "required_lifecycle_stages": list(REQUIRED_LIFECYCLE_STAGES),
        "required_artifact_roles": list(REQUIRED_ARTIFACT_ROLES),
    }


def work_item_spec_payload(item: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable item payload that every result must bind."""
    payload = copy.deepcopy(item)
    payload.pop("spec_sha256", None)
    payload.pop("status", None)
    return payload


def work_item_spec_sha256(item: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(work_item_spec_payload(item))).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_oid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _safe_file_under(base: Path, relative: str) -> Path | None:
    if not _safe_relative_path(relative):
        return None
    base_resolved = base.resolve()
    current = base_resolved
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            return None
    try:
        resolved = (base_resolved / relative).resolve(strict=True)
    except OSError:
        return None
    if resolved != base_resolved and base_resolved not in resolved.parents:
        return None
    return resolved if resolved.is_file() else None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _validate_tensor(item_id: str, tensor: dict[str, Any], errors: list[str]) -> None:
    required = {
        "name",
        "role",
        "dtype",
        "shape",
        "strides",
        "stride_unit",
        "storage_offset",
        "storage_elements",
        "access",
        "alias_group",
    }
    missing = sorted(required - tensor.keys())
    if missing:
        errors.append(f"{item_id}: tensor missing fields: {','.join(missing)}")
        return
    shape = tensor["shape"]
    strides = tensor["strides"]
    if tensor["dtype"] not in VALID_DTYPES:
        errors.append(f"{item_id}: unsupported tensor dtype {tensor['dtype']!r}")
    if (
        not isinstance(shape, list)
        or not shape
        or any(not isinstance(value, int) or value <= 0 for value in shape)
    ):
        errors.append(f"{item_id}: tensor shape must be concrete positive integers")
    if (
        not isinstance(strides, list)
        or not isinstance(shape, list)
        or len(strides) != len(shape)
        or any(not isinstance(value, int) or value <= 0 for value in strides)
    ):
        errors.append(f"{item_id}: tensor strides must match rank and be positive")
    if tensor["stride_unit"] != "elements":
        errors.append(f"{item_id}: tensor stride unit must be elements")
    if not isinstance(tensor["storage_offset"], int) or tensor["storage_offset"] < 0:
        errors.append(f"{item_id}: tensor storage offset is invalid")
    if not isinstance(tensor["storage_elements"], int) or tensor["storage_elements"] <= 0:
        errors.append(f"{item_id}: tensor storage size is invalid")
    elif (
        isinstance(shape, list)
        and isinstance(strides, list)
        and len(shape) == len(strides)
        and all(isinstance(value, int) and value > 0 for value in shape + strides)
    ):
        required_storage = tensor["storage_offset"] + 1 + sum(
            (extent - 1) * stride for extent, stride in zip(shape, strides)
        )
        if tensor["storage_elements"] < required_storage:
            errors.append(f"{item_id}: tensor storage does not cover shape and strides")
    if tensor["role"] not in {"input", "output", "state", "parameter"}:
        errors.append(f"{item_id}: tensor role is invalid")
    if tensor["access"] not in {"read_only", "write_only", "read_write"}:
        errors.append(f"{item_id}: tensor access is invalid")
    if not isinstance(tensor["alias_group"], str) or not tensor["alias_group"]:
        errors.append(f"{item_id}: tensor alias group is invalid")


def _validate_configured_item(
    manifest: dict[str, Any], item: dict[str, Any], errors: list[str]
) -> None:
    item_id = item["id"]
    tensors = item.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        errors.append(f"{item_id}: configured item has no tensors")
        return
    tensor_names: set[str] = set()
    for tensor in tensors:
        if not isinstance(tensor, dict):
            errors.append(f"{item_id}: tensor entry is not an object")
            continue
        _validate_tensor(item_id, tensor, errors)
        name = tensor.get("name")
        if name in tensor_names:
            errors.append(f"{item_id}: duplicate tensor name {name!r}")
        if isinstance(name, str):
            tensor_names.add(name)

    mutation = item.get("mutation", {})
    for key in ("allowed", "must_remain_bitwise_equal", "must_be_disjoint"):
        if not isinstance(mutation.get(key), list):
            errors.append(f"{item_id}: mutation.{key} must be a list")
    referenced = set(mutation.get("must_remain_bitwise_equal", []))
    referenced.update(mutation.get("must_be_disjoint", []))
    for entry in mutation.get("allowed", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("tensor"), str):
            errors.append(f"{item_id}: mutation allowed entry is invalid")
        else:
            referenced.add(entry["tensor"])
    if referenced - tensor_names:
        errors.append(f"{item_id}: mutation references unknown tensors")

    state = item.get("state", {})
    transitions = state.get("transition_count")
    if state.get("kind") not in {"stateless", "persistent"}:
        errors.append(f"{item_id}: configured state kind is invalid")
    if not isinstance(transitions, int) or transitions < 0:
        errors.append(f"{item_id}: state transition count is invalid")
    elif state.get("kind") == "persistent" and transitions < 2:
        errors.append(f"{item_id}: persistent state needs at least two transitions")
    if not isinstance(state.get("slots"), list):
        errors.append(f"{item_id}: state slots must be a list")

    oracle = item.get("oracle", {})
    if not isinstance(oracle.get("reference"), str) or not oracle["reference"]:
        errors.append(f"{item_id}: oracle reference is missing")
    if oracle.get("accumulation_dtype") not in VALID_DTYPES:
        errors.append(f"{item_id}: oracle accumulation dtype is invalid")
    comparisons = oracle.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        errors.append(f"{item_id}: oracle comparisons are missing")
    else:
        for comparison in comparisons:
            if not isinstance(comparison, dict):
                errors.append(f"{item_id}: oracle comparison is not an object")
                continue
            if comparison.get("mode") not in {"exact", "atol_rtol"}:
                errors.append(f"{item_id}: oracle comparison mode is invalid")
            for key in ("atol", "rtol"):
                value = comparison.get(key)
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(f"{item_id}: oracle {key} is invalid")
            if comparison.get("max_mismatches") != 0:
                errors.append(f"{item_id}: oracle must require zero mismatches")
            if comparison.get("finite_required") is not True:
                errors.append(f"{item_id}: oracle must require finite values")
            if comparison.get("equal_nan") is not False:
                errors.append(f"{item_id}: oracle must reject NaN equality")

    kernel = item.get("kernel", {})
    if not isinstance(kernel.get("entrypoint"), str) or not kernel["entrypoint"]:
        errors.append(f"{item_id}: kernel entrypoint is missing")
    symbols = kernel.get("expected_symbols")
    if not isinstance(symbols, list) or not symbols:
        errors.append(f"{item_id}: kernel symbols are missing")
    target = kernel.get("target")
    expected_target = manifest.get("acceptance_policy", {}).get("runtime_target")
    if target != expected_target:
        errors.append(f"{item_id}: kernel target differs from manifest target")
    code_objects = kernel.get("code_objects", {})
    if code_objects.get("identity_policy") != "recorded_sha256":
        errors.append(f"{item_id}: code-object identity must be recorded at runtime")
    if set(code_objects) != {"count", "identity_policy"}:
        errors.append(f"{item_id}: code-object spec contains a precomputed identity")
    if not isinstance(code_objects.get("count"), int) or code_objects["count"] <= 0:
        errors.append(f"{item_id}: configured item needs a positive code-object count")
    launch = kernel.get("launch", {})
    for key in ("grid", "workgroup"):
        value = launch.get(key)
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(not isinstance(extent, int) or extent <= 0 for extent in value)
        ):
            errors.append(f"{item_id}: kernel launch {key} is invalid")
    if not isinstance(launch.get("num_warps"), int) or launch["num_warps"] <= 0:
        errors.append(f"{item_id}: kernel launch num_warps is invalid")


def _source_path(root: Path, entrypoint: dict[str, Any]) -> Path | None:
    repositories = {
        "root": root,
        "vllm": root / "projects" / "vllm",
        "triton": root / "projects" / "triton",
        "pytorch": root / "projects" / "pytorch",
    }
    repository = repositories.get(entrypoint.get("repo"))
    if repository is None or not _safe_relative_path(entrypoint.get("path")):
        return None
    return _safe_file_under(repository, entrypoint["path"])


def _validate_source(item: dict[str, Any], root: Path, errors: list[str]) -> None:
    item_id = item["id"]
    source = item["source"]
    entrypoints = source.get("entrypoints")
    if not isinstance(entrypoints, list) or not entrypoints:
        errors.append(f"{item_id}: source entrypoints are missing")
        return
    for entrypoint in entrypoints:
        if not isinstance(entrypoint, dict):
            errors.append(f"{item_id}: source entrypoint is not an object")
            continue
        path = _source_path(root, entrypoint)
        if path is None or not _is_sha256(entrypoint.get("file_sha256")):
            errors.append(f"{item_id}: source repository, path, or SHA-256 is invalid")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != entrypoint["file_sha256"]:
            errors.append(f"{item_id}: source file hash mismatch: {entrypoint['path']}")
        if not entrypoint.get("symbol") and not entrypoint.get("registration"):
            errors.append(f"{item_id}: source entrypoint lacks symbol or registration")
    runner = source.get("runner")
    if item["configuration_status"] == "configured":
        if not isinstance(runner, dict) or not _safe_relative_path(runner.get("path")):
            errors.append(f"{item_id}: configured item runner is invalid")
        elif not _is_sha256(runner.get("sha256")):
            errors.append(f"{item_id}: configured item runner SHA-256 is invalid")
        else:
            path = _safe_file_under(root, runner["path"])
            if path is None:
                errors.append(f"{item_id}: configured item runner does not exist")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != runner["sha256"]:
                errors.append(f"{item_id}: configured item runner hash mismatch")
    elif runner is not None:
        errors.append(f"{item_id}: unconfigured item cannot name a runner")


def _validate_provenance(
    manifest: dict[str, Any], item: dict[str, Any], root: Path, errors: list[str]
) -> None:
    item_id = item["id"]
    provenance = item["provenance"]
    model = manifest.get("model", {})
    required_revisions = {
        name: revision.get("head")
        for name, revision in manifest.get("source_revisions", {}).items()
        if name in {"vllm", "triton", "pytorch"}
    }
    if set(provenance) != {
        "model_revision",
        "config_sha256",
        "model_manifest_sha256",
        "weight_bytes",
        "weight_sha256",
        "source_lock_sha256",
        "required_repo_commits",
    }:
        errors.append(f"{item_id}: static provenance field set is invalid")
    if provenance.get("model_revision") != model.get("revision"):
        errors.append(f"{item_id}: provenance model revision mismatch")
    if provenance.get("config_sha256") != model.get("config_sha256"):
        errors.append(f"{item_id}: provenance config SHA-256 mismatch")
    if provenance.get("required_repo_commits") != required_revisions:
        errors.append(f"{item_id}: provenance repository commits mismatch")
    for key in (
        "config_sha256",
        "model_manifest_sha256",
        "weight_sha256",
        "source_lock_sha256",
    ):
        if not _is_sha256(provenance.get(key)):
            errors.append(f"{item_id}: provenance {key} is not SHA-256")
    static_files = {
        "model_manifest_sha256": root / "models/Qwen3.5-0.8B/manifest.json",
        "source_lock_sha256": root / "SOURCE_LOCK.json",
    }
    for key, path in static_files.items():
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != provenance.get(key)
        ):
            errors.append(f"{item_id}: provenance {key} does not match source")
    if not isinstance(provenance.get("weight_bytes"), int) or provenance["weight_bytes"] <= 0:
        errors.append(f"{item_id}: provenance weight size is invalid")


def _validate_runtime_policy(item: dict[str, Any], errors: list[str]) -> None:
    item_id = item["id"]
    if item["runtime_evidence"] != runtime_evidence_policy():
        errors.append(f"{item_id}: runtime result policy is not fail-closed")
    fallback = item["fallback"]
    if fallback.get("allowed") is not False:
        errors.append(f"{item_id}: fallback must be forbidden")
    if set(fallback.get("required_zero_counters", [])) != set(
        REQUIRED_FALLBACK_COUNTERS
    ):
        errors.append(f"{item_id}: exact zero fallback counters are missing")
    for key in ("forbidden_backends", "forbidden_dsos", "forbidden_device_nodes"):
        if not isinstance(fallback.get(key), list) or not fallback[key]:
            errors.append(f"{item_id}: fallback {key} policy is missing")


def _validate_artifacts(
    item: dict[str, Any], result: dict[str, Any], root: Path,
    errors: list[str], require_artifacts: bool
) -> dict[str, dict[str, Any]]:
    item_id = item["id"]
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append(f"{item_id}: result artifacts must be a list")
        return {}
    by_role: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {
            "role", "path", "bytes", "sha256"
        }:
            errors.append(f"{item_id}: result artifact descriptor is invalid")
            continue
        role = artifact.get("role")
        if not isinstance(role, str) or role in by_role:
            errors.append(f"{item_id}: result artifact role is invalid or duplicated")
            continue
        by_role[role] = artifact
        if (
            not _safe_relative_path(artifact.get("path"))
            or not isinstance(artifact.get("bytes"), int)
            or artifact["bytes"] < 0
            or not _is_sha256(artifact.get("sha256"))
        ):
            errors.append(f"{item_id}: result artifact identity is invalid")
            continue
        if require_artifacts:
            path = _safe_file_under(root, artifact["path"])
            if path is None:
                errors.append(f"{item_id}: result artifact is absent or escapes root")
            elif (
                path.stat().st_size != artifact["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]
            ):
                errors.append(f"{item_id}: result artifact size or SHA-256 mismatch")
    required_roles = set(item["runtime_evidence"]["required_artifact_roles"])
    if set(by_role) != required_roles:
        errors.append(f"{item_id}: result artifact roles are incomplete or unexpected")
    return by_role


def validate_result(
    item: dict[str, Any], result: dict[str, Any], root: Path = ROOT,
    *, require_artifacts: bool = True
) -> list[str]:
    """Validate one operator-independent runtime result against an item spec."""
    item_id = item.get("id", "<missing-id>")
    errors: list[str] = []
    required_fields = {
        "schema",
        "work_item_id",
        "spec_sha256",
        "run_kind",
        "normal_user_entrypoint",
        "exit_code",
        "target",
        "lifecycle",
        "oracle",
        "fallback",
        "cache",
        "provenance",
        "artifacts",
    }
    if not isinstance(result, dict) or set(result) != required_fields:
        return [f"{item_id}: result field set is invalid"]
    if result.get("schema") != RESULT_SCHEMA:
        errors.append(f"{item_id}: result schema is invalid")
    if result.get("work_item_id") != item_id:
        errors.append(f"{item_id}: result work-item identity mismatch")
    if result.get("spec_sha256") != item.get("spec_sha256"):
        errors.append(f"{item_id}: result spec SHA-256 mismatch")
    run_kind = result.get("run_kind")
    if run_kind not in item["runtime_evidence"]["required_run_kinds"]:
        errors.append(f"{item_id}: result run kind is invalid")
    if result.get("normal_user_entrypoint") is not True or result.get("exit_code") != 0:
        errors.append(f"{item_id}: result did not pass through the normal entrypoint")
    if result.get("target") != item.get("kernel", {}).get("target"):
        errors.append(f"{item_id}: result target differs from the item target")

    lifecycle = result.get("lifecycle")
    required_stages = set(item["runtime_evidence"]["required_lifecycle_stages"])
    if (
        not isinstance(lifecycle, dict)
        or set(lifecycle) != required_stages
        or any(lifecycle.get(stage) is not True for stage in required_stages)
    ):
        errors.append(f"{item_id}: result lifecycle is incomplete")

    oracle = result.get("oracle")
    if (
        not isinstance(oracle, dict)
        or set(oracle) != {
            "passed", "finite", "mismatch_count", "nonfinite_count",
            "output_sha256", "metrics"
        }
        or oracle.get("passed") is not True
        or oracle.get("finite") is not True
        or oracle.get("mismatch_count") != 0
        or oracle.get("nonfinite_count") != 0
        or not _is_sha256(oracle.get("output_sha256"))
        or not isinstance(oracle.get("metrics"), dict)
    ):
        errors.append(f"{item_id}: result oracle did not pass exactly")

    fallback = result.get("fallback")
    counters = fallback.get("counters", {}) if isinstance(fallback, dict) else {}
    if (
        not isinstance(fallback, dict)
        or set(fallback) != {
            "used", "process_audit_passed", "counters", "observed_backends",
            "forbidden_dsos_observed", "forbidden_device_nodes_observed"
        }
        or fallback.get("used") is not False
        or fallback.get("process_audit_passed") is not True
        or fallback.get("observed_backends") != []
        or fallback.get("forbidden_dsos_observed") != []
        or fallback.get("forbidden_device_nodes_observed") != []
        or set(counters) != set(REQUIRED_FALLBACK_COUNTERS)
        or any(counters.get(name) != 0 for name in REQUIRED_FALLBACK_COUNTERS)
    ):
        errors.append(f"{item_id}: result fallback audit is not clean")

    cache = result.get("cache")
    expected_hit = run_kind == "repeat"
    if (
        not isinstance(cache, dict)
        or set(cache) != {"key", "hit"}
        or not isinstance(cache.get("key"), str)
        or not cache["key"]
        or cache.get("hit") is not expected_hit
    ):
        errors.append(f"{item_id}: result cache state is invalid")

    provenance = result.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != RUNTIME_PROVENANCE_FIELDS:
        errors.append(f"{item_id}: result runtime provenance field set is invalid")
        provenance = {}
    else:
        for key in ("gem5_tree", "runtime_commit", "triton_commit"):
            if not _is_git_oid(provenance.get(key)):
                errors.append(f"{item_id}: result provenance {key} is invalid")
        for key in RUNTIME_PROVENANCE_FIELDS - {
            "gem5_tree", "runtime_commit", "triton_commit"
        }:
            if not _is_sha256(provenance.get(key)):
                errors.append(f"{item_id}: result provenance {key} is invalid")
        runner = item.get("source", {}).get("runner", {})
        if provenance.get("runner_sha256") != runner.get("sha256"):
            errors.append(f"{item_id}: result runner identity mismatch")
        expected_triton = item.get("provenance", {}).get(
            "required_repo_commits", {}
        ).get("triton")
        if provenance.get("triton_commit") != expected_triton:
            errors.append(f"{item_id}: result Triton revision mismatch")

    artifacts = _validate_artifacts(item, result, root, errors, require_artifacts)
    code_object = artifacts.get("code_object", {})
    if code_object.get("sha256") != provenance.get("code_object_sha256"):
        errors.append(f"{item_id}: result code-object artifact is not provenance-bound")
    return errors


def derive_work_item_status(
    item: dict[str, Any], results: list[dict[str, Any]] | None = None,
    root: Path = ROOT, errors: list[str] | None = None,
    *, require_artifacts: bool = True
) -> str:
    local_errors = errors if errors is not None else []
    if item.get("configuration_status") != "configured":
        return "unconfigured"
    selected = [
        result for result in (results or [])
        if isinstance(result, dict) and result.get("work_item_id") == item.get("id")
    ]
    if not selected:
        return "ready"
    for result in selected:
        local_errors.extend(
            validate_result(item, result, root, require_artifacts=require_artifacts)
        )
    if local_errors:
        return "failed"
    by_kind = {result["run_kind"]: result for result in selected}
    if len(by_kind) != len(selected) or set(by_kind) != set(REQUIRED_RUN_KINDS):
        local_errors.append(f"{item['id']}: results do not contain one fresh and repeat run")
        return "incomplete"
    stable_fields = (
        lambda result: (
            result["oracle"]["output_sha256"],
            result["cache"]["key"],
            result["provenance"]["code_object_sha256"],
            result["provenance"]["environment_sha256"],
        )
    )
    if stable_fields(by_kind["fresh"]) != stable_fields(by_kind["repeat"]):
        local_errors.append(f"{item['id']}: fresh and repeat identities differ")
        return "failed"
    return "accepted"


def validate_manifest(manifest: dict[str, Any], root: Path = ROOT) -> list[str]:
    """Validate a deterministic source manifest without consulting results."""
    errors: list[str] = []
    if manifest.get("schema") != SCHEMA:
        return [f"manifest schema must be {SCHEMA}"]
    contracts = manifest.get("contracts")
    items = manifest.get("work_items")
    if not isinstance(contracts, list) or len(contracts) != 15:
        errors.append("manifest must contain exactly 15 contracts")
        contracts = contracts if isinstance(contracts, list) else []
    if not isinstance(items, list) or len(items) != 32:
        errors.append("manifest must contain exactly 32 required work items")
        items = items if isinstance(items, list) else []

    contract_ids = [contract.get("id") for contract in contracts]
    item_ids = [item.get("id") for item in items]
    if len(set(contract_ids)) != len(contract_ids):
        errors.append("contract IDs must be unique")
    if len(set(item_ids)) != len(item_ids):
        errors.append("work-item IDs must be unique")
    contract_by_id = {contract.get("id"): contract for contract in contracts}
    item_by_id = {item.get("id"): item for item in items}

    required_item_fields = {
        "id",
        "contract_id",
        "phase",
        "variant",
        "acceptance_role",
        "required_for_contract",
        "configuration_status",
        "configuration_errors",
        "status",
        "spec_sha256",
        "tensors",
        "mutation",
        "state",
        "oracle",
        "dependencies",
        "kernel",
        "source",
        "provenance",
        "runtime_evidence",
        "fallback",
    }
    for item in items:
        if not isinstance(item, dict):
            errors.append("work-item entry is not an object")
            continue
        item_id = item.get("id", "<missing-id>")
        missing = sorted(required_item_fields - item.keys())
        if missing:
            errors.append(f"{item_id}: missing fields: {','.join(missing)}")
            continue
        if item["contract_id"] not in contract_by_id:
            errors.append(f"{item_id}: unknown contract {item['contract_id']!r}")
        if item["phase"] not in VALID_PHASES:
            errors.append(f"{item_id}: invalid phase")
        if item["acceptance_role"] not in VALID_ROLES:
            errors.append(f"{item_id}: invalid acceptance role")
        if item["acceptance_role"] != "required" or item["required_for_contract"] is not True:
            errors.append(f"{item_id}: all materialized work items must be required")
        if item["configuration_status"] not in VALID_CONFIGURATIONS:
            errors.append(f"{item_id}: invalid configuration status")
        if item["status"] not in VALID_SOURCE_STATUSES:
            errors.append(f"{item_id}: source manifest status is invalid")
        if item["configuration_status"] == "unconfigured":
            if not item["configuration_errors"]:
                errors.append(f"{item_id}: unconfigured item needs a blocker")
        else:
            if item["configuration_errors"]:
                errors.append(f"{item_id}: configured item has configuration errors")
            _validate_configured_item(manifest, item, errors)
        if item["spec_sha256"] != work_item_spec_sha256(item):
            errors.append(f"{item_id}: work-item spec SHA-256 mismatch")

        dependencies = item["dependencies"]
        if not isinstance(dependencies.get("all_of_work_items"), list):
            errors.append(f"{item_id}: dependencies must be a list")
        else:
            for dependency in dependencies["all_of_work_items"]:
                if dependency not in item_by_id:
                    errors.append(f"{item_id}: unknown dependency {dependency!r}")
        _validate_runtime_policy(item, errors)
        _validate_source(item, root, errors)
        _validate_provenance(manifest, item, root, errors)
        expected_status = derive_work_item_status(item)
        if item["status"] != expected_status:
            errors.append(
                f"{item_id}: stored status {item['status']!r} != {expected_status!r}"
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(item_id: str) -> None:
        if item_id in visited or item_id not in item_by_id:
            return
        if item_id in visiting:
            errors.append(f"dependency cycle includes {item_id}")
            return
        visiting.add(item_id)
        for dependency in item_by_id[item_id]["dependencies"].get(
            "all_of_work_items", []
        ):
            visit(dependency)
        visiting.remove(item_id)
        visited.add(item_id)

    for item_id in item_by_id:
        visit(item_id)

    referenced_items: set[str] = set()
    for contract in contracts:
        contract_id = contract.get("id", "<missing-id>")
        required_ids = contract.get("required_work_item_ids")
        if contract.get("completion_rule") != "all_required_items_accepted":
            errors.append(f"{contract_id}: invalid completion rule")
        if not isinstance(required_ids, list) or not required_ids:
            errors.append(f"{contract_id}: required work items are missing")
            required_ids = []
        if contract.get("partial_work_item_ids") != []:
            errors.append(f"{contract_id}: source queue cannot embed partial evidence")
        if contract.get("work_queue_status") != "not_accepted":
            errors.append(f"{contract_id}: source-only contract cannot be accepted")
        for item_id in required_ids:
            item = item_by_id.get(item_id)
            if not item or item.get("contract_id") != contract_id:
                errors.append(f"{contract_id}: invalid required item {item_id!r}")
            referenced_items.add(item_id)
    if referenced_items != set(item_by_id):
        errors.append("contract required-item lists do not cover the queue exactly")

    summary = manifest.get("summary", {})
    if (
        summary.get("accepted_contract_count") != 0
        or summary.get("all_contracts_accepted") is not False
        or summary.get("amd_runtime_executed") is not False
        or summary.get("amd_runtime_pass") is not False
    ):
        errors.append("source-only summary contains a runtime acceptance claim")
    return errors


def queue_summary(
    manifest: dict[str, Any], results: list[dict[str, Any]] | None = None,
    root: Path = ROOT, *, require_artifacts: bool = True
) -> dict[str, Any]:
    items = manifest["work_items"]
    status_errors: list[str] = []
    statuses = {
        item["id"]: derive_work_item_status(
            item,
            results,
            root,
            status_errors,
            require_artifacts=require_artifacts,
        )
        for item in items
    }
    changed = True
    while changed:
        changed = False
        for item in items:
            if statuses[item["id"]] != "accepted":
                continue
            dependencies = item["dependencies"].get("all_of_work_items", [])
            if any(statuses.get(dependency) != "accepted" for dependency in dependencies):
                statuses[item["id"]] = "blocked"
                changed = True
    accepted_contracts = sum(
        bool(contract["required_work_item_ids"])
        and all(statuses.get(item_id) == "accepted" for item_id in contract["required_work_item_ids"])
        for contract in manifest["contracts"]
    )
    return {
        "contract_count": len(manifest["contracts"]),
        "accepted_contract_count": accepted_contracts,
        "work_item_count": len(items),
        "configured_work_item_count": sum(
            item["configuration_status"] == "configured" for item in items
        ),
        "ready_work_item_count": sum(status == "ready" for status in statuses.values()),
        "accepted_work_item_count": sum(
            status == "accepted" for status in statuses.values()
        ),
        "all_contracts_accepted": accepted_contracts == len(manifest["contracts"]),
        "result_errors": status_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=ROOT / "tools/qwen35_operator_manifest.json",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="validate a JSON list (or an object with a results list) of external results",
    )
    args = parser.parse_args()
    manifest = _load_json(args.manifest)
    if not isinstance(manifest, dict):
        print(json.dumps({"errors": ["manifest is not a JSON object"]}))
        return 1
    errors = validate_manifest(manifest)
    results: list[dict[str, Any]] = []
    if args.results is not None:
        loaded = _load_json(args.results)
        if isinstance(loaded, dict):
            loaded = loaded.get("results")
        if not isinstance(loaded, list):
            errors.append("results file must contain a JSON list")
        else:
            results = loaded
    summary = queue_summary(manifest, results) if not errors else None
    if summary and summary["result_errors"]:
        errors.extend(summary["result_errors"])
    print(json.dumps({"errors": errors, "summary": summary}, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
