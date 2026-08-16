#!/usr/bin/env python3

import runpy
from pathlib import Path


HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE / "_gemsim_bootstrap.py"))["bootstrap"](
    __file__, "qwen35-tied-lm-head"
)

import argparse
import hashlib
import json
import math

import torch
import triton
from safetensors import safe_open


ROOT = HERE.parents[1]
MODEL_DIR = ROOT / "models/Qwen3.5-0.8B"
MODEL_FILENAME = "model.safetensors-00001-of-00001.safetensors"
MODEL_FILE = MODEL_DIR / MODEL_FILENAME
EMBEDDING_WEIGHT_NAME = "model.language_model.embed_tokens.weight"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_ARTIFACT_SCHEMA = "amdgpu-sim.triton-qwen35-model-backbone-decode.v1"
MODEL_STATE_SCHEMA = "amdgpu-sim.qwen35-model-final-state.v1"
NVIDIA_GOLDEN_SCHEMA = "amdgpu-sim.qwen35-nvidia-lm-head-golden.v1"
NVIDIA_GOLDEN_KIND = "independent_torch_cuda_tied_lm_head_golden"
NVIDIA_GOLDEN_ACCEPTED_SHA256 = {
    "metadata.json": "69fb64f51f622f600596b3d07f88ba154e9ab90bfbada5915fed1385d1841de0",
    "results.safetensors": "be0073d97143d08e1e24da3d256c014a8e0be37d96c84cfca3dac81b539335a7",
    "script_sha256": "b165387b450ffb2a33b3813ed69342abd4080fdf3906a2089977a20d8be4cb05",
}
NVIDIA_BACKBONE_ACCEPTED_SHA256 = {
    "metadata.json": "2a5d43d9c8b068ad15027916db4120782fc99b7ffa48bf225073bc09f909a9fb",
    "results.safetensors": "43a6b9f8d2cc29c728444ead69f7a0df575d634b35593bc1b2490b9ed0adfb9b",
}
NVIDIA_BACKBONE_SCRIPT_SHA256 = (
    "bd630ee9693f5ea79c7ba7c05d4981b985dcb5b8e4cf119ee8ea8aaa20535231"
)
NVIDIA_FINAL_NORM_SHA256 = (
    "f309907965a721aee3ce35e0c300eca4cb34edb8ff203000f82f047dcd6ab994"
)
TIED_WEIGHT_SHA256 = (
    "3247e63f0f265462e2fba5316dfb2819941ea8ed62ab5b2c4904e4aab5b9d7aa"
)
NVIDIA_LOGITS_SHA256 = (
    "90078f44f1f1272d9ebbe505d788d86b2b71d7aef7c3eeaea63764201de049fa"
)
NVIDIA_GPU = {
    "name": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "uuid": "GPU-64aae36b-ef77-b0d4-b1c7-f7ab17a729f1",
    "compute_capability": [12, 0],
}
TOKEN_ID = 248044
VOCAB_SIZE = 248320
HIDDEN_SIZE = 1024
FULL_VOCAB_CHUNK_ROWS = 4096
GUARD_ELEMENTS = 1024
ABSOLUTE_TOLERANCE = 0.03125
RELATIVE_TOLERANCE = 0.03
TOP_K = 20
# Cross-architecture BF16 accumulation profile. These gates are applied only
# to the complete user-visible logits vector; every target kernel still has a
# strict local FP32 oracle and finite/guard/fallback checks.
MODEL_LOGITS_ACCEPTANCE_PROFILE = "bf16_cross_arch_v1"
MODEL_LOGITS_MIN_COSINE_SIMILARITY = 0.975
MODEL_LOGITS_MAX_RELATIVE_L2_ERROR = 0.22
MODEL_LOGITS_MIN_TOP_K_TOKEN_OVERLAP = 0.80
BOUNDARY_VOCAB_IDS = (
    0,
    1,
    15,
    16,
    31,
    32,
    64,
    255,
    256,
    266,
    848,
    1023,
    2022,
    36933,
    65535,
    131071,
    248044,
    248318,
    248319,
)
EXPECTED_LAYER_TYPES = tuple(
    layer_type
    for _ in range(6)
    for layer_type in (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )
)
EXPECTED_ARTIFACTS = {
    "config.json": {
        "bytes": 2907,
        "sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    },
    "model.safetensors.index.json": {
        "bytes": 50900,
        "sha256": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
    },
    MODEL_FILENAME: {
        "bytes": 1746942600,
        "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    },
    "manifest.json": {
        "bytes": 1008,
        "sha256": "de2281cc73a1329d13245cb9658be910cf435e72c4ea0277c4f8811a24edf762",
    },
}


DEVICE = triton.runtime.driver.active.get_active_torch_device()
dense_linear = runpy.run_path(
    str(HERE / "qwen35_dense_linear_correctness.py")
)["dense_linear"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(
        order="C"
    )


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(value)).hexdigest()


def tensor_descriptor(value: torch.Tensor) -> dict:
    raw = tensor_bytes(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def require_sha256(value: object, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"invalid SHA-256 field: {field}")


def validate_checkpoint() -> tuple[dict, dict]:
    artifact_hashes = {}
    for filename, expected in EXPECTED_ARTIFACTS.items():
        path = MODEL_DIR / filename
        observed = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        if observed != expected:
            raise RuntimeError(
                f"checkpoint artifact mismatch for {filename}: {observed!r}"
            )
        artifact_hashes[filename] = observed

    config = json.loads((MODEL_DIR / "config.json").read_text())
    text_config = config.get("text_config", {})
    config_contract = {
        "architecture": config.get("architectures"),
        "root_tie_word_embeddings": config.get("tie_word_embeddings"),
        "text_model_type": text_config.get("model_type"),
        "text_dtype": text_config.get("dtype"),
        "text_hidden_size": text_config.get("hidden_size"),
        "text_vocab_size": text_config.get("vocab_size"),
        "text_tie_word_embeddings": text_config.get("tie_word_embeddings"),
    }
    expected_config_contract = {
        "architecture": ["Qwen3_5ForConditionalGeneration"],
        "root_tie_word_embeddings": True,
        "text_model_type": "qwen3_5_text",
        "text_dtype": "bfloat16",
        "text_hidden_size": HIDDEN_SIZE,
        "text_vocab_size": VOCAB_SIZE,
        "text_tie_word_embeddings": True,
    }
    if config_contract != expected_config_contract:
        raise RuntimeError(f"checkpoint config contract mismatch: {config_contract!r}")

    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map", {})
    independent_lm_heads = sorted(
        name for name in weight_map if name.endswith("lm_head.weight")
    )
    if weight_map.get(EMBEDDING_WEIGHT_NAME) != MODEL_FILENAME:
        raise RuntimeError("embedding index mapping does not name the pinned shard")
    if independent_lm_heads:
        raise RuntimeError(
            f"tied checkpoint contains independent LM head weights: {independent_lm_heads!r}"
        )

    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    if (
        manifest.get("model_id") != MODEL_ID
        or manifest.get("revision") != MODEL_REVISION
        or any(
            manifest.get("files", {}).get(name) != expected
            for name, expected in EXPECTED_ARTIFACTS.items()
            if name != "manifest.json"
        )
    ):
        raise RuntimeError("checkpoint manifest provenance contract mismatch")

    provenance = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "artifact_hashes": artifact_hashes,
        "index_total_size": index.get("metadata", {}).get("total_size"),
        "config_contract": config_contract,
        "embedding_weight_name": EMBEDDING_WEIGHT_NAME,
        "embedding_weight_file": MODEL_FILENAME,
        "embedding_weight_shape": [VOCAB_SIZE, HIDDEN_SIZE],
        "lm_head_storage": "tied_to_embed_tokens",
        "independent_lm_head_present": False,
        "config_contract_verified": True,
        "index_mapping_verified": True,
        "manifest_provenance_verified": True,
    }
    return provenance, weight_map


def resolve_exact_directory(path: Path, expected_names: set[str], label: str) -> dict:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise RuntimeError(f"{label} directory must not be a symlink")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_dir():
        raise RuntimeError(f"{label} path is not a directory: {resolved}")
    entries = {entry.name: entry for entry in resolved.iterdir()}
    if set(entries) != expected_names:
        raise RuntimeError(f"{label} directory contents mismatch: {sorted(entries)!r}")
    for name, entry in entries.items():
        if entry.is_symlink() or not entry.is_file():
            raise RuntimeError(f"{label} entry is not a regular file: {name}")
    entries["__directory__"] = resolved
    return entries


def validate_model_output_dir(path: Path, checkpoint: dict) -> tuple[torch.Tensor, dict]:
    entries = resolve_exact_directory(
        path,
        {"execution_metadata.json", "final_state.safetensors"},
        "model output",
    )
    metadata_path = entries["execution_metadata.json"]
    state_path = entries["final_state.safetensors"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError("model execution metadata must be a JSON object")
    expected_scalars = {
        "schema": MODEL_ARTIFACT_SCHEMA,
        "backend": "gemsim_amd",
        "arch": "gfx950",
        "model": MODEL_ID,
        "mode": "checkpoint_first_token_empty_cache",
        "token_id": TOKEN_ID,
        "layers_executed": 24,
        "final_norm_applied": True,
        "all_layer_states_finite": True,
        "final_state_finite": True,
        "all_finite": True,
        "fallback_count": 0,
        "cpu_fallback_count": 0,
        "nvidia_fallback_count": 0,
        "execution_complete": True,
        "requested_layer_range_complete": True,
        "full_24_layer_backbone_complete": True,
        "execution_status": "completed_with_external_nvidia_golden",
        "golden_provided": True,
        "golden_compared": True,
    }
    for name, expected in expected_scalars.items():
        if metadata.get(name) != expected:
            raise RuntimeError(
                f"model execution contract mismatch for {name}: {metadata.get(name)!r}"
            )

    trajectory_verified = metadata.get("correctness_verified")
    if trajectory_verified not in (True, False):
        raise RuntimeError("model trajectory verdict must be boolean")
    expected_trajectory_status = (
        "verified_against_external_nvidia_golden"
        if trajectory_verified
        else "failed_external_nvidia_golden"
    )
    if (
        metadata.get("correctness_status") != expected_trajectory_status
        or metadata.get("output_correct") is not trajectory_verified
    ):
        raise RuntimeError("model trajectory verdict is internally inconsistent")

    embedded_checkpoint = metadata.get("checkpoint_provenance")
    if not isinstance(embedded_checkpoint, dict):
        raise RuntimeError("model checkpoint provenance is missing")
    expected_checkpoint = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "artifact_hashes": checkpoint["artifact_hashes"],
        "selected_tensor_count": 320,
        "selected_tensor_mapping_verified": True,
        "manifest_provenance_verified": True,
        "index_metadata_verified": True,
        "independent_lm_head_present": False,
    }
    for name, expected in expected_checkpoint.items():
        if embedded_checkpoint.get(name) != expected:
            raise RuntimeError(f"model checkpoint provenance mismatch for {name}")

    layer_records = metadata.get("layer_records")
    if not isinstance(layer_records, list) or len(layer_records) != 24:
        raise RuntimeError("model output must contain exactly 24 layer records")
    for layer, (record, layer_type) in enumerate(
        zip(layer_records, EXPECTED_LAYER_TYPES, strict=True)
    ):
        if (
            not isinstance(record, dict)
            or record.get("layer") != layer
            or record.get("type") != layer_type
            or record.get("finite") is not True
        ):
            raise RuntimeError(f"model layer record mismatch at layer {layer}")
        for name in ("hidden_sha256", "residual_sha256"):
            require_sha256(record.get(name), f"layer_records[{layer}].{name}")
    selected_hashes = metadata.get("selected_tensor_hashes")
    if (
        not isinstance(selected_hashes, dict)
        or metadata.get("selected_tensor_hash_count") != len(selected_hashes)
        or len(selected_hashes) != 320
    ):
        raise RuntimeError("model selected tensor hash coverage mismatch")
    for name, value in selected_hashes.items():
        require_sha256(value, f"selected_tensor_hashes.{name}")
    assembled_hashes = metadata.get("assembled_weight_hashes")
    if (
        not isinstance(assembled_hashes, dict)
        or metadata.get("assembled_weight_hash_count") != len(assembled_hashes)
    ):
        raise RuntimeError("model assembled weight hash coverage mismatch")
    for name, value in assembled_hashes.items():
        require_sha256(value, f"assembled_weight_hashes.{name}")

    nvidia = metadata.get("nvidia_golden")
    if (
        not isinstance(nvidia, dict)
        or nvidia.get("enabled") is not True
        or nvidia.get("comparison_count") != 49
        or nvidia.get("nonfinite_count") != 0
        or nvidia.get("metadata_sha256")
        != NVIDIA_BACKBONE_ACCEPTED_SHA256["metadata.json"]
        or nvidia.get("results_sha256")
        != NVIDIA_BACKBONE_ACCEPTED_SHA256["results.safetensors"]
        or nvidia.get("generator")
        != {
            "path": "tools/qwen35_nvidia_golden.py",
            "sha256": NVIDIA_BACKBONE_SCRIPT_SHA256,
        }
    ):
        raise RuntimeError("model NVIDIA backbone comparison contract mismatch")
    layer_comparisons = nvidia.get("layer_comparisons")
    final_norm_comparison = nvidia.get("final_norm")
    if (
        not isinstance(layer_comparisons, list)
        or len(layer_comparisons) != 24
        or not isinstance(final_norm_comparison, dict)
    ):
        raise RuntimeError("model NVIDIA comparison records are incomplete")
    comparison_records = []
    for layer, (comparison_record, layer_type) in enumerate(
        zip(layer_comparisons, EXPECTED_LAYER_TYPES, strict=True)
    ):
        if (
            not isinstance(comparison_record, dict)
            or comparison_record.get("layer") != layer
            or comparison_record.get("type") != layer_type
        ):
            raise RuntimeError(f"model NVIDIA comparison mismatch at layer {layer}")
        hidden_comparison = comparison_record.get("returned_hidden")
        residual_comparison = comparison_record.get("returned_residual")
        if not isinstance(hidden_comparison, dict) or not isinstance(
            residual_comparison, dict
        ):
            raise RuntimeError(f"model NVIDIA tensor comparison missing at layer {layer}")
        if (
            hidden_comparison.get("actual_sha256")
            != layer_records[layer]["hidden_sha256"]
            or residual_comparison.get("actual_sha256")
            != layer_records[layer]["residual_sha256"]
        ):
            raise RuntimeError(f"model NVIDIA actual hash mismatch at layer {layer}")
        comparison_records.extend((hidden_comparison, residual_comparison))
    comparison_records.append(final_norm_comparison)
    if (
        final_norm_comparison.get("actual_sha256")
        != metadata.get("nvidia_comparison_final_norm_sha256")
        or final_norm_comparison.get("expected_sha256")
        != NVIDIA_FINAL_NORM_SHA256
    ):
        raise RuntimeError("model NVIDIA final-norm comparison identity mismatch")
    aggregate_mismatches = 0
    for index, comparison_record in enumerate(comparison_records):
        if (
            comparison_record.get("all_values_finite") is not True
            or comparison_record.get("nonfinite_count") != 0
            or not isinstance(comparison_record.get("mismatch_count"), int)
            or comparison_record["mismatch_count"] < 0
            or not isinstance(comparison_record.get("correct"), bool)
        ):
            raise RuntimeError(f"model NVIDIA comparison {index} is malformed")
        require_sha256(
            comparison_record.get("actual_sha256"),
            f"nvidia_golden.comparisons[{index}].actual_sha256",
        )
        require_sha256(
            comparison_record.get("expected_sha256"),
            f"nvidia_golden.comparisons[{index}].expected_sha256",
        )
        aggregate_mismatches += comparison_record["mismatch_count"]
    comparisons_correct = all(record["correct"] for record in comparison_records)
    if (
        nvidia.get("all_comparisons_correct") is not comparisons_correct
        or nvidia.get("mismatch_count") != aggregate_mismatches
        or comparisons_correct is not trajectory_verified
    ):
        raise RuntimeError("model NVIDIA aggregate comparison verdict mismatch")

    artifacts = metadata.get("output_artifacts")
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("write_policy")
        != "same_parent_temp_directory_fsync_renameat2_noreplace_parent_fsync"
    ):
        raise RuntimeError("model artifact write policy mismatch")
    state_record = artifacts.get("final_state")
    metadata_record = artifacts.get("execution_metadata")
    if not isinstance(state_record, dict) or not isinstance(metadata_record, dict):
        raise RuntimeError("model output artifact records are missing")
    if Path(state_record.get("path", "")).resolve(strict=True) != state_path:
        raise RuntimeError("model final-state recorded path mismatch")
    if Path(metadata_record.get("path", "")).resolve(strict=True) != metadata_path:
        raise RuntimeError("model execution-metadata recorded path mismatch")
    if (
        state_record.get("bytes") != state_path.stat().st_size
        or state_record.get("sha256") != file_sha256(state_path)
        or state_record.get("tensor_keys")
        != ["final_hidden", "final_residual", "nvidia_comparison_final_norm"]
    ):
        raise RuntimeError("model final-state artifact descriptor mismatch")

    with safe_open(state_path, framework="pt", device="cpu") as tensors:
        if set(tensors.keys()) != {
            "final_hidden",
            "final_residual",
            "nvidia_comparison_final_norm",
        }:
            raise RuntimeError("model final-state tensor set mismatch")
        expected_embedded = {
            "schema": MODEL_STATE_SCHEMA,
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "token_id": str(TOKEN_ID),
            "layers_executed": "24",
            "correctness_verified": str(trajectory_verified).lower(),
        }
        if tensors.metadata() != expected_embedded:
            raise RuntimeError("model final-state embedded provenance mismatch")
        hidden = tensors.get_tensor("final_hidden").clone().contiguous()
        comparison = tensors.get_tensor(
            "nvidia_comparison_final_norm"
        ).clone().contiguous()
        residual = tensors.get_tensor("final_residual").clone().contiguous()
    for name, value in (("final_hidden", hidden), ("final_residual", residual)):
        if value.dtype != torch.bfloat16 or tuple(value.shape) != (1, HIDDEN_SIZE):
            raise RuntimeError(
                f"model {name} contract mismatch: {value.dtype} {tuple(value.shape)}"
            )
    if not torch.equal(hidden, comparison):
        raise RuntimeError("model final_hidden is not the persisted final_norm")
    if tensor_sha256(hidden) != metadata.get("final_hidden_sha256"):
        raise RuntimeError("model final_hidden hash mismatch")
    if tensor_sha256(residual) != metadata.get("final_residual_sha256"):
        raise RuntimeError("model final_residual hash mismatch")
    if tensor_sha256(comparison) != metadata.get(
        "nvidia_comparison_final_norm_sha256"
    ):
        raise RuntimeError("model comparison final_norm hash mismatch")
    if not bool(torch.all(torch.isfinite(hidden.float())).item()):
        raise RuntimeError("model final_hidden contains nonfinite values")
    return hidden, {
        "mode": "validated_24_layer_model_execution_output",
        "directory": str(entries["__directory__"]),
        "execution_metadata_sha256": file_sha256(metadata_path),
        "final_state_sha256": file_sha256(state_path),
        "tensor_key": "final_hidden",
        "tensor_sha256": tensor_sha256(hidden),
        "token_id": TOKEN_ID,
        "layers_executed": 24,
        "final_norm_applied": True,
        "execution_complete": True,
        "backbone_execution_verified": True,
        "backbone_trajectory_verified": trajectory_verified,
        "backbone_trajectory_status": expected_trajectory_status,
        "backbone_trajectory_mismatch_count": aggregate_mismatches,
    }


def stable_decision(logits: torch.Tensor, scope: str, vocab_ids: list[int]) -> dict:
    values = logits.view(-1).to(torch.float32).cpu()
    maximum = torch.max(values)
    local_ties = torch.nonzero(values == maximum, as_tuple=False).view(-1).tolist()
    tie_ids = sorted(vocab_ids[index] for index in local_ties)
    ordered = sorted(
        (
            (float(values[index].item()), vocab_id)
            for index, vocab_id in enumerate(vocab_ids)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    top = ordered[: min(TOP_K, len(ordered))]
    prefix = "greedy" if scope == "full_vocab" else "bounded_argmax"
    return {
        "scope": scope,
        "policy": "maximum BF16 logit; lowest token ID breaks exact ties",
        f"{prefix}_token_id": tie_ids[0],
        f"{prefix}_logit": float(maximum.item()),
        "maximum_tie_count": len(tie_ids),
        "maximum_tie_token_ids": tie_ids,
        "top_k": len(top),
        "top_k_entries": [
            {"rank": rank, "token_id": token_id, "logit": value}
            for rank, (value, token_id) in enumerate(top, start=1)
        ],
    }


def decisions_match(actual: dict, expected: dict) -> tuple[bool, bool]:
    if actual.get("scope") != expected.get("scope"):
        return False, False
    prefix = "greedy" if actual["scope"] == "full_vocab" else "bounded_argmax"
    decision_matches = (
        actual.get(f"{prefix}_token_id") == expected.get(f"{prefix}_token_id")
        and actual.get("maximum_tie_count") == expected.get("maximum_tie_count")
        and actual.get("maximum_tie_token_ids")
        == expected.get("maximum_tie_token_ids")
    )
    top_k_token_ids_match = [
        entry["token_id"] for entry in actual.get("top_k_entries", [])
    ] == [entry["token_id"] for entry in expected.get("top_k_entries", [])]
    return decision_matches, top_k_token_ids_match


def compare_full_logit_vectors(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual_values = actual.to(torch.float64).view(-1)
    expected_values = expected.to(torch.float64).view(-1)
    finite = torch.isfinite(actual_values) & torch.isfinite(expected_values)
    nonfinite_count = int(torch.count_nonzero(~finite).item())
    if nonfinite_count:
        return {
            "values_compared": actual_values.numel(),
            "nonfinite_count": nonfinite_count,
            "cosine_similarity": None,
            "relative_l2_error": None,
            "max_abs_error": None,
            "finite": False,
        }
    actual_norm = torch.linalg.vector_norm(actual_values)
    expected_norm = torch.linalg.vector_norm(expected_values)
    if actual_norm.item() == 0.0 or expected_norm.item() == 0.0:
        raise RuntimeError("full logits have a zero L2 norm")
    error = actual_values - expected_values
    return {
        "values_compared": actual_values.numel(),
        "nonfinite_count": 0,
        "cosine_similarity": float(
            (torch.dot(actual_values, expected_values) / (actual_norm * expected_norm)).item()
        ),
        "relative_l2_error": float(
            (torch.linalg.vector_norm(error) / expected_norm).item()
        ),
        "max_abs_error": float(torch.max(torch.abs(error)).item()),
        "finite": True,
    }


def top_k_token_overlap(actual: dict, expected: dict) -> dict:
    actual_ids = {
        entry["token_id"] for entry in actual.get("top_k_entries", [])
    }
    expected_ids = {
        entry["token_id"] for entry in expected.get("top_k_entries", [])
    }
    if len(actual_ids) != TOP_K or len(expected_ids) != TOP_K:
        raise RuntimeError("full-vocabulary top-k decision does not contain 20 unique IDs")
    overlap = sorted(actual_ids & expected_ids)
    return {
        "k": TOP_K,
        "overlap_count": len(overlap),
        "overlap_fraction": len(overlap) / TOP_K,
        "overlap_token_ids": overlap,
    }


def load_nvidia_golden(
    path: Path, checkpoint: dict, hidden: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    entries = resolve_exact_directory(
        path, {"metadata.json", "results.safetensors"}, "NVIDIA LM-head golden"
    )
    for name in ("metadata.json", "results.safetensors"):
        observed = file_sha256(entries[name])
        if observed != NVIDIA_GOLDEN_ACCEPTED_SHA256[name]:
            raise RuntimeError(f"unaccepted NVIDIA LM-head {name}: {observed}")
    metadata = json.loads(entries["metadata.json"].read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError("NVIDIA LM-head metadata must be a JSON object")
    if (
        metadata.get("schema") != NVIDIA_GOLDEN_SCHEMA
        or metadata.get("kind") != NVIDIA_GOLDEN_KIND
        or metadata.get("results_file_sha256")
        != NVIDIA_GOLDEN_ACCEPTED_SHA256["results.safetensors"]
        or metadata.get("script")
        != {
            "path": "tools/qwen35_nvidia_lm_head_golden.py",
            "sha256": NVIDIA_GOLDEN_ACCEPTED_SHA256["script_sha256"],
        }
        or metadata.get("write_policy")
        != "same_parent_temp_directory_fsync_renameat2_noreplace_parent_fsync"
    ):
        raise RuntimeError("NVIDIA LM-head golden identity mismatch")
    expected_model = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "artifact_hashes": checkpoint["artifact_hashes"],
        "index_total_size": 1746882752,
        "embedding_weight_name": EMBEDDING_WEIGHT_NAME,
        "embedding_weight_file": MODEL_FILENAME,
        "embedding_weight_shape": [VOCAB_SIZE, HIDDEN_SIZE],
        "lm_head_storage": "tied_to_embed_tokens",
        "independent_lm_head_present": False,
        "config_contract_verified": True,
        "index_mapping_verified": True,
        "manifest_provenance_verified": True,
    }
    if metadata.get("model") != expected_model:
        raise RuntimeError("NVIDIA LM-head checkpoint provenance mismatch")
    expected_backbone = {
        "schema": "amdgpu-sim.qwen35-nvidia-backbone-golden.v1",
        "metadata_sha256": NVIDIA_BACKBONE_ACCEPTED_SHA256["metadata.json"],
        "results_sha256": NVIDIA_BACKBONE_ACCEPTED_SHA256["results.safetensors"],
        "generator_sha256": NVIDIA_BACKBONE_SCRIPT_SHA256,
        "token_id": TOKEN_ID,
        "layers": 24,
        "position": 0,
        "cache": "empty_per_layer",
        "final_norm": {
            "shape": [1, HIDDEN_SIZE],
            "dtype": "bfloat16",
            "bytes": HIDDEN_SIZE * 2,
            "sha256": NVIDIA_FINAL_NORM_SHA256,
        },
    }
    backbone = metadata.get("source_backbone_golden")
    if not isinstance(backbone, dict) or any(
        backbone.get(name) != value for name, value in expected_backbone.items()
    ):
        raise RuntimeError("NVIDIA LM-head source backbone contract mismatch")
    if not isinstance(backbone.get("directory"), str) or not backbone["directory"]:
        raise RuntimeError("NVIDIA LM-head source backbone directory is missing")
    expected_case = {
        "scope": "full_vocabulary_tied_lm_head",
        "token_id": TOKEN_ID,
        "tokens": 1,
        "input_shape": [1, HIDDEN_SIZE],
        "weight_shape": [VOCAB_SIZE, HIDDEN_SIZE],
        "output_shape": [1, VOCAB_SIZE],
        "input_dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "output_dtype": "bfloat16",
        "formula": "bf16(fp32(final_norm) @ fp32(embed_tokens.weight).T)",
        "sampling_in_scope": False,
    }
    if metadata.get("case") != expected_case:
        raise RuntimeError("NVIDIA LM-head case contract mismatch")
    expected_hidden_descriptor = {
        "shape": [1, HIDDEN_SIZE],
        "dtype": "bfloat16",
        "bytes": HIDDEN_SIZE * 2,
        "sha256": NVIDIA_FINAL_NORM_SHA256,
    }
    if metadata.get("input_final_norm") != expected_hidden_descriptor:
        raise RuntimeError("NVIDIA input final_norm descriptor mismatch")
    if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (1, HIDDEN_SIZE):
        raise RuntimeError("target hidden tensor contract mismatch")
    if metadata.get("tied_embedding_weight") != {
        "shape": [VOCAB_SIZE, HIDDEN_SIZE],
        "dtype": "bfloat16",
        "bytes": VOCAB_SIZE * HIDDEN_SIZE * 2,
        "sha256": TIED_WEIGHT_SHA256,
    }:
        raise RuntimeError("NVIDIA tied embedding descriptor mismatch")
    expected_logits_descriptor = {
        "shape": [1, VOCAB_SIZE],
        "dtype": "bfloat16",
        "bytes": VOCAB_SIZE * 2,
        "sha256": NVIDIA_LOGITS_SHA256,
    }
    if metadata.get("logits") != expected_logits_descriptor:
        raise RuntimeError("NVIDIA logits descriptor mismatch")
    environment = metadata.get("environment")
    gpu = environment.get("gpu") if isinstance(environment, dict) else None
    if (
        not isinstance(gpu, dict)
        or any(gpu.get(name) != value for name, value in NVIDIA_GPU.items())
        or environment.get("tf32_matmul") is not False
        or environment.get("tf32_cudnn") is not False
        or environment.get("float32_matmul_precision") != "highest"
        or environment.get("deterministic_algorithms") is not True
    ):
        raise RuntimeError("NVIDIA LM-head environment contract mismatch")

    with safe_open(entries["results.safetensors"], framework="pt", device="cpu") as f:
        if set(f.keys()) != {"logits"}:
            raise RuntimeError("NVIDIA LM-head tensor set mismatch")
        embedded = f.metadata()
        if not isinstance(embedded, dict) or set(embedded) != {"provenance"}:
            raise RuntimeError("NVIDIA LM-head embedded provenance mismatch")
        provenance = json.loads(embedded["provenance"])
        expected_provenance = {
            "schema": NVIDIA_GOLDEN_SCHEMA,
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "token_id": TOKEN_ID,
            "input_final_norm_sha256": NVIDIA_FINAL_NORM_SHA256,
            "embedding_weight_sha256": TIED_WEIGHT_SHA256,
            "vocab_size": VOCAB_SIZE,
        }
        if provenance != expected_provenance:
            raise RuntimeError("NVIDIA LM-head embedded provenance mismatch")
        logits = f.get_tensor("logits").clone().contiguous()
    if tensor_descriptor(logits) != expected_logits_descriptor:
        raise RuntimeError("NVIDIA LM-head result tensor mismatch")
    expected_decision = stable_decision(logits, "full_vocab", list(range(VOCAB_SIZE)))
    golden_decision = metadata.get("decision")
    if not isinstance(golden_decision, dict):
        raise RuntimeError("NVIDIA LM-head decision metadata is missing")
    if {
        "policy": expected_decision["policy"],
        "greedy_token_id": expected_decision["greedy_token_id"],
        "greedy_logit": expected_decision["greedy_logit"],
        "maximum_tie_count": expected_decision["maximum_tie_count"],
        "maximum_tie_token_ids": expected_decision["maximum_tie_token_ids"],
        "top_k": expected_decision["top_k"],
        "top_k_entries": expected_decision["top_k_entries"],
    } != golden_decision:
        raise RuntimeError("NVIDIA LM-head decision metadata mismatch")
    if (
        metadata.get("all_logits_finite") is not True
        or metadata.get("nonfinite_count") != 0
        or not bool(torch.all(torch.isfinite(logits.float())).item())
    ):
        raise RuntimeError("NVIDIA LM-head golden contains nonfinite logits")
    return logits, {
        "directory": str(entries["__directory__"]),
        "schema": NVIDIA_GOLDEN_SCHEMA,
        "kind": NVIDIA_GOLDEN_KIND,
        "metadata_sha256": NVIDIA_GOLDEN_ACCEPTED_SHA256["metadata.json"],
        "results_sha256": NVIDIA_GOLDEN_ACCEPTED_SHA256["results.safetensors"],
        "generator": metadata["script"],
        "gpu": gpu,
        "decision": golden_decision,
    }


def deterministic_hidden() -> torch.Tensor:
    columns = torch.arange(HIDDEN_SIZE, dtype=torch.int64, device=DEVICE)
    integers = (columns * 37 + 11) % 257 - 128
    return (integers.to(torch.float32) / 512.0).to(torch.bfloat16).view(1, -1)


def load_hidden(
    safetensors_path: Path | None,
    key: str,
    model_output_dir: Path | None,
    checkpoint: dict,
) -> tuple[torch.Tensor, dict]:
    if model_output_dir is not None:
        hidden, source = validate_model_output_dir(model_output_dir, checkpoint)
    elif safetensors_path is None:
        hidden = deterministic_hidden()
        source = {
            "mode": "deterministic_fixture",
            "fixture_formula": "bf16((((column*37+11)%257)-128)/512)",
        }
    else:
        expanded = safetensors_path.expanduser()
        if expanded.is_symlink():
            raise RuntimeError("hidden safetensors path must not be a symlink")
        resolved = expanded.resolve(strict=True)
        if not resolved.is_file():
            raise RuntimeError("hidden safetensors path is not a regular file")
        with safe_open(resolved, framework="pt", device="cpu") as tensors:
            hidden = tensors.get_tensor(key).clone().contiguous()
        source = {
            "mode": "external_safetensors",
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "file_sha256": file_sha256(resolved),
            "tensor_key": key,
        }
    if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (1, HIDDEN_SIZE):
        raise RuntimeError(
            "hidden tensor must be contiguous BF16 [1,1024], got "
            f"{hidden.dtype} {tuple(hidden.shape)}"
        )
    if not hidden.is_contiguous():
        raise RuntimeError("hidden tensor must be contiguous")
    if hidden.untyped_storage().nbytes() != hidden.numel() * hidden.element_size():
        raise RuntimeError("hidden tensor does not own exact storage")
    source["tensor_sha256"] = tensor_sha256(hidden)
    return hidden, source


def load_weight_rows(embedding, vocab_ids: list[int]) -> torch.Tensor:
    weight = torch.cat(
        [embedding[token_id : token_id + 1] for token_id in vocab_ids], dim=0
    ).clone().contiguous()
    validate_weight_storage(weight, len(vocab_ids))
    return weight


def load_weight_chunk(embedding, start: int, end: int) -> torch.Tensor:
    weight = embedding[start:end].clone().contiguous()
    validate_weight_storage(weight, end - start)
    return weight


def validate_weight_storage(weight: torch.Tensor, rows: int) -> None:
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (rows, HIDDEN_SIZE):
        raise RuntimeError(
            f"materialized weight contract mismatch: {weight.dtype} {tuple(weight.shape)}"
        )
    if weight.untyped_storage().nbytes() != weight.numel() * weight.element_size():
        raise RuntimeError("materialized tied weight does not own exact storage")


def make_guarded_output(width: int) -> tuple[torch.Tensor, dict]:
    storage = torch.full(
        (GUARD_ELEMENTS + width + GUARD_ELEMENTS,),
        -97.0,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    output = storage[GUARD_ELEMENTS : GUARD_ELEMENTS + width].view(1, width)
    return output, {
        "storage": storage,
        "prefix": storage[:GUARD_ELEMENTS].clone(),
        "suffix": storage[GUARD_ELEMENTS + width :].clone(),
        "logical_elements": width,
    }


def output_guard_unchanged(guard: dict) -> bool:
    begin = GUARD_ELEMENTS
    end = begin + guard["logical_elements"]
    return bool(
        torch.equal(guard["storage"][:begin], guard["prefix"])
        and torch.equal(guard["storage"][end:], guard["suffix"])
    )


def compare_logits(
    actual: torch.Tensor,
    expected: torch.Tensor,
    vocab_ids: list[int],
) -> dict:
    actual_float = actual.to(torch.float32)
    expected_float = expected.to(torch.float32)
    error = torch.abs(actual_float - expected_float)
    finite = (
        torch.isfinite(actual_float)
        & torch.isfinite(expected_float)
        & torch.isfinite(error)
    )
    mismatch = (~finite) | (
        error
        > ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * torch.abs(expected_float)
    )
    mismatch_indices = torch.nonzero(mismatch, as_tuple=False)[:8].tolist()
    return {
        "mismatch_count": int(torch.count_nonzero(mismatch).item()),
        "nonfinite_count": int(torch.count_nonzero(~finite).item()),
        "max_abs_error": float(torch.max(torch.where(finite, error, 0.0)).item()),
        "first_mismatches": [
            {
                "vocab_id": vocab_ids[column],
                "actual": float(actual_float[row, column].item()),
                "expected": float(expected_float[row, column].item()),
                "absolute_error": float(error[row, column].item()),
            }
            for row, column in mismatch_indices
        ],
        "correct": bool(torch.count_nonzero(mismatch).item() == 0),
    }


def execute_projection(
    hidden: torch.Tensor,
    full_vocab: bool,
    golden_logits: torch.Tensor | None,
) -> dict:
    hidden_before = hidden.clone()
    all_vocab_ids = (
        list(range(VOCAB_SIZE)) if full_vocab else list(BOUNDARY_VOCAB_IDS)
    )
    output_width = len(all_vocab_ids)
    logits = torch.empty((1, output_width), dtype=torch.bfloat16, device=DEVICE)
    expected_bf16 = torch.empty_like(logits)
    weight_hasher = hashlib.sha256()
    chunk_records = []
    local_mismatch_count = 0
    local_nonfinite_count = 0
    local_max_abs_error = 0.0
    local_first_mismatches = []
    golden_mismatch_count = 0
    golden_nonfinite_count = 0
    golden_max_abs_error = 0.0
    golden_first_mismatches = []
    all_weights_unchanged = True
    all_guards_unchanged = True

    with safe_open(MODEL_FILE, framework="pt", device="cpu") as tensors:
        embedding = tensors.get_slice(EMBEDDING_WEIGHT_NAME)
        if tuple(embedding.get_shape()) != (VOCAB_SIZE, HIDDEN_SIZE):
            raise RuntimeError("embedding source shape mismatch")
        ranges = (
            [
                (start, min(start + FULL_VOCAB_CHUNK_ROWS, VOCAB_SIZE))
                for start in range(0, VOCAB_SIZE, FULL_VOCAB_CHUNK_ROWS)
            ]
            if full_vocab
            else [(0, len(BOUNDARY_VOCAB_IDS))]
        )
        for chunk_index, (start, end) in enumerate(ranges):
            vocab_ids = (
                list(range(start, end))
                if full_vocab
                else list(BOUNDARY_VOCAB_IDS)
            )
            weight = (
                load_weight_chunk(embedding, start, end)
                if full_vocab
                else load_weight_rows(embedding, vocab_ids)
            )
            weight_before = weight.clone()
            weight_bytes = tensor_bytes(weight_before)
            weight_hasher.update(weight_bytes)
            output, guard = make_guarded_output(end - start)
            dense_linear(hidden, weight, output)
            reference_float = torch.matmul(
                hidden_before.to(torch.float32), weight_before.to(torch.float32).T
            )
            reference_bf16 = reference_float.to(torch.bfloat16)
            logits[:, start:end] = output if full_vocab else output
            expected_bf16[:, start:end] = reference_bf16

            local = compare_logits(output, reference_float, vocab_ids)
            local_mismatch_count += local["mismatch_count"]
            local_nonfinite_count += local["nonfinite_count"]
            local_max_abs_error = max(local_max_abs_error, local["max_abs_error"])
            local_first_mismatches.extend(
                local["first_mismatches"][: 8 - len(local_first_mismatches)]
            )
            golden = None
            if golden_logits is not None:
                golden_expected = golden_logits[:, vocab_ids]
                golden = compare_logits(output, golden_expected, vocab_ids)
                golden_mismatch_count += golden["mismatch_count"]
                golden_nonfinite_count += golden["nonfinite_count"]
                golden_max_abs_error = max(
                    golden_max_abs_error, golden["max_abs_error"]
                )
                golden_first_mismatches.extend(
                    golden["first_mismatches"][
                        : 8 - len(golden_first_mismatches)
                    ]
                )
            weight_unchanged = bool(torch.equal(weight, weight_before))
            guard_unchanged = output_guard_unchanged(guard)
            all_weights_unchanged &= weight_unchanged
            all_guards_unchanged &= guard_unchanged
            chunk_records.append(
                {
                    "chunk": chunk_index,
                    "vocab_start": start if full_vocab else None,
                    "vocab_end_exclusive": end if full_vocab else None,
                    "vocab_ids": None if full_vocab else vocab_ids,
                    "rows": end - start,
                    "weight_bytes": len(weight_bytes),
                    "weight_sha256": hashlib.sha256(weight_bytes).hexdigest(),
                    "actual_logits_sha256": tensor_sha256(output),
                    "expected_bf16_sha256": tensor_sha256(reference_bf16),
                    "local_oracle": local,
                    "nvidia_golden": golden,
                    "weight_unchanged": weight_unchanged,
                    "output_guard_unchanged": guard_unchanged,
                }
            )

    input_unchanged = bool(torch.equal(hidden, hidden_before))
    local_correct = (
        local_mismatch_count == 0
        and local_nonfinite_count == 0
        and input_unchanged
        and all_weights_unchanged
        and all_guards_unchanged
    )
    scope = "full_vocab" if full_vocab else "bounded_checkpoint_rows"
    actual_decision = stable_decision(logits, scope, all_vocab_ids)
    local_decision = stable_decision(expected_bf16, scope, all_vocab_ids)
    local_decision_match, local_top_k_token_ids_match = decisions_match(
        actual_decision, local_decision
    )
    golden_decision = None
    golden_decision_match = None
    if golden_logits is not None:
        selected_golden = golden_logits if full_vocab else golden_logits[:, all_vocab_ids]
        golden_decision = stable_decision(selected_golden, scope, all_vocab_ids)
        golden_decision_match, golden_top_k_token_ids_match = decisions_match(
            actual_decision, golden_decision
        )
    else:
        golden_top_k_token_ids_match = None
        selected_golden = None
    golden_vector = None
    golden_top_k_overlap = None
    if selected_golden is not None and full_vocab:
        golden_vector = compare_full_logit_vectors(logits, selected_golden)
        golden_top_k_overlap = top_k_token_overlap(actual_decision, golden_decision)
        golden_correct = bool(
            golden_vector["finite"]
            and golden_vector["cosine_similarity"]
            >= MODEL_LOGITS_MIN_COSINE_SIMILARITY
            and golden_vector["relative_l2_error"]
            <= MODEL_LOGITS_MAX_RELATIVE_L2_ERROR
            and golden_decision_match
            and golden_top_k_overlap["overlap_fraction"]
            >= MODEL_LOGITS_MIN_TOP_K_TOKEN_OVERLAP
        )
    elif selected_golden is not None:
        golden_correct = golden_mismatch_count == 0 and golden_nonfinite_count == 0
    else:
        golden_correct = None
    return {
        "logits": logits,
        "expected_bf16": expected_bf16,
        "vocab_ids": all_vocab_ids,
        "weight_sha256": weight_hasher.hexdigest(),
        "chunk_records": chunk_records,
        "input_unchanged": input_unchanged,
        "all_weights_unchanged": all_weights_unchanged,
        "all_guards_unchanged": all_guards_unchanged,
        "local_oracle": {
            "comparison_scope": "all_emitted_logits",
            "values_compared": output_width,
            "mismatch_count": local_mismatch_count,
            "nonfinite_count": local_nonfinite_count,
            "max_abs_error": local_max_abs_error,
            "first_mismatches": local_first_mismatches,
            "correct": local_correct,
            "expected_bf16_sha256": tensor_sha256(expected_bf16),
            "decision": local_decision,
            "decision_match": local_decision_match,
            "top_k_token_ids_match": local_top_k_token_ids_match,
        },
        "nvidia_golden": (
            None
            if golden_logits is None
            else {
                "comparison_scope": "all_emitted_logits",
                "values_compared": output_width,
                "mismatch_count": golden_mismatch_count,
                "nonfinite_count": golden_nonfinite_count,
                "max_abs_error": golden_max_abs_error,
                "first_mismatches": golden_first_mismatches,
                "correct": golden_correct,
                "decision": golden_decision,
                "decision_match": golden_decision_match,
                "top_k_token_ids_match": golden_top_k_token_ids_match,
                "full_vector_metrics": golden_vector,
                "top_k_token_overlap": golden_top_k_overlap,
                "acceptance_policy": (
                    {
                        "minimum_cosine_similarity": MODEL_LOGITS_MIN_COSINE_SIMILARITY,
                        "maximum_relative_l2_error": MODEL_LOGITS_MAX_RELATIVE_L2_ERROR,
                        "minimum_top_k_token_overlap": MODEL_LOGITS_MIN_TOP_K_TOKEN_OVERLAP,
                        "exact_greedy_and_tie_decision_required": True,
                    }
                    if full_vocab
                    else {
                        "pointwise_zero_mismatch_required": True,
                        "exact_bounded_argmax_required": True,
                    }
                ),
            }
        ),
        "actual_decision": actual_decision,
        "correct": local_correct
        and local_decision_match
        and (golden_logits is None or (golden_correct and golden_decision_match)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the pinned Qwen3.5-0.8B tied LM head on gemsim_amd; "
            "bounded mode verifies selected rows, while --full-vocab verifies "
            "all logits and the global greedy decision"
        )
    )
    hidden_group = parser.add_mutually_exclusive_group()
    hidden_group.add_argument(
        "--hidden-safetensors",
        type=Path,
        help="safetensors file containing a BF16 [1,1024] LM-head input",
    )
    hidden_group.add_argument(
        "--model-output-dir",
        type=Path,
        help=(
            "strictly validated, execution-complete 24-layer "
            "qwen35_model_decode_correctness.py output directory; trajectory "
            "comparison may remain a separately reported diagnostic"
        ),
    )
    parser.add_argument(
        "--hidden-key",
        default="hidden",
        help="tensor key used with --hidden-safetensors (default: hidden)",
    )
    parser.add_argument(
        "--nvidia-golden-dir",
        type=Path,
        help=(
            "accepted independent full-vocabulary NVIDIA LM-head golden; "
            "requires --full-vocab; with --hidden-safetensors it is a "
            "cross-backbone diagnostic and cannot establish model-level "
            "correctness"
        ),
    )
    parser.add_argument(
        "--full-vocab",
        action="store_true",
        help=(
            "execute all 248320 rows as 4096-row owned-storage chunks; only "
            "this mode verifies the global greedy token"
        ),
    )
    args = parser.parse_args()
    if args.model_output_dir is not None and args.hidden_key != "hidden":
        parser.error("--hidden-key applies only to --hidden-safetensors")
    if args.nvidia_golden_dir is not None and not args.full_vocab:
        parser.error("--nvidia-golden-dir requires --full-vocab")

    target = triton.runtime.driver.active.get_current_target()
    if (target.backend, target.arch) != ("gemsim_amd", "gfx950"):
        raise RuntimeError(f"unexpected Triton execution target: {target}")
    if DEVICE.type != "cpu":
        raise RuntimeError(f"unexpected staging device: {DEVICE}")

    checkpoint, _ = validate_checkpoint()
    hidden, hidden_source = load_hidden(
        args.hidden_safetensors,
        args.hidden_key,
        args.model_output_dir,
        checkpoint,
    )
    golden_logits = None
    golden_provenance = None
    if args.nvidia_golden_dir is not None:
        golden_logits, golden_provenance = load_nvidia_golden(
            args.nvidia_golden_dir, checkpoint, hidden
        )
    projection = execute_projection(hidden, args.full_vocab, golden_logits)
    logits = projection.pop("logits")
    expected_bf16 = projection.pop("expected_bf16")
    vocab_ids = projection.pop("vocab_ids")

    full_weight_hash_verified = (
        projection["weight_sha256"] == TIED_WEIGHT_SHA256
        if args.full_vocab
        else None
    )
    if args.full_vocab and not full_weight_hash_verified:
        raise RuntimeError("streamed full tied-weight SHA-256 mismatch")
    output_correct = projection["correct"]
    nvidia_comparison = projection["nvidia_golden"]
    global_greedy_verified = bool(
        args.full_vocab
        and nvidia_comparison is not None
        and nvidia_comparison["correct"]
        and nvidia_comparison["decision_match"]
    )
    backbone_execution_verified = bool(
        hidden_source.get("backbone_execution_verified") is True
    )
    backbone_trajectory_verified = bool(
        hidden_source.get("backbone_trajectory_verified") is True
    )
    model_level_full_logits_verified = bool(
        args.full_vocab
        and backbone_execution_verified
        and golden_provenance is not None
        and global_greedy_verified
        and output_correct
    )
    cross_backbone_golden_diagnostic = bool(
        golden_provenance is not None and not backbone_execution_verified
    )
    payload = {
        "schema": "amdgpu-sim.triton-qwen35-tied-lm-head.v3",
        "backend": target.backend,
        "arch": target.arch,
        "model": MODEL_ID,
        "dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "kernel": "dense_linear_kernel",
        "input_shape": list(hidden.shape),
        "source_weight_shape": [VOCAB_SIZE, HIDDEN_SIZE],
        "output_shape": list(logits.shape),
        "output_bytes": logits.numel() * logits.element_size(),
        "full_vocab": args.full_vocab,
        "vocab_rows_computed": len(vocab_ids),
        "checkpoint_provenance": checkpoint,
        "hidden_source": hidden_source,
        "projection": {
            "mode": "full_vocab_streaming" if args.full_vocab else "bounded_rows",
            "claim_scope": (
                "all_248320_logits_and_global_greedy"
                if args.full_vocab
                else "selected_checkpoint_rows_only_no_global_greedy_claim"
            ),
            "projected_vocab_ids": None if args.full_vocab else vocab_ids,
            "chunk_rows": FULL_VOCAB_CHUNK_ROWS if args.full_vocab else len(vocab_ids),
            "chunk_count": len(projection["chunk_records"]),
            "chunk_weight_storage": "owned_contiguous_exact_size",
            "max_per_kernel_weight_bytes": max(
                record["weight_bytes"] for record in projection["chunk_records"]
            ),
            "aggregate_weight_staging_and_copyback_bytes": sum(
                record["weight_bytes"] for record in projection["chunk_records"]
            ),
            "driver_copyback_boundary": (
                "chunking caps per-dispatch weight D2H but current driver still "
                "copies every staged pointer, so aggregate weight D2H equals "
                "the selected tied-weight bytes"
            ),
        },
        "input_sha256": tensor_sha256(hidden),
        "streamed_weight_sha256": projection["weight_sha256"],
        "full_weight_sha256_verified": full_weight_hash_verified,
        "output_sha256": tensor_sha256(logits),
        "expected_bf16_sha256": tensor_sha256(expected_bf16),
        "input_unchanged": projection["input_unchanged"],
        "all_chunk_weights_unchanged": projection["all_weights_unchanged"],
        "all_output_guards_unchanged": projection["all_guards_unchanged"],
        "chunk_records": projection["chunk_records"],
        "local_fp32_oracle": projection["local_oracle"],
        "actual_decision": projection["actual_decision"],
        "nvidia_golden": (
            {"enabled": False, "role": "independent_oracle_not_provided"}
            if golden_provenance is None
            else {
                "enabled": True,
                "role": "external_oracle_excluded_from_target_fallback_counts",
                **golden_provenance,
                "comparison_scope": (
                    "full_vocabulary" if args.full_vocab else "selected_rows_only"
                ),
                "comparison": nvidia_comparison,
            }
        ),
        "all_values_finite": (
            projection["local_oracle"]["nonfinite_count"] == 0
            and (
                nvidia_comparison is None
                or nvidia_comparison["nonfinite_count"] == 0
            )
        ),
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "model_logits_acceptance_thresholds": {
            "profile": MODEL_LOGITS_ACCEPTANCE_PROFILE,
            "minimum_cosine_similarity": MODEL_LOGITS_MIN_COSINE_SIMILARITY,
            "maximum_relative_l2_error": MODEL_LOGITS_MAX_RELATIVE_L2_ERROR,
            "top_k": TOP_K,
            "minimum_top_k_token_overlap": MODEL_LOGITS_MIN_TOP_K_TOKEN_OVERLAP,
            "exact_greedy_and_tie_decision_required": True,
        },
        "fallback_count": 0,
        "cpu_fallback_count": 0,
        "nvidia_fallback_count": 0,
        "sampling_in_scope": False,
        "bounded_projection_correct": output_correct if not args.full_vocab else None,
        "full_vocabulary_projection_correct": output_correct if args.full_vocab else None,
        "model_backbone_execution_verified": backbone_execution_verified,
        "model_backbone_trajectory_verified": backbone_trajectory_verified,
        "independent_nvidia_logits_compared": golden_provenance is not None,
        "cross_backbone_golden_diagnostic": cross_backbone_golden_diagnostic,
        "global_greedy_verified": global_greedy_verified,
        "model_level_full_logits_verified": model_level_full_logits_verified,
        "scope": {
            "in_scope": [
                "checkpoint tied embedding projection",
                "BF16 logits with FP32 accumulation",
                (
                    "all vocabulary rows and global greedy decision"
                    if args.full_vocab
                    else "only the explicitly listed checkpoint vocabulary rows"
                ),
            ],
            "out_of_scope": [
                "probability normalization",
                "sampling",
                "generation beyond the first token",
                "strict cross-architecture equality of every intermediate layer",
                *(
                    [
                        "model-level correctness from a hidden tensor that is "
                        "not a strictly accepted backbone artifact"
                    ]
                    if cross_backbone_golden_diagnostic
                    else []
                ),
                *(
                    []
                    if args.full_vocab
                    else [
                        "uncomputed vocabulary rows",
                        "global argmax or greedy-token correctness",
                    ]
                ),
            ],
        },
        "output_correct": output_correct,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if output_correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
