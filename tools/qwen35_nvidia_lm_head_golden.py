#!/usr/bin/env python3
"""Generate the independent CUDA tied-LM-head golden for Qwen3.5-0.8B.

This tool is intentionally independent of Triton, gemsim_amd, gem5, and the
target correctness runner.  Its input is the accepted 24-layer NVIDIA backbone
golden's final_norm tensor plus the pinned checkpoint's tied embedding weight.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models/Qwen3.5-0.8B"
DEFAULT_BACKBONE_GOLDEN_DIR = (
    ROOT / "artifacts/qwen35-nvidia-golden/20260811T162220.255674Z"
)
DEFAULT_PYTHON = Path.home() / "miniforge3/envs/triton-dev/bin/python3"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_FILENAME = "model.safetensors-00001-of-00001.safetensors"
EMBEDDING_WEIGHT_NAME = "model.language_model.embed_tokens.weight"
TOKEN_ID = 248044
VOCAB_SIZE = 248320
HIDDEN_SIZE = 1024
TOP_K = 20
AT_FDCWD = -100
RENAME_NOREPLACE = 1
SCHEMA = "amdgpu-sim.qwen35-nvidia-lm-head-golden.v1"
KIND = "independent_torch_cuda_tied_lm_head_golden"
BACKBONE_SCHEMA = "amdgpu-sim.qwen35-nvidia-backbone-golden.v1"
BACKBONE_GOLDEN_SHA256 = {
    "metadata.json": "2a5d43d9c8b068ad15027916db4120782fc99b7ffa48bf225073bc09f909a9fb",
    "results.safetensors": "43a6b9f8d2cc29c728444ead69f7a0df575d634b35593bc1b2490b9ed0adfb9b",
}
BACKBONE_GENERATOR_SHA256 = (
    "bd630ee9693f5ea79c7ba7c05d4981b985dcb5b8e4cf119ee8ea8aaa20535231"
)
BACKBONE_FINAL_NORM_SHA256 = (
    "f309907965a721aee3ce35e0c300eca4cb34edb8ff203000f82f047dcd6ab994"
)
PINNED_GPU = {
    "name": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "uuid": "GPU-64aae36b-ef77-b0d4-b1c7-f7ab17a729f1",
    "compute_capability": [12, 0],
}
PINNED_ARTIFACTS = {
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


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Generate the independent NVIDIA CUDA full-vocabulary tied "
            "LM-head golden from the accepted 24-layer final_norm"
        )
    )
    value.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    value.add_argument(
        "--backbone-golden-dir",
        type=Path,
        default=DEFAULT_BACKBONE_GOLDEN_DIR,
    )
    value.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "new output directory; defaults under "
            "artifacts/qwen35-nvidia-lm-head-golden"
        ),
    )
    value.add_argument("--device", type=int, default=0)
    return value


def maybe_reexec_triton_dev() -> None:
    configured = Path(
        os.environ.get("QWEN35_NVIDIA_PYTHON", str(DEFAULT_PYTHON))
    ).resolve()
    if Path(sys.executable).resolve() == configured:
        return
    if not configured.is_file():
        raise RuntimeError(
            f"triton-dev Python is unavailable: {configured}; set "
            "QWEN35_NVIDIA_PYTHON to an equivalent CUDA PyTorch environment"
        )
    os.execv(
        configured,
        [str(configured), str(Path(__file__).resolve()), *sys.argv[1:]],
    )


if __name__ == "__main__":
    maybe_reexec_triton_dev()
    if any(argument in ("-h", "--help") for argument in sys.argv[1:]):
        parser().parse_args()


import torch
from safetensors import safe_open
from safetensors.torch import save as serialize_safetensors


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(
        order="C"
    )


def tensor_descriptor(value: torch.Tensor) -> dict:
    raw = tensor_bytes(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def validate_checkpoint(model_dir: Path) -> tuple[Path, dict]:
    observed_artifacts = {}
    for filename, expected in PINNED_ARTIFACTS.items():
        path = model_dir / filename
        observed = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        require(
            observed == expected,
            f"pinned checkpoint artifact mismatch for {filename}: {observed}",
        )
        observed_artifacts[filename] = observed

    config = load_json(model_dir / "config.json")
    text = config.get("text_config")
    require(isinstance(text, dict), "checkpoint text_config is missing")
    require(
        {
            "architectures": config.get("architectures"),
            "root_tie_word_embeddings": config.get("tie_word_embeddings"),
            "model_type": text.get("model_type"),
            "dtype": text.get("dtype"),
            "hidden_size": text.get("hidden_size"),
            "vocab_size": text.get("vocab_size"),
            "text_tie_word_embeddings": text.get("tie_word_embeddings"),
        }
        == {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "root_tie_word_embeddings": True,
            "model_type": "qwen3_5_text",
            "dtype": "bfloat16",
            "hidden_size": HIDDEN_SIZE,
            "vocab_size": VOCAB_SIZE,
            "text_tie_word_embeddings": True,
        },
        "checkpoint config contract mismatch",
    )
    index = load_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict), "checkpoint weight_map is missing")
    require(
        weight_map.get(EMBEDDING_WEIGHT_NAME) == MODEL_FILENAME,
        "tied embedding does not map to the pinned shard",
    )
    require(
        not any(name.endswith("lm_head.weight") for name in weight_map),
        "tied checkpoint unexpectedly contains an independent LM head",
    )
    manifest = load_json(model_dir / "manifest.json")
    require(manifest.get("model_id") == MODEL_ID, "checkpoint model ID mismatch")
    require(
        manifest.get("revision") == MODEL_REVISION,
        "checkpoint revision mismatch",
    )
    require(
        all(
            manifest.get("files", {}).get(name) == expected
            for name, expected in PINNED_ARTIFACTS.items()
            if name != "manifest.json"
        ),
        "checkpoint manifest file records mismatch",
    )
    return model_dir / MODEL_FILENAME, {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "artifact_hashes": observed_artifacts,
        "index_total_size": index.get("metadata", {}).get("total_size"),
        "embedding_weight_name": EMBEDDING_WEIGHT_NAME,
        "embedding_weight_file": MODEL_FILENAME,
        "embedding_weight_shape": [VOCAB_SIZE, HIDDEN_SIZE],
        "lm_head_storage": "tied_to_embed_tokens",
        "independent_lm_head_present": False,
        "config_contract_verified": True,
        "index_mapping_verified": True,
        "manifest_provenance_verified": True,
    }


def load_backbone_final_norm(golden_dir: Path) -> tuple[torch.Tensor, dict]:
    expanded = golden_dir.expanduser()
    require(not expanded.is_symlink(), "backbone golden directory is a symlink")
    resolved = expanded.resolve(strict=True)
    require(resolved.is_dir(), "backbone golden path is not a directory")
    entries = {entry.name: entry for entry in resolved.iterdir()}
    require(
        set(entries) == set(BACKBONE_GOLDEN_SHA256),
        f"backbone golden directory contents mismatch: {sorted(entries)}",
    )
    for name, expected in BACKBONE_GOLDEN_SHA256.items():
        path = entries[name]
        require(not path.is_symlink() and path.is_file(), f"invalid {name}")
        require(file_sha256(path) == expected, f"unaccepted backbone {name}")

    metadata = load_json(entries["metadata.json"])
    require(metadata.get("schema") == BACKBONE_SCHEMA, "backbone schema mismatch")
    require(
        metadata.get("kind") == "independent_torch_cuda_backbone_golden",
        "backbone kind mismatch",
    )
    require(
        metadata.get("script")
        == {
            "path": "tools/qwen35_nvidia_golden.py",
            "sha256": BACKBONE_GENERATOR_SHA256,
        },
        "backbone generator identity mismatch",
    )
    case = metadata.get("case")
    require(
        isinstance(case, dict)
        and case.get("max_layers") == 24
        and case.get("token_id") == TOKEN_ID
        and case.get("position") == 0
        and case.get("cache") == "empty_per_layer"
        and case.get("final_norm_applied") is True,
        "backbone case contract mismatch",
    )
    model = metadata.get("model")
    require(
        isinstance(model, dict)
        and model.get("id") == MODEL_ID
        and model.get("revision") == MODEL_REVISION,
        "backbone model identity mismatch",
    )
    require(
        metadata.get("results_file_sha256")
        == BACKBONE_GOLDEN_SHA256["results.safetensors"],
        "backbone results hash metadata mismatch",
    )
    with safe_open(entries["results.safetensors"], framework="pt", device="cpu") as f:
        embedded = f.metadata()
        require(
            isinstance(embedded, dict) and set(embedded) == {"provenance"},
            "backbone embedded provenance is missing",
        )
        provenance = json.loads(embedded["provenance"])
        require(
            provenance.get("schema") == BACKBONE_SCHEMA
            and provenance.get("model_id") == MODEL_ID
            and provenance.get("revision") == MODEL_REVISION
            and provenance.get("token_id") == TOKEN_ID
            and provenance.get("max_layers") == 24
            and provenance.get("position") == 0,
            "backbone embedded provenance mismatch",
        )
        final_norm = f.get_tensor("final_norm").clone().contiguous()
    descriptor = tensor_descriptor(final_norm)
    require(
        descriptor
        == {
            "shape": [1, HIDDEN_SIZE],
            "dtype": "bfloat16",
            "bytes": HIDDEN_SIZE * 2,
            "sha256": BACKBONE_FINAL_NORM_SHA256,
        },
        f"backbone final_norm mismatch: {descriptor}",
    )
    require(
        metadata.get("final_norm")
        == {key: descriptor[key] for key in ("shape", "dtype", "sha256")},
        "backbone final_norm descriptor mismatch",
    )
    return final_norm, {
        "directory": str(resolved),
        "schema": BACKBONE_SCHEMA,
        "metadata_sha256": BACKBONE_GOLDEN_SHA256["metadata.json"],
        "results_sha256": BACKBONE_GOLDEN_SHA256["results.safetensors"],
        "generator_sha256": BACKBONE_GENERATOR_SHA256,
        "token_id": TOKEN_ID,
        "layers": 24,
        "position": 0,
        "cache": "empty_per_layer",
        "final_norm": descriptor,
    }


def gpu_identity(device_index: int) -> dict:
    properties = torch.cuda.get_device_properties(device_index)
    uuid = "GPU-" + str(properties.uuid)
    smi_lines = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    matching = [line.strip() for line in smi_lines if uuid in line]
    require(len(matching) == 1, f"could not bind CUDA UUID via nvidia-smi: {uuid}")
    identity = {
        "cuda_device_index": device_index,
        "name": properties.name,
        "uuid": uuid,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "multiprocessor_count": properties.multi_processor_count,
        "pci": {
            "domain": properties.pci_domain_id,
            "bus": properties.pci_bus_id,
            "device": properties.pci_device_id,
        },
        "nvidia_smi_record": matching[0],
    }
    require(
        all(identity.get(name) == value for name, value in PINNED_GPU.items()),
        f"CUDA GPU identity is not the pinned RTX 5090: {identity}",
    )
    return identity


def decision_record(logits: torch.Tensor) -> dict:
    values = logits.view(-1).to(torch.float32).cpu()
    maximum = torch.max(values)
    tie_ids = torch.nonzero(values == maximum, as_tuple=False).view(-1).tolist()
    ordered = sorted(
        ((float(value), token_id) for token_id, value in enumerate(values.tolist())),
        key=lambda item: (-item[0], item[1]),
    )
    top = ordered[:TOP_K]
    return {
        "policy": "maximum BF16 logit; lowest token ID breaks exact ties",
        "greedy_token_id": tie_ids[0],
        "greedy_logit": float(maximum.item()),
        "maximum_tie_count": len(tie_ids),
        "maximum_tie_token_ids": tie_ids,
        "top_k": TOP_K,
        "top_k_entries": [
            {"rank": rank, "token_id": token_id, "logit": value}
            for rank, (value, token_id) in enumerate(top, start=1)
        ],
    }


def exclusive_write(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable; refusing non-atomic publish")
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


def publish_artifact(
    output_dir: Path, results_bytes: bytes, metadata_bytes: bytes
) -> None:
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        os.lstat(output_dir)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(errno.EEXIST, "output directory already exists", output_dir)

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=parent)
    )
    try:
        exclusive_write(temporary_dir / "results.safetensors", results_bytes)
        exclusive_write(temporary_dir / "metadata.json", metadata_bytes)
        fsync_directory(temporary_dir)
        rename_noreplace(temporary_dir, output_dir)
        temporary_dir = None
        fsync_directory(parent)
    finally:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir)


def main() -> int:
    args = parser().parse_args()
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    backbone_dir = args.backbone_golden_dir.expanduser()
    require(torch.cuda.is_available(), "CUDA is unavailable in triton-dev PyTorch")
    require(0 <= args.device < torch.cuda.device_count(), "invalid CUDA device")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)

    started = time.perf_counter()
    shard, checkpoint = validate_checkpoint(model_dir)
    final_norm, backbone = load_backbone_final_norm(backbone_dir)
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor(EMBEDDING_WEIGHT_NAME).clone().contiguous()
    require(
        weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (VOCAB_SIZE, HIDDEN_SIZE),
        f"embedding weight contract mismatch: {weight.dtype} {tuple(weight.shape)}",
    )
    weight_descriptor = tensor_descriptor(weight)
    load_seconds = time.perf_counter() - started

    gpu = gpu_identity(args.device)
    compute_started = time.perf_counter()
    hidden_cuda = final_norm.to(device)
    weight_cuda = weight.to(device)
    logits_cuda = torch.matmul(
        hidden_cuda.to(torch.float32), weight_cuda.to(torch.float32).T
    ).to(torch.bfloat16)
    torch.cuda.synchronize(device)
    logits = logits_cuda.cpu().contiguous()
    compute_seconds = time.perf_counter() - compute_started
    require(
        logits.dtype == torch.bfloat16
        and tuple(logits.shape) == (1, VOCAB_SIZE),
        "full-vocabulary logits contract mismatch",
    )
    nonfinite_count = int(torch.count_nonzero(~torch.isfinite(logits.float())).item())
    require(nonfinite_count == 0, "NVIDIA LM-head golden contains nonfinite logits")
    logits_descriptor = tensor_descriptor(logits)
    decision = decision_record(logits)

    embedded_provenance = {
        "schema": SCHEMA,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "token_id": TOKEN_ID,
        "input_final_norm_sha256": BACKBONE_FINAL_NORM_SHA256,
        "embedding_weight_sha256": weight_descriptor["sha256"],
        "vocab_size": VOCAB_SIZE,
    }
    results_bytes = serialize_safetensors(
        {"logits": logits},
        metadata={
            "provenance": json.dumps(
                embedded_provenance, sort_keys=True, separators=(",", ":")
            )
        },
    )
    results_sha256 = hashlib.sha256(results_bytes).hexdigest()
    script_path = Path(__file__).resolve()
    metadata = {
        "schema": SCHEMA,
        "kind": KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": checkpoint,
        "source_backbone_golden": backbone,
        "case": {
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
        },
        "input_final_norm": tensor_descriptor(final_norm),
        "tied_embedding_weight": weight_descriptor,
        "logits": logits_descriptor,
        "results_file_sha256": results_sha256,
        "all_logits_finite": True,
        "nonfinite_count": nonfinite_count,
        "decision": decision,
        "script": {
            "path": str(script_path.relative_to(ROOT)),
            "sha256": file_sha256(script_path),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": gpu,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "timing": {
            "load_seconds": load_seconds,
            "compute_seconds": compute_seconds,
        },
        "write_policy": (
            "same_parent_temp_directory_fsync_renameat2_noreplace_parent_fsync"
        ),
    }
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()

    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output_dir = ROOT / "artifacts/qwen35-nvidia-lm-head-golden" / stamp
    else:
        output_dir = args.output_dir.expanduser().absolute()
    publish_artifact(output_dir, results_bytes, metadata_bytes)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "results_sha256": results_sha256,
                "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "script_sha256": metadata["script"]["sha256"],
                "logits_sha256": logits_descriptor["sha256"],
                "greedy_token_id": decision["greedy_token_id"],
                "maximum_tie_count": decision["maximum_tie_count"],
                "total_seconds": time.perf_counter() - started,
                "gpu": gpu["name"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
