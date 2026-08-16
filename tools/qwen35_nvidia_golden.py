#!/usr/bin/env python3
"""Generate an independent CUDA golden for the Qwen3.5-0.8B backbone.

The computation uses only PyTorch CUDA and the pinned checkpoint. It does not
import Triton, the GemSim backend, gem5, or the target correctness runner.
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
DEFAULT_PYTHON = Path.home() / "miniforge3/envs/triton-dev/bin/python3"
PINNED_MODEL_ID = "Qwen/Qwen3.5-0.8B"
PINNED_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DEFAULT_TOKEN_ID = 248044
NUM_LAYERS = 24
AT_FDCWD = -100
RENAME_NOREPLACE = 1
LEGACY_SCHEMA = "amdgpu-sim.qwen35-nvidia-golden.v1"
BACKBONE_SCHEMA = "amdgpu-sim.qwen35-nvidia-backbone-golden.v1"
PREFILL_SCHEMA = "amdgpu-sim.qwen35-nvidia-prefill-golden.v1"
LEGACY_DEFAULT_RESULTS_SHA256 = (
    "a19e29ae8409063bc2cf929bb7c18cf7a69693abb3fcfaa78e85d96165a3a7a1"
)
PINNED_ARTIFACTS = {
    "config.json": {
        "bytes": 2907,
        "sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    },
    "model.safetensors.index.json": {
        "bytes": 50900,
        "sha256": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
    },
    "manifest.json": {
        "bytes": 1008,
        "sha256": "de2281cc73a1329d13245cb9658be910cf435e72c4ea0277c4f8811a24edf762",
    },
    "model.safetensors-00001-of-00001.safetensors": {
        "bytes": 1746942600,
        "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    },
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Generate the independent NVIDIA CUDA golden for pinned "
            "Qwen3.5-0.8B first-token backbone execution"
        )
    )
    value.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="pinned Hugging Face checkpoint directory",
    )
    value.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "new output directory; defaults to a timestamped directory under "
            "artifacts/qwen35-nvidia-golden"
        ),
    )
    value.add_argument("--token-id", type=int, default=DEFAULT_TOKEN_ID)
    value.add_argument(
        "--token-ids",
        type=int,
        nargs="+",
        help=(
            "bounded empty-cache prefill token IDs (2..16); when present, "
            "this selects the sequence golden path"
        ),
    )
    value.add_argument("--device", type=int, default=0, help="CUDA device index")
    value.add_argument(
        "--max-layers",
        type=int,
        choices=range(1, NUM_LAYERS + 1),
        default=1,
        metavar="1..24",
        help="execute this many real decoder layers; default preserves layer0 golden",
    )
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
from safetensors.torch import save as save_safetensors


HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3584
NUM_HEADS = 16
HEAD_DIM = 128
QKV_DIM = 3 * NUM_HEADS * HEAD_DIM
Z_DIM = NUM_HEADS * HEAD_DIM
QKVZ_DIM = QKV_DIM + Z_DIM
CONV_WIDTH = 4
CONV_STATE_WIDTH = CONV_WIDTH - 1
CONV_CACHE_LINES = 3
SELECTED_CONV_STATE = 1
FULL_NUM_HEADS = 8
FULL_NUM_KV_HEADS = 2
FULL_HEAD_DIM = 256
FULL_Q_SIZE = FULL_NUM_HEADS * FULL_HEAD_DIM
FULL_KV_SIZE = FULL_NUM_KV_HEADS * FULL_HEAD_DIM
FULL_Q_GATE_SIZE = 2 * FULL_Q_SIZE
FULL_GQA_GROUP_SIZE = FULL_NUM_HEADS // FULL_NUM_KV_HEADS
FULL_ROTARY_DIM = 64
FULL_ROPE_THETA = 10000000.0
MAX_PREFILL_TOKENS = 16
PREFILL_CACHE_SLOTS = 16
LAYER_TYPES = [
    layer_type
    for _ in range(6)
    for layer_type in (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(value: torch.Tensor) -> bytes:
    raw = value.detach().cpu().contiguous().view(torch.uint8)
    return raw.numpy().tobytes(order="C")


def tensor_record(value: torch.Tensor) -> dict:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": sha256_bytes(tensor_bytes(value)),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def validate_checkpoint(
    model_dir: Path,
) -> tuple[dict, dict, dict, Path, dict[str, dict]]:
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    manifest_path = model_dir / "manifest.json"
    artifact_records = {}
    for filename, expected in PINNED_ARTIFACTS.items():
        path = model_dir / filename
        observed = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        require(
            observed == expected,
            f"pinned checkpoint artifact mismatch for {filename}: {observed}",
        )
        artifact_records[filename] = {
            "expected": dict(expected),
            "observed": observed,
        }
    config = load_json(config_path)
    index = load_json(index_path)
    manifest = load_json(manifest_path)
    require(manifest.get("model_id") == PINNED_MODEL_ID, "model ID mismatch")
    require(manifest.get("revision") == PINNED_REVISION, "model revision mismatch")
    for path in (config_path, index_path):
        entry = manifest.get("files", {}).get(path.name, {})
        require(
            entry == PINNED_ARTIFACTS[path.name],
            f"manifest entry mismatch: {path.name}",
        )

    text = config.get("text_config")
    require(isinstance(text, dict), "checkpoint text_config is missing")
    expected_config = {
        "model_type": "qwen3_5_text",
        "dtype": "bfloat16",
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "linear_num_key_heads": NUM_HEADS,
        "linear_num_value_heads": NUM_HEADS,
        "linear_key_head_dim": HEAD_DIM,
        "linear_value_head_dim": HEAD_DIM,
        "linear_conv_kernel_dim": CONV_WIDTH,
        "num_attention_heads": FULL_NUM_HEADS,
        "num_key_value_heads": FULL_NUM_KV_HEADS,
        "head_dim": FULL_HEAD_DIM,
        "attn_output_gate": True,
        "mamba_ssm_dtype": "float32",
        "num_hidden_layers": NUM_LAYERS,
        "rms_norm_eps": 1.0e-6,
        "vocab_size": 248320,
    }
    observed = {name: text.get(name) for name in expected_config}
    require(observed == expected_config, f"text config mismatch: {observed}")
    require(text.get("layer_types") == LAYER_TYPES, "layer type schedule mismatch")
    rope = text.get("rope_parameters")
    require(isinstance(rope, dict), "rope_parameters is missing")
    require(
        {
            "rope_type": rope.get("rope_type"),
            "rope_theta": rope.get("rope_theta"),
            "partial_rotary_factor": rope.get("partial_rotary_factor"),
            "mrope_interleaved": rope.get("mrope_interleaved"),
            "mrope_section": rope.get("mrope_section"),
        }
        == {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
        "RoPE contract mismatch",
    )

    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict), "checkpoint weight_map is missing")
    shard_names = set(weight_map.values())
    require(len(shard_names) == 1, f"expected one pinned shard, got {shard_names}")
    shard = model_dir / next(iter(shard_names))
    shard_entry = manifest.get("files", {}).get(shard.name, {})
    require(
        shard.name in PINNED_ARTIFACTS
        and shard_entry == PINNED_ARTIFACTS[shard.name],
        "model shard manifest entry mismatch",
    )
    return config, index, manifest, shard, artifact_records


def load_inputs(
    model_dir: Path, token_id: int | list[int], shard: Path
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, str]]:
    token_ids = [token_id] if isinstance(token_id, int) else list(token_id)
    require(token_ids, "at least one token ID is required")
    require(
        all(0 <= value < 248320 for value in token_ids),
        f"token ID is out of range: {token_ids}",
    )
    layer = "model.language_model.layers.0"
    names = {
        "embedding": "model.language_model.embed_tokens.weight",
        "input_norm": f"{layer}.input_layernorm.weight",
        "qkv": f"{layer}.linear_attn.in_proj_qkv.weight",
        "z": f"{layer}.linear_attn.in_proj_z.weight",
        "b": f"{layer}.linear_attn.in_proj_b.weight",
        "a": f"{layer}.linear_attn.in_proj_a.weight",
        "conv": f"{layer}.linear_attn.conv1d.weight",
        "a_log": f"{layer}.linear_attn.A_log",
        "dt_bias": f"{layer}.linear_attn.dt_bias",
        "output_norm": f"{layer}.linear_attn.norm.weight",
        "gdn_out": f"{layer}.linear_attn.out_proj.weight",
        "post_attention_norm": f"{layer}.post_attention_layernorm.weight",
        "gate": f"{layer}.mlp.gate_proj.weight",
        "up": f"{layer}.mlp.up_proj.weight",
        "down": f"{layer}.mlp.down_proj.weight",
    }
    contracts = {
        "embedding": (torch.bfloat16, (248320, HIDDEN_SIZE)),
        "input_norm": (torch.bfloat16, (HIDDEN_SIZE,)),
        "qkv": (torch.bfloat16, (QKV_DIM, HIDDEN_SIZE)),
        "z": (torch.bfloat16, (Z_DIM, HIDDEN_SIZE)),
        "b": (torch.bfloat16, (NUM_HEADS, HIDDEN_SIZE)),
        "a": (torch.bfloat16, (NUM_HEADS, HIDDEN_SIZE)),
        "conv": (torch.bfloat16, (QKV_DIM, 1, CONV_WIDTH)),
        "a_log": (torch.float32, (NUM_HEADS,)),
        "dt_bias": (torch.bfloat16, (NUM_HEADS,)),
        "output_norm": (torch.float32, (HEAD_DIM,)),
        "gdn_out": (torch.bfloat16, (HIDDEN_SIZE, Z_DIM)),
        "post_attention_norm": (torch.bfloat16, (HIDDEN_SIZE,)),
        "gate": (torch.bfloat16, (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        "up": (torch.bfloat16, (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        "down": (torch.bfloat16, (HIDDEN_SIZE, INTERMEDIATE_SIZE)),
    }
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        embedding = tensors.get_slice(names["embedding"])
        require(
            tuple(embedding.get_shape()) == contracts["embedding"][1],
            "embedding shape mismatch",
        )
        hidden_rows = [
            embedding[value : value + 1].clone().contiguous()
            for value in token_ids
        ]
        hidden = torch.cat(hidden_rows, dim=0).contiguous()
        loaded = {
            key: tensors.get_tensor(name).clone().contiguous()
            for key, name in names.items()
            if key != "embedding"
        }
    for key, value in {"embedding": hidden, **loaded}.items():
        expected_dtype, expected_shape = contracts[key]
        observed_shape = (
            contracts["embedding"][1]
            if key == "embedding"
            else tuple(value.shape)
        )
        require(value.dtype == expected_dtype, f"dtype mismatch for {names[key]}")
        if key != "embedding":
            require(
                observed_shape == expected_shape,
                f"shape mismatch for {names[key]}",
            )
    require(
        tuple(hidden.shape) == (len(token_ids), HIDDEN_SIZE),
        "embedding row shape mismatch",
    )

    source_hashes = {
        names["embedding"] + f"[{value}]": tensor_record(hidden_rows[index])[
            "sha256"
        ]
        for index, value in enumerate(token_ids)
    }
    source_hashes.update(
        {names[key]: tensor_record(value)["sha256"] for key, value in loaded.items()}
    )
    weights = {
        "input_norm": loaded["input_norm"],
        "qkvz": torch.cat((loaded["qkv"], loaded["z"]), dim=0).contiguous(),
        "ba": torch.cat((loaded["b"], loaded["a"]), dim=0).contiguous(),
        "conv": loaded["conv"].squeeze(1).contiguous(),
        "a_log": loaded["a_log"],
        "dt_bias": loaded["dt_bias"],
        "output_norm": loaded["output_norm"],
        "gdn_out": loaded["gdn_out"],
        "post_attention_norm": loaded["post_attention_norm"],
        "gate_up": torch.cat((loaded["gate"], loaded["up"]), dim=0).contiguous(),
        "down": loaded["down"],
    }
    return hidden, weights, source_hashes


def layer_weight_spec(
    layer_index: int, layer_type: str
) -> tuple[dict[str, str], dict[str, tuple[torch.dtype, tuple[int, ...]]]]:
    layer = f"model.language_model.layers.{layer_index}"
    common_names = {
        "input_norm": f"{layer}.input_layernorm.weight",
        "post_attention_norm": f"{layer}.post_attention_layernorm.weight",
        "gate": f"{layer}.mlp.gate_proj.weight",
        "up": f"{layer}.mlp.up_proj.weight",
        "down": f"{layer}.mlp.down_proj.weight",
    }
    common_contracts = {
        "input_norm": (torch.bfloat16, (HIDDEN_SIZE,)),
        "post_attention_norm": (torch.bfloat16, (HIDDEN_SIZE,)),
        "gate": (torch.bfloat16, (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        "up": (torch.bfloat16, (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        "down": (torch.bfloat16, (HIDDEN_SIZE, INTERMEDIATE_SIZE)),
    }
    if layer_type == "linear_attention":
        names = {
            **common_names,
            "qkv": f"{layer}.linear_attn.in_proj_qkv.weight",
            "z": f"{layer}.linear_attn.in_proj_z.weight",
            "b": f"{layer}.linear_attn.in_proj_b.weight",
            "a": f"{layer}.linear_attn.in_proj_a.weight",
            "conv": f"{layer}.linear_attn.conv1d.weight",
            "a_log": f"{layer}.linear_attn.A_log",
            "dt_bias": f"{layer}.linear_attn.dt_bias",
            "output_norm": f"{layer}.linear_attn.norm.weight",
            "attention_out": f"{layer}.linear_attn.out_proj.weight",
        }
        contracts = {
            **common_contracts,
            "qkv": (torch.bfloat16, (QKV_DIM, HIDDEN_SIZE)),
            "z": (torch.bfloat16, (Z_DIM, HIDDEN_SIZE)),
            "b": (torch.bfloat16, (NUM_HEADS, HIDDEN_SIZE)),
            "a": (torch.bfloat16, (NUM_HEADS, HIDDEN_SIZE)),
            "conv": (torch.bfloat16, (QKV_DIM, 1, CONV_WIDTH)),
            "a_log": (torch.float32, (NUM_HEADS,)),
            "dt_bias": (torch.bfloat16, (NUM_HEADS,)),
            "output_norm": (torch.float32, (HEAD_DIM,)),
            "attention_out": (torch.bfloat16, (HIDDEN_SIZE, Z_DIM)),
        }
    elif layer_type == "full_attention":
        names = {
            **common_names,
            "q_gate": f"{layer}.self_attn.q_proj.weight",
            "k": f"{layer}.self_attn.k_proj.weight",
            "v": f"{layer}.self_attn.v_proj.weight",
            "q_norm": f"{layer}.self_attn.q_norm.weight",
            "k_norm": f"{layer}.self_attn.k_norm.weight",
            "attention_out": f"{layer}.self_attn.o_proj.weight",
        }
        contracts = {
            **common_contracts,
            "q_gate": (torch.bfloat16, (FULL_Q_GATE_SIZE, HIDDEN_SIZE)),
            "k": (torch.bfloat16, (FULL_KV_SIZE, HIDDEN_SIZE)),
            "v": (torch.bfloat16, (FULL_KV_SIZE, HIDDEN_SIZE)),
            "q_norm": (torch.bfloat16, (FULL_HEAD_DIM,)),
            "k_norm": (torch.bfloat16, (FULL_HEAD_DIM,)),
            "attention_out": (torch.bfloat16, (HIDDEN_SIZE, FULL_Q_SIZE)),
        }
    else:
        raise RuntimeError(f"unsupported layer type: {layer_type}")
    return names, contracts


def load_additional_backbone_weights(
    index: dict,
    shard: Path,
    max_layers: int,
) -> tuple[
    dict[int, dict[str, torch.Tensor]],
    torch.Tensor | None,
    dict[str, str],
    dict[str, dict],
]:
    require(max_layers > 1, "additional backbone loader requires max_layers > 1")
    weight_map = index["weight_map"]
    selected_hashes: dict[str, str] = {}
    execution_weights: dict[str, dict] = {}
    layers: dict[int, dict[str, torch.Tensor]] = {}
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        for layer_index in range(1, max_layers):
            layer_type = LAYER_TYPES[layer_index]
            names, contracts = layer_weight_spec(layer_index, layer_type)
            for name in names.values():
                require(
                    weight_map.get(name) == shard.name,
                    f"checkpoint index mapping mismatch: {name}",
                )
            loaded = {
                key: tensors.get_tensor(name).clone().contiguous()
                for key, name in names.items()
            }
            for key, value in loaded.items():
                dtype, shape = contracts[key]
                require(
                    value.dtype == dtype and tuple(value.shape) == shape,
                    f"checkpoint tensor contract mismatch: {names[key]}",
                )
                selected_hashes[names[key]] = tensor_record(value)["sha256"]
            common = {
                "input_norm": loaded["input_norm"],
                "post_attention_norm": loaded["post_attention_norm"],
                "gate_up": torch.cat(
                    (loaded["gate"], loaded["up"]), dim=0
                ).contiguous(),
                "down": loaded["down"],
            }
            if layer_type == "linear_attention":
                weights = {
                    **common,
                    "qkvz": torch.cat(
                        (loaded["qkv"], loaded["z"]), dim=0
                    ).contiguous(),
                    "ba": torch.cat(
                        (loaded["b"], loaded["a"]), dim=0
                    ).contiguous(),
                    "conv": loaded["conv"].squeeze(1).contiguous(),
                    "a_log": loaded["a_log"],
                    "dt_bias": loaded["dt_bias"],
                    "output_norm": loaded["output_norm"],
                    "attention_out": loaded["attention_out"],
                }
            else:
                weights = {
                    **common,
                    "q_gate": loaded["q_gate"],
                    "k": loaded["k"],
                    "v": loaded["v"],
                    "q_norm": loaded["q_norm"],
                    "k_norm": loaded["k_norm"],
                    "attention_out": loaded["attention_out"],
                }
            layers[layer_index] = weights
            execution_weights.update(
                {
                    f"layers.{layer_index}.{name}": tensor_record(value)
                    for name, value in weights.items()
                }
            )

        final_norm = None
        if max_layers == NUM_LAYERS:
            final_norm_name = "model.language_model.norm.weight"
            require(
                weight_map.get(final_norm_name) == shard.name,
                "checkpoint index mapping mismatch: final norm",
            )
            final_norm = tensors.get_tensor(final_norm_name).clone().contiguous()
    if final_norm is not None:
        require(
            final_norm.dtype == torch.bfloat16
            and tuple(final_norm.shape) == (HIDDEN_SIZE,),
            "final norm tensor contract mismatch",
        )
        selected_hashes[final_norm_name] = tensor_record(final_norm)["sha256"]
        execution_weights["final_norm"] = tensor_record(final_norm)
    return layers, final_norm, selected_hashes, execution_weights


def bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x.float(), weight.float().T).to(torch.bfloat16)


def timed_stage(
    name: str,
    timings: dict[str, float],
    function,
):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    timings[name] = (time.perf_counter() - start) * 1000.0
    return value


def compute_golden(
    hidden_cpu: torch.Tensor,
    weights_cpu: dict[str, torch.Tensor],
    device: torch.device,
    epsilon: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    hidden = hidden_cpu.to(device)
    weights = {name: value.to(device) for name, value in weights_cpu.items()}
    timings: dict[str, float] = {}
    results: dict[str, torch.Tensor] = {"hidden_input": hidden}

    def input_norm_formula():
        x = hidden.float()
        variance = torch.mean(x * x, dim=-1, keepdim=True)
        return (
            x
            * torch.rsqrt(variance + epsilon)
            * (1.0 + weights["input_norm"].float())
        ).to(torch.bfloat16)

    normalized = timed_stage("input_rms_norm", timings, input_norm_formula)
    results["input_rms_norm"] = normalized
    qkvz = timed_stage(
        "qkvz_projection",
        timings,
        lambda: bf16_linear(normalized, weights["qkvz"]),
    )
    results["qkvz_projection"] = qkvz
    ba = timed_stage(
        "ba_projection", timings, lambda: bf16_linear(normalized, weights["ba"])
    )
    results["ba_projection"] = ba

    conv_state = torch.zeros(
        (CONV_CACHE_LINES, QKV_DIM, CONV_STATE_WIDTH),
        dtype=torch.bfloat16,
        device=device,
    )
    mixed_qkv_input = qkvz[:, :QKV_DIM]

    def conv_formula():
        state = conv_state[SELECTED_CONV_STATE]
        x = mixed_qkv_input[0].float()
        weight = weights["conv"].float()
        accumulator = (
            state[:, 0].float() * weight[:, 0]
            + state[:, 1].float() * weight[:, 1]
            + state[:, 2].float() * weight[:, 2]
            + x * weight[:, 3]
        )
        output = (accumulator * torch.sigmoid(accumulator)).to(torch.bfloat16)
        next_state = conv_state.clone()
        next_state[SELECTED_CONV_STATE] = torch.stack(
            (state[:, 1], state[:, 2], mixed_qkv_input[0]), dim=1
        )
        return output.view(1, QKV_DIM), next_state

    mixed_qkv, conv_state = timed_stage("gdn_conv", timings, conv_formula)
    results["gdn_conv_output"] = mixed_qkv
    results["conv_state"] = conv_state

    b = ba[:, :NUM_HEADS]
    a = ba[:, NUM_HEADS:]
    recurrent_state = torch.zeros(
        (NUM_HEADS, HEAD_DIM, HEAD_DIM), dtype=torch.float32, device=device
    )

    def recurrent_formula():
        mixed = mixed_qkv.float().view(-1)
        query = mixed[: NUM_HEADS * HEAD_DIM].view(NUM_HEADS, HEAD_DIM)
        key = mixed[NUM_HEADS * HEAD_DIM : 2 * NUM_HEADS * HEAD_DIM].view(
            NUM_HEADS, HEAD_DIM
        )
        value = mixed[2 * NUM_HEADS * HEAD_DIM :].view(NUM_HEADS, HEAD_DIM)
        query = query * torch.rsqrt(
            torch.sum(query * query, dim=-1, keepdim=True) + epsilon
        )
        key = key * torch.rsqrt(
            torch.sum(key * key, dim=-1, keepdim=True) + epsilon
        )
        query = query * (HEAD_DIM**-0.5)
        softplus_input = a.float().view(NUM_HEADS) + weights["dt_bias"].float()
        softplus = torch.where(
            softplus_input <= 20.0,
            torch.log1p(torch.exp(softplus_input)),
            softplus_input,
        )
        decay = torch.exp(-torch.exp(weights["a_log"]) * softplus)
        beta = torch.sigmoid(b.float().view(NUM_HEADS)).to(torch.bfloat16).float()
        state = recurrent_state * decay.view(NUM_HEADS, 1, 1)
        prediction = torch.sum(state * key[:, None, :], dim=-1)
        delta = (value - prediction) * beta[:, None]
        state = state + delta[:, :, None] * key[:, None, :]
        output = torch.sum(state * query[:, None, :], dim=-1)
        return output.to(torch.bfloat16), state

    recurrent_output, recurrent_state = timed_stage(
        "gdn_recurrent", timings, recurrent_formula
    )
    recurrent_output = recurrent_output.view(1, NUM_HEADS, HEAD_DIM)
    results["gdn_recurrent_output"] = recurrent_output
    results["recurrent_state"] = recurrent_state

    z = qkvz[:, QKV_DIM:].view(1, NUM_HEADS, HEAD_DIM)

    def output_norm_gate_formula():
        x = recurrent_output.float()
        variance = torch.mean(x * x, dim=-1, keepdim=True)
        normalized_x = x * torch.rsqrt(variance + epsilon)
        z_float = z.float()
        return (
            normalized_x
            * weights["output_norm"].float()
            * (z_float * torch.sigmoid(z_float))
        ).to(torch.bfloat16)

    norm_gate = timed_stage(
        "output_rms_norm_gate", timings, output_norm_gate_formula
    )
    results["output_rms_norm_gate"] = norm_gate
    attention_output = timed_stage(
        "gdn_out_projection",
        timings,
        lambda: bf16_linear(norm_gate.view(1, Z_DIM), weights["gdn_out"]),
    )
    results["gdn_out_projection"] = attention_output

    def post_attention_formula():
        summed = attention_output.float() + hidden.float()
        residual = summed.to(torch.bfloat16)
        variance = torch.mean(summed * summed, dim=-1, keepdim=True)
        output = (
            summed
            * torch.rsqrt(variance + epsilon)
            * (1.0 + weights["post_attention_norm"].float())
        ).to(torch.bfloat16)
        return output, residual

    post_norm, residual = timed_stage(
        "post_attention_fused_rms_norm", timings, post_attention_formula
    )
    results["post_attention_rms_norm"] = post_norm
    results["post_attention_residual"] = residual
    gate_up = timed_stage(
        "mlp_gate_up", timings, lambda: bf16_linear(post_norm, weights["gate_up"])
    )
    results["mlp_gate_up"] = gate_up

    def silu_and_mul_formula():
        gate = gate_up[:, :INTERMEDIATE_SIZE].float()
        up = gate_up[:, INTERMEDIATE_SIZE:].float()
        return (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)

    activated = timed_stage("mlp_silu_and_mul", timings, silu_and_mul_formula)
    results["mlp_silu_and_mul"] = activated
    final_hidden = timed_stage(
        "mlp_down", timings, lambda: bf16_linear(activated, weights["down"])
    )
    results["mlp_down"] = final_hidden
    results["final_hidden"] = final_hidden.clone()
    results["final_residual"] = residual.clone()
    return (
        {name: value.detach().cpu().contiguous() for name, value in results.items()},
        timings,
    )


def gemma_rms_formula(
    value: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    value_float = value.float()
    variance = torch.mean(value_float * value_float, dim=-1, keepdim=True)
    return (
        value_float
        * torch.rsqrt(variance + epsilon)
        * (1.0 + weight.float())
    ).to(torch.bfloat16)


def fused_gemma_rms_formula(
    value: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed = value.float() + residual.float()
    next_residual = summed.to(torch.bfloat16)
    variance = torch.mean(summed * summed, dim=-1, keepdim=True)
    normalized = (
        summed
        * torch.rsqrt(variance + epsilon)
        * (1.0 + weight.float())
    ).to(torch.bfloat16)
    return normalized, next_residual


def gdn_attention_formula(
    normalized: torch.Tensor,
    weights: dict[str, torch.Tensor],
    epsilon: float,
) -> torch.Tensor:
    qkvz = bf16_linear(normalized, weights["qkvz"])
    ba = bf16_linear(normalized, weights["ba"])
    mixed_qkv_input = qkvz[:, :QKV_DIM]
    conv_state = torch.zeros(
        (CONV_CACHE_LINES, QKV_DIM, CONV_STATE_WIDTH),
        dtype=torch.bfloat16,
        device=normalized.device,
    )
    state = conv_state[SELECTED_CONV_STATE]
    value = mixed_qkv_input[0].float()
    conv_weight = weights["conv"].float()
    accumulator = (
        state[:, 0].float() * conv_weight[:, 0]
        + state[:, 1].float() * conv_weight[:, 1]
        + state[:, 2].float() * conv_weight[:, 2]
        + value * conv_weight[:, 3]
    )
    mixed_qkv = (accumulator * torch.sigmoid(accumulator)).to(torch.bfloat16)

    mixed = mixed_qkv.float().view(-1)
    query = mixed[: NUM_HEADS * HEAD_DIM].view(NUM_HEADS, HEAD_DIM)
    key = mixed[NUM_HEADS * HEAD_DIM : 2 * NUM_HEADS * HEAD_DIM].view(
        NUM_HEADS, HEAD_DIM
    )
    recurrent_value = mixed[2 * NUM_HEADS * HEAD_DIM :].view(
        NUM_HEADS, HEAD_DIM
    )
    query = query * torch.rsqrt(
        torch.sum(query * query, dim=-1, keepdim=True) + epsilon
    )
    key = key * torch.rsqrt(
        torch.sum(key * key, dim=-1, keepdim=True) + epsilon
    )
    query = query * (HEAD_DIM**-0.5)
    b = ba[:, :NUM_HEADS]
    a = ba[:, NUM_HEADS:]
    softplus_input = a.float().view(NUM_HEADS) + weights["dt_bias"].float()
    softplus = torch.where(
        softplus_input <= 20.0,
        torch.log1p(torch.exp(softplus_input)),
        softplus_input,
    )
    decay = torch.exp(-torch.exp(weights["a_log"]) * softplus)
    beta = torch.sigmoid(b.float().view(NUM_HEADS)).to(torch.bfloat16).float()
    recurrent_state = torch.zeros(
        (NUM_HEADS, HEAD_DIM, HEAD_DIM),
        dtype=torch.float32,
        device=normalized.device,
    )
    recurrent_state = recurrent_state * decay.view(NUM_HEADS, 1, 1)
    prediction = torch.sum(recurrent_state * key[:, None, :], dim=-1)
    delta = (recurrent_value - prediction) * beta[:, None]
    recurrent_state = recurrent_state + delta[:, :, None] * key[:, None, :]
    recurrent_output = torch.sum(
        recurrent_state * query[:, None, :], dim=-1
    ).to(torch.bfloat16)
    recurrent_output = recurrent_output.view(1, NUM_HEADS, HEAD_DIM)

    recurrent_float = recurrent_output.float()
    variance = torch.mean(
        recurrent_float * recurrent_float, dim=-1, keepdim=True
    )
    normalized_recurrent = recurrent_float * torch.rsqrt(variance + epsilon)
    z = qkvz[:, QKV_DIM:].view(1, NUM_HEADS, HEAD_DIM).float()
    gated = (
        normalized_recurrent
        * weights["output_norm"].float()
        * (z * torch.sigmoid(z))
    ).to(torch.bfloat16)
    return bf16_linear(
        gated.view(1, Z_DIM), weights["attention_out"]
    )


def full_attention_position0_formula(
    normalized: torch.Tensor,
    weights: dict[str, torch.Tensor],
    epsilon: float,
) -> torch.Tensor:
    q_gate_projection = bf16_linear(normalized, weights["q_gate"])
    q_gate = q_gate_projection.view(1, FULL_NUM_HEADS, 2 * FULL_HEAD_DIM)
    query, gate = torch.chunk(q_gate, 2, dim=-1)
    key = bf16_linear(normalized, weights["k"]).view(
        1, FULL_NUM_KV_HEADS, FULL_HEAD_DIM
    )
    value = bf16_linear(normalized, weights["v"]).view(
        1, FULL_NUM_KV_HEADS, FULL_HEAD_DIM
    )
    query = gemma_rms_formula(query, weights["q_norm"], epsilon)
    key = gemma_rms_formula(key, weights["k_norm"], epsilon)

    # Position zero makes every RoPE angle zero. With an empty cache, causal
    # attention has one key, so softmax is exactly one for every query head.
    key = key.repeat_interleave(FULL_GQA_GROUP_SIZE, dim=1)
    value = value.repeat_interleave(FULL_GQA_GROUP_SIZE, dim=1)
    scores = torch.sum(query.float() * key.float(), dim=-1, keepdim=True)
    scores = scores * (FULL_HEAD_DIM**-0.5)
    probabilities = torch.softmax(scores, dim=-1)
    require(
        torch.equal(probabilities, torch.ones_like(probabilities)),
        "position0 one-element attention softmax did not degenerate to one",
    )
    attention = (probabilities * value.float()).to(torch.bfloat16)
    attention = attention * torch.sigmoid(gate)
    return bf16_linear(
        attention.view(1, FULL_Q_SIZE), weights["attention_out"]
    )


def gdn_attention_sequence_formula(
    normalized: torch.Tensor,
    weights: dict[str, torch.Tensor],
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = normalized.shape[0]
    require(
        2 <= tokens <= MAX_PREFILL_TOKENS,
        "GDN prefill sequence length is outside the bounded contract",
    )
    qkvz = bf16_linear(normalized, weights["qkvz"])
    ba = bf16_linear(normalized, weights["ba"])
    mixed_qkv_input = qkvz[:, :QKV_DIM]
    conv_state = torch.zeros(
        (CONV_CACHE_LINES, QKV_DIM, CONV_STATE_WIDTH),
        dtype=torch.bfloat16,
        device=normalized.device,
    )
    selected = conv_state[SELECTED_CONV_STATE]
    conv_weight = weights["conv"].float()
    mixed_rows: list[torch.Tensor] = []
    for token in range(tokens):
        current = mixed_qkv_input[token].float()
        accumulator = (
            selected[:, 0].float() * conv_weight[:, 0]
            + selected[:, 1].float() * conv_weight[:, 1]
            + selected[:, 2].float() * conv_weight[:, 2]
            + current * conv_weight[:, 3]
        )
        mixed_rows.append(
            (accumulator * torch.sigmoid(accumulator)).to(torch.bfloat16)
        )
        selected = torch.stack(
            (selected[:, 1], selected[:, 2], mixed_qkv_input[token]), dim=1
        )
    conv_state = conv_state.clone()
    conv_state[SELECTED_CONV_STATE] = selected
    mixed_qkv = torch.stack(mixed_rows, dim=0)

    recurrent_state = torch.zeros(
        (NUM_HEADS, HEAD_DIM, HEAD_DIM),
        dtype=torch.float32,
        device=normalized.device,
    )
    recurrent_rows: list[torch.Tensor] = []
    for token in range(tokens):
        mixed = mixed_qkv[token].float().view(-1)
        query = mixed[: NUM_HEADS * HEAD_DIM].view(NUM_HEADS, HEAD_DIM)
        key = mixed[
            NUM_HEADS * HEAD_DIM : 2 * NUM_HEADS * HEAD_DIM
        ].view(NUM_HEADS, HEAD_DIM)
        value = mixed[2 * NUM_HEADS * HEAD_DIM :].view(NUM_HEADS, HEAD_DIM)
        query = query * torch.rsqrt(
            torch.sum(query * query, dim=-1, keepdim=True) + epsilon
        )
        key = key * torch.rsqrt(
            torch.sum(key * key, dim=-1, keepdim=True) + epsilon
        )
        query = query * (HEAD_DIM**-0.5)
        b = ba[token, :NUM_HEADS].float()
        a = ba[token, NUM_HEADS:].float()
        softplus_input = a + weights["dt_bias"].float()
        softplus = torch.where(
            softplus_input <= 20.0,
            torch.log1p(torch.exp(softplus_input)),
            softplus_input,
        )
        decay = torch.exp(-torch.exp(weights["a_log"]) * softplus)
        recurrent_state = recurrent_state * decay.view(NUM_HEADS, 1, 1)
        prediction = torch.sum(recurrent_state * key[:, None, :], dim=-1)
        beta = torch.sigmoid(b).to(torch.bfloat16).float()
        delta = (value - prediction) * beta[:, None]
        recurrent_state = (
            recurrent_state + delta[:, :, None] * key[:, None, :]
        )
        recurrent_rows.append(
            torch.sum(recurrent_state * query[:, None, :], dim=-1).to(
                torch.bfloat16
            )
        )
    recurrent_output = torch.stack(recurrent_rows, dim=0)
    recurrent_float = recurrent_output.float()
    variance = torch.mean(
        recurrent_float * recurrent_float, dim=-1, keepdim=True
    )
    normalized_recurrent = recurrent_float * torch.rsqrt(variance + epsilon)
    z = qkvz[:, QKV_DIM:].view(tokens, NUM_HEADS, HEAD_DIM).float()
    gated = (
        normalized_recurrent
        * weights["output_norm"].float()
        * (z * torch.sigmoid(z))
    ).to(torch.bfloat16)
    attention_weight = weights.get("attention_out", weights.get("gdn_out"))
    require(attention_weight is not None, "GDN output projection is missing")
    attention = bf16_linear(gated.view(tokens, Z_DIM), attention_weight)
    return attention, conv_state, recurrent_state


def apply_text_rope_sequence(
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = query.shape[0]
    inv_freq = 1.0 / (
        FULL_ROPE_THETA
        ** (
            torch.arange(0, FULL_ROTARY_DIM, 2, dtype=torch.float32)
            / FULL_ROTARY_DIM
        )
    )
    positions = torch.arange(tokens, dtype=torch.float32)
    frequencies = torch.einsum("i,j->ij", positions, inv_freq)
    cosine = frequencies.cos().to(torch.bfloat16).to(query.device).float()
    sine = frequencies.sin().to(torch.bfloat16).to(query.device).float()

    def rotate(value: torch.Tensor) -> torch.Tensor:
        output = value.clone()
        first = value[..., : FULL_ROTARY_DIM // 2].float()
        second = value[..., FULL_ROTARY_DIM // 2 : FULL_ROTARY_DIM].float()
        cos = cosine[:, None, :]
        sin = sine[:, None, :]
        output[..., : FULL_ROTARY_DIM // 2] = (
            first * cos - second * sin
        ).to(torch.bfloat16)
        output[..., FULL_ROTARY_DIM // 2 : FULL_ROTARY_DIM] = (
            second * cos + first * sin
        ).to(torch.bfloat16)
        return output

    return rotate(query), rotate(key)


def full_attention_sequence_formula(
    normalized: torch.Tensor,
    weights: dict[str, torch.Tensor],
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = normalized.shape[0]
    require(
        2 <= tokens <= MAX_PREFILL_TOKENS,
        "full-attention prefill length is outside the bounded contract",
    )
    q_gate_projection = bf16_linear(normalized, weights["q_gate"])
    q_gate = q_gate_projection.view(
        tokens, FULL_NUM_HEADS, 2 * FULL_HEAD_DIM
    )
    query, gate = torch.chunk(q_gate, 2, dim=-1)
    key = bf16_linear(normalized, weights["k"]).view(
        tokens, FULL_NUM_KV_HEADS, FULL_HEAD_DIM
    )
    value = bf16_linear(normalized, weights["v"]).view(
        tokens, FULL_NUM_KV_HEADS, FULL_HEAD_DIM
    )
    query = gemma_rms_formula(query, weights["q_norm"], epsilon)
    key = gemma_rms_formula(key, weights["k_norm"], epsilon)
    query, key = apply_text_rope_sequence(query, key)

    kv_cache = torch.zeros(
        (1, PREFILL_CACHE_SLOTS, FULL_NUM_KV_HEADS, 2 * FULL_HEAD_DIM),
        dtype=torch.bfloat16,
        device=normalized.device,
    )
    kv_cache[0, :tokens, :, :FULL_HEAD_DIM] = key
    kv_cache[0, :tokens, :, FULL_HEAD_DIM:] = value
    attention_rows: list[torch.Tensor] = []
    for token in range(tokens):
        per_head: list[torch.Tensor] = []
        for query_head in range(FULL_NUM_HEADS):
            kv_head = query_head // FULL_GQA_GROUP_SIZE
            cached_key = key[: token + 1, kv_head].float()
            cached_value = value[: token + 1, kv_head].float()
            scores = torch.sum(
                query[token, query_head].float()[None, :] * cached_key,
                dim=-1,
            ) * (FULL_HEAD_DIM**-0.5)
            probabilities = torch.softmax(scores, dim=0)
            per_head.append(
                torch.sum(probabilities[:, None] * cached_value, dim=0).to(
                    torch.bfloat16
                )
            )
        attention_rows.append(torch.stack(per_head, dim=0))
    attention = torch.stack(attention_rows, dim=0)
    sigmoid_gate = torch.sigmoid(gate.float()).to(torch.bfloat16).float()
    gated = (attention.float() * sigmoid_gate).to(torch.bfloat16)
    projected = bf16_linear(
        gated.view(tokens, FULL_Q_SIZE), weights["attention_out"]
    )
    return projected, kv_cache


def compute_backbone_layer_sequence(
    layer_index: int,
    hidden_cpu: torch.Tensor,
    residual_cpu: torch.Tensor | None,
    weights_cpu: dict[str, torch.Tensor],
    device: torch.device,
    epsilon: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, float],
]:
    hidden = hidden_cpu.to(device)
    weights = {name: value.to(device) for name, value in weights_cpu.items()}
    timings: dict[str, float] = {}
    if residual_cpu is None:
        normalized = timed_stage(
            "input_rms_norm",
            timings,
            lambda: gemma_rms_formula(hidden, weights["input_norm"], epsilon),
        )
        residual = hidden
    else:
        residual = residual_cpu.to(device)
        normalized, residual = timed_stage(
            "input_rms_norm",
            timings,
            lambda: fused_gemma_rms_formula(
                hidden, residual, weights["input_norm"], epsilon
            ),
        )
    if LAYER_TYPES[layer_index] == "linear_attention":
        attention_output, conv_state, recurrent_state = timed_stage(
            "gdn_attention",
            timings,
            lambda: gdn_attention_sequence_formula(normalized, weights, epsilon),
        )
        states = {
            "conv_state": conv_state,
            "recurrent_state": recurrent_state,
        }
    else:
        attention_output, kv_cache = timed_stage(
            "full_attention",
            timings,
            lambda: full_attention_sequence_formula(normalized, weights, epsilon),
        )
        states = {"kv_cache": kv_cache}
    post_norm, residual = timed_stage(
        "post_attention_fused_rms_norm",
        timings,
        lambda: fused_gemma_rms_formula(
            attention_output,
            residual,
            weights["post_attention_norm"],
            epsilon,
        ),
    )
    gate_up = timed_stage(
        "mlp_gate_up",
        timings,
        lambda: bf16_linear(post_norm, weights["gate_up"]),
    )

    def silu_and_mul() -> torch.Tensor:
        gate = gate_up[:, :INTERMEDIATE_SIZE].float()
        up = gate_up[:, INTERMEDIATE_SIZE:].float()
        return (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)

    activated = timed_stage("mlp_silu_and_mul", timings, silu_and_mul)
    hidden = timed_stage(
        "mlp_down",
        timings,
        lambda: bf16_linear(activated, weights["down"]),
    )
    return (
        hidden.detach().cpu().contiguous(),
        residual.detach().cpu().contiguous(),
        {
            name: value.detach().cpu().contiguous()
            for name, value in states.items()
        },
        timings,
    )


def compute_backbone_layer(
    layer_index: int,
    hidden_cpu: torch.Tensor,
    residual_cpu: torch.Tensor,
    weights_cpu: dict[str, torch.Tensor],
    device: torch.device,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    hidden = hidden_cpu.to(device)
    residual = residual_cpu.to(device)
    weights = {name: value.to(device) for name, value in weights_cpu.items()}
    timings: dict[str, float] = {}
    normalized, residual = timed_stage(
        "input_rms_norm",
        timings,
        lambda: fused_gemma_rms_formula(
            hidden, residual, weights["input_norm"], epsilon
        ),
    )
    if LAYER_TYPES[layer_index] == "linear_attention":
        attention_output = timed_stage(
            "gdn_attention",
            timings,
            lambda: gdn_attention_formula(normalized, weights, epsilon),
        )
    else:
        attention_output = timed_stage(
            "full_attention_position0",
            timings,
            lambda: full_attention_position0_formula(
                normalized, weights, epsilon
            ),
        )
    post_norm, residual = timed_stage(
        "post_attention_fused_rms_norm",
        timings,
        lambda: fused_gemma_rms_formula(
            attention_output,
            residual,
            weights["post_attention_norm"],
            epsilon,
        ),
    )
    gate_up = timed_stage(
        "mlp_gate_up",
        timings,
        lambda: bf16_linear(post_norm, weights["gate_up"]),
    )

    def silu_and_mul():
        gate = gate_up[:, :INTERMEDIATE_SIZE].float()
        up = gate_up[:, INTERMEDIATE_SIZE:].float()
        return (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)

    activated = timed_stage("mlp_silu_and_mul", timings, silu_and_mul)
    hidden = timed_stage(
        "mlp_down",
        timings,
        lambda: bf16_linear(activated, weights["down"]),
    )
    return (
        hidden.detach().cpu().contiguous(),
        residual.detach().cpu().contiguous(),
        timings,
    )


def compute_prefix_final_norm(
    hidden_cpu: torch.Tensor,
    residual_cpu: torch.Tensor,
    weight_cpu: torch.Tensor,
    device: torch.device,
    epsilon: float,
) -> tuple[torch.Tensor, float]:
    hidden = hidden_cpu.to(device)
    residual = residual_cpu.to(device)
    weight = weight_cpu.to(device)
    timings: dict[str, float] = {}
    normalized, _ = timed_stage(
        "final_norm",
        timings,
        lambda: fused_gemma_rms_formula(hidden, residual, weight, epsilon),
    )
    return normalized.detach().cpu().contiguous(), timings["final_norm"]


def source_identity() -> dict:
    paths = [
        ROOT / "projects/vllm/vllm/model_executor/layers/layernorm.py",
        ROOT / "projects/vllm/vllm/model_executor/models/qwen3_5.py",
        ROOT / "projects/vllm/vllm/model_executor/models/qwen3_next.py",
        ROOT
        / "projects/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    ]
    head = subprocess.check_output(
        ["git", "-C", str(ROOT / "projects/vllm"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return {
        "vllm_git_head": head,
        "formula_source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in paths
        },
    }


def gpu_identity(device_index: int) -> dict:
    properties = torch.cuda.get_device_properties(device_index)
    smi = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    uuid = "GPU-" + str(properties.uuid)
    matching = [line.strip() for line in smi if uuid in line]
    require(
        len(matching) == 1,
        f"could not bind CUDA device UUID via nvidia-smi: {uuid}",
    )
    return {
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


def exclusive_write(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
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
        raise OSError(
            errno.ENOSYS,
            "renameat2 is unavailable; refusing non-atomic publish",
        )
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
        raise FileExistsError(
            errno.EEXIST, "output directory already exists", output_dir
        )
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
    sequence_mode = args.token_ids is not None
    token_ids = list(args.token_ids) if sequence_mode else [args.token_id]
    if sequence_mode:
        require(
            2 <= len(token_ids) <= MAX_PREFILL_TOKENS,
            f"prefill token count must be in [2,{MAX_PREFILL_TOKENS}]",
        )
    model_dir = args.model_dir.resolve()
    require(torch.cuda.is_available(), "CUDA is unavailable in triton-dev PyTorch")
    require(
        0 <= args.device < torch.cuda.device_count(),
        "CUDA device index is invalid",
    )
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)

    total_start = time.perf_counter()
    load_start = time.perf_counter()
    config, index, manifest, shard, artifact_records = validate_checkpoint(
        model_dir
    )
    hidden, weights, source_weight_hashes = load_inputs(
        model_dir, token_ids if sequence_mode else args.token_id, shard
    )
    additional_layers: dict[int, dict[str, torch.Tensor]] = {}
    final_norm_weight = None
    additional_execution_weights: dict[str, dict] = {}
    if args.max_layers > 1:
        (
            additional_layers,
            final_norm_weight,
            additional_source_hashes,
            additional_execution_weights,
        ) = load_additional_backbone_weights(index, shard, args.max_layers)
        source_weight_hashes.update(additional_source_hashes)
    load_seconds = time.perf_counter() - load_start
    compute_start = time.perf_counter()
    epsilon = config["text_config"]["rms_norm_eps"]
    if sequence_mode:
        results: dict[str, torch.Tensor] = {"hidden_input": hidden.clone()}
        stage_ms: dict[str, float] = {}
        returned_hidden = hidden
        returned_residual: torch.Tensor | None = None
        for layer_index in range(args.max_layers):
            layer_weights = (
                weights if layer_index == 0 else additional_layers[layer_index]
            )
            (
                returned_hidden,
                returned_residual,
                layer_states,
                layer_stage_ms,
            ) = compute_backbone_layer_sequence(
                layer_index,
                returned_hidden,
                returned_residual,
                layer_weights,
                device,
                epsilon,
            )
            results[f"layers.{layer_index}.returned_hidden"] = returned_hidden
            results[f"layers.{layer_index}.returned_residual"] = returned_residual
            for state_name, state_value in layer_states.items():
                results[f"layers.{layer_index}.{state_name}"] = state_value
            stage_ms.update(
                {
                    f"layers.{layer_index}.{name}": milliseconds
                    for name, milliseconds in layer_stage_ms.items()
                }
            )
        require(returned_residual is not None, "prefill residual was not produced")
        results["final_hidden"] = returned_hidden.clone()
        results["final_residual"] = returned_residual.clone()
        if args.max_layers == NUM_LAYERS:
            require(
                final_norm_weight is not None,
                "final norm weight was not loaded",
            )
            final_norm, final_norm_ms = compute_prefix_final_norm(
                returned_hidden,
                returned_residual,
                final_norm_weight,
                device,
                epsilon,
            )
            results["final_norm"] = final_norm
            stage_ms["final_norm"] = final_norm_ms
        else:
            require(
                final_norm_weight is None,
                "prefix execution unexpectedly loaded final norm weight",
            )
        execution_weight_hashes = {
            f"layers.0.{name}": tensor_record(value)
            for name, value in weights.items()
        }
        execution_weight_hashes.update(additional_execution_weights)
    else:
        layer0_results, layer0_stage_ms = compute_golden(
            hidden, weights, device, epsilon
        )
        if args.max_layers == 1:
            results = layer0_results
            stage_ms = layer0_stage_ms
            execution_weight_hashes = {
                name: tensor_record(value) for name, value in weights.items()
            }
        else:
            results = dict(layer0_results)
            results["layers.0.returned_hidden"] = layer0_results[
                "final_hidden"
            ].clone()
            results["layers.0.returned_residual"] = layer0_results[
                "final_residual"
            ].clone()
            stage_ms = {
                f"layers.0.{name}": milliseconds
                for name, milliseconds in layer0_stage_ms.items()
            }
            returned_hidden = layer0_results["final_hidden"]
            returned_residual = layer0_results["final_residual"]
            for layer_index in range(1, args.max_layers):
                returned_hidden, returned_residual, layer_stage_ms = (
                    compute_backbone_layer(
                        layer_index,
                        returned_hidden,
                        returned_residual,
                        additional_layers[layer_index],
                        device,
                        epsilon,
                    )
                )
                results[f"layers.{layer_index}.returned_hidden"] = returned_hidden
                results[f"layers.{layer_index}.returned_residual"] = returned_residual
                stage_ms.update(
                    {
                        f"layers.{layer_index}.{name}": milliseconds
                        for name, milliseconds in layer_stage_ms.items()
                    }
                )
            results["final_hidden"] = returned_hidden.clone()
            results["final_residual"] = returned_residual.clone()
            if args.max_layers == NUM_LAYERS:
                require(
                    final_norm_weight is not None,
                    "final norm weight was not loaded",
                )
                final_norm, final_norm_ms = compute_prefix_final_norm(
                    returned_hidden,
                    returned_residual,
                    final_norm_weight,
                    device,
                    epsilon,
                )
                results["final_norm"] = final_norm
                stage_ms["final_norm"] = final_norm_ms
            else:
                require(
                    final_norm_weight is None,
                    "prefix execution unexpectedly loaded final norm weight",
                )
            execution_weight_hashes = {
                f"layers.0.{name}": tensor_record(value)
                for name, value in weights.items()
            }
            execution_weight_hashes.update(additional_execution_weights)
    compute_seconds = time.perf_counter() - compute_start

    result_records = {name: tensor_record(value) for name, value in results.items()}
    result_nonfinite_counts = {
        name: int(torch.count_nonzero(~torch.isfinite(value.float())).item())
        for name, value in results.items()
    }
    require(
        all(count == 0 for count in result_nonfinite_counts.values()),
        f"nonfinite result tensors: {result_nonfinite_counts}",
    )
    if sequence_mode:
        result_provenance = {
            "schema": PREFILL_SCHEMA,
            "model_id": PINNED_MODEL_ID,
            "revision": PINNED_REVISION,
            "token_ids": token_ids,
            "positions": list(range(len(token_ids))),
            "max_layers": args.max_layers,
            "layer_types": LAYER_TYPES[: args.max_layers],
            "cache": "empty_per_layer",
        }
    elif args.max_layers == 1:
        result_provenance = {
            "schema": LEGACY_SCHEMA,
            "model_id": PINNED_MODEL_ID,
            "revision": PINNED_REVISION,
            "layer": 0,
            "token_id": args.token_id,
        }
    else:
        result_provenance = {
            "schema": BACKBONE_SCHEMA,
            "model_id": PINNED_MODEL_ID,
            "revision": PINNED_REVISION,
            "token_id": args.token_id,
            "max_layers": args.max_layers,
            "layer_types": LAYER_TYPES[: args.max_layers],
            "position": 0,
        }
    result_bytes = save_safetensors(
        results,
        metadata={
            "provenance": json.dumps(
                result_provenance, sort_keys=True, separators=(",", ":")
            )
        },
    )
    results_sha256 = sha256_bytes(result_bytes)
    if (
        not sequence_mode
        and args.max_layers == 1
        and args.token_id == DEFAULT_TOKEN_ID
    ):
        require(
            results_sha256 == LEGACY_DEFAULT_RESULTS_SHA256,
            "default layer0 results changed from the pinned byte-identical golden",
        )
    script_path = Path(__file__).resolve()
    shard_manifest = manifest["files"][shard.name]
    model_record = {
        "id": PINNED_MODEL_ID,
        "revision": PINNED_REVISION,
        "directory": str(model_dir),
        "config_sha256": sha256_file(model_dir / "config.json"),
        "index_sha256": sha256_file(
            model_dir / "model.safetensors.index.json"
        ),
        "shard": shard.name,
        "shard_bytes": shard.stat().st_size,
        "shard_manifest_sha256": shard_manifest["sha256"],
        "index_total_size": index.get("metadata", {}).get("total_size"),
        "pinned_artifacts": artifact_records,
    }
    input_record = {
        "hidden": tensor_record(hidden),
        "initial_conv_state": tensor_record(
            torch.zeros(
                (CONV_CACHE_LINES, QKV_DIM, CONV_STATE_WIDTH),
                dtype=torch.bfloat16,
            )
        ),
        "initial_recurrent_state": tensor_record(
            torch.zeros((NUM_HEADS, HEAD_DIM, HEAD_DIM), dtype=torch.float32)
        ),
    }
    formula_record = source_identity()
    script_record = {
        "path": str(script_path.relative_to(ROOT)),
        "sha256": sha256_file(script_path),
    }
    environment_record = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu_identity(args.device),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    timing_record = {
        "checkpoint_load_seconds": load_seconds,
        "compute_seconds": compute_seconds,
        "stage_milliseconds": stage_ms,
    }
    legacy_stage_order = [
            "input_rms_norm",
            "qkvz_projection",
            "ba_projection",
            "gdn_conv_output",
            "conv_state",
            "gdn_recurrent_output",
            "recurrent_state",
            "output_rms_norm_gate",
            "gdn_out_projection",
            "post_attention_rms_norm",
            "post_attention_residual",
            "mlp_gate_up",
            "mlp_silu_and_mul",
            "mlp_down",
    ]
    if sequence_mode:
        layer_results = []
        for layer_index in range(args.max_layers):
            state_names = (
                ["conv_state", "recurrent_state"]
                if LAYER_TYPES[layer_index] == "linear_attention"
                else ["kv_cache"]
            )
            layer_results.append(
                {
                    "index": layer_index,
                    "type": LAYER_TYPES[layer_index],
                    "returned_hidden": result_records[
                        f"layers.{layer_index}.returned_hidden"
                    ],
                    "returned_residual": result_records[
                        f"layers.{layer_index}.returned_residual"
                    ],
                    "states": {
                        name: result_records[f"layers.{layer_index}.{name}"]
                        for name in state_names
                    },
                }
            )
        metadata = {
            "schema": PREFILL_SCHEMA,
            "kind": "independent_torch_cuda_empty_cache_prefill_golden",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_record,
            "case": {
                "scope": "bounded_empty_cache_decoder_backbone_prefill",
                "max_layers": args.max_layers,
                "layer_types": LAYER_TYPES[: args.max_layers],
                "linear_attention_layers": sum(
                    layer_type == "linear_attention"
                    for layer_type in LAYER_TYPES[: args.max_layers]
                ),
                "full_attention_layers": sum(
                    layer_type == "full_attention"
                    for layer_type in LAYER_TYPES[: args.max_layers]
                ),
                "token_ids": token_ids,
                "tokens": len(token_ids),
                "positions": list(range(len(token_ids))),
                "tensor_layout": "NHD",
                "cache": "empty_per_layer",
                "state_initialization": "zero_then_sequential_token_updates",
                "selected_conv_state": SELECTED_CONV_STATE,
                "linear_weight_layout": (
                    "checkpoint [out,in], matmul uses weight.T"
                ),
                "outer_rms_weight": "1 + checkpoint raw weight",
                "gdn_output_norm_weight": "checkpoint weight directly",
                "returned_pair": ["mlp_down", "post_attention_residual"],
                "final_norm_scope": (
                    "full_backbone"
                    if args.max_layers == NUM_LAYERS
                    else "not_applied_for_backbone_prefix"
                ),
                "final_norm_applied": args.max_layers == NUM_LAYERS,
                "excludes": ["final vocabulary projection", "logits"],
            },
            "full_attention_prefill_boundary": {
                "q_projection_layout": "per-head [q, gate]",
                "query_heads": FULL_NUM_HEADS,
                "key_value_heads": FULL_NUM_KV_HEADS,
                "head_dim": FULL_HEAD_DIM,
                "rotary_dim": FULL_ROTARY_DIM,
                "rope_theta": FULL_ROPE_THETA,
                "rope_style": "NeoX split-half text positions",
                "gqa_group_size": FULL_GQA_GROUP_SIZE,
                "causal_key_lengths": list(range(1, len(token_ids) + 1)),
                "output_gate_boundary": (
                    "sigmoid(gate_bf16) rounded to BF16 before multiply"
                ),
                "kv_cache_shape": [
                    1,
                    PREFILL_CACHE_SLOTS,
                    FULL_NUM_KV_HEADS,
                    2 * FULL_HEAD_DIM,
                ],
            },
            "input": input_record,
            "selected_checkpoint_tensor_sha256": source_weight_hashes,
            "execution_weights": execution_weight_hashes,
            "layer_results": layer_results,
            **(
                {"final_norm": result_records["final_norm"]}
                if args.max_layers == NUM_LAYERS
                else {}
            ),
            "results": result_records,
            "results_file_sha256": results_sha256,
            "all_results_finite": True,
            "result_nonfinite_counts": result_nonfinite_counts,
            "result_order": list(results),
            "formula_source": formula_record,
            "script": script_record,
            "environment": environment_record,
            "timing": timing_record,
        }
    elif args.max_layers == 1:
        metadata = {
            "schema": LEGACY_SCHEMA,
            "kind": "independent_torch_cuda_golden",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_record,
            "case": {
                "layer": 0,
                "layer_type": "linear_attention",
                "token_id": args.token_id,
                "tokens": 1,
                "cache": "empty",
                "state_initialization": "zero_first_token",
                "selected_conv_state": SELECTED_CONV_STATE,
                "linear_weight_layout": (
                    "checkpoint [out,in], matmul uses weight.T"
                ),
                "qkvz_order": ["q", "k", "v", "z"],
                "ba_order": ["b", "a"],
                "outer_rms_weight": "1 + checkpoint raw weight",
                "gdn_output_norm_weight": "checkpoint weight directly",
                "returned_pair": ["mlp_down", "post_attention_residual"],
            },
            "input": input_record,
            "selected_checkpoint_tensor_sha256": source_weight_hashes,
            "execution_weights": execution_weight_hashes,
            "results": result_records,
            "results_file_sha256": results_sha256,
            "all_results_finite": True,
            "result_nonfinite_counts": result_nonfinite_counts,
            "stage_order": legacy_stage_order,
            "formula_source": formula_record,
            "script": script_record,
            "environment": environment_record,
            "timing": timing_record,
        }
    else:
        layer_results = [
            {
                "index": layer_index,
                "type": LAYER_TYPES[layer_index],
                "returned_hidden": result_records[
                    f"layers.{layer_index}.returned_hidden"
                ],
                "returned_residual": result_records[
                    f"layers.{layer_index}.returned_residual"
                ],
            }
            for layer_index in range(args.max_layers)
        ]
        metadata = {
            "schema": BACKBONE_SCHEMA,
            "kind": "independent_torch_cuda_backbone_golden",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_record,
            "case": {
                "scope": "decoder_backbone_prefix",
                "max_layers": args.max_layers,
                "layer_types": LAYER_TYPES[: args.max_layers],
                "linear_attention_layers": sum(
                    layer_type == "linear_attention"
                    for layer_type in LAYER_TYPES[: args.max_layers]
                ),
                "full_attention_layers": sum(
                    layer_type == "full_attention"
                    for layer_type in LAYER_TYPES[: args.max_layers]
                ),
                "token_id": args.token_id,
                "tokens": 1,
                "position": 0,
                "tensor_layout": "NHD",
                "cache": "empty_per_layer",
                "state_initialization": "zero_first_token",
                "selected_conv_state": SELECTED_CONV_STATE,
                "linear_weight_layout": (
                    "checkpoint [out,in], matmul uses weight.T"
                ),
                "outer_rms_weight": "1 + checkpoint raw weight",
                "gdn_output_norm_weight": "checkpoint weight directly",
                "returned_pair": ["mlp_down", "post_attention_residual"],
                "final_norm_scope": (
                    "full_backbone"
                    if args.max_layers == NUM_LAYERS
                    else "not_applied_for_backbone_prefix"
                ),
                "final_norm_applied": args.max_layers == NUM_LAYERS,
                "excludes": ["final vocabulary projection", "logits"],
            },
            "full_attention_position0_boundary": {
                "q_projection_layout": "per-head [q, gate]",
                "query_heads": FULL_NUM_HEADS,
                "key_value_heads": FULL_NUM_KV_HEADS,
                "head_dim": FULL_HEAD_DIM,
                "gqa_group_size": FULL_GQA_GROUP_SIZE,
                "rope": "identity because every position0 angle is zero",
                "causal_key_length": 1,
                "softmax_domain_size": 1,
                "softmax_probability": 1.0,
            },
            "input": input_record,
            "selected_checkpoint_tensor_sha256": source_weight_hashes,
            "execution_weights": execution_weight_hashes,
            "layer_results": layer_results,
            **(
                {"final_norm": result_records["final_norm"]}
                if args.max_layers == NUM_LAYERS
                else {}
            ),
            "results": result_records,
            "results_file_sha256": results_sha256,
            "all_results_finite": True,
            "result_nonfinite_counts": result_nonfinite_counts,
            "result_order": list(results),
            "formula_source": formula_record,
            "script": script_record,
            "environment": environment_record,
            "timing": timing_record,
        }
    metadata_bytes = (
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    metadata_sha256 = sha256_bytes(metadata_bytes)

    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output_dir = ROOT / "artifacts/qwen35-nvidia-golden" / stamp
    else:
        output_dir = args.output_dir.resolve()
    write_start = time.perf_counter()
    publish_artifact(output_dir, result_bytes, metadata_bytes)
    write_seconds = time.perf_counter() - write_start
    total_seconds = time.perf_counter() - total_start
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "results_sha256": metadata["results_file_sha256"],
                "metadata_sha256": metadata_sha256,
                "load_seconds": load_seconds,
                "compute_seconds": compute_seconds,
                "write_seconds": write_seconds,
                "total_seconds": total_seconds,
                "gpu": metadata["environment"]["gpu"]["name"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
