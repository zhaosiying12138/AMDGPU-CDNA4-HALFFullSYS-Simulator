#!/usr/bin/env python3

import runpy
from pathlib import Path


HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE / "_gemsim_bootstrap.py"))["bootstrap"](
    __file__, "qwen35-layer0-decode"
)

import argparse
import hashlib
import json

import torch
import triton

from safetensors import safe_open


DEVICE = triton.runtime.driver.active.get_active_torch_device()
ROOT = HERE.parents[1]
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3584
GATE_UP_SIZE = 2 * INTERMEDIATE_SIZE
NUM_HEADS = 16
HEAD_DIM = 128
QKV_DIM = 3 * NUM_HEADS * HEAD_DIM
Z_DIM = NUM_HEADS * HEAD_DIM
QKVZ_DIM = QKV_DIM + Z_DIM
BA_DIM = 2 * NUM_HEADS
CONV_WIDTH = 4
CONV_STATE_WIDTH = CONV_WIDTH - 1
CONV_CACHE_LINES = 3
SELECTED_CONV_STATE = 1
EPSILON = 1.0e-6
GUARD_ELEMENTS = 256
VOCAB_SIZE = 248320
CHECKPOINT_TOTAL_SIZE = 1746882752
MODEL_ID = "Qwen/Qwen3.5-0.8B"
PINNED_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
NVIDIA_GOLDEN_SCHEMA = "amdgpu-sim.qwen35-nvidia-golden.v1"
NVIDIA_GOLDEN_KIND = "independent_torch_cuda_golden"
NVIDIA_GOLDEN_STAGE_ORDER = [
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
NVIDIA_GOLDEN_CONTRACT = {
    "hidden_input": (torch.bfloat16, (1, HIDDEN_SIZE)),
    "input_rms_norm": (torch.bfloat16, (1, HIDDEN_SIZE)),
    "qkvz_projection": (torch.bfloat16, (1, QKVZ_DIM)),
    "ba_projection": (torch.bfloat16, (1, BA_DIM)),
    "gdn_conv_output": (torch.bfloat16, (1, QKV_DIM)),
    "conv_state": (
        torch.bfloat16,
        (CONV_CACHE_LINES, QKV_DIM, CONV_STATE_WIDTH),
    ),
    "gdn_recurrent_output": (
        torch.bfloat16,
        (1, NUM_HEADS, HEAD_DIM),
    ),
    "recurrent_state": (
        torch.float32,
        (NUM_HEADS, HEAD_DIM, HEAD_DIM),
    ),
    "output_rms_norm_gate": (
        torch.bfloat16,
        (1, NUM_HEADS, HEAD_DIM),
    ),
    "gdn_out_projection": (torch.bfloat16, (1, HIDDEN_SIZE)),
    "post_attention_rms_norm": (torch.bfloat16, (1, HIDDEN_SIZE)),
    "post_attention_residual": (torch.bfloat16, (1, HIDDEN_SIZE)),
    "mlp_gate_up": (torch.bfloat16, (1, GATE_UP_SIZE)),
    "mlp_silu_and_mul": (torch.bfloat16, (1, INTERMEDIATE_SIZE)),
    "mlp_down": (torch.bfloat16, (1, HIDDEN_SIZE)),
    "final_hidden": (torch.bfloat16, (1, HIDDEN_SIZE)),
    "final_residual": (torch.bfloat16, (1, HIDDEN_SIZE)),
}
NVIDIA_GOLDEN_TOLERANCES = {
    "input_rms_norm": (0.015625, 0.02),
    "qkvz_projection": (0.03125, 0.03),
    "ba_projection": (0.03125, 0.03),
    "gdn_conv_output": (0.05, 0.01),
    # The BF16 cache stores the pre-convolution qkv projection verbatim.
    "conv_state": (0.03125, 0.03),
    "gdn_recurrent_output": (0.015625, 0.02),
    # This state is FP32, but it is computed from independently rounded BF16
    # q/k/v projections. Keep an FP32-scale absolute bound without requiring
    # cross-platform bitwise identity.
    "recurrent_state": (1.0e-4, 0.03),
    "output_rms_norm_gate": (0.01, 0.01),
    "gdn_out_projection": (0.03125, 0.03),
    "post_attention_rms_norm": (0.01, 0.01),
    "post_attention_residual": (0.03125, 0.03),
    "mlp_gate_up": (0.03125, 0.03),
    "mlp_silu_and_mul": (0.015625, 0.02),
    "mlp_down": (0.03125, 0.03),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_inputs(token_id: int) -> tuple[torch.Tensor, dict, dict]:
    if token_id < 0 or token_id >= VOCAB_SIZE:
        raise ValueError(f"token ID is out of range: {token_id}")
    root = ROOT
    model_dir = root / "models/Qwen3.5-0.8B"
    config_file = model_dir / "config.json"
    index_file = model_dir / "model.safetensors.index.json"
    prefix = "model.language_model.layers.0"
    embedding_name = "model.language_model.embed_tokens.weight"
    contracts = {
        embedding_name: (torch.bfloat16, (VOCAB_SIZE, HIDDEN_SIZE)),
        f"{prefix}.input_layernorm.weight": (torch.bfloat16, (HIDDEN_SIZE,)),
        f"{prefix}.linear_attn.in_proj_qkv.weight": (torch.bfloat16, (QKV_DIM, HIDDEN_SIZE)),
        f"{prefix}.linear_attn.in_proj_z.weight": (torch.bfloat16, (Z_DIM, HIDDEN_SIZE)),
        f"{prefix}.linear_attn.in_proj_b.weight": (torch.bfloat16, (NUM_HEADS, HIDDEN_SIZE)),
        f"{prefix}.linear_attn.in_proj_a.weight": (torch.bfloat16, (NUM_HEADS, HIDDEN_SIZE)),
        f"{prefix}.linear_attn.conv1d.weight": (torch.bfloat16, (QKV_DIM, 1, CONV_WIDTH)),
        f"{prefix}.linear_attn.A_log": (torch.float32, (NUM_HEADS,)),
        f"{prefix}.linear_attn.dt_bias": (torch.bfloat16, (NUM_HEADS,)),
        f"{prefix}.linear_attn.norm.weight": (torch.float32, (HEAD_DIM,)),
        f"{prefix}.linear_attn.out_proj.weight": (torch.bfloat16, (HIDDEN_SIZE, Z_DIM)),
        f"{prefix}.post_attention_layernorm.weight": (torch.bfloat16, (HIDDEN_SIZE,)),
        f"{prefix}.mlp.gate_proj.weight": (torch.bfloat16, (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        f"{prefix}.mlp.up_proj.weight": (torch.bfloat16, (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        f"{prefix}.mlp.down_proj.weight": (torch.bfloat16, (HIDDEN_SIZE, INTERMEDIATE_SIZE)),
    }

    config = json.loads(config_file.read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise RuntimeError("checkpoint config has no text_config object")
    observed_config_contract = {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "text_model_type": text_config.get("model_type"),
        "dtype": text_config.get("dtype"),
        "hidden_act": text_config.get("hidden_act"),
        "hidden_size": text_config.get("hidden_size"),
        "intermediate_size": text_config.get("intermediate_size"),
        "layer_types": text_config.get("layer_types"),
        "linear_conv_kernel_dim": text_config.get("linear_conv_kernel_dim"),
        "linear_key_head_dim": text_config.get("linear_key_head_dim"),
        "linear_num_key_heads": text_config.get("linear_num_key_heads"),
        "linear_num_value_heads": text_config.get("linear_num_value_heads"),
        "linear_value_head_dim": text_config.get("linear_value_head_dim"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "rms_norm_eps": text_config.get("rms_norm_eps"),
        "text_tie_word_embeddings": text_config.get("tie_word_embeddings"),
        "vocab_size": text_config.get("vocab_size"),
        "mamba_ssm_dtype": text_config.get("mamba_ssm_dtype"),
        "attn_output_gate": text_config.get("attn_output_gate"),
        "use_cache": text_config.get("use_cache"),
    }
    expected_config_contract = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": True,
        "text_model_type": "qwen3_5_text",
        "dtype": "bfloat16",
        "hidden_act": "silu",
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "layer_types": [
            layer_type
            for _ in range(6)
            for layer_type in (
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            )
        ],
        "linear_conv_kernel_dim": CONV_WIDTH,
        "linear_key_head_dim": HEAD_DIM,
        "linear_num_key_heads": NUM_HEADS,
        "linear_num_value_heads": NUM_HEADS,
        "linear_value_head_dim": HEAD_DIM,
        "num_hidden_layers": 24,
        "rms_norm_eps": EPSILON,
        "text_tie_word_embeddings": True,
        "vocab_size": VOCAB_SIZE,
        "mamba_ssm_dtype": "float32",
        "attn_output_gate": True,
        "use_cache": True,
    }
    if observed_config_contract != expected_config_contract:
        raise RuntimeError(
            "checkpoint config contract mismatch: "
            f"{observed_config_contract!r}"
        )

    index = json.loads(index_file.read_text(encoding="utf-8"))
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if (
        not isinstance(metadata, dict)
        or metadata.get("total_size") != CHECKPOINT_TOTAL_SIZE
        or not isinstance(weight_map, dict)
    ):
        raise RuntimeError("checkpoint index metadata contract mismatch")
    expected_layer_names = set(contracts) - {embedding_name}
    observed_layer_names = {
        name for name in weight_map if name.startswith(f"{prefix}.")
    }
    if observed_layer_names != expected_layer_names:
        raise RuntimeError(
            "checkpoint layer-0 index contract mismatch: "
            f"{sorted(observed_layer_names)!r}"
        )
    selected_mapping = {name: weight_map.get(name) for name in contracts}
    if any(shard is None for shard in selected_mapping.values()):
        raise RuntimeError(
            f"checkpoint selected tensor is absent from index: {selected_mapping!r}"
        )
    expected_shard = "model.safetensors-00001-of-00001.safetensors"
    if set(selected_mapping.values()) != {expected_shard}:
        raise RuntimeError(
            f"checkpoint selected tensor shard mismatch: {selected_mapping!r}"
        )
    if "lm_head.weight" in weight_map:
        raise RuntimeError(
            "tied checkpoint unexpectedly contains a separate lm_head.weight"
        )
    model_file = model_dir / expected_shard

    with safe_open(model_file, framework="pt", device="cpu") as tensors:
        embedding = tensors.get_slice(embedding_name)
        embedding_shape = tuple(embedding.get_shape())
        hidden = embedding[token_id : token_id + 1].clone().contiguous()
        loaded = {
            name: tensors.get_tensor(name).clone().contiguous()
            for name in contracts
            if name != embedding_name
        }
    observed = {
        embedding_name: (
            hidden.dtype,
            embedding_shape,
        )
    }
    observed.update(
        {
            name: (tensor.dtype, tuple(tensor.shape))
            for name, tensor in loaded.items()
        }
    )
    if observed != contracts:
        raise RuntimeError(f"checkpoint tensor contract mismatch: {observed!r}")
    if tuple(hidden.shape) != (1, HIDDEN_SIZE):
        raise RuntimeError(f"checkpoint embedding row shape mismatch: {hidden.shape}")
    for name, tensor in {
        embedding_name: hidden,
        **loaded,
    }.items():
        expected_bytes = tensor.numel() * tensor.element_size()
        if tensor.untyped_storage().nbytes() != expected_bytes:
            raise RuntimeError(
                f"checkpoint tensor does not own exact storage: {name}"
            )
    qkv = loaded[f"{prefix}.linear_attn.in_proj_qkv.weight"]
    z = loaded[f"{prefix}.linear_attn.in_proj_z.weight"]
    b = loaded[f"{prefix}.linear_attn.in_proj_b.weight"]
    a = loaded[f"{prefix}.linear_attn.in_proj_a.weight"]
    gate = loaded[f"{prefix}.mlp.gate_proj.weight"]
    up = loaded[f"{prefix}.mlp.up_proj.weight"]
    weights = {
        "input_norm": loaded[f"{prefix}.input_layernorm.weight"],
        "qkvz": torch.cat((qkv, z), dim=0),
        "ba": torch.cat((b, a), dim=0),
        "conv": loaded[f"{prefix}.linear_attn.conv1d.weight"].view(QKV_DIM, CONV_WIDTH),
        "a_log": loaded[f"{prefix}.linear_attn.A_log"],
        "dt_bias": loaded[f"{prefix}.linear_attn.dt_bias"],
        "output_norm": loaded[f"{prefix}.linear_attn.norm.weight"],
        "gdn_out": loaded[f"{prefix}.linear_attn.out_proj.weight"],
        "post_attention_norm": loaded[f"{prefix}.post_attention_layernorm.weight"],
        "gate_up": torch.cat((gate, up), dim=0),
        "down": loaded[f"{prefix}.mlp.down_proj.weight"],
    }
    provenance = {
        "mode": "checkpoint",
        "config_file": str(config_file.relative_to(root)),
        "config_sha256": file_sha256(config_file),
        "index_file": str(index_file.relative_to(root)),
        "index_sha256": file_sha256(index_file),
        "model_file": str(model_file.relative_to(root)),
        "selected_shard": expected_shard,
        "selected_shard_bytes": model_file.stat().st_size,
        "selected_tensor_shards": selected_mapping,
        "checkpoint_index_total_size": metadata["total_size"],
        "token_id": token_id,
        "embedding_weight_name": embedding_name,
        "embedding_row_sha256": tensor_sha256(hidden),
        "layer": 0,
        "selected_layer_tensor_count": len(loaded),
        "selected_layer_tensor_bytes": sum(
            tensor.numel() * tensor.element_size()
            for tensor in loaded.values()
        ),
        "selected_layer_tensor_sha256": {
            name: tensor_sha256(tensor)
            for name, tensor in sorted(loaded.items())
        },
        "tensor_digest_scope": "contiguous logical tensor bytes",
        "loaded_weight_bytes": sum(
            tensor.numel() * tensor.element_size()
            for tensor in weights.values()
        ),
    }
    return hidden, weights, provenance


def load_runner(filename: str) -> dict:
    return runpy.run_path(str(HERE / filename))


plain_rms_norm = load_runner("rms_norm_correctness.py")[
    "plain_gemma_rms_norm"
]
dense_linear = load_runner("qwen35_dense_linear_correctness.py")[
    "dense_linear"
]
conv_module = load_runner("qwen35_gdn_conv_decode_correctness.py")
gdn_conv_decode = conv_module["gdn_conv_decode"]
recurrent_module = load_runner("qwen35_gdn_recurrent_decode_correctness.py")
gdn_recurrent_decode = recurrent_module["gdn_recurrent_decode"]
recurrent_reference = recurrent_module["reference_transition"]
gdn_output_norm_gate = load_runner(
    "qwen35_gdn_output_norm_gate_correctness.py"
)["gdn_output_norm_gate"]
fused_residual_rms_norm = load_runner(
    "qwen35_fused_residual_rms_norm_correctness.py"
)["fused_residual_gemma_rms_norm"]
silu_and_mul = load_runner("silu_and_mul_correctness.py")["silu_and_mul"]


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value
    if value.dtype == torch.bfloat16:
        raw = value.view(torch.uint16)
    return hashlib.sha256(
        raw.contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def guarded_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, sentinel: float
) -> tuple[torch.Tensor, dict]:
    elements = 1
    for extent in shape:
        elements *= extent
    storage = torch.full(
        (GUARD_ELEMENTS + elements + GUARD_ELEMENTS,),
        sentinel,
        dtype=dtype,
        device=DEVICE,
    )
    begin = GUARD_ELEMENTS
    end = begin + elements
    value = storage[begin:end].view(shape)
    guard = {
        "storage": storage,
        "elements": elements,
        "prefix": storage[:begin].clone(),
        "suffix": storage[end:].clone(),
    }
    return value, guard


def guard_unchanged(guard: dict) -> bool:
    storage = guard["storage"]
    end = GUARD_ELEMENTS + guard["elements"]
    return bool(
        torch.equal(storage[:GUARD_ELEMENTS], guard["prefix"])
        and torch.equal(storage[end:], guard["suffix"])
    )


def compare_stage(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    *,
    require_bitwise: bool = False,
) -> dict:
    actual_float = actual.to(torch.float32)
    expected_float = expected.to(torch.float32)
    error = torch.abs(actual_float - expected_float)
    tolerance = atol + rtol * torch.abs(expected_float)
    finite = (
        torch.isfinite(actual_float)
        & torch.isfinite(expected_float)
        & torch.isfinite(error)
    )
    mismatch = (~finite) | (error > tolerance)
    finite_error = torch.where(finite, error, torch.zeros_like(error))
    expected_l2 = float(torch.linalg.vector_norm(expected_float).item())
    error_l2 = float(torch.linalg.vector_norm(finite_error).item())
    bitwise_mismatch_count = None
    if actual.dtype == expected.dtype:
        bitwise_mismatch_count = int(
            torch.count_nonzero(actual != expected).item()
        )
    correct = (
        int(torch.count_nonzero(mismatch).item()) == 0
        and int(torch.count_nonzero(~finite).item()) == 0
        and (
            not require_bitwise
            or bitwise_mismatch_count == 0
        )
    )
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype).replace("torch.", ""),
        "mismatch_count": int(torch.count_nonzero(mismatch).item()),
        "bitwise_mismatch_count": bitwise_mismatch_count,
        "bitwise_required": require_bitwise,
        "nonfinite_count": int(torch.count_nonzero(~finite).item()),
        "all_values_finite": bool(torch.all(finite).item()),
        "max_abs_error": float(torch.max(finite_error).item()),
        "relative_l2_error": (
            error_l2 / expected_l2 if expected_l2 != 0.0 else error_l2
        ),
        "atol": atol,
        "rtol": rtol,
        "actual_sha256": tensor_sha256(actual),
        "expected_sha256": tensor_sha256(expected),
        "correct": correct,
    }


def tensor_descriptor(value: torch.Tensor) -> dict:
    return {
        "dtype": str(value.dtype).replace("torch.", ""),
        "shape": list(value.shape),
        "sha256": tensor_sha256(value),
    }


def load_nvidia_golden(
    golden_dir: Path,
    token_id: int,
    hidden_input: torch.Tensor,
    weights: dict[str, torch.Tensor],
    conv_state_initial: torch.Tensor,
    recurrent_state_initial: torch.Tensor,
    checkpoint_provenance: dict,
) -> dict:
    golden_dir = golden_dir.expanduser().resolve(strict=True)
    if not golden_dir.is_dir():
        raise RuntimeError(f"NVIDIA golden path is not a directory: {golden_dir}")
    metadata_file = golden_dir / "metadata.json"
    results_file = golden_dir / "results.safetensors"
    if not metadata_file.is_file() or not results_file.is_file():
        raise RuntimeError(
            "NVIDIA golden directory must contain regular metadata.json and "
            "results.safetensors files"
        )

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError("NVIDIA golden metadata must be a JSON object")
    if metadata.get("schema") != NVIDIA_GOLDEN_SCHEMA:
        raise RuntimeError("NVIDIA golden schema mismatch")
    if metadata.get("kind") != NVIDIA_GOLDEN_KIND:
        raise RuntimeError("NVIDIA golden kind mismatch")
    if metadata.get("stage_order") != NVIDIA_GOLDEN_STAGE_ORDER:
        raise RuntimeError("NVIDIA golden stage order mismatch")

    expected_case = {
        "ba_order": ["b", "a"],
        "cache": "empty",
        "gdn_output_norm_weight": "checkpoint weight directly",
        "layer": 0,
        "layer_type": "linear_attention",
        "linear_weight_layout": "checkpoint [out,in], matmul uses weight.T",
        "outer_rms_weight": "1 + checkpoint raw weight",
        "qkvz_order": ["q", "k", "v", "z"],
        "returned_pair": ["mlp_down", "post_attention_residual"],
        "selected_conv_state": SELECTED_CONV_STATE,
        "state_initialization": "zero_first_token",
        "token_id": token_id,
        "tokens": 1,
    }
    if metadata.get("case") != expected_case:
        raise RuntimeError("NVIDIA golden case provenance mismatch")

    model = metadata.get("model")
    expected_model = {
        "id": MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "config_sha256": checkpoint_provenance["config_sha256"],
        "index_sha256": checkpoint_provenance["index_sha256"],
        "index_total_size": checkpoint_provenance[
            "checkpoint_index_total_size"
        ],
        "shard": checkpoint_provenance["selected_shard"],
        "shard_bytes": checkpoint_provenance["selected_shard_bytes"],
    }
    if not isinstance(model, dict) or any(
        model.get(key) != value for key, value in expected_model.items()
    ):
        raise RuntimeError("NVIDIA golden model provenance mismatch")
    for key in ("directory", "shard_manifest_sha256"):
        if not isinstance(model.get(key), str) or not model[key]:
            raise RuntimeError(f"NVIDIA golden model provenance lacks {key}")

    selected_hashes = dict(
        checkpoint_provenance["selected_layer_tensor_sha256"]
    )
    selected_hashes[
        f'{checkpoint_provenance["embedding_weight_name"]}[{token_id}]'
    ] = checkpoint_provenance["embedding_row_sha256"]
    if metadata.get("selected_checkpoint_tensor_sha256") != selected_hashes:
        raise RuntimeError("NVIDIA golden checkpoint tensor hashes mismatch")
    expected_execution_weights = {
        name: tensor_descriptor(value) for name, value in sorted(weights.items())
    }
    if metadata.get("execution_weights") != expected_execution_weights:
        raise RuntimeError("NVIDIA golden execution weight contract mismatch")
    expected_inputs = {
        "hidden": tensor_descriptor(hidden_input),
        "initial_conv_state": tensor_descriptor(conv_state_initial),
        "initial_recurrent_state": tensor_descriptor(recurrent_state_initial),
    }
    if metadata.get("input") != expected_inputs:
        raise RuntimeError("NVIDIA golden input/cache provenance mismatch")

    environment = metadata.get("environment")
    gpu = environment.get("gpu") if isinstance(environment, dict) else None
    if not isinstance(gpu, dict):
        raise RuntimeError("NVIDIA golden metadata has no GPU identity")
    for key in ("name", "uuid", "nvidia_smi_record"):
        if not isinstance(gpu.get(key), str) or not gpu[key]:
            raise RuntimeError(f"NVIDIA golden GPU identity lacks {key}")
    compute_capability = gpu.get("compute_capability")
    if (
        not isinstance(compute_capability, list)
        or len(compute_capability) != 2
        or not all(isinstance(value, int) for value in compute_capability)
    ):
        raise RuntimeError("NVIDIA golden GPU compute capability is malformed")
    script = metadata.get("script")
    if (
        not isinstance(script, dict)
        or script.get("path") != "tools/qwen35_nvidia_golden.py"
        or not isinstance(script.get("sha256"), str)
        or len(script["sha256"]) != 64
    ):
        raise RuntimeError("NVIDIA golden generator provenance is malformed")

    results_sha256 = file_sha256(results_file)
    if metadata.get("results_file_sha256") != results_sha256:
        raise RuntimeError("NVIDIA golden results file hash mismatch")
    expected_embedded_provenance = {
        "layer": 0,
        "model_id": MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "schema": NVIDIA_GOLDEN_SCHEMA,
        "token_id": token_id,
    }
    with safe_open(results_file, framework="pt", device="cpu") as tensors:
        observed_keys = set(tensors.keys())
        if observed_keys != set(NVIDIA_GOLDEN_CONTRACT):
            raise RuntimeError(
                f"NVIDIA golden tensor set mismatch: {sorted(observed_keys)!r}"
            )
        embedded_metadata = tensors.metadata()
        if (
            not isinstance(embedded_metadata, dict)
            or set(embedded_metadata) != {"provenance"}
        ):
            raise RuntimeError("NVIDIA golden embedded metadata mismatch")
        try:
            embedded_provenance = json.loads(embedded_metadata["provenance"])
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "NVIDIA golden embedded provenance is malformed"
            ) from error
        if embedded_provenance != expected_embedded_provenance:
            raise RuntimeError("NVIDIA golden embedded provenance mismatch")
        loaded = {
            name: tensors.get_tensor(name).clone().contiguous()
            for name in NVIDIA_GOLDEN_CONTRACT
        }

    observed_contract = {
        name: (value.dtype, tuple(value.shape))
        for name, value in loaded.items()
    }
    if observed_contract != NVIDIA_GOLDEN_CONTRACT:
        raise RuntimeError(
            f"NVIDIA golden tensor contract mismatch: {observed_contract!r}"
        )
    observed_results = {
        name: tensor_descriptor(value) for name, value in loaded.items()
    }
    if metadata.get("results") != observed_results:
        raise RuntimeError("NVIDIA golden per-tensor metadata/hash mismatch")
    if not torch.equal(loaded["final_hidden"], loaded["mlp_down"]):
        raise RuntimeError("NVIDIA golden final_hidden alias mismatch")
    if not torch.equal(
        loaded["final_residual"], loaded["post_attention_residual"]
    ):
        raise RuntimeError("NVIDIA golden final_residual alias mismatch")

    return {
        "directory": str(golden_dir),
        "metadata": metadata,
        "metadata_sha256": file_sha256(metadata_file),
        "results_sha256": results_sha256,
        "tensors": loaded,
    }


def compare_nvidia_golden_tensor(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    *,
    require_bitwise: bool = False,
) -> dict:
    if (
        actual.dtype != expected.dtype
        or tuple(actual.shape) != tuple(expected.shape)
    ):
        raise RuntimeError(
            f"NVIDIA golden {name} actual tensor contract mismatch: "
            f"{actual.dtype}, {tuple(actual.shape)}"
        )
    return compare_stage(
        actual,
        expected,
        atol,
        rtol,
        require_bitwise=require_bitwise,
    )


def bf16_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(
        x.to(torch.float32), weight.to(torch.float32).T
    ).to(torch.bfloat16)


def gemma_rms_reference(
    x: torch.Tensor, raw_weight: torch.Tensor
) -> torch.Tensor:
    x_float = x.to(torch.float32)
    variance = torch.mean(x_float * x_float, dim=-1, keepdim=True)
    return (
        x_float
        * torch.rsqrt(variance + EPSILON)
        * (1.0 + raw_weight.to(torch.float32))
    ).to(torch.bfloat16)


def conv_reference(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    state_cache: torch.Tensor,
    state_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_before = state_cache[state_index]
    x_float = mixed_qkv.to(torch.float32)
    accumulator = (
        state_before[:, 0].to(torch.float32)
        * weight[:, 0].to(torch.float32)
        + state_before[:, 1].to(torch.float32)
        * weight[:, 1].to(torch.float32)
        + state_before[:, 2].to(torch.float32)
        * weight[:, 2].to(torch.float32)
        + x_float[0] * weight[:, 3].to(torch.float32)
    )
    output = (accumulator * torch.sigmoid(accumulator)).to(torch.bfloat16)
    next_cache = state_cache.clone()
    next_cache[state_index] = torch.stack(
        (state_before[:, 1], state_before[:, 2], mixed_qkv[0]), dim=1
    ).to(torch.bfloat16)
    return output.view(1, QKV_DIM), next_cache


def norm_gate_reference(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    x_float = x.to(torch.float32)
    z_float = z.to(torch.float32)
    variance = torch.mean(x_float * x_float, dim=-1, keepdim=True)
    normalized = x_float * torch.rsqrt(variance + EPSILON)
    return (
        normalized * weight.to(torch.float32)
        * (z_float * torch.sigmoid(z_float))
    ).to(torch.bfloat16)


def fused_residual_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    raw_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed = x.to(torch.float32) + residual.to(torch.float32)
    residual_out = summed.to(torch.bfloat16)
    variance = torch.mean(summed * summed, dim=-1, keepdim=True)
    output = (
        summed
        * torch.rsqrt(variance + EPSILON)
        * (1.0 + raw_weight.to(torch.float32))
    ).to(torch.bfloat16)
    return output, residual_out


def random_bf16(shape: tuple[int, ...], scale: float) -> torch.Tensor:
    return (
        scale
        * torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    ).to(torch.bfloat16)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Qwen3.5-0.8B layer-0 decode path on gemsim_amd"
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="load the pinned local checkpoint weights and embedding row",
    )
    parser.add_argument(
        "--token-id",
        type=int,
        default=248044,
        help="checkpoint token ID used for the decode embedding input",
    )
    parser.add_argument(
        "--nvidia-golden-dir",
        type=Path,
        help=(
            "directory containing independent NVIDIA metadata.json and "
            "results.safetensors"
        ),
    )
    args = parser.parse_args()
    if args.nvidia_golden_dir is not None and not args.checkpoint:
        parser.error("--nvidia-golden-dir requires --checkpoint")

    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "gemsim_amd" or target.arch != "gfx950":
        raise RuntimeError(f"unexpected Triton execution target: {target}")
    if DEVICE.type != "cpu":
        raise RuntimeError(
            f"gemsim_amd must expose a CPU staging device, got {DEVICE}"
        )

    torch.manual_seed(131)
    hidden, hidden_guard = guarded_tensor(
        (1, HIDDEN_SIZE), torch.bfloat16, -41.0
    )
    if args.checkpoint:
        checkpoint_hidden, weights, provenance = checkpoint_inputs(args.token_id)
        hidden.copy_(checkpoint_hidden)
    else:
        hidden.copy_(random_bf16((1, HIDDEN_SIZE), 0.25))
        weights = {
            "input_norm": random_bf16((HIDDEN_SIZE,), 0.05),
            "qkvz": random_bf16((QKVZ_DIM, HIDDEN_SIZE), 0.03125),
            "ba": random_bf16((BA_DIM, HIDDEN_SIZE), 0.015625),
            "conv": random_bf16((QKV_DIM, CONV_WIDTH), 0.125),
            "a_log": -0.25
            + 0.05
            * torch.randn((NUM_HEADS,), dtype=torch.float32, device=DEVICE),
            "dt_bias": random_bf16((NUM_HEADS,), 0.1),
            "output_norm": 1.0
            + 0.125
            * torch.randn((HEAD_DIM,), dtype=torch.float32, device=DEVICE),
            "gdn_out": random_bf16((HIDDEN_SIZE, Z_DIM), 0.03125),
            "post_attention_norm": random_bf16((HIDDEN_SIZE,), 0.05),
            "gate_up": random_bf16((GATE_UP_SIZE, HIDDEN_SIZE), 0.03125),
            "down": random_bf16((HIDDEN_SIZE, INTERMEDIATE_SIZE), 0.03125),
        }
        provenance = {
            "mode": "deterministic_synthetic",
            "seed": 131,
            "layer": 0,
        }
    hidden_before = hidden.clone()
    residual_input = hidden_before.clone()
    weight_snapshots = {name: value.clone() for name, value in weights.items()}

    conv_state, conv_state_guard = guarded_tensor(
        (CONV_CACHE_LINES, QKV_DIM, CONV_STATE_WIDTH),
        torch.bfloat16,
        -43.0,
    )
    if args.checkpoint:
        conv_state.zero_()
    else:
        conv_state.copy_(
            random_bf16(
                (CONV_CACHE_LINES, QKV_DIM, CONV_STATE_WIDTH), 0.125
            )
        )
    conv_state_initial = conv_state.clone()
    conv_state_index = torch.tensor(
        [SELECTED_CONV_STATE], dtype=torch.int32, device=DEVICE
    )
    conv_state_index_before = conv_state_index.clone()

    recurrent_state, recurrent_state_guard = guarded_tensor(
        (NUM_HEADS, HEAD_DIM, HEAD_DIM), torch.float32, 47.0
    )
    if args.checkpoint:
        recurrent_state.zero_()
    else:
        recurrent_state.copy_(
            0.002
            * torch.randn(
                (NUM_HEADS, HEAD_DIM, HEAD_DIM),
                dtype=torch.float32,
                device=DEVICE,
            )
        )
    recurrent_state_initial = recurrent_state.clone()

    nvidia_golden = None
    if args.nvidia_golden_dir is not None:
        nvidia_golden = load_nvidia_golden(
            args.nvidia_golden_dir,
            args.token_id,
            hidden_before,
            weight_snapshots,
            conv_state_initial,
            recurrent_state_initial,
            provenance,
        )

    guards = {
        "hidden_input": hidden_guard,
        "conv_state": conv_state_guard,
        "recurrent_state": recurrent_state_guard,
    }
    stages = {}
    intermediate_inputs_unchanged = {}

    normalized, guards["input_norm"] = guarded_tensor(
        (1, HIDDEN_SIZE), torch.bfloat16, 51.0
    )
    expected_normalized = gemma_rms_reference(
        hidden_before, weight_snapshots["input_norm"]
    )
    plain_rms_norm(hidden, normalized, weights["input_norm"], 1)
    stages["input_rms_norm"] = compare_stage(
        normalized, expected_normalized, 0.015625, 0.02
    )
    normalized_before_projections = normalized.clone()

    qkvz, guards["qkvz"] = guarded_tensor(
        (1, QKVZ_DIM), torch.bfloat16, 53.0
    )
    expected_qkvz = bf16_matmul(normalized, weight_snapshots["qkvz"])
    dense_linear(normalized, weights["qkvz"], qkvz)
    stages["qkvz_projection"] = compare_stage(
        qkvz, expected_qkvz, 0.03125, 0.03
    )
    intermediate_inputs_unchanged["normalized_by_qkvz_projection"] = bool(
        torch.equal(normalized, normalized_before_projections)
    )

    ba, guards["ba"] = guarded_tensor(
        (1, BA_DIM), torch.bfloat16, 55.0
    )
    expected_ba = bf16_matmul(normalized, weight_snapshots["ba"])
    dense_linear(normalized, weights["ba"], ba)
    stages["ba_projection"] = compare_stage(
        ba, expected_ba, 0.03125, 0.03
    )
    intermediate_inputs_unchanged["normalized_by_ba_projection"] = bool(
        torch.equal(normalized, normalized_before_projections)
    )

    mixed_qkv = qkvz[:, :QKV_DIM]
    z = qkvz[:, QKV_DIM:].view(1, NUM_HEADS, HEAD_DIM)
    z_before_conv = z.clone()
    mixed_qkv_before_conv = mixed_qkv.clone()
    expected_mixed_qkv, expected_conv_state = conv_reference(
        mixed_qkv_before_conv,
        weight_snapshots["conv"],
        conv_state_initial,
        SELECTED_CONV_STATE,
    )
    gdn_conv_decode(
        mixed_qkv, weights["conv"], conv_state, conv_state_index
    )
    stages["gdn_conv_output"] = compare_stage(
        mixed_qkv, expected_mixed_qkv, 0.05, 0.01
    )
    stages["conv_state"] = compare_stage(
        conv_state,
        expected_conv_state,
        0.0,
        0.0,
        require_bitwise=True,
    )
    z_preserved_by_conv = bool(torch.equal(z, z_before_conv))
    conv_state_index_unchanged = bool(
        torch.equal(conv_state_index, conv_state_index_before)
    )

    b = ba[:, :NUM_HEADS]
    a = ba[:, NUM_HEADS:]
    expected_recurrent_output, expected_recurrent_state = recurrent_reference(
        mixed_qkv,
        a,
        b,
        weight_snapshots["a_log"],
        weight_snapshots["dt_bias"],
        recurrent_state_initial,
    )
    mixed_qkv_before_recurrent = mixed_qkv.clone()
    a_before_recurrent = a.clone()
    b_before_recurrent = b.clone()
    recurrent_output, guards["recurrent_output"] = guarded_tensor(
        (1, NUM_HEADS, HEAD_DIM), torch.bfloat16, 57.0
    )
    gdn_recurrent_decode(
        mixed_qkv,
        a,
        b,
        weights["a_log"],
        weights["dt_bias"],
        recurrent_state,
        recurrent_output,
    )
    stages["gdn_recurrent_output"] = compare_stage(
        recurrent_output, expected_recurrent_output, 0.015625, 0.02
    )
    stages["recurrent_state"] = compare_stage(
        recurrent_state, expected_recurrent_state, 2.0e-5, 2.0e-4
    )
    intermediate_inputs_unchanged["mixed_qkv_by_recurrent"] = bool(
        torch.equal(mixed_qkv, mixed_qkv_before_recurrent)
    )
    intermediate_inputs_unchanged["a_by_recurrent"] = bool(
        torch.equal(a, a_before_recurrent)
    )
    intermediate_inputs_unchanged["b_by_recurrent"] = bool(
        torch.equal(b, b_before_recurrent)
    )

    norm_gate_output, guards["output_norm_gate"] = guarded_tensor(
        (1, NUM_HEADS, HEAD_DIM), torch.bfloat16, 59.0
    )
    expected_norm_gate = norm_gate_reference(
        recurrent_output,
        z,
        weight_snapshots["output_norm"],
    )
    recurrent_output_before_norm_gate = recurrent_output.clone()
    z_before_norm_gate = z.clone()
    gdn_output_norm_gate(
        recurrent_output,
        z,
        weights["output_norm"],
        norm_gate_output,
        1,
    )
    stages["output_rms_norm_gate"] = compare_stage(
        norm_gate_output, expected_norm_gate, 0.01, 0.01
    )
    intermediate_inputs_unchanged["recurrent_output_by_norm_gate"] = bool(
        torch.equal(recurrent_output, recurrent_output_before_norm_gate)
    )
    intermediate_inputs_unchanged["z_by_norm_gate"] = bool(
        torch.equal(z, z_before_norm_gate)
    )

    attention_output, guards["gdn_out"] = guarded_tensor(
        (1, HIDDEN_SIZE), torch.bfloat16, 61.0
    )
    expected_attention_output = bf16_matmul(
        norm_gate_output.view(1, Z_DIM), weight_snapshots["gdn_out"]
    )
    norm_gate_output_before_projection = norm_gate_output.clone()
    dense_linear(
        norm_gate_output.view(1, Z_DIM), weights["gdn_out"], attention_output
    )
    stages["gdn_out_projection"] = compare_stage(
        attention_output, expected_attention_output, 0.03125, 0.03
    )
    intermediate_inputs_unchanged["norm_gate_by_gdn_out_projection"] = bool(
        torch.equal(norm_gate_output, norm_gate_output_before_projection)
    )

    post_norm, guards["post_attention_norm"] = guarded_tensor(
        (1, HIDDEN_SIZE), torch.bfloat16, 63.0
    )
    residual_out, guards["post_attention_residual"] = guarded_tensor(
        (1, HIDDEN_SIZE), torch.bfloat16, 65.0
    )
    expected_post_norm, expected_residual_out = fused_residual_reference(
        attention_output,
        residual_input,
        weight_snapshots["post_attention_norm"],
    )
    attention_output_before_residual_norm = attention_output.clone()
    fused_residual_rms_norm(
        attention_output,
        residual_input,
        weights["post_attention_norm"],
        post_norm,
        residual_out,
        1,
    )
    stages["post_attention_rms_norm"] = compare_stage(
        post_norm, expected_post_norm, 0.01, 0.01
    )
    stages["post_attention_residual"] = compare_stage(
        residual_out,
        expected_residual_out,
        0.0,
        0.0,
        require_bitwise=True,
    )
    intermediate_inputs_unchanged["attention_by_residual_norm"] = bool(
        torch.equal(attention_output, attention_output_before_residual_norm)
    )

    gate_up, guards["gate_up"] = guarded_tensor(
        (1, GATE_UP_SIZE), torch.bfloat16, 67.0
    )
    expected_gate_up = bf16_matmul(
        post_norm, weight_snapshots["gate_up"]
    )
    post_norm_before_gate_up = post_norm.clone()
    dense_linear(post_norm, weights["gate_up"], gate_up)
    stages["mlp_gate_up"] = compare_stage(
        gate_up, expected_gate_up, 0.03125, 0.03
    )
    intermediate_inputs_unchanged["post_norm_by_gate_up_projection"] = bool(
        torch.equal(post_norm, post_norm_before_gate_up)
    )

    activated, guards["mlp_activation"] = guarded_tensor(
        (1, INTERMEDIATE_SIZE), torch.bfloat16, 69.0
    )
    scratch = torch.zeros((1,), dtype=torch.uint8, device=DEVICE)
    gate_float = gate_up[:, :INTERMEDIATE_SIZE].to(torch.float32)
    up_float = gate_up[:, INTERMEDIATE_SIZE:].to(torch.float32)
    expected_activated = (
        gate_float * torch.sigmoid(gate_float) * up_float
    ).to(torch.bfloat16)
    gate_up_before_activation = gate_up.clone()
    silu_and_mul(gate_up, activated, scratch, 1)
    stages["mlp_silu_and_mul"] = compare_stage(
        activated, expected_activated, 0.015625, 0.02
    )
    intermediate_inputs_unchanged["gate_up_by_activation"] = bool(
        torch.equal(gate_up, gate_up_before_activation)
    )

    final_hidden, guards["mlp_down"] = guarded_tensor(
        (1, HIDDEN_SIZE), torch.bfloat16, 71.0
    )
    expected_final_hidden = bf16_matmul(
        activated, weight_snapshots["down"]
    )
    activated_before_down = activated.clone()
    dense_linear(activated, weights["down"], final_hidden)
    stages["mlp_down"] = compare_stage(
        final_hidden, expected_final_hidden, 0.03125, 0.03
    )
    intermediate_inputs_unchanged["activation_by_down_projection"] = bool(
        torch.equal(activated, activated_before_down)
    )

    weights_unchanged = {
        name: bool(torch.equal(weights[name], snapshot))
        for name, snapshot in weight_snapshots.items()
    }
    guards_unchanged = {
        name: guard_unchanged(guard) for name, guard in guards.items()
    }
    hidden_input_unchanged = bool(torch.equal(hidden, hidden_before))
    residual_input_unchanged = bool(
        torch.equal(residual_input, hidden_before)
    )
    conv_unselected_lines_unchanged = bool(
        torch.equal(conv_state[0], conv_state_initial[0])
        and torch.equal(conv_state[2], conv_state_initial[2])
    )
    states_nonzero = {
        "conv_initial": int(torch.count_nonzero(conv_state_initial).item()),
        "conv_final": int(torch.count_nonzero(conv_state).item()),
        "recurrent_initial": int(
            torch.count_nonzero(recurrent_state_initial).item()
        ),
        "recurrent_final": int(torch.count_nonzero(recurrent_state).item()),
    }
    state_initialization_valid = (
        states_nonzero["conv_initial"] == 0
        and states_nonzero["recurrent_initial"] == 0
        and states_nonzero["conv_final"] > 0
        and states_nonzero["recurrent_final"] > 0
        if args.checkpoint
        else all(count > 0 for count in states_nonzero.values())
    )

    nvidia_golden_correct = True
    nvidia_golden_payload = {"enabled": False}
    if nvidia_golden is not None:
        golden_tensors = nvidia_golden["tensors"]
        actual_stage_tensors = {
            "input_rms_norm": normalized,
            "qkvz_projection": torch.cat(
                (mixed_qkv_before_conv, z_before_conv.view(1, Z_DIM)), dim=1
            ),
            "ba_projection": ba,
            "gdn_conv_output": mixed_qkv,
            "conv_state": conv_state,
            "gdn_recurrent_output": recurrent_output,
            "recurrent_state": recurrent_state,
            "output_rms_norm_gate": norm_gate_output,
            "gdn_out_projection": attention_output,
            "post_attention_rms_norm": post_norm,
            "post_attention_residual": residual_out,
            "mlp_gate_up": gate_up,
            "mlp_silu_and_mul": activated,
            "mlp_down": final_hidden,
        }
        if list(actual_stage_tensors) != NVIDIA_GOLDEN_STAGE_ORDER:
            raise RuntimeError("internal NVIDIA golden stage binding mismatch")
        golden_stages = {}
        for name, actual in actual_stage_tensors.items():
            atol, rtol = NVIDIA_GOLDEN_TOLERANCES[name]
            golden_stages[name] = compare_nvidia_golden_tensor(
                name, actual, golden_tensors[name], atol, rtol
            )
        golden_boundaries = {
            "hidden_input": compare_nvidia_golden_tensor(
                "hidden_input",
                hidden_before,
                golden_tensors["hidden_input"],
                0.0,
                0.0,
                require_bitwise=True,
            ),
            "final_hidden": compare_nvidia_golden_tensor(
                "final_hidden",
                final_hidden,
                golden_tensors["final_hidden"],
                *NVIDIA_GOLDEN_TOLERANCES["mlp_down"],
            ),
            "final_residual": compare_nvidia_golden_tensor(
                "final_residual",
                residual_out,
                golden_tensors["final_residual"],
                *NVIDIA_GOLDEN_TOLERANCES["post_attention_residual"],
            ),
        }
        golden_comparisons = [
            *golden_stages.values(),
            *golden_boundaries.values(),
        ]
        nvidia_golden_correct = all(
            comparison["correct"] for comparison in golden_comparisons
        )
        golden_metadata = nvidia_golden["metadata"]
        nvidia_golden_payload = {
            "enabled": True,
            "role": "external_oracle_excluded_from_target_and_fallback_counts",
            "directory": nvidia_golden["directory"],
            "schema": golden_metadata["schema"],
            "kind": golden_metadata["kind"],
            "results_sha256": nvidia_golden["results_sha256"],
            "metadata_sha256": nvidia_golden["metadata_sha256"],
            "generator": golden_metadata["script"],
            "model": {
                key: golden_metadata["model"][key]
                for key in (
                    "id",
                    "revision",
                    "config_sha256",
                    "index_sha256",
                )
            },
            "gpu": golden_metadata["environment"]["gpu"],
            "tensor_count": len(golden_comparisons),
            "stage_order": NVIDIA_GOLDEN_STAGE_ORDER,
            "stages": golden_stages,
            "boundaries": golden_boundaries,
            "mismatch_count": sum(
                comparison["mismatch_count"]
                for comparison in golden_comparisons
            ),
            "nonfinite_count": sum(
                comparison["nonfinite_count"]
                for comparison in golden_comparisons
            ),
            "max_abs_error": max(
                comparison["max_abs_error"]
                for comparison in golden_comparisons
            ),
            "max_relative_l2_error": max(
                comparison["relative_l2_error"]
                for comparison in golden_comparisons
            ),
            "correct": nvidia_golden_correct,
        }
    correct = (
        all(stage["correct"] for stage in stages.values())
        and all(weights_unchanged.values())
        and all(guards_unchanged.values())
        and all(intermediate_inputs_unchanged.values())
        and hidden_input_unchanged
        and residual_input_unchanged
        and z_preserved_by_conv
        and conv_state_index_unchanged
        and conv_unselected_lines_unchanged
        and state_initialization_valid
        and nvidia_golden_correct
    )
    payload = {
        "schema": "amdgpu-sim.triton-qwen35-layer0-decode.v1",
        "backend": target.backend,
        "arch": target.arch,
        "model": MODEL_ID,
        "provenance": provenance,
        "scope": {
            "unit": "decoder_layer_0",
            "tokens": 1,
            "decoder_forward_residual_argument": "none",
            "post_attention_residual_input": "layer input",
            "weights": "pinned_checkpoint" if args.checkpoint else "synthetic",
            "input": "checkpoint_embedding_row" if args.checkpoint else "synthetic",
            "cache": (
                "empty_first_token" if args.checkpoint else "synthetic_nonzero"
            ),
            "cache_routing": {
                "conv": "3-line cache selected in kernel by int32 index 1",
                "recurrent": (
                    "single preselected state tensor; no cache index routing"
                ),
            },
            "oracle": "stage_local_reference_from_observed_prior_stage_output",
            "excludes": [
                "embedding_kernel",
                "decoder layers 1-23",
                "final norm",
                "lm head",
                "token sampling",
                "prompt-prefilled cache",
                "multi-sequence recurrent cache routing",
                "full-model end-to-end correctness",
            ],
        },
        "return_contract": {
            "api": "Qwen3NextDecoderLayer.forward",
            "hidden_states": "layer-0 MLP down-projection output",
            "residual": "BF16(layer input FP32 + attention output FP32)",
            "materialized_post_mlp_residual": False,
            "next_layer_consumes_hidden_states_and_residual_separately": True,
        },
        "layer": 0,
        "layer_type": "linear_attention",
        "tokens": 1,
        "dtype": "bfloat16",
        "stage_order": list(stages),
        "stages": stages,
        "nvidia_golden": nvidia_golden_payload,
        "conv_state_shape": [
            CONV_CACHE_LINES,
            QKV_DIM,
            CONV_STATE_WIDTH,
        ],
        "selected_conv_state": SELECTED_CONV_STATE,
        "recurrent_state_shape": [NUM_HEADS, HEAD_DIM, HEAD_DIM],
        "recurrent_state_dtype": "float32",
        "states_nonzero": states_nonzero,
        "state_initialization": (
            "zero_first_token" if args.checkpoint else "deterministic_nonzero"
        ),
        "state_initialization_valid": state_initialization_valid,
        "conv_state_index_unchanged": conv_state_index_unchanged,
        "conv_unselected_lines_unchanged": conv_unselected_lines_unchanged,
        "z_preserved_by_in_place_conv": z_preserved_by_conv,
        "hidden_input_unchanged": hidden_input_unchanged,
        "residual_input_unchanged": residual_input_unchanged,
        "intermediate_inputs_unchanged": intermediate_inputs_unchanged,
        "weights_unchanged": weights_unchanged,
        "guards_unchanged": guards_unchanged,
        "residual_semantics": (
            "(layer_input.float() + gdn_out.float()).bfloat16()"
        ),
        "fused_residual_bitwise_required": True,
        "final_hidden_sha256": tensor_sha256(final_hidden),
        "expected_final_hidden_sha256": tensor_sha256(expected_final_hidden),
        "final_residual_sha256": tensor_sha256(residual_out),
        "expected_final_residual_sha256": tensor_sha256(
            expected_residual_out
        ),
        "mismatch_count": sum(
            stage["mismatch_count"] for stage in stages.values()
        ),
        "bitwise_mismatch_count": sum(
            stage["bitwise_mismatch_count"] or 0
            for stage in stages.values()
            if stage["bitwise_required"]
        ),
        "nonfinite_count": sum(
            stage["nonfinite_count"] for stage in stages.values()
        ),
        "fallback_count": 0,
        "cpu_fallback_count": 0,
        "nvidia_fallback_count": 0,
        "persistent_cache": "qwen35-layer0-decode",
        "output_correct": correct,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
