#!/usr/bin/env python3

import runpy
from pathlib import Path


HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE / "_gemsim_bootstrap.py"))["bootstrap"](
    __file__, "qwen35-layer3-decode"
)

import argparse
import hashlib
import json
import math

import torch
import triton
import triton.language as tl

from safetensors import safe_open


DEVICE = triton.runtime.driver.active.get_active_torch_device()
ROOT = HERE.parents[1]
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3584
GATE_UP_SIZE = 2 * INTERMEDIATE_SIZE
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 256
Q_SIZE = NUM_Q_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
Q_GATE_SIZE = 2 * Q_SIZE
QKV_GATE_SIZE = Q_GATE_SIZE + 2 * KV_SIZE
ROTARY_DIM = 64
HALF_ROTARY = ROTARY_DIM // 2
ROPE_THETA = 10_000_000.0
EPSILON = 1.0e-6
KV_BLOCK_SIZE = 16
DECODE_POSITION = 5
SEQUENCE_LENGTH = DECODE_POSITION + 1
KV_CONTENT_SIZE = 2 * HEAD_DIM
GUARD_ELEMENTS = 1024
VOCAB_SIZE = 248320
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_FILENAME = "model.safetensors-00001-of-00001.safetensors"
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_tensor_sha256(value: torch.Tensor) -> str:
    raw = value.view(torch.uint16) if value.dtype == torch.bfloat16 else value
    return hashlib.sha256(
        raw.contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def checkpoint_inputs(token_id: int) -> tuple[torch.Tensor, dict, dict]:
    if token_id < 0 or token_id >= VOCAB_SIZE:
        raise ValueError(f"token ID is out of range: {token_id}")

    model_dir = ROOT / "models/Qwen3.5-0.8B"
    artifact_hashes = {}
    for filename, expected in EXPECTED_ARTIFACTS.items():
        path = model_dir / filename
        observed = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        if observed != expected:
            raise RuntimeError(
                f"checkpoint artifact mismatch for {filename}: {observed!r}"
            )
        artifact_hashes[filename] = observed

    config = json.loads((model_dir / "config.json").read_text())
    text_config = config.get("text_config", {})
    expected_config = {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_attention_heads": NUM_Q_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": EPSILON,
        "attn_output_gate": True,
        "vocab_size": VOCAB_SIZE,
    }
    observed_config = {
        name: text_config.get(name) for name in expected_config
    }
    if observed_config != expected_config:
        raise RuntimeError(
            f"checkpoint config contract mismatch: {observed_config!r}"
        )
    if text_config.get("layer_types", [])[3] != "full_attention":
        raise RuntimeError("checkpoint layer 3 is not full_attention")
    rope = text_config.get("rope_parameters", {})
    if (
        int(HEAD_DIM * rope.get("partial_rotary_factor", 0.0)) != ROTARY_DIM
        or float(rope.get("rope_theta", 0.0)) != ROPE_THETA
    ):
        raise RuntimeError(f"checkpoint RoPE contract mismatch: {rope!r}")

    manifest = json.loads((model_dir / "manifest.json").read_text())
    if (
        manifest.get("model_id") != "Qwen/Qwen3.5-0.8B"
        or manifest.get("revision") != MODEL_REVISION
        or manifest.get("files", {}).get(MODEL_FILENAME)
        != EXPECTED_ARTIFACTS[MODEL_FILENAME]
        or manifest.get("files", {}).get("config.json")
        != EXPECTED_ARTIFACTS["config.json"]
        or manifest.get("files", {}).get("model.safetensors.index.json")
        != EXPECTED_ARTIFACTS["model.safetensors.index.json"]
    ):
        raise RuntimeError("checkpoint manifest provenance contract mismatch")

    prefix = "model.language_model.layers.3"
    contracts = {
        "model.language_model.embed_tokens.weight": (
            torch.bfloat16,
            (VOCAB_SIZE, HIDDEN_SIZE),
        ),
        f"{prefix}.input_layernorm.weight": (
            torch.bfloat16,
            (HIDDEN_SIZE,),
        ),
        f"{prefix}.self_attn.q_proj.weight": (
            torch.bfloat16,
            (Q_GATE_SIZE, HIDDEN_SIZE),
        ),
        f"{prefix}.self_attn.k_proj.weight": (
            torch.bfloat16,
            (KV_SIZE, HIDDEN_SIZE),
        ),
        f"{prefix}.self_attn.v_proj.weight": (
            torch.bfloat16,
            (KV_SIZE, HIDDEN_SIZE),
        ),
        f"{prefix}.self_attn.q_norm.weight": (
            torch.bfloat16,
            (HEAD_DIM,),
        ),
        f"{prefix}.self_attn.k_norm.weight": (
            torch.bfloat16,
            (HEAD_DIM,),
        ),
        f"{prefix}.self_attn.o_proj.weight": (
            torch.bfloat16,
            (HIDDEN_SIZE, Q_SIZE),
        ),
        f"{prefix}.post_attention_layernorm.weight": (
            torch.bfloat16,
            (HIDDEN_SIZE,),
        ),
        f"{prefix}.mlp.gate_proj.weight": (
            torch.bfloat16,
            (INTERMEDIATE_SIZE, HIDDEN_SIZE),
        ),
        f"{prefix}.mlp.up_proj.weight": (
            torch.bfloat16,
            (INTERMEDIATE_SIZE, HIDDEN_SIZE),
        ),
        f"{prefix}.mlp.down_proj.weight": (
            torch.bfloat16,
            (HIDDEN_SIZE, INTERMEDIATE_SIZE),
        ),
    }
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text()
    )
    index_mapping = {
        name: index.get("weight_map", {}).get(name) for name in contracts
    }
    if any(value != MODEL_FILENAME for value in index_mapping.values()):
        raise RuntimeError(
            f"checkpoint index mapping mismatch: {index_mapping!r}"
        )

    model_file = model_dir / MODEL_FILENAME
    with safe_open(model_file, framework="pt", device="cpu") as tensors:
        embedding = tensors.get_slice(
            "model.language_model.embed_tokens.weight"
        )
        embedding_shape = tuple(embedding.get_shape())
        hidden = embedding[token_id : token_id + 1].clone().contiguous()
        loaded = {
            name: tensors.get_tensor(name).clone().contiguous()
            for name in contracts
            if name != "model.language_model.embed_tokens.weight"
        }
    observed_contracts = {
        "model.language_model.embed_tokens.weight": (
            hidden.dtype,
            embedding_shape,
        ),
        **{
            name: (tensor.dtype, tuple(tensor.shape))
            for name, tensor in loaded.items()
        },
    }
    if observed_contracts != contracts:
        raise RuntimeError(
            f"checkpoint tensor contract mismatch: {observed_contracts!r}"
        )
    if tuple(hidden.shape) != (1, HIDDEN_SIZE):
        raise RuntimeError(f"checkpoint embedding row mismatch: {hidden.shape}")
    for name, tensor in {
        "model.language_model.embed_tokens.weight": hidden,
        **loaded,
    }.items():
        expected_bytes = tensor.numel() * tensor.element_size()
        if tensor.untyped_storage().nbytes() != expected_bytes:
            raise RuntimeError(
                f"checkpoint tensor does not own exact storage: {name}"
            )

    q = loaded[f"{prefix}.self_attn.q_proj.weight"]
    k = loaded[f"{prefix}.self_attn.k_proj.weight"]
    v = loaded[f"{prefix}.self_attn.v_proj.weight"]
    gate = loaded[f"{prefix}.mlp.gate_proj.weight"]
    up = loaded[f"{prefix}.mlp.up_proj.weight"]
    weights = {
        "input_norm": loaded[f"{prefix}.input_layernorm.weight"],
        "qkv_gate": torch.cat((q, k, v), dim=0).contiguous(),
        "q_norm": loaded[f"{prefix}.self_attn.q_norm.weight"],
        "k_norm": loaded[f"{prefix}.self_attn.k_norm.weight"],
        "out": loaded[f"{prefix}.self_attn.o_proj.weight"],
        "post_attention_norm": loaded[
            f"{prefix}.post_attention_layernorm.weight"
        ],
        "gate_up": torch.cat((gate, up), dim=0).contiguous(),
        "down": loaded[f"{prefix}.mlp.down_proj.weight"],
    }
    source_tensor_hashes = {
        "embedding_row": raw_tensor_sha256(hidden),
        **{
            name: raw_tensor_sha256(tensor) for name, tensor in loaded.items()
        },
    }
    provenance = {
        "mode": "checkpoint_weights_embedding_input_empty_cache",
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "model_file": str(model_file.relative_to(ROOT)),
        "token_id": token_id,
        "embedding_weight_name": "model.language_model.embed_tokens.weight",
        "layer": 3,
        "artifact_hashes": artifact_hashes,
        "source_tensor_hashes": source_tensor_hashes,
        "assembled_weight_hashes": {
            name: raw_tensor_sha256(tensor) for name, tensor in weights.items()
        },
        "index_mapping_verified": True,
        "config_contract_verified": True,
        "manifest_provenance_verified": True,
        "model_level_golden_verified": False,
        "oracle_scope": "local_stage_actual_upstream_tensor",
        "loaded_weight_bytes": sum(
            tensor.numel() * tensor.element_size()
            for tensor in weights.values()
        ),
    }
    return hidden, weights, provenance


def load_runner(filename: str) -> dict:
    return runpy.run_path(str(HERE / filename))


dense_linear = load_runner("qwen35_dense_linear_correctness.py")[
    "dense_linear"
]
fused_residual_rms_norm = load_runner(
    "qwen35_fused_residual_rms_norm_correctness.py"
)["fused_residual_gemma_rms_norm"]
silu_and_mul = load_runner("silu_and_mul_correctness.py")["silu_and_mul"]


@triton.jit
def qk_rmsnorm_rope_gate_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_gamma_ptr,
    k_gamma_ptr,
    cos_sin_cache_ptr,
    position_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    EPSILON: tl.constexpr,
):
    head = tl.program_id(0)
    is_k = head >= NUM_Q_HEADS
    local_head = tl.where(is_k, head - NUM_Q_HEADS, head)

    if is_k:
        in_base = k_ptr + local_head * HEAD_DIM
        gamma_ptr = k_gamma_ptr
        out_base = k_out_ptr + local_head * HEAD_DIM
    else:
        in_base = q_gate_ptr + local_head * 2 * HEAD_DIM
        gamma_ptr = q_gamma_ptr
        out_base = q_out_ptr + local_head * HEAD_DIM

    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(in_base + offsets).to(tl.float32)
    gamma = tl.load(gamma_ptr + offsets).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / HEAD_DIM
    inverse_rms = tl.rsqrt(variance + EPSILON)
    normalized = (x * inverse_rms * gamma).to(tl.bfloat16).to(tl.float32)
    tl.store(out_base + offsets, normalized, mask=offsets >= ROTARY_DIM)

    rotary_offsets = tl.arange(0, HALF_ROTARY)
    x1 = tl.load(in_base + rotary_offsets).to(tl.float32)
    x2 = tl.load(in_base + HALF_ROTARY + rotary_offsets).to(tl.float32)
    gamma1 = tl.load(gamma_ptr + rotary_offsets).to(tl.float32)
    gamma2 = tl.load(gamma_ptr + HALF_ROTARY + rotary_offsets).to(tl.float32)
    x1 = (x1 * inverse_rms * gamma1).to(tl.bfloat16).to(tl.float32)
    x2 = (x2 * inverse_rms * gamma2).to(tl.bfloat16).to(tl.float32)
    position = tl.load(position_ptr).to(tl.int64)
    cache_base = position * ROTARY_DIM
    cosine = tl.load(cos_sin_cache_ptr + cache_base + rotary_offsets).to(
        tl.float32
    )
    sine = tl.load(
        cos_sin_cache_ptr + cache_base + HALF_ROTARY + rotary_offsets
    ).to(tl.float32)
    tl.store(out_base + rotary_offsets, x1 * cosine - x2 * sine)
    tl.store(
        out_base + HALF_ROTARY + rotary_offsets,
        x2 * cosine + x1 * sine,
    )

    if not is_k:
        gate_base = q_gate_ptr + local_head * 2 * HEAD_DIM + HEAD_DIM
        gate_out_base = gate_out_ptr + local_head * HEAD_DIM
        gate = tl.load(gate_base + offsets)
        tl.store(gate_out_base + offsets, gate)


@triton.jit
def kv_cache_store_kernel(
    k_ptr,
    v_ptr,
    cache_ptr,
    slot_ptr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    KV_CONTENT_SIZE: tl.constexpr,
):
    kv_head = tl.program_id(0)
    offsets = tl.arange(0, HEAD_DIM)
    slot = tl.load(slot_ptr)
    cache_base = (
        (slot * NUM_KV_HEADS + kv_head) * KV_CONTENT_SIZE
    )
    k = tl.load(k_ptr + kv_head * HEAD_DIM + offsets)
    v = tl.load(v_ptr + kv_head * HEAD_DIM + offsets)
    tl.store(cache_ptr + cache_base + offsets, k)
    tl.store(cache_ptr + cache_base + HEAD_DIM + offsets, v)


@triton.jit
def gqa_decode_scores_kernel(
    q_ptr,
    cache_ptr,
    scores_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    KV_CONTENT_SIZE: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    SCORE_BLOCK: tl.constexpr,
    DIM_BLOCK: tl.constexpr,
):
    q_head = tl.program_id(0)
    kv_group_size = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head = q_head // kv_group_size
    positions = tl.arange(0, SCORE_BLOCK)
    position_mask = positions < SEQUENCE_LENGTH
    accumulator = tl.zeros((SCORE_BLOCK,), tl.float32)

    for dim_start in range(0, HEAD_DIM, DIM_BLOCK):
        dimensions = dim_start + tl.arange(0, DIM_BLOCK)
        q = tl.load(q_ptr + q_head * HEAD_DIM + dimensions).to(tl.float32)
        cache_offsets = (
            (positions[:, None] * NUM_KV_HEADS + kv_head)
            * KV_CONTENT_SIZE
            + dimensions[None, :]
        )
        k = tl.load(
            cache_ptr + cache_offsets,
            mask=position_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(k * q[None, :], axis=1)

    scaled = accumulator * (1.0 / 16.0)
    tl.store(
        scores_ptr + q_head * SEQUENCE_LENGTH + positions,
        scaled,
        mask=position_mask,
    )


@triton.jit
def gqa_decode_output_kernel(
    scores_ptr,
    cache_ptr,
    output_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    KV_CONTENT_SIZE: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    SCORE_BLOCK: tl.constexpr,
    DIM_BLOCK: tl.constexpr,
):
    q_head = tl.program_id(0)
    dim_block = tl.program_id(1)
    kv_group_size = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head = q_head // kv_group_size
    positions = tl.arange(0, SCORE_BLOCK)
    position_mask = positions < SEQUENCE_LENGTH
    scores = tl.load(
        scores_ptr + q_head * SEQUENCE_LENGTH + positions,
        mask=position_mask,
        other=-float("inf"),
    ).to(tl.float32)
    scores -= tl.max(scores, axis=0)
    probabilities = tl.exp(scores)
    probabilities /= tl.sum(probabilities, axis=0)

    dimensions = dim_block * DIM_BLOCK + tl.arange(0, DIM_BLOCK)
    dimension_mask = dimensions < HEAD_DIM
    cache_offsets = (
        (positions[:, None] * NUM_KV_HEADS + kv_head) * KV_CONTENT_SIZE
        + HEAD_DIM
        + dimensions[None, :]
    )
    values = tl.load(
        cache_ptr + cache_offsets,
        mask=position_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    output = tl.sum(probabilities[:, None] * values, axis=0)
    tl.store(
        output_ptr + q_head * HEAD_DIM + dimensions,
        output.to(tl.bfloat16),
        mask=dimension_mask,
    )


@triton.jit
def sigmoid_output_gate_kernel(
    attention_ptr,
    gate_ptr,
    output_ptr,
    ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ELEMENTS
    attention = tl.load(attention_ptr + offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    # torch.sigmoid preserves the BF16 gate dtype in the pinned vLLM path.
    # Materialize that boundary before the BF16 attention multiplication.
    sigmoid_gate = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    output = attention * sigmoid_gate
    tl.store(output_ptr + offsets, output.to(tl.bfloat16), mask=mask)


def qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    gate_out: torch.Tensor,
) -> None:
    qk_rmsnorm_rope_gate_kernel[(NUM_Q_HEADS + NUM_KV_HEADS,)](
        q_gate,
        k,
        q_out,
        k_out,
        gate_out,
        q_gamma,
        k_gamma,
        cos_sin_cache,
        position,
        NUM_Q_HEADS=NUM_Q_HEADS,
        NUM_KV_HEADS=NUM_KV_HEADS,
        HEAD_DIM=HEAD_DIM,
        ROTARY_DIM=ROTARY_DIM,
        HALF_ROTARY=HALF_ROTARY,
        EPSILON=EPSILON,
        num_warps=1,
    )


def gqa_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    slot: torch.Tensor,
    scores: torch.Tensor,
    output: torch.Tensor,
    sequence_length: int,
) -> None:
    kv_cache_store_kernel[(NUM_KV_HEADS,)](
        k,
        v,
        cache,
        slot,
        NUM_KV_HEADS=NUM_KV_HEADS,
        HEAD_DIM=HEAD_DIM,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        KV_CONTENT_SIZE=KV_CONTENT_SIZE,
        num_warps=4,
    )
    gqa_decode_scores_kernel[(NUM_Q_HEADS,)](
        q,
        cache,
        scores,
        NUM_Q_HEADS=NUM_Q_HEADS,
        NUM_KV_HEADS=NUM_KV_HEADS,
        HEAD_DIM=HEAD_DIM,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        KV_CONTENT_SIZE=KV_CONTENT_SIZE,
        SEQUENCE_LENGTH=sequence_length,
        SCORE_BLOCK=8,
        DIM_BLOCK=64,
        num_warps=4,
    )
    gqa_decode_output_kernel[
        (NUM_Q_HEADS, triton.cdiv(HEAD_DIM, 64))
    ](
        scores,
        cache,
        output,
        NUM_Q_HEADS=NUM_Q_HEADS,
        NUM_KV_HEADS=NUM_KV_HEADS,
        HEAD_DIM=HEAD_DIM,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        KV_CONTENT_SIZE=KV_CONTENT_SIZE,
        SEQUENCE_LENGTH=sequence_length,
        SCORE_BLOCK=8,
        DIM_BLOCK=64,
        num_warps=4,
    )


def apply_sigmoid_output_gate(
    attention: torch.Tensor, gate: torch.Tensor, output: torch.Tensor
) -> None:
    sigmoid_output_gate_kernel[(triton.cdiv(Q_SIZE, 256),)](
        attention,
        gate,
        output,
        ELEMENTS=Q_SIZE,
        BLOCK_SIZE=256,
        num_warps=4,
    )


def guarded_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, sentinel: float
) -> tuple[torch.Tensor, dict]:
    elements = math.prod(shape)
    storage = torch.full(
        (GUARD_ELEMENTS + elements + GUARD_ELEMENTS,),
        sentinel,
        dtype=dtype,
        device=DEVICE,
    )
    begin = GUARD_ELEMENTS
    end = begin + elements
    return storage[begin:end].view(shape), {
        "storage": storage,
        "elements": elements,
        "prefix": storage[:begin].clone(),
        "suffix": storage[end:].clone(),
    }


def guard_unchanged(guard: dict) -> bool:
    end = GUARD_ELEMENTS + guard["elements"]
    return bool(
        torch.equal(guard["storage"][:GUARD_ELEMENTS], guard["prefix"])
        and torch.equal(guard["storage"][end:], guard["suffix"])
    )


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.view(torch.uint16) if value.dtype == torch.bfloat16 else value
    return hashlib.sha256(
        raw.contiguous().numpy().tobytes(order="C")
    ).hexdigest()


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
    bitwise_mismatch = None
    if actual.dtype == expected.dtype:
        bitwise_mismatch = int(torch.count_nonzero(actual != expected).item())
    expected_l2 = float(torch.linalg.vector_norm(expected_float).item())
    error_l2 = float(torch.linalg.vector_norm(finite_error).item())
    mismatch_count = int(torch.count_nonzero(mismatch).item())
    nonfinite_count = int(torch.count_nonzero(~finite).item())
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype).replace("torch.", ""),
        "mismatch_count": mismatch_count,
        "bitwise_mismatch_count": bitwise_mismatch,
        "bitwise_required": require_bitwise,
        "nonfinite_count": nonfinite_count,
        "all_values_finite": nonfinite_count == 0,
        "max_abs_error": float(torch.max(finite_error).item()),
        "relative_l2_error": (
            error_l2 / expected_l2 if expected_l2 != 0.0 else error_l2
        ),
        "atol": atol,
        "rtol": rtol,
        "actual_sha256": tensor_sha256(actual),
        "expected_sha256": tensor_sha256(expected),
        "correct": (
            mismatch_count == 0
            and nonfinite_count == 0
            and (not require_bitwise or bitwise_mismatch == 0)
        ),
    }


def random_bf16(shape: tuple[int, ...], scale: float) -> torch.Tensor:
    return (
        scale * torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    ).to(torch.bfloat16)


def bf16_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(
        x.to(torch.float32), weight.to(torch.float32).T
    ).to(torch.bfloat16)


def fused_residual_reference(
    x: torch.Tensor, residual: torch.Tensor, raw_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    summed = x.to(torch.float32) + residual.to(torch.float32)
    residual_out = summed.to(torch.bfloat16)
    variance = torch.mean(summed * summed, dim=-1, keepdim=True)
    normalized = (
        summed
        * torch.rsqrt(variance + EPSILON)
        * (1.0 + raw_weight.to(torch.float32))
    ).to(torch.bfloat16)
    return normalized, residual_out


def make_cos_sin_cache(sequence_length: int) -> torch.Tensor:
    inverse_frequency = 1.0 / (
        ROPE_THETA
        ** (
            torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32, device=DEVICE)
            / ROTARY_DIM
        )
    )
    positions = torch.arange(
        0, sequence_length, dtype=torch.float32, device=DEVICE
    )
    frequencies = positions[:, None] * inverse_frequency[None, :]
    return torch.cat((torch.cos(frequencies), torch.sin(frequencies)), dim=1).to(
        torch.bfloat16
    )


def qk_rope_reference(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_raw_weight: torch.Tensor,
    k_raw_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    decode_position: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_gate_heads = q_gate.view(1, NUM_Q_HEADS, 2 * HEAD_DIM)
    q_source = q_gate_heads[..., :HEAD_DIM]
    gate = q_gate_heads[..., HEAD_DIM:].reshape(1, Q_SIZE).clone()
    k_source = k.view(1, NUM_KV_HEADS, HEAD_DIM)

    def normalize(value: torch.Tensor, raw_weight: torch.Tensor) -> torch.Tensor:
        value_float = value.to(torch.float32)
        variance = torch.mean(value_float * value_float, dim=-1, keepdim=True)
        return (
            value_float
            * torch.rsqrt(variance + EPSILON)
            * (1.0 + raw_weight.to(torch.float32))
        ).to(torch.bfloat16)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        normalized = value.clone()
        cosine = cos_sin_cache[decode_position, :HALF_ROTARY].to(torch.float32)
        sine = cos_sin_cache[decode_position, HALF_ROTARY:].to(torch.float32)
        x1 = value[..., :HALF_ROTARY].to(torch.float32)
        x2 = value[..., HALF_ROTARY:ROTARY_DIM].to(torch.float32)
        normalized[..., :HALF_ROTARY] = (
            x1 * cosine - x2 * sine
        ).to(torch.bfloat16)
        normalized[..., HALF_ROTARY:ROTARY_DIM] = (
            x2 * cosine + x1 * sine
        ).to(torch.bfloat16)
        return normalized

    q = rotate(normalize(q_source, q_raw_weight)).reshape(1, Q_SIZE)
    normalized_k = normalize(k_source, k_raw_weight)
    rotated_k = rotate(normalized_k).reshape(1, KV_SIZE)
    return q, rotated_k, gate


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_initial: torch.Tensor,
    decode_position: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = cache_initial.clone()
    cache[0, decode_position, :, :HEAD_DIM] = k.view(NUM_KV_HEADS, HEAD_DIM)
    cache[0, decode_position, :, HEAD_DIM:] = v.view(NUM_KV_HEADS, HEAD_DIM)
    score_rows = []
    output_rows = []
    group_size = NUM_Q_HEADS // NUM_KV_HEADS
    for q_head in range(NUM_Q_HEADS):
        kv_head = q_head // group_size
        keys = cache[0, :sequence_length, kv_head, :HEAD_DIM].to(torch.float32)
        values = cache[0, :sequence_length, kv_head, HEAD_DIM:].to(torch.float32)
        query = q.view(NUM_Q_HEADS, HEAD_DIM)[q_head].to(torch.float32)
        scores = torch.matmul(keys, query) / math.sqrt(HEAD_DIM)
        probabilities = torch.softmax(scores, dim=0)
        score_rows.append(scores)
        output_rows.append(torch.matmul(probabilities, values))
    return (
        torch.stack(score_rows),
        torch.stack(output_rows).view(1, Q_SIZE).to(torch.bfloat16),
        cache,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Qwen3.5-0.8B layer-3 decode path on gemsim_amd"
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
        help="checkpoint token ID used for the standalone upstream input",
    )
    args = parser.parse_args()

    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "gemsim_amd" or target.arch != "gfx950":
        raise RuntimeError(f"unexpected Triton execution target: {target}")
    if DEVICE.type != "cpu":
        raise RuntimeError(
            f"gemsim_amd must expose a CPU staging device, got {DEVICE}"
        )

    decode_position = 0 if args.checkpoint else DECODE_POSITION
    sequence_length = decode_position + 1
    torch.manual_seed(173)
    hidden, hidden_guard = guarded_tensor(
        (1, HIDDEN_SIZE), torch.bfloat16, -41.0
    )
    residual, residual_guard = guarded_tensor(
        (1, HIDDEN_SIZE), torch.bfloat16, -43.0
    )
    if args.checkpoint:
        checkpoint_hidden, weights, provenance = checkpoint_inputs(args.token_id)
        hidden.copy_(checkpoint_hidden)
        residual.zero_()
    else:
        hidden.copy_(random_bf16((1, HIDDEN_SIZE), 0.125))
        residual.copy_(random_bf16((1, HIDDEN_SIZE), 0.25))
        weights = {
            "input_norm": random_bf16((HIDDEN_SIZE,), 0.05),
            "qkv_gate": random_bf16((QKV_GATE_SIZE, HIDDEN_SIZE), 0.03125),
            "q_norm": random_bf16((HEAD_DIM,), 0.05),
            "k_norm": random_bf16((HEAD_DIM,), 0.05),
            "out": random_bf16((HIDDEN_SIZE, Q_SIZE), 0.03125),
            "post_attention_norm": random_bf16((HIDDEN_SIZE,), 0.05),
            "gate_up": random_bf16((GATE_UP_SIZE, HIDDEN_SIZE), 0.03125),
            "down": random_bf16((HIDDEN_SIZE, INTERMEDIATE_SIZE), 0.03125),
        }
        provenance = {
            "mode": "deterministic_synthetic_weights_synthetic_cache",
            "seed": 173,
            "layer": 3,
            "oracle_scope": "local_stage_actual_upstream_tensor",
            "model_level_golden_verified": False,
        }
    hidden_before = hidden.clone()
    residual_before = residual.clone()
    weight_snapshots = {name: value.clone() for name, value in weights.items()}
    q_gamma = (weights["q_norm"].to(torch.float32) + 1.0).contiguous()
    k_gamma = (weights["k_norm"].to(torch.float32) + 1.0).contiguous()
    q_gamma_before = q_gamma.clone()
    k_gamma_before = k_gamma.clone()
    cos_sin_cache = make_cos_sin_cache(sequence_length)
    cos_sin_before = cos_sin_cache.clone()
    position = torch.tensor([decode_position], dtype=torch.int32, device=DEVICE)
    position_before = position.clone()

    kv_cache, kv_cache_guard = guarded_tensor(
        (1, KV_BLOCK_SIZE, NUM_KV_HEADS, KV_CONTENT_SIZE),
        torch.bfloat16,
        -47.0,
    )
    if args.checkpoint:
        kv_cache.zero_()
    else:
        kv_cache.copy_(random_bf16(tuple(kv_cache.shape), 0.125))
    kv_cache_initial = kv_cache.clone()
    slot = torch.tensor([decode_position], dtype=torch.int32, device=DEVICE)
    slot_before = slot.clone()

    guards = {
        "hidden_input": hidden_guard,
        "residual_input": residual_guard,
        "kv_cache": kv_cache_guard,
    }
    stages = {}

    input_norm, guards["input_norm"] = guarded_tensor(
        (1, HIDDEN_SIZE), torch.bfloat16, 51.0
    )
    input_residual, guards["input_residual"] = guarded_tensor(
        (1, HIDDEN_SIZE), torch.bfloat16, 53.0
    )
    expected_input_norm, expected_input_residual = fused_residual_reference(
        hidden_before, residual_before, weight_snapshots["input_norm"]
    )
    fused_residual_rms_norm(
        hidden,
        residual,
        weights["input_norm"],
        input_norm,
        input_residual,
        1,
    )
    stages["input_rms_norm"] = compare_stage(
        input_norm, expected_input_norm, 0.01, 0.01
    )
    stages["input_residual"] = compare_stage(
        input_residual,
        expected_input_residual,
        0.0,
        0.0,
        require_bitwise=True,
    )

    qkv_gate, guards["qkv_gate"] = guarded_tensor(
        (1, QKV_GATE_SIZE), torch.bfloat16, 55.0
    )
    expected_qkv_gate = bf16_matmul(
        input_norm, weight_snapshots["qkv_gate"]
    )
    dense_linear(input_norm, weights["qkv_gate"], qkv_gate)
    stages["qkv_gate_projection"] = compare_stage(
        qkv_gate, expected_qkv_gate, 0.03125, 0.03
    )

    q_gate = qkv_gate[:, :Q_GATE_SIZE]
    k_source = qkv_gate[:, Q_GATE_SIZE : Q_GATE_SIZE + KV_SIZE]
    v = qkv_gate[:, Q_GATE_SIZE + KV_SIZE :]
    expected_q, expected_k, expected_gate = qk_rope_reference(
        q_gate,
        k_source,
        weight_snapshots["q_norm"],
        weight_snapshots["k_norm"],
        cos_sin_cache,
        decode_position,
    )
    q, guards["q"] = guarded_tensor((1, Q_SIZE), torch.bfloat16, 57.0)
    k, guards["k"] = guarded_tensor((1, KV_SIZE), torch.bfloat16, 59.0)
    gate, guards["attention_gate"] = guarded_tensor(
        (1, Q_SIZE), torch.bfloat16, 61.0
    )
    qk_rmsnorm_rope_gate(
        q_gate,
        k_source,
        q_gamma,
        k_gamma,
        cos_sin_cache,
        position,
        q,
        k,
        gate,
    )
    stages["q_norm_rope"] = compare_stage(q, expected_q, 0.015625, 0.02)
    stages["k_norm_rope"] = compare_stage(k, expected_k, 0.015625, 0.02)
    stages["attention_gate_copy"] = compare_stage(
        gate, expected_gate, 0.0, 0.0, require_bitwise=True
    )

    expected_scores, expected_attention, expected_kv_cache = attention_reference(
        q,
        k,
        v,
        kv_cache_initial,
        decode_position,
        sequence_length,
    )
    scores, guards["attention_scores"] = guarded_tensor(
        (NUM_Q_HEADS, sequence_length), torch.float32, 63.0
    )
    attention, guards["attention_output"] = guarded_tensor(
        (1, Q_SIZE), torch.bfloat16, 65.0
    )
    gqa_decode_attention(
        q,
        k,
        v,
        kv_cache,
        slot,
        scores,
        attention,
        sequence_length,
    )
    stages["attention_scores"] = compare_stage(
        scores, expected_scores, 2.0e-4, 2.0e-4
    )
    stages["kv_cache"] = compare_stage(
        kv_cache,
        expected_kv_cache,
        0.0,
        0.0,
        require_bitwise=True,
    )
    stages["gqa_decode_attention"] = compare_stage(
        attention, expected_attention, 0.015625, 0.02
    )

    gated_attention, guards["sigmoid_gate"] = guarded_tensor(
        (1, Q_SIZE), torch.bfloat16, 67.0
    )
    expected_sigmoid_gate = torch.sigmoid(gate)
    expected_gated_attention = (
        attention.to(torch.float32)
        * expected_sigmoid_gate.to(torch.float32)
    ).to(torch.bfloat16)
    apply_sigmoid_output_gate(attention, gate, gated_attention)
    stages["sigmoid_output_gate"] = compare_stage(
        gated_attention, expected_gated_attention, 0.015625, 0.02
    )

    attention_out, guards["out_projection"] = guarded_tensor(
        (1, HIDDEN_SIZE), torch.bfloat16, 69.0
    )
    expected_attention_out = bf16_matmul(
        gated_attention, weight_snapshots["out"]
    )
    dense_linear(gated_attention, weights["out"], attention_out)
    stages["out_projection"] = compare_stage(
        attention_out, expected_attention_out, 0.03125, 0.03
    )

    post_norm, guards["post_attention_norm"] = guarded_tensor(
        (1, HIDDEN_SIZE), torch.bfloat16, 71.0
    )
    post_residual, guards["post_attention_residual"] = guarded_tensor(
        (1, HIDDEN_SIZE), torch.bfloat16, 73.0
    )
    expected_post_norm, expected_post_residual = fused_residual_reference(
        attention_out,
        input_residual,
        weight_snapshots["post_attention_norm"],
    )
    fused_residual_rms_norm(
        attention_out,
        input_residual,
        weights["post_attention_norm"],
        post_norm,
        post_residual,
        1,
    )
    stages["post_attention_rms_norm"] = compare_stage(
        post_norm, expected_post_norm, 0.01, 0.01
    )
    stages["post_attention_residual"] = compare_stage(
        post_residual,
        expected_post_residual,
        0.0,
        0.0,
        require_bitwise=True,
    )

    gate_up, guards["gate_up"] = guarded_tensor(
        (1, GATE_UP_SIZE), torch.bfloat16, 75.0
    )
    expected_gate_up = bf16_matmul(
        post_norm, weight_snapshots["gate_up"]
    )
    dense_linear(post_norm, weights["gate_up"], gate_up)
    stages["mlp_gate_up"] = compare_stage(
        gate_up, expected_gate_up, 0.03125, 0.03
    )

    activated, guards["mlp_activation"] = guarded_tensor(
        (1, INTERMEDIATE_SIZE), torch.bfloat16, 77.0
    )
    expected_gate = gate_up[:, :INTERMEDIATE_SIZE].to(torch.float32)
    expected_up = gate_up[:, INTERMEDIATE_SIZE:].to(torch.float32)
    expected_activated = (
        expected_gate * torch.sigmoid(expected_gate) * expected_up
    ).to(torch.bfloat16)
    scratch = torch.zeros((1,), dtype=torch.uint8, device=DEVICE)
    silu_and_mul(gate_up, activated, scratch, 1)
    stages["mlp_silu_and_mul"] = compare_stage(
        activated, expected_activated, 0.015625, 0.02
    )

    final_hidden, guards["mlp_down"] = guarded_tensor(
        (1, HIDDEN_SIZE), torch.bfloat16, 79.0
    )
    expected_final_hidden = bf16_matmul(
        activated, weight_snapshots["down"]
    )
    dense_linear(activated, weights["down"], final_hidden)
    stages["mlp_down"] = compare_stage(
        final_hidden, expected_final_hidden, 0.03125, 0.03
    )

    weights_unchanged = {
        name: bool(torch.equal(weights[name], snapshot))
        for name, snapshot in weight_snapshots.items()
    }
    immutable = {
        "hidden_input": bool(torch.equal(hidden, hidden_before)),
        "residual_input": bool(torch.equal(residual, residual_before)),
        "q_gamma": bool(torch.equal(q_gamma, q_gamma_before)),
        "k_gamma": bool(torch.equal(k_gamma, k_gamma_before)),
        "cos_sin_cache": bool(torch.equal(cos_sin_cache, cos_sin_before)),
        "position": bool(torch.equal(position, position_before)),
        "slot": bool(torch.equal(slot, slot_before)),
        "kv_history": bool(
            torch.equal(
                kv_cache[:, :decode_position],
                kv_cache_initial[:, :decode_position],
            )
        ),
        "kv_unused_tail": bool(
            torch.equal(
                kv_cache[:, sequence_length:],
                kv_cache_initial[:, sequence_length:],
            )
        ),
    }
    guards_unchanged = {
        name: guard_unchanged(guard) for name, guard in guards.items()
    }
    cache_nonzero = {
        "initial": int(torch.count_nonzero(kv_cache_initial).item()),
        "final": int(torch.count_nonzero(kv_cache).item()),
        "history": int(
            torch.count_nonzero(
                kv_cache_initial[:, :decode_position]
            ).item()
        ),
    }
    if args.checkpoint:
        cache_initialization = "empty"
        cache_initialization_valid = (
            cache_nonzero["initial"] == 0
            and cache_nonzero["history"] == 0
            and cache_nonzero["final"] > 0
        )
    else:
        cache_initialization = "deterministic_nonzero_synthetic"
        cache_initialization_valid = all(
            value > 0 for value in cache_nonzero.values()
        )
    correct = (
        all(stage["correct"] for stage in stages.values())
        and all(weights_unchanged.values())
        and all(immutable.values())
        and all(guards_unchanged.values())
        and cache_initialization_valid
    )
    payload = {
        "schema": "amdgpu-sim.triton-qwen35-layer3-decode.v1",
        "backend": target.backend,
        "arch": target.arch,
        "model": "Qwen/Qwen3.5-0.8B",
        "layer": 3,
        "layer_type": "full_attention",
        "tokens": 1,
        "dtype": "bfloat16",
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "qkv_gate_layout": "per-q-head [q256|raw_gate256], then k2x256, v2x256",
        "qkv_gate_shape": [1, QKV_GATE_SIZE],
        "rotary_dim": ROTARY_DIM,
        "rope_theta": ROPE_THETA,
        "rope_style": "partial NeoX; text decode position",
        "rms_norm_epsilon": EPSILON,
        "kv_cache_shape": [
            1,
            KV_BLOCK_SIZE,
            NUM_KV_HEADS,
            KV_CONTENT_SIZE,
        ],
        "kv_cache_logical_shape": [
            1,
            NUM_KV_HEADS,
            KV_BLOCK_SIZE,
            KV_CONTENT_SIZE,
        ],
        "kv_cache_layout": "vLLM default physical NHD [block,slot,kv_head,k256|v256]",
        "decode_position": decode_position,
        "sequence_length": sequence_length,
        "stage_order": list(stages),
        "stages": stages,
        "weights_unchanged": weights_unchanged,
        "immutable": immutable,
        "guards_unchanged": guards_unchanged,
        "cache_nonzero": cache_nonzero,
        "cache_initialization": cache_initialization,
        "cache_initialization_valid": cache_initialization_valid,
        "provenance": provenance,
        "input_fused_residual_bitwise_required": True,
        "post_attention_fused_residual_bitwise_required": True,
        "residual_semantics": (
            "input_residual=bf16(hidden.float+prior_residual.float); "
            "post_residual=bf16(attention_out.float+input_residual.float)"
        ),
        "final_hidden_sha256": tensor_sha256(final_hidden),
        "expected_final_hidden_sha256": tensor_sha256(expected_final_hidden),
        "final_residual_sha256": tensor_sha256(post_residual),
        "expected_final_residual_sha256": tensor_sha256(
            expected_post_residual
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
        "persistent_cache": "qwen35-layer3-decode",
        "output_correct": correct,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
