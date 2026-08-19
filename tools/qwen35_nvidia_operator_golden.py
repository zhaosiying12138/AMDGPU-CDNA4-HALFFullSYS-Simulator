#!/usr/bin/env python3
"""Generate the diagnostic operator-boundary golden for Qwen3.5 layer 0.

This is deliberately separate from the production runner.  It executes the
same independent, pure-Torch CUDA formulas as ``qwen35_nvidia_golden.py`` but
publishes every useful boundary in the first (GDN) decoder layer.  The SGLang
diagnostic hook consumes this package and stops at the first boundary that
differs; no golden value is ever fed back into the target.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen35_nvidia_golden as backbone  # noqa: E402


TOKEN_IDS = [248044, 266]
EPSILON = 1.0e-6
SCHEMA = "amdgpu-sim.qwen35-nvidia-operator-golden.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(
        order="C"
    )


def _record(value: torch.Tensor) -> dict:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": _sha256_bytes(_tensor_bytes(value)),
        "finite": bool(torch.all(torch.isfinite(value.float())).item()),
    }


def _bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x.float(), weight.float().T).to(torch.bfloat16)


def _compute(hidden_cpu: torch.Tensor, weights_cpu: dict[str, torch.Tensor], device):
    """Return all layer-0 boundaries in execution order.

    The arithmetic and rounding points intentionally mirror the sequence path
    in qwen35_nvidia_golden.py.  Keeping the intermediate tensors here makes
    the first mismatch attributable to a concrete operation rather than to a
    whole-layer checksum.
    """

    hidden = hidden_cpu.to(device)
    w = {name: value.to(device) for name, value in weights_cpu.items()}
    result: dict[str, torch.Tensor] = {"hidden_input": hidden}

    x = hidden.float()
    variance = torch.mean(x * x, dim=-1, keepdim=True)
    normalized = (
        x * torch.rsqrt(variance + EPSILON) * (1.0 + w["input_norm"].float())
    ).to(torch.bfloat16)
    result["input_rms_norm"] = normalized

    qkvz = _bf16_linear(normalized, w["qkvz"])
    ba = _bf16_linear(normalized, w["ba"])
    result["qkvz_projection"] = qkvz
    result["ba_projection"] = ba

    # This is the exact layout emitted by the Triton split/reshape kernel:
    # [all q | all k | all v] with z, b and a in separate outputs.
    num_heads = backbone.NUM_HEADS
    head_dim = backbone.HEAD_DIM
    qkv_dim = backbone.QKV_DIM
    z = qkvz[:, qkv_dim:].reshape(-1, num_heads, head_dim).contiguous()
    mixed_qkv = qkvz[:, :qkv_dim].contiguous()
    b = ba[:, :num_heads].contiguous()
    a = ba[:, num_heads:].contiguous()
    result["mixed_qkv"] = mixed_qkv
    result["z"] = z
    result["b"] = b
    result["a"] = a

    # Causal depthwise conv, preserving the selected cache line exactly as the
    # independent sequence oracle does.
    conv_state = torch.zeros(
        (backbone.CONV_CACHE_LINES, qkv_dim, backbone.CONV_STATE_WIDTH),
        dtype=torch.bfloat16,
        device=device,
    )
    selected = conv_state[backbone.SELECTED_CONV_STATE]
    conv_weight = w["conv"].float()
    conv_rows = []
    for token in range(normalized.shape[0]):
        current = mixed_qkv[token].float()
        accumulator = (
            selected[:, 0].float() * conv_weight[:, 0]
            + selected[:, 1].float() * conv_weight[:, 1]
            + selected[:, 2].float() * conv_weight[:, 2]
            + current * conv_weight[:, 3]
        )
        conv_rows.append((accumulator * torch.sigmoid(accumulator)).to(torch.bfloat16))
        selected = torch.stack(
            (selected[:, 1], selected[:, 2], mixed_qkv[token]), dim=1
        )
    conv_state[backbone.SELECTED_CONV_STATE] = selected
    mixed_after_conv = torch.stack(conv_rows, dim=0).contiguous()
    result["gdn_conv_output"] = mixed_after_conv
    # The SGLang operator boundary exposes only the active request's cache
    # slot.  Store that selected state, not the oracle's surrounding synthetic
    # three-slot envelope.
    result["conv_state"] = conv_state[backbone.SELECTED_CONV_STATE]

    # Gated-delta recurrence.  Keep the recurrent state in FP32, matching the
    # pinned formula, while the observable output is rounded to BF16.
    recurrent_state = torch.zeros(
        (num_heads, head_dim, head_dim), dtype=torch.float32, device=device
    )
    recurrent_rows = []
    query_rows = []
    key_rows = []
    value_rows = []
    beta_rows = []
    decay_rows = []
    g_rows = []
    for token in range(normalized.shape[0]):
        mixed = mixed_after_conv[token].float().view(-1)
        query = mixed[: num_heads * head_dim].view(num_heads, head_dim)
        key = mixed[num_heads * head_dim : 2 * num_heads * head_dim].view(
            num_heads, head_dim
        )
        value = mixed[2 * num_heads * head_dim :].view(num_heads, head_dim)
        query = query * torch.rsqrt(
            torch.sum(query * query, dim=-1, keepdim=True) + EPSILON
        )
        key = key * torch.rsqrt(
            torch.sum(key * key, dim=-1, keepdim=True) + EPSILON
        )
        query = query * (head_dim**-0.5)
        softplus_input = a[token].float() + w["dt_bias"].float()
        softplus = torch.where(
            softplus_input <= 20.0,
            torch.log1p(torch.exp(softplus_input)),
            softplus_input,
        )
        g = -torch.exp(w["a_log"]) * softplus
        decay = torch.exp(g)
        beta = torch.sigmoid(b[token].float()).to(torch.bfloat16).float()
        recurrent_state = recurrent_state * decay.view(num_heads, 1, 1)
        prediction = torch.sum(recurrent_state * key[:, None, :], dim=-1)
        delta = (value - prediction) * beta[:, None]
        recurrent_state = recurrent_state + delta[:, :, None] * key[:, None, :]
        recurrent_rows.append(
            torch.sum(recurrent_state * query[:, None, :], dim=-1).to(torch.bfloat16)
        )
        query_rows.append(query)
        key_rows.append(key)
        value_rows.append(value)
        beta_rows.append(beta)
        decay_rows.append(decay)
        g_rows.append(g)
    recurrent_output = torch.stack(recurrent_rows, dim=0).contiguous()
    q_raw, k_raw, v_raw = mixed_after_conv.split(
        (num_heads * head_dim, num_heads * head_dim, num_heads * head_dim),
        dim=-1,
    )
    result["gdn_q_raw"] = q_raw.reshape(1, -1, num_heads, head_dim)
    result["gdn_k_raw"] = k_raw.reshape(1, -1, num_heads, head_dim)
    result["gdn_v_raw"] = v_raw.reshape(1, -1, num_heads, head_dim)
    result["gdn_g"] = torch.stack(g_rows, dim=0).reshape(1, -1, num_heads)
    result["gdn_beta_output"] = torch.stack(beta_rows, dim=0).reshape(
        1, -1, num_heads
    )
    result["gdn_recurrent_output"] = recurrent_output
    result["recurrent_state"] = recurrent_state

    recurrent_float = recurrent_output.float()
    variance = torch.mean(recurrent_float * recurrent_float, dim=-1, keepdim=True)
    normalized_recurrent = recurrent_float * torch.rsqrt(variance + EPSILON)
    z_float = z.float()
    gated = (
        normalized_recurrent
        * w["output_norm"].float()
        * (z_float * torch.sigmoid(z_float))
    ).to(torch.bfloat16)
    result["output_rms_norm_gate"] = gated
    attention_output = _bf16_linear(gated.reshape(-1, backbone.Z_DIM), w["gdn_out"])
    result["gdn_out_projection"] = attention_output

    summed = attention_output.float() + hidden.float()
    residual = summed.to(torch.bfloat16)
    variance = torch.mean(summed * summed, dim=-1, keepdim=True)
    post_norm = (
        summed
        * torch.rsqrt(variance + EPSILON)
        * (1.0 + w["post_attention_norm"].float())
    ).to(torch.bfloat16)
    result["post_attention_rms_norm"] = post_norm
    result["post_attention_residual"] = residual
    gate_up = _bf16_linear(post_norm, w["gate_up"])
    result["mlp_gate_up"] = gate_up
    gate = gate_up[:, : backbone.INTERMEDIATE_SIZE].float()
    up = gate_up[:, backbone.INTERMEDIATE_SIZE :].float()
    activated = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
    result["mlp_silu_and_mul"] = activated
    result["mlp_down"] = _bf16_linear(activated, w["down"])
    return {name: value.detach().cpu().contiguous() for name, value in result.items()}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=backbone.DEFAULT_MODEL_DIR)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/qwen35-nvidia-operator-golden/20260819-prefill2-layer0-v3",
    )
    p.add_argument("--device", type=int, default=0)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {args.output_dir}")
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise RuntimeError("requested CUDA device is unavailable")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)

    model_dir = args.model_dir.resolve(strict=True)
    config, _index, manifest, shard, artifact_records = backbone.validate_checkpoint(model_dir)
    hidden, weights, source_hashes = backbone.load_inputs(model_dir, TOKEN_IDS, shard)
    result = _compute(hidden, weights, torch.device("cuda", args.device))

    # Cross-check the terminal boundaries against the already frozen prefill
    # golden before publishing this more granular package.
    frozen = ROOT / "artifacts/qwen35-nvidia-golden/20260812-prefill2-max24-v1/results.safetensors"
    from safetensors import safe_open

    with safe_open(frozen, framework="pt", device="cpu") as source:
        for local, frozen_name in (
            ("hidden_input", "hidden_input"),
            ("mlp_down", "layers.0.returned_hidden"),
            ("post_attention_residual", "layers.0.returned_residual"),
        ):
            expected = source.get_tensor(frozen_name)
            if not torch.equal(result[local], expected):
                raise RuntimeError(f"operator formula disagrees with frozen golden: {local}")

    args.output_dir.mkdir(mode=0o700, parents=True)
    tensor_path = args.output_dir / "results.safetensors"
    save_file(result, str(tensor_path), metadata={"schema": SCHEMA})
    tensor_records = {name: _record(value) for name, value in result.items()}
    metadata = {
        "schema": SCHEMA,
        "kind": "independent_torch_cuda_layer0_operator_boundaries",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": backbone.PINNED_MODEL_ID,
            "revision": backbone.PINNED_REVISION,
            "directory": str(model_dir),
            "manifest_files": artifact_records,
            "shard": shard.name,
            "config_sha256": hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
        },
        "case": {
            "token_ids": TOKEN_IDS,
            "positions": [0, 1],
            "layer": 0,
            "layer_type": "linear_attention",
            "cache": "empty_per_layer",
            "operator_order": list(result),
            "rounding": "pinned qwen35_nvidia_golden sequence formulas",
        },
        "selected_checkpoint_tensor_sha256": source_hashes,
        "results": tensor_records,
        "results_file_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        "all_results_finite": all(item["finite"] for item in tensor_records.values()),
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps({"output": str(args.output_dir), "operators": list(result)}, indent=2))
    return 0


if __name__ == "__main__":
    configured = Path("/home/zhaosiying/miniforge3/envs/triton-dev/bin/python3").resolve()
    if Path(sys.executable).resolve() != configured and configured.is_file():
        import os

        os.execv(str(configured), [str(configured), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit(main())
