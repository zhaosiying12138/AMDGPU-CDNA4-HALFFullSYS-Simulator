#!/usr/bin/env python3
"""Fail-fast layer differential for the real SGLang Qwen3.5 TP1 path.

The gate is diagnostic only.  It observes the normal SGLang module calls with
global PyTorch forward hooks, copies already-computed tensors to the host for
comparison, and never feeds a golden value back into target execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save as save_safetensors


SCHEMA = "amdgpu-sim.qwen35-sglang-layer-gate.v2"
PROMPT_TOKEN_IDS = (248044, 266)
TOKEN_POSITIONS = (0, 1)
NUM_LAYERS = 24
HIDDEN_SHAPE = (2, 1024)
ATOL = 0.03125
RTOL = 0.03
MAX_RELATIVE_L2 = 0.03
MIN_COSINE = 0.98
GOLDEN_FILES = {
    "metadata.json": (
        153970,
        "ff326833bc2a47f760c240af5f441f48a187e686e523a4c0778185ce392d2251",
    ),
    "results.safetensors": (
        21284456,
        "c401c34db3f137ad4b2e371b32e3bcc9c796ba56a2101d50e7e8fc927091cbe7",
    ),
}
MODEL_MODULE = "sglang.srt.models.qwen3_5"
MODEL_CLASS = "Qwen3_5ForCausalLM"
MODEL_MODULE_PREFIX = "sglang.srt.models.qwen3"
MODEL_CLASSES = {
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
}
LAYER_CLASSES = {
    "Qwen3_5LinearDecoderLayer",
    "Qwen3_5AttentionDecoderLayer",
}


class LayerGateError(RuntimeError):
    """The diagnostic contract or a numerical comparison failed."""


class FirstNumericalMismatch(LayerGateError):
    """The first observed SGLang layer boundary differs from golden."""


OPERATOR_GOLDEN_SCHEMA = "amdgpu-sim.qwen35-nvidia-operator-golden.v1"
OPERATOR_GOLDEN_DEFAULT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/qwen35-nvidia-operator-golden/20260819-prefill2-layer0-v3"
)
OPERATOR_ORDER = (
    "hidden_input",
    "input_rms_norm",
    "qkvz_projection",
    "ba_projection",
    "mixed_qkv",
    "z",
    "b",
    "a",
    "gdn_conv_output",
    "conv_state",
    "gdn_q_raw",
    "gdn_k_raw",
    "gdn_v_raw",
    "gdn_g",
    "gdn_beta_output",
    "gdn_recurrent_output",
    "recurrent_state",
    "output_rms_norm_gate",
    "gdn_out_projection",
    "post_attention_rms_norm",
    "post_attention_residual",
    "mlp_gate_up",
    "mlp_silu_and_mul",
    "mlp_down",
)
OPERATOR_TOLERANCES = {
    "hidden_input": (0.0, 0.0, 0.0, 1.0),
    "input_rms_norm": (0.015625, 0.02, 0.02, 0.99),
    "qkvz_projection": (0.03125, 0.03, 0.03, 0.98),
    "ba_projection": (0.03125, 0.03, 0.03, 0.98),
    # These are views/slices of the preceding GEMM results.  They inherit
    # the GEMM's cross-device BF16 tolerance; demanding byte identity here
    # would turn an otherwise valid projection rounding difference into a
    # false first divergence.
    "mixed_qkv": (0.03125, 0.03, 0.03, 0.98),
    "z": (0.03125, 0.03, 0.03, 0.98),
    "b": (0.03125, 0.03, 0.03, 0.98),
    "a": (0.03125, 0.03, 0.03, 0.98),
    "gdn_conv_output": (0.05, 0.01, 0.03, 0.98),
    "conv_state": (0.03125, 0.03, 0.03, 0.98),
    "gdn_q_raw": (0.05, 0.01, 0.03, 0.98),
    "gdn_k_raw": (0.05, 0.01, 0.03, 0.98),
    "gdn_v_raw": (0.05, 0.01, 0.03, 0.98),
    "gdn_g": (1.0e-5, 1.0e-4, 1.0e-4, 0.9999),
    "gdn_beta_output": (1.0e-5, 1.0e-4, 1.0e-4, 0.9999),
    "gdn_recurrent_output": (0.015625, 0.02, 0.03, 0.98),
    # The cache state is stored in FP32 but is a sum of outer products of
    # BF16-rounded intermediates (l2norm, A, u and v are each cast to BF16
    # inside the kernels), so its per-element error scale is BF16 epsilon,
    # not FP32.  An FP64 oracle over the captured inputs showed both
    # architectures land within ~1e-3 of the true value -- with the NVIDIA
    # reference farther from the oracle than the simulator at the worst
    # elements -- so demanding 1e-4 here flagged legitimate cross-device
    # rounding as a first divergence.
    "recurrent_state": (0.00390625, 0.03, 0.03, 0.98),
    "output_rms_norm_gate": (0.01, 0.01, 0.03, 0.98),
    "gdn_out_projection": (0.03125, 0.03, 0.03, 0.98),
    "post_attention_rms_norm": (0.01, 0.01, 0.03, 0.98),
    "post_attention_residual": (0.03125, 0.03, 0.03, 0.98),
    "mlp_gate_up": (0.03125, 0.03, 0.03, 0.98),
    "mlp_silu_and_mul": (0.015625, 0.02, 0.03, 0.98),
    "mlp_down": (0.03125, 0.03, 0.03, 0.98),
}


@dataclass
class OperatorGolden:
    tensors: dict[str, torch.Tensor]
    directory: Path
    file_records: dict[str, dict[str, object]]


def load_operator_golden(directory: Path) -> OperatorGolden:
    """Load and identity-check the separate operator-boundary oracle."""

    directory = directory.resolve(strict=True)
    if not directory.is_dir() or directory.is_symlink():
        raise LayerGateError("operator golden directory is unsafe")
    metadata_path = directory / "metadata.json"
    tensor_path = directory / "results.safetensors"
    if metadata_path.is_symlink() or tensor_path.is_symlink():
        raise LayerGateError("operator golden files may not be symlinks")
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    if metadata.get("schema") != OPERATOR_GOLDEN_SCHEMA:
        raise LayerGateError("operator golden schema mismatch")
    if metadata.get("case", {}).get("token_ids") != list(PROMPT_TOKEN_IDS):
        raise LayerGateError("operator golden prompt mismatch")
    if metadata.get("case", {}).get("positions") != list(TOKEN_POSITIONS):
        raise LayerGateError("operator golden positions mismatch")
    if metadata.get("case", {}).get("operator_order") != list(OPERATOR_ORDER):
        raise LayerGateError("operator golden order mismatch")
    records = metadata.get("results", {})
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as source:
        if set(source.keys()) != set(OPERATOR_ORDER):
            raise LayerGateError("operator golden tensor set mismatch")
        for name in OPERATOR_ORDER:
            value = source.get_tensor(name).clone().contiguous()
            record = records.get(name, {})
            if (
                record.get("sha256") != _tensor_sha256(value)
                or record.get("shape") != list(value.shape)
                or record.get("dtype") != str(value.dtype).removeprefix("torch.")
                or record.get("finite") is not True
            ):
                raise LayerGateError(f"operator golden tensor identity mismatch: {name}")
            tensors[name] = value
    return OperatorGolden(
        tensors=tensors,
        directory=directory,
        file_records={
            "metadata.json": {
                "bytes": metadata_path.stat().st_size,
                "sha256": _sha256_file(metadata_path),
            },
            "results.safetensors": {
                "bytes": tensor_path.stat().st_size,
                "sha256": _sha256_file(tensor_path),
            },
        },
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(
        order="C"
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(value)).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short evidence write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: object) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC,
        0o600,
    )
    try:
        payload = _canonical_json(value)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short comparison log write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cpu_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise LayerGateError(f"{name} is not a tensor")
    return value.detach().cpu().contiguous()


def compare_tensor(
    actual: object,
    expected: torch.Tensor,
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
    max_relative_l2: float = MAX_RELATIVE_L2,
    min_cosine_similarity: float = MIN_COSINE,
) -> dict[str, Any]:
    """Apply the formal online-layer BF16 tolerance and fail closed."""

    actual_cpu = _cpu_tensor(actual, "actual")
    expected_cpu = _cpu_tensor(expected, "expected")
    contract_correct = bool(
        actual_cpu.dtype == expected_cpu.dtype
        and tuple(actual_cpu.shape) == tuple(expected_cpu.shape)
    )
    result: dict[str, Any] = {
        "actual_dtype": str(actual_cpu.dtype).removeprefix("torch."),
        "actual_shape": list(actual_cpu.shape),
        "expected_dtype": str(expected_cpu.dtype).removeprefix("torch."),
        "expected_shape": list(expected_cpu.shape),
        "contract_correct": contract_correct,
        "atol": atol,
        "rtol": rtol,
        "max_relative_l2": max_relative_l2,
        "min_cosine_similarity": min_cosine_similarity,
    }
    if not contract_correct:
        result.update(
            {
                "actual_sha256": _tensor_sha256(actual_cpu),
                "expected_sha256": _tensor_sha256(expected_cpu),
                "mismatch_count": None,
                "nonfinite_count": None,
                "max_abs_error": None,
                "relative_l2_error": None,
                "cosine_similarity": None,
                "correct": False,
            }
        )
        return result

    actual_float = actual_cpu.float()
    expected_float = expected_cpu.float()
    error = torch.abs(actual_float - expected_float)
    finite = (
        torch.isfinite(actual_float)
        & torch.isfinite(expected_float)
        & torch.isfinite(error)
    )
    mismatch = (~finite) | (error > atol + rtol * torch.abs(expected_float))
    mismatch_count = int(torch.count_nonzero(mismatch).item())
    nonfinite_count = int(torch.count_nonzero(~finite).item())
    expected_norm = float(torch.linalg.vector_norm(expected_float).item())
    actual_norm = float(torch.linalg.vector_norm(actual_float).item())
    relative_l2 = float(torch.linalg.vector_norm(error).item()) / max(
        expected_norm, 1.0e-30
    )
    if actual_norm == 0.0 and expected_norm == 0.0:
        cosine = 1.0
    elif actual_norm == 0.0 or expected_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(
            torch.sum(actual_float * expected_float).item()
            / (actual_norm * expected_norm)
        )
    result.update(
        {
            "actual_sha256": _tensor_sha256(actual_cpu),
            "expected_sha256": _tensor_sha256(expected_cpu),
            "mismatch_count": mismatch_count,
            "nonfinite_count": nonfinite_count,
            "max_abs_error": float(torch.where(finite, error, 0.0).max().item()),
            "relative_l2_error": relative_l2,
            "cosine_similarity": cosine,
            "correct": bool(
                mismatch_count == 0
                and nonfinite_count == 0
                and relative_l2 <= max_relative_l2
                and cosine >= min_cosine_similarity
            ),
        }
    )
    return result


def _normalize_positions(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, torch.Tensor):
        return None
    rows = value.detach().cpu().to(torch.int64)
    if rows.ndim == 1:
        return tuple(int(item) for item in rows.tolist())
    if rows.ndim == 2 and rows.shape[0] > 0:
        first = rows[0]
        if bool(torch.all(rows == first).item()):
            return tuple(int(item) for item in first.tolist())
        # SGLang uses [3, tokens] for MRoPE, while a few runner paths expose
        # the equivalent [tokens, 3] view.  Text-only Qwen positions repeat
        # one-dimensional positions across the three MRoPE rows/columns.
        if rows.shape[1] > 0:
            first_column = rows[:, 0]
            if bool(torch.all(rows == first_column[:, None]).item()):
                return tuple(int(item) for item in first_column.tolist())
    return None


def _argument(args: tuple[object, ...], kwargs: dict[str, object], name: str, index: int):
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else None


def _output_pair(value: object) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise LayerGateError("decoder layer output is not a hidden/residual pair")
    hidden = _cpu_tensor(value[0], "layer hidden output")
    residual = _cpu_tensor(value[1], "layer residual output")
    return hidden, residual


@dataclass
class Golden:
    tensors: dict[str, torch.Tensor]
    directory: Path
    file_records: dict[str, dict[str, object]]


def load_golden(directory: Path) -> Golden:
    directory = directory.resolve(strict=True)
    if not directory.is_dir() or directory.is_symlink():
        raise LayerGateError("golden directory is unsafe")
    observed_files = sorted(item.name for item in directory.iterdir())
    if observed_files != sorted(GOLDEN_FILES):
        raise LayerGateError(f"golden file set mismatch: {observed_files}")
    records: dict[str, dict[str, object]] = {}
    for name, (expected_bytes, expected_sha) in GOLDEN_FILES.items():
        path = directory / name
        actual_sha = _sha256_file(path)
        if path.stat().st_size != expected_bytes or actual_sha != expected_sha:
            raise LayerGateError(f"golden identity mismatch: {name}")
        records[name] = {
            "bytes": expected_bytes,
            "sha256": expected_sha,
        }

    metadata = json.loads((directory / "metadata.json").read_text(encoding="ascii"))
    if (
        metadata.get("schema") != "amdgpu-sim.qwen35-nvidia-prefill-golden.v1"
        or metadata.get("case", {}).get("token_ids") != list(PROMPT_TOKEN_IDS)
        or metadata.get("case", {}).get("positions") != list(TOKEN_POSITIONS)
        or metadata.get("case", {}).get("max_layers") != NUM_LAYERS
        or metadata.get("all_results_finite") is not True
    ):
        raise LayerGateError("golden metadata contract mismatch")

    required = {"hidden_input"}
    for layer in range(NUM_LAYERS):
        required.add(f"layers.{layer}.returned_hidden")
        required.add(f"layers.{layer}.returned_residual")
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(
        directory / "results.safetensors", framework="pt", device="cpu"
    ) as source:
        missing = required - set(source.keys())
        if missing:
            raise LayerGateError(f"golden tensors are missing: {sorted(missing)}")
        for key in sorted(required):
            value = source.get_tensor(key).clone().contiguous()
            if value.dtype != torch.bfloat16 or tuple(value.shape) != HIDDEN_SHAPE:
                raise LayerGateError(f"golden tensor contract mismatch: {key}")
            if not bool(torch.all(torch.isfinite(value.float())).item()):
                raise LayerGateError(f"golden tensor is nonfinite: {key}")
            record = metadata.get("results", {}).get(key, {})
            if (
                record.get("sha256") != _tensor_sha256(value)
                or record.get("shape") != list(value.shape)
                or record.get("dtype") != "bfloat16"
            ):
                raise LayerGateError(f"golden tensor identity mismatch: {key}")
            tensors[key] = value
    return Golden(tensors=tensors, directory=directory, file_records=records)


class LayerGate:
    def __init__(self, output: Path, golden: Golden):
        self.output = output
        self.golden = golden
        self.active = False
        self.model_id: int | None = None
        self.next_layer = 0
        self.before: dict[str, torch.Tensor | None] = {}
        self._published_mismatch = False
        self._layer_handles: list[object] = []
        self.prompt_seen = False
        self.operator_golden = load_operator_golden(
            Path(os.environ.get("SAGR_QWEN35_OPERATOR_GOLDEN", OPERATOR_GOLDEN_DEFAULT))
        )
        self.operator_active = False
        self.operator_next = 0
        self.operator_module_paths: dict[int, str] = {}
        self.operator_last: dict[str, torch.Tensor] = {}
        self.operator_inputs: dict[str, torch.Tensor] = {}
        self.operator_metadata: dict[str, object] = {}

    def arm_operator_layer(
        self, layer: torch.nn.Module, _args: tuple[object, ...]
    ) -> None:
        """Arm the layer-0 operator gate before any child module executes."""

        if self.operator_active or not self.prompt_seen:
            return
        if getattr(layer, "layer_id", None) != 0:
            return
        self.operator_module_paths = {
            id(module): (name or "<layer>")
            for name, module in layer.named_modules()
        }
        if len(_patched_functions) != 5:
            _install_function_wrappers()
        if len(_patched_functions) != 5:
            raise LayerGateError(
                "SGLang operator gate could not install all five function wrappers"
            )
        # SGLang invokes Qwen3.5 decoder layers with keyword arguments.  The
        # pinned PyTorch process-global pre-hook intentionally receives only
        # positional args, so the hidden input cannot be read here.  Arm the
        # session now, then capture the real hidden tensor at the first child
        # module (input_layernorm), whose kwargs-aware post-hook has it.
        if not self.active:
            self._start(layer)
        self.operator_active = True
        self.operator_next = 0

    @staticmethod
    def _first_tensor(value: object) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = LayerGate._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    @staticmethod
    def _normalize_operator(name: str, value: torch.Tensor) -> torch.Tensor:
        value = value.detach().cpu().contiguous()
        # GDN returns a leading batch dimension for the recurrent state.  The
        # oracle stores the single selected state without that envelope.
        if name == "recurrent_state" and tuple(value.shape) == (1, 16, 128, 128):
            value = value[0].contiguous()
        if name == "conv_state" and tuple(value.shape) == (1, 6144, 3):
            value = value[0].contiguous()
        return value

    @staticmethod
    def _module_capture_inputs(
        module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[tuple[object, ...], list[str]]:
        """Return call values plus direct parameters/buffers for replay.

        The tensors are only serialized on a mismatch.  Keeping the direct
        module parameters here makes an RMSNorm/GEMM capsule independently
        runnable instead of requiring the entire loaded model checkpoint.
        """

        values: list[object] = list(args)
        names: list[str] = [f"arg_{index}" for index in range(len(args))]
        for key, value in kwargs.items():
            values.append(value)
            names.append(f"kw_{key}")
        for key, value in module.named_parameters(recurse=False):
            values.append(value)
            names.append(f"parameter_{key}")
        for key, value in module.named_buffers(recurse=False):
            values.append(value)
            names.append(f"buffer_{key}")
        return tuple(values), names

    def _record_operator(
        self,
        name: str,
        actual: object,
        *,
        inputs: tuple[object, ...] | list[object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if not self.operator_active:
            return
        if self.operator_next >= len(OPERATOR_ORDER):
            raise LayerGateError("operator gate observed an unexpected extra boundary")
        expected_name = OPERATOR_ORDER[self.operator_next]
        if name != expected_name:
            # A missing boundary is just as unsafe as a wrong value: do not
            # let a later tensor accidentally look like a passing result.
            raise LayerGateError(
                f"operator order mismatch: expected={expected_name} actual={name}"
            )
        actual_tensor = self._first_tensor(actual)
        if actual_tensor is None:
            raise LayerGateError(f"operator boundary {name} did not return a tensor")
        actual_tensor = self._normalize_operator(name, actual_tensor)
        expected = self.operator_golden.tensors[name]
        atol, rtol, relative_l2, cosine = OPERATOR_TOLERANCES[name]
        comparison = compare_tensor(
            actual_tensor,
            expected,
            atol=atol,
            rtol=rtol,
            max_relative_l2=relative_l2,
            min_cosine_similarity=cosine,
        )
        record = {
            "schema": SCHEMA,
            "phase": "operator",
            "layer": 0,
            "operator": name,
            "ordinal": self.operator_next,
            "comparison": comparison,
            "correct": comparison["correct"],
        }
        if metadata:
            record["metadata"] = metadata
        if not comparison["correct"]:
            capture: dict[str, torch.Tensor | None] = {
                "actual": actual_tensor,
                "expected": expected,
            }
            if inputs:
                for index, value in enumerate(inputs):
                    tensor = self._first_tensor(value)
                    if tensor is not None:
                        capture[f"input_{index}"] = tensor
            self.operator_metadata = {
                "operator": name,
                "ordinal": self.operator_next,
                "module_path": metadata.get("module_path") if metadata else None,
                "triton_launch_log": os.environ.get("SAGR_TRITON_LAUNCH_LOG"),
                "triton_launch_lines": self._line_count(
                    os.environ.get("SAGR_TRITON_LAUNCH_LOG")
                ),
                "input_names": (
                    metadata.get("input_names") if metadata else None
                ),
            }
            self._mismatch(record, capture, operator=True)
        self.operator_last[name] = actual_tensor
        _append_jsonl(self.output / "operator-comparisons.jsonl", record)
        self.operator_next += 1

    @staticmethod
    def _line_count(path: str | None) -> int | None:
        if not path:
            return None
        try:
            with open(path, "rb") as stream:
                return sum(1 for _ in stream)
        except OSError:
            return None

    def observe_module(
        self,
        module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        output: object,
    ) -> None:
        if not self.operator_active:
            return
        path = self.operator_module_paths.get(id(module))
        if path is None:
            return
        capture_inputs, capture_names = self._module_capture_inputs(
            module, args, kwargs
        )
        name = None
        if path == "input_layernorm":
            # This is the first kwargs-aware child boundary.  It supplies the
            # hidden input that the layer pre-hook cannot see.
            self._record_operator(
                "hidden_input",
                args[0] if args else kwargs.get("hidden_states"),
                inputs=capture_inputs,
                metadata={"module_path": path, "input_names": capture_names},
            )
            name = "input_rms_norm"
        elif path == "linear_attn.in_proj_qkvz":
            name = "qkvz_projection"
        elif path == "linear_attn.in_proj_ba":
            name = "ba_projection"
        elif path == "linear_attn.norm":
            name = "output_rms_norm_gate"
        elif path == "linear_attn.out_proj":
            name = "gdn_out_projection"
        elif path == "post_attention_layernorm":
            # The fused norm returns (normalized, residual); record both
            # boundaries in the same post-hook, preserving their order.
            self._record_operator(
                "post_attention_rms_norm",
                self._first_tensor(output),
                inputs=capture_inputs,
                metadata={"module_path": path, "input_names": capture_names},
            )
            residual = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            self._record_operator(
                "post_attention_residual",
                residual,
                inputs=capture_inputs,
                metadata={"module_path": path, "input_names": capture_names},
            )
            return
        elif path == "mlp.gate_up_proj":
            name = "mlp_gate_up"
        elif path == "mlp.act_fn":
            name = "mlp_silu_and_mul"
        elif path == "mlp.down_proj":
            name = "mlp_down"
        if name is not None:
            self._record_operator(
                name,
                output,
                inputs=capture_inputs,
                metadata={
                    "module_path": path,
                    "class": module.__class__.__name__,
                    "input_names": capture_names,
                },
            )

    def observe_function(
        self,
        name: str,
        output: object,
        *,
        inputs: tuple[object, ...] | list[object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Record a function/kernel boundary exposed by a diagnostic wrapper."""

        self._record_operator(
            name,
            output,
            inputs=inputs,
            metadata=metadata,
        )

    def note_prompt(self, value: object) -> None:
        if isinstance(value, torch.Tensor):
            ids = tuple(int(item) for item in value.detach().cpu().flatten().tolist())
            if ids == PROMPT_TOKEN_IDS:
                self.prompt_seen = True

    @staticmethod
    def _layer_hidden(
        args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        return _cpu_tensor(
            _argument(args, kwargs, "hidden_states", 0), "layer hidden input"
        )

    @staticmethod
    def _layer_residual(
        args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor | None:
        value = _argument(args, kwargs, "residual", 1)
        return None if value is None else _cpu_tensor(value, "layer residual input")

    def _bind_layers(self, model: torch.nn.Module) -> None:
        target = model
        for _ in range(3):
            if isinstance(getattr(target, "layers", None), torch.nn.ModuleList):
                break
            target = getattr(target, "model", None)
            if not isinstance(target, torch.nn.Module):
                break
        layers = getattr(target, "layers", None)
        if not isinstance(layers, torch.nn.ModuleList) or len(layers) != NUM_LAYERS:
            raise LayerGateError("target model does not expose exactly 24 decoder layers")
        for index, layer in enumerate(layers):
            if not _is_layer(layer) or getattr(layer, "layer_id", None) != index:
                raise LayerGateError(f"decoder layer identity mismatch at index {index}")

    def _start(self, model: torch.nn.Module) -> None:
        if self.output.exists() or self.output.is_symlink():
            raise LayerGateError(f"diagnostic output already exists: {self.output}")
        self.output.mkdir(mode=0o700, parents=False)
        self.active = True
        self.model_id = id(model)
        self.next_layer = 0
        _atomic_write(
            self.output / "session.json",
            _canonical_json(
                {
                    "schema": SCHEMA,
                    "state": "active",
                    "diagnostic_only": True,
                    "oracle_feedback_to_target": False,
                    "pid": os.getpid(),
                    "prompt_token_ids": list(PROMPT_TOKEN_IDS),
                    "positions": list(TOKEN_POSITIONS),
                    "golden": {
                        "directory": str(self.golden.directory),
                        "files": self.golden.file_records,
                    },
                    "operator_golden": {
                        "directory": str(self.operator_golden.directory),
                        "files": self.operator_golden.file_records,
                        "operator_order": list(OPERATOR_ORDER),
                    },
                    "thresholds": {
                        "atol": ATOL,
                        "rtol": RTOL,
                        "max_relative_l2": MAX_RELATIVE_L2,
                        "min_cosine_similarity": MIN_COSINE,
                    },
                    "model_module": model.__class__.__module__,
                    "model_class": model.__class__.__name__,
                }
            ),
        )

    def enter_model(
        self,
        model: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        input_ids = _argument(args, kwargs, "input_ids", 0)
        positions = _argument(args, kwargs, "positions", 1)
        if not isinstance(input_ids, torch.Tensor):
            return
        ids = tuple(int(item) for item in input_ids.detach().cpu().flatten().tolist())
        if ids != PROMPT_TOKEN_IDS or _normalize_positions(positions) != TOKEN_POSITIONS:
            return
        # SGLang's multimodal wrapper calls a text wrapper which calls the
        # decoder body.  The global hook sees all three modules; the outermost
        # fixed-prompt call owns the session and nested calls are observations
        # of the same execution, not a second request.
        if self.active:
            return
        self._start(model)

    def observe_layer(
        self,
        layer: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        output: object,
    ) -> None:
        """Compare one completed layer through the kwargs-aware global hook.

        SGLang's scheduler invokes the multimodal model's ``forward`` method
        directly, so a model-level PyTorch hook is not guaranteed to run.
        Decoder layers are still invoked through ``Module.__call__`` in the
        normal loop.  The pinned PyTorch global pre-hook cannot expose keyword
        arguments, but its post-hook can, so it checks both saved inputs and
        output immediately when each layer retires.  Arm on the first layer-0
        call with the frozen two-token shape.
        """

        layer_index = getattr(layer, "layer_id", None)
        if type(layer_index) is not int:
            return
        hidden = self._layer_hidden(args, kwargs)
        if _normalize_positions(kwargs.get("positions")) != TOKEN_POSITIONS:
            return
        if not self.active:
            if (
                not self.prompt_seen
                or layer_index != 0
                or tuple(hidden.shape) != HIDDEN_SHAPE
            ):
                return
            self._start(layer)
            self.prompt_seen = False
        if layer_index == 0 and self.operator_active:
            if self.operator_next != len(OPERATOR_ORDER):
                raise LayerGateError(
                    "layer 0 retired before all operator boundaries were observed: "
                    f"{self.operator_next}/{len(OPERATOR_ORDER)}"
                )
            # Function wrappers are process-global.  Disarm them after the
            # audited layer so layer 1 cannot be mistaken for an extra layer-0
            # operator boundary.
            self.operator_active = False
            self.prompt_seen = False
        self.before_layer(layer, args, kwargs)
        self.after_layer(layer, args, kwargs, output)

    def before_layer(
        self,
        layer: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        if not self.active:
            return
        layer_index = getattr(layer, "layer_id", None)
        if type(layer_index) is not int or layer_index != self.next_layer:
            raise LayerGateError(
                f"layer order mismatch: expected={self.next_layer} actual={layer_index}"
            )
        hidden = self._layer_hidden(args, kwargs)
        residual = self._layer_residual(args, kwargs)
        expected_hidden_key = (
            "hidden_input"
            if layer_index == 0
            else f"layers.{layer_index - 1}.returned_hidden"
        )
        comparisons = {
            "hidden": compare_tensor(hidden, self.golden.tensors[expected_hidden_key])
        }
        expected_residual = None
        if layer_index == 0:
            residual_correct = residual is None
            comparisons["residual"] = {
                "expected": None,
                "actual": None if residual is None else "tensor",
                "correct": residual_correct,
            }
        else:
            expected_residual = self.golden.tensors[
                f"layers.{layer_index - 1}.returned_residual"
            ]
            comparisons["residual"] = compare_tensor(residual, expected_residual)
        record = {
            "schema": SCHEMA,
            "phase": "layer_input",
            "layer": layer_index,
            "comparisons": comparisons,
            "correct": all(value["correct"] for value in comparisons.values()),
        }
        if not record["correct"]:
            tensors = {"actual_hidden": hidden, "expected_hidden": self.golden.tensors[expected_hidden_key]}
            if residual is not None:
                tensors["actual_residual"] = residual
            if expected_residual is not None:
                tensors["expected_residual"] = expected_residual
            self._mismatch(record, tensors)
        _append_jsonl(self.output / "comparisons.jsonl", record)
        self.before = {"hidden": hidden, "residual": residual}

    def after_layer(
        self,
        layer: torch.nn.Module,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
        output: object,
    ) -> None:
        if not self.active:
            return
        layer_index = getattr(layer, "layer_id", None)
        if type(layer_index) is not int or layer_index != self.next_layer:
            raise LayerGateError("layer output order mismatch")
        hidden, residual = _output_pair(output)
        expected_hidden = self.golden.tensors[f"layers.{layer_index}.returned_hidden"]
        expected_residual = self.golden.tensors[
            f"layers.{layer_index}.returned_residual"
        ]
        comparisons = {
            "hidden": compare_tensor(hidden, expected_hidden),
            "residual": compare_tensor(residual, expected_residual),
        }
        record = {
            "schema": SCHEMA,
            "phase": "layer_output",
            "layer": layer_index,
            "comparisons": comparisons,
            "correct": all(value["correct"] for value in comparisons.values()),
        }
        if not record["correct"]:
            tensors = {
                "before_hidden": self.before["hidden"],
                "actual_hidden": hidden,
                "expected_hidden": expected_hidden,
                "actual_residual": residual,
                "expected_residual": expected_residual,
            }
            if self.before.get("residual") is not None:
                tensors["before_residual"] = self.before["residual"]
            self._mismatch(record, tensors)
        _append_jsonl(self.output / "comparisons.jsonl", record)
        self.next_layer += 1
        self.before = {}
        if self.next_layer == NUM_LAYERS:
            self._publish_completed()

    def _publish_completed(self) -> None:
        if not self.active:
            return
        _atomic_write(
            self.output / "layer-gate-result.json",
            _canonical_json(
                {
                    "schema": SCHEMA,
                    "state": "layer_gate_passed",
                    "correct": True,
                    "layers_completed": self.next_layer,
                    "diagnostic_only": True,
                    "oracle_feedback_to_target": False,
                }
            ),
        )
        self.active = False
        self.model_id = None

    def leave_model(self, model: torch.nn.Module, output: object) -> None:
        if not self.active or id(model) != self.model_id:
            return
        completed = self.next_layer == NUM_LAYERS and output is not None
        if completed:
            self._publish_completed()
            return
        _atomic_write(
            self.output / "layer-gate-result.json",
            _canonical_json(
                {
                    "schema": SCHEMA,
                    "state": "layer_gate_passed" if completed else "model_aborted",
                    "correct": completed,
                    "layers_completed": self.next_layer,
                    "diagnostic_only": True,
                    "oracle_feedback_to_target": False,
                }
            ),
        )
        self.active = False
        self.model_id = None

    def _mismatch(
        self,
        record: dict[str, Any],
        tensors: dict[str, torch.Tensor | None],
        *,
        operator: bool = False,
    ) -> None:
        if self._published_mismatch:
            raise FirstNumericalMismatch("additional mismatch after first mismatch")
        self._published_mismatch = True
        package = self.output / "first-mismatch"
        package.mkdir(mode=0o700)
        clean_tensors = {
            key: value.detach().cpu().contiguous()
            for key, value in tensors.items()
            if isinstance(value, torch.Tensor)
        }
        tensor_payload = save_safetensors(clean_tensors)
        _atomic_write(package / "tensors.safetensors", tensor_payload)
        result = {
            **record,
            "state": "first_numerical_mismatch",
            "diagnostic_only": True,
            "oracle_feedback_to_target": False,
            "replay_capsule": bool(operator),
            **(
                {"capture_metadata": self.operator_metadata}
                if operator
                else {}
            ),
            "tensor_file": {
                "name": "tensors.safetensors",
                "bytes": len(tensor_payload),
                "sha256": hashlib.sha256(tensor_payload).hexdigest(),
                "roles": sorted(clean_tensors),
            },
        }
        _atomic_write(package / "result.json", _canonical_json(result))
        raise FirstNumericalMismatch(
            f"Qwen3.5 first numerical mismatch: phase={record['phase']} "
            f"layer={record['layer']} "
            f"operator={record.get('operator', '<layer-boundary>')} evidence={package}"
        )


_thread = threading.local()
_handles: list[object] = []
_patched_functions: dict[tuple[object, str], object] = {}


def _controller() -> LayerGate:
    controller = getattr(_thread, "controller", None)
    if controller is None:
        output = Path(os.environ["SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT"])
        golden_dir = Path(os.environ["SAGR_QWEN35_SGLANG_LAYER_GATE_GOLDEN"])
        if not output.is_absolute() or output.parent.resolve(strict=True) != output.parent:
            raise LayerGateError("diagnostic output path must be normalized and absolute")
        controller = LayerGate(output, load_golden(golden_dir))
        _thread.controller = controller
    return controller


def _is_model(module: torch.nn.Module) -> bool:
    return (
        module.__class__.__module__.startswith(MODEL_MODULE_PREFIX)
        and module.__class__.__name__ in MODEL_CLASSES
    )


def _is_layer(module: torch.nn.Module) -> bool:
    return (
        module.__class__.__module__ == MODEL_MODULE
        and module.__class__.__name__ in LAYER_CLASSES
    )


def _looks_like_embedding(module: torch.nn.Module) -> bool:
    name = module.__class__.__name__
    return (
        isinstance(module, torch.nn.Embedding)
        or name.endswith("Embedding")
        or name == "VocabParallelEmbedding"
    )


def _pre_hook(
    module: torch.nn.Module,
    args: tuple[object, ...],
) -> None:
    # The pinned ROCm PyTorch has no kwargs-aware process-global pre-hook.
    # SGLang supplies input_ids and positions positionally to the outer model;
    # decoder-layer hooks below remain kwargs-aware.
    if _looks_like_embedding(module) and args:
        controller = _controller()
        controller.note_prompt(args[0])
        # Do not import SGLang/AITER from sitecustomize or identity probes.
        # At this point the real engine has executed its embedding and the
        # model modules are already importable; arm_operator_layer will verify
        # the complete wrapper set before layer-0 execution.
    if _is_layer(module):
        _controller().arm_operator_layer(module, args)
    if _is_model(module):
        _controller().enter_model(module, args, {})


def _post_hook(
    module: torch.nn.Module,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    output: object,
) -> None:
    _controller().observe_module(module, args, kwargs, output)
    if _is_layer(module):
        _controller().observe_layer(module, args, kwargs, output)
    elif _is_model(module):
        _controller().leave_model(module, output)


def _patch_attribute(owner: object, name: str, wrapper_factory) -> None:
    key = (owner, name)
    if key in _patched_functions:
        return
    original = getattr(owner, name)
    wrapped = wrapper_factory(original)
    setattr(owner, name, wrapped)
    _patched_functions[key] = original


def _install_function_wrappers() -> None:
    """Patch diagnostic observation at SGLang's non-Module kernel boundaries."""

    import importlib

    model = importlib.import_module("sglang.srt.models.qwen3_5")
    gdn = importlib.import_module(
        "sglang.srt.layers.attention.linear.gdn_backend"
    )
    gdn_kernel = importlib.import_module(
        "sglang.srt.layers.attention.linear.kernels.gdn_triton"
    )

    def split_wrapper(original):
        def wrapped(mixed_qkvz, mixed_ba, *args, **kwargs):
            output = original(mixed_qkvz, mixed_ba, *args, **kwargs)
            controller = _controller()
            for name, value in zip(("mixed_qkv", "z", "b", "a"), output):
                controller.observe_function(
                    name,
                    value,
                    inputs=(mixed_qkvz, mixed_ba),
                    metadata={
                        "function": original.__module__ + "." + original.__name__,
                        "input_names": ["mixed_qkvz", "mixed_ba"],
                    },
                )
            return output

        return wrapped

    _patch_attribute(
        model,
        "fused_qkvzba_split_reshape_cat_contiguous",
        split_wrapper,
    )

    def conv_wrapper(original):
        def wrapped(*args, **kwargs):
            conv_states = kwargs.get("conv_states")
            cache_indices = kwargs.get("cache_indices")
            selected_before = None
            if isinstance(conv_states, torch.Tensor) and isinstance(
                cache_indices, torch.Tensor
            ):
                selected_before = conv_states.index_select(
                    0, cache_indices.to(device=conv_states.device, dtype=torch.long)
                ).clone()
            output = original(*args, **kwargs)
            controller = _controller()
            replay_inputs = (
                *(args[:3]),
                selected_before,
                kwargs.get("has_initial_state"),
                cache_indices,
                kwargs.get("query_start_loc"),
            )
            replay_names = [
                "x",
                "weight",
                "bias",
                "selected_conv_state_before",
                "has_initial_state",
                "cache_indices",
                "query_start_loc",
            ]
            call_metadata = {
                "function": original.__module__ + "." + original.__name__,
                "input_names": replay_names,
                "activation": kwargs.get("activation"),
                "seq_lens_cpu": kwargs.get("seq_lens_cpu"),
            }
            controller.observe_function(
                "gdn_conv_output",
                output.transpose(0, 1) if isinstance(output, torch.Tensor) else output,
                inputs=replay_inputs,
                metadata=call_metadata,
            )
            if isinstance(conv_states, torch.Tensor) and isinstance(
                cache_indices, torch.Tensor
            ):
                selected = conv_states.index_select(
                    0, cache_indices.to(device=conv_states.device, dtype=torch.long)
                )
                controller.observe_function(
                    "conv_state",
                    selected,
                    inputs=replay_inputs,
                    metadata={
                        **call_metadata,
                        "selected_cache_indices": cache_indices.detach()
                        .cpu()
                        .tolist(),
                    },
                )
            else:
                raise LayerGateError("GDN prefill conv did not expose selected state")
            return output

        return wrapped

    _patch_attribute(gdn, "causal_conv1d_fn", conv_wrapper)

    def qkv_split_wrapper(original):
        def wrapped(mixed_qkv, *args, **kwargs):
            output = original(mixed_qkv, *args, **kwargs)
            for name, value in zip(
                ("gdn_q_raw", "gdn_k_raw", "gdn_v_raw"), output
            ):
                _controller().observe_function(
                    name,
                    value,
                    inputs=(mixed_qkv,),
                    metadata={
                        "function": original.__module__ + "." + original.__name__,
                        "input_names": ["mixed_qkv"],
                    },
                )
            return output

        return wrapped

    _patch_attribute(gdn, "fused_qkv_split_gdn_prefill", qkv_split_wrapper)

    def gating_wrapper(original):
        def wrapped(*args, **kwargs):
            output = original(*args, **kwargs)
            replay_inputs = (*args, *kwargs.values())
            replay_names = [
                *[f"arg_{index}" for index in range(len(args))],
                *[f"kw_{key}" for key in kwargs],
            ]
            call_metadata = {
                "function": original.__module__ + "." + original.__name__,
                "input_names": replay_names,
            }
            _controller().observe_function(
                "gdn_g",
                output[0],
                inputs=replay_inputs,
                metadata=call_metadata,
            )
            _controller().observe_function(
                "gdn_beta_output",
                output[1],
                inputs=replay_inputs,
                metadata=call_metadata,
            )
            return output

        return wrapped

    _patch_attribute(gdn, "fused_gdn_gating", gating_wrapper)

    def recurrent_wrapper(original):
        def wrapped(*args, **kwargs):
            # Save the selected initial state before the in-place kernel.  It is
            # part of the minimal replay capsule if this is the first mismatch.
            initial = kwargs.get("initial_state")
            indices = kwargs.get("initial_state_indices")
            initial_selected = None
            if isinstance(initial, torch.Tensor) and isinstance(indices, torch.Tensor):
                initial_selected = initial.index_select(
                    0, indices.to(device=initial.device, dtype=torch.long)
                ).clone()
            output = original(*args, **kwargs)
            controller = _controller()
            replay_inputs = (
                kwargs.get("q"),
                kwargs.get("k"),
                kwargs.get("v"),
                kwargs.get("g"),
                kwargs.get("beta"),
                initial_selected,
                indices,
                kwargs.get("cu_seqlens"),
            )
            replay_names = [
                "q",
                "k",
                "v",
                "g",
                "beta",
                "selected_initial_state_before",
                "initial_state_indices",
                "cu_seqlens",
            ]
            call_metadata = {
                "function": original.__module__ + "." + original.__name__,
                "input_names": replay_names,
                "head_first": kwargs.get("head_first"),
                "use_qk_l2norm_in_kernel": kwargs.get(
                    "use_qk_l2norm_in_kernel"
                ),
            }
            controller.observe_function(
                "gdn_recurrent_output",
                output[0][0] if isinstance(output, tuple) else output,
                inputs=replay_inputs,
                metadata=call_metadata,
            )
            # The chunk kernel returns its per-chunk h history as output[2],
            # while the persistent final state is written in-place into the
            # selected initial-state pool.  Compare the persistent state, not
            # the history envelope.
            final_state = None
            if isinstance(initial, torch.Tensor) and isinstance(indices, torch.Tensor):
                final_state = initial.index_select(
                    0, indices.to(device=initial.device, dtype=torch.long)
                )
            controller.observe_function(
                "recurrent_state",
                final_state,
                inputs=replay_inputs,
                metadata=call_metadata,
            )
            return output

        return wrapped

    _patch_attribute(gdn_kernel, "chunk_gated_delta_rule", recurrent_wrapper)


def install() -> None:
    """Install exactly one process-local observation hook pair.

    Function wrappers are installed lazily at the first fixed-prompt embedding
    observation.  Importing the complete SGLang model graph from sitecustomize
    makes unrelated identity/helper subprocesses initialize HIP/AITER and can
    recursively start managed gem5 sessions before the engine exists.
    """

    if _handles:
        return
    _handles.extend(
        (
            torch.nn.modules.module.register_module_forward_pre_hook(_pre_hook),
            torch.nn.modules.module.register_module_forward_hook(
                _post_hook, with_kwargs=True, always_call=True
            ),
        )
    )
    print(
        "[sagr] Qwen3.5 SGLang fail-fast layer hook installed",
        flush=True,
    )


def assert_installed() -> None:
    """Fail closed if Python ignored an exception raised by sitecustomize."""

    if len(_handles) != 2:
        raise LayerGateError(
            "SGLang numerical diagnostic requested but its global hook pair "
            "is not installed "
            f"(handles={len(_handles)} wrappers={len(_patched_functions)})"
        )


def assert_completed() -> None:
    """Require all 24 layer boundaries before a diagnostic lane can pass."""

    output = Path(os.environ["SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT"])
    result_path = output / "layer-gate-result.json"
    if not result_path.is_file() or result_path.is_symlink():
        raise LayerGateError("SGLang layer diagnostic produced no completion result")
    try:
        result = json.loads(result_path.read_text(encoding="ascii"))
    except Exception as error:
        raise LayerGateError("SGLang layer diagnostic result is unreadable") from error
    if (
        result.get("schema") != SCHEMA
        or result.get("state") != "layer_gate_passed"
        or result.get("correct") is not True
        or result.get("layers_completed") != NUM_LAYERS
    ):
        raise LayerGateError("SGLang layer diagnostic did not pass all 24 layers")
