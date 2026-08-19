#!/usr/bin/env python3
"""Replay the first failing Qwen3.5 GDN recurrent operator in isolation.

The layer gate writes the inputs at the first numerical mismatch.  This tool
does not regenerate those inputs and does not use the NVIDIA oracle as an
input: it executes the pinned SGLang FLA operator with the captured tensors,
then compares the result with both the simulator result and the frozen oracle.

The captured state is the *selected* cache slot.  The real kernel receives a
slot pool plus ``initial_state_indices`` and uses ``initial_state.stride(0)``;
therefore this capsule reconstructs a three-slot pool and places the captured
state back in the original slot (currently slot 2).  That keeps cache-index
and stride semantics in the reproduction instead of silently rebasing them.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = (
    ROOT
    / "artifacts/lanes/sglang-tp1-layer-gate-v5/layer-gate/first-mismatch"
)


def _sha(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    a = actual.float().reshape(-1)
    b = reference.float().reshape(-1)
    delta = a - b
    denom = torch.linalg.vector_norm(b).item()
    norm_a = torch.linalg.vector_norm(a).item()
    norm_b = torch.linalg.vector_norm(b).item()
    cosine = torch.nn.functional.cosine_similarity(a[None], b[None]).item()
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype).removeprefix("torch."),
        "reference_dtype": str(reference.dtype).removeprefix("torch."),
        "max_abs_error": float(delta.abs().max().item()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(delta).item() / max(denom, 1.0e-30)
        ),
        "cosine_similarity": float(cosine),
        "actual_norm": float(norm_a),
        "reference_norm": float(norm_b),
        "actual_sha256": _sha(actual),
        "reference_sha256": _sha(reference),
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            __import__("os").environ.get(
                "QWEN35_RECURRENT_CAPSULE_OUTPUT",
                str(ROOT / "artifacts/qwen35-recurrent-capsule/20260819-layer0-v2"),
            )
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--standalone-fla",
        action="store_true",
        help="load only the pinned FLA source files (for the CUDA golden env)",
    )
    parser.add_argument(
        "--pool-slots",
        type=int,
        default=3,
        help="number of reconstructed state slots (must exceed selected index)",
    )
    return parser.parse_args()


def _load_standalone_fla_chunk():
    """Load the pinned FLA files without importing all of SGLang.

    The CUDA golden environment intentionally does not carry the server's
    optional dependencies.  The FLA files only require ``torch_release`` from
    SGLang's common utility module; providing that narrow module keeps this
    comparison source-identical without installing or changing SGLang.
    """

    import types

    package_names = (
        "sglang",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.attention",
        "sglang.kernels.ops.attention.fla",
        "sglang.srt",
        "sglang.srt.utils",
    )
    for name in package_names:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
    common = types.ModuleType("sglang.srt.utils.common")
    common.torch_release = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    sys.modules[common.__name__] = common

    fla_root = ROOT / "projects/sglang-0.5.17/sglang/kernels/ops/attention/fla"
    module_order = (
        "utils",
        "op",
        "index",
        "l2norm",
        "cumsum",
        "wy_fast",
        "chunk_fwd",
        "chunk_delta_h",
        "chunk_o",
        "chunk",
    )
    loaded = {}
    for short_name in module_order:
        full_name = f"sglang.kernels.ops.attention.fla.{short_name}"
        spec = importlib.util.spec_from_file_location(
            full_name, fla_root / f"{short_name}.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load pinned FLA source: {short_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        loaded[short_name] = module
    return loaded["chunk"]


def main() -> int:
    args = _parse()
    capture = args.capture.resolve(strict=True)
    tensors_path = capture / "tensors.safetensors"
    metadata_path = capture / "result.json"
    if not tensors_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"incomplete first-mismatch capsule: {capture}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tensors = load_file(str(tensors_path), device="cpu")
    required = {
        "actual",
        "expected",
        "input_0",
        "input_1",
        "input_2",
        "input_3",
        "input_4",
        "input_5",
        "input_6",
        "input_7",
    }
    if set(tensors) != required:
        raise RuntimeError(f"unexpected capsule tensor keys: {sorted(tensors)}")

    selected_index = int(tensors["input_6"].reshape(-1)[0].item())
    if selected_index < 0 or selected_index >= args.pool_slots:
        raise RuntimeError(
            f"selected state index {selected_index} does not fit pool-slots="
            f"{args.pool_slots}"
        )
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("simulated HIP torch device is unavailable")

    device = torch.device(args.device)
    # Import only after the lane has installed its backend and simulator
    # identity.  Importing this module outside the lane can select a real CUDA
    # Triton backend and would invalidate the replay.
    if args.standalone_fla:
        chunk_module = _load_standalone_fla_chunk()
    else:
        import sglang.kernels.ops.attention.fla.chunk as chunk_module

    chunk_gated_delta_rule = chunk_module.chunk_gated_delta_rule
    stages: dict[str, torch.Tensor] = {}
    stage_meta: dict[str, object] = {"l2norm_calls": 0}

    def _save_stage(name: str, value: object) -> None:
        if isinstance(value, torch.Tensor):
            stages[name] = value.detach().cpu().contiguous()

    def _wrap_l2norm(original):
        @functools.wraps(original)
        def wrapped(value, *args, **kwargs):
            output_value = original(value, *args, **kwargs)
            ordinal = int(stage_meta["l2norm_calls"])
            stage_meta["l2norm_calls"] = ordinal + 1
            _save_stage(("q" if ordinal == 0 else "k") + "_l2norm", output_value)
            return output_value

        return wrapped

    def _wrap_cumsum(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            output_value = original(*args, **kwargs)
            _save_stage("g_cumsum", output_value)
            return output_value

        return wrapped

    def _wrap_intra(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            output_value = original(*args, **kwargs)
            if isinstance(output_value, tuple):
                for name, value in zip(("w", "u", "A"), output_value):
                    _save_stage(name, value)
            return output_value

        return wrapped

    def _wrap_h(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            output_value = original(*args, **kwargs)
            if isinstance(output_value, tuple):
                for name, value in zip(("h", "v_new"), output_value):
                    _save_stage(name, value)
            return output_value

        return wrapped

    def _wrap_o(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            output_value = original(*args, **kwargs)
            _save_stage("o", output_value)
            return output_value

        return wrapped

    # These are capsule-only diagnostic hooks.  They target the function names
    # imported by chunk.py, so the calls are the exact code path used by the
    # model and not a reimplemented reference.
    chunk_module.l2norm_fwd = _wrap_l2norm(chunk_module.l2norm_fwd)
    chunk_module.chunk_local_cumsum = _wrap_cumsum(chunk_module.chunk_local_cumsum)
    chunk_module.chunk_gated_delta_rule_fwd_intra = _wrap_intra(
        chunk_module.chunk_gated_delta_rule_fwd_intra
    )
    chunk_module.chunk_gated_delta_rule_fwd_h = _wrap_h(
        chunk_module.chunk_gated_delta_rule_fwd_h
    )
    chunk_module.chunk_fwd_o = _wrap_o(chunk_module.chunk_fwd_o)

    q = tensors["input_0"].to(device)
    k = tensors["input_1"].to(device)
    v = tensors["input_2"].to(device)
    g = tensors["input_3"].to(device)
    beta = tensors["input_4"].to(device)
    cu_seqlens = tensors["input_7"].to(device)
    state_selected = tensors["input_5"].to(device)
    # The production state pool is [slots, H, V, K], and the selected slot is
    # mutated in-place by chunk_gated_delta_rule_fwd_h.  Keep the original
    # index and a contiguous per-slot stride in this first reproduction.
    state_pool = torch.zeros(
        (args.pool_slots, *state_selected.shape[1:]),
        dtype=state_selected.dtype,
        device=device,
    )
    state_pool[selected_index].copy_(state_selected[0])
    indices = tensors["input_6"].to(device=device, dtype=torch.int32).contiguous()

    print(
        "CAPSULE input",
        json.dumps(
            {
                "capture": str(capture),
                "selected_state_index": selected_index,
                "pool_shape": list(state_pool.shape),
                "pool_stride": list(state_pool.stride()),
                "indices_dtype": str(indices.dtype),
                "device": str(device),
                "function": metadata.get("metadata", {}).get("function"),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    output, _, history = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state_pool,
        initial_state_indices=indices,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    # Force all asynchronous device work to complete before copying evidence.
    torch.cuda.synchronize()
    replayed = output[0].detach().cpu().contiguous()
    final_state = state_pool[selected_index].detach().cpu().contiguous()
    history_cpu = history.detach().cpu().contiguous() if history is not None else None

    actual = tensors["actual"].contiguous()
    expected = tensors["expected"].contiguous()
    result = {
        "schema": "amdgpu-sim.qwen35-recurrent-capsule.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "replayed",
        "capture": str(capture),
        "source_result_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "selected_state_index": selected_index,
        "pool_shape": list(state_pool.shape),
        "pool_stride": list(state_pool.stride()),
        "operator_metadata": metadata.get("metadata", {}),
        "replayed_vs_actual": _metrics(replayed, actual),
        "replayed_vs_expected": _metrics(replayed, expected),
        "original_actual_vs_expected": _metrics(actual, expected),
        "history_shape": list(history_cpu.shape) if history_cpu is not None else None,
        "stage_shapes": {name: list(value.shape) for name, value in stages.items()},
        "stage_sha256": {name: _sha(value) for name, value in stages.items()},
        "stage_metadata": stage_meta,
    }
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    save_file(
        {
            "replayed": replayed,
            "actual": actual,
            "expected": expected,
            "final_state": final_state,
            **({"history": history_cpu} if history_cpu is not None else {}),
            **stages,
        },
        str(args.output_dir / "tensors.safetensors"),
        metadata={"schema": result["schema"]},
    )
    result["tensors_sha256"] = hashlib.sha256(
        (args.output_dir / "tensors.safetensors").read_bytes()
    ).hexdigest()
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    # A capsule is a diagnostic reproducer.  It exits nonzero if the simulator
    # cannot reproduce its own captured output, which prevents a false sense
    # of progress when the environment or state layout drifted.
    rel = result["replayed_vs_actual"]["relative_l2_error"]
    cos = result["replayed_vs_actual"]["cosine_similarity"]
    if rel > 0.03 or cos < 0.98:
        print("CAPSULE REPRODUCTION FAIL: replay != captured simulator output", flush=True)
        return 1
    print("CAPSULE REPRODUCTION PASS: replay matches captured simulator output", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
