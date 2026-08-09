#!/usr/bin/env python3
"""Run the deterministic Qwen3.5 operator-contract smoke gate.

The default run is intentionally a *static* smoke: it validates model shapes,
layer counts, registrations, and source symbols.  It does not call a CPU
implementation, and it does not call a CUDA/NVIDIA kernel.  Consequently the
result is ``blocked`` until an AMD/ROCm host-native device runner is connected.
Use ``--require-amd`` in CI when a missing AMD runtime should fail the command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

try:
    from qwen35_operator_manifest import ROOT, build_manifest
except ImportError:  # pragma: no cover - supports importing from the repository root
    from tools.qwen35_operator_manifest import ROOT, build_manifest


def _shape_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    topology = manifest["topology"]
    hidden = int(topology["hidden_size"])
    key_heads = int(topology["linear_num_key_heads"])
    value_heads = int(topology["linear_num_value_heads"])
    key_dim = int(topology["linear_key_head_dim"])
    value_dim = int(topology["linear_value_head_dim"])
    conv_width = int(topology["linear_conv_kernel_dim"])
    qwen_key_dim = key_heads * key_dim
    qwen_value_dim = value_heads * value_dim
    conv_dim = qwen_key_dim * 2 + qwen_value_dim
    qkvz_dim = qwen_key_dim * 2 + qwen_value_dim * 2
    ba_dim = value_heads * 2
    full_q_dim_with_gate = int(topology["num_attention_heads"]) * int(topology["head_dim"]) * 2
    full_kv_dim = int(topology["num_key_value_heads"]) * int(topology["head_dim"])
    full_qkv_dim = full_q_dim_with_gate + full_kv_dim * 2
    expected = {
        "hidden_size": hidden,
        "gdn_key_dim": qwen_key_dim,
        "gdn_value_dim": qwen_value_dim,
        "gdn_conv_dim": conv_dim,
        "gdn_qkvz_projection_dim": qkvz_dim,
        "gdn_ba_projection_dim": ba_dim,
        "gdn_conv_width": conv_width,
        "full_attention_qkv_projection_dim": full_qkv_dim,
    }
    checks = {
        "gdn_key_value_head_dims_match": qwen_key_dim == qwen_value_dim,
        "gdn_qkvz_is_qkv_plus_z": qkvz_dim == conv_dim + qwen_value_dim,
        "gdn_conv_width_is_positive": conv_width > 0,
        "full_attention_qkv_accounts_for_output_gate": full_qkv_dim == 5120,
        "qwen_hidden_size_matches_projection_input": hidden == 1024,
    }
    return {
        "expected": expected,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_smoke() -> dict[str, Any]:
    manifest = build_manifest()
    shape = _shape_contract(manifest)
    runtime = manifest["runtime_probe"]
    static_ok = bool(manifest["summary"]["static_source_ok"] and shape["passed"])
    amd_execution = bool(runtime["amd_ready"] and manifest["summary"]["amd_runtime_executed"])
    if amd_execution:
        runtime_status = "execution_not_implemented"
        blocker = "AMD runtime detected, but no host-native Triton execution runner is registered"
    else:
        runtime_status = "blocked"
        blocker = runtime["reason"]
    return {
        "schema": "amdgpu-sim.qwen35.operator-smoke.v1",
        "model_revision": manifest["model"]["revision"],
        "scope": {
            "text_only": True,
            "vision_deferred": True,
            "cpu_fallback_allowed": False,
            "nvidia_execution_counts_as_pass": False,
            "full_rocm_opencl_cts_required": False,
        },
        "static_contract": {
            "source_and_registration_gate": bool(manifest["summary"]["static_source_ok"]),
            "shape_gate": bool(shape["passed"]),
            "passed": static_ok,
        },
        "shape_contract": shape,
        "runtime": {
            "status": runtime_status,
            "amd_ready": bool(runtime["amd_ready"]),
            "hip_runtime": bool(runtime["hip_runtime"]),
            "cuda_runtime": bool(runtime["cuda_runtime"]),
            "device_name": runtime["device_name"],
            "blocker": blocker,
        },
        "fallback_audit": {
            "cpu_fallback_attempted": False,
            "cpu_fallback_counted_as_pass": False,
            "nvidia_fallback_attempted": False,
            "nvidia_fallback_counted_as_pass": False,
        },
        # A static contract is progress, not end-to-end success.  Keep this
        # false until the AMD device runner executes every contract.
        "passed": False,
        "pass_reason": "static contract only; AMD device execution is still required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("-"))
    parser.add_argument(
        "--require-amd",
        action="store_true",
        help="fail unless an AMD runtime is available (execution remains a separate gate)",
    )
    args = parser.parse_args()
    result = run_smoke()
    payload = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.output == Path("-"):
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(json.dumps({"output": str(args.output), "status": result["runtime"]["status"]}, sort_keys=True))
    if not result["static_contract"]["passed"]:
        return 1
    if args.require_amd and not result["runtime"]["amd_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
