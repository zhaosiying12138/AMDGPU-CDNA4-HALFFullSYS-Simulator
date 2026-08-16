#!/usr/bin/env python3
"""Run the deterministic Qwen3.5 executable operator-work-queue smoke gate.

The default run validates the materialized queue and its static model/source
contracts. It never turns an ambient AMD, CPU, or NVIDIA runtime probe into
operator acceptance. Use ``--require-complete`` when incomplete runtime
evidence should fail the command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# The smoke module is also loaded by the source-contract tests through an
# importlib spec, where Python does not automatically add this directory to
# sys.path. Resolve the repository-local module explicitly so both direct
# execution and `pytest` from the repository root use the same import path.
ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for _path in (str(ROOT), str(TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from qwen35_operator_manifest import ROOT, build_manifest
    from qwen35_operator_work_queue import queue_summary, validate_manifest
except ImportError:  # pragma: no cover - supports importing from the repository root
    from tools.qwen35_operator_manifest import ROOT, build_manifest
    from tools.qwen35_operator_work_queue import queue_summary, validate_manifest


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
    validation_errors = validate_manifest(manifest)
    work_queue = queue_summary(manifest)
    static_ok = bool(
        not validation_errors
        and manifest["summary"]["static_source_ok"]
        and shape["passed"]
    )
    complete = bool(static_ok and work_queue["all_contracts_accepted"])
    return {
        "schema": "amdgpu-sim.qwen35.operator-smoke.v3",
        "model_revision": manifest["model"]["revision"],
        "scope": {
            "text_only": True,
            "vision_deferred": True,
            "cpu_fallback_allowed": False,
            "nvidia_execution_counts_as_pass": False,
            "full_rocm_opencl_cts_required": False,
        },
        "static_contract": {
            "manifest_schema_gate": not validation_errors,
            "manifest_schema_errors": validation_errors,
            "source_and_registration_gate": bool(manifest["summary"]["static_source_ok"]),
            "shape_gate": bool(shape["passed"]),
            "passed": static_ok,
        },
        "shape_contract": shape,
        "work_queue": {
            **work_queue,
            "status": "accepted" if complete else "incomplete",
            "results_external": True,
        },
        "runtime": {
            "status": "accepted" if complete else "work_queue_incomplete",
            "source_only_manifest": True,
            "external_results_required": True,
            "blocker": None if complete else manifest["summary"]["blocker"],
        },
        "fallback_audit": {
            "cpu_fallback_attempted": False,
            "cpu_fallback_counted_as_pass": False,
            "nvidia_fallback_attempted": False,
            "nvidia_fallback_counted_as_pass": False,
        },
        "passed": complete,
        "pass_reason": (
            "all 15 contracts have complete accepted fresh/repeat evidence"
            if complete
            else "work queue is valid but all 15 contracts are not accepted"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("-"))
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless all 15 operator contracts are accepted",
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
    if args.require_complete and not result["passed"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
