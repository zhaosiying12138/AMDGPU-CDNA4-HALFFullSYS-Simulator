#!/usr/bin/env python3
"""Bring up a real Qwen3.5 model rank group through the existing CCL runner.

The runner still owns broker creation, capability passing, gem5 lifecycle and
cleanup.  This thin entrypoint only selects the upstream model script as the
rank payload, so model failures are exposed without creating a second runtime
or collective implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_gemsim_ccl_live_allreduce as base_runner  # noqa: E402
from scripts import run_gemsim_vllm_ccl_live as runner  # noqa: E402
from tools import gemsim_ccl_live_allreduce as design_tool  # noqa: E402


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")


def _product_runtime(prefix: Path) -> Path:
    manifest = json.loads((prefix / "manifest.json").read_text(encoding="ascii"))
    path = Path(manifest["artifacts"]["runtime_library"]["path"])
    if not path.is_absolute():
        raise RuntimeError("product runtime path is not absolute")
    return path.resolve(strict=True)


def _default_root(world: int) -> Path:
    # The managed endpoint is an AF_UNIX path; keep the namespace deliberately
    # short so every rank can pass the runtime's sun_path limit.
    parent = Path("/tmp")
    return parent / f"gsim-qwen-tp{world}-{time.time_ns() % 10**12:012d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-parallel-size", type=int, choices=(2, 4), default=2)
    parser.add_argument("--debug-stop-after-layer", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--cleanup-grace-seconds", type=float, default=10.0)
    parser.add_argument("--product-prefix", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--eager", action="store_true", default=True)
    args = parser.parse_args()
    if not 0 <= args.debug_stop_after_layer < 24:
        parser.error("--debug-stop-after-layer must be in [0,23]")

    product = (args.product_prefix or base_runner.default_product_prefix()).resolve(strict=True)
    root = (args.output_root or _default_root(args.tensor_parallel_size)).resolve()
    if root.exists() or root.is_symlink():
        raise RuntimeError(f"diagnostic root must be absent: {root}")
    root.mkdir(mode=0o700, parents=True)
    # Keep the execution namespace short: the runtime creates nested state/tmp
    # and UNIX endpoints beneath each rank, and Linux sun_path is limited.
    token = root.name.rsplit("-", 1)[-1]
    expected_path = Path("/tmp") / f"gqe{token}.json"
    output = Path("/tmp") / f"gqo{token}"
    execution_root = Path("/tmp") / f"gqx{token}"
    for path in (expected_path, output, execution_root):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"diagnostic path must be absent: {path}")
    runtime = _product_runtime(product)
    model_identity = hashlib.sha256(
        b"Qwen/Qwen3.5-0.8B:model-first:upstream-vllm"
    ).hexdigest()
    design = design_tool.build_design(
        runtime,
        execution_root,
        design_tool.deterministic_config(
            args.tensor_parallel_size,
            1024,
            "bfloat16",
            model_identity_sha256=model_identity,
        ),
    )
    design_tool.publish_expected_wrapper(expected_path, design)
    extra = {
        "worker_mode": "qwen35-model",
        "model_script": str((ROOT / "examples/triton/qwen35_vllm_model_forward.py").resolve(strict=True)),
        "model_inference_mode": "debug-layer-diff",
        "model_stop_after_layer": args.debug_stop_after_layer,
        "model_eager": bool(args.eager),
        "model_skip_oracle": True,
        "dist_init_method": (execution_root / ".vllm-gloo-rendezvous").as_uri(),
    }
    result = runner.supervise(
        expected_path=expected_path,
        output=output,
        execution_root=execution_root,
        product_prefix=product,
        timeout_seconds=args.timeout_seconds,
        cleanup_grace_seconds=args.cleanup_grace_seconds,
        worker_script=(ROOT / "examples/triton/qwen35_vllm_tp_rank.py").resolve(strict=True),
        worker_config_extra=extra,
    )
    print(json.dumps({
        "schema": "amdgpu-sim.qwen35-vllm-tp-bringup.v1",
        "status": result["status"],
        "world_size": args.tensor_parallel_size,
        "debug_stop_after_layer": args.debug_stop_after_layer,
        "root": str(root),
        "expected": str(expected_path),
        "output": str(output),
        "execution_root": str(execution_root),
        "result_manifest": str(output / "result-manifest.json"),
        "ranks": [
            {
                "rank": item["rank"],
                "returncode": item["returncode"],
                "worker_result": str(output / f"rank-{item['rank']:02d}" / "worker-result.json"),
            }
            for item in result["ranks"]
        ],
    }, sort_keys=True))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
