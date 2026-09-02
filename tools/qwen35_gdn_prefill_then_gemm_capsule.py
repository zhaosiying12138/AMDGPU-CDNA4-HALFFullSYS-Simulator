#!/usr/bin/env python3
"""Run the 9B GDN prefill geometry immediately before its first MLP GEMM."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(
    os.environ.get(
        "QWEN35_GDN_PREFILL_OUTPUT",
        str(ROOT / "artifacts/qwen35-gdn-prefill-then-gemm-capsule/v1"),
    )
)


def main() -> int:
    torch.manual_seed(20260829)
    batch, seq, q_heads, v_heads, head_dim = 1, 2, 16, 32, 128
    q = torch.randn((batch, seq, q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn((batch, seq, v_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    g = torch.randn((batch, seq, v_heads), device="cuda", dtype=torch.float32)
    beta = torch.sigmoid(torch.randn_like(g))
    states = torch.zeros((4, v_heads, head_dim, head_dim), device="cuda", dtype=torch.float32)
    indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, seq], device="cuda", dtype=torch.int32)

    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    kernel = TritonGDNKernel()
    gdn_output, _, _ = kernel.extend(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        ssm_states=states,
        cache_indices=indices,
        query_start_loc=cu,
    )
    torch.cuda.synchronize()

    x = torch.randn((2, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((3072, 4096), device="cuda", dtype=torch.bfloat16)
    before_gemm = torch.cuda.memory_allocated()
    from aiter.tuned_gemm import tgemm

    actual = tgemm.mm(x, weight, otype=torch.bfloat16)
    torch.cuda.synchronize()
    expected = torch.nn.functional.linear(x.float(), weight.float()).to(torch.bfloat16)
    actual_cpu = actual.detach().cpu().contiguous()
    expected_cpu = expected.detach().cpu().contiguous()
    delta = (actual_cpu.float() - expected_cpu.float()).abs()
    report = {
        "schema": "amdgpu-sim.qwen35-gdn-prefill-then-gemm-capsule.v1",
        "gdn_geometry": {
            "q": list(q.shape),
            "k": list(k.shape),
            "v": list(v.shape),
            "state": list(states.shape),
        },
        "gemm_shape": [2, 3072, 4096],
        "gdn_output_shape": list(gdn_output.shape),
        "allocated_before_gemm": int(before_gemm),
        "allocated_after_gemm": int(torch.cuda.memory_allocated()),
        "max_abs_error": float(delta.max().item()),
        "mismatch_count": int(
            (delta > 0.03125 + 0.03 * expected_cpu.float().abs()).sum().item()
        ),
        "actual_sha256": hashlib.sha256(
            actual_cpu.view(torch.uint16).numpy().tobytes()
        ).hexdigest(),
        "expected_sha256": hashlib.sha256(
            expected_cpu.view(torch.uint16).numpy().tobytes()
        ).hexdigest(),
    }
    report["passed"] = report["mismatch_count"] == 0
    OUTPUT.mkdir(mode=0o700, parents=True, exist_ok=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
