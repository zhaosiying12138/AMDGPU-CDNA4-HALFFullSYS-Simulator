#!/usr/bin/env python3
"""Replay vLLM's GDN prefill-kernel warmup to bisect the wild-address crash.

zcode-vllm-tp1-v7 panicked right after model load inside the warmup that
vLLM's V1 profiling runs (qwen_gdn_linear_attn._warmup_prefill_kernels ->
chunk_gated_delta_rule): a non-Triton-signature kernel with 72 KB LDS and a
4-pointer kernarg executed `buffer_store_dwordx4` to a wild, unaligned
address (0x...7c22), with s2=s3=0xffffffff live in the scalar bank.  This
capsule replays that exact call with the same dummy construction and the
same environment (constant-time autotune selects config #1 for every FLA
kernel, matching the lane), then bisects the six-op FLA chain
(cumsum -> kkt -> solve_tril -> wy_fast -> fwd_h -> fwd_o) by running each
stage with cached intermediates so the first stage whose kernel computes a
wild address isolates in minutes.

Stages are tagged; the journal line names the stage before it runs so a
panic message identifies the culprit even though the panic kills the
process.  Exit 0 = whole chain completed on-device with finite outputs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get(
    "QWEN35_VLLM_FLA_WARMUP_OUTPUT",
    str(ROOT / "artifacts/qwen35-vllm-fla-warmup-capsule/v1")))
OUT.mkdir(parents=True, exist_ok=True)

T = 64          # FLA_CHUNK_SIZE: what the warmup uses
HK = 16         # linear_num_key_heads (TP1)
DK = 128        # head_k_dim
DV = 128        # head_v_dim
QKV_DIM = 3 * HK * DK


def stage(name: str) -> None:
    print(f"[fla-warmup] === stage: {name}", flush=True)
    (OUT / "stage.log").write_text(name, encoding="ascii")


def main() -> int:
    from vllm.third_party.flash_linear_attention.ops.chunk import (
        chunk_gated_delta_rule_fwd,
    )
    from vllm.third_party.flash_linear_attention.ops.utils import (
        FLA_CHUNK_SIZE,
    )
    assert FLA_CHUNK_SIZE == T

    dev = "cuda"
    dt = torch.bfloat16
    torch.manual_seed(0)

    report = {"schema": "amdgpu-sim.qwen35-vllm-fla-warmup.v1",
              "started_at": datetime.now(timezone.utc).isoformat()}

    q = torch.randn(1, T, HK, DK, device=dev, dtype=dt)
    k = torch.randn(1, T, HK, DK, device=dev, dtype=dt)
    v = torch.randn(1, T, HK, DV, device=dev, dtype=dt)
    g = -torch.rand(1, T, HK, device=dev, dtype=torch.float32)
    beta = torch.rand(1, T, HK, device=dev, dtype=dt)
    state = torch.zeros(1, HK, DV, DK, device=dev, dtype=torch.float32)
    cu = torch.tensor([0, T], device=dev, dtype=torch.int32)

    # l2norm in kernel disabled in the warmup; norm on host to mirror it.
    def l2n(x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    q, k = l2n(q), l2n(k)

    stage("full chunk_gated_delta_rule_fwd (warmup call)")
    g_out, o, A, final_state, _, _, _ = chunk_gated_delta_rule_fwd(
        q=q, k=k, v=v, g=g, beta=beta, scale=DK**-0.5,
        initial_state=state, output_final_state=True,
        cu_seqlens=cu, chunk_indices=None, chunk_offsets=None,
    )
    torch.cuda.synchronize()
    for nm, t in (("o", o), ("A", A), ("final_state", final_state)):
        finite = bool(torch.isfinite(t.float()).all().item())
        report[f"{nm}_finite"] = finite
        report[f"{nm}_mean_abs"] = float(t.float().abs().mean())
        print(f"[fla-warmup] {nm}: finite={finite} mean_abs={report[f'{nm}_mean_abs']:.6f}",
              flush=True)

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = all(report.get(f"{n}_finite", False) for n in ("o", "A", "final_state"))
    (OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n",
                                     encoding="ascii")
    print(f"[fla-warmup] passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
