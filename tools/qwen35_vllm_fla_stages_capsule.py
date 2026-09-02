#!/usr/bin/env python3
"""Stage-by-stage replay of vLLM's FLA GDN chain to isolate the wild store.

The full-chain capsule (qwen35_vllm_fla_warmup_capsule) reproduces the lane
crash inside the T=64 warmup call but dies before any dump.  This capsule
calls the six ops of chunk_gated_delta_rule_fwd one at a time -- cumsum,
kkt, solve_tril, wy_fast, fwd_h, fwd_o -- printing a stage marker before
each launch and dumping every intermediate to the output dir as it is
produced.  Whichever stage's marker is last in the log is the one whose
kernel faulted; its dumped INPUTS then re-drive that single op for a
kernel-level bisect.  Shapes mirror the warmup exactly (T=64, 16 heads,
128 dims, single sequence, cu_seqlens [0, 64]).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get(
    "QWEN35_VLLM_FLA_STAGES_OUTPUT",
    str(ROOT / "artifacts/qwen35-vllm-fla-stages-capsule/v1")))
OUT.mkdir(parents=True, exist_ok=True)

T = 64
HK = 16
DK = 128
DV = 128


def stage(name: str) -> None:
    print(f"[fla-stages] >>> {name}", flush=True)
    (OUT / "stage.log").write_text(name, encoding="ascii")


def dump(name: str, t: torch.Tensor) -> None:
    from safetensors.torch import save_file

    save_file({name: t.detach().float().cpu().contiguous()},
              str(OUT / f"{name}.safetensors"))
    finite = bool(torch.isfinite(t.float()).all().item())
    print(f"[fla-stages] dump {name}: shape={tuple(t.shape)} "
          f"finite={finite} mean_abs={float(t.float().abs().mean()):.6f}",
          flush=True)


def main() -> int:
    from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )
    from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_o
    from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )
    from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
    from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
    from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd

    dev = "cuda"
    dt = torch.bfloat16
    torch.manual_seed(0)

    q = torch.randn(1, T, HK, DK, device=dev, dtype=dt)
    k = torch.randn(1, T, HK, DK, device=dev, dtype=dt)
    v = torch.randn(1, T, HK, DV, device=dev, dtype=dt)
    g = -torch.rand(1, T, HK, device=dev, dtype=torch.float32)
    beta = torch.rand(1, T, HK, device=dev, dtype=dt)
    state = torch.zeros(1, HK, DV, DK, device=dev, dtype=torch.float32)
    cu = torch.tensor([0, T], device=dev, dtype=torch.int32)

    def l2n(x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    q, k = l2n(q), l2n(k)
    for name, t in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        dump(name, t)

    stage("1 chunk_local_cumsum")
    g = chunk_local_cumsum(g, chunk_size=T, cu_seqlens=cu, chunk_indices=None)
    dump("g_cumsum", g)

    stage("2 chunk_scaled_dot_kkt_fwd")
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g, cu_seqlens=cu, chunk_indices=None,
        output_dtype=torch.float32)
    dump("A_kkt", A)

    stage("3 solve_tril")
    A = solve_tril(A=A, cu_seqlens=cu, chunk_indices=None, output_dtype=dt)
    dump("A_solved", A)

    stage("4 recompute_w_u_fwd")
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g_cumsum=g, cu_seqlens=cu,
        chunk_indices=None)
    dump("w", w)
    dump("u", u)

    stage("5 chunk_gated_delta_rule_fwd_h")
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=state, output_final_state=True,
        cu_seqlens=cu, chunk_indices=None)
    dump("h", h)
    dump("v_new", v_new)
    dump("final_state", final_state)

    stage("6 chunk_fwd_o")
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=DK**-0.5,
                    cu_seqlens=cu, chunk_indices=None)
    dump("o", o)

    report = {"schema": "amdgpu-sim.qwen35-vllm-fla-stages.v1",
              "completed": True,
              "finished_at": datetime.now(timezone.utc).isoformat()}
    (OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n",
                                     encoding="ascii")
    print("[fla-stages] all stages completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
