#!/usr/bin/env python3
"""Single-kernel bisect: chunk_gated_delta_rule_fwd_h on dumped-good inputs.

CP-0112 pinned the vLLM wild-store crash to this one kernel: stages 1-4 of
the FLA chain complete with finite outputs, and fwd_h then faults a
buffer_store_dwordx4 to a wild address.  This capsule re-drives fwd_h ALONE
from the stage capsule's dumped tensors (k/w/u/g, regenerated determin-
istically to the same seed), first at full shape (16 heads), then with a
shrinking head count (--heads), so the fault's dependence on the grid's
head dimension isolates cheaply.  A stage marker prints before each launch;
whichever marker is last when a panic hits is the failing shape.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get(
    "QWEN35_VLLM_FWDH_OUTPUT",
    str(ROOT / "artifacts/qwen35-vllm-fwdh-capsule/v1")))
OUT.mkdir(parents=True, exist_ok=True)

T = 64
DK = 128
DV = 128


def stage(name: str) -> None:
    print(f"[fwdh] >>> {name}", flush=True)
    (OUT / "stage.log").write_text(name, encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, nargs="*", default=[1, 4, 8, 16])
    args = parser.parse_args()

    from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )

    dev = "cuda"
    dt = torch.bfloat16
    torch.manual_seed(0)

    for h in args.heads:
        stage(f"fwd_h heads={h}")
        k = torch.randn(1, T, h, DK, device=dev, dtype=dt)
        w = (torch.randn(1, T, h, DK, device=dev, dtype=dt) * 0.001)
        u = torch.randn(1, T, h, DV, device=dev, dtype=dt)
        g = -torch.rand(1, T, h, device=dev, dtype=torch.float32)
        state = torch.zeros(1, h, DV, DK, device=dev, dtype=torch.float32)
        cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
        h_out, v_new, final = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g, initial_state=state,
            output_final_state=True, cu_seqlens=cu, chunk_indices=None)
        torch.cuda.synchronize()
        finite = all(bool(torch.isfinite(t.float()).all().item())
                     for t in (h_out, v_new, final))
        print(f"[fwdh] heads={h}: finite={finite} "
              f"h_mean={float(h_out.float().abs().mean()):.6f}", flush=True)
        del k, w, u, g, state, h_out, v_new, final
        torch.cuda.empty_cache()

    report = {"schema": "amdgpu-sim.qwen35-vllm-fwdh.v1",
              "completed": True,
              "finished_at": datetime.now(timezone.utc).isoformat()}
    (OUT / "report.json").write_text(
        __import__("json").dumps(report, indent=1) + "\n", encoding="ascii")
    print("[fwdh] all shapes completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
