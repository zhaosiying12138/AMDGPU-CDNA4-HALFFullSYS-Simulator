#!/usr/bin/env python3
"""Neighbor-corruption probe: the real Triton conv update vs adjacent memory.

The slot journal shows layer 0's temporal pool correct at forward_decode
entry and different by the packed_decode call, with only the Triton
causal_conv1d_update in between; plain-allocation aliasing is cleared.
The remaining explanation is that the conv-update kernel's device stores
exceed the conv-state tensor's bounds inside the simulator and land in
the adjacent allocation.  This capsule allocates conv pool, temporal
pool and a sentinel in production order and shapes, runs the REAL
production kernel on the conv slot, then checks every neighbor.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(
        os.environ.get("CONV_NEIGHBOR_OUTPUT",
                       str(ROOT / "artifacts/qwen35-conv-neighbor/v1"))))
    args = parser.parse_args()

    from sglang.kernels.ops.mamba.causal_conv1d_triton import (
        causal_conv1d_update,
    )

    device = torch.device("cuda")
    slots, dim, width = 5, 6144, 4
    heads, kdim, vdim = 16, 128, 128

    conv = torch.randn((slots, dim, width), dtype=torch.bfloat16, device=device)
    temporal = torch.full(
        (slots, heads, kdim, vdim), 0.5, dtype=torch.float32, device=device
    )
    sentinel = torch.full(
        (slots, dim, width), 7.0, dtype=torch.bfloat16, device=device
    )
    weight = torch.randn((dim, width), dtype=torch.bfloat16, device=device)
    bias = torch.randn((dim,), dtype=torch.bfloat16, device=device)
    x = torch.randn((1, dim), dtype=torch.bfloat16, device=device)
    indices = torch.tensor([2], dtype=torch.int32, device=device)

    temporal_before = temporal.clone()
    sentinel_before = sentinel.clone()
    conv_before_slot = conv[2].clone()

    out = causal_conv1d_update(
        x,
        conv,
        weight,
        bias,
        activation="silu",
        conv_state_indices=indices,
    )
    torch.cuda.synchronize()

    temporal_changed = not bool(torch.equal(temporal, temporal_before))
    sentinel_changed = not bool(torch.equal(sentinel, sentinel_before))
    conv_changed = not bool(torch.equal(conv[2], conv_before_slot))
    result = {
        "schema": "amdgpu-sim.qwen35-conv-neighbor.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "temporal_changed": temporal_changed,
        "sentinel_changed": sentinel_changed,
        "conv_slot_changed": conv_changed,
        "output_shape": list(out.shape),
        "neighbor_corrupted": temporal_changed or sentinel_changed,
        "pointers": {
            "conv": hex(conv.data_ptr()),
            "temporal": hex(temporal.data_ptr()),
            "sentinel": hex(sentinel.data_ptr()),
        },
    }
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
