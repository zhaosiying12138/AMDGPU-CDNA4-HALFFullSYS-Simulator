#!/usr/bin/env python3
"""Device-allocation aliasing probe for the decode state pools.

The slot journal proved layer 0's temporal pool holds the correct state at
forward_decode entry and a different one by the packed_decode call, with
only causal_conv1d_update (a writer of the SEPARATE conv pool) running in
between and no host-side copy in the interval.  The remaining explanation
is that distinct device allocations alias inside the simulator's memory
layout, so a conv-pool write lands in temporal storage.  This capsule
reproduces the production allocation order and shapes (conv pool first,
then the temporal pool), writes distinct constants to each, and reads
everything back.
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
        os.environ.get("ALLOC_ALIAS_OUTPUT",
                       str(ROOT / "artifacts/qwen35-alloc-alias/v1"))))
    args = parser.parse_args()

    device = torch.device("cuda")
    slots = 5
    # Production layer-0 geometry: conv [slots, 6144, 3] bf16, temporal
    # [slots, 16, 128, 128] fp32 (MambaPool.State layout).
    conv = torch.full((slots, 6144, 3), 0.25, dtype=torch.float32, device=device)
    temporal = torch.full((slots, 16, 128, 128), 0.5, dtype=torch.float32, device=device)
    extra = torch.full((slots, 6144, 3), 0.75, dtype=torch.float32, device=device)
    torch.cuda.synchronize()

    ptrs = {
        "conv": conv.data_ptr(),
        "temporal": temporal.data_ptr(),
        "extra": extra.data_ptr(),
    }
    spans = {
        k: (p, p + n.element_size() * n.numel())
        for (k, p), n in zip(ptrs.items(), (conv, temporal, extra))
    }
    overlaps = []
    keys = list(spans)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            lo = max(spans[a][0], spans[b][0])
            hi = min(spans[a][1], spans[b][1])
            if lo < hi:
                overlaps.append({"a": a, "b": b, "bytes": hi - lo})

    # Simulate the decode conv roll on slot 2 and check every pool.
    conv[2, :, 1:] = conv[2, :, :-1]
    conv[2, :, -1] = 0.125
    torch.cuda.synchronize()
    conv_ok = bool((conv == 0.125)[:, :, -1].all().item())
    temporal_unchanged = bool((temporal == 0.5).all().item())
    extra_unchanged = bool((extra == 0.75).all().item())

    result = {
        "schema": "amdgpu-sim.qwen35-alloc-alias.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pointers": {k: hex(v) for k, v in ptrs.items()},
        "span_overlaps": overlaps,
        "conv_roll_visible": conv_ok,
        "temporal_unchanged_after_conv_write": temporal_unchanged,
        "extra_unchanged": extra_unchanged,
        "aliased": bool(overlaps) or not temporal_unchanged or not extra_unchanged,
    }
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
