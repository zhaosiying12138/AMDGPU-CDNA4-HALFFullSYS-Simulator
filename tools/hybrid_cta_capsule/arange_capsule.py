#!/usr/bin/env python3
"""Minimal reproduction capsule for the hybrid CTA executor on the exact
kernel class that crashed the model lane: a device-side torch.arange fill
(grid 65536 x workgroup 64) whose output feeds RoPE's inv-freq computation.

Runs the fill via torch on the lane's simulator and writes the output
buffer's sha256 + first divergence info to result.json.  Run once per mode
(accurate fastwrap, hybrid) via the lane runner; byte-identical outputs
required.
"""
import argparse
import faulthandler
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
faulthandler.enable()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(
        os.environ.get("ARANGE_CAPSULE_OUTPUT",
                       str(ROOT / "artifacts/hybrid-cta-capsule-v2/arange"))))
    args = parser.parse_args()

    n = int(os.environ.get("ARANGE_N", "65536"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("ARANGE_MODE", "simple") == "invfreq":
        # rotary_embedding/base.py:161 verbatim: tensor-exponent pow
        t = torch.arange(0, n, 2, dtype=torch.float, device="cuda")
        inv = 1.0 / (10000.0 ** (t / n))
    else:
        t = torch.arange(n, dtype=torch.float32, device="cuda")
        inv = 1.0 / (t + 1.0)
    host = inv.cpu().numpy().tobytes()

    result = {
        "schema": "amdgpu-sim.hybrid-cta-capsule.arange.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n": n,
        "mode": os.environ.get("ARANGE_MODE", "simple"),
        "output_sha256": hashlib.sha256(host).hexdigest(),
        "first_values": [float(v) for v in inv[:8].cpu()],
        "gem5": os.environ.get("SAGR_MANAGED_GEM5", ""),
    }
    out = args.output_dir / "result.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
