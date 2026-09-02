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

# Replicate the engine's init sequence up to the RoPE crash: the layer-gate
# log shows "[aiter] import module_aiter_core" immediately before the
# segfault, so the corrupting kernel (if any) runs during aiter import.
if os.environ.get("ARANGE_PREIMPORT_AITER", "0") == "1":
    import aiter  # noqa: F401
faulthandler.enable()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(
        os.environ.get("ARANGE_CAPSULE_OUTPUT",
                       str(ROOT / "artifacts/hybrid-cta-capsule-v2/arange"))))
    args = parser.parse_args()

    n = int(os.environ.get("ARANGE_N", "65536"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mode = os.environ.get("ARANGE_MODE", "simple")
    theta = float(os.environ.get("ARANGE_THETA", "10000000"))
    repeats = int(os.environ.get("ARANGE_REPEATS", "3"))
    sweep = [int(x) for x in
             os.environ.get("ARANGE_SWEEP",
                            "65536,4096,512,256,128,96,64,32,16").split(",")]
    import hashlib as _h
    per_size = {}
    for dim in sweep:
        for rep in range(repeats):
            # OOB canary: guard pages around the working buffer, on device.
            guard = torch.full((4096,), 3.14159, dtype=torch.float32,
                               device="cuda")
            if mode == "invfreq":
                # rotary_embedding/base.py:161 verbatim chain
                t = torch.arange(0, dim, 2, dtype=torch.float, device="cuda")
                inv = 1.0 / (theta ** (t / dim))
                ref = 1.0 / (theta ** (torch.arange(0, dim, 2,
                                 dtype=torch.float) / dim))
            else:
                t = torch.arange(dim, dtype=torch.float32, device="cuda")
                inv = 1.0 / (t + 1.0)
                ref = 1.0 / (torch.arange(dim, dtype=torch.float32) + 1.0)
            canary_ok = bool((guard == 3.14159).all().item())
            host = inv.cpu().numpy().tobytes()
            sha = _h.sha256(host).hexdigest()
            match = bool((inv.cpu() == ref).all().item())
            per_size.setdefault(dim, []).append(
                {"rep": rep, "sha256": sha, "matches_cpu": match,
                 "canary_intact": canary_ok})

    result = {
        "schema": "amdgpu-sim.hybrid-cta-capsule.arange.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n": n,
        "mode": mode,
        "repeats": repeats,
        "output_sha256": hashlib.sha256(host).hexdigest(),
        "first_values": [float(v) for v in inv[:8].cpu()],
        "gem5": os.environ.get("SAGR_MANAGED_GEM5", ""),
    }
    result["per_size"] = per_size
    host = json.dumps(per_size, sort_keys=True).encode()
    out = args.output_dir / "result.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
