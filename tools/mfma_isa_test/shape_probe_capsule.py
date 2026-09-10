#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 AMDGPU-CDNA4-HALFFullSYS-Simulator contributors
# Full license text: LICENSE at the repository root.
import json, os, torch
from pathlib import Path
out = Path(os.environ.get("MFMA_PROBE_OUTPUT", "/tmp/shape-art")); out.mkdir(parents=True, exist_ok=True)
dev = "cuda"
from vllm import _custom_ops as vops
results = {}
for (M, N, K) in [(16,64,64),(16,32,64),(16,64,128),(16,64,256),(16,64,512),(16,64,1024),
                  (16,128,64),(16,64,4096),(32,64,1024),(64,64,1024),(128,64,1024)]:
    try:
        A = torch.randn(M, K, dtype=torch.float16).to(dev)
        B = torch.randn(N, K, dtype=torch.float16).to(dev)
        t = vops.wvSplitKrc(A, B, 1)
        results[f"{M}x{N}x{K}"] = "OK " + str(tuple(t.shape)) + " " + str(t.dtype)
    except Exception as e:
        results[f"{M}x{N}x{K}"] = "ERR " + str(e)[:60]
print(json.dumps(results, indent=1), flush=True)
(out / "result.json").write_text(json.dumps(results))
class R: pass
