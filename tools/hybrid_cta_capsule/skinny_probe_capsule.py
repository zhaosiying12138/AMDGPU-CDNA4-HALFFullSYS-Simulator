#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 AMDGPU-CDNA4-HALFFullSYS-Simulator contributors
# Full license text: LICENSE at the repository root.
"""Skinny-GEMM shape scanner: the vLLM 0.8B ctx16 defect isolator.

The failing ctx16 lane uniquely dispatches grid=(16384,4,1) wg=(64,4,1)
lds=163840 kernels (96x = 24 layers x 4); vLLM switches to its skinny-GEMM
family only when m % 16 == 0 (ctx16 chunk m=16 qualifies, ctx17's 17 does
not).  This capsule re-execs itself with the HIP name shim preloaded and
then runs vLLM's own rocm_unquantized_gemm_impl over candidate (m, n, k)
shapes, comparing each against a CPU reference -- no engine needed.
"""
import json
import os
import subprocess
import sys

SHIM = "/tmp/hipnameshim.so"
LOG = os.environ.get("HIPNAME_LOG", "/tmp/hipnames.log")


def main() -> int:
    if os.environ.get("SKINNY_PROBE_SHIMMED") != "1":
        env = dict(os.environ)
        preload = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = f"{SHIM}:{preload}" if preload else SHIM
        env["SKINNY_PROBE_SHIMMED"] = "1"
        env["HIPNAME_LOG"] = LOG
        os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)], env)

    os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
    os.environ.setdefault("TRITON_DEFAULT_BACKEND", "amd")
    os.environ.setdefault("VLLM_PLUGINS", "")
    os.environ.setdefault("SAGR_TRITON_FAST_AUTOTUNE", "1")
    import torch

    try:
        from vllm.model_executor.layers.utils import \
            rocm_unquantized_gemm_impl
        have_vllm = True
    except Exception as exc:
        print("VLLM-IMPORT-FALLBACK", repr(exc)[:120], flush=True)
        have_vllm = False

    torch.manual_seed(0)
    dev = "cuda"
    results = []
    # m % 16 == 0 selects vLLM's skinny family; sweep the small-n, k>512
    # space the 0.8B per-layer projections live in, plus m=17 control.
    for m in (16, 17):
        for k in (1024,):
            for n in (16, 32, 64, 96, 128):
                decoy = torch.full((n * m + 64,), 777.0,
                                   dtype=torch.bfloat16, device=dev)
                del decoy
                x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
                w = torch.randn(n, k, dtype=torch.bfloat16, device=dev)
                if have_vllm:
                    got = rocm_unquantized_gemm_impl(x, w)
                else:
                    got = torch.nn.functional.linear(x, w)
                ref = (x.float() @ w.float().T)
                g = got.float()
                nan_mask = torch.isnan(g)
                n_nan = int(nan_mask.sum())
                finite_wrong = int(((~nan_mask) & ((g - ref).abs() > 0.5)).sum())
                # distribution of NaN over (rows=m, cols=n): uniform or banded?
                rows_with_nan = sorted(set(divmod(int(i), n)[0] for i in nan_mask.flatten().nonzero().flatten().tolist()))
                untouched = bool((g == 777.0).float().mean() > 0.99)
                ok = (not untouched) and n_nan == 0 and torch.allclose(g, ref, atol=2e-1, rtol=2e-1)
                md = (g - ref).abs().max().item() if n_nan == 0 else float("nan")
                rec = {"m": m, "n": n, "k": k, "match": bool(ok),
                       "nan_count": n_nan, "finite_wrong": finite_wrong,
                       "untouched_777": untouched, "nan_rows": rows_with_nan[:8],
                       "sample": [round(v, 2) for v in g.flatten()[:6].tolist()]}
                results.append(rec)
                print("PROBE " + json.dumps(rec), flush=True)

    out = os.environ.get("SKINNY_PROBE_OUTPUT", "/tmp/skinny_probe_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("PROBE-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
