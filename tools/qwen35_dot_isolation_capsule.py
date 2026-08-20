#!/usr/bin/env python3
"""Isolate MFMA semantics with a Triton tl.dot differential.

The aiter CK attention capsule diverges on every multi-key token while the
single-key identity case is exact, pointing at the matrix-multiply path.
This capsule runs a minimal Triton dot in the exact MFMA shapes the CK
kernels use (32x32x16 and 16x16x16, bf16 inputs, fp32 accumulate) through
the normal Triton launch path on the simulator and compares against a
float32 torch reference.  It separates "v_mfma semantics are wrong in
gem5" from "the CK pipeline around the MFMA is wrong": Triton lowers
tl.dot in these shapes straight to v_mfma_f32_32x32x16_bf16 /
v_mfma_f32_16x16x16_bf16 on gfx950.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]


@triton.jit
def _dot_kernel(a_ptr, b_ptr, c_ptr, ACCUM: tl.constexpr):
    offs_m = tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    offs_k = tl.arange(0, 16)
    a = tl.load(a_ptr + offs_m[:, None] * 16 + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * 32 + offs_n[None, :])
    if ACCUM:
        acc = tl.load(c_ptr + offs_m[:, None] * 32 + offs_n[None, :])
    else:
        acc = tl.zeros((32, 32), dtype=tl.float32)
    acc = tl.dot(a, b, acc)
    tl.store(c_ptr + offs_m[:, None] * 32 + offs_n[None, :], acc)


@triton.jit
def _dot16_kernel(a_ptr, b_ptr, c_ptr):
    offs_m = tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    offs_k = tl.arange(0, 16)
    a = tl.load(a_ptr + offs_m[:, None] * 16 + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * 16 + offs_n[None, :])
    acc = tl.zeros((16, 16), dtype=tl.float32)
    acc = tl.dot(a, b, acc)
    tl.store(c_ptr + offs_m[:, None] * 16 + offs_n[None, :], acc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            __import__("os").environ.get(
                "QWEN35_DOT_OUTPUT",
                str(ROOT / "artifacts/qwen35-dot-isolation/20260820-v1"),
            )
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    results = {}
    tensors = {}
    for name, m, n, k, kernel in (
        ("dot_32x32x16", 32, 32, 16, _dot_kernel),
        ("dot_16x16x16", 16, 16, 16, _dot16_kernel),
    ):
        for accum in ((False, True) if "32" in name else (False,)):
            a = (torch.randn(m, k) * 0.5).to(torch.bfloat16)
            b = (torch.randn(k, n) * 0.5).to(torch.bfloat16)
            ref = a.float() @ b.float()
            tag = name + ("_accum" if accum else "")
            if accum:
                c0 = (torch.randn(m, n) * 0.5).to(torch.float32)
                ref = ref + c0
            else:
                c0 = torch.zeros(m, n, dtype=torch.float32)
            a_d = a.to(device)
            b_d = b.to(device)
            c_d = c0.to(device)
            if "32" in name:
                kernel[(1,)](a_d, b_d, c_d, ACCUM=accum)
            else:
                kernel[(1,)](a_d, b_d, c_d)
            torch.cuda.synchronize()
            sim = c_d.detach().cpu()
            delta = (sim - ref).abs()
            results[tag] = {
                "max_abs_error": float(delta.max().item()),
                "relative_l2_error": float(
                    delta.norm().item() / max(ref.norm().item(), 1e-30)
                ),
                "sim_sha256": hashlib.sha256(
                    sim.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "ref_sha256": hashlib.sha256(
                    ref.contiguous().numpy().tobytes()
                ).hexdigest(),
            }
            tensors[f"a_{tag}"] = a.contiguous()
            tensors[f"b_{tag}"] = b.contiguous()
            tensors[f"sim_{tag}"] = sim.contiguous()
            tensors[f"ref_{tag}"] = ref.contiguous()

    payload = {
        "schema": "amdgpu-sim.qwen35-dot-isolation.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    save_file(
        tensors, str(args.output_dir / "tensors.safetensors"),
        metadata={"schema": payload["schema"]},
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    bad = [k for k, v in results.items() if v["max_abs_error"] > 0.01]
    if bad:
        print("DOT ISOLATION: divergent:", bad, flush=True)
        return 1
    print("DOT ISOLATION: all match", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
