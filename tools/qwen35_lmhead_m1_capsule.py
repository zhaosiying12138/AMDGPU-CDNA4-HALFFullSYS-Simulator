#!/usr/bin/env python3
"""Differential capsule for the decode-phase lm_head matmul (m=1).

The layer gate passes every layer boundary for both decode rows (cos >=
0.999) yet the greedy engine emits token 3 at decode row 2, where golden
ranks 3 at position 5354 (logit 5.43 vs top-1 20.01) and no perturbation
inside the gate's pass-band flips the argmax off 27841.  The remaining
suspect is the engine's own final projection: ``torch.matmul(hidden_bf16,
weight.T)`` at M=1, K=1024, N=248320 -- the m=1 GEMM/GEMV dispatch that
prefill (M=2) never exercises.

This capsule runs THE SAME expression with THE SAME weights and THE SAME
golden final-norm hidden states on the simulated device and compares
against a float64 host reference computed from identical inputs:

  - row 1 (M=1 golden prefill output)   -> engine path proved correct E2E
  - row 2 (M=1 golden decode output)    -> the accused step
  - rows 0..3 stacked (M=4)             -> fat-M control
  - M=1 with a fresh random hidden      -> separates data from shape

Exit 0 when every case matches the reference within bf16 tolerance; exit 1
on the first case whose top-1 diverges from the reference top-1 or whose
max deviation exceeds the tolerance, with the failing signature dumped.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = (
    ROOT
    / "artifacts/qwen35-nvidia-golden/20260812-decode4-max24-v1/results.safetensors"
)
DEFAULT_MODEL = (
    ROOT / "models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
)
EMBED_KEY = "model.language_model.embed_tokens.weight"
# bf16 output rounding plus fp32-accumulate difference: the reference and the
# device agree to well under 0.05 on the prefill shapes (proved E2E); anything
# past that is structural corruption, not rounding.
ATOL = 0.05


def topk_summary(logits: torch.Tensor, k: int = 5) -> dict:
    flat = logits.detach().float().reshape(-1)
    top = torch.topk(flat, min(k, flat.numel()))
    return {
        "top_ids": [int(x) for x in top.indices.tolist()],
        "top_vals": [round(float(v), 4) for v in top.values.tolist()],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            __import__("os").environ.get(
                "QWEN35_LMHEAD_M1_OUTPUT",
                str(ROOT / "artifacts/qwen35-lmhead-m1-capsule/v1"),
            )
        ),
    )
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with safe_open(str(args.model), framework="pt", device="cpu") as ckpt:
        weight = ckpt.get_tensor(EMBED_KEY)
    with safe_open(str(args.golden), framework="pt", device="cpu") as golden:
        final_norm = golden.get_tensor("final_norm")

    n_vocab, k_dim = weight.shape
    report = {
        "schema": "amdgpu-sim.qwen35-lmhead-m1-capsule.v1",
        "weight_shape": [int(n_vocab), int(k_dim)],
        "weight_dtype": str(weight.dtype),
        "golden_shape": list(final_norm.shape),
        "atol": ATOL,
        "cases": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    weight_gpu = weight.to("cuda")
    weight_t_gpu = weight_gpu.T

    cases = []
    # (label, hidden) pairs; every case exercises the engine's exact
    # expression: torch.matmul(hidden.to(bf16), weight.T).
    cases.append(("m1_golden_row1", final_norm[1:2].clone()))
    cases.append(("m1_golden_row2", final_norm[2:3].clone()))
    cases.append(("m1_golden_row3", final_norm[3:4].clone()))
    cases.append(("m4_golden_rows0_3", final_norm[0:4].clone()))
    random_hidden = torch.randn(
        (1, k_dim), dtype=torch.float32
    ).to(torch.bfloat16) * final_norm.float().std()
    cases.append(("m1_random", random_hidden))

    device_ok = torch.cuda.is_available()
    report["cuda_available"] = bool(device_ok)

    any_fail = False
    artifacts: dict[str, torch.Tensor] = {}
    for label, hidden in cases:
        hidden_bf16 = hidden.to(torch.bfloat16)
        # float64 host reference from the identical bf16 inputs.
        ref = (
            hidden_bf16.double() @ weight.double().T
        )
        logits = torch.matmul(
            hidden_bf16.to(weight.dtype).to("cuda"), weight_t_gpu
        ).cpu()
        diff = (logits.double() - ref).abs()
        max_diff = float(diff.max())
        ref_top = topk_summary(ref)
        dev_top = topk_summary(logits)
        case = {
            "label": label,
            "shape": list(hidden_bf16.shape),
            "max_abs_diff": round(max_diff, 6),
            "mean_abs_diff": round(float(diff.mean()), 8),
            "ref": ref_top,
            "device": dev_top,
            "argmax_agree": ref_top["top_ids"][0] == dev_top["top_ids"][0],
            "within_atol": max_diff <= ATOL,
            "logit_at_3": {
                "ref": round(float(ref.reshape(-1)[3]), 4),
                "device": round(float(logits.reshape(-1)[3]), 4),
            },
            "logit_at_27841": {
                "ref": round(float(ref.reshape(-1)[27841]), 4),
                "device": round(float(logits.reshape(-1)[27841]), 4),
            },
        }
        report["cases"].append(case)
        artifacts[f"{label}_logits"] = logits.float()
        artifacts[f"{label}_ref"] = ref.float()
        if not case["argmax_agree"] or not case["within_atol"]:
            any_fail = True
        print(
            f"[lmhead-m1] {label}: max_diff={case['max_abs_diff']} "
            f"argmax_agree={case['argmax_agree']} "
            f"ref_top1={case['ref']['top_ids'][0]} "
            f"dev_top1={case['device']['top_ids'][0]}",
            flush=True,
        )

    report["passed"] = not any_fail
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_file(artifacts, str(args.output_dir / "logits.safetensors"))
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="ascii"
    )
    print(
        f"[lmhead-m1] passed={report['passed']} -> {args.output_dir}",
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
