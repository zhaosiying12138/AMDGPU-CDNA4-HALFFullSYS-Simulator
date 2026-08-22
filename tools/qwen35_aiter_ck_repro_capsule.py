#!/usr/bin/env python3
"""Controlled repro capsule: aiter tuned-GEMM dispatch at decode m=1 shapes.

The SGLang second-token defect was root-caused to aiter's machine-global
tuned-GEMM table: drifted entries steered the decode-phase m=1 GEMMs from
the torch/hipBLASLt path onto JIT CK kernels whose gem5 execution is
defective, producing garbage projections (the lane pin AITER_CONFIG_GEMM_BF16
restores exactness by keeping those shapes on the torch path).  This capsule
makes that dispatch difference controllable and measurable:

  --table package   the lane's pinned package-seeded table (expect torch path,
                    exact results -- the currently accepted configuration)
  --table drifted   a frozen copy of the machine-global /tmp table with its
                    extra M in {1,2} gfx950 rows (candidate repro)
  --table synthetic package rows PLUS explicit M in {1,2} rows for the
                    engine's exact decode shapes (N in {5120, 8192}, K=1024),
                    cloned from the drifted M=1 N=8192 K=1024 entry so the CK
                    kernel selection is forced at the accused shape

For each shape it calls the SAME entry the engine's linear layers use
(aiter.tuned_gemm.tgemm.mm) and compares against a float64 host reference
built from identical bf16 inputs.  A case fails when its argmax-relevant
error leaves the bf16 rounding band (mean 0.0025 / max 0.06 observed on the
correct path).  The aiter logger's found/not-found lines are left in stdout
as the dispatch proof.

Exit 0 = every case exact (no repro); exit 2 = at least one case wrong
(repro achieved -- the report names the table, shape, and error signature).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import shutil
from pathlib import Path

# The table override MUST be in place before aiter reads it at import.
PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--table", choices=("package", "drifted", "synthetic"),
                    default="package")
PARSER.add_argument(
    "--output-dir", type=Path, default=Path(
        __import__("os").environ.get(
            "QWEN35_AITER_CK_REPRO_OUTPUT",
            str(Path(__file__).resolve().parents[1]
                / "artifacts/qwen35-aiter-ck-repro-capsule/v1"))))
ARGS = PARSER.parse_args()

ROOT = Path(__file__).resolve().parents[1]
DRIFTED_SRC = Path("/tmp/aiter_configs/bf16_tuned_gemm.csv")
PKG_SRC = (ROOT / "env/conda/rocm-pytorch-v3-fa8414cce688f934f538163621423376c2"
           "542acff3e4d3e403df4340d90fcd6d/lib/python3.12/site-packages/aiter"
           "/configs/bf16_tuned_gemm.csv")

ARGS.output_dir.mkdir(parents=True, exist_ok=True)
TABLE = ARGS.output_dir / f"table_{ARGS.table}.csv"

if ARGS.table == "package":
    shutil.copyfile(PKG_SRC, TABLE)
else:
    # Both drifted and synthetic start from the drifted machine table.
    shutil.copyfile(DRIFTED_SRC, TABLE)
    if ARGS.table == "synthetic":
        import csv
        with open(TABLE) as f:
            rows = list(csv.reader(f))
        hdr, body = rows[0], rows[1:]
        donor = next(r for r in body
                     if r[0] == "gfx950" and r[2] == "1"
                     and r[3] == "8192" and r[4] == "1024")
        have = {tuple(r[:5]) for r in body}
        added = 0
        for m in ("1", "2"):
            for n in ("5120", "8192"):
                key = ("gfx950", "256", m, n, "1024")
                if key in have:
                    continue
                clone = list(donor)
                clone[2], clone[3], clone[4] = m, n, "1024"
                body.append(clone)
                added += 1
        with open(TABLE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            w.writerows(body)
        print(f"[ckrepro] synthetic table: cloned {added} rows for the "
              f"engine decode shapes", flush=True)

os.environ["AITER_CONFIG_GEMM_BF16"] = str(TABLE)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

torch.manual_seed(0)
SHAPES = [(1, 5120, 1024), (1, 8192, 1024), (2, 5120, 1024), (2, 8192, 1024)]


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO)

    from aiter.tuned_gemm import tgemm

    report = {
        "schema": "amdgpu-sim.qwen35-aiter-ck-repro.v1",
        "table": ARGS.table,
        "table_path": str(TABLE),
        "cases": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    artifacts: dict[str, torch.Tensor] = {}
    failures = 0

    for m, n, k in SHAPES:
        a = (torch.randn(m, k) * 0.5).to(torch.bfloat16)
        w = (torch.randn(n, k) * 0.02).to(torch.bfloat16)
        ref = (a.double() @ w.double().T).to(torch.bfloat16)
        dev = tgemm.mm(a.to("cuda"), w.to("cuda")).cpu()
        diff = (dev.float() - ref.float()).abs()
        maxd = float(diff.max())
        meand = float(diff.mean())
        cos = float(torch.nn.functional.cosine_similarity(
            dev.float().reshape(-1), ref.float().reshape(-1), dim=0))
        # Correct-path band measured by the lm_head capsule: mean 0.0025,
        # max 0.058.  Structural corruption sits far outside it.
        ok = meand <= 0.01 and maxd <= 0.25 and cos >= 0.999
        case = {
            "shape": [m, n, k], "max_abs_diff": round(maxd, 6),
            "mean_abs_diff": round(meand, 8), "cos": round(cos, 6), "ok": ok,
        }
        report["cases"].append(case)
        artifacts[f"dev_{m}_{n}_{k}"] = dev.float()
        artifacts[f"ref_{m}_{n}_{k}"] = ref.float()
        if not ok:
            failures += 1
        print(f"[ckrepro] table={ARGS.table} M={m} N={n} K={k}: "
              f"max={maxd:.6f} mean={meand:.6f} cos={cos:.6f} ok={ok}",
              flush=True)

    report["reproduced"] = failures > 0
    report["passed"] = failures == 0
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_file(artifacts, str(ARGS.output_dir / "tensors.safetensors"))
    (ARGS.output_dir / "report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="ascii")
    print(f"[ckrepro] table={ARGS.table} reproduced={report['reproduced']}",
          flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
