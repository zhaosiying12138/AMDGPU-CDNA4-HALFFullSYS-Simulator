#!/usr/bin/env python3
"""Minimal reproducer for the aiter prebuilt batch-prefill attention path.

The SGLang TP1 lane selects ``--attention-backend aiter``; its NHD
full-attention layers execute aiter's prebuilt CK-tile
``mha_batch_prefill`` gfx950 kernels.  In the layer gate those layers
diverge (layer 3 hidden rel_l2 0.119) and a later dispatch hits a gem5
decode fatal.  This capsule calls the same entry point SGLang uses
(``aiter.ops.mha.mha_batch_prefill_func``) with small deterministic
synthetic inputs in the exact Qwen3.5 NHD geometry (8 q heads, 2 kv
heads, head dim 256) and compares against a float32 torch reference of
paged causal GQA attention computed on CPU from the same inputs.  It
gives a seconds-scale reproducer for both symptoms without a 15-minute
lane, and doubles as the regression capsule once a gem5 repair lands.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]


def _metrics(actual: torch.Tensor, reference: torch.Tensor, atol, rtol) -> dict:
    a = actual.float().reshape(-1)
    b = reference.float().reshape(-1)
    delta = a - b
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(delta.abs().max().item()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(delta).item()
            / max(torch.linalg.vector_norm(b).item(), 1.0e-30)
        ),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(a[None], b[None]).item()
        ),
        "mismatch_count": int(
            ((delta.abs() > (atol + rtol * b.abs())).sum().item())
        ),
        "atol": atol,
        "rtol": rtol,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            __import__("os").environ.get(
                "QWEN35_AITER_ATTN_OUTPUT",
                str(ROOT / "artifacts/qwen35-aiter-attn-capsule/20260820-v1"),
            )
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # Import only after the lane has installed its backend and simulator.
    from aiter.ops.mha import mha_batch_prefill_func

    seq_lens = args.seq_lens
    total_q = sum(seq_lens)
    cu_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), 0)), dtype=torch.int32
    )
    # Prefill: KV length equals the q length of each sequence; pack each
    # sequence's pages contiguously with block_size granularity.
    pages_per_seq = [(l + args.block_size - 1) // args.block_size for l in seq_lens]
    kv_indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(pages_per_seq), 0)), dtype=torch.int32
    )
    page_ids = torch.arange(sum(pages_per_seq), dtype=torch.int32)
    num_pages = sum(pages_per_seq) + 1

    q = (torch.randn(total_q, args.q_heads, args.head_dim) * 0.5).to(torch.bfloat16)
    k_cache = (
        torch.randn(num_pages, args.block_size, args.kv_heads, args.head_dim) * 0.5
    ).to(torch.bfloat16)
    v_cache = (
        torch.randn(num_pages, args.block_size, args.kv_heads, args.head_dim) * 0.5
    ).to(torch.bfloat16)
    scale = args.head_dim ** -0.5

    q_d = q.to(device)
    k_d = k_cache.to(device)
    v_d = v_cache.to(device)
    last_page_lens = [
        l - (p - 1) * args.block_size for l, p in zip(seq_lens, pages_per_seq)
    ]
    cu_q_d = cu_q.to(device)
    kv_indptr_d = kv_indptr.to(device)
    page_ids_d = page_ids.to(device)
    last_page_d = torch.tensor(last_page_lens, dtype=torch.int32).to(device)

    out = mha_batch_prefill_func(
        q_d,
        k_d,
        v_d,
        cu_q_d,
        kv_indptr_d,
        page_ids_d,
        max(seq_lens),
        max(seq_lens),
        softmax_scale=scale,
        causal=True,
        kv_last_page_lens=last_page_d,
    )
    if isinstance(out, tuple):
        out = out[0]
    torch.cuda.synchronize()
    out_sim = out.detach().cpu()

    # float32 CPU reference of paged causal GQA attention.
    ref = torch.empty_like(q, dtype=torch.float32)
    for si, length in enumerate(seq_lens):
        qo0 = int(cu_q[si])
        page0 = int(kv_indptr[si])
        # gather this sequence's K/V rows from its pages, in order
        k_rows = []
        v_rows = []
        for p in range(pages_per_seq[si]):
            base = (page0 + p) * args.block_size
            k_rows.append(
                k_cache.view(-1, args.kv_heads, args.head_dim)[
                    base : base + args.block_size
                ]
            )
            v_rows.append(
                v_cache.view(-1, args.kv_heads, args.head_dim)[
                    base : base + args.block_size
                ]
            )
        K = torch.cat(k_rows)[:length].float()  # [len, kv_heads, dim]
        V = torch.cat(v_rows)[:length].float()
        for h in range(args.q_heads):
            kvh = h // (args.q_heads // args.kv_heads)
            scores = (
                q[qo0 : qo0 + length, h].float() @ K[:, kvh].T * scale
            )  # [len, len]
            mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            ref[qo0 : qo0 + length, h] = probs @ V[:, kvh]

    result = {
        "schema": "amdgpu-sim.qwen35-aiter-attn-capsule.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "seq_lens": seq_lens,
            "block_size": args.block_size,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "seed": args.seed,
            "scale": scale,
        },
        "sim_vs_reference": _metrics(
            out_sim, ref.to(torch.bfloat16).float(), 0.015625, 0.02
        ),
        "q_sha256": hashlib.sha256(
            q.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "out_sha256": hashlib.sha256(
            out_sim.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    save_file(
        {
            "q": q.contiguous(),
            "k_cache": k_cache.contiguous(),
            "v_cache": v_cache.contiguous(),
            "out_sim": out_sim.contiguous(),
            "reference": ref.to(torch.bfloat16).contiguous(),
        },
        str(args.output_dir / "tensors.safetensors"),
        metadata={"schema": result["schema"]},
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    m = result["sim_vs_reference"]
    if m["relative_l2_error"] > 0.03 or m["cosine_similarity"] < 0.98:
        print("AITER ATTN CAPSULE: DIVERGES from fp32 reference", flush=True)
        return 1
    print("AITER ATTN CAPSULE: matches fp32 reference", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
