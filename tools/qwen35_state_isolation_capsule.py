#!/usr/bin/env python3
"""Isolate the recurrent-state divergence: NVIDIA inputs, simulator execution.

The layer gate's ordinal-16 ``recurrent_state`` mismatch could be either (a)
bf16 input noise from earlier operators propagating through the state update
or (b) a genuine cross-architecture divergence inside the state-update path
itself.  This capsule separates the two: it replays the pinned
``chunk_gated_delta_rule`` operator on the simulator with the NVIDIA golden's
*own* q/k/v/g/beta tensors (not the simulator's) and a zeroed initial state,
then compares the produced final state and output against the golden's
``recurrent_state``/``gdn_recurrent_output``.

If the divergence disappears, it was input propagation and the gate's 1e-4
atol for this fp32 state is miscalibrated; if it remains, the state-update
path itself diverges and must be root-caused further.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = (
    ROOT / "artifacts/qwen35-nvidia-operator-golden/20260819-prefill2-layer0-v3"
)


def _sha(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    a = actual.float().reshape(-1)
    b = reference.float().reshape(-1)
    delta = a - b
    denom = torch.linalg.vector_norm(b).item()
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(delta.abs().max().item()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(delta).item() / max(denom, 1.0e-30)
        ),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(a[None], b[None]).item()
        ),
        "over_gate_tol": int(
            ((delta.abs() > (1.0e-4 + 0.03 * b.abs().reshape(-1)))).sum().item()
        ),
        "actual_sha256": _sha(actual),
        "reference_sha256": _sha(reference),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            __import__("os").environ.get(
                "QWEN35_STATE_ISOLATION_OUTPUT",
                str(ROOT / "artifacts/qwen35-state-isolation/20260819-layer0-v1"),
            )
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pool-slots", type=int, default=3)
    args = parser.parse_args()

    golden = load_file(str(args.golden / "results.safetensors"))
    device = torch.device(args.device)
    # Import only after the lane has installed its backend and simulator.
    import sglang.kernels.ops.attention.fla.chunk as chunk_module

    q = golden["gdn_q_raw"].to(device=device, dtype=torch.bfloat16)
    k = golden["gdn_k_raw"].to(device=device, dtype=torch.bfloat16)
    v = golden["gdn_v_raw"].to(device=device, dtype=torch.bfloat16)
    g = golden["gdn_g"].to(device=device, dtype=torch.float32)
    beta = golden["gdn_beta_output"].to(device=device, dtype=torch.float32)

    # Layer 0 prefill from an empty cache: the selected pool slot starts at
    # zero on both sides, so the zeroed pool reproduces the golden's initial
    # condition without needing the golden to have stored it.  The pool is
    # [slots, H, K, V]; q is [B, T, H, K] and v is [B, T, H, V].
    heads, dim_k, dim_v = q.shape[2], q.shape[3], v.shape[3]
    state_pool = torch.zeros(
        (args.pool_slots, heads, dim_k, dim_v), dtype=torch.float32, device=device
    )
    indices = torch.full(
        (1,), 2, device=device, dtype=torch.int32
    )
    cu_seqlens = torch.tensor([0, q.shape[1]], device=device, dtype=torch.int32)

    output, _, _ = chunk_module.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state_pool,
        initial_state_indices=indices,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    replayed_o = output[0].detach().cpu().contiguous()
    replayed_state = state_pool[2].detach().cpu().contiguous()

    result = {
        "schema": "amdgpu-sim.qwen35-state-isolation-capsule.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "replayed",
        "golden": str(args.golden),
        "inputs_from": "nvidia-golden",
        "initial_state": "zeroed-pool-slot-2",
        "output_vs_golden": _metrics(replayed_o, golden["gdn_recurrent_output"]),
        "state_vs_golden": _metrics(replayed_state, golden["recurrent_state"]),
    }
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    save_file(
        {"replayed_o": replayed_o, "replayed_state": replayed_state},
        str(args.output_dir / "tensors.safetensors"),
        metadata={"schema": result["schema"]},
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
