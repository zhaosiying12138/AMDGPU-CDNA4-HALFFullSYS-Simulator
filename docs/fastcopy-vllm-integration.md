# Fast-copy and vLLM Integration Handoff

Updated: 2026-08-16 Asia/Shanghai

## Current Decision

The fast-copy implementation is generic ROCr H2D/D2H behavior. It does not
contain a vLLM, SGLang, Qwen, tensor-name, operator, or shape branch. The
legacy path remains the default and is the required fallback when either gate
is disabled or an allocation/copy semantic is unsupported.

There is no independently passing full-model vLLM TP1 baseline in the source
tree or preserved evidence. Therefore this branch does not start a 0.8B
full-model A/B run, does not claim a vLLM weight-load speedup, and does not
modify vLLM or SGLang inference code.

## vLLM Evidence Boundary

Pinned vLLM checkout:

```text
HEAD  8d9b52f7c2514490bdadfd5eb0c931e58625df2e
```

The following artifacts are useful diagnostics but are not a TP1 acceptance:

| Artifact | Result | Boundary |
| --- | --- | --- |
| `/tmp/qwen35-vllm-weight-probe.stdout` | 248 tensors registered/loaded | weight registration only |
| `/tmp/qwen35-vllm-model-forward-fp32.stdout` | constrained single-token output correct, fallback 0 | excludes scheduler, prefill, multi-token, logits/sampling, CCL, TP |
| `/tmp/qwen35-vllm-model-forward-v2.stdout` | `output_correct=false`, exit 1 | not accepted |
| `/tmp/qwen35-vllm-prefill-full24-golden.stdout` | `output_correct=false`, exit 1 | not accepted |
| `/tmp/qwen35-vllm-decode-window-full24-v2.stdout` | execution completed but output comparison false | not accepted |

`scripts/run_qwen35_vllm_tp.py` currently accepts TP2/TP4, not TP1. The
`examples/triton/qwen35_vllm_model_forward.py --tensor-parallel-size 1`
runner is a constrained model-forward probe and must not be promoted to full
Engine/server acceptance without scheduler, prefill, decode, sampling, and
teardown evidence.

## Fast-copy Activation

All commands below must run from the isolated feature worktree:

```bash
cd /home/zhaosiying/amdgpu-sim-fastcopy
```

Legacy fallback (default behavior):

```bash
source scripts/fastcopy_mode.sh legacy
```

Explicit fast mode:

```bash
source scripts/fastcopy_mode.sh fast
```

The two gates are independent and both must be `1` for the model-provider
fast path:

```text
HSA_ENABLE_DTIF_FAST_COPY=1
SAGR_HSAKMT_MODEL_FAST_COPY=1
```

Any unsupported pointer kind, private/VMM/imported allocation, invalid range,
dependency-bearing copy, gang copy, or profiling-sensitive copy falls back to
the legacy AQL path. To compare modes in a future framework run, execute the
same command in two fresh private process groups, changing only
`source scripts/fastcopy_mode.sh legacy|fast`.

## Bounded Runtime Probe

Build the generic 2 MiB HSA probe in the feature-local build tree:

```bash
cmake --build /home/zhaosiying/amdgpu-sim-fastcopy/build/fastcopy-runtime \
  --target self_amdgpu_runtime_upstream_rocr_fastcopy_probe --parallel 6
```

Run legacy mode:

```bash
python3 scripts/run_fastcopy_rocr_probe.py legacy \
  --output /tmp/fastcopy-vllm-gate-legacy \
  --worker build/fastcopy-runtime/tests/self_amdgpu_runtime_upstream_rocr_fastcopy_probe \
  --allow-idle-gem5
```

Run fast mode:

```bash
python3 scripts/run_fastcopy_rocr_probe.py fast \
  --output /tmp/fastcopy-vllm-gate-fast \
  --worker build/fastcopy-runtime/tests/self_amdgpu_runtime_upstream_rocr_fastcopy_probe \
  --allow-idle-gem5
```

The probe requires byte-exact H2D/D2H results. In fast mode a pure copy may
retire zero AQL copy packets; that is expected. A ready dependency can be
tested explicitly with `--worker-arg=--dependency`, but an unsatisfied
dependency is outside this branch because the current model bridge does not
wake a queue after a host-side signal transition.

The recorded 2 MiB results were:

| mode | copy result | retired copy dispatches | wall time interpretation |
| --- | --- | ---: | --- |
| legacy | byte exact | 2 | includes simulator startup |
| HSA-only | byte exact | 2 | provider gate disabled |
| fast | byte exact | 0 | includes idle-simulator grace; not throughput |
| fast + ready dependency | byte exact | 1 | dependency correctly stays on AQL |

These numbers prove the data path and dispatch selection, not Qwen weight-load
performance. The fast run's elapsed time includes a bounded idle gem5 cleanup
interval and must not be reported as a copy bandwidth measurement.

## Future vLLM TP1 A/B Gate

Only execute this section after an unchanged-upstream full vLLM TP1 baseline
has passed independently.

1. Freeze one vLLM command, checkpoint, model directory, input, seed, and
   feature-local runtime/gem5 identity.
2. Run the command once after `fastcopy_mode.sh legacy` and once after
   `fastcopy_mode.sh fast`, each with a unique private endpoint/output path.
3. Record the exact weight-load interval, provider fast-copy eligibility
   counters, AQL copy retirement count, and process cleanup status.
4. Compare loaded parameter summaries, first-forward/logits or token output,
   and final deterministic output. Any legacy/fast difference is a failure.
5. Only then report weight-load speedup. If the baseline does not pass, stop at
   the bounded runtime probe and leave vLLM/SGLang unchanged.

The future TP16 path must reuse this same two-mode contract per rank. It must
not add a TP-size-specific copy branch; topology/CCL readiness is a separate
checkpoint from fast-copy correctness.

## Zero-context Resume

Start with the feature checkpoint, then this document:

```bash
cd /home/zhaosiying/amdgpu-sim-fastcopy
sed -n '1,260p' FASTCOPY_CHECKPOINT.md
sed -n '1,260p' docs/fastcopy-vllm-integration.md
git status --short --branch
git -C projects/rocm-systems status --short --branch
git -C projects/self-amdgpu-runtime status --short --branch
git -C projects/gem5 status --short --branch
```

Do not point commands at `/home/zhaosiying/amdgpu-sim` build products or send
signals to its live processes. Resolve the current feature commit with
`git rev-parse HEAD`; nested commit IDs are recorded in
`FASTCOPY_CHECKPOINT.md`.
