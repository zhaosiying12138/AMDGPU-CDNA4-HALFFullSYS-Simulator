# Fast-copy engineering lessons

Updated: 2026-08-16 Asia/Shanghai

## 1. A runtime flag is not a correctness proof

ROCr's existing DTIF fast-copy flag selects raw `memcpy`. That is valid only
when a GPU virtual address has a host-readable/writable mapping. The current
ordinary model GPU heap is PUBLIC and eligible, but internal private VRAM still
exists. A process-wide flag therefore needs an operation-level eligibility
gate rather than an assumption that every GPU allocation is CPU accessible.

## 2. Use the allocation registry, not model knowledge

The general fast path is an address-range operation. ROCr pointer information
already reports allocation base, size, type, owner, and HostAccess. Those facts
are sufficient to authorize an ordinary H2D or D2H copy; tensor names, weight
layout, kernels, and model architecture are irrelevant.

## 3. Bypass execution, not ownership

The copy may bypass AQL execution, but it must not bypass allocation lifetime,
mapping state, range checks, or completion ordering. The runtime's pointer and
queue semantics remain authoritative even when the data plane is host memory
copy on a shared mapping.

## 4. Protected mappings should stay protected

Changing private VRAM mappings to RW would broaden CPU-observable semantics for
internal allocations. It is unnecessary: ordinary weight buffers already use
the PUBLIC shared mapping. Keep private mappings private and fall back to AQL.

## 5. Prefer an existing upstream data path over a new bridge protocol

The shared backing already provides one coherent byte store for ROCr and gem5.
Adding a second copy RPC would duplicate range, lifetime, and ordering rules.
The smaller general solution is to harden the existing ROCr/CLR fast path with
explicit provider authorization and per-operation eligibility.

The product should still activate in legacy mode. Fast mode is a runtime
experiment selected after activation, so one binary can run an A/B comparison
and a faulty fast path never becomes the implicit fallback behavior.

## 6. Performance validation should expose the mechanism

The useful comparison is not a broad model matrix. Run the same copy with the
two gates disabled and enabled, compare wall time, and verify that the fast run
updates the same backing while retiring no AQL copy dispatch. This directly
tests both speed and the intended architecture.

## 7. A faster load is invalid if it changes model state

The backing path must preserve the bytes and ordering observed by later GPU
execution. Validate byte-exact round trips first, then require deterministic
model inference to match legacy for the same weights, input, seed, and decode
settings. Timing alone cannot accept this feature.

## 8. Model A/B needs an independently passing baseline

A fast-copy branch cannot use a pre-existing framework inference failure as a
reason to expand into model bring-up. Reuse vLLM TP1 only when it is already
known-good. Otherwise stop at focused runtime API correctness and path evidence
and state that model-level validation was skipped; do not infer correctness
from loading progress alone.

The current repository has accepted vLLM communicator and RowParallel layer
evidence, but its authoritative goal explicitly does not claim full-model vLLM
TP acceptance. Those narrower artifacts are not a substitute for a passing
TP1 inference baseline.

## 13. Preserve the vLLM stop condition

The 2026-08-16 vLLM preflight found no complete TP1 acceptance. Weight
registration and constrained single-token probes are useful for diagnosis, but
they do not prove scheduler, prefill, multi-token decode, sampling, CCL, or
teardown correctness. Full BF16/prefill/decode records include output mismatch.
The correct fast-copy action is to persist the evidence and stop, not to edit
vLLM or spend another long Qwen load on an invalid baseline.

## 14. Make the launch contract resumable

The two explicit gates and the feature-local runner are now documented in
`docs/fastcopy-vllm-integration.md`. A future model run must change only
`source scripts/fastcopy_mode.sh legacy|fast`, use a private endpoint and
output directory, and compare byte/path/dispatch evidence before reporting a
weight-load speedup. This keeps legacy fallback available while allowing a
later SGLang, vLLM TP1, or TP16 run to reuse the same generic contract.


## 9. ROCr image helpers are hidden focused-build prerequisites

Even a three-object ROCr build can trigger the target's image-blit generation
order dependency. In this product, set `HIP_DEVICE_LIB_PATH` to the locked
ROCm 7.2.3 clang bitcode directory. This is a build-input requirement, not a
reason to use or modify the original worktree's build directory.

## 10. A three-mode A/B catches the authorization regression

The small unchanged-upstream HSA worker is a useful stop condition for this
scope: `legacy` and `hsa-only` both retire two copy dispatches, while `fast`
retires only the user kernel. All modes return the same byte-checked result
for its 256-byte payload (64 int32 elements).
This directly tests that the second provider gate is an authorization gate,
not a performance-only hint. It does not measure large weight throughput; that
claim remains intentionally unverified without a passing model baseline.

## 11. Separate pure-copy completion from simulator shutdown

A fast copy can legitimately retire zero AQL packets. A runner that waits for a
normal dispatch-session completion will misclassify this as a gem5 hang. Keep an
explicit idle-simulator mode with a private, bounded termination, and require
the worker's byte result plus a clean gem5 log before calling the probe passed.

## 12. Dependency fallback must remain visible

The 2 MiB probe's ready non-empty dependency retires one legacy copy in fast
mode, while the no-dependency case retires zero copy packets. This is the
generic semantic boundary: dependencies are not silently ignored. The current
model bridge did not wake a queue after an unsatisfied dependency was changed
to zero, so that wakeup path remains an unverified bridge capability rather than
a reason to weaken the fast-copy gate.
