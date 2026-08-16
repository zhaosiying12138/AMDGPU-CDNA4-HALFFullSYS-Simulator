# Generic model-backed memory copy fast path

Updated: 2026-08-16 Asia/Shanghai

## Active Goal

Implement a generic H2D/D2H memory-copy fast path for the unchanged ROCr
runtime. When explicitly enabled, eligible copies must update the same
model-owned shared backing that gem5 uses, without generating an AQL copy
kernel or SDMA packet. The legacy AQL path remains available and is the
default.

The target is to reduce model weight-loading time from simulated copy-kernel
time toward host storage-to-memory copy time. The implementation must apply to
normal runtime copy APIs and must not inspect a framework, model, operator,
tensor name, tensor shape, kernel identity, or program counter.

## Paused Goal

The model/TP work in `GOAL.md`, `PLAN.md`, and `CHECKPOINT.md` is paused while
this feature is developed. This branch must not merge into the main branch
until the user explicitly requests it.

## Required Behavior

1. Legacy is the default: `HSA_ENABLE_DTIF_FAST_COPY=0` uses the existing AQL
   copy path.
2. Fast copy requires both:
   - `HSA_ENABLE_DTIF_FAST_COPY=1`
   - `SAGR_HSAKMT_MODEL_FAST_COPY=1`
   Use `source scripts/fastcopy_mode.sh fast` to set both. Use the same command
   with `legacy` to restore the complete legacy path.
3. An old model provider, disabled provider option, dependency signal,
   profiling requirement, unsupported pointer kind, unmapped allocation, or
   unsupported range must fall back to the legacy path without raw host access
   to a `PROT_NONE` GPU mapping.
4. A request that starts in a known allocation but crosses its boundary is an
   error, not a fallback.
5. Fast-path completion is published with release ordering only after the
   backing update succeeds.
6. No gem5 command is expected for a fast copy. The absence of an AQL copy
   dispatch is part of the intended data path, not a simulator fallback.

## Correctness Boundary

The model provider owns the authoritative GPU-VA-to-backing-offset mapping.
The generated ordinary GPU heap is `FRAME_BUFFER_PUBLIC`, so ROCr creates
eligible weight allocations with `HostAccess=1` and libhsakmt maps the model's
shared `memfd` at the GPU VA with `MAP_SHARED | MAP_FIXED` and read/write host
access. A small number of internal private allocations remain non-host-
accessible and must not take this path. USERPTR, imported memory, sparse/VMM
mappings, MMIO, doorbells, and cross-allocation requests remain on the legacy
path until their semantics are explicitly supported.

The implementation must not walk or bypass a guest page table. It may bypass
packet simulation only after the runtime proves that the entire byte range is
host-accessible and belongs to the supported ordinary allocation class.

## Final Acceptance And Stop Condition

The feature is not complete when the API or microbenchmarks pass if an existing
model baseline is available. First inspect the repository evidence for a
previously passing vLLM TP1 Qwen3.5-0.8B path. Do not use this feature branch to
repair framework/model inference:

- If vLLM TP1 already passes, use that unchanged command for the final model
  A/B below.
- If vLLM TP1 has not passed independently, record that baseline blocker and
  skip model end-to-end validation. Do not debug vLLM or SGLang as part of this
  feature. In that case the bounded acceptance ends at byte-exact runtime API
  A/B, copy-path evidence, and isolated build/tests; model-level correctness
  remains explicitly unverified rather than inferred.

When the passing vLLM baseline exists, final acceptance is a real framework
loading the full local Qwen3.5-0.8B checkpoint through the unchanged runtime
API:

1. Run the same framework command, model files, runtime build, and host under
   the legacy and fast-copy configurations.
2. Both runs must finish the complete weight-loading phase and complete the
   same deterministic inference without changing model or framework source.
3. The fast run must use provider-backed H2D copies and must not retire AQL
   copy kernels for those handled transfers.
4. Validate loaded-memory correctness before claiming speedup: small copies
   must pass byte-exact round trips, and the model runs must use the same input,
   seed, decoding settings, and weights. Compare deterministic parameter
   summaries where available, first-forward output or logits, selected token
   IDs, and final generated output. Any difference from legacy is a failure.
5. Record exact weight-load wall time and speedup. The target is close to host
   file-read plus memory-copy time rather than the former 25-30 minute
   simulated-copy-kernel time.

Do not claim model acceptance without this A/B weight-load gate. Absence of a
passing vLLM TP1 baseline is an explicit reason to skip, not permission to fix
or change the framework in this branch.

## Incremental Validation Scope

Keep validation narrow and failure-oriented:

1. Small H2D and D2H round trips at KB and MB scale.
2. Boundary, disabled-option, unsupported-address, and dependency fallback.
3. Compare fast and legacy wall time and AQL retirement count.
4. Check existing vLLM TP1 evidence without starting a long model run.
5. Only if vLLM TP1 is already known-good, run the full 0.8B legacy/fast
   weight-load and inference A/B after the small gates pass.
