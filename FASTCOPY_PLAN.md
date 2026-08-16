# Fast-copy implementation plan

Updated: 2026-08-16 Asia/Shanghai

## Development Isolation

- Root worktree: `/home/zhaosiying/amdgpu-sim-fastcopy`
- Root branch: `feature/fast-copy-memfd`
- ROCm Systems branch: `feature/fast-copy-memfd-rocm`
- Self-runtime branch: `feature/fast-copy-memfd-runtime`
- gem5 branch: `feature/fast-copy-memfd-gem5`
- All builds and temporary outputs must remain under this worktree or a unique
  `/tmp` directory.
- Do not signal, reuse, or modify processes and build products from
  `/home/zhaosiying/amdgpu-sim`.

## Implementation Steps

1. Require two explicit gates in model mode: the existing ROCr/CLR operation
   switch plus a model-provider authorization switch. Both default off.
2. Reuse the existing upstream CLR direct H2D/D2H path for ordinary public
   buffers. Do not add a KMT copy opcode or a runtime-gem5 bridge data path.
3. Route ROCr kernel-blit and SDMA-blit fast-copy attempts through one generic
   eligibility helper. Only host-to-device or device-to-host ranges with one
   ordinary host-accessible GPU allocation may use raw memory copy; D2D,
   private, VMM, import, IPC, graphics, and invalid ranges fall back or fail
   closed.
4. Preserve asynchronous dependency, gang, and profiling semantics by falling
   back whenever the direct completion contract is insufficient. Publish
   successful async completion with release ordering.
5. Build ROCr/libhsakmt, self-runtime, and any required gem5 artifacts in
   independent build directories. Do not install over an active product.
6. Run focused small-copy gates and an unchanged-upstream HSA runtime worker
   smoke; do not call that a framework/model acceptance.
7. Inspect existing evidence for a passing unchanged vLLM TP1 baseline. If it
   exists, run the full Qwen3.5-0.8B legacy/fast weight-load and inference A/B.
   If it does not exist, skip model E2E and do not repair vLLM/SGLang in this
   branch; close with runtime byte correctness, path, fallback, and timing
   evidence while marking model-level validation unverified.

## Commit Policy

Commit every coherent, buildable stage immediately:

1. Goal/plan/checkpoint/lessons contract.
2. Upstream ROCr/CLR gates and generic eligibility routing.
3. Product activation option and focused tests.
4. Isolated build and runtime verification.
5. Final checkpoint with exact resume instructions.

At each stage update `FASTCOPY_CHECKPOINT.md` and `FASTCOPY_LESSONS.md`. Do not
leave unstaged or untracked source files at a handoff boundary.

## Status

- [x] Isolated root and nested submodule worktrees created.
- [x] Existing ROCr raw-memcpy path and model-backed mapping audited.
- [x] Existing public shared mapping confirmed for ordinary weight buffers.
- [x] Generic upstream fast-copy hardening design selected.
- [x] ROCr/CLR model-mode two-gate initialization implemented.
- [x] ROCr operation eligibility and ordering fallback implemented.
- [x] Product activation option implemented; both gates default to legacy off.
- [x] Focused builds and tests passed: ROCr full library and isolated
      self-runtime/model/worker targets build; the three-mode byte/path probe
      passes.
- [x] Small runtime benchmark passed: fast removes the two copy dispatches and
      preserves the worker's byte-exact result; hsa-only falls back to legacy.
- [x] 2 MiB sync H2D/D2H probe passed in legacy, authorization-only, and fast
      modes; a ready non-empty dependency remains on the AQL path.
- [x] Existing vLLM TP1 baseline status established: no full-model passing TP1
      is claimed, so model A/B is skipped and framework repair is out of scope.
- [x] Feature branches clean and checkpointed.
