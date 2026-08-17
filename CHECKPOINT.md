# Active Checkpoint: AgentENV + fastcopy upstream model bring-up

Updated: 2026-08-17 Asia/Shanghai

This is the blank-context entry point for the active GSIM-001 work. Read this
file first, then `GOAL.md`, `PLAN.md`, `ENGINEERING_CONSTRAINTS.md`,
`AGENTENV_CHECKPOINT.md`, and `FASTCOPY_CHECKPOINT.md`. The archived
CP-0030 pointer in `state/current.json` is historical and is not the active
execution authority.

## 2026-08-17 Resumed Authority

The resumed goal is not the legacy CP-0029 vecadd continuation. The current
authority is revision 16 of `GOAL.md` and `PLAN.md`:

- AgentENV integration and generic fast copy are now on `main` and are required
  foundations for subsequent model tests.
- The active Windows `.wslconfig` contains the corrected literal doubled path
  separators and points at the staged 6.18.40.1 kernel/modules. The restart is
  complete: the running release is
  `6.18.40.1-aenv-ublk-6.18.40.1`, and `/dev/ublk-control` plus `/dev/kvm` are
  present and usable by the current user.
- The modules VHD has an extra release-directory nesting. Current-session
  symlinks made the required modules and devices usable; a permanent image
  layout repair can wait for a future maintenance restart. Do not request or
  perform another restart while the user is away.
- Complete one AgentENV service and sandbox create/collect/stop cycle first.
  If it cannot work without another restart, do not block model bring-up:
  continue on the host, parallel only with proven disjoint resources and
  otherwise serially.
- Model tests default to `source scripts/fastcopy_mode.sh fast`. Legacy remains
  an explicit A/B/fallback, not the default long-run configuration.
- SGLang and vLLM must both be unchanged upstream. Project replacement Triton
  operators, copied model code, monkey patches, or engine edits do not count.
- A TP1 sandbox may own exactly one live gem5 process. SGLang and vLLM may run
  concurrently only in separate AgentENV instances and worktree branches.
  Shared fixes are reviewed against both engines and enter main serially.
- The active ladder is SGLang TP1 + vLLM TP1, then both TP2, then
  Qwen3.5-9B TP16. All Qwen3.5-0.8B TP4 tasks are cancelled.
- During a long model run, assign one read-only monitor and use
  `gpt-5.6-sol` agents at `max` reasoning only for bounded work likely to
  shorten the next iteration: failure capsules, focused replay/tests,
  postmortem tooling, launch-overhead measurement, or the next gate. Do not
  fill concurrency with low-value work, change the running binary, or count
  focused replay as model acceptance.

AgentENV host-only code and its focused tests have been integrated. The main
gem5 gitlink is `30d2511083d7c32ffa75ef1fd1864432c3212cf8`: it preserves the
later Claude-era fixes and the reviewed LGKM-underflow fail-closed invariant.
That inherited change is committed; it is no longer a dirty-worktree item.

## Non-Negotiable Engineering Rule

All production fixes must be generic at the shared ISA, ROCr/HIP, runtime,
bridge, memory, queue, or collective boundary. Do not add branches keyed by a
model, framework, operator, tensor shape, kernel hash, code address, PC, or
golden output. Use the unchanged upstream SGLang/vLLM model path to expose the
next weak shared boundary.

## Historical Model Position Before AgentENV

The active diagnostic is unchanged-upstream SGLang 0.5.17 running local
`Qwen3.5-0.8B`, BF16, TP1, AITER attention, Triton linear attention, disabled
CUDA graphs, and one generated token.

Completed shared-layer fixes include:

- ROCr resident-AQL admission, zero-kernarg handling, queue lifecycle, and
  managed runtime-to-gem5 dispatch.
- V_PACK_B32_F16 semantics and speculative instruction-buffer control-flow
  gating.
- Generic lazy scratch admission/retry, scratch pinning, signed gfx950 FLAT
  scratch offsets, and physical resident-wave scratch-slot mapping for the
  current wave64 full-small profile.
- The complete gfx950 CDNA4 `DS_READ_*_TR_*` B4/B6/B8/B16 family: local-memory
  classification, architecture-derived cross-lane packing, and fail-closed
  ACC/GDS/wave32/partial-EXEC/alignment contracts.
- The six gfx950 MFMA opcode overrides that overlap older VOP3P table slots:
  BF16/F16 16x16x32 and 32x32x16, plus I8 16x16x64 and 32x32x32. The live
  failure word `0xd3b70000` is `v_mfma_f32_32x32x16_bf16`.

The MFMA change reuses the existing parameterized execution classes and the
existing gfx950 timing keys. It does not specialize the model or kernel. It
does not claim ACC_CD=1/AGPR hazard support, scaled MFMA, or F8/F6/F4 MFMA.

## Verification At This Checkpoint

- `gpu_decoder.test.opt`: 11/11 passed, including all six gfx950 overlapping
  MFMA opcodes and the live `0xd3b70000` word.
- `ds_transpose.test.opt`: 6/6 passed.
- The previous DS-fixed real model run reached ticket 1115 and exposed the
  next `0xd3b70000` decoder gap, proving the DS resource/semantic boundary was
  crossed.
- Full `gem5.opt` linked successfully after the MFMA patch.
- Current workspace binary SHA-256:
  `34e94f9129cdf03a43a2be8026f57791c900bd1d427c2a0b0d1cd8d03104d98e`.
- Current self-runtime model DSO SHA-256:
  `ee5f71edbb80ca6c02b8126d12c96fc3d991c2f6327bfdcb1605f8c5b173afdf`.

## Last Real Run And Stop Boundary

The MFMA-fixed rerun used:

- App log: `/tmp/sglang-qwen35-tp1-mfma-v2-20260816.log`
- Scheduler run: `/tmp/self-amdgpu-opencl-run.1000.HAQikR`
- Scheduler trace: the final durable record is execution ticket 1006.
- It completed the 0.8B weight load in 1475.09 seconds, allocated Mamba cache,
  allocated the 16-token KV cache, and logged `Memory pool end`.
- There was no `fatal`, `panic`, `Invalid opcode`, `unmapped`, or Python
  traceback in the target logs.
- The run was deliberately terminated at the user's key-change pause request.
  It did not yet reach the old ticket-1115/next-dispatch MFMA boundary, engine
  ready, generation, output validation, or natural teardown.
- The exact Python process group and its three managed gem5 process groups
  were terminated and verified absent. No target workload is intentionally
  left running.

Therefore the immediate acceptance gate is unchanged: rerun the same TP1 path
and require it to pass ticket 1115, execute the MFMA dispatch, generate the
requested token, and cleanly tear down. Do not start topology/TP work before
that model boundary is known.

## Resume From Zero Context

1. Read the active contracts in this order and verify source state:

   ```bash
   cd /home/zhaosiying/amdgpu-sim
   sed -n '1,280p' CHECKPOINT.md
   sed -n '1,260p' GOAL.md
   sed -n '1,300p' PLAN.md
   sed -n '1,300p' ENGINEERING_CONSTRAINTS.md
   sed -n '1,260p' AGENTENV_CHECKPOINT.md
   sed -n '1,240p' FASTCOPY_CHECKPOINT.md
   git status --short --branch
   git -C projects/gem5 status --short --branch
   ```

2. Confirm `uname -r` is the custom 6.18.40.1 release and both required device
   nodes are usable. If a later AgentENV issue would require another restart,
   record it for the user's return and continue the model lane on the host.
   Never invoke `wsl --shutdown` from the agent.

3. Verify AgentENV prerequisites, then run only host/lifecycle checks before
   selecting AgentENV or the host fallback for a model:

   ```bash
   uname -r
   test -c /dev/ublk-control
   test -r /dev/kvm && test -w /dev/kvm
   python3 -m unittest \
     tests.test_agentenv_wslconfig \
     tests.test_agentenv_bundle \
     tests.test_agentenv_manager \
     tests.test_agentenv_service -q
   python3 tools/agentenv_service.py plan
   ```

4. Build/verify the immutable runtime bundle and one sandbox lifecycle. If its
   service ownership, collection, namespace, and host-process guards are green,
   use it. Otherwise record the failure and use host execution without waiting
   for interaction; parallelize only after proving isolation, else serialize.

5. Create independent SGLang/vLLM branches and AgentENV instances. Enable fast
   copy with `source scripts/fastcopy_mode.sh fast`, run byte-exact probes, and
   enforce exactly one live gem5 process per TP1 sandbox. On failure freeze the
   first shared-layer traceback and make one generic committed fix. Review its
   impact on the peer engine before serially integrating it into main.

6. Require both unchanged-upstream TP1 paths to generate and tear down, then
   both TP2 paths. Skip 0.8B TP4 and proceed directly to the 9B TP16 gate.

## Historical Host Resume Command (Do Not Execute As Current Plan)

1. Open the repository and read the active contracts:

   ```bash
   cd /home/zhaosiying/amdgpu-sim
   sed -n '1,240p' GOAL.md
   sed -n '1,260p' PLAN.md
   sed -n '1,260p' CHECKPOINT.md
   sed -n '1,240p' ENGINEERING_CONSTRAINTS.md
   scripts/resume.sh --verify
   ```

2. Confirm the committed trees are clean and no previous target is running:

   ```bash
   git status --short
   git -C projects/gem5 status --short
   git -C projects/self-amdgpu-runtime status --short
   pgrep -af 'qwen35_inference|gem5.opt.*host_dispatch.py' || true
   ```

3. Rebuild only if the binary is missing or older than the MFMA sources:

   ```bash
   GEM5_JOBS=24 scripts/build_gem5_lld24.sh \
     build/VEGA_X86/arch/amdgpu/vega/gpu_decoder.test.opt \
     build/VEGA_X86/gem5.opt
   projects/gem5/build/VEGA_X86/arch/amdgpu/vega/gpu_decoder.test.opt \
     --gtest_color=no
   ```

4. Launch the exact TP1 diagnostic. The activation script owns the valid
   `HSA_MODEL_TOPOLOGY`; do not override it with the nonexistent
   `build/.../tests/topology` path.

   ```bash
   cd /home/zhaosiying/amdgpu-sim
   source env/conda/rocm-pytorch-v3-fa8414cce688f934f538163621423376c2542acff3e4d3e403df4340d90fcd6d/etc/conda/activate.d/amdgpu-sim-rocm-pytorch.sh
   export PYTHONPATH=/home/zhaosiying/amdgpu-sim/projects/sglang-0.5.17:/home/zhaosiying/amdgpu-sim/env/sglang-overlay-cp312
   export LD_LIBRARY_PATH=/home/zhaosiying/amdgpu-sim/projects/self-amdgpu-runtime/build/cp28-runtime-clang:${LD_LIBRARY_PATH:-}
   export HSA_MODEL_LIB=/home/zhaosiying/amdgpu-sim/projects/self-amdgpu-runtime/build/cp28-runtime-clang/libself_amdgpu_hsakmt_model.so.1
   export SAGR_MANAGED_GEM5=/home/zhaosiying/amdgpu-sim/projects/gem5/build/VEGA_X86/gem5.opt
   export SAGR_MANAGED_GEM5_CONFIG=/home/zhaosiying/amdgpu-sim/projects/gem5/configs/example/gemsim/host_dispatch.py
   export SAGR_MANAGED_REPO_ROOT=/home/zhaosiying/amdgpu-sim
   export HSA_ENABLE_DTIF_FAST_COPY=0
   export SGLANG_USE_AITER=1
   export TRITON_DEFAULT_BACKEND=gemsim_hip
   export TRITON_BACKENDS_IN_TREE=0
   export GEMSIM_HIP_AUTOTUNE_MODE=correctness
   export TRITON_CACHE_AUTOTUNING=1
   export FLA_CACHE_RESULTS=1
   unset SAGR_OPENCL_ENDPOINT SAGR_OPENCL_SOCKET SAGR_OPENCL_GEM5_EXTERNAL
   log=/tmp/sglang-qwen35-tp1-mfma-resume-$(date +%Y%m%dT%H%M%S).log
   python examples/sglang/qwen35_inference.py \
     --tp-size 1 \
     --attention-backend aiter \
     --context-length 16 \
     --max-total-tokens 16 \
     --max-mamba-cache-size 5 \
     --max-new-tokens 1 >"$log" 2>&1
   ```

5. Monitor the scheduler run with the largest growing
   `dispatch-trace.jsonl`. On failure, freeze the exact next AQL packet,
   descriptor, code bytes, dynamic PC/state, and first traceback before
   editing. Repair the lowest shared layer and rerun. On success, record the
   output token and clean session teardown.

6. After TP1 succeeds, implement one coherent product `gpu_count` setting
   (1..16, default 16) through topology generation, self-runtime/provider,
   managed-session arguments, and per-GPU KMT bridge state. Then run TP2 and
   TP4; do not fake devices in PyTorch, SGLang, or vLLM. SMI remains a separate
   16-slot lease-backed ON/OFF observation facade.

## Remaining High-Level Work

1. Activate and validate AgentENV after the user-controlled WSL restart.
2. Complete unchanged-upstream SGLang and vLLM Qwen3.5-0.8B TP1 with fast copy,
   one gem5 process per sandbox, generation, equality checks, and teardown.
3. Generalize logical GPU capacity to 1..16 and validate both engines at TP2.
4. Complete lease-backed 16-slot `rocm-smi`/`gemsim-smi` product behavior.
5. Run upstream AMD Qwen3.5-9B TP16 with `torch.compile` after both TP2 paths
   are stable.
