# AMDGPU simulator engineering constraints

These constraints are mandatory for implementation and validation work in this
repository. They describe the product execution architecture, not merely a
checkpoint-specific test procedure.

## Operator execution

- Running a Triton operator must compile only that operator when its persistent
  Triton cache does not already contain a compatible artifact.
- Running or adding an operator must not configure, compile, relink, or install
  gem5 or `self-amdgpu-runtime`.
- A repository Triton example must be directly executable as
  `python3 examples/triton/<example>.py`. The entry point may transparently
  select the private Python, `gemsim_amd`, the installed runtime library, and a
  persistent Triton cache, but it must not inject gem5/runtime build settings.
- `ccache`, host `CC`/`CXX`, host linker selection, and `MAX_JOBS` are build
  policy, not operator runtime configuration. Triton JIT reuse is provided by
  `TRITON_CACHE_DIR`, not by rebuilding gem5 or the runtime.
- Every production fix is operator-agnostic. Runtime, bridge, compiler backend,
  and simulator admission may depend only on versioned ABI, target,
  code-object metadata, resources, ownership, bounds, synchronization, and ISA
  semantics. They never branch on an operator/model name, tensor shape/role,
  image hash, expected output, or predeclared PC sequence.
- A new-operator failure is repaired once at the lowest shared semantic layer
  that is wrong: generic lowering, code-object/kernarg handling, memory,
  dispatch/synchronization, or ISA execution. Per-operator production routes
  and callbacks are forbidden.
- Operator-specific material belongs only in ordinary user/kernel source,
  external oracle/evidence, and focused tests. It cannot become production
  admission or simulator-side numerical implementation.

## Runtime and simulator boundary

- `self-amdgpu-runtime` is a prebuilt shared library with a stable generic ABI.
  Python/Triton loads and links this installed library; it is not rebuilt per
  operator.
- gem5 is a separately prebuilt simulator process. The runtime communicates
  with it through the versioned Unix-socket protocol. Operator submission must
  remain process- and build-decoupled from gem5.
- The production path is kernel-agnostic: it transports compiler-produced
  HSACO and metadata, kernarg bytes, arbitrary owned allocations, grid,
  workgroup and shared-memory geometry, then returns explicitly requested
  allocations through D2H.
- New kernels are admitted by generic format, ownership, bounds, ABI and
  resource checks. Kernel names, code-object hashes, tensor shapes, per-op
  buffer roles, fixed PC sequences, and operator oracles must not be production
  admission requirements.
- Code-object hashes and exact traces belong to regression/evidence records.
  Numerical oracles belong in the Python caller. Unsupported ISA instructions
  fail with an opcode/PC diagnostic and are fixed once in the shared ISA model.

## Private Python environment

- Simulator Python packages are installed only in the repository-local private
  ROCm/Triton virtual environment. Do not modify the host NVIDIA/CUDA Python
  environment or inherit its packages into operator execution.
- Pin required package versions, wheel identities, URLs, and SHA-256 digests in
  the repository environment setup. A direct operator `.py` entry point must
  transparently re-exec this reproducible private environment.
- Run independent wheel downloads and private-environment maintenance in the
  background when practical. Continue foreground E2E implementation, source
  audit, and tests that do not depend on the pending package instead of waiting
  idle for downloads.

## External golden oracle

- A local NVIDIA GPU may generate golden outputs in a separate CUDA virtual
  environment and process when a CPU oracle would delay model bring-up. This is
  reference computation only; it is never target execution or fallback.
- The `gemsim_amd` target process must still report zero CPU/NVIDIA fallback and
  must not load CUDA/NVIDIA DSOs or open NVIDIA device nodes. Golden and target
  artifacts communicate only through retained inputs, outputs, hashes, and
  comparison reports.
- Bind every golden result to the exact model/config/weight/input hashes,
  tensor metadata, oracle implementation, PyTorch/CUDA versions, GPU identity,
  and output hashes. Do not reuse an uncorrelated golden artifact.
- Numerical BF16/FP32 tensors use finite checks plus source-grounded elementwise
  tolerances and relative-L2 gates. Token IDs, shapes, strides, aliasing,
  mutation, cache/state selection, guards, and discrete outputs remain exact or
  bitwise as required by their contract.

## Unchanged upstream inference-engine integration

- Keep pinned SGLang and vLLM source byte-identical to upstream. The accepted
  model path may use only their normal ROCm/HIP/PyTorch/Triton interfaces plus
  the generic self-runtime facade below those interfaces.
- Project-owned replacement Triton operators, `torch.library` model-operator
  registrations, copied framework/model code, and monkey patches are
  diagnostic history only. They cannot satisfy SGLang or vLLM TP1/TP2/model
  acceptance and must not be active in an accepted target process.
- A temporary diagnostic patch must still be idempotent, reversible, limited to
  named symbols, and fail closed unless the pinned module version, source
  identity, callable signature, model config, dtype, shape, alias, mutation,
  and state contract match. Never apply a broad process-global patch to an
  unknown torch/vLLM checkout.
- Runtime activation must be explicit and observable in the private
  environment, preserve legacy behavior when disabled, and expose fallback
  counters. It must not import CUDA/NVIDIA libraries into the AMD target
  process or turn a reference oracle into target execution.
- Preserve vLLM's standard model, parameter-loader, distributed-initialization,
  named `GroupCoordinator`, and tensor-parallel layer contracts. Project code
  must not assign private parallel-state globals or build a second sharding
  framework. An OOT adapter supplies only the delegated device implementation
  and validates exact compatibility with the pinned upstream contract.
- Gloo may be used for bounded, audited process-group bootstrap and control
  metadata when required by upstream initialization. Model tensor payload and
  arithmetic must reach the OOT device communicator and never fall back to
  Gloo. Evidence distinguishes these two traffic classes.

## AgentENV isolation and serial integration

- Model tests run in AgentENV once its custom-kernel, ublk, KVM, Firecracker,
  service-ownership, and lifecycle gates pass. A configured `.wslconfig` alone
  is not acceptance, and the assistant never invokes `wsl --shutdown` without
  first warning the user and receiving explicit confirmation.
- Concurrent SGLang/vLLM work uses separate AgentENV instances, worktrees,
  branches, builds, caches, sockets, endpoints, tmp directories, SMI lease
  namespaces, logs, and process groups. A sandbox must not mutate another
  sandbox or the host main worktree.
- TP1 has a hard budget of one live gem5 process per sandbox. Preflight refuses
  launch if the target command would create or reuse another gem5 process.
- Parallel lanes discover bugs independently but never commit to main in
  parallel. Each shared-layer fix receives focused tests and an explicit impact
  review for both engines, then enters main serially; the other lane rebases
  before continuing.
- Main is the integration authority. Every coherent feature/fix is committed
  promptly, and no handoff leaves staged, unstaged, or untracked source files.

## Fast-copy model policy

- Model bring-up defaults to the generic two-gate fast-copy mode selected by
  `source scripts/fastcopy_mode.sh fast`. It may bypass an AQL copy only for a
  fully eligible host-accessible allocation and range; dependency-bearing,
  profiling-sensitive, private, imported, VMM, D2D, invalid, or unsupported
  copies retain the specified fallback/error behavior.
- Legacy mode remains available through `source scripts/fastcopy_mode.sh
  legacy`. Use it for one bounded A/B or diagnosis, not as the default long-run
  path.
- Fast-copy acceptance requires byte-exact small/MB probes and model-level
  equality against legacy for loaded parameters, deterministic outputs/logits,
  token IDs, and cleanup before reporting weight-load speedup.

## Active model ladder

- First pass unchanged-upstream SGLang TP1 and vLLM TP1, then both engines at
  TP2. Qwen3.5-0.8B TP4 is cancelled and must not be scheduled.
- After both 0.8B TP2 gates pass, continue directly to Qwen3.5-9B TP16 on the
  upstream AMD execution path. No custom engine operator/model implementation
  can substitute for these gates.

## Upstream-zero-diff compiler integration

- Keep pinned Triton core and AMD lowering unchanged. In particular, do not
  edit Triton's core runtime driver, compiler coordinator, language frontend,
  or generic JIT/cache logic to recognize GemSim, Qwen, or a specific kernel.
- Implement the normal backend contracts in the project-owned out-of-tree
  `gemsim_amd` driver/compiler/launcher and reuse upstream AMD lowering and
  code-object generation. Backend selection, capabilities, cache identity, and
  runtime calls belong to that plugin.
- Framework and Triton users do not construct runtime or gem5 transport
  records. The backend plugin hides the small versioned self-runtime ABI, and
  the self-runtime hides process/session, allocation, module, dispatch, wait,
  copy, and cleanup details.
- Do not equate this transparent path with complete ROCm ABI compatibility.
  HIP/OpenCL/ROCr shims are separate compatibility lanes and are added only for
  a named caller with a frozen ABI and standalone tests.

The complete ownership table and TP integration rule are defined in
`docs/framework-runtime-layering.md`.

## Runtime-gem5 bridge concentration

- The self-runtime public API and the gem5 GPU execution adapter are stable
  facades. All KMD-removal-specific transport, identity, ownership,
  generation, process, deadline, error, and completion translation belongs in
  one runtime-gem5 bridge layer.
- vLLM, PyTorch, and Triton must not see bridge records or gem5 implementation
  objects. gem5's queue, memory, dispatcher, CU, and Vega implementation must
  not see self-runtime message types, framework ranks, tensor roles, model
  names, or collective names.
- Prefer one generated/shared canonical bridge contract over independently
  maintained runtime and gem5 codecs. Both endpoints validate the same fixed
  widths, endian order, reserved fields, capability/version rules, and status
  mapping.
- Collective device arithmetic is submitted as an ordinary generic kernel.
  Do not add a CCL-, vLLM-, or Qwen-specific operation to gem5 admission.
- A bridge path is accepted only after two structurally different retained
  kernels and one previously unseen valid kernel use it without any production
  source change. Fixture-only success cannot establish generality.
- Migrate beside the accepted baseline and switch only after equivalent
  output/trace/fault/cleanup gates pass. Once the common bridge fully replaces
  a fixture route, duplicate codec, copied upstream state machine, or obsolete
  launcher, remove that legacy implementation and its dead build references.

## Build and storage policy

- All project builds use 24 parallel jobs.
- LLVM, Triton, gem5 and runtime maintenance builds use Clang and lld with a
  persistent `ccache` where the build system invokes a C/C++ compiler.
- Reuse compatible completed build directories and caches. Do not create a
  fresh build merely to add or execute an operator.
- Remove only build artifacts proven obsolete and not required by accepted or
  retained non-passing evidence. Do not consume disk with redundant builds.

## End-to-end priority and optimization gate

- The critical path is Qwen3.5-0.8B single-device end-to-end inference first,
  followed by functional CCL and tensor-parallel execution. Simulator
  optimization is subordinate to those milestones.
- Optimize gem5 only when a real operator, layer, model, or TP run identifies a
  measured dominant cost and the proposed change has high expected impact,
  small implementation scope, and strong correctness regressions.
- Stop or defer an optimization when profiling or implementation takes longer
  than advancing the next model blocker, the measured gain is not material, or
  the change introduces correctness/lifecycle risk or follow-on repair work.
- ISA and runtime fixes required for a currently blocked model kernel are
  correctness work, not an invitation to broaden unsupported semantics. Do not
  pursue completeness that is not exercised by the model critical path.
- Prefer functional simulation fast paths only when they preserve observable
  kernel, memory, synchronization, collective, and failure semantics needed by
  single-device and TP acceptance. Every performance change requires retained
  before/after evidence and unchanged operator/layer or model correctness.

## CCL before tensor parallelism

- Stabilize the project CCL API and collective semantics independently before
  connecting vLLM tensor parallelism. A vLLM TP run is not the first test of a
  new collective implementation.
- The CCL API is one bounded, generic `world_size=2..16` contract. Planner,
  codec, state, and carrier tests cover every integer world size in that range,
  including non-power-of-two N=3/5/7/15; they must not contain a TP=2-only or
  power-of-two-only branch.
- Multi-process integration tests cover rank registration, topology, group and
  sequence ordering, dtype/shape/count validation, allreduce, allgather,
  reduce-scatter, broadcast, barriers, zero-size and uneven chunks, timeouts,
  peer failure, stale epochs, retry/reconnect, and deterministic cleanup. Live
  acceptance must include N=2, N=3, N=4, N=8, and N=16. Unit or mock rank-16
  coverage does not count as live rank-16 collective execution.
- Protocol capacity and model sharding legality are separate. Supporting 16
  ranks in CCL never implies that a particular Qwen model shape is valid at
  TP=16; framework admission must independently prove every tensor/head/vocab
  dimension and shard layout for the selected TP degree.
- Tensor reductions execute through gem5 kernels; host transport may move bytes
  but cannot silently compute a collective result. Every test reports host
  reduction and fallback counters.
- CCL v1 SUM has an explicit ordered numeric contract. For every nonzero
  REDUCE_SCATTER receive, BF16 loads both BF16 partials as FP32, performs one
  binary FP32 add, then immediately stores BF16 with round-to-nearest-even
  before that chunk can be sent at the next hop. FP32 performs one binary FP32
  add per hop. A cross-rank FP32 accumulator is a different future protocol
  mode and must never be introduced silently.
- Carrier receive copies into immutable staging disjoint from the private
  workspace. The device SUM must complete successfully before the receiver
  sends `CONSUMED` or uses the chunk in the next planner step. A zero-element
  receive is acknowledged without dispatch. Validation, copy, launch, or wait
  failure atomically aborts the group epoch and never returns credit as if the
  step succeeded.
- Host or NVIDIA arithmetic is allowed only as an external post-target oracle
  with `feedback_to_target=false`; it is separately counted and cannot satisfy
  `host_reduction_count=0`. A host-mock collective accepts only transport,
  ordering, state, fault, and cleanup behavior, never device math.
- The private reduction primitive is in-place on a disposable workspace. The
  vLLM `world_size>1` all-reduce adapter must preserve the public input bytes
  and return a fresh nonalias output only after the entire collective commits.
  Nonuniform `reduce_scatterv` remains explicitly unsupported until it has its
  own planner/API and tests; uniform reduce-scatter must not masquerade as it.
- Only after the standalone CCL suite is stable may the formal PyTorch
  distributed/vLLM communicator plugin map TP collectives onto that API. TP
  tests then focus on framework ordering, model sharding, and end-to-end output
  rather than rediscovering basic collective bugs.

## TP and simulator inventory scope

- Formal Qwen model acceptance covers TP=2 and TP=4 only. Do not create TP=8
  or TP=16 model matrices merely because the reusable CCL/runtime capacity is
  16 ranks.
- Keep all protocol, planner, rendezvous, lifecycle, and device-slot code
  parameterized for 2..16. The narrower model matrix must never introduce a
  TP=2/4 special case below the framework admission and evidence layer.
- The local simulator inventory exposes exactly 16 logical slots. It does not
  eagerly launch 16 gem5 processes: a slot is `ON` only while a managed
  runtime owns a live, identity-validated gem5 lease; all other slots are
  `OFF`.
- `rocm-smi` is an observation facade, not a hardware-management emulation. It
  never opens GPU device nodes, loads production SMI libraries, or fabricates
  clocks, power, temperature, utilization, or health values.

## Framework portability scope

- The formal framework targets are vLLM Qwen3.5-0.8B TP=4 and SGLang
  Qwen3.5-0.8B TP=4. Pinned upstream framework trees remain unmodified, and
  both targets must use the same runtime-gem5 bridge and generic CCL engine.
- vLLM Qwen3.5-9B TP=16 is a separate scale target with upstream
  `torch.compile`; it does not create a TP=8 model matrix.
- On ROCm, framework attention/GEMM selection stays with the upstream selector:
  vLLM AITER/ROCM_ATTN/TRITON_ATTN and SGLang AITER/Triton as compatible. Never
  hard-code FlashInfer or add a project-owned attention replacement for AMD.
- If a framework lacks an official OOT device/communicator/backend hook, first
  record the exact extension gap. A repair may add one generic adapter at the
  shared boundary, but may not copy model layers, patch individual operators,
  or modify the runtime-gem5 bridge per framework.

## Correction policy

- Do not preserve a known-wrong architecture just to close a checkpoint.
  Correct it immediately, rerun the relevant generic regressions, and update
  the active checkpoint to describe the corrected boundary.
- Do not solve a foreseeable model-scale lifetime problem by only increasing a
  fixed object/count limit. Separate active resources from transient staging,
  release or evict resources at their real lifetime boundary, retain bounded
  byte/count backpressure for untrusted clients, and prove reuse beyond the old
  limit. A larger constant is acceptable only when the resource is inherently
  concurrent and the bound is derived from an explicit product contract.
- Treat compiled-kernel storage as three distinct lifetimes: persistent on-disk
  JIT cache, bounded concurrent upload staging, and resident mapped executable
  state. If measured resident code bytes threaten larger-model execution, use a
  byte-budgeted rematerializable kernel proxy: evict only an idle native handle,
  retain immutable name/image/ABI identity, and remap the cached HSACO on the
  next launch without recompilation. Never invalidate a live Triton cache entry
  by closing its handle without a rematerialization path, and never substitute
  another fixed kernel-count limit for this policy.
