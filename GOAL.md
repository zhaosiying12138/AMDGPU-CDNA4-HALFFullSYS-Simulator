# Goal contract: host-side AMDGPU gem5 simulation

**Goal ID:** `GSIM-001`

**Plan:** `AMDGPU-SIM-V1`, revision `16`

**Current state:** `AgentENV and generic model-backed fast copy are integrated on main; the corrected custom-kernel config awaits a user-controlled WSL restart; unchanged-upstream SGLang/vLLM TP1 and TP2 plus Qwen3.5-9B TP16 remain unaccepted`

**Current phase:** `activate AgentENV isolation on main, then complete unchanged-upstream SGLang and vLLM Qwen3.5-0.8B at TP1 and TP2 with generic fast copy enabled; cancel the 0.8B TP4 lane and continue directly to Qwen3.5-9B TP16`

## 2026-08-17 authoritative resumed goal

This section supersedes the CP-0029 resume pointer and every older scheduling
statement that requires Qwen3.5-0.8B TP4. The active goal has resumed on
`main`; historical checkpoints remain evidence, not execution authority.

1. AgentENV and generic model-backed fast copy are mandatory foundations for
   subsequent model bring-up. AgentENV isolates filesystem, PID, network,
   cache, socket, endpoint, log, SMI, build, and temporary state. Fast copy is
   enabled explicitly with `source scripts/fastcopy_mode.sh fast`; legacy mode
   remains available for a bounded A/B or fallback investigation.
2. SGLang and vLLM are pinned, unchanged upstream inference engines. Accepted
   runs do not edit either engine, register project replacement Triton
   operators, copy model code, monkey-patch framework behavior, or branch on a
   model/operator/shape/kernel/PC. Required repairs belong only at a generic
   self-runtime, ROCr/HIP facade, runtime-gem5 bridge, memory, queue,
   synchronization, collective, or ISA boundary.
3. A TP1 sandbox may own exactly one live `gem5.opt` process. Launch
   orchestration must refuse extra managed simulators instead of silently
   consuming them. Parallel tests use distinct AgentENV sandboxes and must not
   share worktrees, builds, caches, endpoints, SMI leases, or process groups.
4. SGLang TP1 and vLLM TP1 may discover failures concurrently on independent
   branches. Each generic fix is a small committed change, receives an impact
   review against both engines, and enters `main` serially. The other branch
   rebases onto that new main before continuing; sandboxes never write main
   concurrently.
5. The model ladder is: unchanged-upstream SGLang TP1 and vLLM TP1, then both
   engines at TP2, then Qwen3.5-9B TP16 on the upstream AMD path. The
   Qwen3.5-0.8B TP4 tasks are cancelled. No successful custom-operator vLLM
   probe counts as upstream vLLM TP1 or TP2 acceptance.
6. Model acceptance with fast copy requires the same model revision, inputs,
   seed, decoding settings, parameter bytes, output/logit checks, token IDs,
   and cleanup behavior as the legacy path. A speedup is reported only after
   correctness equality; a fast-copy failure falls back through the explicit
   legacy switch rather than widening framework-specific code.
7. The corrected AgentENV `.wslconfig` is staged, but the running kernel is
   still stock until the user performs the separately announced second
   `wsl --shutdown`. The assistant must never invoke that shutdown implicitly
   or while unrelated important work is running.

**Execution checkpoint (2026-08-15):** the simulator-aware 16-slot
`rocm-smi` lease inventory passes focused tests and a real managed-gem5
OFF -> ON -> OFF lifecycle probe. The official ROCr Model Interface path now
creates standard AQL queues through the project model DSO, exports one
process-owned shared backing, and sends monotonic QueueDoorbell notifications.
The bridge binds KMT GPU VAs to authenticated backing offsets, authorizes only
the exact same-UID gem5 peer for cross-process memory access, observes real
64-byte AQL publication batches, and converts descriptor VAs into
owner/generation-bound `ResidentKernelView` objects. The same generic command
processor reads descriptor, kernarg, completion-signal object and MQD through
`GPUNativeMemory` for explicit-mapper and standard resident-kernel sources.
A fresh live run now executes three unchanged-upstream ROCr dispatches through
the shared gem5 CU lifecycle: a standard device blit, the user code-object
kernel, and a second device blit. All three decrement their completion signals,
retire their queue slots, release resource pins, and publish durable native
trace records; one terminal record binds clean owner teardown. The user process
verifies its numerical oracle and exits 0, gem5 exits 0 at session completion,
and `host_fallback_count=0`. The absent-only runner and independent verifier
freeze this as `artifacts/evidence/upstream-rocr-aql-v2-accepted`, with exact
argv/environment/PID/start-time identity, four ROCr queue lifecycle observations
(one standard control packet plus three kernel packets), three native execution
tickets, one session-complete record, and all artifact hashes revalidated.
A separate absent-only runner and independent verifier freeze one ordinary
public-HIP vector-add as `artifacts/evidence/hip-facade-runtime-v1-accepted`:
bit-exact output, one durable native retirement, one clean session-complete
record, natural worker/gem5 exit, current identity, and zero fallback. This is
formal generic HIP allocator/copy/stream/module/kernel/synchronize acceptance.
The immutable official ROCm product
`rocm-pytorch-v2-49c6121256fd2ca673cf44f5c8235d5dd601dfdc14bbbbc3cabf6fbc710aad2f`
keeps AMD's HIP, COMGR, RCCL, torch 2.11.0, and Triton 3.6.0 artifacts and
replaces only ROCr/KMD. An ordinary upstream-only Python demo using
`torch.cuda`, copy, add, sigmoid, and sum is independently accepted in
`artifacts/evidence/rocm-pytorch-eager-multiop-wavefront-fix-v2-accepted`:
every tensor is bitwise correct, inputs are unchanged, outputs are
fresh/non-aliasing, eight standard AQL dispatches retire,
`host_fallback_count=0`, and the worker/gem5 session exits naturally. The same
product also runs an unchanged upstream Triton AMD program containing add,
masked branching/transform, and reduction kernels through the official
`HIPDriver`: 12 native dispatches retire, all three outputs are bitwise exact,
inputs are unchanged, outputs do not alias, gem5 exits normally, and all 16 SMI
slots return to `OFF`. That live smoke required one generic gem5 fix: when an
AMD vector memory instruction has an empty execution mask, wavefront execution
now cancels the exact resource reservation made at issue time instead of
leaving a stale global/local-memory pipeline counter. No runtime-gem5 bridge,
Triton, PyTorch, operator, model, tensor-shape, code-hash, or program-counter
special case was added. The absent-only runner and independent verifier freeze
that execution as
`artifacts/evidence/upstream-triton-amd-multiop-v2-accepted`: result SHA-256
`8437bd53841ff9ea44656d2ab331b768e75d80ec45675c9150b9bc71d865f842`,
manifest SHA-256
`a6ef2f15d3cf23fee3319e02c1e5d6c3644339b5e9cab3320efb69cf45a50dfe`,
three distinct HSACO identities, 28 fresh private JIT-cache files, 12 unique
execution tickets, one clean session-complete record, and zero host fallback.
This is not complete event/error, PyTorch API, upstream Triton API,
or torch.compile acceptance. The upstream RowParallel TP2 prerequisite is now
independently accepted in
`artifacts/evidence/vllm-rowparallel-n2-bf16-1024-v5-accepted`: both real Qwen
weight shards load through the upstream parameter hook, inherited
`RowParallelLinear.forward` executes one local dense projection per rank, and
the public out-of-place all-reduce returns the exact independent BF16 ring
oracle. Both local projections and the final result have `max_abs=0`,
`relative_l2=0`, and `mismatch_count=0`; two dense and two device-SUM dispatches
retire durably, model-window Gloo tensor traffic is zero, host/fallback counts
are zero, and all processes, FDs, and 16 SMI slots clean up. This is one real
layer contract, not full-model TP acceptance. No full-model TP2/TP4 vLLM or
SGLang result, and no 9B TP16 `torch.compile` result, is currently claimed. The
upstream-zero-diff rule and the single runtime-gem5 bridge boundary remain
highest priority.

The immediate implementation slice is full-model TP2 on the now accepted
framework-standard boundary: preserve upstream vLLM model construction,
parameter loading, parallel state, layer forwards, attention selection, and
sampling while the OOT backend supplies only the generic device/runtime and
collective capabilities. The next acceptance is a complete Qwen3.5-0.8B
prefill plus predeclared multi-token greedy decode, not another isolated layer.
After TP2 it expands unchanged to TP4. The accepted Triton gate remains a
framework-neutral regression prerequisite. The runtime provider/model DSO and
gem5 GPU adapter remain behind one bridge; PyTorch, Triton, vLLM, and SGLang are
not modified to learn about the simulator. SMI remains a 16-slot regression
gate, not a model-TP claim. CCL stays generic for world sizes 2..16. AMD
attention is selected by the upstream ROCm framework path; FlashInfer is neither
required nor forced.

**Model correction:** the official target is `Qwen/Qwen3.5-0.8B`; there is no
official `Qwen3.5-0.9B` checkpoint in this project.

The pinned Qwen checkpoint is prepared offline under `models/` at revision
`2fc06364715b967f1860aea9cf38778875588b17`. The first model gate is the
text-only path: its 24-layer topology (18 linear-attention/GDN layers and 6
full-attention layers) has a source-grounded 15-contract operator manifest.
Vision is deferred. Full ROCm and OpenCL CTS are not prerequisites for this
goal; every operator used by the checkpoint still needs an AMD execution
result, and CPU or NVIDIA fallback never counts as success.

## 2026-08-15 provider checkpoint

The next immutable user-facing product is being built from the signed AMD
ROCm 7.2.3 Jammy user-space closure and the pinned ROCm vLLM wheel index. The
APT lock contains 63 packages and explicitly excludes a second ROCr/HSA
provider; the wheel lock contains 215 CPython 3.12 manylinux wheels with
content hashes and dist-info checks. The existing native product remains the
only ROCr/KMD boundary and the runtime-gem5 bridge remains the only simulator
translation layer. No torch, Triton, vLLM, or SGLang source modification is
allowed in this product step.

The 16-slot `rocm-smi`/`gemsim-smi` lease contract is already a regression
gate: live authenticated gem5 leases report `ON`, while absent, stale, or
identity-mismatched records report `OFF`; the tools never probe `/dev/kfd` or
`/dev/dri`. After provider and single-card HIP/Triton checks pass, the model
sequence is unchanged-upstream vLLM Qwen3.5-0.8B TP=2, vLLM TP=4, SGLang
TP=4, then Qwen3.5-9B TP=16 with upstream `torch.compile`. AMD attention is
selected by the upstream ROCm path; FlashInfer is not a required dependency
for the acceptance claim.

## 2026-08-15 model-first execution checkpoint

The next work is interactive model bring-up rather than another round of
evidence formatting. Existing accepted products and bundles are preserved;
new Triton cache-count/provider publication checks are off the critical path.
Run the real unchanged-upstream vLLM path first at TP1, then TP2 and TP4, and
then unchanged SGLang TP4. The 9B TP16 `torch.compile` run follows once TP4
starts. Diagnose every failure at the lowest shared layer and repair it
generically; do not add an operator/model/shape-specific bridge workaround.
Prioritize weak-link probes (startup, weight loading, first HIP allocation,
first GEMM/attention/cache update, first collective, and teardown) over broad
matrices that cannot expose a new failure while the model is not running.
Formal acceptance is deferred until a stable model path exists.

## Product-level execution contract and priority

The product is not a protocol test program. The following order is the
authoritative execution goal and overrides any older text that schedules a
profile or simulator optimization before broader operator enablement:

1. **OpenCL direct executable:** compile a user `.cl` program and link its host
   program against only the repository-built OpenCL/runtime stack to produce a
   normal executable. Running that executable alone must transparently start or
   connect to the managed gem5 device path, submit the kernel through our
   runtime, wait, copy results back, verify the oracle, and exit successfully.
   The user must not manually invoke a protocol endpoint or construct transport
   records.
2. **Triton normal Python:** an ordinary Python program using the pinned Triton
   frontend must select the simulator backend through Triton's normal
   driver/runtime APIs. Compilation, code-object load, launch, wait, and result
   copy must reach gem5 implicitly; a manually launched C test endpoint or an
   application-specific launcher does not satisfy this gate.
3. **Model operator coverage:** enumerate every operator, dtype, shape, stride,
   mutation, and state transition required by the pinned text model, then make
   every entry compile and execute through the same user-facing Triton/runtime
   path with a differential oracle and zero CPU/NVIDIA fallback.
4. **Upstream framework integration:** install compatibility/diagnostic plugins
   only in private AMD environments, but make standard ROCm PyTorch the formal
   device contract. Unchanged vLLM and SGLang must auto-select their in-tree
   ROCm platforms, upstream Triton HIP, and RCCL/NCCL interfaces. PyTorch's
   dispatcher, c10d, and official framework entry points remain bounded
   extension surfaces, not an excuse to replace standard device semantics.
   Do not modify pinned open-source torch, Triton, vLLM, or SGLang. Monkey
   patches may be used only for diagnosis or prototyping and must be absent
   from the accepted operator, layer, model, and TP execution path.
   Preserve each framework's standard model, parameter loader, parallel state,
   group coordinator, and tensor-parallel layer semantics; never assign private
   TP globals or replace them with a project-specific sharding system.
   The complete ownership contract is `docs/framework-runtime-layering.md`.
5. **Single-device model:** run the complete text model on one simulated device
   through prefill and a predeclared multi-token decode window. A one-token
   smoke is useful evidence but is not the stable-inference acceptance gate.
6. **CCL before TP:** stabilize the standalone CCL API with unit and
   multi-process tests for collective math, ordering, rank/epoch lifecycle,
   failure propagation, reconnect, and every N=2..16 topology case before exposing
   it through the formal PyTorch/vLLM communicator plugin.
7. **Multi-TP model:** run one rank per independent gem5 daemon and complete the
   full model with stable multi-token inference at TP=2, then the formal
   Qwen3.5-0.8B TP=4 gate. Separately run Qwen3.5-9B TP=16 with upstream
   vLLM `torch.compile`; do not generate a TP=8 matrix.

8. **Upstream attention policy:** do not hard-code FlashInfer on AMD. Let the
   pinned vLLM ROCm selector choose AITER/ROCM_ATTN/TRITON_ATTN, record the
   selected backend, and fail closed if the chosen path reaches CPU fallback or
   an unbound project kernel. `torch.compile` is required in the 9B TP=16 lane.

9. **SGLang portability:** run the same Qwen3.5-0.8B TP=4 workload through
   upstream SGLang without editing SGLang or the runtime-gem5 bridge. Use only
   an official SGLang platform/device/backend extension point; otherwise stop
   at a documented extension-gap result and implement one generic adapter
   below the framework, never a copied model/operator path.

   The selected audit snapshot is SGLang `0.5.17` (full unpacked-wheel identity
   in `tools/sglang_source_manifest.json`). It provides the official
   in-tree `RocmSRTPlatform`, an official `sglang.srt.platforms` diagnostic
   extension point, and standard PyTorch distributed construction. The formal
   model path leaves `SGLANG_PLATFORM` unset so SGLang auto-selects ROCm from
   `torch.version.hip`; the shared third-party c10d `ProcessGroup` is a
   diagnostic/fallback. The historical `0.5.10.post1` snapshot is retained only
   as evidence of the former extension gap. Platform discovery, actual group
   selection, in-place/async semantics, device CCL traces, and cleanup remain
   separate gates.

   SGLang `0.5.17` publishes CPython 3.10--3.13 wheels while the existing vLLM
   product is CPython 3.14. The SGLang lane therefore uses a separate immutable
   repository-owned CPython 3.13 conda product; it must not downgrade or mutate
   the accepted vLLM environment. The product installs the upstream SGLang
   wheel without CUDA-only dependency defaults, pins the AMD dependency set,
   and snapshots only the shared GemSim CCL and SGLang OOT plugins.

10. **Shared device facade before framework models:** the host-tested c10d
    adapter is not a model-execution claim. Both frameworks must ultimately see
    the same HIP-compatible PyTorch device, allocator, stream/event, copy,
    module/launch, and Inductor/Triton semantics backed by self-runtime and the
    runtime-gem5 bridge. The primary path is a self-built ROCr/HIP-compatible
    stack selected by library search paths so unmodified ROCm PyTorch exposes
    its normal `torch.cuda` HIP surface. PrivateUse1 is an isolated fallback
    only if a frozen upstream caller cannot use the compatible ABI. Do not grow
    a per-operator `torch.library` compatibility table to emulate a device.

    Until that facade is present, the SGLang diagnostic OOT platform must fail closed
    instead of advertising the CPU engine: SGLang's CPU path may select its
    own AMX shared-memory collective and would bypass GemSim c10d. Host-only
    ProcessGroup tests remain valid evidence for communication semantics, but
    cannot unlock SGLang TP4.

11. **SGLang portability is a first-class acceptance lane:** Qwen3.5-0.8B
    TP=4 must run through the unchanged pinned SGLang 0.5.17 ROCm path and
    the same facade/CCL/runtime-gem5 bridge used by vLLM. FlashInfer is not
    used as a workaround; AMD upstream AITER/ROCM_ATTN/Triton selection is
    authoritative. If an upstream extension point is insufficient, repair the
    shared ABI/facade once and retest both frameworks rather than adding a
    SGLang branch or operator-specific bridge route.

    The capability state is authoritative in
    `tools/framework_device_facade_manifest.json` and is recomputed by
    `tools/framework_device_facade_audit.py`. The device/model gates remain
    blocked until ROCr provider, HIP runtime, PyTorch ROCm device, and RCCL ABI
    capability families are all accepted.

    The ROCr provider must reuse the pinned upstream `libhsakmt` unchanged via
    its official `HSA_MODEL_LIB` Model Interface 1.1. The project model DSO is
    the single KMD-removal boundary; it must not re-export or clone the 124
    upstream thunk functions. Its accepted first rung is limited to a
    process-owned sparse model memfd and the typed managed
    `OPEN_KFD/GET_VERSION/CLOSE_KFD` lifecycle. Unimplemented KFD/DRM calls
    preserve outputs and fail explicitly. Fork inheritance is process-owned:
    a child cannot use or close the parent's provider and may only discard its
    local copy before reconnecting. Topology, memory coherence, AQL
    queue/doorbell, signals, pointer info, and code-object dispatch remain
    required before ROCr/HIP promotion.

The CCL protocol bound and the model-sharding bound are deliberately separate.
CCL v1 is a bounded generic `world_size=2..16` protocol and must cover every
integer world size in that interval, including non-power-of-two 3/5/7/15. A
specific Qwen TP degree is accepted only when hidden, intermediate, query, KV,
GDN-head, and vocabulary dimensions satisfy the required shard semantics;
protocol support for rank 16 never implies that Qwen3.5-0.8B TP16 is valid.
Live CCL acceptance must include N=2/3/4/8/16; unit or mock coverage at N=16
does not count as a live rank-16 collective result.

The standalone device arithmetic implementation requires no gem5 or managed
runtime special case. The independent `gemsim-ccl` package runs normal Triton
BF16/FP32 SUM through generic gfx950 HSACO; CCL v1 BF16 rounds to nearest even
after every planner receive, zero-length steps do not dispatch, and Qwen-sized
1024/2048/7168 vectors pass the one-step bitwise matrix. The formal product-v1
evidence now binds the exact base Python, noneditable product plugin snapshots,
runtime DSO, gem5 binary/config, all Triton code objects, stats, log and trace.
It proves 24 retired and 24 type20-durable device dispatches, 23 reuse records,
one clean session completion, bitwise BF16/FP32 outputs, and zero authoritative
host fallback. This accepts the one-step private-workspace primitive only.
Device-backed standalone BF16 allreduce is accepted at N=2/3/4/8/16 for 1024
elements. The external verifier reopens every artifact,
recomputes descriptors/plans and the per-hop BF16 RTNE ring oracle, and binds
each normal Triton SUM lifecycle to the exact rank session. N=2/3/4/8/16
execute 1/2/3/7/15 SUMs per rank, including uneven N=3 chunks and 240 total N=16
device reductions. Every rank output is bitwise equal to the versioned ring
oracle; every result has zero host reduction/fallback, measured FD delta zero
and no orphan process. This completes the mandatory standalone live topology
matrix and unlocks the formal out-of-place vLLM communicator adapter. That
adapter now has its own N=2 BF16/1024 live acceptance: the real pinned
`GroupCoordinator.all_reduce` entry point reaches the out-of-tree
`GemsimDeviceCommunicator` and reusable `gemsim-ccl` engine, both ranks execute
one normal Triton SUM, the independent ring oracle matches bitwise, all 24
audited Gloo tensor APIs remain at zero calls, inputs remain immutable, outputs
are fresh, and host reduction/fallback/FD delta/orphans are zero. This accepts
the standalone framework communicator only; it does not claim a sharded vLLM
layer or TP model execution.
Authenticated peer-rank rendezvous and group-wide first-error propagation pass
for live N=2/3/4/8/16, with generic capacity and cleanup checks at every
N=2..16. A separate live host-mock gate now drives the real planner and carrier
through ordered DATA, immutable staging, mock reduction/copy, and CONSUMED at
N=2/3/4/8/16. It covers zero chunks, credit pressure and reordering, replay,
group abort, peer loss, FD conservation, and orphan cleanup. That gate reports
nonzero `host_mock_reduction_count`, zero `device_sum_count`, and
`device_collective_acceptance=false`; it accepts transport, ordering, fault and
cleanup behavior only. The formal device-live matrix and the N=2 framework
communicator gate are complete. TP model execution remains unaccepted until
Qwen's adapters pass real shard construction, per-rank weights/state,
collective sequence, output, trace, fault, and cleanup gates. Start with a
Qwen-sized `RowParallelLinear` local projection plus out-of-place all-reduce;
then add the column/merged/QKV/vocabulary/GDN/full-attention-specific contracts
one at a time. The live protocol models
every planner ordinal as two independently bound transfer chains: outbound
DATA retains sender credit until its exact CONSUMED tuple, and inbound DATA is
acknowledged only after immutable staging plus copy/device completion. A step
finishes only when both chains finish. Payloads larger than the 16 MiB carrier
record bound are divided into contiguous segments with consecutive collective
sequences and explicit global offsets; every segment remains private and the
public fresh output is committed once, after all segments pass. Formal
acceptance comes only from an external absent-only verifier that reopens and
rehashes artifacts, reconstructs planner/order and numerical results, and
binds authoritative traces; self-reported design evidence cannot unlock TP.

Profiling is diagnostic, not a prerequisite between these stages. Record cheap
timing observations during correctness runs, but start a profiling/optimization
checkpoint only when a real OpenCL, Triton, layer, or model workload is slow
enough to threaten operator-matrix or full-model progress. Any optimization
must preserve the user-facing path, output oracle, traces, and no-fallback
boundary. The CP-0026 endpoint and locked `gpuReadWrite` fixture remain required
regressions, but they are not product-level OpenCL or Triton success.

## Success criteria

- A self-contained copy of this directory can prepare and run the pinned stack
  on the same Linux ABI/architecture without network access after preparation.
- A user can compile a `.cl` kernel plus host program into a normal executable
  linked against the repository-local OpenCL/runtime stack, run that executable
  directly, and receive correct gem5-computed output without a manual endpoint.
- A normal Triton Python program implicitly uses the simulator driver/runtime,
  submits real code objects to gem5, and receives correct output without an
  application-specific C endpoint or hidden CPU/NVIDIA execution.
- Running that Python program JIT-compiles only a missing Triton kernel artifact
  into a persistent cache. It never rebuilds gem5 or `self-amdgpu-runtime`;
  both are prebuilt and communicate across the generic Unix-socket boundary.
- A previously unseen valid Triton kernel is handled by the same generic
  HSACO/metadata/kernarg/allocation/dispatch/D2H path without a kernel-name,
  code-hash, shape, PC-sequence, trace-schema, or simulator-oracle C++ change.
- Qwen operator registration is supplied by an out-of-tree package installed in
  the private AMD environment. The pinned torch and vLLM source trees remain
  unmodified; disabling the plugin restores their original behavior, and the
  accepted execution path contains no monkey patch.
- Pinned Triton core, its compiler coordinator, frontend/JIT/cache, and upstream
  AMD lowering remain unmodified. The project-owned `gemsim_amd` backend alone
  implements Triton's backend driver/compiler contracts and hides the
  self-runtime/gem5 transport from Triton and vLLM callers.
- KMD-removal-specific behavior is concentrated in one runtime-gem5 bridge.
  The self-runtime API above it and gem5 GPU adapter below it stay stable;
  vLLM/Torch/Triton never see bridge records, while gem5 dispatcher/CU/Vega
  code never sees framework, model, tensor-role, or collective identities.
- All production changes are generic across operators. Admission and execution
  inspect only target/ABI, code-object/kernarg metadata, resources, ownership/
  bounds, synchronization, and ISA semantics; they never inspect an operator/
  model name, tensor shape/role, image hash, expected output, or fixed PC
  sequence. New-operator failures are fixed once in the shared semantic layer
  and covered by multiple unrelated kernels.
- After equivalent gates pass through the concentrated bridge, remove
  superseded fixture routes, duplicate codecs, copied upstream state machines,
  and obsolete environment launchers. A recoverable source backup and
  historical evidence remain; dead production code does not.
- A host-native simulator daemon/library runs on the physical host without
  `VEGA_X86`, the gem5 x86 CPU model, or x86 system/CPU ports, while reusing
  the existing GPU packet, Vega ISA, memory, queue, signal, and CU functional
  modules and preserving their accepted behavior.
- No production AMD UMD/KMD library is loaded; no `/dev/kfd` or `/dev/dri` is
  opened; no CPU arithmetic fallback occurs in acceptance mode.
- One simulated device executes the complete official Qwen3.5-0.8B text path
  through full prefill and a stable, predeclared multi-token decode window.
- Two independent gem5 daemons and vLLM TP=2 execute the same full model through
  prefill and stable multi-token decode using the project CCL path.
- Before that TP gate, the standalone CCL API passes deterministic unit and
  multi-process suites for all required collectives, N=2..16 rank lifecycle,
  stale epochs, timeout/peer failure, cleanup, and reconnect, with host reduction
  and fallback counters zero.
- The selected greedy token ID exactly matches a pinned reference; numerical
  and trace evidence satisfies the tolerances in PLAN.md.
- The full Qwen3.5-0.8B model passes the same gates at TP=4. The generic N-rank
  protocol remains valid through rank 16, with shape constraints and any model
  limitation explicitly recorded; this does not require a TP=8 matrix.
- Qwen3.5-9B TP=16 passes a separate scale gate with vLLM V1, the upstream
  ROCm attention selector, `torch.compile` enabled, exact rank/shard/resource
  identities, zero fallback, and clean sixteen-daemon teardown. This is a
  scale acceptance target, not a promise that every Qwen model shape is legal
  at every TP degree.
- Qwen3.5-0.8B TP=4 also passes through pinned upstream SGLang with the same
  unchanged runtime-gem5 bridge, shared CCL engine, rank/shard oracle, token
  output, backend-selection, fallback, trace, and teardown gates. The SGLang
  source tree and bridge hashes are explicitly cross-bound to the vLLM lane.
- Triton's user-facing `tutorial/01-vecadd.py` request (mapped, without editing
  upstream, to the pinned checkout's `python/tutorials/01-vector-add.py`) runs
  transparently through the GemSim device path, with no CPU fallback, and its
  output matches the pinned oracle.
- Profiling and 80/20 optimization are triggered only by a measured operator,
  layer, or model bottleneck that threatens end-to-end progress; they are not a
  mandatory gate after the first transparent operator.
- Any CPU-side threadblock parallelism is enabled only behind an explicit
  simulator mode and passes dependency, barrier, atomic, determinism, and
  race-regression gates; it must improve the measured operator path rather
  than merely inflate host utilization.
- The repository conda product installs simulator-aware `rocm-smi` and
  `gemsim-smi` commands. They expose 16 logical slots and report `ON` only for
  live managed gem5 leases whose process, executable, daemon/job identity, and
  private endpoint validate; otherwise they report `OFF`, without probing
  hardware device nodes or loading production management libraries.

## Model-scale acceptance lanes

- `QWEN35-TP4-01`: Qwen3.5-0.8B full text model, TP=2 prerequisite followed by
  TP=4 formal acceptance. All local projections, upstream parameter loaders,
  CCL collectives, attention backend selection, `torch.compile` state, logits,
  token IDs, and lifecycle evidence must be bound to the same product.
- `QWEN35-9B-TP16-COMPILE-01`: Qwen3.5-9B, TP=16, vLLM V1 with
  `torch.compile`. The runner must record the actual ROCm backend selected by
  vLLM and reject any hidden CPU/NVIDIA or project-specific attention fallback.
- `QWEN35-SGLANG-TP4-01`: Qwen3.5-0.8B, TP=4, pinned upstream SGLang
  auto-selecting its in-tree ROCm platform, upstream Triton attention (the
  upstream AMD recommendation), and the standard RCCL/NCCL-compatible path.
  The generic c10d backend is an independently gated fallback, not the required
  model data plane.
  It is a framework-portability gate, not permission to duplicate vLLM's model
  implementation, use SGLang's CPU/AMX fast path, or add bridge branches.
- Neither lane permits per-operator bridge patches. A failure is fixed at the
  lowest generic runtime, compiler, dispatcher, communicator, memory, or
  framework OOT boundary and then replayed across at least two unrelated
  operators before the lane resumes.

## In scope

ROCr/HIP/CLR/libhsakmt-compatible host glue, gem5 host bridge/daemon,
gfx950/MI355X code-object execution, functional N-rank transport and RCCL
semantics, Triton AMD lowering adapter, PyTorch and vLLM integration, Qwen
GDN/full-attention layers, official SGLang platform/c10d integration, operator profiling and correctness-preserving
simulator scheduling, a simulator-aware `rocm-smi` status client,
reproducible tests, evidence, checkpoints, and offline packaging.

## Out of scope until explicitly promoted

Vision execution, real hardware validation, silent CPU or mock execution,
unbounded ROCm API compatibility, full ROCm/OpenCL CTS, timing claims before FabricModel, elastic
rank shrink, unsafe or nondeterministic threadblock parallelism, hardware
`rocm-smi` compatibility claims, and committing weights/build
products/environments.

## Resume contract

Do not infer progress from an old chat or a historical checkpoint next action.
Read `PLAN.md`, this file, `ENGINEERING_CONSTRAINTS.md`, `SOURCE_LOCK.json`, and
`PROJECT_LANES.json`, then inspect current source/tests and the evidence named by
the relevant PLAN gate. `state/checkpoints/`, `state/evidence/`,
`state/bitlessons/`, and `state/current.json` are retained history through
CP-0030; the stale CP-0030 pointer is not an active scheduler and must not
override the current product priority or rerun its obsolete CP-0031/RMSNorm
next action. A result advances only when its current source, exact identities,
tests, failure gates, and evidence pass. Never redo a passed gate merely to
close a checkpoint, silently change the model ID, or preserve a known-wrong
architecture for historical bookkeeping.

### Blank-context continuation prompt

```text
继续执行 amdgpu-sim 计划。当前目录是 /home/zhaosiying/amdgpu-sim。
先读取 PLAN.md、GOAL.md、ENGINEERING_CONSTRAINTS.md、SOURCE_LOCK.json 和
PROJECT_LANES.json，再核对当前源码、测试和相关 evidence。state/current.json
及 CP1-30 只作历史档案，不执行其中过时的 CP31/RMSNorm next_action。当前从
P8/P9 继续：隔离 product environment、device SUM、N=2/3/4/8/16 authenticated
ordered device collectives、真实 GroupCoordinator 到 OOT communicator/engine
的 N=2 allreduce，以及 16-slot rocm-smi OFF/ON/OFF 均已有验证，不要重跑。现在先实现并验证
Qwen 实际 1024 hidden-size 的 RowParallelLinear 分片、本地投影和 out-of-place
allreduce；通过后再逐项开放其余 TP adapter，最后进入完整 Qwen TP2 和 TP4。
已通过的
live host-mock 只验证 transport/order/fault/cleanup，不可以代替 device gate。全部 standalone
不要把 host-only rendezvous、mock arithmetic 或 standalone communicator 称为
RowParallel、模型 TP 或完整 Qwen acceptance。
不要重做
bootstrap、source freeze、CP26/CP27/CP28 已通过的执行门禁，也不要修改已冻结的
SOURCE_LOCK.json 或已登记的 PROJECT_LANES baseline。以下 CP10-CP27 描述都是
历史边界，不是当前 next action。CP-0010 只实现 18 个 typed KMT
操作的固定宽度 shim、版本化 daemon envelope、模拟资源生命周期和
no-device 证据；它不是完整 124-PFN ROCr/libhsakmt provider，也不宣称
KFD attach、HIP、OpenCL、Triton、PyTorch 或 vLLM 能力。CP-0011 已冻结
两份真实 gfx950 HSACO 的 ELF/MsgPack/descriptor/kernarg provenance，但
gem5 gfx950 decoder、unsupported opcode、pinned device-libs/toolchain proof
已完成；CP-0013 A1 仅完成 versioned HSACO fixed-record transport/staging，
不包含 PT_LOAD mapping、动态 AQL/kernarg 或真实 code-object execution。
在该历史边界尚无 Triton 端到端通过案例；当前 CP28 结论见下文。CP-0015 已完成 no-VEGA_X86
control-core/build/audit 边界，CP-0016 又完成了复用既有 memory/queue/signal/dispatch
state 的 functional parity adapter。CP-0017-A 进一步完成了锁定 gfx950 HSACO 的
CP13 staging、PT_LOAD 文件复制/BSS 清零、descriptor/entry 绑定和 page-lifetime
测试，但没有完成动态 AQL/kernarg、GPU instruction execution 或 Triton。CP-0018
B0 又完成了 no-x86 dispatch admission：descriptor/entry 关系、280-byte
hidden kernarg、64-byte AQL materialization、queue-control 和 listener-contract
smoke；这些字段仍只经过 host functional memory，未发布 native queue。CP-0019
B1 新增了提取出的 `HostNativeQueueCore` 和
`HostNativeCommandProcessorCore`：在 no-x86 target 中解析 host-owned GPU VA，
发布、ring、顺序 fetch 一个 64-byte AQL packet，读取 descriptor/MQD/kernarg/
completion-signal object，并复用 `HSAQueueEntry` 完成 native CP-core admission。
这里 `aql_submitted=true` 只表示提取出的 native CP core 已接受 packet；legacy
HSAPP/GPUCommandProcessor SimObject 没有 link 或 instantiate，GPUDispatcher/CU
也未连接，read index、retire、signal decrement、instruction fetch 和 kernel output
仍为 false。CP-0020 的 B2 已在同一 no-x86 front-end 中连接
GPUDispatcher/Shader/ComputeUnit/Vega instruction path，并完成锁定
gfx950 gpuReadWrite 的 4 个 256-item workgroup、16 个 wave64、每 wave 19
个 instruction-start（304 总数）和输出/队列/信号差分；这只是单一 fixture
的功能边界，不是通用 gfx950、timing、fence/atomic、TLB/Ruby 或任意 HSACO
支持。CP-0021/CP-0022/CP-0023 已依次冻结 Triton provenance、payload-v2 wire
codec 和 runtime client/local native admission 边界。CP-0024 又增加了
owner-bound type-18 handler 源码、type-19 ACK 出站 plumbing 和共享 route-policy
harness，并保留 bit 8 未广告时的 canonical reject。CP-0025 完成了真实
runtime-to-gem5 两代 owner control lifecycle：依赖与 bit 8 正向选择，MAP、逻辑
alignment 8 ALLOC（page backing 隐藏）、既有 v1 `MEMORY_COPY_H2D`、daemon-built
AQL admission、type-19 ACK、type-20 retire、UNMAP、disconnect cleanup 和 reconnect
都通过，但当时尚未连接 GPUDispatcher/ComputeUnit。CP-0026 随后把这个 lifecycle
接到真实 GPUDispatcher/CU，并对锁定的 zero-preload `gpuReadWrite` fixture 证明
输出正确。CP-0027 进一步完成首个产品入口：普通 OpenCL `.cl` 程序和 host
程序链接本仓库的 `libOpenCL.so.1` 成为可直接执行的二进制；该进程自动启动
gem5、编译并提交 exact `vecadd`、回读 C、校验 `C=A+B` 并清理退出，零
CPU/NVIDIA fallback。它当前仍限制为一个 OpenCL context 一次 submit。
CP-0028 又完成普通 Python/Triton 产品入口：外部 `gemsim_amd` backend 通过
Triton 正常 driver/JIT/launcher 路径生成并加载 pure-b010 gfx950 `add_kernel`，
在同一 Python 进程、同一 managed gem5 session 中以两组独立输入连续 launch，
两次 bit-exact `float32 C=A+B` 都通过且 fallback 为零。该结论只覆盖连续
float32 vecadd（98,432 elements、BLOCK_SIZE 1,024），不覆盖任意 Triton kernel
或模型算子。CP-0028 的 v6/v7 最终仓库本地安装尝试均明确 NON-PASSING：v6
被 `/opt/rocm` 默认路径污染后中止，v7 因 queue mock 的 pre-ACK 测试竞态
fail closed；它们不能作为安装 authority。CP-0029 已用 test-only 修改稳定
queue gate，并从空 schema-8 prefix 完成安装、OpenCL/Triton、污染和
	active-isolation 验证；它不扩大 Triton vecadd 的算子边界。CP-0030 随后接受
	Qwen3.5 所需的 BF16 SiluAndMul 最小子门禁：normal Triton decode 和
	masked-prefill 两种形状均以 fresh/repeat 执行、精确 trace、零 mismatch、零
	nonfinite 和零 fallback 通过；它仍未完成 gate/up 与 down GEMM，因此不接受
	完整 MLP contract。当前下一步按 P8 的 CCL standalone/device-live 顺序，
	随后再实现多 TP 完整模型稳定多 token。gem5
性能优化从属于这条主线：只有真实算子、层、完整模型或 TP 运行的可复现 profile
证明某项是显著大头，且修复范围小、收益预期高、回归边界强时才实施。优化耗时
超过推进下一个模型 blocker、实测收益不显著，或引入新的正确性和生命周期修复
负担时立即停止或推迟。不安全或不可证明独立的 workgroup 必须保留串行执行。
模型路径和 N-rank registry 可用后，再实现只观察模拟 daemon
lease/epoch 的 `rocm-smi` ON/OFF 工具，绝不探测物理卡。
	每个原子进展同步 GOAL/PLAN、当前测试和内容寻址 evidence。历史 checkpoint
	保持不可变，但不再为了分配新 CP ID 或封账而推迟已知错误修复和产品主线。
```

`CP-0013` is accepted at the A1 code-object transport/staging boundary. It
binds fixed 4096-byte BEGIN/CHUNK/COMMIT records, capability negotiation,
per-chunk CRC-32C, whole-image SHA-256, owner/generation/order checks, and
atomic staging cleanup in both children. Every successful A1 ACK keeps mapping,
descriptor, code, and kernarg addresses zero. It does not claim PT_LOAD mapping,
relocation application, dynamic AQL/kernarg construction, queue submission,
gem5 code-object execution, hardware presence, timing, or performance. Its
historical next action was the scoped `P3-CODEOBJ-03-A2` loader gate; that gate
remains an unproven prerequisite for later code-object execution, while CP-0014
makes `P3-HOST-NATIVE-02` the current unique action.

`CP-0014` is accepted as the source-grounded host-native architecture and
dependency inventory. EV-0036 and EV-0037 bind reusable GPU/Vega/HSA and host
bridge surfaces, x86/Process/TLB blockers, and the standalone runtime ABI;
gem5 boundary tests (4/4), the runtime matrix (16/16), and a focused Clang
ASAN boundary test pass. This checkpoint makes no host-native execution,
Triton end-to-end, hardware, timing, or performance claim. Its recorded
`P3-HOST-NATIVE-02` action is completed by CP-0015 below; the standalone
control core remains a control-plane self-test boundary rather than a
code-object runner.

`CP-0015` is accepted at the `P3-HOST-NATIVE-02` control-core/build boundary.
The gem5 target builds with `BUILD_ISA=n`, `USE_X86_ISA=n`, and `BUILD_GPU=y`;
the symbol/DSO audit passes, the protocol/memory/queue/signal self-tests pass,
and the eight legacy host-state regression binaries remain green. It does not
claim PT_LOAD mapping, code-object execution, Triton E2E, hardware, timing, or
performance. The next unique action is `P3-HOST-NATIVE-03`: connect the
existing runtime protocol and CP13 loader to this target and prove pinned
gfx950 loader/dispatch parity.

`CP-0016` is accepted at the first host-native functional-parity boundary. The
new `HostNativeMemoryContext` reuses `HostGPUMemoryState` and the pinned
dispatch state machine for GPU-VA range checks, functional copies, page leases,
ownership authorization, and cleanup. Its standalone gfx950 fixture target
passes the XOR oracle and negative/lifetime checks with `USE_X86_ISA=n`; the
runtime probe recognizes two pinned gfx950 HSACO metadata records. This still
does not map PT_LOAD segments, construct dynamic AQL/kernarg, connect
HSAPacketProcessor/GPUDispatcher/ComputeUnit, execute an ISA kernel, or pass
Triton. Triton end-to-end remains 0/1. The next unique action is
`P3-HOST-NATIVE-03-A`: implement the bounded loader/translation/AQL adapter and
compare it against the existing gem5 front-end before attempting the Triton
launcher.

`CP-0017` is accepted at the bounded host-native PT_LOAD staging boundary. The
standalone `BUILD_ISA=n; USE_X86_ISA=n; BUILD_GPU=y` target stages the locked
`gpuReadWrite` gfx950 image, validates exact PT_LOAD tuples, copies `filesz`,
zero-fills `memsz-filesz`, binds the descriptor/entry relation, and preserves
allocation leases across Busy/unmap cases. Its negative cases are
pre-allocation manifest/image rejection; the mapper is fixture-scoped and does
not enforce segment permissions, apply relocations, build AQL/kernarg, submit a
queue, or execute an instruction. Triton remains 0/1 and Qwen inference remains
0/1. The next unique action is `P3-HOST-NATIVE-03-B`: native translation plus
dynamic AQL/kernarg parity on the reused GPU pipeline.

`CP-0018` is accepted at the host-native dispatch-admission B0 boundary. The
no-x86 `host_gpu_native_dispatch_admission_core` reuses the CP17 staged image and
host memory/queue state to load the descriptor, verify its entry relation, pack and
read back the 280-byte hidden kernarg, materialize and round-trip a 64-byte AQL
packet, construct an `HSAQueueEntry`, and exercise queue-control plus ordered
lifecycle-listener contract checks. The legacy dispatcher object recompiles with
the shared listener symbols. This is admission and control-state evidence only:
no HSA queue publication, HSAPP/GPUCommandProcessor/GPUDispatcher/ComputeUnit
instantiation, AQL submission, instruction fetch/retirement, or GPU output
differential is claimed. Triton remains 0/1 and Qwen inference remains 0/1; the
static 15-contract Qwen gate is valid but strict AMD execution remains blocked.
The next unique action is `P3-HOST-NATIVE-03-B1` for native address translation
and real HSAPP/command-processor packet publication, retaining the no-CU boundary.

`CP-0019` is accepted at the host-native queue/command-processor-core B1
boundary. The no-x86 `host_gpu_native_b1_core` resolves the CP17/CP18
host-owned GPU virtual addresses, registers a 64-slot queue, publishes and
rings one 64-byte AQL packet, fetches it in order, reads the locked descriptor,
MQD, kernarg, and completion-signal object, and reuses `HSAQueueEntry` for one
native command-processor-core admission. Its `aql_submitted=true` flag is
scoped to the extracted native core: legacy HSAPP and GPUCommandProcessor
SimObjects are neither linked nor instantiated. GPUDispatcher/ComputeUnit
connection, host read-index update, packet retirement, signal decrement,
instruction fetch/retirement, ISA execution, and kernel output remain false.
The CP15-CP18 checkers and eight VEGA_X86 regressions remain green. Triton E2E
and Qwen inference remain 0/1; the downloaded model and static 15-contract gate
remain ready. The historical next action was `P3-HOST-NATIVE-03-B2` for a
no-x86 GPUDispatcher/CU output/trace differential; CP-0020 accepted that
boundary and CP-0021 below records the separate Triton provenance prerequisite.

`CP-0020` is accepted at the bounded no-x86 B2 execution boundary for one
locked `gpuReadWrite` fixture. Its four workgroups, sixteen wave64 waves,
instruction-start trace, output oracle, queue retirement, signal completion,
and pin release do not establish generic gfx950 or arbitrary-HSACO support.

`CP-0021` accepts only the child-side Triton vecadd compile/provenance
boundary. The unmodified pinned tutorial and retained 5,408-byte HSACO hashes
match; the runtime accepts the exact `amdgcn-amd-amdhsa-unknown-gfx950` target
spelling and DEFAULT-visible `vecadd.kd` only in a metadata-only branch. The
descriptor preload is 12 DWORD (48 bytes), runtime CTest is 16/16 with focused
code-object tests 4/4, and caller-local code/kernarg preparation succeeds.
`compiler_invoked`, `jit`, `launcher`, `transport`, `execution`, and `fallback`
are all false; Triton E2E and Qwen inference remain `0/1`. Public A1 mapping,
descriptor, code, and kernarg VAs remain zero and fixture-only.

`CP-0022` accepts the independent generic payload-v2 codec boundary. It keeps
the v1 80-byte framing and records byte-identical, adds opt-in capability bit 8
and message types 18/19/20, and validates owner-scoped MAP, ALLOC_KERNARG,
SUBMIT_AQL, and UNMAP records with canonical zero padding and daemon-issued GPU
VA response fields.

`CP-0023` accepts the next adapter/client/admission boundary. The standalone
runtime keeps v1 behavior and passes CTest 18/18 while exercising its public v2
client lifecycle against a mock transport. The gem5 protocol suite passes
47/47 normally and 47/47 under ASAN/UBSAN. A separate no-x86 native selftest
performs owner-bound MAP, ALLOC_KERNARG, kernarg publication, AQL queue
publication/fetch, command-processor admission, retirement, and UNMAP, including
cancel-fetch rollback after a rejected CP admission. This remains local adapter
state: the daemon does not advertise capability bit 8 or route MessageType 18,
and GPUDispatcher, ComputeUnit, kernel execution, normal Triton launcher,
compiler/JIT, and fallback remain false. The retained Triton 12-DWORD (48-byte)
descriptor preload still fails closed as NOT_SUPPORTED. The next unique action
was CP-0024 / `P5-TRITON-VECADD-03-DAEMON-ROUTE`.

`CP-0024` accepts a bounded partial daemon-handler boundary. Gem5 now contains
an owner-bound MessageType 18 handler, MessageType 19 response plumbing, and a
shared route-policy harness. The runtime adds an opt-in endpoint probe; against
the live VEGA_X86 listener it observes the canonical unsupported-capability
handshake and then reconnects successfully with the baseline capability set.
That negative handshake does not send MessageType 18. Capability bit 8 remains
unadvertised, a positive socket route and daemon H2D publication are unproven,
ALLOC accepts only page-backed 4096/65536 alignment while the normal logical
alignment is 8, and the page-size policy is fixed for the owner session.
SUBMIT ACK, MessageType 20 completion, launcher, compiler/JIT, GPU execution,
and fallback remain false. The next unique action is CP-0025 /
`P5-TRITON-VECADD-04-DAEMON-LIFECYCLE`.

`CP-0025` accepts the bounded positive generic daemon control lifecycle. The
current daemon advertises capability bit 8 with all dependencies, and a fresh
two-generation runtime-to-gem5 run completes MAP, logical-alignment-8 ALLOC
over hidden page backing, existing v1 `MEMORY_COPY_H2D`, daemon-built AQL
admission, a durable MessageType 19 ACK, MessageType 20 retirement, UNMAP,
disconnect cleanup, and reconnect. Packet CRC is nonzero and the recorded
ticks are nonzero and nondecreasing. This proves native control-processor
admission and retirement only: GPUDispatcher, ComputeUnit, kernel execution,
output correctness, normal Triton launcher, compiler/JIT, fallback, and Qwen
remain false at the CP25 boundary. Its historical next action was CP-0026 /
`P5-TRITON-VECADD-05-GPU-EXECUTION`, which is accepted below; preload-aware
Triton launcher work remains separate.

`CP-0026` closes the locked generic execution extension, without changing the
meaning of bit 8. Bit 8 remains the owner-bound control/admission/retire
contract; bit 9 (`GENERIC_EXECUTION_V2`, word 0 bit 9, wire byte 1 bit 1) is
selected only with bit 8 and all of its topology, queue, memory, signal, and
code-object dependencies. A fresh daemon-socket run sends the exact 5,528-byte
gfx950 `gpuReadWrite` image (SHA-256
`7b6a4d2bb7f9c4e7466bcf69f3110ecbfab54d07abd4c70b6bd96b6a6fb9de56`) through
the real `GPUDispatcher`/`ComputeUnit` path: four 256-item workgroups, sixteen
wave64 waves, 304 instruction starts, exact A/B/C output oracle, durable type 20,
three D2H checks including a duplicate read, and UNMAP. The daemon's fsynced
trace is the authority for dispatch/CU/execution and quarantine; the runtime
client is the authority only for bytes actually delivered to it. The positive
output CRC is `0x6f67026f` (A and C `0x4705cdab`, B `0xb28d0486`).

The live route is the `VEGA_X86` bridge configuration; this does not promote the
separate CP-0020 no-x86 functional fixture into a no-x86 daemon claim. The
fixture is intentionally exact (`Gem5IsaSupported=false` while
`LockedGpuReadWriteExecutionSupported=true`), so generic gfx950 or arbitrary
HSACO support remains unproven. The wire `signal_value_bits` remains the
expected value `1` and no wire after-value is observed; trace `1 -> 0` is the
private native AQL completion signal. A separate post-ACK disconnect sample
proves daemon quarantine cleanup with no type-20, D2H, UNMAP, or client output;
normal completed disconnect cleanup and pre-SUBMIT live-lease cleanup remain
separate boundaries. The historical CP-0021 5,408-byte Triton image remains
compile-only evidence. CP-0027 separately accepts one direct OpenCL executable and one
exact 5,160-byte `vecadd` image: the executable manages gem5, reaches
GPUDispatcher/CU, receives C-only D2H, validates bit-exact `C=A+B`, and exits
cleanly without fallback. Reusable multi-dispatch and arbitrary OpenCL remain
false. CP-0028 accepts one exact 5,384-byte pure-b010 Triton `add_kernel`
image through the normal Python driver/JIT/launcher path, including two
same-process launches with session/resource reuse and bit-exact output. This
is still only contiguous float32 vecadd; broadcast, cast, reduction, norm,
GEMM, RoPE, cache, GDN, attention, full-model, TP, and CCL execution remain
false. Its v6/v7 final-prefix attempts are retained only as NON-PASSING
evidence. CP-0029 stabilizes the queue test and accepts the fresh schema-8
repository-local prefix. CP-0030 accepts the first model-required subgate:
the exact BF16 SiluAndMul image executes normal-Python decode and masked-prefill
shapes twice with exact PC/lifecycle evidence, zero mismatch, zero nonfinite
values, and zero fallback while CP26/CP27/CP29 regressions remain green. This is
only the activation/multiply part of `mlp.gate_up_silu_down`; gate/up and down
GEMM, every complete model contract (still 0/15), a complete layer, and the
model remain unaccepted. P5-OPS-01 expands the operator matrix next. Profiling
is deferred until a real workload demonstrates a material operator-, layer-,
or model-level bottleneck.

The current direct-runner boundary has advanced beyond that historical CP30
handoff without widening CP30 itself. Ordinary Python now executes the pinned
checkpoint's complete 24-layer, empty-cache, first-token text backbone through
271 generic Triton dispatches and the tied 248,320-row LM head through 61
streamed dispatches. Every dispatch retires durably and the sessions clean up;
the target local FP32 oracle reports zero LM-head mismatch, all values are
finite, the full tied-weight digest matches, and fallback counters are zero.
Against the independently frozen RTX 5090 golden, the complete logits use the
explicit `bf16_cross_arch_v1` policy: cosine similarity `0.9806992579600918`,
relative-L2 `0.20649629542929238`, top-20 token overlap `18/20`, and exact unique
greedy token `266`. This is a model-level first-token logits proof, not stable
inference acceptance. The older frozen-input trajectory remains retained as an
accumulated-drift diagnostic; its replacement live-AMD-input differential now
passes all 24 decoder layers with zero pointwise mismatch and nonfinite values.
Final norm still needs the same online gate, and uninterrupted production,
multi-token cache evolution, CCL, and TP are still required. The formal
integration package now exists under
`plugins/framework/gemsim_vllm`, is installed only in the private schema-8
environment, and is discovered by pinned vLLM through both platform and
general entry points. It registers ten `torch.library` operations plus formal
OOT linear, RoPE/MRoPE, Gemma RMSNorm, SiluAndMul, RMSNormGated, GDN decode,
full-attention, and model adapters through supported dispatcher,
`CustomOp.register_oot`, `PluggableLayer.register_oot`, attention-backend, and
`ModelRegistry` APIs. The bounded actual vLLM attention path performs NHD KV
update and GQA decode in two GemSim dispatches with exact output/cache results;
the actual GDN adapter performs its eight-dispatch decode path with exact state
routing and zero fallback. The private framework installer is reproducible and
keeps the pinned vLLM checkout untouched; it installs a pure-Python `empty`
vLLM wheel and CPU-only dependencies in the private prefix. The separately
registered text-only Qwen3.5 architecture now constructs and executes all 24
decoder layers with the formal replacements and no source patch. The official
weight loader accepts exactly the 320 text-model tensors and rejects unrelated
checkpoint namespaces. With bounded, explicitly bound one-token metadata, the
registered model completes 278 generic Triton dispatches, updates all 18 GDN
and 6 NHD attention-cache states, and produces the exact direct-runner final
BF16 hidden digest
`a75caaff06236a54e397bc4025f4329c471e9f609c7a84a74e37b1247b2bf084`
with zero mismatch, non-finite result, or fallback. The OOT model preserves the
checkpoint's FP32 GDN output-norm parameters while the surrounding model uses
BF16, and a load-time hash gate covers that dtype boundary. The formal path now
also supports one-request, empty-cache causal prefill for 2..16 tokens. A
checkpoint-backed two-token gate through layers 0..3 completes 141 dispatches
and exactly matches serial decode for both returned BF16 rows, all three GDN
conv/recurrent states, and the first full-attention NHD KV cache, with zero
mismatch, non-finite result, or fallback. This remains a bounded manually
scheduled differential gate. The same registered model now also executes the
complete 24-layer two-token prefill in exactly 278 dispatches: all 24 cache
records are finite and mutated, every dispatch reaches type20 durability, and
the terminal session completes without quarantine. It does not yet pass the
frozen independent-NVIDIA pointwise prefix gate: final hidden relative-L2 is
`0.10932212645964957` with cosine `0.9940072673597194`, but 318 elements exceed
the predeclared `0.25 + 0.15 * abs(reference)` bound. The first observed state
pointwise failure was layer 16. The newer online protocol supplies each NVIDIA
layer with the actual live AMD input rather than a frozen upstream trajectory;
all 24 layers, including layer 16 GDN and layer 23 full attention, pass their
hidden/residual/state, cache-mutation, alias, finite, pointwise, relative-L2,
and cosine gates with zero fallback. This proves decoder-layer intrinsic
correctness for that diagnostic two-token path without converting checkpoint
replay into acceptance. Final norm, one uninterrupted empty-cache production
run, cache-preserving multi-token execution, and full-logit/greedy model-output
gates remain next; worker scheduling, batching, CCL, and TP remain unaccepted.
Standalone CCL unit/multi-process gates remain mandatory before any vLLM TP
binding.

## 2026-08-16 live-model priority checkpoint

The immediate goal is to make the real unchanged-upstream model progress,
not to manufacture another evidence matrix. The active order is:

1. Finish the current Qwen3.5-0.8B SGLang TP1 run and repair each actual
   shared-layer traceback with a generic fix.
2. Repeat the same path at vLLM/SGLang TP2 and TP4, keeping framework source
   and the runtime-gem5 bridge unchanged for framework-specific reasons.
3. Finish the simulator-aware 16-slot `rocm-smi`/`gemsim-smi` facade: a live,
   identity-checked managed gem5 lease is `ON`; absent, stale, or mismatched
   identity is `OFF`; no KFD/DRM node is opened.
4. Attempt Qwen3.5-9B vLLM TP16 with the upstream AMD attention selection and
   upstream `torch.compile` path only after the smaller TP4 path is usable.

The first real model failure exposed a missing ROCr scratch handshake, a
signed FLAT scratch offset bug, and a group-local scratch alias. These are now
fixed at the queue/gem5 semantic layer and verified by an unchanged upstream
AITER spill probe with two concurrent workgroups and bitwise output. The
current static mapping is intentionally limited to gfx950 wave64 full-small
main scratch on the one-CU host profile; reduced/large/alternate scratch still
needs a generic scratch-scoreboard lease allocator before it can be claimed.

Progress is measured by reaching the next real model boundary: startup,
weight load, cache allocation, first forward, first collective, generation,
and clean teardown. Broad matrices and formal publication are deferred while
the model is below that boundary. A full 0.8B load currently takes about 27
minutes; a diagnostic iteration is normally 30--45 minutes, so TP4/TP16 timing
remains contingent on the actual shared-layer failures.

Current live checkpoint: the repaired TP1 path completed weight loading,
Mamba/KV allocation, engine startup, bounded Triton correctness autotune, and
full-model dispatch through ticket 1115 without the former V_PACK, control-flow,
or scratch failures.  The next active packet exposed the shared gfx950 CDNA4
`DS_READ_*_TR_*` boundary: all four transpose-read constructors need the common
local-memory classification, and the existing B6/B16 data-repacking semantics
must be corrected as one instruction family.  The immediate gate is a
family-wide architecture-derived fix with asymmetric golden vectors, followed
by a matched runtime/gem5 rebuild and the same unchanged SGLang TP1 rerun.  A
flag-only patch, kernel-PC exception, or model/operator branch is not accepted.
The local demo still uses the official `watchdog_timeout` ServerArgs field with
a 24-hour simulator default; no SGLang source change is permitted.

Before TP2/TP4, make the product's HSA/KMT topology genuinely configurable
from 1 through 16 and default the simulator product to 16 logical GPUs.  This
must be one coherent runtime-gem5 capability covering GPU IDs, apertures, VM,
memory policy, scratch, and queue node ownership.  SMI's 16 slots remain an
observation facade: an available but idle logical GPU is OFF until a validated
managed gem5 lease is live.

The immediate TP1 bring-up uses the repository-owned `gemsim_hip` OOT Triton
driver policy. It must preserve the upstream `hip/gfx950` target and stock AMD
compiler, run every autotune candidate once with synchronization, and make no
performance claim. This avoids millisecond-budget autotune loops whose launch
counts explode under cycle-level simulation. The separate correctness goal is
to implement KMT clock counters and HIP event start/end timestamps in one gem5
simulated-nanosecond domain. Host wall time is not an acceptable device timing
substitute, and no framework- or kernel-specific branch may be added for this
policy.

## 2026-08-16 model-key pause checkpoint

The active blank-context checkpoint is `CHECKPOINT.md`. The current source
contains the generic CDNA4 DS transpose family repair and the gfx950 six-op
BF16/F16/I8 MFMA decoder-family repair. Focused decoder tests pass 11/11 and a
matching full gem5 binary was linked. The latest real TP1 run completed weight
load, Mamba/KV allocation, and memory-pool initialization through durable
ticket 1006 with no fatal, then was deliberately stopped for the user's model
key change. It has not yet crossed the previous ticket-1115 MFMA failure, so
TP1 generation remains the immediate goal and no MFMA runtime acceptance is
claimed. Resume exactly from `CHECKPOINT.md`; preserve all genericity rules.
