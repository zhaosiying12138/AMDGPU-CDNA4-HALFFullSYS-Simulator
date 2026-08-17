# amdgpu-sim implementation plan

**Plan ID:** `AMDGPU-SIM-V1`

**Revision:** `16`

**Revision date:** `2026-08-17`

**State at this commit:** `AgentENV and generic model-backed fast copy are integrated on main while later gem5 fixes remain preserved. The custom 6.18.40.1 kernel is active with usable KVM and ublk devices. Model acceptance is reset to unchanged-upstream SGLang TP1 and vLLM TP1, then both TP2, then Qwen3.5-9B TP16; 0.8B TP4 is cancelled and custom engine operators do not count.`

## 0.0 Authoritative resumed plan (2026-08-17)

This revision overrides the CP-0029 continuation text and all older 0.8B TP4
steps. Execution is resumed on `main` in this order:

1. Integrate and verify the pinned AgentENV service, deterministic runtime
   bundle, guest bootstrap, pair manager, custom WSL kernel tooling, and
   rollback records without reverting fast-copy or later gem5 commits.
2. Commit or explicitly reject every inherited dirty change after review.
   Keep AgentENV, fastcopy, and Claude-era generic gem5/self-runtime fixes as
   separate auditable commits. End each handoff with root and nested trees
   clean.
3. The user completed the required restart. The running custom 6.18.40.1
   kernel, `/dev/ublk-control`, and `/dev/kvm` are verified. Complete the
   Firecracker, loopback-only AgentENV service ownership, and sandbox
   create/collect/stop gates without requesting another restart. If that is
   not possible in the current session, switch to host TP1 execution instead
   of blocking: run lanes concurrently only when their resources are proven
   disjoint, otherwise run them serially.
4. Build one immutable runtime bundle from the active main product. Every
   sandbox gets a private worktree/branch, build directory, Triton/FLA cache,
   endpoint/socket, SMI lease namespace, logs, tmp, and process group. Host
   process guards refuse accidental use of unrelated gem5/vLLM/SGLang jobs.
5. Enable the generic copy path with `source scripts/fastcopy_mode.sh fast` for
   model bring-up. Run the existing byte-exact probes first. Keep
   `source scripts/fastcopy_mode.sh legacy` as the explicit fallback and use a
   single bounded legacy/fast A/B to validate loaded bytes, model outputs, and
   weight-load speedup; do not duplicate every long run.
6. Bring up pinned unchanged-upstream SGLang TP1 and vLLM TP1. Each TP1
   sandbox is limited to exactly one live gem5 process. Project-specific vLLM
   Triton registrations and modified framework/model paths are diagnostic
   history only and cannot satisfy this gate.
7. The two TP1 lanes may run concurrently only in distinct AgentENV instances
   and branches. A discovered self-runtime/gem5 fix is reviewed for both
   engines, tested narrowly, committed on its lane, and merged to main one at a
   time. The peer lane rebases to the new main before resuming.
8. After both TP1 paths pass generation and clean teardown, run unchanged
   SGLang TP2 and vLLM TP2. Cancel all Qwen3.5-0.8B TP4 work. When both TP2
   gates pass, continue directly to Qwen3.5-9B TP16 on the unchanged upstream
   AMD path, including upstream `torch.compile` where required.
9. Replace repeated full-model-only debugging with a tiered loop. Every long
   run has one read-only monitor and uses `gpt-5.6-sol` subagents at `max`
   reasoning only for bounded work that can materially improve the next gate;
   do not create busywork to occupy slots or spend tokens. Freeze a canonical
   failure capsule first; validate decoder and
   pure semantics in seconds; replay an isolated dispatch only when its code,
   packet, resources, allocation generations, memory inputs, queue/MQD state,
   and synchronization dependencies are complete; otherwise replay the
   smallest proven predecessor sequence. Use gdb/core data for diagnosis, not
   as a substitute for execution. Re-run full TP1 only after the focused gate
   proves the proposed repair and all artifacts for the next failure capture
   are already prepared. Measure kernel-launch overhead separately and optimize
   only a generic, demonstrated bottleneck without weakening dispatch, memory,
   ordering, completion, profiling, or failure semantics.

AgentENV isolation is not model correctness, fast copy is not model
correctness, and a custom-operator vLLM probe is not upstream-engine
acceptance. Each claim retains its own evidence boundary.

## 0.0.1 Execution-efficiency mandate (2026-08-17T22)

Standing user requirement. It constrains *how* every step below is executed and
overrides conflicting scheduling language; it relaxes no correctness, evidence,
or upstream-zero-diff rule. The normative statement is GOAL.md section
"2026-08-17T22 execution-efficiency mandate".

The tiered debug loop in item 9 above is amended as follows.

1. **Identity gate before execution.** A run records the resolved ROCr, HIP,
   model DSO and gem5 identity (path, symlink target, SHA-256) as the first
   lines of its own log, and asserts that they descend from the commit under
   test. Two concrete traps are now known and must be checked explicitly:
   assigning `LD_PRELOAD=<diagnostic>.so` silently displaces the product ROCr
   preload that carries the fix under test, and the content-addressed
   `env/rocm/product-v1-*` prefix named by the conda activate script is not
   rebuilt when `projects/rocm-systems` advances. A run that fails this gate is
   void as evidence in either direction.
2. **Capsule-first repair.** Every memory, copy, dispatch or ISA repair is first
   reproduced and then accepted on the cheapest artifact that exhibits the same
   failure shape — preferably a public-HIP program of a few seconds. Full-model
   runs are spent only on the acceptance gate, never on hypothesis search.
   A capsule must additionally assert that the intended path was actually taken
   (for fast copy: at least one `Fast copy success` and zero blit dispatch
   packets), so a silent fallback cannot masquerade as a pass.
3. **Concurrent bounded work is mandatory during long runs**, using `opus-5`
   subagents with 1M context at `max` reasoning, on: failure capsules, bridge/AQL
   replay, post-weight-load snapshot reuse, launch-overhead measurement, AgentENV
   closure, and the peer engine lane. Findings enter the next iteration only
   after review and never mutate binaries or source under a live run.
4. **One `gem5.opt` per TP1 lane**, with orphan reaping before each launch.
5. **Acceleration backlog**, each generic and semantics-preserving: the capsule
   ladder (done for the copy family), bridge-level record and point-in-time
   replay, post-weight-load device-memory snapshot reuse, and generic
   kernel-launch overhead reduction in gem5.
6. **vLLM parity requirement.** The vLLM lane is re-scoped to unchanged upstream
   with no project Triton operator registration, matching the SGLang lane. The
   earlier operator-registration result is retained as diagnostic history only.
7. **Model/TP pinning.** TP1 and TP2 are Qwen3.5-0.8B lanes; TP16 is a
   Qwen3.5-9B lane. No other pairing is in scope, and 0.8B TP4 is cancelled.
8. **AgentENV lane contract (2026-08-18).** AgentENV is experimental research
   whose required gate is TP1 only; host execution with namespace isolation
   proceeds in parallel and is not blocked on it. The template bakes gem5,
   self-runtime and the product; the Python stack is installed inside the
   sandbox over the host proxy and snapshotted for reuse. Three enablement
   defects are fixed and must not regress: the base image comes from
   `quay.io` because `docker.io` is unreachable, `/dev/ublkc*` needs group
   access through a udev rule because ambient capabilities do not bypass DAC on
   a device node, and `AENV_HOME_PATH` must be published as a short symlink
   because firecracker's API socket otherwise exceeds `sockaddr_un.sun_path`.
   Qwen3.5-9B TP16 inside a sandbox is a stretch demo; simulated VRAM is not the
   constraint because device memory is a sparse memfd, so scale it by measured
   per-rank runtime cost and add swap if required.

## 0.1 Execution checkpoint (2026-08-14)

This checkpoint records the executable boundary today and prevents an
intermediate result from being reported as a framework or model acceptance.

| Area | Current state | Claim allowed now | Next gate |
| --- | --- | --- | --- |
| KMD replacement | Unchanged upstream ROCr/libhsakmt/HIP/COMGR/RCCL; one project model DSO and one runtime-gem5 bridge own translation; the process memfd is exported once and authenticated KMT allocations map GPU VAs to owner-bound backing offsets | formally accepted upstream ROCr QueueCreate/QueueDoorbell/QueueDestroy transport, KFD lifecycle, exact peer-PID memory authorization, bridge layering, KMT memory ownership, and three live CU retirements | promote the accepted ABI into the immutable HIP/PyTorch product facade |
| Standard HIP queue | ROCr creates a standard AQL queue; the model DSO sends monotonic doorbell notifications; gem5 observes live ring batches through `HostNativeQueueCore`; `ResidentKernelView` admits descriptors from either KMT allocations or the explicit mapper through one `GPUNativeMemory` command processor | independently verified ROCr device-blit/user-kernel/device-blit and public-HIP allocator/copy/stream/module/kernel/synchronize chains, durable trace, clean session exit, external oracle, and zero fallback | complete event/error negatives, then exercise official ROCm PyTorch without adding operator routes |
| Simulator status | 16 lease slots with owner/start-time/digest checks; real managed-gem5 OFF -> ON -> OFF probe passes | simulator-aware `rocm-smi` inventory | retain as regression gate |
| Generic CCL | Protocol/device matrices cover logical N=2..16; formal live N=2/3/4/8/16 | standalone device SUM and communicator contracts | model sharding/output gates at TP=2, then TP=4 |
| Frameworks | Pinned Triton/PyTorch/vLLM/SGLang trees remain upstream; the immutable product retains official AMD HIP/COMGR/RCCL and replaces only ROCr/KMD | formal public-HIP, PyTorch eager, unchanged-upstream Triton AMD multi-op, and one inherited vLLM RowParallelLinear TP2 layer with exact local/device-collective evidence | full HIP-backed vLLM TP2/4, then unchanged SGLang TP4 |
| Models | RowParallel TP2 prerequisite accepted; no end-to-end Qwen TP acceptance; 9B compile lane not started | operator/layer evidence including the real upstream RowParallel boundary | 0.8B full TP2 -> TP4, then 9B TP16 with upstream `torch.compile` |

## 0.2 Product/provider continuation (2026-08-15)

The next product generation is an official ROCm 7.2.3 user-space closure, not a
mix of the existing ROCm 7.13 wheel product and a second HSA implementation.
The signed AMD Jammy repository is locked in
`config/rocm-deb-sysroot-7.2.3-jammy-amd64.json`: 63 packages, exact signed
`InRelease`/`Packages.gz` provenance, and explicit replacement of `hsa-rocr`
with the native product's ROCr boundary. The materialized probe has no
`libhsa-runtime64` or `libhsakmt` and exposes the 7.2.3 HIP/COMGR/RCCL ABI
(`librocsolver.so.0`, `librccl.so.1`, and related libraries) under one private
sysroot. Its native ROCr provider remains selected only through the existing
`LD_PRELOAD` boundary.

The official ROCm vLLM wheel index is resolved with
`tools/python_wheel_lock.py` into
`config/rocm-pytorch-vllm-gfx950-7.2.3.json` (215 CPython 3.12 manylinux wheels,
3,394,870,865 bytes). The lock records file-hosted PyPI URLs or the pinned
ROCm index only, rehashes every wheel and its dist-info metadata, and is
installed offline by the product builder. No framework source is patched by
this product step. Product acceptance remains blocked until the new prefix
passes provider ELF/source identity, torch HIP import, upstream Triton launch,
and the existing bridge/SMI regression suite.

The simulator management contract is fixed at 16 logical slots. `rocm-smi`
and `gemsim-smi` read only authenticated lease records and procfs: a live
managed gem5 instance is `ON`; missing, stale, or identity-mismatched leases
are `OFF`. This is a regression prerequisite for every TP lane, not a model
specific implementation. Model work starts only after this product is frozen:
unmodified upstream vLLM Qwen3.5-0.8B TP=2, then TP=4; unmodified upstream
SGLang TP=4; and the separate Qwen3.5-9B TP=16 upstream `torch.compile` scale
lane. AMD attention selection remains upstream-first and does not require
FlashInfer.

The process backing, owner-scoped GPU-VA translation, monotonic doorbell
notification, batched live-ring observation, loader-independent resident
kernel admission, shared asynchronous execution lifecycle, signal completion,
queue retirement, and KMT resource-pin release are complete. The exact path is
frozen in `artifacts/evidence/upstream-rocr-aql-v2-accepted`: result SHA-256
`abc191a38d309be2cca8ed444e08ec0e2b0042e1d094c7c3b5ab636ad5fe48f6`,
manifest SHA-256
`299d50c751c82d96dfece1e07f8a26b40e9f14dfff319c565a5ebe8b7cf2ad24`,
three kernel execution tickets, one standard control packet, and
`host_fallback_count=0`. A second accepted bundle,
`artifacts/evidence/hip-facade-runtime-v1-accepted`, records result SHA-256
`7f1e26613f0f60a7cba0dff94d80af204437e89a29aedb060f46bce74bd04e71`
and manifest SHA-256
`fe81e5e5732333d14c9112df96c382c615b53bb7317cd58fc10d22f24b053a05`
for one ordinary public-HIP vector-add with bit-exact output, one native retire,
one session-complete record, and zero fallback. The current official ROCm
PyTorch product is accepted at the next bounded rung in
`artifacts/evidence/rocm-pytorch-eager-multiop-wavefront-fix-v2-accepted`:
result SHA-256
`af0c079031daf2f001ac88e1a33c98268e862c78a2e46c81b8187c62e5330b7e`,
manifest SHA-256
`b185237375c730cadf19a628a31b364ee4977dd023ce7d9d8a169387d53d4b17`,
eight standard AQL retirements, bitwise copy/add/sigmoid/sum, fresh non-aliasing
outputs, natural process cleanup, and zero host fallback. The same product has
also completed a direct unchanged-upstream Triton AMD live smoke: the official
`triton.backends.amd.driver.HIPDriver` launches add, masked transform, and
reduction kernels; all outputs are bitwise exact and 12 native dispatches plus
one clean session-complete record retire. The only required fix was generic
wavefront accounting in gem5: an empty-lane vector memory instruction now
releases the same GM/LM reservation it acquired at issue. The bridge and all
upstream framework trees remain unchanged for that workload. The absent-only
runner and independent verifier freeze the result in
`artifacts/evidence/upstream-triton-amd-multiop-v2-accepted`: result SHA-256
`8437bd53841ff9ea44656d2ab331b768e75d80ec45675c9150b9bc71d865f842`,
manifest SHA-256
`a6ef2f15d3cf23fee3319e02c1e5d6c3644339b5e9cab3320efb69cf45a50dfe`,
28 fresh private JIT-cache files, three distinct HSACO identities, 12 unique
native execution tickets, one clean session-complete record, and zero fallback.
The lifecycle remains shared
by explicit and KMT-originated tasks; it may not copy `GenericDispatchSession`
into a second KMT state machine. No
bridge or simulator source may branch on an operator, model, tensor role,
framework, code hash, or fixed packet/PC.

The resident-kernel CP refactor is the first of the two bounded shared-layer
fixes allowed for this AQL slice. The shared execution/retirement lifecycle is
the second. If it does not converge within the bounded slice, preserve the
facade/SMI/resident-admission work and return to the prior accepted
explicit-dispatch plan rather than adding a special case.

## 0.3 Model-first execution mode (2026-08-15)

The next work slice prioritizes real model bring-up over additional evidence
packaging. Existing accepted products and bundles remain immutable regression
inputs; new cache inventories, identity refinements, and publication bundles
are deferred unless a model failure requires one. Each iteration uses a short
temporary run and records the actual traceback, selected backend, rank
topology, and first shared-layer boundary that failed.

1. Run unchanged-upstream vLLM Qwen3.5-0.8B at TP1 to expose model/operator
   failures, then the same path at TP2 and TP4 with independent gem5 ranks.
2. Run unchanged-upstream SGLang Qwen3.5-0.8B at TP4 through its native ROCm
   path.
3. After TP4 starts, attempt Qwen3.5-9B vLLM TP16 with upstream
   `torch.compile` enabled as a scale smoke.

The debugging rule is “measure the weak link first”: prioritize model startup,
weight loading, device discovery, process-group initialization, first GEMM,
first attention/cache update, first collective, and teardown. Do not spend a
cycle on another broad correctness matrix when the model has not reached that
boundary. All fixes stay at the lowest shared HIP/PyTorch/c10d/CCL/runtime-
gem5 layer; operator-, model-, shape-, and framework-specific bridge branches
are forbidden. Formal acceptance publication is postponed during this
bring-up slice; a failed run is still useful diagnostic input and does not
change any acceptance claim.

## 1. Outcome and non-negotiable invariants

The deliverable is an industrial, versioned host-side simulator stack, not a
mock demo.  A host process must be able to compile and launch real AMDGPU
HSACO/code objects and have their work executed by gem5's GPU model while the
host has no AMDGPU device and no production KMD/UMD runtime loaded.

The final anchor is:

1. Official `Qwen/Qwen3.5-0.8B`, text-only path (vision is a later extension).
2. A complete Qwen3.5-0.8B model lane: TP=2 is the prerequisite anchor and
   TP=4 is the formal model acceptance target, with one independent gem5
   daemon per rank.
3. A separate scale lane: Qwen3.5-9B at TP=16 with vLLM's upstream
   `torch.compile` path and the same no-fallback/device/lifecycle gates. This
   is one scale target, not a request to generate a TP=8/16 model matrix.
4. A framework portability lane: Qwen3.5-0.8B at TP=4 through the pinned
   upstream SGLang ROCm path, with no SGLang source edit and no
   runtime-gem5 bridge edit. The lane is accepted only if SGLang reaches the
   same generic CCL/runtime bridge through an official platform, device
   communicator, or backend extension point.
5. Complete prefill followed by a stable, predeclared multi-token greedy decode
   window; a one-token smoke alone is not model acceptance.
6. Exact selected token ID agreement with a pinned reference, plus layered
   numerical checks and trace evidence.
7. Zero host CPU arithmetic fallback and zero loads/accesses of
   `libamdhip64.so`, `libhsa-runtime64.so`, `/dev/kfd`, or `/dev/dri`.

The transport, handles, collectives, failure epochs, and rank registry are
parameterized by `world_size=N`; the model lanes must not introduce a
TP-specific protocol. CCL and logical device capacity remain 2..16. Gloo/TCPStore may carry
control metadata, rendezvous, and barriers only.  Tensor values and reductions
must travel through the GemSim transport and execute reduction kernels in
gem5.  A CPU reduction oracle is diagnostic-only and is counted; it cannot
silently satisfy an acceptance run.

### 1.1 User-facing execution contract and priority

The low-level generic endpoint accepted by CP-0026 is a regression interface,
not the product entry point. Work proceeds in this order:

The non-negotiable engineering constraints in `ENGINEERING_CONSTRAINTS.md`
apply to every step below. In particular, an operator invocation may JIT only
the operator into a persistent Triton cache. gem5 and `self-amdgpu-runtime` are
prebuilt, socket-decoupled components and must never be rebuilt as a side
effect of adding or running an operator.

1. Compile a user OpenCL `.cl` kernel and its host program against only the
   repository-local OpenCL/runtime stack. The resulting normal executable is
   the user command: it transparently starts or connects to gem5, submits,
   waits, copies results back, checks the oracle, and exits. No manual endpoint
   or hand-authored transport records are permitted in this acceptance path.
2. Run an ordinary pinned Triton Python program through Triton's normal
   backend/driver/runtime selection. Compilation, code-object load, launch,
   synchronization, and result copy must implicitly traverse our runtime and
   gem5. A C endpoint, standalone fixture runner, or application-specific
   launcher is insufficient.
3. Use the generated model operator manifest as the work queue. Every required
   operator variant must pass compile plus real gem5 execution plus a
   differential oracle with fallback counters zero before full-model claims.
4. Run the complete text model first on one simulated device for stable
   multi-token inference, then use the upstream vLLM Row/Column/attention
   contracts over independent gem5 daemons. Stabilize TP=2, expand to TP=4,
   and only then exercise the separate 9B TP=16 compile-scale lane.

The attention policy is deliberately upstream-first. On ROCm, vLLM's own
platform selector is authoritative: it may choose `ROCM_AITER_UNIFIED_ATTN`,
`ROCM_AITER_FA`, `ROCM_ATTN`, or `TRITON_ATTN` according to availability and
model constraints. The acceptance runner records the selected backend and
requires `torch.compile` to remain enabled for the 9B scale lane. The bridge
must not expose a project-specific attention op or require a bridge edit when
the selected upstream backend changes.

SGLang uses the same rule: its upstream ROCm backend selection (AITER or
Triton, according to the pinned SGLang capability checks) is authoritative.
The bridge is never changed per framework. If SGLang lacks an official
out-of-tree communicator/device hook, record that source-level gap first and
add one generic framework adapter at the lowest shared interface; do not fork
SGLang model code, copy vLLM layers, or add operator-specific runtime routes.

The selected local SGLang snapshot is the unchanged `0.5.17` CPython 3.13
wheel. Its complete unpacked identity is recorded in
`tools/sglang_source_manifest.json` and checked by
`tools/sglang_portability_audit.py`. This release already auto-detects a ROCm
PyTorch build and selects its in-tree `RocmSRTPlatform`. That unchanged path is
the formal model path. Its official `sglang.srt.platforms` entry point remains
useful for a bounded diagnostic platform, and the framework-neutral
`gemsim_ccl.torch_process_group` supplies a PyTorch third-party c10d fallback.
Neither is allowed to replace the standard HIP device identity or upstream
model/layer/group coordinator. The historical `0.5.10.post1` source snapshot
remains a negative extension-gap record only.

The SGLang wheel is ABI-incompatible with the existing CPython 3.14 vLLM
product, so SGLang receives a distinct immutable CPython 3.13 conda product.
That environment binds the same native runtime/gem5 product, Triton source,
CCL engine, and bridge identity, while leaving the accepted vLLM environment
untouched. PyPI's CUDA-oriented default dependency set is not installed
blindly; the product locks only source-grounded AMD dependencies and verifies
the actual imported files.

The c10d adapter's CPU-tensor tests are a host communication gate, not device
or model acceptance. SGLang's CPU branch can bypass c10d via
`sgl_kernel.shm_allreduce` on AMX/ARM hosts. The diagnostic OOT platform must
fail closed until the shared PyTorch HIP device facade is available; formal
model acceptance leaves `SGLANG_PLATFORM` unset and requires SGLang to select
its own `RocmSRTPlatform`. The primary facade reuses the pinned ROCr/HIP/CLR
sources and presents the normal HIP-backed `torch.cuda` contract to unchanged
PyTorch; all KMD replacement remains below it in self-runtime plus the
runtime-gem5 bridge. A PrivateUse1 facade is allowed only as an isolated
compatibility lane, not as the accepted upstream-ROCm path. Do not replace this
missing device contract with an ever-growing project `torch.library` operator
list.

`tools/framework_device_facade_manifest.json` is the machine-readable work
queue for this boundary. It separates ROCr provider, HIP runtime, PyTorch ROCm
device, upstream Triton HIP, RCCL ABI, vLLM ROCm, and SGLang ROCm capability
families. `tools/framework_device_facade_audit.py` rehashes the pinned upstream
callers and derives `model_ready`; no operator or model can override a blocked
generic prerequisite.

The formal ROCr cut reuses the unchanged upstream `libhsakmt` and its official
`HSA_MODEL_LIB` Model Interface 1.1. The project publishes one model DSO whose
three callbacks own the KMD-removal translation: a process-private sparse
memfd, KFD ioctl translation to the typed managed provider, and model DRM
translation. The frozen 124-entry ThunkLoader inventory remains an ABI audit;
it is not a second project implementation of libhsakmt and is not exported by
the model DSO. This preserves the upstream ROCr/libhsakmt public ABI and keeps
all simulation-specific behavior below it.

The first executable sub-gate is complete: unchanged Model ABI discovery
reaches managed `OPEN_KFD -> GET_VERSION -> CLOSE_KFD`, negotiates the generic
KMT capability, and retains per-provider operation sequencing. All other KFD
ioctls and DRM commands still fail atomically. Providers and managed sessions
are PID-owned; after `fork`, inherited calls fail closed and the child may only
discard its local transport copy before reconnecting, without terminating the
parent-owned gem5 process. This is lifecycle evidence, not ROCr device or HIP
acceptance. The next generic capability sequence is topology, memory/mapping
coherence, AQL queue/doorbell, signals/events, pointer info, and preloaded code
object dispatch. ROCjitsu may supply reference state-machine tests, but it must
not become a second execution backend; execution remains the existing generic
runtime-gem5 bridge and gem5 GPU adapter.

Correctness and end-to-end model coverage outrank simulator speed. The critical
path is single-device Qwen3.5-0.8B inference, then functional CCL and TP=2.
Correctness is always the first priority. A performance candidate may be
profiled, compiled, and retained while diagnosing a blocker, but it cannot be
accepted, become the default binary, trigger a full TPOT campaign, or reorder
the product roadmap until the exact affected operator/layer/model workload has
passed its numerical, state, lifecycle, and zero-fallback gates. The fixed
sequence is: find the first failing layer with live-input differential evidence;
fix it; replay the affected suffix from a validated checkpoint; pass one
uninterrupted empty-cache production run; only then run baseline/candidate
performance A/B and report TPOT. Any unexpected tensor, state, trace, simTick,
instruction, memory-traffic, or lifecycle change rejects the optimization.
Cross-cutting infrastructure that shortens many later correctness tasks is
scheduled as early as its prerequisites permit. In particular, isolated
multi-instance execution is not deferred until TP integration: once the
single-instance generic path is stable, build the supervisor, namespace, and
evidence gates so operator/layer differentials and fault regressions can run in
parallel. This work may advance ahead of a model-specific numeric fix because
it does not relax correctness and benefits every subsequent debug, profiling,
CCL, and TP task. Profile timing itself remains serial until a no-interference
calibration proves that pinned concurrent instances produce equivalent timing.
Correctness runs retain cheap wall-clock and simulated-work observations, but
profiling, host-parallel threadblocks, or memory fast paths become scheduled
work only when an actual operator, layer, model, or TP run demonstrates a
material bottleneck. Optimize only measured 80/20 causes with high expected
impact, small scope, and strong regressions. Stop or defer the optimization if
its investigation or repair cost exceeds advancing the next model blocker, if
the measured gain is not material, or if it introduces new correctness or
lifecycle risk. A standalone microbenchmark cannot displace the next missing
product operator. Any active checkpoint whose intent makes profiling an
unconditional prerequisite must be re-scoped before acceptance.

The first accepted performance intervention is deliberately narrow. It replaces
the 1024-slot owner-allocation scan in native range resolution with a checked
O(1) VA-stride slot lookup; it does not change ISA behavior, scheduling,
simulated memory timing, model math, or the user-facing runtime path. Its exact
patch and A/B binaries are retained under
`artifacts/evidence/perf-o1-range-lookup/isolation-v2/`. The formal full-window
measurement is a single causal A/B pair, supported by three short crossover
pairs; it is not presented as a multi-run confidence interval. The reported
`host-validation TPOT` includes the full prefill cost amortized over the two
decoded tokens and must not be relabelled as steady-state serving TPOT. Absolute
`simTicks` include host-paced submission gaps and are diagnostic only; the
accepted invariants are byte-identical results, equal dynamic instruction
count, normalized structural trace equality, and per-dispatch simulated
execution durations equal within one 1000-tick clock period. No further gem5
optimization is scheduled unless a new measured bottleneck has comparably high
expected impact and low correctness risk.

Every change that alters recoverable project state is an atomic progress unit:
one `Checkpoint-ID`, child commit(s) where applicable, evidence, bitlesson (if
there is a new lesson), and a final root coordinator commit.  Git cannot make
multiple repositories one physical atomic transaction, so the protocol is
explicitly two-phase and crash-recoverable (see section 8).

Every project-authored child and root commit carries exactly one
`Checkpoint-ID`, `Goal-ID`, `Plan-Revision`, `Source-Lock-SHA256`,
`Evidence-Manifest-SHA256`, `Change-Kind`, and `Baseline-Commit` trailer. The
tracked `commit-msg` hook rejects missing, duplicate, abbreviated, or malformed
identities. Existing upstream commits are never rewritten to add our trailers.

## 2. Fixed decisions

| Area | Decision | Boundary |
| --- | --- | --- |
| Target | gfx950 / MI355X semantics; wave64 | No silent gfx942 or real-device fallback |
| Simulator | gem5 GPU execution core with a host bridge and daemon | Host threads never mutate SimObjects directly |
| Runtime cut | Reuse unchanged ROCr/libhsakmt and its official Model 1.1 boundary; implement the model provider plus HIP/CLR semantics over GemSim RPC | Linux KFD and ROCjitsu are semantic/test references only, not linked or alternate execution backends |
| HIP ABI | Compatible self-built HIP/ROCr-facing ABI selected by bundle search paths | No production AMD UMD/KMD binaries or `/dev/kfd` |
| Rank topology | N daemons + supervisor, one vLLM rank per daemon | Independent VA/queue/signal namespaces; generation-checked handles |
| Collectives | RCCL-compatible semantics with GemSim functional transport first; device reduction kernels in gem5 | Host only stages/copies bytes |
| Fabric | Functional shared-memory/Unix transport first, timed xGMI/SDMA model later | No timing claims before the timed phase |
| PyTorch | Transparent HIP-compatible path is primary; optional PrivateUse1 adapter is isolated behind the same facade | No backend-specific business logic spread through vLLM |
| Framework integration | Upstream vLLM/SGLang ROCm platforms over the standard HIP-backed `torch.cuda` and RCCL/NCCL contracts; OOT hooks remain diagnostics or compatibility only | No edits to pinned torch/vLLM/SGLang and no monkey patch in an accepted path |
| Triton | Formal path reuses the unchanged upstream AMD `hip` compiler/driver; `gemsim_amd` remains a regression/migration backend | Cache keys include every toolchain/runtime/simulator revision |
| Performance | Profile only when a real OpenCL/Triton operator, layer, or model bottleneck threatens end-to-end progress; then prioritize measured 80/20 causes | Profiling is not a prerequisite to broader operator coverage; no optimization claim without retained before/after evidence and unchanged correctness |
| Host parallelism | Permit CPU-parallel threadblock simulation only with explicit dependency, barrier, atomic, and determinism gates | Host utilization never overrides simulated ordering or synchronization semantics |
| Status tooling | Provide a simulator-aware `rocm-smi` view from the daemon registry | Report simulated instances only; never probe hardware nodes or load production SMI libraries |
| Precision | Bitwise for copies/integers; explicit per-op/layer BF16/FP16 tolerances; exact final greedy token | No claim of all-float bitwise identity |
| Failure | Abort the whole job epoch, checkpoint at request/layer boundaries, deterministic replay | No elastic shrink in the anchor |
| Distribution | Root coordinator plus standard submodule gitlinks and immutable child baseline tags | We never rewrite upstream history |
| Artifacts | Weights, environments, binaries, caches and logs are ignored; scripts fetch/rebuild by fixed revision and hash | Offline execution after preparation is required |
| License | GPL-3.0-or-later for new glue/aggregate where legally possible; preserve every upstream license and notice | No license header is removed or relicensed casually |

The normative ownership and transparency design is
`docs/framework-runtime-layering.md`. Pinned vLLM, PyTorch, Triton core, and
upstream AMD lowering remain zero-diff. vLLM keeps its standard model,
parameter-loader, distributed initialization, named TP `GroupCoordinator`, and
layer semantics; project OOT plugins implement only supported extension
points. Triton keeps its frontend/JIT/compiler coordinator/cache; the OOT
`gemsim_amd` backend implements the standard driver/compiler contracts. The
backend's self-runtime/gem5 and CCL details are invisible to framework users.
This transparency does not promote the current bounded runtime into a complete
ROCm ABI implementation; HIP/OpenCL/ROCr compatibility remains a separately
tested lane.

The next architecture migration treats all current source and evidence as a
recoverable baseline, not as a frozen implementation. First retain a
content-addressed root/gem5/self-runtime source backup. Then introduce one
canonical runtime-gem5 bridge contract and a small stable gem5 GPU adapter;
migrate memory, code-object, dispatch/completion, and lifecycle families one at
a time while keeping self-runtime's public API and gem5's queue/memory/
dispatcher/CU/Vega interfaces stable. Run the complete old-versus-new
output/trace/fault/cleanup matrix after each family. Only after OpenCL, direct
Triton, framework operators, CCL, single-device model, and TP gates pass on the
new path may it become the active product. Remove the superseded fixture
routes, duplicate codecs, copied upstream state machines, and obsolete
launchers in the same migration; do not retain dead implementations for
speculative compatibility.

The executable short-cycle plan, conda entry contract, vertical migration
order, legacy-deletion gates, and mandatory fallback criteria are fixed in
`docs/runtime-gem5-bridge-migration.md`. Bridge extraction is time-bounded: if
the three minimal memory/code-object/dispatch families do not converge without
upstream or GPU-core edits, repeated correctness repair, or duplicate paths,
retain the accepted conda/facade improvements, remove incomplete bridge work,
restore the backed-up implementation, and resume P9 RowParallel immediately.

The highest-priority implementation rule is generality across operators. No
production runtime, bridge, Triton backend, or gem5 path may branch on operator
or model names, tensor shapes/roles, image hashes, expected outputs, or fixed
PC sequences. Every failure is repaired once in the shared ABI/lowering/
resource/ownership/synchronization/ISA semantic layer and gains unrelated
cross-operator tests. A replacement path must run two different retained
kernels and one previously unseen valid kernel unchanged before acceptance.

## 3. Repository and source layout

```text
PLAN.md                         complete staged plan
GOAL.md                         human-readable acceptance contract
SOURCE_LOCK.json                observed and then frozen upstream revisions
PROJECT_LANES.json              immutable project-authored baseline registry
projects/gem5/                  independent gem5 checkout
projects/rocm-systems/         ROCr, libhsakmt, HIP, CLR, RCCL source lane
projects/llvm-project/         LLVM/Clang/device-libs source lane
projects/triton/                Triton source lane
projects/pytorch/               PyTorch source lane
projects/vllm/                  vLLM source lane
projects/self-amdgpu-runtime/  our standalone host glue/runtime project
scripts/                        fetch, build, verify, model and environment tools
state/current.json              historical CP-0030 pointer, not active scheduler
state/checkpoints/              append-only historical machine checkpoints
state/bitlessons/               append-only engineering lessons
state/evidence/                 command/result manifests and hashes
docs/                           architecture, source maps, and human reports
models/ env/ build/ artifacts/  ignored material, created by scripts
```

The first three ROCm components are intentionally taken from one
`rocm-systems` revision so ROCr/HIP/CLR/RCCL interfaces cannot drift.  The
active device-libs source is `ROCm/llvm-project/amd/device-libs`; the archived
standalone Device-Libs repository is not used as the primary lane.  Nested
source repositories preserve their upstream history. The root always records
each child as a standard `.gitmodules` entry plus mode-160000 gitlink. A
pristine checkout is tagged `upstream-baseline/<lane>/<full-40-byte-sha>`; our
work branches and commits must descend from that exact object. A child has only
an `upstream` remote until a durable project fork is configured; local changes
must not advance the root gitlink before their `origin` is reproducible.

Large source lanes may use a non-shallow partial clone with a recorded
`blob:none` or `tree:0` filter and a full, non-sparse checkout. Non-shallow means
that every commit in the ancestry reachable from the locked head is locally
traversable; it does not claim that unrelated historical trees or blobs are
local. The complete locked baseline tree, plus every compatibility tree named
by the lock, must be hydrated and pass
`GIT_NO_LAZY_FETCH=1 git archive <revision>` before acceptance. This provides an
offline-buildable prepared checkout while retaining honest promisor semantics
for unused history. A later offline bundle gate must hydrate all reachable
objects and create a hashed full mirror/bundle when a fresh disconnected clone,
rather than a copied prepared checkout, is required.

`SOURCE_LOCK.json` is byte-immutable after CP-0002 and remains the authority
for upstream provenance. `PROJECT_LANES.json` separately records immutable
baselines for repositories authored by this project. Neither file owns an
evolving work head: each accepted checkpoint records the exact current
commit/tree, and the verifier requires that commit to descend from its declared
baseline. The registered workspace is the exact union of both authorities;
`.gitmodules`, mode-160000 gitlinks, absorbed Git administration directories,
and checkpoint repository records must agree on that same set.

## 4. Architecture to implement

```text
HIP/Triton/OpenCL/PyTorch/vLLM host process
              |
      self-amdgpu-runtime (stable C ABI)
              |
  ROCr core + HIP/CLR-compatible shims + libhsakmt_gem5 provider
              |
       per-rank daemon protocol (Unix control + memfd/shm payload)
              |
       supervisor: registry, epochs, N-rank collectives
              |
       gem5 HostGPUBridge SimObject / event-queue adapter
              |
 gem5 GPU execution core: AQL queues, VA, code objects, CU/shader execution
```

### 4.1 Runtime boundary

Retain ROCr's public HSA ABI, ELF/code-object loading, agent/region discovery,
queue/signal/event semantics, AQL packet construction, kernarg metadata and
allocation rules wherever possible. Replace the KFD/DRM provider, not every
upper-runtime algorithm. Upstream `libhsakmt` continues exporting the complete
symbol surface that ROCr resolves; the project implements only its official
Model Interface 1.1 callbacks. An unsupported model ioctl/DRM capability
returns the documented error and preserves caller output, never a fake
success. Preserve exact layouts and widths for queue resources, doorbells,
memory flags/maps, events, handles, 4 KiB/64 KiB alignment, and generation
checks.

The first provider implements lifecycle, topology, memory, pointer info,
queue creation/destruction/update, ring doorbell, queue masks, signals/events,
fences, and code-object-related synchronization.  Debug/trap/PC-sampling,
DMA-BUF/AIS/images, cooperative launch, and other unverified features are
explicitly capability-gated until tests prove them.

### 4.2 gem5 bridge and daemon

`HostGPUBridge` is a gem5 SimObject.  Host requests enter through a Unix
socket/poll mechanism, are converted into gem5 events, and are serviced on the
simulator event queue.  The bridge owns simulated VA/page-table context,
simulated VRAM, host staging, AQL queues, signals, and execution records.  It
must not call SimObject methods from an arbitrary host thread.  The daemon
offers a stable protocol version, capability negotiation, request IDs, CRCs,
credits, cancellation/epoch handling, and deterministic trace IDs.

Where existing GPUFS code assumes a guest Linux/KFD/PCI path, reuse the GPU
execution machinery but provide a host adapter with equivalent queue/doorbell,
memory, dispatch, and completion semantics.  Keep a quiesced simulated x86
SE context available when gem5's GPU translation code requires a simulated
page-table owner; do not map host pointers as if they were simulated VRAs.

### 4.3 N-rank and collective protocol

Each rank gets an opaque handle containing daemon UUID, allocation ID, virtual
address/offset, byte count, and generation.  The supervisor freezes a canonical
rank map after all leases join a `{job_uuid, epoch, world_size, model_revision,
protocol_version}`.  Stale epochs and handles are rejected.

Collective descriptors include group, sequence, operation, reduction type,
dtype, shape/count, dependency fences, and per-rank buffer handles.  Use ring
reduce-scatter + allgather for large allreduce, ring allgather/reduce-scatter,
and a small-message tree.  Include N=3 tests before claiming generality.

Payload slots use explicit `EMPTY -> READY -> CONSUMED` transitions, sequence
numbers, length, CRC, and credit/back-pressure accounting.  A collective is
source-fence -> D2H staging -> transport -> receiver scratch -> gem5 reduction
kernel -> completion signal.  The first error aborts the epoch and all ranks
observe the same failure; replay begins at a checkpointed request/layer
boundary.

## 5. Ordered implementation phases and gates

Each phase is independently committed and cannot be marked complete without
its stated gate.  The operator matrix is deliberately progressive: no-op and
vector kernels, elementwise, reductions/LDS/barriers/atomics, MFMA/GEMM,
embedding/RMSNorm/MLP/RoPE, GDN, paged attention, logits/sampling, then full
official layers.

### P0 — control plane and source freeze

Deliver PLAN/GOAL/recovery scripts, root initial commit, exact source lock,
upstream checkouts and immutable baseline tags, license inventory, model-fetch
script, and a clean offline verification gate. After `P0-SRC-01`,
`P0-SELF-01` must generalize the repository registry before adding authored
code: immutable upstream baseline identity remains under `SOURCE_LOCK`, while
checkpoint/current head identity may advance only through audited descendant
commits. The registered project set then becomes the union of locked upstream
lanes and separately declared project-authored lanes. That checkpoint creates
`self-amdgpu-runtime` with a pristine initial baseline commit before any runtime
implementation commit. This split is required before gem5, Triton, PyTorch, or
vLLM work heads can advance without weakening source provenance.

Accepted boundary: CP-0003 completes this phase. The standalone runtime has a
GPL-3.0-or-later initial baseline, a versioned C ABI foundation, static/shared
package-consumer tests, and no claimed GPU transport implementation. P1 starts
with a jointly versioned runtime-to-gem5 handshake rather than adding runtime
APIs whose transport contract is still implicit.

### P1 — gem5 host bridge and one daemon

Add the bridge SimObject, request/event protocol, simulated memory/VA ownership,
queue/signal lifecycle, code-object handoff, tracing, and a 1-CU fast config.
Gate: a host client allocates, copies, launches a trivial kernel, waits, and
verifies bytes; no guest KMD/UMD and no host arithmetic.

Accepted sub-boundary: CP-0004 freezes host transport 1.0 before queue or
memory work begins. The runtime and gem5 now share byte-exact framing,
capability negotiation, request and instance identity, topology, deterministic
reject status, one absolute deadline, and a secure pathname `SOCK_SEQPACKET`
endpoint lifecycle. Real-process gates pass for N=1/2/3/4/8 and bounded
admission, but this boundary deliberately implements no allocation, copy,
doorbell, packet submission, or GPU execution. The next sub-gate adds a
versioned single-daemon queue lifecycle and doorbell control path without
advancing the phase-level trivial-kernel claim.

Accepted sub-boundary: CP-0005 adds that queue-control extension without
changing host-transport-v1 framing. Capability bit 1 is explicitly offered and
required; queue IDs, generations, request IDs, and exact doorbell sequences are
validated before bounded create, destroy, and control-only notification mutate
gem5 state on its event queue. ACK and asynchronous completion are distinct
commit points, disconnect cleanup is generation-safe, and backpressure and
deterministic error completion are covered in both child unit suites and the
real-process gate. This boundary still implements no simulated allocation,
byte transfer, packet submission, code-object loading, or GPU execution. The
next sub-gate adds bounded simulated memory ownership and transfer only.

Accepted sub-boundary: CP-0006 adds capability bit 2 and a bounded functional
memory service without changing the frozen envelope, handshake, or queue
frames. The bridge owns sparse zero-filled allocations, deterministic
functional virtual addresses, generation-safe slot reuse, and exact transfer
commit points. H2D and D2H payloads use single sealed memfd carriers transferred
with SCM_RIGHTS; both endpoints validate descriptor cardinality, type, owner,
mode, access, size, link count, CLOEXEC state, seals, bytes, and CRC before
observable mutation. Runtime timeout, cancellation, determinate rejection, and
indeterminate-ACK poison rules preserve caller-buffer atomicity. The
real-process gate covers allocation, three-chunk byte roundtrip, free, and slot
reuse while retaining every CP-0004/CP-0005 gate. This remains bridge-private
functional storage: it is not packet-visible VRAM, SDMA timing, packet
submission, code-object loading, or GPU execution. The next sub-gate adds a
bounded signal and event lifecycle without advancing the phase-level trivial
kernel claim.

Accepted sub-boundary: CP-0007 adds capability bit 3 and a bounded bridge-private
signal/event service without changing the frozen envelope, queue frames, or
simulated-memory carriers. Signal values are signed 64-bit bit patterns with
generation-safe 1024-slot ownership, deterministic event-queue ordering,
one-shot waits, exact one-tick completion relationships, and atomic overflow
rejection. The runtime shares one monotonic request-ID namespace across queue,
memory, and signal records; it buffers only canonical completions for prior ACKed
work, retries an already ACKed wait without resending, poisons on indeterminate
pre-ACK outcomes, and publishes validated signal state atomically. The real
process gate covers timeout, retry, store-triggered completion, load/destroy,
slot reuse, and preserved handshake/queue/memory/isolation behavior. This remains
host transport state: it is not a KFD event handle, GPU-visible signal memory,
packet-visible VRAM, SDMA timing, code-object loading, or GPU execution. The next
sub-gate binds a functional allocation to gem5 GPU VA and executes one traced
trivial gfx950 AQL dispatch.

Accepted sub-boundary: CP-0008 adds only one protocol-pinned gfx950/wave64,
one-CU, one-workgroup AQL fixture. Its acceptance evidence must bind the
generation-safe queue, packet-visible allocation, and completion signal to the
real `HSAPacketProcessor -> GPUCommandProcessor -> GPUDispatcher -> CU` path,
show retired GPU instructions and a global write, update the CP-0007 signal
after native completion, and verify non-identity D2H bytes. A bridge-side byte
transform, direct command-processor call, or synthetic one-tick completion is
not dispatch evidence. CP-0008's retained 12-event trace, exact fixture/code/
packet/kernarg hashes, causal daemon exit, and zero host fallback are accepted
execution evidence. This gate does not add a generic code-object loader or
claim ROCr, HIP, OpenCL, Triton, or P2 compatibility. The next sub-boundary is
the source-exact ROCr ThunkLoader/libhsakmt ABI inventory and provider skeleton.

### P2 — ROCr/libhsakmt provider

Fork the pinned ROCr core and implement `libhsakmt_gem5` against the daemon.
Audit every loaded symbol and ABI structure against the exact source revision.
Gate: lifecycle, memory, queue, signal/event, pointer-info, and error-path
conformance tests pass under a loader audit with no `/dev/kfd`/`/dev/dri`.

CP-0009 closes only the source-inventory portion of this phase. It freezes the
124-entry ThunkLoader/libhsakmt source union, Linux shared/direct conditional
counts (123/122), 17 key-offset-partial layout records, status observations,
and the 1.1 model interface. Its original provider implementation was metadata-only plus
the established transport-open handshake and deterministic NOT_SUPPORTED
boundary; it exports zero typed hsaKmt/DRM functions and makes no KFD or
topology claim. `P2-KMT-ABI-02` is the next gate: implement a typed,
source-compatible libhsakmt shim and a versioned daemon KFD/DRM operation
envelope with fixed-width IDs, copied buffers, ownership/generation checks,
and an explicit no-device/no-production-DSO audit. The current follow-on has
implemented the official Model 1.1 DSO scaffold, managed KMT ownership,
OPEN/GET_VERSION/CLOSE execution, and fork-safe provider ownership. It does
not yet claim topology, memory, queue/doorbell, signal/event, pointer-info,
ROCr, HIP, or framework execution. Do not cast wire integers
back to host pointers or inherit upstream v1.0 fake-success behavior.

### P3 — code-object and kernel ABI gate

The P3 gate is split so source evidence cannot be mistaken for executable
support. `CP-0011` freezes two tracked gfx950 HSACO fixtures and a runtime-side
ELF V4–V6/MsgPack parser that validates PT_LOAD segments, relocations,
descriptor symbols, hidden arguments, kernarg alignment, and LDS/scratch
metadata. Gem5 records the same fixture provenance without embedding code bytes.
The accepted parser/fixture boundary still has explicit blockers: no pinned
LLVM/device-libs executable build, no gfx950-specific decoder feature proof,
`v_fmamk_f32` is absent from the locked decoder, and no reduction/vecadd HSACO
execution fixture is source-locked. `P3-CODEOBJ-02` must close those blockers
with a reproducible toolchain/device-libs build, decoder differential proof,
and a deterministic execution fixture before any HIP/OpenCL or Triton claim.
All CP-0012 and later acceptance links of `build/VEGA_X86/gem5.opt` use the
checked-in `scripts/build_gem5_mold24.sh` wrapper and the recorded
`scons -j24 --linker=mold` procedure in `docs/gem5-build.md`; its exact argv,
tool versions, freshness chain, and final binary hash are evidence.

### P4 — HIP and OpenCL host APIs

Build the standalone `self-amdgpu-runtime` plus compatible HIP registration,
module, stream, event, memory, copy, launch, error, and fatbinary APIs. Add a
separate OpenCL ICD-compatible shim and compiler path. The first product gate
is a `.cl` kernel plus host program compiled and linked against the local stack
into a normal executable. Running that executable alone must transparently
manage the gem5 connection, launch the real code object, wait, copy results,
and validate them. Direct invocation of the generic endpoint is regression
evidence only. Later gates add hipcc and OpenCL conformance subsets, always with
production runtime/device opens absent.

### P5 — Triton operators

Keep AMD TTIR/TTGIR/LLVM lowering; add an out-of-tree `gemsim_amd` backend and
launcher linked only to the stable runtime ABI. Cache keys contain gem5/runtime/
ISA/LLVM/Triton/device-lib revisions and capability bits. The first gate is
ordinary Python execution of the tutorial-equivalent vecadd correctness path
with no fallback. The complete upstream tutorial file, including its benchmark
block and large-size sweep, remains a later scale gate. After that,
expand the model-required operator matrix immediately. Profiling and P5B
parallelism do not block coverage unless measured end-to-end evidence identifies
a material simulator bottleneck.

The production launcher is operator-agnostic. It loads the installed runtime
DSO and submits compiler-produced HSACO, metadata, kernargs, allocation
descriptors and launch geometry over the generic socket protocol to a prebuilt
gem5 process. New operators must not require gem5 C++ routes, fixture hashes,
fixed PC arrays, per-op buffer callbacks, per-op trace schemas, or simulator-side
numerical oracles. Those may exist only in external regression/evidence code.
An unsupported instruction is a shared ISA-model defect, not a reason to add an
operator route.

Kernel resources use three separate lifetimes. A successful MAP consumes and
securely clears its bounded upload-staging slot; the mapped session owns copied
PT_LOAD bytes and generation-bound identity. The persistent Triton disk cache is
the cold source of truth. Resident executable mappings are measured in bytes and
allocation count rather than bounded by a model-specific kernel constant. The
current Qwen path records resident specialization and mapped-byte peaks. Only if
those measurements approach the daemon safety envelope do we enable the planned
idle-kernel LRU: a stable driver proxy retains immutable HSACO/name/ABI identity,
UNMAPs an unpinned native handle, and rematerializes it from the existing HSACO
without invoking LLVM. That change must also add managed-only self-runtime
tombstone reaping and a cross-JITFunction weighted compiled-kernel cache; doing
only one layer would trade one unbounded resource for another. gem5 retains hard
allocation/byte safety caps and never evicts a client handle autonomously.

### P5A — transparent Triton Python vecadd

The first Triton usability checkpoint is ordinary user execution of
`examples/triton/vecadd_correctness.py` through `gemsim_amd`. Its `add_kernel`
and `add()` preserve the pinned `python/tutorials/01-vector-add.py` correctness
semantics, while intentionally omitting the upstream benchmark block and its
large-size sweep. The ordinary Triton driver/launcher selection path must be
used, with no application-specific C endpoint or source-overlay import. After
that gate is accepted,
the next default action is model-operator coverage, not profiling. Always record
basic elapsed time and work counts. If a real operator/layer/model run is slow
enough to block progress, open a bounded profiling checkpoint spanning host
protocol, event queue, AQL/dispatch, CU pipelines, memory, and trace emission;
rank costs by wall-clock contribution and optimize the smallest dominant set.
Every optimization records before/after profiles, output oracle, deterministic
replay, and fallback counts.

### P5B — conditional correctness-preserving host-parallel threadblocks

Investigate a separate, explicit simulator execution mode that can process
independent threadblocks on multiple host CPU workers while preserving gem5's
architectural result. Parallel work may be admitted only when dependency,
memory-order, barrier, atomic, LDS, signal, event, and completion analysis proves
the workgroups independent at that boundary. Cross-workgroup atomics, global
ordering, unresolved aliases, or synchronization force serialization. Gates
include race-oriented adversarial kernels, serial-versus-parallel differential
state and trace checks, repeated deterministic runs, ThreadSanitizer where
practical, and measured host-core scaling. The default serial path remains a
reference and automatic fallback; saturating CPUs is not itself a success
criterion. The parallel path must publish a bounded host scheduler/memory
overhead budget against the serial baseline; if that budget or the proof of
independence fails, retain the serial path.

### P5C — model-required Triton operator matrix

After the normal Python vecadd path is accepted, expand the generated operator
manifest and differential tests immediately through elementwise, reductions,
LDS/barriers, atomics, MFMA/GEMM, embedding, RMSNorm, MLP, RoPE, GDN, paged
attention, logits, and sampling. P5B is optional and may run only when profile
evidence justifies it. Each entry keeps its compiler/runtime/simulator identity
and falls back only by explicit unsupported status, never host arithmetic.

For the official Qwen3.5-0.8B first gate, the generated local manifest is
currently 15 text-only contracts covering embedding/LM head, RMSNorm, GDN
convolution/post-prep/chunk/recurrent/norm/L2 families, six full-attention
layers (QKV/QK norm/partial RoPE/KV cache), and dense MLP. The manifest and
shape smoke are static source evidence only; they report `passed=false` while
the AMD device runner is absent. Acceptance requires every listed contract to
compile and execute on the host-native AMD path, with a differential oracle and
zero CPU/NVIDIA fallback. Vision remains deferred and full ROCm/OpenCL CTS is
not an acceptance prerequisite. The offline model preparation is pinned to
revision `2fc06364715b967f1860aea9cf38778875588b17`; the single safetensors
weight is `1,746,942,600` bytes with SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
CP-0028's pure-b010 compiler path produces the accepted 5,384-byte gfx950
`add_kernel` HSACO with SHA-256
`7308427e69dea6f320178c55863291d4d615338eb295a422a5ff7a2c2b8afa95`.
It has an exact 48-byte kernarg, 12-DWORD preload, and 256-thread workgroup;
normal Python execution validates it twice. Earlier 5,408/5,736-byte artifacts
are historical compile-only or mixed-toolchain evidence and are not authority.
CP-0030 adds the first model-required runtime subgate: the exact 6,672-byte
gfx950 BF16 `silu_and_mul_kernel` image with SHA-256
`2db0d67ff6903a737f3a4d40cf67e2a2cbbacc8b605863d31b7c90386ccb357e`.
Normal Python executes decode `[1,7168] -> [1,3584]` and masked-prefill
`[7,7168] -> [7,3584]` in fresh and repeat runs with exact PC/memory/lifecycle
traces, finite outputs, zero mismatch, and zero fallback. This accepts only the
activation/multiply subgate inside `mlp.gate_up_silu_down`; gate/up projection,
down projection, the full MLP contract, and all 15 complete runtime contracts
remain unaccepted. P5-OPS-01 must preserve that distinction while adding exact
dtype/shape/stride/state/oracle records for broader operators.

The subsequent direct-runner bring-up proves the shared generic path at model
scale without changing CP30's historical acceptance claim. The pinned
checkpoint's 24-layer empty-cache first-token backbone runs 271 generic Triton
dispatches; its full tied LM head streams all 248,320 vocabulary rows in 61
additional dispatches. The backbone and LM-head sessions retire and clean up
without Busy/resource exhaustion, all target-local operator oracles and guards
pass, values are finite, the full tied-weight SHA-256 is verified, and fallback
counters remain zero. The strict 49-point intermediate comparison to the
independent NVIDIA trajectory intentionally remains `false`, because BF16
rounding differences accumulate across layers. Model-output acceptance is a
separate explicit `bf16_cross_arch_v1` full-logit gate: cosine >= `0.975`,
relative-L2 <= `0.22`, top-20 token overlap >= `0.80`, and exact greedy/tie
decision, in addition to the target-local FP32 oracle. The retained run passes
at cosine `0.9806992579600918`, relative-L2 `0.20649629542929238`, overlap
`18/20`, and unique token `266`. This proves direct-runner first-token logits,
not the P7 stable-inference gate. P6/P7 now package these operations behind the
formal out-of-tree PyTorch/vLLM plugin; prefill and multi-token decode remain
required before single-device acceptance, and P8 CCL tests still precede TP.

The first formal framework package now lives at
`plugins/framework/gemsim_vllm`. `scripts/setup_framework_env.sh` installs the
pinned vLLM commit from an immutable archive with `VLLM_TARGET_DEVICE=empty`,
uses the persistent private pip cache, and installs the plugin through entry
point metadata. The pinned vLLM checkout remains unchanged. Pinned vLLM
discovers its `vllm.platform_plugins` and `vllm.general_plugins` entry points
without source modification. The package registers ten project-owned
`torch.library` operations with FakeTensor support. Formal OOT adapters cover
linear variants, RoPE/MRoPE, Gemma RMSNorm, SiluAndMul, RMSNormGated, and Qwen
Gated DeltaNet decode. A bounded GemSim attention backend reuses vLLM metadata
but executes NHD KV-cache update and one-token GQA decode through Triton rather
than CPU attention. Its actual vLLM `Attention.forward` smoke passes in two
dispatches; the actual GDN adapter smoke passes its eight-dispatch projection,
state-transition, norm/gate, and output path with exact cache selection and
zero fallback. A separately registered text-only Qwen3.5 architecture reuses
the upstream model and official weight loader, wraps the six full-attention
modules, and constructs all 24 layers through formal replacements. The actual
registered model now completes an unpatched, checkpoint-backed single-token
forward in 278 generic Triton dispatches. All 18 GDN state records and 6 NHD
attention caches update, the final BF16 hidden tensor exactly matches the
direct runner, and mismatch, non-finite, and fallback counts remain zero. The
OOT model explicitly preserves the checkpoint's FP32 GDN output-norm
parameters instead of silently narrowing them under the surrounding BF16
construction dtype. A bounded 2-token empty-cache prefill through layers 0..3
also matches serial decode exactly for returned hidden rows and all executed
GDN/NHD cache state. A full 24-layer two-token prefill now completes the exact
278-dispatch matrix with 24/24 finite, mutated states and terminal cleanup. Its
retained frozen-input comparison failed after accumulated cross-backend drift
(318 final-hidden outliers; rel-L2 `0.10932212645964957`, cosine
`0.9940072673597194`). The replacement fail-fast online protocol feeds NVIDIA
each actual AMD layer input, never feeds its response back to AMD, and passes
all 24 layers with zero pointwise mismatch/nonfinite/fallback while preserving
non-target caches and storage identities. This includes the formerly
first-observed layer 16 and the deepest full-attention layer 23, without
threshold relaxation. Final norm still needs the same online comparison; one
uninterrupted empty-cache production run then remains mandatory before
acceptance or TPOT A/B. Worker scheduling, batching, cache-preserving
multi-token decode, communicator, and TP remain outside this gate.

### P6 — PyTorch integration

Build unchanged PyTorch against the compatible HIP ABI (with an isolated
PrivateUse1 facade only for APIs that prove impossible to expose
transparently). Implement allocator,
DeviceGuard, streams/events, storage/copy/view, RNG, serialization, AMP and
Inductor/Triton registration. Gate eager and compile paths with CPU fallback
counter zero in acceptance mode. Operator bindings live in a separately
installable project plugin and register schemas, dispatcher implementations,
and Fake/Meta behavior through supported `torch.library` APIs only where a
project-owned operator is genuinely required; the pinned PyTorch source tree
remains unmodified. A temporary monkey patch may diagnose
a missing upstream hook, but it is removed or replaced by a formal extension
before the gate can be accepted.

### P7 — single-device vLLM

Provide a separately packaged `gemsim` platform plugin/worker/communicator and
reuse the official Qwen3.5 model implementation. Discover it through pinned
vLLM's `vllm.platform_plugins` and `vllm.general_plugins` entry points; do not
modify the vLLM checkout. Register model operators by their exact dtype, shape,
alias, mutation and state contracts. If an upstream call site has no supported
hook, an exact-version diagnostic patch may demonstrate the required behavior,
but acceptance waits for a formal plugin/dispatcher/upstream-compatible adapter
and an unpatched-mode regression.
The first gate is exact plugin discovery plus OOT replacement identity and
real registered-op execution. It now includes all 24 construction-time layer
replacements, actual bounded full-attention/KV-cache execution, actual GDN
decode execution, and an unpatched checkpoint-backed single-token forward
through the registered text-only architecture and official weight loader.
The formal adapters now accept one-request empty-cache causal prefill for 2..16
tokens; the first two-token layers-0..3 differential gate is exact against
serial decode. Unsupported attention batching, prefill with prior context,
contexts above 128, workers, and other cache modes fail closed rather than
falling back to CPU. The next formal gate is complete 24-layer prefill followed
by cache-preserving multi-token decode.

Before another long model acceptance run, add a formal online layer
differential mode. At each layer boundary, the AMD runner supplies the exact
live inputs for that layer -- hidden state, residual, selected GDN convolution
and recurrent state, NHD KV cache, token positions, and attention metadata --
to an independent RTX5090 golden process. The golden process executes only the
same official layer and returns diagnostic outputs; it is never a target
fallback and its values never feed AMD execution. Immediately after the AMD
layer completes, compare returned hidden/residual plus every mutable state with
the frozen per-tensor finite, pointwise, relative-L2, cosine, alias, mutation,
and guard rules. The first failing layer aborts the remaining model before more
simulator time is spent and retains a self-contained diagnostic package:
exact layer inputs, AMD and NVIDIA outputs, source/binary/weight identities,
kernel trace, lifecycle records, and first-mismatch statistics.

Every layer-start boundary also has an optional atomic restart checkpoint for
debugging efficiency. It contains all state required to reproduce the next
layer -- hidden/residual, every GDN convolution/recurrent state, every KV cache,
token/position/sequence metadata, RNG state where applicable, and exact model,
plugin, runner, runtime, gem5, HSACO-cache, and checkpoint identities. Files are
written into a same-parent temporary directory, fsynced, and published with a
no-replace atomic rename plus a completion manifest and hashes. Restart refuses
missing, extra, stale, aliased, non-finite, shape/dtype/stride-incompatible, or
identity-mismatched state. A fix is first replayed from the earliest affected
checkpoint through a bounded suffix. Once all online layer comparisons pass,
acceptance still requires one uninterrupted run from empty cache with no
checkpoint restoration and the exact full lifecycle/logit/greedy gates; restart
evidence accelerates diagnosis but cannot substitute for the continuous E2E
run.

Expose these behaviors as explicit inference modes rather than ambient debug
switches. `production` is the only acceptance mode: it disables NVIDIA calls,
layer dumps, and checkpoint restoration and runs continuously from empty cache.
`debug-layer-diff` enables the independent per-layer NVIDIA oracle and stops at
the first failed boundary. `debug-resume` restores one strictly validated layer
checkpoint and runs a bounded suffix. An optional `evidence` mode may retain
expanded traces and tensors, but it must not change target math, ordering, or
fallback policy. Every result records the selected mode and rejects conflicting
options; no debug or resume result may be relabelled as production evidence.
Expand OOT adapters only through supported vLLM interfaces, and keep every
unsupported worker, attention, or cache capability fail-closed rather than
emulating it through a monkey patch.
Disable CUDA graphs, unsupported quantization, speculative decode and other
features until capability-gated. First validate individual GDN and
full-attention layers, then 24-layer prefill and decode on a 1-CU quick config,
followed by a full MI355X-config single daemon. Gate: same
checkpoint/tokenizer/template, a stable predeclared multi-token decode window,
and exact greedy tokens against the reference. One decoded token is a smoke,
not this acceptance gate.

### P8 — N-rank transport and RCCL semantics

Implement rank leases, epochs, allocation namespaces, ring/tree planners,
collective tracing, failure/replay, and functional FabricModel. The CCL API and
state machine use one bounded generic `world_size=2..16` contract, not TP=2 or
power-of-two branches. Gate every integer world size in that interval, with
explicit non-power-of-two coverage at N=3/5/7/15, plus negative/error/restart
cases; host reduction counter remains zero. This phase is a hard prerequisite
for vLLM TP integration, not an implementation detail tested for the first time
inside P9.

First freeze a small versioned CCL API and run pure planner/codec/state-machine
unit tests for rank/group/sequence identity, dtype/shape/count legality,
allreduce, allgather, reduce-scatter, broadcast, barriers, chunking, credits,
timeouts, stale epochs, peer loss, abort, cleanup, and reconnect. These pure
tests iterate all N=2..16 rather than sampling only TP=2/4/8. Then run
multi-process tests with independent daemon namespaces at N=2 and N=3 before
N=4/8/16 live topology gates. Reduction payloads must execute through gem5 kernels;
host-side arithmetic is an observed failure. Only after this suite is stable
does the out-of-tree PyTorch distributed/vLLM communicator plugin bind TP calls
to the CCL API. The first TP tests reuse the already-passing collective vectors
and add only framework ordering and model-sharding assertions.

Before collectives or TP, prove isolated multi-instance execution. A supervisor
launches at least two independent gem5/runtime instances with unique socket and
run directories, daemon UUIDs, job/epoch/rank identities, allocation/queue/
signal namespaces, trace/stats/config outputs, and cache namespaces. No mutable
SimObject, singleton runtime session, file descriptor, signal, allocation,
completion, or cleanup record may cross instances. Correctness regressions may
run concurrently once this isolation gate passes. Performance A/B remains
serial by default, or uses disjoint pinned CPU/core and storage resources with
an explicit no-interference calibration; oversubscribed parallel wall time is
not valid speed evidence. TP then reuses the same isolated-instance launcher:
rank instances exchange tensor payloads only through the versioned CCL
transport, never process globals or shared simulator state. Gates cover
simultaneous success, one-instance failure/timeout/restart, stale endpoint and
epoch rejection, peer cleanup, deterministic evidence attribution, and no
orphan processes/sockets.

Treat this gate as early shared infrastructure rather than waiting for all P7
numerical work to finish. Its first implementation uses existing per-process
managed sessions and socket spawning without a wire extension. Start with two
and four concurrent short correctness workloads, then expose a reusable test
matrix API so layer-differential shards, operator suites, fault cases, and CCL
ranks share the same launch/isolation/evidence code. Only after correctness
isolation passes add explicit CPU affinity, per-instance resource accounting,
and serial-versus-concurrent calibration for profiling. A task scheduler may
parallelize independent correctness jobs by default; it may parallelize
performance samples only when the calibration's slowdown/variance bounds are
met and recorded.

The first supervisor implementation is now available as
`scripts/run_gemsim_instances.py`. Focused mock regressions pass for two and
four workers, independent-failure survival, timeout cleanup, stale endpoint and
epoch rejection, absent-only output roots, per-instance writable caches, and
explicit recursively read-only shared caches with before/after tree hashes. A
live two-instance and four-instance VecAdd gate also passes: every worker has a
unique Triton cache, run directory, socket, daemon UUID, job UUID, and epoch;
all process/device-open audits are clean and no runner or gem5 process remains.
This accepts the reusable correctness-isolation substrate only. Each worker is
still rank 0/world size 1, concurrent wall time is not TPOT evidence, and CCL
group/rank assignment remains the next P8 layer.

The formal group increment is now present. `gemsim-rank-launch.v1` and
`gemsim-live-registry.v1` make all ranks share one exact job UUID, epoch and
world size while covering `0..N-1` exactly once; daemon UUIDs, endpoints, run
roots, writable HOME/tmp/XDG/cache namespaces, allocation state and traces stay
per-rank. Managed-session v2 carries that exact topology into both spawned gem5
argv and HELLO, with no external-endpoint topology wildcard. A supervisor-held
lease plus atomic generation-stamped registry makes READY/OFF, normal stop and
stale epoch observable without log scraping. `scripts/gemsim_smi.py` reads only
that registry and reports daemon/job/epoch/rank/world identity without opening
device nodes or loading ROCm SMI. A real N=2 VecAdd group smoke passes with two
different daemon UUIDs, sockets, traces and caches, shared job/epoch, exact ranks
0/1, correct output and zero fallback. This proves grouped execution identity;
it is not yet a collective or TP result.

After the registry/identity unit tests pass, implement the standalone CCL in
strict order: versioned ABI/status and size-tag tests; descriptor codec, CRC and
sealed-memfd carrier tests; complete N=2..16 ring/tree planner coverage,
including non-power-of-two N=3/5/7/15;
group/sequence/epoch state-machine and canonical abort tests; slot credit,
timeout, peer-loss and cleanup fault tests; single-rank Triton reduction-kernel
proof with `host_reduction_count=0`; then live N=2, N=3 and N=4/8 multiprocess
collectives. SUM BF16/FP32 all-reduce is the first model-critical reduction;
all-gather, reduce-scatter, broadcast and barrier remain required standalone
contracts. Only after this matrix passes may the out-of-tree vLLM
`DeviceCommunicatorBase` adapter expose collectives and begin TP=2 model
sharding.

The first standalone CCL foundation is complete at the API/codec/planner/state
and byte-carrier layer. It exports a size-tagged C ABI, canonical 160-byte
big-endian descriptor
with CRC32C, BF16/FP32 SUM metadata, copy-carrier dtypes, ring
reduce-scatter/allgather/allreduce, broadcast, barrier, and a sequence-gated
fail-closed group state machine. Unit tests iterate every `world_size=2..16`,
all supported collectives and every rank, including uneven N=3 chunks. The
runtime release/ABI is now `0.8.0`/`1.8`; the state-machine corruption matrix
validates every public mutation and every world size, including sequence
exhaustion. Rank-launch descriptors are private no-follow files and are the
single source of rank/cache identity, fork-inherited Python runtime sessions
fail closed, and a READY group aborts all peers if a rank or daemon is lost.
Mock supervisor gates cover N=2/3/8/16, but real gem5 grouped execution is
currently proven only at N=2. The live collective acceptance matrix therefore
remains N=2/3/4/8/16 and cannot be replaced by unit or mock coverage. The
carrier uses a canonical 240-byte `DATA`/`CONSUMED`/`ABORT` record,
rank-independent descriptor SHA-256, and four-seal read-only memfd payloads
passed over nonblocking AF_UNIX SOCK_SEQPACKET with exactly zero or one
SCM_RIGHTS descriptor. A caller-serialized opaque session is the production
ownership boundary: it owns every payload FD and per-peer credit through exact
`CONSUMED`, preserves the immutable first-error record and its original reporter
through relays, atomically reclaims all resources on an ambiguous
send/peer/protocol failure, and rejects stale or replayed slot generations. Its
focused matrix exercises every collective/rank/send edge at every N=2..16,
credit depths 1/4/16, a raw-credit world-16 saturation of all `15 * 16 = 240`
peer slots, and an opaque-session pressure gate with all 16 rank sessions each
owning 16 planner-valid payload FDs before group abort reclaims all 256. It also
covers full session round trips for every collective at every N=2..16,
multi-credit out-of-order
send/ack, output-before-consume, wire/CRC/FD corruption, zero-byte records with
SCM_RIGHTS, zero-length chunks, invalid-descriptor preflight, peer close, exact
automatic/remote abort retrieval and relay, late replay, FD conservation, and
first-error abort, including exact wire-version mismatch propagation, under
Clang/GCC and ASan/UBSan. The carrier
performs byte copy only and contains no tensor arithmetic. The first
normal-Triton device primitive now passes independently through the
out-of-tree `plugins/collectives/gemsim_ccl` package. Its only production
operation is in-place private-workspace SUM with an immutable, disjoint
received source. CCL v1 deliberately defines BF16 as FP32 pairwise addition
followed by RTNE BF16 storage after every planner receive; FP32 performs one
binary FP32 addition per receive; zero elements do not dispatch. A direct
matrix covers both dtypes at counts
0/1/3/127/128/129/255/256/257/1024/1027/2048/7168, including BF16 even/odd
halfway bit patterns and Qwen hidden/intermediate sizes. All 24 nonzero
launches are bitwise equal to the one-step oracle and preserve
source/tail/guards. The formal product-v1 evidence contains exactly 24 retired
and 24 type20-durable SUM dispatches, 23 reuse completions, and one clean
session completion. Its external verifier binds the numerical result,
authoritative trace and log, all four code-object images, product manifest,
base Python, noneditable Triton and CCL snapshots, runtime DSO, gem5
binary/config and postflight identity. This accepts the standalone device
primitive. Live collectives must operate on
private workspace and commit a fresh public output only once all steps succeed;
the single-step primitive does not claim public-output failure atomicity. The
host-only authenticated rendezvous gate is now complete: one generic broker
constructs private rank-indexed nonblocking Unix seqpacket peer tables for every
`world_size=2..16`, binds each capability to exact PID/UID/GID/start-time and
group identity, uses an absolute deadline, and preserves one immutable
group-wide first error. Full live tests cover N=2/3/4/8/16, with configuration,
FD conservation and cleanup coverage at every N=2..16; the broker never reads
DATA peer sockets and performs no tensor copy or reduction. This accepts only
the control-plane rendezvous. The ordered host-mock gate now runs the formal
planner, authenticated peer table and opaque carrier sessions at
N=2/3/4/8/16. It performs DATA -> immutable staging -> explicitly counted host
mock arithmetic -> CONSUMED, with zero chunks, credit BUSY and reordering,
replay rejection, bounded group abort and peer loss, identical per-rank result
hashes, FD conservation and no orphan processes. N=16 executes all 240 reduce
steps; an N=8 count-three case exercises 70 zero-chunk transfers. The result
explicitly reports `device_sum_count=0`,
`device_collective_acceptance=false`, and `live_gem5_acceptance=false`, so it
accepts only transport, ordering, staging, fault and cleanup. The next gate
replaces mock arithmetic with the ordinary Triton device SUM and binds every
rank's authoritative trace. N=2/3/4/8/16 now pass that formal external gate.
Each planner
ordinal is one full-duplex protocol step: the outbound DATA holds a sender
credit until the exact matching CONSUMED tuple returns, while the independent
inbound DATA lands in immutable staging and may emit its exact CONSUMED only
after copy or device SUM completion. Both chains bind group, descriptor SHA,
sequence, phase, step, chunk, source, destination, slot and slot generation;
the step is complete only when both chains reach their terminal state. The
carrier's 16 MiB record limit is not a public collective-size limit: larger
collectives are split into contiguous segments with consecutive descriptor
sequences and explicit global element offsets, all segments use private
workspace, and the fresh public output is committed exactly once after every
segment succeeds. Design-only objects and caller-supplied booleans, counters or
hash strings are never acceptance authority; an absent-only external verifier
must reopen and hash source artifacts, replay the planner and numerical oracle,
and bind per-rank traces and lifecycle. Only those device collective results
can unlock the out-of-place framework communicator. The reusable-engine N=2
bundle is `artifacts/evidence/ccl-engine-live-allreduce-n2-bf16-1024-v1-accepted`
with result SHA-256
`5525bddc7cc397e4a5507679e03b20cc2f5378ae8a1a3025420c82e309f94499`
and manifest SHA-256
`6491e02a8d3255ec912db024dcd161504d2797f1ddb3305e660d9706516cdc99`.
The accepted odd-world N=3 bundle is
`artifacts/evidence/ccl-engine-live-allreduce-n3-bf16-1024-v1-accepted` with
result SHA-256
`d527ade376c2f7c57e790b0547a2078036eeadc0ef1bf7652796a361cf845aac`
and manifest SHA-256
`1a6a0efa490a0b55caeb29f456eedf3a8ef8ab79c031b1bd102d72697d3784c4`.
All five results report zero host reduction/fallback, measured FD delta zero,
and no orphan process. N=2 has one retired/type20 SUM per rank; N=3 has two and
covers uneven 342/341/341 chunks. N=4/8/16 have 3/7/15 SUMs per rank and
12/56/240 total device reductions. The remaining accepted bundle hashes are:

- N=4: result `80d2696b327190c1729cd684874c10c50f92aa6ff0d675eb8733a547cfcc508c`,
  manifest `b4ab91c03806601e69d85fce2507eed2db92a29bba20e28be1eb58e788c66744`.
- N=8: result `0c5a13cada392e4a172df84a27de90295cd431d88303ee203d7d3b80f6009b9b`,
  manifest `a3d112301be4f91d302c7fa7d1fe17fc056403b2d0d2f96d719c2a60e612db72`.
- N=16: result `9c8c4f22bf3ad87c8bea539797eb1a2023aecb031bc33977aefdb83beeddf988`,
  manifest `3ccb26fb56cceae1be20596fcca97a2e1b0960ff4a3ef2d78bdad4881b3a7d03`.

The isolated active product is
`product-v1-7ddf908cad5bd13ed2b0129be730c15d68c4c0581f3d1f8925bd264b62affe8c`.
Its manifest SHA-256 is
`e36465dd3572c2faa10897fd37182c78e1efbe93a8e8988e9cf203ec14a2dbf3`;
the runtime DSO SHA-256 is
`0de2b46c76335f3e026f26c6a22095b15d2228d28d157e73c03ac0bfec217136`.
Reproduce it with `scripts/setup_rocm_env.sh --freeze-product --jobs 24`,
`--product-runtime --jobs 24`, then `--verify-product`; `--print-prefix` is the
constant-time runtime entry. The accepted device-SUM bundle is
`artifacts/evidence/ccl-device-sum-product-v1`: result SHA-256
`b46c2937b6ba9d033db0f923c30e48545e4c3bacdb0597b237f7c639bfc4e70a`,
manifest SHA-256
`2730c60b3c0fa0248d44af45ae37646d50711e96a80c980b2f7e91a1d90de32b`,
and trace SHA-256
`7d8f53992f58a308d3ff1fa00053ee944763f50815e646b06fcd6d20999d253f`.

`P8-FRAMEWORK-COMM-01` is now accepted at N=2/BF16/1024. The formal entry
point is the pinned vLLM `GroupCoordinator.all_reduce`, not a direct
communicator constructor. It reaches the out-of-tree
`GemsimDeviceCommunicator` and the same reusable CCL engine. The two ranks emit
two normal Triton SUM dispatches total, match the independent per-hop BF16 ring
oracle bitwise, preserve input bytes, return fresh output storage, and report
zero calls across 24 audited Gloo tensor APIs, zero host reduction/fallback,
zero FD delta, and zero orphan processes. The accepted bundle is
`artifacts/evidence/vllm-ccl-live-n2-bf16-1024-v1-accepted`: result SHA-256
`22104223acb659a1706623632d54d853209c2cc98a20880924c54c0a92f88b7b`,
manifest SHA-256
`699b24116e662463c8d6523255be78f991a3fe420dd29c8671fd8ce2ec369340`,
and source manifest SHA-256
`122b47eb73ec046ec080342fae155de19d90e90d6b0eacac6103499b21af7327`.
This is not a RowParallel, model-sharding, logits, sampling, or Qwen TP claim.
The next gate is a Qwen hidden-size-1024 `RowParallelLinear` with exact TP2
MLP `down_proj` weight `[1024,3584]`, upstream-loader rank shards
`[1024,1792]`, weight ranges and reconstruction SHA, local dense output, one out-of-place
all-reduce, input immutability/fresh output, single-device oracle comparison,
collective-sequence/trace identity, zero Gloo model payload/fallback, and clean
two-daemon teardown. Only after that passes may the remaining
Column/Merged/QKV/Vocab/GDN/full-attention TP adapters be enabled individually.
The gate must use vLLM's supported group initialization and actual OOT layer
construction/weight loader; it may audit fixed Gloo control traffic but must
not assign private `_TP` state or substitute a test-only sharding framework.

### P9 — TP=2 Qwen anchor (critical milestone)

Launch two independent gem5 daemons and two vLLM ranks. Implement exact Qwen
sharding: vocabulary/LM-head tying, attention Q/KV head rules, row/column MLP
and projection allreduces, and the GDN convolution/QKVZ/BA/state sharding.
Run full prefill then a stable predeclared multi-token greedy decode window and compare token IDs,
selected logits, hidden-state/layer checkpoints, trace completeness, and
fallback/device-open audits. This is the first “usable” release; earlier
phases are prerequisites, not a claimed finished product.

### P10 — TP=4 generalization and framework portability

Run an actual four-daemon job and the required collective suite for TP=4. Exercise
replicated KV-head rules and all shard divisibility constraints. Run the full
Qwen token acceptance whenever the official tensor shapes permit it; otherwise
fail explicitly with a recorded shape/semantic reason and retain the protocol
acceptance. TP=8 is not a required matrix.

The same Qwen3.5-0.8B TP=4 workload then runs through pinned upstream SGLang.
The SGLang checkout and runtime-gem5 bridge must remain byte-identical to their
vLLM-lane identities. Unchanged SGLang must auto-select its in-tree ROCm
platform from the standard PyTorch HIP identity. RCCL/NCCL compatibility is
the primary collective path and the generic c10d backend is a bounded fallback
only where upstream c10d is actually selected. The evidence records the
actual ROCm attention/GEMM backend chosen by SGLang, rank/shard reconstruction,
token output, trace, fallback and teardown; a copied model path or host mock is
not acceptance.

#### P10-SGLANG portability sequence

1. Rehash the unchanged SGLang 0.5.17 wheel and run the official platform-hook
   audit. Retain 0.5.10.post1 only as the historical negative comparison.
2. Implement and unit-test the shared generic c10d fallback, including in-place
   collective semantics, async `Work`, rank/world identity, descriptor binding,
   fork/timeout/abort behavior, and no host reduction.
3. Build the separate CPython 3.13 SGLang product and prove exact imported
   upstream files. The diagnostic OOT platform must remain inactive without a
   real HIP facade; initialization-only Gloo control metadata is audited
   separately.
4. Complete the common HIP-compatible PyTorch device facade. Prove that
   unchanged SGLang allocates HIP tensors, selects `forward_hip`, uses upstream
   Triton attention (the upstream AMD recommendation), and cannot enter the
   CPU/AMX shared-memory collective. Reuse at least two unrelated Triton
   operators before resuming the model gate.
5. Publish the RCCL/NCCL-compatible collective facade and prove that upstream
   SGLang selects its own ROCm platform and device collectives. Reuse the c10d
   diagnostic only for explicit fallback coverage.
6. Run a fake two-rank SGLang collective gate and external verifier, then the
   real TP=4 Qwen3.5-0.8B prefill plus multi-token decode on four daemons.
7. Any missing upstream hook or device semantic is recorded as a generic
   platform/runtime capability gap. Fix it once below the framework and replay
   both vLLM and SGLang probes; never patch SGLang symbols or add a Qwen case.

### P10B — Qwen3.5-9B TP16 compile scale

Prepare the official `Qwen/Qwen3.5-9B` (or its explicit `-Base` checkpoint if
the conversational checkpoint is unavailable) in the private model cache.
Launch sixteen independent simulator ranks with vLLM V1 and
`torch.compile` enabled. Let the upstream ROCm selector choose AITER,
ROCM_ATTN, or TRITON_ATTN; record the selection and reject hidden CPU/NVIDIA
paths. Require at least a short prefill plus multi-token decode, exact rank
and cache lifecycle, compile cache identity, zero fallback, and clean SMI
16-slot ON/OFF evidence. This is a scale gate, not a replacement for the
0.8B TP4 numerical model gate.

### P11 — timing, breadth, and hardening

Add timed xGMI/SDMA/FabricModel, larger MI355X configurations, performance
metrics, broader HIP/OpenCL surface, images/interop/debug features, packaging,
upgrade compatibility matrices, fuzzing, fault injection, and reproducible
offline bundles. No timing result is published from P8/P9 functional mode.

### P11A — simulator-aware `rocm-smi`

Status: complete. The repository conda product installs both `rocm-smi` and
`gemsim-smi`. They expose 16 logical simulator slots without eagerly starting
16 gem5 processes. A managed runtime session atomically claims the first free
slot only after its private gem5 endpoint and handshake are ready, and holds a
file lease for the session lifetime. The reader revalidates the lease, record
SHA-256, owner/daemon PID and start time, gem5 executable inode, daemon/job/
rank/world identity, and private socket before reporting `ON`; an unused,
stopped, crashed, stale, or unverifiable slot is `OFF`, while a locked corrupt
record fails closed. The live product gate observed `0/16 -> 1/15 -> 0/16`
ON/OFF across a real managed gem5 session and left no process or lock behind.
The tool does not open `/dev/kfd` or `/dev/dri`, load ROCm SMI libraries, or
invent temperature, power, clocks, utilization, or hardware health.

### P11B — framework-neutral AMD device facade and SGLang TP4

This is the next execution lane after the KMT/SMI and bridge regression gates.
The SMI lane is complete and remains a regression prerequisite; it exposes 16
logical simulator slots, while model evidence is intentionally limited to the
requested TP=2 prerequisite and TP=4 framework gates.
The implementation order is strict:

1. Build and probe the unchanged upstream ROCr/libhsakmt, HIP/CLR, and pinned
   PyTorch ABI boundary against the existing self-runtime model provider. A
   library load, `hipGetDeviceCount`, allocator, stream/event, copy, module,
   and one generic kernel probe must all use the runtime-gem5 bridge; a
   `torch.library` operator list or CPU/PrivateUse1 emulation cannot mark this
   prerequisite complete.
2. Publish the facade in a private conda product with explicit library search
   paths and no system ROCm/KFD/DRM loads. Reuse the same product for vLLM and
   SGLang; do not copy or patch framework code. `torch.compile` and Triton
   compilation remain upstream operations over the normal HIP device contract.
   The product build must consume a read-only, content-addressed standard
   ROCm stage and the frozen runtime source snapshot; no framework or operator
   source is copied into the bridge. If this product slice does not converge
   within one bounded build/test cycle, preserve the accepted product and
   return to the prior explicit-dispatch lane rather than adding a case split.
3. Run the existing vLLM TP2 prerequisite, then vLLM Qwen3.5-0.8B TP4. In
   parallel only for host/source checks, run the unchanged SGLang 0.5.17
   `RocmSRTPlatform` path and its official device/communicator hook audit.
   SGLang TP4 is accepted only when the same facade, CCL engine, bridge hash,
   shard oracle, token window, trace, and cleanup gates pass. The SGLang
   checkout and bridge must be byte-identical to their vLLM-lane identities.
4. Once TP4 is stable, run Qwen3.5-9B TP16 with upstream vLLM V1 and
   `torch.compile` enabled. The run records the selected AMD attention backend
   (AITER/ROCM_ATTN/TRITON_ATTN/Triton), rejects CPU/NVIDIA/project-specific
   fallback, and is a scale gate rather than a new per-operator path.

If the unchanged upstream stack cannot cross a generic ABI boundary, stop at
the exact failing layer and add one shared facade/bridge capability there.
Never solve the failure by editing SGLang/vLLM/Triton/PyTorch, by adding a
framework-specific branch to runtime-gem5, or by adding an operator/model case.
Every facade repair must pass two unrelated kernels and both framework probes
before model work resumes. TP model evidence is required only at TP=2/4;
CCL and logical simulator capacity remain generic for every world size 2..16,
with 16 simulator-aware SMI slots.

### P12 — top-level user guide and release entry point

Before the final release claim, replace the root `README.md` with a tested,
public-facing entry point rather than a progress log. It must explain the
project's purpose and supported/unsupported boundaries; root and child-project
organization; pinned sources and private environment layout; prerequisites;
one-command incremental builds for gem5, runtime, LLVM/Triton, framework plugin,
and CCL with the documented ccache/Clang/LLD/`-j24` policy; and how build caches
and retained build directories are reused and cleaned safely.

The guide includes copy-paste commands and expected outputs for a normal OpenCL
operator, a direct Triton `.py` operator, operator differential tests, formal
single-node Qwen prefill/decode, the `production`, `debug-layer-diff`,
`debug-resume`, and `evidence` inference modes, layer checkpoint inspection,
concurrent isolated instances, standalone CCL tests, TP=2 and TP=4,
and simulator-aware SMI. Every compile and inference option has its default,
purpose, interaction, safety boundary, and example. It also covers NVIDIA
golden isolation, zero-fallback/device-open audits, performance/TPOT measurement,
evidence locations, offline/cache behavior, troubleshooting, cleanup, license,
and exact unsupported features. Documentation commands run in CI or a retained
release smoke; stale or illustrative-only commands fail the documentation gate.

### Working checkpoint forecast

Checkpoint IDs are allocated only when an atomic transaction begins, so this
table is a planning forecast rather than a fixed total. A difficult phase may
split into additional checkpoints and no later row may skip its prerequisite.
Each remaining row is a distinct forecast gate and must receive its own
transaction/checkpoint ID; IDs are never merged across rows. The table is a
conditional sequence rather than a fixed total: a difficult row may split into
additional checkpoints, and a later row may not skip its prerequisite.

| Forecast gate | Earliest phase | Result |
| --- | --- | --- |
| CP-0008 | P1 | One traced real CU-backed pinned dispatch |
| P2-KMT-ABI-01 | P2 | Source-exact thunk/libhsakmt ABI inventory and provider skeleton |
| P2-KMT-ABI-02 | P2 | Typed libhsakmt shim and versioned daemon KFD/DRM operation envelope |
| P3-CODEOBJ-01 | P3 | Pinned gfx950 code-object and kernarg ABI fixtures |
| P3-CODEOBJ-02 | P3 | Pinned toolchain/device-libs build and gfx950 decoder/execution proof |
| P3-CODEOBJ-03 | P3 | Versioned HSACO transport, PT_LOAD mapping, dynamic AQL/kernarg, and gem5 execution |
| P3-HOST-NATIVE-01 | P3H | GPU/x86 dependency inventory, reusable-core boundary, and host-native compatibility contract |
| P3-HOST-NATIVE-02 | P3H | Standalone no-VEGA_X86 build plus host event/memory/queue/signal adapter |
| P3-HOST-NATIVE-03 | P3H | Pinned gfx950 loader/dispatch parity between host-native and gem5 front-ends |
| P3-HOST-NATIVE-03-A | P3H | Fixture-scoped PT_LOAD staging, descriptor/entry binding, and lifetime checks |
| P3-HOST-NATIVE-03-B0 | P3H | No-x86 descriptor/kernarg/AQL admission and queue/lifecycle contract smoke |
| P3-HOST-NATIVE-03-B1 | P3H | No-x86 native queue/CP-core address resolution, AQL publication/fetch, and admission without legacy SimObjects or CU execution |
| P3-HOST-NATIVE-03-B2 | P3H | No-x86 GPUDispatcher/CU functional output/trace differential for the locked 4-WG/16-wave gfx950 dispatch |
| CP-0027 / P4-OPENCL-E2E-01 | P4 | User `.cl` plus host program builds into a normal executable that transparently runs on gem5 and validates output; low-level endpoint remains regression-only |
| CP-0021 | P5 | Hash-bound Triton vecadd compile/provenance prerequisite; normal launcher remains blocked |
| CP-0022 | P5 | Generic payload-v2 codec/admission boundary; daemon mapping and launcher handoff remain blocked |
| CP-0023 / P5-TRITON-VECADD-02-RUNTIME | P5 | Owner-bound runtime client plus local native MAP/ALLOC/publish/fetch/CP-admission/retire/unmap; daemon route remains blocked |
| CP-0024 / P5-TRITON-VECADD-03-DAEMON-ROUTE | P5 | Bounded handler source/type-19 plumbing, route-policy harness, and live canonical negative handshake; bit 8 and positive type-18/H2D/SUBMIT route remain blocked |
| CP-0025 / P5-TRITON-VECADD-04-DAEMON-LIFECYCLE | P5 | Accepted positive bit-8 daemon control lifecycle through logical-align-8 ALLOC, v1 H2D, native CP admission/type-19 ACK/type-20 retire, cleanup, and reconnect; no GPU execution |
| CP-0026 / P5-TRITON-VECADD-05-GPU-EXECUTION | P5 | Connect the committed daemon lifecycle to GPUDispatcher/CU execution and prove output correctness for the locked zero-preload fixture |
| CP-0028 / P5-TRITON-VECADD-06-NORMAL-PYTHON | P5A | Ordinary Python/Triton compiles and executes exact contiguous float32 vecadd twice in one managed session with resource reuse, exact output, and zero fallback; broader operators and final repo-local install remain bounded |
| CP-0029 / P5-TRITON-VECADD-07-FRESH-SCHEMA8 | P5A | Accepted: test-only queue ordering fix, GCC/Clang serial and 18-worker stability, fresh schema-8 repository-local prefix, independent verify-only, OpenCL/Triton, provenance, pollution, and active-isolation gates |
| CP-0030 / P5-OPS-00-SILU-AND-MUL-MINIMUM | P5C | Accepted bounded BF16 SiluAndMul subgate: decode `[1,7168] -> [1,3584]` and masked-prefill `[7,7168] -> [7,3584]`, fresh/repeat exact image/trace/oracle, finite outputs and zero fallback; complete MLP/model contracts remain unaccepted |
| P5-PROFILE-ON-BLOCKER | P5A | Conditional retained profile and measured 80/20 optimization only after a real operator/layer/model bottleneck is demonstrated |
| P5-PARALLEL-TB-ON-BLOCKER | P5B | Conditional safe serial-versus-parallel threadblock experiment justified by a retained profile |
| P5-OPS-01 | P5C | Broader model-operator manifest and differential gates |
| P5-QWEN35-OPS-01 | P5C | All 15 text-only Qwen3.5-0.8B operator contracts execute on AMD with no fallback |
| P5-TRITON-VECADD-UPSTREAM-SCALE | P5C+ | Deferred full unmodified upstream tutorial file, benchmark block, and large-size sweep; not a prerequisite for the model-specific operator matrix |
| P6-PYTORCH-01 | P6 | PyTorch eager/compile device foundation plus out-of-tree dispatcher/custom-op plugin; pinned PyTorch source remains unmodified |
| P7-VLLM-PLUGIN-01 | P7 | Private-environment vLLM platform/general plugin, exact-version compatibility adapters, and unpatched-mode regression without modifying vLLM |
| P7-VLLM-SINGLE-01 | P7 | Single-daemon Qwen model path through the out-of-tree plugin |
| P7-LAYER-DIFF-01 | P7 | Independent NVIDIA executes each official layer from the exact live AMD layer input; compare outputs and every mutable cache immediately, stop at the first failure, and retain a self-contained diagnostic package with zero fallback |
| P7-RESTART-01 | P7 | Atomically checkpoint every complete layer boundary, fail closed on identity/state drift, replay a bounded suffix after a fix, then require an uninterrupted empty-cache E2E rerun before acceptance |
| P8-MULTI-INSTANCE-01 | P8 | Concurrent isolated gem5/runtime instances have disjoint sockets, namespaces, state, traces and cleanup; correctness may run in parallel, while performance A/B requires calibrated non-interference |
| P7X-MULTI-INSTANCE-EARLY-01 | P7/P8 | Early reusable 2/4-instance supervisor and isolation/fault/evidence gates accelerate layer debug and operator regression, then become the unchanged launcher foundation for CCL/TP |
| P8-CCL-API-01 | P8 | Versioned standalone CCL API plus API/codec/planner/state/carrier/credit/fault tests over every world size 2..16; sealed byte transport is accepted, while device reduction and live multiprocess collectives remain separate gates and host reduction stays forbidden |
| P8-COLLECTIVE-01 | P8 | Independent-daemon N=2..16 multi-process collective semantics; hard prerequisite for framework TP binding |
| P8-FRAMEWORK-COMM-01 | P8 | Accepted N=2 standalone pinned-vLLM GroupCoordinator allreduce through the OOT communicator and reusable CCL engine; no model TP claim |
| P9-ROWPARALLEL-01 | P9 | Qwen hidden-size-1024 TP2 RowParallelLinear sharding, local dense, out-of-place allreduce, single-device oracle, trace and cleanup gate |
| P9-QWEN-TP2-01 | P9 | Full TP=2 prefill and greedy decode anchor |
| P10-QWEN-TP4-01 | P10 | Full TP=4 protocol, sharding, inference, trace, and cleanup gate |
| P10-SGLANG-PORTABILITY-00 | P10 | Pinned SGLang source identity, in-tree ROCm auto-detection, official-hook audit, and diagnostic c10d contract |
| P10-SGLANG-C10D-01 | P10 | Generic out-of-tree PyTorch ProcessGroup backend and host state/fault/async tests; no device/model claim |
| P6-HIP-PYTORCH-FACADE-01 | P6/P10 | Shared ROCr/HIP-compatible PyTorch device, allocator, copy, stream/event, launch and Inductor/Triton contract used unchanged by vLLM and SGLang |
| P8-RCCL-ABI-01 | P8/P10 | Standard libnccl.so.2/RCCL-compatible communicator and collectives over the generic CCL engine, with stream ordering and group abort |
| P10-SGLANG-CONDA-01 | P10 | Immutable CPython 3.13 SGLang product bound to unchanged upstream wheel, shared native product, CCL and bridge identities |
| P10-SGLANG-TP4-01 | P10 | Same Qwen3.5-0.8B TP4 workload through unmodified upstream SGLang and the unchanged runtime-gem5 bridge |
| P10B-QWEN35-9B-TP16-COMPILE-01 | P10B | Qwen3.5-9B TP16 scale run with upstream ROCm backend selection, vLLM torch.compile, rank identity, trace, and cleanup |
| P11-SMI-01 | P11A | Accepted 16-slot runtime lease inventory and real managed-gem5 OFF -> ON -> OFF status tool |
| P11B-FACADE-01 | P11B | Unchanged upstream ROCr/HIP/CLR/PyTorch device probe through self-runtime and the generic bridge, with no KFD/DRM/CPU fallback |
| P11B-VLLM-TP4-01 | P11B | Qwen3.5-0.8B TP2 prerequisite followed by vLLM TP4 on the shared facade |
| P11B-SGLANG-TP4-01 | P11B | Qwen3.5-0.8B TP4 through unchanged upstream SGLang 0.5.17 and the same bridge/facade |
| P11B-VLLM-9B-TP16-01 | P11B | Qwen3.5-9B TP16 vLLM V1 with upstream AMD attention selection and torch.compile |
| P11-HARDEN-01 | P11 | Timing, breadth, packaging, and fault hardening |
| P12-README-01 | P12 | Tested root README covers project layout, builds, OpenCL/Triton operators, single-node and TP inference, CCL/SMI, all compile/inference modes, evidence, cleanup, and troubleshooting |

## 6. Model and operator manifest

The model manifest is generated, not hand-maintained: scan the official vLLM
Qwen registry/import graph, capture FakeTensor/Meta/Export graphs for each
phase, and correlate runtime traces with source, schema, shape, dtype, stride,
mutation/state semantics, lowering path, oracle, test, and last passing commit.
An unregistered observed operation, an unsupported operation that silently
falls back, or a “supported” entry without a differential test fails CI.

Qwen3.5-0.8B is a mixed architecture (24 layers, 18 Gated DeltaNet layers and
6 full-attention layers, BF16, hidden size 1024, vocabulary 248320, tied
embedding/LM head).  The plan therefore gates causal conv1d, recurrent/chunk
GDN state, gated RMSNorm, full attention/RoPE, MLP, logits, and sampling before
the model milestone; a Transformer-only approximation is invalid.

## 7. Verification and evidence

Every phase records commands, exact working directories, exit codes, stdout/
stderr hashes, environment/source/model revisions, limitations, and trace IDs
under `state/evidence/`. Required audit classes are:

- ABI/layout/handle/generation/OOM/double-free/concurrency/DSO tests.
- Code-object metadata, kernarg, dynamic LDS/scratch, queue fences and signal
  ordering tests.
- HIP/OpenCL/Triton/PyTorch API and operator differential tests.
- Per-op and per-layer numeric tolerances, NaN/Inf checks, deterministic repeat
  checks, and exact final token ID.
- `/proc/<pid>/maps`, dynamic dependency and syscall audits proving no official
  UMD/KMD/device nodes; no `amdsmi` probing.
- Device execution record coverage for every accepted operation and CPU
  fallback counter equal to zero.
- OpenCL direct-executable and Triton normal-Python transparency, including
  compile/load/launch/wait/copy/oracle evidence with no manual endpoint.
- When a real blocking workload triggers optimization: profile attribution,
  before/after bottleneck evidence, and any serial-versus-parallel differential.
- Simulator `rocm-smi` registry/lease/epoch ON/OFF behavior with explicit
  absence of hardware device and production management-library access.
- N-rank collective correctness, ordering, CRC/credit, epoch abort/replay, and
  N=3 non-pair topology tests.

Large logs and artifacts are external/ignored. Evidence manifests store their
size and SHA-256, never an unbounded blob in Git. Historical commands whose raw
streams predate the capture helper must declare that limitation explicitly and
cannot independently support an acceptance claim; every new acceptance command
retains a command record plus hashed stdout and stderr through the capture
helper. The exact pre-freeze `SOURCE_LOCK.json` candidate is likewise preserved
as an ignored, hash-addressed artifact and must be bound by the checkpoint's
evidence manifest at acceptance.

## 8. Recovery and commit protocol

`state/current.json` and CP-0001..CP-0030 are retained as the historical
transaction ledger, not as the current scheduler. Current work resumes from
GOAL/PLAN priorities, present source and tests, and content-addressed evidence;
the stale CP-0030 pointer must not restart CP-0031/RMSNorm or override P8. A
historical source-freeze checkpoint carries an exact repository
map (`id`, path, baseline commit/tree/tag, head/tree, administrative Git path,
and clean state) that is verified against the immutable source lock. A
bitlesson is append-only and records the symptom, source evidence,
wrong assumption, decision, confidence, and affected commit range.

Historical immutability is proven from Git history, not trusted from the
current JSON.
The active `SOURCE_LOCK.json` must match the lock blob and checkpoint hash in
the CP-0002 coordinator commit. For each project-authored lane, the verifier
finds its first appearance in `PROJECT_LANES.json`, binds that historical blob
to the same commit's `Checkpoint-ID` and checkpoint hash, and requires the
current lane declaration to remain exact. A later checkpoint may append a new
lane but cannot redefine an existing baseline or move its introduction point.

`scripts/resume.sh --verify` is offline and read-only.  It refuses handoff if
the root is dirty or in merge/rebase/bisect, the current pointer/hash does not
match its checkpoint, required evidence is missing, a source lane is not a
registered clean repository at its locked commit, a forbidden artifact is
tracked/staged, or a transaction journal is unfinished.  `--online` is a
separate optional source-reachability check and may not be required for resume.

Cross-repository work uses a local fsync+rename journal:

1. From a clean, strictly verified accepted checkpoint, allocate exactly the
   next `Checkpoint-ID`, lock the participant set, and record each participant's
   starting commit/tree from the previous root gitlink (or explicit absence for
   a new lane).
2. Commit each changed child from its immutable baseline with the same audit
   trailers. Before the journal records a target, consolidate its incremental
   object closure into a non-thin pack, fsync objects/refs, issue a filesystem
   durability barrier, and revalidate the child identity.
3. Stage only the declared root allowlist. The gitlinks changed from the
   previous root must equal the participant set exactly. Persist the prepared
   root tree and index, issue the barrier, revalidate the tree and children, and
   only then advance the journal from `prepare` to `prepared`.
4. Make the single root coordinator commit, run the full acceptance verifier,
   persist the coordinator object closure/ref and every participant filesystem,
   and revalidate all identities before the journal can become `committed` and
   be retired. Destination-before-source directory fsync makes a crash-recovered
   duplicate journal name safe only when both copies are byte-identical.

A crash after a child commit leaves an auditable prepared object and never
causes an automatic reset. Pending diagnostics report each declared participant
relative to both its immutable initial and target identities. Detached HEAD,
linked-worktree administration, checkpoint sequence gaps, undeclared gitlink
changes, and a durability-barrier failure are fail-closed conditions.

Commit trailers are mandatory:

```text
Checkpoint-ID: CP-xxxx
Goal-ID: GSIM-001
Plan-Revision: <current-plan-revision>
Source-Lock-SHA256: <sha256>
Evidence-Manifest-SHA256: <sha256>
Change-Kind: baseline|bootstrap|source|code|test|lesson|checkpoint
Baseline-Commit: <sha-or-N/A>
```

## 9. Storage, licensing, and offline policy

The root `.gitignore`, pre-commit hook, CI, and resume verifier jointly reject
model formats, virtual environments, native build products, secrets, and files
over the configured size limit.  A model script resolves the official model
revision through `huggingface_hub`, downloads with resume support to `models/`
or an external cache, and verifies every recorded SHA-256.  If the official
endpoint is unavailable, the script must stop or require an explicitly marked
mirror; a mirror observation can never silently become an official lock.

ROCm components retain their own licenses/notices (ROCr NCSA, libhsakmt/CLR/
HIP MIT, RCCL notices, LLVM Apache-2.0 WITH LLVM-exception, gem5 BSD-3-Clause,
and all third-party terms).  New glue is GPL-3.0-or-later where compatible;
the distribution manifest will enumerate exceptions instead of flattening
licenses into a false aggregate claim.

## 10. Current handoff boundary

`CP-0007` retains the CP-0002 official Qwen and six-upstream source freeze plus
the CP-0003 authored runtime baseline. It advances only gem5 and
`self-amdgpu-runtime` from the CP-0006 simulated-memory boundary to the shared
bounded signal/event extension. The accepted gate preserves byte-exact
handshake, endpoint, deadline, failure, queue, memory, completion, and
N=1/2/3/4/8 isolation behavior and adds signed signal lifecycle, generation-safe
one-shot waits, event-queue completion, exact tick validation, bounded outbound
accounting, shared request correlation, and runtime retry/poison/atomicity
contracts. It makes no KFD event, GPU-visible signal memory, packet-visible VRAM,
SDMA timing, packet-submission, code-object, kernel, collective, PyTorch, or vLLM
execution claim.

`CP-0008` is accepted at the P1 dispatch boundary. It preserves the CP-0002
official Qwen and six-upstream source freeze, the CP-0003 authored runtime
baseline, and every CP4-CP7 transport gate while proving one source-pinned
gfx950/wave64, one-CU, one-workgroup dispatch through the real HSA packet and
GPU execution path. The accepted evidence binds queue, allocation, signal,
request, trace, packet, kernarg, VA, tick, statistics, and exact D2H identities;
it explicitly makes no generic code-object, ROCr/libhsakmt, HIP, OpenCL, Triton,
PyTorch, vLLM, multi-CU, collective, model, or performance claim.

`CP-0009` is accepted at the P2 ABI-inventory boundary. It binds the pinned
ROCr/libhsakmt source files, loader order and platform guards, status/layout
records, model ABI, and hardware/build exclusion audit. The child provider is
metadata-only: typed hsaKmt/DRM exports remain zero, `query_lifecycle` means
transport-open only, and no KFD/topology or production DSO is touched.

`CP-0010` is accepted at the typed-shim boundary. It implements the frozen
message types 14/15 and capability bit 5, 18 typed operations, fixed-width
owner/object generations, copied-buffer CRCs, per-provider operation
sequences, canonical gfx950 fixture validation, daemon-owned simulated
allocation/queue/event state, and deterministic unsupported/source-only DRM
statuses. The retained live smoke proves the runtime-to-gem5 envelope and
cleanup path only; it does not claim a complete 124-PFN ROCr provider, KFD
attach, topology discovery, HIP, OpenCL, Triton, PyTorch, or vLLM execution.

`CP-0011` is accepted at the source-locked code-object fixture boundary. It
binds two tracked gfx950 ELF V6 images, their MsgPack metadata, PT_LOAD and
relocation rules, exact protected code/descriptor symbols, hidden kernarg
holes, descriptor resources, and parser/gem5 provenance tests. It deliberately
does not claim code-object execution: the authority records the missing pinned
device-libs build, gfx950 decoder proof, unsupported `v_fmamk_f32`, and missing
reduction fixture as blockers. The next unique action is `P3-CODEOBJ-02`:
freeze the compiler/device-libs output and prove decoder/execution equivalence
before adding HIP/OpenCL or Triton launch claims.

`CP-0012` is accepted at the bounded toolchain and decoder boundary. It records
the pinned LLVM/device-libs build, two independent byte-identical gfx950
device-library manifests and HSACO outputs, the native gem5
`--linker=mold -j24` procedure, gfx942/gfx950 literal-FMA decoder aliases with
ISA-table reset isolation, and runtime-local selected-kernel materialization.
It does not claim HSACO upload, PT_LOAD mapping, dynamic AQL/kernarg creation,
generic gem5 code-object execution, or HIP/OpenCL/Triton support.

`CP-0013` is accepted as the A1 code-object transport/staging boundary. It
freezes the 4096-byte BEGIN/CHUNK/COMMIT envelope, capability and identity
fields, CRC-32C/SHA-256 rules, manifest/PT_LOAD metadata validation, owner and
generation scope, contiguous ordering, atomic failure cleanup, and the
zero-address A1 boundary. It proves copied bytes and digest-bound staging only;
PT_LOAD mapping, relocation application, dynamic AQL/kernarg, queue submission,
and gem5 execution remain separate gates. At CP-0013 acceptance, its recorded
next action was `P3-CODEOBJ-03-A2`; that loader gate remains an unproven later
prerequisite and is not the active CP-0014 next action.
`SOURCE_LOCK.json` remains byte-immutable, and existing `PROJECT_LANES`
declarations remain historically anchored and append-only.

`CP-0014` is accepted at `P3-HOST-NATIVE-01`, the source-grounded
architecture/dependency inventory boundary. It binds the reusable GPU/Vega,
HSA queue, host bridge, Triton AMD compiler/launcher, and runtime C ABI
surfaces, while recording the current `VEGA_X86`, `Process`, `ThreadContext`,
TLB, and SE translation blockers. EV-0037 retains the gem5 boundary inventory
tests (4/4), runtime CTest matrix (16/16), focused Clang ASAN boundary test
(1/1), clean child identities, and the CP14 prepare journal. It does not claim
a host-native execution, Triton E2E, hardware, timing, or performance result.
Its recorded `P3-HOST-NATIVE-02` action is completed by CP-0015 below.

`CP-0015` is accepted at `P3-HOST-NATIVE-02`, the standalone host-native
control-core/build boundary. The gem5 target uses `BUILD_ISA=n`,
`USE_X86_ISA=n`, and `BUILD_GPU=y`; the ELF audit finds only allowed host
libraries and no forbidden x86/Process/ThreadContext/GPU-pipeline symbols. Its
protocol, memory, queue, and signal self-tests pass, and the eight existing
VEGA_X86 host-state regression binaries remain green. This proves a control
plane and state-adapter boundary only: it does not map or execute HSACO, run a
GPU pipeline, or pass Triton. The next unique action is
`P3-HOST-NATIVE-03`: connect the accepted runtime protocol and CP13 loader to
the target and prove pinned gfx950 loader/dispatch parity.

`CP-0016` is accepted at the first functional-parity sub-gate of
`P3-HOST-NATIVE-03`. It adds `HostNativeMemoryContext` and page-lifetime
tokens while reusing the accepted sparse memory, queue, signal, and pinned
dispatch state. The standalone gfx950 fixture target passes the XOR oracle,
cross-allocation/unmapped/foreign-owner rejection, and cleanup lifetime checks;
the runtime probe recognizes two pinned gfx950 HSACO metadata records. This
does not map PT_LOAD segments, apply relocations, build dynamic AQL/kernarg,
connect HSAPacketProcessor/GPUDispatcher/ComputeUnit, execute instructions, or
pass Triton. The next unique action is `P3-HOST-NATIVE-03-A`.

`CP-0017` is accepted at the bounded PT_LOAD staging sub-gate. Its no-x86
`host_gpu_native_ptload_core` reuses the CP13 staged-image state and the
`HostGPUMemoryState`/`HostNativeMemoryContext` ledgers to materialize the locked
gfx950 `gpuReadWrite` image, check exact segment tuples, copy `filesz`, zero
`memsz-filesz`, bind descriptor/entry addresses, and retain leases across Busy
unmap attempts. The negative cases are pre-allocation malformed/unsupported
rejections; the mapper is fixture-scoped and does not enforce segment
permissions, apply relocations, construct kernarg/AQL, submit a queue, or
execute instructions. The next unique action is
`P3-HOST-NATIVE-03-B`: native translation plus dynamic AQL/kernarg parity.

`CP-0018` is accepted at the `P3-HOST-NATIVE-03-B0` dispatch-admission
sub-gate. The no-x86 `host_gpu_native_dispatch_admission_core` reuses the CP17
staged image and shared host memory/queue ledgers to validate the descriptor and
entry relation, pack/read back the 280-byte hidden kernarg, materialize a 64-byte
AQL packet, construct an `HSAQueueEntry`, and exercise queue-control and ordered
lifecycle-listener contracts. The extracted listener header preserves the legacy
`GPUDispatcher` object interface. B0 does not publish an HSA queue, instantiate
HSAPP/GPUCommandProcessor/GPUDispatcher/ComputeUnit, submit AQL to the GPU path,
fetch or retire instructions, or produce a GPU output/trace differential. The
static Qwen 15-contract gate and offline model hashes remain valid, but strict AMD
execution is blocked; Triton E2E and Qwen inference remain 0/1. The next unique
action is `P3-HOST-NATIVE-03-B1`, native address translation plus real
HSAPP/command-processor packet publication with no CU claim; B2 reserves the
four-workgroup/sixteen-wave CU differential.

`CP-0019` is accepted at the `P3-HOST-NATIVE-03-B1` native queue and
command-processor-core sub-gate. The no-x86 `host_gpu_native_b1_core` resolves
host-owned GPU VAs, registers and validates the 64-slot queue, publishes/rings
one 64-byte AQL packet, fetches it in order, reads the locked descriptor, MQD,
kernarg, and completion-signal object, and reuses `HSAQueueEntry` for one
extracted command-processor-core admission. `aql_submitted=true` is local to
that native core; neither legacy HSAPP nor the GPUCommandProcessor SimObject is
linked or instantiated. The host read index is unchanged, and packet retire,
signal decrement, GPUDispatcher/CU connection, wave start, instruction fetch,
ISA execution, and output differential all remain false. The Qwen model is
offline-ready and its 15-contract static gate remains valid, but Triton E2E and
Qwen inference remain 0/1. CP-0020 below closes the bounded execution gate.

`CP-0020` is accepted at `P3-HOST-NATIVE-03-B2` for the locked
`gpuReadWrite` HSACO. The same native queue/CP admission reaches the reused
`GPUDispatcher`, `Shader`, `ComputeUnit`, Vega decoder, and instruction path in
a runtime graph with no CPU, Process, Ruby, TLB, HSAPP, or
`GPUCommandProcessor` objects. The fixture runs four 256-item workgroups as
sixteen wave64 waves; the lifecycle listener observes 19 instruction-start
PCs per wave (304 total), independently matched by CU `numInstrExecuted=304`
and 16 completed waves. A/B/C are separate 4 KiB allocations and the exact
oracle is A unchanged, B=gid, C=A over all 1024 elements; packet retirement,
MQD read-index 0->1, direct-u64 signal 1->0, and pin release are ordered and
validated. This is one locked functional case only: general gfx950/arbitrary
HSACO, timing accuracy, fences/barriers, atomics, LDS/scratch, GPU TLB,
Ruby/coherence, HIP/OpenCL, and performance remain unproven. At the historical
CP20 boundary, Triton E2E and Qwen inference remained 0/1. Its next action was
the Triton vecadd launcher gate later bounded and accepted by CP28.

`CP-0021` accepts a narrower child-side prerequisite for that launcher gate.
The pinned unmodified tutorial and retained vecadd HSACO identities match, the
runtime parser accepts the Triton `amdgcn-amd-amdhsa-unknown-gfx950` spelling
and its DEFAULT-visible descriptor only in a metadata-only branch, and the
descriptor preload is 12 DWORD (48 bytes). Runtime CTest is 16/16 and the
focused code-object set is 4/4; caller-local code/kernarg preparation passes,
but compiler/JIT invocation, normal launcher, transport, simulator execution,
output differential, and fallback are all false. Public A1 mapping remains
zero-address and fixture-only. Triton E2E and Qwen inference remain 0/1. The
`CP-0022` accepts the independent payload-v2 codec/admission boundary: v1
framing and records remain byte-compatible, while owner-scoped MAP,
ALLOC_KERNARG, SUBMIT_AQL, and UNMAP records are fixed-width and canonically
padded. `CP-0023` adds an append-only runtime client API and a separate local
native adapter/state selftest. Runtime CTest passes 18/18; the gem5 protocol
suite passes 47/47 normally and under ASAN/UBSAN. The local native lifecycle
maps the object, allocates and publishes kernarg bytes, publishes/fetches AQL,
performs command-processor admission, retires, and unmaps with owner/generation
checks and cancel-fetch rollback. It deliberately stops before the live daemon:
capability bit 8 advertisement and MessageType 18 routing are false, as are
GPUDispatcher/CU execution, the normal Triton launcher, compiler/JIT, and
fallback. The 12-DWORD descriptor preload remains an explicit NOT_SUPPORTED
   boundary. CP-0024 adds an owner-bound MessageType 18 handler source path,
   MessageType 19 response plumbing, and a shared route-policy harness. Its
   live runtime-to-gem5 probe proves only the canonical unsupported-capability
   handshake and baseline reconnect while bit 8 remains unadvertised. CP-0025
   closes the positive control lifecycle: the daemon advertises and selects
   bit 8 with its dependencies, accepts logical alignment 8 over hidden page
   backing, binds existing v1 MEMORY_COPY_H2D to the owner allocation, builds
   and admits the AQL packet, sends a durable type-19 ACK, emits type-20
   retirement, unmaps, reclaims disconnected leases, and accepts a new owner.
   Packet CRC and nondecreasing lifecycle ticks are retained. This is native
   CP admission/retire only; launcher, compiler/JIT, GPUDispatcher/CU execution,
   output correctness, and fallback remain false at the CP25 boundary. Its
   historical next action was CP-0026 /
   `P5-TRITON-VECADD-05-GPU-EXECUTION`, which is accepted below.

   CP-0026 accepts the exact bit-9 execution extension while preserving bit-8 as
   the control contract. The negotiated bit is word 0 bit 9 (wire byte 1 bit 1)
   and requires bit 8 plus topology, queue, memory, signal, and code-object
   capabilities. The clean daemon route executes only the locked 5,528-byte
   gfx950 `gpuReadWrite` image with zero descriptor preload: four workgroups,
   sixteen wave64 waves, 304 instruction starts, exact A/B/C output, durable
   type-20, three D2H oracles, duplicate D2H, and UNMAP. The fsynced daemon
   trace, not endpoint assertion fields, establishes GPUDispatcher/CU execution
   and post-ACK quarantine cleanup. The wire signal field remains expected `1`;
   trace `1 -> 0` is the private native completion signal. This is a fixture and
   VEGA_X86 bridge proof, not generic gfx950/arbitrary HSACO or standalone
   no-x86 daemon proof. The historical 5,408-byte Triton artifact remains
   compile-only at this boundary. The
   CP-0027 then accepts `P4-OPENCL-E2E-01`: a normal executable linked only to
   the repository-local OpenCL/runtime stack compiles the exact 5,160-byte
   gfx950 `vecadd`, automatically manages gem5, executes four workgroups and
   sixteen waves, receives C-only D2H, validates bit-exact `C=A+B`, and exits
   cleanly without fallback. It remains one submit per OpenCL context. CP-0028
   then accepts the normal Triton Python path for the exact 5,384-byte pure-b010
   `add_kernel`: two deterministic launches in one process observe the same
   managed session, queue, signal, packet VA, and allocation IDs, and both
   produce bit-exact float32 `C=A+B` with zero fallback. The stable image hash,
   packet VA/CRC, trace, ticket, and dispatch IDs support an inference of the
	   same kernel source packet; the trace exposes no kernel mapping ID. The v6/v7
	   final-prefix attempts remain NON-PASSING; CP-0029 fixes the queue mock race and
	   accepts a fresh schema-8 repository-local prefix after independent installation,
	   workload, provenance, pollution, and active-isolation gates. CP-0030 accepts the
	   minimum BF16 SiluAndMul subgate for decode and masked-prefill with fresh/repeat
	   normal-Python execution, exact traces, finite output, zero mismatch, and zero
	   fallback. It does not accept either projection GEMM, a complete MLP contract, or
	   any complete model contract; P5-OPS-01 expands the broader operator matrix next.
	   Profiling is conditional on a demonstrated
operator-, layer-, or model-level bottleneck.

### P3H - host-native simulator and first Triton gate

The physical machine is already the host; launching a full `VEGA_X86` gem5
target to obtain the GPU model adds an avoidable x86 CPU/system cost. This
workstream creates a second front-end over the existing GPU functional core.
The gem5 front-end remains the behavioral reference and is not deleted. The
host-native front-end must reuse the existing GPU packet/queue path,
Vega ISA decoder and instruction classes, `GPUDispatcher`/`ComputeUnit`, and
the sparse memory/queue/signal state covered by CP8-CP13. New code is limited
to an event loop, host memory/page adapter, daemon lifecycle, and build/link
glue; it must not fork a second code-object ABI or silently use a CPU fallback.

The work is staged as follows:

1. `P3-HOST-NATIVE-01` is accepted by CP-0014: its source inventory freezes
   the dependency graph, reusable-core boundary, and compatibility matrix
   against the existing CP8/CP13 wire. It does not claim that the listed
   modules already compile as a host-native binary.
2. `P3-HOST-NATIVE-02` is accepted by CP-0015: its standalone control core
   links without `VEGA_X86`/X86 ISA objects and passes no-x86/no-production-DSO
   audits, with direct and legacy state regressions retained.
3. `P3-HOST-NATIVE-03` is split into explicit loader, admission, publication,
   and execution gates.
   CP-0016 proves only functional memory/dispatch parity. CP-0017-A connects
   the CP13 staged image to fixture-scoped PT_LOAD materialization and
   descriptor/entry validation. CP-0018-B0 adds only descriptor-derived
   kernarg/AQL admission and queue/lifecycle contract smoke. CP-0019-B1 adds
   no-x86 native address resolution plus extracted queue/command-processor-core
   publication, fetch, and admission. It deliberately does not instantiate the
   legacy HSAPP/GPUCommandProcessor SimObjects or claim CU execution.
   CP-0020 accepts this boundary for one locked fixture: four workgroups,
   sixteen wave64 waves, nineteen instruction-start PCs per wave, exact A/B/C
   output coverage, packet retirement, MQD update, and native signal completion.
   It does not establish generic gfx950, timing, fence/atomic, TLB/Ruby, or
   arbitrary HSACO semantics.
4. `P5-TRITON-VECADD-01` runs the tutorial-equivalent correctness path
   (`examples/triton/vecadd_correctness.py`) through the normal Triton launcher,
   with simulator device selection only. It retains compiler, HSACO digest,
   transport, dispatch, output, and CPU-fallback evidence. CP-0021 accepts
   only the compile/provenance prerequisite: the exact tutorial and HSACO
   identities, metadata, and caller-local materialization pass, while
   compiler/JIT, launcher, transport, execution, and fallback remain false.
   Triton end-to-end is still 0/1 at CP-0021. The exact LLVM/Triton pair has a temporary
   AMD-only overlay that produced the retained compile artifact; it is not yet
   a committed launcher or device execution path. CP-0022 freezes the wire-v2
   codec. CP-0023 adds the runtime client contract and local owner-bound native
   adapter/admission lifecycle, but no daemon advertises bit 8 or routes
   MessageType 18 and no GPU execution is claimed. CP-0024 adds the bounded
   handler/policy source and a negative live capability probe without selecting
   bit 8 or sending type 18. CP-0025 then accepts the real positive owner-bound
   daemon control lifecycle through bit-8 selection, logical-align-8 ALLOC,
   v1 H2D, native CP admission, type-19 ACK, type-20 retirement, cleanup, and
   reconnect. It does not issue work to GPUDispatcher/CU or validate output.
   CP-0026 then connects the locked `gpuReadWrite` route to GPUDispatcher/CU,
   and CP-0027 accepts the separate direct OpenCL `vecadd` executable with a
   managed gem5 lifecycle and output oracle. CP-0028 accepts normal Triton
   Python driver/JIT/launcher execution for one exact float32 vecadd, including
   preload-aware mapping and two-launch reuse. It does not accept a second
	   operator family. CP-0029 is accepted: it stabilizes the queue mock test and
	   produces a fresh schema-8 repository-local prefix with independent runtime,
	   OpenCL, Triton, provenance, pollution, and active-isolation evidence. CP-0030
	   accepts the model-required BF16 SiluAndMul decode and masked-prefill subgate;
	   P5-OPS-01 is the next coordinated gate for the broader model-operator manifest
	   and differential coverage.

This workstream is not a cycle-accurate replacement. Timing, wider operator
coverage, host-parallel threadblocks, HIP/OpenCL CTS, PyTorch, vLLM, and Qwen
remain later gates after the first vecadd differential result. Full ROCm and
OpenCL CTS are not prerequisites for the Qwen-specific operator gate. If a
reused gem5 module cannot be separated without changing semantics, keep the
adapter explicit and retain the gem5 path as the oracle.

The exact blank-context continuation contract remains in `GOAL.md`; the
machine-executable argv, prerequisites, expected gate, and rollback boundary
are in `state/current.json` and its referenced checkpoint.

## 0.4 Live-model problem-first checkpoint (2026-08-16)

This is the active execution strategy. A real unchanged-upstream SGLang
Qwen3.5-0.8B TP1 run is the diagnostic driver; acceptance-bundle formatting
and broad matrices are paused until the model reaches the next boundary.

1. Use the smallest real probe that reaches the failing boundary. After every
   fix, rerun that probe first, then resume the full model. Prefer startup,
   allocation, first GEMM/attention/cache, collective, and teardown failures
   over synthetic correctness grids.
2. Repair the lowest shared semantic layer. The current generic fixes are
   ROCr queue scratch admission/retry, gfx950 FLAT scratch signed offsets, and
   physical resident-wave scratch-slot mapping. No model, operator, tensor,
   kernel-PC, or framework branch is allowed.
3. The current host-dispatch profile is gfx950 wave64, one simulated CU,
   four SIMDs, eight wave slots per SIMD, and full-small main scratch. Large or
   reduced main scratch and alternate scratch require a real scoreboard lease
   allocator before being claimed; do not silently map them with the static
   tuple formula.
4. Once TP1 reaches stable generation, bring up the same unchanged upstream
   path at TP2 and TP4. Then run unchanged SGLang TP4, followed by the separate
   vLLM Qwen3.5-9B TP16 upstream `torch.compile` scale lane. Keep TP=2/4 model
   coverage; no TP=8 matrix is required.
5. Implement/finish the 16-slot simulator-aware `rocm-smi` facade after the
   next model boundary: ON only for a live identity-validated managed gem5
   lease, OFF otherwise, without opening KFD/DRM device nodes. Add SMI work
   without changing the runtime-gem5 bridge contract.
6. Preserve the pinned ROCm, PyTorch, Triton, vLLM, and SGLang source trees.
   Keep KMD-removal details in the self-runtime/bridge/domain layers. Remove
   legacy code only after the corresponding real model path and focused tests
   are green; never delete a fallback while it is the only diagnostic route.

The current full-model run is expected to take about 27 minutes to load the
0.8B checkpoint and a few minutes to reach the first new execution traceback.
Each subsequent shared-layer iteration is budgeted at roughly 30--45 minutes;
this is an estimate, not an acceptance promise.

## 0.6 Key-change pause checkpoint (2026-08-16 16:30)

The DS transpose family fix is complete and passed its architecture-derived
golden tests. The following real TP1 run reached ticket 1115 and exposed
gfx950 VOP3P word `0xd3b70000`, decoded by upstream LLVM as
`v_mfma_f32_32x32x16_bf16`. The generic decoder patch now covers the complete
six-op gfx950 BF16/F16/I8 overlap family using existing parameterized MFMA
execution classes and timing keys; `gpu_decoder.test.opt` passes 11/11 and the
full `gem5.opt` links.

The MFMA-fixed real rerun was intentionally stopped for a model-key change
after ticket 1006. Before the stop it completed weight loading, Mamba cache,
KV cache, and `Memory pool end`, with no fatal/panic/invalid-opcode/traceback.
It had not yet crossed the old ticket-1115 boundary, so real MFMA execution is
not accepted yet. Resume authority and the exact command are in
`CHECKPOINT.md`. The next action remains the same unchanged-upstream TP1 run;
do not begin topology, TP2/TP4, or broad evidence work until it generates and
tears down or exposes the next real shared-layer failure.

The latest TP1 rerun completed the 0.8B weight load, Mamba/KV pool, engine
startup, bounded Triton correctness autotune, and retired through ticket 1115,
including the former V_PACK, speculative S_TRAP, and zero-scratch failures.
The next real boundary is the gfx950 CDNA4 DS transpose-read family: the live
kernel reached `ds_read_b64_tr_b16`, whose constructors were missing the common
local-memory classification, while the existing B6/B16 cross-lane repacking
also contains family-level semantic defects.  Fix B4/B6/B8/B16 from the
architecture mapping with asymmetric golden vectors, then rebuild the matched
runtime/gem5 pair and rerun the same TP1 path.  Do not accept a flag-only fix or
add a kernel/model-specific branch.  The repository-owned demo also passes the
official upstream `watchdog_timeout` ServerArgs field with a 24-hour simulator
default; SGLang itself remains unchanged.

TP2/TP4 has one prerequisite before launch: the frozen HSA topology and the
runtime-gem5 KMT state still expose one GPU even though the topology generator
supports 1..16 and SMI exposes 16 observation slots.  Implement a single
product `gpu_count` setting (default 16) through the generated HSA topology,
self-runtime model provider, and per-GPU bridge KMT state.  Keep SMI ON/OFF
lease-backed and independent from device availability.  Do not fake extra
devices in SGLang, vLLM, or PyTorch.

## 0.5 Simulator autotune and profiling policy (2026-08-16)

Model bring-up must not use Triton's millisecond-budget benchmark loop against
cycle-level simulation. A measured 1.157 microsecond simulated kernel would
still request roughly 21,600 warmup launches and 86,400 measured launches under
the upstream 25/100 ms budgets, even after the current zero-duration HIP Event
bug is fixed.

The repository-owned `gemsim_hip` Triton entry point changes only the active
HIP driver's benchmark policy. It preserves the public `hip/gfx950` target and
the stock upstream AMD compiler. Its default correctness mode runs each
candidate exactly once followed by device synchronization and makes no
performance claim. Simulator cache isolation uses a dedicated
`TRITON_CACHE_DIR`, not a different target identity. Device benchmark mode
continues to delegate to upstream HIP and is not the model bring-up default.

In parallel, repair `AMDKFD_IOC_GET_CLOCK_COUNTERS` and `amd_signal_t`
start/end timestamps in the KMT/runtime-gem5 path. All four counters and event
timestamps must share the gem5 simulated-nanosecond domain; do not use host
wall clock as a substitute. Once that ABI is correct, an optional bounded
timing mode may compare a small fixed number of launches. Performance claims
still require that scoped measurement or an upstream AMD hardware tuning
result. Never infer performance from correctness-mode ties.
