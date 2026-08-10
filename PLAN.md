# amdgpu-sim implementation plan

**Plan ID:** `AMDGPU-SIM-V1`

**Revision:** `2`

**Revision date:** `2026-08-10`

**State at this commit:** `CP-0024 bounded daemon-handler boundary accepted; bit8 positive route, submit lifecycle, and normal launcher blocked; next-CP-0025 P5-TRITON-VECADD-04-DAEMON-LIFECYCLE`

## 1. Outcome and non-negotiable invariants

The deliverable is an industrial, versioned host-side simulator stack, not a
mock demo.  A host process must be able to compile and launch real AMDGPU
HSACO/code objects and have their work executed by gem5's GPU model while the
host has no AMDGPU device and no production KMD/UMD runtime loaded.

The final anchor is:

1. Official `Qwen/Qwen3.5-0.8B`, text-only path (vision is a later extension).
2. vLLM tensor parallelism `TP=2` across two independently running gem5
   daemon processes, one rank per daemon.
3. Complete prefill followed by at least one greedy decode token.
4. Exact selected token ID agreement with a pinned reference, plus layered
   numerical checks and trace evidence.
5. Zero host CPU arithmetic fallback and zero loads/accesses of
   `libamdhip64.so`, `libhsa-runtime64.so`, `/dev/kfd`, or `/dev/dri`.

The transport, handles, collectives, failure epochs, and rank registry are
parameterized by `world_size=N`; TP=4 and TP=8 are later acceptance targets,
not a reason to introduce a TP=2-only protocol.  Gloo/TCPStore may carry
control metadata, rendezvous, and barriers only.  Tensor values and reductions
must travel through the GemSim transport and execute reduction kernels in
gem5.  A CPU reduction oracle is diagnostic-only and is counted; it cannot
silently satisfy an acceptance run.

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
| Runtime cut | Fork/reuse ROCr core and HIP/CLR semantics; provide a complete compatible `libhsakmt` provider backed by GemSim RPC | Linux KFD is a semantic reference only, not a user-space link target |
| HIP ABI | Compatible self-built HIP/ROCr-facing ABI selected by bundle search paths | No production AMD UMD/KMD binaries or `/dev/kfd` |
| Rank topology | N daemons + supervisor, one vLLM rank per daemon | Independent VA/queue/signal namespaces; generation-checked handles |
| Collectives | RCCL-compatible semantics with GemSim functional transport first; device reduction kernels in gem5 | Host only stages/copies bytes |
| Fabric | Functional shared-memory/Unix transport first, timed xGMI/SDMA model later | No timing claims before the timed phase |
| PyTorch | Transparent HIP-compatible path is primary; optional PrivateUse1 adapter is isolated behind the same facade | No backend-specific business logic spread through vLLM |
| Triton | Reuse AMD lowering/compiler; add an out-of-tree `gemsim_amd` driver and launcher | Cache keys include every toolchain/runtime/simulator revision |
| Performance | Profile the first transparent Triton vecadd before optimizing; prioritize measured 80/20 bottlenecks | No optimization claim without retained before/after profiles and unchanged correctness |
| Host parallelism | Permit CPU-parallel threadblock simulation only with explicit dependency, barrier, atomic, and determinism gates | Host utilization never overrides simulated ordering or synchronization semantics |
| Status tooling | Provide a simulator-aware `rocm-smi` view from the daemon registry | Report simulated instances only; never probe hardware nodes or load production SMI libraries |
| Precision | Bitwise for copies/integers; explicit per-op/layer BF16/FP16 tolerances; exact final greedy token | No claim of all-float bitwise identity |
| Failure | Abort the whole job epoch, checkpoint at request/layer boundaries, deterministic replay | No elastic shrink in the anchor |
| Distribution | Root coordinator plus standard submodule gitlinks and immutable child baseline tags | We never rewrite upstream history |
| Artifacts | Weights, environments, binaries, caches and logs are ignored; scripts fetch/rebuild by fixed revision and hash | Offline execution after preparation is required |
| License | GPL-3.0-or-later for new glue/aggregate where legally possible; preserve every upstream license and notice | No license header is removed or relicensed casually |

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
state/current.json              the sole resume pointer
state/checkpoints/              append-only machine checkpoints
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
allocation rules wherever possible.  Replace the KFD/DRM provider, not every
upper-runtime algorithm.  The provider must export the complete symbol surface
that the pinned ROCr `ThunkLoader` resolves; an unsupported capability returns
the documented `HSAKMT_STATUS_NOT_SUPPORTED` (or equivalent), never a fake
success.  Preserve exact layouts and widths for queue resources, doorbells,
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
and the 1.1 model interface. Its provider implementation is metadata-only plus
the established transport-open handshake and deterministic NOT_SUPPORTED
boundary; it exports zero typed hsaKmt/DRM functions and makes no KFD or
topology claim. `P2-KMT-ABI-02` is the next gate: implement a typed,
source-compatible libhsakmt shim and a versioned daemon KFD/DRM operation
envelope with fixed-width IDs, copied buffers, ownership/generation checks,
and an explicit no-device/no-production-DSO audit. Do not cast wire integers
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
separate OpenCL ICD-compatible shim and compiler path. Gate: hipcc and OpenCL
conformance subsets run against gem5 with production runtime/device opens absent.

### P5 — Triton operators

Keep AMD TTIR/TTGIR/LLVM lowering; add an out-of-tree `gemsim_amd` backend and
launcher linked only to the stable runtime ABI. Cache keys contain gem5/runtime/
ISA/LLVM/Triton/device-lib revisions and capability bits. The first gate is
unmodified execution of Triton's tutorial vecadd with no fallback. Broader
operator expansion waits for the P5A profile/optimization gate and the P5B
parallelism feasibility decision so later work benefits from early simulator
speedups rather than repeatedly paying an already visible bottleneck.

### P5A — transparent vecadd and measured operator optimization

The first Triton usability checkpoint is unmodified user execution of the
`python tutorial/01-vecadd.py` request through `gemsim_amd`. In the pinned
Triton checkout this request maps to `python/tutorials/01-vector-add.py`; the
upstream file is not edited. The ordinary Triton driver/launcher selection path
must be used, with no application-specific import, source, or environment
rewrite beyond selecting the simulator device. After that gate is accepted,
retain a reproducible profile spanning host protocol, event queue,
AQL/dispatch, CU pipelines, memory translation/cache, and trace emission.
Rank bottlenecks by wall-clock contribution and optimize the smallest set that
explains most runtime. Every optimization records the before/after profile,
host and simulated work counts, output oracle, deterministic replay, and CPU
fallback count. Micro-optimizations outside the dominant path are deferred.

### P5B — correctness-preserving host-parallel threadblocks

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

### P5C — broader Triton operator matrix

After P5A and P5B are accepted (including an evidence-backed decision to retain
serial execution where parallelism is unsafe), expand the generated operator
manifest and differential tests through elementwise, reductions, LDS/barriers,
atomics, MFMA/GEMM, embedding, RMSNorm, MLP, RoPE, GDN, paged attention, logits,
and sampling. Each entry keeps its compiler/runtime/simulator identity and
falls back only by explicit unsupported status, never host arithmetic.

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
The temporary pinned Triton/LLVM overlay has compile-only gfx950 vecadd
HSACO SHA-256
`ee8b0f892da7ab1886f17ee66f88de5c23e05a48f7f361e02bd0707c9a11826e`;
this artifact is not an execution result.

### P6 — PyTorch integration

Build PyTorch against the compatible HIP ABI (with an isolated PrivateUse1
facade for APIs that cannot be made transparent). Implement allocator,
DeviceGuard, streams/events, storage/copy/view, RNG, serialization, AMP and
Inductor/Triton registration. Gate eager and compile paths with CPU fallback
counter zero in acceptance mode.

### P7 — single-device vLLM

Provide a `gemsim` platform plugin/worker/communicator and reuse the official
Qwen3.5 model implementation. Disable CUDA graphs, unsupported quantization,
speculative decode and other features until capability-gated. First validate
individual GDN and full-attention layers, then 24-layer prefill and decode on a
1-CU quick config, followed by a full MI355X-config single daemon. Gate: same
checkpoint/tokenizer/template and exact greedy token against the reference.

### P8 — N-rank transport and RCCL semantics

Implement rank leases, epochs, allocation namespaces, ring/tree planners,
collective tracing, failure/replay, and functional FabricModel. Gate synthetic
N=2, N=3, N=4, and N=8 collectives, including non-power-of-two ranks and
negative/error/restart cases; host reduction counter remains zero.

### P9 — TP=2 Qwen anchor (critical milestone)

Launch two independent gem5 daemons and two vLLM ranks. Implement exact Qwen
sharding: vocabulary/LM-head tying, attention Q/KV head rules, row/column MLP
and projection allreduces, and the GDN convolution/QKVZ/BA/state sharding.
Run full prefill then one or more greedy decode tokens and compare token IDs,
selected logits, hidden-state/layer checkpoints, trace completeness, and
fallback/device-open audits. This is the first “usable” release; earlier
phases are prerequisites, not a claimed finished product.

### P10 — TP=4/8 generalization

Run actual N-daemon jobs and collective suites for TP=4 and TP=8. Exercise
replicated KV-head rules and all shard divisibility constraints. Run the full
Qwen token acceptance whenever the official tensor shapes permit it; otherwise
fail explicitly with a recorded shape/semantic reason and retain the protocol
acceptance.

### P11 — timing, breadth, and hardening

Add timed xGMI/SDMA/FabricModel, larger MI355X configurations, performance
metrics, broader HIP/OpenCL surface, images/interop/debug features, packaging,
upgrade compatibility matrices, fuzzing, fault injection, and reproducible
offline bundles. No timing result is published from P8/P9 functional mode.

### P11A — simulator-aware `rocm-smi`

After the single-device model path and N-rank registry are usable, add a
low-priority status tool with familiar `rocm-smi` discovery and display
conventions backed solely by the GemSim supervisor/daemon registry. With
multiple gem5 processes running it
reports each simulated card's stable UUID/rank/epoch and ON state; after an
orderly stop, crash, lease expiry, or stale epoch it reports OFF without
inventing temperature, power, clocks, utilization, or hardware health. The
tool must not open `/dev/kfd` or `/dev/dri`, load ROCm SMI libraries, or imply
compatibility with physical-card management operations.

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
| P4-HIP-01 | P4 | Minimal transparent HIP/OpenCL launch surface |
| P5-TRITON-VECADD-01 | P5 | Unmodified Triton tutorial vecadd through GemSim |
| CP-0021 | P5 | Hash-bound Triton vecadd compile/provenance prerequisite; normal launcher remains blocked |
| CP-0022 | P5 | Generic payload-v2 codec/admission boundary; daemon mapping and launcher handoff remain blocked |
| CP-0023 / P5-TRITON-VECADD-02-RUNTIME | P5 | Owner-bound runtime client plus local native MAP/ALLOC/publish/fetch/CP-admission/retire/unmap; daemon route remains blocked |
| CP-0024 / P5-TRITON-VECADD-03-DAEMON-ROUTE | P5 | Bounded handler source/type-19 plumbing, route-policy harness, and live canonical negative handshake; bit 8 and positive type-18/H2D/SUBMIT route remain blocked |
| CP-0025 / P5-TRITON-VECADD-04-DAEMON-LIFECYCLE | P5 | Resolve alignment and owner-bound v1 H2D, then wire queue/signal/AQL packet/ticks, SUBMIT ACK, and type-20 completion before reconsidering bit 8 |
| P5-PROFILE-01 | P5A | Retained profile, ranked 80/20 bottlenecks, and at least one measured optimization with before/after evidence |
| P5-PARALLEL-TB-01 | P5B | Safe serial-versus-parallel threadblock experiment |
| P5-OPS-01 | P5C | Broader model-operator manifest and differential gates |
| P5-QWEN35-OPS-01 | P5C | All 15 text-only Qwen3.5-0.8B operator contracts execute on AMD with no fallback |
| P6-PYTORCH-01 | P6 | PyTorch eager/compile device foundation |
| P7-VLLM-SINGLE-01 | P7 | Single-daemon Qwen model path |
| P8-COLLECTIVE-01 | P8 | N-rank functional collective semantics |
| P9-QWEN-TP2-01 | P9 | Full TP=2 prefill and greedy decode anchor |
| P10-QWEN-TP48-01 | P10 | TP=4/8 protocol and permitted model gates |
| P11-SMI-01 | P11A | At least two concurrent simulated gem5 cards with ON/OFF status tool |
| P11-HARDEN-01 | P11 | Timing, breadth, packaging, and fault hardening |

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
- Triton vecadd transparency, profile attribution, before/after bottleneck
  evidence, and serial-versus-parallel threadblock differential checks.
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

`state/current.json` is the only resume pointer.  A checkpoint names exactly
one next action with cwd, argv, non-empty prerequisites, expected gate, and
rollback boundary. A source-freeze checkpoint also carries an exact repository
map (`id`, path, baseline commit/tree/tag, head/tree, administrative Git path,
and clean state) that is verified against the immutable source lock. A
bitlesson is append-only and records the symptom, source evidence,
wrong assumption, decision, confidence, and affected commit range.

Immutability is proven from Git history, not trusted from the current JSON.
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
Ruby/coherence, HIP/OpenCL, and performance remain unproven. Triton E2E and
Qwen inference remain 0/1. The next unique action is
`P5-TRITON-VECADD-01`, the unmodified Triton tutorial through the normal
launcher with simulator device selection only.

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
   handshake and baseline reconnect while bit 8 remains unadvertised; it does
   not send MessageType 18. A positive socket/H2D route, normal alignment 8,
   SUBMIT ACK, MessageType 20 completion, launcher, compiler/JIT, execution,
   and fallback remain false. The next unique action is CP-0025 /
   `P5-TRITON-VECADD-04-DAEMON-LIFECYCLE`.

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
4. `P5-TRITON-VECADD-01` runs the pinned, unmodified Triton tutorial request
   (`python/tutorials/01-vector-add.py`) through the normal Triton launcher,
   with simulator device selection only. It retains compiler, HSACO digest,
   transport, dispatch, output, and CPU-fallback evidence. CP-0021 accepts
   only the compile/provenance prerequisite: the exact tutorial and HSACO
   identities, metadata, and caller-local materialization pass, while
   compiler/JIT, launcher, transport, execution, and fallback remain false.
   Triton end-to-end is still 0/1. The exact LLVM/Triton pair has a temporary
   AMD-only overlay that produced the retained compile artifact; it is not yet
   a committed launcher or device execution path. CP-0022 freezes the wire-v2
   codec. CP-0023 adds the runtime client contract and local owner-bound native
   adapter/admission lifecycle, but no daemon advertises bit 8 or routes
   MessageType 18 and no GPU execution is claimed. CP-0024 adds the bounded
   handler/policy source and a negative live capability probe without selecting
   bit 8 or sending type 18. CP-0025 /
   `P5-TRITON-VECADD-04-DAEMON-LIFECYCLE` is the next coordinated daemon gate:
   resolve ALLOC alignment and owner-bound v1 H2D, then implement queue/signal/
   packet/tick submission and type-20 completion before advertisement.

This workstream is not a cycle-accurate replacement. Timing, wider operator
coverage, host-parallel threadblocks, HIP/OpenCL CTS, PyTorch, vLLM, and Qwen
remain later gates after the first vecadd differential result. Full ROCm and
OpenCL CTS are not prerequisites for the Qwen-specific operator gate. If a
reused gem5 module cannot be separated without changing semantics, keep the
adapter explicit and retain the gem5 path as the oracle.

The exact blank-context continuation contract remains in `GOAL.md`; the
machine-executable argv, prerequisites, expected gate, and rollback boundary
are in `state/current.json` and its referenced checkpoint.
