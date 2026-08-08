# amdgpu-sim implementation plan

**Plan ID:** `AMDGPU-SIM-V1`

**Revision:** `2`

**Revision date:** `2026-08-09`

**State at this commit:** `P2-KMT-ABI-01 accepted; next-P2-KMT-ABI-02`

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

Use the pinned ROCm LLVM/Clang/device-libs to produce gfx950 HSACO. Reuse ROCr
ELF V4–V6/MsgPack metadata parsing, relocations, hidden arguments, kernarg
alignment, LDS/scratch descriptors, and symbol lookup. Audit emitted ISA against
gem5's supported opcode/features. Gate: deterministic launch metadata and
vector/reduction fixtures match a reference.

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
If every remaining row closes in exactly one distinct transaction (checkpoint
IDs are not merged across rows), the nine accepted checkpoints plus these
fourteen forecast gates imply a minimum of twenty-three checkpoints overall;
this is a conditional lower bound, not a promise to compress later work into
those IDs. A difficult row may split into additional checkpoints.

| Forecast gate | Earliest phase | Result |
| --- | --- | --- |
| CP-0008 | P1 | One traced real CU-backed pinned dispatch |
| P2-KMT-ABI-01 | P2 | Source-exact thunk/libhsakmt ABI inventory and provider skeleton |
| P2-KMT-ABI-02 | P2 | Typed libhsakmt shim and versioned daemon KFD/DRM operation envelope |
| P3-CODEOBJ-01 | P3 | Pinned gfx950 code-object and kernarg ABI fixtures |
| P4-HIP-01 | P4 | Minimal transparent HIP/OpenCL launch surface |
| P5-TRITON-VECADD-01 | P5 | Unmodified Triton tutorial vecadd through GemSim |
| P5-PROFILE-01 | P5A | Retained profile, ranked 80/20 bottlenecks, and at least one measured optimization with before/after evidence |
| P5-PARALLEL-TB-01 | P5B | Safe serial-versus-parallel threadblock experiment |
| P5-OPS-01 | P5C | Broader model-operator manifest and differential gates |
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

The next unique action is `P2-KMT-ABI-02`: create the next transaction and
implement the typed libhsakmt shim plus a versioned daemon KFD/DRM operation
envelope. The gate must prove fixed-width pointer/buffer translation,
owner/generation and status precedence, unsupported-call atomicity, and
no-device/no-production-DSO behavior before making a provider-attach claim.
`SOURCE_LOCK.json` remains byte-immutable, and existing `PROJECT_LANES`
declarations remain historically anchored and append-only.

The exact blank-context continuation contract remains in `GOAL.md`; the
machine-executable argv, prerequisites, expected gate, and rollback boundary
are in `state/current.json` and its referenced checkpoint.
