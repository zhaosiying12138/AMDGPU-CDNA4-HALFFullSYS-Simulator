# amdgpu-sim implementation plan

**Plan ID:** `AMDGPU-SIM-V1`  
**Revision:** `1`  
**Revision date:** `2026-08-07`  
**State at this commit:** `P0-control-plane-complete; paused-for-handoff`

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
| Precision | Bitwise for copies/integers; explicit per-op/layer BF16/FP16 tolerances; exact final greedy token | No claim of all-float bitwise identity |
| Failure | Abort the whole job epoch, checkpoint at request/layer boundaries, deterministic replay | No elastic shrink in the anchor |
| Distribution | Root control repo plus independent upstream Git lanes with immutable baseline tags | We never rewrite upstream history |
| Artifacts | Weights, environments, binaries, caches and logs are ignored; scripts fetch/rebuild by fixed revision and hash | Offline execution after preparation is required |
| License | GPL-3.0-or-later for new glue/aggregate where legally possible; preserve every upstream license and notice | No license header is removed or relicensed casually |

## 3. Repository and source layout

```text
PLAN.md                         complete staged plan
GOAL.md                         human-readable acceptance contract
SOURCE_LOCK.json                observed and then frozen upstream revisions
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
source repositories preserve their upstream history.  A pristine checkout is
tagged `upstream-baseline/<lane>/<sha>` (or recorded as a gitlink); our work
branches and commits must descend from that exact object.

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

### P0 — control plane and source freeze (current bootstrap, then next run)

Deliver PLAN/GOAL/recovery scripts, root initial commit, exact source lock,
upstream checkouts and immutable baseline tags, license inventory, model-fetch
script, and a clean offline verification gate.  The current commit stops after
the control plane; the next unique action is `P0-SRC-01` in `state/current.json`.

### P1 — gem5 host bridge and one daemon

Add the bridge SimObject, request/event protocol, simulated memory/VA ownership,
queue/signal lifecycle, code-object handoff, tracing, and a 1-CU fast config.
Gate: a host client allocates, copies, launches a trivial kernel, waits, and
verifies bytes; no guest KMD/UMD and no host arithmetic.

### P2 — ROCr/libhsakmt provider

Fork the pinned ROCr core and implement `libhsakmt_gem5` against the daemon.
Audit every loaded symbol and ABI structure against the exact source revision.
Gate: lifecycle, memory, queue, signal/event, pointer-info, and error-path
conformance tests pass under a loader audit with no `/dev/kfd`/`/dev/dri`.

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
ISA/LLVM/Triton/device-lib revisions and capability bits. Gate the progressive
operator manifest and differential tests for all model primitives.

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
- N-rank collective correctness, ordering, CRC/credit, epoch abort/replay, and
  N=3 non-pair topology tests.

Large logs and artifacts are external/ignored. Evidence manifests store their
size and SHA-256, never an unbounded blob in Git.

## 8. Recovery and commit protocol

`state/current.json` is the only resume pointer.  A checkpoint names exactly
one next action with cwd, argv, prerequisites, expected gate, and rollback
boundary.  A bitlesson is append-only and records the symptom, source evidence,
wrong assumption, decision, confidence, and affected commit range.

`scripts/resume.sh --verify` is offline and read-only.  It refuses handoff if
the root is dirty or in merge/rebase/bisect, the current pointer/hash does not
match its checkpoint, required evidence is missing, a source lane is not a
registered clean repository at its locked commit, a forbidden artifact is
tracked/staged, or a transaction journal is unfinished.  `--online` is a
separate optional source-reachability check and may not be required for resume.

Cross-repository work uses a local fsync+rename journal:

1. Lock a `Checkpoint-ID` and record the previous root head and expected child
   heads (`prepare`).
2. Commit each changed child from its immutable baseline, with the same trailers.
3. Verify children, stage only the allowlisted gitlinks/manifests/state/evidence,
   and make one root coordinator commit (`commit`).
4. Verify the root ref, then retire the journal.  A crash after step 2 leaves
   auditable prepared child commits; resume never resets them automatically and
   reports whether finalization is safe.

Commit trailers are mandatory:

```text
Checkpoint-ID: CP-xxxx
Goal-ID: GSIM-001
Plan-Revision: 1
Source-Lock-SHA256: <sha256>
Evidence-Manifest-SHA256: <sha256>
Change-Kind: bootstrap|source|code|test|lesson|checkpoint
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

This bootstrap commit contains no upstream checkout and no implementation
claim.  It establishes the plan, goal, lock schema, recovery verifier, hooks,
and the first checkpoint.  The next unique action is:

```text
P0-SRC-01: run scripts/resolve_sources.sh --online, revalidate every official
remote HEAD and the official Hugging Face model revision, write immutable
SOURCE_LOCK.json (including tree hashes), then clone the three source lanes and
the ML lanes with their pristine upstream history before any local patch.
```

After this commit the requested workflow intentionally pauses for a model/token
handoff.  The exact blank-context continuation prompt is in `GOAL.md` and
`state/current.json`.
