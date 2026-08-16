# Runtime-gem5 bridge migration plan

## Objective

Concentrate every consequence of removing the Linux KMD into one
runtime-gem5 bridge while preserving two stable facades:

- upward: the versioned `self-amdgpu-runtime` public API used by OpenCL and the
  out-of-tree Triton backend;
- downward: a small GPU-domain gem5 adapter used to allocate/copy memory, map
  code objects, publish queue/kernarg state, dispatch, retire, and clean up.

Pinned vLLM, PyTorch, Triton core/AMD lowering, and gem5 GPU/Vega/CU internals
remain unchanged. Users activate one repository-owned conda environment and
run ordinary Python demos directly. The current implementation is backed up
and used as the numerical, lifecycle, trace, and performance regression
baseline; it does not constrain the final file layout.

## Source backup

The migration baseline is
`artifacts/source-backups/20260813T101500Z-layering-migration-baseline`. It
contains binary Git patches and untracked-source archives for root, gem5, and
self-runtime. Its canonical `manifest.json` records every relative path, byte
count, and SHA-256, and the normal layering test rehashes all six artifacts.
Build trees, toolchain prefixes, model weights, and accepted evidence are not
duplicated.

## Short-cycle plan

### M0: contract and enforcement

Deliverables:

- `docs/framework-runtime-layering.md` and this plan;
- a static ownership test that requires pinned vLLM/PyTorch/Triton trees clean,
  forbids GemSim references in Triton core, forbids production writes to
  vLLM private parallel state, and verifies the project backend/plugin base
  classes and entry points;
- an inventory mapping every current runtime/gem5 protocol, handler, state
  object, fixture route, launcher, and test to its owning future layer.

Exit: documentation and static tests pass; every existing file is classified
as keep, migrate, compatibility shim, evidence-only, or delete-candidate.

Status: complete. The ownership tests and machine-readable inventory are part
of the normal host test suite, and the migration baseline is recoverable from
the source-backup manifest above.

### M1: repository-owned conda product candidate

Create `env/conda/product-v1-<identity>` using an explicit conda lock and exact
package hashes. Install or link the already pinned CPU PyTorch, vLLM export,
Triton build, `gemsim_amd`, framework plugin, CCL plugin, self-runtime DSO, and
product manifest without editable packages or workspace `PYTHONPATH`.

The activation command and direct entry points are:

```bash
./scripts/setup_conda_env.sh --install
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$(./scripts/setup_conda_env.sh --print-prefix)"
python examples/quickstart/triton_vecadd.py
python examples/quickstart/vllm_silu.py
rocm-smi
```

The scripts may bootstrap private socket/cache directories and gem5 processes,
but users do not call transport helpers. CUDA/NVIDIA/host conda variables and
DSOs are scrubbed. The old product stays selectable until the candidate passes.

Exit: conda create/verify is reproducible and idempotent; fresh-shell direct
Triton and vLLM import/smoke pass with exact installed-file identities and zero
editable packages.

Status: complete for the entry-point and simulator-inventory scope. The active
immutable conda product is
`e40b50b3162467827b7e045f0ee38244797dfd1b1d061624e38cb316e776738d`,
bound to native product
`e016ae502aeb9ebd0d64e4d09fbe521f95bf818d081005d98b97412d51224833`,
gem5 SHA-256
`fec967aa97782929394a6f4d64a0b42f1c18e38e5526c5af0ead39675e95d68f`,
and runtime DSO SHA-256
`4ba08ff9d39b6fd175c82ba62e46b344354e14532baf344995894f779b356d40`.
Both direct quickstarts pass after ordinary conda activation, with ordinary
wheel-installed project plugins and no editable workspace package. The same
product installs `rocm-smi`/`gemsim-smi`; a real managed session passed
`0 ON, 16 OFF -> 1 ON, 15 OFF -> 0 ON, 16 OFF` with no residual process.

### M2: canonical bridge contract and adapter seam

Introduce a project-authored, fixed-width canonical contract that is the
single source for runtime and gem5 wire constants, layouts, capabilities,
statuses, and codecs. Prefer generated C/C++ declarations and golden codec
vectors over hand-maintained duplicate structs. Public runtime symbols and
wire behavior remain byte-compatible during migration.

Add a gem5-side adapter interface containing only GPU-domain operations. The
existing bridge initially implements the interface without behavior change;
GPU dispatcher, CU, Vega ISA, and functional memory call sites remain stable.

Exit: codec roundtrip/corruption tests use shared vectors on both endpoints;
the old binary protocol suite and one normal vecadd trace remain exact.

No operation family introduces an operator-name, code-hash, tensor-shape,
model, oracle, or fixed-PC discriminator. Before the adapter seam is accepted,
two structurally different retained kernels and one previously unseen valid
kernel traverse it without a production source edit.

Status: implementation, host/build, and initial live gates complete. Runtime and gem5 consume
generated declarations from one schema and share fixed binary request/response
vectors. On the gem5 side, `host_gpu_domain_types.hh` is independent of the
wire protocol and `host_gpu_protocol_adapter.{hh,cc}` is the only wire/domain
converter. The rebuilt full `gem5.opt`, adapter tests, protocol tests, vecadd
code object, and silu-and-mul code object pass. A fresh source-compiled OpenCL
vecadd run and fresh direct Triton vecadd, vLLM SiluAndMul, and two-launch RMS
norm runs pass with the rebuilt immutable product. The OpenCL trace contains
one retired record, one type-20 durable record, and one clean session-complete
record; the RMS norm trace contains two retired records with one reuse handoff
and a clean session-complete record. These are operation-family regression
gates, not full M4 model or collective acceptance.

### M3: three vertical operation families

Migrate in this order, one reviewable patch and gate per family:

1. memory ownership, allocate/free, H2D/D2H;
2. code-object upload/map/unmap and kernarg lifetime;
3. queue publication, dispatch, completion, timeout/error, and cleanup.

Each patch removes its old duplicate only after both old and new implementations
pass the same unit/fault tests and one real workload. No framework, Triton core,
public runtime ABI, or gem5 GPU core edits are permitted.

If a workload fails, identify and repair the generic violated contract. A
per-operator bridge route, simulator callback, fixture hash, output formula, or
incremental list of accepted shapes is an automatic rejection and rollback.

Exit per family: output SHA, normalized trace, failure status, allocation/FD/
process cleanup, and simulator instruction/tick semantics match the declared
baseline. Host-validation timing is recorded as a regression signal, not
optimized in this migration.

Status: the three generic-dispatch domain implementations and their direct production callers
are migrated and compile without protocol dependencies. Memory, code-object,
mapper/kernarg, native command-processor, and generic dispatch-v2 focused tests
pass; the full gem5 binary links. Pure-domain PT_LOAD/Triton/memory/code-object
targets no longer link the wire codec. Legacy control-only queue/signal and
pinned-dispatch evidence state still use wire DTOs and remain migration or M5
compatibility work; they are not the accepted generic execution state. Fresh
OpenCL, Triton, framework operators, the full live-input 24-layer chain, final
norm, CCL, and production runs now bind output, retire/durability, reuse, and
cleanup to the rebuilt binary. Formal TP-layer execution remains in M4.

The generic dispatch state also rejects an aligned kernarg subrange unless its
complete half-open range was published; checked containment prevents unsigned
underflow when an offset lies above a published end. Native memory reads and
writes are owner/request-bound and range-restricted throughout the complete
execution binding lifetime, including prepare and retire edges, rather than
only while a kernel is active. These are shared execution-contract fixes, not
operator-specific admissions.

The unchanged-upstream ROCr lane now reaches the same domain seam. The model DSO
exports one process backing and reports standard doorbell publication; gem5
maps owner-scoped KMT allocation offsets, observes one or more live AQL packets,
and builds a loader-independent `ResidentKernelView`. The command processor no
longer depends on `HostNativeMemoryContext` or the code-object mapper: it reads
the AMD kernel descriptor, kernarg, completion signal object and MQD from the
stable `GPUNativeMemory` interface. The explicit mapper only constructs the
same resident view and supplies an additional exact metadata expectation.
Focused tests reject stale allocation generations and descriptor/entry ranges
outside the resident allocation. CU execution and retirement of a packet from
this standard ROCr source remain the next shared-lifecycle gate.

### M4: full regression and activation

Run, in order:

1. runtime API/codec/state/fault tests;
2. gem5 bridge/adapter/ISA tests;
3. normal OpenCL and direct Triton operator gates;
4. framework plugin/operator and single-device layer/model gates;
5. CCL N=2/3/4/8/16 and vLLM communicator gates;
6. Qwen production and the next formal TP layer gate.

Publish a new content-addressed conda/product manifest only if every applicable
gate passes without tolerance relaxation or fallback. Switch the active pointer
atomically; retain the previous product until one clean rerun completes.

Status: steps 1-5 and the single-device part of step 6 are complete for the new
immutable products named above. Besides the operator gates, a fresh online
NVIDIA comparison covers decoder layers 0..23 exactly once across one lineage;
every hidden, residual, current mutable cache, non-target cache, storage
identity, checkpoint and final norm passes without target feedback or fallback.
The replacement-product CCL matrix passes at N=2/3/4/8/16 and the real pinned
vLLM `GroupCoordinator.all_reduce` gate passes at N=2. The continuous production
window completes 834 generic dispatches from empty cache through two-token
prefill and two cache-preserving decode tokens: 834 retired, 834 type20 durable,
833 reuse, one clean session completion, zero fallback, 87 trajectory/cache
comparisons within policy, and unchanged postflight identity. Its scope remains
teacher-forced backbone final norm and mutable state; it does not claim logits,
sampling, greedy generation or TP. The simulator-aware 16-slot SMI lifecycle
also passes on this product. Formal model TP work is limited to TP=2 and TP=4;
CCL/runtime capacity remains 2..16. The remaining M4 gate is the formal
RowParallel TP layer. M5 deletion remains limited to items with a proven
replacement and zero active production/build/test/documentation references.

### M5: legacy deletion

Use the M0 inventory and current reference graph. Delete a candidate only when:

- production, build, documentation, and active tests have no reference;
- replacement behavior and negative/fault coverage pass;
- historical evidence remains readable without executing it;
- clean builds from the conda/product entry point pass; and
- removal decreases duplicate ownership rather than hiding it behind wrappers.

Expected candidates include exact-kernel fixture routes after generic dispatch
fully covers them, duplicated runtime/gem5 codec declarations after generation,
copied upstream layer state machines, obsolete environment/bootstrap variants,
and tests whose only purpose was the removed implementation. KMT/HIP/OpenCL
compatibility code with a current named public caller is not deleted merely
because vLLM does not use it.

Current deletion status: the fixed `PINNED_DISPATCH_V1` XOR implementation is
no longer advertised, routed, linked into the production bridge, or present in
the production bridge object. Its wire records and runtime C ABI remain
decodable and return `NOT_SUPPORTED`; the standalone historical harness and
GTest remain for old evidence. Seven zero-reference Qwen smoke CLIs were
removed from the live tree after an exact reference-graph review. A recoverable
manifest with their bytes and SHA-256 values is stored under
`/home/zhaosiying/amdgpu-sim-backups/20260813-generic-bridge-migration`.
The tied LM-head diagnostic, bootstraps, queue/signal state, CCL paths, and
fixtures with active test or evidence readers remain in place.

## Timebox and fallback

This migration must converge early. M0 and M1 are retained improvements even
if bridge extraction stops. M2/M3 are abandoned and restored from the source
backup if any of these occurs:

- two consecutive blockers require editing pinned vLLM, PyTorch, Triton core/
  AMD lowering, or gem5 GPU/Vega/CU implementation;
- the first three operation families cannot share one contract/adapter without
  changing the public runtime ABI or accepted wire behavior;
- a correctness regression repeats after two bounded fixes, or passing would
  require a tolerance/fallback relaxation;
- the extraction adds more duplicate production paths than it removes; or
- the bridge work prevents measurable progress on P9 RowParallel for two
  consecutive work cycles.

On fallback, keep the conda environment, upstream-zero-diff enforcement,
documentation, and any fully accepted adapter seam. Remove incomplete parallel
bridge paths, restore the proven implementation from the backup, and resume the
existing P9 RowParallel plan. Do not leave half-integrated abstractions or
unused compatibility code in the repository.
