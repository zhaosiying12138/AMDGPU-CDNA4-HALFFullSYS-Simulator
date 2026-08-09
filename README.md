# amdgpu-sim

`amdgpu-sim` is a source-backed, host-side AMDGPU simulation stack.  Its
long-term purpose is to let HIP, OpenCL, Triton, PyTorch, and vLLM workloads
submit real AMDGPU code objects to gem5 without a guest kernel, `/dev/kfd`,
`/dev/dri`, or the production AMD UMD/KMD binaries.

The root is a control-plane and transaction-coordinator repository. The six
pinned upstream source lanes and the project-authored `self-amdgpu-runtime`
lane are standard Git submodules under `projects/`. Each child retains
independent history and an immutable annotated baseline tag, while the root
gitlink records its exact current head, tests, checkpoints, and lessons.
`SOURCE_LOCK.json` owns upstream provenance; `PROJECT_LANES.json` owns our
project baselines; accepted checkpoints own descendant work heads. Model
weights, virtual environments, and build
outputs are downloaded or built by scripts into ignored paths and are never
stored in Git.

The first hard acceptance target is the official text-only
`Qwen/Qwen3.5-0.8B` checkpoint, served by vLLM with tensor parallelism across
two independent gem5 instances, producing at least one greedy token with the
same token ID as the reference and no CPU arithmetic fallback.  The protocol
is N-rank from the beginning so TP=4/8 can follow without a pair-specific
rewrite.  The checkpoint is already downloaded at the pinned revision under
`models/`; its first software gate is a source-grounded 15-contract text-only
operator manifest.  Full ROCm/OpenCL CTS is deliberately not a prerequisite;
every operator used by this model still needs an AMD execution result, and
CPU/NVIDIA fallback is never counted as a pass.

The local preparation is reproducible from the recorded Hugging Face mirror
capture: revision `2fc06364715b967f1860aea9cf38778875588b17`, one
`1,746,942,600`-byte safetensors file, SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.

Two later goals are explicit in plan revision 2. Once the user's unmodified
Triton `tutorial/01-vecadd.py` request (the pinned checkout's
`python/tutorials/01-vector-add.py`) runs through GemSim, retained profiles will drive a
small number of high-impact operator/simulator optimizations, followed by a
separately gated experiment for CPU-parallel threadblock simulation that must
preserve dependency and synchronization semantics. After the model path is
usable, a low-priority simulator-aware `rocm-smi` client will report the ON/OFF
state of multiple gem5 daemon instances without probing physical GPUs.

Read [PLAN.md](PLAN.md) for the complete staged plan and [GOAL.md](GOAL.md) for
the immutable acceptance anchor.  A blank-context handoff starts with:

Gem5 acceptance builds use the recorded mold/24-job procedure in
[docs/gem5-build.md](docs/gem5-build.md).

```text
继续执行 amdgpu-sim 计划。从 checkpoint 指定的下一条唯一动作继续：先读取
PLAN.md、GOAL.md、SOURCE_LOCK.json、state/current.json、最新 checkpoint 和
bitlesson，运行 scripts/resume.sh --verify；不要重做已通过的工作。
```

`CP-0007` remains the accepted bridge-private signal/event boundary. The standalone
runtime and gem5 preserve the CP-0004 byte-exact handshake, CP-0005 bounded
queue control, CP-0006 sparse simulated-memory transfer, and N=1/2/3/4/8
isolation gates while adding signed 64-bit signal create/load/store/destroy,
generation-safe one-shot waits, event-queue completion, bounded outbound
accounting, shared request correlation, and retry/poison semantics. The signal
records remain host transport primitives: they do not expose GPU-visible signal
memory or claim packet submission, code-object loading, or GPU execution.

`CP-0008` is now the accepted pinned-dispatch boundary. It preserves every
CP-0004 through CP-0007 gate and proves one source-pinned `gfx950-xor-u8-v1`
wave64, one-CU, one-workgroup AQL execution through the real
`HSAPacketProcessor -> GPUCommandProcessor -> GPUDispatcher -> CU` path, with
exact packet/trace hashes, positive retired/store statistics, CP7 signal
completion at retirement plus one tick, exact non-identity D2H bytes and CRC,
causal clean exit, and zero host fallback. This is still not a generic
code-object, ROCr/libhsakmt, HIP, OpenCL, Triton, PyTorch, vLLM, multi-CU,
collective, or performance claim.

`CP-0009` is now accepted as the source-exact ROCr/libhsakmt ABI boundary. It
records the pinned ThunkLoader source, 124-entry source-union order, Linux
shared/direct effective counts, 17 key-offset-partial layouts, status mapping,
and model ABI 1.1. The runtime child exposes metadata, the existing transport
handshake, and deterministic unsupported-call behavior only: it exports zero
typed hsaKmt/DRM entry points, does not open KFD/topology, and does not load
production GPU libraries or device nodes. The next action is
`P2-KMT-ABI-02`: build the typed libhsakmt shim and a versioned daemon KFD/DRM
operation envelope with fixed-width ownership and pointer translation. That
gate must remain separate from generic ROCr/HIP/OpenCL compatibility.

`CP-0010` is now accepted as the typed KMT shim boundary. It adds the frozen
message types 14/15 and capability bit 5, 18 fixed-width operations, explicit
owner/object generations, copied-buffer CRCs, per-provider sequence scope,
canonical gfx950 fixture checks, and daemon-owned simulated resource state.
The retained gem5 smoke completes the runtime-to-daemon lifecycle with
`failures=0`, but this remains a translated envelope rather than a complete
124-PFN ROCr/libhsakmt provider, KFD attach, HIP, OpenCL, Triton, PyTorch, or
vLLM implementation. The pinned fixture work is recorded by `CP-0011` below;
the next action is the decoder/toolchain proof that follows it.

`CP-0011` is now accepted as the source-locked code-object fixture boundary.
The runtime validates the two tracked gfx950 ELF V6 images, MsgPack metadata,
PT_LOAD/relocation structure, exact descriptor and code symbols, hidden
kernarg offsets, and 64-byte resource descriptors; gem5 binds the same
provenance without embedding HSACO bytes. This is parser and provenance
evidence only.

`CP-0012` is now accepted as the pinned toolchain and decoder boundary. It
records the reproducible device-libraries/HSACO identities, the native
`mold`/24-job gem5 link method, gfx942/gfx950 decoder alias isolation, and
runtime-local selected-kernel byte materialization. It still does not claim
HSACO wire upload, PT_LOAD mapping, dynamic AQL/kernarg, or real gem5 execution;
its historical next action was the A1 transport gate recorded by `CP-0013`.

`CP-0013` is now accepted at the A1 code-object transport/staging boundary. It
adds fixed 4096-byte BEGIN/CHUNK/COMMIT records, pointer-free capability and
identity fields, per-chunk CRC-32C, whole-image SHA-256, owner/generation and
ordering validation, and daemon-owned atomic staging in both children. A1
publishes no mapping, descriptor, code, or kernarg address and makes no
PT_LOAD, AQL, queue-submission, gem5 execution, hardware, timing, or performance
claim. Its historical next action was the separately scoped
`P3-CODEOBJ-03-A2` loader gate; that gate remains unproven, while CP-0014 below
makes `P3-HOST-NATIVE-02` current.

`CP-0014` is accepted at the `P3-HOST-NATIVE-01` source-inventory boundary.
EV-0036/EV-0037 record the reusable gem5 GPU/Vega/HSA and host-bridge surfaces,
the current x86/Process/TLB blockers, and the runtime ABI boundary; gem5's
boundary suite is 4/4, the runtime CTest matrix is 16/16, and the focused
Clang ASAN boundary test is 1/1. The gem5 path remains the behavioral oracle.
No host-native execution, Triton end-to-end, hardware, timing, or performance
claim is made. Its historical `P3-HOST-NATIVE-02` action is completed by
`CP-0015` below; the CP14 inventory remains the architectural record.

`CP-0015` is accepted at the `P3-HOST-NATIVE-02` control-core/build boundary.
The standalone gem5 target uses `BUILD_ISA=n`, `USE_X86_ISA=n`, and
`BUILD_GPU=y`; its ELF/dependency audit and protocol/memory/queue/signal
self-tests pass, and the existing eight legacy state regression binaries remain
green. This is a control-plane boundary, not HSACO mapping or execution: no
GPU pipeline, Triton E2E, hardware, timing, or performance claim is made. The
next action is `P3-HOST-NATIVE-03`, the pinned gfx950 loader/dispatch parity
gate between the host-native and gem5 front-ends.

`CP-0016` is accepted at the first functional-parity sub-gate of that
workstream. It adds a standalone `host_gpu_native_fixture_core` that reuses the
existing protocol, sparse memory, queue, signal, and pinned dispatch state, plus
GPU-VA range access and page-lifetime leases. The gfx950 XOR fixture, negative
access checks, and lifetime cleanup pass with `USE_X86_ISA=n`; the runtime probe
recognizes the pinned HSACO metadata. This is not PT_LOAD mapping, dynamic
AQL/kernarg construction, GPU pipeline execution, hardware validation, or
Triton E2E. Triton remains 0/1, and the next action is
`P3-HOST-NATIVE-03-A` for bounded loader/translation/AQL parity.

`CP-0017` is accepted at the bounded host-native PT_LOAD staging boundary.
The no-x86 target stages the locked gfx950 `gpuReadWrite` image, checks exact
PT_LOAD tuples, copies file bytes, zero-fills BSS, binds descriptor/entry
addresses, and preserves page leases across Busy/unmap cases.  This is a
fixture-scoped staging gate only: no segment permissions, relocations, dynamic
kernarg/AQL, queue submission, GPU instruction execution, or Triton E2E is
claimed.  The next action is native translation plus dynamic AQL/kernarg parity.

The pinned Triton/LLVM overlay currently reaches gfx950 HSACO compilation only
(vecadd SHA-256
`ee8b0f892da7ab1886f17ee66f88de5c23e05a48f7f361e02bd0707c9a11826e`); no
Triton request has executed in GemSim yet, so the user-facing Triton count is
still `0/1`.

The frozen `SOURCE_LOCK.json` and registered project baseline remain immutable.
