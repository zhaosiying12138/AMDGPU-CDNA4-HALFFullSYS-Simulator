# Goal contract: host-side AMDGPU gem5 simulation

**Goal ID:** `GSIM-001`

**Plan:** `AMDGPU-SIM-V1`, revision `2`

**Current state:** `CP-0024-bounded-daemon-handler-boundary-accepted; bit8-positive-route-submit-and-normal-launcher-blocked; next-CP-0025-P5-TRITON-VECADD-04-DAEMON-LIFECYCLE`

**Current phase:** `P5`

**Model correction:** the official target is `Qwen/Qwen3.5-0.8B`; there is no
official `Qwen3.5-0.9B` checkpoint in this project.

The pinned Qwen checkpoint is prepared offline under `models/` at revision
`2fc06364715b967f1860aea9cf38778875588b17`. The first model gate is the
text-only path: its 24-layer topology (18 linear-attention/GDN layers and 6
full-attention layers) has a source-grounded 15-contract operator manifest.
Vision is deferred. Full ROCm and OpenCL CTS are not prerequisites for this
goal; every operator used by the checkpoint still needs an AMD execution
result, and CPU or NVIDIA fallback never counts as success.

## Success criteria

- A self-contained copy of this directory can prepare and run the pinned stack
  on the same Linux ABI/architecture without network access after preparation.
- Host HIP/OpenCL/Triton/PyTorch/vLLM workloads use the standalone
  `self-amdgpu-runtime` ABI and submit real code objects to gem5.
- A host-native simulator daemon/library runs on the physical host without
  `VEGA_X86`, the gem5 x86 CPU model, or x86 system/CPU ports, while reusing
  the existing GPU packet, Vega ISA, memory, queue, signal, and CU functional
  modules and preserving their accepted behavior.
- No production AMD UMD/KMD library is loaded; no `/dev/kfd` or `/dev/dri` is
  opened; no CPU arithmetic fallback occurs in acceptance mode.
- Two independent gem5 daemons and vLLM TP=2 execute the official Qwen3.5-0.8B
  text path through full prefill and at least one decode token.
- The selected greedy token ID exactly matches a pinned reference; numerical
  and trace evidence satisfies the tolerances in PLAN.md.
- The same N-rank protocol passes synthetic TP=4/8 topology and collective
  tests, with shape constraints and any model limitation explicitly recorded.
- Triton's user-facing `tutorial/01-vecadd.py` request (mapped, without editing
  upstream, to the pinned checkout's `python/tutorials/01-vector-add.py`) runs
  transparently through the GemSim device path, with no CPU fallback, and its
  output matches the pinned oracle.
- After transparent vecadd is accepted, a retained gem5 profile identifies
  the dominant operator-execution costs and at least one 80/20 optimization is
  implemented with before/after evidence and unchanged correctness.
- Any CPU-side threadblock parallelism is enabled only behind an explicit
  simulator mode and passes dependency, barrier, atomic, determinism, and
  race-regression gates; it must improve the measured operator path rather
  than merely inflate host utilization.
- A simulator-aware `rocm-smi` command reports the live ON/OFF state and stable
  identity of at least two concurrently running gem5 instances without probing
  hardware device nodes or loading production management libraries.

## In scope

ROCr/HIP/CLR/libhsakmt-compatible host glue, gem5 host bridge/daemon,
gfx950/MI355X code-object execution, functional N-rank transport and RCCL
semantics, Triton AMD lowering adapter, PyTorch and vLLM integration, Qwen
GDN/full-attention layers, operator profiling and correctness-preserving
simulator scheduling, a simulator-aware `rocm-smi` status client,
reproducible tests, evidence, checkpoints, and offline packaging.

## Out of scope until explicitly promoted

Vision execution, real hardware validation, silent CPU or mock execution,
unbounded ROCm API compatibility, full ROCm/OpenCL CTS, timing claims before FabricModel, elastic
rank shrink, unsafe or nondeterministic threadblock parallelism, hardware
`rocm-smi` compatibility claims, and committing weights/build
products/environments.

## Resume contract

Do not infer progress from prose or an old chat. Read `PLAN.md`, this file,
`SOURCE_LOCK.json`, `PROJECT_LANES.json`, `state/current.json`, the referenced
checkpoint, and all bitlessons and evidence named there. Run:

```bash
scripts/resume.sh --verify
```

Then perform only `state/current.json.next_action` (after checking its
prerequisites).  Never redo a passed gate, silently change the model ID, or
advance a phase without a new checkpoint and root coordinator commit.

### Blank-context continuation prompt

```text
继续执行 amdgpu-sim 计划。当前目录是 /home/zhaosiying/amdgpu-sim。
先读取 PLAN.md、GOAL.md、SOURCE_LOCK.json、state/current.json、其引用的
最新 checkpoint/bitlesson/evidence，运行 scripts/resume.sh --verify。
CP-0019 host-native queue/command-processor-core B1 boundary 已 accepted；从唯一
next_action `P3-HOST-NATIVE-03-B2` 继续，不要再次
begin CP-0014，也不要重做 bootstrap、source freeze、authored runtime
baseline 或已通过的 CP4-CP9 gates，也不要修改已冻结的 SOURCE_LOCK.json
或已登记的 PROJECT_LANES baseline。CP-0010 只实现 18 个 typed KMT
操作的固定宽度 shim、版本化 daemon envelope、模拟资源生命周期和
no-device 证据；它不是完整 124-PFN ROCr/libhsakmt provider，也不宣称
KFD attach、HIP、OpenCL、Triton、PyTorch 或 vLLM 能力。CP-0011 已冻结
两份真实 gfx950 HSACO 的 ELF/MsgPack/descriptor/kernarg provenance，但
gem5 gfx950 decoder、unsupported opcode、pinned device-libs/toolchain proof
已完成；CP-0013 A1 仅完成 versioned HSACO fixed-record transport/staging，
不包含 PT_LOAD mapping、动态 AQL/kernarg 或真实 code-object execution。
当前还没有任何 Triton 端到端通过案例。CP-0015 已完成 no-VEGA_X86
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
harness；真实 runtime-to-gem5 probe 只证明 bit 8 未广告时 canonical reject，且
随后 baseline reconnect 成功，没有发送或证明 type-18 socket/H2D daemon route。
当前唯一 next_action 是 CP-0025 的
`P5-TRITON-VECADD-04-DAEMON-LIFECYCLE`：先解决正常 alignment 8 与 page-backed
ALLOC 的契约差异，把既有 v1 `MEMORY_COPY_H2D` carrier 绑定到同一 owner session，
再补齐 queue/signal/AQL packet/tick、SUBMIT ACK 和 type-20 completion；完整
lifecycle 通过前仍不得广告 capability bit 8，也不得把 handler/admission 当作
正常 Triton launcher 或 GPU 执行成功。
在 Triton 用户命令 `tutorial/01-vecadd.py`（当前 pinned checkout 中对应未修改的
`python/tutorials/01-vector-add.py`）首次透明通过后，先完成可复现的 gem5
算子 profile、80/20 优化和 threadblock host-parallel 正确性/可行性门禁，再
扩展更大算子、PyTorch 和模型路径；不安全或不可证明独立的 workgroup 必须
保留串行执行。模型路径和 N-rank registry 可用后，再实现只观察模拟 daemon
lease/epoch 的 `rocm-smi` ON/OFF 工具，绝不探测物理卡。
每个原子进展使用新的 Checkpoint-ID，先提交 child 再提交 root coordinator
commit，并同步 checkpoint、bitlesson、evidence。遇到长耗时或 token 切换，
先形成 coherent partial checkpoint，再暂停。
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
