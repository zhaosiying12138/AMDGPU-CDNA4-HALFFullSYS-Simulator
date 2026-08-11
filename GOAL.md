# Goal contract: host-side AMDGPU gem5 simulation

**Goal ID:** `GSIM-001`

**Plan:** `AMDGPU-SIM-V1`, revision `2`

**Current state:** `CP-0028-normal-Triton-Python-vecadd-accepted; exact-float32-contiguous-only; same-process-two-launch-reuse; final-repository-local-install-not-yet-accepted; next-CP-0029-queue-test-stabilization-and-fresh-prefix; model-operator-matrix-before-performance-work`

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
4. **Single-device model:** run the complete text model on one simulated device
   through prefill and a predeclared multi-token decode window. A one-token
   smoke is useful evidence but is not the stable-inference acceptance gate.
5. **Multi-TP model:** implement the required CCL collective semantics, run one
   rank per independent gem5 daemon, and complete the full model with stable
   multi-token inference at TP=2 before generalizing to supported TP=4/8 shapes.

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
- The selected greedy token ID exactly matches a pinned reference; numerical
  and trace evidence satisfies the tolerances in PLAN.md.
- The same N-rank protocol passes synthetic TP=4/8 topology and collective
  tests, with shape constraints and any model limitation explicitly recorded.
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
prerequisites). If that action or an active prepare-phase transaction conflicts
with the product-level priority above, do not execute or accept it: retire and
re-scope it through the transaction protocol first. Never redo a passed gate,
silently change the model ID, or advance a phase without a new checkpoint and
root coordinator commit.

### Blank-context continuation prompt

```text
继续执行 amdgpu-sim 计划。当前目录是 /home/zhaosiying/amdgpu-sim。
先读取 PLAN.md、GOAL.md、SOURCE_LOCK.json、state/current.json、其引用的
最新 checkpoint/bitlesson/evidence，运行 scripts/resume.sh --verify。
CP-0028 normal-Python Triton vecadd boundary 已 accepted；从唯一 next_action
`P5-TRITON-VECADD-07-FRESH-SCHEMA8` 继续：先完成 CP-0029 的 queue mock
determinism test-only 修复和全新 schema-8 prefix，再进入模型算子矩阵。不要重做
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
fail closed；它们不能作为安装 authority。CP-0029 先用 test-only 修改稳定该
queue gate，再从空 schema-8 prefix 完成安装验证，随后立即跑模型所需算子矩阵、
单卡完整模型稳定多 token，再实现 CCL 和多 TP 完整模型稳定多 token。只有真实
算子/层/模型的耗时阻塞该主线时才开启
可复现 profile、80/20 优化或 threadblock host-parallel 门禁；不安全或不可证明
独立的 workgroup 必须保留串行执行。模型路径和 N-rank registry 可用后，再实现只观察模拟 daemon
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
evidence. CP-0029 stabilizes the queue test and produces a fresh accepted
prefix before the model-required operator matrix. Profiling is deferred until a
real workload demonstrates a material operator-, layer-, or model-level
bottleneck.
