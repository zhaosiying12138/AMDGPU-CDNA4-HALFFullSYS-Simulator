# Goal contract: host-side AMDGPU gem5 simulation

**Goal ID:** `GSIM-001`

**Plan:** `AMDGPU-SIM-V1`, revision `2`

**Current state:** `CP-0016-host-native-functional-parity-accepted; next-P3-HOST-NATIVE-03-A`

**Current phase:** `P3`

**Model correction:** the official target is `Qwen/Qwen3.5-0.8B`; there is no
official `Qwen3.5-0.9B` checkpoint in this project.

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
unbounded ROCm API compatibility, timing claims before FabricModel, elastic
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
CP-0016 host-native functional-parity boundary 已 accepted；从唯一 next_action `P3-HOST-NATIVE-03-A` 继续，不要再次
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
state 的 functional parity adapter，但没有完成 PT_LOAD loader、动态 AQL/kernarg
或 GPU instruction execution。下一步完成 host-native/gem5 loader parity，再以未修改
Triton vecadd 作为第一用户可见验收。
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
