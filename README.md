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
same token ID as the reference and no CPU arithmetic fallback, then sustaining
a predeclared multi-token decode window. The protocol
is N-rank from the beginning so TP=4 can follow without a pair-specific
rewrite. The underlying CCL and simulator-device capacity remains 16 ranks;
that capacity is not a requirement to run a TP=8 or TP=16 model gate. The
checkpoint is already downloaded at the pinned revision under
`models/`; its first software gate is a source-grounded 15-contract text-only
operator manifest.  Full ROCm/OpenCL CTS is deliberately not a prerequisite;
every operator used by this model still needs an AMD execution result, and
CPU/NVIDIA fallback is never counted as a pass.

The local preparation is reproducible from the recorded Hugging Face mirror
capture: revision `2fc06364715b967f1860aea9cf38778875588b17`, one
`1,746,942,600`-byte safetensors file, SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.

The product order is explicit in plan revision 2: first a normal OpenCL
executable that compiles `.cl` and transparently runs through gem5, then an
ordinary Triton Python correctness request through the normal driver/runtime,
then every model-required operator, one-device stable multi-token inference,
and finally CCL-backed multi-TP stable inference. The complete unmodified
upstream tutorial file, including its benchmark sweep, is a later scale gate
rather than a prerequisite for the model-specific operator matrix. Profiling and simulator
optimization are scheduled only when a real operator, layer, or model run
materially blocks that correctness path.

The simulator-aware `rocm-smi` client is now installed with the repository
conda product. It always exposes 16 logical simulator slots. A slot is `ON`
only while a managed gem5 process holds a live runtime lease whose PID,
process start time, executable inode, daemon/job identity, and private socket
all validate; unused or released slots are `OFF`. It never probes physical
GPUs or loads a production ROCm SMI library.

Current accepted collective scope is intentionally narrower than the final
model target. The reusable out-of-tree CCL engine has device-backed BF16/1024
allreduce evidence at N=2/3/4/8/16, and a real pinned-vLLM
`GroupCoordinator.all_reduce` passes separately at N=2 through the out-of-tree
communicator with zero Gloo tensor payload or fallback. No RowParallel layer or
Qwen TP model is accepted yet; the next gate is Qwen-sized TP2
`RowParallelLinear` sharding plus out-of-place allreduce.

The integration architecture deliberately keeps pinned vLLM, PyTorch, and
Triton core unmodified. vLLM uses its official platform/general plugin,
`PluggableLayer`, attention, and `DeviceCommunicator` extension points; Triton
uses the out-of-tree `gemsim_amd` backend while retaining its normal frontend,
JIT/cache, compiler coordinator, and AMD lowering. The backend hides the
versioned self-runtime, CCL engine, and gem5 transport. See
[docs/framework-runtime-layering.md](docs/framework-runtime-layering.md) for
the ownership and TP integration contract.

The bounded migration from the current implementation to a concentrated
runtime-gem5 bridge and repository-owned conda entry point is specified in
[docs/runtime-gem5-bridge-migration.md](docs/runtime-gem5-bridge-migration.md),
including regression, legacy-removal, and rollback gates.

The repository-owned conda entry point is now available. It installs exact
noneditable project wheels over the pinned framework/compiler packages and
keeps mutable caches outside the content-addressed prefix:

```bash
./scripts/setup_conda_env.sh --install
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$(./scripts/setup_conda_env.sh --print-prefix)"
python examples/quickstart/triton_vecadd.py
python examples/quickstart/vllm_silu.py
rocm-smi
rocm-smi --json
```

`./scripts/setup_conda_env.sh --verify` rehashes the product, native runtime,
gem5 inputs, plugin source sets, and pinned upstream identities. These two
quickstarts are user-entry smoke gates; they do not by themselves claim full
Qwen inference or tensor-parallel model acceptance. Formal Qwen TP model gates
are limited to TP=2 and TP=4; the generic CCL/runtime capacity remains 2..16.

Read [PLAN.md](PLAN.md) for the complete staged plan and [GOAL.md](GOAL.md) for
the immutable acceptance anchor.  A blank-context handoff starts with:

Gem5 acceptance builds use the recorded mold/24-job procedure in
[docs/gem5-build.md](docs/gem5-build.md).

```text
继续执行 amdgpu-sim 计划。先读取 PLAN.md、GOAL.md、
ENGINEERING_CONSTRAINTS.md、SOURCE_LOCK.json 和 PROJECT_LANES.json，再核对当前
源码、测试与相关 evidence。state/current.json 和 CP1-30 是历史档案，不执行其
过时的 CP31/RMSNorm next_action；从 PLAN 当前 P8 correctness gate 继续。
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
GPU pipeline, Triton E2E, hardware, timing, or performance claim is made. Its
historical next action was `P3-HOST-NATIVE-03`, the pinned gfx950
loader/dispatch parity gate later split across CP16-CP20.

`CP-0016` is accepted at the first functional-parity sub-gate of that
workstream. It adds a standalone `host_gpu_native_fixture_core` that reuses the
existing protocol, sparse memory, queue, signal, and pinned dispatch state, plus
GPU-VA range access and page-lifetime leases. The gfx950 XOR fixture, negative
access checks, and lifetime cleanup pass with `USE_X86_ISA=n`; the runtime probe
recognizes the pinned HSACO metadata. This is not PT_LOAD mapping, dynamic
AQL/kernarg construction, GPU pipeline execution, hardware validation, or
Triton E2E. At that boundary Triton remained 0/1, and the historical next
action was `P3-HOST-NATIVE-03-A` for bounded loader/translation/AQL parity.

`CP-0017` is accepted at the bounded host-native PT_LOAD staging boundary.
The no-x86 target stages the locked gfx950 `gpuReadWrite` image, checks exact
PT_LOAD tuples, copies file bytes, zero-fills BSS, binds descriptor/entry
addresses, and preserves page leases across Busy/unmap cases.  This is a
fixture-scoped staging gate only: no segment permissions, relocations, dynamic
kernarg/AQL, queue submission, GPU instruction execution, or Triton E2E is
claimed. Its historical next action was native translation plus dynamic
AQL/kernarg parity.

`CP-0018` is accepted at the host-native dispatch-admission B0 boundary. The
no-x86 target reuses the staged gfx950 image and shared host state to load the
descriptor, bind its entry relation, pack/read back the 280-byte hidden kernarg,
materialize a 64-byte AQL packet, construct an `HSAQueueEntry`, and exercise
queue-control and ordered lifecycle-listener contracts. The legacy dispatcher
object recompiles with the extracted listener symbols. This remains admission
and control-state evidence only: no HSA queue publication, HSAPP/
GPUCommandProcessor/GPUDispatcher/ComputeUnit instantiation, AQL submission,
instruction execution, or GPU output/trace differential is claimed. The static
Qwen 15-contract gate and offline model hashes remain valid, but strict AMD
execution is blocked; Triton E2E and Qwen inference remain `0/1`. Its historical
next action was `P3-HOST-NATIVE-03-B1`, now accepted by CP-0019 below.

`CP-0019` is accepted at the host-native queue/command-processor-core B1
boundary. The no-x86 target resolves host-owned GPU virtual addresses,
registers a 64-slot queue, publishes and rings one 64-byte AQL packet, fetches
it in order, reads the locked descriptor/MQD/kernarg/completion-signal object,
and reuses `HSAQueueEntry` for one native CP-core admission. Here
`aql_submitted=true` means accepted by the extracted native core only: legacy
HSAPP and GPUCommandProcessor SimObjects are neither linked nor instantiated.
At the historical CP19 boundary, GPUDispatcher/CU connection, host read-index
update, packet retirement, signal decrement, instruction fetch/retirement, ISA
execution, and kernel output were all false. Its next action was
`P3-HOST-NATIVE-03-B2`, later accepted by CP20. The Qwen model and static
15-contract gate were ready, but strict AMD execution and Qwen inference were
still blocked at that boundary.

`CP-0020` is accepted at the bounded no-x86 B2 execution boundary. The locked
5,528-byte `gfx950` `gpuReadWrite` image travels from the native queue/CP core
through the reused `GPUDispatcher`, `Shader`, `ComputeUnit`, Vega decoder, and
instruction path. The dispatch is four 256-item workgroups and sixteen wave64
waves; the lifecycle records 19 instruction-start PCs per wave (304 total),
independently matched by CU `numInstrExecuted=304` and 16 completed waves.
Separate 4 KiB A/B/C allocations pass A unchanged, B=gid, and C=A over all
1024 elements; packet retirement, MQD read-index `0->1`, direct-u64 signal
`1->0`, and pin release are also checked. The runtime graph has no CPU,
Process, Ruby, TLB, HSAPP, or GPUCommandProcessor objects; generic
`BaseCPU`/`System` symbols retained in the monolithic ELF are not runtime
instantiations. This is one locked functional case only: generic gfx950,
arbitrary HSACO, timing accuracy, fences/barriers, atomics, LDS/scratch,
GPU-TLB/Ruby/coherence, HIP/OpenCL, and performance remain unproven. At the
historical CP20 boundary, Triton and Qwen remained `0/1`; its then-next action
was the Triton vecadd launcher gate later bounded and accepted by CP28.

The Qwen smoke now also supports importlib-loaded source-contract tests by
explicitly adding the repository and `tools/` roots; this fixes test loading only
and does not change the no-fallback execution policy.

At the historical CP21 boundary, the pinned Triton/LLVM overlay reached gfx950 HSACO compilation only
(vecadd SHA-256
`ee8b0f892da7ab1886f17ee66f88de5c23e05a48f7f361e02bd0707c9a11826e`); no
Triton request had executed in GemSim, so the user-facing Triton count was
`0/1`. CP-0021 records that provenance boundary explicitly: the
unmodified tutorial hash is
`842430949e0ccde4fbce07606cce3ac4bac36bf21b2b12619a31b795ca4029b3`, the
HSACO target is `amdgcn-amd-amdhsa-unknown-gfx950`, and its descriptor preload
is 12 DWORD (48 bytes). Runtime CTest is 16/16 (focused code-object tests 4/4),
but compiler/JIT invocation, normal launcher, transport, execution, and
fallback are all false. The public A1 path still publishes zero VAs and remains
fixture-only. CP-0022 accepts the independent payload-v2 codec boundary: v1
framing remains unchanged, bit 8 and records 18/19/20 are opt-in, and
owner-scoped MAP/ALLOC_KERNARG/SUBMIT_AQL/UNMAP records are strictly validated.
CP-0023 now accepts the bounded adapter/client/admission step. The runtime
public v2 lifecycle preserves v1 and passes CTest 18/18; gem5's protocol suite
passes 47/47 normally and under ASAN/UBSAN. A separate local no-x86 adapter
selftest performs owner-bound MAP, ALLOC, kernarg publish, AQL publish/fetch,
CP admission, retire, and UNMAP with rejection rollback. It is not a live
daemon path: capability bit 8 advertisement and MessageType 18 routing are
false, as are GPUDispatcher/CU execution, normal Triton launcher, compiler/JIT,
and fallback. The 12-DWORD (48-byte) Triton preload remains NOT_SUPPORTED.
CP-0024 accepts the next bounded partial step: gem5 contains an owner-bound
type-18 handler, type-19 response plumbing, and a shared route-policy harness,
while the runtime adds an opt-in endpoint probe. The retained live
runtime-to-gem5 result is a canonical unsupported-capability handshake followed
by a successful baseline reconnect; bit 8 is still unadvertised and no type-18
request is sent. A positive socket route, daemon H2D publication, normal
logical alignment 8, SUBMIT ACK, type-20 completion, launcher, compiler/JIT,
GPU execution, and fallback remain false. Its historical next unique action was
CP-0025 / `P5-TRITON-VECADD-04-DAEMON-LIFECYCLE`.

CP-0025 accepts the bounded positive generic daemon control lifecycle. A fresh
two-owner runtime-to-gem5 run selects capability bit 8 and all dependencies,
then completes MAP, logical-alignment-8 ALLOC over hidden page backing, the
existing v1 `MEMORY_COPY_H2D` carrier, daemon-built AQL admission, type-19 ACK,
type-20 retirement, UNMAP, disconnect cleanup, and reconnect. Packet CRC is
nonzero and lifecycle ticks are nonzero and nondecreasing. This remains native
control-processor admission/retire only: GPUDispatcher/CU execution, kernel
output correctness, normal Triton launcher, compiler/JIT, fallback, and Qwen
are false at the CP25 boundary. Its historical next action was CP-0026 /
`P5-TRITON-VECADD-05-GPU-EXECUTION`, which is accepted below; later Triton
preload and launcher work remains separate.

CP-0026 accepts the locked generic execution extension while keeping bit 8
strictly as the control/admission/retire contract. Bit 9 (`GENERIC_EXECUTION_V2`)
maps to runtime word 0 bit 9 and wire byte 1 bit 1, and is selected only with
bit 8 and all dependencies. The live daemon route uses the exact 5,528-byte
gfx950 `gpuReadWrite` image (SHA-256
`7b6a4d2bb7f9c4e7466bcf69f3110ecbfab54d07abd4c70b6bd96b6a6fb9de56`, zero
preload) and reaches `GPUDispatcher`/`ComputeUnit`: four workgroups, sixteen
wave64 waves, 304 instruction starts, A/B/C oracle, durable type 20, duplicate
D2H verification, and UNMAP. The fsynced daemon trace is authoritative for
execution and quarantine; endpoint JSON is authoritative only for bytes
delivered to the client. A post-ACK disconnect trace proves quarantine cleanup
without type 20 or client output. The wire signal field remains expected `1`
with no observed wire after-value; trace `1 -> 0` is the private native AQL
completion signal. This is a fixture-scoped `VEGA_X86` bridge proof, not a
standalone no-x86 daemon, generic gfx950, arbitrary HSACO, Triton, launcher,
compiler/JIT, performance, fallback, or Qwen claim. Its historical next action
was superseded by the product-priority clarification recorded in GOAL/PLAN and
by the accepted CP-0027 OpenCL executable boundary below.

CP-0027 accepts the first direct user-facing OpenCL executable. The repository-
local v5 prefix installs the pinned compiler/device libraries, shared
`self-amdgpu-runtime`, bounded `libOpenCL.so.1`, the normal `opencl-vecadd`
host executable, and its `.cl` source without touching system ROCm. Running the
executable alone compiles a 5,160-byte gfx950 image (SHA-256
`314ede16940432996c9fe190115408bf42744a8ab7d0036bf07b931e39c4cb19`), starts
the managed gem5 daemon, executes four workgroups/sixteen waves/448
instructions through GPUDispatcher/CU, copies only C back, validates bit-exact
`C=A+B`, and exits with zero CPU/NVIDIA fallback. The retained direct run is
about 0.94 seconds wall on this checkout, so profiling is not a prerequisite
to the next gate. This is still one exact vecadd and one execution per OpenCL
context; multi-dispatch and general OpenCL remain unaccepted.

CP-0028 accepts the first normal Triton Python product path. The external
`gemsim_amd` backend uses Triton's normal driver, JIT, and launcher to compile
the exact 5,384-byte pure-b010 gfx950 `add_kernel` (SHA-256
`7308427e69dea6f320178c55863291d4d615338eb295a422a5ff7a2c2b8afa95`), starts
managed gem5 implicitly, and runs two deterministic launches in one Python
process. The same session, queue, signal, packet VA, and allocation IDs are
observed across launches; stable image/packet/trace identifiers infer reuse of
the kernel source packet because the trace has no mapping ID. Both float32
`C=A+B` results are bit-exact and fallback remains zero. This proves only
contiguous float32 vector add at 98,432 elements with `BLOCK_SIZE=1024`.
Broadcast, cast, reduction, softmax, norm, GEMM, RoPE, cache, GDN, attention,
full-model, multi-token, TP, and CCL execution are not accepted.

The CP-0028 v6/v7 final-prefix attempts are retained as NON-PASSING evidence:
v6 was stopped after detecting a baked `/opt/rocm` default, and v7 failed a
deterministic queue mock test race. CP-0029 fixes that test-only race and accepts
a fresh schema-8 repository-local prefix after independent verify-only, OpenCL,
Triton, provenance, pollution, and active-isolation gates. CP-0030 then accepts
the first bounded model-required subgate: the exact BF16 SiluAndMul image runs
decode `[1,7168] -> [1,3584]` and masked-prefill `[7,7168] -> [7,3584]` twice
through normal Python/Triton with finite output, exact traces, zero mismatch,
and zero fallback. This is not a complete MLP contract: the gate/up and down
projection GEMMs, all 15 complete model contracts, a complete layer, and the
model remain unaccepted. P5-OPS-01 expands the operator matrix next.

The frozen `SOURCE_LOCK.json` and registered project baseline remain immutable.

---

## 用户工具链（AMDGPU-CDNA4-SIM · rocm-smi · gem5 一键启停 · Triton demo）

本节面向想在模拟 GPU 上直接动手的使用者（无需了解 lane 体系）。

### 0. 一键创建

```bash
bash scripts/make_amdgpu_tools_env.sh
```

幂等、秒级（零包复制，只建 symlink 和激活脚本）。默认装到
`~/miniforge3/envs/AMDGPU-CDNA4-SIM`（可用 `SAGR_TOOLS_ENV_TARGET`
改目标路径）。前置（缺一即报错）：`build/rocr-stage-zcode`、
`projects/self-amdgpu-runtime/build/cp28-runtime-clang`、
`build/rocr_logging_preload.so`、`projects/gem5/build/VEGA_X86/gem5.opt`、
`tools/gemsim_smi_publish.py`。

### 1. 激活后直接 rocm-smi

```bash
conda activate AMDGPU-CDNA4-SIM
rocm-smi            # 文本表格；rocm-smi --json 供脚本消费
```

输出带 GPU 型号列（AMD Instinct MI350X（虞书欣粉丝特供版））。16 个
模拟 GPU 槽位；每个活着的 gem5 会话一行：`DAEMON_PID`（gem5 进程）、
`RANK/WORLD`（TP 位置）、`JOB_UUID`（同一次启动）、JSON 里的 `endpoint`
（bridge socket）。记录带 PID+starttime+flock 校验，进程死了槽位自动 OFF。

### 2. gem5 一键启停（支持实例数参数）

```bash
gem5-session start           # 启动 1 个实例（默认；适合单节点 softmax 测试）
gem5-session start 1         # 显式启动 1 个实例
gem5-session start 4         # 启动 4 个实例（rank 0..3，同一 job_uuid）
gem5-session status          # 每实例 pid/endpoint/已完成 dispatch 数
gem5-session restart 2       # 停止全部后启动 2 个实例
gem5-session stop            # 停止全部
gem5-session stop 0          # 只停实例 0
gem5-session stop --all      # 同 stop
gem5-session start 2 --accurate  # 用时序精确模式（默认 functional-fast）
```

启动后 endpoint 写入 `/tmp/amdgpu-sim-tools-session/session.env`
（`SAGR_GENERIC_BRIDGE_ENDPOINT` 指向实例 0，另导出
`SAGR_TOOLS_INSTANCE_COUNT` 与 `SAGR_TOOLS_INSTANCE_ENDPOINTS` 列表）；
新 shell 或重新激活自动附加到实例 0。

### 3. Triton softmax demo（单次 kernel，无 benchmark）

```bash
gem5-session start 1
python /home/zhaosiying/zcode-lane/tools/softmax_demo.py
```

单次 Triton kernel：输入在 CPU 生成（GPU 不跑 RNG kernel）、参照
softmax 也在 CPU 算好，设备侧只有一次 H2D 拷贝 + 一次 softmax kernel
+ 一次 D2H 拷贝。默认 4×128 fp32，实测 **约 2 秒** 完成且精确到
0.00000（PASS）。`--rows/--cols/--block/--dtype bfloat16` 可调。若所在
shell 的杂散变量打翻设备栈，脚本自动经干净 env 重启自身，无需手动处理。

### 组件与关键开关

| 工具 | 实现 |
|---|---|
| `rocm-smi` | `tools/gemsim_smi.py`（读 `/tmp/amdgpu-sim-smi-<uid>/` 的 320 字节签名 lease）|
| `gem5-session` | `scripts/gem5_session_control.sh`（listener-mode 多实例 + SMI lease 发布）|
| SMI lease 发布 | `tools/gemsim_smi_publish.py`（flock 持有的 per-slot 记录 daemon）|
| softmax demo | `tools/softmax_demo.py`（`python softmax_demo.py` 直跑）|
| 环境生成 | `scripts/make_amdgpu_tools_env.sh` |

三个已固化进生成器的可移植性陷阱（详因见各工具源码注释）：
gem5 的 evidence 文件拒绝含 symlink 或宽松权限的路径（会话须为真实 0700
/tmp 目录）；WSL 的 /dev/dxg 会把 ROCr thunk 引向 librocdxg 死路
（`HSA_ENABLE_DXG_DETECTION=0` + `HSA_ENABLE_INTERRUPT=0`）；交互 shell
的杂散变量在 hsa_init 打翻设备栈（demo wrapper 经 env -i + unshare 执行）。
