# Goal contract: host-side AMDGPU gem5 simulation

**Goal ID:** `GSIM-001`

**Plan:** `AMDGPU-SIM-V1`, revision `2`

**Current state:** `CP-0008-pinned-dispatch-accepted; next-P2-KMT-ABI-01`

**Current phase:** `P2`

**Model correction:** the official target is `Qwen/Qwen3.5-0.8B`; there is no
official `Qwen3.5-0.9B` checkpoint in this project.

## Success criteria

- A self-contained copy of this directory can prepare and run the pinned stack
  on the same Linux ABI/architecture without network access after preparation.
- Host HIP/OpenCL/Triton/PyTorch/vLLM workloads use the standalone
  `self-amdgpu-runtime` ABI and submit real code objects to gem5.
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
CP-0008 已 accepted；从唯一 next_action `P2-KMT-ABI-01` 继续，不要再次
begin CP-0008，也不要重做 bootstrap、source freeze、authored runtime
baseline 或已通过的 CP4-CP8 handshake/queue-control/memory/signal/dispatch
gates，也不要修改已冻结的 SOURCE_LOCK.json 或已登记的 PROJECT_LANES
baseline。先针对 pinned ROCr/ThunkLoader/libhsakmt source inventory 建立
source-exact ABI map 和 provider skeleton；不要把 CP8 的 single fixture
推广为通用 ROCr、HIP、OpenCL、Triton、PyTorch 或 vLLM 能力。
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
