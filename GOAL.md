# Goal contract: host-side AMDGPU gem5 simulation

**Goal ID:** `GSIM-001`  
**Plan:** `AMDGPU-SIM-V1`, revision `1`  
**Current state:** `paused-after-bootstrap`  
**Current phase:** `P0`  
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

## In scope

ROCr/HIP/CLR/libhsakmt-compatible host glue, gem5 host bridge/daemon,
gfx950/MI355X code-object execution, functional N-rank transport and RCCL
semantics, Triton AMD lowering adapter, PyTorch and vLLM integration, Qwen
GDN/full-attention layers, reproducible tests, evidence, checkpoints, and
offline packaging.

## Out of scope until explicitly promoted

Vision execution, real hardware validation, silent CPU or mock execution,
unbounded ROCm API compatibility, timing claims before FabricModel, elastic
rank shrink, and committing weights/build products/environments.

## Resume contract

Do not infer progress from prose or an old chat.  Read `PLAN.md`, this file,
`SOURCE_LOCK.json`, `state/current.json`, the referenced checkpoint, and all
bitlessons named there.  Run:

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
根据 next_action 从 P0-SRC-01 继续；不要重做已通过的 bootstrap。先在线
复核官方精确版本并拉取上游，保留上游历史和 baseline 标记；每个原子修改
使用新的 Checkpoint-ID，先提交 child 再提交 root coordinator commit，并
同步 checkpoint、bitlesson、evidence。遇到长耗时或 token 切换，先写
partial checkpoint，更新 current.json 的唯一 next_action，再暂停。
```
