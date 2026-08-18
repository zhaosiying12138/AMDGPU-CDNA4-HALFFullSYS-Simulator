# Handoff — amdgpu-sim, CP-0065 (2026-08-19)

Working tree is clean in all four repositories. A vLLM TP1 lane is running.

## Prompt for codex (zero context)

Paste everything between the markers.

```text
=== BEGIN PROMPT ===
你接手 /home/zhaosiying/amdgpu-sim 的工作。先读这四个文件，它们是权威，不要
依赖任何人的口头总结：
  PLAN.md, GOAL.md, ENGINEERING_CONSTRAINTS.md, docs/HANDOFF-CP-0065.md
再读 git log -12 和最近的 artifacts/evidence/*/manifest.json。

目标（GOAL.md）：让**未经修改的上游** SGLang 和 vLLM 在这个 gem5 模拟器上跑通
Qwen3.5-0.8B TP1 → TP2 → Qwen3.5-9B TP16。TP1/TP2 用 0.8B，TP16 用 9B。
0.8B 的 TP4 已取消。引擎不许改：不注册项目算子、不替换模型/层、不打 monkey
patch。所有修复只能落在 self-runtime / ROCr-HIP facade / runtime-gem5 bridge /
memory / queue / 同步 / collective / ISA 这些通用层。

最重要的工作原则（用户明确要求，必须一直遵守）：
  用最本质、最系统、最一般性的方式修复，绝不 case-by-case。
  动手前先回答：机制是什么（不是报错点在哪）？有多少同类（用 rg -c 数）？
  我依据的判据是语义属性还是身份（指令名/模型名/shape/地址）——身份判据禁止？
  树里有没有权威实现可以照搬？修好这层之后会暴露哪个潜藏缺陷？
  这个修复是不是根本不属于崩溃的那个组件？

进度必须可量化，不许回答"还在跑"：
  bash scripts/lane_progress.sh
  一个 token 大约需要 900 次 kernel launch（实测锚定值）。

【第一优先级 — 必须先查清，不要假设无害】
SGLang TP1 跑到 1449 次 dispatch（已过权重加载、prefill、aiter attention 编译）
后死于：
  AssertionError: One of GPU archs of [''] is invalid or not supported
  来自 aiter/jit/core.py validate_and_update_archs()，它读
  archs = os.getenv("GPU_ARCHS", "native").split(";")
所以有东西把 GPU_ARCHS 设成了**空字符串**。但是：
  - 在完全相同的 lane 环境里直接测，aiter get_gfx() 返回 'gfx950'，
    shutil.which("rocminfo") 正确解析到 artifacts/tool-shim/rocminfo；
  - 失败的那次日志里**既没有** "Pre-warmed aiter GPU_ARCHS=" **也没有**
    "Failed to pre-warm aiter chip info"，而 SGLang 的
    projects/sglang-0.5.17/sglang/srt/utils/aiter.py 只要走过那个 pre-warm
    就必定打印其中一条。
  => 机制**尚未查明**，显而易见的解释已被排除。
复现只要跑 lane；秒级探针用：
  bash scripts/run_engine_lane.sh --engine sglang --tp 1 --capsule <你的.py> /tmp/x.log

【关键警告】CP-0065 提交里凡是标了 UNVERIFIED 的，都是跨会话攒下来、编进了当前
二进制、但**从未被任何一次完整模型跑通验证过**的代码。它们完全可能就是上面这个
失败的根源。重点审计（按嫌疑排序）：
  1. projects/self-amdgpu-runtime  src/rocminfo_model.c —— aiter 就是 shell out
     调它拿 gfx arch 的。它在 topology 缺失/不可读时是不是**只返回非零而不输出**？
     调用方如果只 parse stdout 不看 exit code，拿到的就正好是空字符串。
     也看 src/hsakmt_model.c（设备枚举）和 src/managed_session.c（run root、
     fork/spawn 行为——SGLang 的 scheduler 是 spawn 出来的子进程）。
  2. projects/gem5 的 SDWA 通用层（0f27ef552）——225 处 panic 被一次性替换，
     体量大且未经完整验证。
  3. 9f161c3 改了 SOURCE_LOCK.json。PLAN.md 第 3 节说它 CP-0002 之后
     **byte-immutable**。这次只是**追加**了新 lane（llvm-project 等），没有改动
     任何已有条目，但是否合规需要你裁定。它单独一个 commit，便于审计或回滚。

【当前状态】
  SGLang TP1  最深 1449/900 — 死于上面那个 GPU_ARCHS 问题
  vLLM   TP1  最深  671/900 — 交接时仍在跑（artifacts/lanes/vllm-tp1/status.json）
  TP2 尚未在当前栈上尝试；9B TP16 未开始。
  AgentENV：sandbox 生命周期已打通，里面 torch 能看到设备
  （AMD Instinct MI350X）、Triton target 是 gfx950、上游 Triton softmax 跑过。
  sandbox 里跑完整 SGLang 尚未成功。

【长期运行的纪律】
  - lane 必须用 scripts/run_model_lane.sh 起（setsid 脱离会话，双阈值看门狗：
    先看 dispatch，再看 wavefront 是否推进，判定卡死时**先抓 backtrace 再杀**）。
  - 一条 TP1 lane 只应有一个 gem5。现在 SGLang 会起 3 个，其中 parent python 和
    detokenizer 那两个 dispatch 数为 0，纯浪费。根因已定位：ROCr 在**每个**加载它
    的进程的 hsa_init 里就做 ALLOC/FREE，所以只延迟 GET_VERSION 没用；正确修法是
    在 self-runtime 里做延迟挂载（首次真正需要队列/dispatch 时才 spawn gem5）。
  - 每次长跑之前先确认二进制身份：python3 tools/run_identity_gate.py --format text
    （曾经有一整轮 70 分钟跑的是不含修复的旧 product）。
  - /tmp 是 tmpfs（占 RAM）。别把大文件留在那儿。
  - 先用 capsule 在秒级验证修复，再花 40 分钟跑整模型。capsule 必须跑在 lane
    完全相同的环境和 NVML 隔离里，否则 hsa_init 直接返回 4104。

【尚未兑现的承诺】
  用宿主机 RTX 5090 做**异步差分校验**（每次 kernel launch 交叉验证 gem5 结果）。
  用户明确要求过，至今未实现。边界：只能做诊断、必须计数、必须记进 evidence，
  **永远不能**用来满足验收门禁（GOAL.md 禁止验收路径出现 NVIDIA）。
  背景：ds_swizzle_b32 曾经**静默**算成恒等置换好几个小时，没有任何报错。

每个原子进展都要走 checkpoint / evidence / bitlesson / 跨仓库事务协议，
commit trailer 参照 git log。
=== END PROMPT ===
```

## Trap: `GEMSIM_*` environment knobs cannot reach a managed simulator

`configs/example/gemsim/host_dispatch.py` reads `GEMSIM_LDS_BYTES_PER_CU`
(default 160 KiB, line 47) and `GEMSIM_NUM_COMPUTE_UNITS` (default 1, line 237).
Neither can be set from a lane. `projects/self-amdgpu-runtime/src/managed_session.c`
(~lines 446-463) builds a **fixed seven-entry environment** for the gem5
`posix_spawn` — `PATH`, `HOME`, `TMPDIR`, `XDG_CACHE_HOME`, `LC_ALL`,
`PYTHONNOUSERSITE`, `PYTHONDONTWRITEBYTECODE` — and nothing else is inherited.

So exporting a `GEMSIM_*` variable before a lane silently does nothing; the
defaults always apply. The capsules under `tools/*_capsule/` work around this by
launching gem5 through a wrapper script that injects the variable itself. If you
need a different LDS size or CU count in a lane, either extend the environment
in `managed_session.c` or use the wrapper approach — do not assume an export
took effect.

## Known defect in this history: `60aede62f` does not build standalone

Verified, and it is mine. `60aede62f` (CP-0063) added the definition of
`ComputeUnit::dispatchResourceReport` to `src/gpu-compute/compute_unit.cc`, but
its declaration in `compute_unit.hh` — and the `LdsState` accessors it calls —
only arrive in `e0c9dd267` (CP-0065). `git log -S` confirms the split:

```
declaration added by  e0c9dd267   src/gpu-compute/compute_unit.hh
definition  added by  60aede62f   src/gpu-compute/compute_unit.cc
```

`git -C projects/gem5 HEAD` **does** build: rebuilt at handoff, exit 0, zero
errors. So the tree is sound and only the intermediate commit is not
independently compilable. The consequence is a **bisect hazard**: a bisect that
lands on `60aede62f` or `0f27ef552` will fail to build for a reason unrelated to
whatever is being bisected. Rebase the declaration back into `60aede62f` before
relying on bisect across this range.

This is worth stating plainly because the house rule was met at the mechanism
level — the fix was general, not case-by-case — but not at the level of commit
atomicity.

## What changed under CP-0065

Nine commits, grouped by functional layer, across four repositories.

| Repo | Commit | Layer |
| --- | --- | --- |
| rocm-systems | `0401e8cdb6` | ROCr fast-copy refusal diagnostics |
| self-amdgpu-runtime | `a8044ce` | run-root confinement, KFD model, rocminfo |
| gem5 | `0f27ef552` | generic SDWA operand layer (replaces 225 panics) |
| gem5 | `e0c9dd267` | wavefront counter accounting |
| gem5 | `4fe648f7e` | host bridge + gemsim config |
| root | `3395986` | lane supervision and progress |
| root | `1a46a1d` | bridge tape decoder |
| root | `e020e92` | reproduction capsules and probes |
| root | `2547ce0` | model source pinning, engine examples |
| root | `9f161c3` | **SOURCE_LOCK addition — audit this** |
| root | `98afaec` | coordination, gitlinks, PLAN.md |

Earlier in the session, verified by A/B with negative controls:

- `CP-0063` gem5 host-native atomics (`AbstractMemory` semantics, functor-generic)
  and the disassembly selector table. Capsule: 256 lanes atomically incrementing
  one address leave it at exactly base+256, and the returned values are exactly
  the sequence base..base+255 — 256 distinct, no duplicates, no gaps.
- `CP-0064` device discovery: the product's `rocminfo` first on PATH, and vLLM on
  the formal upstream AMD Triton backend (`ACTIVE_COUNT 1`, active `['amd']`).

## Housekeeping done at handoff

- Zombie processes reaped; exactly one live gem5 (the running vLLM lane).
- 9.7 GB of RAM freed — AgentENV staging tarballs moved from tmpfs to
  `artifacts/sandbox-stage/`; 253 stale run directories and 7 superseded gem5
  binaries removed. `gem5.opt.preatomic` is kept deliberately: it is the negative
  control for the atomics capsule.
