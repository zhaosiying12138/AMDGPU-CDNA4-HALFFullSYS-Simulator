# HANDOVER: Hybrid CTA 剩余缺陷修复 + 9B TP4 稳定跑通

**写给**: Codex（接手人） · **写于**: 2026-08-28 · **CP**: CP-0154
**核心目标**: **9B 模型 TP4 快速稳定跑通**（一 token golden compare pass）。
**风格要求**: 直接了当修 bug、系统性解决竞争、不要过度设计。

---

## 1. 仓库/环境速查

| 项 | 值 |
|---|---|
| lane 仓库 | `/home/zhaosiying/zcode-lane`（branch `zcode/sglang-tp1-dpp`，远端 `public`） |
| gem5 worktree | `/home/zhaosiying/zcode-gem5-hybrid2`（branch `zcode/gem5-hybrid-cta2`，基线 `a6e9be550`） |
| 基线保护 tag | `all-models-verified-20260826` → `a8d85a5`（**不允许 force push / amend**） |
| 验证 lane | `scripts/run_engine_lane.sh --engine sglang --tp N [--fast] [--model <path>] <logfile>` |
| 两个 wrapper | `build/VEGA_X86/gem5.opt.fastwrap`（速赢档）、`gem5.opt.hybridfastwrap2`（hybrid+fast） |
| 关键环境 | `SAGR_MANAGED_GEM5=<wrapper>` + `SAGR_MANAGED_GEM5_CONFIG=<worktree>/configs/.../host_dispatch.py` |
| runroot | **必须放 `/home/zhaosiying/r...`**（AF_UNIX ≤108 字节；`/tmp` 会被 WSL 清理器几分钟一轮吞掉产物！） |

**每次跑之前**：`pkill -9 -f qwen35; for p in $(pgrep gem5); do kill -9 $p; done`——残留引擎/gem5 会占 gdb 端口、吃 CPU、毒化后续会话。

---

## 2. 已完成（不需要重做）

| 成果 | 证据 |
|---|---|
| Hybrid CTA 重实现（源码曾丢失） | 3 capsule 字节级 MATCH 历史 hash（`deebc5e9/ad7facb2/ea8a8838`），gem5 commit `88162bb3f` |
| **速赢优化（idle park + 进度追踪默认关）** | 0.8B TP2 golden `[27841]` PASS **742s vs 基线 1466s（1.98×）**，`artifacts/perf-verify/20260828-tp2/lane.log` |
| 缺陷①修复：分片执行（128 WG/tick） | capsule 全 MATCH；commit `9d2d9c250` |
| spin guard + wall watchdog | gem5 最新 commit（32M 指令预算 + 120s 看门狗，都会 panic 出 PC+指令名） |
| 消融旋钮 | `SAGR_LANE_CACHE_ROOT` / `SAGR_LANE_FASTCOPY_MODE` / `SAGR_LANE_AUTOTUNE_MODE`（默认不变） |
| 轻量复现工具 | `tools/hybrid_cta_capsule/mini_engine_construct.py`（TP1+dummy 权重，~90s 到崩溃窗口）；`tools/crashbt/repro_stalk.sh`（gdb 附着 spawn 子进程抓原生栈）；`tools/crashbt/crashbt.so`（原生 SIGSEGV 回溯） |

---

## 3. 当前卡点（缺陷②：hybrid 引擎停滞）——你要修的核心

### 症状（确定性复现，~15 分钟）
1. 跑 `tools/crashbt/run_dbg_lane.sh`（或直接 mini engine lane + `hybridfastwrap2`）
2. 权重加载完后进入 mamba cache 初始化（2048-WG 大 kernel，`grid=[262144]×wg=[128]`）
3. functional WG 以 ~1900/s 速率退役，然后在 **某个 WG 编号（1362/1461/989，每次不同）突然冻结**
4. gem5 保持 90% CPU 但不再产出 `wg compl`；引擎侧 `AsyncEventsLoop` 忙旋（设计常态）
5. **spin guard 和 wall watchdog 都已编入但当时二进制未跑完验证**——最新 gem5 commit 含这两个仪表，重建后重跑应能打出 panic（PC+指令名）

### 已排除的假设
- ❌ 算子级数据错：arange/invfreq/pow 全尺寸（16→1048576）hybrid==accurate 字节一致 + OOB canary 完好
- ❌ KMT EventWait stub 立即返回：忙旋路径根本走不到它（BusyWaitSignal 是设计行为）
- ❌ transport 阻塞：`kmt_shim.c` 的 exchange 是真阻塞 poll

### 最可能的根因方向（按优先级）
1. **decodeAt→native->read 偶发阻塞/极慢**：gem5 在 executor 内读指令字节走 `HostKmtNativeMemory::read`。如果 backing 为 null 走 `state.readMemory`（transport 往返），在某些状态下会挂。**查**：`shader->nativeMemory()` 绑定的是哪个实例、`backing` 是否为 null。
2. **某个指令的解释在特定数据下死循环但每步都"progress"**：watchdog 会抓到（120s panic 带指令名）。
3. **yield 逻辑与 WG0 timing 路径的竞争**：分片 yield 时 WG0 可能还在 timing 管线里，`failStalledDispatch` 只在所有 CU activeWaves==0 时 panic——边缘组合可能静默丢 tick。**查**：yield 返回 false 后 `exec()` 的重排队路径、`scheduleDispatch()` 是否覆盖所有分支。

### 修复后验收阶梯（全部单 lane 串行，不许并行）
```
3 capsule（字节级 MATCH）
→ mini engine construct（MINI_ENGINE_PASS，重复 ×3）
→ TP1 layer gate（--debug-layer-gate，层张量 vs NVIDIA golden）
→ 0.8B TP2 一 token golden [27841]
→ 9B TP4 一 token golden（--model /home/zhaosiying/zcode-lane/models/Qwen3.5-9B）
```

---

## 4. 遗留独立问题

### 4a. 9B TP4 NCCL watchdog 挂死（与 hybrid 无关，非 hybrid 档也挂过）
- **症状**：rank1 `ProcessGroupNCCL watchdog got stuck for 480 seconds`
- **轻量复现工具（现成）**：`tools/gemsim_ccl_live_allreduce.py`（standalone 设备侧 allreduce N=2..16，不经模型）+ `tools/qwen35_nccl_allreduce_capsule.py`（两进程 NCCL allreduce）
- **方向**：大概率是 collective kernel 在模拟器上活锁（历史上 dispatch 1618/1619 处曾确定性活锁过）；或 NCCL init 的 host 侧自旋与 gem5 时序不匹配。用 allreduce capsule 直接压。

### 4b. P0 性能优化（缺陷修完后做）
| 优化 | 预估收益 | 风险 |
|---|---|---|
| decode memoize（executor 内 `decodeAt` 每条指令都重新 decode+new GPUStaticInst）| executor 路径 ~2× | 低（executor 局部缓存 keyed by PC） |
| light_stats 门控（配置管道已接好，热点块未完成）| timing 路径 1.3-1.5× | 极低（纯统计） |
配置管道：`GPU.py` 已有 `light_stats` Param；`host_dispatch.py` 传参行**当时为绕过二进制不匹配临时注释掉了**（`Shader(... hybrid_cta=args.hybrid_cta, ...)`），重建后加回 `light_stats=args.functional_fast`。

---

## 5. 消融实验（最后做，全部单 lane）

| 档 | 设置 | 状态 |
|---|---|---|
| C2 现状 | warm+correctness+fastcopy fast | ✅ golden PASS 1466s（`artifacts/perf-ablation-2026-08/c2-status-quo`）|
| C1 legacy | `SAGR_LANE_FASTCOPY_MODE=legacy` | 未跑 |
| C0' 冷+correct | `SAGR_LANE_CACHE_ROOT=<新目录> SAGR_LANE_FASTCOPY_MODE=legacy` | 未跑 |
| C0 冷+device | 同上 + `SAGR_LANE_AUTOTUNE_MODE=device` | 未跑（2h 硬顶）|

跑法：`tools/perf_ablation/run_ablation.sh <config> [timeout]`；解析：`tools/perf_ablation/parse_results.py`。

---

## 6. 操作纪律（血泪教训）

1. **一次只跑一条 lane**；跑前 `pkill` 残留。残留引擎+gem5 会把 load 顶到 135/24 核并冻死 shell。
2. **所有产物放工作区** `artifacts/...`；`/tmp` 被 WSL 清理器几分钟一轮吞掉（吞过 9B 产物和 core 文件）。
3. runroot 用 `/home/zhaosiying/r...` 短路径（AF_UNIX 108 上限）。
4. 长任务用 `setsid` + 工作区脚本；Bash tool 的 `&` 在工具退出时会被杀。
5. gem5 二进制 1.1GB 带 debug info——gdb attach 加载符号可能超 90s 超时；用 `timeout 200+` 或 raw-address 模式。
6. 每次改 gem5 源码后 `scons -j12`（~8 分钟全量链接）；改 Python 配置不需要重编。
