# AMDGPU-CDNA4-HALFFullSYS-Simulator

[简体中文](README.md) | [English](README_EN.md)

基于 gem5 的 AMD GPU "半全系统"（HALF-FullSYS）模拟器：**去掉 KMD 内核驱动，也不把 ROCm Runtime 装进模拟的 x86 虚拟机**——一座 AF_UNIX bridge（+ sealed memfd 共享显存）把**原封不动的宿主侧 ROCm 软件栈**（ROCr/HIP/Triton/PyTorch/aiter/SGLang/vLLM，原生 wheel）接到 gem5 模拟的 VEGA ISA + gfx950 decoder + Command Processor 上。对上游保持零修改（ROCr 仅 6 commits、+251/−61 行；LLVM/HIP/RCCL/Triton/PyTorch/vLLM/SGLang/aiter 一行未动），多 gem5 实例 + 双 CCL 路径（原版 RCCL 与自研 `gemsim_ccl` ProcessGroup 后端）支撑 **SGLang/vLLM 以 TP2 跑通 Qwen3.5-0.8B、SGLang 以 TP4 跑通 Qwen3.5-9B**，端到端 token golden 全部 PASS；一轮按功能分层的优化把单 token 模拟墙钟从基线的 4 h 超时未完成压到 **703 s（≥20.5×，保守下界）**、权重加载路径 **28.6×**（全精确）、9B 加载 **6.08×**。

| 成果 | 截图 |
|---|---|
| SGLang TP4 · Qwen3.5-9B · 金色 prompt 稳定推理 20 tokens（TTFT/TPOT/加载耗时） | [docs/assets/screenshots/hero-20tok-metrics.png](docs/assets/screenshots/hero-20tok-metrics.png) |
| `rocm-smi`：gem5 实例启动前/后（16 槽位、MI350X 虚拟卡） | [启动前](docs/assets/screenshots/smi-before.png) · [启动后](docs/assets/screenshots/smi-after.png) |
| AgentENV 沙箱内 SGLang TP2 golden token（归档 2026-08-26） | [docs/assets/screenshots/agentenv-vm-tp2.png](docs/assets/screenshots/agentenv-vm-tp2.png) |
| 算子正确性回归（softmax + HIP 双模式胶囊） | [docs/assets/screenshots/operator-correctness.png](docs/assets/screenshots/operator-correctness.png) |

## 已验证能力矩阵

| 能力 | 证据 |
|---|---|
| HIP C 算子（hsa/hipModuleLoadData 胶囊：plain_dp / barrier_lds / atomic_decline） | 双模式（functional-fast vs hybrid）输出 SHA256 逐字节一致；`scripts/regression/operator_correctness.sh` |
| Triton kernel（softmax / vecadd / SiluAndMul） | `examples/quickstart/`、`tools/softmax_demo.py`（CPU 参照，实测 ~2 s PASS） |
| SGLang TP1 / TP2 · Qwen3.5-0.8B（1 token golden `[27841]`；TP2 另有 10-token gate） | `scripts/test_qwen35_tp.sh 0.8b-tp2`；归档 lane 全 PASS |
| vLLM TP1 / TP2 · Qwen3.5-0.8B（同 golden） | lane `zcode-vllm-tp1-v19`、`zcode-vllm-tp2-v4` |
| SGLang TP4 · Qwen3.5-9B（1 token golden `[271]`；另归档 10-token PASS） | `scripts/test_qwen35_tp.sh 9b-tp4`；F1/F2 双二进制复验 |
| vLLM TP4 · Qwen3.5-9B | 未验证 |
| CCL：AllReduce/AllGather/ReduceScatter/Broadcast/Barrier，world 2..16（验证 2/3/4/8/16） | `tests/test_gemsim_ccl_*`、`tools/gemsim_ccl_live_allreduce_acceptance.py` |
| AgentENV 沙箱内端到端（SGLang TP2 golden token） | `tools/agentenv/vm_run_sglang.sh`；归档 `artifacts/agentenv-vm-tp2/vmrun.log` |

> 模型推理的截图均为 **1 token golden PASS** 口径（多 token 稳定性由 20-token 演示与归档 10-token gate 证明）——为了让展示实验只做一次，正确性由 fail-closed 门禁背书。

## 性能（2026-09 冻结实测）

| 指标 | 基线 | 全优化 | 加速 |
|---|---|---|---|
| 0.8B TP1 单 token 墙钟（SGLang，CU16） | ≥14400 s（4 h 截断，下界） | **703 s** | **≥20.5×** |
| 0.8B 权重加载（逐级精确） | 3355.1 s | **117.1 s** | **28.6×** |
| 9B TP4 权重加载（最慢 rank） | 1272.95 s | **209.40 s** | **6.08×** |
| 9B TP4 单 token 墙钟 | 4862 s | **1788 s** | **2.72×** |
| hybrid CTA 接纳率（真实模型负载诊断） | — | 82.5% launch / 83.4% workgroup | fail-closed 静态筛选；被拒 kernel 回落完整时序 |

分层贡献（加载路径）：DTIF fast copy 35.1% > functional-fast 28.4% > KMT mapping cache 26.2% > hybrid CTA 9.3% > idle park/progress 0.9%。口径、消融阶梯与逐层数据见 `docs/blog/2026-09-amdgpu-cdna4-halffullsys/`（数据源 `data/*.json` 可溯源）。

## 快速开始

### 0. 前置条件

- x86_64 Linux（本仓库在 WSL2 Ubuntu 上开发验证）；24 核以上推荐；磁盘 ~500 GB（源码+构建+两个模型）。
- conda（miniforge 即可）；`/dev/kvm` 仅 AgentENV 路径需要。
- 模型 checkpoints 放置：`models/Qwen3.5-0.8B`（HF revision `2fc06364715b967f1860aea9cf38778875588b17`，safetensors SHA-256 `04b1c301…f1fe4696`）与 `models/Qwen3.5-9B`。

### 1. 源码与编译

```bash
git clone <本仓库> && cd AMDGPU-CDNA4-HALFFullSYS-Simulator
./scripts/materialize_sources.sh          # 按 SOURCE_LOCK.json 检出全部上游树（含校验）
bash scripts/build_gem5_mold24.sh         # 构建 projects/gem5/build/VEGA_X86/gem5.opt
./scripts/setup_conda_env.sh --install    # conda 产品前缀 + ROCr stage + runtime
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$(./scripts/setup_conda_env.sh --print-prefix)"
```

构建细节（mold/24-job 记录）见 [docs/gem5-build.md](docs/gem5-build.md)；`./scripts/setup_conda_env.sh --verify` 复核产品/运行时/插件指纹。

### 2. 冒烟：模拟卡可见 + 一次 Triton kernel

```bash
bash scripts/make_amdgpu_tools_env.sh    # 生成 AMDGPU-CDNA4-SIM 工具环境（幂等，秒级）
conda activate AMDGPU-CDNA4-SIM
rocm-smi                                 # 16 槽位全 OFF；--json 供脚本
gem5-session start 1                     # 拉起 1 个 gem5 实例（秒级）
rocm-smi                                 # 槽位 0 = ON，含 pid/rank/endpoint
triton-softmax                           # 单次 Triton kernel vs CPU 参照（~2 s，PASS）
gem5-session stop
```

### 3. 算子正确性回归

```bash
bash scripts/regression/operator_correctness.sh          # 全量（含 2048-WG 压力 ×3）
bash scripts/regression/operator_correctness.sh --quick  # 冒烟级
```

检查项：Triton softmax vs CPU 参照；三个 HIP 胶囊在 functional-fast 与 hybrid 两种模式下输出 SHA256 逐字节一致（plain_dp 带精确 oracle）；2048-WG 压力 ×3。输出 `artifacts/operator-correctness-regression/summary.md`，任何不一致非零退出。

### 4. 性能回归（纯性能，不做正确性断言）

```bash
bash scripts/regression/perf_bench.sh [--with-baseline] [--tokens N] [--out DIR]
```

两臂（SGLang TP1 · 0.8B · 1 token · CU16 · warm cache，串行独占主机）：`legacy`（快拷/空闲停泊关闭）与 `full`（全开），输出 wall/load_weight/kv/调度/请求时延与 retired dispatch 数的汇总表（`<out>/summary.md` 与逐臂 `metrics.json`）。`--with-baseline` 额外加 bugfix-only 二进制臂（`ASIM_GEM5_BASE` 指向 gem5 `8cd1db918` 树）；注意 accurate（无 functional-fast）基线会命中已知 scratch 准入竞态（见"已知限制"），不在快测范围。

### 5. 端到端：SGLang Qwen3.5-9B · TP4 · 金色 prompt

```bash
bash scripts/test_qwen35_tp.sh 9b-tp4 --tokens 1
```

固定 prompt「为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？」，期望 token `[271]`；fail-closed `report.json`（token golden、gem5 panic 扫描、NCCL watchdog、HIP 209、残留进程五项）。0.8B TP2：`bash scripts/test_qwen35_tp.sh 0.8b-tp2 --tokens 1`。多 token 演示（TTFT/TPOT）：`python tools/demos/demo_sglang_tp4.py --max-tokens N`。

### 6. AgentENV 沙箱内运行（可选）

```bash
# 一次性：安装 AgentENV 服务端（需 sudo，详见 docs/AGENTENV_SERVICE.md）
# 常规流程：
aenv start --cold dockerproxy.net/library/ubuntu:24.04 --cpu 8 --memory 32768 \
  --disk-size-mb 65536 <sandbox-id>
./tools/agentenv/publish_sim_stack.sh --full <sandbox-id>   # 一键发布全套工具链（6 个流）
aenv exec <sandbox-id> -- bash tools/agentenv/vm_run_sglang.sh   # 沙箱内 SGLang TP2 golden token
```

策略：上游 wheel 预装镜像；自研源码（gem5/runtime/ROCr stage）host 侧编译 + 一键发布；多沙箱互不干扰（详见 `tools/agentenv/`）。

## 已知限制（如实）

1. **KMT scratch 准入竞态（未修）**：accurate + legacy copy + 无 idle park 的慢速配置可触发 `host_gpu_bridge.cc:3695` 竞态挂起（性能消融的截断臂源于此；全优化配置未触发）。复现材料：`artifacts/blog-perf-2026-09/results/L0-attempt2-stall-forensics/`。
2. hybrid CTA 的功能 WG 步进是串行的（~3–4 ms/WG，不随 CU 数扩展）；decode memoization 与 light_stats 门控在 backlog。
3. TP>1 的 CCL 仅有正确性/稳定性修复，无性能优化。
4. layer gate 的 diffing 钩子有内存累积，24 层比到第 19 层会被 OOM killer 终止（已覆盖层全部通过）。
5. functional-fast/hybrid 模式 simTicks 不可用于时序结论（identity banner 强制登记）。

## 溯源与治理

- 上游基线：`SOURCE_LOCK.json`（每棵树带注释的不可变 tag）；导出补丁：`patches/gem5`（42 个）、`patches/rocm-systems`（5 个）。
- 每条 lane 首行 identity banner：repo head、ROCr/runtime/模型 DSO/gem5 二进制的 SHA-256。
- 设计文档：`docs/host-native-architecture.md`、`docs/runtime-gem5-bridge-migration.md`、`docs/framework-runtime-layering.md`。
- 完整技术报告（含消融阶梯、KMD 职责迁移映射、bug fix 全记录）：`docs/blog/2026-09-amdgpu-cdna4-halffullsys/`。

## 致谢

- [gem5](https://github.com/gem5/gem5) 及其 [Full System AMD GPU model](https://www.gem5.org/documentation/general_docs/gpu_models/gpufs)——本项目的模拟核心与其 VEGA/amdgpu 设备模型。
- [AgentENV](https://github.com/kvcache-ai/AgentENV)（Moonshot AI & kvcache-ai）——沙箱基础设施。
- [ROCm/rocm-systems](https://github.com/ROCm/rocm-systems)——ROCr/ROCclr/RCCL 上游。
