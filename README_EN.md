# AMDGPU-CDNA4-HALFFullSYS-Simulator

[简体中文](README.md) | [English](README_EN.md)

A gem5-based "HALF-FullSYS" AMD GPU simulator: **the KMD kernel driver is removed and the ROCm runtime is never installed inside a simulated x86 VM**. Instead, an AF_UNIX bridge (plus a sealed-memfd shared device memory) connects the **unmodified host-side ROCm stack** (ROCr/HIP/Triton/PyTorch/aiter/SGLang/vLLM, stock wheels) to a gem5-simulated VEGA ISA + gfx950 decoder + Command Processor. Upstream stays untouched (ROCr: 6 commits, +251/−61 lines; LLVM/HIP/RCCL/Triton/PyTorch/vLLM/SGLang/aiter: zero changes). Multiple gem5 instances and a dual CCL path (stock RCCL plus the in-house `gemsim_ccl` ProcessGroup backend) let **SGLang/vLLM serve Qwen3.5-0.8B at TP2 and SGLang serve Qwen3.5-9B at TP4** with every end-to-end token gate PASS. A functionally layered optimization campaign compressed single-token simulated wall time from a baseline that exceeded a 4 h timeout to **703 s (≥20.5×, conservative lower bound)**, the weight-load path by **28.6×** (exact at every step), and 9B loading by **6.08×**.

| Result | Screenshot |
|---|---|
| SGLang TP4 · Qwen3.5-9B · golden prompt, 20 stable tokens (TTFT/TPOT/load time) | [docs/assets/screenshots/hero-20tok-metrics.png](docs/assets/screenshots/hero-20tok-metrics.png) |
| `rocm-smi` before/after gem5 instances (16 slots, virtual MI350X) | [before](docs/assets/screenshots/smi-before.png) · [after](docs/assets/screenshots/smi-after.png) |
| SGLang TP2 golden token inside an AgentENV sandbox (live re-run PASS 2026-09-05, ~12 min) | [docs/assets/screenshots/agentenv-vmrun-pass.png](docs/assets/screenshots/agentenv-vmrun-pass.png) |
| Operator-correctness regression (softmax + dual-mode HIP capsules) | [docs/assets/screenshots/operator-correctness.png](docs/assets/screenshots/operator-correctness.png) |

## Verified capability matrix

| Capability | Evidence |
|---|---|
| HIP C operators (hsa/hipModuleLoadData capsules: plain_dp / barrier_lds / atomic_decline) | byte-identical output SHA256 across functional-fast vs hybrid modes; `scripts/regression/operator_correctness.sh` |
| Triton kernels (softmax / vecadd / SiluAndMul) | `examples/quickstart/`, `tools/softmax_demo.py` (CPU reference, ~2 s PASS) |
| SGLang TP1 / TP2 · Qwen3.5-0.8B (1-token golden `[27841]`; TP2 also has a 10-token gate) | `scripts/test_qwen35_tp.sh 0.8b-tp2`; archived lanes all PASS |
| vLLM TP1 / TP2 · Qwen3.5-0.8B (same golden) | lanes `zcode-vllm-tp1-v19`, `zcode-vllm-tp2-v4` |
| SGLang TP4 · Qwen3.5-9B (1-token golden `[271]`; an archived 10-token PASS also exists) | `scripts/test_qwen35_tp.sh 9b-tp4`; F1/F2 dual-binary re-verification |
| vLLM TP4 · Qwen3.5-9B | not verified |
| CCL: AllReduce/AllGather/ReduceScatter/Broadcast/Barrier, worlds 2..16 (2/3/4/8/16 verified) | `tests/test_gemsim_ccl_*`, `tools/gemsim_ccl_live_allreduce_acceptance.py` |
| End-to-end inside AgentENV sandboxes (SGLang TP2 golden token) | `tools/agentenv/vm_run_sglang.sh`; live re-run PASS 2026-09-05 (in-sandbox weight load ~330 s, ~12 min end to end); archived `artifacts/agentenv-vm-tp2/vmrun.log` (2026-08-26) |

> Model-inference screenshots use the **1-token golden PASS** convention (multi-token stability is covered by the 20-token demo and the archived 10-token gate) — showcase experiments run once, correctness is backed by fail-closed gates.

## Performance (frozen measurements, 2026-09)

| Metric | Baseline | All optimizations | Speedup |
|---|---|---|---|
| 0.8B TP1 single-token wall (SGLang, CU16) | ≥14400 s (4 h censored, lower bound) | **703 s** | **≥20.5×** |
| 0.8B weight load (exact at every step) | 3355.1 s | **117.1 s** | **28.6×** |
| 9B TP4 weight load (slowest rank) | 1272.95 s | **209.40 s** | **6.08×** |
| 9B TP4 single-token wall | 4862 s | **1788 s** | **2.72×** |
| hybrid CTA admission (real-model diagnostic) | — | 82.5% of launches / 83.4% of workgroups | fail-closed static screen; declined kernels fall back to full timing |

Layer contributions (load path): DTIF fast copy 35.1% > functional-fast 28.4% > KMT mapping cache 26.2% > hybrid CTA 9.3% > idle park/progress 0.9%. Methodology, ablation ladder, and per-layer data: `docs/blog/2026-09-amdgpu-cdna4-halffullsys/` (data sources `data/*.json`).

## Quick start

### 0. Prerequisites

- x86_64 Linux (developed and verified on WSL2 Ubuntu); ≥24 cores recommended; ~500 GB disk for sources, builds, and both models.
- conda (miniforge is fine); `/dev/kvm` only for the AgentENV path.
- Model checkpoints: `models/Qwen3.5-0.8B` (HF revision `2fc06364715b967f1860aea9cf38778875588b17`, safetensors SHA-256 `04b1c301…f1fe4696`) and `models/Qwen3.5-9B`.

### 1. Sources and build

```bash
git clone <this repo> && cd AMDGPU-CDNA4-HALFFullSYS-Simulator
./scripts/materialize_sources.sh          # check out all upstream trees per SOURCE_LOCK.json (verified)
bash scripts/build_gem5_mold24.sh         # build projects/gem5/build/VEGA_X86/gem5.opt
./scripts/setup_conda_env.sh --install    # conda product prefix + ROCr stage + runtime
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$(./scripts/setup_conda_env.sh --print-prefix)"
```

Build details (mold/24-job record): [docs/gem5-build.md](docs/gem5-build.md); `./scripts/setup_conda_env.sh --verify` re-verifies product/runtime/plugin fingerprints.

### 2. Smoke: the simulated GPU is visible + one Triton kernel

```bash
bash scripts/make_amdgpu_tools_env.sh    # create the AMDGPU-CDNA4-SIM tools env (idempotent, seconds)
conda activate AMDGPU-CDNA4-SIM
rocm-smi                                 # 16 slots all OFF; --json for scripts
gem5-session start 1                     # start one gem5 instance (seconds)
rocm-smi                                 # slot 0 = ON with pid/rank/endpoint
triton-softmax                           # one Triton kernel vs CPU reference (~2 s, PASS)
gem5-session stop
```

### 3. Operator-correctness regression

```bash
bash scripts/regression/operator_correctness.sh          # full (includes 2048-WG stress ×3)
bash scripts/regression/operator_correctness.sh --quick  # smoke level
```

Checks: Triton softmax vs CPU reference; three HIP capsules byte-identical (SHA256) across functional-fast vs hybrid modes (plain_dp carries an exact oracle); 2048-WG stress ×3. Writes `artifacts/operator-correctness-regression/summary.md`; any mismatch exits non-zero.

### 4. Performance regression (perf-only, no correctness assertion)

```bash
bash scripts/regression/perf_bench.sh [--with-baseline] [--tokens N] [--out DIR]
```

Two arms (SGLang TP1 · 0.8B · 1 token · CU16 · warm cache, serialized on an otherwise idle host): `legacy` (fast copy / idle park off) and `full` (everything on), reporting wall/load_weight/kv/scheduler/request latency and retired dispatch counts (`<out>/summary.md` plus per-arm `metrics.json`). `--with-baseline` adds the bugfix-only binary arm (`ASIM_GEM5_BASE` points at the gem5 `8cd1db918` tree); note the accurate (no functional-fast) baseline hits the known scratch-admission race (see Known limitations) and is not part of the quick bench.

### 5. End to end: SGLang Qwen3.5-9B · TP4 · golden prompt

```bash
bash scripts/test_qwen35_tp.sh 9b-tp4 --tokens 1
```

Uses the fixed prompt 「为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？」 with expected token `[271]`; fail-closed `report.json` (token golden, gem5 panic scan, NCCL watchdog, HIP 209, stray processes). 0.8B TP2: `bash scripts/test_qwen35_tp.sh 0.8b-tp2 --tokens 1`. Multi-token demo (TTFT/TPOT): `python tools/demos/demo_sglang_tp4.py --max-tokens N`.

### 6. Running inside AgentENV sandboxes (optional)

```bash
# One-time: install the AgentENV server (sudo required, see docs/AGENTENV_SERVICE.md)
# After a WSL restart /run is tmpfs and empty; pre-create it as root or the
# service's ExecStartPre fails to mkdir:
#   sudo mkdir -p /run/aenv && sudo chown aenv:aenv /run/aenv
# Regular flow:
aenv start --cold dockerproxy.net/library/ubuntu:24.04 --cpu 8 --memory 32768 \
  --disk-size-mb 65536 <sandbox-id>
./tools/agentenv/publish_sim_stack.sh --full <sandbox-id>   # one-key toolchain publish (6 streams)
aenv exec <sandbox-id> -- bash tools/agentenv/vm_run_sglang.sh   # in-sandbox SGLang TP2 golden token
```

Strategy: upstream wheels preinstalled in the image; in-house sources (gem5/runtime/ROCr stage) compiled host-side and published with one key; sandboxes stay isolated from each other (see `tools/agentenv/`).

## Known limitations (stated honestly)

1. **KMT scratch-admission race (unfixed)**: slow timing configurations (accurate + legacy copy + no idle park) can trigger the race at `host_gpu_bridge.cc:3695` and hang (the censored ablation arms hit this; the fully optimized configuration never has). Reproduction material: `artifacts/blog-perf-2026-09/results/L0-attempt2-stall-forensics/`.
2. The hybrid CTA executor's functional stepping is serial (~3–4 ms/WG, does not scale with CU count); decode memoization and light_stats gating are on the backlog.
3. TP>1 CCL has correctness/stability fixes only, no performance work.
4. The layer gate's diffing hooks accumulate memory; a 24-layer comparison gets OOM-killed at layer 19 (all covered layers pass).
5. simTicks under functional-fast/hybrid must not be used for timing conclusions (the identity banner records this per run).

## Provenance and governance

- Upstream baselines: `SOURCE_LOCK.json` (annotated immutable tags per tree); exported patches: `patches/gem5` (42), `patches/rocm-systems` (5).
- Identity banner on every lane's first lines: repo head plus SHA-256 of ROCr/runtime/model DSO/gem5 binaries.
- Design docs: `docs/host-native-architecture.md`, `docs/runtime-gem5-bridge-migration.md`, `docs/framework-runtime-layering.md`.
- Full technical report (ablation ladder, KMD responsibility migration, complete bug-fix record): `docs/blog/2026-09-amdgpu-cdna4-halffullsys/`.

## Acknowledgments

- [gem5](https://github.com/gem5/gem5) and its [Full System AMD GPU model](https://www.gem5.org/documentation/general_docs/gpu_models/gpufs) — the simulation core and its VEGA/amdgpu device model.
- [AgentENV](https://github.com/kvcache-ai/AgentENV) (Moonshot AI & kvcache-ai) — sandbox infrastructure.
- [ROCm/rocm-systems](https://github.com/ROCm/rocm-systems) — the ROCr/ROCclr/RCCL upstream.
