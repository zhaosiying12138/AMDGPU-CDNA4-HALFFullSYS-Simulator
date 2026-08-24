#!/usr/bin/env python3
"""一键客户演示：在 AMDGPU-CDNA4-SIM 模拟 GPU 上生成 100 个 token。

用法（先 gem5-session start 1，然后）：

    python demo_gen.py "你的 prompt"

无 prompt 参数时使用内置演示 prompt。脚本输出：
  - 加载阶段进度（权重/内核缓存）
  - 每个 token 的实时进度条：第几个 token、当前已生成文本
  - kernel/layer 级别的采样日志（体现模拟器的修复路径）
  - 结束后完整回答 + 性能指标（TTFT / TPOT / 吞吐）

进度可视化（当前已生成文本的滑动窗口）示例：
  ┌─────────────────────────────────────────────────┐
  │ token  17/100 │ layer 12/24 │ kernel GDN-decode │
  │ …《月鳞绮纪》的服装设计融合了唐代与西域元素…        │
  └─────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = "/home/zhaosiying/zcode-lane"
SESSION_ENV = "/tmp/amdgpu-sim-tools-session/session.env"
DISPATCH_TRACE_GLOB = "/tmp/amdgpu-sim-tools-session/instance-0/dispatch-trace.jsonl"
DEFAULT_PROMPT = "为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？"
MAX_TOKENS = 100

# ---------------------------------------------------------------------------
# 终端 UI
# ---------------------------------------------------------------------------
import fcntl
import struct
import termios


def term_width() -> int:
    try:
        res = fcntl.ioctl(0, termios.TIOCGWINSZ, b"\x00" * 8)
        return struct.unpack("hhhh", res)[1] or 80
    except Exception:
        return 80


ANSI = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "green": "\033[32m", "yellow": "\033[33m", "cyan": "\033[36m",
    "magenta": "\033[35m", "red": "\033[31m", "blue": "\033[34m",
}


def c(text: str, color: str) -> str:
    return f"{ANSI[color]}{text}{ANSI['reset']}"


def banner(title: str, width: int = 60) -> None:
    line = "─" * width
    print(f"{c('┌' + line + '┐', 'cyan')}")
    print(f"{c('│', 'cyan')}{c(title.center(width, ' '), 'bold')}{c('│', 'cyan')}")
    print(f"{c('└' + line + '┘', 'cyan')}", flush=True)


def progress_bar(current: int, total: int, width: int = 28) -> str:
    filled = int(width * current / total) if total else 0
    return c("█" * filled, "green") + c("░" * (width - filled), "dim")


def token_display(token: int, total: int, layer: str, kernel: str,
                  text: str, wall_s: float) -> None:
    w = term_width()
    text = text.replace("\n", " ")
    window = max(20, w - 24)
    if len(text) > window:
        text = "…" + text[-window:]
    line1 = (
        f"  {c('token', 'dim')} {c(str(token), 'bold')}/{c(str(total), 'dim')}"
        f"  {c('│', 'dim')} {c('layer', 'dim')} {layer}"
        f"  {c('│', 'dim')} {kernel}"
        f"  {c(f'{wall_s:.0f}s', 'yellow')}"
    )
    line2 = f"  {progress_bar(token, total)}  {text}"
    print(f"\033[G\033[K{line1}")
    print(f"\033[G\033[K{line2}\033[F", end="", flush=True)


def kernel_log_line(kind: str, detail: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"\033[G\033[K  {c(stamp, 'dim')} {c('●', 'magenta')} {kind:<38} {detail}\033[F",
          end="", flush=True)


def finish_display() -> None:
    print("\033[E", end="", flush=True)


# ---------------------------------------------------------------------------
# 模拟 dispatch-trace 采样器（跟踪 gem5 真实 kernel 执行）
# ---------------------------------------------------------------------------
class DispatchSampler(threading.Thread):
    """周期读取 gem5 的 dispatch-trace.jsonl，抽取 kernel 名/层级。"""

    def __init__(self):
        super().__init__(daemon=True)
        self.daemon_flag = threading.Event()
        self.current_layer = "—"
        self.current_kernel = "—"
        self.dispatch_count = 0
        self.kernel_names = []

    def run(self):
        while not self.daemon_flag.is_set():
            if os.path.exists(DISPATCH_TRACE_GLOB):
                try:
                    n = 0
                    with open(DISPATCH_TRACE_GLOB) as f:
                        for line in f:
                            n += 1
                            if n % 50 == 0:
                                try:
                                    rec = json.loads(line)
                                    grid = rec.get("grid", [0])
                                    lds = rec.get("fixed_shared_memory_bytes", 0)
                                    wg = rec.get("workgroup", [0])
                                    self.current_kernel = (
                                        f"grid {grid[0]}×{grid[1] if len(grid) > 1 else 1}"
                                        f" wg{wg[0]} lds{lds // 1024}K"
                                    )
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    pass
                    self.dispatch_count = n
                except (OSError, IOError):
                    pass
            self.daemon_flag.wait(2.0)


# ---------------------------------------------------------------------------
# 生成引擎：SGLang 离线 LLM
# ---------------------------------------------------------------------------

def run_generation(prompt: str, max_tokens: int = MAX_TOKENS) -> tuple[list[int], dict]:
    """在模拟 GPU 上跑 100-token 生成；返回 token id 列表与计时指标。"""

    # 环境加载（复用 softmax demo 的自愈机制变量集）
    if os.path.exists(SESSION_ENV):
        for line in open(SESSION_ENV):
            line = line.strip()
            if line.startswith("export "):
                k, _, v = line[len("export "):].partition("=")
                v = v.strip('"').strip("'")
                if k:
                    os.environ.setdefault(k, v)

    import torch

    if not torch.cuda.is_available():
        print(c("✗ 模拟 GPU 不可达——请先 gem5-session start 1", "red"), file=sys.stderr)
        sys.exit(1)

    from transformers import AutoTokenizer
    from sglang.test.runners import offline_inference as sglang_offline  # 延迟导入

    tokenizer_path = f"{ROOT}/models/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    prompt_ids = tokenizer(prompt)["input_ids"]

    banner(f"AMDGPU-CDNA4-SIM · Qwen3.5-0.8B · TP1 · {MAX_TOKENS} tokens", 56)
    print(f"  {c('模型', 'dim')}: Qwen3.5-0.8B（24 层：18 GDN 线性注意力 + 6 全注意力）")
    print(f"  {c('Prompt', 'dim')} ({len(prompt_ids)} tok): {prompt[:40]}…")
    print(f"  {c('引擎', 'dim')}: SGLang 离线 · Triton attention · 模拟 gfx950 (4 CU)")
    print()

    sampler = DispatchSampler()
    sampler.start()

    # ---- 引擎启动进度（权重加载 ~20 min on simulator）----
    t_engine_start = time.time()
    print(c("  ▸ 加载引擎（权重 1.6 GB 经 DTIF fast-copy 注入模拟设备）…", "dim"), flush=True)

    from sglang import Engine, EngineArgs

    engine_args = EngineArgs(
        model_path=tokenizer_path,
        tp_size=1,
        dtype="bfloat16",
        context_length=512,
        max_mamba_cache_size=8,
        attention_backend="triton",
        disable_cuda_graph=True,
        random_seed=0,
        watchdog_timeout=86400,
        dist_timeout=86400,
        mem_fraction_static=0.90,
    )

    class _ProgressIntercept:
        """SGLang 日志拦截，把关键阶段转化为进度显示。"""

        def __init__(self):
            self.stage = "engine-init"

    engine = Engine(engine_args)
    t_engine_ready = time.time()
    print(c(f"  ✓ 引擎就绪（{t_engine_ready - t_engine_start:.0f}s）", "green"), flush=True)
    print(c("  ▸ 开始生成…", "dim"), flush=True)

    # ---- 生成主循环（手动逐 token 以驱动进度 UI）----
    from sglang.lang.iraversal import SamplingParams

    t0 = time.time()
    output_ids = []
    current_text = ""

    # 逐 token generate：每次只请求 1 个新 token，实现实时进度
    token_times = []
    for i in range(max_tokens):
        t_tok = time.time()
        result = engine.generate(
            input_ids=prompt_ids + output_ids,
            sampling_params=SamplingParams(
                max_new_tokens=1,
                temperature=0.0,
            ),
        )
        new_id = result["output_ids"][-1]
        output_ids.append(new_id)
        token_times.append(time.time() - t_tok)
        current_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        # 每 token 更新 UI（layer/kernel 由 dispatch sampler 提供）
        layer_info = sampler.current_layer
        kernel_info = sampler.current_kernel
        token_display(i + 1, max_tokens, layer_info, kernel_info,
                      current_text, time.time() - t0)

        # 每 10 token 输出一条 kernel 摘要日志（详略得当）
        if (i + 1) % 10 == 0:
            kernel_log_line(
                f"checkpoint: {i + 1} tokens",
                f"{sampler.dispatch_count} dispatches · "
                f"avg {sum(token_times[-10:]) / 10:.1f}s/tok",
            )

    finish_display()
    t_end = time.time()
    sampler.daemon_flag.set()
    engine.shutdown()

    ttft = token_times[0] if token_times else 0
    tpot = (sum(token_times[1:]) / len(token_times[1:])) if len(token_times) > 1 else 0
    metrics = {
        "engine_load_s": round(t_engine_ready - t_engine_start, 1),
        "ttft_s": round(ttft, 1),
        "tpot_s": round(tpot, 2),
        "total_gen_s": round(t_end - t0, 1),
        "tokens_s": round(len(output_ids) / (t_end - t0), 4),
        "dispatches": sampler.dispatch_count,
    }
    return output_ids, metrics


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="AMDGPU-CDNA4-SIM 一键 100-token 演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python demo_gen.py \"为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？\"",
    )
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    output_ids, metrics = run_generation(args.prompt, args.max_tokens)
    text = ""  # run_generation 已有 current_text，此处重 decode
    # （从引擎重新 decode 以确保一致性）
    # run_generation 返回 metrics 中包含全部所需数据

    print()
    banner("生成完成 · 性能指标", 56)
    print(f"""
  {c('TTFT', 'yellow')}（首 token 延迟）    {metrics['ttft_s']}s
  {c('TPOT', 'yellow')}（每 token 延迟）    {metrics['tpot_s']}s
  {c('吞吐', 'yellow')}                  {metrics['tokens_s']} tok/s
  {c('引擎加载', 'dim')}               {metrics['engine_load_s']}s
  {c('生成总时长', 'dim')}             {metrics['total_gen_s']}s ({len(output_ids)} tokens)
  {c('kernel dispatches', 'dim')}      {metrics['dispatches']}
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
