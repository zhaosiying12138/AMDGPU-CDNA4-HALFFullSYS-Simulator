#!/usr/bin/env python3
"""一键客户演示：SGLang 多 TP 生成 100 个 token（默认 TP2 最快；--tp 4 需手动 --model 指定 9B）。

用法（无需手动 gem5-session——TP2 引擎自己按 rank 拉起两个模拟器）：

    python demo_gen.py "你的 prompt"

流程：本脚本以干净环境 + 私有 run root 启动工作子进程；工作进程按
run_engine_lane 的 TP2 配置（2-GPU CU 一致拓扑、NCCL NET/Socket、
triton attention）构造 SGLang Engine(tp=2)，逐 token 生成并把每个新
token + 计时写进进度文件；UI 进程实时渲染进度（token i/N、各 rank
dispatch 签名、已生成文本滑窗）、每 10 token 输出 kernel 检查点日志
（体现 DTIF fast-copy 权重注入与 Triton GDN 内核路径），结束时打印
完整回答与 TTFT/TPOT/吞吐指标。
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import struct
import subprocess
import sys
import termios
import time

ROOT = "/home/zhaosiying/zcode-lane"
CONDA_PREFIX = (
    f"{ROOT}/env/conda/rocm-pytorch-v3-fa8414cce688f934f538163621423376c2"
    "542acff3e4d3e403df4340d90fcd6d"
)
RUNTIME_BUILD = f"{ROOT}/projects/self-amdgpu-runtime/build/cp28-runtime-clang"
ROCR_LIB = f"{ROOT}/build/rocr-stage-zcode/lib"
TOPOLOGY = f"{ROOT}/artifacts/topology/gpu-{{TP}}"
DEFAULT_MODEL_8B = f"{ROOT}/models/Qwen3.5-0.8B"
DEFAULT_MODEL_9B = f"{ROOT}/models/Qwen3.5-9B"
DEFAULT_PROMPT = "为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？"
STATE_DIR = "/tmp/amdgpu-sim-demo-gen"
PROGRESS_FILE = f"{STATE_DIR}/progress.jsonl"
RUN_ROOT = "/tmp/sagr-lane-zcode-demo-tp{TP}"

WORKER_SOURCE = r'''
import json, os, time

progress_path = os.environ["DEMO_PROGRESS_FILE"]
prompt = os.environ["DEMO_PROMPT"]
max_tokens = int(os.environ["DEMO_MAX_TOKENS"])
model = os.environ["DEMO_MODEL"]
tp = int(os.environ["DEMO_TP"])
ctx = int(os.environ["DEMO_CONTEXT"])
mtt = int(os.environ["DEMO_MAX_TOTAL_TOKENS"])

def emit(rec):
    with open(progress_path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()

emit({"event": "worker_start"})
t0 = time.time()
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
prompt_ids = tokenizer(prompt)["input_ids"]
emit({"event": "prompt_tokenized", "n_prompt": len(prompt_ids)})

from sglang.srt.entrypoints.engine import Engine
engine = Engine(
    model_path=model,
    tp_size=tp,
    dtype="bfloat16",
    attention_backend="triton",
    disable_cuda_graph=True,
    disable_custom_all_reduce=True,
    max_total_tokens=mtt,
    max_running_requests=1,
    max_mamba_cache_size=5,
    random_seed=0,
    watchdog_timeout=86400,
    dist_timeout=86400,
    context_length=ctx,
    chunked_prefill_size=-1,
    skip_tokenizer_init=True,
    log_level="info",
)
emit({"event": "engine_ready", "load_s": round(time.time() - t0, 1)})

output_ids = []
token_times = []
for i in range(max_tokens):
    t = time.time()
    out = engine.generate(
        input_ids=prompt_ids + output_ids,
        sampling_params={"max_new_tokens": 1, "temperature": 0.0},
    )
    new = out["output_ids"][-1]
    dt = time.time() - t
    output_ids.append(new)
    token_times.append(dt)
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    emit({
        "event": "token", "i": i + 1, "dt_s": round(dt, 2),
        "id": new, "text": text,
    })

ttft = token_times[0] if token_times else 0
rest = token_times[1:]
tpot = (sum(rest) / len(rest)) if rest else 0
emit({
    "event": "done",
    "ids": output_ids,
    "ttft_s": round(ttft, 1),
    "tpot_s": round(tpot, 2),
    "load_s": round(time.time() - t0 - sum(token_times), 1),
    "total_s": round(time.time() - t0, 1),
})
engine.shutdown()
'''


def term_width() -> int:
    try:
        res = fcntl.ioctl(0, termios.TIOCGWINSZ, b"\x00" * 8)
        return struct.unpack("hhhh", res)[1] or 80
    except Exception:
        return 80


ANSI = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "green": "\033[32m", "yellow": "\033[33m", "cyan": "\033[36m",
    "magenta": "\033[35m", "red": "\033[31m",
}


def c(text: str, color: str) -> str:
    return f"{ANSI[color]}{text}{ANSI['reset']}"


def banner(title: str, width: int = 62) -> None:
    line = "─" * width
    print(f"{c('┌' + line + '┐', 'cyan')}")
    print(f"{c('│', 'cyan')}{c(title.center(width, ' '), 'bold')}{c('│', 'cyan')}")
    print(f"{c('└' + line + '┘', 'cyan')}", flush=True)


def bar(cur: int, total: int, width: int = 26) -> str:
    filled = int(width * cur / total) if total else 0
    return c("█" * filled, "green") + c("░" * (width - filled), "dim")


def dispatch_summary(run_root: str, tp: int):
    """返回 [(count, sig), ...] 每个已出现的会话一份，按名字排序。"""
    import glob as _glob
    out = []
    paths = sorted(
        _glob.glob(f"{run_root}/self-amdgpu-opencl-run.*/dispatch-trace.jsonl")
    )
    for path in paths[:tp]:
        n = 0
        sig = "—"
        try:
            with open(path) as f:
                for line in f:
                    n += 1
                    last = line
            if n:
                rec = json.loads(last)
                g = rec.get("grid", [0])
                wg = rec.get("workgroup", [0])
                lds = rec.get("fixed_shared_memory_bytes", 0)
                sig = f"g{g[0]} w{wg[0]} L{lds // 1024}K"
        except (OSError, json.JSONDecodeError):
            pass
        out.append((n, sig))
    while len(out) < tp:
        out.append((0, "—"))
    return out[:tp]


def build_worker_env(prompt: str, max_tokens: int, model: str, tp: int) -> dict:
    product = (
        f"{ROOT}/env/rocm/product-v1-4d9d40454031c7345f25da81b6781995b09a3"
        "b10e4dd66026e019306fc7ee39b"
    )
    env = {
        "HOME": os.environ.get("HOME", "/home/zhaosiying"),
        "TERM": "dumb",
        "PATH": f"{CONDA_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONPATH": f"{ROOT}/projects/sglang-0.5.17:{ROOT}/env/sglang-overlay-cp312",
        "LD_LIBRARY_PATH": (
            f"{CONDA_PREFIX}/system-runtime/usr/lib/x86_64-linux-gnu:"
            f"{CONDA_PREFIX}/rocm-sysroot/opt/rocm-7.2.3/lib:"
            f"{CONDA_PREFIX}/rocm-sysroot/opt/rocm-7.2.3/lib/rocm_sysdeps/lib:"
            f"{CONDA_PREFIX}/lib:{RUNTIME_BUILD}:{ROCR_LIB}"
        ),
        "LD_PRELOAD": f"{ROOT}/build/rocr_logging_preload.so:{ROCR_LIB}/libhsa-runtime64.so.1",
        "ROCM_SIM_ROOT": product,
        "HSA_PATH": product,
        "HSA_MODEL_LIB": f"{RUNTIME_BUILD}/libself_amdgpu_hsakmt_model.so.1",
        "HSA_MODEL_TOPOLOGY": TOPOLOGY.format(TP=tp),
        "HSA_ENABLE_DXG_DETECTION": "0",
        "HSA_ENABLE_INTERRUPT": "0",
        "HIP_PLATFORM": "amd",
        "SAGR_MANAGED_RUN_ROOT": RUN_ROOT.format(TP=tp),
        "SAGR_ROCR_LIBRARY_DIR": ROCR_LIB,
        "SGLANG_USE_AITER": "1",
        "FLA_CACHE_RESULTS": "1",
        "NCCL_SHM_DISABLE": "1",
        "AITER_CONFIG_GEMM_BF16": f"{STATE_DIR}/aiter-config/bf16_tuned_gemm.csv",
        "TRITON_CACHE_DIR": f"{STATE_DIR}/triton-cache",
        "XDG_CACHE_HOME": f"{STATE_DIR}/xdg",
        "DEMO_PROGRESS_FILE": PROGRESS_FILE,
        "DEMO_PROMPT": prompt,
        "DEMO_MAX_TOKENS": str(max_tokens),
        "DEMO_MODEL": model,
        "DEMO_TP": str(tp),
        "DEMO_CONTEXT": str(256 if "0.8B" in model else 512),
        "DEMO_MAX_TOTAL_TOKENS": str(64 if "0.8B" in model else 160),
    }
    return env


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AMDGPU-CDNA4-SIM 一键 100-token 演示（SGLang TP2 双 gem5）",
    )
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--tp", type=int, default=2,
                        choices=(1, 2, 4),
                        help="TP 度；4 档需手动 --model 指定 9B（TP4 是 9B 的数学最大并行）")
    parser.add_argument("--model", default=None,
                        help="模型路径；--tp 4（9B 档）必须显式指定，"
                             "tp<=2 默认 Qwen3.5-0.8B")
    args = parser.parse_args()
    tp = args.tp
    if args.model:
        model = args.model
    elif tp >= 4:
        parser.error(
            "--tp 4 需要显式 --model（例如 --model "
            f"{DEFAULT_MODEL_9B}）；不会自动选择 9B"
        )
    else:
        model = DEFAULT_MODEL_8B

    shutil.rmtree(STATE_DIR, ignore_errors=True)
    run_root = RUN_ROOT.format(TP=tp)
    shutil.rmtree(run_root, ignore_errors=True)
    os.makedirs(f"{STATE_DIR}/aiter-config", exist_ok=True)
    os.makedirs(f"{STATE_DIR}/triton-cache", exist_ok=True)
    os.makedirs(f"{STATE_DIR}/xdg", exist_ok=True)
    pkg_csv = (
        f"{CONDA_PREFIX}/lib/python3.12/site-packages/aiter/configs/bf16_tuned_gemm.csv"
    )
    if os.path.exists(pkg_csv):
        shutil.copyfile(pkg_csv, f"{STATE_DIR}/aiter-config/bf16_tuned_gemm.csv")
    open(PROGRESS_FILE, "w").close()

    model_name = "Qwen3.5-9B" if tp >= 4 else "Qwen3.5-0.8B"
    layers = "36 层" if tp >= 4 else "24 层：18 GDN + 6 全注意力"
    banner(f"AMDGPU-CDNA4-SIM · {model_name} · SGLang TP{tp} · {tp}× gem5 实例", 58)
    print(f"  {c('模型', 'dim')}: {model_name}（{layers}）")
    print(f"  {c('Prompt', 'dim')}: {args.prompt[:44]}…")
    print(f"  {c('并行', 'dim')}: TP{tp}（rank 0-{tp - 1} 各持一个模拟 gfx950，NCCL Socket）")
    print(f"  {c('目标', 'dim')}: {args.max_tokens} tokens（贪心解码）")
    print()
    print(c("  ▸ 拉起工作进程：2-GPU CU 一致拓扑 → DTIF fast-copy 权重注入…", "dim"), flush=True)

    worker_path = f"{STATE_DIR}/worker.py"
    with open(worker_path, "w") as f:
        f.write(WORKER_SOURCE)
    env = build_worker_env(args.prompt, args.max_tokens, model, tp)
    worker = subprocess.Popen(
        [f"{CONDA_PREFIX}/bin/python", worker_path],
        env=env,
        stdout=open(f"{STATE_DIR}/worker.out", "w"),
        stderr=subprocess.STDOUT,
    )

    t_start = time.time()
    seen = 0
    last_text = ""
    token_dts = []
    checkpoints_logged = 0
    final = None

    try:
        while True:
            if worker.poll() is not None:
                break
            try:
                with open(PROGRESS_FILE) as f:
                    lines = f.readlines()
            except OSError:
                lines = []
            for line in lines[seen:]:
                seen += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = rec.get("event")
                if ev == "engine_ready":
                    print(
                        c(f"  ✓ {tp} rank 引擎就绪（{rec['load_s']}s）", "green"),
                        flush=True,
                    )
                elif ev == "token":
                    token_dts.append(rec["dt_s"])
                    last_text = rec["text"]
                    summaries = dispatch_summary(run_root, tp)
                    w = term_width()
                    text = last_text.replace("\n", " ")
                    window = max(18, w - 30)
                    if len(text) > window:
                        text = "…" + text[-window:]
                    rank_parts = "  ".join(
                        f"{c(f'r{i}', 'cyan')} {s}({d})"
                        for i, (d, s) in enumerate(summaries)
                    )
                    print(
                        f"\033[G\033[K  {c('token', 'dim')} "
                        f"{c(str(rec['i']), 'bold')}/{c(str(args.max_tokens), 'dim')}"
                        f"  {rank_parts}"
                        f"  {c(f'{time.time() - t_start:.0f}s', 'yellow')}"
                    )
                    print(
                        f"\033[G\033[K  {bar(rec['i'], args.max_tokens)}  {text}\033[F",
                        end="", flush=True,
                    )
                    if rec["i"] // 10 > checkpoints_logged:
                        checkpoints_logged = rec["i"] // 10
                        recent = token_dts[-10:]
                        stamp = time.strftime("%H:%M:%S")
                        total_disp = sum(d for d, _ in summaries)
                        print(
                            f"\033[G\033[K  {c(stamp, 'dim')} {c('●', 'magenta')} "
                            f"checkpoint {rec['i']} tok · {total_disp} dispatches · "
                            f"avg {sum(recent) / len(recent):.1f}s/tok\033[F",
                            end="", flush=True,
                        )
                elif ev == "done":
                    final = rec
            time.sleep(1.0)
    except KeyboardInterrupt:
        worker.kill()
        print(c("\n  ✗ 已中断", "red"))
        return 1

    print("\033[E", end="", flush=True)
    if worker.returncode != 0:
        print(
            c(
                f"  ✗ 工作进程异常退出（code {worker.returncode}）；"
                f"日志: {STATE_DIR}/worker.out",
                "red",
            )
        )
        return 1
    if final is None:
        print(c("  ✗ 未收到完成记录", "red"))
        return 1

    ids = final.get("ids", [])
    print()
    banner("生成结果", 58)
    print()
    for line in last_text.split("\n"):
        print(f"  {line}")
    print()
    banner("性能指标", 58)
    print(f"""
  {c('TTFT', 'yellow')}（首 token 延迟）      {final.get('ttft_s')}s
  {c('TPOT', 'yellow')}（每 token 延迟）      {final.get('tpot_s')}s
  {c('吞吐', 'yellow')}                    {len(ids) / max(final.get('total_s', 1), 1):.4f} tok/s
  {c('引擎加载', 'dim')}                 {final.get('load_s')}s（{tp} rank 权重注入）
  {c('生成总时长', 'dim')}               {final.get('total_s')}s（{len(ids)} tokens）
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
