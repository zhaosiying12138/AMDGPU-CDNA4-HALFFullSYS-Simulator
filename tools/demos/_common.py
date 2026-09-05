"""共享支撑：demo_* 一键脚本的引擎环境组装、进度渲染与汇总。"""
from __future__ import annotations

import fcntl
import glob as _glob
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
PRODUCT = (
    f"{ROOT}/env/rocm/product-v1-4d9d40454031c7345f25da81b6781995b09a3"
    "b10e4dd66026e019306fc7ee39b"
)
MODELS = {
    "0.8B": f"{ROOT}/models/Qwen3.5-0.8B",
    "9B": f"{ROOT}/models/Qwen3.5-9B",
}

ANSI = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "green": "\033[32m", "yellow": "\033[33m", "cyan": "\033[36m",
    "magenta": "\033[35m", "red": "\033[31m",
}


def c(text: str, color: str) -> str:
    return f"{ANSI[color]}{text}{ANSI['reset']}"


def term_width() -> int:
    try:
        res = fcntl.ioctl(0, termios.TIOCGWINSZ, b"\x00" * 8)
        return struct.unpack("hhhh", res)[1] or 80
    except Exception:
        return 80


def banner(title: str, width: int = 62) -> None:
    line = "─" * width
    print(f"{c('┌' + line + '┐', 'cyan')}")
    print(f"{c('│', 'cyan')}{c(title.center(width, ' '), 'bold')}{c('│', 'cyan')}")
    print(f"{c('└' + line + '┘', 'cyan')}", flush=True)


def bar(cur: int, total: int, width: int = 26) -> str:
    filled = int(width * cur / total) if total else 0
    return c("█" * filled, "green") + c("░" * (width - filled), "dim")


def dispatch_summary(run_root: str, tp: int):
    out = []
    for path in sorted(
        _glob.glob(f"{run_root}/self-amdgpu-opencl-run.*/dispatch-trace.jsonl")
    )[:tp]:
        n, sig = 0, "—"
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


def tail_worker_log(path: str, n: int = 1) -> str:
    """返回 worker 日志的最后 n 行（剥离 rocr/blit 噪声）。"""
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError:
        return ""
    keep = [
        ln for ln in lines
        if "amd_blit" not in ln and "rocr logging preload" not in ln
    ]
    return "\n".join(keep[-n:])


def build_env(tag: str, tp: int, prompt: str, max_tokens: int,
               model: str, engine: str, state_dir: str) -> dict:
    """按验收 lane 的环境组装 worker env（sglang 与 vllm 各自的形态）。"""
    topology = f"{ROOT}/artifacts/topology/gpu-{tp}"
    run_root = f"/tmp/sagr-lane-zcode-demo-{tag}"
    progress = f"{state_dir}/progress.jsonl"
    env = {
        "HOME": os.environ.get("HOME", "/home/zhaosiying"),
        "TERM": "dumb",
        "PATH": f"{CONDA_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": (
            f"{CONDA_PREFIX}/system-runtime/usr/lib/x86_64-linux-gnu:"
            f"{CONDA_PREFIX}/rocm-sysroot/opt/rocm-7.2.3/lib:"
            f"{CONDA_PREFIX}/rocm-sysroot/opt/rocm-7.2.3/lib/rocm_sysdeps/lib:"
            f"{CONDA_PREFIX}/lib:{RUNTIME_BUILD}:{ROCR_LIB}"
        ),
        "LD_PRELOAD": f"{ROOT}/build/rocr_logging_preload.so:{ROCR_LIB}/libhsa-runtime64.so.1",
        "ROCM_SIM_ROOT": PRODUCT,
        "HSA_PATH": PRODUCT,
        "HSA_MODEL_LIB": f"{RUNTIME_BUILD}/libself_amdgpu_hsakmt_model.so.1",
        "HSA_MODEL_TOPOLOGY": topology,
        "HSA_ENABLE_DXG_DETECTION": "0",
        "HSA_ENABLE_INTERRUPT": "0",
        "HIP_PLATFORM": "amd",
        "ROCM_PATH": f"{CONDA_PREFIX}/rocm-sysroot/opt/rocm-7.2.3",
        "HIP_PATH": f"{CONDA_PREFIX}/rocm-sysroot/opt/rocm-7.2.3",
        "HIP_CLANG_PATH": f"{CONDA_PREFIX}/rocm-sysroot/opt/rocm-7.2.3/lib/llvm/bin",
        "PYTORCH_ROCM_ARCH": "gfx950",
        "GPU_ARCHS": "gfx950",
        "SAGR_MANAGED_RUN_ROOT": run_root,
        "SAGR_ROCR_LIBRARY_DIR": ROCR_LIB,
        "NCCL_SHM_DISABLE": "1",
        "NCCL_SOCKET_IFNAME": "lo",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "86400",
        # ROCr refuses multi-agent collectives without this (the lane runner
        # exports it too; missing it aborts RCCL init outright).
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "AITER_CONFIG_GEMM_BF16": f"{state_dir}/aiter-config/bf16_tuned_gemm.csv",
        "TRITON_CACHE_DIR": f"{state_dir}/triton-cache",
        "XDG_CACHE_HOME": f"{state_dir}/xdg",
        "DEMO_PROGRESS_FILE": progress,
        "DEMO_PROMPT": prompt,
        "DEMO_MAX_TOKENS": str(max_tokens),
        "DEMO_MODEL": model,
        "DEMO_TP": str(tp),
        "DEMO_ENGINE": engine,
    }
    if engine == "sglang":
        env["PYTHONPATH"] = (
            f"{ROOT}/tools:{ROOT}/projects/sglang-0.5.17:{ROOT}/env/sglang-overlay-cp312"
        )
        env["SGLANG_USE_AITER"] = "1"
        env["FLA_CACHE_RESULTS"] = "1"
        env["SAGR_ATTENTION_BACKEND"] = "triton"
    else:
        env["PYTHONPATH"] = f"{ROOT}/env/sglang-overlay-cp312"
        env["SAGR_TRITON_FAST_AUTOTUNE"] = "1"
        env["SAGR_VLLM_RPC_TIMEOUT_SECONDS"] = "86400"
        env["SAGR_VLLM_DIST_TIMEOUT_SECONDS"] = "86400"
        env["SAGR_VLLM_GPU_MEM_UTIL"] = "0.015"
        env["TRITON_BACKENDS_IN_TREE"] = "1"
        env["TRITON_DEFAULT_BACKEND"] = "amd"
        env["VLLM_PLUGINS"] = ""
    return env


def install_shim(state_dir: str, env: dict) -> None:
    shim = f"{state_dir}/tool-shim"
    os.makedirs(shim, exist_ok=True)
    for target, name in (
        (f"{RUNTIME_BUILD}/sagr-rocminfo", "rocminfo"),
        (f"{ROOT}/tools/sim_amdgpu_arch.sh", "amdgpu-arch"),
    ):
        link = f"{shim}/{name}"
        try:
            os.symlink(target, link)
        except FileExistsError:
            pass
    env["PATH"] = f"{shim}:{env['PATH']}"
    env["SAGR_SIM_ROCMINFO"] = f"{shim}/rocminfo"


def prepare_state(state_dir: str, run_root_tag: str) -> None:
    shutil.rmtree(state_dir, ignore_errors=True)
    shutil.rmtree(f"/tmp/sagr-lane-zcode-demo-{run_root_tag}", ignore_errors=True)
    for sub in ("aiter-config", "triton-cache", "xdg"):
        os.makedirs(f"{state_dir}/{sub}", exist_ok=True)
    pkg_csv = (
        f"{CONDA_PREFIX}/lib/python3.12/site-packages/aiter/configs/bf16_tuned_gemm.csv"
    )
    if os.path.exists(pkg_csv):
        shutil.copyfile(pkg_csv, f"{state_dir}/aiter-config/bf16_tuned_gemm.csv")


def launch_worker(state_dir: str, env: dict, worker_source: str):
    """Write the worker spawn-safe and launch it under the lane's NVML mask.

    Two conditions the lane runner (scripts/run_engine_lane.sh) provides and
    a bare Popen does not:
      - sglang's Engine forces multiprocessing "spawn", whose children re-run
        the entry module during bootstrap; wrapping the body in a guarded
        main() keeps it from re-executing there (the idiom of the lane
        examples and of any spawn-safe __main__).
      - This WSL host exposes a real NVIDIA GPU through NVML; without the
        bind-mount mask vLLM's platform resolver sees cuda *and* rocm and
        refuses to activate either platform.
    """
    indented = "\n".join(
        ("    " + line if line.strip() else line)
        for line in worker_source.splitlines()
    )
    worker_path = f"{state_dir}/worker.py"
    with open(worker_path, "w") as f:
        f.write(
            "def _worker_main():\n" + indented + "\n\n\n"
            'if __name__ == "__main__":\n    _worker_main()\n'
        )
    open("/tmp/empty-nvml.so", "w").close()
    return subprocess.Popen(
        ["unshare", "-r", "-m", "bash", "-c",
         'mount --bind /tmp/empty-nvml.so /usr/lib/wsl/lib/libnvidia-ml.so.1; '
         'exec "$0" "$@"',
         f"{CONDA_PREFIX}/bin/python", worker_path],
        env=env,
        stdout=open(f"{state_dir}/worker.out", "w"),
        stderr=subprocess.STDOUT,
    )


def watch(worker, state_dir: str, run_root: str, tp: int, max_tokens: int,
          engine: str, token_hook=None):
    """通用进度循环：消费 progress.jsonl，返回 final 记录（或 None）。"""
    progress = f"{state_dir}/progress.jsonl"
    t_start = time.time()
    seen, final = 0, None
    last_text = ""
    while True:
        if worker.poll() is not None and seen >= _line_count(progress):
            break
        try:
            with open(progress) as f:
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
                print(c(f"  ✓ {tp} rank 引擎就绪（{rec['load_s']}s）", "green"), flush=True)
                print(c("  ▸ 生成中…", "dim"), flush=True)
            elif ev == "log":
                # worker 侧自由日志（权重加载阶段等）
                print(f"    {c(rec.get('text', ''), 'dim')}", flush=True)
            elif ev == "token":
                last_text = rec.get("text", last_text)
                if token_hook:
                    token_hook(rec, tp, run_root, max_tokens, t_start, last_text)
            elif ev == "done":
                final = rec
        if worker.poll() is not None and not _pending(progress, seen):
            break
        time.sleep(2.0)
    return final, last_text


def _line_count(path: str) -> int:
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _pending(path: str, seen: int) -> bool:
    return _line_count(path) > seen


def default_token_hook(rec, tp, run_root, max_tokens, t_start, text):
    summaries = dispatch_summary(run_root, tp)
    rank_parts = "  ".join(f"{c(f'r{i}', 'cyan')} {s}({d})"
                            for i, (d, s) in enumerate(summaries))
    w = term_width()
    show = text.replace("\n", " ")
    window = max(18, w - 34)
    if len(show) > window:
        show = "…" + show[-window:]
    print(
        f"\033[G\033[K  {c('token', 'dim')} "
        f"{c(str(rec['i']), 'bold')}/{c(str(max_tokens), 'dim')}"
        f"  {rank_parts}  {c(f'{time.time() - t_start:.0f}s', 'yellow')}"
    )
    print(f"\033[G\033[K  {bar(rec['i'], max_tokens)}  {show}\033[F", end="", flush=True)


def final_report(final: dict, last_text: str, engine: str, tp: int,
                 model_name: str, state_dir: str) -> None:
    ids = final.get("ids", [])
    print("\033[E", end="", flush=True)
    print()
    banner("生成结果", 58)
    print()
    for line in last_text.split("\n"):
        print(f"  {line}")
    print()
    banner("汇总", 58)
    total = final.get("total_s", 1) or 1
    print(f"""
  {c('引擎', 'dim')}        SGLang {engine} · TP{tp} · {model_name}
  {c('TTFT', 'yellow')}（首 token 延迟）      {final.get('ttft_s')}s
  {c('TPOT', 'yellow')}（每 token 延迟）      {final.get('tpot_s')}s
  {c('吞吐', 'yellow')}                    {len(ids) / max(total, 1):.4f} tok/s
  {c('引擎加载', 'dim')}                 {final.get('load_s')}s（{tp} rank 权重注入）
  {c('生成总时长', 'dim')}               {total}s（{len(ids)} tokens）
  {c('token ids', 'dim')}               {ids}
  {c('text golden', 'dim')}             {('PASS' if final.get('token_gate') is True else 'FAIL')}
  {c('日志', 'dim')}                   {state_dir}/worker.out
""")
