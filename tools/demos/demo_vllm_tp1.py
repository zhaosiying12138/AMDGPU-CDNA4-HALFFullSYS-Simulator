#!/usr/bin/env python3
"""一键演示：vLLM TP1 · Qwen3.5-0.8B · decode 1 token（默认）。

用法：
    python demo_vllm_tp1.py [prompt] [--max-tokens N]

单 gem5 实例；环境与验收 lane（zcode-vllm-tp1-formal-v1）一致：
TRITON_BACKENDS_IN_TREE + amd 后端、fast deterministic autotune、
小缓存上限。
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (MODELS, banner, c, build_env, install_shim,
                     launch_worker, prepare_state, watch,
                     default_token_hook, final_report, tail_worker_log)

TAG = "vllm-tp1"
TP = 1

WORKER = r'''
import json, os, time
progress_path = os.environ["DEMO_PROGRESS_FILE"]
prompt = os.environ["DEMO_PROMPT"]
max_tokens = int(os.environ["DEMO_MAX_TOKENS"])
model = os.environ["DEMO_MODEL"]

def emit(rec):
    with open(progress_path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()

emit({"event": "log", "text": "worker 启动；导入 vllm……"})
t0 = time.time()
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
prompt_ids = tokenizer(prompt)["input_ids"]
emit({"event": "log", "text": f"prompt 已编码：{len(prompt_ids)} tokens"})

emit({"event": "log", "text": "构造 LLM(tp=1)……enforce_eager + FLA warmup autotune"})
llm = LLM(
    model=model, tokenizer=model, skip_tokenizer_init=True,
    tensor_parallel_size=1, dtype="bfloat16",
    max_model_len=256, max_num_seqs=1, seed=0,
    max_num_batched_tokens=256, enforce_eager=True,
    disable_custom_all_reduce=True, gpu_memory_utilization=0.30,
    limit_mm_per_prompt={"image": 0, "video": 0},
)
emit({"event": "engine_ready", "load_s": round(time.time() - t0, 1)})

output_ids, token_times = [], []
for i in range(max_tokens):
    t = time.time()
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids + output_ids)],
        SamplingParams(max_tokens=1, temperature=0.0, seed=0),
    )
    output_ids.append(outs[0].outputs[0].token_ids[-1])
    token_times.append(time.time() - t)
    emit({"event": "token", "i": i + 1, "dt_s": round(token_times[-1], 2),
          "id": output_ids[-1],
          "text": tokenizer.decode(output_ids, skip_special_tokens=True)})

rest = token_times[1:]
emit({"event": "done", "ids": output_ids,
      "ttft_s": round(token_times[0] if token_times else 0, 1),
      "tpot_s": round(sum(rest) / len(rest), 2) if rest else 0,
      "load_s": round(time.time() - t0 - sum(token_times), 1),
      "total_s": round(time.time() - t0, 1)})
'''

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?",
                        default="为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？")
    parser.add_argument("--max-tokens", type=int, default=1)
    args = parser.parse_args()

    state = "/tmp/amdgpu-sim-demo-vllm-tp1"
    prepare_state(state, TAG)
    env = build_env(TAG, TP, args.prompt, args.max_tokens,
                    MODELS["0.8B"], "vllm", state)
    install_shim(state, env)

    banner("AMDGPU-CDNA4-SIM · Qwen3.5-0.8B · vLLM TP1 · 单 gem5 实例", 58)
    print(f"  {c('模型', 'dim')}: Qwen3.5-0.8B（24 层：18 GDN + 6 全注意力）")
    print(f"  {c('Prompt', 'dim')}: {args.prompt[:44]}…")
    print(f"  {c('目标', 'dim')}: {args.max_tokens} token（贪心）")
    print(c(f"  ▸ 拉起工作进程：{TP}-GPU CU 一致拓扑 → DTIF fast-copy 权重注入…", "dim"), flush=True)

    worker = launch_worker(state, env, WORKER)
    final, text = watch(worker, state, env["SAGR_MANAGED_RUN_ROOT"], TP,
                        args.max_tokens, "vllm")
    if worker.returncode != 0:
        print(c(f"  ✗ 工作进程异常退出（code {worker.returncode}）", "red"))
        print(c(f"    日志尾部：{tail_worker_log(f'{state}/worker.out', 5)}", "dim"))
        return 1
    if final is None:
        print(c("  ✗ 未收到完成记录", "red")); return 1
    final_report(final, text, "TP1", TP, "Qwen3.5-0.8B", state)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
