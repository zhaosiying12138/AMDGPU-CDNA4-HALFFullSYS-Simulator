#!/usr/bin/env python3
"""一键演示：SGLang TP2 · Qwen3.5-0.8B · decode 1 token（最快验收配置）。

用法：
    python demo_sglang_tp2.py [prompt] [--max-tokens N]

双 gem5 实例（rank 0/1，NCCL Socket）；环境与验收 lane（zcode-sglang-tp2-formal-v1）一致。
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (MODELS, banner, c, build_env, install_shim,
                     launch_worker, prepare_state, watch,
                     default_token_hook, final_report, tail_worker_log)

TAG = "sglang-tp2"
TP = 2

WORKER = r'''
import json, os, time
progress_path = os.environ["DEMO_PROGRESS_FILE"]
prompt = os.environ["DEMO_PROMPT"]
max_tokens = int(os.environ["DEMO_MAX_TOKENS"])
model = os.environ["DEMO_MODEL"]

def emit(rec):
    with open(progress_path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()

emit({"event": "log", "text": "worker 启动；导入 torch/transformers…"})
t0 = time.time()
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
prompt_ids = tokenizer(prompt)["input_ids"]
emit({"event": "log", "text": f"prompt 已编码：{len(prompt_ids)} tokens"})

from sglang.srt.entrypoints.engine import Engine
from qwen35_text_golden import (
    PROMPT as TEXT_GOLDEN_PROMPT,
    compare_text_token_ids,
)
emit({"event": "log", "text": "构造 Engine(tp=2)……两个 rank 各注入半份权重"})
engine = Engine(
    model_path=model, tp_size=2, dtype="bfloat16",
    attention_backend="triton", disable_cuda_graph=True,
    disable_custom_all_reduce=True, max_total_tokens=64,
    max_running_requests=1, max_mamba_cache_size=5,
    random_seed=0, watchdog_timeout=86400, dist_timeout=86400,
    context_length=256, chunked_prefill_size=-1,
    skip_tokenizer_init=True, log_level="info",
)
emit({"event": "engine_ready", "load_s": round(time.time() - t0, 1)})

output_ids, token_times = [], []
for i in range(max_tokens):
    t = time.time()
    out = engine.generate(
        input_ids=prompt_ids + output_ids,
        sampling_params={"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True},
    )
    output_ids.append(out["output_ids"][-1])
    token_times.append(time.time() - t)
    emit({"event": "token", "i": i + 1, "dt_s": round(token_times[-1], 2),
          "id": output_ids[-1],
          "text": tokenizer.decode(output_ids, skip_special_tokens=True)})

rest = token_times[1:]
if prompt != TEXT_GOLDEN_PROMPT:
    raise ValueError(
        "demo token gate has no independent golden for this prompt; "
        "refusing comparison"
    )
token_gate = compare_text_token_ids(
    output_ids,
    max_tokens,
    model_path=model,
    prompt=prompt,
    prompt_token_ids=prompt_ids,
)
emit({"event": "done", "ids": output_ids,
      "expected_ids": token_gate["expected_token_ids"],
      "token_gate": token_gate["correct"],
      "text_golden": token_gate,
      "ttft_s": round(token_times[0] if token_times else 0, 1),
      "tpot_s": round(sum(rest) / len(rest), 2) if rest else 0,
      "load_s": round(time.time() - t0 - sum(token_times), 1),
      "total_s": round(time.time() - t0, 1)})
engine.shutdown()
'''

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?",
                        default="为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？")
    parser.add_argument("--max-tokens", type=int, default=1)
    args = parser.parse_args()

    state = "/tmp/amdgpu-sim-demo-sglang-tp2"
    prepare_state(state, TAG)
    env = build_env(TAG, TP, args.prompt, args.max_tokens,
                    MODELS["0.8B"], "sglang", state)
    install_shim(state, env)

    banner("AMDGPU-CDNA4-SIM · Qwen3.5-0.8B · SGLang TP2 · 双 gem5 实例", 58)
    print(f"  {c('模型', 'dim')}: Qwen3.5-0.8B（24 层：18 GDN + 6 全注意力）")
    print(f"  {c('Prompt', 'dim')}: {args.prompt[:44]}…")
    print(f"  {c('并行', 'dim')}: TP2（rank 0/1 各持一个模拟 gfx950，NCCL Socket）")
    print(f"  {c('目标', 'dim')}: {args.max_tokens} token（贪心）")
    print(c(f"  ▸ 拉起工作进程：{TP}-GPU CU 一致拓扑 → DTIF fast-copy 权重注入…", "dim"), flush=True)

    worker = launch_worker(state, env, WORKER)
    final, text = watch(worker, state, env["SAGR_MANAGED_RUN_ROOT"], TP,
                        args.max_tokens, "sglang")
    if worker.returncode != 0:
        print(c(f"  ✗ 工作进程异常退出（code {worker.returncode}）", "red"))
        print(c(f"    日志尾部：{tail_worker_log(f'{state}/worker.out', 5)}", "dim"))
        return 1
    if final is None:
        print(c("  ✗ 未收到完成记录", "red")); return 1
    final_report(final, text, "TP2", TP, "Qwen3.5-0.8B", state)
    return 0 if final.get("token_gate") is True and len(final.get("ids", [])) == args.max_tokens else 1

if __name__ == "__main__":
    raise SystemExit(main())
