#!/usr/bin/env python3
"""Generate one token with unchanged upstream vLLM on the simulated GPU.

Mirrors examples/sglang/qwen35_inference.py so the two engines are driven the
same way and their results are comparable. Nothing here registers a project
operator, replaces a model or a layer, or patches vLLM: the only adaptation is
the self-runtime beneath it, which is the thing under test.

This has to be a real file rather than a heredoc piped to python. With
tensor_parallel_size > 1 vLLM spawns worker processes through multiprocessing,
and spawn re-imports the parent's __main__ from its recorded path. A program
fed on stdin records that path as "<stdin>", so every worker dies with

    FileNotFoundError: [Errno 2] No such file or directory: '<stdin>'

before it reaches any GPU code. The __main__ guard below is required for the
same reason.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from qwen35_token_gate import (  # noqa: E402
    EXPECTED_CONTINUATION_TOKEN_IDS,
    PROMPT_TOKEN_IDS,
    compare_token_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=Path, default=REPO_ROOT / "models" / "Qwen3.5-0.8B"
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.30,
        help="Kept small: the simulated device reports a large pool and vLLM "
        "would otherwise size a KV cache that takes far longer to touch than "
        "the run needs.",
    )
    parser.add_argument(
        "--load-format",
        default="auto",
        help="Upstream LoadFormat. 'dummy' initialises random weights. Note "
        "that dummy is not automatically faster here: the random "
        "initialisation is itself a large Philox kernel, and it once "
        "accounted for ~70%% of a run's simulated execution.",
    )
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= len(EXPECTED_CONTINUATION_TOKEN_IDS):
        parser.error(
            "--max-new-tokens must fit the frozen golden continuation "
            f"(1..{len(EXPECTED_CONTINUATION_TOKEN_IDS)})"
        )
    return args


def main() -> int:
    args = parse_args()
    # Imported inside main so that a spawn-imported worker does not pay for a
    # second engine construction at import time.
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model = str(args.model_path)
    llm = LLM(
        model=model,
        tokenizer=model,
        # No tokenizer and no chat template: the frozen token trajectory is
        # supplied directly and checked exactly below.
        skip_tokenizer_init=True,
        tensor_parallel_size=args.tp_size,
        dtype="bfloat16",
        max_model_len=args.context_length,
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
        max_num_batched_tokens=args.context_length,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # models/Qwen3.5-0.8B declares Qwen3_5ForConditionalGeneration, which
        # upstream maps to the multimodal class. A zero budget keeps the vision
        # tower out of the decode path.
        limit_mm_per_prompt={"image": 0, "video": 0},
        load_format=args.load_format,
    )

    # Loaded-library proof, written after weight load so it reflects the run
    # rather than what the launcher intended.
    with open("/proc/self/maps", "r", encoding="ascii") as handle:
        loaded = sorted(
            {
                field
                for line in handle
                for field in [line.split()[-1]]
                if "libamdhip64" in field
                or "libhsa-runtime64" in field
                or "libself_amdgpu_runtime" in field
            }
        )
    for path in loaded:
        print("loaded_library=" + path, flush=True)

    outputs = llm.generate(
        TokensPrompt(prompt_token_ids=list(PROMPT_TOKEN_IDS)),
        SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            seed=args.seed,
        ),
    )
    actual_ids = list(outputs[0].outputs[0].token_ids)
    print("generated_token_ids=" + repr(actual_ids), flush=True)
    token_gate = compare_token_ids(actual_ids, args.max_new_tokens)
    token_gate["checkpoint_weights"] = args.load_format != "dummy"
    if not token_gate["checkpoint_weights"]:
        token_gate["correct"] = False
        token_gate["error"] = "dummy weights cannot pass the token golden gate"
    print(
        "token_golden="
        + json.dumps(token_gate, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0 if token_gate["correct"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
