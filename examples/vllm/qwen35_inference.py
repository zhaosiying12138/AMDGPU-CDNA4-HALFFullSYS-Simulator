#!/usr/bin/env python3
"""Run a minimal Qwen3.5 generation through the unchanged upstream vLLM LLM API.

This is the vLLM peer of ``examples/sglang/qwen35_inference.py``. It uses only
the public upstream entry point: no project operator registration, no
``ModelRegistry`` override, no monkey patch, and no out-of-tree platform
plugin. vLLM must reach its own in-tree ``RocmPlatform`` by ordinary
auto-selection, which requires the simulator-aware AMD SMI provider on the
library path and an environment with no competing NVIDIA device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models" / "Qwen3.5-0.8B",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Upstream fraction of simulated device memory reserved by vLLM.",
    )
    parser.add_argument(
        "--load-format",
        default="auto",
        help="Upstream load format; 'dummy' skips real weight bytes.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=True,
        help="Disable graph capture; the simulator executes each dispatch.",
    )
    parser.add_argument(
        "--skip-tokenizer-init",
        action="store_true",
        help=(
            "Upstream Qwen3.5 carries a multimodal processor whose constructor "
            "dereferences the tokenizer, so skipping tokenizer init aborts "
            "before the model is built. Off by default."
        ),
    )
    args = parser.parse_args()
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()

    # Import after argument parsing so a usage error does not pay vLLM's
    # import cost, and so platform selection failures surface with the run.
    from vllm import LLM, SamplingParams
    from vllm.platforms import current_platform

    print(json.dumps({"selected_platform": type(current_platform).__name__}))

    engine = LLM(
        model=str(args.model_path.resolve()),
        tensor_parallel_size=args.tp_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        load_format=args.load_format,
        skip_tokenizer_init=args.skip_tokenizer_init,
        disable_custom_all_reduce=True,
    )
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    outputs = engine.generate(
        prompts=[{"prompt_token_ids": [248044, 266]}],
        sampling_params=sampling,
    )
    payload = [
        {
            "prompt_token_ids": list(output.prompt_token_ids or []),
            "token_ids": list(output.outputs[0].token_ids),
            "finish_reason": output.outputs[0].finish_reason,
        }
        for output in outputs
    ]
    print(json.dumps(payload, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
