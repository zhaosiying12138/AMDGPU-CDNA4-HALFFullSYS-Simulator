#!/usr/bin/env python3
"""Run a minimal Qwen3.5 generation through the upstream SGLang Engine API."""

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
    parser.add_argument("--attention-backend", default="aiter")
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--max-total-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--max-mamba-cache-size", type=int, default=5)
    parser.add_argument("--watchdog-timeout", type=float, default=86400.0)
    parser.add_argument("--enable-multimodal", action="store_true")
    parser.add_argument("--initialize-tokenizer", action="store_true")
    args = parser.parse_args()
    if args.watchdog_timeout <= 0:
        parser.error("--watchdog-timeout must be positive")
    return args


def main() -> None:
    args = parse_args()
    from sglang.srt.entrypoints.engine import Engine

    engine = None
    try:
        engine = Engine(
            model_path=str(args.model_path.resolve()),
            tp_size=args.tp_size,
            dtype="bfloat16",
            attention_backend=args.attention_backend,
            disable_cuda_graph=True,
            disable_custom_all_reduce=True,
            max_total_tokens=args.max_total_tokens,
            max_running_requests=1,
            max_mamba_cache_size=args.max_mamba_cache_size,
            watchdog_timeout=args.watchdog_timeout,
            context_length=args.context_length,
            chunked_prefill_size=-1,
            enable_multimodal=args.enable_multimodal,
            skip_tokenizer_init=not args.initialize_tokenizer,
            log_level="info",
        )
        output = engine.generate(
            input_ids=[248044, 266],
            sampling_params={"max_new_tokens": args.max_new_tokens, "temperature": 0.0},
        )
        print(json.dumps(output, ensure_ascii=True, default=str))
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
