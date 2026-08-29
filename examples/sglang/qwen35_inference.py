#!/usr/bin/env python3
"""Run a minimal Qwen3.5 generation through the upstream SGLang Engine API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from qwen35_token_gate import (  # noqa: E402
    PROMPT_TOKEN_IDS,
    compare_token_ids,
    expected_continuation_token_ids,
)


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--watchdog-timeout", type=float, default=86400.0)
    parser.add_argument("--dist-timeout", type=int, default=86400)
    parser.add_argument("--enable-multimodal", action="store_true")
    parser.add_argument("--initialize-tokenizer", action="store_true")
    parser.add_argument(
        "--load-format",
        default="auto",
        help=(
            "Upstream ServerArgs load format. 'dummy' initializes random "
            "weights, which keeps simulator bring-up iterations short when "
            "the failure under investigation is not numerical."
        ),
    )
    args = parser.parse_args()
    if args.watchdog_timeout <= 0:
        parser.error("--watchdog-timeout must be positive")
    if args.dist_timeout <= 0:
        parser.error("--dist-timeout must be positive")
    expected_tokens = expected_continuation_token_ids(args.model_path)
    if not 1 <= args.max_new_tokens <= len(expected_tokens):
        parser.error(
            "--max-new-tokens must fit the frozen golden continuation "
            f"(1..{len(expected_tokens)})"
        )
    return args


def main() -> int:
    args = parse_args()
    if os.environ.get("SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT"):
        # CPython reports sitecustomize exceptions and then continues.  A
        # numerical-diagnostic lane must instead stop before constructing an
        # engine if its fail-fast observation hooks did not install.
        from qwen35_sglang_layer_gate import assert_installed

        assert_installed()
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
            random_seed=args.seed,
            watchdog_timeout=args.watchdog_timeout,
            dist_timeout=args.dist_timeout,
            context_length=args.context_length,
            chunked_prefill_size=-1,
            enable_multimodal=args.enable_multimodal,
            load_format=args.load_format,
            skip_tokenizer_init=not args.initialize_tokenizer,
            log_level="info",
        )
        output = engine.generate(
            input_ids=list(PROMPT_TOKEN_IDS),
            sampling_params={"max_new_tokens": args.max_new_tokens, "temperature": 0.0},
        )
        if os.environ.get("SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT"):
            from qwen35_sglang_layer_gate import assert_completed

            assert_completed()
        print(json.dumps(output, ensure_ascii=True, default=str))
        actual_ids = output.get("output_ids") if isinstance(output, dict) else None
        token_gate = compare_token_ids(
            actual_ids,
            args.max_new_tokens,
            expected_token_ids=expected_continuation_token_ids(args.model_path),
        )
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
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
