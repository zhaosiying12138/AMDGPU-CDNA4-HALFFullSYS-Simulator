#!/usr/bin/env python3
"""Fail-closed token golden gate for the pinned Qwen3.5 trajectory."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


SCHEMA = "amdgpu-sim.qwen35-token-golden-gate.v1"
PROMPT_TOKEN_IDS = (248044, 266)
EXPECTED_CONTINUATION_TOKEN_IDS = (27841, 27841)
# The original frozen continuation belongs to the 0.8B checkpoint.  The 9B
# checkpoint has a different greedy first token for the same prompt; this was
# verified with a CPU reference forward over the same safetensors weights.
MODEL_CONTINUATION_TOKEN_IDS = {
    "Qwen3.5-9B": (248044,),
}


def expected_continuation_token_ids(model_path: object) -> tuple[int, ...]:
    """Return the checkpoint-specific greedy continuation oracle."""

    model_name = Path(str(model_path)).name
    return MODEL_CONTINUATION_TOKEN_IDS.get(
        model_name, EXPECTED_CONTINUATION_TOKEN_IDS
    )


def _normalize_actual(value: object) -> tuple[list[int] | None, str | None]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return None, "actual token IDs are not a sequence"
    result: list[int] = []
    for index, token in enumerate(value):
        if isinstance(token, bool) or not isinstance(token, int):
            return None, f"actual token ID at index {index} is not an integer"
        result.append(token)
    return result, None


def compare_token_ids(
    actual: object,
    max_new_tokens: int,
    *,
    expected_token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compare an engine continuation to the exact pinned greedy trajectory."""

    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be an integer")
    expected_source = tuple(
        EXPECTED_CONTINUATION_TOKEN_IDS
        if expected_token_ids is None
        else expected_token_ids
    )
    if not 1 <= max_new_tokens <= len(expected_source):
        raise ValueError(
            "max_new_tokens exceeds the frozen golden continuation: "
            f"requested={max_new_tokens} available={len(expected_source)}"
        )

    expected = list(expected_source[:max_new_tokens])
    observed, error = _normalize_actual(actual)
    first_mismatch: dict[str, int | None] | None = None
    if observed is not None:
        common = min(len(observed), len(expected))
        mismatch_index = next(
            (index for index in range(common) if observed[index] != expected[index]),
            None,
        )
        if mismatch_index is None and len(observed) != len(expected):
            mismatch_index = common
        if mismatch_index is not None:
            first_mismatch = {
                "index": mismatch_index,
                "expected": (
                    expected[mismatch_index]
                    if mismatch_index < len(expected)
                    else None
                ),
                "actual": (
                    observed[mismatch_index]
                    if mismatch_index < len(observed)
                    else None
                ),
            }

    correct = error is None and observed == expected
    return {
        "schema": SCHEMA,
        "prompt_token_ids": list(PROMPT_TOKEN_IDS),
        "expected_token_ids": expected,
        "actual_token_ids": observed,
        "correct": correct,
        "first_mismatch": first_mismatch,
        "error": error,
    }
