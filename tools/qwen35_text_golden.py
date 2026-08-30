#!/usr/bin/env python3
"""Independent text-prompt golden for the Qwen3.5 acceptance prompt.

The older engine gate intentionally used a two-token synthetic input
``[248044, 266]``.  That is useful for layer capsules, but it decodes to
``<|endoftext|>at`` and is not a customer prompt.  This module binds the
default demo prompt to tokenizer IDs and to CPU-reference greedy outputs for
both checkpoints.  Unknown prompts are rejected instead of silently accepting
an oracle that was generated for a different input.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


SCHEMA = "amdgpu-sim.qwen35-text-golden.v1"
PROMPT = "为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？"
PROMPT_TOKEN_IDS = (
    144277,
    103426,
    108169,
    95967,
    236,
    124094,
    26076,
    96212,
    103182,
    108076,
    96799,
    24273,
    95761,
    104224,
    109276,
    95726,
    111104,
    115110,
    10992,
)

# Generated independently with the pinned Transformers checkpoint on CPU,
# greedy decoding, no sampling, and no chat-template wrapping.  The model
# names are checkpoint directory basenames, not model-family heuristics.
MODEL_CONTINUATIONS = {
    "Qwen3.5-0.8B": (
        271,
        248068,
        271,
        248069,
        271,
        103426,
        108169,
        95967,
        236,
        124094,
    ),
    "Qwen3.5-9B": (
        271,
        109455,
        332,
        116752,
        221794,
        109311,
        3709,
        332,
        26076,
        96212,
    ),
}


def _model_name(model_path: object) -> str:
    return Path(str(model_path)).name


def expected_text_continuation_token_ids(
    model_path: object, prompt: str
) -> tuple[int, ...]:
    """Return the frozen text golden, rejecting a different prompt."""

    if prompt != PROMPT:
        raise ValueError(
            "no frozen text golden for this prompt; generate an independent "
            "reference before running a strict comparison"
        )
    try:
        return MODEL_CONTINUATIONS[_model_name(model_path)]
    except KeyError as exc:
        raise ValueError(
            f"no frozen text golden for checkpoint {_model_name(model_path)!r}"
        ) from exc


def compare_text_token_ids(
    actual: object,
    max_new_tokens: int,
    *,
    model_path: object,
    prompt: str,
    prompt_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Compare generated IDs and bind the result to the exact text input."""

    expected_source = expected_text_continuation_token_ids(model_path, prompt)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be an integer")
    if not 1 <= max_new_tokens <= len(expected_source):
        raise ValueError(
            "max_new_tokens exceeds the frozen text golden: "
            f"requested={max_new_tokens} available={len(expected_source)}"
        )
    if tuple(prompt_token_ids) != PROMPT_TOKEN_IDS:
        raise ValueError(
            "tokenizer output does not match the frozen prompt tokenization"
        )
    expected = list(expected_source[:max_new_tokens])
    observed: list[int] | None = None
    error: str | None = None
    if isinstance(actual, (str, bytes, bytearray)) or not isinstance(actual, Sequence):
        error = "actual token IDs are not a sequence"
    else:
        observed = []
        for index, token in enumerate(actual):
            if isinstance(token, bool) or not isinstance(token, int):
                error = f"actual token ID at index {index} is not an integer"
                observed = None
                break
            observed.append(token)
    first_mismatch = None
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
                "expected": expected[mismatch_index]
                if mismatch_index < len(expected)
                else None,
                "actual": observed[mismatch_index]
                if mismatch_index < len(observed)
                else None,
            }
    return {
        "schema": SCHEMA,
        "prompt": prompt,
        "prompt_token_ids": list(prompt_token_ids),
        "expected_token_ids": expected,
        "actual_token_ids": observed,
        "correct": error is None and observed == expected,
        "first_mismatch": first_mismatch,
        "error": error,
    }


def expected_text(model_path: object, max_new_tokens: int) -> str:
    """Return the decoded reference continuation for reporting."""

    # Import lazily so pure-host gate tests do not need the tokenizer runtime.
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        str(Path(str(model_path)) / "tokenizer.json")
    )
    ids = expected_text_continuation_token_ids(model_path, PROMPT)
    return tokenizer.decode(list(ids[:max_new_tokens]), skip_special_tokens=False)
