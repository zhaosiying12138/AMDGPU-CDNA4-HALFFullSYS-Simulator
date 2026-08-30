#!/usr/bin/env python3
"""Pure-host tests for the user-facing Qwen3.5 text golden."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen35_text_golden", ROOT / "tools" / "qwen35_text_golden.py"
)
assert SPEC is not None and SPEC.loader is not None
golden = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(golden)


class Qwen35TextGoldenTests(unittest.TestCase):
    def test_prompt_is_real_text_not_the_legacy_token_fixture(self) -> None:
        self.assertNotEqual(golden.PROMPT_TOKEN_IDS, (248044, 266))
        self.assertEqual(len(golden.PROMPT_TOKEN_IDS), 19)

    def test_checkpoint_specific_reference_sequences_are_exact(self) -> None:
        self.assertEqual(
            golden.expected_text_continuation_token_ids(
                "/models/Qwen3.5-0.8B", golden.PROMPT
            ),
            (271, 248068, 271, 248069, 271, 103426, 108169, 95967, 236, 124094),
        )
        self.assertEqual(
            golden.expected_text_continuation_token_ids(
                "/models/Qwen3.5-9B", golden.PROMPT
            ),
            (271, 109455, 332, 116752, 221794, 109311, 3709, 332, 26076, 96212),
        )

    def test_exact_text_sequences_pass_and_wrong_sequence_reports_mismatch(self) -> None:
        expected = golden.expected_text_continuation_token_ids(
            "/models/Qwen3.5-9B", golden.PROMPT
        )
        result = golden.compare_text_token_ids(
            expected,
            10,
            model_path="/models/Qwen3.5-9B",
            prompt=golden.PROMPT,
            prompt_token_ids=golden.PROMPT_TOKEN_IDS,
        )
        self.assertTrue(result["correct"])
        wrong = golden.compare_text_token_ids(
            [279] + list(expected[1:]),
            10,
            model_path="/models/Qwen3.5-9B",
            prompt=golden.PROMPT,
            prompt_token_ids=golden.PROMPT_TOKEN_IDS,
        )
        self.assertFalse(wrong["correct"])
        self.assertEqual(wrong["first_mismatch"], {"index": 0, "expected": 271, "actual": 279})

    def test_unknown_prompt_and_wrong_tokenization_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            golden.expected_text_continuation_token_ids(
                "/models/Qwen3.5-9B", "a different prompt"
            )
        with self.assertRaises(ValueError):
            golden.compare_text_token_ids(
                golden.MODEL_CONTINUATIONS["Qwen3.5-9B"],
                10,
                model_path="/models/Qwen3.5-9B",
                prompt=golden.PROMPT,
                prompt_token_ids=(248044, 266),
            )

    def test_engine_and_runner_expose_text_prompt_mode(self) -> None:
        for relative in (
            "examples/sglang/qwen35_inference.py",
            "examples/vllm/qwen35_inference.py",
            "scripts/run_engine_lane.sh",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("--prompt", source)
        for relative in (
            "tools/demos/demo_sglang_tp2.py",
            "tools/demos/demo_sglang_tp4.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("compare_text_token_ids", source)
            self.assertNotIn("from qwen35_token_gate import expected_continuation_token_ids", source)

    def test_compute_unit_wrapper_preserves_selected_gem5(self) -> None:
        source = (ROOT / "scripts" / "run_engine_lane.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('gem5_base="$SAGR_MANAGED_GEM5"', source)
        self.assertIn('export SAGR_MANAGED_GEM5="$gem5"', source)


if __name__ == "__main__":
    unittest.main()
