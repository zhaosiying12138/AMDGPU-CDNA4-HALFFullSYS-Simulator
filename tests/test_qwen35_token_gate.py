#!/usr/bin/env python3
"""Fail-closed tests for the engine-level Qwen3.5 token gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen35_token_gate", ROOT / "tools" / "qwen35_token_gate.py"
)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class Qwen35TokenGateTests(unittest.TestCase):
    def test_exact_frozen_continuation_passes(self) -> None:
        one = gate.compare_token_ids([27841], 1)
        two = gate.compare_token_ids((27841, 27841), 2)
        self.assertTrue(one["correct"])
        self.assertTrue(two["correct"])
        self.assertEqual(one["prompt_token_ids"], [248044, 266])
        self.assertEqual(two["expected_token_ids"], [27841, 27841])

    def test_wrong_token_reports_first_mismatch(self) -> None:
        result = gate.compare_token_ids([279], 1)
        self.assertFalse(result["correct"])
        self.assertEqual(
            result["first_mismatch"],
            {"index": 0, "expected": 27841, "actual": 279},
        )

    def test_9b_uses_its_checkpoint_specific_oracle(self) -> None:
        expected = gate.expected_continuation_token_ids("/models/Qwen3.5-9B")
        self.assertEqual(expected, (248044, 266, 506, 506, 506, 506, 506, 506, 506, 506))
        result = gate.compare_token_ids(
            [248044], 1, expected_token_ids=expected
        )
        self.assertTrue(result["correct"])

    def test_missing_or_malformed_output_fails_closed(self) -> None:
        missing = gate.compare_token_ids([], 1)
        malformed = gate.compare_token_ids("27841", 1)
        self.assertFalse(missing["correct"])
        self.assertEqual(
            missing["first_mismatch"],
            {"index": 0, "expected": 27841, "actual": None},
        )
        self.assertFalse(malformed["correct"])
        self.assertIsNotNone(malformed["error"])

    def test_unfrozen_continuation_length_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            gate.compare_token_ids([27841] * 11, 11)

    def test_both_engine_drivers_use_the_same_gate_and_prompt(self) -> None:
        for relative in (
            "examples/sglang/qwen35_inference.py",
            "examples/vllm/qwen35_inference.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("PROMPT_TOKEN_IDS", source)
            self.assertIn("compare_token_ids", source)
            self.assertIn("dummy weights cannot pass", source)
            self.assertNotIn("[9707, 11, 1879, 0]", source)


if __name__ == "__main__":
    unittest.main()
