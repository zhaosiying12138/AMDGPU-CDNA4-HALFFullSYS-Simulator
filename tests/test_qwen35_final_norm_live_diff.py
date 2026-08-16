#!/usr/bin/env python3
"""Pure-host contracts for the checkpoint-bound final-norm diagnostic."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import uuid

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples/triton/qwen35_final_norm_live_diff.py"
MAIN_RUNNER = ROOT / "examples/triton/qwen35_vllm_model_forward.py"
FUNCTIONS = {
    "tensor_sha256",
    "compare_final_hidden",
    "publish_result",
}
ASSIGNMENTS = {
    "EXPECTED_MAIN_RUNNER_SHA256",
    "ATOL",
    "RTOL",
    "MAX_RELATIVE_L2",
    "MIN_COSINE",
}


def load_contracts() -> dict[str, object]:
    tree = ast.parse(RUNNER.read_text(), filename=str(RUNNER))
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if names & ASSIGNMENTS:
                selected.append(node)
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "Path": Path,
        "torch": torch,
        "uuid": uuid,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(RUNNER), "exec"),
        namespace,
    )
    return namespace


class FinalNormLiveDiffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contracts = load_contracts()

    def test_main_runner_identity_is_frozen_and_unchanged(self) -> None:
        observed = hashlib.sha256(MAIN_RUNNER.read_bytes()).hexdigest()
        self.assertEqual(
            observed, self.contracts["EXPECTED_MAIN_RUNNER_SHA256"]
        )
        self.assertEqual(
            observed,
            "878acfa8d37a81a204d2aff1844d7618fa571e274a05a8ba47f5f31312649343",
        )

    def test_strict_final_norm_thresholds(self) -> None:
        self.assertEqual(self.contracts["ATOL"], 0.03125)
        self.assertEqual(self.contracts["RTOL"], 0.03)
        self.assertEqual(self.contracts["MAX_RELATIVE_L2"], 0.03)
        self.assertEqual(self.contracts["MIN_COSINE"], 0.98)
        expected = torch.linspace(-2, 2, 2048, dtype=torch.float32).reshape(
            2, 1024
        ).to(torch.bfloat16)
        exact = self.contracts["compare_final_hidden"](expected.clone(), expected)
        self.assertTrue(exact["correct"])
        self.assertEqual(exact["mismatch_count"], 0)
        self.assertEqual(exact["nonfinite_count"], 0)
        self.assertAlmostEqual(exact["cosine_similarity"], 1.0, places=6)

        actual = expected.clone()
        actual[0, 0] = actual[0, 0] + 1.0
        rejected = self.contracts["compare_final_hidden"](actual, expected)
        self.assertFalse(rejected["correct"])
        self.assertGreater(rejected["mismatch_count"], 0)

    def test_nonfinite_and_tensor_contracts_fail_closed(self) -> None:
        expected = torch.zeros((2, 1024), dtype=torch.bfloat16)
        nonfinite = expected.clone()
        nonfinite[0, 0] = float("nan")
        compared = self.contracts["compare_final_hidden"](nonfinite, expected)
        self.assertFalse(compared["correct"])
        self.assertEqual(compared["nonfinite_count"], 1)
        for actual in (
            torch.zeros((1, 1024), dtype=torch.bfloat16),
            torch.zeros((2, 1024), dtype=torch.float32),
        ):
            with self.subTest(actual=(actual.dtype, actual.shape)):
                with self.assertRaises(RuntimeError):
                    self.contracts["compare_final_hidden"](actual, expected)

    def test_result_publish_is_atomic_and_no_replace(self) -> None:
        publish = self.contracts["publish_result"]
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "result.json"
            payload = {
                "schema": "test",
                "diagnostic_only": True,
                "acceptance_eligible": False,
                "oracle_feedback_to_target": False,
            }
            publish(result, payload)
            self.assertEqual(result.stat().st_mode & 0o777, 0o400)
            self.assertEqual(json.loads(result.read_text()), payload)
            original = result.read_bytes()
            with self.assertRaises(FileExistsError):
                publish(result, {"schema": "replacement"})
            self.assertEqual(result.read_bytes(), original)

    def test_oracle_precedes_target_and_response_is_comparison_only(self) -> None:
        source = RUNNER.read_text()
        oracle_call = source.index("response = run_layer_oracle(")
        target_call = source.index(
            "actual, residual_out = torch.ops.gemsim.fused_add_gemma_rms_norm("
        )
        compare_call = source.index(
            "comparison = compare_final_hidden(actual, response.final_hidden_after)"
        )
        self.assertLess(oracle_call, target_call)
        self.assertLess(target_call, compare_call)
        self.assertNotIn("response.final_hidden_after.copy_", source)
        self.assertNotIn("actual = response.final_hidden_after", source)
        self.assertIn('"oracle_feedback_to_target": False', source)
        self.assertIn('"acceptance_eligible": False', source)
        self.assertIn('"fallback_count": 0', source)
        self.assertIn('"final_norm_runner_sha256": file_sha256(Path(__file__).resolve())', source)

    def test_checkpoint_and_weight_are_exactly_bound(self) -> None:
        source = RUNNER.read_text()
        self.assertIn("load_layer_checkpoint(checkpoint_path)", source)
        self.assertIn("checkpoint.after_layer == 23", source)
        self.assertIn('boundary["resume_action"] == "final_norm"', source)
        self.assertIn("publish_final_norm_oracle_request(", source)
        self.assertIn("source_checkpoint=checkpoint_path", source)
        self.assertIn('FINAL_NORM_WEIGHT_KEY = "model.language_model.norm.weight"', source)
        self.assertIn("FINAL_NORM_WEIGHT_SHA256", source)
        self.assertIn("actual_matches_source_model_dispatch_sha256", source)
        self.assertIn("source_result[\"final_hidden_sha256\"]", source)
        self.assertIn("source_result[\"checkpoint_manifest_sha256\"]", source)

    def test_isolated_entry_bootstraps_before_sibling_imports(self) -> None:
        source = RUNNER.read_text()
        bootstrap = source.index('runpy.run_path(str(HERE / "_gemsim_bootstrap.py"))')
        path_insert = source.index("sys.path.insert(0, str(HERE))")
        checkpoint_import = source.index(
            "from _qwen35_layer_checkpoint import load_layer_checkpoint"
        )
        protocol_import = source.index("from _qwen35_layer_oracle_protocol import (")
        self.assertLess(bootstrap, path_insert)
        self.assertLess(path_insert, checkpoint_import)
        self.assertLess(path_insert, protocol_import)


if __name__ == "__main__":
    unittest.main()
