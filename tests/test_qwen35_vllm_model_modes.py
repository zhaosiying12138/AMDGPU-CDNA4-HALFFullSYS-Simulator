#!/usr/bin/env python3
"""Pure-host contracts for Qwen3.5 explicit inference modes."""

from __future__ import annotations

import argparse
import ast
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
import errno
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save as save_safetensors


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples/triton/qwen35_vllm_model_forward.py"
FUNCTIONS = {
    "tensor_bytes",
    "tensor_sha256",
    "argument_parser",
    "validate_mode_args",
    "inference_mode_description",
    "mismatch_evidence_policy",
    "publish_actual_diagnostic",
    "tensor_sha256",
    "tensor_bytes",
    "snapshot_runtime_cache_guard",
    "validate_runtime_cache_guard",
    "validate_resume_suffix",
}
ASSIGNMENTS = {"INFERENCE_MODES", "PREFILL_TOKEN_IDS", "TOKEN_ID"}


def load_contracts() -> dict[str, object]:
    tree = ast.parse(RUNNER.read_text(), filename=str(RUNNER))
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in FUNCTIONS:
                selected.append(node)
        elif isinstance(node, ast.Assign):
            names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if names & ASSIGNMENTS:
                selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "argparse": argparse,
        "ctypes": ctypes,
        "datetime": datetime,
        "timezone": timezone,
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "errno": errno,
        "Path": Path,
        "tempfile": tempfile,
        "torch": torch,
        "save_safetensors": save_safetensors,
    }
    exec(compile(module, str(RUNNER), "exec"), namespace)
    return namespace


class ModeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contracts = load_contracts()

    def parse(self, arguments: list[str]):
        parser = self.contracts["argument_parser"]()
        return self.contracts["validate_mode_args"](
            parser, parser.parse_args(arguments)
        )

    def test_default_is_injection_free_production(self) -> None:
        args = self.parse([])
        self.assertEqual(args.inference_mode, "production")
        self.assertIsNone(args.debug_output_dir)
        self.assertIsNone(args.resume_checkpoint)
        self.assertIsNone(args.debug_stop_after_layer)
        description = self.contracts["inference_mode_description"]()
        production = description["modes"]["production"]
        self.assertFalse(production["oracle"])
        self.assertFalse(production["checkpoint_restore"])
        self.assertFalse(production["layer_checkpoint_publish"])
        self.assertTrue(production["acceptance_eligible"])

    def test_production_rejects_every_debug_injection(self) -> None:
        for arguments in (
            ["--debug-output-dir", "/tmp/out"],
            ["--resume-checkpoint", "/tmp/checkpoint"],
            ["--debug-stop-after-layer", "0"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.parse(arguments)

    def test_debug_modes_are_fail_closed_and_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            self.parse(["--inference-mode", "debug-layer-diff"])
        with self.assertRaises(SystemExit):
            self.parse(
                [
                    "--inference-mode",
                    "debug-resume",
                    "--debug-output-dir",
                    "/tmp/out",
                ]
            )
        with self.assertRaises(SystemExit):
            self.parse(
                [
                    "--inference-mode",
                    "evidence",
                    "--debug-output-dir",
                    "/tmp/out",
                    "--resume-checkpoint",
                    "/tmp/checkpoint",
                ]
            )
        with self.assertRaises(SystemExit):
            self.parse(
                [
                    "--inference-mode",
                    "debug-layer-diff",
                    "--debug-output-dir",
                    "/tmp/out",
                    "--prefill-golden-dir",
                    "/tmp/golden",
                ]
            )

    def test_debug_suffix_defaults_and_bounds(self) -> None:
        args = self.parse(
            [
                "--inference-mode",
                "debug-layer-diff",
                "--debug-output-dir",
                "/tmp/out",
            ]
        )
        self.assertEqual(args.debug_stop_after_layer, 23)
        with self.assertRaises(SystemExit):
            self.parse(
                [
                    "--inference-mode",
                    "evidence",
                    "--debug-output-dir",
                    "/tmp/out",
                    "--debug-stop-after-layer",
                    "24",
                ]
            )

    def test_first_mismatch_has_actual_evidence_but_no_resume_checkpoint(self) -> None:
        policy = self.contracts["mismatch_evidence_policy"](False)
        self.assertEqual(
            policy,
            {
                "publish_actual_diagnostic": True,
                "publish_resume_checkpoint": False,
                "stop": True,
                "target_feedback": False,
            },
        )
        passed = self.contracts["mismatch_evidence_policy"](True)
        self.assertFalse(passed["publish_actual_diagnostic"])
        self.assertTrue(passed["publish_resume_checkpoint"])
        self.assertFalse(passed["stop"])

    def test_actual_diagnostic_is_read_only_nonreplaceable_and_nonresumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            destination = parent / "layer-00-amd-actual"
            request = {
                "request_id": "a" * 64,
                "package_sha256": "b" * 64,
                "identity_sha256": "c" * 64,
                "payload_sha256": "d" * 64,
            }
            response = {
                "request_id": "a" * 64,
                "package_sha256": "e" * 64,
                "identity_sha256": "c" * 64,
                "payload_sha256": "f" * 64,
                "target_feedback": False,
            }
            result = self.contracts["publish_actual_diagnostic"](
                destination,
                layer_index=0,
                hidden=torch.zeros((2, 4), dtype=torch.bfloat16),
                residual=torch.ones((2, 4), dtype=torch.bfloat16),
                state={
                    "conv_state": torch.zeros((1, 3, 2), dtype=torch.bfloat16),
                    "recurrent_state": torch.zeros((1, 1), dtype=torch.float32),
                },
                request_record=request,
                response_record=response,
                oracle_execution={
                    "argv": ["python", "oracle.py"],
                    "exit_code": 0,
                    "stdout": "ok\n",
                    "stderr": "",
                    "stdout_sha256": hashlib.sha256(b"ok\n").hexdigest(),
                    "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                },
                identity_sha256="1" * 64,
            )
            self.assertFalse(result["resume_eligible"])
            self.assertFalse(result["target_feedback"])
            self.assertEqual(destination.stat().st_mode & 0o777, 0o500)
            self.assertEqual(
                {item.name for item in destination.iterdir()},
                {"actual.safetensors", "manifest.json"},
            )
            self.assertTrue(
                all(item.stat().st_mode & 0o777 == 0o400 for item in destination.iterdir())
            )
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertTrue(manifest["diagnostic_only"])
            self.assertFalse(manifest["resume_eligible"])
            self.assertFalse(manifest["target_feedback"])
            self.assertEqual(manifest["oracle_request"], request)
            self.assertEqual(manifest["oracle_response"], response)
            self.assertEqual(manifest["oracle_execution"]["exit_code"], 0)
            sentinel = destination / "sentinel"
            os.chmod(destination, 0o700)
            sentinel.write_text("preserve-existing")
            os.chmod(sentinel, 0o400)
            os.chmod(destination, 0o500)
            with self.assertRaises(FileExistsError):
                self.contracts["publish_actual_diagnostic"](
                    destination,
                    layer_index=0,
                    hidden=torch.zeros((2, 4), dtype=torch.bfloat16),
                    residual=torch.ones((2, 4), dtype=torch.bfloat16),
                    state={"kv_cache": torch.zeros((1, 1), dtype=torch.bfloat16)},
                    request_record=request,
                    response_record=response,
                    oracle_execution={
                        "argv": ["python", "oracle.py"],
                        "exit_code": 0,
                        "stdout": "ok\n",
                        "stderr": "",
                        "stdout_sha256": hashlib.sha256(b"ok\n").hexdigest(),
                        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                    },
                    identity_sha256="1" * 64,
                )
            self.assertEqual(sentinel.read_text(), "preserve-existing")
            self.assertEqual(
                {item.name for item in destination.iterdir()},
                {"actual.safetensors", "manifest.json", "sentinel"},
            )

    def test_protocol_is_the_only_oracle_identity_builder(self) -> None:
        source = RUNNER.read_text()
        tree = ast.parse(source)
        function_names = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertNotIn("oracle_request_identity", function_names)
        self.assertIn("current_layer_oracle_request_identity(Path(__file__))", source)
        self.assertNotIn("response.hidden_after.copy_", source)
        self.assertNotIn("response.residual_after.copy_", source)
        self.assertNotIn("response.mutable_state_after", source.split("def compare_online_layer", 1)[0])
        self.assertIn("sys.path.insert(0, str(HERE))", source)
        self.assertNotIn('os.environ.get("PYTHONPATH")', source)
        self.assertNotIn("os.environ['PYTHONPATH']", source)

    def test_cache_guard_rejects_non_target_writes_and_alias_changes(self) -> None:
        caches = [
            ("layer0", torch.zeros((2,), dtype=torch.bfloat16)),
            ("layer1", torch.zeros((2,), dtype=torch.bfloat16)),
        ]
        snapshot = self.contracts["snapshot_runtime_cache_guard"](caches)
        caches[0][1].add_(1)
        accepted = self.contracts["validate_runtime_cache_guard"](
            caches, snapshot, 0
        )
        self.assertTrue(accepted["correct"])
        self.assertTrue(accepted["non_target_cache_unchanged"])
        caches[1][1].add_(1)
        rejected = self.contracts["validate_runtime_cache_guard"](
            caches, snapshot, 0
        )
        self.assertFalse(rejected["correct"])
        self.assertFalse(rejected["non_target_cache_unchanged"])

        replacement = torch.ones((2,), dtype=torch.bfloat16)
        aliased = [("layer0", replacement), caches[1]]
        identity_rejected = self.contracts["validate_runtime_cache_guard"](
            aliased, snapshot, 0
        )
        self.assertFalse(identity_rejected["all_storage_identity_preserved"])

        signed_zero_caches = [
            ("layer0", torch.tensor([0.0], dtype=torch.float32)),
            ("layer1", torch.tensor([0.0], dtype=torch.float32)),
        ]
        signed_zero_snapshot = self.contracts["snapshot_runtime_cache_guard"](
            signed_zero_caches
        )
        signed_zero_caches[1][1].copy_(
            torch.tensor([-0.0], dtype=torch.float32)
        )
        self.assertTrue(
            torch.equal(
                signed_zero_caches[1][1],
                signed_zero_snapshot[1]["tensors"][0]["value"],
            )
        )
        bitwise_rejected = self.contracts["validate_runtime_cache_guard"](
            signed_zero_caches, signed_zero_snapshot, 0
        )
        self.assertFalse(bitwise_rejected["non_target_cache_unchanged"])
        self.assertNotEqual(
            bitwise_rejected["records"][1]["tensors"][0]["before_sha256"],
            bitwise_rejected["records"][1]["tensors"][0]["after_sha256"],
        )

    def test_resume_empty_suffix_only_allows_layer23_final_norm(self) -> None:
        validate = self.contracts["validate_resume_suffix"]
        self.assertTrue(
            validate(next_layer=24, stop_after_layer=23, resume_action="final_norm")
        )
        self.assertFalse(
            validate(
                next_layer=7,
                stop_after_layer=7,
                resume_action="decoder_layers_then_final_norm",
            )
        )
        with self.assertRaises(RuntimeError):
            validate(
                next_layer=8,
                stop_after_layer=7,
                resume_action="decoder_layers_then_final_norm",
            )
        with self.assertRaises(RuntimeError):
            validate(
                next_layer=24,
                stop_after_layer=22,
                resume_action="final_norm",
            )


if __name__ == "__main__":
    unittest.main()
