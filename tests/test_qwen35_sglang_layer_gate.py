#!/usr/bin/env python3
"""Pure-host gates for SGLang's first-divergence layer diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = (
    ROOT
    / "tools/qwen35_sglang_layer_gate/qwen35_sglang_layer_gate.py"
)
SPEC = importlib.util.spec_from_file_location("qwen35_sglang_layer_gate", PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


class FakeGolden:
    def __init__(self, root: Path):
        zero = torch.zeros((2, 1024), dtype=torch.bfloat16)
        self.tensors = {"hidden_input": zero.clone()}
        for layer in range(24):
            self.tensors[f"layers.{layer}.returned_hidden"] = zero.clone()
            self.tensors[f"layers.{layer}.returned_residual"] = zero.clone()
        self.directory = root
        self.file_records = {}


class FakeModel(torch.nn.Module):
    pass


class FakeLayer(torch.nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id


class FakeLayerWithNorm(FakeLayer):
    def __init__(self, layer_id: int):
        super().__init__(layer_id)
        self.input_layernorm = torch.nn.Identity()


class Qwen35SglangLayerGateTests(unittest.TestCase):
    def test_tensor_comparison_passes_exact_and_rejects_wrong_or_nonfinite(self):
        expected = torch.zeros((2, 1024), dtype=torch.bfloat16)
        self.assertTrue(gate.compare_tensor(expected.clone(), expected)["correct"])
        wrong = expected.clone()
        wrong[0, 0] = 1
        compared = gate.compare_tensor(wrong, expected)
        self.assertFalse(compared["correct"])
        self.assertEqual(compared["mismatch_count"], 1)
        nonfinite = expected.clone()
        nonfinite[0, 0] = float("nan")
        self.assertFalse(gate.compare_tensor(nonfinite, expected)["correct"])

    def test_first_wrong_layer_publishes_evidence_and_stops(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            output = parent / "diagnostic"
            operator = mock.Mock()
            operator.directory = parent
            operator.file_records = {}
            operator.tensors = {}
            with mock.patch.object(gate, "load_operator_golden", return_value=operator):
                controller = gate.LayerGate(output, FakeGolden(parent))
            model = FakeModel()
            controller._start(model)
            layer = FakeLayer(0)
            hidden = torch.zeros((2, 1024), dtype=torch.bfloat16)
            controller.before_layer(
                layer,
                (),
                {"hidden_states": hidden, "residual": None},
            )
            wrong = hidden.clone()
            wrong[1, 17] = 2
            with self.assertRaises(gate.FirstNumericalMismatch):
                controller.after_layer(layer, (), {}, (wrong, hidden.clone()))
            mismatch = output / "first-mismatch"
            self.assertTrue((mismatch / "result.json").is_file())
            self.assertTrue((mismatch / "tensors.safetensors").is_file())
            self.assertFalse((output / "layer-gate-result.json").exists())

    def test_fixed_prompt_and_repeated_mrope_positions_are_recognized(self):
        self.assertEqual(
            gate._normalize_positions(torch.tensor([[0, 1], [0, 1], [0, 1]])),
            (0, 1),
        )
        self.assertIsNone(gate._normalize_positions(torch.tensor([[0, 1], [1, 0]])))

    def test_sitecustomize_is_strictly_opt_in(self):
        source = (
            ROOT / "tools/qwen35_sglang_layer_gate/sitecustomize.py"
        ).read_text(encoding="utf-8")
        self.assertIn("SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT", source)
        self.assertIn("triton_launch_probe/sitecustomize.py", source)
        self.assertNotIn("setattr", source)
        self.assertNotIn("copy_", source)

    def test_diagnostic_driver_fails_closed_when_hooks_are_absent(self):
        source = (ROOT / "examples/sglang/qwen35_inference.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT", source)
        self.assertIn("assert_installed()", source)
        saved = list(gate._handles)
        try:
            gate._handles.clear()
            with self.assertRaises(gate.LayerGateError):
                gate.assert_installed()
        finally:
            gate._handles[:] = saved

    def test_completion_gate_requires_all_layers(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "diagnostic"
            output.mkdir()
            import os

            previous = os.environ.get("SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT")
            os.environ["SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT"] = str(output)
            try:
                with self.assertRaises(gate.LayerGateError):
                    gate.assert_completed()
                (output / "layer-gate-result.json").write_text(
                    '{"schema":"amdgpu-sim.qwen35-sglang-layer-gate.v2",'
                    '"state":"layer_gate_passed","correct":true,'
                    '"layers_completed":24}',
                    encoding="ascii",
                )
                gate.assert_completed()
            finally:
                if previous is None:
                    os.environ.pop("SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT", None)
                else:
                    os.environ["SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT"] = previous

    def test_lane_mode_is_tp1_checkpoint_only_and_records_identity(self):
        source = (ROOT / "scripts/run_engine_lane.sh").read_text(encoding="utf-8")
        self.assertIn("--debug-layer-gate", source)
        self.assertIn("requires --engine sglang --tp 1", source)
        self.assertIn("requires checkpoint weights", source)
        self.assertIn("layer_gate_sha256=", source)
        self.assertIn("SAGR_TRITON_LAUNCH_LOG", source)
        self.assertIn("SAGR_QWEN35_OPERATOR_GOLDEN", source)

    def test_operator_gate_stops_on_first_wrong_boundary_with_replay_inputs(self):
        class FakeOperator:
            directory = Path("/operator")
            file_records = {}
            tensors = {
                name: torch.zeros((2, 4), dtype=torch.bfloat16)
                for name in gate.OPERATOR_ORDER
            }

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            output = parent / "diagnostic"
            with mock.patch.object(
                gate, "load_operator_golden", return_value=FakeOperator()
            ):
                controller = gate.LayerGate(output, FakeGolden(parent))
            controller.output.mkdir()
            controller.operator_active = True
            controller.operator_next = 1
            actual = torch.zeros((2, 4), dtype=torch.bfloat16)
            actual[0, 0] = 4
            with self.assertRaises(gate.FirstNumericalMismatch):
                controller._record_operator(
                    "input_rms_norm",
                    actual,
                    inputs=(torch.ones((2, 4), dtype=torch.bfloat16),),
                    metadata={"module_path": "input_layernorm"},
                )
            result = (output / "first-mismatch/result.json").read_text(
                encoding="ascii"
            )
            self.assertIn('"replay_capsule":true', result)
            self.assertIn('"operator":"input_rms_norm"', result)
            from safetensors import safe_open

            with safe_open(
                output / "first-mismatch/tensors.safetensors",
                framework="pt",
                device="cpu",
            ) as source:
                self.assertEqual(set(source.keys()), {"actual", "expected", "input_0"})

    def test_empty_layer_args_arm_and_first_norm_captures_hidden(self):
        """SGLang passes decoder-layer inputs entirely through kwargs."""

        class FakeOperator:
            directory = Path("/operator")
            file_records = {}
            tensors = {
                name: torch.zeros((2, 1024), dtype=torch.bfloat16)
                for name in gate.OPERATOR_ORDER
            }

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            with mock.patch.object(
                gate, "load_operator_golden", return_value=FakeOperator()
            ):
                controller = gate.LayerGate(parent / "diagnostic", FakeGolden(parent))
            layer = FakeLayerWithNorm(0)
            controller.prompt_seen = True
            with mock.patch.object(gate, "_install_function_wrappers") as install:
                with mock.patch.object(gate, "_patched_functions", {object(): object()}):
                    # The wrapper-count contract must be satisfied without
                    # importing SGLang in this pure-host unit test.
                    patched = gate._patched_functions
                    patched.clear()
                    for index, name in enumerate(
                        sorted(gate._REQUIRED_FUNCTION_WRAPPERS)
                    ):
                        patched[(index, name)] = object()
                    controller.arm_operator_layer(layer, ())
                install.assert_not_called()
            self.assertTrue(controller.operator_active)
            self.assertEqual(controller.operator_next, 0)
            hidden = torch.zeros((2, 1024), dtype=torch.bfloat16)
            controller.observe_module(
                layer.input_layernorm,
                (hidden,),
                {},
                hidden.clone(),
            )
            self.assertEqual(controller.operator_next, 2)
            records = (parent / "diagnostic/operator-comparisons.jsonl").read_text(
                encoding="ascii"
            )
            self.assertIn('"operator":"hidden_input"', records)
            self.assertIn('"operator":"input_rms_norm"', records)


if __name__ == "__main__":
    unittest.main()
