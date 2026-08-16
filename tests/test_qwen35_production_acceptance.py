#!/usr/bin/env python3
"""Pure-host gates for the continuous Qwen production acceptance policy."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/qwen35_production_acceptance.py"
SPEC = importlib.util.spec_from_file_location("qwen35_production_acceptance", PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def metric(*, mismatch: int = 1, nonfinite: int = 0, l2: float = 0.149, cosine: float = 0.981):
    return {
        "actual_sha256": "a" * 64,
        "expected_sha256": "b" * 64,
        "mismatch_count": mismatch,
        "nonfinite_count": nonfinite,
        "max_abs_error": 0.3,
        "relative_l2_error": l2,
        "cosine_similarity": cosine,
        "atol": 0.25,
        "rtol": 0.15,
        "max_relative_l2": 0.15,
        "correct": mismatch == 0,
    }


class ProductionAcceptanceTests(unittest.TestCase):
    def test_live_prerequisite_and_golden_are_bound(self) -> None:
        result = module.validate_strict_prerequisite()
        self.assertTrue(result["correct"])
        self.assertEqual(result["layers"], list(range(24)))
        self.assertTrue(result["final_norm_exact"])
        self.assertEqual(len(result["artifacts"]), 54)
        self.assertEqual(
            len({record["path"] for record in result["artifacts"]}),
            len(result["artifacts"]),
        )
        self.assertEqual(
            module.STRICT_EVIDENCE_MANIFEST,
            ROOT / "artifacts/qwen35-layer-diff/bridge-m4-evidence-manifest.json",
        )
        self.assertEqual(len(module.validate_golden()), 4)

    def test_atomic_publish_refuses_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = parent / "source"
            destination = parent / "destination"
            source.mkdir()
            destination.mkdir()
            marker = destination / "marker"
            marker.write_text("preserve")
            with self.assertRaises(FileExistsError):
                module.rename_noreplace(source, destination)
            self.assertEqual(marker.read_text(), "preserve")
            self.assertTrue(source.is_dir())

    def test_trajectory_keeps_pointwise_mismatch_diagnostic(self) -> None:
        value = metric(mismatch=7)
        self.assertFalse(value["correct"])
        self.assertTrue(module.trajectory_metric(value))
        self.assertFalse(module.trajectory_metric(metric(nonfinite=1)))
        self.assertFalse(module.trajectory_metric(metric(l2=0.150001)))
        self.assertFalse(module.trajectory_metric(metric(cosine=0.979999)))
        self.assertTrue(module.trajectory_metric(metric(l2=0.15, cosine=0.98)))

    def test_trajectory_aggregation_has_machine_claim_boundary(self) -> None:
        result = json.loads(
            (
                ROOT
                / "artifacts/evidence/P7-live-layer-diff/834-dispatch-baseline/result.json"
            ).read_text()
        )
        result["inference_mode"] = "production"
        accepted = module.validate_trajectory(result)
        self.assertTrue(accepted["correct"])
        self.assertGreater(accepted["pointwise_mismatch_total"], 0)
        self.assertEqual(accepted["comparison_count"], 87)

    def test_trajectory_rejects_step_or_cache_role_substitution(self) -> None:
        result = json.loads(
            (
                ROOT
                / "artifacts/evidence/P7-live-layer-diff/834-dispatch-baseline/result.json"
            ).read_text()
        )
        result["inference_mode"] = "production"
        result["step_comparisons"][1]["input_token_id"] = 1
        with self.assertRaises(module.AcceptanceError):
            module.validate_trajectory(result)
        result["step_comparisons"][1]["input_token_id"] = 27841
        result["prefill_cache_comparison"]["layers"][3]["tensors"][0]["name"] = "conv_state"
        with self.assertRaises(module.AcceptanceError):
            module.validate_trajectory(result)

    def test_trace_validation_rejects_missing_durable(self) -> None:
        source = ROOT / "artifacts/evidence/P7-live-layer-diff/834-dispatch-baseline"
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.jsonl"
            lines = (source / "dispatch-trace.jsonl").read_text().splitlines()
            trace.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaises(module.AcceptanceError):
                module.validate_trace(trace, source / "gem5.log")

    def test_production_command_has_no_debug_oracle_checkpoint_options(self) -> None:
        command = module.command_line(ROOT / "artifacts/unused")
        self.assertIn("production", command)
        joined = " ".join(command)
        for forbidden in (
            "debug-layer-diff",
            "debug-resume",
            "resume-checkpoint",
            "debug-output-dir",
            "teacher-force",
            "stop-after-layer",
        ):
            self.assertNotIn(forbidden, joined)

    def test_gem5_child_identity_is_derived_from_its_command(self) -> None:
        runtime_paths = module.current_runtime_paths()
        executable = runtime_paths[3]
        config = runtime_paths[4]
        run_dir = Path("/tmp") / f"self-amdgpu-opencl-run.{module.os.getuid()}.unit"
        argv = [
            str(executable),
            "--listener-mode=on",
            "--outdir",
            str(run_dir / "m5out"),
            str(config),
            "--endpoint",
            str(run_dir / "dispatch.sock"),
            "--dispatch-trace-path",
            str(run_dir / "dispatch-trace.jsonl"),
            "--epoch",
            "7",
            "--job-uuid",
            "1" * 32,
            "--rank",
            "0",
            "--world-size",
            "1",
        ]
        child = module.parse_gem5_child(123, argv, executable, config)
        self.assertEqual(child.run_dir, str(run_dir))
        self.assertEqual(child.job_uuid, "1" * 32)
        self.assertEqual(child.epoch, 7)
        self.assertEqual(child.executable_sha256, module.file_sha256(executable))
        self.assertEqual(child.config_sha256, module.file_sha256(config))

    def test_postflight_rejects_identity_drift(self) -> None:
        before = {"execution_identity": {"gem5_binary_sha256": "a" * 64}}
        with mock.patch.object(
            module,
            "validate_strict_prerequisite",
            return_value={"execution_identity": {"gem5_binary_sha256": "b" * 64}},
        ), mock.patch.object(module, "validate_golden", return_value=[]):
            with self.assertRaises(module.AcceptanceError):
                module.validate_postflight(before, [], module.file_sha256(module.THIS_FILE))


if __name__ == "__main__":
    unittest.main()
