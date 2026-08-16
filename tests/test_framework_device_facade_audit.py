from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "framework_device_facade_audit",
    ROOT / "tools/framework_device_facade_audit.py",
)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


class FrameworkDeviceFacadeAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (ROOT / "tools/framework_device_facade_manifest.json").read_text()
        )

    def test_current_source_identity_and_incomplete_model_boundary(self) -> None:
        result = AUDIT.audit(ROOT, self.manifest)
        self.assertTrue(result["correct"], result["errors"])
        self.assertFalse(result["model_ready"])
        self.assertEqual(result["capability_status"]["triton_upstream_hip"], "source_ready")
        self.assertEqual(result["capability_status"]["sglang_upstream_rocm"], "source_ready")
        self.assertEqual(result["capability_status"]["rocr_provider"], "in_progress")
        self.assertEqual(result["capability_status"]["hip_runtime"], "in_progress")
        self.assertEqual(result["capability_status"]["rccl_abi"], "in_progress")

    def test_model_ready_is_derived_from_generic_prerequisites(self) -> None:
        changed = copy.deepcopy(self.manifest)
        required = set(changed["acceptance"]["device_facade_prerequisite"])
        required.update(changed["acceptance"]["distributed_prerequisite"])
        for family in changed["capability_families"]:
            if family["id"] in required:
                family["status"] = "accepted"
                family["blocker"] = None
        changed["acceptance"]["model_ready"] = True
        result = AUDIT.audit(ROOT, changed)
        self.assertTrue(result["correct"], result["errors"])
        self.assertTrue(result["model_ready"])

    def test_false_model_claim_fails_closed(self) -> None:
        changed = copy.deepcopy(self.manifest)
        changed["acceptance"]["model_ready"] = True
        result = AUDIT.audit(ROOT, changed)
        self.assertFalse(result["correct"])
        self.assertIn(
            "model_ready differs from prerequisite capability states",
            result["errors"],
        )

    def test_no_operator_or_model_names_in_capability_contract(self) -> None:
        families = json.dumps(
            self.manifest["capability_families"], sort_keys=True
        ).lower()
        for forbidden in ("qwen", "attention", "rms", "gemm", "flashinfer"):
            self.assertNotIn(forbidden, families)


if __name__ == "__main__":
    unittest.main()
