# SPDX-License-Identifier: GPL-3.0-or-later
"""Fail-closed tests for the generic Qwen3.5 operator result queue."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen35_operator_manifest_for_queue_tests",
    ROOT / "tools/qwen35_operator_manifest.py",
)
assert SPEC and SPEC.loader
manifest_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest_tool)

from qwen35_operator_work_queue import (  # noqa: E402
    REQUIRED_ARTIFACT_ROLES,
    REQUIRED_FALLBACK_COUNTERS,
    REQUIRED_LIFECYCLE_STAGES,
    RESULT_SCHEMA,
    derive_work_item_status,
    validate_manifest,
    validate_result,
    work_item_spec_sha256,
)


class Qwen35OperatorWorkQueueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = manifest_tool.build_manifest()

    def _manifest(self) -> dict:
        return copy.deepcopy(self.base)

    def _rehash(self, item: dict) -> None:
        item["spec_sha256"] = work_item_spec_sha256(item)

    def _configured_required(self, manifest: dict) -> dict:
        return next(
            item
            for item in manifest["work_items"]
            if item["configuration_status"] == "configured"
        )

    def _result(
        self, item: dict, root: Path, run_kind: str
    ) -> dict:
        artifacts = []
        for role in REQUIRED_ARTIFACT_ROLES:
            path = root / "artifacts" / run_kind / role
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"{role}-payload".encode("ascii")
            path.write_bytes(payload)
            artifacts.append(
                {
                    "role": role,
                    "path": path.relative_to(root).as_posix(),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        code_object_sha256 = next(
            artifact["sha256"]
            for artifact in artifacts
            if artifact["role"] == "code_object"
        )
        return {
            "schema": RESULT_SCHEMA,
            "work_item_id": item["id"],
            "spec_sha256": item["spec_sha256"],
            "run_kind": run_kind,
            "normal_user_entrypoint": True,
            "exit_code": 0,
            "target": copy.deepcopy(item["kernel"]["target"]),
            "lifecycle": {stage: True for stage in REQUIRED_LIFECYCLE_STAGES},
            "oracle": {
                "passed": True,
                "finite": True,
                "mismatch_count": 0,
                "nonfinite_count": 0,
                "output_sha256": "1" * 64,
                "metrics": {"max_abs_error": 0.0},
            },
            "fallback": {
                "used": False,
                "process_audit_passed": True,
                "counters": {name: 0 for name in REQUIRED_FALLBACK_COUNTERS},
                "observed_backends": [],
                "forbidden_dsos_observed": [],
                "forbidden_device_nodes_observed": [],
            },
            "cache": {"key": "stable-cache-key", "hit": run_kind == "repeat"},
            "provenance": {
                "gem5_tree": "2" * 40,
                "runtime_commit": "3" * 40,
                "triton_commit": item["provenance"]["required_repo_commits"][
                    "triton"
                ],
                "prefix_manifest_sha256": "4" * 64,
                "setup_sha256": "5" * 64,
                "runner_sha256": item["source"]["runner"]["sha256"],
                "code_object_sha256": code_object_sha256,
                "environment_sha256": "6" * 64,
            },
            "artifacts": artifacts,
        }

    def test_generated_source_queue_validates(self) -> None:
        self.assertEqual(validate_manifest(self.base), [])
        self.assertEqual(len(self.base["contracts"]), 15)
        self.assertEqual(len(self.base["work_items"]), 32)

    def test_symbolic_shape_is_rejected_even_with_updated_spec_hash(self) -> None:
        manifest = self._manifest()
        item = self._configured_required(manifest)
        item["tensors"][0]["shape"][0] = "rows"
        self._rehash(item)
        errors = validate_manifest(manifest)
        self.assertTrue(any("shape must be concrete" in error for error in errors))

    def test_nan_tolerant_oracle_is_rejected(self) -> None:
        manifest = self._manifest()
        item = self._configured_required(manifest)
        item["oracle"]["comparisons"][0]["equal_nan"] = True
        self._rehash(item)
        errors = validate_manifest(manifest)
        self.assertTrue(any("must reject NaN equality" in error for error in errors))

    def test_source_manifest_cannot_claim_accepted(self) -> None:
        manifest = self._manifest()
        item = self._configured_required(manifest)
        item["status"] = "accepted"
        errors = validate_manifest(manifest)
        self.assertTrue(any("source manifest status" in error for error in errors))

    def test_source_path_escape_is_rejected(self) -> None:
        manifest = self._manifest()
        item = self._configured_required(manifest)
        item["source"]["entrypoints"][0]["path"] = "../outside.py"
        self._rehash(item)
        errors = validate_manifest(manifest)
        self.assertTrue(any("source repository, path" in error for error in errors))

    def test_fallback_policy_cannot_be_relaxed(self) -> None:
        manifest = self._manifest()
        item = self._configured_required(manifest)
        item["fallback"]["allowed"] = True
        self._rehash(item)
        errors = validate_manifest(manifest)
        self.assertTrue(any("fallback must be forbidden" in error for error in errors))

    def test_dependency_cycle_is_rejected(self) -> None:
        manifest = self._manifest()
        first, second = manifest["work_items"][:2]
        first["dependencies"]["all_of_work_items"] = [second["id"]]
        second["dependencies"]["all_of_work_items"] = [first["id"]]
        self._rehash(first)
        self._rehash(second)
        errors = validate_manifest(manifest)
        self.assertTrue(any("dependency cycle" in error for error in errors))

    def test_precomputed_code_object_identity_is_rejected(self) -> None:
        manifest = self._manifest()
        item = self._configured_required(manifest)
        item["kernel"]["code_objects"]["expected_sha256"] = "a" * 64
        self._rehash(item)
        errors = validate_manifest(manifest)
        self.assertTrue(any("precomputed identity" in error for error in errors))

    def test_generic_fresh_and_repeat_results_are_accepted(self) -> None:
        item = self._configured_required(self.base)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            results = [
                self._result(item, root, "fresh"),
                self._result(item, root, "repeat"),
            ]
            self.assertEqual(validate_result(item, results[0], root), [])
            self.assertEqual(validate_result(item, results[1], root), [])
            errors: list[str] = []
            self.assertEqual(
                derive_work_item_status(item, results, root, errors), "accepted"
            )
            self.assertEqual(errors, [])

    def test_result_wrong_spec_and_fallback_are_rejected(self) -> None:
        item = self._configured_required(self.base)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            result = self._result(item, root, "fresh")
            result["spec_sha256"] = "0" * 64
            result["fallback"]["counters"]["fallback_count"] = 1
            errors = validate_result(item, result, root)
            self.assertTrue(any("spec SHA-256" in error for error in errors))
            self.assertTrue(any("fallback audit" in error for error in errors))

    def test_result_lifecycle_and_artifact_escape_are_rejected(self) -> None:
        item = self._configured_required(self.base)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            result = self._result(item, root, "fresh")
            result["lifecycle"]["launch"] = False
            result["artifacts"][0]["path"] = "../outside"
            errors = validate_result(item, result, root)
            self.assertTrue(any("lifecycle" in error for error in errors))
            self.assertTrue(any("artifact identity" in error for error in errors))

    def test_validator_contains_no_checkpoint_or_operator_special_case(self) -> None:
        source = (ROOT / "tools/qwen35_operator_work_queue.py").read_text(
            encoding="utf-8"
        ).lower()
        for token in (
            "cp-003",
            "ev-00",
            "rms_norm",
            "silu_and_mul",
            "expected_sha256",
            "pc_coverage",
            "root_parent_commit",
        ):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
