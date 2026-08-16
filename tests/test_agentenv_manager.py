# SPDX-License-Identifier: GPL-3.0-or-later
"""Host-only contracts for the feature-local AgentENV manager."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "agentenv_manager", ROOT / "tools" / "agentenv_manager.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def start_args(root: Path, state: Path, **overrides: object) -> Namespace:
    values: dict[str, object] = {
        "repo": str(root),
        "state_dir": str(state),
        "api": "http://127.0.0.1:18080",
        "feature_id": "test-agentenv",
        "instance": None,
        "http_timeout": 0.1,
        "dry_run": True,
        "json": True,
        "template": "fixture-template",
        "bundle_manifest": None,
        "timeout": 120,
        "allow_internet": False,
        "allow_live": False,
        "cpu_count": 12,
        "memory_mb": 24576,
        "disk_size_mb": 98304,
    }
    values.update(overrides)
    return Namespace(**values)


class AgentEnvManagerTest(unittest.TestCase):
    maxDiff = None

    def test_namespace_is_stable_unique_and_feature_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            first = MODULE.namespace_for(repo, "vllm-tp4", "test-feature")
            repeat = MODULE.namespace_for(repo, "vllm-tp4", "test-feature")
            other_instance = MODULE.namespace_for(repo, "sglang-tp1", "test-feature")
            other_feature = MODULE.namespace_for(repo, "vllm-tp4", "other-feature")

        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other_instance)
        self.assertNotEqual(first, other_feature)
        self.assertTrue(first.startswith("aenv-vllm-tp4-"))
        self.assertNotIn("/", first)

    def test_instance_payload_separates_runtime_namespaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            namespace, payload = MODULE._instance_payload(
                repo,
                "vllm-tp4",
                template="fixture-template",
                timeout=120,
                feature_id="test-feature",
                bundle_manifest_sha256="a" * 64,
                allow_internet=False,
                cpu_count=12,
                memory_mb=24576,
                disk_size_mb=98304,
            )

        self.assertEqual(payload["templateID"], "fixture-template")
        self.assertEqual(payload["timeout"], 120)
        self.assertFalse(payload["allow_internet_access"])
        self.assertEqual(payload["metadata"]["agentenv_namespace"], namespace)
        self.assertEqual(payload["envVars"]["AGENTENV_NAMESPACE"], namespace)
        self.assertEqual(payload["envVars"]["TMPDIR"], f"/tmp/{namespace}")
        self.assertEqual(payload["envVars"]["XDG_RUNTIME_DIR"], f"/run/{namespace}")
        self.assertEqual(payload["envVars"]["XDG_CACHE_HOME"], f"/var/cache/{namespace}")
        self.assertEqual(payload["metadata"]["agentenv_cpu_count"], "12")
        self.assertEqual(payload["metadata"]["agentenv_memory_mb"], "24576")
        self.assertEqual(payload["metadata"]["agentenv_disk_size_mb"], "98304")

    def test_start_pair_dry_run_is_pure_and_plans_two_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            state = root / "build" / "agentenv-integration" / "state"
            root.mkdir(parents=True)
            args = start_args(root, state)
            with mock.patch.object(MODULE, "scan_live_workloads", return_value=[]), mock.patch.object(
                MODULE, "api_request", side_effect=AssertionError("dry-run must not call API")
            ):
                result = MODULE.start_pair(args)

            self.assertEqual(result["operation"], "start-pair")
            self.assertEqual(
                result["resource_contract"],
                {"cpu_count": 12, "memory_mb": 24576, "disk_size_mb": 98304},
            )
            self.assertEqual([item["instance"] for item in result["plans"]], ["vllm-tp4", "sglang-tp1"])
            self.assertEqual(result["sandboxes"], [])
            self.assertFalse((state / "instances").exists())

    def test_start_pair_refuses_unrelated_live_processes_before_planning(self) -> None:
        process = MODULE.LiveProcess(
            pid=12345,
            command="python -m vllm.entrypoints.openai.api_server",
            cwd="/home/other-workload",
            feature_owned=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            args = start_args(root, root / "state")
            with mock.patch.object(MODULE, "scan_live_workloads", return_value=[process]):
                with self.assertRaisesRegex(MODULE.ManagerError, "unrelated"):
                    MODULE.start_pair(args)

    def test_start_pair_rolls_back_partial_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            state = root / "state"
            root.mkdir(parents=True)
            args = start_args(root, state, dry_run=False, allow_live=True)
            calls: list[tuple[str, str]] = []

            def api_side_effect(
                _base: str,
                method: str,
                path: str,
                **_kwargs: object,
            ) -> tuple[int, object, dict[str, str]]:
                calls.append((method, path))
                if method == "POST" and path == "/sandboxes" and len(calls) == 1:
                    return 201, {"sandboxID": "sb-first"}, {}
                if method == "POST":
                    raise MODULE.ManagerError("second create failed")
                if method == "DELETE" and path == "/sandboxes/sb-first":
                    return 204, None, {}
                raise AssertionError((method, path))

            with mock.patch.object(MODULE, "scan_live_workloads", return_value=[]), mock.patch.object(
                MODULE, "api_request", side_effect=api_side_effect
            ):
                with self.assertRaisesRegex(MODULE.ManagerError, "rollback"):
                    MODULE.start_pair(args)

            self.assertIn(("DELETE", "/sandboxes/sb-first"), calls)
            state_record = json.loads(
                (state / "instances" / "vllm-tp4" / "sandbox.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIsInstance(state_record.get("deleted_at"), str)
            self.assertEqual(state_record["delete_status"], 204)

    def test_stop_without_confirmation_never_calls_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            state = root / "state"
            instance_dir = state / "instances" / "vllm-tp4"
            instance_dir.mkdir(parents=True)
            (instance_dir / "sandbox.json").write_text(
                json.dumps({"instance": "vllm-tp4", "sandbox_id": "sb-fixture"}),
                encoding="utf-8",
            )
            args = Namespace(
                repo=str(root),
                state_dir=str(state),
                api="http://127.0.0.1:18080",
                instance=["vllm-tp4"],
                http_timeout=0.1,
                dry_run=False,
                confirm=False,
            )
            with mock.patch.object(MODULE, "api_request", side_effect=AssertionError("confirmation required")):
                result = MODULE.stop(args)

        self.assertEqual(result["deleted"], [])
        self.assertIn("--confirm", result["note"])

    def test_cli_help_exposes_all_lifecycle_commands(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "agentenv_manager.py"), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("host-preflight", "status", "collect", "start-pair", "stop"):
            self.assertIn(command, result.stdout)


if __name__ == "__main__":
    unittest.main()
