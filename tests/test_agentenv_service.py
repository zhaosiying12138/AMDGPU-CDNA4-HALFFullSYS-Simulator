# SPDX-License-Identifier: GPL-3.0-or-later
"""Host-only tests for the feature-local AgentENV server lifecycle tool."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "agentenv_service", ROOT / "tools" / "agentenv_service.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


SOURCE_CONFIG = b'''\
home_path = "/var/lib/aenv"

[image.resolver]
default_image = "ubuntu:24.04"
'''


def fixture_repo(temporary: str) -> Path:
    repo = Path(temporary) / "repo"
    config = repo / "projects" / "AgentENV" / "config" / "default.toml"
    config.parent.mkdir(parents=True)
    config.write_bytes(SOURCE_CONFIG)
    (repo / "projects" / "AgentENV").mkdir(parents=True, exist_ok=True)
    return repo


class AgentEnvServiceTest(unittest.TestCase):
    maxDiff = None

    def test_plan_is_loopback_and_feature_local(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            result = MODULE.desired_plan(
                paths, "127.0.0.1:18080", "ubuntu:26.04"
            )

        state = repo / "build" / "agentenv-integration" / "server"
        self.assertEqual(result["state_root"], str(state))
        self.assertEqual(result["api_addr"], "127.0.0.1:18080")
        self.assertEqual(result["environment"]["AENV_HOME_PATH"], str(state / "home"))
        self.assertEqual(result["environment"]["AENV_RUNTIME_PATH"], str(state / "runtime"))
        self.assertEqual(result["environment"]["AENV_DEPS_PATH"], str(state / "deps"))
        self.assertTrue(result["safety"]["loopback_only"])
        self.assertFalse(result["safety"]["wsl_shutdown"])

    def test_non_loopback_api_is_rejected(self) -> None:
        for value in ("0.0.0.0:18080", "192.168.1.1:18080", "localhost:18080"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(MODULE.ServiceError, "loopback|invalid"):
                    MODULE.validate_api_addr(value)

    def test_prepare_generates_private_config_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            hashes = MODULE.prepare_layout(paths, "127.0.0.1:18080", "ubuntu:26.04")
            rendered = paths.config.read_text(encoding="utf-8")
            environment = json.loads(paths.environment_json.read_text(encoding="ascii"))

            self.assertIn('default_image = "ubuntu:26.04"', rendered)
            self.assertNotIn('default_image = "ubuntu:24.04"', rendered)
            self.assertEqual(environment["API_ADDR"], "127.0.0.1:18080")
            self.assertEqual(environment["E2B_API_URL"], "http://127.0.0.1:18080")
            self.assertTrue(environment["AENV_HOME_PATH"].startswith(str(paths.root)))
            self.assertEqual(len(hashes["config_sha256"]), 64)
            self.assertEqual(paths.config.stat().st_mode & 0o777, 0o600)
            self.assertEqual(paths.environment_sh.stat().st_mode & 0o777, 0o600)
            credentials = paths.credentials.read_text(encoding="ascii")
            self.assertIn('url = "http://127.0.0.1:18080"', credentials)
            self.assertIn('api_key = "e2b_000000"', credentials)
            metadata = json.loads(paths.credentials_metadata.read_text(encoding="ascii"))
            self.assertEqual(metadata["scope"], "loopback-development-only")
            self.assertEqual(paths.credentials.stat().st_mode & 0o777, 0o600)

    def test_start_dry_run_is_pure_and_does_not_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            with mock.patch.object(MODULE, "prerequisite_report", return_value={
                "ready": True, "blockers": [], "devices": {}
            }), mock.patch.object(
                MODULE.subprocess,
                "Popen",
                side_effect=AssertionError("dry-run must not spawn"),
            ):
                result = MODULE.start_service(
                    paths,
                    "127.0.0.1:18080",
                    "ubuntu:26.04",
                    dry_run=True,
                )

            self.assertEqual(result["result"], "would-start")
            self.assertFalse(paths.root.exists())

    def test_start_dry_run_reports_missing_prerequisites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            missing = {
                "ready": False,
                "blockers": ["ublk_control"],
                "devices": {},
            }
            with mock.patch.object(MODULE, "prerequisite_report", return_value=missing):
                result = MODULE.start_service(
                    paths,
                    "127.0.0.1:18080",
                    "ubuntu:26.04",
                    dry_run=True,
                )

            self.assertEqual(result["result"], "would-refuse-missing-prereqs")
            self.assertFalse(paths.root.exists())

    def test_start_refuses_missing_prerequisites_without_writing_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            missing = {
                "ready": False,
                "blockers": ["kvm", "ublk_control"],
                "devices": {},
            }
            with mock.patch.object(MODULE, "prerequisite_report", return_value=missing):
                with self.assertRaisesRegex(MODULE.ServiceError, "prerequisites"):
                    MODULE.start_service(
                        paths,
                        "127.0.0.1:18080",
                        "ubuntu:26.04",
                        dry_run=False,
                    )

            self.assertFalse(paths.root.exists())

    def test_symlinked_pidfile_is_rejected_before_read_or_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            paths.root.mkdir(parents=True)
            outside = Path(temporary) / "outside-process.json"
            outside.write_text("{}", encoding="ascii")
            paths.pidfile.symlink_to(outside)
            with self.assertRaisesRegex(MODULE.ServiceError, "symlink"):
                MODULE.service_status(paths, "127.0.0.1:18080", check_health=False)

    def test_stop_requires_confirmation_before_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            paths.root.mkdir(parents=True)
            executable = os.path.realpath(paths.binary)
            record = {
                "schema": MODULE.PROCESS_SCHEMA,
                "pid": 4242,
                "starttime_ticks": 99,
                "executable": executable,
            }
            paths.pidfile.write_bytes(MODULE.canonical_json(record))
            identity = MODULE.ProcessIdentity(4242, 99, executable)
            with mock.patch.object(MODULE, "process_identity", return_value=identity), mock.patch.object(
                MODULE.os, "kill", side_effect=AssertionError("confirmation required")
            ):
                with self.assertRaisesRegex(MODULE.ServiceError, "--confirm"):
                    MODULE.stop_service(
                        paths,
                        "127.0.0.1:18080",
                        dry_run=False,
                        confirm=False,
                        timeout=0,
                        kill_after_timeout=False,
                    )

    def test_mismatched_identity_is_never_signalled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            paths.root.mkdir(parents=True)
            executable = os.path.realpath(paths.binary)
            record = {
                "schema": MODULE.PROCESS_SCHEMA,
                "pid": 4242,
                "starttime_ticks": 99,
                "executable": executable,
            }
            paths.pidfile.write_bytes(MODULE.canonical_json(record))
            reused = MODULE.ProcessIdentity(4242, 100, "/usr/bin/python3")
            with mock.patch.object(MODULE, "process_identity", return_value=reused), mock.patch.object(
                MODULE.os, "kill", side_effect=AssertionError("must not signal")
            ):
                result = MODULE.stop_service(
                    paths,
                    "127.0.0.1:18080",
                    dry_run=False,
                    confirm=True,
                    timeout=0,
                    kill_after_timeout=False,
                )

            self.assertEqual(result["result"], "stale-record-cleared-without-signal")
            self.assertFalse(paths.pidfile.exists())

    def test_cli_help_lists_lifecycle_commands(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "agentenv_service.py"), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("plan", "status", "start", "stop"):
            self.assertIn(command, result.stdout)


if __name__ == "__main__":
    unittest.main()
