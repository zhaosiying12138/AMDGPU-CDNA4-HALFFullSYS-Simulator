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

[pool]
low_watermark = 2
high_watermark = 64
'''


def fixture_repo(temporary: str) -> Path:
    repo = Path(temporary) / "repo"
    config = repo / "projects" / "AgentENV" / "config" / "default.toml"
    config.parent.mkdir(parents=True)
    config.write_bytes(SOURCE_CONFIG)
    (repo / "projects" / "AgentENV").mkdir(parents=True, exist_ok=True)
    return repo


def fixture_process_record(
    paths: MODULE.ServicePaths,
    *,
    pid: int = 4242,
    starttime_ticks: int = 99,
    boot_id: str = "test-boot-id",
) -> dict[str, object]:
    paths.binary.parent.mkdir(parents=True, exist_ok=True)
    paths.binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    paths.binary.chmod(0o700)
    executable = MODULE.executable_identity(paths.binary)
    assert executable is not None
    return {
        "schema": MODULE.PROCESS_SCHEMA,
        "pid": pid,
        "starttime_ticks": starttime_ticks,
        "boot_id": boot_id,
        "executable": executable.as_dict(),
    }


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
        # AENV_HOME_PATH and AENV_RUNTIME_PATH are short /tmp aliases because
        # firecracker binds its API socket beneath them and sun_path is capped
        # at 108 bytes; every artefact still lives in the worktree.
        self.assertEqual(
            result["environment"]["AENV_HOME_PATH"],
            str(MODULE.short_state_link_path(state / "home", "home")),
        )
        self.assertEqual(
            result["environment"]["AENV_RUNTIME_PATH"],
            str(MODULE.short_state_link_path(state / "runtime", "run")),
        )
        self.assertEqual(result["environment"]["AENV_DEPS_PATH"], str(state / "deps"))
        self.assertTrue(result["safety"]["loopback_only"])
        self.assertFalse(result["safety"]["wsl_shutdown"])
        self.assertEqual(result["safety"]["signal_transport"], "pidfd-required")
        self.assertIn("executable_inode", result["safety"]["signal_identity"])

    def test_non_loopback_api_is_rejected(self) -> None:
        for value in ("0.0.0.0:18080", "192.168.1.1:18080", "localhost:18080"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(MODULE.ServiceError, "loopback|invalid"):
                    MODULE.validate_api_addr(value)

    def test_plan_and_status_create_no_state_and_no_short_link(self) -> None:
        """Rendering the environment must not publish anything.

        The short ``/tmp`` alias is a global, per-uid name.  If deriving it also
        created it, then ``plan``, ``status`` and ``start --dry-run`` would
        silently repoint the alias of a *running* server at a different
        worktree's state root.
        """

        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            links = [
                MODULE.short_state_link_path(paths.home, "home"),
                MODULE.short_state_link_path(paths.runtime, "run"),
            ]
            for link in links:
                self.assertFalse(link.exists() or link.is_symlink(), link)

            plan = MODULE.desired_plan(paths, "127.0.0.1:18080", "ubuntu:26.04")
            status = MODULE.service_status(
                paths, "127.0.0.1:18080", check_health=False
            )

            self.assertEqual(status["state"], "stopped")
            self.assertEqual(plan["operation"], "plan")
            self.assertFalse(paths.root.exists())
            for link in links:
                self.assertFalse(link.exists() or link.is_symlink(), link)

    def test_short_state_link_leaves_room_for_the_firecracker_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = MODULE.service_paths(fixture_repo(temporary))
            for target, tag in ((paths.home, "home"), (paths.runtime, "run")):
                link = MODULE.short_state_link_path(target, tag)
                # firecracker appends its own working directory and socket name
                # beneath this path before bind(); sun_path is 108 bytes.
                self.assertLessEqual(
                    len(os.fsencode(link))
                    + MODULE.FIRECRACKER_SOCKET_SUFFIX_BUDGET,
                    MODULE.SUN_PATH_MAX,
                )
                self.assertFalse(link.exists() or link.is_symlink(), link)

    def test_short_state_link_rejects_a_path_over_the_socket_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = MODULE.service_paths(fixture_repo(temporary))
            with mock.patch.object(
                MODULE, "FIRECRACKER_SOCKET_SUFFIX_BUDGET", MODULE.SUN_PATH_MAX
            ):
                with self.assertRaisesRegex(MODULE.ServiceError, "socket budget"):
                    MODULE.short_state_link_path(paths.home, "home")

    def test_publish_short_state_link_is_idempotent_and_repoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = MODULE.service_paths(fixture_repo(temporary))
            link = MODULE.short_state_link_path(paths.home, "home")
            self.addCleanup(lambda: link.unlink(missing_ok=True))
            other = Path(temporary) / "other"
            other.mkdir()
            link.symlink_to(other, target_is_directory=True)

            self.assertEqual(MODULE.publish_short_state_link(paths.home, "home"), link)
            self.assertEqual(link.readlink(), paths.home)
            self.assertTrue(paths.home.is_dir())
            # A second publish of the same target is a no-op.
            self.assertEqual(MODULE.publish_short_state_link(paths.home, "home"), link)
            self.assertEqual(link.readlink(), paths.home)

    def test_publish_short_state_link_refuses_a_non_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = MODULE.service_paths(fixture_repo(temporary))
            link = MODULE.short_state_link_path(paths.home, "home")
            self.addCleanup(lambda: link.unlink(missing_ok=True))
            link.write_bytes(b"")

            with self.assertRaisesRegex(MODULE.ServiceError, "not a symlink"):
                MODULE.publish_short_state_link(paths.home, "home")

    def test_prepare_generates_private_config_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            hashes = MODULE.prepare_layout(paths, "127.0.0.1:18080", "ubuntu:26.04")
            rendered = paths.config.read_text(encoding="utf-8")
            environment = json.loads(paths.environment_json.read_text(encoding="ascii"))

            self.assertIn('default_image = "ubuntu:26.04"', rendered)
            self.assertNotIn('default_image = "ubuntu:24.04"', rendered)
            self.assertIn("low_watermark = 0", rendered)
            self.assertIn("high_watermark = 0", rendered)
            self.assertNotIn("high_watermark = 64", rendered)
            self.assertEqual(environment["API_ADDR"], "127.0.0.1:18080")
            self.assertEqual(environment["E2B_API_URL"], "http://127.0.0.1:18080")
            # The short alias is published by prepare_layout and must resolve
            # back into the feature-local state root.
            for variable, target in (
                ("AENV_HOME_PATH", paths.home),
                ("AENV_RUNTIME_PATH", paths.runtime),
            ):
                link = Path(environment[variable])
                self.assertTrue(link.is_symlink(), variable)
                self.assertEqual(link.readlink(), target, variable)
                self.assertTrue(
                    str(link.resolve()).startswith(str(paths.root.resolve())), variable
                )
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
                "ready": True,
                "blockers": [],
                "devices": {},
                "pidfd": {"ready": True},
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
                "pidfd": {"ready": True},
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
                "pidfd": {"ready": True},
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

    def test_start_records_stored_launch_identity_without_proc_exe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            fixture_process_record(paths)
            process = mock.Mock(pid=4242)
            process.poll.return_value = None
            identity = MODULE.ProcessIdentity(4242, 99, "test-boot-id")
            prerequisites = {
                "ready": True,
                "blockers": [],
                "devices": {},
                "pidfd": {"ready": True},
            }
            descriptor = os.open("/dev/null", os.O_RDONLY)
            pidfd = MODULE.PidfdHandle(descriptor, mock.Mock())
            with mock.patch.object(
                MODULE, "prerequisite_report", return_value=prerequisites
            ), mock.patch.object(
                MODULE.subprocess, "Popen", return_value=process
            ), mock.patch.object(
                MODULE, "process_identity", return_value=identity
            ), mock.patch.object(
                MODULE, "_open_pidfd", return_value=pidfd
            ), mock.patch.object(
                MODULE.os,
                "readlink",
                side_effect=AssertionError("/proc/PID/exe must not be read"),
            ):
                result = MODULE.start_service(
                    paths,
                    "127.0.0.1:18080",
                    "ubuntu:26.04",
                    dry_run=False,
                )

            record = json.loads(paths.pidfile.read_text(encoding="ascii"))

        self.assertEqual(result["result"], "started")
        self.assertEqual(record["schema"], MODULE.PROCESS_SCHEMA)
        self.assertEqual(record["boot_id"], "test-boot-id")
        self.assertEqual(record["starttime_ticks"], 99)
        self.assertEqual(record["executable"]["path"], str(paths.binary))
        self.assertGreater(record["executable"]["inode"], 0)

    def test_start_verification_failure_kills_and_reaps_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            fixture_process_record(paths)
            process = mock.Mock(pid=4242)
            process.poll.return_value = None
            process.wait.return_value = -MODULE.signal.SIGKILL
            send_signal = mock.Mock()
            descriptor = os.open("/dev/null", os.O_RDONLY)
            pidfd = MODULE.PidfdHandle(descriptor, send_signal)
            prerequisites = {
                "ready": True,
                "blockers": [],
                "devices": {},
                "pidfd": {"ready": True},
            }
            with mock.patch.object(
                MODULE, "prerequisite_report", return_value=prerequisites
            ), mock.patch.object(
                MODULE.subprocess, "Popen", return_value=process
            ), mock.patch.object(
                MODULE, "_open_pidfd", return_value=pidfd
            ), mock.patch.object(
                MODULE,
                "process_identity",
                side_effect=MODULE.ServiceError("uninspectable child"),
            ):
                with self.assertRaisesRegex(MODULE.ServiceError, "uninspectable"):
                    MODULE.start_service(
                        paths,
                        "127.0.0.1:18080",
                        "ubuntu:26.04",
                        dry_run=False,
                    )

            self.assertFalse(paths.pidfile.exists())
            send_signal.assert_called_once_with(
                descriptor, MODULE.signal.SIGKILL, None, 0
            )
            process.wait.assert_called_once_with(timeout=5.0)

    def test_process_record_write_failure_kills_and_reaps_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            fixture_process_record(paths)
            process = mock.Mock(pid=4242)
            process.poll.return_value = None
            process.wait.return_value = -MODULE.signal.SIGKILL
            identity = MODULE.ProcessIdentity(4242, 99, "test-boot-id")
            send_signal = mock.Mock()
            descriptor = os.open("/dev/null", os.O_RDONLY)
            pidfd = MODULE.PidfdHandle(descriptor, send_signal)
            prerequisites = {
                "ready": True,
                "blockers": [],
                "devices": {},
                "pidfd": {"ready": True},
            }
            real_atomic_write = MODULE.atomic_write

            def fail_process_record(
                path: Path, payload: bytes, *, mode: int = 0o600
            ) -> None:
                if path == paths.pidfile:
                    raise OSError("simulated process-record write failure")
                real_atomic_write(path, payload, mode=mode)

            with mock.patch.object(
                MODULE, "prerequisite_report", return_value=prerequisites
            ), mock.patch.object(
                MODULE.subprocess, "Popen", return_value=process
            ), mock.patch.object(
                MODULE, "_open_pidfd", return_value=pidfd
            ), mock.patch.object(
                MODULE, "process_identity", return_value=identity
            ), mock.patch.object(
                MODULE, "atomic_write", side_effect=fail_process_record
            ):
                with self.assertRaisesRegex(MODULE.ServiceError, "record write failure"):
                    MODULE.start_service(
                        paths,
                        "127.0.0.1:18080",
                        "ubuntu:26.04",
                        dry_run=False,
                    )

            self.assertFalse(paths.pidfile.exists())
            send_signal.assert_called_once_with(
                descriptor, MODULE.signal.SIGKILL, None, 0
            )
            process.wait.assert_called_once_with(timeout=5.0)

    def test_start_pidfd_requirement_cannot_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            prerequisites = {
                "ready": False,
                "blockers": ["pidfd_api"],
                "devices": {},
                "pidfd": {"ready": False},
            }
            with mock.patch.object(
                MODULE, "prerequisite_report", return_value=prerequisites
            ), mock.patch.object(
                MODULE.subprocess,
                "Popen",
                side_effect=AssertionError("pidfd gate must run before spawn"),
            ):
                with self.assertRaisesRegex(MODULE.ServiceError, "pidfd"):
                    MODULE.start_service(
                        paths,
                        "127.0.0.1:18080",
                        "ubuntu:26.04",
                        dry_run=False,
                        allow_missing_prereqs=True,
                    )

            self.assertFalse(paths.root.exists())

    def test_prerequisite_report_includes_runtime_host_contract(self) -> None:
        device_reports = {
            "kvm": {"ready": True},
            "ublk_control": {"ready": True},
            "tun": {"ready": False},
        }
        capability_report = {
            "ready": False,
            "required": {
                "cap_net_admin": {"ready": True},
                "cap_sys_admin": {"ready": False},
            },
        }

        def device_report(path: Path) -> dict[str, bool]:
            name = next(
                name for name, candidate in MODULE.DEVICE_PATHS if candidate == path
            )
            return device_reports[name]

        with mock.patch.object(
            MODULE, "_device_report", side_effect=device_report
        ), mock.patch.object(
            MODULE, "_ip_forward_report", return_value={"ready": False, "value": "0"}
        ), mock.patch.object(
            MODULE, "_capability_report", return_value=capability_report
        ), mock.patch.object(
            MODULE, "_overlaybd_config_report", return_value={"ready": False}
        ), mock.patch.object(
            MODULE, "_pidfd_report", return_value={"ready": True}
        ):
            result = MODULE.prerequisite_report()

        self.assertFalse(result["ready"])
        self.assertEqual(
            result["blockers"],
            ["tun", "ip_forward", "cap_sys_admin", "overlaybd_config"],
        )
        self.assertIn("tun", result["devices"])
        self.assertEqual(result["ip_forward"]["value"], "0")
        self.assertIn("cap_sys_admin", result["capabilities"]["required"])
        self.assertIn("overlaybd_config", result)
        self.assertTrue(result["pidfd"]["ready"])

    def test_capability_report_matches_non_root_delegation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status = Path(temporary) / "status"
            status.write_text(
                "CapInh:\t0000000000201000\n"
                "CapPrm:\t0000000000201000\n"
                "CapEff:\t0000000000201000\n"
                "CapAmb:\t0000000000001000\n",
                encoding="ascii",
            )
            with mock.patch.object(MODULE.os, "geteuid", return_value=1000):
                result = MODULE._capability_report(status)

        self.assertEqual(
            result["required_sets"],
            ["inheritable", "permitted", "effective", "ambient"],
        )
        self.assertTrue(result["required"]["cap_net_admin"]["ready"])
        self.assertFalse(result["required"]["cap_sys_admin"]["ready"])
        self.assertFalse(result["ready"])

    def test_process_identity_does_not_read_proc_exe(self) -> None:
        stat_fields = ["S", *(["0"] * 18), "99"]
        with mock.patch.object(
            MODULE.Path,
            "read_text",
            return_value=f"4242 (server) {' '.join(stat_fields)}",
        ), mock.patch.object(
            MODULE, "boot_identity", return_value="test-boot-id"
        ), mock.patch.object(
            MODULE.os,
            "readlink",
            side_effect=AssertionError("/proc/PID/exe must not be read"),
        ):
            identity = MODULE.process_identity(4242)

        self.assertEqual(
            identity, MODULE.ProcessIdentity(4242, 99, "test-boot-id")
        )

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
            record = fixture_process_record(paths)
            paths.pidfile.write_bytes(MODULE.canonical_json(record))
            identity = MODULE.ProcessIdentity(4242, 99, "test-boot-id")
            with mock.patch.object(
                MODULE, "process_identity", return_value=identity
            ), mock.patch.object(
                MODULE.os,
                "kill",
                side_effect=AssertionError("confirmation required"),
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
            record = fixture_process_record(paths)
            paths.pidfile.write_bytes(MODULE.canonical_json(record))
            reused = MODULE.ProcessIdentity(4242, 100, "test-boot-id")
            with mock.patch.object(
                MODULE, "process_identity", return_value=reused
            ), mock.patch.object(
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

    def test_pidfd_signal_does_not_use_os_kill_and_closes_handle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            paths.root.mkdir(parents=True)
            record = fixture_process_record(paths)
            paths.pidfile.write_bytes(MODULE.canonical_json(record))
            identity = MODULE.ProcessIdentity(4242, 99, "test-boot-id")
            open_pidfd = mock.Mock(return_value=73)
            send_signal = mock.Mock()
            close = mock.Mock()
            with mock.patch.object(
                MODULE, "process_identity", return_value=identity
            ), mock.patch.object(
                MODULE, "_pidfd_api", return_value=(open_pidfd, send_signal)
            ), mock.patch.object(
                MODULE, "_wait_for_pidfd_exit", return_value=True
            ), mock.patch.object(
                MODULE.os, "kill", side_effect=AssertionError("pidfd must be used")
            ), mock.patch.object(MODULE.os, "close", close):
                result = MODULE.stop_service(
                    paths,
                    "127.0.0.1:18080",
                    dry_run=False,
                    confirm=True,
                    timeout=1,
                    kill_after_timeout=False,
                )

        self.assertEqual(result["result"], "stopped-after-sigterm")
        self.assertEqual(result["signal_method"], "pidfd")
        open_pidfd.assert_called_once_with(4242, 0)
        send_signal.assert_called_once_with(73, MODULE.signal.SIGTERM, None, 0)
        close.assert_called_once_with(73)

    def test_pidfd_term_and_kill_reuse_the_same_handle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            paths.root.mkdir(parents=True)
            record = fixture_process_record(paths)
            paths.pidfile.write_bytes(MODULE.canonical_json(record))
            identity = MODULE.ProcessIdentity(4242, 99, "test-boot-id")
            open_pidfd = mock.Mock(return_value=73)
            send_signal = mock.Mock()
            with mock.patch.object(
                MODULE, "process_identity", return_value=identity
            ), mock.patch.object(
                MODULE, "_pidfd_api", return_value=(open_pidfd, send_signal)
            ), mock.patch.object(
                MODULE, "_wait_for_pidfd_exit", side_effect=[False, True]
            ), mock.patch.object(MODULE.os, "kill") as kill, mock.patch.object(
                MODULE.os, "close"
            ):
                result = MODULE.stop_service(
                    paths,
                    "127.0.0.1:18080",
                    dry_run=False,
                    confirm=True,
                    timeout=0,
                    kill_after_timeout=True,
                )

        self.assertEqual(result["result"], "stopped-after-sigkill")
        self.assertEqual(
            send_signal.call_args_list,
            [
                mock.call(73, MODULE.signal.SIGTERM, None, 0),
                mock.call(73, MODULE.signal.SIGKILL, None, 0),
            ],
        )
        kill.assert_not_called()

    def test_pid_reuse_after_pidfd_open_is_never_signalled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            paths.root.mkdir(parents=True)
            record = fixture_process_record(paths)
            paths.pidfile.write_bytes(MODULE.canonical_json(record))
            matching = MODULE.ProcessIdentity(4242, 99, "test-boot-id")
            reused = MODULE.ProcessIdentity(4242, 100, "test-boot-id")
            open_pidfd = mock.Mock(return_value=73)
            send_signal = mock.Mock()
            with mock.patch.object(
                MODULE,
                "process_identity",
                side_effect=[matching, matching, reused],
            ), mock.patch.object(
                MODULE, "_pidfd_api", return_value=(open_pidfd, send_signal)
            ), mock.patch.object(MODULE.os, "kill") as kill, mock.patch.object(
                MODULE.os, "close"
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
        send_signal.assert_not_called()
        kill.assert_not_called()

    def test_missing_pidfd_refuses_signal_and_retains_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            paths.root.mkdir(parents=True)
            record = fixture_process_record(paths)
            paths.pidfile.write_bytes(MODULE.canonical_json(record))
            identity = MODULE.ProcessIdentity(4242, 99, "test-boot-id")
            with mock.patch.object(
                MODULE, "process_identity", return_value=identity
            ), mock.patch.object(
                MODULE, "_pidfd_api", return_value=None
            ), mock.patch.object(MODULE.os, "kill") as kill:
                with self.assertRaisesRegex(MODULE.ServiceError, "pidfd"):
                    MODULE.stop_service(
                        paths,
                        "127.0.0.1:18080",
                        dry_run=False,
                        confirm=True,
                        timeout=1,
                        kill_after_timeout=False,
                    )

            self.assertTrue(paths.pidfile.exists())
            kill.assert_not_called()

    def test_uninspectable_process_retains_record_without_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            paths.root.mkdir(parents=True)
            record = fixture_process_record(paths)
            paths.pidfile.write_bytes(MODULE.canonical_json(record))
            with mock.patch.object(
                MODULE,
                "process_identity",
                side_effect=MODULE.ServiceError("cannot inspect Linux start time"),
            ), mock.patch.object(MODULE.os, "kill") as kill:
                with self.assertRaisesRegex(MODULE.ServiceError, "cannot inspect"):
                    MODULE.stop_service(
                        paths,
                        "127.0.0.1:18080",
                        dry_run=False,
                        confirm=True,
                        timeout=0,
                        kill_after_timeout=False,
                    )

            self.assertTrue(paths.pidfile.exists())
            kill.assert_not_called()

    def test_pidfd_signal_error_retains_record_and_closes_handle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            paths.root.mkdir(parents=True)
            record = fixture_process_record(paths)
            paths.pidfile.write_bytes(MODULE.canonical_json(record))
            identity = MODULE.ProcessIdentity(4242, 99, "test-boot-id")
            open_pidfd = mock.Mock(return_value=73)
            send_signal = mock.Mock(side_effect=PermissionError("denied"))
            close = mock.Mock()
            with mock.patch.object(
                MODULE, "process_identity", return_value=identity
            ), mock.patch.object(
                MODULE, "_pidfd_api", return_value=(open_pidfd, send_signal)
            ), mock.patch.object(MODULE.os, "kill") as kill, mock.patch.object(
                MODULE.os, "close", close
            ):
                with self.assertRaisesRegex(MODULE.ServiceError, "record retained"):
                    MODULE.stop_service(
                        paths,
                        "127.0.0.1:18080",
                        dry_run=False,
                        confirm=True,
                        timeout=0,
                        kill_after_timeout=False,
                    )

            self.assertTrue(paths.pidfile.exists())
            send_signal.assert_called_once_with(73, MODULE.signal.SIGTERM, None, 0)
            close.assert_called_once_with(73)
            kill.assert_not_called()

    def test_binary_replacement_retains_record_without_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            paths.root.mkdir(parents=True)
            record = fixture_process_record(paths)
            paths.pidfile.write_bytes(MODULE.canonical_json(record))
            replacement = paths.binary.with_name("replacement-server")
            replacement.write_bytes(b"#!/bin/sh\nexit 1\n")
            replacement.chmod(0o700)
            replacement.replace(paths.binary)
            with mock.patch.object(MODULE.os, "kill") as kill:
                with self.assertRaisesRegex(MODULE.ServiceError, "launch identity"):
                    MODULE.stop_service(
                        paths,
                        "127.0.0.1:18080",
                        dry_run=False,
                        confirm=True,
                        timeout=0,
                        kill_after_timeout=False,
                    )

            self.assertTrue(paths.pidfile.exists())
            kill.assert_not_called()

    def test_legacy_process_record_is_retained_without_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = fixture_repo(temporary)
            paths = MODULE.service_paths(repo)
            paths.root.mkdir(parents=True)
            paths.pidfile.write_bytes(
                MODULE.canonical_json(
                    {
                        "schema": "amdgpu-sim.agentenv-service-process.v1",
                        "pid": 4242,
                        "starttime_ticks": 99,
                        "executable": "/tmp/server",
                    }
                )
            )
            with mock.patch.object(MODULE.os, "kill") as kill:
                with self.assertRaisesRegex(MODULE.ServiceError, "unsupported"):
                    MODULE.stop_service(
                        paths,
                        "127.0.0.1:18080",
                        dry_run=False,
                        confirm=True,
                        timeout=0,
                        kill_after_timeout=False,
                    )

            self.assertTrue(paths.pidfile.exists())
            kill.assert_not_called()

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
