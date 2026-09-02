# SPDX-License-Identifier: GPL-3.0-or-later
"""Host-only contract tests for the world=4 RCCL capsule."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / "tools/qwen35_nccl_tp4_capsule.py"
SPEC = importlib.util.spec_from_file_location("qwen35_nccl_tp4_capsule", CAPSULE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeProcess:
    def __init__(self, pid: int, returncode: int | None = None) -> None:
        self.pid = pid
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired(["fake-worker"], timeout)
        return self.returncode


class Qwen35NcclTp4CapsuleTest(unittest.TestCase):
    def test_collective_contract_is_exactly_two_tiny_then_embedding(self) -> None:
        self.assertEqual(MODULE.WORLD_SIZE, 4)
        self.assertEqual(
            [operation.name for operation in MODULE.OPERATIONS],
            [
                "tiny_scalar",
                "tiny_vector",
                "qwen35_9b_prefill_embedding",
            ],
        )
        self.assertEqual(MODULE.OPERATIONS[0].shape, (1,))
        self.assertEqual(MODULE.OPERATIONS[1].shape, (16,))
        production = MODULE.OPERATIONS[2]
        self.assertEqual(production.shape, (2, 4096))
        self.assertEqual(production.dtype, "bfloat16")
        self.assertEqual(MODULE.operation_contract()[2]["bytes"], 16_384)
        self.assertNotIn("reduce_scatter", CAPSULE.read_text(encoding="ascii"))

    def test_every_rank_has_a_distinct_exact_integer_oracle(self) -> None:
        for operation in MODULE.OPERATIONS:
            with self.subTest(operation=operation.name):
                inputs = [
                    MODULE.input_values(operation, rank)
                    for rank in range(MODULE.WORLD_SIZE)
                ]
                self.assertEqual(len({tuple(values) for values in inputs}), 4)
                observed_sum = [sum(values) for values in zip(*inputs, strict=True)]
                self.assertEqual(observed_sum, MODULE.expected_values(operation))
                # All values and sums stay in the exact-integer range of BF16.
                self.assertLessEqual(
                    max(abs(value) for values in inputs for value in values), 256
                )
                self.assertLessEqual(max(abs(value) for value in observed_sum), 256)

    def test_worker_environments_cover_ranks_zero_through_three(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output = Path(temporary) / "out"
            environments = [
                MODULE.worker_environment(
                    {"PRESERVED": "yes"},
                    rank=rank,
                    run_id="run-1",
                    output=output,
                    address="127.0.0.1",
                    port=23456,
                )
                for rank in range(MODULE.WORLD_SIZE)
            ]
        self.assertEqual(
            [environment[MODULE.WORKER_RANK_ENV] for environment in environments],
            ["0", "1", "2", "3"],
        )
        self.assertEqual(
            [environment["RANK"] for environment in environments],
            ["0", "1", "2", "3"],
        )
        self.assertTrue(all(env["WORLD_SIZE"] == "4" for env in environments))
        self.assertTrue(all(env["PRESERVED"] == "yes" for env in environments))
        self.assertTrue(
            all(
                env[MODULE.SUPERVISOR_PID_ENV] == str(os.getpid())
                for env in environments
            )
        )

    def test_transport_report_preserves_unset_and_explicit_knobs(self) -> None:
        value = MODULE.communication_environment(
            {
                "NCCL_SOCKET_IFNAME": "lo",
                "NCCL_SHM_DISABLE": "1",
                "NCCL_PROTO": "Simple",
            }
        )
        self.assertEqual(value["NCCL_SOCKET_IFNAME"], "lo")
        self.assertEqual(value["NCCL_SHM_DISABLE"], "1")
        self.assertEqual(value["NCCL_PROTO"], "Simple")
        self.assertIsNone(value["NCCL_ALGO"])

    def test_supervisor_signal_handler_raises_and_restores(self) -> None:
        previous_term = signal.getsignal(signal.SIGTERM)
        previous_hup = signal.getsignal(signal.SIGHUP)
        previous = MODULE.install_supervisor_signal_handlers()
        try:
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            with self.assertRaises(MODULE.SupervisorTermination) as caught:
                handler(signal.SIGTERM, None)
            self.assertEqual(caught.exception.signal_number, signal.SIGTERM)
            self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_IGN)
            self.assertEqual(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
        finally:
            MODULE.restore_signal_handlers(previous)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_term)
        self.assertEqual(signal.getsignal(signal.SIGHUP), previous_hup)

    def test_atomic_json_publish_is_canonical_and_leaves_no_temporary(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            report = root / "nested/report.json"
            value = {"schema": "test", "passed": False, "ranks": [0, 1, 2, 3]}
            MODULE.write_json_atomic(report, value)
            self.assertEqual(json.loads(report.read_text(encoding="ascii")), value)
            self.assertEqual(report.read_bytes(), MODULE.canonical_json(value))
            self.assertEqual(list(report.parent.glob(".report.json.tmp-*")), [])

    def test_rank_report_validation_fails_closed_on_stale_or_partial_data(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "rank-2.json"
            valid = {
                "schema": MODULE.RANK_REPORT_SCHEMA,
                "run_id": "current",
                "rank": 2,
                "status": "success",
                "passed": True,
                "world_size": 4,
                "process_group_initialized": True,
                "teardown_completed": True,
                "operations": [
                    {
                        "name": operation.name,
                        "ordinal": ordinal,
                        "shape": list(operation.shape),
                        "dtype": operation.dtype,
                        "passed": True,
                        "mismatch_count": 0,
                        "max_abs_error": 0.0,
                        "elapsed_seconds": 1.0,
                        "input_sha256": "1" * 64,
                        "expected_sha256": "2" * 64,
                        "observed_sha256": "2" * 64,
                    }
                    for ordinal, operation in enumerate(MODULE.OPERATIONS)
                ],
            }
            MODULE.write_json_atomic(path, valid)
            value, error = MODULE._load_rank_report(
                path, run_id="current", rank=2
            )
            self.assertEqual(value, valid)
            self.assertIsNone(error)

            stale = dict(valid, run_id="previous")
            MODULE.write_json_atomic(path, stale)
            _, error = MODULE._load_rank_report(path, run_id="current", rank=2)
            self.assertIn("stale", error)

            partial = dict(valid, status="operations_passed", passed=False)
            MODULE.write_json_atomic(path, partial)
            _, error = MODULE._load_rank_report(path, run_id="current", rank=2)
            self.assertIn("operations_passed", error)

            bad_digest = json.loads(json.dumps(valid))
            bad_digest["operations"][1]["observed_sha256"] = "3" * 64
            MODULE.write_json_atomic(path, bad_digest)
            _, error = MODULE._load_rank_report(path, run_id="current", rank=2)
            self.assertIn("digest mismatch", error)

    def test_timeouts_must_be_positive_and_finite(self) -> None:
        with mock.patch.dict(
            MODULE.os.environ, {MODULE.TIMEOUT_ENV: "0"}, clear=False
        ):
            with self.assertRaisesRegex(ValueError, "positive finite"):
                MODULE.parse_positive_seconds(MODULE.TIMEOUT_ENV, 600.0)

    def test_spawn_workers_creates_four_isolated_rank_processes(self) -> None:
        calls: list[tuple[list[str], dict[str, object]]] = []

        def factory(command: list[str], **kwargs: object) -> FakeProcess:
            calls.append((command, kwargs))
            return FakeProcess(10_000 + len(calls), returncode=0)

        runs = MODULE.spawn_workers(
            run_id="run-1",
            output=ROOT / "artifacts/test-only",
            address="127.0.0.1",
            port=23456,
            grace_seconds=0.01,
            process_factory=factory,
        )
        self.assertEqual([run.rank for run in runs], [0, 1, 2, 3])
        self.assertEqual(
            [call[1]["env"][MODULE.WORKER_RANK_ENV] for call in calls],
            ["0", "1", "2", "3"],
        )
        self.assertTrue(all(call[1]["close_fds"] is True for call in calls))
        self.assertTrue(
            all(call[1]["start_new_session"] is True for call in calls)
        )

    def test_partial_spawn_is_reaped_before_error_escapes(self) -> None:
        created: list[FakeProcess] = []

        def factory(_command: list[str], **_kwargs: object) -> FakeProcess:
            if len(created) == 2:
                raise OSError("injected spawn failure")
            process = FakeProcess(15_000 + len(created))
            created.append(process)
            return process

        with (
            mock.patch.object(MODULE, "_signal_live_workers") as signal_workers,
            mock.patch.object(MODULE, "_reap_workers") as reap_workers,
            self.assertRaisesRegex(OSError, "injected spawn failure"),
        ):
            MODULE.spawn_workers(
                run_id="run-1",
                output=ROOT / "artifacts/test-only",
                address="127.0.0.1",
                port=23456,
                grace_seconds=0.01,
                process_factory=factory,
            )
        signaled_runs, number = signal_workers.call_args.args
        self.assertEqual([run.rank for run in signaled_runs], [0, 1])
        self.assertEqual(number, signal.SIGTERM)
        reaped_runs, grace = reap_workers.call_args.args
        self.assertEqual([run.rank for run in reaped_runs], [0, 1])
        self.assertEqual(grace, 0.01)

    def test_deadline_reaps_all_owned_rank_groups(self) -> None:
        runs = [
            MODULE.RankProcess(rank, FakeProcess(20_000 + rank))
            for rank in range(MODULE.WORLD_SIZE)
        ]
        signals: list[int] = []

        def signal_workers(rank_runs: list[MODULE.RankProcess], number: int) -> None:
            signals.append(number)
            if number == signal.SIGKILL:
                for run in rank_runs:
                    if run.process.returncode is None:
                        run.process.returncode = -number

        with mock.patch.object(
            MODULE, "_signal_live_workers", side_effect=signal_workers
        ):
            timed_out, failure = MODULE.wait_for_workers(
                runs, timeout_seconds=0.001, grace_seconds=0.001
            )
        self.assertTrue(timed_out)
        self.assertIn("live ranks=[0, 1, 2, 3]", failure)
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertTrue(all(run.process.returncode is not None for run in runs))

    def test_early_rank_failure_reaps_still_running_peers(self) -> None:
        runs = [
            MODULE.RankProcess(0, FakeProcess(30_000, returncode=7)),
            MODULE.RankProcess(1, FakeProcess(30_001)),
            MODULE.RankProcess(2, FakeProcess(30_002)),
            MODULE.RankProcess(3, FakeProcess(30_003)),
        ]

        def signal_workers(rank_runs: list[MODULE.RankProcess], number: int) -> None:
            if number == signal.SIGKILL:
                for run in rank_runs:
                    if run.process.returncode is None:
                        run.process.returncode = -number

        with mock.patch.object(
            MODULE, "_signal_live_workers", side_effect=signal_workers
        ):
            timed_out, failure = MODULE.wait_for_workers(
                runs, timeout_seconds=1.0, grace_seconds=0.001
            )
        self.assertFalse(timed_out)
        self.assertEqual(failure, "rank 0 exited with status 7")
        self.assertTrue(all(run.process.returncode is not None for run in runs))

    def test_supervisor_timeout_still_publishes_fail_closed_report(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output = Path(temporary) / "report"
            created: list[FakeProcess] = []

            def factory(_command: list[str], **_kwargs: object) -> FakeProcess:
                process = FakeProcess(40_000 + len(created))
                created.append(process)
                return process

            def signal_workers(
                rank_runs: list[MODULE.RankProcess], number: int
            ) -> None:
                if number == signal.SIGKILL:
                    for run in rank_runs:
                        if run.process.returncode is None:
                            run.process.returncode = -number

            environment = {
                MODULE.OUTPUT_ENV: str(output),
                MODULE.TIMEOUT_ENV: "0.001",
                MODULE.TERM_GRACE_ENV: "0.001",
            }
            with (
                mock.patch.dict(MODULE.os.environ, environment, clear=False),
                mock.patch.object(
                    MODULE, "_signal_live_workers", side_effect=signal_workers
                ),
            ):
                code = MODULE.run_supervisor(
                    process_factory=factory, port_factory=lambda: 23456
                )

            self.assertEqual(code, 1)
            report = json.loads(
                (output / "report.json").read_text(encoding="ascii")
            )
            self.assertEqual(report["schema"], MODULE.REPORT_SCHEMA)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["passed"])
            self.assertTrue(report["timed_out"])
            self.assertIn("live ranks=[0, 1, 2, 3]", report["failure"])
            self.assertEqual(len(report["rank_processes"]), 4)
            self.assertTrue(
                all(item["validation_error"] for item in report["rank_processes"])
            )
            self.assertTrue(all(process.returncode is not None for process in created))

    @unittest.skipUnless(sys.platform == "linux", "requires Linux prctl")
    def test_parent_death_signal_kills_detached_worker(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            child_pid_path = root / "child.pid"
            child_ready_path = root / "child.ready"
            child_code = "\n".join(
                [
                    "import importlib.util, os, pathlib, sys, time",
                    f"path = {str(CAPSULE)!r}",
                    "spec = importlib.util.spec_from_file_location('tp4_child', path)",
                    "module = importlib.util.module_from_spec(spec)",
                    "sys.modules[spec.name] = module",
                    "spec.loader.exec_module(module)",
                    "module.arm_parent_death_signal(int(os.environ['EXPECTED_PPID']))",
                    f"pathlib.Path({str(child_ready_path)!r}).write_text('ready')",
                    "time.sleep(60)",
                ]
            )
            controller_code = "\n".join(
                [
                    "import os, pathlib, subprocess, sys, time",
                    "environment = dict(os.environ)",
                    "environment['EXPECTED_PPID'] = str(os.getpid())",
                    f"code = {child_code!r}",
                    "child = subprocess.Popen([sys.executable, '-c', code],",
                    "                         env=environment, start_new_session=True)",
                    f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))",
                    "time.sleep(60)",
                ]
            )
            controller = subprocess.Popen(
                [sys.executable, "-c", controller_code],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            child_pid: int | None = None
            try:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if child_pid_path.exists() and child_ready_path.exists():
                        child_pid = int(child_pid_path.read_text())
                        break
                    if controller.poll() is not None:
                        self.fail("parent-death controller exited before worker armed")
                    time.sleep(0.01)
                self.assertIsNotNone(child_pid)
                os.kill(controller.pid, signal.SIGKILL)
                controller.wait(timeout=5.0)
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and Path(
                    f"/proc/{child_pid}"
                ).exists():
                    time.sleep(0.01)
                self.assertFalse(Path(f"/proc/{child_pid}").exists())
            finally:
                if controller.poll() is None:
                    controller.kill()
                    controller.wait(timeout=5.0)
                if child_pid is not None and Path(f"/proc/{child_pid}").exists():
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    @unittest.skipUnless(sys.platform == "linux", "requires Linux process groups")
    def test_external_sigterm_reaps_rank_sessions_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output = Path(temporary) / "output"
            controller_code = "\n".join(
                [
                    "import importlib.util, subprocess, sys",
                    f"path = {str(CAPSULE)!r}",
                    "spec = importlib.util.spec_from_file_location(",
                    "    'tp4_supervisor', path)",
                    "module = importlib.util.module_from_spec(spec)",
                    "sys.modules[spec.name] = module",
                    "spec.loader.exec_module(module)",
                    "def factory(_command, **kwargs):",
                    "    return subprocess.Popen(",
                    "        [sys.executable, '-c', 'import time; time.sleep(60)'],",
                    "        **kwargs)",
                    "raise SystemExit(module.run_supervisor(",
                    "    process_factory=factory, port_factory=lambda: 23456))",
                ]
            )
            environment = dict(os.environ)
            environment.update(
                {
                    MODULE.OUTPUT_ENV: str(output),
                    MODULE.TIMEOUT_ENV: "60",
                    MODULE.TERM_GRACE_ENV: "0.1",
                }
            )
            controller = subprocess.Popen(
                [sys.executable, "-c", controller_code],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            child_pids: list[int] = []
            try:
                children_path = Path(
                    f"/proc/{controller.pid}/task/{controller.pid}/children"
                )
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if children_path.exists():
                        child_pids = [
                            int(value) for value in children_path.read_text().split()
                        ]
                    if len(child_pids) == MODULE.WORLD_SIZE:
                        break
                    if controller.poll() is not None:
                        self.fail("capsule supervisor exited before rank spawn")
                    time.sleep(0.01)
                self.assertEqual(len(child_pids), MODULE.WORLD_SIZE)
                os.kill(controller.pid, signal.SIGTERM)
                self.assertEqual(controller.wait(timeout=5.0), 1)
                report = json.loads(
                    (output / "report.json").read_text(encoding="ascii")
                )
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["passed"])
                self.assertEqual(report["termination_signal"], "SIGTERM")
                self.assertIn("received SIGTERM", report["failure"])
                self.assertEqual(len(report["rank_processes"]), MODULE.WORLD_SIZE)
                self.assertTrue(
                    all(
                        item["returncode"] is not None
                        for item in report["rank_processes"]
                    )
                )
                self.assertTrue(
                    all(not Path(f"/proc/{pid}").exists() for pid in child_pids)
                )
            finally:
                if controller.poll() is None:
                    controller.kill()
                    controller.wait(timeout=5.0)
                for pid in child_pids:
                    if Path(f"/proc/{pid}").exists():
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass


if __name__ == "__main__":
    unittest.main()
