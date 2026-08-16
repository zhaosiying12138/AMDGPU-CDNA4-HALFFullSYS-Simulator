# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused process-isolation tests for run_gemsim_instances.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts/run_gemsim_instances.py"
WORKER = ROOT / "tests/fixtures/gemsim_instance_worker.py"


class GemsimInstanceSupervisorTest(unittest.TestCase):
    maxDiff = None

    def invoke(
        self,
        temporary: Path,
        *,
        count: int,
        worker_mode: str,
        extra: list[str] | None = None,
        timeout: float = 15.0,
    ) -> tuple[subprocess.CompletedProcess[str], dict]:
        output = temporary / "supervisor-output"
        environment = dict(os.environ)
        environment.update(
            {
                "SAGR_GENERIC_BRIDGE_ENDPOINT": str(temporary / "stale.sock"),
                "SAGR_MANAGED_GEM5": "/stale/gem5.opt",
                "SAGR_TEST_SENTINEL": "must-not-leak",
                "CUDA_VISIBLE_DEVICES": "production-cuda",
                "HIP_VISIBLE_DEVICES": "production-rocm",
            }
        )
        command = [
            sys.executable,
            str(SUPERVISOR),
            "--instances",
            str(count),
            "--output-dir",
            str(output),
            *(extra or []),
            "--",
            sys.executable,
            str(WORKER),
            worker_mode,
        ]
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            timeout=timeout,
            check=False,
        )
        manifest_path = output / "manifest.json"
        self.assertTrue(manifest_path.is_file(), result.stderr)
        return result, json.loads(manifest_path.read_text(encoding="utf-8"))

    def assert_no_runtime_processes(self, manifest: dict) -> None:
        for run in manifest["runs"]:
            for runtime in run["runtime_processes"]:
                self.assertFalse(
                    Path(f"/proc/{runtime['pid']}").exists(),
                    f"runtime process {runtime['pid']} survived supervisor cleanup",
                )

    def test_two_and_four_instances_have_unique_namespaces(self) -> None:
        for count in (2, 4):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temporary:
                result, manifest = self.invoke(
                    Path(temporary), count=count, worker_mode="success"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(manifest["passed"])
                self.assertEqual(len(manifest["runs"]), count)
                self.assertEqual(
                    manifest["environment_boundary"]["removed_sagr_variables"],
                    [
                        "SAGR_GENERIC_BRIDGE_ENDPOINT",
                        "SAGR_MANAGED_GEM5",
                        "SAGR_TEST_SENTINEL",
                    ],
                )
                caches = {run["cache_dir"] for run in manifest["runs"]}
                self.assertEqual(len(caches), count)
                fields = (
                    "endpoint",
                    "run_dir",
                    "output_dir",
                    "trace_path",
                    "stats_path",
                    "gem5_cache_dir",
                    "daemon_uuid",
                    "job_uuid",
                    "epoch",
                )
                runtimes = [run["runtime_processes"][0] for run in manifest["runs"]]
                for field in fields:
                    self.assertEqual(
                        len({runtime[field] for runtime in runtimes}), count, field
                    )
                self.assertTrue(
                    all(
                        (runtime["rank"], runtime["world_size"]) == (0, 1)
                        for runtime in runtimes
                    )
                )
                self.assertTrue(
                    all(run["process_audit"]["samples"] > 0 for run in manifest["runs"])
                )
                self.assertTrue(
                    all(
                        run["process_audit"]["sagr_environment_leaks"] == []
                        and run["process_audit"]["production_device_fds"] == []
                        and run["process_audit"]["errors"] == []
                        for run in manifest["runs"]
                    )
                )
                self.assert_no_runtime_processes(manifest)

    def test_one_failure_does_not_cancel_other_and_leaves_no_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            result, manifest = self.invoke(
                temporary_path, count=2, worker_mode="fail-one"
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            runs = sorted(manifest["runs"], key=lambda run: run["index"])
            self.assertEqual(runs[0]["returncode"], 9)
            self.assertEqual(runs[1]["returncode"], 0)
            self.assertGreater(runs[1]["elapsed_seconds"], runs[0]["elapsed_seconds"])
            self.assertTrue(
                (Path(runs[1]["directory"]) / "completed").is_file(),
                "surviving worker was cancelled after its peer failed",
            )
            self.assert_no_runtime_processes(manifest)

    def test_timeout_cleans_runner_and_separate_gem5_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, manifest = self.invoke(
                Path(temporary),
                count=2,
                worker_mode="hold",
                extra=["--timeout-seconds", "0.35"],
            )
            self.assertEqual(result.returncode, 124, result.stderr)
            self.assertTrue(all(run["timed_out"] for run in manifest["runs"]))
            self.assert_no_runtime_processes(manifest)

    def test_stale_ready_epoch_fails_identity_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, manifest = self.invoke(
                Path(temporary), count=2, worker_mode="stale-epoch"
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse(manifest["passed"])
            for run in manifest["runs"]:
                self.assertEqual(run["returncode"], 0)
                self.assertIn(
                    "gem5 ready identity does not match configured job/epoch/rank/world",
                    run["isolation_errors"],
                )
                runtime = run["runtime_processes"][0]
                self.assertEqual(
                    runtime["observed_identity"]["epoch"], runtime["epoch"] + 1
                )
            self.assert_no_runtime_processes(manifest)

    def test_shared_cache_requires_explicit_read_only_prewarm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            cache = temporary_path / "prewarmed"
            cache.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(SUPERVISOR),
                    "-n",
                    "2",
                    "--output-dir",
                    str(temporary_path / "rejected"),
                    "--shared-readonly-cache",
                    str(cache),
                    "--",
                    sys.executable,
                    str(WORKER),
                    "success",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("prewarmed and recursively read-only", result.stderr)

            cache.chmod(0o555)
            accepted_root = temporary_path / "accepted-case"
            accepted_root.mkdir()
            accepted_cache = accepted_root / "prewarmed"
            accepted_cache.mkdir(mode=0o555)
            result, manifest = self.invoke(
                accepted_root,
                count=2,
                worker_mode="success",
                extra=["--shared-readonly-cache", str(accepted_cache)],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                {run["cache_dir"] for run in manifest["runs"]},
                {str(accepted_cache.resolve())},
            )
            self.assertEqual(
                manifest["cache_policy"]["mode"], "shared-read-only-prewarmed"
            )
            self.assertFalse(manifest["cache_policy"]["concurrent_writes_allowed"])
            self.assertTrue(manifest["cache_policy"]["exact_workload_must_be_prewarmed"])
            self.assertTrue(manifest["cache_policy"]["tree_unchanged"])
            self.assertEqual(
                manifest["cache_policy"]["pre_run_tree_sha256"],
                manifest["cache_policy"]["post_run_tree_sha256"],
            )

    def test_shared_cache_mutation_fails_post_run_identity_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            cache = temporary_path / "prewarmed"
            cache.mkdir(mode=0o555)
            result, manifest = self.invoke(
                temporary_path,
                count=2,
                worker_mode="mutate-cache",
                extra=["--shared-readonly-cache", str(cache)],
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse(manifest["passed"])
            self.assertFalse(manifest["cache_policy"]["tree_unchanged"])
            self.assertIn("writable", manifest["cache_policy"]["verification_error"])

    def test_output_root_must_not_preexist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "already-owned"
            output.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(SUPERVISOR),
                    "-n",
                    "2",
                    "--output-dir",
                    str(output),
                    "--",
                    sys.executable,
                    str(WORKER),
                    "success",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("must not already exist", result.stderr)

    @unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Linux affinity is required")
    def test_affinity_mode_is_calibration_only(self) -> None:
        allowed = sorted(os.sched_getaffinity(0))
        if len(allowed) < 2:
            self.skipTest("two allowed CPUs are required")
        with tempfile.TemporaryDirectory() as temporary:
            result, manifest = self.invoke(
                Path(temporary),
                count=2,
                worker_mode="success",
                extra=[
                    "--mode",
                    "calibrate",
                    "--cpu-set",
                    str(allowed[0]),
                    "--cpu-set",
                    str(allowed[1]),
                ],
                timeout=30.0,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("interference calibration only", manifest["calibration"]["claim_scope"])
            self.assertNotIn("TPOT", manifest["calibration"])
            phases = [run["phase"] for run in manifest["runs"]]
            self.assertEqual(phases.count("warmup"), 2)
            self.assertEqual(phases.count("serial-baseline"), 2)
            self.assertEqual(phases.count("parallel"), 2)
            self.assert_no_runtime_processes(manifest)


if __name__ == "__main__":
    unittest.main()
