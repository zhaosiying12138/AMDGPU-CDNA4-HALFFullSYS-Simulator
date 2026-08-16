# SPDX-License-Identifier: GPL-3.0-or-later
"""Host-only tests for the live allreduce rank runner and supervisor."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_gemsim_ccl_live_allreduce as RUNNER  # noqa: E402


DESIGN = ROOT / "tools/gemsim_ccl_live_allreduce.py"


def active_prefix() -> Path:
    value = subprocess.check_output(
        ["bash", str(ROOT / "scripts/setup_rocm_env.sh"), "--print-prefix"],
        cwd=ROOT,
        text=True,
    ).strip()
    return Path(value).resolve(strict=True)


class FakeProcess:
    def __init__(self, pid: int, returncode: int) -> None:
        self.pid = pid
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


class FakeBroker:
    def __init__(self, failure: SimpleNamespace | None) -> None:
        self.owner = SimpleNamespace(pid=41, start_time_ticks=73)
        self.failure = failure
        self.destroyed = False
        self.bound: list[tuple[int, int]] = []

    def prepare_rank(self, _rank: int) -> int:
        return os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)

    def bind_rank(self, rank: int, identity: SimpleNamespace) -> None:
        self.bound.append((rank, identity.pid))

    def rendezvous(self, _deadline: int) -> None:
        return None

    def progress(self) -> SimpleNamespace | None:
        return self.failure

    def abort(self, failed_rank: int, reason: int, sequence: int) -> SimpleNamespace:
        if self.failure is None:
            self.failure = SimpleNamespace(
                status=reason,
                reporter_rank=failed_rank,
                failed_rank=failed_rank,
                context_sequence=sequence,
            )
        return self.failure

    def first_error(self) -> SimpleNamespace | None:
        return self.failure

    def info(self) -> SimpleNamespace:
        return SimpleNamespace(abort_pending_mask=0)

    def destroy(self) -> None:
        self.destroyed = True


class FakeNative:
    def __init__(self, runtime_sha: str, broker: FakeBroker) -> None:
        self.library_sha256 = runtime_sha
        self.broker = broker
        self.now = 1_000_000_000

    def identity(self, **_kwargs: object) -> object:
        return object()

    def monotonic_time_ns(self) -> int:
        self.now += 1_000
        return self.now

    def live_broker(self, _identity: object) -> FakeBroker:
        return self.broker

    def process_identity(self, pid: int) -> SimpleNamespace:
        return SimpleNamespace(pid=pid, start_time_ticks=pid + 10_000)


class LiveAllreduceRunnerTest(unittest.TestCase):
    maxDiff = None

    def publish_expected(self, root: Path, execution_root: Path) -> Path:
        expected = root / "expected.json"
        runtime = active_prefix() / "lib/libself_amdgpu_runtime.so.1"
        subprocess.run(
            [
                sys.executable,
                str(DESIGN),
                "--design-only",
                "--runtime-library",
                str(runtime),
                "--namespace-root",
                str(execution_root),
                "--world-size",
                "2",
                "--element-count",
                "2",
                "--dtype",
                "float32",
                "--expected-output",
                str(expected),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return expected

    @staticmethod
    def transfer(descriptor_sha256: str, sequence: int = 1) -> dict[str, object]:
        return {
            "descriptor_sha256": descriptor_sha256,
            "sequence": sequence,
            "phase": 1,
            "step_index": 0,
            "chunk_index": 0,
            "source_rank": 1,
            "destination_rank": 0,
            "slot_index": 0,
            "slot_generation": 1,
        }

    def supervise_fake(self, root: Path, *, fail: bool) -> tuple[dict, Path]:
        output = root / "source"
        execution_root = root / "x"
        expected_path = self.publish_expected(root, execution_root)
        expected = json.loads(expected_path.read_text(encoding="ascii"))
        runtime = active_prefix() / "lib/libself_amdgpu_runtime.so.1"
        failure = (
            SimpleNamespace(
                status=RUNNER.PEER_LOST,
                reporter_rank=0,
                failed_rank=1,
                context_sequence=1,
            )
            if fail
            else None
        )
        broker = FakeBroker(failure)
        native = FakeNative(expected["design"]["runtime"]["sha256"], broker)
        identity_record = RUNNER.file_record(Path(__file__))
        source_identity = {
            role: dict(identity_record) for role in RUNNER.IDENTITY_ROLES
        }

        def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
            self.assertIs(kwargs["close_fds"], True)
            self.assertIs(kwargs["start_new_session"], True)
            self.assertEqual(len(kwargs["pass_fds"]), 1)
            config_path = Path(command[command.index("--config") + 1])
            config = json.loads(config_path.read_text(encoding="ascii"))
            rank = int(config["rank"])
            input_payload = RUNNER.deterministic_input("float32", rank, 2)
            input_sha = RUNNER.sha256_bytes(input_payload)
            descriptor = expected["design"]["ranks"][rank]["segments"][0][
                "descriptor_sha256"
            ]
            result = {
                "schema": "amdgpu-sim.ccl-live-allreduce-rank-result.v1",
                "status": "peer_lost" if fail and rank == 0 else "success",
                "rank": rank,
                "world_size": 2,
                "acceptance_authority": False,
                "live_collective_accepted": False,
                "public_result_published": not (fail and rank == 0),
                "public_commit_count": 0 if fail and rank == 0 else 1,
                "input_sha256_before": input_sha,
                "input_sha256_after": input_sha,
                "output_sha256": input_sha,
                "output_storage_fresh": True,
                "first_error": (
                    {
                        "status": "peer_lost",
                        "native_status": RUNNER.PEER_LOST,
                        "reporter_rank": 0,
                        "failed_rank": 1,
                        "context_sequence": 1,
                    }
                    if fail and rank == 0
                    else None
                ),
                "failed_transfer": (
                    self.transfer(descriptor) if fail and rank == 0 else None
                ),
                "failed_ack_sent": False,
            }
            rank_root = Path(config["result_path"]).parent
            RUNNER._exclusive_write(
                Path(config["result_path"]), RUNNER.canonical_json(result)
            )
            RUNNER._exclusive_write(Path(config["journal_path"]), b"")
            RUNNER._exclusive_write(Path(config["input_path"]), input_payload)
            RUNNER._exclusive_write(Path(config["output_path"]), input_payload)
            launch = expected["design"]["ranks"][rank]["rank_launch"]
            trace = Path(launch["paths"]["dispatch_trace_path"])
            log = Path(launch["paths"]["gem5_log_path"])
            stats = Path(launch["paths"]["gem5_output_directory"]) / "stats.txt"
            for path, payload in ((trace, b""), (log, b""), (stats, b"")):
                path.parent.mkdir(parents=True, exist_ok=True)
                RUNNER._exclusive_write(path, payload)
            return FakeProcess(10_000 + rank, 1 if fail and rank == 0 else 0)

        def fake_terminate(runs: list[RUNNER.RankProcess], *_args: object) -> bool:
            for run in runs:
                run.returncode = run.process.poll() if run.process else None
                if run.stdout is not None:
                    run.stdout.close()
                if run.stderr is not None:
                    run.stderr.close()
            return True

        paths = {role: Path(__file__) for role in RUNNER.IDENTITY_ROLES}
        paths["runtime_library"] = runtime
        with (
            mock.patch.object(RUNNER, "load_product", return_value=(
                {"product_id": "fake-product"}, paths
            )),
            mock.patch.object(RUNNER, "source_identity", return_value=source_identity),
            mock.patch.object(RUNNER, "enable_subreaper"),
            mock.patch.object(RUNNER, "owned_fd_snapshot", side_effect=[{}, {}]),
            mock.patch.object(RUNNER, "_direct_child_identities", return_value=set()),
            mock.patch.object(RUNNER, "_capture_owned_processes"),
            mock.patch.object(RUNNER, "_present", return_value=False),
            mock.patch.object(RUNNER, "terminate_group", side_effect=fake_terminate),
        ):
            manifest = RUNNER.supervise(
                expected_path=expected_path,
                output=output,
                execution_root=execution_root,
                product_prefix=active_prefix(),
                timeout_seconds=1.0,
                cleanup_grace_seconds=0.01,
                popen_factory=fake_popen,
                native_factory=lambda _path: native,
            )
        self.assertTrue(broker.destroyed)
        return manifest, output

    def test_fake_n2_supervisor_success_has_atomic_cleanup_evidence(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            manifest, output = self.supervise_fake(Path(temporary), fail=False)
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(manifest["public_commit_count"], 2)
            cleanup = manifest["supervisor_cleanup"]
            self.assertEqual(cleanup["baseline_fds"], [])
            self.assertEqual(cleanup["post_fds"], [])
            self.assertEqual(cleanup["measured_fd_delta"], 0)
            self.assertTrue(cleanup["all_clear"])
            self.assertEqual(
                {child["role"] for child in cleanup["new_child_identities"]},
                {"worker"},
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"result-manifest.json", "rank-00", "rank-01"},
            )
            self.assertFalse((Path(temporary) / "x").exists())

    def test_fake_n2_failure_is_group_wide_and_uses_reporter_tuple(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            manifest, output = self.supervise_fake(Path(temporary), fail=True)
            self.assertEqual(manifest["status"], "peer_lost")
            self.assertEqual(manifest["first_error"]["reporter_rank"], 0)
            self.assertEqual(manifest["failed_transfer"]["sequence"], 1)
            for rank in range(2):
                result = json.loads(
                    (output / f"rank-{rank:02d}/worker-result.json").read_text()
                )
                self.assertEqual(result["status"], "peer_lost")
                self.assertEqual(result["first_error"], manifest["first_error"])
                self.assertEqual(result["failed_transfer"], manifest["failed_transfer"])
                self.assertEqual(result["public_commit_count"], 0)
                self.assertFalse(result["public_result_published"])
                self.assertEqual(
                    (output / f"rank-{rank:02d}/output.bin").stat().st_size, 0
                )

    def test_failure_bundle_never_borrows_a_nonreporter_tuple(self) -> None:
        descriptor = "ab" * 32
        first = SimpleNamespace(
            status=RUNNER.PEER_LOST,
            reporter_rank=1,
            failed_rank=0,
            context_sequence=2,
        )
        results = [
            {"failed_transfer": self.transfer(descriptor, 2)},
            {"failed_transfer": None},
        ]
        _, _, failed, _ = RUNNER.canonical_failure_bundle(
            "device_failure", first, results
        )
        self.assertIsNone(failed)

    def test_stale_gem5_binding_fails_product_preflight(self) -> None:
        prefix = active_prefix()
        manifest = json.loads((prefix / "manifest.json").read_text())
        stale = copy.deepcopy(manifest)
        stale["managed_inputs"]["gem5_binary"]["sha256"] = "00" * 32
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fake_prefix = Path(temporary).resolve()
            stale["prefix"] = str(fake_prefix)
            source_package = (
                ROOT / "plugins/collectives/gemsim_ccl/src/gemsim_ccl"
            )
            stale["artifacts"]["ccl_plugin_init"]["path"] = str(
                source_package / "__init__.py"
            )
            stale["plugins"]["snapshots"]["gemsim-ccl"]["package_path"] = str(
                source_package
            )
            (fake_prefix / "manifest.json").write_bytes(RUNNER.canonical_json(stale))
            with self.assertRaisesRegex(RUNNER.RunnerError, "gem5_binary bytes"):
                RUNNER.load_product(fake_prefix)

    def test_product_without_reusable_engine_is_rejected(self) -> None:
        prefix = active_prefix()
        manifest = json.loads((prefix / "manifest.json").read_text())
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fake_prefix = Path(temporary).resolve()
            package = fake_prefix / "python/gemsim_ccl"
            package.mkdir(parents=True)
            for name in ("__init__.py", "native.py", "device.py"):
                shutil.copyfile(
                    ROOT / "plugins/collectives/gemsim_ccl/src/gemsim_ccl" / name,
                    package / name,
                )
            manifest["prefix"] = str(fake_prefix)
            manifest["artifacts"]["ccl_plugin_init"]["path"] = str(
                package / "__init__.py"
            )
            manifest["plugins"]["snapshots"]["gemsim-ccl"]["package_path"] = str(
                package
            )
            (fake_prefix / "manifest.json").write_bytes(
                RUNNER.canonical_json(manifest)
            )
            with self.assertRaisesRegex(
                RUNNER.RunnerError, "identity is missing: ccl_engine"
            ):
                RUNNER.load_product(fake_prefix)

    def test_malicious_launch_path_is_rejected_before_output_or_delete(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary).resolve()
            output = root / "source"
            execution_root = root / "x"
            expected_path = self.publish_expected(root, execution_root)
            expected = json.loads(expected_path.read_text())
            marker = root / "external-marker"
            marker.write_text("keep", encoding="ascii")
            launch = expected["design"]["ranks"][0]["rank_launch"]
            launch["paths"]["triton_cache_directory"] = str(marker.parent / "triton")
            expected["design"]["ranks"][0]["rank_launch_sha256"] = (
                RUNNER.object_sha256(launch)
            )
            expected_path.unlink()
            expected_path.write_bytes(RUNNER.canonical_json(expected))
            with self.assertRaises(RUNNER.RunnerError):
                RUNNER.supervise(
                    expected_path=expected_path,
                    output=output,
                    execution_root=execution_root,
                    product_prefix=active_prefix(),
                    timeout_seconds=1.0,
                    cleanup_grace_seconds=0.01,
                )
            self.assertEqual(marker.read_text(encoding="ascii"), "keep")
            self.assertFalse(output.exists())

    def test_fd_identity_set_detects_and_clears_a_leak(self) -> None:
        baseline = RUNNER.owned_fd_snapshot()
        read_fd, write_fd = os.pipe()
        try:
            leaked = RUNNER.owned_fd_snapshot()
            self.assertNotEqual(set(leaked.items()), set(baseline.items()))
            delta = set(leaked.items()) - set(baseline.items())
            self.assertGreaterEqual(len(delta), 2)
        finally:
            os.close(read_fd)
            os.close(write_fd)
        self.assertEqual(RUNNER.owned_fd_snapshot(), baseline)

    def test_terminate_group_reaps_descendant_but_not_baseline_child(self) -> None:
        sleeper = "import time; time.sleep(30)"
        baseline = subprocess.Popen(
            [sys.executable, "-c", sleeper], start_new_session=True
        )
        worker = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import subprocess,sys,time;"
                    "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
                    "time.sleep(30)"
                ),
            ],
            start_new_session=True,
        )
        try:
            baseline_identity = RUNNER._proc_identity(baseline.pid)
            self.assertIsNotNone(baseline_identity)
            run = RUNNER.RankProcess(
                rank=0,
                directory=ROOT,
                launch={},
                process=worker,
                start_time_ticks=RUNNER._proc_identity(worker.pid)[0],
            )
            baseline_children = {(baseline.pid, baseline_identity[0])}
            deadline = time.monotonic() + 2.0
            while not run.descendants and time.monotonic() < deadline:
                RUNNER._capture_owned_processes([run], baseline_children)
                time.sleep(0.01)
            self.assertTrue(run.descendants)
            self.assertTrue(RUNNER.terminate_group([run], 0.2, baseline_children))
            self.assertIsNone(worker.poll()) if False else None
            self.assertIsNone(baseline.poll())
            self.assertFalse(
                any(RUNNER._present(pid, started) for pid, started in run.descendants.items())
            )
        finally:
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=2)
            baseline.terminate()
            baseline.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
