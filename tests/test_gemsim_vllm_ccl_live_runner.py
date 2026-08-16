# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
from pathlib import Path
import socket
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
from scripts import run_gemsim_vllm_ccl_live as RUNNER  # noqa: E402


class VllmCCLLiveRunnerTest(unittest.TestCase):
    class FakeProcess:
        def __init__(self, pid: int, returncode: int):
            self.pid = pid
            self.returncode = returncode

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    class FakeBroker:
        def __init__(self, failure):
            self.owner = SimpleNamespace(pid=41, start_time_ticks=73)
            self.failure = failure
            self.peers = []
            self.bound = []

        def prepare_rank(self, rank):
            left, right = socket.socketpair()
            self.peers.append(right)
            return left.detach()

        def bind_rank(self, rank, identity):
            self.bound.append((rank, identity.pid))

        def rendezvous(self, deadline):
            return None

        def progress(self):
            return self.failure

        def abort(self, failed_rank, reason, sequence):
            if self.failure is None:
                self.failure = SimpleNamespace(
                    status=reason,
                    reporter_rank=failed_rank,
                    failed_rank=failed_rank,
                    context_sequence=sequence,
                )
            return self.failure

        def first_error(self):
            return self.failure

        def info(self):
            return SimpleNamespace(abort_pending_mask=0)

        def destroy(self):
            for peer in self.peers:
                peer.close()
            self.peers.clear()

    class FakeNative:
        def __init__(self, runtime_sha, broker):
            self.library_sha256 = runtime_sha
            self.broker = broker
            self.now = 1_000_000_000

        def identity(self, **kwargs):
            return object()

        def monotonic_time_ns(self):
            self.now += 1000
            return self.now

        def live_broker(self, identity):
            return self.broker

        def process_identity(self, pid):
            return SimpleNamespace(pid=pid, start_time_ticks=pid + 10000)

    def test_active_product_and_installed_vllm_bindings_are_exact(self):
        prefix = RUNNER.base.default_product_prefix()
        manifest, paths = RUNNER.load_product(prefix)
        identity = RUNNER.source_identity(paths)
        self.assertEqual(set(identity), set(RUNNER.IDENTITY_ROLES))
        self.assertEqual(manifest["schema"], "amdgpu-sim.product-prefix.v1")
        self.assertEqual(manifest["prefix"], str(prefix.resolve(strict=True)))
        self.assertRegex(manifest["product_id"], r"^[0-9a-f]{64}$")
        for installed, checkout in (
            ("vllm_parallel_state", "vllm_checkout_parallel_state"),
            ("vllm_base_communicator", "vllm_checkout_base_communicator"),
            ("vllm_communication_op", "vllm_checkout_communication_op"),
            ("vllm_version", "vllm_checkout_version"),
        ):
            self.assertEqual(identity[installed]["sha256"], identity[checkout]["sha256"])

    def test_spawn_passes_only_the_rank_capability(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            left, right = socket.socketpair()
            try:
                run = RUNNER.base.RankProcess(
                    rank=0,
                    directory=directory,
                    launch={
                        "paths": {"triton_cache_directory": str(directory / "cache")}
                    },
                )
                run.capability_fd = left.fileno()
                observed = {}

                def fake_popen(command, **kwargs):
                    observed["command"] = command
                    observed.update(kwargs)
                    return object()

                with mock.patch.object(
                    RUNNER, "worker_environment", return_value={"LC_ALL": "C"}
                ):
                    process = RUNNER.spawn_rank_process(
                        run,
                        config_path=directory / "config.json",
                        bootstrap_path=directory / "bootstrap.json",
                        product_prefix=directory,
                        popen_factory=fake_popen,
                    )
                self.assertIsNotNone(process)
                self.assertEqual(observed["pass_fds"], (left.fileno(),))
                self.assertIs(observed["close_fds"], True)
                self.assertIs(observed["start_new_session"], True)
                self.assertIn("vllm_ccl_live_rank.py", observed["command"][1])
                self.assertEqual(
                    observed["command"][-1], str(left.fileno())
                )
            finally:
                if run.stdout is not None:
                    run.stdout.close()
                if run.stderr is not None:
                    run.stderr.close()
                left.close()
                right.close()

    def test_bootstrap_is_generic_and_exact_for_rank_15(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            runtime_path = Path(temporary) / "runtime.so"
            manifest_path.write_bytes(b"{}\n")
            runtime_path.write_bytes(b"runtime")
            manifest = {"prefix": temporary}
            preflight = {
                "product_manifest": RUNNER.base.file_record(manifest_path),
                "runtime_library": RUNNER.base.file_record(runtime_path),
            }
            identity = {
                "world_size": 16,
                "epoch": 9,
                "group_generation": 4,
                "job_uuid": "11" * 16,
                "group_uuid": "22" * 16,
                "model_identity_sha256": "33" * 32,
            }
            document = RUNNER.bootstrap_document(
                manifest=manifest,
                identity=identity,
                preflight=preflight,
                rank=15,
                capability_fd=71,
                broker_pid=123,
                broker_start_time_ticks=456,
                timeout_ns=789,
                credits_per_peer=2,
            )
            group = document["groups"][0]
            self.assertEqual(group["unique_name"], "tp:0")
            self.assertEqual(group["identity"]["world_size"], 16)
            self.assertEqual(group["rank"]["rank"], 15)
            self.assertEqual(group["rank"]["capability_fd"], 71)
            self.assertEqual(document, __import__("json").loads(
                RUNNER.base.canonical_json(document).decode("ascii")
            ))

    def _publish_expected(
        self, root: Path, execution_root: Path,
        model_identity_sha256: str | None = None,
    ) -> Path:
        expected = root / "expected.json"
        runtime = (
            RUNNER.base.default_product_prefix()
            / "lib/libself_amdgpu_runtime.so.1"
        )
        command = [
                __import__("sys").executable,
                str(ROOT / "tools/gemsim_ccl_live_allreduce.py"),
                "--design-only",
                "--runtime-library", str(runtime),
                "--namespace-root", str(execution_root),
                "--world-size", "2",
                "--element-count", "1024",
                "--dtype", "bfloat16",
                "--expected-output", str(expected),
            ]
        if model_identity_sha256 is not None:
            command.extend(["--model-identity-sha256", model_identity_sha256])
        __import__("subprocess").run(
            command,
            cwd=ROOT,
            check=True,
            stdout=__import__("subprocess").PIPE,
            stderr=__import__("subprocess").PIPE,
        )
        return expected

    def _supervise_fake(
        self, root: Path, fail: bool, leak_fd: bool = False,
        fail_before_rank: int | None = None, row_parallel: bool = False,
    ):
        output = root / "source"
        execution_root = root / "execution"
        workload_path = None
        workload = None
        if row_parallel:
            from tools import gemsim_vllm_ccl_live_acceptance as acceptance

            workload = acceptance.build_row_parallel_workload(
                ROOT / "models/Qwen3.5-0.8B"
            )
            workload_path = root / "row-workload.json"
            RUNNER.base._exclusive_write(
                workload_path, RUNNER.base.canonical_json(workload)
            )
        workload_sha = (
            None if workload is None else RUNNER.base.sha256_bytes(
                RUNNER.base.canonical_json(workload)
            )
        )
        expected_path = self._publish_expected(
            root, execution_root, workload_sha
        )
        expected = __import__("json").loads(expected_path.read_text("ascii"))
        prefix = RUNNER.base.default_product_prefix()
        runtime = prefix / "lib/libself_amdgpu_runtime.so.1"
        failure = (
            SimpleNamespace(
                status=RUNNER.PEER_LOST,
                reporter_rank=0,
                failed_rank=1,
                context_sequence=1,
            )
            if fail else None
        )
        broker = self.FakeBroker(failure)
        if fail_before_rank is not None:
            original_prepare = broker.prepare_rank

            def prepare_rank(rank):
                if rank == fail_before_rank:
                    raise RuntimeError("synthetic prepare failure")
                return original_prepare(rank)

            broker.prepare_rank = prepare_rank
        native = self.FakeNative(expected["design"]["runtime"]["sha256"], broker)
        identity_record = RUNNER.base.file_record(Path(__file__))
        identities = {role: dict(identity_record) for role in RUNNER.IDENTITY_ROLES}
        identities["runtime_library"] = RUNNER.base.file_record(runtime)
        identities["product_manifest"] = RUNNER.base.file_record(prefix / "manifest.json")

        def fake_popen(command, **kwargs):
            self.assertIs(kwargs["close_fds"], True)
            self.assertEqual(kwargs["pass_fds"], (int(command[-1]),))
            config_path = Path(command[command.index("--config") + 1])
            config = __import__("json").loads(config_path.read_text("ascii"))
            rank = config["rank"]
            payload = (
                __import__(
                    "tools.gemsim_vllm_ccl_live_acceptance",
                    fromlist=["row_parallel_input"],
                ).row_parallel_input(rank)
                if row_parallel else
                RUNNER.base.deterministic_input("bfloat16", rank, 1024)
            )
            output_payload = (
                b"\0" * 2048 if row_parallel else payload
            )
            result = {
                "schema": RUNNER.RANK_RESULT_SCHEMA,
                "status": "device_failure" if fail and rank == 0 else "success",
                "rank": rank,
                "world_size": 2,
                "acceptance_authority": False,
                "live_adapter_accepted": False,
                "public_result_published": not (fail and rank == 0),
                "input_sha256_before": RUNNER.base.sha256_bytes(payload),
                "input_sha256_after": RUNNER.base.sha256_bytes(payload),
                "output_sha256": RUNNER.base.sha256_bytes(output_payload),
                "output_storage_fresh": True,
                "bootstrap_descriptor_sha256": config["bootstrap_descriptor_sha256"],
                "adapter_evidence_sha256": RUNNER.base.sha256_bytes(
                    RUNNER.base.canonical_json({})
                ),
                "managed_session": None,
                "first_error": None,
                "product": config["product"],
            }
            RUNNER.base._exclusive_write(
                Path(config["result_path"]), RUNNER.base.canonical_json(result)
            )
            RUNNER.base._exclusive_write(
                Path(config["adapter_evidence_path"]), RUNNER.base.canonical_json({})
            )
            RUNNER.base._exclusive_write(Path(config["journal_path"]), b"")
            RUNNER.base._exclusive_write(Path(config["input_path"]), payload)
            RUNNER.base._exclusive_write(Path(config["output_path"]), output_payload)
            launch = expected["design"]["ranks"][rank]["rank_launch"]
            for path, data in (
                (Path(launch["paths"]["dispatch_trace_path"]), b""),
                (Path(launch["paths"]["gem5_log_path"]), b""),
                (Path(launch["paths"]["gem5_output_directory"]) / "stats.txt", b""),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                RUNNER.base._exclusive_write(path, data)
            return self.FakeProcess(10000 + rank, 1 if fail and rank == 0 else 0)

        def fake_terminate(runs, *args):
            for run in runs:
                run.returncode = run.process.poll() if run.process else None
                if run.stdout:
                    run.stdout.close()
                if run.stderr:
                    run.stderr.close()
            return True

        paths = {role: Path(__file__) for role in RUNNER.IDENTITY_ROLES}
        paths["runtime_library"] = runtime
        with (
            mock.patch.object(RUNNER, "load_product", return_value=(
                {"product_id": "fake", "prefix": str(prefix)}, paths
            )),
            mock.patch.object(RUNNER, "source_identity", return_value=identities),
            mock.patch.object(RUNNER.base, "enable_subreaper"),
            mock.patch.object(
                RUNNER.base,
                "owned_fd_snapshot",
                side_effect=[
                    {},
                    {91: (1, 2, stat.S_IFREG, "/synthetic/leak")}
                    if leak_fd else {},
                ],
            ),
            mock.patch.object(RUNNER.base, "_direct_child_identities", return_value=set()),
            mock.patch.object(RUNNER.base, "_capture_owned_processes"),
            mock.patch.object(RUNNER.base, "_present", return_value=False),
            mock.patch.object(RUNNER.base, "terminate_group", side_effect=fake_terminate),
        ):
            manifest = RUNNER.supervise(
                expected_path=expected_path,
                output=output,
                execution_root=execution_root,
                product_prefix=prefix,
                timeout_seconds=5,
                cleanup_grace_seconds=0.1,
                popen_factory=fake_popen,
                native_factory=lambda path: native,
                workload_path=workload_path,
            )
        return manifest, output

    def test_supervise_success_materializes_complete_rank_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, output = self._supervise_fake(Path(temporary), False)
            self.assertEqual(manifest["status"], "success")
            self.assertTrue(manifest["supervisor_cleanup"]["all_clear"])
            self.assertEqual([entry["returncode"] for entry in manifest["ranks"]], [0, 0])
            for rank in range(2):
                self.assertEqual(
                    set((output / f"rank-{rank:02d}").iterdir()),
                    {output / f"rank-{rank:02d}" / name for name in RUNNER.RANK_FILES},
                )

    def test_supervise_row_parallel_binds_workload_to_collective_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, output = self._supervise_fake(
                Path(temporary), False, row_parallel=True
            )
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(
                manifest["workload"]["document"]["kind"], "vllm-row-parallel"
            )
            workload_sha = RUNNER.base.sha256_bytes(
                RUNNER.base.canonical_json(manifest["workload"]["document"])
            )
            expected = __import__("json").loads(
                (Path(temporary) / "expected.json").read_text("ascii")
            )
            self.assertEqual(
                expected["design"]["config"]["model_identity_sha256"],
                workload_sha,
            )
            for rank in range(2):
                self.assertEqual(
                    (output / f"rank-{rank:02d}/input.bin").stat().st_size,
                    3584,
                )

    def test_supervise_failure_normalizes_all_ranks_and_removes_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, output = self._supervise_fake(Path(temporary), True)
            self.assertEqual(manifest["status"], "peer_lost")
            self.assertEqual(manifest["first_error"]["failed_rank"], 1)
            for rank in range(2):
                result = __import__("json").loads(
                    (output / f"rank-{rank:02d}/worker-result.json").read_text("ascii")
                )
                self.assertEqual(result["status"], "peer_lost")
                self.assertFalse(result["public_result_published"])
                self.assertEqual((output / f"rank-{rank:02d}/output.bin").stat().st_size, 0)

    def test_supervise_fd_leak_fails_closed_and_removes_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, output = self._supervise_fake(
                Path(temporary), False, leak_fd=True
            )
            self.assertEqual(manifest["status"], "device_failure")
            self.assertFalse(manifest["supervisor_cleanup"]["all_clear"])
            self.assertEqual(manifest["supervisor_cleanup"]["measured_fd_delta"], 1)
            for rank in range(2):
                result = __import__("json").loads(
                    (output / f"rank-{rank:02d}/worker-result.json").read_text("ascii")
                )
                self.assertEqual(result["status"], "device_failure")
                self.assertFalse(result["public_result_published"])
                self.assertEqual((output / f"rank-{rank:02d}/output.bin").stat().st_size, 0)

    def test_supervise_partial_launch_materializes_unprepared_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, output = self._supervise_fake(
                Path(temporary), False, fail_before_rank=1
            )
            self.assertNotEqual(manifest["status"], "success")
            unprepared = manifest["ranks"][1]
            self.assertIsNone(unprepared["worker_pid"])
            self.assertEqual(unprepared["capability"]["pass_fds"], [])
            bootstrap = __import__("json").loads(
                (output / "rank-01/bootstrap-descriptor.json").read_text("ascii")
            )
            self.assertEqual(bootstrap["status"], "rank_not_prepared")
            self.assertEqual((output / "rank-01/output.bin").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
