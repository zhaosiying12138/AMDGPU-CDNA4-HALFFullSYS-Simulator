from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ACCEPTANCE = load_module(
    "upstream_rocr_aql_acceptance_test_module",
    ROOT / "tools/upstream_rocr_aql_acceptance.py",
)
RUNNER = load_module(
    "run_upstream_rocr_aql_test_module",
    ROOT / "scripts/run_upstream_rocr_aql.py",
)


class AcceptanceFixture:
    def __init__(self, root: Path) -> None:
        self.source = root / "source"
        self.source.mkdir(mode=0o700)
        (self.source / "m5out").mkdir(mode=0o700)
        self.execution_root = root / "deleted-execution-root"
        self.endpoint = self.execution_root / "bridge.sock"
        self.trace_path = self.execution_root / "dispatch-trace.jsonl"
        self.job_uuid = "1" * 32
        self.daemon_uuid = "2" * 32
        self.worker_pid = 1_000_000_001
        self.gem5_pid = 1_000_000_002
        self.identity = {"fixed": "identity"}
        self.gem5_argv = RUNNER.gem5_argv(
            self.execution_root,
            self.endpoint,
            self.trace_path,
            self.job_uuid,
        )
        self.execution = {
            "job_uuid": self.job_uuid,
            "epoch": 1,
            "rank": 0,
            "world_size": 1,
            "execution_root": str(self.execution_root),
            "endpoint": str(self.endpoint),
            "trace_path": str(self.trace_path),
            "m5out_path": str(self.execution_root / "m5out"),
            "gem5_argv": self.gem5_argv,
            "worker_argv": RUNNER.worker_argv(),
            "gem5_environment": RUNNER.common_environment(self.execution_root),
            "worker_environment": RUNNER.worker_environment(
                self.execution_root, self.endpoint
            ),
            "gem5_pid": self.gem5_pid,
            "gem5_start_time_ticks": 101,
            "gem5_process_group": self.gem5_pid,
            "worker_pid": self.worker_pid,
            "worker_start_time_ticks": 102,
            "worker_process_group": self.worker_pid,
            "worker_exit_code": 0,
            "gem5_exit_code": 0,
            "worker_timeout_seconds": 120,
            "gem5_exit_timeout_seconds": 20,
        }
        self.cleanup = {
            "worker_reaped": True,
            "gem5_reaped": True,
            "worker_process_group_absent": True,
            "gem5_process_group_absent": True,
            "endpoint_absent": True,
            "worker_forced_termination": False,
            "gem5_forced_termination": False,
            "all_clear": True,
        }
        self.manifest = {
            "schema": ACCEPTANCE.RUN_SCHEMA,
            "status": "success",
            "claim_scope": ACCEPTANCE.CLAIM_SCOPE,
            "hip_runtime_accepted": False,
            "pytorch_rocm_accepted": False,
            "model_accepted": False,
            "identity_preflight": self.identity,
            "identity_postflight": self.identity,
            "execution": self.execution,
            "cleanup": self.cleanup,
        }
        self._write_artifacts()
        self.refresh_manifest()

    def _retired(self, index: int) -> dict[str, object]:
        queue_object = 11 if index in (0, 2) else 22
        return {
            "schema": ACCEPTANCE.TRACE_SCHEMA,
            "event": "native_execution_retired",
            "source": "upstream_rocr_kmt_aql",
            "sim_tick": 200 + index * 50,
            "daemon_uuid": self.daemon_uuid,
            "job_uuid": self.job_uuid,
            "epoch": 1,
            "rank": 0,
            "world_size": 1,
            "connection_id": 99,
            "owner_fd": 10,
            "owner_generation": 1,
            "queue_object_id": queue_object,
            "execution_ticket": index + 1,
            "dispatch_id": 32,
            "queue_index": [1, 0, 2][index],
            "descriptor_abi": [2, 3, 2][index],
            "kernarg_size": [0, 280, 0][index],
            "grid": [[65536, 1, 1], [64, 1, 1], [65536, 1, 1]][index],
            "workgroup": [64, 1, 1],
            "packet_fetches": 1,
            "command_processor_submissions": 1,
            "dispatcher_starts": 1,
            "workgroups_completed": [1024, 1, 1024][index],
            "start_tick": 100 + index * 50,
            "end_tick": 150 + index * 50,
            "retire_tick": 150 + index * 50,
            "signal_before": 1,
            "signal_after": 0,
            "doorbell_ack_durable": True,
            "queue_retired": True,
            "pins_released": True,
            "cleanup_complete": False,
            "kernel_executed": True,
        }

    def _write_artifacts(self) -> None:
        worker_lines = [
            f"hsakmt-model pid={self.worker_pid} phase=queue-doorbell queue_id=1 slot=0 doorbell=0 notification=1 completion=0 status=0",
            f"hsakmt-model pid={self.worker_pid} phase=queue-doorbell queue_id=1 slot=0 doorbell=1 notification=2 completion=1 status=0",
            f"hsakmt-model pid={self.worker_pid} phase=queue-retired queue_id=1 slot=0 doorbell=1 notification=2 completion=1 status=0",
            f"hsakmt-model pid={self.worker_pid} phase=queue-retired queue_id=1 slot=0 doorbell=1 notification=2 completion=2 status=0",
            f"hsakmt-model pid={self.worker_pid} phase=queue-doorbell queue_id=2 slot=1 doorbell=0 notification=1 completion=0 status=0",
            f"hsakmt-model pid={self.worker_pid} phase=queue-retired queue_id=2 slot=1 doorbell=0 notification=1 completion=1 status=0",
            f"hsakmt-model pid={self.worker_pid} phase=queue-doorbell queue_id=1 slot=0 doorbell=2 notification=3 completion=2 status=0",
            f"hsakmt-model pid={self.worker_pid} phase=queue-retired queue_id=1 slot=0 doorbell=2 notification=3 completion=3 status=0",
            "agent device=1 name=gfx950",
            "upstream-rocr-execution phase=verified",
            "upstream ROCr standard AQL execution passed: elements=64 kernel_object=0x1 kernarg=280",
        ]
        (self.source / "worker.log").write_text(
            "\n".join(worker_lines) + "\n", encoding="ascii"
        )
        gem5_lines = [
            (
                f"host-gpu-ready endpoint={self.endpoint} "
                f"daemon_uuid={self.daemon_uuid} job_uuid={self.job_uuid} "
                "epoch=1 rank=0 world=1 max_record=65536"
            ),
            f"gem5 executing on test, pid {self.gem5_pid}",
            "command line: " + " ".join(self.gem5_argv),
            "host-gpu-handshake status=OK fd=10 generation=1",
            "host-gpu-dispatch-exit cause=host GPU dispatch session complete code=0 tick=400 stats=/tmp/stats.txt",
        ]
        (self.source / "gem5.log").write_text(
            "\n".join(gem5_lines) + "\n", encoding="ascii"
        )
        records = [self._retired(index) for index in range(3)]
        records.append(
            {
                "schema": ACCEPTANCE.TRACE_SCHEMA,
                "event": "native_execution_session_complete",
                "source": "upstream_rocr_kmt_aql",
                "sim_tick": 400,
                "daemon_uuid": self.daemon_uuid,
                "job_uuid": self.job_uuid,
                "epoch": 1,
                "rank": 0,
                "world_size": 1,
                "connection_id": 99,
                "owner_fd": 10,
                "owner_generation": 1,
                "retired_dispatches": 3,
                "owner_disconnected": True,
                "cleanup_complete": True,
                "kernel_executed": True,
            }
        )
        (self.source / "dispatch-trace.jsonl").write_bytes(
            b"".join(ACCEPTANCE.canonical_json(value) for value in records)
        )
        (self.source / "m5out/stats.txt").write_text(
            "simTicks 400\n"
            "finalTick 400\n"
            "hostSeconds 1.25\n"
            "system.host_gpu_bridge.host_fallback_count 0\n",
            encoding="ascii",
        )
        (self.source / "m5out/config.ini").write_text("[root]\n", encoding="ascii")
        (self.source / "m5out/config.json").write_text("{}\n", encoding="ascii")

    def refresh_manifest(self) -> None:
        self.manifest["artifacts"] = {
            relative: ACCEPTANCE.artifact_record(self.source, relative)
            for relative in ACCEPTANCE.SOURCE_ARTIFACTS
        }
        (self.source / "result-manifest.json").write_bytes(
            ACCEPTANCE.canonical_json(self.manifest)
        )

    def validate(self):
        return ACCEPTANCE.validate_source(
            self.source.resolve(), snapshot=lambda: self.identity
        )


class UpstreamRocrAqlAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.fixture = AcceptanceFixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_happy_source_and_atomic_publish(self) -> None:
        validation = self.fixture.validate()
        self.assertTrue(validation["correct"])
        output = self.root / "accepted"
        result = ACCEPTANCE.publish(self.fixture.source.resolve(), output, validation)
        self.assertEqual(result["status"], "accepted")
        self.assertTrue(result["standard_rocr_aql_accepted"])
        self.assertFalse(result["hip_runtime_accepted"])
        self.assertEqual(
            {entry.name for entry in output.iterdir()},
            {"source", "result.json", "manifest.json"},
        )
        with self.assertRaises(ACCEPTANCE.AcceptanceError):
            ACCEPTANCE.publish(self.fixture.source.resolve(), output, validation)

    def test_trace_semantics_are_recomputed(self) -> None:
        path = self.fixture.source / "dispatch-trace.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[1]["signal_after"] = 1
        path.write_bytes(b"".join(ACCEPTANCE.canonical_json(value) for value in records))
        self.fixture.refresh_manifest()
        with self.assertRaisesRegex(ACCEPTANCE.AcceptanceError, "signal transition"):
            self.fixture.validate()

    def test_worker_queue_lifecycle_is_recomputed(self) -> None:
        path = self.fixture.source / "worker.log"
        text = path.read_text(encoding="ascii")
        path.write_text(
            text.replace("notification=3 completion=3", "notification=3 completion=2"),
            encoding="ascii",
        )
        self.fixture.refresh_manifest()
        with self.assertRaisesRegex(ACCEPTANCE.AcceptanceError, "queue lifecycle"):
            self.fixture.validate()

    def test_artifact_byte_drift_is_rejected(self) -> None:
        path = self.fixture.source / "m5out/config.ini"
        path.write_bytes(path.read_bytes() + b"x")
        with self.assertRaisesRegex(ACCEPTANCE.AcceptanceError, "content drifted"):
            self.fixture.validate()

    def test_claim_and_environment_overstatement_are_rejected(self) -> None:
        self.fixture.manifest["hip_runtime_accepted"] = True
        self.fixture.refresh_manifest()
        with self.assertRaisesRegex(ACCEPTANCE.AcceptanceError, "overclaims HIP"):
            self.fixture.validate()
        self.fixture.manifest["hip_runtime_accepted"] = False
        self.fixture.execution["worker_environment"]["CUDA_VISIBLE_DEVICES"] = "0"
        self.fixture.refresh_manifest()
        with self.assertRaisesRegex(ACCEPTANCE.AcceptanceError, "worker environment differs"):
            self.fixture.validate()

    def test_forced_termination_is_not_accepted(self) -> None:
        self.fixture.cleanup["gem5_forced_termination"] = True
        self.fixture.cleanup["all_clear"] = False
        self.fixture.refresh_manifest()
        with self.assertRaisesRegex(ACCEPTANCE.AcceptanceError, "cleanup invariant"):
            self.fixture.validate()

    def test_runner_environment_is_private_and_cuda_free(self) -> None:
        root = self.root / "execution"
        endpoint = root / "bridge.sock"
        environment = RUNNER.worker_environment(root, endpoint)
        self.assertEqual(environment["SAGR_GENERIC_BRIDGE_ENDPOINT"], str(endpoint))
        self.assertEqual(environment["HSA_MODEL_TOPOLOGY"], str(ACCEPTANCE.TOPOLOGY.resolve()))
        for forbidden in (
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDA_VISIBLE_DEVICES",
            "CONDA_PREFIX",
            "PYTHONPATH",
            "LD_PRELOAD",
            "ROCM_PATH",
        ):
            self.assertNotIn(forbidden, environment)

    def test_runner_rejects_relative_and_existing_outputs(self) -> None:
        with self.assertRaisesRegex(RUNNER.RunnerError, "absolute"):
            RUNNER.validate_output(Path("relative"))
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaisesRegex(RUNNER.RunnerError, "absent"):
            RUNNER.validate_output(existing)

    def test_runner_fake_success_uses_private_process_contract(self) -> None:
        calls: list[tuple[list[str], dict[str, object]]] = []
        next_pid = iter((810001, 810002))

        class FakeProcess:
            def __init__(self, argv, **kwargs):
                self.argv = list(argv)
                self.kwargs = kwargs
                self.pid = next(next_pid)
                self.return_code = None
                calls.append((self.argv, kwargs))

            def poll(self):
                return self.return_code

            def wait(self, timeout=None):
                del timeout
                if self.argv[0] == str(ACCEPTANCE.WORKER.resolve()):
                    self.kwargs["stdout"].write(b"worker output\n")
                else:
                    outdir = Path(self.argv[self.argv.index("--outdir") + 1])
                    trace = Path(
                        self.argv[self.argv.index("--dispatch-trace-path") + 1]
                    )
                    (outdir / "stats.txt").write_text("stats\n", encoding="ascii")
                    (outdir / "config.ini").write_text("ini\n", encoding="ascii")
                    (outdir / "config.json").write_text("{}\n", encoding="ascii")
                    trace.write_text("trace\n", encoding="ascii")
                    self.kwargs["stdout"].write(b"gem5 output\n")
                self.return_code = 0
                return 0

        identity = {"identity": "stable"}
        identity_values = iter(((810001, 11, 810001), (810002, 12, 810002)))
        output = self.root / "run-source"
        with (
            mock.patch.object(RUNNER.contract, "identity_snapshot", return_value=identity),
            mock.patch.object(RUNNER.subprocess, "Popen", side_effect=FakeProcess),
            mock.patch.object(RUNNER, "process_identity", side_effect=lambda *_: next(identity_values)),
            mock.patch.object(RUNNER, "wait_for_endpoint"),
            mock.patch.object(RUNNER, "process_group_members", return_value=[]),
        ):
            manifest = RUNNER.run_gate(
                output,
                worker_timeout_seconds=120,
                gem5_exit_timeout_seconds=20,
                startup_timeout_seconds=30,
            )
        self.assertEqual(manifest["status"], "success")
        self.assertEqual(set(manifest["artifacts"]), set(ACCEPTANCE.SOURCE_ARTIFACTS))
        self.assertEqual(len(calls), 2)
        for _, kwargs in calls:
            self.assertIs(kwargs["start_new_session"], True)
            self.assertIs(kwargs["close_fds"], True)
            self.assertEqual(kwargs["stdin"], RUNNER.subprocess.DEVNULL)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", kwargs["env"])

    def test_runner_unexpected_spawn_failure_is_published_and_cleaned(self) -> None:
        output = self.root / "failed-source"
        before = set(Path("/tmp").glob("gs-rocr-aql-*"))
        with (
            mock.patch.object(
                RUNNER.contract,
                "identity_snapshot",
                return_value={"identity": "stable"},
            ),
            mock.patch.object(
                RUNNER.subprocess,
                "Popen",
                side_effect=RuntimeError("injected unexpected spawn failure"),
            ),
        ):
            manifest = RUNNER.run_gate(
                output,
                worker_timeout_seconds=120,
                gem5_exit_timeout_seconds=20,
                startup_timeout_seconds=30,
            )
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("runner-error.txt", manifest["artifacts"])
        self.assertEqual(before, set(Path("/tmp").glob("gs-rocr-aql-*")))
        self.assertIn(
            "injected unexpected spawn failure",
            (output / "runner-error.txt").read_text(encoding="ascii"),
        )


if __name__ == "__main__":
    unittest.main()
