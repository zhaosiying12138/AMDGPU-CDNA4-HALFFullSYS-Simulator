from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import gemsim_single_rank_lifecycle as lifecycle
from scripts import run_rocm_pytorch_runtime as runner


class RocmPytorchRuntimeRunnerTest(unittest.TestCase):
    def test_worker_contract_uses_activation_and_ordinary_demo(self) -> None:
        identity = {
            "files": {
                "activation": {"path": "/product/activate"},
                "python": {"path": "/product/python"},
                "quickstart": {"path": "/repo/examples/quickstart/torch_rocm.py"},
            }
        }
        self.assertEqual(
            runner.worker_argv(identity)[-3:],
            [
                "/product/activate",
                "/product/python",
                "/repo/examples/quickstart/torch_rocm.py",
            ],
        )
        environment = runner.worker_environment(
            Path("/tmp/private"), Path("/tmp/private/bridge.sock")
        )
        self.assertEqual(
            environment["SAGR_GENERIC_BRIDGE_ENDPOINT"],
            "/tmp/private/bridge.sock",
        )
        for forbidden in ("CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX", "PYTHONPATH"):
            self.assertNotIn(forbidden, environment)

    def test_generic_lifecycle_builds_one_rank_command(self) -> None:
        command = lifecycle.gem5_argv(
            binary=runner.contract.GEM5_BINARY,
            config=runner.contract.GEM5_CONFIG,
            execution_root=Path("/tmp/test-root"),
            endpoint=Path("/tmp/test-root/bridge.sock"),
            trace_path=Path("/tmp/test-root/trace.jsonl"),
            job_uuid="a" * 32,
        )
        self.assertEqual(command[command.index("--rank") + 1], "0")
        self.assertEqual(command[command.index("--world-size") + 1], "1")
        self.assertEqual(command[command.index("--epoch") + 1], "1")
        self.assertEqual(
            command[command.index("--dispatch-trace-path") + 1],
            "/tmp/test-root/trace.jsonl",
        )

    def test_run_gate_rejects_identity_drift_without_accepting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execution_root = root / "execution"
            execution_root.mkdir()
            for relative in runner.contract.SOURCE_ARTIFACTS:
                path = execution_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            session = lifecycle.SessionResult(
                execution_root=execution_root,
                execution={"worker_exit_code": 0, "gem5_exit_code": 0},
                cleanup={"all_clear": True},
                failure=None,
                process_success=True,
            )
            published: dict[str, object] = {}

            def capture(**kwargs):
                published.update(kwargs["manifest"])
                published["error"] = kwargs["error"]
                return dict(kwargs["manifest"])

            identity_a = {
                "files": {
                    "activation": {"path": "/a"},
                    "python": {"path": "/p"},
                    "quickstart": {"path": "/q"},
                }
            }
            identity_b = {
                "files": {
                    "activation": {"path": "/changed"},
                    "python": {"path": "/p"},
                    "quickstart": {"path": "/q"},
                }
            }
            with (
                mock.patch.object(
                    runner.contract,
                    "identity_snapshot",
                    side_effect=[identity_a, identity_b],
                ),
                mock.patch.object(runner.lifecycle, "run_session", return_value=session),
                mock.patch.object(runner.lifecycle, "publish_source", side_effect=capture),
            ):
                result = runner.run_gate(
                    root / "result",
                    worker_timeout_seconds=10,
                    gem5_exit_timeout_seconds=10,
                    startup_timeout_seconds=10,
                )
            self.assertEqual(result["status"], "failed")
            self.assertIn("identity drifted", published["error"])
            self.assertFalse(published["ordinary_upstream_pytorch_eager_executed"])

    def test_absent_output_contract_rejects_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "present"
            path.mkdir()
            with self.assertRaisesRegex(lifecycle.LifecycleError, "absent"):
                lifecycle.validate_absent_output(path)


if __name__ == "__main__":
    unittest.main()
