from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import gemsim_single_rank_lifecycle as lifecycle
from scripts import run_upstream_triton_runtime as runner


class UpstreamTritonRuntimeRunnerTest(unittest.TestCase):
    def test_worker_contract_overrides_activation_with_private_cache(self) -> None:
        identity = {
            "files": {
                "activation": {"path": "/product/activate"},
                "python": {"path": "/product/python"},
                "quickstart": {"path": "/repo/examples/quickstart/triton_rocm.py"},
            }
        }
        argv = runner.worker_argv(identity)
        self.assertEqual(
            argv[-3:],
            [
                "/product/activate",
                "/product/python",
                "/repo/examples/quickstart/triton_rocm.py",
            ],
        )
        self.assertIn('export TRITON_CACHE_DIR="$GEMSIM_RUN_TRITON_CACHE_DIR"', argv[4])
        environment = runner.worker_environment(
            Path("/tmp/gs-upstream-triton-test"),
            Path("/tmp/gs-upstream-triton-test/bridge.sock"),
        )
        self.assertEqual(
            environment["GEMSIM_RUN_TRITON_CACHE_DIR"],
            "/tmp/gs-upstream-triton-test/triton-cache",
        )
        self.assertNotIn("TRITON_CACHE_DIR", environment)
        for forbidden in ("CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX", "PYTHONPATH"):
            self.assertNotIn(forbidden, environment)

    def test_run_gate_rejects_identity_drift_without_accepting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execution_root = root / "execution"
            execution_root.mkdir()
            for relative in runner.contract.CORE_ARTIFACTS:
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
            published: dict[str, object] = {}

            def capture(**kwargs):
                published.update(kwargs["manifest"])
                published["error"] = kwargs["error"]
                return dict(kwargs["manifest"])

            with (
                mock.patch.object(
                    runner.contract,
                    "identity_snapshot",
                    side_effect=[identity_a, identity_b],
                ),
                mock.patch.object(runner.lifecycle, "run_session", return_value=session),
                mock.patch.object(
                    runner.contract,
                    "cache_artifact_paths",
                    return_value=("triton-cache/A/file",),
                ),
                mock.patch.object(
                    runner.lifecycle, "publish_source", side_effect=capture
                ),
            ):
                result = runner.run_gate(
                    root / "result",
                    worker_timeout_seconds=10,
                    gem5_exit_timeout_seconds=10,
                    startup_timeout_seconds=10,
                )
            self.assertEqual(result["status"], "failed")
            self.assertIn("identity drifted", published["error"])
            self.assertFalse(published["ordinary_upstream_triton_amd_executed"])
            self.assertFalse(published["triton_upstream_amd_accepted"])

    def test_absent_output_contract_rejects_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "present"
            path.mkdir()
            with self.assertRaisesRegex(lifecycle.LifecycleError, "absent"):
                lifecycle.validate_absent_output(path)


if __name__ == "__main__":
    unittest.main()
