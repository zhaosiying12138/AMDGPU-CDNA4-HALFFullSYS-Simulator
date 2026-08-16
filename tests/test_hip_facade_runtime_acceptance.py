from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ACCEPTANCE = load(
    "hip_facade_runtime_acceptance_test_module",
    ROOT / "tools/hip_facade_runtime_acceptance.py",
)
RUNNER = load(
    "run_hip_facade_runtime_test_module",
    ROOT / "scripts/run_hip_facade_runtime.py",
)
SMOKE = load(
    "hip_facade_runtime_smoke_test_contract",
    ROOT / "tools/hip_facade_runtime_smoke.py",
)


class HipFacadeRuntimeAcceptanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.worker_path = self.root / "worker.log"
        self.trace_path = self.root / "dispatch-trace.jsonl"
        self.identity = {
            "files": {
                "compiler": {"path": "/frozen/bin/clang++", "sha256": "1" * 64},
                "hip_library": {"path": "/frozen/lib/libamdhip64.so.7", "sha256": "2" * 64},
                "activation": {"path": "/frozen/activate"},
                "python": {"path": "/frozen/bin/python"},
                "smoke": {"path": str(ACCEPTANCE.SMOKE)},
                "gem5_binary": {"path": str(ACCEPTANCE.GEM5_BINARY.resolve())},
                "gem5_config": {"path": str(ACCEPTANCE.GEM5_CONFIG.resolve())},
            }
        }
        self.worker = {
            "schema": ACCEPTANCE.SMOKE_SCHEMA,
            "mode": "vector-add",
            "device_count": 1,
            "hip_library": {
                "path": self.identity["files"]["hip_library"]["path"],
                "sha256": self.identity["files"]["hip_library"]["sha256"],
            },
            "compilation": {
                "compiler_path": self.identity["files"]["compiler"]["path"],
                "compiler_sha256": self.identity["files"]["compiler"]["sha256"],
                "source_sha256": hashlib.sha256(SMOKE.KERNEL_SOURCE.encode("ascii")).hexdigest(),
                "image_bytes": 5576,
                "image_sha256": "3" * 64,
                "target": "gfx950",
                "device_only": True,
                "output_format": "elf64-amdgpu",
            },
            "execution": {
                "count": 256,
                "grid": [4, 1, 1],
                "block": [64, 1, 1],
                "mismatch_count": 0,
                "output_sha256": ACCEPTANCE.EXPECTED_OUTPUT_SHA256,
                "correct": True,
            },
            "path": [
                "upstream_hip_api",
                "upstream_rocr",
                "upstream_hsakmt_model_interface",
                "self_runtime_provider",
                "runtime_gem5_bridge",
                "gem5_gpu_model",
            ],
            "fallback": {"cpu": 0, "cuda": 0, "privateuse1": 0, "project_operator": 0},
            "correct": True,
        }
        self.execution = {"job_uuid": "a" * 32}
        common = {
            "schema": ACCEPTANCE.TRACE_SCHEMA,
            "source": "upstream_rocr_kmt_aql",
            "daemon_uuid": "b" * 32,
            "job_uuid": "a" * 32,
            "epoch": 1,
            "rank": 0,
            "world_size": 1,
            "connection_id": 17,
            "kernel_executed": True,
        }
        self.trace = [
            {
                **common,
                "event": "native_execution_retired",
                "execution_ticket": 1,
                "dispatch_id": 32,
                "descriptor_abi": 3,
                "queue_index": 0,
                "kernarg_size": 288,
                "grid": [256, 1, 1],
                "workgroup": [64, 1, 1],
                "workgroups_completed": 4,
                "packet_fetches": 1,
                "command_processor_submissions": 1,
                "dispatcher_starts": 1,
                "signal_before": 0,
                "signal_after": 0,
                "doorbell_ack_durable": True,
                "queue_retired": True,
                "pins_released": True,
                "cleanup_complete": False,
                "start_tick": 100,
                "end_tick": 200,
                "retire_tick": 200,
            },
            {
                **common,
                "event": "native_execution_session_complete",
                "sim_tick": 300,
                "retired_dispatches": 1,
                "owner_disconnected": True,
                "cleanup_complete": True,
            },
        ]
        self.write_worker()
        self.write_trace()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_worker(self) -> None:
        self.worker_path.write_text(json.dumps(self.worker, sort_keys=True) + "\n", encoding="ascii")

    def write_trace(self) -> None:
        self.trace_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in self.trace),
            encoding="ascii",
        )

    def test_worker_and_trace_happy_path(self) -> None:
        worker = ACCEPTANCE.validate_worker(self.worker_path, self.identity)
        trace = ACCEPTANCE.parse_trace(self.trace_path, self.execution)
        self.assertTrue(worker["correct"])
        self.assertEqual(worker["output_sha256"], ACCEPTANCE.EXPECTED_OUTPUT_SHA256)
        self.assertEqual(trace["retired_dispatches"], 1)
        self.assertEqual(trace["terminal_tick"], 300)

    def test_worker_rejects_numerical_and_fallback_tamper(self) -> None:
        for mutate in (
            lambda value: value["execution"].__setitem__("mismatch_count", 1),
            lambda value: value["execution"].__setitem__("output_sha256", "4" * 64),
            lambda value: value["fallback"].__setitem__("cpu", 1),
            lambda value: value.__setitem__("path", ["project_operator"]),
        ):
            with self.subTest(mutate=mutate):
                original = copy.deepcopy(self.worker)
                mutate(self.worker)
                self.write_worker()
                with self.assertRaises(ACCEPTANCE.AcceptanceError):
                    ACCEPTANCE.validate_worker(self.worker_path, self.identity)
                self.worker = original

    def test_trace_rejects_identity_execution_and_cleanup_tamper(self) -> None:
        cases = (
            (0, "job_uuid", "c" * 32),
            (0, "workgroups_completed", 3),
            (0, "signal_before", 1),
            (0, "pins_released", False),
            (1, "retired_dispatches", 2),
            (1, "cleanup_complete", False),
        )
        for index, key, value in cases:
            with self.subTest(index=index, key=key):
                original = copy.deepcopy(self.trace)
                self.trace[index][key] = value
                self.write_trace()
                with self.assertRaises(ACCEPTANCE.AcceptanceError):
                    ACCEPTANCE.parse_trace(self.trace_path, self.execution)
                self.trace = original

    def test_runner_uses_product_activation_and_public_smoke(self) -> None:
        argv = RUNNER.worker_argv(self.identity)
        self.assertEqual(argv[:4], ["/bin/bash", "--noprofile", "--norc", "-c"])
        self.assertIn(self.identity["files"]["activation"]["path"], argv)
        self.assertIn(self.identity["files"]["python"]["path"], argv)
        self.assertEqual(argv[-4:], ["--mode", "vector-add", "--count", "256"])
        source = (ROOT / "scripts/run_hip_facade_runtime.py").read_text(encoding="ascii")
        self.assertNotIn("sagr_managed_kernel", source)
        self.assertNotIn("torch.ops", source)


if __name__ == "__main__":
    unittest.main()
