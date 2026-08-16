from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools import rocm_pytorch_runtime_acceptance as acceptance


class RocmPytorchRuntimeAcceptanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.source = Path(self.temporary.name) / "source"
        self.source.mkdir()
        (self.source / "m5out").mkdir()
        self.execution_root = Path("/tmp/gs-rocm-pytorch-fixture")
        self.job_uuid = "a" * 32
        self.identity = {
            "product": {
                "native_role": "rocr_kmd_boundary_only",
                "runtime_probe": {"torch": "2.11.0+rocm", "torch_hip": "7.13"},
                "sdk_libraries": {
                    role: {"provider": "official_rocm_sdk"}
                    for role in ("hip_library", "comgr_library", "rccl_library")
                },
            },
            "files": {
                "activation": {"path": "/product/activate"},
                "python": {"path": "/product/python"},
                "quickstart": {"path": str(acceptance.QUICKSTART)},
            },
        }
        self.execution = self.make_execution()
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
        self.write_worker()
        self.write_trace()
        self.write_stats()
        self.write_gem5_log()
        (self.source / "m5out/config.ini").write_text("[root]\n", encoding="ascii")
        (self.source / "m5out/config.json").write_text("{}\n", encoding="ascii")
        self.write_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_execution(self) -> dict[str, object]:
        gem5_argv = [
            str(acceptance.GEM5_BINARY.resolve()),
            "--listener-mode=on",
            "--outdir",
            str(self.execution_root / "m5out"),
            str(acceptance.GEM5_CONFIG.resolve()),
            "--endpoint",
            str(self.execution_root / "bridge.sock"),
            "--dispatch-trace-path",
            str(self.execution_root / "dispatch-trace.jsonl"),
            "--epoch",
            "1",
            "--job-uuid",
            self.job_uuid,
            "--rank",
            "0",
            "--world-size",
            "1",
            "--startup-timeout-ms",
            "86400000",
            "--handshake-timeout-ms",
            "15000",
            "--run-timeout-ms",
            "86400000",
        ]
        return {
            "job_uuid": self.job_uuid,
            "epoch": 1,
            "rank": 0,
            "world_size": 1,
            "execution_root": str(self.execution_root),
            "endpoint": str(self.execution_root / "bridge.sock"),
            "trace_path": str(self.execution_root / "dispatch-trace.jsonl"),
            "m5out_path": str(self.execution_root / "m5out"),
            "gem5_argv": gem5_argv,
            "worker_argv": [
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                'set -eu; source "$1"; shift; exec "$@"',
                "rocm-pytorch-worker",
                "/product/activate",
                "/product/python",
                str(acceptance.QUICKSTART),
            ],
            "gem5_environment": {},
            "worker_environment": {
                "SAGR_GENERIC_BRIDGE_ENDPOINT": str(
                    self.execution_root / "bridge.sock"
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            },
            "gem5_pid": 1234,
            "gem5_start_time_ticks": 10,
            "gem5_process_group": 1234,
            "worker_pid": 1235,
            "worker_start_time_ticks": 11,
            "worker_process_group": 1235,
            "worker_exit_code": 0,
            "gem5_exit_code": 0,
            "worker_timeout_seconds": 120,
            "gem5_exit_timeout_seconds": 30,
            "startup_timeout_seconds": 30,
        }

    def worker_payload(self) -> dict[str, object]:
        return {
            "schema": acceptance.QUICKSTART_SCHEMA,
            "torch": "2.11.0+rocm",
            "torch_hip": "7.13",
            "device_count": 1,
            "device_name": "AMD Instinct MI350X",
            "capability": [9, 5],
            "operations": ["copy", "add", "sigmoid", "sum"],
            "tensor_contract": {
                "dtype": "float32",
                "input_shape": [16],
                "input_device": "cuda:0",
                "actual_sha256": acceptance.EXPECTED_TENSOR_SHA256,
                "expected_sha256": acceptance.EXPECTED_TENSOR_SHA256,
            },
            "checks": {
                "copy_bitwise": True,
                "add_bitwise": True,
                "sigmoid_bitwise": True,
                "sum_bitwise": True,
                "input_unchanged": True,
                "outputs_are_cuda": True,
                "outputs_fresh": True,
                "outputs_nonalias": True,
            },
            "correct": True,
        }

    def write_worker(self, payload: dict[str, object] | None = None) -> None:
        (self.source / "worker.log").write_bytes(
            acceptance.canonical_json(payload or self.worker_payload())
        )

    def trace_records(self) -> list[dict[str, object]]:
        common = {
            "schema": acceptance.TRACE_SCHEMA,
            "source": "upstream_rocr_kmt_aql",
            "daemon_uuid": "b" * 32,
            "job_uuid": self.job_uuid,
            "epoch": 1,
            "rank": 0,
            "world_size": 1,
            "connection_id": 77,
            "owner_fd": 10,
            "owner_generation": 1,
            "kernel_executed": True,
        }
        records: list[dict[str, object]] = []
        signals = [(1, 0), (0, 0), (0, 0), (0, 0)] + [(1, 0)] * 4
        for index in range(8):
            tick = (index + 1) * 100
            before, after = signals[index]
            records.append(
                {
                    **common,
                    "event": "native_execution_retired",
                    "sim_tick": tick,
                    "execution_ticket": index + 1,
                    "queue_index": acceptance.EXPECTED_QUEUE_INDEX[index],
                    "descriptor_abi": 3,
                    "kernarg_size": acceptance.EXPECTED_KERNARG_SIZE[index],
                    "grid": acceptance.EXPECTED_GRID[index],
                    "workgroup": acceptance.EXPECTED_WORKGROUP[index],
                    "signal_before": before,
                    "signal_after": after,
                    "source_request_id": 1000 + index,
                    "queue_object_id": 900,
                    "dispatch_id": 32,
                    "packet_fetches": 1,
                    "command_processor_submissions": 1,
                    "dispatcher_starts": 1,
                    "workgroups_completed": 1,
                    "doorbell_ack_durable": True,
                    "queue_retired": True,
                    "pins_released": True,
                    "cleanup_complete": False,
                    "start_tick": tick - 10,
                    "end_tick": tick,
                    "retire_tick": tick,
                }
            )
        records.append(
            {
                **common,
                "event": "native_execution_session_complete",
                "sim_tick": 1000,
                "retired_dispatches": 8,
                "owner_disconnected": True,
                "cleanup_complete": True,
            }
        )
        return records

    def write_trace(self, records: list[dict[str, object]] | None = None) -> None:
        payload = "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in (records or self.trace_records())
        )
        (self.source / "dispatch-trace.jsonl").write_text(payload, encoding="ascii")

    def write_stats(self, fallback: int = 0) -> None:
        (self.source / "m5out/stats.txt").write_text(
            "simTicks 1000\n"
            "finalTick 1000\n"
            "hostSeconds 1.25\n"
            f"system.host_gpu_bridge.host_fallback_count {fallback}\n",
            encoding="ascii",
        )

    def write_gem5_log(self) -> None:
        text = (
            f"host-gpu-ready endpoint={self.execution_root}/bridge.sock "
            f"daemon_uuid={'b' * 32} job_uuid={self.job_uuid} epoch=1 rank=0 world=1\n"
            "gem5 executing on test, pid 1234\n"
            f"command line: {' '.join(self.execution['gem5_argv'])}\n"
            "host-gpu-handshake status=OK fd=10 generation=1\n"
            "host-gpu-dispatch-exit cause=host GPU dispatch session complete "
            "code=0 tick=1000 stats=/tmp/stats.txt\n"
        )
        (self.source / "gem5.log").write_text(text, encoding="ascii")

    def write_manifest(self) -> None:
        artifacts = {
            relative: acceptance.base.artifact_record(self.source, relative)
            for relative in acceptance.SOURCE_ARTIFACTS
        }
        manifest = {
            "schema": acceptance.RUN_SCHEMA,
            "status": "success",
            "claim_scope": acceptance.CLAIM_SCOPE,
            "ordinary_upstream_pytorch_eager_executed": True,
            "runtime_gem5_bridge_modified_for_profile": False,
            "pytorch_rocm_multiop_accepted": False,
            "triton_upstream_amd_accepted": False,
            "torch_compile_accepted": False,
            "vllm_accepted": False,
            "sglang_accepted": False,
            "model_accepted": False,
            "identity_preflight": self.identity,
            "identity_postflight": self.identity,
            "execution": self.execution,
            "cleanup": self.cleanup,
            "artifacts": artifacts,
        }
        (self.source / "result-manifest.json").write_bytes(
            acceptance.canonical_json(manifest)
        )

    def test_full_synthetic_source_is_accepted(self) -> None:
        result = acceptance.validate_source(
            self.source, snapshot=lambda: self.identity
        )
        self.assertTrue(result["output_correct"])
        self.assertEqual(result["trace"]["retired_dispatches"], 8)
        self.assertEqual(result["stats"]["host_fallback_count"], 0)

    def test_worker_rejects_tensor_and_storage_tamper(self) -> None:
        for mutation in ("hash", "fresh", "nonalias"):
            payload = copy.deepcopy(self.worker_payload())
            if mutation == "hash":
                payload["tensor_contract"]["actual_sha256"]["sigmoid"] = "0" * 64
            else:
                payload["checks"][f"outputs_{mutation}"] = False
            self.write_worker(payload)
            with self.assertRaises(acceptance.AcceptanceError):
                acceptance.validate_worker(self.source / "worker.log", self.identity)

    def test_trace_rejects_route_order_and_cleanup_tamper(self) -> None:
        mutations = (
            (0, "source", "other"),
            (1, "queue_index", 9),
            (2, "execution_ticket", 99),
            (3, "doorbell_ack_durable", False),
            (4, "queue_retired", False),
            (8, "cleanup_complete", False),
        )
        for index, key, value in mutations:
            records = self.trace_records()
            records[index][key] = value
            self.write_trace(records)
            with self.assertRaises(acceptance.AcceptanceError):
                acceptance.parse_trace(
                    self.source / "dispatch-trace.jsonl", self.execution
                )

    def test_stats_and_cleanup_fail_closed(self) -> None:
        self.write_stats(fallback=1)
        with self.assertRaisesRegex(acceptance.AcceptanceError, "fallback"):
            acceptance.validate_stats(self.source / "m5out/stats.txt", 1000)
        cleanup = dict(self.cleanup)
        cleanup["gem5_forced_termination"] = True
        cleanup["all_clear"] = False
        with self.assertRaisesRegex(acceptance.AcceptanceError, "cleanup"):
            acceptance.validate_execution(self.execution, cleanup, self.identity)

    def test_quickstart_static_gate_rejects_project_hook(self) -> None:
        path = Path(self.temporary.name) / "bad.py"
        path.write_text(
            "import hashlib\nimport json\nimport torch\nimport gemsim_vllm\n"
            "torch.add(1, 2)\ntorch.sigmoid(torch.tensor(1))\ntorch.sum(torch.tensor(1))\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(acceptance.AcceptanceError, "upstream-only"):
            acceptance.validate_quickstart_source(path)


if __name__ == "__main__":
    unittest.main()
