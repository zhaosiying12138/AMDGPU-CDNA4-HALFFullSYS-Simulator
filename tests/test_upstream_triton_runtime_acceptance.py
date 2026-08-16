from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools import upstream_triton_runtime_acceptance as acceptance


class UpstreamTritonRuntimeAcceptanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.source = Path(self.temporary.name) / "source"
        self.source.mkdir()
        (self.source / "m5out").mkdir()
        self.execution_root = Path("/tmp/gs-upstream-triton-fixture")
        self.job_uuid = "a" * 32
        self.daemon_uuid = "b" * 32
        self.identity = {
            "product": {
                "native_role": "rocr_kmd_boundary_only",
                "runtime_probe": {
                    "torch": "2.11.0+rocm",
                    "torch_hip": "7.13",
                    "triton": "3.6.0",
                },
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
        self.compiled = self.write_cache()
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
        gem5_argv = acceptance.lifecycle.gem5_argv(
            binary=acceptance.GEM5_BINARY,
            config=acceptance.GEM5_CONFIG,
            execution_root=self.execution_root,
            endpoint=self.execution_root / "bridge.sock",
            trace_path=self.execution_root / "dispatch-trace.jsonl",
            job_uuid=self.job_uuid,
        )
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
            "worker_argv": acceptance.expected_worker_argv(self.identity),
            "gem5_environment": {},
            "worker_environment": {
                "SAGR_GENERIC_BRIDGE_ENDPOINT": str(
                    self.execution_root / "bridge.sock"
                ),
                "GEMSIM_RUN_TRITON_CACHE_DIR": str(
                    self.execution_root / "triton-cache"
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
            "worker_timeout_seconds": 180,
            "gem5_exit_timeout_seconds": 30,
            "startup_timeout_seconds": 30,
        }

    def compiled_record(self, name: str, binary: bytes, shared: int) -> dict[str, object]:
        return {
            "name": name,
            "cache_hash": hashlib.sha256(name.encode("ascii")).hexdigest(),
            "binary_bytes": len(binary),
            "binary_sha256": hashlib.sha256(binary).hexdigest(),
            "target": {"backend": "hip", "arch": "gfx950", "warp_size": 64},
            "num_warps": 4,
            "shared_memory_bytes": shared,
        }

    def write_cache(self) -> list[dict[str, object]]:
        compiled: list[dict[str, object]] = []
        keys = ("A" * 52, "B" * 52, "C" * 52)
        shared = (0, 0, 16)
        for name, key, shared_bytes in zip(
            acceptance.KERNEL_NAMES, keys, shared
        ):
            directory = self.source / "triton-cache" / key
            directory.mkdir(parents=True)
            binary = b"\x7fELF" + name.encode("ascii")
            record = self.compiled_record(name, binary, shared_bytes)
            compiled.append(record)
            for suffix in acceptance.KERNEL_CACHE_SUFFIXES:
                path = directory / f"{name}{suffix}"
                if suffix == ".hsaco":
                    path.write_bytes(binary)
                elif suffix == ".json":
                    path.write_text(
                        json.dumps(
                            {
                                "name": name,
                                "hash": record["cache_hash"],
                                "target": record["target"],
                                "arch": "gfx950",
                                "backend_name": "hip",
                                "triton_version": "3.6.0",
                                "num_warps": 4,
                                "shared": shared_bytes,
                            }
                        ),
                        encoding="ascii",
                    )
                else:
                    path.write_text(f"module {name} stage {suffix}\n", encoding="ascii")
            children = {
                f"{name}{suffix}": str(
                    self.execution_root
                    / "triton-cache"
                    / key
                    / f"{name}{suffix}"
                )
                for suffix in acceptance.KERNEL_CACHE_SUFFIXES
            }
            (directory / f"__grp__{name}.json").write_text(
                json.dumps({"child_paths": children}), encoding="ascii"
            )
        for key, name in zip(
            ("D" * 52, "E" * 52, "F" * 52),
            (
                "__triton_launcher.cpython-312-x86_64-linux-gnu.so",
                "__triton_launcher.cpython-312-x86_64-linux-gnu.so",
                "__triton_launcher.cpython-312-x86_64-linux-gnu.so",
            ),
        ):
            directory = self.source / "triton-cache" / key
            directory.mkdir(parents=True)
            (directory / name).write_bytes(b"\x7fELFlauncher" + key[:1].encode())
        directory = self.source / "triton-cache" / ("G" * 52)
        directory.mkdir(parents=True)
        (directory / "hip_utils.cpython-312-x86_64-linux-gnu.so").write_bytes(
            b"\x7fELFhip-utils"
        )
        return compiled

    def worker_payload(self) -> dict[str, object]:
        return {
            "schema": acceptance.QUICKSTART_SCHEMA,
            "torch": "2.11.0+rocm",
            "torch_hip": "7.13",
            "triton": "3.6.0",
            "device_count": 1,
            "device_name": "AMD Instinct MI350X",
            "capability": [9, 5],
            "driver": {
                "module": "triton.backends.amd.driver",
                "class": "HIPDriver",
                "backend": "hip",
                "arch": "gfx950",
                "warp_size": 64,
            },
            "kernels": list(acceptance.KERNEL_NAMES),
            "compiled_kernels": self.compiled,
            "tensor_contract": {
                "dtype": "float32",
                "element_count": 256,
                "input_actual_sha256": acceptance.EXPECTED_INPUT_SHA256,
                "input_expected_sha256": acceptance.EXPECTED_INPUT_SHA256,
                "actual_sha256": acceptance.EXPECTED_OUTPUT_SHA256,
                "expected_sha256": acceptance.EXPECTED_OUTPUT_SHA256,
            },
            "checks": {
                "add_bitwise": True,
                "transform_bitwise": True,
                "reduce_bitwise": True,
                "inputs_unchanged": True,
                "outputs_are_cuda": True,
                "outputs_nonalias": True,
            },
            "target_feedback_from_oracle": False,
            "correct": True,
        }

    def write_worker(self, payload: dict[str, object] | None = None) -> None:
        (self.source / "worker.log").write_bytes(
            b"upstream compiler diagnostic\n"
            + acceptance.canonical_json(payload or self.worker_payload())
        )

    def trace_records(self) -> list[dict[str, object]]:
        common = {
            "schema": acceptance.TRACE_SCHEMA,
            "source": "upstream_rocr_kmt_aql",
            "daemon_uuid": self.daemon_uuid,
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
        signals = [(1, 0)] * 3 + [(0, 0)] * 3 + [(1, 0)] * 6
        kernel_objects = [100, 100, 100, 200, 201, 202] + [100] * 6
        for index in range(12):
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
                    "workgroups_completed": acceptance.EXPECTED_WORKGROUP_COMPLETIONS[index],
                    "signal_before": before,
                    "signal_after": after,
                    "source_request_id": 1000 + index,
                    "queue_object_id": 900,
                    "kernel_object": kernel_objects[index],
                    "dispatch_id": 32,
                    "packet_fetches": 1,
                    "command_processor_submissions": 1,
                    "dispatcher_starts": 1,
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
                "sim_tick": 1300,
                "retired_dispatches": 12,
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
            "simTicks 1300\n"
            "finalTick 1300\n"
            "hostSeconds 1.25\n"
            f"system.host_gpu_bridge.host_fallback_count {fallback}\n",
            encoding="ascii",
        )

    def write_gem5_log(self) -> None:
        text = (
            f"host-gpu-ready endpoint={self.execution_root / 'bridge.sock'} "
            f"daemon_uuid={self.daemon_uuid} job_uuid={self.job_uuid} "
            "epoch=1 rank=0 world=1 max_record=65536\n"
            f"gem5 executing on host, pid {self.execution['gem5_pid']}\n"
            "command line: "
            + " ".join(self.execution["gem5_argv"])
            + "\n"
            + "host-gpu-handshake status=OK fd=10 generation=1\n"
            + "host-gpu-dispatch-exit cause=host GPU dispatch session complete "
            + "code=0 tick=1300 stats=/tmp/stats.txt\n"
        )
        (self.source / "gem5.log").write_text(text, encoding="ascii")

    def write_manifest(self) -> None:
        paths = set(acceptance.CORE_ARTIFACTS) | set(
            acceptance.cache_artifact_paths(self.source)
        )
        artifacts = {
            relative: acceptance.base.artifact_record(self.source, relative)
            for relative in paths
        }
        manifest = {
            "schema": acceptance.RUN_SCHEMA,
            "status": "success",
            "claim_scope": acceptance.CLAIM_SCOPE,
            "ordinary_upstream_triton_amd_executed": True,
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

    def test_full_synthetic_source_is_accepted_and_published(self) -> None:
        validation = acceptance.validate_source(
            self.source, snapshot=lambda: self.identity
        )
        self.assertTrue(validation["output_correct"])
        self.assertEqual(validation["trace"]["retired_dispatches"], 12)
        self.assertEqual(validation["jit_cache"]["file_count"], 28)
        output = Path(self.temporary.name) / "accepted"
        result = acceptance.publish(self.source, output, validation)
        self.assertEqual(result["status"], "accepted")
        self.assertTrue(result["unchanged_upstream_triton_amd_multiop_accepted"])
        self.assertFalse(result["torch_compile_accepted"])
        self.assertTrue((output / "manifest.json").is_file())

    def test_worker_rejects_self_consistent_output_tamper(self) -> None:
        payload = self.worker_payload()
        payload["tensor_contract"]["actual_sha256"] = {
            **acceptance.EXPECTED_OUTPUT_SHA256,
            "add": "f" * 64,
        }
        payload["tensor_contract"]["expected_sha256"] = payload[
            "tensor_contract"
        ]["actual_sha256"]
        self.write_worker(payload)
        with self.assertRaisesRegex(acceptance.AcceptanceError, "output oracle"):
            acceptance.validate_worker(self.source / "worker.log", self.identity)

    def test_worker_log_requires_terminal_canonical_ascii_json(self) -> None:
        payload = acceptance.canonical_json(self.worker_payload())
        worker_log = self.source / "worker.log"
        worker_log.write_bytes(payload + b"late diagnostic\n")
        with self.assertRaisesRegex(acceptance.AcceptanceError, "terminal line"):
            acceptance.validate_worker(worker_log, self.identity)
        worker_log.write_bytes(b"compiler \xff\n" + payload)
        with self.assertRaisesRegex(acceptance.AcceptanceError, "not ASCII"):
            acceptance.validate_worker(worker_log, self.identity)

    def test_jit_cache_rejects_hsaco_tamper(self) -> None:
        worker = acceptance.validate_worker(self.source / "worker.log", self.identity)
        cache_paths = acceptance.cache_artifact_paths(self.source)
        hsaco = next((self.source / "triton-cache").rglob("add_kernel.hsaco"))
        hsaco.write_bytes(hsaco.read_bytes() + b"x")
        with self.assertRaisesRegex(acceptance.AcceptanceError, "binary size"):
            acceptance.validate_jit_cache(
                self.source, cache_paths, self.execution, worker
            )

    def test_trace_rejects_kernel_splice_and_cleanup_tamper(self) -> None:
        records = self.trace_records()
        records[4]["kernel_object"] = records[3]["kernel_object"]
        self.write_trace(records)
        with self.assertRaisesRegex(acceptance.AcceptanceError, "not distinct"):
            acceptance.parse_trace(
                self.source / "dispatch-trace.jsonl", self.execution
            )
        records = self.trace_records()
        records[-1]["cleanup_complete"] = False
        self.write_trace(records)
        with self.assertRaisesRegex(acceptance.AcceptanceError, "terminal cleanup"):
            acceptance.parse_trace(
                self.source / "dispatch-trace.jsonl", self.execution
            )

    def test_stats_and_private_cache_fail_closed(self) -> None:
        self.write_stats(fallback=1)
        with self.assertRaisesRegex(acceptance.AcceptanceError, "fallback"):
            acceptance.validate_stats(self.source / "m5out/stats.txt", 1300)
        self.execution["worker_environment"]["GEMSIM_RUN_TRITON_CACHE_DIR"] = (
            "/tmp/shared-cache"
        )
        with self.assertRaisesRegex(acceptance.AcceptanceError, "private cache"):
            acceptance.validate_execution(
                self.execution, self.cleanup, self.identity
            )

    def test_quickstart_static_gate_rejects_project_hook(self) -> None:
        source = Path(self.temporary.name) / "bad.py"
        source.write_text(
            acceptance.QUICKSTART.read_text(encoding="ascii")
            + "\n# gemsim project hook\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(acceptance.AcceptanceError, "forbidden hook"):
            acceptance.validate_quickstart_source(source)


if __name__ == "__main__":
    unittest.main()
