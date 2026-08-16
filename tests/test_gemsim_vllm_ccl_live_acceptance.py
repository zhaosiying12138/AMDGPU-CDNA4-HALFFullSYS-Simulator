# SPDX-License-Identifier: GPL-3.0-or-later
"""Host-only synthetic tests for the vLLM communicator live verifier."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "tools/gemsim_vllm_ccl_live_acceptance.py"
BASE_TEST_PATH = ROOT / "tests/test_gemsim_ccl_live_allreduce_acceptance.py"


def load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


V = load("_test_gemsim_vllm_ccl_live_acceptance", VERIFIER_PATH)
OLD = load("_test_gemsim_vllm_ccl_live_base_fixture", BASE_TEST_PATH)


class SyntheticAdapterEvidence(OLD.SyntheticEvidence):
    def __init__(self, root: Path) -> None:
        super().__init__(root, world=2, count=3, dtype="bfloat16")
        self.convert_ranks()
        self.manifest = self.adapter_manifest()
        (self.source / "result-manifest.json").unlink()
        OLD.write_json(self.source / "result-manifest.json", self.manifest)
        self.make_private()

    def identities(self) -> dict:
        plugin = ROOT / "plugins/framework/gemsim_vllm/src/gemsim_vllm"
        checkout = ROOT / "projects/vllm/vllm"
        paths = {
            "product_manifest": self.product_prefix / "manifest.json",
            "source_lock": ROOT / "LICENSE",
            "runtime_library": self.runtime,
            "ccl_native": ROOT / "plugins/collectives/gemsim_ccl/src/gemsim_ccl/native.py",
            "ccl_device": ROOT / "plugins/collectives/gemsim_ccl/src/gemsim_ccl/device.py",
            "ccl_engine": ROOT / "plugins/collectives/gemsim_ccl/src/gemsim_ccl/engine.py",
            "triton_driver": ROOT / "plugins/triton/gemsim_amd/backend/driver.py",
            "gem5_binary": ROOT / "LICENSE",
            "gem5_config": ROOT / "README.md",
            "vllm_plugin_init": plugin / "__init__.py",
            "vllm_communicator": plugin / "communicator.py",
            "vllm_ccl_bootstrap": plugin / "ccl_bootstrap.py",
            "vllm_platform": plugin / "platform.py",
            "vllm_parallel_state": checkout / "distributed/parallel_state.py",
            "vllm_base_communicator": (
                checkout / "distributed/device_communicators/base_device_communicator.py"
            ),
            "vllm_communication_op": checkout / "distributed/communication_op.py",
            "vllm_version": checkout / "version.py",
            "vllm_metadata": ROOT / "projects/vllm/pyproject.toml",
            "vllm_checkout_parallel_state": checkout / "distributed/parallel_state.py",
            "vllm_checkout_base_communicator": (
                checkout / "distributed/device_communicators/base_device_communicator.py"
            ),
            "vllm_checkout_communication_op": checkout / "distributed/communication_op.py",
            "vllm_checkout_version": checkout / "version.py",
            "vllm_linear": checkout / "model_executor/layers/linear.py",
            "vllm_config_vllm": checkout / "config/vllm.py",
            "vllm_config_parallel": checkout / "config/parallel.py",
            "vllm_config_model": checkout / "config/model.py",
            "vllm_adapters": plugin / "adapters.py",
            "vllm_row_parallel": plugin / "row_parallel.py",
            "vllm_ops": plugin / "ops.py",
            "vllm_kernels": plugin / "kernels.py",
            "ccl_acceptance_base": V.BASE_VERIFIER_FILE,
            "verifier": V.THIS_FILE,
            "runner": V.RUNNER_FILE,
            "worker": V.WORKER_FILE,
            "bootstrap": V.BOOTSTRAP_FILE,
            "design": V.DESIGN_FILE,
            "rank_registry": ROOT / "scripts/gemsim_live_registry.py",
        }
        result = {}
        for role, path in paths.items():
            payload = path.read_bytes()
            result[role] = {
                "path": str(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        result["runtime_library"] = {
            "path": self.design["runtime"]["path"],
            "bytes": Path(self.design["runtime"]["path"]).stat().st_size,
            "sha256": self.design["runtime"]["sha256"],
        }
        return result

    def product_binding(self) -> dict:
        identity = self.identities()
        return {
            "product_id": self.product_id,
            "manifest_sha256": identity["product_manifest"]["sha256"],
            "prefix": str(self.product_prefix),
            "ccl_engine": identity["ccl_engine"],
            "vllm_plugin_init": identity["vllm_plugin_init"],
            "vllm_communicator": identity["vllm_communicator"],
        }

    def capability(self, rank: int) -> dict:
        descriptor = 70 + rank
        inode = 1000 + rank
        return {
            "fd": descriptor,
            "device": 9,
            "inode": inode,
            "mode": stat.S_IFSOCK | 0o700,
            "target": f"socket:[{inode}]",
        }

    def public_input(self, rank: int) -> bytes:
        return self.inputs[rank]

    def bootstrap(self, rank: int) -> dict:
        config = self.design["config"]
        identity = self.identities()
        return {
            "schema": V.BOOTSTRAP_SCHEMA,
            "product": {
                "prefix": str(self.product_prefix),
                "manifest": identity["product_manifest"],
                "runtime_library": identity["runtime_library"],
            },
            "groups": [{
                "unique_name": "tp:0",
                "identity": {
                    "world_size": self.world,
                    "epoch": config["epoch"],
                    "group_generation": config["group_generation"],
                    "job_uuid": config["job_uuid"],
                    "group_uuid": config["group_uuid"],
                    "model_identity_sha256": config["model_identity_sha256"],
                },
                "rank": {
                    "rank": rank,
                    "capability_fd": self.capability(rank)["fd"],
                    "broker_pid": 3000,
                    "broker_start_time_ticks": 22,
                    "join_timeout_ns": 1_000_000,
                    "collective_timeout_ns": 1_000_000,
                    "credits_per_peer": self.design["limits"]["credits_per_peer"],
                },
            }],
        }

    def events(self, rank: int) -> list[dict]:
        names = (
            "worker_started",
            "default_gloo_group_initialized",
            "coordinator_ready",
            "coordinator_all_reduce_returned",
            "cleanup_complete",
        )
        result = []
        for ordinal, event in enumerate(names):
            record = {
                "schema": V.EVENT_SCHEMA,
                "ordinal": ordinal,
                "rank": rank,
                "event": event,
            }
            if event == "coordinator_ready":
                record["unique_name"] = "tp:0"
            result.append(record)
        return result

    def adapter(self, rank: int, bootstrap_sha: str) -> dict:
        input_sha = hashlib.sha256(self.inputs[rank]).hexdigest()
        output_sha = hashlib.sha256(self.outputs[rank]).hexdigest()
        identities = self.identities()
        return {
            "schema": V.ADAPTER_SCHEMA,
            "rank": rank,
            "world_size": self.world,
            "entrypoint": "vllm.distributed.parallel_state.GroupCoordinator.all_reduce",
            "coordinator_class": "vllm.distributed.parallel_state.GroupCoordinator",
            "communicator_class": "gemsim_vllm.communicator.GemsimDeviceCommunicator",
            "platform_class": "gemsim_vllm.platform.GemsimPlatform",
            "unique_name": "tp:0",
            "control_backend": "gloo",
            "control_process_groups": {
                "default": "gloo", "device_group": "gloo", "cpu_group": "gloo"
            },
            "tensor_data_backend": "gemsim_ccl_engine",
            "message_queue_broadcaster": False,
            "use_custom_op_call": False,
            "coordinator_methods_unmodified": {
                "broadcast": "upstream-object-identity-preserved",
                "broadcast_tensor_dict": "upstream-object-identity-preserved",
            },
            "gloo_tensor_api_counts": {name: 0 for name in V.TENSOR_COLLECTIVE_APIS},
            "gloo_tensor_api_total": 0,
            "gloo_control_records": [],
            "capability_fd_identity": self.capability(rank),
            "bootstrap_descriptor_sha256": bootstrap_sha,
            "input_sha256_before": input_sha,
            "input_sha256_after": input_sha,
            "output_sha256": output_sha,
            "output_storage_fresh": True,
            "engine_rank": rank,
            "engine_world_size": self.world,
            "engine_state_after_collective": "ready",
            "actual_imports": {
                role: identities[role] for role in V.ACTUAL_IMPORT_ROLES
            },
            "vllm_installed_version": V.PINNED_VLLM_VERSION,
            "managed_session": copy.deepcopy(self.sessions[rank]),
            "coordinator_destroyed": True,
            "default_group_destroyed": True,
            "workload_evidence": {"kind": "standalone-allreduce"},
        }

    def convert_ranks(self) -> None:
        new_entries = []
        new_results = []
        for rank in range(self.world):
            directory = self.source / f"rank-{rank:02d}"
            (directory / "step-journal.jsonl").unlink()
            bootstrap = self.bootstrap(rank)
            bootstrap_payload = V.canonical_json(bootstrap)
            bootstrap_sha = hashlib.sha256(bootstrap_payload).hexdigest()
            adapter = self.adapter(rank, bootstrap_sha)
            adapter_payload = V.canonical_json(adapter)
            public_input = self.public_input(rank)
            input_sha = hashlib.sha256(public_input).hexdigest()
            output_sha = hashlib.sha256(self.outputs[rank]).hexdigest()
            result = {
                "schema": V.RANK_RESULT_SCHEMA,
                "status": "success",
                "rank": rank,
                "world_size": self.world,
                "acceptance_authority": False,
                "live_adapter_accepted": False,
                "public_result_published": True,
                "input_sha256_before": input_sha,
                "input_sha256_after": input_sha,
                "output_sha256": output_sha,
                "output_storage_fresh": True,
                "bootstrap_descriptor_sha256": bootstrap_sha,
                "adapter_evidence_sha256": hashlib.sha256(adapter_payload).hexdigest(),
                "managed_session": copy.deepcopy(self.sessions[rank]),
                "first_error": None,
                "product": self.product_binding(),
            }
            payloads = {
                "worker-result.json": V.canonical_json(result),
                "adapter-evidence.json": adapter_payload,
                "adapter-events.jsonl": b"".join(
                    V.canonical_json(item) for item in self.events(rank)
                ),
                "bootstrap-descriptor.json": bootstrap_payload,
                "input.bin": public_input,
                "output.bin": self.outputs[rank],
                "worker-stdout.log": b"",
                "worker-stderr.log": b"",
            }
            for name, payload in payloads.items():
                path = directory / name
                if path.exists():
                    path.unlink()
                path.write_bytes(payload)
            artifacts = {}
            for name in V.RANK_FILES:
                path = directory / name
                payload = path.read_bytes()
                artifacts[name] = {
                    "path": f"rank-{rank:02d}/{name}",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            capability = self.capability(rank)
            new_entries.append({
                "rank": rank,
                "worker_pid": None,
                "worker_start_time_ticks": None,
                "returncode": 0,
                "capability": {
                    "parent_fd_identity": capability,
                    "pass_fds": [capability["fd"]],
                    "bootstrap_descriptor_sha256": bootstrap_sha,
                },
                "artifacts": artifacts,
                "cleanup": {"worker_reaped": True, "daemon_reaped": True},
            })
            new_results.append(result)
        self.rank_entries = new_entries
        self.rank_results = new_results

    def adapter_manifest(self) -> dict:
        config = self.design["config"]
        workload = {
            "schema": V.WORKLOAD_SCHEMA,
            "kind": "standalone-allreduce",
            "model": None,
            "layer": None,
            "input": {"policy": "rank-affine-mod127-v1"},
            "collective": {"dtype": "bfloat16", "element_count": self.count},
        }
        workload_payload = V.canonical_json(workload)
        baseline_fds: list[dict] = []
        children = [{
            "rank": rank,
            "role": "daemon_or_descendant",
            "pid": self.sessions[rank]["child_pid"],
            "start_time_ticks": 100 + rank,
        } for rank in range(self.world)]
        return {
            "schema": V.RUN_SCHEMA,
            "status": "success",
            "acceptance_authority": False,
            "live_adapter_accepted": False,
            "expected": {
                "schema": V.EXPECTED_SCHEMA,
                "bytes": self.expected_record["bytes"],
                "sha256": self.expected_record["sha256"],
            },
            "workload": {
                "schema": V.WORKLOAD_SCHEMA,
                "bytes": len(workload_payload),
                "sha256": hashlib.sha256(workload_payload).hexdigest(),
                "document": workload,
            },
            "world_size": self.world,
            "element_count": self.count,
            "dtype": "bfloat16",
            "unique_name": "tp:0",
            "job_uuid": config["job_uuid"],
            "group_uuid": config["group_uuid"],
            "epoch": config["epoch"],
            "group_generation": config["group_generation"],
            "started_at_ns": 100,
            "completed_at_ns": 200,
            "absolute_deadline_ns": 1000,
            "target_execution_completed": True,
            "target_feedback": False,
            "oracle_phase": "post_target",
            "oracle_feedback": False,
            "first_error": None,
            "supervisor_cleanup": {
                "baseline_fds": baseline_fds,
                "baseline_fd_count": 0,
                "baseline_fd_sha256": V.object_sha256(baseline_fds),
                "post_fds": [],
                "post_fd_count": 0,
                "post_fd_sha256": V.object_sha256(baseline_fds),
                "added_fds": [],
                "removed_fds": [],
                "measured_fd_delta": 0,
                "children_exhausted": True,
                "workers_reaped": True,
                "new_child_identities": children,
                "orphan_identities": [],
                "all_clear": True,
            },
            "source_identity_preflight": copy.deepcopy(self.identities()),
            "source_identity_postflight": copy.deepcopy(self.identities()),
            "vllm_checkout": {
                "path": str(V.VLLM_CHECKOUT.resolve(strict=True)),
                "head": V.PINNED_VLLM_HEAD,
                "tree": V.PINNED_VLLM_TREE,
                "tracked_clean": True,
                "installed_version": V.PINNED_VLLM_VERSION,
            },
            "ranks": self.rank_entries,
        }

    def make_private(self) -> None:
        for path in self.source.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o600)

    def rebind_adapter(self, rank: int, adapter: dict) -> None:
        path = self.source / f"rank-{rank:02d}/adapter-evidence.json"
        path.write_bytes(V.canonical_json(adapter))
        os.chmod(path, 0o600)
        self.rebind(rank, "adapter-evidence.json")
        result_path = self.source / f"rank-{rank:02d}/worker-result.json"
        result = json.loads(result_path.read_text(encoding="ascii"))
        result["adapter_evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result_path.write_bytes(V.canonical_json(result))
        os.chmod(result_path, 0o600)
        self.rebind(rank, "worker-result.json")

    def rewrite_manifest(self) -> None:
        path = self.source / "result-manifest.json"
        if path.exists():
            path.unlink()
        OLD.write_json(path, self.manifest)
        os.chmod(path, 0o600)


class SyntheticRowParallelEvidence(SyntheticAdapterEvidence):
    """Schema-only RowParallel evidence built from the real pinned tensor."""

    def __init__(self, root: Path) -> None:
        model_root = ROOT / "models/Qwen3.5-0.8B"
        self.row_workload = V.build_row_parallel_workload(model_root)
        self.local_partials, _ = V.row_parallel_oracles(self.row_workload)
        from safetensors import safe_open

        with safe_open(
            self.row_workload["model"]["weight_shard"]["path"],
            framework="pt", device="cpu",
        ) as tensors:
            weight = tensors.get_tensor(self.row_workload["model"]["tensor_key"])
        self.weight_shard_shas = [
            hashlib.sha256(
                weight[:, rank * 1792:(rank + 1) * 1792]
                .contiguous().view(__import__("torch").uint8).numpy().tobytes()
            ).hexdigest()
            for rank in range(2)
        ]
        OLD.SyntheticEvidence.__init__(
            self,
            root,
            world=2,
            count=1024,
            dtype="bfloat16",
            model_identity_sha256=V.object_sha256(self.row_workload),
        )
        self.convert_ranks()
        self.manifest = self.adapter_manifest()
        (self.source / "result-manifest.json").unlink()
        OLD.write_json(self.source / "result-manifest.json", self.manifest)
        self.make_private()

    def input_bytes(self, rank: int) -> bytes:
        return self.local_partials[rank]

    def public_input(self, rank: int) -> bytes:
        return V.row_parallel_input(rank)

    def trace(self, rank: int) -> list[dict]:
        records = super().trace(rank)
        if len(records) != 3:
            raise AssertionError("synthetic TP2 RowParallel expects one SUM dispatch")
        for record in records:
            for name in ("sim_tick", "admission_tick", "start_tick", "end_tick", "retire_tick"):
                record[name] += 200
            for name in ("request_id", "trace_id", "ticket_id", "dispatch_id"):
                record[name] += 1

        dense = copy.deepcopy(records[0])
        for name in ("sim_tick", "admission_tick", "start_tick", "end_tick", "retire_tick"):
            dense[name] -= 200
        for name in ("request_id", "trace_id", "ticket_id", "dispatch_id"):
            dense[name] -= 1
        dense.update({
            "kernel": "dense_linear_kernel",
            "image_sha256": "b" * 64,
            "grid": [256, 64, 1],
            "workgroup": [256, 1, 1],
            "allocation_count": 3,
            "workgroups_completed": 64,
            "global_reads": 4096,
            "global_writes": 512,
            "store_dwords": 512,
        })
        dense["allocations"] = copy.deepcopy(dense["allocations"]) + [{
            "allocation_id": 3,
            "generation": 3,
            "gpu_va": 0x5000,
            "bytes": 2048,
        }]
        durable = copy.deepcopy(dense)
        durable["event"] = "generic_execution_type20_durable"
        durable["type20_durable"] = True
        completion = copy.deepcopy(durable)
        completion["event"] = "generic_execution_reuse_complete"
        completion["sim_tick"] = records[0]["admission_tick"]
        return [dense, durable, completion, *records]

    def events(self, rank: int) -> list[dict]:
        return [{
            "schema": V.EVENT_SCHEMA,
            "ordinal": ordinal,
            "rank": rank,
            "event": event,
        } for ordinal, event in enumerate((
            "worker_started",
            "standard_model_parallel_initialized",
            "row_parallel_layer_ready",
            "upstream_row_parallel_forward_returned",
            "cleanup_complete",
        ))]

    @staticmethod
    def process_group_audit(rank: int) -> dict:
        return {
            "schema": V.PROCESS_GROUP_AUDIT_SCHEMA,
            "init": [{
                "ordinal": 0,
                "phase": "initialization",
                "backend": "gloo",
                "rank": rank,
                "world_size": 2,
            }],
            "new": [{
                "ordinal": 0,
                "phase": "initialization",
                "backend": "gloo",
                "ranks": [0, 1],
                "local_member": True,
                "local_token": 1,
            }],
            "destroy": [
                {"ordinal": 0, "phase": "cleanup", "target": 1},
                {"ordinal": 1, "phase": "cleanup", "target": "default"},
            ],
            "local_tokens_created": [1],
            "local_tokens_destroyed": [1],
            "default_destroyed": True,
            "all_local_groups_destroyed": True,
        }

    def adapter(self, rank: int, bootstrap_sha: str) -> dict:
        public_input = self.public_input(rank)
        input_sha = hashlib.sha256(public_input).hexdigest()
        output_sha = hashlib.sha256(self.outputs[rank]).hexdigest()
        partial = self.local_partials[rank]
        identities = self.identities()
        counts = {name: 0 for name in V.TENSOR_COLLECTIVE_APIS}
        counts["barrier"] = 1
        return {
            "schema": V.ADAPTER_SCHEMA,
            "rank": rank,
            "world_size": 2,
            "entrypoint": "vllm.model_executor.layers.linear.RowParallelLinear.forward",
            "coordinator_class": "vllm.distributed.parallel_state.GroupCoordinator",
            "communicator_class": "gemsim_vllm.communicator.GemsimDeviceCommunicator",
            "platform_class": "gemsim_vllm.platform.GemsimPlatform",
            "unique_name": "tp:0",
            "control_backend": "gloo",
            "control_process_groups": self.process_group_audit(rank),
            "tensor_data_backend": "gemsim_ccl_engine",
            "message_queue_broadcaster": True,
            "use_custom_op_call": False,
            "coordinator_methods_unmodified": {
                "broadcast": "upstream-object-identity-preserved",
                "broadcast_tensor_dict": "upstream-object-identity-preserved",
            },
            "gloo_tensor_api_counts": counts,
            "gloo_tensor_api_total": 1,
            "gloo_control_records": [{
                "api": "barrier",
                "phase": "initialization",
                "dtype": "none",
                "shape": [],
                "bytes": 0,
            }],
            "capability_fd_identity": self.capability(rank),
            "bootstrap_descriptor_sha256": bootstrap_sha,
            "input_sha256_before": input_sha,
            "input_sha256_after": input_sha,
            "output_sha256": output_sha,
            "output_storage_fresh": True,
            "engine_rank": rank,
            "engine_world_size": 2,
            "engine_state_after_collective": "ready",
            "actual_imports": {
                role: identities[role] for role in V.ROW_PARALLEL_IMPORT_ROLES
            },
            "vllm_installed_version": V.PINNED_VLLM_VERSION,
            "managed_session": copy.deepcopy(self.sessions[rank]),
            "coordinator_destroyed": True,
            "default_group_destroyed": True,
            "workload_evidence": {
                "kind": "vllm-row-parallel",
                "layer_class": "gemsim_vllm.adapters.GemsimRowParallelLinear",
                "forward_inherited": True,
                "loader": "vllm RowParallelLinear parameter weight_loader hook",
                "loaded_parameters": ["weight"],
                "weight_shard_columns": [rank * 1792, (rank + 1) * 1792],
                "weight_shard_sha256_before": self.weight_shard_shas[rank],
                "weight_shard_sha256_after": self.weight_shard_shas[rank],
                "local_projection": {
                    "schema": V.DISPATCH_CAPTURE_SCHEMA,
                    "operator": V.ROW_PARALLEL_LOCAL_OPERATOR,
                    "dtype": "bfloat16",
                    "shape": [1, 1024],
                    "bytes": len(partial),
                    "sha256": hashlib.sha256(partial).hexdigest(),
                    "payload_hex": partial.hex(),
                },
            },
        }

    def adapter_manifest(self) -> dict:
        manifest = super().adapter_manifest()
        payload = V.canonical_json(self.row_workload)
        manifest["workload"] = {
            "schema": V.WORKLOAD_SCHEMA,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "document": self.row_workload,
        }
        return manifest


class VllmCCLLiveAcceptanceTest(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.fixture = SyntheticAdapterEvidence(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def verify(self):
        return V.verify(
            self.fixture.source,
            self.fixture.expected_path,
            self.fixture.output,
            live_identity=False,
        )

    def test_static_call_chain_uses_frozen_shared_ccl_bootstrap(self) -> None:
        from scripts import run_gemsim_vllm_ccl_live as runner

        _, paths = runner.load_product(runner.base.default_product_prefix())
        identity = runner.source_identity(paths)
        V.validate_static_call_chain(identity)

    def test_n2_success_is_authoritative_and_claim_is_standalone_only(self) -> None:
        result = self.verify()
        self.assertTrue(result["live_adapter_accepted"])
        self.assertEqual(result["gloo_tensor_api_total"], 0)
        self.assertEqual(result["host_reduction_count"], 0)
        self.assertIn("not vLLM tensor-parallel model", result["claim_boundary"])

    def test_row_parallel_n2_full_verifier_separates_dense_and_collective(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fixture = SyntheticRowParallelEvidence(Path(temporary))
            result = V.verify(
                fixture.source, fixture.expected_path, fixture.output,
                live_identity=False,
            )
            self.assertTrue(result["live_adapter_accepted"])
            comparison = result["row_parallel_comparison"]
            self.assertEqual(len(comparison["local_projection_by_rank"]), 2)
            self.assertTrue(comparison["collective_actual_partial_bitwise"])
            self.assertEqual(
                comparison["final_model_oracle"]["mismatch_count"], 0
            )
            self.assertIn("not full-model", result["claim_boundary"])

    def test_row_parallel_rejects_process_group_and_local_partial_forgery(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fixture = SyntheticRowParallelEvidence(Path(temporary))
            path = fixture.source / "rank-00/adapter-evidence.json"
            adapter = json.loads(path.read_text(encoding="ascii"))
            adapter["control_process_groups"]["new"][0]["backend"] = "nccl"
            fixture.rebind_adapter(0, adapter)
            with self.assertRaisesRegex(V.AcceptanceError, "Gloo new_group"):
                V.verify(
                    fixture.source, fixture.expected_path, fixture.output,
                    live_identity=False,
                )

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fixture = SyntheticRowParallelEvidence(Path(temporary))
            path = fixture.source / "rank-00/adapter-evidence.json"
            adapter = json.loads(path.read_text(encoding="ascii"))
            capture = adapter["workload_evidence"]["local_projection"]
            payload = bytearray.fromhex(capture["payload_hex"])
            payload[0:2] = b"\x7f\x7f"
            capture["payload_hex"] = payload.hex()
            capture["sha256"] = hashlib.sha256(payload).hexdigest()
            fixture.rebind_adapter(0, adapter)
            with self.assertRaisesRegex(V.AcceptanceError, "local projection"):
                V.verify(
                    fixture.source, fixture.expected_path, fixture.output,
                    live_identity=False,
                )

    def test_rejects_false_worker_acceptance_claim(self) -> None:
        path = self.fixture.source / "rank-00/worker-result.json"
        result = json.loads(path.read_text(encoding="ascii"))
        result["acceptance_authority"] = True
        path.write_bytes(V.canonical_json(result))
        os.chmod(path, 0o600)
        self.fixture.rebind(0, "worker-result.json")
        with self.assertRaisesRegex(V.AcceptanceError, "authority"):
            self.verify()

    def test_rejects_each_forged_actual_import_even_with_rebound_hashes(self) -> None:
        for role in V.ACTUAL_IMPORT_ROLES:
            with self.subTest(role=role), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                fixture = SyntheticAdapterEvidence(Path(temporary))
                path = fixture.source / "rank-00/adapter-evidence.json"
                adapter = json.loads(path.read_text(encoding="ascii"))
                adapter["actual_imports"][role]["sha256"] = "0" * 64
                fixture.rebind_adapter(0, adapter)
                with self.assertRaisesRegex(V.AcceptanceError, "actual imports"):
                    V.verify(fixture.source, fixture.expected_path, fixture.output,
                             live_identity=False)

    def test_rejects_pinned_version_and_gloo_tensor_attempt(self) -> None:
        path = self.fixture.source / "rank-00/adapter-evidence.json"
        adapter = json.loads(path.read_text(encoding="ascii"))
        adapter["vllm_installed_version"] = "0.0.dev0+forged"
        self.fixture.rebind_adapter(0, adapter)
        with self.assertRaisesRegex(V.AcceptanceError, "version"):
            self.verify()

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fixture = SyntheticAdapterEvidence(Path(temporary))
            path = fixture.source / "rank-00/adapter-evidence.json"
            adapter = json.loads(path.read_text(encoding="ascii"))
            adapter["gloo_tensor_api_counts"]["all_reduce"] = 1
            adapter["gloo_tensor_api_total"] = 1
            fixture.rebind_adapter(0, adapter)
            with self.assertRaisesRegex(V.AcceptanceError, "Gloo"):
                V.verify(fixture.source, fixture.expected_path, fixture.output,
                         live_identity=False)

    def test_rejects_non_singleton_pass_fds_and_bootstrap_rank_forgery(self) -> None:
        self.fixture.manifest["ranks"][0]["capability"]["pass_fds"].append(99)
        self.fixture.rewrite_manifest()
        with self.assertRaisesRegex(V.AcceptanceError, "pass_fds"):
            self.verify()

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fixture = SyntheticAdapterEvidence(Path(temporary))
            path = fixture.source / "rank-00/bootstrap-descriptor.json"
            bootstrap = json.loads(path.read_text(encoding="ascii"))
            bootstrap["groups"][0]["rank"]["rank"] = 1
            path.write_bytes(V.canonical_json(bootstrap))
            os.chmod(path, 0o600)
            fixture.rebind(0, "bootstrap-descriptor.json")
            new_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            fixture.manifest["ranks"][0]["capability"][
                "bootstrap_descriptor_sha256"
            ] = new_sha
            adapter_path = fixture.source / "rank-00/adapter-evidence.json"
            adapter = json.loads(adapter_path.read_text(encoding="ascii"))
            adapter["bootstrap_descriptor_sha256"] = new_sha
            fixture.rebind_adapter(0, adapter)
            result_path = fixture.source / "rank-00/worker-result.json"
            result = json.loads(result_path.read_text(encoding="ascii"))
            result["bootstrap_descriptor_sha256"] = new_sha
            result_path.write_bytes(V.canonical_json(result))
            os.chmod(result_path, 0o600)
            fixture.rebind(0, "worker-result.json")
            fixture.manifest["ranks"][0]["capability"]["pass_fds"] = [
                fixture.capability(0)["fd"]
            ]
            fixture.rewrite_manifest()
            with self.assertRaisesRegex(V.AcceptanceError, "capability binding"):
                V.verify(fixture.source, fixture.expected_path, fixture.output,
                         live_identity=False)

    def test_rejects_oracle_mismatch_after_all_reported_hashes_are_forged(self) -> None:
        path = self.fixture.source / "rank-00/output.bin"
        payload = bytearray(path.read_bytes())
        payload[0] ^= 1
        path.write_bytes(payload)
        os.chmod(path, 0o600)
        self.fixture.rebind(0, "output.bin")
        output_sha = hashlib.sha256(payload).hexdigest()
        adapter_path = self.fixture.source / "rank-00/adapter-evidence.json"
        adapter = json.loads(adapter_path.read_text(encoding="ascii"))
        adapter["output_sha256"] = output_sha
        self.fixture.rebind_adapter(0, adapter)
        result_path = self.fixture.source / "rank-00/worker-result.json"
        result = json.loads(result_path.read_text(encoding="ascii"))
        result["output_sha256"] = output_sha
        result_path.write_bytes(V.canonical_json(result))
        os.chmod(result_path, 0o600)
        self.fixture.rebind(0, "worker-result.json")
        with self.assertRaisesRegex(V.AcceptanceError, "ring-step oracle"):
            self.verify()

    def test_rejects_forged_but_self_consistent_input_stimulus(self) -> None:
        path = self.fixture.source / "rank-00/input.bin"
        payload = bytearray(path.read_bytes())
        payload[0] ^= 1
        path.write_bytes(payload)
        os.chmod(path, 0o600)
        self.fixture.rebind(0, "input.bin")
        input_sha = hashlib.sha256(payload).hexdigest()
        adapter_path = self.fixture.source / "rank-00/adapter-evidence.json"
        adapter = json.loads(adapter_path.read_text(encoding="ascii"))
        adapter["input_sha256_before"] = input_sha
        adapter["input_sha256_after"] = input_sha
        self.fixture.rebind_adapter(0, adapter)
        result_path = self.fixture.source / "rank-00/worker-result.json"
        result = json.loads(result_path.read_text(encoding="ascii"))
        result["input_sha256_before"] = input_sha
        result["input_sha256_after"] = input_sha
        result_path.write_bytes(V.canonical_json(result))
        os.chmod(result_path, 0o600)
        self.fixture.rebind(0, "worker-result.json")
        with self.assertRaisesRegex(V.AcceptanceError, "deterministic stimulus"):
            self.verify()

    def test_rejects_child_capability_trace_and_cleanup_tamper(self) -> None:
        path = self.fixture.source / "rank-00/adapter-evidence.json"
        adapter = json.loads(path.read_text(encoding="ascii"))
        adapter["capability_fd_identity"]["inode"] += 1
        adapter["capability_fd_identity"]["target"] = (
            f"socket:[{adapter['capability_fd_identity']['inode']}]"
        )
        self.fixture.rebind_adapter(0, adapter)
        with self.assertRaisesRegex(V.AcceptanceError, "inherited capability"):
            self.verify()

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fixture = SyntheticAdapterEvidence(Path(temporary))
            path = fixture.source / "rank-00/dispatch-trace.jsonl"
            records = V.BASE.parse_trace_jsonl(path.read_bytes(), "trace")
            records[0]["global_writes"] += 1
            path.write_bytes(b"".join(V.BASE.trace_json(item) for item in records))
            os.chmod(path, 0o600)
            fixture.rebind(0, "dispatch-trace.jsonl")
            with self.assertRaisesRegex(V.AcceptanceError, "retired/type20"):
                V.verify(fixture.source, fixture.expected_path, fixture.output,
                         live_identity=False)

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fixture = SyntheticAdapterEvidence(Path(temporary))
            fixture.manifest["supervisor_cleanup"]["all_clear"] = False
            fixture.rewrite_manifest()
            with self.assertRaisesRegex(V.AcceptanceError, "all-clear"):
                V.verify(fixture.source, fixture.expected_path, fixture.output,
                         live_identity=False)


if __name__ == "__main__":
    unittest.main()
