# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "examples/triton/vllm_ccl_live_rank.py"


def load_worker():
    spec = importlib.util.spec_from_file_location("_test_vllm_ccl_worker", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WORKER = load_worker()


def run_gloo_audit_child(rank: int, rendezvous: Path, output: Path) -> None:
    import torch.distributed as dist

    with WORKER.audit_standard_vllm_control(dist, 2, rank) as audit:
        dist.init_process_group(
            "gloo", rendezvous.as_uri(), None, 2, rank
        )
        all_ranks = dist.new_group(ranks=[0, 1], backend="gloo")
        rank_zero = dist.new_group(ranks=[0], backend="gloo")
        rank_one = dist.new_group(ranks=[1], backend="gloo")
        audit["phase"]["phase"] = "model_ready"
        audit["phase"]["phase"] = "cleanup"
        local_singleton = rank_zero if rank == 0 else rank_one
        dist.destroy_process_group(local_singleton)
        dist.destroy_process_group(all_ranks)
        dist.destroy_process_group()
    output.write_bytes(WORKER.canonical_json(audit["process_groups"]))


class FakeDist:
    class GroupMember:
        NON_GROUP_MEMBER = object()

    class group:
        WORLD = object()

    def __init__(self):
        self.groups = []
        self.destroyed = []

    def init_process_group(self, backend=None, rank=-1, world_size=-1, **_kwargs):
        self.default = (backend, rank, world_size)

    def new_group(self, ranks=None, timeout=None, backend=None, **_kwargs):
        del timeout, backend
        if 0 not in ranks:
            return self.GroupMember.NON_GROUP_MEMBER
        group = object()
        self.groups.append(group)
        return group

    def destroy_process_group(self, group=None):
        self.destroyed.append(group)

    def get_backend(self, group=None):
        del group
        return "gloo"


for _name in WORKER.TENSOR_COLLECTIVE_APIS:
    setattr(FakeDist, _name, lambda *_args, **_kwargs: None)


class VllmCCLLiveWorkerTest(unittest.TestCase):
    def test_dispatch_capture_observes_one_result_without_replacing_operator(self):
        operator = torch.ops.aten.clone.default
        original = operator
        with WORKER.capture_operator_output(str(operator)) as records:
            output = operator(torch.tensor([1.0, 2.0]))
        self.assertEqual(output.tolist(), [1.0, 2.0])
        self.assertIs(torch.ops.aten.clone.default, original)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["operator"], str(operator))
        self.assertEqual(bytes.fromhex(records[0]["payload_hex"]), WORKER.tensor_bytes(output))

    def test_dispatch_capture_requires_exactly_one_result(self):
        with self.assertRaisesRegex(WORKER.RankError, "exactly one"):
            with WORKER.capture_operator_output("aten.clone.default"):
                pass

    def test_standard_vllm_control_allows_only_bounded_initialization_metadata(self):
        import torch.distributed.distributed_c10d as distributed_c10d

        dist = FakeDist()
        calls = []
        for name in WORKER.TENSOR_COLLECTIVE_APIS:
            setattr(dist, name, lambda *_args, _name=name, **_kwargs: calls.append(_name))
        original_object_broadcast = dist.broadcast_object_list
        original_internal_broadcast = distributed_c10d.broadcast
        distributed_c10d.broadcast = lambda *_args, **_kwargs: None

        def object_broadcast(objects, **_kwargs):
            distributed_c10d.broadcast(torch.zeros(1, dtype=torch.int64))
            distributed_c10d.broadcast(torch.zeros(32, dtype=torch.uint8))
            return original_object_broadcast(objects)

        dist.broadcast_object_list = object_broadcast
        try:
            with WORKER.audit_standard_vllm_control(dist, 2, 0) as audit:
                dist.init_process_group(backend="gloo", rank=0, world_size=2)
                local = dist.new_group(ranks=[0, 1], backend="gloo")
                dist.new_group(ranks=[1], backend="gloo")
                dist.all_reduce(torch.zeros(2, dtype=torch.int32))
                dist.broadcast_object_list([{"handle": "private-control"}])
                dist.barrier()
                audit["phase"]["phase"] = "model_ready"
                with self.assertRaisesRegex(WORKER.RankError, "model payload"):
                    dist.all_reduce(torch.zeros(2, dtype=torch.int32))
                audit["phase"]["phase"] = "cleanup"
                dist.destroy_process_group(local)
                dist.destroy_process_group()
        finally:
            distributed_c10d.broadcast = original_internal_broadcast
        self.assertEqual(calls, ["all_reduce", "broadcast_object_list", "barrier"])
        self.assertEqual(
            [record["api"] for record in audit["records"]],
            ["all_reduce", "broadcast_object_list", "broadcast", "broadcast", "barrier"],
        )
        self.assertTrue(audit["process_groups"]["all_local_groups_destroyed"])
        self.assertEqual(audit["process_groups"]["local_tokens_created"], [1])
        self.assertEqual(audit["process_groups"]["local_tokens_destroyed"], [1])

    def test_standard_vllm_control_accepts_positional_init_arguments(self):
        dist = FakeDist()
        with WORKER.audit_standard_vllm_control(dist, 2, 0) as audit:
            dist.init_process_group("gloo", 0, 2)
            local = dist.new_group([0, 1], None, "gloo")
            audit["phase"]["phase"] = "cleanup"
            dist.destroy_process_group(local)
            dist.destroy_process_group()
        self.assertTrue(audit["process_groups"]["all_local_groups_destroyed"])

    def test_standard_vllm_control_real_gloo_two_process_lifecycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rendezvous = root / "gloo-rendezvous"
            processes = []
            for rank in range(2):
                processes.append(subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--gloo-audit-child",
                        str(rank),
                        str(rendezvous),
                        str(root / f"rank-{rank}.json"),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    close_fds=True,
                    start_new_session=True,
                ))
            failures = []
            for rank, process in enumerate(processes):
                try:
                    stdout, stderr = process.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate(timeout=10)
                    failures.append(f"rank {rank} timed out: {stdout}\n{stderr}")
                    continue
                if process.returncode != 0:
                    failures.append(
                        f"rank {rank} exited {process.returncode}: {stdout}\n{stderr}"
                    )
            self.assertEqual(failures, [])
            for rank in range(2):
                audit = json.loads((root / f"rank-{rank}.json").read_text())
                self.assertEqual(audit["schema"], WORKER.PROCESS_GROUP_AUDIT_SCHEMA)
                self.assertEqual(audit["init"][0]["rank"], rank)
                self.assertEqual(audit["local_tokens_created"], [1, 2])
                self.assertEqual(sorted(audit["local_tokens_destroyed"]), [1, 2])
                self.assertEqual(len(audit["new"]), 3)
                self.assertTrue(audit["default_destroyed"])
                self.assertTrue(audit["all_local_groups_destroyed"])

    def test_standard_vllm_control_rejects_model_shaped_initialization_tensor(self):
        dist = FakeDist()
        with WORKER.audit_standard_vllm_control(dist, 2, 0) as audit:
            dist.init_process_group(backend="gloo", rank=0, world_size=2)
            with self.assertRaisesRegex(WORKER.RankError, "int32 control"):
                dist.all_reduce(torch.zeros(1024, dtype=torch.bfloat16))
            audit["phase"]["phase"] = "cleanup"
            dist.destroy_process_group()

    def test_standard_vllm_control_rejects_backend_drift_and_group_leak(self):
        dist = FakeDist()
        with self.assertRaisesRegex(WORKER.RankError, "exact Gloo"):
            with WORKER.audit_standard_vllm_control(dist, 2, 0):
                dist.init_process_group(backend="nccl", rank=0, world_size=2)

        dist = FakeDist()
        with self.assertRaisesRegex(WORKER.RankError, "did not close exactly"):
            with WORKER.audit_standard_vllm_control(dist, 2, 0):
                dist.init_process_group(backend="gloo", rank=0, world_size=2)
                dist.new_group(ranks=[0, 1], backend="gloo")

    def test_tensor_collective_audit_fails_and_restores_every_symbol(self):
        dist = FakeDist()
        originals = {
            name: getattr(dist, name) for name in WORKER.TENSOR_COLLECTIVE_APIS
        }
        with WORKER.reject_tensor_collectives(dist) as counters:
            with self.assertRaisesRegex(WORKER.RankError, "all_reduce"):
                dist.all_reduce(object())
            self.assertEqual(counters["all_reduce"], 1)
            self.assertEqual(sum(counters.values()), 1)
        for name, original in originals.items():
            restored = getattr(dist, name)
            self.assertIs(restored.__func__, original.__func__)

    def test_read_config_requires_private_canonical_descriptor_and_socket(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right = socket.socketpair()
            try:
                product = {"prefix": "/private/product"}
                bootstrap = {
                    "schema": "amdgpu-sim.vllm-ccl-bootstrap.v1",
                    "product": product,
                    "groups": [{
                        "unique_name": "tp:0",
                        "identity": {"world_size": 2},
                        "rank": {"rank": 0, "capability_fd": left.fileno()},
                    }],
                }
                bootstrap_path = root / "bootstrap.json"
                bootstrap_path.write_bytes(WORKER.canonical_json(bootstrap))
                os.chmod(bootstrap_path, 0o400)
                config = {
                    "schema": WORKER.CONFIG_SCHEMA,
                    "rank": 0,
                    "world_size": 2,
                    "element_count": 1024,
                    "dtype": "bfloat16",
                    "unique_name": "tp:0",
                    "rendezvous_path": str(root / "rdzv"),
                    "bootstrap_descriptor_path": str(bootstrap_path),
                    "bootstrap_descriptor_sha256": WORKER.sha256_bytes(
                        bootstrap_path.read_bytes()
                    ),
                    "result_path": str(root / "result.json"),
                    "adapter_evidence_path": str(root / "adapter.json"),
                    "journal_path": str(root / "events.jsonl"),
                    "input_path": str(root / "input.bin"),
                    "output_path": str(root / "output.bin"),
                    "runtime_library": "/private/product/lib/runtime.so",
                    "rank_launch_sha256": "11" * 32,
                    "epoch": 1,
                    "group_generation": 1,
                    "job_uuid": "22" * 16,
                    "group_uuid": "33" * 16,
                    "model_identity_sha256": "44" * 32,
                    "expected_imports": {},
                    "workload": {
                        "schema": "amdgpu-sim.vllm-ccl-workload.v1",
                        "kind": "standalone-allreduce",
                        "model": None,
                        "layer": None,
                        "input": {"policy": "rank-affine-mod127-v1"},
                        "collective": {
                            "dtype": "bfloat16", "element_count": 1024,
                        },
                    },
                    "product": product,
                }
                config_path = root / "config.json"
                config_path.write_bytes(WORKER.canonical_json(config))
                os.chmod(config_path, 0o400)
                self.assertEqual(
                    WORKER.read_config(config_path, left.fileno()), config
                )
                os.chmod(config_path, 0o644)
                with self.assertRaisesRegex(WORKER.RankError, "private owned"):
                    WORKER.read_config(config_path, left.fileno())
            finally:
                left.close()
                right.close()

    def test_coordinator_method_evidence_rejects_runtime_replacement(self):
        class Coordinator:
            def broadcast(self):
                pass

            def broadcast_tensor_dict(self):
                pass

        before = {
            "broadcast": Coordinator.broadcast,
            "broadcast_tensor_dict": Coordinator.broadcast_tensor_dict,
        }
        evidence = WORKER.coordinator_method_evidence(Coordinator, before)
        self.assertEqual(evidence, {
            "broadcast": "upstream-object-identity-preserved",
            "broadcast_tensor_dict": "upstream-object-identity-preserved",
        })

        def replacement(self):
            return None

        Coordinator.broadcast = replacement
        with self.assertRaisesRegex(WORKER.RankError, "replaced"):
            WORKER.coordinator_method_evidence(Coordinator, before)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--gloo-audit-child":
        run_gloo_audit_child(
            int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
        )
    else:
        unittest.main()
