"""Host contract tests for the formal out-of-tree vLLM communicator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_SRC = ROOT / "plugins/framework/gemsim_vllm/src"
CCL_SRC = ROOT / "plugins/collectives/gemsim_ccl/src"
for value in (FRAMEWORK_SRC, CCL_SRC):
    sys.path.insert(0, str(value))
with mock.patch.dict(
    os.environ,
    {"ROCM_SIM_ROOT": "/private/prefix", "TRITON_DEFAULT_BACKEND": "gemsim_amd"},
):
    from gemsim_vllm import ccl_bootstrap  # noqa: E402
    from gemsim_vllm.communicator import (  # noqa: E402
        GemsimDeviceCommunicator,
        _build_engine,
    )


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


class FakeGroup:
    def __init__(self, rank: int = 0, world: int = 2) -> None:
        self._rank = rank
        self._world = world

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._world


class FakeEngine:
    def __init__(self, rank: int = 0, world: int = 2) -> None:
        self.rank = rank
        self.world_size = world
        self.destroy_count = 0
        self.close_count = 0
        self.fail_close = False
        self.calls: list[tuple[torch.Tensor, int]] = []

    def all_reduce(self, tensor: torch.Tensor, *, timeout_ns: int) -> torch.Tensor:
        self.calls.append((tensor, timeout_ns))
        return tensor.clone()

    def destroy(self) -> None:
        self.destroy_count += 1

    def close(self, *, timeout_ns: int) -> None:
        self.close_count += 1
        if self.fail_close:
            raise RuntimeError("close failed")


class GemsimVllmCommunicatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.environment_before = os.environ.copy()
        self.addCleanup(self.restore_environment)
        ccl_bootstrap._reset_claims_for_tests()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.product = self.root / "product"
        self.product.mkdir()
        runtime = self.product / "lib/libself_amdgpu_runtime.so.0.8.0"
        runtime.parent.mkdir()
        runtime.write_bytes(b"runtime-v1")
        runtime.chmod(0o444)
        self.runtime_path = runtime
        runtime_record = {
            "path": str(runtime),
            "bytes": runtime.stat().st_size,
            "sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        }
        self.manifest_path = self.product / "manifest.json"
        manifest = {
            "schema": "amdgpu-sim.product-prefix.v1",
            "prefix": str(self.product),
            "artifacts": {"runtime_library": runtime_record},
        }
        self.manifest_path.write_bytes(canonical_json(manifest))
        self.manifest_path.chmod(0o400)
        self.manifest_record = {
            "path": str(self.manifest_path),
            "bytes": self.manifest_path.stat().st_size,
            "sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
        }
        self.runtime_record = runtime_record
        self.socket, self.peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.descriptor = self.root / "ccl-bootstrap.json"
        self.write_descriptor()

    def tearDown(self) -> None:
        self.socket.close()
        self.peer.close()
        self.temporary.cleanup()
        ccl_bootstrap._reset_claims_for_tests()

    def restore_environment(self) -> None:
        os.environ.clear()
        os.environ.update(self.environment_before)

    def write_descriptor(
        self,
        *,
        unique_name: str = "tp:0",
        rank: int = 0,
        world: int = 2,
        capability_fd: int | None = None,
    ) -> None:
        document = {
            "schema": "amdgpu-sim.vllm-ccl-bootstrap.v1",
            "product": {
                "prefix": str(self.product),
                "manifest": self.manifest_record,
                "runtime_library": self.runtime_record,
            },
            "groups": [
                {
                    "unique_name": unique_name,
                    "identity": {
                        "world_size": world,
                        "epoch": 1,
                        "group_generation": 1,
                        "job_uuid": "01" * 16,
                        "group_uuid": "02" * 16,
                        "model_identity_sha256": "03" * 32,
                    },
                    "rank": {
                        "rank": rank,
                        "capability_fd": (
                            self.socket.fileno()
                            if capability_fd is None
                            else capability_fd
                        ),
                        "broker_pid": os.getpid(),
                        "broker_start_time_ticks": 1,
                        "join_timeout_ns": 1_000_000,
                        "collective_timeout_ns": 2_000_000,
                        "credits_per_peer": 2,
                    },
                }
            ],
        }
        self.descriptor.write_bytes(canonical_json(document))
        self.descriptor.chmod(stat.S_IRUSR)

    def construct(self, engine: FakeEngine | None = None):
        engine = engine or FakeEngine()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "ROCM_SIM_ROOT": str(self.product),
                    "GEMSIM_CCL_BOOTSTRAP_DESCRIPTOR": str(self.descriptor),
                },
                clear=True,
            ),
            mock.patch(
                "gemsim_vllm.communicator.DeviceCommunicatorBase.__init__",
                autospec=True,
            ) as base_init,
            mock.patch("gemsim_vllm.communicator._build_engine", return_value=engine),
        ):
            def initialize(instance, *, cpu_group, device, device_group,
                           unique_name, global_ranks, global_world_size, use_all2all):
                instance.cpu_group = cpu_group
                instance.device_group = device_group
                instance.device = device
                instance.unique_name = unique_name
                instance.rank = cpu_group.rank()
                instance.rank_in_group = instance.rank
                instance.world_size = cpu_group.size()

            base_init.side_effect = initialize
            communicator = GemsimDeviceCommunicator(
                cpu_group=FakeGroup(),
                device=torch.device("cpu:0"),
                device_group=object(),
                unique_name="tp:0",
            )
        return communicator, engine

    def engine_binding(self, capability_fd: int) -> dict:
        return {
            "product": {"runtime_library": {"path": str(self.runtime_path)}},
            "group": {
                "identity": {
                    "world_size": 2,
                    "epoch": 1,
                    "group_generation": 1,
                    "job_uuid": "01" * 16,
                    "group_uuid": "02" * 16,
                    "model_identity_sha256": "03" * 32,
                },
                "rank": {
                    "rank": 0,
                    "capability_fd": capability_fd,
                    "broker_pid": os.getpid(),
                    "broker_start_time_ticks": 1,
                    "join_timeout_ns": 1_000_000,
                    "collective_timeout_ns": 2_000_000,
                    "credits_per_peer": 2,
                },
            },
        }

    def test_all_reduce_is_fresh_and_never_uses_torch_distributed(self) -> None:
        communicator, engine = self.construct()
        input_ = torch.arange(16, dtype=torch.float32).to(torch.bfloat16)
        before = input_.view(torch.uint8).clone()
        with mock.patch(
            "torch.distributed.all_reduce",
            side_effect=AssertionError("tensor fallback is forbidden"),
        ) as fallback:
            output = communicator.all_reduce(input_)
        fallback.assert_not_called()
        self.assertTrue(torch.equal(input_.view(torch.uint8), before))
        self.assertTrue(torch.equal(output, input_))
        self.assertNotEqual(
            output.untyped_storage().data_ptr(), input_.untyped_storage().data_ptr()
        )
        self.assertEqual(engine.calls[0][1], 2_000_000)
        communicator.destroy()
        communicator.destroy()
        self.assertEqual(engine.close_count, 1)
        self.assertEqual(engine.destroy_count, 0)

    def test_destroy_falls_back_to_unconditional_cleanup(self) -> None:
        engine = FakeEngine()
        engine.fail_close = True
        communicator, _ = self.construct(engine)
        communicator.destroy()
        self.assertEqual(engine.close_count, 1)
        self.assertEqual(engine.destroy_count, 1)

    def test_unaccepted_dtype_and_tensor_contract_fail_closed(self) -> None:
        communicator, _ = self.construct()
        with self.assertRaises(TypeError):
            communicator.all_reduce(torch.ones(4, dtype=torch.float32))
        with self.assertRaises(ValueError):
            communicator.all_reduce(torch.empty(0, dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            communicator.all_reduce(
                torch.ones((2, 4), dtype=torch.bfloat16).transpose(0, 1)
            )

    def test_every_unaccepted_tensor_method_is_explicitly_rejected(self) -> None:
        communicator, _ = self.construct()
        tensor = torch.ones(4, dtype=torch.bfloat16)
        calls = (
            lambda: communicator.all_gather(tensor),
            lambda: communicator.all_gatherv(tensor),
            lambda: communicator.reduce_scatter(tensor),
            lambda: communicator.reduce_scatterv(tensor),
            lambda: communicator.gather(tensor),
            lambda: communicator.send(tensor),
            lambda: communicator.recv(torch.Size([4]), torch.bfloat16),
            lambda: communicator.broadcast(tensor),
            lambda: communicator.batch_isend_irecv([]),
            lambda: communicator.dispatch_router_logits(tensor, tensor),
            lambda: communicator.dispatch(tensor, tensor, tensor),
            lambda: communicator.combine(tensor),
            communicator.checkpoint_prepare,
            communicator.checkpoint_restore,
        )
        with mock.patch.multiple(
            torch.distributed,
            all_gather_into_tensor=mock.DEFAULT,
            reduce_scatter_tensor=mock.DEFAULT,
            gather=mock.DEFAULT,
            send=mock.DEFAULT,
            recv=mock.DEFAULT,
            broadcast=mock.DEFAULT,
        ) as fallbacks:
            for call in calls:
                with self.assertRaises(NotImplementedError):
                    call()
            for fallback in fallbacks.values():
                fallback.assert_not_called()

    def test_bootstrap_rejects_identity_drift_and_reuse(self) -> None:
        binding = ccl_bootstrap.claim_group(
            "tp:0",
            expected_rank=0,
            expected_world_size=2,
            descriptor_path=self.descriptor,
        )
        self.assertEqual(binding["group"]["rank"]["rank"], 0)
        with self.assertRaisesRegex(ccl_bootstrap.BootstrapError, "already consumed"):
            ccl_bootstrap.claim_group(
                "tp:0",
                expected_rank=0,
                expected_world_size=2,
                descriptor_path=self.descriptor,
            )
        ccl_bootstrap._reset_claims_for_tests()
        with self.assertRaisesRegex(ccl_bootstrap.BootstrapError, "rank/world"):
            ccl_bootstrap.claim_group(
                "tp:0",
                expected_rank=1,
                expected_world_size=2,
                descriptor_path=self.descriptor,
            )
        ccl_bootstrap._reset_claims_for_tests()
        self.runtime_record["sha256"] = "f" * 64
        self.descriptor.chmod(stat.S_IRUSR | stat.S_IWUSR)
        self.write_descriptor()
        with self.assertRaisesRegex(ccl_bootstrap.BootstrapError, "identity mismatch"):
            ccl_bootstrap.claim_group(
                "tp:0",
                expected_rank=0,
                expected_world_size=2,
                descriptor_path=self.descriptor,
            )

    def test_product_mismatch_is_rejected_before_the_group_is_claimed(self) -> None:
        with self.assertRaisesRegex(ccl_bootstrap.BootstrapError, "active product"):
            ccl_bootstrap.claim_group(
                "tp:0",
                expected_rank=0,
                expected_world_size=2,
                expected_product_prefix=self.root,
                descriptor_path=self.descriptor,
            )
        binding = ccl_bootstrap.claim_group(
            "tp:0",
            expected_rank=0,
            expected_world_size=2,
            expected_product_prefix=self.product,
            descriptor_path=self.descriptor,
        )
        self.assertEqual(binding["product"]["prefix"], self.product)

    def test_non_tp_and_all2all_fail_before_claim(self) -> None:
        with self.assertRaisesRegex(Exception, "tp"):
            GemsimDeviceCommunicator(
                cpu_group=FakeGroup(), unique_name="dp:0", use_all2all=False
            )
        with self.assertRaisesRegex(Exception, "all-to-all"):
            GemsimDeviceCommunicator(
                cpu_group=FakeGroup(), unique_name="tp:0", use_all2all=True
            )

    def test_construct_does_not_leak_ccl_or_platform_environment(self) -> None:
        expected = {
            "ROCM_SIM_ROOT": os.environ.get("ROCM_SIM_ROOT"),
            "TRITON_DEFAULT_BACKEND": os.environ.get("TRITON_DEFAULT_BACKEND"),
            "GEMSIM_CCL_RUNTIME": os.environ.get("GEMSIM_CCL_RUNTIME"),
            "GEMSIM_CCL_RUNTIME_LIBRARY": os.environ.get(
                "GEMSIM_CCL_RUNTIME_LIBRARY"
            ),
            "GEMSIM_CCL_BOOTSTRAP_DESCRIPTOR": os.environ.get(
                "GEMSIM_CCL_BOOTSTRAP_DESCRIPTOR"
            ),
        }
        communicator, _ = self.construct()
        communicator.destroy()
        self.assertEqual(
            {name: os.environ.get(name) for name in expected},
            expected,
        )

    def test_engine_builder_closes_fd_before_ownership_transfer(self) -> None:
        capability_fd = os.dup(self.socket.fileno())
        with mock.patch(
            "gemsim_ccl.native.NativeCCL",
            side_effect=RuntimeError("native construction failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "native construction"):
                _build_engine(self.engine_binding(capability_fd))
        with self.assertRaises(OSError):
            os.fstat(capability_fd)

    def test_engine_builder_does_not_double_close_after_join_entry(self) -> None:
        capability_fd = os.dup(self.socket.fileno())
        native = mock.Mock()
        native.deadline_after.return_value = 9_000_000
        try:
            with (
                mock.patch("gemsim_ccl.native.NativeCCL", return_value=native),
                mock.patch(
                    "gemsim_ccl.engine.AllReduceEngine.join",
                    side_effect=RuntimeError("join failed after transfer"),
                ) as join,
                mock.patch("gemsim_vllm.communicator.os.close") as close,
            ):
                with self.assertRaisesRegex(RuntimeError, "join failed"):
                    _build_engine(self.engine_binding(capability_fd))
            join.assert_called_once()
            close.assert_not_called()
        finally:
            os.close(capability_fd)


if __name__ == "__main__":
    unittest.main()
