from __future__ import annotations

from datetime import timedelta
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
SGLANG_PLUGIN = ROOT / "plugins/framework/gemsim_sglang/src"
CCL_PLUGIN = ROOT / "plugins/collectives/gemsim_ccl/src"
for path in (SGLANG_PLUGIN, CCL_PLUGIN):
    sys.path.insert(0, str(path))

import gemsim_sglang  # noqa: E402
from gemsim_sglang import activate  # noqa: E402
from gemsim_sglang import process_group as PG  # noqa: E402
from gemsim_ccl import torch_process_group as SHARED_PG  # noqa: E402


class FakeEngine:
    def __init__(self, rank: int, world: int, *, add: float = 0.0) -> None:
        self.rank = rank
        self.world_size = world
        self.add = add
        self.calls: list[tuple[torch.Tensor, int]] = []
        self.close_count = 0
        self.destroy_count = 0
        self.fail_at: int | None = None

    def all_reduce(self, tensor: torch.Tensor, *, timeout_ns: int):
        self.calls.append((tensor, timeout_ns))
        if self.fail_at == len(self.calls):
            raise RuntimeError("injected collective failure")
        return (tensor.float() + self.add).to(tensor.dtype)

    def close(self, *, timeout_ns: int) -> None:
        self.close_count += 1

    def destroy(self) -> None:
        self.destroy_count += 1


class GemsimProcessGroupTest(unittest.TestCase):
    def test_framework_module_is_only_a_shared_process_group_facade(self) -> None:
        self.assertIs(PG.GemsimProcessGroup, SHARED_PG.GemsimProcessGroup)
        self.assertIs(PG.register_backend, SHARED_PG.register_backend)

    def group(self, *, rank: int = 0, world: int = 2, engine: FakeEngine | None = None):
        selected = engine or FakeEngine(rank, world, add=1.0)
        group = PG.GemsimProcessGroup(
            dist.HashStore(),
            rank,
            world,
            timedelta(seconds=2),
            group_id=f"tp-{world}",
            engine_factory=lambda: selected,
        )
        self.addCleanup(group.abort)
        return group, selected

    def test_torch_distributed_all_reduce_routes_to_generic_process_group(self) -> None:
        for world in (1, 2, 4):
            with self.subTest(world=world):
                group, engine = self.group(world=world)
                tensor = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
                result = dist.all_reduce(tensor, group=group)
                self.assertIsNone(result)
                self.assertTrue(
                    torch.equal(
                        tensor,
                        torch.tensor([2.0, 3.0], dtype=torch.bfloat16),
                    )
                )
                self.assertEqual(len(engine.calls), 1)
                self.assertGreater(engine.calls[0][1], 0)

    def test_async_work_and_failure_atomic_multi_tensor_commit(self) -> None:
        group, engine = self.group(world=4)
        first = torch.tensor([1.0], dtype=torch.float32)
        work = dist.all_reduce(first, group=group, async_op=True)
        self.assertTrue(work.wait())
        self.assertEqual(first.item(), 2.0)

        left = torch.tensor([4.0], dtype=torch.float32)
        right = torch.tensor([8.0], dtype=torch.float32)
        engine.fail_at = len(engine.calls) + 2
        before = (left.clone(), right.clone())
        options = SimpleNamespace(reduceOp=dist.ReduceOp.SUM)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            group.allreduce([left, right], options)
        self.assertTrue(torch.equal(left, before[0]))
        self.assertTrue(torch.equal(right, before[1]))

    def test_fail_closed_tensor_and_collective_contract(self) -> None:
        group, engine = self.group(world=2)
        invalid = (
            (torch.ones(2, dtype=torch.float64), TypeError),
            (torch.empty(0, dtype=torch.float32), ValueError),
            (torch.ones((2, 3), dtype=torch.float32).t(), ValueError),
        )
        for tensor, error in invalid:
            with self.assertRaises(error):
                dist.all_reduce(tensor, group=group)
        options = SimpleNamespace(reduceOp=dist.ReduceOp.MAX)
        with self.assertRaises(NotImplementedError):
            group.allreduce([torch.ones(1, dtype=torch.float32)], options)
        with self.assertRaises(NotImplementedError):
            group.broadcast([], SimpleNamespace())
        self.assertEqual(engine.calls, [])

    def test_barrier_is_store_control_and_shutdown_is_idempotent(self) -> None:
        group, engine = self.group(world=1)
        work = dist.barrier(group=group, async_op=True)
        self.assertTrue(work.wait())
        self.assertEqual(engine.calls, [])
        group.shutdown()
        group.shutdown()
        self.assertEqual(engine.close_count, 1)

    def test_backend_registration_uses_extended_group_identity(self) -> None:
        previous = SHARED_PG._registered
        SHARED_PG._registered = False
        self.addCleanup(setattr, SHARED_PG, "_registered", previous)
        with mock.patch.object(dist.Backend, "backend_list", ["gloo"]), mock.patch.object(
            dist.Backend, "register_backend"
        ) as register:
            PG.register_backend()
            PG.register_backend()
        register.assert_called_once_with(
            PG.BACKEND_NAME,
            SHARED_PG._creator,
            extended_api=True,
            devices=["cpu", "cuda"],
        )


class GemsimSGLangPlatformTest(unittest.TestCase):
    def test_activation_is_explicit_and_returns_official_class_qualname(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(activate())
        environment = {
            "SGLANG_PLATFORM": "gemsim",
            "ROCM_SIM_ROOT": "/private/product",
            "GEMSIM_HIP_DEVICE_FACADE": "1",
        }
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
            gemsim_sglang, "_hip_facade_ready", return_value=True
        ):
            self.assertEqual(
                activate(), "gemsim_sglang.platform:GemsimSRTPlatform"
            )
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
            gemsim_sglang, "_hip_facade_ready", return_value=False
        ):
            self.assertIsNone(activate())

    def test_platform_contract_uses_upstream_hooks_without_importing_sglang_tree(self) -> None:
        rocm = ModuleType("sglang.srt.platforms.rocm")
        interface = ModuleType("sglang.srt.platforms.interface")
        mixin = ModuleType("sglang.srt.platforms.device_mixin")

        class RocmDeviceMixin:
            pass

        class SRTPlatform:
            pass

        class PlatformEnum:
            OOT = object()

        rocm.RocmDeviceMixin = RocmDeviceMixin
        interface.SRTPlatform = SRTPlatform
        mixin.PlatformEnum = PlatformEnum
        stubs = {
            "sglang": ModuleType("sglang"),
            "sglang.srt": ModuleType("sglang.srt"),
            "sglang.srt.platforms": ModuleType("sglang.srt.platforms"),
            "sglang.srt.platforms.rocm": rocm,
            "sglang.srt.platforms.interface": interface,
            "sglang.srt.platforms.device_mixin": mixin,
        }
        sys.modules.pop("gemsim_sglang.platform", None)
        with mock.patch.dict(sys.modules, stubs):
            platform_module = importlib.import_module("gemsim_sglang.platform")
            platform = platform_module.GemsimSRTPlatform()
        self.assertEqual(platform.device_type, "cuda")
        self.assertEqual(platform.device_name, "rocm")
        self.assertEqual(platform.get_torch_distributed_backend_str(), "gemsim_ccl")
        self.assertEqual(platform.get_default_attention_backend(), "triton")
        self.assertEqual(platform.get_compile_backend(), "inductor")
        self.assertEqual(platform.get_dispatch_key_name(), "hip")
        self.assertFalse(platform.support_cuda_graph())
        self.assertFalse(platform.support_piecewise_cuda_graph())


if __name__ == "__main__":
    unittest.main()
