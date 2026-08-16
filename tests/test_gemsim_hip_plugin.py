"""Host-only tests for the simulator-scoped Triton HIP policy backend."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tomllib
import unittest
from unittest import mock

from triton.backends import _find_concrete_subclasses
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.backends.driver import DriverBase
from triton.compiler.compiler import make_backend


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins/triton/gemsim_hip"
sys.path.insert(0, str(PLUGIN / "src"))

from gemsim_hip import BACKEND_NAME  # noqa: E402
from gemsim_hip import compiler  # noqa: E402
from gemsim_hip import driver  # noqa: E402


class FakeDevice:
    def __init__(self):
        self.synchronize_count = 0

    def synchronize(self):
        self.synchronize_count += 1


class GemsimHIPPluginTest(unittest.TestCase):
    def setUp(self):
        self.instance = object.__new__(driver.GemsimHIPDriver)
        self.device = FakeDevice()
        self.device_patch = mock.patch.object(
            self.instance, "get_device_interface", return_value=self.device
        )
        self.device_patch.start()
        self.addCleanup(self.device_patch.stop)

    def test_package_publishes_official_triton_backend_entry_point(self):
        metadata = tomllib.loads((PLUGIN / "pyproject.toml").read_text())
        self.assertEqual(
            metadata["project"]["entry-points"]["triton.backends"],
            {BACKEND_NAME: "gemsim_hip"},
        )
        self.assertIs(
            _find_concrete_subclasses(compiler, BaseBackend),
            compiler.GemsimHIPBackend,
        )
        self.assertIs(
            _find_concrete_subclasses(driver, DriverBase),
            driver.GemsimHIPDriver,
        )

    def test_compiler_reuses_upstream_hip_for_only_its_target(self):
        target = GPUTarget(BACKEND_NAME, "gfx950", 64)
        backend = compiler.GemsimHIPBackend(target)
        self.assertTrue(backend.supports_target(target))
        self.assertFalse(
            backend.supports_target(GPUTarget("hip", "gfx950", 64))
        )
        self.assertEqual(backend.get_target_name(backend.parse_options({})), "hip:gfx950")

    def test_driver_activation_requires_explicit_backend_selection(self):
        with mock.patch.object(
            driver.amd_driver.HIPDriver, "is_active", return_value=True
        ):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertFalse(driver.GemsimHIPDriver.is_active())
            with mock.patch.dict(
                os.environ, {"TRITON_DEFAULT_BACKEND": BACKEND_NAME}, clear=True
            ):
                self.assertTrue(driver.GemsimHIPDriver.is_active())
        with mock.patch.object(
            driver.amd_driver.HIPDriver, "is_active", return_value=False
        ), mock.patch.dict(
            os.environ, {"TRITON_DEFAULT_BACKEND": BACKEND_NAME}, clear=True
        ):
            self.assertFalse(driver.GemsimHIPDriver.is_active())

    def test_driver_preserves_upstream_hip_target_and_compiler(self):
        upstream = GPUTarget("hip", "gfx950", 64)
        with mock.patch.object(
            driver.amd_driver.HIPDriver,
            "get_current_target",
            return_value=upstream,
        ):
            self.assertEqual(
                self.instance.get_current_target(),
                upstream,
            )
        self.assertEqual(
            type(make_backend(upstream)).__module__,
            "triton.backends.amd.compiler",
        )

    def test_correctness_mode_runs_each_candidate_once(self):
        calls = []
        with mock.patch.dict(os.environ, {}, clear=True):
            benchmark = self.instance.get_benchmarker()
            result = benchmark(
                lambda: calls.append("kernel"), quantiles=(0.5, 0.2, 0.8)
            )
        self.assertEqual(calls, ["kernel"])
        self.assertEqual(self.device.synchronize_count, 1)
        self.assertEqual(result, [1.0, 1.0, 1.0])

    def test_host_mode_has_bounded_counts_and_host_wall_quantiles(self):
        environment = {
            "GEMSIM_HIP_AUTOTUNE_MODE": "host",
            "GEMSIM_HIP_AUTOTUNE_HOST_WARMUP": "1",
            "GEMSIM_HIP_AUTOTUNE_HOST_REPETITIONS": "3",
        }
        clock = iter((0, 2_000_000, 10_000_000, 14_000_000, 20_000_000, 26_000_000))
        calls = []
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
            driver.time, "perf_counter_ns", side_effect=lambda: next(clock)
        ):
            benchmark = self.instance.get_benchmarker()
            result = benchmark(
                lambda: calls.append("kernel"), quantiles=(0.5, 0.2, 0.8)
            )
        self.assertEqual(len(calls), 4)
        self.assertEqual(self.device.synchronize_count, 4)
        self.assertEqual(result, [4.0, 2.8, 5.2])

    def test_device_mode_delegates_to_upstream_amd_driver(self):
        sentinel = object()
        with mock.patch.dict(
            os.environ, {"GEMSIM_HIP_AUTOTUNE_MODE": "device"}, clear=True
        ), mock.patch.object(
            driver.amd_driver.HIPDriver,
            "get_benchmarker",
            return_value=sentinel,
        ) as upstream:
            self.assertIs(self.instance.get_benchmarker(), sentinel)
        upstream.assert_called_once_with()

    def test_policy_environment_is_fail_closed(self):
        for environment, message in (
            ({"GEMSIM_HIP_AUTOTUNE_MODE": "unknown"}, "must be one of"),
            (
                {
                    "GEMSIM_HIP_AUTOTUNE_MODE": "host",
                    "GEMSIM_HIP_AUTOTUNE_HOST_REPETITIONS": "0",
                },
                "must be in",
            ),
            (
                {
                    "GEMSIM_HIP_AUTOTUNE_MODE": "host",
                    "GEMSIM_HIP_AUTOTUNE_HOST_WARMUP": "many",
                },
                "must be an integer",
            ),
        ):
            with self.subTest(environment=environment), mock.patch.dict(
                os.environ, environment, clear=True
            ), self.assertRaisesRegex(RuntimeError, message):
                self.instance.get_benchmarker()


if __name__ == "__main__":
    unittest.main()
