"""Upstream HIP driver with an explicitly selected simulator benchmark policy."""

from __future__ import annotations

import os
import time

from triton.backends.amd import driver as amd_driver
from . import BACKEND_NAME


_MODE_ENV = "GEMSIM_HIP_AUTOTUNE_MODE"
_HOST_WARMUP_ENV = "GEMSIM_HIP_AUTOTUNE_HOST_WARMUP"
_HOST_REPETITIONS_ENV = "GEMSIM_HIP_AUTOTUNE_HOST_REPETITIONS"
_DEFAULT_MODE = "correctness"
_DEFAULT_HOST_WARMUP = 1
_DEFAULT_HOST_REPETITIONS = 3
_MAX_HOST_WARMUP = 16
_MAX_HOST_REPETITIONS = 16
_MIN_POSITIVE_MS = 1e-9


def _bounded_integer(name, default, *, minimum, maximum):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw, 10)
    except ValueError as error:
        raise RuntimeError(
            f"{name} must be an integer in [{minimum}, {maximum}], got {raw!r}"
        ) from error
    if not minimum <= value <= maximum:
        raise RuntimeError(
            f"{name} must be in [{minimum}, {maximum}], got {value}"
        )
    return value


def _requested_quantiles(samples, quantiles):
    ordered = sorted(samples)
    last = len(ordered) - 1
    result = []
    for quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"benchmark quantile must be in [0, 1], got {quantile}")
        point = last * quantile
        lower = int(point)
        upper = min(lower + 1, last)
        weight = point - lower
        result.append((1.0 - weight) * ordered[lower] + weight * ordered[upper])
    return result


class GemsimHIPDriver(amd_driver.HIPDriver):

    @staticmethod
    def is_active():
        return (
            os.environ.get("TRITON_DEFAULT_BACKEND") == BACKEND_NAME
            and amd_driver.HIPDriver.is_active()
        )

    def get_current_target(self):
        # Preserve the public HIP target identity so upstream AMD feature
        # selection and compilation continue to use the stock HIP backend.
        return super().get_current_target()

    def get_benchmarker(self):
        mode = os.environ.get(_MODE_ENV, _DEFAULT_MODE)
        if mode == "device":
            return super().get_benchmarker()
        if mode == "correctness":
            return self._correctness_benchmarker()
        if mode == "host":
            return self._host_benchmarker()
        raise RuntimeError(
            f"{_MODE_ENV} must be one of correctness, host, or device, got {mode!r}"
        )

    def _correctness_benchmarker(self):
        """Execute each candidate once, without making a performance claim."""
        device = self.get_device_interface()

        def benchmark(kernel_call, *, quantiles, **_kwargs):
            kernel_call()
            device.synchronize()
            return [1.0 for _ in quantiles]

        return benchmark

    def _host_benchmarker(self):
        """Bound calls using host wall time only; this is not device timing."""
        warmup = _bounded_integer(
            _HOST_WARMUP_ENV,
            _DEFAULT_HOST_WARMUP,
            minimum=0,
            maximum=_MAX_HOST_WARMUP,
        )
        repetitions = _bounded_integer(
            _HOST_REPETITIONS_ENV,
            _DEFAULT_HOST_REPETITIONS,
            minimum=1,
            maximum=_MAX_HOST_REPETITIONS,
        )
        device = self.get_device_interface()

        def benchmark(kernel_call, *, quantiles, **_kwargs):
            for _ in range(warmup):
                kernel_call()
                device.synchronize()

            samples = []
            for _ in range(repetitions):
                start_ns = time.perf_counter_ns()
                kernel_call()
                device.synchronize()
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                samples.append(max(elapsed_ms, _MIN_POSITIVE_MS))
            return _requested_quantiles(samples, quantiles)

        return benchmark
