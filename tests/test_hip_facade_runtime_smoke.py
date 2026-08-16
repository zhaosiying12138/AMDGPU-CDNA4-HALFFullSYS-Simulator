from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hip_facade_runtime_smoke", ROOT / "tools/hip_facade_runtime_smoke.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HipFacadeRuntimeSmokeTest(unittest.TestCase):
    def test_kernel_is_generic_upstream_hip(self) -> None:
        self.assertIn('extern "C" __global__ void facade_vector_add', MODULE.KERNEL_SOURCE)
        self.assertIn("blockIdx.x * blockDim.x + threadIdx.x", MODULE.KERNEL_SOURCE)
        self.assertNotIn("gemsim", MODULE.KERNEL_SOURCE.lower())
        self.assertNotIn("sagr", MODULE.KERNEL_SOURCE.lower())

    def test_canonical_result_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.canonical_json({"value": float("nan")})

    def test_count_gate_is_bounded(self) -> None:
        self.assertEqual(MODULE.main(["--count", "0"]), 1)
        self.assertEqual(MODULE.main(["--count", "1048577"]), 1)

    def test_public_hip_contract_has_no_private_runtime_symbol(self) -> None:
        source = (ROOT / "tools/hip_facade_runtime_smoke.py").read_text(encoding="ascii")
        self.assertNotIn("ctypes.CDLL(str(runtime", source)
        self.assertNotIn("sagr_managed_", source)
        for symbol in (
            "hipGetDeviceCount",
            "hipMalloc",
            "hipMemcpy",
            "hipStreamCreate",
            "hipModuleLoadData",
            "hipModuleLaunchKernel",
        ):
            self.assertIn(symbol, source)

    def test_compiler_uses_standard_raw_device_code_object_options(self) -> None:
        source = (ROOT / "tools/hip_facade_runtime_smoke.py").read_text(encoding="ascii")
        self.assertIn('"--offload-device-only"', source)
        self.assertIn('"--no-gpu-bundle-output"', source)
        self.assertNotIn('"--genco"', source)


if __name__ == "__main__":
    unittest.main()
