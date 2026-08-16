from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hip_facade_build", ROOT / "tools/hip_facade_build_environment.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HipFacadeBuildEnvironmentTest(unittest.TestCase):
    def test_lock_accepts_conda_and_tar_bz2(self) -> None:
        packages = MODULE.locked_packages(ROOT / MODULE.LOCK_RELATIVE)
        # The lock is an explicit full build environment, so adding an
        # upstream configure dependency must not require changing this test.
        self.assertGreaterEqual(len(packages), 11)
        self.assertEqual(packages["elfutils"]["format"], "tar.bz2")
        self.assertEqual(packages["libdrm"]["format"], "conda")

    def test_current_dependency_prefix_matches_lock(self) -> None:
        result = MODULE.verify_prefix(ROOT)
        self.assertEqual(result["prefix"], str(ROOT / MODULE.PREFIX_RELATIVE))
        self.assertEqual(
            len(result["packages"]),
            len(MODULE.locked_packages(ROOT / MODULE.LOCK_RELATIVE)),
        )

    def test_current_rocr_core_stage_is_complete(self) -> None:
        result = MODULE.verify_core_stage(ROOT)
        self.assertEqual(result["prefix"], str(ROOT / MODULE.CORE_STAGE_RELATIVE))
        self.assertIn("lib/libhsa-runtime64.so", result["artifacts"])

    def test_current_two_root_hip_compile_contract(self) -> None:
        result = MODULE.verify_hip_compile_contract(ROOT)
        self.assertEqual(result["target"], "gfx950")
        self.assertTrue(result["runtime_wrapper_injected"])
        self.assertEqual(result["hip_api_root"], str(ROOT / MODULE.HIP_STAGE_RELATIVE))

    def test_facade_environment_is_amd_only_and_shared(self) -> None:
        result = MODULE.facade_environment(ROOT)
        self.assertEqual(result["set"]["HIP_PLATFORM"], "amd")
        self.assertIn("CUDA_HOME", result["unset"])
        self.assertFalse(result["policy"]["framework_specific_bridge"])
        self.assertFalse(result["policy"]["system_rocm_fallback"])

    def test_dynamic_symbol_reader_handles_versioned_upstream_abi(self) -> None:
        stage = ROOT / MODULE.HIP_STAGE_RELATIVE
        symbols = MODULE.dynamic_symbols(stage / "lib/libamdhip64.so.7")
        self.assertIn("hipGetDeviceCount", symbols)
        self.assertIn("hipModuleLaunchKernel", symbols)

    def test_lock_rejects_unknown_artifact_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.lock"
            path.write_text("https://example.invalid/pkg-1-0.rpm#" + "0" * 64 + "\n", encoding="ascii")
            with self.assertRaises(MODULE.BuildEnvironmentError):
                MODULE.locked_packages(path)


if __name__ == "__main__":
    unittest.main()
