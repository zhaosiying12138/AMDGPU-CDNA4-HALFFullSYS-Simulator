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
    @staticmethod
    def _hip_inventory_fixture(stage: Path) -> str:
        library = stage / "lib"
        library.mkdir(parents=True)
        version = (
            "HIP_PACKAGING_VERSION_PATCH=26331-d140452e1c\n"
            "HIP_VERSION_MAJOR=7\n"
            "HIP_VERSION_MINOR=16\n"
            "HIP_VERSION_PATCH=26331\n"
            "HIP_VERSION_GITHASH=d140452e1c\n"
        )
        for base in MODULE.HIP_VERSIONED_SONAMES:
            soname = f"{base}.7"
            target = f"{soname}.16.26331-d140452e1c"
            (library / target).write_bytes(base.encode("ascii"))
            (library / soname).symlink_to(target)
            (library / base).symlink_to(soname)
        return version

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

    def test_hip_version_inventory_binds_metadata_and_unique_dsos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            version = self._hip_inventory_fixture(stage)
            result = MODULE.hip_versioned_inventory(stage, version)

        self.assertEqual(result["metadata"]["HIP_VERSION_GITHASH"], "d140452e1c")
        self.assertEqual(
            result["libraries"]["libamdhip64.so"]["versioned"],
            "libamdhip64.so.7.16.26331-d140452e1c",
        )

    def test_hip_version_inventory_rejects_stale_versioned_dso(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            version = self._hip_inventory_fixture(stage)
            (stage / "lib/libamdhip64.so.7.16.26315-92115a2941").write_bytes(b"stale")
            with self.assertRaisesRegex(
                MODULE.BuildEnvironmentError, "inventory is not unique"
            ):
                MODULE.hip_versioned_inventory(stage, version)

    def test_hip_version_inventory_rejects_metadata_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            version = self._hip_inventory_fixture(stage).replace(
                "HIP_VERSION_GITHASH=d140452e1c",
                "HIP_VERSION_GITHASH=92115a2941",
            )
            with self.assertRaisesRegex(
                MODULE.BuildEnvironmentError, "packaging version differs"
            ):
                MODULE.hip_versioned_inventory(stage, version)

    def test_lock_rejects_unknown_artifact_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.lock"
            path.write_text("https://example.invalid/pkg-1-0.rpm#" + "0" * 64 + "\n", encoding="ascii")
            with self.assertRaises(MODULE.BuildEnvironmentError):
                MODULE.locked_packages(path)


if __name__ == "__main__":
    unittest.main()
