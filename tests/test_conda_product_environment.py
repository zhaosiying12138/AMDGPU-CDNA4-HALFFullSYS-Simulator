"""Host-only contracts for the repository-owned conda product."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import stat
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/conda_product_environment.py"
SPEC = importlib.util.spec_from_file_location("conda_product_for_tests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CondaProductEnvironmentTest(unittest.TestCase):
    def test_lock_and_builder_are_part_of_identity_contract(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertEqual(MODULE.LOCK_RELATIVE, Path("config/conda-linux-64.lock"))
        self.assertIn('"builder": file_sha256(Path(__file__).resolve())', source)
        self.assertIn('"tools": tool_identities(root)', source)

    def test_smi_sources_are_part_of_product_identity(self) -> None:
        records = MODULE.tool_identities(ROOT)
        self.assertEqual(set(records), {"gemsim_live_registry", "gemsim_smi"})
        for record in records.values():
            self.assertGreater(record["bytes"], 0)
            self.assertEqual(len(record["sha256"]), 64)

    def test_base_exclude_removes_editable_plugins_and_backend(self) -> None:
        for path in (
            Path("__editable__.gemsim_ccl-0.1.0.pth"),
            Path("__editable__.gemsim_vllm_plugin-0.1.0.pth"),
            Path("gemsim_ccl-0.1.0.dist-info/METADATA"),
            Path("gemsim_vllm_plugin-0.1.0.dist-info/METADATA"),
            Path("triton/backends/gemsim_amd/driver.py"),
        ):
            self.assertTrue(MODULE.base_exclude(path), path)
        self.assertFalse(MODULE.base_exclude(Path("vllm/__init__.py")))

    def test_activation_keeps_mutable_state_outside_immutable_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "product"
            state = root / "state"
            native = {
                "prefix": "/native/product",
                "base": {"prefix": "/native/base"},
            }
            MODULE.write_activation(prefix, native, state)
            activation = (
                prefix / "etc/conda/activate.d/amdgpu-sim.sh"
            ).read_text(encoding="ascii")
            self.assertIn(f"TRITON_CACHE_DIR={state}/triton-cache", activation)
            self.assertNotIn(f"TRITON_CACHE_DIR={prefix}", activation)
            self.assertIn("export HIP_PLATFORM=amd", activation)
            self.assertIn("export HIP_PATH=/native/product", activation)
            self.assertIn("export HSA_PATH=/native/product", activation)
            self.assertIn("export HSA_ENABLE_DTIF_FAST_COPY=0", activation)
            self.assertIn(
                "export HSA_MODEL_LIB=/native/product/lib/libself_amdgpu_hsakmt_model.so.1",
                activation,
            )
            self.assertIn(
                "export LD_LIBRARY_PATH=/native/product/lib:/native/base/lib",
                activation,
            )
            self.assertFalse(state.exists())

    def test_product_id_is_canonical_and_sensitive(self) -> None:
        left = {"schema": "x", "items": {"a": 1, "b": 2}}
        right = {"items": {"b": 2, "a": 1}, "schema": "x"}
        self.assertEqual(MODULE.product_id(left), MODULE.product_id(right))
        right["items"]["b"] = 3
        self.assertNotEqual(MODULE.product_id(left), MODULE.product_id(right))

    def test_product_process_environment_is_isolated_and_external(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "product"
            state = root / "state"
            environment = MODULE.product_environment(
                prefix,
                {"prefix": "/native/product", "base": {"prefix": "/native/base"}},
                state,
            )
            self.assertEqual(environment["HOME"], str(state / "home"))
            self.assertEqual(environment["TMPDIR"], str(state / "tmp"))
            self.assertEqual(environment["TRITON_CACHE_DIR"], str(state / "triton-cache"))
            self.assertEqual(environment["HSA_ENABLE_DXG_DETECTION"], "0")
            self.assertEqual(environment["HSA_ENABLE_DTIF_FAST_COPY"], "0")
            self.assertEqual(environment["HSA_ENABLE_INTERRUPT"], "0")
            self.assertEqual(environment["HIP_PLATFORM"], "amd")
            self.assertEqual(environment["HIP_PATH"], "/native/product")
            self.assertEqual(environment["HSA_PATH"], "/native/product")
            self.assertEqual(
                environment["LD_LIBRARY_PATH"],
                "/native/product/lib:/native/base/lib",
            )
            self.assertNotIn("PYTHONPATH", environment)
            self.assertNotIn("CONDA_PREFIX", environment)
            self.assertNotIn("CUDA_HOME", environment)
            self.assertFalse((prefix / "state").exists())

    def test_smi_entry_points_use_only_the_product_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "product"
            (prefix / "bin").mkdir(parents=True)
            entries = MODULE.install_smi_tools(ROOT, prefix)
            wrapper = (prefix / "bin/rocm-smi").read_text(encoding="ascii")
            self.assertIn(f'exec "{prefix}/bin/python"', wrapper)
            self.assertNotIn("/usr/bin/python", wrapper)
            self.assertEqual(entries["rocm_smi"], str(prefix / "bin/rocm-smi"))
            self.assertTrue((prefix / "bin/gemsim-smi").is_symlink())
            self.assertEqual(os.readlink(prefix / "bin/gemsim-smi"), "rocm-smi")
            self.assertEqual(
                stat.S_IMODE((prefix / "bin/rocm-smi").stat().st_mode), 0o555
            )


if __name__ == "__main__":
    unittest.main()
