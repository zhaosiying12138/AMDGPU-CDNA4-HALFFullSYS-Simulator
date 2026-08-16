# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the content-addressed simulator product overlay."""

from __future__ import annotations

import importlib.util
from contextlib import ExitStack
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "product_environment_for_tests", ROOT / "tools/product_environment.py"
)
assert SPEC and SPEC.loader
product = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(product)


class ProductEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        (self.root / "env/rocm").mkdir(parents=True)
        (self.root / "scripts").mkdir()
        (self.root / "scripts/setup_rocm_env.sh").write_bytes(b"#!/bin/sh\n")
        self.base = self.root / "env/rocm/base-v8"
        self.base.mkdir()
        (self.root / "projects/gem5/build/VEGA_X86").mkdir(parents=True)
        (self.root / "projects/gem5/configs/example/gemsim").mkdir(parents=True)
        (self.root / "projects/self-amdgpu-runtime").mkdir(parents=True)
        self.gem5_binary = self.root / "projects/gem5/build/VEGA_X86/gem5.opt"
        self.gem5_config = (
            self.root / "projects/gem5/configs/example/gemsim/host_dispatch.py"
        )
        self.gem5_binary.write_bytes(b"gem5\n")
        self.gem5_binary.chmod(0o755)
        self.gem5_config.write_bytes(b"config\n")
        external_inputs = {
            "projects/rocm-systems/projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/gfx950/binary_search_kernels.hsaco": b"binary search hsaco\n",
            "projects/rocm-systems/projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/gfx950/gpuReadWrite_kernels.hsaco": b"gpu read write hsaco\n",
            "projects/triton/python/tutorials/01-vector-add.py": b"print('vecadd')\n",
            "artifacts/evidence/CP-0017/triton-builds/vecadd-gfx950.hsaco": b"triton vecadd hsaco\n",
        }
        for relative, payload in external_inputs.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        self.plugin_files = {}
        for name, relative in product.PLUGIN_SOURCES.items():
            source = self.root / relative
            if name == "triton-gemsim-amd":
                package = source / "backend"
                files = {
                    "__init__.py": b"SNAPSHOT = 'triton'\n",
                    "driver.py": b"DRIVER = 'snapshot'\n",
                    "compiler.py": b"COMPILER = 'snapshot'\n",
                    "name.conf": b"gemsim_amd\n",
                }
            else:
                package_name = "gemsim_vllm" if name == "gemsim-vllm" else "gemsim_ccl"
                package = source / "src" / package_name
                files = {
                    "__init__.py": f"SNAPSHOT = {name!r}\n".encode("ascii"),
                    "module.py": b"VALUE = 'snapshot'\n",
                }
                (source / "pyproject.toml").parent.mkdir(parents=True, exist_ok=True)
                (source / "pyproject.toml").write_bytes(b"[project]\n")
            package.mkdir(parents=True, exist_ok=True)
            for filename, payload in files.items():
                (package / filename).write_bytes(payload)
            self.plugin_files[name] = package / "__init__.py"
        self.base_binding = {
            "prefix": str(self.base),
            "manifest": {
                "path": str(self.base / "manifest.json"),
                "bytes": 1,
                "sha256": "1" * 64,
            },
            "setup_schema": 8,
            "critical_artifacts": {},
            "python": {
                "path": str(self.base / "venv/bin/python"),
                "bytes": 1,
                "sha256": "2" * 64,
            },
            "triton_plugin": {"source": {}, "package": {}},
        }
        self.repository_sets = {
            "gem5": self.source_set("gem5", "3"),
            "self-amdgpu-runtime": self.source_set("runtime", "4"),
            "rocm-systems": self.source_set("rocm-systems", "5"),
        }
        self.repository_sets["rocm-systems"]["repository"] = str(
            self.root / "projects/rocm-systems"
        )
        self.facade_stage = {
            "schema": product.FACADE_STAGE_SCHEMA,
            "directory": str(self.root / "env/rocm/hip-facade-stage-v1"),
            "file_count": 0,
            "files": [],
            "source_set_sha256": "6" * 64,
        }
        self.facade_build = {
            "schema": "amdgpu-sim.hip-facade-build.v1",
            "prefix": str(self.root / "env/conda/hip-facade-build-deps"),
            "lock": {"path": str(self.root / "config/hip.lock")},
            "packages": [],
        }
        self.facade_tool = {
            "path": str(self.root / "projects/self-amdgpu-runtime/tools/hsakmt-model-topology.py"),
            "bytes": 1,
            "sha256": "7" * 64,
        }
        self.facade = {
            "stage": self.facade_stage,
            "build": self.facade_build,
            "rocm_systems": {
                "repository": str(self.root / "projects/rocm-systems"),
                "head": "5" * 40,
                "tree": "5" * 40,
                "source_set_sha256": "5" * 64,
            },
            "topology_tool": self.facade_tool,
            "rocr_aql_prerequisite": {"schema": "test-rocr"},
            "gpu_topology": {"gpu_count": 1, "architecture": "gfx950"},
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def source_set(name: str, digit: str) -> dict[str, object]:
        return {
            "schema": "amdgpu-sim.repository-source-set.v1",
            "repository": f"/{name}",
            "head": digit * 40,
            "tree": digit * 40,
            "file_count": 1,
            "source_set_sha256": digit * 64,
            "files": [{"path": name, "sha256": digit * 64}],
        }

    def repository_source_set(
        self, path: Path, *, allow_gitlinks: bool = False
    ) -> dict[str, object]:
        if path.name == "gem5":
            name = "gem5"
        elif path.name == "rocm-systems":
            name = "rocm-systems"
        else:
            name = "self-amdgpu-runtime"
        return self.repository_sets[name]

    def patches(self):
        return (
            mock.patch.object(product, "_base_binding", return_value=self.base_binding),
            mock.patch.object(
                product,
                "repository_source_set",
                side_effect=self.repository_source_set,
            ),
            mock.patch.object(product, "facade_stage_source_set", return_value=self.facade_stage),
            mock.patch.object(product, "_facade_build_binding", return_value=self.facade_build),
            mock.patch.object(product, "_regular_tree_binding", return_value=self.facade_tool),
            mock.patch.object(product, "_rocr_aql_binding", return_value=self.facade["rocr_aql_prerequisite"]),
        )

    def freeze(self) -> Path:
        patches = self.patches()
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            return product.freeze_product(self.root, self.base)

    def write_active(self, product_id: str, prefix: Path) -> bytes:
        prefix.mkdir(exist_ok=True)
        record = {
            "schema": product.ACTIVE_SCHEMA,
            "product_id": product_id,
            "prefix": str(prefix),
            "manifest_sha256": "5" * 64,
        }
        payload = product.canonical_json(record)
        path = self.root / "env/rocm/active-product"
        path.write_bytes(payload)
        path.chmod(0o600)
        return payload

    def test_print_prefix_is_constant_time_and_falls_back_only_when_absent(self) -> None:
        fallback = product.print_prefix(self.root, self.base)
        self.assertEqual(fallback, self.base)
        product_id = "a" * 64
        prefix = self.root / "env/rocm" / f"product-v1-{product_id}"
        self.write_active(product_id, prefix)
        with (
            mock.patch.object(
                product, "_regular_sha256", side_effect=AssertionError("hash probe")
            ),
            mock.patch.object(
                product.subprocess,
                "run",
                side_effect=AssertionError("git/build probe"),
            ),
        ):
            self.assertEqual(product.print_prefix(self.root, self.base), prefix)
        (self.root / "env/rocm/active-product").write_bytes(b"{}\n")
        with self.assertRaises(product.ProductEnvironmentError):
            product.print_prefix(self.root, self.base)

    def test_activation_selects_standard_amd_facade_without_host_device_probe(self) -> None:
        prefix = self.root / "env/rocm/product"
        prefix.mkdir()
        activation = prefix / "activate"
        product._write_activation(activation, prefix, self.base)
        text = activation.read_text(encoding="ascii")
        self.assertIn("export HIP_PLATFORM=amd", text)
        self.assertIn("export HSA_ENABLE_DXG_DETECTION=0", text)
        self.assertIn("export HSA_ENABLE_DTIF_FAST_COPY=0", text)
        self.assertIn("export HSA_ENABLE_INTERRUPT=0", text)
        self.assertIn(f"export HIP_PATH={prefix}", text)
        self.assertIn(
            f"export HSA_MODEL_LIB={prefix}/lib/libself_amdgpu_hsakmt_model.so.1",
            text,
        )

    def test_freeze_records_full_sources_and_output_must_be_absent(self) -> None:
        prefix = self.freeze()
        frozen = json.loads((self.root / "env/rocm/frozen-product").read_bytes())
        lock_path = Path(frozen["source_lock"])
        source_lock = json.loads(lock_path.read_bytes())
        self.assertEqual(set(source_lock["plugins"]), set(product.PLUGIN_SOURCES))
        self.assertEqual(
            set(source_lock["repositories"]),
            {"gem5", "self-amdgpu-runtime", "rocm-systems"},
        )
        self.assertEqual(
            source_lock["identity"]["base"]["manifest"]["sha256"], "1" * 64
        )
        self.assertEqual(prefix.name, f"product-v1-{source_lock['product_id']}")
        prefix.mkdir()
        patches = self.patches()
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            with self.assertRaisesRegex(
                product.ProductEnvironmentError, "output already exists"
            ):
                product.freeze_product(self.root, self.base)

    def test_plugin_drift_after_freeze_rejects_before_build_and_preserves_active(self) -> None:
        self.freeze()
        old_id = "b" * 64
        old_prefix = self.root / "env/rocm" / f"product-v1-{old_id}"
        active_before = self.write_active(old_id, old_prefix)
        self.plugin_files["gemsim-vllm"].write_bytes(b"SNAPSHOT = 'drift'\n")
        patches = self.patches()
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            build = stack.enter_context(mock.patch.object(product, "_build_runtime"))
            with self.assertRaisesRegex(
                product.ProductEnvironmentError, "gemsim-vllm actual source set drifted"
            ):
                product.build_product(self.root, self.base)
        build.assert_not_called()
        self.assertEqual(
            (self.root / "env/rocm/active-product").read_bytes(), active_before
        )

    def test_gem5_and_base_drift_reject_before_build(self) -> None:
        self.freeze()
        self.gem5_binary.write_bytes(b"changed gem5\n")
        patches = self.patches()
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            build = stack.enter_context(mock.patch.object(product, "_build_runtime"))
            with self.assertRaisesRegex(
                product.ProductEnvironmentError, "gem5 binary or configuration drifted"
            ):
                product.build_product(self.root, self.base)
        build.assert_not_called()
        self.gem5_binary.write_bytes(b"gem5\n")
        changed_base = dict(self.base_binding)
        changed_base["manifest"] = dict(self.base_binding["manifest"])
        changed_base["manifest"]["sha256"] = "f" * 64
        patches = self.patches()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(product, "_base_binding", return_value=changed_base))
            for patcher in patches[1:]:
                stack.enter_context(patcher)
            build = stack.enter_context(mock.patch.object(product, "_build_runtime"))
            with self.assertRaisesRegex(
                product.ProductEnvironmentError, "base prefix drifted"
            ):
                product.build_product(self.root, self.base)
        build.assert_not_called()

    def test_runtime_source_drift_rejects_before_build(self) -> None:
        self.freeze()
        self.repository_sets["self-amdgpu-runtime"] = self.source_set(
            "runtime", "9"
        )
        patches = self.patches()
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            build = stack.enter_context(mock.patch.object(product, "_build_runtime"))
            with self.assertRaisesRegex(
                product.ProductEnvironmentError,
                "self-amdgpu-runtime actual source set drifted",
            ):
                product.build_product(self.root, self.base)
        build.assert_not_called()

    def test_runtime_build_source_is_a_read_only_frozen_snapshot(self) -> None:
        live = self.root / "runtime-live"
        source = live / "src/runtime.c"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"frozen runtime\n")
        artifact = product._regular_sha256(source)
        source_set = {
            "schema": "amdgpu-sim.repository-source-set.v1",
            "repository": str(live),
            "head": "1" * 40,
            "tree": "2" * 40,
            "file_count": 1,
            "source_set_sha256": "3" * 64,
            "files": [
                {
                    "path": "src/runtime.c",
                    "kind": "regular",
                    "executable": False,
                    "bytes": artifact["bytes"],
                    "sha256": artifact["sha256"],
                    "index_mode": "100644",
                }
            ],
        }
        destination = self.root / "env/product-build/runtime/source"
        snapshot = product._materialize_repository_snapshot(
            source_set, destination
        )
        source.write_bytes(b"workspace drift\n")
        self.assertEqual((snapshot / "src/runtime.c").read_bytes(), b"frozen runtime\n")
        self.assertEqual(stat.S_IMODE(snapshot.lstat().st_mode), 0o555)
        self.assertEqual(
            stat.S_IMODE((snapshot / "src/runtime.c").lstat().st_mode), 0o444
        )

    def test_runtime_build_path_preserves_upstream_hsakmt_path_capacity(self) -> None:
        source_sha = "a" * 64
        tool_sha = "b" * 64
        build_root = product._runtime_build_root(self.root, source_sha)
        topology = product._validate_hsakmt_topology_build_path(
            build_root / "build", tool_sha
        )
        longest = os.fsencode(
            f"{topology}{product.HSAKMT_TOPOLOGY_LONGEST_RELATIVE_PATH}"
        )
        self.assertLess(len(longest), product.HSAKMT_TOPOLOGY_PATH_BUFFER_BYTES)
        self.assertEqual(
            build_root,
            self.root / product.PRODUCT_BUILD_CACHE_RELATIVE / source_sha,
        )

        too_long = Path("/tmp") / ("x" * 180) / "build"
        with self.assertRaisesRegex(
            product.ProductEnvironmentError, "libhsakmt topology path capacity"
        ):
            product._validate_hsakmt_topology_build_path(too_long, tool_sha)

    def test_reused_upstream_subtree_is_reverified(self) -> None:
        live = self.root / "rocm-live"
        header = live / "include/hsakmt/model.h"
        header.parent.mkdir(parents=True)
        header.write_bytes(b"frozen header\n")
        artifact = product._regular_sha256(header)
        source_set = {
            "schema": "amdgpu-sim.repository-source-set.v1",
            "repository": str(live),
            "head": "1" * 40,
            "tree": "2" * 40,
            "file_count": 2,
            "source_set_sha256": "3" * 64,
            "files": [
                {
                    "path": "include/hsakmt/model.h",
                    "kind": "regular",
                    "executable": False,
                    "bytes": artifact["bytes"],
                    "sha256": artifact["sha256"],
                    "index_mode": "100644",
                },
                {
                    "path": "projects/unrelated-gitlink",
                    "kind": "gitlink",
                    "bytes": 40,
                    "sha256": "4" * 64,
                    "index_mode": "160000",
                },
            ],
        }
        destination = self.root / "env/pb/headers"
        product._materialize_source_subtree(source_set, "include/hsakmt", destination)
        snapshotted = destination / "model.h"
        snapshotted.chmod(0o600)
        snapshotted.write_bytes(b"tampered header\n")
        with self.assertRaisesRegex(
            product.ProductEnvironmentError, "subtree snapshot drifted"
        ):
            product._materialize_source_subtree(
                source_set, "include/hsakmt", destination
            )

    @staticmethod
    def mock_runtime_build(
        root: Path,
        source_lock: dict[str, object],
        target: Path,
        staging_root: Path,
        staged_prefix: Path,
    ) -> None:
        files = {
            "bin/sagr-handshake": b"handshake\n",
            "bin/sagr-triton-hsaco-probe": b"probe\n",
            "bin/opencl-vecadd": b"opencl\n",
            "include/self_amdgpu_runtime/runtime.h": b"runtime header\n",
            "include/CL/cl.h": b"opencl header\n",
            "lib/libself_amdgpu_runtime.so.0.8.0": b"runtime dso\n",
            "lib/libOpenCL.so.1.2.0": b"opencl dso\n",
            "lib/libhsa-runtime64.so.1": b"rocr dso\n",
            "lib/libamdhip64.so.7": b"hip dso\n",
            "lib/libamd_comgr.so.3": b"comgr dso\n",
            "lib/librccl.so.1": b"rccl dso\n",
            "lib/libself_amdgpu_hsakmt_model.so.1": b"model dso\n",
            "libexec/amdgpu-sim/generic-dispatch-v2-endpoint-test": b"endpoint\n",
            "share/self-amdgpu-runtime/opencl/vecadd.cl": b"kernel\n",
            "share/self-amdgpu-runtime/hsakmt-topology/manifest.json": b"{\"gpu_count\":1}\n",
            "include/hip/hip_runtime.h": b"hip header\n",
            "include/hsa/hsa.h": b"hsa header\n",
            "include/hsakmt/hsakmtmodeliface.h": b"hsakmt header\n",
            "include/nccl.h": b"nccl header\n",
        }
        for relative, payload in files.items():
            path = staged_prefix / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            if relative.startswith(("bin/", "libexec/")):
                path.chmod(0o755)
        (staged_prefix / "lib/libself_amdgpu_runtime.so.1").symlink_to(
            "libself_amdgpu_runtime.so.0.8.0"
        )
        (staged_prefix / "lib/libOpenCL.so.1").symlink_to("libOpenCL.so.1.2.0")

    def test_mock_build_publishes_snapshot_inventory_and_active_last(self) -> None:
        prefix = self.freeze()
        patches = self.patches()
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            stack.enter_context(
                mock.patch.object(product, "_build_runtime", side_effect=self.mock_runtime_build)
            )
            result = product.build_product(self.root, self.base)
        self.assertEqual(result, prefix)
        manifest = json.loads((prefix / "manifest.json").read_bytes())
        self.assertEqual(manifest["schema"], product.PRODUCT_SCHEMA)
        self.assertEqual(
            manifest["base"]["python"]["path"],
            str(self.base / "venv/bin/python"),
        )
        self.assertEqual(
            manifest["managed_inputs"]["gem5_binary"]["path"],
            str(self.gem5_binary),
        )
        self.assertEqual(
            manifest["artifacts"]["runtime_soname"]["path"],
            str(prefix / "lib/libself_amdgpu_runtime.so.1"),
        )
        self.assertEqual(
            set(manifest["plugins"]["snapshots"]), set(product.PLUGIN_SOURCES)
        )
        self.assertEqual(
            set(manifest["plugins"]["sources"]), set(product.PLUGIN_SOURCES)
        )
        package_paths = {
            "triton-gemsim-amd": "python/triton/backends/gemsim_amd",
            "gemsim-vllm": "python/gemsim_vllm",
            "gemsim-ccl": "python/gemsim_ccl",
        }
        for name, relative in product.PLUGIN_SOURCES.items():
            descriptor = manifest["plugins"]["snapshots"][name]
            self.assertEqual(
                descriptor["source_snapshot"], str(prefix / "python/source" / name)
            )
            self.assertEqual(
                descriptor["package_path"], str(prefix / package_paths[name])
            )
            self.assertEqual(
                manifest["plugins"]["sources"][name]["directory"],
                str((self.root / relative).resolve()),
            )
        self.assertTrue((prefix / "python/gemsim_vllm/ops.py").exists() is False)
        snapshot = prefix / "python/gemsim_vllm/__init__.py"
        self.assertIn("gemsim-vllm", snapshot.read_text(encoding="ascii"))
        inventory_paths = {record["path"] for record in manifest["inventory"]}
        self.assertIn("python/gemsim_ccl/__init__.py", inventory_paths)
        self.assertNotIn("manifest.json", inventory_paths)
        active = json.loads((self.root / "env/rocm/active-product").read_bytes())
        self.assertEqual(active["prefix"], str(prefix))
        self.assertFalse((prefix / "venv").exists())
        self.assertFalse((prefix / "build").exists())
        launcher = (prefix / "python/product_bootstrap.py").read_text(
            encoding="ascii"
        )
        self.assertIn(
            "sys.path[:0] = [str(overlay), str(script.parent)]", launcher
        )
        self.assertIn("ProductTritonBackendFinder", launcher)
        for relative in product.PLUGIN_SOURCES.values():
            self.assertIn(str((self.root / relative).resolve()), launcher)

    def test_failed_mock_build_leaves_old_active_unchanged(self) -> None:
        self.freeze()
        old_id = "c" * 64
        old_prefix = self.root / "env/rocm" / f"product-v1-{old_id}"
        active_before = self.write_active(old_id, old_prefix)
        patches = self.patches()
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            stack.enter_context(
                mock.patch.object(product, "_build_runtime", side_effect=RuntimeError("mock failure"))
            )
            with self.assertRaisesRegex(RuntimeError, "mock failure"):
                product.build_product(self.root, self.base)
        self.assertEqual(
            (self.root / "env/rocm/active-product").read_bytes(), active_before
        )


if __name__ == "__main__":
    unittest.main()
