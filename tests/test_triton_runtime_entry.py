# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the transparent Triton example runtime entry."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gemsim_bootstrap_for_tests", ROOT / "examples/triton/_gemsim_bootstrap.py"
)
assert SPEC and SPEC.loader
bootstrap_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap_module)


class ExecveIntercepted(RuntimeError):
    pass


class TritonRuntimeEntryTest(unittest.TestCase):
    def test_quickstarts_bootstrap_before_framework_imports(self) -> None:
        for relative, framework_import in (
            ("examples/quickstart/triton_vecadd.py", "import torch"),
            ("examples/quickstart/vllm_silu.py", "import torch"),
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            bootstrap = source.index('["bootstrap"](__file__')
            framework = source.index(framework_import)
            self.assertLess(bootstrap, framework, relative)
            self.assertIn(
                'parents[1] / "triton/_gemsim_bootstrap.py"', source, relative
            )

    @staticmethod
    def write_rank_descriptor(path: Path, cache: Path) -> None:
        instance = path.parent / "instance"
        runtime = instance / "runtime"
        document = {
            "schema": "amdgpu-sim.gemsim-rank-launch.v1",
            "job_uuid": "01" * 16,
            "epoch": 1,
            "rank": 0,
            "world_size": 2,
            "paths": {
                "instance_directory": str(instance),
                "triton_cache_directory": str(cache),
                "runtime_directory": str(runtime),
                "endpoint": str(runtime / "bridge.sock"),
                "gem5_output_directory": str(runtime / "m5out"),
                "dispatch_trace_path": str(runtime / "dispatch-trace.jsonl"),
                "gem5_log_path": str(runtime / "gem5.log"),
                "gem5_cache_directory": str(runtime / "cache"),
            },
        }
        path.write_bytes(
            (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "ascii"
            )
        )
        path.chmod(stat.S_IRUSR)

    @staticmethod
    def write_ccl_bootstrap_descriptor(path: Path) -> None:
        document = {
            "schema": "amdgpu-sim.vllm-ccl-bootstrap.v1",
            "product": {"manifest_sha256": "01" * 32},
            "groups": [{"unique_name": "tp:0"}],
        }
        path.write_bytes(
            (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "ascii"
            )
        )
        path.chmod(stat.S_IRUSR)

    def test_bootstrap_execs_with_only_runtime_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            script = root / "examples/triton/example.py"
            prefix = root / "prefix"
            cache = root / "cache/triton"
            script.parent.mkdir(parents=True)
            script.touch()
            (prefix / "venv/bin").mkdir(parents=True)
            (prefix / "venv/bin/python").touch()
            ambient = {
                "HOME": str(root / "ambient-home"),
                "TRITON_CACHE_DIR": str(cache),
                "CC": "/usr/lib/ccache/clang",
                "CXX": "/usr/lib/ccache/clang++",
                "CCACHE_DIR": "/tmp/ccache",
                "LD": "/usr/bin/ld.lld",
                "LDFLAGS": "-fuse-ld=lld",
                "MAX_JOBS": "24",
                "ROCM_SIM_ROOT": str(prefix),
                "CUDA_HOME": "/usr/local/cuda",
                "CONDA_PREFIX": "/ambient/conda",
                "LD_LIBRARY_PATH": "/ambient/lib",
                "SAGR_GEM5_BINARY": "/wrong/gem5.opt",
                "TRITON_DEFAULT_BACKEND": "gemsim_amd",
                "_AMDGPU_SIM_BOOTSTRAPPED": "1",
            }
            with (
                mock.patch.dict(os.environ, ambient, clear=True),
                mock.patch.object(
                    bootstrap_module.sys,
                    "executable",
                    str(prefix / "venv/bin/python"),
                ),
                mock.patch.object(
                    bootstrap_module.sys,
                    "argv",
                    [str(script), "--case", "decode"],
                ),
                mock.patch.object(bootstrap_module, "_prefix", return_value=prefix),
                mock.patch.object(
                    bootstrap_module.pwd,
                    "getpwuid",
                    return_value=SimpleNamespace(pw_dir=str(root / "user-home")),
                ),
                mock.patch.object(
                    bootstrap_module.os,
                    "execve",
                    side_effect=ExecveIntercepted,
                ) as execve,
            ):
                with self.assertRaises(ExecveIntercepted):
                    bootstrap_module.bootstrap(str(script), "example-cache")

            executable, argv, environment = execve.call_args.args
            self.assertEqual(executable, prefix / "venv/bin/python")
            self.assertEqual(
                argv,
                [
                    str(prefix / "venv/bin/python"),
                    "-I",
                    str(script),
                    "--case",
                    "decode",
                ],
            )
            self.assertEqual(
                environment, bootstrap_module._runtime_environment(prefix, cache)
            )
            for name in (
                "CC",
                "CXX",
                "LD",
                "LDFLAGS",
                "MAX_JOBS",
                "CUDA_HOME",
                "CONDA_PREFIX",
                "LD_LIBRARY_PATH",
            ):
                self.assertNotIn(name, environment)
            self.assertFalse(any(name.startswith("CCACHE_") for name in environment))
            self.assertFalse(any(name.startswith("SAGR_") for name in environment))

    def test_clean_private_reentry_does_not_probe_or_exec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            script = root / "examples/triton/example.py"
            prefix = root / "prefix"
            cache = root / "persistent-cache"
            script.parent.mkdir(parents=True)
            script.touch()
            environment = bootstrap_module._runtime_environment(prefix, cache)
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(
                    bootstrap_module.sys,
                    "executable",
                    str(prefix / "venv/bin/python"),
                ),
                mock.patch.object(bootstrap_module, "_prefix") as prefix_probe,
                mock.patch.object(bootstrap_module.os, "execve") as execve,
            ):
                bootstrap_module.bootstrap(str(script), "example-cache")
            prefix_probe.assert_not_called()
            execve.assert_not_called()

    def test_rank_descriptor_is_the_only_supervisor_input_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            script = root / "examples/triton/example.py"
            prefix = root / "prefix"
            cache = root / "cache/triton"
            descriptor = root / "rank-launch.json"
            script.parent.mkdir(parents=True)
            script.touch()
            (prefix / "venv/bin").mkdir(parents=True)
            (prefix / "venv/bin/python").touch()
            self.write_rank_descriptor(descriptor, cache)
            ambient = {
                "TRITON_CACHE_DIR": str(cache),
                "GEMSIM_RANK_LAUNCH_DESCRIPTOR": str(descriptor),
                "RANK": "9",
                "WORLD_SIZE": "99",
                "MASTER_ADDR": "wrong",
                "SAGR_GENERIC_BRIDGE_ENDPOINT": "/wrong.sock",
            }
            with (
                mock.patch.dict(os.environ, ambient, clear=True),
                mock.patch.object(bootstrap_module.sys, "argv", [str(script)]),
                mock.patch.object(bootstrap_module, "_prefix", return_value=prefix),
                mock.patch.object(
                    bootstrap_module.os, "execve", side_effect=ExecveIntercepted
                ) as execve,
            ):
                with self.assertRaises(ExecveIntercepted):
                    bootstrap_module.bootstrap(str(script), "example-cache")
            environment = execve.call_args.args[2]
            self.assertEqual(
                environment["GEMSIM_RANK_LAUNCH_DESCRIPTOR"], str(descriptor)
            )
            for name in (
                "RANK",
                "WORLD_SIZE",
                "MASTER_ADDR",
                "SAGR_GENERIC_BRIDGE_ENDPOINT",
            ):
                self.assertNotIn(name, environment)
            self.assertEqual(environment["TRITON_CACHE_DIR"], str(cache))
            rank_state = descriptor.parent / "instance/state"
            self.assertEqual(environment["HOME"], str(rank_state / "home"))
            self.assertEqual(environment["TMPDIR"], str(rank_state / "tmp"))

    def test_rank_private_tmpdir_keeps_vllm_ipc_uuid_within_sun_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            descriptor = root / "rank-launch.json"
            cache = root / "cache/triton"
            self.write_rank_descriptor(descriptor, cache)
            with mock.patch.dict(
                os.environ,
                {
                    "GEMSIM_RANK_LAUNCH_DESCRIPTOR": str(descriptor),
                    "TRITON_CACHE_DIR": str(cache),
                },
                clear=True,
            ):
                _, _, state = bootstrap_module._rank_descriptor()
            temporary_path = state / "tmp"
            self.assertLessEqual(
                len(os.fsencode(temporary_path)) + 1 + 36,
                bootstrap_module._UNIX_SOCKET_PATH_MAX_BYTES,
            )

    def test_rank_descriptor_rejects_tmpdir_too_long_for_vllm_ipc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / ("x" * 70)
            root.mkdir()
            descriptor = root / "rank-launch.json"
            cache = root / "cache/triton"
            self.write_rank_descriptor(descriptor, cache)
            with mock.patch.dict(
                os.environ,
                {
                    "GEMSIM_RANK_LAUNCH_DESCRIPTOR": str(descriptor),
                    "TRITON_CACHE_DIR": str(cache),
                },
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "vLLM IPC UUID"):
                    bootstrap_module._rank_descriptor()

    def test_private_ccl_bootstrap_descriptor_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            script = root / "examples/triton/example.py"
            prefix = root / "prefix"
            cache = root / "persistent-cache"
            descriptor = root / "ccl-bootstrap.json"
            script.parent.mkdir(parents=True)
            script.touch()
            (prefix / "venv/bin").mkdir(parents=True)
            (prefix / "venv/bin/python").touch()
            self.write_ccl_bootstrap_descriptor(descriptor)
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "TRITON_CACHE_DIR": str(cache),
                        "GEMSIM_CCL_BOOTSTRAP_DESCRIPTOR": str(descriptor),
                        "SAGR_CCL_CAPABILITY_FD": "99",
                        "CUDA_HOME": "/usr/local/cuda",
                    },
                    clear=True,
                ),
                mock.patch.object(bootstrap_module.sys, "argv", [str(script)]),
                mock.patch.object(bootstrap_module, "_prefix", return_value=prefix),
                mock.patch.object(
                    bootstrap_module.os, "execve", side_effect=ExecveIntercepted
                ) as execve,
            ):
                with self.assertRaises(ExecveIntercepted):
                    bootstrap_module.bootstrap(str(script), "example-cache")
            environment = execve.call_args.args[2]
            self.assertEqual(
                environment["GEMSIM_CCL_BOOTSTRAP_DESCRIPTOR"], str(descriptor)
            )
            self.assertNotIn("SAGR_CCL_CAPABILITY_FD", environment)
            self.assertNotIn("CUDA_HOME", environment)

    def test_ccl_bootstrap_descriptor_rejects_nonprivate_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            descriptor = Path(temporary).resolve() / "ccl-bootstrap.json"
            self.write_ccl_bootstrap_descriptor(descriptor)
            descriptor.chmod(0o644)
            with mock.patch.dict(
                os.environ,
                {"GEMSIM_CCL_BOOTSTRAP_DESCRIPTOR": str(descriptor)},
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "private owned file"):
                    bootstrap_module._ccl_bootstrap_descriptor_path()

    def test_rank_descriptor_cache_is_the_single_source_of_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            descriptor = root / "rank-launch.json"
            descriptor_cache = root / "cache/triton"
            self.write_rank_descriptor(descriptor, descriptor_cache)
            with mock.patch.dict(
                os.environ,
                {
                    "GEMSIM_RANK_LAUNCH_DESCRIPTOR": str(descriptor),
                    "TRITON_CACHE_DIR": str(root / "ambient-cache"),
                },
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    bootstrap_module._rank_descriptor()

    def test_rank_descriptor_rejects_nonprivate_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            descriptor = Path(temporary).resolve() / "rank-launch.json"
            descriptor.write_text("{}\n", encoding="ascii")
            descriptor.chmod(0o644)
            with mock.patch.dict(
                os.environ,
                {"GEMSIM_RANK_LAUNCH_DESCRIPTOR": str(descriptor)},
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "private owned file"):
                    bootstrap_module._rank_descriptor_path()

    def test_prefix_probe_cannot_start_a_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            prefix = root / "prefix"
            (prefix / "venv/bin").mkdir(parents=True)
            (prefix / "venv/bin/python").touch()
            with mock.patch.object(
                bootstrap_module.subprocess,
                "check_output",
                return_value=f"{prefix}\n",
            ) as check_output:
                self.assertEqual(bootstrap_module._prefix(root), prefix)
            self.assertEqual(
                check_output.call_args.args[0],
                [
                    "/usr/bin/bash",
                    str(root / "scripts/setup_rocm_env.sh"),
                    "--print-prefix",
                ],
            )

    def test_product_uses_base_python_and_snapshot_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            script = root / "examples/triton/example.py"
            product_prefix = root / "product"
            base = root / "base"
            state = root / "state"
            cache = root / "persistent-cache"
            python = base / "venv/bin/python"
            launcher = product_prefix / "python/product_bootstrap.py"
            script.parent.mkdir(parents=True)
            python.parent.mkdir(parents=True)
            launcher.parent.mkdir(parents=True)
            script.touch()
            python.touch()
            launcher.write_text("# launcher\n", encoding="ascii")
            document = {
                "schema": "amdgpu-sim.product-prefix.v1",
                "prefix": str(product_prefix),
                "base": {
                    "prefix": str(base),
                    "python": {"path": str(python)},
                },
                "runtime_state_root": str(state),
                "plugins": {"bootstrap": str(launcher)},
            }
            manifest = product_prefix / "manifest.json"
            manifest.write_bytes(bootstrap_module._canonical_json(document))
            manifest.chmod(0o400)
            ambient = {
                "TRITON_CACHE_DIR": str(cache),
                "CUDA_HOME": "/usr/local/cuda",
                "CONDA_PREFIX": "/ambient/conda",
                "LD_LIBRARY_PATH": "/ambient/lib",
            }
            with (
                mock.patch.dict(os.environ, ambient, clear=True),
                mock.patch.object(bootstrap_module.sys, "argv", [str(script)]),
                mock.patch.object(
                    bootstrap_module, "_prefix", return_value=product_prefix
                ),
                mock.patch.object(
                    bootstrap_module.os, "execve", side_effect=ExecveIntercepted
                ) as execve,
            ):
                with self.assertRaises(ExecveIntercepted):
                    bootstrap_module.bootstrap(str(script), "product-cache")
            executable, argv, environment = execve.call_args.args
            self.assertEqual(executable, python)
            self.assertEqual(
                argv,
                [str(python), "-I", str(launcher), str(script)],
            )
            self.assertEqual(environment["ROCM_SIM_ROOT"], str(product_prefix))
            for name in ("ROCM_PATH", "HIP_PATH", "HSA_PATH"):
                self.assertEqual(environment[name], str(base))
            self.assertEqual(environment["HSA_ENABLE_DXG_DETECTION"], "0")
            self.assertEqual(environment["HSA_ENABLE_INTERRUPT"], "0")
            for name in ("CUDA_HOME", "CONDA_PREFIX", "LD_LIBRARY_PATH"):
                self.assertNotIn(name, environment)
            self.assertEqual(environment["HOME"], str(state / "home"))

    def test_conda_entry_is_a_clean_non_build_entry(self) -> None:
        wrapper = (ROOT / "scripts/setup_conda_env.sh").read_text(encoding="utf-8")
        builder = (ROOT / "tools/conda_product_environment.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("tools/conda_product_environment.py", wrapper)
        self.assertIn('group.add_argument("--print-prefix"', builder)
        self.assertIn('"PYTHONNOUSERSITE": "1"', builder)
        self.assertIn('state / "triton-cache"', builder)
        self.assertIn("__editable__", builder)
        for token in (
            "CC=",
            "CXX=",
            "CCACHE_",
            "LD=",
            "LDFLAGS=",
            "MAX_JOBS=",
            "--all",
            "cmake ",
            "ninja ",
            "scons ",
            "make ",
        ):
            self.assertNotIn(token, wrapper)


if __name__ == "__main__":
    unittest.main()
