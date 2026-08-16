from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rocm_pytorch_product_environment",
    ROOT / "tools/rocm_pytorch_product_environment.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RocmPytorchProductEnvironmentTest(unittest.TestCase):
    def make_lock(self, root: Path) -> dict[str, object]:
        cache_relative = Path("artifacts/downloads/test")
        cache = root / cache_relative
        cache.mkdir(parents=True)
        packages = []
        names = (
            "rocm",
            "torch",
            "triton",
            "rocm-sdk-core",
            "rocm-sdk-devel",
            "rocm-sdk-libraries-gfx950-dcgpu",
        )
        for index, name in enumerate(names):
            filename = f"{name.replace('-', '_')}-{index}-py3-none-linux_x86_64.whl"
            path = cache / filename
            with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr(f"{name}-{index}.dist-info/METADATA", f"Name: {name}\nVersion: {index}\n")
            record = {
                "name": name,
                "version": str(index),
                "filename": filename,
                "url": f"https://rocm.nightlies.amd.com/test/{filename}",
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            if name == "rocm":
                record["built_from"] = {
                    "kind": "upstream-sdist",
                    "url": "https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/rocm-7.13.0a20260426.tar.gz",
                    "bytes": 17787,
                    "sha256": "02dfcb9374a27e12d005a91f0cde032bafce49948856fbe96a4cbcff878ddaac",
                    "target_family": "gfx950-dcgpu",
                    "python": "3.12.3",
                    "setuptools": "83.0.0",
                    "wheel": "0.47.0",
                }
            packages.append(record)
        document = {
            "schema": MODULE.LOCK_SCHEMA,
            "python": {"implementation": "CPython", "version": "3.12", "abi": "cp312"},
            "architecture": "gfx950",
            "rocm_version": "7.13.0a20260426",
            "index_url": "https://example.invalid/",
            "cache_directory": cache_relative.as_posix(),
            "packages": packages,
        }
        path = root / MODULE.WHEEL_LOCK_RELATIVE
        path.parent.mkdir(parents=True)
        path.write_bytes(MODULE.canonical_json(document))
        return document

    def test_wheel_lock_is_canonical_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.make_lock(root)
            actual, payload = MODULE.wheel_lock(root)
            self.assertEqual(actual, expected)
            self.assertEqual(payload, MODULE.canonical_json(expected))
            package = expected["packages"][0]
            cache = root / expected["cache_directory"] / package["filename"]
            cache.write_bytes(b"changed")
            with self.assertRaisesRegex(MODULE.ProductError, "differs from lock"):
                MODULE.wheel_lock(root)

    def test_product_environment_replaces_only_rocr(self) -> None:
        prefix = Path("/product")
        native = {
            "prefix": "/native",
            "artifacts": {
                "rocr_library": {"path": "/native/lib/libhsa-runtime64.so.1"},
                "hsakmt_model_library": {
                    "path": "/native/lib/libself_amdgpu_hsakmt_model.so.1"
                },
                "topology_manifest": {
                    "path": "/native/share/self-amdgpu-runtime/hsakmt-topology/manifest.json"
                },
            },
        }
        sdk_libraries = {
            "hip_library": {"path": "/sdk-core/lib/libamdhip64.so.7"},
            "comgr_library": {"path": "/sdk-core/lib/libamd_comgr.so.3"},
            "rccl_library": {"path": "/sdk-libraries/lib/librccl.so.1"},
        }
        sdk = Path("/sdk")
        state = Path("/state")
        environment = MODULE.product_environment(prefix, native, sdk, sdk_libraries, state)
        self.assertEqual(
            environment["LD_LIBRARY_PATH"].split(":"),
            [
                "/sdk-core/lib",
                "/sdk-libraries/lib",
                "/sdk/lib",
                "/sdk/lib/rocm_sysdeps/lib",
                "/product/lib",
                "/native/lib",
            ],
        )
        self.assertEqual(environment["ROCM_PATH"], "/sdk")
        self.assertEqual(
            environment["PKG_CONFIG_PATH"].split(":"),
            [
                "/sdk/lib/rocm_sysdeps/lib/pkgconfig",
                "/sdk/lib/pkgconfig",
                "/sdk/share/pkgconfig",
                "/product/lib/pkgconfig",
                "/product/share/pkgconfig",
            ],
        )
        self.assertEqual(environment["LD_PRELOAD"], "/native/lib/libhsa-runtime64.so.1")
        self.assertEqual(environment["HSA_MODEL_LIB"], "/native/lib/libself_amdgpu_hsakmt_model.so.1")
        self.assertEqual(
            environment["HSA_MODEL_TOPOLOGY"],
            "/native/share/self-amdgpu-runtime/hsakmt-topology",
        )
        self.assertEqual(environment["ROCM_SDK_TARGET_FAMILY"], "gfx950-dcgpu")
        self.assertEqual(environment["HSA_ENABLE_DTIF_FAST_COPY"], "0")
        self.assertNotIn("TRITON_DEFAULT_BACKEND", environment)

    def test_sdk_libraries_use_the_official_rocm_locator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "product"
            (prefix / "bin").mkdir(parents=True)
            (prefix / "bin/python").write_bytes(b"")
            core = prefix / "sdk-core/lib"
            libraries = prefix / "sdk-libraries/lib"
            core.mkdir(parents=True)
            libraries.mkdir(parents=True)
            paths = {
                "amdhip64": core / "libamdhip64.so.7",
                "amd_comgr": core / "libamd_comgr.so.3",
                "rccl": libraries / "librccl.so.1",
            }
            for path in paths.values():
                path.write_bytes(path.name.encode("ascii"))
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps({name: str(path) for name, path in paths.items()}),
                stderr="",
            )
            with mock.patch.object(MODULE, "run", return_value=completed) as invoked:
                records = MODULE.sdk_library_records(prefix)
            command = invoked.call_args.args[0]
            environment = invoked.call_args.kwargs["environment"]
            self.assertEqual(command[:2], [str(prefix / "bin/python"), "-I"])
            self.assertIn("rocm_sdk.find_libraries", command[3])
            self.assertEqual(environment["ROCM_SDK_TARGET_FAMILY"], "gfx950-dcgpu")
            self.assertEqual(records["hip_library"]["provider"], "official_rocm_sdk")
            self.assertEqual(records["comgr_library"]["soname"], "libamd_comgr.so.3")
            self.assertEqual(records["rccl_library"]["shortname"], "rccl")

    def test_apt_sysroot_libraries_and_environment_are_product_local(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "product"
            sdk = prefix / "rocm-sysroot/opt/rocm-7.2.3"
            library_directory = sdk / "lib"
            library_directory.mkdir(parents=True)
            for _, soname, _ in MODULE.SDK_LIBRARY_SPECS.values():
                versioned = library_directory / f"{soname}.test"
                versioned.write_bytes(soname.encode("ascii"))
                (library_directory / soname).symlink_to(versioned.name)
            records = MODULE.apt_sysroot_library_records(prefix, sdk)
            self.assertEqual(
                {record["provider"] for record in records.values()},
                {"official_rocm_apt_sysroot"},
            )
            native = {
                "prefix": "/native",
                "artifacts": {
                    "rocr_library": {"path": "/native/lib/libhsa-runtime64.so.1"},
                    "hsakmt_model_library": {
                        "path": "/native/lib/libself_amdgpu_hsakmt_model.so.1"
                    },
                    "topology_manifest": {
                        "path": "/native/share/self-amdgpu-runtime/hsakmt-topology/manifest.json"
                    },
                },
            }
            environment = MODULE.product_environment(
                prefix, native, sdk, records, Path("/state")
            )
            self.assertEqual(environment["HSA_PATH"], "/native")
            self.assertIn(
                str(prefix / "rocm-sysroot/usr/lib/x86_64-linux-gnu"),
                environment["LD_LIBRARY_PATH"].split(":"),
            )
            self.assertEqual(environment["ROCM_PATH"], str(sdk))

    def test_profiles_keep_legacy_and_vllm_providers_separate(self) -> None:
        legacy = MODULE.profile_spec("rocm713")
        vllm = MODULE.profile_spec("vllm-rocm723")
        self.assertEqual(legacy["provider"], {"kind": "sdk-wheel"})
        self.assertEqual(vllm["provider"]["kind"], "signed-apt-sysroot")
        self.assertIn("vllm", vllm["required_packages"])
        with self.assertRaisesRegex(MODULE.ProductError, "unknown"):
            MODULE.profile_spec("other")

    def test_activation_restores_every_overridden_variable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "product"
            native_root = Path(temporary) / "native"
            native = {
                "prefix": str(native_root),
                "artifacts": {
                    "rocr_library": {"path": str(native_root / "lib/libhsa-runtime64.so.1")},
                    "hsakmt_model_library": {
                        "path": str(native_root / "lib/libself_amdgpu_hsakmt_model.so.1")
                    },
                    "topology_manifest": {
                        "path": str(
                            native_root
                            / "share/self-amdgpu-runtime/hsakmt-topology/manifest.json"
                        )
                    },
                },
            }
            sdk_libraries = {
                "hip_library": {"path": str(prefix / "sdk-core/lib/libamdhip64.so.7")},
                "comgr_library": {"path": str(prefix / "sdk-core/lib/libamd_comgr.so.3")},
                "rccl_library": {"path": str(prefix / "sdk-libraries/lib/librccl.so.1")},
            }
            sdk = prefix / "sdk"
            state = Path(temporary) / "state"
            MODULE.write_activation(prefix, native, sdk, sdk_libraries, state)
            activate = (prefix / "etc/conda/activate.d/amdgpu-sim-rocm-pytorch.sh").read_text()
            deactivate = (prefix / "etc/conda/deactivate.d/amdgpu-sim-rocm-pytorch.sh").read_text()
            for name in (
                "LD_LIBRARY_PATH",
                "PKG_CONFIG_PATH",
                "ROCM_PATH",
                "HIP_PATH",
                "HSA_ENABLE_DTIF_FAST_COPY",
                "HSA_MODEL_LIB",
                "PATH",
            ):
                self.assertIn(f"_AMDGPU_SIM_ROCM_OLD_{name}", activate)
                self.assertIn(f"_AMDGPU_SIM_ROCM_OLD_{name}", deactivate)

    def test_lock_rejects_noncanonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = self.make_lock(root)
            path = root / MODULE.WHEEL_LOCK_RELATIVE
            path.write_text(json.dumps(document, indent=2), encoding="ascii")
            with self.assertRaisesRegex(MODULE.ProductError, "not a canonical object"):
                MODULE.wheel_lock(root)


if __name__ == "__main__":
    unittest.main()
