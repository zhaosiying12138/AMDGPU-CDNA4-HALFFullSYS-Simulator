from __future__ import annotations

import gzip
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rocm_deb_sysroot", ROOT / "tools/rocm_deb_sysroot.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def package_record(name: str, depends: str = "") -> str:
    return "\n".join(
        (
            f"Package: {name}",
            "Version: 7.2.3-test",
            "Architecture: amd64",
            f"Filename: pool/{name}_7.2.3-test_amd64.deb",
            f"Size: {len(name.encode())}",
            f"SHA256: {hashlib.sha256(name.encode()).hexdigest()}",
            f"Depends: {depends}",
        )
    )


class RocmDebSysrootTest(unittest.TestCase):
    def test_dependency_closure_replaces_only_rocr(self) -> None:
        payload = gzip.compress(
            (
                package_record("runtime", "math, hsa-rocr, libc6")
                + "\n\n"
                + package_record("math", "rocm-core")
                + "\n\n"
                + package_record("rocm-core")
                + "\n\n"
                + package_record("hsa-rocr")
                + "\n"
            ).encode()
        )
        records = MODULE.parse_packages_index(payload)
        selected, resolutions = MODULE.resolve_packages(
            records,
            ["runtime"],
            {"hsa-rocr": "native_product_rocr", "hsa-rocr-dev": "native_product_hsa_headers"},
            {"libc6"},
        )
        self.assertEqual(selected, ["math", "rocm-core", "runtime"])
        self.assertIn(
            {
                "package": "runtime",
                "alternatives": ["hsa-rocr"],
                "resolution": {
                    "kind": "replacement",
                    "name": "hsa-rocr",
                    "provider": "native_product_rocr",
                },
            },
            resolutions,
        )

    def test_unresolved_dependency_fails_closed(self) -> None:
        records = MODULE.parse_packages_index(
            gzip.compress((package_record("runtime", "unknown") + "\n").encode())
        )
        with self.assertRaisesRegex(MODULE.SysrootError, "unresolved dependency"):
            MODULE.resolve_packages(
                records,
                ["runtime"],
                {"hsa-rocr": "native", "hsa-rocr-dev": "native"},
                set(),
            )

    def test_merge_rejects_rocr_and_absolute_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            (source / "opt/rocm/lib").mkdir(parents=True)
            (source / "opt/rocm/lib/libhsa-runtime64.so.1").write_bytes(b"forbidden")
            destination.mkdir()
            with self.assertRaisesRegex(MODULE.SysrootError, "ROCr provider file"):
                MODULE._merge_package(source, destination, {}, "bad-rocr")

            (source / "opt/rocm/lib/libhsa-runtime64.so.1").unlink()
            (source / "opt/rocm/lib/libamdhip64.so.7").symlink_to("/host/libamdhip64.so.7")
            with self.assertRaisesRegex(MODULE.SysrootError, "absolute symlink"):
                MODULE._merge_package(source, destination, {}, "bad-link")

    def test_merge_allows_identical_collision_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "destination"
            destination.mkdir()
            first = root / "first"
            second = root / "second"
            for source in (first, second):
                (source / "opt/rocm/lib").mkdir(parents=True)
                (source / "opt/rocm/lib/libsame.so").write_bytes(b"same")
            owners: dict[str, list[str]] = {}
            MODULE._merge_package(first, destination, owners, "first")
            MODULE._merge_package(second, destination, owners, "second")
            self.assertEqual(owners["opt/rocm/lib/libsame.so"], ["first", "second"])
            (second / "opt/rocm/lib/libsame.so").write_bytes(b"different")
            with self.assertRaisesRegex(MODULE.SysrootError, "content collision"):
                MODULE._merge_package(second, destination, owners, "drift")

    def test_load_lock_rehashes_signed_index_and_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "artifacts/repository"
            cache = root / "artifacts/packages"
            repository.mkdir(parents=True)
            cache.mkdir(parents=True)
            packages_payload = gzip.compress((package_record("rocm-core") + "\n").encode())
            files = {
                "InRelease": b"inrelease",
                "Release": b"release",
                "Release.gpg": b"signature",
                "Packages.gz": packages_payload,
                "rocm.gpg.key": b"key",
                "rocm.gpg": b"keyring",
            }
            for name, payload in files.items():
                (repository / name).write_bytes(payload)
            package_path = cache / "rocm-core_7.2.3-test_amd64.deb"
            package_path.write_bytes(b"rocm-core")
            document = MODULE.generate_lock(
                root=root,
                repository_directory=repository,
                package_cache_directory=cache,
                repository_base_url="https://repo.radeon.com/rocm/apt/7.2.3/",
                suite="jammy",
                component="main",
                architecture="amd64",
                version="7.2.3",
                roots=["rocm-core"],
                replacements={
                    "hsa-rocr": "native_product_rocr",
                    "hsa-rocr-dev": "native_product_hsa_headers",
                },
                external_dependencies=[],
            )
            lock = root / "config/lock.json"
            lock.parent.mkdir()
            lock.write_bytes(MODULE.canonical_json(document))
            with (
                mock.patch.object(MODULE, "_verify_release"),
                mock.patch.object(
                    MODULE,
                    "_deb_identity",
                    return_value=("rocm-core", "7.2.3-test", "amd64"),
                ),
            ):
                loaded, payload = MODULE.load_lock(root, lock)
            self.assertEqual(loaded, document)
            self.assertEqual(payload, MODULE.canonical_json(document))
            package_path.write_bytes(b"bad")
            with (
                mock.patch.object(MODULE, "_verify_release"),
                self.assertRaisesRegex(MODULE.SysrootError, "cache differs"),
            ):
                MODULE.load_lock(root, lock)


if __name__ == "__main__":
    unittest.main()
