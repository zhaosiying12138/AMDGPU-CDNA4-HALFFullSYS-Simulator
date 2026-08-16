# SPDX-License-Identifier: GPL-3.0-or-later
"""Host-only contracts for deterministic AgentENV runtime bundle manifests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "agentenv_bundle", ROOT / "tools" / "agentenv_bundle.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AgentEnvBundleTest(unittest.TestCase):
    def builder(self, root: Path, *, guest_root: str = "/guest/work") -> object:
        return MODULE.BundleBuilder(source_roots=[root], guest_root=guest_root)

    def test_manifest_is_deterministic_and_hashes_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "z-last.txt").write_text("last\n", encoding="utf-8")
            (root / "a-first.txt").write_text("first\n", encoding="utf-8")
            builder = self.builder(root)
            builder.add(root, label="fixture")

            first = builder.manifest(hash_files=True)
            # A fresh builder exercises traversal ordering rather than relying
            # on the first builder's already-populated entry map.
            second_builder = self.builder(root)
            second_builder.add(root, label="fixture")
            second = second_builder.manifest(hash_files=True)

        self.assertEqual(MODULE.canonical_json(first), MODULE.canonical_json(second))
        self.assertEqual(first["entry_count"], 3)
        self.assertEqual(
            [entry["path"] for entry in first["entries"]],
            ["/guest/work", "/guest/work/a-first.txt", "/guest/work/z-last.txt"],
        )
        files = [entry for entry in first["entries"] if entry["kind"] == "file"]
        self.assertTrue(all(len(entry["sha256"]) == 64 for entry in files))

    def test_symlink_is_retained_and_in_root_target_is_included(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "lib" / "libfixture.so"
            target.parent.mkdir()
            target.write_bytes(b"fixture")
            link = root / "lib" / "libfixture.so.1"
            link.symlink_to("libfixture.so")
            builder = self.builder(root)
            builder.add(root / "lib", label="runtime")
            manifest = builder.manifest(hash_files=True)

        by_path = {entry["path"]: entry for entry in manifest["entries"]}
        self.assertEqual(by_path["/guest/work/lib/libfixture.so.1"]["kind"], "symlink")
        self.assertEqual(by_path["/guest/work/lib/libfixture.so.1"]["target"], "libfixture.so")
        self.assertEqual(by_path["/guest/work/lib/libfixture.so"]["kind"], "file")

    def test_manifest_preserves_file_and_directory_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            runtime.mkdir()
            executable = runtime / "launch.sh"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o751)
            runtime.chmod(0o711)
            builder = self.builder(root)
            builder.add(runtime, label="runtime")
            manifest = builder.manifest(hash_files=False)

        by_path = {entry["path"]: entry for entry in manifest["entries"]}
        self.assertEqual(by_path["/guest/work/runtime"]["mode"], 0o711)
        self.assertEqual(by_path["/guest/work/runtime/launch.sh"]["mode"], 0o751)

    def test_external_symlink_is_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            external = Path(outside) / "outside.so"
            external.write_bytes(b"outside")
            link = root / "escape.so"
            link.symlink_to(external)
            builder = self.builder(root)
            with self.assertRaisesRegex(MODULE.BundleError, "allowed roots"):
                builder.add(link, label="unsafe")

    def test_guest_root_mapping_keeps_absolute_runtime_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tool = root / "bin" / "gem5.opt"
            tool.parent.mkdir()
            tool.write_bytes(b"gem5")
            builder = self.builder(root, guest_root="/home/zhaosiying/amdgpu-sim")
            builder.add(root / "bin", label="gem5")
            manifest = builder.manifest(hash_files=False)

        paths = {entry["path"] for entry in manifest["entries"]}
        self.assertIn("/home/zhaosiying/amdgpu-sim/bin", paths)
        self.assertIn("/home/zhaosiying/amdgpu-sim/bin/gem5.opt", paths)
        self.assertTrue(all(path.startswith("/home/zhaosiying/amdgpu-sim/") for path in paths))


if __name__ == "__main__":
    unittest.main()
