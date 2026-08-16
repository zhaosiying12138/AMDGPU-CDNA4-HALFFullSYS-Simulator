"""Fail-closed ownership inventory for the runtime-gem5 extraction surface."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "tools/runtime_gem5_bridge_inventory.json"
RUNTIME_FILES = {
    "protocol/host-transport-v1.json",
    "protocol/host-transport-v1-kmt.json",
    "protocol/host-transport-v2.json",
    "tools/generate_bridge_wire.py",
    "projects/self-amdgpu-runtime/include/self_amdgpu_runtime/runtime.h",
    "projects/self-amdgpu-runtime/include/self_amdgpu_runtime/generated/bridge_generic_v2.h",
    "projects/self-amdgpu-runtime/include/self_amdgpu_runtime/generated/bridge_kmt_v5.h",
    "projects/self-amdgpu-runtime/include/self_amdgpu_runtime/provider.h",
    "projects/self-amdgpu-runtime/include/self_amdgpu_runtime/kmt_shim.h",
    "projects/self-amdgpu-runtime/src/managed_session.c",
    "projects/self-amdgpu-runtime/src/transport.c",
    "projects/self-amdgpu-runtime/src/transport_codec.c",
    "projects/self-amdgpu-runtime/src/transport_internal.h",
    "projects/gem5/src/dev/amdgpu/generated/bridge_generic_v2.hh",
    "projects/gem5/src/dev/amdgpu/generated/bridge_kmt_v5.hh",
}
FACADE_FILES = {
    "plugins/triton/gemsim_amd/backend/compiler.py",
    "plugins/triton/gemsim_amd/backend/driver.py",
    "projects/gem5/configs/example/gemsim/host_dispatch.py",
}


def discovered_files() -> set[str]:
    result = set(RUNTIME_FILES | FACADE_FILES)
    directory = ROOT / "projects/gem5/src/dev/amdgpu"
    for path in directory.iterdir():
        if path.is_file() and path.name.startswith(("host_gpu_", "host_native_")):
            result.add(path.relative_to(ROOT).as_posix())
    return result


class RuntimeGem5BridgeInventoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads(INVENTORY.read_text(encoding="ascii"))
        cls.entries = cls.document["entries"]

    def test_inventory_is_complete_unique_and_canonical(self) -> None:
        self.assertEqual(
            self.document["schema"], "amdgpu-sim.runtime-gem5-bridge-inventory.v1"
        )
        paths = [entry["path"] for entry in self.entries]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(set(paths), discovered_files())
        self.assertTrue(all((ROOT / path).is_file() for path in paths))

    def test_every_entry_has_one_explicit_migration_disposition(self) -> None:
        allowed = {"keep", "migrate", "compatibility", "evidence-only", "delete-candidate"}
        for entry in self.entries:
            self.assertEqual(set(entry), {"path", "classification", "owner", "target"})
            self.assertIn(entry["classification"], allowed)
            self.assertTrue(entry["owner"])
            self.assertTrue(entry["target"])

    def test_framework_facade_never_owns_bridge_or_gem5_details(self) -> None:
        by_path = {entry["path"]: entry for entry in self.entries}
        self.assertEqual(by_path["plugins/triton/gemsim_amd/backend/compiler.py"]["owner"], "triton-oot-facade")
        self.assertEqual(by_path["plugins/triton/gemsim_amd/backend/driver.py"]["target"], "runtime-public-api")
        self.assertNotIn("gem5", by_path["plugins/triton/gemsim_amd/backend/driver.py"]["target"])

    def test_fixture_routes_are_not_future_production_interfaces(self) -> None:
        for entry in self.entries:
            if "fixture" in Path(entry["path"]).name:
                self.assertIn(entry["classification"], {"evidence-only", "delete-candidate"})


if __name__ == "__main__":
    unittest.main()
