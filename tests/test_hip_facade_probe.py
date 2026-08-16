from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hip_facade_probe", ROOT / "tools/hip_facade_probe.py")
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class HipFacadeProbeTest(unittest.TestCase):
    def test_current_native_product_publishes_complete_standard_facade(self) -> None:
        active = json.loads((ROOT / "env/rocm/active-product").read_text(encoding="ascii"))
        prefix = Path(active["prefix"])
        result = PROBE.probe(prefix)
        self.assertTrue(result["correct"])
        self.assertTrue(result["manifest_present"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["product_id"], active["product_id"])
        self.assertEqual(set(result["libraries"]), set(PROBE.LIBRARY_CONTRACT))
        for record in result["libraries"].values():
            self.assertEqual(record["status"], "present")
            self.assertEqual(record["rejected_candidates"], [])
            library = Path(record["path"])
            library.relative_to(prefix)
            self.assertTrue(library.is_file())
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
        self.assertFalse(result["policy"]["system_loader_fallback"])
        self.assertFalse(result["policy"]["cpu_fallback"])
        self.assertFalse(result["policy"]["privateuse1_fallback"])

    def test_manifest_is_required_unless_explicitly_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            result = PROBE.probe(prefix)
            self.assertFalse(result["correct"])
            self.assertIn("product manifest is missing", result["errors"])
            unmanaged = PROBE.probe(prefix, require_manifest=False)
            self.assertNotIn("product manifest is missing", unmanaged["errors"])
            self.assertFalse(unmanaged["correct"])

    def test_out_of_prefix_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "product"
            (prefix / "lib").mkdir(parents=True)
            outside = Path(directory) / "outside.so"
            outside.write_bytes(b"not a library")
            (prefix / "lib/libamdhip64.so").symlink_to(outside)
            manifest = {"schema": "amdgpu-sim.product-prefix.v1", "product_id": "test"}
            (prefix / "manifest.json").write_bytes(PROBE.canonical_json(manifest))
            result = PROBE.probe(prefix)
            self.assertFalse(result["correct"])
            self.assertIn(
                {"path": str(prefix / "lib/libamdhip64.so"), "reason": "symlink_escapes_product"},
                result["libraries"]["hip"]["rejected_candidates"],
            )

    def test_probe_result_is_canonical_and_declares_no_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / "lib").mkdir()
            (prefix / "manifest.json").write_bytes(
                PROBE.canonical_json({"schema": "amdgpu-sim.product-prefix.v1", "product_id": "x"})
            )
            result = PROBE.probe(prefix)
            payload = PROBE.canonical_json(result)
            self.assertEqual(json.loads(payload.decode("ascii")), result)
            self.assertEqual(result["policy"]["cpu_fallback"], False)
            self.assertEqual(result["policy"]["privateuse1_fallback"], False)


if __name__ == "__main__":
    unittest.main()
