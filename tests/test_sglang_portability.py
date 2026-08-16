from __future__ import annotations

import importlib.util
from pathlib import Path
import json
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sglang_portability_audit", ROOT / "tools/sglang_portability_audit.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SGLangPortabilityAuditTest(unittest.TestCase):
    def _fixture(self, *, hook: bool = False) -> Path:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "sglang/srt/distributed").mkdir(parents=True)
        (root / "sglang/srt/layers/attention").mkdir(parents=True)
        (root / "PKG-INFO").write_text("Name: sglang\nVersion: 0.5.10.post1\n", encoding="ascii")
        (root / "pyproject.toml").write_text("[project]\nname='sglang'\n", encoding="ascii")
        (root / "sglang/_version.py").write_text("__version__ = version = '0.5.10.post1'\n", encoding="ascii")
        (root / "sglang/srt/distributed/parallel_state.py").write_text(
            "class GroupCoordinator:\n    def all_reduce(self, x):\n        return torch.distributed.all_reduce(x)\n",
            encoding="ascii",
        )
        (root / "sglang/srt/layers/attention/attention_registry.py").write_text(
            "BACKENDS = {'aiter': object, 'triton': object}\n", encoding="ascii"
        )
        (root / "sglang/srt/server_args.py").write_text("attention_backend = 'triton'\n", encoding="ascii")
        if hook:
            (root / "sglang/srt/platforms").mkdir()
            (root / "sglang/srt/platforms/__init__.py").write_text(
                "from importlib.metadata import entry_points\n"
                "PLATFORM_PLUGINS_GROUP = 'sglang.srt.platforms'\n"
                "def _load_platform_class(value): return value\n",
                encoding="ascii",
            )
        return root

    def test_reports_missing_srt_hook_without_importing_upstream(self) -> None:
        result = MODULE.audit(self._fixture())
        self.assertEqual(result["upstream"]["version"], "0.5.10.post1")
        self.assertFalse(result["srt_extension_points"]["official_oot_hook"])
        self.assertEqual(result["extension_gap"]["code"], "SRT_NO_PUBLIC_OOT_DEVICE_COMMUNICATOR")
        self.assertTrue(result["attention"]["aiter_registered"])
        self.assertTrue(result["attention"]["triton_registered"])

    def test_detects_official_platform_directory(self) -> None:
        result = MODULE.audit(self._fixture(hook=True))
        self.assertTrue(result["srt_extension_points"]["official_oot_hook"])
        self.assertEqual(result["extension_gap"]["code"], None)

    def test_expected_identity_rejects_drift(self) -> None:
        root = self._fixture()
        expected = MODULE.audit(root)
        (root / "sglang/srt/server_args.py").write_text("changed\n", encoding="ascii")
        with self.assertRaises(MODULE.AuditError):
            MODULE.audit(root, expected=expected)

    def test_real_snapshot_is_upstream_unchanged_and_no_flashinfer_policy(self) -> None:
        root = ROOT / "projects/sglang-0.5.10.post1"
        if not root.is_dir():
            self.skipTest("optional upstream SGLang snapshot is not prepared")
        result = MODULE.audit(root)
        self.assertEqual(result["upstream"]["version"], "0.5.10.post1")
        self.assertTrue(result["attention"]["aiter_registered"])
        self.assertTrue(result["attention"]["triton_registered"])
        self.assertEqual(result["attention"]["policy"], "aiter_or_triton_only")
        self.assertFalse(result["srt_extension_points"]["official_oot_hook"])

    def test_latest_wheel_snapshot_exposes_official_srt_platform_contract(self) -> None:
        root = ROOT / "projects/sglang-0.5.17"
        if not root.is_dir():
            self.skipTest("optional upstream SGLang wheel snapshot is not prepared")
        result = MODULE.audit(root)
        self.assertEqual(result["upstream"]["version"], "0.5.17")
        self.assertTrue(result["srt_extension_points"]["platform_directory"])
        self.assertTrue(result["srt_extension_points"]["platform_discovery_contract"])
        self.assertEqual(
            result["srt_extension_points"]["entry_point_group"],
            "sglang.srt.platforms",
        )
        self.assertTrue(result["srt_extension_points"]["official_oot_hook"])
        self.assertIsNone(result["extension_gap"]["code"])
        self.assertTrue(
            result["srt_extension_points"]["upstream_rocm_auto_detection"]
        )
        self.assertEqual(
            result["srt_extension_points"]["formal_model_platform"],
            "in_tree_RocmSRTPlatform",
        )
        self.assertEqual(
            result["srt_extension_points"]["oot_platform_role"],
            "diagnostic_only",
        )
        manifest = json.loads((ROOT / "tools/sglang_source_manifest.json").read_text())
        MODULE.audit(root, expected=manifest)


if __name__ == "__main__":
    unittest.main()
