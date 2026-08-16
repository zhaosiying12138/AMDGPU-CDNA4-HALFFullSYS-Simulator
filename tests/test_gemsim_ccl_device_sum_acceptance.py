from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gemsim_ccl_device_sum_acceptance",
    ROOT / "tools/gemsim_ccl_device_sum_acceptance.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DeviceSumAcceptanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def minimal_result() -> dict:
        cases = []
        for dtype in ("bfloat16", "float32"):
            for count in MODULE.EXPECTED_COUNTS:
                tie = None
                if dtype == "bfloat16" and count >= 4:
                    tie = {
                        "correct": True,
                        "actual_bits": ["0x3f80", "0x3f82", "0xbf80", "0xbf82"],
                    }
                cases.append(
                    {
                        "dtype": dtype,
                        "element_count": count,
                        "extent": 0 if count == 0 else count + 4,
                        "program_count": 0 if count == 0 else 1,
                        "comparison": {"mismatch_count": 0, "finite": True},
                        "bf16_tie_contract": tie,
                        "right_unchanged": True,
                        "tail_unchanged": True,
                        "guards_unchanged": True,
                        "output_aliases_destination": True,
                        "output_correct": True,
                    }
                )
        return {
            "schema": "amdgpu-sim.ccl-device-sum.v1",
            "backend": "gemsim_amd",
            "arch": "gfx950",
            "output_correct": True,
            "claim_scope": "standalone_device_sum_primitive",
            "planner_binding_accepted": False,
            "trace_evidence_bound": False,
            "live_collective_accepted": False,
            "device_reduction_launch_count": 24,
            "self_reported_counters": {
                "acceptance_authority": False,
                "host_reduction_count": 0,
                "fallback_count": 0,
                "cpu_fallback_count": 0,
                "nvidia_fallback_count": 0,
            },
            "cases": cases,
            "negative_contracts": {"all": True},
        }

    def test_strict_result_rejects_self_acceptance_and_bit_drift(self) -> None:
        result = self.minimal_result()
        parsed = MODULE.strict_result(json.dumps(result, sort_keys=True) + "\n")
        self.assertEqual(parsed["device_reduction_launch_count"], 24)
        result["trace_evidence_bound"] = True
        with self.assertRaises(MODULE.AcceptanceError):
            MODULE.strict_result(json.dumps(result, sort_keys=True) + "\n")
        result["trace_evidence_bound"] = False
        for case in result["cases"]:
            if case["bf16_tie_contract"] is not None:
                case["bf16_tie_contract"]["actual_bits"][3] = "0xbf81"
                break
        with self.assertRaises(MODULE.AcceptanceError):
            MODULE.strict_result(json.dumps(result, sort_keys=True) + "\n")

    def test_rename_noreplace_preserves_existing_output(self) -> None:
        source = self.directory / "source"
        destination = self.directory / "destination"
        source.mkdir()
        destination.mkdir()
        (source / "value").write_bytes(b"new")
        marker = destination / "marker"
        marker.write_bytes(b"old")
        with self.assertRaises(FileExistsError):
            MODULE.rename_noreplace(source, destination)
        self.assertEqual(marker.read_bytes(), b"old")
        self.assertTrue(source.is_dir())

    def test_bind_kernel_images_rejects_missing_trace_image(self) -> None:
        cache = self.directory / "cache"
        entry = cache / "entry"
        entry.mkdir(parents=True)
        for suffix in ("hsaco", "ttir", "ttgir", "llir", "amdgcn"):
            (entry / f"_sum_kernel.{suffix}").write_bytes(suffix.encode("ascii"))
        known = MODULE.file_sha256(entry / "_sum_kernel.hsaco")
        bound = MODULE.bind_kernel_images(cache, {known: 1})
        self.assertTrue(bound["all_trace_images_bound"])
        with self.assertRaises(MODULE.AcceptanceError):
            MODULE.bind_kernel_images(cache, {"0" * 64: 1})


if __name__ == "__main__":
    unittest.main()
