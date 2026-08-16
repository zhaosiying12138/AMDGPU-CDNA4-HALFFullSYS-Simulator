#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/fastcopy_mode.sh"


class FastCopyModeTest(unittest.TestCase):
    def run_mode(self, mode: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1" "$2" && printf "%s %s\\n" '
                '"$HSA_ENABLE_DTIF_FAST_COPY" "$SAGR_HSAKMT_MODEL_FAST_COPY"',
                "bash",
                str(SCRIPT),
                mode,
            ],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_fast_sets_both_gates(self) -> None:
        completed = self.run_mode("fast")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "1 1\n")

    def test_legacy_clears_both_gates(self) -> None:
        completed = self.run_mode("legacy")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "0 0\n")

    def test_unknown_mode_fails(self) -> None:
        completed = self.run_mode("unknown")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("fast|legacy", completed.stderr)


if __name__ == "__main__":
    unittest.main()

