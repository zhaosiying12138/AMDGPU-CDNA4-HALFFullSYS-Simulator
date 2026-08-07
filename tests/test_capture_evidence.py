# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "capture_evidence", ROOT / "scripts/capture_evidence.py"
)
assert SPEC and SPEC.loader
capture_evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture_evidence)


class CaptureEvidenceTest(unittest.TestCase):
    def test_captures_exact_stream_hashes_and_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "artifacts").mkdir()
            with mock.patch.object(capture_evidence, "ROOT", root):
                result = capture_evidence.capture(
                    "artifacts/evidence/fixture.json",
                    ".",
                    [
                        sys.executable,
                        "-c",
                        "import sys; print('out'); print('err', file=sys.stderr); raise SystemExit(7)",
                    ],
                )
            self.assertEqual(result["exit_code"], 7)
            record = json.loads(
                (root / "artifacts/evidence/fixture.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["stdout"]["size"], 4)
            self.assertEqual(record["stderr"]["size"], 4)

    def test_rejects_record_outside_artifacts(self) -> None:
        with self.assertRaises(capture_evidence.CaptureError):
            capture_evidence.capture("state/evidence/bad.json", ".", ["true"])


if __name__ == "__main__":
    unittest.main()
