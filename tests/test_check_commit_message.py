# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from check_commit_message import MessageError, check  # noqa: E402


VALID = """checkpoint subject

Checkpoint-ID: CP-0002
Goal-ID: GSIM-001
Plan-Revision: 1
Source-Lock-SHA256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Evidence-Manifest-SHA256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
Change-Kind: source
Baseline-Commit: N/A
"""


class CommitMessagePolicyTest(unittest.TestCase):
    def test_accepts_complete_progress_identity(self) -> None:
        check(VALID)

    def test_rejects_duplicate_checkpoint(self) -> None:
        with self.assertRaises(MessageError):
            check(VALID + "Checkpoint-ID: CP-9999\n")

    def test_rejects_abbreviated_baseline(self) -> None:
        with self.assertRaises(MessageError):
            check(VALID.replace("Baseline-Commit: N/A", "Baseline-Commit: 1234abcd"))

    def test_rejects_missing_evidence_manifest(self) -> None:
        with self.assertRaises(MessageError):
            check(
                VALID.replace(
                    "Evidence-Manifest-SHA256: " + "b" * 64 + "\n", ""
                )
            )

    def test_rejects_body_text_after_trailers_start(self) -> None:
        with self.assertRaises(MessageError):
            check(VALID.replace("Goal-ID: GSIM-001", "body after start\nGoal-ID: GSIM-001"))


if __name__ == "__main__":
    unittest.main()
