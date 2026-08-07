# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("check_staged", ROOT / "scripts/check_staged.py")
assert SPEC and SPEC.loader
check_staged = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_staged)


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.DEVNULL)


class StagedPolicyTest(unittest.TestCase):
    def test_child_rejects_generated_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git(root, "init", "-b", "main")
            path = root / "models" / "opaque.data"
            path.parent.mkdir(parents=True)
            path.write_text("generated\n", encoding="utf-8")
            git(root, "add", "models/opaque.data")
            with self.assertRaises(check_staged.PolicyError):
                check_staged.check(root)

    def test_child_can_stage_source_under_projects(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git(root, "init", "-b", "main")
            path = root / "projects" / "runtime" / "source.cpp"
            path.parent.mkdir(parents=True)
            path.write_text("int value;\n", encoding="utf-8")
            git(root, "add", "projects/runtime/source.cpp")
            check_staged.check(root)

    def test_coordinator_rejects_non_gitlink_project_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git(root, "init", "-b", "main")
            (root / "state").mkdir()
            (root / "SOURCE_LOCK.json").write_text("{}\n", encoding="utf-8")
            (root / "state" / "current.json").write_text("{}\n", encoding="utf-8")
            (root / "projects").mkdir()
            (root / "projects" / "bad.txt").write_text("bad\n", encoding="utf-8")
            git(root, "add", "projects/bad.txt")
            with self.assertRaises(check_staged.PolicyError):
                check_staged.check(root)

    def test_probable_secret_is_rejected_from_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git(root, "init", "-b", "main")
            path = root / "config.txt"
            probable_secret = "token=" + "hf" + "_" + "abcdefghijklmnopqrstuvwxyz\n"
            path.write_text(probable_secret, encoding="utf-8")
            git(root, "add", "config.txt")
            path.write_text("token=redacted\n", encoding="utf-8")
            with self.assertRaises(check_staged.PolicyError):
                check_staged.check(root)


if __name__ == "__main__":
    unittest.main()
