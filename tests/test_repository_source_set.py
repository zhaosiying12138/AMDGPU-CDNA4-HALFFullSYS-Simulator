from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "repository_source_set", ROOT / "tools/repository_source_set.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RepositorySourceSetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repository = Path(self.temporary.name)
        subprocess.run(["git", "init", "-q", str(self.repository)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repository), "config", "user.name", "test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repository), "config", "user.email", "test@example.invalid"],
            check=True,
        )
        (self.repository / ".gitignore").write_text("ignored\n", encoding="ascii")
        (self.repository / "tracked").write_bytes(b"tracked\n")
        executable = self.repository / "executable"
        executable.write_bytes(b"#!/bin/sh\n")
        executable.chmod(0o755)
        (self.repository / "link").symlink_to("tracked")
        subprocess.run(["git", "-C", str(self.repository), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(self.repository), "commit", "-qm", "fixture"], check=True
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def identity(self):
        return MODULE.source_set(self.repository)

    def test_deterministic_and_covers_actual_worktree(self) -> None:
        (self.repository / "untracked").write_bytes(b"untracked\n")
        (self.repository / "ignored").write_bytes(b"ignored\n")
        first = self.identity()
        second = self.identity()
        self.assertEqual(first, second)
        records = {record["path"]: record for record in first["files"]}
        self.assertIn("untracked", records)
        self.assertNotIn("ignored", records)
        self.assertTrue(records["executable"]["executable"])
        self.assertEqual(records["link"]["kind"], "symlink")

    def test_content_mode_symlink_and_deletion_drift(self) -> None:
        baseline = self.identity()["source_set_sha256"]
        (self.repository / "tracked").write_bytes(b"changed\n")
        changed = self.identity()["source_set_sha256"]
        self.assertNotEqual(baseline, changed)
        executable = self.repository / "executable"
        executable.chmod(0o644)
        mode_changed = self.identity()["source_set_sha256"]
        self.assertNotEqual(changed, mode_changed)
        (self.repository / "link").unlink()
        (self.repository / "link").symlink_to("executable")
        link_changed = self.identity()["source_set_sha256"]
        self.assertNotEqual(mode_changed, link_changed)
        (self.repository / "tracked").unlink()
        deleted = self.identity()
        self.assertNotEqual(link_changed, deleted["source_set_sha256"])
        records = {record["path"]: record for record in deleted["files"]}
        self.assertEqual(records["tracked"]["kind"], "missing")

    def test_gitlinks_require_explicit_mode_and_bind_commit(self) -> None:
        nested = self.repository.parent / "nested"
        nested.mkdir()
        subprocess.run(["git", "-C", str(nested), "init", "-q"], check=True)
        subprocess.run(
            ["git", "-C", str(nested), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(nested), "config", "user.name", "Test"],
            check=True,
        )
        (nested / "source").write_bytes(b"nested\n")
        subprocess.run(["git", "-C", str(nested), "add", "source"], check=True)
        subprocess.run(
            ["git", "-C", str(nested), "commit", "-qm", "nested"], check=True
        )
        commit = subprocess.check_output(
            ["git", "-C", str(nested), "rev-parse", "HEAD"], text=True
        ).strip()
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repository),
                "update-index",
                "--add",
                "--cacheinfo",
                f"160000,{commit},gitlink",
            ],
            check=True,
        )
        (self.repository / "gitlink").mkdir()
        with self.assertRaisesRegex(MODULE.SourceSetError, "gitlink is forbidden"):
            MODULE.source_set(self.repository)
        records = {
            record["path"]: record
            for record in MODULE.source_set(
                self.repository, allow_gitlinks=True
            )["files"]
        }
        self.assertEqual(
            records["gitlink"],
            {
                "path": "gitlink",
                "kind": "gitlink",
                "commit": commit,
                "checkout": "absent",
                "index_mode": "160000",
            },
        )


if __name__ == "__main__":
    unittest.main()
