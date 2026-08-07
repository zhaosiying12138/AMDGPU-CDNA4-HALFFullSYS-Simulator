# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("transaction", ROOT / "scripts/transaction.py")
assert SPEC and SPEC.loader
transaction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transaction)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def commit(root: Path, message: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=amdgpu-sim test",
            "-c",
            "user.email=test@invalid",
            "commit",
            "-m",
            message,
        ],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )


class TransactionIntegrationTest(unittest.TestCase):
    def test_retire_journal_persists_new_directory_before_rename(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            control = Path(temp) / "amdgpu-sim"
            journal = control / "txn" / "CP-0002.json"
            journal.parent.mkdir(parents=True)
            journal.write_text("{}\n", encoding="utf-8")
            events: list[str] = []
            real_replace = os.replace

            def replace(source: Path, destination: Path) -> None:
                events.append("replace")
                real_replace(source, destination)

            def fsync_directory(path: Path) -> None:
                events.append(f"fsync:{path.name}")

            with mock.patch.object(transaction, "fsync_directory", fsync_directory):
                with mock.patch.object(transaction.os, "replace", replace):
                    transaction.retire_journal(journal, control)

            self.assertEqual(
                events,
                ["fsync:amdgpu-sim", "replace", "fsync:txn", "fsync:committed"],
            )
            self.assertFalse(journal.exists())
            self.assertTrue((control / "committed" / journal.name).is_file())

    def test_begin_rejects_empty_participant_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            (root / "state").mkdir()
            (root / "state" / "current.json").write_text(
                '{"checkpoint_id":"CP-0001"}\n', encoding="utf-8"
            )
            subprocess.run(["git", "add", "state/current.json"], cwd=root, check=True)
            commit(root, "bootstrap\n\nCheckpoint-ID: CP-0001")
            with mock.patch.object(transaction, "ROOT", root):
                with self.assertRaisesRegex(transaction.TransactionError, "predeclare"):
                    transaction.command_begin(
                        argparse.Namespace(
                            checkpoint="CP-0002",
                            intent="test",
                            started_at=None,
                            participant=[],
                        )
                    )

    def test_record_child_rejects_undeclared_participant(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-b", "main"], cwd=root, check=True, stdout=subprocess.DEVNULL)
            (root / "state").mkdir()
            (root / "state" / "current.json").write_text(
                '{"checkpoint_id":"CP-0001"}\n', encoding="utf-8"
            )
            subprocess.run(["git", "add", "state/current.json"], cwd=root, check=True)
            commit(root, "bootstrap\n\nCheckpoint-ID: CP-0001")
            with mock.patch.object(transaction, "ROOT", root):
                with redirect_stdout(io.StringIO()):
                    transaction.command_begin(
                        argparse.Namespace(
                            checkpoint="CP-0002",
                            intent="test",
                            started_at=None,
                            participant=["other=projects/other"],
                        )
                    )
                child = root / "projects" / "child"
                child.mkdir(parents=True)
                subprocess.run(
                    ["git", "init", "-b", "main"],
                    cwd=child,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                (child / "source").write_text("x\n", encoding="utf-8")
                subprocess.run(["git", "add", "source"], cwd=child, check=True)
                commit(child, "source")
                with self.assertRaisesRegex(transaction.TransactionError, "not predeclared"):
                    transaction.command_declare_child(
                        argparse.Namespace(
                            checkpoint="CP-0002",
                            id="child",
                            path="projects/child",
                            head=git(child, "rev-parse", "HEAD"),
                            tree=git(child, "rev-parse", "HEAD^{tree}"),
                        )
                    )
                with self.assertRaisesRegex(transaction.TransactionError, "not declared"):
                    transaction.command_record_child(
                        argparse.Namespace(
                            checkpoint="CP-0002",
                            id="child",
                            path="projects/child",
                            head=git(child, "rev-parse", "HEAD"),
                            tree=git(child, "rev-parse", "HEAD^{tree}"),
                        )
                    )

    def test_begin_generates_timestamp_when_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            current = root / "state" / "current.json"
            current.parent.mkdir(parents=True)
            current.write_text('{"checkpoint_id":"CP-0001"}\n', encoding="utf-8")
            subprocess.run(["git", "add", "state/current.json"], cwd=root, check=True)
            commit(root, "bootstrap\n\nCheckpoint-ID: CP-0001")
            with mock.patch.object(transaction, "ROOT", root):
                with redirect_stdout(io.StringIO()):
                    transaction.command_begin(
                        argparse.Namespace(
                            checkpoint="CP-0002",
                            intent="test",
                            started_at=None,
                            participant=["child=projects/child"],
                        )
                    )
                journal = json.loads(
                    (
                        root
                        / ".git"
                        / "amdgpu-sim"
                        / "txn"
                        / "CP-0002.json"
                    ).read_text(encoding="utf-8")
                )
            self.assertRegex(journal["started_at"], r"^20[0-9]{2}-")

    def test_begin_rejects_dirty_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            (root / "state").mkdir()
            (root / "state" / "current.json").write_text(
                '{"checkpoint_id":"CP-0001"}\n', encoding="utf-8"
            )
            subprocess.run(["git", "add", "state/current.json"], cwd=root, check=True)
            commit(root, "bootstrap\n\nCheckpoint-ID: CP-0001")
            (root / "untracked").write_text("dirty\n", encoding="utf-8")
            with mock.patch.object(transaction, "ROOT", root):
                with self.assertRaises(transaction.TransactionError):
                    transaction.command_begin(
                        argparse.Namespace(
                            checkpoint="CP-0002",
                            intent="test",
                            started_at="2026-08-07T00:00:00Z",
                            participant=["child=projects/child"],
                        )
                    )

    def test_prepare_commit_finalize_records_child_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-b", "main"], cwd=root, check=True, stdout=subprocess.DEVNULL)
            (root / "state").mkdir()
            current = root / "state" / "current.json"
            current.write_text('{"checkpoint_id":"CP-0001"}\n', encoding="utf-8")
            subprocess.run(["git", "add", "state/current.json"], cwd=root, check=True)
            commit(root, "bootstrap\n\nCheckpoint-ID: CP-0001")

            with mock.patch.object(transaction, "ROOT", root):
                with redirect_stdout(io.StringIO()):
                    transaction.command_begin(
                        argparse.Namespace(
                            checkpoint="CP-0002",
                            intent="test",
                            started_at="2026-08-07T00:00:00Z",
                            participant=["child=projects/child"],
                        )
                    )
                child = root / "projects" / "child"
                child.mkdir(parents=True)
                subprocess.run(["git", "init", "-b", "main"], cwd=child, check=True, stdout=subprocess.DEVNULL)
                (child / "source.txt").write_text("baseline\n", encoding="utf-8")
                subprocess.run(["git", "add", "source.txt"], cwd=child, check=True)
                commit(child, "baseline")
                head = git(child, "rev-parse", "HEAD")
                tree = git(child, "rev-parse", "HEAD^{tree}")
                with redirect_stdout(io.StringIO()):
                    transaction.command_declare_child(
                        argparse.Namespace(
                            checkpoint="CP-0002",
                            id="child",
                            path="projects/child",
                            head=head,
                            tree=tree,
                        )
                    )
                    transaction.command_record_child(
                        argparse.Namespace(
                            checkpoint="CP-0002",
                            id="child",
                            path="projects/child",
                            head=head,
                            tree=tree,
                        )
                    )
                current.write_text('{"checkpoint_id":"CP-0002"}\n', encoding="utf-8")
                subprocess.run(
                    ["git", "add", "state/current.json", "projects/child"],
                    cwd=root,
                    check=True,
                    stderr=subprocess.DEVNULL,
                )
                with redirect_stdout(io.StringIO()):
                    transaction.command_declare_root(
                        argparse.Namespace(
                            checkpoint="CP-0002",
                            path=["projects/child", "state/current.json"],
                        )
                    )
                    transaction.command_prepare_root(argparse.Namespace(checkpoint="CP-0002"))
                commit(
                    root,
                    "coordinator\n\n"
                    "Checkpoint-ID: CP-0002\n"
                    "Goal-ID: GSIM-001\n"
                    "Plan-Revision: 1\n"
                    f"Source-Lock-SHA256: {'a' * 64}\n"
                    f"Evidence-Manifest-SHA256: {'b' * 64}\n"
                    "Change-Kind: source\n"
                    "Baseline-Commit: N/A",
                )
                with mock.patch.object(
                    transaction,
                    "verify_post_commit",
                    side_effect=transaction.TransactionError("verification failed"),
                ):
                    with self.assertRaisesRegex(transaction.TransactionError, "verification failed"):
                        transaction.command_finalize(
                            argparse.Namespace(
                                checkpoint="CP-0002", committed_at="2026-08-07T00:01:00Z"
                            )
                        )
                active_journal = (
                    Path(git(root, "rev-parse", "--absolute-git-dir"))
                    / "amdgpu-sim"
                    / "txn"
                    / "CP-0002.json"
                )
                self.assertTrue(active_journal.is_file())
                self.assertEqual(json.loads(active_journal.read_text())["phase"], "prepared")
                with mock.patch.object(transaction, "verify_post_commit"):
                    with mock.patch.object(
                        transaction,
                        "retire_journal",
                        side_effect=OSError("simulated crash before journal retirement"),
                    ):
                        with self.assertRaisesRegex(OSError, "simulated crash"):
                            transaction.command_finalize(
                                argparse.Namespace(
                                    checkpoint="CP-0002",
                                    committed_at="2026-08-07T00:01:00Z",
                                )
                            )
                    committed_journal = json.loads(
                        active_journal.read_text(encoding="utf-8")
                    )
                    self.assertEqual(committed_journal["phase"], "committed")
                    self.assertEqual(
                        committed_journal["root_coordinator_commit"],
                        git(root, "rev-parse", "HEAD"),
                    )
                    with redirect_stdout(io.StringIO()):
                        transaction.command_finalize(
                            argparse.Namespace(
                                checkpoint="CP-0002",
                                committed_at="ignored-during-recovery",
                            )
                        )

            control = Path(git(root, "rev-parse", "--absolute-git-dir")) / "amdgpu-sim"
            self.assertFalse((control / "txn" / "CP-0002.json").exists())
            retired = json.loads(
                (control / "committed" / "CP-0002.json").read_text(encoding="utf-8")
            )
            self.assertEqual(retired["phase"], "committed")
            self.assertEqual(retired["root_coordinator_commit"], git(root, "rev-parse", "HEAD"))


if __name__ == "__main__":
    unittest.main()
