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
    def setUp(self) -> None:
        prerequisite_patch = mock.patch.object(
            transaction,
            "verify_begin_prerequisites",
            return_value={"checkpoint_id": "CP-0001"},
        )
        prerequisite_patch.start()
        self.addCleanup(prerequisite_patch.stop)
        durability_patch = mock.patch.object(
            transaction, "sync_repository_filesystems"
        )
        self.durability_barrier = durability_patch.start()
        self.addCleanup(durability_patch.stop)

    def test_transition_lock_persists_new_control_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            git_directory = Path(temp) / ".git"
            git_directory.mkdir()
            events: list[str] = []
            with mock.patch.object(
                transaction, "git_dir", return_value=git_directory
            ), mock.patch.object(
                transaction,
                "fsync_directory",
                side_effect=lambda path: events.append(f"fsync:{path.name}"),
            ):
                with transaction.transition_lock() as control:
                    self.assertEqual(control, git_directory / "amdgpu-sim")

            self.assertEqual(events, ["fsync:.git"])

    def test_git_helper_is_offline_and_read_only_by_default(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["git", "status"], returncode=0, stdout=b"", stderr=b""
        )
        with mock.patch.object(
            transaction.subprocess, "run", return_value=completed
        ) as run:
            transaction.git("status", "--porcelain=v1")

        environment = run.call_args.kwargs["env"]
        self.assertEqual(environment["GIT_NO_LAZY_FETCH"], "1")
        self.assertEqual(environment["GIT_OPTIONAL_LOCKS"], "0")
        self.assertEqual(environment["GIT_TERMINAL_PROMPT"], "0")

    def test_staged_gitlinks_must_equal_declared_participants(self) -> None:
        declared = {
            "previous_root": "a" * 40,
            "declared_children": {
                "declared": {"path": "projects/declared"},
            },
        }
        raw = (
            b":000000 160000 0000000 1111111 A\0projects/declared\0"
            b":000000 160000 0000000 2222222 A\0projects/undeclared\0"
        )
        with mock.patch.object(transaction, "git_bytes", return_value=raw):
            with self.assertRaisesRegex(
                transaction.TransactionError, "extra=.*projects/undeclared"
            ):
                transaction.verify_staged_participant_gitlinks(declared)

    def test_staged_participant_rejects_non_gitlink_project_change(self) -> None:
        declared = {
            "previous_root": "a" * 40,
            "declared_children": {
                "declared": {"path": "projects/declared"},
            },
        }
        raw = b":000000 100644 0000000 1111111 A\0projects/declared\0"
        with mock.patch.object(transaction, "git_bytes", return_value=raw):
            with self.assertRaisesRegex(
                transaction.TransactionError, "not a gitlink publication"
            ):
                transaction.verify_staged_participant_gitlinks(declared)

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
                ["fsync:amdgpu-sim", "replace", "fsync:committed", "fsync:txn"],
            )
            self.assertFalse(journal.exists())
            self.assertTrue((control / "committed" / journal.name).is_file())

    def test_atomic_json_persists_new_parent_before_publishing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            control = Path(temp) / "amdgpu-sim"
            control.mkdir()
            destination = control / "txn" / "CP-0002.json"
            events: list[str] = []
            real_replace = os.replace

            def replace(source: Path, target: Path) -> None:
                events.append("replace")
                real_replace(source, target)

            def fsync_directory(path: Path) -> None:
                events.append(f"fsync:{path.name}")

            with mock.patch.object(transaction, "fsync_directory", fsync_directory):
                with mock.patch.object(transaction.os, "replace", replace):
                    transaction.atomic_json(destination, {"schema": "fixture"})

            self.assertEqual(events, ["fsync:amdgpu-sim", "replace", "fsync:txn"])
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"schema": "fixture"},
            )

    def test_retire_journal_recovers_identical_active_and_retired_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            control = Path(temp) / "amdgpu-sim"
            active = control / "txn" / "CP-0002.json"
            retired = control / "committed" / active.name
            active.parent.mkdir(parents=True)
            retired.parent.mkdir(parents=True)
            payload = b'{"phase":"committed"}\n'
            active.write_bytes(payload)
            retired.write_bytes(payload)
            events: list[str] = []

            with mock.patch.object(
                transaction,
                "fsync_directory",
                side_effect=lambda path: events.append(f"fsync:{path.name}"),
            ):
                transaction.retire_journal(active, control)

            self.assertFalse(active.exists())
            self.assertEqual(retired.read_bytes(), payload)
            self.assertEqual(events, ["fsync:committed", "fsync:txn"])

    def test_retire_journal_rejects_conflicting_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            control = Path(temp) / "amdgpu-sim"
            active = control / "txn" / "CP-0002.json"
            retired = control / "committed" / active.name
            active.parent.mkdir(parents=True)
            retired.parent.mkdir(parents=True)
            active.write_text('{"phase":"committed"}\n', encoding="utf-8")
            retired.write_text('{"phase":"other"}\n', encoding="utf-8")

            with self.assertRaisesRegex(
                transaction.TransactionError, "already exists"
            ):
                transaction.retire_journal(active, control)

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
            self.assertIsNone(journal["declared_children"]["child"]["initial_head"])
            self.assertIsNone(journal["declared_children"]["child"]["initial_tree"])

    def test_begin_records_and_protects_existing_gitlink_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            child = root / "projects" / "child"
            child.mkdir(parents=True)
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=child,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            (child / "source.txt").write_text("baseline\n", encoding="utf-8")
            subprocess.run(["git", "add", "source.txt"], cwd=child, check=True)
            commit(child, "baseline")
            head = git(child, "rev-parse", "HEAD")
            tree = git(child, "rev-parse", "HEAD^{tree}")
            current = root / "state" / "current.json"
            current.parent.mkdir()
            current.write_text('{"checkpoint_id":"CP-0001"}\n', encoding="utf-8")
            subprocess.run(
                ["git", "add", "state/current.json", "projects/child"],
                cwd=root,
                check=True,
                stderr=subprocess.DEVNULL,
            )
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
                journal_path = (
                    root / ".git" / "amdgpu-sim" / "txn" / "CP-0002.json"
                )
                journal = json.loads(journal_path.read_text(encoding="utf-8"))
                declared = journal["declared_children"]["child"]
                self.assertEqual(declared["initial_head"], head)
                self.assertEqual(declared["initial_tree"], tree)

                declared["initial_head"] = "0" * 40
                journal_path.write_text(
                    json.dumps(journal, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    transaction.TransactionError, "initial identity mismatch"
                ):
                    transaction.command_declare_child(
                        argparse.Namespace(
                            checkpoint="CP-0002",
                            id="child",
                            path="projects/child",
                            head=head,
                            tree=tree,
                        )
                    )

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
                    self.durability_barrier.side_effect = transaction.TransactionError(
                        "simulated child durability failure"
                    )
                    with self.assertRaisesRegex(
                        transaction.TransactionError, "child durability failure"
                    ):
                        transaction.command_record_child(
                            argparse.Namespace(
                                checkpoint="CP-0002",
                                id="child",
                                path="projects/child",
                                head=head,
                                tree=tree,
                            )
                        )
                    journal_path = (
                        root / ".git" / "amdgpu-sim" / "txn" / "CP-0002.json"
                    )
                    self.assertEqual(
                        json.loads(journal_path.read_text(encoding="utf-8"))[
                            "expected_children"
                        ],
                        {},
                    )
                    self.durability_barrier.side_effect = None
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
                    self.durability_barrier.side_effect = transaction.TransactionError(
                        "simulated prepared-tree durability failure"
                    )
                    with self.assertRaisesRegex(
                        transaction.TransactionError, "prepared-tree durability failure"
                    ):
                        transaction.command_prepare_root(
                            argparse.Namespace(checkpoint="CP-0002")
                        )
                    self.assertEqual(
                        json.loads(journal_path.read_text(encoding="utf-8"))["phase"],
                        "prepare",
                    )
                    self.durability_barrier.side_effect = None
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
                    self.durability_barrier.side_effect = transaction.TransactionError(
                        "simulated coordinator durability failure"
                    )
                    with self.assertRaisesRegex(
                        transaction.TransactionError, "coordinator durability failure"
                    ):
                        transaction.command_finalize(
                            argparse.Namespace(
                                checkpoint="CP-0002",
                                committed_at="2026-08-07T00:01:00Z",
                            )
                        )
                    self.assertEqual(
                        json.loads(active_journal.read_text(encoding="utf-8"))["phase"],
                        "prepared",
                    )
                    self.durability_barrier.side_effect = None
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


class TransactionDurabilityPrimitiveTest(unittest.TestCase):
    def test_sync_repository_filesystems_deduplicates_devices(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            layouts = [
                (first, first, "refs/heads/main"),
                (second, second, "refs/heads/main"),
            ]
            with mock.patch.object(
                transaction, "repository_layout", side_effect=layouts
            ), mock.patch.object(transaction, "sync_filesystem") as sync:
                transaction.sync_repository_filesystems([first, second])

            sync.assert_called_once()

    def test_repository_layout_rejects_detached_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            (root / "file").write_text("content\n", encoding="utf-8")
            subprocess.run(["git", "add", "file"], cwd=root, check=True)
            commit(root, "initial")
            subprocess.run(
                ["git", "checkout", "--detach"],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with mock.patch.object(transaction, "ROOT", root):
                with self.assertRaisesRegex(
                    transaction.TransactionError, "detached or non-branch HEAD"
                ):
                    transaction.repository_layout(root, "fixture")


class TransactionBeginPrerequisiteTest(unittest.TestCase):
    def write_checkpoint(
        self,
        root: Path,
        *,
        checkpoint_id: str = "CP-0002",
        sequence: int = 2,
        status: str = "accepted",
        state: str = "ready",
    ) -> None:
        checkpoint_dir = root / "state" / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (root / "state" / "current.json").write_text(
            json.dumps(
                {
                    "checkpoint_id": checkpoint_id,
                    "state": state,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (checkpoint_dir / f"{checkpoint_id}.json").write_text(
            json.dumps(
                {
                    "id": checkpoint_id,
                    "sequence": sequence,
                    "status": status,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_begin_prerequisites_require_verified_next_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_checkpoint(root)
            completed = subprocess.CompletedProcess(
                args=["verify_workspace"], returncode=0, stdout="passed\n", stderr=""
            )
            with mock.patch.object(transaction, "ROOT", root), mock.patch.object(
                transaction.subprocess, "run", return_value=completed
            ) as run:
                current = transaction.verify_begin_prerequisites("CP-0003")

            self.assertEqual(current["checkpoint_id"], "CP-0002")
            argv = run.call_args.args[0]
            self.assertEqual(argv[-2:], ["--root", str(root)])
            environment = run.call_args.kwargs["env"]
            self.assertEqual(environment["GIT_NO_LAZY_FETCH"], "1")
            self.assertEqual(environment["GIT_OPTIONAL_LOCKS"], "0")

    def test_begin_prerequisites_reject_skipped_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_checkpoint(root)
            completed = subprocess.CompletedProcess(
                args=["verify_workspace"], returncode=0, stdout="passed\n", stderr=""
            )
            with mock.patch.object(transaction, "ROOT", root), mock.patch.object(
                transaction.subprocess, "run", return_value=completed
            ):
                with self.assertRaisesRegex(
                    transaction.TransactionError, "must be CP-0003"
                ):
                    transaction.verify_begin_prerequisites("CP-0004")

    def test_begin_prerequisites_reject_nonaccepted_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_checkpoint(root, status="partial", state="paused")
            completed = subprocess.CompletedProcess(
                args=["verify_workspace"], returncode=0, stdout="passed\n", stderr=""
            )
            with mock.patch.object(transaction, "ROOT", root), mock.patch.object(
                transaction.subprocess, "run", return_value=completed
            ):
                with self.assertRaisesRegex(
                    transaction.TransactionError, "not an accepted ready state"
                ):
                    transaction.verify_begin_prerequisites("CP-0003")

    def test_begin_prerequisites_propagate_workspace_failure(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["verify_workspace"],
            returncode=1,
            stdout="",
            stderr="resume verification failed: drift\n",
        )
        with mock.patch.object(transaction.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(
                transaction.TransactionError, "workspace verification failed.*drift"
            ):
                transaction.verify_begin_prerequisites("CP-0003")


if __name__ == "__main__":
    unittest.main()
