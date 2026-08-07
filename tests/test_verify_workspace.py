# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "verify_workspace", ROOT / "scripts/verify_workspace.py"
)
assert SPEC and SPEC.loader
verify_workspace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_workspace)


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.DEVNULL)


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def absorbed_child_fixture(root: Path, *, absorb: bool) -> dict[str, object]:
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    child = root / "projects" / "example"
    child.mkdir(parents=True)
    git(child, "init", "-b", "amdgpu-sim/example")
    git(child, "config", "user.name", "test")
    git(child, "config", "user.email", "test@example.invalid")
    (child / "source.txt").write_text("baseline\n", encoding="utf-8")
    (child / "LICENSE").write_text("fixture license\n", encoding="utf-8")
    git(child, "add", "source.txt", "LICENSE")
    git(child, "commit", "-m", "baseline")

    canonical_url = "https://example.invalid/example.git"
    transport_url = "https://transport.invalid/example.git"
    commit = git_output(child, "rev-parse", "HEAD")
    tree = git_output(child, "rev-parse", "HEAD^{tree}")
    tag = f"upstream-baseline/example/{commit}"
    tag_message = (
        "Frozen fixture baseline\n\n"
        f"Upstream-URL: {canonical_url}\n"
        "Upstream-Ref: main\n"
        f"Baseline-Commit: {commit}\n"
        f"Baseline-Tree: {tree}"
    )
    git(child, "tag", "-a", tag, "-m", tag_message)
    tag_object = git_output(child, "rev-parse", tag)
    tag_payload = subprocess.check_output(
        ["git", "cat-file", "tag", tag], cwd=child, text=True
    )
    git(child, "remote", "add", "upstream", transport_url)
    git(child, "remote", "set-url", "--push", "upstream", "no_push")
    git(child, "config", "core.hooksPath", "../../.githooks")

    (root / ".gitmodules").write_text(
        "[submodule \"example\"]\n"
        "\tpath = projects/example\n"
        f"\turl = {canonical_url}\n"
        "\tbranch = main\n",
        encoding="utf-8",
    )
    git(root, "add", ".gitmodules")
    subprocess.run(
        ["git", "add", "projects/example"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if absorb:
        subprocess.run(
            ["git", "submodule", "absorbgitdirs", "projects/example"],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    return {
        "id": "example",
        "path": "projects/example",
        "upstream_url": canonical_url,
        "transport_url": transport_url,
        "upstream_ref": "main",
        "work_head": commit,
        "work_tree": tree,
        "baseline_commit": commit,
        "baseline_tree": tree,
        "baseline_tag": tag,
        "baseline_tag_object": tag_object,
        "baseline_tag_payload": tag_payload,
        "baseline_tag_payload_sha256": hashlib.sha256(tag_payload.encode()).hexdigest(),
        "expected_remotes": {"upstream": transport_url},
        "expected_push_urls": {"upstream": "no_push"},
        "administrative_git_dir": "modules/example",
        "worktree_verification": {
            "full_checkout": True,
            "sparse_checkout": False,
            "skip_worktree_entries": 0,
        },
        "offline_tree_verified": True,
        "history": {
            "shallow": False,
            "commit_ancestry_scope": "all commits reachable from the locked baseline head",
            "commit_ancestry_offline_traversable": True,
            "reachable_commit_count": 1,
            "partial_clone_filter": "none",
            "promisor_remote": False,
            "historical_tree_blob_scope": "all-local",
            "locked_baseline_tree_fully_hydrated": True,
            "fresh_offline_clone_bundle_available": False,
        },
        "nested_submodules": {
            "materialized": False,
            "declared_count": 0,
            "declared_paths": [],
            "gitlink_count": 0,
            "gitlink_paths": [],
            "stale_declarations": [],
            "undeclared_gitlinks": [],
        },
        "compatibility_revisions": [],
        "license_files": [
            {
                "path": "LICENSE",
                "sha256": hashlib.sha256(b"fixture license\n").hexdigest(),
            }
        ],
    }


class VerifyTrackedPolicyTest(unittest.TestCase):
    def test_checkpoint_binds_current_evidence_manifest_and_lessons(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git(root, "init", "-b", "main")
            git(root, "config", "user.name", "test")
            git(root, "config", "user.email", "test@example.invalid")
            (root / "state" / "checkpoints").mkdir(parents=True)
            (root / "state" / "evidence").mkdir()
            (root / "state" / "bitlessons").mkdir()
            for name, payload in (
                ("PLAN.md", b"plan\n"),
                ("GOAL.md", b"goal\n"),
                ("SOURCE_LOCK.json", b"{}\n"),
            ):
                (root / name).write_bytes(payload)
            evidence_path = root / "state" / "evidence" / "EV-0002.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schema": "amdgpu-sim.evidence.v1",
                        "id": "EV-0002",
                        "checkpoint_id": "CP-0002",
                        "type": "fixture",
                        "claim": "fixture evidence",
                        "cwd": ".",
                        "started_at": "2026-08-07T00:00:00Z",
                        "ended_at": "2026-08-07T00:00:01Z",
                        "exit_code": 0,
                        "command_argv": [["true"]],
                        "command_results": [
                            {
                                "id": "fixture-command",
                                "argv": ["true"],
                                "cwd": ".",
                                "exit_code": 0,
                                "result": "passed",
                                "raw_streams_retained": False,
                                "raw_streams_limitation": "unit fixture has no raw streams",
                                "stdout_sha256": None,
                                "stderr_sha256": None,
                            }
                        ],
                        "includes": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            lesson_path = root / "state" / "bitlessons" / "BL-0002.json"
            lesson_path.write_text(
                json.dumps(
                    {
                        "schema": "amdgpu-sim.bitlesson.v1",
                        "id": "BL-0002",
                        "checkpoint_id": "CP-0002",
                        "question_or_symptom": "fixture question",
                        "observation": "fixture observation",
                        "evidence_ids": ["EV-0002"],
                        "conclusion": "fixture conclusion",
                        "decision": "fixture decision",
                        "invalidated_assumptions": ["fixture assumption"],
                        "applies_to": ["fixture"],
                        "confidence": "high",
                        "supersedes": [],
                        "status": "active",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            lesson_sha = hashlib.sha256(lesson_path.read_bytes()).hexdigest()
            source_sha = hashlib.sha256((root / "SOURCE_LOCK.json").read_bytes()).hexdigest()
            checkpoint = {
                "schema": "amdgpu-sim.checkpoint.v1",
                "id": "CP-0002",
                "sequence": 2,
                "parent_checkpoint": "CP-0001",
                "status": "accepted",
                "goal_id": "GSIM-001",
                "phase_id": "P0",
                "plan_sha256": hashlib.sha256((root / "PLAN.md").read_bytes()).hexdigest(),
                "goal_sha256": hashlib.sha256((root / "GOAL.md").read_bytes()).hexdigest(),
                "source_lock_sha256": source_sha,
                "evidence_ids": ["EV-0002"],
                "evidence_sha256": {"EV-0002": evidence_sha},
                "evidence_manifest_id": "EV-0002",
                "evidence_manifest_sha256": evidence_sha,
                "bitlesson_ids": ["BL-0002"],
                "bitlesson_sha256": {"BL-0002": lesson_sha},
                "change_kind": "source",
                "baseline_commit_marker": "N/A",
                "root_parent_commit": None,
                "root_commit": "self-described-by-checkpoint-trailer",
                "next_action": {
                    "id": "NEXT",
                    "cwd": ".",
                    "argv": ["git", "status"],
                    "prerequisites": ["fixture repository exists"],
                    "expected": "git status completes",
                    "rollback_boundary": "no state is changed",
                },
            }
            checkpoint_path = root / "state" / "checkpoints" / "CP-0002.json"
            checkpoint_path.write_text(json.dumps(checkpoint) + "\n", encoding="utf-8")
            current = {
                "schema": "amdgpu-sim.current.v1",
                "checkpoint_id": "CP-0002",
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "goal_id": "GSIM-001",
                "phase_id": "P0",
                "plan_revision": 1,
                "state": "ready",
                "next_action_id": "NEXT",
            }
            (root / "state" / "current.json").write_text(
                json.dumps(current) + "\n", encoding="utf-8"
            )
            git(root, "add", ".")
            message = (
                "checkpoint\n\n"
                "Checkpoint-ID: CP-0002\n"
                "Goal-ID: GSIM-001\n"
                "Plan-Revision: 1\n"
                f"Source-Lock-SHA256: {source_sha}\n"
                f"Evidence-Manifest-SHA256: {evidence_sha}\n"
                "Change-Kind: source\n"
                "Baseline-Commit: N/A"
            )
            subprocess.run(["git", "commit", "-m", message], cwd=root, check=True, stdout=subprocess.DEVNULL)
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            verify_workspace.verify_checkpoint(root, head)
            evidence_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(verify_workspace.VerifyError, "evidence policy mismatch"):
                verify_workspace.verify_checkpoint(root, head)

    def test_capture_record_semantics_must_match_evidence_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record_path = root / "artifacts" / "capture" / "command.json"
            record_path.parent.mkdir(parents=True)
            stdout_descriptor = {
                "path": "artifacts/capture/command.stdout",
                "size": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
                "tracked": False,
                "required_at_acceptance": True,
                "required_for_resume": False,
            }
            stderr_descriptor = {
                "path": "artifacts/capture/command.stderr",
                "size": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
                "tracked": False,
                "required_at_acceptance": True,
                "required_for_resume": False,
            }
            record_path.write_text(
                json.dumps(
                    {
                        "schema": "amdgpu-sim.command-evidence.v1",
                        "argv": ["false"],
                        "cwd": ".",
                        "started_at": "2026-08-07T00:00:00+00:00",
                        "ended_at": "2026-08-07T00:00:01+00:00",
                        "exit_code": 0,
                        "stdout": {
                            key: stdout_descriptor[key]
                            for key in ("path", "size", "sha256")
                        },
                        "stderr": {
                            key: stderr_descriptor[key]
                            for key in ("path", "size", "sha256")
                        },
                    }
                ),
                encoding="utf-8",
            )
            evidence = {
                "command_results": [
                    {
                        "id": "mismatched-command",
                        "argv": ["true"],
                        "cwd": ".",
                        "started_at": "2026-08-07T00:00:00+00:00",
                        "ended_at": "2026-08-07T00:00:01+00:00",
                        "exit_code": 0,
                        "raw_streams_retained": True,
                        "record": {
                            "path": "artifacts/capture/command.json",
                            "size": record_path.stat().st_size,
                            "sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
                            "tracked": False,
                            "required_at_acceptance": True,
                            "required_for_resume": False,
                        },
                        "stdout": stdout_descriptor,
                        "stderr": stderr_descriptor,
                    }
                ]
            }
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "capture record semantics mismatch"
            ):
                verify_workspace.verify_command_capture_records(root, evidence)

    def test_checkpoint_repository_map_matches_frozen_sources(self) -> None:
        source = {
            "id": "example",
            "path": "projects/example",
            "materialization": "gitlink",
            "baseline_commit": "1" * 40,
            "baseline_tree": "2" * 40,
            "work_head": "1" * 40,
            "work_tree": "2" * 40,
            "baseline_tag": "upstream-baseline/example",
            "baseline_tag_object": "3" * 40,
            "administrative_git_dir": "modules/example",
        }
        repository = {
            "id": "example",
            "path": "projects/example",
            "baseline_commit": "1" * 40,
            "baseline_tree": "2" * 40,
            "head": "1" * 40,
            "tree": "2" * 40,
            "baseline_tag": "upstream-baseline/example",
            "baseline_tag_object": "3" * 40,
            "administrative_git_dir": "modules/example",
            "clean": True,
        }
        verify_workspace.verify_checkpoint_repositories(
            {"repositories": [repository]}, {"sources": [source]}
        )
        repository["head"] = "4" * 40
        with self.assertRaisesRegex(
            verify_workspace.VerifyError, "repository identity mismatch"
        ):
            verify_workspace.verify_checkpoint_repositories(
                {"repositories": [repository]}, {"sources": [source]}
            )

    def test_pre_freeze_candidate_requires_exact_manifest_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "artifacts" / "source-lock-pre-freeze.json"
            candidate.parent.mkdir(parents=True)
            candidate.write_text(
                json.dumps(
                    {
                        "schema": "amdgpu-sim.source-lock.v1",
                        "status": "observed-not-fetched",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
            lock = {
                "pre_freeze_candidate_sha256": candidate_sha,
                "pre_freeze_candidate_artifact": {
                    "path": "artifacts/source-lock-pre-freeze.json",
                    "size": candidate.stat().st_size,
                    "sha256": candidate_sha,
                    "required_for_resume": False,
                },
            }
            binding = {
                "path": "artifacts/source-lock-pre-freeze.json",
                "size": candidate.stat().st_size,
                "sha256": candidate_sha,
                "tracked": False,
                "required_at_acceptance": True,
                "required_for_resume": False,
            }
            verify_workspace.verify_pre_freeze_candidate(
                root, lock, {"external_artifacts": [binding]}
            )
            binding["sha256"] = "0" * 64
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "candidate identity mismatch"
            ):
                verify_workspace.verify_pre_freeze_candidate(
                    root, lock, {"external_artifacts": [binding]}
                )

    def test_external_artifact_is_optional_on_resume_but_required_at_acceptance(
        self,
    ) -> None:
        descriptor = {
            "path": "artifacts/missing.json",
            "size": 1,
            "sha256": "0" * 64,
            "tracked": False,
            "required_at_acceptance": True,
            "required_for_resume": False,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            verify_workspace.verify_external_evidence(
                root, [descriptor], require_acceptance_artifacts=False
            )
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "required external evidence is missing"
            ):
                verify_workspace.verify_external_evidence(
                    root, [descriptor], require_acceptance_artifacts=True
                )

    def test_checkpoint_rejects_incomplete_next_action_contract(self) -> None:
        current = {
            "schema": "amdgpu-sim.current.v1",
            "checkpoint_id": "CP-0002",
            "checkpoint_sha256": "fixture-hash",
            "goal_id": "GSIM-001",
            "phase_id": "P0",
            "state": "ready",
            "next_action_id": "P0-SELF-01",
        }
        base_checkpoint = {
            "schema": "amdgpu-sim.checkpoint.v1",
            "id": "CP-0002",
            "sequence": 2,
            "parent_checkpoint": "CP-0001",
            "status": "accepted",
            "goal_id": "GSIM-001",
            "phase_id": "P0",
            "plan_sha256": "fixture-hash",
            "goal_sha256": "fixture-hash",
            "source_lock_sha256": "fixture-hash",
            "next_action": {
                "id": "P0-SELF-01",
                "cwd": ".",
                "argv": ["true"],
                "prerequisites": ["CP-0002 is accepted"],
                "expected": "the next project-authored lane is initialized",
                "rollback_boundary": "leave CP-0002 unchanged",
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for missing in ("prerequisites", "expected", "rollback_boundary"):
                with self.subTest(missing=missing):
                    checkpoint = json.loads(json.dumps(base_checkpoint))
                    checkpoint["next_action"].pop(missing)
                    with mock.patch.object(
                        verify_workspace,
                        "load_json",
                        side_effect=[current, checkpoint],
                    ), mock.patch.object(
                        verify_workspace, "digest", return_value="fixture-hash"
                    ):
                        with self.assertRaisesRegex(
                            verify_workspace.VerifyError,
                            "next action contract is incomplete",
                        ):
                            verify_workspace.verify_checkpoint(root, "1" * 40)

    def test_absorbed_admin_requires_exact_relative_gitfile(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = absorbed_child_fixture(root, absorb=True)
            verify_workspace.verify_child(root, source)

            child = root / "projects" / "example"
            common_dir = git_output(child, "rev-parse", "--git-common-dir")
            common_path = Path(common_dir)
            if not common_path.is_absolute():
                common_path = (child / common_path).resolve()
            (child / ".git").write_text(
                f"gitdir: {common_path}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "exact root module path"
            ):
                verify_workspace.verify_child(root, source)

    def test_absorbed_admin_rejects_embedded_git_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = absorbed_child_fixture(root, absorb=False)
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "absorbed gitfile"
            ):
                verify_workspace.verify_child(root, source)

    def test_source_lock_rejects_wrong_parent_blob_hash(self) -> None:
        parent = "1" * 40
        lock = {
            "schema": "amdgpu-sim.source-lock.v1",
            "lock_id": "SL-0001",
            "status": "frozen",
            "frozen_by_checkpoint": "CP-0002",
            "accepted_parent_root_commit": parent,
            "accepted_parent_source_lock_sha256": "0" * 64,
            "pre_freeze_candidate_sha256": "2" * 64,
            "pre_freeze_candidate_artifact": {
                "path": "artifacts/source-lock-pre-freeze.json",
                "size": 1,
                "sha256": "2" * 64,
                "required_for_resume": False,
            },
            "sources": [],
        }
        checkpoint = {"root_parent_commit": parent, "repositories": []}
        completed = subprocess.CompletedProcess([], 0, b"accepted parent\n", b"")
        with mock.patch.object(
            verify_workspace, "load_json", side_effect=[lock, checkpoint]
        ), mock.patch.object(
            verify_workspace, "verify_checkpoint_repositories"
        ), mock.patch.object(verify_workspace, "run", return_value=completed):
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "accepted parent blob hash mismatch"
            ):
                verify_workspace.verify_sources(Path("unused"), "CP-0002")

    def test_source_lock_rejects_noncanonical_evidence_path(self) -> None:
        parent = "1" * 40
        parent_blob = b"accepted parent\n"
        lock = {
            "schema": "amdgpu-sim.source-lock.v1",
            "lock_id": "SL-0001",
            "status": "frozen",
            "frozen_by_checkpoint": "CP-0002",
            "accepted_parent_root_commit": parent,
            "accepted_parent_source_lock_sha256": hashlib.sha256(parent_blob).hexdigest(),
            "pre_freeze_candidate_sha256": "2" * 64,
            "pre_freeze_candidate_artifact": {
                "path": "artifacts/source-lock-pre-freeze.json",
                "size": 1,
                "sha256": "2" * 64,
                "required_for_resume": False,
            },
            "frozen_at": "2026-08-08T00:00:00+08:00",
            "resolution_evidence_id": "EV-0002",
            "resolution_evidence_path": "state/evidence/alias.json",
            "resolution_evidence_sha256": "3" * 64,
            "sources": [],
        }
        checkpoint = {"root_parent_commit": parent, "repositories": []}
        completed = subprocess.CompletedProcess([], 0, parent_blob, b"")
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            verify_workspace, "load_json", side_effect=[lock, checkpoint]
        ), mock.patch.object(
            verify_workspace, "verify_checkpoint_repositories"
        ), mock.patch.object(verify_workspace, "run", return_value=completed):
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "evidence path is not canonical"
            ):
                verify_workspace.verify_sources(Path(temp), "CP-0002")

    def test_active_transaction_is_reported_as_recoverable_pending_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git(root, "init", "-b", "main")
            git(root, "config", "user.name", "test")
            git(root, "config", "user.email", "test@example.invalid")
            (root / "README.md").write_text("baseline\n", encoding="utf-8")
            git(root, "add", "README.md")
            git(root, "commit", "-m", "baseline")
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            journal = root / ".git" / "amdgpu-sim" / "txn" / "CP-0002.json"
            journal.parent.mkdir(parents=True)
            journal.write_text(
                json.dumps(
                    {
                        "schema": "amdgpu-sim.transaction.v1",
                        "checkpoint_id": "CP-0002",
                        "phase": "prepare",
                        "previous_root": head,
                        "expected_children": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                verify_workspace.PendingTransaction, "CP-0002 phase=prepare"
            ):
                verify_workspace.verify_root_state(root)

    def test_allow_transaction_rejects_unlocked_participant_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git(root, "init", "-b", "main")
            git(root, "config", "user.name", "test")
            git(root, "config", "user.email", "test@example.invalid")
            (root / "README.md").write_text("baseline\n", encoding="utf-8")
            git(root, "add", "README.md")
            git(root, "commit", "-m", "baseline")
            parent = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            (root / "README.md").write_text("coordinator\n", encoding="utf-8")
            git(root, "add", "README.md")
            git(root, "commit", "-m", "coordinator")
            tree = subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
            ).strip()
            journal = root / ".git" / "amdgpu-sim" / "txn" / "CP-0002.json"
            journal.parent.mkdir(parents=True)
            journal.write_text(
                json.dumps(
                    {
                        "schema": "amdgpu-sim.transaction.v1",
                        "checkpoint_id": "CP-0002",
                        "phase": "prepared",
                        "participants_locked": False,
                        "previous_root": parent,
                        "expected_root_tree": tree,
                        "declared_children": {},
                        "expected_children": {},
                        "root_allowlist": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(verify_workspace.VerifyError, "participant/allowlist"):
                verify_workspace.verify_root_state(root, allow_transaction="CP-0002")

    def test_rejects_forced_virtual_environment_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git(root, "init", "-b", "main")
            path = root / ".venv" / "pyvenv.cfg"
            path.parent.mkdir()
            path.write_text("generated\n", encoding="utf-8")
            git(root, "add", "-f", ".venv/pyvenv.cfg")
            with self.assertRaises(verify_workspace.VerifyError):
                verify_workspace.verify_tracked_policy(root)

    def test_parse_trailers_preserves_duplicates_for_rejection(self) -> None:
        parsed = verify_workspace.parse_trailers(
            "subject\n\nCheckpoint-ID: CP-0002\nCheckpoint-ID: CP-9999\n"
        )
        self.assertEqual(parsed["Checkpoint-ID"], ["CP-0002", "CP-9999"])

    def test_annotated_tag_identity_and_offline_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            git(repo, "init", "-b", "main")
            git(repo, "config", "user.name", "test")
            git(repo, "config", "user.email", "test@example.invalid")
            (repo / "source.txt").write_text("baseline\n", encoding="utf-8")
            git(repo, "add", "source.txt")
            git(repo, "commit", "-m", "baseline")
            git(repo, "tag", "-a", "baseline-tag", "-m", "frozen")
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            tree = subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True
            ).strip()
            tag_object = subprocess.check_output(
                ["git", "rev-parse", "baseline-tag"], cwd=repo, text=True
            ).strip()
            payload = subprocess.check_output(
                ["git", "cat-file", "tag", "baseline-tag"], cwd=repo, text=True
            )
            verify_workspace.verify_annotated_tag(
                repo,
                repo,
                "fixture",
                commit=commit,
                tree=tree,
                tag="baseline-tag",
                tag_object=tag_object,
                tag_payload=payload,
                tag_payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),
            )
            with self.assertRaises(verify_workspace.VerifyError):
                verify_workspace.verify_annotated_tag(
                    repo,
                    repo,
                    "fixture",
                    commit=commit,
                    tree="0" * 40,
                    tag="baseline-tag",
                    tag_object=tag_object,
                    tag_payload=payload,
                    tag_payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                )

    def test_accepts_normal_control_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git(root, "init", "-b", "main")
            (root / "README.md").write_text("ok\n", encoding="utf-8")
            git(root, "add", "README.md")
            verify_workspace.verify_tracked_policy(root)

    def test_rejects_forced_model_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git(root, "init", "-b", "main")
            (root / "model.safetensors").write_bytes(b"not a model")
            git(root, "add", "-f", "model.safetensors")
            with self.assertRaises(verify_workspace.VerifyError):
                verify_workspace.verify_tracked_policy(root)

    def test_rejects_non_gitlink_project_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git(root, "init", "-b", "main")
            (root / "projects").mkdir()
            (root / "projects" / "bad.txt").write_text("bad\n", encoding="utf-8")
            git(root, "add", "projects/bad.txt")
            with self.assertRaises(verify_workspace.VerifyError):
                verify_workspace.verify_tracked_policy(root)


if __name__ == "__main__":
    unittest.main()
