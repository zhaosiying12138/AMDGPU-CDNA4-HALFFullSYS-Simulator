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
FIXTURE_SOURCE_LOCK = b"fixture source lock\n"
FIXTURE_SOURCE_LOCK_SHA256 = hashlib.sha256(FIXTURE_SOURCE_LOCK).hexdigest()
FIXTURE_EVIDENCE_SHA256 = hashlib.sha256(b"fixture evidence\n").hexdigest()
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


def transaction_child_fixture(root: Path) -> dict[str, str]:
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    child = root / "projects" / "child"
    child.mkdir(parents=True)
    git(child, "init", "-b", "main")
    git(child, "config", "user.name", "test")
    git(child, "config", "user.email", "test@example.invalid")
    (child / "source.txt").write_text("initial\n", encoding="utf-8")
    git(child, "add", "source.txt")
    git(child, "commit", "-m", "initial")
    initial_head = git_output(child, "rev-parse", "HEAD")
    initial_tree = git_output(child, "rev-parse", "HEAD^{tree}")
    git(root, "add", "projects/child")
    git(root, "commit", "-m", "root initial")
    previous_root = git_output(root, "rev-parse", "HEAD")
    (child / "source.txt").write_text("target\n", encoding="utf-8")
    git(child, "add", "source.txt")
    git(child, "commit", "-m", "target")
    target_head = git_output(child, "rev-parse", "HEAD")
    target_tree = git_output(child, "rev-parse", "HEAD^{tree}")
    return {
        "path": "projects/child",
        "previous_root": previous_root,
        "initial_head": initial_head,
        "initial_tree": initial_tree,
        "target_head": target_head,
        "target_tree": target_tree,
    }


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


def empty_project_lanes() -> dict[str, object]:
    return {
        "schema": "amdgpu-sim.project-lanes.v1",
        "registry_revision": 1,
        "url_policy": {
            "scheme": "sibling-relative-v1",
            "template": "../{project_id}.git",
        },
        "lanes": [],
    }


def authored_lane_identity(
    *, lane_id: str = "self-amdgpu-runtime", baseline: str = "4" * 40
) -> dict[str, object]:
    payload = (
        "fixture tag payload\n\n"
        f"Project-ID: {lane_id}\n"
        f"Project-URL: ../{lane_id}.git\n"
        "Origin-Policy: sibling-relative-v1\n"
        f"Baseline-Commit: {baseline}\n"
        f"Baseline-Tree: {'5' * 40}\n"
        "License-SPDX: GPL-3.0-or-later\n"
        "Created-At: 2026-08-08T03:00:00+08:00\n"
    )
    return {
        "id": lane_id,
        "ownership": "project-authored",
        "role": "host-runtime",
        "path": f"projects/{lane_id}",
        "materialization": "gitlink",
        "origin": {
            "policy": "sibling-relative-v1",
            "remote": "origin",
            "url": f"../{lane_id}.git",
            "push_url": "no_push",
            "reachability": "not-asserted",
            "branch": "main",
        },
        "baseline_commit": baseline,
        "baseline_tree": "5" * 40,
        "baseline_tag": f"project-baseline/{lane_id}/{baseline}",
        "baseline_tag_object": "6" * 40,
        "baseline_tag_payload": payload,
        "baseline_tag_payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "baseline_created_at": "2026-08-08T03:00:00+08:00",
        "baseline_checkpoint_id": "CP-0003",
        "baseline_evidence_id": "EV-0004",
        "baseline_commit_trailers": {
            "checkpoint_id": "CP-0003",
            "goal_id": "GSIM-001",
            "plan_revision": 1,
            "source_lock_sha256": FIXTURE_SOURCE_LOCK_SHA256,
            "evidence_manifest_sha256": FIXTURE_EVIDENCE_SHA256,
            "change_kind": "baseline",
            "baseline_commit_marker": "N/A",
        },
        "administrative_git_dir": f"modules/{lane_id}",
        "license": {
            "spdx_id": "GPL-3.0-or-later",
            "path": "LICENSE",
            "sha256": hashlib.sha256(b"fixture license\n").hexdigest(),
        },
    }


def absorbed_authored_fixture(
    root: Path,
) -> tuple[dict[str, object], dict[str, object], Path]:
    lane_id = "self-amdgpu-runtime"
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    (root / "SOURCE_LOCK.json").write_bytes(FIXTURE_SOURCE_LOCK)
    child = root / "projects" / lane_id
    child.mkdir(parents=True)
    git(child, "init", "-b", "main")
    git(child, "config", "user.name", "test")
    git(child, "config", "user.email", "test@example.invalid")
    license_bytes = b"fixture license\n"
    (child / "LICENSE").write_bytes(license_bytes)
    (child / "runtime.txt").write_text("baseline\n", encoding="utf-8")
    git(child, "add", "LICENSE", "runtime.txt")
    baseline_message = (
        "baseline\n\n"
        "Checkpoint-ID: CP-0003\n"
        "Goal-ID: GSIM-001\n"
        "Plan-Revision: 1\n"
        f"Source-Lock-SHA256: {FIXTURE_SOURCE_LOCK_SHA256}\n"
        f"Evidence-Manifest-SHA256: {FIXTURE_EVIDENCE_SHA256}\n"
        "Change-Kind: baseline\n"
        "Baseline-Commit: N/A"
    )
    git(child, "commit", "-m", baseline_message)
    baseline = git_output(child, "rev-parse", "HEAD")
    baseline_tree = git_output(child, "rev-parse", "HEAD^{tree}")
    origin_url = f"../{lane_id}.git"
    tag = f"project-baseline/{lane_id}/{baseline}"
    tag_message = (
        "Project-authored fixture baseline\n\n"
        f"Project-ID: {lane_id}\n"
        f"Project-URL: {origin_url}\n"
        "Origin-Policy: sibling-relative-v1\n"
        f"Baseline-Commit: {baseline}\n"
        f"Baseline-Tree: {baseline_tree}"
        "\nLicense-SPDX: GPL-3.0-or-later"
        "\nCreated-At: 2026-08-08T03:00:00+08:00"
    )
    git(child, "tag", "-a", tag, "-m", tag_message)
    tag_object = git_output(child, "rev-parse", tag)
    tag_payload = subprocess.check_output(
        ["git", "cat-file", "tag", tag], cwd=child, text=True
    )
    (child / "runtime.txt").write_text("descendant\n", encoding="utf-8")
    git(child, "add", "runtime.txt")
    git(child, "commit", "-m", "descendant")
    head = git_output(child, "rev-parse", "HEAD")
    tree = git_output(child, "rev-parse", "HEAD^{tree}")
    git(child, "remote", "add", "origin", origin_url)
    git(child, "remote", "set-url", "--push", "origin", "no_push")
    git(child, "config", "core.hooksPath", "../../.githooks")
    (root / ".gitmodules").write_text(
        f'[submodule "{lane_id}"]\n'
        f"\tpath = projects/{lane_id}\n"
        f"\turl = {origin_url}\n"
        "\tbranch = main\n",
        encoding="utf-8",
    )
    git(root, "add", ".gitmodules")
    subprocess.run(
        ["git", "add", f"projects/{lane_id}"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "submodule", "absorbgitdirs", f"projects/{lane_id}"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    lane = authored_lane_identity(lane_id=lane_id, baseline=baseline)
    lane.update(
        {
            "baseline_tree": baseline_tree,
            "baseline_tag": tag,
            "baseline_tag_object": tag_object,
            "baseline_tag_payload": tag_payload,
            "baseline_tag_payload_sha256": hashlib.sha256(
                tag_payload.encode()
            ).hexdigest(),
            "license": {
                "spdx_id": "GPL-3.0-or-later",
                "path": "LICENSE",
                "sha256": hashlib.sha256(license_bytes).hexdigest(),
            },
        }
    )
    current = {
        "id": lane_id,
        "path": f"projects/{lane_id}",
        "baseline_commit": baseline,
        "baseline_tree": baseline_tree,
        "head": head,
        "tree": tree,
        "baseline_tag": tag,
        "baseline_tag_object": tag_object,
        "administrative_git_dir": f"modules/{lane_id}",
        "clean": True,
    }
    return lane, current, child


def commit_authored_registry_checkpoint(
    root: Path,
    lane: dict[str, object],
    *,
    include_child_evidence: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    evidence = {
        "schema": "amdgpu-sim.evidence.v1",
        "id": "EV-0004",
        "checkpoint_id": "CP-0003",
    }
    evidence_bytes = (json.dumps(evidence, sort_keys=True) + "\n").encode()
    evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()
    lane["baseline_evidence_id"] = "EV-0004"
    lane["baseline_commit_trailers"]["evidence_manifest_sha256"] = evidence_sha
    registry = empty_project_lanes()
    registry["lanes"] = [lane]
    registry_bytes = (json.dumps(registry, sort_keys=True) + "\n").encode()
    registry_sha = hashlib.sha256(registry_bytes).hexdigest()
    manifest = {
        "schema": "amdgpu-sim.evidence.v1",
        "id": "EV-0005",
        "checkpoint_id": "CP-0003",
        "includes": {"EV-0004": evidence_sha} if include_child_evidence else {},
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    checkpoint = {
        "id": "CP-0003",
        "goal_id": "GSIM-001",
        "source_lock_sha256": lane["baseline_commit_trailers"][
            "source_lock_sha256"
        ],
        "change_kind": "baseline",
        "baseline_commit_marker": "N/A",
        "project_lanes_sha256": registry_sha,
        "evidence_ids": ["EV-0004", "EV-0005"],
        "evidence_sha256": {
            "EV-0004": evidence_sha,
            "EV-0005": manifest_sha,
        },
        "evidence_manifest_id": "EV-0005",
        "evidence_manifest_sha256": manifest_sha,
        "repositories": [],
    }
    for relative, payload in (
        ("PROJECT_LANES.json", registry_bytes),
        ("state/evidence/EV-0004.json", evidence_bytes),
        ("state/evidence/EV-0005.json", manifest_bytes),
        (
            "state/checkpoints/CP-0003.json",
            (json.dumps(checkpoint, sort_keys=True) + "\n").encode(),
        ),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    git(root, "add", ".")
    message = (
        "registry\n\n"
        "Checkpoint-ID: CP-0003\n"
        "Goal-ID: GSIM-001\n"
        "Plan-Revision: 1\n"
        f"Source-Lock-SHA256: {lane['baseline_commit_trailers']['source_lock_sha256']}\n"
        f"Evidence-Manifest-SHA256: {manifest_sha}\n"
        "Change-Kind: baseline\n"
        "Baseline-Commit: N/A"
    )
    git(root, "commit", "-m", message)
    return registry, checkpoint


class VerifyTrackedPolicyTest(unittest.TestCase):
    def test_coordinator_gitlinks_must_equal_transaction_participants(self) -> None:
        raw = (
            b":000000 160000 0000000 1111111 A\0projects/declared\0"
            b":000000 160000 0000000 2222222 A\0projects/undeclared\0"
        )
        completed = subprocess.CompletedProcess(
            args=["git", "diff"], returncode=0, stdout=raw, stderr=b""
        )
        declared = {"declared": {"path": "projects/declared"}}
        with mock.patch.object(verify_workspace, "run", return_value=completed):
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "extra=.*projects/undeclared"
            ):
                verify_workspace.verify_coordinator_participant_gitlinks(
                    Path("."), "a" * 40, "b" * 40, declared
                )

    def test_coordinator_participant_gitlink_diff_is_accepted(self) -> None:
        raw = b":000000 160000 0000000 1111111 A\0projects/declared\0"
        completed = subprocess.CompletedProcess(
            args=["git", "diff"], returncode=0, stdout=raw, stderr=b""
        )
        declared = {"declared": {"path": "projects/declared"}}
        with mock.patch.object(verify_workspace, "run", return_value=completed):
            verify_workspace.verify_coordinator_participant_gitlinks(
                Path("."), "a" * 40, "b" * 40, declared
            )

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
                (
                    "PROJECT_LANES.json",
                    json.dumps(
                        {
                            "schema": "amdgpu-sim.project-lanes.v1",
                            "registry_revision": 1,
                            "url_policy": {
                                "scheme": "sibling-relative-v1",
                                "template": "../{project_id}.git",
                            },
                            "lanes": [],
                        }
                    ).encode()
                    + b"\n",
                ),
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
            lanes_sha = hashlib.sha256(
                (root / "PROJECT_LANES.json").read_bytes()
            ).hexdigest()
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
                "project_lanes_sha256": lanes_sha,
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
            registry_path = root / "PROJECT_LANES.json"
            registry_bytes = registry_path.read_bytes()
            registry_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                verify_workspace.VerifyError,
                "checkpoint hash mismatch: PROJECT_LANES.json",
            ):
                verify_workspace.verify_checkpoint(root, head)
            registry_path.write_bytes(registry_bytes)
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
        # Current head/tree belong to the checkpoint and may advance past the
        # immutable upstream baseline. Repository ancestry is checked against
        # the materialized child by verify_child.
        repository["head"] = "4" * 40
        repository["tree"] = "5" * 40
        verify_workspace.verify_checkpoint_repositories(
            {"repositories": [repository]}, {"sources": [source]}
        )
        repository["clean"] = False
        with self.assertRaisesRegex(
            verify_workspace.VerifyError, "repository identity mismatch"
        ):
            verify_workspace.verify_checkpoint_repositories(
                {"repositories": [repository]}, {"sources": [source]}
            )

    def test_checkpoint_repository_map_is_exact_upstream_authored_union(self) -> None:
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
        lane = authored_lane_identity()
        registry = empty_project_lanes()
        registry["lanes"] = [lane]
        repositories = [
            {
                "id": source["id"],
                "path": source["path"],
                "baseline_commit": source["baseline_commit"],
                "baseline_tree": source["baseline_tree"],
                "head": "7" * 40,
                "tree": "8" * 40,
                "baseline_tag": source["baseline_tag"],
                "baseline_tag_object": source["baseline_tag_object"],
                "administrative_git_dir": source["administrative_git_dir"],
                "clean": True,
            },
            {
                "id": lane["id"],
                "path": lane["path"],
                "baseline_commit": lane["baseline_commit"],
                "baseline_tree": lane["baseline_tree"],
                "head": "9" * 40,
                "tree": "a" * 40,
                "baseline_tag": lane["baseline_tag"],
                "baseline_tag_object": lane["baseline_tag_object"],
                "administrative_git_dir": lane["administrative_git_dir"],
                "clean": True,
            },
        ]
        result = verify_workspace.verify_checkpoint_repositories(
            {"repositories": repositories}, {"sources": [source]}, registry
        )
        self.assertEqual(set(result), {"example", "self-amdgpu-runtime"})
        registry["lanes"][0]["id"] = "example"
        registry["lanes"][0]["path"] = "projects/example"
        registry["lanes"][0]["administrative_git_dir"] = "modules/example"
        registry["lanes"][0]["origin"]["url"] = "../example.git"
        registry["lanes"][0]["baseline_tag"] = (
            f"project-baseline/example/{registry['lanes'][0]['baseline_commit']}"
        )
        registry["lanes"][0]["baseline_tag_payload"] = registry["lanes"][0][
            "baseline_tag_payload"
        ].replace(
            "Project-ID: self-amdgpu-runtime",
            "Project-ID: example",
        ).replace(
            "Project-URL: ../self-amdgpu-runtime.git",
            "Project-URL: ../example.git",
        )
        registry["lanes"][0]["baseline_tag_payload_sha256"] = hashlib.sha256(
            registry["lanes"][0]["baseline_tag_payload"].encode()
        ).hexdigest()
        with self.assertRaisesRegex(
            verify_workspace.VerifyError,
            "upstream/authored lane union",
        ):
            verify_workspace.verify_checkpoint_repositories(
                {"repositories": repositories}, {"sources": [source]}, registry
            )

    def test_project_lanes_rejects_current_head_ownership_and_bad_license(self) -> None:
        registry = empty_project_lanes()
        lane = authored_lane_identity()
        registry["lanes"] = [lane]
        self.assertEqual(verify_workspace.authored_lanes(registry), [lane])
        lane["work_head"] = lane["baseline_commit"]
        with self.assertRaisesRegex(
            verify_workspace.VerifyError, "must not own current head/tree"
        ):
            verify_workspace.authored_lanes(registry)
        lane.pop("work_head")
        lane["license"]["spdx_id"] = "MIT"
        with self.assertRaisesRegex(
            verify_workspace.VerifyError, "identity is incomplete"
        ):
            verify_workspace.authored_lanes(registry)

    def test_project_lanes_schema_rejects_unknown_fields_at_every_level(self) -> None:
        mutations = (
            ("registry", lambda value: value.__setitem__("unexpected", True)),
            (
                "lane",
                lambda value: value["lanes"][0].__setitem__("unexpected", True),
            ),
            (
                "origin",
                lambda value: value["lanes"][0]["origin"].__setitem__(
                    "unexpected", True
                ),
            ),
            (
                "license",
                lambda value: value["lanes"][0]["license"].__setitem__(
                    "unexpected", True
                ),
            ),
        )
        for level, mutate in mutations:
            with self.subTest(level=level):
                registry = empty_project_lanes()
                registry["lanes"] = [authored_lane_identity()]
                mutate(registry)
                with self.assertRaisesRegex(
                    verify_workspace.VerifyError, "unknown fields"
                ):
                    verify_workspace.authored_lanes(registry)

    def test_project_lanes_rejects_bool_revision_bad_branch_and_duplicate_tag_key(self) -> None:
        registry = empty_project_lanes()
        registry["registry_revision"] = True
        with self.assertRaisesRegex(
            verify_workspace.VerifyError, "registry revision"
        ):
            verify_workspace.authored_lanes(registry)

        registry = empty_project_lanes()
        lane = authored_lane_identity()
        registry["lanes"] = [lane]
        lane["origin"]["branch"] = "bad\nbranch"
        with self.assertRaisesRegex(
            verify_workspace.VerifyError, "identity is incomplete"
        ):
            verify_workspace.authored_lanes(registry)

        lane = authored_lane_identity()
        registry["lanes"] = [lane]
        lane["baseline_tag_payload"] += f"Project-ID: {lane['id']}\n"
        with self.assertRaisesRegex(
            verify_workspace.VerifyError, "duplicate.*Project-ID"
        ):
            verify_workspace.authored_lanes(registry)

    def test_root_gitmodules_rejects_include_or_other_unknown_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".gitmodules").write_text(
                '[submodule "example"]\n'
                "\tpath = projects/example\n"
                "\turl = https://example.invalid/example.git\n"
                "\tbranch = main\n"
                "[include]\n"
                "\tpath = extra.conf\n",
                encoding="utf-8",
            )
            source = {
                "id": "example",
                "path": "projects/example",
                "materialization": "gitlink",
            }
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "unexpected keys"
            ):
                verify_workspace.verify_root_gitmodules(
                    root, [source], [], {"projects/example"}
                )

    def test_progress_commits_bind_checkpoint_evidence_and_baseline(self) -> None:
        cases = (
            ("valid", None),
            ("checkpoint", "mismatched Checkpoint-ID"),
            ("evidence", "evidence is not in current checkpoint"),
            ("baseline", "mismatched Baseline-Commit"),
            ("fork", "does not descend from previous checkpoint"),
            ("rollback", "does not descend from previous checkpoint"),
        )
        for case, error in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                git(root, "init", "-b", "main")
                git(root, "config", "user.name", "test")
                git(root, "config", "user.email", "test@example.invalid")
                child = root / "projects" / "example"
                child.mkdir(parents=True)
                git(child, "init", "-b", "main")
                git(child, "config", "user.name", "test")
                git(child, "config", "user.email", "test@example.invalid")
                (child / "payload").write_text("baseline\n", encoding="utf-8")
                git(child, "add", "payload")
                git(child, "commit", "-m", "baseline")
                baseline = git_output(child, "rev-parse", "HEAD")
                parent_head = baseline
                if case == "rollback":
                    (child / "payload").write_text(
                        "prior descendant\n", encoding="utf-8"
                    )
                    git(child, "add", "payload")
                    git(child, "commit", "-m", "prior descendant")
                    parent_head = git_output(child, "rev-parse", "HEAD")
                (root / "state" / "checkpoints").mkdir(parents=True)
                parent = {
                    "id": "CP-0003",
                    "evidence_ids": [],
                    "evidence_sha256": {},
                    "bitlesson_ids": [],
                    "bitlesson_sha256": {},
                    "repositories": [{"id": "example", "head": parent_head}],
                }
                (root / "state" / "checkpoints" / "CP-0003.json").write_text(
                    json.dumps(parent), encoding="utf-8"
                )
                git(root, "add", "state/checkpoints/CP-0003.json")
                git(
                    root,
                    "commit",
                    "-m",
                    "parent\n\nCheckpoint-ID: CP-0003",
                )
                allowed_evidence = "a" * 64
                checkpoint_id = "CP-9999" if case == "checkpoint" else "CP-0004"
                evidence = "b" * 64 if case == "evidence" else allowed_evidence
                declared_baseline = "c" * 40 if case == "baseline" else baseline
                message = (
                    "progress\n\n"
                    f"Checkpoint-ID: {checkpoint_id}\n"
                    "Goal-ID: GSIM-001\n"
                    "Plan-Revision: 1\n"
                    f"Source-Lock-SHA256: {'d' * 64}\n"
                    f"Evidence-Manifest-SHA256: {evidence}\n"
                    "Change-Kind: source\n"
                    f"Baseline-Commit: {declared_baseline}"
                )
                if case == "rollback":
                    head = baseline
                elif case == "fork":
                    empty_tree = subprocess.run(
                        ["git", "mktree"],
                        cwd=child,
                        input="",
                        text=True,
                        check=True,
                        stdout=subprocess.PIPE,
                    ).stdout.strip()
                    head = subprocess.run(
                        ["git", "commit-tree", empty_tree, "-F", "-"],
                        cwd=child,
                        input=message,
                        text=True,
                        check=True,
                        stdout=subprocess.PIPE,
                    ).stdout.strip()
                else:
                    (child / "payload").write_text("progress\n", encoding="utf-8")
                    git(child, "add", "payload")
                    git(child, "commit", "-m", message)
                    head = git_output(child, "rev-parse", "HEAD")
                checkpoint = {
                    "id": "CP-0004",
                    "parent_checkpoint": "CP-0003",
                    "goal_id": "GSIM-001",
                    "source_lock_sha256": "d" * 64,
                    "change_kind": "source",
                    "evidence_sha256": {"EV-0006": allowed_evidence},
                }
                authority = {
                    "id": "example",
                    "path": "projects/example",
                    "baseline_commit": baseline,
                }
                current = {"head": head}
                if error is None:
                    verify_workspace.verify_current_progress_commits(
                        root, checkpoint, 1, authority, current
                    )
                else:
                    with self.assertRaisesRegex(verify_workspace.VerifyError, error):
                        verify_workspace.verify_current_progress_commits(
                            root, checkpoint, 1, authority, current
                        )

    def test_source_lock_history_rejects_synchronized_live_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git(root, "init", "-b", "main")
            git(root, "config", "user.name", "test")
            git(root, "config", "user.email", "test@example.invalid")
            lock = {"frozen_by_checkpoint": "CP-0002", "value": "accepted"}
            lock_bytes = (json.dumps(lock, sort_keys=True) + "\n").encode()
            checkpoint = {
                "id": "CP-0002",
                "source_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            }
            checkpoint_path = root / "state" / "checkpoints" / "CP-0002.json"
            checkpoint_path.parent.mkdir(parents=True)
            (root / "SOURCE_LOCK.json").write_bytes(lock_bytes)
            checkpoint_path.write_text(
                json.dumps(checkpoint, sort_keys=True) + "\n", encoding="utf-8"
            )
            git(root, "add", ".")
            git(
                root,
                "commit",
                "-m",
                "freeze\n\nCheckpoint-ID: CP-0002",
            )
            verify_workspace.verify_source_lock_history(root, lock, checkpoint)

            tampered_lock = {
                "frozen_by_checkpoint": "CP-0002",
                "value": "tampered",
            }
            tampered_bytes = (
                json.dumps(tampered_lock, sort_keys=True) + "\n"
            ).encode()
            tampered_checkpoint = {
                "id": "CP-0002",
                "source_lock_sha256": hashlib.sha256(tampered_bytes).hexdigest(),
            }
            (root / "SOURCE_LOCK.json").write_bytes(tampered_bytes)
            checkpoint_path.write_text(
                json.dumps(tampered_checkpoint, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "differs from accepted history"
            ):
                verify_workspace.verify_source_lock_history(
                    root, tampered_lock, tampered_checkpoint
                )

    def test_historical_checkpoint_rejects_old_evidence_or_lesson_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            git(root, "init", "-b", "main")
            git(root, "config", "user.name", "test")
            git(root, "config", "user.email", "test@example.invalid")
            evidence_bytes = b'{"id":"EV-0001"}\n'
            lesson_bytes = b'{"id":"BL-0001"}\n'
            checkpoint = {
                "id": "CP-0001",
                "parent_checkpoint": None,
                "evidence_ids": ["EV-0001"],
                "evidence_sha256": {
                    "EV-0001": hashlib.sha256(evidence_bytes).hexdigest()
                },
                "bitlesson_ids": ["BL-0001"],
                "bitlesson_sha256": {
                    "BL-0001": hashlib.sha256(lesson_bytes).hexdigest()
                },
            }
            for relative, payload in (
                ("state/evidence/EV-0001.json", evidence_bytes),
                ("state/bitlessons/BL-0001.json", lesson_bytes),
                (
                    "state/checkpoints/CP-0001.json",
                    (json.dumps(checkpoint, sort_keys=True) + "\n").encode(),
                ),
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            git(root, "add", ".")
            git(
                root,
                "commit",
                "-m",
                "checkpoint\n\nCheckpoint-ID: CP-0001",
            )
            verify_workspace.accepted_checkpoint(root, "CP-0001")
            evidence_path = root / "state/evidence/EV-0001.json"
            evidence_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "evidence differs from acceptance"
            ):
                verify_workspace.accepted_checkpoint(root, "CP-0001")
            evidence_path.write_bytes(evidence_bytes)
            (root / "state/bitlessons/BL-0001.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "bitlessons differs from acceptance"
            ):
                verify_workspace.accepted_checkpoint(root, "CP-0001")

    def test_authored_lane_history_requires_layered_child_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry, _checkpoint = commit_authored_registry_checkpoint(
                root, authored_lane_identity()
            )
            accepted = verify_workspace.verify_authored_lane_history(root, registry)
            self.assertEqual(set(accepted), {"self-amdgpu-runtime"})

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry, _checkpoint = commit_authored_registry_checkpoint(
                root,
                authored_lane_identity(),
                include_child_evidence=False,
            )
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "absent from umbrella manifest"
            ):
                verify_workspace.verify_authored_lane_history(root, registry)

    def test_authored_lane_history_rejects_checkpoint_tag_or_baseline_rewrite(self) -> None:
        for field in ("checkpoint", "tag", "baseline"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                registry, _checkpoint = commit_authored_registry_checkpoint(
                    root, authored_lane_identity()
                )
                lane = registry["lanes"][0]
                if field == "checkpoint":
                    lane["baseline_checkpoint_id"] = "CP-0004"
                    lane["baseline_commit_trailers"]["checkpoint_id"] = "CP-0004"
                elif field == "tag":
                    lane["baseline_tag_object"] = "f" * 40
                else:
                    previous = lane["baseline_commit"]
                    lane["baseline_commit"] = "e" * 40
                    lane["baseline_tag"] = (
                        f"project-baseline/{lane['id']}/{lane['baseline_commit']}"
                    )
                    lane["baseline_tag_payload"] = lane[
                        "baseline_tag_payload"
                    ].replace(
                        f"Baseline-Commit: {previous}",
                        f"Baseline-Commit: {lane['baseline_commit']}",
                    )
                    lane["baseline_tag_payload_sha256"] = hashlib.sha256(
                        lane["baseline_tag_payload"].encode()
                    ).hexdigest()
                (root / "PROJECT_LANES.json").write_text(
                    json.dumps(registry, sort_keys=True) + "\n", encoding="utf-8"
                )
                git(root, "add", "PROJECT_LANES.json")
                git(
                    root,
                    "commit",
                    "-m",
                    "rewrite\n\nCheckpoint-ID: CP-0004",
                )
                with self.assertRaisesRegex(
                    verify_workspace.VerifyError, "rewrites an authored baseline"
                ):
                    verify_workspace.verify_authored_lane_history(root, registry)

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
            "project_lanes_sha256": "fixture-hash",
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

    def test_authored_child_accepts_checkpoint_descendant_and_rejects_fork(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lane, current, child = absorbed_authored_fixture(root)
            verify_workspace.verify_authored_child(root, lane, current)
            wrong_identity = json.loads(json.dumps(lane))
            wrong_identity["baseline_commit_trailers"]["goal_id"] = "GSIM-OTHER"
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "mismatched Goal-ID trailer"
            ):
                verify_workspace.verify_authored_child(root, wrong_identity, current)

            empty_tree = subprocess.run(
                ["git", "mktree"],
                cwd=child,
                input="",
                text=True,
                check=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            unrelated = subprocess.run(
                ["git", "commit-tree", empty_tree, "-m", "unrelated"],
                cwd=child,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            git(child, "checkout", "--detach", unrelated)
            current["head"] = unrelated
            current["tree"] = empty_tree
            subprocess.run(
                [
                    "git",
                    "update-index",
                    "--cacheinfo",
                    f"160000,{unrelated},projects/self-amdgpu-runtime",
                ],
                cwd=root,
                check=True,
            )
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "not a baseline descendant"
            ):
                verify_workspace.verify_authored_child(root, lane, current)

    def test_upstream_child_accepts_checkpoint_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = absorbed_child_fixture(root, absorb=True)
            child = root / "projects" / "example"
            (child / "source.txt").write_text("descendant\n", encoding="utf-8")
            git(child, "add", "source.txt")
            git(child, "commit", "-m", "descendant")
            head = git_output(child, "rev-parse", "HEAD")
            tree = git_output(child, "rev-parse", "HEAD^{tree}")
            subprocess.run(
                [
                    "git",
                    "update-index",
                    "--cacheinfo",
                    f"160000,{head},projects/example",
                ],
                cwd=root,
                check=True,
            )
            current = {"head": head, "tree": tree}
            verify_workspace.verify_child(root, source, current)

            empty_tree = subprocess.run(
                ["git", "mktree"],
                cwd=child,
                input="",
                text=True,
                check=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            unrelated = subprocess.run(
                ["git", "commit-tree", empty_tree, "-m", "unrelated"],
                cwd=child,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            git(child, "checkout", "--detach", unrelated)
            current.update({"head": unrelated, "tree": empty_tree})
            subprocess.run(
                [
                    "git",
                    "update-index",
                    "--cacheinfo",
                    f"160000,{unrelated},projects/example",
                ],
                cwd=root,
                check=True,
            )
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "not a baseline descendant"
            ):
                verify_workspace.verify_child(root, source, current)

    def test_authored_child_rejects_modified_or_deleted_current_license(self) -> None:
        for operation in ("modify", "delete"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                lane, current, child = absorbed_authored_fixture(root)
                license_path = child / "LICENSE"
                if operation == "modify":
                    license_path.write_text("changed license\n", encoding="utf-8")
                    git(child, "add", "LICENSE")
                else:
                    git(child, "rm", "LICENSE")
                git(child, "commit", "-m", f"{operation} license")
                current["head"] = git_output(child, "rev-parse", "HEAD")
                current["tree"] = git_output(child, "rev-parse", "HEAD^{tree}")
                subprocess.run(
                    [
                        "git",
                        "update-index",
                        "--cacheinfo",
                        f"160000,{current['head']},projects/self-amdgpu-runtime",
                    ],
                    cwd=root,
                    check=True,
                )
                with self.assertRaisesRegex(
                    verify_workspace.VerifyError, "current license hash mismatch"
                ):
                    verify_workspace.verify_authored_child(root, lane, current)

    def test_authored_child_rejects_additional_fetch_or_push_url(self) -> None:
        for key, message in (
            ("remote.origin.url", "origin URL is not a singleton"),
            ("remote.origin.pushurl", "push policy mismatch"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                lane, current, child = absorbed_authored_fixture(root)
                git(child, "config", "--add", key, "https://example.invalid/extra")
                with self.assertRaisesRegex(verify_workspace.VerifyError, message):
                    verify_workspace.verify_authored_child(root, lane, current)

    def test_authored_child_rejects_stale_nested_submodule_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lane, current, child = absorbed_authored_fixture(root)
            (child / ".gitmodules").write_text(
                '[submodule "stale"]\n'
                "\tpath = nested/stale\n"
                "\turl = ../stale.git\n",
                encoding="utf-8",
            )
            git(child, "add", ".gitmodules")
            git(child, "commit", "-m", "stale declaration")
            current["head"] = git_output(child, "rev-parse", "HEAD")
            current["tree"] = git_output(child, "rev-parse", "HEAD^{tree}")
            subprocess.run(
                [
                    "git",
                    "update-index",
                    "--cacheinfo",
                    f"160000,{current['head']},projects/self-amdgpu-runtime",
                ],
                cwd=root,
                check=True,
            )
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "unexpectedly declares nested submodules"
            ):
                verify_workspace.verify_authored_child(root, lane, current)

    def test_absorbed_admin_rejects_embedded_git_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = absorbed_child_fixture(root, absorb=False)
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "absorbed gitfile"
            ):
                verify_workspace.verify_child(root, source)

    def test_upstream_admin_rejects_symlinked_module_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = absorbed_child_fixture(root, absorb=True)
            modules = root / ".git" / "modules"
            admin = modules / "example"
            backing = modules / "example-real"
            admin.rename(backing)
            admin.symlink_to(backing.name, target_is_directory=True)
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "administrative Git path is a symlink"
            ):
                verify_workspace.verify_child(root, source)

    def test_authored_admin_rejects_symlinked_modules_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lane, current, _child = absorbed_authored_fixture(root)
            modules = root / ".git" / "modules"
            backing = root / ".git" / "modules-real"
            modules.rename(backing)
            modules.symlink_to(backing.name, target_is_directory=True)
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "administrative Git path is a symlink"
            ):
                verify_workspace.verify_authored_child(root, lane, current)

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
        current_checkpoint = {
            "id": "CP-0003",
            "root_parent_commit": "9" * 40,
            "repositories": [],
        }
        source_checkpoint = {
            "id": "CP-0002",
            "root_parent_commit": parent,
            "repositories": [],
        }
        completed = subprocess.CompletedProcess([], 0, b"accepted parent\n", b"")
        with mock.patch.object(
            verify_workspace,
            "load_json",
            side_effect=[
                lock,
                empty_project_lanes(),
                current_checkpoint,
                source_checkpoint,
            ],
        ), mock.patch.object(
            verify_workspace, "verify_checkpoint_repositories"
        ), mock.patch.object(
            verify_workspace, "verify_source_lock_history"
        ), mock.patch.object(
            verify_workspace, "verify_checkpoint_history_chain"
        ), mock.patch.object(verify_workspace, "run", return_value=completed):
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "accepted parent blob hash mismatch"
            ):
                verify_workspace.verify_sources(Path("unused"), "CP-0003")

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
        checkpoint = {
            "id": "CP-0002",
            "root_parent_commit": parent,
            "repositories": [],
        }
        completed = subprocess.CompletedProcess([], 0, parent_blob, b"")
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            verify_workspace,
            "load_json",
            side_effect=[lock, empty_project_lanes(), checkpoint, checkpoint],
        ), mock.patch.object(
            verify_workspace, "verify_checkpoint_repositories"
        ), mock.patch.object(
            verify_workspace, "verify_source_lock_history"
        ), mock.patch.object(
            verify_workspace, "verify_checkpoint_history_chain"
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

    def test_pending_transaction_reports_declared_child_before_recording(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity = transaction_child_fixture(root)
            journal = root / ".git" / "amdgpu-sim" / "txn" / "CP-0003.json"
            journal.parent.mkdir(parents=True)
            journal.write_text(
                json.dumps(
                    {
                        "schema": "amdgpu-sim.transaction.v1",
                        "checkpoint_id": "CP-0003",
                        "phase": "prepare",
                        "participants_locked": True,
                        "previous_root": identity["previous_root"],
                        "declared_children": {
                            "child": {
                                "path": identity["path"],
                                "initial_head": identity["initial_head"],
                                "initial_tree": identity["initial_tree"],
                                "target_head": identity["target_head"],
                                "target_tree": identity["target_tree"],
                            }
                        },
                        "expected_children": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(verify_workspace.PendingTransaction) as raised:
                verify_workspace.verify_root_state(root)
            message = str(raised.exception)
            self.assertIn("child child:", message)
            self.assertIn("position=target", message)
            self.assertIn("initial_identity_match=False", message)
            self.assertIn("target_identity_match=True", message)

    def test_pending_transaction_rejects_unanchored_initial_identity(self) -> None:
        for operation, expected in (
            ("missing", "lacks its initial identity pair"),
            ("tamper", "initial identity mismatch"),
            ("path", "path is invalid or duplicate"),
        ):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                identity = transaction_child_fixture(root)
                declaration = {
                    "path": identity["path"],
                    "initial_head": identity["initial_head"],
                    "initial_tree": identity["initial_tree"],
                    "target_head": identity["target_head"],
                    "target_tree": identity["target_tree"],
                }
                if operation == "missing":
                    declaration.pop("initial_tree")
                elif operation == "tamper":
                    declaration["initial_tree"] = "0" * 40
                else:
                    declaration["path"] = "projects/nested/child"
                journal = {
                    "previous_root": identity["previous_root"],
                    "declared_children": {"child": declaration},
                }
                with self.assertRaisesRegex(verify_workspace.VerifyError, expected):
                    verify_workspace.verify_journal_initial_children(root, journal)

    def test_allow_transaction_rejects_initial_identity_not_in_previous_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity = transaction_child_fixture(root)
            git(root, "add", identity["path"])
            git(root, "commit", "-m", "coordinator")
            coordinator_tree = git_output(root, "rev-parse", "HEAD^{tree}")
            journal = root / ".git" / "amdgpu-sim" / "txn" / "CP-0003.json"
            journal.parent.mkdir(parents=True)
            journal.write_text(
                json.dumps(
                    {
                        "schema": "amdgpu-sim.transaction.v1",
                        "checkpoint_id": "CP-0003",
                        "phase": "prepared",
                        "participants_locked": True,
                        "previous_root": identity["previous_root"],
                        "expected_root_tree": coordinator_tree,
                        "declared_children": {
                            "child": {
                                "path": identity["path"],
                                "initial_head": "0" * 40,
                                "initial_tree": identity["initial_tree"],
                                "target_head": identity["target_head"],
                                "target_tree": identity["target_tree"],
                            }
                        },
                        "expected_children": {
                            "child": {
                                "path": identity["path"],
                                "head": identity["target_head"],
                                "tree": identity["target_tree"],
                            }
                        },
                        "root_allowlist": [identity["path"]],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                verify_workspace.VerifyError, "initial identity mismatch"
            ):
                verify_workspace.verify_root_state(root, allow_transaction="CP-0003")

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
