# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import importlib.util
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("source_manager", ROOT / "scripts/source_manager.py")
assert SPEC and SPEC.loader
source_manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source_manager)


def git(*argv: str, cwd: Path, capture: bool = False) -> str:
    completed = subprocess.run(
        ["git", *argv],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
    )
    return (completed.stdout or "").strip()


def absorption_fixture(root: Path, source_id: str = "example") -> tuple[dict[str, object], Path, str]:
    root.mkdir()
    git("init", "-b", "main", cwd=root)
    git("config", "user.name", "test", cwd=root)
    git("config", "user.email", "test@example.invalid", cwd=root)
    child = root / "projects" / source_id
    child.mkdir(parents=True)
    git("init", "-b", "main", cwd=child)
    git("config", "user.name", "test", cwd=child)
    git("config", "user.email", "test@example.invalid", cwd=child)
    (child / "payload").write_text("source\n", encoding="utf-8")
    git("add", "payload", cwd=child)
    git("commit", "-m", "source", cwd=child)
    commit = git("rev-parse", "HEAD", cwd=child, capture=True)
    relative = f"projects/{source_id}"
    (root / ".gitmodules").write_text(
        f'[submodule "{source_id}"]\n'
        f"\tpath = {relative}\n"
        f"\turl = https://example.invalid/{source_id}.git\n"
        "\tbranch = main\n",
        encoding="utf-8",
    )
    git("add", ".gitmodules", cwd=root)
    git("update-index", "--add", "--cacheinfo", f"160000,{commit},{relative}", cwd=root)
    source = {
        "id": source_id,
        "path": relative,
        "materialization": "pending",
        "observed_head": commit,
        "upstream_url": f"https://example.invalid/{source_id}.git",
        "upstream_ref": "main",
    }
    lock: dict[str, object] = {
        "schema": "amdgpu-sim.source-lock.v1",
        "status": "observed-not-fetched",
        "sources": [
            source,
            {
                "id": "model",
                "path": "models/model",
                "materialization": "external-download",
            },
        ],
    }
    return lock, child, commit


def project_lanes_fixture(source_id: str, commit: str) -> dict[str, object]:
    return {
        "schema": "amdgpu-sim.project-lanes.v1",
        "registry_revision": 1,
        "url_policy": {
            "scheme": "sibling-relative-v1",
            "template": "../{project_id}.git",
        },
        "lanes": [
            {
                "id": source_id,
                "ownership": "project-authored",
                "role": "host-runtime",
                "path": f"projects/{source_id}",
                "materialization": "gitlink",
                "origin": {
                    "policy": "sibling-relative-v1",
                    "remote": "origin",
                    "url": f"../{source_id}.git",
                    "push_url": "no_push",
                    "reachability": "not-asserted",
                    "branch": "main",
                },
                "baseline_commit": commit,
                "baseline_tree": "2" * 40,
                "baseline_tag": f"project-baseline/{source_id}/{commit}",
                "baseline_tag_object": "3" * 40,
                "baseline_tag_payload": "fixture tag payload\n",
                "baseline_tag_payload_sha256": hashlib.sha256(
                    b"fixture tag payload\n"
                ).hexdigest(),
                "baseline_created_at": "2026-08-08T03:00:00+08:00",
                "baseline_checkpoint_id": "CP-0003",
                "baseline_evidence_id": "EV-0004",
                "baseline_commit_trailers": {
                    "checkpoint_id": "CP-0003",
                    "goal_id": "GSIM-001",
                    "plan_revision": 1,
                    "source_lock_sha256": "4" * 64,
                    "evidence_manifest_sha256": "5" * 64,
                    "change_kind": "baseline",
                    "baseline_commit_marker": "N/A",
                },
                "administrative_git_dir": f"modules/{source_id}",
                "license": {
                    "spdx_id": "GPL-3.0-or-later",
                    "path": "LICENSE",
                    "sha256": "6" * 64,
                },
            }
        ],
    }


class SourceManagerTest(unittest.TestCase):
    def test_absorb_moves_embedded_repository_to_canonical_root_git_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            lock, child, commit = absorption_fixture(root)
            with mock.patch.object(source_manager, "ROOT", root):
                result = source_manager.absorb_sources(lock)
                repeated = source_manager.absorb_sources(lock)
            expected_admin = root / ".git" / "modules" / "example"
            self.assertEqual(result[0]["administrative_git_dir"], "modules/example")
            self.assertEqual(repeated, result)
            self.assertEqual(
                (child / ".git").read_text(encoding="utf-8"),
                f"gitdir: {os.path.relpath(expected_admin, child)}\n",
            )
            self.assertEqual(
                Path(git("rev-parse", "--absolute-git-dir", cwd=child, capture=True)),
                expected_admin,
            )
            self.assertEqual(git("rev-parse", "HEAD", cwd=child, capture=True), commit)

    def test_absorb_rejects_wrong_root_gitlink_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            lock, child, _commit = absorption_fixture(root)
            git(
                "update-index",
                "--cacheinfo",
                f"160000,{'f' * 40},projects/example",
                cwd=root,
            )
            with mock.patch.object(source_manager, "ROOT", root):
                with self.assertRaisesRegex(source_manager.SourceError, "gitlink mismatch"):
                    source_manager.absorb_sources(lock)
            self.assertTrue((child / ".git").is_dir())

    def test_absorb_rejects_dirty_child_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            lock, child, _commit = absorption_fixture(root)
            (child / "untracked").write_text("dirty\n", encoding="utf-8")
            with mock.patch.object(source_manager, "ROOT", root):
                with self.assertRaisesRegex(source_manager.SourceError, "dirty source"):
                    source_manager.absorb_sources(lock)
            self.assertTrue((child / ".git").is_dir())

    def test_absorb_rejects_noncanonical_gitmodules_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            lock, child, _commit = absorption_fixture(root)
            modules = root / ".gitmodules"
            modules.write_text(
                '[submodule "alias"]\n'
                "\tpath = projects/example\n"
                "\turl = https://example.invalid/example.git\n",
                encoding="utf-8",
            )
            git("add", ".gitmodules", cwd=root)
            with mock.patch.object(source_manager, "ROOT", root):
                with self.assertRaisesRegex(source_manager.SourceError, "canonical name/path"):
                    source_manager.absorb_sources(lock)
            self.assertTrue((child / ".git").is_dir())

    def test_absorb_rejects_upstream_gitmodules_url_or_branch_drift(self) -> None:
        for key, value in (
            ("url", "https://example.invalid/other.git"),
            ("branch", "other"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "root"
                lock, child, _commit = absorption_fixture(root)
                git(
                    "config",
                    "-f",
                    ".gitmodules",
                    f"submodule.example.{key}",
                    value,
                    cwd=root,
                )
                git("add", ".gitmodules", cwd=root)
                with mock.patch.object(source_manager, "ROOT", root):
                    with self.assertRaisesRegex(
                        source_manager.SourceError, f"{key} mismatch"
                    ):
                        source_manager.absorb_sources(lock)
                self.assertTrue((child / ".git").is_dir())

    def test_absorb_rejects_unknown_gitmodules_key_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            lock, child, _commit = absorption_fixture(root)
            with (root / ".gitmodules").open("a", encoding="utf-8") as stream:
                stream.write("[include]\n\tpath = extra.conf\n")
            git("add", ".gitmodules", cwd=root)
            with mock.patch.object(source_manager, "ROOT", root):
                with self.assertRaisesRegex(source_manager.SourceError, "unexpected keys"):
                    source_manager.absorb_sources(lock)
            self.assertTrue((child / ".git").is_dir())

    def test_absorbed_layout_verifier_rejects_embedded_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            lock, child, _commit = absorption_fixture(root)
            source = lock["sources"][0]
            with mock.patch.object(source_manager, "ROOT", root):
                with self.assertRaisesRegex(source_manager.SourceError, "embedded"):
                    source_manager.verify_absorbed_source(source)
            self.assertTrue((child / ".git").is_dir())

    def test_absorb_rejects_unsafe_source_id_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            lock, child, _commit = absorption_fixture(root)
            lock["sources"][0]["id"] = "../example"
            with mock.patch.object(source_manager, "ROOT", root):
                with self.assertRaisesRegex(source_manager.SourceError, "unsafe submodule name"):
                    source_manager.absorb_sources(lock)
            self.assertTrue((child / ".git").is_dir())

    def test_frozen_absorb_revision_requires_matching_baseline_and_work_head(self) -> None:
        source = {
            "id": "example",
            "path": "projects/example",
            "materialization": "gitlink",
            "baseline_commit": "a" * 40,
            "work_head": "b" * 40,
        }
        lock = {
            "schema": "amdgpu-sim.source-lock.v1",
            "status": "frozen",
            "sources": [source],
        }
        with self.assertRaisesRegex(source_manager.SourceError, "baseline/work head mismatch"):
            source_manager.absorption_sources(lock)
        source["work_head"] = "a" * 40
        self.assertEqual(source_manager.absorption_sources(lock)[0][1], "a" * 40)

    def test_absorb_accepts_frozen_upstream_descendant_gitlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            lock, child, baseline = absorption_fixture(root)
            source = lock["sources"][0]
            lock["status"] = "frozen"
            source.update(
                {
                    "materialization": "gitlink",
                    "baseline_commit": baseline,
                    "work_head": baseline,
                }
            )
            (child / "payload").write_text("descendant\n", encoding="utf-8")
            git("add", "payload", cwd=child)
            git("commit", "-m", "descendant", cwd=child)
            descendant = git("rev-parse", "HEAD", cwd=child, capture=True)
            git(
                "update-index",
                "--cacheinfo",
                f"160000,{descendant},projects/example",
                cwd=root,
            )
            with mock.patch.object(source_manager, "ROOT", root):
                result = source_manager.absorb_sources(lock)
            self.assertEqual(result[0]["id"], "example")
            self.assertEqual(git("rev-parse", "HEAD", cwd=child, capture=True), descendant)

    def test_absorb_rejects_non_descendant_upstream_gitlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            lock, child, baseline = absorption_fixture(root)
            source = lock["sources"][0]
            lock["status"] = "frozen"
            source.update(
                {
                    "materialization": "gitlink",
                    "baseline_commit": baseline,
                    "work_head": baseline,
                }
            )
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
            git("checkout", "--detach", unrelated, cwd=child)
            git(
                "update-index",
                "--cacheinfo",
                f"160000,{unrelated},projects/example",
                cwd=root,
            )
            with mock.patch.object(source_manager, "ROOT", root):
                with self.assertRaisesRegex(
                    source_manager.SourceError, "not a baseline descendant"
                ):
                    source_manager.absorb_sources(lock)
            self.assertTrue((child / ".git").is_dir())

    def test_absorb_supports_project_authored_lane(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            lock, child, baseline = absorption_fixture(
                root, source_id="self-amdgpu-runtime"
            )
            lock["sources"] = [lock["sources"][1]]
            lock_bytes = (json.dumps(lock, sort_keys=True) + "\n").encode()
            (root / "SOURCE_LOCK.json").write_bytes(lock_bytes)
            lock_sha = hashlib.sha256(lock_bytes).hexdigest()
            (child / "LICENSE").write_text("fixture license\n", encoding="utf-8")
            git("add", "LICENSE", cwd=child)
            baseline_message = (
                "baseline\n\n"
                "Checkpoint-ID: CP-0003\n"
                "Goal-ID: GSIM-001\n"
                "Plan-Revision: 1\n"
                f"Source-Lock-SHA256: {lock_sha}\n"
                f"Evidence-Manifest-SHA256: {'5' * 64}\n"
                "Change-Kind: baseline\n"
                "Baseline-Commit: N/A"
            )
            git("commit", "--amend", "-m", baseline_message, cwd=child)
            baseline = git("rev-parse", "HEAD", cwd=child, capture=True)
            tree = git("rev-parse", "HEAD^{tree}", cwd=child, capture=True)
            tag = f"project-baseline/self-amdgpu-runtime/{baseline}"
            tag_message = (
                "fixture baseline\n\n"
                "Project-ID: self-amdgpu-runtime\n"
                "Project-URL: ../self-amdgpu-runtime.git\n"
                "Origin-Policy: sibling-relative-v1\n"
                f"Baseline-Commit: {baseline}\n"
                f"Baseline-Tree: {tree}\n"
                "License-SPDX: GPL-3.0-or-later\n"
                "Created-At: 2026-08-08T03:00:00+08:00"
            )
            git("tag", "-a", tag, "-m", tag_message, cwd=child)
            tag_object = git("rev-parse", tag, cwd=child, capture=True)
            tag_payload = subprocess.check_output(
                ["git", "cat-file", "tag", tag], cwd=child, text=True
            )
            git("remote", "add", "origin", "../self-amdgpu-runtime.git", cwd=child)
            git("remote", "set-url", "--push", "origin", "no_push", cwd=child)
            registry = project_lanes_fixture("self-amdgpu-runtime", baseline)
            lane = registry["lanes"][0]
            lane.update(
                {
                    "baseline_tree": tree,
                    "baseline_tag": tag,
                    "baseline_tag_object": tag_object,
                    "baseline_tag_payload": tag_payload,
                    "baseline_tag_payload_sha256": hashlib.sha256(
                        tag_payload.encode()
                    ).hexdigest(),
                    "license": {
                        "spdx_id": "GPL-3.0-or-later",
                        "path": "LICENSE",
                        "sha256": hashlib.sha256(b"fixture license\n").hexdigest(),
                    },
                }
            )
            lane["baseline_commit_trailers"]["source_lock_sha256"] = lock_sha
            modules = root / ".gitmodules"
            modules.write_text(
                '[submodule "self-amdgpu-runtime"]\n'
                "\tpath = projects/self-amdgpu-runtime\n"
                "\turl = ../self-amdgpu-runtime.git\n"
                "\tbranch = main\n",
                encoding="utf-8",
            )
            git("add", ".gitmodules", cwd=root)
            git(
                "update-index",
                "--cacheinfo",
                f"160000,{baseline},projects/self-amdgpu-runtime",
                cwd=root,
            )
            with mock.patch.object(source_manager, "ROOT", root):
                result = source_manager.absorb_sources(lock, registry)
                repeated = source_manager.absorb_sources(lock, registry)
            self.assertEqual(result, repeated)
            self.assertEqual(result[0]["id"], "self-amdgpu-runtime")
            self.assertTrue((child / ".git").is_file())
            for field, mutation, message in (
                (
                    "tree",
                    lambda value: value.__setitem__("baseline_tree", "f" * 40),
                    "baseline tree mismatch",
                ),
                (
                    "tag",
                    lambda value: value.__setitem__("baseline_tag_object", "f" * 40),
                    "tag object mismatch",
                ),
                (
                    "license",
                    lambda value: value["license"].__setitem__("sha256", "f" * 64),
                    "license mismatch",
                ),
            ):
                with self.subTest(field=field):
                    tampered = json.loads(json.dumps(registry))
                    mutation(tampered["lanes"][0])
                    with mock.patch.object(source_manager, "ROOT", root):
                        with self.assertRaisesRegex(source_manager.SourceError, message):
                            source_manager.absorb_sources(lock, tampered)

    def test_authored_registry_collision_with_upstream_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            lock, _child, baseline = absorption_fixture(root)
            registry = project_lanes_fixture("example", baseline)
            with self.assertRaisesRegex(
                source_manager.SourceError, "across upstream/authored lanes"
            ):
                source_manager.absorption_sources(lock, registry)

    def test_authored_absorption_registry_rejects_unknown_fields(self) -> None:
        registry = project_lanes_fixture("self-amdgpu-runtime", "1" * 40)
        registry["lanes"][0]["unexpected"] = True
        with self.assertRaisesRegex(
            source_manager.SourceError, "unknown fields"
        ):
            source_manager.authored_absorption_sources(registry)

    def test_materializer_recreates_compatibility_revision_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            origin = root / "origin"
            origin.mkdir()
            subprocess.run(["git", "init", "-b", "main"], cwd=origin, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "config", "user.name", "test"], cwd=origin, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=origin, check=True)
            (origin / "base").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "base"], cwd=origin, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=origin, check=True, stdout=subprocess.DEVNULL)
            base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=origin, text=True).strip()
            (origin / "main").write_text("main\n", encoding="utf-8")
            subprocess.run(["git", "add", "main"], cwd=origin, check=True)
            subprocess.run(["git", "commit", "-m", "main"], cwd=origin, check=True, stdout=subprocess.DEVNULL)
            baseline = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=origin, text=True).strip()
            subprocess.run(["git", "checkout", "-b", "compat", base], cwd=origin, check=True, stdout=subprocess.DEVNULL)
            (origin / "compat").write_text("compat\n", encoding="utf-8")
            subprocess.run(["git", "add", "compat"], cwd=origin, check=True)
            subprocess.run(["git", "commit", "-m", "compat"], cwd=origin, check=True, stdout=subprocess.DEVNULL)
            compatibility = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=origin, text=True).strip()
            subprocess.run(["git", "checkout", "main"], cwd=origin, check=True, stdout=subprocess.DEVNULL)
            source = {
                "id": "example",
                "path": "projects/example",
                "upstream_url": str(origin),
                "transport_url": str(origin),
                "work_branch": "amdgpu-sim/example",
                "compatibility_revisions": [
                    {
                        "purpose": "fixture",
                        "required_by_source": "consumer",
                        "required_by_commit": baseline,
                        "commit": compatibility,
                        "tag": f"compatibility/fixture/{compatibility}",
                    }
                ],
            }
            resolved = {
                "commit": baseline,
                "default_branch": "main",
                "resolved_at": "2026-08-07T00:00:00Z",
            }
            identity = {
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "test@example.invalid",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "test@example.invalid",
            }
            with mock.patch.object(source_manager, "ROOT", root), mock.patch.dict(
                os.environ, identity
            ):
                result = source_manager.materialize_one(
                    source,
                    resolved,
                    archive_hydration=False,
                    history_chunk=1,
                    blob_batch_size=1,
                )
            repo = root / "projects" / "example"
            self.assertEqual(result["compatibility_revisions"][0]["commit"], compatibility)
            self.assertEqual(
                subprocess.check_output(
                    ["git", "rev-list", "-n", "1", f"compatibility/fixture/{compatibility}"],
                    cwd=repo,
                    text=True,
                ).strip(),
                compatibility,
            )
            self.assertEqual(source_manager.missing_object_ids(repo, compatibility), [])

    def test_failed_pack_quarantine_preserves_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            subprocess.run(
                ["git", "init", "-b", "main", str(repo)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            git_dir = Path(
                subprocess.check_output(
                    ["git", "rev-parse", "--absolute-git-dir"], cwd=repo, text=True
                ).strip()
            )
            temporary = git_dir / "objects" / "pack" / "tmp_pack_fixture"
            temporary.write_bytes(b"partial-pack")
            moved = source_manager.quarantine_temporary_packs(repo, "test")
            self.assertEqual(len(moved), 1)
            self.assertFalse(temporary.exists())
            self.assertEqual(Path(moved[0]).read_bytes(), b"partial-pack")

    def test_frozen_lock_materialization_does_not_require_resolution_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lock_path = root / "SOURCE_LOCK.json"
            lock = {
                "status": "frozen",
                "frozen_at": "2026-08-07T00:00:00Z",
                "sources": [
                    {
                        "id": "example",
                        "materialization": "gitlink",
                        "baseline_commit": "a" * 40,
                        "upstream_ref": "main",
                    }
                ],
            }
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            resolved, digest, source = source_manager.materialization_resolution(
                lock, root / "absent-resolution.json", lock_path=lock_path
            )
            self.assertEqual(resolved["example"]["commit"], "a" * 40)
            self.assertEqual(digest, hashlib.sha256(lock_path.read_bytes()).hexdigest())
            self.assertEqual(source, "frozen-source-lock")

    def test_refetch_hydration_forces_missing_blob_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            origin = root / "origin"
            subprocess.run(
                ["git", "init", "-b", "main", str(origin)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(["git", "config", "user.name", "test"], cwd=origin, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=origin,
                check=True,
            )
            (origin / "payload").write_text("blob\n", encoding="utf-8")
            subprocess.run(["git", "add", "payload"], cwd=origin, check=True)
            subprocess.run(
                ["git", "commit", "-m", "source"],
                cwd=origin,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=origin, text=True
            ).strip()
            tree = subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=origin, text=True
            ).strip()
            client = root / "client"
            subprocess.run(
                ["git", "init", "-b", "scratch", str(client)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(
                ["git", "remote", "add", "upstream", str(origin)],
                cwd=client,
                check=True,
            )
            for kind, object_id in (("commit", commit), ("tree", tree)):
                payload = subprocess.check_output(
                    ["git", "cat-file", kind, object_id], cwd=origin
                )
                subprocess.run(
                    ["git", "hash-object", "-t", kind, "-w", "--stdin"],
                    cwd=client,
                    input=payload,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            self.assertEqual(len(source_manager.missing_object_ids(client, commit)), 1)
            result = source_manager.hydrate_missing_objects(
                client, str(origin), commit, batch_size=1
            )
            self.assertEqual(result["hydrated_object_count"], 1)
            self.assertEqual(source_manager.missing_object_ids(client, commit), [])

    def test_hydration_starts_from_a_missing_root_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            origin = root / "origin"
            subprocess.run(
                ["git", "init", "-b", "main", str(origin)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(["git", "config", "user.name", "test"], cwd=origin, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=origin,
                check=True,
            )
            nested = origin / "one" / "two"
            nested.mkdir(parents=True)
            (nested / "payload").write_text("nested\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=origin, check=True)
            subprocess.run(
                ["git", "commit", "-m", "nested"],
                cwd=origin,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=origin, text=True
            ).strip()
            client = root / "client"
            subprocess.run(
                ["git", "init", "-b", "scratch", str(client)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(
                ["git", "remote", "add", "upstream", str(origin)],
                cwd=client,
                check=True,
            )
            subprocess.run(
                ["git", "config", "remote.upstream.promisor", "true"],
                cwd=client,
                check=True,
            )
            subprocess.run(
                ["git", "config", "remote.upstream.partialclonefilter", "tree:0"],
                cwd=client,
                check=True,
            )
            commit_payload = subprocess.check_output(
                ["git", "cat-file", "commit", commit], cwd=origin
            )
            subprocess.run(
                ["git", "hash-object", "-t", "commit", "-w", "--stdin"],
                cwd=client,
                input=commit_payload,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            self.assertEqual(len(source_manager.missing_object_ids(client, commit)), 1)
            result = source_manager.hydrate_missing_objects(
                client, str(origin), commit, batch_size=1
            )
            self.assertGreaterEqual(result["hydration_dependency_waves"], 1)
            self.assertGreaterEqual(result["hydrated_object_count"], 1)
            self.assertEqual(source_manager.missing_object_ids(client, commit), [])

    def test_hydration_repeats_when_new_dependencies_are_discovered(self) -> None:
        root_tree = "1" * 40
        nested_tree = "2" * 40
        blob = "3" * 40

        def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            if argv[:4] == ["git", "remote", "get-url", "upstream"]:
                return subprocess.CompletedProcess(argv, 0, "transport\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        with mock.patch.object(
            source_manager,
            "missing_object_ids",
            side_effect=[[root_tree], [nested_tree, blob], []],
        ), mock.patch.object(source_manager, "run", side_effect=fake_run), mock.patch.object(
            source_manager, "require_local_objects"
        ):
            result = source_manager.hydrate_missing_objects(
                Path("unused"), "transport", "a" * 40, batch_size=1
            )
        self.assertEqual(result["hydration_dependency_waves"], 2)
        self.assertEqual(result["hydrated_object_count"], 3)

    def test_resilient_history_fetch_deepens_to_complete_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            origin = root / "origin"
            subprocess.run(
                ["git", "init", "-b", "main", str(origin)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(["git", "config", "user.name", "test"], cwd=origin, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=origin,
                check=True,
            )
            for value in range(5):
                (origin / "value").write_text(f"{value}\n", encoding="utf-8")
                subprocess.run(["git", "add", "value"], cwd=origin, check=True)
                subprocess.run(
                    ["git", "commit", "-m", str(value)],
                    cwd=origin,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=origin, text=True
            ).strip()
            (origin / "value").write_text("upstream moved\n", encoding="utf-8")
            subprocess.run(["git", "add", "value"], cwd=origin, check=True)
            subprocess.run(
                ["git", "commit", "-m", "upstream moved"],
                cwd=origin,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            client = root / "client"
            subprocess.run(
                ["git", "init", "-b", "scratch", str(client)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(
                ["git", "remote", "add", "upstream", str(origin)],
                cwd=client,
                check=True,
            )
            source_manager.resilient_history_fetch(
                client, str(origin), "main", commit, deepen_by=2
            )
            self.assertEqual(
                subprocess.check_output(
                    ["git", "rev-parse", "--is-shallow-repository"],
                    cwd=client,
                    text=True,
                ).strip(),
                "false",
            )
            self.assertEqual(
                subprocess.check_output(
                    ["git", "rev-list", "--count", commit], cwd=client, text=True
                ).strip(),
                "5",
            )
            self.assertEqual(
                subprocess.check_output(
                    ["git", "rev-parse", "refs/amdgpu-sim/resolved/main"],
                    cwd=client,
                    text=True,
                ).strip(),
                commit,
            )

    def test_missing_object_parser_rejects_invalid_oid(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "?not-an-object\n", "")
        with mock.patch.object(source_manager, "run", return_value=completed):
            with self.assertRaises(source_manager.SourceError):
                source_manager.missing_object_ids(Path("unused"), "a" * 40)

    def test_archive_filter_rejects_escaping_symlink(self) -> None:
        member = tarfile.TarInfo("source/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        with self.assertRaises(source_manager.SourceError):
            source_manager.safe_tar_member(member, "unused")

    def test_safe_path_rejects_escape(self) -> None:
        with self.assertRaises(source_manager.SourceError):
            source_manager.safe_workspace_path("../outside")

    def test_github_codeload_url_is_fixed_to_commit(self) -> None:
        commit = "1" * 40
        self.assertEqual(
            source_manager.github_codeload_url(
                "https://github.com/example/project.git", commit
            ),
            f"https://codeload.github.com/example/project/tar.gz/{commit}",
        )

    def test_github_codeload_url_rejects_non_github_source(self) -> None:
        with self.assertRaises(source_manager.SourceError):
            source_manager.github_codeload_url(
                "https://example.com/project.git", "1" * 40
            )

    def test_archive_hydration_rebuilds_exact_tree_without_touching_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            origin = root / "origin"
            origin.mkdir()
            subprocess.run(["git", "init", "-b", "main"], cwd=origin, check=True, stdout=subprocess.DEVNULL)
            (origin / "source.txt").write_text("immutable\n", encoding="utf-8")
            subprocess.run(["git", "add", "source.txt"], cwd=origin, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=test",
                    "-c",
                    "user.email=test@invalid",
                    "commit",
                    "-m",
                    "baseline",
                ],
                cwd=origin,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=origin, text=True
            ).strip()
            tree = subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=origin, text=True
            ).strip()
            archive = root / "archive.tar.gz"
            subprocess.run(
                [
                    "git",
                    "archive",
                    "--format=tar.gz",
                    f"--prefix=project-{commit}/",
                    f"--output={archive}",
                    commit,
                ],
                cwd=origin,
                check=True,
            )
            destination = root / "destination"
            subprocess.run(
                ["git", "clone", "--no-checkout", str(origin), str(destination)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            status_before = subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=destination, text=True
            )

            def provide_archive(_url: str, output: Path) -> str:
                output.write_bytes(archive.read_bytes())
                return "f" * 64

            with mock.patch.object(source_manager, "download_file", side_effect=provide_archive):
                result = source_manager.hydrate_from_github_archive(
                    destination,
                    "https://github.com/example/project.git",
                    commit,
                    tree,
                )
            self.assertEqual(result["archive_sha256"], "f" * 64)
            self.assertEqual(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=destination, text=True
                ),
                status_before,
            )

    def test_resolve_git_uses_explicit_transport_and_canonical_url(self) -> None:
        output = (
            "ref: refs/heads/main\tHEAD\n"
            "0123456789abcdef0123456789abcdef01234567\tHEAD\n"
        )
        completed = subprocess.CompletedProcess([], 0, output, "")
        item = {
            "id": "example",
            "upstream_url": "https://github.com/example/project.git",
            "transport_url": "git@github.com:example/project.git",
            "upstream_ref": "main",
            "observed_head": "f" * 40,
        }
        with mock.patch.object(source_manager, "run", return_value=completed) as mocked:
            result = source_manager.resolve_git(item)
        self.assertEqual(result["url"], item["upstream_url"])
        self.assertEqual(result["transport_url"], item["transport_url"])
        self.assertTrue(result["observation_changed"])
        mocked.assert_called_once_with(
            ["git", "ls-remote", "--symref", item["transport_url"], "HEAD"]
        )

    def test_resolve_git_rejects_default_branch_drift(self) -> None:
        output = "ref: refs/heads/develop\tHEAD\n" + "a" * 40 + "\tHEAD\n"
        completed = subprocess.CompletedProcess([], 0, output, "")
        item = {
            "id": "example",
            "upstream_url": "https://github.com/example/project.git",
            "upstream_ref": "main",
        }
        with mock.patch.object(source_manager, "run", return_value=completed):
            with self.assertRaises(source_manager.SourceError):
                source_manager.resolve_git(item)

    def test_external_command_timeout_is_reported(self) -> None:
        with self.assertRaises(source_manager.SourceError) as caught:
            source_manager.run(
                [sys.executable, "-c", "import time; time.sleep(1)"], timeout=0.01
            )
        self.assertIn("timed out", str(caught.exception))

    def test_frozen_annotated_tag_object_is_recreated_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
            (repo / "value").write_text("x\n", encoding="utf-8")
            subprocess.run(["git", "add", "value"], cwd=repo, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=test",
                    "-c",
                    "user.email=test@invalid",
                    "commit",
                    "-m",
                    "source",
                ],
                cwd=repo,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            payload = (
                f"object {commit}\ntype commit\ntag reproducible\n"
                "tagger test <test@invalid> 1786128000 +0800\n\nBaseline\n"
            )
            expected = subprocess.check_output(
                ["git", "hash-object", "-t", "tag", "-w", "--stdin"],
                cwd=repo,
                text=True,
                input=payload,
            ).strip()
            source_manager.install_frozen_tag(repo, "reproducible", payload, expected)
            self.assertEqual(
                subprocess.check_output(
                    ["git", "rev-parse", "refs/tags/reproducible"], cwd=repo, text=True
                ).strip(),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
