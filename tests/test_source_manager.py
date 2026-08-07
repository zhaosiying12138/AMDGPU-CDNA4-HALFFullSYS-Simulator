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
        f"\turl = https://example.invalid/{source_id}.git\n",
        encoding="utf-8",
    )
    git("add", ".gitmodules", cwd=root)
    git("update-index", "--add", "--cacheinfo", f"160000,{commit},{relative}", cwd=root)
    source = {
        "id": source_id,
        "path": relative,
        "materialization": "pending",
        "observed_head": commit,
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
