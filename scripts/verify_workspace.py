#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline, read-only verification for an amdgpu-sim handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_commit_message import MessageError, check as check_commit_message  # noqa: E402
from evidence_policy import EvidencePolicyError, validate_evidence  # noqa: E402


CP_RE = re.compile(r"^CP-[0-9]{4}$")
EV_RE = re.compile(r"^EV-[0-9]{4}$")
BL_RE = re.compile(r"^BL-[0-9]{4}$")
LOCK_ID_RE = re.compile(r"^SL-[0-9]{4}$")
SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
FORBIDDEN_PREFIXES = (
    "models/",
    "env/",
    ".venv/",
    "build/",
    "artifacts/",
    "logs/",
    "cache/",
    "downloads/",
    "runs/",
    "tmp/",
)
FORBIDDEN_SUFFIX_RE = re.compile(
    r"(?:\.safetensors(?:\.index\.json)?|\.pt|\.pth|\.bin|\.onnx|\.ckpt|"
    r"\.gguf|\.npy|\.npz|\.o|\.a|\.so(?:\..*)?)$",
    re.IGNORECASE,
)
SECRET_PATTERNS = (
    ("private-key", re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")),
    ("aws-access-key", re.compile(rb"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("huggingface-token", re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b")),
    ("openai-token", re.compile(rb"\bsk-[A-Za-z0-9_-]{32,}\b")),
)
PLACEHOLDERS = {f"{prefix}.gitkeep" for prefix in FORBIDDEN_PREFIXES}


def generated_path(path: str) -> bool:
    first = path.split("/", 1)[0]
    return path.startswith(FORBIDDEN_PREFIXES) or first == "build" or first.startswith("build-")


class VerifyError(RuntimeError):
    pass


class PendingTransaction(VerifyError):
    """A recoverable two-phase transaction requires operator finalization."""


def run(
    root: Path,
    argv: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    proc = subprocess.run(
        argv,
        cwd=cwd or root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1"},
    )
    if check and proc.returncode:
        detail = (proc.stderr or proc.stdout).decode(errors="replace").strip()
        raise VerifyError(f"command failed ({proc.returncode}): {argv!r}: {detail}")
    return proc


def text(root: Path, argv: list[str], *, cwd: Path | None = None) -> str:
    return run(root, argv, cwd=cwd).stdout.decode().strip()


def run_discard_stdout(
    root: Path, argv: list[str], *, cwd: Path | None = None
) -> None:
    proc = subprocess.run(
        argv,
        cwd=cwd or root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1"},
    )
    if proc.returncode:
        detail = proc.stderr.decode(errors="replace").strip()
        raise VerifyError(f"command failed ({proc.returncode}): {argv!r}: {detail}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerifyError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerifyError(f"JSON root is not an object: {path}")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_manifest_digest(source: dict[str, Any]) -> str:
    files = []
    for item in source.get("files", []):
        entry: dict[str, Any] = {
            "path": item["path"],
            "blobId": item["blob_id"],
            "size": item["size"],
        }
        if isinstance(item.get("lfs"), dict):
            lfs = item["lfs"]
            entry["lfs"] = {
                "sha256": lfs["sha256"],
                "size": lfs["size"],
                "pointerSize": lfs["pointer_size"],
            }
        files.append(entry)
    files.sort(key=lambda item: item["path"])
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_trailers(message: str) -> dict[str, list[str]]:
    trailers: dict[str, list[str]] = {}
    for line in message.splitlines():
        match = re.fullmatch(r"([A-Za-z0-9-]+): (.+)", line)
        if match:
            trailers.setdefault(match.group(1), []).append(match.group(2))
    return trailers


def safe_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts:
        raise VerifyError(f"unsafe relative path: {relative!r}")
    resolved = (root / posix).resolve()
    if resolved != root and root not in resolved.parents:
        raise VerifyError(f"path escapes workspace: {relative!r}")
    return resolved


def verify_root_state(
    root: Path, *, allow_transaction: str | None = None
) -> tuple[str, str]:
    run(root, ["git", "rev-parse", "--git-dir"])
    git_dir = Path(text(root, ["git", "rev-parse", "--absolute-git-dir"]))
    for marker in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "BISECT_LOG", "rebase-merge", "rebase-apply"):
        if (git_dir / marker).exists():
            raise VerifyError(f"git operation is in progress: {marker}")
    journals = sorted((git_dir / "amdgpu-sim" / "txn").glob("*.json"))
    if journals:
        if allow_transaction is not None:
            if len(journals) != 1 or journals[0].stem != allow_transaction:
                raise VerifyError("allowed transaction does not match the active journal")
            allowed = load_json(journals[0])
            if allowed.get("checkpoint_id") != allow_transaction:
                raise VerifyError("active journal Checkpoint-ID mismatch")
            phase = allowed.get("phase")
            if phase not in {"prepared", "committed"}:
                raise VerifyError(
                    "post-commit verification requires a prepared or committed transaction"
                )
            declared = allowed.get("declared_children")
            recorded = allowed.get("expected_children")
            if (
                allowed.get("participants_locked") is not True
                or not isinstance(declared, dict)
                or not declared
                or not isinstance(recorded, dict)
                or set(declared) != set(recorded)
                or not isinstance(allowed.get("root_allowlist"), list)
                or not allowed["root_allowlist"]
            ):
                raise VerifyError("prepared transaction participant/allowlist set is incomplete")
            for child_id, child in recorded.items():
                declaration = declared[child_id]
                if (
                    declaration.get("path") != child.get("path")
                    or declaration.get("target_head") not in (None, child.get("head"))
                    or declaration.get("target_tree") not in (None, child.get("tree"))
                ):
                    raise VerifyError(f"prepared transaction child identity mismatch: {child_id}")
            coordinator_head = text(root, ["git", "rev-parse", "HEAD"])
            coordinator_tree = text(root, ["git", "rev-parse", "HEAD^{tree}"])
            coordinator_parent = text(root, ["git", "rev-parse", "HEAD^"])
            if (
                allowed.get("expected_root_tree") != coordinator_tree
                or allowed.get("previous_root") != coordinator_parent
            ):
                raise VerifyError("transaction journal does not bind coordinator HEAD")
            if phase == "committed" and (
                allowed.get("root_coordinator_commit") != coordinator_head
                or not isinstance(allowed.get("committed_at"), str)
                or not allowed["committed_at"]
            ):
                raise VerifyError("committed transaction does not bind coordinator HEAD")
            journals = []
    if journals:
        descriptions: list[str] = []
        root_head = text(root, ["git", "rev-parse", "HEAD"])
        for path in journals:
            try:
                journal = load_json(path)
                phase = journal.get("phase", "unknown")
                checkpoint = journal.get("checkpoint_id", path.stem)
                previous = journal.get("previous_root", "unknown")
                descriptions.append(
                    f"{checkpoint} phase={phase} root={root_head} previous_root={previous}"
                )
                for child_id, child in sorted(
                    (journal.get("expected_children") or {}).items()
                ):
                    relative = child.get("path")
                    if not isinstance(relative, str):
                        descriptions.append(f"  child {child_id}: invalid path")
                        continue
                    try:
                        repo = safe_path(root, relative)
                    except VerifyError as exc:
                        descriptions.append(f"  child {child_id}: {exc}")
                        continue
                    if not repo.is_dir():
                        descriptions.append(f"  child {child_id}: missing {relative}")
                        continue
                    actual_head = run(
                        root, ["git", "rev-parse", "HEAD"], cwd=repo, check=False
                    )
                    actual_tree = run(
                        root, ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, check=False
                    )
                    dirty = run(
                        root,
                        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                        cwd=repo,
                        check=False,
                    )
                    head_value = actual_head.stdout.decode().strip()
                    tree_value = actual_tree.stdout.decode().strip()
                    matches = (
                        actual_head.returncode == 0
                        and actual_tree.returncode == 0
                        and head_value == child.get("head")
                        and tree_value == child.get("tree")
                        and not dirty.stdout
                    )
                    descriptions.append(
                        f"  child {child_id}: head={head_value or 'unreadable'} "
                        f"tree={tree_value or 'unreadable'} clean={not bool(dirty.stdout)} "
                        f"recorded_identity_match={matches}"
                    )
                if phase == "prepared" and isinstance(
                    journal.get("expected_root_tree"), str
                ):
                    expected_tree = journal["expected_root_tree"]
                    index_diff = run(
                        root,
                        ["git", "diff", "--cached", "--quiet", expected_tree, "--"],
                        check=False,
                    )
                    descriptions.append(
                        f"  prepared_index_match={index_diff.returncode == 0} "
                        f"expected_root_tree={expected_tree}"
                    )
            except VerifyError as exc:
                descriptions.append(f"{path.name}[invalid: {exc}]")
        raise PendingTransaction(
            "unfinished cross-repository transaction; do not reset children; "
            "inspect/finalize with scripts/transaction.py:\n" + "\n".join(descriptions)
        )
    status = text(root, ["git", "status", "--porcelain=v1", "--untracked-files=all"])
    if status:
        raise VerifyError(f"root worktree is not clean:\n{status}")
    hooks = text(root, ["git", "config", "--get", "core.hooksPath"])
    if hooks not in {".githooks", str(root / ".githooks")}:
        raise VerifyError("core.hooksPath must be .githooks; run scripts/bootstrap.sh")
    for relative in (".githooks/pre-commit", ".githooks/commit-msg"):
        hook = root / relative
        if not hook.is_file() or not os.access(hook, os.X_OK):
            raise VerifyError(f"required hook is missing or not executable: {relative}")
    return text(root, ["git", "rev-parse", "HEAD"]), str(git_dir)


def verify_tracked_policy(root: Path) -> None:
    raw = run(root, ["git", "ls-files", "-z"]).stdout
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        path = encoded.decode(errors="surrogateescape")
        lower = path.lower()
        if path in PLACEHOLDERS or path == "projects/.gitkeep":
            continue
        if generated_path(path) or FORBIDDEN_SUFFIX_RE.search(lower):
            raise VerifyError(f"forbidden artifact is tracked: {path}")
        if path.startswith("projects/"):
            stage = text(root, ["git", "ls-files", "--stage", "--", path])
            if not stage.startswith("160000 "):
                raise VerifyError(f"projects entry is not a gitlink: {path}")
            continue
        file_path = root / path
        if file_path.is_symlink():
            resolved = file_path.resolve()
            if resolved != root and root not in resolved.parents:
                raise VerifyError(f"tracked symlink escapes workspace: {path}")
            continue
        if file_path.is_file() and file_path.stat().st_size > 10 * 1024 * 1024:
            raise VerifyError(f"tracked file exceeds 10 MiB policy: {path}")
        if file_path.is_file():
            blob = file_path.read_bytes()
            for label, pattern in SECRET_PATTERNS:
                if pattern.search(blob):
                    raise VerifyError(f"probable {label} in tracked file: {path}")


def verify_external_evidence(
    root: Path,
    descriptors: list[dict[str, Any]],
    *,
    require_acceptance_artifacts: bool,
) -> None:
    for descriptor in descriptors:
        relative = descriptor["path"]
        if not relative.startswith("artifacts/"):
            raise VerifyError(f"external evidence is outside artifacts/: {relative}")
        path = safe_path(root, relative)
        required = descriptor["required_for_resume"] or (
            require_acceptance_artifacts and descriptor["required_at_acceptance"]
        )
        if not path.exists():
            if required:
                raise VerifyError(f"required external evidence is missing: {relative}")
            continue
        if (
            not path.is_file()
            or path.stat().st_size != descriptor["size"]
            or digest(path) != descriptor["sha256"]
        ):
            raise VerifyError(f"external evidence identity mismatch: {relative}")


def verify_command_capture_records(root: Path, evidence: dict[str, Any]) -> None:
    for result in evidence.get("command_results", []):
        if result.get("raw_streams_retained") is not True:
            continue
        record_path = safe_path(root, result["record"]["path"])
        if not record_path.exists():
            continue
        record = load_json(record_path)
        expected_scalars = {
            "schema": "amdgpu-sim.command-evidence.v1",
            "argv": result["argv"],
            "cwd": result["cwd"],
            "started_at": result["started_at"],
            "ended_at": result["ended_at"],
            "exit_code": result["exit_code"],
        }
        if any(record.get(key) != value for key, value in expected_scalars.items()):
            raise VerifyError(f"capture record semantics mismatch: {result['id']}")
        for stream in ("stdout", "stderr"):
            descriptor = result[stream]
            expected_stream = {
                "path": descriptor["path"],
                "size": descriptor["size"],
                "sha256": descriptor["sha256"],
            }
            if record.get(stream) != expected_stream:
                raise VerifyError(
                    f"capture record {stream} descriptor mismatch: {result['id']}"
                )


def verify_checkpoint(
    root: Path,
    head: str,
    *,
    require_acceptance_artifacts: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    current = load_json(root / "state/current.json")
    if current.get("schema") != "amdgpu-sim.current.v1":
        raise VerifyError("unsupported current.json schema")
    checkpoint_id = current.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not CP_RE.match(checkpoint_id):
        raise VerifyError("malformed current checkpoint_id")
    checkpoint_path = root / "state" / "checkpoints" / f"{checkpoint_id}.json"
    checkpoint = load_json(checkpoint_path)
    sequence = checkpoint.get("sequence")
    if (
        checkpoint.get("schema") != "amdgpu-sim.checkpoint.v1"
        or checkpoint.get("id") != checkpoint_id
        or not isinstance(sequence, int)
        or sequence < 1
        or checkpoint_id != f"CP-{sequence:04d}"
        or checkpoint.get("status") not in {"partial", "accepted", "failed-but-learned"}
    ):
        raise VerifyError("checkpoint ID does not match current pointer")
    expected_parent = None if sequence == 1 else f"CP-{sequence - 1:04d}"
    if checkpoint.get("parent_checkpoint") != expected_parent:
        raise VerifyError("checkpoint parent/sequence chain is invalid")
    if current.get("checkpoint_sha256") != digest(checkpoint_path):
        raise VerifyError("current checkpoint_sha256 does not match the checkpoint file")
    for key in ("goal_id", "phase_id"):
        if checkpoint.get(key) != current.get(key):
            raise VerifyError(f"checkpoint/current {key} mismatch")
    for relative, expected in {
        "PLAN.md": checkpoint.get("plan_sha256"),
        "GOAL.md": checkpoint.get("goal_sha256"),
        "SOURCE_LOCK.json": checkpoint.get("source_lock_sha256"),
    }.items():
        if not isinstance(expected, str) or digest(root / relative) != expected:
            raise VerifyError(f"checkpoint hash mismatch: {relative}")
    action = checkpoint.get("next_action")
    if not isinstance(action, dict) or action.get("id") != current.get("next_action_id"):
        raise VerifyError("checkpoint/current next action mismatch")
    if current.get("state") not in {"ready", "paused", "blocked", "complete"}:
        raise VerifyError("invalid current state")
    if (
        not isinstance(action.get("argv"), list)
        or not action["argv"]
        or not all(isinstance(item, str) and item for item in action["argv"])
    ):
        raise VerifyError("checkpoint next action has no argv")
    if not isinstance(action.get("cwd"), str) or not action["cwd"]:
        raise VerifyError("checkpoint next action has no cwd")
    prerequisites = action.get("prerequisites")
    if (
        not isinstance(prerequisites, list)
        or not prerequisites
        or not all(isinstance(item, str) and item for item in prerequisites)
        or not isinstance(action.get("expected"), str)
        or not action["expected"]
        or not isinstance(action.get("rollback_boundary"), str)
        or not action["rollback_boundary"]
    ):
        raise VerifyError("checkpoint next action contract is incomplete")
    action_cwd = safe_path(root, action["cwd"])
    if not action_cwd.is_dir():
        raise VerifyError("checkpoint next action cwd does not exist")
    command = action["argv"][0]
    if not isinstance(command, str) or not command:
        raise VerifyError("checkpoint next action command is invalid")
    if "/" in command:
        executable = safe_path(action_cwd, command)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise VerifyError("checkpoint next action command is not executable")
    elif shutil.which(command) is None:
        raise VerifyError("checkpoint next action command is not available")
    message = text(root, ["git", "log", "-1", "--format=%B"])
    try:
        check_commit_message(message)
    except MessageError as exc:
        raise VerifyError(f"HEAD commit message violates audit policy: {exc}") from exc
    trailers = parse_trailers(message)
    expected_trailers = {
        "Checkpoint-ID": checkpoint_id,
        "Goal-ID": current.get("goal_id"),
        "Plan-Revision": str(current.get("plan_revision")),
        "Source-Lock-SHA256": checkpoint.get("source_lock_sha256"),
    }
    evidence_manifest = checkpoint.get("evidence_manifest_sha256")
    if not isinstance(evidence_manifest, str):
        raise VerifyError("checkpoint has no evidence manifest identity")
    expected_trailers.update(
        {
            "Evidence-Manifest-SHA256": evidence_manifest,
            "Change-Kind": checkpoint.get("change_kind"),
            "Baseline-Commit": checkpoint.get("baseline_commit_marker"),
        }
    )
    for key, expected in expected_trailers.items():
        if not isinstance(expected, str) or trailers.get(key) != [expected]:
            raise VerifyError(f"HEAD has missing, duplicate, or mismatched {key} trailer")
    recorded = checkpoint.get("root_commit")
    if recorded not in (None, "self-described-by-checkpoint-trailer", head):
        raise VerifyError("checkpoint root_commit does not match HEAD")
    parent = checkpoint.get("root_parent_commit")
    if parent is not None and text(root, ["git", "rev-parse", "HEAD^"]) != parent:
        raise VerifyError("checkpoint root_parent_commit does not match HEAD^")
    evidence_ids = checkpoint.get("evidence_ids")
    evidence_hashes = checkpoint.get("evidence_sha256")
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or len(evidence_ids) != len(set(evidence_ids))
        or not all(isinstance(item, str) and EV_RE.fullmatch(item) for item in evidence_ids)
        or not isinstance(evidence_hashes, dict)
        or set(evidence_hashes) != set(evidence_ids)
    ):
        raise VerifyError("checkpoint evidence identity set is incomplete")
    for evidence_id in evidence_ids:
        path = root / "state" / "evidence" / f"{evidence_id}.json"
        evidence = load_json(path)
        try:
            _, descriptors = validate_evidence(
                evidence, expected_id=evidence_id, checkpoint_id=checkpoint_id
            )
        except EvidencePolicyError as exc:
            raise VerifyError(f"evidence policy mismatch: {evidence_id}: {exc}") from exc
        expected = evidence_hashes.get(evidence_id)
        if not isinstance(expected, str) or digest(path) != expected:
            raise VerifyError(f"evidence hash mismatch: {evidence_id}")
        verify_external_evidence(
            root,
            descriptors,
            require_acceptance_artifacts=require_acceptance_artifacts,
        )
        verify_command_capture_records(root, evidence)
    manifest_id = checkpoint.get("evidence_manifest_id")
    if manifest_id not in evidence_ids:
        raise VerifyError("checkpoint evidence_manifest_id is not an evidence member")
    if checkpoint.get("evidence_manifest_sha256") != evidence_hashes[manifest_id]:
        raise VerifyError("checkpoint evidence manifest hash has no file identity")
    manifest = load_json(root / "state" / "evidence" / f"{manifest_id}.json")
    expected_includes = {
        evidence_id: evidence_hashes[evidence_id]
        for evidence_id in evidence_ids
        if evidence_id != manifest_id
    }
    if manifest.get("includes") != expected_includes:
        raise VerifyError("evidence manifest includes set does not match checkpoint evidence")
    lesson_ids = checkpoint.get("bitlesson_ids")
    lesson_hashes = checkpoint.get("bitlesson_sha256")
    if (
        not isinstance(lesson_ids, list)
        or len(lesson_ids) != len(set(lesson_ids))
        or not all(isinstance(item, str) and BL_RE.fullmatch(item) for item in lesson_ids)
        or not isinstance(lesson_hashes, dict)
        or set(lesson_hashes) != set(lesson_ids)
    ):
        raise VerifyError("checkpoint bitlesson identity set is incomplete")
    for lesson_id in lesson_ids:
        path = root / "state" / "bitlessons" / f"{lesson_id}.json"
        lesson = load_json(path)
        if (
            lesson.get("schema") != "amdgpu-sim.bitlesson.v1"
            or lesson.get("id") != lesson_id
            or lesson.get("checkpoint_id") != checkpoint_id
        ):
            raise VerifyError(f"bitlesson ID mismatch: {lesson_id}")
        lesson_evidence = lesson.get("evidence_ids")
        required_text = (
            "question_or_symptom",
            "observation",
            "conclusion",
            "decision",
        )
        if (
            not all(isinstance(lesson.get(key), str) and lesson[key] for key in required_text)
            or not isinstance(lesson_evidence, list)
            or not lesson_evidence
            or len(lesson_evidence) != len(set(lesson_evidence))
            or not all(item in evidence_ids for item in lesson_evidence)
            or not isinstance(lesson.get("invalidated_assumptions"), list)
            or not lesson["invalidated_assumptions"]
            or not all(
                isinstance(item, str) and item
                for item in lesson["invalidated_assumptions"]
            )
            or not isinstance(lesson.get("applies_to"), list)
            or not lesson["applies_to"]
            or not all(isinstance(item, str) and item for item in lesson["applies_to"])
            or lesson.get("confidence") not in {"low", "medium", "high"}
            or not isinstance(lesson.get("supersedes"), list)
            or lesson.get("status") not in {"active", "superseded"}
        ):
            raise VerifyError(f"bitlesson content is incomplete: {lesson_id}")
        expected = lesson_hashes.get(lesson_id)
        if not isinstance(expected, str) or digest(path) != expected:
            raise VerifyError(f"bitlesson hash mismatch: {lesson_id}")
    return current, checkpoint, action


def verify_checkpoint_repositories(
    checkpoint: dict[str, Any], lock: dict[str, Any]
) -> None:
    repositories = checkpoint.get("repositories")
    if not isinstance(repositories, list):
        raise VerifyError("checkpoint repositories is not a list")
    sources = lock.get("sources")
    if not isinstance(sources, list):
        raise VerifyError("SOURCE_LOCK sources is not a list")
    git_sources = [
        source
        for source in sources
        if isinstance(source, dict) and source.get("materialization") == "gitlink"
    ]
    repository_ids = [
        repository.get("id")
        for repository in repositories
        if isinstance(repository, dict)
    ]
    source_ids = [source.get("id") for source in git_sources]
    if (
        not git_sources
        or len(repository_ids) != len(repositories)
        or len(repository_ids) != len(set(repository_ids))
        or any(not isinstance(source_id, str) for source_id in source_ids)
        or len(source_ids) != len(set(source_ids))
        or set(repository_ids) != set(source_ids)
    ):
        raise VerifyError("checkpoint repository set does not match frozen Git sources")
    by_id = {repository["id"]: repository for repository in repositories}
    for source in git_sources:
        required_source_fields = (
            "id",
            "path",
            "baseline_commit",
            "baseline_tree",
            "work_head",
            "work_tree",
            "baseline_tag",
            "baseline_tag_object",
            "administrative_git_dir",
        )
        if not all(
            isinstance(source.get(field), str) and source[field]
            for field in required_source_fields
        ):
            raise VerifyError(
                f"frozen source identity is incomplete: {source.get('id')}"
            )
        expected = {
            "id": source["id"],
            "path": source["path"],
            "baseline_commit": source["baseline_commit"],
            "baseline_tree": source["baseline_tree"],
            "head": source["work_head"],
            "tree": source["work_tree"],
            "baseline_tag": source["baseline_tag"],
            "baseline_tag_object": source["baseline_tag_object"],
            "administrative_git_dir": source["administrative_git_dir"],
            "clean": True,
        }
        if by_id[source["id"]] != expected:
            raise VerifyError(
                f"checkpoint repository identity mismatch: {source['id']}"
            )


def verify_pre_freeze_candidate(
    root: Path,
    lock: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    candidate_hash = lock.get("pre_freeze_candidate_sha256")
    candidate_artifact = lock.get("pre_freeze_candidate_artifact")
    if not isinstance(candidate_artifact, dict):
        raise VerifyError("SOURCE_LOCK pre-freeze candidate artifact is absent")
    candidate_relative = candidate_artifact.get("path")
    candidate_size = candidate_artifact.get("size")
    if (
        not isinstance(candidate_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", candidate_hash)
        or not isinstance(candidate_relative, str)
        or not candidate_relative.startswith("artifacts/")
        or candidate_artifact.get("sha256") != candidate_hash
        or not isinstance(candidate_size, int)
        or isinstance(candidate_size, bool)
        or candidate_size < 0
        or candidate_artifact.get("required_for_resume") is not False
    ):
        raise VerifyError("SOURCE_LOCK pre-freeze candidate artifact is invalid")
    candidate_path = safe_path(root, candidate_relative)
    if candidate_path.exists():
        if (
            not candidate_path.is_file()
            or candidate_path.stat().st_size != candidate_size
            or digest(candidate_path) != candidate_hash
        ):
            raise VerifyError("SOURCE_LOCK pre-freeze candidate artifact identity mismatch")
        candidate_value = load_json(candidate_path)
        if (
            candidate_value.get("schema") != "amdgpu-sim.source-lock.v1"
            or candidate_value.get("status") == "frozen"
        ):
            raise VerifyError("SOURCE_LOCK pre-freeze candidate artifact is not a candidate")
    manifest_artifacts = manifest.get("external_artifacts")
    if not isinstance(manifest_artifacts, list):
        raise VerifyError("evidence manifest has no external artifact inventory")
    candidate_bindings = [
        item
        for item in manifest_artifacts
        if isinstance(item, dict) and item.get("path") == candidate_relative
    ]
    if len(candidate_bindings) != 1:
        raise VerifyError("evidence manifest does not uniquely bind pre-freeze candidate")
    binding = candidate_bindings[0]
    expected_candidate_binding = {
        "path": candidate_relative,
        "size": candidate_size,
        "sha256": candidate_hash,
        "tracked": False,
        "required_at_acceptance": True,
        "required_for_resume": False,
    }
    if any(binding.get(key) != value for key, value in expected_candidate_binding.items()):
        raise VerifyError("evidence manifest pre-freeze candidate identity mismatch")


def gitmodule_value(root: Path, source_id: str, key: str) -> str:
    return text(
        root,
        ["git", "config", "-f", ".gitmodules", "--get", f"submodule.{source_id}.{key}"],
    )


def verify_annotated_tag(
    root: Path,
    repo: Path,
    source_id: str,
    *,
    commit: str,
    tree: str,
    tag: str,
    tag_object: str,
    tag_payload: str,
    tag_payload_sha256: str,
    required_lines: list[str] | None = None,
) -> None:
    if text(root, ["git", "rev-parse", f"{commit}^{{tree}}"], cwd=repo) != tree:
        raise VerifyError(f"tagged tree mismatch for {source_id}:{commit}")
    if text(root, ["git", "cat-file", "-t", tag], cwd=repo) != "tag":
        raise VerifyError(f"baseline ref is not annotated for {source_id}:{tag}")
    if text(root, ["git", "rev-parse", tag], cwd=repo) != tag_object:
        raise VerifyError(f"tag object mismatch for {source_id}:{tag}")
    actual_payload = run(root, ["git", "cat-file", "tag", tag], cwd=repo).stdout
    if actual_payload.decode() != tag_payload:
        raise VerifyError(f"tag payload mismatch for {source_id}:{tag}")
    if hashlib.sha256(actual_payload).hexdigest() != tag_payload_sha256:
        raise VerifyError(f"tag payload hash mismatch for {source_id}:{tag}")
    if required_lines:
        lines = set(tag_payload.splitlines())
        missing = [line for line in required_lines if line not in lines]
        if missing:
            raise VerifyError(f"tag payload metadata mismatch for {source_id}:{tag}")
    if text(root, ["git", "rev-list", "-n", "1", tag], cwd=repo) != commit:
        raise VerifyError(f"tag target mismatch for {source_id}:{tag}")
    run_discard_stdout(
        root,
        ["git", "archive", "--format=tar", commit],
        cwd=repo,
    )


def verify_child(root: Path, source: dict[str, Any]) -> None:
    source_id = source["id"]
    if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
        raise VerifyError(f"source has an unsafe submodule name: {source_id!r}")
    relative = source["path"]
    repo = safe_path(root, relative)
    if not repo.is_dir():
        raise VerifyError(f"materialized source is missing: {relative}")
    expected_head = source.get("work_head")
    baseline = source.get("baseline_commit")
    baseline_tree = source.get("baseline_tree")
    work_tree = source.get("work_tree", baseline_tree)
    tag = source.get("baseline_tag")
    tag_object = source.get("baseline_tag_object")
    tag_payload = source.get("baseline_tag_payload")
    tag_payload_sha256 = source.get("baseline_tag_payload_sha256")
    if not all(
        isinstance(value, str)
        for value in (
            expected_head,
            baseline,
            baseline_tree,
            work_tree,
            tag,
            tag_object,
            tag_payload,
            tag_payload_sha256,
        )
    ):
        raise VerifyError(f"incomplete frozen identity for {source_id}")
    index = text(root, ["git", "ls-files", "--stage", "--", relative])
    fields = index.split()
    if len(fields) < 4 or fields[0] != "160000" or fields[1] != expected_head:
        raise VerifyError(f"root gitlink mismatch for {source_id}")
    if gitmodule_value(root, source_id, "path") != relative:
        raise VerifyError(f".gitmodules path mismatch for {source_id}")
    if gitmodule_value(root, source_id, "url") != source["upstream_url"]:
        raise VerifyError(f".gitmodules canonical URL mismatch for {source_id}")
    if gitmodule_value(root, source_id, "branch") != source["upstream_ref"]:
        raise VerifyError(f".gitmodules branch mismatch for {source_id}")
    if text(root, ["git", "rev-parse", "HEAD"], cwd=repo) != expected_head:
        raise VerifyError(f"child HEAD mismatch for {source_id}")
    if text(root, ["git", "rev-parse", "HEAD^{tree}"], cwd=repo) != work_tree:
        raise VerifyError(f"child tree mismatch for {source_id}")
    if text(root, ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repo):
        raise VerifyError(f"child worktree is dirty: {source_id}")
    expected_remotes = source.get("expected_remotes", {"upstream": source["transport_url"]})
    actual_remotes = text(root, ["git", "remote"], cwd=repo).splitlines()
    if sorted(actual_remotes) != sorted(expected_remotes):
        raise VerifyError(f"child remote set mismatch for {source_id}")
    for name, expected_url in expected_remotes.items():
        if text(root, ["git", "remote", "get-url", name], cwd=repo) != expected_url:
            raise VerifyError(f"child remote URL mismatch for {source_id}:{name}")
    expected_push_urls = source.get("expected_push_urls", {"upstream": "no_push"})
    for name, expected_url in expected_push_urls.items():
        if text(root, ["git", "remote", "get-url", "--push", name], cwd=repo) != expected_url:
            raise VerifyError(f"child push URL mismatch for {source_id}:{name}")
    expected_hooks = os.path.relpath(root / ".githooks", repo)
    if text(root, ["git", "config", "--get", "core.hooksPath"], cwd=repo) != expected_hooks:
        raise VerifyError(f"child hooksPath mismatch for {source_id}")
    if text(root, ["git", "rev-parse", "--is-shallow-repository"], cwd=repo) != "false":
        raise VerifyError(f"child repository is shallow: {source_id}")
    sparse = run(
        root,
        ["git", "config", "--bool", "core.sparseCheckout"],
        cwd=repo,
        check=False,
    )
    if sparse.returncode not in (0, 1) or sparse.stdout.decode().strip() == "true":
        raise VerifyError(f"child repository uses sparse checkout: {source_id}")
    indexed = run(root, ["git", "ls-files", "-v", "-z"], cwd=repo).stdout
    skipped = sum(
        1
        for entry in indexed.split(b"\0")
        if entry and entry[:1] in {b"S", b"s"}
    )
    worktree = source.get("worktree_verification")
    if skipped or worktree != {
        "full_checkout": True,
        "sparse_checkout": False,
        "skip_worktree_entries": 0,
    }:
        raise VerifyError(f"child worktree is not a frozen full checkout: {source_id}")
    child_git_dir = Path(text(root, ["git", "rev-parse", "--absolute-git-dir"], cwd=repo))
    child_common_dir = Path(text(root, ["git", "rev-parse", "--git-common-dir"], cwd=repo))
    if not child_common_dir.is_absolute():
        child_common_dir = (repo / child_common_dir).resolve()
    else:
        child_common_dir = child_common_dir.resolve()
    root_common_dir = Path(text(root, ["git", "rev-parse", "--git-common-dir"]))
    if not root_common_dir.is_absolute():
        root_common_dir = (root / root_common_dir).resolve()
    else:
        root_common_dir = root_common_dir.resolve()
    expected_admin_relative = (PurePosixPath("modules") / source_id).as_posix()
    expected_child_common_dir = (root_common_dir / expected_admin_relative).resolve()
    expected_gitfile = (
        f"gitdir: {os.path.relpath(expected_child_common_dir, repo)}\n"
    )
    if not (repo / ".git").is_file():
        raise VerifyError(f"child does not use an absorbed gitfile: {source_id}")
    if (
        child_common_dir != expected_child_common_dir
        or child_git_dir.resolve() != child_common_dir
        or (repo / ".git").read_text(encoding="utf-8") != expected_gitfile
    ):
        raise VerifyError(f"child common-dir is not its exact root module path: {source_id}")
    expected_admin = source.get("administrative_git_dir")
    if expected_admin != expected_admin_relative:
        raise VerifyError(f"child administrative Git path mismatch: {source_id}")
    for marker in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "BISECT_LOG", "rebase-merge", "rebase-apply"):
        if (child_git_dir / marker).exists():
            raise VerifyError(f"child Git operation is in progress: {source_id}:{marker}")
    if (child_git_dir / "info" / "grafts").is_file():
        raise VerifyError(f"child history uses a grafts file: {source_id}")
    if text(root, ["git", "replace", "-l"], cwd=repo):
        raise VerifyError(f"child history uses replacement objects: {source_id}")
    alternates = child_git_dir / "objects" / "info" / "alternates"
    if alternates.is_file() and alternates.read_text(encoding="utf-8").strip():
        raise VerifyError(f"child repository still depends on alternates: {source_id}")
    verify_annotated_tag(
        root,
        repo,
        source_id,
        commit=baseline,
        tree=baseline_tree,
        tag=tag,
        tag_object=tag_object,
        tag_payload=tag_payload,
        tag_payload_sha256=tag_payload_sha256,
        required_lines=[
            f"Upstream-URL: {source['upstream_url']}",
            f"Upstream-Ref: {source['upstream_ref']}",
            f"Baseline-Commit: {baseline}",
            f"Baseline-Tree: {baseline_tree}",
        ],
    )
    ancestor = run(root, ["git", "merge-base", "--is-ancestor", baseline, expected_head], cwd=repo, check=False)
    if ancestor.returncode:
        raise VerifyError(f"work head is not a baseline descendant: {source_id}")
    if source.get("offline_tree_verified") is not True:
        raise VerifyError(f"offline baseline tree is not evidenced: {source_id}")
    object_counts = text(root, ["git", "count-objects", "-v"], cwd=repo)
    counts = dict(
        line.split(": ", 1) for line in object_counts.splitlines() if ": " in line
    )
    if counts.get("garbage", "0") != "0":
        raise VerifyError(f"child object database has garbage: {source_id}")
    run_discard_stdout(root, ["git", "fsck", "--connectivity-only"], cwd=repo)
    submodules = text(root, ["git", "submodule", "status", "--recursive"], cwd=repo)
    initialized = [line for line in submodules.splitlines() if line and not line.startswith("-")]
    if initialized:
        raise VerifyError(f"nested submodules unexpectedly materialized: {source_id}")
    history = source.get("history")
    if not isinstance(history, dict) or history.get("shallow") is not False:
        raise VerifyError(f"frozen history metadata is incomplete: {source_id}")
    partial_filter = history.get("partial_clone_filter")
    if partial_filter not in {"blob:none", "tree:0", "none"}:
        raise VerifyError(f"unsupported historical object filter: {source_id}")
    configured_filter = run(
        root,
        ["git", "config", "--get", "remote.upstream.partialclonefilter"],
        cwd=repo,
        check=False,
    )
    if configured_filter.returncode not in (0, 1):
        raise VerifyError(f"cannot inspect partial clone filter: {source_id}")
    actual_filter = configured_filter.stdout.decode().strip() or "none"
    configured_promisor = run(
        root,
        ["git", "config", "--bool", "--get", "remote.upstream.promisor"],
        cwd=repo,
        check=False,
    )
    if configured_promisor.returncode not in (0, 1):
        raise VerifyError(f"cannot inspect promisor configuration: {source_id}")
    actual_promisor = configured_promisor.stdout.decode().strip() == "true"
    if (
        actual_filter != partial_filter
        or history.get("promisor_remote") is not actual_promisor
        or (partial_filter != "none") != actual_promisor
    ):
        raise VerifyError(f"partial clone provenance mismatch: {source_id}")
    expected_object_scope = "all-local" if partial_filter == "none" else "promisor-filtered"
    if (
        history.get("commit_ancestry_scope")
        != "all commits reachable from the locked baseline head"
        or history.get("commit_ancestry_offline_traversable") is not True
        or history.get("historical_tree_blob_scope") != expected_object_scope
        or history.get("fresh_offline_clone_bundle_available") is not False
        or history.get("locked_baseline_tree_fully_hydrated") is not True
    ):
        raise VerifyError(f"locked baseline hydration metadata is missing: {source_id}")
    expected_count = history.get("reachable_commit_count")
    actual_count = int(text(root, ["git", "rev-list", "--count", baseline], cwd=repo))
    if not isinstance(expected_count, int) or actual_count != expected_count:
        raise VerifyError(f"reachable history count mismatch: {source_id}")
    nested = source.get("nested_submodules")
    if not isinstance(nested, dict) or nested.get("materialized") is not False:
        raise VerifyError(f"nested submodule metadata is incomplete: {source_id}")
    modules = repo / ".gitmodules"
    declared_paths: list[str] = []
    if modules.is_file():
        listed = run(
            root,
            [
                "git",
                "config",
                "-f",
                ".gitmodules",
                "--get-regexp",
                r"^submodule\..*\.path$",
            ],
            cwd=repo,
            check=False,
        )
        if listed.returncode not in (0, 1):
            raise VerifyError(f"cannot parse nested submodules: {source_id}")
        declared_paths = sorted(
            line.split(None, 1)[1]
            for line in listed.stdout.decode().splitlines()
            if line.strip()
        )
    raw_tree = run(
        root, ["git", "ls-tree", "-r", "-z", baseline], cwd=repo
    ).stdout
    gitlink_paths = []
    for entry in raw_tree.split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        if metadata.split(b" ", 1)[0] == b"160000":
            gitlink_paths.append(path.decode(errors="surrogateescape"))
    gitlink_paths.sort()
    expected_nested = {
        "declared_count": len(declared_paths),
        "declared_paths": declared_paths,
        "gitlink_count": len(gitlink_paths),
        "gitlink_paths": gitlink_paths,
        "stale_declarations": sorted(set(declared_paths) - set(gitlink_paths)),
        "undeclared_gitlinks": sorted(set(gitlink_paths) - set(declared_paths)),
    }
    for key, expected in expected_nested.items():
        if nested.get(key) != expected:
            raise VerifyError(f"nested submodule {key} mismatch: {source_id}")
    for compatibility in source.get("compatibility_revisions", []):
        required = (
            compatibility.get("commit"),
            compatibility.get("tree"),
            compatibility.get("tag"),
            compatibility.get("tag_object"),
            compatibility.get("tag_payload"),
            compatibility.get("tag_payload_sha256"),
        )
        if not all(isinstance(value, str) for value in required):
            raise VerifyError(f"incomplete compatibility identity for {source_id}")
        if compatibility.get("offline_tree_verified") is not True:
            raise VerifyError(f"compatibility tree is not offline-verified: {source_id}")
        verify_annotated_tag(
            root,
            repo,
            source_id,
            commit=required[0],
            tree=required[1],
            tag=required[2],
            tag_object=required[3],
            tag_payload=required[4],
            tag_payload_sha256=required[5],
            required_lines=[
                f"Upstream-URL: {source['upstream_url']}",
                f"Compatibility-Commit: {required[0]}",
                f"Compatibility-Tree: {required[1]}",
            ],
        )
    for selected in source.get("selected_paths", []):
        selected_path = safe_path(repo, selected)
        if not selected_path.exists():
            raise VerifyError(f"selected source path is missing: {source_id}:{selected}")
    license_files = source.get("license_files")
    if not isinstance(license_files, list) or not license_files:
        raise VerifyError(f"source has no frozen license inventory: {source_id}")
    for license_file in license_files:
        license_path = safe_path(repo, license_file["path"])
        if not license_path.is_file() or digest(license_path) != license_file.get("sha256"):
            raise VerifyError(f"license file identity mismatch: {source_id}:{license_file.get('path')}")
    tracking_ref = source.get("upstream_tracking_ref")
    if tracking_ref and text(root, ["git", "rev-parse", tracking_ref], cwd=repo) != baseline:
        raise VerifyError(f"upstream tracking ref mismatch for {source_id}")


def verify_sources(root: Path, checkpoint_id: str) -> None:
    lock = load_json(root / "SOURCE_LOCK.json")
    if lock.get("schema") != "amdgpu-sim.source-lock.v1":
        raise VerifyError("unsupported SOURCE_LOCK schema")
    if (
        lock.get("status") != "frozen"
        or not LOCK_ID_RE.fullmatch(str(lock.get("lock_id", "")))
        or lock.get("frozen_by_checkpoint") != checkpoint_id
    ):
        raise VerifyError("SOURCE_LOCK is not frozen")
    checkpoint = load_json(root / "state" / "checkpoints" / f"{checkpoint_id}.json")
    verify_checkpoint_repositories(checkpoint, lock)
    parent_commit = lock.get("accepted_parent_root_commit")
    parent_hash = lock.get("accepted_parent_source_lock_sha256")
    candidate_hash = lock.get("pre_freeze_candidate_sha256")
    if (
        not isinstance(parent_commit, str)
        or not SHA_RE.match(parent_commit)
        or parent_commit != checkpoint.get("root_parent_commit")
        or not isinstance(parent_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", parent_hash)
        or not isinstance(candidate_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", candidate_hash)
    ):
        raise VerifyError("SOURCE_LOCK accepted parent identity is incomplete")
    parent_blob = run(
        root,
        ["git", "cat-file", "blob", f"{parent_commit}:SOURCE_LOCK.json"],
    ).stdout
    if hashlib.sha256(parent_blob).hexdigest() != parent_hash:
        raise VerifyError("SOURCE_LOCK accepted parent blob hash mismatch")
    try:
        import datetime as dt

        frozen_at = dt.datetime.fromisoformat(
            str(lock.get("frozen_at", "")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise VerifyError("SOURCE_LOCK frozen_at is invalid") from exc
    if frozen_at.tzinfo is None:
        raise VerifyError("SOURCE_LOCK frozen_at has no timezone")
    evidence_id = lock.get("resolution_evidence_id")
    evidence_relative = lock.get("resolution_evidence_path")
    evidence_sha = lock.get("resolution_evidence_sha256")
    if not all(isinstance(value, str) for value in (evidence_id, evidence_relative, evidence_sha)):
        raise VerifyError("SOURCE_LOCK resolution evidence identity is incomplete")
    if evidence_relative != f"state/evidence/{evidence_id}.json":
        raise VerifyError("SOURCE_LOCK resolution evidence path is not canonical")
    evidence_path = safe_path(root, evidence_relative)
    evidence = load_json(evidence_path)
    if (
        evidence.get("schema") != "amdgpu-sim.evidence.v1"
        or evidence.get("id") != evidence_id
        or evidence.get("checkpoint_id") != checkpoint_id
        or evidence.get("type") != "official-source-resolution"
        or digest(evidence_path) != evidence_sha
        or (checkpoint.get("evidence_sha256") or {}).get(evidence_id) != evidence_sha
    ):
        raise VerifyError("SOURCE_LOCK resolution evidence does not match its file")
    manifest_id = checkpoint.get("evidence_manifest_id")
    if not isinstance(manifest_id, str):
        raise VerifyError("checkpoint has no evidence manifest for source provenance")
    manifest = load_json(root / "state" / "evidence" / f"{manifest_id}.json")
    verify_pre_freeze_candidate(root, lock, manifest)
    records = evidence.get("repo_commits")
    if not isinstance(records, list):
        raise VerifyError("SOURCE_LOCK resolution evidence has no records")
    record_ids = [item.get("id") for item in records if isinstance(item, dict)]
    source_ids = [item.get("id") for item in lock.get("sources", []) if isinstance(item, dict)]
    if (
        len(record_ids) != len(records)
        or len(record_ids) != len(set(record_ids))
        or len(source_ids) != len(lock.get("sources", []))
        or len(source_ids) != len(set(source_ids))
        or set(record_ids) != set(source_ids)
    ):
        raise VerifyError("SOURCE_LOCK resolution record set is incomplete or duplicated")
    evidence_records = {item["id"]: item for item in records}
    command_results = evidence.get("command_results")
    if not isinstance(command_results, list):
        raise VerifyError("SOURCE_LOCK resolution evidence has no command results")
    command_result_ids = [
        item.get("id") for item in command_results if isinstance(item, dict)
    ]
    if (
        len(command_result_ids) != len(command_results)
        or len(command_result_ids) != len(set(command_result_ids))
    ):
        raise VerifyError("SOURCE_LOCK resolution command results are duplicated")
    artifact = lock.get("resolution_artifact")
    if not isinstance(artifact, dict):
        raise VerifyError("SOURCE_LOCK resolution artifact is absent")
    artifact_path = safe_path(root, artifact.get("path", ""))
    if artifact.get("required_for_resume") is not False:
        raise VerifyError("external resolution artifact must not be required for resume")
    if artifact_path.exists():
        if (
            not artifact_path.is_file()
            or digest(artifact_path) != artifact.get("sha256")
            or artifact_path.stat().st_size != artifact.get("size")
        ):
            raise VerifyError("SOURCE_LOCK resolution artifact identity mismatch")
        artifact_value = load_json(artifact_path)
        if (
            artifact_value.get("schema") != "amdgpu-sim.source-resolution.v1"
            or artifact_value.get("status") != artifact.get("status")
        ):
            raise VerifyError("SOURCE_LOCK resolution artifact metadata mismatch")
    registered = set()
    for source in lock.get("sources", []):
        relative = source.get("path")
        if not isinstance(relative, str):
            raise VerifyError("source has no path")
        safe_path(root, relative)
        expected_revision = source.get("official_revision") or source.get("observed_head")
        record = evidence_records[source["id"]]
        expected_record = {
            "id": source["id"],
            "revision": expected_revision,
            "ref": source["upstream_ref"],
            "canonical_url": source["upstream_url"],
            "transport_url": source.get("transport_url"),
        }
        try:
            record_time = dt.datetime.fromisoformat(
                str(record.get("observed_at", "")).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise VerifyError(
                f"source resolution timestamp is invalid: {source.get('id')}"
            ) from exc
        if (
            source.get("resolution_evidence_id") != evidence_id
            or source.get("resolution_record_id") != source.get("id")
            or any(record.get(key) != value for key, value in expected_record.items())
            or record_time.tzinfo is None
            or record_time > frozen_at
            or record.get("command_result_id") not in command_result_ids
        ):
            raise VerifyError(f"source resolution provenance mismatch: {source.get('id')}")
        if source.get("materialization") == "gitlink":
            verify_child(root, source)
            registered.add(relative)
        elif source.get("role") == "acceptance-model":
            revision = source.get("official_revision")
            files = source.get("files")
            if not isinstance(revision, str) or not SHA_RE.match(revision):
                raise VerifyError("acceptance model has no immutable official revision")
            if not isinstance(files, list) or not files:
                raise VerifyError("acceptance model has no frozen file inventory")
            mirror = source.get("mirror_metadata")
            if not isinstance(mirror, dict):
                raise VerifyError("acceptance model mirror cross-check metadata is absent")
            expected_manifest = mirror.get("normalized_revision_manifest_sha256")
            if (
                mirror.get("endpoint") != "https://hf-mirror.com"
                or mirror.get("revision") != revision
                or mirror.get("status")
                != "cross-check-only; never substitutes for official revision evidence"
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(mirror.get("mirror_api_manifest_sha256", ""))
                )
                or not re.fullmatch(r"[0-9a-f]{64}", str(expected_manifest or ""))
            ):
                raise VerifyError("acceptance model mirror cross-check provenance mismatch")
            if model_manifest_digest(source) != expected_manifest:
                raise VerifyError("acceptance model manifest hash mismatch")
            if sum(item["size"] for item in files) != source.get("materialized_size"):
                raise VerifyError("acceptance model materialized size mismatch")
            official_evidence = source.get("official_evidence")
            try:
                official_observed = dt.datetime.fromisoformat(
                    str(source.get("official_evidence_observed_at", "")).replace(
                        "Z", "+00:00"
                    )
                )
            except ValueError as exc:
                raise VerifyError("acceptance model official evidence timestamp is invalid") from exc
            if (
                source.get("official_revision_status")
                != "frozen-from-official-fixed-revision-pages"
                or not isinstance(official_evidence, list)
                or not any(revision in url for url in official_evidence)
                or official_observed.tzinfo is None
                or source.get("official_raw_response_sha256", "missing") is not None
                or "no raw official API archive"
                not in str(source.get("model_manifest_provenance", ""))
            ):
                raise VerifyError("acceptance model provenance boundary is incomplete")
    projects = root / "projects"
    if projects.is_dir():
        for child in projects.iterdir():
            if child.name.startswith("."):
                continue
            relative = child.relative_to(root).as_posix()
            if child.is_dir() and relative not in registered:
                raise VerifyError(f"unregistered project directory: {relative}")
    configured = text(
        root,
        ["git", "config", "-f", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$"],
    )
    module_paths = {line.split(None, 1)[1] for line in configured.splitlines() if line.strip()}
    if module_paths != registered:
        raise VerifyError(".gitmodules paths do not exactly match frozen gitlink sources")
    configured_keys = run(
        root,
        ["git", "config", "-f", ".gitmodules", "--name-only", "--get-regexp", r"^submodule\."],
    ).stdout.decode().splitlines()
    expected_keys = {
        f"submodule.{source['id']}.{key}"
        for source in lock.get("sources", [])
        if source.get("materialization") == "gitlink"
        for key in ("path", "url", "branch")
    }
    if set(configured_keys) != expected_keys or len(configured_keys) != len(expected_keys):
        raise VerifyError(".gitmodules contains missing, duplicate, or unexpected keys")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument(
        "--allow-transaction",
        help=(
            "permit one matching prepared or crash-recovered committed journal "
            "during transaction finalization"
        ),
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        head, _ = verify_root_state(root, allow_transaction=args.allow_transaction)
        verify_tracked_policy(root)
        current, checkpoint, action = verify_checkpoint(
            root,
            head,
            require_acceptance_artifacts=args.allow_transaction is not None,
        )
        verify_sources(root, current["checkpoint_id"])
    except PendingTransaction as exc:
        print(f"resume verification pending: {exc}", file=sys.stderr)
        return 20
    except VerifyError as exc:
        print(f"resume verification failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"resume verification passed: {checkpoint['id']} state={current['state']} "
        f"next={action['id']} head={head}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
