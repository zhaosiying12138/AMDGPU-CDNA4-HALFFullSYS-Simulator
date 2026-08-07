#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline, read-only verification for an amdgpu-sim handoff."""

from __future__ import annotations

import argparse
import datetime as dt
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
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROJECT_LANES_SCHEMA = "amdgpu-sim.project-lanes.v1"
PROJECT_LANES_PATH = "PROJECT_LANES.json"
SIBLING_RELATIVE_POLICY = {
    "scheme": "sibling-relative-v1",
    "template": "../{project_id}.git",
}
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
        env={
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        },
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
        env={
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        },
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


def valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def valid_branch_name(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    proc = subprocess.run(
        ["git", "check-ref-format", "--branch", value],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        },
    )
    return proc.returncode == 0


def json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifyError(f"invalid historical JSON {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerifyError(f"historical JSON root is not an object: {label}")
    return value


def commit_blob(root: Path, commit: str, relative: str) -> bytes:
    proc = run(
        root,
        ["git", "cat-file", "blob", f"{commit}:{relative}"],
        check=False,
    )
    if proc.returncode:
        detail = proc.stderr.decode(errors="replace").strip()
        raise VerifyError(
            f"historical commit lacks {relative}: {commit}: {detail}"
        )
    return proc.stdout


def checkpoint_commit(root: Path, checkpoint_id: str) -> str:
    """Locate one reachable coordinator commit by its exact audit trailer."""

    candidates: list[str] = []
    commits = text(root, ["git", "rev-list", "HEAD"]).splitlines()
    for commit in commits:
        message = text(root, ["git", "log", "-1", "--format=%B", commit])
        if parse_trailers(message).get("Checkpoint-ID") == [checkpoint_id]:
            candidates.append(commit)
    if len(candidates) != 1:
        raise VerifyError(
            f"checkpoint trailer does not identify one reachable commit: {checkpoint_id}"
        )
    return candidates[0]


def accepted_checkpoint(root: Path, checkpoint_id: str) -> dict[str, Any]:
    commit = checkpoint_commit(root, checkpoint_id)
    relative = f"state/checkpoints/{checkpoint_id}.json"
    historical_blob = commit_blob(root, commit, relative)
    if (root / relative).read_bytes() != historical_blob:
        raise VerifyError(f"checkpoint differs from accepted history: {checkpoint_id}")
    checkpoint = json_bytes(historical_blob, f"{commit}:{relative}")
    if checkpoint.get("id") != checkpoint_id:
        raise VerifyError(f"historical checkpoint identity mismatch: {checkpoint_id}")
    for ids_key, hash_key, directory in (
        ("evidence_ids", "evidence_sha256", "evidence"),
        ("bitlesson_ids", "bitlesson_sha256", "bitlessons"),
    ):
        item_ids = checkpoint.get(ids_key)
        identities = checkpoint.get(hash_key)
        if (
            not isinstance(item_ids, list)
            or len(item_ids) != len(set(item_ids))
            or not all(isinstance(item, str) and item for item in item_ids)
            or (identities is not None and not isinstance(identities, dict))
            or (isinstance(identities, dict) and set(identities) != set(item_ids))
        ):
            raise VerifyError(
                f"historical checkpoint has invalid {ids_key}: {checkpoint_id}"
            )
        for identity in item_ids:
            item_relative = f"state/{directory}/{identity}.json"
            item_blob = commit_blob(root, commit, item_relative)
            expected_sha = identities.get(identity) if isinstance(identities, dict) else None
            if (
                (root / item_relative).read_bytes() != item_blob
                or (
                    expected_sha is not None
                    and (
                        not SHA256_RE.fullmatch(str(expected_sha))
                        or hashlib.sha256(item_blob).hexdigest() != expected_sha
                    )
                )
            ):
                raise VerifyError(
                    f"historical checkpoint {directory} differs from acceptance: {identity}"
                )
    return checkpoint


def verify_checkpoint_history_chain(
    root: Path, current_checkpoint: dict[str, Any]
) -> None:
    expected = current_checkpoint
    seen: set[str] = set()
    while True:
        checkpoint_id = expected.get("id")
        if (
            not isinstance(checkpoint_id, str)
            or not CP_RE.fullmatch(checkpoint_id)
            or checkpoint_id in seen
        ):
            raise VerifyError("checkpoint history chain is malformed or cyclic")
        seen.add(checkpoint_id)
        historical = accepted_checkpoint(root, checkpoint_id)
        if historical != expected:
            raise VerifyError(f"live checkpoint JSON drifted: {checkpoint_id}")
        parent_id = historical.get("parent_checkpoint")
        if parent_id is None:
            return
        if not isinstance(parent_id, str) or not CP_RE.fullmatch(parent_id):
            raise VerifyError(f"checkpoint parent identity is invalid: {checkpoint_id}")
        expected = load_json(
            root / "state" / "checkpoints" / f"{parent_id}.json"
        )


def verify_source_lock_history(
    root: Path,
    lock: dict[str, Any],
    source_checkpoint: dict[str, Any],
) -> str:
    """Anchor the live frozen lock and checkpoint file to their accepted commit."""

    checkpoint_id = lock["frozen_by_checkpoint"]
    commit = checkpoint_commit(root, checkpoint_id)
    checkpoint_relative = f"state/checkpoints/{checkpoint_id}.json"
    historical_checkpoint_blob = commit_blob(root, commit, checkpoint_relative)
    historical_checkpoint = json_bytes(
        historical_checkpoint_blob, f"{commit}:{checkpoint_relative}"
    )
    historical_lock_blob = commit_blob(root, commit, "SOURCE_LOCK.json")
    historical_lock_sha = hashlib.sha256(historical_lock_blob).hexdigest()
    live_lock_blob = (root / "SOURCE_LOCK.json").read_bytes()
    live_checkpoint_blob = (root / checkpoint_relative).read_bytes()
    if (
        historical_checkpoint.get("id") != checkpoint_id
        or source_checkpoint != historical_checkpoint
        or live_checkpoint_blob != historical_checkpoint_blob
        or historical_checkpoint.get("source_lock_sha256") != historical_lock_sha
        or hashlib.sha256(live_lock_blob).hexdigest() != historical_lock_sha
        or live_lock_blob != historical_lock_blob
    ):
        raise VerifyError(
            "SOURCE_LOCK or its freeze checkpoint differs from accepted history"
        )
    return commit


def safe_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts:
        raise VerifyError(f"unsafe relative path: {relative!r}")
    resolved = (root / posix).resolve()
    if resolved != root and root not in resolved.parents:
        raise VerifyError(f"path escapes workspace: {relative!r}")
    return resolved


def verify_journal_initial_children(
    root: Path, journal: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Anchor declared participant origins to the journal's previous root."""

    previous_root = journal.get("previous_root")
    declared = journal.get("declared_children")
    if not isinstance(previous_root, str) or not SHA_RE.fullmatch(previous_root):
        raise VerifyError("transaction journal has no valid previous root")
    previous_commit = run(
        root,
        ["git", "cat-file", "-e", f"{previous_root}^{{commit}}"],
        check=False,
    )
    if previous_commit.returncode:
        raise VerifyError("transaction previous root commit is unavailable offline")
    if not isinstance(declared, dict) or not declared:
        raise VerifyError("transaction journal has no declared child participants")
    seen_paths: set[str] = set()
    for child_id, declaration in sorted(declared.items()):
        if (
            not isinstance(child_id, str)
            or not SOURCE_ID_RE.fullmatch(child_id)
            or not isinstance(declaration, dict)
        ):
            raise VerifyError("transaction journal has an invalid declared child")
        relative = declaration.get("path")
        if not isinstance(relative, str):
            raise VerifyError(f"declared child path is invalid: {child_id}")
        relative_path = PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(relative_path.parts) != 2
            or relative_path.parts[0] != "projects"
            or relative_path.as_posix() != relative
            or relative in seen_paths
        ):
            raise VerifyError(f"declared child path is invalid or duplicate: {child_id}")
        seen_paths.add(relative)
        repo = safe_path(root, relative)
        if "initial_head" not in declaration or "initial_tree" not in declaration:
            raise VerifyError(
                f"declared child lacks its initial identity pair: {child_id}"
            )
        if "target_head" not in declaration or "target_tree" not in declaration:
            raise VerifyError(
                f"declared child lacks its target identity pair: {child_id}"
            )
        target = (declaration.get("target_head"), declaration.get("target_tree"))
        if target != (None, None) and (
            not all(isinstance(value, str) and SHA_RE.fullmatch(value) for value in target)
        ):
            raise VerifyError(f"declared child target identity is invalid: {child_id}")
        listing = text(root, ["git", "ls-tree", previous_root, "--", relative])
        if not listing:
            expected_initial: tuple[str | None, str | None] = (None, None)
        else:
            records = listing.splitlines()
            try:
                metadata, listed_path = records[0].split("\t", 1)
                mode, object_type, initial_head = metadata.split()
            except (IndexError, ValueError) as exc:
                raise VerifyError(
                    f"previous root child entry is malformed: {child_id}"
                ) from exc
            if (
                len(records) != 1
                or listed_path != relative
                or mode != "160000"
                or object_type != "commit"
                or not SHA_RE.fullmatch(initial_head)
            ):
                raise VerifyError(
                    f"previous root child entry is not an exact gitlink: {child_id}"
                )
            if not repo.is_dir():
                raise VerifyError(
                    f"previous root child worktree is unavailable: {child_id}"
                )
            tree_proc = run(
                root,
                ["git", "rev-parse", f"{initial_head}^{{tree}}"],
                cwd=repo,
                check=False,
            )
            initial_tree = tree_proc.stdout.decode().strip()
            if tree_proc.returncode or not SHA_RE.fullmatch(initial_tree):
                raise VerifyError(
                    f"previous root child commit is unavailable offline: {child_id}"
                )
            expected_initial = (initial_head, initial_tree)
        observed_initial = (
            declaration.get("initial_head"),
            declaration.get("initial_tree"),
        )
        if observed_initial != expected_initial:
            raise VerifyError(
                f"declared child initial identity mismatch: {child_id}"
            )
    return declared


def verify_coordinator_participant_gitlinks(
    root: Path,
    previous_root: str,
    coordinator_head: str,
    declared: dict[str, dict[str, Any]],
) -> None:
    """Bind coordinator project gitlink changes to the journal participant set."""

    raw = run(
        root,
        [
            "git",
            "diff",
            "--raw",
            "-z",
            "--no-renames",
            previous_root,
            coordinator_head,
            "--",
            "projects",
        ],
    ).stdout
    parts = raw.split(b"\0")
    if parts[-1:] == [b""]:
        parts.pop()
    if len(parts) % 2:
        raise VerifyError("coordinator projects diff has malformed raw records")
    changed: set[str] = set()
    for offset in range(0, len(parts), 2):
        fields = parts[offset].split()
        if len(fields) != 5 or not fields[0].startswith(b":"):
            raise VerifyError("coordinator projects diff has malformed metadata")
        old_mode = fields[0][1:]
        new_mode = fields[1]
        try:
            relative = parts[offset + 1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VerifyError("coordinator project path is not UTF-8") from exc
        posix = PurePosixPath(relative)
        if (
            posix.is_absolute()
            or len(posix.parts) != 2
            or posix.parts[0] != "projects"
            or posix.as_posix() != relative
            or old_mode not in {b"000000", b"160000"}
            or new_mode != b"160000"
        ):
            raise VerifyError(
                f"coordinator project change is not a canonical gitlink: {relative}"
            )
        if relative in changed:
            raise VerifyError(f"coordinator project path is duplicated: {relative}")
        changed.add(relative)
    expected = {entry["path"] for entry in declared.values()}
    if changed != expected:
        raise VerifyError(
            "coordinator gitlinks differ from transaction participants; "
            f"missing={sorted(expected - changed)}, extra={sorted(changed - expected)}"
        )


def describe_journal_children(
    root: Path,
    declared: dict[str, dict[str, Any]],
    recorded: Any,
) -> list[str]:
    """Describe every participant relative to its immutable start and target."""

    recorded_children = recorded if isinstance(recorded, dict) else {}
    descriptions: list[str] = []
    for child_id, declaration in sorted(declared.items()):
        relative = declaration["path"]
        repo = safe_path(root, relative)
        initial = (
            declaration.get("initial_head"),
            declaration.get("initial_tree"),
        )
        target = (
            declaration.get("target_head"),
            declaration.get("target_tree"),
        )
        recorded_child = recorded_children.get(child_id)
        if target == (None, None) and isinstance(recorded_child, dict):
            target = (recorded_child.get("head"), recorded_child.get("tree"))
        initial_label = (
            "absent" if initial == (None, None) else f"{initial[0]}:{initial[1]}"
        )
        target_label = (
            "unrecorded" if target == (None, None) else f"{target[0]}:{target[1]}"
        )
        if not repo.is_dir():
            descriptions.append(
                f"  child {child_id}: missing {relative} initial={initial_label} "
                f"target={target_label}"
            )
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
        actual = (head_value, tree_value)
        readable = actual_head.returncode == 0 and actual_tree.returncode == 0
        initial_match = readable and initial != (None, None) and actual == initial
        target_available = target != (None, None)
        target_match = readable and target_available and actual == target
        if initial_match:
            position = "initial"
        elif target_match:
            position = "target"
        else:
            position = "other"
        descriptions.append(
            f"  child {child_id}: head={head_value or 'unreadable'} "
            f"tree={tree_value or 'unreadable'} clean={not bool(dirty.stdout)} "
            f"position={position} initial={initial_label} target={target_label} "
            f"initial_identity_match={initial_match} "
            f"target_identity_match={target_match if target_available else 'unavailable'}"
        )
    return descriptions


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
            declared = verify_journal_initial_children(root, allowed)
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
            verify_coordinator_participant_gitlinks(
                root,
                allowed["previous_root"],
                coordinator_head,
                declared,
            )
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
                declared = verify_journal_initial_children(root, journal)
                descriptions.extend(
                    describe_journal_children(
                        root, declared, journal.get("expected_children")
                    )
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
        PROJECT_LANES_PATH: checkpoint.get("project_lanes_sha256"),
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


def authored_lanes(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and return immutable project-authored baseline declarations."""

    if set(registry) != {"schema", "registry_revision", "url_policy", "lanes"}:
        raise VerifyError("PROJECT_LANES has missing or unknown fields")
    if registry.get("schema") != PROJECT_LANES_SCHEMA:
        raise VerifyError("unsupported PROJECT_LANES schema")
    if (
        not isinstance(registry.get("registry_revision"), int)
        or isinstance(registry.get("registry_revision"), bool)
        or registry["registry_revision"] != 1
    ):
        raise VerifyError("unsupported PROJECT_LANES registry revision")
    if registry.get("url_policy") != SIBLING_RELATIVE_POLICY:
        raise VerifyError("PROJECT_LANES sibling-relative URL policy mismatch")
    lanes = registry.get("lanes")
    if not isinstance(lanes, list):
        raise VerifyError("PROJECT_LANES lanes is not a list")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_admin: set[str] = set()
    for lane in lanes:
        if not isinstance(lane, dict):
            raise VerifyError("PROJECT_LANES entries must be objects")
        lane_id = lane.get("id")
        forbidden_current = {"head", "tree", "work_head", "work_tree"}.intersection(lane)
        if forbidden_current:
            raise VerifyError(
                f"PROJECT_LANES must not own current head/tree: {lane_id}:{sorted(forbidden_current)}"
            )
        expected_lane_fields = {
            "id",
            "ownership",
            "role",
            "path",
            "materialization",
            "origin",
            "baseline_commit",
            "baseline_tree",
            "baseline_tag",
            "baseline_tag_object",
            "baseline_tag_payload",
            "baseline_tag_payload_sha256",
            "baseline_created_at",
            "baseline_checkpoint_id",
            "baseline_evidence_id",
            "baseline_commit_trailers",
            "administrative_git_dir",
            "license",
        }
        if set(lane) != expected_lane_fields:
            raise VerifyError("authored lane has missing or unknown fields")
        if not isinstance(lane_id, str) or not SOURCE_ID_RE.fullmatch(lane_id):
            raise VerifyError(f"authored lane has an unsafe project id: {lane_id!r}")
        expected_path = f"projects/{lane_id}"
        expected_admin = f"modules/{lane_id}"
        expected_url = SIBLING_RELATIVE_POLICY["template"].format(project_id=lane_id)
        origin = lane.get("origin")
        license_metadata = lane.get("license")
        commit_trailers = lane.get("baseline_commit_trailers")
        if not isinstance(origin, dict) or set(origin) != {
            "policy",
            "remote",
            "url",
            "push_url",
            "reachability",
            "branch",
        }:
            raise VerifyError("authored lane origin has missing or unknown fields")
        if not isinstance(license_metadata, dict) or set(license_metadata) != {
            "spdx_id",
            "path",
            "sha256",
        }:
            raise VerifyError("authored lane license has missing or unknown fields")
        if not isinstance(commit_trailers, dict) or set(commit_trailers) != {
            "checkpoint_id",
            "goal_id",
            "plan_revision",
            "source_lock_sha256",
            "evidence_manifest_sha256",
            "change_kind",
            "baseline_commit_marker",
        }:
            raise VerifyError(
                "authored lane baseline commit trailers have missing or unknown fields"
            )
        baseline_fields = (
            "baseline_commit",
            "baseline_tree",
            "baseline_tag_object",
        )
        if (
            lane.get("ownership") != "project-authored"
            or not isinstance(lane.get("role"), str)
            or not SOURCE_ID_RE.fullmatch(lane["role"])
            or lane.get("materialization") != "gitlink"
            or lane.get("path") != expected_path
            or lane.get("administrative_git_dir") != expected_admin
            or origin.get("policy") != "sibling-relative-v1"
            or origin.get("remote") != "origin"
            or origin.get("url") != expected_url
            or origin.get("push_url") != "no_push"
            or origin.get("reachability") != "not-asserted"
            or not isinstance(origin.get("branch"), str)
            or not valid_branch_name(origin["branch"])
            or license_metadata.get("spdx_id") != "GPL-3.0-or-later"
            or license_metadata.get("path") != "LICENSE"
            or not SHA256_RE.fullmatch(str(license_metadata.get("sha256", "")))
            or not all(SHA_RE.fullmatch(str(lane.get(field, ""))) for field in baseline_fields)
            or lane.get("baseline_tag")
            != f"project-baseline/{lane_id}/{lane.get('baseline_commit')}"
            or not isinstance(lane.get("baseline_tag_payload"), str)
            or not lane["baseline_tag_payload"]
            or not SHA256_RE.fullmatch(str(lane.get("baseline_tag_payload_sha256", "")))
            or not valid_timestamp(lane.get("baseline_created_at"))
            or not CP_RE.fullmatch(str(lane.get("baseline_checkpoint_id", "")))
            or not EV_RE.fullmatch(str(lane.get("baseline_evidence_id", "")))
            or commit_trailers.get("checkpoint_id")
            != lane.get("baseline_checkpoint_id")
            or not isinstance(commit_trailers.get("goal_id"), str)
            or not commit_trailers["goal_id"]
            or not isinstance(commit_trailers.get("plan_revision"), int)
            or isinstance(commit_trailers.get("plan_revision"), bool)
            or commit_trailers["plan_revision"] < 1
            or not SHA256_RE.fullmatch(
                str(commit_trailers.get("source_lock_sha256", ""))
            )
            or not SHA256_RE.fullmatch(
                str(commit_trailers.get("evidence_manifest_sha256", ""))
            )
            or commit_trailers.get("change_kind") != "baseline"
            or commit_trailers.get("baseline_commit_marker") != "N/A"
        ):
            raise VerifyError(f"authored lane identity is incomplete: {lane_id}")
        if lane_id in seen_ids or expected_path in seen_paths or expected_admin in seen_admin:
            raise VerifyError(f"duplicate authored lane identity: {lane_id}")
        seen_ids.add(lane_id)
        seen_paths.add(expected_path)
        seen_admin.add(expected_admin)
        verify_authored_tag_metadata(lane)
    return lanes


def verify_authored_tag_metadata(lane: dict[str, Any]) -> None:
    expected = {
        "Project-ID": lane["id"],
        "Project-URL": lane["origin"]["url"],
        "Origin-Policy": lane["origin"]["policy"],
        "Baseline-Commit": lane["baseline_commit"],
        "Baseline-Tree": lane["baseline_tree"],
        "License-SPDX": lane["license"]["spdx_id"],
        "Created-At": lane["baseline_created_at"],
    }
    observed: dict[str, list[str]] = {key: [] for key in expected}
    for line in lane["baseline_tag_payload"].splitlines():
        match = re.fullmatch(r"([A-Za-z0-9-]+): (.*)", line)
        if match and match.group(1) in observed:
            observed[match.group(1)].append(match.group(2))
    for key, value in expected.items():
        if observed[key] != [value]:
            raise VerifyError(
                f"authored baseline tag has missing, duplicate, or mismatched {key}: {lane['id']}"
            )


def verify_authored_lane_history(
    root: Path,
    registry: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Prove authored baseline declarations are append-only from first acceptance."""

    current = {lane["id"]: lane for lane in authored_lanes(registry)}
    commits = text(
        root,
        ["git", "rev-list", "--reverse", "HEAD", "--", PROJECT_LANES_PATH],
    ).splitlines()
    if current and not commits:
        raise VerifyError("PROJECT_LANES has no reachable acceptance history")
    accepted: dict[str, dict[str, Any]] = {}
    accepted_at: dict[str, str] = {}
    prior_ids: set[str] = set()
    for commit in commits:
        historical_blob = commit_blob(root, commit, PROJECT_LANES_PATH)
        historical_registry = json_bytes(
            historical_blob, f"{commit}:{PROJECT_LANES_PATH}"
        )
        historical_lanes = {
            lane["id"]: lane for lane in authored_lanes(historical_registry)
        }
        if not prior_ids.issubset(historical_lanes):
            raise VerifyError("PROJECT_LANES history deletes an accepted authored lane")
        for lane_id in prior_ids:
            if historical_lanes[lane_id] != accepted[lane_id]:
                raise VerifyError(
                    f"PROJECT_LANES history rewrites an authored baseline: {lane_id}"
                )
        new_ids = set(historical_lanes) - prior_ids
        if not new_ids:
            raise VerifyError(
                "PROJECT_LANES changed without appending an authored lane"
            )
        message = text(root, ["git", "log", "-1", "--format=%B", commit])
        checkpoint_values = parse_trailers(message).get("Checkpoint-ID", [])
        if len(checkpoint_values) != 1 or not CP_RE.fullmatch(checkpoint_values[0]):
            raise VerifyError(
                "PROJECT_LANES acceptance commit has no unique Checkpoint-ID trailer"
            )
        checkpoint_id = checkpoint_values[0]
        if checkpoint_commit(root, checkpoint_id) != commit:
            raise VerifyError(
                f"PROJECT_LANES acceptance checkpoint is ambiguous: {checkpoint_id}"
            )
        checkpoint_relative = f"state/checkpoints/{checkpoint_id}.json"
        checkpoint_blob = commit_blob(root, commit, checkpoint_relative)
        checkpoint = json_bytes(checkpoint_blob, f"{commit}:{checkpoint_relative}")
        historical_registry_sha = hashlib.sha256(historical_blob).hexdigest()
        if (
            checkpoint.get("id") != checkpoint_id
            or checkpoint.get("project_lanes_sha256") != historical_registry_sha
            or (root / checkpoint_relative).read_bytes() != checkpoint_blob
        ):
            raise VerifyError(
                f"authored lane checkpoint does not bind its registry: {checkpoint_id}"
            )
        for lane_id in new_ids:
            lane = historical_lanes[lane_id]
            if lane["baseline_checkpoint_id"] != checkpoint_id:
                raise VerifyError(
                    f"authored lane baseline checkpoint is not its first declaration: {lane_id}"
                )
            evidence_id = lane["baseline_evidence_id"]
            evidence_sha = lane["baseline_commit_trailers"][
                "evidence_manifest_sha256"
            ]
            evidence_ids = checkpoint.get("evidence_ids")
            evidence_hashes = checkpoint.get("evidence_sha256")
            manifest_id = checkpoint.get("evidence_manifest_id")
            if (
                not isinstance(evidence_ids, list)
                or evidence_ids.count(evidence_id) != 1
                or not isinstance(evidence_hashes, dict)
                or evidence_hashes.get(evidence_id) != evidence_sha
                or manifest_id == evidence_id
                or not isinstance(manifest_id, str)
                or checkpoint.get("evidence_manifest_sha256")
                != evidence_hashes.get(manifest_id)
            ):
                raise VerifyError(
                    f"authored baseline evidence is not layered into its checkpoint: {lane_id}"
                )
            evidence_relative = f"state/evidence/{evidence_id}.json"
            evidence_blob = commit_blob(root, commit, evidence_relative)
            if hashlib.sha256(evidence_blob).hexdigest() != evidence_sha:
                raise VerifyError(
                    f"authored baseline evidence blob mismatch: {lane_id}"
                )
            evidence = json_bytes(evidence_blob, f"{commit}:{evidence_relative}")
            if (
                evidence.get("id") != evidence_id
                or evidence.get("checkpoint_id") != checkpoint_id
            ):
                raise VerifyError(
                    f"authored baseline evidence identity mismatch: {lane_id}"
                )
            manifest_relative = f"state/evidence/{manifest_id}.json"
            manifest_blob = commit_blob(root, commit, manifest_relative)
            if (
                hashlib.sha256(manifest_blob).hexdigest()
                != checkpoint["evidence_manifest_sha256"]
            ):
                raise VerifyError(
                    f"authored baseline umbrella evidence blob mismatch: {lane_id}"
                )
            manifest = json_bytes(manifest_blob, f"{commit}:{manifest_relative}")
            if (
                manifest.get("id") != manifest_id
                or manifest.get("checkpoint_id") != checkpoint_id
                or not isinstance(manifest.get("includes"), dict)
                or manifest["includes"].get(evidence_id) != evidence_sha
            ):
                raise VerifyError(
                    f"authored baseline evidence is absent from umbrella manifest: {lane_id}"
                )
            if (
                (root / evidence_relative).read_bytes() != evidence_blob
                or (root / manifest_relative).read_bytes() != manifest_blob
            ):
                raise VerifyError(
                    f"authored baseline evidence differs from accepted history: {lane_id}"
                )
            accepted[lane_id] = lane
            accepted_at[lane_id] = commit
        prior_ids = set(historical_lanes)
    if set(current) != set(accepted):
        raise VerifyError("PROJECT_LANES live set differs from append-only history")
    for lane_id, lane in current.items():
        if lane != accepted[lane_id]:
            raise VerifyError(
                f"PROJECT_LANES live baseline differs from first acceptance: {lane_id}"
            )
    return {
        lane_id: {"lane": accepted[lane_id], "commit": accepted_at[lane_id]}
        for lane_id in sorted(accepted)
    }


def verify_current_progress_commits(
    root: Path,
    checkpoint: dict[str, Any],
    plan_revision: int,
    authority: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Bind every new child commit to the current two-phase checkpoint."""

    baseline = authority["baseline_commit"]
    head = current["head"]
    parent_checkpoint_id = checkpoint.get("parent_checkpoint")
    if not isinstance(parent_checkpoint_id, str) or not CP_RE.fullmatch(
        parent_checkpoint_id
    ):
        raise VerifyError(f"progressed lane has no parent checkpoint: {authority['id']}")
    parent_checkpoint = accepted_checkpoint(root, parent_checkpoint_id)
    parent_repositories = parent_checkpoint.get("repositories")
    if not isinstance(parent_repositories, list):
        raise VerifyError("parent checkpoint repositories is not a list")
    previous_records = [
        item
        for item in parent_repositories
        if isinstance(item, dict) and item.get("id") == authority["id"]
    ]
    if not previous_records:
        if head == baseline:
            return
        raise VerifyError(
            f"new lane advances beyond its accepted baseline: {authority['id']}"
        )
    if len(previous_records) != 1:
        raise VerifyError(
            f"progressed lane has no unique previous head: {authority['id']}"
        )
    previous = previous_records[0].get("head")
    if not isinstance(previous, str) or not SHA_RE.fullmatch(previous):
        raise VerifyError(f"progressed lane previous head is invalid: {authority['id']}")
    repo = safe_path(root, authority["path"])
    if previous == head:
        return
    ancestor = run(
        root,
        ["git", "merge-base", "--is-ancestor", previous, head],
        cwd=repo,
        check=False,
    )
    if ancestor.returncode:
        raise VerifyError(
            f"current lane head does not descend from previous checkpoint: {authority['id']}"
        )
    commits = text(
        root,
        ["git", "rev-list", "--reverse", "--topo-order", f"{previous}..{head}"],
        cwd=repo,
    ).splitlines()
    if not commits or commits[-1] != head:
        raise VerifyError(f"cannot enumerate current lane progress: {authority['id']}")
    evidence_hashes = checkpoint.get("evidence_sha256")
    if not isinstance(evidence_hashes, dict):
        raise VerifyError("current checkpoint evidence map is absent")
    allowed_evidence = {
        value for value in evidence_hashes.values() if isinstance(value, str)
    }
    expected = {
        "Checkpoint-ID": checkpoint["id"],
        "Goal-ID": checkpoint["goal_id"],
        "Plan-Revision": str(plan_revision),
        "Source-Lock-SHA256": checkpoint["source_lock_sha256"],
        "Change-Kind": checkpoint["change_kind"],
        "Baseline-Commit": baseline,
    }
    for commit in commits:
        message = text(root, ["git", "log", "-1", "--format=%B", commit], cwd=repo)
        try:
            check_commit_message(message)
        except MessageError as exc:
            raise VerifyError(
                f"progress commit message violates audit policy: {authority['id']}:{commit}: {exc}"
            ) from exc
        trailers = parse_trailers(message)
        for key, value in expected.items():
            if trailers.get(key) != [value]:
                raise VerifyError(
                    f"progress commit has mismatched {key}: {authority['id']}:{commit}"
                )
        evidence_values = trailers.get("Evidence-Manifest-SHA256")
        if (
            not isinstance(evidence_values, list)
            or len(evidence_values) != 1
            or evidence_values[0] not in allowed_evidence
        ):
            raise VerifyError(
                f"progress commit evidence is not in current checkpoint: {authority['id']}:{commit}"
            )


def verify_checkpoint_repositories(
    checkpoint: dict[str, Any],
    lock: dict[str, Any],
    registry: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
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
    lanes = authored_lanes(
        (
            {
            "schema": PROJECT_LANES_SCHEMA,
            "registry_revision": 1,
            "url_policy": SIBLING_RELATIVE_POLICY,
            "lanes": [],
            }
            if registry is None
            else registry
        )
    )
    repository_ids = [
        repository.get("id")
        for repository in repositories
        if isinstance(repository, dict)
    ]
    authorities = [*git_sources, *lanes]
    source_ids = [source.get("id") for source in authorities]
    source_paths = [source.get("path") for source in authorities]
    source_admin = [source.get("administrative_git_dir") for source in authorities]
    if (
        not git_sources
        or len(repository_ids) != len(repositories)
        or len(repository_ids) != len(set(repository_ids))
        or any(not isinstance(source_id, str) for source_id in source_ids)
        or len(source_ids) != len(set(source_ids))
        or any(not isinstance(path, str) for path in source_paths)
        or len(source_paths) != len(set(source_paths))
        or any(not isinstance(path, str) for path in source_admin)
        or len(source_admin) != len(set(source_admin))
        or set(repository_ids) != set(source_ids)
    ):
        raise VerifyError(
            "checkpoint repository set does not match upstream/authored lane union"
        )
    by_id = {repository["id"]: repository for repository in repositories}
    for source in authorities:
        required_source_fields = (
            "id",
            "path",
            "baseline_commit",
            "baseline_tree",
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
        if source in git_sources and (
            source.get("work_head") != source.get("baseline_commit")
            or source.get("work_tree") != source.get("baseline_tree")
        ):
            raise VerifyError(
                f"frozen upstream source current identity is not its baseline: {source['id']}"
            )
        expected_baseline = {
            "id": source["id"],
            "path": source["path"],
            "baseline_commit": source["baseline_commit"],
            "baseline_tree": source["baseline_tree"],
            "baseline_tag": source["baseline_tag"],
            "baseline_tag_object": source["baseline_tag_object"],
            "administrative_git_dir": source["administrative_git_dir"],
        }
        repository = by_id[source["id"]]
        if (
            set(repository)
            != {*expected_baseline, "head", "tree", "clean"}
            or any(repository.get(key) != value for key, value in expected_baseline.items())
            or not SHA_RE.fullmatch(str(repository.get("head", "")))
            or not SHA_RE.fullmatch(str(repository.get("tree", "")))
            or repository.get("clean") is not True
        ):
            raise VerifyError(
                f"checkpoint repository identity mismatch: {source['id']}"
            )
    return by_id


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


def verify_child(
    root: Path,
    source: dict[str, Any],
    current: dict[str, Any] | None = None,
) -> None:
    source_id = source["id"]
    if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
        raise VerifyError(f"source has an unsafe submodule name: {source_id!r}")
    relative = source["path"]
    repo = safe_path(root, relative)
    if not repo.is_dir():
        raise VerifyError(f"materialized source is missing: {relative}")
    expected_head = (current or {}).get("head", source.get("work_head"))
    baseline = source.get("baseline_commit")
    baseline_tree = source.get("baseline_tree")
    work_tree = (current or {}).get("tree", source.get("work_tree", baseline_tree))
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
        root_common_dir = root / root_common_dir
    modules_dir = root_common_dir / "modules"
    expected_admin_path = modules_dir / source_id
    if modules_dir.is_symlink() or expected_admin_path.is_symlink():
        raise VerifyError(f"child administrative Git path is a symlink: {source_id}")
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


def verify_authored_child(
    root: Path,
    lane: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Verify a locally authored lane without attributing upstream provenance to it."""

    lane_id = lane["id"]
    relative = lane["path"]
    repo = safe_path(root, relative)
    if not repo.is_dir():
        raise VerifyError(f"materialized authored lane is missing: {relative}")
    baseline = lane["baseline_commit"]
    baseline_tree = lane["baseline_tree"]
    expected_head = current["head"]
    expected_tree = current["tree"]
    origin = lane["origin"]
    index = text(root, ["git", "ls-files", "--stage", "--", relative])
    fields = index.split()
    if len(fields) < 4 or fields[0] != "160000" or fields[1] != expected_head:
        raise VerifyError(f"root gitlink mismatch for authored lane {lane_id}")
    if gitmodule_value(root, lane_id, "path") != relative:
        raise VerifyError(f".gitmodules path mismatch for authored lane {lane_id}")
    if gitmodule_value(root, lane_id, "url") != origin["url"]:
        raise VerifyError(f".gitmodules canonical URL mismatch for authored lane {lane_id}")
    if gitmodule_value(root, lane_id, "branch") != origin["branch"]:
        raise VerifyError(f".gitmodules branch mismatch for authored lane {lane_id}")
    if text(root, ["git", "rev-parse", "HEAD"], cwd=repo) != expected_head:
        raise VerifyError(f"authored lane HEAD mismatch: {lane_id}")
    if text(root, ["git", "rev-parse", "HEAD^{tree}"], cwd=repo) != expected_tree:
        raise VerifyError(f"authored lane tree mismatch: {lane_id}")
    if text(root, ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repo):
        raise VerifyError(f"authored lane worktree is dirty: {lane_id}")
    if text(root, ["git", "remote"], cwd=repo).splitlines() != [origin["remote"]]:
        raise VerifyError(f"authored lane remote set mismatch: {lane_id}")
    fetch_urls = run(
        root,
        ["git", "config", "--get-all", f"remote.{origin['remote']}.url"],
        cwd=repo,
        check=False,
    )
    push_urls = run(
        root,
        ["git", "config", "--get-all", f"remote.{origin['remote']}.pushurl"],
        cwd=repo,
        check=False,
    )
    if fetch_urls.returncode or fetch_urls.stdout.decode().splitlines() != [origin["url"]]:
        raise VerifyError(f"authored lane origin URL is not a singleton: {lane_id}")
    if push_urls.returncode or push_urls.stdout.decode().splitlines() != [origin["push_url"]]:
        raise VerifyError(f"authored lane push policy mismatch: {lane_id}")
    expected_hooks = os.path.relpath(root / ".githooks", repo)
    if text(root, ["git", "config", "--get", "core.hooksPath"], cwd=repo) != expected_hooks:
        raise VerifyError(f"authored lane hooksPath mismatch: {lane_id}")
    if text(root, ["git", "rev-parse", "--is-shallow-repository"], cwd=repo) != "false":
        raise VerifyError(f"authored lane repository is shallow: {lane_id}")
    sparse = run(
        root,
        ["git", "config", "--bool", "core.sparseCheckout"],
        cwd=repo,
        check=False,
    )
    if sparse.returncode not in (0, 1) or sparse.stdout.decode().strip() == "true":
        raise VerifyError(f"authored lane uses sparse checkout: {lane_id}")
    indexed = run(root, ["git", "ls-files", "-v", "-z"], cwd=repo).stdout
    if any(entry and entry[:1] in {b"S", b"s"} for entry in indexed.split(b"\0")):
        raise VerifyError(f"authored lane has skip-worktree entries: {lane_id}")
    child_git_dir = Path(text(root, ["git", "rev-parse", "--absolute-git-dir"], cwd=repo))
    child_common_dir = Path(text(root, ["git", "rev-parse", "--git-common-dir"], cwd=repo))
    child_common_dir = (
        (repo / child_common_dir).resolve()
        if not child_common_dir.is_absolute()
        else child_common_dir.resolve()
    )
    root_common_dir = Path(text(root, ["git", "rev-parse", "--git-common-dir"]))
    if not root_common_dir.is_absolute():
        root_common_dir = root / root_common_dir
    modules_dir = root_common_dir / "modules"
    expected_admin_path = modules_dir / lane_id
    if modules_dir.is_symlink() or expected_admin_path.is_symlink():
        raise VerifyError(
            f"authored lane administrative Git path is a symlink: {lane_id}"
        )
    root_common_dir = root_common_dir.resolve()
    expected_admin_relative = f"modules/{lane_id}"
    expected_admin = (root_common_dir / expected_admin_relative).resolve()
    expected_gitfile = f"gitdir: {os.path.relpath(expected_admin, repo)}\n"
    if (
        lane["administrative_git_dir"] != expected_admin_relative
        or not (repo / ".git").is_file()
        or (repo / ".git").is_symlink()
        or (repo / ".git").read_text(encoding="utf-8") != expected_gitfile
        or child_git_dir.resolve() != expected_admin
        or child_common_dir != expected_admin
    ):
        raise VerifyError(f"authored lane administrative Git path mismatch: {lane_id}")
    for marker in (
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "BISECT_LOG",
        "rebase-merge",
        "rebase-apply",
    ):
        if (child_git_dir / marker).exists():
            raise VerifyError(f"authored lane Git operation is in progress: {lane_id}:{marker}")
    if (child_git_dir / "info" / "grafts").is_file():
        raise VerifyError(f"authored lane history uses a grafts file: {lane_id}")
    if text(root, ["git", "replace", "-l"], cwd=repo):
        raise VerifyError(f"authored lane history uses replacement objects: {lane_id}")
    alternates = child_git_dir / "objects" / "info" / "alternates"
    if alternates.is_file() and alternates.read_text(encoding="utf-8").strip():
        raise VerifyError(f"authored lane depends on alternates: {lane_id}")
    verify_annotated_tag(
        root,
        repo,
        lane_id,
        commit=baseline,
        tree=baseline_tree,
        tag=lane["baseline_tag"],
        tag_object=lane["baseline_tag_object"],
        tag_payload=lane["baseline_tag_payload"],
        tag_payload_sha256=lane["baseline_tag_payload_sha256"],
        required_lines=[
            f"Project-ID: {lane_id}",
            f"Project-URL: {origin['url']}",
            "Origin-Policy: sibling-relative-v1",
            f"Baseline-Commit: {baseline}",
            f"Baseline-Tree: {baseline_tree}",
        ],
    )
    commit_message = text(
        root, ["git", "log", "-1", "--format=%B", baseline], cwd=repo
    )
    try:
        check_commit_message(commit_message)
    except MessageError as exc:
        raise VerifyError(
            f"authored baseline commit message violates audit policy: {lane_id}: {exc}"
        ) from exc
    trailers = parse_trailers(commit_message)
    commit_identity = lane["baseline_commit_trailers"]
    expected_trailers = {
        "Checkpoint-ID": commit_identity["checkpoint_id"],
        "Goal-ID": commit_identity["goal_id"],
        "Plan-Revision": str(commit_identity["plan_revision"]),
        "Source-Lock-SHA256": commit_identity["source_lock_sha256"],
        "Evidence-Manifest-SHA256": commit_identity["evidence_manifest_sha256"],
        "Change-Kind": commit_identity["change_kind"],
        "Baseline-Commit": commit_identity["baseline_commit_marker"],
    }
    for key, expected in expected_trailers.items():
        if trailers.get(key) != [expected]:
            raise VerifyError(
                f"authored baseline commit has mismatched {key} trailer: {lane_id}"
            )
    if commit_identity["source_lock_sha256"] != digest(root / "SOURCE_LOCK.json"):
        raise VerifyError(f"authored baseline commit binds another source lock: {lane_id}")
    ancestor = run(
        root,
        ["git", "merge-base", "--is-ancestor", baseline, expected_head],
        cwd=repo,
        check=False,
    )
    if ancestor.returncode:
        raise VerifyError(f"authored lane head is not a baseline descendant: {lane_id}")
    license_metadata = lane["license"]
    for revision, label in ((baseline, "baseline"), (expected_head, "current")):
        license_result = run(
            root,
            ["git", "cat-file", "blob", f"{revision}:{license_metadata['path']}"],
            cwd=repo,
            check=False,
        )
        if (
            license_result.returncode
            or hashlib.sha256(license_result.stdout).hexdigest()
            != license_metadata["sha256"]
        ):
            raise VerifyError(
                f"authored lane {label} license hash mismatch: {lane_id}"
            )
    worktree_license = repo / license_metadata["path"]
    if (
        not worktree_license.is_file()
        or worktree_license.is_symlink()
        or digest(worktree_license) != license_metadata["sha256"]
    ):
        raise VerifyError(f"authored lane worktree license hash mismatch: {lane_id}")
    run_discard_stdout(root, ["git", "archive", "--format=tar", expected_head], cwd=repo)
    run_discard_stdout(root, ["git", "fsck", "--connectivity-only"], cwd=repo)
    tree_entries = run(
        root, ["git", "ls-tree", "-r", "-z", expected_head], cwd=repo
    ).stdout
    gitlinks = [
        entry
        for entry in tree_entries.split(b"\0")
        if entry and entry.split(b" ", 1)[0] == b"160000"
    ]
    module_blob = run(
        root,
        ["git", "cat-file", "-e", f"{expected_head}:.gitmodules"],
        cwd=repo,
        check=False,
    )
    declarations = b""
    if module_blob.returncode == 0:
        declaration_result = run(
            root,
            [
                "git",
                "config",
                "--blob",
                f"{expected_head}:.gitmodules",
                "--get-regexp",
                r"^submodule\..*\.path$",
            ],
            cwd=repo,
            check=False,
        )
        if declaration_result.returncode not in (0, 1):
            detail = declaration_result.stderr.decode(errors="replace").strip()
            raise VerifyError(
                f"cannot inspect authored lane submodule declarations: {lane_id}: {detail}"
            )
        declarations = declaration_result.stdout
    elif module_blob.returncode != 128:
        raise VerifyError(f"cannot inspect authored lane .gitmodules: {lane_id}")
    if gitlinks or declarations:
        raise VerifyError(
            f"authored lane unexpectedly declares nested submodules: {lane_id}"
        )
    if text(root, ["git", "submodule", "status", "--recursive"], cwd=repo):
        raise VerifyError(f"authored lane unexpectedly declares nested submodules: {lane_id}")


def verify_root_gitmodules(
    root: Path,
    sources: list[dict[str, Any]],
    lanes: list[dict[str, Any]],
    registered: set[str],
) -> None:
    configured = text(
        root,
        ["git", "config", "-f", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$"],
    )
    module_paths = {
        line.split(None, 1)[1] for line in configured.splitlines() if line.strip()
    }
    if module_paths != registered:
        raise VerifyError(
            ".gitmodules paths do not exactly match upstream/authored lane union"
        )
    configured_keys = run(
        root,
        ["git", "config", "-f", ".gitmodules", "--name-only", "--get-regexp", r".*"],
    ).stdout.decode().splitlines()
    expected_keys = {
        f"submodule.{source['id']}.{key}"
        for source in sources
        if source.get("materialization") == "gitlink"
        for key in ("path", "url", "branch")
    }
    expected_keys.update(
        f"submodule.{lane['id']}.{key}"
        for lane in lanes
        for key in ("path", "url", "branch")
    )
    if set(configured_keys) != expected_keys or len(configured_keys) != len(expected_keys):
        raise VerifyError(".gitmodules contains missing, duplicate, or unexpected keys")


def verify_sources(root: Path, checkpoint_id: str) -> None:
    lock = load_json(root / "SOURCE_LOCK.json")
    registry = load_json(root / PROJECT_LANES_PATH)
    if lock.get("schema") != "amdgpu-sim.source-lock.v1":
        raise VerifyError("unsupported SOURCE_LOCK schema")
    frozen_by_checkpoint = lock.get("frozen_by_checkpoint")
    if (
        lock.get("status") != "frozen"
        or not LOCK_ID_RE.fullmatch(str(lock.get("lock_id", "")))
        or not CP_RE.fullmatch(str(frozen_by_checkpoint or ""))
    ):
        raise VerifyError("SOURCE_LOCK is not frozen")
    checkpoint = load_json(root / "state" / "checkpoints" / f"{checkpoint_id}.json")
    source_checkpoint = load_json(
        root / "state" / "checkpoints" / f"{frozen_by_checkpoint}.json"
    )
    current_repositories = verify_checkpoint_repositories(checkpoint, lock, registry)
    if source_checkpoint.get("id") != frozen_by_checkpoint:
        raise VerifyError("SOURCE_LOCK freeze checkpoint identity mismatch")
    parent_commit = lock.get("accepted_parent_root_commit")
    parent_hash = lock.get("accepted_parent_source_lock_sha256")
    candidate_hash = lock.get("pre_freeze_candidate_sha256")
    if (
        not isinstance(parent_commit, str)
        or not SHA_RE.match(parent_commit)
        or parent_commit != source_checkpoint.get("root_parent_commit")
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
    verify_source_lock_history(root, lock, source_checkpoint)
    verify_checkpoint_history_chain(root, checkpoint)
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
        or evidence.get("checkpoint_id") != frozen_by_checkpoint
        or evidence.get("type") != "official-source-resolution"
        or digest(evidence_path) != evidence_sha
        or (source_checkpoint.get("evidence_sha256") or {}).get(evidence_id)
        != evidence_sha
    ):
        raise VerifyError("SOURCE_LOCK resolution evidence does not match its file")
    manifest_id = source_checkpoint.get("evidence_manifest_id")
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
    current_state = load_json(root / "state/current.json")
    plan_revision = current_state.get("plan_revision")
    if (
        current_state.get("checkpoint_id") != checkpoint_id
        or not isinstance(plan_revision, int)
        or isinstance(plan_revision, bool)
        or plan_revision < 1
    ):
        raise VerifyError("current plan revision is unavailable for progress verification")
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
            verify_child(root, source, current_repositories[source["id"]])
            verify_current_progress_commits(
                root,
                checkpoint,
                plan_revision,
                source,
                current_repositories[source["id"]],
            )
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
    lanes = authored_lanes(registry)
    verify_authored_lane_history(root, registry)
    for lane in lanes:
        lane_checkpoint_id = lane["baseline_checkpoint_id"]
        lane_checkpoint = (
            checkpoint
            if lane_checkpoint_id == checkpoint_id
            else load_json(
                root
                / "state"
                / "checkpoints"
                / f"{lane_checkpoint_id}.json"
            )
        )
        baseline_repositories = lane_checkpoint.get("repositories")
        if (
            lane_checkpoint.get("id") != lane_checkpoint_id
            or not isinstance(baseline_repositories, list)
        ):
            raise VerifyError(
                f"authored lane baseline checkpoint is invalid: {lane['id']}"
            )
        baseline_records = [
            item
            for item in baseline_repositories
            if isinstance(item, dict) and item.get("id") == lane["id"]
        ]
        if len(baseline_records) != 1:
            raise VerifyError(
                f"authored lane baseline checkpoint has no unique repository: {lane['id']}"
            )
        baseline_record = baseline_records[0]
        if (
            baseline_record.get("baseline_commit") != lane["baseline_commit"]
            or baseline_record.get("baseline_tree") != lane["baseline_tree"]
            or baseline_record.get("head") != lane["baseline_commit"]
            or baseline_record.get("tree") != lane["baseline_tree"]
            or baseline_record.get("baseline_tag") != lane["baseline_tag"]
            or baseline_record.get("baseline_tag_object")
            != lane["baseline_tag_object"]
            or baseline_record.get("clean") is not True
            or lane_checkpoint.get("goal_id")
            != lane["baseline_commit_trailers"]["goal_id"]
            or lane_checkpoint.get("source_lock_sha256")
            != lane["baseline_commit_trailers"]["source_lock_sha256"]
            or not isinstance(lane_checkpoint.get("evidence_sha256"), dict)
            or lane_checkpoint["evidence_sha256"].get(lane["baseline_evidence_id"])
            != lane["baseline_commit_trailers"]["evidence_manifest_sha256"]
            or lane_checkpoint.get("change_kind")
            != lane["baseline_commit_trailers"]["change_kind"]
            or lane_checkpoint.get("baseline_commit_marker")
            != lane["baseline_commit_trailers"]["baseline_commit_marker"]
        ):
            raise VerifyError(
                f"authored lane baseline checkpoint identity mismatch: {lane['id']}"
            )
        verify_authored_child(root, lane, current_repositories[lane["id"]])
        verify_current_progress_commits(
            root,
            checkpoint,
            plan_revision,
            lane,
            current_repositories[lane["id"]],
        )
        registered.add(lane["path"])
    projects = root / "projects"
    if projects.is_dir():
        for child in projects.iterdir():
            if child.name.startswith("."):
                continue
            relative = child.relative_to(root).as_posix()
            if child.is_dir() and relative not in registered:
                raise VerifyError(f"unregistered project directory: {relative}")
    verify_root_gitmodules(root, lock.get("sources", []), lanes, registered)


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
