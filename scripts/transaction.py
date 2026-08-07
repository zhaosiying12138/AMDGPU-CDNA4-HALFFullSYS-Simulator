#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Crash-recoverable coordinator journal for cross-repository checkpoints."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_commit_message import MessageError, check as check_commit_message  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
CP_RE = re.compile(r"^CP-[0-9]{4}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class TransactionError(RuntimeError):
    pass


def git_environment() -> dict[str, str]:
    return {
        **os.environ,
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def git_bytes(*args: str, check: bool = True) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=git_environment(),
    )
    if check and proc.returncode:
        detail = (proc.stderr or proc.stdout).decode(errors="replace").strip()
        raise TransactionError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout


def git(*args: str, check: bool = True) -> str:
    return git_bytes(*args, check=check).decode(errors="replace").strip()


def git_dir() -> Path:
    return Path(git("rev-parse", "--absolute-git-dir"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    parent_created = not path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if parent_created:
        fsync_directory(path.parent.parent)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)
    fsync_directory(path.parent)


@contextmanager
def transition_lock() -> Iterator[Path]:
    control = git_dir() / "amdgpu-sim"
    control_created = not control.exists()
    control.mkdir(parents=True, exist_ok=True)
    if control_created:
        fsync_directory(control.parent)
    lock_path = control / "transition.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield control
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def journal_path(control: Path, checkpoint: str) -> Path:
    if not CP_RE.fullmatch(checkpoint):
        raise TransactionError(f"invalid Checkpoint-ID: {checkpoint!r}")
    return control / "txn" / f"{checkpoint}.json"


def load_journal(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TransactionError(f"cannot read transaction journal {path}: {exc}") from exc
    if value.get("schema") != "amdgpu-sim.transaction.v1":
        raise TransactionError(f"unsupported transaction schema: {path}")
    return value


def verify_child_identity(child_id: str, child: dict[str, str]) -> None:
    relative = child["path"]
    if git("-C", relative, "rev-parse", "HEAD") != child["head"]:
        raise TransactionError(f"child HEAD mismatch: {child_id}")
    if git("-C", relative, "rev-parse", "HEAD^{tree}") != child["tree"]:
        raise TransactionError(f"child tree mismatch: {child_id}")
    if git("-C", relative, "status", "--porcelain=v1", "--untracked-files=all"):
        raise TransactionError(f"child worktree is dirty: {child_id}")


def repository_layout(repo: Path, label: str) -> tuple[Path, Path, str]:
    git_directory = Path(
        git("-C", str(repo), "rev-parse", "--absolute-git-dir")
    ).resolve()
    common_directory = Path(
        git(
            "-C",
            str(repo),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        )
    ).resolve()
    symbolic = git("-C", str(repo), "symbolic-ref", "-q", "HEAD", check=False)
    if git_directory != common_directory:
        raise TransactionError(
            f"{label} uses a linked worktree; transaction durability requires one Git directory"
        )
    if not symbolic.startswith("refs/heads/"):
        raise TransactionError(
            f"{label} has detached or non-branch HEAD; recovery requires a branch ref"
        )
    return git_directory, common_directory, symbolic


def fsync_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise TransactionError(f"durability file is missing for {label}: {path}")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_reference_state(repo: Path, label: str) -> None:
    git_directory, common_directory, symbolic = repository_layout(repo, label)
    fsync_file(git_directory / "HEAD", f"{label} HEAD")
    fsync_file(common_directory / symbolic, f"{label} branch ref")
    packed_refs = common_directory / "packed-refs"
    if packed_refs.exists():
        fsync_file(packed_refs, f"{label} packed refs")
    refs = common_directory / "refs"
    if refs.is_dir():
        directories = {refs}
        for ref in refs.rglob("*"):
            if ref.is_symlink():
                raise TransactionError(f"{label} ref storage contains a symlink: {ref}")
            if ref.is_file():
                fsync_file(ref, f"{label} loose ref")
                directories.add(ref.parent)
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            fsync_directory(directory)
    fsync_directory(common_directory)


def sync_filesystem(path: Path) -> None:
    """Issue Linux syncfs for the filesystem containing path."""

    syncfs = getattr(ctypes.CDLL(None, use_errno=True), "syncfs", None)
    if syncfs is None:
        raise TransactionError("libc does not provide syncfs on this platform")
    syncfs.argtypes = [ctypes.c_int]
    syncfs.restype = ctypes.c_int
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        ctypes.set_errno(0)
        if syncfs(descriptor) != 0:
            error = ctypes.get_errno()
            raise TransactionError(
                f"syncfs failed for {path}: {os.strerror(error)}"
            )
    finally:
        os.close(descriptor)


def sync_repository_filesystems(repositories: list[Path]) -> None:
    """Flush each unique worktree/admin filesystem once before journaling."""

    by_device: dict[int, Path] = {}
    for index, repo in enumerate(repositories):
        git_directory, common_directory, _symbolic = repository_layout(
            repo, f"durability repository {index}"
        )
        for candidate in (repo.resolve(), git_directory, common_directory):
            device = candidate.stat().st_dev
            by_device.setdefault(device, candidate)
    for device in sorted(by_device):
        sync_filesystem(by_device[device])


def harden_object_closure(
    repo: Path,
    positive: str,
    negative: str | None,
    checkpoint: str,
    label: str,
) -> str:
    """Write and fsync a non-thin pack before a journal references an object."""

    _git_directory, common_directory, _symbolic = repository_layout(repo, label)
    pack_directory = common_directory / "objects" / "pack"
    created = not pack_directory.exists()
    pack_directory.mkdir(parents=True, exist_ok=True)
    if created:
        fsync_directory(pack_directory.parent)
    revision_input = f"{positive}\n"
    if negative is not None:
        revision_input += f"^{negative}\n"
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "pack-objects",
            "--quiet",
            "--revs",
            "--no-thin",
            "--include-tag",
            str(pack_directory / "pack"),
        ],
        cwd=ROOT,
        check=False,
        input=revision_input.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=git_environment(),
    )
    pack_hash = proc.stdout.decode(errors="replace").strip()
    if proc.returncode or not SHA_RE.fullmatch(pack_hash):
        detail = proc.stderr.decode(errors="replace").strip()
        raise TransactionError(f"cannot harden object closure for {label}: {detail}")
    pack = pack_directory / f"pack-{pack_hash}.pack"
    index = pack_directory / f"pack-{pack_hash}.idx"
    fsync_file(pack, f"{label} object pack")
    fsync_file(index, f"{label} object index")
    fsync_directory(pack_directory)
    fsync_directory(pack_directory.parent)
    git("-C", str(repo), "cat-file", "-e", positive)
    return pack_hash


def harden_commit(
    repo: Path,
    head: str,
    initial_head: str | None,
    checkpoint: str,
    label: str,
) -> str:
    if git("-C", str(repo), "rev-parse", "HEAD") != head:
        raise TransactionError(f"cannot harden a non-HEAD commit: {label}")
    pack_hash = harden_object_closure(
        repo,
        head,
        initial_head,
        checkpoint,
        label,
    )
    fsync_reference_state(repo, label)
    return pack_hash


def initial_child_identity(
    previous_root: str, child_id: str, relative: str
) -> tuple[str | None, str | None]:
    listing = git("ls-tree", previous_root, "--", relative)
    if not listing:
        return None, None
    records = listing.splitlines()
    if len(records) != 1:
        raise TransactionError(f"previous root has ambiguous child path: {child_id}")
    try:
        metadata, listed_path = records[0].split("\t", 1)
        mode, object_type, head = metadata.split()
    except ValueError as exc:
        raise TransactionError(f"previous root child entry is malformed: {child_id}") from exc
    if (
        listed_path != relative
        or mode != "160000"
        or object_type != "commit"
        or not SHA_RE.fullmatch(head)
    ):
        raise TransactionError(f"previous root child entry is not a gitlink: {child_id}")
    if not (ROOT / relative).is_dir():
        raise TransactionError(f"previous child worktree is missing: {child_id}")
    tree = git("-C", relative, "rev-parse", f"{head}^{{tree}}")
    if not SHA_RE.fullmatch(tree):
        raise TransactionError(f"previous child tree is unavailable offline: {child_id}")
    return head, tree


def verify_declared_initial_children(value: dict[str, Any]) -> None:
    previous_root = value.get("previous_root")
    declared = value.get("declared_children")
    if not isinstance(previous_root, str) or not SHA_RE.fullmatch(previous_root):
        raise TransactionError("transaction has no valid previous root")
    if not isinstance(declared, dict) or not declared:
        raise TransactionError("transaction has no declared child participants")
    for child_id, entry in declared.items():
        if not isinstance(entry, dict) or not re.fullmatch(
            r"[a-z0-9][a-z0-9-]*", child_id
        ):
            raise TransactionError("transaction has an invalid declared child")
        relative = validated_child_path(str(entry.get("path", "")))
        expected_head, expected_tree = initial_child_identity(
            previous_root, child_id, relative
        )
        present = ("initial_head" in entry, "initial_tree" in entry)
        if present == (False, False):
            # Deterministic migration for journals created before initial
            # participant identities became mandatory.
            entry["initial_head"] = expected_head
            entry["initial_tree"] = expected_tree
        elif present != (True, True) or (
            entry.get("initial_head"), entry.get("initial_tree")
        ) != (expected_head, expected_tree):
            raise TransactionError(f"declared child initial identity mismatch: {child_id}")


def validated_child_path(value: str) -> str:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
        or relative.parts[0] != "projects"
    ):
        raise TransactionError(f"unsafe child path: {value!r}")
    return relative.as_posix()


def validated_root_path(value: str) -> str:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or value in {"", "."}:
        raise TransactionError(f"unsafe root allowlist path: {value!r}")
    return relative.as_posix()


def verify_begin_prerequisites(next_checkpoint: str) -> dict[str, Any]:
    """Require one fully accepted workspace before publishing a new journal."""

    verifier = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_workspace.py"),
            "--root",
            str(ROOT),
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=git_environment(),
    )
    if verifier.returncode:
        detail = (verifier.stderr or verifier.stdout).strip()
        raise TransactionError(f"accepted workspace verification failed: {detail}")

    try:
        current = json.loads(
            (ROOT / "state/current.json").read_text(encoding="utf-8")
        )
        checkpoint_id = current["checkpoint_id"]
        checkpoint = json.loads(
            (ROOT / "state/checkpoints" / f"{checkpoint_id}.json").read_text(
                encoding="utf-8"
            )
        )
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        raise TransactionError(f"cannot read the accepted checkpoint: {exc}") from exc
    sequence = checkpoint.get("sequence")
    if (
        current.get("state") != "ready"
        or checkpoint.get("status") != "accepted"
        or checkpoint.get("id") != checkpoint_id
        or not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or checkpoint_id != f"CP-{sequence:04d}"
    ):
        raise TransactionError("current checkpoint is not an accepted ready state")
    expected_next = f"CP-{sequence + 1:04d}"
    if next_checkpoint != expected_next:
        raise TransactionError(
            f"next transaction checkpoint must be {expected_next}, got {next_checkpoint}"
        )
    return current


def command_begin(args: argparse.Namespace) -> None:
    declared_children: dict[str, dict[str, str | None]] = {}
    for encoded in getattr(args, "participant", []) or []:
        participant, separator, target = encoded.partition("@")
        child_id, equals, child_path = participant.partition("=")
        if not equals or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", child_id):
            raise TransactionError(f"invalid participant declaration: {encoded!r}")
        relative = validated_child_path(child_path)
        target_head = target_tree = None
        if separator:
            target_head, colon, target_tree = target.partition(":")
            if (
                not colon
                or not SHA_RE.fullmatch(target_head)
                or not SHA_RE.fullmatch(target_tree)
            ):
                raise TransactionError(f"invalid participant target: {encoded!r}")
        if child_id in declared_children or any(
            entry["path"] == relative for entry in declared_children.values()
        ):
            raise TransactionError(f"duplicate participant declaration: {encoded!r}")
        declared_children[child_id] = {
            "path": relative,
            "target_head": target_head,
            "target_tree": target_tree,
        }
    root_allowlist = sorted(
        {validated_root_path(path) for path in (getattr(args, "root_path", []) or [])}
    )
    if len(root_allowlist) != len(getattr(args, "root_path", []) or []):
        raise TransactionError("root allowlist contains duplicate paths")
    if not declared_children:
        raise TransactionError("a transaction must predeclare at least one child participant")
    with transition_lock() as control:
        path = journal_path(control, args.checkpoint)
        active = sorted((control / "txn").glob("*.json"))
        if active:
            raise TransactionError(
                "another transaction is active: " + ", ".join(item.name for item in active)
            )
        if git("status", "--porcelain=v1", "--untracked-files=all"):
            raise TransactionError("root worktree must be clean before beginning a transaction")
        current = verify_begin_prerequisites(args.checkpoint)
        repository_layout(ROOT, "root coordinator")
        message = git("log", "-1", "--format=%B")
        checkpoint_lines = [
            line for line in message.splitlines() if line.startswith("Checkpoint-ID: ")
        ]
        if checkpoint_lines != [f"Checkpoint-ID: {current['checkpoint_id']}"]:
            raise TransactionError("HEAD has no unique current Checkpoint-ID trailer")
        previous_root = git("rev-parse", "HEAD")
        for child_id, entry in declared_children.items():
            initial_head, initial_tree = initial_child_identity(
                previous_root, child_id, entry["path"] or ""
            )
            if initial_head is None:
                if (ROOT / (entry["path"] or "")).exists():
                    raise TransactionError(
                        f"new child path already exists before transaction begin: {child_id}"
                    )
            else:
                verify_child_identity(
                    child_id,
                    {
                        "path": entry["path"] or "",
                        "head": initial_head,
                        "tree": initial_tree or "",
                    },
                )
            entry["initial_head"] = initial_head
            entry["initial_tree"] = initial_tree
        value = {
            "schema": "amdgpu-sim.transaction.v1",
            "checkpoint_id": args.checkpoint,
            "phase": "prepare",
            "started_at": args.started_at or dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "previous_root": previous_root,
            "previous_checkpoint": current["checkpoint_id"],
            "intent": args.intent,
            "declared_children": declared_children,
            "participants_locked": True,
            "expected_children": {},
            "root_allowlist": root_allowlist,
            "expected_root_tree": None,
            "root_coordinator_commit": None,
        }
        atomic_json(path, value)
        print(path)


def command_declare_child(args: argparse.Namespace) -> None:
    relative = validated_child_path(args.path)
    if (args.head is None) != (args.tree is None):
        raise TransactionError("declared child target needs both head and tree or neither")
    if args.head is not None and (
        not SHA_RE.fullmatch(args.head) or not SHA_RE.fullmatch(args.tree)
    ):
        raise TransactionError("declared child target must use full SHA-1 object IDs")
    with transition_lock() as control:
        path = journal_path(control, args.checkpoint)
        value = load_journal(path)
        if value.get("phase") != "prepare":
            raise TransactionError("participants can only be declared during prepare")
        if value.get("participants_locked") is not True:
            raise TransactionError("transaction participant set is not locked")
        value.pop("expected_child_heads", None)
        value.setdefault("root_allowlist", [])
        verify_declared_initial_children(value)
        declared = value.setdefault("declared_children", {})
        if args.id not in declared:
            raise TransactionError(
                f"child was not predeclared when the transaction began: {args.id}"
            )
        for child_id, existing in declared.items():
            if child_id != args.id and existing.get("path") == relative:
                raise TransactionError(f"child path is already declared by {child_id}")
        previous = declared[args.id]
        if previous.get("path") != relative:
            raise TransactionError(f"declared child path changed: {args.id}")
        target_was_unset = (
            previous.get("target_head") is None and previous.get("target_tree") is None
        )
        if not target_was_unset and (
            previous.get("target_head") != args.head
            or previous.get("target_tree") != args.tree
        ):
            raise TransactionError(f"declared child identity changed: {args.id}")
        recorded = (value.get("expected_children") or {}).get(args.id)
        if recorded is not None:
            if recorded.get("path") != relative:
                raise TransactionError(f"recorded child path conflicts: {args.id}")
            if args.head is not None and (
                recorded.get("head") != args.head or recorded.get("tree") != args.tree
            ):
                raise TransactionError(f"recorded child target conflicts: {args.id}")
        if target_was_unset:
            previous["target_head"] = args.head
            previous["target_tree"] = args.tree
        atomic_json(path, value)
        print(f"declared {args.id} at {relative}")


def command_declare_root(args: argparse.Namespace) -> None:
    paths = sorted({validated_root_path(item) for item in args.path})
    if len(paths) != len(args.path):
        raise TransactionError("root allowlist contains duplicate paths")
    with transition_lock() as control:
        path = journal_path(control, args.checkpoint)
        value = load_journal(path)
        if value.get("phase") != "prepare":
            raise TransactionError("root paths can only be declared during prepare")
        previous = value.get("root_allowlist")
        if previous not in (None, [], paths):
            raise TransactionError("root allowlist changed after declaration")
        value["root_allowlist"] = paths
        atomic_json(path, value)
        print(f"declared {len(paths)} root paths")


def command_record_child(args: argparse.Namespace) -> None:
    if not SHA_RE.fullmatch(args.head) or not SHA_RE.fullmatch(args.tree):
        raise TransactionError("child head and tree must be full SHA-1 object IDs")
    relative = validated_child_path(args.path)
    with transition_lock() as control:
        path = journal_path(control, args.checkpoint)
        value = load_journal(path)
        if value.get("phase") != "prepare":
            raise TransactionError("children can only be recorded during prepare")
        if value.get("participants_locked") is not True:
            raise TransactionError("transaction participant set is not locked")
        verify_declared_initial_children(value)
        declared = (value.get("declared_children") or {}).get(args.id)
        if declared is None:
            raise TransactionError(f"child was not declared at transaction begin: {args.id}")
        if declared.get("path") != relative:
            raise TransactionError(f"child path differs from declaration: {args.id}")
        if declared.get("target_head") not in (None, args.head) or declared.get(
            "target_tree"
        ) not in (None, args.tree):
            raise TransactionError(f"child target differs from declaration: {args.id}")
        children = value.setdefault("expected_children", {})
        entry = {"path": relative, "head": args.head, "tree": args.tree}
        verify_child_identity(args.id, entry)
        for child_id, existing in children.items():
            if child_id != args.id and existing.get("path") == args.path:
                raise TransactionError(f"child path is already recorded by {child_id}")
        previous = children.get(args.id)
        if previous not in (None, entry):
            raise TransactionError(f"child identity changed after recording: {args.id}")
        harden_commit(
            ROOT / relative,
            args.head,
            declared.get("initial_head"),
            args.checkpoint,
            args.id,
        )
        sync_repository_filesystems([ROOT / relative])
        verify_child_identity(args.id, entry)
        children[args.id] = entry
        atomic_json(path, value)
        print(f"recorded {args.id} at {args.head}")


def verify_staged_participant_gitlinks(value: dict[str, Any]) -> None:
    """Require every changed projects/ gitlink to be a declared participant."""

    previous_root = value.get("previous_root")
    declared = value.get("declared_children")
    if not isinstance(previous_root, str) or not SHA_RE.fullmatch(previous_root):
        raise TransactionError("transaction has no valid previous root")
    if not isinstance(declared, dict) or not declared:
        raise TransactionError("transaction has no declared child participants")
    raw = git_bytes(
        "diff",
        "--cached",
        "--raw",
        "-z",
        "--no-renames",
        previous_root,
        "--",
        "projects",
    )
    parts = raw.split(b"\0")
    if parts[-1:] == [b""]:
        parts.pop()
    if len(parts) % 2:
        raise TransactionError("staged projects diff has malformed raw records")
    changed: set[str] = set()
    for offset in range(0, len(parts), 2):
        fields = parts[offset].split()
        if len(fields) != 5 or not fields[0].startswith(b":"):
            raise TransactionError("staged projects diff has malformed metadata")
        old_mode = fields[0][1:]
        new_mode = fields[1]
        try:
            relative = parts[offset + 1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TransactionError("staged project path is not UTF-8") from exc
        relative = validated_child_path(relative)
        if old_mode not in {b"000000", b"160000"} or new_mode != b"160000":
            raise TransactionError(
                f"staged project change is not a gitlink publication: {relative}"
            )
        if relative in changed:
            raise TransactionError(f"staged project path is duplicated: {relative}")
        changed.add(relative)
    expected = {str(entry.get("path", "")) for entry in declared.values()}
    if changed != expected:
        raise TransactionError(
            "changed gitlinks differ from declared participants; "
            f"missing={sorted(expected - changed)}, extra={sorted(changed - expected)}"
        )


def command_prepare_root(args: argparse.Namespace) -> None:
    with transition_lock() as control:
        path = journal_path(control, args.checkpoint)
        value = load_journal(path)
        if value.get("phase") != "prepare":
            raise TransactionError("transaction is not in prepare phase")
        if value.get("participants_locked") is not True:
            raise TransactionError("transaction participant set is not locked")
        verify_declared_initial_children(value)
        declared = value.get("declared_children") or {}
        recorded = value.get("expected_children") or {}
        if not declared or set(declared) != set(recorded):
            missing = sorted(set(declared) - set(recorded))
            extra = sorted(set(recorded) - set(declared))
            raise TransactionError(
                f"participant set is incomplete; missing={missing}, extra={extra}"
            )
        for child_id, child in sorted(value.get("expected_children", {}).items()):
            verify_child_identity(child_id, child)
            index = git("ls-files", "--stage", "--", child["path"])
            fields = index.split()
            if len(fields) < 4 or fields[0] != "160000" or fields[1] != child["head"]:
                raise TransactionError(f"staged gitlink mismatch for {child_id}")
        verify_staged_participant_gitlinks(value)
        allowlist = value.get("root_allowlist")
        if not isinstance(allowlist, list) or not allowlist:
            raise TransactionError("root staged-path allowlist is not declared")
        staged = set(
            filter(
                None,
                git(
                    "diff",
                    "--cached",
                    "--name-only",
                    "--diff-filter=ACDMRTUXB",
                ).splitlines(),
            )
        )
        if staged != set(allowlist):
            raise TransactionError(
                f"staged paths differ from root allowlist; "
                f"missing={sorted(set(allowlist) - staged)}, "
                f"extra={sorted(staged - set(allowlist))}"
            )
        unstaged = git("diff", "--name-only")
        untracked = git("ls-files", "--others", "--exclude-standard")
        if unstaged or untracked:
            raise TransactionError(
                f"root has uncoordinated files; unstaged={unstaged.splitlines()}, "
                f"untracked={untracked.splitlines()}"
            )
        root_tree = git("write-tree")
        harden_object_closure(
            ROOT,
            root_tree,
            f"{value['previous_root']}^{{tree}}",
            args.checkpoint,
            "root prepared tree",
        )
        fsync_file(git_dir() / "index", "root index")
        fsync_directory(git_dir())
        sync_repository_filesystems(
            [ROOT, *(ROOT / child["path"] for child in recorded.values())]
        )
        if git("write-tree") != root_tree:
            raise TransactionError("root index changed across durability barrier")
        for child_id, child in sorted(recorded.items()):
            verify_child_identity(child_id, child)
        staged = git_bytes("diff", "--cached", "--binary", "--full-index")
        value["phase"] = "prepared"
        value["expected_root_tree"] = root_tree
        value["staged_manifest_sha256"] = hashlib.sha256(staged).hexdigest()
        atomic_json(path, value)
        print(f"prepared root tree {root_tree}")


def verify_post_commit(checkpoint: str) -> None:
    verifier = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_workspace.py"),
            "--root",
            str(ROOT),
            "--allow-transaction",
            checkpoint,
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=git_environment(),
    )
    if verifier.returncode:
        detail = (verifier.stderr or verifier.stdout).strip()
        raise TransactionError(f"post-commit workspace verification failed: {detail}")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def retire_journal(path: Path, control: Path) -> None:
    """Durably move a committed journal out of the active transaction set."""

    retired = control / "committed" / path.name
    retired.parent.mkdir(parents=True, exist_ok=True)
    if retired.exists():
        if not path.is_file() or not retired.is_file() or path.read_bytes() != retired.read_bytes():
            raise TransactionError(f"retired transaction journal already exists: {retired}")
        # A power loss after the destination directory was persisted but before
        # the source-directory fsync may recover both names. Keep the durable
        # retired copy and remove only an exact duplicate active name.
        fsync_directory(retired.parent)
        path.unlink()
        fsync_directory(path.parent)
        return
    # Persist the first creation of committed/ before moving the only active
    # journal across directories.
    fsync_directory(control)
    os.replace(path, retired)
    # Persist the new name first. A crash between these fsyncs may leave both
    # names, which is recoverable; persisting the deletion first could lose the
    # only journal name.
    for directory_path in (retired.parent, path.parent):
        fsync_directory(directory_path)


def command_finalize(args: argparse.Namespace) -> None:
    with transition_lock() as control:
        path = journal_path(control, args.checkpoint)
        value = load_journal(path)
        verify_declared_initial_children(value)
        phase = value.get("phase")
        if phase not in {"prepared", "committed"}:
            raise TransactionError("transaction must be prepared before finalization")
        head = git("rev-parse", "HEAD")
        parent = git("rev-parse", "HEAD^")
        tree = git("rev-parse", "HEAD^{tree}")
        message = git("log", "-1", "--format=%B")
        if parent != value.get("previous_root"):
            raise TransactionError("coordinator commit does not directly follow previous_root")
        if tree != value.get("expected_root_tree"):
            raise TransactionError("coordinator commit tree differs from prepared root tree")
        try:
            check_commit_message(message)
        except MessageError as exc:
            raise TransactionError(f"coordinator commit message is invalid: {exc}") from exc
        checkpoint_lines = [
            line for line in message.splitlines() if line.startswith("Checkpoint-ID: ")
        ]
        if checkpoint_lines != [f"Checkpoint-ID: {args.checkpoint}"]:
            raise TransactionError("coordinator commit has a mismatched Checkpoint-ID trailer")
        for child_id, child in sorted(value.get("expected_children", {}).items()):
            relative = child["path"]
            record = git("ls-tree", "HEAD", "--", relative)
            fields = record.split()
            if len(fields) < 4 or fields[0] != "160000" or fields[2] != child["head"]:
                raise TransactionError(f"coordinator gitlink mismatch for {child_id}")
            verify_child_identity(child_id, child)
        current = json.loads((ROOT / "state/current.json").read_text(encoding="utf-8"))
        if current.get("checkpoint_id") != args.checkpoint:
            raise TransactionError("state/current.json does not point to this transaction")
        if phase == "committed":
            if value.get("root_coordinator_commit") != head:
                raise TransactionError(
                    "committed transaction does not bind the coordinator HEAD"
                )
            if not isinstance(value.get("committed_at"), str) or not value["committed_at"]:
                raise TransactionError("committed transaction lacks a commit timestamp")
        if git("status", "--porcelain=v1", "--untracked-files=all"):
            raise TransactionError("root worktree is dirty after coordinator commit")
        verify_post_commit(args.checkpoint)
        if git("rev-parse", "HEAD") != head:
            raise TransactionError("root HEAD changed during post-commit verification")
        if git("status", "--porcelain=v1", "--untracked-files=all"):
            raise TransactionError("root worktree changed during post-commit verification")
        harden_commit(
            ROOT,
            head,
            value["previous_root"],
            args.checkpoint,
            "root coordinator",
        )
        sync_repository_filesystems(
            [ROOT, *(ROOT / child["path"] for child in value["expected_children"].values())]
        )
        if (
            git("rev-parse", "HEAD") != head
            or git("rev-parse", "HEAD^{tree}") != tree
            or git("status", "--porcelain=v1", "--untracked-files=all")
        ):
            raise TransactionError("root identity changed across durability barrier")
        for child_id, child in sorted(value["expected_children"].items()):
            verify_child_identity(child_id, child)
        if phase == "prepared":
            value["phase"] = "committed"
            value["root_coordinator_commit"] = head
            value["committed_at"] = args.committed_at
            atomic_json(path, value)
        retire_journal(path, control)
        print(f"retired committed transaction {args.checkpoint} at {head}")


def command_status(_args: argparse.Namespace) -> None:
    with transition_lock() as control:
        active = sorted((control / "txn").glob("*.json"))
        if not active:
            print("no active transaction")
            return
        for path in active:
            value = load_journal(path)
            print(
                f"{value['checkpoint_id']} phase={value['phase']} "
                f"previous_root={value['previous_root']}"
            )


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser()
    sub = top.add_subparsers(dest="command", required=True)
    begin = sub.add_parser("begin")
    begin.add_argument("--checkpoint", required=True)
    begin.add_argument("--intent", required=True)
    begin.add_argument(
        "--started-at",
        help="ISO-8601 timestamp; defaults to the current UTC time",
    )
    begin.add_argument(
        "--participant",
        action="append",
        default=[],
        metavar="ID=projects/PATH[@HEAD:TREE]",
        help="predeclare a child participant and optional immutable target",
    )
    begin.add_argument(
        "--root-path",
        action="append",
        default=[],
        help="predeclare an exact root path allowed in the coordinator commit",
    )
    begin.set_defaults(func=command_begin)
    child = sub.add_parser("record-child")
    child.add_argument("--checkpoint", required=True)
    child.add_argument("--id", required=True)
    child.add_argument("--path", required=True)
    child.add_argument("--head", required=True)
    child.add_argument("--tree", required=True)
    child.set_defaults(func=command_record_child)
    declare_child = sub.add_parser("declare-child")
    declare_child.add_argument("--checkpoint", required=True)
    declare_child.add_argument("--id", required=True)
    declare_child.add_argument("--path", required=True)
    declare_child.add_argument("--head")
    declare_child.add_argument("--tree")
    declare_child.set_defaults(func=command_declare_child)
    declare_root = sub.add_parser("declare-root")
    declare_root.add_argument("--checkpoint", required=True)
    declare_root.add_argument("--path", action="append", required=True)
    declare_root.set_defaults(func=command_declare_root)
    prepare = sub.add_parser("prepare-root")
    prepare.add_argument("--checkpoint", required=True)
    prepare.set_defaults(func=command_prepare_root)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--checkpoint", required=True)
    finalize.add_argument("--committed-at", required=True)
    finalize.set_defaults(func=command_finalize)
    status = sub.add_parser("status")
    status.set_defaults(func=command_status)
    return top


def main() -> int:
    args = parser().parse_args()
    try:
        args.func(args)
        return 0
    except (TransactionError, OSError, json.JSONDecodeError) as exc:
        print(f"transaction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
