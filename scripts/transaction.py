#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Crash-recoverable coordinator journal for cross-repository checkpoints."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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


def git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if check and proc.returncode:
        detail = (proc.stderr or proc.stdout).strip()
        raise TransactionError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout.strip()


def git_dir() -> Path:
    return Path(git("rev-parse", "--absolute-git-dir"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def transition_lock() -> Iterator[Path]:
    control = git_dir() / "amdgpu-sim"
    control.mkdir(parents=True, exist_ok=True)
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
        current = json.loads((ROOT / "state/current.json").read_text(encoding="utf-8"))
        message = git("log", "-1", "--format=%B")
        if f"Checkpoint-ID: {current['checkpoint_id']}" not in message:
            raise TransactionError("HEAD does not match the current checkpoint pointer")
        value = {
            "schema": "amdgpu-sim.transaction.v1",
            "checkpoint_id": args.checkpoint,
            "phase": "prepare",
            "started_at": args.started_at or dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "previous_root": git("rev-parse", "HEAD"),
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
    entry: dict[str, str | None] = {
        "path": relative,
        "target_head": args.head,
        "target_tree": args.tree,
    }
    with transition_lock() as control:
        path = journal_path(control, args.checkpoint)
        value = load_journal(path)
        if value.get("phase") != "prepare":
            raise TransactionError("participants can only be declared during prepare")
        if value.get("participants_locked") is not True:
            raise TransactionError("transaction participant set is not locked")
        value.pop("expected_child_heads", None)
        value.setdefault("root_allowlist", [])
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
        if not target_was_unset and previous != entry:
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
            declared[args.id] = entry
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
        children[args.id] = entry
        atomic_json(path, value)
        print(f"recorded {args.id} at {args.head}")


def command_prepare_root(args: argparse.Namespace) -> None:
    with transition_lock() as control:
        path = journal_path(control, args.checkpoint)
        value = load_journal(path)
        if value.get("phase") != "prepare":
            raise TransactionError("transaction is not in prepare phase")
        if value.get("participants_locked") is not True:
            raise TransactionError("transaction participant set is not locked")
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
        staged = subprocess.run(
            ["git", "diff", "--cached", "--binary", "--full-index"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        value["phase"] = "prepared"
        value["expected_root_tree"] = root_tree
        value["staged_manifest_sha256"] = hashlib.sha256(staged).hexdigest()
        atomic_json(path, value)
        print(f"prepared root tree {root_tree}")


def fsync_head_ref() -> None:
    directory = git_dir()
    symbolic = git("symbolic-ref", "-q", "HEAD", check=False)
    if symbolic:
        ref_path = directory / symbolic
        if ref_path.is_file():
            descriptor = os.open(ref_path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1"},
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
        raise TransactionError(f"retired transaction journal already exists: {retired}")
    # Persist the first creation of committed/ before moving the only active
    # journal across directories.
    fsync_directory(control)
    os.replace(path, retired)
    for directory_path in (path.parent, retired.parent):
        fsync_directory(directory_path)


def command_finalize(args: argparse.Namespace) -> None:
    with transition_lock() as control:
        path = journal_path(control, args.checkpoint)
        value = load_journal(path)
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
        if phase == "prepared":
            value["phase"] = "committed"
            value["root_coordinator_commit"] = head
            value["committed_at"] = args.committed_at
            atomic_json(path, value)
        fsync_head_ref()
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
