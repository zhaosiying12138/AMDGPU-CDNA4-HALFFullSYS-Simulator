#!/usr/bin/env python3
"""Compute a deterministic identity for the actual contents of a Git worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any


SCHEMA = "amdgpu-sim.repository-source-set.v1"


class SourceSetError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    )
    if completed.stderr:
        raise SourceSetError("git wrote unexpected stderr")
    return completed.stdout


def split_nul(payload: bytes) -> list[bytes]:
    if not payload:
        return []
    if not payload.endswith(b"\0"):
        raise SourceSetError("git path output is not NUL terminated")
    return payload[:-1].split(b"\0")


def tracked_modes(
    repository: Path, *, allow_gitlinks: bool = False
) -> dict[bytes, tuple[int, bytes]]:
    result: dict[bytes, tuple[int, bytes]] = {}
    for record in split_nul(git(repository, "ls-files", "--stage", "-z")):
        prefix, separator, path = record.partition(b"\t")
        if not separator:
            raise SourceSetError("invalid git index record")
        fields = prefix.split(b" ")
        if len(fields) != 3 or fields[2] != b"0":
            raise SourceSetError("unmerged index entries are forbidden")
        try:
            mode = int(fields[0], 8)
        except ValueError as error:
            raise SourceSetError("invalid git index mode") from error
        if mode == 0o160000 and not allow_gitlinks:
            raise SourceSetError(f"gitlink is forbidden: {os.fsdecode(path)}")
        if mode not in (0o100644, 0o100755, 0o120000, 0o160000):
            raise SourceSetError(f"unsupported git index mode: {mode:o}")
        object_id = fields[1]
        result[path] = (mode, object_id)
    return result


def sha256_regular_file(path: Path, before: os.stat_result) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
            or opened.st_mtime_ns != before.st_mtime_ns
        ):
            raise SourceSetError(f"source changed before hashing: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
        ):
            raise SourceSetError(f"source changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def file_record(
    repository: Path,
    encoded_path: bytes,
    index_entry: tuple[int, bytes] | None,
    *,
    allow_gitlinks: bool = False,
) -> dict[str, Any]:
    relative = os.fsdecode(encoded_path)
    if not relative or relative.startswith("/") or "\0" in relative:
        raise SourceSetError("invalid repository-relative path")
    index_mode = None if index_entry is None else index_entry[0]
    index_object = None if index_entry is None else index_entry[1]
    path = repository / relative
    if index_mode == 0o160000:
        if not allow_gitlinks or index_object is None:
            raise SourceSetError(f"gitlink is forbidden: {relative}")
        checkout_marker = path / ".git"
        if checkout_marker.exists():
            try:
                nested_head = git(path, "rev-parse", "HEAD").decode("ascii").strip()
            except (OSError, subprocess.CalledProcessError) as error:
                raise SourceSetError(
                    f"gitlink checkout is unavailable: {relative}"
                ) from error
            if nested_head != index_object.decode("ascii"):
                raise SourceSetError(f"gitlink checkout drifted: {relative}")
            checkout = "present"
        else:
            # An uninitialized submodule is still safely identified by the
            # gitlink object recorded in the parent index.  Do not silently
            # recurse into the parent repository when probing this path.
            checkout = "absent"
        return {
            "path": relative,
            "kind": "gitlink",
            "commit": index_object.decode("ascii"),
            "checkout": checkout,
            "index_mode": "160000",
        }
    try:
        before = path.lstat()
    except FileNotFoundError:
        if index_mode is None:
            raise SourceSetError(f"untracked source disappeared: {relative}")
        return {
            "path": relative,
            "kind": "missing",
            "index_mode": f"{index_mode:o}" if index_mode is not None else None,
        }
    if before.st_uid != os.getuid():
        raise SourceSetError(f"source has wrong owner: {relative}")
    if stat.S_ISREG(before.st_mode):
        return {
            "path": relative,
            "kind": "regular",
            "executable": bool(before.st_mode & stat.S_IXUSR),
            "bytes": before.st_size,
            "sha256": sha256_regular_file(path, before),
            "index_mode": f"{index_mode:o}" if index_mode is not None else None,
        }
    if stat.S_ISLNK(before.st_mode):
        target = os.readlink(path)
        after = path.lstat()
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
        ):
            raise SourceSetError(f"symlink changed while hashing: {relative}")
        encoded_target = os.fsencode(target)
        return {
            "path": relative,
            "kind": "symlink",
            "target": target,
            "bytes": len(encoded_target),
            "sha256": hashlib.sha256(encoded_target).hexdigest(),
            "index_mode": f"{index_mode:o}" if index_mode is not None else None,
        }
    raise SourceSetError(f"special source file is forbidden: {relative}")


def source_set(repository: Path, *, allow_gitlinks: bool = False) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if not repository.is_dir():
        raise SourceSetError("repository is not a directory")
    head = git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    tree = git(repository, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    modes = tracked_modes(repository, allow_gitlinks=allow_gitlinks)
    paths = split_nul(
        git(repository, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    )
    if len(paths) != len(set(paths)):
        raise SourceSetError("git returned duplicate source paths")
    records = [
        file_record(
            repository,
            path,
            modes.get(path),
            allow_gitlinks=allow_gitlinks,
        )
        for path in sorted(paths)
    ]
    core = {"schema": SCHEMA, "head": head, "tree": tree, "files": records}
    return {
        "schema": SCHEMA,
        "repository": str(repository),
        "head": head,
        "tree": tree,
        "file_count": len(records),
        "source_set_sha256": hashlib.sha256(canonical_json(core)).hexdigest(),
        "files": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("--digest-only", action="store_true")
    arguments = parser.parse_args()
    result = source_set(arguments.repository)
    if arguments.digest_only:
        print(result["source_set_sha256"])
    else:
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
