#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run one validation command and retain a hash-addressed external transcript."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent


class CaptureError(RuntimeError):
    pass


def safe_path(relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts:
        raise CaptureError(f"unsafe workspace path: {relative!r}")
    result = (ROOT / value).resolve()
    if result != ROOT and ROOT not in result.parents:
        raise CaptureError(f"path escapes workspace: {relative!r}")
    return result


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def capture(record_relative: str, cwd_relative: str, argv: list[str]) -> dict[str, object]:
    if not argv:
        raise CaptureError("validation argv is empty")
    record = safe_path(record_relative)
    artifacts = (ROOT / "artifacts").resolve()
    if artifacts not in record.parents:
        raise CaptureError("evidence transcripts must remain under artifacts/")
    cwd = safe_path(cwd_relative)
    if not cwd.is_dir():
        raise CaptureError(f"validation cwd does not exist: {cwd_relative}")
    started = dt.datetime.now(dt.timezone.utc)
    proc = subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    ended = dt.datetime.now(dt.timezone.utc)
    stem = record.with_suffix("")
    stdout_path = stem.with_suffix(".stdout")
    stderr_path = stem.with_suffix(".stderr")
    atomic_write(stdout_path, proc.stdout)
    atomic_write(stderr_path, proc.stderr)
    value: dict[str, object] = {
        "schema": "amdgpu-sim.command-evidence.v1",
        "argv": argv,
        "cwd": cwd.relative_to(ROOT).as_posix() or ".",
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": ended.isoformat(timespec="seconds"),
        "exit_code": proc.returncode,
        "stdout": {
            "path": stdout_path.relative_to(ROOT).as_posix(),
            "size": len(proc.stdout),
            "sha256": sha256(proc.stdout),
        },
        "stderr": {
            "path": stderr_path.relative_to(ROOT).as_posix(),
            "size": len(proc.stderr),
            "sha256": sha256(proc.stderr),
        },
    }
    atomic_write(record, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
    print(record.relative_to(ROOT))
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cwd", default=".")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    try:
        result = capture(args.output, args.cwd, argv)
        return int(result["exit_code"])
    except (CaptureError, OSError) as exc:
        print(f"evidence capture failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
