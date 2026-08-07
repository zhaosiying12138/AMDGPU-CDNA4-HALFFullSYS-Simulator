#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reject generated artifacts and likely credentials from a staged commit."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


MAX_BYTES = 10 * 1024 * 1024
ROOT_FORBIDDEN_PREFIXES = (
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
ROOT_PLACEHOLDERS = {f"{prefix}.gitkeep" for prefix in ROOT_FORBIDDEN_PREFIXES}
ROOT_PLACEHOLDERS.update({"projects/.gitkeep", "state/transactions/.gitkeep"})
FORBIDDEN_SUFFIX = re.compile(
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


class PolicyError(RuntimeError):
    pass


def generated_path(path: str) -> bool:
    first = path.split("/", 1)[0]
    return path.startswith(ROOT_FORBIDDEN_PREFIXES) or first == "build" or first.startswith("build-")


def git(repo: Path, *args: str, text: bool = False) -> bytes | str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    if proc.returncode:
        detail = (proc.stderr or proc.stdout).strip()
        raise PolicyError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout


def staged_paths(repo: Path) -> list[str]:
    raw = git(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMRT", "-z")
    assert isinstance(raw, bytes)
    return [item.decode(errors="surrogateescape") for item in raw.split(b"\0") if item]


def index_mode(repo: Path, path: str) -> str:
    output = git(repo, "ls-files", "--stage", "--", path, text=True)
    assert isinstance(output, str)
    fields = output.split()
    if not fields:
        raise PolicyError(f"staged path has no index entry: {path}")
    return fields[0]


def staged_blob(repo: Path, path: str) -> bytes:
    output = git(repo, "show", f":{path}")
    assert isinstance(output, bytes)
    return output


def check(repo: Path) -> None:
    coordinator = (repo / "SOURCE_LOCK.json").is_file() and (repo / "state/current.json").is_file()
    failures = []
    for path in staged_paths(repo):
        mode = index_mode(repo, path)
        if coordinator and path in ROOT_PLACEHOLDERS:
            continue
        if coordinator and path.startswith("projects/"):
            if mode != "160000":
                failures.append(f"root projects entry must be a gitlink: {path}")
            continue
        if generated_path(path):
            scope = "root Git" if coordinator else "child Git"
            failures.append(f"generated/artifact path is forbidden in {scope}: {path}")
            continue
        if FORBIDDEN_SUFFIX.search(path):
            failures.append(f"binary/model artifact suffix is forbidden: {path}")
            continue
        if mode == "160000":
            continue
        blob = staged_blob(repo, path)
        if len(blob) > MAX_BYTES:
            failures.append(f"staged blob exceeds 10 MiB policy: {path} ({len(blob)} bytes)")
            continue
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(blob):
                failures.append(f"probable {label} in staged blob: {path}")
    if failures:
        raise PolicyError("\n".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    try:
        check(Path(args.repo).resolve())
        return 0
    except PolicyError as exc:
        print(f"staged policy failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
