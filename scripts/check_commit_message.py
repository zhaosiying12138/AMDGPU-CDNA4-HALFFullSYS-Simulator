#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Enforce the audit trailers used by every amdgpu-sim progress commit."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


REQUIRED = (
    "Checkpoint-ID",
    "Goal-ID",
    "Plan-Revision",
    "Source-Lock-SHA256",
    "Evidence-Manifest-SHA256",
    "Change-Kind",
    "Baseline-Commit",
)
PATTERNS = {
    "Checkpoint-ID": re.compile(r"CP-[0-9]{4}"),
    "Goal-ID": re.compile(r"[A-Z][A-Z0-9-]*"),
    "Plan-Revision": re.compile(r"[1-9][0-9]*"),
    "Source-Lock-SHA256": re.compile(r"[0-9a-f]{64}"),
    "Evidence-Manifest-SHA256": re.compile(r"[0-9a-f]{64}"),
    "Change-Kind": re.compile(r"[a-z][a-z0-9-]*"),
    "Baseline-Commit": re.compile(r"(?:[0-9a-f]{40}|N/A)"),
}
TRAILER = re.compile(r"^([A-Za-z0-9-]+): (.+)$")


class MessageError(RuntimeError):
    pass


def parse_trailers(message: str) -> dict[str, list[str]]:
    trailers: dict[str, list[str]] = {}
    for line in message.splitlines():
        match = TRAILER.fullmatch(line)
        if match:
            trailers.setdefault(match.group(1), []).append(match.group(2))
    return trailers


def check(message: str) -> None:
    lines = message.rstrip().splitlines()
    trailers = parse_trailers(message)
    failures = []
    for name in REQUIRED:
        values = trailers.get(name, [])
        if len(values) != 1:
            failures.append(f"{name} must appear exactly once")
            continue
        if not PATTERNS[name].fullmatch(values[0]):
            failures.append(f"{name} has an invalid value: {values[0]!r}")
    policy_positions = [
        index
        for index, line in enumerate(lines)
        if (match := TRAILER.fullmatch(line)) and match.group(1) in REQUIRED
    ]
    if policy_positions:
        first = min(policy_positions)
        if first == 0 or lines[first - 1].strip():
            failures.append("audit trailers must follow a blank separator")
        if not any(line.strip() for line in lines[:first]):
            failures.append("commit message has no subject before its audit trailers")
        for line in lines[first:]:
            if line.strip() and not TRAILER.fullmatch(line):
                failures.append("non-trailer text appears after the audit trailer block")
                break
    if failures:
        raise MessageError("\n".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("message_file")
    args = parser.parse_args()
    try:
        check(Path(args.message_file).read_text(encoding="utf-8"))
        return 0
    except (OSError, UnicodeError, MessageError) as exc:
        print(f"commit message policy failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
