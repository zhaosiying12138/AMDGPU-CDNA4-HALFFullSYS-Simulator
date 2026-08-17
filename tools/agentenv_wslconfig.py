#!/usr/bin/env python3
"""Render the AgentENV custom-kernel portion of a Windows .wslconfig."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Sequence


SECTION_RE = re.compile(r"^\s*\[([^]]+)]\s*$")
MANAGED_KEY_RE = re.compile(
    r"^\s*(kernel|kernelModules|nestedVirtualization)\s*=", re.IGNORECASE
)


class ConfigError(ValueError):
    pass


def windows_wsl_path(path: str) -> str:
    match = re.fullmatch(r"/mnt/([A-Za-z])(?:/(.*))?", path.rstrip("/"))
    if not match:
        raise ConfigError("kernel staging path must be below /mnt/<drive>")
    components = [part for part in (match.group(2) or "").split("/") if part]
    if any(part in (".", "..") or "\\" in part for part in components):
        raise ConfigError("kernel staging path contains an unsafe component")
    # WSL's .wslconfig parser requires each Windows separator to remain
    # escaped in the file itself, for example C:\\Users\\name\\bzImage.
    return f"{match.group(1).upper()}:" + "".join(f"\\\\{part}" for part in components)


def render_candidate(current: str, stage: str) -> str:
    lines = current.splitlines()
    section_indexes = [
        index
        for index, line in enumerate(lines)
        if (match := SECTION_RE.match(line)) and match.group(1).lower() == "wsl2"
    ]
    if len(section_indexes) != 1:
        raise ConfigError("active config must contain exactly one [wsl2] section")
    start = section_indexes[0]
    end = next(
        (index for index in range(start + 1, len(lines)) if SECTION_RE.match(lines[index])),
        len(lines),
    )
    preserved = [line for line in lines[start + 1 : end] if not MANAGED_KEY_RE.match(line)]
    win = windows_wsl_path(stage)
    managed = [
        f"kernel={win}\\\\bzImage",
        f"kernelModules={win}\\\\modules.vhdx",
        "nestedVirtualization=true",
    ]
    rendered = lines[: start + 1] + preserved + managed + lines[end:]
    return "\n".join(rendered) + "\n"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--active", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--stage", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    current = args.active.read_text(encoding="utf-8")
    rendered = render_candidate(current, args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
