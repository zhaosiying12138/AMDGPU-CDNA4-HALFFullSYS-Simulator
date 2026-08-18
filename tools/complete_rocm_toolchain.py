#!/usr/bin/env python3
"""Complete the ROCm toolchain so its own linker can start.

The materialized ROCm sysroot ships ``lld`` but not every library ``lld``
itself links against, so any host-side HIP *device* link fails with::

    lld: error while loading shared libraries: libxml2.so.2: ...
    clang++: error: amdgcn-link command failed due to signal

That breaks every runtime JIT that compiles device code -- aiter and Triton
both do -- and it surfaces late and misleadingly: one SGLang lane reached 955
retired dispatches, past prefill, before dying here. It looks like a model or
simulator failure and is neither.

The repair belongs in the toolchain, not in ``LD_LIBRARY_PATH``. ``lld``
carries its own RUNPATH (``$ORIGIN/../lib``), and that directory is *not* on
the runtime library path, so installing the missing libraries there fixes the
linker while leaving every library the model run loads untouched. Prepending a
donor prefix to ``LD_LIBRARY_PATH`` instead would place a second libxml2/ICU in
front of the whole process and is the wrong layer.

Idempotent: re-running when the toolchain is already complete reports
``already-complete`` and copies nothing.

    python3 tools/complete_rocm_toolchain.py --prefix <conda-prefix>
    python3 tools/complete_rocm_toolchain.py --prefix <conda-prefix> --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


SCHEMA = "amdgpu-sim.rocm-toolchain-completion.v1"
ROOT = Path(__file__).resolve().parents[1]

# Where a missing dependency may be sourced from. These are prefixes already
# materialized by this repository; nothing is downloaded.
DONOR_RELATIVE = (
    "env/conda/sglang-build-deps/lib",
)

MAX_RESOLUTION_ROUNDS = 8


class ToolchainError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def llvm_root(prefix: Path) -> Path:
    """Locate the ROCm LLVM directory inside a conda prefix."""
    candidates = sorted(prefix.glob("rocm-sysroot/opt/rocm-*/lib/llvm"))
    for candidate in candidates:
        if (candidate / "bin/lld").is_file():
            return candidate
    raise ToolchainError(f"no ROCm LLVM toolchain with bin/lld under {prefix}")


def runpath_entries(binary: Path) -> list[str]:
    completed = subprocess.run(
        ["readelf", "-d", str(binary)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False,
    )
    for line in completed.stdout.splitlines():
        match = re.search(r"\((?:RUNPATH|RPATH)\).*\[(.*)\]", line)
        if match:
            return match.group(1).split(":")
    return []


def missing_libraries(binary: Path, extra_path: Path | None = None) -> list[str]:
    """Return sonames ``ldd`` cannot resolve for ``binary``."""
    environment = None
    if extra_path is not None:
        import os

        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = str(extra_path)
    completed = subprocess.run(
        ["ldd", str(binary)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        check=False, env=environment,
    )
    missing = []
    for line in completed.stdout.splitlines():
        if "not found" in line:
            missing.append(line.split("=>")[0].strip())
    return missing


def donor_for(soname: str, donors: list[Path]) -> Path | None:
    for donor in donors:
        # Prefer the real file over a symlink so the install is self-contained.
        exact = donor / soname
        if exact.exists():
            return exact.resolve() if exact.is_symlink() else exact
        for candidate in sorted(donor.glob(f"{soname}*")):
            if candidate.is_file():
                return candidate
    return None


def linker_starts(llvm: Path) -> tuple[bool, str]:
    completed = subprocess.run(
        [str(llvm / "bin/lld"), "--version"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
    )
    text = completed.stdout.strip()
    # `lld` is a driver: invoked bare it explains that and exits non-zero. That
    # message is proof it started, which is exactly what is being tested.
    started = "error while loading shared libraries" not in text
    return started, text.splitlines()[0] if text else ""


def complete(prefix: Path, apply: bool) -> dict[str, Any]:
    llvm = llvm_root(prefix)
    linker = llvm / "bin/lld"
    target = llvm / "lib"

    donors = [ROOT / relative for relative in DONOR_RELATIVE]
    donors = [donor for donor in donors if donor.is_dir()]

    started_before, detail_before = linker_starts(llvm)
    installed: list[dict[str, Any]] = []
    unresolved: list[str] = []

    if not started_before and apply:
        target.mkdir(parents=True, exist_ok=True)
        for _ in range(MAX_RESOLUTION_ROUNDS):
            missing = missing_libraries(linker)
            # Dependencies of what we just installed count too.
            for installed_record in installed:
                library = target / installed_record["soname"]
                if library.exists():
                    missing.extend(missing_libraries(library, extra_path=target))
            missing = sorted(set(missing))
            if not missing:
                break
            progressed = False
            for soname in missing:
                if (target / soname).exists():
                    continue
                source = donor_for(soname, donors)
                if source is None:
                    if soname not in unresolved:
                        unresolved.append(soname)
                    continue
                destination = target / soname
                shutil.copyfile(source, destination)
                destination.chmod(0o755)
                installed.append({
                    "soname": soname,
                    "source": str(source),
                    "sha256": sha256_file(destination),
                    "bytes": destination.stat().st_size,
                })
                progressed = True
            if not progressed:
                break

    started_after, detail_after = linker_starts(llvm)

    return {
        "schema": SCHEMA,
        "prefix": str(prefix),
        "llvm_root": str(llvm),
        "install_directory": str(target),
        "linker_runpath": runpath_entries(linker),
        "install_directory_is_on_linker_runpath": any(
            entry.endswith("../lib") or entry.endswith("/lib")
            for entry in runpath_entries(linker)
        ),
        "linker_started_before": started_before,
        "linker_started_after": started_after,
        "linker_detail": detail_after or detail_before,
        "installed": installed,
        "unresolved": unresolved,
        "state": (
            "already-complete" if started_before
            else "completed" if started_after
            else "incomplete"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument(
        "--check", action="store_true",
        help="report without installing; exit non-zero if the linker cannot start",
    )
    arguments = parser.parse_args(argv)

    try:
        prefix = arguments.prefix.resolve(strict=True)
        record = complete(prefix, apply=not arguments.check)
    except (ToolchainError, OSError) as error:
        print(f"complete-rocm-toolchain: {error}", file=sys.stderr)
        return 1

    print(json.dumps(record, sort_keys=True, indent=2))
    if record["state"] == "incomplete":
        print(
            "complete-rocm-toolchain: linker still cannot start; unresolved="
            f"{record['unresolved']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
