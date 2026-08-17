#!/usr/bin/env python3
"""Resolve and assert the binary identity a run is about to execute.

A model run costs 50-70 minutes. A run whose loaded ROCr/HIP does not descend
from the commit under test is worthless as evidence in either direction, and
that failure mode is invisible in the run log: the stale product still emits
upstream DTIF fast-copy records, so the run looks correct while carrying none of
the project fixes.

This gate resolves what the dynamic loader would *actually* bind, not what the
environment claims. It loads the runtime in a throwaway child and reads
``/proc/self/maps``. It deliberately does not call ``hsa_init``, so it never
starts a managed gem5.

Emit the record at the head of every run log:

    python3 tools/run_identity_gate.py --format text >>"$log"

Fail closed in a runner:

    python3 tools/run_identity_gate.py --require-active-product --require-fastcopy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SCHEMA = "amdgpu-sim.run-identity-gate.v1"
ROOT = Path(__file__).resolve().parents[1]

# Sonames whose resolved path decides whether a run is testing the right code.
TRACKED_SONAMES = (
    "libhsa-runtime64.so.1",
    "libamdhip64.so.7",
)


class IdentityError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False,
        )
    except OSError:
        return None
    value = completed.stdout.strip()
    return value or None


def resolve_loaded(soname: str) -> str | None:
    """Return the path the loader really binds for ``soname``.

    Runs in a child so a failed dlopen cannot poison this process, and so the
    child's own ``/proc/self/maps`` reflects the binding.
    """
    program = (
        "import ctypes,os,sys\n"
        f"soname={soname!r}\n"
        "try:\n"
        "    ctypes.CDLL(soname, mode=os.RTLD_NOW | os.RTLD_LOCAL)\n"
        "except OSError as error:\n"
        "    sys.stderr.write(str(error))\n"
        "    raise SystemExit(2)\n"
        "stem=soname.split('.so')[0]\n"
        "for line in open('/proc/self/maps'):\n"
        "    parts=line.rstrip().split(' ')\n"
        "    path=parts[-1]\n"
        "    if path.startswith('/') and stem in os.path.basename(path):\n"
        "        print(path)\n"
        "        break\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if completed.returncode != 0:
        return None
    resolved = completed.stdout.strip()
    return resolved or None


def describe(path_text: str | None) -> dict[str, Any]:
    if not path_text:
        return {"resolved": None}
    path = Path(path_text)
    record: dict[str, Any] = {"resolved": str(path)}
    try:
        real = path.resolve(strict=True)
    except OSError:
        record["error"] = "unresolvable"
        return record
    record["realpath"] = str(real)
    record["basename"] = real.name
    try:
        record["bytes"] = real.stat().st_size
        record["sha256"] = sha256_file(real)
    except OSError as error:
        record["error"] = str(error)
    return record


def active_product() -> dict[str, Any] | None:
    path = ROOT / "env/rocm/active-product"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, ValueError):
        return None


def build_record() -> dict[str, Any]:
    libraries = {name: describe(resolve_loaded(name)) for name in TRACKED_SONAMES}

    model_lib = os.environ.get("HSA_MODEL_LIB")
    gem5 = os.environ.get("SAGR_MANAGED_GEM5")

    fastcopy = {
        "HSA_ENABLE_DTIF_FAST_COPY": os.environ.get("HSA_ENABLE_DTIF_FAST_COPY"),
        "SAGR_HSAKMT_MODEL_FAST_COPY": os.environ.get("SAGR_HSAKMT_MODEL_FAST_COPY"),
    }
    fastcopy["enabled"] = (
        fastcopy["HSA_ENABLE_DTIF_FAST_COPY"] == "1"
        and fastcopy["SAGR_HSAKMT_MODEL_FAST_COPY"] == "1"
    )

    product = active_product()
    product_prefix = product.get("prefix") if product else None
    rocr_real = libraries["libhsa-runtime64.so.1"].get("realpath")
    hip_real = libraries["libamdhip64.so.7"].get("realpath")

    def under_product(value: str | None) -> bool | None:
        if product_prefix is None or value is None:
            return None
        return value.startswith(product_prefix.rstrip("/") + "/")

    return {
        "schema": SCHEMA,
        "repo_head": git_head(ROOT),
        "rocm_systems_head": git_head(ROOT / "projects/rocm-systems"),
        "self_runtime_head": git_head(ROOT / "projects/self-amdgpu-runtime"),
        "gem5_head": git_head(ROOT / "projects/gem5"),
        "active_product": product,
        "libraries": libraries,
        "model_lib": describe(model_lib),
        "gem5_binary": describe(gem5),
        "fastcopy": fastcopy,
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "matches_active_product": {
            "libhsa-runtime64.so.1": under_product(rocr_real),
            "libamdhip64.so.7": under_product(hip_real),
            "model_lib": under_product(describe(model_lib).get("realpath")),
        },
    }


def render_text(record: dict[str, Any]) -> str:
    lines = [f"# {SCHEMA}"]
    lines.append(f"repo_head={record['repo_head']}")
    lines.append(f"rocm_systems_head={record['rocm_systems_head']}")
    lines.append(f"self_runtime_head={record['self_runtime_head']}")
    product = record.get("active_product") or {}
    lines.append(f"active_product={product.get('prefix')}")
    for name, info in record["libraries"].items():
        lines.append(
            f"{name}: realpath={info.get('realpath')} "
            f"sha256={info.get('sha256')} bytes={info.get('bytes')}"
        )
    lines.append(
        f"model_lib: realpath={record['model_lib'].get('realpath')} "
        f"sha256={record['model_lib'].get('sha256')}"
    )
    lines.append(
        f"gem5: realpath={record['gem5_binary'].get('realpath')} "
        f"sha256={record['gem5_binary'].get('sha256')}"
    )
    lines.append(f"fastcopy_enabled={record['fastcopy']['enabled']}")
    lines.append(f"matches_active_product={record['matches_active_product']}")
    lines.append(f"ld_preload={record['ld_preload']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument(
        "--require-active-product",
        nargs="?",
        const="libhsa-runtime64.so.1",
        default=None,
        metavar="SONAMES",
        help=(
            "comma-separated sonames that must resolve inside env/rocm/active-product. "
            "Defaults to libhsa-runtime64.so.1 -- the KMD-replacement boundary this "
            "project owns. Pass 'libhsa-runtime64.so.1,libamdhip64.so.7' to also "
            "require the product HIP, which currently loses to the conda "
            "rocm-sysroot copy on LD_LIBRARY_PATH."
        ),
    )
    parser.add_argument(
        "--require-fastcopy",
        action="store_true",
        help="fail unless both fast-copy gates are exactly 1",
    )
    arguments = parser.parse_args(argv)

    try:
        record = build_record()
    except IdentityError as error:
        print(f"run-identity-gate: {error}", file=sys.stderr)
        return 1

    if arguments.format == "json":
        print(json.dumps(record, sort_keys=True, indent=2))
    else:
        print(render_text(record), end="")

    status = 0
    if arguments.require_active_product is not None:
        matches = record["matches_active_product"]
        required = [n.strip() for n in arguments.require_active_product.split(",") if n.strip()]
        for name in required:
            if name not in matches:
                print(f"run-identity-gate: unknown soname {name!r}", file=sys.stderr)
                status = 1
            elif matches.get(name) is not True:
                print(
                    f"run-identity-gate: {name} does not resolve inside the active "
                    f"product ({record['libraries'].get(name, {}).get('realpath')})",
                    file=sys.stderr,
                )
                status = 1
    # Always surface the known HIP wiring defect, even when not required, so it
    # cannot quietly persist: the product builds its own libamdhip64 but the
    # conda rocm-sysroot copy precedes it on LD_LIBRARY_PATH, leaving the
    # project's CLR changes inert.
    if record["matches_active_product"].get("libamdhip64.so.7") is False:
        print(
            "run-identity-gate: WARNING libamdhip64.so.7 resolves outside the active "
            f"product ({record['libraries']['libamdhip64.so.7'].get('realpath')}); "
            "project CLR/HIP changes are NOT in effect for this run",
            file=sys.stderr,
        )
    if arguments.require_fastcopy and not record["fastcopy"]["enabled"]:
        print(
            "run-identity-gate: fast copy is not enabled "
            "(source scripts/fastcopy_mode.sh fast)",
            file=sys.stderr,
        )
        status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
