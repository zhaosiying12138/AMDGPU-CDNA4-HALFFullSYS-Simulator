#!/usr/bin/env python3
"""Verify the framework-neutral ROCm/HIP device-facade capability manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


SCHEMA = "amdgpu-sim.framework-device-facade-audit.v1"
MANIFEST_SCHEMA = "amdgpu-sim.framework-device-facade.v1"
UPSTREAM_ROOTS = {
    "pytorch": "projects/pytorch",
    "triton": "projects/triton",
    "vllm": "projects/vllm",
    "sglang": "projects/sglang-0.5.17",
    "rocm_systems": "projects/rocm-systems",
}
VALID_STATUSES = {"blocked", "in_progress", "source_ready", "accepted"}


class FacadeAuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def audit(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise FacadeAuditError("device facade manifest schema differs")
    errors: list[str] = []
    upstream_result: dict[str, Any] = {}
    for name, record in manifest.get("upstream", {}).items():
        source_root = root / UPSTREAM_ROOTS[name]
        observed: dict[str, Any] = {"root": str(source_root)}
        if "commit" in record:
            observed["commit"] = _git_head(source_root)
            if observed["commit"] != record["commit"]:
                errors.append(f"{name} commit differs")
        files = {}
        for relative, expected in record.get("files", {}).items():
            path = source_root / relative
            if not path.is_file() or path.is_symlink():
                errors.append(f"{name} file is unavailable: {relative}")
                continue
            actual = _sha256(path)
            files[relative] = actual
            if actual != expected:
                errors.append(f"{name} file differs: {relative}")
        observed["files"] = files
        upstream_result[name] = observed

    families = manifest.get("capability_families", [])
    ids = [family.get("id") for family in families]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        errors.append("capability family IDs are not unique and nonempty")
    statuses: dict[str, str] = {}
    for family in families:
        status = family.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"invalid capability status: {family.get('id')}")
            continue
        statuses[family["id"]] = status
        if status != "accepted" and not family.get("blocker"):
            errors.append(f"non-accepted capability lacks blocker: {family['id']}")

    acceptance = manifest.get("acceptance", {})
    prerequisites = (
        list(acceptance.get("device_facade_prerequisite", []))
        + list(acceptance.get("distributed_prerequisite", []))
    )
    if any(item not in statuses for item in prerequisites):
        errors.append("acceptance references an unknown capability")
    computed_model_ready = bool(prerequisites) and all(
        statuses.get(item) == "accepted" for item in prerequisites
    )
    if acceptance.get("model_ready") is not computed_model_ready:
        errors.append("model_ready differs from prerequisite capability states")

    return {
        "schema": SCHEMA,
        "manifest_schema": manifest["schema"],
        "upstream": upstream_result,
        "capability_status": statuses,
        "model_ready": computed_model_ready,
        "errors": errors,
        "correct": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tools/framework_device_facade_manifest.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="ascii"))
    result = audit(args.root, manifest)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="ascii")
    else:
        print(payload, end="")
    return 0 if result["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
