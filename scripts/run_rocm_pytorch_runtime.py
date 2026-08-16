#!/usr/bin/env python3
"""Run the ordinary upstream ROCm PyTorch quickstart on runner-owned gem5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SCRIPTS = ROOT / "scripts"
for directory in (TOOLS, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import gemsim_single_rank_lifecycle as lifecycle  # noqa: E402
import rocm_pytorch_runtime_acceptance as contract  # noqa: E402


class RunnerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def worker_argv(identity: Mapping[str, Any]) -> list[str]:
    files = identity["files"]
    return [
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
        'set -eu; source "$1"; shift; exec "$@"',
        "rocm-pytorch-worker",
        files["activation"]["path"],
        files["python"]["path"],
        files["quickstart"]["path"],
    ]


def worker_environment(execution_root: Path, endpoint: Path) -> dict[str, str]:
    environment = lifecycle.isolated_environment(execution_root)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "SAGR_GENERIC_BRIDGE_ENDPOINT": str(endpoint),
        }
    )
    return environment


def run_gate(
    output: Path,
    *,
    worker_timeout_seconds: int,
    gem5_exit_timeout_seconds: int,
    startup_timeout_seconds: int,
) -> dict[str, Any]:
    lifecycle.validate_absent_output(output)
    identity_preflight = contract.identity_snapshot()
    session = lifecycle.run_session(
        repository_root=ROOT,
        execution_prefix="gs-rocm-pytorch-",
        gem5_binary=contract.GEM5_BINARY,
        gem5_config=contract.GEM5_CONFIG,
        worker_argv=worker_argv(identity_preflight),
        worker_environment_factory=worker_environment,
        worker_label="ROCm PyTorch worker",
        worker_timeout_seconds=worker_timeout_seconds,
        gem5_exit_timeout_seconds=gem5_exit_timeout_seconds,
        startup_timeout_seconds=startup_timeout_seconds,
    )
    failure = session.failure
    try:
        identity_postflight = contract.identity_snapshot()
    except Exception as error:
        identity_postflight = None
        failure = failure or f"PyTorch postflight identity failed: {error}"

    required_artifacts = all(
        (session.execution_root / relative).is_file()
        for relative in contract.SOURCE_ARTIFACTS
    )
    identity_unchanged = identity_postflight == identity_preflight
    success = (
        session.process_success
        and failure is None
        and required_artifacts
        and identity_unchanged
    )
    if not success and failure is None:
        if not identity_unchanged:
            failure = "PyTorch execution identity drifted"
        elif not required_artifacts:
            failure = "PyTorch execution artifacts are incomplete"
        else:
            failure = "PyTorch execution did not meet its contract"

    manifest_core = {
        "schema": contract.RUN_SCHEMA,
        "status": "success" if success else "failed",
        "claim_scope": contract.CLAIM_SCOPE,
        "ordinary_upstream_pytorch_eager_executed": success,
        "runtime_gem5_bridge_modified_for_profile": False,
        "pytorch_rocm_multiop_accepted": False,
        "triton_upstream_amd_accepted": False,
        "torch_compile_accepted": False,
        "vllm_accepted": False,
        "sglang_accepted": False,
        "model_accepted": False,
        "identity_preflight": identity_preflight,
        "identity_postflight": identity_postflight,
        "execution": session.execution,
        "cleanup": session.cleanup,
    }
    try:
        return lifecycle.publish_source(
            output=output,
            execution_root=session.execution_root,
            artifact_paths=contract.SOURCE_ARTIFACTS,
            manifest=manifest_core,
            canonical_json=contract.canonical_json,
            rename_noreplace=contract.base.rename_noreplace,
            error=failure,
        )
    finally:
        shutil.rmtree(session.execution_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-timeout-seconds", type=int, default=120)
    parser.add_argument("--gem5-exit-timeout-seconds", type=int, default=30)
    parser.add_argument("--startup-timeout-seconds", type=int, default=30)
    arguments = parser.parse_args()
    try:
        result = run_gate(
            arguments.output_dir,
            worker_timeout_seconds=arguments.worker_timeout_seconds,
            gem5_exit_timeout_seconds=arguments.gem5_exit_timeout_seconds,
            startup_timeout_seconds=arguments.startup_timeout_seconds,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "success" else 1
    except (
        RunnerError,
        lifecycle.LifecycleError,
        contract.AcceptanceError,
        FileExistsError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"ROCm PyTorch run failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
