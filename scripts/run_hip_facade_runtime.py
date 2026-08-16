#!/usr/bin/env python3
"""Run one standard upstream HIP kernel against a runner-owned gem5 instance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SCRIPTS = ROOT / "scripts"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import hip_facade_runtime_acceptance as contract  # noqa: E402
import run_upstream_rocr_aql as lifecycle  # noqa: E402


class RunnerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def gem5_argv(execution_root: Path, endpoint: Path, trace: Path, job_uuid: str) -> list[str]:
    return [
        str(contract.GEM5_BINARY.resolve()),
        "--listener-mode=on",
        "--outdir",
        str(execution_root / "m5out"),
        str(contract.GEM5_CONFIG.resolve()),
        "--endpoint",
        str(endpoint),
        "--dispatch-trace-path",
        str(trace),
        "--epoch",
        "1",
        "--job-uuid",
        job_uuid,
        "--rank",
        "0",
        "--world-size",
        "1",
        "--startup-timeout-ms",
        "86400000",
        "--handshake-timeout-ms",
        "15000",
        "--run-timeout-ms",
        "86400000",
    ]


def worker_argv(identity: Mapping[str, Any]) -> list[str]:
    files = identity["files"]
    return [
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
        'set -eu; source "$1"; shift; exec "$@"',
        "hip-facade-worker",
        files["activation"]["path"],
        files["python"]["path"],
        files["smoke"]["path"],
        "--mode",
        "vector-add",
        "--count",
        "256",
    ]


def worker_environment(execution_root: Path, endpoint: Path) -> dict[str, str]:
    environment = lifecycle.common_environment(execution_root)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "SAGR_GENERIC_BRIDGE_ENDPOINT": str(endpoint),
        }
    )
    return environment


def publish_source(
    output: Path,
    execution_root: Path,
    manifest_core: Mapping[str, Any],
    error: str | None,
) -> dict[str, Any]:
    parent = output.parent.resolve(strict=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        for relative in contract.SOURCE_ARTIFACTS:
            source = execution_root / relative
            if not source.is_file() or source.is_symlink():
                continue
            destination = temporary / relative
            record = lifecycle._copy_file(source, destination)
            record["path"] = relative
            artifacts[relative] = record
        if error is not None:
            payload = (error.rstrip() + "\n").encode("ascii", errors="backslashreplace")
            record = lifecycle._write_bytes(temporary / "runner-error.txt", payload)
            artifacts["runner-error.txt"] = {"path": "runner-error.txt", **record}
        manifest = dict(manifest_core)
        manifest["artifacts"] = artifacts
        lifecycle._write_bytes(
            temporary / "result-manifest.json", contract.canonical_json(manifest)
        )
        lifecycle._fsync_tree(temporary)
        contract.base.rename_noreplace(temporary, output)
        temporary = None
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return manifest
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)


def run_gate(
    output: Path,
    *,
    worker_timeout_seconds: int,
    gem5_exit_timeout_seconds: int,
    startup_timeout_seconds: int,
) -> dict[str, Any]:
    lifecycle.validate_output(output)
    require(1 <= worker_timeout_seconds <= 600, "HIP worker timeout is outside 1..600")
    require(1 <= gem5_exit_timeout_seconds <= 120, "HIP gem5 timeout is outside 1..120")
    require(1 <= startup_timeout_seconds <= 120, "HIP startup timeout is outside 1..120")

    identity_preflight = contract.identity_snapshot()
    execution_root = Path(tempfile.mkdtemp(prefix="gs-hip-facade-", dir="/tmp"))
    execution_root.chmod(0o700)
    lifecycle.private_directories(execution_root)
    endpoint = execution_root / "bridge.sock"
    trace_path = execution_root / "dispatch-trace.jsonl"
    require(len(os.fsencode(endpoint)) < 108, "HIP private endpoint exceeds AF_UNIX capacity")
    job_uuid = secrets.token_hex(16)
    gem5_command = gem5_argv(execution_root, endpoint, trace_path, job_uuid)
    worker_command = worker_argv(identity_preflight)
    gem5_environment = lifecycle.common_environment(execution_root)
    hip_environment = worker_environment(execution_root, endpoint)

    gem5_process: subprocess.Popen[bytes] | None = None
    worker_process: subprocess.Popen[bytes] | None = None
    gem5_log = None
    worker_log = None
    gem5_pid = gem5_start = gem5_group = None
    worker_pid = worker_start = worker_group = None
    gem5_exit_code: int | None = None
    worker_exit_code: int | None = None
    gem5_forced = False
    worker_forced = False
    failure: str | None = None
    identity_postflight: dict[str, Any] | None = None

    try:
        gem5_log = (execution_root / "gem5.log").open("wb", buffering=0)
        gem5_process = subprocess.Popen(
            gem5_command,
            cwd=ROOT,
            env=gem5_environment,
            stdin=subprocess.DEVNULL,
            stdout=gem5_log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
        gem5_pid, gem5_start, gem5_group = lifecycle.process_identity(gem5_process, "gem5")
        lifecycle.wait_for_endpoint(endpoint, gem5_process, startup_timeout_seconds)

        worker_log = (execution_root / "worker.log").open("wb", buffering=0)
        worker_process = subprocess.Popen(
            worker_command,
            cwd=ROOT,
            env=hip_environment,
            stdin=subprocess.DEVNULL,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
        worker_pid, worker_start, worker_group = lifecycle.process_identity(worker_process, "HIP worker")
        try:
            worker_exit_code = worker_process.wait(timeout=worker_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise RunnerError("HIP worker timed out") from error
        if worker_exit_code != 0:
            raise RunnerError(f"HIP worker exited {worker_exit_code}")
        try:
            gem5_exit_code = gem5_process.wait(timeout=gem5_exit_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise RunnerError("gem5 did not exit after the HIP session") from error
        if gem5_exit_code != 0:
            raise RunnerError(f"gem5 exited {gem5_exit_code}")
    except Exception as error:
        failure = str(error)
    finally:
        worker_forced = lifecycle.terminate_group(worker_process, worker_group)
        gem5_forced = lifecycle.terminate_group(gem5_process, gem5_group)
        if worker_process is not None:
            try:
                worker_exit_code = worker_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                failure = failure or "HIP worker could not be reaped"
        if gem5_process is not None:
            try:
                gem5_exit_code = gem5_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                failure = failure or "HIP gem5 could not be reaped"
        if worker_log is not None:
            worker_log.close()
        if gem5_log is not None:
            gem5_log.close()
        try:
            identity_postflight = contract.identity_snapshot()
        except Exception as error:
            failure = failure or f"HIP postflight identity failed: {error}"

    worker_reaped = worker_process is not None and worker_process.poll() is not None
    gem5_reaped = gem5_process is not None and gem5_process.poll() is not None
    worker_group_absent = worker_group is not None and not lifecycle.process_group_members(worker_group)
    gem5_group_absent = gem5_group is not None and not lifecycle.process_group_members(gem5_group)
    endpoint_absent = not os.path.lexists(endpoint)
    cleanup = {
        "worker_reaped": worker_reaped,
        "gem5_reaped": gem5_reaped,
        "worker_process_group_absent": worker_group_absent,
        "gem5_process_group_absent": gem5_group_absent,
        "endpoint_absent": endpoint_absent,
        "worker_forced_termination": worker_forced,
        "gem5_forced_termination": gem5_forced,
        "all_clear": (
            worker_reaped
            and gem5_reaped
            and worker_group_absent
            and gem5_group_absent
            and endpoint_absent
            and not worker_forced
            and not gem5_forced
        ),
    }
    required_artifacts = all((execution_root / relative).is_file() for relative in contract.SOURCE_ARTIFACTS)
    identity_unchanged = identity_postflight == identity_preflight
    success = (
        failure is None
        and worker_exit_code == 0
        and gem5_exit_code == 0
        and cleanup["all_clear"]
        and required_artifacts
        and identity_unchanged
    )
    if not success and failure is None:
        if not identity_unchanged:
            failure = "HIP execution identity drifted"
        elif not cleanup["all_clear"]:
            failure = "HIP process cleanup was incomplete"
        elif not required_artifacts:
            failure = "HIP execution artifacts are incomplete"
        else:
            failure = "HIP execution did not meet its contract"

    execution = {
        "job_uuid": job_uuid,
        "epoch": 1,
        "rank": 0,
        "world_size": 1,
        "execution_root": str(execution_root),
        "endpoint": str(endpoint),
        "trace_path": str(trace_path),
        "m5out_path": str(execution_root / "m5out"),
        "gem5_argv": gem5_command,
        "worker_argv": worker_command,
        "gem5_environment": gem5_environment,
        "worker_environment": hip_environment,
        "gem5_pid": gem5_pid,
        "gem5_start_time_ticks": gem5_start,
        "gem5_process_group": gem5_group,
        "worker_pid": worker_pid,
        "worker_start_time_ticks": worker_start,
        "worker_process_group": worker_group,
        "worker_exit_code": worker_exit_code,
        "gem5_exit_code": gem5_exit_code,
        "worker_timeout_seconds": worker_timeout_seconds,
        "gem5_exit_timeout_seconds": gem5_exit_timeout_seconds,
    }
    manifest_core = {
        "schema": contract.RUN_SCHEMA,
        "status": "success" if success else "failed",
        "claim_scope": contract.CLAIM_SCOPE,
        "standard_hip_kernel_executed": success,
        "pytorch_rocm_accepted": False,
        "triton_upstream_hip_accepted": False,
        "framework_accepted": False,
        "model_accepted": False,
        "identity_preflight": identity_preflight,
        "identity_postflight": identity_postflight,
        "execution": execution,
        "cleanup": cleanup,
    }
    try:
        return publish_source(output, execution_root, manifest_core, failure)
    finally:
        shutil.rmtree(execution_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-timeout-seconds", type=int, default=120)
    parser.add_argument("--gem5-exit-timeout-seconds", type=int, default=20)
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
    except (RunnerError, lifecycle.RunnerError, contract.AcceptanceError, FileExistsError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"HIP facade run failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
