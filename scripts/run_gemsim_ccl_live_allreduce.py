#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Supervise one frozen-product live device allreduce run.

This runner produces source evidence only.  It never grants acceptance; the
independent verifier is the sole acceptance authority.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CCL_SOURCE = ROOT / "plugins/collectives/gemsim_ccl/src"
for _path in (CCL_SOURCE,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gemsim_ccl.native as _imported_native  # noqa: E402
from gemsim_ccl.native import (  # noqa: E402
    CANCELLED,
    CCLStatusError,
    NativeCCL,
    PEER_LOST,
    PROTOCOL_ERROR,
    TIMED_OUT,
)


RUN_SCHEMA = "amdgpu-sim.ccl-live-allreduce-run.v1"
EXPECTED_SCHEMA = "amdgpu-sim.ccl-live-allreduce-expected.v1"
WORKER_CONFIG_SCHEMA = "amdgpu-sim.ccl-live-allreduce-rank-config.v1"
PRODUCT_SCHEMA = "amdgpu-sim.product-prefix.v1"
PR_SET_CHILD_SUBREAPER = 36
MAX_JSON_BYTES = 64 * 1024 * 1024
RANK_FILES = (
    "worker-result.json",
    "step-journal.jsonl",
    "dispatch-trace.jsonl",
    "stats.txt",
    "gem5.log",
    "rank-launch.json",
    "input.bin",
    "output.bin",
)
IDENTITY_ROLES = (
    "product_manifest",
    "runtime_library",
    "ccl_native",
    "ccl_device",
    "ccl_engine",
    "triton_driver",
    "gem5_binary",
    "gem5_config",
    "verifier",
    "runner",
    "worker",
    "bootstrap",
    "design",
    "rank_registry",
)


class RunnerError(RuntimeError):
    pass


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def object_sha256(value: object) -> str:
    return sha256_bytes(canonical_json(value))


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RunnerError(f"identity path is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": metadata.st_size, "sha256": digest.hexdigest()}


def artifact_record(root: Path, path: Path) -> dict[str, Any]:
    record = file_record(path)
    record["path"] = path.relative_to(root).as_posix()
    return record


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    if not payload or len(payload) > MAX_JSON_BYTES:
        raise RunnerError(f"{label} has an invalid size")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunnerError(f"{label} is invalid ASCII JSON") from error
    if not isinstance(value, dict):
        raise RunnerError(f"{label} must be a JSON object")
    return value, payload


def _validate_expected_and_paths(
    expected: Mapping[str, Any], output: Path, execution_root: Path
) -> Mapping[str, Any]:
    """Run the bound independent design validator before creating any path."""
    design_path = ROOT / "tools/gemsim_ccl_live_allreduce.py"
    design_name = "_gemsim_live_allreduce_design_preflight"
    design_spec = importlib.util.spec_from_file_location(design_name, design_path)
    if design_spec is None or design_spec.loader is None:
        raise RunnerError("could not load the bound canonical design validator")
    design_module = importlib.util.module_from_spec(design_spec)
    sys.modules[design_name] = design_module
    try:
        design_spec.loader.exec_module(design_module)
        canonical_design = design_module.validate_design(expected.get("design"))
    except Exception as error:
        raise RunnerError(f"canonical design validation failed: {error}") from error
    finally:
        sys.modules.pop(design_name, None)

    verifier_path = ROOT / "tools/gemsim_ccl_live_allreduce_acceptance.py"
    spec = importlib.util.spec_from_file_location(
        "_gemsim_live_allreduce_preflight", verifier_path
    )
    if spec is None or spec.loader is None:
        raise RunnerError("could not load the bound expected-design validator")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        design, _plans = module.validate_expected(expected)
    except Exception as error:
        raise RunnerError(f"expected design validation failed: {error}") from error
    if design != canonical_design:
        raise RunnerError("canonical and independent design validators disagree")

    ranks = design["ranks"]
    for rank, rank_design in enumerate(ranks):
        rank_root = execution_root / f"rank-{rank:02d}"
        instance = rank_root / "correctness"
        cache = rank_root / "cache/triton"
        runtime = instance / "runtime"
        exact = {
            "instance_directory": instance,
            "triton_cache_directory": cache,
            "runtime_directory": runtime,
            "endpoint": runtime / "bridge.sock",
            "gem5_output_directory": runtime / "m5out",
            "dispatch_trace_path": runtime / "dispatch-trace.jsonl",
            "gem5_log_path": runtime / "gem5.log",
            "gem5_cache_directory": runtime / "cache",
        }
        paths = rank_design["rank_launch"]["paths"]
        if any(paths.get(name) != str(path) for name, path in exact.items()):
            raise RunnerError(f"rank {rank} launch path topology is not exact")
    return design


def _exclusive_write(path: Path, payload: bytes, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_manifest(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json(value)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    _exclusive_write(temporary, payload)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def enable_subreaper() -> None:
    if not sys.platform.startswith("linux") or not Path("/proc/self/stat").is_file():
        raise RunnerError("the live CCL supervisor requires Linux procfs")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise RunnerError(f"could not enable child subreaper: errno {ctypes.get_errno()}")


def _proc_identity(pid: int) -> tuple[int, int, int, str] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError:
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2 :].split()
    if len(fields) < 20:
        return None
    return int(fields[19]), int(fields[1]), int(fields[2]), fields[0]


def _alive(pid: int, start_time: int) -> bool:
    value = _proc_identity(pid)
    return value is not None and value[0] == start_time and value[3] != "Z"


def _present(pid: int, start_time: int) -> bool:
    value = _proc_identity(pid)
    return value is not None and value[0] == start_time


def _process_table() -> dict[int, tuple[int, int, int, str]]:
    result: dict[int, tuple[int, int, int, str]] = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            identity = _proc_identity(int(entry.name))
            if identity is not None:
                result[int(entry.name)] = identity
    return result


def _descendants(parent_pid: int) -> set[int]:
    parents = {pid: identity[1] for pid, identity in _process_table().items()}
    result: set[int] = set()
    frontier = {parent_pid}
    while frontier:
        children = {pid for pid, ppid in parents.items() if ppid in frontier and pid not in result}
        result.update(children)
        frontier = children
    return result


@dataclass
class RankProcess:
    rank: int
    directory: Path
    launch: dict[str, Any]
    capability_fd: int = -1
    process: subprocess.Popen[bytes] | None = None
    start_time_ticks: int | None = None
    stdout: Any = None
    stderr: Any = None
    descendants: dict[int, int] = field(default_factory=dict)
    returncode: int | None = None


def _direct_child_identities(parent_pid: int) -> set[tuple[int, int]]:
    return {
        (pid, identity[0])
        for pid, identity in _process_table().items()
        if identity[1] == parent_pid
    }


def _capture_owned_processes(
    runs: Sequence[RankProcess], baseline_children: set[tuple[int, int]]
) -> None:
    """Retain PID/start-time ownership through worker exit and subreparenting."""
    table = _process_table()
    supervisor_pid = os.getpid()
    by_group = {
        run.process.pid: run
        for run in runs
        if run.process is not None
    }
    prior_owner = {
        pid: run
        for run in runs
        for pid in run.descendants
    }
    for pid, identity in table.items():
        started, parent, process_group, _state = identity
        if any(run.process is not None and pid == run.process.pid for run in runs):
            continue
        owner = prior_owner.get(pid) or by_group.get(process_group)
        if owner is None:
            for run in runs:
                if run.process is not None and pid in _descendants(run.process.pid):
                    owner = run
                    break
        if (
            owner is None
            and parent == supervisor_pid
            and (pid, started) not in baseline_children
            and runs
        ):
            owner = runs[0]
        if owner is not None:
            owner.descendants.setdefault(pid, started)


def owned_fd_snapshot() -> dict[int, tuple[int, int, int, str]]:
    """Snapshot every supervisor-owned descriptor without retaining a handle."""
    result: dict[int, tuple[int, int, int, str]] = {}
    for name in os.listdir("/proc/self/fd"):
        if not name.isdigit():
            continue
        descriptor = int(name)
        try:
            metadata = os.fstat(descriptor)
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            continue
        result[descriptor] = (
            int(metadata.st_dev), int(metadata.st_ino), int(metadata.st_mode), target,
        )
    return result


def _signal_group(run: RankProcess, signum: signal.Signals) -> None:
    if run.process is not None and run.process.poll() is None:
        try:
            os.killpg(run.process.pid, signum)
        except ProcessLookupError:
            pass
    for pid, started in tuple(run.descendants.items()):
        if _alive(pid, started):
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass


def _reap_owned_children(runs: Sequence[RankProcess]) -> None:
    """Reap only retained worker descendants, never unrelated baseline children."""
    for run in runs:
        for pid in tuple(run.descendants):
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass


def _fd_documents(
    snapshot: Mapping[int, tuple[int, int, int, str]]
) -> list[dict[str, Any]]:
    return [
        {
            "fd": descriptor,
            "device": value[0],
            "inode": value[1],
            "mode": value[2],
            "target": value[3],
        }
        for descriptor, value in sorted(snapshot.items())
    ]


def _process_documents(
    identities: Sequence[tuple[int, str, int, int]]
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "role": role,
            "pid": pid,
            "start_time_ticks": started,
        }
        for rank, role, pid, started in sorted(identities)
    ]


def _poll_workers(runs: Sequence[RankProcess]) -> None:
    for run in runs:
        if run.process is not None and run.returncode is None:
            value = run.process.poll()
            if value is not None:
                run.returncode = value


def terminate_group(
    runs: Sequence[RankProcess], grace_seconds: float,
    baseline_children: set[tuple[int, int]],
) -> bool:
    for run in runs:
        _capture_owned_processes(runs, baseline_children)
        _signal_group(run, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        _capture_owned_processes(runs, baseline_children)
        _poll_workers(runs)
        _reap_owned_children(runs)
        if all(
            (run.process is None or run.process.poll() is not None)
            and not any(_present(pid, started) for pid, started in run.descendants.items())
            for run in runs
        ):
            break
        time.sleep(0.01)
    for run in runs:
        _signal_group(run, signal.SIGKILL)
        if run.process is not None:
            try:
                run.returncode = run.process.wait(timeout=max(grace_seconds, 0.1))
            except subprocess.TimeoutExpired:
                run.returncode = None
        if run.stdout is not None:
            run.stdout.close()
        if run.stderr is not None:
            run.stderr.close()
    reap_deadline = time.monotonic() + grace_seconds
    while time.monotonic() < reap_deadline:
        _capture_owned_processes(runs, baseline_children)
        _reap_owned_children(runs)
        if not any(
            _present(pid, started)
            for run in runs
            for pid, started in run.descendants.items()
        ):
            break
        time.sleep(0.01)
    _capture_owned_processes(runs, baseline_children)
    return not any(
        _present(pid, started)
        for run in runs
        for pid, started in run.descendants.items()
    )
def segment_worker_config(rank_design: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "segment_id": int(segment["index"]),
            "sequence": int(segment["sequence"]),
            "global_offset_elements": int(segment["base_offset_elements"]),
            "element_count": int(segment["element_count"]),
            "byte_count": int(segment["byte_count"]),
            "descriptor_sha256": segment["descriptor_sha256"],
            "plan_sha256": segment["plan_sha256"],
        }
        for segment in rank_design["segments"]
    ]


def load_product(prefix: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    prefix = prefix.resolve(strict=True)
    manifest_path = prefix / "manifest.json"
    manifest, payload = _read_json(manifest_path, "product manifest")
    if manifest.get("schema") != PRODUCT_SCHEMA or manifest.get("prefix") != str(prefix):
        raise RunnerError("product manifest identity mismatch")
    if payload != canonical_json(manifest):
        raise RunnerError("product manifest is not canonical")
    artifacts = manifest.get("artifacts")
    managed = manifest.get("managed_inputs")
    plugins = manifest.get("plugins")
    snapshots = plugins.get("snapshots") if isinstance(plugins, Mapping) else None
    ccl_snapshot = (
        snapshots.get("gemsim-ccl") if isinstance(snapshots, Mapping) else None
    )
    ccl_package_text = (
        ccl_snapshot.get("package_path")
        if isinstance(ccl_snapshot, Mapping)
        else None
    )
    if (
        not isinstance(artifacts, Mapping)
        or not isinstance(managed, Mapping)
        or not isinstance(ccl_package_text, str)
    ):
        raise RunnerError("product manifest artifact bindings are missing")
    ccl_package = Path(ccl_package_text)
    if (
        not ccl_package.is_absolute()
        or ccl_package != Path(os.path.normpath(ccl_package))
    ):
        raise RunnerError("CCL product package path is not canonical")
    paths = {
        "product_manifest": manifest_path,
        "runtime_library": Path(artifacts["runtime_library"]["path"]),
        "ccl_native": Path(artifacts["ccl_plugin_init"]["path"]).with_name("native.py"),
        "ccl_device": Path(artifacts["ccl_plugin_init"]["path"]).with_name("device.py"),
        "ccl_engine": ccl_package / "engine.py",
        "triton_driver": Path(artifacts["triton_plugin_driver"]["path"]),
        "gem5_binary": Path(managed["gem5_binary"]["path"]),
        "gem5_config": Path(managed["gem5_config"]["path"]),
        "verifier": ROOT / "tools/gemsim_ccl_live_allreduce_acceptance.py",
        "runner": Path(__file__).resolve(),
        "worker": ROOT / "examples/triton/ccl_live_allreduce_rank.py",
        "bootstrap": ROOT / "examples/triton/_gemsim_bootstrap.py",
        "design": ROOT / "tools/gemsim_ccl_live_allreduce.py",
        "rank_registry": ROOT / "scripts/gemsim_live_registry.py",
    }
    for role, path in paths.items():
        try:
            is_file = path.resolve(strict=True).is_file()
        except OSError as error:
            raise RunnerError(
                f"product/source identity is missing: {role}"
            ) from error
        if not is_file:
            raise RunnerError(f"product/source identity is missing: {role}")
    runtime_record = file_record(paths["runtime_library"])
    if runtime_record["sha256"] != artifacts["runtime_library"]["sha256"]:
        raise RunnerError("runtime bytes differ from product manifest")
    manifest_bindings = {
        "runtime_library": artifacts["runtime_library"],
        "triton_driver": artifacts["triton_plugin_driver"],
        "gem5_binary": managed["gem5_binary"],
        "gem5_config": managed["gem5_config"],
    }
    for role, binding in manifest_bindings.items():
        observed = file_record(paths[role])
        if (
            observed["path"] != binding["path"]
            or observed["bytes"] != binding["bytes"]
            or observed["sha256"] != binding["sha256"]
        ):
            raise RunnerError(f"{role} bytes differ from product manifest")
    plugin_root = Path(artifacts["ccl_plugin_init"]["path"]).parent
    if (
        ccl_package != plugin_root
        or paths["ccl_native"].parent != plugin_root
        or paths["ccl_device"].parent != plugin_root
        or paths["ccl_engine"].parent != plugin_root
    ):
        raise RunnerError("CCL product module paths escaped the plugin snapshot")
    try:
        plugin_root.resolve(strict=True).relative_to(prefix)
    except ValueError as error:
        raise RunnerError("CCL product package escaped the product prefix") from error
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list):
        raise RunnerError("product manifest inventory is missing")
    inventory_by_path = {
        item.get("path"): item
        for item in inventory
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }
    for role in ("ccl_native", "ccl_device", "ccl_engine"):
        relative = paths[role].resolve(strict=True).relative_to(prefix).as_posix()
        binding = inventory_by_path.get(relative)
        observed = file_record(paths[role])
        if (
            not isinstance(binding, Mapping)
            or binding.get("kind") != "regular"
            or observed["bytes"] != binding.get("bytes")
            or observed["sha256"] != binding.get("sha256")
        ):
            raise RunnerError(f"{role} bytes differ from product manifest inventory")
    imported_record = file_record(Path(_imported_native.__file__))
    product_native_record = file_record(paths["ccl_native"])
    if (
        imported_record["bytes"] != product_native_record["bytes"]
        or imported_record["sha256"] != product_native_record["sha256"]
    ):
        raise RunnerError("executed NativeCCL source differs from frozen product snapshot")
    return manifest, paths


def source_identity(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    result = {role: file_record(paths[role]) for role in IDENTITY_ROLES}
    if set(result) != set(IDENTITY_ROLES):
        raise RunnerError("source identity roles are incomplete")
    return result


def default_product_prefix() -> Path:
    result = subprocess.run(
        ["/usr/bin/bash", str(ROOT / "scripts/setup_rocm_env.sh"), "--print-prefix"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RunnerError(result.stderr.strip() or "active product query failed")
    return Path(result.stdout.strip())


def _worker_environment(prefix: Path, launch_path: Path, launch: Mapping[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    environment["ROCM_SIM_ROOT"] = str(prefix)
    environment["TRITON_CACHE_DIR"] = launch["paths"]["triton_cache_directory"]
    environment["GEMSIM_RANK_LAUNCH_DESCRIPTOR"] = str(launch_path)
    for name in tuple(environment):
        if name.startswith("SAGR_"):
            del environment[name]
    return environment


def _prepare_rank_tree(output: Path, rank_design: Mapping[str, Any]) -> RankProcess:
    rank = int(rank_design["rank"])
    directory = output / f"rank-{rank:02d}"
    directory.mkdir(mode=0o700)
    launch = dict(rank_design["rank_launch"])
    for path in (
        Path(launch["paths"]["instance_directory"]),
        Path(launch["paths"]["triton_cache_directory"]),
    ):
        path.mkdir(mode=0o700, parents=True)
        os.chmod(path, 0o700)
    launch_path = directory / "rank-launch.json"
    launch_payload = canonical_json(launch)
    _exclusive_write(launch_path, launch_payload)
    os.chmod(launch_path, 0o400, follow_symlinks=False)
    observed, observed_payload = _read_json(launch_path, "rank launch descriptor")
    if observed != launch or observed_payload != launch_payload:
        raise RunnerError("rank launch descriptor did not round-trip exactly")
    return RankProcess(rank=rank, directory=directory, launch=launch)


def spawn_rank_process(
    run: RankProcess,
    *,
    config_path: Path,
    product_prefix: Path,
    popen_factory: Callable[..., subprocess.Popen[bytes]],
) -> subprocess.Popen[bytes]:
    """Launch one worker with only its own capability descriptor inherited."""
    run.stdout = (run.directory / ".worker-stdout.log").open("xb")
    run.stderr = (run.directory / ".worker-stderr.log").open("xb")
    command = [
        sys.executable,
        str(ROOT / "examples/triton/ccl_live_allreduce_rank.py"),
        "--config", str(config_path),
        "--capability-fd", str(run.capability_fd),
    ]
    try:
        return popen_factory(
            command,
            stdin=subprocess.DEVNULL,
            stdout=run.stdout,
            stderr=run.stderr,
            close_fds=True,
            pass_fds=(run.capability_fd,),
            start_new_session=True,
            cwd=run.directory,
            env=_worker_environment(
                product_prefix, run.directory / "rank-launch.json", run.launch
            ),
        )
    except BaseException:
        run.stdout.close()
        run.stderr.close()
        run.stdout = None
        run.stderr = None
        raise


def _copy_runtime_artifacts(run: RankProcess, execution_root: Path) -> None:
    runtime = Path(run.launch["paths"]["runtime_directory"])
    sources = {
        "dispatch-trace.jsonl": Path(run.launch["paths"]["dispatch_trace_path"]),
        "stats.txt": Path(run.launch["paths"]["gem5_output_directory"]) / "stats.txt",
        "gem5.log": Path(run.launch["paths"]["gem5_log_path"]),
    }
    for name, source in sources.items():
        destination = run.directory / name
        if source.is_file():
            _exclusive_write(destination, source.read_bytes())
        else:
            _exclusive_write(destination, b"")
    execution_rank = execution_root / f"rank-{run.rank:02d}"
    instance = Path(run.launch["paths"]["instance_directory"])
    cache = Path(run.launch["paths"]["triton_cache_directory"])
    if (
        instance != execution_rank / "correctness"
        or cache != execution_rank / "cache/triton"
        or execution_rank.is_symlink()
        or not execution_rank.is_dir()
    ):
        raise RunnerError("refusing unsafe execution namespace cleanup")
    shutil.rmtree(execution_rank)
    if execution_rank.exists():
        raise RunnerError("rank execution namespace survived evidence materialization")


def _normalize_failure_status(status: int) -> str:
    if status == TIMED_OUT:
        return "timed_out"
    if status == PEER_LOST:
        return "peer_lost"
    return "device_failure"


def canonical_failure_bundle(
    status: str,
    first_error: Any,
    rank_results: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, bool]:
    """Select the live first error and only its reporter's unmodified tuple."""
    first_error_document = None
    if first_error is not None and all(
        hasattr(first_error, name)
        for name in (
            "status", "reporter_rank", "failed_rank", "context_sequence"
        )
    ):
        status = _normalize_failure_status(int(first_error.status))
        first_error_document = {
            "status": status,
            "native_status": int(first_error.status),
            "reporter_rank": int(first_error.reporter_rank),
            "failed_rank": int(first_error.failed_rank),
            "context_sequence": int(first_error.context_sequence),
        }
    if first_error_document is None:
        for result in rank_results:
            candidate = result.get("first_error")
            if (
                isinstance(candidate, Mapping)
                and candidate.get("status") in (
                    "device_failure", "peer_lost", "timed_out"
                )
                and isinstance(candidate.get("context_sequence"), int)
            ):
                first_error_document = dict(candidate)
                status = str(candidate["status"])
                break

    failed_transfer = None
    failed_ack_sent = False
    if first_error_document is not None:
        failed_sequence = int(first_error_document["context_sequence"])
        reporter_rank = first_error_document.get("reporter_rank")
        candidates = (
            [rank_results[reporter_rank]]
            if (
                isinstance(reporter_rank, int)
                and 0 <= reporter_rank < len(rank_results)
            )
            else []
        )
        for result in candidates:
            candidate = result.get("failed_transfer")
            if (
                isinstance(candidate, Mapping)
                and candidate.get("sequence") == failed_sequence
            ):
                failed_transfer = dict(candidate)
                failed_ack_sent = bool(result.get("failed_ack_sent"))
                break
    return status, first_error_document, failed_transfer, failed_ack_sent


def normalize_rank_failure_results(
    rank_results: Sequence[dict[str, Any]],
    *,
    status: str,
    first_error: Mapping[str, Any] | None,
    failed_transfer: Mapping[str, Any] | None,
    failed_ack_sent: bool,
) -> None:
    sequence = None if first_error is None else int(first_error["context_sequence"])
    for result in rank_results:
        result.update(
            {
                "status": status,
                "first_error": first_error,
                "failed_transfer": failed_transfer,
                "failed_descriptor_sequence": sequence,
                "failed_ack_sent": failed_ack_sent,
                "public_result_published": False,
                "public_commit_count": 0,
            }
        )


def deterministic_input(dtype: str, rank: int, element_count: int) -> bytes:
    """Reproduce the worker input for failure evidence without importing torch."""
    import struct

    values = [(((index * 13 + rank * 29) % 127) - 63) / 16.0
              for index in range(element_count)]
    if dtype == "float32":
        return b"".join(struct.pack("<f", value) for value in values)
    if dtype != "bfloat16":
        raise RunnerError(f"unsupported collective dtype: {dtype}")
    result = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        rounded = bits + 0x7FFF + ((bits >> 16) & 1)
        result.extend(struct.pack("<H", (rounded >> 16) & 0xFFFF))
    return bytes(result)


def supervise(
    *,
    expected_path: Path,
    output: Path,
    execution_root: Path,
    product_prefix: Path,
    timeout_seconds: float,
    cleanup_grace_seconds: float,
    popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    native_factory: Callable[[Path], Any] = NativeCCL,
) -> dict[str, Any]:
    expected_path = expected_path.resolve(strict=True)
    output = Path(os.path.abspath(output))
    execution_root = Path(os.path.abspath(execution_root))
    if output.exists() or output.is_symlink():
        raise RunnerError(f"output directory must be absent: {output}")
    if execution_root.exists() or execution_root.is_symlink():
        raise RunnerError(
            f"execution root must be absent: {execution_root}"
        )
    if (
        output == execution_root
        or output in execution_root.parents
        or execution_root in output.parents
        or execution_root.parent.resolve(strict=True) != execution_root.parent
    ):
        raise RunnerError("execution root is not an independent normalized namespace")
    if timeout_seconds <= 0 or cleanup_grace_seconds <= 0:
        raise RunnerError("timeouts must be positive")
    expected, expected_payload = _read_json(expected_path, "expected wrapper")
    if expected_payload != canonical_json(expected):
        raise RunnerError("expected wrapper is not canonical JSON")
    if expected.get("schema") != EXPECTED_SCHEMA or not isinstance(expected.get("design"), Mapping):
        raise RunnerError("expected wrapper schema mismatch")
    design = _validate_expected_and_paths(expected, output, execution_root)
    config = design["config"]
    world = int(config["world_size"])
    if not 2 <= world <= 16:
        raise RunnerError("world_size must be in 2..16")
    if [rank.get("rank") for rank in design["ranks"]] != list(range(world)):
        raise RunnerError("expected ranks are not exactly 0..world_size-1")
    if any(
        int(rank.get("expected", {}).get("device_sum_launches", 0)) <= 0
        for rank in design["ranks"]
    ):
        raise RunnerError(
            "live execution requires at least one device SUM on every rank; "
            "zero-dispatch session shutdown is not yet acceptance-capable"
        )
    manifest, product_paths = load_product(product_prefix)
    preflight = source_identity(product_paths)
    runtime_path = product_paths["runtime_library"].resolve(strict=True)
    native = native_factory(runtime_path)
    if native.library_sha256 != design["runtime"]["sha256"]:
        raise RunnerError("expected design runtime differs from frozen product")
    identity = native.identity(
        world_size=world, epoch=int(config["epoch"]),
        group_generation=int(config["group_generation"]),
        job_uuid=bytes.fromhex(config["job_uuid"]),
        group_uuid=bytes.fromhex(config["group_uuid"]),
        model_identity_sha256=bytes.fromhex(config["model_identity_sha256"]),
    )
    started_at_ns = native.monotonic_time_ns()
    timeout_ns = max(1, int(timeout_seconds * 1_000_000_000))
    absolute_deadline_ns = started_at_ns + timeout_ns
    output.mkdir(mode=0o700)
    os.chmod(output, 0o700)
    execution_root.mkdir(mode=0o700)
    os.chmod(execution_root, 0o700)
    enable_subreaper()
    baseline_fds = owned_fd_snapshot()
    baseline_children = _direct_child_identities(os.getpid())
    runs = [_prepare_rank_tree(output, item) for item in design["ranks"]]
    first_error = None
    status = "device_failure"
    interrupted = False
    launching_rank = 0
    broker = None
    try:
        broker = native.live_broker(identity)
        owner = broker.owner
        for run, rank_design in zip(runs, design["ranks"]):
            launching_rank = run.rank
            run.capability_fd = broker.prepare_rank(run.rank)
            worker_config = {
                "schema": WORKER_CONFIG_SCHEMA,
                "rank": run.rank,
                "world_size": world,
                "element_count": int(config["element_count"]),
                "dtype": config["dtype"],
                "epoch": int(config["epoch"]),
                "group_generation": int(config["group_generation"]),
                "job_uuid": config["job_uuid"],
                "group_uuid": config["group_uuid"],
                "model_identity_sha256": config["model_identity_sha256"],
                "broker_pid": int(owner.pid),
                "broker_start_time_ticks": int(owner.start_time_ticks),
                "absolute_deadline_ns": absolute_deadline_ns,
                "credits_per_peer": int(design["limits"]["credits_per_peer"]),
                "runtime_library": str(runtime_path),
                "rank_launch_sha256": rank_design["rank_launch_sha256"],
                "segments": segment_worker_config(rank_design),
                "result_path": str(run.directory / "worker-result.json"),
                "journal_path": str(run.directory / "step-journal.jsonl"),
                "input_path": str(run.directory / "input.bin"),
                "output_path": str(run.directory / "output.bin"),
                "product": {
                    "product_id": manifest["product_id"],
                    "manifest_sha256": preflight["product_manifest"]["sha256"],
                    "prefix": str(product_prefix.resolve(strict=True)),
                    "ccl_engine": preflight["ccl_engine"],
                },
            }
            config_path = run.directory / ".worker-config.json"
            _exclusive_write(config_path, canonical_json(worker_config))
            run.process = spawn_rank_process(
                run,
                config_path=config_path,
                product_prefix=product_prefix,
                popen_factory=popen_factory,
            )
            os.close(run.capability_fd)
            run.capability_fd = -1
            process_identity = native.process_identity(run.process.pid)
            run.start_time_ticks = int(process_identity.start_time_ticks)
            broker.bind_rank(run.rank, process_identity)

        broker.rendezvous(absolute_deadline_ns)
        while True:
            _capture_owned_processes(runs, baseline_children)
            _poll_workers(runs)
            observed = broker.progress()
            if observed is not None:
                first_error = observed
                status = _normalize_failure_status(int(observed.status))
            if all(run.returncode is not None for run in runs):
                break
            if first_error is not None:
                break
            if native.monotonic_time_ns() >= absolute_deadline_ns:
                status = "timed_out"
                first_error = broker.abort(0, TIMED_OUT, 1)
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        interrupted = True
        status = "device_failure"
        try:
            first_error = broker.abort(0, CANCELLED, 1)
        except Exception:
            pass
    except CCLStatusError as error:
        status = _normalize_failure_status(int(error.status))
        try:
            first_error = broker.first_error() if broker is not None else None
        except Exception:
            first_error = None
    except Exception:
        status = "device_failure"
        if broker is not None:
            try:
                first_error = broker.abort(launching_rank, PROTOCOL_ERROR, 1)
            except Exception:
                try:
                    first_error = broker.first_error()
                except Exception:
                    first_error = None
    finally:
        for run in runs:
            if run.capability_fd >= 0:
                os.close(run.capability_fd)
                run.capability_fd = -1
        failed_workers = [
            run for run in runs
            if run.process is not None and run.returncode not in (None, 0)
        ]
        if broker is not None and first_error is None and (failed_workers or interrupted):
            failed_rank = failed_workers[0].rank if failed_workers else launching_rank
            try:
                first_error = broker.abort(failed_rank, PEER_LOST, 1)
                status = "peer_lost"
            except Exception:
                pass
        relay_deadline_ns = native.monotonic_time_ns() + max(
            1, int(cleanup_grace_seconds * 1_000_000_000)
        )
        try:
            while (
                broker is not None
                and
                broker.info().abort_pending_mask
                and native.monotonic_time_ns() < relay_deadline_ns
            ):
                relayed = broker.progress()
                if first_error is None and relayed is not None:
                    first_error = relayed
                time.sleep(0.001)
        except Exception:
            pass
        children_exhausted = terminate_group(
            runs, cleanup_grace_seconds, baseline_children
        )
        if broker is not None:
            try:
                broker.destroy()
            except Exception:
                children_exhausted = False

    post_cleanup_fds = owned_fd_snapshot()
    fd_added = dict(set(post_cleanup_fds.items()) - set(baseline_fds.items()))
    fd_removed = dict(set(baseline_fds.items()) - set(post_cleanup_fds.items()))
    owned_fd_delta = len(fd_added) + len(fd_removed)
    orphan_count = sum(
        _present(pid, started)
        for run in runs
        for pid, started in run.descendants.items()
    )
    workers_reaped = all(
        run.process is None or run.process.poll() is not None for run in runs
    )
    cleanup_complete = (
        children_exhausted
        and workers_reaped
        and owned_fd_delta == 0
        and orphan_count == 0
    )
    if not cleanup_complete:
        status = "device_failure"

    rank_results: list[dict[str, Any]] = []
    for run in runs:
        result_path = run.directory / "worker-result.json"
        if result_path.is_file():
            result, _ = _read_json(result_path, f"rank {run.rank} result")
        else:
            input_payload = deterministic_input(
                str(config["dtype"]), run.rank, int(config["element_count"])
            )
            result = {
                "schema": "amdgpu-sim.ccl-live-allreduce-rank-result.v1",
                "status": status,
                "rank": run.rank,
                "world_size": world,
                "acceptance_authority": False,
                "live_collective_accepted": False,
                "public_result_published": False,
                "public_commit_count": 0,
                "failed_transfer": None,
                "first_error": None,
                "input_sha256_before": sha256_bytes(input_payload),
                "input_sha256_after": sha256_bytes(input_payload),
                "product": {
                    "product_id": manifest["product_id"],
                    "manifest_sha256": preflight["product_manifest"]["sha256"],
                    "prefix": str(product_prefix.resolve(strict=True)),
                    "ccl_engine": preflight["ccl_engine"],
                },
            }
            _exclusive_write(result_path, canonical_json(result))
        rank_results.append(result)
        for name in ("step-journal.jsonl", "output.bin"):
            path = run.directory / name
            if not path.exists():
                _exclusive_write(path, b"")
        input_path = run.directory / "input.bin"
        if not input_path.exists():
            input_payload = deterministic_input(
                str(config["dtype"]), run.rank, int(config["element_count"])
            )
            _exclusive_write(input_path, input_payload)
        _copy_runtime_artifacts(run, execution_root)
        for hidden in (".worker-config.json", ".worker-stdout.log", ".worker-stderr.log"):
            try:
                (run.directory / hidden).unlink()
            except FileNotFoundError:
                pass
    try:
        execution_root.rmdir()
    except OSError:
        cleanup_complete = False
        status = "device_failure"

    all_success = cleanup_complete and all(
        run.returncode == 0 and result.get("status") == "success"
        for run, result in zip(runs, rank_results)
    )
    if all_success and first_error is None and not interrupted:
        status = "success"
    elif status == "success":
        status = "device_failure"
    status, first_error_document, failed_transfer, failed_ack_sent = (
        canonical_failure_bundle(status, first_error, rank_results)
    )

    if status != "success":
        normalize_rank_failure_results(
            rank_results,
            status=status,
            first_error=first_error_document,
            failed_transfer=failed_transfer,
            failed_ack_sent=failed_ack_sent,
        )
        for result, run in zip(rank_results, runs):
            output_path = run.directory / "output.bin"
            if output_path.stat().st_size:
                output_path.unlink()
                _exclusive_write(output_path, b"")
            _atomic_manifest(run.directory / "worker-result.json", result)

    postflight = source_identity(product_paths)
    completed_at_ns = native.monotonic_time_ns()
    baseline_fd_documents = _fd_documents(baseline_fds)
    post_fd_documents = _fd_documents(post_cleanup_fds)
    new_children: list[tuple[int, str, int, int]] = []
    for run in runs:
        if run.process is not None and run.start_time_ticks is not None:
            new_children.append(
                (run.rank, "worker", run.process.pid, run.start_time_ticks)
            )
        new_children.extend(
            (run.rank, "daemon_or_descendant", pid, started)
            for pid, started in run.descendants.items()
        )
    new_child_documents = _process_documents(new_children)
    orphan_documents = _process_documents(
        [
            identity
            for identity in new_children
            if _present(identity[2], identity[3])
        ]
    )
    supervisor_cleanup = {
        "baseline_fds": baseline_fd_documents,
        "baseline_fd_count": len(baseline_fd_documents),
        "baseline_fd_sha256": object_sha256(baseline_fd_documents),
        "post_fds": post_fd_documents,
        "post_fd_count": len(post_fd_documents),
        "post_fd_sha256": object_sha256(post_fd_documents),
        "added_fds": _fd_documents(fd_added),
        "removed_fds": _fd_documents(fd_removed),
        "measured_fd_delta": owned_fd_delta,
        "children_exhausted": children_exhausted,
        "workers_reaped": workers_reaped,
        "new_child_identities": new_child_documents,
        "orphan_identities": orphan_documents,
        "all_clear": cleanup_complete,
    }
    rank_entries = []
    for run in runs:
        rank_entries.append(
            {
                "rank": run.rank,
                "worker_pid": None if run.process is None else run.process.pid,
                "worker_start_time_ticks": run.start_time_ticks,
                "returncode": run.returncode,
                "artifacts": {
                    name: artifact_record(output, run.directory / name)
                    for name in RANK_FILES
                },
                "cleanup": {
                    "worker_reaped": run.process is None or run.process.poll() is not None,
                    "daemon_reaped": not any(
                        _present(pid, started)
                        for pid, started in run.descendants.items()
                    ),
                },
            }
        )
    result_manifest = {
        "schema": RUN_SCHEMA,
        "status": status,
        "acceptance_authority": False,
        "live_collective_accepted": False,
        "expected": {
            "schema": EXPECTED_SCHEMA,
            "bytes": len(expected_payload),
            "sha256": sha256_bytes(expected_payload),
        },
        "world_size": world,
        "element_count": int(config["element_count"]),
        "dtype": config["dtype"],
        "job_uuid": config["job_uuid"],
        "group_uuid": config["group_uuid"],
        "epoch": int(config["epoch"]),
        "group_generation": int(config["group_generation"]),
        "segments": design["segmentation"]["segments"],
        "started_at_ns": started_at_ns,
        "completed_at_ns": completed_at_ns,
        "absolute_deadline_ns": absolute_deadline_ns,
        "target_execution_completed": status == "success",
        "target_feedback": False,
        "oracle_phase": "post_target",
        "oracle_feedback": False,
        "public_commit_count": world if status == "success" else 0,
        "failed_ack_sent": failed_ack_sent if status != "success" else None,
        "first_error": first_error_document,
        "failed_transfer": failed_transfer,
        "supervisor_cleanup": supervisor_cleanup,
        "source_identity_preflight": preflight,
        "source_identity_postflight": postflight,
        "ranks": rank_entries,
    }
    _atomic_manifest(output / "result-manifest.json", result_manifest)
    return result_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--product-prefix", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--cleanup-grace-seconds", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prefix = args.product_prefix or default_product_prefix()
    manifest = supervise(
        expected_path=args.expected,
        output=args.output_dir,
        execution_root=args.execution_root,
        product_prefix=prefix,
        timeout_seconds=args.timeout_seconds,
        cleanup_grace_seconds=args.cleanup_grace_seconds,
    )
    return 0 if manifest["status"] == "success" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
