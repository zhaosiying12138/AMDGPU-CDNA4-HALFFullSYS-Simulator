#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run isolated gemsim correctness workers and retain their namespaces."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
from typing import Any, Iterable, Sequence

from gemsim_live_registry import (
    LiveRegistryPublisher,
    RegistryError,
    ensure_private_directory,
    load_rank_launch,
    make_rank_launch,
    validate_rank_launch_group,
    write_rank_launch,
)


PR_SET_CHILD_SUBREAPER = 36
DEVICE_VISIBILITY_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)
PRODUCTION_DEVICE_PREFIXES = ("/dev/dri/", "/dev/nvidia")
PRODUCTION_DEVICE_PATHS = {"/dev/kfd"}
READY_PATTERN = re.compile(
    r"host-gpu-ready .*daemon_uuid=([0-9a-f]{32}) "
    r"job_uuid=([0-9a-f]{32}) epoch=([0-9]+) rank=([0-9]+) world=([0-9]+)"
)
CLEAN_EXIT_PATTERN = re.compile(
    r"host-gpu-dispatch-exit cause=host GPU dispatch session complete code=0 "
)


class SupervisorError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_time: int
    process_group: int
    state: str


@dataclass
class RuntimeArtifacts:
    pid: int
    command: list[str]
    endpoint: str
    run_dir: str
    output_dir: str
    trace_path: str
    stats_path: str
    gem5_log_path: str
    gem5_cache_dir: str
    epoch: int
    job_uuid: str
    rank: int
    world_size: int
    daemon_uuid: str | None = None
    observed_job_uuid: str | None = None
    observed_epoch: int | None = None
    observed_rank: int | None = None
    observed_world_size: int | None = None

    def refresh_ready_identity(self) -> None:
        log = Path(self.gem5_log_path)
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        match = READY_PATTERN.search(text)
        if match is None:
            return
        daemon_uuid, job_uuid, epoch, rank, world_size = match.groups()
        self.daemon_uuid = daemon_uuid
        self.observed_job_uuid = job_uuid
        self.observed_epoch = int(epoch)
        self.observed_rank = int(rank)
        self.observed_world_size = int(world_size)

    def has_clean_exit(self) -> bool:
        try:
            text = Path(self.gem5_log_path).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return False
        return CLEAN_EXIT_PATTERN.search(text) is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "command": self.command,
            "endpoint": self.endpoint,
            "run_dir": self.run_dir,
            "output_dir": self.output_dir,
            "trace_path": self.trace_path,
            "stats_path": self.stats_path,
            "gem5_log_path": self.gem5_log_path,
            "gem5_cache_dir": self.gem5_cache_dir,
            "daemon_uuid": self.daemon_uuid,
            "job_uuid": self.job_uuid,
            "epoch": self.epoch,
            "rank": self.rank,
            "world_size": self.world_size,
            "observed_identity": {
                "job_uuid": self.observed_job_uuid,
                "epoch": self.observed_epoch,
                "rank": self.observed_rank,
                "world_size": self.observed_world_size,
            },
        }


@dataclass
class InstanceRun:
    index: int
    phase: str
    directory: Path
    cache_dir: Path
    command: list[str]
    cpu_set: set[int] | None = None
    process: subprocess.Popen[bytes] | None = None
    stdout: Any = None
    stderr: Any = None
    started_at: float = 0.0
    ended_at: float = 0.0
    returncode: int | None = None
    timed_out: bool = False
    tracked: dict[int, ProcessIdentity] = field(default_factory=dict)
    runtimes: dict[int, RuntimeArtifacts] = field(default_factory=dict)
    isolation_errors: list[str] = field(default_factory=list)
    audit_after: dict[int, float] = field(default_factory=dict)
    audit_samples: int = 0
    audit_errors: set[str] = field(default_factory=set)
    sagr_environment_leaks: set[str] = field(default_factory=set)
    production_device_fds: set[str] = field(default_factory=set)
    rank_launch: dict[str, Any] | None = None
    rank_launch_path: Path | None = None

    @property
    def elapsed_seconds(self) -> float:
        end = self.ended_at or time.monotonic()
        return max(0.0, end - self.started_at)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "phase": self.phase,
            "directory": str(self.directory),
            "cache_dir": str(self.cache_dir),
            "command": self.command,
            "cpu_set": sorted(self.cpu_set) if self.cpu_set is not None else None,
            "pid": self.process.pid if self.process is not None else None,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "elapsed_seconds": self.elapsed_seconds,
            "stdout_path": str(self.directory / "stdout.log"),
            "stderr_path": str(self.directory / "stderr.log"),
            "runtime_processes": [
                runtime.as_dict()
                for runtime in sorted(self.runtimes.values(), key=lambda item: item.pid)
            ],
            "isolation_errors": self.isolation_errors,
            "process_audit": {
                "scope": "sampled Linux procfs environment and open-file audit",
                "samples": self.audit_samples,
                "errors": sorted(self.audit_errors),
                "sagr_environment_leaks": sorted(self.sagr_environment_leaks),
                "production_device_fds": sorted(self.production_device_fds),
            },
            "rank_launch_descriptor": (
                {
                    "path": str(self.rank_launch_path),
                    "document": self.rank_launch,
                }
                if self.rank_launch is not None
                else None
            ),
            "passed": self.returncode == 0 and not self.isolation_errors,
        }


@dataclass
class GroupRegistrySession:
    publisher: LiveRegistryPublisher
    job_uuid: str
    epoch: int
    world_size: int
    state: str | None = None
    abort_reason: str | None = None
    unhealthy_since: dict[str, float] = field(default_factory=dict)

    def _rank_entry(self, run: InstanceRun, state: str) -> dict[str, Any]:
        assert run.rank_launch is not None
        paths = run.rank_launch["paths"]
        runtime = next(iter(run.runtimes.values()), None) if state == "READY" else None
        return {
            "rank": run.rank_launch["rank"],
            "world_size": self.world_size,
            "state": state,
            "worker_pid": run.process.pid if run.process is not None else None,
            "daemon_pid": runtime.pid if runtime is not None else None,
            "daemon_uuid": runtime.daemon_uuid if runtime is not None else None,
            "endpoint": paths["endpoint"],
            "runtime_directory": paths["runtime_directory"],
            "triton_cache_directory": paths["triton_cache_directory"],
            "gem5_cache_directory": paths["gem5_cache_directory"],
        }

    def publish(self, state: str, runs: list[InstanceRun]) -> None:
        if state == self.state:
            return
        ranks = [self._rank_entry(run, state) for run in sorted(runs, key=lambda item: item.index)]
        document = self.publisher.base_document(
            state=state,
            job_uuid=self.job_uuid,
            epoch=self.epoch,
            world_size=self.world_size,
            ranks=ranks,
        )
        self.publisher.publish(document)
        self.state = state

    def refresh(self, runs: list[InstanceRun]) -> None:
        if self.state != "STARTING":
            return
        ready = True
        for run in runs:
            if len(run.runtimes) != 1:
                ready = False
                continue
            runtime = next(iter(run.runtimes.values()))
            runtime.refresh_ready_identity()
            expected = (
                runtime.job_uuid,
                runtime.epoch,
                runtime.rank,
                runtime.world_size,
            )
            observed = (
                runtime.observed_job_uuid,
                runtime.observed_epoch,
                runtime.observed_rank,
                runtime.observed_world_size,
            )
            if observed != expected or runtime.daemon_uuid is None:
                ready = False
        if ready:
            self.publish("READY", runs)

    def ready_health_error(
        self, runs: list[InstanceRun], active_positions: set[int]
    ) -> str | None:
        if self.state != "READY":
            return None
        observed_errors: set[str] = set()
        for position in sorted(active_positions):
            run = runs[position]
            if run.process is None:
                return f"rank {run.index} worker process was not started"
            worker = run.tracked.get(run.process.pid)
            if worker is None or not _identity_alive(worker):
                observed_errors.add(f"rank {run.index} worker identity was lost")
                continue
            if len(run.runtimes) != 1:
                observed_errors.add(
                    f"rank {run.index} no longer has exactly one daemon"
                )
                continue
            runtime = next(iter(run.runtimes.values()))
            daemon = run.tracked.get(runtime.pid)
            if (
                daemon is None or not _identity_alive(daemon)
            ) and not runtime.has_clean_exit():
                observed_errors.add(f"rank {run.index} daemon identity was lost")
        now = time.monotonic()
        for reason in list(self.unhealthy_since):
            if reason not in observed_errors:
                del self.unhealthy_since[reason]
        for reason in sorted(observed_errors):
            first = self.unhealthy_since.setdefault(reason, now)
            if now - first >= 0.10:
                return reason
        return None

    def abort(self, runs: list[InstanceRun], reason: str) -> None:
        if self.abort_reason is None:
            self.abort_reason = reason
        self.publish("OFF", runs)


def _enable_subreaper() -> None:
    if not sys.platform.startswith("linux") or not Path("/proc/self/stat").exists():
        raise SupervisorError("the gemsim supervisor requires Linux procfs")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        native_errno = ctypes.get_errno()
        raise SupervisorError(f"could not enable child subreaper: errno {native_errno}")


def _process_identity(pid: int) -> ProcessIdentity | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError:
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        return ProcessIdentity(
            pid=pid,
            start_time=int(fields[19]),
            process_group=int(fields[2]),
            state=fields[0],
        )
    except ValueError:
        return None


def _children(pid: int) -> list[int]:
    try:
        text = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="ascii")
    except OSError:
        return []
    result = []
    for value in text.split():
        try:
            result.append(int(value))
        except ValueError:
            continue
    return result


def _descendants(pid: int) -> list[int]:
    pending = _children(pid)
    seen: set[int] = set()
    while pending:
        child = pending.pop()
        if child in seen:
            continue
        seen.add(child)
        pending.extend(_children(child))
    return sorted(seen)


def _cmdline(pid: int) -> list[str]:
    try:
        value = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", "replace") for part in value.split(b"\0") if part]


def _argument(command: Sequence[str], name: str) -> str | None:
    try:
        index = command.index(name)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _runtime_from_command(pid: int, command: list[str]) -> RuntimeArtifacts | None:
    endpoint = _argument(command, "--endpoint")
    output_dir = _argument(command, "--outdir")
    trace_path = _argument(command, "--dispatch-trace-path")
    epoch_text = _argument(command, "--epoch")
    job_uuid = _argument(command, "--job-uuid")
    rank_text = _argument(command, "--rank")
    world_text = _argument(command, "--world-size")
    if None in (endpoint, output_dir, trace_path, epoch_text, job_uuid, rank_text, world_text):
        return None
    try:
        epoch = int(str(epoch_text), 0)
        rank = int(str(rank_text), 0)
        world_size = int(str(world_text), 0)
    except ValueError:
        return None
    endpoint_path = Path(str(endpoint))
    run_dir = endpoint_path.parent
    return RuntimeArtifacts(
        pid=pid,
        command=command,
        endpoint=str(endpoint_path),
        run_dir=str(run_dir),
        output_dir=str(Path(str(output_dir))),
        trace_path=str(Path(str(trace_path))),
        stats_path=str(Path(str(output_dir)) / "stats.txt"),
        gem5_log_path=str(run_dir / "gem5.log"),
        gem5_cache_dir=str(run_dir / "cache"),
        epoch=epoch,
        job_uuid=str(job_uuid),
        rank=rank,
        world_size=world_size,
    )


def _audit_process(run: InstanceRun, pid: int) -> None:
    try:
        environment = Path(f"/proc/{pid}/environ").read_bytes()
        for entry in environment.split(b"\0"):
            name = entry.split(b"=", 1)[0]
            if name.startswith(b"SAGR_"):
                run.sagr_environment_leaks.add(
                    f"pid={pid}:{name.decode('ascii', 'replace')}"
                )
    except FileNotFoundError:
        return
    except OSError as error:
        run.audit_errors.add(f"pid={pid}:environ:errno={error.errno}")
    try:
        entries = list(Path(f"/proc/{pid}/fd").iterdir())
    except FileNotFoundError:
        return
    except OSError as error:
        run.audit_errors.add(f"pid={pid}:fd:errno={error.errno}")
        return
    for entry in entries:
        try:
            target = os.readlink(entry)
        except FileNotFoundError:
            continue
        except OSError as error:
            run.audit_errors.add(f"pid={pid}:fd-readlink:errno={error.errno}")
            continue
        normalized = target.removesuffix(" (deleted)")
        if normalized in PRODUCTION_DEVICE_PATHS or normalized.startswith(
            PRODUCTION_DEVICE_PREFIXES
        ):
            run.production_device_fds.add(f"pid={pid}:{normalized}")
    run.audit_samples += 1


def _capture_process(run: InstanceRun, pid: int) -> None:
    identity = _process_identity(pid)
    if identity is None:
        return
    previous = run.tracked.get(pid)
    if previous is not None and previous.start_time != identity.start_time:
        return
    run.tracked[pid] = identity
    now = time.monotonic()
    if now >= run.audit_after.get(pid, 0.0):
        _audit_process(run, pid)
        run.audit_after[pid] = now + 1.0
    if pid not in run.runtimes:
        runtime = _runtime_from_command(pid, _cmdline(pid))
        if runtime is not None:
            run.runtimes[pid] = runtime


def _capture_descendants(run: InstanceRun) -> None:
    if run.process is None:
        return
    _capture_process(run, run.process.pid)
    for pid in _descendants(run.process.pid):
        _capture_process(run, pid)


def _identity_alive(identity: ProcessIdentity) -> bool:
    current = _process_identity(identity.pid)
    return (
        current is not None
        and current.start_time == identity.start_time
        and current.state != "Z"
    )


def _signal_identity(identity: ProcessIdentity, signum: signal.Signals) -> None:
    if not _identity_alive(identity):
        return
    own_group = os.getpgrp()
    if identity.process_group > 1 and identity.process_group != own_group:
        leader = _process_identity(identity.process_group)
        if leader is not None:
            try:
                os.killpg(identity.process_group, signum)
                return
            except ProcessLookupError:
                return
    try:
        os.kill(identity.pid, signum)
    except ProcessLookupError:
        pass


def _reap_adopted_children() -> None:
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid <= 0:
            return


def _reap_tracked_children(identities: Iterable[ProcessIdentity]) -> None:
    for identity in identities:
        try:
            os.waitpid(identity.pid, os.WNOHANG)
        except (ChildProcessError, ProcessLookupError):
            pass


def _terminate_run(run: InstanceRun, grace: float) -> None:
    _capture_descendants(run)
    process = run.process
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGCONT)
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for identity in list(run.tracked.values()):
        _signal_identity(identity, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        _capture_descendants(run)
        direct_alive = process is not None and process.poll() is None
        tracked_alive = any(_identity_alive(item) for item in run.tracked.values())
        if not direct_alive and not tracked_alive:
            break
        time.sleep(0.01)
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for identity in list(run.tracked.values()):
        _signal_identity(identity, signal.SIGKILL)
    if process is not None:
        try:
            process.wait(timeout=max(grace, 0.1))
        except subprocess.TimeoutExpired:
            pass
    _reap_tracked_children(run.tracked.values())


def _parse_cpu_set(value: str) -> set[int]:
    result: set[int] = set()
    try:
        for part in value.split(","):
            bounds = part.split("-", 1)
            first = int(bounds[0])
            last = int(bounds[1]) if len(bounds) == 2 else first
            if first < 0 or last < first:
                raise ValueError
            result.update(range(first, last + 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"invalid CPU set: {value}") from error
    if not result:
        raise argparse.ArgumentTypeError("CPU set must not be empty")
    return result


def _affinity_setter(cpu_set: set[int]):
    def set_affinity() -> None:
        os.sched_setaffinity(0, cpu_set)

    return set_affinity


def _worker_environment(
    source: dict[str, str], run: InstanceRun, instance_count: int
) -> tuple[dict[str, str], list[str]]:
    environment = dict(source)
    removed = sorted(name for name in environment if name.startswith("SAGR_"))
    for name in removed:
        environment.pop(name, None)
    environment.pop("_AMDGPU_SIM_BOOTSTRAPPED", None)
    for name in DEVICE_VISIBILITY_VARIABLES:
        environment[name] = ""
    environment.update(
        {
            "TRITON_CACHE_DIR": str(run.cache_dir),
            "GEMSIM_SUPERVISOR_INSTANCE": str(run.index),
            "GEMSIM_SUPERVISOR_INSTANCE_COUNT": str(instance_count),
            "GEMSIM_SUPERVISOR_PHASE": run.phase,
            "GEMSIM_SUPERVISOR_OUTPUT_DIR": str(run.directory),
            "PYTHONNOUSERSITE": "1",
        }
    )
    if run.rank_launch_path is not None:
        environment["GEMSIM_RANK_LAUNCH_DESCRIPTOR"] = str(run.rank_launch_path)
    return environment, removed


def _start_run(
    run: InstanceRun, source_environment: dict[str, str], instance_count: int
) -> list[str]:
    instance_root = run.directory.parent
    instance_root.mkdir(mode=0o700, exist_ok=True)
    os.chmod(instance_root, 0o700)
    run.directory.mkdir(mode=0o700)
    os.chmod(run.directory, 0o700)
    if run.cache_dir.is_relative_to(instance_root):
        cache_parent = run.cache_dir.parent
        cache_parent.mkdir(mode=0o700, exist_ok=True)
        os.chmod(cache_parent, 0o700)
        run.cache_dir.mkdir(mode=0o700, exist_ok=True)
        os.chmod(run.cache_dir, 0o700)
    if run.rank_launch is not None:
        assert run.rank_launch_path is not None
        write_rank_launch(run.rank_launch_path, run.rank_launch)
        if load_rank_launch(run.rank_launch_path) != run.rank_launch:
            raise SupervisorError("rank launch descriptor did not round-trip exactly")
    environment, removed = _worker_environment(source_environment, run, instance_count)
    run.stdout = (run.directory / "stdout.log").open("wb")
    run.stderr = (run.directory / "stderr.log").open("wb")
    run.started_at = time.monotonic()
    run.process = subprocess.Popen(
        run.command,
        stdin=subprocess.DEVNULL,
        stdout=run.stdout,
        stderr=run.stderr,
        close_fds=True,
        cwd=run.directory,
        env=environment,
        start_new_session=True,
        preexec_fn=(
            _affinity_setter(run.cpu_set) if run.cpu_set is not None else None
        ),
    )
    return removed


def _finish_run(run: InstanceRun) -> None:
    if run.process is not None and run.returncode is None:
        run.returncode = run.process.wait()
    run.ended_at = time.monotonic()
    if run.stdout is not None:
        run.stdout.close()
        run.stdout = None
    if run.stderr is not None:
        run.stderr.close()
        run.stderr = None
    for runtime in run.runtimes.values():
        runtime.refresh_ready_identity()


def _run_phase(
    runs: list[InstanceRun],
    source_environment: dict[str, str],
    instance_count: int,
    timeout_seconds: float,
    cleanup_grace_seconds: float,
    group_registry: GroupRegistrySession | None = None,
) -> tuple[list[str], bool]:
    removed: set[str] = set()
    interrupted = False
    try:
        for run in runs:
            removed.update(_start_run(run, source_environment, instance_count))
        if group_registry is not None:
            group_registry.publish("STARTING", runs)
        deadline = (
            time.monotonic() + timeout_seconds if timeout_seconds > 0.0 else None
        )
        active = set(range(len(runs)))
        while active:
            group_abort_reason = None
            for position in list(active):
                run = runs[position]
                _capture_descendants(run)
                assert run.process is not None
                returncode = run.process.poll()
                if returncode is None:
                    continue
                run.returncode = returncode
                _finish_run(run)
                _terminate_run(run, cleanup_grace_seconds)
                active.remove(position)
                if group_registry is not None and returncode != 0:
                    group_abort_reason = (
                        f"rank {run.index} exited with status {returncode}"
                    )
                    break
            if group_registry is not None:
                group_registry.refresh(runs)
                if group_abort_reason is None:
                    group_abort_reason = group_registry.ready_health_error(runs, active)
                if group_abort_reason is not None:
                    group_registry.abort(runs, group_abort_reason)
                    message = f"group aborted: {group_abort_reason}"
                    for run in runs:
                        if message not in run.isolation_errors:
                            run.isolation_errors.append(message)
                    for position in list(active):
                        _terminate_run(runs[position], cleanup_grace_seconds)
                        if runs[position].process is not None:
                            runs[position].returncode = runs[position].process.returncode
                        _finish_run(runs[position])
                        active.remove(position)
                    break
            if deadline is not None and time.monotonic() >= deadline:
                for position in active:
                    runs[position].timed_out = True
                break
            if active:
                time.sleep(0.02)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        for run in runs:
            if run.returncode is None:
                _terminate_run(run, cleanup_grace_seconds)
                if run.process is not None:
                    run.returncode = run.process.returncode
                _finish_run(run)
        _cleanup_unassigned_children(runs, cleanup_grace_seconds)
        if group_registry is not None:
            group_registry.publish("OFF", runs)
    return sorted(removed), interrupted


def _cleanup_unassigned_children(runs: Iterable[InstanceRun], grace: float) -> None:
    known_direct = {
        run.process.pid for run in runs if run.process is not None
    }
    known_descendants = {
        pid for run in runs for pid in run.tracked
    }
    unknown = []
    for pid in _children(os.getpid()):
        if pid in known_direct or pid in known_descendants:
            continue
        identity = _process_identity(pid)
        if identity is not None:
            unknown.append(identity)
    for identity in unknown:
        _signal_identity(identity, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(
        _identity_alive(identity) for identity in unknown
    ):
        time.sleep(0.01)
    for identity in unknown:
        _signal_identity(identity, signal.SIGKILL)
    _reap_adopted_children()


def _canonical_uuid(value: str | None) -> bool:
    return value is not None and re.fullmatch(r"[0-9a-f]{32}", value) is not None


def _validate_run(run: InstanceRun, group_mode: bool) -> None:
    if run.returncode != 0:
        return
    if run.audit_errors:
        run.isolation_errors.append("process isolation audit encountered procfs errors")
    if run.sagr_environment_leaks:
        run.isolation_errors.append("SAGR environment overrides leaked into a worker process")
    if run.production_device_fds:
        run.isolation_errors.append("worker opened a production GPU device node")
    if len(run.runtimes) != 1:
        run.isolation_errors.append(
            f"expected one private gem5 process, observed {len(run.runtimes)}"
        )
        return
    runtime = next(iter(run.runtimes.values()))
    endpoint = Path(runtime.endpoint)
    run_dir = Path(runtime.run_dir)
    if not endpoint.is_absolute() or endpoint.parent != run_dir:
        run.isolation_errors.append("runtime endpoint is not private to its run directory")
    if Path(runtime.output_dir).parent != run_dir:
        run.isolation_errors.append("gem5 output directory is not private to its run directory")
    if Path(runtime.trace_path).parent != run_dir:
        run.isolation_errors.append("dispatch trace is not private to its run directory")
    required_directories = (run_dir, Path(runtime.output_dir), Path(runtime.gem5_cache_dir))
    for path in required_directories:
        if not path.is_dir():
            run.isolation_errors.append(f"required runtime directory is missing: {path}")
    required_files = (
        Path(runtime.trace_path),
        Path(runtime.stats_path),
        Path(runtime.gem5_log_path),
    )
    for path in required_files:
        if not path.is_file():
            run.isolation_errors.append(f"required runtime artifact is missing: {path}")
    if runtime.epoch == 0:
        run.isolation_errors.append("runtime epoch is zero")
    if not _canonical_uuid(runtime.job_uuid):
        run.isolation_errors.append("runtime job UUID is missing or noncanonical")
    if not _canonical_uuid(runtime.daemon_uuid):
        run.isolation_errors.append("runtime daemon UUID is missing or noncanonical")
    configured_identity = (
        runtime.job_uuid,
        runtime.epoch,
        runtime.rank,
        runtime.world_size,
    )
    observed_identity = (
        runtime.observed_job_uuid,
        runtime.observed_epoch,
        runtime.observed_rank,
        runtime.observed_world_size,
    )
    if observed_identity != configured_identity:
        run.isolation_errors.append(
            "gem5 ready identity does not match configured job/epoch/rank/world"
        )
    if group_mode:
        if run.rank_launch is None or run.rank_launch_path is None:
            run.isolation_errors.append("group worker is missing its rank launch descriptor")
            return
        try:
            persisted = load_rank_launch(run.rank_launch_path)
        except RegistryError as error:
            run.isolation_errors.append(f"rank launch descriptor failed validation: {error}")
            return
        if persisted != run.rank_launch:
            run.isolation_errors.append("rank launch descriptor changed after publication")
        descriptor_identity = (
            persisted["job_uuid"],
            persisted["epoch"],
            persisted["rank"],
            persisted["world_size"],
        )
        if configured_identity != descriptor_identity:
            run.isolation_errors.append(
                "runtime identity does not match its immutable rank launch descriptor"
            )
        paths = persisted["paths"]
        expected_paths = (
            runtime.endpoint,
            runtime.run_dir,
            runtime.output_dir,
            runtime.trace_path,
            runtime.gem5_log_path,
            runtime.gem5_cache_dir,
        )
        descriptor_paths = (
            paths["endpoint"],
            paths["runtime_directory"],
            paths["gem5_output_directory"],
            paths["dispatch_trace_path"],
            paths["gem5_log_path"],
            paths["gem5_cache_directory"],
        )
        if expected_paths != descriptor_paths:
            run.isolation_errors.append(
                "runtime paths do not match its immutable rank launch descriptor"
            )
    elif (runtime.rank, runtime.world_size) != (0, 1):
        run.isolation_errors.append(
            "managed correctness worker must be an independent rank-0/world-1 job"
        )


def _validate_phase(
    runs: list[InstanceRun], shared_cache: bool, group_mode: bool = False
) -> None:
    for run in runs:
        _validate_run(run, group_mode)
    successful = [
        (run, next(iter(run.runtimes.values())))
        for run in runs
        if run.returncode == 0 and len(run.runtimes) == 1
    ]
    unique_fields = (
        "endpoint",
        "run_dir",
        "output_dir",
        "trace_path",
        "stats_path",
        "gem5_cache_dir",
        "daemon_uuid",
    )
    if not group_mode:
        unique_fields += ("epoch", "job_uuid")
    for field_name in unique_fields:
        values = [getattr(runtime, field_name) for _run, runtime in successful]
        if len(values) != len(set(values)):
            for run, _runtime in successful:
                run.isolation_errors.append(
                    f"private runtime field is shared across workers: {field_name}"
                )
    if not shared_cache:
        cache_paths = [str(run.cache_dir) for run in runs]
        if len(cache_paths) != len(set(cache_paths)):
            for run in runs:
                run.isolation_errors.append("writable Triton cache is shared across workers")
    if group_mode and successful:
        identity = {
            (runtime.job_uuid, runtime.epoch, runtime.world_size)
            for _run, runtime in successful
        }
        if len(identity) != 1:
            for run, _runtime in successful:
                run.isolation_errors.append("group runtimes have mixed job/epoch/world identity")
        expected_world = len(runs)
        ranks = [runtime.rank for _run, runtime in successful]
        if len(successful) != expected_world or sorted(ranks) != list(range(expected_world)):
            for run in runs:
                run.isolation_errors.append(
                    "group runtimes must expose ranks 0..world_size-1 exactly once"
                )


def _create_output_root(path: Path) -> Path:
    result = path.expanduser().resolve()
    try:
        result.mkdir(parents=True, mode=0o700)
    except FileExistsError as error:
        raise SupervisorError(f"output directory must not already exist: {result}") from error
    os.chmod(result, 0o700)
    return result


def _cache_tree_identity(root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"amdgpu-sim.read-only-cache-tree.v1\0")

    def add_directory(directory: Path, relative: Path) -> None:
        metadata = directory.stat(follow_symlinks=False)
        mode = stat.S_IMODE(metadata.st_mode)
        if mode & 0o222:
            raise SupervisorError(f"shared cache entry is writable: {directory}")
        digest.update(b"D\0")
        digest.update(os.fsencode(str(relative)))
        digest.update(b"\0")
        digest.update(f"{mode:o}:{metadata.st_uid}:{metadata.st_gid}".encode("ascii"))
        digest.update(b"\0")
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: os.fsencode(entry.name))
        for entry in ordered:
            path = directory / entry.name
            child_relative = relative / entry.name
            child_metadata = entry.stat(follow_symlinks=False)
            child_mode = stat.S_IMODE(child_metadata.st_mode)
            if child_mode & 0o222:
                raise SupervisorError(f"shared cache entry is writable: {path}")
            if stat.S_ISDIR(child_metadata.st_mode):
                add_directory(path, child_relative)
                continue
            if not stat.S_ISREG(child_metadata.st_mode):
                raise SupervisorError(
                    f"shared cache may contain only directories and regular files: {path}"
                )
            digest.update(b"F\0")
            digest.update(os.fsencode(str(child_relative)))
            digest.update(b"\0")
            digest.update(
                (
                    f"{child_mode:o}:{child_metadata.st_uid}:"
                    f"{child_metadata.st_gid}:{child_metadata.st_size}"
                ).encode("ascii")
            )
            digest.update(b"\0")
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")

    add_directory(root, Path("."))
    return digest.hexdigest()


def _shared_cache(path: Path | None) -> tuple[Path | None, str | None]:
    if path is None:
        return None, None
    result = path.expanduser().resolve(strict=True)
    if not result.is_dir():
        raise SupervisorError(f"shared cache is not a directory: {result}")
    try:
        identity = _cache_tree_identity(result)
    except SupervisorError as error:
        raise SupervisorError(
            "shared cache must be prewarmed and recursively read-only; concurrent "
            f"cache writes are forbidden: {error}"
        ) from error
    return result, identity


def _make_runs(
    root: Path,
    phase: str,
    count: int,
    command: list[str],
    shared_cache: Path | None,
    cpu_sets: list[set[int]] | None = None,
    group_identity: tuple[str, int] | None = None,
) -> list[InstanceRun]:
    result = []
    for index in range(count):
        instance_root = root / f"instance-{index:03d}"
        cache_dir = shared_cache or instance_root / "cache/triton"
        directory = instance_root / phase
        run = InstanceRun(
            index=index,
            phase=phase,
            directory=directory,
            cache_dir=cache_dir,
            command=list(command),
            cpu_set=cpu_sets[index] if cpu_sets is not None else None,
        )
        if group_identity is not None:
            job_uuid, epoch = group_identity
            run.rank_launch = make_rank_launch(
                job_uuid=job_uuid,
                epoch=epoch,
                rank=index,
                world_size=count,
                instance_directory=directory,
                triton_cache_directory=cache_dir,
            )
            run.rank_launch_path = directory / "rank-launch.json"
        result.append(run)
    if group_identity is not None:
        validate_rank_launch_group(
            run.rank_launch for run in result if run.rank_launch is not None
        )
    return result


def _write_manifest(root: Path, value: dict[str, Any]) -> None:
    temporary = root / ".manifest.json.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, root / "manifest.json")
    directory_descriptor = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _calibration_summary(
    baselines: list[InstanceRun], parallel: list[InstanceRun]
) -> dict[str, Any]:
    serial_total = sum(run.elapsed_seconds for run in baselines)
    parallel_wall = max((run.ended_at for run in parallel), default=0.0) - min(
        (run.started_at for run in parallel), default=0.0
    )
    per_instance = []
    for baseline, concurrent in zip(baselines, parallel, strict=True):
        per_instance.append(
            {
                "index": baseline.index,
                "serial_seconds": baseline.elapsed_seconds,
                "parallel_seconds": concurrent.elapsed_seconds,
                "slowdown": (
                    concurrent.elapsed_seconds / baseline.elapsed_seconds
                    if baseline.elapsed_seconds > 0.0
                    else None
                ),
            }
        )
    return {
        "claim_scope": "CPU-affinity interference calibration only; not TPOT or TP evidence",
        "serial_total_seconds": serial_total,
        "parallel_wall_seconds": parallel_wall,
        "aggregate_speedup": serial_total / parallel_wall if parallel_wall > 0.0 else None,
        "per_instance": per_instance,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one isolated process/runtime/gem5 session per correctness worker. "
            "The repeated command must open exactly one managed gemsim session."
        )
    )
    parser.add_argument("-n", "--instances", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("correctness", "calibrate"), default="correctness"
    )
    parser.add_argument(
        "--topology",
        choices=("independent", "group"),
        default="independent",
        help=(
            "independent keeps rank-0/world-1 jobs; group creates one immutable "
            "job/epoch/world identity with ranks 0..N-1"
        ),
    )
    parser.add_argument(
        "--live-registry",
        type=Path,
        help=(
            "group-mode live registry path; defaults to "
            "OUTPUT_DIR/live-registry.json"
        ),
    )
    parser.add_argument(
        "--shared-readonly-cache",
        type=Path,
        help=(
            "explicit exact-workload-prewarmed, recursively read-only Triton "
            "cache shared by all workers; its tree is verified before and after"
        ),
    )
    parser.add_argument(
        "--cpu-set",
        action="append",
        type=_parse_cpu_set,
        default=[],
        help="one disjoint Linux CPU list per instance; calibration mode only",
    )
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--cleanup-grace-seconds", type=float, default=2.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SupervisorError("a worker command is required after --")
    supported = "2..16" if args.topology == "group" else "2 or 4"
    valid_instances = (
        2 <= args.instances <= 16
        if args.topology == "group"
        else args.instances in (2, 4)
    )
    if not valid_instances:
        raise SupervisorError(
            f"{args.topology} infrastructure supports instance counts {supported}"
        )
    if args.timeout_seconds < 0.0 or args.cleanup_grace_seconds <= 0.0:
        raise SupervisorError("timeouts must be nonnegative and cleanup grace must be positive")
    if args.mode == "correctness" and args.cpu_set:
        raise SupervisorError("CPU affinity is reserved for calibration mode")
    if args.topology == "group" and args.mode != "correctness":
        raise SupervisorError("group topology is available only in correctness mode")
    if args.topology == "independent" and args.live_registry is not None:
        raise SupervisorError("--live-registry is valid only with --topology group")
    if args.topology == "group" and args.shared_readonly_cache is not None:
        raise SupervisorError(
            "group topology requires one private writable Triton cache per rank"
        )
    if args.mode == "calibrate":
        if len(args.cpu_set) != args.instances:
            raise SupervisorError("calibration requires exactly one --cpu-set per instance")
        allowed = os.sched_getaffinity(0)
        used: set[int] = set()
        for cpu_set in args.cpu_set:
            if not cpu_set <= allowed:
                raise SupervisorError(
                    f"CPU set {sorted(cpu_set)} is outside allowed affinity {sorted(allowed)}"
                )
            if used & cpu_set:
                raise SupervisorError("calibration CPU sets must not overlap")
            used.update(cpu_set)

    root = _create_output_root(args.output_dir)
    shared_cache, shared_cache_before = _shared_cache(args.shared_readonly_cache)
    _enable_subreaper()
    source_environment = dict(os.environ)
    all_runs: list[InstanceRun] = []
    removed_sagr: set[str] = set()
    interrupted = False
    group_mode = args.topology == "group"
    group_job_uuid = uuid.uuid4().hex if group_mode else None
    group_epoch = (
        ((time.monotonic_ns() ^ (os.getpid() << 32)) & ((1 << 63) - 1)) or 1
        if group_mode
        else None
    )
    publisher: LiveRegistryPublisher | None = None
    group_registry: GroupRegistrySession | None = None

    if args.mode == "correctness":
        assert (group_job_uuid is None) == (group_epoch is None)
        group_identity = (
            (group_job_uuid, group_epoch)
            if group_job_uuid is not None and group_epoch is not None
            else None
        )
        runs = _make_runs(
            root,
            "correctness",
            args.instances,
            command,
            shared_cache,
            group_identity=group_identity,
        )
        if group_mode:
            registry_path = (
                Path(os.path.abspath(args.live_registry))
                if args.live_registry is not None
                else root / "live-registry.json"
            )
            if registry_path.parent != root:
                raise SupervisorError(
                    "live registry must be a direct child of the private output directory"
                )
            ensure_private_directory(registry_path.parent)
            publisher = LiveRegistryPublisher(registry_path)
            group_registry = GroupRegistrySession(
                publisher=publisher,
                job_uuid=group_job_uuid,
                epoch=group_epoch,
                world_size=args.instances,
            )
        try:
            removed, interrupted = _run_phase(
                runs,
                source_environment,
                args.instances,
                args.timeout_seconds,
                args.cleanup_grace_seconds,
                group_registry,
            )
        finally:
            if publisher is not None:
                publisher.close()
        removed_sagr.update(removed)
        _validate_phase(runs, shared_cache is not None, group_mode)
        all_runs.extend(runs)
        calibration = None
    else:
        phases: dict[str, list[InstanceRun]] = {}
        for phase in ("warmup", "serial-baseline"):
            phase_runs = _make_runs(
                root, phase, args.instances, command, shared_cache, args.cpu_set
            )
            phases[phase] = phase_runs
            for run in phase_runs:
                removed, was_interrupted = _run_phase(
                    [run],
                    source_environment,
                    args.instances,
                    args.timeout_seconds,
                    args.cleanup_grace_seconds,
                )
                removed_sagr.update(removed)
                interrupted = interrupted or was_interrupted
                _validate_phase([run], shared_cache is not None)
                if run.returncode != 0 or run.isolation_errors or interrupted:
                    break
            all_runs.extend(phase_runs)
            if any(run.returncode != 0 or run.isolation_errors for run in phase_runs):
                break
        parallel: list[InstanceRun] = []
        if not interrupted and all(
            run.returncode == 0 and not run.isolation_errors for run in all_runs
        ):
            parallel = _make_runs(
                root, "parallel", args.instances, command, shared_cache, args.cpu_set
            )
            removed, interrupted = _run_phase(
                parallel,
                source_environment,
                args.instances,
                args.timeout_seconds,
                args.cleanup_grace_seconds,
            )
            removed_sagr.update(removed)
            _validate_phase(parallel, shared_cache is not None)
            all_runs.extend(parallel)
        calibration = (
            _calibration_summary(phases["serial-baseline"], parallel)
            if parallel and "serial-baseline" in phases
            else None
        )

    shared_cache_after = None
    shared_cache_error = None
    if shared_cache is not None:
        try:
            shared_cache_after = _cache_tree_identity(shared_cache)
        except (OSError, SupervisorError) as error:
            shared_cache_error = str(error)
    shared_cache_unchanged = (
        shared_cache is None
        or (
            shared_cache_error is None
            and shared_cache_before == shared_cache_after
        )
    )

    manifest = {
        "schema": "amdgpu-sim.gemsim-instance-supervisor.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "topology": args.topology,
        "instance_count": args.instances,
        "command": command,
        "output_dir": str(root),
        "cache_policy": (
            {
                "mode": "shared-read-only-prewarmed",
                "path": str(shared_cache),
                "concurrent_writes_allowed": False,
                "exact_workload_must_be_prewarmed": True,
                "pre_run_tree_sha256": shared_cache_before,
                "post_run_tree_sha256": shared_cache_after,
                "tree_unchanged": shared_cache_unchanged,
                "verification_error": shared_cache_error,
            }
            if shared_cache is not None
            else {
                "mode": "per-instance-writable",
                "paths_unique": True,
                "concurrent_writes_allowed": True,
            }
        ),
        "environment_boundary": {
            "removed_sagr_variables": sorted(removed_sagr),
            "all_sagr_overrides_removed": True,
            "worker_entry_device_visibility": {
                name: "hidden" for name in DEVICE_VISIBILITY_VARIABLES
            },
            "runtime_enforcement": (
                "successful runs fail on sampled SAGR environment leaks, procfs "
                "audit errors, or open /dev/kfd, /dev/dri, or /dev/nvidia nodes"
            ),
        },
        "namespace_contract": {
            "runner_process": "one process per instance",
            "runtime_session": "one managed session per successful runner",
            "daemon": "one private gem5 process per managed session",
            "transport": "unique socket, daemon UUID, job UUID, and epoch",
            "objects": "allocation, queue, and signal IDs are daemon/connection scoped",
            "topology": (
                "one group job uses one job UUID/epoch/world and ranks 0..N-1 exactly once"
                if group_mode
                else "independent correctness jobs use rank 0 and world size 1"
            ),
            "rank_launch_schema": (
                "amdgpu-sim.gemsim-rank-launch.v1" if group_mode else None
            ),
            "live_registry_schema": (
                "amdgpu-sim.gemsim-live-registry.v1" if group_mode else None
            ),
            "live_registry_path": (
                str(publisher.registry_path) if publisher is not None else None
            ),
            "ccl_boundary": "this layer assigns topology; CCL collectives remain a separate API/test layer",
            "group_abort_reason": (
                group_registry.abort_reason if group_registry is not None else None
            ),
        },
        "interrupted": interrupted,
        "runs": [run.as_dict() for run in all_runs],
        "calibration": calibration,
    }
    passed = (
        not interrupted
        and len(all_runs) > 0
        and shared_cache_unchanged
        and all(run.returncode == 0 and not run.isolation_errors for run in all_runs)
    )
    manifest["passed"] = passed
    _write_manifest(root, manifest)
    print(json.dumps({"manifest": str(root / "manifest.json"), "passed": passed}))
    if interrupted:
        return 130
    if any(run.timed_out for run in all_runs):
        return 124
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SupervisorError, RegistryError) as error:
        print(f"gemsim instance supervisor: {error}", file=sys.stderr)
        raise SystemExit(2)
