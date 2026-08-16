#!/usr/bin/env python3
"""Manage the feature-local AgentENV sandbox pair.

This is a small, dependency-free control plane for the integration branch. It
keeps all local state under ``build/agentenv-integration`` and talks to an
AgentENV server over loopback HTTP.  It never mounts the worktree into a VM;
the runtime bundle is uploaded/extracted by the template bootstrap workflow.

The dangerous operations are explicit:

* ``start-pair`` refuses when unrelated gem5/vLLM/SGLang processes are live.
  Pass ``--allow-live`` only after inspecting the reported process list.
* ``stop`` requires ``--confirm`` before sending DELETE requests.
* all commands support ``--dry-run`` and do not require AgentENV for planning.

The manager intentionally does not repair model/framework failures.  It only
provides stable VM, process, filesystem, network, tmp, socket, cache, and log
namespaces so two existing workload commands can be launched independently.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import sys
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MANAGER_SCHEMA = "amdgpu-sim.agentenv-manager.v1"
FEATURE_ID = "amdgpu-sim-agentenv"
DEFAULT_API_URL = "http://127.0.0.1:18080"
DEFAULT_STATE_RELATIVE = Path("build/agentenv-integration/state")
DEFAULT_INSTANCES = ("vllm-tp4", "sglang-tp1")
DEFAULT_CPU_COUNT = 12
DEFAULT_MEMORY_MB = 24 * 1024
DEFAULT_DISK_SIZE_MB = 96 * 1024
WORKLOAD_WORDS = ("gem5", "vllm", "sglang")
WORKLOAD_RE = re.compile(r"(?:^|[/_. -])(gem5(?:\.opt|\.fast)?|vllm|sglang)(?:[/_. -]|$)", re.I)


class ManagerError(RuntimeError):
    """Raised for an unsafe or incomplete lifecycle operation."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def atomic_write(path: Path, payload: bytes) -> None:
    path = absolute(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def state_root(repo: Path, configured: str | None) -> Path:
    candidate = Path(configured) if configured else repo / DEFAULT_STATE_RELATIVE
    return absolute(candidate)


def namespace_for(repo: Path, instance: str, feature_id: str = FEATURE_ID) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", instance.lower()).strip("-") or "instance"
    digest = hashlib.sha256(f"{feature_id}\0{absolute(repo)}\0{instance}".encode()).hexdigest()[:12]
    return f"aenv-{slug}-{digest}"


def _read_proc_file(path: Path, *, binary: bool = False) -> str:
    try:
        payload = path.read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return ""
    if binary:
        return payload.decode("utf-8", errors="replace")
    return payload.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class LiveProcess:
    pid: int
    command: str
    cwd: str
    feature_owned: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "command": self.command,
            "cwd": self.cwd,
            "feature_owned": self.feature_owned,
        }


def _is_feature_owned(command: str, cwd: str, environ: str, repo: Path) -> bool:
    marker = f"AGENTENV_FEATURE_ID={FEATURE_ID}"
    return (
        str(repo) in cwd
        or str(repo) in command
        or marker in environ
        or "agentenv-sandbox" in command
        or "agentenv-manager.py" in command
    )


def scan_live_workloads(repo: Path) -> list[LiveProcess]:
    """Return gem5/vLLM/SGLang processes unrelated to this feature worktree."""

    records: list[LiveProcess] = []
    proc_root = Path("/proc")
    try:
        process_dirs = sorted(
            (item for item in proc_root.iterdir() if item.name.isdigit()),
            key=lambda item: int(item.name),
        )
    except OSError:
        return records
    for directory in process_dirs:
        try:
            pid = int(directory.name)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        raw_cmdline = _read_proc_file(directory / "cmdline", binary=True)
        command = " ".join(part for part in raw_cmdline.split("\0") if part).strip()
        if not command or not WORKLOAD_RE.search(command):
            continue
        try:
            cwd = os.readlink(directory / "cwd")
        except (FileNotFoundError, PermissionError, OSError):
            cwd = "<unavailable>"
        environ = _read_proc_file(directory / "environ", binary=True)
        records.append(
            LiveProcess(
                pid=pid,
                command=command,
                cwd=cwd,
                feature_owned=_is_feature_owned(command, cwd, environ, repo),
            )
        )
    return records


def _api_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def api_request(
    base: str,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 15.0,
) -> tuple[int, Any, dict[str, str]]:
    payload = None if body is None else canonical_json(body)
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    api_key = os.environ.get("AENV_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    request = Request(_api_url(base, path), data=payload, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
            if raw and "json" in content_type.lower():
                try:
                    value: Any = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    value = raw.decode("utf-8", errors="replace")
            elif raw:
                value = raw.decode("utf-8", errors="replace")
            else:
                value = None
            return response.status, value, dict(response.headers.items())
    except HTTPError as error:
        raw = error.read()
        detail = raw.decode("utf-8", errors="replace")
        try:
            value = json.loads(detail)
        except json.JSONDecodeError:
            value = detail
        raise ManagerError(
            f"AgentENV {method} {path} returned HTTP {error.code}: {value}"
        ) from error
    except URLError as error:
        raise ManagerError(f"AgentENV is unreachable at {base}: {error.reason}") from error
    except TimeoutError as error:
        raise ManagerError(f"AgentENV request timed out at {base}") from error


def host_preflight(api: str, *, check_api: bool) -> dict[str, Any]:
    release = platform.release()
    kvm = Path("/dev/kvm")
    ublk = Path("/dev/ublk-control")
    checks: dict[str, Any] = {
        "kernel_release": release,
        "kernel_meets_agentenv_minimum": _kernel_at_least(release, 6, 8),
        "kvm": _device_status(kvm),
        "ublk_control": _device_status(ublk),
        "ip_forward": _read_text(Path("/proc/sys/net/ipv4/ip_forward")),
        "zstd": shutil.which("zstd") or None,
        "firecracker": shutil.which("firecracker") or None,
        "aenv": shutil.which("aenv") or None,
        "python": sys.executable,
    }
    checks["process_guard"] = {
        "workload_keywords": list(WORKLOAD_WORDS),
        "note": "run start-pair to scan and refuse unrelated workloads",
    }
    if check_api:
        try:
            status, value, _ = api_request(api, "GET", "/health")
            checks["api"] = {"ok": 200 <= status < 300, "status": status, "body": value}
        except ManagerError as error:
            checks["api"] = {"ok": False, "error": str(error)}
    return {"schema": MANAGER_SCHEMA, "checked_at": now_utc(), "checks": checks}


def _kernel_at_least(release: str, major: int, minor: int) -> bool:
    match = re.match(r"(\d+)\.(\d+)", release)
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) >= (major, minor)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None


def _device_status(path: Path) -> dict[str, Any]:
    try:
        metadata = path.stat()
    except OSError as error:
        return {"path": str(path), "present": False, "error": str(error)}
    return {
        "path": str(path),
        "present": True,
        "mode": stat.filemode(metadata.st_mode),
        "readable": os.access(path, os.R_OK),
        "writable": os.access(path, os.W_OK),
    }


def _instance_payload(
    repo: Path,
    instance: str,
    *,
    template: str,
    timeout: int,
    feature_id: str,
    bundle_manifest_sha256: str | None,
    allow_internet: bool,
    cpu_count: int,
    memory_mb: int,
    disk_size_mb: int,
) -> tuple[str, dict[str, Any]]:
    namespace = namespace_for(repo, instance, feature_id)
    metadata: dict[str, str] = {
        "agentenv_feature": feature_id,
        "agentenv_instance": instance,
        "agentenv_namespace": namespace,
        "agentenv_worktree": str(repo),
        "agentenv_cpu_count": str(cpu_count),
        "agentenv_memory_mb": str(memory_mb),
        "agentenv_disk_size_mb": str(disk_size_mb),
    }
    if bundle_manifest_sha256:
        metadata["agentenv_bundle_manifest_sha256"] = bundle_manifest_sha256
    env_vars = {
        "AGENTENV_FEATURE_ID": feature_id,
        "AGENTENV_INSTANCE_ID": instance,
        "AGENTENV_NAMESPACE": namespace,
        "TMPDIR": f"/tmp/{namespace}",
        "XDG_RUNTIME_DIR": f"/run/{namespace}",
        "XDG_CACHE_HOME": f"/var/cache/{namespace}",
        "HF_HOME": f"/var/cache/{namespace}/huggingface",
        "TRITON_CACHE_DIR": f"/var/cache/{namespace}/triton",
    }
    payload = {
        "templateID": template,
        "timeout": timeout,
        "autoPause": False,
        "secure": True,
        "allow_internet_access": allow_internet,
        "metadata": metadata,
        "envVars": env_vars,
    }
    return namespace, payload


def _load_bundle_sha256(path: str | None) -> str | None:
    if not path:
        return None
    candidate = absolute(Path(path))
    if not candidate.exists():
        raise ManagerError(f"bundle manifest does not exist: {candidate}")
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManagerError(f"invalid bundle manifest {candidate}: {error}") from error
    digest = document.get("archive_sha256") or document.get("manifest_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ManagerError(f"bundle manifest has no valid archive_sha256: {candidate}")
    return digest


def _instance_state_path(state: Path, instance: str) -> Path:
    return state / "instances" / instance / "sandbox.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_instance_state(state: Path, instance: str, value: dict[str, Any]) -> None:
    directory = state / "instances" / instance
    for child in ("logs", "tmp", "sockets", "cache", "endpoint"):
        (directory / child).mkdir(parents=True, exist_ok=True)
    atomic_write(_instance_state_path(state, instance), canonical_json(value))


def _sandbox_id(response: Any, headers: dict[str, str]) -> str | None:
    if isinstance(response, dict):
        for key in ("sandboxID", "sandboxId", "id"):
            value = response.get(key)
            if isinstance(value, str) and value:
                return value
    for key, value in headers.items():
        if key.lower() == "x-agentenv-sandbox-id" and value:
            return value
    return None


def _resource_check(local: dict[str, Any], remote: Any) -> dict[str, Any]:
    metadata = local.get("payload", {}).get("metadata", {})
    expected = {
        "cpuCount": metadata.get("agentenv_cpu_count"),
        "memoryMB": metadata.get("agentenv_memory_mb"),
        "diskSizeMB": metadata.get("agentenv_disk_size_mb"),
    }
    result: dict[str, Any] = {"expected": expected, "actual": {}, "matches": True}
    if not isinstance(remote, dict):
        result["matches"] = False
        result["error"] = "remote sandbox record is not an object"
        return result
    for field, wanted in expected.items():
        actual = remote.get(field)
        result["actual"][field] = actual
        if wanted is not None and str(actual) != str(wanted):
            result["matches"] = False
    return result


def start_pair(args: argparse.Namespace) -> dict[str, Any]:
    repo = absolute(Path(args.repo))
    state = state_root(repo, args.state_dir)
    if args.cpu_count <= 0 or args.memory_mb <= 0 or args.disk_size_mb <= 0:
        raise ManagerError("resource contract values must be positive")
    if args.disk_size_mb % 1024:
        raise ManagerError("disk-size-mb must be divisible by 1024")
    live = scan_live_workloads(repo)
    unrelated = [record for record in live if not record.feature_owned]
    if unrelated and not args.allow_live:
        raise ManagerError(
            "refusing start-pair because unrelated gem5/vLLM/SGLang processes are live: "
            + json.dumps([record.as_dict() for record in unrelated], sort_keys=True)
            + "; inspect them or pass --allow-live explicitly"
        )
    template = args.template or "<template-id-required>"
    digest = _load_bundle_sha256(args.bundle_manifest)
    instances = tuple(args.instance or DEFAULT_INSTANCES)
    if len(set(instances)) != len(instances):
        raise ManagerError("instance names must be unique")
    existing: list[dict[str, Any]] = []
    for instance in instances:
        record = _read_json(_instance_state_path(state, instance))
        if record and record.get("sandbox_id") and not record.get("deleted_at"):
            existing.append(record)
    if existing and not getattr(args, "allow_existing", False):
        raise ManagerError(
            "refusing to create duplicate feature sandboxes; stop existing records first "
            "or pass --allow-existing"
        )
    plans = []
    for instance in instances:
        namespace, payload = _instance_payload(
            repo,
            instance,
            template=template,
            timeout=args.timeout,
            feature_id=args.feature_id,
            bundle_manifest_sha256=digest,
            allow_internet=args.allow_internet,
            cpu_count=args.cpu_count,
            memory_mb=args.memory_mb,
            disk_size_mb=args.disk_size_mb,
        )
        plans.append({"instance": instance, "namespace": namespace, "payload": payload})
    result: dict[str, Any] = {
        "schema": MANAGER_SCHEMA,
        "operation": "start-pair",
        "planned_at": now_utc(),
        "repo": str(repo),
        "state_dir": str(state),
        "api": args.api,
        "resource_contract": {
            "cpu_count": args.cpu_count,
            "memory_mb": args.memory_mb,
            "disk_size_mb": args.disk_size_mb,
        },
        "live_workloads": [record.as_dict() for record in live],
        "unrelated_workloads": [record.as_dict() for record in unrelated],
        "existing_feature_sandboxes": [
            {"instance": item.get("instance"), "sandbox_id": item.get("sandbox_id")}
            for item in existing
        ],
        "plans": plans,
        "sandboxes": [],
    }
    if args.dry_run:
        return result
    if not args.template:
        raise ManagerError("--template is required unless --dry-run is used")
    created: list[dict[str, Any]] = []
    try:
        for plan in plans:
            status, response, headers = api_request(
                args.api,
                "POST",
                "/sandboxes",
                body=plan["payload"],
                timeout=args.http_timeout,
            )
            sandbox_id = _sandbox_id(response, headers)
            if not sandbox_id:
                raise ManagerError(
                    "AgentENV created a sandbox without an id for "
                    f"{plan['instance']}: {response}"
                )
            record = {
                "schema": MANAGER_SCHEMA,
                "created_at": now_utc(),
                "instance": plan["instance"],
                "namespace": plan["namespace"],
                "sandbox_id": sandbox_id,
                "template": args.template,
                "payload": plan["payload"],
                "api_status": status,
                "response": response,
                "launch_commands": _launch_commands(plan["instance"], plan["namespace"]),
            }
            # Persist immediately so a later failure can be recovered even if
            # the process is interrupted between API calls.
            _write_instance_state(state, plan["instance"], record)
            created.append(record)
            result["sandboxes"].append(record)
    except Exception as error:
        rollback: list[dict[str, Any]] = []
        for record in reversed(created):
            sandbox_id = record.get("sandbox_id")
            if not isinstance(sandbox_id, str) or not sandbox_id:
                continue
            try:
                code, _, _ = api_request(
                    args.api,
                    "DELETE",
                    f"/sandboxes/{sandbox_id}",
                    timeout=args.http_timeout,
                )
                rollback.append(
                    {"instance": record.get("instance"), "sandbox_id": sandbox_id, "status": code}
                )
                updated = dict(record)
                updated["deleted_at"] = now_utc()
                updated["delete_status"] = code
                instance = record.get("instance")
                if isinstance(instance, str) and instance:
                    _write_instance_state(state, instance, updated)
            except ManagerError as rollback_error:
                rollback.append(
                    {
                        "instance": record.get("instance"),
                        "sandbox_id": sandbox_id,
                        "error": str(rollback_error),
                    }
                )
        detail = f"; rollback={json.dumps(rollback, sort_keys=True)}" if rollback else ""
        raise ManagerError(f"start-pair failed: {error}{detail}") from error
    result["completed_at"] = now_utc()
    return result


def _launch_commands(instance: str, namespace: str) -> list[str]:
    if instance == "vllm-tp4":
        command = "scripts/run_qwen35_vllm_tp.py --tensor-parallel-size 4"
    elif instance == "sglang-tp1":
        command = "examples/sglang/qwen35_inference.py --tp-size 1"
    else:
        command = "<workload command not configured>"
    return [
        f"aenv exec <sandbox-id> /bin/bash -lc 'cd /home/zhaosiying/amdgpu-sim && exec {command}'",
        f"# namespace: {namespace}",
    ]


def status(args: argparse.Namespace) -> dict[str, Any]:
    repo = absolute(Path(args.repo))
    state = state_root(repo, args.state_dir)
    local: list[dict[str, Any]] = []
    for instance in args.instance or DEFAULT_INSTANCES:
        record = _read_json(_instance_state_path(state, instance))
        if record is not None:
            local.append(record)
    result: dict[str, Any] = {
        "schema": MANAGER_SCHEMA,
        "operation": "status",
        "checked_at": now_utc(),
        "repo": str(repo),
        "state_dir": str(state),
        "local": local,
        "live_workloads": [record.as_dict() for record in scan_live_workloads(repo)],
    }
    if args.offline or args.dry_run:
        return result
    try:
        status_code, response, _ = api_request(
            args.api, "GET", "/v2/sandboxes", timeout=args.http_timeout
        )
        result["api"] = {"status": status_code, "sandboxes": response}
        if isinstance(response, list):
            remote_by_id = {
                item.get("sandboxID"): item
                for item in response
                if isinstance(item, dict) and isinstance(item.get("sandboxID"), str)
            }
            result["resource_checks"] = [
                {
                    "instance": item.get("instance"),
                    "sandbox_id": item.get("sandbox_id"),
                    **_resource_check(item, remote_by_id.get(item.get("sandbox_id"))),
                }
                for item in local
            ]
    except ManagerError as error:
        result["api"] = {"error": str(error)}
    return result


def collect(args: argparse.Namespace) -> dict[str, Any]:
    repo = absolute(Path(args.repo))
    state = state_root(repo, args.state_dir)
    report: dict[str, Any] = {
        "schema": MANAGER_SCHEMA,
        "operation": "collect",
        "collected_at": now_utc(),
        "repo": str(repo),
        "state_dir": str(state),
        "host": host_preflight(args.api, check_api=False),
        "live_workloads": [record.as_dict() for record in scan_live_workloads(repo)],
        "instances": [],
    }
    for instance in args.instance or DEFAULT_INSTANCES:
        record = _read_json(_instance_state_path(state, instance))
        item: dict[str, Any] = {"instance": instance, "local": record}
        if record and record.get("sandbox_id") and not args.offline and not args.dry_run:
            try:
                code, response, _ = api_request(
                    args.api,
                    "GET",
                    f"/sandboxes/{record['sandbox_id']}",
                    timeout=args.http_timeout,
                )
                item["remote"] = {"status": code, "sandbox": response}
                item["resource_check"] = _resource_check(record, response)
            except ManagerError as error:
                item["remote"] = {"error": str(error)}
        report["instances"].append(item)
    if args.output and not args.dry_run:
        output = Path(args.output)
        if not output.is_absolute():
            output = state / output
        atomic_write(output, canonical_json(report))
        report["output"] = str(absolute(output))
    return report


def stop(args: argparse.Namespace) -> dict[str, Any]:
    repo = absolute(Path(args.repo))
    state = state_root(repo, args.state_dir)
    records = []
    for instance in args.instance or DEFAULT_INSTANCES:
        record = _read_json(_instance_state_path(state, instance))
        if record is not None:
            records.append(record)
    result: dict[str, Any] = {
        "schema": MANAGER_SCHEMA,
        "operation": "stop",
        "requested_at": now_utc(),
        "repo": str(repo),
        "state_dir": str(state),
        "targets": [
            {"instance": item.get("instance"), "sandbox_id": item.get("sandbox_id")}
            for item in records
        ],
        "deleted": [],
    }
    if args.dry_run or not args.confirm:
        result["note"] = "no DELETE sent; pass --confirm to stop recorded sandboxes"
        return result
    for item in records:
        sandbox_id = item.get("sandbox_id")
        if not isinstance(sandbox_id, str) or not sandbox_id:
            continue
        status_code, _, _ = api_request(
            args.api,
            "DELETE",
            f"/sandboxes/{sandbox_id}",
            timeout=args.http_timeout,
        )
        result["deleted"].append(
            {"instance": item.get("instance"), "sandbox_id": sandbox_id, "status": status_code}
        )
        updated = dict(item)
        updated["deleted_at"] = now_utc()
        updated["delete_status"] = status_code
        instance = item.get("instance")
        if isinstance(instance, str) and instance:
            _write_instance_state(state, instance, updated)
    return result


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    top.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    top.add_argument("--state-dir")
    top.add_argument("--api", default=os.environ.get("AENV_API_URL", DEFAULT_API_URL))
    top.add_argument("--feature-id", default=FEATURE_ID)
    top.add_argument("--instance", action="append")
    top.add_argument("--http-timeout", type=float, default=20.0)
    top.add_argument("--dry-run", action="store_true")
    top.add_argument("--json", action="store_true")
    sub = top.add_subparsers(dest="command", required=True)

    def post_options(command_parser: argparse.ArgumentParser) -> None:
        # Accept common switches both before and after the subcommand.  The
        # suppressed defaults ensure an omitted post-command option does not
        # overwrite a value supplied before the subcommand.
        command_parser.add_argument("--repo", default=argparse.SUPPRESS)
        command_parser.add_argument("--state-dir", default=argparse.SUPPRESS)
        command_parser.add_argument("--api", default=argparse.SUPPRESS)
        command_parser.add_argument("--feature-id", default=argparse.SUPPRESS)
        command_parser.add_argument("--instance", action="append", default=argparse.SUPPRESS)
        command_parser.add_argument("--http-timeout", type=float, default=argparse.SUPPRESS)
        command_parser.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
        command_parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS)

    preflight = sub.add_parser("host-preflight", help="inspect host prerequisites")
    post_options(preflight)
    preflight.add_argument("--check-api", action="store_true")
    preflight.add_argument("--strict", action="store_true")

    status_cmd = sub.add_parser("status", help="show local and AgentENV state")
    post_options(status_cmd)
    status_cmd.add_argument("--offline", action="store_true")

    collect_cmd = sub.add_parser("collect", help="write a lifecycle evidence report")
    post_options(collect_cmd)
    collect_cmd.add_argument("--offline", action="store_true")
    collect_cmd.add_argument("--output", default="collect-report.json")

    start = sub.add_parser("start-pair", help="plan or start the two isolated sandboxes")
    post_options(start)
    start.add_argument("--template")
    start.add_argument("--bundle-manifest")
    start.add_argument("--timeout", type=int, default=3600)
    start.add_argument("--allow-internet", action="store_true")
    start.add_argument("--allow-live", action="store_true")
    start.add_argument("--allow-existing", action="store_true")
    start.add_argument("--cpu-count", type=int, default=DEFAULT_CPU_COUNT)
    start.add_argument("--memory-mb", type=int, default=DEFAULT_MEMORY_MB)
    start.add_argument("--disk-size-mb", type=int, default=DEFAULT_DISK_SIZE_MB)

    stop_cmd = sub.add_parser("stop", help="delete recorded sandboxes")
    post_options(stop_cmd)
    stop_cmd.add_argument("--confirm", action="store_true")

    return top


def _emit(value: Any, *, machine: bool) -> None:
    if machine:
        sys.stdout.buffer.write(canonical_json(value))
    else:
        print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "host-preflight":
            value = host_preflight(args.api, check_api=args.check_api)
            if args.strict:
                checks = value["checks"]
                required = (
                    checks["kernel_meets_agentenv_minimum"],
                    checks["kvm"].get("present"),
                    checks["kvm"].get("readable"),
                    checks["kvm"].get("writable"),
                    checks["ublk_control"].get("present"),
                    checks["ublk_control"].get("readable"),
                    checks["ublk_control"].get("writable"),
                    checks["zstd"] is not None,
                )
                if not all(required):
                    raise ManagerError("host-preflight strict checks failed")
        elif args.command == "status":
            value = status(args)
        elif args.command == "collect":
            value = collect(args)
        elif args.command == "start-pair":
            value = start_pair(args)
        elif args.command == "stop":
            value = stop(args)
        else:  # pragma: no cover - argparse enforces the subcommand
            raise ManagerError(f"unknown command: {args.command}")
        _emit(value, machine=args.json)
        return 0
    except ManagerError as error:
        print(f"agentenv manager: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
