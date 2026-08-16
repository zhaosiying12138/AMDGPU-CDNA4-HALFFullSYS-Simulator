#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Supervise one frozen-product live vLLM GroupCoordinator allreduce gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_gemsim_ccl_live_allreduce as base  # noqa: E402
from gemsim_ccl.native import (  # noqa: E402
    CANCELLED,
    CCLStatusError,
    NativeCCL,
    PEER_LOST,
    PROTOCOL_ERROR,
    TIMED_OUT,
)


RUN_SCHEMA = "amdgpu-sim.vllm-ccl-live-run.v1"
EXPECTED_SCHEMA = base.EXPECTED_SCHEMA
WORKER_CONFIG_SCHEMA = "amdgpu-sim.vllm-ccl-live-rank-config.v1"
BOOTSTRAP_SCHEMA = "amdgpu-sim.vllm-ccl-bootstrap.v1"
RANK_RESULT_SCHEMA = "amdgpu-sim.vllm-ccl-live-rank-result.v1"
RANK_FILES = (
    "worker-result.json",
    "adapter-evidence.json",
    "adapter-events.jsonl",
    "bootstrap-descriptor.json",
    "dispatch-trace.jsonl",
    "stats.txt",
    "gem5.log",
    "rank-launch.json",
    "input.bin",
    "output.bin",
    "worker-stdout.log",
    "worker-stderr.log",
)
IDENTITY_ROLES = (
    "product_manifest",
    "source_lock",
    "runtime_library",
    "ccl_native",
    "ccl_device",
    "ccl_engine",
    "triton_driver",
    "gem5_binary",
    "gem5_config",
    "vllm_plugin_init",
    "vllm_communicator",
    "vllm_ccl_bootstrap",
    "vllm_platform",
    "vllm_parallel_state",
    "vllm_base_communicator",
    "vllm_communication_op",
    "vllm_version",
    "vllm_metadata",
    "vllm_checkout_parallel_state",
    "vllm_checkout_base_communicator",
    "vllm_checkout_communication_op",
    "vllm_checkout_version",
    "vllm_linear",
    "vllm_config_vllm",
    "vllm_config_parallel",
    "vllm_config_model",
    "vllm_adapters",
    "vllm_row_parallel",
    "vllm_ops",
    "vllm_kernels",
    "ccl_acceptance_base",
    "verifier",
    "runner",
    "worker",
    "bootstrap",
    "design",
    "rank_registry",
)


class RunnerError(RuntimeError):
    pass


@dataclass
class CapabilityEvidence:
    parent_fd_identity: dict[str, Any] | None
    pass_fds: list[int]
    bootstrap_descriptor_sha256: str | None


PINNED_VLLM_HEAD = "8d9b52f7c2514490bdadfd5eb0c931e58625df2e"
PINNED_VLLM_TREE = "d7f16cac8369098d7fde19003ab2577171116ecb"


def _socket_identity(descriptor: int) -> dict[str, Any]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISSOCK(metadata.st_mode):
        raise RunnerError("prepared rank capability FD is not a socket")
    return {
        "fd": descriptor,
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": int(metadata.st_mode),
        "target": os.readlink(f"/proc/self/fd/{descriptor}"),
    }


def _one_glob(root: Path, pattern: str, label: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1 or not matches[0].is_file():
        raise RunnerError(f"{label} is not present exactly once in the product base")
    return matches[0].resolve(strict=True)


def load_product(prefix: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest, paths = base.load_product(prefix)
    artifacts = manifest["artifacts"]
    plugins = manifest["plugins"]
    snapshots = plugins["snapshots"]
    vllm_package = Path(snapshots["gemsim-vllm"]["package_path"])
    if vllm_package != Path(artifacts["vllm_plugin_init"]["path"]).parent:
        raise RunnerError("vLLM plugin package differs from its product snapshot")
    base_prefix = Path(manifest["base"]["prefix"]).resolve(strict=True)
    site = _one_glob(base_prefix, "venv/lib/python*/site-packages/vllm/__init__.py",
                     "installed vLLM package").parents[1]
    extended = dict(paths)
    extended.update(
        {
            "source_lock": Path(manifest["source_lock"]["path"]),
            "vllm_plugin_init": vllm_package / "__init__.py",
            "vllm_communicator": vllm_package / "communicator.py",
            "vllm_ccl_bootstrap": vllm_package / "ccl_bootstrap.py",
            "vllm_platform": vllm_package / "platform.py",
            "vllm_parallel_state": site / "vllm/distributed/parallel_state.py",
            "vllm_base_communicator": (
                site / "vllm/distributed/device_communicators/base_device_communicator.py"
            ),
            "vllm_communication_op": site / "vllm/distributed/communication_op.py",
            "vllm_version": site / "vllm/version.py",
            "vllm_metadata": _one_glob(
                site, "vllm-*.dist-info/METADATA", "installed vLLM metadata"
            ),
            "vllm_checkout_parallel_state": (
                ROOT / "projects/vllm/vllm/distributed/parallel_state.py"
            ),
            "vllm_checkout_base_communicator": (
                ROOT / "projects/vllm/vllm/distributed/device_communicators/base_device_communicator.py"
            ),
            "vllm_checkout_communication_op": (
                ROOT / "projects/vllm/vllm/distributed/communication_op.py"
            ),
            "vllm_checkout_version": ROOT / "projects/vllm/vllm/version.py",
            "vllm_linear": site / "vllm/model_executor/layers/linear.py",
            "vllm_config_vllm": site / "vllm/config/vllm.py",
            "vllm_config_parallel": site / "vllm/config/parallel.py",
            "vllm_config_model": site / "vllm/config/model.py",
            "vllm_adapters": vllm_package / "adapters.py",
            "vllm_row_parallel": vllm_package / "row_parallel.py",
            "vllm_ops": vllm_package / "ops.py",
            "vllm_kernels": vllm_package / "kernels.py",
            "ccl_acceptance_base": (
                ROOT / "tools/gemsim_ccl_live_allreduce_acceptance.py"
            ),
            "verifier": ROOT / "tools/gemsim_vllm_ccl_live_acceptance.py",
            "runner": Path(__file__).resolve(),
            "worker": ROOT / "examples/triton/vllm_ccl_live_rank.py",
        }
    )
    for role in IDENTITY_ROLES:
        path = extended.get(role)
        if path is None or not Path(path).resolve(strict=True).is_file():
            raise RunnerError(f"product/source identity is missing: {role}")
    inventory = {
        item.get("path"): item
        for item in manifest.get("inventory", [])
        if isinstance(item, Mapping)
    }
    for role in (
        "vllm_plugin_init", "vllm_communicator", "vllm_ccl_bootstrap", "vllm_platform",
        "vllm_adapters", "vllm_row_parallel", "vllm_ops", "vllm_kernels",
    ):
        path = Path(extended[role]).resolve(strict=True)
        relative = path.relative_to(prefix.resolve(strict=True)).as_posix()
        observed = base.file_record(path)
        binding = inventory.get(relative)
        if (
            not isinstance(binding, Mapping)
            or binding.get("kind") != "regular"
            or binding.get("bytes") != observed["bytes"]
            or binding.get("sha256") != observed["sha256"]
        ):
            raise RunnerError(f"{role} differs from the frozen product inventory")
    metadata = base.file_record(extended["vllm_metadata"])
    text = Path(extended["vllm_metadata"]).read_text(encoding="utf-8")
    if "Version: 0.0.dev0+g8d9b52f7c2\n" not in text:
        raise RunnerError("installed vLLM metadata differs from the pinned version")
    if metadata["bytes"] <= 0:
        raise RunnerError("installed vLLM metadata is empty")
    checkout = ROOT / "projects/vllm"
    head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain=v1", "--untracked-files=no"],
        text=True,
    )
    if head != PINNED_VLLM_HEAD or tree != PINNED_VLLM_TREE or dirty:
        raise RunnerError("pinned vLLM checkout HEAD/tree/clean identity mismatch")
    installed_to_checkout = {
        "vllm_parallel_state": "vllm_checkout_parallel_state",
        "vllm_base_communicator": "vllm_checkout_base_communicator",
        "vllm_communication_op": "vllm_checkout_communication_op",
        "vllm_version": "vllm_checkout_version",
    }
    for installed, source in installed_to_checkout.items():
        left = base.file_record(extended[installed])
        right = base.file_record(extended[source])
        if left["bytes"] != right["bytes"] or left["sha256"] != right["sha256"]:
            raise RunnerError(f"installed {installed} differs from pinned checkout")
    return manifest, extended


def source_identity(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    result = {role: base.file_record(paths[role]) for role in IDENTITY_ROLES}
    if set(result) != set(IDENTITY_ROLES):
        raise RunnerError("vLLM CCL source identity roles are incomplete")
    return result


def bootstrap_document(
    *,
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    preflight: Mapping[str, Mapping[str, Any]],
    rank: int,
    capability_fd: int,
    broker_pid: int,
    broker_start_time_ticks: int,
    timeout_ns: int,
    credits_per_peer: int,
) -> dict[str, Any]:
    prefix = str(Path(manifest["prefix"]).resolve(strict=True))
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "product": {
            "prefix": prefix,
            "manifest": dict(preflight["product_manifest"]),
            "runtime_library": dict(preflight["runtime_library"]),
        },
        "groups": [
            {
                "unique_name": "tp:0",
                "identity": dict(identity),
                "rank": {
                    "rank": rank,
                    "capability_fd": capability_fd,
                    "broker_pid": broker_pid,
                    "broker_start_time_ticks": broker_start_time_ticks,
                    "join_timeout_ns": timeout_ns,
                    "collective_timeout_ns": timeout_ns,
                    "credits_per_peer": credits_per_peer,
                },
            }
        ],
    }


def worker_environment(
    prefix: Path, run: base.RankProcess, bootstrap_path: Path
) -> dict[str, str]:
    environment = base._worker_environment(
        prefix, run.directory / "rank-launch.json", run.launch
    )
    environment["GEMSIM_CCL_BOOTSTRAP_DESCRIPTOR"] = str(bootstrap_path)
    return environment


def spawn_rank_process(
    run: base.RankProcess,
    *,
    config_path: Path,
    bootstrap_path: Path,
    product_prefix: Path,
    worker_script: Path | None = None,
    popen_factory: Callable[..., subprocess.Popen[bytes]],
) -> subprocess.Popen[bytes]:
    run.stdout = (run.directory / ".worker-stdout.log").open("xb")
    run.stderr = (run.directory / ".worker-stderr.log").open("xb")
    command = [
        sys.executable,
        str(worker_script or (ROOT / "examples/triton/vllm_ccl_live_rank.py")),
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
            env=worker_environment(product_prefix, run, bootstrap_path),
        )
    except BaseException:
        run.stdout.close()
        run.stderr.close()
        run.stdout = None
        run.stderr = None
        raise


def _read_rank_result(run: base.RankProcess, status: str, product: Mapping[str, Any],
                      world: int, workload: Mapping[str, Any]) -> dict[str, Any]:
    path = run.directory / "worker-result.json"
    if path.is_file():
        result, payload = base._read_json(path, f"rank {run.rank} result")
        if payload != base.canonical_json(result):
            raise RunnerError(f"rank {run.rank} result is not canonical")
        return result
    input_sha = (
        workload["input"]["sha256_by_rank"][run.rank]
        if workload["kind"] == "vllm-row-parallel" else base.sha256_bytes(
            base.deterministic_input("bfloat16", run.rank, int(run.element_count))
        )
    )
    result = {
        "schema": RANK_RESULT_SCHEMA,
        "status": status,
        "rank": run.rank,
        "world_size": world,
        "acceptance_authority": False,
        "live_adapter_accepted": False,
        "public_result_published": False,
        "input_sha256_before": input_sha,
        "input_sha256_after": input_sha,
        "output_sha256": None,
        "output_storage_fresh": None,
        "bootstrap_descriptor_sha256": run.capability.bootstrap_descriptor_sha256,
        "adapter_evidence_sha256": None,
        "managed_session": None,
        "first_error": None,
        "product": product,
    }
    base._exclusive_write(path, base.canonical_json(result))
    return result


def _normalize_failure(
    results: Sequence[dict[str, Any]], status: str, first_error: Any
) -> dict[str, Any] | None:
    document = None
    if first_error is not None and hasattr(first_error, "status"):
        document = {
            "native_status": int(first_error.status),
            "reporter_rank": int(first_error.reporter_rank),
            "failed_rank": int(first_error.failed_rank),
            "context_sequence": int(first_error.context_sequence),
        }
    for result in results:
        result["status"] = status
        result["public_result_published"] = False
        result["output_sha256"] = None
        result["output_storage_fresh"] = None
        result["first_error"] = document
    return document


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
    workload_path: Path | None = None,
    worker_script: Path | None = None,
    worker_config_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_path = expected_path.resolve(strict=True)
    output = Path(os.path.abspath(output))
    execution_root = Path(os.path.abspath(execution_root))
    if output.exists() or output.is_symlink() or execution_root.exists() or execution_root.is_symlink():
        raise RunnerError("output and execution roots must both be absent")
    if (
        output == execution_root or output in execution_root.parents
        or execution_root in output.parents
        or execution_root.parent.resolve(strict=True) != execution_root.parent
    ):
        raise RunnerError("execution root is not an independent normalized namespace")
    if timeout_seconds <= 0 or cleanup_grace_seconds <= 0:
        raise RunnerError("timeouts must be positive")
    expected, expected_payload = base._read_json(expected_path, "expected wrapper")
    if (
        expected_payload != base.canonical_json(expected)
        or expected.get("schema") != EXPECTED_SCHEMA
    ):
        raise RunnerError("expected wrapper identity is invalid")
    design = base._validate_expected_and_paths(expected, output, execution_root)
    config = design["config"]
    world = int(config["world_size"])
    if not 2 <= world <= 16 or config["dtype"] != "bfloat16":
        raise RunnerError("vLLM live adapter accepts BF16 world_size 2..16")
    if [rank.get("rank") for rank in design["ranks"]] != list(range(world)):
        raise RunnerError("expected ranks are not exactly 0..world_size-1")
    if any(
        int(rank.get("expected", {}).get("device_sum_launches", 0)) <= 0
        for rank in design["ranks"]
    ):
        raise RunnerError("each vLLM adapter rank requires a live device SUM")
    if workload_path is None:
        workload = {
            "schema": "amdgpu-sim.vllm-ccl-workload.v1",
            "kind": "standalone-allreduce",
            "model": None,
            "layer": None,
            "input": {"policy": "rank-affine-mod127-v1"},
            "collective": {
                "dtype": "bfloat16",
                "element_count": int(config["element_count"]),
            },
        }
    else:
        workload, workload_payload = base._read_json(
            workload_path.resolve(strict=True), "vLLM workload"
        )
        if workload_payload != base.canonical_json(workload):
            raise RunnerError("vLLM workload is not canonical")
    try:
        from tools import gemsim_vllm_ccl_live_acceptance as acceptance

        workload = acceptance.validate_workload(workload, live=True)
    except Exception as error:
        raise RunnerError(f"vLLM workload is invalid: {error}") from error
    workload_payload = base.canonical_json(workload)
    workload_sha256 = base.sha256_bytes(workload_payload)
    if workload["kind"] == "vllm-row-parallel":
        if world != 2 or config["element_count"] != 1024:
            raise RunnerError("RowParallel first workload requires N=2 BF16[1024]")
        if config["model_identity_sha256"] != workload_sha256:
            raise RunnerError("collective group does not bind the RowParallel workload")

    manifest, product_paths = load_product(product_prefix)
    preflight = source_identity(product_paths)
    runtime_path = product_paths["runtime_library"].resolve(strict=True)
    native = native_factory(runtime_path)
    if native.library_sha256 != design["runtime"]["sha256"]:
        raise RunnerError("expected runtime differs from frozen product")
    identity_values = {
        "world_size": world,
        "epoch": int(config["epoch"]),
        "group_generation": int(config["group_generation"]),
        "job_uuid": config["job_uuid"],
        "group_uuid": config["group_uuid"],
        "model_identity_sha256": config["model_identity_sha256"],
    }
    native_identity = native.identity(
        world_size=world,
        epoch=identity_values["epoch"],
        group_generation=identity_values["group_generation"],
        job_uuid=bytes.fromhex(identity_values["job_uuid"]),
        group_uuid=bytes.fromhex(identity_values["group_uuid"]),
        model_identity_sha256=bytes.fromhex(identity_values["model_identity_sha256"]),
    )
    timeout_ns = max(1, int(timeout_seconds * 1_000_000_000))
    started_at_ns = native.monotonic_time_ns()
    absolute_deadline_ns = started_at_ns + timeout_ns
    output.mkdir(mode=0o700)
    execution_root.mkdir(mode=0o700)
    os.chmod(output, 0o700)
    os.chmod(execution_root, 0o700)
    rendezvous_path = execution_root / ".vllm-gloo-rendezvous"
    if rendezvous_path.exists() or rendezvous_path.is_symlink():
        raise RunnerError("Gloo rendezvous path must be absent")
    base.enable_subreaper()
    baseline_fds = base.owned_fd_snapshot()
    baseline_children = base._direct_child_identities(os.getpid())
    runs = [base._prepare_rank_tree(output, item) for item in design["ranks"]]
    for run in runs:
        run.element_count = int(config["element_count"])
        run.capability = CapabilityEvidence(
            parent_fd_identity=None,
            pass_fds=[],
            bootstrap_descriptor_sha256=None,
        )
    broker = None
    first_error = None
    status = "device_failure"
    interrupted = False
    launching_rank = 0
    try:
        broker = native.live_broker(native_identity)
        owner = broker.owner
        for run, rank_design in zip(runs, design["ranks"]):
            launching_rank = run.rank
            run.capability_fd = broker.prepare_rank(run.rank)
            parent_identity = _socket_identity(run.capability_fd)
            bootstrap = bootstrap_document(
                manifest=manifest,
                identity=identity_values,
                preflight=preflight,
                rank=run.rank,
                capability_fd=run.capability_fd,
                broker_pid=int(owner.pid),
                broker_start_time_ticks=int(owner.start_time_ticks),
                timeout_ns=timeout_ns,
                credits_per_peer=int(design["limits"]["credits_per_peer"]),
            )
            bootstrap_payload = base.canonical_json(bootstrap)
            bootstrap_path = run.directory / ".bootstrap-descriptor.json"
            base._exclusive_write(bootstrap_path, bootstrap_payload)
            os.chmod(bootstrap_path, 0o400, follow_symlinks=False)
            run.capability = CapabilityEvidence(
                parent_fd_identity=parent_identity,
                pass_fds=[run.capability_fd],
                bootstrap_descriptor_sha256=base.sha256_bytes(bootstrap_payload),
            )
            product_binding = {
                "product_id": manifest["product_id"],
                "manifest_sha256": preflight["product_manifest"]["sha256"],
                "prefix": str(product_prefix.resolve(strict=True)),
                "ccl_engine": preflight["ccl_engine"],
                "vllm_plugin_init": preflight["vllm_plugin_init"],
                "vllm_communicator": preflight["vllm_communicator"],
            }
            expected_import_roles = (
                acceptance.ROW_PARALLEL_IMPORT_ROLES
                if workload["kind"] == "vllm-row-parallel"
                else acceptance.ACTUAL_IMPORT_ROLES
            )
            worker_config = {
                "schema": WORKER_CONFIG_SCHEMA,
                "rank": run.rank,
                "world_size": world,
                "element_count": int(config["element_count"]),
                "dtype": "bfloat16",
                "unique_name": "tp:0",
                "rendezvous_path": str(rendezvous_path),
                "bootstrap_descriptor_path": str(bootstrap_path),
                "bootstrap_descriptor_sha256": run.capability.bootstrap_descriptor_sha256,
                "result_path": str(run.directory / "worker-result.json"),
                "adapter_evidence_path": str(run.directory / "adapter-evidence.json"),
                "journal_path": str(run.directory / "adapter-events.jsonl"),
                "input_path": str(run.directory / "input.bin"),
                "output_path": str(run.directory / "output.bin"),
                "runtime_library": str(runtime_path),
                "rank_launch_sha256": rank_design["rank_launch_sha256"],
                "epoch": identity_values["epoch"],
                "group_generation": identity_values["group_generation"],
                "job_uuid": identity_values["job_uuid"],
                "group_uuid": identity_values["group_uuid"],
                "model_identity_sha256": identity_values["model_identity_sha256"],
                "expected_imports": {
                    role: preflight[role]
                    for role in expected_import_roles
                },
                "workload": workload,
                "product": product_binding,
            }
            if worker_script is not None:
                if worker_config_extra is None:
                    raise RunnerError("custom worker requires an explicit config")
                worker_config.update(dict(worker_config_extra))
            config_path = run.directory / ".worker-config.json"
            base._exclusive_write(config_path, base.canonical_json(worker_config))
            run.process = spawn_rank_process(
                run, config_path=config_path, bootstrap_path=bootstrap_path,
                product_prefix=product_prefix, worker_script=worker_script,
                popen_factory=popen_factory,
            )
            os.close(run.capability_fd)
            run.capability_fd = -1
            process_identity = native.process_identity(run.process.pid)
            run.start_time_ticks = int(process_identity.start_time_ticks)
            broker.bind_rank(run.rank, process_identity)
        broker.rendezvous(absolute_deadline_ns)
        while True:
            base._capture_owned_processes(runs, baseline_children)
            base._poll_workers(runs)
            observed = broker.progress()
            if observed is not None:
                first_error = observed
                status = base._normalize_failure_status(int(observed.status))
            if all(run.returncode is not None for run in runs) or first_error is not None:
                break
            if native.monotonic_time_ns() >= absolute_deadline_ns:
                status = "timed_out"
                first_error = broker.abort(0, TIMED_OUT, 1)
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        interrupted = True
        try:
            first_error = broker.abort(launching_rank, CANCELLED, 1) if broker else None
        except Exception:
            pass
    except CCLStatusError as error:
        status = base._normalize_failure_status(int(error.status))
        try:
            first_error = broker.first_error() if broker else None
        except Exception:
            pass
    except Exception:
        if broker is not None:
            try:
                first_error = broker.abort(launching_rank, PROTOCOL_ERROR, 1)
            except Exception:
                pass
    finally:
        for run in runs:
            if run.capability_fd >= 0:
                os.close(run.capability_fd)
                run.capability_fd = -1
        failed = [run for run in runs if run.process and run.returncode not in (None, 0)]
        if broker is not None and first_error is None and (failed or interrupted):
            failed_rank = failed[0].rank if failed else launching_rank
            try:
                first_error = broker.abort(failed_rank, PEER_LOST, 1)
                status = "peer_lost"
            except Exception:
                pass
        relay_deadline = native.monotonic_time_ns() + max(
            1, int(cleanup_grace_seconds * 1_000_000_000)
        )
        try:
            while (
                broker is not None and broker.info().abort_pending_mask
                and native.monotonic_time_ns() < relay_deadline
            ):
                observed = broker.progress()
                if first_error is None and observed is not None:
                    first_error = observed
                time.sleep(0.001)
        except Exception:
            pass
        children_exhausted = base.terminate_group(
            runs, cleanup_grace_seconds, baseline_children
        )
        if broker is not None:
            try:
                broker.destroy()
            except Exception:
                children_exhausted = False

    post_fds = base.owned_fd_snapshot()
    fd_added = dict(set(post_fds.items()) - set(baseline_fds.items()))
    fd_removed = dict(set(baseline_fds.items()) - set(post_fds.items()))
    owned_fd_delta = len(fd_added) + len(fd_removed)
    orphan_count = sum(
        base._present(pid, started)
        for run in runs for pid, started in run.descendants.items()
    )
    workers_reaped = all(run.process is None or run.process.poll() is not None for run in runs)
    cleanup_complete = children_exhausted and workers_reaped and owned_fd_delta == 0 and orphan_count == 0
    if not cleanup_complete:
        status = "device_failure"

    product_binding = {
        "product_id": manifest["product_id"],
        "manifest_sha256": preflight["product_manifest"]["sha256"],
        "prefix": str(product_prefix.resolve(strict=True)),
        "ccl_engine": preflight["ccl_engine"],
        "vllm_plugin_init": preflight["vllm_plugin_init"],
        "vllm_communicator": preflight["vllm_communicator"],
    }
    results = [
        _read_rank_result(run, status, product_binding, world, workload)
        for run in runs
    ]
    all_success = cleanup_complete and not interrupted and first_error is None and all(
        run.returncode == 0 and result.get("status") == "success"
        for run, result in zip(runs, results)
    )
    status = "success" if all_success else status
    if status == "success" and not all_success:
        status = "device_failure"
    first_error_document = None
    if status != "success":
        first_error_document = _normalize_failure(results, status, first_error)

    for run, result in zip(runs, results):
        output_path = run.directory / "output.bin"
        if status != "success" and output_path.exists() and output_path.stat().st_size:
            output_path.unlink()
            base._exclusive_write(output_path, b"")
        for name in ("adapter-evidence.json", "adapter-events.jsonl", "input.bin", "output.bin"):
            path = run.directory / name
            if not path.exists():
                base._exclusive_write(path, b"" if name.endswith((".bin", ".jsonl")) else base.canonical_json({}))
        bootstrap_source = run.directory / ".bootstrap-descriptor.json"
        if not bootstrap_source.exists():
            placeholder = {
                "schema": BOOTSTRAP_SCHEMA,
                "status": "rank_not_prepared",
                "rank": run.rank,
                "world_size": world,
            }
            placeholder_payload = base.canonical_json(placeholder)
            base._exclusive_write(bootstrap_source, placeholder_payload)
            run.capability.bootstrap_descriptor_sha256 = base.sha256_bytes(
                placeholder_payload
            )
        base._exclusive_write(
            run.directory / "bootstrap-descriptor.json", bootstrap_source.read_bytes()
        )
        if status != "success":
            base._atomic_manifest(run.directory / "worker-result.json", result)
        base._copy_runtime_artifacts(run, execution_root)
        for stream, name in (
            (run.stdout, "worker-stdout.log"),
            (run.stderr, "worker-stderr.log"),
        ):
            if stream is not None and not stream.closed:
                stream.flush()
            source_log = run.directory / f".{name}"
            base._exclusive_write(
                run.directory / name,
                source_log.read_bytes() if source_log.exists() else b"",
            )
        for hidden in (".worker-config.json", ".worker-stdout.log", ".worker-stderr.log", ".bootstrap-descriptor.json"):
            try:
                (run.directory / hidden).unlink()
            except FileNotFoundError:
                pass
    try:
        if rendezvous_path.exists() or rendezvous_path.is_symlink():
            rendezvous_path.unlink()
        execution_root.rmdir()
    except OSError:
        cleanup_complete = False
        status = "device_failure"
    if not cleanup_complete:
        first_error_document = _normalize_failure(
            results, "device_failure", first_error
        )
        for run, result in zip(runs, results):
            output_path = run.directory / "output.bin"
            if output_path.stat().st_size:
                output_path.unlink()
                base._exclusive_write(output_path, b"")
            base._atomic_manifest(run.directory / "worker-result.json", result)

    postflight = source_identity(product_paths)
    cleanup = {
        "baseline_fds": base._fd_documents(baseline_fds),
        "baseline_fd_count": len(baseline_fds),
        "baseline_fd_sha256": base.object_sha256(base._fd_documents(baseline_fds)),
        "post_fds": base._fd_documents(post_fds),
        "post_fd_count": len(post_fds),
        "post_fd_sha256": base.object_sha256(base._fd_documents(post_fds)),
        "added_fds": base._fd_documents(fd_added),
        "removed_fds": base._fd_documents(fd_removed),
        "measured_fd_delta": owned_fd_delta,
        "children_exhausted": children_exhausted,
        "workers_reaped": workers_reaped,
        "new_child_identities": base._process_documents([
            (run.rank, "worker", run.process.pid, run.start_time_ticks)
            for run in runs if run.process is not None and run.start_time_ticks is not None
        ] + [
            (run.rank, "daemon_or_descendant", pid, started)
            for run in runs for pid, started in run.descendants.items()
        ]),
        "orphan_identities": [],
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
                "capability": {
                    "parent_fd_identity": run.capability.parent_fd_identity,
                    "pass_fds": run.capability.pass_fds,
                    "bootstrap_descriptor_sha256": run.capability.bootstrap_descriptor_sha256,
                },
                "artifacts": {
                    name: base.artifact_record(output, run.directory / name)
                    for name in RANK_FILES
                },
                "cleanup": {
                    "worker_reaped": run.process is None or run.process.poll() is not None,
                    "daemon_reaped": not any(
                        base._present(pid, started) for pid, started in run.descendants.items()
                    ),
                },
            }
        )
    result_manifest = {
        "schema": RUN_SCHEMA,
        "status": status,
        "acceptance_authority": False,
        "live_adapter_accepted": False,
        "expected": {
            "schema": EXPECTED_SCHEMA,
            "bytes": len(expected_payload),
            "sha256": base.sha256_bytes(expected_payload),
        },
        "workload": {
            "schema": workload["schema"],
            "bytes": len(workload_payload),
            "sha256": workload_sha256,
            "document": workload,
        },
        "world_size": world,
        "element_count": int(config["element_count"]),
        "dtype": "bfloat16",
        "unique_name": "tp:0",
        "job_uuid": identity_values["job_uuid"],
        "group_uuid": identity_values["group_uuid"],
        "epoch": identity_values["epoch"],
        "group_generation": identity_values["group_generation"],
        "started_at_ns": started_at_ns,
        "completed_at_ns": native.monotonic_time_ns(),
        "absolute_deadline_ns": absolute_deadline_ns,
        "target_execution_completed": status == "success",
        "target_feedback": False,
        "oracle_phase": "post_target",
        "oracle_feedback": False,
        "first_error": first_error_document,
        "supervisor_cleanup": cleanup,
        "source_identity_preflight": preflight,
        "source_identity_postflight": postflight,
        "vllm_checkout": {
            "path": str((ROOT / "projects/vllm").resolve(strict=True)),
            "head": PINNED_VLLM_HEAD,
            "tree": PINNED_VLLM_TREE,
            "tracked_clean": True,
            "installed_version": "0.0.dev0+g8d9b52f7c2",
        },
        "ranks": rank_entries,
    }
    base._atomic_manifest(output / "result-manifest.json", result_manifest)
    return result_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--product-prefix", type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument(
        "--worker-script", type=Path,
        help="use an explicitly supplied diagnostic worker instead of the formal CCL worker",
    )
    parser.add_argument("--worker-config-json", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--cleanup-grace-seconds", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    worker_script = args.worker_script.resolve(strict=True) if args.worker_script else None
    worker_config_extra = None
    if args.worker_config_json is not None:
        worker_config_extra, payload = base._read_json(
            args.worker_config_json.resolve(strict=True), "custom worker config"
        )
        if payload != base.canonical_json(worker_config_extra):
            raise RunnerError("custom worker config is not canonical")
    if worker_script is not None and worker_config_extra is None:
        raise RunnerError("--worker-script requires --worker-config-json")
    manifest = supervise(
        expected_path=args.expected,
        output=args.output_dir,
        execution_root=args.execution_root,
        product_prefix=args.product_prefix or base.default_product_prefix(),
        timeout_seconds=args.timeout_seconds,
        cleanup_grace_seconds=args.cleanup_grace_seconds,
        workload_path=args.workload,
        worker_script=worker_script,
        worker_config_extra=worker_config_extra,
    )
    return 0 if manifest["status"] == "success" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
