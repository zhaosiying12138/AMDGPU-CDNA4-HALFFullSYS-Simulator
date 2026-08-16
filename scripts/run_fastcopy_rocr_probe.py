#!/usr/bin/env python3
"""Run the small unchanged-upstream ROCr H2D/D2H probe in legacy or fast mode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
ROCR_STAGE = ROOT / "build/fastcopy-rocr-eligibility-stage"
RUNTIME_BUILD = ROOT / "build/fastcopy-runtime"
GEM5 = ROOT / "build/FASTCOPY_VEGA_X86/gem5.opt"
GEM5_CONFIG = ROOT / "projects/gem5/configs/example/gemsim/host_dispatch.py"
WORKER = RUNTIME_BUILD / "tests/self_amdgpu_runtime_upstream_rocr_model_test"
MODEL = RUNTIME_BUILD / "libself_amdgpu_hsakmt_model.so.1.1.0"
KERNEL = ROOT / (
    "projects/rocm-systems/projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/"
    "gfx950/gpuReadWrite_kernels.hsaco"
)


def require_file(path: Path, executable: bool = False) -> Path:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"required regular file is missing: {path}")
    if executable and not os.access(path, os.X_OK):
        raise RuntimeError(f"required file is not executable: {path}")
    return path


def topology() -> Path:
    candidates = sorted((RUNTIME_BUILD / "tests").glob("hsakmt-model-topology-*"))
    candidates = [candidate for candidate in candidates if candidate.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one generated topology, found {candidates}")
    return candidates[0]


def members(pgid: int) -> list[int]:
    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = entry.joinpath("stat").read_text(encoding="ascii")
            closing = fields.rfind(")")
            values = fields[closing + 2 :].split()
            if int(values[2]) == pgid:
                result.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return result


def terminate(process: subprocess.Popen[bytes] | None, pgid: int | None) -> bool:
    if process is None or pgid is None:
        return False
    if process.poll() is not None and not members(pgid):
        return False
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if process.poll() is not None and not members(pgid):
            return True
        time.sleep(0.02)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        pass
    return True


def wait_socket(path: Path, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"gem5 exited before endpoint publication: {process.returncode}")
        try:
            mode = path.stat().st_mode
        except FileNotFoundError:
            time.sleep(0.02)
            continue
        if not stat.S_ISSOCK(mode):
            raise RuntimeError("bridge endpoint is not a socket")
        return
    raise RuntimeError("bridge endpoint publication timed out")


def run(
    mode: str,
    output: Path,
    timeout: float,
    worker: Path | None = None,
    worker_args: list[str] | None = None,
    allow_idle_gem5: bool = False,
) -> int:
    selected_worker = (worker or WORKER).resolve()
    selected_args = list(worker_args or [])
    required = [GEM5, GEM5_CONFIG, selected_worker, MODEL]
    if worker is None:
        required.append(KERNEL)
        if not selected_args:
            selected_args = ["--execute", str(KERNEL)]
    for path in required:
        require_file(path, executable=path in (GEM5, selected_worker))
    topo = topology()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"output must be absent: {output}")
    output.mkdir(mode=0o700, parents=True)
    run_dir = output / "run"
    run_dir.mkdir(mode=0o700)
    for relative in ("home", "tmp", "xdg-cache", "xdg-config", "xdg-data", "m5out"):
        (run_dir / relative).mkdir(mode=0o700)
    endpoint = run_dir / "bridge.sock"
    trace = run_dir / "dispatch-trace.jsonl"
    job_uuid = secrets.token_hex(16)
    common = {
        "HOME": str(run_dir / "home"),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TMPDIR": str(run_dir / "tmp"),
        "XDG_CACHE_HOME": str(run_dir / "xdg-cache"),
        "XDG_CONFIG_HOME": str(run_dir / "xdg-config"),
        "XDG_DATA_HOME": str(run_dir / "xdg-data"),
    }
    gem5_argv = [
        str(GEM5), "--listener-mode=on", "--outdir", str(run_dir / "m5out"),
        str(GEM5_CONFIG), "--endpoint", str(endpoint), "--dispatch-trace-path",
        str(trace), "--epoch", "1", "--job-uuid", job_uuid, "--rank", "0",
        "--world-size", "1", "--startup-timeout-ms", "15000",
        "--handshake-timeout-ms", "15000", "--run-timeout-ms", "300000",
    ]
    worker_env = dict(common)
    worker_env.update({
        "HSA_ENABLE_DTIF_FAST_COPY": "0" if mode == "legacy" else "1",
        "SAGR_HSAKMT_MODEL_FAST_COPY": "1" if mode == "fast" else "0",
        "HSA_ENABLE_DXG_DETECTION": "0",
        "HSA_ENABLE_INTERRUPT": "0",
        "HSA_MODEL_LIB": str(MODEL),
        "HSA_MODEL_TOPOLOGY": str(topo),
        "LD_LIBRARY_PATH": f"{RUNTIME_BUILD}:{ROCR_STAGE / 'lib'}",
        "SAGR_GENERIC_BRIDGE_ENDPOINT": str(endpoint),
        "SAGR_HSAKMT_MODEL_TRACE": "1",
        "SAGR_UPSTREAM_ROCR_EXECUTION_TRACE": "1",
    })
    gem5_process: subprocess.Popen[bytes] | None = None
    worker_process: subprocess.Popen[bytes] | None = None
    gem5_pgid: int | None = None
    worker_pgid: int | None = None
    started = time.perf_counter()
    failure: str | None = None
    worker_code: int | None = None
    gem5_code: int | None = None
    gem5_expected_idle = False
    worker_log = (run_dir / "worker.log").open("wb", buffering=0)
    gem5_log = (run_dir / "gem5.log").open("wb", buffering=0)
    try:
        gem5_process = subprocess.Popen(
            gem5_argv, cwd=ROOT, env=common, stdin=subprocess.DEVNULL,
            stdout=gem5_log, stderr=subprocess.STDOUT, close_fds=True,
            start_new_session=True,
        )
        gem5_pgid = gem5_process.pid
        wait_socket(endpoint, gem5_process, 30.0)
        worker_process = subprocess.Popen(
            [str(selected_worker), *selected_args], cwd=ROOT, env=worker_env,
            stdin=subprocess.DEVNULL, stdout=worker_log, stderr=subprocess.STDOUT,
            close_fds=True, start_new_session=True,
        )
        worker_pgid = worker_process.pid
        worker_code = worker_process.wait(timeout=timeout)
        if worker_code != 0:
            raise RuntimeError(f"worker exited {worker_code}")
        try:
            gem5_code = gem5_process.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            if not allow_idle_gem5:
                raise RuntimeError("gem5 did not exit after worker completion")
            gem5_expected_idle = True
            terminate(gem5_process, gem5_pgid)
        if gem5_code != 0:
            if not gem5_expected_idle:
                raise RuntimeError(f"gem5 exited {gem5_code}")
    except Exception as error:  # noqa: BLE001
        failure = str(error)
    finally:
        worker_forced = terminate(worker_process, worker_pgid)
        gem5_forced = terminate(gem5_process, gem5_pgid)
        if worker_process is not None:
            worker_code = worker_process.wait(timeout=3.0)
        if gem5_process is not None:
            gem5_code = gem5_process.wait(timeout=3.0)
        worker_log.close()
        gem5_log.close()
    rows = []
    if trace.is_file():
        for line in trace.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    summary = {
        "schema": "amdgpu-sim.fastcopy-rocr-probe.v1",
        "mode": mode,
        "status": "success"
        if failure is None
        and worker_code == 0
        and (gem5_code == 0 or gem5_expected_idle)
        else "failure",
        "failure": failure,
        "worker_exit_code": worker_code,
        "gem5_exit_code": gem5_code,
        "elapsed_seconds": time.perf_counter() - started,
        "retired_dispatches": sum(row.get("event") == "native_execution_retired" for row in rows),
        "trace_rows": len(rows),
        "output": str(output),
        "fast_copy_gates": mode == "fast",
        "provider_gate_only": mode == "hsa-only",
        "worker": str(selected_worker),
        "worker_args": selected_args,
        "gem5_expected_idle": gem5_expected_idle,
        "worker_output": (run_dir / "worker.log").read_text(encoding="utf-8", errors="replace")[-4096:],
        "gem5_output": (run_dir / "gem5.log").read_text(encoding="utf-8", errors="replace")[-4096:],
    }
    (output / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "success" and not worker_forced and not gem5_forced else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("legacy", "hsa-only", "fast"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--worker-arg", action="append", default=[])
    parser.add_argument("--allow-idle-gem5", action="store_true")
    args = parser.parse_args()
    return run(
        args.mode,
        args.output.resolve(),
        args.timeout,
        args.worker,
        args.worker_arg,
        args.allow_idle_gem5,
    )


if __name__ == "__main__":
    raise SystemExit(main())
