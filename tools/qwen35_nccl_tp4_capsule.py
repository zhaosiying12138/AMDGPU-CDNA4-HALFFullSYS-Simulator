#!/usr/bin/env python3
"""Weight-free world=4 RCCL all-reduce discriminator/regression capsule.

Run it in the SGLang lane environment (the model path is identity only; this
capsule never imports SGLang and never opens model weights):

  QWEN35_NCCL_TP4_OUTPUT=artifacts/qwen35-nccl-tp4/run-1 \
    scripts/run_engine_lane.sh --engine sglang --tp 4 \
    --model models/Qwen3.5-9B \
    --capsule tools/qwen35_nccl_tp4_capsule.py \
    artifacts/qwen35-nccl-tp4/run-1/lane.log

The supervisor launches every rank as an isolated child, including rank 0.
That is deliberate: a collective can wedge rank 0 just as easily as another
rank, so keeping the supervisor out of c10d is what makes the wall timeout and
final report reliable.  Ranks 1..3 are therefore real peers rather than dummy
participants.

The 9B log's PyTorch watchdog message only proves that its monitor could not
enter a HIP API; it does not identify the in-flight kernel as an RCCL kernel.
Consequently, a pass here exonerates this precise RCCL sequence and a failure
reproduces it, but a pass does not by itself explain the full-model watchdog.

Environment:
  QWEN35_NCCL_TP4_OUTPUT            report directory
  QWEN35_NCCL_TP4_TIMEOUT_SECONDS   whole-capsule deadline (default: 600)
  QWEN35_NCCL_TP4_TERM_GRACE_SECONDS process-group TERM grace (default: 5)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import string
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
WORLD_SIZE = 4
HIDDEN_SIZE = 4096
OBSERVED_PREFILL_TOKENS = 2

OUTPUT_ENV = "QWEN35_NCCL_TP4_OUTPUT"
TIMEOUT_ENV = "QWEN35_NCCL_TP4_TIMEOUT_SECONDS"
TERM_GRACE_ENV = "QWEN35_NCCL_TP4_TERM_GRACE_SECONDS"
GEMM_PRESSURE_ENV = "QWEN35_NCCL_TP4_GEMM_PRESSURE_GIB"
ALL_GATHER_ENV = "QWEN35_NCCL_TP4_ALL_GATHER"
ALL_GATHER_LM_HEAD_ENV = "QWEN35_NCCL_TP4_ALL_GATHER_LM_HEAD"
WORKER_RANK_ENV = "_QWEN35_NCCL_TP4_WORKER_RANK"
RUN_ID_ENV = "_QWEN35_NCCL_TP4_RUN_ID"
MASTER_ADDR_ENV = "_QWEN35_NCCL_TP4_MASTER_ADDR"
MASTER_PORT_ENV = "_QWEN35_NCCL_TP4_MASTER_PORT"
SUPERVISOR_PID_ENV = "_QWEN35_NCCL_TP4_SUPERVISOR_PID"

REPORT_SCHEMA = "amdgpu-sim.qwen35-nccl-tp4-capsule.v1"
RANK_REPORT_SCHEMA = "amdgpu-sim.qwen35-nccl-tp4-rank.v1"
PR_SET_PDEATHSIG = 1

COMMUNICATION_ENVIRONMENT_NAMES = (
    "NCCL_SHM_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
)

PRODUCTION_SHAPE_EVIDENCE = {
    "model_config": "models/Qwen3.5-9B/config.json:text_config.hidden_size=4096",
    "frozen_request": "tools/qwen35_token_gate.py:PROMPT_TOKEN_IDS has 2 tokens",
    "observed_tp4_log": (
        "artifacts/perf-verify/20260828-9b-tp4/lane.log:6197 "
        "records M=2,K=4096"
    ),
}


@dataclass(frozen=True)
class Operation:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def elements(self) -> int:
        return math.prod(self.shape)


# Exactly two tiny calls followed by the production embedding call.  The
# production shape comes from the pinned 9B config (hidden_size=4096) and the
# two-token request recorded by the 9B TP4 lane (M=2, K=4096).
OPERATIONS = (
    Operation("tiny_scalar", (1,), "float32"),
    Operation("tiny_vector", (16,), "float32"),
    Operation(
        "qwen35_9b_prefill_embedding",
        (OBSERVED_PREFILL_TOKENS, HIDDEN_SIZE),
        "bfloat16",
    ),
)


@dataclass
class RankProcess:
    rank: int
    process: subprocess.Popen[bytes]


class SupervisorTermination(RuntimeError):
    def __init__(self, signal_number: int) -> None:
        self.signal_number = signal_number
        super().__init__(f"received {signal.Signals(signal_number).name}")


def log(message: str, *, rank: int | str = "supervisor") -> None:
    print(f"[tp4-rccl rank={rank}] {message}", flush=True)


def output_directory() -> Path:
    default = ROOT / "artifacts/qwen35-nccl-tp4-capsule/v1"
    return Path(os.environ.get(OUTPUT_ENV, str(default))).expanduser().resolve()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def write_json_atomic(path: Path, value: object) -> None:
    """Publish a complete JSON file and fsync both file and directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_CLOEXEC,
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _input_integer(operation: Operation, rank: int, index: int) -> int:
    """Return values whose world=4 sums are exactly representable."""

    if operation.name == "tiny_scalar":
        return rank + 1
    if operation.name == "tiny_vector":
        return 2 * (rank + 1) + (index % 7) - 3
    if operation.name == "qwen35_9b_prefill_embedding":
        return rank + 1 + (index % 17) - 8
    raise ValueError(f"unknown operation: {operation.name}")


def input_values(operation: Operation, rank: int) -> list[float]:
    if rank not in range(WORLD_SIZE):
        raise ValueError(f"rank must be in [0, {WORLD_SIZE}): {rank}")
    return [
        float(_input_integer(operation, rank, index))
        for index in range(operation.elements)
    ]


def expected_values(operation: Operation) -> list[float]:
    return [
        float(
            sum(
                _input_integer(operation, rank, index)
                for rank in range(WORLD_SIZE)
            )
        )
        for index in range(operation.elements)
    ]


def operation_contract() -> list[dict[str, object]]:
    return [
        {
            "name": operation.name,
            "collective": "all_reduce_sum",
            "shape": list(operation.shape),
            "dtype": operation.dtype,
            "elements": operation.elements,
            "bytes": operation.elements
            * (2 if operation.dtype == "bfloat16" else 4),
            "comparison": "exact_elementwise_equality",
        }
        for operation in OPERATIONS
    ]


def communication_environment(
    environment: Mapping[str, str] = os.environ,
) -> dict[str, str | None]:
    """Capture transport selection without pretending an unset knob is set."""

    return {
        name: environment.get(name) for name in COMMUNICATION_ENVIRONMENT_NAMES
    }


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def parse_positive_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number: {raw!r}") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number: {raw!r}")
    return value


def parse_nonnegative_gib() -> float:
    raw = os.environ.get(GEMM_PRESSURE_ENV, "0")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{GEMM_PRESSURE_ENV} must be non-negative: {raw!r}") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{GEMM_PRESSURE_ENV} must be non-negative: {raw!r}")
    return value


def worker_environment(
    base: Mapping[str, str],
    *,
    rank: int,
    run_id: str,
    output: Path,
    address: str,
    port: int,
) -> dict[str, str]:
    if rank not in range(WORLD_SIZE):
        raise ValueError(f"rank must be in [0, {WORLD_SIZE}): {rank}")
    environment = dict(base)
    environment.update(
        {
            WORKER_RANK_ENV: str(rank),
            RUN_ID_ENV: run_id,
            MASTER_ADDR_ENV: address,
            MASTER_PORT_ENV: str(port),
            SUPERVISOR_PID_ENV: str(os.getpid()),
            OUTPUT_ENV: str(output),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(WORLD_SIZE),
            "MASTER_ADDR": address,
            "MASTER_PORT": str(port),
        }
    )
    return environment


def arm_parent_death_signal(expected_parent_pid: int) -> None:
    """Kill a detached rank if its supervisor dies before normal cleanup."""

    if sys.platform != "linux":
        raise RuntimeError("the RCCL capsule requires Linux parent-death support")
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    if prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), "prctl")
    # PR_SET_PDEATHSIG cannot report a death that happened before the call.
    # Checking after arming closes that race: a changed parent means the
    # supervisor is already gone and this rank must not initialize a device.
    actual_parent_pid = os.getppid()
    if actual_parent_pid != expected_parent_pid:
        raise RuntimeError(
            "capsule supervisor disappeared before worker startup: "
            f"expected_ppid={expected_parent_pid} actual_ppid={actual_parent_pid}"
        )


def install_supervisor_signal_handlers() -> dict[int, Any]:
    previous = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGHUP: signal.getsignal(signal.SIGHUP),
    }

    def terminate(signal_number: int, _frame: Any) -> None:
        # Ignore repeats while the exception drives owned-rank cleanup and
        # durable failure-report publication.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        raise SupervisorTermination(signal_number)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    return previous


def restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signal_number, handler in previous.items():
        signal.signal(signal_number, handler)


def _tensor_sha256(tensor: Any) -> str:
    import torch

    # View as bytes only after moving to CPU so bfloat16 needs no NumPy dtype
    # support.  The hash records the exact dtype representation, not values
    # after a potentially lossy cast.
    payload = tensor.detach().contiguous().cpu()
    raw = payload.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _rank_report_path(output: Path, rank: int) -> Path:
    return output / f"rank-{rank}.json"


def run_worker(rank: int) -> int:
    run_id = os.environ[RUN_ID_ENV]
    address = os.environ[MASTER_ADDR_ENV]
    port = int(os.environ[MASTER_PORT_ENV])
    output = output_directory()
    timeout_seconds = parse_positive_seconds(TIMEOUT_ENV, 600.0)
    report_path = _rank_report_path(output, rank)
    report: dict[str, Any] = {
        "schema": RANK_REPORT_SCHEMA,
        "run_id": run_id,
        "rank": rank,
        "world_size": WORLD_SIZE,
        "pid": os.getpid(),
        "status": "starting",
        "passed": False,
        "process_group_initialized": False,
        "teardown_completed": False,
        "operation_in_progress": None,
        "operations": [],
        "error": None,
    }
    write_json_atomic(report_path, report)
    log(
        f"TP4_RCCL_CAPSULE_BEGIN run_id={run_id} world={WORLD_SIZE}",
        rank=rank,
    )

    initialized = False
    exit_code = 1
    try:
        import torch
        import torch.distributed as dist

        if torch.cuda.device_count() < WORLD_SIZE:
            raise RuntimeError(
                f"capsule requires {WORLD_SIZE} visible devices, "
                f"found {torch.cuda.device_count()}"
            )
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=WORLD_SIZE,
            init_method=f"tcp://{address}:{port}",
            timeout=timedelta(seconds=timeout_seconds),
        )
        initialized = True
        report["process_group_initialized"] = True
        report["status"] = "running"
        report["device_name"] = torch.cuda.get_device_name(rank)
        write_json_atomic(report_path, report)

        dtype_by_name = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        for ordinal, operation in enumerate(OPERATIONS):
            dtype = dtype_by_name[operation.dtype]
            source = torch.tensor(input_values(operation, rank), dtype=dtype).reshape(
                operation.shape
            )
            expected = torch.tensor(expected_values(operation), dtype=dtype).reshape(
                operation.shape
            )
            actual = source.to(device=f"cuda:{rank}")
            report["operation_in_progress"] = operation.name
            write_json_atomic(report_path, report)
            log(
                "TP4_RCCL_ALL_REDUCE_BEGIN "
                f"ordinal={ordinal} name={operation.name} "
                f"shape={list(operation.shape)} dtype={operation.dtype}",
                rank=rank,
            )

            started = time.monotonic()
            dist.all_reduce(actual, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(rank)
            elapsed = time.monotonic() - started
            observed = actual.cpu()
            equal = bool(torch.equal(observed, expected))
            mismatch_count = int(torch.count_nonzero(observed != expected).item())
            max_abs_error = float(
                (observed.float() - expected.float()).abs().max().item()
            )
            operation_result = {
                "name": operation.name,
                "ordinal": ordinal,
                "shape": list(operation.shape),
                "dtype": operation.dtype,
                "elapsed_seconds": elapsed,
                "input_sha256": _tensor_sha256(source),
                "expected_sha256": _tensor_sha256(expected),
                "observed_sha256": _tensor_sha256(observed),
                "mismatch_count": mismatch_count,
                "max_abs_error": max_abs_error,
                "passed": equal,
            }
            report["operations"].append(operation_result)
            report["operation_in_progress"] = None
            write_json_atomic(report_path, report)
            log(
                "TP4_RCCL_ALL_REDUCE_END "
                f"ordinal={ordinal} name={operation.name} "
                f"status={'PASS' if equal else 'FAIL'} "
                f"mismatch_count={mismatch_count} max_abs_error={max_abs_error:.1f}",
                rank=rank,
            )
            if not equal:
                raise AssertionError(
                    f"{operation.name} exact oracle failed on rank {rank}: "
                    f"mismatches={mismatch_count} max_abs_error={max_abs_error}"
                )

        # Optional TP4 all-gather probe. It is deliberately after the
        # all-reduce sequence so the default capsule remains unchanged.
        if os.environ.get(ALL_GATHER_ENV) == "1":
            gather_shape = (
                (1, 248320 // WORLD_SIZE)
                if os.environ.get(ALL_GATHER_LM_HEAD_ENV) == "1"
                else (OBSERVED_PREFILL_TOKENS, HIDDEN_SIZE)
            )
            source = torch.arange(
                math.prod(gather_shape),
                dtype=torch.int32,
                device=f"cuda:{rank}",
            ).reshape(gather_shape)
            source = (source + rank * 100000).to(torch.bfloat16)
            gathered = torch.empty(
                (WORLD_SIZE * gather_shape[0], gather_shape[1]),
                dtype=torch.bfloat16,
                device=f"cuda:{rank}",
            )
            log(
                "TP4_RCCL_ALL_GATHER_BEGIN "
                f"shape={list(source.shape)} dtype=bfloat16",
                rank=rank,
            )
            dist.all_gather_into_tensor(gathered, source)
            torch.cuda.synchronize(rank)
            expected_gathered = torch.cat(
                [
                    (
                        torch.arange(
                            math.prod(gather_shape),
                            dtype=torch.int32,
                        ).reshape(gather_shape)
                        + peer * 100000
                    ).to(torch.bfloat16)
                    for peer in range(WORLD_SIZE)
                ],
                dim=0,
            )
            observed_gathered = gathered.cpu()
            gather_equal = bool(torch.equal(observed_gathered, expected_gathered))
            report["all_gather"] = {
                "shape": list(source.shape),
                "mode": "lm_head_vocab_shard"
                if len(gather_shape) == 2 and gather_shape[1] == 248320 // WORLD_SIZE
                else "embedding_hidden",
                "world_size": WORLD_SIZE,
                "observed_sha256": _tensor_sha256(observed_gathered),
                "expected_sha256": _tensor_sha256(expected_gathered),
                "mismatch_count": int(
                    torch.count_nonzero(observed_gathered != expected_gathered).item()
                ),
                "passed": gather_equal,
            }
            write_json_atomic(report_path, report)
            log(
                "TP4_RCCL_ALL_GATHER_END "
                f"status={'PASS' if gather_equal else 'FAIL'} "
                f"mismatch_count={report['all_gather']['mismatch_count']}",
                rank=rank,
            )
            if not gather_equal:
                raise AssertionError(
                    f"all_gather exact oracle failed on rank {rank}: "
                    f"mismatches={report['all_gather']['mismatch_count']}"
                )

        # Optional cross-rank allocator/GEMM probe. It is deliberately after
        # the collective sequence so the default capsule remains unchanged.
        pressure_gib = parse_nonnegative_gib()
        if pressure_gib:
            import torch

            pressure = []
            remaining = int(pressure_gib * (1024**3))
            while remaining:
                size = min(256 * 1024 * 1024, remaining)
                pressure.append(
                    torch.empty(size // 2, dtype=torch.bfloat16, device=f"cuda:{rank}")
                )
                remaining -= size
            x = torch.randn((2, 4096), dtype=torch.bfloat16, device=f"cuda:{rank}")
            weight = torch.randn((3072, 4096), dtype=torch.bfloat16, device=f"cuda:{rank}")
            log(f"TP4_GEMM_PRESSURE_BEGIN gib={pressure_gib}", rank=rank)
            from aiter.tuned_gemm import tgemm

            output = tgemm.mm(x, weight, otype=torch.bfloat16)
            torch.cuda.synchronize(rank)
            report["gemm_pressure"] = {
                "gib": pressure_gib,
                "allocated": int(torch.cuda.memory_allocated(rank)),
                "output_sha256": _tensor_sha256(output),
                "passed": True,
            }
            del output, weight, x, pressure
            write_json_atomic(report_path, report)
            log("TP4_GEMM_PRESSURE_END status=PASS", rank=rank)

        # Publish operation success before communicator destruction.  If
        # teardown itself wedges, the supervisor still has the exact last
        # completed operation and will fail the run on its bounded deadline.
        report["status"] = "operations_passed"
        write_json_atomic(report_path, report)
        dist.destroy_process_group()
        initialized = False
        report["teardown_completed"] = True
        report["status"] = "success"
        report["passed"] = True
        exit_code = 0
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        write_json_atomic(report_path, report)
        traceback.print_exc()
    finally:
        if initialized:
            try:
                # No extra collective is introduced during teardown.
                import torch.distributed as dist

                dist.destroy_process_group()
                report["teardown_completed"] = True
            except BaseException as error:
                report["teardown_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                exit_code = 1
                report["passed"] = False
                report["status"] = "failed"
        report["finished_unix_ns"] = time.time_ns()
        write_json_atomic(report_path, report)
        log(
            "TP4_RCCL_CAPSULE_END "
            f"status={'PASS' if exit_code == 0 else 'FAIL'} "
            f"report={report_path}",
            rank=rank,
        )
    return exit_code


def spawn_workers(
    *,
    run_id: str,
    output: Path,
    address: str,
    port: int,
    grace_seconds: float,
    process_factory: Any = subprocess.Popen,
) -> list[RankProcess]:
    runs: list[RankProcess] = []
    command = [sys.executable, str(Path(__file__).resolve())]
    try:
        for rank in range(WORLD_SIZE):
            environment = worker_environment(
                os.environ,
                rank=rank,
                run_id=run_id,
                output=output,
                address=address,
                port=port,
            )
            process = process_factory(
                command,
                env=environment,
                close_fds=True,
                start_new_session=True,
            )
            runs.append(RankProcess(rank=rank, process=process))
            log(f"spawned worker rank={rank} pid={process.pid}")
    except BaseException:
        # A partial spawn is still an owned distributed job.  Reap it here:
        # assignment in the caller has not completed, so the caller cannot
        # otherwise know which ranks already exist.
        _signal_live_workers(runs, signal.SIGTERM)
        _reap_workers(runs, grace_seconds)
        raise
    return runs


def _signal_live_workers(runs: list[RankProcess], signal_number: int) -> None:
    for run in runs:
        if run.process.poll() is not None:
            continue
        try:
            os.killpg(run.process.pid, signal_number)
        except ProcessLookupError:
            pass


def _reap_workers(runs: list[RankProcess], grace_seconds: float) -> None:
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if all(run.process.poll() is not None for run in runs):
            break
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    _signal_live_workers(runs, signal.SIGKILL)
    kill_deadline = time.monotonic() + grace_seconds
    for run in runs:
        if run.process.poll() is not None:
            continue
        remaining = max(0.001, kill_deadline - time.monotonic())
        try:
            run.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            log(f"worker rank={run.rank} resisted SIGKILL", rank="supervisor")


def wait_for_workers(
    runs: list[RankProcess],
    *,
    timeout_seconds: float,
    grace_seconds: float,
) -> tuple[bool, str | None]:
    deadline = time.monotonic() + timeout_seconds
    failure: str | None = None
    timed_out = False
    while True:
        returncodes = [run.process.poll() for run in runs]
        failed = next(
            (
                (run.rank, code)
                for run, code in zip(runs, returncodes, strict=True)
                if code is not None and code != 0
            ),
            None,
        )
        if failed is not None:
            failure = f"rank {failed[0]} exited with status {failed[1]}"
            break
        if all(code is not None for code in returncodes):
            return False, None
        if time.monotonic() >= deadline:
            timed_out = True
            live = [
                run.rank for run, code in zip(runs, returncodes, strict=True)
                if code is None
            ]
            failure = f"capsule deadline expired; live ranks={live}"
            break
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    log(f"{failure}; terminating owned worker process groups")
    _signal_live_workers(runs, signal.SIGTERM)
    _reap_workers(runs, grace_seconds)
    return timed_out, failure


def _load_rank_report(
    path: Path, *, run_id: str, rank: int
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"could not read rank report: {error}"
    if value.get("schema") != RANK_REPORT_SCHEMA:
        return value, "rank report schema mismatch"
    if value.get("run_id") != run_id:
        return value, "rank report is stale (run_id mismatch)"
    if value.get("rank") != rank:
        return value, "rank report rank mismatch"
    if value.get("world_size") != WORLD_SIZE:
        return value, "rank report world-size mismatch"
    if value.get("process_group_initialized") is not True:
        return value, "rank report lacks initialized process group"
    if value.get("status") != "success" or value.get("passed") is not True:
        return value, f"rank report status is {value.get('status')!r}"
    if value.get("teardown_completed") is not True:
        return value, "rank report lacks completed teardown"
    operations = value.get("operations")
    if (
        not isinstance(operations, list)
        or len(operations) != len(OPERATIONS)
        or not all(isinstance(item, dict) for item in operations)
    ):
        return value, "rank report operation sequence mismatch"
    for ordinal, (item, operation) in enumerate(
        zip(operations, OPERATIONS, strict=True)
    ):
        expected_contract = {
            "name": operation.name,
            "ordinal": ordinal,
            "shape": list(operation.shape),
            "dtype": operation.dtype,
            "passed": True,
            "mismatch_count": 0,
            "max_abs_error": 0.0,
        }
        if any(
            item.get(key) != expected
            for key, expected in expected_contract.items()
        ):
            return value, f"rank report operation {ordinal} failed validation"
        digests = [
            item.get("input_sha256"),
            item.get("expected_sha256"),
            item.get("observed_sha256"),
        ]
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in string.hexdigits for character in digest)
            for digest in digests
        ):
            return value, f"rank report operation {ordinal} has invalid digest"
        if item["observed_sha256"] != item["expected_sha256"]:
            return value, f"rank report operation {ordinal} digest mismatch"
        elapsed = item.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            return value, f"rank report operation {ordinal} has invalid elapsed time"
    return value, None


def collect_rank_processes(
    runs: list[RankProcess], *, output: Path, run_id: str
) -> tuple[list[dict[str, Any]], bool]:
    rank_processes: list[dict[str, Any]] = []
    reports_valid = True
    for run in runs:
        rank_path = _rank_report_path(output, run.rank)
        rank_report, validation_error = _load_rank_report(
            rank_path, run_id=run_id, rank=run.rank
        )
        if validation_error is not None:
            reports_valid = False
        rank_processes.append(
            {
                "rank": run.rank,
                "pid": run.process.pid,
                "returncode": run.process.poll(),
                "report_path": str(rank_path),
                "report": rank_report,
                "validation_error": validation_error,
            }
        )
    return rank_processes, reports_valid


def run_supervisor(
    *,
    process_factory: Any = subprocess.Popen,
    port_factory: Any = find_free_port,
) -> int:
    output = output_directory()
    timeout_seconds = parse_positive_seconds(TIMEOUT_ENV, 600.0)
    grace_seconds = parse_positive_seconds(TERM_GRACE_ENV, 5.0)
    run_id = f"{time.time_ns()}-{os.getpid()}"
    address = "127.0.0.1"
    port = port_factory()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "run_id": run_id,
        "status": "running",
        "passed": False,
        "world_size": WORLD_SIZE,
        "backend": "nccl",
        "implementation": "RCCL through PyTorch's nccl backend",
        "model_loaded": False,
        "weights_loaded": False,
        "master_addr": address,
        "master_port": port,
        "timeout_seconds": timeout_seconds,
        "terminate_grace_seconds": grace_seconds,
        "operation_contract": operation_contract(),
        "production_shape_evidence": PRODUCTION_SHAPE_EVIDENCE,
        "communication_environment": communication_environment(),
        "output_directory": str(output),
        "rank_processes": [],
        "failure": None,
        "timed_out": False,
        "termination_signal": None,
        "started_unix_ns": time.time_ns(),
    }
    write_json_atomic(report_path, report)
    log(
        "TP4_RCCL_SUPERVISOR_BEGIN "
        f"run_id={run_id} output={output} timeout_seconds={timeout_seconds}"
    )

    previous_signal_handlers = install_supervisor_signal_handlers()
    runs: list[RankProcess] = []
    try:
        runs = spawn_workers(
            run_id=run_id,
            output=output,
            address=address,
            port=port,
            grace_seconds=grace_seconds,
            process_factory=process_factory,
        )
        timed_out, failure = wait_for_workers(
            runs,
            timeout_seconds=timeout_seconds,
            grace_seconds=grace_seconds,
        )
        rank_processes, reports_valid = collect_rank_processes(
            runs, output=output, run_id=run_id
        )
        returncodes_ok = all(
            item["returncode"] == 0 for item in rank_processes
        )
        if failure is None and not returncodes_ok:
            failure = "one or more ranks exited unsuccessfully"
        if failure is None and not reports_valid:
            failure = "one or more rank reports failed validation"
        passed = not timed_out and failure is None and returncodes_ok and reports_valid
        report.update(
            {
                "status": "success" if passed else "failed",
                "passed": passed,
                "timed_out": timed_out,
                "failure": failure,
                "rank_processes": rank_processes,
            }
        )
    except BaseException as error:
        if runs:
            _signal_live_workers(runs, signal.SIGTERM)
            _reap_workers(runs, grace_seconds)
            rank_processes, _ = collect_rank_processes(
                runs, output=output, run_id=run_id
            )
            report["rank_processes"] = rank_processes
        termination_signal = (
            signal.Signals(error.signal_number).name
            if isinstance(error, SupervisorTermination)
            else None
        )
        report.update(
            {
                "status": "failed",
                "passed": False,
                "failure": f"{type(error).__name__}: {error}",
                "termination_signal": termination_signal,
            }
        )
        if not isinstance(error, SupervisorTermination):
            traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            report["finished_unix_ns"] = time.time_ns()
            write_json_atomic(report_path, report)
            log(
                "TP4_RCCL_SUPERVISOR_END "
                f"status={'PASS' if report['passed'] else 'FAIL'} "
                f"report={report_path}"
            )
        finally:
            restore_signal_handlers(previous_signal_handlers)
    return 0 if report["passed"] else 1


def main() -> int:
    worker_rank = os.environ.get(WORKER_RANK_ENV)
    if worker_rank is None:
        return run_supervisor()
    try:
        rank = int(worker_rank)
    except ValueError as error:
        raise SystemExit(f"invalid {WORKER_RANK_ENV}: {worker_rank!r}") from error
    if rank not in range(WORLD_SIZE):
        raise SystemExit(f"invalid {WORKER_RANK_ENV}: {rank}")
    supervisor_pid = os.environ.get(SUPERVISOR_PID_ENV)
    if supervisor_pid is None:
        raise SystemExit(f"missing {SUPERVISOR_PID_ENV}")
    try:
        expected_parent_pid = int(supervisor_pid)
    except ValueError as error:
        raise SystemExit(
            f"invalid {SUPERVISOR_PID_ENV}: {supervisor_pid!r}"
        ) from error
    arm_parent_death_signal(expected_parent_pid)
    return run_worker(rank)


if __name__ == "__main__":
    raise SystemExit(main())
