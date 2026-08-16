"""Reusable synchronous SUM allreduce engine for the live CCL v1 transport."""

from __future__ import annotations

import dataclasses
from enum import Enum
import hashlib
import os
import threading
import time
from typing import Any, Callable, Final, Sequence

from .native import (
    CANCELLED,
    DTYPE_BF16,
    DTYPE_FP32,
    LIVE_PHASE_READY,
    MAX_WORLD_SIZE,
    MESSAGE_CONSUMED,
    NO_RANK,
    NOT_SUPPORTED,
    OP_ALL_REDUCE,
    OUT_OF_ORDER,
    OUT_OF_RESOURCES,
    PEER_LOST,
    PHASE_ALL_GATHER,
    PHASE_REDUCE_SCATTER,
    PROTOCOL_ERROR,
    REDUCTION_SUM,
    SEQUENCE_MISMATCH,
    TIMED_OUT,
    TOPOLOGY_MISMATCH,
    UINT64_MAX,
    VERSION_MISMATCH,
    IDENTITY_MISMATCH,
    CHECKSUM_ERROR,
    NativeCCL,
    PlannedStep,
)


CARRIER_MAX_PAYLOAD_BYTES: Final = 16 * 1024 * 1024
MANAGED_MAX_SINGLE_ALLOCATION_BYTES: Final = 2 * 1024 * 1024 * 1024
DEVICE_MAX_ELEMENT_COUNT: Final = (1 << 31) - 1
DEFAULT_COLLECTIVE_TIMEOUT_NS: Final = 300_000_000_000
DEFAULT_CLOSE_TIMEOUT_NS: Final = 5_000_000_000
DEFAULT_CREDITS_PER_PEER: Final = 2

_ACTION_SEND_RECEIVE: Final = 3
_MAX_CREDITS_PER_PEER: Final = 16
_VALID_ABORT_STATUSES: Final = frozenset(
    {
        VERSION_MISMATCH,
        TOPOLOGY_MISMATCH,
        IDENTITY_MISMATCH,
        SEQUENCE_MISMATCH,
        NOT_SUPPORTED,
        PROTOCOL_ERROR,
        CHECKSUM_ERROR,
        OUT_OF_ORDER,
        TIMED_OUT,
        PEER_LOST,
        CANCELLED,
        OUT_OF_RESOURCES,
    }
)


class EngineState(str, Enum):
    READY = "ready"
    ACTIVE = "active"
    CLOSING = "closing"
    ABORTED = "aborted"
    CLOSED = "closed"


class EngineError(RuntimeError):
    """Base error for the reusable collective engine."""


class EngineStateError(EngineError):
    """The requested operation is illegal in the current engine state."""


class EngineForkError(EngineStateError):
    """A process tried to reuse an engine created before fork."""


class CollectiveTimeoutError(EngineError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status = TIMED_OUT


class GroupAbortedError(EngineError):
    def __init__(self, first_error: Any) -> None:
        status = int(first_error.status)
        failed_rank = int(first_error.failed_rank)
        super().__init__(
            f"collective observed group abort status={status} "
            f"failed_rank={failed_rank}"
        )
        self.status = status
        self.failed_rank = failed_rank
        self.first_error = first_error


class SequenceExhaustedError(EngineError):
    """The live epoch has no valid descriptor sequence left."""

    status = SEQUENCE_MISMATCH


def _exact_integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _exact_bytes(value: Any, name: str, size: int) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} bytes")
    if not any(value):
        raise ValueError(f"{name} must not be all zero")
    return value


@dataclasses.dataclass(frozen=True)
class GroupSpec:
    """Exact identity shared by every rank in one live CCL epoch."""

    world_size: int
    epoch: int
    group_generation: int
    job_uuid: bytes
    group_uuid: bytes
    model_identity_sha256: bytes

    def __post_init__(self) -> None:
        _exact_integer(self.world_size, "world_size", 2, MAX_WORLD_SIZE)
        _exact_integer(self.epoch, "epoch", 1, UINT64_MAX)
        _exact_integer(
            self.group_generation, "group_generation", 1, UINT64_MAX
        )
        _exact_bytes(self.job_uuid, "job_uuid", 16)
        _exact_bytes(self.group_uuid, "group_uuid", 16)
        _exact_bytes(
            self.model_identity_sha256, "model_identity_sha256", 32
        )


@dataclasses.dataclass(frozen=True)
class RankBootstrap:
    """One rank's inherited broker capability and exact broker identity."""

    rank: int
    capability_fd: int
    broker_pid: int
    broker_start_time_ticks: int
    absolute_deadline_ns: int
    credits_per_peer: int = DEFAULT_CREDITS_PER_PEER

    def __post_init__(self) -> None:
        _exact_integer(self.rank, "rank", 0, MAX_WORLD_SIZE - 1)
        _exact_integer(self.capability_fd, "capability_fd", 0, (1 << 31) - 1)
        _exact_integer(self.broker_pid, "broker_pid", 1, (1 << 31) - 1)
        _exact_integer(
            self.broker_start_time_ticks,
            "broker_start_time_ticks",
            1,
            UINT64_MAX,
        )
        _exact_integer(
            self.absolute_deadline_ns,
            "absolute_deadline_ns",
            1,
            UINT64_MAX,
        )
        _exact_integer(
            self.credits_per_peer,
            "credits_per_peer",
            1,
            _MAX_CREDITS_PER_PEER,
        )


@dataclasses.dataclass(frozen=True)
class AllReduceSegment:
    index: int
    sequence: int
    offset_elements: int
    element_count: int
    byte_count: int


@dataclasses.dataclass(frozen=True)
class TransferInfo:
    descriptor_sha256: str
    sequence: int
    kind: int
    phase: int
    step_index: int
    chunk_index: int
    source_rank: int
    destination_rank: int
    slot_index: int
    slot_generation: int
    payload_bytes: int
    status: int
    failed_rank: int


@dataclasses.dataclass(frozen=True)
class CollectiveEvent:
    """Immutable observation emitted after a protocol transition completes."""

    name: str
    monotonic_ns: int
    rank: int
    world_size: int
    segment: AllReduceSegment | None = None
    step_ordinal: int | None = None
    step: PlannedStep | None = None
    transfer: TransferInfo | None = None
    descriptor_sha256: str | None = None
    payload_sha256: str | None = None
    byte_count: int | None = None
    status: int | None = None
    failed_rank: int | None = None


Observer = Callable[[CollectiveEvent], None]


def _dtype_spec(dtype: Any) -> tuple[str, int, int]:
    if isinstance(dtype, str):
        name = dtype
    else:
        name = str(dtype)
        if name.startswith("torch."):
            name = name[6:]
    if name == "bfloat16":
        return name, DTYPE_BF16, 2
    if name == "float32":
        return name, DTYPE_FP32, 4
    raise TypeError("dtype must be bfloat16 or float32")


def plan_allreduce_segments(
    element_count: int,
    world_size: int,
    dtype: Any,
    *,
    first_sequence: int = 1,
) -> tuple[AllReduceSegment, ...]:
    """Split one allreduce into descriptor-sized contiguous private segments."""

    element_count = _exact_integer(
        element_count, "element_count", 1, (1 << 63) - 1
    )
    _exact_integer(world_size, "world_size", 2, MAX_WORLD_SIZE)
    first_sequence = _exact_integer(
        first_sequence, "first_sequence", 1, UINT64_MAX
    )
    _, _, dtype_bytes = _dtype_spec(dtype)
    if element_count > MANAGED_MAX_SINGLE_ALLOCATION_BYTES // dtype_bytes:
        raise ValueError("public tensor exceeds the managed allocation limit")
    segment_limit = min(
        CARRIER_MAX_PAYLOAD_BYTES // dtype_bytes,
        MANAGED_MAX_SINGLE_ALLOCATION_BYTES // dtype_bytes,
        DEVICE_MAX_ELEMENT_COUNT,
    )
    segment_count = (element_count + segment_limit - 1) // segment_limit
    if first_sequence + segment_count - 1 >= UINT64_MAX:
        raise SequenceExhaustedError(
            "collective descriptor sequence is exhausted"
        )

    result = []
    offset = 0
    for index in range(segment_count):
        count = min(segment_limit, element_count - offset)
        result.append(
            AllReduceSegment(
                index=index,
                sequence=first_sequence + index,
                offset_elements=offset,
                element_count=count,
                byte_count=count * dtype_bytes,
            )
        )
        offset += count
    return tuple(result)


def _tensor_bytes(tensor: Any) -> bytes:
    torch = __import__("torch")
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def _tensor_from_bytes(payload: bytes, dtype: Any, count: int) -> Any:
    torch = __import__("torch")
    if len(payload) != count * dtype.itemsize:
        raise EngineError("carrier staging extent differs from the planner")
    if count == 0:
        return torch.empty(0, dtype=dtype)
    return torch.frombuffer(bytearray(payload), dtype=dtype, count=count).clone()


def _transfer_info(record: Any) -> TransferInfo:
    return TransferInfo(
        descriptor_sha256=bytes(record.descriptor_sha256).hex(),
        sequence=int(record.sequence),
        kind=int(record.kind),
        phase=int(record.phase),
        step_index=int(record.step_index),
        chunk_index=int(record.chunk_index),
        source_rank=int(record.source_rank),
        destination_rank=int(record.destination_rank),
        slot_index=int(record.slot_index),
        slot_generation=int(record.slot_generation),
        payload_bytes=int(record.payload_bytes),
        status=int(record.status),
        failed_rank=int(record.failed_rank),
    )


def _status_for_error(error: BaseException) -> int:
    status = getattr(error, "status", PROTOCOL_ERROR)
    if type(status) is not int or status not in _VALID_ABORT_STATUSES:
        return PROTOCOL_ERROR
    return status


class AllReduceEngine:
    """Caller-serialized, fork-bound live SUM allreduce engine.

    ``join`` takes ownership of the capability descriptor at entry. It closes
    the descriptor on failures before calling ``NativeCCL.join_rank`` and
    transfers ownership to that call on entry, including its failure paths.
    The engine owns the resulting live rank and carrier until graceful close,
    abort, or destroy. A failed collective permanently poisons the engine; no
    sequence or live epoch is reused.
    """

    def __init__(
        self,
        *,
        group: GroupSpec | None,
        rank: int,
        native: Any | None,
        identity: Any | None,
        live_rank: Any | None,
        carrier: Any | None,
        executor: Any | None,
        observer: Observer | None,
        singleton: bool,
    ) -> None:
        self._group = group
        self._rank = rank
        self._native = native
        self._identity = identity
        self._live_rank = live_rank
        self._carrier = carrier
        self._executor = executor
        self._observer = observer
        self._singleton = singleton
        self._owner_pid = os.getpid()
        self._state = EngineState.READY
        self._next_sequence = 1
        self._last_error: BaseException | None = None
        self._active_descriptor: Any | None = None
        self._active_sequence = 0
        self._active_step_ordinal: int | None = None
        self._active_transfer: Any | None = None
        self._call_lock = threading.Lock()

    @classmethod
    def join(
        cls,
        group: GroupSpec,
        bootstrap: RankBootstrap,
        *,
        native: Any | None = None,
        executor: Any | None = None,
        observer: Observer | None = None,
    ) -> "AllReduceEngine":
        capability_fd = (
            bootstrap.capability_fd
            if isinstance(bootstrap, RankBootstrap)
            else None
        )
        owns_capability = capability_fd is not None
        live_rank = None
        carrier = None
        try:
            if not isinstance(group, GroupSpec):
                raise TypeError("group must be GroupSpec")
            if not isinstance(bootstrap, RankBootstrap):
                raise TypeError("bootstrap must be RankBootstrap")
            if bootstrap.rank >= group.world_size:
                raise ValueError("rank is outside the group")
            if observer is not None and not callable(observer):
                raise TypeError("observer must be callable or None")

            runtime = NativeCCL() if native is None else native
            identity = runtime.identity(
                world_size=group.world_size,
                epoch=group.epoch,
                group_generation=group.group_generation,
                job_uuid=group.job_uuid,
                group_uuid=group.group_uuid,
                model_identity_sha256=group.model_identity_sha256,
            )
            owner = runtime.process_identity(bootstrap.broker_pid)
            if (
                int(owner.pid) != bootstrap.broker_pid
                or int(owner.start_time_ticks)
                != bootstrap.broker_start_time_ticks
            ):
                raise EngineError("broker PID/start-time identity mismatch")

            # NativeCCL owns the descriptor on every path once join_rank starts.
            # Clear Python ownership before the call to avoid a double-close.
            owns_capability = False
            live_rank = runtime.join_rank(
                bootstrap.capability_fd,
                identity,
                bootstrap.rank,
                owner,
                bootstrap.absolute_deadline_ns,
            )
            cls._validate_joined_rank(live_rank, group, bootstrap.rank)
            carrier = runtime.carrier_session(
                identity, bootstrap.rank, bootstrap.credits_per_peer
            )
            if executor is None:
                from .device import DeviceSumExecutor

                executor = DeviceSumExecutor()
            return cls(
                group=group,
                rank=bootstrap.rank,
                native=runtime,
                identity=identity,
                live_rank=live_rank,
                carrier=carrier,
                executor=executor,
                observer=observer,
                singleton=False,
            )
        except BaseException as error:
            if owns_capability and capability_fd is not None:
                try:
                    os.close(capability_fd)
                except OSError:
                    pass
            if live_rank is not None:
                try:
                    live_rank.report_abort(
                        bootstrap.rank, _status_for_error(error), 0
                    )
                except Exception:
                    pass
            if carrier is not None:
                try:
                    carrier.close()
                except Exception:
                    pass
            if live_rank is not None:
                try:
                    live_rank.destroy()
                except Exception:
                    pass
            raise

    @classmethod
    def singleton(cls, *, observer: Observer | None = None) -> "AllReduceEngine":
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable or None")
        return cls(
            group=None,
            rank=0,
            native=None,
            identity=None,
            live_rank=None,
            carrier=None,
            executor=None,
            observer=observer,
            singleton=True,
        )

    @staticmethod
    def _validate_joined_rank(live_rank: Any, group: GroupSpec, rank: int) -> None:
        info = live_rank.info()
        identity = info.group
        if (
            int(info.phase) != LIVE_PHASE_READY
            or int(info.self_rank) != rank
            or int(info.world_size) != group.world_size
            or int(identity.world_size) != group.world_size
            or int(identity.epoch) != group.epoch
            or int(identity.group_generation) != group.group_generation
            or bytes(identity.job_uuid) != group.job_uuid
            or bytes(identity.group_uuid) != group.group_uuid
            or bytes(identity.model_identity_sha256)
            != group.model_identity_sha256
        ):
            raise EngineError("joined rank identity/topology mismatch")

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return 1 if self._singleton else self._group.world_size  # type: ignore[union-attr]

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    def _check_process(self) -> None:
        observed = os.getpid()
        if observed != self._owner_pid:
            error = EngineForkError(
                f"engine belongs to PID {self._owner_pid}, not forked PID {observed}"
            )
            self._last_error = error
            self._state = EngineState.ABORTED
            raise error

    def _acquire(self, operation: str) -> None:
        self._check_process()
        if not self._call_lock.acquire(blocking=False):
            raise EngineStateError(
                f"{operation} rejected: engine operations are caller-serialized"
            )
        try:
            self._check_process()
        except BaseException:
            self._call_lock.release()
            raise

    def _require_ready(self, operation: str) -> None:
        if self._state is not EngineState.READY:
            raise EngineStateError(
                f"{operation} requires ready state, got {self._state.value}"
            )

    def _emit(
        self,
        name: str,
        *,
        segment: AllReduceSegment | None = None,
        step_ordinal: int | None = None,
        step: PlannedStep | None = None,
        record: Any | None = None,
        descriptor_sha256: str | None = None,
        payload_sha256: str | None = None,
        byte_count: int | None = None,
        status: int | None = None,
        failed_rank: int | None = None,
    ) -> None:
        if self._observer is None:
            return
        self._observer(
            CollectiveEvent(
                name=name,
                monotonic_ns=time.monotonic_ns(),
                rank=self.rank,
                world_size=self.world_size,
                segment=segment,
                step_ordinal=step_ordinal,
                step=step,
                transfer=None if record is None else _transfer_info(record),
                descriptor_sha256=descriptor_sha256,
                payload_sha256=payload_sha256,
                byte_count=byte_count,
                status=status,
                failed_rank=failed_rank,
            )
        )

    def _emit_safely(self, name: str, **fields: Any) -> None:
        try:
            self._emit(name, **fields)
        except BaseException:
            pass

    def _deadline(self, timeout_ns: int | None, default: int) -> int:
        timeout = default if timeout_ns is None else timeout_ns
        timeout = _exact_integer(timeout, "timeout_ns", 1, UINT64_MAX)
        return self._native.deadline_after(timeout)

    def _wait_busy(
        self,
        action: Callable[[], Any],
        *,
        absolute_deadline_ns: int,
        label: str,
        complete: Callable[[Any], bool],
    ) -> Any:
        while True:
            value = action()
            if complete(value):
                return value
            first_error = self._live_rank.poll_abort()
            if first_error is not None:
                raise GroupAbortedError(first_error)
            if self._native.monotonic_time_ns() >= absolute_deadline_ns:
                raise CollectiveTimeoutError(
                    f"{label} exceeded the absolute collective deadline"
                )
            time.sleep(0.001)

    def _validate_tensor(self, input_: Any) -> tuple[Any, int, int]:
        torch = __import__("torch")
        if not isinstance(input_, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        if input_.device.type != "cpu":
            raise ValueError("input must use the gemsim CPU staging device")
        if input_.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("input must use bfloat16 or float32")
        if not input_.is_contiguous():
            raise ValueError("input must be contiguous")
        count = input_.numel()
        if not self._singleton and count == 0:
            raise ValueError("live allreduce requires at least one element")
        if count * input_.element_size() > MANAGED_MAX_SINGLE_ALLOCATION_BYTES:
            raise ValueError("input exceeds the managed allocation limit")
        _, dtype_code, dtype_bytes = _dtype_spec(input_.dtype)
        return input_.reshape(-1), dtype_code, dtype_bytes

    def _validate_plan(
        self, plan: Sequence[PlannedStep], segment: AllReduceSegment
    ) -> None:
        expected_steps = 2 * (self.world_size - 1)
        if len(plan) != expected_steps:
            raise EngineError("allreduce planner returned an invalid step count")
        for ordinal, step in enumerate(plan):
            expected_phase = (
                PHASE_REDUCE_SCATTER
                if ordinal < self.world_size - 1
                else PHASE_ALL_GATHER
            )
            if int(step.phase) != expected_phase:
                raise EngineError("allreduce planner returned an invalid phase order")
            if int(step.action) != _ACTION_SEND_RECEIVE:
                raise EngineError("allreduce planner returned an invalid action")
            for peer in (int(step.send_rank), int(step.receive_rank)):
                if peer < 0 or peer >= self.world_size or peer == self.rank:
                    raise EngineError("allreduce planner returned an invalid peer")
            for offset, count in (
                (int(step.send_offset_elements), int(step.send_count_elements)),
                (
                    int(step.receive_offset_elements),
                    int(step.receive_count_elements),
                ),
            ):
                if offset < 0 or count < 0 or offset + count > segment.element_count:
                    raise EngineError("allreduce planner escaped the segment workspace")
                if count * (segment.byte_count // segment.element_count) > (
                    CARRIER_MAX_PAYLOAD_BYTES
                ):
                    raise EngineError("allreduce planner exceeded the carrier limit")

    def _execute_segment(
        self,
        flat_input: Any,
        dtype_code: int,
        segment: AllReduceSegment,
        absolute_deadline_ns: int,
    ) -> Any:
        torch = __import__("torch")
        workspace = flat_input.narrow(
            0, segment.offset_elements, segment.element_count
        ).clone()
        descriptor = self._native.descriptor(
            self._identity,
            sequence=segment.sequence,
            input_count=segment.element_count,
            output_count=segment.element_count,
            rank=self.rank,
            operation=OP_ALL_REDUCE,
            reduction=REDUCTION_SUM,
            dtype=dtype_code,
        )
        self._active_descriptor = descriptor
        self._active_sequence = segment.sequence
        plan = self._native.plan(descriptor)
        self._validate_plan(plan, segment)
        descriptor_sha256 = self._native.descriptor_sha256(descriptor)
        self._emit(
            "segment_started",
            segment=segment,
            descriptor_sha256=descriptor_sha256,
        )

        for step_ordinal, step in enumerate(plan):
            self._active_step_ordinal = step_ordinal
            send_view = workspace.narrow(
                0,
                int(step.send_offset_elements),
                int(step.send_count_elements),
            )
            outbound = self._carrier.prepare_data(
                descriptor, step_ordinal, _tensor_bytes(send_view)
            )
            self._active_transfer = outbound
            self._emit(
                "outbound_prepared",
                segment=segment,
                step_ordinal=step_ordinal,
                step=step,
                record=outbound,
                descriptor_sha256=descriptor_sha256,
            )
            send_socket = self._live_rank.peer_socket(int(step.send_rank))
            self._wait_busy(
                lambda: self._carrier.send_data(send_socket, outbound),
                absolute_deadline_ns=absolute_deadline_ns,
                label="outbound DATA send",
                complete=bool,
            )
            self._emit(
                "outbound_DATA_sent",
                segment=segment,
                step_ordinal=step_ordinal,
                step=step,
                record=outbound,
            )

            receive_socket = self._live_rank.peer_socket(int(step.receive_rank))
            inbound = self._wait_busy(
                lambda: self._carrier.receive(
                    receive_socket,
                    descriptor,
                    step_ordinal,
                    int(step.receive_rank),
                ),
                absolute_deadline_ns=absolute_deadline_ns,
                label="inbound DATA receive",
                complete=lambda value: value is not None,
            )
            self._active_transfer = inbound
            self._emit(
                "inbound_DATA_received",
                segment=segment,
                step_ordinal=step_ordinal,
                step=step,
                record=inbound,
            )
            payload, inbound_consumed = self._carrier.consume(
                descriptor, step_ordinal, inbound
            )
            immutable = bytes(payload)
            receive_count = int(step.receive_count_elements)
            staging = _tensor_from_bytes(
                immutable, flat_input.dtype, receive_count
            )
            self._emit(
                "inbound_staged",
                segment=segment,
                step_ordinal=step_ordinal,
                step=step,
                record=inbound,
                payload_sha256=hashlib.sha256(immutable).hexdigest(),
                byte_count=len(immutable),
            )

            destination = workspace.narrow(
                0, int(step.receive_offset_elements), receive_count
            )
            if int(step.phase) == PHASE_REDUCE_SCATTER:
                if receive_count:
                    self._emit(
                        "device_call_enter",
                        segment=segment,
                        step_ordinal=step_ordinal,
                        step=step,
                    )
                    self._executor.sum_in_place(
                        destination, staging, element_count=receive_count
                    )
                    self._emit(
                        "device_call_returned",
                        segment=segment,
                        step_ordinal=step_ordinal,
                        step=step,
                    )
                else:
                    self._emit(
                        "zero_no_dispatch",
                        segment=segment,
                        step_ordinal=step_ordinal,
                        step=step,
                    )
            elif int(step.phase) == PHASE_ALL_GATHER:
                if receive_count:
                    destination.view(torch.uint8).copy_(staging.view(torch.uint8))
                self._emit(
                    "copy_complete",
                    segment=segment,
                    step_ordinal=step_ordinal,
                    step=step,
                    byte_count=len(immutable),
                )
            else:
                raise EngineError("planner returned a non-allreduce phase")

            self._emit(
                "inbound_CONSUMED_send_attempt",
                segment=segment,
                step_ordinal=step_ordinal,
                step=step,
                record=inbound_consumed,
            )
            self._wait_busy(
                lambda: self._carrier.send_consumed(
                    receive_socket, inbound_consumed
                ),
                absolute_deadline_ns=absolute_deadline_ns,
                label="inbound CONSUMED send",
                complete=bool,
            )
            self._emit(
                "inbound_CONSUMED_sent",
                segment=segment,
                step_ordinal=step_ordinal,
                step=step,
                record=inbound_consumed,
            )
            self._active_transfer = outbound
            acknowledged = self._wait_busy(
                lambda: self._carrier.receive(
                    send_socket,
                    descriptor,
                    step_ordinal,
                    int(step.send_rank),
                ),
                absolute_deadline_ns=absolute_deadline_ns,
                label="outbound CONSUMED receive",
                complete=lambda value: value is not None,
            )
            if int(acknowledged.kind) != MESSAGE_CONSUMED:
                raise EngineError(
                    "outbound DATA did not receive matching CONSUMED"
                )
            self._emit(
                "outbound_CONSUMED_received_credit_released",
                segment=segment,
                step_ordinal=step_ordinal,
                step=step,
                record=acknowledged,
            )
            self._active_transfer = None
            self._emit(
                "step_complete",
                segment=segment,
                step_ordinal=step_ordinal,
                step=step,
            )

        info = self._carrier.info()
        if any(
            int(value) != 0
            for value in (
                info.sender_inflight,
                info.receiver_ready,
                info.receiver_consumed,
            )
        ):
            raise EngineError("carrier retained ownership at segment completion")
        self._emit(
            "segment_complete",
            segment=segment,
            descriptor_sha256=descriptor_sha256,
        )
        return workspace

    def all_reduce(
        self, input_: Any, *, timeout_ns: int | None = None
    ) -> Any:
        self._acquire("all_reduce")
        active = False
        try:
            self._require_ready("all_reduce")
            self._state = EngineState.ACTIVE
            active = True
            self._active_descriptor = None
            self._active_sequence = 0
            self._active_step_ordinal = None
            self._active_transfer = None
            try:
                flat_input, dtype_code, _ = self._validate_tensor(input_)
                input_before = _tensor_bytes(input_)
                if self._singleton:
                    if timeout_ns is not None:
                        _exact_integer(timeout_ns, "timeout_ns", 1, UINT64_MAX)
                    self._emit("collective_started")
                    output = input_.clone()
                    self._emit(
                        "public_commit",
                        payload_sha256=hashlib.sha256(
                            _tensor_bytes(output)
                        ).hexdigest(),
                        byte_count=len(input_before),
                    )
                    self._active_sequence = 0
                    self._state = EngineState.READY
                    return output

                absolute_deadline_ns = self._deadline(
                    timeout_ns, DEFAULT_COLLECTIVE_TIMEOUT_NS
                )
                segments = plan_allreduce_segments(
                    flat_input.numel(),
                    self.world_size,
                    flat_input.dtype,
                    first_sequence=self._next_sequence,
                )
                self._next_sequence += len(segments)
                self._active_sequence = segments[0].sequence
                self._emit("collective_started", segment=segments[0])
                completed = [
                    self._execute_segment(
                        flat_input,
                        dtype_code,
                        segment,
                        absolute_deadline_ns,
                    )
                    for segment in segments
                ]
                if _tensor_bytes(input_) != input_before:
                    raise EngineError("public input changed before collective commit")
                torch = __import__("torch")
                output_flat = torch.cat(completed).clone()
                output = output_flat.view(input_.shape)
                if input_.numel() and output.data_ptr() == input_.data_ptr():
                    raise EngineError("public allreduce result aliases its input")
                output_bytes = _tensor_bytes(output)
                self._emit(
                    "public_commit",
                    payload_sha256=hashlib.sha256(output_bytes).hexdigest(),
                    byte_count=len(output_bytes),
                )
                self._active_descriptor = None
                self._active_sequence = 0
                self._active_step_ordinal = None
                self._active_transfer = None
                self._state = EngineState.READY
                return output
            except BaseException as error:
                self._fail_collective(error)
                raise
        finally:
            if active and self._state is EngineState.ACTIVE:
                self._state = EngineState.READY
            self._call_lock.release()

    def _send_abort_record(
        self, record: Any, absolute_deadline_ns: int
    ) -> None:
        for peer in range(self.world_size):
            if peer == self.rank:
                continue
            try:
                descriptor = self._live_rank.peer_socket(peer)
                while not self._carrier.send_abort(descriptor, record):
                    if self._native.monotonic_time_ns() >= absolute_deadline_ns:
                        break
                    time.sleep(0.001)
            except Exception:
                pass

    def _report_live_abort(
        self,
        status: int,
        failed_rank: int,
        sequence: int,
        absolute_deadline_ns: int,
    ) -> None:
        if self._live_rank is None:
            return
        try:
            while not self._live_rank.report_abort(
                failed_rank, status, sequence
            ):
                if self._native.monotonic_time_ns() >= absolute_deadline_ns:
                    return
                time.sleep(0.001)
        except Exception:
            pass

    def _fail_collective(self, error: BaseException) -> None:
        status = _status_for_error(error)
        failed_rank = getattr(error, "failed_rank", self.rank)
        if type(failed_rank) is not int or not 0 <= failed_rank < self.world_size:
            failed_rank = self.rank
        sequence = self._active_sequence
        self._last_error = error
        self._state = EngineState.ABORTED
        deadline = 0
        if self._native is not None:
            try:
                deadline = self._native.deadline_after(DEFAULT_CLOSE_TIMEOUT_NS)
            except Exception:
                pass
        candidate = getattr(error, "abort_record", None)
        abort_record = (
            candidate
            if candidate is not None
            and hasattr(candidate, "descriptor_sha256")
            and hasattr(candidate, "kind")
            else None
        )
        if abort_record is None and self._carrier is not None and (
            self._active_descriptor is not None
        ):
            try:
                abort_record = self._carrier.abort(
                    self._active_descriptor, failed_rank, status
                )
            except Exception:
                try:
                    abort_record = self._carrier.get_abort()
                except Exception:
                    abort_record = None
        if abort_record is not None:
            self._send_abort_record(abort_record, deadline)
        self._report_live_abort(status, failed_rank, sequence, deadline)
        self._emit_safely(
            "collective_failed",
            status=status,
            failed_rank=failed_rank,
        )
        self._release_resources()

    def abort(
        self,
        reason: int = CANCELLED,
        *,
        failed_rank: int | None = None,
        timeout_ns: int | None = None,
    ) -> None:
        self._acquire("abort")
        try:
            self._require_ready("abort")
            reason = _exact_integer(
                reason, "reason", -(1 << 31), (1 << 31) - 1
            )
            if reason not in _VALID_ABORT_STATUSES:
                raise ValueError("reason is not a valid group-abort status")
            target = self.rank if failed_rank is None else failed_rank
            target = _exact_integer(
                target, "failed_rank", 0, self.world_size - 1
            )
            self._state = EngineState.ABORTED
            error = EngineError(f"collective group aborted with status {reason}")
            error.status = reason  # type: ignore[attr-defined]
            error.failed_rank = target  # type: ignore[attr-defined]
            self._last_error = error
            if not self._singleton:
                deadline = self._deadline(timeout_ns, DEFAULT_CLOSE_TIMEOUT_NS)
                self._report_live_abort(reason, target, 0, deadline)
            self._emit_safely(
                "collective_failed", status=reason, failed_rank=target
            )
            self._release_resources()
        finally:
            self._call_lock.release()

    def close(self, *, timeout_ns: int | None = None) -> None:
        self._acquire("close")
        try:
            if self._state is EngineState.CLOSED:
                return
            if self._state is EngineState.ABORTED:
                self._release_resources()
                self._state = EngineState.CLOSED
                return
            self._require_ready("close")
            self._state = EngineState.CLOSING
            try:
                if self._live_rank is not None:
                    deadline = self._deadline(timeout_ns, DEFAULT_CLOSE_TIMEOUT_NS)
                    self._live_rank.close(deadline)
                    self._live_rank = None
                if self._carrier is not None:
                    self._carrier.close()
                    self._carrier = None
                self._state = EngineState.CLOSED
                self._emit_safely("closed")
            except BaseException as error:
                self._fail_collective(error)
                raise
        finally:
            self._call_lock.release()

    def destroy(self) -> None:
        self._acquire("destroy")
        try:
            self._release_resources()
            self._state = EngineState.CLOSED
        finally:
            self._call_lock.release()

    def _release_resources(self) -> None:
        carrier, self._carrier = self._carrier, None
        live_rank, self._live_rank = self._live_rank, None
        if carrier is not None:
            try:
                carrier.close()
            except Exception:
                pass
        if live_rank is not None:
            try:
                live_rank.destroy()
            except Exception:
                pass

    def __enter__(self) -> "AllReduceEngine":
        self._check_process()
        self._require_ready("context entry")
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        if _type is None:
            self.close()
        elif self._state is EngineState.READY:
            self.abort(PROTOCOL_ERROR)
        else:
            self.destroy()

    def __del__(self) -> None:
        try:
            self._release_resources()
        except Exception:
            pass


__all__ = [
    "AllReduceEngine",
    "AllReduceSegment",
    "CollectiveEvent",
    "CollectiveTimeoutError",
    "EngineError",
    "EngineForkError",
    "EngineState",
    "EngineStateError",
    "GroupAbortedError",
    "GroupSpec",
    "RankBootstrap",
    "SequenceExhaustedError",
    "TransferInfo",
    "plan_allreduce_segments",
]
