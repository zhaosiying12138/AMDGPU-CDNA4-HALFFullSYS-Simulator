"""Strict ctypes binding for the versioned CCL v1 runtime."""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import os
from pathlib import Path
from typing import Final


SUCCESS: Final = 0
INVALID_ARGUMENT: Final = 1
BUFFER_TOO_SMALL: Final = 2
VERSION_MISMATCH: Final = 3
TOPOLOGY_MISMATCH: Final = 4
IDENTITY_MISMATCH: Final = 5
SEQUENCE_MISMATCH: Final = 6
NOT_SUPPORTED: Final = 7
PROTOCOL_ERROR: Final = 8
CHECKSUM_ERROR: Final = 9
OUT_OF_ORDER: Final = 10
BUSY: Final = 11
ABORTED: Final = 12
TIMED_OUT: Final = 13
PEER_LOST: Final = 14
CANCELLED: Final = 15
CLOSED: Final = 16
OUT_OF_RESOURCES: Final = 17
NO_RANK: Final = (1 << 32) - 1
NO_CHUNK: Final = (1 << 32) - 1
MAX_WORLD_SIZE: Final = 16
MAX_PLAN_STEPS: Final = 30

OP_ALL_REDUCE: Final = 1
OP_ALL_GATHER: Final = 2
OP_REDUCE_SCATTER: Final = 3
OP_BROADCAST: Final = 4
OP_BARRIER: Final = 5

REDUCTION_NONE: Final = 0
REDUCTION_SUM: Final = 1

DTYPE_NONE: Final = 0
DTYPE_BF16: Final = 1
DTYPE_FP32: Final = 2
DTYPE_UINT8: Final = 3
DTYPE_INT32: Final = 4
DTYPE_UINT32: Final = 5

PHASE_REDUCE_SCATTER: Final = 1
PHASE_ALL_GATHER: Final = 2
PHASE_BROADCAST: Final = 3
PHASE_BARRIER: Final = 4

MESSAGE_DATA: Final = 1
MESSAGE_CONSUMED: Final = 2
MESSAGE_ABORT: Final = 3

LIVE_PHASE_UNINITIALIZED: Final = 0
LIVE_PHASE_CONFIGURING: Final = 1
LIVE_PHASE_JOINING: Final = 2
LIVE_PHASE_READY: Final = 3
LIVE_PHASE_CLOSING: Final = 4
LIVE_PHASE_ABORTED: Final = 5
LIVE_PHASE_CLOSED: Final = 6

EXPECTED_RUNTIME_VERSION: Final = "0.8.0"
EXPECTED_ABI_VERSION: Final = (1 << 16) | 8
UINT32_MAX: Final = (1 << 32) - 1
INT32_MAX: Final = (1 << 31) - 1
INT32_MIN: Final = -(1 << 31)
UINT64_MAX: Final = (1 << 64) - 1


class GroupIdentity(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("protocol_major", ctypes.c_uint16),
        ("protocol_minor", ctypes.c_uint16),
        ("world_size", ctypes.c_uint32),
        ("epoch", ctypes.c_uint64),
        ("group_generation", ctypes.c_uint64),
        ("job_uuid", ctypes.c_uint8 * 16),
        ("group_uuid", ctypes.c_uint8 * 16),
        ("model_identity_sha256", ctypes.c_uint8 * 32),
        ("reserved", ctypes.c_uint8 * 16),
    ]


class Descriptor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("group", GroupIdentity),
        ("sequence", ctypes.c_uint64),
        ("input_count", ctypes.c_uint64),
        ("output_count", ctypes.c_uint64),
        ("rank", ctypes.c_uint32),
        ("operation", ctypes.c_uint32),
        ("reduction", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("root_rank", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 16),
    ]


class PlanStep(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("phase", ctypes.c_uint32),
        ("action", ctypes.c_uint32),
        ("step_index", ctypes.c_uint32),
        ("send_rank", ctypes.c_uint32),
        ("receive_rank", ctypes.c_uint32),
        ("send_chunk", ctypes.c_uint32),
        ("receive_chunk", ctypes.c_uint32),
        ("send_offset_elements", ctypes.c_uint64),
        ("send_count_elements", ctypes.c_uint64),
        ("receive_offset_elements", ctypes.c_uint64),
        ("receive_count_elements", ctypes.c_uint64),
        ("reserved", ctypes.c_uint8 * 16),
    ]


class CarrierRecord(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("group", GroupIdentity),
        ("descriptor_sha256", ctypes.c_uint8 * 32),
        ("sequence", ctypes.c_uint64),
        ("slot_generation", ctypes.c_uint64),
        ("payload_bytes", ctypes.c_uint64),
        ("kind", ctypes.c_uint32),
        ("phase", ctypes.c_uint32),
        ("step_index", ctypes.c_uint32),
        ("chunk_index", ctypes.c_uint32),
        ("source_rank", ctypes.c_uint32),
        ("destination_rank", ctypes.c_uint32),
        ("slot_index", ctypes.c_uint32),
        ("payload_crc32c", ctypes.c_uint32),
        ("status", ctypes.c_int32),
        ("failed_rank", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 12),
    ]


class CarrierSessionInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("phase", ctypes.c_uint32),
        ("self_rank", ctypes.c_uint32),
        ("world_size", ctypes.c_uint32),
        ("credits_per_peer", ctypes.c_uint32),
        ("sender_inflight", ctypes.c_uint32),
        ("receiver_ready", ctypes.c_uint32),
        ("receiver_consumed", ctypes.c_uint32),
        ("first_error", ctypes.c_int32),
        ("failed_rank", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 16),
    ]


class ProcessIdentity(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("pid", ctypes.c_int32),
        ("uid", ctypes.c_uint32),
        ("gid", ctypes.c_uint32),
        ("start_time_ticks", ctypes.c_uint64),
        ("reserved", ctypes.c_uint8 * 16),
    ]


class LiveAbort(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("group", GroupIdentity),
        ("context_sequence", ctypes.c_uint64),
        ("reporter_rank", ctypes.c_uint32),
        ("failed_rank", ctypes.c_uint32),
        ("status", ctypes.c_int32),
        ("reserved0", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 16),
    ]


class LiveRankInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("phase", ctypes.c_uint32),
        ("self_rank", ctypes.c_uint32),
        ("world_size", ctypes.c_uint32),
        ("control_socket", ctypes.c_int32),
        ("reserved0", ctypes.c_uint32),
        ("group", GroupIdentity),
        ("peer_sockets", ctypes.c_int32 * MAX_WORLD_SIZE),
        ("reserved", ctypes.c_uint8 * 16),
    ]


class LiveBrokerInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("phase", ctypes.c_uint32),
        ("world_size", ctypes.c_uint32),
        ("prepared_mask", ctypes.c_uint32),
        ("bound_mask", ctypes.c_uint32),
        ("joined_mask", ctypes.c_uint32),
        ("ready_mask", ctypes.c_uint32),
        ("departed_mask", ctypes.c_uint32),
        ("close_pending_mask", ctypes.c_uint32),
        ("abort_pending_mask", ctypes.c_uint32),
        ("owner", ProcessIdentity),
        ("reserved", ctypes.c_uint8 * 16),
    ]


def _validate_live_layout() -> None:
    expected = (
        (ProcessIdentity, 40, {"start_time_ticks": 16}),
        (
            LiveAbort,
            160,
            {"group": 8, "context_sequence": 120, "reserved": 144},
        ),
        (
            LiveRankInfo,
            216,
            {"group": 24, "peer_sockets": 136, "reserved": 200},
        ),
        (LiveBrokerInfo, 96, {"owner": 40, "reserved": 80}),
    )
    for structure, size, offsets in expected:
        if ctypes.sizeof(structure) != size:
            raise RuntimeError(
                f"{structure.__name__} ctypes size is {ctypes.sizeof(structure)}, "
                f"expected {size}"
            )
        for field, offset in offsets.items():
            actual = getattr(structure, field).offset
            if actual != offset:
                raise RuntimeError(
                    f"{structure.__name__}.{field} ctypes offset is {actual}, "
                    f"expected {offset}"
                )


_validate_live_layout()


@dataclasses.dataclass(frozen=True)
class PlannedStep:
    phase: int
    action: int
    step_index: int
    send_rank: int
    receive_rank: int
    send_chunk: int
    receive_chunk: int
    send_offset_elements: int
    send_count_elements: int
    receive_offset_elements: int
    receive_count_elements: int


class CCLStatusError(RuntimeError):
    def __init__(
        self,
        status: int,
        operation: str,
        status_name: str,
        abort_record: CarrierRecord | LiveAbort | None = None,
    ):
        super().__init__(f"{operation} failed: {status_name} ({status})")
        self.status = status
        self.operation = operation
        self.status_name = status_name
        self.abort_record = abort_record
        self.first_error = abort_record


def _checked_int(value: int, name: str, maximum: int) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return value


def _checked_signed_int(value: int, name: str) -> int:
    if type(value) is not int or value < INT32_MIN or value > INT32_MAX:
        raise ValueError(
            f"{name} must be an integer in [{INT32_MIN}, {INT32_MAX}]"
        )
    return value


def _checked_fd(value: int, name: str) -> int:
    value = _checked_int(value, name, INT32_MAX)
    return value


def _checked_deadline(value: int, name: str = "absolute_deadline_ns") -> int:
    value = _checked_int(value, name, UINT64_MAX)
    if value == 0:
        raise ValueError(f"{name} must be a nonzero absolute monotonic deadline")
    return value


def _require_structure(value, expected: type[ctypes.Structure], name: str):
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be {expected.__name__}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_bytes(target, value: bytes, name: str) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != len(target):
        raise ValueError(f"{name} must contain exactly {len(target)} bytes")
    target[:] = value


class NativeCCL:
    def __init__(self, library: Path | None = None) -> None:
        if library is None:
            prefix = os.environ.get("ROCM_SIM_ROOT")
            if not prefix:
                raise RuntimeError("ROCM_SIM_ROOT is required for the CCL runtime")
            library = Path(prefix).resolve() / "lib/libself_amdgpu_runtime.so.1"
        library = Path(library).resolve(strict=True)
        self.library_path = library
        self.library_sha256 = _sha256_file(library)
        self.lib = ctypes.CDLL(str(library), use_errno=True)
        self._bind()
        self.abi_version = int(self.lib.sagr_abi_version())
        raw_version = self.lib.sagr_version_string()
        self.runtime_version = raw_version.decode("ascii", "strict")
        if (
            self.abi_version != EXPECTED_ABI_VERSION
            or self.runtime_version != EXPECTED_RUNTIME_VERSION
        ):
            raise RuntimeError(
                "CCL runtime identity mismatch: "
                f"version={self.runtime_version}, abi=0x{self.abi_version:08x}"
            )

    def _bind(self) -> None:
        lib = self.lib
        lib.sagr_abi_version.argtypes = []
        lib.sagr_abi_version.restype = ctypes.c_uint32
        lib.sagr_version_string.argtypes = []
        lib.sagr_version_string.restype = ctypes.c_char_p
        lib.sagr_ccl_v1_status_string.argtypes = [ctypes.c_int32]
        lib.sagr_ccl_v1_status_string.restype = ctypes.c_char_p
        lib.sagr_ccl_v1_group_identity_init.argtypes = [
            ctypes.POINTER(GroupIdentity), ctypes.c_uint32
        ]
        lib.sagr_ccl_v1_group_identity_init.restype = ctypes.c_int32
        lib.sagr_ccl_v1_group_identity_validate.argtypes = [
            ctypes.POINTER(GroupIdentity)
        ]
        lib.sagr_ccl_v1_group_identity_validate.restype = ctypes.c_int32
        lib.sagr_ccl_v1_descriptor_init.argtypes = [
            ctypes.POINTER(Descriptor), ctypes.c_uint32
        ]
        lib.sagr_ccl_v1_descriptor_init.restype = ctypes.c_int32
        lib.sagr_ccl_v1_descriptor_validate.argtypes = [ctypes.POINTER(Descriptor)]
        lib.sagr_ccl_v1_descriptor_validate.restype = ctypes.c_int32
        lib.sagr_ccl_v1_descriptor_sha256.argtypes = [
            ctypes.POINTER(Descriptor), ctypes.POINTER(ctypes.c_uint8)
        ]
        lib.sagr_ccl_v1_descriptor_sha256.restype = ctypes.c_int32
        lib.sagr_ccl_v1_plan_rank.argtypes = [
            ctypes.POINTER(Descriptor), ctypes.POINTER(PlanStep),
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)
        ]
        lib.sagr_ccl_v1_plan_rank.restype = ctypes.c_int32

        session = ctypes.c_void_p
        lib.sagr_ccl_v1_carrier_session_create.argtypes = [
            ctypes.POINTER(GroupIdentity), ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(session)
        ]
        lib.sagr_ccl_v1_carrier_session_create.restype = ctypes.c_int32
        lib.sagr_ccl_v1_carrier_session_prepare_data.argtypes = [
            session, ctypes.POINTER(Descriptor), ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(CarrierRecord),
            ctypes.c_uint32
        ]
        lib.sagr_ccl_v1_carrier_session_prepare_data.restype = ctypes.c_int32
        lib.sagr_ccl_v1_carrier_session_send_data.argtypes = [
            session, ctypes.c_int, ctypes.POINTER(CarrierRecord)
        ]
        lib.sagr_ccl_v1_carrier_session_send_data.restype = ctypes.c_int32
        lib.sagr_ccl_v1_carrier_session_receive.argtypes = [
            session, ctypes.c_int, ctypes.POINTER(Descriptor), ctypes.c_uint32,
            ctypes.c_uint32, ctypes.POINTER(CarrierRecord), ctypes.c_uint32
        ]
        lib.sagr_ccl_v1_carrier_session_receive.restype = ctypes.c_int32
        lib.sagr_ccl_v1_carrier_session_consume.argtypes = [
            session, ctypes.POINTER(Descriptor), ctypes.c_uint32,
            ctypes.POINTER(CarrierRecord), ctypes.c_void_p, ctypes.c_uint64,
            ctypes.POINTER(CarrierRecord), ctypes.c_uint32
        ]
        lib.sagr_ccl_v1_carrier_session_consume.restype = ctypes.c_int32
        lib.sagr_ccl_v1_carrier_session_send_consumed.argtypes = [
            session, ctypes.c_int, ctypes.POINTER(CarrierRecord)
        ]
        lib.sagr_ccl_v1_carrier_session_send_consumed.restype = ctypes.c_int32
        lib.sagr_ccl_v1_carrier_session_abort.argtypes = [
            session, ctypes.POINTER(Descriptor), ctypes.c_uint32,
            ctypes.c_int32, ctypes.POINTER(CarrierRecord), ctypes.c_uint32
        ]
        lib.sagr_ccl_v1_carrier_session_abort.restype = ctypes.c_int32
        lib.sagr_ccl_v1_carrier_session_send_abort.argtypes = [
            session, ctypes.c_int, ctypes.POINTER(CarrierRecord)
        ]
        lib.sagr_ccl_v1_carrier_session_send_abort.restype = ctypes.c_int32
        lib.sagr_ccl_v1_carrier_session_get_abort.argtypes = [
            session, ctypes.POINTER(CarrierRecord), ctypes.c_uint32
        ]
        lib.sagr_ccl_v1_carrier_session_get_abort.restype = ctypes.c_int32
        lib.sagr_ccl_v1_carrier_session_info.argtypes = [
            session, ctypes.POINTER(CarrierSessionInfo), ctypes.c_uint32
        ]
        lib.sagr_ccl_v1_carrier_session_info.restype = ctypes.c_int32
        lib.sagr_ccl_v1_carrier_session_destroy.argtypes = [ctypes.POINTER(session)]
        lib.sagr_ccl_v1_carrier_session_destroy.restype = None

        broker = ctypes.c_void_p
        rank = ctypes.c_void_p
        lib.sagr_ccl_live_v1_monotonic_time_ns.argtypes = []
        lib.sagr_ccl_live_v1_monotonic_time_ns.restype = ctypes.c_uint64
        lib.sagr_ccl_live_v1_process_identity.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ProcessIdentity),
            ctypes.c_uint32,
        ]
        lib.sagr_ccl_live_v1_process_identity.restype = ctypes.c_int32
        lib.sagr_ccl_live_v1_broker_create.argtypes = [
            ctypes.POINTER(GroupIdentity),
            ctypes.POINTER(broker),
        ]
        lib.sagr_ccl_live_v1_broker_create.restype = ctypes.c_int32
        lib.sagr_ccl_live_v1_broker_info.argtypes = [
            broker,
            ctypes.POINTER(LiveBrokerInfo),
            ctypes.c_uint32,
        ]
        lib.sagr_ccl_live_v1_broker_info.restype = ctypes.c_int32
        lib.sagr_ccl_live_v1_broker_prepare_rank.argtypes = [
            broker,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.sagr_ccl_live_v1_broker_prepare_rank.restype = ctypes.c_int32
        lib.sagr_ccl_live_v1_broker_bind_rank.argtypes = [
            broker,
            ctypes.c_uint32,
            ctypes.POINTER(ProcessIdentity),
        ]
        lib.sagr_ccl_live_v1_broker_bind_rank.restype = ctypes.c_int32
        lib.sagr_ccl_live_v1_broker_rendezvous.argtypes = [
            broker,
            ctypes.c_uint64,
        ]
        lib.sagr_ccl_live_v1_broker_rendezvous.restype = ctypes.c_int32
        lib.sagr_ccl_live_v1_broker_progress.argtypes = [
            broker,
            ctypes.POINTER(LiveAbort),
            ctypes.c_uint32,
        ]
        lib.sagr_ccl_live_v1_broker_progress.restype = ctypes.c_int32
        lib.sagr_ccl_live_v1_broker_abort.argtypes = [
            broker,
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_uint64,
        ]
        lib.sagr_ccl_live_v1_broker_abort.restype = ctypes.c_int32
        lib.sagr_ccl_live_v1_broker_first_error.argtypes = [
            broker,
            ctypes.POINTER(LiveAbort),
            ctypes.c_uint32,
        ]
        lib.sagr_ccl_live_v1_broker_first_error.restype = ctypes.c_int32
        lib.sagr_ccl_live_v1_broker_destroy.argtypes = [ctypes.POINTER(broker)]
        lib.sagr_ccl_live_v1_broker_destroy.restype = None
        lib.sagr_ccl_live_v1_rank_join.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(GroupIdentity),
            ctypes.c_uint32,
            ctypes.POINTER(ProcessIdentity),
            ctypes.c_uint64,
            ctypes.POINTER(rank),
        ]
        lib.sagr_ccl_live_v1_rank_join.restype = ctypes.c_int32
        lib.sagr_ccl_live_v1_rank_info.argtypes = [
            rank,
            ctypes.POINTER(LiveRankInfo),
            ctypes.c_uint32,
        ]
        lib.sagr_ccl_live_v1_rank_info.restype = ctypes.c_int32
        lib.sagr_ccl_live_v1_rank_report_abort.argtypes = [
            rank,
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_uint64,
        ]
        lib.sagr_ccl_live_v1_rank_report_abort.restype = ctypes.c_int32
        lib.sagr_ccl_live_v1_rank_poll_abort.argtypes = [
            rank,
            ctypes.POINTER(LiveAbort),
            ctypes.c_uint32,
        ]
        lib.sagr_ccl_live_v1_rank_poll_abort.restype = ctypes.c_int32
        lib.sagr_ccl_live_v1_rank_close.argtypes = [rank, ctypes.c_uint64]
        lib.sagr_ccl_live_v1_rank_close.restype = ctypes.c_int32
        lib.sagr_ccl_live_v1_rank_destroy.argtypes = [ctypes.POINTER(rank)]
        lib.sagr_ccl_live_v1_rank_destroy.restype = None

    def status_name(self, status: int) -> str:
        status = _checked_signed_int(status, "status")
        value = self.lib.sagr_ccl_v1_status_string(status)
        return value.decode("ascii", "strict") if value else "UNKNOWN"

    def check(self, status: int, operation: str) -> None:
        status = _checked_signed_int(status, "status")
        if not isinstance(operation, str):
            raise TypeError("operation must be str")
        if status != SUCCESS:
            raise CCLStatusError(status, operation, self.status_name(status))

    def identity(
        self,
        *,
        world_size: int,
        epoch: int,
        group_generation: int,
        job_uuid: bytes,
        group_uuid: bytes,
        model_identity_sha256: bytes,
    ) -> GroupIdentity:
        world_size = _checked_int(world_size, "world_size", UINT32_MAX)
        epoch = _checked_int(epoch, "epoch", UINT64_MAX)
        group_generation = _checked_int(
            group_generation, "group_generation", UINT64_MAX
        )
        identity = GroupIdentity()
        self.check(
            self.lib.sagr_ccl_v1_group_identity_init(
                ctypes.byref(identity), ctypes.sizeof(identity)
            ),
            "group identity initialization",
        )
        identity.world_size = world_size
        identity.epoch = epoch
        identity.group_generation = group_generation
        _copy_bytes(identity.job_uuid, job_uuid, "job_uuid")
        _copy_bytes(identity.group_uuid, group_uuid, "group_uuid")
        _copy_bytes(
            identity.model_identity_sha256,
            model_identity_sha256,
            "model_identity_sha256",
        )
        self.check(
            self.lib.sagr_ccl_v1_group_identity_validate(ctypes.byref(identity)),
            "group identity validation",
        )
        return identity

    def monotonic_time_ns(self) -> int:
        value = int(self.lib.sagr_ccl_live_v1_monotonic_time_ns())
        if value == 0:
            raise RuntimeError("CLOCK_MONOTONIC is unavailable")
        return value

    def deadline_after(self, timeout_ns: int) -> int:
        timeout_ns = _checked_int(timeout_ns, "timeout_ns", UINT64_MAX)
        now = self.monotonic_time_ns()
        if timeout_ns == 0 or timeout_ns > UINT64_MAX - now:
            raise ValueError("timeout_ns must produce a future uint64 deadline")
        return now + timeout_ns

    def process_identity(self, pid: int) -> ProcessIdentity:
        pid = _checked_signed_int(pid, "pid")
        if pid <= 0:
            raise ValueError("pid must be positive")
        identity = ProcessIdentity()
        self.check(
            self.lib.sagr_ccl_live_v1_process_identity(
                pid, ctypes.byref(identity), ctypes.sizeof(identity)
            ),
            "live process identity query",
        )
        return identity

    def live_broker(self, identity: GroupIdentity) -> "LiveBroker":
        _require_structure(identity, GroupIdentity, "identity")
        return LiveBroker(self, identity)

    def broker(self, identity: GroupIdentity) -> "LiveBroker":
        return self.live_broker(identity)

    def join_rank(
        self,
        capability_socket: int,
        identity: GroupIdentity,
        self_rank: int,
        expected_broker: ProcessIdentity,
        absolute_deadline_ns: int,
    ) -> "LiveRank":
        """Join a live group, transferring ownership of ``capability_socket``.

        Python argument validation happens before ownership transfer.  Once
        the C call begins, the runtime closes the descriptor on every return
        path, and the caller must not close or reuse it.
        """
        capability_socket = _checked_fd(capability_socket, "capability_socket")
        _require_structure(identity, GroupIdentity, "identity")
        self_rank = _checked_int(self_rank, "self_rank", UINT32_MAX)
        _require_structure(expected_broker, ProcessIdentity, "expected_broker")
        absolute_deadline_ns = _checked_deadline(absolute_deadline_ns)
        handle = ctypes.c_void_p()
        status = self.lib.sagr_ccl_live_v1_rank_join(
            capability_socket,
            ctypes.byref(identity),
            self_rank,
            ctypes.byref(expected_broker),
            absolute_deadline_ns,
            ctypes.byref(handle),
        )
        if status != SUCCESS:
            raise CCLStatusError(
                status, "live rank join", self.status_name(status)
            )
        return LiveRank(self, handle)

    def rank_join(
        self,
        capability_socket: int,
        identity: GroupIdentity,
        self_rank: int,
        expected_broker: ProcessIdentity,
        absolute_deadline_ns: int,
    ) -> "LiveRank":
        return self.join_rank(
            capability_socket,
            identity,
            self_rank,
            expected_broker,
            absolute_deadline_ns,
        )

    def descriptor(
        self,
        identity: GroupIdentity,
        *,
        sequence: int,
        input_count: int,
        output_count: int,
        rank: int,
        operation: int,
        reduction: int,
        dtype: int,
        root_rank: int = NO_RANK,
    ) -> Descriptor:
        _require_structure(identity, GroupIdentity, "identity")
        sequence = _checked_int(sequence, "sequence", UINT64_MAX)
        input_count = _checked_int(input_count, "input_count", UINT64_MAX)
        output_count = _checked_int(output_count, "output_count", UINT64_MAX)
        rank = _checked_int(rank, "rank", UINT32_MAX)
        operation = _checked_int(operation, "operation", UINT32_MAX)
        reduction = _checked_int(reduction, "reduction", UINT32_MAX)
        dtype = _checked_int(dtype, "dtype", UINT32_MAX)
        root_rank = _checked_int(root_rank, "root_rank", UINT32_MAX)
        descriptor = Descriptor()
        self.check(
            self.lib.sagr_ccl_v1_descriptor_init(
                ctypes.byref(descriptor), ctypes.sizeof(descriptor)
            ),
            "descriptor initialization",
        )
        descriptor.group = identity
        descriptor.sequence = sequence
        descriptor.input_count = input_count
        descriptor.output_count = output_count
        descriptor.rank = rank
        descriptor.operation = operation
        descriptor.reduction = reduction
        descriptor.dtype = dtype
        descriptor.root_rank = root_rank
        self.check(
            self.lib.sagr_ccl_v1_descriptor_validate(ctypes.byref(descriptor)),
            "descriptor validation",
        )
        return descriptor

    def descriptor_sha256(self, descriptor: Descriptor) -> str:
        _require_structure(descriptor, Descriptor, "descriptor")
        digest = (ctypes.c_uint8 * 32)()
        self.check(
            self.lib.sagr_ccl_v1_descriptor_sha256(
                ctypes.byref(descriptor), digest
            ),
            "descriptor hashing",
        )
        return bytes(digest).hex()

    def plan(self, descriptor: Descriptor) -> tuple[PlannedStep, ...]:
        _require_structure(descriptor, Descriptor, "descriptor")
        raw = (PlanStep * MAX_PLAN_STEPS)()
        count = ctypes.c_uint32()
        self.check(
            self.lib.sagr_ccl_v1_plan_rank(
                ctypes.byref(descriptor), raw, len(raw), ctypes.byref(count)
            ),
            "rank planning",
        )
        return tuple(
            PlannedStep(
                phase=item.phase,
                action=item.action,
                step_index=item.step_index,
                send_rank=item.send_rank,
                receive_rank=item.receive_rank,
                send_chunk=item.send_chunk,
                receive_chunk=item.receive_chunk,
                send_offset_elements=item.send_offset_elements,
                send_count_elements=item.send_count_elements,
                receive_offset_elements=item.receive_offset_elements,
                receive_count_elements=item.receive_count_elements,
            )
            for item in raw[: count.value]
        )

    def carrier_session(
        self, identity: GroupIdentity, rank: int, credits_per_peer: int = 4
    ) -> "CarrierSession":
        _require_structure(identity, GroupIdentity, "identity")
        rank = _checked_int(rank, "rank", UINT32_MAX)
        credits_per_peer = _checked_int(
            credits_per_peer, "credits_per_peer", UINT32_MAX
        )
        return CarrierSession(self, identity, rank, credits_per_peer)


class CarrierSession:
    def __init__(
        self,
        native: NativeCCL,
        identity: GroupIdentity,
        rank: int,
        credits_per_peer: int,
    ) -> None:
        self.native = native
        self.handle = ctypes.c_void_p()
        native.check(
            native.lib.sagr_ccl_v1_carrier_session_create(
                ctypes.byref(identity), rank, credits_per_peer,
                ctypes.byref(self.handle)
            ),
            "carrier session creation",
        )

    def close(self) -> None:
        if self.handle.value:
            self.native.lib.sagr_ccl_v1_carrier_session_destroy(
                ctypes.byref(self.handle)
            )

    def __enter__(self) -> "CarrierSession":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def prepare_data(
        self, descriptor: Descriptor, step_index: int, payload: bytes
    ) -> CarrierRecord:
        _require_structure(descriptor, Descriptor, "descriptor")
        step_index = _checked_int(step_index, "step_index", UINT32_MAX)
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        record = CarrierRecord()
        storage = ctypes.create_string_buffer(payload) if payload else None
        pointer = ctypes.cast(storage, ctypes.c_void_p) if storage else None
        self.native.check(
            self.native.lib.sagr_ccl_v1_carrier_session_prepare_data(
                self.handle, ctypes.byref(descriptor), step_index, pointer,
                len(payload), ctypes.byref(record), ctypes.sizeof(record)
            ),
            "carrier data preparation",
        )
        return record

    def send_data(self, socket_descriptor: int, record: CarrierRecord) -> bool:
        socket_descriptor = _checked_fd(socket_descriptor, "socket_descriptor")
        _require_structure(record, CarrierRecord, "record")
        status = self.native.lib.sagr_ccl_v1_carrier_session_send_data(
            self.handle, socket_descriptor, ctypes.byref(record)
        )
        if status == BUSY:
            return False
        self.native.check(status, "carrier data send")
        return True

    def receive(
        self,
        socket_descriptor: int,
        descriptor: Descriptor,
        step_index: int,
        authenticated_peer_rank: int,
    ) -> CarrierRecord | None:
        socket_descriptor = _checked_fd(socket_descriptor, "socket_descriptor")
        _require_structure(descriptor, Descriptor, "descriptor")
        step_index = _checked_int(step_index, "step_index", UINT32_MAX)
        authenticated_peer_rank = _checked_int(
            authenticated_peer_rank, "authenticated_peer_rank", UINT32_MAX
        )
        record = CarrierRecord()
        status = self.native.lib.sagr_ccl_v1_carrier_session_receive(
            self.handle, socket_descriptor, ctypes.byref(descriptor), step_index,
            authenticated_peer_rank, ctypes.byref(record), ctypes.sizeof(record)
        )
        if status == BUSY:
            return None
        if status != SUCCESS:
            abort_record = None
            if self.info().first_error != SUCCESS:
                try:
                    abort_record = self.get_abort()
                except CCLStatusError:
                    abort_record = None
            raise CCLStatusError(
                status,
                "carrier receive",
                self.native.status_name(status),
                abort_record,
            )
        return record

    def consume(
        self,
        descriptor: Descriptor,
        step_index: int,
        record: CarrierRecord,
    ) -> tuple[bytes, CarrierRecord]:
        _require_structure(descriptor, Descriptor, "descriptor")
        _require_structure(record, CarrierRecord, "record")
        step_index = _checked_int(step_index, "step_index", UINT32_MAX)
        capacity = int(record.payload_bytes)
        storage = ctypes.create_string_buffer(capacity) if capacity else None
        pointer = ctypes.cast(storage, ctypes.c_void_p) if storage else None
        consumed = CarrierRecord()
        self.native.check(
            self.native.lib.sagr_ccl_v1_carrier_session_consume(
                self.handle, ctypes.byref(descriptor), step_index,
                ctypes.byref(record), pointer, capacity,
                ctypes.byref(consumed), ctypes.sizeof(consumed)
            ),
            "carrier payload consume",
        )
        payload = storage.raw if storage is not None else b""
        return payload, consumed

    def send_consumed(
        self, socket_descriptor: int, record: CarrierRecord
    ) -> bool:
        socket_descriptor = _checked_fd(socket_descriptor, "socket_descriptor")
        _require_structure(record, CarrierRecord, "record")
        status = self.native.lib.sagr_ccl_v1_carrier_session_send_consumed(
            self.handle, socket_descriptor, ctypes.byref(record)
        )
        if status == BUSY:
            return False
        self.native.check(status, "carrier consumed send")
        return True

    def abort(
        self,
        descriptor: Descriptor,
        failed_rank: int,
        reason: int = PROTOCOL_ERROR,
    ) -> CarrierRecord:
        _require_structure(descriptor, Descriptor, "descriptor")
        failed_rank = _checked_int(failed_rank, "failed_rank", UINT32_MAX)
        reason = _checked_signed_int(reason, "reason")
        record = CarrierRecord()
        status = self.native.lib.sagr_ccl_v1_carrier_session_abort(
            self.handle,
            ctypes.byref(descriptor),
            failed_rank,
            reason,
            ctypes.byref(record),
            ctypes.sizeof(record),
        )
        if status != reason:
            self.native.check(status, "carrier abort")
        return record

    def get_abort(self) -> CarrierRecord:
        record = CarrierRecord()
        self.native.check(
            self.native.lib.sagr_ccl_v1_carrier_session_get_abort(
                self.handle, ctypes.byref(record), ctypes.sizeof(record)
            ),
            "carrier abort query",
        )
        return record

    def send_abort(self, socket_descriptor: int, record: CarrierRecord) -> bool:
        socket_descriptor = _checked_fd(socket_descriptor, "socket_descriptor")
        _require_structure(record, CarrierRecord, "record")
        status = self.native.lib.sagr_ccl_v1_carrier_session_send_abort(
            self.handle, socket_descriptor, ctypes.byref(record)
        )
        if status == BUSY:
            return False
        self.native.check(status, "carrier abort send")
        return True

    def info(self) -> CarrierSessionInfo:
        info = CarrierSessionInfo()
        self.native.check(
            self.native.lib.sagr_ccl_v1_carrier_session_info(
                self.handle, ctypes.byref(info), ctypes.sizeof(info)
            ),
            "carrier session query",
        )
        return info


class LiveBroker:
    """RAII owner for a live broker and its broker-side descriptors."""

    def __init__(self, native: NativeCCL, identity: GroupIdentity) -> None:
        self.native = native
        self.handle = ctypes.c_void_p()
        native.check(
            native.lib.sagr_ccl_live_v1_broker_create(
                ctypes.byref(identity), ctypes.byref(self.handle)
            ),
            "live broker creation",
        )

    def _require_open(self) -> None:
        if not self.handle.value:
            raise RuntimeError("live broker is closed")

    def destroy(self) -> None:
        if self.handle.value:
            self.native.lib.sagr_ccl_live_v1_broker_destroy(
                ctypes.byref(self.handle)
            )

    def close(self) -> None:
        self.destroy()

    def __enter__(self) -> "LiveBroker":
        self._require_open()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.destroy()

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass

    def info(self) -> LiveBrokerInfo:
        self._require_open()
        info = LiveBrokerInfo()
        self.native.check(
            self.native.lib.sagr_ccl_live_v1_broker_info(
                self.handle, ctypes.byref(info), ctypes.sizeof(info)
            ),
            "live broker query",
        )
        return info

    @property
    def owner(self) -> ProcessIdentity:
        return ProcessIdentity.from_buffer_copy(bytes(self.info().owner))

    def prepare_rank(self, rank: int) -> int:
        """Return a caller-owned CLOEXEC capability FD for ``rank``.

        Close it if it will not be passed to ``NativeCCL.join_rank``.  After a
        fork, the parent must close its duplicate; rank join consumes the child
        descriptor on success and failure.
        """
        self._require_open()
        rank = _checked_int(rank, "rank", UINT32_MAX)
        capability_socket = ctypes.c_int(-1)
        self.native.check(
            self.native.lib.sagr_ccl_live_v1_broker_prepare_rank(
                self.handle, rank, ctypes.byref(capability_socket)
            ),
            "live rank preparation",
        )
        if capability_socket.value < 0:
            raise RuntimeError("live rank preparation returned an invalid FD")
        return int(capability_socket.value)

    def bind_rank(self, rank: int, process: ProcessIdentity) -> None:
        self._require_open()
        rank = _checked_int(rank, "rank", UINT32_MAX)
        _require_structure(process, ProcessIdentity, "process")
        self.native.check(
            self.native.lib.sagr_ccl_live_v1_broker_bind_rank(
                self.handle, rank, ctypes.byref(process)
            ),
            "live rank binding",
        )

    def rendezvous(self, absolute_deadline_ns: int) -> None:
        self._require_open()
        absolute_deadline_ns = _checked_deadline(absolute_deadline_ns)
        status = self.native.lib.sagr_ccl_live_v1_broker_rendezvous(
            self.handle, absolute_deadline_ns
        )
        if status != SUCCESS:
            self._raise_with_first_error(status, "live broker rendezvous")

    def first_error(self) -> LiveAbort | None:
        self._require_open()
        first_error = LiveAbort()
        status = self.native.lib.sagr_ccl_live_v1_broker_first_error(
            self.handle, ctypes.byref(first_error), ctypes.sizeof(first_error)
        )
        if status == BUSY:
            return None
        self.native.check(status, "live broker first-error query")
        return first_error

    def progress(self) -> LiveAbort | None:
        """Make nonblocking progress; return a latched error, if any.

        A quiet broker returns ``None``.  CLOSED is also a non-error terminal
        result and can be distinguished through ``info().phase``.
        """
        self._require_open()
        first_error = LiveAbort()
        status = self.native.lib.sagr_ccl_live_v1_broker_progress(
            self.handle, ctypes.byref(first_error), ctypes.sizeof(first_error)
        )
        if status in (SUCCESS, BUSY, CLOSED):
            return None
        if (
            first_error.struct_size == ctypes.sizeof(first_error)
            and first_error.status == status
        ):
            return first_error
        self._raise_with_first_error(status, "live broker progress")
        raise AssertionError("unreachable")

    def abort(
        self,
        failed_rank: int,
        reason: int,
        context_sequence: int = 0,
    ) -> LiveAbort:
        self._require_open()
        failed_rank = _checked_int(failed_rank, "failed_rank", UINT32_MAX)
        reason = _checked_signed_int(reason, "reason")
        context_sequence = _checked_int(
            context_sequence, "context_sequence", UINT64_MAX
        )
        status = self.native.lib.sagr_ccl_live_v1_broker_abort(
            self.handle, failed_rank, reason, context_sequence
        )
        first_error = self.first_error()
        if first_error is not None and first_error.status == status:
            return first_error
        self._raise_with_first_error(status, "live broker abort", first_error)
        raise AssertionError("unreachable")

    def _raise_with_first_error(
        self,
        status: int,
        operation: str,
        first_error: LiveAbort | None = None,
    ) -> None:
        if first_error is None:
            try:
                first_error = self.first_error()
            except CCLStatusError:
                first_error = None
        raise CCLStatusError(
            status, operation, self.native.status_name(status), first_error
        )


class LiveRank:
    """RAII owner for a joined rank, its control FD, and all peer FDs."""

    def __init__(self, native: NativeCCL, handle: ctypes.c_void_p) -> None:
        self.native = native
        self.handle = handle

    def _require_open(self) -> None:
        if not self.handle.value:
            raise RuntimeError("live rank is closed")

    def destroy(self) -> None:
        """Unconditionally release the rank and every FD it owns."""
        if self.handle.value:
            self.native.lib.sagr_ccl_live_v1_rank_destroy(
                ctypes.byref(self.handle)
            )

    def __enter__(self) -> "LiveRank":
        self._require_open()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.destroy()

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass

    def info(self) -> LiveRankInfo:
        """Return a snapshot whose control and peer FDs are borrowed."""
        self._require_open()
        info = LiveRankInfo()
        self.native.check(
            self.native.lib.sagr_ccl_live_v1_rank_info(
                self.handle, ctypes.byref(info), ctypes.sizeof(info)
            ),
            "live rank query",
        )
        return info

    def peer_socket(self, peer_rank: int) -> int:
        """Return a borrowed peer FD; ownership remains with this rank."""
        peer_rank = _checked_int(peer_rank, "peer_rank", UINT32_MAX)
        info = self.info()
        if peer_rank >= info.world_size or peer_rank == info.self_rank:
            raise ValueError("peer_rank must identify another rank in the group")
        descriptor = int(info.peer_sockets[peer_rank])
        if descriptor < 0:
            raise RuntimeError("live rank peer FD is unavailable")
        return descriptor

    def report_abort(
        self,
        failed_rank: int,
        reason: int,
        context_sequence: int = 0,
    ) -> bool:
        self._require_open()
        failed_rank = _checked_int(failed_rank, "failed_rank", UINT32_MAX)
        reason = _checked_signed_int(reason, "reason")
        context_sequence = _checked_int(
            context_sequence, "context_sequence", UINT64_MAX
        )
        status = self.native.lib.sagr_ccl_live_v1_rank_report_abort(
            self.handle, failed_rank, reason, context_sequence
        )
        if status == BUSY:
            return False
        self.native.check(status, "live rank abort report")
        return True

    def poll_abort(self) -> LiveAbort | None:
        """Return ``None`` for BUSY, otherwise return the immutable first error."""
        self._require_open()
        first_error = LiveAbort()
        status = self.native.lib.sagr_ccl_live_v1_rank_poll_abort(
            self.handle, ctypes.byref(first_error), ctypes.sizeof(first_error)
        )
        if status == BUSY:
            return None
        if (
            first_error.struct_size == ctypes.sizeof(first_error)
            and first_error.status == status
        ):
            return first_error
        raise CCLStatusError(
            status, "live rank abort poll", self.native.status_name(status)
        )

    def close(self, absolute_deadline_ns: int | None = None) -> None:
        """Gracefully close by deadline, or destroy immediately when omitted.

        Either form releases all owned FDs before returning or raising.
        """
        if absolute_deadline_ns is None:
            self.destroy()
            return
        if not self.handle.value:
            return
        try:
            absolute_deadline_ns = _checked_deadline(absolute_deadline_ns)
            status = self.native.lib.sagr_ccl_live_v1_rank_close(
                self.handle, absolute_deadline_ns
            )
            first_error = None
            if status not in (SUCCESS, CLOSED):
                try:
                    first_error = self.poll_abort()
                except CCLStatusError:
                    first_error = None
        finally:
            self.destroy()
        if status not in (SUCCESS, CLOSED):
            raise CCLStatusError(
                status,
                "live rank close",
                self.native.status_name(status),
                first_error,
            )

    def graceful_close(self, absolute_deadline_ns: int) -> None:
        self.close(absolute_deadline_ns)
