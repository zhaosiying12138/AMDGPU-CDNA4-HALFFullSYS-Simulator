#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Host-only design gate for a future device-backed live CCL allreduce.

This module deliberately does not launch gem5.  It binds the production CCL
descriptor/planner and rank-launch schemas into a canonical design document,
models the ordered carrier/device acknowledgement gate, and validates the
evidence that a later live run must produce.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CCL_PACKAGE = ROOT / "plugins/collectives/gemsim_ccl/src"
SCRIPTS = ROOT / "scripts"
for _path in (CCL_PACKAGE, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gemsim_ccl.native import (  # noqa: E402
    DTYPE_BF16,
    DTYPE_FP32,
    MESSAGE_CONSUMED,
    MESSAGE_DATA,
    NativeCCL,
    OP_ALL_REDUCE,
    PHASE_ALL_GATHER,
    PHASE_REDUCE_SCATTER,
    REDUCTION_SUM,
)
from gemsim_live_registry import (  # noqa: E402
    make_rank_launch,
    validate_rank_launch_group,
)


DESIGN_SCHEMA = "amdgpu-sim.ccl-live-allreduce-design.v1"
EXPECTED_SCHEMA = "amdgpu-sim.ccl-live-allreduce-expected.v1"
SUCCESS_EVIDENCE_SCHEMA = "amdgpu-sim.ccl-live-allreduce-evidence.v1"
FAILURE_EVIDENCE_SCHEMA = "amdgpu-sim.ccl-live-allreduce-failure.v1"
SYSTEMATIC_WORLDS = tuple(range(2, 17))
LIVE_ACCEPTANCE_WORLDS = (2, 3, 4, 8, 16)
ODD_PLANNER_WORLDS = (3, 5, 7, 15)
ACTION_SEND_RECEIVE = 3
CARRIER_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MANAGED_MAX_SINGLE_ALLOCATION_BYTES = 2 * 1024 * 1024 * 1024
DEVICE_MAX_ELEMENT_COUNT = (1 << 31) - 1
DESIGN_CREDITS_PER_PEER = 2
HEX32 = re.compile(r"[0-9a-f]{32}")
HEX64 = re.compile(r"[0-9a-f]{64}")
DTYPES = {
    "bfloat16": (DTYPE_BF16, 2),
    "float32": (DTYPE_FP32, 4),
}
FAILURE_STATUSES = frozenset(("device_failure", "peer_lost", "timed_out"))
ARITHMETIC_POLICY = {
    "schema": "amdgpu-sim.ccl-ring-sum-arithmetic.v1",
    "bfloat16": (
        "decode-binary32, add-binary32, "
        "round-to-bfloat16-rne-after-each-ring-reduce-step"
    ),
    "float32": (
        "decode-binary32, add-binary32, "
        "round-to-binary32-rne-after-each-ring-reduce-step"
    ),
    "all_gather": "bitwise-copy",
    "oracle_phase": "post_target",
    "oracle_feedback": False,
}


class DesignError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DesignError(message)


def canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise DesignError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def build_expected_wrapper(document: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_design(document)
    return {
        "schema": EXPECTED_SCHEMA,
        "arithmetic_policy": dict(ARITHMETIC_POLICY),
        "design": validated,
    }


def publish_expected_wrapper(path: Path, document: Mapping[str, Any]) -> None:
    output = _normal_absolute(path, "expected output")
    parent = output.parent
    require(parent.is_dir(), "expected output parent must already exist")
    require(
        parent.resolve(strict=True) == parent,
        "expected output parent must not traverse symlinks",
    )
    require(not os.path.lexists(output), "expected output must be absent")
    payload = canonical_json(build_expected_wrapper(document))
    directory_fd = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    temporary = parent / (
        f".{output.name}.tmp-{os.getpid()}-{os.urandom(8).hex()}"
    )
    temporary_fd = -1
    try:
        temporary_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        view = memoryview(payload)
        while view:
            written = os.write(temporary_fd, view)
            require(written > 0, "short write while publishing expected wrapper")
            view = view[written:]
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = -1
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as error:
            raise DesignError("expected output must be absent") from error
        os.fsync(directory_fd)
        os.unlink(temporary)
        os.fsync(directory_fd)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex(value: str, pattern: re.Pattern[str], label: str) -> str:
    require(isinstance(value, str) and pattern.fullmatch(value) is not None,
            f"{label} is not canonical lowercase hex")
    require(set(value) != {"0"}, f"{label} must be nonzero")
    return value


def _integer(value: Any, minimum: int, maximum: int, label: str) -> int:
    require(
        isinstance(value, int)
        and not isinstance(value, bool)
        and minimum <= value <= maximum,
        f"{label} must be an integer in [{minimum}, {maximum}]",
    )
    return value


def _normal_absolute(path: Path, label: str) -> Path:
    path = Path(path)
    require(path.is_absolute(), f"{label} must be absolute")
    require(path == Path(os.path.normpath(path)), f"{label} must be normalized")
    return path


@dataclass(frozen=True)
class DesignConfig:
    world_size: int
    element_count: int
    dtype: str
    epoch: int
    group_generation: int
    job_uuid: str
    group_uuid: str
    model_identity_sha256: str

    def validate(self) -> "DesignConfig":
        _integer(self.world_size, 2, 16, "world_size")
        _integer(self.element_count, 1, (1 << 63) - 1, "element_count")
        require(self.dtype in DTYPES, "dtype must be bfloat16 or float32")
        dtype_bytes = DTYPES[self.dtype][1]
        require(
            self.element_count
            <= MANAGED_MAX_SINGLE_ALLOCATION_BYTES // dtype_bytes,
            "public tensor exceeds the proven managed single-allocation limit",
        )
        _integer(self.epoch, 1, (1 << 63) - 1, "epoch")
        _integer(
            self.group_generation,
            1,
            (1 << 63) - 1,
            "group_generation",
        )
        _hex(self.job_uuid, HEX32, "job_uuid")
        _hex(self.group_uuid, HEX32, "group_uuid")
        _hex(self.model_identity_sha256, HEX64, "model_identity_sha256")
        return self


@dataclass(frozen=True)
class GroupBinding:
    job_uuid: str
    group_uuid: str
    model_identity_sha256: str
    epoch: int
    group_generation: int
    world_size: int

    def validate(self) -> "GroupBinding":
        _hex(self.job_uuid, HEX32, "transfer job_uuid")
        _hex(self.group_uuid, HEX32, "transfer group_uuid")
        _hex(
            self.model_identity_sha256,
            HEX64,
            "transfer model_identity_sha256",
        )
        _integer(self.epoch, 1, (1 << 63) - 1, "transfer epoch")
        _integer(
            self.group_generation,
            1,
            (1 << 63) - 1,
            "transfer group_generation",
        )
        _integer(self.world_size, 2, 16, "transfer world_size")
        return self


@dataclass(frozen=True)
class TransferTuple:
    group: GroupBinding
    descriptor_sha256: str
    sequence: int
    phase: int
    step_index: int
    chunk_index: int
    source_rank: int
    destination_rank: int
    slot_index: int
    slot_generation: int

    def validate(self, credits_per_peer: int) -> "TransferTuple":
        self.group.validate()
        _hex(self.descriptor_sha256, HEX64, "transfer descriptor_sha256")
        _integer(self.sequence, 1, (1 << 63) - 1, "transfer sequence")
        require(
            self.phase in (PHASE_REDUCE_SCATTER, PHASE_ALL_GATHER),
            "transfer phase is invalid",
        )
        _integer(self.step_index, 0, 29, "transfer step_index")
        _integer(self.chunk_index, 0, self.group.world_size - 1, "transfer chunk")
        _integer(self.source_rank, 0, self.group.world_size - 1, "transfer source")
        _integer(
            self.destination_rank,
            0,
            self.group.world_size - 1,
            "transfer destination",
        )
        require(
            self.source_rank != self.destination_rank,
            "transfer endpoints must differ",
        )
        _integer(self.slot_index, 0, credits_per_peer - 1, "transfer slot")
        _integer(
            self.slot_generation,
            1,
            (1 << 63) - 1,
            "transfer generation",
        )
        return self


@dataclass(frozen=True)
class TransferRecord:
    kind: int
    transfer: TransferTuple
    payload_bytes: int

    def validate(self, credits_per_peer: int) -> "TransferRecord":
        require(self.kind in (MESSAGE_DATA, MESSAGE_CONSUMED),
                "transfer record kind is invalid")
        self.transfer.validate(credits_per_peer)
        _integer(
            self.payload_bytes,
            0,
            CARRIER_MAX_PAYLOAD_BYTES,
            "payload_bytes",
        )
        require(
            self.kind == MESSAGE_DATA or self.payload_bytes == 0,
            "CONSUMED must not carry a payload",
        )
        return self


class CreditBusy(DesignError):
    """A nonfatal carrier BUSY result caused by credit exhaustion."""


class CreditLedger:
    """Host model of the carrier's independent sender/receiver slot tables."""

    def __init__(self, self_rank: int, world_size: int, credits_per_peer: int) -> None:
        _integer(world_size, 2, 16, "credit world_size")
        _integer(self_rank, 0, world_size - 1, "credit self_rank")
        _integer(credits_per_peer, 1, 16, "credits_per_peer")
        self.self_rank = self_rank
        self.world_size = world_size
        self.credits_per_peer = credits_per_peer
        self._next_generation = [1] * world_size
        self._outbound: dict[tuple[int, int], int] = {}
        self._inbound: dict[tuple[int, int], int] = {}
        self._inbound_last = [
            [0] * credits_per_peer for _ in range(world_size)
        ]

    def acquire_outbound(self, destination_rank: int) -> tuple[int, int]:
        _integer(destination_rank, 0, self.world_size - 1, "credit destination")
        require(destination_rank != self.self_rank, "cannot credit self transfer")
        for slot in range(self.credits_per_peer):
            key = (destination_rank, slot)
            if key not in self._outbound:
                generation = self._next_generation[destination_rank]
                self._next_generation[destination_rank] += 1
                self._outbound[key] = generation
                return slot, generation
        raise CreditBusy("carrier credit exhausted")

    def release_outbound(self, transfer: TransferTuple) -> None:
        key = (transfer.destination_rank, transfer.slot_index)
        if self._outbound.get(key) != transfer.slot_generation:
            raise DesignError("out_of_order: CONSUMED does not own sender credit")
        del self._outbound[key]

    def discard_outbound(self, transfer: TransferTuple | None) -> None:
        if transfer is None:
            return
        key = (transfer.destination_rank, transfer.slot_index)
        if self._outbound.get(key) == transfer.slot_generation:
            del self._outbound[key]

    def accept_inbound(self, transfer: TransferTuple) -> None:
        key = (transfer.source_rank, transfer.slot_index)
        if (
            key in self._inbound
            or transfer.slot_generation
            <= self._inbound_last[transfer.source_rank][transfer.slot_index]
        ):
            raise DesignError("out_of_order: replayed or occupied DATA slot")
        self._inbound[key] = transfer.slot_generation
        self._inbound_last[transfer.source_rank][transfer.slot_index] = (
            transfer.slot_generation
        )

    def release_inbound(self, transfer: TransferTuple) -> None:
        key = (transfer.source_rank, transfer.slot_index)
        if self._inbound.get(key) != transfer.slot_generation:
            raise DesignError("out_of_order: CONSUMED does not own receiver slot")
        del self._inbound[key]

    def discard_inbound(self, transfer: TransferTuple | None) -> None:
        if transfer is None:
            return
        key = (transfer.source_rank, transfer.slot_index)
        if self._inbound.get(key) == transfer.slot_generation:
            del self._inbound[key]

    @property
    def sender_inflight(self) -> int:
        return len(self._outbound)

    @property
    def receiver_active(self) -> int:
        return len(self._inbound)


class OutboundState(str, Enum):
    INITIAL = "initial"
    PREPARED = "prepared_credit_held"
    DATA_SENT = "data_sent_credit_held"
    COMPLETE = "ack_received_credit_released"
    ABORTED = "aborted"


class InboundState(str, Enum):
    INITIAL = "initial"
    DATA_RECEIVED = "data_received"
    STAGED = "staged"
    DEVICE_SUM_COMPLETE = "device_sum_complete"
    COPY_COMPLETE = "copy_complete"
    COMPLETE = "ack_sent"
    ABORTED = "aborted"


def _fixed_tuple_error(
    observed: TransferTuple,
    *,
    group: GroupBinding,
    descriptor_sha256: str,
    sequence: int,
    phase: int,
    step_index: int,
    chunk_index: int,
    source_rank: int,
    destination_rank: int,
) -> str | None:
    if observed.group != group:
        return "identity_mismatch"
    if (
        observed.descriptor_sha256 != descriptor_sha256
        or observed.sequence != sequence
    ):
        return "sequence_mismatch"
    if (
        observed.source_rank != source_rank
        or observed.destination_rank != destination_rank
    ):
        return "topology_mismatch"
    if observed.step_index != step_index:
        return "topology_mismatch"
    if observed.phase != phase or observed.chunk_index != chunk_index:
        return "protocol_error"
    return None


class OutboundTransferGate:
    """One local DATA send whose credit survives until its exact ACK arrives."""

    def __init__(
        self,
        step: Mapping[str, Any],
        dtype_bytes: int,
        group: GroupBinding,
        descriptor_sha256: str,
        sequence: int,
        self_rank: int,
        ledger: CreditLedger,
    ) -> None:
        self.step = dict(step)
        self.dtype_bytes = dtype_bytes
        self.group = group.validate()
        self.descriptor_sha256 = _hex(
            descriptor_sha256, HEX64, "outbound descriptor_sha256"
        )
        self.sequence = sequence
        self.self_rank = self_rank
        self.ledger = ledger
        require(ledger.self_rank == self_rank, "outbound ledger rank mismatch")
        self.state = OutboundState.INITIAL
        self.record: TransferRecord | None = None
        self.first_error: str | None = None
        self.events: list[str] = []

    def abort(self, reason: str) -> None:
        if self.first_error is None:
            self.first_error = reason
        self.ledger.discard_outbound(
            self.record.transfer if self.record is not None else None
        )
        self.state = OutboundState.ABORTED
        if not self.events or self.events[-1] != "abort":
            self.events.append("abort")

    def _fail(self, reason: str, detail: str) -> None:
        self.abort(reason)
        raise DesignError(f"{reason}: {detail}")

    def prepare_data(self, payload: bytes) -> TransferRecord:
        if self.state != OutboundState.INITIAL:
            self._fail("out_of_order", "DATA was prepared twice")
        expected_bytes = int(self.step["send_count_elements"]) * self.dtype_bytes
        if not isinstance(payload, bytes) or len(payload) != expected_bytes:
            self._fail("protocol_error", "outbound payload extent mismatch")
        slot, generation = self.ledger.acquire_outbound(self.step["send_rank"])
        transfer = TransferTuple(
            group=self.group,
            descriptor_sha256=self.descriptor_sha256,
            sequence=self.sequence,
            phase=int(self.step["phase"]),
            step_index=int(self.step["ordinal"]),
            chunk_index=int(self.step["send_chunk"]),
            source_rank=self.self_rank,
            destination_rank=int(self.step["send_rank"]),
            slot_index=slot,
            slot_generation=generation,
        )
        self.record = TransferRecord(MESSAGE_DATA, transfer, expected_bytes)
        self.record.validate(self.ledger.credits_per_peer)
        self.state = OutboundState.PREPARED
        self.events.append("outbound_prepared_credit_held")
        return self.record

    def send_data(self, record: TransferRecord) -> None:
        if self.state != OutboundState.PREPARED or record != self.record:
            self._fail("out_of_order", "outbound DATA record does not match prepare")
        self.state = OutboundState.DATA_SENT
        self.events.append("outbound_data_sent_credit_held")

    def receive_consumed(
        self, record: TransferRecord, authenticated_peer_rank: int
    ) -> None:
        if self.state != OutboundState.DATA_SENT or self.record is None:
            self._fail("out_of_order", "CONSUMED arrived without inflight DATA")
        try:
            record.validate(self.ledger.credits_per_peer)
        except DesignError as error:
            self._fail("protocol_error", str(error))
        if record.kind != MESSAGE_CONSUMED:
            self._fail("protocol_error", "outbound completion is not CONSUMED")
        if authenticated_peer_rank != self.record.transfer.destination_rank:
            self._fail("topology_mismatch", "ACK arrived from the wrong peer")
        fixed_error = _fixed_tuple_error(
            record.transfer,
            group=self.group,
            descriptor_sha256=self.descriptor_sha256,
            sequence=self.sequence,
            phase=int(self.step["phase"]),
            step_index=int(self.step["ordinal"]),
            chunk_index=int(self.step["send_chunk"]),
            source_rank=self.self_rank,
            destination_rank=int(self.step["send_rank"]),
        )
        if fixed_error is not None:
            self._fail(fixed_error, "ACK fixed transfer tuple mismatch")
        if record.transfer != self.record.transfer:
            self._fail("out_of_order", "ACK slot or generation mismatch")
        try:
            self.ledger.release_outbound(record.transfer)
        except DesignError as error:
            self._fail("out_of_order", str(error))
        self.state = OutboundState.COMPLETE
        self.events.append("outbound_ack_received_credit_released")


class InboundTransferGate:
    """One peer DATA receive whose ACK is gated on staging and device work."""

    def __init__(
        self,
        step: Mapping[str, Any],
        dtype_bytes: int,
        group: GroupBinding,
        descriptor_sha256: str,
        sequence: int,
        self_rank: int,
        ledger: CreditLedger,
    ) -> None:
        self.step = dict(step)
        self.dtype_bytes = dtype_bytes
        self.group = group.validate()
        self.descriptor_sha256 = _hex(
            descriptor_sha256, HEX64, "inbound descriptor_sha256"
        )
        self.sequence = sequence
        self.self_rank = self_rank
        self.ledger = ledger
        require(ledger.self_rank == self_rank, "inbound ledger rank mismatch")
        self.state = InboundState.INITIAL
        self.data_record: TransferRecord | None = None
        self.ack_record: TransferRecord | None = None
        self.first_error: str | None = None
        self.events: list[str] = []
        self.staging_sha256: str | None = None
        self.device_dispatch_count = 0
        self.host_reduction_count = 0

    @property
    def receive_count(self) -> int:
        return int(self.step["receive_count_elements"])

    def abort(self, reason: str) -> None:
        if self.first_error is None:
            self.first_error = reason
        self.ledger.discard_inbound(
            self.data_record.transfer if self.data_record is not None else None
        )
        self.state = InboundState.ABORTED
        if not self.events or self.events[-1] != "abort":
            self.events.append("abort")

    def _fail(self, reason: str, detail: str) -> None:
        self.abort(reason)
        raise DesignError(f"{reason}: {detail}")

    def receive_data(
        self, record: TransferRecord, authenticated_peer_rank: int
    ) -> None:
        if self.state != InboundState.INITIAL:
            self._fail("out_of_order", "DATA arrived on an occupied/replayed step")
        try:
            record.validate(self.ledger.credits_per_peer)
        except DesignError as error:
            self._fail("protocol_error", str(error))
        if record.kind != MESSAGE_DATA:
            self._fail("protocol_error", "inbound record is not DATA")
        if authenticated_peer_rank != int(self.step["receive_rank"]):
            self._fail("topology_mismatch", "DATA arrived from the wrong peer")
        fixed_error = _fixed_tuple_error(
            record.transfer,
            group=self.group,
            descriptor_sha256=self.descriptor_sha256,
            sequence=self.sequence,
            phase=int(self.step["phase"]),
            step_index=int(self.step["ordinal"]),
            chunk_index=int(self.step["receive_chunk"]),
            source_rank=int(self.step["receive_rank"]),
            destination_rank=self.self_rank,
        )
        if fixed_error is not None:
            self._fail(fixed_error, "DATA fixed transfer tuple mismatch")
        expected_bytes = self.receive_count * self.dtype_bytes
        if record.payload_bytes != expected_bytes:
            self._fail("protocol_error", "inbound DATA payload extent mismatch")
        try:
            self.ledger.accept_inbound(record.transfer)
        except DesignError as error:
            self._fail("out_of_order", str(error))
        self.data_record = record
        self.state = InboundState.DATA_RECEIVED
        self.events.append("inbound_data_received")

    def consume_to_immutable_staging(self, payload: bytes) -> TransferRecord:
        if self.state != InboundState.DATA_RECEIVED or self.data_record is None:
            self._fail("out_of_order", "consume requires received DATA")
        expected_bytes = self.receive_count * self.dtype_bytes
        if not isinstance(payload, bytes) or len(payload) != expected_bytes:
            self._fail("protocol_error", "carrier staging extent mismatch")
        self.staging_sha256 = hashlib.sha256(payload).hexdigest()
        self.ack_record = TransferRecord(
            MESSAGE_CONSUMED, self.data_record.transfer, 0
        )
        self.state = InboundState.STAGED
        self.events.append("inbound_immutable_staging")
        return self.ack_record

    def device_sum_complete(self, *, dispatched: bool, succeeded: bool) -> None:
        if self.state != InboundState.STAGED:
            self._fail("out_of_order", "device SUM requires immutable staging")
        if self.step["phase"] != PHASE_REDUCE_SCATTER:
            self._fail("protocol_error", "device SUM is forbidden in all-gather")
        if dispatched != (self.receive_count != 0):
            self._fail("protocol_error", "zero/nonzero dispatch contract violated")
        if not succeeded:
            self._fail("device_failure", "synchronous device SUM failed")
        self.device_dispatch_count += int(dispatched)
        self.state = InboundState.DEVICE_SUM_COMPLETE
        self.events.append(
            "inbound_device_sum_complete" if dispatched else "zero_no_dispatch"
        )

    def copy_complete(self) -> None:
        if self.state != InboundState.STAGED:
            self._fail("out_of_order", "copy requires immutable staging")
        if self.step["phase"] != PHASE_ALL_GATHER:
            self._fail("protocol_error", "copy cannot replace reduce-scatter SUM")
        self.state = InboundState.COPY_COMPLETE
        self.events.append("inbound_copy_complete" if self.receive_count else "zero_no_copy")

    def send_consumed(
        self, record: TransferRecord, socket_peer_rank: int
    ) -> None:
        expected_state = (
            InboundState.DEVICE_SUM_COMPLETE
            if self.step["phase"] == PHASE_REDUCE_SCATTER
            else InboundState.COPY_COMPLETE
        )
        if self.state != expected_state or self.ack_record is None:
            self._fail("out_of_order", "CONSUMED preceded device/copy completion")
        if record != self.ack_record:
            self._fail("out_of_order", "CONSUMED transfer tuple changed")
        if socket_peer_rank != record.transfer.source_rank:
            self._fail("topology_mismatch", "CONSUMED used the wrong peer socket")
        try:
            self.ledger.release_inbound(record.transfer)
        except DesignError as error:
            self._fail("out_of_order", str(error))
        self.state = InboundState.COMPLETE
        self.events.append("inbound_ack_sent")


class StepState(str, Enum):
    ACTIVE = "active"
    COMPLETE = "complete"
    ABORTED = "aborted"


class OrderedStepGate:
    """Full-duplex planner step composed from independent transfer gates."""

    def __init__(
        self,
        step: Mapping[str, Any],
        dtype_bytes: int,
        group: GroupBinding,
        descriptor_sha256: str,
        sequence: int,
        self_rank: int,
        ledger: CreditLedger,
    ) -> None:
        require(dtype_bytes in (2, 4), "dtype_bytes must be 2 or 4")
        self.outbound = OutboundTransferGate(
            step,
            dtype_bytes,
            group,
            descriptor_sha256,
            sequence,
            self_rank,
            ledger,
        )
        self.inbound = InboundTransferGate(
            step,
            dtype_bytes,
            group,
            descriptor_sha256,
            sequence,
            self_rank,
            ledger,
        )

    @property
    def state(self) -> StepState:
        if (
            self.outbound.state == OutboundState.ABORTED
            or self.inbound.state == InboundState.ABORTED
        ):
            return StepState.ABORTED
        if (
            self.outbound.state == OutboundState.COMPLETE
            and self.inbound.state == InboundState.COMPLETE
        ):
            return StepState.COMPLETE
        return StepState.ACTIVE

    @property
    def first_error(self) -> str | None:
        return self.outbound.first_error or self.inbound.first_error

    def abort(self, reason: str) -> None:
        self.outbound.abort(reason)
        self.inbound.abort(reason)

    def _call(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return operation(*args, **kwargs)
        except CreditBusy:
            raise
        except DesignError:
            self.abort(self.first_error or "protocol_error")
            raise

    def prepare_outbound(self, payload: bytes) -> TransferRecord:
        return self._call(self.outbound.prepare_data, payload)

    def send_outbound_data(self, record: TransferRecord) -> None:
        self._call(self.outbound.send_data, record)

    def receive_inbound_data(
        self, record: TransferRecord, authenticated_peer_rank: int
    ) -> None:
        self._call(self.inbound.receive_data, record, authenticated_peer_rank)

    def consume_inbound(self, payload: bytes) -> TransferRecord:
        return self._call(self.inbound.consume_to_immutable_staging, payload)

    def device_sum_complete(self, *, dispatched: bool, succeeded: bool) -> None:
        self._call(
            self.inbound.device_sum_complete,
            dispatched=dispatched,
            succeeded=succeeded,
        )

    def copy_complete(self) -> None:
        self._call(self.inbound.copy_complete)

    def send_inbound_consumed(
        self, record: TransferRecord, socket_peer_rank: int
    ) -> None:
        self._call(self.inbound.send_consumed, record, socket_peer_rank)

    def receive_outbound_consumed(
        self, record: TransferRecord, authenticated_peer_rank: int
    ) -> None:
        self._call(
            self.outbound.receive_consumed, record, authenticated_peer_rank
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "first_error": self.first_error,
            "outbound": {
                "state": self.outbound.state.value,
                "events": list(self.outbound.events),
            },
            "inbound": {
                "state": self.inbound.state.value,
                "events": list(self.inbound.events),
                "staging_sha256": self.inbound.staging_sha256,
                "device_dispatch_count": self.inbound.device_dispatch_count,
                "host_reduction_count": self.inbound.host_reduction_count,
            },
        }


def segment_element_limit(world_size: int, dtype_bytes: int) -> int:
    _integer(world_size, 2, 16, "segment world_size")
    require(dtype_bytes in (2, 4), "segment dtype_bytes must be 2 or 4")
    return min(
        CARRIER_MAX_PAYLOAD_BYTES // dtype_bytes,
        MANAGED_MAX_SINGLE_ALLOCATION_BYTES // dtype_bytes,
        DEVICE_MAX_ELEMENT_COUNT,
    )


def plan_segments(
    element_count: int, world_size: int, dtype_bytes: int
) -> list[dict[str, int]]:
    _integer(element_count, 1, (1 << 63) - 1, "segment element_count")
    limit = segment_element_limit(world_size, dtype_bytes)
    segments: list[dict[str, int]] = []
    base = 0
    sequence = 1
    while base < element_count:
        count = min(limit, element_count - base)
        segments.append(
            {
                "index": len(segments),
                "sequence": sequence,
                "base_offset_elements": base,
                "element_count": count,
                "byte_count": count * dtype_bytes,
            }
        )
        base += count
        sequence += 1
    return segments


def _step_document(
    ordinal: int,
    step: Any,
    dtype_bytes: int,
    segment_base: int,
) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "phase": int(step.phase),
        "phase_step_index": int(step.step_index),
        "action": int(step.action),
        "send_rank": int(step.send_rank),
        "receive_rank": int(step.receive_rank),
        "send_chunk": int(step.send_chunk),
        "receive_chunk": int(step.receive_chunk),
        "send_offset_elements": int(step.send_offset_elements),
        "send_count_elements": int(step.send_count_elements),
        "receive_offset_elements": int(step.receive_offset_elements),
        "receive_count_elements": int(step.receive_count_elements),
        "global_send_offset_elements": (
            segment_base + int(step.send_offset_elements)
        ),
        "global_receive_offset_elements": (
            segment_base + int(step.receive_offset_elements)
        ),
        "send_payload_bytes": int(step.send_count_elements) * dtype_bytes,
        "receive_payload_bytes": int(step.receive_count_elements) * dtype_bytes,
    }


def _rank_launches(config: DesignConfig, namespace_root: Path) -> list[dict[str, Any]]:
    launches = []
    for rank in range(config.world_size):
        rank_root = namespace_root / f"rank-{rank:02d}"
        launches.append(
            make_rank_launch(
                job_uuid=config.job_uuid,
                epoch=config.epoch,
                rank=rank,
                world_size=config.world_size,
                instance_directory=rank_root / "correctness",
                triton_cache_directory=rank_root / "cache/triton",
            )
        )
    return validate_rank_launch_group(launches)


def build_design(
    runtime_library: Path,
    namespace_root: Path,
    config: DesignConfig,
) -> dict[str, Any]:
    config.validate()
    runtime_library = Path(runtime_library).resolve(strict=True)
    namespace_root = _normal_absolute(Path(namespace_root), "namespace_root")
    native = NativeCCL(runtime_library)
    dtype_code, dtype_bytes = DTYPES[config.dtype]
    identity = native.identity(
        world_size=config.world_size,
        epoch=config.epoch,
        group_generation=config.group_generation,
        job_uuid=bytes.fromhex(config.job_uuid),
        group_uuid=bytes.fromhex(config.group_uuid),
        model_identity_sha256=bytes.fromhex(config.model_identity_sha256),
    )
    launches = _rank_launches(config, namespace_root)
    segments = plan_segments(config.element_count, config.world_size, dtype_bytes)
    rank_documents: list[dict[str, Any]] = []
    for rank in range(config.world_size):
        rank_documents.append(
            {
                "rank": rank,
                "rank_launch": launches[rank],
                "rank_launch_sha256": object_sha256(launches[rank]),
                "segments": [],
                "expected": {
                    "data_records": 0,
                    "consumed_records_sent": 0,
                    "consumed_records_received": 0,
                    "ordered_staging_count": 0,
                    "device_sum_launches": 0,
                    "zero_reduce_scatter_receives": 0,
                    "zero_data_sends": 0,
                    "public_commit_count": 1,
                    "host_reduction_count": 0,
                },
            }
        )

    segment_documents: list[dict[str, Any]] = []
    for segment in segments:
        descriptor_hashes: set[str] = set()
        rank_plan_hashes: list[str] = []
        maximum_payload = 0
        for rank, rank_document in enumerate(rank_documents):
            descriptor = native.descriptor(
                identity,
                sequence=segment["sequence"],
                input_count=segment["element_count"],
                output_count=segment["element_count"],
                rank=rank,
                operation=OP_ALL_REDUCE,
                reduction=REDUCTION_SUM,
                dtype=dtype_code,
            )
            descriptor_sha256 = native.descriptor_sha256(descriptor)
            steps = [
                _step_document(
                    ordinal,
                    step,
                    dtype_bytes,
                    segment["base_offset_elements"],
                )
                for ordinal, step in enumerate(native.plan(descriptor))
            ]
            plan_sha256 = object_sha256(steps)
            reduce_steps = steps[: config.world_size - 1]
            expected = {
                "data_records": len(steps),
                "consumed_records_sent": len(steps),
                "consumed_records_received": len(steps),
                "ordered_staging_count": len(steps),
                "device_sum_launches": sum(
                    step["receive_count_elements"] != 0
                    for step in reduce_steps
                ),
                "zero_reduce_scatter_receives": sum(
                    step["receive_count_elements"] == 0
                    for step in reduce_steps
                ),
                "zero_data_sends": sum(
                    step["send_count_elements"] == 0 for step in steps
                ),
            }
            rank_document["segments"].append(
                {
                    **segment,
                    "descriptor_sha256": descriptor_sha256,
                    "steps": steps,
                    "plan_sha256": plan_sha256,
                    "expected": expected,
                }
            )
            for name, value in expected.items():
                rank_document["expected"][name] += value
            descriptor_hashes.add(descriptor_sha256)
            rank_plan_hashes.append(plan_sha256)
            maximum_payload = max(
                maximum_payload,
                *(step["send_payload_bytes"] for step in steps),
                *(step["receive_payload_bytes"] for step in steps),
            )
        require(
            len(descriptor_hashes) == 1,
            "all ranks must share one normalized segment descriptor hash",
        )
        nonzero_chunks = min(segment["element_count"], config.world_size)
        segment_documents.append(
            {
                **segment,
                "descriptor_sha256": next(iter(descriptor_hashes)),
                "rank_plan_chain_sha256": object_sha256(rank_plan_hashes),
                "nonzero_chunks": nonzero_chunks,
                "zero_chunks": config.world_size - nonzero_chunks,
                "maximum_payload_bytes": maximum_payload,
            }
        )

    for rank_document in rank_documents:
        rank_document["descriptor_chain_sha256"] = object_sha256(
            [item["descriptor_sha256"] for item in rank_document["segments"]]
        )
        rank_document["plan_chain_sha256"] = object_sha256(
            [item["plan_sha256"] for item in rank_document["segments"]]
        )

    nonzero_chunks = sum(item["nonzero_chunks"] for item in segment_documents)
    zero_chunks = sum(item["zero_chunks"] for item in segment_documents)
    document = {
        "schema": DESIGN_SCHEMA,
        "execution": {
            "mode": "host_design_only",
            "gem5_started": False,
            "rank_workers_started": False,
            "live_collective_accepted": False,
        },
        "claim_boundary": (
            "This document and its host-only state-machine tests are not "
            "device-backed live collective acceptance."
        ),
        "config": {
            "world_size": config.world_size,
            "element_count": config.element_count,
            "dtype": config.dtype,
            "dtype_bytes": dtype_bytes,
            "first_sequence": 1,
            "last_sequence": len(segments),
            "operation": "allreduce",
            "reduction": "sum",
            "epoch": config.epoch,
            "group_generation": config.group_generation,
            "job_uuid": config.job_uuid,
            "group_uuid": config.group_uuid,
            "model_identity_sha256": config.model_identity_sha256,
        },
        "runtime": {
            "path": str(runtime_library),
            "sha256": native.library_sha256,
            "version": native.runtime_version,
            "abi_version": native.abi_version,
        },
        "limits": {
            "carrier_max_payload_bytes": CARRIER_MAX_PAYLOAD_BYTES,
            "managed_max_single_allocation_bytes": (
                MANAGED_MAX_SINGLE_ALLOCATION_BYTES
            ),
            "device_max_element_count": DEVICE_MAX_ELEMENT_COUNT,
            "credits_per_peer": DESIGN_CREDITS_PER_PEER,
            "sources": {
                "carrier": (
                    "projects/self-amdgpu-runtime/include/"
                    "self_amdgpu_runtime/ccl_carrier_v1.h:"
                    "SAGR_CCL_V1_CARRIER_MAX_PAYLOAD_BYTES"
                ),
                "managed_allocation": (
                    "projects/self-amdgpu-runtime/include/"
                    "self_amdgpu_runtime/runtime.h:"
                    "SAGR_MEMORY_MAX_SINGLE_ALLOCATION_BYTES"
                ),
                "device_index": (
                    "plugins/collectives/gemsim_ccl/src/gemsim_ccl/"
                    "device.py:_MAX_ELEMENT_COUNT"
                ),
            },
        },
        "segmentation": {
            "strategy": "contiguous_16mib_private_storage_segments",
            "segment_element_limit": segment_element_limit(
                config.world_size, dtype_bytes
            ),
            "segment_count": len(segments),
            "sequence_rule": "one descriptor per segment, contiguous 1..K",
            "commit_rule": "all segments succeed, then one fresh public commit",
            "storage_rule": (
                "each source, destination, and staging tensor owns independent "
                "segment-sized storage; never pass a full-public-storage view to "
                "DeviceSumExecutor"
            ),
            "segments": segment_documents,
        },
        "coverage": {
            "systematic_world_sizes": list(SYSTEMATIC_WORLDS),
            "formal_live_acceptance_world_sizes": list(LIVE_ACCEPTANCE_WORLDS),
            "odd_planner_generalization_world_sizes": list(ODD_PLANNER_WORLDS),
            "this_world_is_formal_live_entry": (
                config.world_size in LIVE_ACCEPTANCE_WORLDS
            ),
        },
        "process_architecture": {
            "broker": "NativeCCL.live_broker",
            "capability_creation": "LiveBroker.prepare_rank",
            "worker_spawn": (
                "subprocess.Popen(close_fds=True, pass_fds=(capability_fd,), "
                "start_new_session=True)"
            ),
            "capability_argument_transport": (
                "numeric CLI argument retained across _gemsim_bootstrap exec; "
                "never ambient rank/world environment"
            ),
            "rank_pid_binding": (
                "NativeCCL.process_identity(Popen.pid) then LiveBroker.bind_rank"
            ),
            "managed_session_start": (
                "rank calls existing triton.runtime.driver.active.runtime."
                "_ensure_session() before ccl_live_v1 join; this opens "
                "sagr_managed_session_open_v2 without a kernel dispatch"
            ),
            "peer_table": "NativeCCL.join_rank authenticated ccl_live_v1 FD table",
            "per_rank_daemon": True,
            "per_rank_namespace": True,
            "rank_launch_schema": "amdgpu-sim.gemsim-rank-launch.v1",
            "cleanup": (
                "subreaper-owned worker process groups; bounded TERM/KILL/reap; "
                "rank destroy closes control and every borrowed peer FD"
            ),
        },
        "step_state_machine": {
            "transfer_tuple_fields": [
                "group",
                "descriptor_sha256",
                "sequence",
                "phase",
                "step_index",
                "chunk_index",
                "source_rank",
                "destination_rank",
                "slot_index",
                "slot_generation",
            ],
            "outbound": [
                "prepare_DATA_and_acquire_destination_credit",
                "send_DATA_while_credit_held",
                "receive_exact_matching_CONSUMED",
                "release_sender_credit",
            ],
            "inbound": [
                "receive_authenticated_exact_DATA",
                "consume_to_immutable_bytes_staging",
                "synchronous_device_SUM_for_nonzero_reduce_scatter_or_byte_copy",
                "send_exact_matching_CONSUMED",
            ],
            "step_complete": "outbound and inbound must both be complete",
            "zero_chunk": "no device dispatch, then CONSUMED",
            "device_failure": (
                "latch canonical carrier abort, relay exact record, report live "
                "abort, never acknowledge pending DATA"
            ),
            "deadline_and_peer_loss": (
                "absolute monotonic deadline and live abort polling at every "
                "nonblocking carrier wait"
            ),
            "success_commit": (
                "input remains unchanged; private workspace; allocate fresh "
                "result and copy exactly once after all steps succeed"
            ),
            "host_reduction_count": 0,
        },
        "evidence_contract": {
            "success_schema": SUCCESS_EVIDENCE_SCHEMA,
            "failure_schema": FAILURE_EVIDENCE_SCHEMA,
            "trace_kernel": "_sum_kernel",
            "zero_dispatch_trace": (
                "empty trace with zero lifecycle records; managed daemon identity "
                "comes from session_open_v2 and clean-exit log, not a fake kernel"
            ),
            "trace_identity": [
                "job_uuid",
                "epoch",
                "rank",
                "world_size",
                "daemon_uuid",
                "descriptor_sha256",
                "plan_sha256",
                "rank_launch_sha256",
                "runtime_sha256",
                "trace_sha256",
                "kernel_image_sha256",
            ],
            "oracle": {
                "timing": "post_target_only",
                "allowed_backends": ["cpu", "nvidia"],
                "feedback": False,
            },
            "acceptance_requires": (
                "all ranks exact and same result SHA; input unchanged; fresh "
                "single commit; host/fallback counts zero; exact trace launch "
                "counts; daemon/FD/process cleanup"
            ),
            "acceptance_authority": False,
            "live_collective_accepted": False,
            "future_formal_verifier": {
                "evidence_directory": (
                    "caller-selected absent path created privately and exclusively"
                ),
                "read_policy": (
                    "open regular no-symlink artifacts from that directory only"
                ),
                "rehash": [
                    "rank_result",
                    "dispatch_trace",
                    "gem5_log",
                    "stats",
                ],
                "timing_binding": (
                    "per-step DATA consume precedes SUM retire; retire precedes "
                    "matching CONSUMED; exact transfer tuple and sequence retained"
                ),
                "self_reported_boolean_or_digest_is_authority": False,
            },
        },
        "group_expected": {
            "segment_count": len(segments),
            "planner_steps_per_rank": (
                len(segments) * 2 * (config.world_size - 1)
            ),
            "nonzero_chunks": nonzero_chunks,
            "zero_chunks": zero_chunks,
            "device_sum_launches": nonzero_chunks * (config.world_size - 1),
            "zero_data_sends": 2 * zero_chunks * (config.world_size - 1),
            "host_reduction_count": 0,
            "public_commit_count": 1,
        },
        "ranks": rank_documents,
    }
    validate_design(document)
    return document


def validate_design(document: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(document, Mapping), "design must be an object")
    require(document.get("schema") == DESIGN_SCHEMA, "design schema mismatch")
    execution = document.get("execution")
    require(
        isinstance(execution, Mapping)
        and execution.get("mode") == "host_design_only"
        and execution.get("gem5_started") is False
        and execution.get("rank_workers_started") is False
        and execution.get("live_collective_accepted") is False,
        "host-only execution boundary changed",
    )
    config = document.get("config")
    require(isinstance(config, Mapping), "design config missing")
    world = _integer(config.get("world_size"), 2, 16, "world_size")
    count = _integer(config.get("element_count"), 1, (1 << 63) - 1,
                     "element_count")
    dtype = config.get("dtype")
    require(dtype in DTYPES, "design dtype mismatch")
    dtype_bytes = DTYPES[str(dtype)][1]
    require(
        config.get("dtype_bytes") == dtype_bytes
        and config.get("first_sequence") == 1
        and config.get("operation") == "allreduce"
        and config.get("reduction") == "sum"
        and _integer(config.get("epoch"), 1, (1 << 63) - 1, "epoch") >= 1
        and _integer(
            config.get("group_generation"),
            1,
            (1 << 63) - 1,
            "group_generation",
        )
        >= 1,
        "dtype/sequence/collective identity mismatch",
    )
    _hex(config.get("job_uuid"), HEX32, "design job_uuid")
    _hex(config.get("group_uuid"), HEX32, "design group_uuid")
    _hex(
        config.get("model_identity_sha256"),
        HEX64,
        "design model_identity_sha256",
    )
    require(
        count <= MANAGED_MAX_SINGLE_ALLOCATION_BYTES // dtype_bytes,
        "public tensor exceeds the managed single-allocation limit",
    )
    limits = document.get("limits")
    require(
        isinstance(limits, Mapping)
        and limits.get("carrier_max_payload_bytes")
        == CARRIER_MAX_PAYLOAD_BYTES
        and limits.get("managed_max_single_allocation_bytes")
        == MANAGED_MAX_SINGLE_ALLOCATION_BYTES
        and limits.get("device_max_element_count") == DEVICE_MAX_ELEMENT_COUNT
        and limits.get("credits_per_peer") == DESIGN_CREDITS_PER_PEER
        and isinstance(limits.get("sources"), Mapping)
        and "SAGR_CCL_V1_CARRIER_MAX_PAYLOAD_BYTES"
        in str(limits["sources"].get("carrier"))
        and "SAGR_MEMORY_MAX_SINGLE_ALLOCATION_BYTES"
        in str(limits["sources"].get("managed_allocation"))
        and "_MAX_ELEMENT_COUNT" in str(limits["sources"].get("device_index")),
        "source-grounded execution limits drifted",
    )
    coverage = document.get("coverage")
    require(
        isinstance(coverage, Mapping)
        and coverage.get("systematic_world_sizes") == list(SYSTEMATIC_WORLDS)
        and coverage.get("formal_live_acceptance_world_sizes")
        == list(LIVE_ACCEPTANCE_WORLDS)
        and coverage.get("odd_planner_generalization_world_sizes")
        == list(ODD_PLANNER_WORLDS)
        and coverage.get("this_world_is_formal_live_entry")
        is (world in LIVE_ACCEPTANCE_WORLDS),
        "coverage matrix drifted",
    )
    expected_segments = plan_segments(count, world, dtype_bytes)
    segmentation = document.get("segmentation")
    segment_summaries = (
        segmentation.get("segments") if isinstance(segmentation, Mapping) else None
    )
    require(
        isinstance(segmentation, Mapping)
        and segmentation.get("strategy")
        == "contiguous_16mib_private_storage_segments"
        and segmentation.get("segment_element_limit")
        == segment_element_limit(world, dtype_bytes)
        and segmentation.get("segment_count") == len(expected_segments)
        and segmentation.get("sequence_rule")
        == "one descriptor per segment, contiguous 1..K"
        and segmentation.get("commit_rule")
        == "all segments succeed, then one fresh public commit"
        and "segment-sized storage" in str(segmentation.get("storage_rule"))
        and "never pass a full-public-storage view"
        in str(segmentation.get("storage_rule"))
        and isinstance(segment_summaries, list)
        and len(segment_summaries) == len(expected_segments)
        and config.get("last_sequence") == len(expected_segments),
        "segmentation contract mismatch",
    )
    ranks = document.get("ranks")
    require(isinstance(ranks, list) and len(ranks) == world,
            "rank design count mismatch")
    require([rank.get("rank") for rank in ranks] == list(range(world)),
            "rank designs must be exactly 0..world-1")
    validate_rank_launch_group([rank["rank_launch"] for rank in ranks])
    total_launches = 0
    total_zero_sends = 0
    total_nonzero_chunks = 0
    total_zero_chunks = 0
    aggregate_names = (
        "data_records",
        "consumed_records_sent",
        "consumed_records_received",
        "ordered_staging_count",
        "device_sum_launches",
        "zero_reduce_scatter_receives",
        "zero_data_sends",
    )
    rank_aggregates = [{name: 0 for name in aggregate_names} for _ in ranks]
    for rank_number, rank in enumerate(ranks):
        require(
            rank.get("rank_launch_sha256") == object_sha256(rank["rank_launch"])
            and isinstance(rank.get("segments"), list)
            and len(rank["segments"]) == len(expected_segments),
            "rank launch or segment chain mismatch",
        )
    for segment_index, expected_segment in enumerate(expected_segments):
        summary = segment_summaries[segment_index]
        require(
            all(summary.get(name) == value for name, value in expected_segment.items())
            and summary.get("nonzero_chunks")
            == min(expected_segment["element_count"], world)
            and summary.get("zero_chunks")
            == world - min(expected_segment["element_count"], world),
            "segment extent/sequence summary mismatch",
        )
        descriptor_hashes: set[str] = set()
        plan_hashes: list[str] = []
        observed_max_payload = 0
        segment_count = expected_segment["element_count"]
        segment_base = expected_segment["base_offset_elements"]
        for rank_number, rank in enumerate(ranks):
            rank_segment = rank["segments"][segment_index]
            require(
                all(
                    rank_segment.get(name) == value
                    for name, value in expected_segment.items()
                ),
                "rank segment extent/sequence mismatch",
            )
            descriptor_sha256 = rank_segment.get("descriptor_sha256")
            _hex(descriptor_sha256, HEX64, "segment descriptor_sha256")
            descriptor_hashes.add(descriptor_sha256)
            steps = rank_segment.get("steps")
            require(
                isinstance(steps, list)
                and len(steps) == 2 * (world - 1)
                and rank_segment.get("plan_sha256") == object_sha256(steps),
                "segment planner step/hash mismatch",
            )
            plan_hashes.append(rank_segment["plan_sha256"])
            for ordinal, step in enumerate(steps):
                expected_phase = (
                    PHASE_REDUCE_SCATTER
                    if ordinal < world - 1
                    else PHASE_ALL_GATHER
                )
                require(
                    step.get("ordinal") == ordinal
                    and step.get("phase") == expected_phase
                    and step.get("phase_step_index") == ordinal % (world - 1)
                    and step.get("action") == ACTION_SEND_RECEIVE,
                    "planner order/phase/action mismatch",
                )
                require(
                    0 <= step["send_rank"] < world
                    and 0 <= step["receive_rank"] < world
                    and step["send_rank"] != rank_number
                    and step["receive_rank"] != rank_number
                    and 0 <= step["send_chunk"] < world
                    and 0 <= step["receive_chunk"] < world,
                    "planner peer/chunk index is outside the group",
                )
                require(
                    0 <= step["send_offset_elements"] <= segment_count
                    and 0 <= step["receive_offset_elements"] <= segment_count
                    and 0 <= step["send_count_elements"]
                    <= segment_count - step["send_offset_elements"]
                    and 0 <= step["receive_count_elements"]
                    <= segment_count - step["receive_offset_elements"]
                    and step["global_send_offset_elements"]
                    == segment_base + step["send_offset_elements"]
                    and step["global_receive_offset_elements"]
                    == segment_base + step["receive_offset_elements"]
                    and step["global_send_offset_elements"]
                    + step["send_count_elements"]
                    <= count
                    and step["global_receive_offset_elements"]
                    + step["receive_count_elements"]
                    <= count,
                    "planner local/global range escapes its segment",
                )
                require(
                    step["send_payload_bytes"]
                    == step["send_count_elements"] * dtype_bytes
                    and step["receive_payload_bytes"]
                    == step["receive_count_elements"] * dtype_bytes
                    and step["send_payload_bytes"]
                    <= CARRIER_MAX_PAYLOAD_BYTES
                    and step["receive_payload_bytes"]
                    <= CARRIER_MAX_PAYLOAD_BYTES,
                    "planner payload exceeds the carrier contract",
                )
                observed_max_payload = max(
                    observed_max_payload,
                    step["send_payload_bytes"],
                    step["receive_payload_bytes"],
                )
                peer = ranks[step["send_rank"]]["segments"][segment_index][
                    "steps"
                ][ordinal]
                require(
                    peer["receive_rank"] == rank_number
                    and peer["receive_chunk"] == step["send_chunk"]
                    and peer["receive_offset_elements"]
                    == step["send_offset_elements"]
                    and peer["receive_count_elements"]
                    == step["send_count_elements"],
                    "planner peer symmetry mismatch",
                )
            rank_expected = {
                "data_records": len(steps),
                "consumed_records_sent": len(steps),
                "consumed_records_received": len(steps),
                "ordered_staging_count": len(steps),
                "device_sum_launches": sum(
                    step["receive_count_elements"] != 0
                    for step in steps[: world - 1]
                ),
                "zero_reduce_scatter_receives": sum(
                    step["receive_count_elements"] == 0
                    for step in steps[: world - 1]
                ),
                "zero_data_sends": sum(
                    step["send_count_elements"] == 0 for step in steps
                ),
            }
            require(
                rank_segment.get("expected") == rank_expected,
                "segment expected counters mismatch",
            )
            for name in aggregate_names:
                rank_aggregates[rank_number][name] += rank_expected[name]
        require(
            len(descriptor_hashes) == 1
            and summary.get("descriptor_sha256") == next(iter(descriptor_hashes))
            and summary.get("rank_plan_chain_sha256") == object_sha256(plan_hashes)
            and summary.get("maximum_payload_bytes") == observed_max_payload,
            "segment descriptor/plan/payload identity mismatch",
        )
        total_nonzero_chunks += summary["nonzero_chunks"]
        total_zero_chunks += summary["zero_chunks"]
    for rank_number, rank in enumerate(ranks):
        require(
            rank.get("descriptor_chain_sha256")
            == object_sha256(
                [item["descriptor_sha256"] for item in rank["segments"]]
            )
            and rank.get("plan_chain_sha256")
            == object_sha256([item["plan_sha256"] for item in rank["segments"]])
            and rank.get("expected")
            == {
                **rank_aggregates[rank_number],
                "public_commit_count": 1,
                "host_reduction_count": 0,
            },
            "rank segment chain or aggregate counters mismatch",
        )
        total_launches += rank_aggregates[rank_number]["device_sum_launches"]
        total_zero_sends += rank_aggregates[rank_number]["zero_data_sends"]
    group = document.get("group_expected")
    require(
        isinstance(group, Mapping)
        and group.get("segment_count") == len(expected_segments)
        and group.get("planner_steps_per_rank")
        == len(expected_segments) * 2 * (world - 1)
        and group.get("nonzero_chunks") == total_nonzero_chunks
        and group.get("zero_chunks") == total_zero_chunks
        and group.get("device_sum_launches") == total_launches
        == total_nonzero_chunks * (world - 1)
        and group.get("zero_data_sends") == total_zero_sends
        == 2 * total_zero_chunks * (world - 1)
        and group.get("host_reduction_count") == 0
        and group.get("public_commit_count") == 1,
        "group expected counters mismatch",
    )
    state_machine = document.get("step_state_machine")
    require(
        isinstance(state_machine, Mapping)
        and state_machine.get("host_reduction_count") == 0
        and state_machine.get("transfer_tuple_fields")
        == [
            "group",
            "descriptor_sha256",
            "sequence",
            "phase",
            "step_index",
            "chunk_index",
            "source_rank",
            "destination_rank",
            "slot_index",
            "slot_generation",
        ]
        and state_machine.get("step_complete")
        == "outbound and inbound must both be complete"
        and "release_sender_credit" in state_machine.get("outbound", [])
        and "send_exact_matching_CONSUMED" in state_machine.get("inbound", [])
        and "never acknowledge" in str(state_machine.get("device_failure")),
        "step failure/host arithmetic boundary missing",
    )
    architecture = document.get("process_architecture")
    require(
        isinstance(architecture, Mapping)
        and architecture.get("broker") == "NativeCCL.live_broker"
        and architecture.get("capability_creation") == "LiveBroker.prepare_rank"
        and "pass_fds=(capability_fd,)" in str(architecture.get("worker_spawn"))
        and "NativeCCL.process_identity" in str(
            architecture.get("rank_pid_binding")
        )
        and "_ensure_session()" in str(architecture.get("managed_session_start"))
        and "sagr_managed_session_open_v2" in str(
            architecture.get("managed_session_start")
        )
        and architecture.get("peer_table")
        == "NativeCCL.join_rank authenticated ccl_live_v1 FD table"
        and architecture.get("per_rank_daemon") is True
        and architecture.get("per_rank_namespace") is True
        and "bounded TERM/KILL/reap" in str(architecture.get("cleanup")),
        "broker/capability/process/daemon architecture drifted",
    )
    require(
        document.get("claim_boundary")
        == (
            "This document and its host-only state-machine tests are not "
            "device-backed live collective acceptance."
        ),
        "host-only claim boundary drifted",
    )
    evidence_contract = document.get("evidence_contract")
    oracle = (
        evidence_contract.get("oracle")
        if isinstance(evidence_contract, Mapping)
        else None
    )
    verifier = (
        evidence_contract.get("future_formal_verifier")
        if isinstance(evidence_contract, Mapping)
        else None
    )
    require(
        isinstance(oracle, Mapping)
        and oracle.get("timing") == "post_target_only"
        and oracle.get("feedback") is False
        and evidence_contract.get("acceptance_authority") is False
        and evidence_contract.get("live_collective_accepted") is False
        and isinstance(verifier, Mapping)
        and "absent path" in str(verifier.get("evidence_directory"))
        and set(verifier.get("rehash", []))
        == {"rank_result", "dispatch_trace", "gem5_log", "stats"}
        and verifier.get("self_reported_boolean_or_digest_is_authority") is False
        and "retire precedes" in str(verifier.get("timing_binding")),
        "synthetic/formal evidence authority boundary changed",
    )
    return dict(document)


def _validate_cleanup(cleanup: Any, label: str) -> None:
    require(
        isinstance(cleanup, Mapping)
        and cleanup.get("worker_reaped") is True
        and cleanup.get("daemon_reaped") is True
        and cleanup.get("owned_fd_delta") == 0
        and cleanup.get("orphan_count") == 0,
        f"{label} cleanup is incomplete",
    )


def validate_success_evidence(
    _evidence: Mapping[str, Any], design: Mapping[str, Any]
) -> None:
    validate_design(design)
    raise DesignError(
        "formal live success validation is unavailable: an absent-only artifact "
        "directory verifier that rehashes result/trace/log/stats is required"
    )


def validate_synthetic_success_expectation(
    expectation: Mapping[str, Any], design: Mapping[str, Any]
) -> dict[str, Any]:
    validate_design(design)
    require(
        expectation.get("schema") == SUCCESS_EVIDENCE_SCHEMA
        and expectation.get("mode") == "synthetic_expectation"
        and expectation.get("acceptance_authority") is False
        and expectation.get("live_collective_accepted") is False
        and expectation.get("world_size") == design["config"]["world_size"]
        and expectation.get("segment_count")
        == design["group_expected"]["segment_count"]
        and expectation.get("required_device_sum_launches")
        == design["group_expected"]["device_sum_launches"]
        and expectation.get("required_host_reduction_count") == 0
        and expectation.get("required_public_commit_count") == 1
        and expectation.get("formal_artifact_bundle_present") is False,
        "synthetic success expectation is not a fixed non-acceptance contract",
    )
    return dict(expectation)


def _group_document(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_uuid": config["job_uuid"],
        "group_uuid": config["group_uuid"],
        "model_identity_sha256": config["model_identity_sha256"],
        "epoch": config["epoch"],
        "group_generation": config["group_generation"],
        "world_size": config["world_size"],
    }


def transfer_tuple_document(transfer: TransferTuple) -> dict[str, Any]:
    return {
        "group": {
            "job_uuid": transfer.group.job_uuid,
            "group_uuid": transfer.group.group_uuid,
            "model_identity_sha256": transfer.group.model_identity_sha256,
            "epoch": transfer.group.epoch,
            "group_generation": transfer.group.group_generation,
            "world_size": transfer.group.world_size,
        },
        "descriptor_sha256": transfer.descriptor_sha256,
        "sequence": transfer.sequence,
        "phase": transfer.phase,
        "step_index": transfer.step_index,
        "chunk_index": transfer.chunk_index,
        "source_rank": transfer.source_rank,
        "destination_rank": transfer.destination_rank,
        "slot_index": transfer.slot_index,
        "slot_generation": transfer.slot_generation,
    }


def validate_failure_evidence(
    _evidence: Mapping[str, Any], design: Mapping[str, Any]
) -> None:
    validate_design(design)
    raise DesignError(
        "formal live failure validation is unavailable: an absent-only artifact "
        "directory verifier that rehashes result/trace/log/stats is required"
    )


def validate_synthetic_failure_expectation(
    expectation: Mapping[str, Any], design: Mapping[str, Any]
) -> dict[str, Any]:
    validate_design(design)
    config = design["config"]
    world = config["world_size"]
    require(
        expectation.get("schema") == FAILURE_EVIDENCE_SCHEMA
        and expectation.get("mode") == "synthetic_expectation"
        and expectation.get("acceptance_authority") is False
        and expectation.get("live_collective_accepted") is False
        and expectation.get("formal_artifact_bundle_present") is False
        and expectation.get("world_size") == world,
        "failure expectation is not a fixed non-acceptance contract",
    )
    status = expectation.get("status")
    require(status in FAILURE_STATUSES, "failure status is not canonical")
    first_error = expectation.get("canonical_first_error")
    reporter_rank = first_error.get("reporter_rank") if isinstance(
        first_error, Mapping
    ) else None
    require(
        isinstance(first_error, Mapping)
        and first_error.get("status") == status
        and isinstance(reporter_rank, int)
        and not isinstance(reporter_rank, bool)
        and (0 <= reporter_rank < world or reporter_rank == (1 << 32) - 1)
        and _integer(first_error.get("failed_rank"), 0, world - 1, "failed_rank")
        >= 0
        and _integer(first_error.get("context_sequence"), 1, (1 << 63) - 1,
                     "context_sequence") >= 1,
        "canonical first error is invalid",
    )
    if status == "device_failure":
        require(reporter_rank < world, "device failure must identify its reporter")
    if status == "peer_lost":
        require(
            reporter_rank == (1 << 32) - 1,
            "broker-detected peer loss must use the canonical no-rank reporter",
        )
    binding = expectation.get("failure_binding")
    require(isinstance(binding, Mapping), "failure identity binding is missing")
    segment_index = _integer(
        binding.get("segment_index"),
        0,
        design["group_expected"]["segment_count"] - 1,
        "failure segment_index",
    )
    failed_rank = first_error["failed_rank"]
    rank_design = design["ranks"][failed_rank]
    segment = rank_design["segments"][segment_index]
    sequence = segment["sequence"]
    step_index = _integer(
        binding.get("step_index"),
        0,
        2 * (world - 1) - 1,
        "failure step_index",
    )
    step = segment["steps"][step_index]
    transfer = binding.get("transfer_tuple")
    require(isinstance(transfer, Mapping), "failure transfer tuple is missing")
    expected_fixed_transfer = {
        "group": _group_document(config),
        "descriptor_sha256": segment["descriptor_sha256"],
        "sequence": sequence,
        "phase": step["phase"],
        "step_index": step_index,
        "chunk_index": step["receive_chunk"],
        "source_rank": step["receive_rank"],
        "destination_rank": failed_rank,
    }
    require(
        all(transfer.get(name) == value for name, value in expected_fixed_transfer.items())
        and _integer(
            transfer.get("slot_index"),
            0,
            DESIGN_CREDITS_PER_PEER - 1,
            "failure slot_index",
        )
        >= 0
        and _integer(
            transfer.get("slot_generation"),
            1,
            (1 << 63) - 1,
            "failure slot_generation",
        )
        >= 1
        and binding.get("sequence") == sequence
        and first_error.get("context_sequence") == sequence
        and binding.get("descriptor_sha256") == segment["descriptor_sha256"]
        and binding.get("plan_sha256") == segment["plan_sha256"]
        and binding.get("runtime_sha256") == design["runtime"]["sha256"]
        and binding.get("rank_launch_sha256")
        == rank_design["rank_launch_sha256"],
        "failure segment/sequence/planner/runtime/transfer binding mismatch",
    )
    session = binding.get("managed_session")
    require(
        isinstance(session, Mapping)
        and session.get("job_uuid") == config["job_uuid"]
        and session.get("epoch") == config["epoch"]
        and session.get("rank") == failed_rank
        and session.get("world_size") == world
        and isinstance(session.get("daemon_uuid"), str)
        and HEX32.fullmatch(session["daemon_uuid"]) is not None
        and session["daemon_uuid"] != "0" * 32
        and session.get("runtime_sha256") == design["runtime"]["sha256"]
        and binding.get("managed_session_sha256") == object_sha256(session),
        "failure managed session identity mismatch",
    )
    ranks = expectation.get("ranks")
    require(isinstance(ranks, list) and len(ranks) == world,
            "failure expectation rank count mismatch")
    require([rank.get("rank") for rank in ranks] == list(range(world)),
            "failure expectation ranks are not canonical")
    for rank in ranks:
        require(
            rank.get("first_error") == first_error
            and rank.get("failure_binding") == binding
            and rank.get("public_result_published") is False
            and rank.get("public_commit_count") == 0
            and rank.get("host_reduction_count") == 0,
            "failure was not group-wide and commit-atomic",
        )
        _validate_cleanup(rank.get("cleanup"), f"rank {rank['rank']}")
    pending = expectation.get("failed_step")
    require(
        isinstance(pending, Mapping)
        and pending.get("segment_index") == segment_index
        and pending.get("sequence") == sequence
        and pending.get("step_index") == step_index
        and pending.get("transfer_tuple") == transfer
        and pending.get("consumed_ack_sent") is False,
        "failed DATA was acknowledged",
    )
    if status == "device_failure":
        require(
            pending.get("data_consumed_to_immutable_staging") is True
            and pending.get("receive_count_elements", 0) > 0
            and pending.get("device_dispatch_attempted") is True
            and pending.get("device_dispatch_succeeded") is False,
            "device failure evidence does not bind a failed nonzero SUM",
        )
    require(
        expectation.get("target_feedback") is False
        and _integer(expectation.get("started_at_ns"), 1, (1 << 63) - 1,
                     "started_at_ns") > 0
        and _integer(expectation.get("completed_at_ns"), 1, (1 << 63) - 1,
                     "completed_at_ns") >= expectation["started_at_ns"]
        and _integer(expectation.get("absolute_deadline_ns"), 1, (1 << 63) - 1,
                     "absolute_deadline_ns") > expectation["started_at_ns"]
        and expectation.get("deadline_bounded") is True
        and expectation.get("all_cleanup_complete") is True,
        "failure deadline/cleanup/feedback boundary mismatch",
    )
    relation = expectation.get("deadline_relation")
    if status == "timed_out":
        require(
            relation == "expired"
            and expectation["completed_at_ns"] >= expectation["absolute_deadline_ns"],
            "timeout must complete at or after its absolute deadline",
        )
    else:
        require(
            relation in ("before_deadline", "not_asserted"),
            "non-timeout deadline relation is invalid",
        )
        if relation == "before_deadline":
            require(
                expectation["completed_at_ns"]
                < expectation["absolute_deadline_ns"],
                "asserted pre-deadline failure completed too late",
            )
    return dict(expectation)


def deterministic_config(
    world_size: int,
    element_count: int,
    dtype: str,
    *,
    model_identity_sha256: str | None = None,
) -> DesignConfig:
    seed = f"{world_size}:{element_count}:{dtype}".encode("ascii")

    def digest(label: bytes) -> str:
        return hashlib.sha256(label + b":" + seed).hexdigest()

    return DesignConfig(
        world_size=world_size,
        element_count=element_count,
        dtype=dtype,
        epoch=1,
        group_generation=1,
        job_uuid=digest(b"job")[:32],
        group_uuid=digest(b"group")[:32],
        model_identity_sha256=(
            digest(b"model")
            if model_identity_sha256 is None
            else _hex(model_identity_sha256, HEX64, "model_identity_sha256")
        ),
    )


def default_runtime_library() -> Path:
    prefix = os.environ.get("ROCM_SIM_ROOT")
    if prefix:
        return Path(prefix).resolve() / "lib/libself_amdgpu_runtime.so.1"
    return (
        ROOT
        / "projects/self-amdgpu-runtime/build/cp28-runtime-clang"
        / "libself_amdgpu_runtime.so.1"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit a host-only formal design for live device allreduce"
    )
    parser.add_argument("--design-only", action="store_true")
    parser.add_argument("--runtime-library", type=Path)
    parser.add_argument("--namespace-root", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--element-count", type=int, required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPES), required=True)
    parser.add_argument(
        "--model-identity-sha256",
        help="bind the collective group to an external canonical workload identity",
    )
    parser.add_argument(
        "--expected-output",
        type=Path,
        help=(
            "atomically publish the canonical expected wrapper to this absent "
            "path instead of writing a bare design to stdout"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.design_only:
        raise DesignError(
            "live gem5 execution is intentionally unavailable; pass --design-only"
        )
    if args.expected_output is not None and args.runtime_library is None:
        raise DesignError(
            "formal expected publication requires an explicit --runtime-library"
        )
    namespace_root = Path(os.path.abspath(args.namespace_root))
    document = build_design(
        args.runtime_library or default_runtime_library(),
        namespace_root,
        deterministic_config(
            args.world_size,
            args.element_count,
            args.dtype,
            model_identity_sha256=args.model_identity_sha256,
        ),
    )
    if args.expected_output is None:
        sys.stdout.buffer.write(canonical_json(document))
    else:
        publish_expected_wrapper(
            Path(os.path.abspath(args.expected_output)),
            document,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
