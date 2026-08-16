#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""One product-backed rank for the live device allreduce runner.

The public entry point bootstraps into the frozen product before importing
torch, Triton, or gemsim_ccl.  The small transport helpers intentionally stay
importable by the host-only runner tests.
"""

from __future__ import annotations

if __name__ == "__main__":
    __import__("runpy").run_path(
        __file__.replace("ccl_live_allreduce_rank.py", "_gemsim_bootstrap.py")
    )["bootstrap"](__file__, "ccl-live-allreduce")

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
WORKER_RESULT_SCHEMA = "amdgpu-sim.ccl-live-allreduce-rank-result.v1"
STEP_EVENT_SCHEMA = "amdgpu-sim.ccl-live-allreduce-step-event.v1"
WORKER_CONFIG_SCHEMA = "amdgpu-sim.ccl-live-allreduce-rank-config.v1"
EXPECTED_SCHEMA = "amdgpu-sim.ccl-live-allreduce-expected.v1"
CARRIER_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
NO_RANK = (1 << 32) - 1
DEADLINE_GUARD_NS = 1_000_000
PHASE_REDUCE_SCATTER = 1
PHASE_ALL_GATHER = 2


class RankError(RuntimeError):
    def __init__(self, message: str, *, status: int = 8, failed_rank: int | None = None):
        super().__init__(message)
        self.status = status
        self.failed_rank = failed_rank


def canonical_json(value: object) -> bytes:
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_bytes(tensor: Any) -> bytes:
    return tensor.detach().contiguous().view(__import__("torch").uint8).numpy().tobytes()


def _plan_document(step_ordinal: int, step: Any, segment_base: int, dtype_bytes: int) -> dict[str, int]:
    return {
        "ordinal": step_ordinal,
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
        "global_send_offset_elements": segment_base + int(step.send_offset_elements),
        "global_receive_offset_elements": segment_base + int(step.receive_offset_elements),
        "send_payload_bytes": int(step.send_count_elements) * dtype_bytes,
        "receive_payload_bytes": int(step.receive_count_elements) * dtype_bytes,
    }


def record_document(record: Any) -> dict[str, Any]:
    descriptor_sha256 = record.descriptor_sha256
    if isinstance(descriptor_sha256, str):
        normalized_descriptor_sha256 = descriptor_sha256
    else:
        normalized_descriptor_sha256 = bytes(descriptor_sha256).hex()
    return {
        "descriptor_sha256": normalized_descriptor_sha256,
        "sequence": int(record.sequence),
        "phase": int(record.phase),
        "step_index": int(record.step_index),
        "chunk_index": int(record.chunk_index),
        "source_rank": int(record.source_rank),
        "destination_rank": int(record.destination_rank),
        "slot_index": int(record.slot_index),
        "slot_generation": int(record.slot_generation),
    }


class StepJournal:
    """Durable ordinal event stream used by the external trace verifier."""

    def __init__(self, path: Path, rank: int) -> None:
        self.path = Path(path)
        self.rank = rank
        self.ordinal = 0
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self._fd = os.open(self.path, flags, 0o600)

    def event(
        self,
        name: str,
        *,
        segment: Mapping[str, Any] | None = None,
        step_ordinal: int | None = None,
        step: Any | None = None,
        record: Any | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": STEP_EVENT_SCHEMA,
            "ordinal": self.ordinal,
            "monotonic_ns": time.monotonic_ns(),
            "rank": self.rank,
            "event": name,
            "segment_id": None if segment is None else int(segment["segment_id"]),
            "descriptor_sequence": None if segment is None else int(segment["sequence"]),
            "step_ordinal": step_ordinal,
        }
        if step is not None:
            value.update(
                {
                    "phase": int(step.phase),
                    "phase_step_index": int(step.step_index),
                }
            )
            value["planner"] = {
                "send_rank": int(step.send_rank),
                "receive_rank": int(step.receive_rank),
                "send_chunk": int(step.send_chunk),
                "receive_chunk": int(step.receive_chunk),
                "send_offset_elements": int(step.send_offset_elements),
                "send_count_elements": int(step.send_count_elements),
                "receive_offset_elements": int(step.receive_offset_elements),
                "receive_count_elements": int(step.receive_count_elements),
            }
        if record is not None:
            value["transfer"] = record_document(record)
        value.update(details)
        payload = canonical_json(value)
        offset = 0
        while offset < len(payload):
            offset += os.write(self._fd, payload[offset:])
        os.fsync(self._fd)
        self.ordinal += 1
        return value

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "StepJournal":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


class EngineEvidenceObserver:
    """Validate engine observations and preserve the external evidence schema.

    Expected design bytes are used only as a fail-closed observer.  They are
    never returned to the collective engine or its device executor.
    """

    _STEP_EVENTS = (
        "outbound_prepared",
        "outbound_DATA_sent",
        "inbound_DATA_received",
        "inbound_staged",
        "device_call_enter",
        "device_call_returned",
        "zero_no_dispatch",
        "copy_complete",
        "inbound_CONSUMED_send_attempt",
        "inbound_CONSUMED_sent",
        "outbound_CONSUMED_received_credit_released",
        "step_complete",
    )
    _TRANSFER_EVENTS = frozenset(
        {
            "outbound_prepared",
            "outbound_DATA_sent",
            "inbound_DATA_received",
            "inbound_staged",
            "inbound_CONSUMED_send_attempt",
            "inbound_CONSUMED_sent",
            "outbound_CONSUMED_received_credit_released",
        }
    )

    def __init__(
        self,
        *,
        journal: StepJournal,
        segments: Sequence[Mapping[str, Any]],
        dtype_bytes: int,
        world_size: int,
        execution_context: dict[str, Any],
    ) -> None:
        self.journal = journal
        self.segments = tuple(dict(item) for item in segments)
        self.dtype_bytes = dtype_bytes
        self.world_size = world_size
        self.execution_context = execution_context
        self.segment_results: list[dict[str, Any]] = []
        self.counters = {
            "data_sent_count": 0,
            "data_received_count": 0,
            "consumed_sent_count": 0,
            "consumed_received_count": 0,
            "device_reduction_launch_count": 0,
            "host_reduction_count": 0,
            "public_commit_count": 0,
        }
        self._collective_started = False
        self._active_segment: Mapping[str, Any] | None = None
        self._active_descriptor_sha256: str | None = None
        self._plan_documents: list[dict[str, int]] = []
        self._step_names: list[str] = []
        self._step_ordinal: int | None = None
        self._public_commit: tuple[str, int] | None = None
        self._outbound_transfer: Any | None = None

    @staticmethod
    def _segment_document(segment: Any) -> dict[str, int]:
        return {
            "segment_id": int(segment.index),
            "sequence": int(segment.sequence),
            "global_offset_elements": int(segment.offset_elements),
            "element_count": int(segment.element_count),
            "byte_count": int(segment.byte_count),
        }

    def _expected_segment(self, segment: Any) -> Mapping[str, Any]:
        observed = self._segment_document(segment)
        index = observed["segment_id"]
        if index != len(self.segment_results) or index >= len(self.segments):
            raise RankError("engine segment order differs from expected design")
        expected = self.segments[index]
        for name, value in observed.items():
            if int(expected[name]) != value:
                raise RankError(
                    f"engine segment {name} differs from expected design"
                )
        return expected

    def _expected_step_names(self, step: Any) -> tuple[str, ...]:
        receive_count = int(step.receive_count_elements)
        if int(step.phase) == PHASE_REDUCE_SCATTER:
            middle = (
                ("device_call_enter", "device_call_returned")
                if receive_count
                else ("zero_no_dispatch",)
            )
        elif int(step.phase) == PHASE_ALL_GATHER:
            middle = ("copy_complete",)
        else:
            raise RankError("engine observer saw a non-allreduce phase")
        return (
            "outbound_prepared",
            "outbound_DATA_sent",
            "inbound_DATA_received",
            "inbound_staged",
            *middle,
            "inbound_CONSUMED_send_attempt",
            "inbound_CONSUMED_sent",
            "outbound_CONSUMED_received_credit_released",
            "step_complete",
        )

    def _observe_step(self, event: Any) -> None:
        if self._active_segment is None or event.segment is None:
            raise RankError("engine step event has no active segment")
        expected_segment = self._expected_segment(event.segment)
        step_ordinal = int(event.step_ordinal)
        step = event.step
        if step is None or not 0 <= step_ordinal < 2 * (self.world_size - 1):
            raise RankError("engine step event is missing exact planner identity")
        if self._step_ordinal != step_ordinal:
            if self._step_names:
                raise RankError("engine advanced before the prior step completed")
            if step_ordinal != len(self._plan_documents):
                raise RankError("engine planner step order differs from design")
            self._step_ordinal = step_ordinal
            self._plan_documents.append(
                _plan_document(
                    step_ordinal,
                    step,
                    int(expected_segment["global_offset_elements"]),
                    self.dtype_bytes,
                )
            )
        expected_names = self._expected_step_names(step)
        observed_position = len(self._step_names)
        if (
            observed_position >= len(expected_names)
            or event.name != expected_names[observed_position]
        ):
            raise RankError("engine step event sequence differs from evidence contract")
        self._step_names.append(event.name)

        record = event.transfer
        if event.name in self._TRANSFER_EVENTS:
            if record is None:
                raise RankError("engine transfer event omitted its transfer tuple")
            if (
                record.descriptor_sha256 != self._active_descriptor_sha256
                or int(record.sequence) != int(expected_segment["sequence"])
            ):
                raise RankError("engine transfer identity differs from active segment")
        elif record is not None:
            raise RankError("engine non-transfer event carried a transfer tuple")

        details: dict[str, Any] = {}
        if event.name == "inbound_staged":
            if event.payload_sha256 is None or event.byte_count is None:
                raise RankError("engine staging event omitted immutable payload evidence")
            details.update(
                {
                    "staging_sha256": event.payload_sha256,
                    "immutable_bytes": True,
                    "payload_bytes": int(event.byte_count),
                }
            )
        elif event.name == "copy_complete":
            if event.byte_count is None:
                raise RankError("engine copy event omitted its byte extent")
            details["byte_count"] = int(event.byte_count)

        self.journal.event(
            event.name,
            segment=expected_segment,
            step_ordinal=step_ordinal,
            step=step,
            record=record,
            **details,
        )
        if event.name == "outbound_prepared":
            self.execution_context["step_ordinal"] = step_ordinal
            self.execution_context["failed_transfer"] = record_document(record)
            self._outbound_transfer = record
        elif event.name == "outbound_DATA_sent":
            self.counters["data_sent_count"] += 1
        elif event.name == "inbound_DATA_received":
            self.counters["data_received_count"] += 1
            self.execution_context["failed_transfer"] = record_document(record)
        elif event.name == "device_call_returned":
            self.counters["device_reduction_launch_count"] += 1
        elif event.name == "inbound_CONSUMED_sent":
            self.counters["consumed_sent_count"] += 1
            if self._outbound_transfer is None:
                raise RankError("engine lost the outstanding outbound transfer")
            self.execution_context["failed_transfer"] = record_document(
                self._outbound_transfer
            )
        elif event.name == "outbound_CONSUMED_received_credit_released":
            self.counters["consumed_received_count"] += 1
        elif event.name == "step_complete":
            if tuple(self._step_names) != expected_names:
                raise RankError("engine step completed with incomplete evidence")
            self.execution_context["failed_transfer"] = None
            self.execution_context["failed_ack_sent"] = False
            self._outbound_transfer = None
            self._step_names = []
            self._step_ordinal = None

    def __call__(self, event: Any) -> None:
        if event.name == "collective_started":
            if self._collective_started or event.segment is None:
                raise RankError("engine emitted an invalid collective start")
            self._expected_segment(event.segment)
            self._collective_started = True
            return
        if event.name == "segment_started":
            if (
                not self._collective_started
                or self._active_segment is not None
                or event.segment is None
                or event.descriptor_sha256 is None
            ):
                raise RankError("engine emitted an invalid segment start")
            expected = self._expected_segment(event.segment)
            if event.descriptor_sha256 != expected["descriptor_sha256"]:
                raise RankError("engine descriptor digest differs from expected design")
            self._active_segment = expected
            self._active_descriptor_sha256 = event.descriptor_sha256
            self._plan_documents = []
            self.execution_context.update(
                {
                    "descriptor_sha256": event.descriptor_sha256,
                    "segment_id": int(expected["segment_id"]),
                    "sequence": int(expected["sequence"]),
                    "step_ordinal": None,
                    "failed_transfer": None,
                    "failed_ack_sent": False,
                }
            )
            return
        if event.name in self._STEP_EVENTS:
            self._observe_step(event)
            return
        if event.name == "segment_complete":
            if (
                self._active_segment is None
                or event.segment is None
                or self._step_names
                or event.descriptor_sha256 != self._active_descriptor_sha256
            ):
                raise RankError("engine emitted an invalid segment completion")
            expected = self._expected_segment(event.segment)
            expected_step_count = 2 * (self.world_size - 1)
            plan_sha256 = sha256_bytes(canonical_json(self._plan_documents))
            if (
                len(self._plan_documents) != expected_step_count
                or plan_sha256 != expected["plan_sha256"]
            ):
                raise RankError("engine planner digest differs from expected design")
            self.segment_results.append(
                {
                    "segment_id": int(expected["segment_id"]),
                    "global_offset_elements": int(
                        expected["global_offset_elements"]
                    ),
                    "element_count": int(expected["element_count"]),
                    "sequence": int(expected["sequence"]),
                    "descriptor_sha256": self._active_descriptor_sha256,
                    "plan_sha256": plan_sha256,
                    "plan_step_count": len(self._plan_documents),
                }
            )
            self._active_segment = None
            self._active_descriptor_sha256 = None
            self._plan_documents = []
            return
        if event.name == "public_commit":
            if (
                self._active_segment is not None
                or len(self.segment_results) != len(self.segments)
                or self._public_commit is not None
                or event.payload_sha256 is None
                or event.byte_count is None
            ):
                raise RankError("engine emitted an invalid public commit")
            self._public_commit = (
                event.payload_sha256,
                int(event.byte_count),
            )
            return
        if event.name in {"collective_failed", "closed"}:
            return
        raise RankError(f"unsupported engine evidence event: {event.name}")

    def publish_public_commit(self, output_payload: bytes) -> None:
        observed = (sha256_bytes(output_payload), len(output_payload))
        if self._public_commit != observed:
            raise RankError("engine public commit differs from persisted output")
        self.journal.event(
            "public_commit",
            output_sha256=observed[0],
            output_bytes=observed[1],
        )
        self.counters["public_commit_count"] = 1


def _remaining_timeout_ns(native: Any, absolute_deadline_ns: int, label: str) -> int:
    remaining = absolute_deadline_ns - native.monotonic_time_ns() - DEADLINE_GUARD_NS
    if remaining <= 0:
        raise RankError(f"collective deadline expired before {label}", status=13)
    return remaining


def _file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _wait_busy(
    action: Callable[[], Any],
    *,
    live_rank: Any,
    native: Any,
    absolute_deadline_ns: int,
    label: str,
    is_complete: Callable[[Any], bool],
) -> Any:
    """Retry one nonblocking carrier operation and poll live abort each turn."""
    while True:
        value = action()
        if is_complete(value):
            return value
        first_error = live_rank.poll_abort()
        if first_error is not None:
            error = RankError(
                f"{label} observed live group abort status={int(first_error.status)}",
                status=int(first_error.status),
                failed_rank=int(first_error.failed_rank),
            )
            error.first_error = first_error
            raise error
        if native.monotonic_time_ns() >= absolute_deadline_ns:
            raise RankError(f"{label} exceeded the absolute deadline", status=13)
        time.sleep(0.001)


def _tensor_from_immutable(payload: bytes, dtype: Any, count: int) -> Any:
    torch = __import__("torch")
    if len(payload) != count * dtype.itemsize:
        raise RankError("immutable carrier staging extent mismatch")
    if count == 0:
        return torch.empty(0, dtype=dtype)
    # bytearray owns a disjoint mutable backing; clone removes frombuffer aliasing.
    return torch.frombuffer(bytearray(payload), dtype=dtype, count=count).clone()


def execute_segments(
    *,
    native: Any,
    identity: Any,
    live_rank: Any,
    carrier: Any,
    executor: Any,
    input_tensor: Any,
    segments: Sequence[Mapping[str, Any]],
    dtype_code: int,
    phase_reduce_scatter: int,
    phase_all_gather: int,
    message_consumed: int,
    operation_all_reduce: int,
    reduction_sum: int,
    rank: int,
    absolute_deadline_ns: int,
    journal: StepJournal,
    execution_context: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]], dict[str, int]]:
    """Run every segment privately; the caller owns the one public commit."""
    torch = __import__("torch")
    input_before = input_tensor.clone()
    committed_segments: list[Any] = []
    segment_results: list[dict[str, Any]] = []
    counters = {
        "data_sent_count": 0,
        "data_received_count": 0,
        "consumed_sent_count": 0,
        "consumed_received_count": 0,
        "device_reduction_launch_count": 0,
        "host_reduction_count": 0,
        "public_commit_count": 0,
    }
    for segment in segments:
        offset = int(segment["global_offset_elements"])
        count = int(segment["element_count"])
        if count * input_tensor.element_size() > CARRIER_MAX_PAYLOAD_BYTES:
            raise RankError("segment workspace exceeds the 16 MiB carrier bound")
        workspace = input_tensor.narrow(0, offset, count).clone()
        descriptor = native.descriptor(
            identity,
            sequence=int(segment["sequence"]),
            input_count=count,
            output_count=count,
            rank=rank,
            operation=operation_all_reduce,
            reduction=reduction_sum,
            dtype=dtype_code,
        )
        plan = native.plan(descriptor)
        descriptor_sha256 = native.descriptor_sha256(descriptor)
        plan_document = [
            _plan_document(step_ordinal, step, offset, input_tensor.element_size())
            for step_ordinal, step in enumerate(plan)
        ]
        plan_sha256 = sha256_bytes(canonical_json(plan_document))
        if segment.get("descriptor_sha256") not in (None, descriptor_sha256):
            raise RankError("segment descriptor digest differs from expected design")
        if segment.get("plan_sha256") not in (None, plan_sha256):
            raise RankError("segment planner digest differs from expected design")
        execution_context.update(
            {
                "descriptor": descriptor,
                "descriptor_sha256": descriptor_sha256,
                "segment_id": int(segment["segment_id"]),
                "sequence": int(segment["sequence"]),
                "step_ordinal": None,
                "failed_transfer": None,
                "failed_ack_sent": False,
            }
        )
        for step_ordinal, step in enumerate(plan):
            execution_context["step_ordinal"] = step_ordinal
            execution_context["failed_transfer"] = None
            execution_context["failed_ack_sent"] = False
            send_view = workspace.narrow(
                0, int(step.send_offset_elements), int(step.send_count_elements)
            )
            outbound = carrier.prepare_data(descriptor, step_ordinal, tensor_bytes(send_view))
            execution_context["failed_transfer"] = record_document(outbound)
            journal.event(
                "outbound_prepared", segment=segment, step_ordinal=step_ordinal,
                step=step, record=outbound,
            )
            send_socket = live_rank.peer_socket(int(step.send_rank))
            _wait_busy(
                lambda: carrier.send_data(send_socket, outbound),
                live_rank=live_rank, native=native,
                absolute_deadline_ns=absolute_deadline_ns,
                label="outbound DATA send", is_complete=bool,
            )
            counters["data_sent_count"] += 1
            journal.event(
                "outbound_DATA_sent", segment=segment, step_ordinal=step_ordinal,
                step=step, record=outbound,
            )

            receive_socket = live_rank.peer_socket(int(step.receive_rank))
            inbound = _wait_busy(
                lambda: carrier.receive(
                    receive_socket, descriptor, step_ordinal, int(step.receive_rank)
                ),
                live_rank=live_rank, native=native,
                absolute_deadline_ns=absolute_deadline_ns,
                label="inbound DATA receive", is_complete=lambda value: value is not None,
            )
            counters["data_received_count"] += 1
            execution_context["failed_transfer"] = record_document(inbound)
            journal.event(
                "inbound_DATA_received", segment=segment, step_ordinal=step_ordinal,
                step=step, record=inbound,
            )
            payload, inbound_consumed = carrier.consume(
                descriptor, step_ordinal, inbound
            )
            immutable = bytes(payload)
            receive_count = int(step.receive_count_elements)
            staging = _tensor_from_immutable(immutable, input_tensor.dtype, receive_count)
            journal.event(
                "inbound_staged", segment=segment, step_ordinal=step_ordinal,
                step=step, record=inbound,
                staging_sha256=sha256_bytes(immutable), immutable_bytes=True,
            )

            destination = workspace.narrow(
                0, int(step.receive_offset_elements), receive_count
            )
            if int(step.phase) == phase_reduce_scatter:
                if receive_count:
                    journal.event(
                        "device_call_enter", segment=segment,
                        step_ordinal=step_ordinal, step=step,
                    )
                    executor.sum_in_place(
                        destination, staging, element_count=receive_count
                    )
                    counters["device_reduction_launch_count"] += 1
                    journal.event(
                        "device_call_returned", segment=segment,
                        step_ordinal=step_ordinal, step=step,
                    )
                else:
                    journal.event(
                        "zero_no_dispatch", segment=segment,
                        step_ordinal=step_ordinal, step=step,
                    )
            elif int(step.phase) == phase_all_gather:
                if receive_count:
                    destination.view(torch.uint8).copy_(staging.view(torch.uint8))
                journal.event(
                    "copy_complete", segment=segment, step_ordinal=step_ordinal,
                    step=step, byte_count=len(immutable),
                )
            else:
                raise RankError("planner returned a non-allreduce phase")

            journal.event(
                "inbound_CONSUMED_send_attempt", segment=segment,
                step_ordinal=step_ordinal, step=step, record=inbound_consumed,
            )
            _wait_busy(
                lambda: carrier.send_consumed(receive_socket, inbound_consumed),
                live_rank=live_rank, native=native,
                absolute_deadline_ns=absolute_deadline_ns,
                label="inbound CONSUMED send", is_complete=bool,
            )
            counters["consumed_sent_count"] += 1
            journal.event(
                "inbound_CONSUMED_sent", segment=segment,
                step_ordinal=step_ordinal, step=step, record=inbound_consumed,
            )
            # The inbound transfer is now ACKed.  The outstanding transfer is
            # our outbound DATA, whose matching CONSUMED has not arrived yet.
            execution_context["failed_transfer"] = record_document(outbound)
            execution_context["failed_ack_sent"] = False
            acknowledged = _wait_busy(
                lambda: carrier.receive(
                    send_socket, descriptor, step_ordinal, int(step.send_rank)
                ),
                live_rank=live_rank, native=native,
                absolute_deadline_ns=absolute_deadline_ns,
                label="outbound CONSUMED receive",
                is_complete=lambda value: value is not None,
            )
            if int(acknowledged.kind) != message_consumed:
                raise RankError("outbound DATA did not receive matching CONSUMED")
            counters["consumed_received_count"] += 1
            journal.event(
                "outbound_CONSUMED_received_credit_released", segment=segment,
                step_ordinal=step_ordinal, step=step, record=acknowledged,
            )
            journal.event(
                "step_complete", segment=segment, step_ordinal=step_ordinal,
                step=step,
            )
            execution_context["failed_transfer"] = None
            execution_context["failed_ack_sent"] = False

        info = carrier.info()
        if any(
            int(value) != 0
            for value in (
                info.sender_inflight, info.receiver_ready, info.receiver_consumed
            )
        ):
            raise RankError("carrier retained ownership at segment completion")
        committed_segments.append(workspace)
        segment_results.append(
            {
                "segment_id": int(segment["segment_id"]),
                "global_offset_elements": offset,
                "element_count": count,
                "sequence": int(segment["sequence"]),
                "descriptor_sha256": descriptor_sha256,
                "plan_sha256": plan_sha256,
                "plan_step_count": len(plan),
            }
        )

    if not torch.equal(input_tensor, input_before):
        raise RankError("public input changed before collective commit")
    output = torch.cat(committed_segments).clone() if committed_segments else input_tensor.clone()
    if output.data_ptr() == input_tensor.data_ptr():
        raise RankError("public result aliases the input")
    return output, segment_results, counters


def _read_config(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RankError("rank configuration is invalid JSON") from error
    if not isinstance(value, dict) or value.get("schema") != WORKER_CONFIG_SCHEMA:
        raise RankError("rank configuration schema mismatch")
    if data != canonical_json(value):
        raise RankError("rank configuration is not canonical JSON")
    return value


def _exclusive_bytes(path: Path, payload: bytes, mode: int = 0o600) -> None:
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


def _managed_session_record(driver: Any, rank_launch_sha256: str) -> dict[str, Any]:
    runtime = driver.runtime
    info = runtime.session_info
    if info is None or not runtime.session.value:
        raise RankError("managed simulator session is unavailable")
    return {
        "child_pid": int(info.child_pid),
        "connection_id": int(info.connection_id),
        "epoch": int(info.epoch),
        "rank": int(info.rank),
        "world_size": int(info.world_size),
        "daemon_uuid": bytes(info.daemon_uuid).hex(),
        "job_uuid": bytes(info.job_uuid).hex(),
        "runtime_library": str(Path(runtime.path).resolve(strict=True)),
        "rank_launch_sha256": rank_launch_sha256,
    }


def _relay_abort(
    *,
    carrier: Any,
    live_rank: Any,
    native: Any,
    descriptor: Any | None,
    rank: int,
    world_size: int,
    status: int,
    failed_rank: int,
    sequence: int,
    deadline: int,
) -> dict[str, Any] | None:
    # A CCLStatusError may carry a CarrierRecord in ``first_error``.  That
    # object is valid only for peer transport relay; the canonical group error
    # must come from the live control plane.
    error_record = getattr(sys.exc_info()[1], "abort_record", None)
    abort_record = (
        error_record
        if error_record is not None
        and hasattr(error_record, "descriptor_sha256")
        and hasattr(error_record, "kind")
        else None
    )
    first = None
    try:
        candidate = live_rank.poll_abort()
        if candidate is not None and hasattr(candidate, "context_sequence"):
            first = candidate
    except Exception:
        pass
    if abort_record is None and descriptor is not None:
        try:
            abort_record = carrier.abort(descriptor, failed_rank, status)
        except Exception:
            try:
                abort_record = carrier.get_abort()
            except Exception:
                abort_record = None
    if abort_record is not None:
        for peer in range(world_size):
            if peer == rank:
                continue
            try:
                socket_descriptor = live_rank.peer_socket(peer)
                _wait_busy(
                    lambda fd=socket_descriptor: carrier.send_abort(fd, abort_record),
                    live_rank=live_rank, native=native,
                    absolute_deadline_ns=deadline, label="carrier ABORT relay",
                    is_complete=bool,
                )
            except Exception:
                pass
    try:
        _wait_busy(
            lambda: live_rank.report_abort(failed_rank, status, sequence),
            live_rank=live_rank, native=native,
            absolute_deadline_ns=deadline, label="live abort report",
            is_complete=bool,
        )
    except Exception:
        pass
    if first is None:
        while native.monotonic_time_ns() < deadline:
            try:
                candidate = live_rank.poll_abort()
                first = (
                    candidate
                    if candidate is not None
                    and hasattr(candidate, "context_sequence")
                    else None
                )
            except Exception:
                first = None
            if first is not None:
                break
            time.sleep(0.001)
    if first is not None:
        return {
            "reporter_rank": int(first.reporter_rank),
            "failed_rank": int(first.failed_rank),
            "status": _status_label(int(first.status)),
            "native_status": int(first.status),
            "context_sequence": int(first.context_sequence),
        }
    return None


def _status_label(status: int) -> str:
    if status == 13:
        return "timed_out"
    if status == 14:
        return "peer_lost"
    return "device_failure"


def run_live_rank(args: argparse.Namespace) -> int:
    import torch
    import triton
    from gemsim_ccl import (
        AllReduceEngine,
        DeviceSumExecutor,
        GroupSpec,
        RankBootstrap,
    )
    from gemsim_ccl.native import (
        NativeCCL,
        PROTOCOL_ERROR,
    )

    config = _read_config(args.config)
    rank = int(config["rank"])
    world_size = int(config["world_size"])
    result_path = Path(config["result_path"])
    journal_path = Path(config["journal_path"])
    input_path = Path(config["input_path"])
    output_path = Path(config["output_path"])
    started = time.monotonic_ns()
    result: dict[str, Any] = {
        "schema": WORKER_RESULT_SCHEMA,
        "status": "failed",
        "rank": rank,
        "world_size": world_size,
        "started_monotonic_ns": started,
        "absolute_deadline_ns": int(config["absolute_deadline_ns"]),
        "completed_monotonic_ns": None,
        "live_collective_accepted": False,
        "acceptance_authority": False,
        "public_result_published": False,
        "public_commit_count": 0,
        "failed_transfer": None,
        "error": None,
        "first_error": None,
        "product": config["product"],
    }
    native = None
    engine = None
    capability_owned = True
    execution_context: dict[str, Any] = {
        "descriptor": None,
        "descriptor_sha256": None,
        "segment_id": None,
        "sequence": 1,
        "step_ordinal": None,
        "failed_transfer": None,
        "failed_ack_sent": False,
    }
    journal = StepJournal(journal_path, rank)
    try:
        prefix = Path(os.environ["ROCM_SIM_ROOT"]).resolve(strict=True)
        runtime_path = (prefix / "lib/libself_amdgpu_runtime.so.1").resolve(strict=True)
        if str(runtime_path) != config["runtime_library"]:
            raise RankError("worker runtime path differs from product snapshot")

        dtype_name = config["dtype"]
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
        indices = torch.arange(int(config["element_count"]), dtype=torch.int64)
        values = (((indices * 13 + rank * 29) % 127) - 63).to(torch.float32) / 16.0
        public_input = values.to(dtype).contiguous()
        input_payload = tensor_bytes(public_input)
        _exclusive_bytes(input_path, input_payload)
        result["input_sha256_before"] = sha256_bytes(input_payload)
        result["input_sha256_after"] = sha256_bytes(input_payload)

        driver = triton.runtime.driver.active
        driver.runtime._ensure_session()
        managed = _managed_session_record(driver, config["rank_launch_sha256"])
        if (
            managed["rank"] != rank
            or managed["world_size"] != world_size
            or managed["epoch"] != int(config["epoch"])
            or managed["job_uuid"] != config["job_uuid"]
        ):
            raise RankError("managed session exact topology mismatch")

        native = NativeCCL(runtime_path)
        engine_module = __import__("gemsim_ccl.engine", fromlist=["engine"])
        if _file_record(Path(engine_module.__file__)) != config["product"][
            "ccl_engine"
        ]:
            raise RankError("executed allreduce engine differs from product snapshot")
        group = GroupSpec(
            world_size=world_size,
            epoch=int(config["epoch"]),
            group_generation=int(config["group_generation"]),
            job_uuid=bytes.fromhex(config["job_uuid"]),
            group_uuid=bytes.fromhex(config["group_uuid"]),
            model_identity_sha256=bytes.fromhex(config["model_identity_sha256"]),
        )
        bootstrap = RankBootstrap(
            rank=rank,
            capability_fd=args.capability_fd,
            broker_pid=int(config["broker_pid"]),
            broker_start_time_ticks=int(config["broker_start_time_ticks"]),
            absolute_deadline_ns=int(config["absolute_deadline_ns"]),
            credits_per_peer=int(config["credits_per_peer"]),
        )
        observer = EngineEvidenceObserver(
            journal=journal,
            segments=config["segments"],
            dtype_bytes=public_input.element_size(),
            world_size=world_size,
            execution_context=execution_context,
        )
        executor = DeviceSumExecutor()
        # AllReduceEngine owns the inherited capability on entry, including
        # every exception raised by join itself.
        capability_owned = False
        engine = AllReduceEngine.join(
            group,
            bootstrap,
            native=native,
            executor=executor,
            observer=observer,
        )
        output = engine.all_reduce(
            public_input,
            timeout_ns=_remaining_timeout_ns(
                native,
                int(config["absolute_deadline_ns"]),
                "execution",
            ),
        )
        engine.close(
            timeout_ns=_remaining_timeout_ns(
                native,
                int(config["absolute_deadline_ns"]),
                "close",
            )
        )
        output_payload = tensor_bytes(output)
        _exclusive_bytes(output_path, output_payload)
        observer.publish_public_commit(output_payload)
        segment_results = observer.segment_results
        counters = observer.counters
        result.update(
            {
                "status": "success",
                "managed_session": managed,
                "runtime": {
                    "path": str(runtime_path),
                    "sha256": native.library_sha256,
                    "version": native.runtime_version,
                    "abi_version": native.abi_version,
                },
                "segments": segment_results,
                "input_sha256_before": sha256_bytes(input_payload),
                "input_sha256_after": sha256_bytes(tensor_bytes(public_input)),
                "output_sha256": sha256_bytes(output_payload),
                "output_storage_fresh": output.data_ptr() != public_input.data_ptr(),
                "public_result_published": True,
                "public_commit_count": 1,
                "counters": counters,
            }
        )
        return_code = 0
    except BaseException as error:
        status = int(getattr(error, "status", PROTOCOL_ERROR))
        result["status"] = _status_label(status)
        failed_rank = getattr(error, "failed_rank", None)
        failed_rank = rank if failed_rank is None or failed_rank == NO_RANK else int(failed_rank)
        first_error = getattr(error, "first_error", None)
        if first_error is not None and hasattr(first_error, "context_sequence"):
            result["first_error"] = {
                "reporter_rank": int(first_error.reporter_rank),
                "failed_rank": int(first_error.failed_rank),
                "status": _status_label(int(first_error.status)),
                "native_status": int(first_error.status),
                "context_sequence": int(first_error.context_sequence),
            }
        result["failed_transfer"] = execution_context["failed_transfer"]
        result["failed_descriptor_sequence"] = int(execution_context["sequence"])
        result["failed_ack_sent"] = bool(execution_context["failed_ack_sent"])
        if "input_sha256_before" not in result:
            result["input_sha256_before"] = None
            result["input_sha256_after"] = None
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "status": status,
            "failed_rank": failed_rank,
        }
        return_code = 1
    finally:
        if engine is not None:
            try:
                engine.destroy()
            except Exception:
                pass
        if capability_owned:
            try:
                os.close(args.capability_fd)
            except OSError:
                pass
        if not output_path.exists():
            _exclusive_bytes(output_path, b"")
        journal.close()
        result["completed_monotonic_ns"] = time.monotonic_ns()
        try:
            _exclusive_bytes(result_path, canonical_json(result))
        except FileExistsError:
            return_code = 1
    return return_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--capability-fd", type=int, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_live_rank(parse_args()))
