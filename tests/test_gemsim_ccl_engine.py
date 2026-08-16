from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "plugins/collectives/gemsim_ccl/src"
sys.path.insert(0, str(PACKAGE))

from gemsim_ccl.engine import (  # noqa: E402
    AllReduceEngine,
    CARRIER_MAX_PAYLOAD_BYTES,
    CollectiveEvent,
    CollectiveTimeoutError,
    EngineError,
    EngineForkError,
    EngineState,
    EngineStateError,
    GroupSpec,
    MANAGED_MAX_SINGLE_ALLOCATION_BYTES,
    RankBootstrap,
    SequenceExhaustedError,
    plan_allreduce_segments,
)
from gemsim_ccl.native import (  # noqa: E402
    DTYPE_BF16,
    DTYPE_FP32,
    LIVE_PHASE_READY,
    MESSAGE_CONSUMED,
    MESSAGE_DATA,
    PHASE_ALL_GATHER,
    PHASE_REDUCE_SCATTER,
    PROTOCOL_ERROR,
    SEQUENCE_MISMATCH,
    TIMED_OUT,
    UINT64_MAX,
    PlannedStep,
)


def group_spec(world_size: int = 2) -> GroupSpec:
    return GroupSpec(
        world_size=world_size,
        epoch=7,
        group_generation=3,
        job_uuid=bytes.fromhex("01" * 16),
        group_uuid=bytes.fromhex("02" * 16),
        model_identity_sha256=bytes.fromhex("03" * 32),
    )


def rank_bootstrap(rank: int = 0) -> RankBootstrap:
    return RankBootstrap(
        rank=rank,
        capability_fd=73,
        broker_pid=1234,
        broker_start_time_ticks=5678,
        absolute_deadline_ns=999_999,
        credits_per_peer=2,
    )


@dataclasses.dataclass
class FakeRecord:
    descriptor_sha256: bytes = b"\x11" * 32
    sequence: int = 1
    slot_generation: int = 1
    payload_bytes: int = 0
    kind: int = MESSAGE_DATA
    phase: int = PHASE_REDUCE_SCATTER
    step_index: int = 0
    chunk_index: int = 0
    source_rank: int = 1
    destination_rank: int = 0
    slot_index: int = 0
    status: int = 0
    failed_rank: int = (1 << 32) - 1
    payload: bytes = b""


class FakeLiveRank:
    def __init__(self, group: GroupSpec, rank: int) -> None:
        self.group = group
        self.rank = rank
        self.reported: list[tuple[int, int, int]] = []
        self.destroy_count = 0
        self.close_deadlines: list[int] = []
        self.polled_abort = None

    def info(self):
        identity = SimpleNamespace(
            world_size=self.group.world_size,
            epoch=self.group.epoch,
            group_generation=self.group.group_generation,
            job_uuid=self.group.job_uuid,
            group_uuid=self.group.group_uuid,
            model_identity_sha256=self.group.model_identity_sha256,
        )
        return SimpleNamespace(
            phase=LIVE_PHASE_READY,
            self_rank=self.rank,
            world_size=self.group.world_size,
            group=identity,
        )

    def peer_socket(self, peer: int) -> int:
        if peer == self.rank:
            raise ValueError("self is not a peer")
        return 100 + peer

    def poll_abort(self):
        return self.polled_abort

    def report_abort(self, failed_rank: int, status: int, sequence: int) -> bool:
        self.reported.append((failed_rank, status, sequence))
        return True

    def close(self, deadline: int) -> None:
        self.close_deadlines.append(deadline)

    def destroy(self) -> None:
        self.destroy_count += 1


class FakeExecutor:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def sum_in_place(self, destination, source, *, element_count: int):
        self.calls += 1
        if self.fail:
            error = EngineError("injected device failure")
            error.status = PROTOCOL_ERROR
            raise error
        destination[:element_count].add_(source[:element_count])
        return destination


class FakeCarrier:
    def __init__(self) -> None:
        self.prepared: dict[tuple[int, int], bytes] = {}
        self.receive_counts: dict[tuple[int, int], int] = {}
        self.send_data_count = 0
        self.send_consumed_count = 0
        self.send_abort_count = 0
        self.abort_count = 0
        self.close_count = 0
        self.never_complete_send = False
        self.bad_ack = False

    @staticmethod
    def _record(descriptor, step_index: int, kind: int, payload: bytes):
        phase = (
            PHASE_REDUCE_SCATTER if step_index == 0 else PHASE_ALL_GATHER
        )
        return FakeRecord(
            sequence=descriptor.sequence,
            payload_bytes=len(payload),
            kind=kind,
            phase=phase,
            step_index=step_index,
            chunk_index=0,
            source_rank=1,
            destination_rank=0,
            payload=payload,
        )

    def prepare_data(self, descriptor, step_index: int, payload: bytes):
        self.prepared[(descriptor.sequence, step_index)] = payload
        return self._record(descriptor, step_index, MESSAGE_DATA, payload)

    def send_data(self, _socket: int, _record: FakeRecord) -> bool:
        self.send_data_count += 1
        return not self.never_complete_send

    def receive(self, _socket: int, descriptor, step_index: int, _peer: int):
        key = (descriptor.sequence, step_index)
        call = self.receive_counts.get(key, 0)
        self.receive_counts[key] = call + 1
        if call == 0:
            dtype = (
                torch.bfloat16
                if descriptor.dtype == DTYPE_BF16
                else torch.float32
            )
            if step_index == 0:
                payload = (
                    torch.ones(descriptor.input_count, dtype=dtype)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
            else:
                payload = self.prepared[key]
            return self._record(descriptor, step_index, MESSAGE_DATA, payload)
        kind = MESSAGE_DATA if self.bad_ack else MESSAGE_CONSUMED
        return self._record(descriptor, step_index, kind, b"")

    def consume(self, descriptor, step_index: int, record: FakeRecord):
        consumed = self._record(descriptor, step_index, MESSAGE_CONSUMED, b"")
        return record.payload, consumed

    def send_consumed(self, _socket: int, _record: FakeRecord) -> bool:
        self.send_consumed_count += 1
        return True

    def abort(self, descriptor, failed_rank: int, reason: int):
        self.abort_count += 1
        record = self._record(descriptor, 0, 3, b"")
        record.status = reason
        record.failed_rank = failed_rank
        return record

    def get_abort(self):
        raise RuntimeError("no prior abort")

    def send_abort(self, _socket: int, _record: FakeRecord) -> bool:
        self.send_abort_count += 1
        return True

    def info(self):
        return SimpleNamespace(
            sender_inflight=0,
            receiver_ready=0,
            receiver_consumed=0,
        )

    def close(self) -> None:
        self.close_count += 1


class FakeNative:
    def __init__(self, group: GroupSpec, rank: int = 0) -> None:
        self.group = group
        self.rank = rank
        self.live_rank = FakeLiveRank(group, rank)
        self.carrier = FakeCarrier()
        self.descriptors = []
        self.join_capabilities: list[int] = []
        self._clock = 10_000

    def identity(self, **values):
        if values != dataclasses.asdict(self.group):
            raise AssertionError("engine changed the exact group identity")
        return SimpleNamespace(**values)

    def process_identity(self, pid: int):
        return SimpleNamespace(pid=pid, start_time_ticks=5678)

    def join_rank(self, capability, _identity, rank, _owner, _deadline):
        self.join_capabilities.append(capability)
        if rank != self.rank:
            raise AssertionError("wrong rank joined")
        return self.live_rank

    def carrier_session(self, _identity, rank: int, credits: int):
        if rank != self.rank or credits != 2:
            raise AssertionError("wrong carrier configuration")
        return self.carrier

    def descriptor(self, _identity, **values):
        descriptor = SimpleNamespace(**values)
        self.descriptors.append(descriptor)
        return descriptor

    def plan(self, descriptor):
        count = descriptor.input_count
        return (
            PlannedStep(
                phase=PHASE_REDUCE_SCATTER,
                action=3,
                step_index=0,
                send_rank=1,
                receive_rank=1,
                send_chunk=0,
                receive_chunk=0,
                send_offset_elements=0,
                send_count_elements=count,
                receive_offset_elements=0,
                receive_count_elements=count,
            ),
            PlannedStep(
                phase=PHASE_ALL_GATHER,
                action=3,
                step_index=0,
                send_rank=1,
                receive_rank=1,
                send_chunk=0,
                receive_chunk=0,
                send_offset_elements=0,
                send_count_elements=count,
                receive_offset_elements=0,
                receive_count_elements=count,
            ),
        )

    def descriptor_sha256(self, descriptor) -> str:
        return f"{descriptor.sequence:064x}"

    def deadline_after(self, timeout_ns: int) -> int:
        return self._clock + timeout_ns

    def monotonic_time_ns(self) -> int:
        self._clock += 10
        return self._clock


def joined_engine(
    *,
    executor: FakeExecutor | None = None,
    observer=None,
) -> tuple[AllReduceEngine, FakeNative, FakeExecutor]:
    group = group_spec()
    native = FakeNative(group)
    actual_executor = executor or FakeExecutor()
    engine = AllReduceEngine.join(
        group,
        rank_bootstrap(),
        native=native,
        executor=actual_executor,
        observer=observer,
    )
    return engine, native, actual_executor


class EngineConfigurationTest(unittest.TestCase):
    def test_group_and_bootstrap_are_strict_before_native_side_effects(self):
        with self.assertRaises(ValueError):
            group_spec(1)
        with self.assertRaises(ValueError):
            dataclasses.replace(group_spec(), epoch=0)
        with self.assertRaises(ValueError):
            dataclasses.replace(group_spec(), job_uuid=b"\0" * 16)
        with self.assertRaises(TypeError):
            dataclasses.replace(group_spec(), group_uuid=bytearray(16))
        with self.assertRaises(ValueError):
            dataclasses.replace(rank_bootstrap(), credits_per_peer=17)
        with self.assertRaises(ValueError):
            dataclasses.replace(rank_bootstrap(), absolute_deadline_ns=0)

        native = FakeNative(group_spec())
        with mock.patch("gemsim_ccl.engine.os.close") as close_fd:
            with self.assertRaises(ValueError):
                AllReduceEngine.join(
                    group_spec(),
                    dataclasses.replace(rank_bootstrap(), rank=2),
                    native=native,
                    executor=FakeExecutor(),
                )
        close_fd.assert_called_once_with(73)
        self.assertEqual(native.join_capabilities, [])

    def test_broker_start_identity_is_checked_before_join(self):
        native = FakeNative(group_spec())
        native.process_identity = lambda pid: SimpleNamespace(
            pid=pid, start_time_ticks=1
        )
        with mock.patch("gemsim_ccl.engine.os.close") as close_fd:
            with self.assertRaisesRegex(EngineError, "start-time"):
                AllReduceEngine.join(
                    group_spec(),
                    rank_bootstrap(),
                    native=native,
                    executor=FakeExecutor(),
                )
        close_fd.assert_called_once_with(73)
        self.assertEqual(native.join_capabilities, [])

    def test_capability_ownership_transfers_at_join_call_boundary(self):
        native = FakeNative(group_spec())
        native.process_identity = mock.Mock(side_effect=RuntimeError("identity"))
        with mock.patch("gemsim_ccl.engine.os.close") as close_fd:
            with self.assertRaisesRegex(RuntimeError, "identity"):
                AllReduceEngine.join(
                    group_spec(),
                    rank_bootstrap(),
                    native=native,
                    executor=FakeExecutor(),
                )
        close_fd.assert_called_once_with(73)

        native = FakeNative(group_spec())
        native.join_rank = mock.Mock(side_effect=RuntimeError("join entered"))
        with mock.patch("gemsim_ccl.engine.os.close") as close_fd:
            with self.assertRaisesRegex(RuntimeError, "join entered"):
                AllReduceEngine.join(
                    group_spec(),
                    rank_bootstrap(),
                    native=native,
                    executor=FakeExecutor(),
                )
        close_fd.assert_not_called()

        native = FakeNative(group_spec())
        with mock.patch("gemsim_ccl.engine.os.close") as close_fd:
            engine = AllReduceEngine.join(
                group_spec(),
                rank_bootstrap(),
                native=native,
                executor=FakeExecutor(),
            )
        close_fd.assert_not_called()
        engine.destroy()

    def test_segmentation_is_contiguous_bounded_and_sequence_checked(self):
        for world_size in range(2, 17):
            segments = plan_allreduce_segments(17, world_size, "float32")
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0].element_count, 17)
        for dtype, itemsize in (("bfloat16", 2), ("float32", 4)):
            with self.subTest(dtype=dtype):
                limit = CARRIER_MAX_PAYLOAD_BYTES // itemsize
                segments = plan_allreduce_segments(
                    limit + 3, 3, dtype, first_sequence=41
                )
                self.assertEqual(
                    segments,
                    (
                        dataclasses.replace(
                            segments[0],
                            index=0,
                            sequence=41,
                            offset_elements=0,
                            element_count=limit,
                            byte_count=CARRIER_MAX_PAYLOAD_BYTES,
                        ),
                        dataclasses.replace(
                            segments[1],
                            index=1,
                            sequence=42,
                            offset_elements=limit,
                            element_count=3,
                            byte_count=3 * itemsize,
                        ),
                    ),
                )
        with self.assertRaises(ValueError):
            plan_allreduce_segments(0, 2, "float32")
        with self.assertRaises(TypeError):
            plan_allreduce_segments(1, 2, "float16")
        with self.assertRaises(ValueError):
            plan_allreduce_segments(
                MANAGED_MAX_SINGLE_ALLOCATION_BYTES // 4 + 1,
                2,
                "float32",
            )
        with self.assertRaises(SequenceExhaustedError):
            plan_allreduce_segments(
                CARRIER_MAX_PAYLOAD_BYTES // 4 + 1,
                2,
                "float32",
                first_sequence=UINT64_MAX - 1,
            )


class EngineSuccessTest(unittest.TestCase):
    def test_singleton_is_fresh_and_uses_no_transport(self):
        events: list[CollectiveEvent] = []
        engine = AllReduceEngine.singleton(observer=events.append)
        value = torch.arange(6, dtype=torch.float32).view(2, 3)
        result = engine.all_reduce(value)
        self.assertTrue(torch.equal(result, value))
        self.assertEqual(result.shape, value.shape)
        self.assertNotEqual(result.data_ptr(), value.data_ptr())
        self.assertEqual(engine.next_sequence, 1)
        self.assertEqual([item.name for item in events], [
            "collective_started", "public_commit"
        ])
        engine.close()
        self.assertEqual(engine.state, EngineState.CLOSED)

    def test_bf16_and_fp32_are_out_of_place_and_sequences_are_monotonic(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                events: list[CollectiveEvent] = []
                engine, native, executor = joined_engine(observer=events.append)
                value = torch.arange(1, 7, dtype=torch.float32).to(dtype).view(2, 3)
                before = value.clone()
                first = engine.all_reduce(value)
                second = engine.all_reduce(value)
                expected = before + torch.ones_like(before)
                self.assertTrue(torch.equal(first, expected))
                self.assertTrue(torch.equal(second, expected))
                self.assertTrue(torch.equal(value, before))
                self.assertNotEqual(first.data_ptr(), value.data_ptr())
                self.assertEqual(first.shape, value.shape)
                self.assertEqual(
                    [item.sequence for item in native.descriptors], [1, 2]
                )
                self.assertEqual(engine.next_sequence, 3)
                self.assertEqual(executor.calls, 2)
                self.assertEqual(native.carrier.send_data_count, 4)
                self.assertEqual(native.carrier.send_consumed_count, 4)
                self.assertEqual(
                    sum(item.name == "public_commit" for item in events), 2
                )
                self.assertTrue(all(dataclasses.is_dataclass(item) for item in events))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    events[0].name = "changed"  # type: ignore[misc]

    def test_close_is_graceful_and_idempotent(self):
        engine, native, _ = joined_engine()
        engine.close(timeout_ns=50)
        engine.close(timeout_ns=50)
        self.assertEqual(engine.state, EngineState.CLOSED)
        self.assertEqual(native.live_rank.close_deadlines, [10_050])
        self.assertEqual(native.carrier.close_count, 1)
        with self.assertRaises(EngineStateError):
            engine.all_reduce(torch.ones(1))

    def test_exceptional_context_exit_aborts_the_live_group(self):
        engine, native, _ = joined_engine()
        with self.assertRaisesRegex(RuntimeError, "application failure"):
            with engine:
                raise RuntimeError("application failure")
        self.assertEqual(engine.state, EngineState.ABORTED)
        self.assertEqual(
            native.live_rank.reported, [(0, PROTOCOL_ERROR, 0)]
        )
        self.assertEqual(native.carrier.close_count, 1)

    def test_reentry_is_rejected_without_corrupting_outer_call(self):
        nested_errors = []
        engine_ref = []

        def observer(event: CollectiveEvent) -> None:
            if event.name == "collective_started" and not nested_errors:
                try:
                    engine_ref[0].all_reduce(torch.ones(1))
                except BaseException as error:
                    nested_errors.append(error)

        engine, _, _ = joined_engine(observer=observer)
        engine_ref.append(engine)
        result = engine.all_reduce(torch.ones(1))
        self.assertTrue(torch.equal(result, torch.full((1,), 2.0)))
        self.assertEqual(len(nested_errors), 1)
        self.assertIsInstance(nested_errors[0], EngineStateError)
        self.assertEqual(engine.state, EngineState.READY)


class EngineFailureTest(unittest.TestCase):
    def test_device_failure_aborts_without_consumed_or_public_commit(self):
        events: list[CollectiveEvent] = []
        engine, native, _ = joined_engine(
            executor=FakeExecutor(fail=True), observer=events.append
        )
        value = torch.arange(4, dtype=torch.float32)
        before = value.clone()
        with self.assertRaisesRegex(EngineError, "device failure"):
            engine.all_reduce(value)
        self.assertTrue(torch.equal(value, before))
        self.assertEqual(engine.state, EngineState.ABORTED)
        self.assertEqual(native.carrier.send_consumed_count, 0)
        self.assertEqual(native.carrier.abort_count, 1)
        self.assertEqual(native.carrier.send_abort_count, 1)
        self.assertEqual(native.live_rank.reported, [(0, PROTOCOL_ERROR, 1)])
        self.assertEqual(native.carrier.close_count, 1)
        self.assertEqual(native.live_rank.destroy_count, 1)
        self.assertNotIn("public_commit", [item.name for item in events])
        with self.assertRaises(EngineStateError):
            engine.all_reduce(value)

    def test_timeout_poisoning_preserves_input_and_reports_exact_status(self):
        engine, native, _ = joined_engine()
        native.carrier.never_complete_send = True
        value = torch.ones(2, dtype=torch.float32)
        with self.assertRaises(CollectiveTimeoutError):
            engine.all_reduce(value, timeout_ns=1)
        self.assertTrue(torch.equal(value, torch.ones_like(value)))
        self.assertEqual(engine.state, EngineState.ABORTED)
        self.assertEqual(native.live_rank.reported, [(0, TIMED_OUT, 1)])
        self.assertEqual(native.carrier.send_consumed_count, 0)

    def test_observer_failure_is_a_collective_failure(self):
        def observer(event: CollectiveEvent) -> None:
            if event.name == "inbound_staged":
                raise RuntimeError("observer failed")

        engine, native, _ = joined_engine(observer=observer)
        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            engine.all_reduce(torch.ones(2))
        self.assertEqual(engine.state, EngineState.ABORTED)
        self.assertEqual(native.carrier.send_consumed_count, 0)
        self.assertEqual(native.carrier.abort_count, 1)

    def test_bad_ack_fails_closed_after_inbound_ack(self):
        engine, native, _ = joined_engine()
        native.carrier.bad_ack = True
        with self.assertRaisesRegex(EngineError, "matching CONSUMED"):
            engine.all_reduce(torch.ones(2))
        self.assertEqual(engine.state, EngineState.ABORTED)
        self.assertEqual(native.carrier.send_consumed_count, 1)
        self.assertEqual(native.carrier.abort_count, 1)

    def test_explicit_abort_is_terminal_and_close_reaches_closed(self):
        engine, native, _ = joined_engine()
        engine.abort(PROTOCOL_ERROR, failed_rank=1, timeout_ns=50)
        self.assertEqual(engine.state, EngineState.ABORTED)
        self.assertEqual(native.live_rank.reported, [(1, PROTOCOL_ERROR, 0)])
        self.assertEqual(native.carrier.close_count, 1)
        self.assertEqual(native.live_rank.destroy_count, 1)
        with self.assertRaises(EngineStateError):
            engine.all_reduce(torch.ones(1))
        engine.close()
        self.assertEqual(engine.state, EngineState.CLOSED)

    def test_sequence_exhaustion_aborts_the_epoch(self):
        engine, native, _ = joined_engine()
        engine._next_sequence = UINT64_MAX - 1
        engine.all_reduce(torch.ones(1))
        self.assertEqual(native.descriptors[-1].sequence, UINT64_MAX - 1)
        with self.assertRaisesRegex(SequenceExhaustedError, "exhausted"):
            engine.all_reduce(torch.ones(1))
        self.assertEqual(engine.state, EngineState.ABORTED)
        self.assertEqual(
            native.live_rank.reported[-1], (0, SEQUENCE_MISMATCH, 0)
        )

    def test_fork_and_parallel_entry_fail_closed(self):
        engine, native, _ = joined_engine()
        with mock.patch(
            "gemsim_ccl.engine.os.getpid", return_value=engine._owner_pid + 1
        ):
            with self.assertRaises(EngineForkError):
                engine.all_reduce(torch.ones(1))
        self.assertEqual(engine.state, EngineState.ABORTED)
        engine.destroy()
        self.assertEqual(native.carrier.close_count, 1)

        second, _, _ = joined_engine()
        second._call_lock.acquire()
        try:
            with self.assertRaisesRegex(EngineStateError, "caller-serialized"):
                second.all_reduce(torch.ones(1))
        finally:
            second._call_lock.release()
            second.destroy()


if __name__ == "__main__":
    unittest.main()
