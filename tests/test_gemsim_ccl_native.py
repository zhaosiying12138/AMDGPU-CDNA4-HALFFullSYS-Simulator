from __future__ import annotations

import ctypes
import fcntl
from pathlib import Path
import os
import select
import signal
import socket
import struct
import sys
import time
import traceback
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "plugins/collectives/gemsim_ccl/src"
sys.path.insert(0, str(PACKAGE))

from gemsim_ccl.native import (  # noqa: E402
    BUSY,
    CHECKSUM_ERROR,
    CCLStatusError,
    DTYPE_BF16,
    INVALID_ARGUMENT,
    LIVE_PHASE_CLOSED,
    LIVE_PHASE_READY,
    LiveAbort,
    LiveBrokerInfo,
    LiveRank,
    LiveRankInfo,
    MESSAGE_CONSUMED,
    NativeCCL,
    NO_RANK,
    OP_ALL_REDUCE,
    PEER_LOST,
    ProcessIdentity,
    REDUCTION_SUM,
)


TEST_TIMEOUT_NS = 10_000_000_000
RING_PACKET = struct.Struct("!IIII16s")
RING_MAGIC = 0x52494E47


def current_runtime() -> Path:
    configured = os.environ.get("GEMSIM_CCL_RUNTIME")
    candidate = Path(configured) if configured else (
        ROOT
        / "projects/self-amdgpu-runtime/build/cp28-runtime-clang"
        / "libself_amdgpu_runtime.so.1"
    )
    if not candidate.is_file():
        raise RuntimeError(f"current CCL runtime build is unavailable: {candidate}")
    return candidate


def open_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def wait_fd(descriptor: int, event: int, deadline_ns: int) -> None:
    poller = select.poll()
    poller.register(descriptor, event)
    while True:
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            raise TimeoutError(f"timed out waiting for fd {descriptor}")
        timeout_ms = max(1, (remaining_ns + 999_999) // 1_000_000)
        observed = poller.poll(timeout_ms)
        if not observed:
            continue
        returned = observed[0][1]
        if returned & event:
            return
        raise OSError(f"fd {descriptor} returned poll events 0x{returned:x}")


def send_packet(descriptor: int, payload: bytes, deadline_ns: int) -> None:
    while True:
        try:
            sent = os.write(descriptor, payload)
        except BlockingIOError:
            wait_fd(descriptor, select.POLLOUT, deadline_ns)
            continue
        if sent != len(payload):
            raise OSError(f"short seqpacket write: {sent} != {len(payload)}")
        return


def receive_packet(descriptor: int, size: int, deadline_ns: int) -> bytes:
    while True:
        try:
            payload = os.read(descriptor, size + 1)
        except BlockingIOError:
            wait_fd(descriptor, select.POLLIN, deadline_ns)
            continue
        if len(payload) != size:
            raise OSError(f"invalid seqpacket size: {len(payload)} != {size}")
        return payload


def child_exit(function, *arguments) -> None:
    try:
        function(*arguments)
    except BaseException:
        traceback.print_exc()
        os._exit(100)
    os._exit(0)


def reap_children(children: list[int]) -> None:
    for pid in children:
        waited, status = os.waitpid(pid, 0)
        if waited != pid or not os.WIFEXITED(status) or os.WEXITSTATUS(status):
            raise AssertionError(f"rank child {pid} failed with wait status {status}")
    children.clear()


def terminate_children(children: list[int]) -> None:
    for pid in children:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for pid in children:
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    children.clear()


class NativeCCLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = NativeCCL(current_runtime())

    def identity(self, world: int):
        return self.runtime.identity(
            world_size=world,
            epoch=7,
            group_generation=3,
            job_uuid=bytes.fromhex("01" * 16),
            group_uuid=bytes.fromhex("02" * 16),
            model_identity_sha256=bytes.fromhex("03" * 32),
        )

    @staticmethod
    def assert_fd_properties(descriptor: int) -> None:
        if descriptor < 0:
            raise AssertionError("expected an open descriptor")
        if not fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC:
            raise AssertionError("descriptor is not CLOEXEC")
        if not fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_NONBLOCK:
            raise AssertionError("descriptor is not nonblocking")
        duplicate = socket.socket(fileno=os.dup(descriptor))
        try:
            if duplicate.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != \
                    socket.SOCK_SEQPACKET:
                raise AssertionError("descriptor is not SOCK_SEQPACKET")
            if hasattr(socket, "SO_DOMAIN") and duplicate.getsockopt(
                socket.SOL_SOCKET, socket.SO_DOMAIN
            ) != socket.AF_UNIX:
                raise AssertionError("descriptor is not AF_UNIX")
        finally:
            duplicate.close()

    @classmethod
    def success_rank_child(
        cls,
        broker,
        capability: int,
        identity,
        owner: ProcessIdentity,
        rank_number: int,
    ) -> None:
        broker.destroy()
        descriptors_before = open_fd_count()
        rank = cls.runtime.join_rank(
            capability,
            identity,
            rank_number,
            owner,
            cls.runtime.deadline_after(TEST_TIMEOUT_NS),
        )
        info = rank.info()
        if info.phase != LIVE_PHASE_READY or info.self_rank != rank_number:
            raise AssertionError("rank did not reach READY with its bound rank")
        if info.world_size != identity.world_size:
            raise AssertionError("rank table world size is incorrect")
        cls.assert_fd_properties(info.control_socket)
        for peer in range(len(info.peer_sockets)):
            descriptor = int(info.peer_sockets[peer])
            if peer == rank_number or peer >= info.world_size:
                if descriptor != -1:
                    raise AssertionError("self/out-of-world peer FD must be -1")
            else:
                cls.assert_fd_properties(descriptor)
                if rank.peer_socket(peer) != descriptor:
                    raise AssertionError("peer_socket did not return borrowed FD")

        next_rank = (rank_number + 1) % info.world_size
        previous_rank = (rank_number + info.world_size - 1) % info.world_size
        deadline = cls.runtime.deadline_after(TEST_TIMEOUT_NS)
        payload = RING_PACKET.pack(
            RING_MAGIC,
            rank_number,
            next_rank,
            info.world_size,
            bytes(info.group.group_uuid),
        )
        send_packet(rank.peer_socket(next_rank), payload, deadline)
        received = receive_packet(
            rank.peer_socket(previous_rank), RING_PACKET.size, deadline
        )
        expected = RING_PACKET.pack(
            RING_MAGIC,
            previous_rank,
            rank_number,
            info.world_size,
            bytes(info.group.group_uuid),
        )
        if received != expected:
            raise AssertionError("rank received the wrong peer packet")
        borrowed = [info.control_socket]
        borrowed.extend(
            int(info.peer_sockets[peer])
            for peer in range(info.world_size)
            if peer != rank_number
        )
        rank.close(cls.runtime.deadline_after(TEST_TIMEOUT_NS))
        if rank.handle.value is not None:
            raise AssertionError("rank close did not clear the native handle")
        for descriptor in borrowed:
            try:
                fcntl.fcntl(descriptor, fcntl.F_GETFD)
            except OSError:
                pass
            else:
                raise AssertionError("rank destroy left a borrowed FD open")
        if open_fd_count() != descriptors_before - 1:
            raise AssertionError("rank close leaked an owned descriptor")

    @classmethod
    def abort_rank_child(
        cls,
        broker,
        capability: int,
        identity,
        owner: ProcessIdentity,
        rank_number: int,
        sync_read: int,
        sync_write: int,
    ) -> None:
        broker.destroy()
        os.close(sync_read)
        descriptors_before = open_fd_count()
        rank = cls.runtime.join_rank(
            capability,
            identity,
            rank_number,
            owner,
            cls.runtime.deadline_after(TEST_TIMEOUT_NS),
        )
        if rank.poll_abort() is not None:
            raise AssertionError("quiet rank poll must report BUSY as None")
        if rank_number == 0:
            if not rank.report_abort(1, CHECKSUM_ERROR, 7):
                raise AssertionError("rank 0 abort report unexpectedly returned BUSY")
        elif rank_number == 1:
            if not rank.report_abort(2, PEER_LOST, 8):
                raise AssertionError("rank 1 abort report unexpectedly returned BUSY")
        os.write(sync_write, b"\xa5")
        os.close(sync_write)

        deadline = cls.runtime.deadline_after(TEST_TIMEOUT_NS)
        first_error = None
        while first_error is None:
            first_error = rank.poll_abort()
            if first_error is None:
                if cls.runtime.monotonic_time_ns() >= deadline:
                    raise TimeoutError("rank did not receive broker abort")
                time.sleep(0.001)
        if (
            first_error.reporter_rank != 0
            or first_error.failed_rank != 1
            or first_error.status != CHECKSUM_ERROR
            or first_error.context_sequence != 7
        ):
            raise AssertionError("rank received the wrong immutable first error")
        if bytes(rank.poll_abort()) != bytes(first_error):
            raise AssertionError("rank first error changed after being latched")
        rank.destroy()
        if open_fd_count() != descriptors_before - 2:
            raise AssertionError("aborted rank leaked capability/control/peer FDs")

    def test_planner_is_available_for_every_world_size(self) -> None:
        for world in range(2, 17):
            identity = self.identity(world)
            hashes = set()
            for rank in range(world):
                descriptor = self.runtime.descriptor(
                    identity,
                    sequence=1,
                    input_count=world + 1,
                    output_count=world + 1,
                    rank=rank,
                    operation=OP_ALL_REDUCE,
                    reduction=REDUCTION_SUM,
                    dtype=DTYPE_BF16,
                )
                plan = self.runtime.plan(descriptor)
                self.assertEqual(len(plan), 2 * (world - 1))
                hashes.add(self.runtime.descriptor_sha256(descriptor))
            self.assertEqual(len(hashes), 1)

    def test_runtime_and_integer_identity_fail_closed(self) -> None:
        self.assertEqual(self.runtime.runtime_version, "0.8.0")
        self.assertEqual(self.runtime.abi_version, (1 << 16) | 8)
        self.assertEqual(len(self.runtime.library_sha256), 64)
        invalid_identity = {
            "world_size": 1 << 32,
            "epoch": -1,
            "group_generation": True,
        }
        defaults = {
            "world_size": 2,
            "epoch": 1,
            "group_generation": 1,
            "job_uuid": bytes.fromhex("01" * 16),
            "group_uuid": bytes.fromhex("02" * 16),
            "model_identity_sha256": bytes.fromhex("03" * 32),
        }
        for name, value in invalid_identity.items():
            arguments = dict(defaults)
            arguments[name] = value
            with self.assertRaises(ValueError):
                self.runtime.identity(**arguments)

        identity = self.identity(2)
        for name, value in (
            ("sequence", -1),
            ("input_count", 1 << 64),
            ("output_count", True),
            ("rank", 1 << 32),
        ):
            arguments = {
                "sequence": 1,
                "input_count": 3,
                "output_count": 3,
                "rank": 0,
                "operation": OP_ALL_REDUCE,
                "reduction": REDUCTION_SUM,
                "dtype": DTYPE_BF16,
            }
            arguments[name] = value
            with self.assertRaises(ValueError):
                self.runtime.descriptor(identity, **arguments)

    def test_live_layout_symbols_types_and_path_are_strict(self) -> None:
        self.assertEqual(BUSY, 11)
        self.assertEqual(self.runtime.library_path, current_runtime().resolve())
        self.assertEqual(ctypes.sizeof(ProcessIdentity), 40)
        self.assertEqual(ProcessIdentity.start_time_ticks.offset, 16)
        self.assertEqual(ctypes.sizeof(LiveAbort), 160)
        self.assertEqual(LiveAbort.group.offset, 8)
        self.assertEqual(LiveAbort.context_sequence.offset, 120)
        self.assertEqual(ctypes.sizeof(LiveRankInfo), 216)
        self.assertEqual(LiveRankInfo.peer_sockets.offset, 136)
        self.assertEqual(ctypes.sizeof(LiveBrokerInfo), 96)
        self.assertEqual(LiveBrokerInfo.owner.offset, 40)
        live_symbols = (
            "sagr_ccl_live_v1_monotonic_time_ns",
            "sagr_ccl_live_v1_process_identity",
            "sagr_ccl_live_v1_broker_create",
            "sagr_ccl_live_v1_broker_info",
            "sagr_ccl_live_v1_broker_prepare_rank",
            "sagr_ccl_live_v1_broker_bind_rank",
            "sagr_ccl_live_v1_broker_rendezvous",
            "sagr_ccl_live_v1_broker_progress",
            "sagr_ccl_live_v1_broker_abort",
            "sagr_ccl_live_v1_broker_first_error",
            "sagr_ccl_live_v1_broker_destroy",
            "sagr_ccl_live_v1_rank_join",
            "sagr_ccl_live_v1_rank_info",
            "sagr_ccl_live_v1_rank_report_abort",
            "sagr_ccl_live_v1_rank_poll_abort",
            "sagr_ccl_live_v1_rank_close",
            "sagr_ccl_live_v1_rank_destroy",
        )
        void_symbols = {
            "sagr_ccl_live_v1_broker_destroy",
            "sagr_ccl_live_v1_rank_destroy",
        }
        for symbol in live_symbols:
            function = getattr(self.runtime.lib, symbol)
            self.assertIsNotNone(function.argtypes)
            if symbol in void_symbols:
                self.assertIsNone(function.restype)
            else:
                self.assertIsNotNone(function.restype)

        identity = self.identity(2)
        with self.assertRaises(TypeError):
            self.runtime.live_broker(object())
        with self.assertRaises(ValueError):
            self.runtime.deadline_after(True)
        with self.assertRaises(ValueError):
            self.runtime.process_identity(True)
        with self.assertRaises(ValueError):
            self.runtime.join_rank(-1, identity, 0, ProcessIdentity(), 1)
        with self.runtime.live_broker(identity) as broker:
            self.assertEqual(bytes(broker.owner), bytes(
                self.runtime.process_identity(os.getpid())
            ))
            self.assertIsNone(broker.first_error())
            with self.assertRaises(ValueError):
                broker.prepare_rank(True)
            with self.assertRaises(TypeError):
                broker.bind_rank(0, object())
            with self.assertRaises(ValueError):
                broker.rendezvous(0)
            capability = broker.prepare_rank(0)
            try:
                self.assertGreaterEqual(capability, 0)
                self.assertTrue(
                    fcntl.fcntl(capability, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
                )
                with self.assertRaises(TypeError):
                    self.runtime.join_rank(
                        capability,
                        object(),
                        0,
                        broker.owner,
                        self.runtime.deadline_after(TEST_TIMEOUT_NS),
                    )
                self.assertGreaterEqual(fcntl.fcntl(capability, fcntl.F_GETFD), 0)
                with self.assertRaises(CCLStatusError) as raised:
                    self.runtime.join_rank(
                        capability,
                        identity,
                        identity.world_size,
                        broker.owner,
                        self.runtime.deadline_after(TEST_TIMEOUT_NS),
                    )
                self.assertEqual(raised.exception.status, INVALID_ARGUMENT)
                with self.assertRaises(OSError):
                    fcntl.fcntl(capability, fcntl.F_GETFD)
                capability = -1
            finally:
                if capability >= 0:
                    os.close(capability)

    def test_live_rank_invalid_close_deadline_still_destroys(self) -> None:
        class DestroyTrackingLibrary:
            destroy_calls = 0

            def sagr_ccl_live_v1_rank_destroy(self, handle_pointer) -> None:
                self.destroy_calls += 1
                handle_pointer._obj.value = None

        class DestroyTrackingNative:
            def __init__(self) -> None:
                self.lib = DestroyTrackingLibrary()

        native = DestroyTrackingNative()
        rank = LiveRank(native, ctypes.c_void_p(1))
        with self.assertRaises(ValueError):
            rank.close(True)
        self.assertIsNone(rank.handle.value)
        self.assertEqual(native.lib.destroy_calls, 1)

    def test_live_world2_rank_table_ring_and_collective_close(self) -> None:
        before = open_fd_count()
        identity = self.identity(2)
        children: list[int] = []
        broker = self.runtime.live_broker(identity)
        try:
            owner = broker.owner
            for rank_number in range(2):
                capability = broker.prepare_rank(rank_number)
                pid = os.fork()
                if pid == 0:
                    child_exit(
                        self.success_rank_child,
                        broker,
                        capability,
                        identity,
                        owner,
                        rank_number,
                    )
                children.append(pid)
                os.close(capability)

            for rank_number, pid in enumerate(children):
                broker.bind_rank(rank_number, self.runtime.process_identity(pid))
            broker.rendezvous(self.runtime.deadline_after(TEST_TIMEOUT_NS))
            info = broker.info()
            self.assertEqual(info.phase, LIVE_PHASE_READY)
            self.assertEqual(info.ready_mask, 0b11)

            deadline = self.runtime.deadline_after(TEST_TIMEOUT_NS)
            while broker.info().phase != LIVE_PHASE_CLOSED:
                self.assertIsNone(broker.progress())
                if self.runtime.monotonic_time_ns() >= deadline:
                    self.fail("broker did not observe collective rank close")
                time.sleep(0.001)
            self.assertEqual(broker.info().departed_mask, 0b11)
            reap_children(children)
        finally:
            terminate_children(children)
            broker.destroy()
        self.assertEqual(open_fd_count(), before)

    def test_live_world3_busy_and_first_error_relay(self) -> None:
        before = open_fd_count()
        identity = self.identity(3)
        sync_read, sync_write = os.pipe2(os.O_CLOEXEC)
        children: list[int] = []
        broker = self.runtime.live_broker(identity)
        try:
            owner = broker.owner
            self.assertIsNone(broker.first_error())
            for rank_number in range(3):
                capability = broker.prepare_rank(rank_number)
                pid = os.fork()
                if pid == 0:
                    child_exit(
                        self.abort_rank_child,
                        broker,
                        capability,
                        identity,
                        owner,
                        rank_number,
                        sync_read,
                        sync_write,
                    )
                children.append(pid)
                os.close(capability)

            for rank_number, pid in enumerate(children):
                broker.bind_rank(rank_number, self.runtime.process_identity(pid))
            broker.rendezvous(self.runtime.deadline_after(TEST_TIMEOUT_NS))
            os.close(sync_write)
            sync_write = -1
            markers = b""
            deadline = self.runtime.deadline_after(TEST_TIMEOUT_NS)
            while len(markers) < 3:
                wait_fd(sync_read, select.POLLIN, deadline)
                markers += os.read(sync_read, 3 - len(markers))
            self.assertEqual(markers, b"\xa5" * 3)

            first_error = None
            while first_error is None:
                first_error = broker.progress()
                if first_error is None:
                    if self.runtime.monotonic_time_ns() >= deadline:
                        self.fail("broker did not latch the first rank error")
                    time.sleep(0.001)
            self.assertEqual(first_error.reporter_rank, 0)
            self.assertEqual(first_error.failed_rank, 1)
            self.assertEqual(first_error.status, CHECKSUM_ERROR)
            self.assertEqual(first_error.context_sequence, 7)
            self.assertNotEqual(first_error.reporter_rank, NO_RANK)
            while broker.info().abort_pending_mask:
                relayed = broker.progress()
                self.assertIsNotNone(relayed)
                self.assertEqual(bytes(relayed), bytes(first_error))
                if self.runtime.monotonic_time_ns() >= deadline:
                    self.fail("broker did not relay the abort to every rank")
                time.sleep(0.001)
            self.assertEqual(bytes(broker.first_error()), bytes(first_error))
            reap_children(children)
        finally:
            if sync_write >= 0:
                os.close(sync_write)
            os.close(sync_read)
            terminate_children(children)
            broker.destroy()
        self.assertEqual(open_fd_count(), before)

    def test_explicit_abort_record_is_immutable(self) -> None:
        identity = self.identity(3)
        descriptor = self.runtime.descriptor(
            identity,
            sequence=9,
            input_count=7,
            output_count=7,
            rank=0,
            operation=OP_ALL_REDUCE,
            reduction=REDUCTION_SUM,
            dtype=DTYPE_BF16,
        )
        with self.runtime.carrier_session(identity, 0) as session:
            record = session.abort(descriptor, failed_rank=2)
            stored = session.get_abort()
            self.assertEqual(bytes(record), bytes(stored))
            self.assertEqual(stored.failed_rank, 2)
            self.assertEqual(session.info().failed_rank, 2)

    def test_carrier_session_round_trip(self) -> None:
        identity = self.identity(2)
        source = self.runtime.descriptor(
            identity,
            sequence=1,
            input_count=3,
            output_count=3,
            rank=0,
            operation=OP_ALL_REDUCE,
            reduction=REDUCTION_SUM,
            dtype=DTYPE_BF16,
        )
        destination = self.runtime.descriptor(
            identity,
            sequence=1,
            input_count=3,
            output_count=3,
            rank=1,
            operation=OP_ALL_REDUCE,
            reduction=REDUCTION_SUM,
            dtype=DTYPE_BF16,
        )
        left, right = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_SEQPACKET | socket.SOCK_NONBLOCK | socket.SOCK_CLOEXEC,
        )
        try:
            with self.runtime.carrier_session(identity, 0) as sender, \
                    self.runtime.carrier_session(identity, 1) as receiver:
                data = sender.prepare_data(source, 0, b"\x01\x02")
                self.assertTrue(sender.send_data(left.fileno(), data))
                received = receiver.receive(right.fileno(), destination, 0, 0)
                self.assertIsNotNone(received)
                payload, consumed = receiver.consume(destination, 0, received)
                self.assertEqual(payload, b"\x01\x02")
                self.assertTrue(receiver.send_consumed(right.fileno(), consumed))
                ack = sender.receive(left.fileno(), source, 0, 1)
                self.assertIsNotNone(ack)
                self.assertEqual(ack.kind, MESSAGE_CONSUMED)
                self.assertEqual(sender.info().sender_inflight, 0)
        finally:
            left.close()
            right.close()


if __name__ == "__main__":
    unittest.main()
