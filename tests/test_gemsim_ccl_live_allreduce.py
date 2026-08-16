# SPDX-License-Identifier: GPL-3.0-or-later
"""Host-only tests for the future device-backed live allreduce gate."""

from __future__ import annotations

import copy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/gemsim_ccl_live_allreduce.py"
SPEC = importlib.util.spec_from_file_location("gemsim_ccl_live_allreduce", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def current_runtime() -> Path:
    configured = os.environ.get("GEMSIM_CCL_RUNTIME")
    candidate = Path(configured) if configured else MODULE.default_runtime_library()
    if not candidate.is_file():
        raise RuntimeError(f"current CCL runtime build is unavailable: {candidate}")
    return candidate.resolve()


class LiveAllreduceDesignTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = current_runtime()

    def design(
        self, root: Path, world: int, count: int, dtype: str = "float32"
    ) -> dict:
        namespace = (root / f"w{world}-c{count}-{dtype}").resolve()
        return MODULE.build_design(
            self.runtime,
            namespace,
            MODULE.deterministic_config(world, count, dtype),
        )

    def test_expected_wrapper_builder_uses_the_frozen_arithmetic_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            design = self.design(Path(temporary), 3, 7, "bfloat16")
            wrapper = MODULE.build_expected_wrapper(design)
            self.assertEqual(
                set(wrapper), {"schema", "arithmetic_policy", "design"}
            )
            self.assertEqual(wrapper["schema"], MODULE.EXPECTED_SCHEMA)
            self.assertEqual(
                wrapper["arithmetic_policy"], MODULE.ARITHMETIC_POLICY
            )
            self.assertEqual(wrapper["design"], design)

    def test_expected_wrapper_cli_is_atomic_canonical_and_absent_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary).resolve()
            namespace = root / "run"
            expected = root / "expected.json"
            command = [
                sys.executable,
                str(RUNNER),
                "--design-only",
                "--runtime-library",
                str(self.runtime),
                "--namespace-root",
                str(namespace),
                "--world-size",
                "3",
                "--element-count",
                "7",
                "--dtype",
                "float32",
                "--expected-output",
                str(expected),
            ]
            completed = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.stdout, b"")
            payload = expected.read_bytes()
            wrapper = json.loads(payload)
            self.assertEqual(payload, MODULE.canonical_json(wrapper))
            self.assertEqual(
                wrapper,
                MODULE.build_expected_wrapper(wrapper["design"]),
            )
            self.assertEqual(
                list(root.glob(f".{expected.name}.tmp-*")), []
            )
            repeated = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(expected.read_bytes(), payload)
            self.assertEqual(
                list(root.glob(f".{expected.name}.tmp-*")), []
            )

    def test_expected_wrapper_requires_explicit_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary).resolve()
            expected = root / "expected.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--design-only",
                    "--namespace-root",
                    str(root / "run"),
                    "--world-size",
                    "2",
                    "--element-count",
                    "1024",
                    "--dtype",
                    "bfloat16",
                    "--expected-output",
                    str(expected),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"explicit --runtime-library", result.stderr)
            self.assertFalse(expected.exists())

    def test_systematic_worlds_two_through_sixteen_bind_production_planner(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            for world in MODULE.SYSTEMATIC_WORLDS:
                with self.subTest(world=world):
                    design = self.design(root, world, world + 3)
                    self.assertEqual(len(design["ranks"]), world)
                    self.assertEqual(
                        {
                            len(rank["segments"][0]["steps"])
                            for rank in design["ranks"]
                        },
                        {2 * (world - 1)},
                    )
                    self.assertEqual(design["group_expected"]["zero_chunks"], 0)
                    self.assertEqual(
                        design["group_expected"]["device_sum_launches"],
                        world * (world - 1),
                    )
                    self.assertFalse(design["execution"]["gem5_started"])
                    self.assertFalse(
                        design["execution"]["live_collective_accepted"]
                    )

    def test_count_less_than_world_has_exact_zero_chunk_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            for world in MODULE.SYSTEMATIC_WORLDS:
                with self.subTest(world=world):
                    design = self.design(root, world, 1, "bfloat16")
                    expected = design["group_expected"]
                    self.assertEqual(expected["nonzero_chunks"], 1)
                    self.assertEqual(expected["zero_chunks"], world - 1)
                    self.assertEqual(expected["device_sum_launches"], world - 1)
                    self.assertEqual(
                        expected["zero_data_sends"],
                        2 * (world - 1) * (world - 1),
                    )
                    launch_count = sum(
                        rank["expected"]["device_sum_launches"]
                        for rank in design["ranks"]
                    )
                    self.assertEqual(launch_count, world - 1)

    def test_odd_world_plans_are_generic_and_peer_symmetric(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            for world in MODULE.ODD_PLANNER_WORLDS:
                with self.subTest(world=world):
                    design = self.design(root, world, 2 * world + 1)
                    for rank in design["ranks"]:
                        for ordinal, step in enumerate(
                            rank["segments"][0]["steps"]
                        ):
                            peer = design["ranks"][step["send_rank"]][
                                "segments"
                            ][0]["steps"][ordinal]
                            self.assertEqual(peer["receive_rank"], rank["rank"])
                            self.assertEqual(
                                peer["receive_chunk"], step["send_chunk"]
                            )
                            self.assertEqual(
                                peer["receive_count_elements"],
                                step["send_count_elements"],
                            )

    def test_live_entry_matrix_and_isolated_rank_namespaces(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            for world in MODULE.SYSTEMATIC_WORLDS:
                with self.subTest(world=world):
                    design = self.design(root, world, 3)
                    self.assertIs(
                        design["coverage"]["this_world_is_formal_live_entry"],
                        world in MODULE.LIVE_ACCEPTANCE_WORLDS,
                    )
                    paths = [rank["rank_launch"]["paths"] for rank in design["ranks"]]
                    for key in (
                        "instance_directory",
                        "triton_cache_directory",
                        "runtime_directory",
                        "endpoint",
                        "gem5_output_directory",
                        "dispatch_trace_path",
                        "gem5_log_path",
                        "gem5_cache_directory",
                    ):
                        self.assertEqual(len({item[key] for item in paths}), world)
                    self.assertTrue(design["process_architecture"]["per_rank_daemon"])
                    self.assertIn(
                        "sagr_managed_session_open_v2",
                        design["process_architecture"]["managed_session_start"],
                    )
                    self.assertIn(
                        "pass_fds=(capability_fd,)",
                        design["process_architecture"]["worker_spawn"],
                    )

    def test_segmentation_boundary_and_sequence_for_world_three_and_sixteen(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            for world in (3, 16):
                for dtype, width in (("bfloat16", 2), ("float32", 4)):
                    limit = MODULE.segment_element_limit(world, width)
                    self.assertEqual(
                        limit, MODULE.CARRIER_MAX_PAYLOAD_BYTES // width
                    )
                    for count, expected_segments in (
                        (limit - 1, 1),
                        (limit, 1),
                        (limit + 1, 2),
                    ):
                        with self.subTest(
                            world=world, dtype=dtype, count=count
                        ):
                            design = self.design(root, world, count, dtype)
                            segments = design["segmentation"]["segments"]
                            self.assertEqual(len(segments), expected_segments)
                            self.assertEqual(
                                [item["sequence"] for item in segments],
                                list(range(1, expected_segments + 1)),
                            )
                            self.assertTrue(
                                all(
                                    item["byte_count"]
                                    <= MODULE.CARRIER_MAX_PAYLOAD_BYTES
                                    for item in segments
                                )
                            )
                            self.assertTrue(
                                all(
                                    item["maximum_payload_bytes"]
                                    <= MODULE.CARRIER_MAX_PAYLOAD_BYTES
                                    for item in segments
                                )
                            )
                            self.assertEqual(
                                sum(item["element_count"] for item in segments),
                                count,
                            )
                            self.assertEqual(
                                segments[-1]["base_offset_elements"]
                                + segments[-1]["element_count"],
                                count,
                            )
                            self.assertEqual(
                                design["group_expected"]["public_commit_count"],
                                1,
                            )

    def test_public_tensor_allocation_limit_fails_closed(self) -> None:
        maximum = MODULE.MANAGED_MAX_SINGLE_ALLOCATION_BYTES // 4
        MODULE.deterministic_config(3, maximum, "float32").validate()
        with self.assertRaisesRegex(MODULE.DesignError, "managed"):
            MODULE.deterministic_config(3, maximum + 1, "float32").validate()

    def test_zero_element_collective_fails_before_descriptor_construction(
        self,
    ) -> None:
        config = MODULE.deterministic_config(3, 0, "float32")
        with self.assertRaisesRegex(MODULE.DesignError, r"element_count.*\[1,"):
            config.validate()
        with self.assertRaisesRegex(
            MODULE.DesignError, r"segment element_count.*\[1,"
        ):
            MODULE.plan_segments(0, 3, 4)

    def test_cli_is_design_only_and_never_launches_gem5(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            namespace = (Path(temporary) / "namespaces").resolve()
            base = [
                sys.executable,
                str(RUNNER),
                "--runtime-library",
                str(self.runtime),
                "--namespace-root",
                str(namespace),
                "--world-size",
                "3",
                "--element-count",
                "1",
                "--dtype",
                "float32",
            ]
            rejected = subprocess.run(
                base,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("--design-only", rejected.stderr)
            completed = subprocess.run(
                [*base, "--design-only"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stderr, "")
            document = json.loads(completed.stdout)
            self.assertFalse(document["execution"]["gem5_started"])
            self.assertFalse(document["execution"]["rank_workers_started"])
            self.assertFalse(namespace.exists())

    def test_tampered_planner_and_process_architecture_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            design = self.design(Path(temporary), 5, 3)
            tampered = copy.deepcopy(design)
            tampered["ranks"][2]["segments"][0]["steps"][1][
                "receive_chunk"
            ] = 5
            tampered["ranks"][2]["segments"][0][
                "plan_sha256"
            ] = MODULE.object_sha256(
                tampered["ranks"][2]["segments"][0]["steps"]
            )
            with self.assertRaises(MODULE.DesignError):
                MODULE.validate_design(tampered)
            tampered = copy.deepcopy(design)
            tampered["process_architecture"]["worker_spawn"] = (
                "subprocess.Popen(close_fds=True)"
            )
            with self.assertRaises(MODULE.DesignError):
                MODULE.validate_design(tampered)


class OrderedStepGateTest(unittest.TestCase):
    group = MODULE.GroupBinding(
        job_uuid="1" * 32,
        group_uuid="2" * 32,
        model_identity_sha256="3" * 64,
        epoch=1,
        group_generation=1,
        world_size=3,
    )
    descriptor_sha256 = "4" * 64

    @staticmethod
    def step(
        phase: int,
        count: int,
        *,
        ordinal: int = 0,
        send_count: int | None = None,
    ) -> dict:
        return {
            "ordinal": ordinal,
            "phase": phase,
            "send_rank": 1,
            "receive_rank": 2,
            "send_chunk": 2,
            "receive_chunk": 1,
            "send_count_elements": count if send_count is None else send_count,
            "receive_count_elements": count,
        }

    def gate(
        self,
        phase: int = MODULE.PHASE_REDUCE_SCATTER,
        count: int = 1,
        *,
        sequence: int = 1,
        ledger: object | None = None,
        ordinal: int = 0,
    ) -> tuple[object, object]:
        selected = ledger or MODULE.CreditLedger(0, 3, 2)
        return (
            MODULE.OrderedStepGate(
                self.step(phase, count, ordinal=ordinal),
                4,
                self.group,
                self.descriptor_sha256,
                sequence,
                0,
                selected,
            ),
            selected,
        )

    def inbound_data(
        self,
        gate: object,
        *,
        slot: int = 0,
        generation: int = 1,
    ) -> object:
        step = gate.inbound.step
        transfer = MODULE.TransferTuple(
            group=self.group,
            descriptor_sha256=self.descriptor_sha256,
            sequence=gate.inbound.sequence,
            phase=step["phase"],
            step_index=step["ordinal"],
            chunk_index=step["receive_chunk"],
            source_rank=step["receive_rank"],
            destination_rank=0,
            slot_index=slot,
            slot_generation=generation,
        )
        return MODULE.TransferRecord(
            MODULE.MESSAGE_DATA,
            transfer,
            step["receive_count_elements"] * 4,
        )

    @staticmethod
    def consumed(data: object) -> object:
        return MODULE.TransferRecord(MODULE.MESSAGE_CONSUMED, data.transfer, 0)

    def drive_full_duplex(self, gate: object) -> tuple[object, object]:
        outbound = gate.prepare_outbound(
            b"\x10" * (gate.outbound.step["send_count_elements"] * 4)
        )
        gate.send_outbound_data(outbound)
        inbound = self.inbound_data(gate)
        gate.receive_inbound_data(inbound, 2)
        inbound_ack = gate.consume_inbound(
            b"\x20" * (gate.inbound.step["receive_count_elements"] * 4)
        )
        if gate.inbound.step["phase"] == MODULE.PHASE_REDUCE_SCATTER:
            gate.device_sum_complete(
                dispatched=gate.inbound.receive_count != 0,
                succeeded=True,
            )
        else:
            gate.copy_complete()
        gate.send_inbound_consumed(inbound_ack, 2)
        return outbound, inbound

    def test_full_duplex_step_requires_both_transfer_terminals(self) -> None:
        gate, ledger = self.gate(count=2)
        outbound, _ = self.drive_full_duplex(gate)
        self.assertEqual(gate.state, MODULE.StepState.ACTIVE)
        self.assertEqual(ledger.sender_inflight, 1)
        self.assertEqual(ledger.receiver_active, 0)
        gate.receive_outbound_consumed(self.consumed(outbound), 1)
        self.assertEqual(gate.state, MODULE.StepState.COMPLETE)
        self.assertEqual(ledger.sender_inflight, 0)
        snapshot = gate.snapshot()
        self.assertEqual(snapshot["inbound"]["device_dispatch_count"], 1)
        self.assertEqual(snapshot["inbound"]["host_reduction_count"], 0)

    def test_zero_chunk_has_no_dispatch_then_ack(self) -> None:
        gate, ledger = self.gate(count=0)
        outbound, _ = self.drive_full_duplex(gate)
        self.assertIn("zero_no_dispatch", gate.inbound.events)
        self.assertEqual(gate.inbound.device_dispatch_count, 0)
        gate.receive_outbound_consumed(self.consumed(outbound), 1)
        self.assertEqual(gate.state, MODULE.StepState.COMPLETE)
        self.assertEqual(ledger.sender_inflight, 0)

    def test_inbound_ack_before_device_completion_aborts_both_sides(self) -> None:
        gate, ledger = self.gate()
        outbound = gate.prepare_outbound(b"\x00" * 4)
        gate.send_outbound_data(outbound)
        inbound = self.inbound_data(gate)
        gate.receive_inbound_data(inbound, 2)
        ack = gate.consume_inbound(b"\x01" * 4)
        with self.assertRaisesRegex(MODULE.DesignError, "preceded"):
            gate.send_inbound_consumed(ack, 2)
        self.assertEqual(gate.state, MODULE.StepState.ABORTED)
        self.assertEqual(ledger.sender_inflight, 0)
        self.assertNotIn("inbound_ack_sent", gate.inbound.events)

    def test_device_failure_aborts_without_ack(self) -> None:
        gate, ledger = self.gate()
        outbound = gate.prepare_outbound(b"\x00" * 4)
        gate.send_outbound_data(outbound)
        inbound = self.inbound_data(gate)
        gate.receive_inbound_data(inbound, 2)
        gate.consume_inbound(b"\x01" * 4)
        with self.assertRaisesRegex(MODULE.DesignError, "device SUM failed"):
            gate.device_sum_complete(dispatched=True, succeeded=False)
        self.assertEqual(gate.first_error, "device_failure")
        self.assertEqual(gate.state, MODULE.StepState.ABORTED)
        self.assertEqual(ledger.sender_inflight, 0)
        self.assertNotIn("inbound_ack_sent", gate.inbound.events)

    def test_all_gather_is_copy_with_no_host_reduction(self) -> None:
        gate, ledger = self.gate(MODULE.PHASE_ALL_GATHER, 2)
        outbound, _ = self.drive_full_duplex(gate)
        self.assertIn("inbound_copy_complete", gate.inbound.events)
        self.assertEqual(gate.inbound.host_reduction_count, 0)
        gate.receive_outbound_consumed(self.consumed(outbound), 1)
        self.assertEqual(gate.state, MODULE.StepState.COMPLETE)
        self.assertEqual(ledger.sender_inflight, 0)

    def test_wrong_peer_chunk_slot_and_generation_abort(self) -> None:
        for mutation in ("peer", "chunk", "slot", "generation"):
            with self.subTest(mutation=mutation):
                gate, ledger = self.gate()
                outbound = gate.prepare_outbound(b"\x00" * 4)
                gate.send_outbound_data(outbound)
                if mutation in ("peer", "chunk"):
                    inbound = self.inbound_data(gate)
                    if mutation == "chunk":
                        inbound = replace(
                            inbound,
                            transfer=replace(
                                inbound.transfer,
                                chunk_index=inbound.transfer.chunk_index + 1,
                            ),
                        )
                    with self.assertRaises(MODULE.DesignError):
                        gate.receive_inbound_data(
                            inbound, 1 if mutation == "peer" else 2
                        )
                else:
                    ack = self.consumed(outbound)
                    replacement = (
                        {"slot_index": 1}
                        if mutation == "slot"
                        else {
                            "slot_generation": (
                                ack.transfer.slot_generation + 1
                            )
                        }
                    )
                    ack = replace(
                        ack, transfer=replace(ack.transfer, **replacement)
                    )
                    with self.assertRaises(MODULE.DesignError):
                        gate.receive_outbound_consumed(ack, 1)
                self.assertEqual(gate.state, MODULE.StepState.ABORTED)
                self.assertEqual(ledger.sender_inflight, 0)

    def test_replayed_generation_aborts_new_inbound_gate(self) -> None:
        ledger = MODULE.CreditLedger(0, 3, 2)
        first, _ = self.gate(ledger=ledger)
        data = self.inbound_data(first)
        first.inbound.receive_data(data, 2)
        ack = first.inbound.consume_to_immutable_staging(b"\x01" * 4)
        first.inbound.device_sum_complete(dispatched=True, succeeded=True)
        first.inbound.send_consumed(ack, 2)
        replay, _ = self.gate(ledger=ledger)
        with self.assertRaisesRegex(MODULE.DesignError, "replayed"):
            replay.receive_inbound_data(data, 2)
        self.assertEqual(replay.state, MODULE.StepState.ABORTED)

    def test_two_credits_reverse_data_reverse_ack_and_exhaustion(self) -> None:
        ledger = MODULE.CreditLedger(0, 3, 2)
        gates = [
            MODULE.OutboundTransferGate(
                self.step(MODULE.PHASE_REDUCE_SCATTER, 1),
                4,
                self.group,
                self.descriptor_sha256,
                sequence,
                0,
                ledger,
            )
            for sequence in (1, 2, 3)
        ]
        first = gates[0].prepare_data(b"\x01" * 4)
        second = gates[1].prepare_data(b"\x02" * 4)
        self.assertEqual((first.transfer.slot_index, second.transfer.slot_index), (0, 1))
        self.assertEqual(ledger.sender_inflight, 2)
        with self.assertRaises(MODULE.CreditBusy):
            gates[2].prepare_data(b"\x03" * 4)
        self.assertEqual(gates[2].state, MODULE.OutboundState.INITIAL)
        gates[1].send_data(second)
        gates[0].send_data(first)
        gates[0].receive_consumed(self.consumed(first), 1)
        third = gates[2].prepare_data(b"\x03" * 4)
        self.assertEqual(third.transfer.slot_index, 0)
        self.assertGreater(
            third.transfer.slot_generation, first.transfer.slot_generation
        )
        gates[1].receive_consumed(self.consumed(second), 1)
        gates[2].abort("test_cleanup")
        self.assertEqual(ledger.sender_inflight, 0)


class SyntheticEvidenceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = current_runtime()

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        root = Path(self.temporary.name)
        self.design = MODULE.build_design(
            self.runtime,
            (root / "namespaces").resolve(),
            MODULE.deterministic_config(3, 1, "float32"),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def cleanup() -> dict:
        return {
            "worker_reaped": True,
            "daemon_reaped": True,
            "owned_fd_delta": 0,
            "orphan_count": 0,
        }

    def success(self) -> dict:
        return {
            "schema": MODULE.SUCCESS_EVIDENCE_SCHEMA,
            "mode": "synthetic_expectation",
            "acceptance_authority": False,
            "live_collective_accepted": False,
            "formal_artifact_bundle_present": False,
            "world_size": 3,
            "segment_count": self.design["group_expected"]["segment_count"],
            "required_device_sum_launches": self.design["group_expected"][
                "device_sum_launches"
            ],
            "required_host_reduction_count": 0,
            "required_public_commit_count": 1,
        }

    def failure(self, status: str = "device_failure") -> dict:
        failed_rank = 0
        segment_index = 0
        segment = self.design["ranks"][failed_rank]["segments"][segment_index]
        step_index = next(
            index
            for index, step in enumerate(segment["steps"][:2])
            if step["receive_count_elements"] > 0
        )
        step = segment["steps"][step_index]
        sequence = segment["sequence"]
        first_error = {
            "status": status,
            "reporter_rank": (
                (1 << 32) - 1 if status == "peer_lost" else failed_rank
            ),
            "failed_rank": failed_rank,
            "context_sequence": sequence,
        }
        transfer = {
            "group": MODULE._group_document(self.design["config"]),
            "descriptor_sha256": segment["descriptor_sha256"],
            "sequence": sequence,
            "phase": step["phase"],
            "step_index": step_index,
            "chunk_index": step["receive_chunk"],
            "source_rank": step["receive_rank"],
            "destination_rank": failed_rank,
            "slot_index": 0,
            "slot_generation": 1,
        }
        session = {
            "job_uuid": self.design["config"]["job_uuid"],
            "epoch": self.design["config"]["epoch"],
            "rank": failed_rank,
            "world_size": 3,
            "daemon_uuid": "d" * 32,
            "runtime_sha256": self.design["runtime"]["sha256"],
        }
        binding = {
            "segment_index": segment_index,
            "sequence": sequence,
            "step_index": step_index,
            "descriptor_sha256": segment["descriptor_sha256"],
            "plan_sha256": segment["plan_sha256"],
            "runtime_sha256": self.design["runtime"]["sha256"],
            "rank_launch_sha256": self.design["ranks"][failed_rank][
                "rank_launch_sha256"
            ],
            "managed_session": session,
            "managed_session_sha256": MODULE.object_sha256(session),
            "transfer_tuple": transfer,
        }
        completed = 200 if status == "timed_out" else 150
        expectation = {
            "schema": MODULE.FAILURE_EVIDENCE_SCHEMA,
            "mode": "synthetic_expectation",
            "acceptance_authority": False,
            "live_collective_accepted": False,
            "formal_artifact_bundle_present": False,
            "world_size": 3,
            "status": status,
            "canonical_first_error": first_error,
            "failure_binding": binding,
            "failed_step": {
                "segment_index": segment_index,
                "sequence": sequence,
                "step_index": step_index,
                "transfer_tuple": transfer,
                "data_consumed_to_immutable_staging": status == "device_failure",
                "receive_count_elements": (
                    step["receive_count_elements"]
                    if status == "device_failure"
                    else 0
                ),
                "device_dispatch_attempted": status == "device_failure",
                "device_dispatch_succeeded": False,
                "consumed_ack_sent": False,
            },
            "ranks": [
                {
                    "rank": rank,
                    "first_error": copy.deepcopy(first_error),
                    "failure_binding": copy.deepcopy(binding),
                    "public_result_published": False,
                    "public_commit_count": 0,
                    "host_reduction_count": 0,
                    "cleanup": self.cleanup(),
                }
                for rank in range(3)
            ],
            "target_feedback": False,
            "started_at_ns": 100,
            "completed_at_ns": completed,
            "absolute_deadline_ns": 200,
            "deadline_relation": (
                "expired" if status == "timed_out" else "before_deadline"
            ),
            "deadline_bounded": True,
            "all_cleanup_complete": True,
        }
        return expectation

    def test_synthetic_success_has_no_formal_acceptance_authority(self) -> None:
        expectation = self.success()
        self.assertEqual(
            MODULE.validate_synthetic_success_expectation(
                expectation, self.design
            ),
            expectation,
        )
        with self.assertRaisesRegex(MODULE.DesignError, "absent-only"):
            MODULE.validate_success_evidence(expectation, self.design)
        expectation["acceptance_authority"] = True
        with self.assertRaises(MODULE.DesignError):
            MODULE.validate_synthetic_success_expectation(
                expectation, self.design
            )

    def test_device_failure_binds_exact_identity_without_acceptance(self) -> None:
        expectation = self.failure()
        self.assertEqual(
            MODULE.validate_synthetic_failure_expectation(
                expectation, self.design
            ),
            expectation,
        )
        with self.assertRaisesRegex(MODULE.DesignError, "absent-only"):
            MODULE.validate_failure_evidence(expectation, self.design)
        for field in (
            "sequence",
            "descriptor_sha256",
            "plan_sha256",
            "runtime_sha256",
            "rank_launch_sha256",
            "managed_session_sha256",
        ):
            with self.subTest(field=field):
                tampered = copy.deepcopy(expectation)
                tampered["failure_binding"][field] = (
                    2 if field == "sequence" else "e" * 64
                )
                with self.assertRaises(MODULE.DesignError):
                    MODULE.validate_synthetic_failure_expectation(
                        tampered, self.design
                    )
        for field in ("chunk_index", "slot_index", "slot_generation"):
            with self.subTest(transfer_field=field):
                tampered = copy.deepcopy(expectation)
                tampered["failure_binding"]["transfer_tuple"][field] += 1
                with self.assertRaises(MODULE.DesignError):
                    MODULE.validate_synthetic_failure_expectation(
                        tampered, self.design
                    )
        tampered = copy.deepcopy(expectation)
        tampered["failed_step"]["consumed_ack_sent"] = True
        with self.assertRaises(MODULE.DesignError):
            MODULE.validate_synthetic_failure_expectation(tampered, self.design)

    def test_peer_loss_and_timeout_have_status_specific_deadlines(self) -> None:
        for status in ("peer_lost", "timed_out"):
            with self.subTest(status=status):
                expectation = self.failure(status)
                MODULE.validate_synthetic_failure_expectation(
                    expectation, self.design
                )
                expectation["deadline_bounded"] = False
                with self.assertRaises(MODULE.DesignError):
                    MODULE.validate_synthetic_failure_expectation(
                        expectation, self.design
                    )
                expectation = self.failure(status)
                expectation["completed_at_ns"] = (
                    199 if status == "timed_out" else 200
                )
                with self.assertRaises(MODULE.DesignError):
                    MODULE.validate_synthetic_failure_expectation(
                        expectation, self.design
                    )


if __name__ == "__main__":
    unittest.main()
