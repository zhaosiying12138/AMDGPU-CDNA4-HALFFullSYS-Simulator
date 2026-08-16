from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
CCL_PACKAGE = ROOT / "plugins/collectives/gemsim_ccl/src"
sys.path.insert(0, str(CCL_PACKAGE))

from gemsim_ccl.engine import (  # noqa: E402
    AllReduceSegment,
    CollectiveEvent,
    TransferInfo,
)
from gemsim_ccl.native import (  # noqa: E402
    MESSAGE_CONSUMED,
    MESSAGE_DATA,
    PHASE_ALL_GATHER,
    PHASE_REDUCE_SCATTER,
    PlannedStep,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WORKER_PATH = ROOT / "examples/triton/ccl_live_allreduce_rank.py"
WORKER = load_module("_test_ccl_live_worker", WORKER_PATH)
ACCEPTANCE = load_module(
    "_test_ccl_live_acceptance",
    ROOT / "tools/gemsim_ccl_live_allreduce_acceptance.py",
)
DESCRIPTOR_SHA256 = "11" * 32


def step(phase: int) -> PlannedStep:
    return PlannedStep(
        phase=phase,
        action=3,
        step_index=0,
        send_rank=1,
        receive_rank=1,
        send_chunk=0,
        receive_chunk=0,
        send_offset_elements=0,
        send_count_elements=4,
        receive_offset_elements=0,
        receive_count_elements=4,
    )


def transfer(
    planned: PlannedStep,
    ordinal: int,
    *,
    outbound: bool,
    kind: int,
) -> TransferInfo:
    return TransferInfo(
        descriptor_sha256=DESCRIPTOR_SHA256,
        sequence=1,
        kind=kind,
        phase=planned.phase,
        step_index=ordinal,
        chunk_index=0,
        source_rank=0 if outbound else 1,
        destination_rank=1 if outbound else 0,
        slot_index=0,
        slot_generation=ordinal + 1,
        payload_bytes=16 if kind == MESSAGE_DATA else 0,
        status=0,
        failed_rank=(1 << 32) - 1,
    )


class EngineEvidenceObserverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.segment = AllReduceSegment(
            index=0,
            sequence=1,
            offset_elements=0,
            element_count=4,
            byte_count=16,
        )
        self.steps = (step(PHASE_REDUCE_SCATTER), step(PHASE_ALL_GATHER))
        self.plan = [
            WORKER._plan_document(ordinal, item, 0, 4)
            for ordinal, item in enumerate(self.steps)
        ]
        self.expected_segment = {
            "segment_id": 0,
            "sequence": 1,
            "global_offset_elements": 0,
            "element_count": 4,
            "byte_count": 16,
            "descriptor_sha256": DESCRIPTOR_SHA256,
            "plan_sha256": WORKER.sha256_bytes(WORKER.canonical_json(self.plan)),
        }

    def event(self, name: str, **values: object) -> CollectiveEvent:
        defaults = {
            "name": name,
            "monotonic_ns": 100,
            "rank": 0,
            "world_size": 2,
        }
        defaults.update(values)
        return CollectiveEvent(**defaults)

    def emit_success(self, observer: WORKER.EngineEvidenceObserver) -> bytes:
        observer(
            self.event(
                "collective_started",
                segment=self.segment,
            )
        )
        observer(
            self.event(
                "segment_started",
                segment=self.segment,
                descriptor_sha256=DESCRIPTOR_SHA256,
            )
        )
        for ordinal, planned in enumerate(self.steps):
            outbound = transfer(
                planned, ordinal, outbound=True, kind=MESSAGE_DATA
            )
            inbound = transfer(
                planned, ordinal, outbound=False, kind=MESSAGE_DATA
            )
            inbound_consumed = transfer(
                planned, ordinal, outbound=False, kind=MESSAGE_CONSUMED
            )
            outbound_consumed = transfer(
                planned, ordinal, outbound=True, kind=MESSAGE_CONSUMED
            )
            common = {
                "segment": self.segment,
                "step_ordinal": ordinal,
                "step": planned,
            }
            observer(self.event("outbound_prepared", transfer=outbound, **common))
            observer(self.event("outbound_DATA_sent", transfer=outbound, **common))
            observer(self.event("inbound_DATA_received", transfer=inbound, **common))
            observer(
                self.event(
                    "inbound_staged",
                    transfer=inbound,
                    payload_sha256=hashlib.sha256(b"x" * 16).hexdigest(),
                    byte_count=16,
                    **common,
                )
            )
            if planned.phase == PHASE_REDUCE_SCATTER:
                observer(self.event("device_call_enter", **common))
                observer(self.event("device_call_returned", **common))
            else:
                observer(self.event("copy_complete", byte_count=16, **common))
            observer(
                self.event(
                    "inbound_CONSUMED_send_attempt",
                    transfer=inbound_consumed,
                    **common,
                )
            )
            observer(
                self.event(
                    "inbound_CONSUMED_sent",
                    transfer=inbound_consumed,
                    **common,
                )
            )
            observer(
                self.event(
                    "outbound_CONSUMED_received_credit_released",
                    transfer=outbound_consumed,
                    **common,
                )
            )
            observer(self.event("step_complete", **common))
        observer(
            self.event(
                "segment_complete",
                segment=self.segment,
                descriptor_sha256=DESCRIPTOR_SHA256,
            )
        )
        output = b"result-bytes-000"
        observer(
            self.event(
                "public_commit",
                payload_sha256=hashlib.sha256(output).hexdigest(),
                byte_count=len(output),
            )
        )
        return output

    def test_engine_events_round_trip_through_authoritative_journal_validator(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal_path = Path(temporary) / "journal.jsonl"
            context: dict[str, object] = {}
            journal = WORKER.StepJournal(journal_path, rank=0)
            observer = WORKER.EngineEvidenceObserver(
                journal=journal,
                segments=[self.expected_segment],
                dtype_bytes=4,
                world_size=2,
                execution_context=context,
            )
            output = self.emit_success(observer)
            observer.publish_public_commit(output)
            journal.close()

            records = [
                json.loads(line)
                for line in journal_path.read_text(encoding="ascii").splitlines()
            ]
            verifier_segment = {
                **self.expected_segment,
                "steps": self.plan,
            }
            summary = ACCEPTANCE.validate_journal(
                records,
                rank=0,
                segments=[verifier_segment],
                dtype_bytes=4,
            )
            self.assertEqual(summary["steps"], 2)
            self.assertEqual(summary["public_commit_count"], 1)
            self.assertEqual(summary["device_steps"], [(0, 0, 4)])
            self.assertEqual(observer.segment_results[0]["plan_sha256"], self.expected_segment["plan_sha256"])
            self.assertEqual(
                observer.counters,
                {
                    "data_sent_count": 2,
                    "data_received_count": 2,
                    "consumed_sent_count": 2,
                    "consumed_received_count": 2,
                    "device_reduction_launch_count": 1,
                    "host_reduction_count": 0,
                    "public_commit_count": 1,
                },
            )
            self.assertIsNone(context["failed_transfer"])

    def test_expected_design_is_fail_closed_observation_not_output_feedback(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = WORKER.StepJournal(Path(temporary) / "journal.jsonl", 0)
            tampered = dict(self.expected_segment)
            tampered["descriptor_sha256"] = "22" * 32
            observer = WORKER.EngineEvidenceObserver(
                journal=journal,
                segments=[tampered],
                dtype_bytes=4,
                world_size=2,
                execution_context={},
            )
            observer(self.event("collective_started", segment=self.segment))
            with self.assertRaisesRegex(WORKER.RankError, "descriptor digest"):
                observer(
                    self.event(
                        "segment_started",
                        segment=self.segment,
                        descriptor_sha256=DESCRIPTOR_SHA256,
                    )
                )
            journal.close()
            self.assertEqual(
                (Path(temporary) / "journal.jsonl").read_bytes(),
                b"",
            )

    def test_public_commit_is_published_only_after_persisted_bytes_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal_path = Path(temporary) / "journal.jsonl"
            journal = WORKER.StepJournal(journal_path, 0)
            observer = WORKER.EngineEvidenceObserver(
                journal=journal,
                segments=[self.expected_segment],
                dtype_bytes=4,
                world_size=2,
                execution_context={},
            )
            output = self.emit_success(observer)
            with self.assertRaisesRegex(WORKER.RankError, "persisted output"):
                observer.publish_public_commit(output + b"x")
            journal.close()
            records = [
                json.loads(line)
                for line in journal_path.read_text(encoding="ascii").splitlines()
            ]
            self.assertNotIn("public_commit", [item["event"] for item in records])


class FormalWorkerPathTest(unittest.TestCase):
    def test_success_path_calls_reusable_engine_and_not_legacy_executor(self):
        tree = ast.parse(WORKER_PATH.read_text(encoding="ascii"))
        run_live_rank = next(
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "run_live_rank"
        )
        calls = [
            ast.unparse(item.func)
            for item in ast.walk(run_live_rank)
            if isinstance(item, ast.Call)
        ]
        self.assertIn("AllReduceEngine.join", calls)
        self.assertIn("engine.all_reduce", calls)
        self.assertIn("engine.close", calls)
        self.assertNotIn("execute_segments", calls)
        self.assertNotIn("native.join_rank", calls)
        self.assertNotIn("native.carrier_session", calls)

    def test_remaining_timeout_keeps_an_absolute_deadline_guard(self):
        native = SimpleNamespace(monotonic_time_ns=lambda: 10_000)
        self.assertEqual(
            WORKER._remaining_timeout_ns(
                native,
                10_000 + WORKER.DEADLINE_GUARD_NS + 37,
                "test",
            ),
            37,
        )
        with self.assertRaisesRegex(WORKER.RankError, "before test"):
            WORKER._remaining_timeout_ns(
                native,
                10_000 + WORKER.DEADLINE_GUARD_NS,
                "test",
            )


if __name__ == "__main__":
    unittest.main()
