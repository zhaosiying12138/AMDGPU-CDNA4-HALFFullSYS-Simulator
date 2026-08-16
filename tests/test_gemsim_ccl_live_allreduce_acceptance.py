# SPDX-License-Identifier: GPL-3.0-or-later
"""Host-only synthetic tests for the live allreduce external verifier."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "tools/gemsim_ccl_live_allreduce_acceptance.py"
DESIGN_PATH = ROOT / "tools/gemsim_ccl_live_allreduce.py"


def load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


V = load("gemsim_ccl_live_allreduce_acceptance", VERIFIER_PATH)
D = load("gemsim_ccl_live_allreduce_acceptance_design", DESIGN_PATH)


def write(path: Path, payload: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_json(path: Path, value: object) -> dict:
    return write(path, V.canonical_json(value))


def trace_base(rank: int, launch: int, count: int, *, job_uuid: str,
               epoch: int, world: int, daemon_uuid: str,
               connection_id: int) -> dict:
    admission_tick = 900 + launch * 200
    retire_tick = admission_tick + 100
    return {
        "schema": V.TRACE_SCHEMA,
        "event": "generic_execution_retired",
        "sim_tick": retire_tick,
        "daemon_uuid": daemon_uuid,
        "job_uuid": job_uuid,
        "epoch": epoch,
        "rank": rank,
        "world_size": world,
        "connection_id": connection_id,
        "owner_fd": 10,
        "owner_generation": 1,
        "request_id": 1000 * (rank + 1) + launch,
        "trace_id": launch + 1,
        "ticket_id": launch + 1,
        "dispatch_id": launch + 1,
        "kernel": "_sum_kernel",
        "image_sha256": "a" * 64,
        "grid": [((count + 255) // 256) * 256, 1, 1],
        "workgroup": [256, 1, 1],
        "fixed_shared_memory_bytes": 0,
        "dynamic_shared_memory_bytes": 0,
        "total_shared_memory_bytes": 0,
        "kernarg_va": 0x1000 + launch * 0x100,
        "kernarg_size": 48,
        "packet_va": 0x2000 + launch * 0x100,
        "packet_crc32c": launch + 1,
        "allocation_count": 2,
        "allocations": [
            {"allocation_id": 1, "generation": 1, "gpu_va": 0x3000,
             "bytes": count * 4},
            {"allocation_id": 2, "generation": 2, "gpu_va": 0x4000,
             "bytes": count * 2},
        ],
        "packet_fetches": 1,
        "command_processor_submissions": 1,
        "gpu_dispatcher_starts": 1,
        "waves_started": 1,
        "instructions_started": 4,
        "instruction_wave_count": 1,
        "scalar_reads": 0,
        "global_reads": 2 * count,
        "global_writes": count,
        "store_events": 1,
        "store_dwords": count,
        "workgroups_completed": (count + 255) // 256,
        "signal_before": 1,
        "signal_after": 0,
        "admission_tick": admission_tick,
        "start_tick": admission_tick,
        "end_tick": retire_tick,
        "retire_tick": retire_tick,
        "compatibility_completion_token_crc32c": launch + 1,
        "native_queue_retired": True,
        "application_pins_released": True,
        "adapter_released": True,
        "type20_durable": False,
        "unmap_durable": False,
        "owner_disconnected": False,
        "owner_quarantined": False,
        "cleanup_complete": False,
        "kernel_executed": True,
        "numerical_oracle": "external_user_process",
    }


class SyntheticEvidence:
    def __init__(self, root: Path, *, world: int = 2, count: int = 3,
                 dtype: str = "float32",
                 model_identity_sha256: str | None = None) -> None:
        self.root = root
        self.source = (root / "source").resolve()
        self.output = (root / "accepted").resolve()
        self.expected_path = (root / "expected.json").resolve()
        self.product_prefix = (root / "product").resolve()
        self.product_prefix.mkdir()
        self.product_id = "a" * 64
        write_json(
            self.product_prefix / "manifest.json",
            {
                "schema": "amdgpu-sim.product-prefix.v1",
                "product_id": self.product_id,
                "prefix": str(self.product_prefix),
            },
        )
        self.runtime = D.default_runtime_library()
        self.design = D.build_design(
            self.runtime,
            (root / "namespaces").resolve(),
            D.deterministic_config(
                world, count, dtype,
                model_identity_sha256=model_identity_sha256,
            ),
        )
        self.expected = {
            "schema": V.EXPECTED_SCHEMA,
            "arithmetic_policy": copy.deepcopy(V.ARITHMETIC_POLICY),
            "design": self.design,
        }
        self.expected_record = write_json(self.expected_path, self.expected)
        self.source.mkdir()
        self.world = world
        self.count = count
        self.dtype = dtype
        self.inputs = [self.input_bytes(rank) for rank in range(world)]
        self.plans = V.validate_expected(self.expected)[1]
        self.outputs = V.ring_oracle(self.inputs, dtype, self.plans)
        self.rank_entries: list[dict] = []
        self.rank_results: list[dict] = []
        self.sessions = [self.session(rank) for rank in range(world)]
        for rank in range(world):
            self.make_rank(rank)
        self.manifest = self.manifest_document()
        write_json(self.source / "result-manifest.json", self.manifest)

    def input_bytes(self, rank: int) -> bytes:
        values = [(index * 13 + rank * 29 - 63) / 16.0
                  for index in range(self.count)]
        return b"".join(V._encode_value(value, self.dtype) for value in values)

    def tuple_for(self, segment: dict, step: dict, rank: int, direction: str,
                  slot_generation: int) -> dict:
        value = V.expected_transfer(segment, step, rank, direction)
        value.update({"slot_index": 0, "slot_generation": slot_generation})
        return value

    def session(self, rank: int) -> dict:
        config = self.design["config"]
        run_nonce = hashlib.sha256(str(self.root).encode("utf-8")).digest()
        return {
            "child_pid": 4000 + rank,
            "connection_id": int.from_bytes(run_nonce[:8], "big") + rank + 1,
            "epoch": config["epoch"],
            "rank": rank,
            "world_size": self.world,
            "daemon_uuid": hashlib.sha256(
                run_nonce + rank.to_bytes(4, "big")
            ).hexdigest()[:32],
            "job_uuid": config["job_uuid"],
            "runtime_library": self.design["runtime"]["path"],
            "rank_launch_sha256": self.design["ranks"][rank]["rank_launch_sha256"],
        }

    def journal(self, rank: int) -> list[dict]:
        records: list[dict] = []
        now = 10
        generation = 1
        for segment in self.plans[rank]:
            for step in segment["steps"]:
                outbound = self.tuple_for(segment, step, rank, "outbound", generation)
                inbound = self.tuple_for(segment, step, rank, "inbound", generation)
                nonzero = (step["phase"] == V.PHASE_REDUCE_SCATTER
                           and step["receive_count_elements"] > 0)
                middle = ["device_call_enter", "device_call_returned"] if nonzero else (
                    ["zero_no_dispatch"] if step["phase"] == V.PHASE_REDUCE_SCATTER
                    else ["copy_complete"]
                )
                names = [
                    "outbound_prepared", "outbound_DATA_sent",
                    "inbound_DATA_received", "inbound_staged", *middle,
                    "inbound_CONSUMED_send_attempt", "inbound_CONSUMED_sent",
                    "outbound_CONSUMED_received_credit_released", "step_complete",
                ]
                for name in names:
                    value = {
                        "schema": V.JOURNAL_SCHEMA,
                        "ordinal": len(records),
                        "monotonic_ns": now,
                        "rank": rank,
                        "segment_id": segment["segment_id"],
                        "descriptor_sequence": segment["sequence"],
                        "step_ordinal": step["ordinal"],
                        "phase": step["phase"],
                        "phase_step_index": step["phase_step_index"],
                        "planner": {
                            "send_rank": step["send_rank"],
                            "receive_rank": step["receive_rank"],
                            "send_chunk": step["send_chunk"],
                            "receive_chunk": step["receive_chunk"],
                            "send_offset_elements": step["send_offset_elements"],
                            "send_count_elements": step["send_count_elements"],
                            "receive_offset_elements": step["receive_offset_elements"],
                            "receive_count_elements": step["receive_count_elements"],
                        },
                        "event": name,
                    }
                    if name.startswith("outbound") and name != "step_complete":
                        value["transfer"] = outbound
                    elif name in {
                        "inbound_DATA_received", "inbound_staged",
                        "inbound_CONSUMED_send_attempt", "inbound_CONSUMED_sent",
                    }:
                        value["transfer"] = inbound
                    if name == "inbound_staged":
                        width = V.DTYPES[self.dtype][1]
                        value.update({
                            "staging_sha256": hashlib.sha256(
                                b"\0" * (step["receive_count_elements"] * width)
                            ).hexdigest(),
                            "immutable_bytes": True,
                            "payload_bytes": step["receive_count_elements"] * width,
                        })
                    records.append(value)
                    now += 1
                generation += 1
        records.append({
            "schema": V.JOURNAL_SCHEMA,
            "ordinal": len(records),
            "monotonic_ns": now,
            "rank": rank,
            "segment_id": None,
            "descriptor_sequence": None,
            "step_ordinal": None,
            "event": "public_commit",
        })
        return records

    def trace(self, rank: int) -> list[dict]:
        records: list[dict] = []
        launch = 0
        for segment in self.plans[rank]:
            for step in segment["steps"]:
                if (step["phase"] == V.PHASE_REDUCE_SCATTER
                        and step["receive_count_elements"] > 0):
                    session = self.sessions[rank]
                    config = self.design["config"]
                    retired = trace_base(
                        rank, launch, step["receive_count_elements"],
                        job_uuid=config["job_uuid"], epoch=config["epoch"],
                        world=self.world, daemon_uuid=session["daemon_uuid"],
                        connection_id=session["connection_id"],
                    )
                    durable = copy.deepcopy(retired)
                    durable["event"] = "generic_execution_type20_durable"
                    durable["type20_durable"] = True
                    records.extend((retired, durable))
                    launch += 1
        grouped: list[dict] = []
        for index in range(0, len(records), 2):
            retired, durable = records[index:index + 2]
            completion = copy.deepcopy(durable)
            completion["event"] = (
                "generic_execution_session_complete"
                if index + 2 == len(records)
                else "generic_execution_reuse_complete"
            )
            completion["sim_tick"] += 100
            if completion["event"] == "generic_execution_session_complete":
                completion.update({
                    "unmap_durable": True,
                    "owner_disconnected": True,
                    "cleanup_complete": True,
                })
            grouped.extend((retired, durable, completion))
        records = grouped
        return records

    def gem5_log(self, rank: int, exit_tick: int) -> bytes:
        launch = self.design["ranks"][rank]["rank_launch"]
        paths = launch["paths"]
        session = self.sessions[rank]
        gem5 = self.identities()["gem5_binary"]["path"]
        config = self.identities()["gem5_config"]["path"]
        command = (
            f"{gem5} --listener-mode=on --outdir {paths['gem5_output_directory']} "
            f"{config} --endpoint {paths['endpoint']} "
            f"--dispatch-trace-path {paths['dispatch_trace_path']} "
            f"--epoch {launch['epoch']} --job-uuid {launch['job_uuid']} "
            f"--rank {rank} --world-size {self.world}"
        )
        return (
            f"host-gpu-ready endpoint={paths['endpoint']} "
            f"daemon_uuid={session['daemon_uuid']} job_uuid={launch['job_uuid']} "
            f"epoch={launch['epoch']} rank={rank} world={self.world} max_record=65536\n"
            f"gem5 executing on test, pid {session['child_pid']}\n"
            f"command line: {command}\n"
            "host-gpu-handshake status=OK fd=10 generation=1\n"
            "host-gpu-dispatch-exit cause=host GPU dispatch session complete "
            f"code=0 tick={exit_tick} stats={paths['gem5_output_directory']}/stats.txt\n"
        ).encode("ascii")

    def make_rank(self, rank: int) -> None:
        directory = self.source / f"rank-{rank:02d}"
        directory.mkdir()
        input_sha = hashlib.sha256(self.inputs[rank]).hexdigest()
        output_sha = hashlib.sha256(self.outputs[rank]).hexdigest()
        identities = self.identities()
        result = {
            "schema": V.RANK_RESULT_SCHEMA,
            "status": "success",
            "rank": rank,
            "world_size": self.world,
            "acceptance_authority": False,
            "live_collective_accepted": False,
            "input_sha256_before": input_sha,
            "input_sha256_after": input_sha,
            "output_sha256": output_sha,
            "output_storage_fresh": True,
            "public_commit_count": 1,
            "public_result_published": True,
            "first_error": None,
            "managed_session": copy.deepcopy(self.sessions[rank]),
            "product": {
                "product_id": self.product_id,
                "manifest_sha256": identities["product_manifest"]["sha256"],
                "prefix": str(self.product_prefix),
                "ccl_engine": identities["ccl_engine"],
            },
            # Deliberately false self-reports: verifier must derive authority.
            "device_reduction_launch_count": 999,
            "host_reduction_count": 999,
        }
        self.rank_results.append(result)
        artifacts: dict[str, dict] = {}
        trace = self.trace(rank)
        exit_tick = trace[-1]["sim_tick"] if trace else 1234
        payloads = {
            "worker-result.json": V.canonical_json(result),
            "step-journal.jsonl": b"".join(V.canonical_json(item) for item in self.journal(rank)),
            "dispatch-trace.jsonl": b"".join(V.trace_json(item) for item in trace),
            "stats.txt": (
                f"simTicks {exit_tick}\nhostSeconds 1.0\n".encode("ascii") +
                b"system.host_gpu_bridge.host_fallback_count 0\n"
                b"system.cpu1.CUs.wavefronts0.numInstrExecuted 4\n"
            ),
            "gem5.log": self.gem5_log(rank, exit_tick),
            "rank-launch.json": V.canonical_json(self.design["ranks"][rank]["rank_launch"]),
            "input.bin": self.inputs[rank],
            "output.bin": self.outputs[rank],
        }
        for name, payload in payloads.items():
            record = write(directory / name, payload)
            artifacts[name] = {
                "path": f"rank-{rank:02d}/{name}",
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
        self.rank_entries.append({
            "rank": rank,
            "worker_pid": None,
            "worker_start_time_ticks": None,
            "artifacts": artifacts,
            "cleanup": {
                "worker_reaped": True,
                "daemon_reaped": True,
            },
        })

    def identities(self) -> dict:
        paths = {
            "product_manifest": self.product_prefix / "manifest.json",
            "runtime_library": self.runtime,
            "ccl_native": ROOT / "plugins/collectives/gemsim_ccl/src/gemsim_ccl/native.py",
            "ccl_device": ROOT / "plugins/collectives/gemsim_ccl/src/gemsim_ccl/device.py",
            "ccl_engine": ROOT / "plugins/collectives/gemsim_ccl/src/gemsim_ccl/engine.py",
            "triton_driver": ROOT / "plugins/triton/gemsim_amd/backend/driver.py",
            "gem5_binary": ROOT / "LICENSE",
            "gem5_config": ROOT / "README.md",
            "verifier": V.THIS_FILE,
            "runner": V.RUNNER_FILE,
            "worker": V.WORKER_FILE,
            "bootstrap": V.BOOTSTRAP_FILE,
            "design": V.DESIGN_FILE,
            "rank_registry": ROOT / "scripts/gemsim_live_registry.py",
        }
        values = {}
        for role, path in paths.items():
            if not path.exists():
                path = ROOT / "LICENSE"
            payload = path.read_bytes()
            values[role] = {
                "path": str(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        values["runner"]["path"] = str(V.RUNNER_FILE)
        values["worker"]["path"] = str(V.WORKER_FILE)
        values["runtime_library"] = {
            "path": self.design["runtime"]["path"],
            "bytes": Path(self.design["runtime"]["path"]).stat().st_size,
            "sha256": self.design["runtime"]["sha256"],
        }
        return values

    def manifest_document(self) -> dict:
        config = self.design["config"]
        identities = self.identities()
        baseline_fds: list[dict] = []
        new_children = [
            {
                "rank": rank,
                "role": "daemon_or_descendant",
                "pid": self.sessions[rank]["child_pid"],
                "start_time_ticks": 100 + rank,
            }
            for rank in range(self.world)
        ]
        return {
            "schema": V.RUN_SCHEMA,
            "status": "success",
            "world_size": self.world,
            "element_count": self.count,
            "dtype": self.dtype,
            "job_uuid": config["job_uuid"],
            "group_uuid": config["group_uuid"],
            "epoch": config["epoch"],
            "group_generation": config["group_generation"],
            "expected": {
                "schema": V.EXPECTED_SCHEMA,
                "bytes": self.expected_record["bytes"],
                "sha256": self.expected_record["sha256"],
            },
            "source_identity_preflight": copy.deepcopy(identities),
            "source_identity_postflight": copy.deepcopy(identities),
            "acceptance_authority": False,
            "live_collective_accepted": False,
            "target_execution_completed": True,
            "target_feedback": False,
            "oracle_phase": "post_target",
            "oracle_feedback": False,
            "public_commit_count": self.world,
            "ranks": self.rank_entries,
            "first_error": None,
            "failed_transfer": None,
            "failed_ack_sent": False,
            "started_at_ns": 100,
            "completed_at_ns": 200,
            "absolute_deadline_ns": 1000,
            "supervisor_cleanup": {
                "baseline_fds": baseline_fds,
                "baseline_fd_count": len(baseline_fds),
                "baseline_fd_sha256": V.object_sha256(baseline_fds),
                "post_fds": copy.deepcopy(baseline_fds),
                "post_fd_count": len(baseline_fds),
                "post_fd_sha256": V.object_sha256(baseline_fds),
                "added_fds": [],
                "removed_fds": [],
                "measured_fd_delta": 0,
                "children_exhausted": True,
                "workers_reaped": True,
                "new_child_identities": new_children,
                "orphan_identities": [],
                "all_clear": True,
            },
        }

    def rewrite_manifest(self) -> None:
        (self.source / "result-manifest.json").unlink()
        write_json(self.source / "result-manifest.json", self.manifest)

    def rebind(self, rank: int, name: str) -> None:
        path = self.source / f"rank-{rank:02d}" / name
        payload = path.read_bytes()
        self.manifest["ranks"][rank]["artifacts"][name] = {
            "path": f"rank-{rank:02d}/{name}",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        self.rewrite_manifest()


class LiveAllreduceAcceptanceTest(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.fixture = SyntheticEvidence(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def verify(self):
        return V.verify(self.fixture.source, self.fixture.expected_path,
                        self.fixture.output, live_identity=False)

    def test_n2_success_is_authoritative_despite_false_self_report(self) -> None:
        result = self.verify()
        self.assertTrue(result["live_collective_accepted"])
        self.assertEqual(result["host_reduction_count"], 0)
        self.assertTrue((self.fixture.output / "manifest.json").is_file())
        verifier = self.fixture.output / "acceptance/verifier.py"
        self.assertTrue(verifier.is_file())
        self.assertEqual(
            result["acceptance_verifier"]["sha256"],
            hashlib.sha256(verifier.read_bytes()).hexdigest(),
        )

    def test_trace_encoding_is_fixed_order_duplicate_free_and_finite(self) -> None:
        path = self.fixture.source / "rank-00/dispatch-trace.jsonl"
        line = path.read_bytes().splitlines()[0]
        record = V.parse_trace_jsonl(line + b"\n", "trace")[0]
        self.assertEqual(tuple(record), V.TRACE_KEYS)

        reordered = {"event": record["event"], "schema": record["schema"]}
        reordered.update({key: value for key, value in record.items()
                          if key not in reordered})
        reordered_line = json.dumps(
            reordered, sort_keys=False, separators=(",", ":")
        ).encode("ascii") + b"\n"
        with self.assertRaisesRegex(V.AcceptanceError, "field order"):
            V.parse_trace_jsonl(reordered_line, "trace")

        duplicate = line.replace(b'"rank":0', b'"rank":0,"rank":0', 1) + b"\n"
        with self.assertRaisesRegex(V.AcceptanceError, "duplicate JSON key"):
            V.parse_trace_jsonl(duplicate, "trace")

        nonfinite = line.replace(
            f'"sim_tick":{record["sim_tick"]}'.encode("ascii"),
            b'"sim_tick":NaN', 1,
        ) + b"\n"
        with self.assertRaisesRegex(V.AcceptanceError, "non-finite JSON constant"):
            V.parse_trace_jsonl(nonfinite, "trace")

    def test_execution_verifier_is_historical_and_acceptance_verifier_is_current(self) -> None:
        identity = self.fixture.identities()
        identity["verifier"] = {
            "path": str(V.THIS_FILE),
            "bytes": 1,
            "sha256": "0" * 64,
        }
        validated = V.validate_identity_snapshot(identity, live=True)
        self.assertEqual(validated["verifier"], identity["verifier"])

    def test_read_induced_atime_change_is_not_false_source_drift(self) -> None:
        path = self.fixture.source / "rank-00/input.bin"
        metadata = path.stat()
        os.utime(path, ns=(1, metadata.st_mtime_ns))
        payload, record = V.file_record(path)
        self.assertEqual(payload, self.fixture.inputs[0])
        self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())

    def test_rejects_tamper_symlink_extra_and_missing_file(self) -> None:
        mutations = ("tamper", "symlink", "extra", "missing")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                fixture = SyntheticEvidence(Path(temporary))
                target = fixture.source / "rank-00/stats.txt"
                if mutation == "tamper":
                    target.write_bytes(target.read_bytes() + b"changed\n")
                elif mutation == "symlink":
                    target.unlink()
                    target.symlink_to(ROOT / "LICENSE")
                elif mutation == "extra":
                    (fixture.source / "rank-00/extra.txt").write_text("extra", encoding="ascii")
                else:
                    target.unlink()
                with self.assertRaises(V.AcceptanceError):
                    V.verify(fixture.source, fixture.expected_path, fixture.output,
                             live_identity=False)

    def test_rejects_wrong_tuple_and_ack_before_device_return(self) -> None:
        path = self.fixture.source / "rank-00/step-journal.jsonl"
        records = V.parse_jsonl(path.read_bytes(), "journal")
        records[0]["transfer"]["slot_generation"] += 1
        path.write_bytes(b"".join(V.canonical_json(item) for item in records))
        self.fixture.rebind(0, "step-journal.jsonl")
        with self.assertRaisesRegex(V.AcceptanceError, "tuple"):
            self.verify()

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fixture = SyntheticEvidence(Path(temporary))
            path = fixture.source / "rank-00/step-journal.jsonl"
            records = V.parse_jsonl(path.read_bytes(), "journal")
            returned = next(i for i, item in enumerate(records)
                            if item["event"] == "device_call_returned")
            attempt = next(i for i, item in enumerate(records)
                           if item["event"] == "inbound_CONSUMED_send_attempt")
            records[returned]["event"], records[attempt]["event"] = (
                records[attempt]["event"], records[returned]["event"]
            )
            for ordinal, record in enumerate(records):
                record["ordinal"] = ordinal
            path.write_bytes(b"".join(V.canonical_json(item) for item in records))
            fixture.rebind(0, "step-journal.jsonl")
            with self.assertRaisesRegex(V.AcceptanceError, "event sequence"):
                V.verify(fixture.source, fixture.expected_path, fixture.output,
                         live_identity=False)

    def test_rejects_zero_dispatch_and_unmatched_trace(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fixture = SyntheticEvidence(Path(temporary), count=1)
            with self.assertRaisesRegex(V.AcceptanceError, "zero device dispatches"):
                V.verify(fixture.source, fixture.expected_path, fixture.output,
                         live_identity=False)
        for mutation in ("drop", "extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                fixture = SyntheticEvidence(Path(temporary), count=1)
                path = fixture.source / "rank-00/dispatch-trace.jsonl"
                records = V.parse_trace_jsonl(path.read_bytes(), "trace")
                if mutation == "drop":
                    del records[:2]
                else:
                    records[0:0] = copy.deepcopy(records[:2])
                path.write_bytes(b"".join(V.trace_json(item) for item in records))
                fixture.rebind(0, "dispatch-trace.jsonl")
                with self.assertRaisesRegex(V.AcceptanceError, "trace dispatch count"):
                    V.verify(fixture.source, fixture.expected_path, fixture.output,
                             live_identity=False)

    def test_rejects_cross_rank_trace_replay(self) -> None:
        source = self.fixture.source / "rank-01/dispatch-trace.jsonl"
        target = self.fixture.source / "rank-00/dispatch-trace.jsonl"
        target.write_bytes(source.read_bytes())
        self.fixture.rebind(0, "dispatch-trace.jsonl")
        with self.assertRaisesRegex(V.AcceptanceError, "trace run/rank identity"):
            self.verify()

    def test_rejects_gem5_command_and_log_identity_replay(self) -> None:
        for mutation in ("binary", "config", "endpoint", "other_run"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                fixture = SyntheticEvidence(Path(temporary))
                path = fixture.source / "rank-00/gem5.log"
                text = path.read_text(encoding="ascii")
                if mutation == "binary":
                    text = text.replace(
                        fixture.identities()["gem5_binary"]["path"],
                        str(ROOT / "not-the-bound-gem5"), 1,
                    )
                elif mutation == "config":
                    text = text.replace(
                        fixture.identities()["gem5_config"]["path"],
                        str(ROOT / "not-the-bound-config.py"), 1,
                    )
                elif mutation == "endpoint":
                    endpoint = fixture.design["ranks"][0]["rank_launch"]["paths"]["endpoint"]
                    text = text.replace(endpoint, endpoint + ".replayed", 1)
                else:
                    with tempfile.TemporaryDirectory(dir=ROOT) as replay_root:
                        replay = SyntheticEvidence(Path(replay_root))
                        text = (replay.source / "rank-00/gem5.log").read_text(encoding="ascii")
                path.write_text(text, encoding="ascii")
                fixture.rebind(0, "gem5.log")
                with self.assertRaises(V.AcceptanceError):
                    V.verify(fixture.source, fixture.expected_path, fixture.output,
                             live_identity=False)

    def test_multi_dispatch_reuse_chain_and_splice_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fixture = SyntheticEvidence(Path(temporary), world=3, count=5)
            result = V.verify(fixture.source, fixture.expected_path, fixture.output,
                              live_identity=False)
            self.assertTrue(result["live_collective_accepted"])
            self.assertTrue(all(item["reuse_complete_count"] == 1
                                for item in result["rank_traces"]))
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fixture = SyntheticEvidence(Path(temporary), world=3, count=5)
            path = fixture.source / "rank-00/dispatch-trace.jsonl"
            records = V.parse_trace_jsonl(path.read_bytes(), "trace")
            records[2]["sim_tick"] += 1
            path.write_bytes(b"".join(V.trace_json(item) for item in records))
            fixture.rebind(0, "dispatch-trace.jsonl")
            with self.assertRaisesRegex(V.AcceptanceError, "handoff"):
                V.verify(fixture.source, fixture.expected_path, fixture.output,
                         live_identity=False)

    def test_rejects_source_drift_and_false_acceptance_self_report(self) -> None:
        self.fixture.manifest["source_identity_postflight"]["design"]["sha256"] = "f" * 64
        self.fixture.rewrite_manifest()
        with self.assertRaisesRegex(V.AcceptanceError, "identity drifted"):
            self.verify()
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            fixture = SyntheticEvidence(Path(temporary))
            fixture.manifest["live_collective_accepted"] = True
            fixture.rewrite_manifest()
            with self.assertRaisesRegex(V.AcceptanceError, "falsely claims"):
                V.verify(
                    fixture.source,
                    fixture.expected_path,
                    fixture.output,
                    live_identity=False,
                )

    def test_rejects_legacy_identity_without_engine_role(self) -> None:
        for phase in ("source_identity_preflight", "source_identity_postflight"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                fixture = SyntheticEvidence(Path(temporary))
                del fixture.manifest[phase]["ccl_engine"]
                fixture.rewrite_manifest()
                with self.assertRaisesRegex(
                    V.AcceptanceError, "source identity roles differ"
                ):
                    V.verify(
                        fixture.source,
                        fixture.expected_path,
                        fixture.output,
                        live_identity=False,
                    )

    def test_rejects_rank_engine_binding_rewrite(self) -> None:
        self.fixture.rank_results[0]["product"]["ccl_engine"]["sha256"] = "f" * 64
        result_path = self.fixture.source / "rank-00/worker-result.json"
        result_path.write_bytes(V.canonical_json(self.fixture.rank_results[0]))
        self.fixture.rebind(0, "worker-result.json")
        with self.assertRaisesRegex(
            V.AcceptanceError, "product/CCL engine execution binding mismatch"
        ):
            self.verify()

    def test_supervisor_cleanup_is_recomputed_and_fail_closed(self) -> None:
        for mutation in ("fd_delta", "fd_set", "orphan", "rank_cleanup"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                fixture = SyntheticEvidence(Path(temporary))
                cleanup = fixture.manifest["supervisor_cleanup"]
                if mutation == "fd_delta":
                    cleanup["measured_fd_delta"] = 1
                elif mutation == "fd_set":
                    fake_fd = {
                        "fd": 99,
                        "device": 1,
                        "inode": 2,
                        "mode": 0o100600,
                        "target": "/tmp/replayed-fd",
                    }
                    cleanup["added_fds"] = [fake_fd]
                    cleanup["measured_fd_delta"] = 1
                    cleanup["all_clear"] = False
                elif mutation == "orphan":
                    cleanup["orphan_identities"] = [
                        copy.deepcopy(cleanup["new_child_identities"][0])
                    ]
                    cleanup["all_clear"] = False
                else:
                    fixture.manifest["ranks"][0]["cleanup"][
                        "daemon_reaped"
                    ] = False
                fixture.rewrite_manifest()
                with self.assertRaisesRegex(
                    V.AcceptanceError,
                    "cleanup|FD set algebra|FD delta|lifecycle",
                ):
                    V.verify(
                        fixture.source,
                        fixture.expected_path,
                        fixture.output,
                        live_identity=False,
                    )

    def test_timeout_relation_is_exact(self) -> None:
        ranks = [
            {
                "first_error": {"status": "timed_out", "context_sequence": 1},
                "public_commit_count": 0,
                "public_result_published": False,
                "failed_transfer": {
                    "descriptor_sha256": "a" * 64, "sequence": 1,
                    "phase": 1, "step_index": 0, "chunk_index": 0,
                    "source_rank": 0, "destination_rank": 1,
                    "slot_index": 0, "slot_generation": 1,
                },
            }
            for _ in range(2)
        ]
        manifest = {
            "first_error": ranks[0]["first_error"],
            "failed_transfer": ranks[0]["failed_transfer"],
            "failed_ack_sent": False,
            "public_commit_count": 0,
            "started_at_ns": 100,
            "completed_at_ns": 1000,
            "absolute_deadline_ns": 1000,
        }
        V.validate_failure(manifest, ranks, "timed_out", 2)
        manifest["completed_at_ns"] = 999
        with self.assertRaisesRegex(V.AcceptanceError, "timeout"):
            V.validate_failure(manifest, ranks, "timed_out", 2)
        manifest["completed_at_ns"] = 999
        manifest["first_error"] = {"status": "peer_lost", "context_sequence": 1}
        for rank in ranks:
            rank["first_error"] = manifest["first_error"]
        V.validate_failure(manifest, ranks, "peer_lost", 2)

    def test_world_schema_matrix_and_formal_subset(self) -> None:
        self.assertEqual(V.ALL_WORLDS, tuple(range(2, 17)))
        self.assertEqual(V.FORMAL_WORLDS, (2, 3, 4, 8, 16))
        for world in V.ALL_WORLDS:
            plans = [V.independent_plan(rank, world, 1, 4, 0)
                     for rank in range(world)]
            self.assertTrue(all(len(plan) == 2 * (world - 1) for plan in plans))

    def test_rejects_arithmetic_policy_drift(self) -> None:
        self.assertEqual(D.EXPECTED_SCHEMA, V.EXPECTED_SCHEMA)
        self.assertEqual(D.ARITHMETIC_POLICY, V.ARITHMETIC_POLICY)
        self.assertEqual(
            D.build_expected_wrapper(self.fixture.design), self.fixture.expected
        )
        self.fixture.expected["arithmetic_policy"]["all_gather"] = "numeric-copy"
        write_json(self.fixture.expected_path, self.fixture.expected)
        with self.assertRaisesRegex(V.AcceptanceError, "arithmetic policy"):
            V.verify(self.fixture.source, self.fixture.expected_path,
                     self.fixture.output, live_identity=False)

    def test_bf16_and_fp32_oracles_are_bitwise_and_versioned(self) -> None:
        for dtype in V.DTYPES:
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                fixture = SyntheticEvidence(Path(temporary), dtype=dtype)
                outputs = V.ring_oracle(fixture.inputs, dtype, fixture.plans)
                self.assertEqual(len(set(outputs)), 1)
                self.assertEqual(outputs, fixture.outputs)
        policy = V.ARITHMETIC_POLICY
        self.assertEqual(policy["schema"], "amdgpu-sim.ccl-ring-sum-arithmetic.v1")
        self.assertFalse(policy["oracle_feedback"])


if __name__ == "__main__":
    unittest.main()
