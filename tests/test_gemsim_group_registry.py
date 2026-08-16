# SPDX-License-Identifier: GPL-3.0-or-later
"""Group topology, live registry, and simulator-only SMI tests."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import uuid


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SUPERVISOR = SCRIPTS / "run_gemsim_instances.py"
SMI = SCRIPTS / "gemsim_smi.py"
WORKER = ROOT / "tests/fixtures/gemsim_instance_worker.py"
sys.path.insert(0, str(SCRIPTS))

from gemsim_live_registry import (  # noqa: E402
    LIVE_REGISTRY_SCHEMA,
    RANK_LAUNCH_SCHEMA,
    LiveRegistryPublisher,
    RegistryError,
    load_rank_launch,
    make_rank_launch,
    read_live_snapshot,
    validate_rank_launch,
    validate_rank_launch_group,
    write_rank_launch,
)


DRIVER_SPEC = importlib.util.spec_from_file_location(
    "gemsim_driver_for_group_tests",
    ROOT / "plugins/triton/gemsim_amd/backend/driver.py",
)
assert DRIVER_SPEC and DRIVER_SPEC.loader
driver_module = importlib.util.module_from_spec(DRIVER_SPEC)
DRIVER_SPEC.loader.exec_module(driver_module)


def private_directory(path: Path) -> Path:
    path.mkdir(parents=True, mode=0o700)
    path.chmod(0o700)
    return path


def descriptors(root: Path, count: int) -> list[dict]:
    job_uuid = uuid.uuid4().hex
    epoch = time.monotonic_ns() or 1
    result = []
    for rank in range(count):
        namespace = root / f"instance-{rank:03d}"
        result.append(
            make_rank_launch(
                job_uuid=job_uuid,
                epoch=epoch,
                rank=rank,
                world_size=count,
                instance_directory=namespace / "correctness",
                triton_cache_directory=namespace / "cache/triton",
            )
        )
    return result


def registry_ranks(values: list[dict], state: str) -> list[dict]:
    result = []
    for value in values:
        paths = value["paths"]
        ready = state == "READY"
        result.append(
            {
                "rank": value["rank"],
                "world_size": value["world_size"],
                "state": state,
                "worker_pid": os.getpid(),
                "daemon_pid": os.getpid() if ready else None,
                "daemon_uuid": uuid.uuid4().hex if ready else None,
                "endpoint": paths["endpoint"],
                "runtime_directory": paths["runtime_directory"],
                "triton_cache_directory": paths["triton_cache_directory"],
                "gem5_cache_directory": paths["gem5_cache_directory"],
            }
        )
    return result


class RankLaunchSchemaTest(unittest.TestCase):
    maxDiff = None

    def test_exact_group_descriptors_accept_all_worlds_two_through_sixteen(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for count in range(2, 17):
                with self.subTest(count=count):
                    values = descriptors(root / str(count), count)
                    validated = validate_rank_launch_group(values)
                    self.assertEqual([value["rank"] for value in validated], list(range(count)))
                    self.assertEqual({value["schema"] for value in validated}, {RANK_LAUNCH_SCHEMA})
                    self.assertEqual(len({value["job_uuid"] for value in validated}), 1)
                    self.assertEqual(len({value["epoch"] for value in validated}), 1)

    def test_duplicate_missing_and_mixed_identity_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = descriptors(Path(temporary), 4)
            duplicate = copy.deepcopy(values)
            duplicate[1]["rank"] = 0
            with self.assertRaisesRegex(RegistryError, "ranks must be exactly"):
                validate_rank_launch_group(duplicate)

            with self.assertRaisesRegex(RegistryError, "count does not equal"):
                validate_rank_launch_group(values[:-1])

            mixed = copy.deepcopy(values)
            mixed[2]["job_uuid"] = uuid.uuid4().hex
            with self.assertRaisesRegex(RegistryError, "mixed identity"):
                validate_rank_launch_group(mixed)

            shared = copy.deepcopy(values)
            shared[1]["paths"] = copy.deepcopy(shared[0]["paths"])
            with self.assertRaisesRegex(RegistryError, "path is shared"):
                validate_rank_launch_group(shared)

    def test_managed_endpoint_limit_is_rejected_by_the_shared_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = descriptors(root, 2)[0]
            rank_root = root / ("x" * 100)
            instance = rank_root / "correctness"
            runtime = instance / "runtime"
            value["paths"]["instance_directory"] = str(instance)
            value["paths"]["runtime_directory"] = str(runtime)
            value["paths"]["triton_cache_directory"] = str(
                rank_root / "cache/triton"
            )
            for name, leaf in {
                "endpoint": "bridge.sock",
                "gem5_output_directory": "m5out",
                "dispatch_trace_path": "dispatch-trace.jsonl",
                "gem5_log_path": "gem5.log",
                "gem5_cache_directory": "cache",
            }.items():
                value["paths"][name] = str(runtime / leaf)
            with self.assertRaisesRegex(RegistryError, "path limit"):
                validate_rank_launch(value)

    def test_driver_consumes_canonical_descriptor_without_ambient_topology(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            root.chmod(0o700)
            value = descriptors(root, 2)[1]
            path = root / "rank-launch.json"
            write_rank_launch(path, value)
            with mock.patch.dict(
                os.environ,
                {
                    "GEMSIM_RANK_LAUNCH_DESCRIPTOR": str(path),
                    "TRITON_CACHE_DIR": value["paths"]["triton_cache_directory"],
                    "RANK": "99",
                    "WORLD_SIZE": "100",
                },
                clear=True,
            ):
                parsed = driver_module._rank_launch_descriptor()
            self.assertEqual(parsed["job_uuid"], bytes.fromhex(value["job_uuid"]))
            self.assertEqual(parsed["epoch"], value["epoch"])
            self.assertEqual(parsed["rank"], 1)
            self.assertEqual(parsed["world_size"], 2)
            self.assertEqual(parsed["endpoint"], value["paths"]["endpoint"].encode())

    def test_driver_rejects_descriptor_cache_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            root.chmod(0o700)
            value = descriptors(root, 2)[0]
            path = root / "rank-launch.json"
            write_rank_launch(path, value)
            with mock.patch.dict(
                os.environ,
                {
                    "GEMSIM_RANK_LAUNCH_DESCRIPTOR": str(path),
                    "TRITON_CACHE_DIR": str(root / "wrong-cache"),
                },
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    driver_module._rank_launch_descriptor()

    def test_runtime_rejects_inherited_fork_owner(self) -> None:
        runtime = driver_module._ManagedRuntime.__new__(driver_module._ManagedRuntime)
        runtime.lock = driver_module.threading.RLock()
        runtime.session = driver_module.ctypes.c_void_p(123)
        runtime.session_info = object()
        runtime.owner_pid = os.getpid()
        runtime.forked_child = False
        runtime._after_fork_child()
        self.assertFalse(runtime.session.value)
        self.assertIsNone(runtime.session_info)
        with self.assertRaisesRegex(RuntimeError, "inherited across fork"):
            runtime._check_owner()

    def test_rank_launch_is_canonical_read_only_and_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = private_directory(Path(temporary) / "registry")
            value = descriptors(root, 2)[0]
            path = root / "rank-launch.json"
            write_rank_launch(path, value)
            self.assertEqual(load_rank_launch(path), value)
            self.assertEqual(stat.S_IMODE(path.lstat().st_mode), 0o400)
            self.assertEqual(
                path.read_bytes(),
                (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"),
            )

            link = root / "rank-link.json"
            link.symlink_to(path.name)
            with self.assertRaisesRegex(RegistryError, "symlink"):
                load_rank_launch(link)


class LiveRegistrySchemaTest(unittest.TestCase):
    def make_publisher(self, temporary: str, count: int = 2):
        root = private_directory(Path(temporary) / "live")
        values = descriptors(root / "namespaces", count)
        publisher = LiveRegistryPublisher(root / "registry.json")
        return root, values, publisher

    def test_tamper_and_generation_rollback_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, values, publisher = self.make_publisher(temporary)
            try:
                first = publisher.base_document(
                    state="STARTING",
                    job_uuid=values[0]["job_uuid"],
                    epoch=values[0]["epoch"],
                    world_size=2,
                    ranks=registry_ranks(values, "STARTING"),
                )
                publisher.publish(first)
                first_data = publisher.registry_path.read_bytes()
                second = publisher.base_document(
                    state="READY",
                    job_uuid=values[0]["job_uuid"],
                    epoch=values[0]["epoch"],
                    world_size=2,
                    ranks=registry_ranks(values, "READY"),
                )
                publisher.publish(second)
                snapshot = read_live_snapshot(publisher.registry_path)
                self.assertTrue(snapshot.lease_held)
                self.assertEqual(snapshot.registry["generation"], 2)

                publisher.registry_path.write_bytes(first_data)
                with self.assertRaisesRegex(RegistryError, "generation rollback"):
                    read_live_snapshot(publisher.registry_path, retries=1)

                publisher.registry_path.write_bytes(
                    (json.dumps(second, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
                )
                tampered = copy.deepcopy(second)
                tampered["job_uuid"] = uuid.uuid4().hex
                publisher.registry_path.write_bytes(
                    (json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
                )
                with self.assertRaisesRegex(RegistryError, "tamper"):
                    read_live_snapshot(publisher.registry_path, retries=1)
            finally:
                publisher.close()

    def test_registry_symlinks_and_generation_reuse_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, values, publisher = self.make_publisher(temporary)
            try:
                document = publisher.base_document(
                    state="STARTING",
                    job_uuid=values[0]["job_uuid"],
                    epoch=values[0]["epoch"],
                    world_size=2,
                    ranks=registry_ranks(values, "STARTING"),
                )
                publisher.publish(document)
                with self.assertRaisesRegex(RegistryError, "advance exactly once"):
                    publisher.publish(document)
                link = root / "registry-link.json"
                link.symlink_to(publisher.registry_path.name)
                with self.assertRaisesRegex(RegistryError, "symlink"):
                    read_live_snapshot(link, retries=1)
            finally:
                publisher.close()


class GemsimGroupSupervisorTest(unittest.TestCase):
    maxDiff = None

    def invoke(
        self,
        temporary: Path,
        count: int,
        mode: str = "success",
        extra: list[str] | None = None,
        timeout: float = 20.0,
    ) -> tuple[subprocess.CompletedProcess[str], dict]:
        output = temporary / "supervisor-output"
        result = subprocess.run(
            [
                sys.executable,
                str(SUPERVISOR),
                "--instances",
                str(count),
                "--topology",
                "group",
                "--output-dir",
                str(output),
                *(extra or []),
                "--",
                sys.executable,
                str(WORKER),
                mode,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        manifest_path = output / "manifest.json"
        self.assertTrue(manifest_path.is_file(), result.stderr)
        return result, json.loads(manifest_path.read_text(encoding="utf-8"))

    def smi(self, registry: Path, json_output: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SMI), "--registry", str(registry), *( ["--json"] if json_output else [])],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
            check=False,
        )

    def test_group_representative_worlds_have_exact_identity_and_namespaces(
        self,
    ) -> None:
        for count in (2, 3, 8, 16):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temporary:
                result, manifest = self.invoke(Path(temporary), count)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(manifest["passed"])
                self.assertEqual(manifest["topology"], "group")
                runs = manifest["runs"]
                runtimes = [run["runtime_processes"][0] for run in runs]
                self.assertEqual(sorted(runtime["rank"] for runtime in runtimes), list(range(count)))
                self.assertEqual({runtime["world_size"] for runtime in runtimes}, {count})
                self.assertEqual(len({runtime["job_uuid"] for runtime in runtimes}), 1)
                self.assertEqual(len({runtime["epoch"] for runtime in runtimes}), 1)
                for field in (
                    "daemon_uuid",
                    "endpoint",
                    "run_dir",
                    "output_dir",
                    "trace_path",
                    "gem5_cache_dir",
                ):
                    self.assertEqual(len({runtime[field] for runtime in runtimes}), count, field)
                for run, runtime in zip(runs, runtimes, strict=True):
                    launch = run["rank_launch_descriptor"]
                    self.assertEqual(launch["document"]["rank"], runtime["rank"])
                    self.assertEqual(launch["document"]["job_uuid"], runtime["job_uuid"])
                    self.assertEqual(stat.S_IMODE(Path(launch["path"]).lstat().st_mode), 0o400)
                    for path in (
                        Path(run["directory"]).parent,
                        Path(run["directory"]),
                        Path(run["cache_dir"]),
                        Path(runtime["run_dir"]),
                        Path(runtime["output_dir"]),
                        Path(runtime["gem5_cache_dir"]),
                    ):
                        self.assertEqual(stat.S_IMODE(path.lstat().st_mode), 0o700, path)

                registry = Path(manifest["namespace_contract"]["live_registry_path"])
                smi = self.smi(registry)
                self.assertEqual(smi.returncode, 0, smi.stderr)
                document = json.loads(smi.stdout)
                self.assertEqual(document["schema"], "amdgpu-sim.gemsim-smi.v1")
                self.assertEqual(document["status"], "OFF")
                self.assertEqual(document["registry_state"], "OFF")
                self.assertFalse(document["lease_held"])
                self.assertEqual(document["world_size"], count)
                table = self.smi(registry, json_output=False)
                self.assertEqual(table.returncode, 0, table.stderr)
                self.assertIn("STATUS", table.stdout)
                self.assertEqual(table.stdout.count(" OFF "), count)

    def test_descriptor_tamper_fails_group_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, manifest = self.invoke(Path(temporary), 2, "tamper-descriptor")
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse(manifest["passed"])
            for run in manifest["runs"]:
                self.assertTrue(
                    any("descriptor" in error for error in run["isolation_errors"]),
                    run["isolation_errors"],
                )

    def test_rank_failure_aborts_group_and_publishes_off(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, manifest = self.invoke(Path(temporary), 2, "fail-one")
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse(manifest["passed"])
            reason = manifest["namespace_contract"]["group_abort_reason"]
            self.assertEqual(reason, "rank 0 exited with status 9")
            runs = sorted(manifest["runs"], key=lambda run: run["index"])
            self.assertEqual(runs[0]["returncode"], 9)
            self.assertNotEqual(runs[1]["returncode"], 0)
            self.assertFalse((Path(runs[1]["directory"]) / "completed").exists())
            for run in runs:
                self.assertIn(f"group aborted: {reason}", run["isolation_errors"])
            registry = Path(manifest["namespace_contract"]["live_registry_path"])
            document = json.loads(self.smi(registry).stdout)
            self.assertEqual(document["status"], "OFF")
            self.assertTrue(all(rank["daemon_pid"] is None for rank in document["ranks"]))
            self.assertTrue(all(rank["daemon_uuid"] is None for rank in document["ranks"]))

    def test_ready_daemon_loss_aborts_all_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, manifest = self.invoke(
                Path(temporary), 2, "daemon-dies", timeout=10.0
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            reason = manifest["namespace_contract"]["group_abort_reason"]
            self.assertEqual(reason, "rank 0 daemon identity was lost")
            for run in manifest["runs"]:
                self.assertIn(f"group aborted: {reason}", run["isolation_errors"])
            registry = Path(manifest["namespace_contract"]["live_registry_path"])
            document = json.loads(self.smi(registry).stdout)
            self.assertEqual(document["registry_state"], "OFF")
            self.assertEqual(document["status"], "OFF")

    def test_starting_and_ready_are_live_only_while_lease_is_held(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            output = temporary_path / "supervisor-output"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(SUPERVISOR),
                    "-n",
                    "2",
                    "--topology",
                    "group",
                    "--output-dir",
                    str(output),
                    "--",
                    sys.executable,
                    str(WORKER),
                    "delay-hold",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            registry = output / "live-registry.json"
            ready_document = None
            starting_document = None
            raw_registry = None
            deadline = time.monotonic() + 8.0
            try:
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        stdout, stderr = process.communicate()
                        self.fail(f"supervisor exited early: {stdout}\n{stderr}")
                    if registry.is_file():
                        result = self.smi(registry)
                        if result.returncode == 0:
                            document = json.loads(result.stdout)
                            if document["registry_state"] == "STARTING":
                                starting_document = document
                            if document["registry_state"] == "READY":
                                ready_document = document
                                raw_registry = json.loads(registry.read_text(encoding="ascii"))
                                break
                    time.sleep(0.02)
                self.assertIsNotNone(ready_document, "group never reached READY")
                self.assertIsNotNone(starting_document, "group never exposed STARTING")
                assert starting_document is not None
                self.assertEqual(starting_document["status"], "OFF")
                self.assertTrue(starting_document["lease_held"])
                self.assertTrue(
                    all(rank["daemon_pid"] is None for rank in starting_document["ranks"])
                )
                assert ready_document is not None
                self.assertEqual(ready_document["status"], "ON")
                self.assertTrue(ready_document["lease_held"])
                self.assertTrue(all(rank["daemon_pid"] for rank in ready_document["ranks"]))

                os.kill(process.pid, signal.SIGKILL)
                process.wait(timeout=3.0)
                process.communicate(timeout=1.0)
                after = self.smi(registry)
                self.assertEqual(after.returncode, 0, after.stderr)
                after_document = json.loads(after.stdout)
                self.assertEqual(after_document["registry_state"], "READY")
                self.assertEqual(after_document["status"], "OFF")
                self.assertFalse(after_document["lease_held"])
            finally:
                if process.poll() is None:
                    os.kill(process.pid, signal.SIGKILL)
                    process.wait(timeout=3.0)
                    process.communicate(timeout=1.0)
                if raw_registry is None and registry.is_file():
                    raw_registry = json.loads(registry.read_text(encoding="ascii"))
                if raw_registry is not None:
                    identities = {
                        pid
                        for rank in raw_registry["ranks"]
                        for pid in (rank["worker_pid"], rank["daemon_pid"])
                        if pid is not None
                    }
                    for pid in identities:
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def test_group_rejects_shared_cache_and_external_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            cache = private_directory(temporary_path / "cache")
            cache.chmod(0o555)
            output = temporary_path / "output"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SUPERVISOR),
                    "-n",
                    "2",
                    "--topology",
                    "group",
                    "--output-dir",
                    str(output),
                    "--shared-readonly-cache",
                    str(cache),
                    "--",
                    sys.executable,
                    str(WORKER),
                    "success",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("private writable Triton cache", result.stderr)

            external = private_directory(temporary_path / "external") / "registry.json"
            second_output = temporary_path / "second-output"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SUPERVISOR),
                    "-n",
                    "2",
                    "--topology",
                    "group",
                    "--output-dir",
                    str(second_output),
                    "--live-registry",
                    str(external),
                    "--",
                    sys.executable,
                    str(WORKER),
                    "success",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("direct child", result.stderr)
            self.assertFalse(external.exists())


if __name__ == "__main__":
    unittest.main()
