"""Attribution of managed simulators to the lane that started them.

Two production failures came from getting this wrong, and both are pinned here:
a lane's progress was computed by pooling every run directory on the box, and a
cleanup killed a healthy lane because it matched processes by command-line
substring. Both are cheap to regress into, so the module is tested against real
processes and real run directories rather than mocks.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import textwrap
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/lane_ownership.sh"


def call(function: str, *args: str) -> str:
    """Run one function from the ownership module and return its stdout."""
    script = f'source "{MODULE}"; {function} ' + " ".join(
        f'"{argument}"' for argument in args
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


class LaneRunRootTest(unittest.TestCase):
    def test_root_is_per_lane(self) -> None:
        self.assertNotEqual(call("lane_run_root", "a"), call("lane_run_root", "b"))

    def test_root_leaves_room_for_the_bridge_socket(self) -> None:
        # sun_path is 108 bytes. The run directory adds
        # "/self-amdgpu-opencl-run.<uid>.XXXXXX/bridge.sock" on top of the root,
        # and exceeding the cap fails at bind() deep inside a 50-minute run.
        root = call("lane_run_root", "sglang-tp16")
        suffix = len(f"/self-amdgpu-opencl-run.{os.getuid()}.XXXXXX/bridge.sock")
        self.assertLess(len(root) + suffix, 108)


class LaneDispatchTotalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.roots = []

    def tearDown(self) -> None:
        for root in self.roots:
            shutil.rmtree(root, ignore_errors=True)

    def seed(self, lane: str, *record_counts: int) -> None:
        root = Path(call("lane_run_root", lane))
        self.roots.append(root)
        shutil.rmtree(root, ignore_errors=True)
        for index, count in enumerate(record_counts):
            run = root / f"self-amdgpu-opencl-run.{os.getuid()}.a{index:05d}"
            run.mkdir(parents=True)
            (run / "dispatch-trace.jsonl").write_text("{}\n" * count)

    def test_counts_only_this_lane(self) -> None:
        self.seed("ownership-test-one", 7, 5)
        self.seed("ownership-test-two", 1000)
        # 12, not 1012: pooling the two lanes is the exact defect that reported
        # a 38% lane as 92%.
        self.assertEqual(call("lane_dispatch_total", "ownership-test-one"), "12")
        self.assertEqual(call("lane_dispatch_total", "ownership-test-two"), "1000")

    def test_absent_lane_counts_zero(self) -> None:
        self.assertEqual(call("lane_dispatch_total", "ownership-test-absent"), "0")

    def test_counts_survive_the_simulator_exiting(self) -> None:
        # A finished lane must still report what it achieved; requiring a live
        # process would erase the result at the moment it becomes interesting.
        self.seed("ownership-test-done", 900)
        self.assertEqual(call("lane_dispatch_total", "ownership-test-done"), "900")


class LaneReapTest(unittest.TestCase):
    """Reaping must hit this lane's simulators and nothing else."""

    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="lane-reap-test."))
        # A stand-in for gem5: argv carries an --outdir under a lane run root,
        # which is the only signal the real attribution has to work from.
        self.fake = self.workspace / "gem5.opt"
        self.fake.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                sleep 300
                """
            )
        )
        self.fake.chmod(0o755)
        self.started: list[subprocess.Popen] = []

    def tearDown(self) -> None:
        for process in self.started:
            try:
                process.kill()
            except OSError:
                pass
        shutil.rmtree(self.workspace, ignore_errors=True)
        for lane in ("reap-test-mine", "reap-test-theirs"):
            shutil.rmtree(call("lane_run_root", lane), ignore_errors=True)

    def start(self, lane: str) -> subprocess.Popen:
        root = call("lane_run_root", lane)
        outdir = f"{root}/self-amdgpu-opencl-run.{os.getuid()}.aaaaaa/m5out"
        os.makedirs(outdir, exist_ok=True)
        process = subprocess.Popen([str(self.fake), "--outdir", outdir])
        self.started.append(process)
        return process

    def test_reap_spares_other_lanes(self) -> None:
        mine = self.start("reap-test-mine")
        theirs = self.start("reap-test-theirs")
        time.sleep(0.5)

        pids = call("lane_gem5_pids", "reap-test-mine").split()
        self.assertIn(str(mine.pid), pids)
        self.assertNotIn(str(theirs.pid), pids)

        call("lane_reap_gem5", "reap-test-mine")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and mine.poll() is None:
            time.sleep(0.05)
        self.assertIsNotNone(mine.poll(), "this lane's simulator was not reaped")
        self.assertIsNone(theirs.poll(), "another lane's simulator was killed")


if __name__ == "__main__":
    unittest.main()
