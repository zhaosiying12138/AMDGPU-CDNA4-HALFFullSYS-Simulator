from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "projects/self-amdgpu-runtime/src/rocminfo_model.c"


def _binary() -> Path | None:
    for candidate in sorted(
        (ROOT / "projects/self-amdgpu-runtime").glob("build*/**/sagr-rocminfo")
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _strip_c_comments(text: str) -> str:
    output, index, length = [], 0, len(text)
    while index < length:
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = length if end == -1 else end + 2
        elif text.startswith("//", index):
            end = text.find("\n", index)
            index = length if end == -1 else end
        else:
            output.append(text[index])
            index += 1
    return "".join(output)


def _write_topology(root: Path, nodes: dict[int, tuple[int, int]]) -> Path:
    """nodes maps node id -> (simd_count, gfx_target_version)."""
    topology = root / "hsakmt-topology"
    for node_id, (simd_count, gfx) in nodes.items():
        node = topology / "nodes" / str(node_id)
        node.mkdir(parents=True, exist_ok=True)
        (node / "properties").write_text(
            f"cpu_cores_count {0 if simd_count else 1}\n"
            f"simd_count {simd_count}\n"
            f"gfx_target_version {gfx}\n",
            encoding="ascii",
        )
    return topology


class RocminfoModelSourceTest(unittest.TestCase):
    def test_never_probes_hardware(self) -> None:
        code = _strip_c_comments(SOURCE.read_text(encoding="ascii"))
        for forbidden in ("/dev/kfd", "/dev/dri", "/sys/class/kfd", "/sys/module"):
            self.assertNotIn(forbidden, code)

    def test_reads_the_model_topology(self) -> None:
        self.assertIn("HSA_MODEL_TOPOLOGY", SOURCE.read_text(encoding="ascii"))


class RocminfoModelRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        binary = _binary()
        if binary is None:
            self.skipTest("sagr-rocminfo has not been built")
        self.binary = binary

    def _run(self, topology: str | None) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        if topology is None:
            environment.pop("HSA_MODEL_TOPOLOGY", None)
        else:
            environment["HSA_MODEL_TOPOLOGY"] = topology
        return subprocess.run(
            [str(self.binary)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False, env=environment,
        )

    def test_fails_closed_without_topology(self) -> None:
        completed = self._run(None)
        self.assertNotEqual(completed.returncode, 0)
        self.assertNotIn("gfx", completed.stdout)

    def test_fails_closed_on_relative_topology(self) -> None:
        completed = self._run("relative/path")
        self.assertNotEqual(completed.returncode, 0)

    def test_fails_closed_when_no_simd_bearing_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            topology = _write_topology(Path(directory), {0: (0, 0)})
            completed = self._run(str(topology))
            self.assertNotEqual(completed.returncode, 0)
            self.assertNotIn("gfx", completed.stdout)

    def test_reports_gfx950_from_target_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            topology = _write_topology(Path(directory), {0: (0, 0), 1: (1024, 90500)})
            completed = self._run(str(topology))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("gfx950", completed.stdout)
            self.assertIn("amdgcn-amd-amdhsa--gfx950", completed.stdout)
            self.assertIn("Device Type:             GPU", completed.stdout)

    def test_derives_name_from_topology_not_a_constant(self) -> None:
        # A different target version must produce a different name, proving the
        # value is read rather than hard-coded.
        with tempfile.TemporaryDirectory() as directory:
            # gfx1201 encodes as major 12, minor 0, step 1 -> 120001.
            topology = _write_topology(Path(directory), {0: (0, 0), 1: (1024, 120001)})
            completed = self._run(str(topology))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("gfx1201", completed.stdout)
            self.assertNotIn("gfx950", completed.stdout)

    def test_reports_each_gpu_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            topology = _write_topology(
                Path(directory), {0: (0, 0), 1: (1024, 90500), 2: (1024, 90500)}
            )
            completed = self._run(str(topology))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.count("Device Type:             GPU"), 2)


if __name__ == "__main__":
    unittest.main()
