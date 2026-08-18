"""Cross-repository invariants for the multi-GPU KFD model.

The model DSO publishes one logical agent per topology GPU node but serves all
of them from one managed gem5 session, so it rewrites every outgoing gpu_id to
the single identity that session accepts. That identity is a constant in two
repositories at once. If they drift apart nothing fails at build time -- the
first symptom is an EINVAL from a bridge that no longer recognises the device,
somewhere deep inside hsa_init, which is exactly the failure this whole change
set removed. These tests keep the pair pinned and keep the topology generator
agreeing with the fields the model actually reads.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = ROOT / "projects/self-amdgpu-runtime/src/hsakmt_model.c"
BRIDGE_HEADER = ROOT / "projects/gem5/src/dev/amdgpu/host_gpu_kmt_state.hh"
TOPOLOGY_TOOL = ROOT / "projects/self-amdgpu-runtime/tools/hsakmt-model-topology.py"


def _define(source: str, name: str) -> int:
    match = re.search(rf"^#define\s+{re.escape(name)}\s+(\d+)U?\s*$", source, re.M)
    if match is None:
        raise AssertionError(f"{name} is not defined in {MODEL_SOURCE}")
    return int(match.group(1))


def _constexpr(source: str, name: str) -> int:
    match = re.search(
        rf"static\s+constexpr\s+\w+\s+{re.escape(name)}\s*=\s*(\d+)\s*;", source
    )
    if match is None:
        raise AssertionError(f"{name} is not defined in {BRIDGE_HEADER}")
    return int(match.group(1))


class BridgeIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        if not MODEL_SOURCE.is_file() or not BRIDGE_HEADER.is_file():
            self.skipTest("model source or gem5 bridge header is absent")
        self.model = MODEL_SOURCE.read_text(encoding="utf-8")
        self.bridge = BRIDGE_HEADER.read_text(encoding="utf-8")

    def test_bridge_gpu_id_matches_gem5(self) -> None:
        self.assertEqual(
            _define(self.model, "SAGR_HSAKMT_MODEL_BRIDGE_GPU_ID"),
            _constexpr(self.bridge, "VisibleGpuId"),
        )

    def test_bridge_render_minor_is_inside_the_accepted_window(self) -> None:
        # gem5 rejects an ACQUIRE_VM whose render minor leaves [128, 255], and
        # the model forwards one fixed minor for every logical agent.
        minor = _define(self.model, "SAGR_HSAKMT_MODEL_BRIDGE_RENDER_MINOR")
        self.assertGreaterEqual(
            minor, _define(self.model, "SAGR_HSAKMT_MODEL_RENDER_FIRST")
        )
        self.assertLessEqual(
            minor, _define(self.model, "SAGR_HSAKMT_MODEL_RENDER_LAST")
        )
        acquire = (
            ROOT / "projects/gem5/src/dev/amdgpu/host_gpu_kmt_state.cc"
        ).read_text(encoding="utf-8")
        window = re.search(
            r"request\.argumentWords\[1\]\s*<\s*(\d+)\s*\|\|\s*"
            r"request\.argumentWords\[1\]\s*>\s*(\d+)",
            acquire,
        )
        self.assertIsNotNone(window, "gem5 AcquireVm render-minor window moved")
        self.assertGreaterEqual(minor, int(window.group(1)))
        self.assertLessEqual(minor, int(window.group(2)))

    def test_visible_gpu_ceiling_matches_the_smi_slot_count(self) -> None:
        # Sixteen logical GPUs is also the width of the simulator lease
        # registry, so a topology can never ask for more agents than the host
        # can register simulators for one rank each.
        registry = (
            ROOT / "projects/self-amdgpu-runtime/src/smi_registry_internal.h"
        ).read_text(encoding="utf-8")
        match = re.search(r"SAGR_SMI_DEVICE_COUNT\s*=\s*(\d+)", registry)
        self.assertIsNotNone(match)
        self.assertEqual(
            _define(self.model, "SAGR_HSAKMT_MODEL_MAXIMUM_VISIBLE_GPUS"),
            int(match.group(1)),
        )


class TopologyContractTest(unittest.TestCase):
    """The model reads gpu_id, simd_count and drm_render_minor per node."""

    def setUp(self) -> None:
        if not TOPOLOGY_TOOL.is_file():
            self.skipTest("topology generator is absent")

    def _generate(self, directory: Path, count: int) -> Path:
        output = directory / f"gpu-{count}"
        subprocess.run(
            [
                sys.executable,
                str(TOPOLOGY_TOOL),
                "--output-dir",
                str(output),
                "--gpu-count",
                str(count),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        return output

    def _gpu_nodes(self, topology: Path) -> list[tuple[int, int, int]]:
        """Reproduce the model's node filter: gpu_id != 0 and simd_count != 0."""
        found = []
        for node in sorted(
            (topology / "nodes").iterdir(), key=lambda entry: int(entry.name)
        ):
            gpu_id = int((node / "gpu_id").read_text(encoding="ascii").strip())
            properties = dict(
                (line.split()[0], int(line.split()[1]))
                for line in (node / "properties").read_text(encoding="ascii").splitlines()
                if len(line.split()) == 2 and line.split()[1].lstrip("-").isdigit()
            )
            if gpu_id == 0 or properties.get("simd_count", 0) == 0:
                continue
            found.append((int(node.name), gpu_id, properties["drm_render_minor"]))
        return found

    def test_generated_topologies_are_readable_by_the_model_filter(self) -> None:
        model = MODEL_SOURCE.read_text(encoding="utf-8")
        base = _define(model, "SAGR_HSAKMT_MODEL_BRIDGE_GPU_ID")
        first_minor = _define(model, "SAGR_HSAKMT_MODEL_BRIDGE_RENDER_MINOR")
        maximum = _define(model, "SAGR_HSAKMT_MODEL_MAXIMUM_VISIBLE_GPUS")
        with tempfile.TemporaryDirectory() as directory:
            for count in (1, 2, maximum):
                nodes = self._gpu_nodes(self._generate(Path(directory), count))
                self.assertEqual(len(nodes), count)
                for index, (node_id, gpu_id, minor) in enumerate(nodes):
                    # Node 0 is the CPU, so GPU nodes start at 1.
                    self.assertEqual(node_id, index + 1)
                    self.assertEqual(gpu_id, base + index)
                    self.assertEqual(minor, first_minor + index)
                # Distinct ids and minors are what let the model reject a
                # render descriptor paired with the wrong agent.
                self.assertEqual(len({entry[1] for entry in nodes}), count)
                self.assertEqual(len({entry[2] for entry in nodes}), count)

    def test_a_topology_cannot_exceed_the_model_device_table(self) -> None:
        maximum = _define(
            MODEL_SOURCE.read_text(encoding="utf-8"),
            "SAGR_HSAKMT_MODEL_MAXIMUM_VISIBLE_GPUS",
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOPOLOGY_TOOL),
                    "--output-dir",
                    str(Path(directory) / "too-many"),
                    "--gpu-count",
                    str(maximum + 1),
                ],
                capture_output=True,
            )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
