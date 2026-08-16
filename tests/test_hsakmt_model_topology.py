# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "projects/self-amdgpu-runtime/tools/hsakmt-model-topology.py"
)
SPEC = importlib.util.spec_from_file_location("hsakmt_model_topology", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
TOPOLOGY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOPOLOGY)


class HsakmtModelTopologyTest(unittest.TestCase):
    def test_materialize_and_verify_one_and_sixteen_gpus(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            for count in (1, 16):
                output = base / f"topology-{count}"
                generated = TOPOLOGY.materialize(output, count)
                verified = TOPOLOGY.verify(output, count)
                self.assertEqual(generated, verified)
                self.assertEqual(
                    len([entry for entry in (output / "nodes").iterdir()]),
                    count + 1,
                )
                self.assertEqual(
                    (output / "nodes" / str(count) / "properties").is_file(),
                    True,
                )

    def test_existing_topology_is_absent_or_exact(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "topology"
            expected = TOPOLOGY.materialize(output, 4)
            self.assertEqual(TOPOLOGY.materialize(output, 4), expected)
            with self.assertRaisesRegex(
                TOPOLOGY.TopologyError, "gpu_count differs"
            ):
                TOPOLOGY.materialize(output, 2)

    def test_tampered_and_extra_entries_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            tampered = base / "tampered"
            TOPOLOGY.materialize(tampered, 1)
            with (tampered / "nodes/1/name").open("ab") as stream:
                stream.write(b"x")
            with self.assertRaisesRegex(TOPOLOGY.TopologyError, "drifted"):
                TOPOLOGY.verify(tampered)

            extra = base / "extra"
            TOPOLOGY.materialize(extra, 1)
            (extra / "unexpected").write_text("x", encoding="ascii")
            with self.assertRaisesRegex(TOPOLOGY.TopologyError, "file set differs"):
                TOPOLOGY.verify(extra)

    def test_symlink_is_never_followed(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            output = base / "topology"
            TOPOLOGY.materialize(output, 1)
            target = output / "nodes/1/name"
            payload = target.read_bytes()
            target.unlink()
            target.symlink_to(base / "foreign")
            (base / "foreign").write_bytes(payload)
            with self.assertRaisesRegex(TOPOLOGY.TopologyError, "regular file"):
                TOPOLOGY.verify(output)

    def test_invalid_gpu_capacity_is_rejected_before_writes(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            for count in (0, 17):
                output = base / f"invalid-{count}"
                with self.assertRaisesRegex(TOPOLOGY.TopologyError, r"\[1, 16\]"):
                    TOPOLOGY.materialize(output, count)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
