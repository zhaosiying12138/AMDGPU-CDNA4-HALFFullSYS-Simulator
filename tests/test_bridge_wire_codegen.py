"""Generation and compile gates for the canonical runtime-gem5 wire contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools/generate_bridge_wire.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_bridge_wire", GENERATOR)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load bridge generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BridgeWireCodegenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = load_generator()
        cls.base = json.loads(cls.generator.BASE_SCHEMA.read_text(encoding="ascii"))
        cls.generic = json.loads(
            cls.generator.GENERIC_SCHEMA.read_text(encoding="ascii")
        )
        cls.kmt = json.loads(cls.generator.KMT_SCHEMA.read_text(encoding="ascii"))

    def test_committed_outputs_are_fresh_and_identical_in_authority(self) -> None:
        subprocess.run(
            ["/usr/bin/python3", str(GENERATOR), "--check"],
            cwd=ROOT,
            check=True,
        )
        outputs = self.generator.generated()
        digests = {}
        for path, expected in outputs.items():
            if isinstance(expected, bytes):
                self.assertEqual(path.read_bytes(), expected)
                continue
            self.assertEqual(path.read_text(encoding="ascii"), expected)
            if path.name == "manifest.json":
                continue
            match = re.search(r"Machine contract SHA-256: ([0-9a-f]{64})", expected)
            self.assertIsNotNone(match)
            digests[path] = match.group(1)
            self.assertNotRegex(expected.lower(), r"qwen|vecadd|silu|fixture|kernel_hash")
        self.assertEqual(
            digests[self.generator.C_OUTPUT],
            digests[self.generator.CPP_OUTPUT],
        )
        self.assertEqual(
            digests[self.generator.KMT_C_OUTPUT],
            digests[self.generator.KMT_CPP_OUTPUT],
        )
        self.assertEqual(len(set(digests.values())), 2)

    def test_contract_rejects_inconsistent_record_and_payload_sizes(self) -> None:
        broken = json.loads(json.dumps(self.generic))
        broken["transport"]["record_bytes"] += 1

        def fake_load(path: Path):
            return self.base if path == self.generator.BASE_SCHEMA else broken

        with mock.patch.object(self.generator, "_load", side_effect=fake_load):
            with self.assertRaisesRegex(ValueError, "record size"):
                self.generator.contract()

    def test_schema_offsets_and_bounds_are_exported(self) -> None:
        values = self.generator.contract()["values"]
        self.assertEqual(values["HEADER_BYTES"], self.base["header"]["bytes"])
        self.assertEqual(
            values["REQUEST_SUBMIT_NUM_CTAS_OFFSET"],
            next(
                field["offset"]
                for field in self.generic["request_payload"]["stage_bodies"]
                ["SUBMIT_AQL"]["fields"]
                if field["name"] == "num_ctas"
            ),
        )
        self.assertEqual(
            values["RESPONSE_RETIRE_TICK_OFFSET"],
            next(
                field["offset"]
                for field in self.generic["response_payload"]["fields"]
                if field["name"] == "retire_tick"
            ),
        )
        self.assertEqual(values["MAX_CTAS"], 8)

        kmt_contract = self.generator.kmt_contract()
        kmt_values = kmt_contract["values"]
        self.assertEqual(kmt_values["KMT_PAYLOAD_MAJOR"], 1)
        self.assertEqual(kmt_values["KMT_PAYLOAD_MINOR"], 5)
        self.assertEqual(kmt_values["KMT_HEADER_BYTES"], 80)
        self.assertEqual(kmt_values["KMT_REQUEST_FRAME_BYTES"], 336)
        self.assertEqual(kmt_values["KMT_RESULT_FRAME_BYTES"], 336)
        self.assertEqual(kmt_values["KMT_OPERATION_SET_SCRATCH_BACKING_VA"], 26)
        self.assertEqual(kmt_values["KMT_OPERATION_EXPORT_BACKING"], 27)
        self.assertEqual(kmt_values["KMT_OPERATION_GET_CLOCK_COUNTERS"], 28)
        self.assertEqual(kmt_values["KMT_CLOCK_FREQUENCY_HZ"], 1_000_000_000)
        self.assertEqual(kmt_contract["request_word_masks"]["GET_CLOCK_COUNTERS"], 1)
        self.assertEqual(kmt_contract["result_word_masks"]["GET_CLOCK_COUNTERS"], 255)
        self.assertEqual(kmt_values["KMT_SHARED_BACKING_LAYOUT_MAJOR"], 1)
        self.assertEqual(
            kmt_values["KMT_SHARED_BACKING_DOORBELL_REGION_BYTES"], 8192
        )
        self.assertEqual(kmt_values["KMT_SHARED_BACKING_DOORBELL_SLOT_BYTES"], 8)
        self.assertEqual(
            kmt_values["KMT_SHARED_BACKING_MAXIMUM_DOORBELL_SLOTS"], 128
        )
        self.assertEqual(
            kmt_values["KMT_SHARED_BACKING_DOORBELL_INITIAL_VALUE"],
            (1 << 64) - 1,
        )
        self.assertEqual(
            kmt_values["KMT_SHARED_BACKING_COMPLETION_REGION_BASE_BYTES"], 1024
        )
        self.assertEqual(
            kmt_values["KMT_SHARED_BACKING_COMPLETION_SLOT_BYTES"], 8
        )

    def test_generated_headers_compile_in_c_and_cpp(self) -> None:
        c_source = """
#include <self_amdgpu_runtime/runtime.h>
#include <self_amdgpu_runtime/kmt_shim.h>
_Static_assert(SAGR_GENERIC_DISPATCH_PROTOCOL_MAJOR == 2, "major");
_Static_assert(SAGR_CAPABILITY_GENERIC_EXECUTION_MASK == (UINT64_C(1) << 9), "cap");
_Static_assert(SAGR_KMT_PROTOCOL_MAJOR == 1, "kmt major");
_Static_assert(SAGR_KMT_PROTOCOL_MINOR == 5, "kmt minor");
_Static_assert(SAGR_KMT_OP_SET_SCRATCH_BACKING_VA == 26, "kmt operation");
_Static_assert(SAGR_KMT_OP_EXPORT_BACKING == 27, "kmt export operation");
_Static_assert(SAGR_KMT_OP_GET_CLOCK_COUNTERS == 28, "kmt clock operation");
_Static_assert(SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_REGION_BYTES == 8192, "doorbell region");
_Static_assert(SAGR_BRIDGE_KMT_SHARED_BACKING_COMPLETION_REGION_BASE_BYTES == 1024, "completion base");
int main(void) { return SAGR_GENERIC_MAX_CTAS == 8 ? 0 : 1; }
"""
        cpp_source = """
#include "dev/amdgpu/host_gpu_protocol.hh"
static_assert(gem5::hostgpu::protocol::GenericDispatchRecordBytes == 4096);
static_assert(static_cast<unsigned>(gem5::hostgpu::protocol::GenericDispatchOpcode::SubmitAql) == 3);
static_assert(gem5::hostgpu::protocol::KmtOperationMajor == 1);
static_assert(gem5::hostgpu::protocol::KmtOperationMinor == 5);
static_assert(static_cast<unsigned>(gem5::hostgpu::protocol::KmtOperation::SetScratchBackingVa) == 26);
static_assert(static_cast<unsigned>(gem5::hostgpu::protocol::KmtOperation::ExportBacking) == 27);
static_assert(static_cast<unsigned>(gem5::hostgpu::protocol::KmtOperation::GetClockCounters) == 28);
static_assert(gem5::hostgpu::bridgekmt::KmtSharedBackingDoorbellRegionBytes == 8192);
static_assert(gem5::hostgpu::bridgekmt::KmtSharedBackingCompletionRegionBaseBytes == 1024);
static_assert(gem5::hostgpu::bridgekmt::kmtOperationValid(
    gem5::hostgpu::bridgekmt::KmtOperationQueueCreate));
static_assert(gem5::hostgpu::bridgekmt::kmtResultWordMask(
    gem5::hostgpu::bridgekmt::KmtOperationQueueCreate) == 0x07);
int main() { return 0; }
"""
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            c_path = directory / "contract.c"
            cpp_path = directory / "contract.cc"
            c_path.write_text(c_source, encoding="ascii")
            cpp_path.write_text(cpp_source, encoding="ascii")
            subprocess.run(
                [
                    "/usr/bin/cc",
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-fsyntax-only",
                    "-I",
                    str(ROOT / "projects/self-amdgpu-runtime/include"),
                    str(c_path),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "/usr/bin/c++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-fsyntax-only",
                    "-I",
                    str(ROOT / "projects/gem5/src"),
                    str(cpp_path),
                ],
                check=True,
            )

    def test_consumers_do_not_redeclare_generated_generic_values(self) -> None:
        runtime = (
            ROOT / "projects/self-amdgpu-runtime/src/transport_internal.h"
        ).read_text(encoding="ascii")
        gem5 = (
            ROOT / "projects/gem5/src/dev/amdgpu/host_gpu_protocol.hh"
        ).read_text(encoding="ascii")
        public_runtime = (
            ROOT
            / "projects/self-amdgpu-runtime/include/self_amdgpu_runtime/runtime.h"
        ).read_text(encoding="ascii")
        self.assertNotIn("SAGR_WIRE_GENERIC_PAYLOAD_BYTES = 4016", runtime)
        self.assertNotIn("GenericDispatchPayloadBytes = 4016", gem5)
        self.assertIn("generated/bridge_generic_v2.h", public_runtime)
        self.assertIn("generated/bridge_generic_v2.hh", gem5)
        public_kmt = (
            ROOT
            / "projects/self-amdgpu-runtime/include/self_amdgpu_runtime/kmt_shim.h"
        ).read_text(encoding="ascii")
        self.assertIn("generated/bridge_kmt_v5.h", public_kmt)
        self.assertIn("generated/bridge_kmt_v5.hh", gem5)
        self.assertNotIn("#define SAGR_KMT_PROTOCOL_MINOR UINT16_C(5)", public_kmt)
        self.assertNotIn("constexpr uint16_t KmtOperationMinor = 5", gem5)


if __name__ == "__main__":
    unittest.main()
