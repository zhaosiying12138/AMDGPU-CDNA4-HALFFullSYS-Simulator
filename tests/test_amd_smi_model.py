from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "projects/self-amdgpu-runtime/src/amd_smi_model.c"
UNSUPPORTED = ROOT / "projects/self-amdgpu-runtime/src/amd_smi_unsupported.c"

AMDSMI_STATUS_SUCCESS = 0
AMDSMI_STATUS_INVAL = 1
AMDSMI_STATUS_NOT_SUPPORTED = 2
AMDSMI_STATUS_INIT_ERROR = 18


def _built_library() -> Path | None:
    for candidate in sorted(
        (ROOT / "projects/self-amdgpu-runtime").glob("build*/**/libamd_smi.so.26*")
    ):
        if candidate.is_file():
            return candidate
    for candidate in sorted(
        (ROOT / "projects/self-amdgpu-runtime").glob("build*/libamd_smi.so.26*")
    ):
        if candidate.is_file():
            return candidate
    return None


def _strip_c_comments(text: str) -> str:
    """Assertions below are about code, not prose.

    The file's own header comment legitimately names /dev/kfd to explain why it
    is never opened, so a raw substring check would fail on the documentation
    that proves the point.
    """
    output = []
    index = 0
    length = len(text)
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


def _write_topology(root: Path, nodes: dict[int, int]) -> Path:
    """nodes maps node id -> simd_count."""
    topology = root / "hsakmt-topology"
    for node_id, simd_count in nodes.items():
        node = topology / "nodes" / str(node_id)
        node.mkdir(parents=True, exist_ok=True)
        (node / "properties").write_text(
            f"cpu_cores_count {0 if simd_count else 1}\n"
            f"simd_count {simd_count}\n"
            "gfx_target_version 90500\n"
            "device_id 30112\n",
            encoding="ascii",
        )
    return topology


class AmdSmiModelSourceTest(unittest.TestCase):
    def test_provider_never_touches_hardware_nodes(self) -> None:
        # The whole point of this provider is that ROCm auto-selection must not
        # require a KMD. Probing a hardware node would defeat it and would also
        # violate the project's no-/dev/kfd rule.
        code = _strip_c_comments(SOURCE.read_text(encoding="ascii"))
        for forbidden in ("/dev/kfd", "/dev/dri", "/sys/class/kfd", "amdsmi_lib"):
            self.assertNotIn(forbidden, code)

    def test_provider_fails_closed_without_topology(self) -> None:
        text = SOURCE.read_text(encoding="ascii")
        self.assertIn("HSA_MODEL_TOPOLOGY", text)
        self.assertIn("AMDSMI_STATUS_INIT_ERROR", text)

    def test_gpu_discriminator_is_simd_count(self) -> None:
        # A CPU node reports simd_count 0; only SIMD-bearing nodes are GPUs.
        text = SOURCE.read_text(encoding="ascii")
        self.assertIn("simd_count", text)

    def test_unsupported_surface_is_explicit(self) -> None:
        code = _strip_c_comments(UNSUPPORTED.read_text(encoding="ascii"))
        self.assertIn("AMDSMI_STATUS_NOT_SUPPORTED", code)
        # Every stub must refuse rather than fabricate a reading.
        self.assertNotIn("return AMDSMI_STATUS_SUCCESS", code)

    def test_discovery_entry_points_are_not_stubbed(self) -> None:
        text = UNSUPPORTED.read_text(encoding="ascii")
        for implemented in (
            "amdsmi_init(void)",
            "amdsmi_shut_down(void)",
            "amdsmi_get_socket_handles(void)",
            "amdsmi_get_processor_handles(void)",
        ):
            self.assertNotIn(implemented, text)


class AmdSmiModelRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        library_path = _built_library()
        if library_path is None:
            self.skipTest("libamd_smi.so.26 has not been built")
        self.library = ctypes.CDLL(str(library_path))
        self.library.amdsmi_init.argtypes = [ctypes.c_uint64]
        self.library.amdsmi_init.restype = ctypes.c_uint32
        self.library.amdsmi_shut_down.argtypes = []
        self.library.amdsmi_shut_down.restype = ctypes.c_uint32
        self.library.amdsmi_get_socket_handles.argtypes = [
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p
        ]
        self.library.amdsmi_get_socket_handles.restype = ctypes.c_uint32
        self.library.amdsmi_get_processor_handles.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p
        ]
        self.library.amdsmi_get_processor_handles.restype = ctypes.c_uint32
        self._saved = os.environ.get("HSA_MODEL_TOPOLOGY")

    def tearDown(self) -> None:
        self.library.amdsmi_shut_down()
        if self._saved is None:
            os.environ.pop("HSA_MODEL_TOPOLOGY", None)
        else:
            os.environ["HSA_MODEL_TOPOLOGY"] = self._saved

    def test_init_fails_closed_without_topology(self) -> None:
        os.environ.pop("HSA_MODEL_TOPOLOGY", None)
        self.assertEqual(self.library.amdsmi_init(2), AMDSMI_STATUS_INIT_ERROR)

    def test_init_fails_closed_on_relative_topology(self) -> None:
        os.environ["HSA_MODEL_TOPOLOGY"] = "relative/path"
        self.assertEqual(self.library.amdsmi_init(2), AMDSMI_STATUS_INIT_ERROR)

    def test_enumeration_before_init_is_rejected(self) -> None:
        self.library.amdsmi_shut_down()
        count = ctypes.c_uint32(0)
        self.assertEqual(
            self.library.amdsmi_get_socket_handles(ctypes.byref(count), None),
            AMDSMI_STATUS_INIT_ERROR,
        )

    def test_cpu_only_topology_reports_no_device(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            topology = _write_topology(Path(directory), {0: 0})
            os.environ["HSA_MODEL_TOPOLOGY"] = str(topology)
            # No SIMD-bearing node: discovery must not invent a GPU.
            self.assertNotEqual(self.library.amdsmi_init(2), AMDSMI_STATUS_SUCCESS)

    def test_single_gpu_topology_enumerates_one_processor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            topology = _write_topology(Path(directory), {0: 0, 1: 1024})
            os.environ["HSA_MODEL_TOPOLOGY"] = str(topology)
            self.assertEqual(self.library.amdsmi_init(2), AMDSMI_STATUS_SUCCESS)

            socket_count = ctypes.c_uint32(0)
            self.assertEqual(
                self.library.amdsmi_get_socket_handles(ctypes.byref(socket_count), None),
                AMDSMI_STATUS_SUCCESS,
            )
            self.assertEqual(socket_count.value, 1)

            sockets = (ctypes.c_void_p * socket_count.value)()
            self.assertEqual(
                self.library.amdsmi_get_socket_handles(
                    ctypes.byref(socket_count), ctypes.cast(sockets, ctypes.c_void_p)
                ),
                AMDSMI_STATUS_SUCCESS,
            )

            processor_count = ctypes.c_uint32(0)
            self.assertEqual(
                self.library.amdsmi_get_processor_handles(
                    ctypes.c_void_p(sockets[0]), ctypes.byref(processor_count), None
                ),
                AMDSMI_STATUS_SUCCESS,
            )
            self.assertEqual(processor_count.value, 1)

    def test_multi_gpu_topology_enumerates_each_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            topology = _write_topology(Path(directory), {0: 0, 1: 1024, 2: 1024, 3: 1024})
            os.environ["HSA_MODEL_TOPOLOGY"] = str(topology)
            self.assertEqual(self.library.amdsmi_init(2), AMDSMI_STATUS_SUCCESS)
            socket_count = ctypes.c_uint32(0)
            self.library.amdsmi_get_socket_handles(ctypes.byref(socket_count), None)
            self.assertEqual(socket_count.value, 3)

    def test_unknown_socket_handle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            topology = _write_topology(Path(directory), {0: 0, 1: 1024})
            os.environ["HSA_MODEL_TOPOLOGY"] = str(topology)
            self.assertEqual(self.library.amdsmi_init(2), AMDSMI_STATUS_SUCCESS)
            bogus = ctypes.c_void_p(0xDEAD0000)
            count = ctypes.c_uint32(0)
            self.assertEqual(
                self.library.amdsmi_get_processor_handles(bogus, ctypes.byref(count), None),
                AMDSMI_STATUS_INVAL,
            )

    def test_null_count_pointer_is_rejected(self) -> None:
        self.assertEqual(
            self.library.amdsmi_get_socket_handles(None, None), AMDSMI_STATUS_INVAL
        )

    def test_unsupported_entry_point_refuses(self) -> None:
        probe = self.library.amdsmi_get_gpu_fan_speed
        probe.restype = ctypes.c_uint32
        self.assertEqual(probe(), AMDSMI_STATUS_NOT_SUPPORTED)


if __name__ == "__main__":
    unittest.main()
