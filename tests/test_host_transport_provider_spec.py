#!/usr/bin/env python3
"""Mechanical CP-0009 checks for the source-exact provider authority."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "projects" / "rocm-systems"
AUTHORITY_PATH = ROOT / "protocol" / "host-transport-v1-provider.json"


EXPECTED_STATUS_VALUES = {
    "HSAKMT_STATUS_SUCCESS": 0,
    "HSAKMT_STATUS_ERROR": 1,
    "HSAKMT_STATUS_DRIVER_MISMATCH": 2,
    "HSAKMT_STATUS_INVALID_PARAMETER": 3,
    "HSAKMT_STATUS_INVALID_HANDLE": 4,
    "HSAKMT_STATUS_INVALID_NODE_UNIT": 5,
    "HSAKMT_STATUS_NO_MEMORY": 6,
    "HSAKMT_STATUS_BUFFER_TOO_SMALL": 7,
    "HSAKMT_STATUS_NOT_IMPLEMENTED": 10,
    "HSAKMT_STATUS_NOT_SUPPORTED": 11,
    "HSAKMT_STATUS_UNAVAILABLE": 12,
    "HSAKMT_STATUS_OUT_OF_RESOURCES": 13,
    "HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED": 20,
    "HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR": 21,
    "HSAKMT_STATUS_KERNEL_ALREADY_OPENED": 22,
    "HSAKMT_STATUS_HSAMMU_UNAVAILABLE": 23,
    "HSAKMT_STATUS_WAIT_FAILURE": 30,
    "HSAKMT_STATUS_WAIT_TIMEOUT": 31,
    "HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED": 35,
    "HSAKMT_STATUS_MEMORY_NOT_REGISTERED": 36,
    "HSAKMT_STATUS_MEMORY_ALIGNMENT": 37,
}


# These values were obtained by compiling a C11 probe against the pinned
# hsakmt headers on the authority target (Linux x86-64, pack(4)).
EXPECTED_LAYOUTS = {
    "HsaVersionInfo": {
        "sizeof": 8,
        "fields": {"KernelInterfaceMajorVersion": 0, "KernelInterfaceMinorVersion": 4},
    },
    "HsaSystemProperties": {
        "sizeof": 16,
        "fields": {"NumNodes": 0, "PlatformOem": 4, "PlatformId": 8, "PlatformRev": 12},
    },
    "HsaNodeProperties": {
        "sizeof": 396,
        "fields": {
            "NumCPUCores": 0,
            "Capability": 32,
            "WaveFrontSize": 52,
            "LocalMemSize": 92,
            "MarketingName": 112,
            "AMDName": 240,
            "DebugProperties": 308,
            "UniqueID": 340,
            "WallClockKHz": 384,
            "FabricHandleSupported": 392,
        },
    },
    "HsaMemoryProperties": {
        "sizeof": 32,
        "fields": {
            "HeapType": 0,
            "SizeInBytes": 4,
            "Flags": 12,
            "Width": 16,
            "MemoryClockMax": 20,
            "VirtualBaseAddress": 24,
        },
    },
    "HsaCacheProperties": {
        "sizeof": 1056,
        "fields": {"ProcessorIdLow": 0, "CacheLevel": 4, "CacheType": 28, "SiblingMap": 32},
    },
    "HsaIoLinkProperties": {
        "sizeof": 52,
        "fields": {"IoLinkType": 0, "NodeFrom": 12, "MinimumLatency": 24, "Flags": 48},
    },
    "HsaMemFlags": {"sizeof": 4, "fields": {"Value": 0}},
    "HsaGraphicsResourceInfo": {
        "sizeof": 40,
        "fields": {
            "MemoryAddress": 0,
            "SizeInBytes": 8,
            "Metadata": 16,
            "MetadataSizeInBytes": 24,
            "NodeId": 28,
            "SizeHintInBytes": 32,
        },
    },
    "HsaQueueInfo": {
        "sizeof": 76,
        "fields": {
            "QueueDetailError": 0,
            "QueueTypeExtended": 4,
            "NumCUAssigned": 8,
            "CUMaskInfo": 12,
            "UserContextSaveArea": 20,
            "SaveAreaSizeInBytes": 28,
            "ControlStackTop": 36,
            "SaveAreaHeader": 52,
            "SaveAreaAllocSize": 68,
        },
    },
    "HsaQueueResource": {
        "sizeof": 40,
        "fields": {
            "QueueId": 0,
            "Queue_DoorBell": 8,
            "Queue_write_ptr": 16,
            "Queue_read_ptr": 24,
            "ErrorReason": 32,
        },
    },
    "HsaEventDescriptor": {
        "sizeof": 24,
        "fields": {"EventType": 0, "NodeId": 4, "SyncVar": 8},
    },
    "HsaEvent": {
        "sizeof": 48,
        "fields": {"EventId": 0, "EventData": 4},
    },
    "HsaPointerInfo": {
        "sizeof": 68,
        "fields": {
            "Type": 0,
            "Node": 4,
            "MemFlags": 8,
            "CPUAddress": 12,
            "GPUAddress": 20,
            "SizeInBytes": 28,
            "NRegisteredNodes": 36,
            "NMappedNodes": 40,
            "RegisteredNodes": 44,
            "MappedNodes": 52,
            "UserData": 60,
        },
    },
    "HsaMemoryRange": {
        "sizeof": 16,
        "fields": {"MemoryAddress": 0, "SizeInBytes": 8},
    },
    "HsaHandleImportDesc": {
        "sizeof": 40,
        "fields": {"device_handle": 0, "type": 8, "dmabuf_fd": 12, "mem": 28, "metadata": 36},
    },
    "HsaHandleImportResult": {
        "sizeof": 24,
        "fields": {"buf_handle": 0, "dmabuf_fd": 8, "alloc_size": 12, "metadata": 20},
    },
    "HsaStructureSizes": {
        "sizeof": 16,
        "fields": {
            "StructureSizes": 0,
            "SizeOfHsaNodeProperties": 2,
            "SizeOfHsaExternalHandleDesc": 4,
            "Reserved": 6,
        },
    },
}


def load_authority() -> dict[str, Any]:
    with AUTHORITY_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def source_text(authority: dict[str, Any], file_id: str) -> str:
    record = next(item for item in authority["source_authority"]["files"] if item["id"] == file_id)
    return (SOURCE_ROOT / record["path"]).read_text(encoding="utf-8")


def extract_typedef_order(header: str, macro: str) -> list[str]:
    # Each PFN typedef is one declaration terminated by a semicolon; this
    # avoids also matching class member declarations and macro definitions.
    return re.findall(rf"typedef\s+[^;]*?{macro}\(([^)]+)\)", header, re.S)


def strip_win32_blocks(source: str) -> str:
    """Model the Linux preprocessor for the simple _WIN32 blocks in these files."""
    kept: list[str] = []
    skipping = False
    for line in source.splitlines():
        if re.match(r"\s*#if\s+defined\(_WIN32\)", line):
            skipping = True
            continue
        if skipping and re.match(r"\s*#endif\b", line):
            skipping = False
            continue
        if not skipping:
            kept.append(line)
    if skipping:
        raise AssertionError("unterminated _WIN32 block")
    return "\n".join(kept)


def extract_loader_entries(loader_source: str) -> tuple[list[str], list[str]]:
    start = loader_source.index("void ThunkLoader::LoadThunkApiTable()")
    end = loader_source.index("bool ThunkLoader::CreateThunkInstance()", start)
    function = loader_source[start:end]
    load_error = function.index("LOAD_ERROR:")
    shared = function[:load_error]
    direct = function[function.index("    } else {", load_error):]

    def names(lines: list[str], needle: str) -> list[str]:
        result: list[str] = []
        for line in lines:
            if needle not in line:
                continue
            match = re.search(r"(HSAKMT|DRM)_PFN\(([^)]+)\)", line)
            if match:
                result.append(match.group(2))
        return result

    shared_names = names(shared.splitlines(), "GetExportAddress")
    direct_names = names(direct.splitlines(), "= (")
    return shared_names, direct_names


def version_script_exports(version_script: str) -> list[str]:
    return re.findall(r"^([A-Za-z_][A-Za-z0-9_]*);$", version_script, re.M)


class HostTransportProviderSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.authority = load_authority()
        cls.files = {item["id"]: item for item in cls.authority["source_authority"]["files"]}

    def test_schema_and_pinned_source_identity(self) -> None:
        authority = self.authority
        self.assertEqual(authority["schema"], "amdgpu-sim.host-transport-v1.provider.v1")
        self.assertEqual(authority["checkpoint"], "CP-0009")
        source = authority["source_authority"]
        self.assertEqual(source["repository"], "rocm-systems")
        self.assertEqual(
            subprocess.check_output(
                ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            source["commit"],
        )
        tree_paths = {
            "rocr_runtime_tree": "projects/rocr-runtime",
            "rocr_runtime_core_tree": "projects/rocr-runtime/runtime/hsa-runtime",
            "rocr_runtime_core_subtree_tree": "projects/rocr-runtime/runtime/hsa-runtime/core",
            "libhsakmt_tree": "projects/rocr-runtime/libhsakmt",
            "header_tree": "projects/rocr-runtime/libhsakmt/include/hsakmt",
        }
        for field, path in tree_paths.items():
            actual = subprocess.check_output(
                ["git", "-C", str(SOURCE_ROOT), "rev-parse", f"HEAD:{path}"],
                text=True,
            ).strip()
            self.assertEqual(actual, source[field], field)

        self.assertEqual(len(self.files), 18)
        for file_id, record in self.files.items():
            path = SOURCE_ROOT / record["path"]
            self.assertTrue(path.is_file(), file_id)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"], file_id
            )
            blob = subprocess.check_output(
                ["git", "-C", str(SOURCE_ROOT), "rev-parse", f"HEAD:{record['path']}"],
                text=True,
            ).strip()
            self.assertEqual(blob, record["blob"], file_id)

    def test_thunk_typedef_and_loader_orders_are_exact(self) -> None:
        authority = self.authority
        inventory = authority["symbol_inventory"]
        header = source_text(authority, "thunk_loader_header")
        hsa_typedefs = extract_typedef_order(header, "HSAKMT_DEF")
        drm_typedefs = extract_typedef_order(header, "DRM_DEF")
        self.assertEqual(len(hsa_typedefs), 113)
        self.assertEqual(len(drm_typedefs), 11)
        self.assertEqual(hsa_typedefs, inventory["hsa_thunk_typedef_order"])
        self.assertEqual(inventory["hsa_thunk_typedef_count"], len(hsa_typedefs))
        self.assertEqual(inventory["drm_thunk_typedef_count"], len(drm_typedefs))
        self.assertEqual(inventory["drm_resolution_count"], len(inventory["drm_resolution_order"]))
        self.assertEqual(drm_typedefs, inventory["drm_resolution_order"])

        loader = source_text(authority, "thunk_loader_implementation")
        shared, direct = extract_loader_entries(loader)
        resolution = authority["loader"]["shared_library_resolution"]
        self.assertEqual(shared, resolution["order"])
        direct_authority = authority["loader"]["direct_binding"]
        self.assertEqual(direct, direct_authority["direct_order"])
        self.assertEqual(len(shared), resolution["entry_count"])
        self.assertEqual(len(direct), direct_authority["entry_count"])
        optional = resolution["optional_symbols"]
        self.assertEqual(len(optional), resolution["optional_entry_count"])
        self.assertEqual(
            resolution["mandatory_symbols"], [name for name in shared if name not in optional]
        )
        self.assertEqual(len(resolution["mandatory_symbols"]), resolution["mandatory_entry_count"])
        self.assertEqual(len(set(shared)), len(shared))

        linux_header = strip_win32_blocks(header)
        self.assertEqual(
            len(extract_typedef_order(linux_header, "HSAKMT_DEF")),
            inventory["linux_hsa_thunk_typedef_count"],
        )
        linux_loader = strip_win32_blocks(loader)
        shared_linux, direct_linux = extract_loader_entries(linux_loader)
        self.assertEqual(len(shared_linux), 123)
        self.assertEqual(len(direct_linux), 122)
        self.assertEqual(set(shared) - set(shared_linux), {"hsaKmtGetMemoryHandle"})
        self.assertEqual(
            set(direct) - set(direct_linux),
            {"hsaKmtQueueRingDoorbell", "hsaKmtGetMemoryHandle"},
        )
        self.assertEqual(
            len([name for name in shared_linux if name not in optional]),
            resolution["effective_by_platform"]["linux_x86_64"]["mandatory_entry_count"],
        )
        self.assertEqual(
            len([name for name in direct_linux if name not in optional]),
            authority["loader"]["direct_binding"]["effective_by_platform"]["linux_x86_64"]["mandatory_entry_count"],
        )
        self.assertEqual(
            resolution["effective_by_platform"]["linux_x86_64"],
            {"entry_count": 123, "hsa_symbol_count": 112, "drm_symbol_count": 11, "mandatory_entry_count": 118, "optional_entry_count": 5},
        )
        self.assertEqual(
            authority["loader"]["direct_binding"]["effective_by_platform"]["linux_x86_64"],
            {"entry_count": 122, "hsa_symbol_count": 111, "drm_symbol_count": 11, "mandatory_entry_count": 117, "optional_entry_count": 5},
        )
        conditionals = resolution["conditional_symbols"]
        self.assertFalse(conditionals["hsaKmtGetMemoryHandle"]["linux_shared_effective"])
        self.assertFalse(conditionals["hsaKmtGetMemoryHandle"]["linux_direct_effective"])
        self.assertTrue(conditionals["hsaKmtQueueRingDoorbell"]["linux_shared_effective"])
        self.assertFalse(conditionals["hsaKmtQueueRingDoorbell"]["linux_direct_effective"])

        exports = version_script_exports(source_text(authority, "libhsakmt_version_script"))
        self.assertEqual(exports, inventory["libhsakmt_version_script_exports"])
        self.assertEqual(len(exports), inventory["libhsakmt_version_script_export_count"])
        self.assertEqual(
            sorted(set(hsa_typedefs) - set(exports)),
            sorted(inventory["version_script_difference"]["typedefs_not_exported_by_libhsakmt_ver"]),
        )
        self.assertEqual(
            sorted(set(exports) - set(hsa_typedefs)),
            sorted(inventory["version_script_difference"]["exported_not_typed"]),
        )

    def test_symbol_layers_cover_every_entry_once(self) -> None:
        resolution = self.authority["loader"]["shared_library_resolution"]["order"]
        layers = self.authority["symbol_inventory"]["layers"]
        self.assertEqual(len(layers), 7)
        flattened = [symbol for layer in layers for symbol in layer["symbols"]]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(resolution))
        self.assertEqual(len(flattened), len(resolution))

    def test_status_enum_and_observed_mapping_are_source_grounded(self) -> None:
        authority = self.authority
        types = source_text(authority, "hsakmt_types")
        actual = {
            name: int(value)
            for name, value in re.findall(
                r"^\s*(HSAKMT_STATUS_[A-Z0-9_]+)\s*=\s*(\d+)", types, re.M
            )
        }
        self.assertEqual(actual, EXPECTED_STATUS_VALUES)
        self.assertEqual(authority["statuses"]["values"], EXPECTED_STATUS_VALUES)
        self.assertEqual(len(actual), 21)
        mappings = authority["statuses"]["observed_rocr_mapping"]
        self.assertEqual(len(mappings), 5)
        self.assertIn("HSA_STATUS_ERROR", mappings[0]["non_success"])
        self.assertEqual(mappings[1]["scope"], "KfdDriver::AllocateMemory")
        self.assertIn("229-414", mappings[1]["source"])
        self.assertEqual(mappings[1]["mapped"], "HSA_STATUS_ERROR_OUT_OF_RESOURCES")
        self.assertEqual(mappings[2]["scope"], "KfdDriver::AllocateScratchMemory")
        self.assertEqual(
            mappings[3]["mapping"]["HSAKMT_STATUS_NOT_SUPPORTED"],
            "HSA_STATUS_ERROR_NOT_SUPPORTED",
        )
        self.assertEqual(
            mappings[4]["mapping"]["HSAKMT_STATUS_INVALID_HANDLE"],
            "HSA_STATUS_ERROR_INVALID_ARGUMENT",
        )

    def test_model_interface_and_drm_command_layouts_are_exact(self) -> None:
        authority = self.authority
        model_source = source_text(authority, "hsakmt_model_interface")
        model = authority["model_abi"]
        self.assertEqual(model["interface_version"], {"major": 1, "minor": 1})
        self.assertEqual(model["function_table"]["sizeof"], 32)
        self.assertEqual(
            model["function_table"]["fields"],
            {"version_major": 0, "version_minor": 4, "create_memfd": 8, "handle_ioctl": 16, "handle_drm_call": 24},
        )
        enum_body = model_source.split("enum hsakmt_drm_cmd {", 1)[1].split("};", 1)[0]
        enum_names = re.findall(r"^\s*(HSAKMT_DRM_[A-Z0-9_]+)\s*,?", enum_body, re.M)
        commands = model["drm_commands"]
        self.assertEqual(len(commands), model["drm_command_count"])
        self.assertEqual([item["name"] for item in commands], enum_names)
        self.assertEqual([item["value"] for item in commands], list(range(15)))
        for item in commands:
            self.assertRegex(model_source, rf"struct\s+{re.escape(item['arg_type'].removeprefix('struct '))}\s*\{{")

    def test_upstream_hardware_calls_are_explicitly_excluded(self) -> None:
        authority = self.authority
        audits = authority["upstream_hardware_call_audit"]
        self.assertEqual(
            {item["file_id"] for item in audits},
            {"libhsakmt_memory_hardware_path", "libhsakmt_fmm_hardware_path", "libhsakmt_debug_hardware_path"},
        )
        for item in audits:
            source = source_text(authority, item["file_id"])
            for call in item["direct_or_wrapper_calls"]:
                self.assertIn(call, source, f"{item['file_id']}: {call}")
            self.assertTrue(
                "must not" in item["gem_sim_rule"]
                or "Do not" in item["gem_sim_rule"]
                or "never" in item["gem_sim_rule"]
            )
        gate = authority["provider_gate"]["required_before_provider_claim"]
        self.assertTrue(any("direct-call audit" in rule or "dependency audits" in rule for rule in gate))

    def test_model_dispatch_and_build_boundary_are_source_grounded(self) -> None:
        authority = self.authority
        model_source = source_text(authority, "hsakmt_model_implementation")
        for token in ("getenv", "dlopen", "dlsym", "handle_ioctl", "handle_drm_call", "model_reported_minor"):
            self.assertIn(token, model_source)
        self.assertIn("model_init()", source_text(authority, "libhsakmt_topology_attach"))
        self.assertIn("hsakmt_fmm_init_process_apertures", source_text(authority, "libhsakmt_topology_attach"))
        self.assertIn("AMDKFD_IOCTL_BASE", source_text(authority, "kfd_ioctl_header"))

        boundary = authority["build_boundary"]
        self.assertEqual(
            set(boundary["source_file_ids"]),
            {"libhsakmt_cmake", "rocr_runtime_cmake", "hsa_runtime_cmake"},
        )
        cmake_hsakmt = source_text(authority, "libhsakmt_cmake")
        for source_name in ("src/debug.c", "src/fmm.c", "src/hsakmtmodel.c", "src/memory.c", "src/topology.c"):
            self.assertIn(source_name, cmake_hsakmt)
        for library in ("libdrm", "libdrm_amdgpu", "CMAKE_DL_LIBS"):
            self.assertIn(library, cmake_hsakmt)
        cmake_root = source_text(authority, "rocr_runtime_cmake")
        self.assertIn("add_rocm_subdir(libhsakmt", cmake_root)
        self.assertIn("add_rocm_subdir(runtime/hsa-runtime", cmake_root)
        cmake_runtime = source_text(authority, "hsa_runtime_cmake")
        self.assertIn("core/runtime/thunk_loader.cpp", cmake_runtime)
        self.assertIn("pkg_check_modules(drm", cmake_runtime)
        self.assertIn("no DT_NEEDED", boundary["provider_link_rule"])

    def test_all_seventeen_recorded_layouts_match_probe_values(self) -> None:
        layouts = self.authority["abi_layouts"]
        self.assertEqual(len(layouts), 17)
        layout_scope = self.authority["abi_layouts_scope"]
        self.assertEqual(layout_scope["field_scope"], "key_offsets_partial")
        self.assertIn("not an exhaustive", layout_scope["meaning"])
        actual = {
            item["type"]: {"sizeof": item["sizeof"], "fields": item["fields"]}
            for item in layouts
        }
        self.assertEqual(actual, EXPECTED_LAYOUTS)
        self.assertEqual(set(actual), set(EXPECTED_LAYOUTS))
        target = self.authority["target_abi"]
        self.assertEqual(target["platform"], "linux-x86_64")
        self.assertEqual(target["pointer_bytes"], 8)
        self.assertEqual(target["enum_bytes"], 4)
        self.assertEqual(target["hsakmttypes_pack"], 4)
        self.assertEqual(target["integer_byte_order"], "little-endian")

    def test_loader_resolution_and_hardware_exclusion_contract(self) -> None:
        authority = self.authority
        loader = source_text(authority, "thunk_loader_implementation")
        self.assertIn('return "libdtif.so";', loader)
        self.assertIn('return "dtif64a.dll";', loader)
        self.assertIn('open("/dev/dxg", O_RDWR)', loader)
        self.assertIn('return "librocdxg.so";', loader)
        self.assertIn("return \"\";", loader)
        self.assertIn("GetAdjacentLibraryPath", loader)
        self.assertIn("goto LOAD_ERROR", loader)
        self.assertIn('"DtifCreate"', loader)
        self.assertIn('"DtifDestroy"', loader)
        self.assertIn('"DxgAbiCheck"', loader)

        resolution = authority["loader"]["shared_library_resolution"]
        self.assertEqual(
            set(resolution["order"]),
            set(authority["symbol_inventory"]["hsa_thunk_typedef_order"])
            | set(authority["symbol_inventory"]["drm_resolution_order"]),
        )
        self.assertEqual(
            resolution["optional_symbols"],
            [
                "hsaKmtSetSigbusDelay",
                "hsaKmtImportExternalSemaphore",
                "hsaKmtDestroyExternalSemaphore",
                "hsaKmtQueueSignalExternalSemaphore",
                "hsaKmtQueueWaitExternalSemaphore",
            ],
        )

        unsupported = authority["unsupported_semantics"]
        self.assertEqual(
            unsupported["forbidden_device_nodes"],
            ["/dev/kfd", "/dev/dri", "/dev/dxg", "/dev/udmabuf"],
        )
        self.assertTrue(unsupported["hardware_bypass"].startswith("The provider is a daemon-backed"))
        self.assertIn("HSAKMT_STATUS_NOT_SUPPORTED", unsupported["export_contract"])
        self.assertIn("metadata only", unsupported["export_contract"])
        self.assertIn("does not export the 124 typed", unsupported["export_contract"])
        skeleton = unsupported["current_cp9_metadata_skeleton"]
        self.assertEqual(skeleton["kind"], "metadata-only")
        self.assertEqual(skeleton["typed_hsakmt_drm_exports"], 0)
        self.assertEqual(skeleton["source_union_typed_entries"], 124)
        self.assertFalse(skeleton["implemented_full_provider"])
        future = unsupported["future_full_provider_contract"]
        self.assertEqual(future["required_typed_exports"], 124)
        self.assertEqual(future["gate"], "provider_gate")
        self.assertIn("metadata queries alone", future["not_satisfied_by"])
        query = authority["query_lifecycle"]
        self.assertEqual(query["scope"], "transport-open")
        self.assertIn("hsaKmtOpenKFD", query["does_not_call"])
        self.assertIn("KFD channel open", query["does_not_mean"])
        self.assertIn("KFD topology acquired", query["does_not_mean"])
        self.assertEqual(
            query["hardware_effect"],
            "none; the query performs no device-node, KFD, DRM, or topology operation",
        )
        self.assertIn("future full-provider KFD-channel policy", query["status_boundary"])
        self.assertIn("not a result of this transport-open query", query["status_boundary"])
        self.assertIn("Future full-provider KFD-channel policy only", authority["statuses"]["provider_policy"]["provider_not_open"])
        self.assertIn("transport-open/lifecycle query does not use", authority["statuses"]["provider_policy"]["provider_not_open"])
        self.assertIn("HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR", authority["statuses"]["provider_policy"]["daemon_transport_failure"])


if __name__ == "__main__":
    unittest.main()
