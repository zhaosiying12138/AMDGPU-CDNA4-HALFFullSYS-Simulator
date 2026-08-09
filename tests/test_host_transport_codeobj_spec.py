#!/usr/bin/env python3
"""Independent source and fixture checks for P3-CODEOBJ-01."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import unittest
from pathlib import Path
from typing import Any

try:
    import msgpack
except ImportError:  # pragma: no cover - the authority test remains useful without it.
    msgpack = None


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocol" / "host-transport-v1-codeobj.json"
REPOSITORIES = {
    "gem5": ROOT / "projects" / "gem5",
    "rocm-systems": ROOT / "projects" / "rocm-systems",
    "llvm-project": ROOT / "projects" / "llvm-project",
    "triton": ROOT / "projects" / "triton",
}


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def git_blob(repo: Path, commit: str, path: str) -> tuple[str, bytes]:
    blob = git_output(repo, "rev-parse", f"{commit}:{path}")
    data = subprocess.check_output(["git", "-C", str(repo), "cat-file", "blob", blob])
    return blob, data


def section_table(data: bytes) -> tuple[list[dict[str, Any]], list[tuple[Any, ...]]]:
    ident, _, _, _, _, _, shoff, _, _, _, _, shentsize, shnum, shstrndx = struct.unpack_from(
        "<16sHHIQQQIHHHHHH", data, 0
    )
    if ident[:4] != b"\x7fELF":
        raise AssertionError("not ELF")
    raw = [
        struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
        for i in range(shnum)
    ]
    names = data[raw[shstrndx][4] : raw[shstrndx][4] + raw[shstrndx][5]]

    def name(offset: int) -> str:
        end = names.find(b"\0", offset)
        return names[offset:end].decode("ascii")

    return [
        {
            "name": name(item[0]),
            "type": item[1],
            "flags": item[2],
            "addr": item[3],
            "offset": item[4],
            "size": item[5],
            "link": item[6],
            "info": item[7],
            "align": item[8],
            "entsize": item[9],
        }
        for item in raw
    ], raw


def program_headers(data: bytes) -> list[dict[str, Any]]:
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data, 0)
    phoff, phentsize, phnum = header[5], header[9], header[10]
    result = []
    for index in range(phnum):
        p_type, p_flags, p_offset, p_vaddr, _, p_filesz, p_memsz, p_align = struct.unpack_from(
            "<IIQQQQQQ", data, phoff + index * phentsize
        )
        result.append(
            {
                "type": p_type,
                "flags": p_flags,
                "offset": p_offset,
                "vaddr": p_vaddr,
                "filesz": p_filesz,
                "memsz": p_memsz,
                "align": p_align,
            }
        )
    return result


def dynamic_symbols(data: bytes, sections: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    dynsym = next(section for section in sections if section["name"] == ".dynsym")
    strtab = sections[dynsym["link"]]
    strings = data[strtab["offset"] : strtab["offset"] + strtab["size"]]
    result = {}
    for offset in range(
        dynsym["offset"], dynsym["offset"] + dynsym["size"], dynsym["entsize"]
    ):
        st_name, info, other, shndx, value, size = struct.unpack_from(
            "<IBBHQQ", data, offset
        )
        end = strings.find(b"\0", st_name)
        name = strings[st_name:end].decode("ascii")
        if not name:
            continue
        result[name] = {
            "type": info & 0x0F,
            "bind": info >> 4,
            "visibility": other & 0x03,
            "shndx": shndx,
            "value": value,
            "size": size,
        }
    return result


class CodeObjectSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        cls.base = json.loads(
            (ROOT / "protocol" / "host-transport-v1.json").read_text(
                encoding="utf-8"
            )
        )
        cls.repo_by_id = REPOSITORIES

    def test_schema_scope_and_wire_surface_are_frozen(self) -> None:
        self.assertEqual(
            self.spec["schema"], "amdgpu-sim.host-transport-v1.codeobj.v1"
        )
        self.assertEqual(self.spec["checkpoint"], "P3-CODEOBJ-01")
        self.assertEqual(
            self.spec["base_protocol"], "protocol/host-transport-v1.json"
        )
        self.assertEqual(
            self.spec["capability"]["name"], "CODE_OBJECT_ABI_V1"
        )
        self.assertEqual(
            (self.spec["capability"]["bit"], self.spec["capability"]["mask_hex"]),
            (6, "40"),
        )
        transport = self.spec["transport"]
        self.assertEqual(transport["header_bytes"], self.base["header"]["bytes"])
        self.assertEqual(transport["byte_order"], "big-endian")
        self.assertEqual(transport["message_types"], {})
        self.assertEqual(transport["ancillary_descriptors"], 0)
        self.assertIn("no HSACO bytes", transport["wire_surface"])
        self.assertIn("no message type", self.spec["capability"]["negotiation"])

    def test_lock_and_project_lane_hashes_are_exact(self) -> None:
        for key in ("source_lock", "project_lanes"):
            entry = self.spec[key]
            path = ROOT / entry["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"]
            )
        self.assertEqual(self.spec["source_lock"]["lock_id"], "SL-0001")
        self.assertEqual(self.spec["project_lanes"]["registry_revision"], 1)

    def test_repository_commits_trees_and_source_blobs_are_pinned(self) -> None:
        expected_repositories = {
            item["id"]: item for item in self.spec["source_authority"]["repositories"]
        }
        self.assertEqual(set(expected_repositories), set(REPOSITORIES))
        for repository_id, expected in expected_repositories.items():
            repo = self.repo_by_id[repository_id]
            self.assertEqual(
                git_output(repo, "rev-parse", f"{expected['commit']}^{{tree}}"),
                expected["tree"],
            )
        for source in self.spec["source_authority"]["files"]:
            repo = self.repo_by_id[source["repository"]]
            repository_commit = next(
                item["commit"]
                for item in self.spec["source_authority"]["repositories"]
                if item["id"] == source["repository"]
            )
            blob, data = git_blob(repo, source.get("commit", repository_commit), source["path"])
            self.assertEqual(blob, source["blob"], source["path"])
            self.assertEqual(
                hashlib.sha256(data).hexdigest(), source["sha256"], source["path"]
            )

    def test_tree_anchors_are_exact(self) -> None:
        anchors = self.spec["source_authority"]["tree_anchors"]
        paths = {
            "rocr_runtime": ("rocm-systems", "projects/rocr-runtime"),
            "rocr_gfx950_hsaco_directory": (
                "rocm-systems",
                "projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/gfx950",
            ),
            "llvm_amdgpu": ("llvm-project", "llvm"),
            "llvm_device_libs": ("llvm-project", "amd/device-libs"),
            "triton_amd_backend": ("triton", "third_party/amd"),
            "gem5_amdgpu": ("gem5", "src/arch/amdgpu"),
        }
        for name, (repository, path) in paths.items():
            repo = self.repo_by_id[repository]
            self.assertEqual(
                git_output(
                    repo,
                    "rev-parse",
                    f"{next(item['commit'] for item in self.spec['source_authority']['repositories'] if item['id'] == repository)}:{path}",
                ),
                anchors[name],
                name,
            )

    def test_device_libs_templates_are_source_only_and_pinned(self) -> None:
        repo = REPOSITORIES["llvm-project"]
        commit = next(
            item["commit"]
            for item in self.spec["source_authority"]["repositories"]
            if item["id"] == "llvm-project"
        )
        readme = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{commit}:amd/device-libs/README.md"]
        )
        abi = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{commit}:amd/device-libs/oclc/src/abi_version.cl.in"]
        )
        isa = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{commit}:amd/device-libs/oclc/src/isa_version.cl.in"]
        )
        self.assertIn(b"--rocm-path", readme)
        self.assertIn(b"@ABI_VERSION@", abi)
        self.assertIn(b"@ISA_VERSION@", isa)
        self.assertIn("device-libs", self.spec["blockers"][-1]["statement"])

    def test_relocation_formula_table_is_complete(self) -> None:
        reloc = self.spec["abi_contract"]["relocations"]
        self.assertEqual(set(reloc["ids"].values()), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14})
        self.assertEqual(set(reloc["static_formulas"]), {
            "R_AMDGPU_ABS32_LO", "R_AMDGPU_ABS32_HI", "R_AMDGPU_ABS64",
            "R_AMDGPU_REL32", "R_AMDGPU_REL64", "R_AMDGPU_ABS32",
            "R_AMDGPU_GOTPCREL", "R_AMDGPU_GOTPCREL32_LO",
            "R_AMDGPU_GOTPCREL32_HI", "R_AMDGPU_REL32_LO",
            "R_AMDGPU_REL32_HI", "R_AMDGPU_REL16",
        })
        self.assertIn("B + A", reloc["dynamic_formula"])

    def test_elf_and_segment_headers_match_fixtures(self) -> None:
        for fixture in self.spec["deterministic_fixtures"]:
            # source_path is rooted at the rocm-systems repository.
            path = ROOT / "projects" / "rocm-systems" / fixture["source_path"]
            data = path.read_bytes()
            blob, locked_data = git_blob(
                REPOSITORIES["rocm-systems"], fixture["commit"], fixture["source_path"]
            )
            self.assertEqual(blob, fixture["git_blob"], fixture["id"])
            self.assertEqual(locked_data, data, fixture["id"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), fixture["sha256"])
            self.assertEqual(len(data), fixture["bytes"], fixture["id"])
            ident, e_type, e_machine, version, entry, _, _, flags, _, _, _, _, _, _ = (
                struct.unpack_from("<16sHHIQQQIHHHHHH", data, 0)
            )
            self.assertEqual(ident[:4], b"\x7fELF")
            self.assertEqual((ident[4], ident[5], ident[7], ident[8]), (2, 1, 64, 4))
            self.assertEqual(
                (e_type, e_machine, version, entry, flags),
                (3, 224, 1, 0, int(fixture["elf"]["flags_hex"], 16)),
            )
            sections, _ = section_table(data)
            by_name = {section["name"]: section for section in sections}
            for name, expected in fixture["elf"]["sections"].items():
                actual = by_name[name]
                self.assertEqual(actual["offset"], int(expected["offset_hex"], 16), name)
                self.assertEqual(actual["size"], int(expected["size_hex"], 16), name)
                if "vaddr_hex" in expected:
                    self.assertEqual(actual["addr"], int(expected["vaddr_hex"], 16), name)
                if "alignment" in expected:
                    self.assertEqual(actual["align"], expected["alignment"], name)
            self.assertEqual(
                sum(section["type"] in (4, 9) for section in sections),
                fixture["elf"]["relocation_sections"],
            )
            loads = [item for item in program_headers(data) if item["type"] == 1]
            self.assertEqual(len(loads), len(fixture["segments"]))
            flag_names = {4: "R", 5: "RX", 6: "RW", 7: "RWX"}
            for actual, expected in zip(loads, fixture["segments"]):
                self.assertEqual(actual["offset"], int(expected["offset_hex"], 16))
                self.assertEqual(actual["vaddr"], int(expected["vaddr_hex"], 16))
                self.assertEqual(actual["filesz"], int(expected["filesz_hex"], 16))
                self.assertEqual(actual["memsz"], int(expected["memsz_hex"], 16))
                self.assertEqual(actual["align"], int(expected["align_hex"], 16))
                self.assertEqual(flag_names[actual["flags"]], expected["flags"])

    def test_note_is_amdgpu_msgpack_and_metadata_matches(self) -> None:
        for fixture in self.spec["deterministic_fixtures"]:
            path = ROOT / "projects" / "rocm-systems" / fixture["source_path"]
            data = path.read_bytes()
            sections, _ = section_table(data)
            note = next(section for section in sections if section["name"] == ".note")
            note_bytes = data[note["offset"] : note["offset"] + note["size"]]
            namesz, descsz, note_type = struct.unpack_from("<III", note_bytes, 0)
            name_offset = 12
            desc_offset = 12 + ((namesz + 3) // 4) * 4
            owner = note_bytes[name_offset : name_offset + namesz].rstrip(b"\0")
            descriptor = note_bytes[desc_offset : desc_offset + descsz]
            self.assertEqual(owner, b"AMDGPU")
            self.assertEqual(note_type, 32)
            self.assertEqual(
                hashlib.sha256(descriptor).hexdigest(),
                fixture["metadata"]["desc_sha256"],
            )
            if msgpack is None:
                continue
            decoded = msgpack.unpackb(
                descriptor,
                raw=False,
                strict_map_key=False,
                max_map_len=100000,
                max_array_len=100000,
            )
            metadata = fixture["metadata"]
            self.assertEqual(decoded["amdhsa.version"], metadata["version"])
            self.assertEqual(decoded["amdhsa.target"], metadata["target"])
            actual_kernels = decoded["amdhsa.kernels"]
            self.assertEqual(len(actual_kernels), len(metadata["kernels"]))
            scalar_keys = (
                "name", "symbol", "kernarg_segment_size", "kernarg_segment_align",
                "group_segment_fixed_size", "private_segment_fixed_size",
                "uses_dynamic_stack", "wavefront_size", "sgpr_count", "vgpr_count",
                "agpr_count", "sgpr_spill_count", "vgpr_spill_count",
                "max_flat_workgroup_size",
            )
            for actual, expected in zip(actual_kernels, metadata["kernels"]):
                for key in scalar_keys:
                    self.assertEqual(actual[f".{key}"], expected[key], key)
                self.assertEqual(len(actual[".args"]), len(expected["args"]))
                for actual_arg, expected_arg in zip(actual[".args"], expected["args"]):
                    for key, value in expected_arg.items():
                        self.assertEqual(actual_arg[f".{key}"], value, key)

    def test_descriptor_bytes_fields_and_entry_relation_match(self) -> None:
        for fixture in self.spec["deterministic_fixtures"]:
            path = ROOT / "projects" / "rocm-systems" / fixture["source_path"]
            data = path.read_bytes()
            sections, _ = section_table(data)
            rodata = next(section for section in sections if section["name"] == ".rodata")
            for descriptor in fixture["descriptors"]:
                offset = int(descriptor["file_offset_hex"], 16)
                raw = data[offset : offset + 64]
                self.assertEqual(len(raw), 64, descriptor["symbol"])
                self.assertEqual(raw.hex(), descriptor["hex"], descriptor["symbol"])
                self.assertEqual(
                    hashlib.sha256(raw).hexdigest(), descriptor["sha256"], descriptor["symbol"]
                )
                fields = descriptor["fields"]
                self.assertEqual(int.from_bytes(raw[0:4], "little"), fields["group_segment_fixed_size"])
                self.assertEqual(int.from_bytes(raw[4:8], "little"), fields["private_segment_fixed_size"])
                self.assertEqual(int.from_bytes(raw[8:12], "little"), fields["kernarg_segment_size"])
                self.assertEqual(
                    int.from_bytes(raw[16:24], "little"),
                    int(fields["kernel_code_entry_byte_offset_hex"], 16),
                )
                self.assertEqual(int.from_bytes(raw[44:48], "little"), int(fields["compute_pgm_rsrc3_hex"], 16))
                self.assertEqual(int.from_bytes(raw[48:52], "little"), int(fields["compute_pgm_rsrc1_hex"], 16))
                self.assertEqual(int.from_bytes(raw[52:56], "little"), int(fields["compute_pgm_rsrc2_hex"], 16))
                self.assertEqual(int.from_bytes(raw[56:58], "little"), int(fields["kernel_code_properties_hex"], 16))
                self.assertEqual(int.from_bytes(raw[58:60], "little"), int(fields["kernarg_preload_hex"], 16))
                self.assertEqual(
                    rodata["offset"] + int(descriptor["file_offset_hex"], 16) - int(fixture["elf"]["sections"][".rodata"]["offset_hex"], 16),
                    offset,
                )

    def test_symbol_lookup_is_descriptor_based(self) -> None:
        type_values = {"STT_FUNC": 2, "STT_OBJECT": 1}
        bind_values = {"GLOBAL": 1}
        visibility_values = {"PROTECTED": 3}
        for fixture in self.spec["deterministic_fixtures"]:
            path = ROOT / "projects" / "rocm-systems" / fixture["source_path"]
            symbols = dynamic_symbols(path.read_bytes(), section_table(path.read_bytes())[0])
            expected_by_name = {symbol["name"]: symbol for symbol in fixture["symbols"]}
            self.assertEqual(set(symbols), set(expected_by_name))
            for name, expected in expected_by_name.items():
                actual = symbols[name]
                self.assertEqual(actual["type"], type_values[expected["type"]], name)
                self.assertEqual(actual["bind"], bind_values[expected["binding"]], name)
                self.assertEqual(actual["visibility"], visibility_values[expected["visibility"]], name)
                self.assertEqual(actual["value"], int(expected["value_hex"], 16), name)
                self.assertEqual(actual["size"], expected["size"], name)
            for kernel in fixture["metadata"]["kernels"]:
                self.assertIn(kernel["symbol"], symbols)
                self.assertIn(kernel["name"], symbols)
                self.assertEqual(kernel["symbol"], f"{kernel['name']}.kd")

    def test_kernarg_layout_is_aligned_and_hidden_fields_are_preserved(self) -> None:
        for fixture in self.spec["deterministic_fixtures"]:
            for kernel in fixture["metadata"]["kernels"]:
                align = kernel["kernarg_segment_align"]
                self.assertGreaterEqual(align, 4)
                self.assertEqual(align & (align - 1), 0)
                args = sorted(kernel["args"], key=lambda item: item["offset"])
                previous_end = 0
                for argument in args:
                    self.assertGreaterEqual(argument["offset"], previous_end)
                    previous_end = argument["offset"] + argument["size"]
                    self.assertLessEqual(previous_end, kernel["kernarg_segment_size"])
                    self.assertIn("value_kind", argument)
                    self.assertIn("offset", argument)
                    self.assertIn("size", argument)
                hidden = [argument for argument in args if argument["value_kind"].startswith("hidden_")]
                self.assertTrue(hidden)
                self.assertIn("hidden_global_offset_x", {arg["value_kind"] for arg in hidden})
                self.assertIn("hidden_grid_dims", {arg["value_kind"] for arg in hidden})
                self.assertEqual(kernel["group_segment_fixed_size"], 0)
                self.assertEqual(kernel["private_segment_fixed_size"], 0)

    def test_isa_audit_is_explicitly_blocked(self) -> None:
        audit = self.spec["isa_audit"]
        self.assertFalse(audit["pinned_toolchain"])
        self.assertFalse(audit["gem5_gfx950_feature_proof"])
        self.assertTrue(audit["gem5_evidence"]["enum_has_gfx950"])
        self.assertFalse(audit["gem5_evidence"]["decoder_set_gfx_version_has_gfx950_case"])
        self.assertEqual(audit["current_status"], "blocked")
        fixtures = {fixture["id"]: fixture for fixture in self.spec["deterministic_fixtures"]}
        self.assertEqual(fixtures["gfx950-rdc-binary-search-v1"]["isa_audit"]["unsupported_mnemonics"], ["v_fmamk_f32"])
        self.assertEqual(fixtures["gfx950-rdc-binary-search-v1"]["isa_audit"]["unsupported_occurrences"], 2)
        self.assertEqual(fixtures["gfx950-rdc-gpuReadWrite-v1"]["isa_audit"]["unsupported_mnemonics"], [])
        self.assertTrue(all(fixture["isa_audit"]["status"] != "accepted" for fixture in self.spec["deterministic_fixtures"]))

    def test_document_cross_reference_and_no_forbidden_upload_api(self) -> None:
        docs = (ROOT / "docs" / "host-transport-v1-codeobj.md").read_text(encoding="utf-8")
        self.assertIn("protocol/host-transport-v1-codeobj.json", docs)
        self.assertIn("CODE_OBJECT_ABI_V1", docs)
        self.assertIn("v_fmamk_f32", docs)
        self.assertIn("device-libs", docs)
        serialized = SPEC_PATH.read_text(encoding="utf-8")
        self.assertNotIn('"CODE_OBJECT_REQUEST"', serialized)
        self.assertNotIn('"HSACO_UPLOAD"', serialized)
        self.assertIn("no code-object upload", self.spec["scope"])
        self.assertGreaterEqual(len(self.spec["blockers"]), 5)


if __name__ == "__main__":
    unittest.main()
