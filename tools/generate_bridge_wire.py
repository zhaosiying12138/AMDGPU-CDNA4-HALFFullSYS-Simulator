#!/usr/bin/env python3
"""Generate the shared runtime-gem5 generic-v2 wire declarations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_SCHEMA = ROOT / "protocol/host-transport-v1.json"
GENERIC_SCHEMA = ROOT / "protocol/host-transport-v2.json"
KMT_SCHEMA = ROOT / "protocol/host-transport-v1-kmt.json"
C_OUTPUT = (
    ROOT
    / "projects/self-amdgpu-runtime/include/self_amdgpu_runtime/generated/bridge_generic_v2.h"
)
CPP_OUTPUT = (
    ROOT / "projects/gem5/src/dev/amdgpu/generated/bridge_generic_v2.hh"
)
KMT_C_OUTPUT = (
    ROOT
    / "projects/self-amdgpu-runtime/include/self_amdgpu_runtime/generated/bridge_kmt_v5.h"
)
KMT_CPP_OUTPUT = ROOT / "projects/gem5/src/dev/amdgpu/generated/bridge_kmt_v5.hh"
VECTOR_DIRECTORY = ROOT / "tests/fixtures/runtime-gem5-bridge/generic-v2"


def _load(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: root must be an object")
    return document


def _field(fields: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [field for field in fields if field.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"field {name!r} must occur exactly once")
    return matches[0]


def _range_max(field: dict[str, Any]) -> int:
    match = re.fullmatch(r"(?:[^.]+\.\.)?(\d+)", str(field.get("range", "")))
    if match is None:
        raise ValueError(f"field {field.get('name')!r} has no integer upper bound")
    return int(match.group(1))


def _hex_int(value: Any) -> int:
    text = str(value)
    return int(text, 16) if not text.lower().startswith("0x") else int(text, 0)


def contract() -> dict[str, Any]:
    base = _load(BASE_SCHEMA)
    generic = _load(GENERIC_SCHEMA)
    if base.get("schema") != "amdgpu-sim.host-transport.v1":
        raise ValueError("unexpected base transport schema")
    if generic.get("schema") != "amdgpu-sim.host-transport-v2.generic-dispatch.v2":
        raise ValueError("unexpected generic transport schema")

    base_transport = base["transport"]
    base_header = base["header"]
    transport = generic["transport"]
    request = generic["request_payload"]
    response = generic["response_payload"]
    capability = generic["capability"]
    execution = generic["execution_capability"]
    versioning = generic["versioning"]

    if base_transport["byte_order"] != "big-endian":
        raise ValueError("base transport must be big-endian")
    if transport["byte_order"] != "big-endian for all serialized scalar fields":
        raise ValueError("generic transport must be big-endian")
    if transport["header_bytes"] != base_header["bytes"]:
        raise ValueError("generic header size differs from base envelope")
    if transport["record_bytes"] != transport["header_bytes"] + transport["payload_bytes"]:
        raise ValueError("generic record size is inconsistent")
    if request["bytes"] != transport["payload_bytes"] or response["bytes"] != transport["payload_bytes"]:
        raise ValueError("generic payload sizes are inconsistent")

    common_fields = request["common_fields"]
    stage_bodies = request["stage_bodies"]
    submit_fields = stage_bodies["SUBMIT_AQL"]["fields"]
    map_fields = stage_bodies["MAP_OBJECT"]["fields"]
    alloc_fields = stage_bodies["ALLOC_KERNARG"]["fields"]
    kernel_name = _field(common_fields, "kernel_name")
    kernarg_map = _field(map_fields, "kernarg_segment_size")
    kernarg_alloc = _field(alloc_fields, "size_bytes")
    kernarg_submit = _field(submit_fields, "kernarg_size")
    kernarg_maxima = {
        _range_max(kernarg_map),
        _range_max(kernarg_alloc),
        _range_max(kernarg_submit),
    }
    if len(kernarg_maxima) != 1:
        raise ValueError("kernarg bounds differ across generic stages")

    fields: dict[str, int] = {}
    for prefix, items in (
        ("REQUEST", common_fields),
        ("REQUEST_MAP", map_fields),
        ("REQUEST_ALLOC", alloc_fields),
        ("REQUEST_SUBMIT", submit_fields),
        ("RESPONSE", response["fields"]),
    ):
        for field in items:
            key = re.sub(r"[^A-Z0-9]+", "_", field["name"].upper())
            fields[f"{prefix}_{key}_OFFSET"] = int(field["offset"])
            fields[f"{prefix}_{key}_BYTES"] = int(field["bytes"])

    values: dict[str, int] = {
        "FRAMING_MAJOR": int(versioning["framing_major"]),
        "FRAMING_MINOR": int(versioning["framing_minor"]),
        "PAYLOAD_MAJOR": int(versioning["payload_major"]),
        "PAYLOAD_MINOR": int(versioning["payload_minor"]),
        "HEADER_BYTES": int(transport["header_bytes"]),
        "RECORD_BYTES": int(transport["record_bytes"]),
        "PAYLOAD_BYTES": int(transport["payload_bytes"]),
        "REQUEST_COMMON_BYTES": int(request["common_prefix_bytes"]),
        "REQUEST_MAP_ACTIVE_END": int(stage_bodies["MAP_OBJECT"]["active_end"]),
        "REQUEST_ALLOC_ACTIVE_END": int(stage_bodies["ALLOC_KERNARG"]["active_end"]),
        "REQUEST_SUBMIT_ACTIVE_END": int(stage_bodies["SUBMIT_AQL"]["active_end"]),
        "REQUEST_UNMAP_ACTIVE_END": int(stage_bodies["UNMAP_OBJECT"]["active_end"]),
        "RESPONSE_ACTIVE_END": max(
            int(field["offset"]) + int(field["bytes"])
            for field in response["fields"]
        ),
        "KERNEL_NAME_BYTES": int(kernel_name["bytes"]),
        "MAX_KERNARG_BYTES": kernarg_maxima.pop(),
        "MAX_SHARED_BYTES": _range_max(_field(submit_fields, "shared_memory_bytes")),
        "MAX_WORKGROUP_DIMENSION": _range_max(_field(submit_fields, "workgroup_x")),
        "MAX_WARPS": _range_max(_field(submit_fields, "num_warps")),
        "MAX_CTAS": _range_max(_field(submit_fields, "num_ctas")),
        "GENERIC_CAPABILITY_WORD": int(capability["word_index"]),
        "GENERIC_CAPABILITY_BIT": int(capability["bit"]),
        "GENERIC_CAPABILITY_MASK": _hex_int(capability["mask_hex"]),
        "GENERIC_CAPABILITY_WIRE_BYTE": int(capability["wire_byte_index"]),
        "GENERIC_CAPABILITY_WIRE_MASK": _hex_int(capability["wire_mask_hex"]),
        "EXECUTION_CAPABILITY_WORD": int(execution["word_index"]),
        "EXECUTION_CAPABILITY_BIT": int(execution["bit"]),
        "EXECUTION_CAPABILITY_MASK": _hex_int(execution["mask_hex"]),
        "EXECUTION_CAPABILITY_WIRE_BYTE": int(execution["wire_byte_index"]),
        "EXECUTION_CAPABILITY_WIRE_MASK": _hex_int(execution["wire_mask_hex"]),
    }
    for name, value in transport["message_types"].items():
        values[f"MESSAGE_{name}"] = int(value)
    for name, value in transport["opcodes"].items():
        values[f"OPCODE_{name}"] = int(value)
    for name, value in base["statuses"].items():
        values[f"STATUS_{name}"] = int(value)
    values.update(fields)

    if values["GENERIC_CAPABILITY_MASK"] != 1 << values["GENERIC_CAPABILITY_BIT"]:
        raise ValueError("generic capability word mask is inconsistent")
    if values["EXECUTION_CAPABILITY_MASK"] != 1 << values["EXECUTION_CAPABILITY_BIT"]:
        raise ValueError("execution capability word mask is inconsistent")
    return {"schema": "amdgpu-sim.runtime-gem5-bridge.generic-v2.v1", "values": values}


def kmt_contract() -> dict[str, Any]:
    base = _load(BASE_SCHEMA)
    kmt = _load(KMT_SCHEMA)
    if kmt.get("schema") != "amdgpu-sim.host-transport-v1.kmt.v5":
        raise ValueError("unexpected KMT transport schema")
    if kmt.get("base_protocol") != "protocol/host-transport-v1.json":
        raise ValueError("KMT transport does not name the base wire authority")
    transport = kmt["transport"]
    request = kmt["request_payload"]
    result = kmt["result_payload"]
    if transport["header_bytes"] != base["header"]["bytes"]:
        raise ValueError("KMT header size differs from base envelope")
    if request["bytes"] != transport["request_payload_bytes"]:
        raise ValueError("KMT request payload size is inconsistent")
    if result["bytes"] != transport["result_payload_bytes"]:
        raise ValueError("KMT result payload size is inconsistent")
    if transport["request_frame_bytes"] != (
        transport["header_bytes"] + request["bytes"]
    ):
        raise ValueError("KMT request frame size is inconsistent")
    if transport["result_frame_bytes"] != (
        transport["header_bytes"] + result["bytes"]
    ):
        raise ValueError("KMT result frame size is inconsistent")

    request_fields = request["fields"]
    result_fields = result["fields"]
    request_major = _field(request_fields, "kmt_major")
    request_minor = _field(request_fields, "kmt_minor")
    result_major = _field(result_fields, "kmt_major")
    result_minor = _field(result_fields, "kmt_minor")
    if request_major["constant"] != result_major["constant"]:
        raise ValueError("KMT request/result major versions differ")
    if request_minor["constant"] != result_minor["constant"]:
        raise ValueError("KMT request/result minor versions differ")
    request_buffer = _field(request_fields, "copied_buffer")
    result_buffer = _field(result_fields, "copied_result")
    if request_buffer["bytes"] != result_buffer["bytes"]:
        raise ValueError("KMT request/result copied-buffer capacities differ")
    argument_words = _field(request_fields, "arg_words")
    result_words = _field(result_fields, "result_words")
    if argument_words["bytes"] != result_words["bytes"]:
        raise ValueError("KMT request/result word capacities differ")
    operations = kmt["operations"]
    operation_layouts = kmt["operation_layouts"]
    operation_ids = [int(entry["id"]) for entry in operations.values()]
    if operation_ids != list(range(1, len(operation_ids) + 1)):
        raise ValueError("KMT operation IDs must be contiguous and ordered")

    layout = kmt["shared_backing_layout"]
    layout_major = int(layout["layout_major"])
    alignment_bytes = int(layout["alignment_bytes"])
    doorbell_region_bytes = int(layout["doorbell_region_bytes"])
    doorbell_region_base_bytes = int(layout["doorbell_region_base_bytes"])
    doorbell_slot_bytes = int(layout["doorbell_slot_bytes"])
    maximum_doorbell_slots = int(layout["maximum_doorbell_slots"])
    completion_region_base_bytes = int(layout["completion_region_base_bytes"])
    completion_slot_bytes = int(layout["completion_slot_bytes"])
    completion_initial_value = int(layout["completion_initial_value"])
    userptr_memory_flag_mask = int(layout["userptr_memory_flag_mask"])
    doorbell_memory_flag_mask = int(layout["doorbell_memory_flag_mask"])
    if layout_major != 1:
        raise ValueError("unexpected KMT shared-backing layout version")
    if alignment_bytes < 4096 or alignment_bytes & (alignment_bytes - 1):
        raise ValueError("KMT shared-backing alignment must be a power of two")
    if (
        doorbell_region_bytes == 0
        or doorbell_region_bytes % alignment_bytes
        or doorbell_slot_bytes == 0
        or maximum_doorbell_slots == 0
        or doorbell_region_base_bytes < 0
        or maximum_doorbell_slots * doorbell_slot_bytes
            > doorbell_region_bytes - doorbell_region_base_bytes
        or completion_region_base_bytes < 0
        or completion_slot_bytes == 0
        or completion_region_base_bytes
            < doorbell_region_base_bytes
                + maximum_doorbell_slots * doorbell_slot_bytes
        or maximum_doorbell_slots * completion_slot_bytes
            > doorbell_region_bytes - completion_region_base_bytes
        or layout["doorbell_initial_value"] != "UINT64_MAX"
        or completion_initial_value != 0
    ):
        raise ValueError("invalid KMT doorbell aperture layout")
    if (
        userptr_memory_flag_mask == 0
        or userptr_memory_flag_mask & (userptr_memory_flag_mask - 1)
        or doorbell_memory_flag_mask == 0
        or doorbell_memory_flag_mask & (doorbell_memory_flag_mask - 1)
        or userptr_memory_flag_mask == doorbell_memory_flag_mask
    ):
        raise ValueError("invalid KMT shared-backing memory flag masks")

    clock = kmt["clock_correlation"]
    clock_frequency_hz = int(clock["system_clock_frequency_hz"])
    if clock.get("counter_unit") != "nanoseconds" or clock_frequency_hz != 10**9:
        raise ValueError("KMT clock correlation must use nanoseconds at 1 GHz")

    values: dict[str, int] = {
        "KMT_PAYLOAD_MAJOR": int(request_major["constant"]),
        "KMT_PAYLOAD_MINOR": int(request_minor["constant"]),
        "KMT_HEADER_BYTES": int(transport["header_bytes"]),
        "KMT_REQUEST_PAYLOAD_BYTES": int(transport["request_payload_bytes"]),
        "KMT_RESULT_PAYLOAD_BYTES": int(transport["result_payload_bytes"]),
        "KMT_REQUEST_FRAME_BYTES": int(transport["request_frame_bytes"]),
        "KMT_RESULT_FRAME_BYTES": int(transport["result_frame_bytes"]),
        "KMT_COPIED_BUFFER_BYTES": int(request_buffer["bytes"]),
        "KMT_ARGUMENT_WORD_COUNT": int(argument_words["bytes"]) // 4,
        "KMT_CAPABILITY_BYTE": int(kmt["capability"]["byte_index"]),
        "KMT_CAPABILITY_MASK": _hex_int(kmt["capability"]["mask_hex"]),
        "KMT_SHARED_BACKING_LAYOUT_MAJOR": layout_major,
        "KMT_SHARED_BACKING_ALIGNMENT_BYTES": alignment_bytes,
        "KMT_SHARED_BACKING_DOORBELL_REGION_BYTES": doorbell_region_bytes,
        "KMT_SHARED_BACKING_DOORBELL_REGION_BASE_BYTES": doorbell_region_base_bytes,
        "KMT_SHARED_BACKING_DOORBELL_SLOT_BYTES": doorbell_slot_bytes,
        "KMT_SHARED_BACKING_MAXIMUM_DOORBELL_SLOTS": maximum_doorbell_slots,
        "KMT_SHARED_BACKING_DOORBELL_INITIAL_VALUE": (1 << 64) - 1,
        "KMT_SHARED_BACKING_COMPLETION_REGION_BASE_BYTES": completion_region_base_bytes,
        "KMT_SHARED_BACKING_COMPLETION_SLOT_BYTES": completion_slot_bytes,
        "KMT_SHARED_BACKING_COMPLETION_INITIAL_VALUE": completion_initial_value,
        "KMT_SHARED_BACKING_USERPTR_MEMORY_FLAG_MASK": userptr_memory_flag_mask,
        "KMT_SHARED_BACKING_DOORBELL_MEMORY_FLAG_MASK": doorbell_memory_flag_mask,
        "KMT_CLOCK_FREQUENCY_HZ": clock_frequency_hz,
    }
    for name, value in transport["message_types"].items():
        values[f"KMT_MESSAGE_{name.removeprefix('KMT_')}"] = int(value)
    for name, entry in operations.items():
        values[f"KMT_OPERATION_{name}"] = int(entry["id"])

    request_word_masks: dict[str, int] = {}
    result_word_masks: dict[str, int] = {}
    if list(operation_layouts) != list(operations):
        raise ValueError("KMT operation layouts must match operation order")
    for name, layout_entry in operation_layouts.items():
        if int(layout_entry["id"]) != int(operations[name]["id"]):
            raise ValueError(f"KMT operation layout ID differs for {name}")
        masks = []
        for field_name in ("arguments", "results"):
            mask = 0
            for field in layout_entry[field_name]:
                word = int(field["word"])
                if word < 0 or word >= values["KMT_ARGUMENT_WORD_COUNT"]:
                    raise ValueError(
                        f"KMT {name} {field_name} word {word} is out of range"
                    )
                mask |= 1 << word
            masks.append(mask)
        request_word_masks[name], result_word_masks[name] = masks
    return {
        "schema": "amdgpu-sim.runtime-gem5-bridge.kmt-v5.v1",
        "values": values,
        "request_word_masks": request_word_masks,
        "result_word_masks": result_word_masks,
    }


def _digest(document: dict[str, Any]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def render_c(document: dict[str, Any]) -> str:
    lines = [
        "/* SPDX-License-Identifier: GPL-3.0-or-later */",
        "/* Generated by tools/generate_bridge_wire.py; do not edit. */",
        f"/* Machine contract SHA-256: {_digest(document)} */",
        "",
        "#ifndef SELF_AMDGPU_RUNTIME_GENERATED_BRIDGE_GENERIC_V2_H",
        "#define SELF_AMDGPU_RUNTIME_GENERATED_BRIDGE_GENERIC_V2_H",
        "",
        "#include <stdint.h>",
        "",
    ]
    for name, value in document["values"].items():
        macro = f"SAGR_BRIDGE_GENERIC_V2_{name}"
        if name == "MAX_KERNARG_BYTES":
            suffix = "UINT64_C"
        elif "CAPABILITY_MASK" in name and "WIRE" not in name:
            suffix = "UINT64_C"
        elif name.endswith(("WIRE_BYTE", "WIRE_MASK")):
            suffix = "UINT8_C"
        elif name.startswith(("MESSAGE_", "OPCODE_")) or name in {
            "FRAMING_MAJOR",
            "FRAMING_MINOR",
            "PAYLOAD_MAJOR",
            "PAYLOAD_MINOR",
        }:
            suffix = "UINT16_C"
        else:
            suffix = "UINT32_C"
        lines.append(f"#define {macro} {suffix}({value})")
    lines.extend(["", "#endif", ""])
    return "\n".join(lines)


def render_cpp(document: dict[str, Any]) -> str:
    lines = [
        "/*",
        " * Copyright (c) 2026 The gem5 Project",
        " * All rights reserved.",
        " *",
        " * Generated by tools/generate_bridge_wire.py; do not edit.",
        " */",
        f"/* Machine contract SHA-256: {_digest(document)} */",
        "",
        "#ifndef __DEV_AMDGPU_GENERATED_BRIDGE_GENERIC_V2_HH__",
        "#define __DEV_AMDGPU_GENERATED_BRIDGE_GENERIC_V2_HH__",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace gem5::hostgpu::bridgewire",
        "{",
    ]
    for name, value in document["values"].items():
        identifier = "".join(word.title() for word in name.lower().split("_"))
        if "CAPABILITY_MASK" in name and "WIRE" not in name:
            ctype = "uint64_t"
        elif name.endswith(("WIRE_BYTE", "WIRE_MASK")):
            ctype = "uint8_t"
        elif name.startswith(("MESSAGE_", "OPCODE_")) or name in {
            "FRAMING_MAJOR",
            "FRAMING_MINOR",
            "PAYLOAD_MAJOR",
            "PAYLOAD_MINOR",
        }:
            ctype = "uint16_t"
        else:
            ctype = "uint32_t"
        literal = f"{value}ULL" if ctype == "uint64_t" else str(value)
        lines.append(f"inline constexpr {ctype} {identifier} = {literal};")
    lines.extend(["}", "", "#endif", ""])
    return "\n".join(lines)


def render_kmt_c(document: dict[str, Any]) -> str:
    last_operation = next(reversed(document["request_word_masks"]))
    lines = [
        "/* SPDX-License-Identifier: GPL-3.0-or-later */",
        "/* Generated by tools/generate_bridge_wire.py; do not edit. */",
        f"/* Machine contract SHA-256: {_digest(document)} */",
        "",
        "#ifndef SELF_AMDGPU_RUNTIME_GENERATED_BRIDGE_KMT_V5_H",
        "#define SELF_AMDGPU_RUNTIME_GENERATED_BRIDGE_KMT_V5_H",
        "",
        "#include <stdint.h>",
        "",
    ]
    for name, value in document["values"].items():
        suffix = (
            "UINT8_C"
            if name.endswith(("CAPABILITY_BYTE", "CAPABILITY_MASK"))
            else "UINT16_C"
            if name.startswith(("KMT_PAYLOAD_", "KMT_MESSAGE_", "KMT_OPERATION_"))
            else "UINT64_C"
            if value > (1 << 32) - 1
            else "UINT32_C"
        )
        lines.append(f"#define SAGR_BRIDGE_{name} {suffix}({value})")
    lines.extend(
        [
            "",
            "static inline int",
            "sagr_bridge_kmt_operation_valid(uint16_t operation)",
            "{",
            "  return operation >= SAGR_BRIDGE_KMT_OPERATION_OPEN_KFD &&",
            f"         operation <= SAGR_BRIDGE_KMT_OPERATION_{last_operation};",
            "}",
        ]
    )
    for direction in ("request", "result"):
        lines.extend(
            [
                "",
                "static inline uint32_t",
                f"sagr_bridge_kmt_{direction}_word_mask(uint16_t operation)",
                "{",
                "  switch (operation) {",
            ]
        )
        for name, mask in document[f"{direction}_word_masks"].items():
            lines.append(
                f"    case SAGR_BRIDGE_KMT_OPERATION_{name}: "
                f"return UINT32_C({mask});"
            )
        lines.extend(["    default: return UINT32_C(0);", "  }", "}"])
    lines.extend(["", "#endif", ""])
    return "\n".join(lines)


def render_kmt_cpp(document: dict[str, Any]) -> str:
    last_operation = "".join(
        word.title()
        for word in next(reversed(document["request_word_masks"])).lower().split("_")
    )
    lines = [
        "/*",
        " * Copyright (c) 2026 The gem5 Project",
        " * All rights reserved.",
        " *",
        " * Generated by tools/generate_bridge_wire.py; do not edit.",
        " */",
        f"/* Machine contract SHA-256: {_digest(document)} */",
        "",
        "#ifndef __DEV_AMDGPU_GENERATED_BRIDGE_KMT_V5_HH__",
        "#define __DEV_AMDGPU_GENERATED_BRIDGE_KMT_V5_HH__",
        "",
        "#include <cstdint>",
        "",
        "namespace gem5::hostgpu::bridgekmt",
        "{",
    ]
    for name, value in document["values"].items():
        identifier = "".join(word.title() for word in name.lower().split("_"))
        ctype = (
            "uint8_t"
            if name.endswith(("CAPABILITY_BYTE", "CAPABILITY_MASK"))
            else "uint16_t"
            if name.startswith(("KMT_PAYLOAD_", "KMT_MESSAGE_", "KMT_OPERATION_"))
            else "uint64_t"
            if value > (1 << 32) - 1
            else "uint32_t"
        )
        literal = f"{value}ULL" if ctype == "uint64_t" else str(value)
        lines.append(f"inline constexpr {ctype} {identifier} = {literal};")
    lines.extend(
        [
            "",
            "inline constexpr bool",
            "kmtOperationValid(uint16_t operation)",
            "{",
            "    return operation >= KmtOperationOpenKfd &&",
            f"           operation <= KmtOperation{last_operation};",
            "}",
        ]
    )
    for direction in ("request", "result"):
        lines.extend(
            [
                "",
                "inline constexpr uint32_t",
                f"kmt{direction.title()}WordMask(uint16_t operation)",
                "{",
                "    switch (operation) {",
            ]
        )
        for name, mask in document[f"{direction}_word_masks"].items():
            identifier = "".join(word.title() for word in name.lower().split("_"))
            lines.append(
                f"      case KmtOperation{identifier}: return {mask};"
            )
        lines.extend(["      default: return 0;", "    }", "}"])
    lines.extend(["}", "", "#endif", ""])
    return "\n".join(lines)


def _store(payload: bytearray, offset: int, size: int, value: int) -> None:
    payload[offset : offset + size] = value.to_bytes(size, "big")


def _request_payload(values: dict[str, int], opcode: str) -> bytes:
    payload = bytearray(values["PAYLOAD_BYTES"])
    for field, value in (("PAYLOAD_MAJOR", values["PAYLOAD_MAJOR"]),
                         ("PAYLOAD_MINOR", values["PAYLOAD_MINOR"]),
                         ("OPCODE", values[f"OPCODE_{opcode}"])):
        _store(payload, values[f"REQUEST_{field}_OFFSET"],
               values[f"REQUEST_{field}_BYTES"], value)
    _store(payload, values["REQUEST_OBJECT_ID_OFFSET"], 8, 7)
    _store(payload, values["REQUEST_OBJECT_GENERATION_OFFSET"], 8, 9)
    if opcode in {"ALLOC_KERNARG", "SUBMIT_AQL", "UNMAP_OBJECT"}:
        _store(payload, values["REQUEST_MAPPING_ID_OFFSET"], 8, 11)
        _store(payload, values["REQUEST_MAPPING_GENERATION_OFFSET"], 8, 13)
    if opcode == "SUBMIT_AQL":
        _store(payload, values["REQUEST_QUEUE_ID_OFFSET"], 8, 17)
        _store(payload, values["REQUEST_QUEUE_GENERATION_OFFSET"], 8, 19)
        _store(payload, values["REQUEST_QUEUE_SEQUENCE_OFFSET"], 8, 23)
    if opcode in {"MAP_OBJECT", "SUBMIT_AQL"}:
        digest_offset = values["REQUEST_IMAGE_SHA256_OFFSET"]
        payload[digest_offset : digest_offset + 32] = bytes(range(1, 33))
        name = b"bridge_generic_kernel"
        name_offset = values["REQUEST_KERNEL_NAME_OFFSET"]
        payload[name_offset : name_offset + len(name)] = name
    if opcode == "MAP_OBJECT":
        for name, size, value in (
            ("GFX_TARGET", 4, 950), ("RELOCATION_COUNT", 4, 0),
            ("KERNARG_SEGMENT_SIZE", 4, 48),
            ("KERNARG_SEGMENT_ALIGN", 4, 8),
            ("DESCRIPTOR_PRELOAD_DWORDS", 4, 12), ("PAGE_SIZE", 4, 4096),
        ):
            _store(payload, values[f"REQUEST_MAP_{name}_OFFSET"], size, value)
    elif opcode == "ALLOC_KERNARG":
        _store(payload, values["REQUEST_ALLOC_SIZE_BYTES_OFFSET"], 8, 48)
        _store(payload, values["REQUEST_ALLOC_ALIGNMENT_BYTES_OFFSET"], 8, 8)
    elif opcode == "SUBMIT_AQL":
        fields = (
            ("KERNARG_ALLOCATION_ID", 8, 29), ("KERNARG_GENERATION", 8, 31),
            ("KERNARG_OFFSET", 8, 0), ("KERNARG_SIZE", 8, 48),
            ("SIGNAL_ID", 8, 37), ("SIGNAL_GENERATION", 8, 41),
            ("EXPECTED_SIGNAL_VALUE_BITS", 8, 1), ("GRID_X", 4, 24832),
            ("GRID_Y", 4, 1), ("GRID_Z", 4, 1), ("WORKGROUP_X", 4, 256),
            ("WORKGROUP_Y", 4, 1), ("WORKGROUP_Z", 4, 1),
            ("NUM_WARPS", 4, 4), ("NUM_CTAS", 4, 1),
            ("SHARED_MEMORY_BYTES", 4, 0), ("WAVEFRONT_SIZE", 4, 64),
            ("LAUNCH_FLAGS", 4, 0), ("RESERVED0", 4, 0),
        )
        for name, size, value in fields:
            _store(payload, values[f"REQUEST_SUBMIT_{name}_OFFSET"], size, value)
    return bytes(payload)


def _response_payload(
    values: dict[str, int], opcode: str, *, failure_status: int = 0
) -> bytes:
    payload = bytearray(values["PAYLOAD_BYTES"])
    fields = (
        ("PAYLOAD_MAJOR", values["PAYLOAD_MAJOR"]),
        ("PAYLOAD_MINOR", values["PAYLOAD_MINOR"]),
        ("STATUS", failure_status),
        ("OPCODE", values[f"OPCODE_{opcode}"]),
        ("ERROR_CODE", failure_status),
    )
    for field, value in fields:
        _store(
            payload,
            values[f"RESPONSE_{field}_OFFSET"],
            values[f"RESPONSE_{field}_BYTES"],
            value,
        )
    if failure_status:
        return bytes(payload)

    common = (
        ("OBJECT_ID", 7),
        ("OBJECT_GENERATION", 9),
        ("MAPPING_ID", 11),
        ("MAPPING_GENERATION", 13),
    )
    for field, value in common:
        _store(
            payload,
            values[f"RESPONSE_{field}_OFFSET"],
            values[f"RESPONSE_{field}_BYTES"],
            value,
        )
    if opcode != "UNMAP_OBJECT":
        digest_offset = values["RESPONSE_IMAGE_SHA256_OFFSET"]
        payload[digest_offset : digest_offset + 32] = bytes(range(1, 33))
    if opcode == "MAP_OBJECT":
        response_fields = (
            ("MAPPED_BASE_VA", 0x100000000000),
            ("MAPPED_END_VA", 0x100000003000),
            ("DESCRIPTOR_VA", 0x1000000005C0),
            ("CODE_VA", 0x100000001600),
            ("ENTRY_VA", 0x100000001600),
            ("MAPPED_BYTES", 0x3000),
            ("SEGMENT_COUNT", 3),
            ("DESCRIPTOR_PRELOAD_DWORDS", 12),
        )
    elif opcode == "ALLOC_KERNARG":
        response_fields = (
            ("KERNARG_ALLOCATION_ID", 29),
            ("KERNARG_GENERATION", 31),
            ("KERNARG_VA", 0x100000004000),
            ("KERNARG_SIZE", 48),
            ("KERNARG_ALIGNMENT", 8),
        )
    elif opcode == "SUBMIT_AQL":
        response_fields = (
            ("KERNARG_ALLOCATION_ID", 29),
            ("KERNARG_GENERATION", 31),
            ("KERNARG_VA", 0x100000004000),
            ("KERNARG_SIZE", 48),
            ("KERNARG_ALIGNMENT", 8),
            ("TICKET_ID", 59),
            ("TRACE_ID", 67),
            ("QUEUE_ID", 17),
            ("QUEUE_GENERATION", 19),
            ("QUEUE_SEQUENCE", 23),
            ("SIGNAL_ID", 37),
            ("SIGNAL_GENERATION", 41),
            ("SIGNAL_VALUE_BITS", 1),
            ("PACKET_VA", 0x100000005000),
            ("PACKET_CRC32C", 0x12345678),
            ("ADMISSION_TICK", 100),
            ("START_TICK", 110),
            ("END_TICK", 120),
            ("RETIRE_TICK", 130),
        )
    else:
        response_fields = ()
    for field, value in response_fields:
        _store(
            payload,
            values[f"RESPONSE_{field}_OFFSET"],
            values[f"RESPONSE_{field}_BYTES"],
            value,
        )
    return bytes(payload)


def generated() -> dict[Path, str | bytes]:
    document = contract()
    kmt_document = kmt_contract()
    outputs: dict[Path, str | bytes] = {
        C_OUTPUT: render_c(document),
        CPP_OUTPUT: render_cpp(document),
        KMT_C_OUTPUT: render_kmt_c(kmt_document),
        KMT_CPP_OUTPUT: render_kmt_cpp(kmt_document),
    }
    vector_records = []
    for opcode in ("MAP_OBJECT", "ALLOC_KERNARG", "SUBMIT_AQL", "UNMAP_OBJECT"):
        content = _request_payload(document["values"], opcode)
        path = VECTOR_DIRECTORY / f"request-{opcode.lower().replace('_', '-')}.bin"
        outputs[path] = content
        vector_records.append(
            {
                "bytes": len(content),
                "direction": "request",
                "opcode": opcode,
                "path": path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    for opcode in ("MAP_OBJECT", "ALLOC_KERNARG", "SUBMIT_AQL", "UNMAP_OBJECT"):
        content = _response_payload(document["values"], opcode)
        suffix = opcode.lower().replace("_", "-")
        path = VECTOR_DIRECTORY / f"response-{suffix}-success.bin"
        outputs[path] = content
        vector_records.append(
            {
                "bytes": len(content),
                "direction": "response",
                "opcode": opcode,
                "path": path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "status": "OK",
            }
        )
    failure = _response_payload(
        document["values"],
        "MAP_OBJECT",
        failure_status=document["values"]["STATUS_UNSUPPORTED_VERSION"],
    )
    failure_path = VECTOR_DIRECTORY / "response-map-object-unsupported-version.bin"
    outputs[failure_path] = failure
    vector_records.append(
        {
            "bytes": len(failure),
            "direction": "response",
            "opcode": "MAP_OBJECT",
            "path": failure_path.name,
            "sha256": hashlib.sha256(failure).hexdigest(),
            "status": "UNSUPPORTED_VERSION",
        }
    )
    manifest = {
        "contract_sha256": _digest(document),
        "schema": "amdgpu-sim.runtime-gem5-bridge.generic-v2-vectors.v1",
        "vectors": vector_records,
    }
    outputs[VECTOR_DIRECTORY / "manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = generated()
    if args.write:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="ascii")
        return 0
    stale = []
    for path, content in outputs.items():
        if not path.is_file():
            stale.append(path)
        elif isinstance(content, bytes):
            if path.read_bytes() != content:
                stale.append(path)
        elif path.read_text(encoding="ascii") != content:
            stale.append(path)
    if stale:
        for path in stale:
            print(f"stale generated bridge contract: {path.relative_to(ROOT)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
