/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_CODE_OBJECT_H
#define SELF_AMDGPU_RUNTIME_CODE_OBJECT_H

#include <stddef.h>
#include <stdint.h>

#include <self_amdgpu_runtime/export.h>
#include <self_amdgpu_runtime/runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * This is a byte-oriented loader boundary.  It deliberately does not add a
 * transport message or copy code into the daemon.  A caller supplies an
 * already-owned ELF image and receives validated metadata which can later be
 * attached to an explicit code-object protocol.
 */
#define SAGR_CODE_OBJECT_AUTHORITY_SCHEMA \
  "amdgpu-sim.host-transport-v1.codeobj.v1"
#define SAGR_CODE_OBJECT_TARGET_GFX950 UINT32_C(950)
#define SAGR_CODE_OBJECT_ELF_MACHINE_AMDGPU UINT16_C(224)
#define SAGR_CODE_OBJECT_ELF_TYPE_DYN UINT16_C(3)
#define SAGR_CODE_OBJECT_ELF_OSABI_AMDGPU_HSA UINT8_C(64)
#define SAGR_CODE_OBJECT_ELF_ABI_VERSION UINT8_C(4)
#define SAGR_CODE_OBJECT_AMDHSA_NOTE_TYPE UINT32_C(32)
#define SAGR_CODE_OBJECT_DESCRIPTOR_BYTES UINT32_C(64)
#define SAGR_CODE_OBJECT_MAX_SEGMENTS UINT32_C(16)
#define SAGR_CODE_OBJECT_MAX_KERNELS UINT32_C(16)
#define SAGR_CODE_OBJECT_MAX_ARGS UINT32_C(64)
#define SAGR_CODE_OBJECT_NAME_BYTES UINT32_C(128)
#define SAGR_CODE_OBJECT_VALUE_KIND_BYTES UINT32_C(64)

typedef enum sagr_code_object_arg_kind {
  SAGR_CODE_OBJECT_ARG_VISIBLE = 0,
  SAGR_CODE_OBJECT_ARG_HIDDEN = 1
} sagr_code_object_arg_kind_t;

typedef struct sagr_code_object_arg_info {
  uint32_t struct_size;
  uint32_t index;
  uint32_t offset_bytes;
  uint32_t size_bytes;
  uint32_t kind;
  char name[SAGR_CODE_OBJECT_NAME_BYTES];
  char value_kind[SAGR_CODE_OBJECT_VALUE_KIND_BYTES];
} sagr_code_object_arg_info_t;

typedef struct sagr_code_object_kernel_info {
  uint32_t struct_size;
  uint32_t index;
  uint32_t arg_count;
  uint32_t visible_arg_count;
  uint32_t hidden_arg_count;
  uint32_t kernarg_segment_size;
  uint32_t kernarg_segment_align;
  uint32_t group_segment_fixed_size;
  uint32_t private_segment_fixed_size;
  uint32_t max_flat_workgroup_size;
  uint32_t wavefront_size;
  uint32_t sgpr_count;
  uint32_t vgpr_count;
  uint32_t uses_dynamic_stack;
  uint32_t relocation_count;
  uint32_t isa_supported_by_gemsim;
  uint32_t descriptor_group_segment_fixed_size;
  uint32_t descriptor_private_segment_fixed_size;
  uint32_t descriptor_kernarg_segment_size;
  uint32_t descriptor_compute_pgm_rsrc3;
  uint32_t descriptor_compute_pgm_rsrc1;
  uint32_t descriptor_compute_pgm_rsrc2;
  uint16_t descriptor_kernel_code_properties;
  uint16_t descriptor_kernarg_preload;
  int64_t descriptor_kernel_code_entry_byte_offset;
  uint64_t code_address;
  uint64_t code_file_offset;
  uint64_t code_size;
  uint64_t descriptor_address;
  uint64_t descriptor_file_offset;
  uint32_t descriptor_size;
  uint32_t reserved0;
  char name[SAGR_CODE_OBJECT_NAME_BYTES];
  char symbol[SAGR_CODE_OBJECT_NAME_BYTES];
  uint8_t descriptor[SAGR_CODE_OBJECT_DESCRIPTOR_BYTES];
  sagr_code_object_arg_info_t args[SAGR_CODE_OBJECT_MAX_ARGS];
} sagr_code_object_kernel_info_t;

typedef struct sagr_code_object_segment_info {
  uint32_t struct_size;
  uint32_t type;
  uint32_t flags;
  uint32_t reserved0;
  uint64_t file_offset;
  uint64_t virtual_address;
  uint64_t file_size;
  uint64_t memory_size;
  uint64_t alignment;
} sagr_code_object_segment_info_t;

typedef struct sagr_code_object_info {
  uint32_t struct_size;
  uint32_t flags;
  uint16_t elf_machine;
  uint16_t elf_type;
  uint8_t elf_osabi;
  uint8_t elf_abi_version;
  uint16_t reserved0;
  uint32_t elf_flags;
  uint32_t gfx_target;
  uint32_t code_object_version;
  uint32_t metadata_major;
  uint32_t metadata_minor;
  uint32_t kernel_count;
  uint32_t relocation_count;
  uint32_t isa_supported_by_gemsim;
  uint32_t reserved1;
  uint32_t segment_count;
  uint32_t reserved2;
  uint64_t note_file_offset;
  uint64_t note_file_size;
  uint64_t text_file_offset;
  uint64_t text_file_size;
  uint64_t text_address;
  uint64_t text_alignment;
  uint64_t rodata_file_offset;
  uint64_t rodata_file_size;
  char target[64];
  sagr_code_object_segment_info_t segments[SAGR_CODE_OBJECT_MAX_SEGMENTS];
  sagr_code_object_kernel_info_t kernels[SAGR_CODE_OBJECT_MAX_KERNELS];
} sagr_code_object_info_t;

/* Validate the complete ELF/AMDHSA metadata and fill a caller-sized result. */
SAGR_API sagr_status_t sagr_code_object_validate(
    const void *image, size_t image_size, sagr_code_object_info_t *info,
    uint32_t info_size);

SAGR_API sagr_status_t sagr_code_object_get_kernel(
    const sagr_code_object_info_t *info, const char *name,
    sagr_code_object_kernel_info_t *kernel, uint32_t kernel_size);

/* Pack values in metadata order.  Values are little-endian byte scalars. */
typedef struct sagr_code_object_arg_value {
  uint32_t struct_size;
  uint32_t arg_index;
  uint64_t value;
} sagr_code_object_arg_value_t;

SAGR_API sagr_status_t sagr_code_object_pack_kernarg(
    const sagr_code_object_kernel_info_t *kernel,
    const sagr_code_object_arg_value_t *values, uint32_t value_count,
    uint8_t *destination, uint32_t destination_size, uint32_t *written_size);

/* The current gem5 decoder has no gfx950-specific feature table. */
SAGR_API int sagr_code_object_gemsim_isa_supported(
    const sagr_code_object_info_t *info);

#ifdef __cplusplus
}
#endif

#endif
