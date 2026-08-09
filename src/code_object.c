/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <self_amdgpu_runtime/code_object.h>

#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* The loader intentionally has no ELF or MsgPack dependency.  Both formats
 * are bounded, little-endian inputs at this boundary and are decoded here so
 * a malformed image cannot make a host pointer or an unbounded allocation. */

typedef struct byte_cursor {
  const uint8_t *current;
  const uint8_t *end;
} byte_cursor_t;

typedef enum msg_kind {
  MSG_OTHER = 0,
  MSG_UNSIGNED = 1,
  MSG_SIGNED = 2,
  MSG_STRING = 3,
  MSG_MAP = 4,
  MSG_ARRAY = 5
} msg_kind_t;

typedef struct msg_header {
  msg_kind_t kind;
  uint64_t value;
  int64_t signed_value;
  const uint8_t *string;
  size_t string_size;
} msg_header_t;

typedef struct elf_section {
  uint32_t name;
  uint32_t type;
  uint64_t flags;
  uint64_t address;
  uint64_t offset;
  uint64_t size;
  uint32_t link;
  uint32_t info;
  uint64_t alignment;
  uint64_t entry_size;
} elf_section_t;

typedef struct elf_symbol {
  uint32_t name;
  uint8_t info;
  uint8_t other;
  uint16_t section;
  uint64_t value;
  uint64_t size;
} elf_symbol_t;

enum {
  ELF_PT_LOAD = 1,
  ELF_PF_X = 1,
  ELF_PF_W = 2,
  ELF_PF_R = 4,
  ELF_SHT_SYMTAB = 2,
  ELF_SHT_STRTAB = 3,
  ELF_SHT_RELA = 4,
  ELF_SHT_REL = 9,
  ELF_SHT_NOTE = 7,
  ELF_SHF_EXECINSTR = 4,
  ELF_STT_OBJECT = 1,
  ELF_STT_FUNC = 2,
  ELF_STB_GLOBAL = 1,
  ELF_STV_PROTECTED = 3
};

static uint16_t read_u16(const uint8_t *bytes) {
  return (uint16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8));
}

static uint32_t read_u32(const uint8_t *bytes) {
  return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
         ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static uint64_t read_u64(const uint8_t *bytes) {
  return (uint64_t)read_u32(bytes) | ((uint64_t)read_u32(bytes + 4) << 32);
}

static int cursor_take(byte_cursor_t *cursor, size_t size,
                       const uint8_t **result) {
  if (cursor == NULL || result == NULL || size > (size_t)(cursor->end - cursor->current)) {
    return 0;
  }
  *result = cursor->current;
  cursor->current += size;
  return 1;
}

static int msg_header_read(byte_cursor_t *cursor, msg_header_t *header) {
  const uint8_t *byte;
  uint8_t tag;
  uint64_t length;
  if (cursor == NULL || header == NULL || !cursor_take(cursor, 1U, &byte)) {
    return 0;
  }
  memset(header, 0, sizeof(*header));
  tag = byte[0];
  if (tag <= UINT8_C(0x7f)) {
    header->kind = MSG_UNSIGNED;
    header->value = tag;
    return 1;
  }
  if (tag >= UINT8_C(0xe0)) {
    header->kind = MSG_SIGNED;
    header->signed_value = (int8_t)tag;
    return 1;
  }
  if (tag >= UINT8_C(0x80) && tag <= UINT8_C(0x8f)) {
    header->kind = MSG_MAP;
    header->value = (uint64_t)(tag & UINT8_C(0x0f));
    return 1;
  }
  if (tag >= UINT8_C(0x90) && tag <= UINT8_C(0x9f)) {
    header->kind = MSG_ARRAY;
    header->value = (uint64_t)(tag & UINT8_C(0x0f));
    return 1;
  }
  if (tag >= UINT8_C(0xa0) && tag <= UINT8_C(0xbf)) {
    length = (uint64_t)(tag & UINT8_C(0x1f));
    header->kind = MSG_STRING;
    header->string_size = (size_t)length;
    return cursor_take(cursor, (size_t)length, &header->string);
  }
  switch (tag) {
    case 0xc0:
      return 1;
    case 0xc2:
    case 0xc3:
      header->kind = MSG_UNSIGNED;
      header->value = (uint64_t)(tag - UINT8_C(0xc2));
      return 1;
    case 0xcc:
      if (!cursor_take(cursor, 1U, &byte)) return 0;
      header->kind = MSG_UNSIGNED; header->value = byte[0]; return 1;
    case 0xcd:
      if (!cursor_take(cursor, 2U, &byte)) return 0;
      header->kind = MSG_UNSIGNED; header->value =
          ((uint64_t)byte[0] << 8) | (uint64_t)byte[1]; return 1;
    case 0xce:
      if (!cursor_take(cursor, 4U, &byte)) return 0;
      header->kind = MSG_UNSIGNED; header->value =
          ((uint64_t)byte[0] << 24) | ((uint64_t)byte[1] << 16) |
          ((uint64_t)byte[2] << 8) | (uint64_t)byte[3]; return 1;
    case 0xcf:
      if (!cursor_take(cursor, 8U, &byte)) return 0;
      header->kind = MSG_UNSIGNED;
      header->value = ((uint64_t)byte[0] << 56) |
                      ((uint64_t)byte[1] << 48) |
                      ((uint64_t)byte[2] << 40) |
                      ((uint64_t)byte[3] << 32) |
                      ((uint64_t)byte[4] << 24) |
                      ((uint64_t)byte[5] << 16) |
                      ((uint64_t)byte[6] << 8) | (uint64_t)byte[7];
      return 1;
    case 0xd0:
      if (!cursor_take(cursor, 1U, &byte)) return 0;
      header->kind = MSG_SIGNED; header->signed_value = (int8_t)byte[0]; return 1;
    case 0xd1:
      if (!cursor_take(cursor, 2U, &byte)) return 0;
      header->kind = MSG_SIGNED;
      header->signed_value = (int16_t)(((uint16_t)byte[0] << 8) | byte[1]);
      return 1;
    case 0xd2:
      if (!cursor_take(cursor, 4U, &byte)) return 0;
      header->kind = MSG_SIGNED;
      header->signed_value = (int32_t)(((uint32_t)byte[0] << 24) |
                                       ((uint32_t)byte[1] << 16) |
                                       ((uint32_t)byte[2] << 8) | byte[3]);
      return 1;
    case 0xd3:
      if (!cursor_take(cursor, 8U, &byte)) return 0;
      header->kind = MSG_SIGNED;
      header->signed_value = (int64_t)(((uint64_t)byte[0] << 56) |
                                       ((uint64_t)byte[1] << 48) |
                                       ((uint64_t)byte[2] << 40) |
                                       ((uint64_t)byte[3] << 32) |
                                       ((uint64_t)byte[4] << 24) |
                                       ((uint64_t)byte[5] << 16) |
                                       ((uint64_t)byte[6] << 8) | byte[7]);
      return 1;
    case 0xd9:
      if (!cursor_take(cursor, 1U, &byte)) return 0;
      length = byte[0];
      break;
    case 0xda:
      if (!cursor_take(cursor, 2U, &byte)) return 0;
      length = ((uint64_t)byte[0] << 8) | byte[1];
      break;
    case 0xdb:
      if (!cursor_take(cursor, 4U, &byte)) return 0;
      length = ((uint64_t)byte[0] << 24) | ((uint64_t)byte[1] << 16) |
               ((uint64_t)byte[2] << 8) | byte[3];
      break;
    case 0xdc:
      if (!cursor_take(cursor, 2U, &byte)) return 0;
      header->kind = MSG_ARRAY;
      header->value = ((uint64_t)byte[0] << 8) | byte[1];
      return 1;
    case 0xdd:
      if (!cursor_take(cursor, 4U, &byte)) return 0;
      header->kind = MSG_ARRAY;
      header->value = ((uint64_t)byte[0] << 24) | ((uint64_t)byte[1] << 16) |
                      ((uint64_t)byte[2] << 8) | byte[3];
      return 1;
    case 0xde:
      if (!cursor_take(cursor, 2U, &byte)) return 0;
      header->kind = MSG_MAP;
      header->value = ((uint64_t)byte[0] << 8) | byte[1];
      return 1;
    case 0xdf:
      if (!cursor_take(cursor, 4U, &byte)) return 0;
      header->kind = MSG_MAP;
      header->value = ((uint64_t)byte[0] << 24) | ((uint64_t)byte[1] << 16) |
                      ((uint64_t)byte[2] << 8) | byte[3];
      return 1;
    default:
      break;
  }
  if (tag == UINT8_C(0xc4) || tag == UINT8_C(0xc5) || tag == UINT8_C(0xc6)) {
    size_t width = tag == UINT8_C(0xc4) ? 1U : (tag == UINT8_C(0xc5) ? 2U : 4U);
    if (!cursor_take(cursor, width, &byte)) return 0;
    length = width == 1U ? byte[0] : (width == 2U
        ? (((uint64_t)byte[0] << 8) | byte[1])
        : (((uint64_t)byte[0] << 24) | ((uint64_t)byte[1] << 16) |
           ((uint64_t)byte[2] << 8) | byte[3]));
    if (length > SIZE_MAX) return 0;
    return cursor_take(cursor, (size_t)length, &header->string);
  }
  /* Floats and extension/bin values are opaque to metadata parsing. */
  if ((tag >= UINT8_C(0xca) && tag <= UINT8_C(0xd8)) ||
      tag == UINT8_C(0xc1)) {
    size_t width = 0U;
    if (tag == UINT8_C(0xca)) width = 4U;
    else if (tag == UINT8_C(0xcb)) width = 8U;
    else if (tag == UINT8_C(0xd4)) width = 2U;
    else if (tag == UINT8_C(0xd5)) width = 3U;
    else if (tag == UINT8_C(0xd6)) width = 5U;
    else if (tag == UINT8_C(0xd7)) width = 9U;
    else if (tag == UINT8_C(0xd8)) width = 17U;
    else return 0;
    return cursor_take(cursor, width, &header->string);
  }
  if (tag >= UINT8_C(0xc7) && tag <= UINT8_C(0xc9)) {
    size_t width = tag == UINT8_C(0xc7) ? 1U : (tag == UINT8_C(0xc8) ? 2U : 4U);
    if (!cursor_take(cursor, width, &byte)) return 0;
    length = width == 1U ? byte[0] : width == 2U
        ? (((uint64_t)byte[0] << 8) | byte[1])
        : (((uint64_t)byte[0] << 24) | ((uint64_t)byte[1] << 16) |
           ((uint64_t)byte[2] << 8) | byte[3]);
    if (!cursor_take(cursor, 1U, &byte) || length > SIZE_MAX - 1U) return 0;
    return cursor_take(cursor, (size_t)length, &header->string);
  }
  header->kind = MSG_STRING;
  header->string_size = (size_t)length;
  return cursor_take(cursor, (size_t)length, &header->string);
}

static int msg_skip(byte_cursor_t *cursor, unsigned depth);

static int msg_skip_body(byte_cursor_t *cursor, const msg_header_t *header,
                         unsigned depth) {
  uint64_t index;
  if (cursor == NULL || header == NULL || depth > 64U) return 0;
  if (header->kind == MSG_ARRAY) {
    for (index = 0; index < header->value; ++index) {
      if (!msg_skip(cursor, depth + 1U)) return 0;
    }
  } else if (header->kind == MSG_MAP) {
    for (index = 0; index < header->value; ++index) {
      if (!msg_skip(cursor, depth + 1U) || !msg_skip(cursor, depth + 1U)) return 0;
    }
  }
  return 1;
}

static int msg_skip(byte_cursor_t *cursor, unsigned depth) {
  msg_header_t header;
  if (depth > 64U || !msg_header_read(cursor, &header)) return 0;
  return msg_skip_body(cursor, &header, depth);
}

static int msg_string_is(const msg_header_t *header, const char *text) {
  size_t length = strlen(text);
  return header != NULL && header->kind == MSG_STRING &&
         header->string_size == length &&
         memcmp(header->string, text, length) == 0;
}

static int msg_number(const msg_header_t *header, uint64_t *value) {
  if (header == NULL || value == NULL) return 0;
  if (header->kind == MSG_UNSIGNED) {
    *value = header->value;
    return 1;
  }
  if (header->kind == MSG_SIGNED && header->signed_value >= 0) {
    *value = (uint64_t)header->signed_value;
    return 1;
  }
  return 0;
}

static int copy_string(char *destination, size_t capacity,
                       const msg_header_t *header) {
  if (destination == NULL || capacity == 0U || header == NULL ||
      header->kind != MSG_STRING || header->string_size >= capacity) return 0;
  memcpy(destination, header->string, header->string_size);
  destination[header->string_size] = '\0';
  return 1;
}

static int parse_arg_map(byte_cursor_t *cursor, sagr_code_object_arg_info_t *arg,
                         uint32_t index) {
  msg_header_t map, key, value;
  uint64_t pair;
  int have_offset = 0;
  int have_size = 0;
  int have_kind = 0;
  if (!msg_header_read(cursor, &map) || map.kind != MSG_MAP || arg == NULL) return 0;
  memset(arg, 0, sizeof(*arg));
  arg->struct_size = (uint32_t)sizeof(*arg);
  arg->index = index;
  for (pair = 0; pair < map.value; ++pair) {
    if (!msg_header_read(cursor, &key) || key.kind != MSG_STRING ||
        !msg_header_read(cursor, &value)) return 0;
    if (msg_string_is(&key, ".offset")) {
      uint64_t number; if (!msg_number(&value, &number) || number > UINT32_MAX) return 0;
      arg->offset_bytes = (uint32_t)number; have_offset = 1;
    } else if (msg_string_is(&key, ".size")) {
      uint64_t number; if (!msg_number(&value, &number) || number > UINT32_MAX) return 0;
      arg->size_bytes = (uint32_t)number; have_size = 1;
    } else if (msg_string_is(&key, ".name")) {
      if (!copy_string(arg->name, sizeof(arg->name), &value)) return 0;
    } else if (msg_string_is(&key, ".value_kind")) {
      if (!copy_string(arg->value_kind, sizeof(arg->value_kind), &value)) return 0;
      have_kind = 1;
    } else {
      /* value has already been consumed for scalar/string values.  For an
       * array/map, msg_header_read consumed only its header. */
      if ((value.kind == MSG_ARRAY || value.kind == MSG_MAP) &&
          !msg_skip_body(cursor, &value, 0U)) return 0;
    }
  }
  if (!have_offset || !have_size || !have_kind || arg->size_bytes == 0U) return 0;
  arg->kind = strncmp(arg->value_kind, "hidden_", 7U) == 0
                  ? SAGR_CODE_OBJECT_ARG_HIDDEN
                  : SAGR_CODE_OBJECT_ARG_VISIBLE;
  return 1;
}

static int parse_kernel_map(byte_cursor_t *cursor,
                            sagr_code_object_kernel_info_t *kernel,
                            uint32_t index) {
  msg_header_t map, key, value;
  uint64_t pair;
  int have_name = 0;
  int have_symbol = 0;
  int have_args = 0;
  int have_kernarg_size = 0;
  int have_kernarg_align = 0;
  int have_group_size = 0;
  int have_private_size = 0;
  int have_max_workgroup = 0;
  int have_wavefront = 0;
  int have_sgpr = 0;
  int have_vgpr = 0;
  if (!msg_header_read(cursor, &map) || map.kind != MSG_MAP || kernel == NULL) return 0;
  memset(kernel, 0, sizeof(*kernel));
  kernel->struct_size = (uint32_t)sizeof(*kernel);
  kernel->index = index;
  for (pair = 0; pair < map.value; ++pair) {
    if (!msg_header_read(cursor, &key) || key.kind != MSG_STRING ||
        !msg_header_read(cursor, &value)) return 0;
    if (msg_string_is(&key, ".name")) {
      if (!copy_string(kernel->name, sizeof(kernel->name), &value)) return 0;
      have_name = 1;
    } else if (msg_string_is(&key, ".symbol")) {
      if (!copy_string(kernel->symbol, sizeof(kernel->symbol), &value)) return 0;
      have_symbol = 1;
    } else if (msg_string_is(&key, ".args")) {
      uint64_t count, arg_index;
      if (value.kind != MSG_ARRAY || value.value > SAGR_CODE_OBJECT_MAX_ARGS) return 0;
      count = value.value;
      for (arg_index = 0; arg_index < count; ++arg_index) {
        if (!parse_arg_map(cursor, &kernel->args[arg_index], (uint32_t)arg_index)) return 0;
      }
      kernel->arg_count = (uint32_t)count;
      have_args = 1;
    } else {
      uint64_t number;
      if (msg_string_is(&key, ".kernarg_segment_size")) {
        if (!msg_number(&value, &number) || number > UINT32_MAX) return 0;
        kernel->kernarg_segment_size = (uint32_t)number;
        have_kernarg_size = 1;
      } else if (msg_string_is(&key, ".kernarg_segment_align")) {
        if (!msg_number(&value, &number) || number > UINT32_MAX) return 0;
        kernel->kernarg_segment_align = (uint32_t)number;
        have_kernarg_align = 1;
      } else if (msg_string_is(&key, ".group_segment_fixed_size")) {
        if (!msg_number(&value, &number) || number > UINT32_MAX) return 0;
        kernel->group_segment_fixed_size = (uint32_t)number;
        have_group_size = 1;
      } else if (msg_string_is(&key, ".private_segment_fixed_size")) {
        if (!msg_number(&value, &number) || number > UINT32_MAX) return 0;
        kernel->private_segment_fixed_size = (uint32_t)number;
        have_private_size = 1;
      } else if (msg_string_is(&key, ".max_flat_workgroup_size")) {
        if (!msg_number(&value, &number) || number > UINT32_MAX) return 0;
        kernel->max_flat_workgroup_size = (uint32_t)number;
        have_max_workgroup = 1;
      } else if (msg_string_is(&key, ".wavefront_size")) {
        if (!msg_number(&value, &number) || number > UINT32_MAX) return 0;
        kernel->wavefront_size = (uint32_t)number;
        have_wavefront = 1;
      } else if (msg_string_is(&key, ".sgpr_count")) {
        if (!msg_number(&value, &number) || number > UINT32_MAX) return 0;
        kernel->sgpr_count = (uint32_t)number;
        have_sgpr = 1;
      } else if (msg_string_is(&key, ".vgpr_count")) {
        if (!msg_number(&value, &number) || number > UINT32_MAX) return 0;
        kernel->vgpr_count = (uint32_t)number;
        have_vgpr = 1;
      } else if (msg_string_is(&key, ".uses_dynamic_stack")) {
        if (value.kind == MSG_UNSIGNED) kernel->uses_dynamic_stack = value.value != 0U ? 1U : 0U;
        else if (value.kind != MSG_OTHER) return 0;
      } else if ((value.kind == MSG_ARRAY || value.kind == MSG_MAP) &&
                 !msg_skip_body(cursor, &value, 0U)) {
        return 0;
      }
    }
  }
  return have_name && have_symbol && have_args && have_kernarg_size &&
         have_kernarg_align && have_group_size && have_private_size &&
         have_max_workgroup && have_wavefront && have_sgpr && have_vgpr;
}

static int parse_metadata(const uint8_t *bytes, size_t size,
                          sagr_code_object_info_t *info) {
  byte_cursor_t cursor = {bytes, bytes + size};
  msg_header_t root, key, value;
  uint64_t pair;
  int have_target = 0;
  int have_version = 0;
  int have_kernels = 0;
  if (!msg_header_read(&cursor, &root) || root.kind != MSG_MAP) return 0;
  for (pair = 0; pair < root.value; ++pair) {
    if (!msg_header_read(&cursor, &key) || key.kind != MSG_STRING ||
        !msg_header_read(&cursor, &value)) return 0;
    if (msg_string_is(&key, "amdhsa.target")) {
      if (!copy_string(info->target, sizeof(info->target), &value)) return 0;
      have_target = 1;
    } else if (msg_string_is(&key, "amdhsa.version")) {
      uint64_t count, first, second;
      if (value.kind != MSG_ARRAY || value.value != 2U) return 0;
      count = value.value;
      (void)count;
      if (!msg_header_read(&cursor, &key) || !msg_number(&key, &first) ||
          !msg_header_read(&cursor, &key) || !msg_number(&key, &second) ||
          first > UINT32_MAX || second > UINT32_MAX) return 0;
      info->metadata_major = (uint32_t)first;
      info->metadata_minor = (uint32_t)second;
      have_version = 1;
    } else if (msg_string_is(&key, "amdhsa.kernels")) {
      uint64_t count, kernel_index;
      if (value.kind != MSG_ARRAY || value.value == 0U ||
          value.value > SAGR_CODE_OBJECT_MAX_KERNELS) return 0;
      count = value.value;
      for (kernel_index = 0; kernel_index < count; ++kernel_index) {
        if (!parse_kernel_map(&cursor, &info->kernels[kernel_index],
                              (uint32_t)kernel_index)) return 0;
      }
      info->kernel_count = (uint32_t)count;
      have_kernels = 1;
    } else if ((value.kind == MSG_ARRAY || value.kind == MSG_MAP) &&
               !msg_skip_body(&cursor, &value, 0U)) {
      return 0;
    }
  }
  return have_target && have_version && have_kernels && cursor.current == cursor.end;
}

static int section_in_bounds(const elf_section_t *section, size_t size) {
  return section != NULL && section->offset <= size && section->size <=
         (uint64_t)(size - (size_t)section->offset);
}

static int section_in_load_segment(const elf_section_t *section,
                                   const sagr_code_object_segment_info_t *segment,
                                   uint32_t required_flags) {
  uint64_t section_end;
  uint64_t segment_end;
  if (section == NULL || segment == NULL || section->offset > UINT64_MAX - section->size ||
      segment->file_offset > UINT64_MAX - segment->file_size) return 0;
  section_end = section->offset + section->size;
  segment_end = segment->file_offset + segment->file_size;
  return section->offset >= segment->file_offset && section_end <= segment_end &&
         (segment->flags & required_flags) == required_flags;
}

static int read_section(const uint8_t *image, size_t image_size,
                        uint64_t table_offset, uint16_t count,
                        uint16_t entry_size, uint16_t index,
                        elf_section_t *section) {
  const uint8_t *raw;
  uint64_t offset;
  if (section == NULL || index >= count || entry_size < 64U ||
      (uint64_t)index > (UINT64_MAX - table_offset) / entry_size) return 0;
  offset = table_offset + (uint64_t)index * entry_size;
  if (offset > image_size || entry_size > image_size - (size_t)offset ||
      !cursor_take(&(byte_cursor_t){image + (size_t)offset,
                                    image + image_size}, 64U, &raw)) return 0;
  section->name = read_u32(raw + 0);
  section->type = read_u32(raw + 4);
  section->flags = read_u64(raw + 8);
  section->address = read_u64(raw + 16);
  section->offset = read_u64(raw + 24);
  section->size = read_u64(raw + 32);
  section->link = read_u32(raw + 40);
  section->info = read_u32(raw + 44);
  section->alignment = read_u64(raw + 48);
  section->entry_size = read_u64(raw + 56);
  return section_in_bounds(section, image_size) || section->type == 8;
}

static const char *section_name(const uint8_t *image, size_t image_size,
                                const elf_section_t *strings, uint32_t index) {
  size_t offset;
  const uint8_t *start;
  if (strings == NULL || !section_in_bounds(strings, image_size) ||
      index >= strings->size || strings->offset > image_size) return NULL;
  offset = (size_t)strings->offset + (size_t)index;
  start = image + offset;
  if (memchr(start, '\0', (size_t)strings->size - (size_t)index) == NULL) return NULL;
  return (const char *)start;
}

static int find_symbol(const uint8_t *image, size_t image_size,
                       const elf_section_t *symtab, const elf_section_t *strtab,
                       const char *name, elf_symbol_t *symbol, uint16_t *section_index) {
  uint64_t offset;
  uint64_t count;
  uint64_t index;
  if (symtab == NULL || strtab == NULL || symbol == NULL || section_index == NULL ||
      symtab->entry_size < 24U || !section_in_bounds(symtab, image_size) ||
      !section_in_bounds(strtab, image_size)) return 0;
  count = symtab->size / symtab->entry_size;
  for (index = 0; index < count; ++index) {
    const uint8_t *raw;
    const char *symbol_name;
    offset = symtab->offset + index * symtab->entry_size;
    if (offset > image_size || 24U > image_size - (size_t)offset) return 0;
    raw = image + (size_t)offset;
    symbol_name = section_name(image, image_size, strtab, read_u32(raw));
    if (symbol_name != NULL && strcmp(symbol_name, name) == 0) {
      symbol->name = read_u32(raw);
      symbol->info = raw[4];
      symbol->other = raw[5];
      *section_index = read_u16(raw + 6);
      symbol->value = read_u64(raw + 8);
      symbol->size = read_u64(raw + 16);
      return 1;
    }
  }
  return 0;
}

static int locate_note(const uint8_t *image, size_t image_size,
                       const elf_section_t *note, uint64_t *metadata_offset,
                       uint64_t *metadata_size) {
  uint64_t offset = note->offset;
  const uint64_t end = note->offset + note->size;
  while (offset + 12U <= end) {
    const uint8_t *raw = image + (size_t)offset;
    uint32_t namesz = read_u32(raw);
    uint32_t descsz = read_u32(raw + 4);
    uint32_t type = read_u32(raw + 8);
    uint64_t name_end = offset + 12U +
                        (((uint64_t)namesz + UINT64_C(3)) & ~UINT64_C(3));
    uint64_t desc_end = name_end +
                        (((uint64_t)descsz + UINT64_C(3)) & ~UINT64_C(3));
    if (name_end > end || desc_end > end || desc_end > image_size) return 0;
    if (type == SAGR_CODE_OBJECT_AMDHSA_NOTE_TYPE && namesz >= 6U &&
        memcmp(image + (size_t)(offset + 12U), "AMDGPU", 6U) == 0) {
      *metadata_offset = name_end;
      *metadata_size = descsz;
      return 1;
    }
    offset = desc_end;
  }
  return 0;
}

static int descriptor_valid(const uint8_t *descriptor, size_t size) {
  if (descriptor == NULL || size != SAGR_CODE_OBJECT_DESCRIPTOR_BYTES) return 0;
  /* The descriptor is little-endian and exactly 64 bytes.  The reserved
   * regions are intentionally copied but not interpreted as host pointers. */
  return read_u32(descriptor + 8U) != 0U && read_u32(descriptor + 48U) != 0U &&
         read_u32(descriptor + 52U) != 0U && read_u16(descriptor + 56U) != UINT16_MAX;
}

static int validate_kernel_layout(sagr_code_object_kernel_info_t *kernel) {
  uint32_t index;
  uint32_t other;
  if (kernel->kernarg_segment_size == 0U || kernel->kernarg_segment_align == 0U ||
      (kernel->kernarg_segment_align & (kernel->kernarg_segment_align - 1U)) != 0U ||
      kernel->wavefront_size != 64U || kernel->max_flat_workgroup_size == 0U ||
      kernel->arg_count == 0U) return 0;
  kernel->visible_arg_count = 0U;
  kernel->hidden_arg_count = 0U;
  for (index = 0; index < kernel->arg_count; ++index) {
    sagr_code_object_arg_info_t *arg = &kernel->args[index];
    if (arg->offset_bytes > kernel->kernarg_segment_size ||
        arg->size_bytes > kernel->kernarg_segment_size - arg->offset_bytes) return 0;
    if (arg->kind == SAGR_CODE_OBJECT_ARG_HIDDEN) kernel->hidden_arg_count++;
    else kernel->visible_arg_count++;
    for (other = 0; other < index; ++other) {
      const sagr_code_object_arg_info_t *prior = &kernel->args[other];
      if (arg->offset_bytes < prior->offset_bytes + prior->size_bytes &&
          prior->offset_bytes < arg->offset_bytes + arg->size_bytes) return 0;
    }
  }
  return 1;
}

static int copy_section_info(size_t image_size,
                             const elf_section_t *section,
                             uint64_t *offset, uint64_t *size,
                             uint64_t *address, uint64_t *alignment) {
  if (section == NULL || !section_in_bounds(section, image_size)) return 0;
  *offset = section->offset;
  *size = section->size;
  *address = section->address;
  *alignment = section->alignment;
  return 1;
}

sagr_status_t sagr_code_object_validate(const void *image, size_t image_size,
                                         sagr_code_object_info_t *info,
                                         uint32_t info_size) {
  const uint8_t *bytes = (const uint8_t *)image;
  uint64_t section_offset;
  uint16_t section_count;
  uint16_t section_entry_size;
  uint16_t section_names_index;
  uint64_t program_offset;
  uint16_t program_count;
  uint16_t program_entry_size;
  elf_section_t section_names;
  elf_section_t note = {0};
  elf_section_t text = {0};
  elf_section_t rodata = {0};
  elf_section_t symtab = {0};
  elf_section_t strtab = {0};
  int have_note = 0;
  int have_text = 0;
  int have_rodata = 0;
  int have_symtab = 0;
  int have_strtab = 0;
  uint16_t index;
  uint64_t metadata_offset;
  uint64_t metadata_size;
  uint32_t kernel_index;
  if (image == NULL || info == NULL) return SAGR_STATUS_INVALID_ARGUMENT;
  if (info_size < (uint32_t)sizeof(*info)) return SAGR_STATUS_BUFFER_TOO_SMALL;
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  if (image_size < 64U || bytes[0] != 0x7fU || bytes[1] != 'E' ||
      bytes[2] != 'L' || bytes[3] != 'F' || bytes[4] != 2U || bytes[5] != 1U ||
      bytes[6] != 1U || bytes[7] != SAGR_CODE_OBJECT_ELF_OSABI_AMDGPU_HSA)
    return SAGR_STATUS_PROTOCOL_ERROR;
  if (bytes[8] < 2U || bytes[8] > 4U) return SAGR_STATUS_NOT_SUPPORTED;
  if (read_u16(bytes + 16U) != SAGR_CODE_OBJECT_ELF_TYPE_DYN ||
      read_u16(bytes + 18U) != SAGR_CODE_OBJECT_ELF_MACHINE_AMDGPU) return SAGR_STATUS_NOT_SUPPORTED;
  info->elf_type = read_u16(bytes + 16U);
  info->elf_machine = read_u16(bytes + 18U);
  info->elf_osabi = bytes[7];
  info->elf_abi_version = bytes[8];
  info->elf_flags = read_u32(bytes + 48U);
  if ((info->elf_flags & UINT32_C(0xff)) != UINT32_C(0x4f)) return SAGR_STATUS_NOT_SUPPORTED;
  info->gfx_target = SAGR_CODE_OBJECT_TARGET_GFX950;
  info->code_object_version = (uint32_t)bytes[8] + 2U;
  program_offset = read_u64(bytes + 32U);
  program_entry_size = read_u16(bytes + 54U);
  program_count = read_u16(bytes + 56U);
  if (program_entry_size < 56U ||
      program_offset > image_size ||
      (uint64_t)program_count * program_entry_size >
          image_size - (size_t)program_offset) return SAGR_STATUS_PROTOCOL_ERROR;
  for (index = 0; index < program_count; ++index) {
    const uint8_t *raw = bytes + (size_t)(program_offset +
                                         (uint64_t)index * program_entry_size);
    const uint32_t type = read_u32(raw + 0U);
    const uint32_t flags = read_u32(raw + 4U);
    const uint64_t file_offset = read_u64(raw + 8U);
    const uint64_t virtual_address = read_u64(raw + 16U);
    const uint64_t file_size = read_u64(raw + 32U);
    const uint64_t memory_size = read_u64(raw + 40U);
    const uint64_t alignment = read_u64(raw + 48U);
    if (type != ELF_PT_LOAD) continue;
    if (info->segment_count >= SAGR_CODE_OBJECT_MAX_SEGMENTS ||
        file_size > memory_size || file_offset > image_size ||
        file_size > image_size - (size_t)file_offset ||
        (alignment != 0U && (alignment & (alignment - 1U)) != 0U))
      return SAGR_STATUS_PROTOCOL_ERROR;
    info->segments[info->segment_count].struct_size =
        (uint32_t)sizeof(info->segments[info->segment_count]);
    info->segments[info->segment_count].type = type;
    info->segments[info->segment_count].flags = flags;
    info->segments[info->segment_count].file_offset = file_offset;
    info->segments[info->segment_count].virtual_address = virtual_address;
    info->segments[info->segment_count].file_size = file_size;
    info->segments[info->segment_count].memory_size = memory_size;
    info->segments[info->segment_count].alignment = alignment;
    info->segment_count++;
  }
  if (info->segment_count == 0U) return SAGR_STATUS_PROTOCOL_ERROR;
  section_offset = read_u64(bytes + 40U);
  section_entry_size = read_u16(bytes + 58U);
  section_count = read_u16(bytes + 60U);
  section_names_index = read_u16(bytes + 62U);
  if (section_entry_size < 64U || section_count == 0U ||
      section_names_index >= section_count ||
      section_offset > image_size ||
      (uint64_t)section_count * section_entry_size > image_size - (size_t)section_offset ||
      !read_section(bytes, image_size, section_offset, section_count,
                    section_entry_size, section_names_index, &section_names)) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  for (index = 0; index < section_count; ++index) {
    elf_section_t section;
    const char *name;
    if (!read_section(bytes, image_size, section_offset, section_count,
                      section_entry_size, index, &section) ||
        (section.type != 8U && !section_in_bounds(&section, image_size))) return SAGR_STATUS_PROTOCOL_ERROR;
    name = section_name(bytes, image_size, &section_names, section.name);
    if (name == NULL) return SAGR_STATUS_PROTOCOL_ERROR;
    if (strcmp(name, ".note") == 0 && section.type == ELF_SHT_NOTE) { note = section; have_note = 1; }
    else if (strcmp(name, ".text") == 0 && (section.flags & ELF_SHF_EXECINSTR) != 0U) { text = section; have_text = 1; }
    else if (strcmp(name, ".rodata") == 0) { rodata = section; have_rodata = 1; }
    else if (strcmp(name, ".symtab") == 0 && section.type == ELF_SHT_SYMTAB) { symtab = section; have_symtab = 1; }
    else if (strcmp(name, ".strtab") == 0 && section.type == ELF_SHT_STRTAB) { strtab = section; have_strtab = 1; }
    if (section.type == ELF_SHT_RELA || section.type == ELF_SHT_REL) {
      if (section.entry_size == 0U || section.size % section.entry_size != 0U ||
          section.size / section.entry_size > UINT32_MAX - info->relocation_count) return SAGR_STATUS_PROTOCOL_ERROR;
      info->relocation_count += (uint32_t)(section.size / section.entry_size);
    }
  }
  if (!have_note || !have_text || !have_rodata || !have_symtab || !have_strtab ||
      !locate_note(bytes, image_size, &note, &metadata_offset, &metadata_size) ||
      metadata_offset > image_size || metadata_size > image_size - (size_t)metadata_offset ||
      !copy_section_info(image_size, &text, &info->text_file_offset,
                         &info->text_file_size, &info->text_address,
                         &info->text_alignment) ||
      !section_in_bounds(&rodata, image_size)) return SAGR_STATUS_PROTOCOL_ERROR;
  {
    uint32_t segment_index;
    int text_segment = 0;
    int rodata_segment = 0;
    for (segment_index = 0; segment_index < info->segment_count; ++segment_index) {
      const sagr_code_object_segment_info_t *segment = &info->segments[segment_index];
      if (section_in_load_segment(&text, segment, ELF_PF_R | ELF_PF_X)) text_segment = 1;
      if (section_in_load_segment(&rodata, segment, ELF_PF_R) &&
          (segment->flags & ELF_PF_W) == 0U) rodata_segment = 1;
    }
    if (!text_segment || !rodata_segment) return SAGR_STATUS_PROTOCOL_ERROR;
  }
  info->note_file_offset = note.offset;
  info->note_file_size = note.size;
  info->rodata_file_offset = rodata.offset;
  info->rodata_file_size = rodata.size;
  if (info->text_alignment == 0U || (info->text_alignment & (info->text_alignment - 1U)) != 0U ||
      !parse_metadata(bytes + (size_t)metadata_offset, (size_t)metadata_size, info) ||
      info->metadata_major != 1U || info->kernel_count == 0U) return SAGR_STATUS_PROTOCOL_ERROR;
  if ((bytes[8] == 2U && info->metadata_minor != 1U) ||
      (bytes[8] >= 3U && info->metadata_minor != 2U))
    return SAGR_STATUS_NOT_SUPPORTED;
  /*
   * LLVM's AMD backend emits the explicit gfx target for the pinned RDC
   * fixtures, while the pinned Triton overlay emits the ABI-equivalent
   * ``-unknown-gfx950`` spelling.  Both identify gfx950 here; neither implies
   * that this runtime can execute the image.
   */
  if (strcmp(info->target, "amdgcn-amd-amdhsa--gfx950") != 0 &&
      strcmp(info->target, "amdgcn-amd-amdhsa-unknown-gfx950") != 0)
    return SAGR_STATUS_NOT_SUPPORTED;
  const int triton_target =
      strcmp(info->target, "amdgcn-amd-amdhsa-unknown-gfx950") == 0;
  for (kernel_index = 0; kernel_index < info->kernel_count; ++kernel_index) {
    sagr_code_object_kernel_info_t *kernel = &info->kernels[kernel_index];
    elf_symbol_t code_symbol;
    elf_symbol_t descriptor_symbol;
    uint16_t code_section_index = 0;
    uint16_t descriptor_section_index = 0;
    if (kernel->name[0] == '\0' || kernel->symbol[0] == '\0' ||
        !validate_kernel_layout(kernel))
      return SAGR_STATUS_PROTOCOL_ERROR;
    /* Triton's linker leaves the descriptor symbol DEFAULT-visible; the
     * legacy RDC fixtures use PROTECTED.  This exception stays metadata-only
     * and does not mark the Triton ISA executable. */
    if (!find_symbol(bytes, image_size, &symtab, &strtab, kernel->name,
                     &code_symbol, &code_section_index) ||
        !find_symbol(bytes, image_size, &symtab, &strtab, kernel->symbol,
                     &descriptor_symbol, &descriptor_section_index) ||
        code_section_index >= section_count || descriptor_section_index >= section_count ||
        (code_symbol.info & UINT8_C(0x0f)) != ELF_STT_FUNC ||
        (code_symbol.info >> 4) != ELF_STB_GLOBAL ||
        (code_symbol.other & UINT8_C(0x03)) != ELF_STV_PROTECTED ||
        (descriptor_symbol.info & UINT8_C(0x0f)) != ELF_STT_OBJECT ||
        (descriptor_symbol.info >> 4) != ELF_STB_GLOBAL ||
        ((descriptor_symbol.other & UINT8_C(0x03)) != ELF_STV_PROTECTED &&
         !(triton_target &&
           (descriptor_symbol.other & UINT8_C(0x03)) == 0U)) ||
        code_symbol.size == 0U || descriptor_symbol.size != SAGR_CODE_OBJECT_DESCRIPTOR_BYTES) return SAGR_STATUS_PROTOCOL_ERROR;
    {
      elf_section_t code_section;
      elf_section_t descriptor_section;
      if (!read_section(bytes, image_size, section_offset, section_count,
                        section_entry_size, code_section_index, &code_section) ||
          !read_section(bytes, image_size, section_offset, section_count,
                        section_entry_size, descriptor_section_index, &descriptor_section) ||
          code_symbol.value < code_section.address ||
          code_symbol.value - code_section.address >= code_section.size ||
          descriptor_symbol.value < descriptor_section.address ||
          descriptor_symbol.value - descriptor_section.address + descriptor_symbol.size > descriptor_section.size ||
          !section_in_bounds(&descriptor_section, image_size)) return SAGR_STATUS_PROTOCOL_ERROR;
      kernel->code_address = code_symbol.value;
      kernel->code_file_offset = code_section.offset + (code_symbol.value - code_section.address);
      kernel->code_size = code_symbol.size;
      kernel->descriptor_address = descriptor_symbol.value;
      kernel->descriptor_file_offset = descriptor_section.offset + (descriptor_symbol.value - descriptor_section.address);
      kernel->descriptor_size = (uint32_t)descriptor_symbol.size;
      memcpy(kernel->descriptor, bytes + (size_t)kernel->descriptor_file_offset,
             SAGR_CODE_OBJECT_DESCRIPTOR_BYTES);
      if (!descriptor_valid(kernel->descriptor, sizeof(kernel->descriptor))) return SAGR_STATUS_PROTOCOL_ERROR;
      kernel->descriptor_group_segment_fixed_size = read_u32(kernel->descriptor + 0U);
      kernel->descriptor_private_segment_fixed_size = read_u32(kernel->descriptor + 4U);
      kernel->descriptor_kernarg_segment_size = read_u32(kernel->descriptor + 8U);
      kernel->descriptor_kernel_code_entry_byte_offset =
          (int64_t)read_u64(kernel->descriptor + 16U);
      kernel->descriptor_compute_pgm_rsrc3 = read_u32(kernel->descriptor + 44U);
      kernel->descriptor_compute_pgm_rsrc1 = read_u32(kernel->descriptor + 48U);
      kernel->descriptor_compute_pgm_rsrc2 = read_u32(kernel->descriptor + 52U);
      kernel->descriptor_kernel_code_properties = read_u16(kernel->descriptor + 56U);
      kernel->descriptor_kernarg_preload = read_u16(kernel->descriptor + 58U);
      if (kernel->descriptor_group_segment_fixed_size != kernel->group_segment_fixed_size ||
          kernel->descriptor_private_segment_fixed_size != kernel->private_segment_fixed_size ||
          kernel->descriptor_kernarg_segment_size != kernel->kernarg_segment_size ||
          kernel->descriptor_kernel_code_entry_byte_offset < 0 ||
          descriptor_symbol.value + (uint64_t)kernel->descriptor_kernel_code_entry_byte_offset !=
              code_symbol.value) return SAGR_STATUS_PROTOCOL_ERROR;
    }
    kernel->relocation_count = info->relocation_count;
    kernel->isa_supported_by_gemsim = 0U;
  }
  info->isa_supported_by_gemsim = 0U;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_code_object_get_kernel(
    const sagr_code_object_info_t *info, const char *name,
    sagr_code_object_kernel_info_t *kernel, uint32_t kernel_size) {
  uint32_t index;
  if (info == NULL || name == NULL || kernel == NULL) return SAGR_STATUS_INVALID_ARGUMENT;
  if (kernel_size < (uint32_t)sizeof(*kernel)) return SAGR_STATUS_BUFFER_TOO_SMALL;
  if (info->struct_size != (uint32_t)sizeof(*info) || info->kernel_count > SAGR_CODE_OBJECT_MAX_KERNELS) return SAGR_STATUS_INVALID_ARGUMENT;
  for (index = 0; index < info->kernel_count; ++index) {
    if (strcmp(info->kernels[index].name, name) == 0) {
      memcpy(kernel, &info->kernels[index], sizeof(*kernel));
      return SAGR_STATUS_SUCCESS;
    }
  }
  return SAGR_STATUS_INVALID_ARGUMENT;
}

sagr_status_t sagr_code_object_describe_dispatch(
    const sagr_code_object_info_t *info, const char *kernel_name,
    size_t image_size, sagr_code_object_dispatch_binding_t *binding,
    uint32_t binding_size) {
  sagr_code_object_kernel_info_t kernel;
  if (info == NULL || kernel_name == NULL || binding == NULL || image_size == 0U)
    return SAGR_STATUS_INVALID_ARGUMENT;
  if (binding_size < (uint32_t)sizeof(*binding)) return SAGR_STATUS_BUFFER_TOO_SMALL;
  if (sagr_code_object_get_kernel(info, kernel_name, &kernel,
                                  (uint32_t)sizeof(kernel)) != SAGR_STATUS_SUCCESS)
    return SAGR_STATUS_INVALID_ARGUMENT;
  memset(binding, 0, sizeof(*binding));
  binding->struct_size = (uint32_t)sizeof(*binding);
  binding->flags = SAGR_CODE_OBJECT_DISPATCH_FLAG_METADATA_ONLY |
                   SAGR_CODE_OBJECT_DISPATCH_FLAG_REQUIRES_CODE_OBJECT_TRANSPORT;
  binding->gfx_target = info->gfx_target;
  binding->code_object_version = info->code_object_version;
  binding->metadata_major = info->metadata_major;
  binding->metadata_minor = info->metadata_minor;
  binding->kernel_index = kernel.index;
  binding->wavefront_size = kernel.wavefront_size;
  binding->max_flat_workgroup_size = kernel.max_flat_workgroup_size;
  binding->kernarg_segment_size = kernel.kernarg_segment_size;
  binding->kernarg_segment_align = kernel.kernarg_segment_align;
  binding->descriptor_size = kernel.descriptor_size;
  binding->relocation_count = kernel.relocation_count;
  binding->isa_supported_by_gemsim = kernel.isa_supported_by_gemsim;
  binding->requires_explicit_code_object_transport = 1U;
  binding->image_size = (uint64_t)image_size;
  binding->code_address = kernel.code_address;
  binding->code_file_offset = kernel.code_file_offset;
  binding->code_size = kernel.code_size;
  binding->descriptor_address = kernel.descriptor_address;
  binding->descriptor_file_offset = kernel.descriptor_file_offset;
  binding->descriptor_kernel_code_entry_byte_offset =
      kernel.descriptor_kernel_code_entry_byte_offset;
  memcpy(binding->kernel_name, kernel.name, sizeof(binding->kernel_name));
  memcpy(binding->symbol, kernel.symbol, sizeof(binding->symbol));
  memcpy(binding->descriptor, kernel.descriptor, sizeof(binding->descriptor));
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_code_object_materialize_kernel_code(
    const void *image, size_t image_size,
    const sagr_code_object_dispatch_binding_t *binding,
    uint8_t *destination, size_t destination_size, size_t *written_size) {
  const uint8_t *bytes = (const uint8_t *)image;
  size_t code_offset;
  size_t code_size;
  if (image == NULL || binding == NULL || destination == NULL ||
      written_size == NULL || image_size == 0U ||
      binding->struct_size != (uint32_t)sizeof(*binding) ||
      binding->flags != (SAGR_CODE_OBJECT_DISPATCH_FLAG_METADATA_ONLY |
                         SAGR_CODE_OBJECT_DISPATCH_FLAG_REQUIRES_CODE_OBJECT_TRANSPORT) ||
      binding->requires_explicit_code_object_transport != 1U ||
      binding->reserved0 != 0U ||
      binding->image_size != (uint64_t)image_size ||
      binding->code_size == 0U || binding->code_file_offset > (uint64_t)SIZE_MAX ||
      binding->code_size > (uint64_t)SIZE_MAX)
    return SAGR_STATUS_INVALID_ARGUMENT;
  code_offset = (size_t)binding->code_file_offset;
  code_size = (size_t)binding->code_size;
  if (code_offset > image_size || code_size > image_size - code_offset)
    return SAGR_STATUS_PROTOCOL_ERROR;
  if (destination_size < code_size) return SAGR_STATUS_BUFFER_TOO_SMALL;
  memcpy(destination, bytes + code_offset, code_size);
  *written_size = code_size;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_code_object_pack_kernarg(
    const sagr_code_object_kernel_info_t *kernel,
    const sagr_code_object_arg_value_t *values, uint32_t value_count,
    uint8_t *destination, uint32_t destination_size, uint32_t *written_size) {
  uint32_t index;
  uint32_t value_index;
  uint64_t seen = 0U;
  if (kernel == NULL || destination == NULL || written_size == NULL ||
      (value_count != 0U && values == NULL) ||
      kernel->struct_size != (uint32_t)sizeof(*kernel) ||
      kernel->kernarg_segment_size == 0U) return SAGR_STATUS_INVALID_ARGUMENT;
  if (destination_size < kernel->kernarg_segment_size) return SAGR_STATUS_BUFFER_TOO_SMALL;
  for (value_index = 0; value_index < value_count; ++value_index) {
    const sagr_code_object_arg_value_t *value = &values[value_index];
    const sagr_code_object_arg_info_t *arg = NULL;
    if (value->struct_size < (uint32_t)sizeof(*value) || value->arg_index >= kernel->arg_count) return SAGR_STATUS_INVALID_ARGUMENT;
    for (index = 0; index < kernel->arg_count; ++index) {
      if (kernel->args[index].index == value->arg_index) { arg = &kernel->args[index]; break; }
    }
    if (arg == NULL || arg->size_bytes > sizeof(value->value) ||
        arg->offset_bytes > kernel->kernarg_segment_size ||
        arg->size_bytes > kernel->kernarg_segment_size - arg->offset_bytes ||
        value->arg_index >= 64U || (seen & (UINT64_C(1) << value->arg_index)) != 0U)
      return SAGR_STATUS_INVALID_ARGUMENT;
    seen |= UINT64_C(1) << value->arg_index;
  }
  memset(destination, 0, kernel->kernarg_segment_size);
  for (value_index = 0; value_index < value_count; ++value_index) {
    const sagr_code_object_arg_value_t *value = &values[value_index];
    const sagr_code_object_arg_info_t *arg = NULL;
    for (index = 0; index < kernel->arg_count; ++index) {
      if (kernel->args[index].index == value->arg_index) { arg = &kernel->args[index]; break; }
    }
    if (arg == NULL) return SAGR_STATUS_INTERNAL_ERROR;
    for (index = 0; index < arg->size_bytes; ++index) {
      destination[arg->offset_bytes + index] = (uint8_t)(value->value >> (8U * index));
    }
  }
  *written_size = kernel->kernarg_segment_size;
  return SAGR_STATUS_SUCCESS;
}

int sagr_code_object_gemsim_isa_supported(const sagr_code_object_info_t *info) {
  return info != NULL && info->struct_size == (uint32_t)sizeof(*info) &&
         info->gfx_target == SAGR_CODE_OBJECT_TARGET_GFX950 &&
         info->isa_supported_by_gemsim != 0U;
}
