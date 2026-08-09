/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <self_amdgpu_runtime/code_object.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SAGR_CODE_OBJECT_GPU_READ_WRITE_PATH
#define SAGR_CODE_OBJECT_GPU_READ_WRITE_PATH ""
#endif
#ifndef SAGR_CODE_OBJECT_BINARY_SEARCH_PATH
#define SAGR_CODE_OBJECT_BINARY_SEARCH_PATH ""
#endif

static int expect(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "code-object test: %s\n", message);
    return 1;
  }
  return 0;
}

static int read_file(const char *path, uint8_t **bytes, size_t *size) {
  FILE *file;
  long length;
  size_t read_size;
  if (path == NULL || bytes == NULL || size == NULL || path[0] == '\0') return 0;
  file = fopen(path, "rb");
  if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
    if (file != NULL) fclose(file);
    return 0;
  }
  length = ftell(file);
  if (length <= 0L || fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return 0;
  }
  *size = (size_t)length;
  *bytes = (uint8_t *)malloc(*size);
  if (*bytes == NULL) {
    fclose(file);
    return 0;
  }
  read_size = fread(*bytes, 1U, *size, file);
  fclose(file);
  if (read_size != *size) {
    free(*bytes);
    *bytes = NULL;
    *size = 0U;
    return 0;
  }
  return 1;
}

static int check_fixture(const char *path, const char *kernel_name,
                         uint32_t expected_kernels, uint32_t expected_kernarg) {
  uint8_t *bytes = NULL;
  size_t size = 0U;
  sagr_code_object_info_t info;
  sagr_code_object_kernel_info_t kernel;
  sagr_code_object_arg_value_t value;
  uint8_t *kernarg;
  uint32_t written = 0U;
  int failures = 0;
  if (expect(read_file(path, &bytes, &size), "fixture file is readable")) return 1;
  memset(&info, 0, sizeof(info));
  failures += expect(sagr_code_object_validate(bytes, size, &info,
                                               (uint32_t)sizeof(info)) ==
                         SAGR_STATUS_SUCCESS,
                     "ELF/MsgPack fixture validates");
  failures += expect(info.elf_machine == SAGR_CODE_OBJECT_ELF_MACHINE_AMDGPU &&
                         info.elf_type == SAGR_CODE_OBJECT_ELF_TYPE_DYN &&
                         info.elf_osabi == SAGR_CODE_OBJECT_ELF_OSABI_AMDGPU_HSA &&
                         info.elf_abi_version == SAGR_CODE_OBJECT_ELF_ABI_VERSION,
                     "AMDGPU HSA ELF identity");
  failures += expect(info.gfx_target == SAGR_CODE_OBJECT_TARGET_GFX950 &&
                         info.code_object_version == 6U &&
                         strstr(info.target, "gfx950") != NULL,
                     "gfx950 code-object target");
  failures += expect(info.kernel_count == expected_kernels &&
                         info.relocation_count == 0U &&
                         info.isa_supported_by_gemsim == 0U,
                     "kernel count, relocation and ISA boundary");
  memset(&kernel, 0, sizeof(kernel));
  failures += expect(sagr_code_object_get_kernel(
                         &info, kernel_name, &kernel, (uint32_t)sizeof(kernel)) ==
                         SAGR_STATUS_SUCCESS,
                     "named kernel lookup");
  failures += expect(kernel.kernarg_segment_size == expected_kernarg &&
                         kernel.kernarg_segment_align == 8U &&
                         kernel.wavefront_size == 64U &&
                         kernel.descriptor_size == SAGR_CODE_OBJECT_DESCRIPTOR_BYTES &&
                         kernel.visible_arg_count > 0U && kernel.hidden_arg_count > 0U,
                     "kernarg and descriptor metadata");
  kernarg = (uint8_t *)malloc(kernel.kernarg_segment_size);
  if (kernarg == NULL) {
    free(bytes);
    return failures + 1;
  }
  memset(&value, 0, sizeof(value));
  value.struct_size = (uint32_t)sizeof(value);
  value.arg_index = 0U;
  value.value = UINT64_C(0x1122334455667788);
  failures += expect(sagr_code_object_pack_kernarg(
                         &kernel, &value, 1U, kernarg,
                         kernel.kernarg_segment_size, &written) ==
                         SAGR_STATUS_SUCCESS && written == kernel.kernarg_segment_size &&
                         kernarg[0] == 0x88U && kernarg[7] == 0x11U,
                     "little-endian kernarg packing");
  failures += expect(sagr_code_object_pack_kernarg(
                         &kernel, &value, 1U, kernarg,
                         kernel.kernarg_segment_size - 1U, &written) ==
                         SAGR_STATUS_BUFFER_TOO_SMALL,
                     "undersized kernarg destination rejected");
  free(kernarg);
  free(bytes);
  return failures;
}

static const char *environment_or_default(const char *name, const char *fallback) {
  const char *value = getenv(name);
  return value == NULL || value[0] == '\0' ? fallback : value;
}

int main(void) {
  int failures = 0;
  sagr_code_object_info_t info;
  memset(&info, 0, sizeof(info));
  failures += expect(sagr_code_object_validate(NULL, 0U, &info,
                                               (uint32_t)sizeof(info)) ==
                         SAGR_STATUS_INVALID_ARGUMENT,
                     "null image rejected");
  failures += expect(sagr_code_object_validate("bad", 3U, &info,
                                               (uint32_t)sizeof(info)) ==
                         SAGR_STATUS_PROTOCOL_ERROR,
                     "truncated image rejected");
  failures += check_fixture(environment_or_default(
                                "SAGR_CODE_OBJECT_GPU_READ_WRITE_PATH",
                                SAGR_CODE_OBJECT_GPU_READ_WRITE_PATH),
                            "gpuReadWrite", 1U, 280U);
  failures += check_fixture(environment_or_default(
                                "SAGR_CODE_OBJECT_BINARY_SEARCH_PATH",
                                SAGR_CODE_OBJECT_BINARY_SEARCH_PATH),
                            "binarySearch", 3U, 280U);
  return failures == 0 ? 0 : 1;
}
