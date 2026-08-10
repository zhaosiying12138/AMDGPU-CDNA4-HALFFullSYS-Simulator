/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "sha256_internal.h"

#include <self_amdgpu_runtime/code_object.h>
#include <self_amdgpu_runtime/runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const CanonicalSha256 =
    "7308427e69dea6f320178c55863291d4d615338eb295a422a5ff7a2c2b8afa95";

#define CP28_ELEMENT_COUNT UINT32_C(98432)
#define CP28_BUFFER_BYTES \
  (UINT64_C(98432) * (uint64_t)sizeof(float))

static int expect(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "CP28 Triton artifact: %s\n", message);
    return 1;
  }
  return 0;
}

static int read_image(const char *path, uint8_t **image, size_t *size) {
  FILE *file;
  long length;
  size_t read_size;
  if (path == NULL || image == NULL || size == NULL) {
    return 0;
  }
  file = fopen(path, "rb");
  if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
    if (file != NULL) {
      fclose(file);
    }
    return 0;
  }
  length = ftell(file);
  if (length <= 0L || fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return 0;
  }
  *size = (size_t)length;
  *image = (uint8_t *)malloc(*size);
  if (*image == NULL) {
    fclose(file);
    return 0;
  }
  read_size = fread(*image, 1U, *size, file);
  fclose(file);
  if (read_size != *size) {
    free(*image);
    *image = NULL;
    *size = 0U;
    return 0;
  }
  return 1;
}

static void hex_digest(const uint8_t digest[SAGR_SHA256_BYTES],
                       char output[65]) {
  static const char Hex[] = "0123456789abcdef";
  size_t index;
  for (index = 0U; index < SAGR_SHA256_BYTES; ++index) {
    output[index * 2U] = Hex[digest[index] >> 4U];
    output[index * 2U + 1U] = Hex[digest[index] & UINT8_C(0x0f)];
  }
  output[64] = '\0';
}

static int validate_canonical(const uint8_t *image, size_t image_size,
                              char digest_text[65]) {
  uint8_t digest[SAGR_SHA256_BYTES];
  sagr_code_object_info_t info;
  sagr_code_object_kernel_info_t kernel;
  sagr_code_object_dispatch_binding_t binding;
  uint8_t code[416];
  size_t code_size = 0U;
  sagr_status_t status;
  int failures = 0;

  sagr_sha256(image, image_size, digest);
  hex_digest(digest, digest_text);
  failures += expect(image_size == 5384U, "canonical image size changed");
  failures += expect(strcmp(digest_text, CanonicalSha256) == 0,
                     "canonical image SHA-256 changed");

  memset(&info, 0, sizeof(info));
  status = sagr_code_object_validate(image, image_size, &info,
                                     (uint32_t)sizeof(info));
  failures += expect(status == SAGR_STATUS_SUCCESS,
                     "canonical HSACO validation failed");
  if (status != SAGR_STATUS_SUCCESS) {
    return failures;
  }
  failures += expect(
      info.elf_machine == SAGR_CODE_OBJECT_ELF_MACHINE_AMDGPU &&
          info.elf_type == SAGR_CODE_OBJECT_ELF_TYPE_DYN &&
          info.elf_osabi == SAGR_CODE_OBJECT_ELF_OSABI_AMDGPU_HSA &&
          info.elf_abi_version == 3U && info.code_object_version == 5U &&
          info.gfx_target == SAGR_CODE_OBJECT_TARGET_GFX950 &&
          info.metadata_major == 1U && info.metadata_minor == 2U &&
          info.kernel_count == 1U && info.relocation_count == 0U &&
          info.segment_count == 3U &&
          strcmp(info.target, "amdgcn-amd-amdhsa-unknown-gfx950") == 0,
      "canonical ELF, target, or metadata identity changed");
  failures += expect(
      info.segments[0].type == 1U && info.segments[0].flags == 4U &&
          info.segments[0].file_offset == UINT64_C(0) &&
          info.segments[0].virtual_address == UINT64_C(0) &&
          info.segments[0].file_size == UINT64_C(0x600) &&
          info.segments[0].memory_size == UINT64_C(0x600) &&
          info.segments[0].alignment == UINT64_C(0x1000) &&
          info.segments[1].type == 1U && info.segments[1].flags == 5U &&
          info.segments[1].file_offset == UINT64_C(0x600) &&
          info.segments[1].virtual_address == UINT64_C(0x1600) &&
          info.segments[1].file_size == UINT64_C(0x5c0) &&
          info.segments[1].memory_size == UINT64_C(0x5c0) &&
          info.segments[1].alignment == UINT64_C(0x1000) &&
          info.segments[2].type == 1U && info.segments[2].flags == 6U &&
          info.segments[2].file_offset == UINT64_C(0xbc0) &&
          info.segments[2].virtual_address == UINT64_C(0x2bc0) &&
          info.segments[2].file_size == UINT64_C(0x70) &&
          info.segments[2].memory_size == UINT64_C(0x440) &&
          info.segments[2].alignment == UINT64_C(0x1000),
      "canonical PT_LOAD layout changed");

  memset(&kernel, 0, sizeof(kernel));
  status = sagr_code_object_get_kernel(&info, "add_kernel", &kernel,
                                       (uint32_t)sizeof(kernel));
  failures += expect(status == SAGR_STATUS_SUCCESS,
                     "add_kernel metadata lookup failed");
  if (status != SAGR_STATUS_SUCCESS) {
    return failures;
  }
  failures += expect(
      kernel.kernarg_segment_size == 48U &&
          kernel.kernarg_segment_align == 8U &&
          kernel.max_flat_workgroup_size == 256U &&
          kernel.wavefront_size == 64U && kernel.relocation_count == 0U &&
          kernel.descriptor_size == 64U &&
          kernel.descriptor_address == UINT64_C(0x5c0) &&
          kernel.descriptor_file_offset == UINT64_C(0x5c0) &&
          kernel.code_address == UINT64_C(0x1600) &&
          kernel.code_file_offset == UINT64_C(0x600) &&
          kernel.code_size == UINT64_C(416) &&
          kernel.descriptor_kernel_code_entry_byte_offset == INT64_C(0x1040),
      "add_kernel descriptor or code layout changed");
  failures += expect(
      kernel.descriptor_kernarg_preload == 12U &&
          (kernel.descriptor_kernarg_preload & UINT16_C(0x7f)) == 12U &&
          (kernel.descriptor_kernarg_preload >> 7U) == 0U,
      "add_kernel preload must remain length 12 DWORDs at offset zero");

  memset(&binding, 0, sizeof(binding));
  status = sagr_code_object_describe_dispatch(
      &info, "add_kernel", image_size, &binding, (uint32_t)sizeof(binding));
  failures += expect(
      status == SAGR_STATUS_SUCCESS &&
          binding.code_size == sizeof(code) &&
          binding.descriptor_kernel_code_entry_byte_offset == INT64_C(0x1040),
      "add_kernel dispatch binding changed");
  if (status == SAGR_STATUS_SUCCESS) {
    status = sagr_code_object_materialize_kernel_code(
        image, image_size, &binding, code, sizeof(code), &code_size);
    failures += expect(status == SAGR_STATUS_SUCCESS &&
                           code_size == sizeof(code) &&
                           memcmp(code, image + 0x600U, sizeof(code)) == 0,
                       "add_kernel code materialization failed");
  }
  return failures;
}

static int validate_managed_load(const uint8_t *image, size_t image_size) {
  sagr_managed_session_t session = NULL;
  sagr_managed_kernel_t kernel = NULL;
  sagr_managed_session_info_t session_info;
  sagr_managed_kernel_info_t kernel_info;
  sagr_error_info_t error;
  sagr_status_t status;
  int failures = 0;

  memset(&session_info, 0, sizeof(session_info));
  memset(&error, 0, sizeof(error));
  status = sagr_managed_session_open(
      NULL, &session, &session_info, (uint32_t)sizeof(session_info), &error,
      (uint32_t)sizeof(error));
  failures += expect(status == SAGR_STATUS_SUCCESS,
                     "managed simulator session open failed");
  if (status != SAGR_STATUS_SUCCESS) {
    fprintf(stderr,
            "CP28 Triton artifact: managed open: status=%d wire=%d errno=%d "
            "message=%s\n",
            (int)status, (int)error.wire_status, (int)error.native_errno,
            error.message);
    return failures;
  }

  memset(&kernel_info, 0, sizeof(kernel_info));
  status = sagr_managed_kernel_load(
      session, image, image_size, "add_kernel", &kernel, &kernel_info,
      (uint32_t)sizeof(kernel_info), &error, (uint32_t)sizeof(error));
  failures += expect(status == SAGR_STATUS_SUCCESS,
                     "managed add_kernel upload/MAP failed");
  if (status != SAGR_STATUS_SUCCESS) {
    fprintf(stderr,
            "CP28 Triton artifact: managed load: status=%d wire=%d errno=%d "
            "message=%s\n",
            (int)status, (int)error.wire_status, (int)error.native_errno,
            error.message);
  } else {
    failures += expect(
        kernel_info.version == SAGR_MANAGED_RUNTIME_API_VERSION &&
            kernel_info.kernarg_segment_size == 48U &&
            kernel_info.descriptor_preload_dwords == 12U &&
            kernel_info.entry_va != 0U && kernel_info.kernarg_va != 0U,
        "managed add_kernel identity or preload changed");
    failures += expect(sagr_managed_kernel_unload(
                           &kernel, &error, (uint32_t)sizeof(error)) ==
                           SAGR_STATUS_SUCCESS,
                       "managed add_kernel unmap failed");
  }
  failures += expect(sagr_managed_session_close(
                         &session, &error, (uint32_t)sizeof(error)) ==
                         SAGR_STATUS_SUCCESS,
                     "managed simulator session close failed");
  return failures;
}

static int managed_call_succeeded(const char *operation, sagr_status_t status,
                                  const sagr_error_info_t *error) {
  if (status == SAGR_STATUS_SUCCESS) {
    return 1;
  }
  fprintf(stderr,
          "CP28 Triton artifact: %s: status=%d wire=%d errno=%d message=%s\n",
          operation, (int)status, error != NULL ? (int)error->wire_status : 0,
          error != NULL ? (int)error->native_errno : 0,
          error != NULL ? error->message : "");
  return 0;
}

static void fill_inputs(float *a, float *b, float *c, uint32_t pass) {
  uint32_t index;
  for (index = 0U; index < CP28_ELEMENT_COUNT; ++index) {
    a[index] = (float)(1U + pass * 2U + index % 127U) * 0.25F;
    b[index] = (float)(2U + pass * 3U + index % 31U) * 0.5F;
    memset(&c[index], 0x7f, sizeof(c[index]));
  }
}

static int validate_output(const float *a, const float *b, const float *c,
                           uint32_t pass) {
  uint32_t index;
  uint32_t mismatches = 0U;
  for (index = 0U; index < CP28_ELEMENT_COUNT; ++index) {
    const float expected = a[index] + b[index];
    if (c[index] != expected) {
      if (mismatches == 0U) {
        fprintf(stderr,
                "CP28 Triton artifact: pass %u first mismatch index=%u "
                "actual=%g expected=%g\n",
                pass, index, (double)c[index], (double)expected);
      }
      ++mismatches;
    }
  }
  if (mismatches != 0U) {
    fprintf(stderr, "CP28 Triton artifact: pass %u mismatches=%u\n", pass,
            mismatches);
    return 0;
  }
  return 1;
}

static int validate_managed_launch(
    const uint8_t *image, size_t image_size,
    sagr_generic_dispatch_completion_t completions[2]) {
  sagr_managed_session_t session = NULL;
  sagr_managed_kernel_t kernel = NULL;
  sagr_managed_buffer_t buffers[3] = {NULL, NULL, NULL};
  sagr_memory_info_t buffer_info[3];
  sagr_managed_kernel_info_t kernel_info;
  sagr_managed_launch_options_t launch;
  sagr_error_info_t error;
  float *host[3] = {NULL, NULL, NULL};
  uint8_t kernarg[48];
  uint32_t n_elements = CP28_ELEMENT_COUNT;
  uint32_t index;
  uint32_t pass;
  sagr_status_t status;
  int failures = 0;

  memset(buffer_info, 0, sizeof(buffer_info));
  memset(&kernel_info, 0, sizeof(kernel_info));
  memset(&error, 0, sizeof(error));
  memset(completions, 0, sizeof(completions[0]) * 2U);
  for (index = 0U; index < 3U; ++index) {
    host[index] = (float *)malloc((size_t)CP28_BUFFER_BYTES);
    if (host[index] == NULL) {
      fprintf(stderr, "CP28 Triton artifact: host buffer allocation failed\n");
      ++failures;
      goto cleanup;
    }
  }

  status = sagr_managed_session_open(
      NULL, &session, NULL, 0U, &error, (uint32_t)sizeof(error));
  if (!managed_call_succeeded("managed session open", status, &error)) {
    ++failures;
    goto cleanup;
  }
  status = sagr_managed_kernel_load(
      session, image, image_size, "add_kernel", &kernel, &kernel_info,
      (uint32_t)sizeof(kernel_info), &error, (uint32_t)sizeof(error));
  if (!managed_call_succeeded("managed kernel load", status, &error)) {
    ++failures;
    goto cleanup;
  }
  for (index = 0U; index < 3U; ++index) {
    status = sagr_managed_buffer_allocate(
        session, CP28_BUFFER_BYTES, SAGR_MEMORY_ALIGNMENT_64K, &buffers[index],
        &buffer_info[index], (uint32_t)sizeof(buffer_info[index]), &error,
        (uint32_t)sizeof(error));
    if (!managed_call_succeeded("managed buffer allocate", status, &error)) {
      ++failures;
      goto cleanup;
    }
    failures += expect(buffer_info[index].simulated_va != 0U &&
                           buffer_info[index].size_bytes >= CP28_BUFFER_BYTES,
                       "managed large buffer identity changed");
  }

  memset(kernarg, 0, sizeof(kernarg));
  memcpy(kernarg, &buffer_info[0].simulated_va,
         sizeof(buffer_info[0].simulated_va));
  memcpy(kernarg + 8U, &buffer_info[1].simulated_va,
         sizeof(buffer_info[1].simulated_va));
  memcpy(kernarg + 16U, &buffer_info[2].simulated_va,
         sizeof(buffer_info[2].simulated_va));
  memcpy(kernarg + 24U, &n_elements, sizeof(n_elements));
  status = sagr_managed_launch_options_init(&launch,
                                             (uint32_t)sizeof(launch));
  if (!managed_call_succeeded("managed launch options init", status, &error)) {
    ++failures;
    goto cleanup;
  }
  launch.grid_x = 24832U;
  launch.workgroup_x = 256U;
  launch.num_warps = 4U;
  launch.num_ctas = 1U;

  for (pass = 0U; pass < 2U; ++pass) {
    fill_inputs(host[0], host[1], host[2], pass);
    for (index = 0U; index < 2U; ++index) {
      status = sagr_managed_buffer_copy_from_host(
          buffers[index], 0U, host[index], CP28_BUFFER_BYTES, &error,
          (uint32_t)sizeof(error));
      if (!managed_call_succeeded("managed input H2D", status, &error)) {
        ++failures;
        goto cleanup;
      }
    }
    status = sagr_managed_kernel_launch(
        kernel, kernarg, sizeof(kernarg), &launch, &completions[pass],
        (uint32_t)sizeof(completions[pass]), &error, (uint32_t)sizeof(error));
    if (!managed_call_succeeded("managed kernel launch", status, &error)) {
      ++failures;
      goto cleanup;
    }
    failures += expect(
        completions[pass].status == SAGR_STATUS_SUCCESS &&
            completions[pass].end_tick >= completions[pass].start_tick &&
            completions[pass].retire_tick >= completions[pass].end_tick,
        "managed completion status or tick ordering changed");
    status = sagr_managed_buffer_copy_to_host(
        buffers[2], 0U, host[2], CP28_BUFFER_BYTES, &error,
        (uint32_t)sizeof(error));
    if (!managed_call_succeeded("managed output-only D2H", status, &error)) {
      ++failures;
      goto cleanup;
    }
    failures += expect(validate_output(host[0], host[1], host[2], pass),
                       "managed vecadd oracle failed");
    if (failures != 0) {
      goto cleanup;
    }
  }

cleanup:
  if (kernel != NULL &&
      sagr_managed_kernel_unload(&kernel, &error,
                                 (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "CP28 Triton artifact: managed kernel cleanup failed\n");
    ++failures;
  }
  for (index = 3U; index > 0U; --index) {
    if (buffers[index - 1U] != NULL &&
        sagr_managed_buffer_free(&buffers[index - 1U], &error,
                                 (uint32_t)sizeof(error)) !=
            SAGR_STATUS_SUCCESS) {
      fprintf(stderr, "CP28 Triton artifact: managed buffer cleanup failed\n");
      ++failures;
    }
  }
  if (session != NULL &&
      sagr_managed_session_close(&session, &error,
                                 (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "CP28 Triton artifact: managed session cleanup failed\n");
    ++failures;
  }
  for (index = 0U; index < 3U; ++index) {
    free(host[index]);
  }
  return failures;
}

static void dump_begin_manifest(const uint8_t *image, size_t image_size) {
  sagr_code_object_info_t info;
  sagr_code_object_kernel_info_t kernel;
  char descriptor[129];
  static const char Hex[] = "0123456789abcdef";
  uint32_t index;
  memset(&info, 0, sizeof(info));
  memset(&kernel, 0, sizeof(kernel));
  if (sagr_code_object_validate(image, image_size, &info,
                                (uint32_t)sizeof(info)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_code_object_get_kernel(&info, "add_kernel", &kernel,
                                  (uint32_t)sizeof(kernel)) !=
          SAGR_STATUS_SUCCESS) {
    return;
  }
  for (index = 0U; index < sizeof(kernel.descriptor); ++index) {
    descriptor[index * 2U] = Hex[kernel.descriptor[index] >> 4U];
    descriptor[index * 2U + 1U] =
        Hex[kernel.descriptor[index] & UINT8_C(0x0f)];
  }
  descriptor[128] = '\0';
  fprintf(stderr,
          "CP28 BEGIN image=%zu kernel_index=%u elf_machine=%u elf_type=%u "
          "osabi=%u abi=%u flags=0x%x gfx=%u cov=%u metadata=%u.%u "
          "relocations=%u kernarg=%u/%u group=%u private=%u max_wg=%u "
          "wave=%u sgpr=%u vgpr=%u dynamic_stack=%u\n",
          image_size, kernel.index, (unsigned)info.elf_machine,
          (unsigned)info.elf_type, (unsigned)info.elf_osabi,
          (unsigned)info.elf_abi_version, info.elf_flags, info.gfx_target,
          info.code_object_version, info.metadata_major, info.metadata_minor,
          kernel.relocation_count, kernel.kernarg_segment_size,
          kernel.kernarg_segment_align, kernel.group_segment_fixed_size,
          kernel.private_segment_fixed_size, kernel.max_flat_workgroup_size,
          kernel.wavefront_size, kernel.sgpr_count, kernel.vgpr_count,
          kernel.uses_dynamic_stack);
  fprintf(stderr,
          "CP28 BEGIN name=%s symbol=%s descriptor_va=0x%llx "
          "descriptor_file=0x%llx descriptor_size=%u raw_entry=0x%llx "
          "code_va=0x%llx code_file=0x%llx code_size=0x%llx "
          "preload_raw=%u descriptor=%s\n",
          kernel.name, kernel.symbol,
          (unsigned long long)kernel.descriptor_address,
          (unsigned long long)kernel.descriptor_file_offset,
          kernel.descriptor_size,
          (unsigned long long)kernel.descriptor_kernel_code_entry_byte_offset,
          (unsigned long long)kernel.code_address,
          (unsigned long long)kernel.code_file_offset,
          (unsigned long long)kernel.code_size,
          (unsigned)kernel.descriptor_kernarg_preload, descriptor);
  for (index = 0U; index < info.segment_count; ++index) {
    const sagr_code_object_segment_info_t *segment = &info.segments[index];
    fprintf(stderr,
            "CP28 BEGIN segment[%u]=type:%u flags:%u file:0x%llx "
            "va:0x%llx filesz:0x%llx memsz:0x%llx align:0x%llx\n",
            index, segment->type, segment->flags,
            (unsigned long long)segment->file_offset,
            (unsigned long long)segment->virtual_address,
            (unsigned long long)segment->file_size,
            (unsigned long long)segment->memory_size,
            (unsigned long long)segment->alignment);
  }
}

int main(int argc, char **argv) {
  uint8_t *image = NULL;
  size_t image_size = 0U;
  char digest[65];
  int managed = 0;
  int launch = 0;
  int parser_failures;
  int managed_failures = 0;
  sagr_generic_dispatch_completion_t completions[2];

  if (argc == 1) {
    fprintf(stderr,
            "CP28 Triton artifact: canonical HSACO path is not configured\n");
    return 77;
  }
  if (argc < 2 || argc > 3 ||
      (argc == 3 && strcmp(argv[2], "--managed") != 0 &&
       strcmp(argv[2], "--launch") != 0) ||
      !read_image(argv[1], &image, &image_size)) {
    fprintf(stderr, "usage: %s IMAGE.hsaco [--managed|--launch]\n", argv[0]);
    return 2;
  }
  managed = argc == 3;
  launch = managed != 0 && strcmp(argv[2], "--launch") == 0;
  memset(completions, 0, sizeof(completions));
  parser_failures = validate_canonical(image, image_size, digest);
  if (parser_failures == 0 && managed != 0) {
    dump_begin_manifest(image, image_size);
    managed_failures = launch != 0
                           ? validate_managed_launch(image, image_size,
                                                     completions)
                           : validate_managed_load(image, image_size);
  }
  printf("{\"schema\":\"self-amdgpu-runtime.cp28-triton-artifact.v1\","
         "\"hsaco_sha256\":\"%s\",\"hsaco_bytes\":%zu,"
         "\"kernel\":\"add_kernel\",\"preload_dwords\":12,"
         "\"preload_offset_dwords\":0,\"parser\":%s,\"managed\":%s,"
         "\"launch\":%s,\"reuse\":%s,\"first_sim_tick\":%llu,"
         "\"second_sim_tick\":%llu}\n",
         digest, image_size, parser_failures == 0 ? "true" : "false",
         managed != 0 && parser_failures == 0 && managed_failures == 0
             ? "true"
             : "false",
         launch != 0 && managed_failures == 0 ? "true" : "false",
         launch != 0 && managed_failures == 0 ? "true" : "false",
         (unsigned long long)completions[0].sim_tick,
         (unsigned long long)completions[1].sim_tick);
  free(image);
  return parser_failures == 0 && managed_failures == 0 ? 0 : 1;
}
