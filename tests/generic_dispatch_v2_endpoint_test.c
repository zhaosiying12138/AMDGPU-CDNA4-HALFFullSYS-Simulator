/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <self_amdgpu_runtime/code_object.h>
#include <self_amdgpu_runtime/runtime.h>

#include "transport_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CTest treats this return value as an explicit environment-gated skip. */
#define SAGR_ENDPOINT_TEST_SKIP 77
#define SAGR_ENDPOINT_TEST_HSACO_ENV "SAGR_CODE_OBJECT_GPU_READ_WRITE_PATH"
#define SAGR_ENDPOINT_TEST_MODE_ENV "SAGR_GENERIC_ENDPOINT_MODE"
#define SAGR_ENDPOINT_TEST_MODE_DISCONNECT_AFTER_ACK "disconnect-after-ack"
#define SAGR_ENDPOINT_TEST_KERNEL "gpuReadWrite"
#define SAGR_ENDPOINT_TEST_KERNARG_BYTES UINT32_C(280)
#define SAGR_ENDPOINT_TEST_KERNARG_ALLOCATION_BYTES UINT64_C(512)
#define SAGR_ENDPOINT_TEST_BUFFER_BYTES UINT64_C(4096)
#define SAGR_ENDPOINT_TEST_BUFFER_WORDS UINT32_C(1024)
#define SAGR_ENDPOINT_TEST_KERNARG_OFFSET UINT64_C(64)
#define SAGR_ENDPOINT_TEST_LOGICAL_ALIGNMENT UINT64_C(8)
#define SAGR_ENDPOINT_TEST_BUFFER_ALIGNMENT UINT64_C(4096)
#define SAGR_ENDPOINT_TEST_OUTPUT_CRC32C UINT32_C(0x6f67026f)
#define SAGR_ENDPOINT_TEST_A_CRC32C UINT32_C(0x4705cdab)
#define SAGR_ENDPOINT_TEST_B_CRC32C UINT32_C(0xb28d0486)

typedef enum endpoint_mode {
  ENDPOINT_MODE_POSITIVE = 0,
  ENDPOINT_MODE_DISCONNECT_AFTER_ACK = 1
} endpoint_mode_t;

typedef enum lifecycle_result {
  LIFECYCLE_COMPLETE = 0,
  LIFECYCLE_FAILED = 1,
  LIFECYCLE_DISCONNECT_AFTER_ACK = 2
} lifecycle_result_t;

typedef struct reconnect_snapshot {
  sagr_instance_info_t owner;
  sagr_code_object_remote_info_t object;
  sagr_generic_mapping_info_t mapping;
  sagr_generic_kernarg_info_t kernarg;
} reconnect_snapshot_t;

static int endpoint_unavailable(sagr_status_t status) {
  return status == SAGR_STATUS_ENDPOINT_NOT_FOUND ||
         status == SAGR_STATUS_UNAVAILABLE ||
         status == SAGR_STATUS_TIMED_OUT ||
         status == SAGR_STATUS_CONNECTION_LOST;
}

static int get_endpoint_mode(endpoint_mode_t *mode) {
  const char *value = getenv(SAGR_ENDPOINT_TEST_MODE_ENV);

  if (mode == NULL) {
    return 1;
  }
  if (value == NULL || value[0] == '\0' || strcmp(value, "positive") == 0) {
    *mode = ENDPOINT_MODE_POSITIVE;
    return 0;
  }
  if (strcmp(value, SAGR_ENDPOINT_TEST_MODE_DISCONNECT_AFTER_ACK) == 0) {
    *mode = ENDPOINT_MODE_DISCONNECT_AFTER_ACK;
    return 0;
  }
  fprintf(stderr,
          "generic endpoint probe: %s must be positive or %s\n",
          SAGR_ENDPOINT_TEST_MODE_ENV,
          SAGR_ENDPOINT_TEST_MODE_DISCONNECT_AFTER_ACK);
  return 1;
}

static uint64_t generic_control_dependency_mask(void) {
  return SAGR_CAPABILITY_TOPOLOGY_MASK | SAGR_CAPABILITY_QUEUE_MASK |
         SAGR_CAPABILITY_MEMORY_MASK | SAGR_CAPABILITY_SIGNAL_MASK |
         SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK |
         SAGR_CAPABILITY_GENERIC_DISPATCH_MASK;
}

static uint64_t generic_execution_dependency_mask(void) {
  return generic_control_dependency_mask() |
         SAGR_CAPABILITY_GENERIC_EXECUTION_MASK;
}

static int open_options_for_execution(
    sagr_instance_open_options_t *options) {
  const sagr_status_t status = sagr_instance_open_options_init(
      options, (uint32_t)sizeof(*options));
  if (status != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "generic endpoint probe: options init failed: %s\n",
            sagr_status_string(status));
    return 1;
  }
  /* CP-0026 requires bit 9 plus the existing bit-8 control/dependency set. */
  options->offered_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] |=
      generic_execution_dependency_mask();
  options->required_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] |=
      generic_execution_dependency_mask();
  return 0;
}

static int open_options_for_control(sagr_instance_open_options_t *options) {
  const sagr_status_t status = sagr_instance_open_options_init(
      options, (uint32_t)sizeof(*options));
  if (status != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "generic endpoint probe: control options init failed: %s\n",
            sagr_status_string(status));
    return 1;
  }
  options->offered_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] |=
      generic_control_dependency_mask();
  options->required_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] |=
      generic_control_dependency_mask();
  return 0;
}

static int open_options_for_baseline(
    sagr_instance_open_options_t *options) {
  const sagr_status_t status = sagr_instance_open_options_init(
      options, (uint32_t)sizeof(*options));
  if (status != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "generic endpoint probe: baseline options init failed: %s\n",
            sagr_status_string(status));
    return 1;
  }
  return 0;
}

static int close_instance(sagr_instance_t *instance) {
  const sagr_status_t status = sagr_instance_close(instance);
  if (status != SAGR_STATUS_SUCCESS || *instance != NULL) {
    fprintf(stderr, "generic endpoint probe: instance close failed: %s\n",
            sagr_status_string(status));
    return 1;
  }
  return 0;
}

static void print_open_error(const char *label, sagr_status_t status,
                             const sagr_error_info_t *error) {
  fprintf(stderr,
          "generic endpoint probe: %s: status=%s wire_status=%d errno=%d message=%s\n",
          label, sagr_status_string(status), error->wire_status,
          error->native_errno, error->message[0] == '\0' ? "(none)"
                                                           : error->message);
}

static int call_succeeded(const char *label, sagr_status_t status,
                          const sagr_error_info_t *error) {
  if (status == SAGR_STATUS_SUCCESS) {
    return 1;
  }
  print_open_error(label, status, error);
  return 0;
}

static int require_condition(int condition, const char *message) {
  if (condition != 0) {
    return 1;
  }
  fprintf(stderr, "generic endpoint probe: %s\n", message);
  return 0;
}

static int bytes_are_nonzero(const uint8_t *bytes, size_t size) {
  size_t index;
  uint8_t aggregate = 0U;
  for (index = 0U; index < size; ++index) {
    aggregate = (uint8_t)(aggregate | bytes[index]);
  }
  return aggregate != 0U;
}

static int read_image(const char *path, uint8_t **out_image,
                      size_t *out_size) {
  FILE *file;
  long length;
  size_t size;
  size_t read_size;

  if (path == NULL || path[0] == '\0' || out_image == NULL ||
      out_size == NULL) {
    return 0;
  }
  *out_image = NULL;
  *out_size = 0U;
  file = fopen(path, "rb");
  if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
    if (file != NULL) {
      (void)fclose(file);
    }
    return 0;
  }
  length = ftell(file);
  if (length <= 0L || fseek(file, 0, SEEK_SET) != 0) {
    (void)fclose(file);
    return 0;
  }
  size = (size_t)length;
  *out_image = (uint8_t *)malloc(size);
  if (*out_image == NULL) {
    (void)fclose(file);
    return 0;
  }
  read_size = fread(*out_image, 1U, size, file);
  (void)fclose(file);
  if (read_size != size) {
    free(*out_image);
    *out_image = NULL;
    return 0;
  }
  *out_size = size;
  return 1;
}

static int get_required_info(sagr_instance_t instance, sagr_instance_info_t *info,
                             uint64_t dependencies, const char *label) {
  const sagr_status_t status =
      sagr_instance_get_info(instance, info, (uint32_t)sizeof(*info));
  if (status != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "generic endpoint probe: get_info failed: %s\n",
            sagr_status_string(status));
    return 1;
  }
  if ((info->negotiated_capabilities[SAGR_CAPABILITY_GENERIC_DISPATCH_WORD] &
       dependencies) != dependencies) {
    fprintf(stderr,
            "generic endpoint probe: successful handshake did not select %s and all dependencies\n",
            label);
    return 1;
  }
  return 0;
}

static int get_required_execution_info(sagr_instance_t instance,
                                       sagr_instance_info_t *info) {
  return get_required_info(instance, info, generic_execution_dependency_mask(),
                           "bit 9 execution");
}

static int get_required_control_info(sagr_instance_t instance,
                                     sagr_instance_info_t *info) {
  return get_required_info(instance, info, generic_control_dependency_mask(),
                           "bit 8 control");
}

static int validate_fixture(const uint8_t *image, size_t image_size,
                            sagr_code_object_info_t *info,
                            sagr_code_object_kernel_info_t *kernel,
                            uint8_t kernarg_bytes
                                [SAGR_ENDPOINT_TEST_KERNARG_BYTES]) {
  uint32_t written_size = 0U;
  sagr_status_t status;

  memset(info, 0, sizeof(*info));
  status = sagr_code_object_validate(image, image_size, info,
                                     (uint32_t)sizeof(*info));
  if (!require_condition(status == SAGR_STATUS_SUCCESS,
                         "gpuReadWrite HSACO validation failed")) {
    return 1;
  }
  memset(kernel, 0, sizeof(*kernel));
  status = sagr_code_object_get_kernel(info, SAGR_ENDPOINT_TEST_KERNEL, kernel,
                                       (uint32_t)sizeof(*kernel));
  if (!require_condition(status == SAGR_STATUS_SUCCESS,
                         "gpuReadWrite kernel metadata lookup failed")) {
    return 1;
  }
  if (!require_condition(
          info->gfx_target == SAGR_CODE_OBJECT_TARGET_GFX950 &&
              info->relocation_count == 0U &&
              info->isa_supported_by_gemsim == 0U &&
              kernel->kernarg_segment_size ==
                  SAGR_ENDPOINT_TEST_KERNARG_BYTES &&
              kernel->kernarg_segment_align ==
                  SAGR_ENDPOINT_TEST_LOGICAL_ALIGNMENT &&
              kernel->wavefront_size == 64U &&
              kernel->max_flat_workgroup_size >= 256U &&
              kernel->relocation_count == 0U &&
              kernel->descriptor_kernarg_preload == 0U,
          "gpuReadWrite manifest does not match the generic route contract")) {
    return 1;
  }
  status = sagr_code_object_pack_kernarg(
      kernel, NULL, 0U, kernarg_bytes, SAGR_ENDPOINT_TEST_KERNARG_BYTES,
      &written_size);
  return require_condition(
             status == SAGR_STATUS_SUCCESS &&
                 written_size == SAGR_ENDPOINT_TEST_KERNARG_BYTES,
             "could not materialize the complete 280-byte kernarg manifest")
             ? 0
             : 1;
}

typedef struct execution_buffers {
  sagr_memory_t a;
  sagr_memory_t b;
  sagr_memory_t c;
  sagr_memory_info_t a_info;
  sagr_memory_info_t b_info;
  sagr_memory_info_t c_info;
  uint32_t a_host[SAGR_ENDPOINT_TEST_BUFFER_WORDS];
  uint32_t b_host[SAGR_ENDPOINT_TEST_BUFFER_WORDS];
  uint32_t c_host[SAGR_ENDPOINT_TEST_BUFFER_WORDS];
  uint32_t a_result[SAGR_ENDPOINT_TEST_BUFFER_WORDS];
  uint32_t b_result[SAGR_ENDPOINT_TEST_BUFFER_WORDS];
  uint32_t c_result[SAGR_ENDPOINT_TEST_BUFFER_WORDS];
} execution_buffers_t;

static uint64_t load_le64(const uint8_t *bytes) {
  uint64_t value = 0U;
  uint32_t index;
  for (index = 0U; index < 8U; ++index) {
    value |= (uint64_t)bytes[index] << (index * 8U);
  }
  return value;
}

static int pack_execution_kernarg(
    const sagr_code_object_kernel_info_t *kernel, uint64_t a_va, uint64_t b_va,
    uint64_t c_va, uint8_t destination[SAGR_ENDPOINT_TEST_KERNARG_BYTES]) {
  sagr_code_object_arg_value_t values[4];
  uint32_t hidden_grid_index = UINT32_MAX;
  uint32_t written_size = 0U;
  uint32_t index;
  sagr_status_t status;

  if (kernel == NULL || kernel->arg_count < 4U) {
    return 0;
  }
  for (index = 0U; index < kernel->arg_count; ++index) {
    if (strcmp(kernel->args[index].value_kind, "hidden_grid_dims") == 0) {
      hidden_grid_index = kernel->args[index].index;
      if (kernel->args[index].offset_bytes != 88U ||
          kernel->args[index].size_bytes != 2U ||
          kernel->args[index].kind != SAGR_CODE_OBJECT_ARG_HIDDEN) {
        return 0;
      }
      break;
    }
  }
  if (hidden_grid_index == UINT32_MAX ||
      kernel->args[0].offset_bytes != 0U || kernel->args[0].size_bytes != 8U ||
      kernel->args[1].offset_bytes != 8U || kernel->args[1].size_bytes != 8U ||
      kernel->args[2].offset_bytes != 16U ||
      kernel->args[2].size_bytes != 8U ||
      strcmp(kernel->args[0].value_kind, "global_buffer") != 0 ||
      strcmp(kernel->args[1].value_kind, "global_buffer") != 0 ||
      strcmp(kernel->args[2].value_kind, "global_buffer") != 0 ||
      kernel->args[0].kind != SAGR_CODE_OBJECT_ARG_VISIBLE ||
      kernel->args[1].kind != SAGR_CODE_OBJECT_ARG_VISIBLE ||
      kernel->args[2].kind != SAGR_CODE_OBJECT_ARG_VISIBLE) {
    return 0;
  }
  memset(values, 0, sizeof(values));
  for (index = 0U; index < 4U; ++index) {
    values[index].struct_size = (uint32_t)sizeof(values[index]);
  }
  values[0].arg_index = kernel->args[0].index;
  values[0].value = a_va;
  values[1].arg_index = kernel->args[1].index;
  values[1].value = b_va;
  values[2].arg_index = kernel->args[2].index;
  values[2].value = c_va;
  values[3].arg_index = hidden_grid_index;
  values[3].value = 1U;
  status = sagr_code_object_pack_kernarg(
      kernel, values, 4U, destination, SAGR_ENDPOINT_TEST_KERNARG_BYTES,
      &written_size);
  if (status != SAGR_STATUS_SUCCESS ||
      written_size != SAGR_ENDPOINT_TEST_KERNARG_BYTES ||
      load_le64(destination) != a_va || load_le64(destination + 8U) != b_va ||
      load_le64(destination + 16U) != c_va ||
      load_le64(destination + 24U) != 0U ||
      load_le64(destination + 32U) != 0U ||
      load_le64(destination + 40U) != 0U ||
      load_le64(destination + 64U) != 0U ||
      load_le64(destination + 72U) != 0U ||
      load_le64(destination + 80U) != 0U || destination[88] != 1U ||
      destination[89] != 0U) {
    return 0;
  }
  /* The locked RDC object executes with device-library OLD_ABI global
   * offsets at +24/+32/+40 even though its metadata names block/group fields.
   * Match the CP20 executor: all of +24..+47 and the explicit global offsets
   * at +64/+72/+80 remain zero; only hidden_grid_dims at +88 is one. */
  for (index = 24U; index < SAGR_ENDPOINT_TEST_KERNARG_BYTES; ++index) {
    if (index != 88U && index != 89U && destination[index] != 0U) {
      return 0;
    }
  }
  return 1;
}

static int memory_info_matches_owner(const sagr_memory_info_t *memory,
                                     const sagr_instance_info_t *owner) {
  return memory->struct_size == sizeof(*memory) && memory->allocation_id != 0U &&
         memory->generation != 0U && memory->simulated_va != 0U &&
         memory->size_bytes == SAGR_ENDPOINT_TEST_BUFFER_BYTES &&
         memory->alignment_bytes == SAGR_ENDPOINT_TEST_BUFFER_ALIGNMENT &&
         memory->connection_id == owner->connection_id &&
         memory->epoch == owner->epoch &&
         memcmp(memory->daemon_uuid, owner->daemon_uuid,
                sizeof(memory->daemon_uuid)) == 0;
}

static int allocate_execution_buffers(sagr_instance_t instance,
                                      const sagr_instance_info_t *owner,
                                      execution_buffers_t *buffers) {
  sagr_memory_allocate_options_t options;
  sagr_error_info_t error;
  sagr_memory_t *handles[3] = {&buffers->a, &buffers->b, &buffers->c};
  sagr_memory_info_t *infos[3] = {&buffers->a_info, &buffers->b_info,
                                  &buffers->c_info};
  uint32_t *patterns[3] = {buffers->a_host, buffers->b_host, buffers->c_host};
  uint32_t index;
  uint32_t word;
  sagr_status_t status;

  memset(buffers, 0, sizeof(*buffers));
  status = sagr_memory_allocate_options_init(&options, (uint32_t)sizeof(options));
  if (!require_condition(status == SAGR_STATUS_SUCCESS,
                         "buffer ALLOC options initialization failed")) {
    return 1;
  }
  options.size_bytes = SAGR_ENDPOINT_TEST_BUFFER_BYTES;
  options.alignment_bytes = SAGR_ENDPOINT_TEST_BUFFER_ALIGNMENT;
  for (index = 0U; index < 3U; ++index) {
    for (word = 0U; word < SAGR_ENDPOINT_TEST_BUFFER_WORDS; ++word) {
      patterns[index][word] =
          index == 0U ? (UINT32_C(0x51000000) ^ word)
                      : index == 1U ? (UINT32_C(0x6b000000) ^ word)
                                    : (UINT32_C(0x7c000000) ^ word);
    }
    memset(&error, 0, sizeof(error));
    status = sagr_memory_allocate(instance, &options, NULL, handles[index],
                                  infos[index], (uint32_t)sizeof(*infos[index]),
                                  &error, (uint32_t)sizeof(error));
    if (!call_succeeded("buffer ALLOC", status, &error) ||
        !require_condition(memory_info_matches_owner(infos[index], owner),
                           "buffer ALLOC returned a noncanonical owner lease")) {
      return 1;
    }
    if (!require_condition(
            (index == 0U || infos[index]->simulated_va != buffers->a_info.simulated_va) &&
                (index < 2U || infos[index]->simulated_va != buffers->b_info.simulated_va),
            "buffer ALLOC returned duplicate GPU virtual addresses")) {
      return 1;
    }
    status = sagr_memory_copy_from_host(
        *handles[index], 0U, patterns[index], SAGR_ENDPOINT_TEST_BUFFER_BYTES,
        NULL, &error, (uint32_t)sizeof(error));
    if (!call_succeeded("buffer H2D", status, &error)) {
      return 1;
    }
  }
  return 0;
}

static int verify_execution_buffers(execution_buffers_t *buffers,
                                    uint32_t *output_crc) {
  sagr_memory_t handles[3] = {buffers->a, buffers->b, buffers->c};
  uint32_t *results[3] = {buffers->a_result, buffers->b_result,
                          buffers->c_result};
  uint8_t combined[SAGR_ENDPOINT_TEST_BUFFER_BYTES * 2U];
  sagr_error_info_t error;
  sagr_status_t status;
  uint32_t word;
  uint32_t index;

  for (index = 0U; index < 3U; ++index) {
    memset(&error, 0, sizeof(error));
    status = sagr_memory_copy_to_host(
        handles[index], 0U, results[index], SAGR_ENDPOINT_TEST_BUFFER_BYTES,
        NULL, &error, (uint32_t)sizeof(error));
    if (!call_succeeded("buffer D2H", status, &error)) {
      return 1;
    }
  }
  for (word = 0U; word < SAGR_ENDPOINT_TEST_BUFFER_WORDS; ++word) {
    if (buffers->a_result[word] != buffers->a_host[word] ||
        buffers->b_result[word] != word ||
        buffers->c_result[word] != buffers->a_host[word]) {
      return 1;
    }
  }
  if (sagr_crc32c((const uint8_t *)buffers->a_result,
                  (size_t)SAGR_ENDPOINT_TEST_BUFFER_BYTES) !=
          SAGR_ENDPOINT_TEST_A_CRC32C ||
      sagr_crc32c((const uint8_t *)buffers->b_result,
                  (size_t)SAGR_ENDPOINT_TEST_BUFFER_BYTES) !=
          SAGR_ENDPOINT_TEST_B_CRC32C) {
    return 1;
  }
  memcpy(combined, buffers->b_result, (size_t)SAGR_ENDPOINT_TEST_BUFFER_BYTES);
  memcpy(combined + SAGR_ENDPOINT_TEST_BUFFER_BYTES, buffers->c_result,
         (size_t)SAGR_ENDPOINT_TEST_BUFFER_BYTES);
  *output_crc = sagr_crc32c(combined, sizeof(combined));
  return *output_crc == SAGR_ENDPOINT_TEST_OUTPUT_CRC32C ? 0 : 1;
}

static int verify_duplicate_a_d2h(execution_buffers_t *buffers) {
  uint32_t duplicate_a[SAGR_ENDPOINT_TEST_BUFFER_WORDS];
  sagr_error_info_t error;
  sagr_status_t status;

  memset(&error, 0, sizeof(error));
  memset(duplicate_a, 0, sizeof(duplicate_a));
  status = sagr_memory_copy_to_host(
      buffers->a, 0U, duplicate_a, SAGR_ENDPOINT_TEST_BUFFER_BYTES, NULL,
      &error, (uint32_t)sizeof(error));
  if (!call_succeeded("duplicate buffer A D2H", status, &error)) {
    return 1;
  }
  return require_condition(
             memcmp(duplicate_a, buffers->a_result, sizeof(duplicate_a)) == 0 &&
                 sagr_crc32c((const uint8_t *)duplicate_a,
                             sizeof(duplicate_a)) ==
                     SAGR_ENDPOINT_TEST_A_CRC32C,
             "duplicate buffer A D2H did not preserve the full bytes and CRC")
             ? 0
             : 1;
}

static int free_execution_buffers(execution_buffers_t *buffers) {
  sagr_memory_t *handles[3] = {&buffers->a, &buffers->b, &buffers->c};
  sagr_error_info_t error;
  int failures = 0;
  int index;
  for (index = 2; index >= 0; --index) {
    if (*handles[index] != NULL) {
      memset(&error, 0, sizeof(error));
      if (sagr_memory_free(handles[index], NULL, &error,
                           (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
        print_open_error("buffer FREE", error.status, &error);
        failures = 1;
      }
    }
  }
  return failures;
}

static int upload_object(sagr_instance_t instance, const uint8_t *image,
                         size_t image_size,
                         const sagr_code_object_info_t *info,
                         const sagr_code_object_kernel_info_t *kernel,
                         sagr_code_object_remote_info_t *remote) {
  sagr_error_info_t error;
  sagr_status_t status;

  memset(&error, 0, sizeof(error));
  memset(remote, 0, sizeof(*remote));
  status = sagr_code_object_upload(
      instance, image, image_size, SAGR_ENDPOINT_TEST_KERNEL, NULL, remote,
      (uint32_t)sizeof(*remote), &error, (uint32_t)sizeof(error));
  if (!call_succeeded("code-object upload", status, &error)) {
    return 1;
  }
  return require_condition(
             remote->struct_size == sizeof(*remote) &&
                 remote->flags ==
                     SAGR_CODE_OBJECT_REMOTE_FLAG_STAGED_IDENTITY_ONLY &&
                 remote->object_id != 0U && remote->generation != 0U &&
                 remote->image_size == (uint64_t)image_size &&
                 remote->mapped_base_va == 0U && remote->descriptor_va == 0U &&
                 remote->code_va == 0U && remote->kernarg_va == 0U &&
                 remote->kernel_index == kernel->index &&
                 remote->segment_count == info->segment_count &&
                 bytes_are_nonzero(remote->image_sha256,
                                   sizeof(remote->image_sha256)),
             "code-object upload returned a noncanonical identity")
             ? 0
             : 1;
}

static int map_and_alloc(
    sagr_instance_t instance, const sagr_instance_info_t *owner,
    const sagr_code_object_info_t *info,
    const sagr_code_object_kernel_info_t *kernel,
    const sagr_code_object_remote_info_t *remote,
    sagr_generic_mapping_t *mapping, sagr_generic_mapping_info_t *mapping_info,
    sagr_generic_kernarg_t *kernarg,
    sagr_generic_kernarg_info_t *kernarg_info) {
  sagr_generic_map_options_t map_options;
  sagr_generic_kernarg_allocate_options_t alloc_options;
  sagr_error_info_t error;
  sagr_status_t status;

  memset(&error, 0, sizeof(error));
  status = sagr_generic_map_options_init(&map_options,
                                         (uint32_t)sizeof(map_options));
  if (!require_condition(status == SAGR_STATUS_SUCCESS,
                         "MAP options initialization failed")) {
    return 1;
  }
  map_options.object_id = remote->object_id;
  map_options.object_generation = remote->generation;
  map_options.kernel_index = kernel->index;
  map_options.gfx_target = info->gfx_target;
  map_options.relocation_count = kernel->relocation_count;
  map_options.kernarg_segment_size = kernel->kernarg_segment_size;
  map_options.kernarg_segment_align = kernel->kernarg_segment_align;
  map_options.descriptor_preload_dwords = 0U;
  memcpy(map_options.image_sha256, remote->image_sha256,
         sizeof(map_options.image_sha256));
  memcpy(map_options.kernel_name, SAGR_ENDPOINT_TEST_KERNEL,
         sizeof(SAGR_ENDPOINT_TEST_KERNEL));

  memset(mapping_info, 0, sizeof(*mapping_info));
  status = sagr_generic_map_object(
      instance, &map_options, NULL, mapping, mapping_info,
      (uint32_t)sizeof(*mapping_info), &error, (uint32_t)sizeof(error));
  if (!call_succeeded("generic MAP_OBJECT", status, &error)) {
    return 1;
  }
  if (!require_condition(
          mapping_info->object_id == remote->object_id &&
              mapping_info->object_generation == remote->generation &&
              mapping_info->mapping_id != 0U &&
              mapping_info->mapping_generation != 0U &&
              mapping_info->mapped_base_va != 0U &&
              mapping_info->mapped_end_va > mapping_info->mapped_base_va &&
              mapping_info->descriptor_va >= mapping_info->mapped_base_va &&
              mapping_info->descriptor_va < mapping_info->mapped_end_va &&
              mapping_info->code_va >= mapping_info->mapped_base_va &&
              mapping_info->code_va < mapping_info->mapped_end_va &&
              mapping_info->entry_va >= mapping_info->mapped_base_va &&
              mapping_info->entry_va < mapping_info->mapped_end_va &&
              mapping_info->kernel_index == kernel->index &&
              mapping_info->segment_count == remote->segment_count &&
              mapping_info->descriptor_preload_dwords == 0U &&
              mapping_info->connection_id == owner->connection_id &&
              mapping_info->epoch == owner->epoch &&
              memcmp(mapping_info->daemon_uuid, owner->daemon_uuid,
                     sizeof(mapping_info->daemon_uuid)) == 0 &&
              memcmp(mapping_info->image_sha256, remote->image_sha256,
                     sizeof(mapping_info->image_sha256)) == 0,
          "MAP_OBJECT returned a noncanonical owner-bound mapping")) {
    return 1;
  }

  status = sagr_generic_kernarg_allocate_options_init(
      &alloc_options, (uint32_t)sizeof(alloc_options));
  if (!require_condition(status == SAGR_STATUS_SUCCESS,
                         "ALLOC_KERNARG options initialization failed")) {
    return 1;
  }
  alloc_options.size_bytes = SAGR_ENDPOINT_TEST_KERNARG_ALLOCATION_BYTES;
  alloc_options.alignment_bytes = SAGR_ENDPOINT_TEST_LOGICAL_ALIGNMENT;
  memset(kernarg_info, 0, sizeof(*kernarg_info));
  status = sagr_generic_alloc_kernarg(
      *mapping, &alloc_options, NULL, kernarg, kernarg_info,
      (uint32_t)sizeof(*kernarg_info), &error, (uint32_t)sizeof(error));
  if (!call_succeeded("generic ALLOC_KERNARG", status, &error)) {
    return 1;
  }
  if (!require_condition(
          kernarg_info->object_id == remote->object_id &&
              kernarg_info->object_generation == remote->generation &&
              kernarg_info->mapping_id == mapping_info->mapping_id &&
              kernarg_info->mapping_generation ==
                  mapping_info->mapping_generation &&
              kernarg_info->allocation_id != 0U &&
              kernarg_info->generation != 0U && kernarg_info->kernarg_va != 0U &&
              (kernarg_info->kernarg_va %
                   SAGR_ENDPOINT_TEST_LOGICAL_ALIGNMENT) == 0U &&
              kernarg_info->size_bytes ==
                  SAGR_ENDPOINT_TEST_KERNARG_ALLOCATION_BYTES &&
              kernarg_info->alignment_bytes ==
                  SAGR_ENDPOINT_TEST_LOGICAL_ALIGNMENT &&
              kernarg_info->connection_id == owner->connection_id &&
              kernarg_info->epoch == owner->epoch &&
              memcmp(kernarg_info->daemon_uuid, owner->daemon_uuid,
                     sizeof(kernarg_info->daemon_uuid)) == 0 &&
              memcmp(kernarg_info->image_sha256, remote->image_sha256,
                     sizeof(kernarg_info->image_sha256)) == 0,
          "ALLOC_KERNARG did not preserve the logical 512-byte/align-8 contract")) {
    return 1;
  }

  return 0;
}

static int publish_kernarg(
    sagr_generic_kernarg_t kernarg,
    const uint8_t kernarg_bytes[SAGR_ENDPOINT_TEST_KERNARG_BYTES]) {
  sagr_error_info_t error;
  sagr_status_t status;

  memset(&error, 0, sizeof(error));
  status = sagr_generic_kernarg_copy_from_host(
      kernarg, SAGR_ENDPOINT_TEST_KERNARG_OFFSET, kernarg_bytes,
      SAGR_ENDPOINT_TEST_KERNARG_BYTES, NULL, &error,
      (uint32_t)sizeof(error));
  return call_succeeded("sealed v1 kernarg H2D", status, &error) ? 0 : 1;
}

static int stage_abandoned_owner(
    sagr_instance_t instance, const sagr_instance_info_t *owner,
    const uint8_t *image, size_t image_size,
    const sagr_code_object_info_t *info,
    const sagr_code_object_kernel_info_t *kernel,
    const uint8_t kernarg_bytes[SAGR_ENDPOINT_TEST_KERNARG_BYTES],
    reconnect_snapshot_t *snapshot) {
  sagr_generic_mapping_t mapping = NULL;
  sagr_generic_kernarg_t kernarg = NULL;

  snapshot->owner = *owner;
  if (upload_object(instance, image, image_size, info, kernel,
                    &snapshot->object) != 0 ||
      map_and_alloc(instance, owner, info, kernel, &snapshot->object, &mapping,
                    &snapshot->mapping, &kernarg, &snapshot->kernarg) != 0 ||
      publish_kernarg(kernarg, kernarg_bytes) != 0) {
    return 1;
  }
  /* Deliberately leave mapping and kernarg live.  Instance close must release
   * the local aliases, while EOF makes daemon owner teardown authoritative. */
  return 0;
}

static int create_queue_and_signal(sagr_instance_t instance,
                                   const sagr_instance_info_t *owner,
                                   sagr_queue_t *queue,
                                   sagr_queue_info_t *queue_info,
                                   sagr_signal_t *signal,
                                   sagr_signal_info_t *signal_info) {
  sagr_signal_create_options_t signal_options;
  sagr_error_info_t error;
  sagr_status_t status;

  memset(&error, 0, sizeof(error));
  memset(queue_info, 0, sizeof(*queue_info));
  status = sagr_queue_create(instance, NULL, NULL, queue, queue_info,
                             (uint32_t)sizeof(*queue_info), &error,
                             (uint32_t)sizeof(error));
  if (!call_succeeded("queue CREATE", status, &error)) {
    return 1;
  }
  if (!require_condition(
          queue_info->queue_id != 0U && queue_info->generation != 0U &&
              queue_info->connection_id == owner->connection_id &&
              queue_info->epoch == owner->epoch,
          "queue CREATE returned a noncanonical owner identity")) {
    return 1;
  }

  status = sagr_signal_create_options_init(
      &signal_options, (uint32_t)sizeof(signal_options));
  if (!require_condition(status == SAGR_STATUS_SUCCESS,
                         "signal CREATE options initialization failed")) {
    return 1;
  }
  signal_options.initial_value = 1;
  memset(signal_info, 0, sizeof(*signal_info));
  status = sagr_signal_create(instance, &signal_options, NULL, signal,
                              signal_info, (uint32_t)sizeof(*signal_info),
                              &error, (uint32_t)sizeof(error));
  if (!call_succeeded("signal CREATE", status, &error)) {
    return 1;
  }
  return require_condition(
             signal_info->signal_id != 0U && signal_info->generation != 0U &&
                 signal_info->value == 1 &&
                 signal_info->connection_id == owner->connection_id &&
                 signal_info->epoch == owner->epoch,
             "signal CREATE did not return the required signed-one signal")
             ? 0
             : 1;
}

static int run_positive_lifecycle(
    sagr_instance_t instance, const sagr_instance_info_t *owner,
    const reconnect_snapshot_t *abandoned, const uint8_t *image,
    size_t image_size, const sagr_code_object_info_t *info,
    const sagr_code_object_kernel_info_t *kernel, endpoint_mode_t mode) {
  sagr_code_object_remote_info_t remote;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t queue_info;
  sagr_signal_t signal = NULL;
  sagr_signal_info_t signal_info;
  sagr_generic_mapping_t mapping = NULL;
  sagr_generic_mapping_info_t mapping_info;
  sagr_generic_kernarg_t kernarg = NULL;
  sagr_generic_kernarg_t stale_kernarg;
  sagr_generic_kernarg_info_t kernarg_info;
  sagr_generic_kernarg_info_t stale_info;
  sagr_generic_submit_options_t submit_options;
  sagr_generic_dispatch_ticket_t ticket;
  sagr_generic_dispatch_completion_t completion;
  execution_buffers_t buffers;
  uint8_t execution_kernarg[SAGR_ENDPOINT_TEST_KERNARG_BYTES];
  uint32_t output_crc = 0U;
  sagr_error_info_t error;
  sagr_status_t status;

  if (upload_object(instance, image, image_size, info, kernel, &remote) != 0) {
    return 1;
  }
  if (!require_condition(
          remote.object_id == abandoned->object.object_id &&
              remote.generation > abandoned->object.generation,
          "disconnect cleanup did not recycle the object slot with a new generation")) {
    return 1;
  }
  if (create_queue_and_signal(instance, owner, &queue, &queue_info, &signal,
                              &signal_info) != 0) {
    return 1;
  }
  if (map_and_alloc(instance, owner, info, kernel, &remote, &mapping,
                    &mapping_info, &kernarg, &kernarg_info) != 0) {
    return 1;
  }
  if (!require_condition(
          mapping_info.mapping_id == abandoned->mapping.mapping_id &&
              mapping_info.mapping_generation >
                  abandoned->mapping.mapping_generation &&
              mapping_info.mapped_base_va == abandoned->mapping.mapped_base_va &&
              kernarg_info.allocation_id == abandoned->kernarg.allocation_id &&
              kernarg_info.generation > abandoned->kernarg.generation &&
              kernarg_info.kernarg_va == abandoned->kernarg.kernarg_va,
          "disconnect cleanup did not recycle mapping/allocation slots with new generations")) {
    return 1;
  }
  if (allocate_execution_buffers(instance, owner, &buffers) != 0 ||
      !require_condition(
          pack_execution_kernarg(kernel, buffers.a_info.simulated_va,
                                 buffers.b_info.simulated_va,
                                 buffers.c_info.simulated_va, execution_kernarg),
          "execution kernarg did not match the locked OLD_ABI 280-byte manifest") ||
      publish_kernarg(kernarg, execution_kernarg) != 0) {
    (void)free_execution_buffers(&buffers);
    return 1;
  }

  status = sagr_generic_submit_options_init(
      &submit_options, (uint32_t)sizeof(submit_options));
  if (!require_condition(status == SAGR_STATUS_SUCCESS,
                         "SUBMIT options initialization failed")) {
    return 1;
  }
  submit_options.kernarg_offset = SAGR_ENDPOINT_TEST_KERNARG_OFFSET;
  submit_options.kernarg_size = SAGR_ENDPOINT_TEST_KERNARG_BYTES;
  submit_options.grid_x = 1024U;
  submit_options.grid_y = 1U;
  submit_options.grid_z = 1U;
  submit_options.workgroup_x = 256U;
  submit_options.workgroup_y = 1U;
  submit_options.workgroup_z = 1U;
  submit_options.num_warps = 4U;
  submit_options.num_ctas = 1U;
  submit_options.shared_memory_bytes = kernel->group_segment_fixed_size;
  submit_options.wavefront_size = kernel->wavefront_size;

  memset(&error, 0, sizeof(error));
  memset(&ticket, 0, sizeof(ticket));
  status = sagr_queue_submit_generic_dispatch(
      queue, mapping, kernarg, signal, &submit_options, NULL, &ticket,
      (uint32_t)sizeof(ticket), &error, (uint32_t)sizeof(error));
  if (!call_succeeded("generic SUBMIT_AQL", status, &error)) {
    return 1;
  }
  if (!require_condition(
          ticket.object_id == remote.object_id &&
              ticket.object_generation == remote.generation &&
              ticket.mapping_id == mapping_info.mapping_id &&
              ticket.mapping_generation == mapping_info.mapping_generation &&
              ticket.kernarg_allocation_id == kernarg_info.allocation_id &&
              ticket.kernarg_generation == kernarg_info.generation &&
              ticket.queue_id == queue_info.queue_id &&
              ticket.queue_generation == queue_info.generation &&
              ticket.queue_sequence != 0U &&
              ticket.signal_id == signal_info.signal_id &&
              ticket.signal_generation == signal_info.generation &&
              ticket.ticket_id != 0U && ticket.trace_id != 0U &&
              ticket.packet_va != 0U && (ticket.packet_va % 64U) == 0U &&
              ticket.packet_crc32c != 0U && ticket.admission_tick != 0U &&
              ticket.connection_id == owner->connection_id &&
              ticket.epoch == owner->epoch &&
              memcmp(ticket.image_sha256, remote.image_sha256,
                     sizeof(ticket.image_sha256)) == 0,
          "SUBMIT_AQL ACK did not publish the canonical admission ticket")) {
    return 1;
  }

  if (mode == ENDPOINT_MODE_DISCONNECT_AFTER_ACK) {
    return LIFECYCLE_DISCONNECT_AFTER_ACK;
  }

  memset(&completion, 0, sizeof(completion));
  status = sagr_queue_wait_generic_dispatch(
      queue, &ticket, NULL, &completion, (uint32_t)sizeof(completion), &error,
      (uint32_t)sizeof(error));
  if (!call_succeeded("generic type-20 completion", status, &error)) {
    return 1;
  }
  if (!require_condition(
          completion.status == SAGR_STATUS_SUCCESS &&
              completion.wire_status == SAGR_WIRE_STATUS_OK &&
              completion.request_id == ticket.request_id &&
              completion.object_id == ticket.object_id &&
              completion.object_generation == ticket.object_generation &&
              completion.mapping_id == ticket.mapping_id &&
              completion.mapping_generation == ticket.mapping_generation &&
              completion.kernarg_allocation_id ==
                  ticket.kernarg_allocation_id &&
              completion.kernarg_generation == ticket.kernarg_generation &&
              completion.kernarg_va ==
                  kernarg_info.kernarg_va + SAGR_ENDPOINT_TEST_KERNARG_OFFSET &&
              completion.kernarg_size == SAGR_ENDPOINT_TEST_KERNARG_BYTES &&
              completion.kernarg_alignment ==
                  SAGR_ENDPOINT_TEST_LOGICAL_ALIGNMENT &&
              completion.queue_id == ticket.queue_id &&
              completion.queue_generation == ticket.queue_generation &&
              completion.queue_sequence == ticket.queue_sequence &&
              completion.signal_id == ticket.signal_id &&
              completion.signal_generation == ticket.signal_generation &&
              completion.signal_value_bits == UINT64_C(1) &&
              completion.ticket_id == ticket.ticket_id &&
              completion.trace_id == ticket.trace_id &&
              completion.packet_va == ticket.packet_va &&
              completion.packet_crc32c == ticket.packet_crc32c &&
              completion.output_crc32c == SAGR_ENDPOINT_TEST_OUTPUT_CRC32C &&
              completion.admission_tick == ticket.admission_tick &&
              completion.sim_tick >= completion.admission_tick &&
              completion.start_tick >= completion.admission_tick &&
              completion.end_tick >= completion.start_tick &&
              completion.retire_tick >= completion.end_tick &&
              completion.start_tick != 0U && completion.end_tick != 0U &&
              completion.retire_tick != 0U &&
              completion.connection_id == owner->connection_id &&
              completion.epoch == owner->epoch &&
              memcmp(completion.image_sha256, ticket.image_sha256,
                     sizeof(completion.image_sha256)) == 0,
          "type-20 completion did not preserve the admitted tuple and ticks")) {
    return 1;
  }

  if (!require_condition(verify_execution_buffers(&buffers, &output_crc) == 0,
                         "D2H output oracle did not match gpuReadWrite")) {
    return 1;
  }
  if (!require_condition(output_crc == completion.output_crc32c,
                         "type-20 output CRC did not match D2H B||C")) {
    return 1;
  }
  if (verify_duplicate_a_d2h(&buffers) != 0) {
    return 1;
  }
  if (free_execution_buffers(&buffers) != 0) {
    return 1;
  }

  stale_kernarg = kernarg;
  status = sagr_generic_unmap_object(&mapping, NULL, &error,
                                     (uint32_t)sizeof(error));
  if (!call_succeeded("generic UNMAP_OBJECT", status, &error)) {
    return 1;
  }
  memset(&stale_info, 0, sizeof(stale_info));
  if (!require_condition(
          mapping == NULL &&
              sagr_generic_kernarg_get_info(
                  stale_kernarg, &stale_info, (uint32_t)sizeof(stale_info)) ==
                  SAGR_STATUS_INVALID_HANDLE,
          "UNMAP_OBJECT did not consume the mapping and child kernarg lease")) {
    return 1;
  }
  kernarg = NULL;

  status = sagr_signal_destroy(&signal, NULL, &error,
                               (uint32_t)sizeof(error));
  if (!call_succeeded("signal DESTROY", status, &error)) {
    return 1;
  }
  if (!require_condition(signal == NULL,
                         "signal DESTROY did not consume the public handle")) {
    return 1;
  }
  status = sagr_queue_destroy(&queue, NULL, &error,
                              (uint32_t)sizeof(error));
  if (!call_succeeded("queue DESTROY", status, &error)) {
    return 1;
  }
  return require_condition(queue == NULL,
                           "queue DESTROY did not consume the public handle")
             ? 0
             : 1;
}

static int run_negative_capability_probe(
    const char *endpoint, sagr_instance_open_options_t *control_options,
    sagr_instance_open_options_t *baseline_options, sagr_instance_t *instance,
    sagr_error_info_t *error) {
  sagr_instance_info_t info;
  sagr_status_t status;
  const uint64_t selected_mask =
      SAGR_CAPABILITY_GENERIC_DISPATCH_MASK |
      SAGR_CAPABILITY_GENERIC_EXECUTION_MASK;

  fprintf(stderr,
          "generic endpoint probe: execution capability was canonically rejected\n");
  if (open_options_for_control(control_options) != 0) {
    return 1;
  }
  status = sagr_instance_open(endpoint, control_options, instance, error,
                              (uint32_t)sizeof(*error));
  if (status == SAGR_STATUS_SUCCESS && *instance != NULL) {
    memset(&info, 0, sizeof(info));
    if (get_required_control_info(*instance, &info) != 0 ||
        (info.negotiated_capabilities[0] &
         SAGR_CAPABILITY_GENERIC_EXECUTION_MASK) != 0U ||
        close_instance(instance) != 0) {
      (void)close_instance(instance);
      return 1;
    }
    printf("{\"schema\":\"self-amdgpu-runtime.generic-dispatch-v2-endpoint.v3\","
           "\"handshake\":true,\"required_generic_selected\":true,"
           "\"required_execution_selected\":false,"
           "\"dependencies_selected\":true,"
           "\"bit8_selected\":true,\"bit9_selected\":false,"
           "\"execution_capability_unsupported\":true,"
           "\"canonical_unsupported\":false,\"raw18_sent\":false,"
           "\"execution\":false,\"output_correctness\":false,"
           "\"launcher\":false,\"compiler\":false,\"jit\":false,"
           "\"fallback\":false}\n");
    return 0;
  }
  if (endpoint_unavailable(status)) {
    print_open_error("control handshake after execution rejection", status,
                     error);
    return SAGR_ENDPOINT_TEST_SKIP;
  }
  if (status != SAGR_STATUS_CAPABILITY_MISMATCH ||
      error->wire_status != SAGR_WIRE_STATUS_UNSUPPORTED_CAPABILITY) {
    print_open_error("control handshake after execution rejection", status,
                     error);
    return 1;
  }
  if (open_options_for_baseline(baseline_options) != 0) {
    return 1;
  }
  status = sagr_instance_open(endpoint, baseline_options, instance, error,
                              (uint32_t)sizeof(*error));
  if (endpoint_unavailable(status)) {
    print_open_error("baseline handshake after capability rejection", status,
                     error);
    return SAGR_ENDPOINT_TEST_SKIP;
  }
  if (status != SAGR_STATUS_SUCCESS || *instance == NULL) {
    print_open_error("baseline handshake after capability rejection", status,
                     error);
    return 1;
  }
  memset(&info, 0, sizeof(info));
  status = sagr_instance_get_info(*instance, &info, (uint32_t)sizeof(info));
  if (status != SAGR_STATUS_SUCCESS ||
      (info.negotiated_capabilities[0] & selected_mask) != 0U ||
      close_instance(instance) != 0) {
    if (*instance != NULL) {
      (void)close_instance(instance);
    }
    return 1;
  }
  printf("{\"schema\":\"self-amdgpu-runtime.generic-dispatch-v2-endpoint.v3\","
         "\"handshake\":true,\"required_generic_selected\":false,"
         "\"bit8_selected\":false,\"bit9_selected\":false,"
         "\"canonical_unsupported\":true,"
         "\"execution_capability_unsupported\":true,"
         "\"baseline_reconnect\":true,\"raw18_sent\":false,"
         "\"execution\":false,\"output_correctness\":false,"
         "\"launcher\":false,\"compiler\":false,\"jit\":false,"
         "\"fallback\":false}\n");
  return 0;
}

int main(void) {
  const char *endpoint = getenv("SAGR_GENERIC_BRIDGE_ENDPOINT");
  const char *hsaco_path = getenv(SAGR_ENDPOINT_TEST_HSACO_ENV);
  sagr_instance_open_options_t execution_options;
  sagr_instance_open_options_t control_options;
  sagr_instance_open_options_t baseline_options;
  sagr_instance_t instance = NULL;
  sagr_instance_info_t info;
  sagr_instance_info_t reconnect_info;
  sagr_error_info_t error;
  sagr_status_t status;
  uint8_t *image = NULL;
  size_t image_size = 0U;
  sagr_code_object_info_t code_object;
  sagr_code_object_kernel_info_t kernel;
  uint8_t kernarg_bytes[SAGR_ENDPOINT_TEST_KERNARG_BYTES];
  reconnect_snapshot_t abandoned;
  endpoint_mode_t mode;
  int lifecycle_result;

  if (endpoint == NULL || endpoint[0] == '\0') {
    fprintf(stderr,
            "SKIP: SAGR_GENERIC_BRIDGE_ENDPOINT is unset; no daemon endpoint was probed\n");
    return SAGR_ENDPOINT_TEST_SKIP;
  }
  if (get_endpoint_mode(&mode) != 0) {
    return 1;
  }

  if (open_options_for_execution(&execution_options) != 0) {
    return 1;
  }
  memset(&error, 0, sizeof(error));
  status = sagr_instance_open(endpoint, &execution_options, &instance, &error,
                              (uint32_t)sizeof(error));
  if (status == SAGR_STATUS_CAPABILITY_MISMATCH &&
      error.wire_status == SAGR_WIRE_STATUS_UNSUPPORTED_CAPABILITY) {
    return run_negative_capability_probe(endpoint, &control_options,
                                         &baseline_options, &instance, &error);
  }

  if (endpoint_unavailable(status)) {
    print_open_error("generic handshake", status, &error);
    fprintf(stderr,
            "SKIP: configured endpoint is unavailable; no daemon lifecycle was claimed\n");
    return SAGR_ENDPOINT_TEST_SKIP;
  }
  if (status != SAGR_STATUS_SUCCESS || instance == NULL) {
    print_open_error("generic handshake", status, &error);
    return 1;
  }
  memset(&info, 0, sizeof(info));
  if (get_required_execution_info(instance, &info) != 0) {
    (void)close_instance(&instance);
    return 1;
  }

  if (!read_image(hsaco_path, &image, &image_size)) {
    fprintf(stderr,
            "generic endpoint probe: %s must name a readable gpuReadWrite HSACO for a positive route\n",
            SAGR_ENDPOINT_TEST_HSACO_ENV);
    (void)close_instance(&instance);
    return 1;
  }
  if (validate_fixture(image, image_size, &code_object, &kernel,
                       kernarg_bytes) != 0) {
    free(image);
    (void)close_instance(&instance);
    return 1;
  }

  memset(&abandoned, 0, sizeof(abandoned));
  if (stage_abandoned_owner(instance, &info, image, image_size, &code_object,
                            &kernel, kernarg_bytes, &abandoned) != 0) {
    free(image);
    (void)close_instance(&instance);
    return 1;
  }
  if (close_instance(&instance) != 0) {
    free(image);
    return 1;
  }

  memset(&error, 0, sizeof(error));
  status = sagr_instance_open(endpoint, &execution_options, &instance, &error,
                              (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_SUCCESS || instance == NULL) {
    print_open_error("required-generic reconnect after abandoned owner", status,
                     &error);
    free(image);
    return 1;
  }
  memset(&reconnect_info, 0, sizeof(reconnect_info));
  if (get_required_execution_info(instance, &reconnect_info) != 0 ||
      !require_condition(
          reconnect_info.connection_id != info.connection_id &&
              reconnect_info.epoch == info.epoch &&
              memcmp(reconnect_info.daemon_uuid, info.daemon_uuid,
                     sizeof(info.daemon_uuid)) == 0,
          "reconnect did not establish a new owner on the same daemon epoch")) {
    free(image);
    (void)close_instance(&instance);
    return 1;
  }

  lifecycle_result = run_positive_lifecycle(
      instance, &reconnect_info, &abandoned, image, image_size, &code_object,
      &kernel, mode);
  if (lifecycle_result == LIFECYCLE_DISCONNECT_AFTER_ACK) {
    free(image);
    if (close_instance(&instance) != 0) {
      return 1;
    }
    printf("{\"schema\":\"self-amdgpu-runtime.generic-dispatch-v2-endpoint.v3\","
           "\"mode\":\"disconnect-after-ack\",\"handshake\":true,"
           "\"required_generic_selected\":true,"
           "\"required_execution_selected\":true,"
           "\"dependencies_selected\":true,\"bit8_selected\":true,"
           "\"bit9_selected\":true,\"code_object_upload\":true,"
           "\"queue_signal\":true,\"map\":true,\"alloc\":true,"
           "\"buffer_count\":3,\"buffer_h2d\":true,\"h2d_v1\":true,"
           "\"kernarg_opcode\":false,\"submit_ack\":true,"
           "\"packet_crc_nonzero\":true,\"admission_tick_nonzero\":true,"
           "\"disconnect_after_ack\":true,\"instance_close_success\":true,"
           "\"post_type19_disconnect\":true,\"wait_called\":false,"
           "\"wait_type20_called\":false,\"type20_read_attempted\":false,"
           "\"completion_observed\":false,\"completion_type20\":false,"
           "\"d2h_verified\":false,"
           "\"duplicate_d2h_verified\":false,\"unmap\":false,"
           "\"explicit_resource_destroy\":false,"
           "\"execution_observed\":false,\"execution_claimed\":false,"
           "\"client_output_correctness\":false,"
           "\"output_correctness\":false,"
           "\"remote_cleanup_verified\":false,"
           "\"remote_cleanup_observed\":false,"
           "\"remote_cleanup_unknown\":true,"
           "\"remote_resource_counters_observed\":false,"
           "\"launcher\":false,\"compiler\":false,\"jit\":false,"
           "\"fallback\":false,\"triton\":false,\"qwen\":false}\n");
    return 0;
  }
  if (lifecycle_result != LIFECYCLE_COMPLETE) {
    free(image);
    (void)close_instance(&instance);
    return 1;
  }
  free(image);
  if (close_instance(&instance) != 0) {
    return 1;
  }

  printf("{\"schema\":\"self-amdgpu-runtime.generic-dispatch-v2-endpoint.v3\","
         "\"handshake\":true,\"required_generic_selected\":true,"
         "\"required_execution_selected\":true,"
         "\"dependencies_selected\":true,\"bit8_selected\":true,"
         "\"bit9_selected\":true,\"execution_capability_selected\":true,"
         "\"canonical_unsupported\":false,"
         "\"code_object_upload\":true,\"queue_signal\":true,"
         "\"map\":true,\"alloc\":true,\"logical_alignment_8\":true,"
         "\"backing_alignment_hidden\":true,\"kernarg_allocation_bytes\":512,"
         "\"buffer_allocation_bytes\":4096,\"buffer_count\":3,"
         "\"buffer_owner_bound\":true,"
         "\"buffer_ids_generations_nonzero\":true,"
         "\"buffer_vas_distinct\":true,\"buffer_h2d\":true,"
         "\"kernarg_offset\":64,\"kernarg_manifest_bytes\":280,"
         "\"zero_preload\":true,\"preload_dwords\":0,\"h2d_v1\":true,"
         "\"d2h_v1\":true,\"kernarg_opcode\":false,\"submit_ack\":true,"
         "\"packet_crc_nonzero\":true,\"admission_tick_nonzero\":true,"
         "\"signal_value_bits_expected\":1,"
         "\"signal_after_observed\":false,"
         "\"completion_type20\":true,\"ticks_monotonic\":true,"
         "\"output_crc32c\":1869021807,\"output_crc_nonzero\":true,"
         "\"output_crc_b_then_c\":true,\"input_a_unchanged\":true,"
         "\"output_b_is_gid\":true,\"output_c_is_a\":true,"
         "\"d2h_verified\":true,\"d2h_oracle\":true,"
         "\"duplicate_d2h_verified\":true,\"unmap\":true,"
         "\"disconnect_with_live_leases\":true,"
         "\"reconnect_after_disconnect\":true,\"reconnect_cleanup\":true,"
         "\"remote_cleanup_verified\":true,"
         "\"remote_resource_counters_observed\":false,"
         "\"native_cp_admission\":true,\"native_retire\":true,"
         "\"gpu_dispatcher\":true,\"compute_unit\":true,"
         "\"kernel_executed\":true,\"execution\":true,"
         "\"output_correctness\":true,\"launcher\":false,"
         "\"compiler\":false,\"jit\":false,\"fallback\":false,"
         "\"triton\":false,\"qwen\":false}\n");
  return 0;
}
