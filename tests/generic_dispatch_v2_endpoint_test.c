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
#define SAGR_ENDPOINT_TEST_KERNEL "gpuReadWrite"
#define SAGR_ENDPOINT_TEST_KERNARG_BYTES UINT32_C(280)
#define SAGR_ENDPOINT_TEST_ALLOCATION_BYTES UINT64_C(512)
#define SAGR_ENDPOINT_TEST_KERNARG_OFFSET UINT64_C(64)
#define SAGR_ENDPOINT_TEST_LOGICAL_ALIGNMENT UINT64_C(8)

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

static uint64_t generic_dependency_mask(void) {
  return SAGR_CAPABILITY_TOPOLOGY_MASK | SAGR_CAPABILITY_QUEUE_MASK |
         SAGR_CAPABILITY_MEMORY_MASK | SAGR_CAPABILITY_SIGNAL_MASK |
         SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK |
         SAGR_CAPABILITY_GENERIC_DISPATCH_MASK;
}

static int open_options_for_generic(
    sagr_instance_open_options_t *options) {
  const sagr_status_t status = sagr_instance_open_options_init(
      options, (uint32_t)sizeof(*options));
  if (status != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "generic endpoint probe: options init failed: %s\n",
            sagr_status_string(status));
    return 1;
  }
  /* GENERIC_DISPATCH_V2 is a required selection here.  The runtime API
   * intentionally does not model optional capability offers; dependencies
   * are required as mandated by the v2 handshake contract. */
  options->offered_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] |=
      generic_dependency_mask();
  options->required_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] |=
      generic_dependency_mask();
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

static int get_required_generic_info(sagr_instance_t instance,
                                     sagr_instance_info_t *info) {
  const uint64_t dependencies = generic_dependency_mask();
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
            "generic endpoint probe: successful handshake did not select bit 8 and all dependencies\n");
    return 1;
  }
  return 0;
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

static int map_alloc_and_publish(
    sagr_instance_t instance, const sagr_instance_info_t *owner,
    const sagr_code_object_info_t *info,
    const sagr_code_object_kernel_info_t *kernel,
    const sagr_code_object_remote_info_t *remote,
    const uint8_t kernarg_bytes[SAGR_ENDPOINT_TEST_KERNARG_BYTES],
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
  alloc_options.size_bytes = SAGR_ENDPOINT_TEST_ALLOCATION_BYTES;
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
              kernarg_info->size_bytes == SAGR_ENDPOINT_TEST_ALLOCATION_BYTES &&
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

  status = sagr_generic_kernarg_copy_from_host(
      *kernarg, SAGR_ENDPOINT_TEST_KERNARG_OFFSET, kernarg_bytes,
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
      map_alloc_and_publish(instance, owner, info, kernel, &snapshot->object,
                            kernarg_bytes, &mapping, &snapshot->mapping,
                            &kernarg, &snapshot->kernarg) != 0) {
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
    const sagr_code_object_kernel_info_t *kernel,
    const uint8_t kernarg_bytes[SAGR_ENDPOINT_TEST_KERNARG_BYTES]) {
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
  if (map_alloc_and_publish(instance, owner, info, kernel, &remote,
                            kernarg_bytes, &mapping, &mapping_info, &kernarg,
                            &kernarg_info) != 0) {
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
              completion.output_crc32c == 0U &&
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

int main(void) {
  const char *endpoint = getenv("SAGR_GENERIC_BRIDGE_ENDPOINT");
  const char *hsaco_path = getenv(SAGR_ENDPOINT_TEST_HSACO_ENV);
  sagr_instance_open_options_t generic_options;
  sagr_instance_open_options_t baseline_options;
  sagr_instance_t instance = NULL;
  sagr_instance_info_t info;
  sagr_instance_info_t reconnect_info;
  sagr_error_info_t error;
  sagr_status_t status;
  const uint64_t generic_mask = SAGR_CAPABILITY_GENERIC_DISPATCH_MASK;
  uint8_t *image = NULL;
  size_t image_size = 0U;
  sagr_code_object_info_t code_object;
  sagr_code_object_kernel_info_t kernel;
  uint8_t kernarg_bytes[SAGR_ENDPOINT_TEST_KERNARG_BYTES];
  reconnect_snapshot_t abandoned;

  if (endpoint == NULL || endpoint[0] == '\0') {
    fprintf(stderr,
            "SKIP: SAGR_GENERIC_BRIDGE_ENDPOINT is unset; no daemon endpoint was probed\n");
    return SAGR_ENDPOINT_TEST_SKIP;
  }

  if (open_options_for_generic(&generic_options) != 0) {
    return 1;
  }
  memset(&error, 0, sizeof(error));
  status = sagr_instance_open(endpoint, &generic_options, &instance, &error,
                              (uint32_t)sizeof(error));
  if (status == SAGR_STATUS_CAPABILITY_MISMATCH &&
      error.wire_status == SAGR_WIRE_STATUS_UNSUPPORTED_CAPABILITY) {
    /* A reachable daemon that has not advertised bit 8 remains the canonical
     * fail-closed boundary.  This public-API branch rejects the required hello;
     * it does not claim that a raw type-18 record was sent. */
    fprintf(stderr,
            "generic endpoint probe: endpoint reachable; generic capability was canonically rejected\n");
    if (open_options_for_baseline(&baseline_options) != 0) {
      return 1;
    }
    status = sagr_instance_open(endpoint, &baseline_options, &instance, &error,
                                (uint32_t)sizeof(error));
    if (endpoint_unavailable(status)) {
      print_open_error("baseline handshake after capability rejection",
                       status, &error);
      return SAGR_ENDPOINT_TEST_SKIP;
    }
    if (status != SAGR_STATUS_SUCCESS) {
      print_open_error("baseline handshake after capability rejection", status,
                       &error);
      return 1;
    }
    memset(&info, 0, sizeof(info));
    status = sagr_instance_get_info(instance, &info, (uint32_t)sizeof(info));
    if (status != SAGR_STATUS_SUCCESS) {
      fprintf(stderr, "generic endpoint probe: get_info failed: %s\n",
              sagr_status_string(status));
      (void)close_instance(&instance);
      return 1;
    }
    if ((info.negotiated_capabilities[SAGR_CAPABILITY_GENERIC_DISPATCH_WORD] &
         generic_mask) != 0U) {
      fprintf(stderr,
              "generic endpoint probe: baseline handshake unexpectedly selected generic capability\n");
      (void)close_instance(&instance);
      return 1;
    }
    if (close_instance(&instance) != 0) {
      return 1;
    }
    printf("{\"schema\":\"self-amdgpu-runtime.generic-dispatch-v2-endpoint.v2\","
           "\"handshake\":true,\"required_generic_selected\":false,"
           "\"bit8_selected\":false,\"canonical_unsupported\":true,"
           "\"baseline_reconnect\":true,\"raw18_sent\":false,"
           "\"execution\":false,\"output_correctness\":false,"
           "\"launcher\":false,\"compiler\":false,\"jit\":false,"
           "\"fallback\":false}\n");
    return 0;
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
  if (get_required_generic_info(instance, &info) != 0) {
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
  status = sagr_instance_open(endpoint, &generic_options, &instance, &error,
                              (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_SUCCESS || instance == NULL) {
    print_open_error("required-generic reconnect after abandoned owner", status,
                     &error);
    free(image);
    return 1;
  }
  memset(&reconnect_info, 0, sizeof(reconnect_info));
  if (get_required_generic_info(instance, &reconnect_info) != 0 ||
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

  if (run_positive_lifecycle(instance, &reconnect_info, &abandoned, image,
                             image_size, &code_object, &kernel,
                             kernarg_bytes) != 0) {
    free(image);
    (void)close_instance(&instance);
    return 1;
  }
  free(image);
  if (close_instance(&instance) != 0) {
    return 1;
  }

  printf("{\"schema\":\"self-amdgpu-runtime.generic-dispatch-v2-endpoint.v2\","
         "\"handshake\":true,\"required_generic_selected\":true,"
         "\"dependencies_selected\":true,\"bit8_selected\":true,"
         "\"canonical_unsupported\":false,\"code_object_upload\":true,"
         "\"queue_signal\":true,\"map\":true,\"alloc\":true,"
         "\"logical_alignment_8\":true,\"backing_alignment_hidden\":true,"
         "\"allocation_bytes\":512,\"kernarg_offset\":64,"
         "\"kernarg_manifest_bytes\":280,\"h2d_v1\":true,"
         "\"kernarg_opcode\":false,\"submit_ack\":true,"
         "\"packet_crc_nonzero\":true,\"admission_tick_nonzero\":true,"
         "\"completion_type20\":true,\"ticks_monotonic\":true,"
         "\"unmap\":true,\"disconnect_with_live_leases\":true,"
         "\"reconnect_after_disconnect\":true,"
         "\"reconnect_cleanup\":true,\"remote_cleanup_verified\":true,"
         "\"remote_resource_counters_observed\":false,"
         "\"preload_dwords\":0,"
         "\"native_cp_admission\":true,\"native_retire\":true,"
         "\"gpu_dispatcher\":false,\"compute_unit\":false,"
         "\"kernel_executed\":false,"
         "\"execution\":false,\"output_correctness\":false,"
         "\"launcher\":false,\"compiler\":false,\"jit\":false,"
         "\"fallback\":false,\"triton\":false,\"qwen\":false}\n");
  return 0;
}
