/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_KMT_SHIM_H
#define SELF_AMDGPU_RUNTIME_KMT_SHIM_H

#include <stdint.h>

#include <self_amdgpu_runtime/export.h>
#include <self_amdgpu_runtime/generated/bridge_kmt_v5.h>
#include <self_amdgpu_runtime/runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CP-0010 fixed-width KFD/DRM envelope.  No member is a host pointer. */
#define SAGR_KMT_PROTOCOL_MAJOR SAGR_BRIDGE_KMT_PAYLOAD_MAJOR
#define SAGR_KMT_PROTOCOL_MINOR SAGR_BRIDGE_KMT_PAYLOAD_MINOR
#define SAGR_KMT_MESSAGE_REQUEST SAGR_BRIDGE_KMT_MESSAGE_REQUEST
#define SAGR_KMT_MESSAGE_RESULT SAGR_BRIDGE_KMT_MESSAGE_ACK
#define SAGR_KMT_MESSAGE_ACK SAGR_KMT_MESSAGE_RESULT
#define SAGR_KMT_CAPABILITY_WORD UINT32_C(0)
#define SAGR_KMT_CAPABILITY_MASK ((uint64_t)SAGR_BRIDGE_KMT_CAPABILITY_MASK)
#define SAGR_KMT_PAYLOAD_BYTES SAGR_BRIDGE_KMT_REQUEST_PAYLOAD_BYTES
#define SAGR_KMT_BUFFER_BYTES SAGR_BRIDGE_KMT_COPIED_BUFFER_BYTES
#define SAGR_KMT_ARGUMENT_WORD_COUNT SAGR_BRIDGE_KMT_ARGUMENT_WORD_COUNT
#define SAGR_KMT_COPY_HOST_TO_SIM UINT32_C(1)
#define SAGR_KMT_COPY_SIM_TO_HOST UINT32_C(2)
#define SAGR_KMT_PROCESS_APERTURES_PER_PAGE UINT32_C(2)
#define SAGR_KMT_PROCESS_APERTURE_WIRE_BYTES UINT32_C(56)
#define SAGR_KMT_CACHE_POLICY_COHERENT UINT32_C(0)
#define SAGR_KMT_CACHE_POLICY_NONCOHERENT UINT32_C(1)
#define SAGR_KMT_QUEUE_MAX_DEPTH UINT32_C(1048576)

/* Values are kept independent of source enum typedefs so this header can be
 * included by a C shim without importing production ROCr headers. */
typedef int32_t sagr_kmt_status_t;
enum {
  SAGR_KMT_STATUS_SUCCESS = 0,
  SAGR_KMT_STATUS_ERROR = 1,
  SAGR_KMT_STATUS_INVALID_PARAMETER = 3,
  SAGR_KMT_STATUS_INVALID_HANDLE = 4,
  SAGR_KMT_STATUS_INVALID_NODE_UNIT = 5,
  SAGR_KMT_STATUS_NO_MEMORY = 6,
  SAGR_KMT_STATUS_BUFFER_TOO_SMALL = 7,
  SAGR_KMT_STATUS_NOT_IMPLEMENTED = 10,
  SAGR_KMT_STATUS_NOT_SUPPORTED = 11,
  SAGR_KMT_STATUS_UNAVAILABLE = 12,
  SAGR_KMT_STATUS_OUT_OF_RESOURCES = 13,
  SAGR_KMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED = 20,
  SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR = 21,
  SAGR_KMT_STATUS_WAIT_FAILURE = 30,
  SAGR_KMT_STATUS_WAIT_TIMEOUT = 31,
  SAGR_KMT_STATUS_MEMORY_ALIGNMENT = 37
};

typedef enum sagr_kmt_operation {
  SAGR_KMT_OP_OPEN_KFD = SAGR_BRIDGE_KMT_OPERATION_OPEN_KFD,
  SAGR_KMT_OP_CLOSE_KFD = SAGR_BRIDGE_KMT_OPERATION_CLOSE_KFD,
  SAGR_KMT_OP_GET_VERSION = SAGR_BRIDGE_KMT_OPERATION_GET_VERSION,
  SAGR_KMT_OP_TOPOLOGY_SNAPSHOT = SAGR_BRIDGE_KMT_OPERATION_TOPOLOGY_SNAPSHOT,
  SAGR_KMT_OP_ALLOC_MEMORY = SAGR_BRIDGE_KMT_OPERATION_ALLOC_MEMORY,
  SAGR_KMT_OP_FREE_MEMORY = SAGR_BRIDGE_KMT_OPERATION_FREE_MEMORY,
  SAGR_KMT_OP_COPY_MEMORY = SAGR_BRIDGE_KMT_OPERATION_COPY_MEMORY,
  SAGR_KMT_OP_QUEUE_CREATE = SAGR_BRIDGE_KMT_OPERATION_QUEUE_CREATE,
  SAGR_KMT_OP_QUEUE_DESTROY = SAGR_BRIDGE_KMT_OPERATION_QUEUE_DESTROY,
  SAGR_KMT_OP_QUEUE_DOORBELL = SAGR_BRIDGE_KMT_OPERATION_QUEUE_DOORBELL,
  SAGR_KMT_OP_EVENT_CREATE = SAGR_BRIDGE_KMT_OPERATION_EVENT_CREATE,
  SAGR_KMT_OP_EVENT_DESTROY = SAGR_BRIDGE_KMT_OPERATION_EVENT_DESTROY,
  SAGR_KMT_OP_EVENT_SET = SAGR_BRIDGE_KMT_OPERATION_EVENT_SET,
  SAGR_KMT_OP_EVENT_RESET = SAGR_BRIDGE_KMT_OPERATION_EVENT_RESET,
  SAGR_KMT_OP_EVENT_QUERY = SAGR_BRIDGE_KMT_OPERATION_EVENT_QUERY,
  SAGR_KMT_OP_EVENT_WAIT = SAGR_BRIDGE_KMT_OPERATION_EVENT_WAIT,
  SAGR_KMT_OP_POINTER_INFO = SAGR_BRIDGE_KMT_OPERATION_POINTER_INFO,
  SAGR_KMT_OP_MODEL_DRM_CALL = SAGR_BRIDGE_KMT_OPERATION_MODEL_DRM_CALL,
  SAGR_KMT_OP_PROCESS_APERTURES = SAGR_BRIDGE_KMT_OPERATION_PROCESS_APERTURES,
  SAGR_KMT_OP_ACQUIRE_VM = SAGR_BRIDGE_KMT_OPERATION_ACQUIRE_VM,
  SAGR_KMT_OP_SET_MEMORY_POLICY = SAGR_BRIDGE_KMT_OPERATION_SET_MEMORY_POLICY,
  SAGR_KMT_OP_ALLOC_MEMORY_OF_GPU =
      SAGR_BRIDGE_KMT_OPERATION_ALLOC_MEMORY_OF_GPU,
  SAGR_KMT_OP_FREE_MEMORY_OF_GPU =
      SAGR_BRIDGE_KMT_OPERATION_FREE_MEMORY_OF_GPU,
  SAGR_KMT_OP_MAP_MEMORY_TO_GPU =
      SAGR_BRIDGE_KMT_OPERATION_MAP_MEMORY_TO_GPU,
  SAGR_KMT_OP_UNMAP_MEMORY_FROM_GPU =
      SAGR_BRIDGE_KMT_OPERATION_UNMAP_MEMORY_FROM_GPU,
  SAGR_KMT_OP_SET_SCRATCH_BACKING_VA =
      SAGR_BRIDGE_KMT_OPERATION_SET_SCRATCH_BACKING_VA,
  SAGR_KMT_OP_EXPORT_BACKING = SAGR_BRIDGE_KMT_OPERATION_EXPORT_BACKING,
  SAGR_KMT_OP_GET_CLOCK_COUNTERS =
      SAGR_BRIDGE_KMT_OPERATION_GET_CLOCK_COUNTERS
} sagr_kmt_operation_t;

typedef struct sagr_provider sagr_provider_t;

typedef struct sagr_kmt_handle {
  uint64_t owner_id;
  uint64_t owner_generation;
  uint64_t object_id;
  uint64_t object_generation;
} sagr_kmt_handle_t;

typedef struct sagr_kmt_envelope_request {
  uint16_t major;
  uint16_t minor;
  uint16_t operation;
  uint16_t flags;
  uint64_t operation_sequence;
  uint64_t owner_id;
  uint64_t owner_generation;
  uint64_t object_id;
  uint64_t object_generation;
  uint64_t auxiliary_id;
  uint64_t auxiliary_generation;
  uint32_t argument_words[SAGR_KMT_ARGUMENT_WORD_COUNT];
  uint32_t buffer_bytes;
  uint32_t buffer_crc32c;
  uint8_t buffer[SAGR_KMT_BUFFER_BYTES];
  uint8_t reserved[24];
} sagr_kmt_envelope_request_t;

typedef struct sagr_kmt_envelope_result {
  uint16_t major;
  uint16_t minor;
  uint16_t operation;
  uint16_t flags;
  uint32_t status;
  int32_t wire_status;
  uint64_t operation_sequence;
  uint64_t owner_id;
  uint64_t owner_generation;
  uint64_t object_id;
  uint64_t object_generation;
  uint64_t auxiliary_id;
  uint64_t auxiliary_generation;
  uint32_t result_words[SAGR_KMT_ARGUMENT_WORD_COUNT];
  uint32_t buffer_bytes;
  uint32_t buffer_crc32c;
  uint8_t buffer[SAGR_KMT_BUFFER_BYTES];
  uint8_t reserved[16];
} sagr_kmt_envelope_result_t;

typedef struct sagr_kmt_call_options {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t timeout_ns;
  uint64_t absolute_deadline_ns;
  int32_t cancel_fd;
  uint32_t reserved0;
  uint8_t reserved[16];
} sagr_kmt_call_options_t;

typedef struct sagr_kmt_version {
  uint32_t struct_size;
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
  uint32_t flags;
  uint8_t reserved[16];
} sagr_kmt_version_t;

typedef struct sagr_kmt_clock_counters {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t gpu_clock_counter;
  uint64_t cpu_clock_counter;
  uint64_t system_clock_counter;
  uint64_t system_clock_frequency_hz;
  uint8_t reserved[16];
} sagr_kmt_clock_counters_t;

typedef struct sagr_kmt_topology {
  uint32_t struct_size;
  uint32_t flags;
  uint16_t snapshot_major;
  uint16_t snapshot_minor;
  uint16_t model_major;
  uint16_t model_minor;
  uint32_t node_count;
  uint32_t gpu_node_count;
  uint32_t cpu_node_count;
  uint32_t gfx_target_code;
  uint32_t compute_units;
  uint32_t wavefront_size;
  uint32_t page_size;
  uint32_t va_bits;
  uint32_t maximum_queues;
  uint32_t maximum_allocations;
  uint32_t maximum_events;
  uint64_t topology_generation;
  uint8_t reserved[16];
} sagr_kmt_topology_t;

typedef struct sagr_kmt_process_aperture {
  uint64_t lds_base;
  uint64_t lds_limit;
  uint64_t scratch_base;
  uint64_t scratch_limit;
  uint64_t gpuvm_base;
  uint64_t gpuvm_limit;
  uint32_t gpu_id;
  uint32_t reserved0;
} sagr_kmt_process_aperture_t;

typedef struct sagr_kmt_alloc_options {
  uint32_t struct_size;
  uint32_t flags;
  uint32_t node_id;
  uint32_t reserved0;
  uint64_t size_bytes;
  uint64_t alignment_bytes;
  uint64_t memory_flags;
  uint8_t reserved[16];
} sagr_kmt_alloc_options_t;

typedef struct sagr_kmt_memory_info {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t size_bytes;
  uint64_t alignment_bytes;
  uint64_t simulated_gpu_va;
  uint8_t reserved[16];
} sagr_kmt_memory_info_t;

typedef struct sagr_kmt_queue_options {
  uint32_t struct_size;
  uint32_t flags;
  uint32_t node_id;
  uint32_t queue_type;
  uint32_t depth;
  uint32_t reserved0;
  /* Simulated GPU VA of the AQL ring; never a host pointer. */
  uint64_t ring_base_address;
  uint64_t ring_size_bytes;
  uint64_t read_pointer_address;
  uint64_t write_pointer_address;
  uint8_t reserved[16];
} sagr_kmt_queue_options_t;

typedef struct sagr_kmt_pointer_info {
  uint32_t struct_size;
  uint32_t flags;
  sagr_kmt_handle_t allocation;
  uint64_t offset_bytes;
  uint64_t size_bytes;
  uint8_t reserved[16];
} sagr_kmt_pointer_info_t;

typedef struct sagr_kmt_copy_options {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t offset_bytes;
  uint64_t byte_count;
  uint8_t reserved[16];
} sagr_kmt_copy_options_t;

typedef struct sagr_kmt_event_options {
  uint32_t struct_size;
  uint32_t flags;
  int64_t initial_value;
  uint8_t reserved[16];
} sagr_kmt_event_options_t;

typedef struct sagr_kmt_event_result {
  uint32_t struct_size;
  uint32_t flags;
  int64_t value;
  uint32_t ready;
  uint32_t reserved0;
  uint64_t sequence;
  uint8_t reserved[16];
} sagr_kmt_event_result_t;

typedef struct sagr_kmt_model_drm_call {
  uint32_t struct_size;
  uint32_t command;
  uint32_t flags;
  uint32_t argument_bytes;
  uint8_t argument[SAGR_KMT_BUFFER_BYTES];
} sagr_kmt_model_drm_call_t;

SAGR_API sagr_kmt_status_t sagr_kmt_call_options_init(
    sagr_kmt_call_options_t *options, uint32_t options_size);
SAGR_API sagr_kmt_status_t sagr_kmt_envelope_request_init(
    sagr_kmt_envelope_request_t *request, uint32_t request_size,
    uint16_t operation);
SAGR_API sagr_kmt_status_t sagr_kmt_envelope_result_validate(
    const sagr_kmt_envelope_request_t *request,
    const sagr_kmt_envelope_result_t *result);

SAGR_API sagr_kmt_status_t sagr_kmt_open_kfd(
    sagr_provider_t *provider, sagr_kmt_handle_t *out_handle,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_close_kfd(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_get_version(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    sagr_kmt_version_t *out_version, uint32_t version_size,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_get_clock_counters(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    uint32_t gpu_id, sagr_kmt_clock_counters_t *out_counters,
    uint32_t counters_size, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_topology_snapshot(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    sagr_kmt_topology_t *out_topology, uint32_t topology_size,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
/*
 * Returns one canonical page of process apertures.  capacity may be zero for
 * a count-only query and is otherwise bounded by
 * SAGR_KMT_PROCESS_APERTURES_PER_PAGE.  The caller advances start_index by
 * out_returned until it reaches out_total.  Outputs are committed only after
 * the complete response has passed identity, shape and range validation.
 */
SAGR_API sagr_kmt_status_t sagr_kmt_process_apertures(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    uint32_t start_index, sagr_kmt_process_aperture_t *out_apertures,
    uint32_t capacity, uint32_t *out_returned, uint32_t *out_total,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_acquire_vm(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    uint32_t gpu_id, uint32_t render_minor,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_set_memory_policy(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    uint32_t gpu_id, uint32_t default_policy, uint32_t alternate_policy,
    uint32_t misc_process_flags, uint64_t alternate_aperture_base,
    uint64_t alternate_aperture_size, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_alloc_memory_of_gpu(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    uint64_t virtual_address, uint64_t size_bytes, uint32_t gpu_id,
    uint32_t memory_flags, uint64_t mmap_offset,
    sagr_kmt_handle_t *out_memory, uint64_t *out_mmap_offset,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_free_memory_of_gpu(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *memory, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_map_memory_to_gpu(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *memory, const uint32_t *gpu_ids,
    uint32_t gpu_count, uint32_t *out_success,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_unmap_memory_from_gpu(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *memory, const uint32_t *gpu_ids,
    uint32_t gpu_count, uint32_t *out_success,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_set_scratch_backing_va(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    uint32_t gpu_id, uint64_t va_addr,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
/*
 * Transfer one duplicate of a process-owned shared backing memfd to the
 * bridge. The caller retains descriptor ownership. The descriptor must be
 * CLOEXEC, O_RDWR, regular, exact-sized, and sealed against shrink/grow.
 */
SAGR_API sagr_kmt_status_t sagr_kmt_export_backing(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    int backing_fd, uint64_t backing_bytes, uint32_t page_bytes,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_alloc_memory(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_alloc_options_t *alloc, sagr_kmt_handle_t *out_memory,
    sagr_kmt_memory_info_t *out_info, uint32_t info_size,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_free_memory(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *memory, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_copy_memory(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *memory, const sagr_kmt_copy_options_t *copy,
    const void *source, void *destination, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_queue_create(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_queue_options_t *queue, sagr_kmt_handle_t *out_queue,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_queue_destroy(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *queue, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_queue_doorbell(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *queue, uint64_t command_kind, uint64_t *sequence,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_event_create(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_event_options_t *event, sagr_kmt_handle_t *out_event,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_event_destroy(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *event, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_event_set(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *event, int64_t value,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_event_reset(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *event, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_event_query(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *event, sagr_kmt_event_result_t *out_result,
    uint32_t result_size, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_event_wait(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *event, uint64_t condition, int64_t compare_value,
    sagr_kmt_event_result_t *out_result, uint32_t result_size,
    const sagr_kmt_call_options_t *options, sagr_error_info_t *error,
    uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_pointer_info(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_handle_t *memory, sagr_kmt_pointer_info_t *out_info,
    uint32_t info_size, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size);
SAGR_API sagr_kmt_status_t sagr_kmt_model_drm_call(
    sagr_provider_t *provider, const sagr_kmt_handle_t *handle,
    const sagr_kmt_model_drm_call_t *call, void *result,
    uint32_t result_size, const sagr_kmt_call_options_t *options,
    sagr_error_info_t *error, uint32_t error_size);

#ifdef __cplusplus
}
#endif

#endif
