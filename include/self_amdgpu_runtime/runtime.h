/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_RUNTIME_H
#define SELF_AMDGPU_RUNTIME_RUNTIME_H

#include <stdint.h>

#include <self_amdgpu_runtime/export.h>
#include <self_amdgpu_runtime/version.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t sagr_status_t;

enum {
  SAGR_STATUS_SUCCESS = 0,
  SAGR_STATUS_INVALID_ARGUMENT = 1,
  SAGR_STATUS_NOT_SUPPORTED = 2,
  SAGR_STATUS_INTERNAL_ERROR = 3,
  SAGR_STATUS_VERSION_MISMATCH = 4,
  SAGR_STATUS_CAPABILITY_MISMATCH = 5,
  SAGR_STATUS_ENDPOINT_NOT_FOUND = 6,
  SAGR_STATUS_INSTANCE_MISMATCH = 7,
  SAGR_STATUS_TOPOLOGY_MISMATCH = 8,
  SAGR_STATUS_PROTOCOL_ERROR = 9,
  SAGR_STATUS_CHECKSUM_ERROR = 10,
  SAGR_STATUS_TIMED_OUT = 11,
  SAGR_STATUS_UNAVAILABLE = 12,
  SAGR_STATUS_CONNECTION_LOST = 13,
  SAGR_STATUS_OUT_OF_RESOURCES = 14,
  SAGR_STATUS_INVALID_HANDLE = 15,
  SAGR_STATUS_BUFFER_TOO_SMALL = 16,
  SAGR_STATUS_UNAUTHORIZED = 17,
  SAGR_STATUS_BUSY = 18,
  SAGR_STATUS_CANCELLED = 19
};

#define SAGR_UUID_SIZE UINT32_C(16)
#define SAGR_CAPABILITY_WORD_COUNT UINT32_C(4)
#define SAGR_ERROR_MESSAGE_SIZE UINT32_C(128)
#define SAGR_DEFAULT_OPEN_TIMEOUT_NS UINT64_C(5000000000)
#define SAGR_INSTANCE_RANK_WILDCARD UINT32_MAX

#define SAGR_CAPABILITY_TOPOLOGY_WORD UINT32_C(0)
#define SAGR_CAPABILITY_TOPOLOGY_MASK UINT64_C(1)
#define SAGR_CAPABILITY_QUEUE_WORD UINT32_C(0)
#define SAGR_CAPABILITY_QUEUE_MASK UINT64_C(2)
#define SAGR_CAPABILITY_MEMORY_WORD UINT32_C(0)
#define SAGR_CAPABILITY_MEMORY_MASK UINT64_C(4)
#define SAGR_CAPABILITY_SIGNAL_WORD UINT32_C(0)
#define SAGR_CAPABILITY_SIGNAL_MASK UINT64_C(8)
#define SAGR_CAPABILITY_DISPATCH_WORD UINT32_C(0)
#define SAGR_CAPABILITY_DISPATCH_MASK UINT64_C(16)
#define SAGR_CAPABILITY_KMT_WORD UINT32_C(0)
#define SAGR_CAPABILITY_KMT_MASK (UINT64_C(1) << 5)
#define SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_WORD UINT32_C(0)
#define SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK (UINT64_C(1) << 7)
#define SAGR_CAPABILITY_GENERIC_DISPATCH_WORD UINT32_C(0)
#define SAGR_CAPABILITY_GENERIC_DISPATCH_MASK (UINT64_C(1) << 8)
/* CP-0026 selects the real GPU execution adapter only when the existing
 * generic control capability and all of its dependencies are also selected. */
#define SAGR_CAPABILITY_GENERIC_EXECUTION_WORD UINT32_C(0)
#define SAGR_CAPABILITY_GENERIC_EXECUTION_MASK (UINT64_C(1) << 9)

#define SAGR_QUEUE_PROTOCOL_MAJOR UINT16_C(1)
#define SAGR_QUEUE_PROTOCOL_MINOR UINT16_C(0)
#define SAGR_QUEUE_MAX_DEPTH UINT32_C(64)
#define SAGR_QUEUE_MAX_INFLIGHT UINT32_C(8)
#define SAGR_QUEUE_COMMAND_NOOP UINT64_C(0)
#define SAGR_QUEUE_COMMAND_CONTROL_TEST UINT64_C(1)
#define SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST UINT64_C(2)

#define SAGR_MEMORY_PROTOCOL_MAJOR UINT16_C(1)
#define SAGR_MEMORY_PROTOCOL_MINOR UINT16_C(0)
#define SAGR_MEMORY_MAX_LIVE_ALLOCATIONS UINT32_C(1024)
#define SAGR_MEMORY_MAX_SINGLE_ALLOCATION_BYTES UINT64_C(2147483648)
#define SAGR_MEMORY_MAX_TOTAL_LIVE_BYTES UINT64_C(4294967296)
#define SAGR_MEMORY_MAX_TRANSFER_BYTES UINT64_C(16777216)
#define SAGR_MEMORY_ALIGNMENT_4K UINT64_C(4096)
#define SAGR_MEMORY_ALIGNMENT_64K UINT64_C(65536)

#define SAGR_SIGNAL_PROTOCOL_MAJOR UINT16_C(1)
#define SAGR_SIGNAL_PROTOCOL_MINOR UINT16_C(0)
#define SAGR_SIGNAL_MAX_LIVE_SIGNALS UINT32_C(1024)
#define SAGR_SIGNAL_MAX_PENDING_WAITS UINT32_C(8)
#define SAGR_SIGNAL_CONDITION_EQ UINT64_C(0)
#define SAGR_SIGNAL_CONDITION_NE UINT64_C(1)
#define SAGR_SIGNAL_CONDITION_LT UINT64_C(2)
#define SAGR_SIGNAL_CONDITION_GTE UINT64_C(3)

#define SAGR_DISPATCH_PROTOCOL_MAJOR UINT16_C(1)
#define SAGR_DISPATCH_PROTOCOL_MINOR UINT16_C(0)
#define SAGR_DISPATCH_FIXTURE_GFX950_XOR_U8_V1 UINT64_C(1)
#define SAGR_DISPATCH_FIXED_IO_BYTES UINT64_C(64)
#define SAGR_DISPATCH_EXPECTED_SIGNAL_VALUE INT64_C(0)
#define SAGR_DISPATCH_PACKET_CRC32C UINT32_C(0x8a912d83)
#define SAGR_DISPATCH_OUTPUT_CRC32C UINT32_C(0x796671ec)
#define SAGR_DISPATCH_FIXTURE_MANIFEST_SHA256_HEX \
  "7500741873f9d39848e57f0aa9ffc6454df7db87b93e1c046501f54db1b7543c"

/* CP-0022 generic object/dispatch payload version.  The enclosing transport
 * framing remains protocol 1.0; this version is selected only when the
 * GENERIC_DISPATCH_V2 capability is negotiated. */
#define SAGR_GENERIC_DISPATCH_PROTOCOL_MAJOR UINT16_C(2)
#define SAGR_GENERIC_DISPATCH_PROTOCOL_MINOR UINT16_C(0)
#define SAGR_GENERIC_RUNTIME_API_VERSION UINT32_C(1)
#define SAGR_GENERIC_KERNEL_NAME_BYTES UINT32_C(128)
#define SAGR_GENERIC_PAGE_SIZE_4K UINT32_C(4096)
#define SAGR_GENERIC_PAGE_SIZE_64K UINT32_C(65536)
#define SAGR_GENERIC_MAX_KERNARG_BYTES UINT64_C(16777216)
#define SAGR_GENERIC_MAX_SHARED_BYTES UINT32_C(65536)
#define SAGR_GENERIC_MAX_WORKGROUP_DIMENSION UINT32_C(1024)
#define SAGR_GENERIC_MAX_WARPS UINT32_C(32)
#define SAGR_GENERIC_MAX_CTAS UINT32_C(8)

typedef struct sagr_instance *sagr_instance_t;
typedef struct sagr_queue *sagr_queue_t;
typedef struct sagr_memory *sagr_memory_t;
typedef struct sagr_signal *sagr_signal_t;
typedef struct sagr_generic_mapping *sagr_generic_mapping_t;
typedef struct sagr_generic_kernarg *sagr_generic_kernarg_t;

/*
 * All public structures contain fixed-width fields only. Callers initialize
 * options with sagr_instance_open_options_init() and pass the actual allocation
 * size. Reserved fields and flags must remain zero.
 */
typedef struct sagr_instance_open_options {
  uint32_t struct_size;
  uint32_t flags;
  uint16_t minimum_version_major;
  uint16_t minimum_version_minor;
  uint16_t maximum_version_major;
  uint16_t maximum_version_minor;
  uint64_t open_timeout_ns;
  uint64_t offered_capabilities[SAGR_CAPABILITY_WORD_COUNT];
  uint64_t required_capabilities[SAGR_CAPABILITY_WORD_COUNT];
  uint8_t expected_daemon_uuid[SAGR_UUID_SIZE];
  uint8_t expected_job_uuid[SAGR_UUID_SIZE];
  uint64_t expected_epoch;
  uint32_t expected_rank;
  uint32_t expected_world_size;
  /*
   * When nonzero, absolute_deadline_ns is the CLOCK_MONOTONIC deadline and
   * takes precedence over open_timeout_ns. cancel_fd is -1 when disabled;
   * otherwise readability, hangup, or error cancels the open. The caller owns
   * cancel_fd, must set FD_CLOEXEC, and must keep it open until
   * sagr_instance_open() returns. An expired deadline takes precedence over
   * simultaneous cancellation readiness.
   */
  uint64_t absolute_deadline_ns;
  int32_t cancel_fd;
  uint32_t reserved0;
  uint8_t reserved[8];
} sagr_instance_open_options_t;

typedef struct sagr_instance_info {
  uint32_t struct_size;
  uint32_t flags;
  uint16_t selected_version_major;
  uint16_t selected_version_minor;
  uint32_t maximum_record_bytes;
  uint64_t negotiated_capabilities[SAGR_CAPABILITY_WORD_COUNT];
  uint8_t daemon_uuid[SAGR_UUID_SIZE];
  uint8_t job_uuid[SAGR_UUID_SIZE];
  uint64_t connection_id;
  uint64_t epoch;
  uint32_t rank;
  uint32_t world_size;
  uint32_t peer_uid;
  uint32_t peer_pid;
  uint64_t request_id;
  uint8_t reserved[32];
} sagr_instance_info_t;

typedef struct sagr_error_info {
  uint32_t struct_size;
  sagr_status_t status;
  int32_t wire_status;
  int32_t native_errno;
  char message[SAGR_ERROR_MESSAGE_SIZE];
  uint8_t reserved[16];
} sagr_error_info_t;

/* Queue operation options use the same one-deadline/cancellation contract as
 * the handshake. Callers initialize the structure and leave reserved fields
 * zero. A zero timeout selects SAGR_DEFAULT_OPEN_TIMEOUT_NS. */
typedef struct sagr_queue_operation_options {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t timeout_ns;
  uint64_t absolute_deadline_ns;
  int32_t cancel_fd;
  uint32_t reserved0;
  uint8_t reserved[16];
} sagr_queue_operation_options_t;

typedef struct sagr_queue_create_options {
  uint32_t struct_size;
  uint32_t flags;
  uint32_t depth;
  uint32_t reserved0;
  uint8_t reserved[16];
} sagr_queue_create_options_t;

typedef struct sagr_queue_info {
  uint32_t struct_size;
  uint32_t flags;
  uint32_t depth;
  uint32_t reserved0;
  uint64_t queue_id;
  uint64_t generation;
  uint64_t connection_id;
  uint64_t epoch;
  uint8_t daemon_uuid[SAGR_UUID_SIZE];
  uint8_t reserved[16];
} sagr_queue_info_t;

typedef struct sagr_queue_completion {
  uint32_t struct_size;
  uint32_t flags;
  sagr_status_t status;
  int32_t wire_status;
  uint64_t queue_id;
  uint64_t generation;
  uint64_t sequence;
  uint64_t value;
  uint64_t error_code;
  uint64_t sim_tick;
  uint8_t reserved[16];
} sagr_queue_completion_t;

typedef struct sagr_memory_allocate_options {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t size_bytes;
  uint64_t alignment_bytes;
  uint8_t reserved[16];
} sagr_memory_allocate_options_t;

typedef struct sagr_memory_operation_options {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t timeout_ns;
  uint64_t absolute_deadline_ns;
  int32_t cancel_fd;
  uint32_t reserved0;
  uint8_t reserved[16];
} sagr_memory_operation_options_t;

typedef struct sagr_memory_info {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t allocation_id;
  uint64_t generation;
  uint64_t simulated_va;
  uint64_t size_bytes;
  uint64_t alignment_bytes;
  uint64_t connection_id;
  uint64_t epoch;
  uint8_t daemon_uuid[SAGR_UUID_SIZE];
  uint8_t reserved[16];
} sagr_memory_info_t;

typedef struct sagr_signal_create_options {
  uint32_t struct_size;
  uint32_t flags;
  int64_t initial_value;
  uint8_t reserved[16];
} sagr_signal_create_options_t;

typedef struct sagr_signal_operation_options {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t timeout_ns;
  uint64_t absolute_deadline_ns;
  int32_t cancel_fd;
  uint32_t reserved0;
  uint8_t reserved[16];
} sagr_signal_operation_options_t;

typedef struct sagr_signal_info {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t signal_id;
  uint64_t generation;
  int64_t value;
  uint64_t connection_id;
  uint64_t epoch;
  uint8_t daemon_uuid[SAGR_UUID_SIZE];
  uint8_t reserved[16];
} sagr_signal_info_t;

typedef struct sagr_signal_wait_result {
  uint32_t struct_size;
  uint32_t flags;
  sagr_status_t status;
  int32_t wire_status;
  uint64_t signal_id;
  uint64_t generation;
  uint64_t sequence;
  int64_t observed_value;
  uint64_t admission_tick;
  uint64_t completion_tick;
  uint32_t ready_at_admission;
  uint32_t reserved0;
  uint8_t reserved[16];
} sagr_signal_wait_result_t;

/* CP-0008 exposes one protocol-pinned fixture only.  The options contain no
 * packet, pointer, code-object, kernarg, or descriptor fields. */
typedef struct sagr_pinned_dispatch_options {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t fixture_id;
  uint8_t reserved[16];
} sagr_pinned_dispatch_options_t;

/* A ticket is an immutable identity tuple returned only after admission. */
typedef struct sagr_dispatch_ticket {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t request_id;
  uint64_t queue_id;
  uint64_t queue_generation;
  uint64_t queue_sequence;
  uint64_t fixture_id;
  uint64_t input_allocation_id;
  uint64_t input_generation;
  uint64_t output_allocation_id;
  uint64_t output_generation;
  uint64_t signal_id;
  uint64_t signal_generation;
  uint64_t trace_id;
  uint64_t input_gpu_va;
  uint64_t output_gpu_va;
  uint32_t packet_crc32c;
  uint32_t reserved0;
  uint64_t admission_tick;
  uint8_t reserved[16];
} sagr_dispatch_ticket_t;

/* Completion is published atomically after DISPATCH_COMPLETION validation. */
typedef struct sagr_dispatch_completion {
  uint32_t struct_size;
  uint32_t flags;
  sagr_status_t status;
  int32_t wire_status;
  uint64_t request_id;
  uint64_t queue_id;
  uint64_t queue_generation;
  uint64_t queue_sequence;
  uint64_t fixture_id;
  uint64_t input_allocation_id;
  uint64_t input_generation;
  uint64_t output_allocation_id;
  uint64_t output_generation;
  uint64_t signal_id;
  uint64_t signal_generation;
  uint64_t trace_id;
  uint64_t input_gpu_va;
  uint64_t output_gpu_va;
  uint32_t packet_crc32c;
  uint32_t output_crc32c;
  uint64_t admission_tick;
  uint64_t start_tick;
  uint64_t end_tick;
  uint64_t retire_tick;
  uint8_t reserved[16];
} sagr_dispatch_completion_t;

/*
 * Generic payload-v2 mapping options.  These are metadata values copied from
 * a committed code object and its selected kernel; no host pointer or packet
 * bytes are accepted.  The daemon remains authoritative for object ownership
 * and all returned GPU virtual addresses.
 */
typedef struct sagr_generic_map_options {
  uint32_t struct_size;
  uint32_t version;
  uint32_t flags;
  uint64_t object_id;
  uint64_t object_generation;
  uint32_t kernel_index;
  uint32_t gfx_target;
  uint32_t relocation_count;
  uint32_t kernarg_segment_size;
  uint32_t kernarg_segment_align;
  uint32_t descriptor_preload_dwords;
  uint32_t page_size;
  uint8_t image_sha256[32];
  char kernel_name[SAGR_GENERIC_KERNEL_NAME_BYTES];
  uint8_t reserved[16];
} sagr_generic_map_options_t;

typedef struct sagr_generic_mapping_info {
  uint32_t struct_size;
  uint32_t version;
  uint32_t flags;
  uint64_t object_id;
  uint64_t object_generation;
  uint64_t mapping_id;
  uint64_t mapping_generation;
  uint64_t mapped_base_va;
  uint64_t mapped_end_va;
  uint64_t descriptor_va;
  uint64_t code_va;
  uint64_t entry_va;
  uint64_t mapped_bytes;
  uint32_t kernel_index;
  uint32_t segment_count;
  uint32_t descriptor_preload_dwords;
  uint32_t reserved0;
  uint8_t image_sha256[32];
  uint64_t connection_id;
  uint64_t epoch;
  uint8_t daemon_uuid[SAGR_UUID_SIZE];
  uint8_t reserved[16];
} sagr_generic_mapping_info_t;

typedef struct sagr_generic_kernarg_allocate_options {
  uint32_t struct_size;
  uint32_t version;
  uint32_t flags;
  uint64_t size_bytes;
  uint64_t alignment_bytes;
  uint8_t reserved[16];
} sagr_generic_kernarg_allocate_options_t;

typedef struct sagr_generic_kernarg_info {
  uint32_t struct_size;
  uint32_t version;
  uint32_t flags;
  uint64_t object_id;
  uint64_t object_generation;
  uint64_t mapping_id;
  uint64_t mapping_generation;
  uint64_t allocation_id;
  uint64_t generation;
  uint64_t kernarg_va;
  uint64_t size_bytes;
  uint64_t alignment_bytes;
  uint8_t image_sha256[32];
  uint64_t connection_id;
  uint64_t epoch;
  uint8_t daemon_uuid[SAGR_UUID_SIZE];
  uint8_t reserved[16];
} sagr_generic_kernarg_info_t;

typedef struct sagr_generic_submit_options {
  uint32_t struct_size;
  uint32_t version;
  uint32_t flags;
  uint64_t kernarg_offset;
  uint64_t kernarg_size;
  uint64_t expected_signal_value_bits;
  uint32_t grid_x;
  uint32_t grid_y;
  uint32_t grid_z;
  uint32_t workgroup_x;
  uint32_t workgroup_y;
  uint32_t workgroup_z;
  uint32_t num_warps;
  uint32_t num_ctas;
  uint32_t shared_memory_bytes;
  uint32_t wavefront_size;
  uint32_t launch_flags;
  uint32_t reserved0;
  uint8_t reserved[16];
} sagr_generic_submit_options_t;

typedef struct sagr_generic_dispatch_ticket {
  uint32_t struct_size;
  uint32_t version;
  uint32_t flags;
  uint64_t request_id;
  uint64_t object_id;
  uint64_t object_generation;
  uint64_t mapping_id;
  uint64_t mapping_generation;
  uint64_t kernarg_allocation_id;
  uint64_t kernarg_generation;
  uint64_t queue_id;
  uint64_t queue_generation;
  uint64_t queue_sequence;
  uint64_t signal_id;
  uint64_t signal_generation;
  uint64_t ticket_id;
  uint64_t trace_id;
  uint64_t packet_va;
  uint32_t packet_crc32c;
  uint32_t reserved0;
  uint64_t admission_tick;
  uint8_t image_sha256[32];
  uint64_t connection_id;
  uint64_t epoch;
  uint8_t daemon_uuid[SAGR_UUID_SIZE];
  uint8_t reserved[16];
} sagr_generic_dispatch_ticket_t;

typedef struct sagr_generic_dispatch_completion {
  uint32_t struct_size;
  uint32_t version;
  uint32_t flags;
  sagr_status_t status;
  int32_t wire_status;
  uint64_t request_id;
  uint64_t object_id;
  uint64_t object_generation;
  uint64_t mapping_id;
  uint64_t mapping_generation;
  uint64_t kernarg_allocation_id;
  uint64_t kernarg_generation;
  uint64_t kernarg_va;
  uint64_t kernarg_size;
  uint64_t kernarg_alignment;
  uint64_t queue_id;
  uint64_t queue_generation;
  uint64_t queue_sequence;
  uint64_t signal_id;
  uint64_t signal_generation;
  uint64_t signal_value_bits;
  uint64_t ticket_id;
  uint64_t trace_id;
  uint64_t packet_va;
  uint32_t packet_crc32c;
  uint32_t output_crc32c;
  uint64_t sim_tick;
  uint64_t admission_tick;
  uint64_t start_tick;
  uint64_t end_tick;
  uint64_t retire_tick;
  uint8_t image_sha256[32];
  uint64_t connection_id;
  uint64_t epoch;
  uint8_t daemon_uuid[SAGR_UUID_SIZE];
  uint8_t reserved[16];
} sagr_generic_dispatch_completion_t;

SAGR_API uint32_t sagr_abi_version(void);
SAGR_API const char *sagr_version_string(void);
SAGR_API const char *sagr_status_string(sagr_status_t status);

/* Set protocol 1.0, a five-second relative deadline, and no cancellation FD. */
SAGR_API sagr_status_t sagr_instance_open_options_init(
    sagr_instance_open_options_t *options, uint32_t options_size);

SAGR_API sagr_status_t sagr_queue_operation_options_init(
    sagr_queue_operation_options_t *options, uint32_t options_size);
SAGR_API sagr_status_t sagr_queue_create_options_init(
    sagr_queue_create_options_t *options, uint32_t options_size);
SAGR_API sagr_status_t sagr_memory_allocate_options_init(
    sagr_memory_allocate_options_t *options, uint32_t options_size);
SAGR_API sagr_status_t sagr_memory_operation_options_init(
    sagr_memory_operation_options_t *options, uint32_t options_size);
SAGR_API sagr_status_t sagr_signal_create_options_init(
    sagr_signal_create_options_t *options, uint32_t options_size);
SAGR_API sagr_status_t sagr_signal_operation_options_init(
    sagr_signal_operation_options_t *options, uint32_t options_size);
SAGR_API sagr_status_t sagr_pinned_dispatch_options_init(
    sagr_pinned_dispatch_options_t *options, uint32_t options_size);
SAGR_API sagr_status_t sagr_generic_map_options_init(
    sagr_generic_map_options_t *options, uint32_t options_size);
SAGR_API sagr_status_t sagr_generic_kernarg_allocate_options_init(
    sagr_generic_kernarg_allocate_options_t *options, uint32_t options_size);
SAGR_API sagr_status_t sagr_generic_submit_options_init(
    sagr_generic_submit_options_t *options, uint32_t options_size);

/*
 * Open performs exactly one AF_UNIX SOCK_SEQPACKET handshake attempt. When no
 * absolute deadline is supplied, a zero relative timeout selects the
 * five-second default and UINT64_MAX means no deadline. On every failure, a
 * non-null out_instance is set to NULL. native_errno is diagnostic only; the
 * returned sagr_status_t is authoritative.
 */
SAGR_API sagr_status_t sagr_instance_open(
    const char *endpoint, const sagr_instance_open_options_t *options,
    sagr_instance_t *out_instance, sagr_error_info_t *out_error,
    uint32_t error_size);
SAGR_API sagr_status_t sagr_instance_get_info(
    sagr_instance_t instance, sagr_instance_info_t *info,
    uint32_t info_size);
/*
 * Handles have unique ownership. Copying aliases is unsupported; after close,
 * no alias of the instance or its queues may be passed to any API. Close is
 * local-only, idempotent for *instance == NULL, and clears on success.
 */
SAGR_API sagr_status_t sagr_instance_close(sagr_instance_t *instance);

/*
 * The caller must serialize queue APIs on each instance; concurrent calls on
 * the same instance are not thread-safe. They exchange control metadata only:
 * no host pointer, descriptor, memory payload, packet, or kernel is accepted.
 * A successful doorbell returns a sequence token; wait consumes its matching
 * asynchronous completion. A wait timeout or cancellation keeps that
 * completion pending so wait may be retried. Failure after a request is sent
 * but before a canonical ACK poisons the queue transport; later queue calls
 * return CONNECTION_LOST and the caller must close the instance. A queue
 * handle has unique ownership; copied aliases must not be passed after
 * successful destroy or instance close.
 */
SAGR_API sagr_status_t sagr_queue_create(
    sagr_instance_t instance, const sagr_queue_create_options_t *options,
    const sagr_queue_operation_options_t *operation_options,
    sagr_queue_t *out_queue, sagr_queue_info_t *out_info,
    uint32_t info_size, sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_queue_ring_doorbell(
    sagr_queue_t queue, uint64_t command_kind,
    const sagr_queue_operation_options_t *operation_options,
    uint64_t *out_sequence, sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_queue_wait(
    sagr_queue_t queue, uint64_t sequence,
    const sagr_queue_operation_options_t *operation_options,
    sagr_queue_completion_t *out_completion, uint32_t completion_size,
    sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_queue_destroy(
    sagr_queue_t *queue,
    const sagr_queue_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size);

/*
 * Submit and wait are deliberately separate. Submit admits one fixed fixture
 * and returns a generation-safe ticket; wait consumes the matching completion.
 * A wait timeout/cancellation retains the ticket and never resends the
 * dispatch request. Queue, input/output memory, and signal handles must all
 * belong to the same instance. The fixture requires two distinct allocations
 * of at least 64 bytes and a live signed-one signal with an armed EQ-zero wait.
 */
SAGR_API sagr_status_t sagr_queue_submit_pinned_dispatch(
    sagr_queue_t queue, sagr_memory_t input_memory,
    sagr_memory_t output_memory, sagr_signal_t completion_signal,
    const sagr_pinned_dispatch_options_t *options,
    const sagr_queue_operation_options_t *operation_options,
    sagr_dispatch_ticket_t *out_ticket, uint32_t ticket_size,
    sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_queue_wait_pinned_dispatch(
    sagr_queue_t queue, const sagr_dispatch_ticket_t *ticket,
    const sagr_queue_operation_options_t *operation_options,
    sagr_dispatch_completion_t *out_completion, uint32_t completion_size,
    sagr_error_info_t *out_error, uint32_t error_size);

/*
 * Generic payload-v2 stages are synchronous and caller-serialized on the
 * owning instance.  A successful MAP creates an owner-bound mapping lease;
 * ALLOC creates a child kernarg lease; SUBMIT returns an admission ticket and
 * WAIT consumes its matching completion.  No client packet bytes or host
 * pointers cross the wire.  UNMAP consumes the mapping and its child leases
 * only after no submission remains pending.  Instance close is local-only:
 * it drops local lease state after closing the transport and does not claim a
 * remote UNMAP or execution result; daemon owner/epoch teardown is
 * authoritative for abandoned remote leases.
 */
SAGR_API sagr_status_t sagr_generic_map_object(
    sagr_instance_t instance, const sagr_generic_map_options_t *options,
    const sagr_queue_operation_options_t *operation_options,
    sagr_generic_mapping_t *out_mapping, sagr_generic_mapping_info_t *out_info,
    uint32_t info_size, sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_generic_mapping_get_info(
    sagr_generic_mapping_t mapping, sagr_generic_mapping_info_t *out_info,
    uint32_t info_size);
SAGR_API sagr_status_t sagr_generic_alloc_kernarg(
    sagr_generic_mapping_t mapping,
    const sagr_generic_kernarg_allocate_options_t *options,
    const sagr_queue_operation_options_t *operation_options,
    sagr_generic_kernarg_t *out_kernarg,
    sagr_generic_kernarg_info_t *out_info, uint32_t info_size,
    sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_generic_kernarg_get_info(
    sagr_generic_kernarg_t kernarg, sagr_generic_kernarg_info_t *out_info,
    uint32_t info_size);
/* Publishes bytes through the existing sealed v1 MEMORY_COPY_H2D carrier.
 * The destination is the daemon-owned allocation returned by ALLOC_KERNARG;
 * no host pointer is serialized and the daemon must echo the range and CRC. */
SAGR_API sagr_status_t sagr_generic_kernarg_copy_from_host(
    sagr_generic_kernarg_t kernarg, uint64_t offset, const void *source,
    uint64_t byte_count,
    const sagr_queue_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_queue_submit_generic_dispatch(
    sagr_queue_t queue, sagr_generic_mapping_t mapping,
    sagr_generic_kernarg_t kernarg, sagr_signal_t signal,
    const sagr_generic_submit_options_t *options,
    const sagr_queue_operation_options_t *operation_options,
    sagr_generic_dispatch_ticket_t *out_ticket, uint32_t ticket_size,
    sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_queue_wait_generic_dispatch(
    sagr_queue_t queue, const sagr_generic_dispatch_ticket_t *ticket,
    const sagr_queue_operation_options_t *operation_options,
    sagr_generic_dispatch_completion_t *out_completion,
    uint32_t completion_size, sagr_error_info_t *out_error,
    uint32_t error_size);
SAGR_API sagr_status_t sagr_generic_unmap_object(
    sagr_generic_mapping_t *mapping,
    const sagr_queue_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size);

/*
 * Memory operations are synchronous and caller-serialized with queue APIs on
 * the owning instance. Host pointers are copied through sealed memfd staging
 * and never appear on the wire. A complete send followed by failure before a
 * canonical terminal ACK poisons the transport. Allocation handles have
 * unique ownership; copied aliases must not be used after free or close.
 * COPY_D2H does not modify the caller buffer until final carrier validation,
 * private reread, CRC verification, and its post-ACK deadline/cancel check
 * succeed. Observable carrier mismatches poison; local scratch allocation or
 * reread failure and post-ACK timeout/cancellation leave the known session
 * reusable.
 */
SAGR_API sagr_status_t sagr_memory_allocate(
    sagr_instance_t instance, const sagr_memory_allocate_options_t *options,
    const sagr_memory_operation_options_t *operation_options,
    sagr_memory_t *out_memory, sagr_memory_info_t *out_info,
    uint32_t info_size, sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_memory_get_info(
    sagr_memory_t memory, sagr_memory_info_t *info, uint32_t info_size);
SAGR_API sagr_status_t sagr_memory_copy_from_host(
    sagr_memory_t memory, uint64_t offset, const void *source,
    uint64_t byte_count,
    const sagr_memory_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_memory_copy_to_host(
    sagr_memory_t memory, uint64_t offset, void *destination,
    uint64_t byte_count,
    const sagr_memory_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_memory_free(
    sagr_memory_t *memory,
    const sagr_memory_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size);

/*
 * Signal operations are caller-serialized with queue and memory APIs on the
 * owning instance. A first wait sends one WAIT request and, after its canonical
 * ACK, retains exactly one pending predicate on the signal. Timeout or
 * cancellation while awaiting completion is retryable: an identical wait
 * resumes locally without sending another request. A different predicate or
 * destroy is locally BUSY until the completion is consumed. Signal handles
 * have unique ownership; copied aliases must not be used after destroy or
 * instance close.
 */
SAGR_API sagr_status_t sagr_signal_create(
    sagr_instance_t instance, const sagr_signal_create_options_t *options,
    const sagr_signal_operation_options_t *operation_options,
    sagr_signal_t *out_signal, sagr_signal_info_t *out_info,
    uint32_t info_size, sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_signal_get_info(
    sagr_signal_t signal, sagr_signal_info_t *info, uint32_t info_size);
SAGR_API sagr_status_t sagr_signal_load(
    sagr_signal_t signal,
    const sagr_signal_operation_options_t *operation_options,
    int64_t *out_value, sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_signal_store(
    sagr_signal_t signal, int64_t value,
    const sagr_signal_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_signal_wait(
    sagr_signal_t signal, uint64_t condition, int64_t compare_value,
    const sagr_signal_operation_options_t *operation_options,
    sagr_signal_wait_result_t *out_result, uint32_t result_size,
    sagr_error_info_t *out_error, uint32_t error_size);
SAGR_API sagr_status_t sagr_signal_destroy(
    sagr_signal_t *signal,
    const sagr_signal_operation_options_t *operation_options,
    sagr_error_info_t *out_error, uint32_t error_size);

#ifdef __cplusplus
}
#endif

#endif
