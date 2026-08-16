/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_TRANSPORT_INTERNAL_H
#define SELF_AMDGPU_RUNTIME_TRANSPORT_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include <self_amdgpu_runtime/runtime.h>
#include <self_amdgpu_runtime/kmt_shim.h>

enum {
  SAGR_WIRE_HEADER_BYTES = SAGR_BRIDGE_GENERIC_V2_HEADER_BYTES,
  SAGR_WIRE_HELLO_FIXED_BYTES = 96,
  SAGR_WIRE_ACK_FIXED_BYTES = 80,
  SAGR_WIRE_TOPOLOGY_TLV_BYTES = 32,
  SAGR_WIRE_HELLO_FRAME_BYTES = 208,
  SAGR_WIRE_ACK_FRAME_BYTES = 192,
  SAGR_WIRE_MAX_RECORD_BYTES = 65536,
  SAGR_WIRE_MAX_HANDSHAKE_BYTES = 4096,
  SAGR_WIRE_CAPABILITY_BYTES = 32,
  SAGR_WIRE_QUEUE_PAYLOAD_BYTES = 64,
  SAGR_WIRE_QUEUE_FRAME_BYTES =
      SAGR_WIRE_HEADER_BYTES + SAGR_WIRE_QUEUE_PAYLOAD_BYTES,
  SAGR_WIRE_MEMORY_PAYLOAD_BYTES = 64,
  SAGR_WIRE_MEMORY_FRAME_BYTES =
      SAGR_WIRE_HEADER_BYTES + SAGR_WIRE_MEMORY_PAYLOAD_BYTES,
  SAGR_WIRE_SIGNAL_PAYLOAD_BYTES = 64,
  SAGR_WIRE_SIGNAL_FRAME_BYTES =
      SAGR_WIRE_HEADER_BYTES + SAGR_WIRE_SIGNAL_PAYLOAD_BYTES,
  SAGR_WIRE_DISPATCH_REQUEST_PAYLOAD_BYTES = 128,
  SAGR_WIRE_DISPATCH_RESULT_PAYLOAD_BYTES = 160,
  SAGR_WIRE_DISPATCH_REQUEST_FRAME_BYTES =
      SAGR_WIRE_HEADER_BYTES + SAGR_WIRE_DISPATCH_REQUEST_PAYLOAD_BYTES,
  SAGR_WIRE_DISPATCH_RESULT_FRAME_BYTES =
      SAGR_WIRE_HEADER_BYTES + SAGR_WIRE_DISPATCH_RESULT_PAYLOAD_BYTES,
  SAGR_WIRE_KMT_FRAME_BYTES =
      SAGR_WIRE_HEADER_BYTES + SAGR_KMT_PAYLOAD_BYTES,
  SAGR_WIRE_CODE_OBJECT_PAYLOAD_BYTES = 4016,
  SAGR_WIRE_CODE_OBJECT_FRAME_BYTES =
      SAGR_WIRE_HEADER_BYTES + SAGR_WIRE_CODE_OBJECT_PAYLOAD_BYTES,
  SAGR_WIRE_CODE_OBJECT_PREFIX_BYTES = 48,
  SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES = 3968,
  SAGR_WIRE_CODE_OBJECT_MAX_SEGMENTS = 16,
  SAGR_WIRE_CODE_OBJECT_NAME_BYTES = 128,
  SAGR_WIRE_CODE_OBJECT_DESCRIPTOR_BYTES = 64,
  SAGR_WIRE_CODE_OBJECT_MAX_IMAGE_BYTES = 67108864,
  SAGR_WIRE_GENERIC_PAYLOAD_BYTES = SAGR_BRIDGE_GENERIC_V2_PAYLOAD_BYTES,
  SAGR_WIRE_GENERIC_FRAME_BYTES =
      SAGR_BRIDGE_GENERIC_V2_RECORD_BYTES,
  SAGR_WIRE_GENERIC_COMMON_BYTES = SAGR_BRIDGE_GENERIC_V2_REQUEST_COMMON_BYTES,
  SAGR_WIRE_GENERIC_KERNEL_NAME_BYTES =
      SAGR_BRIDGE_GENERIC_V2_KERNEL_NAME_BYTES,
  SAGR_WIRE_GENERIC_MAX_KERNARG_BYTES =
      SAGR_BRIDGE_GENERIC_V2_MAX_KERNARG_BYTES,
  SAGR_WIRE_GENERIC_MAX_SHARED_BYTES =
      SAGR_BRIDGE_GENERIC_V2_MAX_SHARED_BYTES,
  SAGR_WIRE_GENERIC_MAX_WORKGROUP_DIMENSION =
      SAGR_BRIDGE_GENERIC_V2_MAX_WORKGROUP_DIMENSION,
  SAGR_WIRE_GENERIC_MAX_WARPS = SAGR_BRIDGE_GENERIC_V2_MAX_WARPS,
  SAGR_WIRE_GENERIC_MAX_CTAS = SAGR_BRIDGE_GENERIC_V2_MAX_CTAS,
  SAGR_WIRE_GENERIC_MAX_PRELOAD_DWORDS = 64
};

enum {
  SAGR_WIRE_STATUS_OK = SAGR_BRIDGE_GENERIC_V2_STATUS_OK,
  SAGR_WIRE_STATUS_MALFORMED = SAGR_BRIDGE_GENERIC_V2_STATUS_MALFORMED,
  SAGR_WIRE_STATUS_UNSUPPORTED_VERSION =
      SAGR_BRIDGE_GENERIC_V2_STATUS_UNSUPPORTED_VERSION,
  SAGR_WIRE_STATUS_UNSUPPORTED_CAPABILITY =
      SAGR_BRIDGE_GENERIC_V2_STATUS_UNSUPPORTED_CAPABILITY,
  SAGR_WIRE_STATUS_INSTANCE_MISMATCH =
      SAGR_BRIDGE_GENERIC_V2_STATUS_INSTANCE_MISMATCH,
  SAGR_WIRE_STATUS_TOPOLOGY_MISMATCH =
      SAGR_BRIDGE_GENERIC_V2_STATUS_TOPOLOGY_MISMATCH,
  SAGR_WIRE_STATUS_UNAUTHORIZED = SAGR_BRIDGE_GENERIC_V2_STATUS_UNAUTHORIZED,
  SAGR_WIRE_STATUS_BUSY = SAGR_BRIDGE_GENERIC_V2_STATUS_BUSY,
  SAGR_WIRE_STATUS_RESOURCE_EXHAUSTED =
      SAGR_BRIDGE_GENERIC_V2_STATUS_RESOURCE_EXHAUSTED,
  SAGR_WIRE_STATUS_PROTOCOL_STATE =
      SAGR_BRIDGE_GENERIC_V2_STATUS_PROTOCOL_STATE,
  SAGR_WIRE_STATUS_INTERNAL = SAGR_BRIDGE_GENERIC_V2_STATUS_INTERNAL
};

enum {
  SAGR_WIRE_MESSAGE_QUEUE_REQUEST = 3,
  SAGR_WIRE_MESSAGE_QUEUE_ACK = 4,
  SAGR_WIRE_MESSAGE_QUEUE_COMPLETION = 5,
  SAGR_WIRE_QUEUE_OPCODE_CREATE = 1,
  SAGR_WIRE_QUEUE_OPCODE_DESTROY = 2,
  SAGR_WIRE_QUEUE_OPCODE_DOORBELL = 3
};

enum {
  SAGR_WIRE_MESSAGE_MEMORY_REQUEST = 6,
  SAGR_WIRE_MESSAGE_MEMORY_ACK = 7,
  SAGR_WIRE_MEMORY_OPCODE_ALLOC = 1,
  SAGR_WIRE_MEMORY_OPCODE_FREE = 2,
  SAGR_WIRE_MEMORY_OPCODE_COPY_H2D = 3,
  SAGR_WIRE_MEMORY_OPCODE_COPY_D2H = 4
};

enum {
  SAGR_WIRE_MESSAGE_SIGNAL_REQUEST = 8,
  SAGR_WIRE_MESSAGE_SIGNAL_ACK = 9,
  SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION = 10,
  SAGR_WIRE_SIGNAL_OPCODE_CREATE = 1,
  SAGR_WIRE_SIGNAL_OPCODE_DESTROY = 2,
  SAGR_WIRE_SIGNAL_OPCODE_LOAD = 3,
  SAGR_WIRE_SIGNAL_OPCODE_STORE = 4,
  SAGR_WIRE_SIGNAL_OPCODE_WAIT = 5
};

enum {
  SAGR_WIRE_MESSAGE_DISPATCH_REQUEST = 11,
  SAGR_WIRE_MESSAGE_DISPATCH_ACK = 12,
  SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION = 13,
  SAGR_WIRE_DISPATCH_OPCODE_SUBMIT_PINNED = 1
};

enum {
  SAGR_WIRE_MESSAGE_KMT_REQUEST = SAGR_KMT_MESSAGE_REQUEST,
  SAGR_WIRE_MESSAGE_KMT_RESULT = SAGR_KMT_MESSAGE_RESULT,
  SAGR_WIRE_MESSAGE_KMT_ACK = SAGR_KMT_MESSAGE_ACK
};

enum {
  SAGR_WIRE_MESSAGE_CODE_OBJECT_REQUEST = 16,
  SAGR_WIRE_MESSAGE_CODE_OBJECT_ACK = 17,
  SAGR_WIRE_CODE_OBJECT_OPCODE_BEGIN = 1,
  SAGR_WIRE_CODE_OBJECT_OPCODE_CHUNK = 2,
  SAGR_WIRE_CODE_OBJECT_OPCODE_COMMIT = 3
};

enum {
  SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_REQUEST =
      SAGR_BRIDGE_GENERIC_V2_MESSAGE_GENERIC_DISPATCH_REQUEST,
  SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_ACK =
      SAGR_BRIDGE_GENERIC_V2_MESSAGE_GENERIC_DISPATCH_ACK,
  SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_COMPLETION =
      SAGR_BRIDGE_GENERIC_V2_MESSAGE_GENERIC_DISPATCH_COMPLETION,
  SAGR_WIRE_GENERIC_OPCODE_MAP_OBJECT =
      SAGR_BRIDGE_GENERIC_V2_OPCODE_MAP_OBJECT,
  SAGR_WIRE_GENERIC_OPCODE_ALLOC_KERNARG =
      SAGR_BRIDGE_GENERIC_V2_OPCODE_ALLOC_KERNARG,
  SAGR_WIRE_GENERIC_OPCODE_SUBMIT_AQL =
      SAGR_BRIDGE_GENERIC_V2_OPCODE_SUBMIT_AQL,
  SAGR_WIRE_GENERIC_OPCODE_UNMAP_OBJECT =
      SAGR_BRIDGE_GENERIC_V2_OPCODE_UNMAP_OBJECT
};

typedef struct sagr_wire_queue_request {
  uint16_t major;
  uint16_t minor;
  uint16_t opcode;
  uint16_t flags;
  uint64_t queue_id;
  uint64_t generation;
  uint64_t sequence;
  uint64_t arg0;
  uint64_t arg1;
} sagr_wire_queue_request_t;

typedef struct sagr_wire_queue_response {
  uint16_t major;
  uint16_t minor;
  uint32_t status;
  uint16_t opcode;
  uint64_t queue_id;
  uint64_t generation;
  uint64_t sequence;
  uint64_t value;
  uint64_t error_code;
  uint64_t sim_tick;
  uint64_t request_id;
  uint16_t message_type;
} sagr_wire_queue_response_t;

typedef struct sagr_wire_memory_request {
  uint16_t major;
  uint16_t minor;
  uint16_t opcode;
  uint16_t flags;
  uint64_t allocation_id;
  uint64_t generation;
  uint64_t offset;
  uint64_t byte_count;
  uint64_t argument;
} sagr_wire_memory_request_t;

typedef struct sagr_wire_memory_response {
  uint16_t major;
  uint16_t minor;
  uint32_t status;
  uint16_t opcode;
  uint64_t allocation_id;
  uint64_t generation;
  uint64_t value0;
  uint64_t value1;
  uint64_t value2;
  uint64_t sim_tick;
  uint64_t request_id;
} sagr_wire_memory_response_t;

typedef struct sagr_wire_signal_request {
  uint16_t major;
  uint16_t minor;
  uint16_t opcode;
  uint16_t flags;
  uint64_t signal_id;
  uint64_t generation;
  uint64_t sequence;
  uint64_t value_bits;
  uint64_t condition;
} sagr_wire_signal_request_t;

typedef struct sagr_wire_signal_response {
  uint16_t major;
  uint16_t minor;
  uint32_t status;
  uint16_t opcode;
  uint64_t signal_id;
  uint64_t generation;
  uint64_t sequence;
  uint64_t value_bits;
  uint64_t ready;
  uint64_t sim_tick;
  uint64_t request_id;
  uint16_t message_type;
} sagr_wire_signal_response_t;

typedef struct sagr_wire_dispatch_request {
  uint16_t major;
  uint16_t minor;
  uint16_t opcode;
  uint16_t flags;
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
  uint64_t expected_signal_value_bits;
  uint8_t fixture_manifest_sha256[32];
} sagr_wire_dispatch_request_t;

typedef struct sagr_wire_dispatch_response {
  uint16_t major;
  uint16_t minor;
  uint32_t status;
  uint16_t opcode;
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
  uint64_t request_id;
  uint16_t message_type;
} sagr_wire_dispatch_response_t;

typedef struct sagr_wire_code_object_segment {
  uint32_t type;
  uint32_t flags;
  uint64_t file_offset;
  uint64_t virtual_address;
  uint64_t file_size;
  uint64_t memory_size;
  uint64_t alignment;
} sagr_wire_code_object_segment_t;

typedef struct sagr_wire_code_object_begin {
  uint64_t image_size;
  uint32_t chunk_data_bytes;
  uint32_t chunk_count;
  uint32_t segment_count;
  uint32_t kernel_index;
  uint8_t image_sha256[32];
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
  uint32_t relocation_count;
  uint32_t kernarg_segment_size;
  uint32_t kernarg_segment_align;
  uint32_t group_segment_fixed_size;
  uint32_t private_segment_fixed_size;
  uint32_t max_flat_workgroup_size;
  uint32_t wavefront_size;
  uint32_t sgpr_count;
  uint32_t vgpr_count;
  uint32_t uses_dynamic_stack;
  uint32_t descriptor_size;
  int64_t descriptor_kernel_code_entry_byte_offset;
  uint64_t code_address;
  uint64_t code_file_offset;
  uint64_t code_size;
  uint64_t descriptor_address;
  uint64_t descriptor_file_offset;
  char kernel_name[SAGR_WIRE_CODE_OBJECT_NAME_BYTES];
  char symbol[SAGR_WIRE_CODE_OBJECT_NAME_BYTES];
  uint8_t descriptor[SAGR_WIRE_CODE_OBJECT_DESCRIPTOR_BYTES];
  sagr_wire_code_object_segment_t
      segments[SAGR_WIRE_CODE_OBJECT_MAX_SEGMENTS];
} sagr_wire_code_object_begin_t;

typedef struct sagr_wire_code_object_request {
  uint16_t major;
  uint16_t minor;
  uint16_t opcode;
  uint16_t flags;
  uint64_t object_id;
  uint64_t generation;
  uint64_t image_offset;
  uint32_t byte_count;
  uint32_t chunk_index;
  uint32_t chunk_crc32c;
  union {
    sagr_wire_code_object_begin_t begin;
    uint8_t chunk[SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES];
    uint8_t commit_sha256[32];
  } body;
} sagr_wire_code_object_request_t;

typedef struct sagr_wire_code_object_response {
  uint16_t major;
  uint16_t minor;
  uint32_t status;
  uint16_t opcode;
  uint16_t flags;
  uint64_t object_id;
  uint64_t generation;
  uint64_t accepted_offset;
  uint32_t accepted_count;
  uint32_t chunk_index;
  uint64_t mapped_base_va;
  uint64_t descriptor_va;
  uint64_t code_va;
  uint64_t kernarg_va;
  uint64_t image_size;
  uint32_t kernel_index;
  uint32_t segment_count;
  uint64_t sim_tick;
  uint8_t image_sha256[32];
  uint32_t error_code;
  uint32_t reserved0;
  uint64_t request_id;
} sagr_wire_code_object_response_t;

typedef struct sagr_wire_generic_map_body {
  uint32_t gfx_target;
  uint32_t relocation_count;
  uint32_t kernarg_segment_size;
  uint32_t kernarg_segment_align;
  uint32_t descriptor_preload_dwords;
  uint32_t page_size;
} sagr_wire_generic_map_body_t;

typedef struct sagr_wire_generic_alloc_kernarg_body {
  uint64_t size_bytes;
  uint64_t alignment_bytes;
  uint32_t allocation_flags;
  uint32_t reserved0;
} sagr_wire_generic_alloc_kernarg_body_t;

typedef struct sagr_wire_generic_submit_body {
  uint64_t kernarg_allocation_id;
  uint64_t kernarg_generation;
  uint64_t kernarg_offset;
  uint64_t kernarg_size;
  uint64_t signal_id;
  uint64_t signal_generation;
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
} sagr_wire_generic_submit_body_t;

/* V2 payloads carry identities and scalar launch metadata only.  C struct
 * padding is never serialized; the codec writes each field explicitly. */
typedef struct sagr_wire_generic_request {
  uint16_t major;
  uint16_t minor;
  uint16_t opcode;
  uint16_t flags;
  uint64_t object_id;
  uint64_t object_generation;
  uint64_t mapping_id;
  uint64_t mapping_generation;
  uint64_t queue_id;
  uint64_t queue_generation;
  uint64_t queue_sequence;
  uint32_t kernel_index;
  uint32_t reserved0;
  uint8_t image_sha256[32];
  char kernel_name[SAGR_WIRE_GENERIC_KERNEL_NAME_BYTES];
  union {
    sagr_wire_generic_map_body_t map;
    sagr_wire_generic_alloc_kernarg_body_t alloc_kernarg;
    sagr_wire_generic_submit_body_t submit;
  } body;
} sagr_wire_generic_request_t;

typedef struct sagr_wire_generic_response {
  uint16_t major;
  uint16_t minor;
  uint32_t status;
  uint16_t opcode;
  uint16_t flags;
  uint32_t error_code;
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
  uint64_t kernarg_allocation_id;
  uint64_t kernarg_generation;
  uint64_t kernarg_va;
  uint64_t kernarg_size;
  uint64_t kernarg_alignment;
  uint32_t kernel_index;
  uint32_t segment_count;
  uint32_t descriptor_preload_dwords;
  uint32_t reserved0;
  uint64_t ticket_id;
  uint64_t trace_id;
  uint64_t queue_id;
  uint64_t queue_generation;
  uint64_t queue_sequence;
  uint64_t signal_id;
  uint64_t signal_generation;
  uint64_t signal_value_bits;
  uint64_t packet_va;
  uint32_t packet_crc32c;
  uint32_t output_crc32c;
  uint64_t sim_tick;
  uint64_t admission_tick;
  uint64_t start_tick;
  uint64_t end_tick;
  uint64_t retire_tick;
  uint8_t image_sha256[32];
  uint64_t request_id;
  uint16_t message_type;
} sagr_wire_generic_response_t;

extern const uint8_t sagr_dispatch_fixture_manifest_sha256[32];

typedef struct sagr_wire_ack_fields {
  uint16_t selected_major;
  uint16_t selected_minor;
  uint32_t status;
  uint8_t client_nonce[16];
  uint8_t server_nonce[16];
  uint64_t selected_capabilities[SAGR_CAPABILITY_WORD_COUNT];
  uint32_t maximum_record_bytes;
  uint64_t request_id;
  uint8_t daemon_uuid[16];
  uint64_t connection_id;
  uint64_t epoch;
  uint8_t job_uuid[16];
  uint32_t rank;
  uint32_t world_size;
  uint32_t include_topology;
} sagr_wire_ack_fields_t;

typedef struct sagr_wire_ack_result {
  uint16_t selected_major;
  uint16_t selected_minor;
  uint32_t maximum_record_bytes;
  uint64_t selected_capabilities[SAGR_CAPABILITY_WORD_COUNT];
  uint8_t daemon_uuid[16];
  uint8_t job_uuid[16];
  uint64_t connection_id;
  uint64_t epoch;
  uint32_t rank;
  uint32_t world_size;
  uint64_t request_id;
} sagr_wire_ack_result_t;

uint32_t sagr_crc32c(const uint8_t *data, size_t size);
void sagr_protocol_recompute_frame_crc(uint8_t *frame, size_t frame_size);

sagr_status_t sagr_protocol_encode_hello(
    const sagr_instance_open_options_t *options, uint64_t request_id,
    const uint8_t client_nonce[16], uint8_t *frame, size_t frame_capacity,
    size_t *frame_size);

sagr_status_t sagr_protocol_encode_ack(
    const sagr_wire_ack_fields_t *fields, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_decode_ack(
    const uint8_t *frame, size_t frame_size,
    const sagr_instance_open_options_t *options, uint64_t request_id,
    const uint8_t client_nonce[16], sagr_wire_ack_result_t *result,
    int32_t *wire_status, const char **reason);

sagr_status_t sagr_protocol_encode_queue_request(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_queue_request_t *request, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_decode_queue_response(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, uint16_t expected_message_type,
    sagr_wire_queue_response_t *result, int32_t *wire_status,
    const char **reason);

sagr_status_t sagr_protocol_encode_queue_response(
    const sagr_instance_info_t *info, uint64_t request_id,
    uint16_t message_type, const sagr_wire_queue_response_t *response,
    uint8_t *frame, size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_allocate_request_id(uint64_t *next_request_id,
                                                uint64_t *request_id);

sagr_status_t sagr_protocol_validate_failed_queue_ack(
    const sagr_wire_queue_request_t *request,
    const sagr_wire_queue_response_t *response);

sagr_status_t sagr_protocol_encode_memory_request(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_memory_request_t *request, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_encode_memory_response(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_memory_response_t *response, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_decode_memory_response(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, sagr_wire_memory_response_t *result,
    int32_t *wire_status, const char **reason);

sagr_status_t sagr_protocol_validate_failed_memory_ack(
    const sagr_wire_memory_request_t *request,
    const sagr_wire_memory_response_t *response);

sagr_status_t sagr_protocol_encode_signal_request(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_signal_request_t *request, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_encode_signal_response(
    const sagr_instance_info_t *info, uint64_t request_id,
    uint16_t message_type, const sagr_wire_signal_response_t *response,
    uint8_t *frame, size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_decode_signal_response(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, uint16_t expected_message_type,
    sagr_wire_signal_response_t *result, int32_t *wire_status,
    const char **reason);

sagr_status_t sagr_protocol_validate_failed_signal_ack(
    const sagr_wire_signal_request_t *request,
    const sagr_wire_signal_response_t *response);

sagr_status_t sagr_protocol_encode_dispatch_request(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_dispatch_request_t *request, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_encode_dispatch_response(
    const sagr_instance_info_t *info, uint64_t request_id,
    uint16_t message_type, const sagr_wire_dispatch_response_t *response,
    uint8_t *frame, size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_decode_dispatch_response(
    const uint8_t *frame, size_t frame_size,
    const sagr_instance_info_t *info, uint64_t expected_request_id,
    uint16_t expected_message_type, sagr_wire_dispatch_response_t *result,
    int32_t *wire_status, const char **reason);

sagr_status_t sagr_protocol_validate_failed_dispatch_ack(
    const sagr_wire_dispatch_request_t *request,
    const sagr_wire_dispatch_response_t *response);

sagr_status_t sagr_protocol_encode_kmt_request(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_kmt_envelope_request_t *request, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_encode_kmt_result(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_kmt_envelope_result_t *result, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_decode_kmt_result(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, sagr_kmt_envelope_result_t *result,
    int32_t *wire_status, const char **reason);

sagr_status_t sagr_protocol_encode_code_object_request(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_code_object_request_t *request, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_decode_code_object_request(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    sagr_wire_code_object_request_t *request, uint64_t *request_id,
    const char **reason);

sagr_status_t sagr_protocol_encode_code_object_response(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_code_object_response_t *response, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_decode_code_object_response(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, sagr_wire_code_object_response_t *response,
    int32_t *wire_status, const char **reason);

sagr_status_t sagr_protocol_encode_generic_dispatch_request(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_generic_request_t *request, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_decode_generic_dispatch_request(
    const uint8_t *frame, size_t frame_size,
    const sagr_instance_info_t *info, sagr_wire_generic_request_t *request,
    uint64_t *request_id, const char **reason);

sagr_status_t sagr_protocol_encode_generic_dispatch_response(
    const sagr_instance_info_t *info, uint64_t request_id,
    uint16_t message_type, const sagr_wire_generic_response_t *response,
    uint8_t *frame, size_t frame_capacity, size_t *frame_size);

sagr_status_t sagr_protocol_decode_generic_dispatch_response(
    const uint8_t *frame, size_t frame_size,
    const sagr_instance_info_t *info, uint64_t expected_request_id,
    uint16_t expected_message_type, sagr_wire_generic_response_t *response,
    int32_t *wire_status, const char **reason);

sagr_status_t sagr_protocol_validate_failed_generic_dispatch_ack(
    const sagr_wire_generic_request_t *request,
    const sagr_wire_generic_response_t *response);

sagr_status_t sagr_protocol_map_wire_status(uint32_t status);

/* The provider shim uses this transport-private exchange.  The caller has
 * already validated typed ownership and copied-buffer bounds. */
sagr_status_t sagr_transport_kmt_exchange(
    sagr_instance_t instance, const sagr_kmt_envelope_request_t *request,
    const sagr_kmt_call_options_t *options,
    sagr_kmt_envelope_result_t *result, int32_t *wire_status,
    sagr_error_info_t *error, uint32_t error_size);

sagr_status_t sagr_transport_kmt_exchange_with_descriptor(
    sagr_instance_t instance, const sagr_kmt_envelope_request_t *request,
    int descriptor, const sagr_kmt_call_options_t *options,
    sagr_kmt_envelope_result_t *result, int32_t *wire_status,
    sagr_error_info_t *error, uint32_t error_size);

#endif
