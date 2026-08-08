/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_TRANSPORT_INTERNAL_H
#define SELF_AMDGPU_RUNTIME_TRANSPORT_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include <self_amdgpu_runtime/runtime.h>

enum {
  SAGR_WIRE_HEADER_BYTES = 80,
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
      SAGR_WIRE_HEADER_BYTES + SAGR_WIRE_SIGNAL_PAYLOAD_BYTES
};

enum {
  SAGR_WIRE_STATUS_OK = 0,
  SAGR_WIRE_STATUS_MALFORMED = 1,
  SAGR_WIRE_STATUS_UNSUPPORTED_VERSION = 2,
  SAGR_WIRE_STATUS_UNSUPPORTED_CAPABILITY = 3,
  SAGR_WIRE_STATUS_INSTANCE_MISMATCH = 4,
  SAGR_WIRE_STATUS_TOPOLOGY_MISMATCH = 5,
  SAGR_WIRE_STATUS_UNAUTHORIZED = 6,
  SAGR_WIRE_STATUS_BUSY = 7,
  SAGR_WIRE_STATUS_RESOURCE_EXHAUSTED = 8,
  SAGR_WIRE_STATUS_PROTOCOL_STATE = 9,
  SAGR_WIRE_STATUS_INTERNAL = 10
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

sagr_status_t sagr_protocol_map_wire_status(uint32_t status);

#endif
