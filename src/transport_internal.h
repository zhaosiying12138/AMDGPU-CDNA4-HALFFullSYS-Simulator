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
  SAGR_WIRE_CAPABILITY_BYTES = 32
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

#endif
