/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "transport_internal.h"

#include <self_amdgpu_runtime/code_object.h>

#include <limits.h>
#include <string.h>

static const uint8_t k_magic[8] = {'G', 'S', 'I', 'M', 'R', 'P', 'C', 0};

const uint8_t sagr_dispatch_fixture_manifest_sha256[32] = {
    0x75, 0x00, 0x74, 0x18, 0x73, 0xf9, 0xd3, 0x98,
    0x48, 0xe5, 0x7f, 0x0a, 0xa9, 0xff, 0xc6, 0x45,
    0x4d, 0xf7, 0xdb, 0x87, 0xb9, 0x3e, 0x1c, 0x04,
    0x65, 0x01, 0xf5, 0x4d, 0xb1, 0xb7, 0x54, 0x3c};

static void put_u16(uint8_t *destination, uint16_t value) {
  destination[0] = (uint8_t)(value >> 8);
  destination[1] = (uint8_t)value;
}

static void put_u32(uint8_t *destination, uint32_t value) {
  destination[0] = (uint8_t)(value >> 24);
  destination[1] = (uint8_t)(value >> 16);
  destination[2] = (uint8_t)(value >> 8);
  destination[3] = (uint8_t)value;
}

static void put_u64(uint8_t *destination, uint64_t value) {
  put_u32(destination, (uint32_t)(value >> 32));
  put_u32(destination + 4, (uint32_t)value);
}

static uint16_t get_u16(const uint8_t *source) {
  return (uint16_t)(((uint16_t)source[0] << 8) | (uint16_t)source[1]);
}

static uint32_t get_u32(const uint8_t *source) {
  return ((uint32_t)source[0] << 24) | ((uint32_t)source[1] << 16) |
         ((uint32_t)source[2] << 8) | (uint32_t)source[3];
}

static uint64_t get_u64(const uint8_t *source) {
  return ((uint64_t)get_u32(source) << 32) | (uint64_t)get_u32(source + 4);
}

static int bytes_are_zero(const uint8_t *bytes, size_t size) {
  size_t index;
  uint8_t combined = 0;
  for (index = 0; index < size; ++index) {
    combined = (uint8_t)(combined | bytes[index]);
  }
  return combined == 0;
}

static void capability_words_to_wire(
    const uint64_t words[SAGR_CAPABILITY_WORD_COUNT],
    uint8_t bytes[SAGR_WIRE_CAPABILITY_BYTES]) {
  uint32_t bit;
  memset(bytes, 0, SAGR_WIRE_CAPABILITY_BYTES);
  for (bit = 0; bit < 256; ++bit) {
    const uint64_t mask = UINT64_C(1) << (bit % 64U);
    if ((words[bit / 64U] & mask) != 0) {
      bytes[bit / 8U] = (uint8_t)(bytes[bit / 8U] |
                                  (uint8_t)(UINT8_C(1) << (bit % 8U)));
    }
  }
}

static void capability_wire_to_words(
    const uint8_t bytes[SAGR_WIRE_CAPABILITY_BYTES],
    uint64_t words[SAGR_CAPABILITY_WORD_COUNT]) {
  uint32_t bit;
  memset(words, 0, sizeof(uint64_t) * SAGR_CAPABILITY_WORD_COUNT);
  for (bit = 0; bit < 256; ++bit) {
    const uint8_t mask = (uint8_t)(UINT8_C(1) << (bit % 8U));
    if ((bytes[bit / 8U] & mask) != 0) {
      words[bit / 64U] |= UINT64_C(1) << (bit % 64U);
    }
  }
}

static int capabilities_subset(const uint64_t *subset,
                               const uint64_t *superset) {
  uint32_t index;
  for (index = 0; index < SAGR_CAPABILITY_WORD_COUNT; ++index) {
    if ((subset[index] & ~superset[index]) != 0) {
      return 0;
    }
  }
  return 1;
}

static int capabilities_are_zero(const uint64_t *capabilities) {
  uint32_t index;
  uint64_t combined = 0;
  for (index = 0; index < SAGR_CAPABILITY_WORD_COUNT; ++index) {
    combined |= capabilities[index];
  }
  return combined == 0;
}

static int capabilities_are_valid_selection(const uint64_t *capabilities) {
  uint32_t index;
  const uint64_t allowed = SAGR_CAPABILITY_TOPOLOGY_MASK |
                           SAGR_CAPABILITY_QUEUE_MASK |
                           SAGR_CAPABILITY_MEMORY_MASK |
                           SAGR_CAPABILITY_SIGNAL_MASK |
                           SAGR_CAPABILITY_DISPATCH_MASK |
                           SAGR_CAPABILITY_KMT_MASK |
                           SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK |
                           SAGR_CAPABILITY_GENERIC_DISPATCH_MASK |
                           SAGR_CAPABILITY_GENERIC_EXECUTION_MASK;
  if ((capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] &
       SAGR_CAPABILITY_TOPOLOGY_MASK) == 0 ||
      (capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] & ~allowed) != 0) {
    return 0;
  }
  if ((capabilities[SAGR_CAPABILITY_DISPATCH_WORD] &
       SAGR_CAPABILITY_DISPATCH_MASK) != 0 &&
      (capabilities[SAGR_CAPABILITY_QUEUE_WORD] &
           SAGR_CAPABILITY_QUEUE_MASK) == 0) {
    return 0;
  }
  if ((capabilities[SAGR_CAPABILITY_DISPATCH_WORD] &
       SAGR_CAPABILITY_DISPATCH_MASK) != 0 &&
      (capabilities[SAGR_CAPABILITY_MEMORY_WORD] &
           SAGR_CAPABILITY_MEMORY_MASK) == 0) {
    return 0;
  }
  if ((capabilities[SAGR_CAPABILITY_DISPATCH_WORD] &
       SAGR_CAPABILITY_DISPATCH_MASK) != 0 &&
      (capabilities[SAGR_CAPABILITY_SIGNAL_WORD] &
           SAGR_CAPABILITY_SIGNAL_MASK) == 0) {
    return 0;
  }
  if ((capabilities[SAGR_CAPABILITY_GENERIC_DISPATCH_WORD] &
       SAGR_CAPABILITY_GENERIC_DISPATCH_MASK) != 0 &&
      ((capabilities[SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_WORD] &
        SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK) == 0 ||
       (capabilities[SAGR_CAPABILITY_QUEUE_WORD] &
        SAGR_CAPABILITY_QUEUE_MASK) == 0 ||
       (capabilities[SAGR_CAPABILITY_MEMORY_WORD] &
        SAGR_CAPABILITY_MEMORY_MASK) == 0 ||
       (capabilities[SAGR_CAPABILITY_SIGNAL_WORD] &
        SAGR_CAPABILITY_SIGNAL_MASK) == 0)) {
    return 0;
  }
  if ((capabilities[SAGR_CAPABILITY_GENERIC_EXECUTION_WORD] &
       SAGR_CAPABILITY_GENERIC_EXECUTION_MASK) != 0 &&
      ((capabilities[SAGR_CAPABILITY_GENERIC_DISPATCH_WORD] &
        SAGR_CAPABILITY_GENERIC_DISPATCH_MASK) == 0 ||
       (capabilities[SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_WORD] &
        SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK) == 0 ||
       (capabilities[SAGR_CAPABILITY_QUEUE_WORD] &
        SAGR_CAPABILITY_QUEUE_MASK) == 0 ||
       (capabilities[SAGR_CAPABILITY_MEMORY_WORD] &
        SAGR_CAPABILITY_MEMORY_MASK) == 0 ||
       (capabilities[SAGR_CAPABILITY_SIGNAL_WORD] &
        SAGR_CAPABILITY_SIGNAL_MASK) == 0)) {
    return 0;
  }
  for (index = 1; index < SAGR_CAPABILITY_WORD_COUNT; ++index) {
    if (capabilities[index] != 0) {
      return 0;
    }
  }
  return 1;
}

uint32_t sagr_crc32c(const uint8_t *data, size_t size) {
  uint32_t crc = UINT32_MAX;
  size_t index;

  for (index = 0; index < size; ++index) {
    uint32_t bit;
    crc ^= data[index];
    for (bit = 0; bit < 8; ++bit) {
      const uint32_t mask = (uint32_t)(0U - (crc & UINT32_C(1)));
      crc = (crc >> 1) ^ (UINT32_C(0x82f63b78) & mask);
    }
  }
  return crc ^ UINT32_MAX;
}

static uint32_t frame_crc32c(const uint8_t *frame, size_t frame_size) {
  uint32_t crc = UINT32_MAX;
  size_t index;
  for (index = 0; index < frame_size; ++index) {
    uint8_t byte = frame[index];
    uint32_t bit;
    if (index >= 64 && index < 68) {
      byte = 0;
    }
    crc ^= byte;
    for (bit = 0; bit < 8; ++bit) {
      const uint32_t mask = (uint32_t)(0U - (crc & UINT32_C(1)));
      crc = (crc >> 1) ^ (UINT32_C(0x82f63b78) & mask);
    }
  }
  return crc ^ UINT32_MAX;
}

void sagr_protocol_recompute_frame_crc(uint8_t *frame, size_t frame_size) {
  if (frame == NULL || frame_size < SAGR_WIRE_HEADER_BYTES) {
    return;
  }
  memset(frame + 64, 0, 4);
  put_u32(frame + 64, frame_crc32c(frame, frame_size));
}

static void encode_header(uint8_t *frame, uint16_t message_type,
                          uint32_t payload_size, uint64_t request_id,
                          const uint8_t daemon_uuid[16],
                          uint64_t connection_id, uint64_t epoch) {
  memset(frame, 0, SAGR_WIRE_HEADER_BYTES);
  memcpy(frame, k_magic, sizeof(k_magic));
  put_u16(frame + 8, 1);
  put_u16(frame + 10, 0);
  put_u16(frame + 12, SAGR_WIRE_HEADER_BYTES);
  put_u16(frame + 14, message_type);
  put_u32(frame + 20, payload_size);
  put_u64(frame + 24, request_id);
  memcpy(frame + 32, daemon_uuid, 16);
  put_u64(frame + 48, connection_id);
  put_u64(frame + 56, epoch);
}

static void encode_topology_tlv(uint8_t *destination,
                                const uint8_t job_uuid[16], uint32_t rank,
                                uint32_t world_size) {
  put_u16(destination, 1);
  put_u16(destination + 2, 1);
  put_u32(destination + 4, 24);
  memcpy(destination + 8, job_uuid, 16);
  put_u32(destination + 24, rank);
  put_u32(destination + 28, world_size);
}

sagr_status_t sagr_protocol_encode_hello(
    const sagr_instance_open_options_t *options, uint64_t request_id,
    const uint8_t client_nonce[16], uint8_t *frame, size_t frame_capacity,
    size_t *frame_size) {
  const int include_topology =
      (options != NULL &&
       (options->offered_capabilities[SAGR_CAPABILITY_TOPOLOGY_WORD] &
        SAGR_CAPABILITY_TOPOLOGY_MASK) != 0);
  const size_t encoded_size =
      SAGR_WIRE_HEADER_BYTES + SAGR_WIRE_HELLO_FIXED_BYTES +
      (include_topology != 0 ? SAGR_WIRE_TOPOLOGY_TLV_BYTES : 0U);
  uint8_t *payload;
  if (options == NULL || client_nonce == NULL || frame == NULL ||
      frame_size == NULL || frame_capacity < encoded_size ||
      request_id == 0 || bytes_are_zero(client_nonce, 16)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }

  memset(frame, 0, encoded_size);
  encode_header(frame, 1, (uint32_t)(encoded_size - SAGR_WIRE_HEADER_BYTES),
                request_id, options->expected_daemon_uuid, 0,
                options->expected_epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, options->minimum_version_major);
  put_u16(payload + 2, options->minimum_version_minor);
  put_u16(payload + 4, options->maximum_version_major);
  put_u16(payload + 6, options->maximum_version_minor);
  memcpy(payload + 8, client_nonce, 16);
  capability_words_to_wire(options->offered_capabilities, payload + 24);
  capability_words_to_wire(options->required_capabilities, payload + 56);
  put_u32(payload + 88, SAGR_WIRE_MAX_RECORD_BYTES);
  put_u16(payload + 92, 1);
  if (include_topology != 0) {
    encode_topology_tlv(payload + SAGR_WIRE_HELLO_FIXED_BYTES,
                        options->expected_job_uuid, options->expected_rank,
                        options->expected_world_size);
  }
  sagr_protocol_recompute_frame_crc(frame, encoded_size);
  *frame_size = encoded_size;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_encode_ack(
    const sagr_wire_ack_fields_t *fields, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size) {
  const size_t encoded_size =
      SAGR_WIRE_HEADER_BYTES + SAGR_WIRE_ACK_FIXED_BYTES +
      (fields != NULL && fields->include_topology != 0
           ? SAGR_WIRE_TOPOLOGY_TLV_BYTES
           : 0U);
  uint8_t *payload;
  if (fields == NULL || frame == NULL || frame_size == NULL ||
      frame_capacity < encoded_size || fields->request_id == 0) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }

  memset(frame, 0, encoded_size);
  encode_header(frame, 2, (uint32_t)(encoded_size - SAGR_WIRE_HEADER_BYTES),
                fields->request_id, fields->daemon_uuid,
                fields->connection_id, fields->epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, fields->selected_major);
  put_u16(payload + 2, fields->selected_minor);
  put_u32(payload + 4, fields->status);
  memcpy(payload + 8, fields->client_nonce, 16);
  memcpy(payload + 24, fields->server_nonce, 16);
  capability_words_to_wire(fields->selected_capabilities, payload + 40);
  put_u32(payload + 72, fields->maximum_record_bytes);
  put_u16(payload + 76, 2);
  if (fields->include_topology != 0) {
    encode_topology_tlv(payload + SAGR_WIRE_ACK_FIXED_BYTES,
                        fields->job_uuid, fields->rank, fields->world_size);
  }
  sagr_protocol_recompute_frame_crc(frame, encoded_size);
  *frame_size = encoded_size;
  return SAGR_STATUS_SUCCESS;
}

typedef struct decoded_header {
  uint64_t request_id;
  uint8_t daemon_uuid[16];
  uint64_t connection_id;
  uint64_t epoch;
  const uint8_t *payload;
  size_t payload_size;
} decoded_header_t;

static sagr_status_t decode_ack_header(const uint8_t *frame,
                                       size_t frame_size,
                                       decoded_header_t *header,
                                       const char **reason) {
  uint32_t payload_size;
  if (frame_size < SAGR_WIRE_HEADER_BYTES ||
      frame_size > SAGR_WIRE_MAX_RECORD_BYTES ||
      frame_size > SAGR_WIRE_MAX_HANDSHAKE_BYTES) {
    *reason = "invalid ACK record size";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (memcmp(frame, k_magic, sizeof(k_magic)) != 0 || get_u16(frame + 8) != 1 ||
      get_u16(frame + 10) != 0 ||
      get_u16(frame + 12) != SAGR_WIRE_HEADER_BYTES) {
    *reason = "invalid ACK framing";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  payload_size = get_u32(frame + 20);
  if ((size_t)payload_size != frame_size - SAGR_WIRE_HEADER_BYTES) {
    *reason = "invalid ACK payload length";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (get_u16(frame + 14) != 2 || get_u32(frame + 16) != 0) {
    *reason = "invalid ACK type or flags";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (get_u64(frame + 24) == 0 || get_u32(frame + 68) != 0 ||
      get_u64(frame + 72) != 0) {
    *reason = "invalid ACK request or reserved field";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (frame_crc32c(frame, frame_size) != get_u32(frame + 64)) {
    *reason = "ACK CRC32C mismatch";
    return SAGR_STATUS_CHECKSUM_ERROR;
  }
  header->request_id = get_u64(frame + 24);
  memcpy(header->daemon_uuid, frame + 32, 16);
  header->connection_id = get_u64(frame + 48);
  header->epoch = get_u64(frame + 56);
  header->payload = frame + SAGR_WIRE_HEADER_BYTES;
  header->payload_size = payload_size;
  return SAGR_STATUS_SUCCESS;
}

typedef struct decoded_topology {
  uint32_t present;
  uint8_t job_uuid[16];
  uint32_t rank;
  uint32_t world_size;
} decoded_topology_t;

static sagr_status_t decode_tlvs(const uint8_t *bytes, size_t size,
                                 decoded_topology_t *topology,
                                 const char **reason) {
  uint16_t seen_types[512];
  size_t seen_count = 0;
  size_t offset = 0;
  memset(topology, 0, sizeof(*topology));

  while (offset < size) {
    uint16_t type;
    uint16_t flags;
    uint32_t value_size;
    size_t value_start;
    size_t value_end;
    size_t padded_end;
    size_t index;
    if (size - offset < 8) {
      *reason = "truncated ACK TLV header";
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    type = get_u16(bytes + offset);
    flags = get_u16(bytes + offset + 2);
    value_size = get_u32(bytes + offset + 4);
    if ((flags & (uint16_t)~UINT16_C(1)) != 0) {
      *reason = "invalid ACK TLV flags";
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    value_start = offset + 8;
    if ((size_t)value_size > size - value_start) {
      *reason = "invalid ACK TLV length";
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    value_end = value_start + (size_t)value_size;
    if (value_end > SIZE_MAX - 7U) {
      *reason = "overflowing ACK TLV length";
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    padded_end = (value_end + 7U) & ~(size_t)7U;
    if (padded_end > size) {
      *reason = "truncated ACK TLV padding";
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    for (index = value_end; index < padded_end; ++index) {
      if (bytes[index] != 0) {
        *reason = "nonzero ACK TLV padding";
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
    }
    for (index = 0; index < seen_count; ++index) {
      if (seen_types[index] == type) {
        *reason = "duplicate ACK TLV";
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
    }
    if (seen_count >= sizeof(seen_types) / sizeof(seen_types[0])) {
      *reason = "too many ACK TLVs";
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    seen_types[seen_count++] = type;

    if (type == 1) {
      if (flags != 1 || value_size != 24) {
        *reason = "malformed ACK topology TLV";
        return SAGR_STATUS_PROTOCOL_ERROR;
      }
      topology->present = 1;
      memcpy(topology->job_uuid, bytes + value_start, 16);
      topology->rank = get_u32(bytes + value_start + 16);
      topology->world_size = get_u32(bytes + value_start + 20);
    } else if ((flags & UINT16_C(1)) != 0) {
      *reason = "unsupported critical ACK TLV";
      return SAGR_STATUS_CAPABILITY_MISMATCH;
    }
    offset = padded_end;
  }
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_map_wire_status(uint32_t status) {
  switch (status) {
    case SAGR_WIRE_STATUS_OK:
      return SAGR_STATUS_SUCCESS;
    case SAGR_WIRE_STATUS_MALFORMED:
    case SAGR_WIRE_STATUS_PROTOCOL_STATE:
      return SAGR_STATUS_PROTOCOL_ERROR;
    case SAGR_WIRE_STATUS_UNSUPPORTED_VERSION:
      return SAGR_STATUS_VERSION_MISMATCH;
    case SAGR_WIRE_STATUS_UNSUPPORTED_CAPABILITY:
      return SAGR_STATUS_CAPABILITY_MISMATCH;
    case SAGR_WIRE_STATUS_INSTANCE_MISMATCH:
      return SAGR_STATUS_INSTANCE_MISMATCH;
    case SAGR_WIRE_STATUS_TOPOLOGY_MISMATCH:
      return SAGR_STATUS_TOPOLOGY_MISMATCH;
    case SAGR_WIRE_STATUS_UNAUTHORIZED:
      return SAGR_STATUS_UNAUTHORIZED;
    case SAGR_WIRE_STATUS_BUSY:
      return SAGR_STATUS_BUSY;
    case SAGR_WIRE_STATUS_RESOURCE_EXHAUSTED:
      return SAGR_STATUS_OUT_OF_RESOURCES;
    case SAGR_WIRE_STATUS_INTERNAL:
      return SAGR_STATUS_INTERNAL_ERROR;
    default:
      return SAGR_STATUS_PROTOCOL_ERROR;
  }
}

sagr_status_t sagr_protocol_allocate_request_id(uint64_t *next_request_id,
                                                uint64_t *request_id) {
  if (next_request_id == NULL || request_id == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (*next_request_id == 0) {
    *request_id = 0;
    return SAGR_STATUS_OUT_OF_RESOURCES;
  }
  *request_id = *next_request_id;
  *next_request_id =
      *request_id == UINT64_MAX ? 0 : *request_id + UINT64_C(1);
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_validate_failed_queue_ack(
    const sagr_wire_queue_request_t *request,
    const sagr_wire_queue_response_t *response) {
  if (request == NULL || response == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (response->status == SAGR_WIRE_STATUS_OK ||
      response->opcode != request->opcode ||
      response->queue_id != request->queue_id ||
      response->generation != request->generation ||
      response->sequence != request->sequence || response->value != 0 ||
      response->error_code != 0) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  return SAGR_STATUS_SUCCESS;
}

static int version_in_range(const sagr_instance_open_options_t *options,
                            uint16_t major, uint16_t minor) {
  const uint32_t selected = ((uint32_t)major << 16) | minor;
  const uint32_t minimum =
      ((uint32_t)options->minimum_version_major << 16) |
      options->minimum_version_minor;
  const uint32_t maximum =
      ((uint32_t)options->maximum_version_major << 16) |
      options->maximum_version_minor;
  return selected >= minimum && selected <= maximum;
}

sagr_status_t sagr_protocol_decode_ack(
    const uint8_t *frame, size_t frame_size,
    const sagr_instance_open_options_t *options, uint64_t request_id,
    const uint8_t client_nonce[16], sagr_wire_ack_result_t *result,
    int32_t *wire_status, const char **reason) {
  decoded_header_t header;
  decoded_topology_t topology;
  uint64_t selected[SAGR_CAPABILITY_WORD_COUNT];
  uint16_t selected_major;
  uint16_t selected_minor;
  uint32_t status;
  uint32_t maximum_record;
  sagr_status_t decode_status;

  if (reason != NULL) {
    *reason = "invalid ACK";
  }
  if (wire_status != NULL) {
    *wire_status = -1;
  }
  if (frame == NULL || options == NULL || client_nonce == NULL ||
      result == NULL || reason == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  decode_status = decode_ack_header(frame, frame_size, &header, reason);
  if (decode_status != SAGR_STATUS_SUCCESS) {
    return decode_status;
  }
  if (header.payload_size < SAGR_WIRE_ACK_FIXED_BYTES) {
    *reason = "short ACK payload";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (header.request_id != request_id) {
    *reason = "ACK request ID mismatch";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (memcmp(header.payload + 8, client_nonce, 16) != 0) {
    *reason = "ACK nonce mismatch";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }

  selected_major = get_u16(header.payload);
  selected_minor = get_u16(header.payload + 2);
  status = get_u32(header.payload + 4);
  maximum_record = get_u32(header.payload + 72);
  if (wire_status != NULL) {
    *wire_status = (status <= INT32_MAX) ? (int32_t)status : -1;
  }
  if (get_u16(header.payload + 76) != 2 ||
      get_u16(header.payload + 78) != 0) {
    *reason = "invalid ACK role or reserved field";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (maximum_record < SAGR_WIRE_MAX_HANDSHAKE_BYTES ||
      maximum_record > SAGR_WIRE_MAX_RECORD_BYTES) {
    *reason = "invalid ACK maximum record";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (bytes_are_zero(header.daemon_uuid, 16) || header.epoch == 0) {
    *reason = "invalid ACK daemon identity";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  capability_wire_to_words(header.payload + 40, selected);
  if (status > SAGR_WIRE_STATUS_INTERNAL) {
    *reason = "unknown ACK status";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (status == SAGR_WIRE_STATUS_OK) {
    if (header.connection_id == 0 || bytes_are_zero(header.daemon_uuid, 16) ||
        bytes_are_zero(header.payload + 24, 16)) {
      *reason = "invalid successful ACK session";
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    decode_status = decode_tlvs(
        header.payload + SAGR_WIRE_ACK_FIXED_BYTES,
        header.payload_size - SAGR_WIRE_ACK_FIXED_BYTES, &topology, reason);
    if (decode_status != SAGR_STATUS_SUCCESS) {
      return decode_status;
    }
  } else {
    memset(&topology, 0, sizeof(topology));
    if (header.connection_id != 0 ||
        !bytes_are_zero(header.payload + 24, 16) || selected_major != 0 ||
        selected_minor != 0 || !capabilities_are_zero(selected) ||
        header.payload_size != SAGR_WIRE_ACK_FIXED_BYTES) {
      *reason = "noncanonical failed ACK";
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
  }
  if (status != SAGR_WIRE_STATUS_OK) {
    decode_status = sagr_protocol_map_wire_status(status);
    *reason = decode_status == SAGR_STATUS_PROTOCOL_ERROR
                  ? "daemon rejected handshake protocol"
                  : "daemon rejected handshake";
    return decode_status;
  }

  if (selected_major != 1 || selected_minor != 0 ||
      !version_in_range(options, selected_major, selected_minor)) {
    *reason = "ACK selected unsupported version";
    return SAGR_STATUS_VERSION_MISMATCH;
  }
  if (!capabilities_subset(selected, options->offered_capabilities) ||
      !capabilities_subset(options->required_capabilities, selected)) {
    *reason = "ACK capability selection mismatch";
    return SAGR_STATUS_CAPABILITY_MISMATCH;
  }
  if (!capabilities_are_valid_selection(selected)) {
    *reason = "ACK selected invalid transport capabilities";
    return SAGR_STATUS_CAPABILITY_MISMATCH;
  }
  if (((selected[SAGR_CAPABILITY_QUEUE_WORD] &
        SAGR_CAPABILITY_QUEUE_MASK) != 0) !=
      ((options->required_capabilities[SAGR_CAPABILITY_QUEUE_WORD] &
        SAGR_CAPABILITY_QUEUE_MASK) != 0)) {
    *reason = "ACK queue capability was not both offered and required";
    return SAGR_STATUS_CAPABILITY_MISMATCH;
  }
  if (((selected[SAGR_CAPABILITY_MEMORY_WORD] &
        SAGR_CAPABILITY_MEMORY_MASK) != 0) !=
      ((options->required_capabilities[SAGR_CAPABILITY_MEMORY_WORD] &
        SAGR_CAPABILITY_MEMORY_MASK) != 0)) {
    *reason = "ACK memory capability was not both offered and required";
    return SAGR_STATUS_CAPABILITY_MISMATCH;
  }
  if (((selected[SAGR_CAPABILITY_SIGNAL_WORD] &
        SAGR_CAPABILITY_SIGNAL_MASK) != 0) !=
      ((options->required_capabilities[SAGR_CAPABILITY_SIGNAL_WORD] &
        SAGR_CAPABILITY_SIGNAL_MASK) != 0)) {
    *reason = "ACK signal capability was not both offered and required";
    return SAGR_STATUS_CAPABILITY_MISMATCH;
  }
  if (((selected[SAGR_CAPABILITY_DISPATCH_WORD] &
        SAGR_CAPABILITY_DISPATCH_MASK) != 0) !=
      ((options->required_capabilities[SAGR_CAPABILITY_DISPATCH_WORD] &
        SAGR_CAPABILITY_DISPATCH_MASK) != 0)) {
    *reason = "ACK dispatch capability was not both offered and required";
    return SAGR_STATUS_CAPABILITY_MISMATCH;
  }
  if (((selected[SAGR_CAPABILITY_KMT_WORD] & SAGR_CAPABILITY_KMT_MASK) != 0) !=
      ((options->required_capabilities[SAGR_CAPABILITY_KMT_WORD] &
        SAGR_CAPABILITY_KMT_MASK) != 0)) {
    *reason = "ACK KMT capability was not both offered and required";
    return SAGR_STATUS_CAPABILITY_MISMATCH;
  }
  if (((selected[SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_WORD] &
        SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK) != 0) !=
      ((options->required_capabilities[SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_WORD] &
        SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK) != 0)) {
    *reason = "ACK code-object transport capability was not both offered and required";
    return SAGR_STATUS_CAPABILITY_MISMATCH;
  }
  if (((selected[SAGR_CAPABILITY_GENERIC_DISPATCH_WORD] &
        SAGR_CAPABILITY_GENERIC_DISPATCH_MASK) != 0) !=
      ((options->required_capabilities[SAGR_CAPABILITY_GENERIC_DISPATCH_WORD] &
        SAGR_CAPABILITY_GENERIC_DISPATCH_MASK) != 0)) {
    *reason = "ACK generic dispatch capability was not both offered and required";
    return SAGR_STATUS_CAPABILITY_MISMATCH;
  }
  if (((selected[SAGR_CAPABILITY_GENERIC_EXECUTION_WORD] &
        SAGR_CAPABILITY_GENERIC_EXECUTION_MASK) != 0) !=
      ((options->required_capabilities[SAGR_CAPABILITY_GENERIC_EXECUTION_WORD] &
        SAGR_CAPABILITY_GENERIC_EXECUTION_MASK) != 0)) {
    *reason = "ACK generic execution capability was not both offered and required";
    return SAGR_STATUS_CAPABILITY_MISMATCH;
  }
  if (!bytes_are_zero(options->expected_daemon_uuid, 16) &&
      memcmp(options->expected_daemon_uuid, header.daemon_uuid, 16) != 0) {
    *reason = "ACK daemon instance mismatch";
    return SAGR_STATUS_INSTANCE_MISMATCH;
  }
  if (options->expected_epoch != 0 && options->expected_epoch != header.epoch) {
    *reason = "ACK job epoch mismatch";
    return SAGR_STATUS_TOPOLOGY_MISMATCH;
  }
  if (topology.present == 0 || bytes_are_zero(topology.job_uuid, 16) ||
      topology.world_size == 0 || topology.rank >= topology.world_size) {
    *reason = "ACK topology is missing or invalid";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (!bytes_are_zero(options->expected_job_uuid, 16) ||
      options->expected_rank != SAGR_INSTANCE_RANK_WILDCARD ||
      options->expected_world_size != 0) {
    if (memcmp(options->expected_job_uuid, topology.job_uuid, 16) != 0 ||
        options->expected_rank != topology.rank ||
        options->expected_world_size != topology.world_size) {
      *reason = "ACK topology identity mismatch";
      return SAGR_STATUS_TOPOLOGY_MISMATCH;
    }
  }

  result->selected_major = selected_major;
  result->selected_minor = selected_minor;
  result->maximum_record_bytes = maximum_record;
  memcpy(result->selected_capabilities, selected, sizeof(selected));
  memcpy(result->daemon_uuid, header.daemon_uuid, 16);
  memcpy(result->job_uuid, topology.job_uuid, 16);
  result->connection_id = header.connection_id;
  result->epoch = header.epoch;
  result->rank = topology.rank;
  result->world_size = topology.world_size;
  result->request_id = header.request_id;
  *reason = "success";
  return SAGR_STATUS_SUCCESS;
}

static int queue_opcode_is_valid(uint16_t opcode) {
  return opcode == SAGR_WIRE_QUEUE_OPCODE_CREATE ||
         opcode == SAGR_WIRE_QUEUE_OPCODE_DESTROY ||
         opcode == SAGR_WIRE_QUEUE_OPCODE_DOORBELL;
}

sagr_status_t sagr_protocol_encode_queue_request(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_queue_request_t *request, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size) {
  const size_t encoded_size = SAGR_WIRE_QUEUE_FRAME_BYTES;
  uint8_t *payload;
  if (info == NULL || request == NULL || frame == NULL || frame_size == NULL ||
      frame_capacity < encoded_size || request_id == 0 ||
      bytes_are_zero(info->daemon_uuid, 16) || info->connection_id == 0 ||
      info->epoch == 0 || request->major != SAGR_QUEUE_PROTOCOL_MAJOR ||
      request->minor != SAGR_QUEUE_PROTOCOL_MINOR || request->flags != 0 ||
      !queue_opcode_is_valid(request->opcode)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (request->opcode == SAGR_WIRE_QUEUE_OPCODE_CREATE &&
      (request->queue_id != 0 || request->generation != 0 ||
       request->sequence != 0 ||
       request->arg0 == 0 || request->arg0 > SAGR_QUEUE_MAX_DEPTH ||
       request->arg1 != 0)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (request->opcode == SAGR_WIRE_QUEUE_OPCODE_DESTROY &&
      (request->queue_id == 0 || request->generation == 0 ||
       request->sequence != 0 ||
       request->arg0 != 0 || request->arg1 != 0)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (request->opcode == SAGR_WIRE_QUEUE_OPCODE_DOORBELL &&
      (request->queue_id == 0 || request->generation == 0 ||
       request->sequence == 0 ||
       (request->arg0 != SAGR_QUEUE_COMMAND_NOOP &&
        request->arg0 != SAGR_QUEUE_COMMAND_CONTROL_TEST &&
        request->arg0 != SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST) ||
       request->arg1 != 0)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }

  memset(frame, 0, encoded_size);
  encode_header(frame, SAGR_WIRE_MESSAGE_QUEUE_REQUEST,
                SAGR_WIRE_QUEUE_PAYLOAD_BYTES, request_id, info->daemon_uuid,
                info->connection_id, info->epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, request->major);
  put_u16(payload + 2, request->minor);
  put_u16(payload + 4, request->opcode);
  put_u16(payload + 6, request->flags);
  put_u64(payload + 8, request->queue_id);
  put_u64(payload + 16, request->generation);
  put_u64(payload + 24, request->sequence);
  put_u64(payload + 32, request->arg0);
  put_u64(payload + 40, request->arg1);
  sagr_protocol_recompute_frame_crc(frame, encoded_size);
  *frame_size = encoded_size;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_encode_queue_response(
    const sagr_instance_info_t *info, uint64_t request_id,
    uint16_t message_type, const sagr_wire_queue_response_t *response,
    uint8_t *frame, size_t frame_capacity, size_t *frame_size) {
  const size_t encoded_size = SAGR_WIRE_QUEUE_FRAME_BYTES;
  uint8_t *payload;
  if (info == NULL || response == NULL || frame == NULL || frame_size == NULL ||
      frame_capacity < encoded_size || request_id == 0 ||
      (message_type != SAGR_WIRE_MESSAGE_QUEUE_ACK &&
       message_type != SAGR_WIRE_MESSAGE_QUEUE_COMPLETION) ||
      bytes_are_zero(info->daemon_uuid, 16) || info->connection_id == 0 ||
      info->epoch == 0 || response->major != SAGR_QUEUE_PROTOCOL_MAJOR ||
      response->minor != SAGR_QUEUE_PROTOCOL_MINOR ||
      response->status > SAGR_WIRE_STATUS_INTERNAL ||
      !queue_opcode_is_valid(response->opcode)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  memset(frame, 0, encoded_size);
  encode_header(frame, message_type, SAGR_WIRE_QUEUE_PAYLOAD_BYTES, request_id,
                info->daemon_uuid, info->connection_id, info->epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, response->major);
  put_u16(payload + 2, response->minor);
  put_u32(payload + 4, response->status);
  put_u16(payload + 8, response->opcode);
  put_u64(payload + 16, response->queue_id);
  put_u64(payload + 24, response->generation);
  put_u64(payload + 32, response->sequence);
  put_u64(payload + 40, response->value);
  put_u64(payload + 48, response->error_code);
  put_u64(payload + 56, response->sim_tick);
  sagr_protocol_recompute_frame_crc(frame, encoded_size);
  *frame_size = encoded_size;
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t decode_queue_response_header(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, uint16_t expected_message_type,
    const uint8_t **payload, const char **reason) {
  uint16_t actual_type;
  uint32_t payload_size;
  if (frame == NULL || info == NULL || payload == NULL || reason == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (frame_size != SAGR_WIRE_QUEUE_FRAME_BYTES) {
    *reason = "invalid queue response record size";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (memcmp(frame, k_magic, sizeof(k_magic)) != 0 || get_u16(frame + 8) != 1 ||
      get_u16(frame + 10) != 0 ||
      get_u16(frame + 12) != SAGR_WIRE_HEADER_BYTES) {
    *reason = "invalid queue response framing";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  actual_type = get_u16(frame + 14);
  if (actual_type != expected_message_type ||
      (actual_type != SAGR_WIRE_MESSAGE_QUEUE_ACK &&
       actual_type != SAGR_WIRE_MESSAGE_QUEUE_COMPLETION) ||
      get_u32(frame + 16) != 0) {
    *reason = "invalid queue response type or flags";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  payload_size = get_u32(frame + 20);
  if (payload_size != SAGR_WIRE_QUEUE_PAYLOAD_BYTES ||
      (size_t)payload_size != frame_size - SAGR_WIRE_HEADER_BYTES) {
    *reason = "invalid queue response payload length";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (get_u64(frame + 24) == 0 || get_u32(frame + 68) != 0 ||
      get_u64(frame + 72) != 0) {
    *reason = "invalid queue response request or reserved field";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (expected_request_id != 0 && get_u64(frame + 24) != expected_request_id) {
    *reason = "queue response request ID mismatch";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (frame_crc32c(frame, frame_size) != get_u32(frame + 64)) {
    *reason = "queue response CRC32C mismatch";
    return SAGR_STATUS_CHECKSUM_ERROR;
  }
  if (bytes_are_zero(info->daemon_uuid, 16) || info->connection_id == 0 ||
      info->epoch == 0) {
    *reason = "invalid local queue session identity";
    return SAGR_STATUS_INVALID_HANDLE;
  }
  if (memcmp(frame + 32, info->daemon_uuid, 16) != 0) {
    *reason = "queue response daemon identity mismatch";
    return SAGR_STATUS_INSTANCE_MISMATCH;
  }
  if (get_u64(frame + 48) != info->connection_id ||
      get_u64(frame + 56) != info->epoch) {
    *reason = "queue response session identity mismatch";
    return SAGR_STATUS_TOPOLOGY_MISMATCH;
  }
  *payload = frame + SAGR_WIRE_HEADER_BYTES;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_decode_queue_response(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, uint16_t expected_message_type,
    sagr_wire_queue_response_t *result, int32_t *wire_status,
    const char **reason) {
  const uint8_t *payload = NULL;
  sagr_status_t status;
  uint32_t wire;
  if (result == NULL || wire_status == NULL || reason == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  *wire_status = -1;
  *reason = "invalid queue response";
  status = decode_queue_response_header(frame, frame_size, info,
                                        expected_request_id,
                                        expected_message_type, &payload,
                                        reason);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (get_u16(payload) != SAGR_QUEUE_PROTOCOL_MAJOR ||
      get_u16(payload + 2) != SAGR_QUEUE_PROTOCOL_MINOR ||
      get_u16(payload + 8) == 0 || !queue_opcode_is_valid(get_u16(payload + 8)) ||
      get_u16(payload + 10) != 0) {
    *reason = "invalid queue response fixed fields";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (get_u32(payload + 12) != 0) {
    *reason = "nonzero queue response reserved field";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  wire = get_u32(payload + 4);
  if (wire > SAGR_WIRE_STATUS_INTERNAL) {
    *reason = "unknown queue response status";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  *wire_status = (int32_t)wire;
  result->major = get_u16(payload);
  result->minor = get_u16(payload + 2);
  result->status = wire;
  result->opcode = get_u16(payload + 8);
  result->queue_id = get_u64(payload + 16);
  result->generation = get_u64(payload + 24);
  result->sequence = get_u64(payload + 32);
  result->value = get_u64(payload + 40);
  result->error_code = get_u64(payload + 48);
  result->sim_tick = get_u64(payload + 56);
  result->request_id = get_u64(frame + 24);
  result->message_type = expected_message_type;
  if (wire != SAGR_WIRE_STATUS_OK) {
    *reason = "daemon rejected queue operation";
    return sagr_protocol_map_wire_status(wire);
  }
  *reason = "queue operation succeeded";
  return SAGR_STATUS_SUCCESS;
}

static int memory_opcode_is_valid(uint16_t opcode) {
  return opcode == SAGR_WIRE_MEMORY_OPCODE_ALLOC ||
         opcode == SAGR_WIRE_MEMORY_OPCODE_FREE ||
         opcode == SAGR_WIRE_MEMORY_OPCODE_COPY_H2D ||
         opcode == SAGR_WIRE_MEMORY_OPCODE_COPY_D2H;
}

sagr_status_t sagr_protocol_encode_memory_request(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_memory_request_t *request, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size) {
  const size_t encoded_size = SAGR_WIRE_MEMORY_FRAME_BYTES;
  uint8_t *payload;
  if (info == NULL || request == NULL || frame == NULL || frame_size == NULL ||
      frame_capacity < encoded_size || request_id == 0 ||
      bytes_are_zero(info->daemon_uuid, 16) || info->connection_id == 0 ||
      info->epoch == 0 || request->major != SAGR_MEMORY_PROTOCOL_MAJOR ||
      request->minor != SAGR_MEMORY_PROTOCOL_MINOR || request->flags != 0 ||
      !memory_opcode_is_valid(request->opcode)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (request->opcode == SAGR_WIRE_MEMORY_OPCODE_ALLOC &&
      (request->allocation_id != 0 || request->generation != 0 ||
       request->offset != 0 || request->byte_count == 0 ||
       (request->argument != SAGR_MEMORY_ALIGNMENT_4K &&
        request->argument != SAGR_MEMORY_ALIGNMENT_64K))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (request->opcode == SAGR_WIRE_MEMORY_OPCODE_FREE &&
      (request->allocation_id == 0 || request->generation == 0 ||
       request->offset != 0 || request->byte_count != 0 ||
       request->argument != 0)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if ((request->opcode == SAGR_WIRE_MEMORY_OPCODE_COPY_H2D ||
       request->opcode == SAGR_WIRE_MEMORY_OPCODE_COPY_D2H) &&
      (request->allocation_id == 0 || request->generation == 0 ||
       request->byte_count == 0 ||
       request->offset > UINT64_MAX - request->byte_count ||
       (request->opcode == SAGR_WIRE_MEMORY_OPCODE_COPY_H2D
            ? request->argument > UINT32_MAX
            : request->argument != 0))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }

  memset(frame, 0, encoded_size);
  encode_header(frame, SAGR_WIRE_MESSAGE_MEMORY_REQUEST,
                SAGR_WIRE_MEMORY_PAYLOAD_BYTES, request_id, info->daemon_uuid,
                info->connection_id, info->epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, request->major);
  put_u16(payload + 2, request->minor);
  put_u16(payload + 4, request->opcode);
  put_u16(payload + 6, request->flags);
  put_u64(payload + 8, request->allocation_id);
  put_u64(payload + 16, request->generation);
  put_u64(payload + 24, request->offset);
  put_u64(payload + 32, request->byte_count);
  put_u64(payload + 40, request->argument);
  sagr_protocol_recompute_frame_crc(frame, encoded_size);
  *frame_size = encoded_size;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_encode_memory_response(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_memory_response_t *response, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size) {
  const size_t encoded_size = SAGR_WIRE_MEMORY_FRAME_BYTES;
  uint8_t *payload;
  if (info == NULL || response == NULL || frame == NULL || frame_size == NULL ||
      frame_capacity < encoded_size || request_id == 0 ||
      bytes_are_zero(info->daemon_uuid, 16) || info->connection_id == 0 ||
      info->epoch == 0 || response->major != SAGR_MEMORY_PROTOCOL_MAJOR ||
      response->minor != SAGR_MEMORY_PROTOCOL_MINOR ||
      response->status > SAGR_WIRE_STATUS_INTERNAL ||
      !memory_opcode_is_valid(response->opcode)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  memset(frame, 0, encoded_size);
  encode_header(frame, SAGR_WIRE_MESSAGE_MEMORY_ACK,
                SAGR_WIRE_MEMORY_PAYLOAD_BYTES, request_id, info->daemon_uuid,
                info->connection_id, info->epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, response->major);
  put_u16(payload + 2, response->minor);
  put_u32(payload + 4, response->status);
  put_u16(payload + 8, response->opcode);
  put_u64(payload + 16, response->allocation_id);
  put_u64(payload + 24, response->generation);
  put_u64(payload + 32, response->value0);
  put_u64(payload + 40, response->value1);
  put_u64(payload + 48, response->value2);
  put_u64(payload + 56, response->sim_tick);
  sagr_protocol_recompute_frame_crc(frame, encoded_size);
  *frame_size = encoded_size;
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t decode_memory_response_header(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, const uint8_t **payload,
    const char **reason) {
  if (frame == NULL || info == NULL || payload == NULL || reason == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (frame_size != SAGR_WIRE_MEMORY_FRAME_BYTES) {
    *reason = "invalid memory ACK record size";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (memcmp(frame, k_magic, sizeof(k_magic)) != 0 || get_u16(frame + 8) != 1 ||
      get_u16(frame + 10) != 0 ||
      get_u16(frame + 12) != SAGR_WIRE_HEADER_BYTES ||
      get_u16(frame + 14) != SAGR_WIRE_MESSAGE_MEMORY_ACK ||
      get_u32(frame + 16) != 0 ||
      get_u32(frame + 20) != SAGR_WIRE_MEMORY_PAYLOAD_BYTES) {
    *reason = "invalid memory ACK framing";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (get_u64(frame + 24) == 0 ||
      get_u64(frame + 24) != expected_request_id ||
      get_u32(frame + 68) != 0 || get_u64(frame + 72) != 0) {
    *reason = "invalid memory ACK request or reserved field";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (frame_crc32c(frame, frame_size) != get_u32(frame + 64)) {
    *reason = "memory ACK CRC32C mismatch";
    return SAGR_STATUS_CHECKSUM_ERROR;
  }
  if (bytes_are_zero(info->daemon_uuid, 16) || info->connection_id == 0 ||
      info->epoch == 0) {
    *reason = "invalid local memory session identity";
    return SAGR_STATUS_INVALID_HANDLE;
  }
  if (memcmp(frame + 32, info->daemon_uuid, 16) != 0) {
    *reason = "memory ACK daemon identity mismatch";
    return SAGR_STATUS_INSTANCE_MISMATCH;
  }
  if (get_u64(frame + 48) != info->connection_id ||
      get_u64(frame + 56) != info->epoch) {
    *reason = "memory ACK session identity mismatch";
    return SAGR_STATUS_TOPOLOGY_MISMATCH;
  }
  *payload = frame + SAGR_WIRE_HEADER_BYTES;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_decode_memory_response(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, sagr_wire_memory_response_t *result,
    int32_t *wire_status, const char **reason) {
  const uint8_t *payload = NULL;
  uint32_t wire;
  sagr_status_t status;
  if (result == NULL || wire_status == NULL || reason == NULL ||
      expected_request_id == 0) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  *wire_status = -1;
  *reason = "invalid memory ACK";
  status = decode_memory_response_header(frame, frame_size, info,
                                         expected_request_id, &payload, reason);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  if (get_u16(payload) != SAGR_MEMORY_PROTOCOL_MAJOR ||
      get_u16(payload + 2) != SAGR_MEMORY_PROTOCOL_MINOR ||
      !memory_opcode_is_valid(get_u16(payload + 8)) ||
      get_u16(payload + 10) != 0 || get_u32(payload + 12) != 0) {
    *reason = "invalid memory ACK fixed fields";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  wire = get_u32(payload + 4);
  if (wire > SAGR_WIRE_STATUS_INTERNAL) {
    *reason = "unknown memory ACK status";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  *wire_status = (int32_t)wire;
  result->major = get_u16(payload);
  result->minor = get_u16(payload + 2);
  result->status = wire;
  result->opcode = get_u16(payload + 8);
  result->allocation_id = get_u64(payload + 16);
  result->generation = get_u64(payload + 24);
  result->value0 = get_u64(payload + 32);
  result->value1 = get_u64(payload + 40);
  result->value2 = get_u64(payload + 48);
  result->sim_tick = get_u64(payload + 56);
  result->request_id = get_u64(frame + 24);
  if (wire != SAGR_WIRE_STATUS_OK) {
    *reason = "daemon rejected memory operation";
    return sagr_protocol_map_wire_status(wire);
  }
  *reason = "memory operation succeeded";
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_validate_failed_memory_ack(
    const sagr_wire_memory_request_t *request,
    const sagr_wire_memory_response_t *response) {
  if (request == NULL || response == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (response->status == SAGR_WIRE_STATUS_OK ||
      response->opcode != request->opcode ||
      response->allocation_id != request->allocation_id ||
      response->generation != request->generation || response->value0 != 0 ||
      response->value1 != 0 || response->value2 != 0 ||
      response->sim_tick != 0) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  return SAGR_STATUS_SUCCESS;
}

static int signal_opcode_is_valid(uint16_t opcode) {
  return opcode == SAGR_WIRE_SIGNAL_OPCODE_CREATE ||
         opcode == SAGR_WIRE_SIGNAL_OPCODE_DESTROY ||
         opcode == SAGR_WIRE_SIGNAL_OPCODE_LOAD ||
         opcode == SAGR_WIRE_SIGNAL_OPCODE_STORE ||
         opcode == SAGR_WIRE_SIGNAL_OPCODE_WAIT;
}

sagr_status_t sagr_protocol_encode_signal_request(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_signal_request_t *request, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size) {
  uint8_t *payload;
  if (info == NULL || request == NULL || frame == NULL || frame_size == NULL ||
      frame_capacity < SAGR_WIRE_SIGNAL_FRAME_BYTES || request_id == 0 ||
      bytes_are_zero(info->daemon_uuid, 16) || info->connection_id == 0 ||
      info->epoch == 0 || request->major != SAGR_SIGNAL_PROTOCOL_MAJOR ||
      request->minor != SAGR_SIGNAL_PROTOCOL_MINOR || request->flags != 0 ||
      !signal_opcode_is_valid(request->opcode)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (request->opcode == SAGR_WIRE_SIGNAL_OPCODE_CREATE &&
      (request->signal_id != 0 || request->generation != 0 ||
       request->sequence != 0 || request->condition != 0)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if ((request->opcode == SAGR_WIRE_SIGNAL_OPCODE_DESTROY ||
       request->opcode == SAGR_WIRE_SIGNAL_OPCODE_LOAD) &&
      (request->signal_id == 0 || request->generation == 0 ||
       request->sequence != 0 || request->value_bits != 0 ||
       request->condition != 0)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (request->opcode == SAGR_WIRE_SIGNAL_OPCODE_STORE &&
      (request->signal_id == 0 || request->generation == 0 ||
       request->sequence != 0 || request->condition != 0)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (request->opcode == SAGR_WIRE_SIGNAL_OPCODE_WAIT &&
      (request->signal_id == 0 || request->generation == 0 ||
       request->sequence == 0 || request->condition > SAGR_SIGNAL_CONDITION_GTE)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }

  memset(frame, 0, SAGR_WIRE_SIGNAL_FRAME_BYTES);
  encode_header(frame, SAGR_WIRE_MESSAGE_SIGNAL_REQUEST,
                SAGR_WIRE_SIGNAL_PAYLOAD_BYTES, request_id, info->daemon_uuid,
                info->connection_id, info->epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, request->major);
  put_u16(payload + 2, request->minor);
  put_u16(payload + 4, request->opcode);
  put_u16(payload + 6, request->flags);
  put_u64(payload + 8, request->signal_id);
  put_u64(payload + 16, request->generation);
  put_u64(payload + 24, request->sequence);
  put_u64(payload + 32, request->value_bits);
  put_u64(payload + 40, request->condition);
  sagr_protocol_recompute_frame_crc(frame, SAGR_WIRE_SIGNAL_FRAME_BYTES);
  *frame_size = SAGR_WIRE_SIGNAL_FRAME_BYTES;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_encode_signal_response(
    const sagr_instance_info_t *info, uint64_t request_id,
    uint16_t message_type, const sagr_wire_signal_response_t *response,
    uint8_t *frame, size_t frame_capacity, size_t *frame_size) {
  uint8_t *payload;
  if (info == NULL || response == NULL || frame == NULL || frame_size == NULL ||
      frame_capacity < SAGR_WIRE_SIGNAL_FRAME_BYTES || request_id == 0 ||
      (message_type != SAGR_WIRE_MESSAGE_SIGNAL_ACK &&
       message_type != SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION) ||
      bytes_are_zero(info->daemon_uuid, 16) || info->connection_id == 0 ||
      info->epoch == 0 || response->major != SAGR_SIGNAL_PROTOCOL_MAJOR ||
      response->minor != SAGR_SIGNAL_PROTOCOL_MINOR ||
      response->status > SAGR_WIRE_STATUS_INTERNAL ||
      response->ready > 1 ||
      ((message_type == SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION ||
        response->status == SAGR_WIRE_STATUS_OK) &&
       !signal_opcode_is_valid(response->opcode))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  memset(frame, 0, SAGR_WIRE_SIGNAL_FRAME_BYTES);
  encode_header(frame, message_type, SAGR_WIRE_SIGNAL_PAYLOAD_BYTES,
                request_id, info->daemon_uuid, info->connection_id,
                info->epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, response->major);
  put_u16(payload + 2, response->minor);
  put_u32(payload + 4, response->status);
  put_u16(payload + 8, response->opcode);
  put_u64(payload + 16, response->signal_id);
  put_u64(payload + 24, response->generation);
  put_u64(payload + 32, response->sequence);
  put_u64(payload + 40, response->value_bits);
  put_u64(payload + 48, response->ready);
  put_u64(payload + 56, response->sim_tick);
  sagr_protocol_recompute_frame_crc(frame, SAGR_WIRE_SIGNAL_FRAME_BYTES);
  *frame_size = SAGR_WIRE_SIGNAL_FRAME_BYTES;
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t decode_signal_response_header(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, uint16_t expected_message_type,
    const uint8_t **payload, const char **reason) {
  uint16_t actual_type;
  if (frame == NULL || info == NULL || payload == NULL || reason == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (frame_size != SAGR_WIRE_SIGNAL_FRAME_BYTES) {
    *reason = "invalid signal response record size";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  actual_type = get_u16(frame + 14);
  if (memcmp(frame, k_magic, sizeof(k_magic)) != 0 ||
      get_u16(frame + 8) != 1 || get_u16(frame + 10) != 0 ||
      get_u16(frame + 12) != SAGR_WIRE_HEADER_BYTES ||
      actual_type != expected_message_type ||
      (actual_type != SAGR_WIRE_MESSAGE_SIGNAL_ACK &&
       actual_type != SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION) ||
      get_u32(frame + 16) != 0 ||
      get_u32(frame + 20) != SAGR_WIRE_SIGNAL_PAYLOAD_BYTES) {
    *reason = "invalid signal response framing";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (get_u64(frame + 24) == 0 ||
      (expected_request_id != 0 &&
       get_u64(frame + 24) != expected_request_id) ||
      get_u32(frame + 68) != 0 || get_u64(frame + 72) != 0) {
    *reason = "invalid signal response request or reserved field";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (frame_crc32c(frame, frame_size) != get_u32(frame + 64)) {
    *reason = "signal response CRC32C mismatch";
    return SAGR_STATUS_CHECKSUM_ERROR;
  }
  if (bytes_are_zero(info->daemon_uuid, 16) || info->connection_id == 0 ||
      info->epoch == 0) {
    *reason = "invalid local signal session identity";
    return SAGR_STATUS_INVALID_HANDLE;
  }
  if (memcmp(frame + 32, info->daemon_uuid, 16) != 0) {
    *reason = "signal response daemon identity mismatch";
    return SAGR_STATUS_INSTANCE_MISMATCH;
  }
  if (get_u64(frame + 48) != info->connection_id ||
      get_u64(frame + 56) != info->epoch) {
    *reason = "signal response session identity mismatch";
    return SAGR_STATUS_TOPOLOGY_MISMATCH;
  }
  *payload = frame + SAGR_WIRE_HEADER_BYTES;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_decode_signal_response(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, uint16_t expected_message_type,
    sagr_wire_signal_response_t *result, int32_t *wire_status,
    const char **reason) {
  const uint8_t *payload = NULL;
  uint32_t wire;
  sagr_status_t status;
  if (result == NULL || wire_status == NULL || reason == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  *wire_status = -1;
  *reason = "invalid signal response";
  status = decode_signal_response_header(
      frame, frame_size, info, expected_request_id, expected_message_type,
      &payload, reason);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  wire = get_u32(payload + 4);
  if (get_u16(payload) != SAGR_SIGNAL_PROTOCOL_MAJOR ||
      get_u16(payload + 2) != SAGR_SIGNAL_PROTOCOL_MINOR ||
      get_u16(payload + 10) != 0 || get_u32(payload + 12) != 0 ||
      get_u64(payload + 48) > 1 || wire > SAGR_WIRE_STATUS_INTERNAL ||
      ((expected_message_type == SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION ||
        wire == SAGR_WIRE_STATUS_OK) &&
       !signal_opcode_is_valid(get_u16(payload + 8)))) {
    *reason = "invalid signal response fixed fields";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  *wire_status = (int32_t)wire;
  result->major = get_u16(payload);
  result->minor = get_u16(payload + 2);
  result->status = wire;
  result->opcode = get_u16(payload + 8);
  result->signal_id = get_u64(payload + 16);
  result->generation = get_u64(payload + 24);
  result->sequence = get_u64(payload + 32);
  result->value_bits = get_u64(payload + 40);
  result->ready = get_u64(payload + 48);
  result->sim_tick = get_u64(payload + 56);
  result->request_id = get_u64(frame + 24);
  result->message_type = expected_message_type;
  if (wire != SAGR_WIRE_STATUS_OK) {
    *reason = "daemon rejected signal operation";
    return sagr_protocol_map_wire_status(wire);
  }
  *reason = "signal operation succeeded";
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_validate_failed_signal_ack(
    const sagr_wire_signal_request_t *request,
    const sagr_wire_signal_response_t *response) {
  if (request == NULL || response == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (response->status == SAGR_WIRE_STATUS_OK ||
      response->opcode != request->opcode ||
      response->signal_id != request->signal_id ||
      response->generation != request->generation ||
      response->sequence != request->sequence || response->value_bits != 0 ||
      response->ready != 0 || response->sim_tick != 0) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  return SAGR_STATUS_SUCCESS;
}

static int dispatch_opcode_is_valid(uint16_t opcode) {
  return opcode == SAGR_WIRE_DISPATCH_OPCODE_SUBMIT_PINNED;
}

sagr_status_t sagr_protocol_encode_dispatch_request(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_dispatch_request_t *request, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size) {
  uint8_t *payload;
  if (info == NULL || request == NULL || frame == NULL || frame_size == NULL ||
      frame_capacity < SAGR_WIRE_DISPATCH_REQUEST_FRAME_BYTES ||
      request_id == 0 || bytes_are_zero(info->daemon_uuid, 16) ||
      info->connection_id == 0 || info->epoch == 0 ||
      request->major != SAGR_DISPATCH_PROTOCOL_MAJOR ||
      request->minor != SAGR_DISPATCH_PROTOCOL_MINOR || request->flags != 0 ||
      !dispatch_opcode_is_valid(request->opcode) ||
      request->queue_id == 0 || request->queue_generation == 0 ||
      request->queue_sequence == 0 ||
      request->fixture_id != SAGR_DISPATCH_FIXTURE_GFX950_XOR_U8_V1 ||
      request->input_allocation_id == 0 || request->input_generation == 0 ||
      request->output_allocation_id == 0 || request->output_generation == 0 ||
      request->input_allocation_id == request->output_allocation_id ||
      request->signal_id == 0 || request->signal_generation == 0 ||
      request->expected_signal_value_bits != 0 ||
      memcmp(request->fixture_manifest_sha256,
             sagr_dispatch_fixture_manifest_sha256,
             sizeof(request->fixture_manifest_sha256)) != 0) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }

  memset(frame, 0, SAGR_WIRE_DISPATCH_REQUEST_FRAME_BYTES);
  encode_header(frame, SAGR_WIRE_MESSAGE_DISPATCH_REQUEST,
                SAGR_WIRE_DISPATCH_REQUEST_PAYLOAD_BYTES, request_id,
                info->daemon_uuid, info->connection_id, info->epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, request->major);
  put_u16(payload + 2, request->minor);
  put_u16(payload + 4, request->opcode);
  put_u16(payload + 6, request->flags);
  put_u64(payload + 8, request->queue_id);
  put_u64(payload + 16, request->queue_generation);
  put_u64(payload + 24, request->queue_sequence);
  put_u64(payload + 32, request->fixture_id);
  put_u64(payload + 40, request->input_allocation_id);
  put_u64(payload + 48, request->input_generation);
  put_u64(payload + 56, request->output_allocation_id);
  put_u64(payload + 64, request->output_generation);
  put_u64(payload + 72, request->signal_id);
  put_u64(payload + 80, request->signal_generation);
  put_u64(payload + 88, request->expected_signal_value_bits);
  memcpy(payload + 96, request->fixture_manifest_sha256, 32);
  sagr_protocol_recompute_frame_crc(frame,
                                    SAGR_WIRE_DISPATCH_REQUEST_FRAME_BYTES);
  *frame_size = SAGR_WIRE_DISPATCH_REQUEST_FRAME_BYTES;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_encode_dispatch_response(
    const sagr_instance_info_t *info, uint64_t request_id,
    uint16_t message_type, const sagr_wire_dispatch_response_t *response,
    uint8_t *frame, size_t frame_capacity, size_t *frame_size) {
  uint8_t *payload;
  if (info == NULL || response == NULL || frame == NULL || frame_size == NULL ||
      frame_capacity < SAGR_WIRE_DISPATCH_RESULT_FRAME_BYTES ||
      request_id == 0 ||
      (message_type != SAGR_WIRE_MESSAGE_DISPATCH_ACK &&
       message_type != SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION) ||
      bytes_are_zero(info->daemon_uuid, 16) || info->connection_id == 0 ||
      info->epoch == 0 || response->major != SAGR_DISPATCH_PROTOCOL_MAJOR ||
      response->minor != SAGR_DISPATCH_PROTOCOL_MINOR ||
      response->status > SAGR_WIRE_STATUS_INTERNAL ||
      ((message_type == SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION ||
        response->status == SAGR_WIRE_STATUS_OK) &&
       !dispatch_opcode_is_valid(response->opcode))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }

  memset(frame, 0, SAGR_WIRE_DISPATCH_RESULT_FRAME_BYTES);
  encode_header(frame, message_type, SAGR_WIRE_DISPATCH_RESULT_PAYLOAD_BYTES,
                request_id, info->daemon_uuid, info->connection_id,
                info->epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, response->major);
  put_u16(payload + 2, response->minor);
  put_u32(payload + 4, response->status);
  put_u16(payload + 8, response->opcode);
  put_u64(payload + 16, response->queue_id);
  put_u64(payload + 24, response->queue_generation);
  put_u64(payload + 32, response->queue_sequence);
  put_u64(payload + 40, response->fixture_id);
  put_u64(payload + 48, response->input_allocation_id);
  put_u64(payload + 56, response->input_generation);
  put_u64(payload + 64, response->output_allocation_id);
  put_u64(payload + 72, response->output_generation);
  put_u64(payload + 80, response->signal_id);
  put_u64(payload + 88, response->signal_generation);
  put_u64(payload + 96, response->trace_id);
  put_u64(payload + 104, response->input_gpu_va);
  put_u64(payload + 112, response->output_gpu_va);
  put_u32(payload + 120, response->packet_crc32c);
  put_u32(payload + 124, response->output_crc32c);
  put_u64(payload + 128, response->admission_tick);
  put_u64(payload + 136, response->start_tick);
  put_u64(payload + 144, response->end_tick);
  put_u64(payload + 152, response->retire_tick);
  sagr_protocol_recompute_frame_crc(frame,
                                    SAGR_WIRE_DISPATCH_RESULT_FRAME_BYTES);
  *frame_size = SAGR_WIRE_DISPATCH_RESULT_FRAME_BYTES;
  return SAGR_STATUS_SUCCESS;
}

static sagr_status_t decode_dispatch_response_header(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, uint16_t expected_message_type,
    const uint8_t **payload, const char **reason) {
  uint16_t actual_type;
  if (frame == NULL || info == NULL || payload == NULL || reason == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (frame_size != SAGR_WIRE_DISPATCH_RESULT_FRAME_BYTES) {
    *reason = "invalid dispatch result record size";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  actual_type = get_u16(frame + 14);
  if (memcmp(frame, k_magic, sizeof(k_magic)) != 0 ||
      get_u16(frame + 8) != 1 || get_u16(frame + 10) != 0 ||
      get_u16(frame + 12) != SAGR_WIRE_HEADER_BYTES ||
      actual_type != expected_message_type ||
      (actual_type != SAGR_WIRE_MESSAGE_DISPATCH_ACK &&
       actual_type != SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION) ||
      get_u32(frame + 16) != 0 ||
      get_u32(frame + 20) != SAGR_WIRE_DISPATCH_RESULT_PAYLOAD_BYTES) {
    *reason = "invalid dispatch result framing";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (get_u64(frame + 24) == 0 ||
      (expected_request_id != 0 &&
       get_u64(frame + 24) != expected_request_id) ||
      get_u32(frame + 68) != 0 || get_u64(frame + 72) != 0) {
    *reason = "invalid dispatch result request or reserved field";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (frame_crc32c(frame, frame_size) != get_u32(frame + 64)) {
    *reason = "dispatch result CRC32C mismatch";
    return SAGR_STATUS_CHECKSUM_ERROR;
  }
  if (bytes_are_zero(info->daemon_uuid, 16) || info->connection_id == 0 ||
      info->epoch == 0) {
    *reason = "invalid local dispatch session identity";
    return SAGR_STATUS_INVALID_HANDLE;
  }
  if (memcmp(frame + 32, info->daemon_uuid, 16) != 0) {
    *reason = "dispatch result daemon identity mismatch";
    return SAGR_STATUS_INSTANCE_MISMATCH;
  }
  if (get_u64(frame + 48) != info->connection_id ||
      get_u64(frame + 56) != info->epoch) {
    *reason = "dispatch result session identity mismatch";
    return SAGR_STATUS_TOPOLOGY_MISMATCH;
  }
  *payload = frame + SAGR_WIRE_HEADER_BYTES;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_decode_dispatch_response(
    const uint8_t *frame, size_t frame_size,
    const sagr_instance_info_t *info, uint64_t expected_request_id,
    uint16_t expected_message_type, sagr_wire_dispatch_response_t *result,
    int32_t *wire_status, const char **reason) {
  const uint8_t *payload = NULL;
  uint32_t wire;
  sagr_status_t status;
  if (result == NULL || wire_status == NULL || reason == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  *wire_status = -1;
  *reason = "invalid dispatch result";
  status = decode_dispatch_response_header(
      frame, frame_size, info, expected_request_id, expected_message_type,
      &payload, reason);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  wire = get_u32(payload + 4);
  if (get_u16(payload) != SAGR_DISPATCH_PROTOCOL_MAJOR ||
      get_u16(payload + 2) != SAGR_DISPATCH_PROTOCOL_MINOR ||
      get_u16(payload + 10) != 0 || get_u32(payload + 12) != 0 ||
      wire > SAGR_WIRE_STATUS_INTERNAL ||
      ((expected_message_type == SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION ||
        wire == SAGR_WIRE_STATUS_OK) &&
       !dispatch_opcode_is_valid(get_u16(payload + 8)))) {
    *reason = "invalid dispatch result fixed fields";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  *wire_status = (int32_t)wire;
  result->major = get_u16(payload);
  result->minor = get_u16(payload + 2);
  result->status = wire;
  result->opcode = get_u16(payload + 8);
  result->queue_id = get_u64(payload + 16);
  result->queue_generation = get_u64(payload + 24);
  result->queue_sequence = get_u64(payload + 32);
  result->fixture_id = get_u64(payload + 40);
  result->input_allocation_id = get_u64(payload + 48);
  result->input_generation = get_u64(payload + 56);
  result->output_allocation_id = get_u64(payload + 64);
  result->output_generation = get_u64(payload + 72);
  result->signal_id = get_u64(payload + 80);
  result->signal_generation = get_u64(payload + 88);
  result->trace_id = get_u64(payload + 96);
  result->input_gpu_va = get_u64(payload + 104);
  result->output_gpu_va = get_u64(payload + 112);
  result->packet_crc32c = get_u32(payload + 120);
  result->output_crc32c = get_u32(payload + 124);
  result->admission_tick = get_u64(payload + 128);
  result->start_tick = get_u64(payload + 136);
  result->end_tick = get_u64(payload + 144);
  result->retire_tick = get_u64(payload + 152);
  result->request_id = get_u64(frame + 24);
  result->message_type = expected_message_type;
  if (wire != SAGR_WIRE_STATUS_OK) {
    *reason = "daemon rejected pinned dispatch";
    return sagr_protocol_map_wire_status(wire);
  }
  *reason = "pinned dispatch operation succeeded";
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_validate_failed_dispatch_ack(
    const sagr_wire_dispatch_request_t *request,
    const sagr_wire_dispatch_response_t *response) {
  if (request == NULL || response == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (response->status == SAGR_WIRE_STATUS_OK ||
      response->opcode != request->opcode ||
      response->queue_id != request->queue_id ||
      response->queue_generation != request->queue_generation ||
      response->queue_sequence != request->queue_sequence ||
      response->fixture_id != request->fixture_id ||
      response->input_allocation_id != request->input_allocation_id ||
      response->input_generation != request->input_generation ||
      response->output_allocation_id != request->output_allocation_id ||
      response->output_generation != request->output_generation ||
      response->signal_id != request->signal_id ||
      response->signal_generation != request->signal_generation ||
      response->trace_id != 0 || response->input_gpu_va != 0 ||
      response->output_gpu_va != 0 || response->packet_crc32c != 0 ||
      response->output_crc32c != 0 || response->admission_tick != 0 ||
      response->start_tick != 0 || response->end_tick != 0 ||
      response->retire_tick != 0) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  return SAGR_STATUS_SUCCESS;
}

static int kmt_operation_valid(uint16_t operation) {
  return operation >= (uint16_t)SAGR_KMT_OP_OPEN_KFD &&
         operation <= (uint16_t)SAGR_KMT_OP_MODEL_DRM_CALL;
}

static int kmt_words_zero_outside(const uint32_t words[8], uint16_t operation,
                                  int result_words) {
  uint32_t allowed = 0;
  uint32_t index;
  switch (operation) {
    case SAGR_KMT_OP_GET_VERSION: allowed = 0x0fU; break;
    case SAGR_KMT_OP_TOPOLOGY_SNAPSHOT: allowed = 0x3fU; break;
    case SAGR_KMT_OP_ALLOC_MEMORY: allowed = 0x7fU; break;
    case SAGR_KMT_OP_COPY_MEMORY: allowed = result_words ? 0U : 0x1fU; break;
    case SAGR_KMT_OP_QUEUE_CREATE: allowed = result_words ? 0x01U : 0x1fU; break;
    case SAGR_KMT_OP_QUEUE_DOORBELL: allowed = 0x03U; break;
    case SAGR_KMT_OP_EVENT_CREATE:
    case SAGR_KMT_OP_EVENT_SET: allowed = result_words ? 0U : 0x03U; break;
    case SAGR_KMT_OP_EVENT_QUERY: allowed = result_words ? 0x1fU : 0U; break;
    case SAGR_KMT_OP_EVENT_WAIT: allowed = result_words ? 0x03U : 0x0fU; break;
    case SAGR_KMT_OP_MODEL_DRM_CALL: allowed = result_words ? 0U : 0x03U; break;
    case SAGR_KMT_OP_POINTER_INFO: allowed = result_words ? 0x1fU : 0U; break;
    case SAGR_KMT_OP_OPEN_KFD:
    case SAGR_KMT_OP_CLOSE_KFD:
    case SAGR_KMT_OP_FREE_MEMORY:
    case SAGR_KMT_OP_QUEUE_DESTROY:
    case SAGR_KMT_OP_EVENT_DESTROY:
    case SAGR_KMT_OP_EVENT_RESET: allowed = 0U; break;
    default: return 0;
  }
  for (index = 0; index < 8U; ++index) {
    if ((allowed & (UINT32_C(1) << index)) == 0 && words[index] != 0) {
      return 0;
    }
  }
  return 1;
}

static int kmt_reserved_request_zero(const sagr_kmt_envelope_request_t *request) {
  return bytes_are_zero(request->reserved, sizeof(request->reserved));
}

static int kmt_reserved_result_zero(const sagr_kmt_envelope_result_t *result) {
  return bytes_are_zero(result->reserved, sizeof(result->reserved));
}

sagr_status_t sagr_protocol_encode_kmt_request(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_kmt_envelope_request_t *request, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size) {
  uint8_t *payload;
  uint32_t index;
  if (info == NULL || request == NULL || frame == NULL || frame_size == NULL ||
      frame_capacity < SAGR_WIRE_KMT_FRAME_BYTES || request_id == 0 ||
      request->major != SAGR_KMT_PROTOCOL_MAJOR ||
      request->minor != SAGR_KMT_PROTOCOL_MINOR ||
      !kmt_operation_valid(request->operation) || request->flags != 0 ||
      request->operation_sequence == 0 ||
      (request->operation != SAGR_KMT_OP_OPEN_KFD &&
       (request->owner_id == 0 || request->owner_generation == 0)) ||
      !kmt_words_zero_outside(request->argument_words, request->operation, 0) ||
      request->buffer_bytes > SAGR_KMT_BUFFER_BYTES ||
      (request->buffer_bytes == 0 && request->buffer_crc32c != 0) ||
      (request->buffer_bytes != 0 && request->buffer_crc32c !=
                                      sagr_crc32c(request->buffer,
                                                  request->buffer_bytes)) ||
      !bytes_are_zero(request->buffer + request->buffer_bytes,
                      SAGR_KMT_BUFFER_BYTES - request->buffer_bytes) ||
      !kmt_reserved_request_zero(request)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  memset(frame, 0, SAGR_WIRE_KMT_FRAME_BYTES);
  encode_header(frame, SAGR_WIRE_MESSAGE_KMT_REQUEST, SAGR_KMT_PAYLOAD_BYTES,
                request_id, info->daemon_uuid, info->connection_id,
                info->epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, request->major);
  put_u16(payload + 2, request->minor);
  put_u16(payload + 4, request->operation);
  put_u16(payload + 6, request->flags);
  put_u64(payload + 8, request->operation_sequence);
  put_u64(payload + 16, request->owner_id);
  put_u64(payload + 24, request->owner_generation);
  put_u64(payload + 32, request->object_id);
  put_u64(payload + 40, request->object_generation);
  put_u64(payload + 48, request->auxiliary_id);
  put_u64(payload + 56, request->auxiliary_generation);
  for (index = 0; index < SAGR_KMT_ARGUMENT_WORD_COUNT; ++index) {
    put_u32(payload + 64 + (size_t)index * 4U, request->argument_words[index]);
  }
  put_u32(payload + 96, request->buffer_bytes);
  put_u32(payload + 100, request->buffer_crc32c);
  memcpy(payload + 104, request->buffer, SAGR_KMT_BUFFER_BYTES);
  sagr_protocol_recompute_frame_crc(frame, SAGR_WIRE_KMT_FRAME_BYTES);
  *frame_size = SAGR_WIRE_KMT_FRAME_BYTES;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_encode_kmt_result(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_kmt_envelope_result_t *result, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size) {
  uint8_t *payload;
  uint32_t index;
  if (info == NULL || result == NULL || frame == NULL || frame_size == NULL ||
      frame_capacity < SAGR_WIRE_KMT_FRAME_BYTES || request_id == 0 ||
      result->major != SAGR_KMT_PROTOCOL_MAJOR ||
      result->minor != SAGR_KMT_PROTOCOL_MINOR ||
      !kmt_operation_valid(result->operation) || result->flags != 0 ||
      result->operation_sequence == 0 ||
      (result->operation != SAGR_KMT_OP_OPEN_KFD &&
       (result->owner_id == 0 || result->owner_generation == 0)) ||
      !kmt_words_zero_outside(result->result_words, result->operation, 1) ||
      result->buffer_bytes > SAGR_KMT_BUFFER_BYTES ||
      (result->buffer_bytes == 0 && result->buffer_crc32c != 0) ||
      (result->buffer_bytes != 0 && result->buffer_crc32c !=
                                      sagr_crc32c(result->buffer,
                                                  result->buffer_bytes)) ||
      !bytes_are_zero(result->buffer + result->buffer_bytes,
                      SAGR_KMT_BUFFER_BYTES - result->buffer_bytes) ||
      !kmt_reserved_result_zero(result)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  memset(frame, 0, SAGR_WIRE_KMT_FRAME_BYTES);
  encode_header(frame, SAGR_WIRE_MESSAGE_KMT_RESULT, SAGR_KMT_PAYLOAD_BYTES,
                request_id, info->daemon_uuid, info->connection_id,
                info->epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, result->major);
  put_u16(payload + 2, result->minor);
  put_u16(payload + 4, result->operation);
  put_u16(payload + 6, result->flags);
  put_u32(payload + 8, result->status);
  put_u32(payload + 12, (uint32_t)result->wire_status);
  put_u64(payload + 16, result->operation_sequence);
  put_u64(payload + 24, result->owner_id);
  put_u64(payload + 32, result->owner_generation);
  put_u64(payload + 40, result->object_id);
  put_u64(payload + 48, result->object_generation);
  put_u64(payload + 56, result->auxiliary_id);
  put_u64(payload + 64, result->auxiliary_generation);
  for (index = 0; index < SAGR_KMT_ARGUMENT_WORD_COUNT; ++index) {
    put_u32(payload + 72 + (size_t)index * 4U, result->result_words[index]);
  }
  put_u32(payload + 104, result->buffer_bytes);
  put_u32(payload + 108, result->buffer_crc32c);
  memcpy(payload + 112, result->buffer, SAGR_KMT_BUFFER_BYTES);
  sagr_protocol_recompute_frame_crc(frame, SAGR_WIRE_KMT_FRAME_BYTES);
  *frame_size = SAGR_WIRE_KMT_FRAME_BYTES;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t sagr_protocol_decode_kmt_result(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, sagr_kmt_envelope_result_t *result,
    int32_t *wire_status, const char **reason) {
  const uint8_t *payload;
  uint32_t index;
  uint32_t buffer_bytes;
  if (wire_status != NULL) {
    *wire_status = -1;
  }
  if (reason != NULL) {
    *reason = "malformed KMT result";
  }
  if (frame == NULL || info == NULL || result == NULL ||
      frame_size != SAGR_WIRE_KMT_FRAME_BYTES || expected_request_id == 0 ||
      memcmp(frame, k_magic, sizeof(k_magic)) != 0 ||
      get_u16(frame + 8) != 1 || get_u16(frame + 10) != 0 ||
      get_u16(frame + 12) != SAGR_WIRE_HEADER_BYTES ||
      get_u16(frame + 14) != SAGR_WIRE_MESSAGE_KMT_RESULT ||
      get_u32(frame + 20) != SAGR_KMT_PAYLOAD_BYTES ||
      get_u32(frame + 16) != 0 || get_u32(frame + 68) != 0 ||
      !bytes_are_zero(frame + 72, 8) ||
      get_u64(frame + 24) != expected_request_id ||
      get_u64(frame + 48) != info->connection_id ||
      get_u64(frame + 56) != info->epoch ||
      memcmp(frame + 32, info->daemon_uuid, SAGR_UUID_SIZE) != 0 ||
      get_u32(frame + 64) != frame_crc32c(frame, frame_size)) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  memset(result, 0, sizeof(*result));
  result->major = get_u16(payload);
  result->minor = get_u16(payload + 2);
  result->operation = get_u16(payload + 4);
  result->flags = get_u16(payload + 6);
  result->status = get_u32(payload + 8);
  result->wire_status = (int32_t)get_u32(payload + 12);
  result->operation_sequence = get_u64(payload + 16);
  result->owner_id = get_u64(payload + 24);
  result->owner_generation = get_u64(payload + 32);
  result->object_id = get_u64(payload + 40);
  result->object_generation = get_u64(payload + 48);
  result->auxiliary_id = get_u64(payload + 56);
  result->auxiliary_generation = get_u64(payload + 64);
  for (index = 0; index < SAGR_KMT_ARGUMENT_WORD_COUNT; ++index) {
    result->result_words[index] = get_u32(payload + 72 + (size_t)index * 4U);
  }
  result->buffer_bytes = get_u32(payload + 104);
  result->buffer_crc32c = get_u32(payload + 108);
  memcpy(result->buffer, payload + 112, SAGR_KMT_BUFFER_BYTES);
  buffer_bytes = result->buffer_bytes;
  if (result->major != SAGR_KMT_PROTOCOL_MAJOR ||
      result->minor != SAGR_KMT_PROTOCOL_MINOR ||
      !kmt_operation_valid(result->operation) || result->flags != 0 ||
      result->operation_sequence == 0 ||
      (result->operation != SAGR_KMT_OP_OPEN_KFD &&
       (result->owner_id == 0 || result->owner_generation == 0)) ||
      !kmt_words_zero_outside(result->result_words, result->operation, 1) ||
      buffer_bytes > SAGR_KMT_BUFFER_BYTES ||
      (buffer_bytes == 0 && result->buffer_crc32c != 0) ||
      (buffer_bytes != 0 && result->buffer_crc32c !=
                                  sagr_crc32c(result->buffer, buffer_bytes)) ||
      !bytes_are_zero(result->buffer + buffer_bytes,
                      SAGR_KMT_BUFFER_BYTES - buffer_bytes) ||
      !kmt_reserved_result_zero(result)) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (wire_status != NULL) {
    *wire_status = result->wire_status;
  }
  if (reason != NULL) {
    *reason = result->status == SAGR_KMT_STATUS_SUCCESS
                  ? "KMT operation succeeded"
                  : "KMT operation rejected";
  }
  return SAGR_STATUS_SUCCESS;
}

/* CP-0013 A1: the code-object envelope is deliberately isolated from the
 * older fixed-size operation codecs above.  Every field is written explicitly
 * in network byte order; C struct padding never becomes wire state. */

static int
code_object_opcode_valid(uint16_t opcode)
{
  return opcode == SAGR_WIRE_CODE_OBJECT_OPCODE_BEGIN ||
         opcode == SAGR_WIRE_CODE_OBJECT_OPCODE_CHUNK ||
         opcode == SAGR_WIRE_CODE_OBJECT_OPCODE_COMMIT;
}

static int
code_object_digest_nonzero(const uint8_t digest[32])
{
  return !bytes_are_zero(digest, 32);
}

static uint32_t
code_object_expected_chunk_count(uint64_t image_size)
{
  if (image_size == 0U ||
      image_size > SAGR_WIRE_CODE_OBJECT_MAX_IMAGE_BYTES) {
    return 0U;
  }
  return (uint32_t)((image_size +
                     (uint64_t)SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES - 1U) /
                    (uint64_t)SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES);
}

static int
code_object_nul_padded(const char *bytes, size_t size)
{
  size_t index;
  int terminated = 0;
  if (bytes == NULL || size == 0U || bytes[0] == '\0') {
    return 0;
  }
  for (index = 0; index < size; ++index) {
    if (terminated != 0 && bytes[index] != '\0') {
      return 0;
    }
    if (terminated == 0 && (unsigned char)bytes[index] >= 0x80U) {
      return 0;
    }
    if (bytes[index] == '\0') {
      terminated = 1;
    }
  }
  return terminated;
}

static int
code_object_range_within(uint64_t start, uint64_t length,
                         uint64_t container_start,
                         uint64_t container_length)
{
  uint64_t end;
  uint64_t container_end;
  if (start < container_start || start > UINT64_MAX - length ||
      container_start > UINT64_MAX - container_length) {
    return 0;
  }
  end = start + length;
  container_end = container_start + container_length;
  return end <= container_end;
}

static int
code_object_ranges_overlap(uint64_t left_start, uint64_t left_length,
                           uint64_t right_start, uint64_t right_length)
{
  uint64_t left_end;
  uint64_t right_end;
  if (left_start > UINT64_MAX - left_length ||
      right_start > UINT64_MAX - right_length) {
    return 1;
  }
  left_end = left_start + left_length;
  right_end = right_start + right_length;
  return left_start < right_end && right_start < left_end;
}

static int
code_object_segment_valid(const sagr_wire_code_object_segment_t *segment)
{
  if (segment == NULL || segment->type != 1U ||
      segment->memory_size < segment->file_size ||
      (segment->alignment != 0U &&
       (segment->alignment & (segment->alignment - UINT64_C(1))) != 0U) ||
      segment->file_offset > UINT64_MAX - segment->file_size ||
      segment->virtual_address > UINT64_MAX - segment->memory_size) {
    return 0;
  }
  if (segment->flags != 4U && segment->flags != 5U && segment->flags != 6U) {
    return 0;
  }
  return 1;
}

static int
code_object_begin_valid(const sagr_wire_code_object_begin_t *begin)
{
  uint32_t index;
  uint32_t expected_chunks;
  int executable_code_range = 0;
  int read_only_descriptor_range = 0;
  if (begin == NULL || begin->image_size == 0U ||
      begin->image_size > SAGR_WIRE_CODE_OBJECT_MAX_IMAGE_BYTES ||
      begin->chunk_data_bytes != SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES ||
      begin->chunk_count == 0U || begin->segment_count == 0U ||
      begin->segment_count > SAGR_WIRE_CODE_OBJECT_MAX_SEGMENTS ||
      begin->kernel_index >= SAGR_CODE_OBJECT_MAX_KERNELS ||
      !code_object_digest_nonzero(begin->image_sha256) ||
      begin->elf_machine != SAGR_CODE_OBJECT_ELF_MACHINE_AMDGPU ||
      begin->elf_type != SAGR_CODE_OBJECT_ELF_TYPE_DYN ||
      begin->elf_osabi != SAGR_CODE_OBJECT_ELF_OSABI_AMDGPU_HSA ||
      begin->elf_abi_version < 2U || begin->elf_abi_version > 4U ||
      begin->reserved0 != 0U || (begin->elf_flags & UINT32_C(0xff)) != 0x4fU ||
      begin->gfx_target != SAGR_CODE_OBJECT_TARGET_GFX950 ||
      begin->code_object_version < 4U || begin->code_object_version > 6U ||
      begin->metadata_major != 1U ||
      (begin->metadata_minor != 1U && begin->metadata_minor != 2U) ||
      begin->relocation_count != 0U || begin->kernarg_segment_size == 0U ||
      begin->kernarg_segment_align == 0U ||
      (begin->kernarg_segment_align &
       (begin->kernarg_segment_align - 1U)) != 0U ||
      begin->max_flat_workgroup_size == 0U ||
      begin->wavefront_size != 64U || begin->uses_dynamic_stack != 0U ||
      begin->descriptor_size != SAGR_WIRE_CODE_OBJECT_DESCRIPTOR_BYTES ||
      (begin->descriptor_address & UINT64_C(63)) != 0U ||
      begin->code_size == 0U || begin->code_file_offset > begin->image_size ||
      begin->code_size > begin->image_size - begin->code_file_offset ||
      begin->descriptor_file_offset > begin->image_size ||
      begin->descriptor_size >
          begin->image_size - begin->descriptor_file_offset ||
      !code_object_nul_padded(begin->kernel_name,
                              SAGR_WIRE_CODE_OBJECT_NAME_BYTES) ||
      !code_object_nul_padded(begin->symbol,
                              SAGR_WIRE_CODE_OBJECT_NAME_BYTES)) {
    return 0;
  }
  expected_chunks = (uint32_t)((begin->image_size +
                                SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES - 1U) /
                               SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES);
  if (begin->chunk_count != expected_chunks) {
    return 0;
  }
  if (!bytes_are_zero((const uint8_t *)begin->segments +
                          begin->segment_count *
                              sizeof(sagr_wire_code_object_segment_t),
                      (SAGR_WIRE_CODE_OBJECT_MAX_SEGMENTS -
                       begin->segment_count) *
                          sizeof(sagr_wire_code_object_segment_t))) {
    return 0;
  }
  for (index = 0; index < begin->segment_count; ++index) {
    const sagr_wire_code_object_segment_t *segment = &begin->segments[index];
    if (!code_object_segment_valid(&begin->segments[index]) ||
        segment->file_offset > begin->image_size ||
        segment->file_size > begin->image_size - segment->file_offset) {
      return 0;
    }
    if (segment->flags == 5U &&
        code_object_range_within(begin->code_file_offset, begin->code_size,
                                 segment->file_offset, segment->file_size) &&
        code_object_range_within(begin->code_address, begin->code_size,
                                 segment->virtual_address,
                                 segment->memory_size)) {
      executable_code_range = 1;
    }
    if (segment->flags == 4U &&
        code_object_range_within(begin->descriptor_file_offset,
                                 SAGR_WIRE_CODE_OBJECT_DESCRIPTOR_BYTES,
                                 segment->file_offset, segment->file_size) &&
        code_object_range_within(begin->descriptor_address,
                                 SAGR_WIRE_CODE_OBJECT_DESCRIPTOR_BYTES,
                                 segment->virtual_address,
                                 segment->memory_size)) {
      read_only_descriptor_range = 1;
    }
    for (uint32_t other = index + 1U;
         other < begin->segment_count; ++other) {
      const sagr_wire_code_object_segment_t *right = &begin->segments[other];
      if (code_object_ranges_overlap(segment->file_offset, segment->file_size,
                                     right->file_offset, right->file_size) ||
          code_object_ranges_overlap(segment->virtual_address,
                                     segment->memory_size,
                                     right->virtual_address,
                                     right->memory_size)) {
        return 0;
      }
    }
  }
  if (executable_code_range == 0 || read_only_descriptor_range == 0) {
    return 0;
  }
  if (begin->descriptor_kernel_code_entry_byte_offset >= 0) {
    const uint64_t offset =
        (uint64_t)begin->descriptor_kernel_code_entry_byte_offset;
    if (begin->descriptor_address > UINT64_MAX - offset ||
        begin->descriptor_address + offset != begin->code_address) {
      return 0;
    }
  } else {
    const uint64_t offset =
        (uint64_t)(-(begin->descriptor_kernel_code_entry_byte_offset + 1)) +
        UINT64_C(1);
    if (offset > begin->descriptor_address ||
        begin->descriptor_address - offset != begin->code_address) {
      return 0;
    }
  }
  return 1;
}

static int
code_object_success_response_valid(
    const sagr_wire_code_object_response_t *response)
{
  uint32_t expected_count;
  if (response->object_id == 0U || response->generation == 0U ||
      response->image_size == 0U ||
      response->image_size > SAGR_WIRE_CODE_OBJECT_MAX_IMAGE_BYTES ||
      response->kernel_index >= SAGR_CODE_OBJECT_MAX_KERNELS ||
      response->segment_count == 0U ||
      response->segment_count > SAGR_WIRE_CODE_OBJECT_MAX_SEGMENTS ||
      !code_object_digest_nonzero(response->image_sha256) ||
      response->error_code != 0U) {
    return 0;
  }
  if (response->opcode == SAGR_WIRE_CODE_OBJECT_OPCODE_BEGIN) {
    return response->accepted_offset == 0U && response->accepted_count == 0U &&
           response->chunk_index == 0U;
  }
  if (response->opcode == SAGR_WIRE_CODE_OBJECT_OPCODE_COMMIT) {
    return response->accepted_offset == response->image_size &&
           response->accepted_count == response->image_size &&
           response->chunk_index ==
               code_object_expected_chunk_count(response->image_size);
  }
  if (response->accepted_offset > response->image_size ||
      response->accepted_count == 0U ||
      response->accepted_count > SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES ||
      response->accepted_count >
          response->image_size - response->accepted_offset ||
      response->accepted_offset !=
          (uint64_t)response->chunk_index *
              (uint64_t)SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES) {
    return 0;
  }
  expected_count =
      response->image_size - response->accepted_offset >
              SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES
          ? SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES
          : (uint32_t)(response->image_size - response->accepted_offset);
  return response->accepted_count == expected_count;
}

static int
code_object_failed_response_valid(
    const sagr_wire_code_object_response_t *response)
{
  return response->object_id == 0U && response->generation == 0U &&
         response->accepted_offset == 0U && response->accepted_count == 0U &&
         response->chunk_index == 0U && response->mapped_base_va == 0U &&
         response->descriptor_va == 0U && response->code_va == 0U &&
         response->kernarg_va == 0U && response->image_size == 0U &&
         response->kernel_index == 0U && response->segment_count == 0U &&
         !code_object_digest_nonzero(response->image_sha256);
}

static sagr_status_t
decode_code_object_header(const uint8_t *frame, size_t frame_size,
                          const sagr_instance_info_t *info,
                          uint16_t expected_type, const uint8_t **payload,
                          const char **reason)
{
  if (frame == NULL || info == NULL || payload == NULL || reason == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (frame_size != SAGR_WIRE_CODE_OBJECT_FRAME_BYTES ||
      memcmp(frame, k_magic, sizeof(k_magic)) != 0 || get_u16(frame + 8) != 1U ||
      get_u16(frame + 10) != 0U ||
      get_u16(frame + 12) != SAGR_WIRE_HEADER_BYTES ||
      get_u16(frame + 14) != expected_type || get_u32(frame + 16) != 0U ||
      get_u32(frame + 20) != SAGR_WIRE_CODE_OBJECT_PAYLOAD_BYTES ||
      get_u64(frame + 24) == 0U || get_u32(frame + 68) != 0U ||
      !bytes_are_zero(frame + 72, 8U) || bytes_are_zero(info->daemon_uuid, 16) ||
      info->connection_id == 0U || info->epoch == 0U ||
      memcmp(frame + 32, info->daemon_uuid, 16) != 0 ||
      get_u64(frame + 48) != info->connection_id ||
      get_u64(frame + 56) != info->epoch ||
      get_u32(frame + 64) != frame_crc32c(frame, frame_size)) {
    *reason = "invalid code-object frame header";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  *payload = frame + SAGR_WIRE_HEADER_BYTES;
  return SAGR_STATUS_SUCCESS;
}

static void
encode_code_object_segment(uint8_t *payload, size_t offset,
                           const sagr_wire_code_object_segment_t *segment)
{
  put_u32(payload + offset, segment->type);
  put_u32(payload + offset + 4U, segment->flags);
  put_u64(payload + offset + 8U, segment->file_offset);
  put_u64(payload + offset + 16U, segment->virtual_address);
  put_u64(payload + offset + 24U, segment->file_size);
  put_u64(payload + offset + 32U, segment->memory_size);
  put_u64(payload + offset + 40U, segment->alignment);
}

static void
decode_code_object_segment(const uint8_t *payload, size_t offset,
                           sagr_wire_code_object_segment_t *segment)
{
  segment->type = get_u32(payload + offset);
  segment->flags = get_u32(payload + offset + 4U);
  segment->file_offset = get_u64(payload + offset + 8U);
  segment->virtual_address = get_u64(payload + offset + 16U);
  segment->file_size = get_u64(payload + offset + 24U);
  segment->memory_size = get_u64(payload + offset + 32U);
  segment->alignment = get_u64(payload + offset + 40U);
}

sagr_status_t
sagr_protocol_encode_code_object_request(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_code_object_request_t *request, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size)
{
  uint8_t *payload;
  uint32_t index;
  if (info == NULL || request == NULL || frame == NULL || frame_size == NULL ||
      frame_capacity < SAGR_WIRE_CODE_OBJECT_FRAME_BYTES || request_id == 0U ||
      bytes_are_zero(info->daemon_uuid, 16) || info->connection_id == 0U ||
      info->epoch == 0U || request->major != SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MAJOR ||
      request->minor != SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MINOR ||
      request->flags != 0U || !code_object_opcode_valid(request->opcode)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (request->opcode == SAGR_WIRE_CODE_OBJECT_OPCODE_BEGIN) {
    if (request->object_id != 0U || request->generation != 0U ||
        request->image_offset != 0U || request->byte_count != 0U ||
        request->chunk_index != 0U || request->chunk_crc32c != 0U ||
        !code_object_begin_valid(&request->body.begin) ||
        !bytes_are_zero(request->body.begin.descriptor +
                            SAGR_WIRE_CODE_OBJECT_DESCRIPTOR_BYTES,
                        0U)) {
      return SAGR_STATUS_INVALID_ARGUMENT;
    }
  } else if (request->opcode == SAGR_WIRE_CODE_OBJECT_OPCODE_CHUNK) {
    const uint32_t count = request->byte_count;
    if (request->object_id == 0U || request->generation == 0U || count == 0U ||
        count > SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES ||
        request->image_offset > SAGR_WIRE_CODE_OBJECT_MAX_IMAGE_BYTES ||
        request->image_offset + count > SAGR_WIRE_CODE_OBJECT_MAX_IMAGE_BYTES ||
        request->chunk_crc32c !=
            sagr_crc32c(request->body.chunk, count) ||
        !bytes_are_zero(request->body.chunk + count,
                        SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES - count)) {
      return SAGR_STATUS_INVALID_ARGUMENT;
    }
  } else {
    if (request->object_id == 0U || request->generation == 0U ||
        request->image_offset != 0U || request->byte_count == 0U ||
        request->byte_count > SAGR_WIRE_CODE_OBJECT_MAX_IMAGE_BYTES ||
        request->chunk_index !=
            code_object_expected_chunk_count(request->byte_count) ||
        request->chunk_crc32c != 0U ||
        !code_object_digest_nonzero(request->body.commit_sha256)) {
      return SAGR_STATUS_INVALID_ARGUMENT;
    }
  }

  memset(frame, 0, SAGR_WIRE_CODE_OBJECT_FRAME_BYTES);
  encode_header(frame, SAGR_WIRE_MESSAGE_CODE_OBJECT_REQUEST,
                SAGR_WIRE_CODE_OBJECT_PAYLOAD_BYTES, request_id,
                info->daemon_uuid, info->connection_id, info->epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, request->major);
  put_u16(payload + 2U, request->minor);
  put_u16(payload + 4U, request->opcode);
  put_u16(payload + 6U, request->flags);
  put_u64(payload + 8U, request->object_id);
  put_u64(payload + 16U, request->generation);
  put_u64(payload + 24U, request->image_offset);
  put_u32(payload + 32U, request->byte_count);
  put_u32(payload + 36U, request->chunk_index);
  put_u32(payload + 40U, request->chunk_crc32c);
  if (request->opcode == SAGR_WIRE_CODE_OBJECT_OPCODE_BEGIN) {
    const sagr_wire_code_object_begin_t *begin = &request->body.begin;
    put_u64(payload + 48U, begin->image_size);
    put_u32(payload + 56U, begin->chunk_data_bytes);
    put_u32(payload + 60U, begin->chunk_count);
    put_u32(payload + 64U, begin->segment_count);
    put_u32(payload + 68U, begin->kernel_index);
    memcpy(payload + 72U, begin->image_sha256, 32U);
    put_u16(payload + 104U, begin->elf_machine);
    put_u16(payload + 106U, begin->elf_type);
    payload[108U] = begin->elf_osabi;
    payload[109U] = begin->elf_abi_version;
    put_u16(payload + 110U, begin->reserved0);
    put_u32(payload + 112U, begin->elf_flags);
    put_u32(payload + 116U, begin->gfx_target);
    put_u32(payload + 120U, begin->code_object_version);
    put_u32(payload + 124U, begin->metadata_major);
    put_u32(payload + 128U, begin->metadata_minor);
    put_u32(payload + 132U, begin->relocation_count);
    put_u32(payload + 136U, begin->kernarg_segment_size);
    put_u32(payload + 140U, begin->kernarg_segment_align);
    put_u32(payload + 144U, begin->group_segment_fixed_size);
    put_u32(payload + 148U, begin->private_segment_fixed_size);
    put_u32(payload + 152U, begin->max_flat_workgroup_size);
    put_u32(payload + 156U, begin->wavefront_size);
    put_u32(payload + 160U, begin->sgpr_count);
    put_u32(payload + 164U, begin->vgpr_count);
    put_u32(payload + 168U, begin->uses_dynamic_stack);
    put_u32(payload + 172U, begin->descriptor_size);
    put_u64(payload + 176U,
            (uint64_t)begin->descriptor_kernel_code_entry_byte_offset);
    put_u64(payload + 184U, begin->code_address);
    put_u64(payload + 192U, begin->code_file_offset);
    put_u64(payload + 200U, begin->code_size);
    put_u64(payload + 208U, begin->descriptor_address);
    put_u64(payload + 216U, begin->descriptor_file_offset);
    memcpy(payload + 224U, begin->kernel_name,
           SAGR_WIRE_CODE_OBJECT_NAME_BYTES);
    memcpy(payload + 352U, begin->symbol, SAGR_WIRE_CODE_OBJECT_NAME_BYTES);
    memcpy(payload + 480U, begin->descriptor,
           SAGR_WIRE_CODE_OBJECT_DESCRIPTOR_BYTES);
    for (index = 0; index < SAGR_WIRE_CODE_OBJECT_MAX_SEGMENTS; ++index) {
      encode_code_object_segment(payload, 544U + (size_t)index * 48U,
                                 &begin->segments[index]);
    }
  } else if (request->opcode == SAGR_WIRE_CODE_OBJECT_OPCODE_CHUNK) {
    memcpy(payload + 48U, request->body.chunk,
           SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES);
  } else {
    memcpy(payload + 48U, request->body.commit_sha256, 32U);
  }
  sagr_protocol_recompute_frame_crc(frame, SAGR_WIRE_CODE_OBJECT_FRAME_BYTES);
  *frame_size = SAGR_WIRE_CODE_OBJECT_FRAME_BYTES;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t
sagr_protocol_decode_code_object_request(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    sagr_wire_code_object_request_t *request, uint64_t *request_id,
    const char **reason)
{
  const uint8_t *payload;
  uint32_t index;
  sagr_status_t status;
  if (reason != NULL) {
    *reason = "malformed code-object request";
  }
  if (request == NULL || request_id == NULL || reason == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  status = decode_code_object_header(frame, frame_size, info,
                                     SAGR_WIRE_MESSAGE_CODE_OBJECT_REQUEST,
                                     &payload, reason);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  memset(request, 0, sizeof(*request));
  request->major = get_u16(payload);
  request->minor = get_u16(payload + 2U);
  request->opcode = get_u16(payload + 4U);
  request->flags = get_u16(payload + 6U);
  request->object_id = get_u64(payload + 8U);
  request->generation = get_u64(payload + 16U);
  request->image_offset = get_u64(payload + 24U);
  request->byte_count = get_u32(payload + 32U);
  request->chunk_index = get_u32(payload + 36U);
  request->chunk_crc32c = get_u32(payload + 40U);
  if (get_u32(payload + 44U) != 0U ||
      request->major != SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MAJOR ||
      request->minor != SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MINOR ||
      request->flags != 0U || !code_object_opcode_valid(request->opcode)) {
    *reason = "invalid code-object request prefix";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (request->opcode == SAGR_WIRE_CODE_OBJECT_OPCODE_BEGIN) {
    sagr_wire_code_object_begin_t *begin = &request->body.begin;
    begin->image_size = get_u64(payload + 48U);
    begin->chunk_data_bytes = get_u32(payload + 56U);
    begin->chunk_count = get_u32(payload + 60U);
    begin->segment_count = get_u32(payload + 64U);
    begin->kernel_index = get_u32(payload + 68U);
    memcpy(begin->image_sha256, payload + 72U, 32U);
    begin->elf_machine = get_u16(payload + 104U);
    begin->elf_type = get_u16(payload + 106U);
    begin->elf_osabi = payload[108U];
    begin->elf_abi_version = payload[109U];
    begin->reserved0 = get_u16(payload + 110U);
    begin->elf_flags = get_u32(payload + 112U);
    begin->gfx_target = get_u32(payload + 116U);
    begin->code_object_version = get_u32(payload + 120U);
    begin->metadata_major = get_u32(payload + 124U);
    begin->metadata_minor = get_u32(payload + 128U);
    begin->relocation_count = get_u32(payload + 132U);
    begin->kernarg_segment_size = get_u32(payload + 136U);
    begin->kernarg_segment_align = get_u32(payload + 140U);
    begin->group_segment_fixed_size = get_u32(payload + 144U);
    begin->private_segment_fixed_size = get_u32(payload + 148U);
    begin->max_flat_workgroup_size = get_u32(payload + 152U);
    begin->wavefront_size = get_u32(payload + 156U);
    begin->sgpr_count = get_u32(payload + 160U);
    begin->vgpr_count = get_u32(payload + 164U);
    begin->uses_dynamic_stack = get_u32(payload + 168U);
    begin->descriptor_size = get_u32(payload + 172U);
    begin->descriptor_kernel_code_entry_byte_offset =
        (int64_t)get_u64(payload + 176U);
    begin->code_address = get_u64(payload + 184U);
    begin->code_file_offset = get_u64(payload + 192U);
    begin->code_size = get_u64(payload + 200U);
    begin->descriptor_address = get_u64(payload + 208U);
    begin->descriptor_file_offset = get_u64(payload + 216U);
    memcpy(begin->kernel_name, payload + 224U,
           SAGR_WIRE_CODE_OBJECT_NAME_BYTES);
    memcpy(begin->symbol, payload + 352U, SAGR_WIRE_CODE_OBJECT_NAME_BYTES);
    memcpy(begin->descriptor, payload + 480U,
           SAGR_WIRE_CODE_OBJECT_DESCRIPTOR_BYTES);
    for (index = 0; index < SAGR_WIRE_CODE_OBJECT_MAX_SEGMENTS; ++index) {
      decode_code_object_segment(payload, 544U + (size_t)index * 48U,
                                 &begin->segments[index]);
    }
    if (request->object_id != 0U || request->generation != 0U ||
        request->image_offset != 0U || request->byte_count != 0U ||
        request->chunk_index != 0U || request->chunk_crc32c != 0U ||
        !code_object_begin_valid(begin) ||
        !bytes_are_zero(payload + 1312U,
                        SAGR_WIRE_CODE_OBJECT_PAYLOAD_BYTES - 1312U)) {
      *reason = "invalid code-object BEGIN manifest";
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
  } else if (request->opcode == SAGR_WIRE_CODE_OBJECT_OPCODE_CHUNK) {
    memcpy(request->body.chunk, payload + 48U,
           SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES);
    if (request->object_id == 0U || request->generation == 0U ||
        request->byte_count == 0U ||
        request->byte_count > SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES ||
        request->image_offset > SAGR_WIRE_CODE_OBJECT_MAX_IMAGE_BYTES ||
        request->image_offset + request->byte_count >
            SAGR_WIRE_CODE_OBJECT_MAX_IMAGE_BYTES ||
        request->chunk_crc32c !=
            sagr_crc32c(request->body.chunk, request->byte_count) ||
        !bytes_are_zero(request->body.chunk + request->byte_count,
                        SAGR_WIRE_CODE_OBJECT_CHUNK_BYTES -
                            request->byte_count) ||
        !bytes_are_zero(payload + 4016U, 0U)) {
      *reason = "invalid code-object CHUNK";
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
  } else {
    memcpy(request->body.commit_sha256, payload + 48U, 32U);
    if (request->object_id == 0U || request->generation == 0U ||
        request->image_offset != 0U || request->byte_count == 0U ||
        request->byte_count > SAGR_WIRE_CODE_OBJECT_MAX_IMAGE_BYTES ||
        request->chunk_index !=
            code_object_expected_chunk_count(request->byte_count) ||
        request->chunk_crc32c != 0U ||
        !code_object_digest_nonzero(request->body.commit_sha256) ||
        !bytes_are_zero(payload + 80U,
                        SAGR_WIRE_CODE_OBJECT_PAYLOAD_BYTES - 80U)) {
      *reason = "invalid code-object COMMIT";
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
  }
  *request_id = get_u64(frame + 24U);
  *reason = "code-object request decoded";
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t
sagr_protocol_encode_code_object_response(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_code_object_response_t *response, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size)
{
  uint8_t *payload;
  if (info == NULL || response == NULL || frame == NULL || frame_size == NULL ||
      frame_capacity < SAGR_WIRE_CODE_OBJECT_FRAME_BYTES || request_id == 0U ||
      bytes_are_zero(info->daemon_uuid, 16) || info->connection_id == 0U ||
      info->epoch == 0U || response->major != SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MAJOR ||
      response->minor != SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MINOR ||
      response->flags != 0U || response->status > SAGR_WIRE_STATUS_INTERNAL ||
      !code_object_opcode_valid(response->opcode) || response->reserved0 != 0U ||
      response->mapped_base_va != 0U || response->descriptor_va != 0U ||
      response->code_va != 0U || response->kernarg_va != 0U) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if ((response->status == SAGR_WIRE_STATUS_OK &&
       !code_object_success_response_valid(response)) ||
      (response->status != SAGR_WIRE_STATUS_OK &&
       !code_object_failed_response_valid(response))) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  memset(frame, 0, SAGR_WIRE_CODE_OBJECT_FRAME_BYTES);
  encode_header(frame, SAGR_WIRE_MESSAGE_CODE_OBJECT_ACK,
                SAGR_WIRE_CODE_OBJECT_PAYLOAD_BYTES, request_id,
                info->daemon_uuid, info->connection_id, info->epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, response->major);
  put_u16(payload + 2U, response->minor);
  put_u32(payload + 4U, response->status);
  put_u16(payload + 8U, response->opcode);
  put_u16(payload + 10U, response->flags);
  put_u64(payload + 16U, response->object_id);
  put_u64(payload + 24U, response->generation);
  put_u64(payload + 32U, response->accepted_offset);
  put_u32(payload + 40U, response->accepted_count);
  put_u32(payload + 44U, response->chunk_index);
  put_u64(payload + 48U, response->mapped_base_va);
  put_u64(payload + 56U, response->descriptor_va);
  put_u64(payload + 64U, response->code_va);
  put_u64(payload + 72U, response->kernarg_va);
  put_u64(payload + 80U, response->image_size);
  put_u32(payload + 88U, response->kernel_index);
  put_u32(payload + 92U, response->segment_count);
  put_u64(payload + 96U, response->sim_tick);
  memcpy(payload + 104U, response->image_sha256, 32U);
  put_u32(payload + 136U, response->error_code);
  sagr_protocol_recompute_frame_crc(frame, SAGR_WIRE_CODE_OBJECT_FRAME_BYTES);
  *frame_size = SAGR_WIRE_CODE_OBJECT_FRAME_BYTES;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t
sagr_protocol_decode_code_object_response(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, sagr_wire_code_object_response_t *response,
    int32_t *wire_status, const char **reason)
{
  const uint8_t *payload;
  sagr_status_t status;
  if (wire_status != NULL) {
    *wire_status = -1;
  }
  if (reason != NULL) {
    *reason = "malformed code-object ACK";
  }
  if (response == NULL || wire_status == NULL || reason == NULL ||
      expected_request_id == 0U) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  status = decode_code_object_header(frame, frame_size, info,
                                     SAGR_WIRE_MESSAGE_CODE_OBJECT_ACK,
                                     &payload, reason);
  if (status != SAGR_STATUS_SUCCESS || get_u64(frame + 24U) != expected_request_id) {
    if (status == SAGR_STATUS_SUCCESS) {
      *reason = "code-object ACK request identity mismatch";
      status = SAGR_STATUS_PROTOCOL_ERROR;
    }
    return status;
  }
  memset(response, 0, sizeof(*response));
  response->major = get_u16(payload);
  response->minor = get_u16(payload + 2U);
  response->status = get_u32(payload + 4U);
  response->opcode = get_u16(payload + 8U);
  response->flags = get_u16(payload + 10U);
  response->object_id = get_u64(payload + 16U);
  response->generation = get_u64(payload + 24U);
  response->accepted_offset = get_u64(payload + 32U);
  response->accepted_count = get_u32(payload + 40U);
  response->chunk_index = get_u32(payload + 44U);
  response->mapped_base_va = get_u64(payload + 48U);
  response->descriptor_va = get_u64(payload + 56U);
  response->code_va = get_u64(payload + 64U);
  response->kernarg_va = get_u64(payload + 72U);
  response->image_size = get_u64(payload + 80U);
  response->kernel_index = get_u32(payload + 88U);
  response->segment_count = get_u32(payload + 92U);
  response->sim_tick = get_u64(payload + 96U);
  memcpy(response->image_sha256, payload + 104U, 32U);
  response->error_code = get_u32(payload + 136U);
  response->reserved0 = get_u32(payload + 140U);
  response->request_id = get_u64(frame + 24U);
  if (get_u32(payload + 12U) != 0U ||
      response->reserved0 != 0U ||
      !bytes_are_zero(payload + 144U,
                      SAGR_WIRE_CODE_OBJECT_PAYLOAD_BYTES - 144U) ||
      response->major != SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MAJOR ||
      response->minor != SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MINOR ||
      response->flags != 0U || !code_object_opcode_valid(response->opcode) ||
      response->status > SAGR_WIRE_STATUS_INTERNAL ||
      response->mapped_base_va != 0U || response->descriptor_va != 0U ||
      response->code_va != 0U || response->kernarg_va != 0U ||
      (response->status == SAGR_WIRE_STATUS_OK &&
       !code_object_success_response_valid(response)) ||
      (response->status != SAGR_WIRE_STATUS_OK &&
       !code_object_failed_response_valid(response))) {
    *reason = "invalid code-object ACK fields";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  *wire_status = (int32_t)response->status;
  if (response->status != SAGR_WIRE_STATUS_OK) {
    *reason = "daemon rejected code-object operation";
    return sagr_protocol_map_wire_status(response->status);
  }
  *reason = "code-object operation succeeded";
  return SAGR_STATUS_SUCCESS;
}

/* CP-0022 generic object/dispatch records.  This section deliberately uses
 * the v1 framing helpers but has an independent payload version and state
 * validator.  No host pointer, FD, or client-owned AQL bytes are serialized. */

static int
generic_opcode_valid(uint16_t opcode)
{
  return opcode == SAGR_WIRE_GENERIC_OPCODE_MAP_OBJECT ||
         opcode == SAGR_WIRE_GENERIC_OPCODE_ALLOC_KERNARG ||
         opcode == SAGR_WIRE_GENERIC_OPCODE_SUBMIT_AQL ||
         opcode == SAGR_WIRE_GENERIC_OPCODE_UNMAP_OBJECT;
}

static int
generic_capability_selected(const sagr_instance_info_t *info)
{
  const uint64_t selected = info->negotiated_capabilities[
      SAGR_CAPABILITY_GENERIC_DISPATCH_WORD];
  return (selected & SAGR_CAPABILITY_GENERIC_DISPATCH_MASK) != 0 &&
         (selected & SAGR_CAPABILITY_TOPOLOGY_MASK) != 0 &&
         (selected & SAGR_CAPABILITY_QUEUE_MASK) != 0 &&
         (selected & SAGR_CAPABILITY_MEMORY_MASK) != 0 &&
         (selected & SAGR_CAPABILITY_SIGNAL_MASK) != 0 &&
         (selected & SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK) != 0;
}

static int
generic_power_of_two(uint64_t value)
{
  return value != 0U && (value & (value - UINT64_C(1))) == 0U;
}

static int
generic_request_common_valid(const sagr_wire_generic_request_t *request)
{
  return request != NULL &&
         request->major == SAGR_GENERIC_DISPATCH_PROTOCOL_MAJOR &&
         request->minor == SAGR_GENERIC_DISPATCH_PROTOCOL_MINOR &&
         request->flags == 0U && request->reserved0 == 0U &&
         generic_opcode_valid(request->opcode);
}

static int
generic_common_hash_and_name_valid(
    const sagr_wire_generic_request_t *request)
{
  return !bytes_are_zero(request->image_sha256, sizeof(request->image_sha256)) &&
         code_object_nul_padded(request->kernel_name,
                                SAGR_WIRE_GENERIC_KERNEL_NAME_BYTES);
}

static int
generic_common_hash_and_name_zero(
    const sagr_wire_generic_request_t *request)
{
  return bytes_are_zero(request->image_sha256, sizeof(request->image_sha256)) &&
         bytes_are_zero((const uint8_t *)request->kernel_name,
                        SAGR_WIRE_GENERIC_KERNEL_NAME_BYTES);
}

static sagr_status_t
generic_request_validate(const sagr_wire_generic_request_t *request)
{
  if (!generic_request_common_valid(request)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }

  switch (request->opcode) {
    case SAGR_WIRE_GENERIC_OPCODE_MAP_OBJECT: {
      const sagr_wire_generic_map_body_t *body = &request->body.map;
      if (request->object_id == 0U || request->object_generation == 0U ||
          request->mapping_id != 0U || request->mapping_generation != 0U ||
          request->queue_id != 0U || request->queue_generation != 0U ||
          request->queue_sequence != 0U ||
          request->kernel_index >= SAGR_CODE_OBJECT_MAX_KERNELS ||
          !generic_common_hash_and_name_valid(request) ||
          body->kernarg_segment_size == 0U ||
          body->kernarg_segment_size > SAGR_WIRE_GENERIC_MAX_KERNARG_BYTES ||
          body->kernarg_segment_align < 8U ||
          body->kernarg_segment_align > SAGR_MEMORY_ALIGNMENT_64K ||
          !generic_power_of_two(body->kernarg_segment_align) ||
          (body->page_size != SAGR_MEMORY_ALIGNMENT_4K &&
           body->page_size != SAGR_MEMORY_ALIGNMENT_64K)) {
        return SAGR_STATUS_INVALID_ARGUMENT;
      }
      if (body->gfx_target != SAGR_CODE_OBJECT_TARGET_GFX950 ||
          body->relocation_count != 0U) {
        return SAGR_STATUS_NOT_SUPPORTED;
      }
      if (body->descriptor_preload_dwords >
          SAGR_WIRE_GENERIC_MAX_PRELOAD_DWORDS) {
        return SAGR_STATUS_INVALID_ARGUMENT;
      }
      return SAGR_STATUS_SUCCESS;
    }
    case SAGR_WIRE_GENERIC_OPCODE_ALLOC_KERNARG: {
      const sagr_wire_generic_alloc_kernarg_body_t *body =
          &request->body.alloc_kernarg;
      if (request->object_id == 0U || request->object_generation == 0U ||
          request->mapping_id == 0U || request->mapping_generation == 0U ||
          request->queue_id != 0U || request->queue_generation != 0U ||
          request->queue_sequence != 0U || request->kernel_index != 0U ||
          !generic_common_hash_and_name_zero(request) || body->size_bytes == 0U ||
          body->size_bytes > SAGR_WIRE_GENERIC_MAX_KERNARG_BYTES ||
          body->alignment_bytes < 8U ||
          body->alignment_bytes > SAGR_MEMORY_ALIGNMENT_64K ||
          !generic_power_of_two(body->alignment_bytes) ||
          body->allocation_flags != 0U || body->reserved0 != 0U) {
        return SAGR_STATUS_INVALID_ARGUMENT;
      }
      return SAGR_STATUS_SUCCESS;
    }
    case SAGR_WIRE_GENERIC_OPCODE_SUBMIT_AQL: {
      const sagr_wire_generic_submit_body_t *body = &request->body.submit;
      const uint64_t workgroup_size =
          (uint64_t)body->workgroup_x * (uint64_t)body->workgroup_y *
          (uint64_t)body->workgroup_z;
      const uint64_t expected_workgroup_size =
          (uint64_t)body->num_warps * (uint64_t)body->wavefront_size;
      if (request->object_id == 0U || request->object_generation == 0U ||
          request->mapping_id == 0U || request->mapping_generation == 0U ||
          request->queue_id == 0U || request->queue_generation == 0U ||
          request->queue_sequence == 0U ||
          request->kernel_index >= SAGR_CODE_OBJECT_MAX_KERNELS ||
          !generic_common_hash_and_name_valid(request) ||
          body->kernarg_allocation_id == 0U ||
          body->kernarg_generation == 0U || body->kernarg_size == 0U ||
          body->kernarg_size > SAGR_WIRE_GENERIC_MAX_KERNARG_BYTES ||
          body->kernarg_offset >
              SAGR_WIRE_GENERIC_MAX_KERNARG_BYTES - body->kernarg_size ||
          body->signal_id == 0U || body->signal_generation == 0U ||
          body->expected_signal_value_bits != UINT64_C(1) ||
          body->grid_x == 0U || body->grid_y == 0U || body->grid_z == 0U ||
          body->workgroup_x == 0U || body->workgroup_y == 0U ||
          body->workgroup_z == 0U ||
          body->workgroup_x > SAGR_WIRE_GENERIC_MAX_WORKGROUP_DIMENSION ||
          body->workgroup_y > SAGR_WIRE_GENERIC_MAX_WORKGROUP_DIMENSION ||
          body->workgroup_z > SAGR_WIRE_GENERIC_MAX_WORKGROUP_DIMENSION ||
          workgroup_size > SAGR_WIRE_GENERIC_MAX_WORKGROUP_DIMENSION ||
          body->grid_x < body->workgroup_x ||
          body->grid_y < body->workgroup_y ||
          body->grid_z < body->workgroup_z ||
          body->num_warps == 0U || body->num_warps > SAGR_WIRE_GENERIC_MAX_WARPS ||
          body->num_ctas == 0U || body->num_ctas > SAGR_WIRE_GENERIC_MAX_CTAS ||
          body->shared_memory_bytes > SAGR_WIRE_GENERIC_MAX_SHARED_BYTES ||
          body->wavefront_size != 64U ||
          expected_workgroup_size != workgroup_size ||
          body->launch_flags != 0U ||
          body->reserved0 != 0U) {
        return SAGR_STATUS_INVALID_ARGUMENT;
      }
      return SAGR_STATUS_SUCCESS;
    }
    case SAGR_WIRE_GENERIC_OPCODE_UNMAP_OBJECT:
      if (request->object_id == 0U || request->object_generation == 0U ||
          request->mapping_id == 0U || request->mapping_generation == 0U ||
          request->queue_id != 0U || request->queue_generation != 0U ||
          request->queue_sequence != 0U || request->kernel_index != 0U ||
          !generic_common_hash_and_name_zero(request) ||
          !bytes_are_zero((const uint8_t *)&request->body,
                          sizeof(request->body))) {
        return SAGR_STATUS_INVALID_ARGUMENT;
      }
      return SAGR_STATUS_SUCCESS;
    default:
      return SAGR_STATUS_INVALID_ARGUMENT;
  }
}

static sagr_status_t
decode_generic_header(const uint8_t *frame, size_t frame_size,
                      const sagr_instance_info_t *info, uint16_t expected_type,
                      const uint8_t **payload, const char **reason)
{
  if (frame == NULL || info == NULL || payload == NULL || reason == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (frame_size != SAGR_WIRE_GENERIC_FRAME_BYTES ||
      memcmp(frame, k_magic, sizeof(k_magic)) != 0 || get_u16(frame + 8U) != 1U ||
      get_u16(frame + 10U) != 0U ||
      get_u16(frame + 12U) != SAGR_WIRE_HEADER_BYTES ||
      get_u16(frame + 14U) != expected_type || get_u32(frame + 16U) != 0U ||
      get_u32(frame + 20U) != SAGR_WIRE_GENERIC_PAYLOAD_BYTES ||
      get_u64(frame + 24U) == 0U || get_u32(frame + 68U) != 0U ||
      !bytes_are_zero(frame + 72U, 8U) || bytes_are_zero(info->daemon_uuid, 16U) ||
      info->connection_id == 0U || info->epoch == 0U ||
      memcmp(frame + 32U, info->daemon_uuid, 16U) != 0 ||
      get_u64(frame + 48U) != info->connection_id ||
      get_u64(frame + 56U) != info->epoch) {
    *reason = "invalid generic dispatch frame header";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (get_u32(frame + 64U) != frame_crc32c(frame, frame_size)) {
    *reason = "generic dispatch CRC32C mismatch";
    return SAGR_STATUS_CHECKSUM_ERROR;
  }
  if (!generic_capability_selected(info)) {
    *reason = "generic dispatch capability was not negotiated";
    return SAGR_STATUS_CAPABILITY_MISMATCH;
  }
  *payload = frame + SAGR_WIRE_HEADER_BYTES;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t
sagr_protocol_encode_generic_dispatch_request(
    const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_generic_request_t *request, uint8_t *frame,
    size_t frame_capacity, size_t *frame_size)
{
  uint8_t *payload;
  sagr_status_t status;
  if (info == NULL || request == NULL || frame == NULL || frame_size == NULL ||
      frame_capacity < SAGR_WIRE_GENERIC_FRAME_BYTES || request_id == 0U ||
      bytes_are_zero(info->daemon_uuid, 16U) || info->connection_id == 0U ||
      info->epoch == 0U) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (!generic_capability_selected(info)) {
    return SAGR_STATUS_CAPABILITY_MISMATCH;
  }
  status = generic_request_validate(request);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }

  memset(frame, 0, SAGR_WIRE_GENERIC_FRAME_BYTES);
  encode_header(frame, SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_REQUEST,
                SAGR_WIRE_GENERIC_PAYLOAD_BYTES, request_id, info->daemon_uuid,
                info->connection_id, info->epoch);
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  put_u16(payload, request->major);
  put_u16(payload + 2U, request->minor);
  put_u16(payload + 4U, request->opcode);
  put_u16(payload + 6U, request->flags);
  put_u64(payload + 8U, request->object_id);
  put_u64(payload + 16U, request->object_generation);
  put_u64(payload + 24U, request->mapping_id);
  put_u64(payload + 32U, request->mapping_generation);
  put_u64(payload + 40U, request->queue_id);
  put_u64(payload + 48U, request->queue_generation);
  put_u64(payload + 56U, request->queue_sequence);
  put_u32(payload + 64U, request->kernel_index);
  memcpy(payload + 72U, request->image_sha256, 32U);
  memcpy(payload + 104U, request->kernel_name,
         SAGR_WIRE_GENERIC_KERNEL_NAME_BYTES);
  if (request->opcode == SAGR_WIRE_GENERIC_OPCODE_MAP_OBJECT) {
    const sagr_wire_generic_map_body_t *body = &request->body.map;
    put_u32(payload + 232U, body->gfx_target);
    put_u32(payload + 236U, body->relocation_count);
    put_u32(payload + 240U, body->kernarg_segment_size);
    put_u32(payload + 244U, body->kernarg_segment_align);
    put_u32(payload + 248U, body->descriptor_preload_dwords);
    put_u32(payload + 252U, body->page_size);
  } else if (request->opcode == SAGR_WIRE_GENERIC_OPCODE_ALLOC_KERNARG) {
    const sagr_wire_generic_alloc_kernarg_body_t *body =
        &request->body.alloc_kernarg;
    put_u64(payload + 232U, body->size_bytes);
    put_u64(payload + 240U, body->alignment_bytes);
    put_u32(payload + 248U, body->allocation_flags);
    put_u32(payload + 252U, body->reserved0);
  } else if (request->opcode == SAGR_WIRE_GENERIC_OPCODE_SUBMIT_AQL) {
    const sagr_wire_generic_submit_body_t *body = &request->body.submit;
    put_u64(payload + 232U, body->kernarg_allocation_id);
    put_u64(payload + 240U, body->kernarg_generation);
    put_u64(payload + 248U, body->kernarg_offset);
    put_u64(payload + 256U, body->kernarg_size);
    put_u64(payload + 264U, body->signal_id);
    put_u64(payload + 272U, body->signal_generation);
    put_u64(payload + 280U, body->expected_signal_value_bits);
    put_u32(payload + 288U, body->grid_x);
    put_u32(payload + 292U, body->grid_y);
    put_u32(payload + 296U, body->grid_z);
    put_u32(payload + 300U, body->workgroup_x);
    put_u32(payload + 304U, body->workgroup_y);
    put_u32(payload + 308U, body->workgroup_z);
    put_u32(payload + 312U, body->num_warps);
    put_u32(payload + 316U, body->num_ctas);
    put_u32(payload + 320U, body->shared_memory_bytes);
    put_u32(payload + 324U, body->wavefront_size);
    put_u32(payload + 328U, body->launch_flags);
    put_u32(payload + 332U, body->reserved0);
  }
  sagr_protocol_recompute_frame_crc(frame, SAGR_WIRE_GENERIC_FRAME_BYTES);
  *frame_size = SAGR_WIRE_GENERIC_FRAME_BYTES;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t
sagr_protocol_decode_generic_dispatch_request(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    sagr_wire_generic_request_t *request, uint64_t *request_id,
    const char **reason)
{
  const uint8_t *payload = NULL;
  sagr_status_t status;
  uint32_t active_end = SAGR_WIRE_GENERIC_COMMON_BYTES;
  if (reason != NULL) {
    *reason = "malformed generic dispatch request";
  }
  if (request == NULL || request_id == NULL || reason == NULL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  status = decode_generic_header(frame, frame_size, info,
                                 SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_REQUEST,
                                 &payload, reason);
  if (status != SAGR_STATUS_SUCCESS) {
    return status;
  }
  memset(request, 0, sizeof(*request));
  request->major = get_u16(payload);
  request->minor = get_u16(payload + 2U);
  request->opcode = get_u16(payload + 4U);
  request->flags = get_u16(payload + 6U);
  request->object_id = get_u64(payload + 8U);
  request->object_generation = get_u64(payload + 16U);
  request->mapping_id = get_u64(payload + 24U);
  request->mapping_generation = get_u64(payload + 32U);
  request->queue_id = get_u64(payload + 40U);
  request->queue_generation = get_u64(payload + 48U);
  request->queue_sequence = get_u64(payload + 56U);
  request->kernel_index = get_u32(payload + 64U);
  request->reserved0 = get_u32(payload + 68U);
  memcpy(request->image_sha256, payload + 72U, 32U);
  memcpy(request->kernel_name, payload + 104U,
         SAGR_WIRE_GENERIC_KERNEL_NAME_BYTES);
  if (request->opcode == SAGR_WIRE_GENERIC_OPCODE_MAP_OBJECT) {
    sagr_wire_generic_map_body_t *body = &request->body.map;
    body->gfx_target = get_u32(payload + 232U);
    body->relocation_count = get_u32(payload + 236U);
    body->kernarg_segment_size = get_u32(payload + 240U);
    body->kernarg_segment_align = get_u32(payload + 244U);
    body->descriptor_preload_dwords = get_u32(payload + 248U);
    body->page_size = get_u32(payload + 252U);
    active_end = 256U;
  } else if (request->opcode == SAGR_WIRE_GENERIC_OPCODE_ALLOC_KERNARG) {
    sagr_wire_generic_alloc_kernarg_body_t *body =
        &request->body.alloc_kernarg;
    body->size_bytes = get_u64(payload + 232U);
    body->alignment_bytes = get_u64(payload + 240U);
    body->allocation_flags = get_u32(payload + 248U);
    body->reserved0 = get_u32(payload + 252U);
    active_end = 256U;
  } else if (request->opcode == SAGR_WIRE_GENERIC_OPCODE_SUBMIT_AQL) {
    sagr_wire_generic_submit_body_t *body = &request->body.submit;
    body->kernarg_allocation_id = get_u64(payload + 232U);
    body->kernarg_generation = get_u64(payload + 240U);
    body->kernarg_offset = get_u64(payload + 248U);
    body->kernarg_size = get_u64(payload + 256U);
    body->signal_id = get_u64(payload + 264U);
    body->signal_generation = get_u64(payload + 272U);
    body->expected_signal_value_bits = get_u64(payload + 280U);
    body->grid_x = get_u32(payload + 288U);
    body->grid_y = get_u32(payload + 292U);
    body->grid_z = get_u32(payload + 296U);
    body->workgroup_x = get_u32(payload + 300U);
    body->workgroup_y = get_u32(payload + 304U);
    body->workgroup_z = get_u32(payload + 308U);
    body->num_warps = get_u32(payload + 312U);
    body->num_ctas = get_u32(payload + 316U);
    body->shared_memory_bytes = get_u32(payload + 320U);
    body->wavefront_size = get_u32(payload + 324U);
    body->launch_flags = get_u32(payload + 328U);
    body->reserved0 = get_u32(payload + 332U);
    active_end = 336U;
  } else if (request->opcode != SAGR_WIRE_GENERIC_OPCODE_UNMAP_OBJECT) {
    *reason = "unknown generic dispatch opcode";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (get_u32(payload + 68U) != 0U ||
      !bytes_are_zero(payload + active_end,
                      SAGR_WIRE_GENERIC_PAYLOAD_BYTES - active_end)) {
    *reason = "noncanonical generic dispatch request padding";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  status = generic_request_validate(request);
  if (status != SAGR_STATUS_SUCCESS) {
    *reason = status == SAGR_STATUS_NOT_SUPPORTED
                  ? "generic dispatch request uses unsupported loader semantics"
                  : "invalid generic dispatch request fields";
    return status == SAGR_STATUS_NOT_SUPPORTED ? status
                                                : SAGR_STATUS_PROTOCOL_ERROR;
  }
  *request_id = get_u64(frame + 24U);
  *reason = "generic dispatch request decoded";
  return SAGR_STATUS_SUCCESS;
}

enum {
  GENERIC_RESPONSE_TAIL = 304,
  GENERIC_RESPONSE_MAJOR = 0,
  GENERIC_RESPONSE_MINOR = 2,
  GENERIC_RESPONSE_STATUS = 4,
  GENERIC_RESPONSE_OPCODE = 8,
  GENERIC_RESPONSE_FLAGS = 10,
  GENERIC_RESPONSE_ERROR = 12,
  GENERIC_RESPONSE_OBJECT = 16,
  GENERIC_RESPONSE_OBJECT_GEN = 24,
  GENERIC_RESPONSE_MAPPING = 32,
  GENERIC_RESPONSE_MAPPING_GEN = 40,
  GENERIC_RESPONSE_MAPPED_BASE = 48,
  GENERIC_RESPONSE_MAPPED_END = 56,
  GENERIC_RESPONSE_DESCRIPTOR = 64,
  GENERIC_RESPONSE_CODE = 72,
  GENERIC_RESPONSE_ENTRY = 80,
  GENERIC_RESPONSE_MAPPED_BYTES = 88,
  GENERIC_RESPONSE_KERNARG = 96,
  GENERIC_RESPONSE_KERNARG_GEN = 104,
  GENERIC_RESPONSE_KERNARG_VA = 112,
  GENERIC_RESPONSE_KERNARG_SIZE = 120,
  GENERIC_RESPONSE_KERNARG_ALIGN = 128,
  GENERIC_RESPONSE_KERNEL_INDEX = 136,
  GENERIC_RESPONSE_SEGMENTS = 140,
  GENERIC_RESPONSE_PRELOAD = 144,
  GENERIC_RESPONSE_RESERVED = 148,
  GENERIC_RESPONSE_TICKET = 152,
  GENERIC_RESPONSE_TRACE = 160,
  GENERIC_RESPONSE_QUEUE = 168,
  GENERIC_RESPONSE_QUEUE_GEN = 176,
  GENERIC_RESPONSE_QUEUE_SEQ = 184,
  GENERIC_RESPONSE_SIGNAL = 192,
  GENERIC_RESPONSE_SIGNAL_GEN = 200,
  GENERIC_RESPONSE_SIGNAL_VALUE = 208,
  GENERIC_RESPONSE_PACKET_VA = 216,
  GENERIC_RESPONSE_PACKET_CRC = 224,
  GENERIC_RESPONSE_OUTPUT_CRC = 228,
  GENERIC_RESPONSE_SIM_TICK = 232,
  GENERIC_RESPONSE_ADMISSION_TICK = 240,
  GENERIC_RESPONSE_START_TICK = 248,
  GENERIC_RESPONSE_END_TICK = 256,
  GENERIC_RESPONSE_RETIRE_TICK = 264,
  GENERIC_RESPONSE_DIGEST = 272
};

static int
generic_response_fields_zero(const sagr_wire_generic_response_t *response)
{
  return response->object_id == 0U && response->object_generation == 0U &&
         response->mapping_id == 0U && response->mapping_generation == 0U &&
         response->mapped_base_va == 0U && response->mapped_end_va == 0U &&
         response->descriptor_va == 0U && response->code_va == 0U &&
         response->entry_va == 0U && response->mapped_bytes == 0U &&
         response->kernarg_allocation_id == 0U &&
         response->kernarg_generation == 0U && response->kernarg_va == 0U &&
         response->kernarg_size == 0U && response->kernarg_alignment == 0U &&
         response->kernel_index == 0U && response->segment_count == 0U &&
         response->descriptor_preload_dwords == 0U && response->reserved0 == 0U &&
         response->ticket_id == 0U && response->trace_id == 0U &&
         response->queue_id == 0U && response->queue_generation == 0U &&
         response->queue_sequence == 0U && response->signal_id == 0U &&
         response->signal_generation == 0U && response->signal_value_bits == 0U &&
         response->packet_va == 0U && response->packet_crc32c == 0U &&
         response->output_crc32c == 0U && response->sim_tick == 0U &&
         response->admission_tick == 0U && response->start_tick == 0U &&
         response->end_tick == 0U && response->retire_tick == 0U &&
         bytes_are_zero(response->image_sha256, sizeof(response->image_sha256));
}

static int
generic_response_success_valid(const sagr_wire_generic_response_t *response,
                               uint16_t message_type)
{
  if (response->error_code != 0U || response->flags != 0U ||
      response->reserved0 != 0U || response->object_id == 0U ||
      response->object_generation == 0U || response->mapping_id == 0U ||
      response->mapping_generation == 0U ||
      (response->opcode != SAGR_WIRE_GENERIC_OPCODE_UNMAP_OBJECT &&
       bytes_are_zero(response->image_sha256,
                      sizeof(response->image_sha256)))) {
    return 0;
  }
  switch (response->opcode) {
    case SAGR_WIRE_GENERIC_OPCODE_MAP_OBJECT:
      return message_type == SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_ACK &&
             response->mapped_base_va != 0U &&
             response->mapped_end_va > response->mapped_base_va &&
             response->mapped_end_va - response->mapped_base_va >=
                 response->mapped_bytes &&
             response->descriptor_va != 0U && response->code_va != 0U &&
             response->entry_va != 0U && response->mapped_bytes != 0U &&
             response->segment_count != 0U &&
             response->descriptor_preload_dwords <=
                 SAGR_WIRE_GENERIC_MAX_PRELOAD_DWORDS &&
             response->kernarg_allocation_id == 0U &&
             response->kernarg_generation == 0U && response->kernarg_va == 0U &&
             response->kernarg_size == 0U && response->kernarg_alignment == 0U &&
             response->ticket_id == 0U && response->trace_id == 0U &&
             response->queue_id == 0U && response->queue_generation == 0U &&
             response->queue_sequence == 0U && response->signal_id == 0U &&
             response->signal_generation == 0U &&
             response->signal_value_bits == 0U && response->packet_va == 0U &&
             response->packet_crc32c == 0U && response->output_crc32c == 0U &&
             response->sim_tick == 0U && response->admission_tick == 0U &&
             response->start_tick == 0U && response->end_tick == 0U &&
             response->retire_tick == 0U;
    case SAGR_WIRE_GENERIC_OPCODE_ALLOC_KERNARG:
      return message_type == SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_ACK &&
             response->mapped_base_va == 0U && response->mapped_end_va == 0U &&
             response->descriptor_va == 0U && response->code_va == 0U &&
             response->entry_va == 0U && response->mapped_bytes == 0U &&
             response->kernarg_allocation_id != 0U &&
             response->kernarg_generation != 0U && response->kernarg_va != 0U &&
             response->kernarg_size != 0U &&
             response->kernarg_size <= SAGR_WIRE_GENERIC_MAX_KERNARG_BYTES &&
             generic_power_of_two(response->kernarg_alignment) &&
             response->kernarg_alignment >= 8U &&
             response->kernarg_alignment <= SAGR_MEMORY_ALIGNMENT_64K &&
             response->ticket_id == 0U && response->trace_id == 0U &&
             response->queue_id == 0U && response->queue_generation == 0U &&
             response->queue_sequence == 0U && response->signal_id == 0U &&
             response->signal_generation == 0U &&
             response->signal_value_bits == 0U && response->packet_va == 0U &&
             response->packet_crc32c == 0U && response->output_crc32c == 0U &&
             response->sim_tick == 0U && response->admission_tick == 0U &&
             response->start_tick == 0U && response->end_tick == 0U &&
             response->retire_tick == 0U && response->kernel_index == 0U &&
             response->segment_count == 0U &&
             response->descriptor_preload_dwords == 0U;
    case SAGR_WIRE_GENERIC_OPCODE_UNMAP_OBJECT:
      return message_type == SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_ACK &&
             response->mapped_base_va == 0U && response->mapped_end_va == 0U &&
             response->descriptor_va == 0U && response->code_va == 0U &&
             response->entry_va == 0U && response->mapped_bytes == 0U &&
             response->kernarg_allocation_id == 0U &&
             response->kernarg_generation == 0U && response->kernarg_va == 0U &&
             response->kernarg_size == 0U && response->kernarg_alignment == 0U &&
             response->ticket_id == 0U && response->trace_id == 0U &&
             response->queue_id == 0U && response->queue_generation == 0U &&
             response->queue_sequence == 0U && response->signal_id == 0U &&
             response->signal_generation == 0U && response->signal_value_bits == 0U &&
             response->packet_va == 0U && response->packet_crc32c == 0U &&
             response->output_crc32c == 0U && response->sim_tick == 0U &&
             response->admission_tick == 0U && response->start_tick == 0U &&
             response->end_tick == 0U && response->retire_tick == 0U &&
             response->kernel_index == 0U && response->segment_count == 0U &&
             response->descriptor_preload_dwords == 0U;
    case SAGR_WIRE_GENERIC_OPCODE_SUBMIT_AQL:
      return (message_type == SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_ACK ||
              message_type == SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_COMPLETION) &&
             response->queue_id != 0U && response->queue_generation != 0U &&
             response->queue_sequence != 0U && response->signal_id != 0U &&
             response->signal_generation != 0U && response->ticket_id != 0U &&
             response->trace_id != 0U && response->packet_va != 0U &&
             response->kernarg_allocation_id != 0U &&
             response->kernarg_generation != 0U && response->kernarg_va != 0U &&
             response->kernarg_size != 0U &&
             response->kernarg_size <= SAGR_WIRE_GENERIC_MAX_KERNARG_BYTES &&
             generic_power_of_two(response->kernarg_alignment) &&
             response->kernarg_alignment >= 8U &&
             response->kernarg_alignment <= SAGR_MEMORY_ALIGNMENT_64K;
    default:
      return 0;
  }
}

static void
generic_response_encode_payload(uint8_t *payload,
                                const sagr_wire_generic_response_t *response)
{
  put_u16(payload + GENERIC_RESPONSE_MAJOR, response->major);
  put_u16(payload + GENERIC_RESPONSE_MINOR, response->minor);
  put_u32(payload + GENERIC_RESPONSE_STATUS, response->status);
  put_u16(payload + GENERIC_RESPONSE_OPCODE, response->opcode);
  put_u16(payload + GENERIC_RESPONSE_FLAGS, response->flags);
  put_u32(payload + GENERIC_RESPONSE_ERROR, response->error_code);
  put_u64(payload + GENERIC_RESPONSE_OBJECT, response->object_id);
  put_u64(payload + GENERIC_RESPONSE_OBJECT_GEN, response->object_generation);
  put_u64(payload + GENERIC_RESPONSE_MAPPING, response->mapping_id);
  put_u64(payload + GENERIC_RESPONSE_MAPPING_GEN, response->mapping_generation);
  put_u64(payload + GENERIC_RESPONSE_MAPPED_BASE, response->mapped_base_va);
  put_u64(payload + GENERIC_RESPONSE_MAPPED_END, response->mapped_end_va);
  put_u64(payload + GENERIC_RESPONSE_DESCRIPTOR, response->descriptor_va);
  put_u64(payload + GENERIC_RESPONSE_CODE, response->code_va);
  put_u64(payload + GENERIC_RESPONSE_ENTRY, response->entry_va);
  put_u64(payload + GENERIC_RESPONSE_MAPPED_BYTES, response->mapped_bytes);
  put_u64(payload + GENERIC_RESPONSE_KERNARG, response->kernarg_allocation_id);
  put_u64(payload + GENERIC_RESPONSE_KERNARG_GEN, response->kernarg_generation);
  put_u64(payload + GENERIC_RESPONSE_KERNARG_VA, response->kernarg_va);
  put_u64(payload + GENERIC_RESPONSE_KERNARG_SIZE, response->kernarg_size);
  put_u64(payload + GENERIC_RESPONSE_KERNARG_ALIGN, response->kernarg_alignment);
  put_u32(payload + GENERIC_RESPONSE_KERNEL_INDEX, response->kernel_index);
  put_u32(payload + GENERIC_RESPONSE_SEGMENTS, response->segment_count);
  put_u32(payload + GENERIC_RESPONSE_PRELOAD,
          response->descriptor_preload_dwords);
  put_u32(payload + GENERIC_RESPONSE_RESERVED, response->reserved0);
  put_u64(payload + GENERIC_RESPONSE_TICKET, response->ticket_id);
  put_u64(payload + GENERIC_RESPONSE_TRACE, response->trace_id);
  put_u64(payload + GENERIC_RESPONSE_QUEUE, response->queue_id);
  put_u64(payload + GENERIC_RESPONSE_QUEUE_GEN, response->queue_generation);
  put_u64(payload + GENERIC_RESPONSE_QUEUE_SEQ, response->queue_sequence);
  put_u64(payload + GENERIC_RESPONSE_SIGNAL, response->signal_id);
  put_u64(payload + GENERIC_RESPONSE_SIGNAL_GEN, response->signal_generation);
  put_u64(payload + GENERIC_RESPONSE_SIGNAL_VALUE,
          response->signal_value_bits);
  put_u64(payload + GENERIC_RESPONSE_PACKET_VA, response->packet_va);
  put_u32(payload + GENERIC_RESPONSE_PACKET_CRC, response->packet_crc32c);
  put_u32(payload + GENERIC_RESPONSE_OUTPUT_CRC, response->output_crc32c);
  put_u64(payload + GENERIC_RESPONSE_SIM_TICK, response->sim_tick);
  put_u64(payload + GENERIC_RESPONSE_ADMISSION_TICK,
          response->admission_tick);
  put_u64(payload + GENERIC_RESPONSE_START_TICK, response->start_tick);
  put_u64(payload + GENERIC_RESPONSE_END_TICK, response->end_tick);
  put_u64(payload + GENERIC_RESPONSE_RETIRE_TICK, response->retire_tick);
  memcpy(payload + GENERIC_RESPONSE_DIGEST, response->image_sha256, 32U);
}

static void
generic_response_decode_payload(const uint8_t *payload,
                                sagr_wire_generic_response_t *response)
{
  memset(response, 0, sizeof(*response));
  response->major = get_u16(payload + GENERIC_RESPONSE_MAJOR);
  response->minor = get_u16(payload + GENERIC_RESPONSE_MINOR);
  response->status = get_u32(payload + GENERIC_RESPONSE_STATUS);
  response->opcode = get_u16(payload + GENERIC_RESPONSE_OPCODE);
  response->flags = get_u16(payload + GENERIC_RESPONSE_FLAGS);
  response->error_code = get_u32(payload + GENERIC_RESPONSE_ERROR);
  response->object_id = get_u64(payload + GENERIC_RESPONSE_OBJECT);
  response->object_generation = get_u64(payload + GENERIC_RESPONSE_OBJECT_GEN);
  response->mapping_id = get_u64(payload + GENERIC_RESPONSE_MAPPING);
  response->mapping_generation = get_u64(payload + GENERIC_RESPONSE_MAPPING_GEN);
  response->mapped_base_va = get_u64(payload + GENERIC_RESPONSE_MAPPED_BASE);
  response->mapped_end_va = get_u64(payload + GENERIC_RESPONSE_MAPPED_END);
  response->descriptor_va = get_u64(payload + GENERIC_RESPONSE_DESCRIPTOR);
  response->code_va = get_u64(payload + GENERIC_RESPONSE_CODE);
  response->entry_va = get_u64(payload + GENERIC_RESPONSE_ENTRY);
  response->mapped_bytes = get_u64(payload + GENERIC_RESPONSE_MAPPED_BYTES);
  response->kernarg_allocation_id = get_u64(payload + GENERIC_RESPONSE_KERNARG);
  response->kernarg_generation = get_u64(payload + GENERIC_RESPONSE_KERNARG_GEN);
  response->kernarg_va = get_u64(payload + GENERIC_RESPONSE_KERNARG_VA);
  response->kernarg_size = get_u64(payload + GENERIC_RESPONSE_KERNARG_SIZE);
  response->kernarg_alignment = get_u64(payload + GENERIC_RESPONSE_KERNARG_ALIGN);
  response->kernel_index = get_u32(payload + GENERIC_RESPONSE_KERNEL_INDEX);
  response->segment_count = get_u32(payload + GENERIC_RESPONSE_SEGMENTS);
  response->descriptor_preload_dwords = get_u32(payload + GENERIC_RESPONSE_PRELOAD);
  response->reserved0 = get_u32(payload + GENERIC_RESPONSE_RESERVED);
  response->ticket_id = get_u64(payload + GENERIC_RESPONSE_TICKET);
  response->trace_id = get_u64(payload + GENERIC_RESPONSE_TRACE);
  response->queue_id = get_u64(payload + GENERIC_RESPONSE_QUEUE);
  response->queue_generation = get_u64(payload + GENERIC_RESPONSE_QUEUE_GEN);
  response->queue_sequence = get_u64(payload + GENERIC_RESPONSE_QUEUE_SEQ);
  response->signal_id = get_u64(payload + GENERIC_RESPONSE_SIGNAL);
  response->signal_generation = get_u64(payload + GENERIC_RESPONSE_SIGNAL_GEN);
  response->signal_value_bits = get_u64(payload + GENERIC_RESPONSE_SIGNAL_VALUE);
  response->packet_va = get_u64(payload + GENERIC_RESPONSE_PACKET_VA);
  response->packet_crc32c = get_u32(payload + GENERIC_RESPONSE_PACKET_CRC);
  response->output_crc32c = get_u32(payload + GENERIC_RESPONSE_OUTPUT_CRC);
  response->sim_tick = get_u64(payload + GENERIC_RESPONSE_SIM_TICK);
  response->admission_tick = get_u64(payload + GENERIC_RESPONSE_ADMISSION_TICK);
  response->start_tick = get_u64(payload + GENERIC_RESPONSE_START_TICK);
  response->end_tick = get_u64(payload + GENERIC_RESPONSE_END_TICK);
  response->retire_tick = get_u64(payload + GENERIC_RESPONSE_RETIRE_TICK);
  memcpy(response->image_sha256, payload + GENERIC_RESPONSE_DIGEST, 32U);
}

sagr_status_t
sagr_protocol_encode_generic_dispatch_response(
    const sagr_instance_info_t *info, uint64_t request_id,
    uint16_t message_type, const sagr_wire_generic_response_t *response,
    uint8_t *frame, size_t frame_capacity, size_t *frame_size)
{
  if (info == NULL || response == NULL || frame == NULL || frame_size == NULL ||
      request_id == 0U || frame_capacity < SAGR_WIRE_GENERIC_FRAME_BYTES ||
      (message_type != SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_ACK &&
       message_type != SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_COMPLETION) ||
      bytes_are_zero(info->daemon_uuid, 16U) || info->connection_id == 0U ||
      info->epoch == 0U) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (!generic_capability_selected(info)) {
    return SAGR_STATUS_CAPABILITY_MISMATCH;
  }
  if (response->major != SAGR_GENERIC_DISPATCH_PROTOCOL_MAJOR ||
      response->minor != SAGR_GENERIC_DISPATCH_PROTOCOL_MINOR ||
      response->flags != 0U || !generic_opcode_valid(response->opcode) ||
      response->status > SAGR_WIRE_STATUS_INTERNAL) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  if (response->status == SAGR_WIRE_STATUS_OK) {
    if (!generic_response_success_valid(response, message_type)) {
      return SAGR_STATUS_INVALID_ARGUMENT;
    }
  } else if (response->error_code == 0U ||
             !generic_response_fields_zero(response)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  memset(frame, 0, SAGR_WIRE_GENERIC_FRAME_BYTES);
  encode_header(frame, message_type, SAGR_WIRE_GENERIC_PAYLOAD_BYTES,
                request_id, info->daemon_uuid, info->connection_id,
                info->epoch);
  generic_response_encode_payload(frame + SAGR_WIRE_HEADER_BYTES, response);
  sagr_protocol_recompute_frame_crc(frame, SAGR_WIRE_GENERIC_FRAME_BYTES);
  *frame_size = SAGR_WIRE_GENERIC_FRAME_BYTES;
  return SAGR_STATUS_SUCCESS;
}

sagr_status_t
sagr_protocol_decode_generic_dispatch_response(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t expected_request_id, uint16_t expected_message_type,
    sagr_wire_generic_response_t *response, int32_t *wire_status,
    const char **reason)
{
  const uint8_t *payload = NULL;
  uint64_t request_id = 0U;
  sagr_status_t status;
  if (reason != NULL) *reason = "malformed generic dispatch response";
  if (response == NULL || wire_status == NULL || reason == NULL ||
      (expected_message_type != SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_ACK &&
       expected_message_type != SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_COMPLETION)) {
    return SAGR_STATUS_INVALID_ARGUMENT;
  }
  status = decode_generic_header(frame, frame_size, info, expected_message_type,
                                 &payload, reason);
  if (status != SAGR_STATUS_SUCCESS) return status;
  request_id = get_u64(frame + 24U);
  if (expected_request_id != 0U && expected_request_id != request_id) {
    *reason = "generic dispatch request ID mismatch";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  generic_response_decode_payload(payload, response);
  response->request_id = request_id;
  response->message_type = expected_message_type;
  *wire_status = response->status <= INT32_MAX ? (int32_t)response->status : -1;
  if (response->major != SAGR_GENERIC_DISPATCH_PROTOCOL_MAJOR ||
      response->minor != SAGR_GENERIC_DISPATCH_PROTOCOL_MINOR ||
      response->flags != 0U || !generic_opcode_valid(response->opcode) ||
      response->status > SAGR_WIRE_STATUS_INTERNAL ||
      !bytes_are_zero(payload + GENERIC_RESPONSE_TAIL,
                      SAGR_WIRE_GENERIC_PAYLOAD_BYTES - GENERIC_RESPONSE_TAIL)) {
    *reason = "invalid generic dispatch response fields";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  if (response->status == SAGR_WIRE_STATUS_OK) {
    if (!generic_response_success_valid(response, expected_message_type)) {
      *reason = "invalid successful generic dispatch response";
      return SAGR_STATUS_PROTOCOL_ERROR;
    }
    *reason = "generic dispatch response decoded";
    return SAGR_STATUS_SUCCESS;
  }
  if (response->error_code == 0U || !generic_response_fields_zero(response)) {
    *reason = "noncanonical failed generic dispatch response";
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  *reason = "generic dispatch stage rejected";
  return sagr_protocol_map_wire_status(response->status);
}

sagr_status_t
sagr_protocol_validate_failed_generic_dispatch_ack(
    const sagr_wire_generic_request_t *request,
    const sagr_wire_generic_response_t *response)
{
  if (request == NULL || response == NULL || response->status == 0U ||
      response->opcode != request->opcode || response->error_code == 0U ||
      !generic_response_fields_zero(response)) {
    return SAGR_STATUS_PROTOCOL_ERROR;
  }
  return SAGR_STATUS_SUCCESS;
}
