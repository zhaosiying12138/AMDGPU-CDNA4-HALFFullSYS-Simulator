/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "transport_internal.h"

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
                           SAGR_CAPABILITY_KMT_MASK;
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
