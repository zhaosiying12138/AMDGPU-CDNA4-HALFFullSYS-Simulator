/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "transport_internal.h"

static const char k_golden_hello[] =
    "4753494d5250430000010000005000010000000000000080"
    "0123456789abcdef00112233445566778899aabbccddeeff"
    "00000000000000000102030405060708508ae01200000000"
    "000000000000000000010000000100000001020304050607"
    "08090a0b0c0d0e0f01000000000000000000000000000000"
    "000000000000000000000000000000000100000000000000"
    "000000000000000000000000000000000000000000000000"
    "000100000001000000010001000000181021324354657687"
    "98a9bacbdcedfe0f0000000300000008";

static const char k_golden_ack[] =
    "4753494d5250430000010000005000020000000000000070"
    "0123456789abcdef00112233445566778899aabbccddeeff"
    "11223344556677880102030405060708c09c261200000000"
    "000000000000000000010000000000000001020304050607"
    "08090a0b0c0d0e0ff0e0d0c0b0a090807060504030201001"
    "010000000000000000000000000000000000000000000000"
    "000000000000000000010000000200000001000100000018"
    "102132435465768798a9bacbdcedfe0f0000000300000008";

static const uint8_t k_daemon_uuid[16] = {
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
static const uint8_t k_job_uuid[16] = {
    0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87,
    0x98, 0xa9, 0xba, 0xcb, 0xdc, 0xed, 0xfe, 0x0f};
static const uint8_t k_client_nonce[16] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
static const uint8_t k_server_nonce[16] = {
    0xf0, 0xe0, 0xd0, 0xc0, 0xb0, 0xa0, 0x90, 0x80,
    0x70, 0x60, 0x50, 0x40, 0x30, 0x20, 0x10, 0x01};

static int decode_hex(const char *hex, uint8_t *bytes, size_t capacity,
                      size_t *size) {
  size_t index;
  const size_t hex_size = strlen(hex);
  if ((hex_size & 1U) != 0 || hex_size / 2U > capacity) {
    return -1;
  }
  for (index = 0; index < hex_size / 2U; ++index) {
    unsigned int value;
    if (sscanf(hex + index * 2U, "%2x", &value) != 1) {
      return -1;
    }
    bytes[index] = (uint8_t)value;
  }
  *size = hex_size / 2U;
  return 0;
}

static int expect_equal(const char *name, const uint8_t *actual,
                        size_t actual_size, const char *golden_hex) {
  uint8_t golden[SAGR_WIRE_MAX_HANDSHAKE_BYTES];
  size_t golden_size = 0;
  if (decode_hex(golden_hex, golden, sizeof(golden), &golden_size) != 0 ||
      actual_size != golden_size || memcmp(actual, golden, golden_size) != 0) {
    fprintf(stderr, "%s does not match the canonical golden frame\n", name);
    return 1;
  }
  return 0;
}

static void initialize_golden_options(sagr_instance_open_options_t *options) {
  (void)sagr_instance_open_options_init(options, (uint32_t)sizeof(*options));
  memcpy(options->expected_daemon_uuid, k_daemon_uuid, 16);
  memcpy(options->expected_job_uuid, k_job_uuid, 16);
  options->expected_epoch = UINT64_C(0x0102030405060708);
  options->expected_rank = 3;
  options->expected_world_size = 8;
}

static void initialize_success_ack(sagr_wire_ack_fields_t *fields) {
  memset(fields, 0, sizeof(*fields));
  fields->selected_major = 1;
  fields->status = SAGR_WIRE_STATUS_OK;
  memcpy(fields->client_nonce, k_client_nonce, 16);
  memcpy(fields->server_nonce, k_server_nonce, 16);
  fields->selected_capabilities[0] = 1;
  fields->maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  fields->request_id = UINT64_C(0x0123456789abcdef);
  memcpy(fields->daemon_uuid, k_daemon_uuid, 16);
  fields->connection_id = UINT64_C(0x1122334455667788);
  fields->epoch = UINT64_C(0x0102030405060708);
  memcpy(fields->job_uuid, k_job_uuid, 16);
  fields->rank = 3;
  fields->world_size = 8;
  fields->include_topology = 1;
}

static int test_crc32c(void) {
  static const uint8_t check[] = "123456789";
  if (sagr_crc32c(check, sizeof(check) - 1U) != UINT32_C(0xe3069283)) {
    fprintf(stderr, "CRC32C canonical check failed\n");
    return 1;
  }
  return 0;
}

static int test_golden_hello(void) {
  sagr_instance_open_options_t options;
  uint8_t frame[SAGR_WIRE_HELLO_FRAME_BYTES];
  size_t frame_size = 0;
  initialize_golden_options(&options);
  if (sagr_protocol_encode_hello(
          &options, UINT64_C(0x0123456789abcdef), k_client_nonce, frame,
          sizeof(frame), &frame_size) != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "HELLO encoder failed\n");
    return 1;
  }
  return expect_equal("HELLO", frame, frame_size, k_golden_hello);
}

static int test_golden_ack(void) {
  sagr_wire_ack_fields_t fields;
  sagr_instance_open_options_t options;
  sagr_wire_ack_result_t result;
  uint8_t frame[SAGR_WIRE_ACK_FRAME_BYTES];
  size_t frame_size = 0;
  int32_t wire_status = -1;
  const char *reason = NULL;
  initialize_success_ack(&fields);
  if (sagr_protocol_encode_ack(&fields, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "ACK encoder failed\n");
    return 1;
  }
  if (expect_equal("ACK", frame, frame_size, k_golden_ack) != 0) {
    return 1;
  }

  initialize_golden_options(&options);
  if (sagr_protocol_decode_ack(
          frame, frame_size, &options, fields.request_id, k_client_nonce,
          &result, &wire_status, &reason) != SAGR_STATUS_SUCCESS ||
      wire_status != SAGR_WIRE_STATUS_OK ||
      result.connection_id != fields.connection_id || result.rank != 3 ||
      result.world_size != 8 ||
      memcmp(result.daemon_uuid, k_daemon_uuid, 16) != 0 ||
      memcmp(result.job_uuid, k_job_uuid, 16) != 0) {
    fprintf(stderr, "canonical ACK decode failed: %s\n",
            reason == NULL ? "no reason" : reason);
    return 1;
  }
  return 0;
}

static int decode_ack_status(const uint8_t *frame, size_t frame_size,
                             const sagr_instance_open_options_t *options,
                             uint64_t request_id,
                             const uint8_t client_nonce[16],
                             int32_t *wire_status) {
  sagr_wire_ack_result_t result;
  const char *reason = NULL;
  return sagr_protocol_decode_ack(frame, frame_size, options, request_id,
                                  client_nonce, &result, wire_status, &reason);
}

static int test_capability_wire_numbering(void) {
  sagr_instance_open_options_t options;
  uint8_t frame[SAGR_WIRE_HELLO_FRAME_BYTES];
  size_t frame_size = 0;
  (void)sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));
  options.offered_capabilities[0] |= UINT64_C(1) << 8;
  if (sagr_protocol_encode_hello(
          &options, UINT64_C(1), k_client_nonce, frame, sizeof(frame),
          &frame_size) != SAGR_STATUS_SUCCESS ||
      frame[80 + 24] != 1 || frame[80 + 25] != 1) {
    fprintf(stderr, "capability bit numbering is not byte-local\n");
    return 1;
  }

  options.offered_capabilities[0] = 0;
  options.required_capabilities[0] = 0;
  if (sagr_protocol_encode_hello(
          &options, UINT64_C(1), k_client_nonce, frame, sizeof(frame),
          &frame_size) != SAGR_STATUS_SUCCESS ||
      frame_size != SAGR_WIRE_HEADER_BYTES + SAGR_WIRE_HELLO_FIXED_BYTES) {
    fprintf(stderr, "HELLO topology TLV is not conditional on capability 0\n");
    return 1;
  }
  return 0;
}

static int test_failure_ack_mapping_and_shape(void) {
  static const struct {
    uint32_t wire;
    sagr_status_t runtime;
  } cases[] = {
      {SAGR_WIRE_STATUS_MALFORMED, SAGR_STATUS_PROTOCOL_ERROR},
      {SAGR_WIRE_STATUS_UNSUPPORTED_VERSION, SAGR_STATUS_VERSION_MISMATCH},
      {SAGR_WIRE_STATUS_UNSUPPORTED_CAPABILITY,
       SAGR_STATUS_CAPABILITY_MISMATCH},
      {SAGR_WIRE_STATUS_INSTANCE_MISMATCH, SAGR_STATUS_INSTANCE_MISMATCH},
      {SAGR_WIRE_STATUS_TOPOLOGY_MISMATCH, SAGR_STATUS_TOPOLOGY_MISMATCH},
      {SAGR_WIRE_STATUS_UNAUTHORIZED, SAGR_STATUS_UNAUTHORIZED},
      {SAGR_WIRE_STATUS_BUSY, SAGR_STATUS_BUSY},
      {SAGR_WIRE_STATUS_RESOURCE_EXHAUSTED, SAGR_STATUS_OUT_OF_RESOURCES},
      {SAGR_WIRE_STATUS_PROTOCOL_STATE, SAGR_STATUS_PROTOCOL_ERROR},
      {SAGR_WIRE_STATUS_INTERNAL, SAGR_STATUS_INTERNAL_ERROR},
  };
  sagr_instance_open_options_t options;
  size_t index;
  initialize_golden_options(&options);
  for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
    sagr_wire_ack_fields_t fields;
    uint8_t frame[SAGR_WIRE_ACK_FRAME_BYTES];
    size_t frame_size = 0;
    int32_t wire_status = -1;
    initialize_success_ack(&fields);
    fields.selected_major = 0;
    fields.status = cases[index].wire;
    memset(fields.server_nonce, 0, sizeof(fields.server_nonce));
    memset(fields.selected_capabilities, 0,
           sizeof(fields.selected_capabilities));
    fields.connection_id = 0;
    fields.include_topology = 0;
    if (sagr_protocol_encode_ack(&fields, frame, sizeof(frame), &frame_size) !=
            SAGR_STATUS_SUCCESS ||
        decode_ack_status(frame, frame_size, &options, fields.request_id,
                          k_client_nonce, &wire_status) != cases[index].runtime ||
        wire_status != (int32_t)cases[index].wire) {
      fprintf(stderr, "failed ACK mapping failed for wire status %u\n",
              cases[index].wire);
      return 1;
    }
  }
  return 0;
}

static int test_ack_mutations(void) {
  sagr_instance_open_options_t options;
  uint8_t golden[SAGR_WIRE_ACK_FRAME_BYTES];
  size_t golden_size = 0;
  int32_t wire_status;
  uint8_t mutated[SAGR_WIRE_MAX_HANDSHAKE_BYTES];
  initialize_golden_options(&options);
  if (decode_hex(k_golden_ack, golden, sizeof(golden), &golden_size) != 0) {
    return 1;
  }

#define EXPECT_MUTATION(OFFSET, VALUE, EXPECTED)                              \
  do {                                                                        \
    memcpy(mutated, golden, golden_size);                                     \
    mutated[(OFFSET)] = (VALUE);                                              \
    sagr_protocol_recompute_frame_crc(mutated, golden_size);                  \
    wire_status = -1;                                                         \
    if (decode_ack_status(mutated, golden_size, &options,                     \
                          UINT64_C(0x0123456789abcdef), k_client_nonce,        \
                          &wire_status) != (EXPECTED)) {                       \
      fprintf(stderr, "unexpected ACK mutation result at offset %u\n",      \
              (unsigned int)(OFFSET));                                        \
      return 1;                                                               \
    }                                                                         \
  } while (0)

  memcpy(mutated, golden, golden_size);
  mutated[golden_size - 1U] ^= 1;
  if (decode_ack_status(mutated, golden_size, &options,
                        UINT64_C(0x0123456789abcdef), k_client_nonce,
                        &wire_status) != SAGR_STATUS_CHECKSUM_ERROR) {
    fprintf(stderr, "ACK checksum mutation was not classified\n");
    return 1;
  }
  EXPECT_MUTATION(14, 1, SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_MUTATION(19, 1, SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_MUTATION(68, 1, SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_MUTATION(24 + 7, 0xee, SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_MUTATION(80 + 8, 0xff, SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_MUTATION(32, 0x01, SAGR_STATUS_INSTANCE_MISMATCH);
  EXPECT_MUTATION(56 + 7, 0x09, SAGR_STATUS_TOPOLOGY_MISMATCH);
  EXPECT_MUTATION(80 + 40, 0x00, SAGR_STATUS_CAPABILITY_MISMATCH);
  EXPECT_MUTATION(80 + 80 + 8 + 16 + 3, 0x04,
                  SAGR_STATUS_TOPOLOGY_MISMATCH);

  memcpy(mutated, golden, golden_size);
  mutated[80 + 80 + 4] = 0xff;
  mutated[80 + 80 + 5] = 0xff;
  mutated[80 + 80 + 6] = 0xff;
  mutated[80 + 80 + 7] = 0xff;
  sagr_protocol_recompute_frame_crc(mutated, golden_size);
  if (decode_ack_status(mutated, golden_size, &options,
                        UINT64_C(0x0123456789abcdef), k_client_nonce,
                        &wire_status) != SAGR_STATUS_PROTOCOL_ERROR) {
    fprintf(stderr, "overflowing ACK TLV length was accepted\n");
    return 1;
  }
#undef EXPECT_MUTATION
  return 0;
}

static int test_noncanonical_failure_ack(void) {
  sagr_instance_open_options_t options;
  sagr_wire_ack_fields_t fields;
  uint8_t frame[SAGR_WIRE_ACK_FRAME_BYTES];
  size_t frame_size;
  int32_t wire_status;
  initialize_golden_options(&options);
  initialize_success_ack(&fields);
  fields.selected_major = 0;
  fields.status = SAGR_WIRE_STATUS_UNSUPPORTED_VERSION;
  memset(fields.server_nonce, 0, 16);
  memset(fields.selected_capabilities, 0, sizeof(fields.selected_capabilities));
  fields.connection_id = 0;
  fields.include_topology = 0;

#define EXPECT_NONCANONICAL(MUTATION)                                         \
  do {                                                                        \
    sagr_wire_ack_fields_t changed = fields;                                  \
    MUTATION;                                                                 \
    if (sagr_protocol_encode_ack(&changed, frame, sizeof(frame), &frame_size) \
            != SAGR_STATUS_SUCCESS ||                                         \
        decode_ack_status(frame, frame_size, &options, changed.request_id,    \
                          k_client_nonce, &wire_status) !=                     \
            SAGR_STATUS_PROTOCOL_ERROR) {                                     \
      fprintf(stderr, "noncanonical failed ACK was accepted\n");            \
      return 1;                                                               \
    }                                                                         \
  } while (0)

  EXPECT_NONCANONICAL(changed.selected_major = 1);
  EXPECT_NONCANONICAL(changed.connection_id = 1);
  EXPECT_NONCANONICAL(changed.server_nonce[0] = 1);
  EXPECT_NONCANONICAL(changed.selected_capabilities[0] = 1);
  EXPECT_NONCANONICAL(changed.include_topology = 1);
  EXPECT_NONCANONICAL(memset(changed.daemon_uuid, 0, 16));
  EXPECT_NONCANONICAL(changed.epoch = 0);
  EXPECT_NONCANONICAL(changed.status = UINT32_MAX);
#undef EXPECT_NONCANONICAL
  return 0;
}

static int test_invalid_success_topology(void) {
  sagr_instance_open_options_t options;
  sagr_wire_ack_fields_t fields;
  uint8_t frame[SAGR_WIRE_ACK_FRAME_BYTES];
  size_t frame_size;
  int32_t wire_status;
  initialize_golden_options(&options);
  initialize_success_ack(&fields);
  fields.include_topology = 0;
  if (sagr_protocol_encode_ack(&fields, frame, sizeof(frame), &frame_size) !=
          SAGR_STATUS_SUCCESS ||
      decode_ack_status(frame, frame_size, &options, fields.request_id,
                        k_client_nonce, &wire_status) !=
          SAGR_STATUS_PROTOCOL_ERROR) {
    fprintf(stderr, "successful ACK without topology was accepted\n");
    return 1;
  }
  initialize_success_ack(&fields);
  memset(fields.job_uuid, 0, 16);
  if (sagr_protocol_encode_ack(&fields, frame, sizeof(frame), &frame_size) !=
          SAGR_STATUS_SUCCESS ||
      decode_ack_status(frame, frame_size, &options, fields.request_id,
                        k_client_nonce, &wire_status) !=
          SAGR_STATUS_PROTOCOL_ERROR) {
    fprintf(stderr, "successful ACK with invalid topology was accepted\n");
    return 1;
  }
  return 0;
}

int main(void) {
  int failures = 0;
  failures += test_crc32c();
  failures += test_golden_hello();
  failures += test_golden_ack();
  failures += test_capability_wire_numbering();
  failures += test_failure_ack_mapping_and_shape();
  failures += test_ack_mutations();
  failures += test_noncanonical_failure_ack();
  failures += test_invalid_success_topology();
  return failures == 0 ? 0 : 1;
}
