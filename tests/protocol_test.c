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

static const char k_golden_queue_request[] =
    "4753494d5250430000010000005000030000000000000040"
    "0123456789abcdf000112233445566778899aabbccddeeff"
    "11223344556677880102030405060708dfd3614c00000000"
    "000000000000000000010000000300001020304050607080"
    "887766554433221101000000000000020000000000000001"
    "000000000000000000000000000000000000000000000000";

static const char k_golden_queue_ack[] =
    "4753494d5250430000010000005000040000000000000040"
    "0123456789abcdf000112233445566778899aabbccddeeff"
    "1122334455667788010203040506070821b4ea0e00000000"
    "000000000000000000010000000000000003000000000000"
    "102030405060708088776655443322110100000000000002"
    "00000000000000000000000000000000123456789abcdef0";

static const char k_golden_queue_completion[] =
    "4753494d5250430000010000005000050000000000000040"
    "0123456789abcdf000112233445566778899aabbccddeeff"
    "11223344556677880102030405060708cf22e2cd00000000"
    "000000000000000000010000000000000003000000000000"
    "102030405060708088776655443322110100000000000002"
    "00000000000000010000000000000000123456789abcdef1";

static const char k_golden_queue_error_completion[] =
    "4753494d5250430000010000005000050000000000000040"
    "0123456789abcdf000112233445566778899aabbccddeeff"
    "11223344556677880102030405060708ee1cda8100000000"
    "0000000000000000000100000000000a0003000000000000"
    "102030405060708088776655443322110100000000000002"
    "00000000000000020000000000000001123456789abcdef1";

static const char k_golden_memory_alloc_request[] =
    "4753494d5250430000010000005000060000000000000040"
    "0123456789abcdf100112233445566778899aabbccddeeff"
    "1122334455667788010203040506070891fdb84e00000000"
    "000000000000000000010000000100000000000000000000"
    "000000000000000000000000000000000000000000010000"
    "000000000001000000000000000000000000000000000000";

static const char k_golden_memory_alloc_ack[] =
    "4753494d5250430000010000005000070000000000000040"
    "0123456789abcdf100112233445566778899aabbccddeeff"
    "11223344556677880102030405060708c6f7e53000000000"
    "000000000000000000010000000000000001000000000000"
    "000000000000000788776655443322110000100300000000"
    "00000000000100000000000000010000123456789abcdef2";

static const char k_golden_memory_h2d_request[] =
    "4753494d5250430000010000005000060000000000000040"
    "0123456789abcdf200112233445566778899aabbccddeeff"
    "112233445566778801020304050607085d4ceda600000000"
    "000000000000000000010000000300000000000000000007"
    "887766554433221100000000000010000000000000000010"
    "0000000048dfe98200000000000000000000000000000000";

static const char k_golden_memory_h2d_ack[] =
    "4753494d5250430000010000005000070000000000000040"
    "0123456789abcdf200112233445566778899aabbccddeeff"
    "112233445566778801020304050607087f6468ee00000000"
    "000000000000000000010000000000000003000000000000"
    "000000000000000788776655443322110000000000001000"
    "00000000000000100000000048dfe982123456789abcdef3";

static const char k_golden_memory_d2h_request[] =
    "4753494d5250430000010000005000060000000000000040"
    "0123456789abcdf300112233445566778899aabbccddeeff"
    "112233445566778801020304050607089cc6b0a700000000"
    "000000000000000000010000000400000000000000000007"
    "887766554433221100000000000010000000000000000010"
    "000000000000000000000000000000000000000000000000";

static const char k_golden_memory_d2h_ack[] =
    "4753494d5250430000010000005000070000000000000040"
    "0123456789abcdf300112233445566778899aabbccddeeff"
    "112233445566778801020304050607089465fa5e00000000"
    "000000000000000000010000000000000004000000000000"
    "000000000000000788776655443322110000000000001000"
    "00000000000000100000000048dfe982123456789abcdef4";

static const char k_golden_memory_free_request[] =
    "4753494d5250430000010000005000060000000000000040"
    "0123456789abcdf400112233445566778899aabbccddeeff"
    "11223344556677880102030405060708a966c7a900000000"
    "000000000000000000010000000200000000000000000007"
    "887766554433221100000000000000000000000000000000"
    "000000000000000000000000000000000000000000000000";

static const char k_golden_memory_free_ack[] =
    "4753494d5250430000010000005000070000000000000040"
    "0123456789abcdf400112233445566778899aabbccddeeff"
    "11223344556677880102030405060708e6b8578e00000000"
    "000000000000000000010000000000000002000000000000"
    "000000000000000788776655443322110000000000000000"
    "00000000000000000000000000000000123456789abcdef5";

static const char k_golden_signal_create_request[] =
    "4753494d5250430000010000005000080000000000000040"
    "0123456789abcdf500112233445566778899aabbccddeeff"
    "11223344556677880102030405060708b095ba9900000000"
    "000000000000000000010000000100000000000000000000"
    "00000000000000000000000000000000fffffffffffffff9"
    "000000000000000000000000000000000000000000000000";

static const char k_golden_signal_create_ack[] =
    "4753494d5250430000010000005000090000000000000040"
    "0123456789abcdf500112233445566778899aabbccddeeff"
    "11223344556677880102030405060708dda8e6fc00000000"
    "000000000000000000010000000000000001000000000000"
    "000000000000000788776655443322110000000000000000"
    "fffffffffffffff90000000000000000123456789abcdef5";

static const char k_golden_signal_load_request[] =
    "4753494d5250430000010000005000080000000000000040"
    "0123456789abcdf600112233445566778899aabbccddeeff"
    "11223344556677880102030405060708601dbdee00000000"
    "000000000000000000010000000300000000000000000007"
    "887766554433221100000000000000000000000000000000"
    "000000000000000000000000000000000000000000000000";

static const char k_golden_signal_load_ack[] =
    "4753494d5250430000010000005000090000000000000040"
    "0123456789abcdf600112233445566778899aabbccddeeff"
    "112233445566778801020304050607082fa9d5b600000000"
    "000000000000000000010000000000000003000000000000"
    "000000000000000788776655443322110000000000000000"
    "fffffffffffffff90000000000000000123456789abcdef6";

static const char k_golden_signal_wait_request[] =
    "4753494d5250430000010000005000080000000000000040"
    "0123456789abcdf700112233445566778899aabbccddeeff"
    "1122334455667788010203040506070860ce276f00000000"
    "000000000000000000010000000500000000000000000007"
    "887766554433221101000000000000020000000000000000"
    "000000000000000300000000000000000000000000000000";

static const char k_golden_signal_wait_ack[] =
    "4753494d5250430000010000005000090000000000000040"
    "0123456789abcdf700112233445566778899aabbccddeeff"
    "11223344556677880102030405060708593564c300000000"
    "000000000000000000010000000000000005000000000000"
    "000000000000000788776655443322110100000000000002"
    "fffffffffffffff90000000000000000123456789abcdef7";

static const char k_golden_signal_store_request[] =
    "4753494d5250430000010000005000080000000000000040"
    "0123456789abcdf800112233445566778899aabbccddeeff"
    "11223344556677880102030405060708302bee2300000000"
    "000000000000000000010000000400000000000000000007"
    "88776655443322110000000000000000000000000000002a"
    "000000000000000000000000000000000000000000000000";

static const char k_golden_signal_store_ack[] =
    "4753494d5250430000010000005000090000000000000040"
    "0123456789abcdf800112233445566778899aabbccddeeff"
    "11223344556677880102030405060708c6b5892c00000000"
    "000000000000000000010000000000000004000000000000"
    "000000000000000788776655443322110000000000000000"
    "000000000000002a0000000000000000123456789abcdef8";

static const char k_golden_signal_wait_completion[] =
    "4753494d52504300000100000050000a0000000000000040"
    "0123456789abcdf700112233445566778899aabbccddeeff"
    "112233445566778801020304050607085b15367900000000"
    "000000000000000000010000000000000005000000000000"
    "000000000000000788776655443322110100000000000002"
    "000000000000002a0000000000000000123456789abcdef9";

static const char k_golden_signal_destroy_request[] =
    "4753494d5250430000010000005000080000000000000040"
    "0123456789abcdf900112233445566778899aabbccddeeff"
    "112233445566778801020304050607084580308000000000"
    "000000000000000000010000000200000000000000000007"
    "887766554433221100000000000000000000000000000000"
    "000000000000000000000000000000000000000000000000";

static const char k_golden_signal_destroy_ack[] =
    "4753494d5250430000010000005000090000000000000040"
    "0123456789abcdf900112233445566778899aabbccddeeff"
    "11223344556677880102030405060708544d9c8300000000"
    "000000000000000000010000000000000002000000000000"
    "000000000000000788776655443322110000000000000000"
    "00000000000000000000000000000000123456789abcdefa";

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

static void initialize_queue_info(sagr_instance_info_t *info) {
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  info->maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  info->negotiated_capabilities[0] =
      SAGR_CAPABILITY_TOPOLOGY_MASK | SAGR_CAPABILITY_QUEUE_MASK;
  memcpy(info->daemon_uuid, k_daemon_uuid, 16);
  info->connection_id = UINT64_C(0x1122334455667788);
  info->epoch = UINT64_C(0x0102030405060708);
}

static void initialize_signal_info(sagr_instance_info_t *info) {
  initialize_queue_info(info);
  info->negotiated_capabilities[0] =
      SAGR_CAPABILITY_TOPOLOGY_MASK | SAGR_CAPABILITY_SIGNAL_MASK;
}

static void initialize_queue_request(sagr_wire_queue_request_t *request) {
  memset(request, 0, sizeof(*request));
  request->major = SAGR_QUEUE_PROTOCOL_MAJOR;
  request->minor = SAGR_QUEUE_PROTOCOL_MINOR;
  request->opcode = SAGR_WIRE_QUEUE_OPCODE_DOORBELL;
  request->queue_id = UINT64_C(0x1020304050607080);
  request->generation = UINT64_C(0x8877665544332211);
  request->sequence = UINT64_C(0x0100000000000002);
  request->arg0 = SAGR_QUEUE_COMMAND_CONTROL_TEST;
}

static void initialize_queue_response(sagr_wire_queue_response_t *response,
                                      uint64_t sim_tick) {
  memset(response, 0, sizeof(*response));
  response->major = SAGR_QUEUE_PROTOCOL_MAJOR;
  response->minor = SAGR_QUEUE_PROTOCOL_MINOR;
  response->status = SAGR_WIRE_STATUS_OK;
  response->opcode = SAGR_WIRE_QUEUE_OPCODE_DOORBELL;
  response->queue_id = UINT64_C(0x1020304050607080);
  response->generation = UINT64_C(0x8877665544332211);
  response->sequence = UINT64_C(0x0100000000000002);
  response->value = SAGR_QUEUE_COMMAND_CONTROL_TEST;
  response->sim_tick = sim_tick;
}

static void store_u16(uint8_t *destination, uint16_t value) {
  destination[0] = (uint8_t)(value >> 8);
  destination[1] = (uint8_t)value;
}

static void store_u32(uint8_t *destination, uint32_t value) {
  destination[0] = (uint8_t)(value >> 24);
  destination[1] = (uint8_t)(value >> 16);
  destination[2] = (uint8_t)(value >> 8);
  destination[3] = (uint8_t)value;
}

static void store_u64(uint8_t *destination, uint64_t value) {
  store_u32(destination, (uint32_t)(value >> 32));
  store_u32(destination + 4, (uint32_t)value);
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

static int test_queue_golden_frames(void) {
  sagr_instance_info_t info;
  sagr_wire_queue_request_t request;
  sagr_wire_queue_response_t response;
  sagr_wire_queue_response_t decoded;
  uint8_t frame[SAGR_WIRE_QUEUE_FRAME_BYTES];
  size_t frame_size = 0;
  int32_t wire_status = -1;
  const char *reason = NULL;
  const uint64_t request_id = UINT64_C(0x0123456789abcdf0);
  const uint64_t ack_tick = UINT64_C(0x123456789abcdef0);

  initialize_queue_info(&info);
  initialize_queue_request(&request);
  if (sagr_protocol_encode_queue_request(
          &info, request_id, &request, frame, sizeof(frame), &frame_size) !=
          SAGR_STATUS_SUCCESS ||
      expect_equal("QUEUE_REQUEST", frame, frame_size,
                   k_golden_queue_request) != 0) {
      fprintf(stderr, "queue request golden encode failed\n");
    return 1;
  }
  request.arg0 = SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST;
  if (sagr_protocol_encode_queue_request(
          &info, request_id, &request, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "CONTROL_ERROR_TEST request encode failed\n");
    return 1;
  }
  request.arg0 = UINT64_C(3);
  if (sagr_protocol_encode_queue_request(
          &info, request_id, &request, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_INVALID_ARGUMENT) {
    fprintf(stderr, "unknown queue command kind was encoded\n");
    return 1;
  }

  initialize_queue_response(&response, ack_tick);
  response.value = 0;
  if (sagr_protocol_encode_queue_response(
          &info, request_id, SAGR_WIRE_MESSAGE_QUEUE_ACK, &response, frame,
          sizeof(frame), &frame_size) != SAGR_STATUS_SUCCESS ||
      expect_equal("QUEUE_ACK", frame, frame_size, k_golden_queue_ack) != 0 ||
      sagr_protocol_decode_queue_response(
          frame, frame_size, &info, request_id,
          SAGR_WIRE_MESSAGE_QUEUE_ACK, &decoded, &wire_status, &reason) !=
          SAGR_STATUS_SUCCESS ||
      wire_status != SAGR_WIRE_STATUS_OK ||
      decoded.request_id != request_id ||
      decoded.queue_id != response.queue_id ||
      decoded.generation != response.generation ||
      decoded.sequence != response.sequence ||
      decoded.value != response.value || decoded.sim_tick != response.sim_tick) {
    fprintf(stderr, "queue ACK golden encode/decode failed: %s\n",
            reason == NULL ? "no reason" : reason);
    return 1;
  }

  initialize_queue_response(&response, ack_tick + UINT64_C(1));
  if (sagr_protocol_encode_queue_response(
          &info, request_id, SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &response,
          frame, sizeof(frame), &frame_size) != SAGR_STATUS_SUCCESS ||
      expect_equal("QUEUE_COMPLETION", frame, frame_size,
                   k_golden_queue_completion) != 0 ||
      sagr_protocol_decode_queue_response(
          frame, frame_size, &info, request_id,
          SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &decoded, &wire_status,
          &reason) != SAGR_STATUS_SUCCESS ||
      decoded.message_type != SAGR_WIRE_MESSAGE_QUEUE_COMPLETION ||
      decoded.request_id != request_id || decoded.sim_tick != response.sim_tick ||
      decoded.sim_tick != ack_tick + UINT64_C(1)) {
    fprintf(stderr, "queue completion golden encode/decode failed: %s\n",
            reason == NULL ? "no reason" : reason);
    return 1;
  }

  response.status = SAGR_WIRE_STATUS_INTERNAL;
  response.value = SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST;
  response.error_code = UINT64_C(1);
  if (sagr_protocol_encode_queue_response(
          &info, request_id, SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &response,
          frame, sizeof(frame), &frame_size) != SAGR_STATUS_SUCCESS ||
      expect_equal("QUEUE_ERROR_COMPLETION", frame, frame_size,
                   k_golden_queue_error_completion) != 0 ||
      sagr_protocol_decode_queue_response(
          frame, frame_size, &info, request_id,
          SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &decoded, &wire_status,
          &reason) != SAGR_STATUS_INTERNAL_ERROR ||
      wire_status != SAGR_WIRE_STATUS_INTERNAL ||
      decoded.value != SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST ||
      decoded.error_code != UINT64_C(1) ||
      decoded.sim_tick != ack_tick + UINT64_C(1)) {
    fprintf(stderr, "queue error completion golden failed: %s\n",
            reason == NULL ? "no reason" : reason);
    return 1;
  }
  return 0;
}

static int expect_queue_decode_status(const uint8_t *frame, size_t frame_size,
                                      const sagr_instance_info_t *info,
                                      uint64_t request_id,
                                      uint16_t message_type,
                                      sagr_status_t expected) {
  sagr_wire_queue_response_t decoded;
  int32_t wire_status = -1;
  const char *reason = NULL;
  const sagr_status_t actual = sagr_protocol_decode_queue_response(
      frame, frame_size, info, request_id, message_type, &decoded,
      &wire_status, &reason);
  if (actual != expected) {
    fprintf(stderr, "queue mutation status=%d expected=%d: %s\n", actual,
            expected, reason == NULL ? "no reason" : reason);
    return 1;
  }
  return 0;
}

static int test_queue_response_mutations(void) {
  sagr_instance_info_t info;
  uint8_t golden[SAGR_WIRE_QUEUE_FRAME_BYTES];
  uint8_t mutated[SAGR_WIRE_QUEUE_FRAME_BYTES];
  size_t golden_size = 0;
  const uint64_t request_id = UINT64_C(0x0123456789abcdf0);
  initialize_queue_info(&info);
  if (decode_hex(k_golden_queue_ack, golden, sizeof(golden), &golden_size) !=
      0) {
    return 1;
  }

#define EXPECT_QUEUE_MUTATION(MUTATION, EXPECTED)                            \
  do {                                                                        \
    memcpy(mutated, golden, golden_size);                                     \
    MUTATION;                                                                 \
    sagr_protocol_recompute_frame_crc(mutated, golden_size);                  \
    if (expect_queue_decode_status(mutated, golden_size, &info, request_id,   \
                                   SAGR_WIRE_MESSAGE_QUEUE_ACK, EXPECTED) !=  \
        0) {                                                                  \
      return 1;                                                               \
    }                                                                         \
  } while (0)

  EXPECT_QUEUE_MUTATION(store_u16(mutated + 14,
                                  SAGR_WIRE_MESSAGE_QUEUE_COMPLETION),
                        SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_QUEUE_MUTATION(store_u64(mutated + 24, request_id + 1),
                        SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_QUEUE_MUTATION(mutated[32] ^= 1, SAGR_STATUS_INSTANCE_MISMATCH);
  EXPECT_QUEUE_MUTATION(store_u64(mutated + 48, info.connection_id + 1),
                        SAGR_STATUS_TOPOLOGY_MISMATCH);
  EXPECT_QUEUE_MUTATION(store_u64(mutated + 56, info.epoch + 1),
                        SAGR_STATUS_TOPOLOGY_MISMATCH);
  EXPECT_QUEUE_MUTATION(store_u16(mutated + SAGR_WIRE_HEADER_BYTES, 2),
                        SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_QUEUE_MUTATION(store_u32(mutated + SAGR_WIRE_HEADER_BYTES + 4,
                                  UINT32_MAX),
                        SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_QUEUE_MUTATION(store_u16(mutated + SAGR_WIRE_HEADER_BYTES + 8, 99),
                        SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_QUEUE_MUTATION(store_u16(mutated + SAGR_WIRE_HEADER_BYTES + 10, 1),
                        SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_QUEUE_MUTATION(store_u32(mutated + SAGR_WIRE_HEADER_BYTES + 12, 1),
                        SAGR_STATUS_PROTOCOL_ERROR);
#undef EXPECT_QUEUE_MUTATION

  memcpy(mutated, golden, golden_size);
  mutated[golden_size - 1U] ^= 1;
  if (expect_queue_decode_status(mutated, golden_size, &info, request_id,
                                 SAGR_WIRE_MESSAGE_QUEUE_ACK,
                                 SAGR_STATUS_CHECKSUM_ERROR) != 0 ||
      expect_queue_decode_status(golden, golden_size - 1U, &info, request_id,
                                 SAGR_WIRE_MESSAGE_QUEUE_ACK,
                                 SAGR_STATUS_PROTOCOL_ERROR) != 0) {
    return 1;
  }
  return 0;
}

static int test_queue_capability_must_be_required(void) {
  sagr_wire_ack_fields_t fields;
  sagr_instance_open_options_t options;
  sagr_wire_ack_result_t result;
  uint8_t frame[SAGR_WIRE_ACK_FRAME_BYTES];
  size_t frame_size = 0;
  int32_t wire_status = -1;
  const char *reason = NULL;
  initialize_success_ack(&fields);
  fields.selected_capabilities[0] |= SAGR_CAPABILITY_QUEUE_MASK;
  initialize_golden_options(&options);
  options.offered_capabilities[0] |= SAGR_CAPABILITY_QUEUE_MASK;
  if (sagr_protocol_encode_ack(&fields, frame, sizeof(frame), &frame_size) !=
          SAGR_STATUS_SUCCESS ||
      sagr_protocol_decode_ack(frame, frame_size, &options, fields.request_id,
                               k_client_nonce, &result, &wire_status,
                               &reason) != SAGR_STATUS_CAPABILITY_MISMATCH) {
    fprintf(stderr, "offered-only queue capability ACK was accepted\n");
    return 1;
  }
  options.required_capabilities[0] |= SAGR_CAPABILITY_QUEUE_MASK;
  if (sagr_protocol_decode_ack(frame, frame_size, &options, fields.request_id,
                               k_client_nonce, &result, &wire_status,
                               &reason) != SAGR_STATUS_SUCCESS ||
      (result.selected_capabilities[0] & SAGR_CAPABILITY_QUEUE_MASK) == 0) {
    fprintf(stderr, "required queue capability ACK was rejected: %s\n",
            reason == NULL ? "no reason" : reason);
    return 1;
  }
  return 0;
}

static int test_failed_queue_ack_shape(void) {
  sagr_wire_queue_request_t request;
  sagr_wire_queue_response_t response;
  initialize_queue_request(&request);
  initialize_queue_response(&response, 0);
  response.status = SAGR_WIRE_STATUS_PROTOCOL_STATE;
  response.value = 0;

  if (sagr_protocol_validate_failed_queue_ack(&request, &response) !=
      SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "canonical failed queue ACK was rejected\n");
    return 1;
  }

#define EXPECT_FAILED_ACK_REJECTED(MUTATION)                                 \
  do {                                                                        \
    sagr_wire_queue_response_t changed = response;                            \
    MUTATION;                                                                 \
    if (sagr_protocol_validate_failed_queue_ack(&request, &changed) !=        \
        SAGR_STATUS_PROTOCOL_ERROR) {                                         \
      fprintf(stderr, "noncanonical failed queue ACK was accepted\n");      \
      return 1;                                                               \
    }                                                                         \
  } while (0)

  EXPECT_FAILED_ACK_REJECTED(changed.opcode = SAGR_WIRE_QUEUE_OPCODE_CREATE);
  EXPECT_FAILED_ACK_REJECTED(changed.queue_id ^= UINT64_C(1));
  EXPECT_FAILED_ACK_REJECTED(changed.generation ^= UINT64_C(1));
  EXPECT_FAILED_ACK_REJECTED(changed.sequence ^= UINT64_C(1));
  EXPECT_FAILED_ACK_REJECTED(changed.value = UINT64_C(1));
  EXPECT_FAILED_ACK_REJECTED(changed.error_code = UINT64_C(1));
  EXPECT_FAILED_ACK_REJECTED(changed.status = SAGR_WIRE_STATUS_OK);
#undef EXPECT_FAILED_ACK_REJECTED
  return 0;
}

static int expect_memory_case(
    const char *name, const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_memory_request_t *request,
    const sagr_wire_memory_response_t *response, const char *request_golden,
    const char *ack_golden) {
  uint8_t frame[SAGR_WIRE_MEMORY_FRAME_BYTES];
  size_t frame_size = 0;
  sagr_wire_memory_response_t decoded;
  int32_t wire_status = -1;
  const char *reason = NULL;
  if (sagr_protocol_encode_memory_request(
          info, request_id, request, frame, sizeof(frame), &frame_size) !=
          SAGR_STATUS_SUCCESS ||
      expect_equal(name, frame, frame_size, request_golden) != 0) {
    fprintf(stderr, "%s memory request golden encode failed\n", name);
    return 1;
  }
  if (sagr_protocol_encode_memory_response(
          info, request_id, response, frame, sizeof(frame), &frame_size) !=
          SAGR_STATUS_SUCCESS ||
      expect_equal(name, frame, frame_size, ack_golden) != 0 ||
      sagr_protocol_decode_memory_response(
          frame, frame_size, info, request_id, &decoded, &wire_status,
          &reason) != SAGR_STATUS_SUCCESS ||
      wire_status != SAGR_WIRE_STATUS_OK ||
      decoded.request_id != request_id || decoded.opcode != response->opcode ||
      decoded.allocation_id != response->allocation_id ||
      decoded.generation != response->generation ||
      decoded.value0 != response->value0 || decoded.value1 != response->value1 ||
      decoded.value2 != response->value2 ||
      decoded.sim_tick != response->sim_tick) {
    fprintf(stderr, "%s memory ACK golden encode/decode failed: %s\n", name,
            reason == NULL ? "no reason" : reason);
    return 1;
  }
  return 0;
}

static int test_memory_golden_frames(void) {
  sagr_instance_info_t info;
  sagr_wire_memory_request_t request;
  sagr_wire_memory_response_t response;

  initialize_queue_info(&info);
  info.negotiated_capabilities[0] =
      SAGR_CAPABILITY_TOPOLOGY_MASK | SAGR_CAPABILITY_MEMORY_MASK;
  memset(&request, 0, sizeof(request));
  request.major = SAGR_MEMORY_PROTOCOL_MAJOR;
  request.minor = SAGR_MEMORY_PROTOCOL_MINOR;
  request.opcode = SAGR_WIRE_MEMORY_OPCODE_ALLOC;
  request.byte_count = UINT64_C(65536);
  request.argument = SAGR_MEMORY_ALIGNMENT_64K;
  memset(&response, 0, sizeof(response));
  response.major = SAGR_MEMORY_PROTOCOL_MAJOR;
  response.minor = SAGR_MEMORY_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = request.opcode;
  response.allocation_id = UINT64_C(7);
  response.generation = UINT64_C(0x8877665544332211);
  response.value0 = UINT64_C(0x0000100300000000);
  response.value1 = request.byte_count;
  response.value2 = request.argument;
  response.sim_tick = UINT64_C(0x123456789abcdef2);
  if (expect_memory_case(
          "MEMORY_ALLOC", &info, UINT64_C(0x0123456789abcdf1), &request,
          &response, k_golden_memory_alloc_request,
          k_golden_memory_alloc_ack) != 0) {
    return 1;
  }

  request.opcode = SAGR_WIRE_MEMORY_OPCODE_COPY_H2D;
  request.allocation_id = response.allocation_id;
  request.generation = response.generation;
  request.offset = UINT64_C(4096);
  request.byte_count = UINT64_C(16);
  request.argument = UINT64_C(0x48dfe982);
  response.opcode = request.opcode;
  response.value0 = request.offset;
  response.value1 = request.byte_count;
  response.value2 = request.argument;
  response.sim_tick = UINT64_C(0x123456789abcdef3);
  if (expect_memory_case(
          "MEMORY_H2D", &info, UINT64_C(0x0123456789abcdf2), &request,
          &response, k_golden_memory_h2d_request,
          k_golden_memory_h2d_ack) != 0) {
    return 1;
  }

  request.opcode = SAGR_WIRE_MEMORY_OPCODE_COPY_D2H;
  request.argument = 0;
  response.opcode = request.opcode;
  response.sim_tick = UINT64_C(0x123456789abcdef4);
  if (expect_memory_case(
          "MEMORY_D2H", &info, UINT64_C(0x0123456789abcdf3), &request,
          &response, k_golden_memory_d2h_request,
          k_golden_memory_d2h_ack) != 0) {
    return 1;
  }

  request.opcode = SAGR_WIRE_MEMORY_OPCODE_FREE;
  request.offset = 0;
  request.byte_count = 0;
  response.opcode = request.opcode;
  response.value0 = 0;
  response.value1 = 0;
  response.value2 = 0;
  response.sim_tick = UINT64_C(0x123456789abcdef5);
  if (expect_memory_case(
          "MEMORY_FREE", &info, UINT64_C(0x0123456789abcdf4), &request,
          &response, k_golden_memory_free_request,
          k_golden_memory_free_ack) != 0) {
    return 1;
  }
  return 0;
}

static int test_memory_request_structural_limits(void) {
  sagr_instance_info_t info;
  sagr_wire_memory_request_t request;
  uint8_t frame[SAGR_WIRE_MEMORY_FRAME_BYTES];
  size_t frame_size = 0;
  initialize_queue_info(&info);
  memset(&request, 0, sizeof(request));
  request.major = SAGR_MEMORY_PROTOCOL_MAJOR;
  request.minor = SAGR_MEMORY_PROTOCOL_MINOR;
  request.opcode = SAGR_WIRE_MEMORY_OPCODE_ALLOC;
  request.byte_count = SAGR_MEMORY_MAX_SINGLE_ALLOCATION_BYTES + UINT64_C(1);
  request.argument = SAGR_MEMORY_ALIGNMENT_4K;
  if (sagr_protocol_encode_memory_request(
          &info, UINT64_C(1), &request, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "resource-exhausting ALLOC was rejected structurally\n");
    return 1;
  }
  request.opcode = SAGR_WIRE_MEMORY_OPCODE_COPY_D2H;
  request.allocation_id = UINT64_C(1);
  request.generation = UINT64_C(1);
  request.offset = UINT64_C(4);
  request.byte_count = SAGR_MEMORY_MAX_TRANSFER_BYTES + UINT64_C(1);
  request.argument = 0;
  if (sagr_protocol_encode_memory_request(
          &info, UINT64_C(2), &request, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "resource-exhausting COPY was rejected structurally\n");
    return 1;
  }
  request.offset = UINT64_MAX;
  request.byte_count = UINT64_C(1);
  if (sagr_protocol_encode_memory_request(
          &info, UINT64_C(3), &request, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_INVALID_ARGUMENT) {
    fprintf(stderr, "overflowing COPY range was encoded\n");
    return 1;
  }
  return 0;
}

static int expect_memory_decode_status(const uint8_t *frame,
                                       size_t frame_size,
                                       const sagr_instance_info_t *info,
                                       uint64_t request_id,
                                       sagr_status_t expected) {
  sagr_wire_memory_response_t decoded;
  int32_t wire_status = -1;
  const char *reason = NULL;
  const sagr_status_t actual = sagr_protocol_decode_memory_response(
      frame, frame_size, info, request_id, &decoded, &wire_status, &reason);
  if (actual != expected) {
    fprintf(stderr, "memory mutation status=%d expected=%d: %s\n", actual,
            expected, reason == NULL ? "no reason" : reason);
    return 1;
  }
  return 0;
}

static int test_memory_response_mutations(void) {
  sagr_instance_info_t info;
  uint8_t golden[SAGR_WIRE_MEMORY_FRAME_BYTES];
  uint8_t mutated[SAGR_WIRE_MEMORY_FRAME_BYTES];
  size_t golden_size = 0;
  const uint64_t request_id = UINT64_C(0x0123456789abcdf1);
  initialize_queue_info(&info);
  if (decode_hex(k_golden_memory_alloc_ack, golden, sizeof(golden),
                 &golden_size) != 0) {
    return 1;
  }

#define EXPECT_MEMORY_MUTATION(MUTATION, EXPECTED)                             \
  do {                                                                          \
    memcpy(mutated, golden, golden_size);                                       \
    MUTATION;                                                                   \
    sagr_protocol_recompute_frame_crc(mutated, golden_size);                    \
    if (expect_memory_decode_status(mutated, golden_size, &info, request_id,    \
                                    EXPECTED) != 0) {                           \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

  EXPECT_MEMORY_MUTATION(store_u16(mutated + 14,
                                   SAGR_WIRE_MESSAGE_MEMORY_REQUEST),
                         SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_MEMORY_MUTATION(store_u64(mutated + 24, request_id + UINT64_C(1)),
                         SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_MEMORY_MUTATION(mutated[32] ^= 1, SAGR_STATUS_INSTANCE_MISMATCH);
  EXPECT_MEMORY_MUTATION(store_u64(mutated + 48, info.connection_id + 1),
                         SAGR_STATUS_TOPOLOGY_MISMATCH);
  EXPECT_MEMORY_MUTATION(store_u64(mutated + 56, info.epoch + 1),
                         SAGR_STATUS_TOPOLOGY_MISMATCH);
  EXPECT_MEMORY_MUTATION(store_u16(mutated + SAGR_WIRE_HEADER_BYTES, 2),
                         SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_MEMORY_MUTATION(store_u32(mutated + SAGR_WIRE_HEADER_BYTES + 4,
                                   UINT32_MAX),
                         SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_MEMORY_MUTATION(store_u16(mutated + SAGR_WIRE_HEADER_BYTES + 8, 99),
                         SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_MEMORY_MUTATION(store_u16(mutated + SAGR_WIRE_HEADER_BYTES + 10, 1),
                         SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_MEMORY_MUTATION(store_u32(mutated + SAGR_WIRE_HEADER_BYTES + 12, 1),
                         SAGR_STATUS_PROTOCOL_ERROR);
#undef EXPECT_MEMORY_MUTATION

  memcpy(mutated, golden, golden_size);
  mutated[golden_size - 1U] ^= 1;
  if (expect_memory_decode_status(mutated, golden_size, &info, request_id,
                                  SAGR_STATUS_CHECKSUM_ERROR) != 0 ||
      expect_memory_decode_status(golden, golden_size - 1U, &info, request_id,
                                  SAGR_STATUS_PROTOCOL_ERROR) != 0) {
    return 1;
  }
  return 0;
}

static int test_memory_capability_must_be_required(void) {
  sagr_wire_ack_fields_t fields;
  sagr_instance_open_options_t options;
  sagr_wire_ack_result_t result;
  uint8_t frame[SAGR_WIRE_ACK_FRAME_BYTES];
  size_t frame_size = 0;
  int32_t wire_status = -1;
  const char *reason = NULL;
  initialize_success_ack(&fields);
  fields.selected_capabilities[0] |= SAGR_CAPABILITY_MEMORY_MASK;
  initialize_golden_options(&options);
  options.offered_capabilities[0] |= SAGR_CAPABILITY_MEMORY_MASK;
  if (sagr_protocol_encode_ack(&fields, frame, sizeof(frame), &frame_size) !=
          SAGR_STATUS_SUCCESS ||
      sagr_protocol_decode_ack(frame, frame_size, &options, fields.request_id,
                               k_client_nonce, &result, &wire_status,
                               &reason) != SAGR_STATUS_CAPABILITY_MISMATCH) {
    fprintf(stderr, "offered-only memory capability ACK was accepted\n");
    return 1;
  }
  options.required_capabilities[0] |= SAGR_CAPABILITY_MEMORY_MASK;
  if (sagr_protocol_decode_ack(frame, frame_size, &options, fields.request_id,
                               k_client_nonce, &result, &wire_status,
                               &reason) != SAGR_STATUS_SUCCESS ||
      (result.selected_capabilities[0] & SAGR_CAPABILITY_MEMORY_MASK) == 0) {
    fprintf(stderr, "required memory capability ACK was rejected: %s\n",
            reason == NULL ? "no reason" : reason);
    return 1;
  }
  return 0;
}

static int test_failed_memory_ack_shape(void) {
  sagr_wire_memory_request_t request;
  sagr_wire_memory_response_t response;
  memset(&request, 0, sizeof(request));
  request.opcode = SAGR_WIRE_MEMORY_OPCODE_COPY_H2D;
  request.allocation_id = UINT64_C(7);
  request.generation = UINT64_C(0x8877665544332211);
  memset(&response, 0, sizeof(response));
  response.status = SAGR_WIRE_STATUS_PROTOCOL_STATE;
  response.opcode = request.opcode;
  response.allocation_id = request.allocation_id;
  response.generation = request.generation;
  if (sagr_protocol_validate_failed_memory_ack(&request, &response) !=
      SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "canonical failed memory ACK was rejected\n");
    return 1;
  }

#define EXPECT_FAILED_MEMORY_ACK_REJECTED(MUTATION)                            \
  do {                                                                          \
    sagr_wire_memory_response_t changed = response;                             \
    MUTATION;                                                                   \
    if (sagr_protocol_validate_failed_memory_ack(&request, &changed) !=         \
        SAGR_STATUS_PROTOCOL_ERROR) {                                           \
      fprintf(stderr, "noncanonical failed memory ACK was accepted\n");     \
      return 1;                                                                 \
    }                                                                           \
  } while (0)
  EXPECT_FAILED_MEMORY_ACK_REJECTED(
      changed.opcode = SAGR_WIRE_MEMORY_OPCODE_FREE);
  EXPECT_FAILED_MEMORY_ACK_REJECTED(changed.allocation_id ^= UINT64_C(1));
  EXPECT_FAILED_MEMORY_ACK_REJECTED(changed.generation ^= UINT64_C(1));
  EXPECT_FAILED_MEMORY_ACK_REJECTED(changed.value0 = UINT64_C(1));
  EXPECT_FAILED_MEMORY_ACK_REJECTED(changed.value1 = UINT64_C(1));
  EXPECT_FAILED_MEMORY_ACK_REJECTED(changed.value2 = UINT64_C(1));
  EXPECT_FAILED_MEMORY_ACK_REJECTED(changed.sim_tick = UINT64_C(1));
  EXPECT_FAILED_MEMORY_ACK_REJECTED(changed.status = SAGR_WIRE_STATUS_OK);
#undef EXPECT_FAILED_MEMORY_ACK_REJECTED
  return 0;
}

static int expect_signal_case(
    const char *name, const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_signal_request_t *request,
    const sagr_wire_signal_response_t *response, uint16_t response_type,
    const char *request_golden, const char *response_golden) {
  uint8_t frame[SAGR_WIRE_SIGNAL_FRAME_BYTES];
  size_t frame_size = 0;
  sagr_wire_signal_response_t decoded;
  int32_t wire_status = -1;
  const char *reason = NULL;
  if (request_golden != NULL &&
      (sagr_protocol_encode_signal_request(
           info, request_id, request, frame, sizeof(frame), &frame_size) !=
           SAGR_STATUS_SUCCESS ||
       expect_equal(name, frame, frame_size, request_golden) != 0)) {
    fprintf(stderr, "%s signal request golden encode failed\n", name);
    return 1;
  }
  if (sagr_protocol_encode_signal_response(
          info, request_id, response_type, response, frame, sizeof(frame),
          &frame_size) != SAGR_STATUS_SUCCESS ||
      expect_equal(name, frame, frame_size, response_golden) != 0 ||
      sagr_protocol_decode_signal_response(
          frame, frame_size, info, request_id, response_type, &decoded,
          &wire_status, &reason) != SAGR_STATUS_SUCCESS ||
      wire_status != SAGR_WIRE_STATUS_OK ||
      decoded.opcode != response->opcode ||
      decoded.signal_id != response->signal_id ||
      decoded.generation != response->generation ||
      decoded.sequence != response->sequence ||
      decoded.value_bits != response->value_bits ||
      decoded.ready != response->ready ||
      decoded.sim_tick != response->sim_tick ||
      decoded.request_id != request_id || decoded.message_type != response_type) {
    fprintf(stderr, "%s signal response golden encode/decode failed: %s\n",
            name, reason == NULL ? "no reason" : reason);
    return 1;
  }
  return 0;
}

static void initialize_signal_request(sagr_wire_signal_request_t *request,
                                      uint16_t opcode) {
  memset(request, 0, sizeof(*request));
  request->major = SAGR_SIGNAL_PROTOCOL_MAJOR;
  request->minor = SAGR_SIGNAL_PROTOCOL_MINOR;
  request->opcode = opcode;
  if (opcode != SAGR_WIRE_SIGNAL_OPCODE_CREATE) {
    request->signal_id = UINT64_C(7);
    request->generation = UINT64_C(0x8877665544332211);
  }
}

static void initialize_signal_response(sagr_wire_signal_response_t *response,
                                       const sagr_wire_signal_request_t *request,
                                       uint64_t value_bits,
                                       uint64_t sim_tick) {
  memset(response, 0, sizeof(*response));
  response->major = SAGR_SIGNAL_PROTOCOL_MAJOR;
  response->minor = SAGR_SIGNAL_PROTOCOL_MINOR;
  response->status = SAGR_WIRE_STATUS_OK;
  response->opcode = request->opcode;
  response->signal_id = request->signal_id;
  response->generation = request->generation;
  response->sequence = request->sequence;
  response->value_bits = value_bits;
  response->sim_tick = sim_tick;
}

static int test_signal_golden_frames(void) {
  sagr_instance_info_t info;
  sagr_wire_signal_request_t request;
  sagr_wire_signal_response_t response;
  const uint64_t initial = UINT64_C(0xfffffffffffffff9);
  const uint64_t generation = UINT64_C(0x8877665544332211);
  initialize_signal_info(&info);

  initialize_signal_request(&request, SAGR_WIRE_SIGNAL_OPCODE_CREATE);
  request.value_bits = initial;
  initialize_signal_response(&response, &request, initial,
                             UINT64_C(0x123456789abcdef5));
  response.signal_id = UINT64_C(7);
  response.generation = generation;
  if (expect_signal_case("SIGNAL_CREATE", &info,
                         UINT64_C(0x0123456789abcdf5), &request, &response,
                         SAGR_WIRE_MESSAGE_SIGNAL_ACK,
                         k_golden_signal_create_request,
                         k_golden_signal_create_ack) != 0) {
    return 1;
  }

  initialize_signal_request(&request, SAGR_WIRE_SIGNAL_OPCODE_LOAD);
  initialize_signal_response(&response, &request, initial,
                             UINT64_C(0x123456789abcdef6));
  if (expect_signal_case("SIGNAL_LOAD", &info,
                         UINT64_C(0x0123456789abcdf6), &request, &response,
                         SAGR_WIRE_MESSAGE_SIGNAL_ACK,
                         k_golden_signal_load_request,
                         k_golden_signal_load_ack) != 0) {
    return 1;
  }

  initialize_signal_request(&request, SAGR_WIRE_SIGNAL_OPCODE_WAIT);
  request.sequence = UINT64_C(0x0100000000000002);
  request.condition = SAGR_SIGNAL_CONDITION_GTE;
  initialize_signal_response(&response, &request, initial,
                             UINT64_C(0x123456789abcdef7));
  if (expect_signal_case("SIGNAL_WAIT", &info,
                         UINT64_C(0x0123456789abcdf7), &request, &response,
                         SAGR_WIRE_MESSAGE_SIGNAL_ACK,
                         k_golden_signal_wait_request,
                         k_golden_signal_wait_ack) != 0) {
    return 1;
  }

  initialize_signal_request(&request, SAGR_WIRE_SIGNAL_OPCODE_STORE);
  request.value_bits = UINT64_C(42);
  initialize_signal_response(&response, &request, UINT64_C(42),
                             UINT64_C(0x123456789abcdef8));
  if (expect_signal_case("SIGNAL_STORE", &info,
                         UINT64_C(0x0123456789abcdf8), &request, &response,
                         SAGR_WIRE_MESSAGE_SIGNAL_ACK,
                         k_golden_signal_store_request,
                         k_golden_signal_store_ack) != 0) {
    return 1;
  }

  initialize_signal_request(&request, SAGR_WIRE_SIGNAL_OPCODE_WAIT);
  request.sequence = UINT64_C(0x0100000000000002);
  request.condition = SAGR_SIGNAL_CONDITION_GTE;
  initialize_signal_response(&response, &request, UINT64_C(42),
                             UINT64_C(0x123456789abcdef9));
  if (expect_signal_case("SIGNAL_COMPLETION", &info,
                         UINT64_C(0x0123456789abcdf7), &request, &response,
                         SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION, NULL,
                         k_golden_signal_wait_completion) != 0) {
    return 1;
  }

  initialize_signal_request(&request, SAGR_WIRE_SIGNAL_OPCODE_DESTROY);
  initialize_signal_response(&response, &request, 0,
                             UINT64_C(0x123456789abcdefa));
  return expect_signal_case("SIGNAL_DESTROY", &info,
                            UINT64_C(0x0123456789abcdf9), &request, &response,
                            SAGR_WIRE_MESSAGE_SIGNAL_ACK,
                            k_golden_signal_destroy_request,
                            k_golden_signal_destroy_ack);
}

static int test_signal_request_structural_rules(void) {
  sagr_instance_info_t info;
  sagr_wire_signal_request_t request;
  uint8_t frame[SAGR_WIRE_SIGNAL_FRAME_BYTES];
  size_t frame_size = 0;
  initialize_signal_info(&info);
  initialize_signal_request(&request, SAGR_WIRE_SIGNAL_OPCODE_WAIT);
  request.sequence = 1;
  request.condition = SAGR_SIGNAL_CONDITION_GTE;
  if (sagr_protocol_encode_signal_request(
          &info, 1, &request, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_SUCCESS) {
    return 1;
  }
  request.condition = UINT64_C(4);
  if (sagr_protocol_encode_signal_request(
          &info, 1, &request, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_INVALID_ARGUMENT) {
    fprintf(stderr, "unknown signal wait condition was encoded\n");
    return 1;
  }
  request.condition = SAGR_SIGNAL_CONDITION_EQ;
  request.sequence = 0;
  if (sagr_protocol_encode_signal_request(
          &info, 1, &request, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_INVALID_ARGUMENT) {
    fprintf(stderr, "zero signal wait sequence was encoded\n");
    return 1;
  }
  initialize_signal_request(&request, SAGR_WIRE_SIGNAL_OPCODE_LOAD);
  request.value_bits = 1;
  if (sagr_protocol_encode_signal_request(
          &info, 1, &request, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_INVALID_ARGUMENT) {
    fprintf(stderr, "noncanonical signal load was encoded\n");
    return 1;
  }
  return 0;
}

static int expect_signal_decode_status(
    const uint8_t *frame, size_t frame_size, const sagr_instance_info_t *info,
    uint64_t request_id, uint16_t message_type, sagr_status_t expected) {
  sagr_wire_signal_response_t decoded;
  int32_t wire_status = -1;
  const char *reason = NULL;
  const sagr_status_t actual = sagr_protocol_decode_signal_response(
      frame, frame_size, info, request_id, message_type, &decoded,
      &wire_status, &reason);
  if (actual != expected) {
    fprintf(stderr, "signal mutation status=%d expected=%d: %s\n", actual,
            expected, reason == NULL ? "no reason" : reason);
    return 1;
  }
  return 0;
}

static int test_signal_response_mutations(void) {
  sagr_instance_info_t info;
  uint8_t golden[SAGR_WIRE_SIGNAL_FRAME_BYTES];
  uint8_t mutated[SAGR_WIRE_SIGNAL_FRAME_BYTES];
  size_t golden_size = 0;
  const uint64_t request_id = UINT64_C(0x0123456789abcdf7);
  initialize_signal_info(&info);
  if (decode_hex(k_golden_signal_wait_ack, golden, sizeof(golden),
                 &golden_size) != 0) {
    return 1;
  }
#define EXPECT_SIGNAL_MUTATION(MUTATION, EXPECTED)                           \
  do {                                                                        \
    memcpy(mutated, golden, golden_size);                                     \
    MUTATION;                                                                 \
    sagr_protocol_recompute_frame_crc(mutated, golden_size);                  \
    if (expect_signal_decode_status(mutated, golden_size, &info, request_id,  \
                                    SAGR_WIRE_MESSAGE_SIGNAL_ACK, EXPECTED) != \
        0) {                                                                  \
      return 1;                                                               \
    }                                                                         \
  } while (0)
  EXPECT_SIGNAL_MUTATION(store_u16(mutated + 14,
                                   SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION),
                         SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_SIGNAL_MUTATION(store_u64(mutated + 24, request_id + 1),
                         SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_SIGNAL_MUTATION(mutated[32] ^= 1, SAGR_STATUS_INSTANCE_MISMATCH);
  EXPECT_SIGNAL_MUTATION(store_u64(mutated + 48, info.connection_id + 1),
                         SAGR_STATUS_TOPOLOGY_MISMATCH);
  EXPECT_SIGNAL_MUTATION(store_u16(mutated + SAGR_WIRE_HEADER_BYTES, 2),
                         SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_SIGNAL_MUTATION(store_u16(mutated + SAGR_WIRE_HEADER_BYTES + 8, 99),
                         SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_SIGNAL_MUTATION(store_u16(mutated + SAGR_WIRE_HEADER_BYTES + 10, 1),
                         SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_SIGNAL_MUTATION(store_u32(mutated + SAGR_WIRE_HEADER_BYTES + 12, 1),
                         SAGR_STATUS_PROTOCOL_ERROR);
  EXPECT_SIGNAL_MUTATION(store_u64(mutated + SAGR_WIRE_HEADER_BYTES + 48, 2),
                         SAGR_STATUS_PROTOCOL_ERROR);
#undef EXPECT_SIGNAL_MUTATION
  memcpy(mutated, golden, golden_size);
  mutated[golden_size - 1U] ^= 1;
  return expect_signal_decode_status(
             mutated, golden_size, &info, request_id,
             SAGR_WIRE_MESSAGE_SIGNAL_ACK, SAGR_STATUS_CHECKSUM_ERROR) ||
         expect_signal_decode_status(
             golden, golden_size - 1U, &info, request_id,
             SAGR_WIRE_MESSAGE_SIGNAL_ACK, SAGR_STATUS_PROTOCOL_ERROR);
}

static int test_signal_capability_and_failed_ack(void) {
  sagr_wire_ack_fields_t fields;
  sagr_instance_open_options_t options;
  sagr_wire_ack_result_t result;
  sagr_instance_info_t signal_info;
  sagr_wire_signal_request_t request;
  sagr_wire_signal_response_t response;
  sagr_wire_signal_response_t decoded;
  uint8_t frame[SAGR_WIRE_ACK_FRAME_BYTES];
  size_t frame_size = 0;
  int32_t wire_status = -1;
  const char *reason = NULL;
  initialize_success_ack(&fields);
  fields.selected_capabilities[0] |= SAGR_CAPABILITY_SIGNAL_MASK;
  initialize_golden_options(&options);
  options.offered_capabilities[0] |= SAGR_CAPABILITY_SIGNAL_MASK;
  if (sagr_protocol_encode_ack(&fields, frame, sizeof(frame), &frame_size) !=
          SAGR_STATUS_SUCCESS ||
      sagr_protocol_decode_ack(frame, frame_size, &options, fields.request_id,
                               k_client_nonce, &result, &wire_status,
                               &reason) != SAGR_STATUS_CAPABILITY_MISMATCH) {
    fprintf(stderr, "offered-only signal capability ACK was accepted\n");
    return 1;
  }
  options.required_capabilities[0] |= SAGR_CAPABILITY_SIGNAL_MASK;
  if (sagr_protocol_decode_ack(frame, frame_size, &options, fields.request_id,
                               k_client_nonce, &result, &wire_status,
                               &reason) != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "required signal capability ACK was rejected\n");
    return 1;
  }
  initialize_signal_request(&request, SAGR_WIRE_SIGNAL_OPCODE_WAIT);
  request.sequence = 1;
  request.condition = SAGR_SIGNAL_CONDITION_GTE;
  initialize_signal_response(&response, &request, 0, 0);
  response.status = SAGR_WIRE_STATUS_BUSY;
  if (sagr_protocol_validate_failed_signal_ack(&request, &response) !=
      SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "canonical failed signal ACK was rejected\n");
    return 1;
  }
#define EXPECT_FAILED_SIGNAL_ACK_REJECTED(MUTATION)                          \
  do {                                                                        \
    sagr_wire_signal_response_t changed = response;                           \
    MUTATION;                                                                 \
    if (sagr_protocol_validate_failed_signal_ack(&request, &changed) !=       \
        SAGR_STATUS_PROTOCOL_ERROR) {                                         \
      fprintf(stderr, "noncanonical failed signal ACK was accepted\n");   \
      return 1;                                                               \
    }                                                                         \
  } while (0)
  EXPECT_FAILED_SIGNAL_ACK_REJECTED(changed.signal_id ^= UINT64_C(1));
  EXPECT_FAILED_SIGNAL_ACK_REJECTED(changed.generation ^= UINT64_C(1));
  EXPECT_FAILED_SIGNAL_ACK_REJECTED(changed.sequence ^= UINT64_C(1));
  EXPECT_FAILED_SIGNAL_ACK_REJECTED(changed.value_bits = UINT64_C(1));
  EXPECT_FAILED_SIGNAL_ACK_REJECTED(changed.ready = UINT64_C(1));
  EXPECT_FAILED_SIGNAL_ACK_REJECTED(changed.sim_tick = UINT64_C(1));
  EXPECT_FAILED_SIGNAL_ACK_REJECTED(changed.status = SAGR_WIRE_STATUS_OK);
#undef EXPECT_FAILED_SIGNAL_ACK_REJECTED

  initialize_signal_info(&signal_info);
  request.opcode = UINT16_C(99);
  response.status = SAGR_WIRE_STATUS_MALFORMED;
  response.opcode = request.opcode;
  if (sagr_protocol_encode_signal_response(
          &signal_info, UINT64_C(77), SAGR_WIRE_MESSAGE_SIGNAL_ACK, &response,
          frame, sizeof(frame), &frame_size) != SAGR_STATUS_SUCCESS ||
      sagr_protocol_decode_signal_response(
          frame, frame_size, &signal_info, UINT64_C(77),
          SAGR_WIRE_MESSAGE_SIGNAL_ACK, &decoded, &wire_status, &reason) !=
          SAGR_STATUS_PROTOCOL_ERROR ||
      wire_status != SAGR_WIRE_STATUS_MALFORMED ||
      decoded.opcode != request.opcode ||
      sagr_protocol_validate_failed_signal_ack(&request, &decoded) !=
          SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "raw-opcode MALFORMED signal ACK was not preserved\n");
    return 1;
  }
  response.status = SAGR_WIRE_STATUS_OK;
  if (sagr_protocol_encode_signal_response(
          &signal_info, UINT64_C(78), SAGR_WIRE_MESSAGE_SIGNAL_ACK, &response,
          frame, sizeof(frame), &frame_size) != SAGR_STATUS_INVALID_ARGUMENT) {
    fprintf(stderr, "raw opcode was accepted in successful signal ACK\n");
    return 1;
  }
  return 0;
}

static int test_request_id_exhaustion(void) {
  uint64_t next_request_id = UINT64_MAX - UINT64_C(1);
  uint64_t request_id = 0;
  if (sagr_protocol_allocate_request_id(&next_request_id, &request_id) !=
          SAGR_STATUS_SUCCESS ||
      request_id != UINT64_MAX - UINT64_C(1) ||
      next_request_id != UINT64_MAX ||
      sagr_protocol_allocate_request_id(&next_request_id, &request_id) !=
          SAGR_STATUS_SUCCESS ||
      request_id != UINT64_MAX || next_request_id != 0) {
    fprintf(stderr, "request ID allocator did not reach exhaustion exactly\n");
    return 1;
  }
  request_id = UINT64_C(42);
  if (sagr_protocol_allocate_request_id(&next_request_id, &request_id) !=
          SAGR_STATUS_OUT_OF_RESOURCES ||
      request_id != 0 || next_request_id != 0) {
    fprintf(stderr, "exhausted request ID allocator wrapped or reused an ID\n");
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
  failures += test_queue_golden_frames();
  failures += test_queue_response_mutations();
  failures += test_queue_capability_must_be_required();
  failures += test_failed_queue_ack_shape();
  failures += test_memory_golden_frames();
  failures += test_memory_request_structural_limits();
  failures += test_memory_response_mutations();
  failures += test_memory_capability_must_be_required();
  failures += test_failed_memory_ack_shape();
  failures += test_signal_golden_frames();
  failures += test_signal_request_structural_rules();
  failures += test_signal_response_mutations();
  failures += test_signal_capability_and_failed_ack();
  failures += test_request_id_exhaustion();
  return failures == 0 ? 0 : 1;
}
