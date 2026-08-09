/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <poll.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <self_amdgpu_runtime/code_object.h>
#include <self_amdgpu_runtime/runtime.h>

#include "transport_internal.h"

#ifndef SAGR_CODE_OBJECT_GPU_READ_WRITE_PATH
#define SAGR_CODE_OBJECT_GPU_READ_WRITE_PATH ""
#endif

#define TEST_OBJECT_ID UINT64_C(0x434f420000000001)
#define TEST_OBJECT_GENERATION UINT64_C(1)
#define TEST_QUEUE_ID UINT64_C(0x1020304050607080)
#define TEST_QUEUE_GENERATION UINT64_C(0x8877665544332211)
#define TEST_INPUT_ID UINT64_C(7)
#define TEST_INPUT_GENERATION UINT64_C(0x1000000000000001)
#define TEST_INPUT_VA UINT64_C(0x0000100300000000)
#define TEST_OUTPUT_ID UINT64_C(8)
#define TEST_OUTPUT_GENERATION UINT64_C(0x1000000000000002)
#define TEST_OUTPUT_VA UINT64_C(0x0000100380000000)
#define TEST_SIGNAL_ID UINT64_C(9)
#define TEST_SIGNAL_GENERATION UINT64_C(0x2000000000000001)
#define TEST_TRACE_ID UINT64_C(0x5452434500000001)
#define TEST_ADMISSION_TICK UINT64_C(100)
#define TEST_START_TICK UINT64_C(110)
#define TEST_END_TICK UINT64_C(120)
#define TEST_RETIRE_TICK UINT64_C(130)

static const uint8_t k_daemon_uuid[16] = {
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
static const uint8_t k_job_uuid[16] = {
    0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87,
    0x98, 0xa9, 0xba, 0xcb, 0xdc, 0xed, 0xfe, 0x0f};
static const uint8_t k_server_nonce[16] = {
    0xf0, 0xe0, 0xd0, 0xc0, 0xb0, 0xa0, 0x90, 0x80,
    0x70, 0x60, 0x50, 0x40, 0x30, 0x20, 0x10, 0x01};

enum upload_server_behavior {
  UPLOAD_SERVER_INTERLEAVED_SUCCESS,
  UPLOAD_SERVER_INTERLEAVED_DISPATCH_SUCCESS,
  UPLOAD_SERVER_CANONICAL_REJECTION,
  UPLOAD_SERVER_MALFORMED_REJECTION,
  UPLOAD_SERVER_WRONG_OPCODE_REJECTION,
  UPLOAD_SERVER_ACK_TIMEOUT,
  UPLOAD_SERVER_ACK_CANCEL,
  UPLOAD_SERVER_PRE_SEND_EXHAUSTION
};

typedef struct upload_server {
  char directory[128];
  char endpoint[160];
  int listener;
  pthread_t thread;
  enum upload_server_behavior behavior;
  int cancel_write_fd;
  int thread_error;
  int saw_disconnect;
  int cleanup_observed;
} upload_server_t;

typedef struct upload_state {
  int active;
  int rejected_once;
  uint64_t image_size;
  uint64_t accepted_offset;
  uint32_t chunk_count;
  uint32_t next_chunk;
  uint32_t kernel_index;
  uint32_t segment_count;
  uint8_t digest[32];
} upload_state_t;

typedef struct dispatch_setup_state {
  uint64_t signal_wait_request_id;
  uint64_t signal_wait_sequence;
  uint64_t dispatch_request_id;
  sagr_wire_dispatch_request_t request;
} dispatch_setup_state_t;

/* Test-only prefix used to force request-ID exhaustion after BEGIN succeeds. */
typedef struct test_instance_prefix {
  uint64_t magic;
  int socket_fd;
  uint64_t next_request_id;
} test_instance_prefix_t;

_Static_assert(offsetof(test_instance_prefix_t, next_request_id) == 16U,
               "test instance prefix layout drifted");

static uint16_t
get_u16(const uint8_t *source)
{
  return (uint16_t)(((uint16_t)source[0] << 8) | source[1]);
}

static uint32_t
get_u32(const uint8_t *source)
{
  return ((uint32_t)source[0] << 24) | ((uint32_t)source[1] << 16) |
         ((uint32_t)source[2] << 8) | source[3];
}

static uint64_t
get_u64(const uint8_t *source)
{
  return ((uint64_t)get_u32(source) << 32) | get_u32(source + 4);
}

static int
send_frame(int peer, const uint8_t *frame, size_t frame_size)
{
  ssize_t sent;
  do {
    sent = send(peer, frame, frame_size, MSG_NOSIGNAL);
  } while (sent < 0 && errno == EINTR);
  return sent == (ssize_t)frame_size ? 0 : -1;
}

static int
receive_frame(int peer, uint8_t *frame, size_t capacity, size_t *frame_size)
{
  ssize_t received;
  do {
    received = recv(peer, frame, capacity, 0);
  } while (received < 0 && errno == EINTR);
  if (received == 0) {
    *frame_size = 0U;
    return 1;
  }
  if (received < 0) {
    return -1;
  }
  *frame_size = (size_t)received;
  return 0;
}

static int
send_handshake_ack(int peer, const uint8_t *hello, size_t hello_size,
                   uint64_t capabilities, sagr_instance_info_t *info,
                   uint64_t *last_request_id)
{
  sagr_wire_ack_fields_t fields;
  uint8_t ack[SAGR_WIRE_ACK_FRAME_BYTES];
  size_t ack_size = 0U;
  if (hello_size < SAGR_WIRE_HEADER_BYTES + SAGR_WIRE_HELLO_FIXED_BYTES) {
    return -1;
  }
  memset(&fields, 0, sizeof(fields));
  fields.selected_major = 1U;
  fields.status = SAGR_WIRE_STATUS_OK;
  memcpy(fields.client_nonce, hello + SAGR_WIRE_HEADER_BYTES + 8U, 16U);
  memcpy(fields.server_nonce, k_server_nonce, sizeof(k_server_nonce));
  fields.selected_capabilities[0] = capabilities;
  fields.maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  fields.request_id = get_u64(hello + 24U);
  memcpy(fields.daemon_uuid, k_daemon_uuid, sizeof(k_daemon_uuid));
  fields.connection_id = UINT64_C(0x1122334455667788);
  fields.epoch = UINT64_C(0x0102030405060708);
  memcpy(fields.job_uuid, k_job_uuid, sizeof(k_job_uuid));
  fields.rank = 3U;
  fields.world_size = 8U;
  fields.include_topology = 1U;
  if (fields.request_id == 0U ||
      sagr_protocol_encode_ack(&fields, ack, sizeof(ack), &ack_size) !=
          SAGR_STATUS_SUCCESS ||
      send_frame(peer, ack, ack_size) != 0) {
    return -1;
  }
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  info->maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  info->negotiated_capabilities[0] = capabilities;
  memcpy(info->daemon_uuid, fields.daemon_uuid, 16U);
  info->connection_id = fields.connection_id;
  info->epoch = fields.epoch;
  *last_request_id = fields.request_id;
  return 0;
}

static int
send_queue_response(int peer, const sagr_instance_info_t *info,
                    uint64_t request_id, uint16_t message_type,
                    const sagr_wire_queue_response_t *response)
{
  uint8_t frame[SAGR_WIRE_QUEUE_FRAME_BYTES];
  size_t frame_size = 0U;
  if (sagr_protocol_encode_queue_response(
          info, request_id, message_type, response, frame, sizeof(frame),
          &frame_size) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static int
receive_typed_frame(int peer, const sagr_instance_info_t *info,
                    uint64_t *last_request_id, uint16_t message_type,
                    size_t expected_size, uint8_t *frame,
                    uint64_t *request_id)
{
  uint8_t copy[SAGR_WIRE_CODE_OBJECT_FRAME_BYTES];
  size_t frame_size = 0U;
  if (expected_size > sizeof(copy) ||
      receive_frame(peer, frame, SAGR_WIRE_CODE_OBJECT_FRAME_BYTES,
                    &frame_size) != 0 ||
      frame_size != expected_size || get_u16(frame + 8U) != 1U ||
      get_u16(frame + 10U) != 0U ||
      get_u16(frame + 12U) != SAGR_WIRE_HEADER_BYTES ||
      get_u16(frame + 14U) != message_type || get_u32(frame + 16U) != 0U ||
      get_u32(frame + 20U) != expected_size - SAGR_WIRE_HEADER_BYTES ||
      get_u64(frame + 24U) == 0U ||
      get_u64(frame + 24U) <= *last_request_id ||
      memcmp(frame + 32U, info->daemon_uuid, 16U) != 0 ||
      get_u64(frame + 48U) != info->connection_id ||
      get_u64(frame + 56U) != info->epoch || get_u32(frame + 68U) != 0U ||
      get_u64(frame + 72U) != 0U) {
    return -1;
  }
  memcpy(copy, frame, expected_size);
  memset(copy + 64U, 0, 4U);
  if (sagr_crc32c(copy, expected_size) != get_u32(frame + 64U)) {
    return -1;
  }
  *request_id = get_u64(frame + 24U);
  *last_request_id = *request_id;
  return 0;
}

static int
send_memory_allocate_response(int peer, const sagr_instance_info_t *info,
                              uint64_t request_id, uint64_t allocation_id,
                              uint64_t generation, uint64_t simulated_va)
{
  sagr_wire_memory_response_t response;
  uint8_t frame[SAGR_WIRE_MEMORY_FRAME_BYTES];
  size_t frame_size = 0U;
  memset(&response, 0, sizeof(response));
  response.major = SAGR_MEMORY_PROTOCOL_MAJOR;
  response.minor = SAGR_MEMORY_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = SAGR_WIRE_MEMORY_OPCODE_ALLOC;
  response.allocation_id = allocation_id;
  response.generation = generation;
  response.value0 = simulated_va;
  response.value1 = SAGR_DISPATCH_FIXED_IO_BYTES;
  response.value2 = SAGR_MEMORY_ALIGNMENT_4K;
  response.sim_tick = request_id + UINT64_C(10);
  if (sagr_protocol_encode_memory_response(
          info, request_id, &response, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static int
send_signal_record(int peer, const sagr_instance_info_t *info,
                   uint64_t request_id, uint16_t message_type,
                   uint16_t opcode, uint64_t sequence, uint64_t value_bits,
                   uint64_t sim_tick)
{
  sagr_wire_signal_response_t response;
  uint8_t frame[SAGR_WIRE_SIGNAL_FRAME_BYTES];
  size_t frame_size = 0U;
  memset(&response, 0, sizeof(response));
  response.major = SAGR_SIGNAL_PROTOCOL_MAJOR;
  response.minor = SAGR_SIGNAL_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = opcode;
  response.signal_id = TEST_SIGNAL_ID;
  response.generation = TEST_SIGNAL_GENERATION;
  response.sequence = sequence;
  response.value_bits = value_bits;
  response.sim_tick = sim_tick;
  if (sagr_protocol_encode_signal_response(
          info, request_id, message_type, &response, frame, sizeof(frame),
          &frame_size) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static void
initialize_dispatch_response(const dispatch_setup_state_t *dispatch,
                             sagr_wire_dispatch_response_t *response)
{
  const sagr_wire_dispatch_request_t *request = &dispatch->request;
  memset(response, 0, sizeof(*response));
  response->major = SAGR_DISPATCH_PROTOCOL_MAJOR;
  response->minor = SAGR_DISPATCH_PROTOCOL_MINOR;
  response->status = SAGR_WIRE_STATUS_OK;
  response->opcode = request->opcode;
  response->queue_id = request->queue_id;
  response->queue_generation = request->queue_generation;
  response->queue_sequence = request->queue_sequence;
  response->fixture_id = request->fixture_id;
  response->input_allocation_id = request->input_allocation_id;
  response->input_generation = request->input_generation;
  response->output_allocation_id = request->output_allocation_id;
  response->output_generation = request->output_generation;
  response->signal_id = request->signal_id;
  response->signal_generation = request->signal_generation;
  response->trace_id = TEST_TRACE_ID;
  response->input_gpu_va = TEST_INPUT_VA;
  response->output_gpu_va = TEST_OUTPUT_VA;
  response->packet_crc32c = SAGR_DISPATCH_PACKET_CRC32C;
  response->admission_tick = TEST_ADMISSION_TICK;
}

static int
send_dispatch_record(int peer, const sagr_instance_info_t *info,
                     const dispatch_setup_state_t *dispatch,
                     uint16_t message_type,
                     const sagr_wire_dispatch_response_t *response)
{
  uint8_t frame[SAGR_WIRE_DISPATCH_RESULT_FRAME_BYTES];
  size_t frame_size = 0U;
  if (sagr_protocol_encode_dispatch_response(
          info, dispatch->dispatch_request_id, message_type, response, frame,
          sizeof(frame), &frame_size) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static int
handle_dispatch_setup(int peer, const sagr_instance_info_t *info,
                      uint64_t *last_request_id,
                      dispatch_setup_state_t *dispatch)
{
  uint8_t frame[SAGR_WIRE_CODE_OBJECT_FRAME_BYTES];
  const uint8_t *payload;
  uint64_t request_id = 0U;
  sagr_wire_queue_response_t queue_response;
  sagr_wire_dispatch_response_t dispatch_response;
  unsigned allocation_index;

  if (receive_typed_frame(peer, info, last_request_id,
                          SAGR_WIRE_MESSAGE_QUEUE_REQUEST,
                          SAGR_WIRE_QUEUE_FRAME_BYTES, frame,
                          &request_id) != 0) {
    return -1;
  }
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  if (get_u16(payload) != SAGR_QUEUE_PROTOCOL_MAJOR ||
      get_u16(payload + 2U) != SAGR_QUEUE_PROTOCOL_MINOR ||
      get_u16(payload + 4U) != SAGR_WIRE_QUEUE_OPCODE_CREATE ||
      get_u16(payload + 6U) != 0U || get_u64(payload + 8U) != 0U ||
      get_u64(payload + 16U) != 0U || get_u64(payload + 24U) != 0U ||
      get_u64(payload + 32U) != 1U || get_u64(payload + 40U) != 0U) {
    return -1;
  }
  memset(&queue_response, 0, sizeof(queue_response));
  queue_response.major = SAGR_QUEUE_PROTOCOL_MAJOR;
  queue_response.status = SAGR_WIRE_STATUS_OK;
  queue_response.opcode = SAGR_WIRE_QUEUE_OPCODE_CREATE;
  queue_response.queue_id = TEST_QUEUE_ID;
  queue_response.generation = TEST_QUEUE_GENERATION;
  queue_response.value = 1U;
  queue_response.sim_tick = request_id + UINT64_C(10);
  if (send_queue_response(peer, info, request_id,
                          SAGR_WIRE_MESSAGE_QUEUE_ACK,
                          &queue_response) != 0) {
    return -1;
  }

  for (allocation_index = 0U; allocation_index < 2U; ++allocation_index) {
    if (receive_typed_frame(peer, info, last_request_id,
                            SAGR_WIRE_MESSAGE_MEMORY_REQUEST,
                            SAGR_WIRE_MEMORY_FRAME_BYTES, frame,
                            &request_id) != 0) {
      return -1;
    }
    payload = frame + SAGR_WIRE_HEADER_BYTES;
    if (get_u16(payload) != SAGR_MEMORY_PROTOCOL_MAJOR ||
        get_u16(payload + 2U) != SAGR_MEMORY_PROTOCOL_MINOR ||
        get_u16(payload + 4U) != SAGR_WIRE_MEMORY_OPCODE_ALLOC ||
        get_u16(payload + 6U) != 0U || get_u64(payload + 8U) != 0U ||
        get_u64(payload + 16U) != 0U || get_u64(payload + 24U) != 0U ||
        get_u64(payload + 32U) != SAGR_DISPATCH_FIXED_IO_BYTES ||
        get_u64(payload + 40U) != SAGR_MEMORY_ALIGNMENT_4K ||
        send_memory_allocate_response(
            peer, info, request_id,
            allocation_index == 0U ? TEST_INPUT_ID : TEST_OUTPUT_ID,
            allocation_index == 0U ? TEST_INPUT_GENERATION
                                   : TEST_OUTPUT_GENERATION,
            allocation_index == 0U ? TEST_INPUT_VA : TEST_OUTPUT_VA) != 0) {
      return -1;
    }
  }

  if (receive_typed_frame(peer, info, last_request_id,
                          SAGR_WIRE_MESSAGE_SIGNAL_REQUEST,
                          SAGR_WIRE_SIGNAL_FRAME_BYTES, frame,
                          &request_id) != 0) {
    return -1;
  }
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  if (get_u16(payload) != SAGR_SIGNAL_PROTOCOL_MAJOR ||
      get_u16(payload + 2U) != SAGR_SIGNAL_PROTOCOL_MINOR ||
      get_u16(payload + 4U) != SAGR_WIRE_SIGNAL_OPCODE_CREATE ||
      get_u16(payload + 6U) != 0U || get_u64(payload + 8U) != 0U ||
      get_u64(payload + 16U) != 0U || get_u64(payload + 24U) != 0U ||
      get_u64(payload + 32U) != 1U || get_u64(payload + 40U) != 0U ||
      send_signal_record(peer, info, request_id,
                         SAGR_WIRE_MESSAGE_SIGNAL_ACK,
                         SAGR_WIRE_SIGNAL_OPCODE_CREATE, 0U, 1U,
                         request_id + UINT64_C(10)) != 0) {
    return -1;
  }

  if (receive_typed_frame(peer, info, last_request_id,
                          SAGR_WIRE_MESSAGE_SIGNAL_REQUEST,
                          SAGR_WIRE_SIGNAL_FRAME_BYTES, frame,
                          &request_id) != 0) {
    return -1;
  }
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  if (get_u16(payload) != SAGR_SIGNAL_PROTOCOL_MAJOR ||
      get_u16(payload + 2U) != SAGR_SIGNAL_PROTOCOL_MINOR ||
      get_u16(payload + 4U) != SAGR_WIRE_SIGNAL_OPCODE_WAIT ||
      get_u16(payload + 6U) != 0U ||
      get_u64(payload + 8U) != TEST_SIGNAL_ID ||
      get_u64(payload + 16U) != TEST_SIGNAL_GENERATION ||
      get_u64(payload + 24U) != 1U || get_u64(payload + 32U) != 0U ||
      get_u64(payload + 40U) != SAGR_SIGNAL_CONDITION_EQ ||
      send_signal_record(peer, info, request_id,
                         SAGR_WIRE_MESSAGE_SIGNAL_ACK,
                         SAGR_WIRE_SIGNAL_OPCODE_WAIT, 1U, 1U,
                         UINT64_C(80)) != 0) {
    return -1;
  }
  dispatch->signal_wait_request_id = request_id;
  dispatch->signal_wait_sequence = 1U;

  if (receive_typed_frame(peer, info, last_request_id,
                          SAGR_WIRE_MESSAGE_DISPATCH_REQUEST,
                          SAGR_WIRE_DISPATCH_REQUEST_FRAME_BYTES, frame,
                          &request_id) != 0) {
    return -1;
  }
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  memset(&dispatch->request, 0, sizeof(dispatch->request));
  dispatch->request.major = get_u16(payload);
  dispatch->request.minor = get_u16(payload + 2U);
  dispatch->request.opcode = get_u16(payload + 4U);
  dispatch->request.flags = get_u16(payload + 6U);
  dispatch->request.queue_id = get_u64(payload + 8U);
  dispatch->request.queue_generation = get_u64(payload + 16U);
  dispatch->request.queue_sequence = get_u64(payload + 24U);
  dispatch->request.fixture_id = get_u64(payload + 32U);
  dispatch->request.input_allocation_id = get_u64(payload + 40U);
  dispatch->request.input_generation = get_u64(payload + 48U);
  dispatch->request.output_allocation_id = get_u64(payload + 56U);
  dispatch->request.output_generation = get_u64(payload + 64U);
  dispatch->request.signal_id = get_u64(payload + 72U);
  dispatch->request.signal_generation = get_u64(payload + 80U);
  dispatch->request.expected_signal_value_bits = get_u64(payload + 88U);
  memcpy(dispatch->request.fixture_manifest_sha256, payload + 96U, 32U);
  dispatch->dispatch_request_id = request_id;
  if (dispatch->request.major != SAGR_DISPATCH_PROTOCOL_MAJOR ||
      dispatch->request.minor != SAGR_DISPATCH_PROTOCOL_MINOR ||
      dispatch->request.opcode != SAGR_WIRE_DISPATCH_OPCODE_SUBMIT_PINNED ||
      dispatch->request.flags != 0U ||
      dispatch->request.queue_id != TEST_QUEUE_ID ||
      dispatch->request.queue_generation != TEST_QUEUE_GENERATION ||
      dispatch->request.queue_sequence != 1U ||
      dispatch->request.fixture_id != SAGR_DISPATCH_FIXTURE_GFX950_XOR_U8_V1 ||
      dispatch->request.input_allocation_id != TEST_INPUT_ID ||
      dispatch->request.input_generation != TEST_INPUT_GENERATION ||
      dispatch->request.output_allocation_id != TEST_OUTPUT_ID ||
      dispatch->request.output_generation != TEST_OUTPUT_GENERATION ||
      dispatch->request.signal_id != TEST_SIGNAL_ID ||
      dispatch->request.signal_generation != TEST_SIGNAL_GENERATION ||
      dispatch->request.expected_signal_value_bits != 0U ||
      memcmp(dispatch->request.fixture_manifest_sha256,
             sagr_dispatch_fixture_manifest_sha256, 32U) != 0) {
    return -1;
  }
  initialize_dispatch_response(dispatch, &dispatch_response);
  return send_dispatch_record(peer, info, dispatch,
                              SAGR_WIRE_MESSAGE_DISPATCH_ACK,
                              &dispatch_response);
}

static int
send_dispatch_completions(int peer, const sagr_instance_info_t *info,
                          const dispatch_setup_state_t *dispatch)
{
  sagr_wire_dispatch_response_t completion;
  initialize_dispatch_response(dispatch, &completion);
  completion.output_crc32c = SAGR_DISPATCH_OUTPUT_CRC32C;
  completion.start_tick = TEST_START_TICK;
  completion.end_tick = TEST_END_TICK;
  completion.retire_tick = TEST_RETIRE_TICK;
  return send_signal_record(peer, info, dispatch->signal_wait_request_id,
                            SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION,
                            SAGR_WIRE_SIGNAL_OPCODE_WAIT,
                            dispatch->signal_wait_sequence, 0U,
                            TEST_RETIRE_TICK + UINT64_C(1)) != 0 ||
                 send_dispatch_record(peer, info, dispatch,
                                      SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION,
                                      &completion) != 0
             ? -1
             : 0;
}

static int
handle_queue_setup(int peer, const sagr_instance_info_t *info,
                   uint64_t *last_request_id, uint64_t *doorbell_request_id)
{
  uint8_t frame[SAGR_WIRE_CODE_OBJECT_FRAME_BYTES];
  size_t frame_size = 0U;
  uint64_t request_id;
  const uint8_t *payload;
  sagr_wire_queue_response_t response;
  if (receive_frame(peer, frame, sizeof(frame), &frame_size) != 0 ||
      frame_size != SAGR_WIRE_QUEUE_FRAME_BYTES ||
      get_u16(frame + 14U) != SAGR_WIRE_MESSAGE_QUEUE_REQUEST) {
    return -1;
  }
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  request_id = get_u64(frame + 24U);
  if (request_id <= *last_request_id ||
      get_u16(payload + 4U) != SAGR_WIRE_QUEUE_OPCODE_CREATE ||
      get_u64(payload + 8U) != 0U || get_u64(payload + 16U) != 0U ||
      get_u64(payload + 24U) != 0U || get_u64(payload + 32U) != 4U) {
    return -1;
  }
  *last_request_id = request_id;
  memset(&response, 0, sizeof(response));
  response.major = SAGR_QUEUE_PROTOCOL_MAJOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = SAGR_WIRE_QUEUE_OPCODE_CREATE;
  response.queue_id = TEST_QUEUE_ID;
  response.generation = TEST_QUEUE_GENERATION;
  response.value = 4U;
  response.sim_tick = 100U;
  if (send_queue_response(peer, info, request_id, SAGR_WIRE_MESSAGE_QUEUE_ACK,
                          &response) != 0 ||
      receive_frame(peer, frame, sizeof(frame), &frame_size) != 0 ||
      frame_size != SAGR_WIRE_QUEUE_FRAME_BYTES ||
      get_u16(frame + 14U) != SAGR_WIRE_MESSAGE_QUEUE_REQUEST) {
    return -1;
  }
  payload = frame + SAGR_WIRE_HEADER_BYTES;
  request_id = get_u64(frame + 24U);
  if (request_id <= *last_request_id ||
      get_u16(payload + 4U) != SAGR_WIRE_QUEUE_OPCODE_DOORBELL ||
      get_u64(payload + 8U) != TEST_QUEUE_ID ||
      get_u64(payload + 16U) != TEST_QUEUE_GENERATION ||
      get_u64(payload + 24U) != 1U || get_u64(payload + 32U) != 0U) {
    return -1;
  }
  *last_request_id = request_id;
  *doorbell_request_id = request_id;
  memset(&response, 0, sizeof(response));
  response.major = SAGR_QUEUE_PROTOCOL_MAJOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = SAGR_WIRE_QUEUE_OPCODE_DOORBELL;
  response.queue_id = TEST_QUEUE_ID;
  response.generation = TEST_QUEUE_GENERATION;
  response.sequence = 1U;
  response.sim_tick = 101U;
  return send_queue_response(peer, info, request_id,
                             SAGR_WIRE_MESSAGE_QUEUE_ACK, &response);
}

static int
send_prior_queue_completion(int peer, const sagr_instance_info_t *info,
                            uint64_t request_id)
{
  sagr_wire_queue_response_t completion;
  memset(&completion, 0, sizeof(completion));
  completion.major = SAGR_QUEUE_PROTOCOL_MAJOR;
  completion.status = SAGR_WIRE_STATUS_OK;
  completion.opcode = SAGR_WIRE_QUEUE_OPCODE_DOORBELL;
  completion.queue_id = TEST_QUEUE_ID;
  completion.generation = TEST_QUEUE_GENERATION;
  completion.sequence = 1U;
  completion.value = SAGR_QUEUE_COMMAND_NOOP;
  completion.sim_tick = 102U;
  return send_queue_response(peer, info, request_id,
                             SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &completion);
}

static void
fill_success_response(const upload_state_t *state, uint16_t opcode,
                      sagr_wire_code_object_response_t *response)
{
  memset(response, 0, sizeof(*response));
  response->major = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MAJOR;
  response->minor = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MINOR;
  response->status = SAGR_WIRE_STATUS_OK;
  response->opcode = opcode;
  response->object_id = TEST_OBJECT_ID;
  response->generation = TEST_OBJECT_GENERATION;
  response->image_size = state->image_size;
  response->kernel_index = state->kernel_index;
  response->segment_count = state->segment_count;
  response->sim_tick = UINT64_C(200) + state->next_chunk;
  memcpy(response->image_sha256, state->digest, sizeof(response->image_sha256));
}

static int
send_code_response(int peer, const sagr_instance_info_t *info,
                   uint64_t request_id,
                   const sagr_wire_code_object_response_t *response)
{
  uint8_t frame[SAGR_WIRE_CODE_OBJECT_FRAME_BYTES];
  size_t frame_size = 0U;
  if (sagr_protocol_encode_code_object_response(
          info, request_id, response, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static int
send_failed_chunk(int peer, const sagr_instance_info_t *info,
                  uint64_t request_id, enum upload_server_behavior behavior)
{
  sagr_wire_code_object_response_t response;
  uint8_t frame[SAGR_WIRE_CODE_OBJECT_FRAME_BYTES];
  size_t frame_size = 0U;
  memset(&response, 0, sizeof(response));
  response.major = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MAJOR;
  response.minor = SAGR_CODE_OBJECT_TRANSPORT_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_MALFORMED;
  response.opcode = behavior == UPLOAD_SERVER_WRONG_OPCODE_REJECTION
                        ? SAGR_WIRE_CODE_OBJECT_OPCODE_COMMIT
                        : SAGR_WIRE_CODE_OBJECT_OPCODE_CHUNK;
  response.error_code = SAGR_WIRE_STATUS_MALFORMED;
  if (sagr_protocol_encode_code_object_response(
          info, request_id, &response, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_SUCCESS) {
    return -1;
  }
  if (behavior == UPLOAD_SERVER_MALFORMED_REJECTION) {
    frame[SAGR_WIRE_HEADER_BYTES + 23U] = 1U;
    sagr_protocol_recompute_frame_crc(frame, frame_size);
  }
  return send_frame(peer, frame, frame_size);
}

static int
wait_for_disconnect(int peer)
{
  struct pollfd descriptor;
  uint8_t byte;
  int ready;
  descriptor.fd = peer;
  descriptor.events = POLLIN | POLLHUP;
  descriptor.revents = 0;
  do {
    ready = poll(&descriptor, 1, 2000);
  } while (ready < 0 && errno == EINTR);
  if (ready <= 0) {
    return -1;
  }
  return recv(peer, &byte, sizeof(byte), 0) == 0 ? 0 : -1;
}

static void *
upload_server_main(void *argument)
{
  upload_server_t *server = (upload_server_t *)argument;
  sagr_instance_info_t info;
  upload_state_t state;
  uint8_t frame[SAGR_WIRE_CODE_OBJECT_FRAME_BYTES];
  uint64_t last_request_id = 0U;
  uint64_t doorbell_request_id = 0U;
  dispatch_setup_state_t dispatch;
  size_t frame_size = 0U;
  int peer = -1;
  int committed = 0;
  uint64_t capabilities = SAGR_CAPABILITY_TOPOLOGY_MASK |
                          SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK;

  memset(&state, 0, sizeof(state));
  memset(&dispatch, 0, sizeof(dispatch));
  do {
    peer = accept4(server->listener, NULL, NULL, SOCK_CLOEXEC);
  } while (peer < 0 && errno == EINTR);
  if (peer < 0 || receive_frame(peer, frame, sizeof(frame), &frame_size) != 0) {
    server->thread_error = 1;
    goto done;
  }
  if (server->behavior == UPLOAD_SERVER_INTERLEAVED_SUCCESS) {
    capabilities |= SAGR_CAPABILITY_QUEUE_MASK;
  } else if (server->behavior == UPLOAD_SERVER_INTERLEAVED_DISPATCH_SUCCESS) {
    capabilities |= SAGR_CAPABILITY_QUEUE_MASK | SAGR_CAPABILITY_MEMORY_MASK |
                    SAGR_CAPABILITY_SIGNAL_MASK | SAGR_CAPABILITY_DISPATCH_MASK;
  }
  if (send_handshake_ack(peer, frame, frame_size, capabilities, &info,
                         &last_request_id) != 0) {
    server->thread_error = 1;
    goto done;
  }
  if (server->behavior == UPLOAD_SERVER_INTERLEAVED_SUCCESS &&
      handle_queue_setup(peer, &info, &last_request_id,
                         &doorbell_request_id) != 0) {
    server->thread_error = 1;
    goto done;
  }
  if (server->behavior == UPLOAD_SERVER_INTERLEAVED_DISPATCH_SUCCESS &&
      handle_dispatch_setup(peer, &info, &last_request_id, &dispatch) != 0) {
    server->thread_error = 1;
    goto done;
  }

  for (;;) {
    sagr_wire_code_object_request_t request;
    sagr_wire_code_object_response_t response;
    uint64_t request_id = 0U;
    const char *reason = NULL;
    const int received = receive_frame(peer, frame, sizeof(frame), &frame_size);
    if (received == 1) {
      if (server->behavior == UPLOAD_SERVER_PRE_SEND_EXHAUSTION ||
          server->behavior == UPLOAD_SERVER_ACK_TIMEOUT ||
          server->behavior == UPLOAD_SERVER_ACK_CANCEL ||
          server->behavior == UPLOAD_SERVER_MALFORMED_REJECTION ||
          server->behavior == UPLOAD_SERVER_WRONG_OPCODE_REJECTION) {
        server->saw_disconnect = 1;
        state.active = 0;
        server->cleanup_observed = 1;
      }
      break;
    }
    if (received != 0 || frame_size != SAGR_WIRE_CODE_OBJECT_FRAME_BYTES ||
        sagr_protocol_decode_code_object_request(
            frame, frame_size, &info, &request, &request_id, &reason) !=
            SAGR_STATUS_SUCCESS ||
        request_id <= last_request_id) {
      server->thread_error = 1;
      break;
    }
    last_request_id = request_id;
    if (request.opcode == SAGR_WIRE_CODE_OBJECT_OPCODE_BEGIN) {
      if (state.active != 0) {
        server->thread_error = 1;
        break;
      }
      state.active = 1;
      state.image_size = request.body.begin.image_size;
      state.chunk_count = request.body.begin.chunk_count;
      state.kernel_index = request.body.begin.kernel_index;
      state.segment_count = request.body.begin.segment_count;
      memcpy(state.digest, request.body.begin.image_sha256,
             sizeof(state.digest));
      fill_success_response(&state, request.opcode, &response);
      if ((server->behavior == UPLOAD_SERVER_INTERLEAVED_SUCCESS &&
           send_prior_queue_completion(peer, &info, doorbell_request_id) != 0) ||
          (server->behavior == UPLOAD_SERVER_INTERLEAVED_DISPATCH_SUCCESS &&
           send_dispatch_completions(peer, &info, &dispatch) != 0) ||
          send_code_response(peer, &info, request_id, &response) != 0) {
        server->thread_error = 1;
        break;
      }
      continue;
    }
    if (request.opcode == SAGR_WIRE_CODE_OBJECT_OPCODE_CHUNK) {
      if (state.active == 0 || request.object_id != TEST_OBJECT_ID ||
          request.generation != TEST_OBJECT_GENERATION ||
          request.image_offset != state.accepted_offset ||
          request.chunk_index != state.next_chunk) {
        server->thread_error = 1;
        break;
      }
      if (server->behavior == UPLOAD_SERVER_ACK_TIMEOUT) {
        if (wait_for_disconnect(peer) != 0) {
          server->thread_error = 1;
        } else {
          server->saw_disconnect = 1;
          state.active = 0;
          server->cleanup_observed = 1;
        }
        break;
      }
      if (server->behavior == UPLOAD_SERVER_ACK_CANCEL) {
        const uint8_t cancel_byte = UINT8_C(1);
        if (server->cancel_write_fd < 0 ||
            write(server->cancel_write_fd, &cancel_byte,
                  sizeof(cancel_byte)) != (ssize_t)sizeof(cancel_byte) ||
            wait_for_disconnect(peer) != 0) {
          server->thread_error = 1;
        } else {
          server->saw_disconnect = 1;
          state.active = 0;
          server->cleanup_observed = 1;
        }
        break;
      }
      if (server->behavior == UPLOAD_SERVER_PRE_SEND_EXHAUSTION) {
        server->thread_error = 1;
        break;
      }
      if ((server->behavior == UPLOAD_SERVER_CANONICAL_REJECTION &&
           state.rejected_once == 0) ||
          server->behavior == UPLOAD_SERVER_MALFORMED_REJECTION ||
          server->behavior == UPLOAD_SERVER_WRONG_OPCODE_REJECTION) {
        if (send_failed_chunk(peer, &info, request_id, server->behavior) != 0) {
          server->thread_error = 1;
          break;
        }
        state.active = 0;
        state.accepted_offset = 0U;
        state.next_chunk = 0U;
        if (server->behavior == UPLOAD_SERVER_CANONICAL_REJECTION) {
          state.rejected_once = 1;
          continue;
        }
        if (wait_for_disconnect(peer) != 0) {
          server->thread_error = 1;
        } else {
          server->saw_disconnect = 1;
          state.active = 0;
          server->cleanup_observed = 1;
        }
        break;
      }
      fill_success_response(&state, request.opcode, &response);
      response.accepted_offset = request.image_offset;
      response.accepted_count = request.byte_count;
      response.chunk_index = request.chunk_index;
      if (send_code_response(peer, &info, request_id, &response) != 0) {
        server->thread_error = 1;
        break;
      }
      state.accepted_offset += request.byte_count;
      ++state.next_chunk;
      continue;
    }
    if (request.opcode != SAGR_WIRE_CODE_OBJECT_OPCODE_COMMIT ||
        state.active == 0 || request.object_id != TEST_OBJECT_ID ||
        request.generation != TEST_OBJECT_GENERATION ||
        request.byte_count != state.image_size ||
        request.chunk_index != state.chunk_count ||
        state.accepted_offset != state.image_size ||
        state.next_chunk != state.chunk_count ||
        memcmp(request.body.commit_sha256, state.digest,
               sizeof(state.digest)) != 0) {
      server->thread_error = 1;
      break;
    }
    fill_success_response(&state, request.opcode, &response);
    response.accepted_offset = state.image_size;
    response.accepted_count = (uint32_t)state.image_size;
    response.chunk_index = state.chunk_count;
    if (send_code_response(peer, &info, request_id, &response) != 0) {
      server->thread_error = 1;
      break;
    }
    state.active = 0;
    committed = 1;
    if (server->behavior != UPLOAD_SERVER_CANONICAL_REJECTION ||
        state.rejected_once != 0) {
      continue;
    }
  }
  if (committed == 0 &&
      (server->behavior == UPLOAD_SERVER_INTERLEAVED_SUCCESS ||
       server->behavior == UPLOAD_SERVER_INTERLEAVED_DISPATCH_SUCCESS)) {
    server->thread_error = 1;
  }

done:
  if (peer >= 0) {
    (void)close(peer);
  }
  return NULL;
}

static int
start_server_with_cancel(upload_server_t *server,
                          enum upload_server_behavior behavior,
                          int cancel_write_fd)
{
  struct sockaddr_un address;
  char template_path[] = "/tmp/sagr-code-upload-XXXXXX";
  char *directory;
  memset(server, 0, sizeof(*server));
  server->listener = -1;
  server->behavior = behavior;
  server->cancel_write_fd = cancel_write_fd;
  directory = mkdtemp(template_path);
  if (directory == NULL ||
      snprintf(server->directory, sizeof(server->directory), "%s",
               directory) >= (int)sizeof(server->directory) ||
      snprintf(server->endpoint, sizeof(server->endpoint), "%s/socket",
               directory) >= (int)sizeof(server->endpoint) ||
      strlen(server->endpoint) >= sizeof(address.sun_path)) {
    return -1;
  }
  server->listener = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (server->listener < 0) {
    return -1;
  }
  memset(&address, 0, sizeof(address));
  address.sun_family = AF_UNIX;
  memcpy(address.sun_path, server->endpoint, strlen(server->endpoint) + 1U);
  if (bind(server->listener, (const struct sockaddr *)&address,
           (socklen_t)sizeof(address)) != 0 ||
      listen(server->listener, 1) != 0 ||
      pthread_create(&server->thread, NULL, upload_server_main, server) != 0) {
    (void)close(server->listener);
    server->listener = -1;
    return -1;
  }
  return 0;
}

static int
start_server(upload_server_t *server, enum upload_server_behavior behavior)
{
  return start_server_with_cancel(server, behavior, -1);
}

static int
finish_server(upload_server_t *server)
{
  int failed = 0;
  if (pthread_join(server->thread, NULL) != 0 || server->thread_error != 0) {
    failed = 1;
  }
  if (server->listener >= 0) {
    (void)close(server->listener);
  }
  if (unlink(server->endpoint) != 0 || rmdir(server->directory) != 0) {
    failed = 1;
  }
  return failed;
}

static int
read_fixture(uint8_t **image, size_t *image_size)
{
  FILE *file = fopen(SAGR_CODE_OBJECT_GPU_READ_WRITE_PATH, "rb");
  long length;
  if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
    if (file != NULL) {
      (void)fclose(file);
    }
    return -1;
  }
  length = ftell(file);
  if (length <= 0L || fseek(file, 0, SEEK_SET) != 0) {
    (void)fclose(file);
    return -1;
  }
  *image_size = (size_t)length;
  *image = (uint8_t *)malloc(*image_size);
  if (*image == NULL || fread(*image, 1U, *image_size, file) != *image_size) {
    free(*image);
    *image = NULL;
    *image_size = 0U;
    (void)fclose(file);
    return -1;
  }
  return fclose(file) == 0 ? 0 : -1;
}

static int
open_instance(const upload_server_t *server, int capability_mode,
              sagr_instance_t *instance)
{
  sagr_instance_open_options_t options;
  sagr_error_info_t error;
  (void)sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));
  options.offered_capabilities[0] |=
      SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK;
  options.required_capabilities[0] |=
      SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK;
  if (capability_mode >= 1) {
    options.offered_capabilities[0] |= SAGR_CAPABILITY_QUEUE_MASK;
    options.required_capabilities[0] |= SAGR_CAPABILITY_QUEUE_MASK;
  }
  if (capability_mode >= 2) {
    options.offered_capabilities[0] |= SAGR_CAPABILITY_MEMORY_MASK |
                                       SAGR_CAPABILITY_SIGNAL_MASK |
                                       SAGR_CAPABILITY_DISPATCH_MASK;
    options.required_capabilities[0] |= SAGR_CAPABILITY_MEMORY_MASK |
                                        SAGR_CAPABILITY_SIGNAL_MASK |
                                        SAGR_CAPABILITY_DISPATCH_MASK;
  }
  memcpy(options.expected_daemon_uuid, k_daemon_uuid, sizeof(k_daemon_uuid));
  memcpy(options.expected_job_uuid, k_job_uuid, sizeof(k_job_uuid));
  options.expected_epoch = UINT64_C(0x0102030405060708);
  options.expected_rank = 3U;
  options.expected_world_size = 8U;
  options.open_timeout_ns = UINT64_C(1000000000);
  return sagr_instance_open(server->endpoint, &options, instance, &error,
                            (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS
             ? 0
             : -1;
}

static int
test_interleaved_upload(const uint8_t *image, size_t image_size)
{
  upload_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_create_options_t create;
  sagr_queue_info_t queue_info;
  sagr_queue_completion_t completion;
  sagr_code_object_remote_info_t remote;
  sagr_error_info_t error;
  uint64_t sequence = 0U;
  int failed = 0;
  if (start_server(&server, UPLOAD_SERVER_INTERLEAVED_SUCCESS) != 0 ||
      open_instance(&server, 1, &instance) != 0) {
    return 1;
  }
  (void)sagr_queue_create_options_init(&create, (uint32_t)sizeof(create));
  create.depth = 4U;
  if (sagr_queue_create(instance, &create, NULL, &queue, &queue_info,
                        (uint32_t)sizeof(queue_info), &error,
                        (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      sagr_queue_ring_doorbell(queue, SAGR_QUEUE_COMMAND_NOOP, NULL, &sequence,
                               &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_code_object_upload(instance, image, image_size, "gpuReadWrite", NULL,
                              &remote, (uint32_t)sizeof(remote), &error,
                              (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      remote.object_id != TEST_OBJECT_ID ||
      remote.generation != TEST_OBJECT_GENERATION ||
      remote.image_size != image_size || remote.mapped_base_va != 0U ||
      sagr_queue_wait(queue, sequence, NULL, &completion,
                      (uint32_t)sizeof(completion), &error,
                      (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      completion.sequence != sequence || completion.sim_tick != 102U) {
    fprintf(stderr, "code-object upload test: interleaved upload failed: %s\n",
            error.message);
    failed = 1;
  }
  (void)sagr_instance_close(&instance);
  failed |= finish_server(&server);
  return failed;
}

static int
test_interleaved_dispatch_upload(const uint8_t *image, size_t image_size)
{
  upload_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_memory_t input = NULL;
  sagr_memory_t output = NULL;
  sagr_signal_t signal = NULL;
  sagr_memory_allocate_options_t memory_options;
  sagr_signal_create_options_t signal_options;
  sagr_signal_operation_options_t signal_operation;
  sagr_signal_wait_result_t signal_result;
  sagr_pinned_dispatch_options_t dispatch_options;
  sagr_dispatch_ticket_t ticket;
  sagr_dispatch_completion_t completion;
  sagr_code_object_remote_info_t remote;
  sagr_error_info_t error;
  sagr_status_t status;
  int failed = 0;

  memset(&error, 0, sizeof(error));
  if (start_server(&server, UPLOAD_SERVER_INTERLEAVED_DISPATCH_SUCCESS) != 0 ||
      open_instance(&server, 2, &instance) != 0) {
    if (instance != NULL) {
      (void)sagr_instance_close(&instance);
    }
    if (server.listener >= 0) {
      failed |= finish_server(&server);
    }
    return 1;
  }
  if (sagr_queue_create(instance, NULL, NULL, &queue, NULL, 0, &error,
                        (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      sagr_memory_allocate_options_init(&memory_options,
                                        (uint32_t)sizeof(memory_options)) !=
          SAGR_STATUS_SUCCESS) {
    failed = 1;
    goto done;
  }
  memory_options.size_bytes = SAGR_DISPATCH_FIXED_IO_BYTES;
  if (sagr_memory_allocate(instance, &memory_options, NULL, &input, NULL, 0,
                           &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_memory_allocate(instance, &memory_options, NULL, &output, NULL, 0,
                           &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_signal_create_options_init(&signal_options,
                                      (uint32_t)sizeof(signal_options)) !=
          SAGR_STATUS_SUCCESS) {
    failed = 1;
    goto done;
  }
  signal_options.initial_value = INT64_C(1);
  if (sagr_signal_create(instance, &signal_options, NULL, &signal, NULL, 0,
                         &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_signal_operation_options_init(&signal_operation,
                                         (uint32_t)sizeof(signal_operation)) !=
          SAGR_STATUS_SUCCESS) {
    failed = 1;
    goto done;
  }
  signal_operation.timeout_ns = UINT64_C(1000000);
  memset(&signal_result, 0xa5, sizeof(signal_result));
  status = sagr_signal_wait(signal, SAGR_SIGNAL_CONDITION_EQ, INT64_C(0),
                            &signal_operation, &signal_result,
                            (uint32_t)sizeof(signal_result), &error,
                            (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_TIMED_OUT || signal_result.signal_id != 0U) {
    failed = 1;
    goto done;
  }
  if (sagr_pinned_dispatch_options_init(
          &dispatch_options, (uint32_t)sizeof(dispatch_options)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_queue_submit_pinned_dispatch(
          queue, input, output, signal, &dispatch_options, NULL, &ticket,
          (uint32_t)sizeof(ticket), &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS) {
    failed = 1;
    goto done;
  }
  if (sagr_code_object_upload(instance, image, image_size, "gpuReadWrite", NULL,
                              &remote, (uint32_t)sizeof(remote), &error,
                              (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      remote.object_id != TEST_OBJECT_ID ||
      remote.generation != TEST_OBJECT_GENERATION) {
    failed = 1;
    goto done;
  }
  memset(&completion, 0, sizeof(completion));
  if (sagr_queue_wait_pinned_dispatch(
          queue, &ticket, NULL, &completion, (uint32_t)sizeof(completion),
          &error, (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      completion.request_id != ticket.request_id ||
      completion.trace_id != TEST_TRACE_ID ||
      completion.output_crc32c != SAGR_DISPATCH_OUTPUT_CRC32C ||
      completion.retire_tick != TEST_RETIRE_TICK) {
    failed = 1;
    goto done;
  }
  memset(&signal_result, 0, sizeof(signal_result));
  if (sagr_signal_wait(signal, SAGR_SIGNAL_CONDITION_EQ, INT64_C(0), NULL,
                       &signal_result, (uint32_t)sizeof(signal_result), &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      signal_result.signal_id != TEST_SIGNAL_ID ||
      signal_result.observed_value != 0 ||
      signal_result.completion_tick != TEST_RETIRE_TICK + UINT64_C(1)) {
    failed = 1;
  }

done:
  (void)sagr_instance_close(&instance);
  failed |= finish_server(&server);
  return failed;
}

static int
test_canonical_rejection_is_determinate(const uint8_t *image,
                                        size_t image_size)
{
  upload_server_t server;
  sagr_instance_t instance = NULL;
  sagr_code_object_remote_info_t remote;
  sagr_error_info_t error;
  int failed = 0;
  if (start_server(&server, UPLOAD_SERVER_CANONICAL_REJECTION) != 0 ||
      open_instance(&server, 0, &instance) != 0) {
    return 1;
  }
  if (sagr_code_object_upload(instance, image, image_size, "gpuReadWrite", NULL,
                              &remote, (uint32_t)sizeof(remote), &error,
                              (uint32_t)sizeof(error)) !=
          SAGR_STATUS_PROTOCOL_ERROR ||
      error.wire_status != SAGR_WIRE_STATUS_MALFORMED ||
      remote.object_id != 0U ||
      sagr_code_object_upload(instance, image, image_size, "gpuReadWrite", NULL,
                              &remote, (uint32_t)sizeof(remote), &error,
                              (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      remote.object_id != TEST_OBJECT_ID) {
    fprintf(stderr,
            "code-object upload test: canonical rejection was not determinate\n");
    failed = 1;
  }
  (void)sagr_instance_close(&instance);
  failed |= finish_server(&server);
  return failed;
}

static int
test_poisoning_failure(const uint8_t *image, size_t image_size,
                       enum upload_server_behavior behavior,
                       sagr_status_t expected_status)
{
  upload_server_t server;
  sagr_instance_t instance = NULL;
  sagr_code_object_remote_info_t remote;
  sagr_queue_operation_options_t operation;
  sagr_error_info_t error;
  sagr_status_t status;
  int cancel_pipe[2] = {-1, -1};
  int failed = 0;
  if (behavior == UPLOAD_SERVER_ACK_CANCEL &&
      pipe2(cancel_pipe, O_CLOEXEC) != 0) {
    return 1;
  }
  if ((behavior == UPLOAD_SERVER_ACK_CANCEL
           ? start_server_with_cancel(&server, behavior, cancel_pipe[1])
           : start_server(&server, behavior)) != 0 ||
      open_instance(&server, 0, &instance) != 0) {
    if (cancel_pipe[0] >= 0) {
      (void)close(cancel_pipe[0]);
      (void)close(cancel_pipe[1]);
    }
    return 1;
  }
  if (behavior == UPLOAD_SERVER_PRE_SEND_EXHAUSTION) {
    test_instance_prefix_t *prefix = (test_instance_prefix_t *)instance;
    prefix->next_request_id = UINT64_MAX;
  }
  (void)sagr_queue_operation_options_init(&operation,
                                          (uint32_t)sizeof(operation));
  if (behavior == UPLOAD_SERVER_ACK_TIMEOUT) {
    operation.timeout_ns = UINT64_C(50000000);
  } else if (behavior == UPLOAD_SERVER_ACK_CANCEL) {
    operation.cancel_fd = cancel_pipe[0];
  }
  memset(&remote, 0xa5, sizeof(remote));
  status = sagr_code_object_upload(
      instance, image, image_size, "gpuReadWrite",
      (behavior == UPLOAD_SERVER_ACK_TIMEOUT ||
       behavior == UPLOAD_SERVER_ACK_CANCEL)
          ? &operation
          : NULL,
      &remote,
      (uint32_t)sizeof(remote), &error, (uint32_t)sizeof(error));
  if (status != expected_status || remote.object_id != 0U ||
      remote.generation != 0U ||
      sagr_code_object_upload(instance, image, image_size, "gpuReadWrite", NULL,
                              &remote, (uint32_t)sizeof(remote), &error,
                              (uint32_t)sizeof(error)) !=
          SAGR_STATUS_CONNECTION_LOST) {
    fprintf(stderr,
            "code-object upload test: failure did not poison transport (%d)\n",
            (int)behavior);
    failed = 1;
  }
  (void)sagr_instance_close(&instance);
  failed |= finish_server(&server);
  if (cancel_pipe[0] >= 0) {
    (void)close(cancel_pipe[0]);
    (void)close(cancel_pipe[1]);
  }
  if (server.saw_disconnect == 0) {
    fprintf(stderr,
            "code-object upload test: server did not observe disconnect (%d)\n",
            (int)behavior);
    failed = 1;
  }
  if (server.cleanup_observed == 0) {
    fprintf(stderr,
            "code-object upload test: disconnect did not clean staging (%d)\n",
            (int)behavior);
    failed = 1;
  }
  return failed;
}

int
main(void)
{
  uint8_t *image = NULL;
  size_t image_size = 0U;
  int failures = 0;
  if (read_fixture(&image, &image_size) != 0) {
    fprintf(stderr, "code-object upload test: fixture is unavailable\n");
    return 1;
  }
  failures += test_interleaved_upload(image, image_size);
  failures += test_interleaved_dispatch_upload(image, image_size);
  failures += test_canonical_rejection_is_determinate(image, image_size);
  failures += test_poisoning_failure(image, image_size,
                                     UPLOAD_SERVER_MALFORMED_REJECTION,
                                     SAGR_STATUS_PROTOCOL_ERROR);
  failures += test_poisoning_failure(image, image_size,
                                     UPLOAD_SERVER_WRONG_OPCODE_REJECTION,
                                     SAGR_STATUS_PROTOCOL_ERROR);
  failures += test_poisoning_failure(image, image_size,
                                     UPLOAD_SERVER_ACK_TIMEOUT,
                                     SAGR_STATUS_TIMED_OUT);
  failures += test_poisoning_failure(image, image_size,
                                     UPLOAD_SERVER_ACK_CANCEL,
                                     SAGR_STATUS_CANCELLED);
  failures += test_poisoning_failure(image, image_size,
                                     UPLOAD_SERVER_PRE_SEND_EXHAUSTION,
                                     SAGR_STATUS_OUT_OF_RESOURCES);
  free(image);
  return failures == 0 ? 0 : 1;
}
