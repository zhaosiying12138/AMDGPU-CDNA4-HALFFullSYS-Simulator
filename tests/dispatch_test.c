/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <self_amdgpu_runtime/runtime.h>

#include "transport_internal.h"

static const uint8_t k_daemon_uuid[16] = {
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
static const uint8_t k_job_uuid[16] = {
    0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87,
    0x98, 0xa9, 0xba, 0xcb, 0xdc, 0xed, 0xfe, 0x0f};
static const uint8_t k_server_nonce[16] = {
    0xf0, 0xe0, 0xd0, 0xc0, 0xb0, 0xa0, 0x90, 0x80,
    0x70, 0x60, 0x50, 0x40, 0x30, 0x20, 0x10, 0x01};

#define MOCK_QUEUE_ID UINT64_C(0x1020304050607080)
#define MOCK_QUEUE_GENERATION UINT64_C(0x8877665544332211)
#define MOCK_INPUT_ID UINT64_C(7)
#define MOCK_INPUT_GENERATION UINT64_C(0x1000000000000001)
#define MOCK_INPUT_VA UINT64_C(0x0000100300000000)
#define MOCK_OUTPUT_ID UINT64_C(8)
#define MOCK_OUTPUT_GENERATION UINT64_C(0x1000000000000002)
#define MOCK_OUTPUT_VA UINT64_C(0x0000100380000000)
#define MOCK_SIGNAL_ID UINT64_C(9)
#define MOCK_SIGNAL_GENERATION UINT64_C(0x2000000000000001)
#define MOCK_TRACE_ID UINT64_C(0x5452434500000001)
#define MOCK_ADMISSION_TICK UINT64_C(100)
#define MOCK_START_TICK UINT64_C(110)
#define MOCK_END_TICK UINT64_C(120)
#define MOCK_RETIRE_TICK UINT64_C(130)

enum dispatch_server_behavior {
  DISPATCH_SERVER_SUCCESS,
  DISPATCH_SERVER_REJECT_ONCE,
  DISPATCH_SERVER_BAD_ACK_PACKET_CRC,
  DISPATCH_SERVER_BAD_SIGNAL_TICK,
  DISPATCH_SERVER_COMPLETION_BEFORE_SIGNAL
};

typedef struct dispatch_server {
  char directory[128];
  char endpoint[160];
  int listener;
  pthread_t thread;
  enum dispatch_server_behavior behavior;
  int thread_error;
  uint32_t dispatch_requests;
  uint32_t signal_wait_requests;
} dispatch_server_t;

typedef struct mock_request {
  uint16_t message_type;
  uint64_t request_id;
  int descriptor;
  sagr_wire_queue_request_t queue;
  sagr_wire_memory_request_t memory;
  sagr_wire_signal_request_t signal;
  sagr_wire_dispatch_request_t dispatch;
} mock_request_t;

typedef struct mock_state {
  sagr_instance_info_t info;
  uint64_t last_request_id;
  uint64_t signal_wait_request_id;
  uint64_t signal_wait_sequence;
  uint64_t signal_wait_ack_tick;
  uint64_t signal_value_bits;
  uint32_t allocations;
  uint32_t live_allocations;
  uint8_t input_bytes[64];
  uint8_t output_bytes[64];
  int queue_live;
  int signal_live;
  int signal_wait_pending;
} mock_state_t;

typedef struct dispatch_resources {
  sagr_instance_t instance;
  sagr_queue_t queue;
  sagr_memory_t input;
  sagr_memory_t output;
  sagr_signal_t signal;
} dispatch_resources_t;

static uint16_t get_u16(const uint8_t *source) {
  return (uint16_t)(((uint16_t)source[0] << 8) | source[1]);
}

static uint32_t get_u32(const uint8_t *source) {
  return ((uint32_t)source[0] << 24) | ((uint32_t)source[1] << 16) |
         ((uint32_t)source[2] << 8) | source[3];
}

static uint64_t get_u64(const uint8_t *source) {
  return ((uint64_t)get_u32(source) << 32) | get_u32(source + 4);
}

static int bytes_are_zero(const uint8_t *bytes, size_t size) {
  uint8_t combined = 0;
  size_t index;
  for (index = 0; index < size; ++index) {
    combined = (uint8_t)(combined | bytes[index]);
  }
  return combined == 0;
}

static int send_frame(int peer, const uint8_t *frame, size_t frame_size) {
  return send(peer, frame, frame_size, MSG_NOSIGNAL) == (ssize_t)frame_size
             ? 0
             : -1;
}

static int read_all_at(int descriptor, uint8_t *bytes, size_t size) {
  size_t offset = 0;
  while (offset < size) {
    const ssize_t count =
        pread(descriptor, bytes + offset, size - offset, (off_t)offset);
    if (count > 0) {
      offset += (size_t)count;
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      return -1;
    }
  }
  return 0;
}

static int write_all_at(int descriptor, const uint8_t *bytes, size_t size) {
  size_t offset = 0;
  while (offset < size) {
    const ssize_t count =
        pwrite(descriptor, bytes + offset, size - offset, (off_t)offset);
    if (count > 0) {
      offset += (size_t)count;
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      return -1;
    }
  }
  return 0;
}

static int send_handshake_ack(int peer, const uint8_t *hello,
                              ssize_t hello_size, mock_state_t *state) {
  const uint64_t capabilities =
      SAGR_CAPABILITY_TOPOLOGY_MASK | SAGR_CAPABILITY_QUEUE_MASK |
      SAGR_CAPABILITY_MEMORY_MASK | SAGR_CAPABILITY_SIGNAL_MASK |
      SAGR_CAPABILITY_DISPATCH_MASK;
  sagr_wire_ack_fields_t fields;
  uint8_t frame[SAGR_WIRE_ACK_FRAME_BYTES];
  size_t frame_size = 0;
  if (hello_size < (ssize_t)(SAGR_WIRE_HEADER_BYTES +
                             SAGR_WIRE_HELLO_FIXED_BYTES)) {
    return -1;
  }
  memset(&fields, 0, sizeof(fields));
  fields.selected_major = 1;
  fields.status = SAGR_WIRE_STATUS_OK;
  memcpy(fields.client_nonce, hello + SAGR_WIRE_HEADER_BYTES + 8, 16);
  memcpy(fields.server_nonce, k_server_nonce, 16);
  fields.selected_capabilities[0] = capabilities;
  fields.maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  fields.request_id = get_u64(hello + 24);
  memcpy(fields.daemon_uuid, k_daemon_uuid, 16);
  fields.connection_id = UINT64_C(0x1122334455667788);
  fields.epoch = UINT64_C(0x0102030405060708);
  memcpy(fields.job_uuid, k_job_uuid, 16);
  fields.rank = 3;
  fields.world_size = 8;
  fields.include_topology = 1;
  if (fields.request_id == 0 ||
      sagr_protocol_encode_ack(&fields, frame, sizeof(frame), &frame_size) !=
          SAGR_STATUS_SUCCESS ||
      send_frame(peer, frame, frame_size) != 0) {
    return -1;
  }
  memset(&state->info, 0, sizeof(state->info));
  state->info.struct_size = (uint32_t)sizeof(state->info);
  state->info.maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  state->info.negotiated_capabilities[0] = capabilities;
  memcpy(state->info.daemon_uuid, k_daemon_uuid, 16);
  state->info.connection_id = fields.connection_id;
  state->info.epoch = fields.epoch;
  state->last_request_id = fields.request_id;
  return 0;
}

static int receive_request(int peer, mock_state_t *state,
                           mock_request_t *request) {
  static const uint8_t magic[8] = {'G', 'S', 'I', 'M', 'R', 'P', 'C', 0};
  uint8_t frame[SAGR_WIRE_DISPATCH_RESULT_FRAME_BYTES];
  uint8_t crc_frame[SAGR_WIRE_DISPATCH_RESULT_FRAME_BYTES];
  unsigned char control[CMSG_SPACE(sizeof(int))];
  struct iovec vector;
  struct msghdr message;
  struct cmsghdr *control_message;
  const uint8_t *payload = frame + SAGR_WIRE_HEADER_BYTES;
  size_t expected_size;
  uint32_t expected_payload;
  ssize_t received;

  memset(request, 0, sizeof(*request));
  request->descriptor = -1;
  memset(&message, 0, sizeof(message));
  memset(control, 0, sizeof(control));
  vector.iov_base = frame;
  vector.iov_len = sizeof(frame);
  message.msg_iov = &vector;
  message.msg_iovlen = 1;
  message.msg_control = control;
  message.msg_controllen = sizeof(control);
  do {
    received = recvmsg(peer, &message, MSG_CMSG_CLOEXEC);
  } while (received < 0 && errno == EINTR);
  if (received == 0) {
    return 1;
  }
  if (received < 0 || (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0) {
    return -1;
  }
  for (control_message = CMSG_FIRSTHDR(&message); control_message != NULL;
       control_message = CMSG_NXTHDR(&message, control_message)) {
    int descriptor = -1;
    if (control_message->cmsg_level != SOL_SOCKET ||
        control_message->cmsg_type != SCM_RIGHTS ||
        control_message->cmsg_len != CMSG_LEN(sizeof(descriptor)) ||
        request->descriptor >= 0) {
      if (request->descriptor >= 0) {
        (void)close(request->descriptor);
        request->descriptor = -1;
      }
      return -1;
    }
    memcpy(&descriptor, CMSG_DATA(control_message), sizeof(descriptor));
    request->descriptor = descriptor;
  }
  request->message_type = get_u16(frame + 14);
  request->request_id = get_u64(frame + 24);
  if (request->message_type == SAGR_WIRE_MESSAGE_DISPATCH_REQUEST) {
    expected_size = SAGR_WIRE_DISPATCH_REQUEST_FRAME_BYTES;
    expected_payload = SAGR_WIRE_DISPATCH_REQUEST_PAYLOAD_BYTES;
  } else {
    expected_size = SAGR_WIRE_QUEUE_FRAME_BYTES;
    expected_payload = SAGR_WIRE_QUEUE_PAYLOAD_BYTES;
  }
  if ((size_t)received != expected_size ||
      memcmp(frame, magic, sizeof(magic)) != 0 || get_u16(frame + 8) != 1 ||
      get_u16(frame + 10) != 0 ||
      get_u16(frame + 12) != SAGR_WIRE_HEADER_BYTES ||
      get_u32(frame + 16) != 0 || get_u32(frame + 20) != expected_payload ||
      request->request_id == 0 ||
      request->request_id <= state->last_request_id ||
      memcmp(frame + 32, state->info.daemon_uuid, 16) != 0 ||
      get_u64(frame + 48) != state->info.connection_id ||
      get_u64(frame + 56) != state->info.epoch ||
      get_u32(frame + 68) != 0 || get_u64(frame + 72) != 0) {
    if (request->descriptor >= 0) {
      (void)close(request->descriptor);
      request->descriptor = -1;
    }
    return -1;
  }
  memcpy(crc_frame, frame, (size_t)received);
  memset(crc_frame + 64, 0, 4);
  if (sagr_crc32c(crc_frame, (size_t)received) != get_u32(frame + 64)) {
    if (request->descriptor >= 0) {
      (void)close(request->descriptor);
      request->descriptor = -1;
    }
    return -1;
  }
  state->last_request_id = request->request_id;
  if (request->message_type == SAGR_WIRE_MESSAGE_QUEUE_REQUEST) {
    if (get_u16(payload) != SAGR_QUEUE_PROTOCOL_MAJOR ||
        get_u16(payload + 2) != SAGR_QUEUE_PROTOCOL_MINOR ||
        get_u16(payload + 6) != 0 || !bytes_are_zero(payload + 48, 16)) {
      return -1;
    }
    request->queue.major = get_u16(payload);
    request->queue.minor = get_u16(payload + 2);
    request->queue.opcode = get_u16(payload + 4);
    request->queue.flags = get_u16(payload + 6);
    request->queue.queue_id = get_u64(payload + 8);
    request->queue.generation = get_u64(payload + 16);
    request->queue.sequence = get_u64(payload + 24);
    request->queue.arg0 = get_u64(payload + 32);
    request->queue.arg1 = get_u64(payload + 40);
  } else if (request->message_type == SAGR_WIRE_MESSAGE_MEMORY_REQUEST) {
    if (get_u16(payload) != SAGR_MEMORY_PROTOCOL_MAJOR ||
        get_u16(payload + 2) != SAGR_MEMORY_PROTOCOL_MINOR ||
        get_u16(payload + 6) != 0 || !bytes_are_zero(payload + 48, 16)) {
      return -1;
    }
    request->memory.major = get_u16(payload);
    request->memory.minor = get_u16(payload + 2);
    request->memory.opcode = get_u16(payload + 4);
    request->memory.flags = get_u16(payload + 6);
    request->memory.allocation_id = get_u64(payload + 8);
    request->memory.generation = get_u64(payload + 16);
    request->memory.offset = get_u64(payload + 24);
    request->memory.byte_count = get_u64(payload + 32);
    request->memory.argument = get_u64(payload + 40);
  } else if (request->message_type == SAGR_WIRE_MESSAGE_SIGNAL_REQUEST) {
    if (get_u16(payload) != SAGR_SIGNAL_PROTOCOL_MAJOR ||
        get_u16(payload + 2) != SAGR_SIGNAL_PROTOCOL_MINOR ||
        get_u16(payload + 6) != 0 || !bytes_are_zero(payload + 48, 16)) {
      return -1;
    }
    request->signal.major = get_u16(payload);
    request->signal.minor = get_u16(payload + 2);
    request->signal.opcode = get_u16(payload + 4);
    request->signal.flags = get_u16(payload + 6);
    request->signal.signal_id = get_u64(payload + 8);
    request->signal.generation = get_u64(payload + 16);
    request->signal.sequence = get_u64(payload + 24);
    request->signal.value_bits = get_u64(payload + 32);
    request->signal.condition = get_u64(payload + 40);
  } else if (request->message_type == SAGR_WIRE_MESSAGE_DISPATCH_REQUEST) {
    if (get_u16(payload) != SAGR_DISPATCH_PROTOCOL_MAJOR ||
        get_u16(payload + 2) != SAGR_DISPATCH_PROTOCOL_MINOR ||
        get_u16(payload + 6) != 0) {
      return -1;
    }
    request->dispatch.major = get_u16(payload);
    request->dispatch.minor = get_u16(payload + 2);
    request->dispatch.opcode = get_u16(payload + 4);
    request->dispatch.flags = get_u16(payload + 6);
    request->dispatch.queue_id = get_u64(payload + 8);
    request->dispatch.queue_generation = get_u64(payload + 16);
    request->dispatch.queue_sequence = get_u64(payload + 24);
    request->dispatch.fixture_id = get_u64(payload + 32);
    request->dispatch.input_allocation_id = get_u64(payload + 40);
    request->dispatch.input_generation = get_u64(payload + 48);
    request->dispatch.output_allocation_id = get_u64(payload + 56);
    request->dispatch.output_generation = get_u64(payload + 64);
    request->dispatch.signal_id = get_u64(payload + 72);
    request->dispatch.signal_generation = get_u64(payload + 80);
    request->dispatch.expected_signal_value_bits = get_u64(payload + 88);
    memcpy(request->dispatch.fixture_manifest_sha256, payload + 96, 32);
  } else {
    return -1;
  }
  return 0;
}

static int send_queue_ack(int peer, mock_state_t *state,
                          const mock_request_t *request, uint64_t queue_id,
                          uint64_t generation, uint64_t value) {
  sagr_wire_queue_response_t response;
  uint8_t frame[SAGR_WIRE_QUEUE_FRAME_BYTES];
  size_t frame_size = 0;
  memset(&response, 0, sizeof(response));
  response.major = SAGR_QUEUE_PROTOCOL_MAJOR;
  response.minor = SAGR_QUEUE_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = request->queue.opcode;
  response.queue_id = queue_id;
  response.generation = generation;
  response.sequence = request->queue.sequence;
  response.value = value;
  response.sim_tick = request->request_id + UINT64_C(10);
  if (sagr_protocol_encode_queue_response(
          &state->info, request->request_id, SAGR_WIRE_MESSAGE_QUEUE_ACK,
          &response, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static int send_memory_ack(int peer, mock_state_t *state,
                           const mock_request_t *request,
                           uint64_t allocation_id, uint64_t generation,
                           uint64_t value0, uint64_t value1,
                           uint64_t value2) {
  sagr_wire_memory_response_t response;
  uint8_t frame[SAGR_WIRE_MEMORY_FRAME_BYTES];
  size_t frame_size = 0;
  memset(&response, 0, sizeof(response));
  response.major = SAGR_MEMORY_PROTOCOL_MAJOR;
  response.minor = SAGR_MEMORY_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = request->memory.opcode;
  response.allocation_id = allocation_id;
  response.generation = generation;
  response.value0 = value0;
  response.value1 = value1;
  response.value2 = value2;
  response.sim_tick = request->request_id + UINT64_C(10);
  if (sagr_protocol_encode_memory_response(
          &state->info, request->request_id, &response, frame, sizeof(frame),
          &frame_size) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static int send_signal_record(int peer, mock_state_t *state,
                              uint64_t request_id, uint16_t message_type,
                              uint16_t opcode, uint64_t sequence,
                              uint64_t value_bits, uint64_t ready,
                              uint64_t sim_tick) {
  sagr_wire_signal_response_t response;
  uint8_t frame[SAGR_WIRE_SIGNAL_FRAME_BYTES];
  size_t frame_size = 0;
  memset(&response, 0, sizeof(response));
  response.major = SAGR_SIGNAL_PROTOCOL_MAJOR;
  response.minor = SAGR_SIGNAL_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = opcode;
  response.signal_id = opcode == SAGR_WIRE_SIGNAL_OPCODE_CREATE
                           ? MOCK_SIGNAL_ID
                           : MOCK_SIGNAL_ID;
  response.generation = MOCK_SIGNAL_GENERATION;
  response.sequence = sequence;
  response.value_bits = value_bits;
  response.ready = ready;
  response.sim_tick = sim_tick;
  if (sagr_protocol_encode_signal_response(
          &state->info, request_id, message_type, &response, frame,
          sizeof(frame), &frame_size) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static void initialize_dispatch_result(
    sagr_wire_dispatch_response_t *response,
    const sagr_wire_dispatch_request_t *request, uint32_t status) {
  memset(response, 0, sizeof(*response));
  response->major = SAGR_DISPATCH_PROTOCOL_MAJOR;
  response->minor = SAGR_DISPATCH_PROTOCOL_MINOR;
  response->status = status;
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
}

static int send_dispatch_record(
    int peer, mock_state_t *state, uint64_t request_id, uint16_t message_type,
    const sagr_wire_dispatch_response_t *response) {
  uint8_t frame[SAGR_WIRE_DISPATCH_RESULT_FRAME_BYTES];
  size_t frame_size = 0;
  if (sagr_protocol_encode_dispatch_response(
          &state->info, request_id, message_type, response, frame,
          sizeof(frame), &frame_size) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static int dispatch_request_is_expected(const mock_state_t *state,
                                        const mock_request_t *received) {
  const sagr_wire_dispatch_request_t *request = &received->dispatch;
  return state->queue_live != 0 && state->live_allocations == 2 &&
         state->signal_live != 0 && state->signal_wait_pending != 0 &&
         state->signal_value_bits == UINT64_C(1) &&
         request->opcode == SAGR_WIRE_DISPATCH_OPCODE_SUBMIT_PINNED &&
         request->queue_id == MOCK_QUEUE_ID &&
         request->queue_generation == MOCK_QUEUE_GENERATION &&
         request->queue_sequence == UINT64_C(1) &&
         request->fixture_id == SAGR_DISPATCH_FIXTURE_GFX950_XOR_U8_V1 &&
         request->input_allocation_id == MOCK_INPUT_ID &&
         request->input_generation == MOCK_INPUT_GENERATION &&
         request->output_allocation_id == MOCK_OUTPUT_ID &&
         request->output_generation == MOCK_OUTPUT_GENERATION &&
         request->signal_id == MOCK_SIGNAL_ID &&
         request->signal_generation == MOCK_SIGNAL_GENERATION &&
         request->expected_signal_value_bits == UINT64_C(0) &&
         memcmp(request->fixture_manifest_sha256,
                sagr_dispatch_fixture_manifest_sha256, 32) == 0;
}

static int handle_dispatch_request(dispatch_server_t *server, int peer,
                                   mock_state_t *state,
                                   const mock_request_t *received) {
  sagr_wire_dispatch_response_t response;
  sagr_wire_dispatch_response_t completion;
  uint64_t signal_tick = MOCK_RETIRE_TICK + UINT64_C(1);
  if (!dispatch_request_is_expected(state, received)) {
    return -1;
  }
  {
    size_t index;
    for (index = 0; index < sizeof(state->output_bytes); ++index) {
      state->output_bytes[index] =
          (uint8_t)(state->input_bytes[index] ^ UINT8_C(0x5a));
    }
  }
  ++server->dispatch_requests;
  initialize_dispatch_result(&response, &received->dispatch,
                             SAGR_WIRE_STATUS_OK);
  if (server->behavior == DISPATCH_SERVER_REJECT_ONCE &&
      server->dispatch_requests == 1U) {
    response.status = SAGR_WIRE_STATUS_BUSY;
    return send_dispatch_record(peer, state, received->request_id,
                                SAGR_WIRE_MESSAGE_DISPATCH_ACK, &response);
  }
  response.trace_id = MOCK_TRACE_ID;
  response.input_gpu_va = MOCK_INPUT_VA;
  response.output_gpu_va = MOCK_OUTPUT_VA;
  response.packet_crc32c = SAGR_DISPATCH_PACKET_CRC32C;
  response.admission_tick = MOCK_ADMISSION_TICK;
  if (server->behavior == DISPATCH_SERVER_BAD_ACK_PACKET_CRC) {
    response.packet_crc32c ^= UINT32_C(1);
  }
  if (send_dispatch_record(peer, state, received->request_id,
                           SAGR_WIRE_MESSAGE_DISPATCH_ACK, &response) != 0) {
    return -1;
  }
  if (server->behavior == DISPATCH_SERVER_BAD_ACK_PACKET_CRC) {
    return 0;
  }
  initialize_dispatch_result(&completion, &received->dispatch,
                             SAGR_WIRE_STATUS_OK);
  completion.trace_id = response.trace_id;
  completion.input_gpu_va = response.input_gpu_va;
  completion.output_gpu_va = response.output_gpu_va;
  completion.packet_crc32c = response.packet_crc32c;
  completion.output_crc32c = SAGR_DISPATCH_OUTPUT_CRC32C;
  completion.admission_tick = MOCK_ADMISSION_TICK;
  completion.start_tick = MOCK_START_TICK;
  completion.end_tick = MOCK_END_TICK;
  completion.retire_tick = MOCK_RETIRE_TICK;
  if (server->behavior == DISPATCH_SERVER_SUCCESS) {
    const struct timespec delay = {0, 20000000};
    (void)nanosleep(&delay, NULL);
  }
  if (server->behavior == DISPATCH_SERVER_COMPLETION_BEFORE_SIGNAL) {
    return send_dispatch_record(peer, state, received->request_id,
                                SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION,
                                &completion);
  }
  if (server->behavior == DISPATCH_SERVER_BAD_SIGNAL_TICK) {
    ++signal_tick;
  }
  if (send_signal_record(peer, state, state->signal_wait_request_id,
                         SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION,
                         SAGR_WIRE_SIGNAL_OPCODE_WAIT,
                         state->signal_wait_sequence, UINT64_C(0), UINT64_C(0),
                         signal_tick) != 0 ||
      send_dispatch_record(peer, state, received->request_id,
                           SAGR_WIRE_MESSAGE_DISPATCH_COMPLETION,
                           &completion) != 0) {
    return -1;
  }
  state->signal_value_bits = 0;
  return 0;
}

static int handle_request(dispatch_server_t *server, int peer,
                          mock_state_t *state,
                          mock_request_t *request) {
  if (request->message_type == SAGR_WIRE_MESSAGE_QUEUE_REQUEST) {
    if (request->descriptor >= 0) {
      return -1;
    }
    if (request->queue.opcode == SAGR_WIRE_QUEUE_OPCODE_CREATE &&
        state->queue_live == 0 && request->queue.queue_id == 0 &&
        request->queue.generation == 0 && request->queue.sequence == 0 &&
        request->queue.arg0 == UINT64_C(1) && request->queue.arg1 == 0) {
      state->queue_live = 1;
      return send_queue_ack(peer, state, request, MOCK_QUEUE_ID,
                            MOCK_QUEUE_GENERATION, UINT64_C(1));
    }
    if (request->queue.opcode == SAGR_WIRE_QUEUE_OPCODE_DESTROY &&
        state->queue_live != 0 && request->queue.queue_id == MOCK_QUEUE_ID &&
        request->queue.generation == MOCK_QUEUE_GENERATION &&
        request->queue.sequence == 0 && request->queue.arg0 == 0 &&
        request->queue.arg1 == 0) {
      state->queue_live = 0;
      return send_queue_ack(peer, state, request, MOCK_QUEUE_ID,
                            MOCK_QUEUE_GENERATION, 0);
    }
    return -1;
  }
  if (request->message_type == SAGR_WIRE_MESSAGE_MEMORY_REQUEST) {
    if (request->memory.opcode == SAGR_WIRE_MEMORY_OPCODE_ALLOC &&
        request->descriptor < 0 &&
        request->memory.allocation_id == 0 && request->memory.generation == 0 &&
        request->memory.offset == 0 &&
        request->memory.byte_count == SAGR_DISPATCH_FIXED_IO_BYTES &&
        request->memory.argument == SAGR_MEMORY_ALIGNMENT_4K &&
        state->allocations < 2U) {
      const uint64_t allocation_id = state->allocations == 0U
                                         ? MOCK_INPUT_ID
                                         : MOCK_OUTPUT_ID;
      const uint64_t generation = state->allocations == 0U
                                      ? MOCK_INPUT_GENERATION
                                      : MOCK_OUTPUT_GENERATION;
      const uint64_t simulated_va = state->allocations == 0U
                                        ? MOCK_INPUT_VA
                                        : MOCK_OUTPUT_VA;
      ++state->allocations;
      ++state->live_allocations;
      return send_memory_ack(peer, state, request, allocation_id, generation,
                             simulated_va, SAGR_DISPATCH_FIXED_IO_BYTES,
                             SAGR_MEMORY_ALIGNMENT_4K);
    }
    if (request->memory.opcode == SAGR_WIRE_MEMORY_OPCODE_FREE &&
        request->descriptor < 0 &&
        request->memory.offset == 0 && request->memory.byte_count == 0 &&
        request->memory.argument == 0 && state->live_allocations != 0 &&
        ((request->memory.allocation_id == MOCK_INPUT_ID &&
          request->memory.generation == MOCK_INPUT_GENERATION) ||
         (request->memory.allocation_id == MOCK_OUTPUT_ID &&
          request->memory.generation == MOCK_OUTPUT_GENERATION))) {
      --state->live_allocations;
      return send_memory_ack(peer, state, request,
                             request->memory.allocation_id,
                             request->memory.generation, 0, 0, 0);
    }
    if ((request->memory.opcode == SAGR_WIRE_MEMORY_OPCODE_COPY_H2D ||
         request->memory.opcode == SAGR_WIRE_MEMORY_OPCODE_COPY_D2H) &&
        request->descriptor >= 0 && request->memory.offset == 0 &&
        request->memory.byte_count == SAGR_DISPATCH_FIXED_IO_BYTES) {
      uint8_t *bytes = NULL;
      uint32_t crc;
      int result;
      if (request->memory.allocation_id == MOCK_INPUT_ID &&
          request->memory.generation == MOCK_INPUT_GENERATION) {
        bytes = state->input_bytes;
      } else if (request->memory.allocation_id == MOCK_OUTPUT_ID &&
                 request->memory.generation == MOCK_OUTPUT_GENERATION) {
        bytes = state->output_bytes;
      }
      if (bytes == NULL) {
        return -1;
      }
      if (request->memory.opcode == SAGR_WIRE_MEMORY_OPCODE_COPY_H2D) {
        if (request->memory.argument > UINT32_MAX ||
            read_all_at(request->descriptor, bytes, 64) != 0) {
          return -1;
        }
        crc = sagr_crc32c(bytes, 64);
        if (crc != (uint32_t)request->memory.argument) {
          return -1;
        }
      } else {
        if (request->memory.argument != 0 ||
            write_all_at(request->descriptor, bytes, 64) != 0 ||
            fcntl(request->descriptor, F_ADD_SEALS,
                  F_SEAL_WRITE | F_SEAL_SEAL) != 0) {
          return -1;
        }
        crc = sagr_crc32c(bytes, 64);
      }
      result = send_memory_ack(
          peer, state, request, request->memory.allocation_id,
          request->memory.generation, 0, SAGR_DISPATCH_FIXED_IO_BYTES, crc);
      (void)close(request->descriptor);
      request->descriptor = -1;
      return result;
    }
    return -1;
  }
  if (request->message_type == SAGR_WIRE_MESSAGE_SIGNAL_REQUEST) {
    if (request->descriptor >= 0) {
      return -1;
    }
    if (request->signal.opcode == SAGR_WIRE_SIGNAL_OPCODE_CREATE &&
        state->signal_live == 0 && request->signal.signal_id == 0 &&
        request->signal.generation == 0 && request->signal.sequence == 0 &&
        request->signal.condition == 0 && request->signal.value_bits == 1) {
      state->signal_live = 1;
      state->signal_value_bits = 1;
      return send_signal_record(peer, state, request->request_id,
                                SAGR_WIRE_MESSAGE_SIGNAL_ACK,
                                SAGR_WIRE_SIGNAL_OPCODE_CREATE, 0,
                                state->signal_value_bits, 0,
                                request->request_id + UINT64_C(10));
    }
    if (request->signal.opcode == SAGR_WIRE_SIGNAL_OPCODE_WAIT &&
        state->signal_live != 0 && state->signal_wait_pending == 0 &&
        request->signal.signal_id == MOCK_SIGNAL_ID &&
        request->signal.generation == MOCK_SIGNAL_GENERATION &&
        request->signal.sequence == UINT64_C(1) &&
        request->signal.condition == SAGR_SIGNAL_CONDITION_EQ &&
        request->signal.value_bits == 0) {
      ++server->signal_wait_requests;
      state->signal_wait_pending = 1;
      state->signal_wait_request_id = request->request_id;
      state->signal_wait_sequence = request->signal.sequence;
      state->signal_wait_ack_tick = UINT64_C(80);
      return send_signal_record(peer, state, request->request_id,
                                SAGR_WIRE_MESSAGE_SIGNAL_ACK,
                                SAGR_WIRE_SIGNAL_OPCODE_WAIT,
                                request->signal.sequence,
                                state->signal_value_bits, 0,
                                state->signal_wait_ack_tick);
    }
    if (request->signal.opcode == SAGR_WIRE_SIGNAL_OPCODE_DESTROY &&
        state->signal_live != 0 && request->signal.signal_id == MOCK_SIGNAL_ID &&
        request->signal.generation == MOCK_SIGNAL_GENERATION &&
        request->signal.sequence == 0 && request->signal.value_bits == 0 &&
        request->signal.condition == 0) {
      state->signal_live = 0;
      state->signal_wait_pending = 0;
      return send_signal_record(peer, state, request->request_id,
                                SAGR_WIRE_MESSAGE_SIGNAL_ACK,
                                SAGR_WIRE_SIGNAL_OPCODE_DESTROY, 0, 0, 0,
                                request->request_id + UINT64_C(10));
    }
    return -1;
  }
  if (request->message_type == SAGR_WIRE_MESSAGE_DISPATCH_REQUEST) {
    if (request->descriptor >= 0) {
      return -1;
    }
    return handle_dispatch_request(server, peer, state, request);
  }
  return -1;
}

static void *dispatch_server_main(void *opaque) {
  dispatch_server_t *server = (dispatch_server_t *)opaque;
  mock_state_t state;
  uint8_t hello[SAGR_WIRE_HELLO_FRAME_BYTES];
  int peer = -1;
  ssize_t hello_size;
  memset(&state, 0, sizeof(state));
  do {
    peer = accept4(server->listener, NULL, NULL, SOCK_CLOEXEC);
  } while (peer < 0 && errno == EINTR);
  if (peer < 0) {
    server->thread_error = 1;
    return NULL;
  }
  do {
    hello_size = recv(peer, hello, sizeof(hello), 0);
  } while (hello_size < 0 && errno == EINTR);
  if (send_handshake_ack(peer, hello, hello_size, &state) != 0) {
    server->thread_error = 1;
    goto done;
  }
  for (;;) {
    mock_request_t request;
    const int received = receive_request(peer, &state, &request);
    if (received == 1) {
      break;
    }
    if (received != 0 || handle_request(server, peer, &state, &request) != 0) {
      if (request.descriptor >= 0) {
        (void)close(request.descriptor);
      }
      server->thread_error = 1;
      break;
    }
  }
done:
  (void)close(peer);
  return NULL;
}

static int start_server(dispatch_server_t *server,
                        enum dispatch_server_behavior behavior) {
  struct sockaddr_un address;
  char template_path[] = "/tmp/sagr-dispatch-XXXXXX";
  char *directory;
  memset(server, 0, sizeof(*server));
  server->listener = -1;
  server->behavior = behavior;
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
      pthread_create(&server->thread, NULL, dispatch_server_main, server) != 0) {
    (void)close(server->listener);
    server->listener = -1;
    return -1;
  }
  return 0;
}

static int stop_server(dispatch_server_t *server) {
  int result = 0;
  if (pthread_join(server->thread, NULL) != 0) {
    result = -1;
  }
  if (server->listener >= 0 && close(server->listener) != 0) {
    result = -1;
  }
  if (unlink(server->endpoint) != 0 && errno != ENOENT) {
    result = -1;
  }
  if (rmdir(server->directory) != 0 && errno != ENOENT) {
    result = -1;
  }
  return result;
}

static void initialize_open_options(sagr_instance_open_options_t *options) {
  const uint64_t capabilities =
      SAGR_CAPABILITY_TOPOLOGY_MASK | SAGR_CAPABILITY_QUEUE_MASK |
      SAGR_CAPABILITY_MEMORY_MASK | SAGR_CAPABILITY_SIGNAL_MASK |
      SAGR_CAPABILITY_DISPATCH_MASK;
  (void)sagr_instance_open_options_init(options, (uint32_t)sizeof(*options));
  options->offered_capabilities[0] = capabilities;
  options->required_capabilities[0] = capabilities;
}

static int open_instance(const char *endpoint, sagr_instance_t *instance,
                         sagr_error_info_t *error) {
  sagr_instance_open_options_t options;
  initialize_open_options(&options);
  return sagr_instance_open(endpoint, &options, instance, error,
                            (uint32_t)sizeof(*error)) == SAGR_STATUS_SUCCESS
             ? 0
             : -1;
}

static int create_resources(dispatch_resources_t *resources,
                            const char *endpoint, int create_output,
                            int create_signal, sagr_error_info_t *error) {
  sagr_memory_allocate_options_t memory_options;
  sagr_signal_create_options_t signal_options;
  memset(resources, 0, sizeof(*resources));
  if (open_instance(endpoint, &resources->instance, error) != 0 ||
      sagr_queue_create(resources->instance, NULL, NULL, &resources->queue,
                        NULL, 0, error, (uint32_t)sizeof(*error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_memory_allocate_options_init(
          &memory_options, (uint32_t)sizeof(memory_options)) !=
          SAGR_STATUS_SUCCESS) {
    return -1;
  }
  memory_options.size_bytes = SAGR_DISPATCH_FIXED_IO_BYTES;
  if (sagr_memory_allocate(resources->instance, &memory_options, NULL,
                           &resources->input, NULL, 0, error,
                           (uint32_t)sizeof(*error)) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  if (create_output != 0 &&
      sagr_memory_allocate(resources->instance, &memory_options, NULL,
                           &resources->output, NULL, 0, error,
                           (uint32_t)sizeof(*error)) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  if (create_signal != 0) {
    if (sagr_signal_create_options_init(
            &signal_options, (uint32_t)sizeof(signal_options)) !=
        SAGR_STATUS_SUCCESS) {
      return -1;
    }
    signal_options.initial_value = INT64_C(1);
    if (sagr_signal_create(resources->instance, &signal_options, NULL,
                           &resources->signal, NULL, 0, error,
                           (uint32_t)sizeof(*error)) != SAGR_STATUS_SUCCESS) {
      return -1;
    }
  }
  return 0;
}

static int arm_signal_wait(sagr_signal_t signal, sagr_error_info_t *error) {
  sagr_signal_operation_options_t options;
  sagr_signal_wait_result_t result;
  if (sagr_signal_operation_options_init(
          &options, (uint32_t)sizeof(options)) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  options.timeout_ns = UINT64_C(1000000);
  memset(&result, 0xa5, sizeof(result));
  return sagr_signal_wait(signal, SAGR_SIGNAL_CONDITION_EQ, INT64_C(0),
                          &options, &result, (uint32_t)sizeof(result), error,
                          (uint32_t)sizeof(*error)) == SAGR_STATUS_TIMED_OUT &&
                 result.struct_size == sizeof(result) && result.status == 0 &&
                 result.signal_id == 0
             ? 0
             : -1;
}

static sagr_status_t submit_dispatch(
    dispatch_resources_t *resources, sagr_dispatch_ticket_t *ticket,
    sagr_error_info_t *error) {
  sagr_pinned_dispatch_options_t options;
  if (sagr_pinned_dispatch_options_init(
          &options, (uint32_t)sizeof(options)) != SAGR_STATUS_SUCCESS) {
    return SAGR_STATUS_INTERNAL_ERROR;
  }
  memset(ticket, 0xa5, sizeof(*ticket));
  return sagr_queue_submit_pinned_dispatch(
      resources->queue, resources->input, resources->output, resources->signal,
      &options, NULL, ticket, (uint32_t)sizeof(*ticket), error,
      (uint32_t)sizeof(*error));
}

static int wait_and_cleanup(dispatch_resources_t *resources,
                            const sagr_dispatch_ticket_t *ticket,
                            int retry_timeout, sagr_error_info_t *error) {
  sagr_dispatch_completion_t completion;
  sagr_signal_wait_result_t signal_result;
  sagr_queue_operation_options_t wait_options;
  sagr_status_t status;
  if (retry_timeout != 0) {
    int cancel_pipe[2] = {-1, -1};
    const uint8_t cancel_byte = UINT8_C(1);
    if (sagr_queue_operation_options_init(
            &wait_options, (uint32_t)sizeof(wait_options)) !=
        SAGR_STATUS_SUCCESS) {
      return -1;
    }
    wait_options.timeout_ns = UINT64_C(1000000);
    memset(&completion, 0xa5, sizeof(completion));
    status = sagr_queue_wait_pinned_dispatch(
        resources->queue, ticket, &wait_options, &completion,
        (uint32_t)sizeof(completion), error, (uint32_t)sizeof(*error));
    if (status != SAGR_STATUS_TIMED_OUT ||
        completion.struct_size != sizeof(completion) || completion.status != 0 ||
        completion.trace_id != 0) {
      return -1;
    }
    if (pipe2(cancel_pipe, O_CLOEXEC) != 0 ||
        write(cancel_pipe[1], &cancel_byte, sizeof(cancel_byte)) !=
            (ssize_t)sizeof(cancel_byte) ||
        sagr_queue_operation_options_init(
            &wait_options, (uint32_t)sizeof(wait_options)) !=
            SAGR_STATUS_SUCCESS) {
      if (cancel_pipe[0] >= 0) {
        (void)close(cancel_pipe[0]);
        (void)close(cancel_pipe[1]);
      }
      return -1;
    }
    wait_options.cancel_fd = cancel_pipe[0];
    memset(&completion, 0xa5, sizeof(completion));
    status = sagr_queue_wait_pinned_dispatch(
        resources->queue, ticket, &wait_options, &completion,
        (uint32_t)sizeof(completion), error, (uint32_t)sizeof(*error));
    (void)close(cancel_pipe[0]);
    (void)close(cancel_pipe[1]);
    if (status != SAGR_STATUS_CANCELLED ||
        completion.struct_size != sizeof(completion) || completion.status != 0 ||
        completion.trace_id != 0) {
      return -1;
    }
  }
  memset(&completion, 0, sizeof(completion));
  status = sagr_queue_wait_pinned_dispatch(
      resources->queue, ticket, NULL, &completion,
      (uint32_t)sizeof(completion), error, (uint32_t)sizeof(*error));
  if (status != SAGR_STATUS_SUCCESS ||
      completion.request_id != ticket->request_id ||
      completion.queue_id != ticket->queue_id ||
      completion.queue_generation != ticket->queue_generation ||
      completion.queue_sequence != ticket->queue_sequence ||
      completion.fixture_id != ticket->fixture_id ||
      completion.input_allocation_id != ticket->input_allocation_id ||
      completion.input_generation != ticket->input_generation ||
      completion.output_allocation_id != ticket->output_allocation_id ||
      completion.output_generation != ticket->output_generation ||
      completion.signal_id != ticket->signal_id ||
      completion.signal_generation != ticket->signal_generation ||
      completion.trace_id != ticket->trace_id ||
      completion.input_gpu_va != ticket->input_gpu_va ||
      completion.output_gpu_va != ticket->output_gpu_va ||
      completion.packet_crc32c != SAGR_DISPATCH_PACKET_CRC32C ||
      completion.output_crc32c != SAGR_DISPATCH_OUTPUT_CRC32C ||
      completion.admission_tick != MOCK_ADMISSION_TICK ||
      completion.start_tick != MOCK_START_TICK ||
      completion.end_tick != MOCK_END_TICK ||
      completion.retire_tick != MOCK_RETIRE_TICK) {
    return -1;
  }
  memset(&signal_result, 0, sizeof(signal_result));
  if (sagr_signal_wait(resources->signal, SAGR_SIGNAL_CONDITION_EQ, INT64_C(0),
                       NULL, &signal_result,
                       (uint32_t)sizeof(signal_result), error,
                       (uint32_t)sizeof(*error)) != SAGR_STATUS_SUCCESS ||
      signal_result.observed_value != 0 ||
      signal_result.completion_tick != MOCK_RETIRE_TICK + UINT64_C(1) ||
      sagr_memory_free(&resources->output, NULL, error,
                       (uint32_t)sizeof(*error)) != SAGR_STATUS_SUCCESS ||
      sagr_memory_free(&resources->input, NULL, error,
                       (uint32_t)sizeof(*error)) != SAGR_STATUS_SUCCESS ||
      sagr_signal_destroy(&resources->signal, NULL, error,
                          (uint32_t)sizeof(*error)) != SAGR_STATUS_SUCCESS ||
      sagr_queue_destroy(&resources->queue, NULL, error,
                         (uint32_t)sizeof(*error)) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return 0;
}

static int run_success_case(enum dispatch_server_behavior behavior,
                            int reject_once) {
  dispatch_server_t server;
  dispatch_resources_t resources;
  sagr_dispatch_ticket_t ticket;
  sagr_dispatch_ticket_t changed;
  sagr_dispatch_completion_t completion;
  sagr_error_info_t error;
  sagr_status_t status;
  int failed = 0;
  memset(&error, 0, sizeof(error));
  if (start_server(&server, behavior) != 0 ||
      create_resources(&resources, server.endpoint, 1, 1, &error) != 0) {
    return 1;
  }
  status = submit_dispatch(&resources, &ticket, &error);
  if (status != SAGR_STATUS_INVALID_ARGUMENT || ticket.struct_size !=
          sizeof(ticket) || ticket.request_id != 0 ||
      arm_signal_wait(resources.signal, &error) != 0) {
    fprintf(stderr, "unarmed dispatch prerequisite gate failed: %s\n",
            error.message);
    failed = 1;
    goto done;
  }
  status = submit_dispatch(&resources, &ticket, &error);
  if (reject_once != 0) {
    if (status != SAGR_STATUS_BUSY || ticket.request_id != 0 ||
        ticket.trace_id != 0) {
      fprintf(stderr, "canonical dispatch rejection was not atomic\n");
      failed = 1;
      goto done;
    }
    status = submit_dispatch(&resources, &ticket, &error);
  }
  if (status != SAGR_STATUS_SUCCESS || ticket.request_id == 0 ||
      ticket.queue_id != MOCK_QUEUE_ID || ticket.queue_sequence != 1 ||
      ticket.trace_id != MOCK_TRACE_ID || ticket.input_gpu_va != MOCK_INPUT_VA ||
      ticket.output_gpu_va != MOCK_OUTPUT_VA ||
      ticket.packet_crc32c != SAGR_DISPATCH_PACKET_CRC32C ||
      ticket.admission_tick != MOCK_ADMISSION_TICK) {
    fprintf(stderr, "dispatch admission ticket mismatch: %s\n", error.message);
    failed = 1;
    goto done;
  }
  if (sagr_memory_free(&resources.input, NULL, &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_BUSY ||
      sagr_signal_destroy(&resources.signal, NULL, &error,
                          (uint32_t)sizeof(error)) != SAGR_STATUS_BUSY ||
      sagr_queue_destroy(&resources.queue, NULL, &error,
                         (uint32_t)sizeof(error)) != SAGR_STATUS_BUSY) {
    fprintf(stderr, "dispatch resources were not pinned\n");
    failed = 1;
    goto done;
  }
  changed = ticket;
  ++changed.request_id;
  memset(&completion, 0xa5, sizeof(completion));
  if (sagr_queue_wait_pinned_dispatch(
          resources.queue, &changed, NULL, &completion,
          (uint32_t)sizeof(completion), &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_BUSY ||
      completion.struct_size != sizeof(completion) || completion.trace_id != 0) {
    fprintf(stderr, "mutated admission ticket was accepted\n");
    failed = 1;
    goto done;
  }
  if (wait_and_cleanup(&resources, &ticket,
                       behavior == DISPATCH_SERVER_SUCCESS, &error) != 0) {
    fprintf(stderr, "dispatch wait/cleanup failed: %s\n", error.message);
    failed = 1;
  }
done:
  (void)sagr_instance_close(&resources.instance);
  if (stop_server(&server) != 0 || server.thread_error != 0 ||
      server.signal_wait_requests != 1U ||
      server.dispatch_requests != (reject_once != 0 ? 2U : 1U)) {
    failed = 1;
  }
  return failed;
}

static int run_poison_case(enum dispatch_server_behavior behavior,
                           int failure_at_submit) {
  dispatch_server_t server;
  dispatch_resources_t resources;
  sagr_dispatch_ticket_t ticket;
  sagr_dispatch_completion_t completion;
  sagr_error_info_t error;
  sagr_status_t status;
  int failed = 0;
  memset(&error, 0, sizeof(error));
  if (start_server(&server, behavior) != 0 ||
      create_resources(&resources, server.endpoint, 1, 1, &error) != 0 ||
      arm_signal_wait(resources.signal, &error) != 0) {
    return 1;
  }
  status = submit_dispatch(&resources, &ticket, &error);
  if (failure_at_submit != 0) {
    if (status != SAGR_STATUS_PROTOCOL_ERROR || ticket.request_id != 0) {
      fprintf(stderr, "bad dispatch ACK was accepted\n");
      failed = 1;
    }
  } else if (status != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "poison test dispatch was not admitted: %s\n",
            error.message);
    failed = 1;
  } else {
    memset(&completion, 0xa5, sizeof(completion));
    status = sagr_queue_wait_pinned_dispatch(
        resources.queue, &ticket, NULL, &completion,
        (uint32_t)sizeof(completion), &error, (uint32_t)sizeof(error));
    if (status != SAGR_STATUS_PROTOCOL_ERROR || completion.trace_id != 0) {
      fprintf(stderr, "noncanonical completion pair was accepted\n");
      failed = 1;
    }
  }
  if (sagr_queue_destroy(&resources.queue, NULL, &error,
                         (uint32_t)sizeof(error)) !=
      SAGR_STATUS_CONNECTION_LOST) {
    fprintf(stderr, "dispatch protocol error did not poison the session\n");
    failed = 1;
  }
  (void)sagr_instance_close(&resources.instance);
  if (stop_server(&server) != 0 || server.thread_error != 0 ||
      server.dispatch_requests != 1U) {
    failed = 1;
  }
  return failed;
}

static int test_same_instance_gate(void) {
  dispatch_server_t first_server;
  dispatch_server_t second_server;
  dispatch_resources_t first;
  dispatch_resources_t second;
  sagr_pinned_dispatch_options_t options;
  sagr_dispatch_ticket_t ticket;
  sagr_error_info_t error;
  int failed = 0;
  memset(&error, 0, sizeof(error));
  if (start_server(&first_server, DISPATCH_SERVER_SUCCESS) != 0 ||
      start_server(&second_server, DISPATCH_SERVER_SUCCESS) != 0 ||
      create_resources(&first, first_server.endpoint, 0, 0, &error) != 0 ||
      create_resources(&second, second_server.endpoint, 1, 1, &error) != 0 ||
      sagr_pinned_dispatch_options_init(
          &options, (uint32_t)sizeof(options)) != SAGR_STATUS_SUCCESS) {
    return 1;
  }
  memset(&ticket, 0, sizeof(ticket));
  if (sagr_queue_submit_pinned_dispatch(
          first.queue, first.input, second.output, second.signal, &options,
          NULL, &ticket, (uint32_t)sizeof(ticket), &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_INSTANCE_MISMATCH ||
      ticket.request_id != 0) {
    fprintf(stderr, "cross-instance dispatch handles were accepted\n");
    failed = 1;
  }
  (void)sagr_instance_close(&first.instance);
  (void)sagr_instance_close(&second.instance);
  if (stop_server(&first_server) != 0 || stop_server(&second_server) != 0 ||
      first_server.thread_error != 0 || second_server.thread_error != 0 ||
      first_server.dispatch_requests != 0 || second_server.dispatch_requests != 0) {
    failed = 1;
  }
  return failed;
}

#ifdef SAGR_DISPATCH_CLI_PATH
static int test_dispatch_cli_json(void) {
  dispatch_server_t server;
  int output_pipe[2] = {-1, -1};
  char output[8192];
  size_t output_size = 0;
  pid_t child;
  int child_status = 0;
  int failed = 0;
  if (start_server(&server, DISPATCH_SERVER_SUCCESS) != 0 ||
      pipe2(output_pipe, O_CLOEXEC) != 0) {
    return 1;
  }
  child = fork();
  if (child == 0) {
    (void)dup2(output_pipe[1], STDOUT_FILENO);
    (void)close(output_pipe[0]);
    (void)close(output_pipe[1]);
    execl(SAGR_DISPATCH_CLI_PATH, SAGR_DISPATCH_CLI_PATH, "--endpoint",
          server.endpoint, "--dispatch-fixture", "gfx950-xor-u8-v1",
          (char *)NULL);
    _exit(127);
  }
  (void)close(output_pipe[1]);
  output_pipe[1] = -1;
  if (child < 0) {
    failed = 1;
  } else {
    for (;;) {
      const ssize_t count = read(output_pipe[0], output + output_size,
                                 sizeof(output) - output_size - 1U);
      if (count > 0) {
        output_size += (size_t)count;
      } else if (count < 0 && errno == EINTR) {
        continue;
      } else {
        break;
      }
      if (output_size + 1U >= sizeof(output)) {
        failed = 1;
        break;
      }
    }
    if (waitpid(child, &child_status, 0) != child ||
        !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
      failed = 1;
    }
  }
  output[output_size] = '\0';
  if (strstr(output, "\"capability_words\":[\"0x000000000000001f\"") ==
          NULL ||
      strstr(output, "\"dispatch\":{\"status\":0") == NULL ||
      strstr(output, "\"fixture\":\"gfx950-xor-u8-v1\"") == NULL ||
      strstr(output, SAGR_DISPATCH_FIXTURE_MANIFEST_SHA256_HEX) == NULL ||
      strstr(output,
             "\"input_hex\":\"000102030405060708090a0b0c0d0e0f"
             "101112131415161718191a1b1c1d1e1f2021222324252627"
             "28292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f\"") ==
          NULL ||
      strstr(output,
             "\"initial_output_hex\":\"00000000000000000000000000000000"
             "00000000000000000000000000000000"
             "00000000000000000000000000000000"
             "00000000000000000000000000000000\"") == NULL ||
      strstr(output,
             "\"expected_output_hex\":\"5a5b58595e5f5c5d5253505156575455"
             "4a4b48494e4f4c4d42434041464744457a7b78797e7f7c7d"
             "72737071767774756a6b68696e6f6c6d6263606166676465\"") ==
          NULL ||
      strstr(output,
             "\"d2h_output_hex\":\"5a5b58595e5f5c5d5253505156575455"
             "4a4b48494e4f4c4d42434041464744457a7b78797e7f7c7d"
             "72737071767774756a6b68696e6f6c6d6263606166676465\"") ==
          NULL ||
      strstr(output, "\"request_id\":\"0x") == NULL ||
      strstr(output, "\"queue_sequence\":\"0x0000000000000001\"") ==
          NULL ||
      strstr(output, "\"packet_crc32c\":\"0x8a912d83\"") == NULL ||
      strstr(output,
             "\"first_wait\":{\"status\":19,\"status_name\":\"cancelled\""
             ",\"wire_status\":-1,\"retried_without_send\":true}") == NULL ||
      strstr(output, "\"completion\":{\"status\":0,\"wire_status\":0,"
                     "\"request_id\":\"0x") == NULL ||
      strstr(output, "\"output_crc32c\":\"0x796671ec\"") == NULL ||
      strstr(output, "\"admission_tick\":\"0x0000000000000064\"") ==
          NULL ||
      strstr(output, "\"start_tick\":\"0x000000000000006e\"") == NULL ||
      strstr(output, "\"end_tick\":\"0x0000000000000078\"") == NULL ||
      strstr(output, "\"retire_tick\":\"0x0000000000000082\"") == NULL ||
      strstr(output,
             "\"signal_completion_tick\":\"0x0000000000000083\"") ==
          NULL ||
      strstr(output, "\"retried_without_send\":true") == NULL ||
      strstr(output, "\"output_match\":true") == NULL ||
      strstr(output,
             "\"cleanup\":{\"queue_destroyed\":true,\"input_freed\":true,"
             "\"output_freed\":true,\"signal_destroyed\":true}") == NULL ||
      server.dispatch_requests != 1U || server.signal_wait_requests != 1U) {
    fprintf(stderr, "dispatch CLI JSON gate failed: %s\n", output);
    failed = 1;
  }
  (void)close(output_pipe[0]);
  if (stop_server(&server) != 0 || server.thread_error != 0) {
    failed = 1;
  }
  return failed;
}
#endif

int main(void) {
  int failures = 0;
  failures += run_success_case(DISPATCH_SERVER_SUCCESS, 0);
  failures += run_success_case(DISPATCH_SERVER_REJECT_ONCE, 1);
  failures += run_poison_case(DISPATCH_SERVER_BAD_ACK_PACKET_CRC, 1);
  failures += run_poison_case(DISPATCH_SERVER_BAD_SIGNAL_TICK, 0);
  failures += run_poison_case(DISPATCH_SERVER_COMPLETION_BEFORE_SIGNAL, 0);
  failures += test_same_instance_gate();
#ifdef SAGR_DISPATCH_CLI_PATH
  failures += test_dispatch_cli_json();
#endif
  return failures == 0 ? 0 : 1;
}
