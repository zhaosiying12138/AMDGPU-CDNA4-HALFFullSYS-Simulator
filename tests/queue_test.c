/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stddef.h>
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

enum queue_server_behavior {
  QUEUE_SERVER_SUCCESS,
  QUEUE_SERVER_NO_CAPABILITY,
  QUEUE_SERVER_FAILED_ACK_BAD_QUEUE,
  QUEUE_SERVER_FAILED_ACK_BAD_SEQUENCE,
  QUEUE_SERVER_FAILED_ACK_BAD_VALUE,
  QUEUE_SERVER_FAILED_ACK_BAD_ERROR,
  QUEUE_SERVER_DOORBELL_ACK_BAD_VALUE,
  QUEUE_SERVER_DOORBELL_ACK_MAX_TICK,
  QUEUE_SERVER_COMPLETION_BAD_REQUEST,
  QUEUE_SERVER_COMPLETION_BAD_TICK,
  QUEUE_SERVER_NONCANONICAL_ERROR_COMPLETION,
  QUEUE_SERVER_COMPLETION_BEFORE_ACK,
  QUEUE_SERVER_INTERLEAVED_PRIOR_COMPLETION,
  QUEUE_SERVER_INTERLEAVED_BAD_TICK,
  QUEUE_SERVER_CONTROL_ERROR,
  QUEUE_SERVER_NO_COMPLETION,
  QUEUE_SERVER_INFLIGHT_LIMIT,
  QUEUE_SERVER_DUPLICATE_CREATE,
  QUEUE_SERVER_CREATE_ACK_TIMEOUT,
  QUEUE_SERVER_CREATE_ACK_CANCEL,
  QUEUE_SERVER_DOORBELL_ACK_TIMEOUT,
  QUEUE_SERVER_DESTROY_ACK_TIMEOUT
};

typedef struct queue_server {
  char directory[128];
  char endpoint[160];
  int listener;
  pthread_t thread;
  enum queue_server_behavior behavior;
  int cancel_write_fd;
  int thread_error;
} queue_server_t;

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
  size_t index;
  uint8_t combined = 0;
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

static int expect_peer_close(int peer) {
  struct pollfd descriptor;
  uint8_t byte;
  int result;
  descriptor.fd = peer;
  descriptor.events = POLLIN;
  descriptor.revents = 0;
  do {
    result = poll(&descriptor, 1, 1000);
  } while (result < 0 && errno == EINTR);
  return result > 0 && recv(peer, &byte, sizeof(byte), 0) == 0 ? 0 : -1;
}

static int receive_queue_request(int peer, const sagr_instance_info_t *info,
                                 uint64_t *last_request_id,
                                 sagr_wire_queue_request_t *request,
                                 uint64_t *request_id) {
  static const uint8_t magic[8] = {'G', 'S', 'I', 'M', 'R', 'P', 'C', 0};
  uint8_t frame[SAGR_WIRE_QUEUE_FRAME_BYTES];
  uint8_t crc_frame[SAGR_WIRE_QUEUE_FRAME_BYTES];
  const uint8_t *payload = frame + SAGR_WIRE_HEADER_BYTES;
  const ssize_t received = recv(peer, frame, sizeof(frame), 0);
  uint32_t actual_crc;
  if (received != (ssize_t)sizeof(frame)) {
    return -1;
  }
  actual_crc = get_u32(frame + 64);
  memcpy(crc_frame, frame, sizeof(frame));
  memset(crc_frame + 64, 0, 4);
  if (memcmp(frame, magic, sizeof(magic)) != 0 || get_u16(frame + 8) != 1 ||
      get_u16(frame + 10) != 0 ||
      get_u16(frame + 12) != SAGR_WIRE_HEADER_BYTES ||
      get_u16(frame + 14) != SAGR_WIRE_MESSAGE_QUEUE_REQUEST ||
      get_u32(frame + 16) != 0 ||
      get_u32(frame + 20) != SAGR_WIRE_QUEUE_PAYLOAD_BYTES ||
      get_u32(frame + 68) != 0 || get_u64(frame + 72) != 0 ||
      actual_crc != sagr_crc32c(crc_frame, sizeof(crc_frame)) ||
      memcmp(frame + 32, info->daemon_uuid, 16) != 0 ||
      get_u64(frame + 48) != info->connection_id ||
      get_u64(frame + 56) != info->epoch ||
      get_u16(payload) != SAGR_QUEUE_PROTOCOL_MAJOR ||
      get_u16(payload + 2) != SAGR_QUEUE_PROTOCOL_MINOR ||
      get_u16(payload + 6) != 0 ||
      !bytes_are_zero(payload + 48, 16)) {
    return -1;
  }
  *request_id = get_u64(frame + 24);
  if (*request_id == 0 || *request_id <= *last_request_id) {
    return -1;
  }
  *last_request_id = *request_id;
  memset(request, 0, sizeof(*request));
  request->major = get_u16(payload);
  request->minor = get_u16(payload + 2);
  request->opcode = get_u16(payload + 4);
  request->flags = get_u16(payload + 6);
  request->queue_id = get_u64(payload + 8);
  request->generation = get_u64(payload + 16);
  request->sequence = get_u64(payload + 24);
  request->arg0 = get_u64(payload + 32);
  request->arg1 = get_u64(payload + 40);
  return 0;
}

static int send_queue_response(int peer, const sagr_instance_info_t *info,
                               uint64_t request_id, uint16_t message_type,
                               const sagr_wire_queue_response_t *response) {
  uint8_t frame[SAGR_WIRE_QUEUE_FRAME_BYTES];
  size_t frame_size = 0;
  if (sagr_protocol_encode_queue_response(
          info, request_id, message_type, response, frame, sizeof(frame),
          &frame_size) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static void initialize_queue_response(sagr_wire_queue_response_t *response,
                                      uint16_t opcode) {
  memset(response, 0, sizeof(*response));
  response->major = SAGR_QUEUE_PROTOCOL_MAJOR;
  response->minor = SAGR_QUEUE_PROTOCOL_MINOR;
  response->status = SAGR_WIRE_STATUS_OK;
  response->opcode = opcode;
}

static int validate_create(const sagr_wire_queue_request_t *request) {
  return request->opcode == SAGR_WIRE_QUEUE_OPCODE_CREATE &&
         request->queue_id == 0 && request->generation == 0 &&
         request->sequence == 0 && request->arg0 == 4 && request->arg1 == 0;
}

static int validate_doorbell(const sagr_wire_queue_request_t *request,
                             uint64_t sequence, uint64_t command_kind) {
  return request->opcode == SAGR_WIRE_QUEUE_OPCODE_DOORBELL &&
         request->queue_id == UINT64_C(0x1020304050607080) &&
         request->generation == UINT64_C(0x8877665544332211) &&
         request->sequence == sequence && request->arg0 == command_kind &&
         request->arg1 == 0;
}

static int send_handshake_ack(int peer, const uint8_t *hello,
                              ssize_t hello_size, int queue_capability,
                              sagr_instance_info_t *info,
                              uint64_t *last_request_id) {
  sagr_wire_ack_fields_t fields;
  uint8_t ack[SAGR_WIRE_ACK_FRAME_BYTES];
  size_t ack_size = 0;
  if (hello_size < SAGR_WIRE_HEADER_BYTES + SAGR_WIRE_HELLO_FIXED_BYTES) {
    return -1;
  }
  memset(&fields, 0, sizeof(fields));
  fields.selected_major = 1;
  fields.status = SAGR_WIRE_STATUS_OK;
  memcpy(fields.client_nonce, hello + SAGR_WIRE_HEADER_BYTES + 8, 16);
  memcpy(fields.server_nonce, k_server_nonce, 16);
  fields.selected_capabilities[0] =
      SAGR_CAPABILITY_TOPOLOGY_MASK |
      (queue_capability != 0 ? SAGR_CAPABILITY_QUEUE_MASK : 0);
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
      sagr_protocol_encode_ack(&fields, ack, sizeof(ack), &ack_size) !=
          SAGR_STATUS_SUCCESS ||
      send_frame(peer, ack, ack_size) != 0) {
    return -1;
  }
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  info->maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  info->negotiated_capabilities[0] = fields.selected_capabilities[0];
  memcpy(info->daemon_uuid, fields.daemon_uuid, 16);
  info->connection_id = fields.connection_id;
  info->epoch = fields.epoch;
  *last_request_id = fields.request_id;
  return 0;
}

static int handle_failed_create(queue_server_t *server, int peer,
                                const sagr_instance_info_t *info,
                                uint64_t *last_request_id) {
  sagr_wire_queue_request_t request;
  sagr_wire_queue_response_t response;
  uint64_t request_id = 0;
  if (receive_queue_request(peer, info, last_request_id, &request,
                            &request_id) != 0 ||
      !validate_create(&request)) {
    return -1;
  }
  initialize_queue_response(&response, SAGR_WIRE_QUEUE_OPCODE_CREATE);
  response.status = SAGR_WIRE_STATUS_PROTOCOL_STATE;
  if (server->behavior == QUEUE_SERVER_FAILED_ACK_BAD_QUEUE) {
    response.queue_id = 1;
  } else if (server->behavior == QUEUE_SERVER_FAILED_ACK_BAD_SEQUENCE) {
    response.sequence = 1;
  } else if (server->behavior == QUEUE_SERVER_FAILED_ACK_BAD_VALUE) {
    response.value = 1;
  } else if (server->behavior == QUEUE_SERVER_FAILED_ACK_BAD_ERROR) {
    response.error_code = 1;
  }
  return send_queue_response(peer, info, request_id,
                             SAGR_WIRE_MESSAGE_QUEUE_ACK, &response);
}

static int handle_successful_create(int peer,
                                    const sagr_instance_info_t *info,
                                    uint64_t *last_request_id) {
  sagr_wire_queue_request_t request;
  sagr_wire_queue_response_t response;
  uint64_t request_id = 0;
  if (receive_queue_request(peer, info, last_request_id, &request,
                            &request_id) != 0 ||
      !validate_create(&request)) {
    return -1;
  }
  initialize_queue_response(&response, SAGR_WIRE_QUEUE_OPCODE_CREATE);
  response.queue_id = UINT64_C(0x1020304050607080);
  response.generation = UINT64_C(0x8877665544332211);
  response.value = request.arg0;
  response.sim_tick = UINT64_C(100);
  return send_queue_response(peer, info, request_id,
                             SAGR_WIRE_MESSAGE_QUEUE_ACK, &response);
}

static int handle_duplicate_create(int peer, const sagr_instance_info_t *info,
                                   uint64_t *last_request_id) {
  if (handle_successful_create(peer, info, last_request_id) != 0) {
    return -1;
  }
  return expect_peer_close(peer);
}

static int handle_doorbell(int peer, const sagr_instance_info_t *info,
                           uint64_t *last_request_id, uint64_t sequence,
                           uint64_t command_kind,
                           enum queue_server_behavior behavior) {
  sagr_wire_queue_request_t request;
  sagr_wire_queue_response_t response;
  uint64_t request_id = 0;
  if (receive_queue_request(peer, info, last_request_id, &request,
                            &request_id) != 0 ||
      !validate_doorbell(&request, sequence, command_kind)) {
    return -1;
  }
  initialize_queue_response(&response, SAGR_WIRE_QUEUE_OPCODE_DOORBELL);
  response.queue_id = request.queue_id;
  response.generation = request.generation;
  response.sequence = request.sequence;
  response.value = behavior == QUEUE_SERVER_DOORBELL_ACK_BAD_VALUE
                       ? command_kind + UINT64_C(1)
                       : 0;
  response.sim_tick = UINT64_C(100) + sequence * UINT64_C(2);
  if (behavior == QUEUE_SERVER_DOORBELL_ACK_MAX_TICK) {
    response.sim_tick = UINT64_MAX;
  }
  if (behavior == QUEUE_SERVER_COMPLETION_BEFORE_ACK) {
    response.value = command_kind;
    ++response.sim_tick;
    if (send_queue_response(peer, info, request_id,
                            SAGR_WIRE_MESSAGE_QUEUE_COMPLETION,
                            &response) != 0) {
      return -1;
    }
    response.value = 0;
    --response.sim_tick;
  }
  if (send_queue_response(peer, info, request_id,
                          SAGR_WIRE_MESSAGE_QUEUE_ACK, &response) != 0) {
    return -1;
  }
  if (behavior == QUEUE_SERVER_DOORBELL_ACK_BAD_VALUE ||
      behavior == QUEUE_SERVER_DOORBELL_ACK_MAX_TICK ||
      behavior == QUEUE_SERVER_INFLIGHT_LIMIT ||
      behavior == QUEUE_SERVER_COMPLETION_BEFORE_ACK) {
    return 0;
  }
  if (behavior == QUEUE_SERVER_NO_COMPLETION) {
    const struct timespec delay = {.tv_sec = 0, .tv_nsec = 100000000L};
    (void)nanosleep(&delay, NULL);
  }
  response.value = command_kind;
  response.sim_tick +=
      behavior == QUEUE_SERVER_COMPLETION_BAD_TICK ? UINT64_C(2) : UINT64_C(1);
  if (behavior == QUEUE_SERVER_CONTROL_ERROR ||
      behavior == QUEUE_SERVER_NONCANONICAL_ERROR_COMPLETION) {
    response.status = SAGR_WIRE_STATUS_INTERNAL;
    response.error_code =
        behavior == QUEUE_SERVER_CONTROL_ERROR ? UINT64_C(1) : 0;
  }
  return send_queue_response(
      peer, info,
      behavior == QUEUE_SERVER_COMPLETION_BAD_REQUEST ? request_id + 1
                                                      : request_id,
      SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &response);
}

static int handle_interleaved_doorbells(
    int peer, const sagr_instance_info_t *info, uint64_t *last_request_id,
    int bad_tick) {
  sagr_wire_queue_request_t first;
  sagr_wire_queue_request_t second;
  sagr_wire_queue_response_t response;
  uint64_t first_request_id = 0;
  uint64_t second_request_id = 0;
  if (receive_queue_request(peer, info, last_request_id, &first,
                            &first_request_id) != 0 ||
      !validate_doorbell(&first, 1, SAGR_QUEUE_COMMAND_NOOP)) {
    return -1;
  }
  initialize_queue_response(&response, SAGR_WIRE_QUEUE_OPCODE_DOORBELL);
  response.queue_id = first.queue_id;
  response.generation = first.generation;
  response.sequence = first.sequence;
  response.sim_tick = UINT64_C(102);
  if (send_queue_response(peer, info, first_request_id,
                          SAGR_WIRE_MESSAGE_QUEUE_ACK, &response) != 0 ||
      receive_queue_request(peer, info, last_request_id, &second,
                            &second_request_id) != 0 ||
      !validate_doorbell(&second, 2, SAGR_QUEUE_COMMAND_CONTROL_TEST)) {
    return -1;
  }

  response.value = SAGR_QUEUE_COMMAND_NOOP;
  response.sim_tick = bad_tick != 0 ? UINT64_C(104) : UINT64_C(103);
  if (send_queue_response(peer, info, first_request_id,
                          SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &response) != 0) {
    return -1;
  }
  if (bad_tick != 0) {
    return expect_peer_close(peer);
  }
  response.sequence = second.sequence;
  response.value = 0;
  response.sim_tick = UINT64_C(104);
  if (send_queue_response(peer, info, second_request_id,
                          SAGR_WIRE_MESSAGE_QUEUE_ACK, &response) != 0) {
    return -1;
  }
  response.value = SAGR_QUEUE_COMMAND_CONTROL_TEST;
  response.sim_tick = UINT64_C(105);
  return send_queue_response(peer, info, second_request_id,
                             SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &response);
}

static int handle_destroy(int peer, const sagr_instance_info_t *info,
                          uint64_t *last_request_id) {
  sagr_wire_queue_request_t request;
  sagr_wire_queue_response_t response;
  uint64_t request_id = 0;
  if (receive_queue_request(peer, info, last_request_id, &request,
                            &request_id) != 0 ||
      request.opcode != SAGR_WIRE_QUEUE_OPCODE_DESTROY ||
      request.queue_id != UINT64_C(0x1020304050607080) ||
      request.generation != UINT64_C(0x8877665544332211) ||
      request.sequence != 0 || request.arg0 != 0 || request.arg1 != 0) {
    return -1;
  }
  initialize_queue_response(&response, SAGR_WIRE_QUEUE_OPCODE_DESTROY);
  response.queue_id = request.queue_id;
  response.generation = request.generation;
  response.sim_tick = UINT64_C(200);
  return send_queue_response(peer, info, request_id,
                             SAGR_WIRE_MESSAGE_QUEUE_ACK, &response);
}

static int withhold_ack_until_disconnect(
    int peer, const sagr_instance_info_t *info, uint64_t *last_request_id,
    uint16_t opcode, int cancel_write_fd) {
  sagr_wire_queue_request_t request;
  uint64_t request_id = 0;
  if (receive_queue_request(peer, info, last_request_id, &request,
                            &request_id) != 0) {
    return -1;
  }
  if ((opcode == SAGR_WIRE_QUEUE_OPCODE_CREATE &&
       !validate_create(&request)) ||
      (opcode == SAGR_WIRE_QUEUE_OPCODE_DOORBELL &&
       !validate_doorbell(&request, 1,
                          SAGR_QUEUE_COMMAND_CONTROL_TEST)) ||
      (opcode == SAGR_WIRE_QUEUE_OPCODE_DESTROY &&
       (request.opcode != SAGR_WIRE_QUEUE_OPCODE_DESTROY ||
        request.queue_id != UINT64_C(0x1020304050607080) ||
        request.generation != UINT64_C(0x8877665544332211) ||
        request.sequence != 0 || request.arg0 != 0 || request.arg1 != 0))) {
    return -1;
  }
  if (cancel_write_fd >= 0 && write(cancel_write_fd, "x", 1) != 1) {
    return -1;
  }
  return expect_peer_close(peer);
}

static void *queue_server_main(void *argument) {
  queue_server_t *server = (queue_server_t *)argument;
  sagr_instance_info_t info;
  uint8_t hello[SAGR_WIRE_MAX_HANDSHAKE_BYTES];
  uint64_t last_request_id = 0;
  ssize_t hello_size;
  int peer = accept4(server->listener, NULL, NULL, SOCK_CLOEXEC);
  if (peer < 0) {
    server->thread_error = errno;
    return NULL;
  }
  hello_size = recv(peer, hello, sizeof(hello), 0);
  if (send_handshake_ack(peer, hello, hello_size,
                         server->behavior != QUEUE_SERVER_NO_CAPABILITY, &info,
                         &last_request_id) != 0) {
    server->thread_error = EPROTO;
    (void)close(peer);
    return NULL;
  }
  if (server->behavior == QUEUE_SERVER_NO_CAPABILITY) {
    (void)close(peer);
    return NULL;
  }
  if (server->behavior == QUEUE_SERVER_CREATE_ACK_TIMEOUT ||
      server->behavior == QUEUE_SERVER_CREATE_ACK_CANCEL) {
    if (withhold_ack_until_disconnect(peer, &info, &last_request_id,
                                      SAGR_WIRE_QUEUE_OPCODE_CREATE,
                                      server->cancel_write_fd) != 0) {
      server->thread_error = EPROTO;
    }
    (void)close(peer);
    return NULL;
  }
  if (server->behavior >= QUEUE_SERVER_FAILED_ACK_BAD_QUEUE &&
      server->behavior <= QUEUE_SERVER_FAILED_ACK_BAD_ERROR) {
    if (handle_failed_create(server, peer, &info, &last_request_id) != 0) {
      server->thread_error = EPROTO;
    }
    (void)close(peer);
    return NULL;
  }
  if (handle_successful_create(peer, &info, &last_request_id) != 0) {
    server->thread_error = EPROTO;
    (void)close(peer);
    return NULL;
  }
  if (server->behavior == QUEUE_SERVER_DUPLICATE_CREATE) {
    if (handle_duplicate_create(peer, &info, &last_request_id) != 0) {
      server->thread_error = EPROTO;
    }
    (void)close(peer);
    return NULL;
  }
  if (server->behavior == QUEUE_SERVER_DOORBELL_ACK_TIMEOUT) {
    if (withhold_ack_until_disconnect(peer, &info, &last_request_id,
                                      SAGR_WIRE_QUEUE_OPCODE_DOORBELL, -1) !=
        0) {
      server->thread_error = EPROTO;
    }
    (void)close(peer);
    return NULL;
  }
  if (server->behavior == QUEUE_SERVER_DESTROY_ACK_TIMEOUT) {
    if (withhold_ack_until_disconnect(peer, &info, &last_request_id,
                                      SAGR_WIRE_QUEUE_OPCODE_DESTROY, -1) !=
        0) {
      server->thread_error = EPROTO;
    }
    (void)close(peer);
    return NULL;
  }
  if (server->behavior == QUEUE_SERVER_INFLIGHT_LIMIT) {
    uint64_t sequence;
    for (sequence = 1; sequence <= SAGR_QUEUE_MAX_INFLIGHT; ++sequence) {
      if (handle_doorbell(peer, &info, &last_request_id, sequence,
                          SAGR_QUEUE_COMMAND_NOOP, server->behavior) != 0) {
        server->thread_error = EPROTO;
        break;
      }
    }
    (void)close(peer);
    return NULL;
  }
  if (server->behavior == QUEUE_SERVER_INTERLEAVED_PRIOR_COMPLETION ||
      server->behavior == QUEUE_SERVER_INTERLEAVED_BAD_TICK) {
    if (handle_interleaved_doorbells(
            peer, &info, &last_request_id,
            server->behavior == QUEUE_SERVER_INTERLEAVED_BAD_TICK) != 0 ||
        (server->behavior == QUEUE_SERVER_INTERLEAVED_PRIOR_COMPLETION &&
         handle_destroy(peer, &info, &last_request_id) != 0)) {
      server->thread_error = EPROTO;
    }
    (void)close(peer);
    return NULL;
  }
  if (handle_doorbell(peer, &info, &last_request_id, 1,
                      server->behavior == QUEUE_SERVER_SUCCESS
                          ? SAGR_QUEUE_COMMAND_NOOP
                          : ((server->behavior ==
                                      QUEUE_SERVER_CONTROL_ERROR ||
                              server->behavior ==
                                  QUEUE_SERVER_NONCANONICAL_ERROR_COMPLETION)
                                 ? SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST
                                 : SAGR_QUEUE_COMMAND_CONTROL_TEST),
                      server->behavior) != 0) {
    server->thread_error = EPROTO;
    (void)close(peer);
    return NULL;
  }
  if (server->behavior == QUEUE_SERVER_NO_COMPLETION ||
      server->behavior == QUEUE_SERVER_CONTROL_ERROR) {
    if (handle_destroy(peer, &info, &last_request_id) != 0) {
      server->thread_error = EPROTO;
    }
    (void)close(peer);
    return NULL;
  }
  if (server->behavior != QUEUE_SERVER_SUCCESS) {
    (void)close(peer);
    return NULL;
  }
  if (handle_doorbell(peer, &info, &last_request_id, 2,
                      SAGR_QUEUE_COMMAND_CONTROL_TEST,
                      server->behavior) != 0 ||
      handle_destroy(peer, &info, &last_request_id) != 0) {
    server->thread_error = EPROTO;
  }
  (void)close(peer);
  return NULL;
}

static int start_server_with_cancel(queue_server_t *server,
                                    enum queue_server_behavior behavior,
                                    int cancel_write_fd) {
  struct sockaddr_un address;
  size_t endpoint_size;
  memset(server, 0, sizeof(*server));
  server->listener = -1;
  server->cancel_write_fd = cancel_write_fd;
  server->behavior = behavior;
  (void)snprintf(server->directory, sizeof(server->directory),
                 "/tmp/sagr-queue-test-XXXXXX");
  if (mkdtemp(server->directory) == NULL) {
    return -1;
  }
  if (snprintf(server->endpoint, sizeof(server->endpoint), "%s/socket",
               server->directory) >= (int)sizeof(server->endpoint)) {
    (void)rmdir(server->directory);
    return -1;
  }
  server->listener = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (server->listener < 0) {
    (void)rmdir(server->directory);
    return -1;
  }
  memset(&address, 0, sizeof(address));
  address.sun_family = AF_UNIX;
  endpoint_size = strlen(server->endpoint);
  memcpy(address.sun_path, server->endpoint, endpoint_size + 1U);
  if (bind(server->listener, (const struct sockaddr *)&address,
           (socklen_t)(offsetof(struct sockaddr_un, sun_path) + endpoint_size +
                       1U)) != 0 ||
      listen(server->listener, 1) != 0 ||
      pthread_create(&server->thread, NULL, queue_server_main, server) != 0) {
    (void)close(server->listener);
    (void)unlink(server->endpoint);
    (void)rmdir(server->directory);
    return -1;
  }
  return 0;
}

static int start_server(queue_server_t *server,
                        enum queue_server_behavior behavior) {
  return start_server_with_cancel(server, behavior, -1);
}

static int finish_server(queue_server_t *server) {
  int failure = 0;
  if (pthread_join(server->thread, NULL) != 0 || server->thread_error != 0) {
    fprintf(stderr, "queue mock server failed: %d\n", server->thread_error);
    failure = 1;
  }
  (void)close(server->listener);
  if (unlink(server->endpoint) != 0 || rmdir(server->directory) != 0) {
    fprintf(stderr, "queue mock server cleanup failed\n");
    failure = 1;
  }
  return failure;
}

static void initialize_open_options(sagr_instance_open_options_t *options,
                                    int queue_capability) {
  (void)sagr_instance_open_options_init(options, (uint32_t)sizeof(*options));
  memcpy(options->expected_daemon_uuid, k_daemon_uuid, 16);
  memcpy(options->expected_job_uuid, k_job_uuid, 16);
  options->expected_epoch = UINT64_C(0x0102030405060708);
  options->expected_rank = 3;
  options->expected_world_size = 8;
  options->open_timeout_ns = UINT64_C(1000000000);
  if (queue_capability != 0) {
    options->offered_capabilities[0] |= SAGR_CAPABILITY_QUEUE_MASK;
    options->required_capabilities[0] |= SAGR_CAPABILITY_QUEUE_MASK;
  }
}

static int open_instance(queue_server_t *server, int queue_capability,
                         sagr_instance_t *instance) {
  sagr_instance_open_options_t options;
  sagr_error_info_t error;
  initialize_open_options(&options, queue_capability);
  if (sagr_instance_open(server->endpoint, &options, instance, &error,
                         (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "queue test handshake failed: %s\n", error.message);
    return -1;
  }
  return 0;
}

static sagr_status_t create_queue_with_operation(
    sagr_instance_t instance,
    const sagr_queue_operation_options_t *operation_options,
    sagr_queue_t *queue, sagr_queue_info_t *info, sagr_error_info_t *error) {
  sagr_queue_create_options_t create_options;
  (void)sagr_queue_create_options_init(&create_options,
                                       (uint32_t)sizeof(create_options));
  create_options.depth = 4;
  return sagr_queue_create(instance, &create_options, operation_options, queue,
                           info, (uint32_t)sizeof(*info), error,
                           (uint32_t)sizeof(*error));
}

static sagr_status_t create_queue(sagr_instance_t instance,
                                  sagr_queue_t *queue,
                                  sagr_queue_info_t *info,
                                  sagr_error_info_t *error) {
  return create_queue_with_operation(instance, NULL, queue, info, error);
}

static int check_completion(const sagr_queue_completion_t *completion,
                            uint64_t sequence, uint64_t command_kind) {
  return completion->struct_size == sizeof(*completion) &&
         completion->status == SAGR_STATUS_SUCCESS &&
         completion->wire_status == SAGR_WIRE_STATUS_OK &&
         completion->queue_id == UINT64_C(0x1020304050607080) &&
         completion->generation == UINT64_C(0x8877665544332211) &&
         completion->sequence == sequence &&
         completion->value == command_kind && completion->error_code == 0 &&
         completion->sim_tick == UINT64_C(101) + sequence * UINT64_C(2);
}

static int test_queue_lifecycle(void) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t info;
  sagr_queue_completion_t completion;
  sagr_queue_create_options_t create_options;
  sagr_error_info_t error;
  uint64_t sequence = 0;
  int failure = 0;
  if (start_server(&server, QUEUE_SERVER_SUCCESS) != 0 ||
      open_instance(&server, 1, &instance) != 0) {
    return 1;
  }
  (void)sagr_queue_create_options_init(&create_options,
                                       (uint32_t)sizeof(create_options));
  create_options.depth = 4;
  queue = (sagr_queue_t)(uintptr_t)1;
  memset(&info, 0xa5, sizeof(info));
  if (sagr_queue_create(instance, &create_options, NULL, &queue, &info,
                        (uint32_t)sizeof(info) - 1U, &error,
                        (uint32_t)sizeof(error)) !=
          SAGR_STATUS_BUFFER_TOO_SMALL ||
      queue != NULL || info.struct_size != (uint32_t)sizeof(info)) {
    fprintf(stderr, "short queue info output changed protocol state\n");
    failure = 1;
  }
  if (create_queue(instance, &queue, &info, &error) != SAGR_STATUS_SUCCESS ||
      queue == NULL || info.struct_size != sizeof(info) || info.depth != 4 ||
      info.queue_id != UINT64_C(0x1020304050607080) ||
      info.generation != UINT64_C(0x8877665544332211)) {
    fprintf(stderr, "queue create returned invalid metadata: %s\n",
            error.message);
    failure = 1;
  }
  if (failure == 0 &&
      (sagr_queue_ring_doorbell(queue, SAGR_QUEUE_COMMAND_NOOP, NULL,
                                &sequence, &error,
                                (uint32_t)sizeof(error)) !=
           SAGR_STATUS_SUCCESS ||
       sequence != 1 ||
       sagr_queue_wait(queue, sequence, NULL, &completion,
                       (uint32_t)sizeof(completion) - 1U, &error,
                       (uint32_t)sizeof(error)) !=
           SAGR_STATUS_BUFFER_TOO_SMALL ||
       completion.struct_size != (uint32_t)sizeof(completion) ||
       sagr_queue_wait(queue, sequence, NULL, &completion,
                       (uint32_t)sizeof(completion), &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
       !check_completion(&completion, 1, SAGR_QUEUE_COMMAND_NOOP) ||
       sagr_queue_wait(queue, sequence, NULL, &completion,
                       (uint32_t)sizeof(completion), &error,
                       (uint32_t)sizeof(error)) !=
           SAGR_STATUS_INVALID_ARGUMENT)) {
    fprintf(stderr, "NOOP doorbell lifecycle failed: %s\n", error.message);
    failure = 1;
  }
  if (failure == 0 &&
      (sagr_queue_ring_doorbell(queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL,
                                &sequence, &error,
                                (uint32_t)sizeof(error)) !=
           SAGR_STATUS_SUCCESS ||
       sequence != 2 ||
       sagr_queue_wait(queue, sequence, NULL, &completion,
                       (uint32_t)sizeof(completion), &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
       !check_completion(&completion, 2,
                         SAGR_QUEUE_COMMAND_CONTROL_TEST))) {
    fprintf(stderr, "CONTROL_TEST doorbell lifecycle failed: %s\n",
            error.message);
    failure = 1;
  }
  if (failure == 0 &&
      (sagr_queue_destroy(&queue, NULL, &error, (uint32_t)sizeof(error)) !=
           SAGR_STATUS_SUCCESS ||
       queue != NULL ||
       sagr_queue_destroy(&queue, NULL, &error,
                          (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS)) {
    fprintf(stderr, "queue destroy failed: %s\n", error.message);
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

static int test_capability_absent(void) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = (sagr_queue_t)(uintptr_t)1;
  sagr_queue_info_t info;
  sagr_error_info_t error;
  int failure = 0;
  if (start_server(&server, QUEUE_SERVER_NO_CAPABILITY) != 0 ||
      open_instance(&server, 0, &instance) != 0) {
    return 1;
  }
  if (create_queue(instance, &queue, &info, &error) !=
          SAGR_STATUS_NOT_SUPPORTED ||
      queue != NULL || error.status != SAGR_STATUS_NOT_SUPPORTED) {
    fprintf(stderr, "queue API accepted an unnegotiated capability\n");
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

static int test_duplicate_create_handle(void) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t first = NULL;
  sagr_queue_t second = NULL;
  sagr_queue_info_t info;
  sagr_error_info_t error;
  uint64_t sequence = 0;
  int failure = 0;
  if (start_server(&server, QUEUE_SERVER_DUPLICATE_CREATE) != 0 ||
      open_instance(&server, 1, &instance) != 0 ||
      create_queue(instance, &first, &info, &error) != SAGR_STATUS_SUCCESS) {
    return 1;
  }
  if (create_queue(instance, &second, &info, &error) !=
          SAGR_STATUS_PROTOCOL_ERROR ||
      second != NULL ||
      sagr_queue_ring_doorbell(first, SAGR_QUEUE_COMMAND_NOOP, NULL,
                               &sequence, &error,
                               (uint32_t)sizeof(error)) !=
          SAGR_STATUS_CONNECTION_LOST) {
    fprintf(stderr, "duplicate active queue ID was accepted or not poisoned\n");
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

static int test_failed_ack_mutations(void) {
  static const enum queue_server_behavior cases[] = {
      QUEUE_SERVER_FAILED_ACK_BAD_QUEUE,
      QUEUE_SERVER_FAILED_ACK_BAD_SEQUENCE,
      QUEUE_SERVER_FAILED_ACK_BAD_VALUE,
      QUEUE_SERVER_FAILED_ACK_BAD_ERROR};
  size_t index;
  int failures = 0;
  for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
    queue_server_t server;
    sagr_instance_t instance = NULL;
    sagr_queue_t queue = NULL;
    sagr_queue_info_t info;
    sagr_error_info_t error;
    if (start_server(&server, cases[index]) != 0 ||
        open_instance(&server, 1, &instance) != 0) {
      ++failures;
      continue;
    }
    if (create_queue(instance, &queue, &info, &error) !=
            SAGR_STATUS_PROTOCOL_ERROR ||
        queue != NULL || error.status != SAGR_STATUS_PROTOCOL_ERROR ||
        error.wire_status != SAGR_WIRE_STATUS_PROTOCOL_STATE) {
      fprintf(stderr, "failed queue ACK mutation %zu was accepted\n", index);
      ++failures;
    }
    if (create_queue(instance, &queue, &info, &error) !=
            SAGR_STATUS_CONNECTION_LOST ||
        queue != NULL) {
      fprintf(stderr, "failed ACK mutation did not poison the transport\n");
      ++failures;
    }
    (void)sagr_instance_close(&instance);
    failures += finish_server(&server);
  }
  return failures;
}

static int test_doorbell_ack_value(void) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t info;
  sagr_error_info_t error;
  uint64_t sequence = UINT64_MAX;
  int failure = 0;
  if (start_server(&server, QUEUE_SERVER_DOORBELL_ACK_BAD_VALUE) != 0 ||
      open_instance(&server, 1, &instance) != 0 ||
      create_queue(instance, &queue, &info, &error) != SAGR_STATUS_SUCCESS) {
    return 1;
  }
  if (sagr_queue_ring_doorbell(
          queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL, &sequence, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_PROTOCOL_ERROR ||
      sequence != 0 || error.status != SAGR_STATUS_PROTOCOL_ERROR) {
    fprintf(stderr, "nonzero successful DOORBELL ACK value was accepted\n");
    failure = 1;
  }
  if (sagr_queue_ring_doorbell(
          queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL, &sequence, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_CONNECTION_LOST) {
    fprintf(stderr, "malformed DOORBELL ACK did not poison the transport\n");
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

static int test_completion_request_id(void) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t info;
  sagr_queue_completion_t completion;
  sagr_error_info_t error;
  uint64_t sequence = 0;
  int failure = 0;
  if (start_server(&server, QUEUE_SERVER_COMPLETION_BAD_REQUEST) != 0 ||
      open_instance(&server, 1, &instance) != 0 ||
      create_queue(instance, &queue, &info, &error) != SAGR_STATUS_SUCCESS ||
      sagr_queue_ring_doorbell(
          queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL, &sequence, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    return 1;
  }
  if (sagr_queue_wait(queue, sequence, NULL, &completion,
                      (uint32_t)sizeof(completion), &error,
                      (uint32_t)sizeof(error)) != SAGR_STATUS_PROTOCOL_ERROR ||
      error.status != SAGR_STATUS_PROTOCOL_ERROR) {
    fprintf(stderr, "completion with wrong request ID was accepted\n");
    failure = 1;
  }
  if (sagr_queue_ring_doorbell(
          queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL, &sequence, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_CONNECTION_LOST) {
    fprintf(stderr, "foreign completion did not poison the transport\n");
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

static int test_doorbell_ack_max_tick(void) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t info;
  sagr_error_info_t error;
  uint64_t sequence = 0;
  int failure = 0;
  if (start_server(&server, QUEUE_SERVER_DOORBELL_ACK_MAX_TICK) != 0 ||
      open_instance(&server, 1, &instance) != 0 ||
      create_queue(instance, &queue, &info, &error) != SAGR_STATUS_SUCCESS) {
    return 1;
  }
  if (sagr_queue_ring_doorbell(
          queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL, &sequence, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_PROTOCOL_ERROR ||
      sequence != 0 ||
      sagr_queue_ring_doorbell(
          queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL, &sequence, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_CONNECTION_LOST) {
    fprintf(stderr, "UINT64_MAX DOORBELL ACK tick was accepted\n");
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

static int run_bad_completion(enum queue_server_behavior behavior,
                              uint64_t command_kind) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t info;
  sagr_queue_completion_t completion;
  sagr_error_info_t error;
  uint64_t sequence = 0;
  int failure = 0;
  if (start_server(&server, behavior) != 0 ||
      open_instance(&server, 1, &instance) != 0 ||
      create_queue(instance, &queue, &info, &error) != SAGR_STATUS_SUCCESS ||
      sagr_queue_ring_doorbell(queue, command_kind, NULL, &sequence, &error,
                               (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS) {
    return 1;
  }
  if (sagr_queue_wait(queue, sequence, NULL, &completion,
                      (uint32_t)sizeof(completion), &error,
                      (uint32_t)sizeof(error)) != SAGR_STATUS_PROTOCOL_ERROR ||
      sagr_queue_ring_doorbell(queue, command_kind, NULL, &sequence, &error,
                               (uint32_t)sizeof(error)) !=
          SAGR_STATUS_CONNECTION_LOST) {
    fprintf(stderr, "noncanonical completion was accepted or did not poison\n");
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

static int test_control_error_completion(void) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t info;
  sagr_queue_completion_t completion;
  sagr_error_info_t error;
  uint64_t sequence = 0;
  int failure = 0;
  if (start_server(&server, QUEUE_SERVER_CONTROL_ERROR) != 0 ||
      open_instance(&server, 1, &instance) != 0 ||
      create_queue(instance, &queue, &info, &error) != SAGR_STATUS_SUCCESS ||
      sagr_queue_ring_doorbell(
          queue, SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST, NULL, &sequence,
          &error, (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    return 1;
  }
  if (sagr_queue_wait(queue, sequence, NULL, &completion,
                      (uint32_t)sizeof(completion), &error,
                      (uint32_t)sizeof(error)) != SAGR_STATUS_INTERNAL_ERROR ||
      completion.status != SAGR_STATUS_INTERNAL_ERROR ||
      completion.wire_status != SAGR_WIRE_STATUS_INTERNAL ||
      completion.value != SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST ||
      completion.error_code != UINT64_C(1) ||
      completion.sim_tick != UINT64_C(103) ||
      sagr_queue_destroy(&queue, NULL, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "canonical CONTROL_ERROR_TEST completion failed\n");
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

static int test_interleaved_bad_completion(void) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t info;
  sagr_error_info_t error;
  uint64_t first_sequence = 0;
  uint64_t second_sequence = 0;
  int failure = 0;
  if (start_server(&server, QUEUE_SERVER_INTERLEAVED_BAD_TICK) != 0 ||
      open_instance(&server, 1, &instance) != 0 ||
      create_queue(instance, &queue, &info, &error) != SAGR_STATUS_SUCCESS ||
      sagr_queue_ring_doorbell(queue, SAGR_QUEUE_COMMAND_NOOP, NULL,
                               &first_sequence, &error,
                               (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS) {
    return 1;
  }
  if (sagr_queue_ring_doorbell(
          queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL, &second_sequence,
          &error, (uint32_t)sizeof(error)) != SAGR_STATUS_PROTOCOL_ERROR ||
      second_sequence != 0 ||
      sagr_queue_ring_doorbell(queue, SAGR_QUEUE_COMMAND_NOOP, NULL,
                               &second_sequence, &error,
                               (uint32_t)sizeof(error)) !=
          SAGR_STATUS_CONNECTION_LOST) {
    fprintf(stderr, "bad interleaved completion was buffered or not poisoned\n");
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

static int run_no_completion(sagr_status_t expected, int cancel) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t info;
  sagr_queue_completion_t completion;
  sagr_queue_operation_options_t operation;
  sagr_error_info_t error;
  uint64_t sequence = 0;
  int cancellation_pipe[2] = {-1, -1};
  int failure = 0;
  if (start_server(&server, QUEUE_SERVER_NO_COMPLETION) != 0 ||
      open_instance(&server, 1, &instance) != 0 ||
      create_queue(instance, &queue, &info, &error) != SAGR_STATUS_SUCCESS ||
      sagr_queue_ring_doorbell(
          queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL, &sequence, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    return 1;
  }
  (void)sagr_queue_operation_options_init(&operation,
                                          (uint32_t)sizeof(operation));
  operation.timeout_ns = UINT64_C(10000000);
  if (cancel != 0) {
    if (pipe2(cancellation_pipe, O_CLOEXEC | O_NONBLOCK) != 0 ||
        write(cancellation_pipe[1], "x", 1) != 1) {
      failure = 1;
    } else {
      operation.timeout_ns = UINT64_MAX;
      operation.cancel_fd = cancellation_pipe[0];
    }
  }
  if (failure == 0 &&
      (sagr_queue_wait(queue, sequence, &operation, &completion,
                       (uint32_t)sizeof(completion), &error,
                       (uint32_t)sizeof(error)) != expected ||
       error.status != expected)) {
    fprintf(stderr, "queue wait deadline/cancellation result mismatch\n");
    failure = 1;
  }
  if (failure == 0 &&
      (sagr_queue_wait(queue, sequence, NULL, &completion,
                       (uint32_t)sizeof(completion), &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
       !check_completion(&completion, sequence,
                         SAGR_QUEUE_COMMAND_CONTROL_TEST))) {
    fprintf(stderr, "queue wait was not retryable after timeout/cancel\n");
    failure = 1;
  }
  if (cancellation_pipe[0] >= 0) {
    (void)close(cancellation_pipe[0]);
    (void)close(cancellation_pipe[1]);
  }
  if (failure == 0 &&
      sagr_queue_destroy(&queue, NULL, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "queue destroy after wait retry failed\n");
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

static int test_completion_before_ack(void) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t info;
  sagr_error_info_t error;
  uint64_t sequence = 0;
  int failure = 0;
  if (start_server(&server, QUEUE_SERVER_COMPLETION_BEFORE_ACK) != 0 ||
      open_instance(&server, 1, &instance) != 0 ||
      create_queue(instance, &queue, &info, &error) != SAGR_STATUS_SUCCESS) {
    return 1;
  }
  if (sagr_queue_ring_doorbell(
          queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL, &sequence, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_PROTOCOL_ERROR ||
      sequence != 0 ||
      sagr_queue_ring_doorbell(
          queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL, &sequence, &error,
          (uint32_t)sizeof(error)) != SAGR_STATUS_CONNECTION_LOST) {
    fprintf(stderr, "pre-ACK completion was accepted or did not poison\n");
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

static int test_interleaved_prior_completion(void) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t info;
  sagr_queue_completion_t completion;
  sagr_error_info_t error;
  uint64_t first_sequence = 0;
  uint64_t second_sequence = 0;
  int failure = 0;
  if (start_server(&server, QUEUE_SERVER_INTERLEAVED_PRIOR_COMPLETION) != 0 ||
      open_instance(&server, 1, &instance) != 0 ||
      create_queue(instance, &queue, &info, &error) != SAGR_STATUS_SUCCESS ||
      sagr_queue_ring_doorbell(queue, SAGR_QUEUE_COMMAND_NOOP, NULL,
                               &first_sequence, &error,
                               (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_queue_ring_doorbell(
          queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL, &second_sequence,
          &error, (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      first_sequence != 1 || second_sequence != 2 ||
      sagr_queue_wait(queue, first_sequence, NULL, &completion,
                      (uint32_t)sizeof(completion), &error,
                      (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      !check_completion(&completion, first_sequence,
                        SAGR_QUEUE_COMMAND_NOOP) ||
      sagr_queue_wait(queue, second_sequence, NULL, &completion,
                      (uint32_t)sizeof(completion), &error,
                      (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      !check_completion(&completion, second_sequence,
                        SAGR_QUEUE_COMMAND_CONTROL_TEST) ||
      sagr_queue_destroy(&queue, NULL, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "prior acknowledged completion interleave failed: %s\n",
            error.message);
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

static int run_buffered_precondition(int expired) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t info;
  sagr_queue_completion_t completion;
  sagr_queue_operation_options_t operation;
  sagr_error_info_t error;
  uint64_t first_sequence = 0;
  uint64_t second_sequence = 0;
  int cancellation_pipe[2] = {-1, -1};
  const sagr_status_t expected =
      expired != 0 ? SAGR_STATUS_TIMED_OUT : SAGR_STATUS_CANCELLED;
  int failure = 0;
  if (start_server(&server, QUEUE_SERVER_INTERLEAVED_PRIOR_COMPLETION) != 0 ||
      open_instance(&server, 1, &instance) != 0 ||
      create_queue(instance, &queue, &info, &error) != SAGR_STATUS_SUCCESS ||
      sagr_queue_ring_doorbell(queue, SAGR_QUEUE_COMMAND_NOOP, NULL,
                               &first_sequence, &error,
                               (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_queue_ring_doorbell(
          queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL, &second_sequence,
          &error, (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      pipe2(cancellation_pipe, O_CLOEXEC | O_NONBLOCK) != 0 ||
      write(cancellation_pipe[1], "x", 1) != 1) {
    return 1;
  }
  (void)sagr_queue_operation_options_init(&operation,
                                          (uint32_t)sizeof(operation));
  operation.cancel_fd = cancellation_pipe[0];
  if (expired != 0) {
    operation.absolute_deadline_ns = UINT64_C(1);
  } else {
    operation.timeout_ns = UINT64_MAX;
  }
  if (sagr_queue_wait(queue, first_sequence, &operation, &completion,
                      (uint32_t)sizeof(completion), &error,
                      (uint32_t)sizeof(error)) != expected ||
      error.status != expected) {
    fprintf(stderr, "buffered completion ignored deadline/cancellation\n");
    failure = 1;
  }
  (void)close(cancellation_pipe[0]);
  (void)close(cancellation_pipe[1]);
  if (failure == 0 &&
      (sagr_queue_wait(queue, first_sequence, NULL, &completion,
                       (uint32_t)sizeof(completion), &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
       !check_completion(&completion, first_sequence,
                         SAGR_QUEUE_COMMAND_NOOP) ||
       sagr_queue_wait(queue, second_sequence, NULL, &completion,
                       (uint32_t)sizeof(completion), &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
       !check_completion(&completion, second_sequence,
                         SAGR_QUEUE_COMMAND_CONTROL_TEST) ||
       sagr_queue_destroy(&queue, NULL, &error,
                          (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS)) {
    fprintf(stderr, "buffered completion was consumed before retry\n");
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

static int run_ack_timeout(enum queue_server_behavior behavior) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t info;
  sagr_queue_operation_options_t operation;
  sagr_error_info_t error;
  uint64_t sequence = 0;
  sagr_status_t status;
  int failure = 0;
  if (start_server(&server, behavior) != 0 ||
      open_instance(&server, 1, &instance) != 0) {
    return 1;
  }
  (void)sagr_queue_operation_options_init(&operation,
                                          (uint32_t)sizeof(operation));
  operation.timeout_ns = UINT64_C(10000000);
  if (behavior == QUEUE_SERVER_CREATE_ACK_TIMEOUT) {
    status = create_queue_with_operation(instance, &operation, &queue, &info,
                                         &error);
    if (status != SAGR_STATUS_TIMED_OUT || queue != NULL ||
        create_queue(instance, &queue, &info, &error) !=
            SAGR_STATUS_CONNECTION_LOST) {
      fprintf(stderr, "CREATE ACK timeout did not poison the transport\n");
      failure = 1;
    }
  } else if (create_queue(instance, &queue, &info, &error) !=
             SAGR_STATUS_SUCCESS) {
    failure = 1;
  } else if (behavior == QUEUE_SERVER_DOORBELL_ACK_TIMEOUT) {
    status = sagr_queue_ring_doorbell(
        queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, &operation, &sequence, &error,
        (uint32_t)sizeof(error));
    if (status != SAGR_STATUS_TIMED_OUT || sequence != 0 ||
        sagr_queue_ring_doorbell(
            queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL, &sequence, &error,
            (uint32_t)sizeof(error)) != SAGR_STATUS_CONNECTION_LOST) {
      fprintf(stderr, "DOORBELL ACK timeout did not poison the transport\n");
      failure = 1;
    }
  } else {
    status = sagr_queue_destroy(&queue, &operation, &error,
                                (uint32_t)sizeof(error));
    if (status != SAGR_STATUS_TIMED_OUT || queue == NULL ||
        sagr_queue_destroy(&queue, NULL, &error,
                           (uint32_t)sizeof(error)) !=
            SAGR_STATUS_CONNECTION_LOST) {
      fprintf(stderr, "DESTROY ACK timeout did not poison the transport\n");
      failure = 1;
    }
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

static int test_ack_cancellation_after_send(void) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t info;
  sagr_queue_operation_options_t operation;
  sagr_error_info_t error;
  int cancellation_pipe[2] = {-1, -1};
  int failure = 0;
  if (pipe2(cancellation_pipe, O_CLOEXEC | O_NONBLOCK) != 0 ||
      start_server_with_cancel(&server, QUEUE_SERVER_CREATE_ACK_CANCEL,
                               cancellation_pipe[1]) != 0) {
    return 1;
  }
  if (open_instance(&server, 1, &instance) != 0) {
    return 1;
  }
  (void)sagr_queue_operation_options_init(&operation,
                                          (uint32_t)sizeof(operation));
  operation.timeout_ns = UINT64_MAX;
  operation.cancel_fd = cancellation_pipe[0];
  if (create_queue_with_operation(instance, &operation, &queue, &info,
                                  &error) != SAGR_STATUS_CANCELLED ||
      queue != NULL ||
      create_queue(instance, &queue, &info, &error) !=
          SAGR_STATUS_CONNECTION_LOST) {
    fprintf(stderr, "post-send ACK cancellation did not poison transport\n");
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  failure += finish_server(&server);
  (void)close(cancellation_pipe[0]);
  (void)close(cancellation_pipe[1]);
  return failure;
}

static int test_inflight_limit(void) {
  queue_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_queue_info_t info;
  sagr_error_info_t error;
  uint64_t sequence = 0;
  uint32_t index;
  int failure = 0;
  if (start_server(&server, QUEUE_SERVER_INFLIGHT_LIMIT) != 0 ||
      open_instance(&server, 1, &instance) != 0 ||
      create_queue(instance, &queue, &info, &error) != SAGR_STATUS_SUCCESS) {
    return 1;
  }
  for (index = 0; index < SAGR_QUEUE_MAX_INFLIGHT; ++index) {
    if (sagr_queue_ring_doorbell(queue, SAGR_QUEUE_COMMAND_NOOP, NULL,
                                 &sequence, &error,
                                 (uint32_t)sizeof(error)) !=
            SAGR_STATUS_SUCCESS ||
        sequence != (uint64_t)index + 1) {
      fprintf(stderr, "in-flight doorbell %u failed\n", index);
      failure = 1;
      break;
    }
  }
  if (failure == 0 &&
      (sagr_queue_ring_doorbell(queue, SAGR_QUEUE_COMMAND_NOOP, NULL,
                                &sequence, &error,
                                (uint32_t)sizeof(error)) !=
           SAGR_STATUS_OUT_OF_RESOURCES ||
       sequence != 0)) {
    fprintf(stderr, "ninth in-flight doorbell was not rejected locally\n");
    failure = 1;
  }
  if (failure == 0 &&
      (sagr_queue_destroy(&queue, NULL, &error,
                          (uint32_t)sizeof(error)) != SAGR_STATUS_BUSY ||
       queue == NULL)) {
    fprintf(stderr, "destroy accepted a queue with pending completions\n");
    failure = 1;
  }
  (void)sagr_instance_close(&instance);
  return failure + finish_server(&server);
}

#ifdef SAGR_QUEUE_CLI_PATH
static int test_queue_cli_json(void) {
  queue_server_t server;
  char output[4096];
  size_t output_size = 0;
  int output_pipe[2] = {-1, -1};
  int child_status = 0;
  pid_t child;
  int failure = 0;
  if (start_server(&server, QUEUE_SERVER_CONTROL_ERROR) != 0 ||
      pipe2(output_pipe, O_CLOEXEC) != 0) {
    return 1;
  }
  child = fork();
  if (child == 0) {
    if (dup2(output_pipe[1], STDOUT_FILENO) < 0) {
      _exit(126);
    }
    (void)close(output_pipe[0]);
    (void)close(output_pipe[1]);
    execl(SAGR_QUEUE_CLI_PATH, "sagr-handshake", "--endpoint",
          server.endpoint, "--queue-depth", "4", "--doorbells", "1",
          "--command-kind", "2", "--timeout-ms", "1000", (char *)NULL);
    _exit(127);
  }
  (void)close(output_pipe[1]);
  if (child < 0) {
    failure = 1;
  } else {
    for (;;) {
      const ssize_t count =
          read(output_pipe[0], output + output_size,
               sizeof(output) - output_size - 1U);
      if (count > 0) {
        output_size += (size_t)count;
        if (output_size + 1U == sizeof(output)) {
          break;
        }
        continue;
      }
      if (count < 0 && errno == EINTR) {
        continue;
      }
      break;
    }
    output[output_size] = '\0';
    if (waitpid(child, &child_status, 0) != child ||
        !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0 ||
        strstr(output, "\"queue\":{\"status\":0") == NULL ||
        strstr(output, "\"queue_id\":\"0x1020304050607080\"") == NULL ||
        strstr(output, "\"generation\":\"0x8877665544332211\"") == NULL ||
        strstr(output, "\"command_kind\":2") == NULL ||
        strstr(output, "\"completion_status\":3") == NULL ||
        strstr(output, "\"completion_wire_status\":10") == NULL ||
        strstr(output, "\"completion_error_code\":1") == NULL ||
        strstr(output, "\"sequences\":[\"0x0000000000000001\"]") == NULL) {
      fprintf(stderr, "queue CLI JSON gate failed: %s\n", output);
      failure = 1;
    }
  }
  (void)close(output_pipe[0]);
  return failure + finish_server(&server);
}
#endif

int main(void) {
  int failures = 0;
  failures += test_queue_lifecycle();
  failures += test_capability_absent();
  failures += test_duplicate_create_handle();
  failures += test_failed_ack_mutations();
  failures += test_doorbell_ack_value();
  failures += test_doorbell_ack_max_tick();
  failures += test_completion_request_id();
  failures += run_bad_completion(QUEUE_SERVER_COMPLETION_BAD_TICK,
                                 SAGR_QUEUE_COMMAND_CONTROL_TEST);
  failures += run_bad_completion(
      QUEUE_SERVER_NONCANONICAL_ERROR_COMPLETION,
      SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST);
  failures += test_control_error_completion();
  failures += test_completion_before_ack();
  failures += test_interleaved_prior_completion();
  failures += test_interleaved_bad_completion();
  failures += run_buffered_precondition(1);
  failures += run_buffered_precondition(0);
  failures += run_no_completion(SAGR_STATUS_TIMED_OUT, 0);
  failures += run_no_completion(SAGR_STATUS_CANCELLED, 1);
  failures += run_ack_timeout(QUEUE_SERVER_CREATE_ACK_TIMEOUT);
  failures += run_ack_timeout(QUEUE_SERVER_DOORBELL_ACK_TIMEOUT);
  failures += run_ack_timeout(QUEUE_SERVER_DESTROY_ACK_TIMEOUT);
  failures += test_ack_cancellation_after_send();
  failures += test_inflight_limit();
#ifdef SAGR_QUEUE_CLI_PATH
  failures += test_queue_cli_json();
#endif
  return failures == 0 ? 0 : 1;
}
