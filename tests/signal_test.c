/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
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

enum signal_server_behavior {
  SIGNAL_SERVER_LIFECYCLE,
  SIGNAL_SERVER_NO_CAPABILITY,
  SIGNAL_SERVER_BAD_COMPLETION,
  SIGNAL_SERVER_WAIT_MAX_TICK,
  SIGNAL_SERVER_STORE_MAX_TICK,
  SIGNAL_SERVER_FAILED_CREATE_ONCE,
  SIGNAL_SERVER_DUPLICATE_GENERATION,
  SIGNAL_SERVER_QUEUE_INTERLEAVE_ACK,
  SIGNAL_SERVER_QUEUE_INTERLEAVE_COMPLETION,
  SIGNAL_SERVER_FOREIGN_UNTRIGGERED_DURING_STORE
};

typedef struct signal_server {
  char directory[128];
  char endpoint[160];
  int listener;
  pthread_t thread;
  enum signal_server_behavior behavior;
  int thread_error;
  uint64_t last_request_id;
  uint64_t signal_id;
  uint64_t generation;
  int signal_exists;
  int wait_pending;
  uint64_t wait_sequence;
  uint64_t wait_request_id;
  uint64_t wait_condition;
  uint64_t wait_compare_bits;
  uint64_t value_bits;
  uint64_t next_tick;
  uint32_t create_failures;
  uint32_t wait_requests;
  uint64_t queue_id;
  uint64_t queue_generation;
  uint64_t queue_sequence;
  uint64_t queue_request_id;
  uint64_t queue_ack_tick;
  uint64_t queue_kind;
  int queue_exists;
  int queue_pending;
  uint32_t signal_create_count;
  int foreign_wait_pending;
  uint64_t foreign_signal_id;
  uint64_t foreign_generation;
  uint64_t foreign_wait_sequence;
  uint64_t foreign_wait_request_id;
} signal_server_t;

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

static int receive_hello(int peer, uint64_t *request_id,
                         uint8_t client_nonce[16]) {
  uint8_t frame[SAGR_WIRE_MAX_HANDSHAKE_BYTES];
  const ssize_t received = recv(peer, frame, sizeof(frame), 0);
  if (received < (ssize_t)(SAGR_WIRE_HEADER_BYTES +
                           SAGR_WIRE_HELLO_FIXED_BYTES) ||
      get_u16(frame + 14) != 1 || get_u64(frame + 24) == 0 ||
      get_u16(frame + 8) != 1 || get_u16(frame + 10) != 0 ||
      get_u16(frame + 12) != SAGR_WIRE_HEADER_BYTES ||
      get_u32(frame + 16) != 0 || get_u32(frame + 20) != 128 ||
      get_u32(frame + 68) != 0 || get_u64(frame + 72) != 0) {
    return -1;
  }
  {
    uint8_t copy[SAGR_WIRE_MAX_HANDSHAKE_BYTES];
    memcpy(copy, frame, (size_t)received);
    memset(copy + 64, 0, 4);
    if (sagr_crc32c(copy, (size_t)received) != get_u32(frame + 64)) {
      return -1;
    }
  }
  memcpy(client_nonce, frame + SAGR_WIRE_HEADER_BYTES + 8, 16);
  *request_id = get_u64(frame + 24);
  return bytes_are_zero(client_nonce, 16) ? -1 : 0;
}

static int send_handshake_ack(signal_server_t *server, int peer,
                              uint64_t request_id,
                              const uint8_t client_nonce[16],
                              sagr_instance_info_t *info) {
  sagr_wire_ack_fields_t fields;
  uint8_t frame[SAGR_WIRE_ACK_FRAME_BYTES];
  size_t frame_size = 0;
  memset(&fields, 0, sizeof(fields));
  fields.selected_major = 1;
  fields.status = SAGR_WIRE_STATUS_OK;
  memcpy(fields.client_nonce, client_nonce, 16);
  memcpy(fields.server_nonce, k_server_nonce, 16);
  fields.selected_capabilities[0] = SAGR_CAPABILITY_TOPOLOGY_MASK;
  if (server->behavior != SIGNAL_SERVER_NO_CAPABILITY) {
    fields.selected_capabilities[0] |= SAGR_CAPABILITY_SIGNAL_MASK;
  }
  if (server->behavior == SIGNAL_SERVER_QUEUE_INTERLEAVE_ACK ||
      server->behavior == SIGNAL_SERVER_QUEUE_INTERLEAVE_COMPLETION) {
    fields.selected_capabilities[0] |= SAGR_CAPABILITY_QUEUE_MASK;
  }
  fields.maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  fields.request_id = request_id;
  memcpy(fields.daemon_uuid, k_daemon_uuid, 16);
  fields.connection_id = UINT64_C(0x1122334455667788);
  fields.epoch = UINT64_C(0x0102030405060708);
  memcpy(fields.job_uuid, k_job_uuid, 16);
  fields.rank = 3;
  fields.world_size = 8;
  fields.include_topology = 1;
  if (sagr_protocol_encode_ack(&fields, frame, sizeof(frame), &frame_size) !=
          SAGR_STATUS_SUCCESS ||
      send_frame(peer, frame, frame_size) != 0) {
    return -1;
  }
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  info->maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  memcpy(info->daemon_uuid, k_daemon_uuid, 16);
  info->connection_id = fields.connection_id;
  info->epoch = fields.epoch;
  info->negotiated_capabilities[0] = fields.selected_capabilities[0];
  return 0;
}

static int receive_signal_request(int peer, const sagr_instance_info_t *info,
                                  uint64_t *last_request_id,
                                  sagr_wire_signal_request_t *request,
                                  uint64_t *request_id) {
  static const uint8_t magic[8] = {'G', 'S', 'I', 'M', 'R', 'P', 'C', 0};
  uint8_t frame[SAGR_WIRE_SIGNAL_FRAME_BYTES];
  uint8_t copy[SAGR_WIRE_SIGNAL_FRAME_BYTES];
  const ssize_t received = recv(peer, frame, sizeof(frame), 0);
  const uint8_t *payload = frame + SAGR_WIRE_HEADER_BYTES;
  if (received != (ssize_t)sizeof(frame) ||
      memcmp(frame, magic, sizeof(magic)) != 0 || get_u16(frame + 8) != 1 ||
      get_u16(frame + 10) != 0 ||
      get_u16(frame + 12) != SAGR_WIRE_HEADER_BYTES ||
      get_u16(frame + 14) != SAGR_WIRE_MESSAGE_SIGNAL_REQUEST ||
      get_u32(frame + 16) != 0 ||
      get_u32(frame + 20) != SAGR_WIRE_SIGNAL_PAYLOAD_BYTES ||
      get_u64(frame + 24) == 0 || get_u32(frame + 68) != 0 ||
      get_u64(frame + 72) != 0 ||
      memcmp(frame + 32, info->daemon_uuid, 16) != 0 ||
      get_u64(frame + 48) != info->connection_id ||
      get_u64(frame + 56) != info->epoch) {
    return -1;
  }
  memcpy(copy, frame, sizeof(copy));
  memset(copy + 64, 0, 4);
  if (sagr_crc32c(copy, sizeof(copy)) != get_u32(frame + 64) ||
      get_u16(payload) != SAGR_SIGNAL_PROTOCOL_MAJOR ||
      get_u16(payload + 2) != SAGR_SIGNAL_PROTOCOL_MINOR ||
      get_u16(payload + 6) != 0 || !bytes_are_zero(payload + 48, 16)) {
    return -1;
  }
  *request_id = get_u64(frame + 24);
  if (*request_id <= *last_request_id) {
    return -1;
  }
  *last_request_id = *request_id;
  memset(request, 0, sizeof(*request));
  request->major = get_u16(payload);
  request->minor = get_u16(payload + 2);
  request->opcode = get_u16(payload + 4);
  request->flags = get_u16(payload + 6);
  request->signal_id = get_u64(payload + 8);
  request->generation = get_u64(payload + 16);
  request->sequence = get_u64(payload + 24);
  request->value_bits = get_u64(payload + 32);
  request->condition = get_u64(payload + 40);
  return 0;
}

static int send_signal_response(int peer, const sagr_instance_info_t *info,
                                uint64_t request_id, uint16_t message_type,
                                const sagr_wire_signal_response_t *response) {
  uint8_t frame[SAGR_WIRE_SIGNAL_FRAME_BYTES];
  size_t frame_size = 0;
  if (sagr_protocol_encode_signal_response(
          info, request_id, message_type, response, frame, sizeof(frame),
          &frame_size) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static void initialize_response(sagr_wire_signal_response_t *response,
                                uint16_t opcode, uint32_t status) {
  memset(response, 0, sizeof(*response));
  response->major = SAGR_SIGNAL_PROTOCOL_MAJOR;
  response->minor = SAGR_SIGNAL_PROTOCOL_MINOR;
  response->status = status;
  response->opcode = opcode;
}

static int predicate_satisfied(uint64_t condition, uint64_t value_bits,
                               uint64_t compare_bits) {
  int64_t value;
  int64_t compare;
  memcpy(&value, &value_bits, sizeof(value));
  memcpy(&compare, &compare_bits, sizeof(compare));
  switch (condition) {
    case SAGR_SIGNAL_CONDITION_EQ:
      return value == compare;
    case SAGR_SIGNAL_CONDITION_NE:
      return value != compare;
    case SAGR_SIGNAL_CONDITION_LT:
      return value < compare;
    case SAGR_SIGNAL_CONDITION_GTE:
      return value >= compare;
    default:
      return 0;
  }
}

static int receive_queue_request(int peer, const sagr_instance_info_t *info,
                                 uint64_t *last_request_id,
                                 sagr_wire_queue_request_t *request,
                                 uint64_t *request_id) {
  uint8_t frame[SAGR_WIRE_QUEUE_FRAME_BYTES];
  uint8_t copy[SAGR_WIRE_QUEUE_FRAME_BYTES];
  const ssize_t received = recv(peer, frame, sizeof(frame), 0);
  const uint8_t *payload = frame + SAGR_WIRE_HEADER_BYTES;
  if (received != (ssize_t)sizeof(frame) ||
      get_u16(frame + 14) != SAGR_WIRE_MESSAGE_QUEUE_REQUEST ||
      get_u16(frame + 8) != 1 || get_u16(frame + 10) != 0 ||
      get_u16(frame + 12) != SAGR_WIRE_HEADER_BYTES ||
      get_u32(frame + 16) != 0 ||
      get_u32(frame + 20) != SAGR_WIRE_QUEUE_PAYLOAD_BYTES ||
      get_u64(frame + 24) == 0 || get_u32(frame + 68) != 0 ||
      get_u64(frame + 72) != 0 ||
      memcmp(frame + 32, info->daemon_uuid, 16) != 0 ||
      get_u64(frame + 48) != info->connection_id ||
      get_u64(frame + 56) != info->epoch) {
    return -1;
  }
  memcpy(copy, frame, sizeof(copy));
  memset(copy + 64, 0, 4);
  if (sagr_crc32c(copy, sizeof(copy)) != get_u32(frame + 64) ||
      get_u16(payload) != SAGR_QUEUE_PROTOCOL_MAJOR ||
      get_u16(payload + 2) != SAGR_QUEUE_PROTOCOL_MINOR ||
      get_u16(payload + 6) != 0 || !bytes_are_zero(payload + 48, 16)) {
    return -1;
  }
  *request_id = get_u64(frame + 24);
  if (*request_id <= *last_request_id) {
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

static int send_pending_queue_completion(signal_server_t *server, int peer,
                                         const sagr_instance_info_t *info) {
  sagr_wire_queue_response_t completion;
  if (server->queue_pending == 0) {
    return -1;
  }
  memset(&completion, 0, sizeof(completion));
  completion.major = SAGR_QUEUE_PROTOCOL_MAJOR;
  completion.minor = SAGR_QUEUE_PROTOCOL_MINOR;
  completion.status = SAGR_WIRE_STATUS_INTERNAL;
  completion.opcode = SAGR_WIRE_QUEUE_OPCODE_DOORBELL;
  completion.queue_id = server->queue_id;
  completion.generation = server->queue_generation;
  completion.sequence = server->queue_sequence;
  completion.value = server->queue_kind;
  completion.error_code = UINT64_C(1);
  completion.sim_tick = server->queue_ack_tick + UINT64_C(1);
  if (send_queue_response(peer, info, server->queue_request_id,
                          SAGR_WIRE_MESSAGE_QUEUE_COMPLETION,
                          &completion) != 0) {
    return -1;
  }
  server->queue_pending = 0;
  return 0;
}

static int handle_queue_request(signal_server_t *server, int peer,
                                const sagr_instance_info_t *info,
                                uint64_t request_id,
                                const sagr_wire_queue_request_t *request) {
  sagr_wire_queue_response_t response;
  memset(&response, 0, sizeof(response));
  response.major = SAGR_QUEUE_PROTOCOL_MAJOR;
  response.minor = SAGR_QUEUE_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = request->opcode;
  response.sim_tick = server->next_tick++;
  if (request->opcode == SAGR_WIRE_QUEUE_OPCODE_CREATE) {
    server->queue_id = UINT64_C(0x1020304050607080);
    server->queue_generation = UINT64_C(0x8877665544332211);
    server->queue_exists = 1;
    response.queue_id = server->queue_id;
    response.generation = server->queue_generation;
    response.value = request->arg0;
  } else if (request->opcode == SAGR_WIRE_QUEUE_OPCODE_DOORBELL &&
             server->queue_exists != 0) {
    response.queue_id = server->queue_id;
    response.generation = server->queue_generation;
    response.sequence = request->sequence;
    server->queue_sequence = request->sequence;
    server->queue_request_id = request_id;
    server->queue_ack_tick = response.sim_tick;
    server->queue_kind = request->arg0;
    server->queue_pending = 1;
  } else if (request->opcode == SAGR_WIRE_QUEUE_OPCODE_DESTROY &&
             server->queue_exists != 0 && server->queue_pending == 0) {
    response.queue_id = server->queue_id;
    response.generation = server->queue_generation;
    server->queue_exists = 0;
  } else {
    return -1;
  }
  return send_queue_response(peer, info, request_id,
                             SAGR_WIRE_MESSAGE_QUEUE_ACK, &response);
}

static int handle_signal_request(signal_server_t *server, int peer,
                                 const sagr_instance_info_t *info,
                                 uint64_t request_id,
                                 const sagr_wire_signal_request_t *request) {
  sagr_wire_signal_response_t response;
  uint64_t tick = server->next_tick++;
  initialize_response(&response, request->opcode, SAGR_WIRE_STATUS_OK);
  response.signal_id = request->signal_id;
  response.generation = request->generation;
  response.sequence = request->sequence;
  response.sim_tick = tick;

  if (request->opcode == SAGR_WIRE_SIGNAL_OPCODE_CREATE) {
    if (server->behavior == SIGNAL_SERVER_FAILED_CREATE_ONCE &&
        server->create_failures++ == 0U) {
      response.status = SAGR_WIRE_STATUS_RESOURCE_EXHAUSTED;
      response.signal_id = 0;
      response.generation = 0;
      response.sequence = 0;
      response.sim_tick = 0;
      response.value_bits = 0;
      return send_signal_response(peer, info, request_id,
                                  SAGR_WIRE_MESSAGE_SIGNAL_ACK, &response);
    }
    if (server->behavior ==
            SIGNAL_SERVER_FOREIGN_UNTRIGGERED_DURING_STORE &&
        server->signal_create_count != 0U && server->wait_pending != 0) {
      server->foreign_wait_pending = 1;
      server->foreign_signal_id = server->signal_id;
      server->foreign_generation = server->generation;
      server->foreign_wait_sequence = server->wait_sequence;
      server->foreign_wait_request_id = server->wait_request_id;
    }
    server->signal_id =
        server->behavior == SIGNAL_SERVER_FOREIGN_UNTRIGGERED_DURING_STORE
            ? UINT64_C(7) + server->signal_create_count
            : UINT64_C(7);
    ++server->signal_create_count;
    if (server->generation == 0) {
      server->generation = 1;
    } else if (server->behavior != SIGNAL_SERVER_DUPLICATE_GENERATION) {
      ++server->generation;
    }
    server->signal_exists = 1;
    server->wait_pending = 0;
    server->value_bits = request->value_bits;
    response.signal_id = server->signal_id;
    response.generation = server->generation;
    response.sequence = 0;
    response.value_bits = server->value_bits;
    response.ready = 0;
    if (server->behavior == SIGNAL_SERVER_QUEUE_INTERLEAVE_ACK &&
        server->queue_pending != 0 &&
        send_pending_queue_completion(server, peer, info) != 0) {
      return -1;
    }
    return send_signal_response(peer, info, request_id,
                                SAGR_WIRE_MESSAGE_SIGNAL_ACK, &response);
  }

  if (!server->signal_exists || request->signal_id != server->signal_id ||
      request->generation != server->generation) {
    response.status = SAGR_WIRE_STATUS_PROTOCOL_STATE;
    response.sequence = request->sequence;
    response.value_bits = 0;
    response.ready = 0;
    response.sim_tick = 0;
    return send_signal_response(peer, info, request_id,
                                SAGR_WIRE_MESSAGE_SIGNAL_ACK, &response);
  }

  if (request->opcode == SAGR_WIRE_SIGNAL_OPCODE_LOAD) {
    response.value_bits = server->value_bits;
    response.sequence = 0;
    response.ready = 0;
  } else if (request->opcode == SAGR_WIRE_SIGNAL_OPCODE_STORE) {
    if (server->behavior ==
            SIGNAL_SERVER_FOREIGN_UNTRIGGERED_DURING_STORE &&
        server->foreign_wait_pending != 0) {
      sagr_wire_signal_response_t completion;
      initialize_response(&completion, SAGR_WIRE_SIGNAL_OPCODE_WAIT,
                          SAGR_WIRE_STATUS_OK);
      completion.signal_id = server->foreign_signal_id;
      completion.generation = server->foreign_generation;
      completion.sequence = server->foreign_wait_sequence;
      completion.value_bits = UINT64_C(42);
      completion.ready = 0;
      completion.sim_tick = tick + UINT64_C(1);
      server->foreign_wait_pending = 0;
      return send_signal_response(peer, info, server->foreign_wait_request_id,
                                  SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION,
                                  &completion);
    }
    server->value_bits = request->value_bits;
    response.value_bits = server->value_bits;
    response.sequence = 0;
    response.ready = 0;
    if (server->behavior == SIGNAL_SERVER_STORE_MAX_TICK) {
      response.sim_tick = UINT64_MAX;
    }
  } else if (request->opcode == SAGR_WIRE_SIGNAL_OPCODE_WAIT) {
    ++server->wait_requests;
    response.value_bits = server->value_bits;
    response.ready = (uint64_t)predicate_satisfied(
        request->condition, server->value_bits, request->value_bits);
    if (server->behavior == SIGNAL_SERVER_WAIT_MAX_TICK) {
      response.sim_tick = UINT64_MAX;
    }
    if (response.ready == 0) {
      server->wait_pending = 1;
      server->wait_sequence = request->sequence;
      server->wait_request_id = request_id;
      server->wait_condition = request->condition;
      server->wait_compare_bits = request->value_bits;
    }
  } else if (request->opcode == SAGR_WIRE_SIGNAL_OPCODE_DESTROY) {
    if (server->wait_pending != 0) {
      response.status = SAGR_WIRE_STATUS_BUSY;
      response.signal_id = request->signal_id;
      response.generation = request->generation;
      response.sequence = 0;
      response.value_bits = 0;
      response.ready = 0;
      response.sim_tick = 0;
      return send_signal_response(peer, info, request_id,
                                  SAGR_WIRE_MESSAGE_SIGNAL_ACK, &response);
    }
    server->signal_exists = 0;
    response.signal_id = request->signal_id;
    response.generation = request->generation;
    response.sequence = 0;
    response.value_bits = 0;
    response.ready = 0;
  }

  if (send_signal_response(peer, info, request_id,
                           SAGR_WIRE_MESSAGE_SIGNAL_ACK, &response) != 0) {
    return -1;
  }

  if (server->behavior == SIGNAL_SERVER_STORE_MAX_TICK &&
      request->opcode == SAGR_WIRE_SIGNAL_OPCODE_STORE) {
    return 0;
  }

  if (request->opcode == SAGR_WIRE_SIGNAL_OPCODE_STORE &&
      server->wait_pending != 0 &&
      predicate_satisfied(server->wait_condition, server->value_bits,
                          server->wait_compare_bits)) {
    sagr_wire_signal_response_t completion;
    initialize_response(&completion, SAGR_WIRE_SIGNAL_OPCODE_WAIT,
                        SAGR_WIRE_STATUS_OK);
    completion.signal_id = server->signal_id;
    completion.generation = server->generation;
    completion.sequence = server->wait_sequence;
    completion.value_bits = server->value_bits;
    completion.ready = 0;
    completion.sim_tick = tick + UINT64_C(1);
    if (server->behavior == SIGNAL_SERVER_BAD_COMPLETION) {
      completion.value_bits = UINT64_C(0xdeadbeef);
    }
    if (send_signal_response(peer, info, server->wait_request_id,
                             SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION,
                             &completion) != 0) {
      return -1;
    }
    server->wait_pending = 0;
  }
  if (request->opcode == SAGR_WIRE_SIGNAL_OPCODE_WAIT && response.ready != 0) {
    sagr_wire_signal_response_t completion;
    if (server->behavior == SIGNAL_SERVER_QUEUE_INTERLEAVE_COMPLETION &&
        server->queue_pending != 0 &&
        send_pending_queue_completion(server, peer, info) != 0) {
      return -1;
    }
    initialize_response(&completion, SAGR_WIRE_SIGNAL_OPCODE_WAIT,
                        SAGR_WIRE_STATUS_OK);
    completion.signal_id = server->signal_id;
    completion.generation = server->generation;
    completion.sequence = request->sequence;
    completion.value_bits = response.value_bits;
    completion.ready = 0;
    completion.sim_tick = tick + UINT64_C(1);
    if (send_signal_response(peer, info, request_id,
                             SAGR_WIRE_MESSAGE_SIGNAL_COMPLETION,
                             &completion) != 0) {
      return -1;
    }
  }
  return 0;
}

static void *signal_server_main(void *opaque) {
  signal_server_t *server = (signal_server_t *)opaque;
  struct sockaddr_un address;
  socklen_t address_size = sizeof(address);
  int peer;
  uint64_t hello_request_id = 0;
  uint8_t client_nonce[16];
  sagr_instance_info_t info;
  peer = accept(server->listener, (struct sockaddr *)&address, &address_size);
  if (peer < 0) {
    server->thread_error = 1;
    return NULL;
  }
  (void)close(server->listener);
  server->listener = -1;
  if (receive_hello(peer, &hello_request_id, client_nonce) != 0 ||
      send_handshake_ack(server, peer, hello_request_id, client_nonce,
                         &info) != 0) {
    server->thread_error = 1;
    (void)close(peer);
    return NULL;
  }
  if (server->behavior == SIGNAL_SERVER_NO_CAPABILITY) {
    (void)close(peer);
    return NULL;
  }
  server->last_request_id = hello_request_id;
  server->next_tick = UINT64_C(100);
  for (;;) {
    uint8_t header[16];
    uint64_t request_id = 0;
    const ssize_t count = recv(peer, header, sizeof(header), MSG_PEEK);
    if (count == 0) {
      break;
    }
    if (count != (ssize_t)sizeof(header)) {
      server->thread_error = 1;
      break;
    }
    if (get_u16(header + 14) == SAGR_WIRE_MESSAGE_SIGNAL_REQUEST) {
      sagr_wire_signal_request_t request;
      if (receive_signal_request(peer, &info, &server->last_request_id,
                                 &request, &request_id) != 0 ||
          handle_signal_request(server, peer, &info, request_id, &request) !=
              0) {
        server->thread_error = 1;
        break;
      }
    } else if (get_u16(header + 14) == SAGR_WIRE_MESSAGE_QUEUE_REQUEST) {
      sagr_wire_queue_request_t request;
      if (receive_queue_request(peer, &info, &server->last_request_id,
                                &request, &request_id) != 0 ||
          handle_queue_request(server, peer, &info, request_id, &request) != 0) {
        server->thread_error = 1;
        break;
      }
    } else {
      server->thread_error = 1;
      break;
    }
  }
  (void)close(peer);
  return NULL;
}

static int start_server(signal_server_t *server,
                        enum signal_server_behavior behavior) {
  struct sockaddr_un address;
  memset(server, 0, sizeof(*server));
  server->listener = -1;
  server->behavior = behavior;
  if (snprintf(server->directory, sizeof(server->directory),
               "/tmp/sagr-signal-XXXXXX") >=
      (int)sizeof(server->directory) || mkdtemp(server->directory) == NULL ||
      snprintf(server->endpoint, sizeof(server->endpoint), "%s/socket",
               server->directory) >= (int)sizeof(server->endpoint)) {
    return -1;
  }
  server->listener = socket(AF_UNIX, SOCK_SEQPACKET, 0);
  if (server->listener < 0) {
    return -1;
  }
  memset(&address, 0, sizeof(address));
  address.sun_family = AF_UNIX;
  if (strlen(server->endpoint) >= sizeof(address.sun_path)) {
    (void)close(server->listener);
    server->listener = -1;
    (void)unlink(server->endpoint);
    (void)rmdir(server->directory);
    return -1;
  }
  memcpy(address.sun_path, server->endpoint, strlen(server->endpoint) + 1U);
  if (bind(server->listener, (struct sockaddr *)&address,
           (socklen_t)(offsetof(struct sockaddr_un, sun_path) +
                       strlen(server->endpoint) + 1U)) != 0 ||
      listen(server->listener, 1) != 0 ||
      pthread_create(&server->thread, NULL, signal_server_main, server) != 0) {
    if (server->listener >= 0) {
      (void)close(server->listener);
      server->listener = -1;
    }
    (void)unlink(server->endpoint);
    (void)rmdir(server->directory);
    return -1;
  }
  return 0;
}

static void stop_server(signal_server_t *server) {
  if (server->listener >= 0) {
    (void)close(server->listener);
    server->listener = -1;
  }
  (void)pthread_join(server->thread, NULL);
  (void)unlink(server->endpoint);
  (void)rmdir(server->directory);
}

static int open_signal_instance(const signal_server_t *server,
                                sagr_instance_t *instance,
                                sagr_error_info_t *error) {
  sagr_instance_open_options_t options;
  if (sagr_instance_open_options_init(&options, (uint32_t)sizeof(options)) !=
      SAGR_STATUS_SUCCESS) {
    return -1;
  }
  options.open_timeout_ns = UINT64_C(2000000000);
  options.offered_capabilities[0] |= SAGR_CAPABILITY_SIGNAL_MASK;
  options.required_capabilities[0] |= SAGR_CAPABILITY_SIGNAL_MASK;
  if (server->behavior == SIGNAL_SERVER_QUEUE_INTERLEAVE_ACK ||
      server->behavior == SIGNAL_SERVER_QUEUE_INTERLEAVE_COMPLETION) {
    options.offered_capabilities[0] |= SAGR_CAPABILITY_QUEUE_MASK;
    options.required_capabilities[0] |= SAGR_CAPABILITY_QUEUE_MASK;
  }
  return sagr_instance_open(server->endpoint, &options, instance, error,
                             (uint32_t)sizeof(*error));
}

static int test_lifecycle_retry_reuse(void) {
  signal_server_t server;
  sagr_instance_t instance = NULL;
  sagr_signal_t signal = NULL;
  sagr_signal_t reuse_signal = NULL;
  sagr_signal_create_options_t create_options;
  sagr_signal_operation_options_t operation;
  sagr_signal_operation_options_t short_operation;
  sagr_signal_info_t info;
  sagr_signal_info_t reuse_info;
  sagr_signal_wait_result_t result;
  sagr_error_info_t error;
  int64_t value = 0;
  sagr_status_t status;
  int failures = 0;
  if (start_server(&server, SIGNAL_SERVER_LIFECYCLE) != 0 ||
      open_signal_instance(&server, &instance, &error) != SAGR_STATUS_SUCCESS) {
    stop_server(&server);
    return 1;
  }
  status = sagr_signal_create_options_init(&create_options,
                                           (uint32_t)sizeof(create_options));
  if (status == SAGR_STATUS_SUCCESS) {
    status = sagr_signal_operation_options_init(&operation,
                                                (uint32_t)sizeof(operation));
  }
  if (status == SAGR_STATUS_SUCCESS) {
    status = sagr_signal_operation_options_init(&short_operation,
                                                (uint32_t)sizeof(short_operation));
  }
  create_options.initial_value = -7;
  operation.timeout_ns = UINT64_C(1000000000);
  short_operation.timeout_ns = UINT64_C(10000000);
  status = sagr_signal_create(instance, &create_options, &operation, &signal,
                              &info, (uint32_t)sizeof(info), &error,
                              (uint32_t)sizeof(error));
  if (status != SAGR_STATUS_SUCCESS || info.signal_id != 7 ||
      info.generation == 0 || info.value != -7) {
    failures = 1;
  }
  if (!failures &&
      sagr_signal_load(signal, &operation, &value, &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    failures = 1;
  }
  if (!failures && value != -7) {
    failures = 1;
  }
  memset(&result, 0xa5, sizeof(result));
  if (!failures &&
      sagr_signal_wait(signal, SAGR_SIGNAL_CONDITION_GTE, 0,
                       &short_operation, &result, (uint32_t)sizeof(result),
                       &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_TIMED_OUT) {
    failures = 1;
  }
  if (!failures && error.wire_status != -1) {
    fprintf(stderr,
            "local signal wait timeout exposed a daemon wire status: %" PRId32
            "\n",
            error.wire_status);
    failures = 1;
  }
  if (!failures &&
      (result.struct_size != sizeof(result) || result.signal_id != 0 ||
       result.generation != 0 || result.sequence != 0 ||
       result.observed_value != 0 || result.completion_tick != 0)) {
    failures = 1;
  }
  if (!failures &&
      sagr_signal_store(signal, 42, &operation, &error,
                        (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    failures = 1;
  }
  memset(&result, 0, sizeof(result));
  if (!failures &&
      sagr_signal_wait(signal, SAGR_SIGNAL_CONDITION_GTE, 0, &short_operation,
                       &result, (uint32_t)sizeof(result), &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    failures = 1;
  }
  if (!failures &&
      (result.observed_value != 42 || result.sequence == 0 ||
       result.completion_tick <= result.admission_tick)) {
    failures = 1;
  }
  if (!failures &&
      sagr_signal_load(signal, &operation, &value, &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    failures = 1;
  }
  if (!failures && value != 42) {
    failures = 1;
  }
  if (!failures &&
      sagr_signal_destroy(&signal, &operation, &error,
                          (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    failures = 1;
  }
  if (!failures &&
      sagr_signal_create(instance, &create_options, &operation, &reuse_signal,
                         &reuse_info, (uint32_t)sizeof(reuse_info), &error,
                         (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    failures = 1;
  }
  if (!failures &&
      (reuse_info.signal_id != info.signal_id ||
       reuse_info.generation <= info.generation || reuse_info.value != -7)) {
    failures = 1;
  }
  if (!failures &&
      sagr_signal_destroy(&reuse_signal, &operation, &error,
                          (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    failures = 1;
  }
  (void)sagr_instance_close(&instance);
  stop_server(&server);
  if (server.thread_error != 0) {
    failures = 1;
  }
  return failures;
}

static int test_busy_and_poison(void) {
  signal_server_t server;
  sagr_instance_t instance = NULL;
  sagr_signal_t signal = NULL;
  sagr_signal_create_options_t create_options;
  sagr_signal_operation_options_t operation;
  sagr_signal_operation_options_t short_operation;
  sagr_signal_wait_result_t result;
  sagr_error_info_t error;
  int64_t value = 0;
  int failures = 0;
  if (start_server(&server, SIGNAL_SERVER_BAD_COMPLETION) != 0 ||
      open_signal_instance(&server, &instance, &error) != SAGR_STATUS_SUCCESS) {
    stop_server(&server);
    return 1;
  }
  (void)sagr_signal_create_options_init(&create_options,
                                        (uint32_t)sizeof(create_options));
  (void)sagr_signal_operation_options_init(&operation,
                                           (uint32_t)sizeof(operation));
  (void)sagr_signal_operation_options_init(&short_operation,
                                           (uint32_t)sizeof(short_operation));
  create_options.initial_value = -7;
  operation.timeout_ns = UINT64_C(1000000000);
  short_operation.timeout_ns = UINT64_C(10000000);
  if (sagr_signal_create(instance, &create_options, &operation, &signal, NULL,
                         0, &error, (uint32_t)sizeof(error)) !=
      SAGR_STATUS_SUCCESS ||
      sagr_signal_wait(signal, SAGR_SIGNAL_CONDITION_GTE, 0, &short_operation,
                       &result, (uint32_t)sizeof(result), &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_TIMED_OUT) {
    failures = 1;
  }
  if (!failures &&
      sagr_signal_wait(signal, SAGR_SIGNAL_CONDITION_LT, 0, &short_operation,
                       &result, (uint32_t)sizeof(result), &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_BUSY) {
    failures = 1;
  }
  if (!failures &&
      sagr_signal_destroy(&signal, &operation, &error,
                          (uint32_t)sizeof(error)) != SAGR_STATUS_BUSY) {
    failures = 1;
  }
  if (!failures &&
      sagr_signal_store(signal, 42, &operation, &error,
                        (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    failures = 1;
  }
  if (!failures &&
      sagr_signal_wait(signal, SAGR_SIGNAL_CONDITION_GTE, 0, &operation,
                       &result, (uint32_t)sizeof(result), &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_PROTOCOL_ERROR) {
    failures = 1;
  }
  if (!failures &&
      sagr_signal_load(signal, &operation, &value, &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_CONNECTION_LOST) {
    failures = 1;
  }
  (void)sagr_instance_close(&instance);
  stop_server(&server);
  return failures || server.thread_error != 0;
}

static int test_capability_and_tick_rejection(void) {
  signal_server_t server;
  sagr_instance_t instance = NULL;
  sagr_error_info_t error;
  int failures = 0;
  if (start_server(&server, SIGNAL_SERVER_NO_CAPABILITY) != 0) {
    return 1;
  }
  if (open_signal_instance(&server, &instance, &error) !=
      SAGR_STATUS_CAPABILITY_MISMATCH || instance != NULL) {
    failures = 1;
    if (instance != NULL) {
      (void)sagr_instance_close(&instance);
    }
  }
  stop_server(&server);
  if (server.thread_error != 0) {
    failures = 1;
  }

  if (start_server(&server, SIGNAL_SERVER_WAIT_MAX_TICK) != 0 ||
      open_signal_instance(&server, &instance, &error) != SAGR_STATUS_SUCCESS) {
    stop_server(&server);
    return 1;
  }
  {
    sagr_signal_t signal = NULL;
    sagr_signal_create_options_t create_options;
    sagr_signal_operation_options_t operation;
    sagr_signal_wait_result_t result;
    (void)sagr_signal_create_options_init(&create_options,
                                          (uint32_t)sizeof(create_options));
    (void)sagr_signal_operation_options_init(&operation,
                                             (uint32_t)sizeof(operation));
    create_options.initial_value = -7;
    operation.timeout_ns = UINT64_C(1000000000);
    if (sagr_signal_create(instance, &create_options, &operation, &signal,
                           NULL, 0, &error, (uint32_t)sizeof(error)) !=
            SAGR_STATUS_SUCCESS ||
        sagr_signal_wait(signal, SAGR_SIGNAL_CONDITION_GTE, 0, &operation,
                         &result, (uint32_t)sizeof(result), &error,
                         (uint32_t)sizeof(error)) != SAGR_STATUS_PROTOCOL_ERROR) {
      failures = 1;
    }
  }
  (void)sagr_instance_close(&instance);
  stop_server(&server);
  if (server.thread_error != 0) {
    failures = 1;
  }

  if (start_server(&server, SIGNAL_SERVER_STORE_MAX_TICK) != 0 ||
      open_signal_instance(&server, &instance, &error) != SAGR_STATUS_SUCCESS) {
    stop_server(&server);
    return 1;
  }
  {
    sagr_signal_t signal = NULL;
    sagr_signal_create_options_t create_options;
    sagr_signal_operation_options_t operation;
    sagr_signal_operation_options_t short_operation;
    sagr_signal_wait_result_t result;
    sagr_signal_info_t info;
    int64_t value = 0;
    (void)sagr_signal_create_options_init(&create_options,
                                          (uint32_t)sizeof(create_options));
    (void)sagr_signal_operation_options_init(&operation,
                                             (uint32_t)sizeof(operation));
    (void)sagr_signal_operation_options_init(&short_operation,
                                             (uint32_t)sizeof(short_operation));
    create_options.initial_value = -7;
    operation.timeout_ns = UINT64_C(1000000000);
    short_operation.timeout_ns = UINT64_C(10000000);
    if (sagr_signal_create(instance, &create_options, &operation, &signal,
                           NULL, 0, &error, (uint32_t)sizeof(error)) !=
            SAGR_STATUS_SUCCESS ||
        sagr_signal_wait(signal, SAGR_SIGNAL_CONDITION_GTE, 0,
                         &short_operation, &result,
                         (uint32_t)sizeof(result), &error,
                         (uint32_t)sizeof(error)) != SAGR_STATUS_TIMED_OUT ||
        sagr_signal_store(signal, 42, &operation, &error,
                          (uint32_t)sizeof(error)) !=
            SAGR_STATUS_PROTOCOL_ERROR ||
        sagr_signal_get_info(signal, &info, (uint32_t)sizeof(info)) !=
            SAGR_STATUS_SUCCESS ||
        info.value != -7 ||
        sagr_signal_load(signal, &operation, &value, &error,
                         (uint32_t)sizeof(error)) !=
            SAGR_STATUS_CONNECTION_LOST) {
      failures = 1;
    }
  }
  (void)sagr_instance_close(&instance);
  stop_server(&server);
  return failures || server.thread_error != 0;
}

static int test_failed_ack_keeps_session_reusable(void) {
  signal_server_t server;
  sagr_instance_t instance = NULL;
  sagr_signal_t signal = NULL;
  sagr_signal_create_options_t create_options;
  sagr_signal_operation_options_t operation;
  sagr_error_info_t error;
  int failures = 0;
  if (start_server(&server, SIGNAL_SERVER_FAILED_CREATE_ONCE) != 0 ||
      open_signal_instance(&server, &instance, &error) != SAGR_STATUS_SUCCESS) {
    stop_server(&server);
    return 1;
  }
  (void)sagr_signal_create_options_init(&create_options,
                                        (uint32_t)sizeof(create_options));
  (void)sagr_signal_operation_options_init(&operation,
                                           (uint32_t)sizeof(operation));
  create_options.initial_value = -7;
  operation.timeout_ns = UINT64_C(1000000000);
  if (sagr_signal_create(instance, &create_options, &operation, &signal, NULL,
                         0, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_OUT_OF_RESOURCES ||
      signal != NULL || error.wire_status != SAGR_WIRE_STATUS_RESOURCE_EXHAUSTED ||
      sagr_signal_create(instance, &create_options, &operation, &signal, NULL,
                         0, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      signal == NULL ||
      sagr_signal_destroy(&signal, &operation, &error,
                          (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    failures = 1;
  }
  (void)sagr_instance_close(&instance);
  stop_server(&server);
  return failures || server.thread_error != 0;
}

static int test_duplicate_generation_poisons(void) {
  signal_server_t server;
  sagr_instance_t instance = NULL;
  sagr_signal_t signal = NULL;
  sagr_signal_create_options_t create_options;
  sagr_signal_operation_options_t operation;
  sagr_error_info_t error;
  int failures = 0;
  if (start_server(&server, SIGNAL_SERVER_DUPLICATE_GENERATION) != 0 ||
      open_signal_instance(&server, &instance, &error) != SAGR_STATUS_SUCCESS) {
    stop_server(&server);
    return 1;
  }
  (void)sagr_signal_create_options_init(&create_options,
                                        (uint32_t)sizeof(create_options));
  (void)sagr_signal_operation_options_init(&operation,
                                           (uint32_t)sizeof(operation));
  operation.timeout_ns = UINT64_C(1000000000);
  if (sagr_signal_create(instance, &create_options, &operation, &signal, NULL,
                         0, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_signal_destroy(&signal, &operation, &error,
                          (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      sagr_signal_create(instance, &create_options, &operation, &signal, NULL,
                         0, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_PROTOCOL_ERROR ||
      signal != NULL ||
      sagr_signal_create(instance, &create_options, &operation, &signal, NULL,
                         0, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_CONNECTION_LOST) {
    failures = 1;
  }
  (void)sagr_instance_close(&instance);
  stop_server(&server);
  return failures || server.thread_error != 0;
}

static int run_queue_error_interleave(enum signal_server_behavior behavior) {
  signal_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_signal_t signal = NULL;
  sagr_queue_create_options_t queue_create;
  sagr_queue_operation_options_t queue_operation;
  sagr_queue_completion_t queue_completion;
  sagr_signal_create_options_t signal_create;
  sagr_signal_operation_options_t signal_operation;
  sagr_signal_wait_result_t wait_result;
  sagr_error_info_t error;
  uint64_t sequence = 0;
  int failures = 0;
  if (start_server(&server, behavior) != 0 ||
      open_signal_instance(&server, &instance, &error) != SAGR_STATUS_SUCCESS) {
    stop_server(&server);
    return 1;
  }
  (void)sagr_queue_create_options_init(&queue_create,
                                       (uint32_t)sizeof(queue_create));
  (void)sagr_queue_operation_options_init(&queue_operation,
                                          (uint32_t)sizeof(queue_operation));
  (void)sagr_signal_create_options_init(&signal_create,
                                        (uint32_t)sizeof(signal_create));
  (void)sagr_signal_operation_options_init(&signal_operation,
                                           (uint32_t)sizeof(signal_operation));
  queue_create.depth = 4;
  queue_operation.timeout_ns = UINT64_C(1000000000);
  signal_operation.timeout_ns = UINT64_C(1000000000);
  signal_create.initial_value = 0;
  if (sagr_queue_create(instance, &queue_create, &queue_operation, &queue, NULL,
                        0, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_queue_ring_doorbell(queue, SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST,
                               &queue_operation, &sequence, &error,
                               (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      sagr_signal_create(instance, &signal_create, &signal_operation, &signal,
                         NULL, 0, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS) {
    failures = 1;
  }
  if (!failures &&
      behavior == SIGNAL_SERVER_QUEUE_INTERLEAVE_COMPLETION &&
      sagr_signal_wait(signal, SAGR_SIGNAL_CONDITION_GTE, 0,
                       &signal_operation, &wait_result,
                       (uint32_t)sizeof(wait_result), &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    failures = 1;
  }
  memset(&queue_completion, 0, sizeof(queue_completion));
  if (!failures &&
      sagr_queue_wait(queue, sequence, &queue_operation, &queue_completion,
                      (uint32_t)sizeof(queue_completion), &error,
                      (uint32_t)sizeof(error)) != SAGR_STATUS_INTERNAL_ERROR) {
    failures = 1;
  }
  if (!failures &&
      (queue_completion.status != SAGR_STATUS_INTERNAL_ERROR ||
       queue_completion.wire_status != SAGR_WIRE_STATUS_INTERNAL ||
       queue_completion.value != SAGR_QUEUE_COMMAND_CONTROL_ERROR_TEST ||
       queue_completion.error_code != UINT64_C(1))) {
    failures = 1;
  }
  if (!failures &&
      sagr_signal_destroy(&signal, &signal_operation, &error,
                          (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    failures = 1;
  }
  if (!failures &&
      sagr_queue_destroy(&queue, &queue_operation, &error,
                         (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    failures = 1;
  }
  (void)sagr_instance_close(&instance);
  stop_server(&server);
  return failures || server.thread_error != 0;
}

static int test_foreign_untriggered_completion_poisons(void) {
  signal_server_t server;
  sagr_instance_t instance = NULL;
  sagr_signal_t signal_a = NULL;
  sagr_signal_t signal_b = NULL;
  sagr_signal_create_options_t create_options;
  sagr_signal_operation_options_t operation;
  sagr_signal_operation_options_t short_operation;
  sagr_signal_wait_result_t wait_result;
  sagr_error_info_t error;
  int64_t value = 0;
  int failures = 0;
  if (start_server(&server,
                   SIGNAL_SERVER_FOREIGN_UNTRIGGERED_DURING_STORE) != 0 ||
      open_signal_instance(&server, &instance, &error) != SAGR_STATUS_SUCCESS) {
    stop_server(&server);
    return 1;
  }
  (void)sagr_signal_create_options_init(&create_options,
                                        (uint32_t)sizeof(create_options));
  (void)sagr_signal_operation_options_init(&operation,
                                           (uint32_t)sizeof(operation));
  (void)sagr_signal_operation_options_init(&short_operation,
                                           (uint32_t)sizeof(short_operation));
  create_options.initial_value = -7;
  operation.timeout_ns = UINT64_C(1000000000);
  short_operation.timeout_ns = UINT64_C(10000000);
  if (sagr_signal_create(instance, &create_options, &operation, &signal_b,
                         NULL, 0, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_signal_wait(signal_b, SAGR_SIGNAL_CONDITION_GTE, 0,
                       &short_operation, &wait_result,
                       (uint32_t)sizeof(wait_result), &error,
                       (uint32_t)sizeof(error)) != SAGR_STATUS_TIMED_OUT ||
      sagr_signal_create(instance, &create_options, &operation, &signal_a,
                         NULL, 0, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_signal_store(signal_a, 42, &operation, &error,
                        (uint32_t)sizeof(error)) !=
          SAGR_STATUS_PROTOCOL_ERROR ||
      sagr_signal_load(signal_a, &operation, &value, &error,
                       (uint32_t)sizeof(error)) !=
          SAGR_STATUS_CONNECTION_LOST) {
    failures = 1;
  }
  (void)sagr_instance_close(&instance);
  stop_server(&server);
  return failures || server.thread_error != 0;
}

#ifdef SAGR_SIGNAL_CLI_PATH
static int test_signal_cli_json(void) {
  signal_server_t server;
  char output[8192];
  size_t output_size = 0;
  int output_pipe[2] = {-1, -1};
  int child_status = 0;
  pid_t child;
  int failure = 0;
  if (start_server(&server, SIGNAL_SERVER_LIFECYCLE) != 0 ||
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
    execl(SAGR_SIGNAL_CLI_PATH, "sagr-handshake", "--endpoint",
          server.endpoint, "--signal-initial", "-7",
          "--signal-wait-condition", "gte", "--signal-wait-compare", "0",
          "--signal-wait-timeout-ms", "50", "--signal-store", "42",
          "--signal-reuse", "--timeout-ms", "1000", (char *)NULL);
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
    if (getenv("SAGR_TEST_PRINT_CLI_JSON") != NULL) {
      fputs(output, stdout);
    }
    if (waitpid(child, &child_status, 0) != child ||
        !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0 ||
        strstr(output, "\"signal\":{\"status\":0") == NULL ||
        strstr(output, "\"signal_id\":\"0x0000000000000007\"") == NULL ||
        strstr(output, "\"initial_value\":-7") == NULL ||
        strstr(output, "\"load_before\":-7") == NULL ||
        strstr(output, "\"condition\":\"gte\"") == NULL ||
        strstr(output, "\"compare\":0") == NULL ||
        strstr(output, "\"first_status\":11") == NULL ||
        strstr(output, "\"completion_status\":0") == NULL ||
        strstr(output, "\"observed_value\":42") == NULL ||
        strstr(output, "\"sequence\":\"0x0000000000000001\"") == NULL ||
        strstr(output, "\"retried_without_send\":true") == NULL ||
        strstr(output, "\"stored_value\":42") == NULL ||
        strstr(output, "\"load_after\":42") == NULL ||
        strstr(output, "\"destroyed\":true") == NULL ||
        strstr(output, "\"reuse\":{\"signal_id\":"
                       "\"0x0000000000000007\"") == NULL ||
        strstr(output, "\"generation\":\"0x0000000000000002\"") == NULL ||
        server.wait_requests != 1U) {
      fprintf(stderr, "signal CLI JSON gate failed: %s\n", output);
      failure = 1;
    }
  }
  (void)close(output_pipe[0]);
  stop_server(&server);
  return failure || server.thread_error != 0;
}
#endif

int main(void) {
  int failures = 0;
  {
    const int result = test_lifecycle_retry_reuse();
    if (result != 0) {
      fprintf(stderr, "lifecycle test failed (%d)\n", result);
    }
    failures += result;
  }
#ifdef SAGR_SIGNAL_CLI_PATH
  {
    const int result = test_signal_cli_json();
    if (result != 0) {
      fprintf(stderr, "signal CLI test failed (%d)\n", result);
    }
    failures += result;
  }
#endif
  {
    const int result = test_busy_and_poison();
    if (result != 0) {
      fprintf(stderr, "busy/poison test failed (%d)\n", result);
    }
    failures += result;
  }
  {
    const int result = test_capability_and_tick_rejection();
    if (result != 0) {
      fprintf(stderr, "capability/tick test failed (%d)\n", result);
    }
    failures += result;
  }
  {
    const int result = test_failed_ack_keeps_session_reusable();
    if (result != 0) {
      fprintf(stderr, "failed ACK reuse test failed (%d)\n", result);
    }
    failures += result;
  }
  {
    const int result = test_duplicate_generation_poisons();
    if (result != 0) {
      fprintf(stderr, "duplicate generation test failed (%d)\n", result);
    }
    failures += result;
  }
  {
    const int result =
        run_queue_error_interleave(SIGNAL_SERVER_QUEUE_INTERLEAVE_ACK);
    if (result != 0) {
      fprintf(stderr, "queue error interleave at ACK failed (%d)\n", result);
    }
    failures += result;
  }
  {
    const int result = run_queue_error_interleave(
        SIGNAL_SERVER_QUEUE_INTERLEAVE_COMPLETION);
    if (result != 0) {
      fprintf(stderr, "queue error interleave at completion failed (%d)\n",
              result);
    }
    failures += result;
  }
  {
    const int result = test_foreign_untriggered_completion_poisons();
    if (result != 0) {
      fprintf(stderr, "foreign untriggered completion test failed (%d)\n",
              result);
    }
    failures += result;
  }
  if (failures != 0) {
    fprintf(stderr, "signal tests failed: %d\n", failures);
  }
  return failures == 0 ? 0 : 1;
}
