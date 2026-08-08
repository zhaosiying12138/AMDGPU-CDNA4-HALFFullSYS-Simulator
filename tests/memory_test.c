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
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
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

#define MOCK_ALLOCATION_ID UINT64_C(7)
#define MOCK_SIMULATED_VA UINT64_C(0x0000100300000000)
#define MOCK_QUEUE_ID UINT64_C(0x1020304050607080)
#define MOCK_QUEUE_GENERATION UINT64_C(0x8877665544332211)

enum memory_server_behavior {
  MEMORY_SERVER_SUCCESS,
  MEMORY_SERVER_NO_CAPABILITY,
  MEMORY_SERVER_BAD_D2H_CRC,
  MEMORY_SERVER_BAD_D2H_SETUID,
  MEMORY_SERVER_BAD_D2H_SETGID,
  MEMORY_SERVER_BAD_D2H_STICKY,
  MEMORY_SERVER_BAD_D2H_SEALS,
  MEMORY_SERVER_POST_ACK_CANCEL,
  MEMORY_SERVER_POST_ACK_DEADLINE_CANCEL,
  MEMORY_SERVER_ALLOC_TIMEOUT,
  MEMORY_SERVER_INTERLEAVED_COMPLETION
};

typedef struct memory_server {
  char directory[128];
  char endpoint[160];
  int listener;
  pthread_t thread;
  enum memory_server_behavior behavior;
  int cancel_write_fd;
  uint64_t post_ack_deadline_ns;
  int thread_error;
} memory_server_t;

typedef struct received_request {
  uint16_t message_type;
  uint64_t request_id;
  int descriptor;
  sagr_wire_memory_request_t memory;
  sagr_wire_queue_request_t queue;
} received_request_t;

typedef struct server_memory_state {
  uint8_t *bytes;
  uint64_t size;
  uint64_t generation;
  uint64_t tick;
  int allocated;
  int corrupted_d2h_sent;
} server_memory_state_t;

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

static int send_frame(int peer, const uint8_t *frame, size_t frame_size) {
  return send(peer, frame, frame_size, MSG_NOSIGNAL) == (ssize_t)frame_size
             ? 0
             : -1;
}

static int send_memory_response(
    int peer, const sagr_instance_info_t *info, uint64_t request_id,
    const sagr_wire_memory_response_t *response) {
  uint8_t frame[SAGR_WIRE_MEMORY_FRAME_BYTES];
  size_t frame_size = 0;
  if (sagr_protocol_encode_memory_response(
          info, request_id, response, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
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

static int send_handshake_ack(int peer, const uint8_t *hello,
                              ssize_t hello_size, uint64_t capabilities,
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
      sagr_protocol_encode_ack(&fields, ack, sizeof(ack), &ack_size) !=
          SAGR_STATUS_SUCCESS ||
      send_frame(peer, ack, ack_size) != 0) {
    return -1;
  }
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  info->maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  info->negotiated_capabilities[0] = capabilities;
  memcpy(info->daemon_uuid, fields.daemon_uuid, 16);
  info->connection_id = fields.connection_id;
  info->epoch = fields.epoch;
  *last_request_id = fields.request_id;
  return 0;
}

static int receive_request(int peer, const sagr_instance_info_t *info,
                           uint64_t *last_request_id,
                           received_request_t *request) {
  static const uint8_t magic[8] = {'G', 'S', 'I', 'M', 'R', 'P', 'C', 0};
  uint8_t frame[SAGR_WIRE_MEMORY_FRAME_BYTES];
  uint8_t crc_frame[SAGR_WIRE_MEMORY_FRAME_BYTES];
  unsigned char control[CMSG_SPACE(sizeof(int) * 2U)];
  struct iovec vector;
  struct msghdr message;
  struct cmsghdr *control_message;
  const uint8_t *payload = frame + SAGR_WIRE_HEADER_BYTES;
  ssize_t received;
  int descriptor_count = 0;

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
  if (received != (ssize_t)sizeof(frame) ||
      (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0) {
    return -1;
  }
  for (control_message = CMSG_FIRSTHDR(&message); control_message != NULL;
       control_message = CMSG_NXTHDR(&message, control_message)) {
    int descriptor;
    if (control_message->cmsg_level != SOL_SOCKET ||
        control_message->cmsg_type != SCM_RIGHTS ||
        control_message->cmsg_len != CMSG_LEN(sizeof(descriptor))) {
      return -1;
    }
    memcpy(&descriptor, CMSG_DATA(control_message), sizeof(descriptor));
    ++descriptor_count;
    if (descriptor_count == 1) {
      request->descriptor = descriptor;
    } else {
      (void)close(descriptor);
    }
  }
  if (descriptor_count > 1) {
    (void)close(request->descriptor);
    request->descriptor = -1;
    return -1;
  }

  memcpy(crc_frame, frame, sizeof(frame));
  memset(crc_frame + 64, 0, 4);
  request->message_type = get_u16(frame + 14);
  request->request_id = get_u64(frame + 24);
  if (memcmp(frame, magic, sizeof(magic)) != 0 || get_u16(frame + 8) != 1 ||
      get_u16(frame + 10) != 0 ||
      get_u16(frame + 12) != SAGR_WIRE_HEADER_BYTES ||
      (request->message_type != SAGR_WIRE_MESSAGE_MEMORY_REQUEST &&
       request->message_type != SAGR_WIRE_MESSAGE_QUEUE_REQUEST) ||
      get_u32(frame + 16) != 0 ||
      get_u32(frame + 20) != SAGR_WIRE_MEMORY_PAYLOAD_BYTES ||
      get_u32(frame + 68) != 0 || get_u64(frame + 72) != 0 ||
      get_u32(frame + 64) != sagr_crc32c(crc_frame, sizeof(crc_frame)) ||
      memcmp(frame + 32, info->daemon_uuid, 16) != 0 ||
      get_u64(frame + 48) != info->connection_id ||
      get_u64(frame + 56) != info->epoch || request->request_id == 0 ||
      request->request_id <= *last_request_id) {
    if (request->descriptor >= 0) {
      (void)close(request->descriptor);
      request->descriptor = -1;
    }
    return -1;
  }
  *last_request_id = request->request_id;
  if (request->message_type == SAGR_WIRE_MESSAGE_MEMORY_REQUEST) {
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
  } else {
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
  }
  return 0;
}

static int validate_staging_descriptor(int descriptor, uint64_t byte_count,
                                       int d2h, int final) {
  struct stat attributes;
  const int descriptor_flags = fcntl(descriptor, F_GETFD);
  const int status_flags = fcntl(descriptor, F_GETFL);
  const int seals = fcntl(descriptor, F_GET_SEALS);
  int expected_seals = F_SEAL_SHRINK | F_SEAL_GROW;
  if (d2h == 0 || final != 0) {
    expected_seals |= F_SEAL_WRITE | F_SEAL_SEAL;
  }
  if (descriptor_flags < 0 || status_flags < 0 || seals < 0 ||
      fstat(descriptor, &attributes) != 0 ||
      !S_ISREG(attributes.st_mode) || attributes.st_nlink != 0 ||
      attributes.st_uid != geteuid() ||
      (attributes.st_mode & (mode_t)07777) != (mode_t)0600 ||
      attributes.st_size < 0 ||
      (uint64_t)attributes.st_size != byte_count ||
      (descriptor_flags & FD_CLOEXEC) == 0 || seals != expected_seals) {
    return -1;
  }
  if (d2h != 0) {
    return (status_flags & O_ACCMODE) == O_RDWR ? 0 : -1;
  }
  return (status_flags & O_ACCMODE) != O_WRONLY ? 0 : -1;
}

static void initialize_memory_response(
    sagr_wire_memory_response_t *response,
    const sagr_wire_memory_request_t *request, uint32_t status,
    uint64_t sim_tick) {
  memset(response, 0, sizeof(*response));
  response->major = SAGR_MEMORY_PROTOCOL_MAJOR;
  response->minor = SAGR_MEMORY_PROTOCOL_MINOR;
  response->status = status;
  response->opcode = request->opcode;
  response->allocation_id = request->allocation_id;
  response->generation = request->generation;
  response->sim_tick = sim_tick;
}

static int handle_memory_request(
    int peer, const sagr_instance_info_t *info,
    enum memory_server_behavior behavior, server_memory_state_t *state,
    received_request_t *received, int cancel_write_fd,
    uint64_t post_ack_deadline_ns) {
  const sagr_wire_memory_request_t *request = &received->memory;
  sagr_wire_memory_response_t response;
  uint8_t *scratch = NULL;
  uint32_t crc = 0;
  int result = -1;

  initialize_memory_response(&response, request, SAGR_WIRE_STATUS_OK,
                             ++state->tick);
  if (request->opcode == SAGR_WIRE_MEMORY_OPCODE_ALLOC) {
    if (received->descriptor >= 0 || request->allocation_id != 0 ||
        request->generation != 0 || request->offset != 0 ||
        request->byte_count == 0 ||
        (request->argument != SAGR_MEMORY_ALIGNMENT_4K &&
         request->argument != SAGR_MEMORY_ALIGNMENT_64K)) {
      return -1;
    }
    if (request->byte_count > SAGR_MEMORY_MAX_SINGLE_ALLOCATION_BYTES) {
      initialize_memory_response(&response, request,
                                 SAGR_WIRE_STATUS_RESOURCE_EXHAUSTED, 0);
      return send_memory_response(peer, info, received->request_id, &response);
    }
    if (state->allocated != 0 || request->byte_count > (uint64_t)SIZE_MAX) {
      return -1;
    }
    state->bytes = (uint8_t *)calloc((size_t)request->byte_count, 1);
    if (state->bytes == NULL) {
      return -1;
    }
    state->size = request->byte_count;
    ++state->generation;
    state->allocated = 1;
    response.allocation_id = MOCK_ALLOCATION_ID;
    response.generation = state->generation;
    response.value0 = MOCK_SIMULATED_VA;
    response.value1 = request->byte_count;
    response.value2 = request->argument;
    return send_memory_response(peer, info, received->request_id, &response);
  }

  if (state->allocated == 0 || request->allocation_id != MOCK_ALLOCATION_ID ||
      request->generation != state->generation) {
    initialize_memory_response(&response, request,
                               SAGR_WIRE_STATUS_PROTOCOL_STATE, 0);
    return send_memory_response(peer, info, received->request_id, &response);
  }
  if (request->opcode == SAGR_WIRE_MEMORY_OPCODE_FREE) {
    if (received->descriptor >= 0 || request->offset != 0 ||
        request->byte_count != 0 || request->argument != 0) {
      return -1;
    }
    free(state->bytes);
    state->bytes = NULL;
    state->size = 0;
    state->allocated = 0;
    response.sim_tick = ++state->tick;
    return send_memory_response(peer, info, received->request_id, &response);
  }
  if ((request->opcode != SAGR_WIRE_MEMORY_OPCODE_COPY_H2D &&
       request->opcode != SAGR_WIRE_MEMORY_OPCODE_COPY_D2H) ||
      received->descriptor < 0 || request->byte_count == 0 ||
      request->offset > state->size ||
      request->byte_count > state->size - request->offset ||
      request->byte_count > (uint64_t)SIZE_MAX) {
    return -1;
  }
  scratch = (uint8_t *)malloc((size_t)request->byte_count);
  if (scratch == NULL) {
    return -1;
  }
  if (request->opcode == SAGR_WIRE_MEMORY_OPCODE_COPY_H2D) {
    if (request->argument > UINT32_MAX ||
        validate_staging_descriptor(received->descriptor,
                                    request->byte_count, 0, 1) != 0 ||
        read_all_at(received->descriptor, scratch,
                    (size_t)request->byte_count) != 0) {
      goto done;
    }
    crc = sagr_crc32c(scratch, (size_t)request->byte_count);
    if (crc != (uint32_t)request->argument) {
      goto done;
    }
    memcpy(state->bytes + (size_t)request->offset, scratch,
           (size_t)request->byte_count);
  } else {
    if (request->argument != 0 ||
        validate_staging_descriptor(received->descriptor,
                                    request->byte_count, 1, 0) != 0) {
      goto done;
    }
    memcpy(scratch, state->bytes + (size_t)request->offset,
           (size_t)request->byte_count);
    if (write_all_at(received->descriptor, scratch,
                     (size_t)request->byte_count) != 0) {
      goto done;
    }
    if (behavior != MEMORY_SERVER_BAD_D2H_SEALS &&
        (fcntl(received->descriptor, F_ADD_SEALS,
               F_SEAL_WRITE | F_SEAL_SEAL) != 0 ||
         validate_staging_descriptor(received->descriptor,
                                     request->byte_count, 1, 1) != 0)) {
      goto done;
    }
    if (read_all_at(received->descriptor, scratch,
                    (size_t)request->byte_count) != 0) {
      goto done;
    }
    crc = sagr_crc32c(scratch, (size_t)request->byte_count);
    if ((behavior == MEMORY_SERVER_BAD_D2H_SETUID &&
         fchmod(received->descriptor, (mode_t)04600) != 0) ||
        (behavior == MEMORY_SERVER_BAD_D2H_SETGID &&
         fchmod(received->descriptor, (mode_t)02600) != 0) ||
        (behavior == MEMORY_SERVER_BAD_D2H_STICKY &&
         fchmod(received->descriptor, (mode_t)01600) != 0)) {
      goto done;
    }
  }
  response.value0 = request->offset;
  response.value1 = request->byte_count;
  response.value2 = crc;
  if (behavior == MEMORY_SERVER_BAD_D2H_CRC &&
      request->opcode == SAGR_WIRE_MEMORY_OPCODE_COPY_D2H &&
      state->corrupted_d2h_sent == 0) {
    response.value2 ^= UINT64_C(1);
    state->corrupted_d2h_sent = 1;
  }
  if (behavior == MEMORY_SERVER_POST_ACK_DEADLINE_CANCEL &&
      request->opcode == SAGR_WIRE_MEMORY_OPCODE_COPY_D2H &&
      post_ack_deadline_ns > UINT64_C(5000000)) {
    struct timespec send_time;
    int sleep_result;
    const uint64_t send_ns = post_ack_deadline_ns - UINT64_C(5000000);
    send_time.tv_sec = (time_t)(send_ns / UINT64_C(1000000000));
    send_time.tv_nsec = (long)(send_ns % UINT64_C(1000000000));
    do {
      sleep_result =
          clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &send_time, NULL);
    } while (sleep_result == EINTR);
    if (sleep_result != 0) {
      goto done;
    }
  }
  result = send_memory_response(peer, info, received->request_id, &response);
  if (result == 0 && behavior == MEMORY_SERVER_POST_ACK_CANCEL &&
      request->opcode == SAGR_WIRE_MEMORY_OPCODE_COPY_D2H &&
      cancel_write_fd >= 0) {
    const struct timespec delay = {0, 5000000};
    const uint8_t signal = 1;
    (void)nanosleep(&delay, NULL);
    if (write(cancel_write_fd, &signal, sizeof(signal)) !=
        (ssize_t)sizeof(signal)) {
      result = -1;
    }
  }
  if (result == 0 &&
      behavior == MEMORY_SERVER_POST_ACK_DEADLINE_CANCEL &&
      request->opcode == SAGR_WIRE_MEMORY_OPCODE_COPY_D2H &&
      cancel_write_fd >= 0 &&
      post_ack_deadline_ns <= UINT64_MAX - UINT64_C(1000000)) {
    struct timespec cancel_time;
    int sleep_result;
    const uint64_t cancel_ns = post_ack_deadline_ns + UINT64_C(1000000);
    const uint8_t signal = 1;
    cancel_time.tv_sec = (time_t)(cancel_ns / UINT64_C(1000000000));
    cancel_time.tv_nsec = (long)(cancel_ns % UINT64_C(1000000000));
    do {
      sleep_result =
          clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &cancel_time, NULL);
    } while (sleep_result == EINTR);
    if (sleep_result != 0 ||
        write(cancel_write_fd, &signal, sizeof(signal)) !=
            (ssize_t)sizeof(signal)) {
      result = -1;
    }
  }

done:
  free(scratch);
  (void)close(received->descriptor);
  received->descriptor = -1;
  return result;
}

static int handle_interleaved_queue_request(
    int peer, const sagr_instance_info_t *info, received_request_t *received,
    uint32_t *stage, uint64_t *doorbell_request_id) {
  sagr_wire_queue_response_t response;
  const sagr_wire_queue_request_t *request = &received->queue;
  if (received->descriptor >= 0) {
    return -1;
  }
  memset(&response, 0, sizeof(response));
  response.major = SAGR_QUEUE_PROTOCOL_MAJOR;
  response.minor = SAGR_QUEUE_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = request->opcode;
  if (*stage == 0 && request->opcode == SAGR_WIRE_QUEUE_OPCODE_CREATE &&
      request->queue_id == 0 && request->generation == 0 &&
      request->sequence == 0 && request->arg0 == UINT64_C(4) &&
      request->arg1 == 0) {
    response.queue_id = MOCK_QUEUE_ID;
    response.generation = MOCK_QUEUE_GENERATION;
    response.value = request->arg0;
    response.sim_tick = UINT64_C(10);
    ++*stage;
  } else if (*stage == 1 &&
             request->opcode == SAGR_WIRE_QUEUE_OPCODE_DOORBELL &&
             request->queue_id == MOCK_QUEUE_ID &&
             request->generation == MOCK_QUEUE_GENERATION &&
             request->sequence == UINT64_C(1) &&
             request->arg0 == SAGR_QUEUE_COMMAND_CONTROL_TEST &&
             request->arg1 == 0) {
    response.queue_id = request->queue_id;
    response.generation = request->generation;
    response.sequence = request->sequence;
    response.sim_tick = UINT64_C(20);
    *doorbell_request_id = received->request_id;
    ++*stage;
  } else if (*stage == 4 &&
             request->opcode == SAGR_WIRE_QUEUE_OPCODE_DESTROY &&
             request->queue_id == MOCK_QUEUE_ID &&
             request->generation == MOCK_QUEUE_GENERATION &&
             request->sequence == 0 && request->arg0 == 0 &&
             request->arg1 == 0) {
    response.queue_id = request->queue_id;
    response.generation = request->generation;
    response.sim_tick = UINT64_C(30);
    ++*stage;
  } else {
    return -1;
  }
  return send_queue_response(peer, info, received->request_id,
                             SAGR_WIRE_MESSAGE_QUEUE_ACK, &response);
}

static int send_prior_queue_completion(int peer,
                                       const sagr_instance_info_t *info,
                                       uint64_t request_id) {
  sagr_wire_queue_response_t completion;
  memset(&completion, 0, sizeof(completion));
  completion.major = SAGR_QUEUE_PROTOCOL_MAJOR;
  completion.minor = SAGR_QUEUE_PROTOCOL_MINOR;
  completion.status = SAGR_WIRE_STATUS_OK;
  completion.opcode = SAGR_WIRE_QUEUE_OPCODE_DOORBELL;
  completion.queue_id = MOCK_QUEUE_ID;
  completion.generation = MOCK_QUEUE_GENERATION;
  completion.sequence = UINT64_C(1);
  completion.value = SAGR_QUEUE_COMMAND_CONTROL_TEST;
  completion.sim_tick = UINT64_C(21);
  return send_queue_response(peer, info, request_id,
                             SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, &completion);
}

static void *memory_server_main(void *argument) {
  memory_server_t *server = (memory_server_t *)argument;
  sagr_instance_info_t info;
  server_memory_state_t state;
  uint8_t hello[SAGR_WIRE_HELLO_FRAME_BYTES];
  uint64_t last_request_id = 0;
  uint64_t capabilities =
      SAGR_CAPABILITY_TOPOLOGY_MASK | SAGR_CAPABILITY_MEMORY_MASK;
  uint64_t doorbell_request_id = 0;
  uint32_t interleave_stage = 0;
  int peer = -1;
  ssize_t hello_size;

  memset(&state, 0, sizeof(state));
  state.generation = UINT64_C(0x8877665544332210);
  state.tick = UINT64_C(100);
  if (server->behavior == MEMORY_SERVER_NO_CAPABILITY) {
    capabilities = SAGR_CAPABILITY_TOPOLOGY_MASK;
  } else if (server->behavior == MEMORY_SERVER_INTERLEAVED_COMPLETION) {
    capabilities |= SAGR_CAPABILITY_QUEUE_MASK;
  }
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
  if (send_handshake_ack(peer, hello, hello_size, capabilities, &info,
                         &last_request_id) != 0) {
    server->thread_error = 1;
    goto done;
  }
  if (server->behavior == MEMORY_SERVER_NO_CAPABILITY) {
    goto done;
  }

  for (;;) {
    received_request_t request;
    const int received =
        receive_request(peer, &info, &last_request_id, &request);
    if (received == 1) {
      break;
    }
    if (received != 0) {
      server->thread_error = 1;
      break;
    }
    if (server->behavior == MEMORY_SERVER_ALLOC_TIMEOUT) {
      struct pollfd descriptor;
      uint8_t byte;
      int poll_result;
      if (request.message_type != SAGR_WIRE_MESSAGE_MEMORY_REQUEST ||
          request.memory.opcode != SAGR_WIRE_MEMORY_OPCODE_ALLOC ||
          request.descriptor >= 0) {
        server->thread_error = 1;
        break;
      }
      descriptor.fd = peer;
      descriptor.events = POLLIN;
      descriptor.revents = 0;
      do {
        poll_result = poll(&descriptor, 1, 2000);
      } while (poll_result < 0 && errno == EINTR);
      if (poll_result <= 0 || recv(peer, &byte, sizeof(byte), 0) != 0) {
        server->thread_error = 1;
      }
      break;
    }
    if (server->behavior == MEMORY_SERVER_INTERLEAVED_COMPLETION &&
        request.message_type == SAGR_WIRE_MESSAGE_QUEUE_REQUEST) {
      if (handle_interleaved_queue_request(
              peer, &info, &request, &interleave_stage,
              &doorbell_request_id) != 0) {
        server->thread_error = 1;
        break;
      }
      continue;
    }
    if (request.message_type != SAGR_WIRE_MESSAGE_MEMORY_REQUEST) {
      server->thread_error = 1;
      break;
    }
    if (server->behavior == MEMORY_SERVER_INTERLEAVED_COMPLETION &&
        interleave_stage == 2 &&
        request.memory.opcode == SAGR_WIRE_MEMORY_OPCODE_ALLOC) {
      if (doorbell_request_id == 0 ||
          send_prior_queue_completion(peer, &info, doorbell_request_id) != 0) {
        server->thread_error = 1;
        break;
      }
      ++interleave_stage;
    }
    if (handle_memory_request(peer, &info, server->behavior, &state, &request,
                              server->cancel_write_fd,
                              server->post_ack_deadline_ns) != 0) {
      server->thread_error = 1;
      break;
    }
    if (server->behavior == MEMORY_SERVER_INTERLEAVED_COMPLETION &&
        interleave_stage == 3 &&
        request.memory.opcode == SAGR_WIRE_MEMORY_OPCODE_FREE) {
      ++interleave_stage;
    }
  }

done:
  free(state.bytes);
  if (peer >= 0) {
    (void)close(peer);
  }
  return NULL;
}

static int start_memory_server_internal(memory_server_t *server,
                                        enum memory_server_behavior behavior,
                                        int cancel_write_fd,
                                        uint64_t post_ack_deadline_ns) {
  struct sockaddr_un address;
  char template_path[] = "/tmp/sagr-memory-XXXXXX";
  char *directory;
  memset(server, 0, sizeof(*server));
  server->listener = -1;
  server->behavior = behavior;
  server->cancel_write_fd = cancel_write_fd;
  server->post_ack_deadline_ns = post_ack_deadline_ns;
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
      pthread_create(&server->thread, NULL, memory_server_main, server) != 0) {
    (void)close(server->listener);
    server->listener = -1;
    return -1;
  }
  return 0;
}

static int start_memory_server(memory_server_t *server,
                               enum memory_server_behavior behavior) {
  return start_memory_server_internal(server, behavior, -1, 0);
}

static int stop_memory_server(memory_server_t *server) {
  int result = 0;
  if (pthread_join(server->thread, NULL) != 0) {
    result = -1;
  }
  if (server->listener >= 0 && close(server->listener) != 0) {
    result = -1;
  }
  if (unlink(server->endpoint) != 0 || rmdir(server->directory) != 0) {
    result = -1;
  }
  return server->thread_error == 0 ? result : -1;
}

static sagr_status_t open_memory_instance(const char *endpoint,
                                          int include_queue,
                                          sagr_instance_t *instance,
                                          sagr_error_info_t *error) {
  sagr_instance_open_options_t options;
  (void)sagr_instance_open_options_init(&options, (uint32_t)sizeof(options));
  options.offered_capabilities[0] |= SAGR_CAPABILITY_MEMORY_MASK;
  options.required_capabilities[0] |= SAGR_CAPABILITY_MEMORY_MASK;
  if (include_queue != 0) {
    options.offered_capabilities[0] |= SAGR_CAPABILITY_QUEUE_MASK;
    options.required_capabilities[0] |= SAGR_CAPABILITY_QUEUE_MASK;
  }
  return sagr_instance_open(endpoint, &options, instance, error,
                            (uint32_t)sizeof(*error));
}

static int test_memory_lifecycle(void) {
  memory_server_t server;
  sagr_instance_t instance = NULL;
  sagr_memory_t memory = NULL;
  sagr_memory_t reused = NULL;
  sagr_memory_allocate_options_t allocate_options;
  sagr_memory_info_t info;
  sagr_memory_info_t queried;
  sagr_memory_info_t reuse_info;
  sagr_error_info_t error;
  uint8_t *expected = NULL;
  uint8_t *returned = NULL;
  const uint64_t allocation_size = UINT64_C(70000);
  const uint64_t copy_offset = UINT64_C(60000);
  const uint64_t copy_size = UINT64_C(10000);
  uint64_t index;
  int failed = 0;

  if (start_memory_server(&server, MEMORY_SERVER_SUCCESS) != 0 ||
      open_memory_instance(server.endpoint, 0, &instance, &error) !=
          SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "memory lifecycle server/open failed\n");
    return 1;
  }
  expected = (uint8_t *)calloc((size_t)allocation_size, 1);
  returned = (uint8_t *)malloc((size_t)allocation_size);
  (void)sagr_memory_allocate_options_init(
      &allocate_options, (uint32_t)sizeof(allocate_options));
  allocate_options.size_bytes = allocation_size;
  allocate_options.alignment_bytes = SAGR_MEMORY_ALIGNMENT_64K;
  if (expected == NULL || returned == NULL ||
      sagr_memory_allocate(instance, &allocate_options, NULL, &memory, &info,
                           (uint32_t)sizeof(info), &error,
                           (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      memory == NULL || info.allocation_id != MOCK_ALLOCATION_ID ||
      info.generation == 0 || info.simulated_va != MOCK_SIMULATED_VA ||
      info.size_bytes != allocation_size ||
      info.alignment_bytes != SAGR_MEMORY_ALIGNMENT_64K ||
      sagr_memory_get_info(memory, &queried, (uint32_t)sizeof(queried)) !=
          SAGR_STATUS_SUCCESS ||
      memcmp(&info, &queried, sizeof(info)) != 0) {
    fprintf(stderr, "memory allocate/info failed: %s\n", error.message);
    failed = 1;
    goto done;
  }
  memset(returned, 0xa5, (size_t)allocation_size);
  if (sagr_memory_copy_to_host(memory, 0, returned, allocation_size, NULL,
                               &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      !bytes_are_zero(returned, (size_t)allocation_size)) {
    fprintf(stderr, "initial zero D2H failed: %s\n", error.message);
    failed = 1;
    goto done;
  }
  for (index = 0; index < copy_size; ++index) {
    expected[(size_t)(copy_offset + index)] =
        (uint8_t)((index * UINT64_C(37) + UINT64_C(11)) & UINT64_C(0xff));
  }
  if (sagr_memory_copy_from_host(
          memory, copy_offset, expected + (size_t)copy_offset, copy_size, NULL,
          &error, (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "cross-range H2D failed: %s\n", error.message);
    failed = 1;
    goto done;
  }
  memset(returned, 0xa5, (size_t)allocation_size);
  if (sagr_memory_copy_to_host(memory, 0, returned, allocation_size, NULL,
                               &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      memcmp(expected, returned, (size_t)allocation_size) != 0 ||
      sagr_memory_copy_to_host(memory, allocation_size - UINT64_C(1), returned,
                               UINT64_C(2), NULL, &error,
                               (uint32_t)sizeof(error)) !=
          SAGR_STATUS_INVALID_ARGUMENT) {
    fprintf(stderr, "memory roundtrip/range validation failed: %s\n",
            error.message);
    failed = 1;
    goto done;
  }
  if (sagr_memory_free(&memory, NULL, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      memory != NULL ||
      sagr_memory_allocate(instance, &allocate_options, NULL, &reused,
                           &reuse_info, (uint32_t)sizeof(reuse_info), &error,
                           (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      reuse_info.allocation_id != info.allocation_id ||
      reuse_info.simulated_va != info.simulated_va ||
      reuse_info.generation == 0 || reuse_info.generation == info.generation) {
    fprintf(stderr, "memory free/reuse identity failed: %s\n", error.message);
    failed = 1;
    goto done;
  }
  memset(returned, 0xa5, (size_t)allocation_size);
  if (sagr_memory_copy_to_host(reused, 0, returned, allocation_size, NULL,
                               &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      !bytes_are_zero(returned, (size_t)allocation_size) ||
      sagr_memory_free(&reused, NULL, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      reused != NULL) {
    fprintf(stderr, "reused allocation zero/free failed: %s\n", error.message);
    failed = 1;
  }

done:
  free(returned);
  free(expected);
  (void)sagr_instance_close(&instance);
  if (stop_memory_server(&server) != 0) {
    fprintf(stderr, "memory lifecycle server failed\n");
    failed = 1;
  }
  return failed;
}

static int test_resource_ack_is_retryable(void) {
  memory_server_t server;
  sagr_instance_t instance = NULL;
  sagr_memory_t memory = NULL;
  sagr_memory_allocate_options_t options;
  sagr_error_info_t error;
  int failed = 0;
  if (start_memory_server(&server, MEMORY_SERVER_SUCCESS) != 0 ||
      open_memory_instance(server.endpoint, 0, &instance, &error) !=
          SAGR_STATUS_SUCCESS) {
    return 1;
  }
  (void)sagr_memory_allocate_options_init(&options,
                                          (uint32_t)sizeof(options));
  options.size_bytes = SAGR_MEMORY_MAX_SINGLE_ALLOCATION_BYTES + UINT64_C(1);
  if (sagr_memory_allocate(instance, &options, NULL, &memory, NULL, 0, &error,
                           (uint32_t)sizeof(error)) !=
          SAGR_STATUS_OUT_OF_RESOURCES ||
      error.wire_status != SAGR_WIRE_STATUS_RESOURCE_EXHAUSTED ||
      memory != NULL) {
    fprintf(stderr, "oversize ALLOC did not return determinate resource ACK\n");
    failed = 1;
    goto done;
  }
  options.size_bytes = UINT64_C(4096);
  if (sagr_memory_allocate(instance, &options, NULL, &memory, NULL, 0, &error,
                           (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      sagr_memory_free(&memory, NULL, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "session was not reusable after resource ACK\n");
    failed = 1;
  }
done:
  (void)sagr_instance_close(&instance);
  if (stop_memory_server(&server) != 0) {
    failed = 1;
  }
  return failed;
}

static int test_bad_d2h_crc_is_atomic_and_poisoning(void) {
  memory_server_t server;
  sagr_instance_t instance = NULL;
  sagr_memory_t memory = NULL;
  sagr_memory_allocate_options_t options;
  sagr_error_info_t error;
  uint8_t destination[32];
  uint8_t sentinel[32];
  int failed = 0;
  memset(destination, 0xa5, sizeof(destination));
  memcpy(sentinel, destination, sizeof(sentinel));
  if (start_memory_server(&server, MEMORY_SERVER_BAD_D2H_CRC) != 0 ||
      open_memory_instance(server.endpoint, 0, &instance, &error) !=
          SAGR_STATUS_SUCCESS) {
    return 1;
  }
  (void)sagr_memory_allocate_options_init(&options,
                                          (uint32_t)sizeof(options));
  options.size_bytes = sizeof(destination);
  if (sagr_memory_allocate(instance, &options, NULL, &memory, NULL, 0, &error,
                           (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      sagr_memory_copy_to_host(memory, 0, destination, sizeof(destination),
                               NULL, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_CHECKSUM_ERROR ||
      memcmp(destination, sentinel, sizeof(destination)) != 0 ||
      sagr_memory_free(&memory, NULL, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_CONNECTION_LOST) {
    fprintf(stderr, "bad D2H CRC did not preserve output and poison session\n");
    failed = 1;
  }
  (void)sagr_instance_close(&instance);
  if (stop_memory_server(&server) != 0) {
    failed = 1;
  }
  return failed;
}

static int test_bad_d2h_mode_is_atomic_and_poisoning(
    enum memory_server_behavior behavior, const char *mode_name) {
  memory_server_t server;
  sagr_instance_t instance = NULL;
  sagr_memory_t memory = NULL;
  sagr_memory_allocate_options_t options;
  sagr_error_info_t error;
  uint8_t destination[32];
  uint8_t sentinel[32];
  int failed = 0;
  memset(destination, 0xa5, sizeof(destination));
  memcpy(sentinel, destination, sizeof(sentinel));
  if (start_memory_server(&server, behavior) != 0 ||
      open_memory_instance(server.endpoint, 0, &instance, &error) !=
          SAGR_STATUS_SUCCESS) {
    return 1;
  }
  (void)sagr_memory_allocate_options_init(&options,
                                          (uint32_t)sizeof(options));
  options.size_bytes = sizeof(destination);
  if (sagr_memory_allocate(instance, &options, NULL, &memory, NULL, 0, &error,
                           (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      sagr_memory_copy_to_host(memory, 0, destination, sizeof(destination),
                               NULL, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_PROTOCOL_ERROR ||
      memcmp(destination, sentinel, sizeof(destination)) != 0 ||
      sagr_memory_free(&memory, NULL, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_CONNECTION_LOST) {
    fprintf(stderr,
            "D2H %s carrier mode did not preserve output and poison session\n",
            mode_name);
    failed = 1;
  }
  (void)sagr_instance_close(&instance);
  if (stop_memory_server(&server) != 0) {
    failed = 1;
  }
  return failed;
}

static int test_post_ack_cancel_is_atomic_and_retryable(void) {
  memory_server_t server;
  sagr_instance_t instance = NULL;
  sagr_memory_t memory = NULL;
  sagr_memory_allocate_options_t allocate_options;
  sagr_memory_operation_options_t operation_options;
  sagr_error_info_t error;
  uint8_t *destination = NULL;
  int cancel_pipe[2] = {-1, -1};
  const uint64_t byte_count = SAGR_MEMORY_MAX_TRANSFER_BYTES;
  size_t index;
  int failed = 0;
  if (pipe2(cancel_pipe, O_CLOEXEC) != 0 ||
      start_memory_server_internal(&server, MEMORY_SERVER_POST_ACK_CANCEL,
                                   cancel_pipe[1], 0) != 0 ||
      open_memory_instance(server.endpoint, 0, &instance, &error) !=
          SAGR_STATUS_SUCCESS) {
    if (cancel_pipe[0] >= 0) {
      (void)close(cancel_pipe[0]);
      (void)close(cancel_pipe[1]);
    }
    return 1;
  }
  destination = (uint8_t *)malloc((size_t)byte_count);
  (void)sagr_memory_allocate_options_init(
      &allocate_options, (uint32_t)sizeof(allocate_options));
  (void)sagr_memory_operation_options_init(
      &operation_options, (uint32_t)sizeof(operation_options));
  allocate_options.size_bytes = byte_count;
  operation_options.cancel_fd = cancel_pipe[0];
  if (destination == NULL ||
      sagr_memory_allocate(instance, &allocate_options, NULL, &memory, NULL, 0,
                           &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS) {
    failed = 1;
    goto done;
  }
  memset(destination, 0xa5, (size_t)byte_count);
  if (sagr_memory_copy_to_host(memory, 0, destination, byte_count,
                               &operation_options, &error,
                               (uint32_t)sizeof(error)) !=
      SAGR_STATUS_CANCELLED) {
    fprintf(stderr, "post-ACK D2H cancellation was not observed\n");
    failed = 1;
    goto done;
  }
  for (index = 0; index < (size_t)byte_count; ++index) {
    if (destination[index] != UINT8_C(0xa5)) {
      fprintf(stderr, "post-ACK cancellation changed caller output\n");
      failed = 1;
      goto done;
    }
  }
  if (sagr_memory_free(&memory, NULL, &error, (uint32_t)sizeof(error)) !=
      SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "post-ACK cancellation poisoned a known session\n");
    failed = 1;
  }

done:
  free(destination);
  (void)sagr_instance_close(&instance);
  (void)close(cancel_pipe[0]);
  (void)close(cancel_pipe[1]);
  if (stop_memory_server(&server) != 0) {
    failed = 1;
  }
  return failed;
}

static int test_post_ack_deadline_precedes_cancel(void) {
  memory_server_t server;
  sagr_instance_t instance = NULL;
  sagr_memory_t memory = NULL;
  sagr_memory_allocate_options_t allocate_options;
  sagr_memory_operation_options_t operation_options;
  sagr_error_info_t error;
  struct timespec now;
  uint8_t *destination = NULL;
  int cancel_pipe[2] = {-1, -1};
  const uint64_t byte_count = UINT64_C(8388608);
  uint64_t deadline_ns;
  size_t index;
  int failed = 0;

  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 || now.tv_sec < 0 ||
      (uint64_t)now.tv_sec >
          (UINT64_MAX - UINT64_C(2000000000)) / UINT64_C(1000000000)) {
    return 1;
  }
  deadline_ns = (uint64_t)now.tv_sec * UINT64_C(1000000000) +
                (uint64_t)now.tv_nsec + UINT64_C(2000000000);
  if (pipe2(cancel_pipe, O_CLOEXEC) != 0 ||
      start_memory_server_internal(
          &server, MEMORY_SERVER_POST_ACK_DEADLINE_CANCEL, cancel_pipe[1],
          deadline_ns) != 0 ||
      open_memory_instance(server.endpoint, 0, &instance, &error) !=
          SAGR_STATUS_SUCCESS) {
    if (cancel_pipe[0] >= 0) {
      (void)close(cancel_pipe[0]);
      (void)close(cancel_pipe[1]);
    }
    return 1;
  }
  destination = (uint8_t *)malloc((size_t)byte_count);
  (void)sagr_memory_allocate_options_init(
      &allocate_options, (uint32_t)sizeof(allocate_options));
  (void)sagr_memory_operation_options_init(
      &operation_options, (uint32_t)sizeof(operation_options));
  allocate_options.size_bytes = byte_count;
  operation_options.absolute_deadline_ns = deadline_ns;
  operation_options.cancel_fd = cancel_pipe[0];
  if (destination == NULL ||
      sagr_memory_allocate(instance, &allocate_options, NULL, &memory, NULL, 0,
                           &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS) {
    failed = 1;
    goto done;
  }
  memset(destination, 0xa5, (size_t)byte_count);
  if (sagr_memory_copy_to_host(memory, 0, destination, byte_count,
                               &operation_options, &error,
                               (uint32_t)sizeof(error)) !=
      SAGR_STATUS_TIMED_OUT) {
    fprintf(stderr, "post-ACK deadline did not precede ready cancellation\n");
    failed = 1;
    goto done;
  }
  for (index = 0; index < (size_t)byte_count; ++index) {
    if (destination[index] != UINT8_C(0xa5)) {
      fprintf(stderr, "post-ACK deadline changed caller output\n");
      failed = 1;
      goto done;
    }
  }
  if (sagr_memory_free(&memory, NULL, &error, (uint32_t)sizeof(error)) !=
      SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "post-ACK deadline poisoned a known session\n");
    failed = 1;
  }

done:
  free(destination);
  (void)sagr_instance_close(&instance);
  (void)close(cancel_pipe[0]);
  (void)close(cancel_pipe[1]);
  if (stop_memory_server(&server) != 0) {
    failed = 1;
  }
  return failed;
}

static int test_alloc_ack_timeout_poisons(void) {
  memory_server_t server;
  sagr_instance_t instance = NULL;
  sagr_memory_t memory = NULL;
  sagr_memory_allocate_options_t allocate_options;
  sagr_memory_operation_options_t operation_options;
  sagr_error_info_t error;
  int failed = 0;
  if (start_memory_server(&server, MEMORY_SERVER_ALLOC_TIMEOUT) != 0 ||
      open_memory_instance(server.endpoint, 0, &instance, &error) !=
          SAGR_STATUS_SUCCESS) {
    return 1;
  }
  (void)sagr_memory_allocate_options_init(
      &allocate_options, (uint32_t)sizeof(allocate_options));
  (void)sagr_memory_operation_options_init(
      &operation_options, (uint32_t)sizeof(operation_options));
  allocate_options.size_bytes = UINT64_C(4096);
  operation_options.timeout_ns = UINT64_C(30000000);
  if (sagr_memory_allocate(instance, &allocate_options, &operation_options,
                           &memory, NULL, 0, &error,
                           (uint32_t)sizeof(error)) != SAGR_STATUS_TIMED_OUT ||
      memory != NULL ||
      sagr_memory_allocate(instance, &allocate_options, NULL, &memory, NULL, 0,
                           &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_CONNECTION_LOST) {
    fprintf(stderr, "ambiguous ALLOC timeout did not poison session\n");
    failed = 1;
  }
  (void)sagr_instance_close(&instance);
  if (stop_memory_server(&server) != 0) {
    failed = 1;
  }
  return failed;
}

static int test_memory_capability_rejection(void) {
  memory_server_t server;
  sagr_instance_t instance = NULL;
  sagr_error_info_t error;
  int failed = 0;
  if (start_memory_server(&server, MEMORY_SERVER_NO_CAPABILITY) != 0) {
    return 1;
  }
  if (open_memory_instance(server.endpoint, 0, &instance, &error) !=
          SAGR_STATUS_CAPABILITY_MISMATCH ||
      instance != NULL) {
    fprintf(stderr, "missing required memory capability was accepted\n");
    failed = 1;
  }
  if (stop_memory_server(&server) != 0) {
    failed = 1;
  }
  return failed;
}

static int test_queue_completion_interleaves_with_memory_ack(void) {
  memory_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_memory_t memory = NULL;
  sagr_queue_create_options_t queue_options;
  sagr_memory_allocate_options_t memory_options;
  sagr_queue_completion_t completion;
  sagr_error_info_t error;
  uint64_t sequence = 0;
  int failed = 0;
  if (start_memory_server(&server, MEMORY_SERVER_INTERLEAVED_COMPLETION) != 0 ||
      open_memory_instance(server.endpoint, 1, &instance, &error) !=
          SAGR_STATUS_SUCCESS) {
    return 1;
  }
  (void)sagr_queue_create_options_init(&queue_options,
                                       (uint32_t)sizeof(queue_options));
  queue_options.depth = 4;
  (void)sagr_memory_allocate_options_init(
      &memory_options, (uint32_t)sizeof(memory_options));
  memory_options.size_bytes = UINT64_C(4096);
  if (sagr_queue_create(instance, &queue_options, NULL, &queue, NULL, 0,
                        &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_queue_ring_doorbell(queue, SAGR_QUEUE_COMMAND_CONTROL_TEST, NULL,
                               &sequence, &error,
                               (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      sagr_memory_allocate(instance, &memory_options, NULL, &memory, NULL, 0,
                           &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_queue_wait(queue, sequence, NULL, &completion,
                      (uint32_t)sizeof(completion), &error,
                      (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS ||
      completion.sequence != sequence || completion.sim_tick != UINT64_C(21) ||
      sagr_memory_free(&memory, NULL, &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_queue_destroy(&queue, NULL, &error,
                         (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "queue completion was not buffered across memory ACK: %s\n",
            error.message);
    failed = 1;
  }
  (void)sagr_instance_close(&instance);
  if (stop_memory_server(&server) != 0) {
    failed = 1;
  }
  return failed;
}

#ifdef SAGR_MEMORY_CLI_PATH
static int test_memory_cli_roundtrip(void) {
  memory_server_t server;
  int output_pipe[2] = {-1, -1};
  char output[4096];
  size_t output_size = 0;
  pid_t child;
  int child_status = 0;
  int failed = 0;
  if (start_memory_server(&server, MEMORY_SERVER_SUCCESS) != 0 ||
      pipe2(output_pipe, O_CLOEXEC) != 0) {
    return 1;
  }
  child = fork();
  if (child == 0) {
    (void)dup2(output_pipe[1], STDOUT_FILENO);
    (void)close(output_pipe[0]);
    (void)close(output_pipe[1]);
    execl(SAGR_MEMORY_CLI_PATH, SAGR_MEMORY_CLI_PATH, "--endpoint",
          server.endpoint, "--memory-bytes", "70000", "--memory-alignment",
          "65536", "--memory-reuse", (char *)NULL);
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
  if (strstr(output, "\"memory\":{\"status\":0") == NULL ||
      strstr(output, "\"size_bytes\":70000") == NULL ||
      strstr(output, "\"alignment_bytes\":65536") == NULL ||
      strstr(output, "\"initial_zero\":true") == NULL ||
      strstr(output, "\"pattern_crc32c\":\"0x") == NULL ||
      strstr(output, "\"returned_crc32c\":\"0x") == NULL ||
      strstr(output, "\"match\":true,\"freed\":true,\"reuse\":{") ==
          NULL ||
      strstr(output, "\"simulated_va\":\"0x0000100300000000\"") == NULL ||
      strstr(output, "\"initial_zero\":true,\"freed\":true}") == NULL) {
    fprintf(stderr, "memory CLI JSON schema/roundtrip mismatch: %s\n", output);
    failed = 1;
  }
  if (output_pipe[0] >= 0) {
    (void)close(output_pipe[0]);
  }
  if (stop_memory_server(&server) != 0) {
    failed = 1;
  }
  return failed;
}
#endif

int main(void) {
  int failures = 0;
  failures += test_memory_lifecycle();
  failures += test_resource_ack_is_retryable();
  failures += test_bad_d2h_crc_is_atomic_and_poisoning();
  failures += test_bad_d2h_mode_is_atomic_and_poisoning(
      MEMORY_SERVER_BAD_D2H_SETUID, "setuid");
  failures += test_bad_d2h_mode_is_atomic_and_poisoning(
      MEMORY_SERVER_BAD_D2H_SETGID, "setgid");
  failures += test_bad_d2h_mode_is_atomic_and_poisoning(
      MEMORY_SERVER_BAD_D2H_STICKY, "sticky");
  failures += test_bad_d2h_mode_is_atomic_and_poisoning(
      MEMORY_SERVER_BAD_D2H_SEALS, "missing-final-seals");
  failures += test_post_ack_cancel_is_atomic_and_retryable();
  failures += test_post_ack_deadline_precedes_cancel();
  failures += test_alloc_ack_timeout_poisons();
  failures += test_memory_capability_rejection();
  failures += test_queue_completion_interleaves_with_memory_ack();
#ifdef SAGR_MEMORY_CLI_PATH
  failures += test_memory_cli_roundtrip();
#endif
  return failures == 0 ? 0 : 1;
}
