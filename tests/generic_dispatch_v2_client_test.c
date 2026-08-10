/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include "transport_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <self_amdgpu_runtime/runtime.h>

static const uint8_t k_daemon_uuid[16] = {
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
static const uint8_t k_job_uuid[16] = {
    0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87,
    0x98, 0xa9, 0xba, 0xcb, 0xdc, 0xed, 0xfe, 0x0f};
static const uint8_t k_server_nonce[16] = {
    0xf0, 0xe0, 0xd0, 0xc0, 0xb0, 0xa0, 0x90, 0x80,
    0x70, 0x60, 0x50, 0x40, 0x30, 0x20, 0x10, 0x01};

static const uint64_t k_all_caps =
    SAGR_CAPABILITY_TOPOLOGY_MASK | SAGR_CAPABILITY_QUEUE_MASK |
    SAGR_CAPABILITY_MEMORY_MASK | SAGR_CAPABILITY_SIGNAL_MASK |
    SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK |
    SAGR_CAPABILITY_GENERIC_DISPATCH_MASK;

enum mock_mode {
  MOCK_FULL,
  MOCK_FULL_FAILURE,
  MOCK_NO_GENERIC,
  MOCK_SIGNAL_ONLY
};

typedef struct mock_server {
  char directory[128];
  char endpoint[160];
  int listener;
  pthread_t thread;
  enum mock_mode mode;
  int thread_error;
} mock_server_t;

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

static int send_frame(int peer, const uint8_t *frame, size_t frame_size) {
  return send(peer, frame, frame_size, MSG_NOSIGNAL) == (ssize_t)frame_size
             ? 0
             : -1;
}

static uint64_t request_id_from_frame(const uint8_t *frame) {
  return get_u64(frame + 24U);
}

static uint16_t message_type_from_frame(const uint8_t *frame) {
  return get_u16(frame + 14U);
}

static int send_handshake_ack(int peer, const uint8_t *hello,
                              ssize_t hello_size, uint64_t capabilities,
                              sagr_instance_info_t *info,
                              uint64_t *last_request_id) {
  sagr_wire_ack_fields_t fields;
  uint8_t ack[SAGR_WIRE_ACK_FRAME_BYTES];
  size_t ack_size = 0U;
  if (hello_size < (ssize_t)(SAGR_WIRE_HEADER_BYTES +
                             SAGR_WIRE_HELLO_FIXED_BYTES)) {
    return -1;
  }
  memset(&fields, 0, sizeof(fields));
  fields.selected_major = 1U;
  fields.selected_minor = 0U;
  fields.status = SAGR_WIRE_STATUS_OK;
  memcpy(fields.client_nonce, hello + SAGR_WIRE_HEADER_BYTES + 8U, 16U);
  memcpy(fields.server_nonce, k_server_nonce, sizeof(fields.server_nonce));
  fields.selected_capabilities[0] = capabilities;
  fields.maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  fields.request_id = request_id_from_frame(hello);
  memcpy(fields.daemon_uuid, k_daemon_uuid, sizeof(fields.daemon_uuid));
  fields.connection_id = UINT64_C(0x1122334455667788);
  fields.epoch = UINT64_C(0x0102030405060708);
  memcpy(fields.job_uuid, k_job_uuid, sizeof(fields.job_uuid));
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
  memcpy(info->daemon_uuid, fields.daemon_uuid, sizeof(info->daemon_uuid));
  info->connection_id = fields.connection_id;
  info->epoch = fields.epoch;
  *last_request_id = fields.request_id;
  return 0;
}

static int receive_record_with_fd(int peer, uint8_t *frame, size_t capacity,
                                  size_t *frame_size, int *descriptor) {
  struct msghdr message;
  struct iovec vector;
  unsigned char control[CMSG_SPACE(sizeof(int) * 2U)];
  struct cmsghdr *control_message;
  ssize_t received;
  *descriptor = -1;
  memset(&message, 0, sizeof(message));
  vector.iov_base = frame;
  vector.iov_len = capacity;
  message.msg_iov = &vector;
  message.msg_iovlen = 1U;
  message.msg_control = control;
  message.msg_controllen = sizeof(control);
  received = recvmsg(peer, &message, MSG_CMSG_CLOEXEC);
  if (received <= 0 || (message.msg_flags & MSG_TRUNC) != 0) {
    return received == 0 ? 1 : -1;
  }
  for (control_message = CMSG_FIRSTHDR(&message); control_message != NULL;
       control_message = CMSG_NXTHDR(&message, control_message)) {
    if (control_message->cmsg_level == SOL_SOCKET &&
        control_message->cmsg_type == SCM_RIGHTS &&
        control_message->cmsg_len >= CMSG_LEN(sizeof(int))) {
      memcpy(descriptor, CMSG_DATA(control_message), sizeof(int));
      break;
    }
  }
  *frame_size = (size_t)received;
  return 0;
}

static int send_queue_response(int peer, const sagr_instance_info_t *info,
                               uint64_t request_id, uint16_t message_type,
                               uint16_t opcode, uint64_t queue_id,
                               uint64_t generation, uint64_t sequence,
                               uint64_t value, uint64_t sim_tick) {
  sagr_wire_queue_response_t response;
  uint8_t frame[SAGR_WIRE_QUEUE_FRAME_BYTES];
  size_t frame_size = 0U;
  memset(&response, 0, sizeof(response));
  response.major = SAGR_QUEUE_PROTOCOL_MAJOR;
  response.minor = SAGR_QUEUE_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = opcode;
  response.queue_id = queue_id;
  response.generation = generation;
  response.sequence = sequence;
  response.value = value;
  response.sim_tick = sim_tick;
  if (sagr_protocol_encode_queue_response(
          info, request_id, message_type, &response, frame, sizeof(frame),
          &frame_size) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static int send_signal_response(int peer, const sagr_instance_info_t *info,
                                uint64_t request_id, uint16_t opcode,
                                uint64_t signal_id, uint64_t generation,
                                uint64_t value_bits) {
  sagr_wire_signal_response_t response;
  uint8_t frame[SAGR_WIRE_SIGNAL_FRAME_BYTES];
  size_t frame_size = 0U;
  memset(&response, 0, sizeof(response));
  response.major = SAGR_SIGNAL_PROTOCOL_MAJOR;
  response.minor = SAGR_SIGNAL_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = opcode;
  response.signal_id = signal_id;
  response.generation = generation;
  response.value_bits = value_bits;
  response.sim_tick = 2U;
  if (sagr_protocol_encode_signal_response(
          info, request_id, SAGR_WIRE_MESSAGE_SIGNAL_ACK, &response, frame,
          sizeof(frame), &frame_size) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static int send_memory_response(int peer, const sagr_instance_info_t *info,
                                uint64_t request_id, uint16_t opcode,
                                uint64_t allocation_id, uint64_t generation,
                                uint64_t offset, uint64_t byte_count,
                                uint32_t crc) {
  sagr_wire_memory_response_t response;
  uint8_t frame[SAGR_WIRE_MEMORY_FRAME_BYTES];
  size_t frame_size = 0U;
  memset(&response, 0, sizeof(response));
  response.major = SAGR_MEMORY_PROTOCOL_MAJOR;
  response.minor = SAGR_MEMORY_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = opcode;
  response.allocation_id = allocation_id;
  response.generation = generation;
  response.value0 = offset;
  response.value1 = byte_count;
  response.value2 = crc;
  response.sim_tick = 3U;
  if (sagr_protocol_encode_memory_response(
          info, request_id, &response, frame, sizeof(frame), &frame_size) !=
      SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static int send_generic_response(int peer, const sagr_instance_info_t *info,
                                  uint64_t request_id, uint16_t message_type,
                                  const sagr_wire_generic_response_t *response) {
  uint8_t frame[SAGR_WIRE_GENERIC_FRAME_BYTES];
  size_t frame_size = 0U;
  if (sagr_protocol_encode_generic_dispatch_response(
          info, request_id, message_type, response, frame, sizeof(frame),
          &frame_size) != SAGR_STATUS_SUCCESS) {
    return -1;
  }
  return send_frame(peer, frame, frame_size);
}

static int check_sealed_descriptor(int descriptor, uint64_t byte_count,
                                   const uint8_t *expected,
                                   uint32_t expected_crc) {
  struct stat attributes;
  uint8_t *bytes;
  uint32_t actual_crc;
  int seals;
  if (descriptor < 0 || fstat(descriptor, &attributes) != 0 ||
      attributes.st_size != (off_t)byte_count ||
      (fcntl(descriptor, F_GETFD) & FD_CLOEXEC) == 0 ||
      (fcntl(descriptor, F_GETFL) & O_ACCMODE) != O_RDWR) {
    return -1;
  }
  seals = fcntl(descriptor, F_GET_SEALS);
  if (seals != (F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL)) {
    return -1;
  }
  bytes = (uint8_t *)malloc((size_t)byte_count);
  if (bytes == NULL ||
      pread(descriptor, bytes, (size_t)byte_count, 0) !=
          (ssize_t)byte_count) {
    free(bytes);
    return -1;
  }
  actual_crc = sagr_crc32c(bytes, (size_t)byte_count);
  if (actual_crc != expected_crc ||
      (expected != NULL && memcmp(bytes, expected, (size_t)byte_count) != 0)) {
    free(bytes);
    return -1;
  }
  free(bytes);
  return 0;
}

static int handle_generic_request(
    int peer, const sagr_instance_info_t *info, const uint8_t *frame,
    size_t frame_size, uint64_t request_id, const uint8_t image_sha256[32],
    int fail_submit_completion) {
  sagr_wire_generic_request_t request;
  sagr_wire_generic_response_t response;
  uint64_t decoded_request_id = 0U;
  const char *reason = NULL;
  sagr_status_t status;

  status = sagr_protocol_decode_generic_dispatch_request(
      frame, frame_size, info, &request, &decoded_request_id, &reason);
  if (status != SAGR_STATUS_SUCCESS || decoded_request_id != request_id) {
    return -1;
  }
  memset(&response, 0, sizeof(response));
  response.major = SAGR_GENERIC_DISPATCH_PROTOCOL_MAJOR;
  response.minor = SAGR_GENERIC_DISPATCH_PROTOCOL_MINOR;
  response.status = SAGR_WIRE_STATUS_OK;
  response.opcode = request.opcode;
  response.object_id = request.object_id;
  response.object_generation = request.object_generation;
  response.mapping_id = request.opcode == SAGR_WIRE_GENERIC_OPCODE_MAP_OBJECT
                            ? UINT64_C(0x101)
                            : request.mapping_id;
  response.mapping_generation =
      request.opcode == SAGR_WIRE_GENERIC_OPCODE_MAP_OBJECT
          ? UINT64_C(0x202)
          : request.mapping_generation;
  if (request.opcode == SAGR_WIRE_GENERIC_OPCODE_MAP_OBJECT) {
    response.mapped_base_va = UINT64_C(0x0000200000000000);
    response.mapped_end_va = UINT64_C(0x0000200000003000);
    response.descriptor_va = UINT64_C(0x0000200000000400);
    response.code_va = UINT64_C(0x0000200000001000);
    response.entry_va = UINT64_C(0x0000200000001000);
    response.mapped_bytes = UINT64_C(0x3000);
    response.kernel_index = request.kernel_index;
    response.segment_count = 3U;
    memcpy(response.image_sha256, request.image_sha256,
           sizeof(response.image_sha256));
  } else if (request.opcode == SAGR_WIRE_GENERIC_OPCODE_ALLOC_KERNARG) {
    response.kernarg_allocation_id = UINT64_C(0x303);
    response.kernarg_generation = UINT64_C(0x404);
    response.kernarg_va = UINT64_C(0x0000200000004000);
    response.kernarg_size = request.body.alloc_kernarg.size_bytes;
    response.kernarg_alignment = request.body.alloc_kernarg.alignment_bytes;
    memcpy(response.image_sha256, image_sha256,
           sizeof(response.image_sha256));
  } else if (request.opcode == SAGR_WIRE_GENERIC_OPCODE_SUBMIT_AQL) {
    response.mapping_id = request.mapping_id;
    response.mapping_generation = request.mapping_generation;
    response.kernarg_allocation_id = request.body.submit.kernarg_allocation_id;
    response.kernarg_generation = request.body.submit.kernarg_generation;
    response.kernarg_va = UINT64_C(0x0000200000004000) +
                          request.body.submit.kernarg_offset;
    response.kernarg_size = request.body.submit.kernarg_size;
    response.kernarg_alignment = 8U;
    response.queue_id = request.queue_id;
    response.queue_generation = request.queue_generation;
    response.queue_sequence = request.queue_sequence;
    response.signal_id = request.body.submit.signal_id;
    response.signal_generation = request.body.submit.signal_generation;
    response.signal_value_bits = UINT64_C(1);
    response.ticket_id = UINT64_C(0x505);
    response.trace_id = UINT64_C(0x606);
    response.packet_va = UINT64_C(0x0000200000005000);
    response.packet_crc32c = UINT32_C(0x12345678);
    response.admission_tick = UINT64_C(100);
    memcpy(response.image_sha256, request.image_sha256,
           sizeof(response.image_sha256));
  } else if (request.opcode == SAGR_WIRE_GENERIC_OPCODE_UNMAP_OBJECT) {
    response.mapping_id = request.mapping_id;
    response.mapping_generation = request.mapping_generation;
    /* UNMAP is identity-only; the response hash is intentionally zero. */
  } else {
    return -1;
  }
  if (send_generic_response(peer, info, request_id,
                            SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_ACK,
                            &response) != 0) {
    return -1;
  }
  if (request.opcode == SAGR_WIRE_GENERIC_OPCODE_SUBMIT_AQL) {
    if (fail_submit_completion != 0) {
      memset(&response, 0, sizeof(response));
      response.major = SAGR_GENERIC_DISPATCH_PROTOCOL_MAJOR;
      response.minor = SAGR_GENERIC_DISPATCH_PROTOCOL_MINOR;
      response.status = SAGR_WIRE_STATUS_INTERNAL;
      response.opcode = SAGR_WIRE_GENERIC_OPCODE_SUBMIT_AQL;
      response.error_code = 1U;
      return send_generic_response(
          peer, info, request_id,
          SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_COMPLETION, &response);
    }
    response.sim_tick = UINT64_C(130);
    response.start_tick = UINT64_C(110);
    response.end_tick = UINT64_C(120);
    response.retire_tick = UINT64_C(125);
    response.output_crc32c = UINT32_C(0xabcdef01);
    if (send_generic_response(
            peer, info, request_id,
            SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_COMPLETION, &response) != 0) {
      return -1;
    }
  }
  return 0;
}

static int handle_memory_request(int peer, const sagr_instance_info_t *info,
                                 const uint8_t *frame, size_t frame_size,
                                 int descriptor, const uint8_t *expected,
                                 uint64_t expected_size,
                                 uint64_t expected_offset) {
  const uint8_t *payload = frame + SAGR_WIRE_HEADER_BYTES;
  const uint64_t request_id = request_id_from_frame(frame);
  const uint16_t opcode = get_u16(payload + 4U);
  const uint64_t allocation_id = get_u64(payload + 8U);
  const uint64_t generation = get_u64(payload + 16U);
  const uint64_t offset = get_u64(payload + 24U);
  const uint64_t byte_count = get_u64(payload + 32U);
  const uint32_t expected_crc = (uint32_t)get_u64(payload + 40U);
  uint8_t *copy = NULL;
  int result = -1;
  if (frame_size != SAGR_WIRE_MEMORY_FRAME_BYTES ||
      opcode != SAGR_WIRE_MEMORY_OPCODE_COPY_H2D || allocation_id == 0U ||
      generation == 0U || byte_count != expected_size ||
      offset != expected_offset || expected_crc == 0U || descriptor < 0) {
    goto done;
  }
  copy = (uint8_t *)malloc((size_t)byte_count);
  if (copy == NULL ||
      pread(descriptor, copy, (size_t)byte_count, 0) != (ssize_t)byte_count ||
      sagr_crc32c(copy, (size_t)byte_count) != expected_crc ||
      memcmp(copy, expected, (size_t)byte_count) != 0 ||
      check_sealed_descriptor(descriptor, byte_count, expected, expected_crc) !=
          0) {
    goto done;
  }
  result = send_memory_response(peer, info, request_id, opcode, allocation_id,
                                generation, offset, byte_count, expected_crc);
done:
  free(copy);
  if (descriptor >= 0) {
    (void)close(descriptor);
  }
  return result;
}

static int handle_queue_request(int peer, const sagr_instance_info_t *info,
                                const uint8_t *frame, size_t frame_size) {
  const uint8_t *payload = frame + SAGR_WIRE_HEADER_BYTES;
  const uint64_t request_id = request_id_from_frame(frame);
  const uint16_t opcode = get_u16(payload + 4U);
  const uint64_t queue_id = get_u64(payload + 8U);
  const uint64_t generation = get_u64(payload + 16U);
  const uint64_t sequence = get_u64(payload + 24U);
  const uint64_t argument = get_u64(payload + 32U);
  if (frame_size != SAGR_WIRE_QUEUE_FRAME_BYTES) {
    return -1;
  }
  if (opcode == SAGR_WIRE_QUEUE_OPCODE_CREATE) {
    return send_queue_response(peer, info, request_id,
                               SAGR_WIRE_MESSAGE_QUEUE_ACK, opcode,
                               UINT64_C(0x1020304050607080),
                               UINT64_C(0x8877665544332211), 0U, argument, 1U);
  }
  if (opcode == SAGR_WIRE_QUEUE_OPCODE_DESTROY) {
    return send_queue_response(peer, info, request_id,
                               SAGR_WIRE_MESSAGE_QUEUE_ACK, opcode, queue_id,
                               generation, 0U, 0U, 4U);
  }
  if (opcode == SAGR_WIRE_QUEUE_OPCODE_DOORBELL) {
    if (send_queue_response(peer, info, request_id,
                            SAGR_WIRE_MESSAGE_QUEUE_ACK, opcode, queue_id,
                            generation, sequence, 0U, 10U) != 0) {
      return -1;
    }
    return send_queue_response(peer, info, request_id,
                               SAGR_WIRE_MESSAGE_QUEUE_COMPLETION, opcode,
                               queue_id, generation, sequence, argument, 11U);
  }
  return -1;
}

static int handle_signal_request(int peer, const sagr_instance_info_t *info,
                                 const uint8_t *frame, size_t frame_size) {
  const uint8_t *payload = frame + SAGR_WIRE_HEADER_BYTES;
  const uint64_t request_id = request_id_from_frame(frame);
  const uint16_t opcode = get_u16(payload + 4U);
  const uint64_t signal_id = get_u64(payload + 8U);
  const uint64_t generation = get_u64(payload + 16U);
  const uint64_t value_bits = get_u64(payload + 32U);
  if (frame_size != SAGR_WIRE_SIGNAL_FRAME_BYTES) {
    return -1;
  }
  if (opcode == SAGR_WIRE_SIGNAL_OPCODE_CREATE) {
    return send_signal_response(peer, info, request_id, opcode, 7U, 8U,
                                value_bits);
  }
  if (opcode == SAGR_WIRE_SIGNAL_OPCODE_DESTROY) {
    return send_signal_response(peer, info, request_id, opcode, signal_id,
                                generation, 0U);
  }
  return -1;
}

static void *mock_server_main(void *opaque) {
  mock_server_t *server = (mock_server_t *)opaque;
  struct sockaddr_un address;
  socklen_t address_size = (socklen_t)sizeof(address);
  uint8_t hello[SAGR_WIRE_MAX_HANDSHAKE_BYTES];
  uint8_t frame[SAGR_WIRE_MAX_RECORD_BYTES];
  sagr_instance_info_t info;
  uint64_t last_request_id = 0U;
  uint64_t capabilities = k_all_caps;
  int peer;

  peer = accept4(server->listener, (struct sockaddr *)&address, &address_size,
                 SOCK_CLOEXEC);
  if (peer < 0) {
    server->thread_error = errno;
    return NULL;
  }
  if (server->mode == MOCK_NO_GENERIC) {
    capabilities &= ~SAGR_CAPABILITY_GENERIC_DISPATCH_MASK;
  }
  if (server->mode == MOCK_SIGNAL_ONLY) {
    capabilities = k_all_caps;
  }
  {
    const ssize_t received = recv(peer, hello, sizeof(hello), 0);
    if (received <= 0 ||
        send_handshake_ack(peer, hello, received, capabilities, &info,
                           &last_request_id) != 0) {
      server->thread_error = EPROTO;
      (void)close(peer);
      return NULL;
    }
  }
  for (;;) {
    size_t frame_size = 0U;
    int descriptor = -1;
    int receive_status = receive_record_with_fd(
        peer, frame, sizeof(frame), &frame_size, &descriptor);
    uint16_t message_type;
    uint64_t request_id;
    if (receive_status == 1) {
      break;
    }
    if (receive_status != 0) {
      server->thread_error = errno == 0 ? EPROTO : errno;
      break;
    }
    message_type = message_type_from_frame(frame);
    request_id = request_id_from_frame(frame);
    if (request_id == 0U || request_id <= last_request_id) {
      if (descriptor >= 0) {
        (void)close(descriptor);
      }
      server->thread_error = EPROTO;
      break;
    }
    last_request_id = request_id;
    if (message_type == SAGR_WIRE_MESSAGE_QUEUE_REQUEST) {
      if (server->mode == MOCK_SIGNAL_ONLY ||
          handle_queue_request(peer, &info, frame, frame_size) != 0) {
        server->thread_error = EPROTO;
        if (descriptor >= 0) {
          (void)close(descriptor);
        }
        break;
      }
    } else if (message_type == SAGR_WIRE_MESSAGE_SIGNAL_REQUEST) {
      if (handle_signal_request(peer, &info, frame, frame_size) != 0) {
        server->thread_error = EPROTO;
        if (descriptor >= 0) {
          (void)close(descriptor);
        }
        break;
      }
    } else if (message_type == SAGR_WIRE_MESSAGE_MEMORY_REQUEST) {
      static const uint8_t expected_bytes[48] = {
          0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
          0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
          0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
          0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
          0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
          0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f};
      if (handle_memory_request(peer, &info, frame, frame_size, descriptor,
                                expected_bytes, sizeof(expected_bytes), 8U) !=
          0) {
        server->thread_error = EPROTO;
        break;
      }
      descriptor = -1;
    } else if (message_type == SAGR_WIRE_MESSAGE_GENERIC_DISPATCH_REQUEST) {
      if (server->mode == MOCK_NO_GENERIC ||
          handle_generic_request(peer, &info, frame, frame_size, request_id,
                                 (const uint8_t[32]){1, 2, 3, 4, 5, 6, 7, 8,
                                                     9, 10, 11, 12, 13, 14,
                                                     15, 16, 17, 18, 19, 20,
                                                     21, 22, 23, 24, 25, 26,
                                                     27, 28, 29, 30, 31, 32},
                                 server->mode == MOCK_FULL_FAILURE) !=
              0) {
        server->thread_error = EPROTO;
        if (descriptor >= 0) {
          (void)close(descriptor);
        }
        break;
      }
    } else {
      server->thread_error = EPROTO;
      if (descriptor >= 0) {
        (void)close(descriptor);
      }
      break;
    }
    if (descriptor >= 0) {
      (void)close(descriptor);
    }
  }
  (void)close(peer);
  return NULL;
}

static int start_server(mock_server_t *server, enum mock_mode mode) {
  struct sockaddr_un address;
  size_t endpoint_size;
  if (server == NULL) {
    return -1;
  }
  memset(server, 0, sizeof(*server));
  server->listener = -1;
  server->mode = mode;
  (void)snprintf(server->directory, sizeof(server->directory),
                 "/tmp/sagr-generic-client-XXXXXX");
  if (mkdtemp(server->directory) == NULL ||
      snprintf(server->endpoint, sizeof(server->endpoint), "%s/socket",
               server->directory) >= (int)sizeof(server->endpoint)) {
    return -1;
  }
  server->listener = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (server->listener < 0) {
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
      pthread_create(&server->thread, NULL, mock_server_main, server) != 0) {
    (void)close(server->listener);
    server->listener = -1;
    (void)unlink(server->endpoint);
    (void)rmdir(server->directory);
    return -1;
  }
  return 0;
}

static int finish_server(mock_server_t *server) {
  int failures = 0;
  if (pthread_join(server->thread, NULL) != 0 || server->thread_error != 0) {
    fprintf(stderr, "generic client mock server failed: %d\n",
            server->thread_error);
    failures = 1;
  }
  if (server->listener >= 0) {
    (void)close(server->listener);
  }
  if (unlink(server->endpoint) != 0 || rmdir(server->directory) != 0) {
    failures = 1;
  }
  return failures;
}

static void initialize_open_options(sagr_instance_open_options_t *options,
                                    int generic_capability) {
  (void)sagr_instance_open_options_init(options, (uint32_t)sizeof(*options));
  memcpy(options->expected_daemon_uuid, k_daemon_uuid,
         sizeof(options->expected_daemon_uuid));
  memcpy(options->expected_job_uuid, k_job_uuid,
         sizeof(options->expected_job_uuid));
  options->expected_epoch = UINT64_C(0x0102030405060708);
  options->expected_rank = 3U;
  options->expected_world_size = 8U;
  options->open_timeout_ns = UINT64_C(1000000000);
  options->offered_capabilities[0] = k_all_caps;
  options->required_capabilities[0] = k_all_caps;
  if (generic_capability == 0) {
    options->offered_capabilities[0] &= ~SAGR_CAPABILITY_GENERIC_DISPATCH_MASK;
    options->required_capabilities[0] &= ~SAGR_CAPABILITY_GENERIC_DISPATCH_MASK;
  }
}

static int open_instance(const mock_server_t *server, int generic_capability,
                         sagr_instance_t *out_instance) {
  sagr_instance_open_options_t options;
  sagr_error_info_t error;
  initialize_open_options(&options, generic_capability);
  if (sagr_instance_open(server->endpoint, &options, out_instance, &error,
                         (uint32_t)sizeof(error)) != SAGR_STATUS_SUCCESS) {
    fprintf(stderr, "generic client handshake failed: %s\n", error.message);
    return -1;
  }
  return 0;
}

static int expect(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "generic dispatch v2 client: %s\n", message);
    return 1;
  }
  return 0;
}

static int run_full_lifecycle(void) {
  mock_server_t server;
  mock_server_t owner_server;
  sagr_instance_t instance = NULL;
  sagr_instance_t other_instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_signal_t signal = NULL;
  sagr_signal_t other_signal = NULL;
  sagr_generic_mapping_t mapping = NULL;
  sagr_generic_kernarg_t kernarg = NULL;
  sagr_queue_info_t queue_info;
  sagr_signal_info_t signal_info;
  sagr_generic_mapping_info_t mapping_info;
  sagr_generic_kernarg_info_t kernarg_info;
  sagr_generic_map_options_t map_options;
  sagr_generic_kernarg_allocate_options_t alloc_options;
  sagr_generic_submit_options_t submit_options;
  sagr_generic_dispatch_ticket_t ticket;
  sagr_generic_dispatch_ticket_t stale_ticket;
  sagr_generic_dispatch_completion_t completion;
  sagr_queue_completion_t queue_completion;
  sagr_signal_create_options_t signal_options;
  sagr_error_info_t error;
  uint8_t kernarg_bytes[48];
  uint64_t sequence = 0U;
  int failures = 0;
  size_t index;

  if (start_server(&server, MOCK_FULL) != 0 ||
      open_instance(&server, 1, &instance) != 0) {
    return 1;
  }
  memset(&queue_info, 0, sizeof(queue_info));
  failures += expect(sagr_queue_create(
                         instance, NULL, NULL, &queue, &queue_info,
                         (uint32_t)sizeof(queue_info), &error,
                         (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                     "v1 queue creation remains available");
  (void)sagr_signal_create_options_init(&signal_options,
                                        (uint32_t)sizeof(signal_options));
  signal_options.initial_value = 1;
  memset(&signal_info, 0, sizeof(signal_info));
  failures += expect(sagr_signal_create(
                         instance, &signal_options, NULL, &signal, &signal_info,
                         (uint32_t)sizeof(signal_info), &error,
                         (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                     "v1 signal creation remains available");

  (void)sagr_generic_map_options_init(&map_options,
                                      (uint32_t)sizeof(map_options));
  map_options.object_id = 7U;
  map_options.object_generation = 9U;
  map_options.kernarg_segment_size = 48U;
  map_options.kernarg_segment_align = 8U;
  for (index = 0; index < sizeof(map_options.image_sha256); ++index) {
    map_options.image_sha256[index] = (uint8_t)(index + 1U);
  }
  (void)snprintf(map_options.kernel_name, sizeof(map_options.kernel_name),
                 "%s", "vecadd");
  failures += expect(sagr_generic_map_object(
                         instance, &map_options, NULL, &mapping, &mapping_info,
                         (uint32_t)sizeof(mapping_info), &error,
                         (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                     "MAP creates an owner-bound mapping lease");
  failures += expect(mapping_info.mapping_id == UINT64_C(0x101) &&
                         mapping_info.mapped_base_va != 0U,
                     "MAP publishes daemon-issued mapping metadata");

  (void)sagr_generic_kernarg_allocate_options_init(
      &alloc_options, (uint32_t)sizeof(alloc_options));
  alloc_options.size_bytes = 64U;
  failures += expect(sagr_generic_alloc_kernarg(
                         mapping, &alloc_options, NULL, &kernarg, &kernarg_info,
                         (uint32_t)sizeof(kernarg_info), &error,
                         (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                     "ALLOC creates an owner-bound kernarg lease");
  failures += expect(kernarg_info.size_bytes == 64U &&
                         kernarg_info.kernarg_va != 0U,
                     "ALLOC publishes daemon-issued kernarg metadata");
  for (index = 0; index < sizeof(kernarg_bytes); ++index) {
    kernarg_bytes[index] = (uint8_t)index;
  }
  failures += expect(sagr_generic_kernarg_copy_from_host(
                         kernarg, 8U, kernarg_bytes, sizeof(kernarg_bytes),
                         NULL, &error, (uint32_t)sizeof(error)) ==
                         SAGR_STATUS_SUCCESS,
                     "kernarg bytes use the sealed v1 H2D carrier");

  (void)sagr_generic_submit_options_init(&submit_options,
                                         (uint32_t)sizeof(submit_options));
  /* The full manifest is a legal subrange of a larger allocation. */
  submit_options.kernarg_offset = 8U;
  submit_options.kernarg_size = 48U;
  submit_options.grid_x = 24832U;
  submit_options.grid_y = 1U;
  submit_options.grid_z = 1U;
  submit_options.workgroup_x = 256U;
  submit_options.workgroup_y = 1U;
  submit_options.workgroup_z = 1U;
  submit_options.num_warps = 4U;
  submit_options.num_ctas = 1U;
  failures += expect(sagr_queue_submit_generic_dispatch(
                         queue, mapping, kernarg, signal, &submit_options, NULL,
                         &ticket, (uint32_t)sizeof(ticket), &error,
                         (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                     "SUBMIT returns an admission ticket without host packet bytes");
  stale_ticket = ticket;
  stale_ticket.mapping_generation++;
  memset(&completion, 0, sizeof(completion));
  failures += expect(sagr_queue_wait_generic_dispatch(
                         queue, &stale_ticket, NULL, &completion,
                         (uint32_t)sizeof(completion), &error,
                         (uint32_t)sizeof(error)) == SAGR_STATUS_BUSY,
                     "stale ticket generation is rejected before receive");

  /* A second owner provides a real cross-instance handle for the local gate. */
  if (start_server(&owner_server, MOCK_SIGNAL_ONLY) != 0 ||
      open_instance(&owner_server, 1, &other_instance) != 0) {
    failures++;
  } else {
    memset(&signal_info, 0, sizeof(signal_info));
    failures += expect(sagr_signal_create(
                           other_instance, &signal_options, NULL, &other_signal,
                           &signal_info, (uint32_t)sizeof(signal_info), &error,
                           (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                       "second owner creates a signal handle");
    failures += expect(sagr_queue_submit_generic_dispatch(
                           queue, mapping, kernarg, other_signal,
                           &submit_options, NULL, &stale_ticket,
                           (uint32_t)sizeof(stale_ticket), &error,
                           (uint32_t)sizeof(error)) ==
                         SAGR_STATUS_INSTANCE_MISMATCH,
                       "cross-owner signal is rejected before wire success");
    if (other_signal != NULL) {
      failures += expect(sagr_signal_destroy(
                             &other_signal, NULL, &error,
                             (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                         "cross-owner test signal is destroyable");
    }
    failures += expect(sagr_instance_close(&other_instance) ==
                           SAGR_STATUS_SUCCESS,
                       "cross-owner instance closes cleanly");
    failures += finish_server(&owner_server);
  }

  failures += expect(sagr_queue_wait_generic_dispatch(
                         queue, &ticket, NULL, &completion,
                         (uint32_t)sizeof(completion), &error,
                         (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                     "WAIT consumes the daemon completion");
  failures += expect(completion.status == SAGR_STATUS_SUCCESS &&
                         completion.kernarg_va ==
                             UINT64_C(0x0000200000004008) &&
                         completion.kernarg_size == 48U &&
                         completion.start_tick == 110U &&
                         completion.retire_tick == 125U,
                     "completion exposes canonical daemon ticks");
  {
    const sagr_status_t unmap_status = sagr_generic_unmap_object(
        &mapping, NULL, &error, (uint32_t)sizeof(error));
    if (unmap_status != SAGR_STATUS_SUCCESS) {
      fprintf(stderr, "generic client UNMAP status=%d wire=%d msg=%s\n",
              unmap_status, error.wire_status, error.message);
    }
    failures += expect(unmap_status == SAGR_STATUS_SUCCESS &&
                           mapping == NULL &&
                           sagr_generic_kernarg_get_info(
                               kernarg, &kernarg_info,
                               (uint32_t)sizeof(kernarg_info)) ==
                               SAGR_STATUS_INVALID_HANDLE,
                       "UNMAP consumes mapping and child kernarg leases");
    if (unmap_status == SAGR_STATUS_SUCCESS) {
      kernarg = NULL;
    }
  }

  failures += expect(sagr_queue_ring_doorbell(
                         queue, SAGR_QUEUE_COMMAND_NOOP, NULL, &sequence, &error,
                         (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                     "v1 queue doorbell remains usable after v2 stages");
  failures += expect(sagr_queue_wait(
                         queue, sequence, NULL, &queue_completion,
                         (uint32_t)sizeof(queue_completion), &error,
                         (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                     "v1 queue completion remains usable after v2 stages");
  failures += expect(sagr_signal_destroy(&signal, NULL, &error,
                                         (uint32_t)sizeof(error)) ==
                         SAGR_STATUS_SUCCESS,
                     "v1 signal destruction remains usable");
  failures += expect(sagr_queue_destroy(&queue, NULL, &error,
                                        (uint32_t)sizeof(error)) ==
                         SAGR_STATUS_SUCCESS,
                     "v1 queue destruction remains usable");
  failures += expect(sagr_instance_close(&instance) == SAGR_STATUS_SUCCESS,
                     "full owner closes cleanly");
  failures += finish_server(&server);
  return failures;
}

static int run_no_capability_gate(void) {
  mock_server_t server;
  sagr_instance_t instance = NULL;
  sagr_generic_mapping_t mapping = NULL;
  sagr_generic_map_options_t options;
  sagr_error_info_t error;
  size_t index;
  int failures = 0;
  if (start_server(&server, MOCK_NO_GENERIC) != 0 ||
      open_instance(&server, 0, &instance) != 0) {
    return 1;
  }
  (void)sagr_generic_map_options_init(&options, (uint32_t)sizeof(options));
  options.object_id = 7U;
  options.object_generation = 9U;
  options.kernarg_segment_size = 48U;
  options.kernarg_segment_align = 8U;
  for (index = 0; index < sizeof(options.image_sha256); ++index) {
    options.image_sha256[index] = (uint8_t)(index + 1U);
  }
  (void)snprintf(options.kernel_name, sizeof(options.kernel_name), "%s",
                 "vecadd");
  failures += expect(sagr_generic_map_object(
                         instance, &options, NULL, &mapping, NULL, 0U, &error,
                         (uint32_t)sizeof(error)) == SAGR_STATUS_NOT_SUPPORTED,
                     "generic API fails closed when capability 8 is absent");
  failures += expect(mapping == NULL &&
                         error.wire_status ==
                             SAGR_WIRE_STATUS_UNSUPPORTED_CAPABILITY,
                     "unsupported capability cannot publish local state");
  failures += expect(sagr_instance_close(&instance) == SAGR_STATUS_SUCCESS,
                     "no-capability instance closes cleanly");
  failures += finish_server(&server);
  return failures;
}

static int run_failed_completion(void) {
  mock_server_t server;
  sagr_instance_t instance = NULL;
  sagr_queue_t queue = NULL;
  sagr_signal_t signal = NULL;
  sagr_generic_mapping_t mapping = NULL;
  sagr_generic_kernarg_t kernarg = NULL;
  sagr_signal_create_options_t signal_options;
  sagr_generic_map_options_t map_options;
  sagr_generic_kernarg_allocate_options_t alloc_options;
  sagr_generic_submit_options_t submit_options;
  sagr_generic_dispatch_ticket_t ticket;
  sagr_generic_dispatch_completion_t completion;
  sagr_error_info_t error;
  int failures = 0;
  size_t index;

  if (start_server(&server, MOCK_FULL_FAILURE) != 0 ||
      open_instance(&server, 1, &instance) != 0) {
    return 1;
  }
  (void)sagr_queue_create(instance, NULL, NULL, &queue, NULL, 0U, &error,
                          (uint32_t)sizeof(error));
  (void)sagr_signal_create_options_init(&signal_options,
                                        (uint32_t)sizeof(signal_options));
  signal_options.initial_value = 1;
  (void)sagr_signal_create(instance, &signal_options, NULL, &signal, NULL, 0U,
                           &error, (uint32_t)sizeof(error));
  (void)sagr_generic_map_options_init(&map_options,
                                      (uint32_t)sizeof(map_options));
  map_options.object_id = 7U;
  map_options.object_generation = 9U;
  map_options.kernarg_segment_size = 48U;
  map_options.kernarg_segment_align = 8U;
  for (index = 0; index < sizeof(map_options.image_sha256); ++index) {
    map_options.image_sha256[index] = (uint8_t)(index + 1U);
  }
  (void)snprintf(map_options.kernel_name, sizeof(map_options.kernel_name),
                 "%s", "vecadd");
  if (sagr_generic_map_object(instance, &map_options, NULL, &mapping, NULL, 0U,
                              &error, (uint32_t)sizeof(error)) !=
          SAGR_STATUS_SUCCESS ||
      sagr_generic_kernarg_allocate_options_init(
          &alloc_options, (uint32_t)sizeof(alloc_options)) !=
          SAGR_STATUS_SUCCESS) {
    failures++;
  }
  alloc_options.size_bytes = 48U;
  failures += expect(sagr_generic_alloc_kernarg(
                         mapping, &alloc_options, NULL, &kernarg, NULL, 0U,
                         &error, (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                     "failure path allocates a kernarg lease");
  (void)sagr_generic_submit_options_init(&submit_options,
                                         (uint32_t)sizeof(submit_options));
  submit_options.kernarg_size = 48U;
  submit_options.grid_x = 256U;
  submit_options.grid_y = 1U;
  submit_options.grid_z = 1U;
  submit_options.workgroup_x = 256U;
  submit_options.workgroup_y = 1U;
  submit_options.workgroup_z = 1U;
  submit_options.num_warps = 4U;
  submit_options.num_ctas = 1U;
  failures += expect(sagr_queue_submit_generic_dispatch(
                         queue, mapping, kernarg, signal, &submit_options, NULL,
                         &ticket, (uint32_t)sizeof(ticket), &error,
                         (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                     "failure path still returns an admission ticket");
  failures += expect(sagr_queue_wait_generic_dispatch(
                         queue, &ticket, NULL, &completion,
                         (uint32_t)sizeof(completion), &error,
                         (uint32_t)sizeof(error)) == SAGR_STATUS_INTERNAL_ERROR,
                     "canonical failed completion returns daemon status");
  failures += expect(completion.status == SAGR_STATUS_INTERNAL_ERROR &&
                         completion.request_id == ticket.request_id &&
                         completion.mapping_id == ticket.mapping_id,
                     "failed completion publishes the admitted tuple");
  failures += expect(sagr_generic_unmap_object(
                         &mapping, NULL, &error, (uint32_t)sizeof(error)) ==
                         SAGR_STATUS_SUCCESS,
                     "failed ticket still permits explicit UNMAP");
  kernarg = NULL;
  failures += expect(sagr_signal_destroy(&signal, NULL, &error,
                                         (uint32_t)sizeof(error)) ==
                         SAGR_STATUS_SUCCESS,
                     "failure path signal cleanup succeeds");
  failures += expect(sagr_queue_destroy(&queue, NULL, &error,
                                        (uint32_t)sizeof(error)) ==
                         SAGR_STATUS_SUCCESS,
                     "failure path queue cleanup succeeds");
  failures += expect(sagr_instance_close(&instance) == SAGR_STATUS_SUCCESS,
                     "failure path instance closes cleanly");
  failures += finish_server(&server);
  return failures;
}

int main(void) {
  int failures = 0;
  failures += run_full_lifecycle();
  failures += run_failed_completion();
  failures += run_no_capability_gate();
  if (failures == 0) {
    puts("{\"api_lifecycle\":true,\"failed_completion\":true,"
         "\"no_capability_gate\":true,\"v1_compatibility\":true,"
         "\"owner_validation\":true,\"kernarg_subrange\":true,"
         "\"full_manifest\":true,\"h2d_offset_nonzero\":true,"
         "\"execution\":false,\"fallback\":false}");
  }
  return failures == 0 ? 0 : 1;
}
