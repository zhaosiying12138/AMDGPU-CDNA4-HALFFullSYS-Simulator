/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef SAGR_HSAKMT_MODEL_PATH
#include <hsakmt/hsakmtmodeliface.h>
#include <hsakmt/linux/kfd_ioctl.h>
#endif
#include <self_amdgpu_runtime/kmt_shim.h>
#include <self_amdgpu_runtime/provider.h>

#include "transport_internal.h"

extern char **environ;

static const uint8_t k_daemon_uuid[16] = {
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
static const uint8_t k_job_uuid[16] = {
    0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87,
    0x98, 0xa9, 0xba, 0xcb, 0xdc, 0xed, 0xfe, 0x0f};
static const uint8_t k_server_nonce[16] = {
    0xf0, 0xe0, 0xd0, 0xc0, 0xb0, 0xa0, 0x90, 0x80,
    0x70, 0x60, 0x50, 0x40, 0x30, 0x20, 0x10, 0x01};

#define TEST_MODEL_LOCAL_MEMORY_BYTES UINT64_C(309237645312)
#define TEST_MODEL_BACKING_BYTES                                       \
  (TEST_MODEL_LOCAL_MEMORY_BYTES +                                    \
   (uint64_t)SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_REGION_BYTES)
#define TEST_LARGE_GPU_ALLOCATION_BYTES UINT64_C(5368709120)

typedef struct mock_server {
  char directory[128];
  char endpoint[160];
  int listener;
  int thread_error;
  int serve_kmt;
  uint64_t first_kmt_sequence;
  uint32_t kmt_request_count;
  uint16_t kmt_operations[64];
  uint32_t backing_export_count;
  uint64_t backing_bytes;
  uint32_t backing_page_bytes;
  int backing_descriptor;
  uint32_t queue_create_count;
  uint32_t stop_after_kmt_requests;
  int unequal_clock_counters;
  pthread_t thread;
} mock_server_t;

typedef struct mock_process_server {
  mock_server_t *shared;
  pid_t pid;
} mock_process_server_t;

static int expect(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "KMT shim test: %s\n", message);
    return 1;
  }
  return 0;
}

static int create_shared_backing(uint64_t byte_count) {
  int descriptor = memfd_create("sagr-kmt-test-backing",
                                MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (descriptor < 0 || ftruncate(descriptor, (off_t)byte_count) != 0 ||
      fcntl(descriptor, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW) != 0) {
    if (descriptor >= 0) {
      (void)close(descriptor);
    }
    return -1;
  }
  return descriptor;
}

static int write_exact_at(int descriptor, uint64_t offset,
                          const void *source, size_t byte_count) {
  const uint8_t *cursor = (const uint8_t *)source;
  size_t remaining = byte_count;

  if (source == NULL || offset > (uint64_t)INT64_MAX) {
    return -1;
  }
  while (remaining != 0U) {
    ssize_t written = pwrite(descriptor, cursor, remaining, (off_t)offset);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    if (written == 0) {
      return -1;
    }
    cursor += (size_t)written;
    remaining -= (size_t)written;
    offset += (uint64_t)written;
  }
  return 0;
}

static int initialize_queue_control(int descriptor, uint64_t backing_bytes) {
  const uint64_t region_bytes =
      SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_REGION_BYTES;
  const uint64_t tail = backing_bytes - region_bytes;
  const uint64_t initial_doorbell =
      SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_INITIAL_VALUE;
  const uint64_t initial_completion = 0;
  uint32_t slot;

  if (descriptor < 0 || backing_bytes < region_bytes) {
    return -1;
  }
  for (slot = 0;
       slot < SAGR_BRIDGE_KMT_SHARED_BACKING_MAXIMUM_DOORBELL_SLOTS; ++slot) {
    if (write_exact_at(
            descriptor,
            tail + SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_REGION_BASE_BYTES +
                (uint64_t)slot *
                    SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_SLOT_BYTES,
            &initial_doorbell, sizeof(initial_doorbell)) != 0 ||
        write_exact_at(
            descriptor,
            tail + SAGR_BRIDGE_KMT_SHARED_BACKING_COMPLETION_REGION_BASE_BYTES +
                (uint64_t)slot *
                    SAGR_BRIDGE_KMT_SHARED_BACKING_COMPLETION_SLOT_BYTES,
            &initial_completion, sizeof(initial_completion)) != 0) {
      return -1;
    }
  }
  return 0;
}

static uint64_t get_be_u64(const uint8_t *source) {
  uint64_t value = 0;
  uint32_t index;
  for (index = 0; index < 8U; ++index) {
    value = (value << 8) | source[index];
  }
  return value;
}

static uint16_t get_be_u16(const uint8_t *source) {
  return (uint16_t)(((uint16_t)source[0] << 8) | source[1]);
}

static uint32_t get_be_u32(const uint8_t *source) {
  return ((uint32_t)source[0] << 24) | ((uint32_t)source[1] << 16) |
         ((uint32_t)source[2] << 8) | source[3];
}

static void put_be_u16(uint8_t *destination, uint16_t value) {
  destination[0] = (uint8_t)(value >> 8);
  destination[1] = (uint8_t)value;
}

static void put_be_u32(uint8_t *destination, uint32_t value) {
  destination[0] = (uint8_t)(value >> 24);
  destination[1] = (uint8_t)(value >> 16);
  destination[2] = (uint8_t)(value >> 8);
  destination[3] = (uint8_t)value;
}

static void put_be_u64(uint8_t *destination, uint64_t value) {
  put_be_u32(destination, (uint32_t)(value >> 32));
  put_be_u32(destination + 4, (uint32_t)value);
}

static void decode_request(const uint8_t *frame,
                           sagr_kmt_envelope_request_t *request) {
  const uint8_t *payload = frame + SAGR_WIRE_HEADER_BYTES;
  uint32_t index;
  memset(request, 0, sizeof(*request));
  request->major = get_be_u16(payload);
  request->minor = get_be_u16(payload + 2);
  request->operation = get_be_u16(payload + 4);
  request->flags = get_be_u16(payload + 6);
  request->operation_sequence = get_be_u64(payload + 8);
  request->owner_id = get_be_u64(payload + 16);
  request->owner_generation = get_be_u64(payload + 24);
  request->object_id = get_be_u64(payload + 32);
  request->object_generation = get_be_u64(payload + 40);
  request->auxiliary_id = get_be_u64(payload + 48);
  request->auxiliary_generation = get_be_u64(payload + 56);
  for (index = 0; index < SAGR_KMT_ARGUMENT_WORD_COUNT; ++index) {
    request->argument_words[index] =
        get_be_u32(payload + 64 + (size_t)index * 4U);
  }
  request->buffer_bytes = get_be_u32(payload + 96);
  request->buffer_crc32c = get_be_u32(payload + 100);
  memcpy(request->buffer, payload + 104, SAGR_KMT_BUFFER_BYTES);
}

static void encode_topology_record(uint8_t buffer[64], uint64_t generation) {
  memset(buffer, 0, 64);
  put_be_u16(buffer, 1);
  put_be_u16(buffer + 2, 0);
  put_be_u32(buffer + 4, 1);
  put_be_u64(buffer + 8, generation);
  put_be_u32(buffer + 16, 950);
  put_be_u32(buffer + 20, 1);
  put_be_u32(buffer + 24, 64);
  put_be_u32(buffer + 28, 65536);
  put_be_u32(buffer + 32, 48);
  put_be_u32(buffer + 36, 8);
  put_be_u32(buffer + 40, 1024);
  put_be_u32(buffer + 44, 1024);
  put_be_u16(buffer + 48, 1);
  put_be_u16(buffer + 50, 1);
}

static void encode_process_aperture(uint8_t buffer[56]) {
  memset(buffer, 0, 56);
  put_be_u64(buffer, UINT64_C(0x0001000000000000));
  put_be_u64(buffer + 8, UINT64_C(0x00010000ffffffff));
  put_be_u64(buffer + 16, UINT64_C(0x0002000000000000));
  /* The dGPU HSAKMT path reserves scratch backing from this aperture.  The
   * upstream gfx950 queue-scratch policy may request up to 32 GiB, so the
   * model advertises a 128 GiB bounded VA window with room for alignment and
   * multiple queues.  This is address-space bookkeeping only; no host bytes
   * are committed by the model. */
  put_be_u64(buffer + 24, UINT64_C(0x0002001fffffffff));
  put_be_u64(buffer + 32, UINT64_C(0x0000000000010000));
  put_be_u64(buffer + 40, UINT64_C(0x00007fffffffffff));
  put_be_u32(buffer + 48, 38144U);
}

static int serve_one_kmt(int peer, const sagr_instance_info_t *unused_info,
                         mock_server_t *server) {
  uint8_t frame[SAGR_WIRE_KMT_FRAME_BYTES];
  uint8_t response[SAGR_WIRE_KMT_FRAME_BYTES];
  uint8_t control[CMSG_SPACE(sizeof(int) * 4U)];
  size_t response_size = 0;
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  struct iovec iov;
  struct msghdr message;
  int received_fds[4] = {-1, -1, -1, -1};
  size_t received_fd_count = 0;
  ssize_t count;
  struct cmsghdr *header;
  (void)unused_info;
  memset(&iov, 0, sizeof(iov));
  iov.iov_base = frame;
  iov.iov_len = sizeof(frame);
  memset(&message, 0, sizeof(message));
  message.msg_iov = &iov;
  message.msg_iovlen = 1;
  message.msg_control = control;
  message.msg_controllen = sizeof(control);
  count = recvmsg(peer, &message, MSG_CMSG_CLOEXEC);
  if (count != (ssize_t)sizeof(frame) || get_be_u16(frame + 14) !=
                                             SAGR_KMT_MESSAGE_REQUEST ||
      (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0) {
    return -1;
  }
  for (header = CMSG_FIRSTHDR(&message); header != NULL;
       header = CMSG_NXTHDR(&message, header)) {
    size_t payload_bytes;
    size_t descriptor_count;
    size_t index;
    if (header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS ||
        header->cmsg_len < CMSG_LEN(0)) {
      goto descriptor_failure;
    }
    payload_bytes = header->cmsg_len - CMSG_LEN(0);
    if (payload_bytes == 0 || payload_bytes % sizeof(int) != 0)
      goto descriptor_failure;
    descriptor_count = payload_bytes / sizeof(int);
    if (descriptor_count > 4U - received_fd_count)
      goto descriptor_failure;
    for (index = 0; index < descriptor_count; ++index) {
      memcpy(&received_fds[received_fd_count],
             CMSG_DATA(header) + index * sizeof(int), sizeof(int));
      received_fd_count++;
    }
  }
  decode_request(frame, &request);
  if ((request.operation == SAGR_KMT_OP_EXPORT_BACKING) !=
      (received_fd_count == 1U)) {
    goto descriptor_failure;
  }
  if (request.operation == SAGR_KMT_OP_EXPORT_BACKING) {
    const uint64_t requested_bytes =
        ((uint64_t)request.argument_words[0] << 32) |
        (uint64_t)request.argument_words[1];
    const uint32_t requested_page_bytes = request.argument_words[2];
    const int descriptor_flags = fcntl(received_fds[0], F_GETFD);
    const int status_flags = fcntl(received_fds[0], F_GETFL);
    const int seals = fcntl(received_fds[0], F_GET_SEALS);
    struct stat metadata;
    memset(&metadata, 0, sizeof(metadata));
    if (descriptor_flags < 0 || (descriptor_flags & FD_CLOEXEC) == 0 ||
        status_flags < 0 || (status_flags & O_ACCMODE) != O_RDWR ||
        seals < 0 ||
        (seals & (F_SEAL_SHRINK | F_SEAL_GROW)) !=
            (F_SEAL_SHRINK | F_SEAL_GROW) ||
        fstat(received_fds[0], &metadata) != 0 ||
        !S_ISREG(metadata.st_mode) || metadata.st_uid != getuid() ||
        metadata.st_size < 0 ||
        (uint64_t)metadata.st_size != requested_bytes ||
        requested_page_bytes < 4096U ||
        (requested_page_bytes & (requested_page_bytes - 1U)) != 0U ||
        requested_bytes % requested_page_bytes != 0U) {
      goto descriptor_failure;
    }
    if (server->backing_descriptor >= 0 ||
        initialize_queue_control(received_fds[0], requested_bytes) != 0) {
      goto descriptor_failure;
    }
    server->backing_descriptor = received_fds[0];
    received_fds[0] = -1;
    received_fd_count = 0;
    server->backing_export_count++;
    server->backing_bytes = requested_bytes;
    server->backing_page_bytes = requested_page_bytes;
  }
  if (received_fd_count == 1U) {
    (void)close(received_fds[0]);
    received_fds[0] = -1;
    received_fd_count = 0;
  }
  if (server->kmt_request_count == 0U) {
    server->first_kmt_sequence = request.operation_sequence;
  }
  if (server->kmt_request_count <
      (uint32_t)(sizeof(server->kmt_operations) /
                 sizeof(server->kmt_operations[0]))) {
    server->kmt_operations[server->kmt_request_count] = request.operation;
  }
  server->kmt_request_count++;
  memset(&result, 0, sizeof(result));
  result.major = SAGR_KMT_PROTOCOL_MAJOR;
  result.minor = SAGR_KMT_PROTOCOL_MINOR;
  result.operation = request.operation;
  result.status = SAGR_KMT_STATUS_SUCCESS;
  result.wire_status = SAGR_WIRE_STATUS_OK;
  result.operation_sequence = request.operation_sequence;
  result.owner_id = request.owner_id;
  result.owner_generation = request.owner_generation;
  result.object_id = request.object_id;
  result.object_generation = request.object_generation;
  result.auxiliary_id = request.auxiliary_id;
  result.auxiliary_generation = request.auxiliary_generation;
  if (request.operation == SAGR_KMT_OP_OPEN_KFD) {
    result.owner_id = UINT64_C(0x4b4d540000000001);
    result.owner_generation = 1;
    result.object_id = 0;
    result.object_generation = 0;
  } else if (request.operation == SAGR_KMT_OP_GET_VERSION) {
    result.result_words[0] = 1;
    result.result_words[1] = 9;
    result.result_words[2] = 0;
  } else if (request.operation == SAGR_KMT_OP_GET_CLOCK_COUNTERS) {
    const uint64_t sample = UINT64_C(1000000) + request.operation_sequence;
    const uint64_t cpu_sample =
        server->unequal_clock_counters != 0 ? sample + 1U : sample;
    const uint64_t system_sample =
        server->unequal_clock_counters != 0 ? sample + 2U : sample;
    result.result_words[0] = (uint32_t)(sample >> 32);
    result.result_words[1] = (uint32_t)sample;
    result.result_words[2] = (uint32_t)(cpu_sample >> 32);
    result.result_words[3] = (uint32_t)cpu_sample;
    result.result_words[4] = (uint32_t)(system_sample >> 32);
    result.result_words[5] = (uint32_t)system_sample;
    result.result_words[6] = 0U;
    result.result_words[7] = UINT32_C(1000000000);
  } else if (request.operation == SAGR_KMT_OP_TOPOLOGY_SNAPSHOT) {
    result.object_id = 1;
    result.object_generation = UINT64_C(0x0102030405060708);
    result.result_words[0] = 1;
    result.result_words[1] = 1;
    result.result_words[2] = 0;
    result.result_words[4] = (uint32_t)(UINT64_C(0x0102030405060708) >> 32);
    result.result_words[5] = (uint32_t)UINT64_C(0x0102030405060708);
    result.buffer_bytes = 64;
    encode_topology_record(result.buffer,
                           UINT64_C(0x0102030405060708));
    result.buffer_crc32c = sagr_crc32c(result.buffer, result.buffer_bytes);
  } else if (request.operation == SAGR_KMT_OP_PROCESS_APERTURES) {
    const uint32_t start = request.argument_words[0];
    const uint32_t capacity = request.argument_words[1];
    const uint32_t returned = start == 0U && capacity != 0U ? 1U : 0U;
    result.result_words[0] = start;
    result.result_words[1] = returned;
    result.result_words[2] = 1U;
    if (returned != 0U) {
      result.buffer_bytes = SAGR_KMT_PROCESS_APERTURE_WIRE_BYTES;
      encode_process_aperture(result.buffer);
      result.buffer_crc32c = sagr_crc32c(result.buffer, result.buffer_bytes);
    }
  } else if (request.operation == SAGR_KMT_OP_ALLOC_MEMORY_OF_GPU) {
    /* The facade owns its backing mapping. The peer validates and echoes its
     * opaque offset, including the valid first offset zero. */
    result.object_id = UINT64_C(0x4d454d0000000000) |
                       (uint64_t)request.operation_sequence;
    result.object_generation = UINT64_C(0x0102030405060000) |
                               (uint64_t)request.operation_sequence;
    {
      const uint64_t requested_offset =
          ((uint64_t)request.argument_words[4] << 32) |
          (uint64_t)request.argument_words[5];
      result.result_words[0] = (uint32_t)(requested_offset >> 32);
      result.result_words[1] = (uint32_t)requested_offset;
    }
  } else if (request.operation == SAGR_KMT_OP_ALLOC_MEMORY) {
    result.object_id = UINT64_C(0x4d454d0000000001);
    result.object_generation = UINT64_C(0x0102030405060708);
    result.result_words[1] = request.argument_words[1];
    result.result_words[2] = request.argument_words[2];
    result.result_words[3] = request.argument_words[3];
    result.result_words[4] = request.argument_words[4];
    result.result_words[5] = 0;
    result.result_words[6] = 0x1000;
  } else if (request.operation == SAGR_KMT_OP_MAP_MEMORY_TO_GPU ||
             request.operation == SAGR_KMT_OP_UNMAP_MEMORY_FROM_GPU) {
    /* The upstream HSAKMT path expects every requested GPU mapping to be
     * acknowledged; this is a model response, not host-side arithmetic. */
    result.result_words[0] = request.argument_words[0];
  } else if (request.operation == SAGR_KMT_OP_SET_SCRATCH_BACKING_VA) {
    /* Scratch backing is a model-side VA registration; no host arithmetic or
     * allocation is performed by the transport peer. */
  } else if (request.operation == SAGR_KMT_OP_COPY_MEMORY &&
             request.argument_words[0] == SAGR_KMT_COPY_SIM_TO_HOST) {
    static const uint8_t k_copy_result[] = {'s', 'i', 'm', 'o', 'k'};
    result.buffer_bytes = (uint32_t)sizeof(k_copy_result);
    memcpy(result.buffer, k_copy_result, sizeof(k_copy_result));
    result.buffer_crc32c = sagr_crc32c(result.buffer, result.buffer_bytes);
  } else if (request.operation == SAGR_KMT_OP_QUEUE_CREATE) {
    const uint32_t slot = server->queue_create_count;
    const uint64_t control_tail =
        server->backing_bytes -
        SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_REGION_BYTES;
    const uint64_t doorbell_offset =
        control_tail +
        SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_REGION_BASE_BYTES +
        (uint64_t)slot *
            SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_SLOT_BYTES;
    const uint64_t doorbell = 0;
    const uint64_t completion = 1;
    const uint64_t completion_offset =
        control_tail +
        SAGR_BRIDGE_KMT_SHARED_BACKING_COMPLETION_REGION_BASE_BYTES +
        (uint64_t)slot *
            SAGR_BRIDGE_KMT_SHARED_BACKING_COMPLETION_SLOT_BYTES;
    if (server->backing_descriptor < 0 ||
        slot >= SAGR_BRIDGE_KMT_SHARED_BACKING_MAXIMUM_DOORBELL_SLOTS ||
        write_exact_at(server->backing_descriptor, doorbell_offset, &doorbell,
                       sizeof(doorbell)) != 0 ||
        write_exact_at(server->backing_descriptor, completion_offset,
                       &completion, sizeof(completion)) != 0) {
      goto descriptor_failure;
    }
    server->queue_create_count++;
    result.object_id = UINT64_C(0x5155450000000000) | (uint64_t)(slot + 1U);
    result.object_generation = UINT64_C(0x0102030405060708);
    result.result_words[0] = request.argument_words[2];
    result.result_words[1] = (uint32_t)(doorbell_offset >> 32);
    result.result_words[2] = (uint32_t)doorbell_offset;
  } else if (request.operation == SAGR_KMT_OP_QUEUE_DOORBELL) {
    result.result_words[0] = request.argument_words[0];
    result.result_words[1] = request.argument_words[1];
  } else if (request.operation == SAGR_KMT_OP_EVENT_CREATE) {
    result.object_id = UINT64_C(0x45564e0000000001);
    result.object_generation = UINT64_C(0x0102030405060708);
  } else if (request.operation == SAGR_KMT_OP_EVENT_QUERY) {
    result.result_words[1] = 0;
    result.result_words[2] = 1;
    result.result_words[4] = 1;
  } else if (request.operation == SAGR_KMT_OP_EVENT_WAIT) {
    result.result_words[1] = 0;
  } else if (request.operation == SAGR_KMT_OP_POINTER_INFO) {
    result.result_words[2] = 0;
    result.result_words[3] = 0;
    result.result_words[4] = 4096;
  }
  if (sagr_protocol_encode_kmt_result(
          unused_info, get_be_u64(frame + 24), &result, response,
          sizeof(response), &response_size) != SAGR_STATUS_SUCCESS ||
      send(peer, response, response_size, MSG_NOSIGNAL) !=
          (ssize_t)response_size) {
    return -1;
  }
  return request.operation == SAGR_KMT_OP_CLOSE_KFD ? 1 : 0;

descriptor_failure:
  while (received_fd_count != 0U) {
    received_fd_count--;
    if (received_fds[received_fd_count] >= 0)
      (void)close(received_fds[received_fd_count]);
  }
  return -1;
}

static void *mock_server_main(void *argument) {
  mock_server_t *server = (mock_server_t *)argument;
  uint8_t hello[SAGR_WIRE_MAX_HANDSHAKE_BYTES];
  uint8_t ack[SAGR_WIRE_ACK_FRAME_BYTES];
  sagr_wire_ack_fields_t fields;
  sagr_instance_info_t server_info;
  size_t ack_size = 0;
  ssize_t hello_size;
  int peer = accept4(server->listener, NULL, NULL, SOCK_CLOEXEC);
  struct timeval receive_timeout;
  if (peer < 0) {
    server->thread_error = errno;
    return NULL;
  }
  memset(&receive_timeout, 0, sizeof(receive_timeout));
  receive_timeout.tv_sec = 5;
  if (setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &receive_timeout,
                 (socklen_t)sizeof(receive_timeout)) != 0) {
    server->thread_error = errno;
    (void)close(peer);
    return NULL;
  }
  hello_size = recv(peer, hello, sizeof(hello), 0);
  if (hello_size < SAGR_WIRE_HEADER_BYTES + SAGR_WIRE_HELLO_FIXED_BYTES) {
    server->thread_error = EPROTO;
    (void)close(peer);
    return NULL;
  }
  memset(&fields, 0, sizeof(fields));
  fields.selected_major = 1;
  fields.status = SAGR_WIRE_STATUS_OK;
  memcpy(fields.client_nonce, hello + SAGR_WIRE_HEADER_BYTES + 8, 16);
  memcpy(fields.server_nonce, k_server_nonce, 16);
  fields.selected_capabilities[0] = SAGR_CAPABILITY_TOPOLOGY_MASK;
  if (server->serve_kmt != 0) {
    fields.selected_capabilities[0] |= SAGR_KMT_CAPABILITY_MASK;
  }
  if (server->serve_kmt == 2) {
    fields.selected_capabilities[0] |=
        SAGR_CAPABILITY_QUEUE_MASK | SAGR_CAPABILITY_MEMORY_MASK |
        SAGR_CAPABILITY_SIGNAL_MASK |
        SAGR_CAPABILITY_CODE_OBJECT_TRANSPORT_MASK |
        SAGR_CAPABILITY_GENERIC_DISPATCH_MASK |
        SAGR_CAPABILITY_GENERIC_EXECUTION_MASK;
  }
  fields.maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  fields.request_id = get_be_u64(hello + 24);
  memcpy(fields.daemon_uuid, k_daemon_uuid, 16);
  fields.connection_id = UINT64_C(0x1122334455667788);
  fields.epoch = UINT64_C(0x0102030405060708);
  memcpy(fields.job_uuid, k_job_uuid, 16);
  fields.rank = 0;
  fields.world_size = 1;
  fields.include_topology = 1;
  if (sagr_protocol_encode_ack(&fields, ack, sizeof(ack), &ack_size) !=
          SAGR_STATUS_SUCCESS ||
      send(peer, ack, ack_size, MSG_NOSIGNAL) != (ssize_t)ack_size) {
    server->thread_error = errno == 0 ? EPROTO : errno;
    (void)close(peer);
    return NULL;
  }
  memset(&server_info, 0, sizeof(server_info));
  memcpy(server_info.daemon_uuid, k_daemon_uuid, 16);
  server_info.connection_id = fields.connection_id;
  server_info.epoch = fields.epoch;
  if (server->serve_kmt != 0) {
    for (;;) {
      const int operation_result = serve_one_kmt(peer, &server_info, server);
      if (operation_result < 0) {
        server->thread_error = EPROTO;
        break;
      }
      if (operation_result > 0) {
        break;
      }
      if (server->stop_after_kmt_requests != 0U &&
          server->kmt_request_count >= server->stop_after_kmt_requests) {
        break;
      }
    }
    (void)close(peer);
    return NULL;
  }
  /* Every valid wrapper must reject the missing KMT capability locally; any
   * record after the handshake is a test failure. */
  {
    uint8_t unexpected[512];
    ssize_t count = recv(peer, unexpected, sizeof(unexpected), 0);
    if (count > 0) {
      server->thread_error = EPROTO;
    } else if (count < 0) {
      server->thread_error = errno;
    }
  }
  (void)close(peer);
  return NULL;
}

static int prepare_server(mock_server_t *server, int serve_kmt,
                          uint32_t stop_after_kmt_requests) {
  struct sockaddr_un address;
  size_t endpoint_size;
  memset(server, 0, sizeof(*server));
  server->listener = -1;
  server->backing_descriptor = -1;
  server->serve_kmt = serve_kmt;
  server->stop_after_kmt_requests = stop_after_kmt_requests;
  (void)snprintf(server->directory, sizeof(server->directory),
                 "/tmp/sagr-kmt-test-XXXXXX");
  if (mkdtemp(server->directory) == NULL) {
    return -1;
  }
  if (snprintf(server->endpoint, sizeof(server->endpoint), "%s/socket",
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
      chmod(server->endpoint, 0600) != 0 || listen(server->listener, 1) != 0) {
    return -1;
  }
  return 0;
}

static int start_server_with_options(mock_server_t *server, int serve_kmt,
                                     uint32_t stop_after_kmt_requests,
                                     int unequal_clock_counters) {
  if (prepare_server(server, serve_kmt, stop_after_kmt_requests) != 0) {
    return -1;
  }
  server->unequal_clock_counters = unequal_clock_counters;
  if (pthread_create(&server->thread, NULL, mock_server_main, server) != 0) {
    return -1;
  }
  return 0;
}

static int start_server_with_limit(mock_server_t *server, int serve_kmt,
                                   uint32_t stop_after_kmt_requests) {
  return start_server_with_options(server, serve_kmt,
                                   stop_after_kmt_requests, 0);
}

static int start_server(mock_server_t *server, int serve_kmt) {
  return start_server_with_limit(server, serve_kmt, 0U);
}

static int start_server_with_unequal_clock_counters(mock_server_t *server,
                                                     int serve_kmt) {
  return start_server_with_options(server, serve_kmt, 0U, 1);
}

static int stop_server(mock_server_t *server) {
  int failure = 0;
  if (pthread_join(server->thread, NULL) != 0) {
    failure = 1;
  }
  if (server->listener >= 0) {
    (void)close(server->listener);
  }
  if (server->backing_descriptor >= 0) {
    (void)close(server->backing_descriptor);
    server->backing_descriptor = -1;
  }
  if (server->endpoint[0] != '\0') {
    (void)unlink(server->endpoint);
  }
  if (server->directory[0] != '\0') {
    (void)rmdir(server->directory);
  }
  return failure != 0 || server->thread_error != 0 ? -1 : 0;
}

static int start_process_server(mock_process_server_t *process,
                                int serve_kmt) {
  uint8_t ready = 0U;
  int ready_pipe[2] = {-1, -1};
  pid_t child;
  memset(process, 0, sizeof(*process));
  process->pid = -1;
  process->shared = mmap(NULL, sizeof(*process->shared), PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (process->shared == MAP_FAILED) {
    process->shared = NULL;
    return -1;
  }
  if (pipe2(ready_pipe, O_CLOEXEC) != 0) {
    (void)munmap(process->shared, sizeof(*process->shared));
    process->shared = NULL;
    return -1;
  }
  child = fork();
  if (child < 0) {
    (void)close(ready_pipe[0]);
    (void)close(ready_pipe[1]);
    (void)munmap(process->shared, sizeof(*process->shared));
    process->shared = NULL;
    return -1;
  }
  if (child == 0) {
    (void)close(ready_pipe[0]);
    if (prepare_server(process->shared, serve_kmt, 0U) != 0) {
      (void)write(ready_pipe[1], &ready, sizeof(ready));
      (void)close(ready_pipe[1]);
      _exit(1);
    }
    ready = 1U;
    if (write(ready_pipe[1], &ready, sizeof(ready)) != (ssize_t)sizeof(ready)) {
      (void)close(ready_pipe[1]);
      _exit(1);
    }
    (void)close(ready_pipe[1]);
    (void)mock_server_main(process->shared);
    if (process->shared->listener >= 0) {
      (void)close(process->shared->listener);
    }
    if (process->shared->backing_descriptor >= 0) {
      (void)close(process->shared->backing_descriptor);
    }
    _exit(process->shared->thread_error == 0 ? 0 : 1);
  }
  process->pid = child;
  (void)close(ready_pipe[1]);
  if (read(ready_pipe[0], &ready, sizeof(ready)) != (ssize_t)sizeof(ready) ||
      ready != 1U) {
    int child_status = 0;
    (void)close(ready_pipe[0]);
    (void)waitpid(child, &child_status, 0);
    if (process->shared->endpoint[0] != '\0') {
      (void)unlink(process->shared->endpoint);
    }
    if (process->shared->directory[0] != '\0') {
      (void)rmdir(process->shared->directory);
    }
    (void)munmap(process->shared, sizeof(*process->shared));
    process->shared = NULL;
    process->pid = -1;
    return -1;
  }
  (void)close(ready_pipe[0]);
  return 0;
}

static int stop_process_server(mock_process_server_t *process,
                               mock_server_t *snapshot) {
  int child_status = 0;
  pid_t waited = 0;
  unsigned attempt;
  int failure = 0;
  if (process == NULL || process->shared == NULL || process->pid <= 0 ||
      snapshot == NULL) {
    return -1;
  }
  for (attempt = 0U; attempt != 700U; ++attempt) {
    waited = waitpid(process->pid, &child_status, WNOHANG);
    if (waited == process->pid || waited < 0) {
      break;
    }
    (void)usleep(10000U);
  }
  if (waited != process->pid) {
    (void)kill(process->pid, SIGKILL);
    (void)waitpid(process->pid, &child_status, 0);
    failure = 1;
  } else if (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
    failure = 1;
  }
  memcpy(snapshot, process->shared, sizeof(*snapshot));
  if (process->shared->endpoint[0] != '\0') {
    (void)unlink(process->shared->endpoint);
  }
  if (process->shared->directory[0] != '\0') {
    (void)rmdir(process->shared->directory);
  }
  if (process->shared->thread_error != 0) {
    failure = 1;
  }
  (void)munmap(process->shared, sizeof(*process->shared));
  process->shared = NULL;
  process->pid = -1;
  return failure == 0 ? 0 : -1;
}

static int check_carrier_layout_and_codec(void) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_kmt_envelope_result_t decoded_result;
  sagr_instance_info_t info;
  uint8_t frame[SAGR_WIRE_KMT_FRAME_BYTES];
  size_t frame_size = 0;
  int32_t wire_status = -1;
  const char *reason = NULL;
  int failures = 0;
  _Static_assert(sizeof(sagr_kmt_handle_t) == 32,
                 "KMT handle must contain four u64 identities");
  _Static_assert(sizeof(sagr_kmt_envelope_request_t) == 256,
                 "KMT request payload size");
  _Static_assert(sizeof(sagr_kmt_envelope_result_t) == 256,
                 "KMT result payload size");
  _Static_assert(SAGR_KMT_PROTOCOL_MINOR == 5,
                 "KMT payload minor version");
  _Static_assert(SAGR_KMT_OP_SET_SCRATCH_BACKING_VA == 26,
                 "KMT operation surface");
  _Static_assert(SAGR_KMT_OP_EXPORT_BACKING == 27,
                 "KMT backing export operation");
  _Static_assert(SAGR_KMT_OP_GET_CLOCK_COUNTERS == 28,
                 "KMT clock-counter operation");
  _Static_assert(SAGR_KMT_QUEUE_MAX_DEPTH == UINT32_C(1048576),
                 "KMT queue capacity matches the bridge boundary");
  _Static_assert(sizeof(sagr_kmt_process_aperture_t) == 56,
                 "process aperture value ABI size");
  _Static_assert(offsetof(sagr_kmt_envelope_request_t, owner_generation) == 24,
                 "request owner generation offset");
  _Static_assert(offsetof(sagr_kmt_envelope_request_t, buffer) == 104,
                 "request copied-buffer offset");
  _Static_assert(offsetof(sagr_kmt_envelope_result_t, owner_generation) == 32,
                 "result owner generation offset");
  _Static_assert(offsetof(sagr_kmt_envelope_result_t, buffer) == 112,
                 "result copied-buffer offset");
  memset(&request, 0, sizeof(request));
  memset(&result, 0, sizeof(result));
  memset(&info, 0, sizeof(info));
  memcpy(info.daemon_uuid, k_daemon_uuid, sizeof(info.daemon_uuid));
  info.connection_id = UINT64_C(0x1122334455667788);
  info.epoch = UINT64_C(0x0102030405060708);
  failures += expect(
      sagr_kmt_envelope_request_init(
          &request, (uint32_t)sizeof(request), SAGR_KMT_OP_MODEL_DRM_CALL) ==
          SAGR_KMT_STATUS_SUCCESS,
      "request initializer accepts a known operation");
  request.operation_sequence = 7;
  request.owner_id = 9;
  request.owner_generation = 11;
  request.buffer_bytes = 4;
  memcpy(request.buffer, "KMT!", 4);
  request.buffer_crc32c = sagr_crc32c(request.buffer, request.buffer_bytes);
  failures += expect(sagr_protocol_encode_kmt_request(
                         &info, 5, &request, frame, sizeof(frame), &frame_size) ==
                         SAGR_STATUS_SUCCESS &&
                         frame_size == SAGR_WIRE_KMT_FRAME_BYTES,
                     "request codec emits fixed frame");
  failures += expect(frame[SAGR_WIRE_HEADER_BYTES + 24 + 7] == 11U &&
                         memcmp(frame + SAGR_WIRE_HEADER_BYTES + 104, "KMT!",
                                4) == 0,
                     "wire owner generation and copied buffer offsets");
  result.major = SAGR_KMT_PROTOCOL_MAJOR;
  result.minor = SAGR_KMT_PROTOCOL_MINOR;
  result.operation = request.operation;
  result.status = SAGR_KMT_STATUS_NOT_SUPPORTED;
  result.wire_status = SAGR_WIRE_STATUS_OK;
  result.operation_sequence = request.operation_sequence;
  result.owner_id = request.owner_id;
  result.owner_generation = request.owner_generation;
  failures += expect(sagr_kmt_envelope_result_validate(&request, &result) ==
                         SAGR_KMT_STATUS_SUCCESS,
                     "unsupported result preserves validated identity");
  result.owner_generation ^= 1U;
  failures += expect(sagr_kmt_envelope_result_validate(&request, &result) ==
                         SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR,
                     "owner generation mismatch is rejected");
  result.owner_generation = request.owner_generation;
  failures += expect(
      sagr_protocol_encode_kmt_result(
          &info, 5, &result, frame, sizeof(frame), &frame_size) ==
              SAGR_STATUS_SUCCESS &&
          frame_size == SAGR_WIRE_KMT_FRAME_BYTES,
      "result codec emits the generated KMT v5 frame");
  memset(&decoded_result, 0, sizeof(decoded_result));
  failures += expect(
      sagr_protocol_decode_kmt_result(
          frame, frame_size, &info, 5, &decoded_result, &wire_status,
          &reason) == SAGR_STATUS_SUCCESS &&
          decoded_result.minor == SAGR_KMT_PROTOCOL_MINOR &&
          decoded_result.operation == result.operation &&
          decoded_result.owner_generation == result.owner_generation &&
          wire_status == SAGR_WIRE_STATUS_OK,
      "result codec decodes the generated KMT v5 identity");
  put_be_u16(frame + SAGR_WIRE_HEADER_BYTES + 2, 2);
  sagr_protocol_recompute_frame_crc(frame, frame_size);
  reason = NULL;
  failures += expect(
      sagr_protocol_decode_kmt_result(
          frame, frame_size, &info, 5, &decoded_result, &wire_status,
          &reason) == SAGR_STATUS_PROTOCOL_ERROR &&
          reason != NULL &&
          strcmp(reason, "unsupported KMT result payload version") == 0,
      "result codec rejects a stale KMT payload minor before publication");
  return failures;
}

static int check_missing_capability_wrappers(void) {
  mock_server_t server;
  sagr_instance_open_options_t open_options;
  sagr_provider_t *provider = NULL;
  sagr_error_info_t error;
  sagr_kmt_handle_t kfd = {7, 9, 0, 0};
  sagr_kmt_handle_t resource = {7, 9, 11, 13};
  sagr_kmt_handle_t output_handle = {0xa5, 0xa5, 0xa5, 0xa5};
  sagr_kmt_handle_t sentinel_handle = output_handle;
  sagr_kmt_version_t version;
  sagr_kmt_topology_t topology;
  sagr_kmt_process_aperture_t apertures[SAGR_KMT_PROCESS_APERTURES_PER_PAGE];
  sagr_kmt_alloc_options_t alloc;
  sagr_kmt_memory_info_t memory_info;
  sagr_kmt_copy_options_t copy;
  sagr_kmt_queue_options_t queue;
  sagr_kmt_event_options_t event;
  sagr_kmt_event_result_t event_result;
  sagr_kmt_pointer_info_t pointer_info;
  sagr_kmt_model_drm_call_t drm;
  uint8_t bytes[48];
  uint8_t output[48];
  uint64_t sequence = UINT64_C(0xa5a5a5a5a5a5a5a5);
  uint32_t aperture_returned = UINT32_C(0xa5a5a5a5);
  uint32_t aperture_total = UINT32_C(0xa5a5a5a5);
  int failures = 0;
  if (start_server(&server, 0) != 0) {
    return expect(0, "could not start missing-capability server");
  }
  memset(&open_options, 0, sizeof(open_options));
  (void)sagr_instance_open_options_init(
      &open_options, (uint32_t)sizeof(open_options));
  memcpy(open_options.expected_daemon_uuid, k_daemon_uuid, 16);
  memcpy(open_options.expected_job_uuid, k_job_uuid, 16);
  open_options.expected_epoch = UINT64_C(0x0102030405060708);
  open_options.expected_rank = 0;
  open_options.expected_world_size = 1;
  memset(&error, 0, sizeof(error));
  failures += expect(sagr_provider_open(
                         server.endpoint, &open_options, &provider, &error,
                         (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                     "provider opens without KMT capability");
  memset(&version, 0xa5, sizeof(version));
  memset(&topology, 0xa5, sizeof(topology));
  memset(apertures, 0xa5, sizeof(apertures));
  memset(&alloc, 0, sizeof(alloc));
  alloc.struct_size = (uint32_t)sizeof(alloc);
  alloc.node_id = 1;
  alloc.size_bytes = 4096;
  alloc.alignment_bytes = 4096;
  memset(&memory_info, 0xa5, sizeof(memory_info));
  memset(&copy, 0, sizeof(copy));
  copy.struct_size = (uint32_t)sizeof(copy);
  copy.flags = SAGR_KMT_COPY_HOST_TO_SIM;
  copy.byte_count = sizeof(bytes);
  memset(bytes, 0x5a, sizeof(bytes));
  memset(output, 0xa5, sizeof(output));
  memset(&queue, 0, sizeof(queue));
  queue.struct_size = (uint32_t)sizeof(queue);
  queue.node_id = 1;
  queue.depth = 16384;
  queue.ring_base_address = UINT64_C(0x0000400000000000);
  queue.ring_size_bytes = UINT64_C(1048576);
  queue.read_pointer_address = UINT64_C(0x0000500000000000);
  queue.write_pointer_address = UINT64_C(0x0000500000000008);
  memset(&event, 0, sizeof(event));
  event.struct_size = (uint32_t)sizeof(event);
  memset(&event_result, 0xa5, sizeof(event_result));
  memset(&pointer_info, 0xa5, sizeof(pointer_info));
  memset(&drm, 0, sizeof(drm));
  drm.struct_size = (uint32_t)sizeof(drm);
  drm.argument_bytes = 48;

#define EXPECT_UNSUPPORTED(expression, label)                                  \
  failures += expect((expression) == SAGR_KMT_STATUS_NOT_SUPPORTED, (label))
  EXPECT_UNSUPPORTED(sagr_kmt_open_kfd(provider, &output_handle, NULL, &error,
                                       (uint32_t)sizeof(error)),
                     "open returns unsupported without capability");
  EXPECT_UNSUPPORTED(sagr_kmt_close_kfd(provider, &kfd, NULL, &error,
                                        (uint32_t)sizeof(error)),
                     "close wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_get_version(provider, &kfd, &version,
                                          (uint32_t)sizeof(version), NULL,
                                          &error, (uint32_t)sizeof(error)),
                     "version wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_topology_snapshot(
                         provider, &kfd, &topology,
                         (uint32_t)sizeof(topology), NULL, &error,
                         (uint32_t)sizeof(error)),
                     "topology wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_process_apertures(
                         provider, &kfd, 0U, apertures,
                         SAGR_KMT_PROCESS_APERTURES_PER_PAGE,
                         &aperture_returned, &aperture_total, NULL, &error,
                         (uint32_t)sizeof(error)),
                     "process apertures wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_acquire_vm(provider, &kfd, 38144U, 128U, NULL,
                                         &error, (uint32_t)sizeof(error)),
                     "acquire VM wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_alloc_memory(
                         provider, &kfd, &alloc, &output_handle, &memory_info,
                         (uint32_t)sizeof(memory_info), NULL, &error,
                         (uint32_t)sizeof(error)),
                     "allocation wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_free_memory(provider, &kfd, &resource, NULL,
                                          &error, (uint32_t)sizeof(error)),
                     "free wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_copy_memory(
                         provider, &kfd, &resource, &copy, bytes, NULL, NULL,
                         &error, (uint32_t)sizeof(error)),
                     "copy wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_queue_create(
                         provider, &kfd, &queue, &output_handle, NULL, &error,
                         (uint32_t)sizeof(error)),
                     "queue create wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_queue_destroy(provider, &kfd, &resource, NULL,
                                            &error, (uint32_t)sizeof(error)),
                     "queue destroy wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_queue_doorbell(
                         provider, &kfd, &resource, 1, &sequence, NULL, &error,
                         (uint32_t)sizeof(error)),
                     "doorbell wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_event_create(
                         provider, &kfd, &event, &output_handle, NULL, &error,
                         (uint32_t)sizeof(error)),
                     "event create wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_event_destroy(provider, &kfd, &resource, NULL,
                                            &error, (uint32_t)sizeof(error)),
                     "event destroy wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_event_set(provider, &kfd, &resource, 1, NULL,
                                        &error, (uint32_t)sizeof(error)),
                     "event set wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_event_reset(provider, &kfd, &resource, NULL,
                                          &error, (uint32_t)sizeof(error)),
                     "event reset wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_event_query(
                         provider, &kfd, &resource, &event_result,
                         (uint32_t)sizeof(event_result), NULL, &error,
                         (uint32_t)sizeof(error)),
                     "event query wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_event_wait(
                         provider, &kfd, &resource, SAGR_SIGNAL_CONDITION_EQ, 0,
                         &event_result, (uint32_t)sizeof(event_result), NULL,
                         &error, (uint32_t)sizeof(error)),
                     "event wait wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_pointer_info(
                         provider, &kfd, &resource, &pointer_info,
                         (uint32_t)sizeof(pointer_info), NULL, &error,
                         (uint32_t)sizeof(error)),
                     "pointer-info wrapper");
  EXPECT_UNSUPPORTED(sagr_kmt_model_drm_call(
                         provider, &kfd, &drm, output,
                         (uint32_t)sizeof(output), NULL, &error,
                         (uint32_t)sizeof(error)),
                     "model DRM wrapper");
#undef EXPECT_UNSUPPORTED

  failures += expect(memcmp(&output_handle, &sentinel_handle,
                            sizeof(output_handle)) == 0,
                     "unsupported creates do not mutate output handles");
  failures += expect(sequence == UINT64_C(0xa5a5a5a5a5a5a5a5),
                     "unsupported doorbell does not mutate sequence");
  failures += expect(aperture_returned == UINT32_C(0xa5a5a5a5) &&
                         aperture_total == UINT32_C(0xa5a5a5a5) &&
                         apertures[0].gpu_id == UINT32_C(0xa5a5a5a5),
                     "unsupported aperture query leaves outputs unchanged");
  failures += expect(output[0] == 0xa5U && output[47] == 0xa5U,
                     "unsupported DRM call does not mutate output bytes");
  {
    sagr_kmt_handle_t foreign = resource;
    foreign.owner_generation++;
    failures += expect(sagr_kmt_free_memory(
                           provider, &kfd, &foreign, NULL, &error,
                           (uint32_t)sizeof(error)) ==
                           SAGR_KMT_STATUS_INVALID_HANDLE,
                       "foreign owner generation precedes unsupported");
  }
  failures += expect(sagr_provider_close(&provider) == SAGR_STATUS_SUCCESS,
                     "provider closes");
  failures += expect(stop_server(&server) == 0,
                     "missing-capability server saw no KMT record");
  return failures;
}

static int check_negotiated_kmt_wrappers(void) {
  mock_server_t server;
  mock_server_t second_server;
  sagr_instance_open_options_t open_options;
  sagr_provider_t *provider = NULL;
  sagr_provider_t *second_provider = NULL;
  sagr_error_info_t error;
  sagr_kmt_call_options_t call_options;
  sagr_kmt_handle_t kfd = {0, 0, 0, 0};
  sagr_kmt_handle_t memory = {0, 0, 0, 0};
  sagr_kmt_handle_t large_gpu_memory = {0, 0, 0, 0};
  sagr_kmt_handle_t queue_handle = {0, 0, 0, 0};
  sagr_kmt_handle_t event_handle = {0, 0, 0, 0};
  sagr_kmt_version_t version;
  sagr_kmt_clock_counters_t clock_counters;
  sagr_kmt_topology_t topology;
  sagr_kmt_process_aperture_t apertures[SAGR_KMT_PROCESS_APERTURES_PER_PAGE];
  sagr_kmt_alloc_options_t alloc;
  sagr_kmt_memory_info_t memory_info;
  sagr_kmt_copy_options_t copy;
  sagr_kmt_queue_options_t queue;
  sagr_kmt_event_options_t event;
  sagr_kmt_event_result_t event_result;
  sagr_kmt_pointer_info_t pointer_info;
  sagr_kmt_model_drm_call_t drm;
  uint8_t source[5] = {'h', 'o', 's', 't', '!'};
  uint8_t destination[5] = {0, 0, 0, 0, 0};
  uint8_t drm_result[48];
  uint64_t sequence = 0;
  uint64_t large_mmap_offset = 0;
  uint32_t aperture_returned = UINT32_C(0xa5a5a5a5);
  uint32_t aperture_total = UINT32_C(0xa5a5a5a5);
  int backing_fd = -1;
  int failures = 0;
  if (start_server(&server, 1) != 0) {
    return expect(0, "could not start KMT server");
  }
  memset(&open_options, 0, sizeof(open_options));
  (void)sagr_instance_open_options_init(
      &open_options, (uint32_t)sizeof(open_options));
  memcpy(open_options.expected_daemon_uuid, k_daemon_uuid, 16);
  memcpy(open_options.expected_job_uuid, k_job_uuid, 16);
  open_options.expected_epoch = UINT64_C(0x0102030405060708);
  open_options.expected_rank = 0;
  open_options.expected_world_size = 1;
  open_options.offered_capabilities[0] |= SAGR_KMT_CAPABILITY_MASK;
  open_options.required_capabilities[0] |= SAGR_KMT_CAPABILITY_MASK;
  memset(&error, 0, sizeof(error));
  failures += expect(sagr_kmt_call_options_init(
                         &call_options, (uint32_t)sizeof(call_options)) ==
                         SAGR_KMT_STATUS_SUCCESS,
                     "KMT call options initialize");
  failures += expect(sagr_provider_open(
                         server.endpoint, &open_options, &provider, &error,
                         (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                     "provider opens with KMT capability");
  failures += expect(sagr_kmt_open_kfd(
                         provider, &kfd, &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS &&
                         kfd.owner_id == UINT64_C(0x4b4d540000000001) &&
                         kfd.owner_generation == 1 &&
                         kfd.object_id == 0 && kfd.object_generation == 0,
                     "open creates owner-only KFD handle");
  backing_fd = create_shared_backing(UINT64_C(65536));
  failures += expect(backing_fd >= 0, "shared KMT backing is created");
  if (backing_fd >= 0) {
    failures += expect(
        sagr_kmt_export_backing(provider, &kfd, backing_fd,
                                UINT64_C(65536), 4096U, &call_options,
                                &error, (uint32_t)sizeof(error)) ==
            SAGR_KMT_STATUS_SUCCESS,
        "typed shared-backing descriptor export");
    failures += expect(close(backing_fd) == 0,
                       "sender retains and closes its backing descriptor");
    backing_fd = -1;
  }
  failures += expect(sagr_kmt_get_version(
                         provider, &kfd, &version, (uint32_t)sizeof(version),
                         &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS &&
                         version.major == 1 && version.minor == 9,
                     "typed version operation");
  memset(&clock_counters, 0, sizeof(clock_counters));
  failures += expect(
      sagr_kmt_get_clock_counters(
          provider, &kfd, 38144U, &clock_counters,
          (uint32_t)sizeof(clock_counters), &call_options, &error,
          (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS &&
          clock_counters.gpu_clock_counter != 0U &&
          clock_counters.gpu_clock_counter ==
              clock_counters.cpu_clock_counter &&
          clock_counters.gpu_clock_counter ==
              clock_counters.system_clock_counter &&
          clock_counters.system_clock_frequency_hz == UINT64_C(1000000000),
      "typed coherent clock-counter operation");
  failures += expect(sagr_kmt_topology_snapshot(
                         provider, &kfd, &topology,
                         (uint32_t)sizeof(topology), &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS &&
                         topology.gfx_target_code == 950 &&
                         topology.wavefront_size == 64 &&
                         topology.maximum_allocations == 1024,
                     "typed topology record operation");
  memset(apertures, 0xa5, sizeof(apertures));
  failures += expect(sagr_kmt_process_apertures(
                         provider, &kfd, 0U, NULL, 0U,
                         &aperture_returned, &aperture_total, &call_options,
                         &error, (uint32_t)sizeof(error)) ==
                             SAGR_KMT_STATUS_SUCCESS &&
                         aperture_returned == 0U && aperture_total == 1U,
                     "typed process aperture count operation");
  failures += expect(sagr_kmt_process_apertures(
                         provider, &kfd, 0U, apertures,
                         SAGR_KMT_PROCESS_APERTURES_PER_PAGE,
                         &aperture_returned, &aperture_total, &call_options,
                         &error, (uint32_t)sizeof(error)) ==
                             SAGR_KMT_STATUS_SUCCESS &&
                         aperture_returned == 1U && aperture_total == 1U &&
                         apertures[0].gpu_id == 38144U &&
                         apertures[0].gpuvm_base == UINT64_C(0x10000) &&
                         apertures[0].gpuvm_limit ==
                             UINT64_C(0x00007fffffffffff),
                     "typed process aperture page operation");
  failures += expect(sagr_kmt_process_apertures(
                         provider, &kfd, 1U, apertures,
                         SAGR_KMT_PROCESS_APERTURES_PER_PAGE,
                         &aperture_returned, &aperture_total, &call_options,
                         &error, (uint32_t)sizeof(error)) ==
                             SAGR_KMT_STATUS_SUCCESS &&
                         aperture_returned == 0U && aperture_total == 1U,
                     "typed process aperture terminal page operation");
  failures += expect(sagr_kmt_acquire_vm(
                         provider, &kfd, 38144U, 128U, &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS,
                     "typed VM acquisition operation");
  failures += expect(
      sagr_kmt_set_scratch_backing_va(
          provider, &kfd, 38144U, UINT64_C(0x0002000000000000),
          &call_options, &error, (uint32_t)sizeof(error)) ==
          SAGR_KMT_STATUS_SUCCESS,
      "typed scratch backing VA operation");
  failures += expect(
      sagr_kmt_alloc_memory_of_gpu(
          provider, &kfd, UINT64_C(0x0000300000000000),
          TEST_LARGE_GPU_ALLOCATION_BYTES, 38144U,
          SAGR_BRIDGE_KMT_SHARED_BACKING_USERPTR_MEMORY_FLAG_MASK,
          UINT64_C(0x0000100000000000), &large_gpu_memory,
          &large_mmap_offset, &call_options, &error,
          (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS &&
          large_gpu_memory.object_id != 0U &&
          large_mmap_offset == UINT64_C(0x0000100000000000),
      "GPU allocation wrapper preserves sizes above the managed limit");
  failures += expect(
      sagr_kmt_free_memory_of_gpu(
          provider, &kfd, &large_gpu_memory, &call_options, &error,
          (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS,
      "large GPU allocation wrapper releases its typed resource");
  memset(&alloc, 0, sizeof(alloc));
  alloc.struct_size = (uint32_t)sizeof(alloc);
  alloc.size_bytes = 4096;
  alloc.alignment_bytes = 4096;
  failures += expect(sagr_kmt_alloc_memory(
                         provider, &kfd, &alloc, &memory, &memory_info,
                         (uint32_t)sizeof(memory_info), &call_options, &error,
                         (uint32_t)sizeof(error)) ==
                         SAGR_KMT_STATUS_INVALID_PARAMETER,
                     "allocation rejects noncanonical node zero");
  alloc.node_id = 1;
  failures += expect(sagr_kmt_alloc_memory(
                         provider, &kfd, &alloc, &memory, &memory_info,
                         (uint32_t)sizeof(memory_info), &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS &&
                         memory.object_id == UINT64_C(0x4d454d0000000001) &&
                         memory.object_generation ==
                             UINT64_C(0x0102030405060708) &&
                         memory_info.simulated_gpu_va == 0x1000,
                     "typed allocation operation");
  memset(&copy, 0, sizeof(copy));
  copy.struct_size = (uint32_t)sizeof(copy);
  copy.flags = SAGR_KMT_COPY_HOST_TO_SIM;
  copy.byte_count = sizeof(source);
  failures += expect(sagr_kmt_copy_memory(
                         provider, &kfd, &memory, &copy, source, NULL,
                         &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS,
                     "typed host-to-sim copy operation");
  memset(&copy, 0, sizeof(copy));
  copy.struct_size = (uint32_t)sizeof(copy);
  copy.flags = SAGR_KMT_COPY_SIM_TO_HOST;
  copy.byte_count = sizeof(destination);
  failures += expect(sagr_kmt_copy_memory(
                         provider, &kfd, &memory, &copy, NULL, destination,
                         &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS &&
                         memcmp(destination, "simok", 5) == 0,
                     "typed sim-to-host copied result");
  memset(&queue, 0, sizeof(queue));
  queue.struct_size = (uint32_t)sizeof(queue);
  queue.depth = 16384;
  queue.ring_base_address = UINT64_C(0x0000400000000000);
  queue.ring_size_bytes = UINT64_C(1048576);
  queue.read_pointer_address = UINT64_C(0x0000500000000000);
  queue.write_pointer_address = UINT64_C(0x0000500000000008);
  failures += expect(sagr_kmt_queue_create(
                         provider, &kfd, &queue, &queue_handle, &call_options,
                         &error, (uint32_t)sizeof(error)) ==
                         SAGR_KMT_STATUS_INVALID_PARAMETER,
                     "queue creation rejects noncanonical node zero");
  queue.node_id = 1;
  failures += expect(sagr_kmt_queue_create(
                         provider, &kfd, &queue, &queue_handle, &call_options,
                         &error, (uint32_t)sizeof(error)) ==
                         SAGR_KMT_STATUS_SUCCESS &&
                         queue_handle.object_id == UINT64_C(0x5155450000000001),
                     "typed queue create operation");
  failures += expect(sagr_kmt_queue_doorbell(
                         provider, &kfd, &queue_handle, 1, &sequence,
                         &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS &&
                         sequence == 1,
                     "typed queue doorbell operation");
  memset(&event, 0, sizeof(event));
  event.struct_size = (uint32_t)sizeof(event);
  failures += expect(sagr_kmt_event_create(
                         provider, &kfd, &event, &event_handle, &call_options,
                         &error, (uint32_t)sizeof(error)) ==
                         SAGR_KMT_STATUS_SUCCESS &&
                         event_handle.object_id == UINT64_C(0x45564e0000000001),
                     "typed event create operation");
  memset(&event_result, 0, sizeof(event_result));
  failures += expect(sagr_kmt_event_query(
                         provider, &kfd, &event_handle, &event_result,
                         (uint32_t)sizeof(event_result), &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS &&
                         event_result.ready == 1,
                     "typed event query operation");
  failures += expect(sagr_kmt_event_wait(
                         provider, &kfd, &event_handle, SAGR_SIGNAL_CONDITION_EQ,
                         0, &event_result, (uint32_t)sizeof(event_result),
                         &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS,
                     "typed event wait operation");
  failures += expect(sagr_kmt_event_set(
                         provider, &kfd, &event_handle, 1, &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS,
                     "typed event set operation");
  failures += expect(sagr_kmt_event_reset(
                         provider, &kfd, &event_handle, &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS,
                     "typed event reset operation");
  memset(&pointer_info, 0, sizeof(pointer_info));
  failures += expect(sagr_kmt_pointer_info(
                         provider, &kfd, &memory, &pointer_info,
                         (uint32_t)sizeof(pointer_info), &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS &&
                         pointer_info.size_bytes == 4096,
                     "typed pointer-info operation");
  memset(&drm, 0, sizeof(drm));
  drm.struct_size = (uint32_t)sizeof(drm);
  drm.argument_bytes = 48;
  memset(drm_result, 0xa5, sizeof(drm_result));
  failures += expect(sagr_kmt_model_drm_call(
                         provider, &kfd, &drm, drm_result,
                         (uint32_t)sizeof(drm_result), &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS,
                     "typed model DRM envelope operation");
  failures += expect(sagr_kmt_event_destroy(
                         provider, &kfd, &event_handle, &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS,
                     "typed event destroy operation");
  failures += expect(sagr_kmt_queue_destroy(
                         provider, &kfd, &queue_handle, &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS,
                     "typed queue destroy operation");
  failures += expect(sagr_kmt_free_memory(
                         provider, &kfd, &memory, &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS,
                     "typed memory free operation");
  failures += expect(sagr_kmt_close_kfd(
                         provider, &kfd, &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS,
                     "typed close operation");
  failures += expect(sagr_provider_close(&provider) == SAGR_STATUS_SUCCESS,
                     "KMT provider closes");
  failures += expect(server.backing_export_count == 1U &&
                         server.backing_bytes == UINT64_C(65536) &&
                         server.backing_page_bytes == 4096U,
                     "server validates exactly one shared backing descriptor");
  failures += expect(stop_server(&server) == 0,
                     "KMT server completed every operation");

  /* Operation sequences are scoped to the provider/session, not process
   * global.  A fresh provider must begin at sequence one. */
  if (start_server_with_unequal_clock_counters(&second_server, 1) != 0) {
    failures += expect(0, "could not start second KMT server");
  } else {
    failures += expect(sagr_provider_open(
                           second_server.endpoint, &open_options,
                           &second_provider, &error,
                           (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                       "second provider opens with KMT capability");
    if (second_provider != NULL) {
      sagr_kmt_handle_t second_kfd = {0, 0, 0, 0};
      sagr_kmt_clock_counters_t rejected_counters;
      sagr_kmt_clock_counters_t rejected_sentinel;
      failures += expect(sagr_kmt_open_kfd(
                             second_provider, &second_kfd, &call_options,
                             &error, (uint32_t)sizeof(error)) ==
                             SAGR_KMT_STATUS_SUCCESS,
                         "second provider opens KFD");
      memset(&rejected_counters, 0xa5, sizeof(rejected_counters));
      memcpy(&rejected_sentinel, &rejected_counters,
             sizeof(rejected_sentinel));
      failures += expect(
          sagr_kmt_get_clock_counters(
              second_provider, &second_kfd, 38144U, &rejected_counters,
              (uint32_t)sizeof(rejected_counters), &call_options, &error,
              (uint32_t)sizeof(error)) ==
                  SAGR_KMT_STATUS_KERNEL_COMMUNICATION_ERROR &&
              memcmp(&rejected_counters, &rejected_sentinel,
                     sizeof(rejected_counters)) == 0,
          "unequal clock-counter triple is rejected without output commit");
      failures += expect(sagr_kmt_close_kfd(
                             second_provider, &second_kfd, &call_options,
                             &error, (uint32_t)sizeof(error)) ==
                             SAGR_KMT_STATUS_SUCCESS,
                         "second provider closes KFD");
      failures += expect(sagr_provider_close(&second_provider) ==
                             SAGR_STATUS_SUCCESS,
                         "second provider closes");
    }
    failures += expect(second_server.first_kmt_sequence == 1U,
                       "KMT operation sequence resets per provider");
    failures += expect(stop_server(&second_server) == 0,
                       "second KMT server completed operations");
  }
  return failures;
}

static int check_managed_provider_ownership(void) {
  mock_server_t server;
  sagr_managed_session_options_v2_t options;
  sagr_managed_session_info_t session_info;
  sagr_provider_info_t provider_info;
  sagr_provider_t *provider = NULL;
  sagr_kmt_handle_t kfd = {0, 0, 0, 0};
  sagr_error_info_t error;
  int failures = 0;
  if (start_server(&server, 2) != 0) {
    return expect(0, "could not start managed KMT server");
  }
  memset(&options, 0, sizeof(options));
  failures += expect(
      sagr_managed_session_options_v2_init(
          &options, (uint32_t)sizeof(options)) == SAGR_STATUS_SUCCESS,
      "managed KMT exact options initialize");
  options.flags = SAGR_MANAGED_SESSION_V2_FLAG_EXTERNAL_ENDPOINT;
  options.epoch = UINT64_C(0x0102030405060708);
  options.rank = 0U;
  options.world_size = 1U;
  memcpy(options.job_uuid, k_job_uuid, sizeof(options.job_uuid));
  memcpy(options.endpoint, server.endpoint, strlen(server.endpoint) + 1U);
  memset(&session_info, 0, sizeof(session_info));
  memset(&error, 0, sizeof(error));
  failures += expect(
      sagr_provider_open_managed_v2(
          &options, &provider, &session_info, (uint32_t)sizeof(session_info),
          &error, (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS &&
          provider != NULL && session_info.external_endpoint == 1U &&
          session_info.rank == 0U && session_info.world_size == 1U &&
          session_info.child_pid == 0U &&
          memcmp(session_info.job_uuid, k_job_uuid, sizeof(k_job_uuid)) == 0,
      "managed provider preserves exact external topology");
  memset(&provider_info, 0, sizeof(provider_info));
  if (provider != NULL) {
    pid_t child;
    int child_status = 0;
    failures += expect(
        sagr_provider_get_info(provider, &provider_info,
                               (uint32_t)sizeof(provider_info)) ==
                SAGR_STATUS_SUCCESS &&
            (provider_info.negotiated_capabilities[SAGR_CAPABILITY_KMT_WORD] &
             SAGR_CAPABILITY_KMT_MASK) != 0U &&
            provider_info.rank == 0U && provider_info.world_size == 1U,
        "managed provider records the negotiated KMT capability");
    failures += expect(
        sagr_kmt_open_kfd(provider, &kfd, NULL, &error,
                          (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS,
        "managed provider owns a usable KMT transport");
    failures += expect(
        sagr_provider_discard_inherited(&provider) ==
                SAGR_STATUS_INVALID_ARGUMENT &&
            provider != NULL,
        "creating process cannot discard its live provider");
    child = fork();
    if (child == 0) {
      sagr_kmt_version_t version;
      int child_failures = 0;
      memset(&version, 0xa5, sizeof(version));
      child_failures +=
          sagr_kmt_get_version(provider, &kfd, &version,
                               (uint32_t)sizeof(version), NULL, NULL, 0U) ==
                  SAGR_KMT_STATUS_INVALID_HANDLE &&
              version.major == UINT32_C(0xa5a5a5a5)
          ? 0
          : 1;
      child_failures +=
          sagr_provider_close(&provider) == SAGR_STATUS_INVALID_HANDLE &&
                  provider != NULL
              ? 0
              : 1;
      child_failures +=
          sagr_provider_discard_inherited(&provider) == SAGR_STATUS_SUCCESS &&
                  provider == NULL
              ? 0
              : 1;
      _exit(child_failures == 0 ? 0 : 2);
    }
    failures += expect(
        child > 0 && waitpid(child, &child_status, 0) == child &&
            WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
        "fork child fails closed then discards only its local provider copy");
    failures += expect(
        sagr_kmt_close_kfd(provider, &kfd, NULL, &error,
                           (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS,
        "managed provider closes the KMT owner");
    failures += expect(sagr_provider_close(&provider) == SAGR_STATUS_SUCCESS &&
                           provider == NULL,
                       "managed provider closes its session exactly once");
  }
  failures += expect(server.first_kmt_sequence == 1U &&
                         server.kmt_request_count == 2U,
                     "managed provider sequence is session-local");
  failures += expect(stop_server(&server) == 0,
                     "managed KMT server observed clean disconnect");
  return failures;
}

#ifdef SAGR_HSAKMT_MODEL_PATH
static int check_hsakmt_model_get_version(void) {
  mock_server_t server;
  mock_process_server_t process_server;
  get_hsakmt_model_functions_t getter;
  const struct hsakmt_model_functions *functions = NULL;
  struct kfd_ioctl_get_version_args version;
  struct kfd_ioctl_get_process_apertures_new_args aperture_query;
  struct kfd_process_device_apertures aperture;
  struct hsakmt_drm_open_render_args open_render;
  struct hsakmt_drm_close_args close_render;
  struct kfd_ioctl_acquire_vm_args acquire_vm;
  struct kfd_ioctl_alloc_memory_of_gpu_args first_allocation;
  struct kfd_ioctl_alloc_memory_of_gpu_args second_allocation;
  struct kfd_ioctl_alloc_memory_of_gpu_args reused_allocation;
  struct kfd_ioctl_alloc_memory_of_gpu_args userptr_allocation;
  struct kfd_ioctl_alloc_memory_of_gpu_args capacity_first;
  struct kfd_ioctl_alloc_memory_of_gpu_args capacity_second;
  struct kfd_ioctl_alloc_memory_of_gpu_args capacity_doorbell;
  struct kfd_ioctl_alloc_memory_of_gpu_args capacity_overflow;
  struct kfd_ioctl_free_memory_of_gpu_args free_allocation;
  const char *previous_endpoint = getenv("SAGR_GENERIC_BRIDGE_ENDPOINT");
  char *saved_endpoint = previous_endpoint != NULL ? strdup(previous_endpoint)
                                                   : NULL;
  void *library = NULL;
  int render_fd = -1;
  void *mapped_backing = MAP_FAILED;
  int failures = 0;
  if (previous_endpoint != NULL && saved_endpoint == NULL) {
    return expect(0, "could not preserve managed endpoint environment");
  }
  if (start_process_server(&process_server, 2) != 0) {
    free(saved_endpoint);
    return expect(0, "could not start HSAKMT model server");
  }
  failures += expect(setenv("SAGR_GENERIC_BRIDGE_ENDPOINT",
                            process_server.shared->endpoint, 1) == 0,
                     "HSAKMT model endpoint is explicit");
  library = dlopen(SAGR_HSAKMT_MODEL_PATH, RTLD_NOW | RTLD_LOCAL);
  failures += expect(library != NULL, "HSAKMT model DSO loads for KMT gate");
  getter = library != NULL
               ? (get_hsakmt_model_functions_t)dlsym(
                     library, "get_hsakmt_model_functions")
               : NULL;
  failures += expect(getter != NULL, "HSAKMT model getter resolves");
  if (getter != NULL) {
    functions = getter();
  }
  memset(&version, 0xa5, sizeof(version));
  memset(&aperture_query, 0, sizeof(aperture_query));
  memset(&aperture, 0xa5, sizeof(aperture));
  if (functions != NULL) {
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_GET_VERSION, &version) == 0 &&
                           version.major_version == 1U &&
                           version.minor_version == 9U,
                       "official HSAKMT GET_VERSION crosses typed KMT bridge");
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_GET_PROCESS_APERTURES_NEW,
                           &aperture_query) == 0 &&
                           aperture_query.num_of_nodes == 1U,
                       "official HSAKMT aperture count crosses typed bridge");
    aperture_query.kfd_process_device_apertures_ptr =
        (uint64_t)(uintptr_t)&aperture;
    aperture_query.num_of_nodes = 1U;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_GET_PROCESS_APERTURES_NEW,
                           &aperture_query) == 0 &&
                           aperture_query.num_of_nodes == 1U &&
                           aperture.gpu_id == 38144U &&
                           aperture.gpuvm_base == UINT64_C(0x10000) &&
                           aperture.gpuvm_limit ==
                               UINT64_C(0x00007fffffffffff),
                       "official HSAKMT aperture record is canonical");
    open_render.minor = 128U;
    open_render.fd_out = &render_fd;
    failures += expect(functions->handle_drm_call(
                           HSAKMT_DRM_OPEN_RENDER, &open_render) == 0 &&
                           render_fd >= 0,
                       "official model opens a private render identity");
    memset(&acquire_vm, 0, sizeof(acquire_vm));
    acquire_vm.drm_fd = (uint32_t)render_fd;
    acquire_vm.gpu_id = 38144U;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_ACQUIRE_VM, &acquire_vm) == 0,
                       "official HSAKMT VM acquisition crosses typed bridge");
    memset(&first_allocation, 0, sizeof(first_allocation));
    first_allocation.va_addr = UINT64_C(0x100000000);
    first_allocation.size = UINT64_C(0x200000);
    first_allocation.gpu_id = 38144U;
    first_allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                             KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
                           &first_allocation) == 0 &&
                           first_allocation.handle != 0U &&
                           first_allocation.mmap_offset == 0U,
                       "facade assigns the first file-backed interval at zero");
    mapped_backing = mmap(NULL, (size_t)first_allocation.size,
                          PROT_READ | PROT_WRITE, MAP_SHARED, render_fd,
                          (off_t)first_allocation.mmap_offset);
    if (mapped_backing != MAP_FAILED) {
      ((volatile unsigned char *)mapped_backing)[0] = 0x3cU;
      ((volatile unsigned char *)mapped_backing)[first_allocation.size - 1U] =
          0xc3U;
    }
    failures += expect(mapped_backing != MAP_FAILED &&
                           ((volatile unsigned char *)mapped_backing)[0] == 0x3cU &&
                           ((volatile unsigned char *)mapped_backing)
                               [first_allocation.size - 1U] == 0xc3U,
                       "facade backing offset maps writable pages inside its memfd");
    if (mapped_backing != MAP_FAILED) {
      failures += expect(munmap(mapped_backing,
                                (size_t)first_allocation.size) == 0,
                         "facade backing mapping unmaps cleanly");
      mapped_backing = MAP_FAILED;
    }
    memset(&second_allocation, 0, sizeof(second_allocation));
    second_allocation.va_addr = UINT64_C(0x100200000);
    second_allocation.size = UINT64_C(0x3000);
    second_allocation.gpu_id = 38144U;
    second_allocation.flags = first_allocation.flags;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
                           &second_allocation) == 0 &&
                           second_allocation.handle != 0U &&
                           second_allocation.mmap_offset == UINT64_C(0x200000),
                       "facade assigns adjacent page-aligned intervals");
    memset(&free_allocation, 0, sizeof(free_allocation));
    free_allocation.handle = first_allocation.handle;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_FREE_MEMORY_OF_GPU,
                           &free_allocation) == 0,
                       "facade releases the first backing interval");
    memset(&reused_allocation, 0, sizeof(reused_allocation));
    reused_allocation.va_addr = UINT64_C(0x100400000);
    reused_allocation.size = UINT64_C(0x1000);
    reused_allocation.gpu_id = 38144U;
    reused_allocation.flags = first_allocation.flags;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
                           &reused_allocation) == 0 &&
                           reused_allocation.mmap_offset == 0U,
                       "facade reuses a released interval without overlap");
    memset(&userptr_allocation, 0, sizeof(userptr_allocation));
    userptr_allocation.va_addr = UINT64_C(0x100500000);
    userptr_allocation.size = UINT64_C(0x1000);
    userptr_allocation.gpu_id = 38144U;
    userptr_allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR |
                               KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
    userptr_allocation.mmap_offset = UINT64_C(0x70000000);
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
                           &userptr_allocation) == 0 &&
                           userptr_allocation.mmap_offset == UINT64_C(0x70000000),
                       "facade preserves the upstream USERPTR address token");
    free_allocation.handle = second_allocation.handle;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_FREE_MEMORY_OF_GPU,
                           &free_allocation) == 0,
                       "facade releases the adjacent interval");
    free_allocation.handle = reused_allocation.handle;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_FREE_MEMORY_OF_GPU,
                           &free_allocation) == 0,
                       "facade releases the reused interval");
    free_allocation.handle = userptr_allocation.handle;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_FREE_MEMORY_OF_GPU,
                           &free_allocation) == 0,
                       "facade releases USERPTR metadata without a backing interval");
    memset(&capacity_first, 0, sizeof(capacity_first));
    capacity_first.va_addr = UINT64_C(0x10000000000);
    capacity_first.size = TEST_LARGE_GPU_ALLOCATION_BYTES;
    capacity_first.gpu_id = 38144U;
    capacity_first.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                           KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
                           &capacity_first) == 0 &&
                           capacity_first.mmap_offset == 0U,
                       "facade admits a VRAM interval larger than four GiB");
    memset(&capacity_second, 0, sizeof(capacity_second));
    capacity_second.va_addr = capacity_first.va_addr + capacity_first.size;
    capacity_second.size =
        TEST_MODEL_LOCAL_MEMORY_BYTES - capacity_first.size;
    capacity_second.gpu_id = 38144U;
    capacity_second.flags = capacity_first.flags;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
                           &capacity_second) == 0 &&
                           capacity_second.mmap_offset ==
                               TEST_LARGE_GPU_ALLOCATION_BYTES,
                       "facade fills the ordinary region below the doorbell tail");
    memset(&capacity_doorbell, 0, sizeof(capacity_doorbell));
    capacity_doorbell.va_addr =
        capacity_second.va_addr + capacity_second.size;
    capacity_doorbell.size =
        SAGR_BRIDGE_KMT_SHARED_BACKING_DOORBELL_REGION_BYTES;
    capacity_doorbell.gpu_id = 38144U;
    capacity_doorbell.flags = KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL |
                              KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                              KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
                           &capacity_doorbell) == 0 &&
                           capacity_doorbell.mmap_offset ==
                               TEST_MODEL_LOCAL_MEMORY_BYTES,
                       "facade reserves the complete shared-backing tail for doorbells");
    memset(&capacity_overflow, 0, sizeof(capacity_overflow));
    capacity_overflow.va_addr =
        capacity_doorbell.va_addr + capacity_doorbell.size;
    capacity_overflow.size = UINT64_C(0x1000);
    capacity_overflow.gpu_id = 38144U;
    capacity_overflow.flags = first_allocation.flags;
    errno = 0;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
                           &capacity_overflow) == -1 && errno == ENOMEM &&
                           capacity_overflow.handle == 0U &&
                           capacity_overflow.mmap_offset == 0U,
                       "facade rejects backing capacity overflow atomically");
    free_allocation.handle = capacity_first.handle;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_FREE_MEMORY_OF_GPU,
                           &free_allocation) == 0,
                       "facade releases the first capacity interval");
    free_allocation.handle = capacity_second.handle;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_FREE_MEMORY_OF_GPU,
                           &free_allocation) == 0,
                       "facade releases the second capacity interval");
    free_allocation.handle = capacity_doorbell.handle;
    failures += expect(functions->handle_ioctl(
                           AMDKFD_IOC_FREE_MEMORY_OF_GPU,
                           &free_allocation) == 0,
                       "facade releases the reserved doorbell interval");
    close_render.fd = render_fd;
    failures += expect(functions->handle_drm_call(
                           HSAKMT_DRM_CLOSE, &close_render) == 0,
                       "official model render identity closes cleanly");
  } else {
    failures += expect(0, "HSAKMT model function table is available");
  }
  if (library != NULL) {
    failures += expect(dlclose(library) == 0,
                       "HSAKMT model closes its managed provider");
  }
  if (saved_endpoint != NULL) {
    failures += expect(setenv("SAGR_GENERIC_BRIDGE_ENDPOINT", saved_endpoint,
                              1) == 0,
                       "managed endpoint environment is restored");
  } else {
    failures += expect(unsetenv("SAGR_GENERIC_BRIDGE_ENDPOINT") == 0,
                       "managed endpoint environment is cleared");
  }
  free(saved_endpoint);
  failures += expect(stop_process_server(&process_server, &server) == 0,
                     "HSAKMT model server observed clean disconnect");
  failures += expect(server.first_kmt_sequence == 1U &&
                         server.kmt_request_count == 22U &&
                         server.backing_export_count == 1U &&
                         server.backing_bytes == TEST_MODEL_BACKING_BYTES &&
                         server.backing_page_bytes == 4096U,
                     "HSAKMT model owns discovery, VM, backing, and close sequence");
  return failures;
}
#endif

#if defined(SAGR_HSAKMT_MODEL_PATH) && defined(SAGR_UPSTREAM_HSAKMT_TEST)
static int restore_environment(const char *name, char *saved_value) {
  const int result = saved_value != NULL ? setenv(name, saved_value, 1)
                                         : unsetenv(name);
  free(saved_value);
  return result;
}

static char *saved_environment(const char *name) {
  const char *value = getenv(name);
  return value != NULL ? strdup(value) : NULL;
}

static int check_upstream_hsakmt_model_open(void) {
  mock_server_t server;
  char *const arguments[] = {(char *)SAGR_UPSTREAM_HSAKMT_TEST_PATH, NULL};
  char *saved_endpoint = saved_environment("SAGR_GENERIC_BRIDGE_ENDPOINT");
  char *saved_model_library = saved_environment("HSA_MODEL_LIB");
  char *saved_topology = saved_environment("HSA_MODEL_TOPOLOGY");
  char *saved_dxg_detection = saved_environment("HSA_ENABLE_DXG_DETECTION");
  char *saved_interrupt = saved_environment("HSA_ENABLE_INTERRUPT");
  pid_t child = -1;
  int child_status = 0;
  int failures = 0;
  if ((getenv("SAGR_GENERIC_BRIDGE_ENDPOINT") != NULL &&
       saved_endpoint == NULL) ||
      (getenv("HSA_MODEL_LIB") != NULL && saved_model_library == NULL) ||
      (getenv("HSA_MODEL_TOPOLOGY") != NULL && saved_topology == NULL) ||
      (getenv("HSA_ENABLE_DXG_DETECTION") != NULL &&
       saved_dxg_detection == NULL) ||
      (getenv("HSA_ENABLE_INTERRUPT") != NULL && saved_interrupt == NULL)) {
    free(saved_endpoint);
    free(saved_model_library);
    free(saved_topology);
    free(saved_dxg_detection);
    free(saved_interrupt);
    return expect(0, "could not preserve upstream Model environment");
  }
  if (start_server(&server, 2) != 0) {
    free(saved_endpoint);
    free(saved_model_library);
    free(saved_topology);
    free(saved_dxg_detection);
    free(saved_interrupt);
    return expect(0, "could not start upstream HSAKMT model server");
  }
  failures += expect(
      setenv("SAGR_GENERIC_BRIDGE_ENDPOINT", server.endpoint, 1) == 0 &&
          setenv("HSA_MODEL_LIB", SAGR_HSAKMT_MODEL_PATH, 1) == 0 &&
          setenv("HSA_MODEL_TOPOLOGY", SAGR_HSAKMT_TOPOLOGY_PATH, 1) == 0 &&
          setenv("HSA_ENABLE_DXG_DETECTION", "0", 1) == 0 &&
          setenv("HSA_ENABLE_INTERRUPT", "0", 1) == 0,
      "upstream HSAKMT Model inputs are explicit");

  failures += expect(
      posix_spawn(&child, SAGR_UPSTREAM_HSAKMT_TEST_PATH, NULL, NULL,
                  arguments, environ) == 0 &&
          waitpid(child, &child_status, 0) == child &&
          WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
      "unchanged upstream libhsakmt opens the Model and discovers gfx950");

  failures += expect(restore_environment("SAGR_GENERIC_BRIDGE_ENDPOINT",
                                         saved_endpoint) == 0 &&
                         restore_environment("HSA_MODEL_LIB",
                                             saved_model_library) == 0 &&
                         restore_environment("HSA_MODEL_TOPOLOGY",
                                             saved_topology) == 0 &&
                         restore_environment("HSA_ENABLE_DXG_DETECTION",
                                             saved_dxg_detection) == 0 &&
                         restore_environment("HSA_ENABLE_INTERRUPT",
                                             saved_interrupt) == 0,
                     "upstream Model environment is restored");
  failures += expect(stop_server(&server) == 0,
                     "upstream HSAKMT server and descriptors are clean");
  if (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
    uint32_t index;
    fprintf(stderr, "upstream HSAKMT exit=%d operations=",
            WIFEXITED(child_status) ? WEXITSTATUS(child_status) : -1);
    for (index = 0U; index < server.kmt_request_count; ++index) {
      fprintf(stderr, "%s%u", index == 0U ? "" : ",",
              (unsigned)server.kmt_operations[index]);
    }
    fputc('\n', stderr);
  }
  failures += expect(server.first_kmt_sequence == 1U &&
                         server.kmt_request_count >= 3U,
                     "upstream process owns a monotonic KMT initialization sequence");
  if (server.kmt_request_count > 0U) {
    uint32_t index;
    fprintf(stderr, "upstream observed %u KMT operations:",
            (unsigned)server.kmt_request_count);
    for (index = 0U; index < server.kmt_request_count; ++index)
      fprintf(stderr, " %u", (unsigned)server.kmt_operations[index]);
    fputc('\n', stderr);
  }
  return failures;
}
#endif

#if defined(SAGR_HSAKMT_MODEL_PATH) && defined(SAGR_UPSTREAM_ROCR_TEST)
static int check_upstream_rocr_model_init(void) {
  mock_server_t server;
  char *const arguments[] = {(char *)SAGR_UPSTREAM_ROCR_TEST_PATH, NULL};
  char *saved_endpoint = saved_environment("SAGR_GENERIC_BRIDGE_ENDPOINT");
  char *saved_model_library = saved_environment("HSA_MODEL_LIB");
  char *saved_topology = saved_environment("HSA_MODEL_TOPOLOGY");
  char *saved_dxg_detection = saved_environment("HSA_ENABLE_DXG_DETECTION");
  char *saved_interrupt = saved_environment("HSA_ENABLE_INTERRUPT");
  pid_t child = -1;
  int child_status = 0;
  int failures = 0;
  uint32_t queue_create_count = 0U;
  uint32_t queue_doorbell_count = 0U;
  uint32_t queue_destroy_count = 0U;
  if ((getenv("SAGR_GENERIC_BRIDGE_ENDPOINT") != NULL &&
       saved_endpoint == NULL) ||
      (getenv("HSA_MODEL_LIB") != NULL && saved_model_library == NULL) ||
      (getenv("HSA_MODEL_TOPOLOGY") != NULL && saved_topology == NULL) ||
      (getenv("HSA_ENABLE_DXG_DETECTION") != NULL &&
       saved_dxg_detection == NULL) ||
      (getenv("HSA_ENABLE_INTERRUPT") != NULL && saved_interrupt == NULL)) {
    free(saved_endpoint);
    free(saved_model_library);
    free(saved_topology);
    free(saved_dxg_detection);
    free(saved_interrupt);
    return expect(0, "could not preserve upstream ROCr environment");
  }
  if (start_server(&server, 2) != 0) {
    free(saved_endpoint);
    free(saved_model_library);
    free(saved_topology);
    free(saved_dxg_detection);
    free(saved_interrupt);
    return expect(0, "could not start upstream ROCr model server");
  }
  failures += expect(
      setenv("SAGR_GENERIC_BRIDGE_ENDPOINT", server.endpoint, 1) == 0 &&
          setenv("HSA_MODEL_LIB", SAGR_HSAKMT_MODEL_PATH, 1) == 0 &&
          setenv("HSA_MODEL_TOPOLOGY", SAGR_HSAKMT_TOPOLOGY_PATH, 1) == 0 &&
          setenv("HSA_ENABLE_DXG_DETECTION", "0", 1) == 0 &&
          setenv("HSA_ENABLE_INTERRUPT", "0", 1) == 0,
      "upstream ROCr model inputs are explicit");
  failures += expect(
      posix_spawn(&child, SAGR_UPSTREAM_ROCR_TEST_PATH, NULL, NULL, arguments,
                  environ) == 0 &&
          waitpid(child, &child_status, 0) == child &&
          WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
      "unchanged upstream ROCr initializes and enumerates the model");
  failures += expect(restore_environment("SAGR_GENERIC_BRIDGE_ENDPOINT",
                                         saved_endpoint) == 0 &&
                         restore_environment("HSA_MODEL_LIB",
                                             saved_model_library) == 0 &&
                         restore_environment("HSA_MODEL_TOPOLOGY",
                                             saved_topology) == 0 &&
                         restore_environment("HSA_ENABLE_DXG_DETECTION",
                                             saved_dxg_detection) == 0 &&
                         restore_environment("HSA_ENABLE_INTERRUPT",
                                             saved_interrupt) == 0,
                     "upstream ROCr environment is restored");
  failures += expect(stop_server(&server) == 0,
                     "upstream ROCr server and descriptors are clean");
  {
    uint32_t index;
    for (index = 0U; index < server.kmt_request_count; ++index) {
      if (server.kmt_operations[index] == SAGR_KMT_OP_QUEUE_CREATE) {
        ++queue_create_count;
      } else if (server.kmt_operations[index] ==
                 SAGR_KMT_OP_QUEUE_DOORBELL) {
        ++queue_doorbell_count;
      } else if (server.kmt_operations[index] ==
                 SAGR_KMT_OP_QUEUE_DESTROY) {
        ++queue_destroy_count;
      }
    }
  }
  failures += expect(queue_create_count >= 1U &&
                         queue_doorbell_count >= 1U &&
                         queue_destroy_count >= 1U,
                     "unchanged upstream ROCr doorbells cross the KMT bridge");
  if (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
    uint32_t index;
    fprintf(stderr, "upstream ROCr exit=%d signal=%d operations=",
            WIFEXITED(child_status) ? WEXITSTATUS(child_status) : -1,
            WIFSIGNALED(child_status) ? WTERMSIG(child_status) : 0);
    for (index = 0U; index < server.kmt_request_count; ++index) {
      fprintf(stderr, "%s%u", index == 0U ? "" : ",",
              (unsigned)server.kmt_operations[index]);
    }
    fputc('\n', stderr);
  }
  return failures;
}
#endif

int main(void) {
  int failures = 0;
  failures += check_carrier_layout_and_codec();
  failures += check_missing_capability_wrappers();
  failures += check_negotiated_kmt_wrappers();
  failures += check_managed_provider_ownership();
#ifdef SAGR_HSAKMT_MODEL_PATH
  failures += check_hsakmt_model_get_version();
#endif
#if defined(SAGR_HSAKMT_MODEL_PATH) && defined(SAGR_UPSTREAM_HSAKMT_TEST)
  failures += check_upstream_hsakmt_model_open();
#endif
#if defined(SAGR_HSAKMT_MODEL_PATH) && defined(SAGR_UPSTREAM_ROCR_TEST)
  failures += check_upstream_rocr_model_init();
#endif
  if (failures != 0) {
    fprintf(stderr, "KMT shim failures: %d\n", failures);
    return 1;
  }
  puts("KMT shim tests passed");
  return 0;
}
