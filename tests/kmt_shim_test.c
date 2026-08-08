/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <errno.h>
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

#include <self_amdgpu_runtime/kmt_shim.h>
#include <self_amdgpu_runtime/provider.h>

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

typedef struct mock_server {
  char directory[128];
  char endpoint[160];
  int listener;
  int thread_error;
  int serve_kmt;
  uint64_t first_kmt_sequence;
  uint32_t kmt_request_count;
  pthread_t thread;
} mock_server_t;

static int expect(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "KMT shim test: %s\n", message);
    return 1;
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

static int serve_one_kmt(int peer, const sagr_instance_info_t *unused_info,
                         mock_server_t *server) {
  uint8_t frame[SAGR_WIRE_KMT_FRAME_BYTES];
  uint8_t response[SAGR_WIRE_KMT_FRAME_BYTES];
  size_t response_size = 0;
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  ssize_t count;
  (void)unused_info;
  count = recv(peer, frame, sizeof(frame), 0);
  if (count != (ssize_t)sizeof(frame) || get_be_u16(frame + 14) !=
                                             SAGR_KMT_MESSAGE_REQUEST) {
    return -1;
  }
  decode_request(frame, &request);
  if (server->kmt_request_count == 0U) {
    server->first_kmt_sequence = request.operation_sequence;
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
    result.result_words[1] = 0;
    result.result_words[2] = 0;
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
  } else if (request.operation == SAGR_KMT_OP_ALLOC_MEMORY) {
    result.object_id = UINT64_C(0x4d454d0000000001);
    result.object_generation = UINT64_C(0x0102030405060708);
    result.result_words[1] = request.argument_words[1];
    result.result_words[2] = request.argument_words[2];
    result.result_words[3] = request.argument_words[3];
    result.result_words[4] = request.argument_words[4];
    result.result_words[5] = 0;
    result.result_words[6] = 0x1000;
  } else if (request.operation == SAGR_KMT_OP_COPY_MEMORY &&
             request.argument_words[0] == SAGR_KMT_COPY_SIM_TO_HOST) {
    static const uint8_t k_copy_result[] = {'s', 'i', 'm', 'o', 'k'};
    result.buffer_bytes = (uint32_t)sizeof(k_copy_result);
    memcpy(result.buffer, k_copy_result, sizeof(k_copy_result));
    result.buffer_crc32c = sagr_crc32c(result.buffer, result.buffer_bytes);
  } else if (request.operation == SAGR_KMT_OP_QUEUE_CREATE) {
    result.object_id = UINT64_C(0x5155450000000001);
    result.object_generation = UINT64_C(0x0102030405060708);
  } else if (request.operation == SAGR_KMT_OP_QUEUE_DOORBELL) {
    result.result_words[1] = 1;
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
  if (peer < 0) {
    server->thread_error = errno;
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

static int start_server(mock_server_t *server, int serve_kmt) {
  struct sockaddr_un address;
  size_t endpoint_size;
  memset(server, 0, sizeof(*server));
  server->listener = -1;
  server->serve_kmt = serve_kmt;
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
  if (pthread_create(&server->thread, NULL, mock_server_main, server) != 0) {
    return -1;
  }
  return 0;
}

static int stop_server(mock_server_t *server) {
  int failure = 0;
  if (pthread_join(server->thread, NULL) != 0) {
    failure = 1;
  }
  if (server->listener >= 0) {
    (void)close(server->listener);
  }
  if (server->endpoint[0] != '\0') {
    (void)unlink(server->endpoint);
  }
  if (server->directory[0] != '\0') {
    (void)rmdir(server->directory);
  }
  return failure != 0 || server->thread_error != 0 ? -1 : 0;
}

static int check_carrier_layout_and_codec(void) {
  sagr_kmt_envelope_request_t request;
  sagr_kmt_envelope_result_t result;
  sagr_instance_info_t info;
  uint8_t frame[SAGR_WIRE_KMT_FRAME_BYTES];
  size_t frame_size = 0;
  int failures = 0;
  _Static_assert(sizeof(sagr_kmt_handle_t) == 32,
                 "KMT handle must contain four u64 identities");
  _Static_assert(sizeof(sagr_kmt_envelope_request_t) == 256,
                 "KMT request payload size");
  _Static_assert(sizeof(sagr_kmt_envelope_result_t) == 256,
                 "KMT result payload size");
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
  queue.depth = 1;
  queue.ring_size_bytes = 4096;
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
  sagr_kmt_handle_t queue_handle = {0, 0, 0, 0};
  sagr_kmt_handle_t event_handle = {0, 0, 0, 0};
  sagr_kmt_version_t version;
  sagr_kmt_topology_t topology;
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
  failures += expect(sagr_kmt_get_version(
                         provider, &kfd, &version, (uint32_t)sizeof(version),
                         &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS &&
                         version.major == 1 && version.minor == 0,
                     "typed version operation");
  failures += expect(sagr_kmt_topology_snapshot(
                         provider, &kfd, &topology,
                         (uint32_t)sizeof(topology), &call_options, &error,
                         (uint32_t)sizeof(error)) == SAGR_KMT_STATUS_SUCCESS &&
                         topology.gfx_target_code == 950 &&
                         topology.wavefront_size == 64 &&
                         topology.maximum_allocations == 1024,
                     "typed topology record operation");
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
  queue.depth = 1;
  queue.ring_size_bytes = 4096;
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
  failures += expect(stop_server(&server) == 0,
                     "KMT server completed every operation");

  /* Operation sequences are scoped to the provider/session, not process
   * global.  A fresh provider must begin at sequence one. */
  if (start_server(&second_server, 1) != 0) {
    failures += expect(0, "could not start second KMT server");
  } else {
    failures += expect(sagr_provider_open(
                           second_server.endpoint, &open_options,
                           &second_provider, &error,
                           (uint32_t)sizeof(error)) == SAGR_STATUS_SUCCESS,
                       "second provider opens with KMT capability");
    if (second_provider != NULL) {
      sagr_kmt_handle_t second_kfd = {0, 0, 0, 0};
      failures += expect(sagr_kmt_open_kfd(
                             second_provider, &second_kfd, &call_options,
                             &error, (uint32_t)sizeof(error)) ==
                             SAGR_KMT_STATUS_SUCCESS,
                         "second provider opens KFD");
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

int main(void) {
  int failures = 0;
  failures += check_carrier_layout_and_codec();
  failures += check_missing_capability_wrappers();
  failures += check_negotiated_kmt_wrappers();
  if (failures != 0) {
    fprintf(stderr, "KMT shim failures: %d\n", failures);
    return 1;
  }
  puts("KMT shim tests passed");
  return 0;
}
