/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
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

enum server_behavior {
  SERVER_SUCCESS,
  SERVER_SUCCESS_ABSOLUTE_DEADLINE,
  SERVER_SUCCESS_WIDE_RANGE,
  SERVER_SUCCESS_EXTRA_OFFER,
  SERVER_VERSION_REJECTION,
  SERVER_REQUIRED_CAPABILITY_REJECTION,
  SERVER_WIRE_FAILURE,
  SERVER_TIMEOUT,
  SERVER_EOF,
  SERVER_TRUNCATED_RECORD,
  SERVER_TRUNCATED_CONTROL,
  SERVER_ZERO_LENGTH_CONTROL,
  SERVER_BAD_CRC,
  SERVER_WRONG_REQUEST,
  SERVER_WRONG_NONCE,
  SERVER_WRONG_DAEMON,
  SERVER_WRONG_EPOCH,
  SERVER_WRONG_TOPOLOGY,
  SERVER_WRONG_CAPABILITIES,
  SERVER_EXTRA_SELECTED_CAPABILITY,
  SERVER_ZERO_CONNECTION,
  SERVER_ZERO_SERVER_NONCE,
  SERVER_ZERO_EPOCH,
  SERVER_BAD_MAXIMUM_RECORD,
  SERVER_BAD_FLAGS,
  SERVER_BAD_RESERVED,
  SERVER_BAD_PAYLOAD_RESERVED,
  SERVER_BAD_TYPE
};

typedef struct mock_server {
  char directory[128];
  char endpoint[160];
  int listener;
  pthread_t thread;
  enum server_behavior behavior;
  uint32_t wire_status;
  int thread_error;
} mock_server_t;

typedef struct cancellation_sender {
  int descriptor;
  int error;
} cancellation_sender_t;

static uint64_t get_u64(const uint8_t *source) {
  uint64_t result = 0;
  uint32_t index;
  for (index = 0; index < 8; ++index) {
    result = (result << 8) | source[index];
  }
  return result;
}

static void initialize_options(sagr_instance_open_options_t *options) {
  (void)sagr_instance_open_options_init(options, (uint32_t)sizeof(*options));
  memcpy(options->expected_daemon_uuid, k_daemon_uuid, 16);
  memcpy(options->expected_job_uuid, k_job_uuid, 16);
  options->expected_epoch = UINT64_C(0x0102030405060708);
  options->expected_rank = 3;
  options->expected_world_size = 8;
  options->open_timeout_ns = UINT64_C(1000000000);
}

static int send_control_record(int peer, const uint8_t *frame,
                               size_t frame_size) {
  struct iovec vector;
  struct msghdr message;
  struct cmsghdr *control_message;
  unsigned char control[CMSG_SPACE(sizeof(int) * 16U)];
  int descriptors[16];
  uint32_t index;
  memset(&message, 0, sizeof(message));
  memset(control, 0, sizeof(control));
  for (index = 0; index < 16; ++index) {
    descriptors[index] = peer;
  }
  vector.iov_base = (void *)frame;
  vector.iov_len = frame_size;
  message.msg_iov = &vector;
  message.msg_iovlen = 1;
  message.msg_control = control;
  message.msg_controllen = sizeof(control);
  control_message = CMSG_FIRSTHDR(&message);
  if (control_message == NULL) {
    return -1;
  }
  control_message->cmsg_level = SOL_SOCKET;
  control_message->cmsg_type = SCM_RIGHTS;
  control_message->cmsg_len = CMSG_LEN(sizeof(descriptors));
  memcpy(CMSG_DATA(control_message), descriptors, sizeof(descriptors));
  return sendmsg(peer, &message, MSG_NOSIGNAL) == (ssize_t)frame_size ? 0 : -1;
}

static void *mock_server_main(void *argument) {
  mock_server_t *server = (mock_server_t *)argument;
  uint8_t hello[SAGR_WIRE_MAX_HANDSHAKE_BYTES];
  uint8_t ack[SAGR_WIRE_ACK_FRAME_BYTES];
  sagr_wire_ack_fields_t fields;
  size_t ack_size = 0;
  ssize_t hello_size;
  int peer = accept4(server->listener, NULL, NULL, SOCK_CLOEXEC);
  if (peer < 0) {
    server->thread_error = errno;
    return NULL;
  }
  hello_size = recv(peer, hello, sizeof(hello), 0);
  if (hello_size < SAGR_WIRE_HELLO_FIXED_BYTES + SAGR_WIRE_HEADER_BYTES) {
    server->thread_error = EPROTO;
    (void)close(peer);
    return NULL;
  }
  if (server->behavior == SERVER_TIMEOUT) {
    const struct timespec delay = {.tv_sec = 0, .tv_nsec = 100000000L};
    (void)nanosleep(&delay, NULL);
    (void)close(peer);
    return NULL;
  }
  if (server->behavior == SERVER_EOF) {
    (void)close(peer);
    return NULL;
  }
  if (server->behavior == SERVER_TRUNCATED_RECORD) {
    uint8_t oversized[SAGR_WIRE_MAX_HANDSHAKE_BYTES + 1U];
    memset(oversized, 0xa5, sizeof(oversized));
    if (send(peer, oversized, sizeof(oversized), MSG_NOSIGNAL) !=
        (ssize_t)sizeof(oversized)) {
      server->thread_error = errno == 0 ? EIO : errno;
    }
    (void)close(peer);
    return NULL;
  }

  memset(&fields, 0, sizeof(fields));
  fields.selected_major = 1;
  fields.status = SAGR_WIRE_STATUS_OK;
  memcpy(fields.client_nonce, hello + SAGR_WIRE_HEADER_BYTES + 8, 16);
  memcpy(fields.server_nonce, k_server_nonce, 16);
  fields.selected_capabilities[0] = 1;
  fields.maximum_record_bytes = SAGR_WIRE_MAX_RECORD_BYTES;
  fields.request_id = get_u64(hello + 24);
  memcpy(fields.daemon_uuid, k_daemon_uuid, 16);
  fields.connection_id = UINT64_C(0x1122334455667788);
  fields.epoch = UINT64_C(0x0102030405060708);
  memcpy(fields.job_uuid, k_job_uuid, 16);
  fields.rank = 3;
  fields.world_size = 8;
  fields.include_topology = 1;
  if (server->behavior == SERVER_WIRE_FAILURE ||
      server->behavior == SERVER_VERSION_REJECTION ||
      server->behavior == SERVER_REQUIRED_CAPABILITY_REJECTION) {
    fields.selected_major = 0;
    fields.status =
        server->behavior == SERVER_VERSION_REJECTION
            ? SAGR_WIRE_STATUS_UNSUPPORTED_VERSION
            : (server->behavior == SERVER_REQUIRED_CAPABILITY_REJECTION
                   ? SAGR_WIRE_STATUS_UNSUPPORTED_CAPABILITY
                   : server->wire_status);
    memset(fields.server_nonce, 0, 16);
    memset(fields.selected_capabilities, 0,
           sizeof(fields.selected_capabilities));
    fields.connection_id = 0;
    fields.include_topology = 0;
    if (server->wire_status == SAGR_WIRE_STATUS_INSTANCE_MISMATCH) {
      fields.daemon_uuid[0] ^= 1;
    }
    if (server->wire_status == SAGR_WIRE_STATUS_TOPOLOGY_MISMATCH) {
      ++fields.epoch;
    }
  }
  if (sagr_protocol_encode_ack(&fields, ack, sizeof(ack), &ack_size) !=
      SAGR_STATUS_SUCCESS) {
    server->thread_error = EPROTO;
    (void)close(peer);
    return NULL;
  }

  switch (server->behavior) {
    case SERVER_BAD_CRC:
      ack[ack_size - 1U] ^= 1;
      break;
    case SERVER_WRONG_REQUEST:
      ack[31] ^= 1;
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    case SERVER_WRONG_NONCE:
      ack[SAGR_WIRE_HEADER_BYTES + 8] ^= 1;
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    case SERVER_WRONG_DAEMON:
      ack[32] ^= 1;
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    case SERVER_WRONG_EPOCH:
      ack[63] ^= 1;
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    case SERVER_WRONG_TOPOLOGY:
      ack[SAGR_WIRE_HEADER_BYTES + SAGR_WIRE_ACK_FIXED_BYTES + 8 + 16 + 3] =
          4;
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    case SERVER_WRONG_CAPABILITIES:
      ack[SAGR_WIRE_HEADER_BYTES + 40] = 0;
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    case SERVER_EXTRA_SELECTED_CAPABILITY:
      ack[SAGR_WIRE_HEADER_BYTES + 40] = 3;
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    case SERVER_ZERO_CONNECTION:
      memset(ack + 48, 0, 8);
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    case SERVER_ZERO_SERVER_NONCE:
      memset(ack + SAGR_WIRE_HEADER_BYTES + 24, 0, 16);
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    case SERVER_ZERO_EPOCH:
      memset(ack + 56, 0, 8);
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    case SERVER_BAD_MAXIMUM_RECORD:
      ack[SAGR_WIRE_HEADER_BYTES + 72] = 0;
      ack[SAGR_WIRE_HEADER_BYTES + 73] = 0;
      ack[SAGR_WIRE_HEADER_BYTES + 74] = 0;
      ack[SAGR_WIRE_HEADER_BYTES + 75] = 1;
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    case SERVER_BAD_FLAGS:
      ack[19] = 1;
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    case SERVER_BAD_RESERVED:
      ack[68] = 1;
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    case SERVER_BAD_PAYLOAD_RESERVED:
      ack[SAGR_WIRE_HEADER_BYTES + 79] = 1;
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    case SERVER_BAD_TYPE:
      ack[15] = 1;
      sagr_protocol_recompute_frame_crc(ack, ack_size);
      break;
    default:
      break;
  }

  if (server->behavior == SERVER_TRUNCATED_CONTROL ||
      server->behavior == SERVER_ZERO_LENGTH_CONTROL) {
    const size_t payload_size =
        server->behavior == SERVER_ZERO_LENGTH_CONTROL ? 0 : ack_size;
    if (send_control_record(peer, ack, payload_size) != 0) {
      server->thread_error = errno == 0 ? EIO : errno;
    }
  } else if (send(peer, ack, ack_size, MSG_NOSIGNAL) != (ssize_t)ack_size) {
    server->thread_error = errno == 0 ? EIO : errno;
  }
  (void)close(peer);
  return NULL;
}

static int start_server(mock_server_t *server, enum server_behavior behavior,
                        uint32_t wire_status) {
  struct sockaddr_un address;
  size_t endpoint_size;
  memset(server, 0, sizeof(*server));
  server->listener = -1;
  server->behavior = behavior;
  server->wire_status = wire_status;
  (void)snprintf(server->directory, sizeof(server->directory),
                 "/tmp/sagr-runtime-test-XXXXXX");
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
      pthread_create(&server->thread, NULL, mock_server_main, server) != 0) {
    (void)close(server->listener);
    (void)unlink(server->endpoint);
    (void)rmdir(server->directory);
    return -1;
  }
  return 0;
}

static int finish_server(mock_server_t *server) {
  int failure = 0;
  if (pthread_join(server->thread, NULL) != 0 || server->thread_error != 0) {
    fprintf(stderr, "mock server failed: %d\n", server->thread_error);
    failure = 1;
  }
  (void)close(server->listener);
  if (unlink(server->endpoint) != 0 || rmdir(server->directory) != 0) {
    fprintf(stderr, "mock server cleanup failed\n");
    failure = 1;
  }
  return failure;
}

static int run_scenario(enum server_behavior behavior, uint32_t wire_status,
                        sagr_status_t expected_status,
                        int32_t expected_wire_status) {
  mock_server_t server;
  sagr_instance_open_options_t options;
  sagr_error_info_t error;
  sagr_instance_t instance = (sagr_instance_t)(uintptr_t)1;
  sagr_status_t status;
  int failure = 0;
  if (start_server(&server, behavior, wire_status) != 0) {
    fprintf(stderr, "could not start mock server\n");
    return 1;
  }
  initialize_options(&options);
  if (behavior == SERVER_SUCCESS_WIDE_RANGE) {
    options.minimum_version_major = 0;
    options.minimum_version_minor = 9;
    options.maximum_version_major = 1;
    options.maximum_version_minor = 1;
  }
  if (behavior == SERVER_SUCCESS_EXTRA_OFFER) {
    options.offered_capabilities[0] |= UINT64_C(2);
  }
  if (behavior == SERVER_SUCCESS_ABSOLUTE_DEADLINE) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 || now.tv_sec < 0 ||
        (uint64_t)now.tv_sec >
            (UINT64_MAX - UINT64_C(2000000000)) /
                UINT64_C(1000000000)) {
      fprintf(stderr, "could not construct absolute test deadline\n");
      failure = 1;
    } else {
      options.open_timeout_ns = 1;
      options.absolute_deadline_ns =
          (uint64_t)now.tv_sec * UINT64_C(1000000000) +
          (uint64_t)now.tv_nsec + UINT64_C(2000000000);
    }
  }
  if (behavior == SERVER_VERSION_REJECTION) {
    options.minimum_version_major = 2;
    options.maximum_version_major = 2;
  }
  if (behavior == SERVER_REQUIRED_CAPABILITY_REJECTION) {
    options.offered_capabilities[0] |= UINT64_C(2);
    options.required_capabilities[0] |= UINT64_C(2);
  }
  if (behavior == SERVER_TIMEOUT) {
    options.open_timeout_ns = UINT64_C(20000000);
  }
  status = sagr_instance_open(server.endpoint, &options, &instance, &error,
                              (uint32_t)sizeof(error));
  if (status != expected_status || error.status != expected_status ||
      error.wire_status != expected_wire_status) {
    fprintf(stderr,
            "scenario %d: status=%d error=%d wire=%d, expected=%d/%d\n",
            (int)behavior, status, error.status, error.wire_status,
            expected_status, expected_wire_status);
    failure = 1;
  }
  if (status == SAGR_STATUS_SUCCESS) {
    sagr_instance_info_t info;
    if (instance == NULL ||
        sagr_instance_get_info(instance, &info, (uint32_t)sizeof(info)) !=
            SAGR_STATUS_SUCCESS ||
        info.selected_version_major != 1 ||
        info.selected_version_minor != 0 ||
        info.negotiated_capabilities[0] != 1 ||
        info.maximum_record_bytes != SAGR_WIRE_MAX_RECORD_BYTES ||
        memcmp(info.daemon_uuid, k_daemon_uuid, 16) != 0 ||
        memcmp(info.job_uuid, k_job_uuid, 16) != 0 ||
        info.connection_id != UINT64_C(0x1122334455667788) ||
        info.epoch != UINT64_C(0x0102030405060708) || info.rank != 3 ||
        info.world_size != 8 || info.request_id == 0 ||
        info.peer_uid != (uint32_t)geteuid() ||
        info.peer_pid != (uint32_t)getpid()) {
      fprintf(stderr, "successful handshake returned incorrect info\n");
      failure = 1;
    }
    if (sagr_instance_close(&instance) != SAGR_STATUS_SUCCESS ||
        instance != NULL ||
        sagr_instance_close(&instance) != SAGR_STATUS_SUCCESS) {
      fprintf(stderr, "instance close is not local and idempotent\n");
      failure = 1;
    }
  } else if (instance != NULL) {
    fprintf(stderr, "failed open did not clear the output handle\n");
    failure = 1;
  }
  failure += finish_server(&server);
  return failure;
}

static int test_wire_failures(void) {
  static const struct {
    uint32_t wire;
    sagr_status_t status;
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
  size_t index;
  int failures = 0;
  for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
    failures += run_scenario(SERVER_WIRE_FAILURE, cases[index].wire,
                             cases[index].status, (int32_t)cases[index].wire);
  }
  return failures;
}

static int test_endpoint_validation_and_stale_preservation(void) {
  char missing[] = "/tmp/sagr-endpoint-does-not-exist";
  char long_path[200];
  char directory[] = "/tmp/sagr-stale-test-XXXXXX";
  char endpoint[160];
  struct sockaddr_un address;
  struct stat metadata;
  sagr_error_info_t error;
  sagr_instance_t instance = (sagr_instance_t)(uintptr_t)1;
  int socket_fd;
  size_t endpoint_size;
  int failures = 0;

  if (sagr_instance_open("relative", NULL, &instance, &error,
                         (uint32_t)sizeof(error)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      instance != NULL) {
    fprintf(stderr, "relative endpoint was accepted\n");
    ++failures;
  }
  memset(long_path, 'x', sizeof(long_path));
  long_path[0] = '/';
  long_path[sizeof(long_path) - 1U] = '\0';
  if (sagr_instance_open(long_path, NULL, &instance, &error,
                         (uint32_t)sizeof(error)) !=
      SAGR_STATUS_INVALID_ARGUMENT) {
    fprintf(stderr, "overlong endpoint was accepted\n");
    ++failures;
  }
  if (sagr_instance_open(missing, NULL, &instance, &error,
                         (uint32_t)sizeof(error)) !=
          SAGR_STATUS_ENDPOINT_NOT_FOUND ||
      error.native_errno != ENOENT) {
    fprintf(stderr, "missing endpoint status is incorrect\n");
    ++failures;
  }
  if (mkdtemp(directory) == NULL) {
    return failures + 1;
  }
  (void)snprintf(endpoint, sizeof(endpoint), "%s/socket", directory);
  socket_fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  memset(&address, 0, sizeof(address));
  address.sun_family = AF_UNIX;
  endpoint_size = strlen(endpoint);
  memcpy(address.sun_path, endpoint, endpoint_size + 1U);
  if (socket_fd < 0 ||
      bind(socket_fd, (const struct sockaddr *)&address,
           (socklen_t)(offsetof(struct sockaddr_un, sun_path) + endpoint_size +
                       1U)) != 0) {
    if (socket_fd >= 0) {
      (void)close(socket_fd);
    }
    (void)rmdir(directory);
    return failures + 1;
  }
  (void)close(socket_fd);
  if (sagr_instance_open(endpoint, NULL, &instance, &error,
                         (uint32_t)sizeof(error)) != SAGR_STATUS_UNAVAILABLE ||
      lstat(endpoint, &metadata) != 0 || !S_ISSOCK(metadata.st_mode)) {
    fprintf(stderr, "client altered or misclassified stale endpoint\n");
    ++failures;
  }
  (void)unlink(endpoint);
  (void)rmdir(directory);
  return failures;
}

static int test_public_validation(void) {
  sagr_instance_open_options_t options;
  sagr_error_info_t error;
  sagr_instance_t instance = (sagr_instance_t)(uintptr_t)1;
  initialize_options(&options);
  options.required_capabilities[0] = 0;
  if (sagr_instance_open("/unused", &options, &instance, &error,
                         (uint32_t)sizeof(error)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      instance != NULL) {
    fprintf(stderr, "missing required topology capability was accepted\n");
    return 1;
  }
  initialize_options(&options);
  options.required_capabilities[0] |= UINT64_C(2);
  instance = (sagr_instance_t)(uintptr_t)1;
  if (sagr_instance_open("/unused", &options, &instance, &error,
                         (uint32_t)sizeof(error)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      instance != NULL) {
    fprintf(stderr, "required capability outside offered set was accepted\n");
    return 1;
  }
  initialize_options(&options);
  options.minimum_version_major = 2;
  options.maximum_version_major = 1;
  if (sagr_instance_open("/unused", &options, &instance, &error,
                         (uint32_t)sizeof(error)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      instance != NULL) {
    fprintf(stderr, "inverted version range was accepted\n");
    return 1;
  }
  instance = (sagr_instance_t)(uintptr_t)1;
  if (sagr_instance_open("/unused", NULL, &instance, &error,
                         (uint32_t)sizeof(error) - 1U) !=
          SAGR_STATUS_BUFFER_TOO_SMALL ||
      instance != NULL) {
    fprintf(stderr, "short error buffer handling is incorrect\n");
    return 1;
  }
  return 0;
}

static int count_open_descriptors(void) {
  DIR *directory = opendir("/proc/self/fd");
  struct dirent *entry;
  int count = 0;
  if (directory == NULL) {
    return -1;
  }
  while ((entry = readdir(directory)) != NULL) {
    if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
      ++count;
    }
  }
  (void)closedir(directory);
  return count;
}

static int test_zero_length_control_does_not_leak(void) {
  const int before = count_open_descriptors();
  int failures;
  int after;
  if (before < 0) {
    fprintf(stderr, "could not count descriptors before ancillary test\n");
    return 1;
  }
  failures = run_scenario(SERVER_ZERO_LENGTH_CONTROL, 0,
                          SAGR_STATUS_PROTOCOL_ERROR, -1);
  after = count_open_descriptors();
  if (after != before) {
    fprintf(stderr, "zero-length ancillary record leaked descriptors: %d -> %d\n",
            before, after);
    ++failures;
  }
  return failures;
}

static int test_deadline_and_cancellation_inputs(void) {
  sagr_instance_open_options_t options;
  sagr_error_info_t error;
  sagr_instance_t instance = (sagr_instance_t)(uintptr_t)1;
  int cancellation_pipe[2] = {-1, -1};
  int closed_fd;
  int failures = 0;

  initialize_options(&options);
  options.absolute_deadline_ns = 1;
  if (sagr_instance_open("/unused", &options, &instance, &error,
                         (uint32_t)sizeof(error)) != SAGR_STATUS_TIMED_OUT ||
      error.status != SAGR_STATUS_TIMED_OUT || instance != NULL) {
    fprintf(stderr, "expired absolute deadline was not enforced\n");
    ++failures;
  }

  if (pipe2(cancellation_pipe, O_CLOEXEC | O_NONBLOCK) != 0 ||
      write(cancellation_pipe[1], "x", 1) != 1) {
    fprintf(stderr, "could not create cancellation fixture\n");
    if (cancellation_pipe[0] >= 0) {
      (void)close(cancellation_pipe[0]);
    }
    if (cancellation_pipe[1] >= 0) {
      (void)close(cancellation_pipe[1]);
    }
    return failures + 1;
  }
  initialize_options(&options);
  options.absolute_deadline_ns = 1;
  options.cancel_fd = cancellation_pipe[0];
  instance = (sagr_instance_t)(uintptr_t)1;
  if (sagr_instance_open("/unused", &options, &instance, &error,
                         (uint32_t)sizeof(error)) != SAGR_STATUS_TIMED_OUT ||
      error.status != SAGR_STATUS_TIMED_OUT || instance != NULL) {
    fprintf(stderr, "expired deadline did not precede ready cancellation\n");
    ++failures;
  }
  initialize_options(&options);
  options.cancel_fd = cancellation_pipe[0];
  instance = (sagr_instance_t)(uintptr_t)1;
  if (sagr_instance_open("/unused", &options, &instance, &error,
                         (uint32_t)sizeof(error)) != SAGR_STATUS_CANCELLED ||
      error.status != SAGR_STATUS_CANCELLED || instance != NULL) {
    fprintf(stderr, "ready cancellation descriptor did not cancel open\n");
    ++failures;
  }
  if (close(cancellation_pipe[0]) != 0 ||
      close(cancellation_pipe[1]) != 0) {
    fprintf(stderr, "runtime closed a caller-owned cancellation descriptor\n");
    ++failures;
  }

  if (pipe(cancellation_pipe) != 0) {
    fprintf(stderr, "could not create non-CLOEXEC cancellation fixture\n");
    return failures + 1;
  }
  initialize_options(&options);
  options.cancel_fd = cancellation_pipe[0];
  instance = (sagr_instance_t)(uintptr_t)1;
  if (sagr_instance_open("/unused", &options, &instance, &error,
                         (uint32_t)sizeof(error)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      error.native_errno != EINVAL || instance != NULL) {
    fprintf(stderr, "non-CLOEXEC cancellation descriptor was accepted\n");
    ++failures;
  }
  (void)close(cancellation_pipe[0]);
  (void)close(cancellation_pipe[1]);

  closed_fd = dup(STDIN_FILENO);
  if (closed_fd < 0) {
    fprintf(stderr, "could not create invalid cancellation descriptor\n");
    return failures + 1;
  }
  (void)close(closed_fd);
  initialize_options(&options);
  options.cancel_fd = closed_fd;
  instance = (sagr_instance_t)(uintptr_t)1;
  if (sagr_instance_open("/unused", &options, &instance, &error,
                         (uint32_t)sizeof(error)) !=
          SAGR_STATUS_INVALID_ARGUMENT ||
      error.native_errno != EBADF || instance != NULL) {
    fprintf(stderr, "closed cancellation descriptor was accepted\n");
    ++failures;
  }
  return failures;
}

static void *send_cancellation(void *argument) {
  cancellation_sender_t *sender = (cancellation_sender_t *)argument;
  const struct timespec delay = {.tv_sec = 0, .tv_nsec = 10000000L};
  if (nanosleep(&delay, NULL) != 0 ||
      write(sender->descriptor, "x", 1) != 1) {
    sender->error = errno == 0 ? EIO : errno;
  }
  return NULL;
}

static int test_blocking_cancellation_does_not_consume_fd(void) {
  mock_server_t server;
  sagr_instance_open_options_t options;
  sagr_error_info_t error;
  sagr_instance_t instance = (sagr_instance_t)(uintptr_t)1;
  cancellation_sender_t sender;
  pthread_t sender_thread;
  int cancellation_pipe[2] = {-1, -1};
  char byte = 0;
  int failure = 0;

  if (pipe2(cancellation_pipe, O_CLOEXEC | O_NONBLOCK) != 0) {
    fprintf(stderr, "could not create blocking cancellation fixture\n");
    return 1;
  }
  if (start_server(&server, SERVER_TIMEOUT, 0) != 0) {
    fprintf(stderr, "could not start blocking cancellation server\n");
    (void)close(cancellation_pipe[0]);
    (void)close(cancellation_pipe[1]);
    return 1;
  }
  sender.descriptor = cancellation_pipe[1];
  sender.error = 0;
  if (pthread_create(&sender_thread, NULL, send_cancellation, &sender) != 0) {
    fprintf(stderr, "could not start cancellation sender\n");
    (void)close(cancellation_pipe[0]);
    (void)close(cancellation_pipe[1]);
    (void)pthread_cancel(server.thread);
    (void)pthread_join(server.thread, NULL);
    (void)close(server.listener);
    (void)unlink(server.endpoint);
    (void)rmdir(server.directory);
    return 1;
  }

  initialize_options(&options);
  options.open_timeout_ns = UINT64_MAX;
  options.cancel_fd = cancellation_pipe[0];
  if (sagr_instance_open(server.endpoint, &options, &instance, &error,
                         (uint32_t)sizeof(error)) != SAGR_STATUS_CANCELLED ||
      error.status != SAGR_STATUS_CANCELLED || instance != NULL) {
    fprintf(stderr, "blocked handshake did not observe cancellation\n");
    ++failure;
  }
  if (pthread_join(sender_thread, NULL) != 0 || sender.error != 0 ||
      read(cancellation_pipe[0], &byte, 1) != 1 || byte != 'x') {
    fprintf(stderr, "runtime consumed, closed, or corrupted cancellation fd\n");
    ++failure;
  }
  (void)close(cancellation_pipe[0]);
  (void)close(cancellation_pipe[1]);
  failure += finish_server(&server);
  return failure;
}

int main(void) {
  int failures = 0;
  (void)alarm(20);
  failures += run_scenario(SERVER_SUCCESS, 0, SAGR_STATUS_SUCCESS,
                           -1);
  failures += run_scenario(SERVER_SUCCESS_ABSOLUTE_DEADLINE, 0,
                           SAGR_STATUS_SUCCESS, -1);
  failures += run_scenario(SERVER_SUCCESS_WIDE_RANGE, 0,
                           SAGR_STATUS_SUCCESS, -1);
  failures += run_scenario(SERVER_SUCCESS_EXTRA_OFFER, 0,
                           SAGR_STATUS_SUCCESS, -1);
  failures += run_scenario(SERVER_VERSION_REJECTION, 0,
                           SAGR_STATUS_VERSION_MISMATCH,
                           SAGR_WIRE_STATUS_UNSUPPORTED_VERSION);
  failures += run_scenario(SERVER_REQUIRED_CAPABILITY_REJECTION, 0,
                           SAGR_STATUS_CAPABILITY_MISMATCH,
                           SAGR_WIRE_STATUS_UNSUPPORTED_CAPABILITY);
  failures += test_wire_failures();
  failures += run_scenario(SERVER_TIMEOUT, 0, SAGR_STATUS_TIMED_OUT, -1);
  failures += run_scenario(SERVER_EOF, 0, SAGR_STATUS_CONNECTION_LOST, -1);
  failures += run_scenario(SERVER_TRUNCATED_RECORD, 0,
                           SAGR_STATUS_PROTOCOL_ERROR, -1);
  failures += run_scenario(SERVER_TRUNCATED_CONTROL, 0,
                           SAGR_STATUS_PROTOCOL_ERROR, -1);
  failures += test_zero_length_control_does_not_leak();
  failures += run_scenario(SERVER_BAD_CRC, 0, SAGR_STATUS_CHECKSUM_ERROR, -1);
  failures += run_scenario(SERVER_WRONG_REQUEST, 0,
                           SAGR_STATUS_PROTOCOL_ERROR, -1);
  failures += run_scenario(SERVER_WRONG_NONCE, 0, SAGR_STATUS_PROTOCOL_ERROR,
                           -1);
  failures += run_scenario(SERVER_WRONG_DAEMON, 0,
                           SAGR_STATUS_INSTANCE_MISMATCH,
                           SAGR_WIRE_STATUS_OK);
  failures += run_scenario(SERVER_WRONG_EPOCH, 0,
                           SAGR_STATUS_TOPOLOGY_MISMATCH,
                           SAGR_WIRE_STATUS_OK);
  failures += run_scenario(SERVER_WRONG_TOPOLOGY, 0,
                           SAGR_STATUS_TOPOLOGY_MISMATCH,
                           SAGR_WIRE_STATUS_OK);
  failures += run_scenario(SERVER_WRONG_CAPABILITIES, 0,
                           SAGR_STATUS_CAPABILITY_MISMATCH,
                           SAGR_WIRE_STATUS_OK);
  failures += run_scenario(SERVER_EXTRA_SELECTED_CAPABILITY, 0,
                           SAGR_STATUS_CAPABILITY_MISMATCH,
                           SAGR_WIRE_STATUS_OK);
  failures += run_scenario(SERVER_ZERO_CONNECTION, 0,
                           SAGR_STATUS_PROTOCOL_ERROR,
                           SAGR_WIRE_STATUS_OK);
  failures += run_scenario(SERVER_ZERO_SERVER_NONCE, 0,
                           SAGR_STATUS_PROTOCOL_ERROR,
                           SAGR_WIRE_STATUS_OK);
  failures += run_scenario(SERVER_ZERO_EPOCH, 0,
                           SAGR_STATUS_PROTOCOL_ERROR,
                           SAGR_WIRE_STATUS_OK);
  failures += run_scenario(SERVER_BAD_MAXIMUM_RECORD, 0,
                           SAGR_STATUS_PROTOCOL_ERROR,
                           SAGR_WIRE_STATUS_OK);
  failures += run_scenario(SERVER_BAD_FLAGS, 0, SAGR_STATUS_PROTOCOL_ERROR,
                           -1);
  failures += run_scenario(SERVER_BAD_RESERVED, 0,
                           SAGR_STATUS_PROTOCOL_ERROR, -1);
  failures += run_scenario(SERVER_BAD_PAYLOAD_RESERVED, 0,
                           SAGR_STATUS_PROTOCOL_ERROR,
                           SAGR_WIRE_STATUS_OK);
  failures += run_scenario(SERVER_BAD_TYPE, 0, SAGR_STATUS_PROTOCOL_ERROR, -1);
  failures += test_endpoint_validation_and_stale_preservation();
  failures += test_public_validation();
  failures += test_deadline_and_cancellation_inputs();
  failures += test_blocking_cancellation_does_not_consume_fd();
  return failures == 0 ? 0 : 1;
}
