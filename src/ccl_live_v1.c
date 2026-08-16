/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <self_amdgpu_runtime/ccl_live_v1.h>

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <poll.h>
#include <sched.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define LIVE_BROKER_MAGIC UINT64_C(0x534147524c425231)
#define LIVE_RANK_MAGIC UINT64_C(0x534147524c524b31)
#define LIVE_WIRE_MAGIC UINT32_C(0x53434c56)
#define LIVE_WIRE_MAJOR UINT16_C(1)
#define LIVE_WIRE_MINOR UINT16_C(0)
#define LIVE_WIRE_BYTES UINT32_C(192)
#define LIVE_WIRE_CRC_OFFSET UINT32_C(188)
#define LIVE_MAX_CONTROL_FDS SAGR_CCL_V1_MAX_WORLD_SIZE

typedef enum live_message_kind {
  LIVE_MESSAGE_INVALID = 0,
  LIVE_MESSAGE_JOIN = 1,
  LIVE_MESSAGE_TABLE = 2,
  LIVE_MESSAGE_TABLE_ACK = 3,
  LIVE_MESSAGE_READY = 4,
  LIVE_MESSAGE_ABORT = 5,
  LIVE_MESSAGE_LEAVE = 6,
  LIVE_MESSAGE_CLOSED = 7
} live_message_kind_t;

typedef struct live_control_record {
  uint32_t kind;
  uint32_t claimed_rank;
  uint32_t failed_rank;
  int32_t status;
  uint32_t descriptor_count;
  uint32_t peer_mask;
  uint64_t context_sequence;
  sagr_ccl_v1_group_identity_t group;
} live_control_record_t;

struct sagr_ccl_live_v1_broker {
  uint64_t magic;
  uint32_t phase;
  uint32_t world_size;
  sagr_ccl_v1_group_identity_t group;
  sagr_ccl_live_v1_process_identity_t owner;
  int control[SAGR_CCL_V1_MAX_WORLD_SIZE];
  sagr_ccl_live_v1_process_identity_t
      expected_process[SAGR_CCL_V1_MAX_WORLD_SIZE];
  int transfer[SAGR_CCL_V1_MAX_WORLD_SIZE][SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint32_t prepared_mask;
  uint32_t bound_mask;
  uint32_t joined_mask;
  uint32_t table_sent_mask;
  uint32_t table_acked_mask;
  uint32_t ready_mask;
  uint32_t departed_mask;
  uint32_t close_pending_mask;
  uint32_t abort_pending_mask;
  atomic_uint first_error_state;
  sagr_ccl_live_v1_abort_t first_error;
};

struct sagr_ccl_live_v1_rank {
  uint64_t magic;
  uint32_t phase;
  uint32_t self_rank;
  uint32_t world_size;
  uint32_t leave_sent;
  int control;
  int peers[SAGR_CCL_V1_MAX_WORLD_SIZE];
  sagr_ccl_v1_group_identity_t group;
  sagr_ccl_live_v1_process_identity_t expected_broker;
  uint32_t has_first_error;
  sagr_ccl_live_v1_abort_t first_error;
};

static int bytes_zero(const uint8_t *bytes, size_t count) {
  size_t index;
  for (index = 0U; index < count; ++index) {
    if (bytes[index] != 0U) {
      return 0;
    }
  }
  return 1;
}

static uint32_t world_mask(uint32_t world_size) {
  return (UINT32_C(1) << world_size) - UINT32_C(1);
}

static void put_u16(uint8_t *destination, uint16_t value) {
  destination[0] = (uint8_t)(value >> 8U);
  destination[1] = (uint8_t)value;
}

static void put_u32(uint8_t *destination, uint32_t value) {
  destination[0] = (uint8_t)(value >> 24U);
  destination[1] = (uint8_t)(value >> 16U);
  destination[2] = (uint8_t)(value >> 8U);
  destination[3] = (uint8_t)value;
}

static void put_u64(uint8_t *destination, uint64_t value) {
  put_u32(destination, (uint32_t)(value >> 32U));
  put_u32(destination + 4, (uint32_t)value);
}

static uint16_t get_u16(const uint8_t *source) {
  return (uint16_t)(((uint16_t)source[0] << 8U) | (uint16_t)source[1]);
}

static uint32_t get_u32(const uint8_t *source) {
  return ((uint32_t)source[0] << 24U) | ((uint32_t)source[1] << 16U) |
         ((uint32_t)source[2] << 8U) | (uint32_t)source[3];
}

static uint64_t get_u64(const uint8_t *source) {
  return ((uint64_t)get_u32(source) << 32U) | (uint64_t)get_u32(source + 4);
}

static uint32_t crc32c(const uint8_t *bytes, size_t count) {
  uint32_t crc = UINT32_MAX;
  size_t index;
  for (index = 0U; index < count; ++index) {
    uint32_t value = crc ^ bytes[index];
    unsigned bit;
    for (bit = 0U; bit < 8U; ++bit) {
      const uint32_t mask = (uint32_t)-(int32_t)(value & UINT32_C(1));
      value = (value >> 1U) ^ (UINT32_C(0x82f63b78) & mask);
    }
    crc = value;
  }
  return ~crc;
}

static int valid_abort_status(sagr_ccl_v1_status_t status) {
  return status == SAGR_CCL_V1_STATUS_VERSION_MISMATCH ||
         status == SAGR_CCL_V1_STATUS_TOPOLOGY_MISMATCH ||
         status == SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH ||
         status == SAGR_CCL_V1_STATUS_SEQUENCE_MISMATCH ||
         status == SAGR_CCL_V1_STATUS_NOT_SUPPORTED ||
         status == SAGR_CCL_V1_STATUS_PROTOCOL_ERROR ||
         status == SAGR_CCL_V1_STATUS_CHECKSUM_ERROR ||
         status == SAGR_CCL_V1_STATUS_OUT_OF_ORDER ||
         status == SAGR_CCL_V1_STATUS_TIMED_OUT ||
         status == SAGR_CCL_V1_STATUS_PEER_LOST ||
         status == SAGR_CCL_V1_STATUS_CANCELLED ||
         status == SAGR_CCL_V1_STATUS_OUT_OF_RESOURCES;
}

static sagr_ccl_v1_status_t status_from_errno(int error) {
  switch (error) {
    case EAGAIN:
#if EWOULDBLOCK != EAGAIN
    case EWOULDBLOCK:
#endif
      return SAGR_CCL_V1_STATUS_BUSY;
    case EPIPE:
    case ECONNRESET:
    case ENOTCONN:
      return SAGR_CCL_V1_STATUS_PEER_LOST;
    case EMFILE:
    case ENFILE:
    case ENOMEM:
    case ENOBUFS:
    case ENOSPC:
      return SAGR_CCL_V1_STATUS_OUT_OF_RESOURCES;
    default:
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
}

uint64_t sagr_ccl_live_v1_monotonic_time_ns(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0 || value.tv_sec < 0) {
    return 0U;
  }
  return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
         (uint64_t)value.tv_nsec;
}

static int read_small_file(const char *path, char *buffer, size_t capacity) {
  int descriptor;
  size_t used = 0U;
  if (path == NULL || buffer == NULL || capacity < 2U) {
    return -1;
  }
  descriptor = open(path, O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return -1;
  }
  while (used + 1U < capacity) {
    ssize_t result = read(descriptor, buffer + used, capacity - used - 1U);
    if (result > 0) {
      used += (size_t)result;
      continue;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result < 0) {
      (void)close(descriptor);
      return -1;
    }
    break;
  }
  if (used + 1U == capacity) {
    char extra;
    ssize_t result;
    do {
      result = read(descriptor, &extra, 1U);
    } while (result < 0 && errno == EINTR);
    if (result != 0) {
      (void)close(descriptor);
      return -1;
    }
  }
  (void)close(descriptor);
  buffer[used] = '\0';
  return 0;
}

static int process_start_time(int32_t pid, uint64_t *start_time) {
  char path[64];
  char buffer[8192];
  char *close_parenthesis;
  char *save = NULL;
  char *token;
  unsigned index = 0U;
  if (pid <= 0 || start_time == NULL ||
      snprintf(path, sizeof(path), "/proc/%" PRId32 "/stat", pid) <= 0 ||
      read_small_file(path, buffer, sizeof(buffer)) != 0) {
    return -1;
  }
  close_parenthesis = strrchr(buffer, ')');
  if (close_parenthesis == NULL || close_parenthesis[1] != ' ') {
    return -1;
  }
  token = strtok_r(close_parenthesis + 2, " ", &save);
  while (token != NULL) {
    if (index == 19U) {
      char *end = NULL;
      unsigned long long parsed;
      errno = 0;
      parsed = strtoull(token, &end, 10);
      if (errno != 0 || end == token || (*end != '\0' && *end != '\n') ||
          parsed == 0ULL) {
        return -1;
      }
      *start_time = (uint64_t)parsed;
      return 0;
    }
    ++index;
    token = strtok_r(NULL, " ", &save);
  }
  return -1;
}

static int process_ids(int32_t pid, uint32_t *uid, uint32_t *gid) {
  char path[64];
  char buffer[8192];
  char *line;
  char *save = NULL;
  int have_uid = 0;
  int have_gid = 0;
  if (pid <= 0 || uid == NULL || gid == NULL ||
      snprintf(path, sizeof(path), "/proc/%" PRId32 "/status", pid) <= 0 ||
      read_small_file(path, buffer, sizeof(buffer)) != 0) {
    return -1;
  }
  line = strtok_r(buffer, "\n", &save);
  while (line != NULL) {
    unsigned long parsed;
    char tail;
    if (sscanf(line, "Uid:\t%lu%c", &parsed, &tail) >= 1 &&
        parsed <= UINT32_MAX) {
      *uid = (uint32_t)parsed;
      have_uid = 1;
    } else if (sscanf(line, "Gid:\t%lu%c", &parsed, &tail) >= 1 &&
               parsed <= UINT32_MAX) {
      *gid = (uint32_t)parsed;
      have_gid = 1;
    }
    line = strtok_r(NULL, "\n", &save);
  }
  return have_uid && have_gid ? 0 : -1;
}

sagr_ccl_v1_status_t sagr_ccl_live_v1_process_identity(
    int32_t pid, sagr_ccl_live_v1_process_identity_t *identity,
    uint32_t identity_size) {
  uint64_t first_start;
  uint64_t second_start;
  uint32_t uid;
  uint32_t gid;
  if (identity == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (identity_size < sizeof(*identity)) {
    if (identity_size >= sizeof(identity->struct_size)) {
      identity->struct_size = (uint32_t)sizeof(*identity);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  if (pid <= 0 || process_start_time(pid, &first_start) != 0 ||
      process_ids(pid, &uid, &gid) != 0 ||
      process_start_time(pid, &second_start) != 0 ||
      first_start != second_start) {
    return SAGR_CCL_V1_STATUS_PEER_LOST;
  }
  memset(identity, 0, sizeof(*identity));
  identity->struct_size = (uint32_t)sizeof(*identity);
  identity->pid = pid;
  identity->uid = uid;
  identity->gid = gid;
  identity->start_time_ticks = first_start;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static int process_identity_valid(
    const sagr_ccl_live_v1_process_identity_t *identity) {
  return identity != NULL && identity->struct_size == sizeof(*identity) &&
         identity->pid > 0 && identity->start_time_ticks != 0U &&
         bytes_zero(identity->reserved, sizeof(identity->reserved));
}

static int process_identity_equal(
    const sagr_ccl_live_v1_process_identity_t *left,
    const sagr_ccl_live_v1_process_identity_t *right) {
  return process_identity_valid(left) && process_identity_valid(right) &&
         left->pid == right->pid && left->uid == right->uid &&
         left->gid == right->gid &&
         left->start_time_ticks == right->start_time_ticks;
}

static int process_identity_is_current(
    const sagr_ccl_live_v1_process_identity_t *expected) {
  sagr_ccl_live_v1_process_identity_t current;
  return process_identity_valid(expected) &&
         sagr_ccl_live_v1_process_identity(
             expected->pid, &current, (uint32_t)sizeof(current)) ==
             SAGR_CCL_V1_STATUS_SUCCESS &&
         process_identity_equal(expected, &current);
}

static int credential_matches(const struct ucred *credential,
                              const sagr_ccl_live_v1_process_identity_t *expected) {
  return credential != NULL && process_identity_valid(expected) &&
         credential->pid == expected->pid &&
         (uint32_t)credential->uid == expected->uid &&
         (uint32_t)credential->gid == expected->gid &&
         process_identity_is_current(expected);
}

static sagr_ccl_v1_status_t encode_record(const live_control_record_t *record,
                                          uint8_t wire[LIVE_WIRE_BYTES]) {
  if (record == NULL || wire == NULL ||
      sagr_ccl_v1_group_identity_validate(&record->group) !=
          SAGR_CCL_V1_STATUS_SUCCESS) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  memset(wire, 0, LIVE_WIRE_BYTES);
  put_u32(wire + 0, LIVE_WIRE_MAGIC);
  put_u16(wire + 4, LIVE_WIRE_MAJOR);
  put_u16(wire + 6, LIVE_WIRE_MINOR);
  put_u32(wire + 8, record->kind);
  put_u32(wire + 12, record->claimed_rank);
  put_u32(wire + 16, record->failed_rank);
  put_u32(wire + 20, (uint32_t)record->status);
  put_u32(wire + 24, record->descriptor_count);
  put_u32(wire + 28, record->peer_mask);
  put_u64(wire + 32, record->context_sequence);
  put_u16(wire + 40, record->group.protocol_major);
  put_u16(wire + 42, record->group.protocol_minor);
  put_u32(wire + 44, record->group.world_size);
  put_u64(wire + 48, record->group.epoch);
  put_u64(wire + 56, record->group.group_generation);
  memcpy(wire + 64, record->group.job_uuid,
         SAGR_CCL_V1_UUID_BYTES);
  memcpy(wire + 80, record->group.group_uuid,
         SAGR_CCL_V1_UUID_BYTES);
  memcpy(wire + 96, record->group.model_identity_sha256,
         SAGR_CCL_V1_SHA256_BYTES);
  put_u32(wire + LIVE_WIRE_CRC_OFFSET,
          crc32c(wire, LIVE_WIRE_CRC_OFFSET));
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static sagr_ccl_v1_status_t decode_record(
    const uint8_t wire[LIVE_WIRE_BYTES], live_control_record_t *record) {
  sagr_ccl_v1_status_t status;
  uint32_t expected_crc;
  if (wire == NULL || record == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (get_u32(wire + 0) != LIVE_WIRE_MAGIC ||
      get_u16(wire + 4) != LIVE_WIRE_MAJOR ||
      get_u16(wire + 6) != LIVE_WIRE_MINOR) {
    return SAGR_CCL_V1_STATUS_VERSION_MISMATCH;
  }
  if (!bytes_zero(wire + 128, LIVE_WIRE_CRC_OFFSET - 128U)) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  expected_crc = get_u32(wire + LIVE_WIRE_CRC_OFFSET);
  if (expected_crc != crc32c(wire, LIVE_WIRE_CRC_OFFSET)) {
    return SAGR_CCL_V1_STATUS_CHECKSUM_ERROR;
  }
  memset(record, 0, sizeof(*record));
  record->kind = get_u32(wire + 8);
  record->claimed_rank = get_u32(wire + 12);
  record->failed_rank = get_u32(wire + 16);
  record->status = (int32_t)get_u32(wire + 20);
  record->descriptor_count = get_u32(wire + 24);
  record->peer_mask = get_u32(wire + 28);
  record->context_sequence = get_u64(wire + 32);
  status = sagr_ccl_v1_group_identity_init(
      &record->group, (uint32_t)sizeof(record->group));
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  record->group.protocol_major = get_u16(wire + 40);
  record->group.protocol_minor = get_u16(wire + 42);
  record->group.world_size = get_u32(wire + 44);
  record->group.epoch = get_u64(wire + 48);
  record->group.group_generation = get_u64(wire + 56);
  memcpy(record->group.job_uuid, wire + 64, SAGR_CCL_V1_UUID_BYTES);
  memcpy(record->group.group_uuid, wire + 80, SAGR_CCL_V1_UUID_BYTES);
  memcpy(record->group.model_identity_sha256, wire + 96,
         SAGR_CCL_V1_SHA256_BYTES);
  return sagr_ccl_v1_group_identity_validate(&record->group);
}

static void initialize_record(live_control_record_t *record, uint32_t kind,
                              const sagr_ccl_v1_group_identity_t *group,
                              uint32_t rank) {
  memset(record, 0, sizeof(*record));
  record->kind = kind;
  record->claimed_rank = rank;
  record->failed_rank = SAGR_CCL_LIVE_V1_NO_RANK;
  record->group = *group;
}

static sagr_ccl_v1_status_t validate_socket_shape(int descriptor,
                                                  int require_cloexec) {
  int type = 0;
  int domain = 0;
  int descriptor_flags;
  int status_flags;
  socklen_t size = (socklen_t)sizeof(type);
  if (descriptor < 0 ||
      getsockopt(descriptor, SOL_SOCKET, SO_TYPE, &type, &size) != 0 ||
      size != sizeof(type) || type != SOCK_SEQPACKET) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  size = (socklen_t)sizeof(domain);
  if (getsockopt(descriptor, SOL_SOCKET, SO_DOMAIN, &domain, &size) != 0 ||
      size != sizeof(domain) || domain != AF_UNIX) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  descriptor_flags = fcntl(descriptor, F_GETFD);
  status_flags = fcntl(descriptor, F_GETFL);
  if (descriptor_flags < 0 || status_flags < 0 ||
      (require_cloexec && (descriptor_flags & FD_CLOEXEC) == 0) ||
      (status_flags & O_NONBLOCK) == 0) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static sagr_ccl_v1_status_t validate_peer_identity(
    int descriptor,
    const sagr_ccl_live_v1_process_identity_t *expected_peer) {
  struct ucred credential;
  socklen_t size = (socklen_t)sizeof(credential);
  sagr_ccl_v1_status_t status = validate_socket_shape(descriptor, 1);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (!process_identity_valid(expected_peer) ||
      getsockopt(descriptor, SOL_SOCKET, SO_PEERCRED, &credential, &size) != 0 ||
      size != sizeof(credential) ||
      !credential_matches(&credential, expected_peer)) {
    return SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static void close_descriptors(int *descriptors, uint32_t count) {
  uint32_t index;
  if (descriptors == NULL) {
    return;
  }
  for (index = 0U; index < count; ++index) {
    if (descriptors[index] >= 0) {
      (void)close(descriptors[index]);
      descriptors[index] = -1;
    }
  }
}

static sagr_ccl_v1_status_t send_control(
    int descriptor, const live_control_record_t *record,
    const int *descriptors, uint32_t descriptor_count) {
  uint8_t wire[LIVE_WIRE_BYTES];
  struct iovec vector;
  struct msghdr message;
  unsigned char control[CMSG_SPACE(sizeof(int) * LIVE_MAX_CONTROL_FDS)];
  ssize_t sent;
  sagr_ccl_v1_status_t status;
  if (descriptor_count > LIVE_MAX_CONTROL_FDS ||
      (descriptor_count != 0U && descriptors == NULL)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  status = validate_socket_shape(descriptor, 1);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  status = encode_record(record, wire);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  memset(&message, 0, sizeof(message));
  memset(control, 0, sizeof(control));
  vector.iov_base = wire;
  vector.iov_len = sizeof(wire);
  message.msg_iov = &vector;
  message.msg_iovlen = 1U;
  if (descriptor_count != 0U) {
    struct cmsghdr *header;
    message.msg_control = control;
    message.msg_controllen = CMSG_SPACE(sizeof(int) * descriptor_count);
    header = CMSG_FIRSTHDR(&message);
    if (header == NULL) {
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    }
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(int) * descriptor_count);
    memcpy(CMSG_DATA(header), descriptors, sizeof(int) * descriptor_count);
  }
  do {
    sent = sendmsg(descriptor, &message, MSG_NOSIGNAL | MSG_DONTWAIT);
  } while (sent < 0 && errno == EINTR);
  if (sent == (ssize_t)sizeof(wire)) {
    return SAGR_CCL_V1_STATUS_SUCCESS;
  }
  if (sent < 0) {
    return status_from_errno(errno);
  }
  return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
}

static sagr_ccl_v1_status_t receive_control(
    int descriptor, live_control_record_t *record, int *descriptors,
    uint32_t descriptor_capacity, uint32_t *descriptor_count,
    struct ucred *credential) {
  uint8_t wire[LIVE_WIRE_BYTES];
  struct iovec vector;
  struct msghdr message;
  unsigned char control[CMSG_SPACE(sizeof(int) * LIVE_MAX_CONTROL_FDS) +
                        CMSG_SPACE(sizeof(struct ucred))];
  struct cmsghdr *header;
  uint32_t count = 0U;
  uint32_t credential_count = 0U;
  int malformed = 0;
  ssize_t received;
  sagr_ccl_v1_status_t status;
  if (record == NULL || descriptor_count == NULL || credential == NULL ||
      descriptor_capacity > LIVE_MAX_CONTROL_FDS ||
      (descriptor_capacity != 0U && descriptors == NULL)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  *descriptor_count = 0U;
  if (descriptors != NULL) {
    uint32_t index;
    for (index = 0U; index < descriptor_capacity; ++index) {
      descriptors[index] = -1;
    }
  }
  status = validate_socket_shape(descriptor, 1);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  memset(&message, 0, sizeof(message));
  memset(control, 0, sizeof(control));
  memset(credential, 0, sizeof(*credential));
  vector.iov_base = wire;
  vector.iov_len = sizeof(wire);
  message.msg_iov = &vector;
  message.msg_iovlen = 1U;
  message.msg_control = control;
  message.msg_controllen = sizeof(control);
  do {
    received = recvmsg(descriptor, &message,
                       MSG_CMSG_CLOEXEC | MSG_DONTWAIT);
  } while (received < 0 && errno == EINTR);
  if (received < 0) {
    return status_from_errno(errno);
  }
  for (header = CMSG_FIRSTHDR(&message); header != NULL;
       header = CMSG_NXTHDR(&message, header)) {
    if (header->cmsg_level == SOL_SOCKET &&
        header->cmsg_type == SCM_RIGHTS &&
        header->cmsg_len >= CMSG_LEN(0)) {
      const size_t bytes = header->cmsg_len - CMSG_LEN(0);
      const uint32_t observed = (uint32_t)(bytes / sizeof(int));
      const int *values = (const int *)CMSG_DATA(header);
      uint32_t index;
      if (bytes == 0U || bytes % sizeof(int) != 0U) {
        malformed = 1;
      }
      for (index = 0U; index < observed; ++index) {
        if (count < descriptor_capacity) {
          descriptors[count] = values[index];
        } else {
          (void)close(values[index]);
          malformed = 1;
        }
        ++count;
      }
    } else if (header->cmsg_level == SOL_SOCKET &&
               header->cmsg_type == SCM_CREDENTIALS &&
               header->cmsg_len == CMSG_LEN(sizeof(struct ucred))) {
      ++credential_count;
      if (credential_count == 1U) {
        memcpy(credential, CMSG_DATA(header), sizeof(*credential));
      } else {
        malformed = 1;
      }
    } else {
      malformed = 1;
    }
  }
  *descriptor_count = count > descriptor_capacity ? descriptor_capacity : count;
  if (received == 0) {
    close_descriptors(descriptors, *descriptor_count);
    return malformed || count != 0U ? SAGR_CCL_V1_STATUS_PROTOCOL_ERROR
                                    : SAGR_CCL_V1_STATUS_PEER_LOST;
  }
  if (received != (ssize_t)sizeof(wire) ||
      (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0 || malformed ||
      credential_count != 1U || count > descriptor_capacity) {
    close_descriptors(descriptors, *descriptor_count);
    *descriptor_count = 0U;
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  status = decode_record(wire, record);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    close_descriptors(descriptors, *descriptor_count);
    *descriptor_count = 0U;
  }
  return status;
}

static int deadline_timeout_ms(uint64_t deadline) {
  const uint64_t now = sagr_ccl_live_v1_monotonic_time_ns();
  uint64_t remaining;
  uint64_t milliseconds;
  if (deadline == 0U || now == 0U || now >= deadline) {
    return 0;
  }
  remaining = deadline - now;
  milliseconds = (remaining + UINT64_C(999999)) / UINT64_C(1000000);
  return milliseconds > (uint64_t)INT_MAX ? INT_MAX : (int)milliseconds;
}

static sagr_ccl_v1_status_t wait_one(int descriptor, short events,
                                     uint64_t deadline) {
  struct pollfd item;
  int result;
  for (;;) {
    const int timeout = deadline_timeout_ms(deadline);
    if (timeout == 0) {
      return SAGR_CCL_V1_STATUS_TIMED_OUT;
    }
    item.fd = descriptor;
    item.events = events;
    item.revents = 0;
    do {
      result = poll(&item, 1U, timeout);
    } while (result < 0 && errno == EINTR);
    if (result == 0) {
      return SAGR_CCL_V1_STATUS_TIMED_OUT;
    }
    if (result < 0) {
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    }
    if ((item.revents & events) != 0) {
      return SAGR_CCL_V1_STATUS_SUCCESS;
    }
    if ((item.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return SAGR_CCL_V1_STATUS_PEER_LOST;
    }
  }
}

static sagr_ccl_v1_status_t send_until(
    int descriptor, const live_control_record_t *record,
    const int *descriptors, uint32_t descriptor_count, uint64_t deadline) {
  for (;;) {
    sagr_ccl_v1_status_t status =
        send_control(descriptor, record, descriptors, descriptor_count);
    if (status != SAGR_CCL_V1_STATUS_BUSY) {
      return status;
    }
    status = wait_one(descriptor, POLLOUT, deadline);
    if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
      return status;
    }
  }
}

static sagr_ccl_v1_status_t receive_until(
    int descriptor, live_control_record_t *record, int *descriptors,
    uint32_t descriptor_capacity, uint32_t *descriptor_count,
    struct ucred *credential, uint64_t deadline) {
  for (;;) {
    sagr_ccl_v1_status_t status =
        receive_control(descriptor, record, descriptors, descriptor_capacity,
                        descriptor_count, credential);
    if (status != SAGR_CCL_V1_STATUS_BUSY) {
      return status;
    }
    status = wait_one(descriptor, POLLIN, deadline);
    if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
      return status;
    }
  }
}

static sagr_ccl_v1_status_t validate_abort_value(
    const sagr_ccl_live_v1_abort_t *value,
    const sagr_ccl_v1_group_identity_t *group) {
  if (value == NULL || value->struct_size != sizeof(*value) ||
      value->flags != 0U || value->reserved0 != 0U ||
      !bytes_zero(value->reserved, sizeof(value->reserved)) ||
      !sagr_ccl_v1_group_identity_equal(&value->group, group) ||
      (value->reporter_rank != SAGR_CCL_LIVE_V1_NO_RANK &&
       value->reporter_rank >= group->world_size) ||
      (value->failed_rank != SAGR_CCL_LIVE_V1_NO_RANK &&
       value->failed_rank >= group->world_size) ||
      !valid_abort_status(value->status)) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static void make_abort(sagr_ccl_live_v1_abort_t *value,
                       const sagr_ccl_v1_group_identity_t *group,
                       uint32_t reporter_rank, uint32_t failed_rank,
                       sagr_ccl_v1_status_t reason,
                       uint64_t context_sequence) {
  memset(value, 0, sizeof(*value));
  value->struct_size = (uint32_t)sizeof(*value);
  value->group = *group;
  value->context_sequence = context_sequence;
  value->reporter_rank = reporter_rank;
  value->failed_rank = failed_rank < group->world_size
                           ? failed_rank
                           : SAGR_CCL_LIVE_V1_NO_RANK;
  value->status = valid_abort_status(reason)
                      ? reason
                      : SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
}

static live_control_record_t record_from_abort(
    const sagr_ccl_live_v1_abort_t *value) {
  live_control_record_t record;
  initialize_record(&record, LIVE_MESSAGE_ABORT, &value->group,
                    value->reporter_rank);
  record.failed_rank = value->failed_rank;
  record.status = value->status;
  record.context_sequence = value->context_sequence;
  return record;
}

static sagr_ccl_v1_status_t abort_from_record(
    const live_control_record_t *record,
    const sagr_ccl_v1_group_identity_t *group,
    sagr_ccl_live_v1_abort_t *value) {
  if (record == NULL || value == NULL || record->kind != LIVE_MESSAGE_ABORT ||
      record->descriptor_count != 0U || record->peer_mask != 0U ||
      !valid_abort_status(record->status)) {
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  if (!sagr_ccl_v1_group_identity_equal(&record->group, group)) {
    return SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
  }
  make_abort(value, group, record->claimed_rank, record->failed_rank,
             record->status, record->context_sequence);
  return validate_abort_value(value, group);
}

static int validate_broker(sagr_ccl_live_v1_broker_t broker) {
  return broker != NULL && broker->magic == LIVE_BROKER_MAGIC &&
         broker->world_size == broker->group.world_size &&
         sagr_ccl_v1_group_identity_validate(&broker->group) ==
             SAGR_CCL_V1_STATUS_SUCCESS &&
         process_identity_is_current(&broker->owner) &&
         broker->owner.pid == (int32_t)getpid();
}

static int validate_rank(sagr_ccl_live_v1_rank_t rank) {
  return rank != NULL && rank->magic == LIVE_RANK_MAGIC &&
         rank->world_size == rank->group.world_size &&
         rank->self_rank < rank->world_size && rank->control >= 0 &&
         sagr_ccl_v1_group_identity_validate(&rank->group) ==
             SAGR_CCL_V1_STATUS_SUCCESS &&
         process_identity_valid(&rank->expected_broker);
}

static void broker_close_transfers(sagr_ccl_live_v1_broker_t broker) {
  uint32_t rank;
  uint32_t peer;
  for (rank = 0U; rank < SAGR_CCL_V1_MAX_WORLD_SIZE; ++rank) {
    for (peer = 0U; peer < SAGR_CCL_V1_MAX_WORLD_SIZE; ++peer) {
      if (broker->transfer[rank][peer] >= 0) {
        (void)close(broker->transfer[rank][peer]);
        broker->transfer[rank][peer] = -1;
      }
    }
  }
}

static int broker_latch_first_error(
    sagr_ccl_live_v1_broker_t broker,
    const sagr_ccl_live_v1_abort_t *candidate) {
  unsigned expected = 0U;
  if (validate_abort_value(candidate, &broker->group) !=
      SAGR_CCL_V1_STATUS_SUCCESS) {
    return 0;
  }
  if (atomic_compare_exchange_strong_explicit(
          &broker->first_error_state, &expected, 1U, memory_order_acq_rel,
          memory_order_acquire)) {
    broker->first_error = *candidate;
    broker->phase = SAGR_CCL_LIVE_V1_PHASE_ABORTED;
    broker->close_pending_mask = 0U;
    broker->abort_pending_mask = broker->prepared_mask;
    broker_close_transfers(broker);
    atomic_store_explicit(&broker->first_error_state, 2U,
                          memory_order_release);
    return 1;
  }
  while (atomic_load_explicit(&broker->first_error_state,
                              memory_order_acquire) == 1U) {
    (void)sched_yield();
  }
  return 0;
}

static void broker_latch_reason(sagr_ccl_live_v1_broker_t broker,
                                uint32_t reporter_rank,
                                uint32_t failed_rank,
                                sagr_ccl_v1_status_t reason,
                                uint64_t sequence) {
  sagr_ccl_live_v1_abort_t candidate;
  make_abort(&candidate, &broker->group, reporter_rank, failed_rank, reason,
             sequence);
  (void)broker_latch_first_error(broker, &candidate);
}

static sagr_ccl_v1_status_t broker_first_status(
    sagr_ccl_live_v1_broker_t broker) {
  if (atomic_load_explicit(&broker->first_error_state,
                           memory_order_acquire) != 2U) {
    return SAGR_CCL_V1_STATUS_SUCCESS;
  }
  return broker->first_error.status;
}

sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_create(
    const sagr_ccl_v1_group_identity_t *group,
    sagr_ccl_live_v1_broker_t *broker) {
  sagr_ccl_live_v1_broker_t created;
  uint32_t rank;
  uint32_t peer;
  sagr_ccl_v1_status_t status;
  if (broker == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  *broker = NULL;
  status = sagr_ccl_v1_group_identity_validate(group);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  created = (sagr_ccl_live_v1_broker_t)calloc(1U, sizeof(*created));
  if (created == NULL) {
    return SAGR_CCL_V1_STATUS_OUT_OF_RESOURCES;
  }
  created->magic = LIVE_BROKER_MAGIC;
  created->phase = SAGR_CCL_LIVE_V1_PHASE_CONFIGURING;
  created->world_size = group->world_size;
  created->group = *group;
  for (rank = 0U; rank < SAGR_CCL_V1_MAX_WORLD_SIZE; ++rank) {
    created->control[rank] = -1;
    for (peer = 0U; peer < SAGR_CCL_V1_MAX_WORLD_SIZE; ++peer) {
      created->transfer[rank][peer] = -1;
    }
  }
  atomic_init(&created->first_error_state, 0U);
  status = sagr_ccl_live_v1_process_identity(
      (int32_t)getpid(), &created->owner,
      (uint32_t)sizeof(created->owner));
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    free(created);
    return status;
  }
  *broker = created;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_info(
    sagr_ccl_live_v1_broker_t broker,
    sagr_ccl_live_v1_broker_info_t *info, uint32_t info_size) {
  if (info == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (info_size < sizeof(*info)) {
    if (info_size >= sizeof(info->struct_size)) {
      info->struct_size = (uint32_t)sizeof(*info);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  if (!validate_broker(broker)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  info->phase = broker->phase;
  info->world_size = broker->world_size;
  info->prepared_mask = broker->prepared_mask;
  info->bound_mask = broker->bound_mask;
  info->joined_mask = broker->joined_mask;
  info->ready_mask = broker->ready_mask;
  info->departed_mask = broker->departed_mask;
  info->close_pending_mask = broker->close_pending_mask;
  info->abort_pending_mask = broker->abort_pending_mask;
  info->owner = broker->owner;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_prepare_rank(
    sagr_ccl_live_v1_broker_t broker, uint32_t rank,
    int *rank_capability_socket) {
  int pair[2] = {-1, -1};
  int enabled = 1;
  uint32_t bit;
  if (rank_capability_socket == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  *rank_capability_socket = -1;
  if (!validate_broker(broker) ||
      broker->phase != SAGR_CCL_LIVE_V1_PHASE_CONFIGURING ||
      rank >= broker->world_size) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  bit = UINT32_C(1) << rank;
  if ((broker->prepared_mask & bit) != 0U) {
    return SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
  }
  if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0,
                 pair) != 0) {
    return status_from_errno(errno);
  }
  if (setsockopt(pair[0], SOL_SOCKET, SO_PASSCRED, &enabled,
                 (socklen_t)sizeof(enabled)) != 0 ||
      setsockopt(pair[1], SOL_SOCKET, SO_PASSCRED, &enabled,
                 (socklen_t)sizeof(enabled)) != 0) {
    (void)close(pair[0]);
    (void)close(pair[1]);
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  broker->control[rank] = pair[0];
  broker->prepared_mask |= bit;
  *rank_capability_socket = pair[1];
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_bind_rank(
    sagr_ccl_live_v1_broker_t broker, uint32_t rank,
    const sagr_ccl_live_v1_process_identity_t *process) {
  uint32_t other;
  uint32_t bit;
  if (!validate_broker(broker) ||
      broker->phase != SAGR_CCL_LIVE_V1_PHASE_CONFIGURING ||
      rank >= broker->world_size || !process_identity_is_current(process) ||
      process->uid != broker->owner.uid || process->gid != broker->owner.gid ||
      process->pid == broker->owner.pid) {
    return SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
  }
  bit = UINT32_C(1) << rank;
  if ((broker->prepared_mask & bit) == 0U) {
    return SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
  }
  if ((broker->bound_mask & bit) != 0U) {
    return SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
  }
  for (other = 0U; other < broker->world_size; ++other) {
    if ((broker->bound_mask & (UINT32_C(1) << other)) != 0U &&
        broker->expected_process[other].pid == process->pid) {
      return SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
    }
  }
  broker->expected_process[rank] = *process;
  broker->bound_mask |= bit;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static sagr_ccl_v1_status_t create_peer_tables(
    sagr_ccl_live_v1_broker_t broker) {
  uint32_t left;
  uint32_t right;
  for (left = 0U; left < broker->world_size; ++left) {
    for (right = left + 1U; right < broker->world_size; ++right) {
      int pair[2] = {-1, -1};
      if (socketpair(AF_UNIX,
                     SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0,
                     pair) != 0) {
        broker_close_transfers(broker);
        return status_from_errno(errno);
      }
      broker->transfer[left][right] = pair[0];
      broker->transfer[right][left] = pair[1];
    }
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static sagr_ccl_v1_status_t broker_receive_rank(
    sagr_ccl_live_v1_broker_t broker, uint32_t rank,
    live_control_record_t *record, uint64_t deadline, int blocking) {
  int unexpected[LIVE_MAX_CONTROL_FDS];
  uint32_t descriptor_count = 0U;
  struct ucred credential;
  sagr_ccl_v1_status_t status =
      blocking ? receive_until(broker->control[rank], record, unexpected,
                               LIVE_MAX_CONTROL_FDS, &descriptor_count,
                               &credential, deadline)
               : receive_control(broker->control[rank], record, unexpected,
                                 LIVE_MAX_CONTROL_FDS, &descriptor_count,
                                 &credential);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (!credential_matches(&credential, &broker->expected_process[rank])) {
    close_descriptors(unexpected, descriptor_count);
    return SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
  }
  if (descriptor_count != 0U) {
    close_descriptors(unexpected, descriptor_count);
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static uint32_t first_missing_rank(uint32_t mask, uint32_t world_size) {
  uint32_t rank;
  for (rank = 0U; rank < world_size; ++rank) {
    if ((mask & (UINT32_C(1) << rank)) == 0U) {
      return rank;
    }
  }
  return SAGR_CCL_LIVE_V1_NO_RANK;
}

static sagr_ccl_v1_status_t broker_wait_for_messages(
    sagr_ccl_live_v1_broker_t broker, uint32_t wanted_mask,
    uint64_t deadline, uint32_t expected_kind) {
  struct pollfd items[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint32_t rank;
  for (;;) {
    int result;
    int timeout;
    uint32_t observed_mask = expected_kind == LIVE_MESSAGE_JOIN
                                 ? broker->joined_mask
                                 : broker->table_acked_mask;
    if ((observed_mask & wanted_mask) == wanted_mask) {
      return SAGR_CCL_V1_STATUS_SUCCESS;
    }
    timeout = deadline_timeout_ms(deadline);
    if (timeout == 0) {
      broker_latch_reason(
          broker, SAGR_CCL_LIVE_V1_NO_RANK,
          first_missing_rank(observed_mask, broker->world_size),
          SAGR_CCL_V1_STATUS_TIMED_OUT, 0U);
      return SAGR_CCL_V1_STATUS_TIMED_OUT;
    }
    for (rank = 0U; rank < broker->world_size; ++rank) {
      items[rank].fd = broker->control[rank];
      items[rank].events = POLLIN;
      items[rank].revents = 0;
    }
    do {
      result = poll(items, broker->world_size, timeout);
    } while (result < 0 && errno == EINTR);
    if (result == 0) {
      broker_latch_reason(
          broker, SAGR_CCL_LIVE_V1_NO_RANK,
          first_missing_rank(observed_mask, broker->world_size),
          SAGR_CCL_V1_STATUS_TIMED_OUT, 0U);
      return SAGR_CCL_V1_STATUS_TIMED_OUT;
    }
    if (result < 0) {
      broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK,
                          SAGR_CCL_LIVE_V1_NO_RANK,
                          SAGR_CCL_V1_STATUS_PROTOCOL_ERROR, 0U);
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    }
    for (rank = 0U; rank < broker->world_size; ++rank) {
      live_control_record_t record;
      sagr_ccl_v1_status_t status;
      uint32_t bit = UINT32_C(1) << rank;
      if ((items[rank].revents & POLLIN) == 0) {
        if ((items[rank].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
          broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK, rank,
                              SAGR_CCL_V1_STATUS_PEER_LOST, 0U);
          return SAGR_CCL_V1_STATUS_PEER_LOST;
        }
        continue;
      }
      status = broker_receive_rank(broker, rank, &record, deadline, 0);
      if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
        broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK, rank, status,
                            0U);
        return status;
      }
      if (record.kind == LIVE_MESSAGE_ABORT) {
        sagr_ccl_live_v1_abort_t received_abort;
        status = abort_from_record(&record, &broker->group, &received_abort);
        if (status == SAGR_CCL_V1_STATUS_SUCCESS &&
            received_abort.reporter_rank == rank) {
          (void)broker_latch_first_error(broker, &received_abort);
          return received_abort.status;
        }
        broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK, rank,
                            SAGR_CCL_V1_STATUS_PROTOCOL_ERROR, 0U);
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      if (record.kind != expected_kind || record.claimed_rank != rank ||
          record.failed_rank != SAGR_CCL_LIVE_V1_NO_RANK ||
          record.status != SAGR_CCL_V1_STATUS_SUCCESS ||
          record.descriptor_count != 0U || record.peer_mask != 0U ||
          record.context_sequence != 0U) {
        broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK, rank,
                            SAGR_CCL_V1_STATUS_PROTOCOL_ERROR, 0U);
        return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
      }
      if (!sagr_ccl_v1_group_identity_equal(&record.group, &broker->group)) {
        broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK, rank,
                            SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH, 0U);
        return SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
      }
      if ((observed_mask & bit) != 0U) {
        broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK, rank,
                            SAGR_CCL_V1_STATUS_OUT_OF_ORDER, 0U);
        return SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
      }
      observed_mask |= bit;
      if (expected_kind == LIVE_MESSAGE_JOIN) {
        broker->joined_mask = observed_mask;
      } else {
        broker->table_acked_mask = observed_mask;
      }
    }
  }
}

static sagr_ccl_v1_status_t broker_send_table(
    sagr_ccl_live_v1_broker_t broker, uint32_t rank, uint64_t deadline) {
  live_control_record_t record;
  int descriptors[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint32_t count = 0U;
  uint32_t peer;
  sagr_ccl_v1_status_t status;
  initialize_record(&record, LIVE_MESSAGE_TABLE, &broker->group, rank);
  record.peer_mask = world_mask(broker->world_size) & ~(UINT32_C(1) << rank);
  for (peer = 0U; peer < broker->world_size; ++peer) {
    if (peer == rank) {
      continue;
    }
    if (broker->transfer[rank][peer] < 0) {
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    }
    descriptors[count++] = broker->transfer[rank][peer];
  }
  record.descriptor_count = count;
  status = send_until(broker->control[rank], &record, descriptors, count,
                      deadline);
  if (status == SAGR_CCL_V1_STATUS_SUCCESS) {
    for (peer = 0U; peer < broker->world_size; ++peer) {
      if (peer != rank && broker->transfer[rank][peer] >= 0) {
        (void)close(broker->transfer[rank][peer]);
        broker->transfer[rank][peer] = -1;
      }
    }
    broker->table_sent_mask |= UINT32_C(1) << rank;
  }
  return status;
}

static void broker_flush_abort(sagr_ccl_live_v1_broker_t broker) {
  live_control_record_t record;
  uint32_t rank;
  if (atomic_load_explicit(&broker->first_error_state,
                           memory_order_acquire) != 2U) {
    return;
  }
  record = record_from_abort(&broker->first_error);
  for (rank = 0U; rank < broker->world_size; ++rank) {
    const uint32_t bit = UINT32_C(1) << rank;
    sagr_ccl_v1_status_t status;
    if ((broker->abort_pending_mask & bit) == 0U ||
        broker->control[rank] < 0) {
      continue;
    }
    status = send_control(broker->control[rank], &record, NULL, 0U);
    if (status == SAGR_CCL_V1_STATUS_SUCCESS ||
        status == SAGR_CCL_V1_STATUS_PEER_LOST ||
        status == SAGR_CCL_V1_STATUS_PROTOCOL_ERROR) {
      broker->abort_pending_mask &= ~bit;
    }
  }
}

sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_rendezvous(
    sagr_ccl_live_v1_broker_t broker, uint64_t absolute_deadline_ns) {
  const uint32_t complete =
      broker != NULL ? world_mask(broker->world_size) : 0U;
  uint32_t rank;
  sagr_ccl_v1_status_t status;
  if (!validate_broker(broker) ||
      broker->phase != SAGR_CCL_LIVE_V1_PHASE_CONFIGURING ||
      broker->prepared_mask != complete || broker->bound_mask != complete ||
      absolute_deadline_ns == 0U) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  broker->phase = SAGR_CCL_LIVE_V1_PHASE_JOINING;
  status = broker_wait_for_messages(broker, complete, absolute_deadline_ns,
                                    LIVE_MESSAGE_JOIN);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    broker_flush_abort(broker);
    return broker_first_status(broker);
  }
  status = create_peer_tables(broker);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK,
                        SAGR_CCL_LIVE_V1_NO_RANK, status, 0U);
    broker_flush_abort(broker);
    return broker_first_status(broker);
  }
  for (rank = 0U; rank < broker->world_size; ++rank) {
    status = broker_send_table(broker, rank, absolute_deadline_ns);
    if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
      broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK, rank, status, 0U);
      broker_flush_abort(broker);
      return broker_first_status(broker);
    }
  }
  status = broker_wait_for_messages(broker, complete, absolute_deadline_ns,
                                    LIVE_MESSAGE_TABLE_ACK);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    broker_flush_abort(broker);
    return broker_first_status(broker);
  }
  for (rank = 0U; rank < broker->world_size; ++rank) {
    live_control_record_t ready;
    initialize_record(&ready, LIVE_MESSAGE_READY, &broker->group, rank);
    status = send_until(broker->control[rank], &ready, NULL, 0U,
                        absolute_deadline_ns);
    if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
      broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK, rank, status, 0U);
      broker_flush_abort(broker);
      return broker_first_status(broker);
    }
    broker->ready_mask |= UINT32_C(1) << rank;
  }
  broker->phase = SAGR_CCL_LIVE_V1_PHASE_READY;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static sagr_ccl_v1_status_t copy_first_error(
    const sagr_ccl_live_v1_abort_t *source,
    sagr_ccl_live_v1_abort_t *destination, uint32_t destination_size) {
  if (destination == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (destination_size < sizeof(*destination)) {
    if (destination_size >= sizeof(destination->struct_size)) {
      destination->struct_size = (uint32_t)sizeof(*destination);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  *destination = *source;
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_first_error(
    sagr_ccl_live_v1_broker_t broker,
    sagr_ccl_live_v1_abort_t *first_error, uint32_t first_error_size) {
  if (!validate_broker(broker)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (atomic_load_explicit(&broker->first_error_state,
                           memory_order_acquire) != 2U) {
    return SAGR_CCL_V1_STATUS_BUSY;
  }
  return copy_first_error(&broker->first_error, first_error,
                          first_error_size);
}

sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_abort(
    sagr_ccl_live_v1_broker_t broker, uint32_t failed_rank,
    sagr_ccl_v1_status_t reason, uint64_t context_sequence) {
  if (!validate_broker(broker) || !valid_abort_status(reason) ||
      (failed_rank != SAGR_CCL_LIVE_V1_NO_RANK &&
       failed_rank >= broker->world_size) ||
      broker->phase == SAGR_CCL_LIVE_V1_PHASE_CLOSED) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK, failed_rank, reason,
                      context_sequence);
  broker_flush_abort(broker);
  return broker_first_status(broker);
}

static void broker_flush_closed(sagr_ccl_live_v1_broker_t broker) {
  live_control_record_t record;
  uint32_t rank;
  initialize_record(&record, LIVE_MESSAGE_CLOSED, &broker->group,
                    SAGR_CCL_LIVE_V1_NO_RANK);
  for (rank = 0U; rank < broker->world_size; ++rank) {
    const uint32_t bit = UINT32_C(1) << rank;
    sagr_ccl_v1_status_t status;
    if ((broker->close_pending_mask & bit) == 0U) {
      continue;
    }
    status = send_control(broker->control[rank], &record, NULL, 0U);
    if (status == SAGR_CCL_V1_STATUS_SUCCESS) {
      broker->close_pending_mask &= ~bit;
    } else if (status != SAGR_CCL_V1_STATUS_BUSY) {
      broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK, rank,
                          SAGR_CCL_V1_STATUS_PEER_LOST, 0U);
      return;
    }
  }
  if (broker->close_pending_mask == 0U &&
      broker->phase == SAGR_CCL_LIVE_V1_PHASE_CLOSING) {
    broker->phase = SAGR_CCL_LIVE_V1_PHASE_CLOSED;
  }
}

sagr_ccl_v1_status_t sagr_ccl_live_v1_broker_progress(
    sagr_ccl_live_v1_broker_t broker,
    sagr_ccl_live_v1_abort_t *first_error, uint32_t first_error_size) {
  uint32_t rank;
  const uint32_t complete =
      broker != NULL ? world_mask(broker->world_size) : 0U;
  if (!validate_broker(broker)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (broker->phase == SAGR_CCL_LIVE_V1_PHASE_CLOSED) {
    return SAGR_CCL_V1_STATUS_CLOSED;
  }
  if (atomic_load_explicit(&broker->first_error_state,
                           memory_order_acquire) == 2U) {
    broker_flush_abort(broker);
    if (first_error != NULL) {
      sagr_ccl_v1_status_t copied =
          copy_first_error(&broker->first_error, first_error, first_error_size);
      if (copied != SAGR_CCL_V1_STATUS_SUCCESS) {
        return copied;
      }
    }
    return broker->first_error.status;
  }
  if (broker->phase != SAGR_CCL_LIVE_V1_PHASE_READY &&
      broker->phase != SAGR_CCL_LIVE_V1_PHASE_CLOSING) {
    return SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
  }
  for (rank = 0U; rank < broker->world_size; ++rank) {
    const uint32_t bit = UINT32_C(1) << rank;
    if (broker->phase == SAGR_CCL_LIVE_V1_PHASE_CLOSING &&
        (broker->close_pending_mask & bit) == 0U) {
      continue;
    }
    for (;;) {
      live_control_record_t record;
      sagr_ccl_v1_status_t status =
          broker_receive_rank(broker, rank, &record, 0U, 0);
      if (status == SAGR_CCL_V1_STATUS_BUSY) {
        break;
      }
      if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
        broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK, rank,
                            status == SAGR_CCL_V1_STATUS_PEER_LOST
                                ? SAGR_CCL_V1_STATUS_PEER_LOST
                                : status,
                            0U);
        break;
      }
      if (record.kind == LIVE_MESSAGE_ABORT) {
        sagr_ccl_live_v1_abort_t received_abort;
        status = abort_from_record(&record, &broker->group, &received_abort);
        if (status == SAGR_CCL_V1_STATUS_SUCCESS &&
            received_abort.reporter_rank == rank) {
          (void)broker_latch_first_error(broker, &received_abort);
        } else {
          broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK, rank,
                              SAGR_CCL_V1_STATUS_PROTOCOL_ERROR, 0U);
        }
        break;
      }
      if (broker->phase == SAGR_CCL_LIVE_V1_PHASE_CLOSING) {
        broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK, rank,
                            SAGR_CCL_V1_STATUS_PROTOCOL_ERROR, 0U);
        break;
      }
      if (record.kind != LIVE_MESSAGE_LEAVE ||
          record.claimed_rank != rank ||
          record.failed_rank != SAGR_CCL_LIVE_V1_NO_RANK ||
          record.status != SAGR_CCL_V1_STATUS_SUCCESS ||
          record.descriptor_count != 0U || record.peer_mask != 0U ||
          record.context_sequence != 0U ||
          !sagr_ccl_v1_group_identity_equal(&record.group, &broker->group) ||
          (broker->departed_mask & (UINT32_C(1) << rank)) != 0U) {
        broker_latch_reason(broker, SAGR_CCL_LIVE_V1_NO_RANK, rank,
                            SAGR_CCL_V1_STATUS_PROTOCOL_ERROR, 0U);
        break;
      }
      broker->departed_mask |= UINT32_C(1) << rank;
      /* Drain an abort queued immediately after LEAVE before sending CLOSED. */
    }
    if (atomic_load_explicit(&broker->first_error_state,
                             memory_order_acquire) == 2U) {
      break;
    }
  }
  if (atomic_load_explicit(&broker->first_error_state,
                           memory_order_acquire) == 2U) {
    broker_flush_abort(broker);
    if (first_error != NULL) {
      sagr_ccl_v1_status_t copied =
          copy_first_error(&broker->first_error, first_error, first_error_size);
      if (copied != SAGR_CCL_V1_STATUS_SUCCESS) {
        return copied;
      }
    }
    return broker->first_error.status;
  }
  if (broker->departed_mask == complete &&
      broker->phase == SAGR_CCL_LIVE_V1_PHASE_READY) {
    broker->phase = SAGR_CCL_LIVE_V1_PHASE_CLOSING;
    broker->close_pending_mask = complete;
    /* A later progress turn drains failures queued behind the final LEAVE. */
    return SAGR_CCL_V1_STATUS_SUCCESS;
  }
  if (broker->phase == SAGR_CCL_LIVE_V1_PHASE_CLOSING) {
    broker_flush_closed(broker);
  }
  return broker->phase == SAGR_CCL_LIVE_V1_PHASE_CLOSED
             ? SAGR_CCL_V1_STATUS_CLOSED
             : SAGR_CCL_V1_STATUS_SUCCESS;
}

void sagr_ccl_live_v1_broker_destroy(sagr_ccl_live_v1_broker_t *broker) {
  sagr_ccl_live_v1_broker_t value;
  uint32_t rank;
  if (broker == NULL || *broker == NULL) {
    return;
  }
  value = *broker;
  if (value->magic != LIVE_BROKER_MAGIC) {
    *broker = NULL;
    return;
  }
  broker_close_transfers(value);
  for (rank = 0U; rank < SAGR_CCL_V1_MAX_WORLD_SIZE; ++rank) {
    if (value->control[rank] >= 0) {
      (void)close(value->control[rank]);
      value->control[rank] = -1;
    }
  }
  value->phase = SAGR_CCL_LIVE_V1_PHASE_CLOSED;
  value->magic = 0U;
  free(value);
  *broker = NULL;
}

static void rank_close_owned(sagr_ccl_live_v1_rank_t rank) {
  uint32_t peer;
  if (rank == NULL) {
    return;
  }
  if (rank->control >= 0) {
    (void)close(rank->control);
    rank->control = -1;
  }
  for (peer = 0U; peer < SAGR_CCL_V1_MAX_WORLD_SIZE; ++peer) {
    if (rank->peers[peer] >= 0) {
      (void)close(rank->peers[peer]);
      rank->peers[peer] = -1;
    }
  }
}

static sagr_ccl_v1_status_t rank_receive(
    sagr_ccl_live_v1_rank_t rank, live_control_record_t *record,
    int *descriptors, uint32_t descriptor_capacity,
    uint32_t *descriptor_count, uint64_t deadline, int blocking) {
  struct ucred credential;
  sagr_ccl_v1_status_t status =
      blocking ? receive_until(rank->control, record, descriptors,
                               descriptor_capacity, descriptor_count,
                               &credential, deadline)
               : receive_control(rank->control, record, descriptors,
                                 descriptor_capacity, descriptor_count,
                                 &credential);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (!credential_matches(&credential, &rank->expected_broker)) {
    close_descriptors(descriptors, *descriptor_count);
    *descriptor_count = 0U;
    return SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

static void rank_store_abort(sagr_ccl_live_v1_rank_t rank,
                             const sagr_ccl_live_v1_abort_t *value) {
  if (!rank->has_first_error) {
    rank->first_error = *value;
    rank->has_first_error = 1U;
    rank->phase = SAGR_CCL_LIVE_V1_PHASE_ABORTED;
  }
}

static void rank_report_join_failure(sagr_ccl_live_v1_rank_t rank,
                                     sagr_ccl_v1_status_t reason) {
  sagr_ccl_live_v1_abort_t value;
  live_control_record_t record;
  make_abort(&value, &rank->group, rank->self_rank, rank->self_rank,
             valid_abort_status(reason) ? reason
                                        : SAGR_CCL_V1_STATUS_PROTOCOL_ERROR,
             0U);
  record = record_from_abort(&value);
  (void)send_control(rank->control, &record, NULL, 0U);
}

sagr_ccl_v1_status_t sagr_ccl_live_v1_rank_join(
    int capability_socket, const sagr_ccl_v1_group_identity_t *group,
    uint32_t self_rank,
    const sagr_ccl_live_v1_process_identity_t *expected_broker,
    uint64_t absolute_deadline_ns, sagr_ccl_live_v1_rank_t *rank) {
  sagr_ccl_live_v1_rank_t created = NULL;
  live_control_record_t record;
  int received_fds[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint32_t received_count = 0U;
  uint32_t peer;
  uint32_t expected_mask;
  sagr_ccl_v1_status_t status;
  if (rank == NULL) {
    if (capability_socket >= 0) {
      (void)close(capability_socket);
    }
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  *rank = NULL;
  if (capability_socket < 0 || group == NULL ||
      sagr_ccl_v1_group_identity_validate(group) !=
          SAGR_CCL_V1_STATUS_SUCCESS ||
      self_rank >= group->world_size || !process_identity_valid(expected_broker) ||
      absolute_deadline_ns == 0U) {
    if (capability_socket >= 0) {
      (void)close(capability_socket);
    }
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  {
    int flags = fcntl(capability_socket, F_GETFD);
    if (flags < 0 || fcntl(capability_socket, F_SETFD, flags | FD_CLOEXEC) != 0) {
      (void)close(capability_socket);
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    }
  }
  status = validate_peer_identity(capability_socket, expected_broker);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    (void)close(capability_socket);
    return status;
  }
  created = (sagr_ccl_live_v1_rank_t)calloc(1U, sizeof(*created));
  if (created == NULL) {
    (void)close(capability_socket);
    return SAGR_CCL_V1_STATUS_OUT_OF_RESOURCES;
  }
  created->magic = LIVE_RANK_MAGIC;
  created->phase = SAGR_CCL_LIVE_V1_PHASE_JOINING;
  created->self_rank = self_rank;
  created->world_size = group->world_size;
  created->control = capability_socket;
  created->group = *group;
  created->expected_broker = *expected_broker;
  for (peer = 0U; peer < SAGR_CCL_V1_MAX_WORLD_SIZE; ++peer) {
    created->peers[peer] = -1;
    received_fds[peer] = -1;
  }
  initialize_record(&record, LIVE_MESSAGE_JOIN, group, self_rank);
  status = send_until(created->control, &record, NULL, 0U,
                      absolute_deadline_ns);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    goto fail;
  }
  status = rank_receive(created, &record, received_fds,
                        SAGR_CCL_V1_MAX_WORLD_SIZE, &received_count,
                        absolute_deadline_ns, 1);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    goto fail;
  }
  if (record.kind == LIVE_MESSAGE_ABORT) {
    sagr_ccl_live_v1_abort_t value;
    status = abort_from_record(&record, group, &value);
    if (status == SAGR_CCL_V1_STATUS_SUCCESS) {
      rank_store_abort(created, &value);
      status = value.status;
    }
    goto fail;
  }
  expected_mask = world_mask(group->world_size) &
                  ~(UINT32_C(1) << self_rank);
  if (record.kind != LIVE_MESSAGE_TABLE ||
      record.claimed_rank != self_rank ||
      record.failed_rank != SAGR_CCL_LIVE_V1_NO_RANK ||
      record.status != SAGR_CCL_V1_STATUS_SUCCESS ||
      record.context_sequence != 0U || record.peer_mask != expected_mask ||
      record.descriptor_count != group->world_size - 1U ||
      received_count != record.descriptor_count ||
      !sagr_ccl_v1_group_identity_equal(&record.group, group)) {
    status = SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    goto fail;
  }
  received_count = 0U;
  for (peer = 0U; peer < group->world_size; ++peer) {
    if (peer == self_rank) {
      continue;
    }
    status = validate_peer_identity(received_fds[received_count],
                                    expected_broker);
    if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
      goto fail;
    }
    created->peers[peer] = received_fds[received_count];
    received_fds[received_count] = -1;
    ++received_count;
  }
  initialize_record(&record, LIVE_MESSAGE_TABLE_ACK, group, self_rank);
  status = send_until(created->control, &record, NULL, 0U,
                      absolute_deadline_ns);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    goto fail;
  }
  received_count = 0U;
  status = rank_receive(created, &record, received_fds,
                        SAGR_CCL_V1_MAX_WORLD_SIZE, &received_count,
                        absolute_deadline_ns, 1);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    goto fail;
  }
  if (record.kind == LIVE_MESSAGE_ABORT) {
    sagr_ccl_live_v1_abort_t value;
    status = abort_from_record(&record, group, &value);
    if (status == SAGR_CCL_V1_STATUS_SUCCESS) {
      rank_store_abort(created, &value);
      status = value.status;
    }
    goto fail;
  }
  if (record.kind != LIVE_MESSAGE_READY ||
      record.claimed_rank != self_rank ||
      record.failed_rank != SAGR_CCL_LIVE_V1_NO_RANK ||
      record.status != SAGR_CCL_V1_STATUS_SUCCESS ||
      record.descriptor_count != 0U || received_count != 0U ||
      record.peer_mask != 0U || record.context_sequence != 0U ||
      !sagr_ccl_v1_group_identity_equal(&record.group, group)) {
    status = SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    goto fail;
  }
  created->phase = SAGR_CCL_LIVE_V1_PHASE_READY;
  *rank = created;
  return SAGR_CCL_V1_STATUS_SUCCESS;

fail:
  close_descriptors(received_fds, SAGR_CCL_V1_MAX_WORLD_SIZE);
  if (created != NULL && !created->has_first_error &&
      valid_abort_status(status)) {
    rank_report_join_failure(created, status);
  }
  rank_close_owned(created);
  if (created != NULL) {
    created->magic = 0U;
    free(created);
  }
  return status;
}

sagr_ccl_v1_status_t sagr_ccl_live_v1_rank_info(
    sagr_ccl_live_v1_rank_t rank, sagr_ccl_live_v1_rank_info_t *info,
    uint32_t info_size) {
  uint32_t peer;
  if (info == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (info_size < sizeof(*info)) {
    if (info_size >= sizeof(info->struct_size)) {
      info->struct_size = (uint32_t)sizeof(*info);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  if (!validate_rank(rank)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  info->phase = rank->phase;
  info->self_rank = rank->self_rank;
  info->world_size = rank->world_size;
  info->control_socket = rank->control;
  info->group = rank->group;
  for (peer = 0U; peer < SAGR_CCL_V1_MAX_WORLD_SIZE; ++peer) {
    info->peer_sockets[peer] = rank->peers[peer];
  }
  return SAGR_CCL_V1_STATUS_SUCCESS;
}

sagr_ccl_v1_status_t sagr_ccl_live_v1_rank_report_abort(
    sagr_ccl_live_v1_rank_t rank, uint32_t failed_rank,
    sagr_ccl_v1_status_t reason, uint64_t context_sequence) {
  sagr_ccl_live_v1_abort_t value;
  live_control_record_t record;
  if (!validate_rank(rank) ||
      (rank->phase != SAGR_CCL_LIVE_V1_PHASE_READY &&
       rank->phase != SAGR_CCL_LIVE_V1_PHASE_CLOSING) ||
      !valid_abort_status(reason) ||
      (failed_rank != SAGR_CCL_LIVE_V1_NO_RANK &&
       failed_rank >= rank->world_size)) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  make_abort(&value, &rank->group, rank->self_rank, failed_rank, reason,
             context_sequence);
  record = record_from_abort(&value);
  return send_control(rank->control, &record, NULL, 0U);
}

sagr_ccl_v1_status_t sagr_ccl_live_v1_rank_poll_abort(
    sagr_ccl_live_v1_rank_t rank, sagr_ccl_live_v1_abort_t *first_error,
    uint32_t first_error_size) {
  live_control_record_t record;
  int descriptors[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint32_t descriptor_count = 0U;
  sagr_ccl_v1_status_t status;
  if (!validate_rank(rank) || first_error == NULL) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (first_error_size < sizeof(*first_error)) {
    if (first_error_size >= sizeof(first_error->struct_size)) {
      first_error->struct_size = (uint32_t)sizeof(*first_error);
    }
    return SAGR_CCL_V1_STATUS_BUFFER_TOO_SMALL;
  }
  if (rank->has_first_error) {
    *first_error = rank->first_error;
    return rank->first_error.status;
  }
  status = rank_receive(rank, &record, descriptors,
                        SAGR_CCL_V1_MAX_WORLD_SIZE, &descriptor_count, 0U, 0);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  if (descriptor_count != 0U || record.kind != LIVE_MESSAGE_ABORT) {
    close_descriptors(descriptors, descriptor_count);
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
  status = abort_from_record(&record, &rank->group, first_error);
  if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
    return status;
  }
  rank_store_abort(rank, first_error);
  return first_error->status;
}

sagr_ccl_v1_status_t sagr_ccl_live_v1_rank_close(
    sagr_ccl_live_v1_rank_t rank, uint64_t absolute_deadline_ns) {
  live_control_record_t record;
  int descriptors[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint32_t descriptor_count;
  sagr_ccl_v1_status_t status;
  if (!validate_rank(rank) || absolute_deadline_ns == 0U) {
    return SAGR_CCL_V1_STATUS_INVALID_ARGUMENT;
  }
  if (rank->has_first_error) {
    return rank->first_error.status;
  }
  if (rank->phase == SAGR_CCL_LIVE_V1_PHASE_CLOSED) {
    return SAGR_CCL_V1_STATUS_CLOSED;
  }
  if (rank->phase != SAGR_CCL_LIVE_V1_PHASE_READY &&
      rank->phase != SAGR_CCL_LIVE_V1_PHASE_CLOSING) {
    return SAGR_CCL_V1_STATUS_OUT_OF_ORDER;
  }
  if (!rank->leave_sent) {
    initialize_record(&record, LIVE_MESSAGE_LEAVE, &rank->group,
                      rank->self_rank);
    status = send_until(rank->control, &record, NULL, 0U,
                        absolute_deadline_ns);
    if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
      return status;
    }
    rank->leave_sent = 1U;
    rank->phase = SAGR_CCL_LIVE_V1_PHASE_CLOSING;
  }
  for (;;) {
    sagr_ccl_live_v1_abort_t value;
    descriptor_count = 0U;
    status = rank_receive(rank, &record, descriptors,
                          SAGR_CCL_V1_MAX_WORLD_SIZE, &descriptor_count,
                          absolute_deadline_ns, 1);
    if (status == SAGR_CCL_V1_STATUS_TIMED_OUT) {
      (void)sagr_ccl_live_v1_rank_report_abort(
          rank, rank->self_rank, SAGR_CCL_V1_STATUS_TIMED_OUT, 0U);
      return status;
    }
    if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
      return status;
    }
    if (descriptor_count != 0U) {
      close_descriptors(descriptors, descriptor_count);
      return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
    }
    if (record.kind == LIVE_MESSAGE_ABORT) {
      status = abort_from_record(&record, &rank->group, &value);
      if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
        return status;
      }
      rank_store_abort(rank, &value);
      return value.status;
    }
    if (record.kind == LIVE_MESSAGE_CLOSED &&
        record.claimed_rank == SAGR_CCL_LIVE_V1_NO_RANK &&
        record.failed_rank == SAGR_CCL_LIVE_V1_NO_RANK &&
        record.status == SAGR_CCL_V1_STATUS_SUCCESS &&
        record.descriptor_count == 0U && record.peer_mask == 0U &&
        record.context_sequence == 0U &&
        sagr_ccl_v1_group_identity_equal(&record.group, &rank->group)) {
      rank->phase = SAGR_CCL_LIVE_V1_PHASE_CLOSED;
      return SAGR_CCL_V1_STATUS_SUCCESS;
    }
    return SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  }
}

void sagr_ccl_live_v1_rank_destroy(sagr_ccl_live_v1_rank_t *rank) {
  sagr_ccl_live_v1_rank_t value;
  if (rank == NULL || *rank == NULL) {
    return;
  }
  value = *rank;
  if (value->magic != LIVE_RANK_MAGIC) {
    *rank = NULL;
    return;
  }
  rank_close_owned(value);
  value->phase = SAGR_CCL_LIVE_V1_PHASE_CLOSED;
  value->magic = 0U;
  free(value);
  *rank = NULL;
}
