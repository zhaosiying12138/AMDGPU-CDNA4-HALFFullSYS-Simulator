/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <self_amdgpu_runtime/ccl_carrier_v1.h>
#include <self_amdgpu_runtime/ccl_live_v1.h>

#include "sha256_internal.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define REQUIRE(expression)                                                    \
  do {                                                                         \
    if (!(expression)) {                                                       \
      fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, \
              #expression);                                                    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHILD_CHECK(expression)                                                \
  do {                                                                         \
    if (!(expression)) {                                                       \
      fprintf(stderr, "rank child %s:%d: requirement failed: %s\n", __FILE__, \
              __LINE__, #expression);                                          \
      result = 1;                                                              \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

#define TEST_CHECK(expression)                                                 \
  do {                                                                         \
    if (!(expression)) {                                                       \
      fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, \
              #expression);                                                    \
      result = 1;                                                              \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

#define REPORT_MAGIC UINT32_C(0x43434c4d)
#define COLLECTIVE_DEADLINE_NS UINT64_C(10000000000)
#define FAULT_DEADLINE_NS UINT64_C(5000000000)
#define CHILD_WAIT_NS UINT64_C(5000000000)
#define INJECTED_WAIT_NS UINT64_C(50000000)
#define MOCK_SEQUENCE UINT64_C(1)
#define WAIT_OUTCOME_NONZERO UINT32_C(1)
#define WAIT_OUTCOME_TIMEOUT UINT32_C(2)
#define WAIT_OUTCOME_ERROR UINT32_C(4)

enum child_mode {
  CHILD_MODE_COLLECTIVE = 1,
  CHILD_MODE_GROUP_ABORT = 2,
  CHILD_MODE_PEER_LOSS = 3,
  CHILD_MODE_INJECTED_EXIT = 4,
  CHILD_MODE_INJECTED_HANG = 5
};

typedef struct child_report {
  uint32_t magic;
  uint32_t mode;
  uint32_t world_size;
  uint32_t rank;
  uint64_t element_count;
  int32_t status;
  uint32_t data_count;
  uint32_t consumed_count;
  uint32_t ordered_stage_count;
  uint32_t zero_chunk_send_count;
  uint32_t host_mock_reduction_count;
  uint32_t device_sum_count;
  uint32_t credit_busy_observed;
  uint32_t out_of_order_observed;
  uint32_t deadline_ok;
  uint32_t fd_cleanup_ok;
  uint32_t abort_reporter_rank;
  uint32_t abort_failed_rank;
  uint64_t abort_sequence;
  uint8_t result_sha256[SAGR_CCL_V1_SHA256_BYTES];
} child_report_t;

typedef struct child_process {
  pid_t pid;
  uint32_t rank;
  int report_descriptor;
} child_process_t;

static void initialize_children(child_process_t *children, uint32_t count) {
  uint32_t index;
  for (index = 0U; index < count; ++index) {
    children[index].pid = (pid_t)-1;
    children[index].rank = index;
    children[index].report_descriptor = -1;
  }
}

static int close_owned_descriptor(int *descriptor) {
  int owned;
  if (descriptor == NULL || *descriptor < 0) {
    return 0;
  }
  owned = *descriptor;
  *descriptor = -1;
  return close(owned);
}

static uint64_t deadline_after(uint64_t nanoseconds) {
  uint64_t now = sagr_ccl_live_v1_monotonic_time_ns();
  return now == 0U || UINT64_MAX - now < nanoseconds ? 0U : now + nanoseconds;
}

static int set_cloexec(int descriptor, int enabled) {
  int flags = fcntl(descriptor, F_GETFD);
  if (flags < 0) {
    return -1;
  }
  flags = enabled ? flags | FD_CLOEXEC : flags & ~FD_CLOEXEC;
  return fcntl(descriptor, F_SETFD, flags);
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

static int wait_descriptor(int descriptor, short events, uint64_t deadline) {
  struct pollfd item;
  for (;;) {
    uint64_t now = sagr_ccl_live_v1_monotonic_time_ns();
    uint64_t remaining;
    uint64_t milliseconds;
    int timeout;
    int result;
    if (now == 0U || now >= deadline) {
      return -1;
    }
    remaining = deadline - now;
    milliseconds = (remaining + UINT64_C(999999)) / UINT64_C(1000000);
    timeout = milliseconds > (uint64_t)INT_MAX ? INT_MAX : (int)milliseconds;
    item.fd = descriptor;
    item.events = events;
    item.revents = 0;
    do {
      result = poll(&item, 1U, timeout);
    } while (result < 0 && errno == EINTR);
    if (result <= 0) {
      return -1;
    }
    if ((item.revents & events) != 0) {
      return 0;
    }
    if ((item.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return -1;
    }
  }
}

static int write_all(int descriptor, const void *bytes, size_t count,
                     uint64_t deadline) {
  const uint8_t *cursor = (const uint8_t *)bytes;
  size_t written = 0U;
  while (written < count) {
    ssize_t value = write(descriptor, cursor + written, count - written);
    if (value > 0) {
      written += (size_t)value;
      continue;
    }
    if (value < 0 && errno == EINTR) {
      continue;
    }
    if (value < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) &&
        wait_descriptor(descriptor, POLLOUT, deadline) == 0) {
      continue;
    }
    return -1;
  }
  return 0;
}

static int read_all(int descriptor, void *bytes, size_t count,
                    uint64_t deadline) {
  uint8_t *cursor = (uint8_t *)bytes;
  size_t received = 0U;
  while (received < count) {
    ssize_t value;
    if (wait_descriptor(descriptor, POLLIN, deadline) != 0) {
      return -1;
    }
    do {
      value = read(descriptor, cursor + received, count - received);
    } while (value < 0 && errno == EINTR);
    if (value <= 0) {
      return -1;
    }
    received += (size_t)value;
  }
  return 0;
}

static int parse_u32(const char *text, uint32_t *value) {
  char *end = NULL;
  unsigned long parsed;
  if (text == NULL || value == NULL) {
    return -1;
  }
  errno = 0;
  parsed = strtoul(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed > UINT32_MAX) {
    return -1;
  }
  *value = (uint32_t)parsed;
  return 0;
}

static int parse_u64(const char *text, uint64_t *value) {
  char *end = NULL;
  unsigned long long parsed;
  if (text == NULL || value == NULL) {
    return -1;
  }
  errno = 0;
  parsed = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    return -1;
  }
  *value = (uint64_t)parsed;
  return 0;
}

static int parse_i32(const char *text, int32_t *value) {
  char *end = NULL;
  long parsed;
  if (text == NULL || value == NULL) {
    return -1;
  }
  errno = 0;
  parsed = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed < INT32_MIN ||
      parsed > INT32_MAX) {
    return -1;
  }
  *value = (int32_t)parsed;
  return 0;
}

static sagr_ccl_v1_group_identity_t make_identity(uint32_t world_size,
                                                   uint32_t tag) {
  sagr_ccl_v1_group_identity_t identity;
  uint32_t index;
  (void)sagr_ccl_v1_group_identity_init(&identity,
                                        (uint32_t)sizeof(identity));
  identity.world_size = world_size;
  identity.epoch = UINT64_C(0x43434c1000000000) + tag;
  identity.group_generation = UINT64_C(0x43434c2000000000) + tag;
  for (index = 0U; index < SAGR_CCL_V1_UUID_BYTES; ++index) {
    identity.job_uuid[index] = (uint8_t)(tag + index + 3U);
    identity.group_uuid[index] = (uint8_t)(tag * 3U + index + 11U);
  }
  for (index = 0U; index < SAGR_CCL_V1_SHA256_BYTES; ++index) {
    identity.model_identity_sha256[index] =
        (uint8_t)(tag * 5U + index + 29U);
  }
  return identity;
}

static sagr_ccl_v1_descriptor_t make_all_reduce_descriptor(
    const sagr_ccl_v1_group_identity_t *identity, uint32_t rank,
    uint64_t element_count, uint64_t sequence) {
  sagr_ccl_v1_descriptor_t descriptor;
  (void)sagr_ccl_v1_descriptor_init(&descriptor,
                                    (uint32_t)sizeof(descriptor));
  descriptor.group = *identity;
  descriptor.sequence = sequence;
  descriptor.input_count = element_count;
  descriptor.output_count = element_count;
  descriptor.rank = rank;
  descriptor.operation = SAGR_CCL_V1_OPERATION_ALL_REDUCE;
  descriptor.reduction = SAGR_CCL_V1_REDUCTION_SUM;
  descriptor.dtype = SAGR_CCL_V1_DTYPE_FP32;
  descriptor.root_rank = SAGR_CCL_V1_NO_RANK;
  return descriptor;
}

static uint32_t mock_rank_value(uint32_t rank, uint64_t element) {
  return (rank + UINT32_C(1)) * UINT32_C(257) +
         ((uint32_t)element + UINT32_C(1)) * UINT32_C(3);
}

static uint32_t mock_expected_value(uint32_t world, uint64_t element) {
  uint32_t rank;
  uint32_t value = 0U;
  for (rank = 0U; rank < world; ++rank) {
    value += mock_rank_value(rank, element);
  }
  return value;
}

static void hash_words(const uint32_t *words, uint64_t count,
                       uint8_t digest[SAGR_CCL_V1_SHA256_BYTES]) {
  sagr_sha256_context_t context;
  uint64_t index;
  sagr_sha256_init(&context);
  for (index = 0U; index < count; ++index) {
    uint8_t encoded[4];
    encoded[0] = (uint8_t)(words[index] >> 24U);
    encoded[1] = (uint8_t)(words[index] >> 16U);
    encoded[2] = (uint8_t)(words[index] >> 8U);
    encoded[3] = (uint8_t)words[index];
    sagr_sha256_update(&context, encoded, sizeof(encoded));
  }
  sagr_sha256_final(&context, digest);
}

static int validate_peer_table(const sagr_ccl_live_v1_rank_info_t *info,
                               const sagr_ccl_v1_group_identity_t *identity) {
  uint32_t peer;
  if (info->phase != SAGR_CCL_LIVE_V1_PHASE_READY ||
      info->self_rank >= info->world_size || info->control_socket < 0 ||
      !sagr_ccl_v1_group_identity_equal(&info->group, identity)) {
    return -1;
  }
  for (peer = 0U; peer < SAGR_CCL_V1_MAX_WORLD_SIZE; ++peer) {
    if (peer == info->self_rank || peer >= info->world_size) {
      if (info->peer_sockets[peer] != -1) {
        return -1;
      }
    } else {
      int type = 0;
      socklen_t size = (socklen_t)sizeof(type);
      if (info->peer_sockets[peer] < 0 ||
          getsockopt(info->peer_sockets[peer], SOL_SOCKET, SO_TYPE, &type,
                     &size) != 0 ||
          type != SOCK_SEQPACKET ||
          (fcntl(info->peer_sockets[peer], F_GETFD) & FD_CLOEXEC) == 0 ||
          (fcntl(info->peer_sockets[peer], F_GETFL) & O_NONBLOCK) == 0) {
        return -1;
      }
    }
  }
  return 0;
}

static sagr_ccl_v1_status_t send_data_until(
    sagr_ccl_v1_carrier_session_t session, int descriptor,
    const sagr_ccl_v1_carrier_record_t *record, uint64_t deadline) {
  for (;;) {
    sagr_ccl_v1_status_t status =
        sagr_ccl_v1_carrier_session_send_data(session, descriptor, record);
    if (status != SAGR_CCL_V1_STATUS_BUSY) {
      return status;
    }
    if (wait_descriptor(descriptor, POLLOUT, deadline) != 0) {
      return SAGR_CCL_V1_STATUS_TIMED_OUT;
    }
  }
}

static sagr_ccl_v1_status_t send_consumed_until(
    sagr_ccl_v1_carrier_session_t session, int descriptor,
    const sagr_ccl_v1_carrier_record_t *record, uint64_t deadline) {
  for (;;) {
    sagr_ccl_v1_status_t status = sagr_ccl_v1_carrier_session_send_consumed(
        session, descriptor, record);
    if (status != SAGR_CCL_V1_STATUS_BUSY) {
      return status;
    }
    if (wait_descriptor(descriptor, POLLOUT, deadline) != 0) {
      return SAGR_CCL_V1_STATUS_TIMED_OUT;
    }
  }
}

static sagr_ccl_v1_status_t receive_until(
    sagr_ccl_v1_carrier_session_t session, int descriptor,
    const sagr_ccl_v1_descriptor_t *collective, uint32_t step_index,
    uint32_t peer_rank, sagr_ccl_v1_carrier_record_t *record,
    uint64_t deadline) {
  for (;;) {
    sagr_ccl_v1_status_t status = sagr_ccl_v1_carrier_session_receive(
        session, descriptor, collective, step_index, peer_rank, record,
        (uint32_t)sizeof(*record));
    if (status != SAGR_CCL_V1_STATUS_BUSY) {
      return status;
    }
    if (wait_descriptor(descriptor, POLLIN, deadline) != 0) {
      return SAGR_CCL_V1_STATUS_TIMED_OUT;
    }
  }
}

static sagr_ccl_v1_status_t raw_send_until(
    int descriptor, const sagr_ccl_v1_carrier_record_t *record,
    int payload_descriptor, uint64_t deadline) {
  for (;;) {
    sagr_ccl_v1_status_t status =
        sagr_ccl_v1_carrier_send(descriptor, record, payload_descriptor);
    if (status != SAGR_CCL_V1_STATUS_BUSY) {
      return status;
    }
    if (wait_descriptor(descriptor, POLLOUT, deadline) != 0) {
      return SAGR_CCL_V1_STATUS_TIMED_OUT;
    }
  }
}

/*
 * The world-2 preflight uses the live broker-provided FD, not a local
 * socketpair. It deliberately exhausts two credits, reverses DATA order,
 * reverses ACK order, and finally replays a released generation.
 */
static int run_live_credit_probe(const sagr_ccl_live_v1_rank_info_t *info,
                                 child_report_t *report,
                                 uint64_t deadline) {
  sagr_ccl_v1_carrier_session_t session = NULL;
  sagr_ccl_v1_descriptor_t first = make_all_reduce_descriptor(
      &info->group, info->self_rank, 2U, UINT64_C(101));
  sagr_ccl_v1_descriptor_t second = first;
  sagr_ccl_v1_descriptor_t third = first;
  sagr_ccl_v1_carrier_record_t first_data;
  sagr_ccl_v1_carrier_record_t second_data;
  sagr_ccl_v1_carrier_record_t ignored_data;
  sagr_ccl_v1_carrier_record_t first_received;
  sagr_ccl_v1_carrier_record_t second_received;
  sagr_ccl_v1_carrier_record_t first_ack;
  sagr_ccl_v1_carrier_record_t second_ack;
  sagr_ccl_v1_carrier_record_t received_ack;
  sagr_ccl_v1_carrier_session_info_t session_info;
  uint32_t first_payload = UINT32_C(0x10203040);
  uint32_t second_payload = UINT32_C(0x50607080);
  uint32_t staging = 0U;
  int replay_descriptor = -1;
  uint32_t replay_crc = 0U;
  int result = 0;
  second.sequence = UINT64_C(102);
  third.sequence = UINT64_C(103);
  CHILD_CHECK(info->world_size == 2U);
  CHILD_CHECK(sagr_ccl_v1_carrier_session_create(
                  &info->group, info->self_rank, 2U, &session) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
  if (info->self_rank == 0U) {
    CHILD_CHECK(sagr_ccl_v1_carrier_session_prepare_data(
                    session, &first, 0U, &first_payload,
                    (uint64_t)sizeof(first_payload), &first_data,
                    (uint32_t)sizeof(first_data)) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(sagr_ccl_v1_carrier_session_prepare_data(
                    session, &second, 0U, &second_payload,
                    (uint64_t)sizeof(second_payload), &second_data,
                    (uint32_t)sizeof(second_data)) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(first_data.slot_index != second_data.slot_index);
    CHILD_CHECK(sagr_ccl_v1_carrier_session_prepare_data(
                    session, &third, 0U, &first_payload,
                    (uint64_t)sizeof(first_payload), &ignored_data,
                    (uint32_t)sizeof(ignored_data)) ==
                SAGR_CCL_V1_STATUS_BUSY);
    report->credit_busy_observed = 1U;
    CHILD_CHECK(send_data_until(session, info->peer_sockets[1], &second_data,
                                deadline) == SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(send_data_until(session, info->peer_sockets[1], &first_data,
                                deadline) == SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(receive_until(session, info->peer_sockets[1], &first, 0U, 1U,
                              &received_ack, deadline) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(received_ack.kind == SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED);
    CHILD_CHECK(receive_until(session, info->peer_sockets[1], &second, 0U, 1U,
                              &received_ack, deadline) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(received_ack.kind == SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED);
    CHILD_CHECK(sagr_ccl_v1_carrier_payload_create(
                    &first_payload, (uint64_t)sizeof(first_payload),
                    &replay_descriptor, &replay_crc) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(replay_crc == first_data.payload_crc32c);
    CHILD_CHECK(raw_send_until(info->peer_sockets[1], &first_data,
                               replay_descriptor, deadline) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(close(replay_descriptor) == 0);
    replay_descriptor = -1;
  } else {
    CHILD_CHECK(receive_until(session, info->peer_sockets[0], &second, 0U, 0U,
                              &second_received, deadline) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(receive_until(session, info->peer_sockets[0], &first, 0U, 0U,
                              &first_received, deadline) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(sagr_ccl_v1_carrier_session_consume(
                    session, &second, 0U, &second_received, &staging,
                    (uint64_t)sizeof(staging), &second_ack,
                    (uint32_t)sizeof(second_ack)) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(staging == second_payload);
    CHILD_CHECK(sagr_ccl_v1_carrier_session_consume(
                    session, &first, 0U, &first_received, &staging,
                    (uint64_t)sizeof(staging), &first_ack,
                    (uint32_t)sizeof(first_ack)) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(staging == first_payload);
    CHILD_CHECK(send_consumed_until(session, info->peer_sockets[0], &first_ack,
                                    deadline) == SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(send_consumed_until(session, info->peer_sockets[0], &second_ack,
                                    deadline) == SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(receive_until(session, info->peer_sockets[0], &first, 0U, 0U,
                              &first_received, deadline) ==
                SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
    CHILD_CHECK(sagr_ccl_v1_carrier_session_info(
                    session, &session_info,
                    (uint32_t)sizeof(session_info)) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(session_info.phase ==
                    SAGR_CCL_V1_CARRIER_SESSION_ABORTED &&
                session_info.first_error == SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
    report->out_of_order_observed = 1U;
  }

cleanup:
  if (replay_descriptor >= 0) {
    (void)close(replay_descriptor);
  }
  sagr_ccl_v1_carrier_session_destroy(&session);
  return result;
}

static int run_mock_collective(sagr_ccl_live_v1_rank_t live_rank,
                               uint64_t element_count,
                               child_report_t *report) {
  sagr_ccl_live_v1_rank_info_t info;
  sagr_ccl_v1_descriptor_t descriptor;
  sagr_ccl_v1_plan_step_t steps[SAGR_CCL_V1_MAX_PLAN_STEPS];
  sagr_ccl_v1_carrier_session_t session = NULL;
  sagr_ccl_v1_carrier_session_info_t session_info;
  uint32_t *workspace = NULL;
  uint32_t *staging = NULL;
  uint32_t step_count = 0U;
  uint32_t step_index;
  uint64_t element;
  uint64_t deadline = deadline_after(COLLECTIVE_DEADLINE_NS);
  int result = 0;
  CHILD_CHECK(deadline != 0U);
  CHILD_CHECK(element_count <= (uint64_t)SIZE_MAX / sizeof(uint32_t));
  CHILD_CHECK(sagr_ccl_live_v1_rank_info(live_rank, &info,
                                        (uint32_t)sizeof(info)) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
  CHILD_CHECK(validate_peer_table(&info, &info.group) == 0);
  descriptor = make_all_reduce_descriptor(&info.group, info.self_rank,
                                          element_count, MOCK_SEQUENCE);
  CHILD_CHECK(sagr_ccl_v1_descriptor_validate(&descriptor) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
  CHILD_CHECK(sagr_ccl_v1_plan_rank(&descriptor, steps,
                                    SAGR_CCL_V1_MAX_PLAN_STEPS,
                                    &step_count) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
  CHILD_CHECK(step_count == 2U * (info.world_size - 1U));
  workspace = (uint32_t *)calloc(element_count == 0U ? 1U : (size_t)element_count,
                                 sizeof(*workspace));
  staging = (uint32_t *)calloc(element_count == 0U ? 1U : (size_t)element_count,
                               sizeof(*staging));
  CHILD_CHECK(workspace != NULL && staging != NULL);
  for (element = 0U; element < element_count; ++element) {
    workspace[element] = mock_rank_value(info.self_rank, element);
  }
  if (info.world_size == 2U) {
    CHILD_CHECK(run_live_credit_probe(&info, report, deadline) == 0);
  }
  CHILD_CHECK(sagr_ccl_v1_carrier_session_create(
                  &info.group, info.self_rank, 2U, &session) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
  for (step_index = 0U; step_index < step_count; ++step_index) {
    const sagr_ccl_v1_plan_step_t *step = &steps[step_index];
    sagr_ccl_v1_carrier_record_t data;
    sagr_ccl_v1_carrier_record_t received;
    sagr_ccl_v1_carrier_record_t consumed;
    sagr_ccl_v1_carrier_record_t acknowledged;
    uint64_t send_bytes;
    uint64_t receive_bytes;
    uint32_t *send_pointer;
    uint32_t *staging_pointer;
    uint64_t index;
    CHILD_CHECK(step->action == SAGR_CCL_V1_PLAN_ACTION_SEND_RECEIVE);
    CHILD_CHECK(step->send_rank < info.world_size &&
                step->receive_rank < info.world_size);
    CHILD_CHECK(step->send_offset_elements <= element_count &&
                step->send_count_elements <=
                    element_count - step->send_offset_elements);
    CHILD_CHECK(step->receive_offset_elements <= element_count &&
                step->receive_count_elements <=
                    element_count - step->receive_offset_elements);
    send_bytes = step->send_count_elements * (uint64_t)sizeof(uint32_t);
    receive_bytes = step->receive_count_elements * (uint64_t)sizeof(uint32_t);
    send_pointer = send_bytes == 0U
                       ? NULL
                       : workspace + (size_t)step->send_offset_elements;
    CHILD_CHECK(sagr_ccl_v1_carrier_session_prepare_data(
                    session, &descriptor, step_index, send_pointer, send_bytes,
                    &data, (uint32_t)sizeof(data)) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(data.payload_bytes == send_bytes &&
                data.kind == SAGR_CCL_V1_CARRIER_MESSAGE_DATA);
    CHILD_CHECK(send_data_until(session, info.peer_sockets[step->send_rank],
                                &data, deadline) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    ++report->data_count;
    if (send_bytes == 0U) {
      ++report->zero_chunk_send_count;
    }
    CHILD_CHECK(receive_until(session,
                              info.peer_sockets[step->receive_rank],
                              &descriptor, step_index, step->receive_rank,
                              &received, deadline) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(received.kind == SAGR_CCL_V1_CARRIER_MESSAGE_DATA &&
                received.payload_bytes == receive_bytes);
    staging_pointer = receive_bytes == 0U ? NULL : staging;
    CHILD_CHECK(sagr_ccl_v1_carrier_session_consume(
                    session, &descriptor, step_index, &received,
                    staging_pointer, receive_bytes, &consumed,
                    (uint32_t)sizeof(consumed)) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(sagr_ccl_v1_carrier_session_info(
                    session, &session_info,
                    (uint32_t)sizeof(session_info)) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(session_info.receiver_consumed >= 1U);
    ++report->ordered_stage_count;

    if (step->phase == SAGR_CCL_V1_PLAN_PHASE_REDUCE_SCATTER) {
      if (step->receive_count_elements != 0U) {
        for (index = 0U; index < step->receive_count_elements; ++index) {
          workspace[step->receive_offset_elements + index] += staging[index];
        }
        ++report->host_mock_reduction_count;
      }
    } else {
      CHILD_CHECK(step->phase == SAGR_CCL_V1_PLAN_PHASE_ALL_GATHER);
      for (index = 0U; index < step->receive_count_elements; ++index) {
        workspace[step->receive_offset_elements + index] = staging[index];
      }
    }

    CHILD_CHECK(send_consumed_until(
                    session, info.peer_sockets[step->receive_rank], &consumed,
                    deadline) == SAGR_CCL_V1_STATUS_SUCCESS);
    ++report->consumed_count;
    CHILD_CHECK(receive_until(session, info.peer_sockets[step->send_rank],
                              &descriptor, step_index, step->send_rank,
                              &acknowledged, deadline) ==
                SAGR_CCL_V1_STATUS_SUCCESS);
    CHILD_CHECK(acknowledged.kind ==
                SAGR_CCL_V1_CARRIER_MESSAGE_CONSUMED);
  }
  CHILD_CHECK(sagr_ccl_v1_carrier_session_info(
                  session, &session_info, (uint32_t)sizeof(session_info)) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
  CHILD_CHECK(session_info.phase == SAGR_CCL_V1_CARRIER_SESSION_RUNNING &&
              session_info.sender_inflight == 0U &&
              session_info.receiver_ready == 0U &&
              session_info.receiver_consumed == 0U);
  for (element = 0U; element < element_count; ++element) {
    CHILD_CHECK(workspace[element] ==
                mock_expected_value(info.world_size, element));
  }
  hash_words(workspace, element_count, report->result_sha256);
  report->status = SAGR_CCL_V1_STATUS_SUCCESS;
  report->device_sum_count = 0U;

cleanup:
  sagr_ccl_v1_carrier_session_destroy(&session);
  free(staging);
  free(workspace);
  return result;
}

static sagr_ccl_v1_status_t wait_live_abort(
    sagr_ccl_live_v1_rank_t rank, sagr_ccl_live_v1_abort_t *value,
    uint64_t deadline) {
  for (;;) {
    sagr_ccl_v1_status_t status = sagr_ccl_live_v1_rank_poll_abort(
        rank, value, (uint32_t)sizeof(*value));
    if (status != SAGR_CCL_V1_STATUS_BUSY) {
      return status;
    }
    if (sagr_ccl_live_v1_monotonic_time_ns() >= deadline) {
      return SAGR_CCL_V1_STATUS_TIMED_OUT;
    }
    (void)usleep(1000U);
  }
}

static int report_live_abort_until(sagr_ccl_live_v1_rank_t rank,
                                   int control_descriptor,
                                   uint64_t deadline) {
  for (;;) {
    sagr_ccl_v1_status_t status = sagr_ccl_live_v1_rank_report_abort(
        rank, 1U, SAGR_CCL_V1_STATUS_CHECKSUM_ERROR, UINT64_C(77));
    if (status == SAGR_CCL_V1_STATUS_SUCCESS) {
      return 0;
    }
    if (status != SAGR_CCL_V1_STATUS_BUSY ||
        wait_descriptor(control_descriptor, POLLOUT, deadline) != 0) {
      return -1;
    }
  }
}

static int run_fault_child(sagr_ccl_live_v1_rank_t *rank, uint32_t mode,
                           child_report_t *report) {
  sagr_ccl_live_v1_rank_info_t info;
  sagr_ccl_live_v1_abort_t abort_value;
  sagr_ccl_v1_status_t status;
  uint64_t start = sagr_ccl_live_v1_monotonic_time_ns();
  uint64_t deadline = deadline_after(FAULT_DEADLINE_NS);
  int result = 0;
  CHILD_CHECK(start != 0U && deadline != 0U);
  CHILD_CHECK(sagr_ccl_live_v1_rank_info(*rank, &info,
                                        (uint32_t)sizeof(info)) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
  CHILD_CHECK(validate_peer_table(&info, &info.group) == 0);
  if (mode == CHILD_MODE_GROUP_ABORT) {
    if (info.self_rank == 0U) {
      CHILD_CHECK(report_live_abort_until(*rank, info.control_socket,
                                          deadline) == 0);
    }
    status = wait_live_abort(*rank, &abort_value, deadline);
    CHILD_CHECK(status == SAGR_CCL_V1_STATUS_CHECKSUM_ERROR);
    CHILD_CHECK(abort_value.reporter_rank == 0U &&
                abort_value.failed_rank == 1U &&
                abort_value.context_sequence == UINT64_C(77));
    report->abort_reporter_rank = abort_value.reporter_rank;
    report->abort_failed_rank = abort_value.failed_rank;
    report->abort_sequence = abort_value.context_sequence;
  } else {
    CHILD_CHECK(mode == CHILD_MODE_PEER_LOSS);
    if (info.self_rank == 0U) {
      sagr_ccl_live_v1_rank_destroy(rank);
      status = SAGR_CCL_V1_STATUS_PEER_LOST;
      report->abort_reporter_rank = SAGR_CCL_LIVE_V1_NO_RANK;
      report->abort_failed_rank = 0U;
      report->abort_sequence = 0U;
    } else {
      status = wait_live_abort(*rank, &abort_value, deadline);
      CHILD_CHECK(status == SAGR_CCL_V1_STATUS_PEER_LOST);
      CHILD_CHECK(abort_value.reporter_rank == SAGR_CCL_LIVE_V1_NO_RANK &&
                  abort_value.failed_rank == 0U &&
                  abort_value.context_sequence == 0U);
      report->abort_reporter_rank = abort_value.reporter_rank;
      report->abort_failed_rank = abort_value.failed_rank;
      report->abort_sequence = abort_value.context_sequence;
    }
  }
  report->status = status;
  report->deadline_ok =
      sagr_ccl_live_v1_monotonic_time_ns() < deadline ? 1U : 0U;
  CHILD_CHECK(report->deadline_ok == 1U);

cleanup:
  return result;
}

static int child_main(int argc, char **argv) {
  const char *mode_text;
  int32_t capability;
  int32_t report_descriptor;
  uint32_t world;
  uint32_t rank_index;
  uint64_t element_count;
  uint32_t tag;
  int32_t owner_pid;
  uint32_t owner_uid;
  uint32_t owner_gid;
  uint64_t owner_start;
  uint32_t mode;
  sagr_ccl_v1_group_identity_t identity;
  sagr_ccl_live_v1_process_identity_t owner;
  sagr_ccl_live_v1_rank_t rank = NULL;
  child_report_t report;
  sagr_ccl_v1_status_t join_status;
  int descriptors_before;
  int result;
  if (argc != 13 || strcmp(argv[1], "--child") != 0) {
    return 90;
  }
  mode_text = argv[2];
  mode = strcmp(mode_text, "collective") == 0
             ? CHILD_MODE_COLLECTIVE
             : strcmp(mode_text, "group-abort") == 0
                   ? CHILD_MODE_GROUP_ABORT
                   : strcmp(mode_text, "peer-loss") == 0
                         ? CHILD_MODE_PEER_LOSS
                         : strcmp(mode_text, "injected-exit") == 0
                               ? CHILD_MODE_INJECTED_EXIT
                               : strcmp(mode_text, "injected-hang") == 0
                                     ? CHILD_MODE_INJECTED_HANG
                         : 0U;
  if (mode == 0U || parse_i32(argv[3], &capability) != 0 ||
      parse_i32(argv[4], &report_descriptor) != 0 ||
      parse_u32(argv[5], &world) != 0 ||
      parse_u32(argv[6], &rank_index) != 0 ||
      parse_u64(argv[7], &element_count) != 0 ||
      parse_u32(argv[8], &tag) != 0 || parse_i32(argv[9], &owner_pid) != 0 ||
      parse_u32(argv[10], &owner_uid) != 0 ||
      parse_u32(argv[11], &owner_gid) != 0 ||
      parse_u64(argv[12], &owner_start) != 0) {
    return 91;
  }
  if (mode == CHILD_MODE_INJECTED_EXIT) {
    return 94;
  }
  if (mode == CHILD_MODE_INJECTED_HANG) {
    for (;;) {
      (void)pause();
    }
  }
  descriptors_before = count_open_descriptors();
  if (descriptors_before < 0) {
    return 92;
  }
  memset(&report, 0, sizeof(report));
  report.magic = REPORT_MAGIC;
  report.mode = mode;
  report.world_size = world;
  report.rank = rank_index;
  report.element_count = element_count;
  report.status = SAGR_CCL_V1_STATUS_PROTOCOL_ERROR;
  report.abort_reporter_rank = SAGR_CCL_LIVE_V1_NO_RANK;
  report.abort_failed_rank = SAGR_CCL_LIVE_V1_NO_RANK;
  identity = make_identity(world, tag);
  memset(&owner, 0, sizeof(owner));
  owner.struct_size = (uint32_t)sizeof(owner);
  owner.pid = owner_pid;
  owner.uid = owner_uid;
  owner.gid = owner_gid;
  owner.start_time_ticks = owner_start;
  join_status = sagr_ccl_live_v1_rank_join(
      capability, &identity, rank_index, &owner,
      deadline_after(COLLECTIVE_DEADLINE_NS), &rank);
  if (join_status != SAGR_CCL_V1_STATUS_SUCCESS || rank == NULL) {
    report.status = join_status;
    result = 1;
  } else if (mode == CHILD_MODE_COLLECTIVE) {
    result = run_mock_collective(rank, element_count, &report);
    if (result == 0) {
      sagr_ccl_v1_status_t close_status = sagr_ccl_live_v1_rank_close(
          rank, deadline_after(COLLECTIVE_DEADLINE_NS));
      if (close_status != SAGR_CCL_V1_STATUS_SUCCESS) {
        report.status = close_status;
        result = 1;
      }
    }
  } else {
    result = run_fault_child(&rank, mode, &report);
  }
  sagr_ccl_live_v1_rank_destroy(&rank);
  report.fd_cleanup_ok =
      count_open_descriptors() == descriptors_before - 1 ? 1U : 0U;
  if (report.fd_cleanup_ok == 0U) {
    result = 1;
  }
  if (write_all(report_descriptor, &report, sizeof(report),
                deadline_after(CHILD_WAIT_NS)) != 0) {
    result = 1;
  }
  (void)close(report_descriptor);
  return result == 0 ? 0 : 93;
}

static int spawn_rank(sagr_ccl_live_v1_broker_t broker,
                      const sagr_ccl_live_v1_process_identity_t *owner,
                      uint32_t world, uint32_t rank, uint64_t element_count,
                      uint32_t tag, const char *mode,
                      child_process_t *child) {
  int capability = -1;
  int report_pipe[2] = {-1, -1};
  pid_t pid;
  char capability_text[32];
  char report_text[32];
  char world_text[32];
  char rank_text[32];
  char count_text[32];
  char tag_text[32];
  char owner_pid_text[32];
  char owner_uid_text[32];
  char owner_gid_text[32];
  char owner_start_text[32];
  if (child == NULL || child->pid > 0 || child->report_descriptor >= 0 ||
      pipe2(report_pipe, O_CLOEXEC | O_NONBLOCK) != 0) {
    return -1;
  }
  if (sagr_ccl_live_v1_broker_prepare_rank(broker, rank, &capability) !=
      SAGR_CCL_V1_STATUS_SUCCESS) {
    (void)close_owned_descriptor(&report_pipe[0]);
    (void)close_owned_descriptor(&report_pipe[1]);
    return -1;
  }
  pid = fork();
  if (pid < 0) {
    (void)close_owned_descriptor(&capability);
    (void)close_owned_descriptor(&report_pipe[0]);
    (void)close_owned_descriptor(&report_pipe[1]);
    return -1;
  }
  if (pid == 0) {
    if (set_cloexec(capability, 0) != 0 ||
        set_cloexec(report_pipe[1], 0) != 0) {
      _exit(97);
    }
    (void)snprintf(capability_text, sizeof(capability_text), "%d", capability);
    (void)snprintf(report_text, sizeof(report_text), "%d", report_pipe[1]);
    (void)snprintf(world_text, sizeof(world_text), "%" PRIu32, world);
    (void)snprintf(rank_text, sizeof(rank_text), "%" PRIu32, rank);
    (void)snprintf(count_text, sizeof(count_text), "%" PRIu64, element_count);
    (void)snprintf(tag_text, sizeof(tag_text), "%" PRIu32, tag);
    (void)snprintf(owner_pid_text, sizeof(owner_pid_text), "%" PRId32,
                   owner->pid);
    (void)snprintf(owner_uid_text, sizeof(owner_uid_text), "%" PRIu32,
                   owner->uid);
    (void)snprintf(owner_gid_text, sizeof(owner_gid_text), "%" PRIu32,
                   owner->gid);
    (void)snprintf(owner_start_text, sizeof(owner_start_text), "%" PRIu64,
                   owner->start_time_ticks);
    execl("/proc/self/exe", "ccl_live_collective_v1_test", "--child", mode,
          capability_text, report_text, world_text, rank_text, count_text,
          tag_text, owner_pid_text, owner_uid_text, owner_gid_text,
          owner_start_text, (char *)NULL);
    _exit(98);
  }
  child->pid = pid;
  child->rank = rank;
  child->report_descriptor = report_pipe[0];
  report_pipe[0] = -1;
  {
    int close_failed = 0;
    if (close_owned_descriptor(&capability) != 0) {
      close_failed = 1;
    }
    if (close_owned_descriptor(&report_pipe[1]) != 0) {
      close_failed = 1;
    }
    if (close_failed != 0) {
      return -1;
    }
  }
  return 0;
}

static int bind_children(sagr_ccl_live_v1_broker_t broker,
                         child_process_t *children, uint32_t world) {
  uint32_t rank;
  for (rank = 0U; rank < world; ++rank) {
    sagr_ccl_live_v1_process_identity_t identity;
    if (children[rank].pid <= 0 ||
        sagr_ccl_live_v1_process_identity(
            (int32_t)children[rank].pid, &identity,
            (uint32_t)sizeof(identity)) != SAGR_CCL_V1_STATUS_SUCCESS ||
        sagr_ccl_live_v1_broker_bind_rank(broker, rank, &identity) !=
            SAGR_CCL_V1_STATUS_SUCCESS) {
      return -1;
    }
  }
  return 0;
}

static int spawn_world(sagr_ccl_live_v1_broker_t broker,
                       const sagr_ccl_live_v1_process_identity_t *owner,
                       uint32_t world, uint64_t element_count, uint32_t tag,
                       const char *mode, child_process_t *children) {
  uint32_t rank;
  for (rank = 0U; rank < world; ++rank) {
    if (spawn_rank(broker, owner, world, rank, element_count, tag, mode,
                   &children[rank]) != 0) {
      return -1;
    }
  }
  return bind_children(broker, children, world);
}

static int drive_until_closed(sagr_ccl_live_v1_broker_t broker,
                              uint64_t deadline) {
  for (;;) {
    sagr_ccl_v1_status_t status =
        sagr_ccl_live_v1_broker_progress(broker, NULL, 0U);
    if (status == SAGR_CCL_V1_STATUS_CLOSED) {
      return 0;
    }
    if (status != SAGR_CCL_V1_STATUS_SUCCESS ||
        sagr_ccl_live_v1_monotonic_time_ns() >= deadline) {
      return -1;
    }
    (void)usleep(1000U);
  }
}

static int drive_until_abort_flushed(sagr_ccl_live_v1_broker_t broker,
                                     sagr_ccl_v1_status_t expected,
                                     sagr_ccl_live_v1_abort_t *first_error,
                                     uint64_t deadline) {
  int observed = 0;
  for (;;) {
    sagr_ccl_live_v1_broker_info_t info;
    sagr_ccl_v1_status_t status = sagr_ccl_live_v1_broker_progress(
        broker, first_error, (uint32_t)sizeof(*first_error));
    if (status == expected) {
      observed = 1;
    } else if (status != SAGR_CCL_V1_STATUS_SUCCESS) {
      return -1;
    }
    if (sagr_ccl_live_v1_broker_info(broker, &info,
                                     (uint32_t)sizeof(info)) !=
        SAGR_CCL_V1_STATUS_SUCCESS) {
      return -1;
    }
    if (observed && info.abort_pending_mask == 0U) {
      return 0;
    }
    if (sagr_ccl_live_v1_monotonic_time_ns() >= deadline) {
      return -1;
    }
    (void)usleep(1000U);
  }
}

static int read_reports(child_process_t *children, child_report_t *reports,
                        uint32_t world, uint64_t deadline) {
  uint32_t rank;
  int result = 0;
  for (rank = 0U; rank < world; ++rank) {
    if (children[rank].report_descriptor < 0 ||
        read_all(children[rank].report_descriptor, &reports[rank],
                 sizeof(reports[rank]), deadline) != 0) {
      result = -1;
    }
    if (close_owned_descriptor(&children[rank].report_descriptor) != 0) {
      result = -1;
    }
  }
  return result;
}

static int wait_children_bounded(child_process_t *children, uint32_t world,
                                 uint64_t deadline, uint32_t *outcome) {
  uint32_t remaining = 0U;
  uint32_t rank;
  uint32_t observed = 0U;
  for (rank = 0U; rank < world; ++rank) {
    if (children[rank].pid > 0) {
      ++remaining;
    }
  }
  while (remaining != 0U) {
    uint64_t now = sagr_ccl_live_v1_monotonic_time_ns();
    if (now == 0U || now >= deadline) {
      observed |= WAIT_OUTCOME_TIMEOUT;
      break;
    }
    for (rank = 0U; rank < world; ++rank) {
      int status = 0;
      pid_t value;
      if (children[rank].pid <= 0) {
        continue;
      }
      do {
        value = waitpid(children[rank].pid, &status, WNOHANG);
      } while (value < 0 && errno == EINTR);
      if (value == 0) {
        continue;
      }
      if (value == children[rank].pid) {
        children[rank].pid = (pid_t)-1;
        --remaining;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
          observed |= WAIT_OUTCOME_NONZERO;
        }
      } else if (value < 0 && errno == ECHILD) {
        children[rank].pid = (pid_t)-1;
        --remaining;
        observed |= WAIT_OUTCOME_ERROR;
      } else {
        observed |= WAIT_OUTCOME_ERROR;
      }
    }
    if (remaining != 0U) {
      (void)usleep(1000U);
    }
  }
  if (outcome != NULL) {
    *outcome = observed;
  }
  return observed == 0U ? 0 : -1;
}

static int cleanup_children(child_process_t *children, uint32_t world,
                            uint64_t deadline) {
  uint32_t rank;
  uint32_t remaining = 0U;
  int result = 0;
  for (rank = 0U; rank < world; ++rank) {
    if (close_owned_descriptor(&children[rank].report_descriptor) != 0) {
      result = -1;
    }
    if (children[rank].pid > 0) {
      if (kill(children[rank].pid, SIGKILL) != 0 && errno != ESRCH) {
        result = -1;
      }
      ++remaining;
    }
  }
  while (remaining != 0U) {
    uint64_t now = sagr_ccl_live_v1_monotonic_time_ns();
    if (now == 0U || now >= deadline) {
      return -1;
    }
    for (rank = 0U; rank < world; ++rank) {
      int status = 0;
      pid_t value;
      if (children[rank].pid <= 0) {
        continue;
      }
      do {
        value = waitpid(children[rank].pid, &status, WNOHANG);
      } while (value < 0 && errno == EINTR);
      if (value == children[rank].pid || (value < 0 && errno == ECHILD)) {
        children[rank].pid = (pid_t)-1;
        --remaining;
      } else if (value < 0) {
        result = -1;
      }
    }
    if (remaining != 0U) {
      (void)usleep(1000U);
    }
  }
  return result;
}

static void print_digest(const uint8_t digest[SAGR_CCL_V1_SHA256_BYTES]) {
  uint32_t index;
  for (index = 0U; index < SAGR_CCL_V1_SHA256_BYTES; ++index) {
    (void)printf("%02x", (unsigned)digest[index]);
  }
}

static int test_collective_world(uint32_t world, uint64_t element_count,
                                 uint32_t tag) {
  sagr_ccl_v1_group_identity_t identity = make_identity(world, tag);
  sagr_ccl_live_v1_broker_t broker = NULL;
  sagr_ccl_live_v1_broker_info_t broker_info;
  child_process_t children[SAGR_CCL_V1_MAX_WORLD_SIZE];
  child_report_t reports[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint32_t *expected = NULL;
  uint8_t expected_digest[SAGR_CCL_V1_SHA256_BYTES];
  uint64_t host_reductions = 0U;
  uint64_t zero_transfers = 0U;
  uint64_t data_count = 0U;
  uint64_t consumed_count = 0U;
  uint64_t nonzero_chunks = element_count < world ? element_count : world;
  uint32_t rank;
  uint64_t element;
  int before = count_open_descriptors();
  uint64_t deadline;
  int result = 0;
  initialize_children(children, SAGR_CCL_V1_MAX_WORLD_SIZE);
  memset(reports, 0, sizeof(reports));
  TEST_CHECK(before >= 0);
  TEST_CHECK(sagr_ccl_live_v1_broker_create(&identity, &broker) ==
             SAGR_CCL_V1_STATUS_SUCCESS);
  TEST_CHECK(sagr_ccl_live_v1_broker_info(broker, &broker_info,
                                         (uint32_t)sizeof(broker_info)) ==
             SAGR_CCL_V1_STATUS_SUCCESS);
  TEST_CHECK(spawn_world(broker, &broker_info.owner, world, element_count, tag,
                         "collective", children) == 0);
  deadline = deadline_after(COLLECTIVE_DEADLINE_NS);
  TEST_CHECK(deadline != 0U);
  TEST_CHECK(sagr_ccl_live_v1_broker_rendezvous(broker, deadline) ==
             SAGR_CCL_V1_STATUS_SUCCESS);
  deadline = deadline_after(COLLECTIVE_DEADLINE_NS);
  TEST_CHECK(deadline != 0U && drive_until_closed(broker, deadline) == 0);
  deadline = deadline_after(CHILD_WAIT_NS);
  TEST_CHECK(deadline != 0U &&
             read_reports(children, reports, world, deadline) == 0);
  deadline = deadline_after(CHILD_WAIT_NS);
  TEST_CHECK(deadline != 0U &&
             wait_children_bounded(children, world, deadline, NULL) == 0);
  expected = (uint32_t *)calloc(element_count == 0U ? 1U : (size_t)element_count,
                                sizeof(*expected));
  TEST_CHECK(expected != NULL);
  for (element = 0U; element < element_count; ++element) {
    expected[element] = mock_expected_value(world, element);
  }
  hash_words(expected, element_count, expected_digest);
  free(expected);
  expected = NULL;
  for (rank = 0U; rank < world; ++rank) {
    const child_report_t *report = &reports[rank];
    TEST_CHECK(report->magic == REPORT_MAGIC &&
               report->mode == CHILD_MODE_COLLECTIVE &&
               report->world_size == world && report->rank == rank &&
               report->element_count == element_count &&
               report->status == SAGR_CCL_V1_STATUS_SUCCESS);
    TEST_CHECK(report->data_count == 2U * (world - 1U) &&
               report->consumed_count == report->data_count &&
               report->ordered_stage_count == report->data_count);
    TEST_CHECK(report->device_sum_count == 0U &&
               report->fd_cleanup_ok == 1U);
    TEST_CHECK(memcmp(report->result_sha256, expected_digest,
                      sizeof(expected_digest)) == 0);
    if (world == 2U) {
      TEST_CHECK(report->credit_busy_observed == (rank == 0U ? 1U : 0U));
      TEST_CHECK(report->out_of_order_observed == (rank == 1U ? 1U : 0U));
    } else {
      TEST_CHECK(report->credit_busy_observed == 0U &&
                 report->out_of_order_observed == 0U);
    }
    host_reductions += report->host_mock_reduction_count;
    zero_transfers += report->zero_chunk_send_count;
    data_count += report->data_count;
    consumed_count += report->consumed_count;
  }
  TEST_CHECK(host_reductions == nonzero_chunks * (world - 1U));
  TEST_CHECK(zero_transfers ==
             UINT64_C(2) * (world - nonzero_chunks) * (world - 1U));
  TEST_CHECK(data_count ==
             (uint64_t)world * UINT64_C(2) * (world - 1U));
  TEST_CHECK(consumed_count == data_count);
  (void)printf("ccl_live_collective_v1: world=%" PRIu32
               " count=%" PRIu64 " sha256=",
               world, element_count);
  print_digest(expected_digest);
  (void)printf(" host_mock_reduction_count=%" PRIu64
               " device_sum_count=0 zero_chunk_transfers=%" PRIu64 "\n",
               host_reductions, zero_transfers);
cleanup:
  free(expected);
  deadline = deadline_after(CHILD_WAIT_NS);
  if (deadline == 0U || cleanup_children(children, world, deadline) != 0) {
    fprintf(stderr, "collective child cleanup exceeded its bounded deadline\n");
    result = 1;
  }
  sagr_ccl_live_v1_broker_destroy(&broker);
  if (before >= 0 && count_open_descriptors() != before) {
    fprintf(stderr, "collective parent descriptor baseline was not restored\n");
    result = 1;
  }
  return result;
}

static int test_fault_world(const char *mode, uint32_t expected_mode,
                            sagr_ccl_v1_status_t expected_status,
                            uint32_t expected_reporter,
                            uint32_t expected_failed_rank,
                            uint64_t expected_sequence, uint32_t tag) {
  const uint32_t world = 3U;
  sagr_ccl_v1_group_identity_t identity = make_identity(world, tag);
  sagr_ccl_live_v1_broker_t broker = NULL;
  sagr_ccl_live_v1_broker_info_t broker_info;
  sagr_ccl_live_v1_abort_t first_error;
  child_process_t children[SAGR_CCL_V1_MAX_WORLD_SIZE];
  child_report_t reports[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint64_t start;
  uint64_t deadline;
  uint32_t rank;
  int before = count_open_descriptors();
  int result = 0;
  initialize_children(children, SAGR_CCL_V1_MAX_WORLD_SIZE);
  memset(reports, 0, sizeof(reports));
  TEST_CHECK(before >= 0);
  TEST_CHECK(sagr_ccl_live_v1_broker_create(&identity, &broker) ==
             SAGR_CCL_V1_STATUS_SUCCESS);
  TEST_CHECK(sagr_ccl_live_v1_broker_info(broker, &broker_info,
                                         (uint32_t)sizeof(broker_info)) ==
             SAGR_CCL_V1_STATUS_SUCCESS);
  TEST_CHECK(spawn_world(broker, &broker_info.owner, world, 0U, tag, mode,
                         children) == 0);
  deadline = deadline_after(FAULT_DEADLINE_NS);
  TEST_CHECK(deadline != 0U);
  TEST_CHECK(sagr_ccl_live_v1_broker_rendezvous(broker, deadline) ==
             SAGR_CCL_V1_STATUS_SUCCESS);
  start = sagr_ccl_live_v1_monotonic_time_ns();
  deadline = deadline_after(FAULT_DEADLINE_NS);
  TEST_CHECK(start != 0U && deadline != 0U);
  TEST_CHECK(drive_until_abort_flushed(broker, expected_status, &first_error,
                                       deadline) == 0);
  TEST_CHECK(sagr_ccl_live_v1_monotonic_time_ns() < deadline);
  TEST_CHECK(first_error.reporter_rank == expected_reporter &&
             first_error.failed_rank == expected_failed_rank &&
             first_error.status == expected_status &&
             first_error.context_sequence == expected_sequence);
  deadline = deadline_after(CHILD_WAIT_NS);
  TEST_CHECK(deadline != 0U &&
             read_reports(children, reports, world, deadline) == 0);
  deadline = deadline_after(CHILD_WAIT_NS);
  TEST_CHECK(deadline != 0U &&
             wait_children_bounded(children, world, deadline, NULL) == 0);
  for (rank = 0U; rank < world; ++rank) {
    TEST_CHECK(reports[rank].magic == REPORT_MAGIC &&
               reports[rank].mode == expected_mode &&
               reports[rank].world_size == world &&
               reports[rank].rank == rank &&
               reports[rank].status == expected_status &&
               reports[rank].deadline_ok == 1U &&
               reports[rank].fd_cleanup_ok == 1U &&
               reports[rank].abort_reporter_rank == expected_reporter &&
               reports[rank].abort_failed_rank == expected_failed_rank &&
               reports[rank].abort_sequence == expected_sequence);
  }
  (void)printf("ccl_live_collective_v1: %s world=3 status=%s "
               "bounded_deadline_ms<5000\n",
               mode, sagr_ccl_v1_status_string(expected_status));
cleanup:
  deadline = deadline_after(CHILD_WAIT_NS);
  if (deadline == 0U || cleanup_children(children, world, deadline) != 0) {
    fprintf(stderr, "fault child cleanup exceeded its bounded deadline\n");
    result = 1;
  }
  sagr_ccl_live_v1_broker_destroy(&broker);
  if (before >= 0 && count_open_descriptors() != before) {
    fprintf(stderr, "fault parent descriptor baseline was not restored\n");
    result = 1;
  }
  return result;
}

static int test_post_spawn_failure_cleanup(void) {
  const uint32_t world = 3U;
  const uint32_t spawned = 2U;
  sagr_ccl_v1_group_identity_t identity = make_identity(world, 250U);
  sagr_ccl_live_v1_broker_t broker = NULL;
  sagr_ccl_live_v1_broker_info_t broker_info;
  child_process_t children[SAGR_CCL_V1_MAX_WORLD_SIZE];
  child_report_t reports[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint32_t outcome = 0U;
  uint32_t rank;
  uint64_t deadline;
  int before = count_open_descriptors();
  int result = 0;
  initialize_children(children, SAGR_CCL_V1_MAX_WORLD_SIZE);
  memset(reports, 0, sizeof(reports));
  TEST_CHECK(before >= 0);
  TEST_CHECK(sagr_ccl_live_v1_broker_create(&identity, &broker) ==
             SAGR_CCL_V1_STATUS_SUCCESS);
  TEST_CHECK(sagr_ccl_live_v1_broker_info(broker, &broker_info,
                                         (uint32_t)sizeof(broker_info)) ==
             SAGR_CCL_V1_STATUS_SUCCESS);
  TEST_CHECK(spawn_rank(broker, &broker_info.owner, world, 0U, 0U, 250U,
                        "injected-exit", &children[0]) == 0);
  TEST_CHECK(spawn_rank(broker, &broker_info.owner, world, 1U, 0U, 250U,
                        "injected-hang", &children[1]) == 0);
  deadline = deadline_after(INJECTED_WAIT_NS);
  TEST_CHECK(deadline != 0U &&
             read_reports(children, reports, spawned, deadline) != 0);
  deadline = deadline_after(INJECTED_WAIT_NS);
  TEST_CHECK(deadline != 0U &&
             wait_children_bounded(children, spawned, deadline, &outcome) !=
                 0);
  TEST_CHECK((outcome & WAIT_OUTCOME_NONZERO) != 0U &&
             (outcome & WAIT_OUTCOME_TIMEOUT) != 0U &&
             (outcome & WAIT_OUTCOME_ERROR) == 0U);

cleanup:
  deadline = deadline_after(CHILD_WAIT_NS);
  if (deadline == 0U || cleanup_children(children, spawned, deadline) != 0) {
    fprintf(stderr, "injected child cleanup exceeded its bounded deadline\n");
    result = 1;
  }
  sagr_ccl_live_v1_broker_destroy(&broker);
  for (rank = 0U; rank < spawned; ++rank) {
    if (children[rank].pid > 0 || children[rank].report_descriptor >= 0) {
      fprintf(stderr, "injected child ownership remained after cleanup\n");
      result = 1;
    }
  }
  {
    int status = 0;
    pid_t value;
    errno = 0;
    value = waitpid((pid_t)-1, &status, WNOHANG);
    if (value != (pid_t)-1 || errno != ECHILD) {
      fprintf(stderr, "injected cleanup left an orphan or zombie child\n");
      result = 1;
    }
  }
  if (before >= 0 && count_open_descriptors() != before) {
    fprintf(stderr, "injected cleanup did not restore the FD baseline\n");
    result = 1;
  }
  if (result == 0) {
    puts("ccl_live_collective_v1: post_spawn_failure_cleanup=true "
         "report_timeout=true child_nonzero=true child_timeout=true "
         "bounded_kill_reap=true fd_delta=0 no_orphan=true");
  }
  return result;
}

int main(int argc, char **argv) {
  if (argc > 1) {
    return child_main(argc, argv);
  }
  REQUIRE(test_collective_world(2U, 9U, 102U) == 0);
  REQUIRE(test_collective_world(3U, 11U, 103U) == 0);
  REQUIRE(test_collective_world(4U, 13U, 104U) == 0);
  REQUIRE(test_collective_world(8U, 3U, 108U) == 0);
  REQUIRE(test_collective_world(16U, 19U, 116U) == 0);
  REQUIRE(test_fault_world("group-abort", CHILD_MODE_GROUP_ABORT,
                           SAGR_CCL_V1_STATUS_CHECKSUM_ERROR, 0U, 1U, 77U,
                           201U) == 0);
  REQUIRE(test_fault_world("peer-loss", CHILD_MODE_PEER_LOSS,
                           SAGR_CCL_V1_STATUS_PEER_LOST,
                           SAGR_CCL_LIVE_V1_NO_RANK, 0U, 0U, 202U) == 0);
  REQUIRE(test_post_spawn_failure_cleanup() == 0);
  puts("ccl_live_collective_v1: standalone_host_mock=true "
       "ordering_transport_staging_acceptance=true "
       "device_collective_acceptance=false live_gem5_acceptance=false "
       "mock_payload=rank_coded_uint32 descriptor=fp32_allreduce "
       "host_arithmetic_only=true device_sum_count=0 fd_orphan_cleanup=true");
  return 0;
}
