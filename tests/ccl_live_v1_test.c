/* SPDX-License-Identifier: GPL-3.0-or-later */

#define _GNU_SOURCE

#include <self_amdgpu_runtime/ccl_live_v1.h>

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <poll.h>
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

#define TEST_DEADLINE_NS UINT64_C(5000000000)

typedef struct child_process {
  pid_t pid;
  uint32_t rank;
  int expected_exit;
} child_process_t;

typedef struct ring_packet {
  uint32_t magic;
  uint32_t source_rank;
  uint32_t destination_rank;
  uint32_t world_size;
  uint8_t group_uuid[SAGR_CCL_V1_UUID_BYTES];
} ring_packet_t;

static uint64_t deadline_after(uint64_t nanoseconds) {
  uint64_t now = sagr_ccl_live_v1_monotonic_time_ns();
  return now == 0U || UINT64_MAX - now < nanoseconds ? 0U : now + nanoseconds;
}

static sagr_ccl_v1_group_identity_t make_identity(uint32_t world_size,
                                                   uint32_t tag) {
  sagr_ccl_v1_group_identity_t identity;
  uint32_t index;
  (void)sagr_ccl_v1_group_identity_init(&identity,
                                        (uint32_t)sizeof(identity));
  identity.world_size = world_size;
  identity.epoch = UINT64_C(0x1020304000000000) + tag;
  identity.group_generation = UINT64_C(0x5060708000000000) + tag;
  for (index = 0U; index < SAGR_CCL_V1_UUID_BYTES; ++index) {
    identity.job_uuid[index] = (uint8_t)(tag + index + 1U);
    identity.group_uuid[index] = (uint8_t)(tag * 3U + index + 17U);
  }
  for (index = 0U; index < SAGR_CCL_V1_SHA256_BYTES; ++index) {
    identity.model_identity_sha256[index] =
        (uint8_t)(tag * 5U + index + 33U);
  }
  return identity;
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
    if (strcmp(entry->d_name, ".") != 0 &&
        strcmp(entry->d_name, "..") != 0) {
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
    timeout =
        milliseconds > (uint64_t)INT_MAX ? INT_MAX : (int)milliseconds;
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

static int send_packet(int descriptor, const void *bytes, size_t count,
                       uint64_t deadline) {
  for (;;) {
    ssize_t result = send(descriptor, bytes, count, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (result == (ssize_t)count) {
      return 0;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) &&
        wait_descriptor(descriptor, POLLOUT, deadline) == 0) {
      continue;
    }
    return -1;
  }
}

static int receive_packet(int descriptor, void *bytes, size_t count,
                          uint64_t deadline) {
  for (;;) {
    ssize_t result = recv(descriptor, bytes, count, MSG_DONTWAIT);
    if (result == (ssize_t)count) {
      return 0;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) &&
        wait_descriptor(descriptor, POLLIN, deadline) == 0) {
      continue;
    }
    return -1;
  }
}

static int run_ring(sagr_ccl_live_v1_rank_t rank,
                    const sagr_ccl_v1_group_identity_t *expected_group) {
  sagr_ccl_live_v1_rank_info_t info;
  ring_packet_t sent;
  ring_packet_t received;
  uint32_t next;
  uint32_t previous;
  uint32_t peer;
  uint64_t deadline = deadline_after(TEST_DEADLINE_NS);
  REQUIRE(sagr_ccl_live_v1_rank_info(rank, &info,
                                    (uint32_t)sizeof(info)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(info.phase == SAGR_CCL_LIVE_V1_PHASE_READY);
  REQUIRE(info.self_rank < info.world_size);
  REQUIRE(info.control_socket >= 0);
  REQUIRE(sagr_ccl_v1_group_identity_equal(&info.group, expected_group));
  for (peer = 0U; peer < SAGR_CCL_V1_MAX_WORLD_SIZE; ++peer) {
    if (peer == info.self_rank || peer >= info.world_size) {
      REQUIRE(info.peer_sockets[peer] == -1);
    } else {
      int type = 0;
      int domain = 0;
      socklen_t size = (socklen_t)sizeof(type);
      REQUIRE(info.peer_sockets[peer] >= 0);
      REQUIRE(getsockopt(info.peer_sockets[peer], SOL_SOCKET, SO_TYPE, &type,
                         &size) == 0 &&
              type == SOCK_SEQPACKET);
      size = (socklen_t)sizeof(domain);
      REQUIRE(getsockopt(info.peer_sockets[peer], SOL_SOCKET, SO_DOMAIN,
                         &domain, &size) == 0 &&
              domain == AF_UNIX);
      REQUIRE((fcntl(info.peer_sockets[peer], F_GETFD) & FD_CLOEXEC) != 0);
      REQUIRE((fcntl(info.peer_sockets[peer], F_GETFL) & O_NONBLOCK) != 0);
    }
  }
  next = (info.self_rank + 1U) % info.world_size;
  previous = (info.self_rank + info.world_size - 1U) % info.world_size;
  memset(&sent, 0, sizeof(sent));
  sent.magic = UINT32_C(0x52494e47);
  sent.source_rank = info.self_rank;
  sent.destination_rank = next;
  sent.world_size = info.world_size;
  memcpy(sent.group_uuid, info.group.group_uuid, sizeof(sent.group_uuid));
  REQUIRE(send_packet(info.peer_sockets[next], &sent, sizeof(sent), deadline) ==
          0);
  REQUIRE(receive_packet(info.peer_sockets[previous], &received,
                         sizeof(received), deadline) == 0);
  REQUIRE(received.magic == UINT32_C(0x52494e47));
  REQUIRE(received.source_rank == previous);
  REQUIRE(received.destination_rank == info.self_rank);
  REQUIRE(received.world_size == info.world_size);
  REQUIRE(memcmp(received.group_uuid, info.group.group_uuid,
                 sizeof(received.group_uuid)) == 0);
  return 0;
}

static int notify_parent(int descriptor) {
  const uint8_t marker = UINT8_C(0xa5);
  if (descriptor < 0) {
    return 0;
  }
  for (;;) {
    ssize_t result = write(descriptor, &marker, sizeof(marker));
    if (result == (ssize_t)sizeof(marker)) {
      return 0;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    return -1;
  }
}

static sagr_ccl_v1_status_t wait_abort(
    sagr_ccl_live_v1_rank_t rank, sagr_ccl_live_v1_abort_t *value) {
  uint64_t deadline = deadline_after(TEST_DEADLINE_NS);
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

static int child_main(int argc, char **argv) {
  const char *mode;
  int32_t capability;
  int32_t sync_descriptor;
  uint32_t world;
  uint32_t actual_rank;
  uint32_t tag;
  int32_t owner_pid;
  uint32_t owner_uid;
  uint32_t owner_gid;
  uint64_t owner_start;
  uint32_t claimed_rank;
  sagr_ccl_v1_group_identity_t identity;
  sagr_ccl_live_v1_process_identity_t owner;
  sagr_ccl_live_v1_rank_t rank = NULL;
  sagr_ccl_v1_status_t status;
  int descriptors_before;
  if (argc != 12 || strcmp(argv[1], "--child") != 0) {
    return 90;
  }
  mode = argv[2];
  if (parse_i32(argv[3], &capability) != 0 ||
      parse_i32(argv[4], &sync_descriptor) != 0 ||
      parse_u32(argv[5], &world) != 0 ||
      parse_u32(argv[6], &actual_rank) != 0 ||
      parse_u32(argv[7], &tag) != 0 ||
      parse_i32(argv[8], &owner_pid) != 0 ||
      parse_u32(argv[9], &owner_uid) != 0 ||
      parse_u32(argv[10], &owner_gid) != 0 ||
      parse_u64(argv[11], &owner_start) != 0) {
    return 91;
  }
  identity = make_identity(world, tag);
  descriptors_before = count_open_descriptors();
  if (descriptors_before < 0) {
    return 99;
  }
  memset(&owner, 0, sizeof(owner));
  owner.struct_size = (uint32_t)sizeof(owner);
  owner.pid = owner_pid;
  owner.uid = owner_uid;
  owner.gid = owner_gid;
  owner.start_time_ticks = owner_start;
  if (strcmp(mode, "hold-no-join") == 0) {
    (void)usleep(600000U);
    (void)close(capability);
    return count_open_descriptors() == descriptors_before - 1 ? 0 : 99;
  }
  if (strcmp(mode, "wrong-group") == 0 && actual_rank == 0U) {
    ++identity.group_generation;
  }
  claimed_rank =
      strcmp(mode, "wrong-claim") == 0 && actual_rank == 0U ? 1U
                                                              : actual_rank;
  status = sagr_ccl_live_v1_rank_join(
      capability, &identity, claimed_rank, &owner,
      deadline_after(TEST_DEADLINE_NS), &rank);
  if (strcmp(mode, "wrong-claim") == 0 ||
      strcmp(mode, "wrong-group") == 0 ||
      strcmp(mode, "swapped-binding") == 0) {
    sagr_ccl_v1_status_t expected_status =
        strcmp(mode, "wrong-claim") == 0
            ? SAGR_CCL_V1_STATUS_PROTOCOL_ERROR
            : SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH;
    if (sync_descriptor >= 0) {
      (void)close(sync_descriptor);
    }
    return status == expected_status &&
                   count_open_descriptors() == descriptors_before - 1
               ? 0
               : 92;
  }
  if (strcmp(mode, "expect-join-timeout") == 0) {
    if (sync_descriptor >= 0) {
      (void)close(sync_descriptor);
    }
    return status == SAGR_CCL_V1_STATUS_TIMED_OUT &&
                   count_open_descriptors() == descriptors_before - 1
               ? 0
               : 93;
  }
  if (status != SAGR_CCL_V1_STATUS_SUCCESS || rank == NULL) {
    if (sync_descriptor >= 0) {
      (void)close(sync_descriptor);
    }
    return 94;
  }
  if (strcmp(mode, "success") == 0) {
    int result = run_ring(rank, &identity);
    if (result == 0) {
      status = sagr_ccl_live_v1_rank_close(
          rank, deadline_after(TEST_DEADLINE_NS));
      result = status == SAGR_CCL_V1_STATUS_SUCCESS ? 0 : 95;
    }
    sagr_ccl_live_v1_rank_destroy(&rank);
    if (result == 0 &&
        count_open_descriptors() != descriptors_before - 1) {
      result = 99;
    }
    return result;
  }
  if (strcmp(mode, "close-timeout") == 0) {
    sagr_ccl_live_v1_abort_t value;
    REQUIRE(notify_parent(sync_descriptor) == 0);
    status = sagr_ccl_live_v1_rank_close(
        rank, deadline_after(UINT64_C(500000000)));
    REQUIRE(status == SAGR_CCL_V1_STATUS_TIMED_OUT);
    REQUIRE(notify_parent(sync_descriptor) == 0);
    (void)close(sync_descriptor);
    status = wait_abort(rank, &value);
    REQUIRE(status == SAGR_CCL_V1_STATUS_TIMED_OUT);
    REQUIRE(value.reporter_rank == 0U && value.failed_rank == 0U &&
            value.context_sequence == 0U);
    sagr_ccl_live_v1_rank_destroy(&rank);
    return count_open_descriptors() == descriptors_before - 2 ? 0 : 99;
  }
  if (strcmp(mode, "first-error") == 0) {
    sagr_ccl_live_v1_abort_t value;
    if (actual_rank == 0U) {
      REQUIRE(sagr_ccl_live_v1_rank_report_abort(
                  rank, 1U, SAGR_CCL_V1_STATUS_CHECKSUM_ERROR, 7U) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
    } else if (actual_rank == 1U) {
      REQUIRE(sagr_ccl_live_v1_rank_report_abort(
                  rank, 2U, SAGR_CCL_V1_STATUS_PEER_LOST, 8U) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
    }
    REQUIRE(notify_parent(sync_descriptor) == 0);
    (void)close(sync_descriptor);
    status = wait_abort(rank, &value);
    REQUIRE(status == SAGR_CCL_V1_STATUS_CHECKSUM_ERROR);
    REQUIRE(value.reporter_rank == 0U && value.failed_rank == 1U &&
            value.context_sequence == 7U);
    sagr_ccl_live_v1_rank_destroy(&rank);
    return count_open_descriptors() == descriptors_before - 2 ? 0 : 99;
  }
  if (strcmp(mode, "peer-loss") == 0) {
    sagr_ccl_live_v1_abort_t value;
    REQUIRE(notify_parent(sync_descriptor) == 0);
    (void)close(sync_descriptor);
    if (actual_rank == 0U) {
      sagr_ccl_live_v1_rank_destroy(&rank);
      return count_open_descriptors() == descriptors_before - 2 ? 41 : 42;
    }
    status = wait_abort(rank, &value);
    REQUIRE(status == SAGR_CCL_V1_STATUS_PEER_LOST);
    REQUIRE(value.reporter_rank == SAGR_CCL_LIVE_V1_NO_RANK &&
            value.failed_rank == 0U && value.context_sequence == 0U);
    sagr_ccl_live_v1_rank_destroy(&rank);
    return count_open_descriptors() == descriptors_before - 2 ? 0 : 99;
  }
  sagr_ccl_live_v1_rank_destroy(&rank);
  if (sync_descriptor >= 0) {
    (void)close(sync_descriptor);
  }
  return 96;
}

static int spawn_rank(sagr_ccl_live_v1_broker_t broker,
                      const sagr_ccl_live_v1_process_identity_t *owner,
                      uint32_t world, uint32_t rank, uint32_t tag,
                      const char *mode, int sync_descriptor,
                      child_process_t *child) {
  int capability = -1;
  pid_t pid;
  char capability_text[32];
  char sync_text[32];
  char world_text[32];
  char rank_text[32];
  char tag_text[32];
  char owner_pid_text[32];
  char owner_uid_text[32];
  char owner_gid_text[32];
  char owner_start_text[32];
  REQUIRE(sagr_ccl_live_v1_broker_prepare_rank(broker, rank, &capability) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    if (set_cloexec(capability, 0) != 0 ||
        (sync_descriptor >= 0 && set_cloexec(sync_descriptor, 0) != 0)) {
      _exit(97);
    }
    (void)snprintf(capability_text, sizeof(capability_text), "%d", capability);
    (void)snprintf(sync_text, sizeof(sync_text), "%d", sync_descriptor);
    (void)snprintf(world_text, sizeof(world_text), "%" PRIu32, world);
    (void)snprintf(rank_text, sizeof(rank_text), "%" PRIu32, rank);
    (void)snprintf(tag_text, sizeof(tag_text), "%" PRIu32, tag);
    (void)snprintf(owner_pid_text, sizeof(owner_pid_text), "%" PRId32,
                   owner->pid);
    (void)snprintf(owner_uid_text, sizeof(owner_uid_text), "%" PRIu32,
                   owner->uid);
    (void)snprintf(owner_gid_text, sizeof(owner_gid_text), "%" PRIu32,
                   owner->gid);
    (void)snprintf(owner_start_text, sizeof(owner_start_text), "%" PRIu64,
                   owner->start_time_ticks);
    execl("/proc/self/exe", "ccl_live_v1_test", "--child", mode,
          capability_text, sync_text, world_text, rank_text, tag_text,
          owner_pid_text, owner_uid_text, owner_gid_text, owner_start_text,
          (char *)NULL);
    _exit(98);
  }
  REQUIRE(close(capability) == 0);
  child->pid = pid;
  child->rank = rank;
  child->expected_exit =
      strcmp(mode, "peer-loss") == 0 && rank == 0U ? 41 : 0;
  return 0;
}

static int bind_children(sagr_ccl_live_v1_broker_t broker,
                         child_process_t *children, uint32_t world,
                         int swapped) {
  sagr_ccl_live_v1_process_identity_t identities[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint32_t rank;
  for (rank = 0U; rank < world; ++rank) {
    REQUIRE(sagr_ccl_live_v1_process_identity(
                (int32_t)children[rank].pid, &identities[rank],
                (uint32_t)sizeof(identities[rank])) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
  }
  for (rank = 0U; rank < world; ++rank) {
    uint32_t identity_rank = swapped ? (rank + 1U) % world : rank;
    REQUIRE(sagr_ccl_live_v1_broker_bind_rank(
                broker, rank, &identities[identity_rank]) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
  }
  return 0;
}

static int wait_children(child_process_t *children, uint32_t world) {
  uint32_t rank;
  for (rank = 0U; rank < world; ++rank) {
    int status = 0;
    pid_t result;
    do {
      result = waitpid(children[rank].pid, &status, 0);
    } while (result < 0 && errno == EINTR);
    REQUIRE(result == children[rank].pid);
    REQUIRE(WIFEXITED(status));
    if (WEXITSTATUS(status) != children[rank].expected_exit) {
      fprintf(stderr, "rank %" PRIu32 " exited %d, expected %d\n", rank,
              WEXITSTATUS(status), children[rank].expected_exit);
      return 1;
    }
  }
  {
    int status = 0;
    pid_t result;
    errno = 0;
    result = waitpid(-1, &status, WNOHANG);
    REQUIRE(result == (pid_t)-1);
    REQUIRE(errno == ECHILD);
  }
  return 0;
}

static int spawn_world(sagr_ccl_live_v1_broker_t broker,
                       const sagr_ccl_live_v1_process_identity_t *owner,
                       uint32_t world, uint32_t tag, const char *mode,
                       int sync_descriptor, child_process_t *children,
                       int swapped) {
  uint32_t rank;
  for (rank = 0U; rank < world; ++rank) {
    REQUIRE(spawn_rank(broker, owner, world, rank, tag, mode,
                       sync_descriptor, &children[rank]) == 0);
  }
  REQUIRE(bind_children(broker, children, world, swapped) == 0);
  return 0;
}

static int drive_until_closed(sagr_ccl_live_v1_broker_t broker) {
  uint64_t deadline = deadline_after(TEST_DEADLINE_NS);
  for (;;) {
    sagr_ccl_v1_status_t status =
        sagr_ccl_live_v1_broker_progress(broker, NULL, 0U);
    if (status == SAGR_CCL_V1_STATUS_CLOSED) {
      return 0;
    }
    if (status != SAGR_CCL_V1_STATUS_SUCCESS ||
        sagr_ccl_live_v1_monotonic_time_ns() >= deadline) {
      return 1;
    }
    (void)usleep(1000U);
  }
}

static int test_success_world(uint32_t world, uint32_t tag) {
  sagr_ccl_v1_group_identity_t identity = make_identity(world, tag);
  sagr_ccl_live_v1_broker_t broker = NULL;
  sagr_ccl_live_v1_broker_info_t info;
  child_process_t children[SAGR_CCL_V1_MAX_WORLD_SIZE];
  int before = count_open_descriptors();
  REQUIRE(before >= 0);
  REQUIRE(sagr_ccl_live_v1_broker_create(&identity, &broker) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_live_v1_broker_info(broker, &info,
                                      (uint32_t)sizeof(info)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(spawn_world(broker, &info.owner, world, tag, "success", -1,
                      children, 0) == 0);
  REQUIRE(sagr_ccl_live_v1_broker_rendezvous(
              broker, deadline_after(TEST_DEADLINE_NS)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(drive_until_closed(broker) == 0);
  REQUIRE(wait_children(children, world) == 0);
  sagr_ccl_live_v1_broker_destroy(&broker);
  REQUIRE(count_open_descriptors() == before);
  return 0;
}

static int test_generic_configuration_world2_16(void) {
  uint32_t world;
  for (world = SAGR_CCL_V1_MIN_WORLD_SIZE;
       world <= SAGR_CCL_V1_MAX_WORLD_SIZE; ++world) {
    sagr_ccl_v1_group_identity_t identity =
        make_identity(world, 80U + world);
    sagr_ccl_live_v1_broker_t broker = NULL;
    sagr_ccl_live_v1_broker_info_t info;
    int capabilities[SAGR_CCL_V1_MAX_WORLD_SIZE];
    int before = count_open_descriptors();
    uint32_t rank;
    REQUIRE(before >= 0);
    REQUIRE(sagr_ccl_live_v1_broker_create(&identity, &broker) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    for (rank = 0U; rank < SAGR_CCL_V1_MAX_WORLD_SIZE; ++rank) {
      capabilities[rank] = -1;
    }
    for (rank = 0U; rank < world; ++rank) {
      REQUIRE(sagr_ccl_live_v1_broker_prepare_rank(
                  broker, rank, &capabilities[rank]) ==
              SAGR_CCL_V1_STATUS_SUCCESS);
      REQUIRE(capabilities[rank] >= 0);
    }
    REQUIRE(sagr_ccl_live_v1_broker_info(broker, &info,
                                        (uint32_t)sizeof(info)) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(info.world_size == world);
    REQUIRE(info.prepared_mask == (UINT32_C(1) << world) - UINT32_C(1));
    REQUIRE(info.bound_mask == 0U && info.joined_mask == 0U &&
            info.ready_mask == 0U);
    for (rank = 0U; rank < world; ++rank) {
      REQUIRE(close(capabilities[rank]) == 0);
    }
    sagr_ccl_live_v1_broker_destroy(&broker);
    REQUIRE(count_open_descriptors() == before);
  }
  return 0;
}

static int wait_markers(int descriptor, uint32_t expected) {
  uint8_t buffer[SAGR_CCL_V1_MAX_WORLD_SIZE];
  uint32_t count = 0U;
  uint64_t deadline = deadline_after(TEST_DEADLINE_NS);
  while (count < expected) {
    ssize_t result;
    REQUIRE(wait_descriptor(descriptor, POLLIN, deadline) == 0);
    do {
      result = read(descriptor, buffer + count, expected - count);
    } while (result < 0 && errno == EINTR);
    REQUIRE(result > 0);
    count += (uint32_t)result;
  }
  while (count > 0U) {
    REQUIRE(buffer[--count] == UINT8_C(0xa5));
  }
  return 0;
}

static int drive_abort(sagr_ccl_live_v1_broker_t broker,
                       sagr_ccl_v1_status_t expected,
                       sagr_ccl_live_v1_abort_t *first_error) {
  uint64_t deadline = deadline_after(TEST_DEADLINE_NS);
  for (;;) {
    sagr_ccl_v1_status_t status = sagr_ccl_live_v1_broker_progress(
        broker, first_error, (uint32_t)sizeof(*first_error));
    if (status == expected) {
      break;
    }
    REQUIRE(status == SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(sagr_ccl_live_v1_monotonic_time_ns() < deadline);
    (void)usleep(1000U);
  }
  for (;;) {
    sagr_ccl_live_v1_broker_info_t info;
    REQUIRE(sagr_ccl_live_v1_broker_info(broker, &info,
                                        (uint32_t)sizeof(info)) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    if (info.abort_pending_mask == 0U) {
      return 0;
    }
    REQUIRE(sagr_ccl_live_v1_broker_progress(
                broker, first_error, (uint32_t)sizeof(*first_error)) ==
            expected);
    REQUIRE(sagr_ccl_live_v1_monotonic_time_ns() < deadline);
    (void)usleep(1000U);
  }
}

static int test_first_error_world3(void) {
  const uint32_t world = 3U;
  const uint32_t tag = 31U;
  sagr_ccl_v1_group_identity_t identity = make_identity(world, tag);
  sagr_ccl_live_v1_broker_t broker = NULL;
  sagr_ccl_live_v1_broker_info_t info;
  sagr_ccl_live_v1_abort_t first_error;
  child_process_t children[SAGR_CCL_V1_MAX_WORLD_SIZE];
  int sync_pipe[2] = {-1, -1};
  int before = count_open_descriptors();
  REQUIRE(pipe2(sync_pipe, O_CLOEXEC) == 0);
  REQUIRE(sagr_ccl_live_v1_broker_create(&identity, &broker) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_live_v1_broker_info(broker, &info,
                                      (uint32_t)sizeof(info)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(spawn_world(broker, &info.owner, world, tag, "first-error",
                      sync_pipe[1], children, 0) == 0);
  REQUIRE(close(sync_pipe[1]) == 0);
  sync_pipe[1] = -1;
  REQUIRE(sagr_ccl_live_v1_broker_rendezvous(
              broker, deadline_after(TEST_DEADLINE_NS)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(wait_markers(sync_pipe[0], world) == 0);
  REQUIRE(drive_abort(broker, SAGR_CCL_V1_STATUS_CHECKSUM_ERROR,
                      &first_error) == 0);
  REQUIRE(first_error.reporter_rank == 0U && first_error.failed_rank == 1U &&
          first_error.status == SAGR_CCL_V1_STATUS_CHECKSUM_ERROR &&
          first_error.context_sequence == 7U);
  REQUIRE(wait_children(children, world) == 0);
  REQUIRE(close(sync_pipe[0]) == 0);
  sagr_ccl_live_v1_broker_destroy(&broker);
  REQUIRE(count_open_descriptors() == before);
  return 0;
}

static int test_close_timeout_abort_world2(void) {
  const uint32_t world = 2U;
  const uint32_t tag = 32U;
  sagr_ccl_v1_group_identity_t identity = make_identity(world, tag);
  sagr_ccl_live_v1_broker_t broker = NULL;
  sagr_ccl_live_v1_broker_info_t info;
  sagr_ccl_live_v1_abort_t first_error;
  child_process_t children[SAGR_CCL_V1_MAX_WORLD_SIZE];
  int sync_pipe[2] = {-1, -1};
  int before = count_open_descriptors();
  uint64_t progress_deadline = deadline_after(TEST_DEADLINE_NS);
  REQUIRE(pipe2(sync_pipe, O_CLOEXEC) == 0);
  REQUIRE(sagr_ccl_live_v1_broker_create(&identity, &broker) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_live_v1_broker_info(broker, &info,
                                      (uint32_t)sizeof(info)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(spawn_world(broker, &info.owner, world, tag, "close-timeout",
                      sync_pipe[1], children, 0) == 0);
  REQUIRE(close(sync_pipe[1]) == 0);
  sync_pipe[1] = -1;
  REQUIRE(sagr_ccl_live_v1_broker_rendezvous(
              broker, deadline_after(TEST_DEADLINE_NS)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  /* Enter CLOSING after both LEAVEs, then stop progress until ABORT is queued. */
  REQUIRE(wait_markers(sync_pipe[0], world) == 0);
  for (;;) {
    REQUIRE(sagr_ccl_live_v1_broker_progress(broker, NULL, 0U) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    REQUIRE(sagr_ccl_live_v1_broker_info(broker, &info,
                                        (uint32_t)sizeof(info)) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
    if (info.phase == SAGR_CCL_LIVE_V1_PHASE_CLOSING) {
      break;
    }
    REQUIRE(sagr_ccl_live_v1_monotonic_time_ns() < progress_deadline);
    (void)usleep(1000U);
  }
  REQUIRE(info.departed_mask == (UINT32_C(1) << world) - UINT32_C(1));
  REQUIRE(info.close_pending_mask ==
          (UINT32_C(1) << world) - UINT32_C(1));
  REQUIRE(wait_markers(sync_pipe[0], world) == 0);
  REQUIRE(drive_abort(broker, SAGR_CCL_V1_STATUS_TIMED_OUT, &first_error) == 0);
  REQUIRE(first_error.reporter_rank == 0U && first_error.failed_rank == 0U &&
          first_error.status == SAGR_CCL_V1_STATUS_TIMED_OUT &&
          first_error.context_sequence == 0U);
  REQUIRE(wait_children(children, world) == 0);
  memset(&first_error, 0, sizeof(first_error));
  REQUIRE(sagr_ccl_live_v1_broker_first_error(
              broker, &first_error, (uint32_t)sizeof(first_error)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(first_error.reporter_rank == 0U && first_error.failed_rank == 0U &&
          first_error.status == SAGR_CCL_V1_STATUS_TIMED_OUT &&
          first_error.context_sequence == 0U);
  REQUIRE(close(sync_pipe[0]) == 0);
  sagr_ccl_live_v1_broker_destroy(&broker);
  REQUIRE(count_open_descriptors() == before);
  return 0;
}

static int test_peer_loss_world3(void) {
  const uint32_t world = 3U;
  const uint32_t tag = 41U;
  sagr_ccl_v1_group_identity_t identity = make_identity(world, tag);
  sagr_ccl_live_v1_broker_t broker = NULL;
  sagr_ccl_live_v1_broker_info_t info;
  sagr_ccl_live_v1_abort_t first_error;
  child_process_t children[SAGR_CCL_V1_MAX_WORLD_SIZE];
  int sync_pipe[2] = {-1, -1};
  int before = count_open_descriptors();
  REQUIRE(pipe2(sync_pipe, O_CLOEXEC) == 0);
  REQUIRE(sagr_ccl_live_v1_broker_create(&identity, &broker) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_live_v1_broker_info(broker, &info,
                                      (uint32_t)sizeof(info)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(spawn_world(broker, &info.owner, world, tag, "peer-loss",
                      sync_pipe[1], children, 0) == 0);
  REQUIRE(close(sync_pipe[1]) == 0);
  sync_pipe[1] = -1;
  REQUIRE(sagr_ccl_live_v1_broker_rendezvous(
              broker, deadline_after(TEST_DEADLINE_NS)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(wait_markers(sync_pipe[0], world) == 0);
  REQUIRE(drive_abort(broker, SAGR_CCL_V1_STATUS_PEER_LOST,
                      &first_error) == 0);
  REQUIRE(first_error.reporter_rank == SAGR_CCL_LIVE_V1_NO_RANK &&
          first_error.failed_rank == 0U &&
          first_error.status == SAGR_CCL_V1_STATUS_PEER_LOST);
  REQUIRE(wait_children(children, world) == 0);
  REQUIRE(close(sync_pipe[0]) == 0);
  sagr_ccl_live_v1_broker_destroy(&broker);
  REQUIRE(count_open_descriptors() == before);
  return 0;
}

static int test_join_timeout_world2(void) {
  const uint32_t world = 2U;
  const uint32_t tag = 51U;
  sagr_ccl_v1_group_identity_t identity = make_identity(world, tag);
  sagr_ccl_live_v1_broker_t broker = NULL;
  sagr_ccl_live_v1_broker_info_t info;
  sagr_ccl_live_v1_abort_t first_error;
  child_process_t children[SAGR_CCL_V1_MAX_WORLD_SIZE];
  int before = count_open_descriptors();
  REQUIRE(sagr_ccl_live_v1_broker_create(&identity, &broker) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_live_v1_broker_info(broker, &info,
                                      (uint32_t)sizeof(info)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(spawn_rank(broker, &info.owner, world, 0U, tag,
                     "expect-join-timeout", -1, &children[0]) == 0);
  REQUIRE(spawn_rank(broker, &info.owner, world, 1U, tag, "hold-no-join", -1,
                     &children[1]) == 0);
  REQUIRE(bind_children(broker, children, world, 0) == 0);
  REQUIRE(sagr_ccl_live_v1_broker_rendezvous(
              broker, deadline_after(UINT64_C(250000000))) ==
          SAGR_CCL_V1_STATUS_TIMED_OUT);
  REQUIRE(sagr_ccl_live_v1_broker_first_error(
              broker, &first_error, (uint32_t)sizeof(first_error)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(first_error.reporter_rank == SAGR_CCL_LIVE_V1_NO_RANK &&
          first_error.failed_rank == 1U &&
          first_error.status == SAGR_CCL_V1_STATUS_TIMED_OUT);
  REQUIRE(wait_children(children, world) == 0);
  sagr_ccl_live_v1_broker_destroy(&broker);
  REQUIRE(count_open_descriptors() == before);
  return 0;
}

static int test_auth_failure(const char *mode, uint32_t tag, int swapped,
                             sagr_ccl_v1_status_t expected) {
  const uint32_t world = 2U;
  sagr_ccl_v1_group_identity_t identity = make_identity(world, tag);
  sagr_ccl_live_v1_broker_t broker = NULL;
  sagr_ccl_live_v1_broker_info_t info;
  sagr_ccl_live_v1_abort_t first_error;
  child_process_t children[SAGR_CCL_V1_MAX_WORLD_SIZE];
  int before = count_open_descriptors();
  REQUIRE(sagr_ccl_live_v1_broker_create(&identity, &broker) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_live_v1_broker_info(broker, &info,
                                      (uint32_t)sizeof(info)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(spawn_world(broker, &info.owner, world, tag, mode, -1, children,
                      swapped) == 0);
  REQUIRE(sagr_ccl_live_v1_broker_rendezvous(
              broker, deadline_after(TEST_DEADLINE_NS)) == expected);
  REQUIRE(sagr_ccl_live_v1_broker_first_error(
              broker, &first_error, (uint32_t)sizeof(first_error)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(first_error.status == expected);
  REQUIRE(wait_children(children, world) == 0);
  sagr_ccl_live_v1_broker_destroy(&broker);
  REQUIRE(count_open_descriptors() == before);
  return 0;
}

static int test_bind_identity_rejections(void) {
  const uint32_t world = 2U;
  const uint32_t tag = 71U;
  sagr_ccl_v1_group_identity_t identity = make_identity(world, tag);
  sagr_ccl_live_v1_broker_t broker = NULL;
  sagr_ccl_live_v1_broker_info_t info;
  sagr_ccl_live_v1_process_identity_t child_identity[2];
  sagr_ccl_live_v1_process_identity_t stale;
  child_process_t children[2];
  uint32_t rank;
  int before = count_open_descriptors();
  REQUIRE(sagr_ccl_live_v1_broker_create(&identity, &broker) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_live_v1_broker_info(broker, &info,
                                      (uint32_t)sizeof(info)) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  for (rank = 0U; rank < world; ++rank) {
    REQUIRE(spawn_rank(broker, &info.owner, world, rank, tag, "hold-no-join",
                       -1, &children[rank]) == 0);
    REQUIRE(sagr_ccl_live_v1_process_identity(
                (int32_t)children[rank].pid, &child_identity[rank],
                (uint32_t)sizeof(child_identity[rank])) ==
            SAGR_CCL_V1_STATUS_SUCCESS);
  }
  stale = child_identity[0];
  ++stale.start_time_ticks;
  REQUIRE(sagr_ccl_live_v1_broker_bind_rank(broker, 0U, &stale) ==
          SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH);
  REQUIRE(sagr_ccl_live_v1_broker_bind_rank(broker, 0U,
                                            &child_identity[0]) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(sagr_ccl_live_v1_broker_bind_rank(broker, 0U,
                                            &child_identity[0]) ==
          SAGR_CCL_V1_STATUS_OUT_OF_ORDER);
  REQUIRE(sagr_ccl_live_v1_broker_bind_rank(broker, 1U,
                                            &child_identity[0]) ==
          SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH);
  REQUIRE(sagr_ccl_live_v1_broker_bind_rank(broker, 1U,
                                            &child_identity[1]) ==
          SAGR_CCL_V1_STATUS_SUCCESS);
  REQUIRE(wait_children(children, world) == 0);
  sagr_ccl_live_v1_broker_destroy(&broker);
  REQUIRE(count_open_descriptors() == before);
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1) {
    return child_main(argc, argv);
  }
  REQUIRE(test_generic_configuration_world2_16() == 0);
  REQUIRE(test_success_world(2U, 2U) == 0);
  REQUIRE(test_success_world(3U, 3U) == 0);
  REQUIRE(test_success_world(4U, 4U) == 0);
  REQUIRE(test_success_world(8U, 8U) == 0);
  REQUIRE(test_success_world(16U, 16U) == 0);
  REQUIRE(test_auth_failure("wrong-claim", 21U, 0,
                            SAGR_CCL_V1_STATUS_PROTOCOL_ERROR) == 0);
  REQUIRE(test_auth_failure("wrong-group", 22U, 0,
                            SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH) == 0);
  REQUIRE(test_auth_failure("swapped-binding", 23U, 1,
                            SAGR_CCL_V1_STATUS_IDENTITY_MISMATCH) == 0);
  REQUIRE(test_bind_identity_rejections() == 0);
  REQUIRE(test_first_error_world3() == 0);
  REQUIRE(test_close_timeout_abort_world2() == 0);
  REQUIRE(test_peer_loss_world3() == 0);
  REQUIRE(test_join_timeout_world2() == 0);
  puts("ccl_live_v1: generic world2-16 and live world2/3/4/8/16 "
       "auth/fault/cleanup tests passed");
  return 0;
}
